"""
Process pools that work on a laptop and on a cluster node: CPU count, start method, BLAS/OpenMP
thread limits, a worker initializer that builds per-worker state, a watchdog for lost tasks, and
the split of items over ranks (nodes).

Usage
-----
Workers build their state in an initializer instead of reading module globals of the caller, so
the same code runs with the start methods 'fork' (Linux default) and 'spawn' (macOS, Windows, or
``PPMPY_SYNSPEC_START_METHOD=spawn``)::

    from ppmpy.synspec import parallel as par

    def setup(path):                       # runs once in every worker (and in replacements)
        return dict(lib=np.load(path))     # the returned object is the worker state

    def work(dump):                        # module-level function (picklable)
        st = par.worker_state()
        ...
        return result

    if __name__ == "__main__":             # required for 'spawn'
        par.login_node_warning(nproc)
        with par.make_pool(nproc, initializer=setup, initargs=(path,), maxtasksperchild=20) as pool:
            for res in par.imap_watchdog(pool, work, dumps, timeout=900):
                ...

With 'fork', ``initargs`` reach the workers without pickling (copy-on-write), so a large object
built once in the parent can be passed as an initarg and is shared between the workers; with
'spawn' every worker receives a pickled copy, so pass file names there and build (or memory-map)
in the initializer.

Thread limits: environment variables such as OPENBLAS_NUM_THREADS act only on libraries loaded
after they are set (and on child processes). In a pool worker numpy and its OpenBLAS are usually
loaded already (inherited with 'fork'; imported with the pickled initializer under 'spawn'), so
:func:`limit_blas_threads` also changes the loaded BLAS: through threadpoolctl when it is
installed, else by calling ``openblas_set_num_threads`` of the OpenBLAS bundled with numpy and
scipy wheels (ctypes). A BLAS from elsewhere (e.g. conda's MKL or a system OpenBLAS) needs
threadpoolctl or the variables set before numpy is imported.

Initializer errors: if the initializer raises in a pool worker, the worker records the traceback and
stays alive, and every task it receives raises :class:`WorkerInitError` with that traceback (instead of
multiprocessing restarting the dying workers for ever while the parent waits). This does not cover an
initializer or initargs that cannot be unpickled in a 'spawn' worker (e.g. a function defined in an
interactive session or notebook): that fails before the initializer runs, so use module-level functions
of importable modules.

PP 2026-10-01: ported from fw_disc_dumps.py (Pool with maxtasksperchild, the imap_unordered
watchdog, the rank split) and the login-node limits in the project CLAUDE.md; new: start-method
selection, thread limits, worker state.
"""
import functools
import glob
import multiprocessing as mp
import os
import sys
import traceback
import warnings

__all__ = ["available_cpus", "get_context", "set_thread_env", "limit_blas_threads", "blas_threads", "init_worker",
           "worker_state", "make_pool", "imap_watchdog", "PoolStalled", "WorkerInitError", "split_items",
           "login_node_warning", "LoginNodeWarning", "THREAD_ENV_VARS", "START_METHOD_ENV"]

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS")
START_METHOD_ENV = "PPMPY_SYNSPEC_START_METHOD"

_STATE = {}          # per-process worker state (init_worker)


class PoolStalled(RuntimeError):
    """
    No task of a pool finished within the watchdog timeout (a worker was killed, e.g. by the
    out-of-memory killer or a CPU-time limit, and its task is lost, or a task hangs).

    Attributes
    ----------
    missing: list
        The items whose results had not arrived (running, lost or not started).
    timeout: float
        The timeout that expired [s].
    """

    def __init__(self, missing, timeout):
        self.missing = list(missing)
        self.timeout = timeout
        super().__init__("no task finished for {:g} s (worker killed or stuck?); {} items missing, e.g. {}"
                         .format(timeout, len(self.missing), self.missing[:10]))

    def __reduce__(self):
        # PP 2026-10-01: picklable (e.g. raised in a worker of an outer pool): args holds the message only
        return type(self), (self.missing, self.timeout)


class WorkerInitError(RuntimeError):
    """
    The initializer of a pool worker raised (:func:`make_pool`); the message holds the worker's
    traceback. Raised by every task that worker receives, and by :func:`worker_state` there.
    """


class LoginNodeWarning(UserWarning):
    """Many worker processes outside a Slurm job (shared login node)."""


def available_cpus():
    """
    CPUs this process may run on.

    Returns
    -------
    int
        ``len(os.sched_getaffinity(0))`` where available (Linux; honours Slurm/cgroup CPU binding and
        taskset), else ``os.cpu_count()`` (at least 1).
    """
    # PP 2026-10-01: new
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def get_context(start_method=None):
    """
    multiprocessing context for a start method.

    Parameters
    ----------
    start_method: {None, 'fork', 'spawn', 'forkserver'}
        None: the environment variable PPMPY_SYNSPEC_START_METHOD if set, else 'fork' on Linux
        (cheap start, copy-on-write sharing of the parent's arrays, as the legacy scripts) and
        'spawn' elsewhere ('fork' is unsafe on macOS).

    Returns
    -------
    multiprocessing.context.BaseContext
    """
    # PP 2026-10-01: new
    if start_method is None:
        start_method = os.environ.get(START_METHOD_ENV) or ("fork" if sys.platform.startswith("linux") else "spawn")
    if start_method not in mp.get_all_start_methods():
        raise ValueError("start method {!r} not available here (have {})".format(start_method, mp.get_all_start_methods()))
    return mp.get_context(start_method)


def set_thread_env(n=1, blas=True):
    """
    Limit the threads of BLAS, OpenMP, MKL and numexpr to n in this process and its children.

    Sets OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, NUMEXPR_NUM_THREADS and
    VECLIB_MAXIMUM_THREADS, which act on libraries loaded afterwards and on child processes (e.g.
    external codes), and with ``blas`` also limits the BLAS libraries already loaded
    (:func:`limit_blas_threads`). An OpenMP runtime already loaded keeps its thread count unless
    threadpoolctl is installed; for a hard limit, set the variables before numpy is imported.

    Parameters
    ----------
    n: int
        Threads per process.
    blas: bool
        Also call :func:`limit_blas_threads` (n).

    Returns
    -------
    dict
        The previous values of the variables (None where unset), e.g. to restore them.
    """
    # PP 2026-10-01: new (the legacy scripts set no thread limits)
    n = _positive_int(n, "n")
    old = {k: os.environ.get(k) for k in THREAD_ENV_VARS}
    for k in THREAD_ENV_VARS:
        os.environ[k] = str(n)
    if blas:
        limit_blas_threads(n)
    return old


# OpenBLAS thread functions: numpy wheels (64-bit integer build, suffix 64_), scipy-openblas builds (prefix scipy_),
# plain OpenBLAS (scipy <= 1.13 wheels, system libraries)
_OPENBLAS_SET = ("openblas_set_num_threads64_", "scipy_openblas_set_num_threads64_", "openblas_set_num_threads",
                 "scipy_openblas_set_num_threads")
_OPENBLAS_GET = ("openblas_get_num_threads64_", "scipy_openblas_get_num_threads64_", "openblas_get_num_threads",
                 "scipy_openblas_get_num_threads")


def _loaded_bundled_openblas():
    """{path: (get, set)} of the OpenBLAS libraries bundled with the numpy and scipy wheels that are imported and
    already loaded in this process (dlopen with RTLD_NOLOAD: nothing is loaded, nothing promoted to global scope)."""
    # PP 2026-10-01: new. Library locations of the wheels: <site-packages>/<pkg>.libs (Linux, auditwheel) and
    # <pkg>/.dylibs (macOS, delocate). /proc/self/maps is not read.
    import ctypes
    nol = getattr(os, "RTLD_NOLOAD", None)
    if nol is None:                       # e.g. Windows: no way to open only an already loaded library
        return {}
    found = {}
    for name in ("numpy", "scipy"):
        f = getattr(sys.modules.get(name), "__file__", None)
        if not f:
            continue
        d = os.path.dirname(f)
        for pat in (os.path.join(d, os.pardir, name + ".libs", "lib*openblas*"),
                    os.path.join(d, ".dylibs", "lib*openblas*")):
            for p in sorted(glob.glob(pat)):
                p = os.path.realpath(p)
                if p in found:
                    continue
                try:
                    lib = ctypes.CDLL(p, mode=nol)
                except OSError:           # not loaded (e.g. scipy imported without scipy.linalg)
                    continue
                get = next((getattr(lib, s) for s in _OPENBLAS_GET if hasattr(lib, s)), None)
                set_ = next((getattr(lib, s) for s in _OPENBLAS_SET if hasattr(lib, s)), None)
                if get is None or set_ is None:
                    continue
                get.restype, get.argtypes = ctypes.c_int, []
                set_.restype, set_.argtypes = None, [ctypes.c_int]
                found[p] = (get, set_)
    return found


def blas_threads():
    """
    Thread counts of the BLAS libraries loaded in this process.

    Returns
    -------
    dict
        {library path: threads}: from threadpoolctl when it is installed, else for the OpenBLAS
        bundled with the numpy and scipy wheels (ctypes). Empty when neither finds a library
        (then the thread count cannot be queried here).
    """
    # PP 2026-10-01: new
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return {p: int(get()) for p, (get, _) in _loaded_bundled_openblas().items()}
    return {d["filepath"]: int(d["num_threads"]) for d in threadpool_info() if d.get("user_api") == "blas"}


def limit_blas_threads(n):
    """
    Set the thread count of the BLAS (and, with threadpoolctl, OpenMP) libraries already loaded in
    this process.

    Parameters
    ----------
    n: int
        Threads.

    Returns
    -------
    dict
        {library path: previous threads} of the BLAS libraries changed (empty when none was found:
        a BLAS loaded later follows OPENBLAS_NUM_THREADS etc., see :func:`set_thread_env`).

    Notes
    -----
    Uses ``threadpoolctl.threadpool_limits(n)`` when threadpoolctl is installed (all its
    libraries), else ``openblas_set_num_threads`` of the OpenBLAS bundled with numpy and scipy
    wheels through ctypes. Other libraries (conda MKL, system OpenBLAS, OpenMP runtimes) are left
    alone without threadpoolctl.
    """
    # PP 2026-10-01: new (reviewer: the environment variables alone do not limit an OpenBLAS that is already loaded,
    # which is the normal case in fork and spawn pool workers)
    n = _positive_int(n, "n")
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        old = {}
        for p, (get, set_) in _loaded_bundled_openblas().items():
            old[p] = int(get())
            set_(n)
        return old
    old = blas_threads()
    threadpool_limits(n)
    return old


def _positive_int(x, name):
    try:
        ok = int(x) == x and x >= 1
    except (TypeError, ValueError, OverflowError):
        ok = False
    if not ok:
        raise ValueError("{} must be a positive integer, got {!r}".format(name, x))
    return int(x)


def init_worker(initializer=None, initargs=(), threads=None):
    """
    Worker initializer used by :func:`make_pool`: limits the threads (:func:`set_thread_env`), runs
    ``initializer(*initargs)`` and keeps its return value as this process's worker state
    (:func:`worker_state`). Call it directly to run worker functions serially in the current
    process (exceptions of the initializer propagate then; in a pool see :class:`WorkerInitError`).

    Parameters
    ----------
    initializer: callable, optional
        Builds the worker state; must be picklable (a module-level function) for 'spawn'.
    initargs: tuple
        Its arguments.
    threads: int or None
        Threads per process (:func:`set_thread_env`: environment and loaded BLAS); None (default)
        leaves both alone, which is what a serial call in the main process wants. :func:`make_pool`
        passes its own ``threads`` (default 1).
    """
    # PP 2026-10-01: new
    _STATE.clear()
    if threads is not None:
        set_thread_env(threads)
    if initializer is not None:
        _STATE["state"] = initializer(*initargs)


def _pool_init(initializer, initargs, threads):
    """init_worker in a pool worker: an exception is recorded (WorkerInitError in every task) instead of killing the
    worker, which multiprocessing would replace by a new one running the failing initializer again, for ever."""
    # PP 2026-10-01: new (reviewer: a failing initializer gave thousands of worker restarts per second)
    try:
        init_worker(initializer, initargs, threads)
    except BaseException:
        _STATE.clear()
        _STATE["error"] = traceback.format_exc()


def _check_init_error():
    if "error" in _STATE:
        raise WorkerInitError("worker initializer failed in process {}:\n{}".format(os.getpid(), _STATE["error"]))


def worker_state():
    """
    The object returned by the initializer of this worker (:func:`init_worker`).

    Raises
    ------
    WorkerInitError
        If the initializer of this pool worker raised.
    RuntimeError
        If no initializer has run in this process.
    """
    _check_init_error()
    if "state" not in _STATE:
        raise RuntimeError("no worker state: pass an initializer to make_pool (or call init_worker in this process)")
    return _STATE["state"]


def make_pool(nproc, initializer=None, initargs=(), maxtasksperchild=None, start_method=None, threads=1):
    """
    A multiprocessing Pool whose workers limit their threads and build their state in an initializer.

    Parameters
    ----------
    nproc: int
        Worker processes (:func:`available_cpus`; on a shared login node see :func:`login_node_warning`).
    initializer: callable, optional
        See :func:`init_worker`. Also run in workers that replace retired ones. If it raises, the
        tasks of that worker raise :class:`WorkerInitError` (with the worker's traceback).
    initargs: tuple or list
        Its arguments (a bare string or other object raises TypeError: ``tuple('lib.npz')`` would
        split it into characters).
    maxtasksperchild: int, optional
        Tasks per worker before it is replaced (a fresh process restarts the CPU-time count of a
        ``ulimit -t`` limit, e.g. 3600 s on the Trillium login nodes; legacy fw_disc_dumps.py: 20).
    start_method: str, optional
        See :func:`get_context`.
    threads: int or None
        Threads per worker, default 1: :func:`set_thread_env` in every worker, i.e. the
        environment variables (libraries loaded later, child processes) and the BLAS already loaded
        (:func:`limit_blas_threads`: threadpoolctl, or ctypes for the OpenBLAS of numpy/scipy
        wheels; another BLAS that is already loaded keeps its threads without threadpoolctl).
        None leaves the workers' threads alone.

    Returns
    -------
    multiprocessing.pool.Pool
        Use as a context manager (``with``): leaving it terminates the workers.
    """
    # PP 2026-10-01: ported from fw_disc_dumps.py:123 (Pool(nproc, maxtasksperchild=...)); state via the initializer
    nproc = _positive_int(nproc, "nproc")
    if not isinstance(initargs, (tuple, list)):
        raise TypeError("initargs must be a tuple or list, got {}".format(type(initargs).__name__))
    if initializer is not None and not callable(initializer):
        raise TypeError("initializer must be callable")
    if threads is not None:
        threads = _positive_int(threads, "threads")
    ctx = get_context(start_method)
    return ctx.Pool(nproc, initializer=_pool_init, initargs=(initializer, tuple(initargs), threads),
                    maxtasksperchild=maxtasksperchild)


def _call_chunk(func, chunk):
    _check_init_error()
    return [(i, func(x)) for i, x in chunk]


def imap_watchdog(pool, func, items, timeout=900.0, chunksize=1):
    """
    ``pool.imap_unordered(func, items)`` with a watchdog: a worker killed by a signal (out of memory,
    CPU-time limit) never returns its task, and imap would wait for ever.

    Parameters
    ----------
    pool: multiprocessing.pool.Pool
        E.g. from :func:`make_pool`. The caller owns it (leaving its ``with`` block terminates the
        workers after :class:`PoolStalled`).
    func: callable
        Picklable function of one item.
    items: iterable
        Task arguments (read into a list).
    timeout: float or None
        Raise :class:`PoolStalled` when no result arrives for this long [s] (since the start or the
        previous result); None waits for ever. Legacy fw_disc_dumps.py: 900 s. Worker start-up
        counts against it: the first result needs a started worker whose initializer has finished,
        and with ``maxtasksperchild`` every replacement worker runs the initializer again (e.g.
        DiscImu setup ~2 min on a login node), so choose timeout > initializer time + the longest
        task.
    chunksize: int
        Items sent to a worker at a time (the timeout then applies to whole chunks).

    Yields
    ------
    The results of func, in completion order. Exceptions raised by func are re-raised here.

    Raises
    ------
    PoolStalled
        With ``missing`` = the items whose results had not arrived.
    WorkerInitError
        At the first task of a worker whose initializer raised (pools from :func:`make_pool`).
    """
    # PP 2026-10-01: ported from fw_disc_dumps.py:124-134 (it.next(timeout=...) -> abort with the missing dumps)
    # PP 2026-10-01: chunks are built here and sent one per task: with chunksize > 1, Pool.imap_unordered returns a
    # plain generator without next(timeout=...)
    if int(chunksize) != chunksize or chunksize < 1:
        raise ValueError("chunksize must be a positive integer")
    chunksize = int(chunksize)
    items = list(items)
    done = [False] * len(items)
    indexed = list(enumerate(items))
    chunks = [indexed[i:i + chunksize] for i in range(0, len(indexed), chunksize)]
    it = pool.imap_unordered(functools.partial(_call_chunk, func), chunks)
    for _ in range(len(chunks)):
        try:
            results = it.next(timeout=timeout)
        except mp.TimeoutError:
            raise PoolStalled([x for x, d in zip(items, done) if not d], timeout) from None
        for i, res in results:
            done[i] = True
            yield res


def split_items(items, rank, nranks):
    """
    The items of one rank (node, array task): item i goes to rank i % nranks.

    Parameters
    ----------
    items: sequence
    rank, nranks: int
        0 <= rank < nranks.

    Returns
    -------
    list

    Notes
    -----
    The rule of fw_disc_dumps.py (--rank/--nranks) and fw_sphere_task.sh (task k takes lines k,
    k + K, ...). sphere_sample.py splits by dump number instead (dump d to worker d % n), which is
    the same for consecutive dumps only when the first one is a multiple of n.
    """
    # PP 2026-10-01: ported from fw_disc_dumps.py:117
    if int(nranks) != nranks or nranks < 1 or int(rank) != rank or not 0 <= rank < nranks:
        raise ValueError("need integers 0 <= rank < nranks, got rank={!r}, nranks={!r}".format(rank, nranks))
    return [x for i, x in enumerate(items) if i % nranks == rank]


def login_node_warning(nproc, limit=20):
    """
    Warn when more than ``limit`` worker processes are started outside a Slurm job (SLURM_JOB_ID
    unset), i.e. probably on a shared login node.

    Parameters
    ----------
    nproc: int
        Planned worker processes.
    limit: int
        Largest number accepted silently (Trillium login node: ~20 workers; each process also has
        a 3600 s CPU-time limit, see ``maxtasksperchild`` of :func:`make_pool`).

    Returns
    -------
    bool
        True if a :class:`LoginNodeWarning` was issued.
    """
    # PP 2026-10-01: new (project CLAUDE.md: login-node limits)
    if os.environ.get("SLURM_JOB_ID") or nproc <= limit:
        return False
    warnings.warn("{} worker processes outside a Slurm job (shared login node?); the limit used here is {}. "
                  "Use fewer workers or a compute node.".format(nproc, limit), LoginNodeWarning, stacklevel=2)
    return True
