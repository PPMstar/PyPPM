"""Tests of ppmpy.synspec.parallel: CPU count, start methods, thread limits, worker state, watchdog, rank split.

The worker functions are module-level so that 'spawn' workers can import them (as test_parallel, from the
sys.path the parent passes on).
"""
import json
import os
import pickle
import signal
import subprocess
import sys
import textwrap
import threading
import time
import types

import numpy as np
import pytest

from conftest import ROOT
from ppmpy.synspec import parallel as par
from ppmpy.synspec import sphere as sph

LEGACY = os.environ.get("PPMPY_SYNSPEC_M424_PROJECT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy"))
LINUX = sys.platform.startswith("linux")


# ----------------------------------------------------------------------------------------------
# worker functions (module level: picklable for 'spawn')
# ----------------------------------------------------------------------------------------------
_PARENT = {}            # set by a test in the parent: visible to 'fork' workers only


def _setup_scale(scale):
    return dict(scale=scale, pid=os.getpid())


def _work_scale(x):
    st = par.worker_state()
    return x, x * st["scale"], os.getpid(), st["pid"], os.environ.get("OPENBLAS_NUM_THREADS"), _PARENT.get("mark")


def _setup_geometry(n, seed):
    """Worker state built in the worker: grid, projections and a random velocity field (float32, like the samples)."""
    th, ph = sph.fibonacci_sphere(n)
    rng = np.random.default_rng(seed)
    u = [(rng.normal(size=n) * 50.0).astype(np.float32) for _ in range(3)]
    return dict(proj=sph.project_los(th, ph, "thompson2024"), u=u)


def _work_vlos(k):
    st = par.worker_state()
    mu, tn, pn = (p[k] for p in st["proj"])
    v = sph.los_velocity(*st["u"], mu, tn, pn)
    return k, v, float(np.sum(sph.disc_weights(mu) * v))


def _setup_lock(lock):
    return dict(lock_type=type(lock).__name__)


def _work_lock(x):
    return par.worker_state()["lock_type"]


def _sleep_on_3(x):
    if x == 3:
        time.sleep(120)
    return x


def _kill_on_2(x):
    if x == 2:
        os.kill(os.getpid(), signal.SIGKILL)
    return x


def _fail_on_1(x):
    if x == 1:
        raise ValueError("item 1 failed")
    return x


def _work_blas(x):
    """The real BLAS thread counts of this worker (not just the environment variable)."""
    return x, par.blas_threads(), os.environ.get("OPENBLAS_NUM_THREADS")


def _setup_open(path, log):
    """An initializer that fails (missing library file) after logging its start."""
    with open(log, "a") as f:
        f.write("{}\n".format(os.getpid()))
    with open(path, "rb") as f:
        return f.read()


def _check_partition(got, missing, items):
    """Results and missing items of a stalled pool: disjoint, together all items, no duplicates."""
    assert len(got) == len(set(got))
    assert not set(got) & set(missing)
    assert set(got) | set(missing) == set(items)


def _start_methods():
    import multiprocessing
    return [m for m in ("fork", "spawn") if m in multiprocessing.get_all_start_methods()]


# ----------------------------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------------------------
def test_available_cpus(monkeypatch):
    n = par.available_cpus()
    assert isinstance(n, int) and 1 <= n <= (os.cpu_count() or n)
    if hasattr(os, "sched_getaffinity"):
        assert n == len(os.sched_getaffinity(0))
        monkeypatch.delattr(os, "sched_getaffinity")
    assert par.available_cpus() == max(1, os.cpu_count() or 1)


def test_get_context(monkeypatch):
    monkeypatch.delenv(par.START_METHOD_ENV, raising=False)
    assert par.get_context().get_start_method() == ("fork" if LINUX else "spawn")
    assert par.get_context("spawn").get_start_method() == "spawn"
    monkeypatch.setenv(par.START_METHOD_ENV, "spawn")
    assert par.get_context().get_start_method() == "spawn"
    with pytest.raises(ValueError):
        par.get_context("thread")


def test_set_thread_env(monkeypatch):
    for k in par.THREAD_ENV_VARS:
        monkeypatch.setenv(k, "7")                      # registered, so monkeypatch restores them
    before = par.blas_threads()
    old = par.set_thread_env(3, blas=False)                 # environment only: this process's BLAS is kept
    assert old == {k: "7" for k in par.THREAD_ENV_VARS}
    assert all(os.environ[k] == "3" for k in par.THREAD_ENV_VARS)
    assert par.blas_threads() == before
    for bad in (0, 2.5, "x"):
        with pytest.raises(ValueError):
            par.set_thread_env(bad)
        with pytest.raises(ValueError):
            par.limit_blas_threads(bad)
    assert all(os.environ[k] == "3" for k in par.THREAD_ENV_VARS)


def test_split_items_legacy_rule():
    items = list(range(3200, 3223))
    parts = [par.split_items(items, r, 4) for r in range(4)]
    assert sorted(sum(parts, [])) == items
    for r in range(4):
        assert parts[r] == items[r::4]
        # the line of the frozen fw_disc_dumps.py
        src = next(s for s in open(os.path.join(LEGACY, "fw_disc_dumps.py")).read().splitlines()
                   if s.strip().startswith("dumps = [d for i, d in enumerate(dumps)"))
        ns = dict(dumps=list(items), a=types.SimpleNamespace(rank=r, nranks=4))
        exec(textwrap.dedent(src), ns)
        assert parts[r] == ns["dumps"]
    assert par.split_items(items, 0, 1) == items
    assert par.split_items([], 0, 3) == []
    for rank, nranks in ((4, 4), (-1, 4), (0, 0), (0.5, 2), (0, 2.5)):
        with pytest.raises(ValueError):
            par.split_items(items, rank, nranks)


def test_login_node_warning(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.warns(par.LoginNodeWarning):
        assert par.login_node_warning(21) is True
    with pytest.warns(par.LoginNodeWarning):
        assert par.login_node_warning(9, limit=8) is True
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert par.login_node_warning(20) is False
        monkeypatch.setenv("SLURM_JOB_ID", "2457804")
        assert par.login_node_warning(1000) is False


def test_worker_state_serial(monkeypatch, tmp_path):
    monkeypatch.setattr(par, "_STATE", {})
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    with pytest.raises(RuntimeError):
        par.worker_state()
    before, env0 = par.blas_threads(), {k: os.environ.get(k) for k in par.THREAD_ENV_VARS}
    par.init_worker(_setup_scale, (2.5,))                 # default: serial use, threads (environment, BLAS) untouched
    assert _work_scale(4)[:2] == (4, 10.0)
    assert {k: os.environ.get(k) for k in par.THREAD_ENV_VARS} == env0 and env0["OPENBLAS_NUM_THREADS"] == "8"
    assert par.blas_threads() == before
    par.init_worker(_setup_scale, (1.5,), threads=None)
    assert _work_scale(4)[:2] == (4, 6.0)
    par.init_worker()                                     # no initializer: no state
    with pytest.raises(RuntimeError):
        par.worker_state()
    # serially, an initializer error propagates (no WorkerInitError, no recorded state)
    with pytest.raises(FileNotFoundError):
        par.init_worker(_setup_open, (str(tmp_path / "missing.npz"), str(tmp_path / "log")))
    with pytest.raises(RuntimeError, match="no worker state"):
        par.worker_state()


def test_exceptions_pickle():
    e = par.PoolStalled([3, 5], 2.5)
    f = pickle.loads(pickle.dumps(e))
    assert type(f) is par.PoolStalled and f.missing == [3, 5] and f.timeout == 2.5 and str(f) == str(e)
    w = pickle.loads(pickle.dumps(par.WorkerInitError("initializer failed")))
    assert type(w) is par.WorkerInitError and str(w) == "initializer failed"
    assert issubclass(par.WorkerInitError, RuntimeError) and issubclass(par.PoolStalled, RuntimeError)


def test_make_pool_arguments():
    """Bad arguments raise in the parent, before any worker starts."""
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError):
            par.make_pool(bad)
    with pytest.raises(TypeError, match="initargs"):
        par.make_pool(1, initializer=_setup_scale, initargs="lib.npz")      # tuple('lib.npz') would split it
    with pytest.raises(TypeError, match="initargs"):
        par.make_pool(1, initializer=_setup_scale, initargs=2.0)
    with pytest.raises(TypeError, match="callable"):
        par.make_pool(1, initializer="setup")
    for bad in (0, 1.5):
        with pytest.raises(ValueError):
            par.make_pool(1, threads=bad)


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_pool_threads_and_state(method, monkeypatch):
    if method not in __import__("multiprocessing").get_all_start_methods():
        pytest.skip("start method not available")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "5")
    monkeypatch.setitem(_PARENT, "mark", "parent")
    with par.make_pool(2, initializer=_setup_scale, initargs=(3.0,), start_method=method) as pool:
        res = sorted(par.imap_watchdog(pool, _work_scale, range(6), timeout=120))
    assert all(r[5] == ("parent" if method == "fork" else None) for r in res)      # spawn: a fresh interpreter
    assert [r[:2] for r in res] == [(i, 3.0 * i) for i in range(6)]
    assert all(r[2] == r[3] for r in res)                  # the state was built in that worker
    assert all(r[2] != os.getpid() for r in res)
    assert all(r[4] == "1" for r in res)                   # environment of the workers (the BLAS: test below)
    with par.make_pool(1, initializer=_setup_scale, initargs=(1.0,), start_method=method, threads=None) as pool:
        assert [r[4] for r in par.imap_watchdog(pool, _work_scale, [0], timeout=120)] == ["5"]
    assert os.environ["OPENBLAS_NUM_THREADS"] == "5"       # the parent is not changed


_BLAS_SCRIPT = r"""
import json, sys
import numpy                     # numpy and its BLAS are loaded here, with OPENBLAS_NUM_THREADS=2
from ppmpy.synspec import parallel as par
import test_parallel as tp       # 'spawn' workers import it (and numpy) when they unpickle the initializer,
                                 # i.e. before the pool initializer limits the threads
out = dict(parent=par.blas_threads())
for m in sys.argv[1:]:
    with par.make_pool(2, initializer=tp._setup_scale, initargs=(1.0,), start_method=m) as pool:
        out[m] = [r[1:] for r in par.imap_watchdog(pool, tp._work_blas, range(4), timeout=120)]
    with par.make_pool(1, initializer=tp._setup_scale, initargs=(1.0,), start_method=m, threads=None) as pool:
        out[m + "_none"] = [r[1:] for r in par.imap_watchdog(pool, tp._work_blas, [0], timeout=120)]
out["parent_after"] = par.blas_threads()
print(json.dumps(out))
"""


def test_pool_limits_loaded_blas():
    """The workers' BLAS really runs with 1 thread, although numpy (OpenBLAS) was loaded before the limit: inherited
    with 'fork', imported with the pickled initializer under 'spawn'. Run in a subprocess started with
    OPENBLAS_NUM_THREADS=2 (the container sets OMP_NUM_THREADS=1, which would hide the problem)."""
    methods = _start_methods()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="2",
               PYTHONPATH=os.pathsep.join([ROOT, os.path.dirname(os.path.abspath(__file__))]))
    env.pop(par.START_METHOD_ENV, None)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env.pop(k, None)
    out = subprocess.run([sys.executable, "-c", _BLAS_SCRIPT] + methods, capture_output=True, text=True, env=env,
                         cwd=ROOT, timeout=300)
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout.strip().splitlines()[-1])
    if not res["parent"]:
        pytest.skip("BLAS thread counts cannot be queried here (no threadpoolctl, no OpenBLAS of a numpy/scipy wheel)")
    if set(res["parent"].values()) != {2}:
        pytest.skip("BLAS did not start with 2 threads here: {}".format(res["parent"]))
    for m in methods:
        assert len(res[m]) == 4
        for blas, env_var in res[m]:
            assert env_var == "1"
            assert set(blas) == set(res["parent"]) and set(blas.values()) == {1}, (m, blas)
        for blas, env_var in res[m + "_none"]:                  # threads=None: inherited / from the environment
            assert env_var == "2" and blas == res["parent"], (m, blas)
    assert res["parent_after"] == res["parent"]                 # the parent is not changed


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_initializer_error_reported(method, tmp_path):
    """A failing initializer (missing library file) raises WorkerInitError with the worker's traceback at the first
    task, within seconds, instead of endless worker restarts and a PoolStalled after the timeout."""
    if method not in _start_methods():
        pytest.skip("start method not available")
    missing, log = str(tmp_path / "no_such_library.npz"), str(tmp_path / "init.log")
    t0 = time.time()
    with par.make_pool(2, initializer=_setup_open, initargs=(missing, log), start_method=method) as pool:
        with pytest.raises(par.WorkerInitError, match="FileNotFoundError") as e:
            list(par.imap_watchdog(pool, _work_scale, range(4), timeout=120))
    assert "no_such_library.npz" in str(e.value) and "_setup_open" in str(e.value)
    assert time.time() - t0 < 30
    starts = open(log).read().split()
    assert 1 <= len(starts) <= 2                            # each worker ran the initializer once: no restart loop
    # worker_state raises in such a worker as well
    with par.make_pool(1, initializer=_setup_open, initargs=(missing, log), start_method=method) as pool:
        with pytest.raises(par.WorkerInitError):
            list(par.imap_watchdog(pool, _work_scale, [0], timeout=120))


def test_fork_equals_spawn_equals_serial(monkeypatch):
    """Worker state built in the initializer: identical results with 'fork', 'spawn' and in-process."""
    methods = [m for m in ("fork", "spawn") if m in __import__("multiprocessing").get_all_start_methods()]
    out = {}
    for m in methods:
        with par.make_pool(3, initializer=_setup_geometry, initargs=(20011, 7), start_method=m) as pool:
            out[m] = {k: (v, s) for k, v, s in par.imap_watchdog(pool, _work_vlos, range(8), timeout=300)}
    monkeypatch.setattr(par, "_STATE", {})
    par.init_worker(_setup_geometry, (20011, 7), threads=None)
    out["serial"] = {k: (v, s) for k, v, s in map(_work_vlos, range(8))}
    ref = out["serial"]
    for m in methods:
        assert sorted(out[m]) == list(range(8))
        for k in range(8):
            np.testing.assert_array_equal(out[m][k][0], ref[k][0])
            assert out[m][k][1] == ref[k][1]


@pytest.mark.skipif(not LINUX, reason="fork start method")
def test_fork_initargs_not_pickled():
    """With 'fork' initargs reach the workers without pickling (copy-on-write); 'spawn' must pickle them."""
    lock = threading.Lock()
    with par.make_pool(1, initializer=_setup_lock, initargs=(lock,), start_method="fork") as pool:
        assert list(par.imap_watchdog(pool, _work_lock, [0], timeout=60)) == [type(lock).__name__]


def test_maxtasksperchild_reruns_initializer():
    with par.make_pool(1, initializer=_setup_scale, initargs=[2.0], maxtasksperchild=1) as pool:     # a list is fine
        res = list(par.imap_watchdog(pool, _work_scale, range(4), timeout=120))
    assert sorted(r[1] for r in res) == [0.0, 2.0, 4.0, 6.0]
    assert all(r[2] == r[3] for r in res)                  # every replacement worker built its own state
    assert len({r[2] for r in res}) >= 2


# The watchdog tests assert robust properties (the stuck or killed item is missing; results and missing items
# partition the items) with 5 s timeouts: on a loaded login node an extra item can be missing when a normal task
# takes longer than the timeout.
@pytest.mark.skipif(not LINUX, reason="fork start method")
def test_watchdog_stuck_worker():
    t0 = time.time()
    got = []
    with par.make_pool(2, start_method="fork") as pool:
        with pytest.raises(par.PoolStalled) as e:
            for r in par.imap_watchdog(pool, _sleep_on_3, range(6), timeout=5.0):
                got.append(r)
    assert 3 in e.value.missing and e.value.timeout == 5.0
    _check_partition(got, e.value.missing, range(6))
    assert "{} items missing".format(len(e.value.missing)) in str(e.value)
    assert time.time() - t0 < 60                           # the sleeping worker was terminated with the pool


@pytest.mark.skipif(not LINUX, reason="fork start method")
def test_watchdog_killed_worker():
    """A worker killed by a signal (OOM killer, CPU-time limit) loses its task; imap alone would wait for ever."""
    got = []
    with par.make_pool(2, start_method="fork") as pool:
        with pytest.raises(par.PoolStalled) as e:
            for r in par.imap_watchdog(pool, _kill_on_2, range(8), timeout=5.0):
                got.append(r)
    assert 2 in e.value.missing
    _check_partition(got, e.value.missing, range(8))


def test_watchdog_errors_and_no_timeout():
    with par.make_pool(2) as pool:
        with pytest.raises(ValueError, match="item 1 failed"):
            list(par.imap_watchdog(pool, _fail_on_1, range(4), timeout=60))
    with par.make_pool(2) as pool:
        assert sorted(par.imap_watchdog(pool, _fail_on_1, [0, 2, 3, 4], timeout=None, chunksize=2)) == [0, 2, 3, 4]
        assert sorted(par.imap_watchdog(pool, _fail_on_1, range(2, 9), timeout=60, chunksize=3)) == list(range(2, 9))
        assert list(par.imap_watchdog(pool, _fail_on_1, [], timeout=1)) == []
        with pytest.raises(ValueError):
            list(par.imap_watchdog(pool, _fail_on_1, [0], chunksize=0))


@pytest.mark.skipif(not LINUX, reason="fork start method")
def test_watchdog_chunks():
    """With chunks the missing items are those of the unfinished chunk(s)."""
    got = []
    with par.make_pool(2, start_method="fork") as pool:
        with pytest.raises(par.PoolStalled) as e:
            for r in par.imap_watchdog(pool, _sleep_on_3, range(9), timeout=5.0, chunksize=2):
                got.append(r)
    missing = e.value.missing
    assert {2, 3} <= set(missing)
    _check_partition(got, missing, range(9))
    for c in ([0, 1], [2, 3], [4, 5], [6, 7], [8]):          # whole chunks are missing or done
        assert set(c) <= set(missing) or not set(c) & set(missing)
