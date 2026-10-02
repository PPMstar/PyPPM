"""
Run FASTWIND for many sphere points on one node: fw_sphere_task.sh in Python (standard library only).

:func:`run_models` takes one task's share of a points table ('idx teff' lines), runs the models with a pool of
worker threads (each drives :func:`.model.run_model`, i.e. one pnlte + pformalsol at a time in its own process
groups), and has a packer child process move the finished results into the shared results directory. The output
is the legacy layout, byte-compatible, so ``ppmpy.synspec.fwresults`` (merge_task, combine, read_ledger,
locate_points, extract_points, check_indat_premise) reads it unchanged::

    <results_dir>/<tag>/part_*.tar.gz + part_*.idx      tag = task_%04d, or task_<list>_%04d for a point list

The legacy design is kept: no shared writes except the packer's (atomic renames), one read of the table per node,
the FASTWIND install and all run directories in node memory, a restartable done ledger, and a stop at SIGUSR1 (15
min before the time limit) that packs what is finished. What differs:

* split: line i of the table (0-based, blank and comment lines counted, as awk's ``(NR - 1) % K == k``) goes to
  task ``i % K``; a point listed twice in one task's share runs once;
* done list: the union of the ledgers ``<tag>/*.idx`` (any status), after :func:`.archive.recover` has rebuilt the
  ledger of an archive left without one by a crash and removed stale temporaries;
* staging: ``/dev/shm`` when it has room for the install, the run directories and the results of a pack interval,
  else a temporary directory (``tempfile``); removed at the end (kept when the final pack failed);
* the packer is a child process (``python -m ppmpy.synspec.fastwind.archive pack``, same interpreter) started every
  ``pack_interval`` s and once at the end; :func:`.archive.pack` is crash-safe;
* stop (SIGUSR1, SIGTERM, SIGINT, SIGHUP, SIGXCPU): :func:`.model.stop_all` kills the process groups of this
  runner's models only (never ``pkill`` by name, which also hit other jobs' and the user's other FASTWIND runs);
  interrupted models record nothing and run again after a restart; what is finished is packed; exit status 3
  (resumable). SIGHUP: an ssh session that drops while the run is in the foreground of a login shell; the log then
  goes nowhere (a dead terminal is ignored). SIGXCPU: the soft CPU-time limit of the runner process (login node
  ``ulimit -St`` 3600 s; the hard limit, 5400 s there, would SIGKILL it);
* parent-death guard: a watchdog child process (own session, the stop signals ignored) reads a pipe from the runner;
  when the runner dies without saying goodbye (SIGKILL, OOM killer, a crash), the pipe ends and the watchdog kills
  the runner's leftover FASTWIND process groups (those running in this runner's own run directories; with a staged
  root given by the caller only the groups the runner reported), waits for a running packer child, packs the
  finished results and removes the local root (the legacy runner lost them, and killed its workers only through the
  terminal's SIGHUP of its process group);
* status 'ok' needs complete OUT files (:func:`.formal.out_problem`), not just their existence;
* optional immediate retries of failed models (``retry`` > 0) with T_eff + ``retry_step`` per attempt, the rule of
  fwresults.combine's missing list (failures are deterministic: production point 571348 fails at 37298.910 K and
  converges at +1 K). Only the last attempt is packed (meta.txt with its T_eff, as a missing-list round would
  record it); the meta lines of the earlier attempts are kept in ``attempts.txt`` of the point. Default 0: the
  legacy behaviour (failures recorded; ``fwresults.combine`` lists them in missing.txt with T_eff + 1 K);
* exit status: 0 every point of the share is in the ledgers (ok or failed), 1 errors (some points neither
  recorded nor run: run again; or the final pack failed; or more than ``max_errors`` exceptions), 3 stopped;
* the number of tasks K is never guessed from Slurm: a task index from $SLURM_PROCID / $SLURM_ARRAY_TASK_ID in a
  step or array of several tasks needs K given (:func:`resolve_tasks`);
* a JSON record of each run, ``<tag>/runner_<UTC>_<host>_<pid>_<n>.json`` (counts, parts, exit status; the
  watchdog's ``..._watchdog.json``), and :func:`status` (ledgers, orphans, temporaries, progress against the table).

PP 2026-10-02: new (M6); ported from fw_sphere_task.sh (whole), fw_sphere_node.sbatch and fw_sphere_multinode.sbatch
(task index, tags) of the project stellar-atmosphere-KU-Leuven.
"""
import argparse
import collections
import fnmatch
import itertools
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

from . import archive
from . import model as mo
from .formal import OUT_NROW, VTURB_M424
from .install import DATA_DIRS, HOPF_FILES, FastwindInstall, StagedRoot, atomic_write

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_STOPPED = 3
"""Exit status of a run stopped by a signal (resumable: run again with the same arguments)."""

STOP_SIGNALS = tuple(n for n in ("SIGUSR1", "SIGTERM", "SIGINT", "SIGHUP", "SIGXCPU") if hasattr(signal, n))
"""Signals that stop a run in order (models killed, finished results packed, exit 3)."""
CHILD_IGNORED = archive.CHILD_IGNORED
"""Signals the packer and watchdog children ignore; blocked from the fork to their own SIG_IGN, so a step-wide Slurm
signal in their first milliseconds cannot kill them."""
TEFF_NUDGE_MAX = 10.0
"""Largest total T_eff nudge of the retries (fwresults.TEFF_NUDGE_MAX: merge_task refuses larger ones)."""
RETRY_ON = ("pnlte_failed", "formal_failed")
"""Statuses retried by default (a timed-out model would likely cost another full time limit)."""
ATTEMPTS_FILE = "attempts.txt"
RUNNER_PREFIX = "runner_"
WATCHDOG_LOG = "watchdog.log"
_record_seq = itertools.count(1)
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN_DIR_BYTES = 48 * 2 ** 20
"""Room per running model in the local root (legacy: ~7.5 GB for 192 run directories)."""
RESULT_BYTES = {"profiles": 2 ** 17, "model": 8 * 2 ** 20}
"""Room per finished point waiting for the packer (profiles ~50 kB with logs; with the model files ~6.7 MB)."""
MODEL_SECONDS = 150.0
"""Typical model time (M424 at 192 per node: 127-208 s), for the room of a pack interval's results."""
SETTLE = 1.0
"""Seconds a failed model waits before it is recorded, so that a stop signal that also hit the FASTWIND processes
(Slurm signals every process of a step) is seen first and the model is rerun instead of recorded as failed."""


class _Flag:
    """A flag without locks (safe to set in a signal handler)."""
    value = False


def _log_default(msg):
    """A time-stamped line on stdout; a dead stdout (closed pipe, hung-up terminal after SIGHUP) is ignored."""
    try:
        sys.stdout.write("{} {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S"), msg))
        sys.stdout.flush()
    except (OSError, ValueError):
        _drop_stdout()


def _drop_stdout():
    """Replace a dead sys.stdout by /dev/null: Python's flush of stdout at exit would fail and turn the exit status
    into 120."""
    # PP 2026-10-02: new (reviewer: a run stopped by the SIGHUP of a dropped ssh session writes to a dead terminal)
    try:
        sys.stdout = open(os.devnull, "w")
    except OSError:
        pass


def _child_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = PKG_ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _popen_signals_blocked(cmd, **kw):
    """
    subprocess.Popen with :data:`CHILD_IGNORED` blocked in this thread around the fork: the child inherits the
    blocked mask through fork and exec and sets SIG_IGN before it unblocks them (a pending signal is then discarded).
    Without it, a signal sent to the whole Slurm step in the child's first ~10-30 ms killed the packer (exit -10).
    """
    # PP 2026-10-02: new (reviewer)
    sigs = {getattr(signal, n) for n in CHILD_IGNORED}
    can = hasattr(signal, "pthread_sigmask")
    old = signal.pthread_sigmask(signal.SIG_BLOCK, sigs) if can else None
    try:
        return subprocess.Popen(cmd, **kw)
    finally:
        if can:
            signal.pthread_sigmask(signal.SIG_SETMASK, old)


# ----------------------------------------------------------------------------------------------------------------
# task index, tag, table split
# ----------------------------------------------------------------------------------------------------------------
def task_index(task=None, environ=None):
    """
    This task's index k: ``task``, else $FW_TASK, else $SLURM_ARRAY_TASK_ID, else $SLURM_PROCID (srun rank of a
    multi-node job), else 0; empty variables count as unset (bash ``${FW_TASK:-...}``).
    """
    # PP 2026-10-02: ported from fw_sphere_task.sh:22
    if task is not None:
        return int(task)
    env = os.environ if environ is None else environ
    for k in ("FW_TASK", "SLURM_ARRAY_TASK_ID", "SLURM_PROCID"):
        v = env.get(k, "")
        if v != "":
            return int(v)
    return 0


def task_count(ntasks=None, environ=None):
    """The number of tasks K: ``ntasks``, else $FW_NTASKS, else 1 (fw_sphere_task.sh's default). See
    :func:`resolve_tasks` for the check against Slurm."""
    # PP 2026-10-02: ported from fw_sphere_task.sh:21 (K=${2:-1})
    if ntasks is not None:
        return int(ntasks)
    v = (os.environ if environ is None else environ).get("FW_NTASKS", "")
    return int(v) if v != "" else 1


def resolve_tasks(task=None, ntasks=None, environ=None):
    """
    (k, K) of this runner: :func:`task_index` and :func:`task_count`, refusing a K of 1 by default when k comes from
    Slurm and Slurm runs several tasks.

    Raises
    ------
    ValueError
        ``ntasks`` and $FW_NTASKS not given, k taken from $SLURM_ARRAY_TASK_ID in an array of more than one task
        ($SLURM_ARRAY_TASK_COUNT), or from $SLURM_PROCID in a step of more than one task ($SLURM_STEP_NUM_TASKS,
        else $SLURM_NTASKS). With K = 1 rank 0 would run the whole table and the other ranks fail.

    Notes
    -----
    K is not taken from Slurm: a resubmitted subset of an array (``--array=3,17``) has another task count than the
    split, and a runner started in the batch script itself sees $SLURM_PROCID = 0 with $SLURM_NTASKS = the job's
    tasks (e.g. ``--ntasks-per-node=192``), where K = $SLURM_NTASKS would run 1/192 of the table without a word.
    """
    # PP 2026-10-02: new (reviewer: srun without -K ran the whole table on rank 0)
    env = os.environ if environ is None else environ
    k = task_index(task, env)
    if ntasks is not None or env.get("FW_NTASKS", "") != "" or task is not None or env.get("FW_TASK", "") != "":
        return k, task_count(ntasks, env)

    def count(*names):
        for n in names:
            v = env.get(n, "")
            if v != "":
                try:
                    return int(v)
                except ValueError:
                    return None
        return None

    if env.get("SLURM_ARRAY_TASK_ID", "") != "":
        n, what = count("SLURM_ARRAY_TASK_COUNT"), "an array of {} tasks ($SLURM_ARRAY_TASK_ID)"
    elif env.get("SLURM_PROCID", "") != "":
        n, what = count("SLURM_STEP_NUM_TASKS", "SLURM_NTASKS"), "a Slurm step of {} tasks ($SLURM_PROCID)"
    else:
        n, what = None, ""
    if n is not None and n > 1:
        raise ValueError(("task index {} from {}, but the number of tasks K is not given: pass -K / ntasks (or set "
                          "$FW_NTASKS); K is not guessed from Slurm").format(k, what.format(n)))
    return k, 1


def build_nfobs(build_dir):
    """
    ID_NFOBS of a FASTWIND build (``<build_dir>/nlte_dim.f90``: the rows of every OUT table), or None when the file
    or the parameter is absent.
    """
    # PP 2026-10-02: new (reviewer: a build with another NFOBS than 161 would record every model formal_failed)
    import re
    try:
        with open(os.path.join(build_dir, "nlte_dim.f90"), errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    m = re.search(r"\bID_NFOBS\s*=\s*(\d+)", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def list_name(path):
    """``basename LIST .txt`` (the list part of the tag)."""
    b = os.path.basename(os.path.normpath(os.fspath(path)))
    return b[:-4] if b.endswith(".txt") and b != ".txt" else b


def task_tag(task, list_name=None):
    """``task_%04d``, or ``task_<list>_%04d`` for a point list (fw_sphere_task.sh:28)."""
    # PP 2026-10-02: ported from fw_sphere_task.sh:28
    if list_name:
        return "task_{}_{:04d}".format(list_name, int(task))
    return "task_{:04d}".format(int(task))


def _teff_text(x):
    if isinstance(x, (bytes, bytearray)):
        x = bytes(x).decode()
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, bool):
        raise ValueError("T_eff must be a number or text, got a bool")
    return str(x) if isinstance(x, int) else repr(float(x))


def _entry(i, idx_t, teff_t, where):
    try:
        idx = int(idx_t)
    except (TypeError, ValueError):
        raise ValueError("{}: point index {!r} is not an integer".format(where, idx_t)) from None
    if idx < 0:
        raise ValueError("{}: negative point index {}".format(where, idx))
    text = _teff_text(teff_t)
    try:
        v = float(text)
    except ValueError:
        raise ValueError("{}: T_eff {!r} is not a number".format(where, text)) from None
    if not text or any(c.isspace() for c in text) or not math.isfinite(v) or v <= 0:
        raise ValueError("{}: T_eff {!r} must be a positive finite number".format(where, text))
    return (i, idx, text)


def split_table(table, ntasks=1, task=0):
    """
    This task's share of a points table: the entries of the lines i (0-based) with ``i % ntasks == task``.

    Parameters
    ----------
    table: str, os.PathLike or iterable
        A file of 'idx teff' lines (points.txt, missing.txt, a pilot list), or an iterable of such lines or of
        (idx, teff) pairs. Every line counts for the split (blank lines and '#' comments too, as awk's NR), but
        only 'idx teff' lines give entries; T_eff text is kept verbatim (meta.txt records it as given).
    ntasks, task: int
        K and k.

    Returns
    -------
    list of (line, idx, teff_text)

    Raises
    ------
    ValueError
        A selected line that is not 'idx teff' with an integer idx >= 0 and a positive finite T_eff.
    """
    # PP 2026-10-02: ported from fw_sphere_task.sh:45 (awk '(NR - 1) % K == k')
    ntasks, task = int(ntasks), int(task)
    if ntasks < 1 or not 0 <= task < ntasks:
        raise ValueError("task {} of {}: need 0 <= task < ntasks".format(task, ntasks))
    if isinstance(table, (str, os.PathLike)):
        with open(table, "rb") as f:
            return _split(f, ntasks, task, os.fspath(table))
    return _split(table, ntasks, task, "table")


def _split(lines, ntasks, task, src):
    out = []
    for i, item in enumerate(lines):
        if i % ntasks != task:
            continue
        where = "{}:{}".format(src, i + 1)
        if isinstance(item, (bytes, bytearray, str)):
            tok = (item.decode("latin-1") if isinstance(item, (bytes, bytearray)) else item).split()
            if not tok or tok[0].startswith("#"):
                continue
            if len(tok) != 2:
                raise ValueError("{}: expected 'idx teff', got {!r}".format(where, " ".join(tok)))
            out.append(_entry(i, tok[0], tok[1], where))
        else:
            try:
                a, b = item
            except (TypeError, ValueError):
                raise ValueError("{}: expected an (idx, teff) pair, got {!r}".format(where, item)) from None
            out.append(_entry(i, a, b, where))
    return out


# ----------------------------------------------------------------------------------------------------------------
# local root
# ----------------------------------------------------------------------------------------------------------------
def _tree_bytes(path):
    total = 0
    if os.path.isfile(path):
        return os.path.getsize(path)
    for dp, dn, fn in os.walk(path):
        for f in fn:
            try:
                total += os.lstat(os.path.join(dp, f)).st_size
            except OSError:
                pass
    return total


def local_need(install, nworkers, keep="profiles", stage_mode="copy", pack_interval=900.0):
    """
    Bytes the node-local root needs: the staged install (copy mode), ``nworkers`` run directories and the results
    of one pack interval (:data:`RESULT_BYTES` per point at one model per :data:`MODEL_SECONDS` per worker).
    """
    # PP 2026-10-02: new (fw_sphere_task.sh assumed /dev/shm: ~0.4 GB install + ~7.5 GB run directories)
    inst = 0
    if isinstance(install, FastwindInstall) and stage_mode == "copy":
        inst = sum(_tree_bytes(install.data_dir(d)) for d in DATA_DIRS)
        inst += sum(_tree_bytes(install.hopf_file(h)) for h in HOPF_FILES)
        inst += sum(_tree_bytes(p) for p in install.bin_sources().values() if os.path.exists(p))
    npack = nworkers * max(1.0, float(pack_interval) / MODEL_SECONDS + 1.0)
    return int(inst + nworkers * RUN_DIR_BYTES + npack * RESULT_BYTES.get(keep, RESULT_BYTES["model"]))


def _free_bytes(path):
    try:
        s = os.statvfs(path)
    except OSError:
        return 0
    return s.f_bavail * s.f_frsize


def choose_local_base(local_root=None, need=0, shm="/dev/shm"):
    """
    (directory, reason) for the node-local root: ``local_root`` when given, else ``shm`` (node memory, as the
    legacy runner) when it exists and has ``need`` bytes free, else the temporary directory (``tempfile``, $TMPDIR).
    """
    # PP 2026-10-02: new; fw_sphere_task.sh:31 used /dev/shm unconditionally
    if local_root:
        return os.path.abspath(local_root), "given"
    if shm and os.path.isdir(shm) and os.access(shm, os.W_OK) and _free_bytes(shm) >= need:
        return shm, "node memory, {:.1f} GB free".format(_free_bytes(shm) / 1e9)
    d = tempfile.gettempdir()
    return d, "{} too small or absent (need {:.1f} GB); temporary directory".format(shm, need / 1e9)


# ----------------------------------------------------------------------------------------------------------------
# the run
# ----------------------------------------------------------------------------------------------------------------
class _PackChild:
    """One packer child process (python -m ppmpy.synspec.fastwind.archive pack ...)."""

    def __init__(self, run, final=False):
        self.run = run
        self.final = final
        run.pack_seq += 1
        self.part = archive.part_name(pid=os.getpid(), seq=run.pack_seq)
        self.report = os.path.join(run.local, "pack_{:04d}.json".format(run.pack_seq))
        self.proc = None
        self.t0 = time.monotonic()

    def start(self):
        cmd = [sys.executable, "-m", "ppmpy.synspec.fastwind.archive", "pack", self.run.res, self.run.stage,
               self.run.out, "--part", self.part, "--report", self.report, "--ignore-signals",
               "--compresslevel", str(self.run.compresslevel)]
        with open(os.path.join(self.run.local, "packer.log"), "ab") as logf:
            self.proc = _popen_signals_blocked(cmd, cwd=self.run.local, env=_child_env(), stdin=subprocess.DEVNULL,
                                               stdout=logf, stderr=subprocess.STDOUT, close_fds=True,
                                               start_new_session=True)
        self.run.tell_watchdog(packer=self.proc.pid)
        return self

    def poll(self):
        return self.proc.poll()

    def wait(self):
        return self.proc.wait()

    def result(self):
        """The pack report, or None when the child failed (logged)."""
        rc = self.proc.wait()
        self.run.tell_watchdog(packer=0)
        rep = None
        try:
            with open(self.report) as f:
                rep = json.load(f)
        except (OSError, ValueError):
            pass
        if rc != 0 or not rep or not rep.get("ok"):
            self.run.log("packer child failed (exit {}): {}; see the end of {}".format(
                rc, (rep or {}).get("error", "no report"), os.path.join(self.run.local, "packer.log")))
            return None
        return rep


class _Watchdog:
    """
    The runner's side of the parent-death guard (:func:`watchdog_main` in a child process): a pipe whose end tells
    the watchdog that the runner is gone. The runner sends the process groups of its models and the pid of a running
    packer child when they change, and ``DONE`` at its orderly end. Writes never block the runner (non-blocking pipe;
    a line that does not fit is sent with the next update).
    """

    def __init__(self, run, root, private):
        self.run, self.root, self.private = run, root, private
        self.proc = None
        self.fd = None
        self.dead = False
        self.state = dict(G=(), P=0)
        self.sent = {}

    def start(self):
        r = self.run
        cmd = [sys.executable, "-m", "ppmpy.synspec.fastwind.batch", "watchdog", r.local, r.res, r.stage, r.out,
               self.root, "--runner-pid", str(os.getpid()), "--grace", str(r.grace), "--compresslevel",
               str(r.compresslevel)] + (["--private"] if self.private else []) + (["--keep-local"] if r.keep_local
                                                                                  else [])
        with open(os.path.join(r.local, WATCHDOG_LOG), "ab") as logf:
            self.proc = _popen_signals_blocked(cmd, cwd=r.local, env=_child_env(), stdin=subprocess.PIPE, stdout=logf,
                                               stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
        self.fd = self.proc.stdin.fileno()
        os.set_blocking(self.fd, False)
        return self

    def _line(self, key):
        if key == "P":
            return "P {}\n".format(self.state["P"]).encode()
        out = "G"
        for g in self.state["G"]:
            if len(out) + 12 > 4000:           # PIPE_BUF: one atomic write; the cwd scan covers a private root
                break
            out += " {}".format(g)
        return (out + "\n").encode()

    def update(self, groups=None, packer=None):
        """Send what changed (groups: pgids of the running models; packer: pid of a packer child, 0 none)."""
        if groups is not None:
            self.state["G"] = tuple(sorted(groups))
        if packer is not None:
            self.state["P"] = int(packer)
        if self.dead or self.fd is None:
            return
        for key in ("P", "G"):
            if self.sent.get(key) == self.state[key]:
                continue
            try:
                os.write(self.fd, self._line(key))
                self.sent[key] = self.state[key]
            except BlockingIOError:
                return                            # pipe full: sent with the next update
            except OSError:
                self.dead = True
                self.run.log("the watchdog is gone (exit {}): no parent-death guard".format(self.proc.poll()))
                return

    def done(self, timeout=10.0):
        """Tell the watchdog that the runner ends in order, and reap it."""
        if self.proc is None:
            return
        try:
            os.set_blocking(self.fd, True)
            os.write(self.fd, b"DONE\n")
        except OSError:
            pass
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.run.log("watchdog did not end: killed")
            self.proc.kill()
            self.proc.wait()


class _Run:
    """State of one :func:`run_models` call (shared by the worker threads and the main thread)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.lock = threading.Lock()
        self.stop = _Flag()
        self.signal = None
        self.aborted = False
        self.counts = collections.Counter()
        self.errors = []
        self.parts = []
        self.packed = 0
        self.pack_failures = 0
        self.pack_seq = 0
        self.retried = 0
        self.retry_ok = 0
        self.interrupted = 0
        self.running = 0
        self.watchdog = None
        self.periodic_packs = 0

    def tell_watchdog(self, groups=None, packer=None):
        if self.watchdog is not None:
            self.watchdog.update(groups=groups, packer=packer)

    # -- queue ------------------------------------------------------------------------------------------------
    def next_item(self):
        with self.lock:
            if self.stop.value or not self.queue:
                return None
            self.running += 1
            return self.queue.popleft()

    def finished_item(self):
        with self.lock:
            self.running -= 1

    # -- one point -----------------------------------------------------------------------------------------
    def _attempt_teff(self, teff_text, k):
        return teff_text if k == 0 else "%.3f" % (float(teff_text) + k * self.retry_step)

    def run_point(self, hold, idx, teff_text):
        """Run one point (with retries); returns the final run_model record, or None if interrupted."""
        attempts = []
        r = None
        for k in range(self.retry + 1):
            try:
                r = mo.run_model(self.staged, (idx, self._attempt_teff(teff_text, k)), hold, self.template,
                                 self.formal, vturb=self.vturb, iescat=self.iescat, keep=self.keep,
                                 pnlte_timeout=self.pnlte_timeout, formal_timeout=self.formal_timeout,
                                 extras=self.extras, grace=self.grace, env=self.env, nrow=self.nrow)
            except mo.ModelInterrupted:
                return None                           # nothing recorded: the point runs again after a restart
            if r["status"] == "ok" or r["status"] not in self.retry_on or k == self.retry:
                break
            if not self.stop.value:
                time.sleep(SETTLE)
            if self.stop.value:
                return None                           # a stop that may have hit this model: rerun it later
            attempts.append(r["meta"])
            self.log("P{:06d} {} at T_eff {} (flags {}); retry at {}".format(
                idx, r["status"], r["teff"], ",".join(r["flags"]) or "-", self._attempt_teff(teff_text, k + 1)))
        if r["status"] != "ok":
            if not self.stop.value:
                time.sleep(SETTLE)
            if self.stop.value:
                return None
        src = r["result_dir"]
        if attempts:
            with open(os.path.join(src, ATTEMPTS_FILE), "w") as f:
                f.write("".join(attempts))
        mo._move_into_place(src, os.path.join(self.res, os.path.basename(src)))
        r = dict(r, attempts=len(attempts))
        return r

    def worker(self, wid):
        hold = os.path.join(self.local, "hold", "w{:03d}".format(wid))
        os.makedirs(hold, exist_ok=True)
        while True:
            item = self.next_item()
            if item is None:
                return
            _, idx, teff_text = item
            try:
                r = self.run_point(hold, idx, teff_text)
            except Exception as e:                    # recorded nowhere: the point runs again after a restart
                with self.lock:
                    self.errors.append((idx, repr(e)))
                    nerr = len(self.errors)
                self.log("P{:06d}: {!r}".format(idx, e))
                if self.max_errors is not None and nerr > self.max_errors and not self.stop.value:
                    self.log("more than {} errors: stopping".format(self.max_errors))
                    self.aborted = True
                    self.stop.value = True
                    mo.stop_all()
                self.finished_item()
                continue
            with self.lock:
                if r is None:
                    self.interrupted += 1
                else:
                    self.counts[r["status"]] += 1
                    if r["attempts"]:
                        self.retried += 1
                        self.retry_ok += r["status"] == "ok"
            if r is not None and r["status"] != "ok":
                self.log("P{:06d} {} (T_eff {}, niter {}, flags {})".format(idx, r["status"], r["teff"], r["niter"],
                                                                         ",".join(r["flags"]) or "-"))
            self.finished_item()

    # -- packing ------------------------------------------------------------------------------------------
    def note_pack(self, rep):
        if rep is None:
            self.pack_failures += 1
            return False
        if rep.get("part"):
            self.parts.append(rep["part"])
            self.packed += rep["npoint"]
            self.log("packed {} points -> {} ({:.1f} MB, {:.1f} s)".format(rep["npoint"], rep["part"],
                                                                          rep["bytes"] / 1e6, rep["seconds"]))
        return True

    def final_pack(self):
        """Pack what is left: a child process, else (it failed) in this process. True on success."""
        ok = False
        if self.pack_child:
            try:
                ok = self.note_pack(_PackChild(self, final=True).start().result())
            except OSError as e:
                self.log("cannot start the packer child: {!r}".format(e))
        if not ok:
            try:
                self.pack_seq += 1
                rep = archive.pack(self.res, self.stage, self.out, compresslevel=self.compresslevel,
                                   part=archive.part_name(pid=os.getpid(), seq=self.pack_seq))
                ok = self.note_pack(dict(rep, ok=True))
            except Exception as e:
                self.log("final pack failed: {!r}; results kept in {}".format(e, self.local))
                self.pack_failures += 1
                ok = False
        return ok


def run_models(table, install, template, formal, results_dir, task=None, ntasks=None, tag=None, list_name=None,
               nworkers=2, local_root=None, stage_mode="copy", keep="profiles", pack_interval=900,
               pnlte_timeout=3600, formal_timeout=600, retry=0, retry_step=1.0, retry_on=RETRY_ON, extras=True,
               vturb=VTURB_M424, iescat=0, grace=5.0, env=None, nrow=OUT_NROW, max_errors=20, signals=True,
               pack_child=True, compresslevel=archive.COMPRESSLEVEL, keep_local=False, poll=0.2, watchdog=True,
               log=None):
    """
    Run FASTWIND for one task's share of a points table (fw_sphere_task.sh); see the module notes.

    Parameters
    ----------
    table: str or iterable
        'idx teff' lines (points.txt, missing.txt, a pilot list) or (idx, teff) pairs (:func:`split_table`).
    install: FastwindInstall, StagedRoot or None
        The FASTWIND install (staged into the local root, ``stage_mode``), or a root staged earlier (used as it is),
        or None (:meth:`FastwindInstall.from_env`).
    template, formal:
        INDAT template and FORMAL_INPUT (objects, paths or text; :func:`.model.run_model`).
    results_dir: str
        ``RUN_DIR/results``; this task writes only to ``<results_dir>/<tag>``.
    task, ntasks: int, optional
        k and K (:func:`resolve_tasks`: from the environment when not given; K must be given when k comes from a
        Slurm step or array of several tasks).
    tag: str, optional
        Output directory name (default :func:`task_tag` of ``task`` and ``list_name``).
    list_name: str, optional
        The list part of the default tag (``task_<list>_%04d``; :func:`list_name` of the list's path).
    nworkers: int
        Models at once (M424 production: 192 per node; the login node: at most 4).
    local_root: str, optional
        Parent of the node-local root (default :func:`choose_local_base`).
    stage_mode: str
        'copy' (node-local copies, legacy) or 'link'.
    keep: str
        'profiles' (KEEP_MODEL=0) or 'model' (KEEP_MODEL=1).
    pack_interval: float
        Seconds between packs (legacy 900).
    pnlte_timeout, formal_timeout: float
        Time limits per model (legacy PNLTE_TIMEOUT 3600 s).
    retry: int
        Immediate retries of a failed model (status in ``retry_on``) at T_eff + k ``retry_step`` (k = 1..retry);
        0 (default) as the legacy runner. ``retry * retry_step`` must not exceed :data:`TEFF_NUDGE_MAX`.
    extras: bool or str
        :func:`.model.run_model` extras ('full' by default; False: exactly the legacy files).
    vturb, iescat, grace, env, nrow:
        Passed to :func:`.model.run_model`.
    max_errors: int or None
        Stop (exit 1) after more than this many exceptions of :func:`.model.run_model` (not failed models: errors of
        the runner, e.g. a full disk). None: never. Fewer exceptions still give exit 1 (those points are neither
        recorded nor done; a run again runs them).
    signals: bool
        Install the stop handlers for :data:`STOP_SIGNALS` (SIGUSR1, SIGTERM, SIGINT, SIGHUP, SIGXCPU; only possible
        in the main thread; restored at the end).
    pack_child: bool
        Pack in a child process (default); False packs in this process (tests).
    keep_local: bool
        Keep the local root (debugging).
    poll: float
        Seconds between the main loop's checks (packer, watchdog updates).
    watchdog: bool
        Start the parent-death guard (:func:`watchdog_main`; default True).
    log: callable, optional
        Receives the log lines (default: stdout with a time stamp).

    Returns
    -------
    dict
        tag, out_dir, task, ntasks, host, pid, local_root, assigned (entries of this task), duplicates, done_before,
        todo, status (counts of this run), interrupted, not_started, errors, n_errors, retried, retry_ok, parts,
        packed, pack_failures, periodic_packs (packs before the final one that succeeded), stopped, signal,
        aborted, exit_code (0 every point of the share recorded; 1 errors: points neither recorded nor run, a failed
        final pack, or an abort; 3 stopped: resumable), seconds, cpu_seconds (of this process), watchdog (the guard
        ran), record (the runner_*.json written).

    Notes
    -----
    Every model of the share is run once unless it is already in the ledgers; a stopped run is resumed by calling
    again with the same table, task and K. A model interrupted by the stop writes nothing; a finished one is
    always packed (a failed model whose failure may come from the stop signal itself is rerun instead,
    :data:`SETTLE`). The stop flag of :mod:`.model` is cleared at the start and at the end.
    """
    # PP 2026-10-02: ported from fw_sphere_task.sh (whole)
    say = log or _log_default
    t_start = time.monotonic()
    nworkers = int(nworkers)
    if nworkers < 1:
        raise ValueError("nworkers must be >= 1")
    retry = int(retry)
    if retry < 0 or (retry and not retry_step > 0):
        raise ValueError("retry must be >= 0 and retry_step > 0")
    if retry * float(retry_step) > TEFF_NUDGE_MAX + 1e-9:
        raise ValueError("retry * retry_step = {} K exceeds the nudge merge_task accepts ({} K)".format(
            retry * retry_step, TEFF_NUDGE_MAX))
    if keep not in mo.KEEP_MODES:
        raise ValueError("keep must be one of {}, got {!r}".format(mo.KEEP_MODES, keep))
    mo._extras_mode(extras)
    k, K = resolve_tasks(task, ntasks)
    tag = tag or task_tag(k, list_name)
    if os.sep in tag or tag in ("", ".", ".."):
        raise ValueError("bad tag {!r}".format(tag))
    template = mo._as_indat(template)
    formal = mo._as_formal(formal)
    if install is None:
        install = FastwindInstall.from_env()
    out = os.path.join(os.path.abspath(results_dir), tag)

    run = _Run(log=say, retry=retry, retry_step=float(retry_step), retry_on=tuple(retry_on), template=template,
               formal=formal, vturb=vturb, iescat=iescat, keep=keep, pnlte_timeout=pnlte_timeout,
               formal_timeout=formal_timeout, extras=extras, grace=grace, env=env, nrow=nrow, max_errors=max_errors,
               pack_child=pack_child, compresslevel=compresslevel, out=out, queue=collections.deque(),
               keep_local=keep_local)

    def handler(signum, frame):              # signal-safe: plain attributes and model.stop_all
        if run.signal is None:
            run.signal = signum
        run.stop.value = True
        mo.stop_all()

    installed = {}
    if signals and threading.current_thread() is threading.main_thread():
        for name in STOP_SIGNALS:
            s = getattr(signal, name)
            installed[s] = signal.signal(s, handler)
    mo.reset_stop()
    local = None
    summary = dict(tag=tag, out_dir=out, task=k, ntasks=K, host=archive.short_host(), pid=os.getpid())
    threads = []
    packer = None
    pack_ok = True
    try:
        # ---- this task's points, minus those already done ---------------------------------------------------
        share = split_table(table, K, k)
        seen, todo, dup = set(), [], 0
        for e in share:
            if e[1] in seen:
                dup += 1
                continue
            seen.add(e[1])
            todo.append(e)
        os.makedirs(out, exist_ok=True)
        rec = archive.recover(out, log=say)
        done = archive.done_set(out)
        before = sum(1 for e in todo if e[1] in done)
        todo = [e for e in todo if e[1] not in done]
        run.queue.extend(todo)
        summary.update(assigned=len(share), duplicates=dup, done_before=before, todo=len(todo), recovered=rec)
        say("task {}/{} tag {} host {} pid {}: {} assigned{}, {} done before, {} to run; {} workers".format(
            k, K, tag, summary["host"], os.getpid(), len(share), " ({} duplicates)".format(dup) if dup else "",
            before, len(todo), nworkers))

        # ---- node-local root: install, run directories, results, stage ------------------------------------
        if todo and not run.stop.value:
            base, why = choose_local_base(local_root, local_need(install, min(nworkers, len(todo)), keep, stage_mode,
                                                                 pack_interval))
            jid = os.environ.get("SLURM_JOB_ID") or "local{}".format(os.getpid())
            os.makedirs(base, exist_ok=True)
            local = tempfile.mkdtemp(prefix="fwsphere_{}_{}_".format(jid, k), dir=base)
            run.local, run.res, run.stage = local, os.path.join(local, "res"), os.path.join(local, "stage")
            os.makedirs(run.res)
            os.makedirs(run.stage)
            if isinstance(install, StagedRoot):
                run.staged = install
            else:
                t0 = time.monotonic()
                run.staged = install.stage(os.path.join(local, "fw"), mode=stage_mode)
                say("local root {} ({}); staged FASTWIND ({}) in {:.1f} s".format(local, why, stage_mode,
                                                                                 time.monotonic() - t0))
            summary["local_root"] = local
            if watchdog:
                try:
                    run.watchdog = _Watchdog(run, run.staged.root, not isinstance(install, StagedRoot)).start()
                except OSError as e:
                    say("cannot start the watchdog: {!r}; no parent-death guard".format(e))

            # ---- workers and packer ----------------------------------------------------------------------
            for i in range(min(nworkers, len(todo))):
                th = threading.Thread(target=run.worker, args=(i,), name="fw-worker-{}".format(i), daemon=True)
                th.start()
                threads.append(th)
            next_pack = time.monotonic() + float(pack_interval)
            noted = False
            while True:
                alive = [th for th in threads if th.is_alive()]
                if not alive:
                    break
                alive[0].join(poll)
                run.tell_watchdog(groups=mo.active_groups())
                if run.stop.value and not noted:
                    noted = True
                    if run.signal is not None:
                        say("{} received: stopping the models of this runner".format(signal.Signals(run.signal).name))
                if packer is not None and packer.poll() is not None:
                    run.periodic_packs += run.note_pack(packer.result())
                    packer = None
                now = time.monotonic()
                if now >= next_pack and not run.stop.value:
                    next_pack = now + float(pack_interval)
                    if packer is None:
                        with run.lock:
                            done_n = sum(run.counts.values())
                            running = run.running
                        say("progress: {} of {} finished ({}), {} running".format(
                            done_n, len(todo), ", ".join("{} {}".format(s, n) for s, n in sorted(run.counts.items()))
                            or "-", running))
                        if pack_child:
                            try:
                                packer = _PackChild(run).start()
                            except OSError as e:
                                say("cannot start the packer child: {!r}".format(e))
                        else:
                            run.pack_seq += 1
                            run.periodic_packs += run.note_pack(dict(archive.pack(
                                run.res, run.stage, run.out, compresslevel=compresslevel,
                                part=archive.part_name(seq=run.pack_seq)), ok=True))
    finally:
        if threads and any(th.is_alive() for th in threads):     # an exception in the main loop: stop the models
            run.stop.value = True
            mo.stop_all()
            for th in threads:
                th.join()
        try:
            if packer is not None:
                run.periodic_packs += run.note_pack(packer.result())
            if local is not None:
                pack_ok = run.final_pack()
        finally:
            if run.watchdog is not None:
                run.watchdog.done()
            for s, h in installed.items():
                signal.signal(s, h)
            mo.reset_stop()
            if local is not None and pack_ok and not keep_local:
                shutil.rmtree(local, ignore_errors=True)

    with run.lock:
        status = dict(sorted(run.counts.items()))
        nstarted = sum(status.values()) + run.interrupted + len(run.errors)
    stopped = run.stop.value and not run.aborted
    # PP 2026-10-02: reviewer: points whose run_model raised (or never started without a stop) are neither recorded
    # nor done, so a run with them is not 'done' (an afterok merge job would take an incomplete task)
    incomplete = bool(run.errors) or (not stopped and len(run.queue) > 0)
    if run.aborted or not pack_ok:
        code = EXIT_ERROR
    elif stopped:
        code = EXIT_STOPPED
    else:
        code = EXIT_ERROR if incomplete else EXIT_OK
    summary.update(status=status, interrupted=run.interrupted, not_started=len(run.queue),
                   errors=run.errors[:100], n_errors=len(run.errors), retried=run.retried, retry_ok=run.retry_ok,
                   parts=run.parts, packed=run.packed, pack_failures=run.pack_failures,
                   periodic_packs=run.periodic_packs, stopped=stopped,
                   signal=signal.Signals(run.signal).name if run.signal else None, aborted=run.aborted,
                   exit_code=code, seconds=round(time.monotonic() - t_start, 3), cpu_seconds=_cpu_seconds(),
                   nworkers=nworkers, started=nstarted, local_kept=bool(local and (not pack_ok or keep_local)),
                   watchdog=run.watchdog is not None and not run.watchdog.dead)
    summary.setdefault("local_root", local)
    rec_path = os.path.join(out, record_name(summary["host"], os.getpid()))
    try:
        atomic_write(rec_path, json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")
        summary["record"] = rec_path
    except OSError as e:
        say("cannot write {}: {!r}".format(rec_path, e))
    ledger_ok = 0
    for p in archive.ledger_paths(out):
        with open(p) as f:
            ledger_ok += sum(1 for ln in f if ln.split()[2:3] == ["ok"])
    how = ("stopped" if stopped else "aborted" if run.aborted else "incomplete" if incomplete else "finished")
    say("task {} {}: {} run ({}), {} interrupted, {} not started, {} errors; {} packed in {} parts; ledgers: {} ok; "
        "exit {}".format(tag, how, nstarted, ", ".join("{} {}".format(s, n) for s, n in status.items()) or "-",
                         run.interrupted, len(run.queue), len(run.errors), run.packed, len(run.parts), ledger_ok,
                         code))
    return summary


def record_name(host, pid, suffix=None):
    """``runner_<UTC>_<host>_<pid>_<n:04d>.json`` (n counts the records of this process; ``suffix`` replaces it)."""
    # PP 2026-10-02: reviewer: two runs of one process within a second wrote the same record name
    return "{}{}_{}_{}_{}.json".format(RUNNER_PREFIX, time.strftime("%Y%m%d_%H%M%SZ", time.gmtime()), host, int(pid),
                                       suffix if suffix else "{:04d}".format(next(_record_seq)))


def _cpu_seconds():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF)
        return round(r.ru_utime + r.ru_stime, 3)
    except (ImportError, OSError):
        return None


# ----------------------------------------------------------------------------------------------------------------
# parent-death guard
# ----------------------------------------------------------------------------------------------------------------
def _under(path, root):
    root = root.rstrip("/")
    return path == root or path.startswith(root + "/")


def _my_processes():
    """(pid, pgid, tty_nr, cwd) of this user's live processes (zombies and unreadable ones skipped)."""
    uid = os.getuid()
    try:
        names = os.listdir("/proc")
    except OSError:
        return
    for name in names:
        if not name.isdigit():
            continue
        d = "/proc/" + name
        try:
            if os.stat(d).st_uid != uid:
                continue
            with open(d + "/stat", "rb") as f:
                st = f.read()
            fl = st[st.rindex(b")") + 2:].split()       # state ppid pgrp session tty_nr ...
            cwd = os.readlink(d + "/cwd")
            yield int(name), int(fl[2]), int(fl[4]), cwd
        except (OSError, ValueError, IndexError):
            continue


def leftover_groups(root, groups=(), private=False):
    """
    Process groups of a dead runner's models that still run: groups with a process whose working directory is below
    ``root`` (the staged root, where the run directories are) and that has no controlling terminal (a user's shell
    in such a directory is left alone), restricted to ``groups`` unless ``private`` (the staged root belongs to that
    runner alone). Never this process's own group.
    """
    # PP 2026-10-02: new (reviewer: a runner killed with SIGKILL left its FASTWIND processes running as orphans)
    own = {os.getpgrp(), os.getpid()}
    groups = set(groups)
    out = set()
    for pid, pgid, tty, cwd in _my_processes():
        if pgid in own or pgid <= 1 or tty != 0 or not _under(cwd, root):
            continue
        if private or pgid in groups:
            out.add(pgid)
    return out


def kill_leftover_groups(root, groups=(), private=False, grace=3.0, log=None):
    """
    SIGTERM the :func:`leftover_groups`, SIGKILL those still there after ``grace`` seconds. Returns the sorted pgids
    signalled.
    """
    # PP 2026-10-02: new
    say = log or (lambda m: None)
    found = leftover_groups(root, groups, private)
    for g in sorted(found):
        mo._killpg(g, signal.SIGTERM)
    t_end = time.monotonic() + float(grace)
    left = set(found)
    while left and time.monotonic() < t_end:
        time.sleep(0.1)
        left &= leftover_groups(root, left, private)
    for g in sorted(left):
        mo._killpg(g, signal.SIGKILL)
    if found:
        say("killed {} leftover process groups ({} needed SIGKILL)".format(len(found), len(left)))
    return sorted(found)


def _alive_cmd(pid, needle):
    """True while ``pid`` runs a command line containing ``needle`` (not a zombie, not a reused pid)."""
    if pid <= 0:
        return False
    try:
        with open("/proc/{}/cmdline".format(pid), "rb") as f:
            cmd = f.read()
        with open("/proc/{}/stat".format(pid), "rb") as f:
            st = f.read()
    except OSError:
        return False
    return needle.encode() in cmd and st[st.rindex(b")") + 2:st.rindex(b")") + 3] != b"Z"


def watchdog_main(argv=None):
    """
    The parent-death guard (``python -m ppmpy.synspec.fastwind.batch watchdog LOCAL RES STAGE OUT ROOT ...``): reads
    the runner's pipe (lines 'G pgid ...', 'P pid', 'DONE') until it ends. After 'DONE' nothing is done. When the pipe
    ends without it (the runner died), it kills the runner's leftover model groups (:func:`kill_leftover_groups`),
    waits for a running packer child, packs the finished results of RES into OUT (:func:`.archive.pack`), writes
    ``OUT/runner_<UTC>_<host>_<runner pid>_watchdog.json`` and removes LOCAL (kept when the pack failed or with
    --keep-local). Returns the exit status.
    """
    # PP 2026-10-02: new (reviewer: parent-death guard for hard kills of the runner)
    ap = argparse.ArgumentParser(prog="python -m ppmpy.synspec.fastwind.batch watchdog")
    ap.add_argument("cmd", choices=["watchdog"])
    for n in ("local", "res", "stage", "out", "root"):
        ap.add_argument(n)
    ap.add_argument("--runner-pid", type=int, required=True)
    ap.add_argument("--private", action="store_true", help="ROOT belongs to this runner alone")
    ap.add_argument("--grace", type=float, default=3.0)
    ap.add_argument("--compresslevel", type=int, default=archive.COMPRESSLEVEL)
    ap.add_argument("--keep-local", action="store_true")
    a = ap.parse_args(argv)
    archive.ignore_signals(CHILD_IGNORED)
    groups, packer, clean = set(), 0, False
    for line in sys.stdin.buffer:
        tok = line.split()
        try:
            if tok[:1] == [b"G"]:
                groups = {int(x) for x in tok[1:]}
            elif tok[:1] == [b"P"]:
                packer = int(tok[1])
            elif tok[:1] == [b"DONE"]:
                clean = True
                break
        except (ValueError, IndexError):
            continue
    if clean:
        return 0
    say = _log_default
    t0 = time.monotonic()
    say("runner {} gone without DONE: cleaning up {}".format(a.runner_pid, a.local))
    killed = kill_leftover_groups(a.root, groups, a.private, a.grace, log=say)
    if _alive_cmd(packer, "ppmpy.synspec.fastwind.archive"):
        say("waiting for the packer child {}".format(packer))
        while _alive_cmd(packer, "ppmpy.synspec.fastwind.archive"):
            time.sleep(0.2)
    rec = dict(watchdog=True, watchdog_pid=os.getpid(), runner_pid=a.runner_pid, host=archive.short_host(),
               local_root=a.local, killed_groups=len(killed), exit_code=None, stopped=False, signal=None, status=None,
               parts=[], packed=0, pack_ok=None)
    ok = True
    if os.path.isdir(a.local) and os.path.isdir(a.out):
        try:
            r = archive.pack(a.res, a.stage, a.out, compresslevel=a.compresslevel, log=say)
            rec.update(parts=[r["part"]] if r["part"] else [], packed=r["npoint"], pack_ok=True)
        except Exception as e:
            ok = False
            rec.update(pack_ok=False, error=repr(e))
            say("pack failed: {!r}; results kept in {}".format(e, a.local))
        rec["seconds"] = round(time.monotonic() - t0, 3)
        try:
            atomic_write(os.path.join(a.out, record_name(rec["host"], a.runner_pid, "watchdog")),
                         json.dumps(rec, indent=1, sort_keys=True) + "\n")
        except OSError as e:
            say("cannot write the record: {!r}".format(e))
    if ok and not a.keep_local:
        shutil.rmtree(a.local, ignore_errors=True)
    say("done: {} groups killed, {} points packed".format(len(killed), rec["packed"]))
    return 0 if ok else 1


# ----------------------------------------------------------------------------------------------------------------
# status
# ----------------------------------------------------------------------------------------------------------------
def _tag_dirs(results_dir, tags):
    try:
        subdirs = sorted(d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))
                         and not d.startswith("."))
    except FileNotFoundError:
        return []
    if tags is None:
        return subdirs
    if isinstance(tags, str):
        return [d for d in subdirs if fnmatch.fnmatchcase(d, tags)]
    return list(tags)


def status(results_dir, tags=None, table=None):
    """
    The state of a run from its ledgers (no archive is opened).

    Parameters
    ----------
    results_dir: str
        ``RUN_DIR/results``.
    tags: None, str or sequence of str
        Tag directories: all (default), a glob pattern, or a list.
    table: str, optional
        A points table ('idx teff' lines): adds the progress against it.

    Returns
    -------
    dict
        tags: per tag parts, ledgers, records, points, duplicates (records - points), status (per point: 'ok' if
        any record is ok, else its first status), orphan_parts, orphan_ledgers, tmp, corrupt, last_runner (exit
        status and counts of the newest runner record, by modification time); total: the same summed over tags
        (points: union);
        table (when given): points, done, todo, ok, failed.
    """
    # PP 2026-10-02: new (the legacy task printed ok / failed counts at its end)
    res = dict(results_dir=os.path.abspath(results_dir), tags={})
    best = {}
    total = collections.Counter()
    for tag in _tag_dirs(results_dir, tags):
        d = os.path.join(results_dir, tag)
        st = archive.scan(d)
        per = {}
        nrec = 0
        for p in archive.ledger_paths(d):
            with open(p) as f:
                for line in f:
                    fl = line.split()
                    if not fl:
                        continue
                    nrec += 1
                    i, s = int(fl[0]), (fl[2] if len(fl) > 2 else "?")
                    if i not in per or (per[i] != "ok" and s == "ok"):
                        per[i] = s
        for i, s in per.items():
            if i not in best or (best[i] != "ok" and s == "ok"):
                best[i] = s
        last = None
        if st["runner_records"]:
            try:
                newest = max(st["runner_records"], key=lambda n: (os.stat(os.path.join(d, n)).st_mtime, n))
                with open(os.path.join(d, newest)) as f:
                    r = json.load(f)
                last = {k: r.get(k) for k in ("exit_code", "stopped", "signal", "status", "todo", "packed",
                                              "interrupted", "not_started", "host", "seconds")}
            except (OSError, ValueError):
                last = None
        info = dict(parts=len(st["parts"]), ledgers=len(st["ledgers"]), records=nrec, points=len(per),
                    duplicates=nrec - len(per), status=dict(sorted(collections.Counter(per.values()).items())),
                    orphan_parts=st["orphan_parts"], orphan_ledgers=st["orphan_ledgers"], tmp=st["tmp"],
                    corrupt=st["corrupt"], last_runner=last)
        res["tags"][tag] = info
        for key in ("parts", "ledgers", "records"):
            total[key] += info[key]
        for key in ("orphan_parts", "orphan_ledgers", "tmp", "corrupt"):
            total[key] += len(info[key])
    res["total"] = dict(total, points=len(best), status=dict(sorted(collections.Counter(best.values()).items())))
    if table is not None:
        idx = set()
        with open(table, "rb") as f:
            for line in f:
                tok = line.split(None, 1)
                if tok and not tok[0].startswith(b"#"):
                    idx.add(int(tok[0]))
        done = idx & set(best)
        nok = sum(1 for i in done if best[i] == "ok")
        res["table"] = dict(points=len(idx), done=len(done), todo=len(idx) - len(done), ok=nok,
                            failed=len(done) - nok, not_in_table=len(set(best) - idx))
    return res


def format_status(st):
    """Text lines of a :func:`status` record."""
    out = ["{}: {} tags".format(st["results_dir"], len(st["tags"]))]
    for tag, t in st["tags"].items():
        extra = []
        for k in ("orphan_parts", "orphan_ledgers", "tmp", "corrupt"):
            if t[k]:
                extra.append("{} {}".format(len(t[k]), k))
        if t["duplicates"]:
            extra.append("{} duplicate records".format(t["duplicates"]))
        lr = t["last_runner"]
        if lr:
            extra.append("last runner exit {}{}".format(lr["exit_code"], " ({})".format(lr["signal"])
                                                        if lr.get("signal") else ""))
        out.append("  {:<28s} {:4d} parts {:8d} points  {}{}".format(
            tag, t["parts"], t["points"], ", ".join("{} {}".format(s, n) for s, n in t["status"].items()) or "-",
            ("; " + "; ".join(extra)) if extra else ""))
    tot = st["total"]
    out.append("  total: {} parts, {} points: {}".format(tot.get("parts", 0), tot["points"], ", ".join(
        "{} {}".format(s, n) for s, n in tot["status"].items()) or "-"))
    if "table" in st:
        tb = st["table"]
        out.append("  table: {points} points, {done} done ({ok} ok, {failed} failed), {todo} to run".format(**tb))
    return out


# ----------------------------------------------------------------------------------------------------------------
# self test (host python, no pytest)
# ----------------------------------------------------------------------------------------------------------------
MODULES = ("ppmpy.synspec.fastwind.batch", "ppmpy.synspec.fastwind.archive", "ppmpy.synspec.fastwind.__main__")


def selftest(workdir=None, verbose=True):
    """
    Checks without numpy and pytest (for the host python): stdlib-only imports of batch / archive / __main__, a
    fake run of 16 points with 4 workers (a deterministic failure retried at +1 K; the packer child and the watchdog
    without failure), a CLI run stopped by SIGUSR1 (exit 3) and resumed (every point exactly once in the ledgers),
    and the recovery of an archive whose ledger was lost. Raises AssertionError on a failure; returns a dict of
    timings.
    """
    # PP 2026-10-02: new (M6)
    from . import fake
    t0 = time.time()

    def say(*a):
        if verbose:
            print(*a)
            sys.stdout.flush()

    ok, msg = fake.check_stdlib_only(modules=fake.MODULES + MODULES)
    assert ok, msg
    say("stdlib-only:", msg)
    own = workdir is None
    work = workdir or tempfile.mkdtemp(prefix="fwbatch_")
    try:
        inst = fake.install_fake(os.path.join(work, "fw"))
        tpl, fi = os.path.join(inst.root, "INDAT.template"), os.path.join(inst.root, "FORMAL_INPUT")
        pts = os.path.join(work, "points.txt")
        teffs = ["%.3f" % (37000.0 + 37.5 * i) for i in range(16)]
        with open(pts, "w") as f:
            f.write("".join("{} {}\n".format(i, t) for i, t in enumerate(teffs)))
        fake.set_fake_config(inst.build, fail=[teffs[5]])
        s = run_models(pts, inst, tpl, fi, os.path.join(work, "res1"), task=0, ntasks=1, nworkers=4, retry=1,
                       local_root=os.path.join(work, "local"), log=(lambda m: None))
        assert s["exit_code"] == 0 and s["status"] == {"ok": 16} and s["retried"] == 1, s
        assert s["pack_failures"] == 0 and s["watchdog"], s
        led = archive.ledger_paths(os.path.join(work, "res1", "task_0000"))
        recs = [r for p in led for r in archive.read_index(p)]
        assert sorted(r.idx for r in recs) == list(range(16)), recs
        r5 = [r for r in recs if r.idx == 5][0]
        assert r5.teff == "%.3f" % (float(teffs[5]) + 1) and r5.status == "ok", r5
        st = status(os.path.join(work, "res1"), table=pts)
        assert st["table"]["done"] == 16 and st["table"]["todo"] == 0, st
        say("run of 16 points with a retry ok")
        # CLI run stopped by SIGUSR1, then resumed
        fake.set_fake_config(inst.build, sleep=0.25)
        res2 = os.path.join(work, "res2")
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("SLURM_") and k not in ("FW_TASK", "FW_NTASKS", "NW", "KEEP_MODEL")}
        env["PYTHONPATH"] = PKG_ROOT
        cmd = [sys.executable, "-m", "ppmpy.synspec.fastwind", "run", work, "--table", pts, "--results", res2,
               "--root", inst.root, "--build", os.path.basename(inst.build), "--template", tpl, "--formal", fi,
               "--nworkers", "4", "--pack-interval", "0.5", "--local-root", os.path.join(work, "local")]
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out2 = os.path.join(res2, "task_0000")
        t = time.time()
        while time.time() - t < 60 and not archive.ledger_paths(out2):
            time.sleep(0.05)
        p.send_signal(signal.SIGUSR1)
        o, _ = p.communicate(timeout=120)
        assert p.returncode == EXIT_STOPPED, (p.returncode, o.decode()[-2000:])
        n1 = len(archive.done_set(out2))
        assert 0 < n1 < 16, n1
        p = subprocess.run(cmd + ["--json"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        assert p.returncode == 0, p.stdout.decode()[-2000:]
        o = p.stdout.decode()
        s = json.loads(o[o.index("\n{") + 1:])
        assert s["pack_failures"] == 0 and s["watchdog"], s
        recs = [r for q in archive.ledger_paths(out2) for r in archive.read_index(q)]
        assert sorted(r.idx for r in recs) == list(range(16)), sorted(r.idx for r in recs)
        say("CLI stop (exit 3 after {} points) and resume ok".format(n1))
        # an archive whose ledger was lost
        q = archive.ledger_paths(out2)[0]
        data = open(q, "rb").read()
        os.unlink(q)
        r = archive.recover(out2)
        assert len(r["rebuilt"]) == 1 and open(q, "rb").read() == data, r
        say("orphan archive recovered")
    finally:
        if own:
            shutil.rmtree(work, ignore_errors=True)
    secs = time.time() - t0
    say("batch selftest passed in %.1f s" % secs)
    return dict(seconds=secs)


if __name__ == "__main__":
    if sys.argv[1:2] == ["watchdog"]:
        sys.exit(watchdog_main())
    sys.exit("usage: python -m ppmpy.synspec.fastwind.batch watchdog LOCAL RES STAGE OUT ROOT --runner-pid PID ... "
             "(started by run_models; the command line is python3 -m ppmpy.synspec.fastwind)")
