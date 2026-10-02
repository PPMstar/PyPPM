"""
Run one FASTWIND model (pnlte + pformalsol) in its own run directory, and rerun pformalsol from saved model files
(standard library only).

:func:`run_model` is fw_sphere_point.sh in Python. Its output directory ``<res_dir>/<name>/`` holds what the legacy
worker wrote, byte-compatible, so the packer's parts and ``ppmpy.synspec.fwresults`` (merge_task, locate_points,
extract_points, check_indat_premise) read them unchanged:

``INDAT.DAT``
    The template with MODNAM = name and TEFF = '%.3f' of T_eff (:mod:`.indat`; byte-identical to the awk edit).
``OUT.*``
    pformalsol's line profiles (all ``OUT.*`` files of the model, as ``cp $name/OUT.*``).
``MODEL NLTE_POP LTE_POP ENION TAU_ROS FLUXCONT CLUMPING_OUTPUT CONT_FORMAL CONT_FORMAL_ALL``
    Only with ``keep='model'`` and status ok (KEEP_MODEL=1): what a pformalsol rerun needs.
``pnlte_tail.log``
    Last 40 lines of pnlte.log, for a status other than ok.
``meta.txt``
    ``idx teff status niter T_tau23 t_pnlte t_formal`` (awk ``"%d %s %s %d %s %.1f %.1f\\n"``; T_eff as given,
    T_tau23 the verbatim last field of the last 'T(TAUROSS=2/3)' line or 'nan').

Status: as the legacy worker ('ok', 'formal_failed', 'pnlte_failed', 'pnlte_timeout'), except that pformalsol
counts as successful only when it exited 0 within ``formal_timeout`` and every expected output is complete
(:func:`.formal.out_problem`): pformalsol opens each OUT file before computing it, so a killed or crashed run leaves
empty or truncated files that the legacy ``ls OUT.* | wc -l`` test took for success.

Additions (extra files, ignored by the readers), chosen with ``extras``:

``'full'`` (or True, the default)
    ``CONVERG`` and ``MAXTCORR.dat`` (the convergence history) and ``convergence.json`` (status, flags,
    convergence verdict of :mod:`.logs`, return codes, times, host). Cost: ~6 kB raw per point (real CONVERG
    ~3.7 kB, MAXTCORR ~0.4 kB, JSON 1.2-2 kB), ~2 kB after gzip, i.e. ~+25 % on the 7.7 kB/point profiles-only
    parts (~2-3 GB for 1.24 M points).
``'digest'``
    Only ``convergence.txt``, one line of ``key=value`` pairs (:data:`DIGEST_KEYS`; ~0.3 kB): enough to select
    unconverged or capped models in a production run.
``False``
    Nothing: exactly the legacy file set.

``OUT_IMU.*`` are copied too when the formal build writes them (``keep_imu``).

The result appears atomically: everything is written to ``<res_dir>/<name>.tmp`` and renamed. The run directory
``<staged root>/<name>`` (links to bin/, INDAT.DAT, FORMAL_INPUT, the model directory) is removed afterwards.
Replacing an existing result of the same name is not atomic for readers: the old one is first renamed to a hidden
name (``.<name>.old...``, skipped by ``ls``-based packers), so for a moment neither exists under the final name.

Processes
---------
pnlte and pformalsol run in their own session / process group (``start_new_session``), with stdout and stderr
in ``pnlte.log`` / ``pformalsol.log``. At the time limit the whole group gets SIGTERM, then SIGKILL after a grace
period; after a normal exit any process left in the group is killed too. The leader is waited for without being
reaped (``waitid(WNOWAIT)``) until its group has been signalled, so a process group id is never reused by another
process when we signal it. Nothing is killed by name. The soft stack limit is raised to the hard limit once per
process (``ulimit -s unlimited`` of the legacy scripts; FASTWIND needs a large stack).

:func:`stop_all` kills every group started by this module and makes running :func:`run_model` calls raise
:class:`ModelInterrupted` without writing a result (the legacy SIGUSR1 stop: an interrupted point records nothing
and is rerun later). It may be called from a signal handler: the registry lock is re-entrant (a handler that
interrupts the main thread inside the lock does not deadlock), the stop flag is a plain attribute (no lock), and a
group is removed from the registry before its leader is reaped, so a group id that is signalled is never one the
system has reused. Time limits use :func:`time.monotonic` (immune to wall-clock steps); the times in meta.txt are
durations on the same clock.

PP 2026-10-02: new (M6); ported from fw_sphere_point.sh (whole) and fw_imu_run.sh:12-20 (one model) of the project
stellar-atmosphere-KU-Leuven.
"""
import glob
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import threading
import time

from .formal import OUT_NROW, FormalInput, VTURB_M424, formal_suffix, formalsol_stdin, out_name, out_problem
from .indat import Indat
from .install import StagedRoot, atomic_symlink, atomic_write, tmp_path
from .logs import ACABOSE, classify, convergence, parse_pnlte_log, tail_bytes

MODEL_FILES = ("MODEL", "NLTE_POP", "LTE_POP", "ENION", "TAU_ROS", "FLUXCONT", "CLUMPING_OUTPUT", "CONT_FORMAL",
               "CONT_FORMAL_ALL")
"""Model files kept with keep='model' (fw_sphere_point.sh:47) and linked by a pformalsol rerun (fw_imu_run.sh:17)."""

EXTRA_FILES = ("CONVERG", "MAXTCORR.dat")
"""Convergence history copied into the result (an addition to the legacy layout)."""

CONVERGENCE_FILE = "convergence.json"
DIGEST_FILE = "convergence.txt"
DIGEST_KEYS = ("status", "flags", "niter", "n_iter", "converged", "converged_it", "temp_converged_it", "cap",
               "emax_last", "meanerr_last", "n_tcorr", "returncode", "formal_returncode")
"""Keys of the one-line digest of ``extras='digest'`` (flags joined by ',', '-' for none; None as 'None')."""
EXTRAS_MODES = ("full", "digest", "none")
META_FILE = "meta.txt"
TAIL_FILE = "pnlte_tail.log"
TAIL_LINES = 40
NAME_FMT = "P{idx:06d}"
"""Run / result directory of a point (fw_sphere_point.sh: ``printf P%06d``)."""

KEEP_MODES = ("profiles", "model")

_lock = threading.RLock()     # re-entrant: stop_all may run in a signal handler of a thread inside the lock
_active = {}                 # pgid -> command, for stop_all; removed before the leader is reaped


class _Flag:
    """A stop flag without locks (threading.Event.set takes a lock, unsafe in a signal handler)."""
    value = False


_stop = _Flag()
_stack_raised = False


class ModelInterrupted(RuntimeError):
    """:func:`stop_all` was called while a model ran; nothing was written for it."""


class ModelBusy(RuntimeError):
    """A model of the same name is already running in the same root (another thread or process)."""


def _raise_stack_limit():
    """Raise the soft stack limit to the hard limit, once per process (children inherit it)."""
    global _stack_raised
    with _lock:
        if _stack_raised:
            return
        _stack_raised = True
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        if soft != hard and (hard == resource.RLIM_INFINITY or soft < hard):
            resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))
    except (ImportError, ValueError, OSError):
        pass


def active_groups():
    """Process group ids of the FASTWIND processes this module is running (dict pgid -> command)."""
    with _lock:
        return dict(_active)


def stop_all(sig=signal.SIGTERM):
    """
    Stop every running model of this process: set the stop flag (running :func:`run_model` calls raise
    :class:`ModelInterrupted` and write nothing) and send ``sig`` to their process groups (escalated to SIGKILL by
    the waiting call after its grace period). Returns the number of groups signalled.

    Safe in a signal handler (SIGUSR1 / SIGTERM of a runner): see the module notes.
    """
    # PP 2026-10-02: replaces stop() of fw_sphere_task.sh:70-77 (pkill by name) with process groups
    _stop.value = True
    n = 0
    with _lock:                  # held while signalling: no listed group can be reaped (and its id reused) meanwhile
        for pgid in list(_active):
            if _killpg(pgid, sig):
                n += 1
    return n


def reset_stop():
    """Clear the stop flag set by :func:`stop_all`."""
    _stop.value = False


def stop_requested():
    """True after :func:`stop_all` (until :func:`reset_stop`)."""
    return _stop.value


def _exited(pid):
    """True when the child ``pid`` has exited (not reaped: WNOWAIT keeps its pid and process group id reserved)."""
    try:
        r = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return True
    return r is not None and r.si_pid == pid


def _killpg(pgid, sig):
    try:
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def run_group(cmd, cwd, log_path, stdin=None, timeout=None, grace=5.0, env=None):
    """
    Run ``cmd`` in its own process group with stdout + stderr in ``log_path``.

    Parameters
    ----------
    cmd: list of str
    cwd: str
    log_path: str
        Created / truncated.
    stdin: bytes, optional
        Written to the process's standard input, which is then closed (default /dev/null).
    timeout: float, optional
        Seconds; then SIGTERM to the group, SIGKILL after ``grace`` seconds.
    env: dict, optional
        Environment (default inherited).

    Returns
    -------
    returncode: int
        Exit status (negative: killed by that signal).
    timed_out: bool
    elapsed: float
        Seconds (:func:`time.monotonic`).
    """
    # PP 2026-10-02: replaces `timeout N ./pnlte...` (fw_sphere_point.sh:28), which signals the child only
    t0 = time.monotonic()
    with open(log_path, "wb") as log:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=env, close_fds=True,
                             stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                             start_new_session=True)
    pgid = p.pid
    with _lock:
        _active[pgid] = list(cmd)
    timed_out = False
    try:
        if stdin is not None:
            try:
                p.stdin.write(stdin)
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    p.stdin.close()
                except OSError:
                    pass
        deadline = None if timeout is None else t0 + float(timeout)
        nap = 0.01
        term_at = None
        while not _exited(p.pid):
            now = time.monotonic()
            if term_at is None and deadline is not None and now >= deadline:
                timed_out = True
                term_at = now
                _killpg(pgid, signal.SIGTERM)
            elif term_at is None and _stop.value:
                term_at = now                    # stop_all signalled the group (or ran before it was registered)
                _killpg(pgid, signal.SIGTERM)
            if term_at is not None and now >= term_at + grace:
                _killpg(pgid, signal.SIGKILL)
            time.sleep(nap)
            nap = min(nap * 1.5, 0.1)
    finally:
        if not _exited(p.pid):                   # interrupted (exception in this thread): never leave it running
            _killpg(pgid, signal.SIGKILL)
        _killpg(pgid, signal.SIGKILL)            # leader exited (not reaped): remove what is left in its group
        with _lock:                              # unregister while the zombie leader still holds the group id,
            _active.pop(pgid, None)              # so stop_all never signals an id the system may have reused
        rc = p.wait()
    return rc, timed_out, time.monotonic() - t0


def _teff_text(teff):
    """meta.txt's T_eff: the text as given (awk '%s' of the points.txt field), else repr / str of the number."""
    if isinstance(teff, str):
        return teff
    if isinstance(teff, bool):
        raise ValueError("T_eff must be a number or text, got a bool")
    if isinstance(teff, int):
        return str(teff)
    return repr(float(teff))


def _spec(spec, template):
    """(idx, meta T_eff text, INDAT fields) of a model specification."""
    try:
        idx, x = spec
    except (TypeError, ValueError):
        raise ValueError("spec must be (idx, teff) or (idx, {{field: value}}), got {!r}".format(spec)) from None
    idx = int(idx)
    if idx < 0:
        raise ValueError("idx must be >= 0, got {}".format(idx))
    if isinstance(x, dict):
        fields = dict(x)
        if "MODNAM" in fields:
            raise ValueError("MODNAM is set from name_fmt, not from the spec")
        if "TEFF" not in fields:
            return idx, template.raw("TEFF"), fields
        x = fields["TEFF"]
    else:
        fields = {}
    text = _teff_text(x)
    v = float(x)
    if not math.isfinite(v) or v <= 0:
        raise ValueError("T_eff must be a positive finite number, got {!r}".format(x))
    fields["TEFF"] = v                       # written '%.3f' (awk sprintf of the points.txt text)
    return idx, text, fields


def _as_indat(template):
    if isinstance(template, Indat):
        return template
    if isinstance(template, (bytes, bytearray)) or (isinstance(template, str) and "\n" in template):
        return Indat(template)
    return Indat.read(template)


def _as_formal(formal):
    if isinstance(formal, FormalInput):
        return formal
    if isinstance(formal, (bytes, bytearray)) or (isinstance(formal, str) and "\n" in formal):
        return FormalInput.from_text(formal)
    return FormalInput.read(formal)


def _rmtree(path):
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _jsonable(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def format_meta(idx, teff_text, status, niter, t_tau23, t_pnlte, t_formal):
    """One meta.txt line: ``"%d %s %s %d %s %.1f %.1f\\n"`` (fw_sphere_point.sh:52-53)."""
    # PP 2026-10-02: ported from fw_sphere_point.sh:52-53 (awk printf)
    return "%d %s %s %d %s %.1f %.1f\n" % (int(idx), teff_text, status, int(niter),
                                           t_tau23 if t_tau23 else "nan", t_pnlte, t_formal)


def _move_into_place(tmp, final):
    """
    Rename the finished directory to its final name. An existing result of the same name is replaced: it is first
    renamed to a hidden name (``.<name>.old<pid>.<thread>.<n>``, skipped by ``ls``-based packers such as
    fw_sphere_task.sh's ``ls RES | grep -v '\\.tmp$'``), so the replacement is not atomic for readers.
    """
    try:
        os.rename(tmp, final)
        return
    except OSError:
        if not os.path.lexists(final):
            raise
    old = tmp_path(final, "old")
    os.rename(final, old)
    os.rename(tmp, final)
    _rmtree(old)


def _extras_mode(extras):
    if extras is True:
        return "full"
    if extras is False or extras is None:
        return "none"
    if extras in EXTRAS_MODES:
        return extras
    raise ValueError("extras must be True / 'full', 'digest' or False / 'none', got {!r}".format(extras))


def format_digest(rec):
    """
    The one-line digest of ``extras='digest'``: ``key=value`` for :data:`DIGEST_KEYS`, blank-separated, from a
    record with the keys of :func:`run_model`'s result and its ``convergence`` dict merged in.
    """
    # PP 2026-10-02: new (M6); reviewer: the full CONVERG history costs ~+25 % of a profiles-only part
    out = []
    for k in DIGEST_KEYS:
        v = rec.get(k)
        if k == "flags":
            v = ",".join(v) if v else "-"
        elif isinstance(v, float):
            v = repr(v)
        out.append("{}={}".format(k, v))
    return " ".join(out) + "\n"


def parse_digest(text):
    """dict of a :func:`format_digest` line (values as text; flags as a tuple)."""
    # PP 2026-10-02: new (M6)
    out = {}
    for item in text.split():
        k, v = item.split("=", 1)
        out[k] = tuple(v.split(",")) if k == "flags" and v != "-" else (() if k == "flags" else v)
    return out


def formal_problems(model_dir, names, nrow=OUT_NROW, kind="OUT"):
    """dict name -> :func:`.formal.out_problem` for the expected outputs ``names`` of ``model_dir`` not complete."""
    # PP 2026-10-02: new (M6)
    out = {}
    for n in names:
        why = out_problem(os.path.join(model_dir, n), kind, nrow)
        if why is not None:
            out[n] = why
    return out


def run_model(staged, spec, res_dir, template, formal, vturb=VTURB_M424, iescat=0, keep="profiles",
              pnlte_timeout=3600, formal_timeout=600, name_fmt=NAME_FMT, extras=True, keep_imu=True, grace=5.0,
              env=None, keep_run=False, nrow=OUT_NROW):
    """
    Run pnlte and pformalsol for one model and write its result directory (fw_sphere_point.sh).

    Parameters
    ----------
    staged: StagedRoot or str
        Run root from :meth:`FastwindInstall.stage` (or its path).
    spec: tuple
        ``(idx, teff)``: point index and T_eff (number, or the text of points.txt, kept verbatim in meta.txt);
        or ``(idx, {field: value})``: INDAT fields to set (:class:`Indat` names; meta.txt's T_eff is the TEFF given,
        else the template's).
    res_dir: str
        Directory of finished results (created); the result is ``<res_dir>/<name>``.
    template: Indat, str
        INDAT template (object, path or text). MODNAM is set to the name.
    formal: FormalInput, str
        Line list (object, path or text); written as the run's FORMAL_INPUT (its original text when read from a
        file) and used for the expected OUT names.
    vturb, iescat:
        pformalsol's answers (:func:`formalsol_stdin`; M424: '10 0.1', 0 -> OUT.<line>_VTV010).
    keep: str
        'profiles' (KEEP_MODEL=0) or 'model' (KEEP_MODEL=1: also the files of :data:`MODEL_FILES`, for status ok).
    pnlte_timeout, formal_timeout: float
        Seconds before the process group is killed (pnlte: status 'pnlte_timeout'; pformalsol: 'formal_failed',
        flag 'formal_timeout').
    name_fmt: str
        Name of the run and result directories and MODNAM, formatted with ``idx``.
    extras: bool or str
        True / 'full': also write CONVERG, MAXTCORR.dat and convergence.json (~2 kB per point gzipped); 'digest':
        only the one-line convergence.txt; False / 'none': the legacy file set (see the module notes).
    keep_imu: bool
        Also copy OUT_IMU.* (written by a patched pformalsol); with a staged root whose pformalsol has the patch,
        they must be complete for status ok.
    grace: float
        Seconds between SIGTERM and SIGKILL.
    env: dict, optional
        Environment of the FASTWIND processes.
    keep_run: bool
        Keep the run directory (debugging).
    nrow: int or None
        Rows every OUT table must have for status ok (:data:`.formal.OUT_NROW` = 161, M424); None: at least one.

    Returns
    -------
    dict
        idx, name, teff (meta text), status, flags, niter, T_tau23, t_pnlte, t_formal, result_dir, meta (the
        meta.txt line), returncode, formal_returncode, timed_out, formal_timed_out, formal_problems (name -> why
        for incomplete expected outputs), convergence, log (digest without the tail).

    Raises
    ------
    ModelInterrupted
        :func:`stop_all` was called meanwhile (nothing written; the run directory is removed).
    ModelBusy
        The same name is running in the same staged root (nothing touched).
    """
    # PP 2026-10-02: ported from fw_sphere_point.sh:14-56
    if isinstance(staged, str):
        staged = StagedRoot.open(staged)
    if keep not in KEEP_MODES:
        raise ValueError("keep must be one of {}, got {!r}".format(KEEP_MODES, keep))
    xmode = _extras_mode(extras)
    template = _as_indat(template)
    formal = _as_formal(formal)
    idx, teff_text, fields = _spec(spec, template)
    if any(c.isspace() for c in teff_text) or not teff_text:
        raise ValueError("T_eff text {!r} would break meta.txt".format(teff_text))
    name = name_fmt.format(idx=idx)
    suffix = formal_suffix(vturb, iescat)
    stdin = formalsol_stdin(name, vturb, iescat)
    indat = template.copy().set(MODNAM=name, **fields)
    if _stop.value:
        raise ModelInterrupted(name)

    with _NameLock(staged.root, name):
        return _run_model_locked(staged, name, idx, teff_text, indat, formal, stdin, suffix, res_dir, keep,
                                 pnlte_timeout, formal_timeout, xmode, keep_imu, grace, env, keep_run, nrow)


def _run_model_locked(staged, name, idx, teff_text, indat, formal, stdin, suffix, res_dir, keep, pnlte_timeout,
                      formal_timeout, xmode, keep_imu, grace, env, keep_run, nrow):
    run = os.path.join(staged.root, name)
    os.makedirs(res_dir, exist_ok=True)
    final = os.path.join(os.path.abspath(res_dir), name)
    out = final + ".tmp"
    _rmtree(run)
    _rmtree(out)
    mdir = os.path.join(run, name)
    os.makedirs(mdir)
    finished = False
    try:
        for f in staged.bin_files():
            os.symlink(os.path.join(staged.bin_dir, f), os.path.join(run, f))
        indat.write(os.path.join(run, "INDAT.DAT"))
        atomic_write(os.path.join(run, "FORMAL_INPUT"), formal.to_text().encode("latin-1"))
        _raise_stack_limit()

        rc, timed_out, t_pnlte = run_group(list(staged.launcher) + ["./" + staged.pnlte], run,
                                           os.path.join(run, "pnlte.log"), timeout=pnlte_timeout, grace=grace, env=env)
        t1 = time.monotonic()
        if _stop.value:
            raise ModelInterrupted(name)
        with open(os.path.join(run, "pnlte.log"), "rb") as fh:
            acabose = ACABOSE in fh.read()          # grep -q "ESTO ES EL ACABOSE" (fw_sphere_point.sh:30)
        expected = [out_name(ln, suffix) for ln in formal.names]
        formal_ok, rc_f, to_f, probs = None, None, False, {}
        if acabose:
            rc_f, to_f, _ = run_group(list(staged.launcher) + ["./" + staged.pformalsol], run,
                                      os.path.join(run, "pformalsol.log"), stdin=stdin, timeout=formal_timeout,
                                      grace=grace, env=env)
            if _stop.value:
                raise ModelInterrupted(name)
            # stricter than fw_sphere_point.sh:31-32 (ls OUT.* | wc -l): pformalsol opens each OUT file before
            # computing it, so a killed or crashed run leaves empty / truncated files under the final names
            probs = formal_problems(mdir, expected, nrow, "OUT")
            if keep_imu:
                imu = [out_name(ln, suffix, "OUT_IMU") for ln in formal.names]
                if not staged.has_imu:           # unknown or unpatched: check only what was written
                    imu = [n for n in imu if os.path.lexists(os.path.join(mdir, n))]
                probs.update(formal_problems(mdir, imu, nrow, "OUT_IMU"))
            formal_ok = (not to_f) and rc_f == 0 and not probs
        t_formal = time.monotonic() - t1
        log = parse_pnlte_log(os.path.join(run, "pnlte.log"), tail=0)
        try:
            itstart, itmore = int(indat.get("ITSTART")), int(indat.get("ITMORE"))
            enatcor = bool(indat.get("ENATCOR"))
        except (KeyError, ValueError):
            itstart, itmore, enatcor = 0, None, True
        conv = convergence(mdir, itstart, itmore, enatcor)
        status, flags = classify(log, formal_ok, timed_out, rc, conv, formal_timed_out=to_f, formal_returncode=rc_f,
                                 formal_problems=probs)

        os.makedirs(out)
        shutil.copyfile(os.path.join(run, "INDAT.DAT"), os.path.join(out, "INDAT.DAT"))
        for f in sorted(glob.glob(os.path.join(glob.escape(mdir), "OUT.*"))):
            shutil.copyfile(f, os.path.join(out, os.path.basename(f)))
        if keep_imu:
            for f in sorted(glob.glob(os.path.join(glob.escape(mdir), "OUT_IMU.*"))):
                shutil.copyfile(f, os.path.join(out, os.path.basename(f)))
        if keep == "model" and status == "ok":
            for f in MODEL_FILES:
                src = os.path.join(mdir, f)
                if os.path.isfile(src):
                    shutil.copyfile(src, os.path.join(out, f))
        if status != "ok":
            with open(os.path.join(out, TAIL_FILE), "wb") as fh:
                fh.write(tail_bytes(os.path.join(run, "pnlte.log"), TAIL_LINES))
        meta = format_meta(idx, teff_text, status, log["niter"], log["T_tau23"], t_pnlte, t_formal)
        if xmode == "full":
            for f in EXTRA_FILES:
                src = os.path.join(mdir, f)
                if os.path.isfile(src):
                    shutil.copyfile(src, os.path.join(out, f))
            digest = {k: v for k, v in log.items() if k != "tail"}
            rec = dict(idx=idx, name=name, teff=teff_text, status=status, flags=list(flags), niter=log["niter"],
                       T_tau23=log["T_tau23"], t_pnlte=t_pnlte, t_formal=t_formal, returncode=rc,
                       formal_returncode=rc_f, timed_out=timed_out, formal_timed_out=to_f, formal_problems=probs,
                       suffix=suffix, expected=expected, convergence=conv, log=digest, host=socket.gethostname(),
                       launcher=list(staged.launcher))
            with open(os.path.join(out, CONVERGENCE_FILE), "w") as fh:
                json.dump(_jsonable(rec), fh, indent=1, sort_keys=True)
                fh.write("\n")
        elif xmode == "digest":
            rec = dict(conv, status=status, flags=flags, niter=log["niter"], returncode=rc, formal_returncode=rc_f)
            with open(os.path.join(out, DIGEST_FILE), "w") as fh:
                fh.write(format_digest(rec))
        with open(os.path.join(out, META_FILE), "w") as fh:
            fh.write(meta)
        _move_into_place(out, final)             # atomic: a packer only ever sees complete results
        finished = True
    finally:
        if not finished:
            _rmtree(out)
        if not keep_run:
            _rmtree(run)
    return dict(idx=idx, name=name, teff=teff_text, status=status, flags=flags, niter=log["niter"],
                T_tau23=log["T_tau23"], t_pnlte=t_pnlte, t_formal=t_formal, result_dir=final, meta=meta,
                returncode=rc, formal_returncode=rc_f, timed_out=timed_out, formal_timed_out=to_f,
                formal_problems=probs, convergence=conv, log={k: v for k, v in log.items() if k != "tail"})


class _NameLock:
    """
    Exclusive use of the run directory ``<root>/<name>`` while a model runs in it: an flock on the hidden file
    ``<root>/.<name>.lock`` (released by the kernel if the process dies, so a stale run directory is still cleaned
    up by the next run), removed again before release. A second run of the same name in the same root (another
    thread or process) raises :class:`ModelBusy` instead of deleting the first one's directory.
    """

    # PP 2026-10-02: new (M6); the legacy workers never ran one point twice at once in a node-local root
    def __init__(self, root, name):
        self.path = os.path.join(root, ".{}.lock".format(name))
        self.name = name
        self.fh = None

    def __enter__(self):
        import fcntl
        while True:
            fh = open(self.path, "a+")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fh.close()
                raise ModelBusy("model {} is already running in {}".format(self.name, os.path.dirname(self.path)))
            try:
                same = os.fstat(fh.fileno()).st_ino == os.stat(self.path).st_ino
            except FileNotFoundError:
                same = False
            if same:                              # not unlinked by a previous holder meanwhile
                self.fh = fh
                return self
            fh.close()

    def __exit__(self, *exc):
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        self.fh.close()                           # releases the lock
        return False


def _ensure_root_links(root, staged):
    """``root/inicalc`` and ``root/HOPFPARA_ALL_*`` pointing at a staged root (fw_imu_run.sh:9-11)."""
    os.makedirs(root, exist_ok=True)
    if os.path.abspath(root) == staged.root:
        return
    for n in ("inicalc", "HOPFPARA_ALL_HHe", "HOPFPARA_ALL_met"):
        dst = os.path.join(root, n)
        if not os.path.lexists(dst):
            atomic_symlink(os.path.join(staged.root, n), dst)


def rerun_formal(staged, model_dir, formal, run_root=None, name=None, vturb=VTURB_M424, iescat=0, kind="OUT_IMU",
                 timeout=600, overwrite=False, grace=5.0, env=None, nrow=OUT_NROW):
    """
    Rerun pformalsol for a saved model (fw_imu_run.sh, one model): the model files are linked, not copied.

    Parameters
    ----------
    staged: StagedRoot or str
        Run root; its pformalsol is used (stage from an install with ``formal_build`` = the patched build for
        OUT_IMU).
    model_dir: str
        Directory with the files of :data:`MODEL_FILES` and INDAT.DAT (an extracted point, ``P<idx>/``).
    formal: FormalInput or str
        Line list.
    run_root: str, optional
        Where the run directory ``<run_root>/<name>`` is made (default the staged root); links to the staged
        inicalc and Hopf tables are added when it is another directory.
    name: str, optional
        Model name (default the basename of ``model_dir``, as the legacy script).
    kind: str
        Outputs that must appear: 'OUT_IMU' (needs the patched pformalsol) or 'OUT'.
    overwrite: bool
        Rerun even when every expected output is complete (default: skip, as fw_imu_run.sh, which tests only that
        they exist; an empty or truncated output from a killed run is rerun here).
    nrow: int or None
        Rows the output tables must have (:data:`.formal.OUT_NROW`).

    Returns
    -------
    dict
        name, run_dir, model_dir (``<run>/<name>``, where the outputs are), outputs (expected paths), status
        ('ok': exit 0 within ``timeout`` and every output complete; 'failed'; 'skipped'), problems (output name ->
        why it is not complete), returncode, timed_out, elapsed, missing_model_files.

    Notes
    -----
    Thread-safe: many calls (other models) may share one ``run_root``; its links are made with unique temporaries.
    """
    # PP 2026-10-02: ported from fw_imu_run.sh:12-20
    if isinstance(staged, str):
        staged = StagedRoot.open(staged)
    formal = _as_formal(formal)
    if kind == "OUT_IMU" and staged.has_imu is False:
        raise ValueError("the staged pformalsol has no intensity patch (no OUT_IMU); stage with formal_build")
    src = os.path.abspath(model_dir)
    name = name or os.path.basename(os.path.normpath(src))
    root = os.path.abspath(run_root) if run_root else staged.root
    _ensure_root_links(root, staged)
    run = os.path.join(root, name)
    m = os.path.join(run, name)
    suffix = formal_suffix(vturb, iescat)
    outputs = [os.path.join(m, out_name(ln, suffix, kind)) for ln in formal.names]
    res = dict(name=name, run_dir=run, model_dir=m, outputs=outputs, returncode=None, timed_out=False, elapsed=0.0,
               missing_model_files=[f for f in MODEL_FILES if not os.path.isfile(os.path.join(src, f))])
    names = [os.path.basename(p) for p in outputs]
    if not overwrite:
        probs = formal_problems(m, names, nrow, kind)
        if not probs:
            res.update(status="skipped", problems={})
            return res
    with _NameLock(root, name):
        return _rerun_formal_locked(staged, src, name, run, m, formal, outputs, vturb, iescat, timeout, grace, env,
                                    res, kind, nrow)


def _rerun_formal_locked(staged, src, name, run, m, formal, outputs, vturb, iescat, timeout, grace, env, res, kind,
                         nrow):
    os.makedirs(m, exist_ok=True)
    for f in (staged.pformalsol, "ATOM_FILE", staged.atom):
        atomic_symlink(os.path.join(staged.bin_dir, f), os.path.join(run, f))
    for f in MODEL_FILES:
        atomic_symlink(os.path.join(src, f), os.path.join(m, f))
    shutil.copyfile(os.path.join(src, "INDAT.DAT"), os.path.join(run, "INDAT.DAT"))
    atomic_write(os.path.join(run, "FORMAL_INPUT"), formal.to_text().encode("latin-1"))
    for p in outputs:
        if os.path.lexists(p):
            os.unlink(p)
    _raise_stack_limit()
    rc, to, el = run_group(list(staged.launcher) + ["./" + staged.pformalsol], run,
                           os.path.join(run, "pformalsol.log"), stdin=formalsol_stdin(name, vturb, iescat),
                           timeout=timeout, grace=grace, env=env)
    probs = formal_problems(m, [os.path.basename(p) for p in outputs], nrow, kind)
    res.update(returncode=rc, timed_out=to, elapsed=el, problems=probs,
               status="ok" if (rc == 0 and not to and not probs) else "failed")
    return res
