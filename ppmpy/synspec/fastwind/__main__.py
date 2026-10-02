"""
Command line of ppmpy.synspec.fastwind (standard library only; run it with the host python, where the FASTWIND
binaries run)::

    python3 -m ppmpy.synspec.fastwind check  [--fingerprint]
    python3 -m ppmpy.synspec.fastwind stage  DEST [--mode link|copy]
    python3 -m ppmpy.synspec.fastwind one    IDX TEFF --template T --formal F --out RES
    python3 -m ppmpy.synspec.fastwind run    RUN_DIR [LIST] --template T --formal F [-K N] [-k i] [--nworkers N]
    python3 -m ppmpy.synspec.fastwind status RESULTS_DIR [--table points.txt] [--tags PATTERN] [--json]
    python3 -m ppmpy.synspec.fastwind recover DIR [--all]
    python3 -m ppmpy.synspec.fastwind rerun-formal [MODEL_DIR ...] [--list FILE] --formal F --run-root R

The FASTWIND install is given with ``--root`` / ``--build`` / ``--formal-build`` / ``--launcher`` or the
environment (PPMPY_FASTWIND_ROOT, _BUILD, _FORMAL_BUILD, _LAUNCHER). ``run`` is fw_sphere_task.sh: RUN_DIR holds
points.txt and results/; LIST (a file of 'idx teff' lines, e.g. missing.txt) replaces points.txt and gives the tag
task_<list>_%04d; the task index comes from -k, $FW_TASK, $SLURM_ARRAY_TASK_ID or $SLURM_PROCID, and K from -K or
$FW_NTASKS (required when the index comes from a Slurm step or array of several tasks; never guessed). The legacy
environment variables NW, KEEP_MODEL and PNLTE_TIMEOUT are honoured as defaults; in a Slurm job without --nworkers
or $NW the models at once default to $SLURM_CPUS_PER_TASK (legacy NW 192 = the --cpus-per-task of its sbatch
scripts), with a warning. The rows of the OUT tables (--nrow) default to ID_NFOBS of the formal build's
nlte_dim.f90 (161 for the M424 builds; 161 when the file is absent). Exit status of ``run``: 0 every point of the
share recorded, 1 errors (points neither recorded nor run, a failed final pack; run again), 3 stopped by SIGUSR1 /
SIGTERM / SIGINT / SIGHUP / SIGXCPU (resumable: submit again).

A Slurm script (one node per srun task, as fw_sphere_multinode.sbatch). The host python needs ppmpy on its path:
the project's ``sbatch`` shell function passes --export=NONE, so set PYTHONPATH in the script itself::

    #SBATCH --nodes=40 --ntasks-per-node=1 --cpus-per-task=192 --mem=0
    #SBATCH --signal=USR1@900
    export PYTHONPATH=/home/ppathak/PyPPM
    K=$SLURM_NNODES
    srun --nodes=$K --ntasks=$K --ntasks-per-node=1 --cpus-per-task=192 --cpu-bind=none --kill-on-bad-exit=0 \\
         /usr/bin/python3 -m ppmpy.synspec.fastwind run RUN_DIR -K $K --nworkers 192 \\
         --root /scratch/ppathak/FW_10.6.4.1 --build v10.6_HHe --template T --formal F

On the login node run it in tmux (or with nohup) and at most ~20 models at once; a dropped ssh session (SIGHUP)
stops it in order (exit 3), and a runner killed outright is cleaned up by its watchdog (models killed, finished
results packed).

PP 2026-10-02: new (M6); replaces fastwind_run.sh, fw_sphere_task.sh (with fw_sphere_node.sbatch /
fw_sphere_multinode.sbatch) and fw_imu_run.sh of the project stellar-atmosphere-KU-Leuven.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import archive, batch
from . import model as mo
from .install import ENV_BUILD, ENV_FORMAL_BUILD, ENV_LAUNCHER, ENV_ROOT, FastwindInstall, StagedRoot


def _env_int(name, default):
    v = os.environ.get(name, "")
    return int(v) if v != "" else default


def _install(a):
    root = a.root or os.environ.get(ENV_ROOT)
    build = a.build or os.environ.get(ENV_BUILD)
    if not root or not build:
        raise SystemExit("FASTWIND location not given: --root and --build (or {} and {})".format(ENV_ROOT, ENV_BUILD))
    fb = a.formal_build or os.environ.get(ENV_FORMAL_BUILD) or None
    launcher = a.launcher if a.launcher is not None else os.environ.get(ENV_LAUNCHER, "")
    return FastwindInstall(root, build, formal_build=fb, launcher=tuple(launcher.split()))


def _add_install(p):
    g = p.add_argument_group("FASTWIND install")
    g.add_argument("--root", help="install root with inicalc/ (or ${})".format(ENV_ROOT))
    g.add_argument("--build", help="build directory, a path or a name below the root (or ${})".format(ENV_BUILD))
    g.add_argument("--formal-build", help="build whose pformalsol is used (e.g. the intensity build; or ${})".format(
        ENV_FORMAL_BUILD))
    g.add_argument("--launcher", help="command prefix for the executables, e.g. 'taskset -c 3' (or ${})".format(
        ENV_LAUNCHER))


def _add_model(p):
    p.add_argument("--template", required=True, help="INDAT template (MODNAM and TEFF are set per point)")
    p.add_argument("--formal", required=True, help="FORMAL_INPUT (line list)")
    p.add_argument("--keep", choices=mo.KEEP_MODES, help="profiles (default; $KEEP_MODEL=1: model)")
    p.add_argument("--extras", choices=("full", "digest", "none"), default="full",
                   help="extra files per point: CONVERG + MAXTCORR.dat + convergence.json, a one-line digest, or "
                        "none (exactly the legacy files)")
    p.add_argument("--pnlte-timeout", type=float, help="s (default $PNLTE_TIMEOUT or 3600)")
    p.add_argument("--formal-timeout", type=float, default=600.0, help="s (default 600)")
    p.add_argument("--vturb", default="10 0.1", help="pformalsol's turbulence answer (default '10 0.1')")
    p.add_argument("--iescat", type=int, default=0)
    _add_nrow(p)


def _add_nrow(p):
    p.add_argument("--nrow", type=int, help="rows every OUT table must have (default ID_NFOBS of the formal "
                                            "build's nlte_dim.f90, else 161)")


def _model_kw(a):
    keep = a.keep or ("model" if os.environ.get("KEEP_MODEL", "0") == "1" else "profiles")
    return dict(keep=keep, extras=False if a.extras == "none" else a.extras,
                pnlte_timeout=a.pnlte_timeout if a.pnlte_timeout is not None else _env_int("PNLTE_TIMEOUT", 3600),
                formal_timeout=a.formal_timeout, vturb=a.vturb, iescat=a.iescat)


def _print_json(obj):
    try:
        print(json.dumps(obj, indent=1, sort_keys=True, default=str))
        sys.stdout.flush()
    except (OSError, ValueError):            # a dead terminal or pipe (e.g. after SIGHUP): the record has it all
        batch._drop_stdout()


def _nrow(a, inst):
    """--nrow, else ID_NFOBS of the formal build (nlte_dim.f90), else formal.OUT_NROW."""
    if getattr(a, "nrow", None):
        return a.nrow
    n = batch.build_nfobs(inst.formal_build) if isinstance(inst, FastwindInstall) else None
    return n or batch.OUT_NROW


def _nworkers(a, environ=None):
    """--nworkers, else $NW, else (in a Slurm job, with a warning) $SLURM_CPUS_PER_TASK, else 2."""
    env = os.environ if environ is None else environ
    if a.nworkers:
        return a.nworkers
    if env.get("NW", "") != "":
        return int(env["NW"])
    if env.get("SLURM_JOB_ID", "") != "":
        n = int(env.get("SLURM_CPUS_PER_TASK", "") or 2)
        batch._log_default("warning: --nworkers / $NW not given in Slurm job {}: {} models at once ({}; the legacy "
                           "default was NW=192)".format(env["SLURM_JOB_ID"], n, "$SLURM_CPUS_PER_TASK"
                                                        if env.get("SLURM_CPUS_PER_TASK") else "default"))
        return n
    return 2


# ----------------------------------------------------------------------------------------------------------------
def cmd_check(a):
    inst = _install(a)
    probs = inst.check()
    print("install: {}".format(inst))
    for p in probs:
        print("problem: {}".format(p))
    try:
        print("intensity patch (OUT_IMU) in pformalsol: {}".format(inst.has_imu_patch()))
    except OSError as e:
        print("cannot read pformalsol: {!r}".format(e))
    if a.fingerprint:
        _print_json(inst.fingerprint())
    print("ok" if not probs else "{} problem(s)".format(len(probs)))
    return 0 if not probs else 1


def cmd_stage(a):
    st = _install(a).stage(a.dest, mode=a.mode)
    print("staged {} (tag {}, mode {}, intensity patch {})".format(st.root, st.tag, st.mode, st.has_imu))
    probs = st.check()
    for p in probs:
        print("problem: {}".format(p))
    return 0 if not probs else 1


def cmd_one(a):
    kw = _model_kw(a)
    if a.nrow:
        kw["nrow"] = a.nrow
    tmp = None
    if a.stage:
        st = StagedRoot.open(a.stage) if os.path.isfile(os.path.join(a.stage, "FASTWIND_STAGE.txt")) else \
            _install(a).stage(a.stage, mode=a.mode)
    else:
        tmp = tempfile.mkdtemp(prefix="fwone_")
        st = _install(a).stage(tmp, mode=a.mode)
    if "nrow" not in kw:
        try:
            kw["nrow"] = _nrow(a, _install(a))
        except SystemExit:                  # an existing staged root without --root / --build: the default rows
            pass
    if a.name:
        kw["name_fmt"] = a.name                 # PP 2026-10-02: own run/result name (fastwind_run.sh NAME; no clashes)
    # PP 2026-10-02: stop handlers as for 'run': on SIGTERM/SIGHUP/SIGINT/SIGUSR1 kill the model's process group
    # (pnlte runs in its own session and would otherwise outlive this process), then exit with 128 + signal
    import signal as _signal

    def _stop(signum, frame):
        mo.stop_all(_signal.SIGTERM)
        raise SystemExit(128 + signum)
    old_handlers = {}
    for _n in ("SIGUSR1", "SIGTERM", "SIGINT", "SIGHUP"):
        if hasattr(_signal, _n):
            _s = getattr(_signal, _n)
            old_handlers[_s] = _signal.signal(_s, _stop)
    try:
        r = mo.run_model(st, (a.idx, a.teff), a.out, a.template, a.formal, keep_run=a.keep_run, **kw)
    finally:
        for _s, _h in old_handlers.items():
            _signal.signal(_s, _h)
        if tmp and not a.keep_run:
            shutil.rmtree(tmp, ignore_errors=True)
    if a.json:
        _print_json(r)
    else:
        print(r["meta"].rstrip("\n"))
        print("status {} flags {} -> {}".format(r["status"], ",".join(r["flags"]) or "-", r["result_dir"]))
    return 0 if r["status"] == "ok" else 1


def cmd_run(a):
    if a.umask is not None:
        os.umask(int(a.umask, 8))
    table = a.table or a.list or os.path.join(a.run_dir, "points.txt")
    lname = a.list_name if a.list_name is not None else (batch.list_name(a.list) if a.list else None)
    results = a.results or os.path.join(a.run_dir, "results")
    kw = _model_kw(a)
    try:
        k, K = batch.resolve_tasks(a.task, a.ntasks)
    except ValueError as e:
        raise SystemExit("error: {}".format(e))
    inst = _install(a)
    s = batch.run_models(table, inst, a.template, a.formal, results, task=k, ntasks=K, tag=a.tag,
                         list_name=lname, nworkers=_nworkers(a), local_root=a.local_root,
                         stage_mode=a.stage_mode, pack_interval=a.pack_interval, retry=a.retry,
                         retry_step=a.retry_step, max_errors=a.max_errors, keep_local=a.keep_local,
                         nrow=_nrow(a, inst), **kw)
    if a.json:
        _print_json(s)
    return s["exit_code"]


def cmd_status(a):
    tags = a.tags[0] if a.tags and len(a.tags) == 1 else a.tags
    st = batch.status(a.results_dir, tags=tags, table=a.table)
    if a.json:
        _print_json(st)
    else:
        print("\n".join(batch.format_status(st)))
    return 0


def cmd_recover(a):
    dirs = [os.path.join(a.dir, d) for d in batch._tag_dirs(a.dir, None)] if a.all else [a.dir]
    for d in dirs:
        r = archive.recover(d, min_age=a.min_age, log=batch._log_default)
        if a.json:
            _print_json(dict(r, out_dir=d))
        else:
            print("{}: {} ledgers rebuilt, {} corrupt, {} orphan ledgers, {} temporaries removed, {} kept".format(
                d, len(r["rebuilt"]), len(r["corrupt"]), len(r["orphan_ledgers"]), len(r["removed_tmp"]),
                len(r["kept_tmp"])))
    return 0


def _model_dirs(a):
    dirs = list(a.model_dir or [])
    if a.list:
        with open(a.list) as f:
            for line in f:
                tok = line.split()
                if not tok or tok[0].startswith("#"):
                    continue
                dirs.append(tok[a.column - 1] if a.column else tok[-1])
    return dirs


def cmd_rerun_formal(a):
    dirs = _model_dirs(a)
    if not dirs:
        raise SystemExit("no model directories given")
    stage = a.stage or os.path.join(a.run_root, ".fwstage")
    inst = _install(a)
    nrow = _nrow(a, inst)
    st = inst.stage(stage, mode="link")
    os.makedirs(a.run_root, exist_ok=True)
    counts = {}
    lock = threading.Lock()

    def one(src):
        try:
            r = mo.rerun_formal(st, src, a.formal, run_root=a.run_root, kind=a.kind, timeout=a.timeout,
                                overwrite=a.overwrite, vturb=a.vturb, iescat=a.iescat, nrow=nrow)
            key = r["status"]
            if key == "failed":
                batch._log_default("{}: failed (exit {}, {})".format(src, r["returncode"], r["problems"]))
        except Exception as e:
            key = "error"
            batch._log_default("{}: {!r}".format(src, e))
        with lock:
            counts[key] = counts.get(key, 0) + 1

    t0 = time.time()
    with ThreadPoolExecutor(max(1, a.nworkers)) as ex:
        list(ex.map(one, dirs))
    print("{} models in {:.1f} s: {}".format(len(dirs), time.time() - t0, ", ".join(
        "{} {}".format(k, v) for k, v in sorted(counts.items()))))
    return 0 if set(counts) <= {"ok", "skipped"} else 1


def main(argv=None):
    """The command line; returns the exit status."""
    # PP 2026-10-02: new (M6)
    ap = argparse.ArgumentParser(prog="python3 -m ppmpy.synspec.fastwind",
                                 description="Run FASTWIND (pnlte + pformalsol) for one point or a points table.")
    sub = ap.add_subparsers(dest="cmd")
    sub.required = True

    p = sub.add_parser("check", help="check the FASTWIND install (data, executables, ELF interpreter)")
    _add_install(p)
    p.add_argument("--fingerprint", action="store_true", help="also print the sha256 of the executables")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("stage", help="stage a run root (inicalc, Hopf tables, bin/) in DEST")
    _add_install(p)
    p.add_argument("dest")
    p.add_argument("--mode", choices=("link", "copy"), default="link")
    p.set_defaults(func=cmd_stage)

    p = sub.add_parser("one", help="run one model (fastwind_run.sh / fw_sphere_point.sh)")
    _add_install(p)
    p.add_argument("idx", type=int)
    p.add_argument("teff", help="T_eff [K] (written '%%.3f' into INDAT, verbatim into meta.txt)")
    p.add_argument("--out", required=True, help="result directory (the result is OUT/P<idx>)")
    p.add_argument("--stage", help="staged root to use or create (default: a temporary one)")
    p.add_argument("--mode", choices=("link", "copy"), default="link")
    p.add_argument("--keep-run", action="store_true", help="keep the run directory")
    p.add_argument("--name", help="model/run name or format (default P{idx:06d}), e.g. M424_T38230")
    p.add_argument("--json", action="store_true")
    _add_model(p)
    p.set_defaults(func=cmd_one)

    p = sub.add_parser("run", help="run one task's share of a points table (fw_sphere_task.sh)")
    _add_install(p)
    p.add_argument("run_dir", help="run directory (points.txt, results/)")
    p.add_argument("list", nargs="?", help="point list replacing points.txt (tag task_<list>_%%04d)")
    p.add_argument("-K", "--ntasks", type=int, help="number of tasks (default $FW_NTASKS or 1)")
    p.add_argument("-k", "--task", type=int, help="task index (default $FW_TASK, $SLURM_ARRAY_TASK_ID, "
                                                  "$SLURM_PROCID or 0)")
    p.add_argument("--nworkers", type=int, help="models at once (default $NW; in a Slurm job "
                                                "$SLURM_CPUS_PER_TASK, with a warning; else 2)")
    p.add_argument("--table", help="points table (overrides RUN_DIR/points.txt and LIST)")
    p.add_argument("--results", help="results directory (default RUN_DIR/results)")
    p.add_argument("--tag", help="output directory name (default task_%%04d / task_<list>_%%04d)")
    p.add_argument("--list-name", help="list part of the tag")
    p.add_argument("--local-root", help="parent of the node-local root (default /dev/shm if it has room)")
    p.add_argument("--stage-mode", choices=("copy", "link"), default="copy")
    p.add_argument("--pack-interval", type=float, default=900.0)
    p.add_argument("--retry", type=int, default=0, help="immediate retries of a failed model (default 0)")
    p.add_argument("--retry-step", type=float, default=1.0, help="K added to T_eff per retry (default 1)")
    p.add_argument("--max-errors", type=int, default=20)
    p.add_argument("--keep-local", action="store_true")
    p.add_argument("--umask", default="007", help="octal umask of the outputs (legacy 007; '' keeps the current)")
    p.add_argument("--json", action="store_true", help="print the summary as JSON")
    _add_model(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status", help="state of a run from its ledgers")
    p.add_argument("results_dir")
    p.add_argument("--table", help="points table: progress against it")
    p.add_argument("--tags", nargs="*", help="tags or one glob pattern (default all)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("recover", help="rebuild missing ledgers and drop stale temporaries of a tag directory")
    p.add_argument("dir", help="tag directory (or the results directory with --all)")
    p.add_argument("--all", action="store_true", help="every tag directory of DIR")
    p.add_argument("--min-age", type=float, default=0.0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_recover)

    p = sub.add_parser("rerun-formal", help="rerun pformalsol from saved model files (fw_imu_run.sh)")
    _add_install(p)
    p.add_argument("model_dir", nargs="*", help="directories with the model files and INDAT.DAT (P<idx>/)")
    p.add_argument("--list", help="file with a model directory per line (column --column, default the last)")
    p.add_argument("--column", type=int, help="1-based column of the model directory in --list")
    p.add_argument("--formal", required=True)
    p.add_argument("--run-root", required=True, help="where the run directories are made")
    p.add_argument("--stage", help="staged root (default RUN_ROOT/.fwstage, links)")
    p.add_argument("--kind", choices=("OUT_IMU", "OUT"), default="OUT_IMU")
    p.add_argument("--nworkers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--vturb", default="10 0.1")
    p.add_argument("--iescat", type=int, default=0)
    _add_nrow(p)
    p.set_defaults(func=cmd_rerun_formal)

    a = ap.parse_args(argv)
    if getattr(a, "umask", None) == "":
        a.umask = None
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
