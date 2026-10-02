"""Tests of ppmpy.synspec.fastwind.batch / archive / __main__ (the runner of fw_sphere_task.sh in Python).

With the fake FASTWIND (fast; ~1-2 min in total):

* stdlib only: batch, archive and __main__ import with numpy blocked, with this python and the host's /usr/bin/python3,
  which also runs batch.selftest() (a run with a retry, a CLI stop by SIGUSR1 and its resumption, a recovery);
* the split equals the legacy awk rule ``(NR - 1) % K == k`` (blank and comment lines counted), task index and tags;
  K is never guessed from Slurm (a Slurm rank in a step of several tasks without K is refused);
* a run of 40 points with 4 workers and the packer child (no packer failure, periodic packs by the child);
  fwresults.merge_task reads the parts, and its arrays equal a merge of the parts written by the LEGACY
  fw_sphere_task.sh + fw_sphere_point.sh (copies with the paths adapted) on the same fake build, K = 2 tasks, for
  points.txt with KEEP_MODEL=0 and for a point list with KEEP_MODEL=1 (tag task_<list>_%04d, model files); same
  ledger lines and same archive members and bytes;
* SIGUSR1 / SIGINT / SIGHUP (with the runner's stdout gone, as after a dropped ssh session) mid-run: exit 3,
  nothing of the interrupted models recorded, no process of the run left; the restart runs the rest; every point
  exactly once in the ledgers;
* SIGTERM with hanging models: their process groups (helper children included) are killed, a decoy process with the
  same executable name survives;
* the runner killed with SIGKILL: its watchdog kills the leftover model groups (not a decoy), packs the finished
  results and removes the local root; the restart runs the rest, every point once;
* a point whose run_model raised: exit 1, not recorded; the next run runs it; runner records have unique names;
* the packer child ignores the stop signals from its first instruction (blocked across the fork);
* a deterministic failure recorded (legacy, retry 0) and found by fwresults.combine's missing list with +1 K; the
  immediate retry (retry 1): ok at +1 K, the earlier attempt in attempts.txt, teff_nudge 1 in the merge;
* archive: an orphan archive (ledger lost) gets its ledger back byte for byte (also for a GNU tar part), corrupt
  orphans and ledgers without archive are set aside (a read error that is not corruption is raised instead), stale
  temporaries removed; a pack interrupted at any step is completed by the next one without losing or duplicating a
  point;
* the command line: check, stage, one, run, status, recover, rerun-formal; --nrow and ID_NFOBS of the build's
  nlte_dim.f90; the models-at-once default in a Slurm job.

With the real FASTWIND (markers fastwind + slow; 8 points, 4 at a time, ~6-7 min): an 8-point pilot of the M424
dump-3200 run (one capped model, point 571348 with its deterministic failure and the retry at +1 K) against the
production ledgers and profiles. PPMPY_SYNSPEC_FW_PILOT=<results dir> checks a pilot run made earlier with the
command line instead of running FASTWIND again.
"""
import glob
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time

import pytest

from conftest import ROOT
from ppmpy.synspec.fastwind import archive, batch, fake
from ppmpy.synspec.fastwind import install as ins
from ppmpy.synspec.fastwind import model as mo

# frozen copies of the original shell runner and the INDAT templates (tests/synspec/legacy/README.txt)
PROJECT = os.environ.get("PPMPY_SYNSPEC_M424_ANALYSIS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy"))
FW_ROOT = os.environ.get("PPMPY_FASTWIND_ROOT", "/scratch/ppathak/FW_10.6.4.1")
M424_RUN = os.environ.get("PPMPY_SYNSPEC_M424_RUN", "/scratch/ppathak/fastwind_sphere/d3200_r4050_N1236544")
SHADOW = os.environ.get("PPMPY_SYNSPEC_FASTWIND_SCRATCH", "/scratch/ppathak/synspec_shadow/m6/batch")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fastwind")
HOST_PYTHON = "/usr/bin/python3"
META_RE = re.compile(r"^(\d+) (\S+) (ok|formal_failed|pnlte_failed|pnlte_timeout) (\d+) (\S+) (\d+\.\d) (\d+\.\d)\n$")
PILOT = (0, 154567, 309134, 463701, 571348, 618268, 1081969, 1236536)
"""The 8-point pilot: M424 dump-3200 points; 154567 stops at the iteration cap; 571348 fails at 37298.910 K."""


def _need(path):
    if not os.path.exists(path):
        pytest.skip("not available: {}".format(path))
    return path


RUNNER_ENV = ("FW_TASK", "FW_NTASKS", "NW", "KEEP_MODEL", "PNLTE_TIMEOUT")


def _clean_environ(env):
    """``env`` without the variables that steer the runner (Slurm's, FW_*, the legacy NW / KEEP_MODEL / ...)."""
    return {k: v for k, v in env.items() if not k.startswith("SLURM_") and k not in RUNNER_ENV}


def _env():
    env = _clean_environ(os.environ)
    env["PYTHONPATH"] = ROOT + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    return env


@pytest.fixture(autouse=True)
def _no_runner_environ(monkeypatch):
    """The tests may run inside a Slurm job: its task variables must not steer run_models."""
    for k in list(os.environ):
        if k.startswith("SLURM_") or k in RUNNER_ENV:
            monkeypatch.delenv(k)


def _teffs(n, start=37000.0, step=37.5):
    return ["%.3f" % (start + step * i) for i in range(n)]


def _write_points(path, teffs):
    with open(path, "w") as f:
        f.write("".join("{} {}\n".format(i, t) for i, t in enumerate(teffs)))
    return path


def _records(out_dir):
    return [r for p in archive.ledger_paths(out_dir) for r in archive.read_index(p)]


def _quiet(msg):
    pass


def _procs_with_cwd_under(prefix):
    """pids of this user's processes whose working directory is below ``prefix``."""
    out = []
    for d in glob.glob("/proc/[0-9]*"):
        try:
            cwd = os.readlink(os.path.join(d, "cwd"))
        except OSError:
            continue
        if cwd.startswith(prefix):
            out.append(int(os.path.basename(d)))
    return out


def _newest_record(out_dir):
    recs = archive.scan(out_dir)["runner_records"]
    newest = max(recs, key=lambda n: os.stat(os.path.join(out_dir, n)).st_mtime)
    with open(os.path.join(out_dir, newest)) as f:
        return json.load(f)


def _wait_for(cond, timeout=60.0, step=0.05):
    t = time.time()
    while time.time() - t < timeout:
        if cond():
            return True
        time.sleep(step)
    return False


@pytest.fixture
def fk(tmp_path):
    """A fake install (both builds), its template and line list, points and a local root, in tmp_path."""
    inst = fake.install_fake(str(tmp_path / "fw"))
    return dict(inst=inst, tpl=os.path.join(inst.root, "INDAT.template"),
                formal=os.path.join(inst.root, "FORMAL_INPUT"), local=str(tmp_path / "local"), dir=str(tmp_path))


def _cli(fk, run_dir, *extra, table=None, results=None, nworkers=4, build=None):
    cmd = [sys.executable, "-m", "ppmpy.synspec.fastwind", "run", run_dir, "--root", fk["inst"].root,
           "--build", build or os.path.basename(fk["inst"].build), "--template", fk["tpl"], "--formal", fk["formal"],
           "--nworkers", str(nworkers), "--local-root", fk["local"]]
    if table:
        cmd += ["--table", table]
    if results:
        cmd += ["--results", results]
    return cmd + list(extra)


# ------------------------------------------------------------------------------------------------------------------
# stdlib only
# ------------------------------------------------------------------------------------------------------------------
def test_stdlib_only_this_python():
    ok, msg = fake.check_stdlib_only(modules=batch.MODULES)
    assert ok, msg


def test_stdlib_only_host_python_and_selftest(tmp_path):
    if not os.path.exists(HOST_PYTHON):
        pytest.skip("no " + HOST_PYTHON)
    ok, msg = fake.check_stdlib_only(HOST_PYTHON, modules=batch.MODULES)
    assert ok, msg
    code = ("import sys; sys.path.insert(0, {!r}); from ppmpy.synspec.fastwind import batch; "
            "batch.selftest(workdir={!r})").format(ROOT, str(tmp_path))
    p = subprocess.run([HOST_PYTHON, "-c", code], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "batch selftest passed" in p.stdout


# ------------------------------------------------------------------------------------------------------------------
# split, task index, tags, local root
# ------------------------------------------------------------------------------------------------------------------
def test_split_equals_awk(tmp_path):
    if not shutil.which("awk"):
        pytest.skip("no awk")
    rng = random.Random(3)
    lines = []
    for i in range(203):
        u = rng.random()
        if u < 0.05:
            lines.append("")
        elif u < 0.08:
            lines.append("# comment {}".format(i))
        else:
            lines.append("{} {}".format(rng.randrange(10 ** 6), "%.3f" % rng.uniform(35000, 39000)))
    for ending in ("\n", ""):                         # with and without a final newline
        p = tmp_path / "pts{}.txt".format(len(ending))
        p.write_text("\n".join(lines) + ending)
        for K in (1, 2, 3, 7, 41):
            for k in range(K):
                awk = subprocess.run(["awk", "-v", "K={}".format(K), "-v", "k={}".format(k), "(NR - 1) % K == k",
                                      str(p)], capture_output=True, text=True, check=True).stdout
                want = [tuple(ln.split()) for ln in awk.split("\n") if ln.split() and not ln.startswith("#")]
                got = batch.split_table(str(p), K, k)
                assert [(str(i), t) for _, i, t in got] == want, (K, k)
                assert all(ln % K == k for ln, _, _ in got)
    # pairs and lines in memory
    assert batch.split_table([(5, 38000.0), (6, "38000.125")], 2, 1) == [(1, 6, "38000.125")]
    assert batch.split_table(["5 38000.0\n", b"6 38000.5\n"]) == [(0, 5, "38000.0"), (1, 6, "38000.5")]
    for bad in (["5"], ["5 1 2"], ["x 38000"], ["5 -1"], ["5 nan"], ["-5 38000"], [(5,)]):
        with pytest.raises(ValueError):
            batch.split_table(bad)
    with pytest.raises(ValueError):
        batch.split_table(["1 38000"], 2, 2)


def test_task_index_count_and_tags():
    assert batch.task_index(environ={}) == 0
    assert batch.task_index(environ={"SLURM_PROCID": "3"}) == 3
    assert batch.task_index(environ={"SLURM_PROCID": "3", "SLURM_ARRAY_TASK_ID": "5"}) == 5
    assert batch.task_index(environ={"SLURM_PROCID": "3", "SLURM_ARRAY_TASK_ID": "5", "FW_TASK": "7"}) == 7
    assert batch.task_index(environ={"SLURM_PROCID": "3", "FW_TASK": ""}) == 3          # ${FW_TASK:-...}
    assert batch.task_index(2, environ={"FW_TASK": "7"}) == 2
    assert batch.task_count(environ={}) == 1 and batch.task_count(environ={"FW_NTASKS": "40"}) == 40
    assert batch.task_tag(7) == "task_0007" and batch.task_tag(0, "missing") == "task_missing_0000"
    assert batch.list_name("/x/y/missing2.txt") == "missing2" and batch.list_name("pilot.list") == "pilot.list"
    assert batch.list_name("dir/") == "dir"
    # K never guessed from Slurm: a rank of a step / array of several tasks needs K
    r = batch.resolve_tasks
    assert r(environ={}) == (0, 1) and r(environ={"SLURM_PROCID": "0", "SLURM_NTASKS": "1"}) == (0, 1)
    for env in ({"SLURM_PROCID": "1", "SLURM_NTASKS": "4"}, {"SLURM_PROCID": "0", "SLURM_NTASKS": "192"},
                {"SLURM_PROCID": "2", "SLURM_STEP_NUM_TASKS": "3", "SLURM_NTASKS": "1"},
                {"SLURM_ARRAY_TASK_ID": "3", "SLURM_ARRAY_TASK_COUNT": "2", "SLURM_PROCID": "0"}):
        with pytest.raises(ValueError, match="not guessed"):
            r(environ=env)
        k = batch.task_index(environ=env)
        assert r(ntasks=40, environ=env) == (k, 40) and r(environ=dict(env, FW_NTASKS="40")) == (k, 40)
    assert r(environ={"SLURM_PROCID": "1", "SLURM_NTASKS": "4", "FW_TASK": "2"}) == (2, 1)   # explicit k: legacy K
    assert r(1, environ={"SLURM_PROCID": "1", "SLURM_NTASKS": "4"}) == (1, 1)
    # the legacy bash rules themselves
    for k, lst in ((3, ""), (12, "/a/missing.txt")):
        sh = ('k={}; LIST={}; if [ -n "$LIST" ]; then TAG=$(printf "task_%s_%04d" "$(basename "$LIST" .txt)" "$k");'
              ' else TAG=$(printf "task_%04d" "$k"); fi; echo $TAG').format(k, lst or '""')
        want = subprocess.run(["bash", "-c", sh], capture_output=True, text=True).stdout.strip()
        assert batch.task_tag(k, batch.list_name(lst) if lst else None) == want


def test_choose_local_base(tmp_path):
    assert batch.choose_local_base(str(tmp_path), 10 ** 30)[0] == str(tmp_path)
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        assert batch.choose_local_base(None, 1)[0] == "/dev/shm"
    import tempfile
    assert batch.choose_local_base(None, 10 ** 30)[0] == tempfile.gettempdir()
    assert batch.choose_local_base(None, 1, shm=str(tmp_path / "absent"))[0] == tempfile.gettempdir()
    inst = fake.install_fake(str(tmp_path / "fw"))
    n1, n2 = batch.local_need(inst, 4), batch.local_need(inst, 4, keep="model")
    assert 4 * batch.RUN_DIR_BYTES < n1 < n2


def test_run_models_argument_checks(fk, tmp_path):
    pts = _write_points(str(tmp_path / "p.txt"), _teffs(2))
    for kw in (dict(nworkers=0), dict(retry=11), dict(retry=3, retry_step=4.0), dict(retry=1, retry_step=0),
               dict(keep="all"), dict(extras="yes"), dict(task=2, ntasks=2), dict(tag="a/b")):
        with pytest.raises(ValueError):
            batch.run_models(pts, fk["inst"], fk["tpl"], fk["formal"], str(tmp_path / "r"), local_root=fk["local"],
                             log=_quiet, **kw)
    assert not os.path.exists(fk["local"]) or not os.listdir(fk["local"])


# ------------------------------------------------------------------------------------------------------------------
# a run, and the legacy runner on the same fake build
# ------------------------------------------------------------------------------------------------------------------
def test_run_40_points_4_workers(fk, tmp_path):
    from ppmpy.synspec import fwresults
    import numpy as np
    teffs = _teffs(40)
    pts = _write_points(str(tmp_path / "points.txt"), teffs)
    fake.set_fake_config(fk["inst"].build, fail=[teffs[7]], formal_fail=[teffs[11]], sleep=0.05)
    res = str(tmp_path / "results")
    t = time.time()
    s = batch.run_models(pts, fk["inst"], fk["tpl"], fk["formal"], res, nworkers=4, local_root=fk["local"],
                         pack_interval=0.5, keep_local=True, log=_quiet)
    el = time.time() - t
    assert s["exit_code"] == 0 and not s["stopped"] and s["todo"] == 40 and s["n_errors"] == 0, s
    assert s["status"] == {"ok": 38, "pnlte_failed": 1, "formal_failed": 1} and s["packed"] == 40
    assert len(s["parts"]) >= 1 and s["tag"] == "task_0000" and el < 120
    # the packer CHILD packed (a broken child would fall back to packing in this process): no failure, periodic packs
    # by the child (its log and reports in the kept local root), the watchdog ran and was told DONE
    assert s["pack_failures"] == 0 and s["periodic_packs"] >= 1 and s["watchdog"] and s["local_kept"], s
    plog = open(os.path.join(s["local_root"], "packer.log")).read()
    assert "packed " in plog and "Traceback" not in plog, plog
    reps = [json.load(open(f)) for f in glob.glob(os.path.join(s["local_root"], "pack_*.json"))]
    assert len(reps) >= 2 and all(r["ok"] for r in reps)                   # periodic ones and the final one
    assert sum(r["npoint"] for r in reps) == 40 and sorted(r["part"] for r in reps if r["part"]) == sorted(s["parts"])
    assert "Traceback" not in open(os.path.join(s["local_root"], batch.WATCHDOG_LOG)).read()
    shutil.rmtree(s["local_root"])
    out = os.path.join(res, "task_0000")
    recs = _records(out)
    assert sorted(r.idx for r in recs) == list(range(40))
    assert all(r.teff == teffs[r.idx] for r in recs)
    st = archive.scan(out)
    assert st["orphan_parts"] == st["orphan_ledgers"] == st["tmp"] == [] and len(st["runner_records"]) == 1
    assert sorted(st["parts"]) == sorted(s["parts"])
    for p in st["parts"]:
        info = archive.parse_part_name(p)
        assert info["host"] == archive.short_host() and info["pid"] == os.getpid()
    # the record is there
    rec = json.load(open(s["record"]))
    assert rec["exit_code"] == 0 and rec["status"] == s["status"]
    # merge_task reads the parts; status() agrees
    points = dict(idx=np.arange(40), teff=np.array([float(t) for t in teffs]))
    m = fwresults.merge_task(res, "task_0000", str(tmp_path / "m.npz"), points, copy_keys=())
    assert m["npoint"] == 40 and m["status"] == {"ok": 38, "pnlte_failed": 1, "formal_failed": 1}
    z = np.load(str(tmp_path / "m.npz"))
    assert np.isfinite(z["fnorm"][z["status"] == "ok"]).all() and (z["teff_nudge"] == 0).all()
    sts = batch.status(res, table=pts)
    assert sts["table"] == dict(points=40, done=40, todo=0, ok=38, failed=2, not_in_table=0)
    assert sts["tags"]["task_0000"]["duplicates"] == 0 and sts["tags"]["task_0000"]["last_runner"]["exit_code"] == 0
    assert "table: 40 points, 40 done" in "\n".join(batch.format_status(sts))
    # a second and third call (within a second) find everything done: no local root, no new part; their runner
    # records do not replace each other
    for _ in range(2):
        s2 = batch.run_models(pts, fk["inst"], fk["tpl"], fk["formal"], res, nworkers=4, local_root=fk["local"],
                              log=_quiet)
        assert s2["exit_code"] == 0 and s2["todo"] == 0 and s2["done_before"] == 40 and s2["parts"] == []
    assert sorted(archive.scan(out)["parts"]) == sorted(s["parts"])
    recs = archive.scan(out)["runner_records"]
    assert len(recs) == 3 and os.path.basename(s2["record"]) == recs[-1], recs
    # every INDAT.DAT is the template with the point's MODNAM and TEFF (fwresults' premise check)
    prem = fwresults.check_indat_premise(res, template=fk["tpl"])
    assert prem["passed"] and prem["n_points"] == 40, prem


def _legacy_task_copies(fk, tmp_path):
    """fw_sphere_task.sh and fw_sphere_point.sh copied, with ANA / FWTOP / /dev/shm pointing into tmp_path."""
    task = _need(os.path.join(PROJECT, "fw_sphere_task.sh"))
    point = _need(os.path.join(PROJECT, "fw_sphere_point.sh"))
    for tool in ("bash", "awk", "timeout", "tar", "xargs"):
        if not shutil.which(tool):
            pytest.skip("no " + tool)
    ana = tmp_path / "ana"
    (ana / "fastwind").mkdir(parents=True)
    shutil.copyfile(point, str(ana / "fw_sphere_point.sh"))
    os.chmod(str(ana / "fw_sphere_point.sh"), 0o755)
    shutil.copyfile(fk["tpl"], str(ana / "fastwind" / "INDAT_M424test.DAT"))
    shutil.copyfile(fk["formal"], str(ana / "fastwind" / "FORMAL_INPUT_He3"))
    text = open(task).read()
    subs = {"ANA=/home/ppathak/stellar-atmosphere-KU-Leuven/project/analysis\n": "ANA={}\n".format(ana),
            "FWTOP=/scratch/ppathak/FW_10.6.4.1\n": "FWTOP={}\n".format(fk["inst"].root),
            "LOCAL=/dev/shm/fwsphere_": "LOCAL={}/shm/fwsphere_".format(tmp_path)}
    for a, b in subs.items():
        assert text.count(a) == 1, a
        text = text.replace(a, b)
    (tmp_path / "shm").mkdir()
    p = tmp_path / "fw_sphere_task.sh"
    p.write_text(text)
    return str(p)


def _members(part):
    """{member name: bytes or None (directory)} of an archive, without './'."""
    out = {}
    with tarfile.open(part) as tf:
        for m in tf:
            n = m.name
            while n.startswith("./"):
                n = n[2:]
            out[n.rstrip("/")] = tf.extractfile(m).read() if m.isfile() else None
    return out


@pytest.mark.parametrize("variant", ["table", "list_keep_model"])
def test_equals_legacy_task_script(fk, tmp_path, variant):
    """
    The same points (ok, deterministic failure, formal failure, crash) run by the legacy fw_sphere_task.sh +
    fw_sphere_point.sh and by run_models (extras=False), K = 2: same ledger lines (columns 1-5), same archive members
    and bytes (meta.txt columns 1-5), and fwresults.merge_task gives the same arrays (the run times aside).
    'table': 40 points of points.txt, KEEP_MODEL=0; 'list_keep_model': 16 points of a point list (tag
    task_pilot_%04d), KEEP_MODEL=1 / keep='model' (the model files byte for byte).
    """
    from ppmpy.synspec import fwresults
    import numpy as np
    script = _legacy_task_copies(fk, tmp_path)
    use_list = variant == "list_keep_model"
    n = 16 if use_list else 40
    teffs = _teffs(n)
    fake.set_fake_config(fk["inst"].build, fail=[teffs[3]], formal_fail=[teffs[8]], crash=[teffs[13]])
    leg, new = tmp_path / "legacy", tmp_path / "new"
    leg.mkdir()
    new.mkdir()
    name = "pilot.txt" if use_list else "points.txt"
    pts = _write_points(str(leg / name), teffs)
    shutil.copyfile(pts, str(new / name))
    keep_model = "1" if use_list else "0"
    K = 2
    t = time.time()
    for k in range(K):
        env = dict(_clean_environ(os.environ), FW_TASK=str(k), NW="4", KEEP_MODEL=keep_model, PNLTE_TIMEOUT="60")
        logp = tmp_path / "legacy_{}.log".format(k)
        args = ["bash", script, str(leg), str(K)] + ([str(leg / name)] if use_list else [])
        with open(str(logp), "wb") as lf:                # not a pipe: the legacy packer leaves a 'sleep 900' behind
            p = subprocess.Popen(args, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = p.wait(timeout=300)
            finally:
                try:
                    os.killpg(p.pid, signal.SIGKILL)     # that sleep (the session of the script started here)
                except ProcessLookupError:
                    pass
        text = logp.read_text()
        assert rc == 0 and "finished: " in text, text
    t_legacy = time.time() - t
    t = time.time()
    for k in range(K):
        s = batch.run_models(str(new / name), fk["inst"], fk["tpl"], fk["formal"], str(new / "results"),
                             task=k, ntasks=K, nworkers=4, extras=False, local_root=fk["local"],
                             list_name=batch.list_name(name) if use_list else None,
                             keep="model" if use_list else "profiles", log=_quiet)
        assert s["exit_code"] == 0 and s["tag"] == batch.task_tag(k, "pilot" if use_list else None)
        assert s["pack_failures"] == 0
    t_new = time.time() - t
    assert not os.listdir(str(tmp_path / "shm")) and not os.listdir(fk["local"])
    points = dict(idx=np.arange(n), teff=np.array([float(x) for x in teffs]))
    for k in range(K):
        tag = batch.task_tag(k, "pilot" if use_list else None)
        assert os.path.isdir(str(leg / "results" / tag))
        la, lb = _records(str(leg / "results" / tag)), _records(str(new / "results" / tag))
        assert sorted(r[:5] for r in la) == sorted(r[:5] for r in lb)
        assert sorted(r.idx for r in lb) == list(range(k, n, K))
        ma, mb = {}, {}
        for d, m in ((leg, ma), (new, mb)):
            for part in glob.glob(str(d / "results" / tag / "part_*.tar.gz")):
                m.update(_members(part))
        assert sorted(ma) == sorted(mb)
        if use_list:                                       # the model files are there (and compared below)
            ok_pt = ["P%06d" % i for i in range(k, n, K) if i not in (3, 8, 13)][0]
            assert all("{}/{}".format(ok_pt, f) in mb for f in mo.MODEL_FILES)
        for m in ma:
            if m.endswith("meta.txt"):
                assert ma[m].split()[:5] == mb[m].split()[:5], m
            else:
                assert ma[m] == mb[m], m
        za = fwresults.merge_task(str(leg / "results"), tag, str(tmp_path / "a.npz"), points, copy_keys=())
        zb = fwresults.merge_task(str(new / "results"), tag, str(tmp_path / "b.npz"), points, copy_keys=())
        assert za["status"] == zb["status"] and za["npoint"] == zb["npoint"] == n // K
        a, b = np.load(str(tmp_path / "a.npz")), np.load(str(tmp_path / "b.npz"))
        assert sorted(a.files) == sorted(b.files)
        for key in a.files:
            if key in ("t_pnlte", "t_formal"):
                continue
            assert a[key].dtype == b[key].dtype, key
            assert np.array_equal(a[key], b[key], equal_nan=a[key].dtype.kind == "f"), key
    st = {r.idx: r.status for k in range(K)
          for r in _records(str(new / "results" / batch.task_tag(k, "pilot" if use_list else None)))}
    assert (st[3], st[8], st[13]) == ("pnlte_failed", "formal_failed", "pnlte_failed")
    print("legacy {:.1f} s, run_models {:.1f} s".format(t_legacy, t_new))


# ------------------------------------------------------------------------------------------------------------------
# stop and restart
# ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("sig,dead_stdout", [(signal.SIGUSR1, False), (signal.SIGINT, False), (signal.SIGHUP, True)])
def test_stop_exit3_and_restart(fk, tmp_path, sig, dead_stdout):
    """dead_stdout: the runner's stdout and stderr are gone before the signal (a dropped ssh session: SIGHUP, then
    writes to a hung-up terminal fail); the stop is still orderly, exit 3 (not Python's 120 of a failed flush)."""
    teffs = _teffs(40)
    pts = _write_points(str(tmp_path / "points.txt"), teffs)
    fake.set_fake_config(fk["inst"].build, sleep=0.3)
    res = str(tmp_path / "results")
    out = os.path.join(res, "task_0000")
    cmd = _cli(fk, str(tmp_path), "--pack-interval", "0.5", "--json", table=pts, results=res)
    p = subprocess.Popen(cmd, env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert _wait_for(lambda: archive.ledger_paths(out) or p.poll() is not None)
    assert p.poll() is None, p.communicate()[0].decode()
    assert _wait_for(lambda: _procs_with_cwd_under(fk["local"]), timeout=10)      # models are running
    if dead_stdout:
        p.stdout.close()
    p.send_signal(sig)
    t = time.time()
    if dead_stdout:
        p.wait(timeout=120)
        o = ""
        s = _newest_record(out)
    else:
        o = p.communicate(timeout=120)[0].decode()
        assert "{} received".format(signal.Signals(sig).name) in o
        s = json.loads(o[o.index("\n{") + 1:])
    assert p.returncode == batch.EXIT_STOPPED, o
    assert time.time() - t < 20
    assert s["stopped"] and s["signal"] == signal.Signals(sig).name and s["exit_code"] == 3
    assert s["pack_failures"] == 0
    assert s["interrupted"] >= 1 and s["not_started"] >= 1
    assert _procs_with_cwd_under(fk["local"]) == []                           # nothing of the run left
    assert os.listdir(fk["local"]) == []
    done1 = _records(out)
    n1 = len(done1)
    assert 0 < n1 < 40 and len({r.idx for r in done1}) == n1 and n1 == s["packed"]
    assert all(r.status == "ok" for r in done1)
    st = archive.scan(out)
    assert st["tmp"] == [] and st["orphan_parts"] == []
    # restart: the rest, each point once
    p2 = subprocess.run(cmd, env=_env(), capture_output=True, text=True, timeout=300)
    assert p2.returncode == 0, p2.stdout + p2.stderr
    s2 = json.loads(p2.stdout[p2.stdout.index("\n{") + 1:])
    assert s2["done_before"] == n1 and s2["todo"] == 40 - n1 and s2["status"] == {"ok": 40 - n1}
    assert s2["pack_failures"] == 0
    recs = _records(out)
    assert sorted(r.idx for r in recs) == list(range(40))
    assert all(r.teff == teffs[r.idx] for r in recs)
    assert len(archive.scan(out)["runner_records"]) == 2


def test_sigterm_kills_own_groups_not_decoy(fk, tmp_path):
    """Hanging models (pnlte with a helper child) are killed by the stop; a decoy with the same executable name
    (started outside the runner) survives; the hung points run after the restart."""
    teffs = _teffs(8)
    pts = _write_points(str(tmp_path / "points.txt"), teffs)
    fake.set_fake_config(fk["inst"].build, hang=[teffs[2], teffs[5]])
    decoy_dir = tmp_path / "decoy"
    decoy_dir.mkdir()
    decoy = subprocess.Popen([fk["inst"].pnlte], cwd=str(decoy_dir), env=dict(os.environ, **{fake.ENV_MODE: "hang"}),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        assert _wait_for(lambda: (decoy_dir / "hang_child.pid").exists())
        decoy_child = int((decoy_dir / "hang_child.pid").read_text())
        res = str(tmp_path / "results")
        cmd = _cli(fk, str(tmp_path), "--pnlte-timeout", "300", "--json", table=pts, results=res)
        p = subprocess.Popen(cmd, env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        pidfiles = lambda: glob.glob(os.path.join(fk["local"], "fwsphere_*", "fw", "P*", "hang_child.pid"))
        assert _wait_for(lambda: len(pidfiles()) == 2 or p.poll() is not None, timeout=60)
        assert p.poll() is None
        finished = lambda: glob.glob(os.path.join(fk["local"], "fwsphere_*", "res", "P*"))
        assert _wait_for(lambda: len(finished()) == 6, timeout=60)      # the six other points are finished
        helpers = [int(open(f).read()) for f in pidfiles()]
        assert all(fake._alive(h) for h in helpers)
        p.send_signal(signal.SIGTERM)
        o = p.communicate(timeout=60)[0].decode()
        assert p.returncode == 3, o
        time.sleep(0.2)
        assert not any(fake._alive(h) for h in helpers)
        assert decoy.poll() is None and fake._alive(decoy_child)
        out = os.path.join(res, "task_0000")
        assert sorted(r.idx for r in _records(out)) == [0, 1, 3, 4, 6, 7]
        fake.set_fake_config(fk["inst"].build)
        p2 = subprocess.run(cmd, env=_env(), capture_output=True, text=True, timeout=120)
        assert p2.returncode == 0, p2.stdout + p2.stderr
        assert sorted(r.idx for r in _records(out)) == list(range(8))
        assert decoy.poll() is None
    finally:
        os.killpg(decoy.pid, signal.SIGKILL)
        decoy.wait()


def test_stop_in_process_and_handlers_restored(fk, tmp_path):
    """run_models in this process stopped by SIGUSR1 from a timer thread: exit 3, handlers and model stop flag
    restored, so a following run works."""
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signal handlers need the main thread")
    teffs = _teffs(24)
    pts = _write_points(str(tmp_path / "points.txt"), teffs)
    fake.set_fake_config(fk["inst"].build, sleep=0.3)
    before = {s: signal.getsignal(getattr(signal, s)) for s in batch.STOP_SIGNALS}
    timer = threading.Timer(1.0, os.kill, args=(os.getpid(), signal.SIGUSR1))
    timer.start()
    res = str(tmp_path / "results")
    s = batch.run_models(pts, fk["inst"], fk["tpl"], fk["formal"], res, nworkers=4, local_root=fk["local"],
                         pack_child=False, log=_quiet)
    timer.join()
    assert s["exit_code"] == 3 and s["signal"] == "SIGUSR1" and s["interrupted"] >= 1
    assert {s_: signal.getsignal(getattr(signal, s_)) for s_ in batch.STOP_SIGNALS} == before
    assert not mo.stop_requested() and mo.active_groups() == {}
    fake.set_fake_config(fk["inst"].build)
    s2 = batch.run_models(pts, fk["inst"], fk["tpl"], fk["formal"], res, nworkers=4, local_root=fk["local"],
                          pack_child=False, log=_quiet)
    assert s2["exit_code"] == 0 and s2["todo"] == 24 - s["packed"]
    assert sorted(r.idx for r in _records(os.path.join(res, "task_0000"))) == list(range(24))


def test_watchdog_after_sigkill(fk, tmp_path):
    """The runner killed outright (SIGKILL: no handler runs): its watchdog kills the models still running in its run
    directories (not a decoy elsewhere with the same executable), packs the finished but unpacked results, writes a
    record and removes the local root; the restart runs the rest, every point exactly once."""
    teffs = _teffs(12)
    pts = _write_points(str(tmp_path / "points.txt"), teffs)
    fake.set_fake_config(fk["inst"].build, sleep=1.0)
    decoy_dir = tmp_path / "decoy"
    decoy_dir.mkdir()
    decoy = subprocess.Popen([fk["inst"].pnlte], cwd=str(decoy_dir), env=dict(os.environ, **{fake.ENV_MODE: "hang"}),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        res = str(tmp_path / "results")
        out = os.path.join(res, "task_0000")
        cmd = _cli(fk, str(tmp_path), "--pack-interval", "3600", table=pts, results=res)
        p = subprocess.Popen(cmd, env=_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finished = lambda: glob.glob(os.path.join(fk["local"], "fwsphere_*", "res", "P*"))
        assert _wait_for(lambda: len(finished()) >= 2 or p.poll() is not None, timeout=60)
        assert p.poll() is None
        assert _wait_for(lambda: _procs_with_cwd_under(fk["local"]), timeout=10)
        p.kill()
        p.wait()
        assert _wait_for(lambda: not os.listdir(fk["local"]), timeout=30)       # cleaned up by the watchdog
        assert _procs_with_cwd_under(fk["local"]) == []
        assert decoy.poll() is None
        rec = _newest_record(out)
        assert rec["watchdog"] and rec["runner_pid"] == p.pid and rec["pack_ok"] and rec["killed_groups"] >= 1
        recs = _records(out)
        assert len(recs) == rec["packed"] >= 2 and len({r.idx for r in recs}) == len(recs)
        assert all(r.status == "ok" for r in recs) and archive.scan(out)["tmp"] == []
        fake.set_fake_config(fk["inst"].build)
        p2 = subprocess.run(cmd + ["--json"], env=_env(), capture_output=True, text=True, timeout=120)
        assert p2.returncode == 0, p2.stdout + p2.stderr
        assert sorted(r.idx for r in _records(out)) == list(range(12))
        st = batch.status(res)["tags"]["task_0000"]
        assert st["duplicates"] == 0 and st["last_runner"]["exit_code"] == 0
    finally:
        os.killpg(decoy.pid, signal.SIGKILL)
        decoy.wait()


def test_exit_1_for_points_that_raised(fk, tmp_path, monkeypatch):
    """A point whose run_model raised (fewer than max_errors) is neither recorded nor done: exit 1, not 0 (an
    afterok merge job must not start); the next run runs it."""
    teffs = _teffs(6)
    pts = _write_points(str(tmp_path / "points.txt"), teffs)
    real = mo.run_model

    def flaky(staged, spec, *a, **k):
        if spec[0] == 3:
            raise OSError(28, "No space left on device")
        return real(staged, spec, *a, **k)

    monkeypatch.setattr(mo, "run_model", flaky)
    res = str(tmp_path / "results")
    s = batch.run_models(pts, fk["inst"], fk["tpl"], fk["formal"], res, nworkers=2, local_root=fk["local"],
                         log=_quiet)
    assert s["exit_code"] == batch.EXIT_ERROR and s["n_errors"] == 1 and s["status"] == {"ok": 5}
    assert not s["aborted"] and not s["stopped"]
    out = os.path.join(res, "task_0000")
    assert sorted(r.idx for r in _records(out)) == [0, 1, 2, 4, 5]
    monkeypatch.setattr(mo, "run_model", real)
    s2 = batch.run_models(pts, fk["inst"], fk["tpl"], fk["formal"], res, nworkers=2, local_root=fk["local"],
                          log=_quiet)
    assert s2["exit_code"] == 0 and s2["todo"] == 1 and s2["status"] == {"ok": 1}
    assert sorted(r.idx for r in _records(out)) == list(range(6))


def test_packer_child_ignores_signals_from_the_start(tmp_path):
    """The stop signals sent right after the fork (during the child's start-up, before it can install SIG_IGN) do not
    kill the packer child: they are blocked across the fork and discarded."""
    res, stage, out = str(tmp_path / "res"), str(tmp_path / "stage"), str(tmp_path / "out")
    for trial in range(3):
        _fake_result(res, trial)
        cmd = [sys.executable, "-m", "ppmpy.synspec.fastwind.archive", "pack", res, stage, out, "--ignore-signals"]
        p = batch._popen_signals_blocked(cmd, env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         start_new_session=True)
        for name in batch.CHILD_IGNORED:
            os.kill(p.pid, getattr(signal, name))
        o = p.communicate(timeout=60)[0].decode()
        assert p.returncode == 0, (p.returncode, o)
    assert sorted(r.idx for r in _records(out)) == [0, 1, 2]
    # the runner's own mask is restored
    if hasattr(signal, "pthread_sigmask"):
        blocked = signal.pthread_sigmask(signal.SIG_BLOCK, [])
        assert not any(getattr(signal, n) in blocked for n in batch.CHILD_IGNORED)


# ------------------------------------------------------------------------------------------------------------------
# failures, retries, missing list
# ------------------------------------------------------------------------------------------------------------------
def test_deterministic_failure_recorded_and_retry(fk, tmp_path):
    from ppmpy.synspec import fwresults
    import numpy as np
    teffs = _teffs(12)
    pts = _write_points(str(tmp_path / "points.txt"), teffs)
    fake.set_fake_config(fk["inst"].build, fail=[teffs[4]], formal_partial=[teffs[6]], crash=[teffs[9]])
    points = dict(idx=np.arange(12), teff=np.array([float(t) for t in teffs]))
    # legacy behaviour (retry 0): failures recorded, missing.txt lists them at T_eff + 1 K
    res = str(tmp_path / "r0")
    s = batch.run_models(pts, fk["inst"], fk["tpl"], fk["formal"], res, nworkers=3, local_root=fk["local"],
                         log=_quiet)
    assert s["status"] == {"ok": 9, "pnlte_failed": 2, "formal_failed": 1} and s["retried"] == 0
    recs = {r.idx: r for r in _records(os.path.join(res, "task_0000"))}
    assert (recs[4].status, recs[4].niter) == ("pnlte_failed", 19) and recs[6].status == "formal_failed"
    fwresults.merge_task(res, "task_0000", str(tmp_path / "m0.npz"), points, copy_keys=())
    fwresults.combine([str(tmp_path / "m0.npz")], pts, str(tmp_path / "c0"))
    miss = open(str(tmp_path / "c0" / "missing.txt")).read().split("\n")
    assert miss[:3] == ["4 %.3f" % (float(teffs[4]) + 1), "6 %.3f" % (float(teffs[6]) + 1),
                        "9 %.3f" % (float(teffs[9]) + 1)]
    # the missing list run as a point list (tag task_missing_0000): every failure converges at +1 K
    s = batch.run_models(str(tmp_path / "c0" / "missing.txt"), fk["inst"], fk["tpl"], fk["formal"], res,
                         list_name="missing", nworkers=3, local_root=fk["local"], log=_quiet)
    assert s["tag"] == "task_missing_0000" and s["status"] == {"ok": 3}
    # immediate retry: the same outcome in one run
    res1 = str(tmp_path / "r1")
    s = batch.run_models(pts, fk["inst"], fk["tpl"], fk["formal"], res1, nworkers=3, local_root=fk["local"],
                         retry=1, log=_quiet)
    assert s["status"] == {"ok": 12} and s["retried"] == 3 and s["retry_ok"] == 3
    recs = {r.idx: r for r in _records(os.path.join(res1, "task_0000"))}
    for i in (4, 6, 9):
        assert recs[i].teff == "%.3f" % (float(teffs[i]) + 1) and recs[i].status == "ok"
    for i in set(range(12)) - {4, 6, 9}:
        assert recs[i].teff == teffs[i]
    with tarfile.open(glob.glob(os.path.join(res1, "task_0000", "part_*.tar.gz"))[0]) as tf:
        att = tf.extractfile("./P000004/attempts.txt").read().decode()
        ind = tf.extractfile("./P000004/INDAT.DAT").read().decode()
    assert META_RE.match(att) and att.split()[:4] == ["4", teffs[4], "pnlte_failed", "19"]
    assert ind.split("\n")[3].startswith("%.3f," % (float(teffs[4]) + 1))
    m = fwresults.merge_task(res1, "task_0000", str(tmp_path / "m1.npz"), points, copy_keys=())
    z = np.load(str(tmp_path / "m1.npz"))
    assert m["nnudge"] == 3 and sorted(np.nonzero(z["teff_nudge"])[0]) == [4, 6, 9]
    assert np.all(z["teff_nudge"][[4, 6, 9]] == 1.0)
    assert fwresults.check_indat_premise(res1, template=fk["tpl"])["passed"]
    # a failure that persists: recorded once, at the last attempt's T_eff, both attempts in attempts.txt
    fake.set_fake_config(fk["inst"].build, fail=[teffs[2], "%.3f" % (float(teffs[2]) + 2),
                                                 "%.3f" % (float(teffs[2]) + 4)])
    res2 = str(tmp_path / "r2")
    s = batch.run_models([(2, teffs[2])], fk["inst"], fk["tpl"], fk["formal"], res2, nworkers=1, retry=2,
                         retry_step=2.0, local_root=fk["local"], log=_quiet)
    assert s["status"] == {"pnlte_failed": 1} and s["retried"] == 1 and s["retry_ok"] == 0
    (r,) = _records(os.path.join(res2, "task_0000"))
    assert r.teff == "%.3f" % (float(teffs[2]) + 4) and r.status == "pnlte_failed"


# ------------------------------------------------------------------------------------------------------------------
# archive: pack, ledgers, recovery
# ------------------------------------------------------------------------------------------------------------------
def _fake_result(res, idx, teff="38000.000", status="ok", extra=b""):
    d = os.path.join(res, "P%06d" % idx)
    os.makedirs(d)
    with open(os.path.join(d, "meta.txt"), "w") as f:
        f.write(mo.format_meta(idx, teff, status, 33, "39000.5", 1.0, 0.5))
    with open(os.path.join(d, "OUT.HEI4026_VTV010"), "wb") as f:
        f.write(b"x" * 100 + extra)
    return d


def test_pack_layout_and_ledger(tmp_path):
    res, stage, out = str(tmp_path / "res"), str(tmp_path / "stage"), str(tmp_path / "out")
    for i in (5, 2, 9):
        _fake_result(res, i, extra=str(i).encode())
    os.makedirs(os.path.join(res, "P000011.tmp"))             # being written
    os.makedirs(os.path.join(res, ".P000012.old123"))          # being replaced
    r = archive.pack(res, stage, out, part="part_test_1")
    assert r["npoint"] == 3 and r["points"] == ["P000002", "P000005", "P000009"]
    assert sorted(os.listdir(res)) == [".P000012.old123", "P000011.tmp"]
    assert os.listdir(stage) == []
    with tarfile.open(r["archive"]) as tf:
        names = [m.name for m in tf]
        assert all(m.isdir() for m in tf if m.name.count("/") < 2)
    assert names == [".", "./P000002", "./P000002/OUT.HEI4026_VTV010", "./P000002/meta.txt", "./P000005",
                     "./P000005/OUT.HEI4026_VTV010", "./P000005/meta.txt", "./P000009", "./P000009/OUT.HEI4026_VTV010",
                     "./P000009/meta.txt"]                    # tarfile strips the '/' of directories when reading
    gnu = subprocess.run(["tar", "-tzf", r["archive"]], capture_output=True, text=True, check=True).stdout.split()
    assert gnu[:2] == ["./", "./P000002/"]                    # as GNU tar -C STAGE . (fw_sphere_task.sh)
    led = open(r["ledger"], "rb").read()
    (tmp_path / "x").mkdir()
    cat = subprocess.run(["bash", "-c", "cd {} && tar -xzf {} && cat ./*/meta.txt".format(tmp_path / "x",
                                                                                          r["archive"])],
                         capture_output=True).stdout
    assert led == cat == archive.ledger_bytes_from_part(r["archive"])
    assert [x.idx for x in archive.read_index(r["ledger"])] == [2, 5, 9]
    assert archive.done_set(out) == {2, 5, 9}
    assert archive.pack(res, stage, out)["part"] is None                # nothing finished
    with pytest.raises(ValueError):
        archive.parse_meta_line("1 2 3")


def test_pack_interrupted_is_completed(tmp_path, monkeypatch):
    """A pack killed after the archive (no ledger) or during the archive is completed by the next pack: every point
    once in the ledgers, no temporaries."""
    res, stage, out = str(tmp_path / "res"), str(tmp_path / "stage"), str(tmp_path / "out")
    for i in range(4):
        _fake_result(res, i)
    real = archive._write_ledger
    monkeypatch.setattr(archive, "_write_ledger", lambda *a, **k: (_ for _ in ()).throw(OSError("killed")))
    with pytest.raises(OSError):
        archive.pack(res, stage, out, part="part_a")
    monkeypatch.setattr(archive, "_write_ledger", real)
    assert os.path.exists(os.path.join(out, "part_a.tar.gz")) and not os.path.exists(os.path.join(out, "part_a.idx"))
    assert os.path.exists(os.path.join(stage, archive.MARKER))
    for i in range(4, 6):
        _fake_result(res, i)
    r = archive.pack(res, stage, out, part="part_b")
    assert r["resumed"]["action"] == "completed" and r["points"] == ["P000004", "P000005"]
    assert sorted(x.idx for x in _records(out)) == list(range(6))
    # killed while writing the archive: a partial .tmp and the marker are left; the stage is packed again
    for i in range(6, 9):
        _fake_result(res, i)
    archive._write_marker(stage, "part_c")
    for i in range(6, 8):
        os.rename(os.path.join(res, "P%06d" % i), os.path.join(stage, "P%06d" % i))
    open(os.path.join(out, "part_c.tar.gz.tmp"), "wb").write(b"\x1f\x8b partial")
    r = archive.pack(res, stage, out, part="part_d")
    assert r["resumed"]["action"] == "discarded" and r["points"] == ["P000006", "P000007", "P000008"]
    assert sorted(x.idx for x in _records(out)) == list(range(9))
    st = archive.scan(out)
    assert st["tmp"] == [] and st["orphan_parts"] == [] and sorted(st["parts"]) == ["part_a", "part_b", "part_d"]
    # an exception inside the archive writer: temporary removed, stage kept for the next pack
    _fake_result(res, 9)
    monkeypatch.setattr(archive, "_write_tar", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        archive.pack(res, stage, out, part="part_e")
    monkeypatch.undo()
    assert archive.scan(out)["tmp"] == [] and os.listdir(stage) == ["P000009"]
    assert archive.pack(res, stage, out)["points"] == ["P000009"]
    assert sorted(x.idx for x in _records(out)) == list(range(10))


def test_recover_orphans_and_tmp(tmp_path):
    res, stage, out = str(tmp_path / "res"), str(tmp_path / "stage"), str(tmp_path / "out")
    parts = []
    for grp in ((1, 2), (3, 4, 5), (6,)):
        for i in grp:
            _fake_result(res, i)
        parts.append(archive.pack(res, stage, out)["part"])
    a, b, c = parts
    led_a = open(os.path.join(out, a + ".idx"), "rb").read()
    os.unlink(os.path.join(out, a + ".idx"))                                 # orphan archive
    # a legacy part (GNU tar -C STAGE .) without its ledger
    leg = tmp_path / "legstage"
    for i in (7, 8):
        _fake_result(str(leg), i)
    subprocess.run(["tar", "-czf", os.path.join(out, "part_20260925_121512_2637.tar.gz"), "-C", str(leg), "."],
                   check=True)
    led_leg = subprocess.run(["bash", "-c", "cat {}/*/meta.txt".format(leg)], capture_output=True).stdout
    # a truncated orphan, a ledger without archive, temporaries
    data = open(os.path.join(out, b + ".tar.gz"), "rb").read()
    open(os.path.join(out, "part_20260101_000000Z_h_1_0001.tar.gz"), "wb").write(data[:len(data) // 2])
    open(os.path.join(out, "part_20260101_000000Z_h_1_0002.idx"), "w").write("99 1.0 ok 1 nan 0.0 0.0\n")
    old = time.time() - 7200
    for n in ("part_x.tar.gz.tmp", "part_y.idx.tmp"):
        open(os.path.join(out, n), "w").write("partial")
        os.utime(os.path.join(out, n), (old, old))
    mine = "part_20260101_000000Z_{}_{}_0007.tar.gz.tmp".format(archive.short_host(), os.getpid())
    open(os.path.join(out, mine), "w").write("being written")
    os.utime(os.path.join(out, mine), (old, old))
    st = batch.status(str(tmp_path), tags=["out"])["tags"]["out"]
    assert len(st["orphan_parts"]) == 3 and len(st["orphan_ledgers"]) == 1 and len(st["tmp"]) == 3
    assert 99 in archive.done_set(out)
    r = archive.recover(out)
    assert sorted(r["rebuilt"]) == sorted([a, "part_20260925_121512_2637"])
    assert r["corrupt"] == ["part_20260101_000000Z_h_1_0001"]
    assert r["orphan_ledgers"] == ["part_20260101_000000Z_h_1_0002.idx"]
    assert sorted(r["removed_tmp"]) == ["part_x.tar.gz.tmp", "part_y.idx.tmp"] and r["kept_tmp"] == [mine]
    assert open(os.path.join(out, a + ".idx"), "rb").read() == led_a
    assert open(os.path.join(out, "part_20260925_121512_2637.idx"), "rb").read() == led_leg
    assert archive.done_set(out) == set(range(1, 9))
    st = archive.scan(out)
    assert st["orphan_parts"] == st["orphan_ledgers"] == [] and st["tmp"] == [mine]
    assert st["corrupt"] and st["orphaned"]
    assert archive.recover(out) == dict(rebuilt=[], corrupt=[], orphan_ledgers=[], removed_tmp=[], kept_tmp=[mine])
    # a read error that is not corruption (EACCES, EIO on scratch) is raised: the valid orphan is not set aside
    os.unlink(os.path.join(out, b + ".idx"))
    real = archive.ledger_bytes_from_part
    for exc in (PermissionError(13, "Permission denied"), OSError(5, "Input/output error"), MemoryError()):
        archive.ledger_bytes_from_part = lambda part, e=exc: (_ for _ in ()).throw(e)
        try:
            with pytest.raises(type(exc)):
                archive.recover(out)
        finally:
            archive.ledger_bytes_from_part = real
        assert os.path.exists(os.path.join(out, b + ".tar.gz")) and archive.scan(out)["orphan_parts"] == [b]
    assert archive.recover(out)["rebuilt"] == [b]
    # the runner recovers its tag directory at the start (here: an archive whose ledger was lost)
    os.unlink(os.path.join(out, c + ".idx"))
    assert 6 not in archive.done_set(out)
    p = subprocess.run([sys.executable, "-m", "ppmpy.synspec.fastwind", "recover", out], env=_env(),
                       capture_output=True, text=True)
    assert p.returncode == 0 and "1 ledgers rebuilt" in p.stdout, p.stdout + p.stderr
    assert 6 in archive.done_set(out)


def test_runner_recovers_lost_ledger_before_split(fk, tmp_path):
    teffs = _teffs(6)
    pts = _write_points(str(tmp_path / "points.txt"), teffs)
    res = str(tmp_path / "results")
    s = batch.run_models(pts, fk["inst"], fk["tpl"], fk["formal"], res, nworkers=2, local_root=fk["local"],
                         log=_quiet)
    out = os.path.join(res, "task_0000")
    (led,) = archive.ledger_paths(out)
    os.unlink(led)
    s2 = batch.run_models(pts, fk["inst"], fk["tpl"], fk["formal"], res, nworkers=2, local_root=fk["local"],
                          log=_quiet)
    assert s2["recovered"]["rebuilt"] == s["parts"] and s2["todo"] == 0 and s2["done_before"] == 6


# ------------------------------------------------------------------------------------------------------------------
# command line
# ------------------------------------------------------------------------------------------------------------------
def _main(*args):
    return subprocess.run([sys.executable, "-m", "ppmpy.synspec.fastwind"] + list(args), env=_env(),
                          capture_output=True, text=True, timeout=300)


def test_cli_subcommands(fk, tmp_path):
    inst = fk["inst"]
    common = ["--root", inst.root, "--build", "v10.6_HHe"]
    p = _main("check", *common, "--fingerprint")
    assert p.returncode == 0 and "intensity patch (OUT_IMU) in pformalsol: False" in p.stdout, p.stdout + p.stderr
    p = _main("check", "--root", str(tmp_path / "nowhere"), "--build", "v10.6_HHe")
    assert p.returncode != 0
    p = _main("stage", *common, "--formal-build", "v10.6_HHe_imu", str(tmp_path / "st"), "--mode", "copy")
    assert p.returncode == 0 and "intensity patch True" in p.stdout, p.stdout + p.stderr
    p = _main("one", *common, "7", "38123.4567", "--template", fk["tpl"], "--formal", fk["formal"],
              "--out", str(tmp_path / "one"), "--keep", "model", "--extras", "none")
    assert p.returncode == 0, p.stdout + p.stderr
    assert sorted(os.listdir(str(tmp_path / "one" / "P000007"))) == sorted(
        ["INDAT.DAT", "meta.txt"] + list(mo.MODEL_FILES) + ["OUT.{}_VTV010".format(n)
                                                           for n in ("HEI4026", "HEII4200", "HEI4922")])
    assert open(str(tmp_path / "one" / "P000007" / "meta.txt")).read().split()[:3] == ["7", "38123.4567", "ok"]
    fake.set_fake_config(inst.build, fail=[38000.0])
    p = _main("one", *common, "8", "38000", "--template", fk["tpl"], "--formal", fk["formal"],
              "--out", str(tmp_path / "one"), "--json", "--stage", str(tmp_path / "st"))
    assert p.returncode == 1 and json.loads(p.stdout)["status"] == "pnlte_failed"
    fake.set_fake_config(inst.build)
    # run (a point list -> tag task_<list>_0000; legacy environment defaults), status, rerun-formal
    rd = tmp_path / "run"
    rd.mkdir()
    lst = _write_points(str(tmp_path / "pilot.txt"), _teffs(5))
    env = dict(_env(), NW="2", KEEP_MODEL="1", PNLTE_TIMEOUT="100")
    p = subprocess.run([sys.executable, "-m", "ppmpy.synspec.fastwind", "run", str(rd), lst, *common, "--template",
                        fk["tpl"], "--formal", fk["formal"], "--local-root", fk["local"], "--json"], env=env,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    s = json.loads(p.stdout[p.stdout.index("\n{") + 1:])
    assert s["tag"] == "task_pilot_0000" and s["nworkers"] == 2 and s["status"] == {"ok": 5}
    out = rd / "results" / "task_pilot_0000"
    part = glob.glob(str(out / "part_*.tar.gz"))[0]
    assert oct(os.stat(part).st_mode & 0o777) == oct(0o660)                 # umask 007, as the legacy runner
    assert "./P000003/MODEL" in [m.name for m in tarfile.open(part)]        # KEEP_MODEL=1
    p = _main("status", str(rd / "results"), "--table", lst)
    assert p.returncode == 0 and "table: 5 points, 5 done (5 ok, 0 failed), 0 to run" in p.stdout, p.stdout
    st = json.loads(_main("status", str(rd / "results"), "--json").stdout)
    assert st["total"]["points"] == 5
    ex = tmp_path / "extracted"
    ex.mkdir()
    subprocess.run(["tar", "-xzf", part, "-C", str(ex)], check=True)
    reps = tmp_path / "reps.txt"
    reps.write_text("".join("0 {} x {}\n".format(i, ex / ("P%06d" % i)) for i in range(5)))
    p = _main("rerun-formal", *common, "--formal-build", "v10.6_HHe_imu", "--list", str(reps), "--formal",
              fk["formal"], "--run-root", str(tmp_path / "imu"), "--nworkers", "3")
    assert p.returncode == 0 and "5 models" in p.stdout and "ok 5" in p.stdout, p.stdout + p.stderr
    assert len(glob.glob(str(tmp_path / "imu" / "P*" / "P*" / "OUT_IMU.*"))) == 15
    p = _main("rerun-formal", *common, "--formal-build", "v10.6_HHe_imu", str(ex / "P000001"), "--formal",
              fk["formal"], "--run-root", str(tmp_path / "imu"))
    assert p.returncode == 0 and "skipped 1" in p.stdout
    p = _main("recover", str(rd / "results"), "--all", "--json")
    assert p.returncode == 0 and json.loads(p.stdout)["rebuilt"] == []


def test_cli_nrow_and_slurm_defaults(fk, tmp_path):
    """--nrow, else ID_NFOBS of the formal build's nlte_dim.f90 (a build with more rows than pformalsol writes records
    formal_failed); in a Slurm job without --nworkers / $NW the models at once are $SLURM_CPUS_PER_TASK, with a
    warning; a Slurm rank of a step of several tasks without -K is refused before anything runs."""
    inst = fk["inst"]
    common = ["--root", inst.root, "--build", "v10.6_HHe", "--template", fk["tpl"], "--formal", fk["formal"]]
    assert batch.build_nfobs(inst.build) is None
    dim = os.path.join(inst.build, "nlte_dim.f90")
    with open(dim, "w") as f:
        f.write("MODULE nlte_dim\n INTEGER(I4B), PARAMETER :: ID_NFOBS = 170\nEND MODULE\n")
    assert batch.build_nfobs(inst.build) == 170
    p = _main("one", "1", "38000", *common, "--out", str(tmp_path / "one"), "--json")
    r = json.loads(p.stdout)
    assert p.returncode == 1 and r["status"] == "formal_failed", p.stdout + p.stderr
    assert "161 table rows < 170" in json.dumps(r["formal_problems"])
    p = _main("one", "2", "38000", *common, "--out", str(tmp_path / "one"), "--nrow", "161")
    assert p.returncode == 0, p.stdout + p.stderr
    if os.path.exists(FW_ROOT):
        assert batch.build_nfobs(os.path.join(FW_ROOT, "v10.6_HHe")) == batch.OUT_NROW
    os.unlink(dim)
    rd = tmp_path / "run"
    rd.mkdir()
    _write_points(str(rd / "points.txt"), _teffs(3))
    base = ["run", str(rd)] + common + ["--local-root", fk["local"], "--json"]
    env = dict(_env(), SLURM_JOB_ID="4242", SLURM_CPUS_PER_TASK="3")
    p = subprocess.run([sys.executable, "-m", "ppmpy.synspec.fastwind"] + base, env=env, capture_output=True,
                       text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    s = json.loads(p.stdout[p.stdout.index("\n{") + 1:])
    assert s["nworkers"] == 3 and "warning: --nworkers / $NW not given in Slurm job 4242" in p.stdout
    env = dict(_env(), SLURM_JOB_ID="4242", SLURM_PROCID="1", SLURM_NTASKS="4", NW="2")
    p = subprocess.run([sys.executable, "-m", "ppmpy.synspec.fastwind"] + base, env=env, capture_output=True,
                       text=True, timeout=120)
    assert p.returncode == 1 and "K is not guessed from Slurm" in p.stderr, p.stdout + p.stderr
    assert not os.path.exists(str(rd / "results" / "task_0001"))
    p = subprocess.run([sys.executable, "-m", "ppmpy.synspec.fastwind"] + base + ["-K", "4"], env=env,
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    s = json.loads(p.stdout[p.stdout.index("\n{") + 1:])
    assert s["tag"] == "task_0001" and s["ntasks"] == 4 and s["assigned"] == 1


# ------------------------------------------------------------------------------------------------------------------
# the real FASTWIND: an 8-point pilot of the M424 run
# ------------------------------------------------------------------------------------------------------------------
def _production_records(idx):
    led = sorted(glob.glob(os.path.join(M424_RUN, "results", "*", "part_*.idx")))
    want = {str(i) for i in idx}
    out = {}
    for p in led:
        with open(p) as f:
            for ln in f:
                tok = ln.split(None, 1)
                if tok and tok[0] in want:
                    out.setdefault(int(tok[0]), []).append(ln.split())
    return out


@pytest.fixture(scope="module")
def pilot():
    given = os.environ.get("PPMPY_SYNSPEC_FW_PILOT")
    if given:
        return dict(results=_need(given), summary=None, seconds=None)
    if not (os.path.isdir(FW_ROOT) and os.path.isdir("/cvmfs")):
        pytest.skip("FASTWIND install or /cvmfs not available")
    inst = ins.FastwindInstall(FW_ROOT, "v10.6_HHe")
    probs = inst.check()
    if probs:
        pytest.skip("FASTWIND not runnable here: " + "; ".join(probs))
    pts = _need(os.path.join(M424_RUN, "points.txt"))
    rows = {}
    with open(pts) as f:
        for ln in f:
            tok = ln.split()
            if int(tok[0]) in PILOT:
                rows[int(tok[0])] = tok[1]
    work = os.path.join(SHADOW, "pilot8_pytest_{}".format(os.getpid()))
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    table = os.path.join(work, "points.txt")
    with open(table, "w") as f:
        f.write("".join("{} {}\n".format(i, rows[i]) for i in PILOT))
    t = time.time()
    s = batch.run_models(table, inst, os.path.join(DATA, "INDAT_M424test.DAT"), os.path.join(DATA, "FORMAL_INPUT_He3"),
                         os.path.join(work, "results"), nworkers=4, retry=1, pnlte_timeout=1800,
                         local_root=os.path.join(work, "local"), log=print)
    return dict(results=os.path.join(work, "results"), summary=s, seconds=time.time() - t)


@pytest.mark.fastwind
@pytest.mark.slow
def test_real_pilot_8_points(pilot):
    """The 8 points as in the production run: ledger columns 1-5 (idx, T_eff, status, niter, T(tau=2/3)) equal; 571348
    fails at 37298.910 K and is recorded ok at 37299.910 K with the failed attempt in attempts.txt (the production
    run needed a missing-list round for it); the profiles equal those of profiles.npz."""
    from ppmpy.synspec import fwresults
    import numpy as np
    s = pilot["summary"]
    if s is not None:
        assert s["exit_code"] == 0 and s["status"] == {"ok": 8} and s["retried"] == 1 and s["retry_ok"] == 1, s
        print("pilot: {:.0f} s".format(pilot["seconds"]))
    out = os.path.join(pilot["results"], "task_0000")
    recs = {r.idx: r for r in _records(out)}
    assert sorted(recs) == sorted(PILOT) and all(r.status == "ok" for r in recs.values())
    prod = _production_records(PILOT)
    if not prod:
        pytest.skip("production ledgers not available")
    for i, r in recs.items():
        ok = [p for p in prod[i] if p[2] == "ok"]
        assert ok and [str(x) for x in r[:5]] == ok[0][:5], (r, ok)
    assert recs[154567].niter == 102                                    # stopped at the cap, as in production
    files = {}
    for part in glob.glob(os.path.join(out, "part_*.tar.gz")):
        files.update({k: v for k, v in _members(part).items() if k.startswith("P571348/")})
    p571 = sorted(files)
    att = files["P571348/attempts.txt"].decode()
    conv = json.loads(files["P571348/convergence.json"])
    fail = [p for p in prod[571348] if p[2] != "ok"][0]
    assert att.split()[:5] == fail[:5]
    assert conv["status"] == "ok" and conv["convergence"]["converged"]
    assert "P571348/pnlte_tail.log" not in p571 and "P571348/CONVERG" in p571
    pts = os.path.join(M424_RUN, "points.npz")
    prof = os.path.join(M424_RUN, "profiles.npz")
    if not (os.path.exists(pts) and os.path.exists(prof)):
        pytest.skip("production points.npz / profiles.npz not available")
    m = os.path.join(os.path.dirname(pilot["results"]), "merged_pilot.npz")
    info = fwresults.merge_task(pilot["results"], "task_0000", m, pts)
    assert info["npoint"] == 8 and info["nnudge"] == 1
    z = np.load(m)
    store = fwresults.ProfileStore.open(prof)
    rows = np.searchsorted(np.asarray(store.idx), z["idx"])
    assert np.array_equal(np.asarray(store.idx)[rows], z["idx"])
    assert np.array_equal(np.asarray(store["teff_nudge"])[rows], z["teff_nudge"])
    for key in ("lam", "fcont", "fnorm"):
        a, b = np.asarray(store[key][rows]), z[key]
        assert np.array_equal(a, b), (key, np.nanmax(np.abs(a - b)))
