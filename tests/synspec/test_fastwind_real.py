"""
M6 end-to-end checks of ppmpy.synspec.fastwind: (a) the project scripts migrated to thin callers of the runner
(stellar-atmosphere-KU-Leuven/project/analysis: fastwind_run.sh, fw_sphere_task.sh, fw_sphere_point.sh,
fw_sphere_node.sbatch, fw_sphere_multinode.sbatch, fw_imu_run.sh), driven with the fake FASTWIND (no batch job is
submitted: the sbatch scripts run under bash with a Slurm-like environment, srun replaced by a shim); (b) the real
FASTWIND against the legacy products, byte for byte (reference model M424_T38230, point 571348, an 8-point pilot
through the CLI with a SIGUSR1 stop and a resume, OUT_IMU reruns).

Real tests are marked ``fastwind`` (skipped without the install or /cvmfs; in the container bind /cvmfs:
``apptainer exec --bind /home,/cvmfs SIF python -m pytest tests/synspec/test_fastwind_real.py``) and ``slow``. At most
4 real models run at once. The pilot (~12 min with 4 workers on the login node) reuses a finished pilot results
directory when PPMPY_SYNSPEC_FW_REAL_PILOT names one (the shadow run of 2026-10-02:
/scratch/ppathak/synspec_shadow/m6/real/pilot/results).

PP 2026-10-02: new (M6).
"""
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

import pytest

from ppmpy.synspec.fastwind import fake
from ppmpy.synspec.fastwind import install as ins
from ppmpy.synspec.fastwind import model as mo

PPMPY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT = os.environ.get("PPMPY_SYNSPEC_PROJECT_ANALYSIS",
                         "/home/ppathak/stellar-atmosphere-KU-Leuven/project/analysis")
FW_ROOT = os.environ.get("PPMPY_FASTWIND_ROOT", "/scratch/ppathak/FW_10.6.4.1")
FW_RUNS = os.environ.get("PPMPY_SYNSPEC_M424_FWRUNS", "/scratch/ppathak/fastwind_runs")
M424_RUN = os.environ.get("PPMPY_SYNSPEC_M424_RUN", "/scratch/ppathak/fastwind_sphere/d3200_r4050_N1236544")
IMU = os.environ.get("PPMPY_SYNSPEC_M424_IMU", "/scratch/ppathak/fastwind_imu")
SHADOW = os.environ.get("PPMPY_SYNSPEC_FASTWIND_SCRATCH_REAL", "/scratch/ppathak/synspec_shadow/m6/real")
LINES = ("HEI4026", "HEII4200", "HEI4922")
OUTS = tuple("OUT.{}_VTV010".format(n) for n in LINES)
LEGACY_OK = {"INDAT.DAT", "meta.txt"} | set(OUTS)             # fw_sphere_point.sh, KEEP_MODEL=0, status ok
PILOT = (0, 40, 80, 120, 160, 200, 520, 680)                  # production task_0000, one part; 80 120 520 680 capped
CAPPED = (80, 120, 520, 680)
IMU_REPS = ("P009729", "P001652")


def _need(path):
    if not os.path.exists(path):
        pytest.skip("not available: {}".format(path))
    return path


def _script(name):
    return _need(os.path.join(PROJECT, name))


def _env(**kw):
    """A clean environment for the scripts (no inherited Slurm, NW or exported shell functions such as a site 'srun'
    wrapper); the runner in this python."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "FW_", "PPMPY_FAKE", "BASH_FUNC_")) and
           k not in ("NW", "KEEP_MODEL", "PNLTE_TIMEOUT", "PYTHONPATH")}
    env.update(FW_PYTHON=sys.executable, PPMPY=PPMPY)
    env.update({k: str(v) for k, v in kw.items()})
    return env


def _run(cmd, env, timeout=600, **kw):
    return subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                          timeout=timeout, **kw)


def _tar_members(part):
    import tarfile
    out = {}
    with tarfile.open(part) as t:
        for m in t.getmembers():
            if m.isfile():
                out[m.name.lstrip("./")] = t.extractfile(m).read()
    return out


def _ledger(out_dir):
    recs = []
    for p in sorted(glob.glob(os.path.join(out_dir, "part_*.idx"))):
        with open(p) as f:
            recs += [ln.split() for ln in f if ln.strip()]
    return recs


def _points(path, n, t0=38000.0, dt=7.25):
    with open(path, "w") as f:
        f.write("".join("{} {:.3f}\n".format(i, t0 + i * dt) for i in range(n)))
    return path


@pytest.fixture()
def fakefw(tmp_path):
    root = str(tmp_path / "fakefw")
    fake.install_fake(root, python=sys.executable, sleep=0.05)
    return root


def _set(root, **cfg):
    for b in (fake.FAKE_BUILD, fake.FAKE_IMU_BUILD):
        fake.set_fake_config(os.path.join(root, b), **cfg)


# ------------------------------------------------------------------------------------------------------------------
# (a) the migrated project scripts with the fake FASTWIND
# ------------------------------------------------------------------------------------------------------------------
SUMMARY = re.compile(r"^(\S+): pnlte ([0-9.]+) s, pformalsol ([0-9.]+) s, (\d+) line profiles in (\S+), status (\S+), "
                     r"converged (\S+)$")


def test_fastwind_run_sh_fake(fakefw, tmp_path):
    """fastwind_run.sh: same CLI, run directory kept as $FW_ROOT/NAME, old paths via the MODNAM link, summary line
    parsable as fastwind_scaling.sbatch does (field 3 = pnlte seconds), the '_warmup /dev/null' call, failures."""
    sh = _script("fastwind_run.sh")
    indat = os.path.join(PROJECT, "fastwind", "INDAT_M424test.DAT")
    formal = os.path.join(PROJECT, "fastwind", "FORMAL_INPUT_He3")
    root = str(tmp_path / "runs")
    env = _env(FW_ROOT=root, FW_BUILD=os.path.join(fakefw, fake.FAKE_BUILD))
    r = _run(["bash", sh, "_warmup", "/dev/null", "/dev/null"], env)
    assert r.returncode == 1 and "staged only" in r.stdout, r.stdout
    assert os.path.isfile(os.path.join(root, ins.STAGE_MANIFEST)) and os.path.isdir(os.path.join(root, "bin"))
    procs = [subprocess.Popen(["bash", sh, "w{:03d}".format(i), indat, formal], env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, universal_newlines=True) for i in range(1, 5)]
    outs = [p.communicate(timeout=120)[0] for p in procs]
    assert [p.returncode for p in procs] == [0] * 4, outs
    for i, o in enumerate(outs, 1):
        name = "w{:03d}".format(i)
        m = SUMMARY.match(o.strip())
        assert m and m.group(1) == name and m.group(4) == "3" and m.group(6) == "ok" and m.group(7) == "True", o
        assert o.split()[2] == m.group(2)                  # fastwind_scaling.sbatch: awk '{print $3}'
        run = os.path.join(root, name)
        assert os.path.islink(os.path.join(run, "M424test"))      # INDAT catalogue name -> the model directory
        for f in OUTS:
            assert os.path.isfile(os.path.join(run, "M424test", f)) and os.path.isfile(os.path.join(run, name, f))
        assert os.path.isfile(os.path.join(run, "pnlte.log"))
        res = os.path.join(root, "results", name)
        assert open(os.path.join(res, "meta.txt")).read().split()[2] == "ok"
        assert set(mo.MODEL_FILES) <= set(os.listdir(res))
        with open(os.path.join(run, "INDAT.DAT")) as f:
            assert f.readline().split()[0] == name
    _set(fakefw, fail=["38230.000"], sleep=0.05)
    r = _run(["bash", sh, "wfail", indat, formal], env)
    assert r.returncode == 1 and r.stdout.startswith("wfail: pnlte FAILED (") and "error in ne" in r.stdout, r.stdout
    assert open(os.path.join(root, "results", "wfail", "meta.txt")).read().split()[2] == "pnlte_failed"
    r = _run(["bash", sh, "only2"], env)
    assert r.returncode == 2 and "usage" in r.stdout


def test_fastwind_run_sh_legacy_root(fakefw, tmp_path):
    """An old FW_ROOT whose inicalc and Hopf entries link into the install: staged in place, the install untouched."""
    sh = _script("fastwind_run.sh")
    root = tmp_path / "old"
    root.mkdir()
    ini = os.path.join(fakefw, "inicalc")
    os.symlink(ini, str(root / "inicalc"))
    for h in ins.HOPF_FILES:
        os.symlink(os.path.join(ini, h), str(root / h))
    before = sorted(os.listdir(ini))
    r = _run(["bash", sh, "M424test", os.path.join(PROJECT, "fastwind", "INDAT_M424test.DAT"),
              os.path.join(PROJECT, "fastwind", "FORMAL_INPUT_He3")],
             _env(FW_ROOT=str(root), FW_BUILD=os.path.join(fakefw, fake.FAKE_BUILD)))
    assert r.returncode == 0, r.stdout
    assert sorted(os.listdir(ini)) == before and os.path.islink(str(root / "inicalc"))
    assert all(os.path.isfile(str(root / "M424test" / "M424test" / f)) for f in OUTS)


def _node(fakefw, run_dir, K, k, extra=(), **env):
    e = _env(SLURM_JOB_ID="dry{}".format(os.getpid()), SLURM_ARRAY_TASK_ID=k, SLURM_PROCID=0,
             SLURM_CPUS_PER_TASK=192, NW=env.pop("NW", 3), FW_TOP=fakefw, **env)
    return _run(["bash", _script("fw_sphere_node.sbatch"), run_dir, str(K)] + list(extra), e)


def test_fw_sphere_node_sbatch_fake(fakefw, tmp_path):
    """Array task k of K: lines k, k+K, ...; the legacy member set with FW_EXTRAS=none; lists; resubmission skips."""
    run = str(tmp_path / "run")
    os.makedirs(run)
    _points(os.path.join(run, "points.txt"), 10)
    _set(fakefw, fail=["38021.750"], sleep=0.02)                 # point 3
    r = _node(fakefw, run, 2, 1, FW_EXTRAS="none")
    assert r.returncode == 0, r.stdout
    out = os.path.join(run, "results", "task_0001")
    recs = _ledger(out)
    assert [int(x[0]) for x in recs] == [1, 3, 5, 7, 9] or sorted(int(x[0]) for x in recs) == [1, 3, 5, 7, 9]
    st = {int(x[0]): x[2] for x in recs}
    assert st[3] == "pnlte_failed" and all(st[i] == "ok" for i in (1, 5, 7, 9))
    files = {}
    for p in glob.glob(os.path.join(out, "part_*.tar.gz")):
        files.update(_tar_members(p))
    for i in (1, 5, 7, 9):
        assert {k.split("/", 1)[1] for k in files if k.startswith("P{:06d}/".format(i))} == LEGACY_OK
    assert {k.split("/", 1)[1] for k in files if k.startswith("P000003/")} == {"INDAT.DAT", "meta.txt",
                                                                               "pnlte_tail.log"}
    # resubmission: nothing to do
    r = _node(fakefw, run, 2, 1)
    assert r.returncode == 0 and "0 to run" in r.stdout, r.stdout
    # a list (the retry of point 3 at +1 K), KEEP_MODEL=1, extras full
    with open(os.path.join(run, "missing.txt"), "w") as f:
        f.write("3 38022.750\n")
    _set(fakefw, sleep=0.02)
    r = _node(fakefw, run, 1, 0, extra=[os.path.join(run, "missing.txt")], KEEP_MODEL=1)
    assert r.returncode == 0, r.stdout
    out = os.path.join(run, "results", "task_missing_0000")
    assert _ledger(out)[0][:3] == ["3", "38022.750", "ok"]
    mem = {}
    for p in glob.glob(os.path.join(out, "part_*.tar.gz")):
        mem.update(_tar_members(p))
    names = {k.split("/", 1)[1] for k in mem}
    assert LEGACY_OK | set(mo.MODEL_FILES) | {"CONVERG", "MAXTCORR.dat", "convergence.json"} == names


def test_fw_sphere_multinode_sbatch_fake(fakefw, tmp_path):
    """One srun task per node (shim): ranks 0..K-1 split points.txt as K array tasks; srun line as before."""
    run = str(tmp_path / "run")
    os.makedirs(run)
    _points(os.path.join(run, "points.txt"), 9)
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "srun").write_text(
        "#!/bin/bash\necho \"srun $*\" > \"$DRY_DIR/srun_cmd.txt\"\nn=1\n"
        "while [ $# -gt 0 ]; do case \"$1\" in --ntasks=*) n=${1#--ntasks=}; shift;; --*) shift;; *) break;; esac; done\n"
        "pids=()\nfor ((t = 0; t < n; t++)); do SLURM_PROCID=$t \"$@\" > \"$DRY_DIR/task$t.out\" 2>&1 & pids+=($!); done\n"
        "rc=0; for p in \"${pids[@]}\"; do wait $p || rc=$?; done; exit $rc\n")
    os.chmod(str(shim / "srun"), 0o755)
    env = _env(SLURM_JOB_ID="drym", SLURM_NNODES=3, SLURM_JOB_NODELIST="n[1-3]", SLURM_CPUS_PER_TASK=192, NW=2,
               FW_TOP=fakefw, DRY_DIR=str(tmp_path), PATH="{}:{}".format(shim, os.environ.get("PATH", "/usr/bin:/bin")))
    r = _run(["bash", _script("fw_sphere_multinode.sbatch"), run], env)
    assert r.returncode == 0, r.stdout
    cmd = (tmp_path / "srun_cmd.txt").read_text()
    assert "--ntasks=3 --ntasks-per-node=1 --cpus-per-task=192 --cpu-bind=none --kill-on-bad-exit=0" in cmd
    assert cmd.rstrip().endswith("fw_sphere_task.sh {} 3".format(run))
    for k in range(3):
        recs = _ledger(os.path.join(run, "results", "task_{:04d}".format(k)))
        assert sorted(int(x[0]) for x in recs) == list(range(k, 9, 3)) and all(x[2] == "ok" for x in recs)
    text = open(_script("fw_sphere_multinode.sbatch")).read()
    assert "#SBATCH --signal=USR1@900" in text
    assert "#SBATCH --signal=B:USR1@900" in open(_script("fw_sphere_node.sbatch")).read()


def test_fw_sphere_task_sigusr1_and_resume_fake(fakefw, tmp_path):
    """SIGUSR1 to the batch shell's process (the execs make it the runner): exit 3, finished points packed, no
    leftovers; the resubmission finishes the share (exit 0)."""
    run = str(tmp_path / "run")
    os.makedirs(run)
    _points(os.path.join(run, "points.txt"), 8)
    _set(fakefw, sleep=1.5)
    env = _env(SLURM_JOB_ID="drys", SLURM_ARRAY_TASK_ID=0, SLURM_PROCID=0, SLURM_CPUS_PER_TASK=192, NW=2,
               FW_TOP=fakefw)
    p = subprocess.Popen(["bash", _script("fw_sphere_node.sbatch"), run, "1"], env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, universal_newlines=True)
    time.sleep(4.0)
    p.send_signal(signal.SIGUSR1)
    out = p.communicate(timeout=120)[0]
    assert p.returncode == 3 and "SIGUSR1" in out, out
    done = _ledger(os.path.join(run, "results", "task_0000"))
    assert 0 < len(done) < 8
    assert not glob.glob("/dev/shm/fwsphere_drys_0_*")
    _set(fakefw, sleep=0.02)
    r = _run(["bash", _script("fw_sphere_node.sbatch"), run, "1"], env)
    assert r.returncode == 0, r.stdout
    recs = _ledger(os.path.join(run, "results", "task_0000"))
    assert sorted(int(x[0]) for x in recs) == list(range(8)) and all(x[2] == "ok" for x in recs)


def test_fw_sphere_point_sh_fake(fakefw, tmp_path):
    """The kept single-point caller with a pre-runner node-local root (FWL with copies) and RES."""
    fwl = tmp_path / "fwl"
    shutil.copytree(os.path.join(fakefw, "inicalc"), str(fwl / "inicalc"), symlinks=True,
                    ignore=shutil.ignore_patterns("inicalc", "HOPFPARA_ALL_*"))
    for h in ins.HOPF_FILES:
        shutil.copy(os.path.join(fakefw, "inicalc", h), str(fwl / h))
    os.makedirs(str(fwl / "bin"))
    for f in os.listdir(os.path.join(fakefw, fake.FAKE_BUILD)):
        if f != fake.CONFIG_NAME:
            shutil.copy2(os.path.join(fakefw, fake.FAKE_BUILD, f), str(fwl / "bin" / f))
    shutil.copy(os.path.join(PROJECT, "fastwind", "INDAT_M424test.DAT"), str(fwl / "INDAT.template"))
    shutil.copy(os.path.join(PROJECT, "fastwind", "FORMAL_INPUT_He3"), str(fwl / "FORMAL_INPUT"))
    res = str(tmp_path / "res")
    r = _run(["bash", _script("fw_sphere_point.sh"), "3", "38100.125"],
             _env(FWL=str(fwl), RES=res, FW_TOP=fakefw, KEEP_MODEL=1))
    assert r.returncode == 0, r.stdout
    meta = open(os.path.join(res, "P000003", "meta.txt")).read().split()
    assert meta[:3] == ["3", "38100.125", "ok"]
    assert set(mo.MODEL_FILES) <= set(os.listdir(os.path.join(res, "P000003")))
    assert not os.path.islink(str(fwl / "bin" / "pnlte_A10HHe.eo"))       # copies kept (copy staging)


def test_fw_imu_run_sh_fake(fakefw, tmp_path):
    """fw_imu_run.sh: 'bin idx teff model_dir' lines -> RUNS/P<idx>/P<idx>/OUT_IMU.*; a second call skips."""
    st = ins.FastwindInstall(fakefw, fake.FAKE_BUILD).stage(str(tmp_path / "st"))
    tpl = os.path.join(PROJECT, "fastwind", "INDAT_M424test.DAT")
    formal = os.path.join(PROJECT, "fastwind", "FORMAL_INPUT_He3")
    lines = []
    for i, t in ((5, "38001.000"), (9, "38002.000")):
        r = mo.run_model(st, (i, t), str(tmp_path / "raw"), tpl, formal, keep="model")
        assert r["status"] == "ok"
        lines.append("{} {} {} {}\n".format(i, i, t, r["result_dir"]))
    reps = tmp_path / "reps.txt"
    reps.write_text("".join(lines))
    runs = str(tmp_path / "runs")
    r = _run(["bash", _script("fw_imu_run.sh"), str(reps), runs, "2"], _env(FW_TOP=fakefw))
    assert r.returncode == 0 and "2 of 2 models have OUT_IMU files" in r.stdout and "ok 2" in r.stdout, r.stdout
    for name in ("P000005", "P000009"):
        for n in LINES:
            assert mo.out_problem(os.path.join(runs, name, name, "OUT_IMU.{}_VTV010".format(n)), "OUT_IMU") is None
    r = _run(["bash", _script("fw_imu_run.sh"), str(reps), runs, "2"], _env(FW_TOP=fakefw))
    assert r.returncode == 0 and "skipped 2" in r.stdout, r.stdout


# ------------------------------------------------------------------------------------------------------------------
# (b) the real FASTWIND, byte for byte against the legacy products
# ------------------------------------------------------------------------------------------------------------------
def _real():
    if not (os.path.isdir(FW_ROOT) and os.path.isdir("/cvmfs")):
        pytest.skip("FASTWIND install or /cvmfs not available")
    probs = ins.FastwindInstall(FW_ROOT, "v10.6_HHe").check()
    if probs:
        pytest.skip("FASTWIND not runnable here: " + "; ".join(probs))


def _work(name):
    w = os.path.join(SHADOW, "pytest_{}_{}".format(name, os.getpid()))
    shutil.rmtree(w, ignore_errors=True)
    os.makedirs(w)
    return w


def _same(a, b):
    with open(a, "rb") as f, open(b, "rb") as g:
        return f.read() == g.read()


@pytest.mark.fastwind
@pytest.mark.slow
def test_real_fastwind_run_sh_reference():
    """fastwind_run.sh on the reference INDAT: OUT.*, CONVERG, MAXTCORR.dat and the model files byte-identical to
    /scratch/ppathak/fastwind_runs/M424_T38230 (same login-node class of CPU); converged at 59 CONVERG iterations."""
    _real()
    ref = _need(os.path.join(FW_RUNS, "M424_T38230"))
    w = _work("ref")
    t = time.time()
    r = _run(["bash", _script("fastwind_run.sh"), "M424_T38230", os.path.join(ref, "INDAT.DAT"),
              os.path.join(ref, "FORMAL_INPUT")],
             _env(FW_ROOT=w, FW_BUILD=os.path.join(FW_ROOT, "v10.6_HHe")), timeout=1800)
    print("reference model: {:.0f} s".format(time.time() - t))
    assert r.returncode == 0 and r.stdout.rstrip().endswith("status ok, converged True"), r.stdout
    new = os.path.join(w, "M424_T38230", "M424_T38230")
    for f in OUTS + ("CONVERG", "MAXTCORR.dat") + mo.MODEL_FILES:
        assert _same(os.path.join(ref, "M424_T38230", f), os.path.join(new, f)), f
    assert _same(os.path.join(ref, "INDAT.DAT"), os.path.join(w, "M424_T38230", "INDAT.DAT"))
    conv = json.load(open(os.path.join(w, "results", "M424_T38230", "convergence.json")))
    c = conv["convergence"]
    assert conv["status"] == "ok" and conv["niter"] == 61
    assert c["converged"] and c["n_iter"] == 59 and c["converged_it"] == 58 and not c["at_cap"]
    shutil.rmtree(w, ignore_errors=True)


def _extract(idx, dest):
    from ppmpy.synspec import fwresults
    res = _need(os.path.join(M424_RUN, "results"))
    return fwresults.extract_points(res, idx, dest, nproc=2)


@pytest.mark.fastwind
@pytest.mark.slow
def test_real_cli_one_point_571348():
    """'one' (CLI) for point 571348: pnlte_failed at 37298.910 K ('error in ne -- nlteopt', ledger columns 1-5 as in
    production); at +1 K every archived file of the production retry (OUT.*, model files, INDAT) is byte-identical."""
    _real()
    w = _work("p571348")
    arch = os.path.join(w, "archived")
    _extract([571348], arch)
    tpl = os.path.join(PROJECT, "fastwind", "INDAT_M424test.DAT")
    formal = os.path.join(PROJECT, "fastwind", "FORMAL_INPUT_He3")
    env = _env(PYTHONPATH=PPMPY)
    procs = {}
    for key, teff in (("fail", "37298.910"), ("retry", "37299.910")):
        cmd = [sys.executable, "-m", "ppmpy.synspec.fastwind", "one", "571348", teff, "--out", os.path.join(w, key),
               "--stage", os.path.join(w, "stage_" + key), "--root", FW_ROOT, "--build", "v10.6_HHe", "--template",
               tpl, "--formal", formal, "--keep", "model", "--json"]
        procs[key] = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      universal_newlines=True)
    out = {k: p.communicate(timeout=1800) for k, p in procs.items()}
    assert procs["fail"].returncode == 1 and procs["retry"].returncode == 0, out
    rf, rr = json.loads(out["fail"][0]), json.loads(out["retry"][0])
    assert rf["status"] == "pnlte_failed" and rf["niter"] == 19 and "error" in rf["flags"]
    assert rr["status"] == "ok" and rr["niter"] == 63 and rr["convergence"]["converged"]
    led = glob.glob(os.path.join(M424_RUN, "results", "task_0028", "part_*.idx"))
    prod = [ln.split() for p in led for ln in open(p) if ln.startswith("571348 ")]
    assert open(os.path.join(w, "fail", "P571348", "meta.txt")).read().split()[:5] == prod[0][:5]
    a, b = os.path.join(arch, "P571348"), os.path.join(w, "retry", "P571348")
    assert open(os.path.join(a, "meta.txt")).read().split()[:5] == open(os.path.join(b, "meta.txt")).read().split()[:5]
    names = sorted(f for f in os.listdir(a) if f != "meta.txt")
    assert set(OUTS) | set(mo.MODEL_FILES) <= set(names)
    for f in names:
        assert _same(os.path.join(a, f), os.path.join(b, f)), f
    shutil.rmtree(w, ignore_errors=True)


@pytest.mark.fastwind
@pytest.mark.slow
def test_real_fw_imu_run_sh():
    """fw_imu_run.sh (rerun-formal with the patched build) for two representatives: OUT_IMU.* and OUT.* byte-identical
    to /scratch/ppathak/fastwind_imu/runs."""
    _real()
    reps = _need(os.path.join(IMU, "representatives.txt"))
    w = _work("imu")
    lines = [ln for ln in open(reps) if ln.split() and os.path.basename(ln.split()[3]) in IMU_REPS]
    assert len(lines) == len(IMU_REPS)
    for ln in lines:
        _need(ln.split()[3])
    sel = os.path.join(w, "reps.txt")
    with open(sel, "w") as f:
        f.write("".join(lines))
    r = _run(["bash", _script("fw_imu_run.sh"), sel, os.path.join(w, "runs"), "2"], _env(), timeout=1200)
    assert r.returncode == 0 and "2 of 2 models have OUT_IMU files" in r.stdout, r.stdout
    for name in IMU_REPS:
        for n in LINES:
            for kind in ("OUT_IMU", "OUT"):
                f = "{}.{}_VTV010".format(kind, n)
                assert _same(os.path.join(IMU, "runs", name, name, f), os.path.join(w, "runs", name, name, f)), (name, f)
    shutil.rmtree(w, ignore_errors=True)


def _pilot_cli(w):
    """The 8-point pilot through the CLI, 4 workers: SIGUSR1 after 60 s (exit 3), then the resume (exit 0)."""
    with open(_need(os.path.join(M424_RUN, "points.txt"))) as f:
        rows = {int(t[0]): t[1] for t in (ln.split() for ln in f) if int(t[0]) in PILOT}
    with open(os.path.join(w, "points.txt"), "w") as f:
        f.write("".join("{} {}\n".format(i, rows[i]) for i in PILOT))
    cmd = [sys.executable, "-m", "ppmpy.synspec.fastwind", "run", w, "-K", "1", "-k", "0", "--nworkers", "4",
           "--keep", "model", "--pack-interval", "120", "--root", FW_ROOT, "--build", "v10.6_HHe",
           "--template", os.path.join(PROJECT, "fastwind", "INDAT_M424test.DAT"),
           "--formal", os.path.join(PROJECT, "fastwind", "FORMAL_INPUT_He3")]
    env = _env(PYTHONPATH=PPMPY)
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    time.sleep(60)
    p.send_signal(signal.SIGUSR1)
    out = p.communicate(timeout=300)[0]
    assert p.returncode == 3, out
    r = _run(cmd, env, timeout=3600)
    assert r.returncode == 0, r.stdout
    return os.path.join(w, "results")


@pytest.mark.fastwind
@pytest.mark.slow
def test_real_pilot_cli_vs_archive():
    """8 production points (4 capped at the iteration limit) through the CLI with a stop and a resume: every archived
    file (OUT.*, model files, INDAT.DAT) byte-identical, ledger columns 1-5 equal, merge_task reads the parts; capped
    points 'ok' but not converged."""
    from ppmpy.synspec import fwresults
    import numpy as np
    given = os.environ.get("PPMPY_SYNSPEC_FW_REAL_PILOT")
    w = _work("pilot")
    if given:
        results = _need(given)
    else:
        _real()
        results = _pilot_cli(w)
    new = os.path.join(w, "new")
    arch = os.path.join(w, "archived")
    fwresults.extract_points(results, list(PILOT), new, nproc=2)
    _extract(list(PILOT), arch)
    for i in PILOT:
        name = "P{:06d}".format(i)
        a, b = os.path.join(arch, name), os.path.join(new, name)
        ma, mb = open(os.path.join(a, "meta.txt")).read().split(), open(os.path.join(b, "meta.txt")).read().split()
        assert ma[:5] == mb[:5], (ma, mb)
        names = sorted(f for f in os.listdir(a) if f != "meta.txt")
        assert set(OUTS) | set(mo.MODEL_FILES) <= set(names)
        for f in names:
            assert _same(os.path.join(a, f), os.path.join(b, f)), (name, f)
        conv = json.load(open(os.path.join(b, "convergence.json")))
        c = conv["convergence"]
        if i in CAPPED:
            assert mb[3] == "102" and not c["converged"] and c["at_cap"], (i, c)
        else:
            assert c["converged"] and not c["at_cap"], (i, c)
    pts = _need(os.path.join(M424_RUN, "points.npz"))
    m = os.path.join(w, "merged.npz")
    info = fwresults.merge_task(results, "task_0000", m, pts)
    assert info["npoint"] == 8 and info["status"] == {"ok": 8}
    z = np.load(m)
    for row, i in enumerate(z["idx"]):
        for j, n in enumerate(LINES):
            o = fwresults.read_out(os.path.join(arch, "P{:06d}".format(int(i)), "OUT.{}_VTV010".format(n)), nrow=161)
            assert np.array_equal(z["fnorm"][row, j], o["fnorm"].astype(np.float32))
    shutil.rmtree(w, ignore_errors=True)
