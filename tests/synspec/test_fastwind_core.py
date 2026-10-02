"""Tests of ppmpy.synspec.fastwind (core: install, indat, formal, logs, model, fake).

* stdlib only: every module imports with numpy / scipy / matplotlib blocked, with this python and with the host's
  /usr/bin/python3 (where present), which also runs fake.selftest() (no pytest there);
* Indat: byte identity with the legacy awk edit of fw_sphere_point.sh on 10^4 random T_eff values (awk run as a
  subprocess), round trips, column rules, validation; the schema equals fwresults.INDAT_SCHEMA;
* formal: FORMAL_INPUT parsing (':' comments, 80 columns), formal_suffix table (formalsol.f90), stdin answers, names;
* logs: real logs and convergence files of the reference runs (copies in data/fastwind, and the full files when
  /scratch is there), the convergence verdict against fw_itmore_check.py on the 48 ITMORE-check models, legacy grep /
  awk / tail commands as subprocesses;
* model with the fake FASTWIND: ok, keep modes, deterministic failure and the +1 K retry, crash, cap, formal failure,
  timeouts that kill the process group but not a same-named decoy, stop_all, rerun_formal with OUT_IMU, concurrency,
  and the packed parts read by fwresults (merge_task, read_ledger, check_indat_premise);
* review fixes: pformalsol killed / crashed / timed out after opening its OUT files (formal_failed, never ok;
  out_problem on every truncation), rerun_formal skipping only complete outputs, rerun_formal and stage() from many
  threads, staging another build into the same root, hidden temporaries and old results, monotonic time limits,
  stop_all from a signal handler inside the lock, '\n'-only line splitting (\x85, \x0c in comments), the
  convergence digest, and fw_sphere_point.sh itself against run_model (same files and bytes);
* real FASTWIND (markers fastwind + slow; 3 models at once, ~3-6 min): the M424 reference model, point 571348
  ('error in ne -- nlteopt' at 37298.910 K) and its +1 K retry against the legacy runs and the production ledger, and
  a pformalsol rerun with the intensity build.
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
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import pytest

from conftest import ROOT
from ppmpy.synspec import fastwind as fwm
from ppmpy.synspec.fastwind import fake, formal as fo, indat as ind, install as ins, logs, model as mo

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fastwind")
# frozen copies of the original shell runner and the INDAT templates (tests/synspec/legacy/README.txt)
PROJECT = os.environ.get("PPMPY_SYNSPEC_M424_ANALYSIS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy"))
FW_ROOT = os.environ.get("PPMPY_FASTWIND_ROOT", "/scratch/ppathak/FW_10.6.4.1")
FW_RUNS = os.environ.get("PPMPY_SYNSPEC_M424_FWRUNS", "/scratch/ppathak/fastwind_runs")
M424_RUN = os.environ.get("PPMPY_SYNSPEC_M424_RUN", "/scratch/ppathak/fastwind_sphere/d3200_r4050_N1236544")
SHADOW = os.environ.get("PPMPY_SYNSPEC_FASTWIND_SCRATCH", "/scratch/ppathak/synspec_shadow/m6/core")
HOST_PYTHON = "/usr/bin/python3"
LINES = ["HEI4026", "HEII4200", "HEI4922"]


def _data(name):
    return os.path.join(DATA, name)


def _need(path):
    if not os.path.exists(path):
        pytest.skip("not available: {}".format(path))
    return path


def _alive(pid):
    return fake._alive(pid)


@pytest.fixture(scope="module")
def fakefw(tmp_path_factory):
    """A fake install with both builds, a link-staged root and a copy-staged intensity root."""
    d = tmp_path_factory.mktemp("fakefw")
    inst = fake.install_fake(str(d / "fw"))
    st = inst.stage(str(d / "stage"))
    inst_imu = ins.FastwindInstall(inst.root, fake.FAKE_BUILD, formal_build=fake.FAKE_IMU_BUILD)
    st_imu = inst_imu.stage(str(d / "stage_imu"), mode="copy")
    return dict(dir=str(d), inst=inst, st=st, inst_imu=inst_imu, st_imu=st_imu,
                tpl=os.path.join(inst.root, "INDAT.template"), formal=os.path.join(inst.root, "FORMAL_INPUT"))


@pytest.fixture
def cfg(fakefw):
    """Set the fake behaviour for one test and restore the default afterwards."""
    def setter(**kw):
        fake.set_fake_config(fakefw["inst"].build, **kw)
        fake.set_fake_config(fakefw["inst_imu"].formal_build, **kw)
    yield setter
    setter()


# ------------------------------------------------------------------------------------------------------------------
# stdlib only
# ------------------------------------------------------------------------------------------------------------------
def test_stdlib_only_this_python():
    ok, msg = fake.check_stdlib_only()
    assert ok, msg


def test_stdlib_only_host_python_and_selftest(tmp_path):
    if not os.path.exists(HOST_PYTHON):
        pytest.skip("no " + HOST_PYTHON)
    ok, msg = fake.check_stdlib_only(HOST_PYTHON)
    assert ok, msg
    code = ("import sys; sys.path.insert(0, {!r}); from ppmpy.synspec.fastwind import fake; "
            "fake.selftest(workdir={!r}, n_awk=300)").format(ROOT, str(tmp_path))
    p = subprocess.run([HOST_PYTHON, "-c", code], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "selftest passed" in p.stdout


def test_package_exports():
    for name in fwm.__all__:
        assert hasattr(fwm, name), name


# ------------------------------------------------------------------------------------------------------------------
# Indat
# ------------------------------------------------------------------------------------------------------------------
def _templates():
    out = [_data("INDAT_M424test.DAT")]
    out += sorted(glob.glob(os.path.join(PROJECT, "fastwind", "INDAT_*.DAT")))
    return out


def test_indat_schema_equals_fwresults():
    from ppmpy.synspec import fwresults
    assert ind.INDAT_SCHEMA == fwresults.INDAT_SCHEMA


@pytest.mark.parametrize("path", _templates())
def test_indat_roundtrip(path):
    raw = open(path, "rb").read()
    I = ind.Indat.read(path)
    assert I.to_bytes() == raw
    assert I.copy().to_bytes() == raw
    assert I.validate() == []


def test_indat_fields_and_get():
    I = ind.Indat.read(_data("INDAT_M424test.DAT"))
    f = I.fields()
    assert list(f)[:5] == ["MODNAM", "OPTNEUPDATE", "HE_ONE", "ITSTART", "ITMORE"]
    assert f["MODNAM"] == "M424test" and f["TEFF"] == 38230.0 and f["LOGG"] == 4.25 and f["RSTAR"] == 6.21
    assert f["ITSTART"] == 0 and f["ITMORE"] == 100 and f["OPTTLUCY"] is True and f["OPTMOD"] is False
    assert f["MDOT"] == 1e-10 and f["VINF"] == 2500.0 and f["SET_FIRST"] == 1 and f["SET_STEP"] == 2
    assert I.raw("TEFF") == "38230.0" and I.get("ENATCOR") is True
    assert I.extra_lines() == ["1., 0.1, 0.2                                   CLF, VCLSTART, VCLMAX"]
    with pytest.raises(KeyError):
        I.get("NOPE")
    # the numpy-side reader of check_indat_premise sees the same values
    from ppmpy.synspec import fwresults
    fr = fwresults._indat_fields(I.to_text())
    for k, v in f.items():
        if k in ind.INTEGER_FIELDS:
            assert fr[k][0] == float(v)
        else:
            assert fr[k][0] == v, k


def _awk_values(rng, n):
    vals = []
    for i in range(n):
        r = i % 6
        x = rng.uniform(33000.0, 42000.0)
        if r == 0:
            vals.append("%.3f" % x)                     # points.txt
        elif r == 1:
            vals.append(repr(x))                        # full double
        elif r == 2:
            vals.append("%.4f" % (round(x * 16) / 16))  # exact binary ties of '%.3f' (x.0625, x.1875, ...)
        elif r == 3:
            vals.append("%.4f" % (round(x * 2000) / 2000 + 0.0005))   # decimal ties (inexact in binary)
        elif r == 4:
            vals.append(str(int(x)))
        else:
            vals.append("%.6e" % x)
    return vals


def _names(rng, n):
    out = []
    for i in range(n):
        if i % 50 == 7:
            out.append("M424_" + "x" * rng.randint(1, 41))   # long names, < 47 characters
        else:
            out.append("P%06d" % rng.randrange(1236544))
    return out


@pytest.mark.parametrize("path", _templates())
def test_indat_edit_equals_awk(path, tmp_path):
    """Byte identity with fw_sphere_point.sh's awk edit: 10^4 values for the M424 template, 300 for the others."""
    if not shutil.which("awk"):
        pytest.skip("no awk")
    n = 10000 if path.endswith("INDAT_M424test.DAT") else 300
    seed = zlib.crc32(os.path.basename(path).encode())        # reproducible (str hash() is salted per process)
    rng = random.Random(seed)
    pairs = list(zip(_names(rng, n), _awk_values(rng, n)))
    got = fake.awk_edit_batch(path, pairs)
    assert len(got) == n
    text = open(path, "rb").read().decode("latin-1")
    tpl = ind.Indat(text)
    bad = []
    for (name, t), g in zip(pairs, got):
        mine = tpl.copy().set(MODNAM=name, TEFF=float(t)).to_bytes()
        if mine != g or ind.edit_like_awk(text, name, t).encode("latin-1") != g:
            bad.append((name, t))
    assert not bad, "seed {}: {}".format(seed, bad[:5])
    # the batch (FNR, assignments) equals the legacy program itself (NR, -v) on a sample
    for name, t in pairs[:15]:
        assert fake.awk_edit(path, name, t) == tpl.copy().set(MODNAM=name, TEFF=float(t)).to_bytes()


def test_awk_program_is_the_legacy_one():
    p = _need(os.path.join(PROJECT, "fw_sphere_point.sh"))
    assert fake.AWK_EDIT in open(p).read()


def test_indat_set_rules():
    I = ind.Indat.read(_data("INDAT_M424test.DAT"))
    J = I.copy().set(ITMORE=50, ITSTART=10, ENATCOR=False, MDOT=2e-9, LOGG="4.000")
    lines = J.to_text().splitlines()
    assert lines[1] == "T  T   10  50                                  OPTNEUPDATE,HE_ONE,ITSTART,ITMORE"
    assert lines[1].index("OPTNEUPDATE") == I.to_text().splitlines()[1].index("OPTNEUPDATE")   # column kept
    assert lines[3].startswith("38230.0,     4.000,    6.2100")
    assert lines[5].startswith("2e-09,  0.1,")
    assert lines[9].startswith("F F 1 2 ")
    assert J.get("ITMORE") == 50 and J.get("ENATCOR") is False and J.get("MDOT") == 2e-9
    # long names keep one blank before the comment; > 50 characters are refused (CHARACTER*50)
    K = I.copy().set(MODNAM="N" * 48)
    assert K.to_text().splitlines()[0] == "N" * 48 + " CATALOG" and K.get("MODNAM") == "N" * 48
    for bad in (dict(MODNAM="N" * 51), dict(TEFF=float("nan")), dict(ITMORE=1.5), dict(ENATCOR=1),
                dict(TEFF="38 000"), dict(LOGG=True), dict(MODNAM=5), dict(OPTTLUCY="X")):
        with pytest.raises(ValueError):
            I.copy().set(**bad)
    with pytest.raises(KeyError):
        I.copy().set(NOPE=1)
    with pytest.raises(KeyError):
        ind.Indat("ONLY_ONE_LINE\n").set(TEFF=1.0)
    assert I.copy().set(TEFF=37298.9104999).raw("TEFF") == "37298.910"
    assert I.copy().set(TEFF="37298.91").raw("TEFF") == "37298.91"     # str: verbatim


def test_indat_write_atomic(tmp_path):
    I = ind.Indat.read(_data("INDAT_M424test.DAT")).set(MODNAM="P000001")
    p = str(tmp_path / "INDAT.DAT")
    I.write(p)
    assert open(p, "rb").read() == I.to_bytes()
    assert os.listdir(str(tmp_path)) == ["INDAT.DAT"]


def test_indat_validate_problems():
    t = open(_data("INDAT_M424test.DAT")).read()
    good = ind.Indat(t)
    assert good.validate() == []
    cases = {
        "OPTTLUCY": good.copy().set(OPTTLUCY=False),
        "MDOT": good.copy().set(MDOT=0.0),
        "ITMORE": good.copy().set(ITMORE=0),
        "clumping": ind.Indat(t.replace("1., 0.1, 0.2  ", "0.5, 0.1, 0.2 ")),
        "missing": ind.Indat("\n".join(t.splitlines()[:6]) + "\n"),
        "LINES": ind.Indat(t.replace("10., 1.0, T  T", "10., 0.0, T  T")),
    }
    for what, I in cases.items():
        probs = I.validate()
        assert probs, what
    assert any("OPTHOPF" in p for p in cases["OPTTLUCY"].validate())


# ------------------------------------------------------------------------------------------------------------------
# formal
# ------------------------------------------------------------------------------------------------------------------
def test_formal_input_read_roundtrip(tmp_path):
    F = fo.FormalInput.read(_data("FORMAL_INPUT_He3"))
    assert F.vsini == 0.0 and F.names == LINES
    assert F.lines[0].components == [("HE12P3", "HE15D3", 0, 1), ("HE24", "HE213", 0, 1)]
    assert F.lines[2].components == [("HE12P1", "HE14D1", 0, 1)]
    assert F.to_text() == open(_data("FORMAL_INPUT_He3")).read()
    G = F.subset(["HEI4922", "HEI4026"])
    assert G.names == ["HEI4922", "HEI4026"]
    H = fo.FormalInput.from_text(G.to_text())
    assert H == G and H.names == ["HEI4922", "HEI4026"]
    p = G.write(str(tmp_path / "FORMAL_INPUT"))
    assert fo.FormalInput.read(p) == G


def test_formal_names_match_pformalsol_log():
    """pformalsol prints the lines it treats: the names read from FORMAL_INPUT are those."""
    log = open(_data("M424_T38230.pformalsol_head.log")).read().splitlines()
    i = [k for k, s in enumerate(log) if "LINES TO BE TREATED" in s][0]
    names = []
    for s in log[i + 1:]:
        if "END OF FORMAL-INPUT" in s:
            break
        names.append(s.strip())
    assert names == fo.FormalInput.read(_data("FORMAL_INPUT_He3")).names


def test_formal_free_field_rules():
    text = (":T comment line, ignored HEIGNORED 1 A B 0 1\n"
            "0. :inline comment: \n"
            "LINEA 1 L1 U1 0 1 :trailing comment LINEX 1 A B 0 1\n"
            "LINEB 2 L1 U1 0 1\n"
            "     L2 U2 3 0\n"
            + "LINEC 1 L1 U1 0 1".ljust(80) + "LINED 1 A B 0 1\n")      # beyond column 80: ignored
    F = fo.FormalInput.from_text(text)
    assert F.names == ["LINEA", "LINEB", "LINEC"]
    assert F.lines[1].components == [("L1", "U1", 0, 1), ("L2", "U2", 3, 0)]
    with pytest.raises(ValueError):
        fo.FormalInput.from_text("0.\nLINEA 2 L1 U1 0 1\n")
    with pytest.raises(ValueError):                         # a name longer than 19 characters
        fo.FormalInput.from_text("0.\n" + "X" * 20 + " 1 a b 0 1\n")


def test_formal_install_line_lists():
    files = glob.glob(os.path.join(FW_ROOT, "inicalc", "FORMAL_INPUT_A10*"))
    if not files:
        pytest.skip("FASTWIND install not available")
    for f in files:
        F = fo.FormalInput.read(f)
        assert F.names and all(len(n) <= fo.LINE_NAME_MAX for n in F.names), f


SUFFIX_TABLE = [
    ((10, 0.1), 0, "VTV010"), ("10 0.1", 0, "VTV010"), ("10, 0.1", 0, "VTV010"), ((10, 10), 0, "VT010"),
    (10, 0, "VT010"), ("10", 0, "VT010"), (5.7, 0, "VT005"), ((7.99, 0.2), 0, "VTV007"), (999.9, 0, "VT999"),
    (0, 0, ""), ((0, 0), 0, ""), (0, 1, "ESC"), (10, 1, "ESC_VT010"), ((12, 12), 1, "ESC_VT012"),
    ((100, 300), 0, "VTV100"),
]


@pytest.mark.parametrize("vturb,iescat,suffix", SUFFIX_TABLE)
def test_formal_suffix_table(vturb, iescat, suffix):
    assert fo.formal_suffix(vturb, iescat) == suffix


@pytest.mark.parametrize("vturb,iescat", [((0, 5), 0), ((10, 0.1), 1), (10, 2), (1000, 0), ((1000, 1000), 0),
                                          (-1, 0), ((), 0), ((1, 2, 3), 0)])
def test_formal_suffix_refused(vturb, iescat):
    with pytest.raises(ValueError):
        fo.formal_suffix(vturb, iescat)


def test_formalsol_stdin_and_names():
    assert fo.formalsol_stdin("P000001") == b"P000001\n10 0.1\n0\n"        # printf "%s\n10 0.1\n0\n"
    assert fo.formalsol_stdin("M", 10.0, 1) == b"M\n10.0\n1\n"
    assert fo.formalsol_stdin("M", "15 0.2") == b"M\n15 0.2\n0\n"
    for bad in ("", "x" * 61, "a\nb"):
        with pytest.raises(ValueError):
            fo.formalsol_stdin(bad)
    with pytest.raises(ValueError):
        fo.formalsol_stdin("M", (0, 5))
    assert fo.out_name("HEI4026", "VTV010") == "OUT.HEI4026_VTV010"
    assert fo.out_name("HEI4026", "") == "OUT.HEI4026"
    assert fo.out_name("HEI4026", "VTV010", kind="OUT_IMU") == "OUT_IMU.HEI4026_VTV010"
    with pytest.raises(ValueError):
        fo.out_name("HEI4026", "VTV010", kind="OUTX")
    assert fo.out_names(_data("FORMAL_INPUT_He3")) == ["OUT.{}_VTV010".format(n) for n in LINES]
    ref = os.path.join(FW_RUNS, "M424_T38230", "M424_T38230")
    if os.path.isdir(ref):
        assert sorted(f for f in os.listdir(ref) if f.startswith("OUT.")) == sorted(fo.out_names(
            _data("FORMAL_INPUT_He3")))


# ------------------------------------------------------------------------------------------------------------------
# logs
# ------------------------------------------------------------------------------------------------------------------
def _model_dir(tmp_path, prefix):
    d = tmp_path / prefix
    d.mkdir()
    for f in ("CONVERG", "MAXTCORR.dat"):
        shutil.copyfile(_data("{}.{}".format(prefix, f)), str(d / f))
    return str(d)


def test_parse_pnlte_log_digests():
    r = logs.parse_pnlte_log(_data("M424_T38230.pnlte_digest.log"))
    assert r["acabose"] and r["niter"] == 61 and r["last_iteration"] == 59
    assert r["T_tau23"] == "39591.1272317698" and r["T_tau23_value"] == 39591.1272317698
    assert r["temp_converged"] and r["n_tcorr"] == 19 and r["all_levels_ok"] and r["errors"] == []
    assert r["cpu_time"] == pytest.approx(145.328628) and r["last_line"] == "! ESTO ES EL ACABOSE !"
    r = logs.parse_pnlte_log(_data("R571348a.pnlte_digest.log"))
    assert not r["acabose"] and r["niter"] == 19 and set(r["errors"]) == {"error in ne -- nlteopt"}
    assert r["last_line"] == "error in ne -- nlteopt" and r["T_tau23"] == "38827.3222565564"
    # content given as bytes / text
    raw = open(_data("R571348a.pnlte_digest.log"), "rb").read()
    assert logs.parse_pnlte_log(raw)["niter"] == 19
    assert logs.parse_pnlte_log(raw.decode("latin-1"))["niter"] == 19
    assert logs.parse_pnlte_log(b"")["niter"] == 0


@pytest.mark.parametrize("run", ["M424_T38230", "R571348a", "R571348b"])
def test_parse_pnlte_log_vs_legacy_commands(run):
    """niter, T_tau23 and the tail equal fw_sphere_point.sh's grep / awk / tail pipelines on the full logs."""
    log = _need(os.path.join(FW_RUNS, run, "pnlte.log"))
    r = logs.parse_pnlte_log(log)
    sh = lambda c: subprocess.run(["bash", "-c", c], capture_output=True).stdout
    assert r["niter"] == int(sh('grep -c "ITERATION NO" {}'.format(log)))
    assert (r["T_tau23"] or "").encode() == sh('grep "T(TAUROSS=2/3)" {} | tail -1 | awk \'{{print $NF}}\''.format(
        log)).strip()
    assert r["acabose"] == (subprocess.run(["grep", "-q", "ESTO ES EL ACABOSE", log]).returncode == 0)
    assert logs.tail_bytes(log, 40) == sh("tail -40 {}".format(log))


def test_tail_bytes(tmp_path):
    for content in (b"", b"a", b"a\n", b"a\nb", b"\n\n\n", b"".join(b"%d\n" % i for i in range(100)),
                    b"".join(b"%d\n" % i for i in range(100)) + b"last"):
        p = str(tmp_path / "f")
        open(p, "wb").write(content)
        assert logs.tail_bytes(p, 40) == subprocess.run(["tail", "-40", p], capture_output=True).stdout, content


def test_convergence_reference_and_failure(tmp_path):
    c = logs.convergence(_model_dir(tmp_path, "M424_T38230"), 0, 100)
    assert c["converged"] and c["converged_it"] == 58 and c["temp_converged_it"] == 41 and c["n_iter"] == 59
    assert c["consistent"] and not c["at_cap"] and c["n_tcorr"] == 19 and c["emaxtc_last"] == pytest.approx(2.821e-3)
    assert c["meanerr_last"] == pytest.approx(-4.61783944470346)
    rows = logs.parse_converg(_data("M424_T38230.CONVERG"))
    assert len(rows) == 59 and rows[0] == (1, 0.688225535139492, -0.456005819355345)
    mt = logs.parse_maxtcorr(_data("M424_T38230.MAXTCORR.dat"))
    assert len(mt) == 19 and mt[0] == (0, 0.064337, 1) and mt[-1] == (18, 0.002821, 41)
    c = logs.convergence(_model_dir(tmp_path, "R571348a"), 0, 100)
    assert not c["converged"] and c["temp_converged_it"] is None and c["n_iter"] == 16 and not c["consistent"]
    c = logs.convergence(str(tmp_path / "nothing"), 0, 100)
    assert not c["has_converg"] and c["n_iter"] == 0 and not c["converged"]
    assert logs.parse_maxtcorr(b"   0  0.64337E-01 ***\n")[0][2] is None


def test_convergence_itmore_copies(tmp_path):
    c = logs.convergence(_model_dir(tmp_path, "itmore_P025059"), 0, 300)       # never T-converged, capped
    assert not c["converged"] and c["temp_converged_it"] is None and c["at_cap"] and c["n_iter"] == 300
    assert c["consistent"]
    c = logs.convergence(_model_dir(tmp_path, "itmore_P021304"), 0, 300)       # converged late
    assert c["converged"] and c["temp_converged_it"] == 97 and c["converged_it"] == 99 and c["n_iter"] == 100
    c = logs.convergence(_model_dir(tmp_path, "itmore_P007951"), 0, 100)
    assert c["converged"] and c["temp_converged_it"] == 41 and c["n_iter"] == 59
    c = logs.convergence(str(tmp_path / "itmore_P025059"), 0, 300, enatcor=False)
    assert c["temp_converged_it"] == 0


def test_convergence_vs_fw_itmore_check():
    """The verdict of fw_itmore_check.py (T converged, last row meets the criterion) on all 48 ITMORE-check models."""
    root = _need(os.path.join(FW_RUNS, "itmore_check"))
    rows = [ln.split() for ln in open(os.path.join(root, "compare.txt")).read().splitlines()[1:]
            if ln.startswith("P")]
    assert len(rows) == 48
    for r in rows:
        name, tconv, conv = r[0], int(r[5]), r[6] == "True"
        itmore = ind.Indat.read(os.path.join(root, "runs", name, "INDAT.DAT")).get("ITMORE")
        c = logs.convergence(os.path.join(root, "runs", name, name), 0, itmore)
        assert c["converged"] == conv, name
        assert (c["temp_converged_it"] if c["temp_converged_it"] is not None else -1) == tconv, name
        assert c["consistent"], name
        assert c["n_iter"] == int(r[4]), name


def test_classify():
    ok_log = dict(acabose=True, errors=[], all_levels_ok=True)
    conv_ok = dict(has_converg=True, converged=True, criterion_last=True, temp_converged_it=41, consistent=True)
    conv_cap = dict(has_converg=True, converged=False, criterion_last=False, temp_converged_it=None, consistent=True)
    assert logs.classify(ok_log, True, False, 0, conv_ok) == ("ok", ())
    assert logs.classify(ok_log, False) == ("formal_failed", ())
    assert logs.classify(ok_log, None)[0] == "formal_failed"
    assert logs.classify(ok_log, True, False, 0, conv_cap) == ("ok", ("not_converged", "temp_not_converged"))
    bad = dict(acabose=False, errors=["error in ne -- nlteopt"])
    assert logs.classify(bad, None, False, 0) == ("pnlte_failed", ("error",))
    assert logs.classify(dict(acabose=False, errors=[]), None, True, -15) == ("pnlte_timeout", ("signal", "timeout"))
    assert logs.classify(dict(acabose=False, errors=[]), None, False, 174) == ("pnlte_failed", ("nonzero_exit",))


# ------------------------------------------------------------------------------------------------------------------
# install
# ------------------------------------------------------------------------------------------------------------------
def _elf64(path, interp):
    """A minimal ELF64 little-endian file with one PT_INTERP program header."""
    import struct
    s = interp.encode() + b"\0"
    phoff, ph = 64, 56
    off = phoff + ph
    hdr = b"\x7fELF" + bytes([2, 1, 1]) + bytes(9)
    hdr += struct.pack("<HHIQQQIHHHHHH", 2, 62, 1, 0, phoff, 0, 0, 64, ph, 1, 64, 0, 0)
    prog = struct.pack("<IIQQQQQQ", 3, 4, off, off, off, len(s), len(s), 1)
    with open(path, "wb") as f:
        f.write(hdr + prog + s)
    os.chmod(path, 0o755)


def _elf32(path, interp):
    import struct
    s = interp.encode() + b"\0"
    phoff, ph = 52, 32
    off = phoff + ph
    hdr = b"\x7fELF" + bytes([1, 1, 1]) + bytes(9)
    hdr += struct.pack("<HHIIIIIHHHHHH", 2, 3, 1, 0, phoff, 0, 0, 52, ph, 1, 40, 0, 0)
    prog = struct.pack("<IIIIIIII", 3, off, off, off, len(s), len(s), 4, 1)
    with open(path, "wb") as f:
        f.write(hdr + prog + s)


def test_elf_interpreter(tmp_path):
    _elf64(str(tmp_path / "a64"), "/nonexistent/ld-test.so.2")
    _elf32(str(tmp_path / "a32"), "/lib/ld-linux.so.2")
    assert ins.elf_interpreter(str(tmp_path / "a64")) == "/nonexistent/ld-test.so.2"
    assert ins.elf_interpreter(str(tmp_path / "a32")) == "/lib/ld-linux.so.2"
    (tmp_path / "s").write_text("#!/bin/sh\n")
    assert ins.elf_interpreter(str(tmp_path / "s")) is None
    (tmp_path / "t").write_bytes(b"\x7fELF")
    assert ins.elf_interpreter(str(tmp_path / "t")) is None
    py = os.path.realpath(sys.executable)
    assert ins.elf_interpreter(py) is None or os.path.exists(ins.elf_interpreter(py))
    p = os.path.join(FW_ROOT, "v10.6_HHe", "pnlte_A10HHe.eo")
    if os.path.exists(p):
        assert ins.elf_interpreter(p).startswith("/cvmfs/") and ins.elf_interpreter(p).endswith("ld-linux-x86-64.so.2")


def test_install_fake_check_and_problems(tmp_path):
    inst = fake.install_fake(str(tmp_path / "fw"))
    assert inst.tag == "A10HHe" and inst.atom == "A10HHe.dat" and inst.check() == []
    assert os.path.islink(os.path.join(inst.root, "inicalc", "inicalc"))
    assert not inst.has_imu_patch()
    assert ins.FastwindInstall(inst.root, fake.FAKE_BUILD, formal_build=fake.FAKE_IMU_BUILD).has_imu_patch()
    assert ins.FastwindInstall(inst.root, os.path.join(inst.root, fake.FAKE_BUILD)).build == inst.build
    fp = inst.fingerprint()
    assert set(fp) >= {"sha256:pnlte_A10HHe.eo", "sha256:pformalsol_A10HHe.eo", "sha256:ATOM_FILE",
                       "sha256:A10HHe.dat", "tag", "root"}
    with open(os.path.join(inst.build, "A10HHe.dat"), "a") as f:
        f.write("changed\n")
    assert inst.fingerprint()["sha256:A10HHe.dat"] != fp["sha256:A10HHe.dat"]
    assert inst.fingerprint()["sha256:pnlte_A10HHe.eo"] == fp["sha256:pnlte_A10HHe.eo"]
    # break it
    shutil.rmtree(os.path.join(inst.root, "inicalc", "RaymondSmith"))
    os.unlink(os.path.join(inst.root, "inicalc", "HOPFPARA_ALL_met"))
    os.chmod(inst.pformalsol, 0o644)
    _elf64(inst.pnlte, "/nonexistent/ld-test.so.2")
    probs = inst.check()
    assert any("RaymondSmith" in p for p in probs) and any("HOPFPARA_ALL_met" in p for p in probs)
    assert any("not executable" in p for p in probs) and any("ELF interpreter /nonexistent" in p for p in probs)
    assert any("ATOM_FILE names" in p for p in ins.FastwindInstall(inst.root, fake.FAKE_BUILD, atom="A11HHe").check())
    assert ins.FastwindInstall(str(tmp_path / "none"), "b", atom="A10HHe").check()[0].endswith("does not exist")
    with pytest.raises(FileNotFoundError):
        inst.stage(str(tmp_path / "st"))


def test_install_from_env(tmp_path):
    inst = fake.install_fake(str(tmp_path / "fw"))
    env = {ins.ENV_ROOT: inst.root, ins.ENV_BUILD: fake.FAKE_BUILD, ins.ENV_FORMAL_BUILD: fake.FAKE_IMU_BUILD,
           ins.ENV_LAUNCHER: "nice -n 5"}
    i = ins.FastwindInstall.from_env(env)
    assert i.formal_build.endswith(fake.FAKE_IMU_BUILD) and i.launcher == ("nice", "-n", "5") and i.has_imu_patch()
    assert ins.FastwindInstall.from_env(env, launcher=()).launcher == ()
    with pytest.raises(KeyError):
        ins.FastwindInstall.from_env({ins.ENV_ROOT: inst.root})


def test_stage_link_and_copy(tmp_path):
    inst = fake.install_fake(str(tmp_path / "fw"))
    os.symlink("..", os.path.join(inst.root, "inicalc", "DATA", "uplink"))     # a link inside a data dir
    st = inst.stage(str(tmp_path / "L"))
    assert st.check() == [] and st.mode == "link" and st.has_imu is False
    assert os.readlink(os.path.join(st.root, "inicalc", "DATA")) == inst.data_dir("DATA")
    assert os.readlink(os.path.join(st.root, "HOPFPARA_ALL_HHe")) == inst.hopf_file("HOPFPARA_ALL_HHe")
    assert sorted(os.listdir(os.path.join(st.root, "inicalc"))) == sorted(ins.DATA_DIRS)    # no inicalc/inicalc
    assert st.bin_files() == sorted(["A10HHe.dat", "ATOM_FILE", "pformalsol_A10HHe.eo", "pnlte_A10HHe.eo"])
    st2 = ins.StagedRoot.open(st.root)
    assert (st2.tag, st2.atom, st2.mode, st2.has_imu) == ("A10HHe", "A10HHe.dat", "link", False)
    assert inst.stage(st.root).check() == []                                    # idempotent
    stc = inst.stage(str(tmp_path / "C"), mode="copy")
    for d in ins.DATA_DIRS:
        p = os.path.join(stc.root, "inicalc", d)
        assert os.path.isdir(p) and not os.path.islink(p)
    assert os.path.islink(os.path.join(stc.root, "inicalc", "DATA", "uplink"))   # copied as a link
    assert open(os.path.join(stc.bin_dir, "pnlte_A10HHe.eo"), "rb").read() == open(inst.pnlte, "rb").read()
    assert not [f for f in os.listdir(stc.root) if f.startswith(".")]
    # a link staging replaced by a copy staging (and the files of bin/)
    inst.stage(st.root, mode="copy")
    assert not os.path.islink(os.path.join(st.root, "inicalc", "DATA"))
    assert not os.path.islink(os.path.join(st.root, "bin", "pnlte_A10HHe.eo"))
    with pytest.raises(ValueError):
        inst.stage(str(tmp_path / "X"), mode="move")


def test_stage_concurrent(tmp_path):
    """8 processes stage the same directory at once (copy and link): all succeed, nothing temporary is left."""
    inst = fake.install_fake(str(tmp_path / "fw"))
    for mode in ("copy", "link"):
        dest = str(tmp_path / ("S" + mode))
        code = ("import sys; sys.path.insert(0, {!r}); from ppmpy.synspec.fastwind import install as i; "
                "s = i.FastwindInstall({!r}, {!r}).stage({!r}, mode={!r}); assert s.check() == [], s.check()"
                ).format(ROOT, inst.root, fake.FAKE_BUILD, dest, mode)
        ps = [subprocess.Popen([sys.executable, "-c", code], stderr=subprocess.PIPE) for _ in range(8)]
        errs = [p.communicate()[1] for p in ps]
        assert all(p.returncode == 0 for p in ps), errs
        left = [os.path.join(r, f) for r, ds, fs in os.walk(dest) for f in ds + fs if f.startswith(".")]
        assert left == []
        assert ins.StagedRoot.open(dest).check() == []


# ------------------------------------------------------------------------------------------------------------------
# run_model with the fake FASTWIND
# ------------------------------------------------------------------------------------------------------------------
META_RE = re.compile(r"^(\d+) (\S+) (ok|formal_failed|pnlte_failed|pnlte_timeout) (\d+) (\S+) (\d+\.\d) (\d+\.\d)\n$")


def test_run_model_ok(fakefw, tmp_path):
    from ppmpy.synspec import fwresults
    res = str(tmp_path / "res")
    r = mo.run_model(fakefw["st"], (1563, "38230.000"), res, fakefw["tpl"], fakefw["formal"])
    d = os.path.join(res, "P001563")
    assert r["status"] == "ok" and r["flags"] == () and r["result_dir"] == d and r["niter"] == 33
    assert sorted(os.listdir(res)) == ["P001563"]
    assert sorted(os.listdir(d)) == sorted(["INDAT.DAT", "meta.txt", "CONVERG", "MAXTCORR.dat", "convergence.json"]
                                           + ["OUT.{}_VTV010".format(n) for n in LINES])
    meta = open(os.path.join(d, "meta.txt")).read()
    m = META_RE.match(meta)
    assert m and m.group(1) == "1563" and m.group(2) == "38230.000" and m.group(3) == "ok" and m.group(4) == "33"
    assert meta == r["meta"]
    pm = fwresults.parse_meta(meta)
    assert pm["idx"] == 1563 and pm["teff"] == 38230.0 and pm["niter"] == 33 and pm["T_tau23"] > 38230
    tpl = open(fakefw["tpl"]).read()
    assert open(os.path.join(d, "INDAT.DAT")).read() == ind.edit_like_awk(tpl, "P001563", "38230.000")
    o = fwresults.read_out(os.path.join(d, "OUT.HEI4026_VTV010"))
    assert o["nrow"] == 161 and o["ew_fastwind"] < 0 and abs(o["lam"][80] - 4026.0) < 1e-9
    js = json.load(open(os.path.join(d, "convergence.json")))
    assert js["status"] == "ok" and js["convergence"]["converged"] and js["convergence"]["converged_it"] == 30
    assert js["convergence"]["temp_converged_it"] == 23 and js["expected"] == fo.out_names(fakefw["formal"])
    assert not os.path.exists(os.path.join(fakefw["st"].root, "P001563"))           # run directory removed
    assert mo.active_groups() == {}
    # keep='model' and extras=False (exactly the legacy KEEP_MODEL=1 file set)
    r = mo.run_model(fakefw["st"], (1564, 38230.5), res, fakefw["tpl"], fakefw["formal"], keep="model", extras=False)
    d = os.path.join(res, "P001564")
    assert sorted(os.listdir(d)) == sorted(["INDAT.DAT", "meta.txt"] + list(mo.MODEL_FILES)
                                           + ["OUT.{}_VTV010".format(n) for n in LINES])
    assert open(os.path.join(d, "meta.txt")).read().split()[1] == "38230.5"
    assert ind.Indat.read(os.path.join(d, "INDAT.DAT")).raw("TEFF") == "38230.500"
    # rerun of the same point replaces the result
    r = mo.run_model(fakefw["st"], (1564, 38230.5), res, fakefw["tpl"], fakefw["formal"])
    assert "MODEL" not in os.listdir(d) and not [f for f in os.listdir(res) if ".old" in f or ".tmp" in f]
    with pytest.raises(ValueError):
        mo.run_model(fakefw["st"], (1, 38230.0), res, fakefw["tpl"], fakefw["formal"], keep="all")
    with pytest.raises(ValueError):
        mo.run_model(fakefw["st"], (1, float("nan")), res, fakefw["tpl"], fakefw["formal"])
    with pytest.raises(ValueError):
        mo.run_model(fakefw["st"], 5, res, fakefw["tpl"], fakefw["formal"])


def test_run_model_failures(fakefw, cfg, tmp_path):
    cfg(fail=[37298.91], crash=[37400.0], formal_fail=[37500.0], cap=[37600.0])
    res = str(tmp_path / "res")
    r = mo.run_model(fakefw["st"], (571348, "37298.910"), res, fakefw["tpl"], fakefw["formal"], keep="model",
                     keep_run=True)
    d = os.path.join(res, "P571348")
    assert r["status"] == "pnlte_failed" and r["niter"] == 19 and "error" in r["flags"]
    assert sorted(os.listdir(d)) == sorted(["INDAT.DAT", "meta.txt", "pnlte_tail.log", "CONVERG", "MAXTCORR.dat",
                                           "convergence.json"])
    log = os.path.join(fakefw["st"].root, "P571348", "pnlte.log")
    assert open(os.path.join(d, "pnlte_tail.log"), "rb").read() == subprocess.run(
        ["tail", "-40", log], capture_output=True).stdout
    assert open(os.path.join(d, "meta.txt")).read().split()[2:4] == ["pnlte_failed", "19"]
    shutil.rmtree(os.path.join(fakefw["st"].root, "P571348"))
    # deterministic: the same T_eff fails again, +1 K converges (the retry rule of fw_sphere_merge.py --combine)
    assert mo.run_model(fakefw["st"], (571348, "37298.910"), res, fakefw["tpl"], fakefw["formal"])["status"] == \
        "pnlte_failed"
    r = mo.run_model(fakefw["st"], (571348, "37299.910"), str(tmp_path / "res2"), fakefw["tpl"], fakefw["formal"])
    assert r["status"] == "ok"
    r = mo.run_model(fakefw["st"], (2, 37400.0), res, fakefw["tpl"], fakefw["formal"])
    assert r["status"] == "pnlte_failed" and r["returncode"] == 174 and "nonzero_exit" in r["flags"]
    r = mo.run_model(fakefw["st"], (3, 37500.0), res, fakefw["tpl"], fakefw["formal"], keep="model")
    assert r["status"] == "formal_failed" and r["formal_returncode"] == 0
    assert not [f for f in os.listdir(os.path.join(res, "P000003")) if f.startswith(("OUT", "MODEL"))]
    # capped: finished (ok, as the legacy status) but not converged; niter = ITMORE + 2 for ITMORE = 40
    r = mo.run_model(fakefw["st"], (4, {"TEFF": 37600.0, "ITMORE": 40}), res, fakefw["tpl"], fakefw["formal"])
    assert r["status"] == "ok" and r["niter"] == 42 and "not_converged" in r["flags"]
    assert r["convergence"]["at_cap"] and r["convergence"]["cap"] == 40 and r["teff"] == "37600.0"
    assert ind.Indat.read(os.path.join(res, "P000004", "INDAT.DAT")).get("ITMORE") == 40
    # field dict without TEFF: meta.txt takes the template's TEFF text
    r = mo.run_model(fakefw["st"], (5, {"LOGG": 4.0}), res, fakefw["tpl"], fakefw["formal"])
    assert r["status"] == "ok" and r["teff"] == "38230.0"


def test_run_model_timeout_kills_group_not_decoy(fakefw, cfg, tmp_path):
    cfg(hang=[37200.0], formal_hang=[37300.0])
    decoy_dir = tmp_path / "decoy"
    decoy_dir.mkdir()
    env = dict(os.environ, **{fake.ENV_MODE: "hang"})
    decoy = subprocess.Popen([fakefw["inst"].pnlte], cwd=str(decoy_dir), env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        for _ in range(100):
            if (decoy_dir / "hang_child.pid").exists():
                break
            time.sleep(0.05)
        decoy_child = int((decoy_dir / "hang_child.pid").read_text())
        res = str(tmp_path / "res")
        t = time.time()
        r = mo.run_model(fakefw["st"], (7, 37200.0), res, fakefw["tpl"], fakefw["formal"], pnlte_timeout=2, grace=1,
                         keep_run=True)
        el = time.time() - t
        assert r["status"] == "pnlte_timeout" and r["timed_out"] and "timeout" in r["flags"] and el < 10
        assert r["returncode"] == -signal.SIGTERM
        child = int(open(os.path.join(fakefw["st"].root, "P000007", "hang_child.pid")).read())
        time.sleep(0.2)
        assert not _alive(child)
        assert decoy.poll() is None and _alive(decoy_child)
        assert open(os.path.join(res, "P000007", "meta.txt")).read().split()[2] == "pnlte_timeout"
        # a pformalsol hanging inside the last line: killed at formal_timeout, status formal_failed although every
        # OUT file exists (the last one open and empty, as formalsol.f90 leaves it; the legacy test said ok)
        r = mo.run_model(fakefw["st"], (8, 37300.0), res, fakefw["tpl"], fakefw["formal"], formal_timeout=2, grace=1)
        assert r["status"] == "formal_failed" and r["formal_timed_out"]
        assert {"formal_timeout", "formal_signal", "formal_incomplete"} <= set(r["flags"]), r["flags"]
        assert r["formal_problems"] == {"OUT.HEI4922_VTV010": "empty"}
        d = os.path.join(res, "P000008")
        assert sorted(f for f in os.listdir(d) if f.startswith("OUT.")) == sorted(fo.out_names(fakefw["formal"]))
        assert open(os.path.join(d, "meta.txt")).read().split()[2] == "formal_failed"
        assert decoy.poll() is None
        assert mo.active_groups() == {}
    finally:
        os.killpg(decoy.pid, signal.SIGKILL)
        decoy.wait()


def test_run_model_sigterm_ignored_escalates(fakefw, tmp_path):
    """A process that ignores SIGTERM is killed with SIGKILL after the grace period."""
    st = fakefw["inst"].stage(str(tmp_path / "st"))
    st.launcher = (sys.executable, "-c", "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                   "while True: time.sleep(0.1)")
    t = time.time()
    r = mo.run_model(st, (9, 38000.0), str(tmp_path / "res"), fakefw["tpl"], fakefw["formal"], pnlte_timeout=1,
                     grace=1)
    assert r["status"] == "pnlte_timeout" and r["returncode"] == -signal.SIGKILL and time.time() - t < 8


def test_stop_all(fakefw, cfg, tmp_path):
    cfg(hang=[37200.0])
    res = str(tmp_path / "res")
    out = {}

    def work():
        try:
            mo.run_model(fakefw["st"], (11, 37200.0), res, fakefw["tpl"], fakefw["formal"], pnlte_timeout=60, grace=1)
            out["r"] = "returned"
        except mo.ModelInterrupted as e:
            out["r"] = e
    th = threading.Thread(target=work)
    th.start()
    try:
        for _ in range(200):
            if mo.active_groups():
                break
            time.sleep(0.05)
        time.sleep(0.5)
        assert mo.stop_all() == 1
        th.join(20)
        assert isinstance(out.get("r"), mo.ModelInterrupted)
        assert not os.listdir(res) and not os.path.exists(os.path.join(fakefw["st"].root, "P000011"))
        assert mo.active_groups() == {}
        with pytest.raises(mo.ModelInterrupted):
            mo.run_model(fakefw["st"], (12, 38000.0), res, fakefw["tpl"], fakefw["formal"])
    finally:
        mo.reset_stop()
    assert mo.run_model(fakefw["st"], (12, 38000.0), res, fakefw["tpl"], fakefw["formal"])["status"] == "ok"


def test_rerun_formal_imu(fakefw, tmp_path):
    from ppmpy.synspec import fwresults
    res = str(tmp_path / "res")
    mo.run_model(fakefw["st"], (21, 38100.0), res, fakefw["tpl"], fakefw["formal"], keep="model")
    src = os.path.join(res, "P000021")
    runs = str(tmp_path / "runs")
    r = mo.rerun_formal(fakefw["st_imu"], src, fakefw["formal"], run_root=runs)
    assert r["status"] == "ok" and r["missing_model_files"] == []
    m = os.path.join(runs, "P000021", "P000021")
    assert r["model_dir"] == m and os.path.islink(os.path.join(m, "MODEL"))
    assert os.readlink(os.path.join(runs, "inicalc")) == os.path.join(fakefw["st_imu"].root, "inicalc")
    lam, p, ic, il, cnt = fwresults.read_out_imu(os.path.join(m, "OUT_IMU.HEI4026_VTV010"), counts=True)
    assert lam.shape == (161,) and ic.shape == (161, 5) and cnt == dict(nray=5, ncore=4)
    for n in LINES:          # same model files -> the same OUT as the original run
        assert open(os.path.join(m, "OUT.{}_VTV010".format(n)), "rb").read() == open(
            os.path.join(src, "OUT.{}_VTV010".format(n)), "rb").read()
    assert mo.rerun_formal(fakefw["st_imu"], src, fakefw["formal"], run_root=runs)["status"] == "skipped"
    assert mo.rerun_formal(fakefw["st_imu"], src, fakefw["formal"], run_root=runs, overwrite=True)["status"] == "ok"
    with pytest.raises(ValueError):
        mo.rerun_formal(fakefw["st"], src, fakefw["formal"], run_root=runs)
    r = mo.rerun_formal(fakefw["st"], src, fakefw["formal"], run_root=str(tmp_path / "runs2"), kind="OUT")
    assert r["status"] == "ok"
    # the result of run_model with the intensity build keeps OUT_IMU.* too
    r = mo.run_model(fakefw["st_imu"], (22, 38100.0), res, fakefw["tpl"], fakefw["formal"])
    assert sum(f.startswith("OUT_IMU.") for f in os.listdir(r["result_dir"])) == 3


def _pack(res, stage, outdir):
    """fw_sphere_task.sh's pack(): move finished results to STAGE, tar -czf -C STAGE ., cat */meta.txt > .idx."""
    fin = [p for p in sorted(os.listdir(res)) if not p.endswith(".tmp")]
    os.makedirs(stage, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    for p in fin:
        shutil.move(os.path.join(res, p), stage)
    part = os.path.join(outdir, "part_test_{}".format(len(os.listdir(outdir))))
    subprocess.run(["tar", "-czf", part + ".tar.gz.tmp", "-C", stage, "."], check=True)
    os.replace(part + ".tar.gz.tmp", part + ".tar.gz")
    subprocess.run(["bash", "-c", "cat {}/*/meta.txt > {}.idx".format(stage, part)], check=True)
    shutil.rmtree(stage)
    return fin


def test_parts_read_by_fwresults(fakefw, cfg, tmp_path):
    """Results of run_model packed like fw_sphere_task.sh are read unchanged by merge_task / ledgers / premise."""
    import numpy as np
    from ppmpy.synspec import fwresults
    cfg(fail=[37000.0])
    teffs = ["38230.000", "37000.000", "36500.250", "39999.999", "37000.000", "38000.125"]
    res = str(tmp_path / "res")
    out = {}
    for i, t in enumerate(teffs):
        tt = "37001.000" if i == 4 else t                  # point 4: the +1 K retry of point 1's T_eff
        out[i] = mo.run_model(fakefw["st"], (i, tt), res, fakefw["tpl"], fakefw["formal"],
                              keep="model" if i % 2 else "profiles")
    results = str(tmp_path / "results")
    _pack(res, str(tmp_path / "stage"), os.path.join(results, "task_0000"))
    n = len(teffs)
    points = dict(idx=np.arange(n), teff=np.array([float(t) for t in teffs]))
    s = fwresults.merge_task(results, "task_0000", str(tmp_path / "m.npz"), points, copy_keys=())
    assert s["npoint"] == n and s["status"] == {"ok": 5, "pnlte_failed": 1} and s["nnudge"] == 1
    z = np.load(str(tmp_path / "m.npz"))
    assert list(z["status"]) == ["ok", "pnlte_failed", "ok", "ok", "ok", "ok"]
    assert list(z["niter"]) == [33, 19, 33, 33, 33, 33] and z["teff_nudge"][4] == 1.0
    for i in (0, 2):
        assert np.isfinite(z["fnorm"][i]).all() and np.isnan(z["fnorm"][1]).all()
    led = glob.glob(os.path.join(results, "task_0000", "*.idx"))[0]
    assert sorted(i for i, _ in fwresults.read_ledger(led)) == list(range(n))
    prem = fwresults.check_indat_premise(results, template=fakefw["tpl"])
    assert prem["passed"] and prem["n_points"] == n, prem


def test_same_name_twice_is_refused(fakefw, cfg, tmp_path):
    """A second run of a name already running in the same root raises ModelBusy and leaves the first alone."""
    cfg(hang=[37200.0])
    res = str(tmp_path / "res")
    out = {}
    th = threading.Thread(target=lambda: out.update(r=mo.run_model(fakefw["st"], (31, 37200.0), res, fakefw["tpl"],
                                                                    fakefw["formal"], pnlte_timeout=3, grace=1)))
    th.start()
    for _ in range(200):
        if os.path.exists(os.path.join(fakefw["st"].root, "P000031", "pnlte.log")):
            break
        time.sleep(0.02)
    with pytest.raises(mo.ModelBusy):
        mo.run_model(fakefw["st"], (31, 38000.0), str(tmp_path / "res2"), fakefw["tpl"], fakefw["formal"])
    with pytest.raises(mo.ModelBusy):
        mo.rerun_formal(fakefw["st"], str(tmp_path), fakefw["formal"], name="P000031", kind="OUT")
    th.join(30)
    assert out["r"]["status"] == "pnlte_timeout"
    assert not os.path.exists(str(tmp_path / "res2")) or not os.listdir(str(tmp_path / "res2"))
    assert not [f for f in os.listdir(fakefw["st"].root) if f.endswith(".lock")]


def test_concurrent_models(fakefw, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    res = str(tmp_path / "res")
    with ThreadPoolExecutor(8) as ex:
        rs = list(ex.map(lambda i: mo.run_model(fakefw["st"], (100 + i, 38000.0 + i), res, fakefw["tpl"],
                                                fakefw["formal"]), range(16)))
    assert all(r["status"] == "ok" for r in rs) and len(os.listdir(res)) == 16
    assert not [f for f in os.listdir(fakefw["st"].root) if f.startswith("P0001")]


# ------------------------------------------------------------------------------------------------------------------
# review fixes (2026-10-02): complete outputs, threads, signal handlers, clocks, control bytes, legacy script
# ------------------------------------------------------------------------------------------------------------------
def _cuts(n, step):
    return sorted(set(list(range(0, n, step)) + [n - 1, n - 2]))


def test_out_problem_complete_and_truncated(fakefw, tmp_path):
    """Complete OUT / OUT_IMU files (fake and real) pass; every truncation of them is reported."""
    r = mo.run_model(fakefw["st_imu"], (71, 38000.0), str(tmp_path / "res"), fakefw["tpl"], fakefw["formal"],
                     extras=False)
    assert r["status"] == "ok" and r["formal_problems"] == {}
    d = r["result_dir"]
    outs = [os.path.join(d, n) for n in fo.out_names(fakefw["formal"])]
    imus = [os.path.join(d, n) for n in fo.out_names(fakefw["formal"], kind="OUT_IMU")]
    real_out = sorted(glob.glob(os.path.join(FW_RUNS, "M424_T38230", "M424_T38230", "OUT.*")))
    real_imu = sorted(glob.glob(os.path.join("/scratch/ppathak/fastwind_imu/runs", "P001563", "P001563",
                                             "OUT_IMU.*")))
    for f in outs + real_out:
        assert fo.out_problem(f) is None, f
    for f in imus + real_imu:
        assert fo.out_problem(f, "OUT_IMU") is None, f
    assert fo.out_problem(str(tmp_path / "nope")) == "missing"
    for src, kind, step in [(outs[0], "OUT", 7), (imus[0], "OUT_IMU", 53)] + \
            [(f, "OUT", 211) for f in real_out[:1]] + [(f, "OUT_IMU", 4099) for f in real_imu[:1]]:
        data = open(src, "rb").read()
        p = str(tmp_path / "cut")
        for k in _cuts(len(data), step):
            with open(p, "wb") as fh:
                fh.write(data[:k])
            assert fo.out_problem(p, kind) is not None, (src, k)
    # the specific reasons
    data = open(outs[0], "rb").read()
    rows = data.split(b"\n")
    for content, why in ((b"", "empty"), (data[:-1], "no final newline"),
                         (b"\n".join(rows[:161]) + b"\n", "no EW trailer"),
                         (b"\n".join(rows[:100]) + b"\n" + rows[161] + b"\n", "100 table rows < 161")):
        with open(p, "wb") as fh:
            fh.write(content)
        assert fo.out_problem(p) == why
    assert fo.out_problem(p, nrow=100) is None and fo.out_problem(p, nrow=None) is None
    with pytest.raises(ValueError):
        fo.out_problem(p, "OUTX")


def test_run_model_formal_partial(fakefw, cfg, tmp_path):
    """pformalsol crashed (exit 174) or killed after writing part of the last table: formal_failed, not ok."""
    cfg(formal_partial=[37310.0], formal_partial_hang=[37320.0])
    res = str(tmp_path / "res")
    r = mo.run_model(fakefw["st"], (91, 37310.0), res, fakefw["tpl"], fakefw["formal"])
    assert r["status"] == "formal_failed" and r["formal_returncode"] == 174
    assert {"formal_nonzero_exit", "formal_incomplete"} <= set(r["flags"]) and "formal_timeout" not in r["flags"]
    assert r["formal_problems"] == {"OUT.HEI4922_VTV010": "20 table rows < 161"}
    d = r["result_dir"]
    # every OUT file exists, so the legacy ls | wc -l test would have recorded 'ok'
    assert sorted(f for f in os.listdir(d) if f.startswith("OUT.")) == sorted(fo.out_names(fakefw["formal"]))
    assert open(os.path.join(d, "meta.txt")).read().split()[2] == "formal_failed"
    assert "pnlte_tail.log" in os.listdir(d)
    js = json.load(open(os.path.join(d, "convergence.json")))
    assert js["formal_problems"] == r["formal_problems"] and js["formal_returncode"] == 174
    r = mo.run_model(fakefw["st"], (92, 37320.0), res, fakefw["tpl"], fakefw["formal"], formal_timeout=2, grace=1)
    assert r["status"] == "formal_failed" and {"formal_timeout", "formal_incomplete"} <= set(r["flags"])
    # with the intensity build the last OUT_IMU is complete (written first) but OUT is not
    r = mo.run_model(fakefw["st_imu"], (93, 37310.0), res, fakefw["tpl"], fakefw["formal"])
    assert r["status"] == "formal_failed" and list(r["formal_problems"]) == ["OUT.HEI4922_VTV010"]
    assert mo.active_groups() == {}


def test_rerun_formal_complete_outputs(fakefw, cfg, tmp_path):
    """rerun_formal skips only complete outputs, and reports 'failed' for a crash with complete files."""
    res = str(tmp_path / "res")
    src = mo.run_model(fakefw["st"], (94, 37330.0), res, fakefw["tpl"], fakefw["formal"], keep="model")["result_dir"]
    runs = str(tmp_path / "runs")
    r = mo.rerun_formal(fakefw["st_imu"], src, fakefw["formal"], run_root=runs)
    assert r["status"] == "ok" and r["problems"] == {}
    imu = os.path.join(r["model_dir"], "OUT_IMU.HEI4922_VTV010")
    good = open(imu, "rb").read()
    with open(imu, "wb") as fh:                          # a killed earlier rerun: truncated file
        fh.write(good[:len(good) // 2])
    r = mo.rerun_formal(fakefw["st_imu"], src, fakefw["formal"], run_root=runs)
    assert r["status"] == "ok" and open(imu, "rb").read() == good       # rerun, not skipped
    assert mo.rerun_formal(fakefw["st_imu"], src, fakefw["formal"], run_root=runs)["status"] == "skipped"
    # a crash after the OUT_IMU tables are complete: still 'failed' (exit status)
    cfg(formal_partial=[37330.0])
    r = mo.rerun_formal(fakefw["st_imu"], src, fakefw["formal"], run_root=runs, overwrite=True)
    assert r["status"] == "failed" and r["returncode"] == 174 and r["problems"] == {}
    r = mo.rerun_formal(fakefw["st_imu"], src, fakefw["formal"], run_root=runs, overwrite=True, kind="OUT")
    assert r["status"] == "failed" and list(r["problems"]) == ["OUT.HEI4922_VTV010"]
    # a hanging rerun is killed at its time limit
    cfg(formal_partial_hang=[37330.0])
    r = mo.rerun_formal(fakefw["st_imu"], src, fakefw["formal"], run_root=runs, overwrite=True, kind="OUT",
                        timeout=2, grace=1)
    assert r["status"] == "failed" and r["timed_out"]


def test_rerun_formal_threads_fresh_root(fakefw, tmp_path):
    """8 threads rerun pformalsol into one fresh run root (its links are made concurrently)."""
    res = str(tmp_path / "res")
    one = mo.run_model(fakefw["st"], (41, 38100.0), res, fakefw["tpl"], fakefw["formal"], keep="model")["result_dir"]
    srcs = []
    for i in range(8):
        d = str(tmp_path / "models" / ("P%06d" % (300 + i)))
        shutil.copytree(one, d)
        srcs.append(d)
    runs = str(tmp_path / "runs_fresh")
    with ThreadPoolExecutor(8) as ex:
        rs = list(ex.map(lambda s: mo.rerun_formal(fakefw["st_imu"], s, fakefw["formal"], run_root=runs), srcs))
    assert [r["status"] for r in rs] == ["ok"] * 8
    assert sorted(f for f in os.listdir(runs)) == sorted(["inicalc", "HOPFPARA_ALL_HHe", "HOPFPARA_ALL_met"] +
                                                          [os.path.basename(s) for s in srcs])


def _dotfiles(top):
    return [os.path.join(r, f) for r, ds, fs in os.walk(top) for f in ds + fs if f.startswith(".")]


@pytest.mark.parametrize("mode", ["copy", "link"])
def test_stage_threads(tmp_path, mode):
    """16 threads of one process stage the same directory at once: all succeed, nothing temporary is left."""
    inst = fake.install_fake(str(tmp_path / "fw"))
    dest = str(tmp_path / "S")
    with ThreadPoolExecutor(16) as ex:
        sts = list(ex.map(lambda _: inst.stage(dest, mode=mode), range(16)))
    assert all(s.check() == [] for s in sts) and _dotfiles(dest) == []
    assert ins.StagedRoot.open(dest).check() == []
    # the helpers themselves, from many threads on one name
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda i: ins.atomic_symlink(inst.root if i % 2 else inst.build, str(tmp_path / "lnk")), range(64)))
        list(ex.map(lambda i: ins.atomic_write(str(tmp_path / "w"), b"%d" % i), range(64)))
    assert os.readlink(str(tmp_path / "lnk")) in (inst.root, inst.build) and int(open(str(tmp_path / "w")).read()) < 64
    assert _dotfiles(str(tmp_path)) == []


def test_stage_switch_build_refresh_and_guard(tmp_path):
    inst = fake.install_fake(str(tmp_path / "fw"))
    imu = ins.FastwindInstall(inst.root, fake.FAKE_BUILD, formal_build=fake.FAKE_IMU_BUILD)
    dest = str(tmp_path / "S")
    staged_pf = os.path.join(dest, "bin", "pformalsol_A10HHe.eo")
    assert inst.stage(dest, mode="copy").has_imu is False
    st = imu.stage(dest, mode="copy")                       # another formal build into the same dest
    assert st.has_imu is True and ins.StagedRoot.open(dest).has_imu is True
    assert open(staged_pf, "rb").read() == open(imu.pformalsol, "rb").read()
    st = inst.stage(dest, mode="copy")
    assert st.has_imu is False and open(staged_pf, "rb").read() == open(inst.pformalsol, "rb").read()
    with open(inst.pnlte, "a") as f:                        # a rebuilt binary replaces the staged copy
        f.write("# rebuilt\n")
    inst.stage(dest, mode="copy")
    assert open(os.path.join(dest, "bin", "pnlte_A10HHe.eo"), "rb").read() == open(inst.pnlte, "rb").read()
    st = imu.stage(dest, mode="link")                       # links over copies: data dirs kept, bin/ linked
    assert st.has_imu is True and os.path.islink(staged_pf) and not os.path.islink(os.path.join(dest, "inicalc",
                                                                                                 "DATA"))
    for junk in (".pnlte_A10HHe.eo.cp123", ".X.lnk9"):      # leftovers of a crashed staging are never linked
        open(os.path.join(dest, "bin", junk), "w").close()
    assert not [f for f in st.bin_files() if f.startswith(".")]
    other = fake.install_fake(str(tmp_path / "fw2"))
    with pytest.raises(ValueError):                         # copied data dirs of another install would be kept
        other.stage(dest, mode="copy")
    L = str(tmp_path / "L")
    inst.stage(L)
    assert other.stage(L).check() == [] and os.readlink(os.path.join(L, "inicalc", "DATA")) == other.data_dir("DATA")


def test_replace_result_hidden_old_name(fakefw, tmp_path, monkeypatch):
    res = str(tmp_path / "res")
    mo.run_model(fakefw["st"], (51, 38000.0), res, fakefw["tpl"], fakefw["formal"])
    seen = []
    real = os.rename

    def rec(a, b, *k, **kw):
        seen.append((str(a), str(b)))
        return real(a, b, *k, **kw)
    monkeypatch.setattr(mo.os, "rename", rec)
    mo.run_model(fakefw["st"], (51, 38000.0), res, fakefw["tpl"], fakefw["formal"], extras=False)
    monkeypatch.undo()
    final = os.path.join(res, "P000051")
    olds = [b for a, b in seen if a == final]
    assert len(olds) == 1 and os.path.basename(olds[0]).startswith(".P000051.old")
    assert sorted(os.listdir(res)) == ["P000051"] and "convergence.json" not in os.listdir(final)


def test_run_group_monotonic(tmp_path, monkeypatch):
    """A wall-clock step neither kills a model early nor inflates its elapsed time."""
    real = time.time
    n = [0]

    def jumpy():
        n[0] += 1
        return real() + (1.0e6 if n[0] % 2 else -1.0e6)
    monkeypatch.setattr(mo.time, "time", jumpy)
    rc, to, el = mo.run_group([sys.executable, "-c", "import time; time.sleep(0.6)"], str(tmp_path),
                              str(tmp_path / "log"), timeout=30, grace=1)
    monkeypatch.undo()
    assert rc == 0 and not to and 0.5 < el < 10


def test_stop_all_from_signal_handler(fakefw, cfg, tmp_path):
    """A SIGUSR1 handler calling stop_all while its thread holds the registry lock: no deadlock, model stopped."""
    cfg(hang=[37200.0])
    code = r'''
import os, signal, sys, threading, time
sys.path.insert(0, sys.argv[1])
from ppmpy.synspec.fastwind import model as mo
st, tpl, formal, res = sys.argv[2:6]
out = {}
def work():
    try:
        mo.run_model(st, (61, 37200.0), res, tpl, formal, pnlte_timeout=60, grace=1)
        out["r"] = "returned"
    except mo.ModelInterrupted:
        out["r"] = "interrupted"
th = threading.Thread(target=work)
th.start()
for _ in range(500):
    if mo.active_groups():
        break
    time.sleep(0.02)
time.sleep(0.3)
hits = []
signal.signal(signal.SIGUSR1, lambda s, f: hits.append(mo.stop_all()))
with mo._lock:                       # the handler runs in this thread while it holds the lock
    os.kill(os.getpid(), signal.SIGUSR1)
    t = time.monotonic()
    while not hits and time.monotonic() - t < 5:
        time.sleep(0.01)
th.join(30)
print(hits, out.get("r"), len(mo.active_groups()), os.listdir(res) if os.path.isdir(res) else [])
'''
    p = subprocess.run([sys.executable, "-c", code, ROOT, fakefw["st"].root, fakefw["tpl"], fakefw["formal"],
                        str(tmp_path / "res")], capture_output=True, text=True, timeout=90)
    assert p.returncode == 0, p.stderr
    assert p.stdout.split() == ["[1]", "interrupted", "0", "[]"], p.stdout + p.stderr


def test_indat_control_bytes_in_comments(tmp_path):
    """\\x85 (cp1252 ellipsis as latin-1) and \\x0c in comments do not shift the INDAT lines (as Fortran / awk)."""
    lines = open(_data("INDAT_M424test.DAT"), "rb").read().decode("latin-1").split("\n")
    lines[2] += " \x85 cp1252 ellipsis"
    lines[4] += " \x0c form feed \x0b \x1c \x1d \x1e"
    text = "\n".join(lines)
    I = ind.Indat(text)
    assert I.to_text() == text and len(I.extra_lines()) == 1
    assert I.get("TEFF") == 38230.0 and I.get("RMAX") == 120.0 and I.get("MDOT") == 1e-10 and I.get("OPTMIXED") == 0.0
    assert I.validate() == []
    J = I.copy().set(MODNAM="P571348", TEFF=37298.91)
    jl = J.to_text().split("\n")
    assert J.get("TEFF") == 37298.91 and jl[3].startswith("37298.910,") and jl[2] == lines[2] and jl[4] == lines[4]
    if shutil.which("awk"):
        p = str(tmp_path / "tpl")
        open(p, "wb").write(text.encode("latin-1"))
        got = subprocess.run(["awk", "-v", "n=P571348", "-v", "t=37298.91", fake.AWK_EDIT, p], capture_output=True,
                             env=dict(os.environ, LC_ALL="C")).stdout
        assert got == J.to_bytes()
    assert ind.split_lines("a\x85b\x0cc\r\nd\n\ne") == ["a\x85b\x0cc\r\n", "d\n", "\n", "e"]
    assert ind.split_lines("") == [] and ind.split_lines("x\n") == ["x\n"]


def test_formal_control_bytes_in_comments():
    text = open(_data("FORMAL_INPUT_He3")).read()
    t2 = text.replace(":T He II 4199.90 (n=4-11)", ":T He II 4199.90 \x85 (n=4-11)\x0c HEIX 1 A B 0 1")
    t2 = t2.replace("HEII4200  1  HE24   HE211 0  1", "HEII4200  1  HE24   HE211 0  1 :c\x85 HEIY 1 A B 0 1:")
    assert t2 != text
    F = fo.FormalInput.from_text(t2)
    assert F.names == LINES and F.to_text() == t2
    assert fo.FormalInput.from_text(text.replace("\n", "\r\n")).names == LINES


def test_classify_formal_flags():
    ok_log = dict(acabose=True, errors=[], all_levels_ok=True)
    assert logs.classify(ok_log, False, formal_timed_out=True, formal_returncode=-15,
                         formal_problems={"OUT.A": "empty", "OUT.B": "missing"}) == (
        "formal_failed", ("formal_incomplete", "formal_missing", "formal_signal", "formal_timeout"))
    assert logs.classify(ok_log, False, formal_returncode=174) == ("formal_failed", ("formal_nonzero_exit",))
    assert logs.classify(ok_log, True, formal_returncode=0, formal_problems={}) == ("ok", ())


def test_extras_digest(fakefw, tmp_path):
    res = str(tmp_path / "res")
    r = mo.run_model(fakefw["st"], (81, 38000.0), res, fakefw["tpl"], fakefw["formal"], extras="digest")
    d = r["result_dir"]
    assert sorted(os.listdir(d)) == sorted(["INDAT.DAT", "meta.txt", mo.DIGEST_FILE]
                                           + ["OUT.{}_VTV010".format(n) for n in LINES])
    text = open(os.path.join(d, mo.DIGEST_FILE)).read()
    g = mo.parse_digest(text)
    assert text.count("\n") == 1 and len(text) < 400 and list(g) == list(mo.DIGEST_KEYS)
    assert g["status"] == "ok" and g["flags"] == () and g["converged"] == "True" and g["n_iter"] == "31"
    assert g["temp_converged_it"] == "23" and g["formal_returncode"] == "0" and g["niter"] == "33"
    with pytest.raises(ValueError):
        mo.run_model(fakefw["st"], (82, 38000.0), res, fakefw["tpl"], fakefw["formal"], extras="yes")


def test_legacy_point_script_equals_run_model(fakefw, cfg, tmp_path):
    """fw_sphere_point.sh itself and run_model(extras=False) on the fake: same files, same bytes, same meta.txt
    columns 1-5, for ok / pnlte failure / formal failure / crash points and KEEP_MODEL=0 / 1."""
    script = _need(os.path.join(PROJECT, "fw_sphere_point.sh"))
    if not (shutil.which("bash") and shutil.which("timeout") and shutil.which("awk")):
        pytest.skip("bash, timeout or awk missing")
    cfg(fail=[37000.0], formal_fail=[37100.0], crash=[37400.0])
    inst = fakefw["inst"]
    fwl = inst.stage(str(tmp_path / "fwl"), mode="copy")
    shutil.copyfile(fakefw["tpl"], os.path.join(fwl.root, "INDAT.template"))
    shutil.copyfile(fakefw["formal"], os.path.join(fwl.root, "FORMAL_INPUT"))
    st2 = inst.stage(str(tmp_path / "py"), mode="copy")
    points = [(1, "38230.000"), (2, "37000.000"), (3, "37100.000"), (4, "37400.000"), (5, "36999.5")]
    for keep in (0, 1):
        res_l, res_p = str(tmp_path / "legacy{}".format(keep)), str(tmp_path / "py{}".format(keep))
        os.makedirs(res_l)
        env = dict(os.environ, FWL=fwl.root, RES=res_l, KEEP_MODEL=str(keep), PNLTE_TIMEOUT="60")
        for idx, t in points:
            subprocess.run(["bash", script, str(idx), t], env=env, check=True, timeout=120, capture_output=True)
            mo.run_model(st2, (idx, t), res_p, fakefw["tpl"], fakefw["formal"],
                         keep="model" if keep else "profiles", extras=False)
        assert sorted(os.listdir(res_l)) == sorted(os.listdir(res_p)) == ["P%06d" % i for i, _ in points]
        status = {}
        for name in os.listdir(res_l):
            a, b = os.path.join(res_l, name), os.path.join(res_p, name)
            assert sorted(os.listdir(a)) == sorted(os.listdir(b)), (keep, name)
            for f in os.listdir(a):
                if f != "meta.txt":
                    assert open(os.path.join(a, f), "rb").read() == open(os.path.join(b, f), "rb").read(), (name, f)
            ma, mb = open(os.path.join(a, "meta.txt")).read(), open(os.path.join(b, "meta.txt")).read()
            assert ma.split()[:5] == mb.split()[:5] and META_RE.match(ma) and META_RE.match(mb), (ma, mb)
            status[name] = ma.split()[2]
        assert status == {"P000001": "ok", "P000002": "pnlte_failed", "P000003": "formal_failed",
                          "P000004": "pnlte_failed", "P000005": "ok"}
    assert not [f for f in os.listdir(fwl.root) if f.startswith("P0")]


# ------------------------------------------------------------------------------------------------------------------
# the real FASTWIND (3 models at once; ~3-6 min)
# ------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real():
    if not (os.path.isdir(FW_ROOT) and os.path.isdir("/cvmfs")):
        pytest.skip("FASTWIND install or /cvmfs not available")
    inst = ins.FastwindInstall(FW_ROOT, "v10.6_HHe")
    inst_imu = ins.FastwindInstall(FW_ROOT, "v10.6_HHe", formal_build="v10.6_HHe_imu")
    probs = inst.check() + inst_imu.check()
    if probs:
        pytest.skip("FASTWIND not runnable here: " + "; ".join(probs))
    work = os.path.join(SHADOW, "real_{}".format(os.getpid()))
    shutil.rmtree(work, ignore_errors=True)
    st = inst.stage(os.path.join(work, "stage"))
    st2 = inst.stage(os.path.join(work, "stage2"))         # the retry has the same name: its own root
    st_imu = inst_imu.stage(os.path.join(work, "stage_imu"))
    tpl, formal = _data("INDAT_M424test.DAT"), _data("FORMAL_INPUT_He3")
    jobs = {"ref": ((1, "38230.000"), "res_ref", "model", st),
            "fail": ((571348, "37298.910"), "res_fail", "profiles", st),
            "retry": ((571348, "37299.910"), "res_retry", "profiles", st2)}
    out = {}

    def one(k):
        spec, rd, keep, root = jobs[k]
        try:
            out[k] = mo.run_model(root, spec, os.path.join(work, rd), tpl, formal, keep=keep, pnlte_timeout=1800)
        except Exception as e:                  # reported by the tests
            out[k] = dict(status="exception: {!r}".format(e), flags=(), niter=-1, convergence={})
    t = time.time()
    ths = [threading.Thread(target=one, args=(k,)) for k in jobs]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    out["wall"] = time.time() - t
    out.update(inst=inst, st=st, st_imu=st_imu, work=work)
    yield out
    shutil.rmtree(work, ignore_errors=True)


@pytest.mark.fastwind
@pytest.mark.slow
def test_real_reference_model(real):
    """INDAT_M424test at 38230 K: converged like the reference run M424_T38230 (59 iterations), same profiles."""
    from ppmpy.synspec import fwresults
    import numpy as np
    r = real["ref"]
    assert r["status"] == "ok" and r["flags"] == (), r
    assert r["convergence"]["converged"] and r["convergence"]["n_iter"] == 59 and r["niter"] == 61
    assert r["convergence"]["temp_converged_it"] == 41 and r["convergence"]["converged_it"] == 58
    ref = os.path.join(FW_RUNS, "M424_T38230", "M424_T38230")
    if os.path.isdir(ref):
        assert r["T_tau23"] == logs.parse_pnlte_log(os.path.join(FW_RUNS, "M424_T38230", "pnlte.log"))["T_tau23"]
        for n in LINES:
            a = fwresults.read_out(os.path.join(r["result_dir"], "OUT.{}_VTV010".format(n)))
            b = fwresults.read_out(os.path.join(ref, "OUT.{}_VTV010".format(n)))
            assert np.array_equal(a["lam"], b["lam"]) and np.abs(a["fnorm"] - b["fnorm"]).max() < 1e-4, n
    assert sorted(f for f in os.listdir(r["result_dir"]) if f in mo.MODEL_FILES) == sorted(mo.MODEL_FILES)


@pytest.mark.fastwind
@pytest.mark.slow
def test_real_failing_point_and_retry(real):
    """Point 571348: 'error in ne -- nlteopt' at 37298.910 K (19 'ITERATION NO' lines), ok at +1 K; as production."""
    f, g = real["fail"], real["retry"]
    assert f["status"] == "pnlte_failed" and f["niter"] == 19 and "error" in f["flags"]
    assert "error in ne -- nlteopt" in open(os.path.join(f["result_dir"], "pnlte_tail.log")).read()
    assert g["status"] == "ok" and g["niter"] == 63 and g["convergence"]["converged"]
    led = sorted(glob.glob(os.path.join(M424_RUN, "results", "*", "part_*.idx")))      # all tags (~1 s)
    prod = [ln.split() for p in led for ln in open(p) if ln.startswith("571348 ")]
    if prod:
        for mine, pr in ((f, [x for x in prod if x[1] == "37298.910"]), (g, [x for x in prod if x[1] == "37299.910"])):
            assert pr, "production ledger has no record for " + mine["teff"]
            got = open(os.path.join(mine["result_dir"], "meta.txt")).read().split()
            assert got[:5] == pr[0][:5], (got, pr[0])         # idx teff status niter T_tau23 as in production


@pytest.mark.fastwind
@pytest.mark.slow
def test_real_rerun_formal_imu(real):
    from ppmpy.synspec import fwresults
    assert real["st"].has_imu is False and real["st_imu"].has_imu is True
    r = mo.rerun_formal(real["st_imu"], real["ref"]["result_dir"], _data("FORMAL_INPUT_He3"),
                        run_root=os.path.join(real["work"], "imu"))
    assert r["status"] == "ok", r
    lam, p, ic, il, cnt = fwresults.read_out_imu(os.path.join(r["model_dir"], "OUT_IMU.HEI4922_VTV010"), counts=True)
    assert lam.size == 161 and cnt["nray"] == p.size == ic.shape[1]
    for n in LINES:
        assert open(os.path.join(r["model_dir"], "OUT.{}_VTV010".format(n)), "rb").read() == open(
            os.path.join(real["ref"]["result_dir"], "OUT.{}_VTV010".format(n)), "rb").read()


@pytest.mark.fastwind
@pytest.mark.slow
def test_real_formal_killed_is_failed(real):
    """
    The real pformalsol (~0.5 s) killed partway leaves the current line's OUT file empty under its final name; never
    'ok'. ifort's runtime catches SIGTERM ('forrtl: error (78): process killed (SIGTERM)') and exits 1, so the exit
    status is 1, not -15.
    """
    hits = 0
    for i, t in enumerate((0.1, 0.2, 0.3, 0.45)):
        r = mo.rerun_formal(real["st_imu"], real["ref"]["result_dir"], _data("FORMAL_INPUT_He3"),
                            run_root=os.path.join(real["work"], "killed{}".format(i)), kind="OUT", timeout=t, grace=0.2)
        if r["timed_out"]:
            hits += 1
            assert r["status"] == "failed" and r["returncode"] != 0 and r["problems"], r
        else:
            assert r["status"] == "ok" and r["problems"] == {}, r
    assert hits >= 1
    assert mo.active_groups() == {}
