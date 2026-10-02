"""Tests of ppmpy.synspec.validate.

* Report objects (CheckResult, ValidationReport: verdicts, table, JSON / npz round trips).
* Legacy equivalence on a synthetic run on the M424 grid (3 lines, 8 lines of sight): the frozen
  fw_disc_dumps_validate.py and fw_disc_holdout.py are run as scripts on toy files (fw_disc.RUN / SAMPLES / DISC_DUMPS
  patched to tmp; Pool -> builtin map) and their validate.npz / holdout.npz are compared bit for bit with the port.
  The frozen fw_disc_validate.py likewise for exact_continuous and NearestBin.
* A general synthetic run (2 lines, 3 lines of sight, a small grid) on which every check passes on correct inputs and
  the faults of the module's sensitivity table are (or, by design, are not) detected (V2 through faulty factories
  and a replaced DiscFlux); a second toy with T_eff' within the bins for faulty interpolation weights; the sign
  conventions checked analytically.
* Smoothed and corrected integrators (V4, V5), nodes that are not the integrator's, empty inputs, NaN inputs (V1, EW
  conservation, LPV ratios).
* Brute force (naive per-point loop, DiscFlux, parallel paths), EW conservation, T_eff' ranges, run_validation and
  the LPV comparison (also of the recorded M424 values).
* M424 regressions (marker m424; slow where noted): validate.npz (V1 flux, V3-V6; V1 imu with the frozen DiscImu),
  holdout.npz, teff_ranges_all_dumps.npy, the brute force against the stored flux products, EW conservation and the
  LPV residual rms of the production time series.

PPMPY_SYNSPEC_M424_SAMPLES (default /scratch/ppathak/fastwind_sphere/samples_r4050_N1236544) locates the per-dump
samples, PPMPY_SYNSPEC_M424_R3 (default the project's analysis/r3_out.npz) the 2026-09-29 review output (optional).
"""
import json
import os
import resource
import sys
import time
import types
import warnings

import numpy as np
import pytest

import conftest
from conftest import m424_path
from ppmpy.synspec import disc
from ppmpy.synspec import library as lb
from ppmpy.synspec import sphere as sph
from ppmpy.synspec import validate as va
from ppmpy.synspec.conventions import C_KMS, los_array
from ppmpy.synspec.io import npz_member_memmap, read_meta
from ppmpy.synspec.spectral import LineSet, VelocityGrid

LREF = np.array([4026.22, 4199.90, 4921.93])
LINES = ["HEI4026", "HEII4200", "HEI4922"]
LINESET = LineSet(LINES, LREF)
LEGACY = os.environ.get("PPMPY_SYNSPEC_M424_PROJECT",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy"))
SAMPLES_M424 = os.environ.get("PPMPY_SYNSPEC_M424_SAMPLES", "/scratch/ppathak/fastwind_sphere/samples_r4050_N1236544")
R3_OUT = os.environ.get("PPMPY_SYNSPEC_M424_R3",
                        "/home/ppathak/stellar-atmosphere-KU-Leuven/project/analysis/r3_out.npz")


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _legacy_fw_disc():
    """The frozen fw_disc.py (skips when it is not there); stays in sys.modules for the scripts' 'import fw_disc'."""
    if not os.path.exists(os.path.join(LEGACY, "fw_disc.py")):
        pytest.skip("legacy fw_disc.py not available in {}".format(LEGACY))
    sys.path.insert(0, LEGACY)
    try:
        import fw_disc as fd
    finally:
        sys.path.remove(LEGACY)
    return fd


class _FakePool:
    """multiprocessing.Pool stand-in for the legacy scripts: `with Pool(n) as pool: pool.map(...)` -> builtin map."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return types.SimpleNamespace(map=map)

    def __exit__(self, *exc):
        return False


def _run_legacy_script(name, argv, replace=(), ns=None):
    """Execute a frozen legacy script as a whole (its argparse sees argv), after text replacements."""
    path = os.path.join(LEGACY, name)
    if not os.path.exists(path):
        pytest.skip("legacy {} not available in {}".format(name, LEGACY))
    src = open(path).read()
    for a, b in replace:
        assert a in src, (name, a)
        src = src.replace(a, b)
    old = sys.argv
    sys.argv = [name] + [str(x) for x in argv]
    try:
        g = dict(ns or {}, __name__="legacy_" + name[:-3])
        exec(compile(src, path, "exec"), g)
    finally:
        sys.argv = old
    return g


def _samples_m424(dump):
    p = os.path.join(SAMPLES_M424, "d{:04d}.npz".format(dump))
    if not os.path.exists(p):
        pytest.skip("M424 product not available: {}".format(p))
    return SAMPLES_M424


def _write_sample(d, dump, teff, rng, vsig=40.0, nfast=0):
    N = teff.size
    vel = {k: (vsig * rng.standard_normal(N)).astype(np.float32) for k in ("ur", "uth", "uph")}
    if nfast:
        vel["ur"][:nfast] = 600.0                                # beyond 400 km/s: clipped by DiscFlux
    np.savez(os.path.join(d, "d{:04d}.npz".format(dump)), teff=np.asarray(teff, np.float32), t_s=2835.0 * dump,
             **vel)


# ----------------------------------------------------------------------------------------------
# report objects
# ----------------------------------------------------------------------------------------------
def test_no_heavy_imports():
    import subprocess
    code = ("import sys; sys.path.insert(0, {!r}); import ppmpy.synspec.validate; "
            "bad = [m for m in ('matplotlib', 'ppmpy.ppm', 'h5py') if m in sys.modules]; print(bad)").format(
                conftest.ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"


def test_checkresult_verdicts():
    c = va.CheckResult("a", 1e-6, 2e-6)
    assert c.passed is True and c.status == "PASS"
    c = va.CheckResult("a", 3e-6, 2e-6)
    assert c.passed is False and c.status == "FAIL"
    c = va.CheckResult("a", np.nan, 2e-6)
    assert c.passed is False                                       # NaN never passes
    c = va.CheckResult("a", 5.0, None)
    assert c.passed is None and c.status == "info"
    c = va.CheckResult("a", 5.0, 1.0, passed=True)                 # explicit verdict kept
    assert c.passed is True
    with pytest.raises(ValueError):
        va.CheckResult("a", 1.0, unit="km/s")
    with pytest.raises(ValueError, match="NaN"):
        va.CheckResult("a", 1.0, np.nan)                           # informational is None, not NaN
    c = va.CheckResult("a", np.inf, np.inf)
    assert c.passed is True and va.CheckResult("a", np.nan, np.inf).passed is False
    assert "PASS" in repr(va.CheckResult("x", 1e-7, 1e-6))
    # an empty report does not pass; one with only informational checks does
    assert not va.ValidationReport([]).passed()
    assert va.ValidationReport([va.CheckResult("i", 1.0)]).passed()


def _small_report():
    a = va.CheckResult("V1_flux", 3e-6, 5e-6, details=dict(per_line=np.array([1e-6, 3e-6]), lines=["a", "b"]))
    b = va.CheckResult("V6", 8e-5, 5e-5, details=dict(per_line=np.array([8e-5, 1e-6])))
    c = va.CheckResult("V3_leaveout", 1e-4, None, details=dict(per_line=np.array([np.nan, 1e-4])))
    d = va.CheckResult("ew", 1e-8, 1e-6, unit="relative")
    r1 = va.ValidationReport([a, b], meta=dict(x=1),
                             arrays=dict(V1_flux_F=np.array([1e-6, 3e-6]), lines=np.array(["a", "b"])))
    r2 = va.ValidationReport([c, d], arrays=dict(V3_n=np.array([[1.0, 2.0]]), V3_dumps=np.array([7], np.int64),
                                                   lines=np.array(["a", "b"]), seed=np.int64(11)))
    return va.ValidationReport.merge(r1, r2, meta=dict(label="toy"))


def test_report_table_and_verdicts():
    r = _small_report()
    assert r.names() == ["V1_flux", "V6", "V3_leaveout", "ew"]
    assert not r.passed() and [c.name for c in r.failed()] == ["V6"]
    assert "V6" in r and r["V1_flux"].value == 3e-6
    t = r.table()
    assert "FAIL" in t and "info" in t and "4 checks: 2 passed, 1 failed, 1 informational" in t
    assert "1.00e-06 / 3.00e-06" in t
    with pytest.raises(ValueError):
        va.ValidationReport.merge(r, r)                            # repeated names
    with pytest.raises(KeyError):
        r["nope"]


def test_report_json_npz_roundtrip(tmp_path):
    r = _small_report()
    r.checks.append(va.CheckResult("inf_tol", 1.0, np.inf, details=dict(per_line=[1.0, np.nan])))
    with pytest.warns(UserWarning) as rec:                        # lpv_ratio with NaN (the V3_leaveout per_line)
        va.compare_lpv(r, dict(rms=np.array([1e-4, 1e-4])))
    assert {str(w.message).split(":")[0] for w in rec} == {"V6", "inf_tol"}               # NaN ratio: warned
    assert r["V3_leaveout"].details["lpv_warn"] and r["V3_leaveout"].details["lpv_nonfinite"] == 1
    t0 = r.table()
    p = r.to_json(str(tmp_path / "rep.json"))
    q = va.ValidationReport.from_json(p)
    assert q.names() == r.names()
    for a, b in zip(q, r):
        assert (a.value == b.value or (np.isnan(a.value) and np.isnan(b.value))) and a.tolerance == b.tolerance
        assert a.passed == b.passed and a.unit == b.unit
    assert q["inf_tol"].tolerance == np.inf and q["inf_tol"].passed is True
    np.testing.assert_array_equal(q["V3_leaveout"].details["per_line"], [np.nan, 1e-4])      # None -> NaN
    np.testing.assert_allclose(q["V1_flux"].details["lpv_ratio"], [0.01, 0.03])
    for k, v in r.arrays.items():
        np.testing.assert_array_equal(q.arrays[k], v)
        assert q.arrays[k].dtype == np.asarray(v).dtype
    json.load(open(p))                                             # strict JSON (no NaN tokens)
    assert q.table() == t0                                         # printable after the round trip, same text
    p = r.to_npz(str(tmp_path / "rep.npz"))
    z = np.load(p)
    assert list(z["check_names"]) == r.names()
    np.testing.assert_array_equal(z["check_passed"], [1, 0, -1, 1, 1])
    assert np.isnan(z["check_tolerances"][2]) and z["check_tolerances"][4] == np.inf
    np.testing.assert_array_equal(z["V3_dumps"], [7])
    assert read_meta(z)["kind"] == "synspec.validation"
    q = va.ValidationReport.from_npz(p)
    assert q.names() == r.names() and q["V6"].passed is False
    np.testing.assert_array_equal(q["V1_flux"].details["per_line"], [1e-6, 3e-6])
    assert q.table() == t0
    # a caller's meta: the checks are still written (and read back), its own entries kept
    p = r.to_npz(str(tmp_path / "rep2.npz"), meta=dict(kind="mine", params=dict(a=1), checks="stale"))
    m = read_meta(p)
    assert m["kind"] == "mine" and m["params"] == dict(a=1)
    q = va.ValidationReport.from_npz(p)
    assert q.names() == r.names() and q.meta == dict(a=1) and q.table() == t0
    with pytest.raises(ValueError):
        va.ValidationReport([], arrays=dict(check_names=np.zeros(1))).to_npz(str(tmp_path / "x.npz"))


def test_tolerance_tables():
    """The defaults cover the recorded M424 values with a margin; the per-line records agree with the maxima."""
    R, A = va.RECORDED_M424, va.RECORDED_M424_ARRAYS
    T = va.DEFAULT_TOLERANCES
    for name, tol in (("V1_flux", "V1"), ("V1_imu", "V1"), ("V1_flux_dEW", "V1_dEW"), ("V2_holdout", "V2"),
                      ("V2_insample", "V2"), ("V2_holdout_dEW", "V2_dEW"), ("V3_extrap", "V3"), ("V4", "V4"),
                      ("V5", "V5"), ("V6", "V6"), ("ew_flux", "ew"), ("ew_imu", "ew")):
        assert R[name] < T[tol] <= 7 * R[name], name
    # the brute force vs stored products: 8 x the float32 rounding bound 2^-25 (what M424 gives), far below the
    # 2026-09-29 review's 7.6e-6 and below V1 and V4 (it must see misaligned lines of sight or points)
    assert R["brute_stored"] <= 2.0 ** -25 < T["brute"] <= 10 * R["brute_stored"]
    assert T["brute"] < R["V4"] < R["V1_flux"] < R["brute_review"]
    assert R["V1_flux"] == max(max(A["V1_flux_F"]), max(A["V1_flux_F0"]))
    assert R["V2_holdout"] == max(max(A[k]) for k in A if k.startswith("holdout_") and not k.endswith("dEW"))
    assert R["V6"] == np.max(A["V6"]) and R["V4"] == np.max(A["V4"]) and R["V3_extrap"] == np.max(A["V3_extrap"])
    assert R["ew_flux"] == max(A["ew_flux"]) and R["ew_imu"] == max(A["ew_imu"])          # full values, not rounded
    assert R["ew_flux"] == 1.4861438631683864e-07


def test_recorded_lpv_comparison():
    """compare_lpv on the recorded M424 values (V1, V2 hold-out and in-sample, V3-V6) with the recorded imu residual
    rms and flux EW rms: V6 and the lambda4026 EW checks V1_flux_dEW (5.7 %), V2_holdout_dEW (18 %) and V2_insample_dEW
    (12 %) are warned about, the V3 leave-out (35 %, informational) is flagged; V2's profiles reach 2.9 % (hold-out)
    and 2.4 % (in-sample), every other profile check stays below 1.2 % (module notes)."""
    A = va.RECORDED_M424_ARRAYS

    def per_line(*keys):
        return np.max([np.max(np.atleast_2d(A[k]), axis=0) for k in keys], axis=0)

    def chk(name, keys, unit="continuum", tol=1.0):
        return va.CheckResult(name, per_line(*keys).max(), tol, unit=unit, details=dict(per_line=per_line(*keys)))

    halves = ("A", "B")
    checks = [chk("V1_flux", ("V1_flux_F", "V1_flux_F0")), chk("V1_flux_dEW", ("V1_flux_dEW",), "A"),
              chk("V1_imu", ("V1_imu_F", "V1_imu_F0")),
              chk("V2_holdout", ["holdout_{}_{}".format(X, q) for X in halves for q in ("F", "F0")]),
              chk("V2_holdout_dEW", ["holdout_{}_dEW".format(X) for X in halves], "A"),
              chk("V2_insample", ["insample_{}_{}".format(X, q) for X in halves for q in ("F", "F0")]),
              chk("V2_insample_dEW", ["insample_{}_dEW".format(X) for X in halves], "A"),
              chk("V3_extrap", ("V3_extrap",)), chk("V3_leaveout", ("V3_leaveout",), tol=None), chk("V4", ("V4",)),
              chk("V5", ("V5_nmin1", "V5_nmin5", "V5_nmin100")), chk("V6", ("V6",))]
    r = va.ValidationReport(checks)
    lpv = dict(rms=np.array(A["lpv_rms_imu"]), ew_rms=np.array(A["lpv_ew_rms_flux"]))
    with pytest.warns(UserWarning) as rec:
        va.compare_lpv(r, lpv)
    warned = {str(w.message).split(":")[0] for w in rec}
    assert warned == {"V6", "V1_flux_dEW", "V2_holdout_dEW", "V2_insample_dEW"}
    assert {c.name for c in r.warnings()} == warned | {"V3_leaveout"}
    worst = {c.name: float(np.max(c.details["lpv_ratio"])) for c in r}
    assert abs(worst["V2_holdout"] - 0.0286) < 5e-4 and abs(worst["V2_insample"] - 0.0242) < 5e-4
    assert abs(worst["V1_flux_dEW"] - 0.0572) < 5e-4 and abs(worst["V2_holdout_dEW"] - 0.1807) < 5e-4
    assert abs(worst["V2_insample_dEW"] - 0.1243) < 5e-4 and abs(worst["V6"] - 0.1131) < 5e-4
    for name in ("V1_flux", "V1_imu", "V3_extrap", "V4", "V5"):
        assert worst[name] < 0.012, name
    # with the imu EW rms the same checks warn
    r2 = va.ValidationReport([va.CheckResult(c.name, c.value, c.tolerance, unit=c.unit,
                                             details=dict(per_line=c.details["per_line"])) for c in checks])
    with pytest.warns(UserWarning):
        va.compare_lpv(r2, dict(rms=np.array(A["lpv_rms_imu"]), ew_rms=np.array(A["lpv_ew_rms_imu"])))
    assert {c.name for c in r2.warnings()} == warned | {"V3_leaveout"}


# ----------------------------------------------------------------------------------------------
# legacy equivalence on the M424 grid
# ----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def legacy_toy(tmp_path_factory):
    """A synthetic per-point run on the M424 grid in the legacy file layout: RUN/{profiles, points, library_dT10,
    disc_los8}.npz (library and exact sums from the synspec code, block 500, stride 2), SAMPLES/d3200-d3204.npz
    (d3200: the models' T_eff' in float32; later dumps: T_eff' scattered by 150 K, beyond the nodes at both ends, a few
    shifts beyond 400 km/s), a T_eff' range table for V3."""
    root = tmp_path_factory.mktemp("legacy_toy")
    run, smp, dd = (str(root / x) for x in ("run", "samples", "disc"))
    for d in (run, smp, dd):
        os.makedirs(d)
    rng = np.random.default_rng(5)
    N, nrow, span = 3000, 161, 3000.0
    theta, phi = sph.fibonacci_sphere(N)
    teff = 38230.0 + 300.0 * rng.standard_normal(N)
    teff[:3] = [37000.0, 39500.0, 38230.0]                        # sparse tails: empty bins, merged nodes
    x = (teff - 38230.0) / 300.0
    yn = np.linspace(-span, span, nrow)
    jit = rng.uniform(-0.4, 0.4, N)
    lam = (LREF[None, :, None] * np.exp((yn[None, None, :] + jit[:, None, None]) / C_KMS)).astype(np.float32)
    fnorm = np.empty((N, 3, nrow), np.float32)
    fcont = np.empty((N, 3, nrow), np.float32)
    saw = 0.01 * np.sign(np.sin(teff / 40.0))                     # a sawtooth in T_eff' (cf. the M424 EW branches)
    for j, sig in enumerate((60.0, 140.0, 40.0)):
        depth = (0.4 + 0.05 * (j + 1) * x + saw)[:, None] * np.exp(
            -0.5 * ((yn[None, :] - 2.0 * x[:, None] - j) / (sig * (1.0 + 0.03 * x[:, None]))) ** 2)
        fnorm[:, j] = 1.0 - depth
        fcont[:, j] = ((teff / 38230.0) ** 4 * (1.0 + 0.05 * j))[:, None] * (1.0 + 1e-4 * yn / span)[None, :]
    np.savez(os.path.join(run, "profiles.npz"), idx=np.arange(N, dtype=np.int32), teff=teff, theta=theta, phi=phi,
             lam=lam, fnorm=fnorm, fcont=fcont, lines=np.array(LINES))
    np.savez(os.path.join(run, "points.npz"), theta=theta, phi=phi, teff=teff)
    lib = lb.FluxLibrary.build(teff, lam, fnorm, fcont[:, :, 0], VelocityGrid(), LREF, block=500, stride=2)
    lib.save(os.path.join(run, "library_dT10.npz"))
    _write_sample(smp, 3200, teff, rng, nfast=2)
    for d in range(3201, 3205):
        t = teff + 150.0 * rng.standard_normal(N)
        t[:2] = [36000.0 - 100.0 * (d - 3200), 39900.0 + 10.0 * d % 7]
        _write_sample(smp, d, t, rng)
    ex = disc.integrate_exact_stream(os.path.join(run, "profiles.npz"), os.path.join(smp, "d3200.npz"), LINESET,
                                     checks=False, block=500, stride=2)
    disc.save_disc_los(os.path.join(run, "disc_los8.npz"), ex)
    rg = va.teff_ranges(smp, range(3200, 3205), trange=(teff.min(), teff.max()))
    np.save(os.path.join(str(root), "ranges.npy"), rg)
    return dict(root=str(root), run=run, samples=smp, disc=dd, ranges=os.path.join(str(root), "ranges.npy"),
                theta=theta, phi=phi, teff=teff)


def _patch_fd(monkeypatch, fd, toy):
    monkeypatch.setattr(fd, "RUN", toy["run"])
    monkeypatch.setattr(fd, "SAMPLES", toy["samples"])
    monkeypatch.setattr(fd, "DISC_DUMPS", toy["disc"])


def test_matches_legacy_validate(legacy_toy, monkeypatch):
    """V1 (flux), V3 (dumps from the ranges table), V4, V5, V6 equal the arrays of the frozen fw_disc_dumps_validate.py
    (run as a script, --no-imu) bit for bit."""
    fd = _legacy_fw_disc()
    _patch_fd(monkeypatch, fd, legacy_toy)
    dl = [3201, 3203]
    _run_legacy_script("fw_disc_dumps_validate.py", ["--no-imu", "--dumps", ",".join(map(str, dl)), "--nsub", 400,
                                                     "--ranges", legacy_toy["ranges"]])
    ref = np.load(os.path.join(legacy_toy["disc"], "validate.npz"))
    T = legacy_toy
    lib = lb.FluxLibrary.load(os.path.join(T["run"], "library_dT10.npz"))
    nodes = lb.lib_nodes(lib, nmin=20)
    FX = disc.DiscFlux(nodes)
    MU, TN, PN = sph.project_los(T["theta"], T["phi"], "thompson2024")
    v3 = va.v3_select(np.load(T["ranges"]))
    rep = va.ValidationReport.merge(
        va.v1_exact(FX, os.path.join(T["run"], "disc_los8.npz"), os.path.join(T["samples"], "d3200.npz"), MU, TN, PN,
                    lref=LINESET),
        va.v4_nearest(FX, lib, T["samples"], dl, MU, TN, PN),
        va.v5_node_merging(FX, lib, T["samples"], dl[0], MU, TN, PN),
        va.v3_tails(FX, T["samples"], v3, MU, TN, PN),
        va.v6_rounding(FX, nodes, T["samples"], dl, MU, TN, PN, nsub=400))
    keys = [k for k in ref.files if k not in ("lines", "nmin")]
    assert set(keys) <= set(rep.arrays)
    for k in keys:
        np.testing.assert_array_equal(rep.arrays[k], ref[k], err_msg=k)
        assert rep.arrays[k].dtype == ref[k].dtype, k
    assert ref["V3_n"].sum() > 0                                   # the tails are exercised
    assert np.all(ref["V6"] > 0) and np.all(ref["V4"] > 0)


@pytest.mark.parametrize("nproc,method,library", [(1, None, "file"), (1, None, None), (2, "fork", None),
                                                  (2, "spawn", "file")])
def test_matches_legacy_holdout(legacy_toy, monkeypatch, nproc, method, library):
    """v2_holdout equals the holdout.npz of the frozen fw_disc_holdout.py (run as a script with --nproc 2 --block 500,
    Pool -> map) bit for bit: serial and with 2 workers (fork, spawn), with the library file or the library built in
    the pass. ew_jacobian=True: the frozen fw_disc.diagnostics has the factor lambda/lref (the M424 file does not)."""
    fd = _legacy_fw_disc()
    _patch_fd(monkeypatch, fd, legacy_toy)
    out = os.path.join(legacy_toy["disc"], "holdout.npz")
    if not os.path.exists(out):
        _run_legacy_script("fw_disc_holdout.py", ["--nproc", 2, "--block", 500],
                           replace=[("from multiprocessing import Pool\n", "")], ns=dict(Pool=_FakePool))
    ref = np.load(out)
    T = legacy_toy
    lib = os.path.join(T["run"], "library_dT10.npz") if library == "file" else None
    rep = va.v2_holdout(os.path.join(T["run"], "profiles.npz"), os.path.join(T["samples"], "d3200.npz"), None, None,
                        "thompson2024", VelocityGrid(), LINESET, stride=2, block=500, nproc=nproc, start_method=method,
                        library=lib, ew_jacobian=True, rows=97)
    assert list(rep.arrays) == list(ref.files)                     # same members, same order
    for k in ref.files:
        np.testing.assert_array_equal(rep.arrays[k], ref[k], err_msg=k)
        assert np.asarray(rep.arrays[k]).dtype == ref[k].dtype, k
    if library is None:
        L0 = lb.FluxLibrary.load(os.path.join(T["run"], "library_dT10.npz"))
        for k in ("edges", "tmean", "count", "prof", "fc"):
            np.testing.assert_array_equal(rep.data["libraries"]["all"][k], L0[k], err_msg=k)
    # the toy's coarse library (sawtooth, sparse tails) deviates more than M424's: here just finite and non-zero
    assert all(np.isfinite(c.value) and c.value > 0 for c in rep)


def test_matches_legacy_fw_disc_validate(legacy_toy, monkeypatch):
    """The first library-vs-exact check: the frozen fw_disc_validate.py (run as a script, --ndir 2) on the toy;
    exact_continuous (continuous shifts) equals its fw_disc.integrate_exact, and NearestBin through v1_exact (with the
    models' own T_eff' in the sample, as the script binned them) its fw_disc.integrate_lib, bit for bit; the
    difference includes the 1 km/s rounding, so it exceeds that against the rounded exact sums (disc_los8.npz)."""
    fd = _legacy_fw_disc()
    _patch_fd(monkeypatch, fd, legacy_toy)
    T = legacy_toy
    prof = os.path.join(T["run"], "profiles.npz")
    libp = os.path.join(T["run"], "library_dT10.npz")
    st = os.stat(prof)
    os.utime(libp, (st.st_atime + 10, st.st_mtime + 10))       # the cache must look newer (fd.library would rebuild)
    g = _run_legacy_script("fw_disc_validate.py", ["--ndir", 2],
                           replace=[("fd.library(dT=a.dT)", "fd.library(fd.RUN, dT=a.dT)"),
                                    ("fd.load_run()", "fd.load_run(fd.RUN)")])
    assert g["k"] == 1
    MU, TN, PN = sph.project_los(T["theta"], T["phi"], fd.directions(2), method="matvec")
    smp = os.path.join(T["samples"], "d3200.npz")
    ex = va.exact_continuous(prof, smp, MU, TN, PN, LINESET)
    np.testing.assert_array_equal(ex["F"][1], g["Fx"])
    s = np.load(smp)
    models = dict(teff=T["teff"], ur=s["ur"], uth=s["uth"], uph=s["uph"])           # bins of the models' T_eff'
    lib = lb.FluxLibrary.load(libp)
    rep = va.v1_exact(va.NearestBin(lib, VelocityGrid()), ex, models, MU, TN, PN, lref=LINESET, label="nearest",
                      tolerance=1.0, dew_tolerance=1.0)
    np.testing.assert_array_equal(rep.data["V1_nearest"]["F"][1], g["Fl"])
    np.testing.assert_array_equal(rep["V1_nearest"].details["F"], np.abs(rep.data["V1_nearest"]["F"] - ex["F"])
                                  .max(axis=(0, 2)))
    # without velocities the continuous and rounded references agree (no shifts; the interpolation offsets of other
    # blocks change the last bits, ~1e-11)
    rounded = np.load(os.path.join(T["run"], "disc_los8.npz"))
    exr = va.exact_continuous(prof, smp, *sph.project_los(T["theta"], T["phi"], "thompson2024", method="matvec"),
                              LINESET)
    np.testing.assert_allclose(exr["F0"], rounded["F0"], rtol=0, atol=1e-10)
    assert np.abs(exr["F"] - rounded["F"]).max() > 1e-5                                   # the rounding


# ----------------------------------------------------------------------------------------------
# a general synthetic run and its faults
# ----------------------------------------------------------------------------------------------
GEN_GRID = VelocityGrid(dv=1.0, vmax=600.0, vshift=150.0)
GEN_LS = LineSet(["L1", "L2"], [4471.5, 5875.6])
GEN_LOS = los_array([[1.0, 0.2, 0.1], [-0.3, 1.0, 0.5], [0.2, -0.4, -1.0]])
# tolerances of the toy (its own scales: float32 library 1e-8, rounding of 1 km/s steps on 35 km/s lines 1e-4)
GEN_TOL = dict(V1=1e-6, V1_dEW=1e-6, V2=1e-6, V2_dEW=1e-6, V3=1e-3, V4=1e-6, V5=1e-6, V6=4e-4, brute=2.5e-7,
               brute_f64=1e-10, ew=1e-6)


class _FlipV:
    """An integrator with the sign of v flipped (a convention error inside the pipeline)."""

    def __init__(self, integ):
        self.i, self.t, self.grid, self.nl = integ, integ.t, integ.grid, integ.nl

    def pairs(self, teff):
        return self.i.pairs(teff)

    def __call__(self, mu, v, k0, k1, a, novel=True):
        return self.i(mu, -v, k0, k1, a, novel=novel)


class _MixF:
    """F of one integrator with F0 (and the rest) of another: profiles with and without velocities inconsistent."""

    def __init__(self, a, b):
        self.a, self.b, self.t, self.grid, self.nl = a, b, b.t, b.grid, b.nl

    def pairs(self, teff):
        return self.b.pairs(teff)

    def __call__(self, mu, v, k0, k1, a, novel=True):
        out = self.b(mu, v, k0, k1, a, novel=novel)
        return (self.a(mu, v, k0, k1, a, novel=False)[0],) + tuple(out[1:])


class _WeightFault:
    """An integrator whose T_eff' interpolation weights are wrong: 'swap' gives (k0, k1, 1 - a), 'square' (k0, k1,
    a^2) (the same at a = 0 and 1)."""

    def __init__(self, integ, kind):
        self.i, self.kind, self.t, self.grid, self.nl = integ, kind, integ.t, integ.grid, integ.nl

    def pairs(self, teff):
        k0, k1, a = self.i.pairs(teff)
        return k0, k1, (1.0 - a if self.kind == "swap" else a ** 2)

    def __call__(self, mu, v, k0, k1, a, novel=True):
        return self.i(mu, v, k0, k1, a, novel=novel)


class _FlipDiscFlux(disc.DiscFlux):
    """DiscFlux with the sign of v flipped (to replace validate.DiscFlux, which v2_holdout builds by default)."""

    def __call__(self, mu, v, k0, k1, a, novel=True):
        return super().__call__(mu, -v, k0, k1, a, novel=novel)


def _offset_nodes(nodes, win, eps=1e-5):
    return dict(t=nodes.t, count=nodes.count, prof=nodes["prof"] + eps * win, fc=nodes.fc)


def _drop_node(nodes):
    keep = np.arange(nodes.nn) != nodes.nn // 2
    return dict(t=nodes.t[keep], count=nodes.count[keep], prof=nodes["prof"][keep], fc=nodes.fc[keep])


def _fault_factory(fault, win, grid=None):
    """v2_holdout factory (FluxLibrary -> integrator) with the fault of the sensitivity table."""
    grid = GEN_GRID if grid is None else grid

    def make(L):
        nodes = lb.lib_nodes(L, nmin=20)
        good = disc.DiscFlux(nodes, grid)
        if fault == "offset":
            return disc.DiscFlux(_offset_nodes(nodes, win), grid)
        if fault == "missing":
            return disc.DiscFlux(_drop_node(nodes), grid)
        if fault == "vflip":
            return _FlipV(good)
        if fault in ("swap", "square"):
            return _WeightFault(good, fault)
        if fault == "f_f0":
            return _MixF(disc.DiscFlux(_offset_nodes(nodes, win), grid), good)
        return good
    return make


def _gen_toy(N=4000, nbin=24, seed=3, jitter=0.02):
    """Models in 24 T_eff' bins of 10 K, all models of a bin identical (T_eff' at the bin centre, so the library's own
    dump needs no interpolation: V1 and V2 are exact up to the float32 library), profiles changing smoothly with
    T_eff' plus a random depth jitter per bin (so a missing node shows). Dumps 1, 2: T_eff' on the node values,
    random velocities; dump 3: continuous T_eff' reaching 15 K beyond the nodes."""
    rng = np.random.default_rng(seed)
    th, ph = sph.fibonacci_sphere(N)
    Tb = 38005.0 + 10.0 * np.arange(nbin)
    b = rng.integers(0, nbin, N)
    teff = Tb[b]
    x = (np.arange(nbin) - (nbin - 1) / 2) / ((nbin - 1) / 2)
    yn = np.linspace(-800.0, 800.0, 321)
    nl = len(GEN_LS)
    A = np.array([0.35 + 0.05 * x + jitter * rng.standard_normal(nbin),
                  0.25 - 0.04 * x + jitter * rng.standard_normal(nbin)])
    c = np.array([2.0 * x, -1.5 * x + 1.0])
    sig = np.array([35.0 * (1 + 0.05 * x), 50.0 * (1 - 0.04 * x)])
    lam = np.empty((N, nl, yn.size), np.float32)
    fn, fcont = np.empty_like(lam), np.empty_like(lam)
    for j in range(nl):
        lam[:, j] = (GEN_LS.lref[j] * np.exp(yn / C_KMS))[None, :].astype(np.float32)
        pb = 1.0 - A[j][:, None] * np.exp(-0.5 * ((yn[None, :] - c[j][:, None]) / sig[j][:, None]) ** 2)
        fn[:, j] = pb[b].astype(np.float32)
        fcont[:, j] = ((teff / 38000.0) ** 4 * (1 + 0.05 * j))[:, None].astype(np.float32)
    prof = dict(lam=lam, fnorm=fn, fcont=fcont, teff=teff, theta=th, phi=ph, lines=np.array(GEN_LS.names),
                idx=np.arange(N))
    samples = {}
    for d in range(4):
        if d == 0:
            t = teff
        elif d < 3:
            t = Tb[rng.integers(0, nbin, N)]
        else:
            t = rng.uniform(Tb[0] - 15.0, Tb[-1] + 15.0, N)
        samples[d] = dict(teff=t.astype(np.float32), t_s=float(d),
                          **{k: (25.0 * rng.standard_normal(N)).astype(np.float32) for k in ("ur", "uth", "uph")})
    lib = lb.FluxLibrary.build(teff, lam, fn, fcont[:, :, 0], GEN_GRID, GEN_LS, block=1000, stride=2)
    nodes = lb.lib_nodes(lib, nmin=20)
    integ = disc.DiscFlux(nodes, GEN_GRID)
    ex = disc.integrate_exact_stream(prof, samples[0], GEN_LS, los=GEN_LOS, grid=GEN_GRID, checks=False, block=1000,
                                     stride=2)
    proj = sph.project_los(th, ph, GEN_LOS)
    win = (np.abs(GEN_GRID.y) <= GEN_GRID.vmax - GEN_GRID.vshift).astype(float)
    # the faulty integrators and the nodes each was built from (what a run would pass to V6 and the brute force)
    fnodes = dict(offset=_offset_nodes(nodes, win), missing=_drop_node(nodes))
    faulty = dict(offset=disc.DiscFlux(fnodes["offset"], GEN_GRID), missing=disc.DiscFlux(fnodes["missing"], GEN_GRID),
                  vflip=_FlipV(integ), swap=_WeightFault(integ, "swap"), square=_WeightFault(integ, "square"))
    faulty["f_f0"] = _MixF(faulty["offset"], integ)
    stored = {}
    for d in (1, 2):
        F, F0 = va._run(integ, va._sample_dict(samples[d]), *proj)
        stored[d] = (F.astype(np.float32), F0.astype(np.float32))
    return dict(prof=prof, samples=samples, lib=lib, nodes=nodes, integ=integ, exact=ex, proj=proj, faulty=faulty,
                fnodes=fnodes, win=win, stored=stored)


@pytest.fixture(scope="module")
def gen():
    return _gen_toy()


def _permuted(samples, seed=9):
    out = {}
    for d, s in samples.items():
        p = np.random.default_rng(seed).permutation(s["teff"].size)
        out[d] = {k: (v[p] if np.ndim(v) else v) for k, v in s.items()}
    return out


def _check_values(g, integ, samples, proj, nodes=None):
    """Value of every check for a pipeline (integrator, its nodes, samples, projections); references (exact sums,
    library, stored products of the intact pipeline) fixed."""
    MU, TN, PN = proj
    nodes = g["nodes"] if nodes is None else nodes
    kw = dict(tolerances=GEN_TOL)
    out = {}
    out["V1"] = va.v1_exact(integ, g["exact"], samples[0], MU, TN, PN, lref=GEN_LS, **kw)["V1_flux"]
    out["V3"] = va.v3_tails(integ, samples, [3], MU, TN, PN, margin=50.0, **kw)["V3_extrap"]
    out["V4"] = va.v4_nearest(integ, g["lib"], samples, [1, 2], MU, TN, PN, **kw)["V4"]
    out["V5"] = va.v5_node_merging(integ, g["lib"], samples, 1, MU, TN, PN, **kw)["V5"]
    out["V6"] = va.v6_rounding(integ, nodes, samples, [1, 2], MU, TN, PN, nsub=1500, **kw)["V6"]
    rb = va.brute_force_check(nodes, samples, MU, TN, PN, GEN_GRID, dumps=[1], integ=integ,
                              stored=lambda d: g["stored"][d], **kw)
    out["brute"], out["brute_stored"] = rb["brute_vs_integrator"], rb["brute_vs_stored"]
    F, F0 = va._run(integ, va._sample_dict(samples[1]), MU, TN, PN)
    out["EW"] = va.ew_conservation(F, F0, GEN_GRID.y, GEN_LS, **kw)
    return out


# which checks see which fault ('x'); the module docstring's sensitivity table (V2: test_general_v2_faults)
FAULTS = {
    "offset": {"V1", "V4", "V5", "brute_stored"},
    "missing": {"V1", "V4", "V5", "brute_stored"},
    "vflip": {"V1", "V4", "V5", "V6", "brute"},
    "swap": {"V1", "V3", "V4", "V5", "V6", "brute"},
    "los_swap": {"V1", "brute_stored"},
    "permuted": {"V1", "brute_stored"},
    "f_f0": {"V1", "V4", "V5", "brute", "EW"},
}


def test_general_correct_inputs_pass(gen):
    vals = _check_values(gen, gen["integ"], gen["samples"], gen["proj"])
    for k, c in vals.items():
        assert c.passed, (k, c)
    assert vals["V1"].value < 1e-7 and vals["V4"].value < 1e-12 and vals["V5"].value == 0.0
    assert vals["brute"].value < 1e-13 and vals["brute_stored"].value < 1e-7


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_general_faults(gen, fault):
    integ, samples, proj = gen["integ"], gen["samples"], gen["proj"]
    if fault in gen["faulty"]:
        integ = gen["faulty"][fault]
    elif fault == "los_swap":
        proj = tuple(p[[1, 0, 2]] for p in proj)
    elif fault == "permuted":
        samples = _permuted(samples)
    vals = _check_values(gen, integ, samples, proj, nodes=gen["fnodes"].get(fault))
    for k, c in vals.items():
        if k in FAULTS[fault]:
            assert c.passed is False, (fault, k, c)
        else:
            assert c.passed is True, (fault, k, c)


def test_general_square_weights_invisible(gen):
    """The blind spot of this toy (every model at its bin centre, so a = 0 or 1 at the nodes): weights a^2 instead of
    a change nothing for V1; the spread toy (test_spread_interpolation_faults) sees them."""
    MU, TN, PN = gen["proj"]
    r = va.v1_exact(gen["faulty"]["square"], gen["exact"], gen["samples"][0], MU, TN, PN, lref=GEN_LS,
                    tolerances=GEN_TOL)
    assert r["V1_flux"].passed and r["V1_flux"].value < 1e-7


@pytest.mark.parametrize("fault", ["offset", "missing", "vflip", "swap", "f_f0"])
def test_general_v2_faults(gen, fault):
    """V2 with the faulty integrator as its factory: hold-out and in-sample fail (the V2 column of the sensitivity
    table); the intact factory passes with the same bits as the default."""
    g = gen
    kw = dict(block=1000, stride=2, tolerances=GEN_TOL)
    r = va.v2_holdout(g["prof"], g["samples"][0], None, None, GEN_LOS, GEN_GRID, GEN_LS,
                      factory=_fault_factory(fault, g["win"]), **kw)
    assert r["V2_holdout"].passed is False and r["V2_insample"].passed is False, r.table()


def test_general_v2_factory_and_discflux(gen, monkeypatch):
    """The intact factory reproduces the default bit for bit; DiscFlux replaced by a v-flipping subclass (the default
    factory looks it up at call time) fails V2."""
    g = gen
    kw = dict(block=1000, stride=2, tolerances=GEN_TOL)
    args = (g["prof"], g["samples"][0], None, None, GEN_LOS, GEN_GRID, GEN_LS)
    r0 = va.v2_holdout(*args, **kw)
    r1 = va.v2_holdout(*args, factory=_fault_factory(None, g["win"]), **kw)
    for k in r0.arrays:
        np.testing.assert_array_equal(r0.arrays[k], r1.arrays[k], err_msg=k)
    assert r0.meta["factory"] == "DiscFlux(lib_nodes(nmin=20))" and "make" in r1.meta["factory"]
    monkeypatch.setattr(va, "DiscFlux", _FlipDiscFlux)
    r = va.v2_holdout(*args, **kw)
    assert r["V2_holdout"].passed is False and r["V2_insample"].passed is False


def test_general_v3_tails(gen):
    """V3 sees T_eff' far beyond the nodes (clamping then matters); the leave-out comparison is informational and NaN
    (with no warning) when no point is inside the node range."""
    MU, TN, PN = gen["proj"]
    hot = {d: dict(s, teff=s["teff"] + 300.0) for d, s in gen["samples"].items()}
    r = va.v3_tails(gen["integ"], hot, [3], MU, TN, PN, margin=50.0, tolerances=GEN_TOL)
    assert r["V3_extrap"].passed is False and r["V3_extrap"].value > 1e-2
    assert r["V3_leaveout"].passed is None
    np.testing.assert_array_equal(r.arrays["V3_n"], [[0.0, float(MU.shape[1])]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = va.v3_tails(gen["integ"], gen["samples"], [3, 1], MU, TN, PN, margin=50.0)
    np.testing.assert_array_equal(r.arrays["V3_dumps"], [3, 1])
    assert r.arrays["V3_n"][1].sum() == 0 and r.arrays["V3_extrap"][1].max() == 0.0     # dump 1: inside the nodes


def test_general_v2_holdout(gen):
    """Hold-out on the toy: exact (all models of a bin are identical); permuted model T_eff' labels fail both parts;
    an offset in-sample library fails only the in-sample part."""
    g = gen
    kw = dict(block=1000, stride=2, tolerances=GEN_TOL)
    r = va.v2_holdout(g["prof"], g["samples"][0], None, None, GEN_LOS, GEN_GRID, GEN_LS, **kw)
    assert r.passed() and r["V2_holdout"].value < 1e-10
    assert r.arrays["exact_A_F"].shape == (3, 2, GEN_GRID.ny) and list(r.arrays["lines"]) == GEN_LS.names
    p = np.random.default_rng(9).permutation(g["prof"]["teff"].size)
    bad = dict(g["prof"], teff=g["prof"]["teff"][p])
    r = va.v2_holdout(bad, g["samples"][0], None, None, GEN_LOS, GEN_GRID, GEN_LS, check_teff=False, **kw)
    assert r["V2_holdout"].passed is False and r["V2_insample"].passed is False
    L = g["lib"]
    off = lb.FluxLibrary(L.edges, L.tmean, L.count, L.prof + 1e-5 * g["win"], L.fc, L.dT)
    r = va.v2_holdout(g["prof"], g["samples"][0], None, None, GEN_LOS, GEN_GRID, GEN_LS, library=off, **kw)
    assert r["V2_holdout"].passed is True and r["V2_insample"].passed is False
    # samples of another dump are refused (their T_eff' are not the models')
    with pytest.raises(ValueError, match="differ from the models"):
        va.v2_holdout(g["prof"], g["samples"][3], None, None, GEN_LOS, GEN_GRID, GEN_LS, **kw)


def test_nearest_bin_integrator(gen):
    """NearestBin (fw_disc_validate.py's library method) through v1_exact: the toy's dump 0 has T_eff' at the bin
    centres, so it reproduces the exact sums to the float32 library; F equals integrate_library_nearest."""
    MU, TN, PN = gen["proj"]
    nb = va.NearestBin(gen["lib"], GEN_GRID)
    r = va.v1_exact(nb, gen["exact"], gen["samples"][0], MU, TN, PN, lref=GEN_LS, label="nearest", tolerances=GEN_TOL)
    assert r.passed() and r["V1_nearest"].value < 1e-7
    s = va._sample_dict(gen["samples"][3])
    v = sph.los_velocity(s["ur"], s["uth"], s["uph"], MU[1], TN[1], PN[1])
    b = gen["lib"].bin_index(s["teff"])
    w = sph.disc_weights(MU[1], 1.0)[:, None] * gen["lib"].fc[b]
    F = disc.integrate_library_nearest(gen["lib"], s["teff"], v, w, GEN_GRID)
    np.testing.assert_array_equal(nb(MU[1], v, *nb.pairs(s["teff"]))[0], F)
    with pytest.raises(ValueError):
        va.NearestBin(gen["lib"], VelocityGrid(vmax=500.0, vshift=100.0))


def test_v1_inputs(gen):
    MU, TN, PN = gen["proj"]
    with pytest.raises(ValueError, match="exact sums have shape"):
        va.v1_exact(gen["integ"], dict(F=gen["exact"]["F"][:2], F0=gen["exact"]["F0"][:2]), gen["samples"][0], MU,
                    TN, PN, lref=GEN_LS)
    with pytest.raises(ValueError, match="lref is required"):
        va.v1_exact(gen["integ"], gen["exact"], gen["samples"][0], MU, TN, PN)
    with pytest.raises(ValueError, match="projections are for"):
        va.v1_exact(gen["integ"], gen["exact"], gen["samples"][0], MU[:, :10], TN[:, :10], PN[:, :10], lref=GEN_LS)
    r = va.v1_exact(gen["integ"], gen["exact"], gen["samples"][0], MU, TN, PN, lref=GEN_LS, label="x", tolerance=0.0)
    assert r.names() == ["V1_x", "V1_x_dEW"] and r["V1_x"].passed is False      # explicit tolerance wins
    assert r["V1_x"].details["lines"] == GEN_LS.names and r["V1_x_dEW"].unit == "A"


@pytest.mark.parametrize("member", ["F", "F0"])
def test_v1_nan(gen, member):
    """A NaN in only F or only F0 of the exact sums makes V1 NaN, which fails."""
    MU, TN, PN = gen["proj"]
    ex = {k: np.array(gen["exact"][k], copy=True) for k in ("F", "F0")}
    ex[member][1, 1, 100] = np.nan
    r = va.v1_exact(gen["integ"], ex, gen["samples"][0], MU, TN, PN, lref=GEN_LS, tolerances=GEN_TOL)
    c = r["V1_flux"]
    assert np.isnan(c.value) and c.passed is False and np.isnan(c.details["per_line"][1])
    assert not r.passed()


# ----------------------------------------------------------------------------------------------
# a second toy: T_eff' spread within the bins, profiles linear in T_eff' (interpolation matters)
# ----------------------------------------------------------------------------------------------
def _spread_toy(N=3000, nbin=20, seed=12):
    """Models with T_eff' spread uniformly within interior 10 K bins (the end bins: one value each, so no point lies
    beyond the end nodes), line depths linear in T_eff' and F_c constant: the bin means and the linear interpolation
    between nodes are exact up to the float32 profiles, while the points sit anywhere between nodes (0 < a < 1)."""
    rng = np.random.default_rng(seed)
    th, ph = sph.fibonacci_sphere(N)
    T0 = 38000.0
    b = rng.integers(0, nbin, N)
    teff = T0 + 10.0 * b + 10.0 * rng.random(N)
    teff[b == 0] = T0 + 5.0
    teff[b == nbin - 1] = T0 + 10.0 * nbin - 5.0
    x = (teff - (T0 + 5.0 * nbin)) / 100.0
    yn = np.linspace(-800.0, 800.0, 321)
    nl = len(GEN_LS)
    lam = np.empty((N, nl, yn.size), np.float32)
    fn, fcont = np.empty_like(lam), np.empty_like(lam)
    for j, (A, B, c, sig) in enumerate(((0.35, 0.05, 0.0, 35.0), (0.25, -0.04, 1.0, 50.0))):
        lam[:, j] = (GEN_LS.lref[j] * np.exp(yn / C_KMS))[None, :].astype(np.float32)
        fn[:, j] = (1.0 - (A + B * x)[:, None] * np.exp(-0.5 * ((yn[None, :] - c) / sig) ** 2)).astype(np.float32)
        fcont[:, j] = 1.0 + 0.05 * j
    prof = dict(lam=lam, fnorm=fn, fcont=fcont, teff=teff, theta=th, phi=ph, lines=np.array(GEN_LS.names))
    smp = dict(teff=teff.astype(np.float32), **{k: (25.0 * rng.standard_normal(N)).astype(np.float32)
                                                 for k in ("ur", "uth", "uph")})
    lib = lb.FluxLibrary.build(teff, lam, fn, fcont[:, :, 0], GEN_GRID, GEN_LS, block=1000, stride=2)
    nodes = lb.lib_nodes(lib, nmin=20)
    ex = disc.integrate_exact_stream(prof, smp, GEN_LS, los=GEN_LOS, grid=GEN_GRID, checks=False, block=1000,
                                     stride=2)
    return dict(prof=prof, sample=smp, lib=lib, nodes=nodes, integ=disc.DiscFlux(nodes, GEN_GRID), exact=ex,
                proj=sph.project_los(th, ph, GEN_LOS))


@pytest.fixture(scope="module")
def spread():
    return _spread_toy()


def test_spread_interpolation_faults(spread):
    """With T_eff' between nodes, V1, V2 and the brute force see wrong interpolation weights (a swapped with 1 - a,
    or a^2), which the bin-centre toy cannot (test_general_square_weights_invisible)."""
    g = spread
    MU, TN, PN = g["proj"]
    k0, k1, a = g["integ"].pairs(g["sample"]["teff"])
    assert 0.3 < np.mean((a > 0.05) & (a < 0.95))                   # many points well between nodes
    kw = dict(lref=GEN_LS, tolerances=GEN_TOL)
    vkw = dict(block=1000, stride=2, tolerances=GEN_TOL)
    r = va.v1_exact(g["integ"], g["exact"], g["sample"], MU, TN, PN, **kw)
    assert r.passed() and r["V1_flux"].value < 2e-7, r.table()
    r = va.v2_holdout(g["prof"], g["sample"], None, None, GEN_LOS, GEN_GRID, GEN_LS, **vkw)
    assert r.passed() and r["V2_holdout"].value < 2e-7, r.table()
    smp = {0: g["sample"]}
    for kind in ("swap", "square"):
        bad = _WeightFault(g["integ"], kind)
        r = va.v1_exact(bad, g["exact"], g["sample"], MU, TN, PN, **kw)
        assert r["V1_flux"].passed is False and r["V1_flux"].value > 1e-5, (kind, r.table())
        r = va.v2_holdout(g["prof"], g["sample"], None, None, GEN_LOS, GEN_GRID, GEN_LS,
                          factory=_fault_factory(kind, None), **vkw)
        assert r["V2_holdout"].passed is False and r["V2_insample"].passed is False, (kind, r.table())
        r = va.brute_force_check(g["nodes"], smp, MU, TN, PN, GEN_GRID, dumps=[0], integ=bad, tolerances=GEN_TOL)
        assert r["brute_vs_integrator"].passed is False, kind


def test_sign_convention_analytic():
    """The conventions shared by the pipeline and its references, checked without los_velocity: observer on +z; a
    point at the disc centre moving outwards (towards the observer) at 50 km/s puts the line at y = -50 km/s
    (blueshift, lambda (1 - v/c)); a point at theta = 60 deg moving along +theta_hat (away from the observer:
    theta_hat . z = -sin 60) at 50 km/s puts it at +43 km/s (redshift). Checked for DiscFlux, the brute force and the
    exact sums with continuous shifts."""
    grid = VelocityGrid(dv=1.0, vmax=400.0, vshift=100.0)
    line = 1.0 - 0.5 * np.exp(-0.5 * (grid.y / 10.0) ** 2)
    nodes = dict(t=np.array([38000.0, 38100.0]), count=np.array([20.0, 20.0]), prof=np.array([[line], [line]]),
                 fc=np.ones((2, 1)))
    integ = disc.DiscFlux(nodes, grid)
    los = np.array([[0.0, 0.0, 1.0]])
    yn = np.linspace(-300.0, 300.0, 1201)
    lref = np.array([5000.0])
    for theta, comp, mu_hand, tn_hand, ypeak in ((0.0, "ur", 1.0, 0.0, -50.0),
                                                 (np.pi / 3, "uth", 0.5, -np.sin(np.pi / 3), 43.0)):
        mu, tn, pn = sph.project_los(np.array([theta]), np.array([0.0]), los)
        # by hand: r_hat = (sin th, 0, cos th), theta_hat = (cos th, 0, -sin th), phi_hat = (0, 1, 0) at phi = 0
        np.testing.assert_allclose([mu[0, 0], tn[0, 0], pn[0, 0]], [mu_hand, tn_hand, 0.0], rtol=0, atol=1e-15)
        smp = dict(teff=np.array([38050.0]), ur=np.zeros(1), uth=np.zeros(1), uph=np.zeros(1))
        smp[comp] = np.array([50.0])
        F = va._run(integ, va._sample_dict(smp), mu, tn, pn)[0][0, 0]
        assert grid.y[np.argmin(F)] == ypeak, (comp, grid.y[np.argmin(F)])
        Fb = va.brute_force(nodes, smp, mu, tn, pn, grid)[0][0, 0]
        assert grid.y[np.argmin(Fb)] == ypeak
        prof = dict(lam=(lref[0] * np.exp(yn / C_KMS))[None, None, :], fnorm=(1.0 - 0.5 * np.exp(
            -0.5 * (yn / 10.0) ** 2))[None, None, :], fcont=np.ones((1, 1, yn.size)))
        Fx = va.exact_continuous(prof, smp, mu, tn, pn, lref, grid=grid)["F"][0, 0]
        vexp = 50.0 * (1.0 if comp == "ur" else -np.sin(np.pi / 3))                       # v = u . n by hand
        yexp = C_KMS * np.log(1.0 - vexp / C_KMS)
        assert abs(grid.y[np.argmin(Fx)] - yexp) <= 0.5 and np.sign(yexp) == np.sign(ypeak)


# ----------------------------------------------------------------------------------------------
# variants of the flux integrator (smoothed, corrected), wrong nodes, empty inputs
# ----------------------------------------------------------------------------------------------
def test_v4_v5_smoothed_and_corrected(gen, tmp_path):
    """V4 and V5 of a smoothed or corrected integrator (the flux_sm335 / flux_lamfix situation): V5 rebuilds the nodes
    with the integrator's smoothing (and the given correction) and passes; V4 is informational with smoothing and
    passes with the correction given; a missing correction raises."""
    from ppmpy.synspec.dumps import flux_integrator
    g = gen
    MU, TN, PN = g["proj"]
    kw = dict(tolerances=GEN_TOL)
    sm = flux_integrator(g["lib"], nmin=20, smooth=60.0, grid=GEN_GRID)
    r4 = va.v4_nearest(sm, g["lib"], g["samples"], [1, 2], MU, TN, PN, **kw)["V4"]
    assert r4.passed is None and r4.value > 1e-4 and "smooth" in r4.details["note"]
    r4 = va.v4_nearest(sm, g["lib"], g["samples"], [1, 2], MU, TN, PN, tolerance=1e-6)["V4"]
    assert r4.passed is False                                      # an explicit tolerance makes it a check
    r5 = va.v5_node_merging(sm, g["lib"], g["samples"], 1, MU, TN, PN, **kw)["V5"]
    assert r5.passed and r5.details["smooth"] == 60.0
    corr = 1e-4 * np.broadcast_to(g["win"], g["lib"].prof.shape).copy()
    co = flux_integrator(g["lib"], nmin=20, corr=corr, grid=GEN_GRID)
    with pytest.raises(ValueError, match="correction"):
        va.v4_nearest(co, g["lib"], g["samples"], [1], MU, TN, PN, **kw)
    with pytest.raises(ValueError, match="correction"):
        va.v5_node_merging(co, g["lib"], g["samples"], 1, MU, TN, PN, **kw)
    with pytest.raises(ValueError, match="carry none"):
        va.v4_nearest(g["integ"].__class__(lb.lib_nodes(g["lib"], nmin=20), GEN_GRID), g["lib"], g["samples"], [1],
                      MU, TN, PN, corr=corr, **kw)
    r4 = va.v4_nearest(co, g["lib"], g["samples"], [1, 2], MU, TN, PN, corr=corr, **kw)["V4"]
    assert r4.passed and r4.value < 1e-12
    r5 = va.v5_node_merging(co, g["lib"], g["samples"], 1, MU, TN, PN, corr=corr, **kw)["V5"]
    assert r5.passed and r5.value == 0.0
    # the correction from an .npz, through run_validation
    np.savez(str(tmp_path / "corr.npz"), corr=corr)
    r = va.run_validation(co, MU, TN, PN, library=g["lib"], samples=g["samples"], dumps=[1],
                          corr=str(tmp_path / "corr.npz"), nmins=(5,), tolerances=GEN_TOL)
    assert r.names() == ["V4", "V5"] and r.passed()


def test_nodes_must_be_the_integrators(gen):
    """V6, the brute force and run_validation refuse nodes the integrator was not built from (another nmin; an
    integrator with other node T_eff', e.g. the intensity method's); run_validation skips V4-V6 and the brute force
    for an integrator without fc."""
    g = gen
    MU, TN, PN = g["proj"]
    other = lb.lib_nodes(g["lib"], nmin=400)
    assert other.nn < g["nodes"].nn
    with pytest.raises(ValueError, match="not these nodes"):
        va.v6_rounding(g["integ"], other, g["samples"], [1], MU, TN, PN, nsub=100)
    with pytest.raises(ValueError, match="not these nodes"):
        va.brute_force_check(other, g["samples"], MU, TN, PN, GEN_GRID, dumps=[1], integ=g["integ"])
    with pytest.raises(ValueError, match="not these nodes"):
        va.run_validation(g["integ"], MU, TN, PN, nodes=other, samples=g["samples"], dumps=[1])
    fc2 = dict(t=g["nodes"].t, count=g["nodes"].count, prof=g["nodes"].prof, fc=g["nodes"].fc * 1.01)
    with pytest.raises(ValueError, match="fc differ"):
        va.v6_rounding(g["integ"], fc2, g["samples"], [1], MU, TN, PN, nsub=100)
    nb = va.NearestBin(g["lib"], GEN_GRID)                           # no fc: not a flux integrator
    r = va.run_validation(nb, MU, TN, PN, sample=g["samples"][0], exact=g["exact"], library=g["lib"],
                          nodes=other, samples=g["samples"], dumps=[1], brute_dumps=[1], lref=GEN_LS, label="nb",
                          tolerances=GEN_TOL)
    assert r.names() == ["V1_nb", "V1_nb_dEW"]
    assert all(r.meta["skipped"][k] == ["flux_integrator"] for k in ("V4", "V5", "V6", "brute"))


def test_empty_inputs(gen):
    """A check of nothing raises instead of passing with value 0."""
    g = gen
    MU, TN, PN = g["proj"]
    with pytest.raises(ValueError, match="no dumps"):
        va.v3_tails(g["integ"], g["samples"], [], MU, TN, PN)
    with pytest.raises(ValueError, match="no dumps"):
        va.v4_nearest(g["integ"], g["lib"], g["samples"], [], MU, TN, PN)
    with pytest.raises(ValueError, match="no dumps"):
        va.v6_rounding(g["integ"], g["nodes"], g["samples"], [], MU, TN, PN)
    with pytest.raises(ValueError, match="no dumps"):
        va.brute_force_check(g["nodes"], g["samples"], MU, TN, PN, GEN_GRID, dumps=[])
    with pytest.raises(ValueError, match="no nmins"):
        va.v5_node_merging(g["integ"], g["lib"], g["samples"], 1, MU, TN, PN, nmins=())
    for margin in (0.0, -10.0, np.nan):
        with pytest.raises(ValueError, match="margin"):
            va.v3_tails(g["integ"], g["samples"], [3], MU, TN, PN, margin=margin)
    r = va.v3_tails(g["integ"], g["samples"], [3], MU, TN, PN, margin=1.0)        # below the node spacing: fine
    assert np.isfinite(r["V3_extrap"].value) and r.meta["margin"] == 1.0
    r = va.run_validation(g["integ"], MU, TN, PN, library=g["lib"], samples=g["samples"], dumps=[1], nmins=())
    assert r.names() == ["V4"] and r.meta["skipped"]["V5"] == ["nmins"]           # V5 skipped: no nmins
    with pytest.warns(UserWarning, match="no check ran"):
        r = va.run_validation()
    assert len(r) == 0 and not r.passed() and r.meta["ran"] == []


# ----------------------------------------------------------------------------------------------
# brute force
# ----------------------------------------------------------------------------------------------
def _naive(nodes, smp, mu, v, grid):
    """Point by point in Python: interpolated node depth, shifted, added."""
    t, P, FC = nodes["t"], nodes["prof"], nodes["fc"]
    nl, ny = P.shape[1], grid.ny
    D, den = np.zeros((nl, ny)), np.zeros(nl)
    D0 = np.zeros((nl, ny))
    for i in np.where(mu > 0)[0]:
        T = float(smp["teff"][i])
        k0 = min(max(int(np.searchsorted(t, T, side="right")) - 1, 0), t.size - 2)
        a = min(max((T - t[k0]) / (t[k0 + 1] - t[k0]), 0.0), 1.0)
        s = int(np.rint(-C_KMS * np.log(1.0 - v[i] / C_KMS) / grid.dv))
        s = max(-grid.nshift, min(grid.nshift, s))
        for j in range(nl):
            w0, w1 = mu[i] * (1 - a) * FC[k0, j], mu[i] * a * FC[k0 + 1, j]
            row = w0 * (1.0 - P[k0, j]) + w1 * (1.0 - P[k0 + 1, j])
            den[j] += w0 + w1
            D0[j] += row
            for iy in range(ny):
                if 0 <= iy + s < ny:
                    D[j, iy] += row[iy + s]
    return 1.0 - D / den[:, None], 1.0 - D0 / den[:, None]


def test_brute_force_naive_and_discflux():
    """brute_force equals a naive per-point loop and DiscFlux to ~1e-15, with shifts beyond +-nshift (clipped) and
    T_eff' beyond the nodes (clamped)."""
    grid = VelocityGrid(dv=1.0, vmax=300.0, vshift=60.0)
    rng = np.random.default_rng(2)
    nn, nl = 6, 2
    t = 38000.0 + 15.0 * np.arange(nn)
    prof = np.ones((nn, nl, grid.ny))
    for k in range(nn):
        for j in range(nl):
            prof[k, j] = 1.0 - (0.3 + 0.02 * k + 0.05 * j) * np.exp(-0.5 * ((grid.y - k) / (20.0 + 5 * j)) ** 2)
    nodes = dict(t=t, prof=prof, fc=1.0 + 0.1 * rng.random((nn, nl)), count=np.full(nn, 20.0))
    N = 60
    th, ph = sph.fibonacci_sphere(N)
    proj = sph.project_los(th, ph, GEN_LOS[:2])
    smp = dict(teff=rng.uniform(t[0] - 20, t[-1] + 20, N), ur=30 * rng.standard_normal(N),
               uth=30 * rng.standard_normal(N), uph=30 * rng.standard_normal(N))
    smp["ur"][:3] = [120.0, -150.0, 90.0]                          # some shifts beyond 60 km/s
    F, F0 = va.brute_force(nodes, smp, *proj, grid, chunk=7)
    integ = disc.DiscFlux(nodes, grid)
    Fi, F0i = va._run(integ, va._sample_dict(smp), *proj)
    for k in range(2):
        v = sph.los_velocity(smp["ur"], smp["uth"], smp["uph"], proj[0][k], proj[1][k], proj[2][k])
        Fn, F0n = _naive(nodes, smp, proj[0][k], v, grid)
        np.testing.assert_allclose(F[k], Fn, rtol=0, atol=1e-14)
        np.testing.assert_allclose(F0[k], F0n, rtol=0, atol=1e-14)
    np.testing.assert_allclose(F, Fi, rtol=0, atol=1e-14)
    np.testing.assert_allclose(F0, F0i, rtol=0, atol=1e-14)
    F2, F02 = va.brute_force(nodes, smp, *proj, grid, chunk=1000)  # chunking changes the last bits only
    np.testing.assert_allclose(F2, F, rtol=0, atol=1e-15)


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_brute_force_parallel(gen, method):
    """nproc 2 (fork, spawn) gives the same bits as the serial run."""
    smp = gen["samples"][1]
    F, F0 = va.brute_force(gen["nodes"], smp, *gen["proj"], GEN_GRID)
    Fp, F0p = va.brute_force(gen["nodes"], smp, *gen["proj"], GEN_GRID, nproc=2, start_method=method)
    np.testing.assert_array_equal(F, Fp)
    np.testing.assert_array_equal(F0, F0p)


def test_brute_force_check_files(gen, tmp_path):
    """Stored products as a directory of dNNNN.npz and as (outdir, name); a reference set with Fl/Fl0 (the layout of
    r3_out.npz) is informational; arrays and report layout."""
    d = tmp_path / "flux"
    d.mkdir()
    for k, (F, F0) in gen["stored"].items():
        np.savez(str(d / "d{:04d}.npz".format(k)), F=F, F0=F0)
    F1 = gen["stored"][1][0].astype(np.float64) + 1e-3
    np.savez(str(tmp_path / "ref.npz"), dumps=np.array([1, 7]), Fl=np.array([F1, F1]),
             Fl0=np.array([gen["stored"][1][1]] * 2))
    for stored in (str(d), (str(tmp_path), "flux")):
        r = va.brute_force_check(gen["nodes"], gen["samples"], *gen["proj"], GEN_GRID, lref=GEN_LS, dumps=[1, 2],
                                 stored=stored, reference=str(tmp_path / "ref.npz"), tolerances=GEN_TOL)
        assert r.names() == ["brute_vs_integrator", "brute_vs_stored", "brute_vs_reference"]
        assert r["brute_vs_integrator"].passed and r["brute_vs_stored"].passed
        assert r["brute_vs_reference"].passed is None and abs(r["brute_vs_reference"].value - 1e-3) < 1e-6
        np.testing.assert_array_equal(r.arrays["brute_dumps"], [1, 2])
        np.testing.assert_array_equal(r.arrays["brute_reference_dumps"], [1])
        assert r.arrays["brute_integ"].shape == (2, 2) and r["brute_vs_stored"].details["lines"] == GEN_LS.names
    # stored products offset by 5e-6 (below V1 and the 2026-09-29 review's 7.6e-6) fail under the default tolerance
    off = {k: (F + np.float32(5e-6), F0) for k, (F, F0) in gen["stored"].items()}
    r = va.brute_force_check(gen["nodes"], gen["samples"], *gen["proj"], GEN_GRID, dumps=[1], stored=lambda d: off[d])
    assert r["brute_vs_stored"].passed is False and r["brute_vs_stored"].tolerance == 2.5e-7
    with pytest.raises(ValueError, match="grid"):
        va.brute_force_check(gen["nodes"], gen["samples"], *gen["proj"], dumps=[1])
    # one sample, no dumps
    r = va.brute_force_check(gen["nodes"], gen["samples"][1], *gen["proj"], GEN_GRID)
    np.testing.assert_array_equal(r.arrays["brute_dumps"], [-1])
    assert r["brute_vs_integrator"].value < 1e-13


# ----------------------------------------------------------------------------------------------
# EW conservation
# ----------------------------------------------------------------------------------------------
def test_ew_conservation(gen, tmp_path):
    MU, TN, PN = gen["proj"]
    s = va._sample_dict(gen["samples"][1])
    F, F0 = va._run(gen["integ"], s, MU, TN, PN)
    c = va.ew_conservation(F, F0, GEN_GRID.y, GEN_LS)
    assert c.passed and c.value < 1e-13 and c.unit == "relative" and c.details["lines"] == GEN_LS.names
    cj = va.ew_conservation(F, F0, GEN_GRID.y, GEN_LS, ew_jacobian=True)
    assert 1e-7 < cj.value < 1e-4                                  # exp(-s/c): O(v/c)
    Fo, _ = va._run(gen["faulty"]["offset"], s, MU, TN, PN)
    bad = va.ew_conservation(Fo, F0, GEN_GRID.y, GEN_LS)
    assert bad.passed is False and bad.value > 1e-5
    # a time series (dumps, los, lines, ny) read in blocks, float32, from a file with Y and LREF
    Fs = np.array([F, Fo, F]).astype(np.float32)
    F0s = np.array([F0, F0, F0]).astype(np.float32)
    p = str(tmp_path / "ts.npz")
    np.savez(p, F=Fs, F0=F0s, Y=GEN_GRID.y, LREF=GEN_LS.lref)
    c = va.ew_conservation(p, block=2)
    c1 = va.ew_conservation(Fs, F0s, GEN_GRID.y, GEN_LS.lref, block=1)
    assert c.value == c1.value and c.passed is False and abs(c.value - bad.value) < 1e-6        # float32
    assert c.details["lines"] == ["line0", "line1"]
    lead = [divmod(int(w), 3) for w in c.details["argmax_flat"]]
    assert all(i == 1 for i, _ in lead)                            # the maximum is in the corrupted dump
    with pytest.raises(ValueError):
        va.ew_conservation(F, F0[:, :1], GEN_GRID.y, GEN_LS)
    with pytest.raises(ValueError):
        va.ew_conservation(F, F0)


@pytest.mark.parametrize("block", [1, 2, 64])
@pytest.mark.parametrize("case", ["one_middle", "first_dump", "last_dump", "F0_only"])
def test_ew_conservation_nan(gen, case, block):
    """A non-finite profile anywhere fails the check, whatever the block order (NaN is sticky), and is counted."""
    MU, TN, PN = gen["proj"]
    F, F0 = va._run(gen["integ"], va._sample_dict(gen["samples"][1]), MU, TN, PN)
    Fs, F0s = np.array([F, F, F]), np.array([F0, F0, F0])
    if case == "one_middle":
        Fs[1, 2, 0, 50] = np.nan
    elif case == "first_dump":
        Fs[0] = np.nan
    elif case == "last_dump":
        Fs[2] = np.nan
    else:
        F0s[1, 0, 1, 10] = np.nan
    c = va.ew_conservation(Fs, F0s, GEN_GRID.y, GEN_LS, block=block)
    assert np.isnan(c.value) and c.passed is False, c
    nf = c.details["nonfinite"]
    expect = {"one_middle": [1, 0], "first_dump": [3, 3], "last_dump": [3, 3], "F0_only": [0, 1]}[case]
    np.testing.assert_array_equal(nf, expect)
    bad = int(np.argmax(nf))
    assert np.isnan(c.details["per_line"][bad]) and "non-finite" in c.details["note"]
    first = {"one_middle": 1 * 3 + 2, "first_dump": 0, "last_dump": 6, "F0_only": 1 * 3 + 0}[case]
    assert c.details["argmax_flat"][bad] == first                 # the first non-finite profile
    if case in ("one_middle", "F0_only"):                          # the other line is fine
        assert np.isfinite(c.details["per_line"][1 - bad]) and c.details["per_line"][1 - bad] < 1e-13


# ----------------------------------------------------------------------------------------------
# T_eff' ranges
# ----------------------------------------------------------------------------------------------
def test_teff_ranges_and_select(tmp_path):
    rng = np.random.default_rng(4)
    N = 500
    rows = []
    for d in range(10, 16):
        t = (38000.0 + 300 * rng.standard_normal(N)).astype(np.float32)
        np.savez(str(tmp_path / "d{:04d}.npz".format(d)), teff=t, ur=t, uth=t, uph=t, t_s=0.0)
        tf = t.astype(np.float64)
        rows.append([d, tf.min(), tf.max(), (tf < 37500.0).sum(), (tf > 38600.0).sum(), tf.std()])
    rg = va.teff_ranges(str(tmp_path), range(10, 16), trange=(37500.0, 38600.0))
    np.testing.assert_array_equal(rg, np.array(rows))
    np.testing.assert_array_equal(va.teff_ranges(str(tmp_path), range(10, 16), trange=(37500.0, 38600.0), nproc=2), rg)
    assert np.isnan(va.teff_ranges(str(tmp_path), [12])[0, 3])
    sel = va.v3_select(rg)
    expect = {int(rg[np.argmax(rg[:, 3] + rg[:, 4]), 0]), int(rg[np.argmin(rg[:, 1]), 0]),
              int(rg[np.argmax(rg[:, 2]), 0])}
    assert sel == sorted(expect)
    with pytest.raises(ValueError, match="trange"):               # n_lo, n_hi NaN: no silent first dump
        va.v3_select(va.teff_ranges(str(tmp_path), range(10, 16)))
    with pytest.raises(ValueError):
        va.v3_select(np.zeros((0, 6)))


# ----------------------------------------------------------------------------------------------
# driver and LPV comparison
# ----------------------------------------------------------------------------------------------
def test_run_validation_and_lpv(gen, tmp_path):
    g = gen
    MU, TN, PN = g["proj"]
    # a toy time series of the run: dumps 1, 2, 3 (residual rms ~1e-3 here)
    F, F0 = [], []
    for d in (1, 2, 3):
        a, b = va._run(g["integ"], va._sample_dict(g["samples"][d]), MU, TN, PN)
        F.append(a)
        F0.append(b)
    ts = dict(F=np.array(F).astype(np.float32), F0=np.array(F0).astype(np.float32), Y=GEN_GRID.y, LREF=GEN_LS.lref)
    lpv = va.lpv_residual_rms(ts)
    assert lpv["rms"].shape == (2,) and np.all(lpv["rms"] > 0) and lpv["ew_rms"] is None
    with pytest.warns(UserWarning, match="of the LPV residual rms"):
        r = va.run_validation(g["integ"], MU, TN, PN, sample=g["samples"][0], exact=g["exact"], library=g["lib"],
                              nodes=g["nodes"], samples=g["samples"], dumps=[1, 2], v3_dumps=[3], lref=GEN_LS,
                              nsub=1000, margin=50.0, brute_dumps=[1],
                              holdout=dict(profiles=g["prof"], sample=g["samples"][0], theta=None, phi=None,
                                           los=GEN_LOS, grid=GEN_GRID, lref=GEN_LS, block=1000, stride=2),
                              timeseries=ts, tolerances=GEN_TOL)
    assert r.names() == ["V1_flux", "V1_flux_dEW", "V2_holdout", "V2_holdout_dEW", "V2_insample", "V2_insample_dEW",
                         "V3_extrap", "V3_leaveout", "V4", "V5", "V6", "brute_vs_integrator", "ew_conservation"]
    assert r.passed() and r.meta["skipped"] == {} and r.meta["ran"] == r.names()
    assert r.meta["tolerances"]["V6"] == GEN_TOL["V6"] and r.meta["tolerances"]["V2"] == GEN_TOL["V2"]
    for c in r:
        if c.unit == "continuum":
            np.testing.assert_allclose(c.details["lpv_ratio"], np.asarray(c.details["per_line"]) / lpv["rms"])
    warned = {c.name for c in r.warnings()}
    assert "V6" in warned and "V1_flux" not in warned               # 1 km/s rounding of 1500 points vs 3 dumps
    assert "!" in r.table()
    # V2 and the arrays of every part are kept; a report of the driver saves
    assert "holdout_A_F" in r.arrays and "V4" in r.arrays and "brute_integ" in r.arrays
    r.to_npz(str(tmp_path / "v.npz"))
    r.to_json(str(tmp_path / "v.json"))
    # nothing given: everything skipped
    with pytest.warns(UserWarning, match="no check ran"):
        r = va.run_validation()
    assert len(r) == 0 and set(r.meta["skipped"]) == {"V1", "V2", "V3", "V4", "V5", "V6", "brute", "ew_conservation"}
    # only V4 / V5 / V6 inputs
    r = va.run_validation(g["integ"], MU, TN, PN, library=g["lib"], nodes=g["nodes"], samples=g["samples"],
                          dumps=[1], nsub=500, tolerances=GEN_TOL)
    assert r.names() == ["V4", "V5", "V6"] and "V1" in r.meta["skipped"]


def test_compare_lpv_units(tmp_path):
    r = va.ValidationReport([va.CheckResult("a", 2e-5, 1e-4, details=dict(per_line=[2e-5, 1e-6])),
                             va.CheckResult("b", 1e-5, 1e-4, unit="A", details=dict(per_line=[1e-5, 1e-5])),
                             va.CheckResult("c", 1e-9, 1e-6, unit="relative", details=dict(per_line=[1e-9, 1e-9])),
                             va.CheckResult("d", 1.0, None, details=dict(per_line=[1.0, 1.0]))])
    lpv = dict(rms=np.array([1e-4, 1e-4]), ew_rms=np.array([1e-3, 1e-3]), ew_rel_rms=np.array([1e-4, 1e-4]))
    with pytest.warns(UserWarning) as rec:
        va.compare_lpv(r, lpv)
    assert len(rec) == 1 and "a:" in str(rec[0].message)           # informational checks are not warned about
    np.testing.assert_allclose(r["a"].details["lpv_ratio"], [0.2, 0.01])
    np.testing.assert_allclose(r["b"].details["lpv_ratio"], [0.01, 0.01])
    assert r["c"].details["lpv_warn"] is False and r["d"].details["lpv_warn"] is True
    assert [c.name for c in r.warnings()] == ["a", "d"]
    # all ratios NaN (a NaN check value): flagged, and warned for a check with a tolerance
    r = va.ValidationReport([va.CheckResult("n", np.nan, 1e-4, details=dict(per_line=[np.nan, np.nan])),
                             va.CheckResult("z", 1e-6, 1e-4, details=dict(per_line=[1e-6, 1e-6]))])
    with pytest.warns(UserWarning) as rec:
        va.compare_lpv(r, dict(rms=np.array([1e-4, 0.0])))           # z: scale 0 -> inf ratio, also flagged
    msgs = sorted(str(w.message) for w in rec)
    assert len(msgs) == 2 and msgs[0].startswith("n: deviation up to nan") and msgs[1].startswith("z: ")
    assert r["n"].details["lpv_warn"] and r["n"].details["lpv_nonfinite"] == 2
    assert r["z"].details["lpv_warn"] and r["z"].details["lpv_nonfinite"] == 1
    assert "nan" in r.table()
    q = va.ValidationReport.from_json(r.to_json(str(tmp_path / "lpv.json")))     # NaN -> None, inf -> 'inf' and back
    np.testing.assert_array_equal(q["z"].details["lpv_ratio"], r["z"].details["lpv_ratio"])
    assert q["z"].details["lpv_ratio"][1] == np.inf
    assert np.all(np.isnan(q["n"].details["lpv_ratio"])) and q.table() == r.table()


# ----------------------------------------------------------------------------------------------
# M424 regressions
# ----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def m424():
    """Library, nodes, DiscFlux (nmin 20) and the 'matmul' projections of the M424 points (legacy validate)."""
    lib = lb.FluxLibrary.load(m424_path("run", "library_dT10.npz"))
    nodes = lb.lib_nodes(lib, nmin=20)
    pts = m424_path("run", "points.npz")
    th, ph = np.asarray(npz_member_memmap(pts, "theta")), np.asarray(npz_member_memmap(pts, "phi"))
    return dict(lib=lib, nodes=nodes, FX=disc.DiscFlux(nodes), proj=sph.project_los(th, ph, "thompson2024"))


def _ranges_m424():
    p = os.path.join(os.path.dirname(conftest.M424["run"]), "teff_ranges_all_dumps.npy")
    if not os.path.exists(p):
        pytest.skip("M424 product not available: {}".format(p))
    return np.load(p)


@pytest.mark.m424
def test_m424_teff_ranges():
    """teff_ranges reproduces columns 0-5 of teff_ranges_all_dumps.npy (n_lo, n_hi against the models' T_eff' range),
    and v3_select of the table gives the stored V3 dumps."""
    rg = _ranges_m424()
    samples = _samples_m424(3200)
    t = np.asarray(npz_member_memmap(m424_path("run", "profiles.npz"), "teff"))
    dl = [3200, 3201, 3334, 4169, 4391, 4800]
    mine = va.teff_ranges(samples, dl, trange=(t.min(), t.max()), nproc=2)
    np.testing.assert_array_equal(mine, rg[np.array(dl) - 3200, :6])
    assert va.v3_select(rg) == list(va.RECORDED_M424_ARRAYS["V3_dumps"])


@pytest.mark.m424
@pytest.mark.slow
def test_m424_validate_flux(m424):
    """V1 (flux), V3, V4, V5, V6 reproduce validate.npz bit for bit (~75 s on a login node, 1.7 GB)."""
    ref = np.load(m424_path("disc", "validate.npz"))
    samples = _samples_m424(3600)
    MU, TN, PN = m424["proj"]
    FX, lib = m424["FX"], m424["lib"]
    dl = list(ref["V4_dumps"])
    t0 = time.time()
    rep = va.ValidationReport.merge(
        va.v1_exact(FX, m424_path("run", "disc_los8.npz"), os.path.join(samples, "d3200.npz"), MU, TN, PN,
                    lref=LINESET),
        va.v4_nearest(FX, lib, samples, dl, MU, TN, PN, lref=LINESET),
        va.v5_node_merging(FX, lib, samples, dl[0], MU, TN, PN, lref=LINESET),
        va.v3_tails(FX, samples, va.v3_select(_ranges_m424()), MU, TN, PN, lref=LINESET),
        va.v6_rounding(FX, m424["nodes"], samples, dl, MU, TN, PN, lref=LINESET))
    print("M424 V1, V3-V6: {:.0f} s".format(time.time() - t0))
    print(rep.table())
    for k in ref.files:
        if k.startswith("V1_imu") or k in ("lines", "nmin"):
            continue
        np.testing.assert_array_equal(rep.arrays[k], ref[k], err_msg=k)
    for k, v in va.RECORDED_M424_ARRAYS.items():
        if k in ref.files:
            np.testing.assert_array_equal(np.array(v), ref[k], err_msg="RECORDED " + k)
    assert rep.passed()
    assert rep["V1_flux"].value == va.RECORDED_M424["V1_flux"] and rep["V6"].value == va.RECORDED_M424["V6"]


@pytest.mark.m424
@pytest.mark.slow
def test_m424_validate_imu(m424):
    """V1 of the intensity method with the frozen fw_disc.DiscImu (T_eff' interpolation; setup ~2 min, ~8 GB) vs
    disc_los8_imu.npz: V1_imu_F, V1_imu_F0 of validate.npz bit for bit."""
    fd = _legacy_fw_disc()
    if not os.path.exists(fd.IMU_LIB):
        pytest.skip("M424 product not available: {}".format(fd.IMU_LIB))
    ref = np.load(m424_path("disc", "validate.npz"))
    MU, TN, PN = m424["proj"]
    t0 = time.time()
    imu = fd.DiscImu()
    t1 = time.time()
    rep = va.v1_exact(imu, m424_path("run", "disc_los8_imu.npz"), os.path.join(_samples_m424(3200), "d3200.npz"), MU,
                      TN, PN, label="imu", lref=LINESET, grid=VelocityGrid())
    print("DiscImu setup {:.0f} s, V1 {:.0f} s, max RSS {:.1f} GB".format(
        t1 - t0, time.time() - t1, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
    print(rep.table())
    for k in ("V1_imu_F", "V1_imu_F0"):
        np.testing.assert_array_equal(rep.arrays[k], ref[k], err_msg=k)
    assert rep["V1_imu"].value == va.RECORDED_M424["V1_imu"] and rep.passed()
    # V4-V6 and the brute force are for flux integrators: the intensity integrator's nodes are not the flux nodes
    samples = _samples_m424(4800)
    assert not np.array_equal(imu.t, m424["nodes"].t)
    with pytest.raises(ValueError, match="not these nodes"):
        va.v6_rounding(imu, m424["nodes"], samples, [4800], MU, TN, PN, grid=VelocityGrid())
    with pytest.raises(ValueError, match="not these nodes"):
        va.brute_force_check(m424["nodes"], samples, MU, TN, PN, VelocityGrid(), dumps=[4800], integ=imu)
    with pytest.warns(UserWarning, match="no check ran"):
        r = va.run_validation(imu, MU, TN, PN, library=m424["lib"], nodes=m424["nodes"], samples=samples,
                              dumps=[4800], brute_dumps=[4800], lref=LINESET, grid=VelocityGrid())
    assert all(r.meta["skipped"][k] == ["flux_integrator"] for k in ("V4", "V5", "V6", "brute"))


@pytest.mark.m424
@pytest.mark.slow
def test_m424_holdout():
    """v2_holdout (8 workers, fork; in-sample library built in the pass) reproduces every member of holdout.npz bit for
    bit (ew_jacobian=False: the file predates the Jacobian in the EW), and its in-sample library is library_dT10.npz
    bit for bit (~75 s, parent ~6 GB)."""
    ref = np.load(m424_path("disc", "holdout.npz"))
    L0 = np.load(m424_path("run", "library_dT10.npz"))
    t0 = time.time()
    rep = va.v2_holdout(m424_path("run", "profiles.npz"), os.path.join(_samples_m424(3200), "d3200.npz"), None, None,
                        "thompson2024", VelocityGrid(), LINESET, nproc=8, start_method="fork", log=print)
    print("M424 hold-out: {:.0f} s, parent max RSS {:.1f} GB".format(
        time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
    print(rep.table())
    assert list(rep.arrays) == list(ref.files)
    for k in ref.files:
        np.testing.assert_array_equal(rep.arrays[k], ref[k], err_msg=k)
    for k in ("edges", "tmean", "count", "prof", "fc"):
        np.testing.assert_array_equal(rep.data["libraries"]["all"][k], L0[k], err_msg=k)
    assert rep.passed() and rep["V2_holdout"].value == va.RECORDED_M424["V2_holdout"]


@pytest.mark.m424
@pytest.mark.slow
def test_m424_brute_force(m424):
    """The direct per-point sum (no FFT, no histogram) for dumps 3200 and 4800 (8 workers): DiscFlux to ~1e-15
    (float64), the stored flux products to their float32 rounding; r3_out.npz (if present) differs by ~1e-3
    (informational: not a same-input re-implementation, module notes)."""
    samples = _samples_m424(4800)
    stored = os.path.join(m424_path("disc", "flux"))
    ref = R3_OUT if os.path.exists(R3_OUT) else None
    t0 = time.time()
    rep = va.brute_force_check(m424["nodes"], samples, *m424["proj"], VelocityGrid(), lref=LINESET, dumps=[3200, 4800],
                               stored=stored, reference=ref, nproc=8, log=print)
    print("M424 brute force, 2 dumps: {:.0f} s".format(time.time() - t0))
    print(rep.table())
    assert rep["brute_vs_integrator"].value < 1e-12
    # the float32 rounding bound 2^-25 of values in [0.5, 1); the brute force's own last bits (~1e-15) may move it
    assert abs(rep["brute_vs_stored"].value - va.RECORDED_M424["brute_stored"]) <= 1e-15 and rep.passed()
    if ref is not None:
        c = rep["brute_vs_reference"]
        assert c.passed is None and 5e-4 < c.value < 2e-3


@pytest.mark.m424
@pytest.mark.slow
@pytest.mark.parametrize("run", ["flux", "imu"])
def test_m424_ew_conservation_timeseries(run):
    """ew_conservation of the whole production time series (1601 dumps, read in blocks) equals the recorded values
    (~65 s each)."""
    t0 = time.time()
    c = va.ew_conservation(m424_path("disc", run + "_timeseries.npz"))
    print("M424 {} time series EW conservation: {:.0f} s".format(run, time.time() - t0))
    assert c.value == va.RECORDED_M424["ew_" + run] and c.passed
    np.testing.assert_array_equal(c.details["per_line"], va.RECORDED_M424_ARRAYS["ew_" + run])
    assert not np.any(c.details["nonfinite"])


@pytest.mark.m424
@pytest.mark.parametrize("run", ["flux", "imu"])
def test_m424_ew_conservation(run):
    """Whole-step Doppler shifts conserve the EW of the stored per-dump products to their float32 rounding (<= 1.6e-7
    relative over all M424 dumps); with the Jacobian the O(v/c) compression shows (~1e-5)."""
    for d in (3200, 4000, 4800):
        p = m424_path("disc", run, "d{:04d}.npz".format(d))
        with np.load(p) as z:
            F, F0 = z["F"], z["F0"]
        c = va.ew_conservation(F, F0, VelocityGrid().y, LINESET)
        assert c.passed and c.value <= va.RECORDED_M424["ew_" + run], (d, c)
        cj = va.ew_conservation(F, F0, VelocityGrid().y, LINESET, ew_jacobian=True)
        assert 1e-7 < cj.value < 2e-5


@pytest.mark.m424
@pytest.mark.slow
def test_m424_lpv_and_run_validation(m424):
    """lpv_residual_rms of the production time series equals the recorded values; run_validation (V1, V4, V5, V6 of
    one dump, EW conservation of the flux time series) relates every check to the LPV: V6 of lambda4026/4922 (a
    20000-point subset) and V1 dEW of lambda4026 exceed 5 %, the V1 profile, V5 and EW checks are far below."""
    for run in ("imu", "flux"):
        lpv = va.lpv_residual_rms(m424_path("disc", run + "_timeseries.npz"))
        np.testing.assert_allclose(lpv["rms"], va.RECORDED_M424_ARRAYS["lpv_rms_" + run], rtol=1e-12, atol=0)
        np.testing.assert_allclose(lpv["ew_rms"], va.RECORDED_M424_ARRAYS["lpv_ew_rms_" + run], rtol=1e-12, atol=0)
    samples = _samples_m424(4000)
    MU, TN, PN = m424["proj"]
    t0 = time.time()
    with pytest.warns(UserWarning) as rec:
        r = va.run_validation(m424["FX"], MU, TN, PN, sample=os.path.join(samples, "d3200.npz"),
                              exact=m424_path("run", "disc_los8.npz"), library=m424["lib"], nodes=m424["nodes"],
                              samples=samples, dumps=[4800], lref=LINESET,
                              timeseries=m424_path("disc", "flux_timeseries.npz"))
    print("M424 run_validation: {:.0f} s".format(time.time() - t0))
    print(r.table())
    assert r.passed()
    # V6 (a 20000-point subset) and the lambda4026 EW of V1 (5.7 % of its EW rms) exceed 5 % of the LPV
    assert {str(w.message).split(":")[0] for w in rec} == {c.name for c in r.warnings()} == {"V6", "V1_flux_dEW"}
    for name in ("V1_flux", "V5", "ew_conservation"):
        assert np.nanmax(r[name].details["lpv_ratio"]) < 0.05, name
