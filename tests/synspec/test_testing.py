"""Tests of ppmpy.synspec.testing (toy data and the self-test) and of fwresults.check_indat_premise.

* The toy data: line set and parameters, the analytic depth (exact zeros beyond the taper) and continuum, the
  T_eff' range of the library, the analytic library (node count, zero padding, DiscFlux and flux_integrator accept
  it), the sphere (equal-area points, field amplitudes, smoothness, the plume, the uniform library-dump field), the
  per-point store (members, dtypes, rows, noise, block independence, the library built from it vs the analytic one)
  and a small toy run (the integrator reproduces the stored per-dump products).
* The self-test: passes (default; seed 1; nproc 2 with fork and spawn, bit for bit the serial results; quick=False,
  slow) within 60 s and 2 GB; every mutation fails the report, its 'fail' checks and none of its 'blind' ones; options
  (keep, tolerances, unknown mutation) and the command line.
* check_indat_premise on synthetic parts (the INDAT.DAT files written as fw_sphere_point.sh writes them): passes
  when only MODNAM and TEFF differ, detects a changed LOGG / VINF / extra line, accepts values written differently,
  counts missing INDATs and inconsistent MODNAM / TEFF, tags / max_parts / template / nproc options. M424 (marker
  m424, slow): one part each of task_0000, task_0039 and task_missing_0000.
"""
import glob
import hashlib
import io
import os
import subprocess
import sys
import tarfile

import numpy as np
import pytest

import conftest
from conftest import m424_path
from ppmpy.synspec import fwresults as fw
from ppmpy.synspec import library as lb
from ppmpy.synspec import sphere as sph
from ppmpy.synspec import testing as tt
from ppmpy.synspec.disc import DiscFlux
from ppmpy.synspec.io import read_meta
from ppmpy.synspec.spectral import LineSet, VelocityGrid, y_of_lam

# PP 2026-10-01: the M424 per-point INDAT template, frozen from the project (git show 67e042f:
# project/analysis/fastwind/INDAT_M424test.DAT; fw_sphere_task.sh copies it to INDAT.template, fw_sphere_point.sh
# replaces line 1 and the first value of line 4 per point)
INDAT_M424 = """M424test                                       CATALOG
T  T   0   100                                 OPTNEUPDATE,HE_ONE,ITSTART,ITMORE
0.                                             OPTMIXED
38230.0,     4.25000,    6.2100                TEFF, LOG G, RSTAR
120.,  0.6                                     RMAX, TMIN
1.0e-10,  0.1,  2500.00,   1.00000,  0.1       MDOT, VMIN(START), VINF, BETA, VDIV
0.100000,  2.00000                             YHE, IHE(START)
F T F T T                                      OPTMOD,OPTTLUCY,MEGAS,ACCEL,OPTCMF
10., 1.0, T  T                                 VTURB,METALLICITY,LINES,LINES_IN_MODEL
T F 1 2                                        ENATCOR, EXPANSION,SET_FIRST, SET_STEP
1., 0.1, 0.2                                   CLF, VCLSTART, VCLMAX
"""
CHECKS = ["V1_flux", "V1_flux_dEW", "V2_holdout", "V2_holdout_dEW", "V2_insample", "V2_insample_dEW", "V3_extrap",
          "V3_leaveout", "V4", "V5", "V6", "brute_vs_integrator", "brute_vs_stored", "ew_conservation",
          "library_vs_analytic", "library_fc_vs_analytic", "convention"]
N_MUT = 8000                     # points of the mutation runs (~5 s each; tolerances checked at n 8000 and 20 000)


# ----------------------------------------------------------------------------------------------
# imports
# ----------------------------------------------------------------------------------------------
def test_no_heavy_imports():
    code = ("import sys; sys.path.insert(0, {!r}); import ppmpy.synspec.testing; "
            "bad = [m for m in ('matplotlib', 'ppmpy.ppm', 'h5py') if m in sys.modules]; print(bad)").format(
                conftest.ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"


# ----------------------------------------------------------------------------------------------
# the analytic models
# ----------------------------------------------------------------------------------------------
def test_toy_lines_and_params():
    ls = tt.toy_lineset(3)
    assert ls.names == ["L4026", "L4200", "L4922"] and np.array_equal(ls.lref, [4026.22, 4199.90, 4921.93])
    ls10 = tt.toy_lineset(10)
    assert len(set(ls10.names)) == 10 and ls10.lref[9] == 4900.0
    mine = LineSet(["a"], [5000.0])
    assert tt.toy_lineset(mine) is mine
    with pytest.raises(ValueError):
        tt.toy_lineset(0)
    p3, p5 = tt.toy_line_params(3, seed=4), tt.toy_line_params(5, seed=4)
    assert p5[:3] == p3                                            # line j does not depend on the number of lines
    assert tt.toy_line_params(3, seed=4) == p3 and tt.toy_line_params(3, seed=5) != p3
    for p in p5:
        assert 0.25 <= p["A0"] <= 0.5 and 20 <= p["sig0"] <= 60 and 0.1 <= p["eta"] <= 0.5 and 1 <= p["p"] <= 2
        assert 0.5 * p["sig0"] <= p["gam"] <= p["sig0"] and abs(p["c0"]) <= 5 and abs(p["a1"]) <= 0.3


@pytest.mark.parametrize("grid", [tt.TOY_GRID, VelocityGrid()], ids=["toy", "m424"])
def test_toy_depth_and_continuum(grid):
    """Exact zero depth from 0.9 (vmax - vshift) on (what the FFT zero padding needs), 0 < depth < 1 in the line,
    smooth in T_eff', the broadcast shapes; the continuum at the first row by default."""
    y = grid.y
    Y = grid.vmax - grid.vshift
    teff = np.linspace(36000.0, 40500.0, 7)
    for p in tt.toy_line_params(3, seed=2):
        d = tt.toy_depth(teff, y, p, grid)
        assert d.shape == (teff.size, y.size)
        assert np.all(d[:, np.abs(y) >= 0.9 * Y] == 0.0)
        assert np.all(d[:, np.abs(y) <= 0.6 * Y] > 0.0) and d.max() < 1.0
        assert np.all(d[:, np.abs(y) < 5.0] > 0.01)                 # a real line at the centre
        assert grid.check_zero_padding(d, 0.0)["ok"]
        h = 10.0                                                       # smooth in T_eff': tiny second differences
        d2 = tt.toy_depth(teff + h, y, p, grid) - 2 * d + tt.toy_depth(teff - h, y, p, grid)
        assert np.abs(d2).max() < 5e-5
        assert tt.toy_depth(38000.0, y, p, grid).shape == y.shape
        rows = np.tile(np.linspace(-Y, Y, 11), (teff.size, 1))
        assert tt.toy_depth(teff, rows, p, grid).shape == rows.shape
        fc = tt.toy_continuum(teff, p, grid=grid)
        assert fc.shape == teff.shape and np.all(np.diff(fc) > 0)     # rises with T_eff'
        np.testing.assert_allclose(fc, tt.toy_continuum(teff, p, np.full((teff.size, 1), -Y), grid)[:, 0], rtol=1e-15)
        assert tt.toy_continuum(teff, p, rows, grid).shape == rows.shape
    with pytest.raises(ValueError):
        tt.toy_depth(38000.0, y, p, VelocityGrid(vmax=300.0, vshift=300.0))


def test_toy_teff_range():
    lo, hi, dT = tt.toy_teff_range(40)
    assert (lo, hi, dT) == (37180.0, 39260.0, 52.0)
    assert lo % dT == 0 and hi - lo == 40 * dT
    assert tt.toy_teff_range(7)[2] == round(6 * 0.009 * 38230.0 / 7)
    t = np.linspace(lo + 0.5 * dT, hi - 0.5 * dT, 1000)
    np.testing.assert_array_equal(lb.teff_edges(t, dT), lo + dT * np.arange(41))
    with pytest.raises(ValueError):
        tt.toy_teff_range(1)


def test_toy_library():
    lib = tt.toy_library()
    lo, hi, dT = tt.toy_teff_range(40)
    assert (lib.nb, lib.nl, lib.ny, lib.dT) == (40, 3, tt.TOY_GRID.ny, dT) and lib.prof.dtype == np.float32
    assert np.all(lib.count == 25) and lib.edges[0] == lo and lib.edges[-1] == hi
    np.testing.assert_array_equal(lib.tmean, lib.centres)
    assert tt.TOY_GRID.check_zero_padding(1.0 - lib.prof, 0.0)["ok"]
    nodes = lb.lib_nodes(lib, nmin=20)
    assert nodes.nn == 40 and np.array_equal(nodes.t, lib.centres)
    DiscFlux(nodes, tt.TOY_GRID, pad_tol=0.0)                        # exact zeros beyond vmax - vshift
    from ppmpy.synspec.dumps import flux_integrator
    fx = flux_integrator(lib, grid=tt.TOY_GRID)
    np.testing.assert_array_equal(fx.lref, tt.toy_lineset(3).lref)
    with pytest.raises(ValueError, match="grid"):
        flux_integrator(lib, grid=VelocityGrid())                    # the library records its grid
    p = tt.toy_line_params(3)
    np.testing.assert_array_equal(lib.fc[:, 1], tt.toy_continuum(lib.centres, p[1]))
    np.testing.assert_array_equal(lib.prof[:, 2], (1.0 - tt.toy_depth(lib.centres, tt.TOY_GRID.y, p[2])).astype(
        np.float32))
    big = tt.toy_library(nnode=12, lines=2, grid=VelocityGrid(), seed=3)
    assert big.prof.shape == (12, 2, 5401)
    DiscFlux(lb.lib_nodes(big), VelocityGrid(), pad_tol=0.0)


def _nn_roughness(s, keys=("teff", "ur", "uth", "uph")):
    """Mean |f(i) - f(nearest neighbour of i)| / rms of f, per field."""
    from scipy.spatial import cKDTree
    xyz = np.column_stack(sph.sphere_xyz(s["theta"], s["phi"]))
    nn = cKDTree(xyz).query(xyz, k=2)[1][:, 1]
    return {k: float(np.mean(np.abs(s[k] - s[k][nn])) / s[k].std()) for k in keys}


def test_toy_sphere():
    n = 20000
    s = tt.toy_sphere(n, seed=0, dump=1)
    th, ph = sph.fibonacci_sphere(n)
    assert np.array_equal(s["theta"], th) and np.array_equal(s["phi"], ph)
    assert s["t_s"] == 2835.0 and s["dump"] == 1
    sig = tt.TEFF_REL_RMS * tt.TEFF0
    assert abs(s["teff"].mean() - tt.TEFF0) < 10.0 and 0.95 < s["teff"].std() / sig < 1.05
    for k in ("ur", "uth", "uph"):
        assert 44.0 < s[k].std() < 46.5, k
    assert max(_nn_roughness(s).values()) < 0.08                       # smooth: neighbours nearly equal
    # the plume: a compact cool downflow, a few points beyond the toy library's nodes
    lo, hi, dT = tt.toy_teff_range(40)
    assert s["teff"].min() < tt.TEFF0 - 4 * sig and s["ur"].min() < -120.0
    for d in (1, 2, 3):
        sd = tt.toy_sphere(n, seed=0, dump=d)
        nout = int(((sd["teff"] < lo + 0.5 * dT) | (sd["teff"] > hi - 0.5 * dT)).sum())
        assert 1 <= nout <= n // 200, (d, nout)
    flat = tt.toy_sphere(n, seed=0, dump=1, plume=False)
    assert tt.TEFF0 - 3.5 * sig < flat["teff"].min() and flat["teff"].max() < tt.TEFF0 + 3.5 * sig
    assert np.array_equal(flat["uth"], s["uth"]) and not np.array_equal(flat["ur"], s["ur"])
    # deterministic; other dumps and seeds differ
    s2 = tt.toy_sphere(n, seed=0, dump=1)
    assert all(np.array_equal(s[k], s2[k]) for k in ("teff", "ur", "uth", "uph"))
    assert not np.array_equal(tt.toy_sphere(n, seed=0, dump=2)["teff"], s["teff"])
    assert not np.array_equal(tt.toy_sphere(n, seed=1, dump=1)["teff"], s["teff"])
    # the library dump: uniform over the range, every bin equally filled, still smooth
    u = tt.toy_sphere(n, seed=0, teff_range=(lo, hi))
    assert lo < u["teff"].min() and u["teff"].max() < hi
    np.testing.assert_array_equal(np.bincount(lb.teff_bins(u["teff"], lo + dT * np.arange(41)), minlength=40),
                                  np.full(40, n // 40))
    assert _nn_roughness(u, ("teff",))["teff"] < 0.08
    assert tt.toy_sphere(10, t_s=5.0)["t_s"] == 5.0
    with pytest.raises(ValueError):
        tt.toy_sphere(10, teff_range=(2.0, 1.0))
    for bad in (0, 1):                                                # standardised over the points: needs 2
        with pytest.raises(ValueError, match="at least 2"):
            tt.toy_sphere(bad)
    two = tt.toy_sphere(2, plume=False)
    assert all(np.all(np.isfinite(two[k])) for k in ("teff", "ur", "uth", "uph"))


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_toy_profiles_store(tmp_path):
    lo, hi, dT = tt.toy_teff_range(10)
    s = tt.toy_sphere(4000, seed=0, teff_range=(lo, hi))
    p = tt.toy_profiles_store(str(tmp_path / "p.npz"), s)
    st = fw.ProfileStore.open(p, strict=True)                        # uncompressed: every member memory-mapped
    # the member order of fwresults.combine and of the M424 profiles.npz ('lines' after the row members)
    assert list(st.keys()) == ["idx", "teff", "status", "niter", "T_tau23", "t_pnlte", "t_formal", "lam", "fcont",
                               "fnorm", "r", "theta", "phi", "x", "y", "z", "ur_kms", "relT", "teff_nudge", "lines"]
    import zipfile
    with zipfile.ZipFile(p) as zf:
        assert [i.filename for i in zf.infolist()][-2:] == ["lines.npy", "_meta.npy"]
    dt = {k: np.asarray(st[k]).dtype.str for k in st.keys()}
    assert dt["idx"] == "<i4" and dt["niter"] == "<i2" and dt["lam"] == dt["fnorm"] == dt["fcont"] == "<f4"
    assert dt["teff"] == "<f8" and dt["teff_nudge"] == "<f4" and dt["status"] == "<U12"     # M424's status width
    assert st.n == 4000 and st.lines == tt.toy_lineset(3).names and st.nrow == tt.NROW
    assert np.array_equal(st.idx, np.arange(4000)) and np.all(np.asarray(st.status) == "ok")
    assert np.array_equal(st.teff, s["teff"]) and not np.any(st.teff_nudge)
    np.testing.assert_array_equal(st["x"], sph.sphere_xyz(s["theta"], s["phi"])[0])
    assert read_meta(p)["kind"] == "synspec.toy_profiles" and st.meta["params"]["noise"] == 0.003
    lam, fn = np.asarray(st["lam"]), np.asarray(st["fnorm"])
    assert np.all(np.diff(lam, axis=2) > 0)                           # increasing rows
    assert not np.array_equal(lam[0, 0], lam[1, 0])                   # rows differ from model to model
    Y = tt.TOY_GRID.vmax - tt.TOY_GRID.vshift
    for j, lr in enumerate(tt.toy_lineset(3).lref):
        yr = y_of_lam(lam[:, j], lr)
        assert np.abs(yr[:, 0] + Y).max() < 1.0 and np.abs(yr[:, -1] - Y).max() < 1.0
        assert np.all(fn[:, j][np.abs(yr) >= 0.9 * Y + 0.1] == 1.0)    # exactly the continuum beyond the taper
    assert np.all(np.asarray(st["fcont"]) > 0)
    # the depth scatter: per model and line a constant factor, rms = noise
    p0 = tt.toy_profiles_store(str(tmp_path / "p0.npz"), s, noise=0.0, meta=False)
    f0 = np.asarray(fw.ProfileStore.open(p0)["fnorm"]).astype(np.float64)
    deep = (1.0 - f0) > 0.05
    ratio = np.where(deep, (1.0 - fn) / np.where(deep, 1.0 - f0, 1.0), np.nan)
    eps = np.nanmean(ratio, axis=2) - 1.0
    assert np.nanmax(np.nanstd(ratio, axis=2)) < 1e-5                  # constant over the rows (float32)
    assert 0.0027 < np.std(eps) < 0.0033
    # without noise the models are the analytic lines on their rows (to float32 rounding of lambda)
    for j, par in enumerate(tt.toy_line_params(3)):
        yr = y_of_lam(np.asarray(fw.ProfileStore.open(p0)["lam"])[:50, j], par["lref"])
        np.testing.assert_allclose(f0[:50, j], 1.0 - tt.toy_depth(s["teff"][:50], yr, par), rtol=0, atol=1e-3)
    # the file does not depend on the block size
    a = tt.toy_profiles_store(str(tmp_path / "a.npz"), s, meta=False, block=7)
    b = tt.toy_profiles_store(str(tmp_path / "b.npz"), s, meta=False, block=4000)
    assert _sha(a) == _sha(b)
    # the library built from it agrees with the analytic one (bin means vs centres, the per-model scatter)
    L = lb.FluxLibrary.build(st.teff, st["lam"], st["fnorm"], st.fc(None), tt.TOY_GRID, tt.toy_lineset(3), dT=dT)
    A = tt.toy_library(10)
    np.testing.assert_array_equal(L.edges, A.edges)
    np.testing.assert_array_equal(L.count, np.full(10, 400.0))
    np.testing.assert_allclose(L.tmean, A.tmean, rtol=0, atol=1e-9)
    np.testing.assert_allclose(L.fc, A.fc, rtol=2e-4, atol=0)
    assert np.abs(L.prof.astype(np.float64) - A.prof).max() < 2e-3
    with pytest.raises(ValueError):
        tt.toy_profiles_store(str(tmp_path / "x.npz"), dict(s, teff=s["teff"][:10]))
    assert not os.path.exists(str(tmp_path / "x.npz")) and not [f for f in os.listdir(str(tmp_path)) if "tmp" in f]


def test_toy_run(tmp_path):
    """A small run: files in place, nnode nodes, the integrator reproduces the stored per-dump products (float32)
    and the time series; the exact sums of the library dump agree with the integrator to the toy's V1 level."""
    from ppmpy.synspec.dumps import load_sample
    run = tt.toy_run(str(tmp_path / "run"), n=4000, nnode=20, ndumps=2)
    for k in ("profiles", "sample0", "library", "exact", "timeseries"):
        assert os.path.isfile(run[k]), k
    assert run["dumps"] == [0, 1, 2] and run["later"] == [1, 2] and run["nodes"].nn == 20
    MU, TN, PN = sph.project_los(run["theta"], run["phi"], run["los"])
    with np.load(run["timeseries"]) as z:
        F_ts = z["F"]
        assert F_ts.shape == (3, 8, 3, tt.TOY_GRID.ny)
    for d in run["dumps"]:
        s = load_sample(run["samples"], d)
        k0, k1, a = run["integ"].pairs(s["teff"])
        with np.load(os.path.join(run["root"], "flux", "d{:04d}.npz".format(d))) as z:
            for k in range(8):
                v = sph.los_velocity(s["ur"], s["uth"], s["uph"], MU[k], TN[k], PN[k])
                F = run["integ"](MU[k], v, k0, k1, a)[0]
                np.testing.assert_array_equal(F.astype(np.float32), z["F"][k])
            np.testing.assert_array_equal(z["F"], F_ts[d])
    s0 = load_sample(run["samples"], 0)
    with np.load(run["profiles"]) as z:
        assert np.abs(s0["teff"] - z["teff"]).max() < 0.01                            # float32 of the models' T_eff'
    with np.load(run["exact"]) as z, np.load(os.path.join(run["root"], "flux", "d0000.npz")) as f:
        assert np.abs(z["F"].astype(np.float32) - f["F"]).max() < 3e-4


# ----------------------------------------------------------------------------------------------
# the self-test
# ----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def serial():
    return tt.selftest()


def test_selftest_passes(serial):
    """The default self-test (n 20 000, quick, seed 0): every check passes, in < 60 s and < 2 GB."""
    r = serial
    print(r.table())
    assert r.passed(), r.table()
    assert r.names() == CHECKS
    assert r["V3_leaveout"].passed is None and r["V5"].value == 0.0
    assert r["brute_vs_integrator"].value < 1e-13 and r["brute_vs_stored"].value <= 2.0 ** -25 * (1 + 1e-9)
    assert r.arrays["V3_n"].sum() > 0                                 # points beyond the nodes: V3 checks something
    st = r.meta["selftest"]
    assert st["nodes"] == 40 and st["n"] == 20000 and st["root"] is None and st["mutate"] is None
    assert st["leftover"] is None and st["nmins"] == [1, 5, 100]
    assert st["tol_scale"] == dict(interp=1.0, interp2=1.0, noise=1.0, dT=52.0)
    assert st["wall"] < 60.0 and st["maxrss_gb"] < 2.0, st
    assert r.meta["skipped"] == {} and r.meta["tolerances"]["V1"] == tt.TOY_TOLERANCES["V1"]
    assert r.meta["tolerances"]["convention"] == tt.TOY_TOLERANCES["convention"]
    for c in r:                                                       # well inside the toy's tolerances
        if c.tolerance:
            assert c.value <= 0.5 * c.tolerance, c
    assert "17 checks: 16 passed, 0 failed, 1 informational" in r.table()
    # the analytic anchors: Lambert's 2/3 and the blueshift to ~1e-4, the hidden hemisphere contributes nothing
    cv = r["convention"]
    assert cv.unit == "relative" and cv.value < 1e-3 and np.all(np.asarray(cv.details["hemisphere"]) == 0.0)
    np.testing.assert_allclose(cv.details["vmean"], 200.0 / 3.0, rtol=1e-4)
    np.testing.assert_allclose(cv.details["dc"], -200.0 / 3.0, rtol=1e-3)
    assert r["library_vs_analytic"].value < 2e-3 and r["library_fc_vs_analytic"].value < 1e-5


def test_selftest_parallel(serial):
    """nproc 2: the parallel exact sums and library, per-dump products and brute force equal the serial ones bit for
    bit with 'fork' and 'spawn'; every other check has the same value as the serial self-test."""
    r = tt.selftest(nproc=2)
    assert r.passed(), r.table()
    assert r.names() == CHECKS + ["parallel_fork", "parallel_spawn"]
    for m in ("fork", "spawn"):
        c = r["parallel_" + m]
        assert c.value == 0.0 and c.tolerance == 0.0 and c.passed
        assert c.details["exact"] == c.details["dumps"] == c.details["brute"] == 0.0 and c.details["nproc"] == 2
    for c in serial:
        assert r[c.name].value == c.value or (np.isnan(c.value) and np.isnan(r[c.name].value)), c.name
    assert r.meta["selftest"]["wall"] < 60.0


def test_selftest_other_seed():
    r = tt.selftest(n=N_MUT, seed=1)
    assert r.passed(), r.table()


@pytest.mark.parametrize("n, nnode", [(3000, 40), (3900, 40), (N_MUT, 100), (N_MUT, 10)])
def test_selftest_other_sizes(n, nnode):
    """Intact runs pass outside the calibration point: V5 uses only the nmins every bin satisfies (n // nnode < 100
    gave V5 1.6e-5 at n 3900 and 4.0e-6 at n 8000 / nnode 100), the tolerances scale with n and the node spacing
    (nnode 10 failed V1, V3, V4 at up to 1.9 x before)."""
    r = tt.selftest(n=n, nnode=nnode)
    assert r.passed(), r.table()
    st = r.meta["selftest"]
    assert st["nmins"] == [m for m in (1, 5, 100) if m <= n // nnode] and r["V5"].value == 0.0
    assert r["V5"].details["nmins"] == st["nmins"]
    tol, sc = tt.toy_tolerances(n, nnode)
    assert st["tol_scale"] == sc and r["V1_flux"].tolerance == tol["V1"]
    if n < tt.TOY_N_CAL:
        assert sc["noise"] == pytest.approx(np.sqrt(tt.TOY_N_CAL / n)) and sc["interp"] == 1.0
    if nnode == 10:
        assert sc["dT"] == 206.0 and sc["interp2"] == pytest.approx((206.0 / 52.0) ** 2)
        assert r["V4"].tolerance == pytest.approx(tt.TOY_TOLERANCES["V4"] * (206.0 / 52.0) ** 2)
    for c in r:
        if c.tolerance:
            assert c.value <= 0.7 * c.tolerance, c


def test_selftest_size_guards():
    with pytest.raises(ValueError, match="n >= 20 nnode"):
        tt.selftest(n=700, nnode=40)                                  # 17 models per bin: nodes would merge
    with pytest.warns(UserWarning, match="outside the checked range"):
        r = tt.selftest(n=2000, nnode=40)
    assert r.meta["selftest"]["nmins"] == [1, 5]
    tol, sc = tt.toy_tolerances(20000, 40, dict(V1=0.5))
    assert tol["V1"] == 0.5 and tol["V2"] == tt.TOY_TOLERANCES["V2"] and sc["noise"] == 1.0


@pytest.mark.parametrize("mutate", sorted(tt.MUTATIONS))
def test_selftest_mutations(mutate):
    """Each fault fails the report and its 'fail' checks; the 'blind' checks (which compare the faulty pipeline with
    itself, or intact references with each other) pass (module data SELFTEST_EXPECT)."""
    r = tt.selftest(n=N_MUT, mutate=mutate)
    exp = tt.SELFTEST_EXPECT[mutate]
    failed = {c.name for c in r.failed()}
    assert not r.passed()
    assert exp["fail"] <= failed, (mutate, sorted(exp["fail"] - failed), r.table())
    assert not (exp["blind"] & failed), (mutate, sorted(exp["blind"] & failed), r.table())
    assert set(tt.SELFTEST_EXPECT) == set(tt.MUTATIONS) and not (exp["fail"] & exp["blind"])
    assert r.meta["selftest"]["mutate"] == mutate


def _patch_everywhere(monkeypatch, module, name, new):
    """Replace a function in its module and in every ppmpy.synspec module that imported it by name."""
    orig = getattr(module, name)
    for mn, m in list(sys.modules.items()):
        if mn.startswith("ppmpy.synspec") and getattr(m, name, None) is orig:
            monkeypatch.setattr(m, name, new)


def _flip_v(monkeypatch):
    orig = sph.los_velocity
    _patch_everywhere(monkeypatch, sph, "los_velocity", lambda *a: -orig(*a))


def _fc0_ones(monkeypatch):
    from ppmpy.synspec import disc
    _patch_everywhere(monkeypatch, disc, "_fc0", lambda fcont, block=100000: np.ones(np.shape(fcont)[:2]))


def _abs_mu(monkeypatch):
    orig = sph.project_los

    def project_los(*a, **k):
        MU, TN, PN = orig(*a, **k)
        return np.abs(MU), TN, PN
    _patch_everywhere(monkeypatch, sph, "project_los", project_los)


ANCHOR_FAULTS = {"los_velocity_sign": (_flip_v, {"convention"}),
                 "fc0_ones": (_fc0_ones, {"library_fc_vs_analytic"}),
                 "project_los_abs_mu": (_abs_mu, {"convention"})}


@pytest.mark.parametrize("fault", sorted(ANCHOR_FAULTS))
def test_selftest_anchor_faults(monkeypatch, fault):
    """Faults in the shared helpers, applied to the whole pipeline (references included): the line-of-sight velocity
    with the wrong sign, the F_c weight dropped (fcont[:, :, 0] -> 1), the far hemisphere visible (|mu|). The checks
    that compare the pipeline with references made by the same helpers pass; the analytic anchors fail."""
    setup, expect = ANCHOR_FAULTS[fault]
    setup(monkeypatch)
    r = tt.selftest(n=N_MUT)
    failed = {c.name for c in r.failed()}
    assert not r.passed() and expect <= failed, (fault, r.table())
    blind = {"V1_flux", "V2_holdout", "V2_insample", "V4", "V5", "V6", "brute_vs_integrator", "brute_vs_stored",
             "ew_conservation"}
    assert not (blind & failed), (fault, sorted(blind & failed))      # what the reviewer found: invisible before
    if fault == "los_velocity_sign":
        assert r["convention"].value == pytest.approx(2.0, rel=1e-3)
    if fault == "project_los_abs_mu":
        assert np.max(r["convention"].details["hemisphere"]) > 1e-2


def test_selftest_options(tmp_path):
    with pytest.raises(ValueError, match="unknown mutation"):
        tt.selftest(mutate="nope")
    r = tt.selftest(n=4000, keep=True, workdir=str(tmp_path), tolerances=dict(V1=0.0))
    root = r.meta["selftest"]["root"]
    assert os.path.isdir(root) and os.path.isfile(os.path.join(root, "flux_timeseries.npz"))
    assert r["V1_flux"].passed is False and r["V1_flux"].tolerance == 0.0 and not r.passed()
    rj = type(r).from_json(r.to_json(str(tmp_path / "rep.json")))
    assert rj.table() == r.table() and rj.meta["selftest"]["n"] == 4000
    r.to_npz(str(tmp_path / "rep.npz"))
    # without keep nothing stays behind
    before = set(os.listdir(str(tmp_path)))
    tt.selftest(n=4000, workdir=str(tmp_path))
    assert set(os.listdir(str(tmp_path))) == before


def test_selftest_cleanup_on_error(tmp_path, monkeypatch):
    """An error inside the run: the directory is removed (memory maps of its files held by the traceback's frames
    are released first), the error propagates with its traceback; with keep=True the directory stays."""
    def failing_run(root, **kw):
        path = os.path.join(root, "a.npy")
        np.save(path, np.arange(1000.0))
        held = np.load(path, mmap_mode="r")                           # an open memory map in the failing frame
        assert held[3] == 3.0
        raise RuntimeError("toy_run failed")
    monkeypatch.setattr(tt, "toy_run", failing_run)
    wd = tmp_path / "wd"
    wd.mkdir()
    with pytest.raises(RuntimeError, match="toy_run failed") as ei:
        tt.selftest(n=4000, workdir=str(wd))
    assert os.listdir(str(wd)) == []
    assert any(e.name == "failing_run" for e in ei.traceback)        # the traceback keeps its lines
    with pytest.raises(RuntimeError):
        tt.selftest(n=4000, workdir=str(wd), keep=True)
    kept = os.listdir(str(wd))
    assert len(kept) == 1 and os.path.isfile(os.path.join(str(wd), kept[0], "a.npy"))


def test_selftest_leftover_warns(tmp_path, monkeypatch):
    """A directory that cannot be removed: a UserWarning with the path, meta['selftest']['leftover']."""
    import shutil
    monkeypatch.setattr(tt, "_remove_tree", lambda root, tries=5: False)
    with pytest.warns(UserWarning, match="could not remove the run directory"):
        r = tt.selftest(n=4000, workdir=str(tmp_path))
    left = r.meta["selftest"]["leftover"]
    assert r.passed() and r.meta["selftest"]["root"] is None and os.path.isdir(left)
    assert os.path.dirname(left) == str(tmp_path)
    shutil.rmtree(left)


def test_parallel_checks_detect(tmp_path):
    """The parallel checks compare every numeric member of the per-dump products and the library's count, tmean and
    edges: a perturbed stored member (here vmean_w, then F) is reported."""
    run = tt.toy_run(str(tmp_path / "run"), n=4000, nnode=20, ndumps=1)
    proj = sph.project_los(run["theta"], run["phi"], run["los"])
    c = tt._parallel_checks(run, proj, 2, ("fork",))[0]
    assert c.passed and c.value == 0.0
    assert {"F", "F0", "diag_F", "vmean_w", "sigma_w", "n_clip", "wout"} <= set(c.details["dump_members"])
    assert c.details["library_members"] == ["prof", "fc", "count", "tmean", "edges"]
    path = os.path.join(run["root"], "flux", "d0001.npz")

    def perturb(key, delta):
        with np.load(path) as z:
            d = {k: z[k] for k in z.files}
        d[key] = (d[key] + delta).astype(d[key].dtype)
        np.savez(path, **d)
    perturb("vmean_w", 1e-9)
    c = tt._parallel_checks(run, proj, 2, ("fork",))[0]
    assert not c.passed and c.details["dumps"] > 0 and c.details["exact"] == 0.0 and c.details["brute"] == 0.0
    perturb("vmean_w", -1e-9)
    perturb("F", 1e-7)
    c = tt._parallel_checks(run, proj, 2, ("fork",))[0]
    assert not c.passed and c.value >= 1e-7 and c.details["dumps"] >= 1e-7
    # the library: tmean of the stored library perturbed
    lib = lb.FluxLibrary.load(run["library"])
    lib.tmean = lib.tmean + 1e-6
    lib.save(run["library"])
    c = tt._parallel_checks(run, proj, 2, ("fork",))[0]
    assert not c.passed and c.details["exact"] > 0


def test_maxdiff():
    assert tt._maxdiff([1.0, np.nan, np.inf], [1.0, np.nan, np.inf]) == 0.0
    assert tt._maxdiff([1.0, np.nan], [1.0, 2.0]) == np.inf
    assert tt._maxdiff([1.0, 2.0], [1.0, 2.5]) == 0.5 and tt._maxdiff([1.0], [1.0, 2.0]) == np.inf
    assert tt._maxdiff([], []) == 0.0 and tt._maxdiff(np.float64(2.0), 2.0) == 0.0 and tt._maxdiff(1, 3) == 2.0


def test_main(capsys, tmp_path):
    assert tt.main(["--n", str(N_MUT), "--json", str(tmp_path / "r.json")]) == 0
    out = capsys.readouterr().out
    assert "selftest: PASSED" in out and "ew_conservation" in out and os.path.isfile(str(tmp_path / "r.json"))
    assert tt.main(["--n", str(N_MUT), "--mutate", "v_sign"]) == 1
    assert "selftest: FAILED" in capsys.readouterr().out


@pytest.mark.slow
def test_selftest_full():
    """quick=False: the M424 grid (5401 points), 6 later dumps, V4 / V6 on 4 dumps (5000-point subsets)."""
    r = tt.selftest(quick=False)
    print(r.table())
    assert r.passed(), r.table()
    st = r.meta["selftest"]
    assert st["grid"] == VelocityGrid().to_dict() and st["nsub"] == 5000 and st["check_dumps"] == [1, 2, 3, 4]
    assert r.arrays["V4"].shape == (4, 3) and r.arrays["brute_dumps"].tolist() == [1, 2]


# ----------------------------------------------------------------------------------------------
# check_indat_premise
# ----------------------------------------------------------------------------------------------
def _indat(idx, teff, template=INDAT_M424):
    """INDAT.DAT of one point as fw_sphere_point.sh writes it: line 1 'P<idx>' padded to 47 columns + 'CATALOG',
    the first value of line 4 replaced by the T_eff with '%.3f' (awk sprintf of the command-line value)."""
    lines = template.splitlines()
    lines[0] = "{:<47s}CATALOG".format("P{:06d}".format(idx))
    lines[3] = "{:.3f},".format(float(teff)) + lines[3].split(",", 1)[1]
    return "\n".join(lines) + "\n"


def _tar_add(tf, name, data):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    ti.mtime = 1758800000
    tf.addfile(ti, io.BytesIO(data))


def _write_part(path, records):
    """records: (idx, teff, indat text or None, status); a point also gets a dummy model file, as the M424 parts.
    meta.txt holds teff verbatim when it is a str (fw_sphere_point.sh: awk '%s' of the command-line value), else
    with '%.3f' (as points.txt and missing.txt)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tarfile.open(path, "w:gz") as tf:
        for idx, teff, indat, status in records:
            d = "./P{:06d}".format(idx)
            tt_ = teff if isinstance(teff, str) else "{:.3f}".format(teff)
            _tar_add(tf, d + "/meta.txt", "{} {} {} 61 39000.5 150.2 0.5\n".format(idx, tt_, status).encode())
            _tar_add(tf, d + "/MODEL", b"x" * 64)
            if indat is not None:
                _tar_add(tf, d + "/INDAT.DAT", indat.encode())


def _good_run(root, edit=None):
    """results/ with 3 tags, 8 points; edit(idx, teff) -> INDAT text or None changes single points."""
    res = os.path.join(str(root), "results")
    teff = {i: 38230.25 + 113.375 * (i - 4) for i in range(8)}          # none equal to the template's 38230.0

    def rec(i, nudge=0.0, status="ok"):
        t = teff[i] + nudge
        text = edit(i, t) if edit is not None else False
        return (i, t, _indat(i, t) if text is False else text, status)
    _write_part(os.path.join(res, "task_0000", "part_1.tar.gz"), [rec(0), rec(1), rec(2, status="pnlte_failed")])
    _write_part(os.path.join(res, "task_0000", "part_2.tar.gz"), [rec(3), rec(4)])
    _write_part(os.path.join(res, "task_0001", "part_1.tar.gz"), [rec(5), rec(6)])
    _write_part(os.path.join(res, "task_missing_0000", "part_1.tar.gz"), [rec(2, 1.0), rec(7)])
    return res, teff


def test_indat_fields():
    f = fw._indat_fields(INDAT_M424)
    names = [n for line in fw.INDAT_SCHEMA for n in line]
    assert list(f)[:len(names)] == names and list(f)[len(names):] == ["LINE11"]
    assert f["MODNAM"] == ("M424test", "M424test") and f["TEFF"] == (38230.0, "38230.0")
    assert f["OPTNEUPDATE"] == (True, "T") and f["MEGAS"] == (False, "F") and f["ITMORE"] == (100.0, "100")
    assert f["MDOT"][0] == 1e-10 and f["LINE11"][0] == (1.0, 0.1, 0.2, "CLF", "VCLSTART", "VCLMAX")
    alt = (INDAT_M424.replace("1.0e-10,", "1.0D-10,").replace("4.25000", "4.25")
           .replace("F T F T T", ".FALSE. .TRUE. F T T"))
    assert fw._indat_diff(f, fw._indat_fields(alt)) == []             # the same values, written differently
    short = "\n".join(INDAT_M424.splitlines()[:3]) + "\n\n\n"
    g = fw._indat_fields(short)
    assert g["TEFF"] == (None, None) and "LINE11" not in g
    assert set(fw._indat_diff(f, g)) == set(names[names.index("TEFF"):]) | {"LINE11"}
    assert fw._indat_same(1.0, 1.0) and not fw._indat_same(1.0, True) and fw._indat_same(np.nan, np.nan)


def test_iter_part_points_needs_meta(tmp_path):
    """The quirk check_indat_premise works around: a point without meta.txt among the wanted files is dropped."""
    res, _ = _good_run(tmp_path)
    part = os.path.join(res, "task_0001", "part_1.tar.gz")
    assert list(fw.iter_part_points(part, want=("INDAT.DAT",))) == []
    assert [p for p, _ in fw.iter_part_points(part, want=("meta.txt", "INDAT.DAT"))] == ["P000005", "P000006"]


def test_check_indat_premise_passes(tmp_path):
    res, teff = _good_run(tmp_path)
    r = fw.check_indat_premise(res, template=INDAT_M424)
    assert r["passed"] and r["complete"] and r["require_indat"]
    assert r["n_points"] == 9 and r["n_unique"] == 8 and r["n_offending"] == 0
    assert r["n_premise_ok"] == 9 and r["n_no_indat"] == 0 and r["offending"] == []
    assert r["differ"] == {"MODNAM": 9, "TEFF": 9}
    assert r["tags"] == ["task_0000", "task_0001", "task_missing_0000"] and len(r["parts"]) == 4
    assert r["status"] == {"ok": 8, "pnlte_failed": 1}
    assert r["consistency"] == dict(modnam_mismatch=0, teff_meta_mismatch=0, teff_meta_maxdiff=0.0,
                                    teff_range=[min(teff.values()), max(teff.values())])
    assert r["reference"]["source"] == "template" and r["reference"]["fields"]["LOGG"] == "4.25000"
    # the template as a file, the first INDAT as the reference, tags, max_parts
    p = tmp_path / "INDAT.DAT"
    p.write_text(INDAT_M424)
    rp = fw.check_indat_premise(res, template=str(p))
    assert rp["reference"]["path"] == str(p) and rp["differ"] == r["differ"]
    rf = fw.check_indat_premise(res)
    assert rf["passed"] and rf["reference"]["source"] == "first" and rf["reference"]["idx"] == 0
    assert rf["differ"] == {"MODNAM": 8, "TEFF": 8}                    # all but point 0 itself
    rt = fw.check_indat_premise(res, tags="task_000*", max_parts=1, template=INDAT_M424)
    assert rt["tags"] == ["task_0000", "task_0001"] and rt["n_points"] == 5 and len(rt["parts"]) == 2
    re_ = fw.check_indat_premise(res, tags=["task_0001", "nothing"], template=INDAT_M424)
    assert re_["empty_tags"] == ["nothing"] and re_["n_points"] == 2 and re_["passed"]
    # in parallel: the same result
    r2 = fw.check_indat_premise(res, template=INDAT_M424, nproc=2)
    assert {k: v for k, v in r2.items() if k != "wall"} == {k: v for k, v in r.items() if k != "wall"}


def test_check_indat_premise_detects(tmp_path):
    """A changed LOGG (and other fields, an extra line) is reported with the point; MODNAM / TEFF other than the
    directory / meta.txt are counted as inconsistencies; missing INDATs are counted."""
    def edit(i, t):
        text = _indat(i, t)
        if i == 3:
            return text.replace("4.25000", "4.30000")                  # log g
        if i == 5:
            return text.replace("2500.00", "2400.00").replace("4.25000", "4.2")   # v_inf and log g
        if i == 6:
            return text + "XRAYS 0.1\n"                                # an extra line
        if i == 1:
            return None                                                # no INDAT.DAT
        if i == 7:
            return text.replace("P000007", "P000099").replace("{:.3f}".format(t), "{:.3f}".format(t + 5.0))
        return text
    res, _ = _good_run(tmp_path, edit)
    r = fw.check_indat_premise(res, template=INDAT_M424)
    assert not r["passed"] and r["n_offending"] == 3 and r["n_points"] == 8 and r["n_no_indat"] == 1
    assert not r["complete"]
    off = {o["idx"]: o for o in r["offending"]}
    assert off[3]["fields"] == {"LOGG": ["4.25000", "4.30000"]} and off[3]["pdir"] == "P000003"
    assert off[3]["tag"] == "task_0000" and off[3]["part"].endswith("task_0000/part_2.tar.gz")
    assert off[5]["fields"] == {"LOGG": ["4.25000", "4.2"], "VINF": ["2500.00", "2400.00"]}
    assert off[6]["fields"] == {"LINE12": [None, "XRAYS 0.1"]}
    assert [o["idx"] for o in r["offending"]] == [3, 5, 6]              # in part order
    assert r["differ"]["LOGG"] == 2 and r["differ"]["VINF"] == 1 and r["differ"]["LINE12"] == 1
    assert list(r["differ"])[:2] == ["MODNAM", "TEFF"]
    assert r["consistency"]["modnam_mismatch"] == 1 and r["consistency"]["teff_meta_mismatch"] == 1
    assert abs(r["consistency"]["teff_meta_maxdiff"] - 5.0) < 1e-9
    assert len(fw.check_indat_premise(res, template=INDAT_M424, max_report=1)["offending"]) == 1
    # TEFF not allowed: every model offends
    rt = fw.check_indat_premise(res, template=INDAT_M424, fields_allowed=("MODNAM",))
    assert rt["n_offending"] == rt["n_points"] == 8
    # LOGG allowed too: only the v_inf and the extra line remain
    rl = fw.check_indat_premise(res, template=INDAT_M424, fields_allowed=("MODNAM", "TEFF", "LOGG"))
    assert sorted(o["idx"] for o in rl["offending"]) == [5, 6]


def _consistency_run(root, edit):
    """_good_run with ``edit`` applied, checked against the template: (result, result without require_indat)."""
    res, _ = _good_run(root, edit)
    return (fw.check_indat_premise(res, template=INDAT_M424),
            fw.check_indat_premise(res, template=INDAT_M424, require_indat=False))


@pytest.mark.parametrize("case", ["teff_off", "modnam_other", "teff_text", "teff_nan", "modnam_missing"])
def test_check_indat_premise_consistency_fails(tmp_path, case):
    """Only a consistency check fails (no field outside MODNAM / TEFF differs from the template): the verdict is
    False. A TEFF 50 K off meta.txt's puts the model in the library under the wrong T_eff'; a MODNAM of another
    point is that point's INDAT; a TEFF that is not a finite number, or a missing MODNAM, cannot be checked."""
    def edit(i, t):
        text = _indat(i, t)
        if i != 4:
            return text
        if case == "teff_off":
            return text.replace("{:.3f},".format(t), "{:.3f},".format(t + 50.0))
        if case == "modnam_other":
            return text.replace("P000004", "P000003")
        if case == "teff_text":
            return text.replace("{:.3f},".format(t), "T38230,")
        if case == "teff_nan":
            return text.replace("{:.3f},".format(t), "nan,")
        lines = text.splitlines()                                    # MODNAM missing: a blank first line
        lines[0] = ""
        return "\n".join(lines) + "\n"
    r, r_noreq = _consistency_run(tmp_path, edit)
    c = r["consistency"]
    assert r["n_points"] == 9 and r["n_no_indat"] == 0 and r["complete"]
    if case in ("modnam_missing", "modnam_other"):
        assert c["modnam_mismatch"] == 1 and c["teff_meta_mismatch"] == 0
    else:
        assert c["teff_meta_mismatch"] == 1 and c["modnam_mismatch"] == 0
    if case == "teff_off":
        assert abs(c["teff_meta_maxdiff"] - 50.0) < 1e-9
    if case in ("teff_text", "teff_nan"):
        assert c["teff_meta_maxdiff"] == 0.0                         # only finite TEFF enter the maximum
    assert r["n_offending"] == 0 and r["offending"] == []             # only MODNAM / TEFF differ
    assert not r["passed"] and not r_noreq["passed"]


def test_check_indat_premise_missing_indat(tmp_path):
    """Points without INDAT.DAT fail the check unless require_indat=False (then they are only counted)."""
    def edit(i, t):
        return _indat(i, t) if i == 0 else None                      # 8 of 9 records without INDAT.DAT
    r, r_noreq = _consistency_run(tmp_path, edit)
    assert r["n_points"] == 1 and r["n_no_indat"] == 8 and not r["complete"] and not r["passed"]
    assert r_noreq["passed"] and not r_noreq["complete"] and r_noreq["require_indat"] is False
    assert r_noreq["n_no_indat"] == 8


def test_check_indat_premise_teff_rounding(tmp_path):
    """meta.txt holds the command-line T_eff verbatim, INDAT its '%.3f' rounding: consistent for any number of
    decimals (also on an exact tie, 38230.0625 -> 38230.062), inconsistent for another rounding."""
    res = os.path.join(str(tmp_path), "results")
    recs = [(0, "38230.12345", _indat(0, 38230.12345), "ok"),
            (1, "38230.0625", _indat(1, 38230.0625), "ok"),
            (2, "38230.1", _indat(2, 38230.1), "ok"),
            (3, "38230.06251", _indat(3, 38230.06251), "ok")]
    assert "38230.062," in recs[1][2] and "38230.063," in recs[3][2]
    _write_part(os.path.join(res, "task_0000", "part_1.tar.gz"), recs)
    r = fw.check_indat_premise(res, template=INDAT_M424)
    assert r["passed"] and r["consistency"]["teff_meta_mismatch"] == 0
    # the tie differs by 5e-4 up to float parsing (5.00000002e-4: on the old threshold d > 5e-4 it would fail)
    assert 4.9e-4 < r["consistency"]["teff_meta_maxdiff"] < 5e-4 + 1e-8
    # an INDAT rounded the other way (38230.063 for 38230.0625) is not meta.txt's T_eff
    bad = [(1, "38230.0625", _indat(1, 38230.0625).replace("38230.062,", "38230.063,"), "ok")]
    _write_part(os.path.join(res, "task_0001", "part_1.tar.gz"), bad)
    r = fw.check_indat_premise(res, tags=["task_0001"], template=INDAT_M424)
    assert not r["passed"] and r["consistency"]["teff_meta_mismatch"] == 1


def test_check_indat_premise_glob_chars(tmp_path):
    """A results directory and tags with glob metacharacters are taken literally."""
    res, _ = _good_run(tmp_path / "run[1]")
    assert glob.escape(res) != res
    r = fw.check_indat_premise(res, template=INDAT_M424)
    assert r["passed"] and len(r["parts"]) == 4 and r["n_points"] == 9 and r["empty_tags"] == []
    os.rename(os.path.join(res, "task_0001"), os.path.join(res, "task_[1]"))
    r = fw.check_indat_premise(res, tags=["task_[1]"], template=INDAT_M424)
    assert r["passed"] and r["n_points"] == 2 and r["empty_tags"] == []


def test_check_indat_premise_serial_recycled(tmp_path, monkeypatch):
    """nproc=1 with more than INDAT_SERIAL_MAX parts goes through one recycled worker: the same result."""
    from ppmpy.synspec import parallel as par
    res, _ = _good_run(tmp_path)
    r = fw.check_indat_premise(res, template=INDAT_M424)
    calls, make_pool = [], par.make_pool

    def counting_pool(nproc, **kw):
        calls.append((nproc, kw.get("maxtasksperchild")))
        return make_pool(nproc, **kw)
    monkeypatch.setattr(par, "make_pool", counting_pool)
    assert fw.check_indat_premise(res, template=INDAT_M424, nproc=1)["passed"] and calls == []    # 4 parts <= 50
    monkeypatch.setattr(fw, "INDAT_SERIAL_MAX", 2)
    rp = fw.check_indat_premise(res, template=INDAT_M424, nproc=1)
    assert calls == [(1, 8)]                                          # one worker, replaced every 8 parts
    assert {k: v for k, v in rp.items() if k != "wall"} == {k: v for k, v in r.items() if k != "wall"}


def test_check_indat_premise_nothing(tmp_path):
    os.makedirs(str(tmp_path / "results" / "task_0000"))
    r = fw.check_indat_premise(str(tmp_path / "results"))
    assert not r["passed"] and r["n_points"] == 0 and r["parts"] == [] and r["reference"]["fields"] is None
    r = fw.check_indat_premise(str(tmp_path / "results"), tags=["task_0000"], template=INDAT_M424)
    assert not r["passed"] and r["empty_tags"] == ["task_0000"]


@pytest.mark.m424
@pytest.mark.slow
def test_m424_check_indat_premise():
    """The first part of task_0000, task_0039 and task_missing_0000 (1503 models; ~16 s with 3 workers): every
    archived INDAT.DAT differs from the template only in MODNAM (= the point's directory) and TEFF (= meta.txt's)."""
    res = m424_path("run", "results")
    for t in ("task_0000", "task_0039", "task_missing_0000"):
        m424_path("run", "results", t)
    r = fw.check_indat_premise(res, tags=["task_0000", "task_0039", "task_missing_0000"], max_parts=1,
                               template=INDAT_M424, nproc=3, log=print)
    print({k: r[k] for k in ("n_points", "differ", "status", "consistency", "wall")})
    assert r["passed"] and r["complete"] and r["n_points"] == r["n_unique"] == 1503 and r["n_no_indat"] == 0
    assert r["differ"] == {"MODNAM": 1503, "TEFF": 1503}
    c = r["consistency"]
    assert c["modnam_mismatch"] == 0 and c["teff_meta_mismatch"] == 0
    assert 35402.0 < c["teff_range"][0] < c["teff_range"][1] < 38906.0                # the models' T_eff' range
