"""Tests of ppmpy.synspec.diagnostics: analytic / synthetic checks, and regression (m424) against the
M424 products and the CSV tables of the project's fig_disc_vmac.py / fig_disc_profiles.py."""
import csv
import functools
import inspect
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

from conftest import ROOT, m424_path
from ppmpy.synspec import diagnostics as dg
from ppmpy.synspec import io as sio
from ppmpy.synspec.conventions import C_KMS
from ppmpy.synspec.spectral import VelocityGrid

LREF = np.array([4026.22, 4199.90, 4921.93])
LINES = ["HEI4026", "HEII4200", "HEI4922"]
# frozen copies of the original project sources and tables (tests/synspec/legacy/README.txt)
PROJECT = os.environ.get("PPMPY_SYNSPEC_M424_PROJECT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy"))
TAG = "d3200_r4050_N1236544"


def _project_file(*parts):
    p = os.path.join(PROJECT, *parts)
    if not os.path.exists(p):
        pytest.skip("project file not available: {}".format(p))
    return p


def _legacy_module():
    """The project's fw_disc.py itself (skips when the checkout is not there)."""
    _project_file("fw_disc.py")
    sys.path.insert(0, PROJECT)
    try:
        import fw_disc as fd
    finally:
        sys.path.remove(PROJECT)
    return fd


_trapz = getattr(np, "trapezoid", None) or np.trapz

# Bitwise reproduction of the stored M424 products is guaranteed under numpy 1.26 (the production container);
# under numpy >= 2, np.exp and np.fft differ at the ulp level (diagnostics.py, module Notes).
PRODUCTION_NUMPY = np.lib.NumpyVersion(np.__version__) < "2.0.0"


def _legacy_diagnostics(F, y, lref, vwin, dv=1.0, jacobian=True, inclusive=True):
    """fw_disc.diagnostics (fw_disc.py:180-196) verbatim, with Y -> y, LREF[j] -> lref, DV -> dv, np.trapz ->
    _trapz (numpy 2); jacobian / inclusive False = the version before 2026-09-29."""
    d = 1.0 - F
    if jacobian:
        ew = _trapz(d * np.exp(y / C_KMS), y) * lref / C_KMS
    else:
        ew = _trapz(d, y) * lref / C_KMS
    m = np.abs(y) <= vwin if inclusive else np.abs(y) < vwin
    m0 = _trapz(d[m], y[m])
    v1 = _trapz(y[m] * d[m], y[m]) / m0
    sig = np.sqrt(_trapz((y[m] - v1) ** 2 * d[m], y[m]) / m0)
    k = np.argmax(d)
    half = d[k] / 2
    lo = k - np.argmax(d[k::-1] < half)
    hi = k + np.argmax(d[k:] < half)
    return dict(ew=ew, v1=v1, sigma=sig, fwhm=(hi - lo) * dv, depth=d[k])


def _same(a, b):
    """Equal, or both NaN (the legacy code gives NaN moments for rows with no net absorption)."""
    return bool(a == b) or bool(np.isnan(a) and np.isnan(b))


def _gauss_line(y, depth=0.4, y0=0.0, s=50.0):
    return 1.0 - depth * np.exp(-0.5 * ((y - y0) / s) ** 2)


def _random_profiles(rng, shape, y):
    """Sums of random Gaussian absorption lines with noise (some emission, some flat rows)."""
    F = np.ones(shape + (y.size,))
    for ix in np.ndindex(*shape):
        for _ in range(rng.integers(1, 4)):
            F[ix] -= rng.uniform(-0.1, 0.5) * np.exp(-0.5 * ((y - rng.uniform(-150, 150)) / rng.uniform(5, 80)) ** 2)
        F[ix] += 1e-3 * rng.standard_normal(y.size)
    return F


# PP 2026-10-01: fig_disc_vmac.py:41-62 (shifted, fit), :120-139 (GOF loop) and :151-158 (FT) verbatim; only the
# script's globals (Y, fitw, VWIN, a.snr, fd.broaden / fd.k_*, fd.DV) became arguments.
def _legacy_vmac_fit(Fobs, Ftemp, kernel, grid, Y, broaden, dvs=np.arange(-15.0, 15.01, 0.5), VWIN=500.0):
    fitw = np.abs(Y) < VWIN

    def shifted(G, dv):
        return np.interp(Y, Y + dv, G)

    best = (np.inf, 0.0, 0.0)
    for par in grid:
        B = broaden(Ftemp, kernel(par))
        for dv in dvs:
            r = np.sqrt(np.mean((Fobs[fitw] - shifted(B, dv)[fitw]) ** 2))
            if r < best[0]:
                best = (r, par, dv)
    r, par, dv = best
    fine = np.arange(max(par - 2, 0), par + 2.01, 0.1)             # refine
    for par2 in fine:
        B = broaden(Ftemp, kernel(par2))
        for dv2 in np.arange(dv - 0.5, dv + 0.51, 0.1):
            r2 = np.sqrt(np.mean((Fobs[fitw] - shifted(B, dv2)[fitw]) ** 2))
            if r2 < r:
                r, par, dv = r2, par2, dv2
    return par, dv, r


def _legacy_gof(Fobs, Ftemp, shift, Y, broaden, k_rot, k_rt, snr=300.0, VWIN=500.0,
                vs=np.arange(0.0, 151.0, 3.0), zs=np.arange(0.0, 151.0, 3.0)):
    fitw = np.abs(Y) < VWIN

    def shifted(G, dv):
        return np.interp(Y, Y + dv, G)

    chi = np.zeros((zs.size, vs.size))
    for iv, vr in enumerate(vs):
        R = broaden(Ftemp, k_rot(vr))
        for iz, z in enumerate(zs):
            B = shifted(broaden(R, k_rt(z)), shift)
            chi[iz, iv] = np.sum(((Fobs - B)[fitw] * snr) ** 2)
    return chi


def _legacy_ft(Fobs, Y, DV=1.0, VWIN=500.0):
    dd = 1.0 - Fobs
    wdw = np.abs(Y) < VWIN
    ft = np.abs(np.fft.rfft(dd[wdw] - 0.0, n=1 << 16))
    freq = np.fft.rfftfreq(1 << 16, d=DV)                      # cycles per km/s
    ft /= ft[0]
    mins = np.where((ft[1:-1] < ft[:-2]) & (ft[1:-1] < ft[2:]))[0] + 1
    return freq, ft, mins


# ------------------------------------------------------------------ module hygiene

def test_no_heavy_imports():
    code = ("import sys; import ppmpy.synspec.diagnostics;"
            "bad=[m for m in ('ppmpy.ppm','matplotlib','nugridpy','pyshtools') if m in sys.modules];"
            "print(','.join(bad))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                         env=dict(os.environ, PYTHONPATH=ROOT))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", "heavy modules imported: " + out.stdout


# ------------------------------------------------------------------ line_diagnostics: analytic

def test_gaussian_line_analytic():
    y = np.arange(-1500.0, 1500.5, 1.0)
    A, y0, s, lref = 0.4, 12.0, 50.0, 4026.22
    r = dg.line_diagnostics(_gauss_line(y, A, y0, s), y, lref)
    area = A * s * np.sqrt(2 * np.pi)
    # int A exp(-(y-y0)^2 / 2s^2) exp(y/c) dy = area exp(y0/c + s^2 / 2c^2)
    assert r["ew"] == pytest.approx(area * np.exp(y0 / C_KMS + 0.5 * (s / C_KMS) ** 2) * lref / C_KMS, rel=1e-12)
    old = dg.line_diagnostics(_gauss_line(y, A, y0, s), y, lref, ew_jacobian=False)
    assert old["ew"] == pytest.approx(area * lref / C_KMS, rel=1e-12)
    assert r["v1"] == pytest.approx(y0, abs=1e-9)
    assert r["sigma"] == pytest.approx(s, rel=1e-9)
    assert r["depth"] == pytest.approx(A, rel=1e-15)
    true_fwhm = 2 * np.sqrt(2 * np.log(2)) * s
    # integer grid steps between the first points below half depth on either side
    assert true_fwhm < r["fwhm"] <= true_fwhm + 2.0
    assert set(r) == set(dg.DIAG_KEYS)
    assert np.ndim(r["ew"]) == 0


def test_centroid_follows_shift_on_whole_grid():
    """With the whole grid (vwin None) v1 follows a shift exactly; a window cuts the wings."""
    y = np.arange(-2700.0, 2700.5, 1.0)
    wing = lambda x: 1.0 - 0.3 * np.exp(-np.abs(x) / 150.0)       # extended wings, ~0 at the grid ends
    a = dg.line_diagnostics(wing(y), y, 4026.22)
    b = dg.line_diagnostics(wing(y - 20.0), y, 4026.22)
    assert b["v1"] - a["v1"] == pytest.approx(20.0, abs=1e-3)
    aw = dg.line_diagnostics(wing(y), y, 4026.22, vwin=400.0)
    bw = dg.line_diagnostics(wing(y - 20.0), y, 4026.22, vwin=400.0)
    assert bw["v1"] - aw["v1"] < 19.0


def test_window_inclusive_vs_strict():
    y = np.arange(-600.0, 600.5, 1.0)
    F = _gauss_line(y, 0.3, 30.0, 150.0)
    inc = dg.line_diagnostics(F, y, 4000.0, vwin=400.0)
    strict = dg.line_diagnostics(F, y, 4000.0, vwin=400.0, inclusive=False)
    assert inc["v1"] != strict["v1"] and inc["ew"] == strict["ew"]
    # strict 400 == inclusive 399.5 on an integer grid
    assert dg.line_diagnostics(F, y, 4000.0, vwin=399.5)["v1"] == strict["v1"]


@pytest.mark.parametrize("vwin", [None, 400.0, 120.0])
@pytest.mark.parametrize("jacobian,inclusive", [(True, True), (False, False)])
def test_matches_legacy_algorithm_bitwise(vwin, jacobian, inclusive):
    """Vectorised over rows == the legacy one-profile-per-call code, bit for bit."""
    rng = np.random.default_rng(7)
    y = np.arange(-700.0, 700.5, 1.0)
    F = _random_profiles(rng, (4, 3), y)
    F[1, 2] = 1.0 + 0.2 * np.exp(-0.5 * (y / 30.0) ** 2)            # pure emission (meaningless, but as legacy)
    with np.errstate(invalid="ignore"):                                 # sqrt of negative moments: NaN, as legacy
        r = dg.line_diagnostics(F, y, LREF, vwin=vwin, ew_jacobian=jacobian, inclusive=inclusive)
    vw = 1e9 if vwin is None else vwin                                 # whole grid == legacy vwin beyond the grid
    with np.errstate(invalid="ignore"):
        for i, j in np.ndindex(4, 3):
            ref = _legacy_diagnostics(F[i, j], y, LREF[j], vw, jacobian=jacobian, inclusive=inclusive)
            for k in dg.DIAG_KEYS:
                assert _same(r[k][i, j], ref[k]), (k, i, j)
    assert r["depth"][1, 2] == 0.0                                     # emission: the deepest point is continuum


def test_float32_like_legacy_and_chunks():
    rng = np.random.default_rng(3)
    y = np.arange(-500.0, 500.5, 1.0)
    F = _random_profiles(rng, (5,), y).astype(np.float32)
    with np.errstate(invalid="ignore"):
        r = dg.line_diagnostics(F, y, 4199.9, vwin=300.0)
        r1 = dg.line_diagnostics(F, y, 4199.9, vwin=300.0, chunk=1)
        for k in dg.DIAG_KEYS:
            np.testing.assert_array_equal(r[k], r1[k])
            for i in range(5):
                assert _same(r[k][i], _legacy_diagnostics(F[i], y, 4199.9, 300.0)[k])
    assert r["depth"].dtype == np.float32 and r["ew"].dtype == np.float64


def test_nan_rows_like_legacy():
    y = np.arange(-300.0, 300.5, 1.0)
    F = np.stack([_gauss_line(y), _gauss_line(y)])
    F[1, 250] = np.nan
    with np.errstate(invalid="ignore"):
        r = dg.line_diagnostics(F, y, 4026.22)
        ref = _legacy_diagnostics(F[1], y, 4026.22, 300.0)
    assert np.isnan(r["ew"][1]) and np.isnan(r["v1"][1]) and np.isnan(r["depth"][1])
    assert r["fwhm"][1] == ref["fwhm"] == 0.0
    assert np.isfinite(r["ew"][0])


def test_nan_outside_window_keeps_moments():
    """A NaN outside the moment window: NaN ew and depth, fwhm 0, but finite (unchanged) v1 and sigma."""
    y = np.arange(-300.0, 300.5, 1.0)
    F = _gauss_line(y, s=20.0)
    G = F.copy()
    G[10] = np.nan                                                      # y = -290
    with np.errstate(invalid="ignore"):
        r = dg.line_diagnostics(G, y, 4026.22, vwin=100.0)
        ref = _legacy_diagnostics(G, y, 4026.22, 100.0)
    ok = dg.line_diagnostics(F, y, 4026.22, vwin=100.0)
    assert np.isnan(r["ew"]) and np.isnan(r["depth"]) and r["fwhm"] == 0.0
    assert r["v1"] == ok["v1"] == ref["v1"] and r["sigma"] == ok["sigma"] == ref["sigma"]


@pytest.mark.parametrize("vwin", [None, 400.0])
@pytest.mark.parametrize("jacobian", [True, False])
def test_memory_layout_independent(vwin, jacobian):
    """Fortran-ordered, last-axis-strided and transposed inputs give the C-order numbers bit for bit (the rows
    are made contiguous, as the legacy 1-D d always was; without that the sums run in a different order)."""
    rng = np.random.default_rng(5)
    y = np.arange(-2700.0, 2700.5, 1.0)
    F = _random_profiles(rng, (24,), y)
    big = np.full((24, 2 * y.size), np.nan)
    big[:, ::2] = F
    variants = (np.asfortranarray(F), big[:, ::2], np.ascontiguousarray(F.T).T,
                np.asfortranarray(F.reshape(4, 6, y.size)))
    with np.errstate(invalid="ignore"):
        ref = dg.line_diagnostics(F, y, 4026.22, vwin=vwin, ew_jacobian=jacobian)
        for G in variants:
            r = dg.line_diagnostics(G, y, 4026.22, vwin=vwin, ew_jacobian=jacobian)
            for k in dg.DIAG_KEYS:
                np.testing.assert_array_equal(r[k].reshape(-1), ref[k], err_msg=k)
        for i in (0, 7, 23):                                           # == the legacy one-row code
            leg = _legacy_diagnostics(F[i], y, 4026.22, 1e9 if vwin is None else vwin, jacobian=jacobian)
            assert all(_same(ref[k][i], leg[k]) for k in dg.DIAG_KEYS)


def test_input_validation():
    y = np.arange(-50.0, 50.5, 1.0)
    F = _gauss_line(y, s=10.0)
    for bad_y in (y[::-1], np.r_[y[:10], y[9:]], y[:1]):                # decreasing, repeated point, 1 point
        with pytest.raises(ValueError, match="increasing"):
            dg.line_diagnostics(F[:bad_y.size], bad_y, 4026.22)
    for vw, inc in ((0.3, True), (0.0, True), (1.0, False), (-5.0, True)):   # <= 1 grid point in the window
        with pytest.raises(ValueError, match="moment window"):
            dg.line_diagnostics(F, y, 4026.22, vwin=vw, inclusive=inc)
    assert np.isfinite(dg.line_diagnostics(F, y, 4026.22, vwin=1.0)["sigma"])          # 3 points
    r = dg.line_diagnostics(F, y, 4026.22, vwin=0.3, keys=("ew", "fwhm", "depth"))   # no moments: no window check
    assert set(r) == {"ew", "fwhm", "depth"}
    # empty leading axes
    e = dg.line_diagnostics(np.ones((0, 3, y.size)), y, 4026.22)
    assert all(e[k].shape == (0, 3) for k in dg.DIAG_KEYS)


def test_diagnostics_array_and_keys():
    y = np.arange(-400.0, 400.5, 1.0)
    F = np.stack([[_gauss_line(y, 0.3 + 0.1 * j, 5.0 * k) for j in range(3)] for k in range(2)])
    a = dg.diagnostics_array(F, y, LREF)
    assert a.shape == (2, 3, 5)
    r = dg.line_diagnostics(F, y, LREF)
    for i, k in enumerate(dg.DIAG_KEYS):
        np.testing.assert_array_equal(a[..., i], r[k])
    sub = dg.line_diagnostics(F, y, LREF, keys=("sigma", "depth"))
    assert list(sub) == ["sigma", "depth"]
    np.testing.assert_array_equal(sub["sigma"], r["sigma"])
    np.testing.assert_array_equal(dg.equivalent_width(F, y, LREF), r["ew"])
    with pytest.raises(ValueError):
        dg.line_diagnostics(F, y, LREF, keys=("ew", "skew"))
    with pytest.raises(ValueError):
        dg.line_diagnostics(F, y[:-1], LREF)
    with pytest.raises(ValueError):
        dg.grid_step(np.array([0.0, 1.0, 3.0]))


def test_summary_and_residuals():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 3, 5))
    m, s = dg.diagnostics_summary(x)
    np.testing.assert_array_equal(m, x.mean(axis=0))
    np.testing.assert_array_equal(s, x.std(axis=0))
    F = rng.standard_normal((8, 3, 11))
    mean, res = dg.los_mean_residuals(F)
    np.testing.assert_array_equal(mean, F.mean(axis=0))
    np.testing.assert_allclose(res.mean(axis=0), 0.0, atol=1e-15)


def test_equivalent_width_native():
    lam = np.linspace(4000.0, 4010.0, 161)
    f = 1.0 - 0.5 * np.clip(1.0 - np.abs(lam - 4005.0) / 2.0, 0.0, None)   # triangle, area 1 A
    assert dg.equivalent_width_native(lam, f) == pytest.approx(1.0, rel=1e-12)
    F = np.stack([f, f, 1.0 + 0 * f]).astype(np.float32)
    L = np.broadcast_to(lam, F.shape).astype(np.float32)
    e = dg.equivalent_width_native(L, F)
    assert e.shape == (3,) and e[2] == 0.0
    assert e[0] == _trapz(1.0 - F[0].astype(np.float64), L[0].astype(np.float64))
    # memory layout does not matter (C order, as fig_fw_sphere_ew.py's arrays)
    rng = np.random.default_rng(2)
    G = 1.0 - 0.3 * rng.random((40, 161))
    LL = np.broadcast_to(lam, G.shape)
    ref = dg.equivalent_width_native(np.ascontiguousarray(LL), G)
    np.testing.assert_array_equal(dg.equivalent_width_native(np.asfortranarray(LL), np.asfortranarray(G)), ref)
    np.testing.assert_array_equal(dg.equivalent_width_native(lam, np.asfortranarray(G)), ref)
    np.testing.assert_array_equal(ref, [_trapz(1.0 - G[i], lam) for i in range(40)])


# ------------------------------------------------------------------ kernels and broadening

@pytest.mark.parametrize("fn", [dg.kernel_rotation, dg.kernel_gauss, dg.kernel_rt])
def test_kernels_basic(fn):
    for p in (0.0, -3.0):
        np.testing.assert_array_equal(fn(p), [1.0])
    k = fn(37.3)
    assert k.size % 2 == 1 and k.sum() == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_allclose(k, k[::-1], rtol=0, atol=1e-17)
    assert np.all(k >= 0)


def test_kernel_second_moments():
    dv = 0.1
    v = lambda k: (np.arange(k.size) - k.size // 2) * dv
    # Gaussian exp(-(v/vmac)^2): <v^2> = vmac^2 / 2
    k = dg.kernel_gauss(60.0, dv=dv)
    assert np.sum(k * v(k) ** 2) == pytest.approx(60.0 ** 2 / 2, rel=1e-6)
    # RT, A_R = A_T: int x^2 M / int M = (sqrt(pi)/16) / (sqrt(pi)/4) = 1/4
    k = dg.kernel_rt(80.0, dv=dv)
    assert np.sum(k * v(k) ** 2) == pytest.approx(80.0 ** 2 / 4, rel=1e-4)
    # rotation, eps: <x^2> = (a pi/8 + b 4/15) / (a pi/2 + b 4/3), a = 2(1 - eps), b = pi eps / 2
    for eps in (0.0, 0.6, 1.0):
        a, b = 2 * (1 - eps), np.pi * eps / 2
        x2 = (a * np.pi / 8 + b * 4 / 15) / (a * np.pi / 2 + b * 4 / 3)
        k = dg.kernel_rotation(200.0, dv=dv, eps=eps)
        assert np.sum(k * v(k) ** 2) == pytest.approx(x2 * 200.0 ** 2, rel=2e-3)
    assert dg.kernel_rotation(0.7, dv=1.0).tolist() == [1.0]          # v sin i < dv: no broadening


def test_broaden_conserves_ew_and_adds_variance():
    y = np.arange(-1500.0, 1500.5, 1.0)
    F = _gauss_line(y, 0.4, 0.0, 30.0)
    B = dg.broaden(F, dg.kernel_gauss(50.0))
    r0, r1 = dg.line_diagnostics(F, y, 4026.22), dg.line_diagnostics(B, y, 4026.22)
    # int d dy is conserved; with d lambda = lambda dy / c the EW grows by ~ <v_k^2> / 2c^2 = 7e-9
    assert dg.equivalent_width(B, y, 4026.22, ew_jacobian=False) == pytest.approx(
        dg.equivalent_width(F, y, 4026.22, ew_jacobian=False), rel=1e-12)
    assert r1["ew"] / r0["ew"] - 1 == pytest.approx(50.0 ** 2 / 2 / (2 * C_KMS ** 2), rel=1e-3)
    assert r1["sigma"] ** 2 == pytest.approx(30.0 ** 2 + 50.0 ** 2 / 2, rel=1e-6)
    B2 = dg.broaden(np.stack([F, F]), dg.kernel_gauss(50.0), dg.kernel_rt(20.0))
    np.testing.assert_array_equal(B2[1], dg.broaden(F, dg.kernel_gauss(50.0), dg.kernel_rt(20.0)))
    np.testing.assert_array_equal(dg.broaden(F), 1.0 - (1.0 - F))
    with pytest.raises(ValueError):
        dg.broaden(F[:11], dg.kernel_gauss(50.0))


def test_broaden_empty_rows():
    k = dg.kernel_gauss(5.0)
    for shape in ((0, 101), (2, 0, 101)):
        B = dg.broaden(np.ones(shape, dtype=np.float32), k)
        assert B.shape == shape and B.dtype == np.float64
    assert dg.broaden(np.ones((0, 101), dtype=np.float32)).dtype == np.float32     # as the 1-D path
    assert dg.broaden(np.ones(101, dtype=np.float32), k).dtype == np.float64


def test_shift_profile():
    y = np.arange(-100.0, 100.5, 1.0)
    F = _gauss_line(y, 0.5, 0.0, 10.0)
    S = dg.shift_profile(F, y, 7.0)
    np.testing.assert_allclose(S[20:-20], _gauss_line(y, 0.5, 7.0, 10.0)[20:-20], atol=1e-15)
    w = np.abs(y) < 50
    np.testing.assert_array_equal(dg.shift_profile(F, y, 2.3, at=y[w]), dg.shift_profile(F, y, 2.3)[w])


# ------------------------------------------------------------------ fits, GOF map, Fourier transform

def _template(y):
    return 1.0 - 0.35 * np.exp(-0.5 * (y / 25.0) ** 2) - 0.08 / (1.0 + ((y + 41.0) / 60.0) ** 2)


def test_fit_broadening_recovers_injected():
    y = np.arange(-1200.0, 1200.5, 1.0)
    T = _template(y)
    for kname, kfun, par in (("gauss", dg.kernel_gauss, 41.0), ("rt", dg.kernel_rt, 87.0)):
        obs = dg.shift_profile(dg.broaden(T, kfun(par)), y, -2.6)
        p, s, r = dg.fit_broadening(obs, T, y, kname)
        assert abs(p - par) < 0.15 and abs(s + 2.6) < 0.15 and r < 2e-4, (kname, p, s, r)
    # callable kernel == named kernel
    obs = dg.broaden(T, dg.kernel_gauss(41.0))
    assert dg.fit_broadening(obs, T, y, lambda v: dg.kernel_gauss(v, dv=1.0)) == dg.fit_broadening(obs, T, y, "gauss")
    # no refinement: grid values only
    p, s, _ = dg.fit_broadening(obs, T, y, "gauss", refine=None)
    assert p in dg.DEFAULT_PAR_GRID and s in dg.DEFAULT_SHIFTS


def test_refinement_stops_equal_legacy_literals():
    """fit_broadening's stops par + (h + step / 10) and s + (hs + step / 10) with the default (h, step) are
    fig_disc_vmac.py's literals par + 2.01 and dv + 0.51 for every starting value, because the offsets are
    the same doubles."""
    assert 2.0 + 0.1 / 10 == 2.01 and 0.5 + 0.1 / 10 == 0.51
    p = inspect.signature(dg.fit_broadening).parameters
    assert p["refine"].default == (2.0, 0.1) and p["shift_refine"].default == (0.5, 0.1)
    assert p["par_min"].default == 0.0 and p["vwin"].default == 500.0


def test_fit_broadening_matches_verbatim_legacy():
    """fit_broadening == fig_disc_vmac.py's fit (verbatim copy) bit for bit, including the re-centring shift
    window: with one coarse point (39, 0) the refined shift walks to the true 3 km/s, beyond hs = 0.5."""
    y = np.arange(-1200.0, 1200.5, 1.0)
    T = _template(y)
    rng = np.random.default_rng(11)
    grid = dg.DEFAULT_PAR_GRID
    for kname, kfun, par, s in (("gauss", dg.kernel_gauss, 41.37, -2.63), ("rt", dg.kernel_rt, 87.21, 3.17)):
        obs = dg.shift_profile(dg.broaden(T, kfun(par)), y, s) + 2e-4 * rng.standard_normal(y.size)
        assert dg.fit_broadening(obs, T, y, kname) == _legacy_vmac_fit(obs, T, kfun, grid, y, dg.broaden)
    obs = dg.shift_profile(dg.broaden(T, dg.kernel_gauss(41.0)), y, 3.0)
    r = dg.fit_broadening(obs, T, y, "gauss", grid=[39.0], shifts=[0.0])
    assert r == _legacy_vmac_fit(obs, T, dg.kernel_gauss, [39.0], y, dg.broaden, dvs=np.array([0.0]))
    assert r[0] == pytest.approx(41.0, abs=1e-9) and r[1] == pytest.approx(3.0, abs=1e-9)


def test_fit_broadening_nan_and_shapes():
    y = np.arange(-600.0, 600.5, 1.0)
    T = _template(y)
    obs = T.copy()
    obs[600] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # every rms NaN (NaN in the window, or an empty window): the initial best is returned
        assert dg.fit_broadening(obs, T, y, "gauss", grid=[0.0, 10.0], shifts=[0.0]) == (0.0, 0.0, np.inf)
        assert dg.fit_broadening(T, T, y, "gauss", grid=[0.0, 10.0], shifts=[0.0], vwin=0.0) == (0.0, 0.0, np.inf)
    F2 = np.stack([T, T])
    for call in (lambda: dg.shift_profile(F2, y, 1.0),
                 lambda: dg.fit_broadening(F2, F2, y, "rt"),
                 lambda: dg.fit_broadening(T, F2, y, "rt"),
                 lambda: dg.gof_map(F2, T, y),
                 lambda: dg.gof_map(T, F2, y),
                 lambda: dg.fourier_amplitude(F2, y),
                 lambda: dg.fourier_first_zero(F2, y)):
        with pytest.raises(ValueError, match="1-D"):
            call()
    with pytest.raises(ValueError, match="points"):
        dg.fit_broadening(T[:-1], T, y, "rt")


def test_macroturbulence_fits_layout():
    y = np.arange(-1000.0, 1000.5, 1.0)
    T = _template(y)
    F = np.stack([dg.broaden(T, dg.kernel_gauss(v)) for v in (20.0, 45.0)])
    res = dg.macroturbulence_fits(F, np.stack([T, T]), y, grid=np.arange(0.0, 61.0, 5.0))
    assert res.shape == (2, 6)
    for i in range(2):
        assert tuple(res[i, :3]) == dg.fit_broadening(F[i], T, y, "rt", np.arange(0.0, 61.0, 5.0))
    assert abs(res[0, 3] - 20.0) < 0.15 and abs(res[1, 3] - 45.0) < 0.15


def test_gof_map_finds_injected_point():
    y = np.arange(-1000.0, 1000.5, 1.0)
    T = _template(y)
    obs = dg.shift_profile(dg.broaden(dg.broaden(T, dg.kernel_rotation(60.0)), dg.kernel_rt(30.0)), y, 1.5)
    g = dg.gof_map(obs, T, y, vsini=np.arange(0.0, 91.0, 6.0), zeta=np.arange(0.0, 61.0, 6.0), shift=1.5)
    assert (g["best_vsini"], g["best_zeta"]) == (60.0, 30.0)
    assert g["chi2"].shape == (11, 16) and g["chi2"].min() == pytest.approx(0.0, abs=1e-18)
    assert g["levels"] == dg.GOF_LEVELS


def test_gof_map_matches_verbatim_legacy():
    """gof_map == fig_disc_vmac.py's GOF loop (verbatim copy, two broaden calls) bit for bit; the one-call form
    broaden(T, rot, rt) gives different bits, so this test tells them apart."""
    y = np.arange(-1000.0, 1000.5, 1.0)
    T = _template(y)
    rng = np.random.default_rng(4)
    obs = dg.shift_profile(dg.broaden(dg.broaden(T, dg.kernel_rotation(60.0)), dg.kernel_rt(30.0)), y, 1.5)
    obs = obs + 1e-3 * rng.standard_normal(y.size)
    vs, zs = np.arange(0.0, 91.0, 6.0), np.arange(0.0, 61.0, 6.0)
    g = dg.gof_map(obs, T, y, vsini=vs, zeta=zs, shift=1.3)
    leg = _legacy_gof(obs, T, 1.3, y, dg.broaden, dg.kernel_rotation, dg.kernel_rt, vs=vs, zs=zs)
    np.testing.assert_array_equal(g["chi2"], leg)
    w = np.abs(y) < 500.0
    one = np.array([[np.sum(((obs - dg.shift_profile(dg.broaden(T, dg.kernel_rotation(v), dg.kernel_rt(z)), y, 1.3))[w]
                              * 300.0) ** 2) for v in vs] for z in zs])
    assert not np.array_equal(one, leg)
    np.testing.assert_allclose(one, leg, rtol=1e-9)


def test_fourier_first_zero_of_rotation_profile():
    """A narrow line broadened by rotation (eps = 0.6): the first zero gives v sin i (Gray's 0.660)."""
    y = np.arange(-1500.0, 1500.5, 1.0)
    narrow = 1.0 - 0.6 * np.exp(-0.5 * (y / 1.0) ** 2)
    for vsini in (60.0, 120.0, 200.0):
        F = dg.broaden(narrow, dg.kernel_rotation(vsini))
        r = dg.fourier_first_zero(F, y, vwin=vsini + 50.0)
        assert r["vsini"] == pytest.approx(vsini, rel=0.02), (vsini, r["vsini"])
        assert r["amp"][0] == 1.0 and r["freq"][1] == pytest.approx(1.0 / 65536)
    r = dg.fourier_first_zero(narrow, y, vwin=8.0)                 # ~ Gaussian: no minimum
    assert np.isnan(r["freq1"]) and np.isnan(r["vsini"]) and r["minima"].size == 0
    # the window must fit into nfft (np.fft.rfft would crop it silently)
    with pytest.raises(ValueError, match="nfft"):
        dg.fourier_amplitude(narrow, y, vwin=600.0, nfft=512)       # 1199 points
    with pytest.raises(ValueError, match="nfft"):
        dg.fourier_first_zero(narrow, y, vwin=600.0, nfft=1198)
    f, a = dg.fourier_amplitude(narrow, y, vwin=600.0, nfft=1199)
    assert f.size == a.size == 600 and a[0] == 1.0
    # == the legacy FT code (same numpy)
    freq, ft, mins = _legacy_ft(F, y)
    r = dg.fourier_first_zero(F, y)
    np.testing.assert_array_equal(r["freq"], freq)
    np.testing.assert_array_equal(r["amp"], ft)
    np.testing.assert_array_equal(r["minima"], mins)


# ------------------------------------------------------------------ M424 regression

@pytest.mark.m424
@pytest.mark.parametrize("method", ["imu", "flux"])
def test_m424_per_dump_diag(method):
    """disc_dumps/<method>/d3200.npz: diag_F / diag_F0 were computed (vwin = VY = 2700, whole grid) from
    the float64 F before F was stored as float32 (checked bit for bit by
    test_m424_per_dump_diag_from_rebuilt_float64). The stored F carry an absolute rounding error of up
    to 2^-24 ~ 6e-8 near F = 1, so the recomputed values agree (observed, imu and flux) to <= 7.0e-8
    relative in ew, <= 3e-8 in depth, exactly in fwhm, <= 5.8e-5 km/s in v1 (v1 ~ 1 km/s) and <= 1.4e-6
    relative in sigma (whose whole-grid moments weight the far Stark wings by y^2 ~ 7e6 km^2/s^2).
    Asserted with margins: rtol 2e-7 (ew), atol 2e-4 km/s (v1), rtol 5e-6 (sigma), atol 6e-8 (depth)."""
    z = np.load(m424_path("disc", method, "d3200.npz"))
    assert list(z["diag_keys"]) == list(dg.DIAG_KEYS) and float(z["diag_vwin"]) == 2700.0
    y = VelocityGrid().y
    for X, st in (("F", "diag_F"), ("F0", "diag_F0")):
        ref = z[st]
        r = dg.diagnostics_array(z[X], y, LREF)
        r64 = dg.diagnostics_array(z[X].astype(np.float64), y, LREF)
        for a in (r, r64):
            np.testing.assert_allclose(a[..., 0], ref[..., 0], rtol=2e-7, atol=0)            # ew
            np.testing.assert_allclose(a[..., 1], ref[..., 1], rtol=0, atol=2e-4)            # v1 [km/s]
            np.testing.assert_allclose(a[..., 2], ref[..., 2], rtol=5e-6, atol=0)            # sigma
            np.testing.assert_array_equal(a[..., 3], ref[..., 3])                            # fwhm
            np.testing.assert_allclose(a[..., 4], ref[..., 4], rtol=0, atol=6e-8)            # depth


@pytest.mark.m424
def test_m424_vwin_none_equals_legacy_vy():
    """vwin=None is the legacy vwin = VY (all grid points inside |y| <= 2700): bit for bit."""
    d = np.load(m424_path("run", "disc_los8.npz"))
    y = d["Y"]
    a = dg.diagnostics_array(d["F"], y, d["LREF"])
    b = dg.diagnostics_array(d["F"], y, d["LREF"], vwin=2700.0)
    np.testing.assert_array_equal(a, b)
    for k, j in ((0, 0), (5, 1), (7, 2)):
        ref = _legacy_diagnostics(d["F"][k, j], y, d["LREF"][j], 2700.0)
        assert tuple(a[k, j]) == tuple(ref[q] for q in dg.DIAG_KEYS)


@pytest.mark.m424
def test_m424_timeseries_diag_sample():
    """imu_timeseries.npz (collected per-dump diagnostics), every 40th dump, from the float32 F; observed
    <= 7.3e-8 relative (ew), 6.1e-5 km/s (v1), 1.8e-6 relative (sigma), 3e-8 (depth), fwhm exact; the
    same tolerances as test_m424_per_dump_diag."""
    p = m424_path("disc", "imu_timeseries.npz")
    z = np.load(p)
    sel = np.arange(0, z["dumps"].size, 40)
    F = np.asarray(sio.npz_member_memmap(p, "F")[sel]).astype(np.float64)
    ref = z["diag_F"][sel]
    r = dg.diagnostics_array(F, z["Y"], z["LREF"])
    np.testing.assert_allclose(r[..., 0], ref[..., 0], rtol=2e-7, atol=0)
    np.testing.assert_allclose(r[..., 1], ref[..., 1], rtol=0, atol=2e-4)
    np.testing.assert_allclose(r[..., 2], ref[..., 2], rtol=5e-6, atol=0)
    np.testing.assert_array_equal(r[..., 3], ref[..., 3])
    np.testing.assert_allclose(r[..., 4], ref[..., 4], rtol=0, atol=6e-8)


@pytest.mark.m424
@pytest.mark.slow
def test_m424_per_dump_diag_from_rebuilt_float64():
    """The premise of the tolerances above: rebuild the float64 F, F0 of flux dumps 3200 and 4800 with the
    project's fw_disc (lib_nodes + DiscFlux, verbatim as fw_disc_dumps.py:68-104). They round to the stored
    float32 F, F0 exactly, and diagnostics_array (defaults) of the float64 profiles gives the stored diag_F,
    diag_F0 bit for bit. ~3-7 s with a warm page cache (up to ~1 min cold). Bitwise under numpy 1.26 (the
    container). Under numpy >= 2 (with its scipy) np.exp, einsum and the FFTs round differently: the rebuilt
    F64 still round to the stored float32, but the diagnostics differ by up to ~4e-11 relative (sigma of F0,
    whose whole-grid moment weights the wings by y^2; host numpy 2.2.2 / scipy 1.15.1), so there they are
    compared to 1e-9."""
    fd = _legacy_module()
    samples = os.environ.get("PPMPY_SYNSPEC_M424_SAMPLES", fd.SAMPLES)
    lib = np.load(m424_path("run", "library_dT10.npz"))
    pts = np.load(m424_path("run", "points.npz"))
    INT = fd.DiscFlux(fd.lib_nodes(lib, nmin=20, smooth=0.0, lamfix=False))
    rhat, that, phat = fd.unit_vectors(pts["theta"], pts["phi"])
    LOS = fd.los8()
    MU, TN, PN = rhat @ LOS.T, that @ LOS.T, phat @ LOS.T
    MU, TN, PN = (np.ascontiguousarray(x.T) for x in (MU, TN, PN))
    del rhat, that, phat
    y = VelocityGrid().y
    np.testing.assert_array_equal(y, fd.Y)
    for dump in (3200, 4800):
        sp = os.path.join(samples, "d{:04d}.npz".format(dump))
        if not os.path.exists(sp):
            pytest.skip("M424 sphere sample not available: {}".format(sp))
        z = np.load(m424_path("disc", "flux", "d{:04d}.npz".format(dump)))
        assert int(z["nmin"]) == 20 and not bool(z["lamfix"]) and float(z["smooth"]) == 0.0
        smp = np.load(sp)
        teff = smp["teff"].astype(np.float64)
        ur, uth, uph = (smp[k].astype(np.float64) for k in ("ur", "uth", "uph"))
        k0, k1, w1 = INT.pairs(teff)
        F, F0 = np.zeros((8, 3, y.size)), np.zeros((8, 3, y.size))
        for k in range(8):
            v = ur * MU[k] + uth * TN[k] + uph * PN[k]
            F[k], F0[k] = INT(MU[k], v, k0, k1, w1)[:2]
        for X64, X, st in ((F, "F", "diag_F"), (F0, "F0", "diag_F0")):
            r = dg.diagnostics_array(X64, y, LREF)
            if PRODUCTION_NUMPY:
                np.testing.assert_array_equal(X64.astype(np.float32), z[X])
                np.testing.assert_array_equal(r, z[st])
            else:
                np.testing.assert_allclose(X64, z[X], rtol=0, atol=2.0 ** -24)
                np.testing.assert_allclose(r, z[st], rtol=1e-9, atol=1e-9)      # atol: v1 ~ 0 km/s
                np.testing.assert_array_equal(r[..., 3], z[st][..., 3])


@pytest.mark.m424
@pytest.mark.parametrize("name", ["disc_los8.npz", "disc_los8_imu.npz"])
def test_m424_disc_los8_legacy_mode(name):
    """RUN/disc_los8*.npz (2026-09-28): vwin = 400 with |y| < vwin and the EW without exp(y/c).
    ew_jacobian=False, inclusive=False reproduces diag_F and diag_F0 bit for bit (F stored as float64).
    The current definition changes the EW by -(8.4..8.9)e-5 A for lambda4026 (the documented "+9e-5 A"
    of the old formula) and by <= 1e-5 A for the others; adding the grid points +-400 km/s changes v1 by
    <= 0.03 km/s and sigma by <= 0.3 km/s, fwhm and depth not at all."""
    d = np.load(m424_path("run", name))
    assert list(d["diag_keys"]) == list(dg.DIAG_KEYS)
    y, lref = d["Y"], d["LREF"]
    for X, st in (("F", "diag_F"), ("F0", "diag_F0")):
        old = dg.diagnostics_array(d[X], y, lref, vwin=400.0, ew_jacobian=False, inclusive=False)
        np.testing.assert_array_equal(old, d[st])
        new = dg.diagnostics_array(d[X], y, lref, vwin=400.0)
        dew = d[st][..., 0] - new[..., 0]                                  # old - new
        assert np.all((dew[:, 0] > 8.0e-5) & (dew[:, 0] < 9.5e-5)), dew[:, 0]
        assert np.all(np.abs(dew[:, 1:]) <= 1e-5)
        assert 0 < np.abs(new[..., 1] - d[st][..., 1]).max() < 0.05
        assert 0 < np.abs(new[..., 2] - d[st][..., 2]).max() < 0.5
        np.testing.assert_array_equal(new[..., 3:], d[st][..., 3:])


@pytest.mark.m424
def test_m424_native_ew():
    """RUN/ew.npz (fig_fw_sphere_ew.py) from profiles.npz lam/fnorm, bit for bit (subsets)."""
    p = m424_path("run", "profiles.npz")
    ew = np.load(m424_path("run", "ew.npz"))["ew"]
    lam, fn = sio.npz_member_memmap(p, "lam"), sio.npz_member_memmap(p, "fnorm")
    n = lam.shape[0]
    for sl in (slice(0, 2000), slice(n - 2000, n), slice(618270, 618280)):
        np.testing.assert_array_equal(dg.equivalent_width_native(lam[sl], fn[sl]), ew[sl])
    assert dg.equivalent_width_native(lam[4242, 2], fn[4242, 2]) == ew[4242, 2]


@functools.lru_cache(maxsize=None)
def _module_fits(suffix):
    """macroturbulence_fits of RUN/disc_los8<suffix>.npz (computed once per session; do not modify)."""
    d = np.load(m424_path("run", "disc_los8{}.npz".format(suffix)))
    return dg.macroturbulence_fits(d["F"], d["F0"], d["Y"])


@pytest.mark.m424
@pytest.mark.parametrize("suffix", ["", "_imu"])
def test_m424_vmac_bitwise_vs_verbatim_legacy(suffix):
    """fit_broadening (all 24 profiles, RT and Gaussian), gof_map (los1, 3 lines) and the Fourier transform
    (los1) against verbatim copies of fig_disc_vmac.py's code with the project's fw_disc (fd.broaden, fd.k_rt,
    fd.k_gauss, fd.k_rot): bit for bit. Both run under the same numpy, so this holds for numpy 1.26 and 2."""
    fd = _legacy_module()
    d = np.load(m424_path("run", "disc_los8{}.npz".format(suffix)))
    Y, F, F0 = d["Y"], d["F"], d["F0"]
    np.testing.assert_array_equal(Y, fd.Y)
    res = _module_fits(suffix)
    grid = np.arange(0.0, 151.0, 2.0)
    for k in range(8):
        for j in range(3):
            np.testing.assert_array_equal(res[k, j, :3], _legacy_vmac_fit(F[k, j], F0[k, j], fd.k_rt, grid, Y, fd.broaden))
            np.testing.assert_array_equal(res[k, j, 3:], _legacy_vmac_fit(F[k, j], F0[k, j], fd.k_gauss, grid, Y,
                                                                          fd.broaden))
    K = 0                                                               # --los 1 (the default)
    for j in range(3):
        leg = _legacy_gof(F[K, j], F0[K, j], res[K, j, 1], Y, fd.broaden, fd.k_rot, fd.k_rt, snr=300.0)
        np.testing.assert_array_equal(dg.gof_map(F[K, j], F0[K, j], Y, shift=res[K, j, 1], snr=300.0)["chi2"], leg)
        freq, ft, mins = _legacy_ft(F[K, j], Y, DV=fd.DV)
        r = dg.fourier_first_zero(F[K, j], Y)
        np.testing.assert_array_equal(r["freq"], freq)
        np.testing.assert_array_equal(r["amp"], ft)
        np.testing.assert_array_equal(r["minima"], mins)
        assert r["freq1"] == freq[mins[0]] and r["vsini"] == 0.660 / freq[mins[0]]


@pytest.mark.m424
@pytest.mark.parametrize("suffix", ["", "_imu"])
def test_m424_vmac_csv(suffix):
    """figures/fw_disc_vmac<suffix>_<tag>.csv (fig_disc_vmac.py) from disc_los8<suffix>.npz: the CSV strings
    ('{:.4g}') are reproduced (the bitwise check is test_m424_vmac_bitwise_vs_verbatim_legacy)."""
    rows = list(csv.reader(open(_project_file("figures", "fw_disc_vmac{}_{}.csv".format(suffix, TAG)))))
    d = np.load(m424_path("run", "disc_los8{}.npz".format(suffix)))
    res = _module_fits(suffix)
    assert rows[0] == ["los", "line"] + list(dg.MACRO_COLUMNS) + ["sigma_los"]
    mine = [[str(k + 1), LINES[j]] + ["{:.4g}".format(x) for x in res[k, j]] + ["{:.2f}".format(d["sigma_w"][k, j])]
            for k in range(8) for j in range(3)]
    assert rows[1:] == mine


# printout of fig_disc_vmac.py (run 2026-10-01 into /scratch/ppathak/synspec_shadow/diagnostics/, identical
# CSVs; los1 flux also in project_log.txt 2026-09-28): (vsini, zeta) of the GOF minimum, f1 [1e-3 / (km/s)]
# to 2 decimals and the FT "v sin i" rounded, per line
LEGACY_GOF_FT = {
    ("", 1): [((75, 57), "5.95", 111), ((78, 48), "3.72", 177), ((75, 57), "17.93", 37)],
    ("", 3): [((78, 51), "5.97", 111), ((78, 48), "3.72", 177), ((78, 51), "17.91", 37)],
    ("", 5): [((78, 51), "5.98", 110), ((78, 48), "3.72", 177), ((78, 51), "17.94", 37)],
    ("_imu", 1): [((75, 57), "5.95", 111), ((75, 54), "3.72", 177), ((75, 57), "17.97", 37)],
}


@pytest.mark.m424
@pytest.mark.parametrize("key", sorted(LEGACY_GOF_FT))
def test_m424_gof_and_fourier(key):
    suffix, los = key
    d = np.load(m424_path("run", "disc_los8{}.npz".format(suffix)))
    y, F, F0 = d["Y"], d["F"][los - 1], d["F0"][los - 1]
    for j, ((bv, bz), f1, vft) in enumerate(LEGACY_GOF_FT[key]):
        shift = _module_fits(suffix)[los - 1, j, 1]                    # GOF uses the zeta_RT-only fit's shift
        g = dg.gof_map(F[j], F0[j], y, shift=shift, snr=300.0)
        assert (g["best_vsini"], g["best_zeta"]) == (bv, bz), (j, g["best_vsini"], g["best_zeta"])
        r = dg.fourier_first_zero(F[j], y)
        assert "{:.2f}".format(r["freq1"] * 1e3) == f1 and "{:.0f}".format(r["vsini"]) == str(vft)


@pytest.mark.m424
@pytest.mark.parametrize("suffix", ["", "_imu"])
def test_m424_profiles_csv(suffix):
    """figures/fw_disc_profiles<suffix>_<tag>.csv (fig_disc_profiles.py): the stored diagnostics, here
    recomputed with the 2026-09-28 definitions."""
    rows = list(csv.reader(open(_project_file("figures", "fw_disc_profiles{}_{}.csv".format(suffix, TAG)))))
    d = np.load(m424_path("run", "disc_los8{}.npz".format(suffix)))
    opts = dict(vwin=400.0, ew_jacobian=False, inclusive=False)
    dF = dg.diagnostics_array(d["F"], d["Y"], d["LREF"], **opts)
    dF0 = dg.diagnostics_array(d["F0"], d["Y"], d["LREF"], **opts)
    keys = list(dg.DIAG_KEYS)
    assert rows[0] == ["los", "line", "vmean_w_kms", "sigma_w_kms"] + keys + [k + "_novel" for k in keys]
    mine = [[str(k + 1), LINES[j], "{:.3f}".format(d["vmean_w"][k, j]), "{:.3f}".format(d["sigma_w"][k, j])]
            + ["{:.5g}".format(x) for x in dF[k, j]] + ["{:.5g}".format(x) for x in dF0[k, j]]
            for k in range(8) for j in range(3)]
    assert rows[1:] == mine
    m, s = dg.diagnostics_summary(dF)
    assert m.shape == s.shape == (3, 5)


@pytest.mark.m424
def test_m424_kernels_match_legacy_module():
    """Kernels and broaden against the project's fw_disc.py itself (when the checkout is there)."""
    fd = _legacy_module()
    for p in (0.0, 0.4, 1.0, 7.3, 44.6, 93.1, 150.0):
        np.testing.assert_array_equal(dg.kernel_rotation(p), fd.k_rot(p))
        np.testing.assert_array_equal(dg.kernel_gauss(p), fd.k_gauss(p))
        np.testing.assert_array_equal(dg.kernel_rt(p), fd.k_rt(p))
    d = np.load(m424_path("run", "disc_los8.npz"))
    np.testing.assert_array_equal(dg.broaden(d["F0"][0, 0], dg.kernel_rt(93.1), dg.kernel_gauss(10.0)),
                                  fd.broaden(d["F0"][0, 0], fd.k_rt(93.1), fd.k_gauss(10.0)))
    for k, j in ((0, 0), (3, 1), (6, 2)):
        for vw in (400.0, fd.VY):
            ref = fd.diagnostics(d["F"][k, j], j, vwin=vw)
            r = dg.line_diagnostics(d["F"][k, j], d["Y"], d["LREF"][j], vwin=vw)
            assert all(r[q] == ref[q] for q in dg.DIAG_KEYS)
