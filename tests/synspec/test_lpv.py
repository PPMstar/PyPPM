"""Tests of ppmpy.synspec.lpv: synthetic/analytic checks and M424 regressions."""
import numpy as np
import pytest

from conftest import m424_path
from ppmpy.synspec import lpv
from ppmpy.synspec.conventions import C_KMS

LINES = ["HEI4026", "HEII4200", "HEI4922"]


# ---------------------------------------------------------------------------------------------
# legacy transcriptions (the project scripts, verbatim apart from variable names)
# ---------------------------------------------------------------------------------------------
def legacy_zerocross(Y, F, lref, vwin=150.0):
    """fig_disc_zerocross.py:33-46 for one line of sight and line."""
    nd = F.shape[0]
    A = np.full(nd, np.nan)
    F = F.astype(np.float64)
    Fm = F.mean(axis=0)
    core = np.abs(Y) <= 400
    ymin = Y[core][np.argmin(Fm[core])]
    m = np.abs(Y - ymin) <= vwin
    y, R = Y[m], F[:, m] - Fm[m]
    for i in range(nd):
        s = np.sign(R[i])
        c = np.where(s[:-1] * s[1:] < 0)[0]
        if c.size == 0:
            continue
        yc = y[c] - R[i, c] * (y[c + 1] - y[c]) / (R[i, c + 1] - R[i, c])
        yz = yc[np.argmin(np.abs(yc - ymin))]
        A[i] = lref * np.exp(yz / C_KMS)
    return A


def legacy_stats(Rr, Ro):
    """fw_disc_systematics.py:40-43."""
    d = Ro - Rr
    return dict(sig_rms=Rr.std(), sig_max=np.abs(Rr).max(), d_rms=d.std(), d_max=np.abs(d).max(), ratio=d.std() / Rr.std(),
                amp=np.sum(Ro * Rr) / np.sum(Rr * Rr), corr=np.corrcoef(Ro.ravel(), Rr.ravel())[0, 1])


def legacy_acf(R, lag, k):
    """scratch acf.py (project log 2026-09-29): one line of sight k of R (dumps, los, velocity)."""
    return np.sum(R[:-lag, k] * R[lag:, k]) / np.sqrt(np.sum(R[:-lag, k] ** 2) * np.sum(R[lag:, k] ** 2))


def synthetic_series(nt=64, nlos=3, nline=2, ny=401, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(-(ny // 2), ny // 2 + 1, 1.0)
    v = rng.normal(0, 5.0, size=(nt, nlos, nline, 1))
    F = 1 - 0.3 * np.exp(-((y - v) / 40.0) ** 2) + 1e-4 * rng.standard_normal((nt, nlos, nline, ny))
    return y, F.astype(np.float32)


# ---------------------------------------------------------------------------------------------
# residual spectra, summaries, colour limits
# ---------------------------------------------------------------------------------------------
def test_residual_spectra_mean():
    _, F = synthetic_series()
    R, Fref = lpv.residual_spectra(F)
    assert R.dtype == np.float64 and R.shape == F.shape and Fref.shape == F.shape[1:]
    np.testing.assert_allclose(R.mean(axis=0), 0, atol=1e-15)
    X = F.astype(np.float64)
    np.testing.assert_array_equal(R, X - X.mean(axis=0))            # legacy expression, bit for bit


def test_residual_spectra_ref_rows_and_array():
    _, F = synthetic_series()
    rows = np.arange(10, 30)
    R, Fref = lpv.residual_spectra(F, ref_rows=rows)
    X = F.astype(np.float64)
    np.testing.assert_array_equal(Fref, X[rows].mean(axis=0))
    np.testing.assert_array_equal(R, X - X[rows].mean(axis=0))
    bmask = np.zeros(F.shape[0], bool)
    bmask[rows] = True
    np.testing.assert_array_equal(lpv.residual_spectra(F, ref_rows=bmask)[1], Fref)
    R2, F2 = lpv.residual_spectra(F, ref=Fref)
    np.testing.assert_array_equal(R2, R)
    R3, _ = lpv.residual_spectra(F, ref=Fref[None])
    np.testing.assert_array_equal(R3, R)
    with pytest.raises(ValueError):
        lpv.residual_spectra(F, ref="median")


def test_residual_spectra_ref_broadcast():
    """A lower-dimensional reference broadcasts against F WITHOUT the time axis (review 2026-10-01)."""
    rng = np.random.default_rng(2)
    F = rng.standard_normal((4, 6, 6))                                 # (los, t, v), time on axis 1
    prof = rng.standard_normal(6)
    R, Fref = lpv.residual_spectra(F, axis=1, ref=prof)
    np.testing.assert_array_equal(R, F - prof)                         # a 1-D profile runs along the last axis
    assert Fref.shape == (4, 6) and (Fref == prof).all()                # documented shape: F without the time axis
    R1, _ = lpv.residual_spectra(F, axis=1, ref=prof[None, None, :])    # (1, 1, v): time axis of length 1
    np.testing.assert_array_equal(R1, R)
    per_los = rng.standard_normal((4, 1))                              # (los, 1): one offset per line of sight
    np.testing.assert_array_equal(lpv.residual_spectra(F, axis=1, ref=per_los)[0], F - per_los[:, None, :])
    _, G = synthetic_series()                                          # (t, los, lines, v)
    p = G.astype(np.float64)[:, 0, 0].mean(axis=0)
    R4, F4 = lpv.residual_spectra(G, ref=p)
    np.testing.assert_array_equal(R4, G.astype(np.float64) - p)
    assert F4.shape == G.shape[1:]
    for bad in (rng.standard_normal(5), rng.standard_normal((2, 6)), rng.standard_normal((4, 2, 6)),
                rng.standard_normal((1, 4, 6, 6))):
        with pytest.raises(ValueError):
            lpv.residual_spectra(F, axis=1, ref=bad)


def test_residual_spectra_other_axis():
    _, F = synthetic_series()
    Ft = np.moveaxis(F, 0, 2)                                          # time on axis 2
    R, Fref = lpv.residual_spectra(Ft, axis=2)
    np.testing.assert_allclose(np.moveaxis(R, 2, 0), lpv.residual_spectra(F)[0], rtol=0, atol=1e-14)
    assert Fref.shape == Ft.shape[:2] + Ft.shape[3:]


def test_residual_summary_and_limit():
    rng = np.random.default_rng(3)
    R = rng.standard_normal((500, 50))
    s = lpv.residual_summary(R)
    np.testing.assert_array_equal(s["rms_t"], R.std(axis=0))
    assert s["rms"] == R.std() and s["max_abs"] == np.abs(R).max()
    assert s["pct_abs"] == np.percentile(np.abs(R), 99.5)
    assert 2.5 < s["pct_abs"] < 3.1                                    # Gaussian: 99.5 % of |x| at 2.81 sigma
    R2 = 3 * R
    assert lpv.symmetric_limit([R, R2]) == np.percentile(np.abs(R2), 99.5)
    assert lpv.symmetric_limit(R) == s["pct_abs"]
    assert lpv.symmetric_limit([R], pct=100) == s["max_abs"]
    assert lpv.symmetric_limit((R, R2)) == lpv.symmetric_limit([R, R2])
    # a plain list of numbers is ONE array (review 2026-10-01: was max|x| = 5.0)
    x = [0.1, -0.2, 0.3, -5.0] + [0.01] * 996
    assert lpv.symmetric_limit(x) == np.percentile(np.abs(x), 99.5) == pytest.approx(0.01)
    assert lpv.symmetric_limit(tuple(x)) == lpv.symmetric_limit(np.array(x))
    with pytest.raises(ValueError):
        lpv.symmetric_limit([])


# ---------------------------------------------------------------------------------------------
# zero-crossing tracker
# ---------------------------------------------------------------------------------------------
def test_zero_crossing_linear_residuals():
    # mean profile with its minimum at y = 2; each pair of rows carries +-a (y - y0) g(y), so the time mean is
    # the base profile (exactly zero mean of the perturbations) and R crosses zero at y0 with a linear shape.
    y = np.arange(-300.0, 301.0)
    base = 1 - 0.3 * np.exp(-((y - 2.0) / 60.0) ** 2)
    y0 = np.array([3.3, -17.25, 40.6, 0.5])
    rows = []
    for v in y0:
        p = 1e-3 * (y - v) / 100.0
        rows += [base + p, base - p]
    F = np.array(rows)
    out = lpv.zero_crossing_track(y, F, lref=4026.22)
    assert out["ymin"] == 2.0 and out["n_found"] == F.shape[0]
    np.testing.assert_allclose(out["y"], np.repeat(y0, 2), rtol=0, atol=1e-9)
    np.testing.assert_allclose(out["lam"], 4026.22 * np.exp(out["y"] / C_KMS), rtol=1e-15)
    np.testing.assert_array_equal(out["n_cross"], 1)
    lam_legacy = legacy_zerocross(y, F, 4026.22)
    np.testing.assert_array_equal(out["lam"], lam_legacy)


def test_zero_crossing_nearest_none_window_and_exact_zero():
    y = np.arange(-20.0, 21.0)
    base = 0.5 + np.abs(y) / 8.0                                       # minimum at 0
    shifted = (y + 10.5) * (y - 0.5) * (y - 6.5) / 4096.0               # crossings at -10.5, 0.5, 6.5
    const = np.full_like(y, 0.25)
    F = np.array([base + shifted, base - shifted, base + const, base - const, base + y / 16.0, base - y / 16.0])
    out = lpv.zero_crossing_track(y, F, halfwidth=20.0, core=5.0)
    assert out["ymin"] == 0.0
    # the crossing nearest ymin of the three, linearly interpolated between y = 0 and 1 (all values here are exact dyadics)
    r0, r1 = 34.125 / 4096, -31.625 / 4096
    np.testing.assert_allclose(out["y"][:2], r0 / (r0 - r1), rtol=0, atol=1e-14)
    np.testing.assert_array_equal(out["n_cross"][:2], 3)
    assert np.isnan(out["y"][2:4]).all() and (out["n_cross"][2:4] == 0).all()   # no sign change
    # exact zero on a pixel (R = +-y/16 is exactly 0 at y = 0): not a sign change (legacy behaviour, documented)
    assert np.isnan(out["y"][4:]).all()
    assert out["n_found"] == 2
    np.testing.assert_array_equal(lpv.zero_crossing_track(y, F, lref=1.0, halfwidth=20.0)["lam"],
                                  legacy_zerocross(y, F, 1.0, vwin=20.0))
    # a narrow window that excludes the crossings
    out2 = lpv.zero_crossing_track(y, F[:2], halfwidth=0.4, core=5.0)
    assert np.isnan(out2["y"]).all() and "lam" not in out2


def test_zero_crossing_matches_legacy_synthetic():
    y, F = synthetic_series(nt=200, nlos=1, nline=1, ny=1201)
    F = F[:, 0, 0]
    for hw in (150.0, 30.0):
        out = lpv.zero_crossing_track(y, F, lref=4199.9, halfwidth=hw)
        np.testing.assert_array_equal(out["lam"], legacy_zerocross(y, F, 4199.9, vwin=hw))
    with pytest.raises(ValueError):
        lpv.zero_crossing_track(y, F[:, :-1])


# ---------------------------------------------------------------------------------------------
# gaps, lag correlation, coherence time
# ---------------------------------------------------------------------------------------------
def test_fill_gaps():
    t = np.linspace(0.0, 10.0, 11)
    x = 2.0 * t + 1.0
    x[[0, 3, 4, 10]] = np.nan
    xf, bad = lpv.fill_gaps(t, x)
    np.testing.assert_array_equal(bad, np.isnan(x))
    np.testing.assert_allclose(xf[3:5], [7.0, 9.0], rtol=1e-15)        # linear inside
    assert xf[0] == 3.0 and xf[10] == 19.0                              # ends held (np.interp)
    assert np.isnan(x[3])                                                # input untouched
    X = np.stack([x, np.full_like(x, np.nan), np.arange(11.0)], axis=1)
    Xf, B = lpv.fill_gaps(t, X)
    np.testing.assert_array_equal(Xf[:, 0], xf)
    assert np.isnan(Xf[:, 1]).all() and B[:, 1].all()                   # nothing to interpolate from
    np.testing.assert_array_equal(Xf[:, 2], np.arange(11.0))
    with pytest.raises(ValueError):
        lpv.fill_gaps(t[:-1], x)


def test_fill_gaps_non_c_order():
    """Non-C-ordered input with >= 3 dimensions (review 2026-10-01: the NaNs stayed while the mask said filled)."""
    rng = np.random.default_rng(9)
    t = np.cumsum(rng.uniform(0.5, 1.5, 20))
    Z = rng.standard_normal((20, 4, 30))
    m = np.arange(30) % 3 != 1
    for x in (Z[:, :, m], np.asfortranarray(Z), np.moveaxis(rng.standard_normal((20, 30, 4)), 2, 1)):
        x = np.array(x, order="K")                                       # keep the non-C layout
        assert not x.flags["C_CONTIGUOUS"]
        x[7, 2, 3] = np.nan
        x[0, 1, 1] = np.nan
        x[13:16, 0, 4] = np.nan
        xf, bad = lpv.fill_gaps(t, x)
        assert np.isfinite(xf).all() and bad.sum() == 5
        np.testing.assert_array_equal(bad, np.isnan(x))
        ref, refbad = lpv.fill_gaps(t, np.ascontiguousarray(x))         # C-ordered copy: the reference
        np.testing.assert_array_equal(xf, ref)
        np.testing.assert_array_equal(bad, refbad)
        good = ~bad
        np.testing.assert_array_equal(xf[good], x[good])
        np.testing.assert_allclose(xf[7, 2, 3], np.interp(t[7], t[[6, 8]], x[[6, 8], 2, 3]), rtol=1e-14)
        assert xf[0, 1, 1] == x[1, 1, 1]                                 # leading gap: first finite value
        np.testing.assert_allclose(xf[13:16, 0, 4], np.interp(t[13:16], t[[12, 16]], x[[12, 16], 0, 4]), rtol=1e-14)


def test_lag_correlation_ar1():
    rng = np.random.default_rng(5)
    nt, npix, phi = 20000, 8, 0.6
    e = rng.standard_normal((nt, npix))
    x = np.empty_like(e)
    x[0] = e[0]
    for i in range(1, nt):
        x[i] = phi * x[i - 1] + np.sqrt(1 - phi ** 2) * e[i]
    lags = [0, 1, 2, 3, 5]
    c = lpv.lag_correlation(x, lags)
    assert c[0] == pytest.approx(1.0, abs=1e-15)
    np.testing.assert_allclose(c, phi ** np.array(lags), atol=0.02)
    cp = lpv.lag_correlation(x, lags, demean=True)
    np.testing.assert_allclose(cp, c, atol=0.01)
    assert lpv.lag_correlation(x, 1) == pytest.approx(c[1], rel=1e-14)    # scalar lag
    np.testing.assert_allclose(lpv.lag_correlation(x, [-2, 2]), lpv.lag_correlation(x, 2), rtol=1e-12)
    assert np.isnan(lpv.lag_correlation(x, [nt])).all()
    # coherence time: phi^L = 1/e at L = -1/ln(phi) = 1.96
    tc = lpv.coherence_time(np.arange(11), lpv.lag_correlation(x, np.arange(11)))
    assert 1.6 < tc < 2.2
    # independent second series: cross-correlation ~ 0
    other = rng.standard_normal((nt, npix))
    assert abs(lpv.lag_correlation(x, 0, other=other)) < 0.03
    # lags are whole numbers of rows (review 2026-10-01: 1.7 was truncated to 1)
    np.testing.assert_array_equal(lpv.lag_correlation(x, [1.0, 2.0]), lpv.lag_correlation(x, [1, 2]))
    for bad in (1.7, [1, 2.5], np.nan, "1"):
        with pytest.raises(ValueError):
            lpv.lag_correlation(x, bad)
    with pytest.raises(ValueError):
        lpv.lag_correlation(x, 1, demean="median")


def test_lag_correlation_demean_conventions():
    """demean=True/'pooled' is np.corrcoef of the flattened segments; 'pixel' removes per-pixel means."""
    rng = np.random.default_rng(13)
    x = rng.standard_normal((4000, 3)) + np.array([0.0, 5.0, -5.0])     # white noise, per-pixel offsets
    a, b = x[:-1], x[1:]
    pooled = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    assert lpv.lag_correlation(x, 1, demean=True) == pytest.approx(pooled, rel=1e-12)
    assert lpv.lag_correlation(x, 1, demean="pooled") == lpv.lag_correlation(x, 1, demean=np.True_)
    assert pooled > 0.9                                                 # dominated by the offsets
    ap, bp = a - a.mean(axis=0), b - b.mean(axis=0)
    pix = np.sum(ap * bp) / np.sqrt(np.sum(ap ** 2) * np.sum(bp ** 2))
    assert lpv.lag_correlation(x, 1, demean="pixel") == pix              # bit for bit
    assert abs(pix) < 0.05                                               # white noise: uncorrelated
    # keep_axes: per pixel, 'pixel' and 'pooled' coincide (one pixel per kept index)
    np.testing.assert_allclose(lpv.lag_correlation(x, [1, 3], keep_axes=1, demean="pixel"),
                               lpv.lag_correlation(x, [1, 3], keep_axes=1, demean="pooled"), rtol=1e-12)
    # 1-D: both are the Pearson correlation of the segments
    y = x[:, 1]
    assert lpv.lag_correlation(y, 2, demean="pixel") == pytest.approx(np.corrcoef(y[:-2], y[2:])[0, 1], rel=1e-12)


def test_lag_correlation_keep_axes_matches_formula():
    rng = np.random.default_rng(7)
    R = rng.standard_normal((300, 4, 50))
    R -= R.mean(axis=0)
    lags = [1, 2, 5]
    c = lpv.lag_correlation(R, lags, keep_axes=1)
    assert c.shape == (3, 4)
    for il, L in enumerate(lags):
        for k in range(4):
            assert c[il, k] == legacy_acf(R, L, k)                      # bit for bit
    # time on another axis, two kept axes
    Rt = np.moveaxis(R, 0, 2)
    c2 = lpv.lag_correlation(Rt, lags, axis=2, keep_axes=(0,))
    np.testing.assert_allclose(c2, c, rtol=1e-12)
    c3 = lpv.lag_correlation(R, lags, keep_axes=(1, 2))
    assert c3.shape == (3, 4, 50)
    with pytest.raises(ValueError):
        lpv.lag_correlation(R, lags, keep_axes=0)


def test_coherence_time_edge_cases():
    lags = np.arange(5.0)
    assert np.isnan(lpv.coherence_time(lags, [1, 0.9, 0.8, 0.7, 0.6]))
    assert lpv.coherence_time(lags, [1, 0.5, 0.3, 0.1, 0.0], level=0.4) == pytest.approx(1.5)
    out = lpv.coherence_time(lags, np.array([[1, 1], [0.5, 0.9], [0.3, 0.8], [0.1, 0.2], [0, 0]]), level=0.4)
    np.testing.assert_allclose(out, [1.5, 2 + 0.4 / 0.6])


# ---------------------------------------------------------------------------------------------
# comparison of two runs
# ---------------------------------------------------------------------------------------------
def test_compare_timeseries_identities():
    y, F = synthetic_series()
    m = np.abs(y) <= 120
    st = lpv.compare_timeseries(F, F, mask=m)
    assert set(st) == {"sig_rms", "sig_max", "d_rms", "d_max", "ratio", "amp", "corr", "static"}
    assert st["d_rms"].shape == (F.shape[2],)
    np.testing.assert_array_equal(st["d_rms"], 0)
    np.testing.assert_array_equal(st["static"], 0)
    np.testing.assert_allclose(st["amp"], 1, rtol=1e-14)
    np.testing.assert_allclose(st["corr"], 1, rtol=1e-14)
    # run = ref + 2 (time-variable part) + an offset: amplitude ratio 2, ratio 1, corr 1, static = offset
    X = F.astype(np.float64)
    G = X + (X - X.mean(axis=0)) + 0.25
    st2 = lpv.compare_timeseries(X, G, mask=m)
    np.testing.assert_allclose(st2["amp"], 2, rtol=1e-12)
    np.testing.assert_allclose(st2["ratio"], 1, rtol=1e-12)
    np.testing.assert_allclose(st2["corr"], 1, rtol=1e-12)
    np.testing.assert_allclose(st2["static"], 0.25, rtol=1e-12)
    # pooled (no line axis) gives floats
    assert isinstance(lpv.compare_timeseries(F[:, :, 0], F[:, :, 0], line_axis=None)["corr"], float)
    with pytest.raises(ValueError):
        lpv.compare_timeseries(F, F[:-1])


def test_compare_timeseries_matches_legacy_and_static_quirk():
    y, F = synthetic_series(seed=1)
    _, G = synthetic_series(seed=2)
    m = np.abs(y) <= 100
    st = lpv.compare_timeseries(F, G, mask=m)
    for j in range(F.shape[2]):
        A, B = F[:, :, j][..., m].astype(np.float64), G[:, :, j][..., m].astype(np.float64)
        ref = legacy_stats(A - A.mean(axis=0, keepdims=True), B - B.mean(axis=0, keepdims=True))
        ref["static"] = np.abs(G[:, :, j].astype(np.float64).mean(0) - F[:, :, j].astype(np.float64).mean(0)).max()
        for k, v in ref.items():
            assert st[k][j] == v, k                                      # bit for bit
    # static within the window instead of the whole grid
    stw = lpv.compare_timeseries(F, G, mask=m, static_full_grid=False)
    for j in range(F.shape[2]):
        sw = np.abs(G[:, :, j][..., m].astype(np.float64).mean(0) - F[:, :, j][..., m].astype(np.float64).mean(0)).max()
        assert stw["static"][j] == sw
    # time on another axis
    st_t = lpv.compare_timeseries(np.moveaxis(F, 0, 1), np.moveaxis(G, 0, 1), axis=1, line_axis=2, mask=m)
    for k in st:
        np.testing.assert_allclose(st_t[k], st[k], rtol=1e-12)


def test_compare_series():
    rng = np.random.default_rng(11)
    x = rng.standard_normal((400, 8, 3))
    z = x + 0.1 * rng.standard_normal(x.shape) + 5.0
    st = lpv.compare_series(x, z)
    assert st["corr"].shape == (3,)
    for j in range(3):
        xr = x[:, :, j] - x[:, :, j].mean(axis=0)
        xo = z[:, :, j] - z[:, :, j].mean(axis=0)
        assert st["sig_rms"][j] == xr.std() and st["d_rms"][j] == (xo - xr).std()
        assert st["corr"][j] == np.corrcoef(xo.ravel(), xr.ravel())[0, 1]
    np.testing.assert_allclose(st["ratio"], 0.1, rtol=0.1)
    one = lpv.compare_series(x[:, :, 0], z[:, :, 0], line_axis=None)
    assert one["sig_rms"] == st["sig_rms"][0]


def test_systematics_synthetic():
    y, F = synthetic_series(nline=3)
    _, G = synthetic_series(nline=3, seed=4)
    rng = np.random.default_rng(0)
    D = rng.standard_normal((F.shape[0], F.shape[1], 3, 5))
    ref = dict(F=F, F0=F, diag_F=D, Y=y, dumps=np.arange(F.shape[0]), diag_keys=np.array(["ew", "v1", "sigma", "fwhm", "depth"]))
    run = dict(F=G, F0=F, diag_F=D * 1.1, Y=y, dumps=np.arange(F.shape[0]))
    out = lpv.systematics(ref, {"x": run}, ref_name="r", vwin=120.0, lines=LINES)
    assert str(out["ref"]) == "r" and list(out["lines"]) == LINES and out["vwin"] == 120.0
    st = lpv.compare_timeseries(F, G, mask=np.abs(y) <= 120.0)
    for k, v in st.items():
        np.testing.assert_array_equal(out["x_F_" + k], v)
    np.testing.assert_array_equal(out["x_F0_d_rms"], 0)
    np.testing.assert_allclose(out["x_ew_ratio"], 0.1, rtol=1e-12)
    np.testing.assert_allclose(out["x_sigma_corr"], 1, rtol=1e-12)
    # no M424 names by default (review 2026-10-01): ref['lines'] if present, else line<j>
    assert list(lpv.systematics(ref, {"x": run}, quantities=())["lines"]) == ["line0", "line1", "line2"]
    assert list(lpv.systematics(dict(ref, lines=np.array(["a", "b", "c"])), {"x": run}, quantities=())["lines"]) == ["a", "b", "c"]
    with pytest.raises(ValueError):
        lpv.systematics(ref, {"x": run}, quantities=(), lines=["a", "b"])
    # diagnostics located by name in each mapping's own diag_keys (a run with another key order)
    perm = [3, 2, 4, 0, 1]
    keys = ref["diag_keys"]
    run2 = dict(run, diag_F=run["diag_F"][..., perm], diag_keys=keys[perm])
    out2 = lpv.systematics(ref, {"x": run2}, ref_name="r", vwin=120.0, quantities=(), lines=LINES)
    assert sorted(out2) == sorted(k for k in out if not k.startswith(("x_F_", "x_F0_")))
    for k in out2:
        np.testing.assert_array_equal(out2[k], out[k])
    run3 = dict(run2, diag_keys=keys[perm][:4], diag_F=run2["diag_F"][..., :4])     # 'v1' missing
    with pytest.raises(ValueError):
        lpv.systematics(ref, {"x": run3}, quantities=())
    run["dumps"] = run["dumps"] + 1
    with pytest.raises(ValueError):
        lpv.systematics(ref, {"x": run})


def test_open_timeseries(tmp_path):
    p = str(tmp_path / "ts.npz")
    F = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    np.savez(p, F=F, Y=np.arange(4.0))
    d = lpv.open_timeseries(p, mmap=("F",))
    assert isinstance(d["F"], np.memmap)
    np.testing.assert_array_equal(d["F"], F)
    np.testing.assert_array_equal(d["Y"], np.arange(4.0))
    # one name as a string (review 2026-10-01: 'F0' also mapped 'F' by a substring test)
    np.savez(p, F=F, F0=F + 1, Y=np.arange(4.0))
    d = lpv.open_timeseries(p, mmap="F0")
    assert isinstance(d["F0"], np.memmap) and not isinstance(d["F"], np.memmap)
    d = lpv.open_timeseries(p, mmap=None)
    assert not any(isinstance(v, np.memmap) for v in d.values())


# ---------------------------------------------------------------------------------------------
# M424 regressions
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def imu():
    return lpv.open_timeseries(m424_path("disc", "imu_timeseries.npz"))


@pytest.mark.m424
@pytest.mark.slow
def test_zero_crossing_track_m424(imu):
    Y, LREF = imu["Y"], imu["LREF"]
    nfound = {}
    for k in range(8):
        z = np.load(m424_path("disc", "zerocross_imu_los{}.npz".format(k + 1)))
        assert float(z["vwin"]) == 150.0
        np.testing.assert_array_equal(z["dumps"], imu["dumps"])
        Fk = np.asarray(imu["F"][:, k])                                  # (dumps, lines, velocity), 104 MB
        for j in range(3):
            out = lpv.zero_crossing_track(Y, Fk[:, j], lref=LREF[j])
            np.testing.assert_array_equal(out["lam"], z["A"][:, j])      # bit for bit, NaNs included
            nfound[k, j] = out["n_found"]
            assert out["n_found"] == np.isfinite(z["A"][:, j]).sum()
    assert nfound[0, 1] == 1601 - 41                                     # 41 dumps without a crossing (lambda4200, los1)


@pytest.mark.m424
def test_fill_gaps_m424():
    nbad = {}
    for k in range(8):
        z = np.load(m424_path("disc", "zerocross_imu_los{}.npz".format(k + 1)))
        t = z["t_s"] - z["t_s"][0]
        for j in range(3):
            A = z["A"][:, j].copy()                                      # fig_disc_zerocross_spectrum.py series()
            bad = ~np.isfinite(A)
            A[bad] = np.interp(t[bad], t[~bad], A[~bad])
            xf, mask = lpv.fill_gaps(t, z["A"][:, j])
            np.testing.assert_array_equal(xf, A)
            np.testing.assert_array_equal(mask, bad)
            nbad[k, j] = int(mask.sum())
        Xf, B = lpv.fill_gaps(t, z["A"])                                  # all three lines at once
        for j in range(3):
            np.testing.assert_array_equal(Xf[:, j], lpv.fill_gaps(t, z["A"][:, j])[0])
    assert nbad[0, 1] == 41


@pytest.mark.m424
@pytest.mark.slow
def test_residual_spectra_and_limit_m424(imu):
    """fig_disc_dynspec.py: residuals, rms curve and common colour limit (los1, |v| <= 400 km/s)."""
    Y, dumps = imu["Y"], imu["dumps"]
    m = np.abs(Y) <= 400.0
    F1 = np.asarray(imu["F"][:, 0])                                       # los1 (dumps, lines, velocity)
    lims = {}
    for zoom in (None, (3200, 3260)):
        isel = np.arange(dumps.size) if zoom is None else np.where((dumps >= zoom[0]) & (dumps <= zoom[1]))[0]
        RES = [F1[isel, j][:, m].astype(np.float64) for j in range(3)]  # legacy, standard layout
        LEG = [R - R.mean(axis=0) for R in RES]
        Rs = [lpv.residual_spectra(F1[isel, j][:, m])[0] for j in range(3)]
        for a, b in zip(Rs, LEG):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(lpv.residual_summary(Rs[0])["rms_t"], LEG[0].std(axis=0))
        lims[zoom] = lpv.symmetric_limit(Rs)
        assert lims[zoom] == max(np.percentile(np.abs(R), 99.5) for R in LEG)
        # --mean-all: mean of all dumps while showing the window
        MALL = [F1[:, j][:, m].astype(np.float64).mean(axis=0) for j in range(3)]
        for j in range(3):
            Ra, Fref = lpv.residual_spectra(F1[:, j][:, m])
            np.testing.assert_array_equal(Fref, MALL[j])
            np.testing.assert_array_equal(Ra[isel], RES[j] - MALL[j])
            Rb, _ = lpv.residual_spectra(F1[:, j][:, m], ref_rows=isel)
            np.testing.assert_array_equal(Rb[isel], LEG[j])
            # the documented recipe: mean of all dumps applied to the window -> the rms curve bit for bit
            Rw, _ = lpv.residual_spectra(F1[isel, j][:, m], ref=Fref)
            rms_leg = (RES[j] - MALL[j]).std(axis=0)
            np.testing.assert_array_equal(Rw, RES[j] - MALL[j])
            np.testing.assert_array_equal(lpv.residual_summary(Rw)["rms_t"], rms_leg)
            # slicing R of the whole series: same R, rms curve equal to ~1e-15 only (C order, documented)
            np.testing.assert_allclose(lpv.residual_summary(Ra[isel])["rms_t"], rms_leg, rtol=1e-12, atol=0)
    # project log 2026-09-29: common colour scale +-1.63e-3 (full) and +-1.36e-3 (zoom 3200-3260)
    assert round(lims[None], 5) == pytest.approx(1.63e-3) and round(lims[(3200, 3260)], 5) == pytest.approx(1.36e-3)


@pytest.mark.m424
@pytest.mark.slow
def test_lag_correlation_m424(imu):
    """Project log 2026-09-29: residual-spectrum correlation vs lag (mean over the 8 LOS, |v| <= 600, imu)."""
    logged = {0: [0.77, 0.55, 0.37, 0.14, 0.04], 1: [0.83, 0.63, 0.44, 0.17, 0.04], 2: [0.69, 0.46, 0.30, 0.12, 0.05]}
    opposite_resid = {0: [-0.0056, 0.0036, 0.0006, 0.0005], 1: [-0.0429, -0.0242, -0.0596, -0.0402],
                      2: [-0.0101, -0.0012, 0.0089, -0.0022]}
    lags = [1, 2, 3, 5, 10]
    m = np.abs(imu["Y"]) <= 600.0
    for j in range(3):
        F = np.asarray(imu["F"][:, :, j])[:, :, m].astype(np.float64)
        R = F - F.mean(0)
        c = lpv.lag_correlation(R, lags, keep_axes=1)
        for il, L in enumerate(lags):
            for k in (0, 5):
                assert c[il, k] == legacy_acf(R, L, k)
        np.testing.assert_array_equal(np.round(c.mean(axis=1), 2), logged[j])
        tc = lpv.coherence_time(lags, c.mean(axis=1))
        assert 2.0 < tc < 4.0                                             # 1/e after ~2.5-3.5 dumps ("~2-3 dumps")
        # new check (not in the log): lag-0 cross-correlation of the residual spectra of opposite LOS k, k+4,
        # measured 2026-10-01 (all |r| <= 0.06; lambda4200 LOS 3/7 = -0.0596)
        r = [lpv.lag_correlation(R[:, k], 0, other=R[:, k + 4]) for k in range(4)]
        np.testing.assert_allclose(r, opposite_resid[j], rtol=0, atol=1e-4)
    # project log 2026-09-29 "opposite LOS uncorrelated (|r| <= 0.06)": scratch acf.py, Pearson correlation of the
    # centroid <v> (diag_F[..., 1]) of lambda4922 for LOS k vs k+4: -0.012/0.018/-0.061/-0.022
    assert list(imu["diag_keys"]).index("v1") == 1
    v1 = imu["diag_F"][:, :, 2, 1]
    v1 = v1 - v1.mean(0)
    rv = [np.corrcoef(v1[:, k], v1[:, k + 4])[0, 1] for k in range(4)]
    assert [round(abs(x), 2) for x in rv] == [0.01, 0.02, 0.06, 0.02]
    assert all(round(abs(x), 2) <= 0.06 for x in rv)
    np.testing.assert_allclose([lpv.lag_correlation(v1[:, k], 0, other=v1[:, k + 4], demean=True) for k in range(4)],
                               rv, rtol=1e-12)


@pytest.mark.m424
@pytest.mark.slow
def test_systematics_m424():
    ref_path = m424_path("disc", "systematics.npz")
    ref = np.load(ref_path)
    runs = {n: lpv.open_timeseries(m424_path("disc", "{}_timeseries.npz".format(n)))
            for n in ("imu", "flux_lamfix", "flux_sm335")}
    out = lpv.systematics(lpv.open_timeseries(m424_path("disc", "flux_timeseries.npz")), runs, ref_name="flux",
                          vwin=float(ref["vwin"]), lines=LINES)
    assert sorted(out) == sorted(ref.files)
    worst = 0.0
    for k in ref.files:
        if ref[k].dtype.kind in "US":
            np.testing.assert_array_equal(out[k], ref[k])
            continue
        np.testing.assert_allclose(out[k], ref[k], rtol=1e-12, atol=0, err_msg=k)
        worst = max(worst, float(np.max(np.abs(np.asarray(out[k]) - ref[k]) / np.maximum(np.abs(ref[k]), 1e-300))))
    assert worst == 0.0, "systematics not bit-identical (max rel. diff {:.2e})".format(worst)


@pytest.mark.m424
def test_compare_timeseries_m424_memmap(imu):
    """compare_timeseries reads memory maps line by line; equals the legacy stats on in-memory copies
    (32 dumps, every 50th, imu against the following dump)."""
    A, B = imu["F"][0:1600:50], imu["F"][1:1601:50]
    assert isinstance(A, np.memmap) and A.shape == B.shape
    m = np.abs(imu["Y"]) <= 600.0
    st = lpv.compare_timeseries(A, B, mask=m)
    A, B = np.array(A), np.array(B)
    for j in range(3):
        a, b = A[:, :, j][..., m].astype(np.float64), B[:, :, j][..., m].astype(np.float64)
        leg = legacy_stats(a - a.mean(axis=0, keepdims=True), b - b.mean(axis=0, keepdims=True))
        leg["static"] = np.abs(B[:, :, j].astype(np.float64).mean(0) - A[:, :, j].astype(np.float64).mean(0)).max()
        for k, v in leg.items():
            assert st[k][j] == v, k
