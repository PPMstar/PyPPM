"""Tests of ppmpy.synspec.spectrum (temporal power spectra)."""
import ast
import os
import subprocess
import sys
import types
import warnings

import numpy as np
import pytest

from conftest import ROOT, m424_path
from ppmpy.synspec import conventions as cv
from ppmpy.synspec import lpv
from ppmpy.synspec import spectrum as sp

DT = 2834.54            # M424 mean dump spacing [s]
# frozen copies of the original project sources and tables (tests/synspec/legacy/README.txt)
PROJECT = os.environ.get("PPMPY_SYNSPEC_M424_PROJECT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy"))
ZEROCROSS_SCRIPT = "fig_disc_zerocross_spectrum.py"
LONG_DOUBLE = np.finfo(np.longdouble).nmant >= 63      # x87 extended precision (u n of a float64 u is exact)


def _series(n=1601, seed=3, mean=5.0):
    """Positive series with a trend, two sinusoids and noise."""
    rng = np.random.default_rng(seed)
    t = DT * np.arange(n)
    return (mean + 0.01 * t / t[-1] + 0.05 * np.sin(2 * np.pi * 150e-6 * t)
            + 0.02 * np.cos(2 * np.pi * 37e-6 * t + 0.3) + 0.02 * rng.standard_normal(n))


def _legacy_zerocross_spectrum(x, dt, pad, no_hann=False):
    """Hand copy of spectrum() in fig_disc_zerocross_spectrum.py (2026-09-30), with a.pad/a.no_hann as arguments
    (checked against the script itself in test_matches_legacy_zerocross_formula[script])."""
    N1 = x.size
    w = np.hanning(N1) if not no_hann else 1.0
    xf = w * x
    if pad:
        xf = np.pad(xf, (pad // 2, pad // 2), "mean")
    dft = np.fft.fft(xf)
    f = np.fft.fftfreq(xf.size, dt)
    pos = f > 0
    return f[pos] * 1e6, np.sqrt(8 / 3) * (1e-6 * dt / N1) * np.abs(dft[pos]) ** 2 * 1e12     # ppm^2 / muHz


def _legacy_script(dt, t=None, pad=10_000_000, no_hann=False, zdir=None, name="imu"):
    """
    spectrum() and series() of the project's fig_disc_zerocross_spectrum.py itself, extracted with ast and
    bound to the script's module globals (``a`` = its argparse namespace, dt, t, np, os); skips when the
    project checkout is not there (PPMPY_SYNSPEC_M424_PROJECT).
    """
    path = os.path.join(PROJECT, ZEROCROSS_SCRIPT)
    if not os.path.exists(path):
        pytest.skip("project script not available: {}".format(path))
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    funcs = [nd for nd in tree.body if isinstance(nd, ast.FunctionDef) and nd.name in ("spectrum", "series")]
    assert sorted(nd.name for nd in funcs) == ["series", "spectrum"]
    ns = dict(np=np, os=os, dt=dt, t=t, a=types.SimpleNamespace(pad=pad, no_hann=no_hann, zdir=zdir, name=name))
    exec(compile(ast.Module(body=funcs, type_ignores=[]), path, "exec"), ns)
    return types.SimpleNamespace(spectrum=ns["spectrum"], series=ns["series"])


def _longdouble_power(y, k, M):
    """|sum_n y_n exp(-2 pi i k n / M)|^2 in long double, phases (k n mod M) / M reduced exactly in integers."""
    twopi = 4 * np.arcsin(np.longdouble(1))
    yl = np.asarray(y, dtype=np.longdouble)
    r = np.outer(np.asarray(k, dtype=np.int64), np.arange(yl.size, dtype=np.int64)) % M
    ph = twopi * (r.astype(np.longdouble) / np.longdouble(M))
    return (np.cos(ph) @ yl) ** 2 + (np.sin(ph) @ yl) ** 2


# ----------------------------------------------------------------------------------------------
# API and detrending
# ----------------------------------------------------------------------------------------------
def test_no_heavy_imports():
    code = ("import sys; import ppmpy.synspec.spectrum;"
            "bad=[m for m in ('ppmpy.ppm','matplotlib','nugridpy','pyshtools') if m in sys.modules];"
            "print(','.join(bad))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                         env=dict(os.environ, PYTHONPATH=ROOT))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", "heavy modules imported: " + out.stdout


def test_detrend_is_required():
    with pytest.raises(TypeError):
        sp.temporal_power_spectrum(_series(), DT)
    with pytest.raises(TypeError):
        sp.temporal_power_spectra(_series()[:, None], DT)


@pytest.mark.parametrize("method", ["fft", "dft"])
def test_detrend_modes(method):
    x = _series(201)
    for det, ref in ((None, x), ("mean", x - x.mean()), ("ratio", x / x.mean() - 1)):
        _, _, xd = sp.temporal_power_spectrum(x, DT, detrend=det, pad=0, method=method, return_detrended=True)
        np.testing.assert_array_equal(xd, ref)
    for bad in ("poly", ("poly", 3), ("poly", 3, "ratio"), "linear"):
        with pytest.raises(ValueError):
            sp.temporal_power_spectrum(x, DT, detrend=bad, pad=0, method=method)


def test_return_detrended_dft_equals_fft():
    X = np.stack([_series(301, seed=s) for s in range(3)], axis=1)
    for det in ("ratio", ("poly", 2, "divisive"), ("poly", 1, "subtractive")):
        _, _, D0 = sp.temporal_power_spectra(X, DT, detrend=det, pad=1000, return_detrended=True)
        _, _, D1 = sp.temporal_power_spectra(X, DT, detrend=det, pad=1000, method="dft", return_detrended=True)
        np.testing.assert_array_equal(D1, D0)


def test_input_errors():
    x = _series(101)
    xn = x.copy()
    xn[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        sp.temporal_power_spectrum(xn, DT, detrend="mean", pad=0)
    with pytest.raises(ValueError):
        sp.temporal_power_spectrum(x, DT, detrend="mean", pad=0, freq_muhz=[10.0])      # fft + grid
    for kw in (dict(norm="amplitude"), dict(method="lomb"), dict(window="hamming"), dict(window=np.ones(3)),
               dict(pad=-2), dict(pad=-2, method="dft")):
        with pytest.raises(ValueError):
            sp.temporal_power_spectrum(x, DT, detrend="mean", **kw)
    for bad_dt in (-DT, 0.0, np.nan):
        with pytest.raises(ValueError):
            sp.temporal_power_spectrum(x, bad_dt, detrend="mean", pad=0)
    # window weights must be finite and not all zero
    w = np.hanning(x.size)
    w[3] = np.nan
    for win in (w, np.zeros(x.size)):
        for method in ("fft", "dft"):
            with pytest.raises(ValueError, match="window"):
                sp.temporal_power_spectrum(x, DT, detrend="mean", pad=0, window=win, method=method)
    # explicit dft frequencies must be finite and positive
    for fbad in ([0.0, 10.0], [-5.0], [np.nan], [np.inf]):
        with pytest.raises(ValueError, match="freq_muhz"):
            sp.temporal_power_spectrum(x, DT, detrend="mean", method="dft", freq_muhz=fbad)
    with pytest.raises(ValueError, match="frange"):
        sp.temporal_power_spectrum(x, DT, detrend="mean", pad=0, frange=(60.0, 20.0))
    # no series / no time axis
    with pytest.raises(ValueError, match="no series"):
        sp.temporal_power_spectra(np.zeros((101, 0)), DT, detrend="mean")
    with pytest.raises(ValueError, match="time axis"):
        sp.temporal_power_spectra(np.float64(1.0), DT, detrend="mean")


@pytest.mark.parametrize("method", ["fft", "dft"])
def test_pad_must_be_integer(method):
    """A float pad is rejected in both paths (np.pad itself rejects it in the legacy FFT)."""
    x = _series(101)
    with pytest.raises(TypeError, match="pad must be an integer"):
        sp.temporal_power_spectrum(x, DT, detrend="mean", pad=1e4, method=method)
    f0, p0 = sp.temporal_power_spectrum(x, DT, detrend="mean", pad=np.int64(1000), method=method)
    f1, p1 = sp.temporal_power_spectrum(x, DT, detrend="mean", pad=1000, method=method)
    np.testing.assert_array_equal(f0, f1)
    np.testing.assert_array_equal(p0, p1)


def test_fft_frequencies_validation():
    with pytest.raises(ValueError, match="pad"):
        sp.fft_frequencies(301, DT, pad=-2)
    with pytest.raises(TypeError, match="pad"):
        sp.fft_frequencies(301, DT, pad=1e4)
    with pytest.raises(ValueError):
        sp.fft_frequencies(0, DT)
    with pytest.raises(ValueError):
        sp.fft_frequencies(301, 0.0)
    assert sp.fft_frequencies(301, DT, pad=None).size == 150
    assert sp.fft_frequencies(301, DT, pad=2).size == 151


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_chunk_must_be_positive(bad):
    """chunk < 1 raises in both dft paths (it used to return uninitialised memory on the general path)."""
    x = _series(301)
    f_general = np.sort(np.random.default_rng(0).uniform(1.0, 170.0, 50))
    for kw in (dict(pad=2000), dict(freq_muhz=f_general)):
        with pytest.raises(ValueError, match="chunk"):
            sp.temporal_power_spectrum(x, DT, detrend="mean", method="dft", chunk=bad, **kw)
    with pytest.raises(ValueError, match="chunk"):
        sp.temporal_power_spectra(x[:, None], DT, detrend="mean", method="dft", chunk=bad)
    with pytest.raises(TypeError, match="chunk"):
        sp.temporal_power_spectrum(x, DT, detrend="mean", method="dft", chunk=2.5)


def test_nonfinite_detrended_series_raises():
    """An all-zero series divided by its mean (or trend) gives 0/0: ValueError, not a silent NaN spectrum."""
    z = np.zeros(301)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for det in ("ratio", ("poly", 2, "divisive")):
            for method in ("fft", "dft"):
                with pytest.raises(ValueError, match="detrended series is not finite"):
                    sp.temporal_power_spectrum(z, DT, detrend=det, pad=0, method=method)
    assert not [w for w in rec if issubclass(w.category, RuntimeWarning)]           # numpy's 0/0 warning is silenced
    assert any("not bounded away from zero" in str(w.message) for w in rec)


def test_ratio_detrend_warns_for_zero_mean_series():
    """x / mean - 1 of a zero-mean series is meaningless (powers ~1e31 in the review): warn, values unchanged."""
    x = 1e-3 * np.random.default_rng(8).standard_normal(801)
    with pytest.warns(UserWarning, match="detrend='ratio' divides by a mean that is not bounded away from zero"):
        _, _, xd = sp.temporal_power_spectrum(x, DT, detrend="ratio", pad=0, return_detrended=True)
    np.testing.assert_array_equal(xd, x / x.mean() - 1)                                # legacy expression


def test_poly_divisive_detrend_warns_for_zero_mean_series():
    spectra = pytest.importorskip("ppmpy.spectra")
    x = 1e-3 * np.random.default_rng(9).standard_normal(801)
    with pytest.warns(UserWarning, match=r"detrend=\('poly', 3, 'divisive'\) divides by a trend"):
        f1, p1, d1 = sp.temporal_power_spectrum(x, DT, detrend=("poly", 3, "divisive"), pad=500, return_detrended=True)
    f0, p0, d0 = spectra.lums_temporal_spectrum(x, DT, pad=500, detrend_order=3, detrend_mode="divisive")
    np.testing.assert_array_equal(d1, d0)                                              # values unchanged (legacy)
    np.testing.assert_array_equal(p1, p0)
    assert np.abs(d1).max() > 1.0                                                      # the hazard: garbage


def test_divisive_detrends_quiet_for_positive_series():
    x = _series(801)
    x0 = x - x.mean()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for det in ("ratio", ("poly", 3, "divisive"), ("poly", 0, "divisive")):
            sp.temporal_power_spectrum(x, DT, detrend=det, pad=0)
        for det in (None, "mean", ("poly", 3, "subtractive")):                          # no divisor: no check
            sp.temporal_power_spectrum(x0, DT, detrend=det, pad=0)
        sp.temporal_power_spectrum(-x, DT, detrend="ratio", pad=0)                     # negative, bounded away: fine


# ----------------------------------------------------------------------------------------------
# analytic checks
# ----------------------------------------------------------------------------------------------
def test_sinusoid_on_bin_exact():
    """Unwindowed, unpadded cosine exactly on bin k0: |Z_k0| = A N / 2, no leakage."""
    n, k0, A = 1000, 137, 3e-3
    t = DT * np.arange(n)
    f0 = k0 / (n * DT)
    x = A * np.cos(2 * np.pi * f0 * t + 0.7)
    f, p = sp.temporal_power_spectrum(x, DT, detrend=None, pad=0, window=None)
    i = int(np.argmax(p))
    assert f[i] == pytest.approx(f0 * 1e6, rel=1e-14) and i == k0 - 1
    assert p[i] == pytest.approx(np.sqrt(8 / 3) * 1e-6 * DT * n * A ** 2 / 4, rel=1e-12)
    assert np.delete(p, i).max() < 1e-20 * p[i]
    # 'psd': the peak bin integrates to the variance A^2/2 of the sinusoid
    f, q = sp.temporal_power_spectrum(x, DT, detrend=None, pad=0, window=None, norm="psd")
    df = 1e6 / (n * DT)
    assert q[i] * df == pytest.approx(A ** 2 / 2, rel=1e-12)
    # direct DFT at f0
    _, pd = sp.temporal_power_spectrum(x, DT, detrend=None, window=None, method="dft", freq_muhz=[f0 * 1e6])
    assert pd[0] == pytest.approx(p[i], rel=1e-10)


def test_sinusoid_hann_padded_peak():
    """Off-bin sinusoid, Hann window, mean padding: peak within one padded bin, height (A (N-1)/4)^2 scale."""
    n, A, f0 = 1601, 2e-3, 151.2345e-6
    t = DT * np.arange(n)
    x = 1.0 + A * np.sin(2 * np.pi * f0 * t)
    pad = 200_000
    f, p = sp.temporal_power_spectrum(x, DT, detrend="mean", pad=pad)
    i = int(np.argmax(p))
    df = 1e6 / ((n + pad) * DT)
    assert abs(f[i] - f0 * 1e6) <= df
    expect = np.sqrt(8 / 3) * (1e-6 * DT / n) * (A * (n - 1) / 4) ** 2     # sum(np.hanning(n)) = (n - 1) / 2
    # the padded grid misses f0 by <= df/2: main-lobe curvature ~ (N / 2M)^2 ~ 1e-5
    assert p[i] == pytest.approx(expect, rel=1e-4)
    _, pd = sp.temporal_power_spectrum(x, DT, detrend="mean", method="dft", freq_muhz=[f0 * 1e6])
    assert pd[0] == pytest.approx(expect, rel=1e-8)
    fpk, cpd, pmax = sp.peak_frequency(f, p, frange=(100.0, 200.0))
    assert fpk == f[i] and cpd == pytest.approx(f[i] * 0.0864) and pmax == p[i]


@pytest.mark.parametrize("window", ["hann", None])
def test_parseval(window):
    """
    Exact identities (odd padded length): sum(P) df = sqrt(8/3)/2 mean(y^2) ('ppmstar') and
    mean(y^2)/mean(w^2) ('psd'), y = w x - mean(w x). For white noise the measured ratio to var(x) is
    sqrt(3/32) = 0.306 (Hann) or sqrt(2/3) = 0.816 (no window) for 'ppmstar' and 1 for 'psd'.
    """
    n = 20001
    x = np.random.default_rng(11).standard_normal(n)
    w = np.hanning(n) if window == "hann" else np.ones(n)
    y = w * x - np.mean(w * x)
    df = 1e6 / (n * DT)
    ratios = {}
    for norm, exact in (("ppmstar", np.sqrt(8 / 3) / 2 * np.mean(y ** 2)),
                        ("psd", np.mean(y ** 2) / np.mean(w ** 2))):
        f, p = sp.temporal_power_spectrum(x, DT, detrend=None, pad=0, window=window, norm=norm)
        total = p.sum() * df
        assert total == pytest.approx(exact, rel=1e-10)
        ratios[norm] = total / x.var()
    print("\nParseval ({} window, white noise, N={}): sum(P) df / var(x) = {:.4f} ('ppmstar'), {:.4f} ('psd')"
          .format(window, n, ratios["ppmstar"], ratios["psd"]))
    expect = np.sqrt(3 / 32) if window == "hann" else np.sqrt(2 / 3)
    assert ratios["ppmstar"] == pytest.approx(expect, rel=0.03)
    assert ratios["psd"] == pytest.approx(1.0, rel=0.03)
    assert ratios["ppmstar"] / ratios["psd"] == pytest.approx(expect, rel=2e-3)


@pytest.mark.parametrize("method", ["fft", "dft"])
def test_parseval_custom_window(method):
    """
    An arbitrary window array, mean padding, odd padded length M: sum(P) df = mean(y^2)/mean(w^2) ('psd',
    normalised by sum(w^2) of that window) and sqrt(8/3)/2 mean(y^2) ('ppmstar'), df = 1e6 / (M dt).
    """
    n, pad = 2001, 3000
    M = n + 2 * (pad // 2)
    rng = np.random.default_rng(12)
    x = rng.standard_normal(n)
    w = rng.uniform(0.2, 1.0, n)
    y = w * x - np.mean(w * x)
    df = 1e6 / (M * DT)
    for norm, exact in (("psd", np.mean(y ** 2) / np.mean(w ** 2)), ("ppmstar", np.sqrt(8 / 3) / 2 * np.mean(y ** 2))):
        f, p = sp.temporal_power_spectrum(x, DT, detrend=None, pad=pad, window=w, norm=norm, method=method)
        assert f.size == (M - 1) // 2
        assert p.sum() * df == pytest.approx(exact, rel=1e-10)


@pytest.mark.parametrize("method", ["fft", "dft"])
def test_window_array_equals_hann(method):
    x = _series(1601)
    kw = dict(detrend="ratio", pad=4000, method=method)
    f0, p0 = sp.temporal_power_spectrum(x, DT, window="hann", **kw)
    for w in (np.hanning(x.size), list(np.hanning(x.size))):
        f1, p1 = sp.temporal_power_spectrum(x, DT, window=w, **kw)
        np.testing.assert_array_equal(f1, f0)
        np.testing.assert_array_equal(p1, p0)


# ----------------------------------------------------------------------------------------------
# legacy equivalence (bitwise)
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("order", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("mode", ["divisive", "subtractive"])
@pytest.mark.parametrize("pad", [0, 999, 4000, 65536])
def test_matches_lums_temporal_spectrum(order, mode, pad):
    spectra = pytest.importorskip("ppmpy.spectra")
    x = _series(1601, seed=order + 7)
    dt = 2838.0
    f0, p0, d0 = spectra.lums_temporal_spectrum(x, dt, pad=pad, detrend_order=order, detrend_mode=mode)
    f1, p1, d1 = sp.temporal_power_spectrum(x, dt, detrend=("poly", order, mode), pad=pad, return_detrended=True)
    np.testing.assert_array_equal(f1, f0)
    np.testing.assert_array_equal(p1, p0)
    np.testing.assert_array_equal(d1, d0)


def test_matches_lums_temporal_spectrum_list_input():
    spectra = pytest.importorskip("ppmpy.spectra")
    x = list(_series(301))
    f0, p0, _ = spectra.lums_temporal_spectrum(x, DT, pad=1000)
    f1, p1 = sp.temporal_power_spectrum(x, DT, detrend=("poly", 3, "divisive"), pad=1000)
    np.testing.assert_array_equal(f1, f0)
    np.testing.assert_array_equal(p1, p0)


@pytest.mark.parametrize("source", ["copy", "script"])
@pytest.mark.parametrize("no_hann", [False, True])
@pytest.mark.parametrize("pad", [0, 10_000, 10_001])
def test_matches_legacy_zerocross_formula(source, no_hann, pad):
    """Against the hand copy and against spectrum() of the script itself (ast; skips without the checkout)."""
    A = 4026.0 * (1 + 1e-6 * _series(1601, seed=5, mean=0.0))
    x = A / A.mean() - 1
    if source == "copy":
        f0, p0 = _legacy_zerocross_spectrum(x, DT, pad, no_hann=no_hann)
    else:
        f0, p0 = _legacy_script(dt=DT, pad=pad, no_hann=no_hann).spectrum(x)
    f1, p1 = sp.temporal_power_spectrum(A, DT, detrend="ratio", pad=pad, window=None if no_hann else "hann")
    np.testing.assert_array_equal(f1, f0)
    np.testing.assert_array_equal(p1 * cv.PPM2_PER_REL2, p0)


# ----------------------------------------------------------------------------------------------
# direct DFT
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("window", ["hann", None])
@pytest.mark.parametrize("pad", [0, 4000, 20001])
@pytest.mark.parametrize("norm", ["ppmstar", "psd"])
def test_dft_equals_fft(window, pad, norm):
    """
    At the FFT frequencies: default grid (exact integer phases) <= 1e-11 pointwise where P > 1e-6 max;
    explicit freq_muhz (float frequencies, a few ulp off the bins k / (M dt)) <= 1e-11 of the peak.
    """
    x = _series(1601)
    kw = dict(detrend="ratio", pad=pad, window=window, norm=norm)
    f, p = sp.temporal_power_spectrum(x, DT, **kw)
    fd, pd = sp.temporal_power_spectrum(x, DT, method="dft", **kw)            # uniform-grid path
    np.testing.assert_array_equal(fd, f)
    big = p > 1e-6 * p.max()
    assert np.abs(pd - p).max() <= 1e-12 * p.max()
    assert (np.abs(pd - p) / p)[big].max() <= 1e-11
    sel = np.sort(np.random.default_rng(1).choice(f.size, 400, replace=False))
    fg, pg = sp.temporal_power_spectrum(x, DT, method="dft", freq_muhz=f[sel], **kw)   # general path
    np.testing.assert_array_equal(fg, f[sel])
    assert np.abs(pg - p[sel]).max() <= 1e-11 * p.max()


@pytest.mark.skipif(not LONG_DOUBLE, reason="needs an 80-bit long double")
def test_phase_cycles_exact():
    """
    frac(u n) without rounding u n: <= 2^-53 cycles from the exact value (float u: split u_hi + u_lo;
    integer numerators: (k n mod M) / M). Rounding u n directly is off by up to ~|u| n 2^-53 ~ 1e-13 cycles.
    """
    rng = np.random.default_rng(2)
    N = 1601
    u = np.r_[rng.uniform(0.0, 0.5, 200), rng.uniform(-3.0, 3.0, 50), 1e-9 * rng.uniform(0, 1, 10), 0.5, 0.25]
    nl = np.arange(N, dtype=np.longdouble)

    def dist(ph, ref):                                        # distance on the unit circle [cycles]
        d = np.asarray(ph, dtype=np.longdouble) - ref
        return float(np.max(np.abs(d - np.round(d))))

    ref = np.outer(u.astype(np.longdouble), nl)               # exact: 53-bit u times 11-bit n in a 64-bit mantissa
    ref = ref - np.floor(ref)
    assert dist(sp._phase_cycles(u, N), ref) <= 2.0 ** -53
    naive = np.outer(u, np.arange(N, dtype=np.float64))
    assert dist(naive - np.floor(naive), ref) > 1e-14         # what the split avoids
    M = 10_001_601
    k = rng.integers(1, M // 2, 300)
    ph = sp._phase_cycles(k, N, denom=M)
    ref = (np.outer(k, np.arange(N)) % M).astype(np.longdouble) / M
    assert dist(ph, ref) <= 2.0 ** -54 + 2.0 ** -62           # one rounding of a value < 1


@pytest.mark.skipif(not LONG_DOUBLE, reason="needs an 80-bit long double")
@pytest.mark.parametrize("pad", [100_000, 1_000_000])
def test_dft_against_long_double(pad):
    """
    Pointwise accuracy against a long-double DFT with exact phases, 1500 frequencies with P > 1e-6 max.
    Measured (default grid, exact integer phases): max 1.0e-13, median 9e-16 (general path max 5e-14); the
    FFT max 6e-14. Rounding u n directly gave max 1-3e-11, median 8e-14; the float split u_hi + u_lo alone
    max 1.1e-12, median 9e-15 (both fail the bounds below).
    """
    x = _series(1601, seed=4)
    n = x.size
    M = n + 2 * (pad // 2)
    f, p = sp.temporal_power_spectrum(x, DT, detrend="ratio", pad=pad)
    _, pd = sp.temporal_power_spectrum(x, DT, detrend="ratio", pad=pad, method="dft")
    xf = np.hanning(n) * (x / x.mean() - 1)
    scale = sp.HANN_POWER_FACTOR * (1e-6 * DT / n)
    big = np.flatnonzero(p > 1e-6 * p.max())
    idx = np.sort(np.random.default_rng(pad).choice(big, 1500, replace=False))
    ref = (scale * _longdouble_power(xf - np.mean(xf), idx + 1, M)).astype(np.float64)
    e_dft = np.abs(pd[idx] - ref) / ref
    e_fft = np.abs(p[idx] - ref) / ref
    # the general (non-uniform) path with the same exact phases
    e_gen = np.abs(scale * sp._dft_abs2((xf - np.mean(xf))[:, None], idx + 1, denom=M)[:, 0] - ref) / ref
    print("\npad {}: vs long double, dft max {:.1e} median {:.1e}; general path max {:.1e}; fft max {:.1e} "
          "median {:.1e}".format(pad, e_dft.max(), np.median(e_dft), e_gen.max(), e_fft.max(), np.median(e_fft)))
    for e in (e_dft, e_gen):
        assert e.max() <= 1e-12 and np.median(e) <= 1e-14


def test_dft_float32_unwindowed():
    """The mean that np.pad uses is computed in the series' dtype; the DFT path must use the same one."""
    x = (_series(801) * 1e3).astype(np.float32)
    kw = dict(detrend=None, pad=3000, window=None)
    f, p = sp.temporal_power_spectrum(x, DT, **kw)
    _, pd = sp.temporal_power_spectrum(x, DT, method="dft", **kw)
    assert np.abs(pd - p).max() <= 1e-10 * p.max()


def test_dft_chunking_and_band():
    x = _series(1601)
    f, p = sp.temporal_power_spectrum(x, DT, detrend="mean", pad=30_000, frange=(20.0, 60.0))
    fa, pa = sp.temporal_power_spectrum(x, DT, detrend="mean", pad=30_000)
    keep = (fa >= 20.0) & (fa <= 60.0)
    np.testing.assert_array_equal(f, fa[keep])                                  # frange is a pure selection
    np.testing.assert_array_equal(p, pa[keep])
    f_lo, _ = sp.temporal_power_spectrum(x, DT, detrend="mean", pad=30_000, frange=(None, 60.0))
    np.testing.assert_array_equal(f_lo, fa[fa <= 60.0])
    fg = f[::7][:300] * (1 + 1e-9 * np.arange(300) ** 2)                       # non-uniform: general path
    for chunk in (1, 7, 1000):
        fd, pd = sp.temporal_power_spectrum(x, DT, detrend="mean", pad=30_000, frange=(20.0, 60.0),
                                            method="dft", chunk=chunk)
        np.testing.assert_array_equal(fd, f)
        assert np.abs(pd - p).max() <= 1e-10 * p.max()
        _, pg = sp.temporal_power_spectrum(x, DT, detrend="mean", method="dft", freq_muhz=fg, chunk=chunk)
        _, pg0 = sp.temporal_power_spectrum(x, DT, detrend="mean", method="dft", freq_muhz=fg)
        assert np.abs(pg - pg0).max() <= 1e-12 * pg0.max()


def test_fft_frequencies_match_fft():
    x = _series(1601)
    for pad in (None, 0, 1, 999, 1000):
        f, _ = sp.temporal_power_spectrum(x, DT, detrend="mean", pad=pad)
        np.testing.assert_array_equal(sp.fft_frequencies(1601, DT, pad), f)


# ----------------------------------------------------------------------------------------------
# many series
# ----------------------------------------------------------------------------------------------
def test_spectra_many_series():
    rng = np.random.default_rng(4)
    X = 2.0 + 0.1 * rng.standard_normal((601, 4, 3))
    kw = dict(detrend="ratio", pad=5000)
    f, P = sp.temporal_power_spectra(X, DT, **kw)
    assert P.shape == (f.size, 4, 3)
    for k in range(4):
        for j in range(3):
            fs_, ps = sp.temporal_power_spectrum(X[:, k, j], DT, **kw)
            np.testing.assert_array_equal(fs_, f)
            np.testing.assert_array_equal(P[:, k, j], ps)                      # fft: bitwise per series
    fd, Pd = sp.temporal_power_spectra(X, DT, method="dft", **kw)
    np.testing.assert_array_equal(fd, f)
    assert np.abs(Pd - P).max() <= 1e-10 * P.max()
    # frange: the band is kept per series inside the loop; equal to selecting it afterwards
    fb, Pb = sp.temporal_power_spectra(X, DT, frange=(10.0, 40.0), **kw)
    keep = (f >= 10.0) & (f <= 40.0)
    np.testing.assert_array_equal(fb, f[keep])
    np.testing.assert_array_equal(Pb, P[keep])
    # time along another axis; detrended series returned in X's layout
    Xt = np.moveaxis(X, 0, 1)                                                   # (4, 601, 3)
    f2, P2, D2 = sp.temporal_power_spectra(Xt, DT, axis=1, return_detrended=True, **kw)
    np.testing.assert_array_equal(P2, np.moveaxis(P, 0, 1))
    for k in range(4):
        for j in range(3):
            x = Xt[k, :, j]
            np.testing.assert_array_equal(D2[k, :, j], x / x.mean() - 1)


# ----------------------------------------------------------------------------------------------
# sampling, peaks
# ----------------------------------------------------------------------------------------------
def test_sample_spacing():
    t = 100.0 + DT * np.arange(50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dtm, spread = sp.sample_spacing(t)
    assert dtm == pytest.approx(DT, rel=1e-14) and spread < 1e-12
    tj = t + np.r_[0.0, 20.0, np.zeros(48)]
    with pytest.warns(UserWarning, match="non-uniform"):
        sp.sample_spacing(tj)
    with pytest.raises(ValueError):
        sp.sample_spacing(t[::-1])
    with pytest.raises(ValueError):
        sp.sample_spacing(t[:1])
    for bad in ([0.0, 1.0, np.nan, 3.0], [0.0, 1.0, np.inf]):
        with pytest.raises(ValueError, match="finite"):
            sp.sample_spacing(bad)


def test_peak_frequency():
    f = np.linspace(1.0, 100.0, 991)
    p = np.exp(-(f - 30.0) ** 2) + 2.0 * np.exp(-(f - 70.0) ** 2)
    fpk, cpd, pmax = sp.peak_frequency(f, p)                                     # frange=None: all frequencies
    assert fpk == f[np.argmax(p)] == pytest.approx(70.0) and pmax == p.max()
    assert cpd == fpk * cv.CPD_PER_MUHZ
    assert abs(cpd - fpk * 0.0864) <= np.spacing(cpd)                           # legacy literal: <= 1 ulp
    assert sp.peak_frequency(f, p, frange=(None, 50.0))[0] == pytest.approx(30.0)
    assert sp.peak_frequency(f, p, frange=(20.0, 50.0))[0] == pytest.approx(30.0)
    with pytest.raises(ValueError, match="no frequencies in frange"):
        sp.peak_frequency(f, p, frange=(200.0, 300.0))
    with pytest.raises(ValueError, match="frange"):
        sp.peak_frequency(f, p, frange=(50.0, 20.0))
    with pytest.raises(ValueError, match="1-D"):
        sp.peak_frequency(f, np.stack([p, p], axis=1))                           # temporal_power_spectra output
    with pytest.raises(ValueError, match="1-D"):
        sp.peak_frequency(f, p[:-1])


# ----------------------------------------------------------------------------------------------
# M424 regression
# ----------------------------------------------------------------------------------------------
LOS1_PEAKS_MUHZ = (12.8048, 29.1453, 30.8771)      # highest power in [1, 0.5e6/dt], legacy pad-1e7 grid
LOS1_NBAD = (0, 41, 0)                             # dumps without a zero crossing (filled)


@pytest.mark.m424
def test_m424_sample_spacing():
    z = np.load(m424_path("disc", "zerocross_imu_los1.npz"))
    with pytest.warns(UserWarning, match="non-uniform"):
        dtm, spread = sp.sample_spacing(z["t_s"])
    assert dtm == pytest.approx(2834.54, abs=1e-6)
    assert spread == pytest.approx(17.0 / 2834.54, rel=1e-12)                  # spacings 2826..2843 s
    assert z["t_s"][1] - z["t_s"][0] == 2838.0


@pytest.mark.m424
@pytest.mark.slow
def test_m424_zerocross_spectrum_los1():
    """
    The arrays behind fig_disc_zerocross_spectrum.py for los1 (defaults: --name imu, pad 1e7, Hann), bit for
    bit against spectrum() and series() of the script itself: x = A/<A> - 1 of the gap-filled zero-crossing
    wavelength, dt = t_s[1] - t_s[0]; the plotted arrays are f, P in [1, fnyq].
    """
    zfile = m424_path("disc", "zerocross_imu_los1.npz")
    z = np.load(zfile)
    t = z["t_s"] - z["t_s"][0]
    dt = z["t_s"][1] - z["t_s"][0]
    fnyq = 0.5e6 / dt
    leg = _legacy_script(dt=dt, t=t, zdir=os.path.dirname(zfile), name="imu")
    A, bad = lpv.fill_gaps(t, z["A"])
    fb, Pb = sp.temporal_power_spectra(A, dt, detrend="ratio", frange=(1.0, fnyq))       # the module's band
    fd, Pd = sp.temporal_power_spectra(A, dt, detrend="ratio", frange=(1.0, fnyq), method="dft")
    np.testing.assert_array_equal(fd, fb)
    lines = ["HEI4026", "HEII4200", "HEI4922"]
    nbad = []
    print()
    for j in range(3):
        x0, nb = leg.series(1, j)
        nbad.append(nb)
        f0, p0 = leg.spectrum(x0)
        keep = (f0 >= 1.0) & (f0 <= fnyq)                                      # script lines 72-73
        np.testing.assert_array_equal(fb, f0[keep])
        np.testing.assert_array_equal(Pb[:, j] * cv.PPM2_PER_REL2, p0[keep])
        if j == 0:                                                             # the whole output, incl. f < 1 muHz
            f1, p1, x1 = sp.temporal_power_spectrum(A[:, 0], dt, detrend="ratio", return_detrended=True)
            np.testing.assert_array_equal(x1, x0)
            np.testing.assert_array_equal(f1, f0)
            np.testing.assert_array_equal(p1 * cv.PPM2_PER_REL2, p0)
        # direct DFT on the same band (default grid: exact integer phases), pointwise
        big = Pb[:, j] > 1e-6 * Pb[:, j].max()
        rel = (np.abs(Pd[:, j] - Pb[:, j]) / Pb[:, j])[big].max()
        assert rel <= 5e-11                                                    # measured 7.8e-12 (FFT-limited)
        fpk, cpd, _ = sp.peak_frequency(fb, Pb[:, j])
        assert fpk == pytest.approx(LOS1_PEAKS_MUHZ[j], abs=1e-4)
        i = int(np.argmax(p0))                                                 # the script's printout: whole range
        print("los1 {}: {} dumps filled; highest power at {:.4f} muHz ({:.4f} d^-1) [legacy printout, whole "
              "range], in [1, {:.1f}] muHz at {:.4f} muHz ({:.4f} d^-1); trapz power / var(x) = {:.4f}; dft vs fft "
              "{:.1e} pointwise".format(lines[j], nb, f0[i], f0[i] * 0.0864, fnyq, fpk, cpd,
                                        np.trapz(p0, f0) / (x0.var() * 1e12), rel))
    assert tuple(nbad) == LOS1_NBAD
    assert tuple(int(v) for v in bad.sum(axis=0)) == LOS1_NBAD
