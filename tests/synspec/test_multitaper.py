"""Tests of the multitaper spectra of ppmpy.synspec.spectrum (multitaper_spectra, dpss_tapers, mt_bandwidth, ftest_threshold).

PP 2026-10-03: new.
"""
import numpy as np
import pytest

from ppmpy.synspec import spectrum as sp

DT = 2834.54            # M424 mean dump spacing [s]
N = 1601


def _ar1(n, phi, nser, seed, burn=2000):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n + burn, nser))
    x = np.zeros_like(e)
    for i in range(1, n + burn):
        x[i] = phi * x[i - 1] + e[i]
    return x[burn:]


def test_tapers_normalised_and_concentrated():
    v, lam = sp.dpss_tapers(N, 3.0, 5)
    assert v.shape == (5, N)
    np.testing.assert_allclose(np.sum(v * v, axis=1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(v @ v.T, np.eye(5), atol=1e-10)
    assert np.all(lam[:4] > 0.99) and lam[4] > 0.94
    assert sp.mt_bandwidth(N, DT, 3.0) == pytest.approx(3e6 / (N * DT))


def test_tapers_warn_for_too_many():
    with pytest.warns(UserWarning):
        sp.dpss_tapers(100, 2.0, 5)


def test_white_noise_variance_conserved():
    x = np.random.default_rng(1).standard_normal((N, 64))
    for adaptive in (False, True):
        f, P = sp.multitaper_spectra(x, DT, detrend="mean", adaptive=adaptive)
        ratio = np.sum(P, axis=0) * (f[1] - f[0]) / x.var(axis=0)
        assert abs(ratio.mean() - 1) < 0.01
        # flat: the expected one-sided density 2e-6 dt var
        assert abs(np.median(P) / (2e-6 * DT) - 1) < 0.1


def test_sinusoid_power_amplitude_and_ftest():
    t = DT * np.arange(N)
    f0, A = 22.07e-6, 3.0
    rng = np.random.default_rng(2)
    x = A * np.cos(2 * np.pi * f0 * t + 0.4) + 0.3 * rng.standard_normal(N)
    f, P, F, mu = sp.multitaper_spectrum(x, DT, detrend="mean", ftest=True)
    W = sp.mt_bandwidth(N, DT)
    m = np.abs(f - f0 * 1e6) <= 1.5 * W
    assert np.sum(P[m]) * (f[1] - f[0]) == pytest.approx(A ** 2 / 2, rel=0.05)
    i = np.argmax(F)
    assert abs(f[i] - f0 * 1e6) < 0.05
    assert 2 * abs(mu[i]) == pytest.approx(A, rel=0.03)
    assert F[i] > sp.ftest_threshold(1e-6)


def test_ftest_null_distribution():
    from scipy.stats import f as fdist
    x = np.random.default_rng(3).standard_normal((N, 40))
    f, P, F, mu = sp.multitaper_spectra(x, DT, detrend="mean", over=1, ftest=True)
    q = np.mean(F > fdist.isf(0.05, 2, 8))
    assert 0.035 < q < 0.065


def test_adaptive_unbiased_on_red_noise():
    phi = 0.995
    x = _ar1(N, phi, 100, seed=4)
    f, P = sp.multitaper_spectra(x, DT, detrend="mean", over=1)
    u = f * 1e-6 * DT
    true = 2e-6 * DT / (1 + phi ** 2 - 2 * phi * np.cos(2 * np.pi * u))
    r = P.mean(axis=1) / true
    assert abs(r[f > 20].mean() - 1) < 0.05


def test_axis_shape_frange_and_returns():
    X = np.random.default_rng(5).standard_normal((3, N, 2))
    f, P, F, mu, dof, D = sp.multitaper_spectra(X, DT, axis=1, detrend="mean", frange=(1.0, 20.0), ftest=True,
                                                return_dof=True, return_detrended=True)
    assert P.shape == (3, f.size, 2) and F.shape == P.shape and mu.dtype.kind == "c" and dof.shape == P.shape
    assert D.shape == X.shape
    assert f.min() >= 1.0 and f.max() <= 20.0
    assert np.all(dof <= 10.0 + 1e-9) and np.all(dof > 1.0)
    f1, P1 = sp.multitaper_spectrum(X[1, :, 0], DT, detrend="mean", frange=(1.0, 20.0))
    np.testing.assert_allclose(P1, P[1, :, 0], rtol=1e-8)       # adaptive iterations stop per block


def test_grid_spacing_and_nyquist():
    f, P = sp.multitaper_spectrum(np.random.default_rng(6).standard_normal(N), DT, detrend="mean", over=20)
    np.testing.assert_allclose(np.diff(f), 1e6 / (20 * N * DT), rtol=1e-9)
    assert f[-1] <= 0.5e6 / DT * (1 + 1e-12)


def test_matches_tapify():
    tapify = pytest.importorskip("tapify")
    t = DT * np.arange(N)
    rng = np.random.default_rng(7)
    x = _ar1(N, 0.95, 1, seed=8)[:, 0] + 2.0 * np.cos(2 * np.pi * 30e-6 * t) + 0.5 * rng.standard_normal(N)
    xd = x - x.mean()
    f, P, F, mu = sp.multitaper_spectrum(x, DT, detrend="mean", over=20, adaptive=False, ftest=True)
    mt = tapify.MultiTaper(xd, delta_t=DT, NW=3.0, K=5)
    ft, pt, Ft = mt.periodogram(method="fft", N_padded=20 * N, adaptive_weighting=False, ftest=True)
    L = len(pt) - 1
    np.testing.assert_allclose(np.asarray(ft)[1:] * 1e6, f[:L], rtol=1e-12)
    np.testing.assert_allclose(P[:L], np.asarray(pt)[1:] * 2e-6, rtol=1e-10)
    np.testing.assert_allclose(F[:L], np.asarray(Ft)[1:], rtol=1e-9)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        sp.multitaper_spectrum(np.r_[np.ones(50), np.nan], DT, detrend="mean")
    with pytest.raises(ValueError):
        sp.multitaper_spectrum(np.ones(50), DT, detrend="mean", norm="ppmstar")
    with pytest.raises(ValueError):
        sp.dpss_tapers(20, 12.0, 3)
