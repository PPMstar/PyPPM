"""
Temporal power spectra of line-profile-variability time series (zero-crossing positions,
residual amplitudes, equivalent widths, light curves, ...).

Two evaluations of one quantity:

* ``method='fft'``: the legacy PPMstar pipeline (``get_temporal_spectrum`` of
  Temporal_spectra-threaded.ipynb, ``ppmpy.spectra.lums_temporal_spectrum``,
  the project's fig_disc_zerocross_spectrum.py): detrend, Hann window, pad with the
  mean to a long series (default 1e7 extra samples), FFT, keep positive frequencies.
  Bit-identical to the legacy code.
* ``method='dft'``: the same values evaluated directly on any frequency grid, which is
  much cheaper when only a band or a modestly oversampled grid is wanted.

Conventions
-----------
* Time in s, frequency in microHz (``conventions.CPD_PER_MUHZ`` converts to d^-1).
* Norm 'ppmstar' (legacy): P(f) = sqrt(8/3) (1e-6 dt / N) |Z(f)|^2, N = number of
  samples (not the padded length), Z = DFT of the windowed, mean-padded series. For a
  relative fluctuation it is in relative^2 per microHz; times ``PPM2_PER_REL2`` (1e12)
  gives ppm^2 per microHz. The factor sqrt(8/3) is applied for every window (legacy
  fig_disc_zerocross_spectrum.py --no-hann does the same).
* Norm 'psd': the one-sided, window-corrected power spectral density
  P(f) = 2 (1e-6 dt) |Z(f)|^2 / sum(w^2), which conserves the variance (Parseval).

Normalisation (measured in tests/synspec/test_spectrum.py)
----------------------------------------------------------
With y = w x - mean(w x) and an odd padded length, the positive-frequency sum is exactly
``sum(P) df = sqrt(8/3) / 2 * mean(y^2)`` ('ppmstar') and ``mean(y^2) / mean(w^2)``
('psd'). For a stationary zero-mean series mean(y^2) ~ mean(w^2) var(x) with
mean(w^2) = 3 (N - 1) / (8 N) for np.hanning, so the legacy 'ppmstar' power integrates to
sqrt(3/32) = 0.306 of the variance with the Hann window (sqrt(2/3) = 0.816 without):
the Hann power correction 8/3 enters as its square root and the one-sided factor 2 is
missing. Peak shapes and frequencies are unaffected; absolute ppm^2 levels are below the
variance-conserving one-sided PSD ('psd') by 16 / (3 sqrt(8/3)) = 3.27 (Hann) or
2 / sqrt(8/3) = 1.22 (no window). Measured on white noise (N = 20001): 0.3065 and 0.8165;
on the M424 zero-crossing series (fig_disc_zerocross_spectrum.py) trapz(P) / var(x) = 0.30-0.33.

PP 2026-10-01: new module; the 'fft' path is ported from
fig_disc_zerocross_spectrum.py:39-48 (spectrum()) and :51-56 (series(): A / A.mean() - 1),
and from ppmpy/spectra.py:78-141 (lums_temporal_spectrum).
"""
import math
import operator
import warnings

import numpy as np

from .conventions import CPD_PER_MUHZ

__all__ = ["HANN_POWER_FACTOR", "sample_spacing", "fft_frequencies", "temporal_power_spectrum",
           "temporal_power_spectra", "peak_frequency"]

# Hann factor of the legacy normalisation (applied to the power, as in the legacy code).
HANN_POWER_FACTOR = np.sqrt(8.0 / 3.0)

_NORMS = ("ppmstar", "psd")
_METHODS = ("fft", "dft")
_UNIFORM_TOL = 1e-12          # cycles of phase error accepted at the last sample for the uniform-grid DFT
_WORK_ELEMENTS = 1 << 22      # complex elements per DFT work array (64 MB)
_DIVISOR_NSIGMA = 3.0         # 'ratio' / divisive detrend: warn when |divisor| < this many rms of the series about it


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def sample_spacing(t, rtol=5e-3):
    """
    Mean sample spacing of a time series and its relative spread; warns when non-uniform.

    Parameters
    ----------
    t: array-like
        Sample times [s], finite and strictly increasing.
    rtol: float
        Largest accepted (max - min) spacing relative to the mean spacing.

    Returns
    -------
    dt_mean: float
        (t[-1] - t[0]) / (len(t) - 1) [s].
    rel_spread: float
        (max(diff(t)) - min(diff(t))) / dt_mean.

    Notes
    -----
    The spectra here assume uniform sampling. M424 dumps 3200-4800: spacings 2826-2843 s,
    dt_mean = 2834.54 s, rel_spread = 6.0e-3 (warns at the default rtol), and the times
    stay within 8.0 s (0.0028 dt) of the uniform grid t[0] + n dt_mean. The legacy
    fig_disc_zerocross_spectrum.py used dt = t[1] - t[0] = 2838 s, which scales every
    frequency by dt_mean / 2838 = 0.99878 (peaks 0.12 % too low).
    """
    t = np.asarray(t, dtype=np.float64)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("need a 1-D array of at least two times")
    if not np.all(np.isfinite(t)):
        raise ValueError("sample times must be finite")
    d = np.diff(t)
    if np.any(d <= 0):
        raise ValueError("sample times must increase strictly")
    dt_mean = float((t[-1] - t[0]) / (t.size - 1))
    rel_spread = float((d.max() - d.min()) / dt_mean)
    if rel_spread > rtol:
        dev = np.max(np.abs(t - (t[0] + dt_mean * np.arange(t.size))))
        warnings.warn("non-uniform sampling: spacings {:.6g}-{:.6g} s (first {:.6g} s, mean {:.6g} s, "
                      "spread {:.2e} > rtol {:.1e}); times deviate by up to {:.3g} s ({:.2e} dt) from a "
                      "uniform grid".format(d.min(), d.max(), d[0], dt_mean, rel_spread, rtol, dev,
                                            dev / dt_mean), stacklevel=2)
    return dt_mean, rel_spread


def _int_arg(name, value, minimum):
    """An integer argument (operator.index: floats are rejected) that must be >= minimum."""
    try:
        v = operator.index(value)
    except TypeError:
        raise TypeError("{} must be an integer, got {!r}".format(name, value)) from None
    if v < minimum:
        raise ValueError("{} must be >= {}, got {}".format(name, minimum, v))
    return v


def _pad_arg(pad):
    """Normalised pad: None -> 0; an integer >= 0 (floats are rejected in every path)."""
    return 0 if pad is None else _int_arg("pad", pad, 0)


def _check_dt(dt):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite, got {!r}".format(dt))


def fft_frequencies(n, dt, pad=10_000_000):
    """
    Positive frequencies [microHz] of the mean-padded FFT of n samples, exactly as the
    legacy pipeline (``np.fft.fftfreq(n + 2 (pad // 2), dt)``, positive values, times 1e6).

    Parameters
    ----------
    n: int
        Number of samples (>= 1).
    dt: float
        Sample spacing [s].
    pad: int or None
        Mean-pad length (pad // 2 samples on each side); 0 or None for no padding.
        Must be an integer >= 0 (as for the FFT itself, where np.pad rejects floats).

    Returns
    -------
    np.ndarray
        Frequencies in microHz; spacing 1e6 / ((n + 2 (pad // 2)) dt).
    """
    n = _int_arg("n", n, 1)
    pad = _pad_arg(pad)
    _check_dt(dt)
    f = np.fft.fftfreq(n + 2 * (pad // 2), dt)
    return f[f > 0] * 1e6


def peak_frequency(freq_muhz, power, frange=None):
    """
    Frequency of the highest power.

    Parameters
    ----------
    freq_muhz, power: np.ndarray
        1-D spectrum of one series, equal shapes (for :func:`temporal_power_spectra` pass
        one column, e.g. ``power[:, k]``).
    frange: (float or None, float or None), optional
        Inclusive search band [microHz]; None searches all frequencies.

    Returns
    -------
    f_muhz: float
    f_cpd: float
        The same in d^-1, ``f_muhz * conventions.CPD_PER_MUHZ``. CPD_PER_MUHZ =
        86400 * 1e-6 = 0.08639999999999999 is one ulp below the literal 0.0864 of the legacy
        printout (``f[i] * 0.0864``), so f_cpd can differ from it by 1 ulp (~1e-17 relative).
    p_max: float

    Raises
    ------
    ValueError
        For arrays that are not 1-D of equal shape, or when no frequency lies in frange.
    """
    f = np.asarray(freq_muhz)
    p = np.asarray(power)
    if f.ndim != 1 or p.shape != f.shape:
        raise ValueError("peak_frequency needs 1-D freq_muhz and power of equal shape, got {} and {} "
                         "(pass one series, e.g. power[:, k])".format(f.shape, p.shape))
    keep = _band(f, frange)
    if not keep.any():
        raise ValueError("no frequencies in frange {!r} (spectrum covers {:.6g}-{:.6g} microHz)".format(
            frange, f.min() if f.size else np.nan, f.max() if f.size else np.nan))
    i = np.flatnonzero(keep)[np.argmax(p[keep])]
    return float(f[i]), float(f[i] * CPD_PER_MUHZ), float(p[i])


def _band(f, frange):
    if frange is None:
        return np.ones(f.shape, dtype=bool)
    lo, hi = frange
    if lo is not None and hi is not None and lo > hi:
        raise ValueError("frange lower bound {!r} exceeds the upper bound {!r}".format(lo, hi))
    keep = np.ones(f.shape, dtype=bool)
    if lo is not None:
        keep &= f >= lo
    if hi is not None:
        keep &= f <= hi
    return keep


def _warn_divisor(what, divisor, resid):
    """Warn (values unchanged) when a detrend divisor is not bounded away from zero."""
    d = np.asarray(divisor, dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.square(np.asarray(resid, dtype=np.float64)))))
    dmin, dmax, amin = float(d.min()), float(d.max()), float(np.abs(d).min())
    if (dmin <= 0.0 <= dmax) or amin < _DIVISOR_NSIGMA * rms:
        val = "{:.4g}".format(dmin) if d.ndim == 0 else "{:.4g} to {:.4g}".format(dmin, dmax)
        warnings.warn("{} divides by a {} that is not bounded away from zero ({}; rms of the series about it "
                      "{:.4g}); the relative fluctuation is meaningless for a zero-mean series: use detrend='mean' "
                      "or ('poly', k, 'subtractive')".format(what[0], what[1], val, rms), UserWarning, stacklevel=5)


def _detrend(x, dt, detrend):
    """
    Apply the requested detrending to a 1-D series (see temporal_power_spectrum).

    'mean' and 'ratio' keep the dtype of x (legacy 'ratio' is literally ``A / A.mean() - 1``);
    ('poly', ...) works in float64 as ppmpy.spectra.lums_temporal_spectrum. The divisive modes
    warn when the divisor is within _DIVISOR_NSIGMA rms of zero (or, for a trend, changes
    sign); the values are those of the legacy expressions either way.
    """
    if detrend is None:
        return x
    if isinstance(detrend, str):
        if detrend == "mean":
            return x - x.mean()
        if detrend == "ratio":
            m = x.mean()
            _warn_divisor(("detrend='ratio'", "mean"), m, x - m)
            with np.errstate(divide="ignore", invalid="ignore"):
                return x / m - 1                     # == x / x.mean() - 1 (legacy), bit for bit
        raise ValueError("unknown detrend {!r}".format(detrend))
    if isinstance(detrend, (tuple, list)) and len(detrend) == 3 and detrend[0] == "poly":
        order, mode = int(detrend[1]), detrend[2]
        if mode not in ("divisive", "subtractive"):
            raise ValueError("polynomial detrend mode must be 'divisive' or 'subtractive', got {!r}".format(mode))
        # PP 2026-10-01: verbatim from ppmpy/spectra.py:112-123 (trend fitted against dt * arange(N))
        L_time = np.asarray(x, dtype=np.float64)
        times = dt * np.arange(len(L_time))
        coefs = np.polyfit(times, L_time, order)
        trend = np.polyval(coefs, times)
        if mode == "divisive":
            _warn_divisor(("detrend=('poly', {}, 'divisive')".format(order), "trend"), trend, L_time - trend)
            with np.errstate(divide="ignore", invalid="ignore"):
                return L_time / trend - 1.0
        return L_time - trend
    raise ValueError("detrend must be None, 'mean', 'ratio' or ('poly', order, 'divisive'|'subtractive'); "
                     "got {!r}".format(detrend))


def _window(n, window):
    """Window weights: np.hanning(n) for 'hann', the scalar 1.0 for None (as legacy), or an array."""
    if window is None:
        return 1.0
    if isinstance(window, str):
        if window == "hann":
            return np.hanning(n)
        raise ValueError("unknown window {!r} (use 'hann', None or an array)".format(window))
    w = np.asarray(window, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError("window array must have shape ({},), got {}".format(n, w.shape))
    if not np.all(np.isfinite(w)):
        raise ValueError("window weights must be finite")
    if not np.any(w):
        raise ValueError("window weights are all zero")
    return w


def _scale(norm, dt, n, w):
    """Power normalisation factor."""
    if norm == "ppmstar":
        return HANN_POWER_FACTOR * (1e-6 * dt / n)
    if norm == "psd":
        sw2 = float(n) if np.ndim(w) == 0 else float(np.sum(np.square(w)))
        return 2e-6 * dt / sw2
    raise ValueError("unknown norm {!r} (use {})".format(norm, " or ".join(repr(s) for s in _NORMS)))


def _check_inputs(n, dt, pad, method, chunk):
    """Validate the shared arguments; returns the normalised (pad, chunk)."""
    if n < 2:
        raise ValueError("need at least two samples")
    _check_dt(dt)
    pad = _pad_arg(pad)
    if chunk is not None:
        chunk = _int_arg("chunk", chunk, 1)
    if method not in _METHODS:
        raise ValueError("unknown method {!r} (use 'fft' or 'dft')".format(method))
    return pad, chunk


def _fft_power(xf, dt, pad, scale):
    """Legacy padded FFT of one windowed series (fig_disc_zerocross_spectrum.py:39-48)."""
    if pad > 0:
        xf = np.pad(xf, (pad // 2, pad // 2), "mean")
    dft = np.fft.fft(xf)
    f = np.fft.fftfreq(xf.size, dt)
    pos = f > 0
    return f[pos] * 1e6, scale * np.abs(dft[pos]) ** 2


def _cycles(a):
    """Fractional part (phase in cycles), so that the trigonometric arguments stay small."""
    return a - np.floor(a)


def _phase_cycles(u, N, denom=None):
    """
    Fractional part of the outer product u_k n [cycles], n = 0..N-1, without the rounding of u n.

    Rounding u n directly costs up to ~|u| n 2^-53 cycles (~1e-13 at u n ~ 800), which limited the
    DFT to ~7e-11 relative pointwise on the M424 series. Two exact reductions are used instead:

    * ``denom`` given (u = k / denom with integer k, the FFT bins): (k n mod denom) / denom in
      int64, rounded once (<= 2^-54 cycles), i.e. the twiddle factors of the FFT itself.
    * float u: u = u_hi + u_lo with u_hi on a 2^-b grid, b chosen so that
      max|u_hi| 2^b (N - 1) < 2^53. Then u_hi n is exact, so is its fractional part, and
      frac(u_hi n) + u_lo n (|u_lo| <= 2^-b-1) is rounded once (~1e-16 cycles).

    Parameters
    ----------
    u: np.ndarray
        (K,) frequencies [cycles per sample], or integer numerators k when denom is given.
    N: int
        Number of samples.
    denom: int, optional
        Common denominator of u = k / denom; requires max|k| (N - 1) < 2^62.

    Returns
    -------
    np.ndarray
        (K, N) phases [cycles] in [0, 1) up to ~1e-16.
    """
    if denom is not None:
        k = np.asarray(u, dtype=np.int64)
        return (np.outer(k, np.arange(N, dtype=np.int64)) % denom) / float(denom)
    n = np.arange(N, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    prod = (float(np.max(np.abs(u))) if u.size else 0.0) * float(N - 1)
    if prod == 0.0:
        return np.zeros((u.size, N))
    b = max(0, min(60, 52 - math.frexp(prod)[1]))          # max|u| (N - 1) 2^b < 2^52
    s = float(2 ** b)
    u_hi = np.round(u * s) / s                                # exact (power-of-two scaling)
    u_lo = u - u_hi                                           # exact
    return _cycles(np.outer(u_hi, n)) + np.outer(u_lo, n)


def _dft_abs2(Y, u, chunk=None, denom=None):
    """
    |sum_n Y[n, s] exp(-2 pi i u_k n)|^2 for all frequencies u_k [cycles per sample] and series s.

    A uniform grid (equal integer steps of k with ``denom``; otherwise to _UNIFORM_TOL cycles of
    phase at the last sample) is evaluated in blocks of ``chunk`` frequencies,
    exp(-2 pi i (u_b + j du) n) = exp(-2 pi i u_b n) exp(-2 pi i j du n): one shared block matrix
    and complex matrix products, no trigonometry per frequency. Other grids use cos/sin matrices
    per chunk. Phases are reduced without rounding u n (:func:`_phase_cycles`).

    Parameters
    ----------
    Y: np.ndarray
        (N, S) float64.
    u: np.ndarray
        (K,) frequencies in cycles per sample, or integer numerators k of u = k / denom.
    chunk: int, optional
        Frequencies per block (>= 1).
    denom: int, optional
        Common denominator (the padded length M for the FFT bins k / M).

    Returns
    -------
    np.ndarray
        (K, S) float64.
    """
    if chunk is not None:
        chunk = _int_arg("chunk", chunk, 1)
    N, S = Y.shape
    if denom is not None:
        denom = _int_arg("denom", denom, 1)
        u = np.asarray(u)
        if u.dtype.kind not in "iu":
            raise TypeError("with denom, u must hold integer numerators")
        u = u.astype(np.int64)
        if u.size and int(np.max(np.abs(u))) * (N - 1) >= 2 ** 62:
            u, denom = u / float(denom), None                 # int64 products would overflow
    K = u.size
    out = np.empty((K, S))
    if K == 0:
        return out
    uniform = False
    if K >= 3:
        if denom is not None:
            du = u[1] - u[0]
            uniform = du != 0 and bool(np.all(np.diff(u) == du))
        else:
            du = (u[-1] - u[0]) / (K - 1)
            uniform = du != 0 and float(np.max(np.abs(u - (u[0] + du * np.arange(K))))) * N <= _UNIFORM_TOL
    if uniform:
        c = chunk if chunk is not None else max(16, (1 << 19) // N)
        c = min(c, K)
        B = np.exp(-2j * np.pi * _phase_cycles(np.arange(c) * du, N, denom))             # (c, N)
        nblk = -(-K // c)
        nb = max(1, _WORK_ELEMENTS // ((N + c) * S))                                      # blocks per product
        for g0 in range(0, nblk, nb):
            g1 = min(nblk, g0 + nb)
            A = np.exp(-2j * np.pi * _phase_cycles(u[np.arange(g0, g1) * c], N, denom)).T  # (N, g)
            V = (A[:, :, None] * Y[:, None, :]).reshape(N, -1)                           # (N, g S)
            Z = (B @ V).reshape(c, g1 - g0, S)
            P = (Z.real ** 2 + Z.imag ** 2).transpose(1, 0, 2).reshape(-1, S)            # row = block c + j
            k0, k1 = g0 * c, min(K, g1 * c)
            out[k0:k1] = P[:k1 - k0]
        return out
    c = chunk if chunk is not None else max(1, _WORK_ELEMENTS // (2 * N))
    for k0 in range(0, K, c):
        ph = (2 * np.pi) * _phase_cycles(u[k0:k0 + c], N, denom)
        re = np.cos(ph) @ Y
        im = np.sin(ph) @ Y
        out[k0:k0 + c] = re ** 2 + im ** 2
    return out


def _spectra(series, dt, detrend, pad, window, norm, method, freq_muhz, frange, chunk):
    """Shared engine: list of 1-D series -> (freq, P (K, S), detrended list)."""
    if len(series) == 0:
        raise ValueError("no series given")
    n = series[0].size
    pad, chunk = _check_inputs(n, dt, pad, method, chunk)
    if method == "fft" and freq_muhz is not None:
        raise ValueError("freq_muhz is only used with method='dft' (the FFT grid is set by pad)")
    w = _window(n, window)
    scale = _scale(norm, dt, n, w)
    detrended, prepared = [], []
    for x in series:
        if x.shape != (n,):
            raise ValueError("all series must have the same length")
        if not np.all(np.isfinite(x)):
            raise ValueError("series contains non-finite values; fill the gaps first (e.g. linear "
                             "interpolation in time)")
        xd = _detrend(x, dt, detrend)
        if not np.all(np.isfinite(xd)):
            raise ValueError("the detrended series is not finite (detrend={!r} divided by zero?)".format(detrend))
        detrended.append(xd)
        prepared.append(w * xd)
    if method == "fft":
        # the band is selected per series, so only the kept part of each ~5e6-point spectrum is stored
        f = keep = out = None
        for s, xf in enumerate(prepared):
            fs_, p = _fft_power(xf, dt, pad, scale)
            if out is None:
                keep = None if frange is None else _band(fs_, frange)
                f = fs_ if keep is None else fs_[keep]
                out = np.empty((f.size, len(prepared)), dtype=p.dtype)
            out[:, s] = p if keep is None else p[keep]
            del p
        return f, out, detrended
    denom = None
    if freq_muhz is None:
        # default grid: the FFT bins u = k / M (k = 1, 2, ...), phases reduced exactly in integers
        f = fft_frequencies(n, dt, pad)
        keep = _band(f, frange)
        f = f[keep]
        u = np.flatnonzero(keep) + 1
        denom = n + 2 * (pad // 2)
    else:
        f = np.asarray(freq_muhz, dtype=np.float64).ravel()
        if not np.all(np.isfinite(f)) or np.any(f <= 0):
            raise ValueError("freq_muhz must be finite and > 0")
        f = f[_band(f, frange)]
        u = f * (1e-6 * dt)
    # mean-padding equivalence: subtract the mean np.pad would use, computed the same way
    Y = np.stack([np.asarray(xf, dtype=np.float64) - np.float64(np.mean(xf)) for xf in prepared], axis=1)
    return f, scale * _dft_abs2(Y, u, chunk, denom), detrended


# ----------------------------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------------------------
def temporal_power_spectrum(x, dt, *, detrend, pad=10_000_000, window="hann", norm="ppmstar",
                            method="fft", freq_muhz=None, frange=None, chunk=None,
                            return_detrended=False):
    """
    Temporal power spectrum of one uniformly sampled series.

    Parameters
    ----------
    x: array-like
        1-D series, finite (fill gaps before).
    dt: float
        Sample spacing [s] (see :func:`sample_spacing`).
    detrend: None, 'mean', 'ratio' or ('poly', order, 'divisive' | 'subtractive')
        Required, because a divisive polynomial trend is unsafe for zero-mean LPV series.
        None: x as given. 'mean': x - mean(x). 'ratio': ``x / x.mean() - 1`` (legacy
        fig_disc_zerocross_spectrum.py, e.g. A/<A> - 1 of a zero-crossing wavelength).
        ('poly', order, mode): polynomial of the given order fitted against
        ``dt * arange(N)`` (the uniform times, not the actual ones), then x / trend - 1
        ('divisive', positive series only) or x - trend ('subtractive'); bit-identical to
        ``ppmpy.spectra.lums_temporal_spectrum``. 'ratio' and 'divisive' warn (UserWarning,
        values unchanged) when the divisor changes sign or lies within 3 rms of the series
        about it from zero; a non-finite detrended series raises ValueError.
    pad: int or None
        Mean-pad length: pad // 2 samples equal to mean(w x) on each side (legacy 1e7).
        An integer >= 0 (floats raise TypeError, as np.pad does); None = 0. With
        method='dft' it only sets the default frequency grid.
    window: 'hann', None or array-like
        'hann' = np.hanning(N); None = no window (the legacy scalar 1.0); or N finite
        weights.
    norm: 'ppmstar' or 'psd'
        'ppmstar' (legacy): sqrt(8/3) (1e-6 dt / N) |Z|^2 for every window, in
        (unit of the detrended x)^2 per microHz; times ``conventions.PPM2_PER_REL2`` for
        ppm^2 when x is relative. 'psd': 2 (1e-6 dt) |Z|^2 / sum(w^2), the one-sided
        variance-conserving PSD. See the module docstring for the integrals.
    method: 'fft' or 'dft'
        'fft': legacy padded FFT. 'dft': direct evaluation on ``freq_muhz`` (default the
        FFT grid of ``pad``, see :func:`fft_frequencies`).
    freq_muhz: array-like, optional
        Frequencies [microHz] for method='dft', finite and > 0. Y(f) is periodic in 1e6/dt:
        f above the Nyquist frequency 0.5e6/dt aliases, and multiples of 1e6/dt alias to
        f = 0, where the mean-subtracted DFT is ~0 (the FFT's DC term c M is never returned).
    frange: (float or None, float or None), optional
        Keep only frequencies in this inclusive band [microHz] (a selection; values are
        unchanged). Legacy figure: (1.0, 0.5e6 / dt).
    chunk: int, optional
        Frequencies per block for method='dft' (an integer >= 1).
    return_detrended: bool
        Also return the detrended (pre-window) series.

    Returns
    -------
    freq_muhz: np.ndarray
        Positive frequencies [microHz].
    power: np.ndarray
        Power at those frequencies.
    detrended: np.ndarray
        Only with return_detrended.

    Notes
    -----
    Mean padding and the direct DFT. Let z_n = w_n x_n (n = 0..N-1, x detrended),
    c = mean(z), L = pad // 2 and M = N + 2L. The padded series is p_m = c + q_m with
    q_{L+n} = y_n = z_n - c and q_m = 0 outside the data. Its DFT is
    P_k = c M delta_k0 + exp(-2 pi i k L / M) sum_n y_n exp(-2 pi i k n / M), so for every
    k != 0 (mod M), |P_k|^2 = |Y(f_k)|^2 with Y(f) = sum_n y_n exp(-2 pi i f n dt) and
    f_k = k / (M dt). Without padding (M = N) the same holds because the constant c has no
    power at k != 0. Mean padding therefore only interpolates |Y(f)|^2 onto a grid M/N
    times finer than the natural spacing 1/(N dt); method='dft' evaluates Y(f) on any grid
    (y in float64, with c computed by np.mean like np.pad does; on the default grid at
    u = k / M cycles per sample, otherwise at u = f 1e-6 dt).

    Accuracy (M424 zero-crossing series, pad 1e7, 5.0e6 frequencies; reference: a long-double
    DFT with exact integer phase reduction; pointwise relative errors where P > 1e-6 of the
    peak; PP 2026-10-01). Default grid: the phases are (k n mod M) / M in integers, the FFT's
    own twiddle factors, and the dft is within 8e-13 of the reference (median 1e-15), the legacy
    FFT within 8e-12 (median 3e-15); dft vs fft <= 7.8e-12 over all 24 series (8 LOS x 3
    lines) in the plotted band [1, 0.5e6/dt]. Rounding u n directly (u n reaches N/2 ~ 800
    cycles, i.e. ~1e-13 cycles lost) gave 7e-11 (median 2e-13) and dft vs fft up to 5.2e-11.
    Explicit freq_muhz: the phases of u = f 1e-6 dt are exact (u = u_hi + u_lo, see
    _phase_cycles), but a float frequency in microHz differs from the FFT bin k / (M dt) by a
    few ulp, which moves P by up to 1.3e-10 relative near zeros of Y(f) (1e-13 of the peak);
    a uniform float grid is evaluated as u_b + j du, within _UNIFORM_TOL = 1e-12 cycles of
    phase at the last sample. Cost: N K complex multiply-adds for K frequencies (M424: the full
    pad-1e7 grid, 5e6 frequencies, 1.2 s per series; the pad-1e7 FFT 1-20 s per series, mostly
    page faults on the login node).

    Validation
    ----------
    Default arguments with detrend='ratio' reproduce fig_disc_zerocross_spectrum.py
    (spectrum() applied to series()) bit for bit, and detrend=('poly', k, mode) reproduces
    ppmpy.spectra.lums_temporal_spectrum bit for bit (tests/synspec/test_spectrum.py).
    """
    x = np.ascontiguousarray(x)                 # contiguous, as the legacy scripts' series (same summation order)
    if x.ndim != 1:
        raise ValueError("x must be 1-D (use temporal_power_spectra for many series)")
    f, P, xd = _spectra([x], dt, detrend, pad, window, norm, method, freq_muhz, frange, chunk)
    if return_detrended:
        return f, P[:, 0], xd[0]
    return f, P[:, 0]


def temporal_power_spectra(X, dt, axis=0, *, detrend, pad=10_000_000, window="hann", norm="ppmstar",
                           method="fft", freq_muhz=None, frange=None, chunk=None,
                           return_detrended=False):
    """
    Temporal power spectra of many series sampled at the same times, on one frequency grid.

    Parameters
    ----------
    X: array-like
        Series with time along ``axis`` (e.g. (n_dumps, n_los, n_lines)); at least one series.
    dt: float
        Sample spacing [s].
    axis: int
        Time axis of X.
    detrend, pad, window, norm, method, freq_muhz, frange, chunk:
        As :func:`temporal_power_spectrum`; detrending is per series.
    return_detrended: bool
        Also return the detrended series (shape of X).

    Returns
    -------
    freq_muhz: np.ndarray
        (K,) frequencies [microHz].
    power: np.ndarray
        X's shape with the time axis replaced by the K frequencies (same position).
    detrended: np.ndarray
        Only with return_detrended.

    Notes
    -----
    method='fft' runs the legacy FFT per series (results bit-identical to
    :func:`temporal_power_spectrum`) and stores only the ``frange`` band of each spectrum.
    Memory: a pad-1e7 FFT (Bluestein, length 10001601) has a transient peak of ~1.5 GB
    (measured peak RSS 1.6 GB for one series); the output adds 40 MB per series for the full
    5e6-frequency grid (12 series without frange: 2.0 GB peak; with frange=(1, 20): 1.6 GB).
    method='dft' evaluates all series together in frequency blocks (matrix products), equal
    to the single-series results to rounding.
    """
    X = np.asarray(X)
    if X.ndim == 0:
        raise ValueError("X must have a time axis")
    Xm = np.moveaxis(X, axis, 0)
    n, rest = Xm.shape[0], Xm.shape[1:]
    flat = Xm.reshape(n, -1)
    if flat.shape[1] == 0:
        raise ValueError("X contains no series (shape {})".format(X.shape))
    series = [np.ascontiguousarray(flat[:, s]) for s in range(flat.shape[1])]
    f, P, xd = _spectra(series, dt, detrend, pad, window, norm, method, freq_muhz, frange, chunk)
    power = np.moveaxis(P.reshape((f.size,) + rest), 0, axis)
    if return_detrended:
        D = np.moveaxis(np.stack(xd, axis=1).reshape((n,) + rest), 0, axis)
        return f, power, D
    return f, power
