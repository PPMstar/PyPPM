"""
Line diagnostics and broadening on the logarithmic velocity grid.

* :func:`line_diagnostics` / :func:`diagnostics_array`: equivalent width, centroid, width, FWHM and
  central depth of continuum-normalised profiles on y = c ln(lambda / lambda_ref);
  :func:`equivalent_width_native` on a native FASTWIND wavelength grid.
* :func:`kernel_rotation`, :func:`kernel_gauss`, :func:`kernel_rt`, :func:`broaden`: unit-area
  broadening kernels on the velocity grid (Gray 2005) and their convolution with the absorption depth.
* :func:`fit_broadening`, :func:`macroturbulence_fits`, :func:`gof_map`, :func:`fourier_first_zero`:
  the macroturbulence / v sin i analysis of mock observations (as for observed O stars: Sundqvist et
  al. 2013; Delbroek et al. 2025, 2026), template = the same disc integration without velocities.

Conventions
-----------
* Profiles F are continuum-normalised (F = 1 in the continuum); the absorption depth is d = 1 - F.
* The last axis of F is the velocity grid y [km/s], strictly increasing. :func:`line_diagnostics`,
  :func:`diagnostics_array`, :func:`equivalent_width`, :func:`equivalent_width_native` and
  :func:`broaden` accept leading axes and process them row by row (the numbers are identical to one
  call per row, whatever the memory layout of F). :func:`shift_profile`, :func:`fit_broadening`,
  :func:`gof_map`, :func:`fourier_amplitude` and :func:`fourier_first_zero` take one 1-D profile
  (:func:`macroturbulence_fits` loops the fits over leading axes).
* Fits shift a model profile by +s on the y grid: model(y) = B(y - s), i.e. ``np.interp(y, y + s, B)``.
* Windows are |y| <= vwin for the moments (:func:`line_diagnostics`, ``inclusive=True``) and
  |y| < vwin for the fits, GOF maps and Fourier transforms, as in the legacy scripts.

Notes
-----
Bitwise reproduction of the stored M424 products (diag_F / diag_F0, the fw_disc_vmac numbers) is
guaranteed under numpy 1.26, the version of the production container. Under numpy >= 2 (checked
with 2.2.2) whatever goes through np.exp or np.fft differs at the ulp level (np.exp is
SIMD-vectorised differently; np.fft was rewritten, pocketfft C++):

* the default EW: 2317 of the 5401 factors exp(y/c) of the M424 grid differ in the last bit, so
  ew differs by <= 2.2e-16 A;
* :func:`kernel_gauss` and :func:`kernel_rt` (1 ulp in some entries), hence everything broadened
  with them: fit rms <= 5e-15 relative, GOF chi^2 ~1e-14 relative. The GOF minima and the fw_disc_vmac
  CSV strings of the M424 products do not change; a fitted value can move by rounding only (imu
  los8 lambda4026, RT: the refined zeta 97.99999999999989 and the coarse 98.0 give the same rms to
  rounding; numpy 1.26 picks the refined, numpy 2 the coarse point, shifts -3 +- 4e-16 likewise);
* the Fourier amplitudes (~3e-16 absolute).

v1, sigma, fwhm, depth, the EW without the Jacobian, :func:`equivalent_width_native`,
:func:`kernel_rotation` and :func:`shift_profile` are bitwise equal for the same input (the
disc-integrated F themselves, rebuilt under numpy 2, differ at the 1e-16 level). Bitwise tests
against stored products (or numbers from another numpy) should therefore avoid exp and FFT or run
in the container; comparisons with the legacy code run under the same numpy are bitwise on both.

PP 2026-10-01: ported from the project's fw_disc.py (diagnostics, k_rot, k_gauss, k_rt, broaden),
fig_disc_vmac.py (fit, GOF map, Fourier transform) and fig_disc_profiles.py (summaries over the
lines of sight). Defaults reproduce the legacy numbers bit for bit (tests/synspec/test_diagnostics.py).
"""
import numpy as np
from scipy.special import erf

from .conventions import C_KMS

# PP 2026-10-01: numpy >= 2.0 renamed trapz to trapezoid (same algorithm; trapz is a deprecated alias there).
# numpy 1.26 (the M424 production container) has only trapz; the sums are the same on both (see Notes above
# for the parts, np.exp and the FFT, that are not bitwise equal across numpy versions).
_trapz = getattr(np, "trapezoid", None) or np.trapz

DIAG_KEYS = ("ew", "v1", "sigma", "fwhm", "depth")
"""Diagnostics of :func:`line_diagnostics`, in the order of the stored ``diag_keys`` of the M424 products."""

MACRO_COLUMNS = ("zeta_RT", "dv_RT", "rms_RT", "vmac_G", "dv_G", "rms_G")
"""Columns of :func:`macroturbulence_fits` (as figures/fw_disc_vmac_<tag>.csv): zeta_RT, v_mac [km/s];
dv_RT, dv_G [km/s] = the fitted shift of the template on the y grid (> 0 = redward, so about -<v_los>
in the v > 0 towards-the-observer convention); rms_* = rms residual in the fit window."""

GOF_LEVELS = (2.30, 6.18, 11.83)
"""Delta chi^2 of the 1, 2, 3 sigma regions for two free parameters (contours of :func:`gof_map`)."""

FT_Q1_ROT = 0.660
"""sigma_1 v sin i of the first zero of the rotation profile's Fourier transform (Gray 2005, eps = 0.6)."""

DEFAULT_PAR_GRID = np.arange(0.0, 151.0, 2.0)          # fig_disc_vmac.py: grid of zeta_RT / v_mac [km/s]
DEFAULT_SHIFTS = np.arange(-15.0, 15.01, 0.5)          # fig_disc_vmac.py: velocity shifts [km/s]
DEFAULT_GOF_GRID = np.arange(0.0, 151.0, 3.0)          # fig_disc_vmac.py: v sin i and zeta_RT axes [km/s]


def grid_step(y, rtol=1e-9):
    """
    Step of a uniform velocity grid.

    Parameters
    ----------
    y: array-like
        Velocity grid [km/s], increasing.
    rtol: float
        Allowed relative deviation of any step from the mean step.

    Returns
    -------
    float
        (y[-1] - y[0]) / (len(y) - 1); exactly 1.0 for the M424 grid.

    Raises
    ------
    ValueError
        If the grid is not uniform.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or y.size < 2:
        raise ValueError("need a 1-D grid with at least 2 points")
    dv = (y[-1] - y[0]) / (y.size - 1)
    if not np.allclose(np.diff(y), dv, rtol=rtol, atol=0.0):
        raise ValueError("velocity grid is not uniform")
    return float(dv)


def line_diagnostics(F, y, lref, vwin=None, keys=DIAG_KEYS, ew_jacobian=True, inclusive=True, dv=None,
                     chunk=2048):
    """
    Equivalent width, centroid, width, FWHM and central depth of continuum-normalised profiles.

    With d = 1 - F (absorption depth):

    * ew     [Angstrom] = lref / c * int d exp(y/c) dy over the whole grid (d lambda = lambda dy / c);
    * v1     [km/s]     = int y d dy / int d dy within the window (centroid; about -<v> for a
      Doppler shift v > 0 towards the observer);
    * sigma  [km/s]     = sqrt(int (y - v1)^2 d dy / int d dy) within the window;
    * fwhm   [km/s]     = (hi - lo) dv, with lo / hi the nearest grid points left / right of the
      deepest point k where d < d[k] / 2 (whole grid; integer grid steps, no interpolation);
    * depth             = d[k], the largest absorption depth.

    Integrals are trapezoidal on the grid points (np.trapz).

    Parameters
    ----------
    F: array-like
        Normalised profiles, last axis = y. Leading axes are allowed.
    y: array-like
        Velocity grid [km/s], 1-D, strictly increasing, >= 2 points; uniform for 'fwhm'.
    lref: float or array-like
        Velocity zero point [Angstrom]; broadcast against F.shape[:-1] (e.g. one per line).
    vwin: float or None
        Moment window |y| <= vwin [km/s] for v1 and sigma; it must hold >= 2 grid points (checked
        only when v1 or sigma is requested). None = whole grid, the same numbers as
        the legacy vwin >= the grid half-width (fw_disc_dumps.py, vwin = VY = 2700). The legacy
        default 400 km/s (dump-3200 products) cuts the Stark wings of lambda4026/4200, so that v1
        follows a Doppler shift only by 0.80/0.72.
    keys: sequence of str
        Diagnostics to compute, from :data:`DIAG_KEYS`.
    ew_jacobian: bool
        True (default): EW with the factor lambda/lref = exp(y/c) (fw_disc.py since 2026-09-29).
        False: the earlier formula without it, lref / c int d dy, used for RUN/disc_los8.npz and
        disc_los8_imu.npz; it gives a larger EW, by 8.4-8.9e-5 A for lambda4026 (the documented
        "+9e-5 A") and by <= 1e-5 A for the others (dump 3200).
    inclusive: bool
        True (default): window |y| <= vwin (fw_disc.py since 2026-09-29). False: |y| < vwin, as for
        RUN/disc_los8*.npz (vwin = 400, so the grid points +-400 km/s were left out).
    dv: float, optional
        Grid step for 'fwhm'; default :func:`grid_step` (y). The legacy code used DV = 1.0.
    chunk: int
        Rows processed at once (bounds the memory; the numbers do not depend on it).

    Returns
    -------
    dict
        key -> array of shape F.shape[:-1] (numpy scalars for a single profile). ew, v1, sigma and
        fwhm are float64; depth has the dtype of 1 - F.

    Raises
    ------
    ValueError
        Unknown keys; y not 1-D, not strictly increasing or shorter than 2 points; last axis of F
        not matching y; a moment window with fewer than 2 grid points (v1 / sigma requested).

    Validation
    ----------
    Bit-identical with fw_disc.diagnostics(F, j, vwin) for float64 profiles (the legacy code, one
    profile per call), for any memory layout of F (each block of rows is made C-contiguous, as the
    legacy 1-D d always was). With vwin=400, ew_jacobian=False, inclusive=False it reproduces
    diag_F / diag_F0 of RUN/disc_los8.npz and disc_los8_imu.npz bit for bit; with the defaults it
    reproduces diag_F / diag_F0 of the per-dump products bit for bit from the float64 F rebuilt
    with the project's fw_disc (flux dumps 3200 and 4800 checked; under numpy 1.26, see the module
    Notes). A NaN in a profile gives, as in the legacy code, NaN ew, NaN v1 / sigma if it lies
    inside the moment window, fwhm 0 and depth NaN.

    Notes
    -----
    float32 profiles are processed like the legacy code did (d = 1 - F stays float32, so the
    trapezoid sums of d alone add float32 neighbours); cast to float64 first for full precision.
    The stored M424 float32 F (rounded by <= 2^-24 ~ 6e-8) reproduce the stored diagnostics (computed
    from float64 F) to 7.3e-8 relative in ew, 3e-8 in depth, 6.1e-5 km/s in v1, 1.8e-6 relative in
    sigma (whole grid: the far Stark wings carry weight y^2) and exactly in fwhm (imu and flux dump
    3200, every 40th dump of imu_timeseries.npz).
    """
    keys = tuple(keys)
    bad = [k for k in keys if k not in DIAG_KEYS]
    if bad:
        raise ValueError("unknown diagnostics {} (choose from {})".format(bad, DIAG_KEYS))
    F = np.asarray(F)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or y.size < 2 or np.any(np.diff(y) <= 0):
        raise ValueError("y must be a 1-D, strictly increasing grid with at least 2 points")
    if F.ndim < 1 or F.shape[-1] != y.size:
        raise ValueError("last axis of F ({}) must match the grid y ({})".format(F.shape, y.shape))
    lead = F.shape[:-1]
    ny = y.size
    rows = F.reshape(int(np.prod(lead)), ny)
    nrow = rows.shape[0]
    lr = np.broadcast_to(np.asarray(lref, dtype=np.float64), lead).reshape(-1)
    if "fwhm" in keys and dv is None:
        dv = grid_step(y)
    need_mom = "v1" in keys or "sigma" in keys
    if vwin is None:
        m = None
    else:
        m = np.abs(y) <= vwin if inclusive else np.abs(y) < vwin
        ym = y[m]
        if need_mom and ym.size < 2:
            raise ValueError("moment window |y| {} {} holds {} grid point(s); need at least 2".format(
                "<=" if inclusive else "<", vwin, ym.size))
    e = np.exp(y / C_KMS) if ew_jacobian else None
    ddtype = (1.0 - rows[:1, :1]).dtype
    out = {k: np.empty(nrow, dtype=ddtype if k == "depth" else np.float64) for k in keys}
    idx = np.arange(ny)
    for i0 in range(0, nrow, chunk):
        sl = slice(i0, min(i0 + chunk, nrow))
        # C order: numpy sums a non-contiguous last axis (Fortran-ordered or strided F) in a different
        # order (up to ~1e-13 km/s off); the legacy 1-D d was always contiguous. A no-op for C input.
        d = np.ascontiguousarray(1.0 - rows[sl])
        if "ew" in keys:
            # PP 2026-10-01: fw_disc.diagnostics, ew = trapz(d * exp(Y / c), Y) * LREF[j] / c (left to right)
            s = _trapz(d * e, y, axis=-1) if ew_jacobian else _trapz(d, y, axis=-1)
            out["ew"][sl] = s * lr[sl] / C_KMS
        if need_mom:
            if m is None:
                dm, yy = d, y
            else:
                # C order matters: numpy sums a non-contiguous window in a different order (~1e-13 off)
                dm, yy = np.ascontiguousarray(d[:, m]), ym
            m0 = _trapz(dm, yy, axis=-1)
            v1 = _trapz(yy * dm, yy, axis=-1) / m0
            if "v1" in keys:
                out["v1"][sl] = v1
            if "sigma" in keys:
                out["sigma"][sl] = np.sqrt(_trapz((yy - v1[:, None]) ** 2 * dm, yy, axis=-1) / m0)
        if "fwhm" in keys or "depth" in keys:
            # PP 2026-10-01: vectorised form of k = argmax(d); lo = k - argmax(d[k::-1] < d[k] / 2);
            # hi = k + argmax(d[k:] < d[k] / 2) (no point below half on a side -> that side stays at k)
            k = np.argmax(d, axis=-1)
            dk = d[np.arange(d.shape[0]), k]
            if "depth" in keys:
                out["depth"][sl] = dk
            if "fwhm" in keys:
                half = dk / 2
                below = d < half[:, None]
                left = below & (idx[None, :] <= k[:, None])
                right = below & (idx[None, :] >= k[:, None])
                lo = np.where(left.any(axis=-1), ny - 1 - np.argmax(left[:, ::-1], axis=-1), k)
                hi = np.where(right.any(axis=-1), np.argmax(right, axis=-1), k)
                out["fwhm"][sl] = (hi - lo) * dv
    return {k: (out[k].reshape(lead)[()] if lead == () else out[k].reshape(lead)) for k in keys}


def diagnostics_array(F, y, lref, vwin=None, keys=DIAG_KEYS, **kwargs):
    """
    :func:`line_diagnostics` stacked on a last axis, the layout of the stored ``diag_F`` / ``diag_F0``
    (with ``diag_keys``) of the M424 products.

    Parameters
    ----------
    F, y, lref, vwin, keys:
        As :func:`line_diagnostics`; e.g. F (8, 3, ny) with lref (3,) gives (8, 3, 5).
    **kwargs:
        Further options of :func:`line_diagnostics` (ew_jacobian, inclusive, dv, chunk).

    Returns
    -------
    np.ndarray
        F.shape[:-1] + (len(keys),), float64.

    Validation
    ----------
    PP 2026-10-01: as fw_disc_los.py:169 / fw_disc_dumps.py:99,
    ``np.array([[[fd.diagnostics(F[k, j], j, vwin)[q] for q in keys] for j in ...] for k in ...])``.
    """
    r = line_diagnostics(F, y, lref, vwin=vwin, keys=keys, **kwargs)
    return np.stack([np.asarray(r[k], dtype=np.float64) for k in keys], axis=-1)


def diagnostics_summary(diag, axis=0):
    """
    Mean and standard deviation (ddof 0) of diagnostics over an axis, e.g. over the lines of sight.

    Parameters
    ----------
    diag: array-like
        e.g. (8, 3, 5) from :func:`diagnostics_array`.
    axis: int
        Axis to summarise (default 0, the lines of sight).

    Returns
    -------
    mean, std: np.ndarray

    Notes
    -----
    PP 2026-10-01: the printout of fig_disc_profiles.py:62-65 (mean (std) over the 8 lines of sight).
    Bit-identical to that printout only when called on each 1-D slice (diag[:, j, i]): numpy reduces
    axis 0 of a 3-D array sequentially but a 1-D array pairwise, so the whole-array call differs by
    up to ~3e-14 relative (invisible in the printed digits).
    """
    diag = np.asarray(diag)
    return diag.mean(axis=axis), diag.std(axis=axis)


def los_mean_residuals(F, axis=0):
    """
    Mean profile over an axis (the lines of sight) and every profile's deviation from it.

    Returns
    -------
    mean: np.ndarray
        F.mean(axis).
    resid: np.ndarray
        F - mean (broadcast back along axis).

    Notes
    -----
    PP 2026-10-01: fig_disc_profiles.py:35-38 (bottom row, F - <F>_los).
    """
    F = np.asarray(F)
    mean = F.mean(axis=axis)
    return mean, F - np.expand_dims(mean, axis)


def equivalent_width(F, y, lref, ew_jacobian=True):
    """
    Equivalent width [Angstrom] of normalised profiles on the velocity grid (whole grid).

    Parameters
    ----------
    F: array-like
        Normalised profiles, last axis = y.
    y: array-like
        Velocity grid [km/s].
    lref: float or array-like
        Velocity zero point [Angstrom], broadcast against F.shape[:-1].
    ew_jacobian: bool
        Include d lambda / dy = lambda / c (default); False = the pre-2026-09-29 formula.

    Returns
    -------
    np.ndarray or numpy scalar
        F.shape[:-1].
    """
    return line_diagnostics(F, y, lref, keys=("ew",), ew_jacobian=ew_jacobian)["ew"]


def equivalent_width_native(lam, fnorm, axis=-1):
    """
    Equivalent width [Angstrom] on a native (FASTWIND OUT) wavelength grid: trapz(1 - F, lambda).

    Parameters
    ----------
    lam: array-like
        Wavelengths [Angstrom], increasing along axis; 1-D or the shape of fnorm.
    fnorm: array-like
        Normalised flux.
    axis: int
        Wavelength axis.

    Returns
    -------
    np.ndarray or numpy scalar
        Positive for absorption (FASTWIND prints the opposite sign).

    Validation
    ----------
    PP 2026-10-01: as fig_fw_sphere_ew.py:57-62 (both cast to float64, C-ordered, axis = last),
    which wrote RUN/ew.npz; bit-identical with it. Inputs are made C-contiguous, so a Fortran-ordered
    or strided input gives the same numbers.
    """
    lam = np.ascontiguousarray(lam, dtype=np.float64)
    f = np.asarray(fnorm, dtype=np.float64)
    return _trapz(np.ascontiguousarray(1.0 - f), lam, axis=axis)


# ---- broadening kernels on the velocity grid (unit area; Gray 2005) ----
# PP 2026-10-01: ported from fw_disc.py:199-229 (k_rot, k_gauss, k_rt, broaden); identical for dv = 1.

def kernel_rotation(vsini, dv=1.0, eps=0.6):
    """
    Rotation kernel with linear limb darkening (Gray 2005), sampled on the velocity grid.

    G(x) ~ 2 (1 - eps) sqrt(1 - x^2) + (pi / 2) eps (1 - x^2), x = v / (v sin i), on the grid points
    |v| <= v sin i (floor(v sin i / dv) points each side), normalised to unit sum.

    Parameters
    ----------
    vsini: float
        Projected rotation velocity [km/s]; <= 0 gives the identity kernel [1].
    dv: float
        Grid step [km/s].
    eps: float
        Linear limb-darkening coefficient.

    Returns
    -------
    np.ndarray
        Odd-length kernel, sum 1. v sin i < dv gives [1] (no broadening).
    """
    if vsini <= 0:
        return np.array([1.0])
    x = np.arange(-np.floor(vsini / dv), np.floor(vsini / dv) + 1) * dv / vsini
    # the clip only guards against |x| = 1 + rounding for dv != 1 (identity for the legacy grid)
    g = 2 * (1 - eps) * np.sqrt(np.clip(1 - x ** 2, 0.0, None)) + 0.5 * np.pi * eps * (1 - x ** 2)
    return g / g.sum()


def kernel_gauss(vmac, dv=1.0, nsig=4.0):
    """
    Isotropic Gaussian macroturbulence kernel exp(-(v / vmac)^2) on |v| <= ceil(nsig vmac / dv) dv.

    Parameters
    ----------
    vmac: float
        Gaussian macroturbulence [km/s] (1/e half-width; sigma = vmac / sqrt 2); <= 0 gives [1].
    dv: float
        Grid step [km/s].
    nsig: float
        Half-width of the kernel in units of vmac.

    Returns
    -------
    np.ndarray
        Odd-length kernel, sum 1.
    """
    if vmac <= 0:
        return np.array([1.0])
    v = np.arange(-np.ceil(nsig * vmac / dv), np.ceil(nsig * vmac / dv) + 1) * dv
    g = np.exp(-(v / vmac) ** 2)
    return g / g.sum()


def kernel_rt(zeta, dv=1.0, nsig=4.0):
    """
    Radial-tangential macroturbulence kernel with equal radial and tangential parts (Gray 1975),
    disc-integrated without limb darkening: exp(-x^2) - sqrt(pi) x erfc(x), x = |v| / zeta.

    Parameters
    ----------
    zeta: float
        zeta_RT [km/s]; <= 0 gives [1].
    dv: float
        Grid step [km/s].
    nsig: float
        Half-width of the kernel in units of zeta.

    Returns
    -------
    np.ndarray
        Odd-length kernel, sum 1.
    """
    if zeta <= 0:
        return np.array([1.0])
    x = np.abs(np.arange(-np.ceil(nsig * zeta / dv), np.ceil(nsig * zeta / dv) + 1) * dv / zeta)
    g = np.exp(-x ** 2) - np.sqrt(np.pi) * x * (1 - erf(x))
    return g / g.sum()


def broaden(F, *kernels):
    """
    Convolve the absorption depth 1 - F with kernels, one after the other (np.convolve, mode 'same').

    Parameters
    ----------
    F: array-like
        Normalised profile(s), last axis = velocity grid (same step as the kernels).
    *kernels: np.ndarray
        Odd-length, unit-sum kernels (:func:`kernel_rotation`, :func:`kernel_gauss`, :func:`kernel_rt`),
        each shorter than the profile.

    Returns
    -------
    np.ndarray
        1 - (((1 - F) * k1) * k2 ...), float64 for float64 kernels; same shape as F (also for
        zero rows, e.g. F of shape (0, n)).

    Notes
    -----
    Beyond the grid ends the depth is taken as 0 (continuum). ``broaden(broaden(F, k1), k2)`` is not
    bit-identical with ``broaden(F, k1, k2)`` (1 - (1 - d) != d in floating point); the GOF map of
    fig_disc_vmac.py uses the two-call form, and :func:`gof_map` keeps it.
    """
    F = np.asarray(F)
    n = F.shape[-1]
    for k in kernels:
        if np.size(k) > n:
            raise ValueError("kernel ({} points) longer than the profile ({})".format(np.size(k), n))
    if F.ndim == 1:
        d = 1.0 - F
        for k in kernels:
            d = np.convolve(d, k, mode="same")
        return 1.0 - d
    rows = F.reshape(int(np.prod(F.shape[:-1])), n)
    if rows.shape[0] == 0:
        return np.empty(F.shape, dtype=np.result_type(1.0 - rows, *[np.asarray(k) for k in kernels]))
    return np.stack([broaden(r, *kernels) for r in rows]).reshape(F.shape)


# ---- macroturbulence / v sin i analysis of mock observations (fig_disc_vmac.py) ----

def _one_profile(F, y, what="profile", hint="loop over the leading axes"):
    """F as an array; ValueError unless it is one 1-D profile on the grid y."""
    F = np.asarray(F)
    if F.ndim != 1:
        raise ValueError("one {} (1-D) expected, got shape {}; {}".format(what, F.shape, hint))
    if F.size != np.size(y):
        raise ValueError("{} has {} points but the grid y has {}".format(what, F.size, np.size(y)))
    return F


def shift_profile(F, y, shift, at=None):
    """
    Profile moved by +shift [km/s] on the y grid: ``np.interp(at, y + shift, F)`` (constant beyond the ends).

    Parameters
    ----------
    F: array-like
        One profile on y.
    y: array-like
        Velocity grid [km/s].
    shift: float
        Shift [km/s]; > 0 moves the profile to larger y (redwards).
    at: array-like, optional
        Where to evaluate (default y). Values are pointwise, so evaluating on a window gives the same
        numbers as evaluating on y and cutting out the window.

    Notes
    -----
    PP 2026-10-01: fig_disc_vmac.py:41-42 ``shifted(G, dv)``.
    """
    y = np.asarray(y, dtype=np.float64)
    F = _one_profile(F, y)
    return np.interp(y if at is None else at, y + shift, F)


def _kernel_function(kernel, dv):
    if callable(kernel):
        return kernel
    if kernel == "rt":
        return lambda p: kernel_rt(p, dv=dv)
    if kernel == "gauss":
        return lambda p: kernel_gauss(p, dv=dv)
    if kernel == "rot":
        return lambda p: kernel_rotation(p, dv=dv)
    raise ValueError("unknown kernel {!r} (use 'rt', 'gauss', 'rot' or a function of the parameter)".format(kernel))


def fit_broadening(Fobs, Ftemp, y, kernel, grid=None, shifts=None, vwin=500.0, refine=(2.0, 0.1),
                   shift_refine=(0.5, 0.1), par_min=0.0):
    """
    Least-squares fit of one broadening parameter and a velocity shift: the template broadened by
    kernel(par) and shifted by s is compared with the observed profile within |y| < vwin.

    Two stages, as fig_disc_vmac.py: (1) all (par, s) of ``grid`` x ``shifts``; (2) refinement in
    steps (refine, shift_refine): par runs over [max(par_c - h, par_min), par_c + h] around the coarse
    best par_c (range fixed once), and for each refined par the shift runs over [s - hs, s + hs]
    around the running best s, i.e. the shift window re-centres whenever a candidate improves the fit
    (fig_disc_vmac.py evaluates np.arange(dv - 0.5, dv + 0.51, 0.1) inside the par loop with the
    updated dv). The refined shift can therefore drift more than hs from the coarse best. A candidate
    replaces the best only if its rms is strictly smaller (first minimum wins; par is the outer loop).

    Parameters
    ----------
    Fobs, Ftemp: array-like
        Observed and template profiles on y (1-D each; use :func:`macroturbulence_fits` for arrays).
    y: array-like
        Uniform velocity grid [km/s].
    kernel: {'rt', 'gauss', 'rot'} or callable
        'rt' = :func:`kernel_rt` (zeta_RT), 'gauss' = :func:`kernel_gauss` (v_mac), 'rot' =
        :func:`kernel_rotation` (v sin i), on the grid step; or a function par -> kernel array.
    grid: array-like, optional
        Coarse parameter grid [km/s]; default 0, 2, ..., 150.
    shifts: array-like, optional
        Coarse shifts [km/s]; default -15, -14.5, ..., 15.
    vwin: float
        Fit window |y| < vwin [km/s] (strict).
    refine: (float, float) or None
        Half-width and step of the refinement in par; None = no refinement.
    shift_refine: (float, float)
        Half-width and step of the refinement in the shift.
    par_min: float
        Lower bound of the refined parameter.

    Returns
    -------
    par, shift, rms: float
        Best parameter [km/s], shift [km/s] on the y grid (> 0 = redward, about -<v_los>) and rms
        residual; (0.0, 0.0, inf) if every rms is NaN (e.g. NaN in the window or an empty window).

    Raises
    ------
    ValueError
        If Fobs or Ftemp is not one 1-D profile on y.

    Validation
    ----------
    PP 2026-10-01: fig_disc_vmac.py:45-62 ``fit``; the refinement ranges are
    np.arange(max(par - 2, 0), par + 2.01, 0.1) and np.arange(dv - 0.5, dv + 0.51, 0.1) there, here
    stop = par + (h + step / 10): 2.0 + 0.1 / 10 == 2.01 and 0.5 + 0.1 / 10 == 0.51 exactly, so the
    aranges are the same doubles for every starting value. Bit-identical with the legacy fit
    (verified against a verbatim copy: RT and Gaussian fits of all 24 profiles of RUN/disc_los8.npz
    and disc_los8_imu.npz);
    reproduces the strings of figures/fw_disc_vmac_<tag>.csv. Only the window is interpolated
    (pointwise, so identical to interpolating the whole grid and cutting the window).
    """
    y = np.asarray(y, dtype=np.float64)
    Fobs = _one_profile(Fobs, y, "observed profile", "loop or use macroturbulence_fits")
    _one_profile(Ftemp, y, "template", "loop or use macroturbulence_fits")
    kern = _kernel_function(kernel, grid_step(y))
    grid = DEFAULT_PAR_GRID if grid is None else np.asarray(grid, dtype=np.float64)
    shifts = DEFAULT_SHIFTS if shifts is None else np.asarray(shifts, dtype=np.float64)
    w = np.abs(y) < vwin
    yw, Fw = y[w], Fobs[w]

    def rms(B, s):
        return np.sqrt(np.mean((Fw - np.interp(yw, y + s, B)) ** 2))

    best = (np.inf, 0.0, 0.0)
    for par in grid:
        B = broaden(Ftemp, kern(par))
        for s in shifts:
            r = rms(B, s)
            if r < best[0]:
                best = (r, par, s)
    r, par, s = best
    if refine is not None:
        h, st = refine
        hs, sts = shift_refine
        for par2 in np.arange(max(par - h, par_min), par + (h + st / 10), st):
            B = broaden(Ftemp, kern(par2))
            for s2 in np.arange(s - hs, s + (hs + sts / 10), sts):
                r2 = rms(B, s2)
                if r2 < r:
                    r, par, s = r2, par2, s2
    return float(par), float(s), float(r)


def macroturbulence_fits(F, F0, y, grid=None, shifts=None, vwin=500.0, **kwargs):
    """
    Radial-tangential and Gaussian macroturbulence fits of every profile against its template.

    Parameters
    ----------
    F, F0: array-like
        Profiles with velocities and templates without (same shape, last axis = y), e.g. (8, 3, ny).
    y: array-like
        Uniform velocity grid [km/s].
    grid, shifts, vwin, **kwargs:
        As :func:`fit_broadening`.

    Returns
    -------
    np.ndarray
        F.shape[:-1] + (6,): columns :data:`MACRO_COLUMNS` (zeta_RT, dv_RT, rms_RT, vmac_G, dv_G, rms_G);
        dv_* are shifts of the template on the y grid [km/s] (> 0 = redward, about -<v_los> for
        v_los > 0 towards the observer).

    Notes
    -----
    For a non-rotating star the Gaussian v_mac ~ sqrt(2) sigma_los (disc rms line-of-sight velocity):
    M424 dump 3200 gives v_mac 63-66 km/s, zeta_RT 92-104 km/s for sigma_los 44.6 km/s.
    PP 2026-10-01: fig_disc_vmac.py:67-72.
    """
    F, F0 = np.asarray(F), np.asarray(F0)
    if F.shape != F0.shape:
        raise ValueError("F and F0 must have the same shape")
    lead = F.shape[:-1]
    res = np.zeros(lead + (6,))
    for ix in np.ndindex(*lead):
        res[ix + (slice(0, 3),)] = fit_broadening(F[ix], F0[ix], y, "rt", grid, shifts, vwin, **kwargs)
        res[ix + (slice(3, 6),)] = fit_broadening(F[ix], F0[ix], y, "gauss", grid, shifts, vwin, **kwargs)
    return res


def gof_map(Fobs, Ftemp, y, vsini=None, zeta=None, shift=0.0, snr=300.0, vwin=500.0, eps=0.6):
    """
    Goodness-of-fit map chi^2(v sin i, zeta_RT): the template broadened by rotation and then by RT
    macroturbulence, shifted by a fixed velocity, compared with the observed profile.

        chi^2 = sum_{|y| < vwin} ((Fobs - B) snr)^2,  B = shift(broaden(broaden(Ftemp, rot), rt))

    Parameters
    ----------
    Fobs, Ftemp: array-like
        Observed and template profiles on y (1-D each).
    y: array-like
        Uniform velocity grid [km/s].
    vsini, zeta: array-like, optional
        Axes [km/s]; default 0, 3, ..., 150 each.
    shift: float
        Fixed shift [km/s] (fig_disc_vmac.py uses the shift of the zeta_RT-only fit).
    snr: float
        Signal-to-noise ratio per grid point (the contour sizes scale with it).
    vwin: float
        Window |y| < vwin [km/s] (strict).
    eps: float
        Limb-darkening coefficient of the rotation kernel.

    Returns
    -------
    dict
        chi2 (nzeta, nvsini), vsini, zeta, best_vsini, best_zeta (first minimum in C order),
        levels (:data:`GOF_LEVELS`, Delta chi^2 of 1, 2, 3 sigma for two parameters), snr.

    Notes
    -----
    For the M424 mock observations (true v sin i = 0) the best fit sits at v sin i ~ 75-78 km/s with
    zeta_RT ~ 48-57 km/s: the v sin i - zeta degeneracy valley.
    PP 2026-10-01: fig_disc_vmac.py:129-139 (rotation and RT applied in two broaden calls, as there);
    bit-identical with the legacy loop (verified against a verbatim copy for los1 of
    RUN/disc_los8.npz and disc_los8_imu.npz).
    """
    y = np.asarray(y, dtype=np.float64)
    Fobs = _one_profile(Fobs, y, "observed profile")
    _one_profile(Ftemp, y, "template")
    dv = grid_step(y)
    vs = DEFAULT_GOF_GRID if vsini is None else np.asarray(vsini, dtype=np.float64)
    zs = DEFAULT_GOF_GRID if zeta is None else np.asarray(zeta, dtype=np.float64)
    w = np.abs(y) < vwin
    yw, Fw = y[w], Fobs[w]
    xp = y + shift
    chi = np.zeros((zs.size, vs.size))
    for iv, vr in enumerate(vs):
        R = broaden(Ftemp, kernel_rotation(vr, dv=dv, eps=eps))
        for iz, z in enumerate(zs):
            B = np.interp(yw, xp, broaden(R, kernel_rt(z, dv=dv)))
            chi[iz, iv] = np.sum(((Fw - B) * snr) ** 2)
    iz, iv = np.unravel_index(np.argmin(chi), chi.shape)
    return dict(chi2=chi, vsini=vs, zeta=zs, best_vsini=float(vs[iv]), best_zeta=float(zs[iz]),
                levels=GOF_LEVELS, snr=float(snr))


def fourier_amplitude(F, y, vwin=500.0, nfft=1 << 16):
    """
    Normalised Fourier amplitude of the absorption depth within |y| < vwin.

    Parameters
    ----------
    F: array-like
        One profile on y.
    y: array-like
        Uniform velocity grid [km/s].
    vwin: float
        Window |y| < vwin [km/s] (strict); the rest is zero-padded to nfft points.
    nfft: int
        FFT length (default 65536); at least the number of grid points in the window.

    Returns
    -------
    freq: np.ndarray
        Frequencies [cycles per km/s] (rfftfreq with the grid step).
    amp: np.ndarray
        |FT(1 - F)| / |FT(1 - F)|(0).

    Raises
    ------
    ValueError
        If F is not one 1-D profile on y, or the window holds more than nfft points (np.fft.rfft
        would silently crop it).

    Notes
    -----
    PP 2026-10-01: fig_disc_vmac.py:154-158. Under numpy >= 2 the amplitudes differ from numpy 1.26
    by ~3e-16 (absolute, normalised amplitude; np.fft rewrite); see the module Notes.
    """
    y = np.asarray(y, dtype=np.float64)
    F = _one_profile(F, y)
    w = np.abs(y) < vwin
    if w.sum() > nfft:
        raise ValueError("the window |y| < {} holds {} points, more than nfft = {}".format(vwin, int(w.sum()), nfft))
    dd = 1.0 - F
    amp = np.abs(np.fft.rfft(dd[w], n=nfft))
    freq = np.fft.rfftfreq(nfft, d=grid_step(y))
    amp /= amp[0]
    return freq, amp


def fourier_first_zero(F, y, vwin=500.0, nfft=1 << 16, q1=FT_Q1_ROT):
    """
    First minimum of the Fourier amplitude of the absorption depth and the v sin i it implies if the
    profile were rotationally broadened: v sin i = q1 / sigma_1 (Gray 2005; q1 = 0.660 for eps = 0.6).

    Parameters
    ----------
    F: array-like
        One profile on y.
    y: array-like
        Uniform velocity grid [km/s].
    vwin, nfft:
        As :func:`fourier_amplitude`.
    q1: float
        sigma_1 v sin i of the first zero.

    Returns
    -------
    dict
        freq1 (sigma_1 [cycles per km/s], NaN if there is no local minimum), vsini [km/s], freq, amp,
        minima (indices of all strict local minima).

    Raises
    ------
    ValueError
        As :func:`fourier_amplitude` (one 1-D profile; window <= nfft points).

    Notes
    -----
    Strict local minima of the sampled amplitude (amp[i] < both neighbours); no interpolation, so
    sigma_1 is quantised to 1 / (nfft dv). For the M424 dump-3200 profiles (no rotation) the first
    minimum gives "v sin i" = 111 / 177 / 37 km/s for lambda4026 / 4200 / 4922 (the intrinsic Stark
    profiles dominate). PP 2026-10-01: fig_disc_vmac.py:159-170.
    """
    freq, amp = fourier_amplitude(F, y, vwin=vwin, nfft=nfft)
    mins = np.where((amp[1:-1] < amp[:-2]) & (amp[1:-1] < amp[2:]))[0] + 1
    f1 = float(freq[mins[0]]) if mins.size else np.nan
    vsini = q1 / f1 if mins.size else np.nan
    return dict(freq1=f1, vsini=vsini, freq=freq, amp=amp, minima=mins)
