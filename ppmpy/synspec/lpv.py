"""
Line-profile variability: residual spectra, colour limits, the central zero-crossing tracker,
gap filling, lag correlations (coherence time) and the comparison of two time series of spectra.

Time series of disc-integrated profiles are arrays F with time along one axis (axis 0 by
default) and the velocity grid along the last axis, e.g. the M424 products
``<name>_timeseries.npz`` with F of shape (dumps, lines of sight, lines, velocity) =
(1601, 8, 3, 5401), float32. Residual spectra are R = F - <F>_t.

Conventions
-----------
* All statistics are computed in float64. Inputs are converted as the legacy scripts did, with
  ``.astype(np.float64)``, which keeps the memory order of the input (a fancy index on the last
  axis, e.g. ``F[rows, k, j][:, mask]``, gives a non-C-ordered array). Time means of float32
  profiles are exact (the sums need fewer than 53 bits) and so independent of the summation order;
  std and sums of products of the float64 residuals are not (~1e-15 relative). Pass arrays made by
  the same indexing as a legacy script to reproduce its numbers bit for bit.
* Functions accept numpy memory maps (:func:`open_timeseries`, :func:`ppmpy.synspec.io.npz_member_memmap`)
  and read only the slices they need.

PP 2026-10-01: ported from the project scripts fig_disc_dynspec.py, fig_disc_zerocross.py,
fig_disc_zerocross_spectrum.py, fw_disc_systematics.py and the coherence check behind the
project-log entry of 2026-09-29 (scratch acf.py); see the provenance comments per function.
"""
import numpy as np

from .spectral import lam_of_y


# ----------------------------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------------------------
def open_timeseries(path, mmap=("F", "F0")):
    """
    Open a time-series product (e.g. M424 ``imu_timeseries.npz``) without reading its big arrays.

    Parameters
    ----------
    path: str
        Uncompressed .npz file.
    mmap: str or sequence of str
        Members returned as read-only memory maps (:func:`ppmpy.synspec.io.npz_member_memmap`);
        all other members are read into memory. A single name may be given as a string.

    Returns
    -------
    dict
        member name -> array (memory map for the members in ``mmap``).
    """
    from .io import npz_member_memmap
    # PP 2026-10-01 (review): a string is one name ('F0' must not also map 'F' by a substring test)
    if mmap is None:
        mmap = ()
    elif isinstance(mmap, str):
        mmap = (mmap,)
    mmap = set(mmap)
    out = {}
    with np.load(path) as z:
        for k in z.files:
            out[k] = npz_member_memmap(path, k) if k in mmap else z[k]
    return out


def _f64(a):
    """float64 version of a, memory order kept (the legacy ``.astype(np.float64)``; no copy if already float64)."""
    return np.asarray(a).astype(np.float64, copy=False)


def _read(a):
    """In-memory array: memory maps are read once (a C-ordered copy), other arrays are used as they are.
    The copy does not change any result: a fancy index on the last axis gives the same layout for a view
    and for a copy."""
    return np.array(a) if isinstance(a, np.memmap) else np.asarray(a)


def _index(ndim, axis, i):
    """Tuple index selecting position i of ``axis`` (a view, also of memory maps)."""
    idx = [slice(None)] * ndim
    idx[axis] = i
    return tuple(idx)


# ----------------------------------------------------------------------------------------------
# residual spectra and colour scales
# ----------------------------------------------------------------------------------------------
def residual_spectra(F, axis=0, ref="mean", ref_rows=None):
    """
    Residual spectra R = F - F_ref, F_ref = the time mean of F (or a given array).

    Parameters
    ----------
    F: array-like
        Spectra, time along ``axis`` (e.g. (dumps, velocity) or (dumps, los, lines, velocity)).
    axis: int
        Time axis.
    ref: 'mean' or array-like
        'mean': F_ref is the mean over time, of all rows or of ``ref_rows`` only. An array:
        subtracted as it is. It is either broadcastable to the shape of F without the time axis
        (e.g. one profile (ny,) for all lines of sight), or has F's number of dimensions with
        length 1 on the time axis; it is broadcast to the shape of F without the time axis
        before the time axis is re-inserted, so a 1-D profile always runs along the last axis.
    ref_rows: int array or bool array, optional
        Rows (along ``axis``) that define the mean. R is still returned for all rows of F.
        With the default (mean of the rows passed) pass only the window you show.

    Returns
    -------
    R: np.ndarray
        float64, the shape of F.
    Fref: np.ndarray
        float64, the shape of F without the time axis.

    Notes
    -----
    Kalita et al. (2025) subtract the mean of ALL spectra while showing a window
    (fig_disc_dynspec.py --mean-all). To reproduce that figure bit for bit, take the mean of the
    whole series and apply it to the window::

        Fref_all = residual_spectra(F[:, k, j][:, m])[1]            # mean of all dumps
        R_win, _ = residual_spectra(F[isel, k, j][:, m], ref=Fref_all)

    (the indexing of the script, so that R_win has its memory layout).
    ``residual_spectra(F[:, k, j][:, m])[0][isel]`` (or ``ref_rows``) gives the same R values, but as a
    C-ordered slice, so statistics over time of it (e.g. the rms curve of
    :func:`residual_summary`) differ from the legacy ones by ~1e-15 relative (summation order).

    Validation
    ----------
    Bit-identical to fig_disc_dynspec.py (``R - R.mean(axis=0)`` on ``F.astype(np.float64)``, and
    ``R - MREF`` for --mean-all with the recipe above), including the rms curves.
    """
    # PP 2026-10-01: ported from fig_disc_dynspec.py:71-72 and :121-122 (--mean-all)
    X = _f64(F)
    axis = axis % X.ndim
    tshape = X.shape[:axis] + X.shape[axis + 1:]
    if isinstance(ref, str):
        if ref != "mean":
            raise ValueError("ref must be 'mean' or an array, got {!r}".format(ref))
        if ref_rows is None:
            S = X
        else:
            rows = np.asarray(ref_rows)
            if rows.dtype == bool:
                if rows.shape != (X.shape[axis],):
                    raise ValueError("a boolean ref_rows needs one entry per row")
                rows = np.nonzero(rows)[0]
            S = np.take(X, rows, axis=axis)
        if S.shape[axis] == 0:
            raise ValueError("ref_rows selects no rows")
        Fref = S.mean(axis=axis)
    else:
        # PP 2026-10-01 (review): broadcast to the shape without the time axis BEFORE re-inserting the
        # time axis (np.expand_dims of a lower-dimensional ref put the time axis in the wrong place)
        Fref = np.asarray(ref, dtype=np.float64)
        if Fref.ndim > X.ndim:
            raise ValueError("ref has more dimensions ({}) than F ({})".format(Fref.ndim, X.ndim))
        if Fref.ndim == X.ndim:
            if Fref.shape[axis] != 1:
                raise ValueError("a reference with the time axis must have length 1 there")
            Fref = np.squeeze(Fref, axis=axis)
        if Fref.shape != tshape:
            try:
                Fref = np.array(np.broadcast_to(Fref, tshape))
            except ValueError:
                raise ValueError("ref of shape {} does not broadcast to F without the time axis {}"
                                 .format(np.shape(ref), tshape)) from None
    R = X - np.expand_dims(Fref, axis)
    return R, Fref


def residual_summary(R, axis=0, pct=99.5):
    """
    Amplitude summary of residual spectra.

    Parameters
    ----------
    R: array-like
        Residuals, time along ``axis``.
    axis: int
        Time axis (``rms_t`` is the std along it).
    pct: float
        Percentile of |R| (the dynamic-spectrum colour scale uses 99.5).

    Returns
    -------
    dict
        rms_t: std over time per pixel (the "rms of F - <F>_t" curve of the dynamic-spectrum figures);
        rms: std of all values; pct_abs: the ``pct`` percentile of |R|; max_abs: max |R|.
    """
    # PP 2026-10-01: ported from fig_disc_dynspec.py:142 (rms curve) and :161 (printed summary)
    R = np.asarray(R, dtype=np.float64)
    aR = np.abs(R)
    return dict(rms_t=R.std(axis=axis), rms=R.std(), pct_abs=np.percentile(aR, pct), max_abs=aR.max())


def symmetric_limit(arrays, pct=99.5):
    """
    Symmetric colour limit for residual dynamic spectra: the largest, over the arrays, of the
    ``pct`` percentile of |a| (one common scale for several lines; set by the strongest LPV).

    Parameters
    ----------
    arrays: array-like or list/tuple of array-like
        One array, or a list/tuple of arrays (e.g. the residuals of each line). A list or tuple
        whose first element has ndim >= 1 is a sequence of arrays; anything else (an ndarray, a
        list of numbers) is one array. A nested list of numbers therefore counts as several
        arrays (one per row); pass ``np.asarray`` of it for one array.
    pct: float
        Percentile.

    Returns
    -------
    float
        In data units (the figures show it x 1e3).
    """
    # PP 2026-10-01: ported from fig_disc_dynspec.py:73/:125 and fig_disc_dynspec_slide.py:37 (LIM, without the 1e3)
    # PP 2026-10-01 (review): a plain list of numbers is one array (was taken as a list of scalars -> max|x|)
    if isinstance(arrays, (list, tuple)) and len(arrays) == 0:
        raise ValueError("no arrays given")
    if not isinstance(arrays, (list, tuple)) or np.ndim(arrays[0]) == 0:
        arrays = [np.asarray(arrays)]
    return max(np.percentile(np.abs(a), pct) for a in arrays)


# ----------------------------------------------------------------------------------------------
# central zero crossing of the residual spectra
# ----------------------------------------------------------------------------------------------
def zero_crossing_track(y, F, lref=None, halfwidth=150.0, core=400.0):
    """
    Central zero crossing of the residual spectra R(t) = F(t) - <F>_t, one per row: the "white
    space" running through the line core of a residual dynamic spectrum.

    Algorithm (exactly fig_disc_zerocross.py): <F>_t is the mean over all rows; its minimum is
    searched within |y| <= ``core`` (``ymin``); in the window |y - ymin| <= ``halfwidth`` every
    sign change of R between neighbouring pixels c, c+1 (``sign(R_c) sign(R_c+1) < 0``) is located
    by linear interpolation, y_c - R_c (y_c+1 - y_c) / (R_c+1 - R_c), and the one closest to ymin
    is kept (the first on ties). Rows without a sign change give NaN.

    Parameters
    ----------
    y: array-like
        (ny,) velocity grid y = c ln(lambda / lref) [km/s]: redshift positive, i.e. about -v_los in
        the sign convention of :mod:`ppmpy.synspec.conventions` (v > 0 towards the observer).
    F: array-like
        (nt, ny) spectra (one line of sight, one line); converted to float64.
    lref: float, optional
        Zero point of the velocity grid [Angstrom]; if given, the crossings are also returned as
        wavelengths lref exp(y/c) (the 'A' of the M424 zerocross_<name>_los<k>.npz products).
    halfwidth: float
        Half-width of the search window around ymin [km/s] (legacy --vwin, 150).
    core: float
        The mean-profile minimum is searched within |y| <= core [km/s] (400).

    Returns
    -------
    dict
        y: (nt,) the crossing on the velocity grid y = c ln(lambda / lref) [km/s] (redshift
        positive, about -v_los), NaN where none; lam: (nt,) the same as wavelength
        [Angstrom] (only if lref is given); ymin: grid coordinate y of the mean-profile minimum;
        n_cross: (nt,) number of sign changes in the window; n_found: rows with a crossing;
        Fmean: (ny,) the time-mean profile; halfwidth, core: the parameters.

    Conventions
    -----------
    A pixel where R is exactly 0 has sign 0 and does not form a sign change with either
    neighbour (legacy behaviour, kept): a crossing that falls exactly on a pixel is missed.
    For float32 profiles this is very rare; it never occurred for M424.

    Validation
    ----------
    Reproduces the 'A' arrays of the M424 zerocross_imu_los{1..8}.npz bit for bit, NaNs included
    (tests/synspec/test_lpv.py). The loop over rows is vectorised; the arithmetic per crossing is
    the legacy one.
    """
    # PP 2026-10-01: ported from fig_disc_zerocross.py:32-46 (vectorised over dumps)
    Y = np.asarray(y, dtype=np.float64)
    F = _f64(F)
    if F.ndim != 2 or F.shape[1] != Y.size:
        raise ValueError("F must be (nt, ny) with ny = len(y); got {} for ny = {}".format(F.shape, Y.size))
    nt = F.shape[0]
    Fm = F.mean(axis=0)
    cmask = np.abs(Y) <= core
    if not cmask.any():
        raise ValueError("no grid point within |y| <= core = {}".format(core))
    ymin = Y[cmask][np.argmin(Fm[cmask])]
    m = np.abs(Y - ymin) <= halfwidth
    yw, R = Y[m], F[:, m] - Fm[m]
    out = dict(y=np.full(nt, np.nan), ymin=ymin, n_cross=np.zeros(nt, dtype=np.int64), Fmean=Fm,
               halfwidth=float(halfwidth), core=float(core))
    if yw.size >= 2:
        s = np.sign(R)
        cross = s[:, :-1] * s[:, 1:] < 0                       # sign changes between pixel c and c+1
        out["n_cross"] = cross.sum(axis=1)
        ii, cc = np.nonzero(cross)
        yc = yw[cc] - R[ii, cc] * (yw[cc + 1] - yw[cc]) / (R[ii, cc + 1] - R[ii, cc])
        D = np.full(cross.shape, np.inf)
        D[ii, cc] = np.abs(yc - ymin)
        YC = np.zeros(cross.shape)
        YC[ii, cc] = yc
        rows = np.nonzero(cross.any(axis=1))[0]
        best = np.argmin(D[rows], axis=1)                       # first minimum = legacy argmin over ascending c
        out["y"][rows] = YC[rows, best]
    out["n_found"] = int(np.isfinite(out["y"]).sum())
    if lref is not None:
        out["lam"] = lam_of_y(out["y"], lref)
    return out


# ----------------------------------------------------------------------------------------------
# gaps, lag correlation
# ----------------------------------------------------------------------------------------------
def fill_gaps(t, x):
    """
    Fill non-finite samples of a time series by linear interpolation in time.

    Parameters
    ----------
    t: array-like
        (nt,) increasing times.
    x: array-like
        (nt,) or (nt, ...) series, time along axis 0; every column is filled separately.

    Returns
    -------
    x_filled: np.ndarray
        float64 C-ordered copy of x (any input layout). Gaps before the first / after the last
        finite sample take that sample's value (np.interp); columns without any finite sample stay NaN.
    mask: np.ndarray of bool
        True where x was not finite (the filled samples, and the NaNs of all-NaN columns).

    Validation
    ----------
    Identical to series() of fig_disc_zerocross_spectrum.py (before its A/<A> - 1), which used
    t = t_s - t_s[0]; pass the same t to reproduce it bit for bit.
    """
    # PP 2026-10-01: ported from fig_disc_zerocross_spectrum.py:53-55
    # PP 2026-10-01 (review): order='C' so that the reshape below is a view of xf (a non-C-ordered input with
    # >= 3 dimensions made it a copy, and the filled values never reached xf)
    t = np.asarray(t, dtype=np.float64)
    xf = np.array(x, dtype=np.float64, order="C")
    if xf.shape[0] != t.size:
        raise ValueError("x must have len(t) = {} rows, got {}".format(t.size, xf.shape[0]))
    bad = ~np.isfinite(xf)
    cols = xf.reshape(t.size, -1)                             # a view of the C-ordered xf
    bcols = bad.reshape(t.size, -1)
    for c in range(cols.shape[1]):
        b = bcols[:, c]
        if b.any() and not b.all():
            cols[b, c] = np.interp(t[b], t[~b], cols[~b, c])
    return xf, bad


def lag_correlation(R, lags, axis=0, keep_axes=(), other=None, demean=False):
    """
    Correlation of residual spectra with themselves (or with ``other``) at time lags: the
    coherence-time analysis.

    For a lag L (rows),
        c(L) = sum R(t) S(t+L) / sqrt( sum R(t)^2  sum S(t+L)^2 ),
    the sums running over the overlapping rows and over all axes except ``axis`` and
    ``keep_axes`` (S = ``other`` or R). Negative lags correlate R(t) with S(t - |L|).

    Parameters
    ----------
    R: array-like
        Residuals (time mean removed), time along ``axis``, e.g. (dumps, los, velocity).
    lags: int or sequence of int
        Lags in rows (dumps), whole numbers (integer-valued floats are accepted, anything else
        raises ValueError). The result is per row lag; multiply the lags by the cadence for time.
    axis: int
        Time axis.
    keep_axes: int or sequence of int
        Axes NOT summed over (one correlation per index, e.g. per line of sight).
    other: array-like, optional
        Second series of the same shape (cross-correlation, e.g. opposite lines of sight at lag 0).
    demean: bool or {'pooled', 'pixel'}
        Which mean is removed from the two overlapping segments (R(t) and S(t+L)) before the sums:

        * False (default): none; normalised inner product of the residuals as they are (they have
          zero time mean already; the M424 convention).
        * True or 'pooled': each segment minus its single overall mean, taken over all its rows
          and all pooled pixels (and lines of sight) together, i.e. np.corrcoef of the flattened
          segments. Per-pixel offsets are NOT removed and can dominate the result.
        * 'pixel': each segment minus its own time mean per pooled pixel (per element of the
          summed axes), i.e. the Pearson correlation with per-pixel segment means.

    Returns
    -------
    np.ndarray
        float64, shape (len(lags),) + the shapes of ``keep_axes`` (a scalar lag gives no lag axis);
        NaN where |L| >= nt.

    Validation
    ----------
    With keep_axes=(1,) on the (dumps, los, |v| <= 600 km/s) residuals of M424 imu, the mean over
    the 8 lines of sight reproduces the project-log values for lags 1/2/3/5/10 dumps
    (0.77/0.55/0.37/0.14/0.04 for lambda4026 ...; coherent over ~2-3 dumps, ~2 h of M424 time),
    bit-identical to the original per-LOS formula.
    """
    # PP 2026-10-01: ported from the scratch script acf.py behind project_log.txt (2026-09-29 results)
    R = np.asarray(R)
    S = R if other is None else np.asarray(other)
    if S.shape != R.shape:
        raise ValueError("other must have the shape of R: {} vs {}".format(S.shape, R.shape))
    nd = R.ndim
    axis = axis % nd
    keep = [k % nd for k in np.atleast_1d(keep_axes).astype(int)] if np.size(keep_axes) else []
    if axis in keep:
        raise ValueError("the time axis cannot be kept")
    if isinstance(demean, (bool, np.bool_)):
        demean = "pooled" if demean else False
    if demean is not False and demean not in ("pooled", "pixel"):
        raise ValueError("demean must be False, True, 'pooled' or 'pixel', got {!r}".format(demean))
    scalar = np.ndim(lags) == 0
    # PP 2026-10-01 (review): lags are rows; a fractional lag (e.g. a time) was silently truncated
    lag_in = np.atleast_1d(np.asarray(lags))
    if lag_in.dtype.kind not in "iu":
        if lag_in.dtype.kind != "f" or not np.all(np.isfinite(lag_in)) or np.any(lag_in != np.rint(lag_in)):
            raise ValueError("lags must be whole numbers of rows, got {!r}".format(lags))
    lags = lag_in.astype(np.int64)
    order = [axis] + keep + [k for k in range(nd) if k != axis and k not in keep]
    Rt, St = np.transpose(R, order), np.transpose(S, order)
    nt = Rt.shape[0]
    kshape = Rt.shape[1:1 + len(keep)]
    nk = int(np.prod(kshape)) if keep else 1
    Rt = Rt.reshape((nt, nk, -1))
    St = Rt if other is None else St.reshape((nt, nk, -1))
    out = np.full((lags.size, nk), np.nan)
    for il, L in enumerate(lags):
        if abs(L) >= nt:
            continue
        for k in range(nk):
            a, b = Rt[:, k], St[:, k]
            if L >= 0:
                a, b = a[:nt - L], b[L:]
            else:
                a, b = a[-L:], b[:nt + L]
            a = np.asarray(a, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)
            if demean == "pooled":
                a = a - a.mean()
                b = b - b.mean()
            elif demean == "pixel":
                a = a - a.mean(axis=0)
                b = b - b.mean(axis=0)
            out[il, k] = np.sum(a * b) / np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    out = out.reshape((lags.size,) + tuple(kshape))
    return out[0] if scalar else out


def coherence_time(lags, corr, level=np.exp(-1.0)):
    """
    First lag at which a correlation curve falls below ``level``, by linear interpolation
    between the neighbouring lags.

    Parameters
    ----------
    lags: array-like
        (n,) increasing lags (rows or time).
    corr: array-like
        (n,) or (n, ...) correlations (lag along axis 0), e.g. from :func:`lag_correlation`.
    level: float
        Threshold (default 1/e).

    Returns
    -------
    float or np.ndarray
        In the units of ``lags``; NaN where the curve never falls below ``level`` (or starts below it).
    """
    # PP 2026-10-01: new (summarises lag_correlation)
    lags = np.asarray(lags, dtype=np.float64)
    c = np.asarray(corr, dtype=np.float64)
    c2 = c.reshape(lags.size, -1)
    out = np.full(c2.shape[1], np.nan)
    for k in range(c2.shape[1]):
        below = np.nonzero(c2[:, k] < level)[0]
        if below.size == 0 or below[0] == 0:
            continue
        i = below[0]
        c0, c1 = c2[i - 1, k], c2[i, k]
        out[k] = lags[i - 1] + (c0 - level) / (c0 - c1) * (lags[i] - lags[i - 1])
    return out[0] if c.ndim == 1 else out.reshape(c.shape[1:])


# ----------------------------------------------------------------------------------------------
# comparison of two runs (systematics)
# ----------------------------------------------------------------------------------------------
def _resid(A, mask):
    """Legacy fw_disc_systematics.resid(): window, float64, minus the time mean (time on axis 0)."""
    A = np.asarray(A)
    if mask is not None:
        A = A[..., mask]
    A = A.astype(np.float64)
    return A - A.mean(axis=0, keepdims=True)


def residual_stats(Rref, Rrun):
    """
    Statistics of the difference of two residual arrays (pooled over all elements).

    Parameters
    ----------
    Rref, Rrun: np.ndarray
        Residuals (time mean removed) of the reference and of the run, same shape.

    Returns
    -------
    dict
        sig_rms, sig_max: std and max |.| of Rref (the signal); d_rms, d_max: of Rrun - Rref;
        ratio = d_rms / sig_rms; amp = sum(Rrun Rref) / sum(Rref^2) (best-fit amplitude ratio);
        corr: Pearson correlation of Rrun and Rref.
    """
    # PP 2026-10-01: ported from fw_disc_systematics.py:40-43 (stats)
    d = Rrun - Rref
    return dict(sig_rms=Rref.std(), sig_max=np.abs(Rref).max(), d_rms=d.std(), d_max=np.abs(d).max(),
                ratio=d.std() / Rref.std(), amp=np.sum(Rrun * Rref) / np.sum(Rref * Rref),
                corr=np.corrcoef(Rrun.ravel(), Rref.ravel())[0, 1])


def compare_timeseries(ref, run, axis=0, line_axis=-2, mask=None, static_full_grid=True):
    """
    Compare the TIME-VARIABLE parts of two time series of spectra (e.g. two methods or library
    variants), per line, pooling all other axes (lines of sight, pixels, time).

    For each line: R = F - <F>_t within ``mask`` for both arrays; signal = std and max |R_ref|;
    difference = R_run - R_ref (std, max |.|, std ratio to the signal); best-fit amplitude ratio
    and correlation (:func:`residual_stats`); plus the static difference max |<F_run>_t - <F_ref>_t|.

    Parameters
    ----------
    ref, run: array-like
        Spectra of the same shape, time along ``axis``, velocity along the last axis, e.g.
        (dumps, los, lines, velocity). Memory maps are read one line at a time.
    axis: int
        Time axis.
    line_axis: int or None
        Axis of the lines (one set of statistics per index); None pools everything (scalars).
    mask: bool array, optional
        Window on the last (velocity) axis, e.g. ``np.abs(y) <= 600``.
    static_full_grid: bool
        True (legacy default): the static difference is taken over the whole velocity grid,
        not within ``mask`` (fw_disc_systematics.py quirk). False: within ``mask``.

    Returns
    -------
    dict
        sig_rms, sig_max, d_rms, d_max, ratio, amp, corr, static: (nlines,) float64 arrays
        (floats if line_axis is None).

    Validation
    ----------
    With the M424 time series (mask |y| <= 600 km/s, F and F0) this reproduces the corresponding
    entries of systematics.npz bit for bit (see :func:`systematics`).
    """
    # PP 2026-10-01: ported from fw_disc_systematics.py:35-37, :55-57 as a function of two arrays
    # PP 2026-10-01 (review): the per-line loop is shared with systematics (_compare_lines)
    out = _compare_lines(ref, [run], axis, line_axis, mask, static_full_grid)[0]
    if line_axis is None:
        return {k: float(v[0]) for k, v in out.items()}
    return out


_CMP_KEYS = ("sig_rms", "sig_max", "d_rms", "d_max", "ratio", "amp", "corr", "static")


def _compare_lines(ref, runs, axis, line_axis, mask, static_full_grid):
    """
    Per-line statistics of several runs against one reference (:func:`compare_timeseries`,
    :func:`systematics`). Each line of the reference is read, and its residuals and time mean
    computed, once; the runs are then read one line at a time.

    Parameters
    ----------
    ref: array-like
        Reference spectra, time along ``axis``, velocity along the last axis.
    runs: sequence of array-like
        Spectra of the same shape as ``ref``.
    axis, line_axis, mask, static_full_grid:
        As in :func:`compare_timeseries`.

    Returns
    -------
    list of dict
        One dict per run: statistic -> (nlines,) float64 array ((1,) if line_axis is None).
    """
    # PP 2026-10-01 (review): factored out of compare_timeseries; the operations are those of the
    # previous compare_timeseries and systematics, so the results are unchanged (bit for bit)
    shape = np.shape(ref)
    for B in runs:
        if np.shape(B) != shape:
            raise ValueError("ref and run must have the same shape: {} vs {}".format(shape, np.shape(B)))
    nd = len(shape)
    axis = axis % nd
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
    smask = None if static_full_grid else mask
    if line_axis is None:
        groups, la = [None], None
    else:
        la = line_axis % nd
        if la == axis or la == nd - 1:
            raise ValueError("line_axis must differ from the time axis and from the last (velocity) axis")
        groups = range(shape[la])

    def block(X, j):
        """Line j of X in memory (memory maps are read here), time on axis 0."""
        ta = axis
        if j is not None:
            X = X[_index(nd, la, j)]
            ta = axis if axis < la else axis - 1
        return np.moveaxis(_read(X), ta, 0)

    def static_mean(X):
        """Time mean for the static difference: whole velocity grid, or within the mask."""
        Xs = X if smask is None else X[..., smask]
        return Xs.astype(np.float64).mean(0)

    outs = [{k: np.zeros(len(groups)) for k in _CMP_KEYS} for _ in runs]
    for g, j in enumerate(groups):
        A = block(ref, j)
        Rr, mr = _resid(A, mask), static_mean(A)
        del A
        for out, run in zip(outs, runs):
            B = block(run, j)
            st = residual_stats(Rr, _resid(B, mask))
            st["static"] = np.abs(static_mean(B) - mr).max()
            for k in _CMP_KEYS:
                out[k][g] = st[k]
    return outs


def compare_series(ref, run, axis=0, line_axis=-1):
    """
    Compare two scalar time series (EW, centroid, width ...), each minus its time mean, per line,
    pooling the other axes (lines of sight).

    Parameters
    ----------
    ref, run: array-like
        Same shape, time along ``axis``, e.g. (dumps, los, lines).
    axis: int
        Time axis.
    line_axis: int or None
        One set of statistics per index of this axis; None pools everything.

    Returns
    -------
    dict
        sig_rms (std of the reference), d_rms (std of the difference), ratio = d_rms / sig_rms,
        corr (Pearson): (nlines,) arrays, or floats if line_axis is None.
    """
    # PP 2026-10-01: ported from fw_disc_systematics.py:61-67
    ref, run = np.asarray(ref), np.asarray(run)
    if ref.shape != run.shape:
        raise ValueError("ref and run must have the same shape: {} vs {}".format(ref.shape, run.shape))
    nd = ref.ndim
    axis = axis % nd
    keys = ("sig_rms", "d_rms", "ratio", "corr")
    groups = [None] if line_axis is None else range(ref.shape[line_axis % nd])
    out = {k: np.zeros(len(groups)) for k in keys}
    for g, j in enumerate(groups):
        a, b, ta = ref, run, axis
        if j is not None:
            la = line_axis % nd
            if la == axis:
                raise ValueError("line_axis must differ from the time axis")
            a, b = a[_index(nd, la, j)], b[_index(nd, la, j)]
            ta = axis if axis < la else axis - 1
        xr = a - a.mean(axis=ta, keepdims=True)
        xo = b - b.mean(axis=ta, keepdims=True)
        for k, v in (("sig_rms", xr.std()), ("d_rms", (xo - xr).std()), ("ratio", (xo - xr).std() / xr.std()),
                     ("corr", np.corrcoef(xo.ravel(), xr.ravel())[0, 1])):
            out[k][g] = v
    if line_axis is None:
        return {k: float(v[0]) for k, v in out.items()}
    return out


def systematics(ref, runs, ref_name="flux", y=None, vwin=600.0, quantities=("F", "F0"), diag="diag_F",
                diag_names=("ew", "v1", "sigma"), lines=None):
    """
    The systematics table of the M424 all-dump products (systematics.npz): every run compared
    with the reference for the residual spectra with velocities (F), with T_eff' only (F0), and the
    EW, centroid and width time series.

    Parameters
    ----------
    ref: mapping
        The reference time series (e.g. :func:`open_timeseries` of flux_timeseries.npz): F, F0
        (dumps, los, lines, velocity), diag_F (dumps, los, lines, keys), dumps, Y, and optionally
        diag_keys and LREF.
    runs: mapping
        run name -> mapping like ``ref`` (e.g. 'imu', 'flux_lamfix', 'flux_sm335').
    ref_name: str
        Name stored as 'ref'.
    y: array-like, optional
        Velocity grid; default ref['Y'].
    vwin: float
        Residuals within |y| <= vwin [km/s] (600).
    quantities: sequence of str
        Spectra members compared per line exactly as :func:`compare_timeseries` does (the same code,
        with the static difference over the whole velocity grid, as in the legacy script).
    diag: str or None
        Member with the per-dump diagnostics, compared with :func:`compare_series`.
    diag_names: sequence of str
        Diagnostics to compare, located by name in each mapping's own 'diag_keys' (so runs may
        store them in a different order). A reference without 'diag_keys' is taken to hold
        ``diag_names`` as its first entries (legacy indices 0, 1, 2 = ew, v1, sigma); a run without
        'diag_keys' is taken to have the reference's layout. A missing name raises ValueError.
    lines: sequence of str, optional
        Line names stored as 'lines'; default ref['lines'] if present, else 'line0', 'line1', ...
        (pass the names for M424, e.g. ('HEI4026', 'HEII4200', 'HEI4922')).

    Returns
    -------
    dict
        'ref', 'lines', 'vwin' and '<run>_<q>_<stat>' -> (nlines,) arrays, the keys and values of
        systematics.npz (save with :func:`ppmpy.synspec.io.save_npz`).

    Validation
    ----------
    Reproduces /scratch/.../disc_dumps_r4050_N1236544/systematics.npz bit for bit from the four
    M424 time series (tests/synspec/test_lpv.py).
    """
    # PP 2026-10-01: ported from fw_disc_systematics.py:30-79; the reference residuals are read once per
    # quantity and line instead of once per run.
    # PP 2026-10-01 (review): spectra via the helper shared with compare_timeseries; diagnostics located by
    # name in each mapping's diag_keys; no M424 line names by default.
    Y = np.asarray(ref["Y"] if y is None else y, dtype=np.float64)
    mask = np.abs(Y) <= vwin
    nline = np.shape(ref[quantities[0]])[2] if quantities else np.shape(ref[diag])[2]
    if lines is None:
        lines = [str(n) for n in ref["lines"]] if "lines" in ref else ["line{}".format(j) for j in range(nline)]
    if len(lines) != nline:
        raise ValueError("{} line names for {} lines".format(len(lines), nline))
    out = {"ref": np.array(ref_name), "lines": np.array(list(lines)), "vwin": vwin}
    names = list(runs)
    for name in names:
        run = runs[name]
        if "dumps" in ref and "dumps" in run and not np.array_equal(np.asarray(run["dumps"]), np.asarray(ref["dumps"])):
            raise ValueError("{}: different dumps than the reference".format(name))
    for q in quantities:
        res = _compare_lines(ref[q], [runs[name][q] for name in names], axis=0, line_axis=2, mask=mask,
                             static_full_grid=True)
        for name, st in zip(names, res):
            for k in _CMP_KEYS:
                out["{}_{}_{}".format(name, q, k)] = st[k]
    if diag is not None:
        Dr = np.asarray(ref[diag])
        rkeys = [str(k) for k in ref["diag_keys"]] if "diag_keys" in ref else list(diag_names)

        def position(keys, key, who):
            if key not in keys:
                raise ValueError("diagnostic {!r} not in the diag_keys of {}: {}".format(key, who, keys))
            return keys.index(key)

        for name in names:
            run = runs[name]
            Do = np.asarray(run[diag])
            okeys = [str(k) for k in run["diag_keys"]] if "diag_keys" in run else rkeys
            for key in diag_names:
                ir, io = position(rkeys, key, "the reference"), position(okeys, key, name)
                for j in range(nline):
                    st = compare_series(Dr[:, :, j, ir], Do[:, :, j, io], axis=0, line_axis=None)
                    for k in ("sig_rms", "d_rms", "ratio", "corr"):
                        out.setdefault("{}_{}_{}".format(name, key, k), np.zeros(nline))[j] = st[k]
    return out
