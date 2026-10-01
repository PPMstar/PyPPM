"""
Plotting helpers for line-profile variability: symmetric colour scales for residual spectra,
dynamic spectra, profile bundles with an rms panel, time and dump axes, and power spectra with
a d^-1 axis.

Every drawing function draws on a matplotlib Axes supplied by the caller and returns the
artists it made. matplotlib is imported inside the functions (importing this module does not
import it); nothing here changes rcParams, creates a figure or saves one. Fonts, sizes and the
layout are left to the caller. Colours and line widths default to those of the M424 figures and
are overridden with the usual matplotlib keywords: aliases (lw, ls, c, va, fc, ec, ...) are
normalised to matplotlib's canonical names before the defaults are applied, so a caller's
keyword always wins, whichever spelling it uses.

PP 2026-10-01: extracted from the project's fig_disc_dynspec.py, fig_disc_dynspec_slide.py,
fig_disc_zerocross_spectrum.py and fig_disc_timeseries.py.

Validation
----------
Shadow copies of those four scripts rewritten on top of these helpers (with the arguments noted
in each docstring) reproduce all 17 M424 figure variants exactly: identical PNG pixels and, with
SOURCE_DATE_EPOCH fixed, identical PDF bytes (2026-10-01; standard, zoomed, --mean-all,
--symlog 1/2 and --tall dynamic spectra, the slide, the zero-crossing spectra for 1 and 8 lines
of sight, the time series; rechecked after the review fixes of the same day).
tests/synspec/test_plotting.py keeps panel-level pixel checks against transcriptions of the
scripts' drawing code.

Conventions
-----------
* A residual spectrum R has shape (n_t, n_x): time along axis 0 (drawn upwards), wavelength or
  velocity along axis 1. Values are converted to float64 before any arithmetic, as the project
  scripts do (``astype(np.float64)``), so float32 products give the legacy pixels.
* imshow draws evenly spaced pixels. The M424 wavelength axis lambda = lref exp(y/c) on the
  uniform velocity grid y is not evenly spaced (see :func:`dynspec_extent`); for wide windows
  plot against y, or use ``plot_dynamic_spectrum(..., mesh=True)``.
* Colour scales are symmetric about zero, [-lim, lim]. ``decades > 0`` selects a symmetric-log
  scale: linear within +-lim / base**decades, then ``decades`` logarithmic decades to +-lim.
* Frequencies are in microHz; :func:`add_cpd_axis` adds d^-1 (1 microHz = 0.0864 d^-1).
"""
import sys
import warnings

import numpy as np

from .conventions import CPD_PER_MUHZ

GREY = (0.5, 0.5, 0.5)                                          # figstyle.GREY of the project
ACCENT = (1.0, 0.5019607843137255, 0.054901960784313725)       # nugridpy linestylecb(1) orange
MEAN_LABEL = r"$\langle F\rangle_t$"
UNEVEN_TOL = 0.5                                                # default tolerance of dynspec_extent [steps]


# --------------------------------------------------------------------------- keyword handling
def _merged_kw(defaults, user, artist="line"):
    """
    ``defaults`` (canonical matplotlib names) updated with the caller's keywords, whose aliases
    (lw, ls, c, va, fc, ...) are first normalised with matplotlib.cbook.normalize_kwargs.

    artist: 'line' (Line2D), 'text' (Text) or 'patch' (FancyBboxPatch, a text box).
    Passing an alias and its full name together raises matplotlib's TypeError.
    """
    # PP 2026-10-01 (review): merging un-normalised keywords gave "Got both 'lw' and 'linewidth'" or
    # silently kept the default (mark_frequencies(linewidth=2), panel_text(verticalalignment=...))
    from matplotlib import cbook
    if artist == "line":
        from matplotlib.lines import Line2D as cls
    elif artist == "text":
        from matplotlib.text import Text as cls
    elif artist == "patch":
        from matplotlib.patches import FancyBboxPatch as cls
    else:
        raise ValueError("artist must be 'line', 'text' or 'patch', got {!r}".format(artist))
    kw = dict(defaults)
    kw.update(cbook.normalize_kwargs(dict(user or {}), cls))
    return kw


def _check_lim(lim):
    if not (np.isfinite(lim) and lim > 0):
        raise ValueError("lim must be finite and > 0, got {!r}".format(lim))


# --------------------------------------------------------------------------- colour scales
def percentile_limit(arrays, q=99.5, scale=1.0, nan="raise", stacked=False):
    """
    Common symmetric colour limit: the largest q-th percentile of |a| over the arrays, times scale.

    Parameters
    ----------
    arrays: np.ndarray or list/tuple of array-like
        One array or several (e.g. the residual spectra of all lines, so that they share one
        colour scale). An ndarray is ONE array, pooled over all its axes (unless ``stacked``);
        a list or tuple whose first element has ndim >= 1 is a sequence of arrays; a list or
        tuple of numbers is one array (as :func:`ppmpy.synspec.lpv.symmetric_limit`).
    q: float
        Percentile (M424 figures: 99.5).
    scale: float
        Applied after the percentile (e.g. 1e3 for units of 10^-3).
    nan: str
        'raise' (default): a percentile that is not finite (NaN or inf in an array) raises
        ValueError. 'omit': ignore NaN (np.nanpercentile); an array that is all NaN still raises.
    stacked: bool
        Treat a single ndarray as a stack of arrays along its first axis (e.g. (n_lines, n_t,
        n_x)): the result is the largest per-array percentile, not the percentile of the pooled
        values.

    Returns
    -------
    float
        ``max(np.percentile(np.abs(a), q) for a in arrays) * scale``.

    Validation
    ----------
    Bit-identical to the LIM of fig_disc_dynspec.py and fig_disc_dynspec_slide.py with the
    defaults (the per-array percentiles are collected in a float64 array; its max is the
    builtin max).
    """
    # PP 2026-10-01: ported from fig_disc_dynspec.py:73,125 and fig_disc_dynspec_slide.py:37
    # PP 2026-10-01 (review): order-independent NaN handling (builtin max gave nan or a number depending on the
    # position of the NaN array), stacked option, a list of numbers is one array
    if nan not in ("raise", "omit"):
        raise ValueError("nan must be 'raise' or 'omit', got {!r}".format(nan))
    if isinstance(arrays, np.ndarray):
        if stacked:
            if arrays.ndim < 2:
                raise ValueError("stacked=True needs an array with at least two dimensions, got {}".format(arrays.shape))
            arrays = list(arrays)
        else:
            arrays = [arrays]
    elif isinstance(arrays, (list, tuple)):
        if len(arrays) == 0:
            raise ValueError("no arrays given")
        if np.ndim(arrays[0]) == 0:
            arrays = [np.asarray(arrays)]
    else:
        arrays = [np.asarray(arrays)]
    pct = np.nanpercentile if nan == "omit" else np.percentile
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN slice: reported below
        vals = np.array([pct(np.abs(np.asarray(a)), q) for a in arrays], dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        bad = [i for i in range(vals.size) if not np.isfinite(vals[i])]
        raise ValueError("the {}th percentile of |a| is not finite for array(s) {} (NaN or inf in the data; "
                         "nan='omit' ignores NaN)".format(q, bad))
    return float(vals.max() * scale)


def symlog_linthresh(lim, decades=1.0, base=10):
    """Linear threshold of the symmetric-log scale: ``lim / base**decades``."""
    return lim / base ** decades


def symlog_norm(lim, decades=1.0, linscale=1.0, base=10):
    """
    Symmetric-log colour norm on [-lim, lim]: linear within +-lim / base**decades, logarithmic
    (``decades`` decades of ``base``) beyond.

    Parameters
    ----------
    lim: float
        Colour limit (finite, > 0).
    decades: float
        Number of logarithmic decades between the linear threshold and lim (> 0).
    linscale: float
        Width of the linear part, in decades, on each side of zero (matplotlib's ``linscale``).
    base: float
        Base of the logarithm.

    Returns
    -------
    matplotlib.colors.SymLogNorm
        ``SymLogNorm(linthresh=lim / base**decades, linscale=linscale, vmin=-lim, vmax=lim, base=base)``.

    Validation
    ----------
    lim=LIM, decades=SYMLOG reproduces ``fig_disc_dynspec.py --symlog SYMLOG``; decades=1 the slide
    figure (``linthresh = LIM / 10``), bit for bit.
    """
    # PP 2026-10-01: ported from fig_disc_dynspec.py:74,127 and fig_disc_dynspec_slide.py:38-39
    from matplotlib.colors import SymLogNorm
    _check_lim(lim)
    if not decades > 0:
        raise ValueError("decades must be > 0 for a symmetric-log norm (use linear_norm), got {!r}".format(decades))
    return SymLogNorm(linthresh=symlog_linthresh(lim, decades, base), linscale=linscale, vmin=-lim, vmax=lim,
                      base=base)


def linear_norm(lim):
    """
    Linear colour norm on [-lim, lim] centred on zero.

    Parameters
    ----------
    lim: float
        Colour limit (finite, > 0; an all-zero residual has no colour scale).

    Returns
    -------
    matplotlib.colors.TwoSlopeNorm
        ``TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)`` (the default of fig_disc_dynspec.py).
    """
    # PP 2026-10-01: ported from fig_disc_dynspec.py:75,129
    # PP 2026-10-01 (review): check lim as symlog_norm does (0 gave matplotlib's 'ascending order' error, NaN passed)
    from matplotlib.colors import TwoSlopeNorm
    _check_lim(lim)
    return TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)


def residual_norm(lim, decades=0.0, linscale=1.0, base=10):
    """
    The colour norm of the residual dynamic spectra: :func:`symlog_norm` if decades > 0, else
    :func:`linear_norm` (the ``--symlog`` option of fig_disc_dynspec.py, default 0 = linear).
    """
    if decades > 0:
        return symlog_norm(lim, decades=decades, linscale=linscale, base=base)
    return linear_norm(lim)


def symlog_ticks(lim, decades=1.0, base=10, fmt="{:.2g}"):
    """
    Colour-bar ticks of a symmetric-log scale: -lim, -linthresh, 0, linthresh, lim.

    Parameters
    ----------
    lim, decades, base: float
        As in :func:`symlog_norm`.
    fmt: str
        Format of the labels (zero is always labelled '0').

    Returns
    -------
    values: list of float
    labels: list of str
    """
    # PP 2026-10-01: ported from fig_disc_dynspec.py:104-106,166-168 and fig_disc_dynspec_slide.py:76-77
    lt = symlog_linthresh(lim, decades, base)
    return [-lim, -lt, 0, lt, lim], [fmt.format(-lim), fmt.format(-lt), "0", fmt.format(lt), fmt.format(lim)]


def colorbar_ticks(cb, lim, decades=1.0, base=10, fmt="{:.2g}", minor=False):
    """
    Ticks of a residual colour bar as in the M424 figures: for a symmetric-log scale
    (decades > 0) the five ticks of :func:`symlog_ticks` with their labels; for a linear scale
    (decades <= 0) matplotlib's default ticks. Minor ticks are switched off unless minor=True.

    Parameters
    ----------
    cb: matplotlib.colorbar.Colorbar
    lim, decades, base, fmt:
        As in :func:`symlog_ticks`.
    minor: bool
        Keep the minor ticks.

    Returns
    -------
    list of float or None
        The tick values set, None for a linear scale.

    Notes
    -----
    The minor ticks are switched off after the ticks are set; the --tall layout of
    fig_disc_dynspec.py does it before, with the same result (tests/synspec/test_plotting.py).
    """
    # PP 2026-10-01: ported from fig_disc_dynspec.py:102-106,164-169 and fig_disc_dynspec_slide.py:76-78
    ticks = None
    if decades > 0:
        ticks, labels = symlog_ticks(lim, decades=decades, base=base, fmt=fmt)
        cb.set_ticks(ticks)
        cb.set_ticklabels(labels)
    if not minor:
        cb.ax.minorticks_off()
    return ticks


# --------------------------------------------------------------------------- dynamic spectra
def spacing_error(v):
    """
    How unevenly a coordinate is spaced: the largest distance of v from the evenly spaced grid
    between v[0] and v[-1], in units of that grid's step.

    Parameters
    ----------
    v: array-like
        (n,) strictly monotonic coordinate, n >= 2.

    Returns
    -------
    float
        ``max_i |v_i - (v_0 + i s)| / |s|`` with s = (v[-1] - v[0]) / (n - 1); 0 for an evenly
        spaced v (to rounding).

    Notes
    -----
    For lambda = lref exp(y/c) on a uniform velocity grid y = -V..V with step dv, the largest
    distance is in the middle, about V**2 / (2 c dv) steps (lref (2V/c)**2 / 8 in Angstrom):
    0.27 for V = 400 km/s, 1.67 for 1000 km/s, 12.2 for the full M424 grid (2700 km/s), on
    dv = 1 km/s. The M424 dump times (dt = 2826-2843 s) give <= 0.003.
    """
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1 or v.size < 2:
        raise ValueError("need a 1-D coordinate with at least two values, got shape {}".format(v.shape))
    d = np.diff(v)
    if not (np.all(d > 0) or np.all(d < 0)):
        raise ValueError("the coordinate must be strictly monotonic (and finite)")
    s = (v[-1] - v[0]) / (v.size - 1)
    return float(np.max(np.abs(v - (v[0] + np.arange(v.size) * s))) / abs(s))


def _caller_stacklevel():
    """stacklevel of the first frame outside this module, for warnings.warn called from the caller of this function."""
    f, level = sys._getframe(1), 1
    while f is not None and f.f_code.co_filename == __file__:
        f, level = f.f_back, level + 1
    return level


def _check_spacing(v, name, uneven, tol):
    if uneven not in ("warn", "raise", "ignore"):
        raise ValueError("uneven must be 'warn', 'raise' or 'ignore', got {!r}".format(uneven))
    err = spacing_error(v)                                      # also checks n >= 2 and monotonic
    if uneven == "ignore" or err <= tol:
        return
    msg = ("{0} is unevenly spaced: imshow draws evenly spaced pixels, so a pixel centre lies up to {1:.3g} steps "
           "from its {0} value (tolerance {2:g}). Plot against an evenly spaced coordinate (e.g. the velocity grid y "
           "rather than lambda = lref exp(y/c)), use plot_dynamic_spectrum(..., mesh=True) for exact cell edges, or "
           "pass uneven='ignore'.".format(name, err, tol))
    if uneven == "raise":
        raise ValueError(msg)
    warnings.warn(msg, UserWarning, stacklevel=_caller_stacklevel())


def dynspec_extent(x, t, xpad=False, uneven="warn", uneven_tol=UNEVEN_TOL):
    """
    imshow extent of a dynamic spectrum, as in the M424 figures.

    Parameters
    ----------
    x: array-like
        (n_x,) abscissa of the columns (wavelength or velocity), strictly monotonic, n_x >= 2.
    t: array-like
        (n_t,) times of the rows, strictly monotonic, n_t >= 2.
    xpad: bool
        Also pad x by half the end steps, so that the first and last pixel centres fall on x[0]
        and x[-1]. Default False, as the legacy figures. The pixel centres in between fall on x
        only if x is evenly spaced (see Notes).
    uneven: str
        What to do when x or t is unevenly spaced by more than ``uneven_tol`` steps
        (:func:`spacing_error`): 'warn' (default), 'raise' (ValueError) or 'ignore'.
    uneven_tol: float
        Tolerated :func:`spacing_error`, in steps (default 0.5, the offset of the end pixels the
        legacy extent already has; the legacy +-400 km/s wavelength windows, 0.27, pass).

    Returns
    -------
    list
        ``[x[0], x[-1], t[0] - dt/2, t[-1] + dt/2]`` with dt = t[1] - t[0].

    Notes
    -----
    imshow spaces the n_x columns (and n_t rows) evenly over the extent. Consequences, kept by
    default because the legacy figures have them:

    1. x is not padded, so the n_x pixels span x[0]..x[-1] and their centres are off by up to
       half a step at the ends (0.5 km/s on the 1 km/s grid).
    2. An unevenly spaced x is drawn as if it were even. The M424 abscissa lambda = lref exp(y/c)
       on the uniform y grid puts the middle column ~V**2 / (2 c dv) steps away from its
       wavelength, for a window |y| <= V: 0.27 px (0.27 km/s) for V = 400 km/s (the figures),
       1.67 px for 1000 km/s, 12.2 px (about 12 km/s) for the full grid (2700 km/s). xpad=True
       fixes only the ends. For wide windows plot against y, or use
       ``plot_dynamic_spectrum(..., mesh=True)``; anything above ``uneven_tol`` warns.
    3. The time padding uses the first step at both ends. For the M424 dumps (dt = 2826-2843 s)
       the row centres are within 0.4 % of a step of the dump times.
    """
    # PP 2026-10-01: ported from fig_disc_dynspec.py:79,150 and fig_disc_dynspec_slide.py:69
    # PP 2026-10-01 (review): spacing check, at least two x values (one gave a zero-width extent), corrected Notes
    x = np.asarray(x)
    t = np.asarray(t)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("a dynamic spectrum needs at least two x values (one column has no width), got shape {}"
                         .format(x.shape))
    if t.ndim != 1 or t.size < 2:
        raise ValueError("a dynamic spectrum needs at least two times, got shape {}".format(t.shape))
    _check_spacing(x, "x", uneven, uneven_tol)
    _check_spacing(t, "t", uneven, uneven_tol)
    ht = 0.5 * (t[1] - t[0])
    if xpad:
        return [x[0] - 0.5 * (x[1] - x[0]), x[-1] + 0.5 * (x[-1] - x[-2]), t[0] - ht, t[-1] + ht]
    return [x[0], x[-1], t[0] - ht, t[-1] + ht]


def cell_edges(v):
    """
    Cell edges of a (possibly unevenly spaced) coordinate, for pcolormesh: the midpoints between
    neighbours, and half the end steps beyond the first and last value.

    Parameters
    ----------
    v: array-like
        (n,) strictly monotonic, n >= 2.

    Returns
    -------
    np.ndarray
        (n + 1,) float64 edges; cell i spans edges[i]..edges[i + 1] and contains v[i].
    """
    v = np.asarray(v, dtype=np.float64)
    spacing_error(v)                                            # checks 1-D, n >= 2, strictly monotonic
    mid = 0.5 * (v[1:] + v[:-1])
    return np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]])


def plot_dynamic_spectrum(ax, x, t, R, norm=None, cmap="RdBu_r", scale=1.0, xpad=False, mesh=False, uneven="warn",
                          uneven_tol=UNEVEN_TOL, **kw):
    """
    Residual dynamic spectrum R(x, t) as an image, time upwards.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
    x: array-like
        (n_x,) abscissa (e.g. wavelength in Angstrom, or velocity).
    t: array-like
        (n_t,) times.
    R: array-like
        (n_t, n_x) values (e.g. F - <F>_t); converted to float64 (as the legacy scripts do).
    norm: matplotlib.colors.Normalize, optional
        Colour norm (:func:`residual_norm`); None: ``linear_norm(percentile_limit(scale * R,
        nan='omit'))`` (NaN pixels are left blank; an all-zero R raises ValueError).
    cmap: str or Colormap
    scale: float
        R is drawn as ``R * scale`` (e.g. 1e3 for a colour bar in units of 10^-3).
    xpad: bool
        See :func:`dynspec_extent` (imshow only).
    mesh: bool
        False (default, the legacy figures): ``imshow`` with the extent of :func:`dynspec_extent`,
        which assumes evenly spaced x and t. True: ``pcolormesh`` with the cell edges of
        :func:`cell_edges`, for any spacing: each column covers the points nearer to its x than
        to the neighbouring ones, each row likewise in t (rasterized by default, so vector
        output stays small).
    uneven, uneven_tol:
        Spacing check of x and t for imshow, see :func:`dynspec_extent`. The default warns
        when a pixel centre is more than half a step from its coordinate.
    **kw:
        Override or extend the imshow keywords (origin='lower', aspect='auto',
        interpolation='nearest', extent from :func:`dynspec_extent`), or with mesh=True the
        pcolormesh keywords (shading='flat', rasterized=True).

    Returns
    -------
    matplotlib.image.AxesImage, or matplotlib.collections.QuadMesh with mesh=True

    Validation
    ----------
    With norm given and the defaults, the same image as fig_disc_dynspec.py and
    fig_disc_dynspec_slide.py (identical pixels), also for float32 input.
    """
    # PP 2026-10-01: ported from fig_disc_dynspec.py:79-80,150-153 and fig_disc_dynspec_slide.py:69-70
    # PP 2026-10-01 (review): float64 conversion (the scripts cast before R * 1e3), NaN-aware automatic norm,
    # spacing check, mesh option
    A = np.asarray(R, dtype=np.float64)
    if A.ndim != 2 or A.shape != (np.size(t), np.size(x)):
        raise ValueError("R must have shape (len(t), len(x)) = ({}, {}), got {}".format(np.size(t), np.size(x), A.shape))
    if scale != 1.0:
        A = A * scale
    if norm is None:
        norm = linear_norm(percentile_limit(A, nan="omit"))
    if mesh:
        mkw = dict(cmap=cmap, norm=norm, shading="flat", rasterized=True)
        mkw.update(kw)
        return ax.pcolormesh(cell_edges(x), cell_edges(t), A, **mkw)
    ikw = dict(origin="lower", aspect="auto",
               extent=dynspec_extent(x, t, xpad=xpad, uneven=uneven, uneven_tol=uneven_tol),
               cmap=cmap, norm=norm, interpolation="nearest")
    ikw.update(kw)
    return ax.imshow(A, **ikw)


def overlay_track(ax, xtrack, t, keep_xlim=True, **plot_kw):
    """
    Curve x(t) over a dynamic spectrum (e.g. the central zero crossing A(t)); NaN leaves gaps.

    Parameters
    ----------
    xtrack, t: array-like
        (n_t,) abscissa of the track and times.
    keep_xlim: bool
        Restore the x limits after plotting (the track must not widen the image).
    **plot_kw:
        Line2D keywords, any spelling (linewidth or lw, color or c, ...); defaults color='k',
        linewidth=0.35.

    Returns
    -------
    matplotlib.lines.Line2D

    Notes
    -----
    The track must come from the same residual definition as the image. The zero crossings of
    zerocross_<name>_los<k>.npz (fig_disc_zerocross.py, :func:`ppmpy.synspec.lpv.zero_crossing_track`
    on all dumps) are those of F - <F>_all; they are not the zero crossings of residuals about a
    window mean (a zoom with --d0/--d1). For imu los1, dumps 3200-3260, the two differ by a median
    of 2.9-3.9 km/s and by up to 71-148 km/s (the three lines). In a zoom either subtract the
    all-dump mean in the image, or recompute the track on the rows shown
    (``zero_crossing_track(y, F[window], lref)``). The legacy --tall zoom overlays the all-dump
    track on window-mean residuals.
    """
    # PP 2026-10-01: ported from fig_disc_dynspec.py:85-87
    xl = ax.get_xlim()
    line, = ax.plot(xtrack, t, **_merged_kw(dict(color="k", linewidth=0.35), plot_kw))
    if keep_xlim:
        ax.set_xlim(xl)
    return line


# --------------------------------------------------------------------------- profiles
def default_bundle_alpha(n, nmax_few=100, alpha_few=0.15, alpha_many=0.03):
    """Opacity of the individual profiles of a bundle: 0.15 for <= 100 profiles, else 0.03."""
    # PP 2026-10-01: ported from fig_disc_dynspec.py:136
    return alpha_few if n <= nmax_few else alpha_many


def plot_profile_bundle(ax, x, F, color=GREY, alpha=None, lw=0.6, mean=True, mean_kw=None, proxy_label=None,
                        proxy_kw=None, rasterized=True, zorder=1):
    """
    All profiles F[i](x) as one (rasterised) LineCollection, plus their mean.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
    x: array-like
        (n_x,) abscissa.
    F: array-like
        (n, n_x) profiles; converted to float64 (as the legacy scripts do).
    color:
        Colour of the individual profiles.
    alpha: float, optional
        Their opacity; None: :func:`default_bundle_alpha` (0.15 for <= 100 profiles, else 0.03).
    lw: float
        Their line width.
    mean: bool
        Also plot the mean profile F.mean(axis=0).
    mean_kw: dict, optional
        Line2D keywords of the mean line, any spelling; defaults color='k', linewidth=0.5,
        label=r'$\\langle F\\rangle_t$', zorder=2.
    proxy_label: str, optional
        Add an empty line as legend entry for the bundle; '{n}' is replaced by the number of
        profiles (fig_disc_dynspec.py: '{n} dumps').
    proxy_kw: dict, optional
        Line2D keywords of the proxy line, any spelling; defaults color=color, linewidth=0.8,
        alpha=0.7.
    rasterized: bool
        Rasterise the collection (vector output stays small for thousands of profiles).
    zorder: float
        zorder of the collection.

    Returns
    -------
    collection: matplotlib.collections.LineCollection
    mean_line: matplotlib.lines.Line2D or None
    proxy_line: matplotlib.lines.Line2D or None

    Notes
    -----
    Artists are added in the legacy order (collection, proxy, mean), so legend entries come out
    as in fig_disc_dynspec.py. The slide figure uses lw=0.8, alpha=0.15, mean_kw=dict(lw=1.0).
    """
    # PP 2026-10-01: ported from fig_disc_dynspec.py:136-140 and fig_disc_dynspec_slide.py:51-53
    from matplotlib.collections import LineCollection
    x = np.asarray(x, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    if F.ndim != 2 or F.shape[1] != x.size:
        raise ValueError("F must have shape (n, len(x)) with len(x) = {}, got {}".format(x.size, F.shape))
    if alpha is None:
        alpha = default_bundle_alpha(F.shape[0])
    lc = LineCollection(np.stack([np.broadcast_to(x, F.shape), F], axis=-1), colors=[color], linewidths=lw,
                        alpha=alpha, rasterized=rasterized, zorder=zorder)
    ax.add_collection(lc)
    proxy = None
    if proxy_label is not None:
        kw = _merged_kw(dict(color=color, linewidth=0.8, alpha=0.7, label=proxy_label.format(n=F.shape[0])), proxy_kw)
        proxy, = ax.plot([], [], **kw)
    mline = None
    if mean:
        kw = _merged_kw(dict(color="k", linewidth=0.5, label=MEAN_LABEL, zorder=2), mean_kw)
        mline, = ax.plot(x, F.mean(axis=0), **kw)
    else:
        ax.autoscale_view()              # add_collection alone does not rescale the view
    return lc, mline, proxy


def plot_rms_twin(ax, x, residuals, color=ACCENT, labels=None, linestyles=(None, "--", ":", "-."), lw=0.8, scale=1.0,
                  nbins=4, ylabel=None, stat="std"):
    """
    Spread over time of residual spectra on a twin y axis (right), starting at zero.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
    x: array-like
        (n_x,) abscissa.
    residuals: array-like or sequence of array-like
        One (n_t, n_x) residual spectrum, or several (a list, or an (n, n_t, n_x) array);
        converted to float64.
    color:
        Colour of the curves, the twin axis ticks and its label.
    labels: sequence of str, optional
        Legend labels, exactly one per residual (None entries get no label).
    linestyles: sequence
        Line style per residual (cycled); None = the rcParams default (solid).
    lw: float
    scale: float
        e.g. 1e3 for units of 10^-3.
    nbins: int
        MaxNLocator bins of the twin axis (fig_disc_dynspec.py 4, the slide 3).
    ylabel: str, optional
        Label of the twin axis (in ``color``).
    stat: str
        'std' (default, the legacy curve): ``R.std(axis=0) * scale``, the spread about the mean
        over the rows given. 'rms': ``sqrt(mean(R**2, axis=0)) * scale``.

    Returns
    -------
    matplotlib.axes.Axes
        The twin axis.

    Notes
    -----
    'std' equals the rms of R only when R has zero mean over the rows drawn, i.e. for residuals
    about the mean of the same window. The legacy figures label it "rms" in all cases; with
    fig_disc_dynspec.py --mean-all (R = F - <F>_all over a window) it is smaller than the rms:
    for imu los1, dumps 3200-3205, the median rms/std is 1.11/1.07/1.10 for the three lines, and
    the lambda4922 peak is 7.94e-4 (rms) vs 7.08e-4 (std). Use stat='rms' there, or label the
    curve 'std'.
    """
    # PP 2026-10-01: ported from fig_disc_dynspec.py:141-148 and fig_disc_dynspec_slide.py:54-58,66-67
    # PP 2026-10-01 (review): stat option, label count check, float64 conversion
    from matplotlib.ticker import MaxNLocator
    if stat not in ("std", "rms"):
        raise ValueError("stat must be 'std' or 'rms', got {!r}".format(stat))
    if isinstance(residuals, np.ndarray) and residuals.ndim == 2:
        residuals = [residuals]
    residuals = [np.asarray(R, dtype=np.float64) for R in residuals]
    nx = np.size(x)
    for R in residuals:
        if R.ndim != 2 or R.shape[1] != nx:
            raise ValueError("each residual must have shape (n_t, len(x)) with len(x) = {}, got {}".format(nx, R.shape))
    if labels is not None and len(labels) != len(residuals):
        raise ValueError("{} labels for {} residuals".format(len(labels), len(residuals)))
    tw = ax.twinx()
    for i, R in enumerate(residuals):
        kw = dict(color=color, linewidth=lw)
        ls = linestyles[i % len(linestyles)]
        if ls is not None:
            kw["linestyle"] = ls
        if labels is not None and labels[i] is not None:
            kw["label"] = labels[i]
        y = R.std(axis=0) if stat == "std" else np.sqrt(np.mean(R ** 2, axis=0))
        tw.plot(x, y * scale, **kw)
    tw.set_ylim(0, None)
    tw.yaxis.set_major_locator(MaxNLocator(nbins))
    tw.tick_params(axis="y", colors=color, which="both")
    if ylabel:
        tw.set_ylabel(ylabel, color=color)
    return tw


def legend_handles(*axes):
    """Legend handles and labels of several axes (e.g. an axis and its twin), concatenated in order."""
    # PP 2026-10-01: ported from fig_disc_dynspec.py:158-160
    handles, labels = [], []
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        handles += h
        labels += lab
    return handles, labels


# --------------------------------------------------------------------------- time axes
def time_since_start(t_s, unit="auto", hours_max_s=3 * 86400.0):
    """
    Times relative to the first one, in hours or days.

    Parameters
    ----------
    t_s: array-like
        Times [s].
    unit: str
        'auto' (hours if t_s[-1] - t_s[0] <= hours_max_s, else days), 'h', 'd' or 's'.
    hours_max_s: float
        Longest span shown in hours with unit='auto' (fig_disc_dynspec.py: 3 days).

    Returns
    -------
    t: np.ndarray
        ``(t_s - t_s[0]) / {3600, 86400, 1}``.
    unit: str
        'h', 'd' or 's'.
    """
    # PP 2026-10-01: ported from fig_disc_dynspec.py:55-56
    t_s = np.asarray(t_s, dtype=np.float64)
    if unit == "auto":
        unit = "h" if (t_s[-1] - t_s[0]) <= hours_max_s else "d"
    div = {"h": 3600.0, "d": 86400.0, "s": 1.0}
    if unit not in div:
        raise ValueError("unit must be 'auto', 'h', 'd' or 's', got {!r}".format(unit))
    return (t_s - t_s[0]) / div[unit], unit


def add_dump_axis(ax, t, dumps, label="dump", location="right", hide_ticks=True):
    """
    Secondary y axis with dump numbers, interpolated from the times of the rows.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        Axis whose y coordinate is time t.
    t, dumps: array-like
        (n,) increasing times and their dump numbers.
    label: str or None
    location: str
        'right' or 'left'.
    hide_ticks: bool
        Switch off ax's own y ticks on that side (they would overlap the dump ticks).

    Returns
    -------
    matplotlib.axes.Axes
        The secondary axis.
    """
    # PP 2026-10-01: ported from fig_disc_dynspec.py:94-97
    tt = np.asarray(t, dtype=np.float64)
    dd = np.asarray(dumps).astype(float)
    sec = ax.secondary_yaxis(location, functions=(lambda v: np.interp(v, tt, dd), lambda v: np.interp(v, dd, tt)))
    if label:
        sec.set_ylabel(label)
    if hide_ticks:
        ax.tick_params(axis="y", which="both", **{location: False})
    return sec


def plot_highlighted_series(ax, t, ys, highlight=0, offset=None, color="k", lw=0.5, label=None, other_color=GREY,
                            other_lw=0.3, other_alpha=0.5, other_label=None):
    """
    Several time series (e.g. one per line of sight) in grey with one of them highlighted on top.

    Parameters
    ----------
    t: array-like
        (n_t,) times.
    ys: array-like
        (n_t, n) series, one per column.
    highlight: int
        Column drawn last, in ``color``; -n <= highlight < n (negative counts from the end).
    offset: float, optional
        Subtracted from every series (e.g. the mean over all of them).
    color, lw, label:
        Style of the highlighted series (alpha 1).
    other_color, other_lw, other_alpha:
        Style of the others.
    other_label: str, optional
        Legend label of the first other series.

    Returns
    -------
    list of matplotlib.lines.Line2D
        n lines in drawing order (the others by column, then the highlighted one).
    """
    # PP 2026-10-01: ported from fig_disc_timeseries.py:40-45 (ys = diag[:, :, j, q] * scale, offset = its mean)
    # PP 2026-10-01 (review): ys must be 2-D; highlight range-checked and wrapped (-1 drew the last column twice)
    ys = np.asarray(ys)
    if ys.ndim != 2:
        raise ValueError("ys must be 2-D (n_t, n), got shape {}".format(ys.shape))
    n = ys.shape[1]
    highlight = int(highlight)
    if not -n <= highlight < n:
        raise ValueError("highlight must be in [-{0}, {0}) for {0} series, got {1}".format(n, highlight))
    highlight %= n
    order = [k for k in range(n) if k != highlight] + [highlight]
    lines = []
    first_other = True
    for k in order:
        y = ys[:, k]
        if offset is not None:
            y = y - offset
        if k == highlight:
            kw = dict(color=color, linewidth=lw, alpha=1.0, label=label)
        else:
            kw = dict(color=other_color, linewidth=other_lw, alpha=other_alpha, label=other_label if first_other else None)
            first_other = False
        lines.append(ax.plot(t, y, **kw)[0])
    return lines


# --------------------------------------------------------------------------- small layout helpers
def expand_ylim(ax, top=0.45, bottom=0.0):
    """Widen the y range by fractions of its height (room for a text block); returns the new limits."""
    # PP 2026-10-01: ported from fig_disc_timeseries.py:51-52
    lo, hi = ax.get_ylim()
    h = hi - lo
    ax.set_ylim(lo if bottom == 0 else lo - bottom * h, hi + top * h)
    return ax.get_ylim()


def panel_text(ax, text, x=0.03, y=0.97, va="top", box=False, box_kw=None, **text_kw):
    """
    Text in axes coordinates (panel annotations).

    Parameters
    ----------
    va: str
        Vertical alignment (``verticalalignment`` in text_kw takes precedence).
    box: bool
        Draw a translucent white box behind the text (fig_disc_dynspec.py: facecolor='w',
        edgecolor='none', alpha=0.7, pad=1.0).
    box_kw: dict, optional
        Overrides of the box keywords, any spelling (fc or facecolor, ...), and FancyBboxPatch
        keywords such as boxstyle.
    **text_kw:
        Further Text keywords, any spelling (e.g. fontsize, ha).

    Returns
    -------
    matplotlib.text.Text
    """
    # PP 2026-10-01: ported from fig_disc_dynspec.py:154-155, fig_disc_timeseries.py:53, fig_disc_zerocross_spectrum.py:95
    kw = dict(transform=ax.transAxes, verticalalignment=va)
    if box:
        kw["bbox"] = _merged_kw(dict(facecolor="w", edgecolor="none", alpha=0.7, pad=1.0), box_kw, "patch")
    kw.update(_merged_kw({}, text_kw, "text"))
    return ax.text(x, y, text, **kw)


def multiple_locators(axis, major, minor=None):
    """Major (and minor) ticks at multiples of the given steps on an Axis (ax.xaxis or ax.yaxis)."""
    # PP 2026-10-01: ported from fig_disc_dynspec.py:83-84,91-93
    from matplotlib.ticker import MultipleLocator
    axis.set_major_locator(MultipleLocator(major))
    if minor is not None:
        axis.set_minor_locator(MultipleLocator(minor))


# --------------------------------------------------------------------------- power spectra
def mark_frequencies(ax, freqs, color=ACCENT, ls="--", lw=0.6, zorder=0, **kw):
    """
    Dashed vertical lines at given frequencies (e.g. eigenmodes); returns the lines.

    Line2D keywords in ``**kw`` may use any spelling and take precedence over color, ls, lw and
    zorder (``linewidth=2`` gives width 2).
    """
    # PP 2026-10-01: ported from fig_disc_zerocross_spectrum.py:61-62
    kw = _merged_kw(dict(color=color, linestyle=ls, linewidth=lw, zorder=zorder), kw)
    return [ax.axvline(f, **kw) for f in np.atleast_1d(freqs)]


def plot_power_spectrum(ax, freq_muhz, power, markers_muhz=None, marker_kw=None, fmin=None, fmax=None, loglog=True,
                        clip=True, **line_kw):
    """
    Power spectrum as a thin line, optionally with frequency markers.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
    freq_muhz, power: array-like
        (n,) frequencies [microHz] and power.
    markers_muhz: array-like, optional
        Frequencies marked by :func:`mark_frequencies`.
    marker_kw: dict, optional
        Keywords for :func:`mark_frequencies` (defaults: ACCENT, '--', linewidth 0.6, zorder 0).
    fmin, fmax: float, optional
        x limits (either may be None).
    loglog: bool
        Log axes (``ax.loglog``); else ``ax.plot``.
    clip: bool
        Plot only the points with fmin <= f <= fmax (what the legacy script plots).
    **line_kw:
        Line2D keywords, any spelling; defaults color='k', linewidth=0.4, rasterized=True.

    Returns
    -------
    line: matplotlib.lines.Line2D
    markers: list of matplotlib.lines.Line2D

    Notes
    -----
    Draws in the legacy order: line, x limits, markers (fig_disc_zerocross_spectrum.py with
    marker_kw=dict(color=fs.cbcolor(1)), fmin=1.0, fmax=f_Nyquist).
    """
    # PP 2026-10-01: ported from fig_disc_zerocross_spectrum.py:59-62,85-86
    f = np.asarray(freq_muhz)
    p = np.asarray(power)
    if f.shape != p.shape or f.ndim != 1:
        raise ValueError("freq_muhz and power must be 1-D arrays of equal length, got {} and {}".format(f.shape, p.shape))
    if clip and (fmin is not None or fmax is not None):
        keep = np.ones(f.size, dtype=bool)
        if fmin is not None:
            keep &= f >= fmin
        if fmax is not None:
            keep &= f <= fmax
        f, p = f[keep], p[keep]
    kw = _merged_kw(dict(color="k", linewidth=0.4, rasterized=True), line_kw)
    line, = (ax.loglog if loglog else ax.plot)(f, p, **kw)
    if fmin is not None or fmax is not None:
        ax.set_xlim(fmin, fmax)
    markers = []
    if markers_muhz is not None:
        markers = mark_frequencies(ax, markers_muhz, **(marker_kw or {}))
    return line, markers


def add_cpd_axis(ax, label=r"frequency (d$^{-1}$)", location="top", factor=CPD_PER_MUHZ, **label_kw):
    """
    Secondary x axis in d^-1 for an axis in microHz.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        Axis with frequency in microHz along x.
    label: str or None
    location: str
        'top' or 'bottom'.
    factor: float
        d^-1 per microHz (conventions.CPD_PER_MUHZ = 0.0864).
    **label_kw:
        Keywords of the label (fig_disc_zerocross_spectrum.py: fontsize=7).

    Returns
    -------
    matplotlib.axes.Axes
        The secondary axis.

    Notes
    -----
    The legacy script used the literal 86400e-6, one ulp (1.39e-17) above CPD_PER_MUHZ
    (86400.0 * 1e-6 = 0.08639999999999999); the figures are identical.
    """
    # PP 2026-10-01: ported from fig_disc_zerocross_spectrum.py:92-93,112-113
    sec = ax.secondary_xaxis(location, functions=(lambda v: v * factor, lambda v: v / factor))
    if label:
        sec.set_xlabel(label, **label_kw)
    return sec
