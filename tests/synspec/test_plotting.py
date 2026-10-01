"""Tests of ppmpy.synspec.plotting (Agg canvases, no pyplot).

Besides unit tests of every helper, the *_pixel_identical tests draw the same panels twice, once
with transcriptions of the drawing code of the project scripts (fig_disc_dynspec.py standard and
--tall layouts, fig_disc_zerocross_spectrum.py, fig_disc_timeseries.py; single panels or one
column, with the script's colours written out, e.g. fs.cbcolor(1) as ORANGE) and once with the
helpers, and compare the rendered RGBA buffers exactly. Byte identity of the complete figures of
the real scripts was checked separately with shadow copies of the scripts (see the module
docstring of ppmpy.synspec.plotting); that harness lives outside the repository.
"""
import ast
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

from conftest import ROOT, m424_path

mpl = pytest.importorskip("matplotlib")
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import SymLogNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import MaxNLocator, MultipleLocator, NullLocator  # noqa: E402

from ppmpy.synspec import io as sio  # noqa: E402
from ppmpy.synspec import plotting as pl  # noqa: E402

GREY = (0.5, 0.5, 0.5)                                         # figstyle.GREY
ORANGE = (1.0, 0.5019607843137255, 0.054901960784313725)      # figstyle.cbcolor(1)
FIGSTYLE = os.environ.get("PPMPY_SYNSPEC_M424_FIGSTYLE",
                          "/home/ppathak/stellar-atmosphere-KU-Leuven/project/analysis/figstyle.py")
# the project's figstyle.set_style() without the savefig entries
STYLE = {"figure.dpi": 100, "font.family": "serif", "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
         "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
         "ytick.labelsize": 8, "axes.linewidth": 0.7, "lines.linewidth": 1.0, "lines.markersize": 3.6,
         "lines.markeredgewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
         "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True, "legend.frameon": False,
         "legend.handlelength": 2.2}


def _fig(**kw):
    fig = Figure(**kw)
    FigureCanvasAgg(fig)
    return fig


def _render(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def _synthetic(nt=40, nx=161, seed=3, lref=4026.22):
    """Profiles F (nt, nx) on a wavelength grid with a moving bump, residuals R = F - <F>_t."""
    rng = np.random.default_rng(seed)
    lam = lref * np.exp(np.linspace(-200, 200, nx) / 299792.458)
    t = np.cumsum(np.r_[0.0, 0.787 + 0.003 * rng.standard_normal(nt - 1)])
    base = 1 - 0.3 * np.exp(-0.5 * ((lam - lref) / 0.9) ** 2)
    shift = 0.05 * np.sin(2 * np.pi * t / 7.0)[:, None]
    F = base - 2e-3 * np.exp(-0.5 * ((lam - lref - shift) / 0.4) ** 2) + 2e-4 * rng.standard_normal((nt, nx))
    R = F - F.mean(axis=0)
    return lam, t, F, R


def _lam_grid(vwin, lref=4026.22, dv=1.0):
    """The M424 abscissa: lambda = lref exp(y/c) on the uniform velocity grid |y| <= vwin."""
    y = np.arange(-vwin, vwin + 0.5 * dv, dv)
    return lref * np.exp(y / 299792.458)


# --------------------------------------------------------------------------- module hygiene
def test_import_does_not_import_matplotlib():
    code = ("import sys; import ppmpy.synspec.plotting;"
            "print(','.join(m for m in ('matplotlib', 'ppmpy.ppm') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                         env=dict(os.environ, PYTHONPATH=ROOT))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", "imported at module load: " + out.stdout


def test_accent_is_nugrid_linestylecb_1():
    """ACCENT (the rms curves and eigenmode markers) = figstyle.cbcolor(1) = nugridpy linestylecb(1)[2]."""
    ut = pytest.importorskip("nugridpy.utils")
    assert tuple(ut.linestylecb(1)[2]) == pl.ACCENT == ORANGE


@pytest.mark.m424
def test_grey_is_project_figstyle_grey():
    """GREY = figstyle.GREY of the M424 project (read from the source, without importing figstyle)."""
    if not os.path.exists(FIGSTYLE):
        pytest.skip("project figstyle.py not available: {}".format(FIGSTYLE))
    with open(FIGSTYLE) as fh:
        tree = ast.parse(fh.read())
    vals = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(tg, ast.Name) and tg.id == "GREY" for tg in n.targets)]
    assert vals and tuple(vals[-1]) == pl.GREY == GREY


def test_no_rcparams_change():
    before = dict(mpl.rcParams)
    lam, t, F, R = _synthetic()
    fig = _fig()
    ax = fig.add_subplot(2, 1, 1)
    pl.plot_profile_bundle(ax, lam, F, proxy_label="{n} dumps")
    pl.plot_rms_twin(ax, lam, [R, 0.5 * R], ylabel="rms")
    im = pl.plot_dynamic_spectrum(fig.add_subplot(2, 1, 2), lam, t, R, norm=pl.residual_norm(1e-3, 1.0))
    pl.colorbar_ticks(fig.colorbar(im), 1e-3, 1.0)
    ax2 = fig.add_subplot(2, 2, 4)
    pl.plot_power_spectrum(ax2, np.arange(1.0, 100.0), np.arange(1.0, 100.0), markers_muhz=[10.0], fmin=2, fmax=50)
    pl.add_cpd_axis(ax2)
    _render(fig)
    assert dict(mpl.rcParams) == before


# --------------------------------------------------------------------------- colour scales
def test_percentile_limit_matches_legacy():
    rng = np.random.default_rng(0)
    RES = [rng.standard_normal((30, 50)) * s for s in (1e-3, 3e-4, 2e-3)]
    legacy = max(np.percentile(np.abs(R), 99.5) for R in RES) * 1e3
    assert pl.percentile_limit(RES, scale=1e3) == legacy
    assert pl.percentile_limit(RES[0]) == np.percentile(np.abs(RES[0]), 99.5)
    assert pl.percentile_limit(RES, q=100) == max(np.abs(R).max() for R in RES)


def test_percentile_limit_nan_stacked_and_scalars():
    rng = np.random.default_rng(1)
    R = rng.standard_normal((30, 50)) * 1e-3
    Rn = R.copy()
    Rn[3, 7] = np.nan
    # NaN raises in either position (the builtin max over percentiles depended on the order)
    for arrays in ([Rn, R], [R, Rn], Rn):
        with pytest.raises(ValueError, match="not finite"):
            pl.percentile_limit(arrays)
    with pytest.raises(ValueError, match="not finite"):
        pl.percentile_limit([R, np.where(R > 0, np.inf, R)])
    # nan='omit': order-independent, np.nanpercentile
    a = pl.percentile_limit([Rn, R], nan="omit")
    assert a == pl.percentile_limit([R, Rn], nan="omit")
    assert a == max(np.nanpercentile(np.abs(Rn), 99.5), np.percentile(np.abs(R), 99.5))
    with pytest.raises(ValueError, match="not finite"):
        pl.percentile_limit([R, np.full_like(R, np.nan)], nan="omit")
    with pytest.raises(ValueError):
        pl.percentile_limit(R, nan="ignore")
    # a stacked ndarray is pooled by default; stacked=True takes the per-array maximum
    S = np.stack([R, 3 * R, 0.5 * R])
    assert pl.percentile_limit(S) == np.percentile(np.abs(S), 99.5)
    assert pl.percentile_limit(S, stacked=True) == pl.percentile_limit(list(S)) == np.percentile(np.abs(3 * R), 99.5)
    assert pl.percentile_limit(S, stacked=True) > pl.percentile_limit(S)
    with pytest.raises(ValueError):
        pl.percentile_limit(R[0], stacked=True)
    # a list of numbers is one array, not several scalars
    v = [1.0, -3.0, 2.0, 0.5]
    assert pl.percentile_limit(v) == np.percentile(np.abs(v), 99.5) < 3.0
    assert pl.percentile_limit(v, q=50) == np.percentile(np.abs(v), 50)
    with pytest.raises(ValueError):
        pl.percentile_limit([])


def test_symlog_norm():
    lim = 2.37
    n = pl.symlog_norm(lim, decades=1.0)
    assert isinstance(n, SymLogNorm)
    assert n.vmin == -lim and n.vmax == lim and n.linthresh == lim / 10
    legacy = SymLogNorm(linthresh=lim / 10 ** 1.0, linscale=1.0, vmin=-lim, vmax=lim, base=10)
    v = np.linspace(-3, 3, 1001)
    np.testing.assert_array_equal(n(v), legacy(v))
    # 0 -> 1/2, +-lim -> 1/0, +-linthresh at 1/2 +- (linscale_adj / (linscale_adj + decades)) / 2
    adj = 1.0 / (1 - 1 / 10)
    np.testing.assert_allclose(n([-lim, 0.0, lim]), [0.0, 0.5, 1.0], atol=1e-15)
    np.testing.assert_allclose(n([lim / 10]), 0.5 + 0.5 * adj / (adj + 1.0), rtol=1e-12)
    n2 = pl.symlog_norm(1.0, decades=2.0, base=10)
    assert n2.linthresh == 0.01
    np.testing.assert_allclose(n2([0.1]), 0.5 + 0.5 * (adj + 1) / (adj + 2), rtol=1e-12)
    with pytest.raises(ValueError):
        pl.symlog_norm(1.0, decades=0.0)
    with pytest.raises(ValueError):
        pl.symlog_norm(0.0)


def test_linear_and_residual_norm():
    n = pl.linear_norm(4.0)
    assert isinstance(n, TwoSlopeNorm) and n.vcenter == 0.0 and n.vmin == -4.0 and n.vmax == 4.0
    np.testing.assert_allclose(n([-4.0, -2.0, 0.0, 4.0]), [0.0, 0.25, 0.5, 1.0])
    assert isinstance(pl.residual_norm(4.0), TwoSlopeNorm)
    assert isinstance(pl.residual_norm(4.0, decades=0.0), TwoSlopeNorm)
    s = pl.residual_norm(4.0, decades=1.5)
    assert isinstance(s, SymLogNorm) and s.linthresh == 4.0 / 10 ** 1.5


@pytest.mark.parametrize("lim", [0.0, -1.0, np.nan, np.inf])
def test_norms_reject_bad_limits(lim):
    for f in (pl.linear_norm, pl.symlog_norm, pl.residual_norm):
        with pytest.raises(ValueError, match="lim must be"):
            f(lim)


def test_symlog_ticks():
    v, lab = pl.symlog_ticks(3.0, 1.0)
    assert v == [-3.0, -0.3, 0, 0.3, 3.0]
    assert lab == ["-3", "-0.3", "0", "0.3", "3"]
    LIM = 2.8471
    v, lab = pl.symlog_ticks(LIM)
    lt = LIM / 10
    assert lab == [f"{-LIM:.2g}", f"{-lt:.2g}", "0", f"{lt:.2g}", f"{LIM:.2g}"]
    assert pl.symlog_ticks(1.0, 2.0, fmt="{:.1e}")[1][3] == "1.0e-02"


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_colorbar_ticks(orientation):
    lam, t, F, R = _synthetic()
    lim = pl.percentile_limit(R, scale=1e3)
    fig = _fig()
    ax = fig.add_subplot()
    im = pl.plot_dynamic_spectrum(ax, lam, t, R, norm=pl.symlog_norm(lim), scale=1e3)
    cb = fig.colorbar(im, orientation=orientation)
    ticks = pl.colorbar_ticks(cb, lim, 1.0)
    _render(fig)
    np.testing.assert_allclose(cb.get_ticks(), [-lim, -lim / 10, 0, lim / 10, lim])
    long = cb.ax.xaxis if orientation == "horizontal" else cb.ax.yaxis
    assert [x.get_text() for x in long.get_ticklabels()] == pl.symlog_ticks(lim)[1]
    assert isinstance(long.get_minor_locator(), NullLocator)
    assert ticks == pl.symlog_ticks(lim)[0]
    # linear scale: matplotlib's ticks, minor ticks off
    fig = _fig()
    im = pl.plot_dynamic_spectrum(fig.add_subplot(), lam, t, R, norm=pl.linear_norm(lim), scale=1e3)
    cb = fig.colorbar(im, orientation=orientation)
    assert pl.colorbar_ticks(cb, lim, 0.0) is None
    _render(fig)
    long = cb.ax.xaxis if orientation == "horizontal" else cb.ax.yaxis
    assert len(cb.get_ticks()) >= 3 and isinstance(long.get_minor_locator(), NullLocator)


def test_colorbar_ticks_keep_minor():
    lam, t, F, R = _synthetic()
    with mpl.rc_context({"ytick.minor.visible": True}):
        fig = _fig()
        im = pl.plot_dynamic_spectrum(fig.add_subplot(), lam, t, R, norm=pl.linear_norm(1e-3))
        cb = fig.colorbar(im)
        assert pl.colorbar_ticks(cb, 1e-3, 0.0, minor=True) is None
        _render(fig)
        assert not isinstance(cb.ax.yaxis.get_minor_locator(), NullLocator)
        assert len(cb.ax.yaxis.get_minorticklocs()) > 0


# --------------------------------------------------------------------------- dynamic spectra
def test_dynspec_extent():
    x = np.array([10.0, 11.0, 12.0])
    t = np.array([0.0, 1.0, 2.5])
    assert pl.dynspec_extent(x, t) == [10.0, 12.0, -0.5, 3.0]
    assert pl.dynspec_extent(x, t, xpad=True) == [9.5, 12.5, -0.5, 3.0]
    lam, t, F, R = _synthetic()
    legacy = [lam[0], lam[-1], t[0] - 0.5 * (t[1] - t[0]), t[-1] + 0.5 * (t[1] - t[0])]
    assert pl.dynspec_extent(lam, t) == legacy
    with pytest.raises(ValueError):
        pl.dynspec_extent(x, [1.0])
    with pytest.raises(ValueError, match="two x values"):          # one column used to give a zero-width extent
        pl.dynspec_extent([4026.0], t)
    with pytest.raises(ValueError, match="monotonic"):
        pl.dynspec_extent([1.0, 3.0, 2.0], t)
    with pytest.raises(ValueError, match="monotonic"):
        pl.dynspec_extent(x, [0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="uneven"):
        pl.dynspec_extent(x, t, uneven="silent")
    assert pl.dynspec_extent(x[::-1], [0.0, 1.0, 2.5]) == [12.0, 10.0, -0.5, 3.0]   # decreasing x is fine


def test_spacing_error():
    assert pl.spacing_error(np.arange(10.0)) < 1e-12
    assert pl.spacing_error(np.linspace(5.0, -3.0, 17)) < 1e-12
    assert pl.spacing_error([0.0, 2.0]) == 0.0
    c = 299792.458
    for vwin, expect in ((400.0, 0.2669), (1000.0, 1.6678), (2700.0, 12.158)):
        e = pl.spacing_error(_lam_grid(vwin))
        assert e == pytest.approx(expect, abs=1e-3)
        assert e == pytest.approx(vwin ** 2 / (2 * c), rel=2e-3)       # ~ V^2 / (2 c dv) steps, middle column
    with pytest.raises(ValueError):
        pl.spacing_error([1.0])
    with pytest.raises(ValueError):
        pl.spacing_error([0.0, 1.0, np.nan])


def _column_centre_offsets(ext, x):
    """Distance of imshow's (evenly spaced) column centres from x, in mean steps."""
    n = x.size
    w = (ext[1] - ext[0]) / n
    return (ext[0] + (np.arange(n) + 0.5) * w - x) / np.diff(x).mean()


def test_dynspec_extent_uneven_lambda_grid():
    t = np.arange(5.0)
    # the legacy +-400 km/s window: middle column 0.27 px off, below the tolerance -> no warning
    lam = _lam_grid(400.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ext = pl.dynspec_extent(lam, t)
    off = _column_centre_offsets(ext, lam)
    assert off[lam.size // 2] == pytest.approx(0.267, abs=2e-3) and abs(off[0]) == pytest.approx(0.5, abs=2e-3)
    off = _column_centre_offsets(pl.dynspec_extent(lam, t, xpad=True), lam)
    assert abs(off[0]) < 2e-3 and abs(off[-1]) < 2e-3 and off[lam.size // 2] == pytest.approx(0.267, abs=2e-3)
    # +-1000 km/s: 1.67 px in the middle -> warns (xpad does not help), raises on request, silent with 'ignore'
    lam = _lam_grid(1000.0)
    for xpad in (False, True):
        with pytest.warns(UserWarning, match="x is unevenly spaced"):
            ext = pl.dynspec_extent(lam, t, xpad=xpad)
        assert _column_centre_offsets(ext, lam)[lam.size // 2] == pytest.approx(1.67, abs=0.01)
    with pytest.raises(ValueError, match="unevenly spaced"):
        pl.dynspec_extent(lam, t, uneven="raise")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert pl.dynspec_extent(lam, t, uneven="ignore") == [lam[0], lam[-1], -0.5, 4.5]
        pl.dynspec_extent(lam, t, uneven_tol=2.0)
    # an evenly spaced coordinate (the velocity grid) never warns, and a gap in time does
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pl.dynspec_extent(np.arange(-2700.0, 2701.0), t)
    with pytest.warns(UserWarning, match="t is unevenly spaced"):
        pl.dynspec_extent(np.arange(3.0), np.r_[np.arange(10.0), np.arange(12.0, 20.0)])


def test_uneven_warning_points_at_caller():
    lam = _lam_grid(1000.0)
    t = np.arange(4.0)
    fig = _fig()
    with pytest.warns(UserWarning) as rec:
        pl.plot_dynamic_spectrum(fig.add_subplot(), lam, t, np.zeros((t.size, lam.size)) + 1e-3)
    assert [os.path.basename(w.filename) for w in rec] == [os.path.basename(__file__)]


def test_cell_edges():
    x = np.array([0.0, 1.0, 3.0, 4.0])
    np.testing.assert_array_equal(pl.cell_edges(x), [-0.5, 0.5, 2.0, 3.5, 4.5])
    np.testing.assert_allclose(pl.cell_edges(np.arange(5.0)), np.arange(6.0) - 0.5, atol=1e-15)
    np.testing.assert_array_equal(pl.cell_edges(x[::-1]), [4.5, 3.5, 2.0, 0.5, -0.5])
    lam = _lam_grid(2700.0)
    e = pl.cell_edges(lam)
    assert e.size == lam.size + 1 and np.all((e[:-1] < lam) & (lam < e[1:]))
    # every column is the set of points nearer to its lambda than to the neighbours
    np.testing.assert_allclose(e[1:-1], 0.5 * (lam[1:] + lam[:-1]), rtol=0, atol=0)
    with pytest.raises(ValueError):
        pl.cell_edges([1.0])


def test_plot_dynamic_spectrum_mesh_uneven_lambda():
    lam = _lam_grid(2700.0, dv=10.0)
    t = np.array([0.0, 1.0, 2.0, 4.0])                           # uneven in time as well
    R = np.random.default_rng(7).standard_normal((t.size, lam.size)) * 1e-3
    fig = _fig()
    ax = fig.add_subplot()
    with warnings.catch_warnings():
        warnings.simplefilter("error")                            # no spacing warning on the mesh path
        qm = pl.plot_dynamic_spectrum(ax, lam, t, R, norm=pl.linear_norm(3.0), scale=1e3, mesh=True)
    from matplotlib.collections import QuadMesh
    assert isinstance(qm, QuadMesh) and qm.get_rasterized()
    np.testing.assert_array_equal(np.asarray(qm.get_array()).reshape(R.shape), R * 1e3)
    xy = qm.get_coordinates()
    np.testing.assert_array_equal(xy[0, :, 0], pl.cell_edges(lam))
    np.testing.assert_array_equal(xy[:, 0, 1], pl.cell_edges(t))
    _render(fig)
    assert ax.get_xlim() == (pl.cell_edges(lam)[0], pl.cell_edges(lam)[-1])
    qm2 = pl.plot_dynamic_spectrum(fig.add_subplot(2, 1, 1), lam, t, R, mesh=True, rasterized=False, cmap="PuOr")
    assert not qm2.get_rasterized() and qm2.get_cmap().name == "PuOr"
    # the same data through imshow warns (12 px off in the middle at 1 km/s; 1.2 px here at 10 km/s)
    with pytest.warns(UserWarning, match="unevenly spaced"):
        pl.plot_dynamic_spectrum(fig.add_subplot(2, 1, 2), lam, t, R)


def test_plot_dynamic_spectrum():
    lam, t, F, R = _synthetic()
    fig = _fig()
    ax = fig.add_subplot()
    norm = pl.linear_norm(2.0)
    im = pl.plot_dynamic_spectrum(ax, lam, t, R, norm=norm, scale=1e3)
    assert list(im.get_extent()) == pl.dynspec_extent(lam, t)
    assert im.origin == "lower" and im.get_interpolation() == "nearest" and ax.get_aspect() == "auto"
    assert im.norm is norm and im.get_cmap().name == "RdBu_r"
    np.testing.assert_array_equal(np.asarray(im.get_array()), R * 1e3)
    assert ax.get_xlim() == (lam[0], lam[-1])
    im2 = pl.plot_dynamic_spectrum(fig.add_subplot(2, 1, 1), lam, t, R, cmap="PuOr", interpolation="bilinear")
    assert isinstance(im2.norm, TwoSlopeNorm) and im2.norm.vmax == pl.percentile_limit(R)
    assert im2.get_interpolation() == "bilinear" and im2.get_cmap().name == "PuOr"
    with pytest.raises(ValueError):
        pl.plot_dynamic_spectrum(ax, lam, t, R.T)
    _render(fig)


def test_plot_dynamic_spectrum_xpad():
    lam, t, F, R = _synthetic()
    fig = _fig()
    ax = fig.add_subplot()
    im = pl.plot_dynamic_spectrum(ax, lam, t, R, norm=pl.linear_norm(1e-3), xpad=True)
    ext = pl.dynspec_extent(lam, t, xpad=True)
    assert list(im.get_extent()) == ext
    assert ext[0] == lam[0] - 0.5 * (lam[1] - lam[0]) and ext[1] == lam[-1] + 0.5 * (lam[-1] - lam[-2])
    assert ax.get_xlim() == (ext[0], ext[1])
    _render(fig)


def test_plot_dynamic_spectrum_auto_norm_nan_and_zero():
    lam, t, F, R = _synthetic()
    Rn = R.copy()
    Rn[2, 5] = np.nan
    fig = _fig()
    im = pl.plot_dynamic_spectrum(fig.add_subplot(), lam, t, Rn, scale=1e3)
    assert np.isfinite(im.norm.vmax) and im.norm.vmax == np.nanpercentile(np.abs(Rn * 1e3), 99.5)
    _render(fig)
    with pytest.raises(ValueError, match="not finite"):
        pl.plot_dynamic_spectrum(fig.add_subplot(), lam, t, np.full_like(R, np.nan))
    with pytest.raises(ValueError, match="lim must be"):
        pl.plot_dynamic_spectrum(fig.add_subplot(), lam, t, np.zeros_like(R))


def test_float32_input_is_cast_like_the_scripts():
    """The scripts cast to float64 before R * 1e3, std and mean; the helpers do the same."""
    lam, t, F, R = _synthetic()
    F32 = F.astype(np.float32)
    R32 = R.astype(np.float32)                    # full float32 mantissa (a float32 difference F32 - <F32> often is exact)
    assert not np.array_equal(R32 * np.float32(1e3), R32.astype(np.float64) * 1e3)   # float32 arithmetic differs
    fig = _fig()
    ax = fig.add_subplot()
    im = pl.plot_dynamic_spectrum(ax, lam, t, R32, norm=pl.linear_norm(1.0), scale=1e3)
    A = np.asarray(im.get_array())
    assert A.dtype == np.float64 and np.array_equal(A, R32.astype(np.float64) * 1e3)
    tw = pl.plot_rms_twin(ax, lam, R32, scale=1e3)
    np.testing.assert_array_equal(tw.get_lines()[0].get_ydata(), R32.astype(np.float64).std(axis=0) * 1e3)
    _, mline, _ = pl.plot_profile_bundle(fig.add_subplot(2, 1, 1), lam, F32)
    np.testing.assert_array_equal(mline.get_ydata(), F32.astype(np.float64).mean(axis=0))
    # pixels: helper on float32 = legacy imshow on the float64 cast
    out = []
    for helper in (False, True):
        fig = _fig(figsize=(2.0, 2.0))
        ax = fig.add_subplot()
        norm = pl.symlog_norm(float(np.percentile(np.abs(R32.astype(np.float64)), 99.5) * 1e3))
        if helper:
            pl.plot_dynamic_spectrum(ax, lam, t, R32, norm=norm, scale=1e3)
        else:
            ax.imshow(R32.astype(np.float64) * 1e3, origin="lower", aspect="auto", extent=pl.dynspec_extent(lam, t),
                      cmap="RdBu_r", norm=norm, interpolation="nearest")
        out.append(_render(fig))
    assert np.array_equal(out[0], out[1])


def test_overlay_track_keeps_xlim():
    lam, t, F, R = _synthetic()
    fig = _fig()
    ax = fig.add_subplot()
    pl.plot_dynamic_spectrum(ax, lam, t, R)
    xl = ax.get_xlim()
    track = np.full(t.size, lam.mean())
    track[3] = lam[-1] + 5.0                                    # would widen the axis
    track[5] = np.nan
    line = pl.overlay_track(ax, track, t)
    _render(fig)
    assert ax.get_xlim() == xl
    assert line.get_color() == "k" and line.get_linewidth() == 0.35
    np.testing.assert_array_equal(line.get_ydata(), t)
    # keep_xlim=False lets the track widen the axis
    ax2 = fig.add_subplot(2, 1, 1)
    pl.plot_dynamic_spectrum(ax2, lam, t, R)
    pl.overlay_track(ax2, track, t, keep_xlim=False)
    assert ax2.get_xlim()[1] >= lam[-1] + 5.0


def test_overlay_track_long_keyword_names():
    lam, t, F, R = _synthetic()
    fig = _fig()
    ax = fig.add_subplot()
    track = np.full(t.size, lam.mean())
    line = pl.overlay_track(ax, track, t, linewidth=1, c="r", linestyle=":")
    assert line.get_linewidth() == 1 and line.get_color() == "r" and line.get_linestyle() == ":"
    line = pl.overlay_track(ax, track, t, lw=2, color="b", ls="--")
    assert line.get_linewidth() == 2 and line.get_color() == "b" and line.get_linestyle() == "--"
    with pytest.raises(TypeError):                                # both spellings at once: matplotlib's error
        pl.overlay_track(ax, track, t, lw=1, linewidth=2)


# --------------------------------------------------------------------------- profiles
def test_plot_profile_bundle():
    lam, t, F, R = _synthetic(nt=50)
    fig = _fig()
    ax = fig.add_subplot()
    lc, mline, proxy = pl.plot_profile_bundle(ax, lam, F, proxy_label="{n} dumps")
    assert isinstance(lc, LineCollection) and lc.get_rasterized() and lc.get_alpha() == 0.15
    segs = lc.get_segments()
    assert len(segs) == 50
    np.testing.assert_array_equal(segs[7], np.stack([lam, F[7]], axis=-1))
    np.testing.assert_array_equal(mline.get_ydata(), F.mean(axis=0))
    assert mline.get_zorder() == 2 and mline.get_linewidth() == 0.5
    assert proxy.get_label() == "50 dumps" and proxy.get_alpha() == 0.7 and len(proxy.get_xdata()) == 0
    assert ax.get_legend_handles_labels()[1] == ["50 dumps", r"$\langle F\rangle_t$"]
    assert pl.default_bundle_alpha(100) == 0.15 and pl.default_bundle_alpha(101) == 0.03
    _, _, F2, _ = _synthetic(nt=150)
    lc2, m2, p2 = pl.plot_profile_bundle(fig.add_subplot(2, 1, 1), lam, F2, mean=False)
    assert lc2.get_alpha() == 0.03 and m2 is None and p2 is None
    ylo, yhi = lc2.axes.get_ylim()
    assert ylo <= F2.min() and yhi >= F2.max()
    with pytest.raises(ValueError):
        pl.plot_profile_bundle(ax, lam[:-1], F)


def test_plot_profile_bundle_long_keyword_names():
    lam, t, F, R = _synthetic()
    fig = _fig()
    ax = fig.add_subplot()
    _, mline, proxy = pl.plot_profile_bundle(ax, lam, F, proxy_label="{n}", mean_kw=dict(linewidth=2, c="b"),
                                             proxy_kw=dict(linewidth=3, linestyle=":", color="r"))
    assert mline.get_linewidth() == 2 and mline.get_color() == "b" and mline.get_zorder() == 2
    assert proxy.get_linewidth() == 3 and proxy.get_linestyle() == ":" and proxy.get_color() == "r"
    assert proxy.get_alpha() == 0.7
    _, mline, proxy = pl.plot_profile_bundle(ax, lam, F, proxy_label="{n}", mean_kw=dict(lw=1.0), proxy_kw=dict(lw=1.5))
    assert mline.get_linewidth() == 1.0 and proxy.get_linewidth() == 1.5


def test_plot_rms_twin_and_legend():
    lam, t, F, R = _synthetic()
    fig = _fig()
    ax = fig.add_subplot()
    ax.plot(lam, F.mean(axis=0), label="mean")
    tw = pl.plot_rms_twin(ax, lam, [R, 0.5 * R], labels=["a", "b"], scale=1e3, nbins=3, ylabel="rms")
    l0, l1 = tw.get_lines()
    np.testing.assert_array_equal(l0.get_ydata(), R.std(axis=0) * 1e3)
    np.testing.assert_array_equal(l1.get_ydata(), (0.5 * R).std(axis=0) * 1e3)
    assert l0.get_linestyle() == "-" and l1.get_linestyle() == "--"
    assert tw.get_ylim()[0] == 0 and isinstance(tw.yaxis.get_major_locator(), MaxNLocator)
    assert tw.yaxis.get_major_locator()._nbins == 3
    assert tw.get_ylabel() == "rms" and tw.yaxis.label.get_color() == pl.ACCENT
    assert pl.legend_handles(ax, tw)[1] == ["mean", "a", "b"]
    tw1 = pl.plot_rms_twin(fig.add_subplot(2, 1, 1), lam, R)
    assert len(tw1.get_lines()) == 1
    _render(fig)


def test_plot_rms_twin_stat_and_labels():
    lam, t, F, R = _synthetic()
    Rall = R[:10] + 3e-4                       # residuals about another mean (--mean-all): non-zero window mean
    fig = _fig()
    ax = fig.add_subplot()
    tw = pl.plot_rms_twin(ax, lam, [R, Rall], stat="rms", scale=1e3)
    l0, l1 = tw.get_lines()
    np.testing.assert_array_equal(l0.get_ydata(), np.sqrt(np.mean(R ** 2, axis=0)) * 1e3)
    np.testing.assert_allclose(l0.get_ydata(), R.std(axis=0) * 1e3, rtol=1e-12)   # zero mean over the rows: rms = std
    np.testing.assert_array_equal(l1.get_ydata(), np.sqrt(np.mean(Rall ** 2, axis=0)) * 1e3)
    assert np.all(l1.get_ydata() > Rall.std(axis=0) * 1e3)                       # std (legacy) < rms here
    tws = pl.plot_rms_twin(ax, lam, Rall, stat="std")
    np.testing.assert_array_equal(tws.get_lines()[0].get_ydata(), Rall.std(axis=0))
    with pytest.raises(ValueError, match="stat"):
        pl.plot_rms_twin(ax, lam, R, stat="var")
    # one label per residual (a shorter list used to raise IndexError)
    with pytest.raises(ValueError, match="labels"):
        pl.plot_rms_twin(ax, lam, [R, Rall], labels=["only one"])
    with pytest.raises(ValueError, match="labels"):
        pl.plot_rms_twin(ax, lam, R, labels=["a", "b"])
    tw2 = pl.plot_rms_twin(ax, lam, R, labels=["single"])                      # the slide's call
    assert tw2.get_lines()[0].get_label() == "single"
    tw3 = pl.plot_rms_twin(ax, lam, np.stack([R, 2 * R]), labels=["a", None])   # (n, n_t, n_x) array
    assert len(tw3.get_lines()) == 2 and tw3.get_legend_handles_labels()[1] == ["a"]
    with pytest.raises(ValueError):
        pl.plot_rms_twin(ax, lam[:-1], R)


# --------------------------------------------------------------------------- time axes
def test_time_since_start():
    t_s = 1e5 + np.arange(0.0, 2.9 * 86400, 2835.0)
    t, u = pl.time_since_start(t_s)
    assert u == "h"
    np.testing.assert_array_equal(t, (t_s - t_s[0]) / 3600.0)
    t_s2 = 1e5 + np.arange(0.0, 3.2 * 86400, 2835.0)
    t, u = pl.time_since_start(t_s2)
    assert u == "d"
    np.testing.assert_array_equal(t, (t_s2 - t_s2[0]) / 86400.0)
    assert pl.time_since_start(t_s2, unit="h")[1] == "h"
    np.testing.assert_array_equal(pl.time_since_start(t_s2, unit="s")[0], t_s2 - t_s2[0])
    with pytest.raises(ValueError):
        pl.time_since_start(t_s, unit="min")


def test_add_dump_axis():
    lam, t, F, R = _synthetic()
    dumps = 3200 + np.arange(t.size)
    fig = _fig()
    ax = fig.add_subplot()
    pl.plot_dynamic_spectrum(ax, lam, t, R)
    sec = pl.add_dump_axis(ax, t, dumps)
    _render(fig)
    np.testing.assert_allclose(sec.get_ylim(), np.interp(ax.get_ylim(), t, dumps.astype(float)))
    assert sec.get_ylabel() == "dump"
    assert ax.yaxis.get_tick_params(which="major")["right"] is False
    assert ax.yaxis.get_tick_params(which="minor")["right"] is False


def test_add_dump_axis_left_and_keep_ticks():
    lam, t, F, R = _synthetic()
    dumps = 3200 + np.arange(t.size)
    with mpl.rc_context({"ytick.left": True, "ytick.right": True}):
        fig = _fig()
        ax = fig.add_subplot()
        pl.plot_dynamic_spectrum(ax, lam, t, R)
        sec = pl.add_dump_axis(ax, t, dumps, label=None, location="left")
        _render(fig)
        np.testing.assert_allclose(sec.get_ylim(), np.interp(ax.get_ylim(), t, dumps.astype(float)))
        assert sec.get_ylabel() == ""
        assert ax.yaxis.get_tick_params(which="major")["left"] is False
        assert ax.yaxis.get_tick_params(which="major").get("right", True) is not False
        ax2 = fig.add_subplot(2, 1, 1)
        pl.plot_dynamic_spectrum(ax2, lam, t, R)
        pl.add_dump_axis(ax2, t, dumps, hide_ticks=False)
        _render(fig)
        assert ax2.yaxis.get_tick_params(which="major").get("right", True) is not False
        assert ax2.yaxis.get_tick_params(which="minor").get("right", True) is not False


def test_plot_highlighted_series():
    rng = np.random.default_rng(5)
    t = np.arange(20.0)
    ys = rng.standard_normal((20, 8))
    fig = _fig()
    ax = fig.add_subplot()
    lines = pl.plot_highlighted_series(ax, t, ys, offset=ys.mean(), label="los1", other_label="los2-8")
    assert len(lines) == 8
    np.testing.assert_array_equal(lines[-1].get_ydata(), ys[:, 0] - ys.mean())
    np.testing.assert_array_equal(lines[0].get_ydata(), ys[:, 1] - ys.mean())
    assert lines[-1].get_color() == "k" and lines[-1].get_alpha() == 1.0 and lines[-1].get_linewidth() == 0.5
    assert lines[0].get_color() == pl.GREY and lines[0].get_alpha() == 0.5 and lines[0].get_linewidth() == 0.3
    assert ax.get_legend_handles_labels()[1] == ["los2-8", "los1"]
    lines = pl.plot_highlighted_series(ax, t, ys, highlight=3)
    np.testing.assert_array_equal(lines[-1].get_ydata(), ys[:, 3])


def test_plot_highlighted_series_negative_and_bad_index():
    rng = np.random.default_rng(6)
    t = np.arange(20.0)
    ys = rng.standard_normal((20, 8))
    fig = _fig()
    ax = fig.add_subplot()
    lines = pl.plot_highlighted_series(ax, t, ys, highlight=-1)
    assert len(lines) == 8 and len(ax.get_lines()) == 8            # -1 used to draw the last column twice
    np.testing.assert_array_equal(lines[-1].get_ydata(), ys[:, 7])
    assert lines[-1].get_color() == "k"
    assert [list(ln.get_ydata()) for ln in lines[:-1]] == [list(ys[:, k]) for k in range(7)]
    for bad in (8, -9):
        with pytest.raises(ValueError, match="highlight"):
            pl.plot_highlighted_series(ax, t, ys, highlight=bad)
    with pytest.raises(ValueError, match="2-D"):
        pl.plot_highlighted_series(ax, t, ys[:, 0])


# --------------------------------------------------------------------------- small layout helpers
def test_layout_helpers():
    fig = _fig()
    ax = fig.add_subplot()
    ax.set_ylim(0.0, 2.0)
    assert pl.expand_ylim(ax) == (0.0, 2.9)
    assert pl.expand_ylim(ax, top=0.0, bottom=0.5) == pytest.approx((-1.45, 2.9))
    txt = pl.panel_text(ax, "with velocities", box=True, fontsize=6.5)
    assert txt.get_transform() is ax.transAxes and txt.get_va() == "top" and txt.get_fontsize() == 6.5
    assert txt.get_bbox_patch() is not None and txt.get_bbox_patch().get_alpha() == 0.7
    txt2 = pl.panel_text(ax, "los1", y=0.06, va="baseline")
    assert txt2.get_bbox_patch() is None and txt2.get_position() == (0.03, 0.06)
    pl.multiple_locators(ax.xaxis, 5, 1)
    assert isinstance(ax.xaxis.get_major_locator(), MultipleLocator)
    assert isinstance(ax.xaxis.get_minor_locator(), MultipleLocator)
    _render(fig)


def test_panel_text_long_keyword_names():
    from matplotlib.colors import to_rgba
    fig = _fig()
    ax = fig.add_subplot()
    txt = pl.panel_text(ax, "a", verticalalignment="bottom", horizontalalignment="right")
    assert txt.get_verticalalignment() == "bottom" and txt.get_horizontalalignment() == "right"
    assert pl.panel_text(ax, "b", va="center").get_va() == "center"
    for box_kw in (dict(facecolor="y"), dict(fc="y")):
        txt = pl.panel_text(ax, "c", box=True, box_kw=box_kw)
        patch = txt.get_bbox_patch()
        assert patch.get_facecolor() == to_rgba("y", 0.7)            # the default alpha 0.7 is kept
    txt = pl.panel_text(ax, "d", box=True, box_kw=dict(edgecolor="k", alpha=1.0, boxstyle="round"))
    assert txt.get_bbox_patch().get_edgecolor() == to_rgba("k") and txt.get_bbox_patch().get_facecolor() == to_rgba("w")
    _render(fig)
    with pytest.raises(TypeError):
        pl.panel_text(ax, "e", box=True, box_kw=dict(fc="y", facecolor="r"))


# --------------------------------------------------------------------------- power spectra
def test_plot_power_spectrum_and_cpd_axis():
    f = np.linspace(0.01, 200.0, 20000)
    p = 1.0 / f
    fig = _fig()
    ax = fig.add_subplot()
    line, marks = pl.plot_power_spectrum(ax, f, p, markers_muhz=[10.95588503, 85.2], fmin=1.0, fmax=176.0)
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    x = line.get_xdata()
    assert x.min() >= 1.0 and x.max() <= 176.0 and x.size == ((f >= 1.0) & (f <= 176.0)).sum()
    assert ax.get_xlim() == (1.0, 176.0)
    assert [m.get_xdata()[0] for m in marks] == [10.95588503, 85.2]
    assert marks[0].get_linestyle() == "--" and marks[0].get_zorder() == 0 and marks[0].get_color() == pl.ACCENT
    assert line.get_rasterized() and line.get_linewidth() == 0.4
    sec = pl.add_cpd_axis(ax, fontsize=7)
    _render(fig)
    np.testing.assert_allclose(sec.get_xlim(), np.array(ax.get_xlim()) * 0.0864, rtol=1e-14)
    assert sec.get_xlabel() == r"frequency (d$^{-1}$)" and sec.xaxis.label.get_fontsize() == 7
    ax2 = fig.add_subplot(2, 1, 1)
    line2, marks2 = pl.plot_power_spectrum(ax2, f, p, loglog=False, clip=False, fmax=50.0, color="r")
    assert ax2.get_xscale() == "linear" and line2.get_xdata().size == f.size and marks2 == []
    assert ax2.get_xlim()[1] == 50.0 and line2.get_color() == "r"
    with pytest.raises(ValueError):
        pl.plot_power_spectrum(ax2, f, p[:-1])


def test_power_spectrum_and_markers_long_keyword_names():
    f = np.linspace(1.0, 100.0, 500)
    fig = _fig()
    ax = fig.add_subplot()
    line, marks = pl.plot_power_spectrum(ax, f, 1 / f, markers_muhz=[10.0, 20.0], linewidth=1, c="g",
                                         marker_kw=dict(linewidth=2, linestyle=":", c="m"))
    assert line.get_linewidth() == 1 and line.get_color() == "g" and line.get_rasterized()
    assert all(m.get_linewidth() == 2 and m.get_linestyle() == ":" and m.get_color() == "m" for m in marks)
    marks = pl.mark_frequencies(ax, [1.0], linewidth=2)            # used to stay at lw 0.6
    assert marks[0].get_linewidth() == 2 and marks[0].get_linestyle() == "--"
    marks = pl.mark_frequencies(ax, [1.0, 2.0], lw=1.5, ls="-", color="k")
    assert marks[1].get_linewidth() == 1.5 and marks[1].get_linestyle() == "-" and marks[1].get_color() == "k"
    _render(fig)


# --------------------------------------------------------------------------- pixel identity with the legacy code
def _legacy_dynspec(fig, lam, t, F, R, R0, LIM, symlog, ZC=None, dumps=None):
    """Transcription of fig_disc_dynspec.py: one column of the standard layout, plus the A(t) overlay and dump axis of
    the tall layout moved onto the middle panel."""
    norm = (SymLogNorm(linthresh=LIM / 10 ** symlog, linscale=1.0, vmin=-LIM, vmax=LIM, base=10) if symlog > 0
            else TwoSlopeNorm(vcenter=0.0, vmin=-LIM, vmax=LIM))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.6, 1.6, 0.06], hspace=0.12, top=0.9)
    ax = fig.add_subplot(gs[0])
    axd = [fig.add_subplot(gs[r], sharex=ax) for r in (1, 2)]
    cax = fig.add_subplot(gs[3])
    alpha = 0.15 if F.shape[0] <= 100 else 0.03
    ax.add_collection(LineCollection(np.stack([np.broadcast_to(lam, F.shape), F], axis=-1), colors=[GREY], linewidths=0.6,
                                     alpha=alpha, rasterized=True, zorder=1))
    ax.plot([], [], color=GREY, lw=0.8, alpha=0.7, label=f"{F.shape[0]} dumps")
    ax.plot(lam, F.mean(axis=0), color="k", lw=0.5, label=r"$\langle F\rangle_t$", zorder=2)
    tw = ax.twinx()
    tw.plot(lam, R.std(axis=0) * 1e3, color=ORANGE, lw=0.8, label="rms, with velocities")
    tw.plot(lam, R0.std(axis=0) * 1e3, color=ORANGE, lw=0.8, ls="--", label=r"rms, $T_{\rm eff}'$ only")
    tw.set_ylim(0, None)
    tw.yaxis.set_major_locator(MaxNLocator(4))
    tw.tick_params(axis="y", colors=ORANGE, which="both")
    tw.set_ylabel(r"rms of $F-\langle F\rangle_t$ ($10^{-3}$)", color=ORANGE)
    ext = [lam[0], lam[-1], t[0] - 0.5 * (t[1] - t[0]), t[-1] + 0.5 * (t[1] - t[0])]
    for r, (A, lab) in enumerate(((R, "with velocities"), (R0, r"$T_{\rm eff}'$ only"))):
        im = axd[r].imshow(A * 1e3, origin="lower", aspect="auto", extent=ext, cmap="RdBu_r", norm=norm,
                           interpolation="nearest")
        axd[r].text(0.03, 0.97, lab, transform=axd[r].transAxes, fontsize=6.5, va="top",
                    bbox=dict(fc="w", ec="none", alpha=0.7, pad=1.0))
    axd[1].xaxis.set_major_locator(MultipleLocator(5))
    axd[1].xaxis.set_minor_locator(MultipleLocator(1))
    if ZC is not None:
        axd[0].plot(ZC, t, color="k", lw=0.35, label=r"$A(t)$")
        axd[0].set_xlim(lam[0], lam[-1])
        sec = axd[0].secondary_yaxis("right", functions=(lambda x: np.interp(x, t, dumps), lambda x: np.interp(x, dumps, t)))
        sec.set_ylabel("dump")
        axd[0].tick_params(axis="y", which="both", right=False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = tw.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper center", ncol=4, handlelength=1.8, bbox_to_anchor=(0.5, 0.955))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7, length=2)
    if symlog > 0:
        lt = LIM / 10 ** symlog
        cb.set_ticks([-LIM, -lt, 0, lt, LIM])
        cb.set_ticklabels([f"{-LIM:.2g}", f"{-lt:.2g}", "0", f"{lt:.2g}", f"{LIM:.2g}"])
    cb.ax.minorticks_off()
    cb.set_label(r"$F-\langle F\rangle_t$ ($10^{-3}$)", fontsize=7)


def _helper_dynspec(fig, lam, t, F, R, R0, LIM, symlog, ZC=None, dumps=None):
    """The same panel drawn with ppmpy.synspec.plotting."""
    norm = pl.residual_norm(LIM, symlog)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.6, 1.6, 0.06], hspace=0.12, top=0.9)
    ax = fig.add_subplot(gs[0])
    axd = [fig.add_subplot(gs[r], sharex=ax) for r in (1, 2)]
    cax = fig.add_subplot(gs[3])
    pl.plot_profile_bundle(ax, lam, F, color=GREY, proxy_label="{n} dumps")
    tw = pl.plot_rms_twin(ax, lam, [R, R0], color=ORANGE, labels=["rms, with velocities", r"rms, $T_{\rm eff}'$ only"],
                          scale=1e3, ylabel=r"rms of $F-\langle F\rangle_t$ ($10^{-3}$)")
    for r, (A, lab) in enumerate(((R, "with velocities"), (R0, r"$T_{\rm eff}'$ only"))):
        im = pl.plot_dynamic_spectrum(axd[r], lam, t, A, norm=norm, scale=1e3)
        pl.panel_text(axd[r], lab, box=True, fontsize=6.5)
    pl.multiple_locators(axd[1].xaxis, 5, 1)
    if ZC is not None:
        pl.overlay_track(axd[0], ZC, t, label=r"$A(t)$")
        pl.add_dump_axis(axd[0], t, dumps)
    h, lab = pl.legend_handles(ax, tw)
    fig.legend(h, lab, fontsize=7, loc="upper center", ncol=4, handlelength=1.8, bbox_to_anchor=(0.5, 0.955))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7, length=2)
    pl.colorbar_ticks(cb, LIM, symlog)
    cb.set_label(r"$F-\langle F\rangle_t$ ($10^{-3}$)", fontsize=7)


def _compare_dynspec(lam, t, F, R, R0, LIM, symlog, ZC=None, dumps=None):
    out = []
    for draw in (_legacy_dynspec, _helper_dynspec):
        with mpl.rc_context(STYLE):
            fig = _fig(figsize=(2.6, 5.0))
            draw(fig, lam, t, F, R, R0, LIM, symlog, ZC=ZC, dumps=dumps)
            out.append(_render(fig))
    assert out[0].shape == out[1].shape
    assert np.array_equal(out[0], out[1]), "{} pixels differ".format(int(np.any(out[0] != out[1], axis=-1).sum()))


@pytest.mark.parametrize("symlog", [0.0, 1.0, 2.0])
def test_dynspec_pixel_identical(symlog):
    lam, t, F, R = _synthetic()
    R0 = 0.3 * R[::-1]
    LIM = max(np.percentile(np.abs(A), 99.5) for A in (R, R0)) * 1e3
    assert pl.percentile_limit([R, R0], scale=1e3) == LIM
    ZC = lam.mean() + 0.02 * np.sin(t)
    ZC[4] = np.nan
    _compare_dynspec(lam, t, F, R, R0, LIM, symlog, ZC=ZC, dumps=(3200 + np.arange(t.size)).astype(float))


def _legacy_tall(fig, lams, t, RES, LIM, symlog, ZC, dumps, hours):
    """Transcription of the --tall branch of fig_disc_dynspec.py (lines 76-107): sharey, overlay then set_xlim,
    minorticks_off BEFORE the colour-bar ticks are set."""
    norm = (SymLogNorm(linthresh=LIM / 10 ** symlog, linscale=1.0, vmin=-LIM, vmax=LIM, base=10) if symlog > 0
            else TwoSlopeNorm(vcenter=0.0, vmin=-LIM, vmax=LIM))
    axs = fig.subplots(1, 3, sharey=True)
    for j in range(3):
        lam = lams[j]
        ext = [lam[0], lam[-1], t[0] - 0.5 * (t[1] - t[0]), t[-1] + 0.5 * (t[1] - t[0])]
        im = axs[j].imshow(RES[j] * 1e3, origin="lower", aspect="auto", extent=ext, cmap="RdBu_r", norm=norm,
                           interpolation="nearest")
        axs[j].set_title("line {}".format(j), pad=4)
        axs[j].set_xlabel(r"$\lambda$ ($\mathrm{\AA}$)")
        axs[j].xaxis.set_major_locator(MultipleLocator(5))
        axs[j].xaxis.set_minor_locator(MultipleLocator(1))
        if ZC is not None:
            axs[j].plot(ZC[:, j], t, color="k", lw=0.35, label=r"$A(t)$: $F = \langle F\rangle_t$" if j == 0 else None)
            axs[j].set_xlim(lam[0], lam[-1])
    if ZC is not None:
        fig.legend(loc="lower left", bbox_to_anchor=(0.06, 0.03), fontsize=7, handlelength=1.5)
    axs[0].set_ylabel("time since dump 3200 ({})".format("h" if hours else "d"))
    if not hours:
        axs[0].yaxis.set_major_locator(MultipleLocator(5))
        axs[0].yaxis.set_minor_locator(MultipleLocator(1))
    tt, dd = t, dumps
    sec = axs[2].secondary_yaxis("right", functions=(lambda x: np.interp(x, tt, dd), lambda x: np.interp(x, dd, tt)))
    sec.set_ylabel("dump")
    axs[2].tick_params(axis="y", which="both", right=False)
    fig.subplots_adjust(wspace=0.08, bottom=0.12, top=0.96)
    cax = fig.add_axes([0.3, 0.045, 0.4, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7, length=2)
    cb.ax.minorticks_off()
    if symlog > 0:
        lt = LIM / 10 ** symlog
        cb.set_ticks([-LIM, -lt, 0, lt, LIM])
        cb.set_ticklabels([f"{-LIM:.2g}", f"{-lt:.2g}", "0", f"{lt:.2g}", f"{LIM:.2g}"])
    cb.set_label(r"$F-\langle F\rangle_t$ ($10^{-3}$), with velocities", fontsize=7)


def _helper_tall(fig, lams, t, RES, LIM, symlog, ZC, dumps, hours):
    norm = pl.residual_norm(LIM, symlog)
    axs = fig.subplots(1, 3, sharey=True)
    for j in range(3):
        im = pl.plot_dynamic_spectrum(axs[j], lams[j], t, RES[j], norm=norm, scale=1e3)
        axs[j].set_title("line {}".format(j), pad=4)
        axs[j].set_xlabel(r"$\lambda$ ($\mathrm{\AA}$)")
        pl.multiple_locators(axs[j].xaxis, 5, 1)
        if ZC is not None:
            pl.overlay_track(axs[j], ZC[:, j], t, label=r"$A(t)$: $F = \langle F\rangle_t$" if j == 0 else None)
    if ZC is not None:
        fig.legend(loc="lower left", bbox_to_anchor=(0.06, 0.03), fontsize=7, handlelength=1.5)
    axs[0].set_ylabel("time since dump 3200 ({})".format("h" if hours else "d"))
    if not hours:
        pl.multiple_locators(axs[0].yaxis, 5, 1)
    pl.add_dump_axis(axs[2], t, dumps)
    fig.subplots_adjust(wspace=0.08, bottom=0.12, top=0.96)
    cax = fig.add_axes([0.3, 0.045, 0.4, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7, length=2)
    pl.colorbar_ticks(cb, LIM, symlog)
    cb.set_label(r"$F-\langle F\rangle_t$ ($10^{-3}$), with velocities", fontsize=7)


@pytest.mark.parametrize("symlog,hours,zc", [(0.0, False, True), (1.0, False, True), (2.0, True, True), (1.0, False, False)])
def test_tall_layout_pixel_identical(symlog, hours, zc):
    lams, RES, ZC = [], [], []
    for j, lref in enumerate((4026.22, 4199.9, 4921.93)):
        lam, t, F, R = _synthetic(seed=10 + j, lref=lref)
        lams.append(lam)
        RES.append(R * (1.0, 0.3, 1.4)[j])
        zc_j = lref + 0.03 * np.sin(t + j)
        zc_j[3 + j] = np.nan
        zc_j[7] = lam[-1] + 1.0                                  # outside the image: set_xlim / keep_xlim must clip
        ZC.append(zc_j)
    ZC = np.stack(ZC, axis=1)
    LIM = max(np.percentile(np.abs(R), 99.5) for R in RES) * 1e3
    assert pl.percentile_limit(RES, scale=1e3) == LIM
    dumps = (3200 + np.arange(t.size)).astype(float)
    out = []
    for draw in (_legacy_tall, _helper_tall):
        with mpl.rc_context(STYLE):
            fig = _fig(figsize=(3.6, 4.4))
            draw(fig, lams, t, RES, LIM, symlog, ZC if zc else None, dumps, hours)
            out.append(_render(fig))
    assert out[0].shape == out[1].shape
    assert np.array_equal(out[0], out[1]), "{} pixels differ".format(int(np.any(out[0] != out[1], axis=-1).sum()))


def _legacy_spectrum(fig, f, P, fnyq, eigen):
    """Transcription of fig_disc_zerocross_spectrum.py (one panel)."""
    ax = fig.add_subplot()
    keep = (f >= 1.0) & (f <= fnyq)
    ax.loglog(f[keep], P[keep], color="k", lw=0.4, rasterized=True)
    ax.set_xlim(1.0, fnyq)
    for fe in eigen:
        ax.axvline(fe, color=ORANGE, ls="--", lw=0.6, zorder=0)
    ax.set_title("He I")
    sec = ax.secondary_xaxis("top", functions=(lambda x: x * 86400e-6, lambda x: x / 86400e-6))
    sec.set_xlabel(r"frequency (d$^{-1}$)", fontsize=7)
    ax.text(0.03, 0.06, "los1", transform=ax.transAxes, fontsize=6.5)


def _helper_spectrum(fig, f, P, fnyq, eigen):
    ax = fig.add_subplot()
    pl.plot_power_spectrum(ax, f, P, markers_muhz=eigen, marker_kw=dict(color=ORANGE), fmin=1.0, fmax=fnyq)
    ax.set_title("He I")
    pl.add_cpd_axis(ax, fontsize=7)
    pl.panel_text(ax, "los1", y=0.06, va="baseline", fontsize=6.5)


def test_power_spectrum_pixel_identical():
    rng = np.random.default_rng(2)
    dt = 2834.5
    x = rng.standard_normal(1601) * np.hanning(1601)
    xf = np.pad(x, (5000, 5000), "mean")
    f = np.fft.fftfreq(xf.size, dt)
    pos = f > 0
    f, P = f[pos] * 1e6, np.abs(np.fft.fft(xf)[pos]) ** 2
    eigen = np.array([10.95588503, 13.52081278, 85.22377711, 99.36265539, 128.59128465])
    out = []
    for draw in (_legacy_spectrum, _helper_spectrum):
        with mpl.rc_context(STYLE):
            fig = _fig(figsize=(2.6, 2.4))
            draw(fig, f, P, 0.5e6 / dt, eigen)
            out.append(_render(fig))
    assert np.array_equal(out[0], out[1])


def test_timeseries_pixel_identical():
    rng = np.random.default_rng(4)
    t = np.arange(300) * 2834.5 / 86400.0
    dg = rng.standard_normal((300, 8, 5)).cumsum(axis=0) * 1e-3 + 1.0
    out = []
    for helper in (False, True):
        with mpl.rc_context(STYLE):
            fig = _fig(figsize=(2.6, 2.4))
            ax = fig.add_subplot()
            q, scale = 0, 1e3
            if helper:
                pl.plot_highlighted_series(ax, t, dg[:, :, q] * scale, offset=dg[:, :, q].mean() * scale, label="los1",
                                           other_label="los2-8")
                pl.expand_ylim(ax, 0.45)
                pl.panel_text(ax, "EW text\nsecond line", y=0.96, fontsize=6)
            else:
                for k in list(range(1, 8)) + [0]:
                    y = dg[:, k, q] * scale
                    y = y - dg[:, :, q].mean() * scale
                    ax.plot(t, y, color="k" if k == 0 else GREY, lw=0.5 if k == 0 else 0.3, alpha=1.0 if k == 0 else 0.5,
                            label="los1" if k == 0 else ("los2-8" if k == 1 else None))
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo, hi + 0.45 * (hi - lo))
                ax.text(0.03, 0.96, "EW text\nsecond line", transform=ax.transAxes, fontsize=6, va="top")
            fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2)
            out.append(_render(fig))
    assert np.array_equal(out[0], out[1])


# --------------------------------------------------------------------------- M424 regression
def _m424_window(name="imu", los=1, d0=3200, d1=3260, vwin=400.0):
    p = m424_path("disc", "{}_timeseries.npz".format(name))
    z = np.load(p)
    dumps, t_s, Y, LREF = z["dumps"], z["t_s"], z["Y"], z["LREF"]
    isel = np.where((dumps >= d0) & (dumps <= d1))[0]
    m = np.abs(Y) <= vwin
    F = sio.npz_member_memmap(p, "F")
    F0 = sio.npz_member_memmap(p, "F0")
    k = los - 1
    FF = [np.asarray(F[isel, k, j])[:, m].astype(np.float64) for j in range(3)]
    FF0 = [np.asarray(F0[isel, k, j])[:, m].astype(np.float64) for j in range(3)]
    lam = [LREF[j] * np.exp(Y / 299792.458)[m] for j in range(3)]
    return dumps[isel], t_s[isel], lam, FF, FF0


@pytest.mark.m424
def test_m424_dynspec_pixel_identical():
    dumps, t_s, lam, FF, FF0 = _m424_window()
    t, unit = pl.time_since_start(t_s)
    assert unit == "h"
    np.testing.assert_array_equal(t, (t_s - t_s[0]) / 3600.0)
    R = [f - f.mean(axis=0) for f in FF]
    R0 = [f - f.mean(axis=0) for f in FF0]
    LIM = max(np.percentile(np.abs(r), 99.5) for r in R) * 1e3                  # fig_disc_dynspec_slide.py:37
    assert pl.percentile_limit(R, scale=1e3) == LIM
    assert 1e-1 < LIM < 10                                                     # residuals ~1e-4..3e-3 -> 0.1..10 x 1e-3
    zc = np.load(m424_path("disc", "zerocross_imu_los1.npz"))
    i = np.searchsorted(zc["dumps"], dumps)
    assert np.array_equal(zc["dumps"][i], dumps)
    _compare_dynspec(lam[0], t, FF[0], R[0], R0[0], LIM, 1.0, ZC=zc["A"][i, 0], dumps=dumps.astype(float))


@pytest.mark.m424
def test_m424_timeseries_values():
    p = m424_path("disc", "imu_timeseries.npz")
    z = np.load(p)
    dg = z["diag_F"]
    t, unit = pl.time_since_start(z["t_s"], unit="d")
    np.testing.assert_array_equal(t, (z["t_s"] - z["t_s"][0]) / 86400.0)
    fig = _fig()
    lines = pl.plot_highlighted_series(fig.add_subplot(), t, dg[:, :, 0, 0] * 1e3, offset=dg[:, :, 0, 0].mean() * 1e3)
    np.testing.assert_array_equal(lines[-1].get_ydata(), dg[:, 0, 0, 0] * 1e3 - dg[:, :, 0, 0].mean() * 1e3)
