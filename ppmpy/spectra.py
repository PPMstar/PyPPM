"""
Line-of-sight `lums` spectra: disk-integrated light-curve power spectra.

PP 2026-06-15: promoted into ppmpy from recreate_lums_mpi.py
(H-core-M25/analysis_fullstar-2). PP 2026-09-30: moved out of ppm.py into this
separate module; the former methods ``RprofSet.lums_spectra``,
``MomsDataSet.compute_lums_spectra`` and ``MomsDataSet.compare_lums_with_rprof``
are now the functions :func:`lums_spectra_rprof`, :func:`lums_spectra_moms` and
:func:`compare_lums_with_rprof`, which take the RprofSet / MomsDataSet as their
first argument. The numerics are unchanged.

Unlike the ell-resolved visible k-omega diagram, the `lums` spectrum
hemisphere-integrates the surface flux along a line of sight into ONE scalar
per dump (a synthetic light curve), then takes a temporal power spectrum
P(nu) -- the directly-observable integrated-light variability, and the
post-processing analogue of the PPMstar Fortran rprof lum1..lum8 columns
(PPM2F-...-O2.F). The 1-D pipeline :func:`lums_temporal_spectrum` is shared by
the moms-derived side (:func:`lums_spectra_moms`) and the rprof side
(:func:`lums_spectra_rprof`), so the two are directly comparable.

Example
-------
>>> from ppmpy import ppm, spectra
>>> rp = ppm.RprofSet('/path/to/prfs/')
>>> res = spectra.lums_spectra_rprof(rp, 3200, 4800, 4050., makefigure=True)
>>> moms = ppm.MomsDataSet('/path/to/moms/', 3200, rprofset=rp, var_list=[...])
>>> both = spectra.compare_lums_with_rprof(moms, 3200, 4800, 4050.)

The module imports only numpy and scipy at load time; ppm.py, matplotlib and
tqdm are imported when a function needs them.
"""
import contextlib
import os
import pickle

import numpy as np
import scipy.interpolate

__all__ = ['SIGMA_SB', 'LUMS_PPM2_PER_RELATIVE2', 'lums_temporal_spectrum',
           'precompute_los_kernels', 'hemisphere_integrate_grid',
           'plot_lums_spectra', 'plot_lums_comparison', 'lums_spectra_rprof',
           'lums_spectra_moms', 'compare_lums_with_rprof']

# Stefan-Boltzmann constant (SI). Any constant prefactor cancels under the
# divisive detrend, so its exact value does not affect the spectrum.
SIGMA_SB = 5.670367e-8

# The lums pipeline detrends to a *relative* (dimensionless) fluctuation, so its
# power spectrum is in relative^2 / microHz. Since 1 (relative) = 1e6 ppm, the
# power in ppm^2 / microHz is the relative^2 spectrum times (1e6)^2 = 1e12.
# (This is the LUMS_LM_SCALE = 1/1e-12 used by the old plot scripts.)
LUMS_PPM2_PER_RELATIVE2 = 1e12


def _ppm():
    """ppmpy.ppm, imported on first use (it pulls in nugridpy, pyshtools, ...)."""
    from . import ppm
    return ppm


def _tqdm(iterable, **kwargs):
    """tqdm progress bar if tqdm is installed, else the iterable itself."""
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


def _lums_is_relative(varname, detrend_mode):
    """True when the lums-detrended series is a relative (dimensionless)
    fluctuation, i.e. the ppm^2 unit conversion applies. The divisive detrend
    (``F/trend-1``) always yields a relative fluctuation; 'rel_lum-1' is already
    relative for either detrend mode."""
    return detrend_mode == 'divisive' or varname == 'rel_lum-1'


def lums_temporal_spectrum(L_time, dt, pad=10_000_000, detrend_order=3,
                           detrend_mode='divisive'):
    """
    Temporal power spectrum of a single 1-D luminosity time series, matching
    the rprof `lums` pipeline (get_temporal_spectrum in
    Temporal_spectra-threaded.ipynb): polynomial detrend, Hann window,
    mean-pad, FFT, energy-conserving power normalization.

    Parameters
    ----------
    L_time: array-like
        Luminosity (or flux) time series, one value per dump.
    dt: float
        Time between dumps in seconds.
    pad: int
        Mean-pad length applied symmetrically before the FFT (frequency
        interpolation). Use 0/None to disable.
    detrend_order: int
        Order of the polynomial trend removed before windowing.
    detrend_mode: 'divisive' or 'subtractive'
        'divisive': ``things / trend - 1`` (rprof verbatim; requires
        positive-valued input, e.g. varname='abs_lum').
        'subtractive': ``things - trend`` (for zero-mean inputs, e.g.
        varname='rel_lum-1').

    Returns
    -------
    freq_muHz: np.ndarray
        Positive temporal frequencies in microHz.
    power: np.ndarray
        Power spectral density at those frequencies.
    L_detrend: np.ndarray
        The detrended (pre-window) time series, for inspection.
    """
    L_time = np.asarray(L_time, dtype=np.float64)
    N_1 = len(L_time)
    times = dt * np.arange(N_1)

    coefs = np.polyfit(times, L_time, detrend_order)
    trend = np.polyval(coefs, times)
    if detrend_mode == 'divisive':
        L_detrend = L_time / trend - 1.0
    elif detrend_mode == 'subtractive':
        L_detrend = L_time - trend
    else:
        raise ValueError("unknown detrend_mode: {!r}".format(detrend_mode))

    L_win = np.hanning(N_1) * L_detrend

    if pad is not None and pad > 0:
        L_fft = np.pad(L_win, (pad // 2, pad // 2), 'mean')
    else:
        L_fft = L_win

    dft = np.fft.fft(L_fft)
    freqs = np.fft.fftfreq(len(L_fft), dt)
    pos = freqs > 0
    freq_muHz = freqs[pos] * 1e6

    sampling_factor = 1e-6 * dt / N_1
    window_power_factor = np.sqrt(8.0 / 3.0)
    power = window_power_factor * sampling_factor * np.abs(dft[pos])**2

    return freq_muHz, power, L_detrend


def precompute_los_kernels(theta_grid, phi_grid, los_list):
    """
    Precompute, per line of sight, the Lambert kernel times the solid-angle
    weight on a Driscoll-Healy (theta, phi) grid:

        K_k(theta, phi) = max(0, r_hat . n_hat_k) * sin(theta) * dtheta * dphi

    so that the hemisphere integral of a surface field is a plain weighted sum
    (see :func:`hemisphere_integrate_grid`). Mirrors the cell-by-cell visible
    accumulation in PPM2F-...-O2.F.

    Parameters
    ----------
    theta_grid, phi_grid: np.ndarray
        1-D colatitude / longitude grids (as returned by
        ``sphericalHarmonics_format(..., get_theta_phi_grids=True)``).
    los_list: list of array-like
        Line-of-sight vectors (need not be normalized).

    Returns
    -------
    np.ndarray
        Kernels of shape (n_los, N_theta, N_phi).
    """
    N_theta = len(theta_grid)
    N_phi = len(phi_grid)
    dtheta = float(np.pi) / N_theta
    dphi = 2.0 * float(np.pi) / N_phi
    sin_theta = np.sin(theta_grid)
    cos_theta = np.cos(theta_grid)
    sin_phi = np.sin(phi_grid)
    cos_phi = np.cos(phi_grid)

    rx = sin_theta[:, None] * cos_phi[None, :]
    ry = sin_theta[:, None] * sin_phi[None, :]
    rz = np.broadcast_to(cos_theta[:, None], (N_theta, N_phi))
    weight = (sin_theta * dtheta * dphi)[:, None]

    kernels = np.empty((len(los_list), N_theta, N_phi), dtype=np.float64)
    for k, los in enumerate(los_list):
        n = np.asarray(los, dtype=np.float64)
        n /= np.linalg.norm(n)
        cos_gamma = rx * n[0] + ry * n[1] + rz * n[2]
        kernels[k] = np.maximum(0.0, cos_gamma) * weight
    return kernels


def hemisphere_integrate_grid(quantity_grid, kernels):
    """
    Hemisphere-integrate a surface field for each line of sight, given the
    precomputed kernels from :func:`precompute_los_kernels`.

    Parameters
    ----------
    quantity_grid: np.ndarray
        Surface field of shape (N_theta, N_phi).
    kernels: np.ndarray
        Per-LOS kernels of shape (n_los, N_theta, N_phi).

    Returns
    -------
    np.ndarray
        One hemisphere integral per LOS, shape (n_los,).
    """
    return np.einsum('ij,kij->k', quantity_grid, kernels)


def _lums_surface_quantity(T9, varname):
    """Build the surface field hemisphere-integrated for a `lums` spectrum."""
    if varname == 'abs_lum':
        # sigma * T^4 with T = T9 * 1e9 K (W/m^2). Prefactor cancels in detrend.
        return SIGMA_SB * (T9 * 1e9)**4
    elif varname == 'T9^4':
        return T9**4
    elif varname == 'rel_lum-1':
        L_base = float(np.mean(T9**4))
        return (T9**4) / L_base - 1.0
    else:
        raise ValueError(
            "lums integrator: varname={!r} not supported "
            "(use 'abs_lum', 'T9^4', or 'rel_lum-1')".format(varname))


def plot_lums_spectra(freq_muHz, power_per_los, power_mean, run_id='', varname='',
                      radius=None, numin=1.0, numax=180.0, outpath=None, ifig=1,
                      color='viridis', to_ppm=True, ylims=None, logx=False):
    """
    Plot the line-of-sight `lums` power spectra: the individual LOS curves
    (faint) plus their mean.

    Parameters
    ----------
    freq_muHz: np.ndarray
        Positive frequencies in microHz (x axis).
    power_per_los: np.ndarray
        Power spectra in relative^2/microHz, shape (n_los, n_freq).
    power_mean: np.ndarray
        LOS-mean power spectrum, shape (n_freq,).
    run_id, varname: str
        Labels for the title.
    radius: float, optional
        Radius (Mm) for the title.
    numin, numax: float
        Frequency window (microHz).
    outpath: str, optional
        If given, save the figure here.
    ifig: int, optional
        Figure number; the figure is closed and recreated under this number on
        each call (default 1), so re-running a cell reuses one clean figure.
    color: str
        Matplotlib colormap name for the per-LOS curves.
    to_ppm: bool
        If True (default) plot the power in ppm^2/microHz (multiply the
        relative^2 spectrum by 1e12); otherwise plot relative^2/microHz.
    ylims: tuple, optional
        (ymin, ymax) for the log y-axis. If None (default) matplotlib autoscales.
    logx: bool
        If True, use a log-scaled frequency (x) axis; default False (linear).
    """
    import matplotlib.pyplot as pl
    scale = LUMS_PPM2_PER_RELATIVE2 if to_ppm else 1.0
    unit = r'\mathrm{ppm}^2/\mu\mathrm{Hz}' if to_ppm else r'\mathcal{L}^2/\mu\mathrm{Hz}'
    pl.close(ifig)
    fig = pl.figure(ifig, figsize=(8, 4.5))
    ax = fig.gca()
    n_los = power_per_los.shape[0]
    cmap = pl.get_cmap(color)
    for k in range(n_los):
        ax.semilogy(freq_muHz, power_per_los[k] * scale, color=cmap(k / max(1, n_los - 1)),
                    lw=0.3, alpha=0.55)
    ax.semilogy(freq_muHz, power_mean * scale, color='k', lw=1.0, zorder=5,
                label=r'$\langle\mathrm{LOS}\rangle$')
    if logx:
        ax.set_xscale('log')
    ax.set_xlim(numin, numax)
    if ylims is not None:
        ax.set_ylim(*ylims)
    ax.set_xlabel(r'$\nu$  ($\mu$Hz)')
    ax.set_ylabel(r'$P_{{\mathrm{{lums}}}}(\nu)$  $({})$'.format(unit))
    rad_str = '' if radius is None else ' @ {:.0f} Mm'.format(radius)
    ax.set_title(r'LOS-specific $P_{{\mathrm{{lums}}}}(\nu)$ — {} — {}{}'.format(
        run_id, varname, rad_str), fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    if outpath is not None:
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
    # No `return fig`: a bare call in a notebook would otherwise echo the figure
    # a second time (on top of the backend's own display). Matches the rest of
    # ppmpy's plotting helpers.


def plot_lums_comparison(moms_result, rprof_result, run_id='', numin=1.0, numax=180.0,
                         outpath=None, ifig=2, per_los_panels=False, to_ppm=True,
                         ylims=None, logx=False):
    """
    Overlay the moms-derived `lums` spectra against the native rprof
    ``lum1..lum8`` spectra for comparison. Both sides use the identical
    temporal pipeline, so no scale factor is applied between them.

    Parameters
    ----------
    moms_result: dict
        Output of :func:`lums_spectra_moms`.
    rprof_result: dict
        Output of :func:`lums_spectra_rprof`.
    run_id: str
        Label for the title.
    numin, numax: float
        Frequency window (microHz).
    outpath: str, optional
        If given, save the figure here.
    ifig: int, optional
        Figure number; the figure is closed and recreated under this number on
        each call (default 2), so re-running a cell reuses one clean figure.
    per_los_panels: bool
        If True, draw a 2x4 grid pairing moms LOS k with rprof ``lum_{k+1}``
        instead of a single overlay panel.
    to_ppm: bool
        If True (default) plot power in ppm^2/microHz (relative^2 spectrum times
        1e12); otherwise plot relative^2/microHz.
    ylims: tuple, optional
        (ymin, ymax) applied to every panel's log y-axis. If None (default)
        matplotlib autoscales.
    logx: bool
        If True, use a log-scaled frequency (x) axis; default False (linear).

    Notes
    -----
    With the Fortran LOS ordering (``los_convention='fortran'``) the moms LOS
    index ``k`` corresponds to rprof ``lum_{k+1}``.
    """
    import matplotlib.pyplot as pl
    from matplotlib.lines import Line2D
    s = LUMS_PPM2_PER_RELATIVE2 if to_ppm else 1.0
    unit = r'\mathrm{ppm}^2/\mu\mathrm{Hz}' if to_ppm else r'\mathcal{L}^2/\mu\mathrm{Hz}'
    fm, Pm, Pm_mean = (moms_result['freq_muHz'], moms_result['power_per_los'] * s,
                       moms_result['power_mean'] * s)
    fr, Pr, Pr_mean = (rprof_result['freq_muHz'], rprof_result['power_per_los'] * s,
                       rprof_result['power_mean'] * s)
    n_los = min(Pm.shape[0], Pr.shape[0])
    radius = moms_result.get('radius')
    rad_str = '' if radius is None else ' @ {:.0f} Mm'.format(radius)

    if per_los_panels:
        pl.close(ifig)
        fig = pl.figure(ifig, figsize=(20, 9))
        axes = fig.subplots(2, 4, sharex=True, sharey=True)
        axes = np.atleast_1d(axes).flatten()
        for k in range(n_los):
            ax = axes[k]
            ax.semilogy(fm, Pm[k], color='tab:blue', lw=0.8, label='moms')
            ax.semilogy(fr, Pr[k], color='crimson', lw=0.8, ls='--', label='rprof lum')
            ax.set_xlim(numin, numax)
            ax.set_title('LOS {} / lum{}'.format(k + 1, k + 1), fontsize=9)
            if k == 0:
                ax.legend(fontsize=8)
        if ylims is not None:
            for ax in axes:
                ax.set_ylim(*ylims)
        if logx:
            for ax in axes:
                ax.set_xscale('log')
        fig.suptitle(r'moms vs rprof $P_{{\mathrm{{lums}}}}(\nu)$ — {}{}'.format(run_id, rad_str),
                     fontsize=12)
        fig.supxlabel(r'$\nu$ ($\mu$Hz)')
        fig.supylabel(r'$P_{{\mathrm{{lums}}}}(\nu)$  $({})$'.format(unit))
        fig.tight_layout()
        if outpath is not None:
            fig.savefig(outpath, dpi=150, bbox_inches='tight')
        return            # no `return fig` — see note below

    pl.close(ifig)
    fig = pl.figure(ifig, figsize=(9, 5))
    ax = fig.gca()
    # moms: blue family; rprof: orange/crimson family.
    for k in range(n_los):
        ax.semilogy(fm, Pm[k], color='#6baed6', lw=0.3, alpha=0.5)
        ax.semilogy(fr, Pr[k], color='#fd8d3c', lw=0.3, alpha=0.5)
    ax.semilogy(fm, Pm_mean, color='#08519c', lw=1.4, zorder=6, label='moms  $\\langle$LOS$\\rangle$')
    ax.semilogy(fr, Pr_mean, color='#a63603', lw=1.4, zorder=6, ls='--',
                label='rprof lum1..8  $\\langle$LOS$\\rangle$')
    if logx:
        ax.set_xscale('log')
    ax.set_xlim(numin, numax)
    if ylims is not None:
        ax.set_ylim(*ylims)
    ax.set_xlabel(r'$\nu$  ($\mu$Hz)')
    ax.set_ylabel(r'$P_{{\mathrm{{lums}}}}(\nu)$  $({})$'.format(unit))
    ax.set_title(r'moms-derived vs rprof $P_{{\mathrm{{lums}}}}(\nu)$ — {}{}'.format(run_id, rad_str),
                 fontsize=11)
    # Faint-curve legend proxies.
    handles = [Line2D([0], [0], color='#6baed6', lw=0.8),
               Line2D([0], [0], color='#fd8d3c', lw=0.8),
               Line2D([0], [0], color='#08519c', lw=1.4),
               Line2D([0], [0], color='#a63603', lw=1.4, ls='--')]
    labels = ['moms LOS 1..8', 'rprof lum1..8',
              r'moms $\langle$LOS$\rangle$', r'rprof $\langle$LOS$\rangle$']
    ax.legend(handles, labels, loc='upper right', fontsize=9)
    fig.tight_layout()
    if outpath is not None:
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
    # No `return fig`: avoids the duplicate figure echo in notebooks (see the
    # per-panels branch above and the other ppmpy plot_* helpers).


def lums_spectra_rprof(rprofset, dump_start, dump_stop, radius, lum_vars=None,
                       detrend_order=3, detrend_mode='divisive', pad=10_000_000,
                       makefigure=False, returnvalues=True, run_id=None,
                       numin=1.0, numax=180.0, verbose=3):
    '''
    Temporal power spectra of the in-code rprof line-of-sight luminosities
    ``lum1..lum8`` at a fixed radius.

    These are the PPMstar Fortran's runtime disk-integrated luminosities
    (one scalar per dump per LOS). This reads each ``lum_k`` at ``radius``
    across the dump range and applies the shared
    :func:`lums_temporal_spectrum` pipeline, returning the same dict shape
    as :func:`lums_spectra_moms` so the two can be overlaid directly (see
    :func:`plot_lums_comparison`). Formerly ``RprofSet.lums_spectra``.

    Parameters
    ----------
    rprofset: ppm.RprofSet
        The run's rprof set.
    dump_start, dump_stop: integer
        Inclusive dump range.
    radius: float
        Radius in Mm at which to sample ``lum1..lum8``.
    lum_vars: list of str, optional
        Variable names to read; default ``['lum1', ..., 'lum8']``.
    detrend_order: integer, optional
        Polynomial detrend order (default 3).
    detrend_mode: {'divisive', 'subtractive'}, optional
        Detrend mode (default 'divisive', matching the rprof pipeline;
        ``lum`` values are absolute fluxes so divisive is appropriate).
    pad: integer, optional
        Mean-pad length before the FFT (default 1e7).
    makefigure: boolean, optional
        If True, draw the 8-LOS + mean spectrum.
    returnvalues: boolean, optional
        If True, return the results dict.
    run_id: str, optional
        Label for the figure; defaults to the run id of ``rprofset``.
    numin, numax: float, optional
        Frequency window (microHz) for the figure.
    verbose: integer, optional
        Verbosity of the warnings about skipped dumps (ppm.Messenger levels).

    Returns
    -------
    dict or None
        Same shape as :func:`lums_spectra_moms`, with 'source' == 'rprof'
        and 'lum_vars' added.
    '''
    ppm = _ppm()
    messenger = ppm.Messenger(verbose=verbose)
    if lum_vars is None:
        lum_vars = ['lum{:d}'.format(k) for k in range(1, 9)]
    n_los = len(lum_vars)

    all_dumps = np.arange(dump_start, dump_stop + 1)
    L_time = np.full((len(all_dumps), n_los), np.nan, dtype=np.float64)
    used_dumps = []
    for i, dump in enumerate(_tqdm(all_dumps, desc='rprof lums', ncols=80)):
        try:
            R = rprofset.get('R', fname=int(dump))
            idx = ppm.index_nearest_value(R, radius)
            for k, lv in enumerate(lum_vars):
                L_time[i, k] = rprofset.get(lv, fname=int(dump))[idx]
            used_dumps.append(int(dump))
        except Exception:
            messenger.warning('Skipping rprof dump {}'.format(dump))
            continue

    # Keep only successfully-read dumps (contiguous expected; this also
    # drops any trailing NaN rows if a dump was missing).
    good = ~np.isnan(L_time).any(axis=1)
    L_time = L_time[good]

    history = rprofset.get_history()
    dump_to_time = dict(zip(history.get('NDump'), history.get('time(mins)')))
    dump_times = np.array([dump_to_time[d] for d in used_dumps if d in dump_to_time])
    dt_s = float(np.median(np.diff(dump_times)) * 60)

    power_per_los = None
    freq_muHz = None
    for k in range(n_los):
        f_muHz, P, _ = lums_temporal_spectrum(
            L_time[:, k], dt=dt_s, pad=pad,
            detrend_order=detrend_order, detrend_mode=detrend_mode)
        if power_per_los is None:
            freq_muHz = f_muHz
            power_per_los = np.zeros((n_los, len(f_muHz)), dtype=np.float64)
        power_per_los[k] = P
    power_mean = power_per_los.mean(axis=0)

    if run_id is None:
        run_id = rprofset.get_run_id() or ''

    result = {'freq_muHz': freq_muHz, 'power_per_los': power_per_los,
              'power_mean': power_mean, 'L_vis_time': L_time,
              'los_list': lum_vars, 'radius': float(radius), 'varname': 'lum',
              'detrend_order': detrend_order, 'detrend_mode': detrend_mode,
              'pad': pad, 'dt_s': dt_s, 'used_dumps': used_dumps,
              'n_dumps': len(used_dumps), 'run_id': run_id,
              'lum_vars': lum_vars, 'source': 'rprof'}

    if makefigure:
        plot_lums_spectra(freq_muHz, power_per_los, power_mean,
                          run_id=run_id, varname='rprof lum1..8',
                          radius=float(radius), numin=numin, numax=numax,
                          to_ppm=_lums_is_relative('lum', detrend_mode))

    if returnvalues:
        return result
    return None


def lums_spectra_moms(moms, dump_start, dump_stop, varname='abs_lum',
                      lmax_crop=None, radius=None, mass=None,
                      los_list=None, los_convention='fortran',
                      detrend_order=3, detrend_mode='divisive', pad=10_000_000,
                      n_threads=1, per_thread_moms=False,
                      use_mpi=False, mapping='rr',
                      makefigure=True, returnvalues=True,
                      save=False, outdir=None, run_id=None,
                      numin=1.0, numax=180.0):
    """
    Compute the line-of-sight `lums` power spectra from the 3D moms data:
    for each line of sight, hemisphere-integrate the surface flux into one
    scalar per dump (a synthetic light curve), then take a temporal power
    spectrum P(nu). Formerly ``MomsDataSet.compute_lums_spectra``.

    This is the moms-derived, post-processing analogue of the PPMstar
    Fortran rprof ``lum1..lum8`` columns; pair it with
    :func:`lums_spectra_rprof` (or :func:`compare_lums_with_rprof`) to
    compare against the in-code values. The surface field is sampled on a
    Driscoll-Healy grid and integrated with a Lambert ``max(0, r_hat.n_hat)``
    kernel (grid integration, mirroring O2.F); the 1-D time series then goes
    through the shared :func:`lums_temporal_spectrum` pipeline.

    Like :meth:`ppm.MomsDataSet.visible_k_omega_diagram`, Phase 1 runs
    serially, threaded (``n_threads``), or under MPI (``use_mpi``), with
    identical results.

    Parameters
    ----------
    moms: ppm.MomsDataSet
        The moms data set (with an rprofset, which provides the dump times).
    dump_start, dump_stop: int
        Inclusive dump range (the time series).
    varname: str, optional
        Surface field hemisphere-integrated per dump. Default 'abs_lum'
        (``sigma*(T9*1e9)^4``, matches the Fortran flux). Also 'T9^4' and
        'rel_lum-1' (``T9^4/<T9^4>-1``).
    lmax_crop: int, optional
        Cap on the spherical-harmonics grid resolution; None uses the grid
        Nyquist (:meth:`ppm.MomsDataSet.sphericalHarmonics_lmax`). Sets the
        (theta, phi) sampling density only (no SH expansion is done here).
    radius: float, optional
        Target radius in Mm (mutually exclusive with ``mass``).
    mass: float, optional
        Target enclosed mass in Msun (requires an rprofset).
    los_list: list of array-like, optional
        Custom lines of sight; None uses the 8 defaults.
    los_convention: {'fortran', 'cross'}, optional
        Default LOS set. 'fortran' (default) aligns LOS k with rprof
        ``lum_{k+1}``; 'cross' uses the visible-komega convention.
    detrend_order: int, optional
        Polynomial detrend order (default 3).
    detrend_mode: {'divisive', 'subtractive'}, optional
        'divisive' (default, ``F/trend-1``) for positive fields like
        'abs_lum'; 'subtractive' for zero-mean fields like 'rel_lum-1'.
    pad: int, optional
        Mean-pad length before the FFT (default 1e7).
    n_threads, per_thread_moms, use_mpi, mapping:
        Parallel backend controls (see
        :meth:`ppm.MomsDataSet.visible_k_omega_diagram`).
    makefigure: bool, optional
        If True, draw the 8-LOS + mean spectrum (:func:`plot_lums_spectra`).
    returnvalues: bool, optional
        If True, return the results dict (see Returns).
    save: bool, optional
        If True (and ``outdir`` set), write per-LOS .npz and a bundle .pickle.
    outdir, run_id: str, optional
        On-disk output directory and run label.
    numin, numax: float, optional
        Frequency window (microHz) for the figure.

    Returns
    -------
    dict or None
        With keys: 'freq_muHz', 'power_per_los' ((n_los, n_freq)),
        'power_mean', 'L_vis_time' ((n_dumps, n_los)), 'los_list', 'radius',
        'varname', 'detrend_order', 'detrend_mode', 'pad', 'dt_s',
        'used_dumps', 'n_dumps'. None on non-root MPI ranks.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    ppm = _ppm()

    # ---- MPI setup (lazy import so notebooks need no mpi4py) ----
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm, rank, size = None, 0, 1
    is_root = (rank == 0)

    radiusinput = radius
    massinput = mass

    # ---- input validation ----
    if (radius is None and mass is None) or (radius is not None and mass is not None):
        moms._messenger.error('You must select either a radius or a mass where to calculate the spectrum')
        return None
    if massinput is not None and not isinstance(moms._rprofset, ppm.RprofSet):
        moms._messenger.error('Mass coordinate requires rprofset for mass-to-radius conversion, '
                              'but no rprofset is available')
        return None

    def _radius_for(dump_number):
        if radiusinput is not None:
            return float(radiusinput)
        r = moms._rprofset.get('R', fname=dump_number)
        m = moms._rprofset.compute_m(dump_number) * 5.025e-07  # code units -> Msun
        return float(scipy.interpolate.interp1d(m, r, fill_value="extrapolate")(massinput))

    rep_radius = _radius_for(dump_start)
    lmax, N, npoints = moms.sphericalHarmonics_lmax(rep_radius)
    if lmax_crop is not None:
        lmax = int(lmax_crop)

    # ---- lines of sight ----
    if los_list is None:
        los_list = (ppm.make_los_vectors_fortran() if los_convention == 'fortran'
                    else ppm.make_los_vectors())
    los_list = [np.asarray(l, dtype=float) for l in los_list]
    n_los = len(los_list)

    # ---- dump layout ----
    all_dumps = np.arange(dump_start, dump_stop + 1)
    n_dumps = len(all_dumps)
    if use_mpi:
        if mapping == 'chunk':
            my_indices = np.array_split(np.arange(n_dumps), size)[rank]
        else:
            my_indices = np.where((np.arange(n_dumps) % size) == rank)[0]
    else:
        my_indices = np.arange(n_dumps)
    my_dumps = all_dumps[my_indices]

    # ---- precompute the (theta, phi) grid + per-LOS Lambert kernels once ----
    # A throwaway grid sample gives the DH theta/phi axes for this lmax.
    _, theta_grid, phi_grid = moms.sphericalHarmonics_format(
        'T9', rep_radius, int(my_dumps[0]) if len(my_dumps) else int(dump_start),
        lmax=lmax, get_theta_phi_grids=True)
    kernels = precompute_los_kernels(theta_grid, phi_grid, los_list)

    # ---- moms factory for the data reads ----
    init_dump = int(my_dumps[0]) if len(my_dumps) else int(dump_start)
    if per_thread_moms:
        tls = threading.local()

        def get_moms():
            if not hasattr(tls, 'moms'):
                tls.moms = ppm.MomsDataSet(moms._dir_name, init_dump_read=init_dump,
                                           dumps_in_mem=1, var_list=moms._var_list,
                                           rprofset=moms._rprofset, verbose=0)
            return tls.moms
        io_lock = None
    else:
        def get_moms():
            return moms
        io_lock = threading.Lock()

    # ---- per-dump task: sample T9 on the grid, hemisphere-integrate ----
    def _task(global_i, dump_number):
        rad = _radius_for(dump_number)
        m = get_moms()
        lock = io_lock if io_lock is not None else contextlib.nullcontext()
        with lock:
            T9 = m.sphericalHarmonics_format('T9', rad, int(dump_number), lmax=lmax)
        quantity = _lums_surface_quantity(T9, varname)
        vals = hemisphere_integrate_grid(quantity, kernels)
        return int(global_i), int(dump_number), vals

    # Full-length buffer so an MPI SUM reconstructs the full time axis.
    my_L_vis = np.zeros((n_dumps, n_los), dtype=np.float64)
    used_dumps_local = []

    # ---- Phase 1: hemisphere integration across dumps ----
    if n_threads and n_threads > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futures = [ex.submit(_task, int(gi), int(dn))
                       for gi, dn in zip(my_indices, my_dumps)]
            completed = as_completed(futures)
            if is_root:
                completed = _tqdm(completed, total=len(futures),
                                  desc='lums phase 1', ncols=80)
            for fut in completed:
                try:
                    gi, dn, vals = fut.result()
                    my_L_vis[gi] = vals
                    used_dumps_local.append(dn)
                except Exception as exc:
                    print('Skipping dump (thread):', exc)
    else:
        pairs = list(zip(my_indices, my_dumps))
        if is_root:
            pairs = _tqdm(pairs, desc='lums phase 1', ncols=80)
        for gi, dn in pairs:
            try:
                gi, dn, vals = _task(int(gi), int(dn))
                my_L_vis[gi] = vals
                used_dumps_local.append(dn)
            except KeyboardInterrupt:
                print('Stopping early at dump', dn)
                break
            except Exception:
                print('Skipping dump', dn)
                continue

    # ---- reduce across ranks (or pass through) ----
    if use_mpi:
        all_used_lists = comm.gather(used_dumps_local, root=0)
        comm.Barrier()
        global_L_vis = np.zeros_like(my_L_vis) if is_root else None
        comm.Reduce(my_L_vis, global_L_vis, op=MPI.SUM, root=0)
    else:
        all_used_lists = [used_dumps_local]
        global_L_vis = my_L_vis

    if use_mpi and not is_root:
        return None

    # ---- time step ----
    used_dumps = sorted({d for sub in all_used_lists for d in sub})
    if len(used_dumps) == 0:
        moms._messenger.error('No dumps were successfully processed.')
        return None
    history = moms._rprofset.get_history()
    dump_to_time = dict(zip(history.get('NDump'), history.get('time(mins)')))
    dump_times = np.array([dump_to_time[d] for d in used_dumps if d in dump_to_time])
    dt_s = float(np.median(np.diff(dump_times)) * 60)

    # ---- Phase 2: temporal spectrum per LOS ----
    power_per_los = None
    freq_muHz = None
    for k in range(n_los):
        f_muHz, P, _ = lums_temporal_spectrum(
            global_L_vis[:, k], dt=dt_s, pad=pad,
            detrend_order=detrend_order, detrend_mode=detrend_mode)
        if power_per_los is None:
            freq_muHz = f_muHz
            power_per_los = np.zeros((n_los, len(f_muHz)), dtype=np.float64)
        power_per_los[k] = P
    power_mean = power_per_los.mean(axis=0)

    if run_id is None:
        run_id = getattr(moms, '_run_id', '') or ''

    result = {'freq_muHz': freq_muHz, 'power_per_los': power_per_los,
              'power_mean': power_mean, 'L_vis_time': global_L_vis,
              'los_list': los_list, 'radius': rep_radius, 'varname': varname,
              'detrend_order': detrend_order, 'detrend_mode': detrend_mode,
              'pad': pad, 'dt_s': dt_s, 'used_dumps': used_dumps,
              'n_dumps': n_dumps, 'run_id': run_id, 'source': 'moms'}

    # ---- on-disk artifacts ----
    if save and outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        tag = '{}-{}-{:.0f}Mm'.format(run_id, varname, rep_radius)
        for k in range(n_los):
            np.savez_compressed(
                os.path.join(outdir, 'lums-{}-los{}.npz'.format(tag, k + 1)),
                freq_muHz=freq_muHz, power=power_per_los[k],
                L_vis_time=global_L_vis[:, k], los_vec=los_list[k],
                run_id=run_id, varname=varname, radius=rep_radius,
                n_dumps=n_dumps, dt_s=dt_s, pad=pad,
                detrend_order=detrend_order, detrend_mode=detrend_mode)
        with open(os.path.join(outdir, 'lums-{}.pickle'.format(tag)), 'wb') as f:
            pickle.dump(result, f)

    # ---- figure ----
    if makefigure:
        fig_out = (os.path.join(outdir, 'lums-{}-{}-{:.0f}Mm.png'.format(
            run_id, varname, rep_radius)) if (save and outdir is not None) else None)
        plot_lums_spectra(freq_muHz, power_per_los, power_mean,
                          run_id=run_id, varname=varname, radius=rep_radius,
                          numin=numin, numax=numax, outpath=fig_out,
                          to_ppm=_lums_is_relative(varname, detrend_mode))

    if returnvalues:
        return result
    return None


def compare_lums_with_rprof(moms, dump_start, dump_stop, radius,
                            varname='abs_lum', lmax_crop=None,
                            los_convention='fortran',
                            detrend_order=3, detrend_mode='divisive',
                            pad=10_000_000, n_threads=1, per_thread_moms=False,
                            use_mpi=False, mapping='rr',
                            makefigure=True, returnvalues=True,
                            outdir=None, run_id=None, numin=1.0, numax=180.0):
    """
    Compute the moms-derived `lums` spectra and the native rprof
    ``lum1..lum8`` spectra and overlay them for comparison. Formerly
    ``MomsDataSet.compare_lums_with_rprof``.

    Runs :func:`lums_spectra_moms` (moms) and :func:`lums_spectra_rprof`
    (rprof, via the rprofset of ``moms``) with the same pipeline settings,
    then draws :func:`plot_lums_comparison`.

    Parameters
    ----------
    moms: ppm.MomsDataSet
        The moms data set; it must have an rprofset.
    Other parameters:
        As :func:`lums_spectra_moms`.

    Returns
    -------
    dict or None
        ``{'moms': <moms result>, 'rprof': <rprof result>}`` when
        ``returnvalues`` is True; None on non-root MPI ranks.
    """
    ppm = _ppm()
    if not isinstance(moms._rprofset, ppm.RprofSet):
        moms._messenger.error('compare_lums_with_rprof requires this MomsDataSet '
                              'to have an rprofset (for the rprof lum1..8 side).')
        return None

    moms_res = lums_spectra_moms(
        moms, dump_start, dump_stop, varname=varname, lmax_crop=lmax_crop,
        radius=radius, los_convention=los_convention,
        detrend_order=detrend_order, detrend_mode=detrend_mode, pad=pad,
        n_threads=n_threads, per_thread_moms=per_thread_moms,
        use_mpi=use_mpi, mapping=mapping,
        makefigure=False, returnvalues=True, run_id=run_id)

    # Non-root MPI ranks are done (moms_res is None there).
    if moms_res is None:
        return None

    rprof_res = lums_spectra_rprof(
        moms._rprofset, dump_start, dump_stop, radius,
        detrend_order=detrend_order, detrend_mode=detrend_mode, pad=pad,
        makefigure=False, returnvalues=True, numin=numin, numax=numax)

    if run_id is None:
        run_id = getattr(moms, '_run_id', '') or ''
    if makefigure:
        out = None
        if outdir is not None:
            os.makedirs(outdir, exist_ok=True)
            out = os.path.join(outdir, 'lums-compare-{}-{}-{:.0f}Mm.png'.format(
                run_id, varname, moms_res['radius']))
        plot_lums_comparison(moms_res, rprof_res, run_id=run_id,
                             numin=numin, numax=numax, outpath=out,
                             to_ppm=_lums_is_relative(varname, detrend_mode))

    if returnvalues:
        return {'moms': moms_res, 'rprof': rprof_res}
    return None
