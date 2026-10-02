"""
Disc integration of local flux profiles (the flux method) and of emergent intensities (the intensity method):
disc-integrated, continuum-normalised line profiles of a star whose surface is sampled by an equal-area grid of
local models.

For an observer in direction n (unit vector, star -> observer) the disc-integrated profile is

    F(y) / F_c = sum_i w_i f_i(y_i') / sum_i w_i,      w_i = mu_i F_c,i   (mu_i = r_i . n > 0),

with f_i the continuum-normalised rest-frame flux profile of point i, F_c,i its continuum flux, and
y_i' the rest-frame velocity coordinate of point i that is seen at y: a line-of-sight velocity
v_i = u_i . n > 0 (towards the observer) gives the blueshift lambda_obs = lambda (1 - v_i / c), i.e. the
profile moves by -c ln(1 - v_i/c) on the grid y = c ln(lambda / lambda_ref). I(mu) = const (Lambert's
cosine law: the only angle dependence is the projected area mu dA; equal-area grid, so dA is common).

Three evaluations, all on a :class:`~ppmpy.synspec.spectral.VelocityGrid`:

* :class:`DiscFlux` (the all-dump method): the local profile of a point is interpolated linearly in
  T_eff' between library nodes (:func:`ppmpy.synspec.library.lib_nodes`) and shifted by its Doppler shift
  rounded to whole grid steps; per node, the profile is convolved (FFT) with the histogram of shifts.
* :func:`integrate_exact_stream` (the dump-3200 reference): every point's own profile, Doppler shifts
  rounded to grid steps, streamed through a sparse weight matrix (bounded memory, optional worker
  processes), for several lines of sight at once; also yields the T_eff' flux library and two checks.
* :func:`integrate_exact` (brute force with continuous shifts, validation) and
  :func:`integrate_library_nearest` (nearest 10 K library bin, the first library method).

The intensity method (the SPAMMS approach, Abdul-Masih et al. 2020) replaces mu f_i F_c,i by the emergent
intensities of point i's model in its direction mu_i, so limb darkening and the centre-to-limb change of the line
enter: F / F_c = sum_i mu_i I_l,i(y_i', mu_i) / sum_i mu_i I_c,i(y_i', mu_i). The intensity library holds, per
T_eff' bin, the rays of one representative model (modified pformalsol, OUT_IMU files; rays with p <= R_max mapped
to mu = sqrt(1 - (p / R_max)^2), I linear in s = p / R_max between them):

* :class:`DiscImu` (the all-dump 'imu' run): intensities interpolated linearly in T_eff' between the
  representatives and in s between their rays; FFT convolution per (node, ray) row; precomputed library FFTs
  (default, 7.7 GB for M424) or a lazy low-memory mode (the float32 intensities only, ~2-3 GB per process).
* :func:`integrate_imu_nearest` (the dump-3200 intensity reference, disc_los8_imu.npz): the nearest library bin
  of every point, and :func:`uniform_star_imu` (a uniform star returns the representative's FASTWIND flux).

Conventions
-----------
* Profiles F are continuum-normalised; the absorption depth is d = 1 - F. The last axis is the grid y.
* Doppler shifts: v [km/s] > 0 towards the observer; the shift in grid steps is
  rint(-c ln(1 - v/c) / dv) (:meth:`VelocityGrid.shift_steps`), the profile seen at y is the rest profile
  at y + s dv (blueshift for s > 0).
* Line-of-sight sets, mu, the local basis and v: :mod:`ppmpy.synspec.sphere` (``project_los``,
  ``los_velocity``); weights mu F_c: :func:`ppmpy.synspec.sphere.disc_weights`.
* The rest profiles must have zero depth wherever a shift can bring in material from beyond the grid:
  |y| > vmax - vshift (:meth:`VelocityGrid.check_zero_padding`; :class:`DiscFlux` checks it). The exact
  sums handle any shift (contributions beyond the grid are dropped, i.e. taken as continuum).

Validation
----------
tests/synspec/test_disc.py. Synthetic: :class:`DiscFlux`, :func:`integrate_exact`,
:func:`integrate_library_nearest` equal the frozen fw_disc.py bit for bit; :func:`integrate_exact_stream`
equals the frozen fw_disc_los.py (its source lines run on a synthetic star) bit for bit, for any
``nproc``, ``rows`` and start method; its library equals :meth:`ppmpy.synspec.library.FluxLibrary.build`;
analytic cases (uniform star, uniform velocity, other grids and numbers of lines). M424 (marker m424):
:class:`DiscFlux` reproduces the stored products of dumps 3200, 4000 and 4800 of the flux, flux_sm335 and
flux_lamfix runs (F, F0 after the float32 cast, vmean_w, sigma_w, n_clip) and
:func:`integrate_exact_stream` reproduces disc_los8.npz and
library_dT10.npz bit for bit (numpy 1.26 with the AVX512 SVML np.log on the Trillium nodes; see the
library and sphere module notes for what is hardware-dependent). Intensity method: tests/synspec/test_discimu.py.
:class:`DiscImu` equals the frozen fw_disc.DiscImu and :func:`integrate_imu_nearest` the frozen fw_disc_imu.py
(run as a script on a toy star) bit for bit; analytic cases (mu-independent intensities = :class:`DiscFlux`,
per-point brute force, linear limb darkening, rigid rotation); M424: the stored imu/dNNNN.npz of dumps 3200,
4000, 4800 (every member) and disc_los8_imu.npz (F, F0, vmean_w, sigma_w, diagnostics, uniform-star checks)
bit for bit.

PP 2026-10-01: ported from the project's fw_disc.py (DiscFlux, integrate_lib, integrate_exact, shift_steps)
and fw_disc_los.py (the streamed exact sums, the library by-product and the checks); see the provenance
comments per function.
PP 2026-10-02: intensity method ported from fw_disc.py (DiscImu) and fw_disc_imu.py (nearest-bin integration,
uniform-star check); new: the lazy and float32 modes, line subsets, blockwise set-up FFTs, fingerprint, pickling.
PP 2026-10-02: opt-in sub-grid Doppler shifts, ``deposit='linear'`` of :class:`DiscFlux` and :class:`DiscImu`: each
point's histogram weight is split between the two neighbouring shift steps s0 = floor(x), s0 + 1 with weights 1 - w1,
w1 (x = -c ln(1 - v/c) / dv, w1 = x - s0; :meth:`VelocityGrid.shift_steps`), i.e. the shifted profile is
interpolated linearly between grid steps (error O(dv^2) instead of O(dv) for smooth profiles); the FFT machinery is
unchanged. Needed when the line-of-sight velocities are not >> dv (M487, IGW only: ~0.5 km/s on the 1 km/s grid, where
whole steps put almost every point at shift 0). The default 'nearest' is the code of the M424 products, bit for bit.
'linear' is exact for the node profiles taken as piecewise linear between grid points; how the real profiles behave
between grid points is a separate, resolution-limited systematic (DiscFlux notes). Reviewer:
:meth:`DiscFlux.with_deposit` and :meth:`DiscImu.with_deposit` (the other deposit without a new set-up, for the
validation's rounded-shift references).
"""
import hashlib
import json
import mmap
import os
import queue
import shutil
import sys
import tempfile
import time

import numpy as np
from scipy import fft as sfft
from scipy import sparse

from . import parallel as par
from .conventions import C_KMS
from .diagnostics import DIAG_KEYS, diagnostics_array, line_diagnostics
from .io import file_identity, make_meta, npz_member_memmap, save_npz
from .library import FluxLibrary, _file_identity, _share, _unshare, node_pairs, teff_bins, teff_edges
from .sphere import _los_vectors, disc_weights, los_velocity, project_los
from .spectral import LineSet, VelocityGrid, check_deposit, interp_rows, y_of_lam

__all__ = ["DiscFlux", "integrate_exact_stream", "integrate_exact", "integrate_library_nearest", "save_disc_los",
           "LEGACY_DIAG_KW", "DISC_LOS_KEYS", "PAD_TOL", "INTERP_BYTES", "TEFF_MATCH_TOL", "WORKER_MALLOC",
           "DiscImu", "integrate_imu_nearest", "uniform_star_imu", "IMU_LIBRARY_KEYS", "IMU_FFT_MODES", "IMU_DTYPES",
           "IMU_FFT_BLOCK"]

PAD_TOL = 1e-12
"""Default largest |depth| of the :class:`DiscFlux` node profiles at |y| > vmax - vshift: rounding level. Node profiles
built with float64 weights that do not sum to exactly 1 keep a residue of a few 1e-16 there (M424 lib_nodes with
smooth=335: 1.78e-15 = 8 eps); 1e-12 of the continuum is far below any line depth and below the float32 precision of
the stored products (6e-8)."""

INTERP_BYTES = 1 << 24
"""Size [bytes] of one (rows, ny) float64 interpolation temporary for ``rows='auto'``: 16 MiB, well below glibc's
largest mmap threshold (32 MiB), so the ~25 temporaries of :func:`ppmpy.synspec.spectral.interp_rows` are reused from
the heap instead of being mapped, page-faulted and unmapped on every call (M424, ny = 5401: 388 rows)."""

TEFF_MATCH_TOL = 0.01
"""Tolerance [K] of the T_eff' check of :func:`integrate_exact_stream` (velocity samples vs models) on top of the
largest teff_nudge: float32 rounding of the sample T_eff' (<= 0.002 K at 38 000 K)."""

LEGACY_DIAG_KW = dict(vwin=400.0, ew_jacobian=False, inclusive=False)
"""Diagnostics options of the M424 disc_los8.npz (written 2026-09-28, before the EW got the factor lambda/lref and
the moment window became |y| <= vwin): the default of :func:`integrate_exact_stream`. The frozen fw_disc.py of
2026-10-01 corresponds to dict(vwin=400.0, ew_jacobian=True, inclusive=True)."""

DISC_LOS_KEYS = ("Y", "LREF", "los", "F", "F0", "diag_keys", "diag_F", "diag_F0", "vmean_w", "sigma_w", "check_keys",
                 "check_vals")
"""Members of the legacy disc_los8.npz (:func:`save_disc_los` adds '_meta')."""


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------
def _grid(grid):
    """A VelocityGrid (default: the M424 grid)."""
    if grid is None:
        return VelocityGrid()
    if not isinstance(grid, VelocityGrid):
        raise TypeError("grid must be a ppmpy.synspec.spectral.VelocityGrid, got {}".format(type(grid).__name__))
    return grid


def _grid_y(grid):
    """The abscissa of a VelocityGrid or a 1-D array (default: the M424 grid)."""
    if grid is None:
        return VelocityGrid().y
    y = np.asarray(getattr(grid, "y", grid), dtype=np.float64)
    if y.ndim != 1 or y.size < 2:
        raise ValueError("grid must be a VelocityGrid or a 1-D array of velocities")
    return y


def _rows(rows, ny, name="rows"):
    """A ``rows`` argument: 'auto' -> the largest count whose (rows, ny) float64 temporaries fit in INTERP_BYTES (at
    least 1; M424: 388); None stays None (whole blocks); else an int >= 1."""
    if isinstance(rows, str):
        if rows != "auto":
            raise ValueError("{} must be 'auto', None or a positive integer, got {!r}".format(name, rows))
        return max(1, INTERP_BYTES // (8 * int(ny)))
    if rows is None:
        return None
    try:
        ok = int(rows) == rows and rows >= 1
    except (TypeError, ValueError, OverflowError):
        ok = False
    if not ok:
        raise ValueError("{} must be 'auto', None or a positive integer, got {!r}".format(name, rows))
    return int(rows)


def _pad_error(depth, grid, tol):
    """The message of a failed zero-padding check: a rounding residue or a profile that is really cut."""
    if depth <= 1e-9:
        hint = ("this is far below any line depth, i.e. a rounding residue of the node arithmetic (e.g. smoothed or "
                "corrected nodes: ~1e-15), not a cut line: pass pad_tol >= {:.3g}".format(depth))
    else:
        hint = ("the profiles are not back at the continuum within vshift of the grid ends, so the FFT convolution "
                "would cut them: use a wider grid (larger vmax) or a smaller vshift (pad_tol >= {:.3g} accepts the "
                "cut)".format(depth))
    return "node profiles have depth up to {:.3g} at |y| > vmax - vshift = {:g} km/s (pad_tol {:g}): {}".format(
        depth, grid.vmax - grid.vshift, tol, hint)


def _lref(lref, nl=None):
    """(names or None, lref float64 (nl,)) from a LineSet or an array."""
    if lref is None:
        raise ValueError("lref is required (a LineSet or one reference wavelength per line)")
    names = getattr(lref, "names", None)
    lr = np.atleast_1d(np.asarray(getattr(lref, "lref", lref), dtype=np.float64))
    if lr.ndim != 1:
        raise ValueError("lref must be 1-D (one wavelength per line)")
    if nl is not None and lr.size != nl:
        raise ValueError("need one reference wavelength per line ({}), got {}".format(nl, lr.size))
    return (list(names) if names is not None else None), lr


def _shift_steps_exact(v, dv):
    """Doppler shift in grid steps without clipping (the exact sums take any shift)."""
    # PP 2026-10-01: ported from fw_disc_los.py:62 (SH = np.rint(-C ln(1 - V/C) / DV)); dv = 1 -> same bits
    return np.rint(-C_KMS * np.log(1.0 - v / C_KMS) / dv).astype(np.int64)


def _shift_add(Arows, Wrows, shifts, ny):
    """
    Sum over velocity groups of the weighted absorption depth W_g - A_g, each moved by its shift s_g
    (grid steps): depth(y_i) += (W_g - A_g)(y_i + s_g). Shifts with |s| >= ny fall off the grid entirely.
    """
    # PP 2026-10-01: ported from fw_disc_los.py:86-95 (shift_add); new: |s| >= ny is skipped (the legacy code raised
    # for s > ny and added nothing for s = ny)
    depth = np.zeros(ny)
    for g, sh in enumerate(shifts):
        if abs(int(sh)) >= ny:
            continue
        dg = Wrows[g] - Arows[g]
        if sh >= 0:
            depth[:ny - sh] += dg[sh:]
        else:
            depth[-sh:] += dg[:ny + sh]
    return depth


# ----------------------------------------------------------------------------------------------
# the flux method with T_eff' interpolation between library nodes (all dumps)
# ----------------------------------------------------------------------------------------------
class DiscFlux:
    """
    Disc integration with T_eff' interpolation between library nodes (port of fw_disc.DiscFlux).

    Point i contributes mu_i F_line,i(lambda (1 - v_i/c)), with the line flux F_line = F_c f interpolated
    linearly in T_eff' between nodes k0, k1 (weights 1 - a, a), and F_c likewise:

        F / F_c = 1 - sum_k F_c,k sum_s H_k(s) d_k(y + s) / sum_k F_c,k H0_k,     d = 1 - f,

    H_k(s) = sum of mu (1 - a) resp. mu a over the visible points (mu > 0) of node k with Doppler shift s
    (whole grid steps, :meth:`VelocityGrid.shift_steps`, clipped to +-nshift and counted), H0_k = sum_s H_k(s).
    The convolutions are products of the FFTs of the fixed node depths (computed once) and of the
    histograms; the FFT length is ``next_fast_len(ny + 2 nshift, real=True)``, so nothing wraps around.

    Parameters
    ----------
    nodes: LibraryNodes or mapping
        Interpolation nodes (:func:`ppmpy.synspec.library.lib_nodes`, or the legacy dict) with t (nn,)
        increasing node T_eff' [K], fc (nn, nl) continuum flux, prof (nn, nl, ny) float64 normalised profiles on
        ``grid``.
    grid: VelocityGrid, optional
        The velocity grid of the node profiles and of the output (default: the M424 grid, dv 1, vmax 2700,
        vshift 400 km/s).
    pad_tol: float
        Largest |depth| allowed at |y| > vmax - vshift (:meth:`VelocityGrid.check_zero_padding`); the FFT
        convolution takes the depth beyond the grid as 0, so a profile that is not back at the continuum
        within vshift of the grid ends would be cut. Default :data:`PAD_TOL` = 1e-12, rounding level: the M424
        nodes of lib_nodes(nmin=20) with or without the lamfix correction have exactly 0 there, those with
        smooth=335 (the flux_sm335 run) up to 1.78e-15 (the local-linear weights do not sum to exactly 1).
        0 demands exact zeros (the default before 2026-10-01). The check changes no result.
    deposit: {'nearest', 'linear'}
        Doppler shifts as whole grid steps ('nearest', default: the M424 products, bit for bit) or split between
        the two neighbouring steps s0 = floor(x), s0 + 1 with weights 1 - w1, w1 ('linear',
        :meth:`VelocityGrid.shift_steps`): H_k(s0) += mu (1 - a) (1 - w1), H_k(s0 + 1) += mu (1 - a) w1 (node k0;
        likewise with a for k1), i.e. linear interpolation of each shifted profile between grid steps. Use
        'linear' when the line-of-sight velocities are not >> dv. F0, vmean and vsig do not depend on it (H0 to
        rounding).

    Attributes
    ----------
    t, fc: np.ndarray
        Node T_eff' (nn,) and continuum flux (nn, nl) (the arrays of ``nodes``, not copied).
    deposit: str
        The deposit option.
    pad_depth: float
        The largest |depth| of the node profiles at |y| > vmax - vshift (0 for exact zeros).
    pad_tol: float
        The tolerance it was checked against.
    node_params: dict
        ``nodes.params`` (nmin, smooth, corr of :func:`ppmpy.synspec.library.lib_nodes`; {} for a dict).
    nn, nl, ny: int
        Nodes, lines, grid points.
    vs, nv: int
        Largest shift in grid steps (grid.nshift) and histogram width 2 vs + 1.
    L: int
        FFT length.
    P: np.ndarray
        (nl, nn, ny) F_c,k f_k (for the profile without Doppler shifts).
    Dhat: np.ndarray
        (nl, nn, L // 2 + 1) rfft of F_c,k d_k.

    Raises
    ------
    ValueError
        Inconsistent shapes, fewer than 2 nodes, or depth beyond the padding tolerance (the message says
        whether the excess looks like a rounding residue, <= 1e-9, or a profile that is really cut).

    Validation
    ----------
    Bit for bit the frozen fw_disc.DiscFlux (synthetic nodes on the M424 grid; same expressions and order:
    histogram rows (node, shift), H0, den = fc.T @ H0, F0 by einsum, the rfft sizes, the einsum of the
    spectra, the vmean/vsig weights). M424: with :func:`ppmpy.synspec.library.lib_nodes` of library_dT10.npz,
    the 'matmul' projections and the per-dump samples it reproduces the stored products of dumps 3200, 4000
    and 4800 of all three flux runs: flux (nmin=20), flux_sm335 (nmin=20, smooth=335) and flux_lamfix
    (nmin=20, corr = lamfix_dT10.npz) (F, F0 after the float32 cast; vmean_w, sigma_w, n_clip exactly). Other
    grids and numbers of lines: equal to a direct (shift-and-add) evaluation to ~1e-15
    (tests/synspec/test_disc.py).

    Notes
    -----
    Memory: P and Dhat, nl nn (ny + L + 2) x 8 bytes (M424: 245 nodes, 3 lines: 69 MB). A call needs a few
    arrays of the number of visible points and the (nn, 2 vs + 1) histogram and its FFT.

    Sub-grid shifts (deposit='linear'; tests/synspec/test_linear_deposit.py): exact for the node profiles taken as
    piecewise linear between their grid points, to rounding: equal to the brute force with continuous shifts and
    that interpolation model (:func:`ppmpy.synspec.validate.brute_force` with continuous=True: every point's profile
    evaluated at y + x by linear interpolation between its grid values): M424 dumps 3200, 4000 on 20 000-point
    subsets of the 8 lines of sight <= 4.3e-15 ('nearest': 2.2e-6 - 4.9e-5), M487 dump 3200 (IGW only, v_los rms
    0.45 km/s) <= 4.9e-15 ('nearest': 6.8e-4 / 8.0e-5 / 1.0e-3 for the lines 4026 / 4200 / 4922, about half the
    velocity signal max|F - F0| of 1.3e-3 / 1.7e-4 / 2.5e-3). A single point with v = 0.3 km/s gives the linearly
    interpolated shifted profile, 'nearest' the unshifted one; against an analytically shifted smooth profile the
    error falls as dv^2 ('linear') and dv ('nearest'). With deposit='nearest' the M424 flux products are reproduced
    bit for bit as before.

    That agreement is with a reference of the same interpolation model; it does not make 'linear' exact relative to
    a continuous shift of the true line profiles. The node profiles are FASTWIND profiles, sampled more coarsely than
    the 1 km/s grid and interpolated linearly onto it, so they are piecewise linear with kinks (M424 flux nodes:
    |second differences| of median 6e-8 in the line cores but isolated kinks up to 6.7e-3, e.g. the core of
    lambda4026 at y = 0). For shifts well below dv (M487) the profile change F - F0 is then partly second order
    (broadening) and concentrated at those kinks, and depends on how the profiles are interpolated between grid
    points: :func:`ppmpy.synspec.validate.brute_force` with continuous='cubic' (Catmull-Rom) gives that sensitivity.
    M487 with the M424 flux nodes, 20 000-point subsets of lines of sight 1-2, lines 4026 / 4200 / 4922 (measured
    2026-10-02; the reviewer's FFT cubic deposit on all points gave the same to two digits): dump 3200, linear minus
    cubic max 3.4e-4 / 1.2e-5 / 3.0e-4 = 41 / 16 / 28 % of max|F - F0| 8.4e-4 / 7.7e-5 / 1.1e-3 (rms over |y| <= 600
    km/s 18 / 6 / 20 %); time-variable parts, linear (nearest) minus cubic: F(3201) - F(3200) 6.6e-6 / 2.5e-7 /
    6.5e-6 (1.1e-5 / 9.3e-7 / 9.5e-6) of 4.2e-5 / 1.1e-5 / 7.3e-5, F(3300) - F(3200) 5.4e-5 / 2.4e-6 / 4.2e-5
    (2.3e-4 / 2.7e-5 / 4.3e-4) of 1.06e-3 / 1.4e-4 / 1.8e-3. 'linear' is 2-5 times closer than 'nearest', but static
    and second-order features of a sub-km/s F - F0 are limited by the line sampling of the library, not by the
    deposit (a library on finer wavelength sampling would be needed): report linear vs cubic as a systematic.
    """

    deposit = "nearest"          # PP 2026-10-02: class default (instances set it; objects pickled before the option)

    def __init__(self, nodes, grid=None, pad_tol=PAD_TOL, deposit="nearest"):
        # PP 2026-10-01: ported from fw_disc.py:373-379 (DiscFlux.__init__); Y.size -> grid.ny, VSHIFT -> grid.nshift
        # PP 2026-10-02: deposit (opt-in sub-grid shifts)
        self.deposit = check_deposit(deposit)
        grid = _grid(grid)
        pad_tol = float(pad_tol)
        if not pad_tol >= 0.0:
            raise ValueError("pad_tol must be >= 0, got {!r}".format(pad_tol))
        t, fc, prof = nodes["t"], nodes["fc"], nodes["prof"]
        t, fc, prof = np.asarray(t), np.asarray(fc), np.asarray(prof)
        if t.ndim != 1 or t.size < 2:
            raise ValueError("need at least 2 nodes (t of shape (nn,)), got shape {}".format(t.shape))
        if not np.all(np.diff(t) > 0):
            raise ValueError("node T_eff' must increase")
        if prof.ndim != 3 or prof.shape[0] != t.size or prof.shape[2] != grid.ny:
            raise ValueError("prof must have shape (nn, nl, ny) = ({}, nl, {}), got {}".format(
                t.size, grid.ny, prof.shape))
        if fc.shape != prof.shape[:2]:
            raise ValueError("fc must have shape (nn, nl) = {}, got {}".format(prof.shape[:2], fc.shape))
        chk = grid.check_zero_padding(1.0 - prof, pad_tol)
        if not chk["ok"]:
            raise ValueError(_pad_error(chk["max_depth"], grid, pad_tol))
        self.pad_depth, self.pad_tol = chk["max_depth"], pad_tol
        self.node_params = dict(getattr(nodes, "params", None) or {})
        self.grid = grid
        self.t, self.fc = t, fc
        self.nn, self.vs, self.nv = self.t.size, grid.nshift, 2 * grid.nshift + 1
        self.nl, self.ny = prof.shape[1], grid.ny
        self.L = sfft.next_fast_len(self.ny + self.nv - 1, real=True)
        self.P = np.ascontiguousarray(np.transpose(self.fc[:, :, None] * prof, (1, 0, 2)))      # (nl, nn, ny)
        d = np.transpose(self.fc[:, :, None] * (1.0 - prof), (1, 0, 2))
        self.Dhat = sfft.rfft(d, n=self.L, axis=-1)                                              # (nl, nn, nf)

    def __repr__(self):
        return "DiscFlux(nn={}, nl={}, {}, T {:.0f}-{:.0f} K{})".format(
            self.nn, self.nl, self.grid, self.t[0], self.t[-1],
            "" if self.deposit == "nearest" else ", deposit=" + self.deposit)

    def with_deposit(self, deposit):
        """
        This integrator with another Doppler-shift deposit, without a new set-up: a shallow copy that shares every
        array (the node spectra P, Dhat do not depend on the deposit). :mod:`ppmpy.synspec.validate` runs the
        library checks (V1, V2, V4, V5), whose references round the shifts, on the 'nearest' twin of a 'linear'
        integrator.

        Parameters
        ----------
        deposit: {'nearest', 'linear'}

        Returns
        -------
        DiscFlux
            ``self`` for its own deposit, else the copy (bit for bit a DiscFlux built with that deposit from the same
            nodes).
        """
        # PP 2026-10-02: new (reviewer: V1, V4, V5 compared a 'linear' integrator with rounded-shift references)
        deposit = check_deposit(deposit)
        if deposit == self.deposit:
            return self
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new.deposit = deposit
        return new

    def pairs(self, teff, mode="clamp"):
        """
        Interpolation nodes and weights of T_eff' (:func:`ppmpy.synspec.library.node_pairs` with ``self.t``).

        Returns
        -------
        k0, k1: np.ndarray of int
        a: np.ndarray
            Weight of k1.
        """
        return node_pairs(self.t, teff, mode=mode)

    def __call__(self, mu, v, k0, k1, a, novel=True):
        """
        Disc-integrated profiles for one line of sight.

        Parameters
        ----------
        mu: np.ndarray
            (N,) r_hat . n of every point; points with mu <= 0 are hidden.
        v: np.ndarray
            (N,) line-of-sight velocity [km/s, > 0 towards the observer].
        k0, k1, a: np.ndarray
            (N,) interpolation nodes and weight of k1 (:meth:`pairs`).
        novel: bool
            Also compute the profile without Doppler shifts (F0).

        Returns
        -------
        F: np.ndarray
            (nl, ny) float64, with Doppler shifts.
        F0: np.ndarray or None
            (nl, ny) without Doppler shifts (None if not ``novel``).
        vmean, vsig: np.ndarray
            (nl,) mean and rms of v weighted by mu F_c (interpolated F_c of each point).
        n_clip: int
            Visible points whose |shift| exceeded nshift (clipped to it; 'nearest': |rint(x)| > nshift, 'linear':
            |x| > nshift, x = -c ln(1 - v/c) / dv).
        """
        # PP 2026-10-01: ported from fw_disc.py:384-400 (DiscFlux.__call__), same operation order
        # PP 2026-10-02: deposit='linear' (the weights of each point split between steps s0 and s0 + 1)
        vis = mu > 0
        m, v, k0, k1, a = mu[vis], v[vis], k0[vis], k1[vis], a[vis]
        if self.deposit == "nearest":
            s, clip = self.grid.shift_steps(v)
            H = np.bincount(np.concatenate([k0, k1]) * self.nv + np.concatenate([s, s]) + self.vs,
                            weights=np.concatenate([m * (1.0 - a), m * a]),
                            minlength=self.nn * self.nv).reshape(self.nn, self.nv)
        else:
            s0, w1, clip = self.grid.shift_steps(v, deposit="linear")
            wa, wb = m * (1.0 - a), m * a
            H = np.bincount(np.concatenate([k0, k1, k0, k1]) * self.nv
                            + np.concatenate([s0, s0, s0 + 1, s0 + 1]) + self.vs,
                            weights=np.concatenate([wa * (1.0 - w1), wb * (1.0 - w1), wa * w1, wb * w1]),
                            minlength=self.nn * self.nv).reshape(self.nn, self.nv)
        H0 = H.sum(axis=1)
        den = self.fc.T @ H0                                                                     # (nl,)
        F0 = np.einsum("k,jky->jy", H0, self.P) / den[:, None] if novel else None
        Hhat = sfft.rfft(H[:, ::-1], n=self.L, axis=1)
        D = sfft.irfft(np.einsum("jkf,kf->jf", self.Dhat, Hhat), n=self.L, axis=-1)[:, self.vs:self.vs + self.ny]
        F = 1.0 - D / den[:, None]
        w = m[:, None] * ((1.0 - a)[:, None] * self.fc[k0] + a[:, None] * self.fc[k1])          # (nvis, nl)
        vm = (w * v[:, None]).sum(axis=0) / w.sum(axis=0)
        sd = np.sqrt((w * (v[:, None] - vm) ** 2).sum(axis=0) / w.sum(axis=0))
        return F, F0, vm, sd, int(clip.sum())

    def integrate_los(self, mu, v, teff=None, pairs=None, novel=True):
        """
        :meth:`__call__` for several lines of sight (the per-LOS loop of fw_disc_dumps.py).

        Parameters
        ----------
        mu, v: np.ndarray
            (nlos, N) projections and line-of-sight velocities (:func:`ppmpy.synspec.sphere.project_los`,
            :func:`ppmpy.synspec.sphere.los_velocity`).
        teff: np.ndarray, optional
            (N,) T_eff' of the points (float64 for the legacy bits); used when ``pairs`` is not given.
        pairs: tuple, optional
            (k0, k1, a) from :meth:`pairs`.
        novel: bool
            Also compute F0.

        Returns
        -------
        dict
            F, F0 (nlos, nl, ny) float64 (F0 None if not ``novel``), vmean_w, sigma_w (nlos, nl),
            n_clip (nlos,) int.
        """
        # PP 2026-10-01: ported from fw_disc_dumps.py:87-95 (the loop over the lines of sight in process())
        mu, v = np.asarray(mu), np.asarray(v)
        if mu.ndim != 2 or mu.shape != v.shape:
            raise ValueError("mu and v must have the same shape (nlos, N), got {} and {}".format(mu.shape, v.shape))
        if pairs is None:
            if teff is None:
                raise ValueError("give teff or pairs")
            pairs = self.pairs(teff)
        k0, k1, a = pairs
        nlos = mu.shape[0]
        F = np.zeros((nlos, self.nl, self.ny))
        F0 = np.zeros((nlos, self.nl, self.ny)) if novel else None
        vm, sd = np.zeros((nlos, self.nl)), np.zeros((nlos, self.nl))
        ncl = np.zeros(nlos, int)
        for k in range(nlos):
            Fk, F0k, vm[k], sd[k], ncl[k] = self(mu[k], v[k], k0, k1, a, novel=novel)
            F[k] = Fk
            if novel:
                F0[k] = F0k
        return dict(F=F, F0=F0, vmean_w=vm, sigma_w=sd, n_clip=ncl)


# ----------------------------------------------------------------------------------------------
# reference methods: nearest library bin, brute force with continuous shifts
# ----------------------------------------------------------------------------------------------
def integrate_library_nearest(lib, teff, v, w, grid=None):
    """
    Disc-integrated profiles from the nearest T_eff' library bin of every point (port of
    fw_disc.integrate_lib; the first library method, the reference of check (b) of
    :func:`integrate_exact_stream`).

    Per bin, the rest absorption depth is convolved with the weighted histogram of the points' Doppler
    shifts (whole grid steps, clipped to +-nshift): F = 1 - sum_b sum_s H_b(s) d_b(y + s) / sum H.

    Parameters
    ----------
    lib: FluxLibrary or mapping
        Flux library (edges (nb + 1,), prof (nb, nl, ny) on ``grid``).
    teff: np.ndarray
        (N,) T_eff' of the points (selects the bin, :func:`ppmpy.synspec.library.teff_bins`).
    v: np.ndarray
        (N,) line-of-sight velocities [km/s, > 0 towards the observer].
    w: np.ndarray
        (N,) or (N, nl) weights (e.g. mu F_c; 0 for hidden points). For 2-D weights the visible points are
        those with w[:, 0] > 0 (legacy rule).
    grid: VelocityGrid, optional
        Default the M424 grid.

    Returns
    -------
    np.ndarray
        (nl, ny) float64.

    Validation
    ----------
    Bit for bit the frozen fw_disc.integrate_lib (synthetic library on the M424 grid; scipy.signal.fftconvolve
    as there).
    """
    # PP 2026-10-01: ported from fw_disc.py:146-162 (integrate_lib); 3 lines -> nl, VSHIFT -> grid.nshift
    from scipy.signal import fftconvolve
    grid = _grid(grid)
    edges = np.asarray(lib["edges"])
    prof = lib["prof"]
    nl = np.shape(prof)[1]
    w = np.asarray(w)
    vis = w[..., 0] > 0 if w.ndim == 2 else w > 0
    b = np.clip(np.digitize(teff[vis], edges) - 1, 0, edges.size - 2)
    s = grid.shift_steps(v[vis])[0] + grid.nshift
    nb, nv = edges.size - 1, 2 * grid.nshift + 1
    out = np.zeros((nl, grid.ny))
    for j in range(nl):
        wj = (w[vis, j] if w.ndim == 2 else w[vis])
        H = np.bincount(b * nv + s, weights=wj, minlength=nb * nv).reshape(nb, nv)
        use = np.where(H.sum(axis=1) > 0)[0]
        d = 1.0 - prof[use, j].astype(np.float64)                     # absorption depth, 0 at the grid ends
        conv = fftconvolve(d, H[use, ::-1], mode="full", axes=1)      # D(y) = sum_s H(s) d(y + s)
        D = conv[:, grid.nshift:grid.nshift + grid.ny].sum(axis=0)
        out[j] = 1.0 - D / H.sum()
    return out


def integrate_exact(lam, fnorm, lref, v, w, idx=None, chunk=20000, grid=None, lines=None, rows="auto"):
    """
    Disc-integrated profiles summing the individual models with continuous Doppler shifts (brute force;
    port of fw_disc.integrate_exact, the reference of check (a) of :func:`integrate_exact_stream`).

    F_j(y) = sum_i w_i f_ij(y - c ln(1 - v_i/c)) / sum_i w_i: each model's profile is placed on its own
    shifted abscissa y_i = c ln(lambda / lref) + c ln(1 - v_i/c) (lambda_obs = lambda (1 - v_i/c)) and
    interpolated linearly onto the grid (:func:`ppmpy.synspec.spectral.interp_rows`, constant beyond the
    model's band). No rounding of the shifts.

    Parameters
    ----------
    lam, fnorm: array-like
        (N, nl, nrow) wavelengths [A, increasing along the last axis] and normalised fluxes of the models
        (arrays or memory maps; only the rows of ``idx`` are read).
    lref: LineSet or array-like
        (nl,) velocity zero points [A].
    v: np.ndarray
        (N,) line-of-sight velocities [km/s, > 0 towards the observer].
    w: np.ndarray
        (N,) or (N, nl) weights (e.g. mu F_c).
    idx: np.ndarray, optional
        Points to sum (default those with w > 0; w[:, 0] for 2-D weights).
    chunk: int
        Points per interpolation offset block and per weighted sum (the legacy 20000; changes the last bits).
    grid: VelocityGrid or np.ndarray, optional
        Output grid (default the M424 grid).
    lines: sequence of int, optional
        Lines to compute (default all); the others stay 0, as in the legacy code.
    rows: int, 'auto' or None
        Rows interpolated at a time inside a chunk (memory: ~25 temporaries of rows x ny x 8 bytes per call,
        ~10 of them alive at once, plus the chunk's (chunk, ny) profiles). Does not change any bit (the rows
        keep the offsets of their chunk positions). 'auto' (default): temporaries of :data:`INTERP_BYTES` =
        16 MiB (M424: 388 rows); larger ones (> 32 MiB, e.g. 1000 rows = 43 MB on the M424 grid) are mapped
        and page-faulted afresh on every call, mostly system time. None = one call per chunk (legacy;
        ~10 x chunk x ny x 8 bytes, 8.6 GB for M424).

    Returns
    -------
    np.ndarray
        (nl, ny) float64.

    Validation
    ----------
    Bit for bit the frozen fw_disc.integrate_exact (synthetic; any ``rows``). The weighted sums are BLAS
    dgemv products, so another BLAS can differ in the last bits.
    """
    # PP 2026-10-01: ported from fw_disc.py:165-177 (integrate_exact); rows is new (bounded memory, same bits);
    # default 'auto' (16 MiB temporaries) instead of 1000 rows after the review (page faults of 43 MB temporaries)
    y = _grid_y(grid)
    _, lr = _lref(lref)
    rows = _rows(rows, y.size)
    nl = np.shape(lam)[1]
    if lr.size != nl:
        raise ValueError("need one reference wavelength per line ({}), got {}".format(nl, lr.size))
    w = np.asarray(w)
    vis = np.where((w[:, 0] if w.ndim == 2 else w) > 0)[0] if idx is None else np.asarray(idx)
    out = np.zeros((nl, y.size))
    chunk = max(int(chunk), 1)
    for j in (range(nl) if lines is None else lines):
        wj = (w[:, j] if w.ndim == 2 else w)
        num = np.zeros(y.size)
        for i0 in range(0, vis.size, chunk):
            ii = vis[i0:i0 + chunk]
            yl = y_of_lam(lam[ii, j], lr[j]) + C_KMS * np.log(1.0 - v[ii, None] / C_KMS)
            num += wj[ii] @ _interp_block(yl, np.asarray(fnorm[ii, j]), y, rows)
        out[j] = num / wj[vis].sum()
    return out


def _interp_block(yl, F, y, rows, out=None):
    """interp_rows of a block, ``rows`` rows at a time with the offsets of their block positions (the same bits as
    one call), into ``out`` (n, ny) if given."""
    n = yl.shape[0]
    if rows is None or rows >= n:
        f = interp_rows(yl, F, y)
        if out is None:
            return f
        out[:n] = f
        return out[:n]
    f = np.empty((n, y.size)) if out is None else out[:n]
    for r0 in range(0, n, rows):
        r1 = min(r0 + rows, n)
        f[r0:r1] = interp_rows(yl[r0:r1], F[r0:r1], y, row0=r0)
    return f


# ----------------------------------------------------------------------------------------------
# exact per-point sums for several lines of sight, streamed (fw_disc_los.py)
# ----------------------------------------------------------------------------------------------
def _weight_matrix(MU, SH, fcj, bins, nb, sub):
    """
    The sparse weight matrix of one line (rows x points) and its row layout: per line of sight its velocity
    groups (visible points, weight mu F_c), per line of sight one row without velocities, one row per T_eff'
    bin (weight 1: the library), and the velocity groups of the check subset of line of sight 0.

    Returns
    -------
    MC: scipy.sparse.csc_matrix
    W: np.ndarray
        Row sums (of the CSR matrix, as the legacy code).
    lay: dict
        groups [(first row, shifts) per line of sight], r_novel, r_lib, r_sub, sv_sub, nrow.
    """
    # PP 2026-10-01: ported from fw_disc_los.py:105-130 (rows, cols, vals per line; same order and dtypes)
    nlos, N = MU.shape
    rows, cols, vals, groups = [], [], [], []
    r0 = 0
    for k in range(nlos):                                # velocity groups per line of sight
        w = disc_weights(MU[k], fcj)
        vis = np.where(w > 0)[0]
        sv, gi = np.unique(SH[k, vis], return_inverse=True)
        rows.append(r0 + gi)
        cols.append(vis)
        vals.append(w[vis])
        groups.append((r0, sv))
        r0 += sv.size
    r_novel = r0
    for k in range(nlos):                                # no velocities
        w = disc_weights(MU[k], fcj)
        vis = np.where(w > 0)[0]
        rows.append(np.full(vis.size, r0 + k))
        cols.append(vis)
        vals.append(w[vis])
    r0 += nlos
    r_lib = r0
    rows.append(r0 + bins)
    cols.append(np.arange(N))
    vals.append(np.ones(N))
    r0 += nb
    r_sub = r0                                           # check (a): velocity groups of the subset of LOS 0
    sv_sub = np.zeros(0, np.int64)
    if sub is not None:
        w1 = disc_weights(MU[0], fcj)
        sv_sub, gi_sub = np.unique(SH[0, sub], return_inverse=True)
        rows.append(r0 + gi_sub)
        cols.append(sub)
        vals.append(w1[sub])
        r0 += sv_sub.size
    M = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r0, N))
    W = np.asarray(M.sum(axis=1)).ravel()
    MC = M.tocsc()
    return MC, W, dict(groups=groups, r_novel=r_novel, r_lib=r_lib, r_sub=r_sub, sv_sub=sv_sub, nrow=r0)


class _ExactStreamer:
    """
    Partial sums M[:, blocks] @ P_blocks of one line (P = the rest profiles on the grid, interpolated block by
    block): partial q sums blocks q, q + stride, q + 2 stride, ... in order (worker q of fw_disc_los.py run with
    --nproc stride).
    """

    def __init__(self, lam, fnorm, mats, y, lref, block, stride, rows):
        self.lam, self.fnorm, self.mats = lam, fnorm, mats
        self.y, self.lref = y, lref
        self.block, self.stride, self.rows = int(block), int(stride), rows

    def partial(self, j, q):
        # PP 2026-10-01: ported from fw_disc_los.py:73-83 (stream(worker)); the interpolation goes `rows` models at a
        # time into one block buffer with the offsets of the block positions (same bits), then one sparse product
        MC = self.mats[j]
        N, ny = MC.shape[1], self.y.size
        A = np.zeros((MC.shape[0], ny))
        buf = None
        for i0 in range(q * self.block, N, self.stride * self.block):
            i1 = min(i0 + self.block, N)
            blk = MC[:, i0:i1]
            if blk.nnz == 0:
                continue
            yl = y_of_lam(np.asarray(self.lam[i0:i1, j]), self.lref[j])
            Fn = np.asarray(self.fnorm[i0:i1, j])
            if self.rows is not None and self.rows < i1 - i0 and buf is None:
                buf = np.empty((self.block, ny))
            A += blk @ _interp_block(yl, Fn, self.y, self.rows, out=buf)
        return A


def _spill(a, d, name):
    """A whole file-backed memory map of ``a``: ``a`` itself if it is one (e.g. a member of profiles.npz), else ``a``
    written to d/name.npy and mapped read-only (start methods other than 'fork')."""
    if isinstance(a, np.memmap) and isinstance(getattr(a, "base", None), mmap.mmap) and a.filename:
        return a
    path = os.path.join(d, name + ".npy")
    np.save(path, np.asarray(a))
    return np.load(path, mmap_mode="r")


def _spill_matrix(M, d, name, method):
    """Share records (library._share) of the data, indices, indptr of a CSC matrix written to d (non-fork pools)."""
    return ("csc", M.shape, tuple(_share(_spill(getattr(M, k), d, "{}_{}".format(name, k)), method)
                                  for k in ("data", "indices", "indptr")))


def _unshare_matrix(rec):
    """The CSC matrix of a _spill_matrix record (memory-mapped arrays), or the matrix itself (fork)."""
    if isinstance(rec, tuple) and rec and rec[0] == "csc":
        return sparse.csc_matrix(tuple(_unshare(r) for r in rec[2]), shape=rec[1])
    return rec


WORKER_MALLOC = dict(mmap_threshold=32 << 20, trim_threshold=512 << 20)
"""glibc malloc settings of the :func:`integrate_exact_stream` pool workers (:func:`_tune_malloc`)."""


def _tune_malloc(mmap_threshold=32 << 20, trim_threshold=512 << 20):
    """
    glibc only (no-op elsewhere): serve allocations below ``mmap_threshold`` from the heap and keep up to
    ``trim_threshold`` bytes of freed heap in this process (mallopt M_MMAP_THRESHOLD, M_TRIM_THRESHOLD).

    A fresh process (a 'spawn' worker) starts with glibc's dynamic thresholds (128 KiB, raised by the first frees):
    the interpolation temporaries of a few to 16 MiB are then given back to the kernel after every
    :func:`ppmpy.synspec.spectral.interp_rows` call and page-faulted again on the next one. Measured on M424
    (integrate_exact_stream, spawn, 8 workers): 145 s wall and 441 s worker system time without, 69 s and 26 s
    with these settings. Returns True if both were set. Changes no result.
    """
    # PP 2026-10-01: new (the spawn workers' system time; called in the pool workers only, never in the caller)
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes
        mallopt = ctypes.CDLL(None).mallopt
    except (OSError, AttributeError):
        return False
    mallopt.argtypes, mallopt.restype = [ctypes.c_int, ctypes.c_int], ctypes.c_int
    M_TRIM_THRESHOLD, M_MMAP_THRESHOLD = -1, -3
    return bool(mallopt(M_MMAP_THRESHOLD, int(mmap_threshold))) & bool(mallopt(M_TRIM_THRESHOLD, int(trim_threshold)))


def _stream_init(spec):
    """Pool initializer: the worker's own _ExactStreamer (fork: inherited arrays and matrices; other start methods:
    memory maps reopened by file name, so the pickled payload stays small), after _tune_malloc."""
    _tune_malloc(**WORKER_MALLOC)
    return _ExactStreamer(_unshare(spec["lam"]), _unshare(spec["fnorm"]), [_unshare_matrix(m) for m in spec["mats"]],
                          VelocityGrid(**spec["grid"]).y, spec["lref"], spec["block"], spec["stride"], spec["rows"])


def _stream_task(task):
    j, q = task
    return par.worker_state().partial(j, q)


def _ordered_window(pool, func, tasks, window, timeout):
    """
    Yield (task, func(task)) in the order of ``tasks``, computed by ``pool``, with at most ``window`` tasks submitted
    beyond the oldest one not yet yielded: this process holds at most ``window`` finished results. (Pool.imap queues
    every task at once, so when one worker stalls the others can finish everything else and all those results wait
    here.) Raises :class:`ppmpy.synspec.parallel.PoolStalled` when no result arrives for ``timeout`` s (None: wait for
    ever) and re-raises the exception of a failed task (e.g. :class:`ppmpy.synspec.parallel.WorkerInitError`).
    """
    # PP 2026-10-01: new (review: the out-of-order partial sums of imap_watchdog had no bound in the parent)
    arrived = queue.Queue()                      # filled by the pool's result-handler thread
    n, window = len(tasks), max(1, int(window))
    held, nsub = {}, 0
    for i in range(n):
        while nsub < min(n, i + window):
            pool.apply_async(func, (tasks[nsub],), callback=lambda r, k=nsub: arrived.put((k, True, r)),
                             error_callback=lambda e, k=nsub: arrived.put((k, False, e)))
            nsub += 1
        while i not in held:
            try:
                k, ok, res = arrived.get(timeout=timeout)
            except queue.Empty:
                raise par.PoolStalled([tasks[k] for k in range(i, n) if k not in held], timeout) from None
            if not ok:
                raise res
            held[k] = res
            del res
        yield tasks[i], held.pop(i)


def _npz_filename(z):
    """The file name of an np.load()ed NpzFile (None if it was opened from a file object without one)."""
    name = getattr(getattr(z, "fid", None), "name", None)
    return os.path.abspath(name) if isinstance(name, str) and os.path.isfile(name) else None


def _profile_arrays(profiles, skip=()):
    """
    lam, fnorm (N, nl, nrow) and the optional members of a profile source (fcont, teff, theta, phi, teff_nudge; not
    those in ``skip``), each accessed once; 'lines' (list of str or None) and 'path'. A path or an np.load()ed
    NpzFile is opened as a ProfileStore (memory maps): reading the members of an NpzFile would load them whole.
    """
    from .fwresults import ProfileStore
    if isinstance(profiles, np.lib.npyio.NpzFile):
        # PP 2026-10-01: np.load(profiles.npz) is the legacy idiom; its members would be read whole (M424: 7.2 GB)
        name = _npz_filename(profiles)
        if name is None:
            raise ValueError("profiles is an np.load()ed .npz without a file name: pass the path of the file "
                             "(memory-mapped members) instead; reading its members would load them whole")
        profiles = name
    if isinstance(profiles, (str, os.PathLike)):
        profiles = ProfileStore.open(os.fspath(profiles))
    keys = set(profiles.keys())
    for k in ("lam", "fnorm"):
        if k not in keys:
            raise KeyError("profiles need a member {!r}".format(k))
    out = {k: profiles[k] for k in ("lam", "fnorm", "fcont", "teff", "theta", "phi", "teff_nudge")
           if k in keys and k not in skip}
    lines = profiles["lines"] if "lines" in keys else getattr(profiles, "lines", None)
    out["lines"] = None if lines is None else [str(s) for s in np.asarray(lines).tolist()]
    out["path"] = getattr(profiles, "path", None)
    return out


def _velocities(velocities):
    """
    (ur, uth, uph) [km/s], the samples' teff (None if absent) and the source file (None for arrays) of a sample
    file, an np.load()ed sample file, a mapping or a 3-sequence.
    """
    if isinstance(velocities, (str, os.PathLike)):
        path = os.path.abspath(os.fspath(velocities))
        with np.load(path) as z:
            u = tuple(np.array(z[k]) for k in ("ur", "uth", "uph"))
            return u, (np.array(z["teff"]) if "teff" in z.files else None), path
    if isinstance(velocities, (tuple, list)) and len(velocities) == 3:
        return tuple(np.asarray(u) for u in velocities), None, None
    try:
        u = tuple(np.asarray(velocities[k]) for k in ("ur", "uth", "uph"))
    except (KeyError, TypeError, IndexError):
        raise ValueError("velocities must be a sample .npz path, a mapping with ur, uth, uph or a 3-sequence") from None
    try:
        t = np.asarray(velocities["teff"]) if "teff" in velocities else None
    except TypeError:
        t = None
    path = _npz_filename(velocities) if isinstance(velocities, np.lib.npyio.NpzFile) else None
    return u, t, path


def _check_teff(vteff, teff, nudge, vpath):
    """Raise if the velocity samples' T_eff' are not the models' (|dT| > max |teff_nudge| + TEFF_MATCH_TOL)."""
    vteff = np.asarray(vteff, dtype=np.float64)
    if vteff.shape != teff.shape:
        raise ValueError("the velocity samples' teff has shape {}, the models' {}".format(vteff.shape, teff.shape))
    tol = TEFF_MATCH_TOL + (float(np.max(np.abs(nudge))) if nudge is not None and np.size(nudge) else 0.0)
    dT = np.abs(vteff - teff)
    bad = dT > tol
    if bad.any():
        raise ValueError("the T_eff' of the velocity samples{} differ from the models' by up to {:.3g} K at {} of {} "
                         "points (tolerance {:.3g} K = largest teff_nudge + {:g} K): samples of another dump? Pass "
                         "check_teff=False to combine them deliberately".format(
                             "" if vpath is None else " " + vpath, float(dT[bad].max()), int(bad.sum()), teff.size,
                             tol, TEFF_MATCH_TOL))
    return float(np.max(dT, initial=0.0, where=np.isfinite(dT)))


def _fc0(fcont, block=100000):
    """fcont[:, :, 0] as float64 (N, nl), read in blocks (memory maps)."""
    N, nl = fcont.shape[:2]
    out = np.empty((N, nl))
    for i0 in range(0, N, block):
        out[i0:i0 + block] = np.asarray(fcont[i0:i0 + block, :, 0]).astype(np.float64)
    return out


def integrate_exact_stream(profiles, velocities, lref, los="thompson2024", grid=None, theta=None, phi=None,
                           teff=None, fc0=None, dT=10.0, checks=True, nsub=20000, seed=1, block=5000, stride=20,
                           rows="auto", nproc=1, start_method=None, maxtasksperchild=None, timeout=900.0,
                           project_method="matvec", diag_kw=None, exact_rows="auto", tmpdir=None, check_teff=True,
                           progress=None, log=None):
    """
    Exact per-point disc integrals for several lines of sight, streamed through a sparse weight matrix
    (port of fw_disc_los.py: the dump-3200 reference product disc_los8.npz and the T_eff' flux library).

    Every point's own profile is used (no library, no interpolation between models), shifted by its
    Doppler shift rounded to whole grid steps. Per line, one sparse weight matrix M (rows x points) holds

    * for each line of sight and each group of points with the same shift: the visible points (mu > 0)
      with weight mu F_c (:func:`ppmpy.synspec.sphere.disc_weights`);
    * for each line of sight: the same weights in one row (the profile without velocities, F0);
    * for each T_eff' bin of width dT: weight 1 (the library sums);
    * with ``checks``: the shift groups of a random subset of line of sight 0.

    The rest-frame profiles are streamed in blocks of ``block`` points: each block is interpolated onto the
    grid (:func:`ppmpy.synspec.spectral.interp_rows`) and M[:, block] @ P_block (float64) is added to partial
    sum q = (block number) mod ``stride``; the partial sums are added in order. Each velocity group's sum is
    then shifted once and the groups are added (F = 1 - sum_g shift(W_g - A_g) / sum_g W_g).

    Parameters
    ----------
    profiles: ProfileStore, str, os.PathLike, np.load()ed .npz or mapping
        The per-point models: lam, fnorm (N, nl, nrow) [A; normalised flux] and, unless given separately,
        fcont (N, nl, nrow), teff, theta, phi (N,); optional lines (names), teff_nudge (N,). A path, or an
        ``np.load(path)`` NpzFile (the legacy idiom), is opened as :class:`ppmpy.synspec.fwresults.ProfileStore`
        (memory maps of profiles.npz; nothing is loaded). Only the members needed are accessed (not fcont
        when ``fc0`` is given, not teff, theta, phi when given).
    velocities: str, os.PathLike, np.load()ed .npz, mapping or 3-sequence
        u_r, u_theta, u_phi (N,) [km/s] of the points (M424: the dump-3200 samples
        samples_r4050_N1236544/d3200.npz, float32, as fw_disc_los.py; converted to float64 exactly). A sample
        file's or mapping's 'teff' is checked against the models' T_eff' (``check_teff``).
    lref: LineSet or array-like
        (nl,) velocity zero points [A]. A LineSet's names name the checks and must equal the profiles' 'lines'
        (in order; line j of the profiles is placed with lref[j]), else ValueError; for an array the names are
        the profiles' 'lines', else 'line<j>'.
    los: str or array-like
        Lines of sight (:func:`ppmpy.synspec.sphere.project_los`; default the 8 of Thompson et al. 2024).
    grid: VelocityGrid, optional
        Output grid (default the M424 grid; dv 1 km/s is the shift rounding).
    theta, phi, teff: np.ndarray, optional
        (N,) point coordinates [rad] and model T_eff' [K] (default: the profiles' members). Bitwise M424
        results need theta, phi of profiles.npz (sphere module notes).
    fc0: np.ndarray, optional
        (N, nl) continuum flux of the weights (default fcont[:, :, 0] as float64).
    dT: float
        Library bin width [K].
    checks: bool
        Run check (a) (rounding vs :func:`integrate_exact` on ``nsub`` random visible points of LOS 0, drawn
        with ``np.random.default_rng(seed)`` as the legacy code) and check (b) (:func:`integrate_library_nearest`
        vs the exact sum for LOS 0).
    nsub, seed: int
        Check (a) subset size and seed (legacy 20000, 1).
    block: int
        Points per streamed block (legacy --block 5000). Sets the interpolation offsets, hence the last bits.
    stride: int
        Number of interleaved partial sums (legacy --nproc; M424 production 20). Changes the float64 sums in
        the last bits, not the work distribution: ``nproc`` workers process the stride x nl tasks.
    rows: int, 'auto' or None
        Models interpolated at a time inside a block (memory and speed; same bits). 'auto' (default): 16 MiB
        interpolation temporaries (:data:`INTERP_BYTES`; M424: 388 rows). None = whole blocks (legacy).
    nproc: int
        Worker processes (each task returns one partial sum; the parent adds them in order, so the result does
        not depend on nproc or the start method, bit for bit). More than stride x nl workers are not used. At
        most 2 nproc tasks are submitted beyond the oldest partial sum not yet added, which bounds the partial
        sums held by the parent (Memory).
    start_method, maxtasksperchild, timeout:
        Pool options (:func:`ppmpy.synspec.parallel.make_pool`); timeout [s] raises
        :class:`ppmpy.synspec.parallel.PoolStalled` when no partial sum arrives for that long (a killed worker).
        'fork' workers inherit everything. With 'spawn' / 'forkserver' nothing large is pickled: memory-mapped
        profiles are reopened by file name (identity-checked), and in-memory profile arrays and the weight
        matrices are written to a temporary directory (``tmpdir``) and memory-mapped by the workers (shared
        pages; the directory is removed at the end, also after an error, but not if the process is killed).
        Scripts using 'spawn' need the ``if __name__ == "__main__":`` guard.
    tmpdir: str, optional
        Parent directory of that temporary directory (default ``tempfile.gettempdir()``). It receives the
        weight matrices (M424: ~0.4 GB) and, for profiles held in memory rather than memory-mapped from a
        file, lam and fnorm at their full size (M424: 4.8 GB); a path or ProfileStore writes no profiles.
    project_method: str
        :func:`ppmpy.synspec.sphere.project_los` method; 'matvec' = fw_disc.mu_vlos as fw_disc_los.py.
    diag_kw: dict, optional
        Options of :func:`ppmpy.synspec.diagnostics.diagnostics_array` for diag_F, diag_F0 and the EW of check
        (b); default :data:`LEGACY_DIAG_KW` (reproduces the stored disc_los8.npz).
    exact_rows: int, 'auto' or None
        ``rows`` of :func:`integrate_exact` in check (a).
    check_teff: bool
        When the velocity source has a 'teff' member (sample files do), require
        |teff_samples - teff_models| <= max |teff_nudge| + :data:`TEFF_MATCH_TOL` at every point, else
        ValueError: velocities of another dump would be combined with these models (M424 d3200.npz: <= 1.0 K,
        the 2 nudged points; d3201: up to 2922 K). False allows a deliberate mix.
    progress: callable, optional
        progress(tasks_done, tasks_total).
    log: callable, optional
        log(message) for progress messages (e.g. print).

    Returns
    -------
    dict
        y, lref, names, los (nlos, 3); F, F0 (nlos, nl, ny) float64; vmean_w, sigma_w (nlos, nl) (mean and rms
        of v weighted by mu F_c); diag_keys, diag_F, diag_F0 (nlos, nl, 5); checks (dict: round_<line>,
        lib_<line>, lib_dEW_<line>, in the legacy order; empty without ``checks``); library
        (:class:`ppmpy.synspec.library.FluxLibrary`, float32 profiles: equal to ``FluxLibrary.build(...,
        block=block, stride=stride)``); stats (N, nb, rows per line, v range, n_visible, the largest
        |teff_samples - teff_models| or None, wall time); params (with 'inputs': path, size and mtime of the
        profiles and velocity files at the time of the call); inputs (the same).

    Memory
    ------
    Parent: the per-point arrays (mu, v, shifts: 3 nlos N x 8 bytes), the nl sparse matrices (~12 bytes per
    entry, about (2 x visible + 1) N entries per line: 138 MB per M424 line), the running (rows of M, ny)
    float64 sum of the current line and at most 2 nproc finished partial sums of that size (M424: 3296 rows,
    142 MB each; at most 2.3 GB for nproc 8, reached when one worker stalls and the others run ahead).
    Each worker: the matrices (shared: inherited under 'fork', memory-mapped from ``tmpdir`` otherwise), one
    partial sum, one block of profiles on the grid (block x ny x 8 bytes) and the interpolation temporaries
    (~10 x rows x ny x 8 bytes alive). Check (a) holds one (nsub, ny) block (M424: 0.86 GB) plus the
    interpolation temporaries of ``exact_rows`` rows. Memory maps add the file pages a process reads to its RSS
    (clean, reclaimable).

    Measured (M424, Trillium login node, 2026-10-01; block 5000, stride 20, nproc 8; bitwise equal to
    disc_los8.npz in every run): defaults (rows 'auto' = 388) with fork 69 s wall, workers 440 s user + 36 s
    system, parent anonymous peak 2.8 GB (VmHWM 7.1 GB with the file pages of the memory maps), 1.5 GB per worker
    (inherited pages counted in each); spawn 70 s, 451 s + 18 s, parent 3.0 GB, 0.58 GB per worker (4.6 GB for
    all 8). Before the review fixes: rows 1000 with fork 106 s, 445 s + 295 s (page faults of the 43 MB
    interpolation temporaries, mapped afresh on every call); rows 'auto' with spawn but without the worker
    malloc settings (:data:`WORKER_MALLOC`) 145 s, 418 s + 441 s (freed heap given back to the kernel after every
    interpolation call). The production fw_disc_los.py took 199 s with 20 workers.

    Validation
    ----------
    Bit for bit the frozen fw_disc_los.py on synthetic stars (its own source lines; any nproc, rows, start
    method); M424 (block 5000, stride 20 = the production run with 20 workers): F, F0, vmean_w, sigma_w,
    diag_F, diag_F0, checks of disc_los8.npz and library_dT10.npz bit for bit.

    Notes
    -----
    Legacy quirks kept: the check subset is drawn from the visible points of LOS 0 with
    ``default_rng(seed).choice(..., replace=False)`` and sorted; the library's empty bins copy the nearest
    filled bin (lower on a tie), tmean = bin centre; diagnostics use the options of the stored product
    (:data:`LEGACY_DIAG_KW`). New: shifts of a whole grid width or more drop out (legacy raised); a LineSet
    must list the profiles' lines in their order; sample T_eff' must match the models' (``check_teff``); an
    np.load()ed profiles file is reopened with memory maps; the parent holds a bounded number of partial sums;
    the pool workers (not the caller) set glibc's malloc thresholds (:data:`WORKER_MALLOC`).
    """
    # PP 2026-10-01: ported from fw_disc_los.py:50-186 (data, MU/V/SH, edges/bins, check subset, per-line weight
    # matrices, streamed partial sums, shift-add, checks (a) and (b), library assembly, diagnostics, vmean/vsig)
    T0 = time.time()

    def _log(msg):
        if log is not None:
            log("[{:7.1f} s] {}".format(time.time() - T0, msg))

    grid = _grid(grid)
    y, ny = grid.y, grid.ny
    diag_kw = dict(LEGACY_DIAG_KW if diag_kw is None else diag_kw)
    skip = [k for k, x in (("fcont", fc0), ("teff", teff), ("theta", theta), ("phi", phi)) if x is not None]
    src = _profile_arrays(profiles, skip=skip)
    lam3, fn3 = src["lam"], src["fnorm"]
    if lam3.ndim != 3 or lam3.shape != fn3.shape:
        raise ValueError("lam and fnorm must have the same shape (N, nl, nrow), got {} and {}".format(
            lam3.shape, fn3.shape))
    N, nl = lam3.shape[:2]
    names, lr = _lref(lref, nl)
    if names is not None and src["lines"] is not None and list(names) != src["lines"]:
        # PP 2026-10-01: review: a LineSet in another order placed line j's profiles with another line's lref
        raise ValueError("the LineSet's lines {} are not the profiles' lines {} (in this order): line j of the "
                         "profiles is placed with lref[j]; reorder the LineSet or pass the reference wavelengths as "
                         "an array".format(list(names), src["lines"]))
    if names is None:
        names = src["lines"] if src["lines"] is not None else ["line{}".format(j) for j in range(nl)]
    if len(names) != nl:
        raise ValueError("need one line name per line ({}), got {}".format(nl, len(names)))

    def _member(x, key):
        if x is not None:
            return np.asarray(x)
        if key not in src:
            raise ValueError("{} is needed (no profiles member {!r})".format(key, key))
        return np.asarray(src[key])

    teff = _member(teff, "teff")
    theta, phi = _member(theta, "theta"), _member(phi, "phi")
    if fc0 is None:
        if "fcont" not in src:
            raise ValueError("fc0 is needed (no profiles member 'fcont')")
        fc_all = _fc0(src["fcont"])
    else:
        fc_all = np.asarray(fc0, dtype=np.float64)
    (ur, uth, uph), vteff, vpath = _velocities(velocities)
    for name, a in (("teff", teff), ("theta", theta), ("phi", phi), ("ur", ur), ("uth", uth), ("uph", uph)):
        if a.shape != (N,):
            raise ValueError("{} must have shape (N,) = ({},), got {}".format(name, N, a.shape))
    if fc_all.shape != (N, nl):
        raise ValueError("fc0 must have shape (N, nl) = ({}, {}), got {}".format(N, nl, fc_all.shape))
    # PP 2026-10-01: review: samples of another dump were accepted silently (M424 d3201 vs d3200 models: up to 2922 K)
    dteff = _check_teff(vteff, teff, src.get("teff_nudge"), vpath) if (check_teff and vteff is not None) else None
    inputs = {}
    for name, p in (("profiles", src["path"]), ("velocities", vpath)):
        if p:
            inputs[name] = file_identity(p)
    block, stride = int(block), int(stride)
    if block < 1 or stride < 1:
        raise ValueError("block and stride must be >= 1")
    rows = _rows(rows, ny)
    exact_rows = _rows(exact_rows, ny, "exact_rows")

    # lines of sight, velocities, shifts (fw_disc_los.py:56-62)
    L = _los_vectors(los)
    nlos = L.shape[0]
    MU, TN, PN = project_los(theta, phi, L, method=project_method)
    V = los_velocity(ur, uth, uph, MU, TN, PN)
    del TN, PN
    SH = _shift_steps_exact(V, grid.dv)
    # library bins (fw_disc_los.py:63-66)
    edges = teff_edges(teff, dT)
    bins = teff_bins(teff, edges)
    nb = edges.size - 1
    cnt = np.bincount(bins, minlength=nb).astype(float)
    # check (a) subset (fw_disc_los.py:67)
    sub = None
    if checks and nsub > 0:
        vis0 = np.where(MU[0] > 0)[0]
        sub = np.sort(np.random.default_rng(seed).choice(vis0, min(int(nsub), int((MU[0] > 0).sum())), replace=False))
    _log("{} points, {} lines of sight, {} lines, {} library bins, {} check points; v_los {:.1f} .. {:.1f} km/s; "
         "{} workers, stride {}, block {}".format(N, nlos, nl, nb, 0 if sub is None else sub.size, V.min(), V.max(),
                                                 nproc, stride, block))

    mats, Ws, lays = [], [], []
    for j in range(nl):
        MC, W, lay = _weight_matrix(MU, SH, fc_all[:, j], bins, nb, sub)
        mats.append(MC)
        Ws.append(W)
        lays.append(lay)
    _log("weight matrices: rows {}, entries {}".format([l["nrow"] for l in lays], [m.nnz for m in mats]))

    F, F0 = np.zeros((nlos, nl, ny)), np.zeros((nlos, nl, ny))
    lib_prof, lib_fc = np.zeros((nb, nl, ny)), np.zeros((nb, nl))
    ok = cnt > 0
    round_chk = [None] * nl

    def finish(j, A):
        # fw_disc_los.py:134-144
        W, lay = Ws[j], lays[j]
        for k in range(nlos):
            g0, sv = lay["groups"][k]
            F[k, j] = 1.0 - _shift_add(A[g0:g0 + sv.size], W[g0:g0 + sv.size], sv, ny) / W[g0:g0 + sv.size].sum()
            F0[k, j] = A[lay["r_novel"] + k] / W[lay["r_novel"] + k]
        r_lib = lay["r_lib"]
        lib_prof[ok, j] = A[r_lib:r_lib + nb][ok] / cnt[ok, None]
        lib_fc[:, j] = np.bincount(bins, weights=fc_all[:, j], minlength=nb) / np.maximum(cnt, 1)
        if sub is not None:
            r_sub = lay["r_sub"]
            Fr_sub = 1.0 - _shift_add(A[r_sub:], W[r_sub:], lay["sv_sub"], ny) / W[r_sub:].sum()
            w1 = disc_weights(MU[0], fc_all[:, j])
            Fc_sub = integrate_exact(lam3, fn3, lr, V[0], w1, idx=sub, grid=y, lines=(j,), rows=exact_rows)[j]
            round_chk[j] = float(np.abs(Fr_sub - Fc_sub).max())
            _log("{} check (a): 1 grid-step rounding vs continuous, {} points of LOS 0: max |dF| {:.1e}".format(
                names[j], sub.size, round_chk[j]))
        mats[j] = None

    # stream: tasks (line, partial sum) in this order; the partial sums of a line are added in order q = 0, 1, ...
    # (sum(pool.map(stream, range(nproc)))), the lines one after the other
    nblocks = -(-N // block)
    nq = min(stride, nblocks)
    tasks = [(j, q) for j in range(nl) for q in range(nq)]
    acc = [None]

    def consume(results):
        for done, ((j, q), Aq) in enumerate(results, 1):
            if progress is not None:
                progress(done, len(tasks))
            if q == 0:
                Aq += 0.0                               # sum() starts from 0: 0 + A_0
                acc[0] = Aq
            else:
                acc[0] += Aq
            del Aq
            if q == nq - 1:
                # partial sums q >= nblocks (no blocks) are zeros: adding them changes nothing
                _log("{}: streamed {} profiles through {} weight rows".format(names[j], N, lays[j]["nrow"]))
                A, acc[0] = acc[0], None
                finish(j, A)
                del A

    nproc = max(1, min(int(nproc), len(tasks)))
    if nproc <= 1:
        streamer = _ExactStreamer(lam3, fn3, mats, y, lr, block, stride, rows)
        consume(((j, q), streamer.partial(j, q)) for j, q in tasks)
    else:
        method = par.get_context(start_method).get_start_method()
        par.login_node_warning(nproc)
        spill = None
        try:
            if method == "fork":
                # inherited without pickling; finish() drops a line's matrix from this list (and from later workers)
                spec = dict(lam=("array", lam3), fnorm=("array", fn3), mats=mats)
            else:
                # PP 2026-10-01: keep the pickled payload small. CPython's spawn launcher writes it into a pipe whose
                # read end the parent itself keeps open, so a child that dies while bootstrapping (e.g. a script
                # without the __main__ guard) blocks the parent for ever once the payload exceeds the pipe buffer
                # (seen with the 415 MB of M424 weight matrices). Files also let the workers share the pages.
                spill = tempfile.mkdtemp(prefix="synspec_disc_", dir=tmpdir)
                spec = dict(lam=_share(_spill(lam3, spill, "lam"), method),
                            fnorm=_share(_spill(fn3, spill, "fnorm"), method),
                            mats=[_spill_matrix(M, spill, "m{}".format(j), method) for j, M in enumerate(mats)])
            spec.update(grid=grid.to_dict(), lref=lr, block=block, stride=stride, rows=rows)
            with par.make_pool(nproc, initializer=_stream_init, initargs=(spec,), maxtasksperchild=maxtasksperchild,
                               start_method=method) as pool:
                consume(_ordered_window(pool, _stream_task, tasks, window=2 * nproc, timeout=timeout))
            del spec
        finally:
            if spill is not None:
                shutil.rmtree(spill, ignore_errors=True)
    mats = None

    # library (same format as the fw_disc.library cache); empty bins take the nearest filled bin
    # (fw_disc_los.py:147-155)
    filled = np.where(ok)[0]
    for i in np.where(~ok)[0]:
        kk = filled[np.argmin(np.abs(filled - i))]
        lib_prof[i], lib_fc[i] = lib_prof[kk], lib_fc[kk]
    tsum = np.bincount(bins, weights=teff, minlength=nb)
    tmean = np.where(ok, tsum / np.maximum(cnt, 1), 0.5 * (edges[:-1] + edges[1:]))
    lib_params = dict(dT=float(dT), block=block, stride=stride, rows=rows, n_models=int(N), n_selected=int(N),
                      select=False, lref=lr.tolist(), ny=int(ny), y0=float(y[0]), y1=float(y[-1]), edges_given=False,
                      prof_dtype="float32", fill_empty=True, source="synspec.disc.integrate_exact_stream")
    lib = FluxLibrary(edges, tmean, cnt, lib_prof.astype(np.float32), lib_fc, dT, params=lib_params,
                      inputs=dict(profiles=src["path"]) if src["path"] else None)
    del lib_prof

    chk = {}
    if sub is not None:
        for j in range(nl):
            chk["round_" + names[j]] = round_chk[j]
    ew_kw = dict(keys=("ew",), ew_jacobian=diag_kw.get("ew_jacobian", True))
    if checks:
        # check (b): nearest-bin library vs the exact sum, LOS 0 (fw_disc_los.py:159-165)
        wl = disc_weights(MU[0], 1.0)[:, None] * lib_fc[bins]
        Fl = integrate_library_nearest(lib, teff, V[0], wl, grid)
        for j in range(nl):
            chk["lib_" + names[j]] = float(np.abs(Fl[j] - F[0, j]).max())
            chk["lib_dEW_" + names[j]] = float(line_diagnostics(Fl[j], y, lr[j], **ew_kw)["ew"]
                                               - line_diagnostics(F[0, j], y, lr[j], **ew_kw)["ew"])
            _log("{} check (b): library vs exact, LOS 0: max |dF| {:.1e}, dEW {:+.1e} A".format(
                names[j], chk["lib_" + names[j]], chk["lib_dEW_" + names[j]]))

    # diagnostics and the weighted line-of-sight velocity moments (fw_disc_los.py:167-177)
    keys = list(DIAG_KEYS)
    with np.errstate(invalid="ignore", divide="ignore"):
        dF = diagnostics_array(F, y, lr, keys=keys, **diag_kw)
        dF0 = diagnostics_array(F0, y, lr, keys=keys, **diag_kw)
    vmean, vsig = np.zeros((nlos, nl)), np.zeros((nlos, nl))
    nvis = np.zeros(nlos, int)
    for k in range(nlos):
        nvis[k] = int((MU[k] > 0).sum())
        for j in range(nl):
            w = disc_weights(MU[k], fc_all[:, j])
            if w.sum() > 0:
                vmean[k, j] = np.sum(w * V[k]) / w.sum()
                vsig[k, j] = np.sqrt(np.sum(w * (V[k] - vmean[k, j]) ** 2) / w.sum())
    stats = dict(N=int(N), nb=int(nb), rows=[int(l["nrow"]) for l in lays], v_min=float(V.min()),
                 v_max=float(V.max()), n_visible=nvis.tolist(), n_check=0 if sub is None else int(sub.size),
                 teff_samples_max_diff=dteff, wall=time.time() - T0)
    params = dict(los=L.tolist(), grid=grid.to_dict(), lref=lr.tolist(), names=list(names), dT=float(dT),
                  checks=bool(checks), nsub=int(nsub), seed=int(seed), block=block, stride=stride, rows=rows,
                  nproc=int(nproc), project_method=project_method, diag_kw=diag_kw, exact_rows=exact_rows,
                  check_teff=bool(check_teff), velocities=vpath, inputs=inputs)
    _log("done: {:.0f} s".format(stats["wall"]))
    return dict(y=y, lref=lr, names=list(names), los=L, F=F, F0=F0, vmean_w=vmean, sigma_w=vsig,
                diag_keys=np.array(keys), diag_F=dF, diag_F0=dF0, checks=chk, library=lib, stats=stats, params=params,
                inputs=inputs)


def save_disc_los(path, result, meta=None):
    """
    Write the result of :func:`integrate_exact_stream` in the layout of the legacy disc_los8.npz
    (:data:`DISC_LOS_KEYS`: Y, LREF, los, F, F0, diag_keys, diag_F, diag_F0, vmean_w, sigma_w, check_keys,
    check_vals) plus '_meta', atomically. The library is saved separately (``result['library'].save(...)``).

    Parameters
    ----------
    path: str or os.PathLike
        Target .npz.
    result: dict
        From :func:`integrate_exact_stream`.
    meta: dict, optional
        Provenance record; default :func:`ppmpy.synspec.io.make_meta` ('synspec.disc_los') with the result's
        params (which hold the identity of the profiles and velocity files at the time of the computation) and
        stats, and the same files as inputs (identity at the time of saving).

    Returns
    -------
    str
        path.
    """
    # PP 2026-10-01: ported from fw_disc_los.py:178-181 (np.savez of disc_los8.npz)
    path = os.fspath(path)
    if meta is None:
        inputs = {k: d["path"] for k, d in (result.get("inputs") or {}).items() if d and d.get("path")}
        if "profiles" not in inputs and "library" in result:
            src = result["library"].inputs.get("profiles")
            if src:
                inputs["profiles"] = src
        meta = make_meta("synspec.disc_los", params=result.get("params", {}), inputs=inputs,
                         stats=result.get("stats", {}))
    chk = result["checks"]
    arrays = dict(Y=result["y"], LREF=result["lref"], los=result["los"], F=result["F"], F0=result["F0"],
                  diag_keys=np.asarray(result["diag_keys"]), diag_F=result["diag_F"], diag_F0=result["diag_F0"],
                  vmean_w=result["vmean_w"], sigma_w=result["sigma_w"], check_keys=np.array(list(chk)),
                  check_vals=np.array(list(chk.values())))
    return save_npz(path, arrays, meta=meta)


# ----------------------------------------------------------------------------------------------
# the intensity method (SPAMMS approach): emergent intensities I(y, mu) of representative models
# ----------------------------------------------------------------------------------------------
IMU_LIBRARY_KEYS = ("src", "teff_rep", "s", "nnode", "Il", "Ic")
"""Members of an intensity library used by :class:`DiscImu` (the legacy imu_library_dT10.npz of fw_imu_library.py,
which also holds edges, tmean, count, idx_rep, rmax): src (nb,) bin whose representative serves bin b; teff_rep (nb,)
its T_eff' [K]; s (nb, nl, K) ray nodes s = p / R_max = sqrt(1 - mu^2), increasing, NaN beyond nnode; nnode (nb, nl);
Il, Ic (nb, nl, K, ny) line and continuum intensity of every ray on the velocity grid."""

IMU_FFT_MODES = ("precomputed", "lazy")
"""Library FFT modes of :class:`DiscImu`."""

IMU_DTYPES = ("float64", "float32")
"""Precisions of the library spectra of :class:`DiscImu`."""

IMU_FFT_BLOCK = 512
"""Rows per block of the library FFTs when :class:`DiscImu` precomputes them (bounds the temporaries). scipy's
pocketfft transforms the rows of a batch a SIMD vector (up to 8 float64 rows with AVX512) at a time and the remaining
rows one by one; 512 is a multiple of every SIMD width, so each row takes the same code path (vector or scalar tail)
as in the legacy single call over all rows, and the bits are the legacy ones on any platform."""


def _imu_has(m, k):
    """Whether mapping m has key k (mappings without __contains__: through keys())."""
    try:
        return k in m
    except TypeError:
        pass
    try:
        return k in list(m.keys())
    except (AttributeError, TypeError):
        return False


def _meta_params(z):
    """The build parameters ('params') of the '_meta' record (JSON text, :func:`ppmpy.synspec.io.save_npz`) of an
    NpzFile or a mapping holding the member '_meta'; {} without one (the legacy files)."""
    rec = json.loads(str(np.asarray(z["_meta"])))
    params = rec.get("params") if isinstance(rec, dict) else None
    return dict(params) if isinstance(params, dict) else {}


def _imu_open(lib, need=IMU_LIBRARY_KEYS, optional=("edges", "idx_rep", "teff_rep", "lref", "lines")):
    """
    The members of an intensity library, each accessed once: the small ones as arrays; Il and Ic as read-only memory
    maps where possible (an uncompressed .npz given by path, or an np.load()ed NpzFile that has a file name), else as
    given (arrays of a mapping; members of a compressed file are read). The build parameters of a '_meta' member (a
    file written by :meth:`ppmpy.synspec.library.ImuLibrary.save`, or a mapping holding '_meta') are returned as the
    entry '_params' (grid, lines, lref; {} for the legacy file). Returns (members, path or None, whether lib was a
    file: a path or an NpzFile with a file name).
    """
    big = ("Il", "Ic")
    if isinstance(lib, np.lib.npyio.NpzFile):
        name = _npz_filename(lib)
        if name is not None:
            lib = name                          # reopen with memory maps: an NpzFile reads a member on every access
    if isinstance(lib, (str, os.PathLike)):
        path = os.path.abspath(os.fspath(lib))
        out = {}
        with np.load(path) as z:
            files = set(z.files)
            miss = [k for k in need if k not in files]
            if miss:
                raise KeyError("{} has no intensity-library member(s) {}".format(path, miss))
            for k in tuple(need) + tuple(optional):
                if k in files and k not in big and k not in out:
                    out[k] = np.asarray(z[k])
            # PP 2026-10-02: the recorded grid / lines of a library saved by ImuLibrary.save (reviewer)
            out["_params"] = _meta_params(z) if "_meta" in files else {}
        for k in big:
            if k in need:
                try:
                    out[k] = npz_member_memmap(path, k)
                except ValueError:              # a compressed member cannot be mapped: read it
                    with np.load(path) as z:
                        out[k] = z[k]
        return out, path, True
    miss = [k for k in need if not _imu_has(lib, k)]
    if miss:
        raise KeyError("the intensity library has no member(s) {} (need {})".format(miss, list(need)))
    out = {}
    for k in tuple(need) + tuple(optional):
        if k not in out and (k in need or _imu_has(lib, k)):
            a = lib[k]
            out[k] = a if (isinstance(a, np.ndarray) or k not in big) else np.asarray(a)
    out["_params"] = _meta_params(lib) if _imu_has(lib, "_meta") else {}
    path = getattr(lib, "path", None)
    return out, (os.path.abspath(os.fspath(path)) if isinstance(path, (str, os.PathLike)) else None), False


def _imu_grid(lib, m):
    """The VelocityGrid recorded by an intensity library (an attribute 'grid', e.g. ImuLibrary.grid, or the 'grid' of
    its '_meta' build parameters), or None (legacy file; a library built on a plain array of velocities)."""
    g = getattr(lib, "grid", None)
    if isinstance(g, VelocityGrid):
        return g
    g = (m.get("_params") or {}).get("grid")
    return VelocityGrid(**g) if isinstance(g, dict) else None


def _mapped_imu_library(lib, m, path):
    """Whether lib is a :class:`ppmpy.synspec.library.ImuLibrary` read from its file with memory maps
    (``ImuLibrary.load(mmap=True)``: a path, the file's identity, Il and Ic read-only memory maps of that file), so
    that an integrator over it can be rebuilt from the file (pickled as a recipe, identity-checked)."""
    if path is None or not getattr(lib, "_mmap", False) or getattr(lib, "_ident", None) is None:
        return False
    for k in ("Il", "Ic"):
        a = m.get(k)
        if not isinstance(a, np.memmap) or a.mode != "r" or getattr(a, "filename", None) is None \
                or os.path.abspath(a.filename) != path:
            return False
    return True


def _same_grid_points(a, b):
    """Whether two VelocityGrids have the same points (ny, first, last), whatever their vshift (an integration
    option: the intensities depend on y only); the rule of :func:`ppmpy.synspec.dumps.flux_integrator`."""
    return (a.ny, float(a.y[0]), float(a.y[-1])) == (b.ny, float(b.y[0]), float(b.y[-1]))


def _imu_lines(lib, m, nl):
    """(LineSet or None, lref float64 (nl,) or None, names or None) recorded by an intensity library: a LineSet
    attribute 'lines', or members / attributes 'lref' and 'lines' (names), or 'lref' and 'lines' of the build
    parameters of its '_meta' record (a file written by ImuLibrary.save). The legacy file records none."""
    ls = getattr(lib, "lines", None)
    if isinstance(ls, LineSet):
        if len(ls) != nl:
            raise ValueError("the library's LineSet has {} lines, its intensities {}".format(len(ls), nl))
        return ls, ls.lref.copy(), list(ls.names)
    params = m.get("_params") or {}
    lref = m.get("lref", getattr(lib, "lref", params.get("lref")))
    names = m.get("lines", ls if ls is not None else params.get("lines"))
    if lref is not None:
        lref = np.atleast_1d(np.asarray(lref, dtype=np.float64))
        if lref.shape != (nl,):
            raise ValueError("the library's lref has shape {}, expected ({},)".format(lref.shape, nl))
    if names is not None:
        names = [str(x) for x in np.atleast_1d(np.asarray(names)).tolist()]
        if len(names) != nl:
            raise ValueError("the library names {} lines, its intensities have {}".format(len(names), nl))
    if lref is not None and names is not None:
        return LineSet(names, lref), lref, names
    return None, lref, names


def _used_rows(H):
    """Rows of a velocity histogram H (rows, nv) holding any nonzero weight: the rows of the F sums.

    The legacy code took the rows with a positive row sum (``H0 > 0``). That is the same set while every weight is
    >= 0 (clamped T_eff' pairs: every M424 product), but it drops rows whose weights are negative or cancel, which
    T_eff' extrapolation (``pairs(mode='extrapolate')``, validate.v3_tails) produces.
    """
    # PP 2026-10-02: replaces fw_disc.py:470 (used = np.where(H0 > 0)[0]); same rows for weights >= 0 (reviewer)
    return np.flatnonzero((H != 0).any(axis=1))


class _SlotCache:
    """Rows of the library spectra (and source intensities) computed once and kept until no pending chunk of any
    line of sight needs them (:meth:`DiscImu._lazy_profiles`): free slots in growing buffers."""

    def __init__(self, nf, ny, cdtype, sdtype, keep_src, start):
        self.nf, self.ny, self.cdtype, self.sdtype, self.keep_src = nf, ny, cdtype, sdtype, keep_src
        self.cap, self.free = 0, []
        self.bl = self.bc = self.ql = self.qc = None
        self.peak = 0
        self.grow(start)

    def grow(self, extra):
        new = max(2 * self.cap, self.cap + int(extra), 1)
        arrs = []
        for old, w, dt, on in ((self.bl, self.nf, self.cdtype, True), (self.bc, self.nf, self.cdtype, True),
                               (self.ql, self.ny, self.sdtype, self.keep_src),
                               (self.qc, self.ny, self.sdtype, self.keep_src)):
            if not on:
                arrs.append(None)
                continue
            a = np.empty((new, w), dt)
            if old is not None:
                a[:self.cap] = old
            arrs.append(a)
        self.bl, self.bc, self.ql, self.qc = arrs
        self.free.extend(range(new - 1, self.cap - 1, -1))
        self.cap = new

    def take(self, n):
        if len(self.free) < n:
            self.grow(n - len(self.free))
        sl = np.array(self.free[len(self.free) - n:], dtype=np.int64)
        del self.free[len(self.free) - n:]
        self.peak = max(self.peak, self.cap - len(self.free))
        return sl

    def nbytes(self):
        return sum(a.nbytes for a in (self.bl, self.bc, self.ql, self.qc) if a is not None)


class DiscImu:
    """
    Disc integration with emergent intensities I(y, mu) (the intensity method, the SPAMMS approach of
    Abdul-Masih et al. 2020; port of fw_disc.DiscImu, the integrator of the M424 'imu' run of fw_disc_dumps.py).

    Point i (mu_i = r_hat_i . n > 0) contributes mu_i I(y + s_i dv, mu_i) of its T_eff', with s_i its Doppler
    shift in whole grid steps (:meth:`VelocityGrid.shift_steps`, clipped to +-nshift and counted):

        F(y) / F_c(y) = sum_i mu_i I_l,i(y + s_i dv, mu_i) / sum_i mu_i I_c,i(y + s_i dv, mu_i).

    I is interpolated linearly in T_eff' between the representative models of the library (nodes: the bins
    with their own representative, ``np.unique(src)``, at their T_eff' ``teff_rep``; :meth:`pairs`, clamped at
    the ends) and linearly in s = sqrt(1 - mu^2) = p / R_max between the rays of each model (this mapping keeps
    FASTWIND's flux; fw_imu_library.py). The weights mu (1 - a)(1 - t), mu (1 - a) t, mu a (1 - t'), mu a t' go
    to the rows (T_eff' node, ray) of a velocity histogram H (rows x (2 nshift + 1)); per row, the line and
    continuum intensities, edge-padded by nshift on both sides, are convolved with the row's histogram by FFTs
    (length ``next_fast_len(ny + 2 nshift + 2 nshift, real=True)``, so nothing wraps around) and summed over the
    rows holding any nonzero weight. Without Doppler shifts F0 = sum_r H0_r I_l,r / sum_r H0_r I_c,r (H0 = row
    sums of H). The line-of-sight velocity moments are weighted by mu I_c(mu) at the line centre (y = 0,
    ``grid.icentre``). The pairs (k0, k1, a) of a call may be any nodes with weights of any sign (T_eff'
    extrapolation: ``pairs(teff, mode='extrapolate')``, or the non-adjacent pairs of
    :func:`ppmpy.synspec.validate.v3_tails`).

    Parameters
    ----------
    lib: str, os.PathLike or mapping
        The intensity library: a path of an .npz (legacy imu_library_dT10.npz, or a file written by
        :meth:`ppmpy.synspec.library.ImuLibrary.save`; Il and Ic are memory-mapped when the file is
        uncompressed), an np.load()ed NpzFile (reopened from its file name with memory maps), or a mapping (e.g.
        :class:`ppmpy.synspec.library.ImuLibrary`) with the members :data:`IMU_LIBRARY_KEYS`. What the library
        records is taken: a LineSet attribute 'lines' and a VelocityGrid attribute 'grid' (ImuLibrary), members /
        attributes 'lref' and 'lines' (names), or the 'grid', 'lines' and 'lref' of the build parameters in the
        '_meta' member of a file (or of a mapping holding '_meta'); then :attr:`lines` and :attr:`lref` are set.
        The legacy file records nothing, so pass lref to :func:`ppmpy.synspec.dumps.disc_dump` there.
    grid: VelocityGrid or dict, optional
        The velocity grid of the library intensities (a dict is passed to VelocityGrid). Default: the grid the
        library records, else the M424 grid (dv 1, vmax 2700, vshift 400 km/s). A given grid must have the
        recorded grid's points (ny, y[0], y[-1]; ValueError otherwise) but may have another vshift (an
        integration option: the intensities depend on y only; its vshift is used). The library must have grid.ny
        points.
    lines: sequence of int or str, optional
        Lines to build (default all): a per-line build holds only those lines' library; the profiles of the
        other lines are NaN (as the legacy ``lines`` argument of a call). Names need a library that records them.
    dtype: {'float64', 'float32'} or a numpy dtype
        Precision of the library spectra, any spelling numpy accepts ('f4', 'single', np.float32, ...; stored as
        the name). 'float64' (default) = legacy. 'float32': precomputed spectra are
        computed in float64 and stored as complex64, lazy ones are computed in float32; the intensities for F0
        are kept in float32 (exact for the float32 M424 library). Histograms and sums stay float64.
    fft: {'precomputed', 'lazy'}
        'precomputed' (default, legacy): the FFTs of all library rows are computed once (M424: 7.6 GB in
        float64) and shared by forked workers. 'lazy': only the library's float32 intensities are referenced
        (memory maps of the file, or the arrays of a mapping) and the FFTs of the rows a call needs are
        computed inside it; :meth:`integrate_los` computes each row once for all lines of sight of a dump.
    chunk: int
        Histogram rows per FFT block (legacy 128; sets the accumulation order, hence the last bits).
    deposit: {'nearest', 'linear'}
        Doppler shifts as whole grid steps ('nearest', default: the M424 imu products, bit for bit) or split between
        the two neighbouring steps s0 = floor(x), s0 + 1 with weights 1 - w1, w1 ('linear',
        :meth:`VelocityGrid.shift_steps`): each of a point's four (node, ray) weights goes to the histogram at s0
        times 1 - w1 and at s0 + 1 times w1, i.e. linear interpolation of the shifted intensities between grid
        steps (as :class:`DiscFlux`). F0 and the velocity moments do not depend on it (F0 to rounding).

    Attributes
    ----------
    grid: VelocityGrid
    t: np.ndarray
        (nn,) node T_eff' [K] (increasing).
    nn, K, nl, ny: int
        Nodes, rays per node (padded), lines of the library, grid points.
    built: tuple of int
        The lines built (``lines``).
    vs, nv, L: int
        nshift, histogram width 2 nshift + 1, FFT length.
    nnode: np.ndarray
        (nn, nl) rays of each node and line.
    Sflat: list of np.ndarray
        Per line, the ray nodes of all T_eff' nodes, NaN entries padded into (1, 2) and node i offset by 10 i
        (one searchsorted call finds every point's rays).
    I0, Ihat: list
        Per built line (None otherwise): (Il, Ic) (nn K, ny) float64 intensities and their FFTs (nn K, L/2 + 1)
        complex128 ('precomputed', 'float64'); float32 / complex64 with dtype 'float32'; None with 'lazy'.
    ic0: list of np.ndarray
        Per built line, I_c at the line centre (nn K,) float64 (velocity weights).
    lines, lref, names:
        LineSet / reference wavelengths / names recorded by the library, or None.
    method: str
        'imu' (:func:`ppmpy.synspec.dumps.run_fields`).
    node_params: dict
        {} (no flux-library node options).
    path: str or None
        The library file (None for a mapping without a 'path').
    dtype, fft, chunk, deposit:
        The options.

    Raises
    ------
    KeyError
        Library members missing.
    ValueError
        Bad options, inconsistent shapes, fewer than 2 nodes, node T_eff' not increasing, ray nodes outside
        [0, 1] or not increasing, a library on another grid (other points than the given grid's).
    TypeError
        A grid that is neither a VelocityGrid nor a dict.

    Validation
    ----------
    'precomputed' with 'float64' (default) is bit for bit the frozen fw_disc.DiscImu (toy libraries on the M424
    grid, empty bins and nnode < K rows included: the arrays t, L, Sflat, I0, Ihat, ic0 and every output of calls
    with clipped shifts, T_eff' beyond the nodes, the disc centre, novel on and off, line subsets). M424: through
    :func:`ppmpy.synspec.dumps.disc_dump` it reproduces the stored files of the imu run of fw_disc_dumps.py for
    dumps 3200, 4000, 4800, every member (F, F0 after the float32 cast, diagnostics, vmean_w, sigma_w, n_clip, ...)
    exactly. 'lazy' gives vmean_w, sigma_w bit for bit the default ones and F bit for bit on x86-64 (scipy 1.13;
    M424 dumps 3200, 4000, 4800 x 8 lines of sight and synthetic cases, per call and in :meth:`integrate_los`:
    each line of sight accumulates the default's chunks in the default's order, and pocketfft gives a row the same
    bits whether it is transformed in a SIMD vector or in the scalar tail of a batch). On other platforms (e.g.
    aarch64, where FMA contraction can make the vector and scalar paths round differently) the lazy F equals the
    default to rounding (~1e-15). F0 to rounding (chunked sums instead of one matrix-vector product; M424
    7.1e-15). 'float32': M424 max |dF| 2.0e-9 (precomputed) and 7.1e-9 (lazy), F0 7.1e-15 (dumps 3200, 4000,
    4800, 8 lines of sight); the diagnostics of the per-dump files (moments over the whole grid, |y| <= 2700 km/s,
    weighted by y and y^2) amplify this systematic error. Max deviations of diag_F (dumps.disc_dump; same dumps and
    lines of sight): precomputed float32 EW 6.8e-8 A, v1 5.0e-5 km/s, sigma 1.3e-3 km/s (biased low), depth 1.2e-9;
    lazy float32 EW 2.6e-7 A, v1 4.8e-4 km/s, sigma 6.3e-3 km/s, depth 7.0e-9 (fwhm unchanged; lazy float64: diag_F
    identical, diag_F0 to 2e-10). For scale, the time rms of the stored imu diagnostics (1601 dumps, per line of
    sight and line): EW 1.2-2.9e-4 A, v1 0.28-0.32 km/s, sigma 0.04-0.12 km/s; lazy float32 is ~0.2 % of the EW
    and v1 rms but 5-16 % of the sigma rms (mostly a bias, -6.3e-3 .. +0.4e-3 km/s; precomputed float32 1-3 %):
    use float64 for the diagnostics of record.
    Analytic and brute-force cases (any grid, 1-3 lines): a per-point sum of the interpolated, shifted,
    edge-padded intensities (<= 2e-14 in float64, every mode; also with extrapolated T_eff' weights < 0 or > 1,
    non-adjacent nodes and rows whose weights cancel, <= 4e-15); intensities independent of mu equal
    :class:`DiscFlux` (<= 2e-14); linear limb darkening matches the analytic disc integrals (line depth 4e-6 D0;
    velocity dispersion of a rigid rotation 2e-6; Gray's rotation profile to the 1 km/s shift rounding)
    (tests/synspec/test_discimu.py). M424 with the V3 extrapolation pairs (margin 300 K; dumps 3334 and 4391, with
    24 and 66 (line of sight, line) cases of rows whose net weight is <= 0, up to 9.35 of mu-weight): lazy F bit for
    bit the default, lazy F0 to 6e-15, the default F equal to a direct convolution without FFTs over every row with
    a nonzero weight to 3e-15 (dump 4391, line of sight 1); the legacy row rule was off by 2.2e-6 (3334) and 1.1e-5
    (4391) in F, and its lazy F0 by up to 2.6e-6.

    Memory and time
    ---------------
    M424 library: nn 303 nodes, K 42 rays, 3 lines, ny 5401, L 7200, i.e. 12 726 rows of 5401 intensities per line
    (1.65 GB in the file's float32 for the 3 lines). Measured on a Trillium login node (2026-10-02; dump 4000, 8 lines
    of sight, one process, glibc malloc settings :data:`WORKER_MALLOC` as in pool workers, 1 BLAS thread):

    ================================  ========  ========  ==========  ==================  =============
    mode                              held      set-up    peak RSS    per dump            dF vs default
    ================================  ========  ========  ==========  ==================  =============
    precomputed, float64 (default)    7.70 GB   21-110 s  9.3 GB      9.1-9.8 s           0
    precomputed, float32              3.85 GB   3 s       5.5 GB      9.3 s               2.0e-9
    lazy, float64                     1 MB (*)  < 0.1 s   2.2-2.8 GB  11.1 s (19.4 s)     0 (F0 7e-15)
    lazy, float32                     1 MB (*)  < 0.1 s   2.2-2.8 GB  10.6 s (14.9 s)     7.1e-9
    ================================  ========  ========  ==========  ==================  =============

    Set-up of the default: 2.5 s user, the rest system time (first touch of 7.7 GB; 20-105 s on a login node,
    depending on its state). (*) plus the library's float32 intensities, memory-mapped from an uncompressed file
    (clean file pages, shared by all processes; counted in RSS once read) or the arrays of a mapping. Peak RSS
    (VmHWM) is the whole process (fresh process, 2026-10-02; imports 0.06 GB), including the projections, velocities
    and samples of the dump (0.4 GB) and those file pages: the lazy peak is 2.2 GB through
    :func:`ppmpy.synspec.dumps.disc_dump` (one call per line of sight) and 2.7-2.8 GB with :meth:`integrate_los`
    (8 histograms at once), of which ~1.6 GB are clean pages of the memory-mapped library (0.83 GB of them mapped
    already at set-up: reading I_c at the line centre of every row faults in the pages around it), reclaimable
    under memory pressure; the anonymous memory peaks at ~0.6 GB (disc_dump) and ~1.1 GB (integrate_los).
    Per dump: 8 lines of sight
    with :meth:`integrate_los`; (19.4 s) with one call per line of sight (each call transforms the library rows it
    needs; :func:`ppmpy.synspec.dumps.disc_dump` calls per line of sight). Without the malloc settings the default
    mode takes 16-23 s per dump on a login node, mostly system time (page faults of the per-call temporaries). Per
    call (one line of sight and line): the histogram (nn K (2 nshift + 1) x 8 bytes, M424 82 MB; lazy
    :meth:`integrate_los`: one per line of sight, 0.65 GB), the per-point arrays (~10 x 8 bytes per visible point
    and contribution, ~0.1 GB) and the FFT blocks (``chunk`` rows; lazy: the row cache, M424 <= 250 rows, 41 MB).
    ``lines=(j,)`` holds a third of the library (one line). Pool workers: 'fork' shares the precomputed library
    (copy-on-write); 'spawn' builds one per worker (pickled as the file name and options, rebuilt and
    identity-checked), so with many spawn workers prefer 'lazy'.

    Notes
    -----
    Pickling: an integrator built from a file, or from an ImuLibrary memory-mapped from its file
    (``ImuLibrary.load(path)``, the default ``mmap=True``), pickles as the file name, the file's identity, the
    options, the recorded lines and a digest of the node T_eff', ray nodes and nnode (a few hundred bytes, either
    mode); unpickling rebuilds it from the file and raises RuntimeError if the file changed (device, inode, size,
    mtime) or those small arrays differ from the file's. One built from any other mapping pickles its arrays:
    'precomputed' its intensities and spectra (``memory()['library']``; M424 7.7 GB, e.g. for every 'spawn' worker
    given it through make_pool initargs), 'lazy' the mapping (M424 1.9 GB); an ImuLibrary read with mmap=False
    counts as such a mapping.

    Rows: F sums the rows of the histogram holding any nonzero weight. The legacy code took the rows with a
    positive row sum, the same rows while every weight is >= 0 (clamped pairs: every M424 product, hence the
    legacy bits), but rows with negative or cancelling weights (extrapolation) were dropped together with every
    other point's contribution to them. Calls raise ValueError for inputs that would give wrong profiles silently:
    non-finite mu; for the visible points a non-finite v or v >= c (the shift would be INT64_MIN, clipped and not
    counted), a non-finite a (a NaN T_eff'), node indices that are not integers in [0, nn).

    Legacy quirks kept: F0 is NaN (not None) without ``novel``; lines not computed are NaN; the ray coordinate of
    node i is offset by 10 i (searchsorted over all nodes at once), which rounds s to ~4e-13.
    """
    method = "imu"
    CHUNK = 128
    deposit = "nearest"          # PP 2026-10-02: class default (instances set it; objects pickled before the option)

    def __init__(self, lib, grid=None, lines=None, dtype="float64", fft="precomputed", chunk=CHUNK, deposit="nearest"):
        # PP 2026-10-02: ported from fw_disc.py:413-435 (DiscImu.__init__): node selection np.unique(src), the padding
        # of s beyond nnode into (1, 2), Sflat and its assertion, the edge-padded rfft of length
        # next_fast_len(ny + 2 vshift + nv - 1); Y.size -> grid.ny, VSHIFT -> grid.nshift, ny // 2 -> grid.icentre,
        # range(3) -> the library's lines; new: dtype, fft, chunk, lines, blockwise FFTs (same bits), checks
        # PP 2026-10-02 (reviewer): a dict grid, any spelling of the dtype, the grid recorded in a file's '_meta',
        # grids compared by their points only (vshift is free), a memory-mapped ImuLibrary pickled as its file
        # PP 2026-10-02: deposit (opt-in sub-grid shifts)
        self.deposit = check_deposit(deposit)
        if isinstance(grid, dict):
            grid = VelocityGrid(**grid)
        if grid is not None:
            grid = _grid(grid)                                   # TypeError for anything but a VelocityGrid
        try:
            dtype = np.dtype(dtype).name                         # 'f4', 'single', np.float32 -> 'float32'
        except TypeError:
            raise ValueError("dtype must be one of {}, got {!r}".format(IMU_DTYPES, dtype)) from None
        if dtype not in IMU_DTYPES:
            raise ValueError("dtype must be one of {}, got {!r}".format(IMU_DTYPES, dtype))
        if fft not in IMU_FFT_MODES:
            raise ValueError("fft must be one of {}, got {!r}".format(IMU_FFT_MODES, fft))
        try:
            ok = int(chunk) == chunk and chunk >= 1
        except (TypeError, ValueError, OverflowError):
            ok = False
        if not ok:
            raise ValueError("chunk must be a positive integer, got {!r}".format(chunk))
        m, path, from_file = _imu_open(lib)
        lgrid = _imu_grid(lib, m)                                # recorded by the library (ImuLibrary, file '_meta')
        if lgrid is not None:
            if grid is None:
                grid = lgrid
            elif not _same_grid_points(grid, lgrid):
                raise ValueError("grid {} (ny {}, y {:g}..{:g} km/s) differs from the library's recorded grid {} "
                                 "(ny {}, y {:g}..{:g} km/s)".format(grid, grid.ny, grid.y[0], grid.y[-1], lgrid,
                                                                     lgrid.ny, lgrid.y[0], lgrid.y[-1]))
        grid = _grid(grid)
        self.grid, self.dtype, self.fft, self.chunk = grid, dtype, fft, int(chunk)
        self._init_args = dict(lines=lines, dtype=dtype, fft=fft, chunk=int(chunk), grid=grid.to_dict(),
                               deposit=self.deposit)
        self.path = path
        # pickled as a recipe (file name, identity, options) when built from a file, or from an ImuLibrary whose
        # intensities are memory maps of its file (ImuLibrary.load(mmap=True)); else as the arrays
        if from_file:
            self._ident = _file_identity(path)
        elif _mapped_imu_library(lib, m, path):
            self._ident = tuple(lib._ident)
        else:
            self._ident = None
        src = np.asarray(m["src"])
        teff_rep = np.asarray(m["teff_rep"])
        S_all = np.asarray(m["s"])
        nnode_all = np.asarray(m["nnode"])
        Il_src, Ic_src = m["Il"], m["Ic"]
        if S_all.ndim != 3:
            raise ValueError("s must have shape (nb, nl, K), got {}".format(S_all.shape))
        nb, nl, K = S_all.shape
        if src.shape != (nb,) or src.dtype.kind not in "iu" or (nb and (src.min() < 0 or src.max() >= nb)):
            raise ValueError("src must hold one bin index in [0, {}) per bin, got shape {} dtype {}".format(
                nb, src.shape, src.dtype))
        if teff_rep.shape != (nb,):
            raise ValueError("teff_rep must have shape ({},), got {}".format(nb, teff_rep.shape))
        if nnode_all.shape != (nb, nl) or nnode_all.dtype.kind not in "iu":
            raise ValueError("nnode must be integers of shape ({}, {}), got {} {}".format(nb, nl, nnode_all.shape,
                                                                                       nnode_all.dtype))
        for name, A in (("Il", Il_src), ("Ic", Ic_src)):
            if tuple(np.shape(A)) != (nb, nl, K, grid.ny):
                raise ValueError("{} must have shape (nb, nl, K, ny) = ({}, {}, {}, {}) (ny of {}), got {}".format(
                    name, nb, nl, K, grid.ny, grid, tuple(np.shape(A))))
        u = np.unique(src)                                       # bins with their own representative model
        t = teff_rep[u]
        if t.size < 2:
            raise ValueError("need at least 2 representative models (nodes), got {}".format(t.size))
        if not np.all(np.diff(t) > 0):
            raise ValueError("the representatives' T_eff' (teff_rep[unique(src)]) must increase")
        nnode = nnode_all[u]
        if np.any(nnode < 2) or np.any(nnode > K):
            raise ValueError("nnode must be in [2, K = {}], got {}..{}".format(K, int(nnode.min()), int(nnode.max())))
        S = S_all[u].astype(np.float64)                         # (nn, nl, K), NaN beyond nnode
        real = np.arange(K)[None, None, :] < nnode[:, :, None]
        sr = S[real]
        if not (np.all(np.isfinite(sr)) and sr.min() >= 0.0 and sr.max() <= 1.0):
            raise ValueError("ray nodes s = p / R_max must be finite and in [0, 1] (the first nnode of every row)")
        self.nn, self.K, self.nl, self.ny = int(u.size), int(K), int(nl), int(grid.ny)
        self.vs, self.nv = grid.nshift, 2 * grid.nshift + 1
        self.t = t
        self.nnode = nnode
        self._u = u
        pad = 1.0 + (1.0 + np.arange(K)) / (K + 1)              # in (1, 2): beyond s <= 1, inside the row's band of 10
        S = np.where(np.isnan(S), pad[None, None, :], S)
        self.Sflat = [(S[:, j, :] + 10.0 * np.arange(self.nn)[:, None]).ravel() for j in range(nl)]
        if not all(np.all(np.diff(x) > 0) for x in self.Sflat):
            raise ValueError("ray nodes not increasing (s of every node and line must increase)")
        self.lines, self.lref, self.names = _imu_lines(lib, m, nl)
        self.built = self._line_list(lines, all_lines=True)
        self._init_args["lines"] = list(self.built)
        self.L = sfft.next_fast_len(self.ny + 2 * self.vs + self.nv - 1, real=True)
        self.node_params = {}
        nr = self.nn * self.K
        h = hashlib.sha256()
        h.update(json.dumps(dict(nn=self.nn, K=self.K, nl=self.nl, ny=self.ny, built=list(self.built),
                                 sdtype=str(np.dtype(getattr(Il_src, "dtype", np.float64)))),
                            sort_keys=True).encode())
        for a in (np.ascontiguousarray(t, dtype=np.float64), np.ascontiguousarray(S),
                  np.ascontiguousarray(nnode, dtype=np.int64)):
            h.update(memoryview(a).cast("B"))
        self._head = h.copy().hexdigest()                       # the small arrays: checked when rebuilt from a recipe
        self.I0, self.Ihat, self.ic0 = [None] * nl, [None] * nl, [None] * nl
        for j in self.built:
            self.ic0[j] = np.asarray(Ic_src[u, j, :, grid.icentre]).reshape(nr).astype(np.float64)
        if fft == "precomputed":
            cdt = np.complex128 if dtype == "float64" else np.complex64
            for j in self.built:
                Il32, Ic32 = np.ascontiguousarray(Il_src[u, j]), np.ascontiguousarray(Ic_src[u, j])
                h.update(memoryview(Il32).cast("B"))
                h.update(memoryview(Ic32).cast("B"))
                if dtype == "float64":
                    Il = Il32.reshape(nr, self.ny).astype(np.float64)
                    Ic = Ic32.reshape(nr, self.ny).astype(np.float64)
                else:
                    Il = Il32.reshape(nr, self.ny).astype(np.float32)
                    Ic = Ic32.reshape(nr, self.ny).astype(np.float32)
                del Il32, Ic32
                self.I0[j] = (Il, Ic)
                self.Ihat[j] = (self._rfft_rows(Il, cdt), self._rfft_rows(Ic, cdt))
            self._sha = h.hexdigest()
            self._src = None
        else:
            self._sha_head = h                                   # completed by fingerprint()
            self._sha = None
            self._src = (Il_src, Ic_src)
        # a mapping in lazy mode (not a memory-mapped ImuLibrary, which pickles as a recipe): pickled through the
        # mapping, i.e. its arrays
        self._lib_ref = lib if (fft == "lazy" and self._ident is None) else None

    # ---- helpers ----
    def __repr__(self):
        return "DiscImu(nn={}, K={}, nl={}, built={}, {}, fft={}, dtype={}{}, T {:.0f}-{:.0f} K)".format(
            self.nn, self.K, self.nl, list(self.built), self.grid, self.fft, self.dtype,
            "" if self.deposit == "nearest" else ", deposit=" + self.deposit, self.t[0], self.t[-1])

    def with_deposit(self, deposit):
        """
        This integrator with another Doppler-shift deposit, without a new set-up (as :meth:`DiscFlux.with_deposit`):
        a shallow copy that shares the library spectra and sources (the deposit is read at call time only); its
        pickling recipe and fingerprint carry the new deposit.

        Parameters
        ----------
        deposit: {'nearest', 'linear'}

        Returns
        -------
        DiscImu
            ``self`` for its own deposit, else the copy (bit for bit a DiscImu built with that deposit and the same
            options).
        """
        # PP 2026-10-02: new (validate: the checks with rounded-shift references run on the 'nearest' twin)
        deposit = check_deposit(deposit)
        if deposit == self.deposit:
            return self
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new.deposit = deposit
        new._init_args = dict(self._init_args, deposit=deposit)
        return new

    def _line_list(self, lines, all_lines=False):
        """Line indices (sorted, unique) of a lines argument; default all lines (constructor) or the built ones."""
        if lines is None:
            return tuple(range(self.nl)) if all_lines else self.built
        if isinstance(lines, (str, int, np.integer)):
            lines = [lines]
        out = []
        for x in lines:
            if isinstance(x, str):
                if self.names is None or x not in self.names:
                    raise ValueError("unknown line {!r} (library lines: {})".format(x, self.names))
                x = self.names.index(x)
            try:
                ok = int(x) == x and 0 <= int(x) < self.nl
            except (TypeError, ValueError, OverflowError):
                ok = False
            if not ok:
                raise ValueError("line indices must be integers in [0, {}), got {!r}".format(self.nl, x))
            out.append(int(x))
        out = tuple(sorted(set(out)))
        if not out:
            raise ValueError("no lines given")
        if not all_lines:
            missing = [j for j in out if j not in self.built]
            if missing:
                raise ValueError("lines {} were not built (built: {})".format(missing, list(self.built)))
        return out

    def _rfft_rows(self, A, cdtype):
        """Edge-padded rfft (length L) of every row of A, in blocks of IMU_FFT_BLOCK rows, computed in float64."""
        n = A.shape[0]
        out = np.empty((n, self.L // 2 + 1), cdtype)
        for r0 in range(0, n, IMU_FFT_BLOCK):
            P = np.pad(np.asarray(A[r0:r0 + IMU_FFT_BLOCK], dtype=np.float64), ((0, 0), (self.vs, self.vs)),
                       mode="edge")
            out[r0:r0 + IMU_FFT_BLOCK] = sfft.rfft(P, n=self.L, axis=1)
        return out

    def _rfft_src(self, A):
        """Edge-padded rfft of source rows (lazy mode): float64, or float32 with dtype 'float32'."""
        dt = np.float64 if self.dtype == "float64" else np.float32
        P = np.pad(np.asarray(A, dtype=dt), ((0, 0), (self.vs, self.vs)), mode="edge")
        return sfft.rfft(P, n=self.L, axis=1)

    def _source_rows(self, j, rows):
        """Source intensities (Il, Ic) of rows (T_eff' node, ray) of line j (lazy mode)."""
        i, k = np.divmod(rows, self.K)
        b = self._u[i]
        return np.asarray(self._src[0][b, j, k]), np.asarray(self._src[1][b, j, k])

    def pairs(self, teff, mode="clamp"):
        """
        Interpolation nodes and weights of T_eff' (:func:`ppmpy.synspec.library.node_pairs` with ``self.t``).

        Returns
        -------
        k0, k1: np.ndarray of int
        a: np.ndarray
            Weight of k1.
        """
        return node_pairs(self.t, teff, mode=mode)

    def _rays(self, i, s, j):
        """Lower ray node kk and fraction tt (linear in s) of points with T_eff' node i."""
        # PP 2026-10-02: ported from fw_disc.py:440-446 (DiscImu._rays)
        sf = self.Sflat[j]
        g = np.searchsorted(sf, s + 10.0 * i, side="right") - 1 - i * self.K
        kk = np.clip(g, 0, self.nnode[i, j] - 2)
        x0, x1 = sf[i * self.K + kk] - 10.0 * i, sf[i * self.K + kk + 1] - 10.0 * i
        return kk, np.clip((s - x0) / (x1 - x0), 0.0, 1.0)

    def _hist(self, j, m, s, sh, k0, k1, a, w1=None):
        """rows, weights (4 per point) and the velocity histogram H (nn K, nv) of line j; with the upper-step weights
        w1 (deposit 'linear', sh = the lower steps s0) every weight is split between s0 (1 - w1) and s0 + 1 (w1)."""
        # PP 2026-10-02: ported from fw_disc.py:461-465 (the rows, weights and bincount of __call__)
        # PP 2026-10-02: w1 (deposit='linear'); the returned rows, wts (velocity moments) are the unsplit ones
        kA, tA = self._rays(k0, s, j)
        kB, tB = self._rays(k1, s, j)
        rows = np.concatenate([k0 * self.K + kA, k0 * self.K + kA + 1, k1 * self.K + kB, k1 * self.K + kB + 1])
        wts = np.concatenate([m * (1 - a) * (1 - tA), m * (1 - a) * tA, m * a * (1 - tB), m * a * tB])
        nr = self.nn * self.K
        if w1 is None:
            H = np.bincount(rows * self.nv + np.tile(sh, 4) + self.vs, weights=wts,
                            minlength=nr * self.nv).reshape(nr, self.nv)
        else:
            sh4, w14 = np.tile(sh, 4), np.tile(w1, 4)
            H = np.bincount(np.concatenate([rows * self.nv + sh4, rows * self.nv + sh4 + 1]) + self.vs,
                            weights=np.concatenate([wts * (1.0 - w14), wts * w14]),
                            minlength=nr * self.nv).reshape(nr, self.nv)
        return rows, wts, H

    def _vmoments(self, j, rows, wts, vv):
        """Mean and rms of v weighted by mu I_c(mu) at the line centre."""
        # PP 2026-10-02: ported from fw_disc.py:479-481
        wp = wts * self.ic0[j][rows]
        vm = np.sum(wp * vv) / wp.sum()
        return vm, np.sqrt(np.sum(wp * (vv - vm) ** 2) / wp.sum())

    def _finish(self, acc_l, acc_c):
        """F = num / den from the accumulated spectra."""
        lo = 2 * self.vs
        num = sfft.irfft(acc_l, n=self.L)[lo:lo + self.ny]
        den = sfft.irfft(acc_c, n=self.L)[lo:lo + self.ny]
        return num / den

    def _precomputed_profiles(self, j, H, H0, novel):
        """F (and F0) of one line of sight from the precomputed library spectra (legacy loop)."""
        # PP 2026-10-02: ported from fw_disc.py:466-478 (F0, the CHUNK loop over the used rows, irfft, num / den)
        Il, Ic = self.I0[j]
        F0 = None
        used = _used_rows(H)
        if novel:
            if self.dtype == "float64":
                F0 = (H0 @ Il) / (H0 @ Ic)
            else:
                n0, d0 = np.zeros(self.ny), np.zeros(self.ny)
                for r0 in range(0, used.size, self.chunk):
                    rr = used[r0:r0 + self.chunk]
                    n0 += H0[rr] @ Il[rr].astype(np.float64)
                    d0 += H0[rr] @ Ic[rr].astype(np.float64)
                F0 = n0 / d0
        nf = self.L // 2 + 1
        acc_l, acc_c = np.zeros(nf, complex), np.zeros(nf, complex)
        Ihl, Ihc = self.Ihat[j]
        for r0 in range(0, used.size, self.chunk):               # small blocks: no huge temporaries (page faults)
            rr = used[r0:r0 + self.chunk]
            Hhat = sfft.rfft(H[rr, ::-1], n=self.L, axis=1)
            acc_l += np.einsum("kf,kf->f", Ihl[rr], Hhat)
            acc_c += np.einsum("kf,kf->f", Ihc[rr], Hhat)
        return self._finish(acc_l, acc_c), F0

    def _lazy_profiles(self, j, Hs, H0s, novel):
        """
        F (and F0) of several lines of sight of line j with library spectra computed on the fly. The chunks of every
        line of sight are the default's (its used rows, ``chunk`` at a time, in order); they are processed in the
        order of their last row, and the spectrum of a row is computed once and kept until the last chunk needing
        it is done, so every row is transformed once per call while each line of sight accumulates exactly the
        default's chunk sums.
        """
        # PP 2026-10-02: new (the lazy mode); per line of sight the arithmetic of fw_disc.py:470-478
        nf, C, nr = self.L // 2 + 1, self.chunk, self.nn * self.K
        nlos = len(Hs)
        acc_l = [np.zeros(nf, complex) for _ in range(nlos)]
        acc_c = [np.zeros(nf, complex) for _ in range(nlos)]
        f0n = [np.zeros(self.ny) for _ in range(nlos)] if novel else None
        f0d = [np.zeros(self.ny) for _ in range(nlos)] if novel else None
        chunks, need = [], np.zeros(nr, np.int64)
        for k, H in enumerate(Hs):
            used = _used_rows(H)
            for r0 in range(0, used.size, C):
                rr = used[r0:r0 + C]
                chunks.append((int(rr[-1]), k, r0, rr))
                need[rr] += 1
        chunks.sort(key=lambda c: c[:3])
        cdt = np.complex128 if self.dtype == "float64" else np.complex64
        sdt = np.dtype(getattr(self._src[0], "dtype", np.float64))
        cache = _SlotCache(nf, self.ny, cdt, sdt, novel, start=2 * C)
        slot = np.full(nr, -1, np.int64)
        for _, k, _, rr in chunks:
            miss = rr[slot[rr] < 0]
            if miss.size:
                sl = cache.take(miss.size)
                slot[miss] = sl
                Al, Ac = self._source_rows(j, miss)
                cache.bl[sl] = self._rfft_src(Al)
                cache.bc[sl] = self._rfft_src(Ac)
                if novel:
                    cache.ql[sl] = Al
                    cache.qc[sl] = Ac
                del Al, Ac
            ss = slot[rr]
            Hhat = sfft.rfft(Hs[k][rr, ::-1], n=self.L, axis=1)
            acc_l[k] += np.einsum("kf,kf->f", cache.bl[ss], Hhat)
            acc_c[k] += np.einsum("kf,kf->f", cache.bc[ss], Hhat)
            if novel:
                h0 = H0s[k][rr]
                f0n[k] += h0 @ cache.ql[ss].astype(np.float64)
                f0d[k] += h0 @ cache.qc[ss].astype(np.float64)
            need[rr] -= 1
            done = rr[need[rr] == 0]
            if done.size:
                cache.free.extend(slot[done].tolist())
                slot[done] = -1
        self._last_cache = dict(rows=cache.peak, bytes=cache.nbytes())
        F = [self._finish(acc_l[k], acc_c[k]) for k in range(nlos)]
        F0 = [f0n[k] / f0d[k] for k in range(nlos)] if novel else [None] * nlos
        return F, F0

    def _prepare(self, mu, v, k0, k1, a):
        """The visible points' mu, s, shifts, T_eff' pairs and v, and the clipped count; ValueError for inputs that
        would silently give wrong profiles (non-finite mu, v or a; v >= c; nodes that are not integers in [0, nn))."""
        # PP 2026-10-02: the checks are new (reviewer): a NaN T_eff' (a = NaN) or v (a shift of INT64_MIN, clipped and
        # not counted) gave finite but wrong profiles
        mu = np.asarray(mu)
        v, k0, k1, a = np.asarray(v), np.asarray(k0), np.asarray(k1), np.asarray(a)
        if mu.ndim != 1 or any(x.shape != mu.shape for x in (v, k0, k1, a)):
            raise ValueError("mu, v, k0, k1, a must be 1-D of one length, got shapes {}, {}, {}, {}, {}".format(
                mu.shape, v.shape, k0.shape, k1.shape, a.shape))
        if not np.all(np.isfinite(mu)):
            raise ValueError("mu has {} non-finite values".format(int(np.sum(~np.isfinite(mu)))))
        vis = mu > 0
        m, v, k0, k1, a = mu[vis], v[vis], k0[vis], k1[vis], a[vis]
        if m.size:
            if k0.dtype.kind not in "iu" or k1.dtype.kind not in "iu":
                raise ValueError("k0, k1 must be integer node indices, got {} and {}".format(k0.dtype, k1.dtype))
            lo, hi = min(int(k0.min()), int(k1.min())), max(int(k0.max()), int(k1.max()))
            if lo < 0 or hi >= self.nn:
                raise ValueError("node indices k0, k1 must be in [0, {}), got {}..{}".format(self.nn, lo, hi))
            bad = ~(np.isfinite(v) & (v < C_KMS))
            if bad.any():
                raise ValueError("{} visible points have a line-of-sight velocity that is not finite or not < c (e.g. "
                                 "{!r} km/s)".format(int(bad.sum()), float(v[bad][0])))
            bad = ~np.isfinite(a)
            if bad.any():
                raise ValueError("{} visible points have a non-finite T_eff' weight a (a NaN T_eff'?)".format(
                    int(bad.sum())))
        s = np.sqrt(np.clip(1.0 - m ** 2, 0.0, 1.0))
        if self.deposit == "nearest":
            sh, clip = self.grid.shift_steps(v)
            w1 = None
        else:
            sh, w1, clip = self.grid.shift_steps(v, deposit="linear")      # PP 2026-10-02: sub-grid shifts
        return dict(m=m, v=v, k0=k0, k1=k1, a=a, s=s, sh=sh, w1=w1, n_clip=int(clip.sum()))

    # ---- the disc integral ----
    def __call__(self, mu, v, k0, k1, a, novel=True, lines=None):
        """
        Disc-integrated profiles for one line of sight.

        Parameters
        ----------
        mu: np.ndarray
            (N,) r_hat . n of every point; points with mu <= 0 are hidden.
        v: np.ndarray
            (N,) line-of-sight velocity [km/s, > 0 towards the observer].
        k0, k1, a: np.ndarray
            (N,) T_eff' interpolation nodes (integers in [0, nn); any two nodes) and weight of k1 (:meth:`pairs`;
            weights 1 - a, a of any sign, e.g. extrapolation).
        novel: bool
            Also compute the profile without Doppler shifts (F0; else F0 stays NaN, as the legacy code).
        lines: sequence of int or str, optional
            Lines to compute (default the built ones; legacy (0, 1, 2)); the others stay NaN.

        Returns
        -------
        F, F0: np.ndarray
            (nl, ny) float64 with / without Doppler shifts.
        vmean, vsig: np.ndarray
            (nl,) mean and rms of v weighted by mu I_c(mu) at the line centre.
        n_clip: int
            Visible points whose |shift| exceeded nshift (clipped to it; 'linear': |x| > nshift, see
            :meth:`VelocityGrid.shift_steps`).

        Raises
        ------
        ValueError
            Inputs of different shapes, a non-finite mu, or for a visible point a non-finite or >= c velocity, a
            non-finite a, or nodes that are not integers in [0, nn).
        """
        # PP 2026-10-02: ported from fw_disc.py:448-482 (DiscImu.__call__), same operation order
        lines = self._line_list(lines)
        p = self._prepare(mu, v, k0, k1, a)
        F, F0 = np.full((self.nl, self.ny), np.nan), np.full((self.nl, self.ny), np.nan)
        vm, sd = np.full(self.nl, np.nan), np.full(self.nl, np.nan)
        vv = np.tile(p["v"], 4)
        for j in lines:
            rows, wts, H = self._hist(j, p["m"], p["s"], p["sh"], p["k0"], p["k1"], p["a"], p["w1"])
            H0 = H.sum(axis=1)
            vm[j], sd[j] = self._vmoments(j, rows, wts, vv)
            del rows, wts
            if self.fft == "precomputed":
                Fj, F0j = self._precomputed_profiles(j, H, H0, novel)
            else:
                (Fj,), (F0j,) = self._lazy_profiles(j, [H], [H0], novel)
            F[j] = Fj
            if novel:
                F0[j] = F0j
            del H, H0
        return F, F0, vm, sd, p["n_clip"]

    def integrate_los(self, mu, v, teff=None, pairs=None, novel=True, lines=None):
        """
        :meth:`__call__` for several lines of sight (the per-LOS loop of fw_disc_dumps.py).

        With fft='lazy', the library spectrum of every row is computed once for all lines of sight (all
        histograms of a line are built first: nlos (nn K) (2 nshift + 1) x 8 bytes, M424 8 x 82 MB); with
        'precomputed' the lines of sight are done one after the other. The results equal the per-call ones
        bit for bit in both modes.

        Parameters
        ----------
        mu, v: np.ndarray
            (nlos, N) projections and line-of-sight velocities.
        teff: np.ndarray, optional
            (N,) T_eff' of the points; used when ``pairs`` is not given.
        pairs: tuple, optional
            (k0, k1, a) from :meth:`pairs`.
        novel: bool
            Also compute F0.
        lines: sequence of int or str, optional
            Lines to compute (default the built ones).

        Returns
        -------
        dict
            F, F0 (nlos, nl, ny) float64 (NaN for lines not computed; F0 NaN without ``novel``), vmean_w,
            sigma_w (nlos, nl), n_clip (nlos,) int.
        """
        # PP 2026-10-02: ported from fw_disc_dumps.py:87-95 (the loop over the lines of sight in process());
        # the lazy batch is new
        mu, v = np.asarray(mu), np.asarray(v)
        if mu.ndim != 2 or mu.shape != v.shape:
            raise ValueError("mu and v must have the same shape (nlos, N), got {} and {}".format(mu.shape, v.shape))
        if pairs is None:
            if teff is None:
                raise ValueError("give teff or pairs")
            pairs = self.pairs(teff)
        k0, k1, a = pairs
        lines = self._line_list(lines)
        nlos = mu.shape[0]
        F = np.full((nlos, self.nl, self.ny), np.nan)
        F0 = np.full((nlos, self.nl, self.ny), np.nan)
        vm, sd = np.full((nlos, self.nl), np.nan), np.full((nlos, self.nl), np.nan)
        ncl = np.zeros(nlos, int)
        if self.fft == "precomputed":
            for k in range(nlos):
                F[k], F0[k], vm[k], sd[k], ncl[k] = self(mu[k], v[k], k0, k1, a, novel=novel, lines=lines)
            return dict(F=F, F0=F0, vmean_w=vm, sigma_w=sd, n_clip=ncl)
        P = [self._prepare(mu[k], v[k], k0, k1, a) for k in range(nlos)]
        for k in range(nlos):
            ncl[k] = P[k]["n_clip"]
        for j in lines:
            Hs, H0s = [], []
            for k, p in enumerate(P):
                rows, wts, H = self._hist(j, p["m"], p["s"], p["sh"], p["k0"], p["k1"], p["a"], p["w1"])
                vm[k, j], sd[k, j] = self._vmoments(j, rows, wts, np.tile(p["v"], 4))
                del rows, wts
                Hs.append(H)
                H0s.append(H.sum(axis=1))
            Fl, F0l = self._lazy_profiles(j, Hs, H0s, novel)
            del Hs, H0s
            for k in range(nlos):
                F[k, j] = Fl[k]
                if novel:
                    F0[k, j] = F0l[k]
        return dict(F=F, F0=F0, vmean_w=vm, sigma_w=sd, n_clip=ncl)

    # ---- identity, memory, pickling ----
    def _library_sha256(self):
        """sha256 of the library content the integrator uses (completed from the source in lazy mode)."""
        if self._sha is None:
            h = self._sha_head.copy()
            for j in self.built:
                for A in self._src:
                    for i0 in range(0, self.nn, 16):
                        blk = np.ascontiguousarray(A[self._u[i0:i0 + 16], j])
                        h.update(memoryview(blk).cast("B"))
            self._sha = h.hexdigest()
        return self._sha

    def fingerprint(self):
        """
        A JSON-able record of everything that determines the results (for the run records of
        :func:`ppmpy.synspec.dumps.run_disc_dumps`): the options, the grid, the built lines and the sha256 of
        the library content used (node T_eff', ray nodes, nnode and the intensities of the representatives of
        the built lines; in lazy mode it is computed on the first call, reading the library once).
        """
        # PP 2026-10-02: new
        # PP 2026-10-02: 'deposit' only when not 'nearest' (the records of the nearest runs stay as they were)
        rec = dict(cls="ppmpy.synspec.disc.DiscImu", method=self.method, fft=self.fft, dtype=self.dtype,
                   chunk=self.chunk, grid=self.grid.to_dict(), lines=list(self.built), nl=self.nl, nn=self.nn,
                   K=self.K, t_range=[float(self.t[0]), float(self.t[-1])], library_sha256=self._library_sha256(),
                   lref=None if self.lref is None else [float(x) for x in self.lref])
        if self.deposit != "nearest":
            rec["deposit"] = self.deposit
        return rec

    def memory(self):
        """
        Bytes held by the integrator: 'library' (its own arrays: spectra, intensities, ray nodes, ic0), 'source'
        (arrays of the library it references in lazy mode, of which 'source_mapped' are memory-mapped file pages,
        clean and shared between processes) and 'per_call' (a rough estimate of the largest temporaries of one call
        for one line of sight, without the per-point arrays: histogram, its FFT block, the lazy row cache).
        """
        own = 0
        for x in self.I0 + self.Ihat:
            if x is not None:
                own += sum(a.nbytes for a in x)
        own += sum(a.nbytes for a in self.ic0 if a is not None) + sum(a.nbytes for a in self.Sflat)
        src = mapped = 0
        if self._src is not None:
            per_line = 2 * self.nn * self.K * self.ny * np.dtype(getattr(self._src[0], "dtype", np.float64)).itemsize
            src = per_line * len(self.built)
            if all(isinstance(a, np.memmap) for a in self._src):
                mapped = src
        nr, nf = self.nn * self.K, self.L // 2 + 1
        per_call = nr * self.nv * 8 + self.chunk * (self.nv + 3 * nf) * 16
        if self.fft == "lazy":
            csz = 16 if self.dtype == "float64" else 8
            per_call += 4 * self.chunk * (2 * nf * csz + 2 * self.ny * 4)
        return dict(library=int(own), source=int(src), source_mapped=int(mapped), per_call=int(per_call))

    def __getstate__(self):
        # PP 2026-10-02: new. Built from a file, or from an ImuLibrary memory-mapped from its file: pickle only how to
        # rebuild it (spawn workers, small payload) with the recorded lines (an ImuLibrary's params may name lines
        # its file does not record) and the digest of the small arrays; the file's identity and that digest are
        # checked when it is rebuilt. Built from any other mapping: the arrays themselves.
        if self._ident is not None:
            return dict(_rebuild=(self.path, self._ident, dict(self._init_args)), _head=self._head,
                        _record=(self.lines, None if self.lref is None else self.lref.copy(),
                                 None if self.names is None else list(self.names)))
        state = self.__dict__.copy()
        state.pop("_sha_head", None)                             # hashlib objects do not pickle
        if state.get("_sha") is None and self.fft == "lazy":
            state["_sha"] = self._library_sha256()
        if state.get("_lib_ref") is not None:
            state.pop("_src")                                    # taken from the mapping again when unpickled
        return state

    def __setstate__(self, state):
        if "_rebuild" not in state:
            self.__dict__.update(state)
            if "_src" not in state:
                m = _imu_open(self._lib_ref)[0]
                self._src = (m["Il"], m["Ic"])
            return
        path, ident, kw = state["_rebuild"]
        now = _file_identity(path)
        if now != tuple(ident):
            raise RuntimeError("the intensity library {} changed since this DiscImu was built (device, inode, size, "
                               "mtime {} -> {})".format(path, tuple(ident), now))
        kw = dict(kw)
        grid = VelocityGrid(**kw.pop("grid"))
        new = DiscImu(path, grid=grid, **kw)
        if state.get("_head") is not None and new._head != state["_head"]:
            raise RuntimeError("the node T_eff', ray nodes or nnode of the intensity library {} differ from those this "
                               "DiscImu was built with (an ImuLibrary changed in memory?)".format(path))
        if "_record" in state:
            new.lines, new.lref, new.names = state["_record"]
        self.__dict__.update(new.__dict__)


# ----------------------------------------------------------------------------------------------
# the intensity method with the nearest library bin (fw_disc_imu.py: dump 3200, disc_los8_imu.npz)
# ----------------------------------------------------------------------------------------------
def _imu_rows(A, rows, j, K):
    """Rows (bin, ray) = divmod(rows, K) of line j of a library array A (nb, nl, K, ny): A[:, j] reshaped to
    (nb K, ny) and indexed by rows, without copying A[:, j]."""
    b, k = np.divmod(rows, K)
    return np.asarray(A[b, j, k])


def _imu_nearest_line(IL, IC, S, NN, b, s, mu, v, j, grid, chunk, shift=True):
    """
    Disc-integrated profile of line j with the nearest-bin intensity library (fw_disc_imu.py integrate()): points in
    bins b at s = sqrt(1 - mu^2), weights mu (1 - t), mu t to the bracketing rays (linear in s), Doppler shifts in
    whole grid steps (zero without ``shift``), FFT convolution of the edge-padded intensities with the histograms
    (scipy.signal.fftconvolve, ``chunk`` rows at a time). Returns (F (ny,), vmean, vsig), the moments weighted by
    mu I_c(mu) at the line centre.
    """
    # PP 2026-10-02: ported from fw_disc_imu.py:52-87 (node_weights, integrate); a.chunk -> chunk, fd.Y.size ->
    # grid.ny, V -> grid.nshift, ny // 2 -> grid.icentre; IL[:, j].reshape(nb K, ny)[rr] -> _imu_rows (same values,
    # without the copy of IL[:, j])
    from scipy.signal import fftconvolve
    nb, _, K = S.shape
    ny, V = grid.ny, grid.nshift
    k = np.empty(b.size, int)
    t = np.empty(b.size)
    for bb in np.unique(b):
        m = b == bb
        n = NN[bb, j]
        sn = S[bb, j, :n]
        kk = np.clip(np.searchsorted(sn, s[m], side="right") - 1, 0, n - 2)
        k[m] = kk
        t[m] = np.clip((s[m] - sn[kk]) / (sn[kk + 1] - sn[kk]), 0.0, 1.0)
    sh = grid.shift_steps(v)[0] if shift else np.zeros(b.size, int)
    nv = 2 * V + 1
    row = np.concatenate([b * K + k, b * K + k + 1])              # (bin, node) rows
    col = np.concatenate([sh, sh]) + V
    wgt = np.concatenate([mu * (1 - t), mu * t])
    H = np.bincount(row * nv + col, weights=wgt, minlength=nb * K * nv).reshape(nb * K, nv)
    rows = np.where(H.sum(axis=1) > 0)[0]
    num, den = np.zeros(ny), np.zeros(ny)
    for r0 in range(0, rows.size, chunk):
        rr = rows[r0:r0 + chunk]
        h = H[rr][:, ::-1]
        for A, acc in ((IL, num), (IC, den)):
            Ip = np.pad(_imu_rows(A, rr, j, K).astype(np.float64), ((0, 0), (V, V)), mode="edge")
            acc += fftconvolve(Ip, h, mode="valid", axes=1).sum(axis=0)
    ic = grid.icentre
    ic0 = (1 - t) * np.asarray(IC[b, j, k, ic]) + t * np.asarray(IC[b, j, k + 1, ic])
    w = mu * ic0
    vm = np.sum(w * v) / w.sum()
    return num / den, vm, np.sqrt(np.sum(w * (v - vm) ** 2) / w.sum())


def _imu_point_arrays(points, teff=None, theta=None, phi=None):
    """teff, theta, phi (N,) (given arrays, else the members of ``points``: a path, an np.load()ed .npz or a
    mapping), the optional teff_nudge and the file name (or None)."""
    want = [k for k, x in (("teff", teff), ("theta", theta), ("phi", phi)) if x is None]
    out = dict(teff=teff, theta=theta, phi=phi)
    nudge, path = None, None
    if points is not None:
        if isinstance(points, np.lib.npyio.NpzFile):
            path = _npz_filename(points)
            src = points
        elif isinstance(points, (str, os.PathLike)):
            path = os.path.abspath(os.fspath(points))
            src = None
        else:
            src = points
        if src is None:
            with np.load(path) as z:
                for k in want:
                    if k not in z.files:
                        raise ValueError("{} has no member {!r}".format(path, k))
                    out[k] = z[k]
                nudge = z["teff_nudge"] if "teff_nudge" in z.files else None
        else:
            for k in want:
                if not _imu_has(src, k):
                    raise ValueError("points have no member {!r}".format(k))
                out[k] = src[k]
            nudge = src["teff_nudge"] if _imu_has(src, "teff_nudge") else None
    elif want:
        raise ValueError("{} needed: give points (profiles.npz / points.npz) or the arrays".format(want))
    return ({k: np.asarray(v) for k, v in out.items()}, None if nudge is None else np.asarray(nudge), path)


def uniform_star_imu(lib, mu, lines, teff=38230.0, runs=None, rep_dir=None, layout="P{idx:06d}/P{idx:06d}",
                     suffix="VTV010", grid=None, chunk=2000, nrow=None):
    """
    Check of the intensity method (fw_disc_imu.py check (1)): a uniform star -- every visible point in the
    library bin of ``teff``, no velocities -- must return that bin's representative model's own FASTWIND flux
    profile.

    Parameters
    ----------
    lib: str, os.PathLike or mapping
        Intensity library with edges, s, nnode, Il, Ic and idx_rep (legacy imu_library_dT10.npz).
    mu: np.ndarray
        (N,) r_hat . n of the points for one line of sight (M424: LOS 1, 'matvec' projections of profiles.npz's
        theta, phi); the visible ones (mu > 0) are used.
    lines: LineSet
        The library's lines (names of the OUT files, reference wavelengths).
    teff: float
        T_eff' of the uniform star [K] (M424 reference 38230).
    runs, rep_dir: str, optional
        Directory holding the representatives' model directories (M424 /scratch/ppathak/fastwind_imu/runs; the
        directory is ``runs/layout.format(idx=idx_rep[bin])``), or the model directory itself. The model directory
        must hold OUT.<line>_<suffix> and OUT_IMU.<line>_<suffix> (modified pformalsol).
    grid: VelocityGrid, optional
        Default the M424 grid.
    chunk: int
        FFT rows per block (legacy --chunk 2000).
    nrow: int, optional
        Rows of the OUT table (default: detected; legacy 161).

    Returns
    -------
    dict
        checks {'uniform_<line>': max |F_uniform - F_FASTWIND|}, F (nl, ny) the uniform-star profiles, Fref (nl, ny)
        the FASTWIND flux profiles (OUT fnorm at the precise OUT_IMU wavelengths, interpolated onto the grid), bin,
        idx_rep, teff_rep, dir.

    Validation
    ----------
    Bit for bit the check_vals of the frozen fw_disc_imu.py (synthetic, test_discimu.py); M424: the check_vals of
    disc_los8_imu.npz (3.9e-6, 4.8e-6, 6.1e-6 for HEI4026, HEII4200, HEI4922).
    """
    # PP 2026-10-02: ported from fw_disc_imu.py:101-114 (check (1)); genfromtxt(OUT, usecols=[4], max_rows=161) ->
    # fwresults.read_out (same numbers), fd.read_imu -> fwresults.read_out_imu (same numbers)
    from .fwresults import read_out, read_out_imu
    grid = _grid(grid)
    if not isinstance(lines, LineSet):
        raise TypeError("lines must be a LineSet (names of the OUT files and reference wavelengths)")
    m = _imu_open(lib, need=("edges", "s", "nnode", "Il", "Ic"), optional=("idx_rep", "teff_rep"))[0]
    edges = np.asarray(m["edges"])
    S, NN = np.asarray(m["s"]), np.asarray(m["nnode"])
    nb, nl, K = S.shape
    if len(lines) != nl:
        raise ValueError("the library has {} lines, lines names {}".format(nl, len(lines)))
    bref = int(np.clip(np.digitize(teff, edges) - 1, 0, nb - 1))
    idx = None
    if rep_dir is None:
        if runs is None or "idx_rep" not in m:
            raise ValueError("give rep_dir, or runs with a library that has idx_rep")
        idx = int(np.asarray(m["idx_rep"])[bref])
        rep_dir = os.path.join(os.fspath(runs), layout.format(idx=idx))
    elif "idx_rep" in m:
        idx = int(np.asarray(m["idx_rep"])[bref])
    mu = np.asarray(mu)
    vis = mu > 0
    nvis = int(vis.sum())
    F, Fref = np.zeros((nl, grid.ny)), np.zeros((nl, grid.ny))
    checks = {}
    for j in range(nl):
        F[j] = _imu_nearest_line(m["Il"], m["Ic"], S, NN, np.full(nvis, bref),
                                 np.sqrt(np.clip(1.0 - mu[vis] ** 2, 0.0, 1.0)), mu[vis], np.zeros(nvis), j, grid,
                                 chunk, shift=False)[0]
        out = read_out(os.path.join(rep_dir, "OUT.{}_{}".format(lines.names[j], suffix)), nrow=nrow)
        w0 = read_out_imu(os.path.join(rep_dir, "OUT_IMU.{}_{}".format(lines.names[j], suffix)))[0]
        f0 = out["fnorm"]
        if w0.shape != f0.shape:
            raise ValueError("{}: OUT and OUT_IMU of {} have different numbers of rows ({} vs {})".format(
                rep_dir, lines.names[j], f0.size, w0.size))
        Fref[j] = interp_rows(C_KMS * np.log(w0[None, :] / lines.lref[j]), f0[None, :], grid.y)[0]
        checks["uniform_" + lines.names[j]] = float(np.abs(F[j] - Fref[j]).max())
    return dict(checks=checks, F=F, Fref=Fref, bin=bref, idx_rep=idx,
                teff_rep=float(np.asarray(m["teff_rep"])[bref]) if "teff_rep" in m else None, dir=rep_dir)


def integrate_imu_nearest(lib, points, velocities, lref, los="thompson2024", grid=None, teff=None, theta=None,
                          phi=None, chunk=2000, project_method="matvec", diag_kw=None, uniform_teff=38230.0,
                          runs=None, rep_dir=None, layout="P{idx:06d}/P{idx:06d}", suffix="VTV010", nrow=None,
                          check_teff=True, log=None):
    """
    Disc-integrated profiles with the nearest-bin intensity library (port of fw_disc_imu.py: the first intensity
    method, dump 3200, disc_los8_imu.npz).

    Every visible point (mu > 0) contributes the intensities of the representative model of its own T_eff' bin
    (no interpolation in T_eff'), interpolated linearly in s = sqrt(1 - mu^2) between the rays and shifted by
    its Doppler shift in whole grid steps:

        F(y) / F_c(y) = sum_i mu_i I_l,i(y + s_i dv, mu_i) / sum_i mu_i I_c,i(y + s_i dv, mu_i);

    F0 likewise without shifts. Per (bin, ray) row the weighted shift histogram is convolved with the edge-padded
    intensities (scipy.signal.fftconvolve, ``chunk`` rows at a time; also for F0, as the legacy script). The
    velocity moments are weighted by mu I_c(mu) at the line centre. With ``runs`` or ``rep_dir`` the uniform-star
    check (:func:`uniform_star_imu`) of line of sight 1 is added.

    Parameters
    ----------
    lib: str, os.PathLike or mapping
        Intensity library with edges, s, nnode, Il, Ic (and idx_rep, teff_rep for the check); Il, Ic are
        memory-mapped from an uncompressed file.
    points: str, os.PathLike, np.load()ed .npz, mapping or None
        Source of teff, theta, phi (N,) (M424: profiles.npz, whose teff includes the +1 K nudges of the two retried
        models; only those members are read) and of teff_nudge (check_teff); None with the arrays given.
    velocities: str, os.PathLike, np.load()ed .npz, mapping or 3-sequence
        u_r, u_theta, u_phi (N,) [km/s] (M424: samples_r4050_N1236544/d3200.npz, float32, as the legacy script).
    lref: LineSet or array-like
        (nl,) velocity zero points [A]; a LineSet's names name the checks and the OUT files (a LineSet is required
        with ``runs`` or ``rep_dir``: ValueError for a plain array there; without the check an array is fine).
    los: str or array-like
        Lines of sight (default the 8 of Thompson et al. 2024).
    grid: VelocityGrid, optional
        Default the M424 grid (the library's).
    teff, theta, phi: np.ndarray, optional
        (N,) instead of the members of ``points``.
    chunk: int
        Rows per FFT convolution block (legacy --chunk 2000; changes the last bits).
    project_method: str
        :func:`ppmpy.synspec.sphere.project_los` method; 'matvec' = fw_disc.mu_vlos (the legacy bits).
    diag_kw: dict, optional
        Options of :func:`ppmpy.synspec.diagnostics.diagnostics_array`; default :data:`LEGACY_DIAG_KW` (the stored
        disc_los8_imu.npz, written 2026-09-28 before the EW got the factor lambda/lref).
    uniform_teff, runs, rep_dir, layout, suffix, nrow:
        The uniform-star check (:func:`uniform_star_imu`; skipped when neither runs nor rep_dir is given).
    check_teff: bool
        Require the velocity samples' teff (if they have one) to match the points' (:func:`integrate_exact_stream`).
    log: callable, optional
        log(message).

    Returns
    -------
    dict
        y, lref, names, los; F, F0 (nlos, nl, ny); vmean_w, sigma_w (nlos, nl); diag_keys, diag_F, diag_F0;
        checks (uniform_<line>); stats, params, inputs. :func:`save_disc_los` writes it in the layout of
        disc_los8_imu.npz.

    Validation
    ----------
    Bit for bit the frozen fw_disc_imu.py run as a script on a synthetic star and library (F, F0, vmean_w,
    sigma_w, diagnostics, check_vals). M424: disc_los8_imu.npz (F, F0, vmean_w, sigma_w, diag_F, diag_F0,
    check_vals) bit for bit (numpy 1.26 on the AVX512 Trillium nodes).

    Notes
    -----
    Memory: the projections and velocities (4 nlos N x 8 bytes; M424 316 MB), per call the histogram
    (nb K (2 nshift + 1) x 8 bytes; M424 94 MB) and the FFT temporaries of ``chunk`` rows (M424 ~0.5 GB); the
    library intensities are memory-mapped (file pages read as needed). Measured (M424, Trillium login node,
    2026-10-02): 74 s (53 s user, 22 s system) on a quiet node, 301 s on a busy one; peak RSS (VmHWM, in a fresh
    process) 2.4 GB, mostly clean pages of the memory-mapped library (69 s on a quiet node in that run); two
    integrations (F and F0) per line and line of sight, each through ~12 700 rows.
    """
    # PP 2026-10-02: ported from fw_disc_imu.py:40-49 (inputs, bins, LOS), :90-99 (the loop over the lines of sight),
    # :101-114 (check (1): uniform_star_imu), :116-118 (diagnostics), :125-128 (the members of the output)
    T0 = time.time()

    def _log(msg):
        if log is not None:
            log("[{:7.1f} s] {}".format(time.time() - T0, msg))

    grid = _grid(grid)
    diag_kw = dict(LEGACY_DIAG_KW if diag_kw is None else diag_kw)
    m, lpath, from_file = _imu_open(lib, need=("edges", "s", "nnode", "Il", "Ic"), optional=("idx_rep", "teff_rep"))
    edges = np.asarray(m["edges"])
    S, NN = np.asarray(m["s"]), np.asarray(m["nnode"])
    if S.ndim != 3:
        raise ValueError("s must have shape (nb, nl, K), got {}".format(S.shape))
    nb, nl, K = S.shape
    for name in ("Il", "Ic"):
        if tuple(np.shape(m[name])) != (nb, nl, K, grid.ny):
            raise ValueError("{} must have shape ({}, {}, {}, {}), got {}".format(name, nb, nl, K, grid.ny,
                                                                                 tuple(np.shape(m[name]))))
    if edges.shape != (nb + 1,):
        raise ValueError("edges must have shape ({},), got {}".format(nb + 1, edges.shape))
    names, lr = _lref(lref, nl)
    if names is None:
        if runs is not None or rep_dir is not None:
            # PP 2026-10-02: the check reads OUT.<line>_<suffix>: fail here, not deep in read_out (reviewer)
            raise ValueError("the uniform-star check (runs / rep_dir) reads the files OUT.<line>_{}: give lref as a "
                             "LineSet (line names and reference wavelengths), not a plain array".format(suffix))
        names = ["line{}".format(j) for j in range(nl)]
    pts, nudge, ppath = _imu_point_arrays(points, teff, theta, phi)
    teff, theta, phi = pts["teff"], pts["theta"], pts["phi"]
    N = teff.size
    (ur, uth, uph), vteff, vpath = _velocities(velocities)
    for name, a in (("teff", teff), ("theta", theta), ("phi", phi), ("ur", ur), ("uth", uth), ("uph", uph)):
        if a.shape != (N,):
            raise ValueError("{} must have shape (N,) = ({},), got {}".format(name, N, a.shape))
    dteff = _check_teff(vteff, teff, nudge, vpath) if (check_teff and vteff is not None) else None
    inputs = {k: file_identity(p) for k, p in (("library", lpath), ("points", ppath), ("velocities", vpath)) if p}
    chunk = int(chunk)
    if chunk < 1:
        raise ValueError("chunk must be >= 1")

    bins_all = teff_bins(teff, edges)
    L = _los_vectors(los)
    nlos = L.shape[0]
    MU, TN, PN = project_los(theta, phi, L, method=project_method)
    V = los_velocity(ur, uth, uph, MU, TN, PN)
    del TN, PN
    _log("intensity library: {} bins, up to {} ray nodes; {} points, {} lines of sight, {} lines".format(
        nb, K, N, nlos, nl))
    F, F0 = np.zeros((nlos, nl, grid.ny)), np.zeros((nlos, nl, grid.ny))
    vmean, vsig = np.zeros((nlos, nl)), np.zeros((nlos, nl))
    nvis = []
    for kk in range(nlos):
        mu, v = MU[kk], V[kk]
        vis = mu > 0
        s = np.sqrt(np.clip(1.0 - mu[vis] ** 2, 0.0, 1.0))
        for j in range(nl):
            F[kk, j], vmean[kk, j], vsig[kk, j] = _imu_nearest_line(m["Il"], m["Ic"], S, NN, bins_all[vis], s,
                                                                    mu[vis], v[vis], j, grid, chunk)
            F0[kk, j] = _imu_nearest_line(m["Il"], m["Ic"], S, NN, bins_all[vis], s, mu[vis], v[vis], j, grid, chunk,
                                          shift=False)[0]
        nvis.append(int(vis.sum()))
        _log("los{}: {} visible points, <v> {:+.2f}, sigma_v {:.2f} km/s".format(kk + 1, nvis[-1], vmean[kk, 0],
                                                                                vsig[kk, 0]))
    chk = {}
    u = None
    if runs is not None or rep_dir is not None:
        u = uniform_star_imu(lpath if from_file else lib, MU[0], LineSet(names, lr), teff=uniform_teff,
                             runs=runs, rep_dir=rep_dir, layout=layout, suffix=suffix, grid=grid, chunk=chunk,
                             nrow=nrow)
        chk.update(u["checks"])
        for j in range(nl):
            _log("{} check (1): uniform star at {} K (bin {}), no velocities vs its FASTWIND flux: max|dF| {:.1e}"
                 .format(names[j], uniform_teff, u["bin"], chk["uniform_" + names[j]]))
    keys = list(DIAG_KEYS)
    with np.errstate(invalid="ignore", divide="ignore"):
        dF = diagnostics_array(F, grid.y, lr, keys=keys, **diag_kw)
        dF0 = diagnostics_array(F0, grid.y, lr, keys=keys, **diag_kw)
    stats = dict(N=int(N), nb=int(nb), K=int(K), n_visible=nvis, v_min=float(V.min()), v_max=float(V.max()),
                 teff_samples_max_diff=dteff, wall=time.time() - T0)
    params = dict(los=L.tolist(), grid=grid.to_dict(), lref=lr.tolist(), names=list(names), chunk=chunk,
                  project_method=project_method, diag_kw=diag_kw, uniform_teff=float(uniform_teff),
                  check_teff=bool(check_teff), velocities=vpath, method="imu_nearest", inputs=inputs)
    _log("done: {:.0f} s".format(stats["wall"]))
    return dict(y=grid.y, lref=lr, names=list(names), los=L, F=F, F0=F0, vmean_w=vmean, sigma_w=vsig,
                diag_keys=np.array(keys), diag_F=dF, diag_F0=dF0, checks=chk, stats=stats, params=params,
                inputs=inputs, uniform=u)    # PP 2026-10-02: uniform-star profiles (F, Fref, bin, ...) for callers' logs
