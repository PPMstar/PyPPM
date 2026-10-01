"""
Disc integration of local flux profiles (the flux method): disc-integrated, continuum-normalised line
profiles of a star whose surface is sampled by an equal-area grid of local models.

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
library and sphere module notes for what is hardware-dependent).

PP 2026-10-01: ported from the project's fw_disc.py (DiscFlux, integrate_lib, integrate_exact, shift_steps)
and fw_disc_los.py (the streamed exact sums, the library by-product and the checks); see the provenance
comments per function.
"""
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
from .io import file_identity, make_meta, save_npz
from .library import FluxLibrary, _share, _unshare, node_pairs, teff_bins, teff_edges
from .sphere import _los_vectors, disc_weights, los_velocity, project_los
from .spectral import VelocityGrid, interp_rows, y_of_lam

__all__ = ["DiscFlux", "integrate_exact_stream", "integrate_exact", "integrate_library_nearest", "save_disc_los",
           "LEGACY_DIAG_KW", "DISC_LOS_KEYS", "PAD_TOL", "INTERP_BYTES", "TEFF_MATCH_TOL", "WORKER_MALLOC"]

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

    Attributes
    ----------
    t, fc: np.ndarray
        Node T_eff' (nn,) and continuum flux (nn, nl) (the arrays of ``nodes``, not copied).
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
    """

    def __init__(self, nodes, grid=None, pad_tol=PAD_TOL):
        # PP 2026-10-01: ported from fw_disc.py:373-379 (DiscFlux.__init__); Y.size -> grid.ny, VSHIFT -> grid.nshift
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
        return "DiscFlux(nn={}, nl={}, {}, T {:.0f}-{:.0f} K)".format(self.nn, self.nl, self.grid, self.t[0],
                                                                       self.t[-1])

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
            Visible points whose |shift| exceeded nshift (clipped to it).
        """
        # PP 2026-10-01: ported from fw_disc.py:384-400 (DiscFlux.__call__), same operation order
        vis = mu > 0
        m, v, k0, k1, a = mu[vis], v[vis], k0[vis], k1[vis], a[vis]
        s, clip = self.grid.shift_steps(v)
        H = np.bincount(np.concatenate([k0, k1]) * self.nv + np.concatenate([s, s]) + self.vs,
                        weights=np.concatenate([m * (1.0 - a), m * a]),
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
