"""
Synthetic data and a self-test of the line-profile pipeline: a toy per-point run (analytic local line profiles on an
equal-area sphere with smooth random T_eff' and velocity fields) that is built, disc-integrated and validated with
the production code, without FASTWIND and without M424 data.

* :func:`toy_lineset`, :func:`toy_line_params`, :func:`toy_depth`, :func:`toy_continuum`: the analytic local models.
  Line j is a pseudo-Voigt absorption line (Gaussian core, Lorentzian wings) whose depth, centre and width vary
  smoothly with T_eff' (quadratic, linear, linear), tapered to exactly zero depth at |y| >= 0.9 (vmax - vshift),
  so that Doppler shifts up to vshift never bring in depth from beyond the grid (the zero padding of
  :class:`ppmpy.synspec.disc.DiscFlux`); the continuum flux varies as (T_eff' / T0)^p.
* :func:`toy_library`: an analytic :class:`ppmpy.synspec.library.FluxLibrary` (``nnode`` filled bins: as many
  interpolation nodes), e.g. for a quick :class:`~ppmpy.synspec.disc.DiscFlux`.
* :func:`toy_sphere`: the points (:func:`ppmpy.synspec.sphere.fibonacci_sphere`) with smooth random fields of
  realistic amplitude: T_eff' with relative rms 0.9 % (M424: 338-345 K at 38 230 K) and u_r, u_theta, u_phi with
  rms 45 km/s each (M424: disc rms v_los 44.6 km/s); or T_eff' mapped onto a uniform distribution over a range (the
  library dump, below).
* :func:`toy_profiles_store`: an uncompressed profiles.npz-like file (the members, their order and dtypes of
  :func:`ppmpy.synspec.fwresults.combine`) holding every point's own model on its own wavelength rows (161 rows,
  denser at the line centre, shifted per point as FASTWIND's frequency grids, float32), so that exact per-point
  sums can be computed (:func:`ppmpy.synspec.disc.integrate_exact_stream`).
* :func:`toy_run`: a whole run in a directory, made with the production functions: per-dump samples, exact sums
  and library of the library dump (integrate_exact_stream), the flux integrator (:func:`ppmpy.synspec.dumps.
  flux_integrator`), the per-dump products of every dump (:func:`ppmpy.synspec.dumps.run_disc_dumps`) and their time
  series (:func:`ppmpy.synspec.dumps.collect_timeseries`).
* :func:`library_vs_analytic`, :func:`convention_check`: analytic anchors, which compare the pipeline with the
  toy's analytic truth (its line profiles and continuum; Lambert's law and the blueshift convention for a uniform
  outflow, the hidden hemisphere) instead of with references made by the same shared helpers.
* :func:`selftest`: builds a toy run and validates it (:func:`ppmpy.synspec.validate.run_validation`: V1-V6, the
  hold-out test, the brute force against the integrator and the stored products, EW conservation, the LPV
  comparison; the analytic anchors), plus, with nproc > 1, the equality of serial, 'fork' and 'spawn' runs;
  ``mutate`` injects one of the faults of :data:`MUTATIONS` into the pipeline under test, which must then fail
  (:data:`SELFTEST_EXPECT`). :func:`toy_tolerances` gives its tolerances for other n and nnode. Also
  ``python -m ppmpy.synspec.testing`` (:func:`main`).

The toy run
-----------
* Library dump (dump 0): its T_eff' field is the smooth random field mapped onto a uniform distribution over the
  library's range (rank transform; +-3 sigma_T, :func:`toy_teff_range`) with ``nnode`` bins of width dT = round(6
  sigma_T / nnode) [K] (52 K for 40 nodes): every bin holds ~n / nnode models, so the library has exactly ``nnode``
  nodes (``nmin`` 20) and no sparse tails; the end bins' models sit at the bin centres (no point of the library dump
  beyond the end nodes). Its models carry a random relative depth scatter (``noise``, default 0.3 %; FASTWIND's
  convergence noise was 0.5 % rms in EW), which the library averages and the exact sums keep (what V1 and V2
  measure, as on M424).
* Later dumps 1 .. ndumps: independent smooth T_eff' fields (rms 0.9 %, extremes 1.5-2.9 sigma) with one compact
  cool downflow each (:func:`toy_sphere` ``plume``), which puts 1-26 of 20 000 points per dump beyond the node range
  (clamped, V3; M424 <= 0.02 %), and velocity fields.
* Lines of sight: the 8 of Thompson et al. (2024); grid :data:`TOY_GRID` (dv 1, vmax 1000, vshift 300 km/s; quick)
  or the M424 grid (``quick=False``).

The toy's deviations are its own (node spacing, line shapes, n): :data:`TOY_TOLERANCES` replace the M424
:data:`ppmpy.synspec.validate.DEFAULT_TOLERANCES` for it (calibrated at nnode 40, n >= 8000; scaled by
:func:`toy_tolerances` for nnode 10-100 and n >= 3000).

Validation
----------
tests/synspec/test_testing.py: the self-test passes (seeds 0 and 1; n 3000, 3900, 8000, 20 000; nnode 10, 40, 100;
serially and with 2 workers, 'fork' and 'spawn', which equal the serial results bit for bit); every mutation of
:data:`MUTATIONS` fails the report, fails its 'fail' checks and passes its 'blind' ones (:data:`SELFTEST_EXPECT`);
faults of the shared helpers (los_velocity sign, F_c weight dropped, |mu| from project_los), monkeypatched into the
whole pipeline, fail the analytic anchors and nothing else; the parallel checks see a perturbed stored member; the
run directory is removed also when the run raises; the toy data have the stated properties (equal-area points,
field amplitudes and smoothness, the plume, exact zero depth beyond the taper, the node count, the members, order
and dtypes of the store, the toy library against the library built from the store). Measured (Trillium login node,
2026-10-01, n 20 000): selftest() 10 s serially (parent max RSS 0.57 GB), 17.5 s with nproc=2 (fork 2 s, spawn 4 s;
workers <= 0.41 GB); quick=False 39 s (1.35 GB); n = 8000 5 s, n 3000 2.5 s. Seeds 0-4 pass on both grids, every
check below 0.7 of its (scaled) tolerance for n 3000-20 000 and nnode 10-100.

PP 2026-10-01: new (synthetic data for users and the tests of ppmpy.synspec; the toy runs of test_validate.py and
test_dumps.py were written in the tests themselves).
"""
import atexit
import os
import shutil
import tempfile
import time
import traceback
import warnings

import numpy as np

from .conventions import C_KMS
from .spectral import LineSet, VelocityGrid

__all__ = ["toy_lineset", "toy_line_params", "toy_depth", "toy_continuum", "toy_teff_range", "toy_library",
           "toy_sphere", "toy_profiles_store", "toy_run", "library_vs_analytic", "convention_check", "toy_tolerances",
           "selftest", "main", "TOY_GRID", "TOY_TOLERANCES", "TOY_TOL_INTERP", "TOY_TOL_INTERP2", "TOY_TOL_NOISE",
           "TOY_N_CAL", "TOY_N_MIN", "TOY_NNODE_RANGE", "V5_NMINS", "MUTATIONS", "SELFTEST_EXPECT", "TEFF0",
           "TEFF_REL_RMS", "V_RMS", "TSCALE", "NODE_SPAN_SIGMA", "NROW", "TOY_LREF"]

TEFF0 = 38230.0
"""Reference T_eff [K] of the toy models (the M424 FASTWIND reference model)."""
TEFF_REL_RMS = 0.009
"""Relative rms of the toy T_eff' fields (M424 per-dump rms 338-345 K at 38 230 K)."""
V_RMS = 45.0
"""Rms of each toy velocity component [km/s] (M424: u_r std 46.7 km/s, disc rms v_los 44.6 km/s)."""
TSCALE = 1000.0
"""T_eff' scale [K] of the line-parameter polynomials."""
NODE_SPAN_SIGMA = 6.0
"""Width of the toy library's T_eff' range in units of the T_eff' rms (+-3 sigma, beyond the extremes of the smooth
fields; M424: the nodes end at +1.9 sigma on the hot side, where its T_eff' fields have no tail)."""
NROW = 161
"""Wavelength rows of a toy model (FASTWIND OUT files: 161)."""
TOY_LREF = (4026.22, 4199.90, 4921.93, 4471.48, 5875.62, 6678.15, 4541.59, 5411.52)
"""Velocity zero points [A] of the first toy lines (He I / He II lines; further lines at 4000 + 100 j A)."""
TOY_GRID = VelocityGrid(dv=1.0, vmax=1000.0, vshift=300.0)
"""Velocity grid of the quick toy run: 2001 points, Doppler shifts up to 300 km/s."""

TOY_TOLERANCES = dict(
    V1=1e-4, V1_dEW=2e-4, V2=2.5e-4, V2_dEW=4e-4, V3=6e-4, V4=4e-4, V5=1e-9, V6=3e-4, brute=2.5e-7, brute_f64=1e-10,
    ew=2e-6, library=3e-3, library_fc=3e-5, convention=1e-3)
"""Tolerances of the toy run at the calibration point nnode 40 (dT 52 K), n >= :data:`TOY_N_CAL` (names as
:data:`ppmpy.synspec.validate.DEFAULT_TOLERANCES`, plus those of the analytic anchors); :func:`toy_tolerances` scales
them for other nnode and n. The largest values of the self-test over seeds 0-4 on both grids (quick and full, n =
20 000) times 3-4. Measured maxima (2026-10-01): V1 2.6e-5, V1_dEW 5.5e-5 A, V2 7.3e-5 (hold-out; in-sample 5.1e-5),
V2_dEW 1.2e-4 A, V3 1.8e-4, V4 1.3e-4, V6 8.7e-5 (2000-point subsets; 5000 points with quick=False: 7.4e-5), brute vs
stored 3.0e-8 (float32 rounding), EW conservation 5.0e-7 (M424 grid; 1.2e-7 on TOY_GRID). V5 is exactly 0 while every
bin holds at least max(nmins) models (the toy's nodes are then its bins for every nmin, and V5 compares identical
integrators): :func:`selftest` uses only the nmins <= n // nnode (with 1, 5, 100 at n 3900 V5 was 1.6e-5, at n 8000 /
nnode 100 4.0e-6, deterministic). The analytic anchors (n 3000-20 000, nnode 10-100, both grids): library 1.3e-3
(row interpolation of the models at the line centres, ~1e-3, plus the depth scatter / sqrt(count); valid for >= 20
models per bin), library_fc 3.6e-6 (relative), convention 2.9e-4 (relative; shifts rounded to whole steps and the
second-order Doppler term of -c ln(1 - v/c)). The toy deviates 5-15 x more than M424 (V1 3.2e-6, V4 9e-7): fewer
points (20 000 vs 1.24 million) average its per-model noise less, and its nodes are 52 K apart (M424 12.5 K). Below
n 8000 the noise-driven checks grow faster than the tolerances (V1 1.08 x its tolerance at n 3000, seed 0) and
are scaled (:func:`toy_tolerances`)."""

TOY_TOL_INTERP = ("V1", "V1_dEW", "V2", "V2_dEW", "V3")
"""Toy tolerances partly driven by the T_eff' interpolation between nodes (and by the per-model scatter or the
extrapolation distance): scaled by max(1, dT / 52 K) (:func:`toy_tolerances`)."""
TOY_TOL_INTERP2 = ("V4",)
"""Toy tolerances driven by the interpolation error alone (~ dT^2): scaled by max(1, (dT / 52 K)^2)."""
TOY_TOL_NOISE = ("V1", "V1_dEW", "V2", "V2_dEW")
"""Toy tolerances driven by the per-model depth scatter: scaled by max(1, sqrt(TOY_N_CAL / n))."""
TOY_N_CAL = 8000
"""Number of points below which the noise-driven tolerances grow as 1 / sqrt(n) (the mutation expectations
:data:`SELFTEST_EXPECT` hold at n >= TOY_N_CAL, where nothing is scaled)."""
TOY_NNODE_RANGE = (10, 100)
"""Node counts for which the scaled toy tolerances were checked (:func:`selftest` warns outside)."""
TOY_N_MIN = 3000
"""Smallest n for which the toy tolerances were checked (:func:`selftest` warns below)."""
V5_NMINS = (1, 5, 100)
"""nmins of V5 in the self-test (those of fw_disc_dumps_validate.py), limited to n // nnode."""
_DT_CAL = 52.0               # node spacing [K] at which TOY_TOLERANCES were calibrated (nnode 40)
_NMIN = 20                   # models per node of the toy integrator (lib_nodes nmin, as M424)

MUTATIONS = dict(
    node_offset="node profiles offset by 1e-3 inside |y| <= vmax - vshift (the run's nodes and integrator)",
    node_missing="the middle node dropped (the run's nodes and integrator)",
    v_sign="the sign of the line-of-sight velocity flipped in the integrator",
    weights_swapped="T_eff' interpolation weights a and 1 - a swapped in the integrator",
    los_mixup="lines of sight 1 and 2 swapped in the projections the pipeline uses",
    misaligned="the samples permuted against the points (and the models' T_eff' labels of the hold-out)",
    f_offset="F offset by 1e-3 inside |y| <= vmax - vshift, F0 not",
)
"""Faults :func:`selftest` can inject into the pipeline under test (the integrator, its nodes, the samples or the
projections handed to the checks, and the hold-out's factory); the references (exact sums, library, stored products
of the intact pipeline) stay intact. The faults of the sensitivity table of :mod:`ppmpy.synspec.validate`, with
offsets of 1e-3 instead of 1e-5 (~10 x the toy's own V1, V2, V4 deviations; M424's are 3e-6-9e-6)."""

_LIB_ANCHORS = {"library_vs_analytic", "library_fc_vs_analytic"}

SELFTEST_EXPECT = dict(
    node_offset=dict(fail={"V1_flux", "V1_flux_dEW", "V2_holdout", "V2_holdout_dEW", "V2_insample", "V2_insample_dEW",
                           "V4", "V5", "brute_vs_stored"},
                     blind={"V3_extrap", "V6", "brute_vs_integrator", "ew_conservation", "convention"} | _LIB_ANCHORS),
    node_missing=dict(fail={"V5", "brute_vs_stored"},
                      blind={"V3_extrap", "V6", "brute_vs_integrator", "ew_conservation", "convention"}
                      | _LIB_ANCHORS),
    v_sign=dict(fail={"V1_flux", "V2_holdout", "V2_insample", "V4", "V5", "V6", "brute_vs_integrator", "convention"},
                blind={"V3_extrap", "brute_vs_stored", "ew_conservation"} | _LIB_ANCHORS),
    weights_swapped=dict(fail={"V1_flux", "V5", "brute_vs_integrator"},
                         blind={"brute_vs_stored", "ew_conservation", "convention"} | _LIB_ANCHORS),
    los_mixup=dict(fail={"V1_flux", "V1_flux_dEW", "brute_vs_stored", "convention"},
                   blind={"V2_holdout", "V2_holdout_dEW", "V2_insample", "V2_insample_dEW", "V3_extrap", "V4", "V5",
                          "V6", "brute_vs_integrator", "ew_conservation"} | _LIB_ANCHORS),
    misaligned=dict(fail={"V1_flux", "V1_flux_dEW", "V2_holdout", "V2_holdout_dEW", "V2_insample", "V2_insample_dEW",
                          "brute_vs_stored"},
                    blind={"V3_extrap", "V4", "V5", "V6", "brute_vs_integrator", "ew_conservation", "convention"}
                    | _LIB_ANCHORS),
    f_offset=dict(fail={"V1_flux", "V1_flux_dEW", "V2_holdout", "V2_holdout_dEW", "V2_insample", "V2_insample_dEW",
                        "V4", "V5", "V6", "brute_vs_integrator", "ew_conservation", "convention"},
                  blind={"V3_extrap", "brute_vs_stored"} | _LIB_ANCHORS),
)
"""What each mutation does to the self-test (the sensitivity table of :mod:`ppmpy.synspec.validate` on the toy):
'fail' = checks that fail (by at least ~2 x their tolerance at seeds 0, 1, 3 and n 8000, 20 000), 'blind' = checks that
cannot see the fault by construction and pass (they compare the faulty pipeline with itself, or the intact
references with each other). The others depend on the toy's scatter and are not asserted: the dEW of v_sign (O(v/c)
of the EW, 1.2-1.4 x the tolerance), V2 and the dEW checks of weights_swapped (1.2-6 x), V1-V4 of node_missing (a
missing node of a smooth library costs ~1e-5, below the toy's scatter; the exact V5 and the stored products see it).
V5 sees every fault of the integrator here because the toy's nodes do not depend on nmin (V5 = 0 when intact). The
analytic anchors: the library checks are blind to all of these (the library is an intact reference); 'convention'
sees the velocity sign (value 2), the swapped projections (the far hemisphere of line of sight 1 becomes visible) and
f_offset (the 1e-3 offset of F moves the depth centroid by ~0.3 %), and is blind to the node faults (a single
T_eff' interpolated from any nodes keeps the centroid theorem) and to the misalignment (its sample is its own). The
faults of the shared helpers themselves (project_los, los_velocity, the F_c weight), which the comparisons with
references made by the same helpers cannot see, are tested in tests/synspec/test_testing.py by monkeypatching them
(the anchors fail, every other check passes)."""

_TAPER = (0.6, 0.9)          # depth taper from 0.6 to 0.9 x (vmax - vshift): exactly 0 beyond
_ROW_DY0 = 2.6               # rows y = Y sinh(s u) / sinh(s), s such that they are 2.6 km/s apart at the centre
_ROW_SHIFT = 0.3             # per-point shift of the rows, in units of the smallest row spacing
_EPS = 1e-3                  # size of the offsets of the mutations (~10 x the toy's own V1 / V2 / V4 scatter)
_PLUME = (2.5, 5.0, 3.0)     # toy_sphere plume: sigma [deg], T_eff' and u_r drop at its centre [rms]


# ----------------------------------------------------------------------------------------------
# the analytic local models
# ----------------------------------------------------------------------------------------------
def toy_lineset(lines=3):
    """
    The toy lines.

    Parameters
    ----------
    lines: int or LineSet
        Number of lines (names 'L<lref>' with the zero points :data:`TOY_LREF`, then 4000 + 100 j A), or a LineSet
        (returned as it is).

    Returns
    -------
    LineSet
    """
    # PP 2026-10-01: new
    if isinstance(lines, LineSet):
        return lines
    nl = int(lines)
    if nl < 1:
        raise ValueError("need at least one line, got {!r}".format(lines))
    lref = [TOY_LREF[j] if j < len(TOY_LREF) else 4000.0 + 100.0 * j for j in range(nl)]
    return LineSet(["L{:d}".format(int(round(l))) for l in lref], lref)


def toy_line_params(lines=3, seed=0):
    """
    Parameters of the toy lines, drawn per line from ``np.random.default_rng([seed, j, 17])`` (so line j is the
    same for any number of lines).

    Returns
    -------
    list of dict
        Per line: lref [A]; depth A0 (0.25-0.5), a1 (-0.3-0.3), a2 (-0.03-0.03): A = A0 (1 + a1 x + a2 x^2) with
        x = (T_eff' - :data:`TEFF0`) / 1000 K; centre c0 (+-5 km/s) + c1 x (c1 +-3 km/s); Gaussian sigma0 (20-60
        km/s) (1 + s1 x) (s1 +-0.08); Lorentzian fraction eta (0.1-0.5) and half width gam (0.5-1 sigma0) (1 + s1 x);
        continuum fc0 (7.4e-7 x 0.8-1.2) (T_eff' / T0)^p (p 1-2) with a slope fslope (+-0.02) across the band.
    """
    # PP 2026-10-01: new (pseudo-Voigt He-line stand-ins; widths and depths of the M424 lines: FWHM 56-246 km/s)
    ls = toy_lineset(lines)
    out = []
    for j, lr in enumerate(ls.lref):
        r = np.random.default_rng([int(seed), j, 17])
        sig0 = r.uniform(20.0, 60.0)
        out.append(dict(lref=float(lr), A0=r.uniform(0.25, 0.5), a1=r.uniform(-0.3, 0.3), a2=r.uniform(-0.03, 0.03),
                        c0=r.uniform(-5.0, 5.0), c1=r.uniform(-3.0, 3.0), sig0=sig0, s1=r.uniform(-0.08, 0.08),
                        eta=r.uniform(0.1, 0.5), gam=sig0 * r.uniform(0.5, 1.0), fc0=7.4e-7 * r.uniform(0.8, 1.2),
                        p=r.uniform(1.0, 2.0), fslope=r.uniform(-0.02, 0.02)))
    return out


def _grid(grid):
    if grid is None:
        return TOY_GRID
    if isinstance(grid, dict):
        return VelocityGrid(**grid)
    if not isinstance(grid, VelocityGrid):
        raise TypeError("grid must be a VelocityGrid, got {}".format(type(grid).__name__))
    return grid


def _ymax(grid):
    """Y = vmax - vshift: the line depth must vanish beyond it (zero padding of DiscFlux)."""
    Y = grid.vmax - grid.vshift
    if not Y > 0:
        raise ValueError("vmax - vshift must be > 0, got {}".format(grid))
    return Y


def toy_depth(teff, y, par, grid=None):
    """
    Absorption depth 1 - F/F_c of a toy line.

    Parameters
    ----------
    teff: float or array-like
        T_eff' [K], scalar or (N,).
    y: array-like
        Rest-frame velocity [km/s]: (ny,) for all T_eff' (result (N, ny)), or (N, nrow) per T_eff'.
    par: dict
        One entry of :func:`toy_line_params`.
    grid: VelocityGrid, optional
        Sets the taper of the pseudo-Voigt depth: taper factor 1 for |y| <= 0.6 Y, a cosine taper to 0 at 0.9 Y
        and exactly 0 beyond, with Y = vmax - vshift (default :data:`TOY_GRID`).

    Returns
    -------
    np.ndarray
        float64 depth, broadcast shape of (teff[:, None], y).
    """
    # PP 2026-10-01: new (zero depth beyond 0.9 (vmax - vshift): the zero padding of disc.DiscFlux)
    Y = _ymax(_grid(grid))
    t = np.asarray(teff, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = (t - TEFF0) / TSCALE
    if x.ndim == 1:
        x = x[:, None]
    A = par["A0"] * (1.0 + par["a1"] * x + par["a2"] * x * x)
    s = par["sig0"] * (1.0 + par["s1"] * x)
    g = par["gam"] * (1.0 + par["s1"] * x)
    u = y - (par["c0"] + par["c1"] * x)
    prof = (1.0 - par["eta"]) * np.exp(-0.5 * (u / s) ** 2) + par["eta"] / (1.0 + (u / g) ** 2)
    ay = np.abs(y)
    y1, y2 = _TAPER[0] * Y, _TAPER[1] * Y
    with np.errstate(invalid="ignore"):
        w = np.where(ay <= y1, 1.0, np.where(ay >= y2, 0.0, 0.5 * (1.0 + np.cos(np.pi * (ay - y1) / (y2 - y1)))))
    return np.where(w > 0.0, A * prof * w, 0.0)


def toy_continuum(teff, par, y=None, grid=None):
    """
    Continuum flux of a toy line: fc0 (T_eff' / T0)^p (1 + fslope y / Y), Y = vmax - vshift.

    Parameters
    ----------
    teff: float or array-like
        T_eff' [K], scalar or (N,).
    par: dict
        One entry of :func:`toy_line_params`.
    y: array-like, optional
        Rest-frame velocity [km/s]: (nrow,) or (N, nrow). Default -Y, the first frequency point of the band, whose
        F_c the pipeline uses as the disc-integration weight (``fcont[:, :, 0]``; the library's ``fc``).
    grid: VelocityGrid, optional
        Sets Y (default :data:`TOY_GRID`).

    Returns
    -------
    np.ndarray
        float64, shape of teff (y None) or broadcast of (teff[:, None], y).
    """
    # PP 2026-10-01: new (weight F_c = fcont[:, :, 0], as fw_disc.library and fw_disc_los.py)
    Y = _ymax(_grid(grid))
    base = par["fc0"] * (np.asarray(teff, dtype=np.float64) / TEFF0) ** par["p"]
    if y is None:
        return base * (1.0 - par["fslope"])
    y = np.asarray(y, dtype=np.float64)
    if base.ndim == 1:
        base = base[:, None]
    return base * (1.0 + par["fslope"] * y / Y)


def toy_teff_range(nnode=40, teff0=TEFF0, teff_rel_rms=TEFF_REL_RMS):
    """
    T_eff' range and bin width of a toy library: ``nnode`` bins of dT = max(1, round(6 sigma / nnode)) K
    (sigma = teff_rel_rms teff0, :data:`NODE_SPAN_SIGMA`), starting at a multiple of dT (so that
    :func:`ppmpy.synspec.library.teff_edges` of T_eff' inside the range gives exactly these bins).

    Returns
    -------
    lo, hi, dT: float
        [K]; hi = lo + nnode dT (default 37180, 39260, 52).
    """
    # PP 2026-10-01: new
    nnode = int(nnode)
    if nnode < 2:
        raise ValueError("need at least 2 nodes, got {!r}".format(nnode))
    dT = max(1.0, float(round(NODE_SPAN_SIGMA * teff_rel_rms * teff0 / nnode)))
    lo = float(np.floor((teff0 - 0.5 * nnode * dT) / dT) * dT)
    return lo, lo + nnode * dT, dT


def toy_library(nnode=40, lines=3, grid=None, seed=0, teff0=TEFF0, teff_rel_rms=TEFF_REL_RMS, count=25):
    """
    An analytic flux library: ``nnode`` filled T_eff' bins (:func:`toy_teff_range`), each with the noise-free toy
    profile (:func:`toy_depth`) and continuum (:func:`toy_continuum` at the first frequency point, as the library's
    F_c) at its centre.

    Parameters
    ----------
    nnode: int
        Bins, each with ``count`` models, so :func:`ppmpy.synspec.library.lib_nodes` (nmin <= count) gives
        ``nnode`` nodes.
    lines: int or LineSet
        :func:`toy_lineset`.
    grid: VelocityGrid, optional
        Grid of the profiles (default :data:`TOY_GRID`).
    seed: int
        Line parameters (:func:`toy_line_params`).
    teff0, teff_rel_rms: float
        Centre and rms that set the node range.
    count: int
        Models per bin (the library's ``count``).

    Returns
    -------
    FluxLibrary
        prof float32 (nnode, nl, ny) as the legacy files, fc float64; ``params`` record the grid (ny, y0, y1) and
        lref, so :func:`ppmpy.synspec.dumps.flux_integrator` takes it as it is (``DiscFlux(lib_nodes(lib))`` works
        too).
    """
    # PP 2026-10-01: new
    from .library import FluxLibrary
    grid = _grid(grid)
    ls = toy_lineset(lines)
    pars = toy_line_params(ls, seed)
    lo, hi, dT = toy_teff_range(nnode, teff0, teff_rel_rms)
    edges = lo + dT * np.arange(int(nnode) + 1)
    tc = 0.5 * (edges[:-1] + edges[1:])
    prof = np.empty((tc.size, len(ls), grid.ny), np.float32)
    fc = np.empty((tc.size, len(ls)))
    for j, p in enumerate(pars):
        prof[:, j] = 1.0 - toy_depth(tc, grid.y, p, grid)
        fc[:, j] = toy_continuum(tc, p, grid=grid)
    params = dict(source="synspec.testing.toy_library", nnode=int(nnode), seed=int(seed), dT=dT, lref=ls.lref.tolist(),
                  names=list(ls.names), ny=int(grid.ny), y0=float(grid.y[0]), y1=float(grid.y[-1]))
    return FluxLibrary(edges, tc, np.full(tc.size, float(count)), prof, fc, dT, params=params)


# ----------------------------------------------------------------------------------------------
# the sphere
# ----------------------------------------------------------------------------------------------
def _smooth_field(xyz, rng, kmax=4.0, nmodes=24, chunk=1 << 16):
    """
    A smooth random field on the unit sphere: sum of ``nmodes`` plane waves cos(k . r + phase) with random
    directions, |k| uniform in [1, kmax] and amplitudes 1 / |k|, standardised to mean 0 and rms 1 over the points.
    Elementwise arithmetic only (no BLAS), so the values do not depend on threads or chunks.
    """
    # PP 2026-10-01: new
    x, y, z = xyz
    kdir = rng.standard_normal((nmodes, 3))
    kdir /= np.sqrt((kdir ** 2).sum(axis=1))[:, None]
    kmag = rng.uniform(1.0, kmax, nmodes)
    ph = rng.uniform(0.0, 2.0 * np.pi, nmodes)
    K = kdir * kmag[:, None]
    amp = 1.0 / kmag
    f = np.empty(x.size)
    for i0 in range(0, x.size, chunk):
        s = slice(i0, i0 + chunk)
        arg = x[s, None] * K[:, 0] + y[s, None] * K[:, 1] + z[s, None] * K[:, 2] + ph
        f[s] = (np.cos(arg) * amp).sum(axis=1)
    f -= f.mean()
    return f / f.std()


def toy_sphere(npoints, seed=0, dump=0, teff0=TEFF0, teff_rel_rms=TEFF_REL_RMS, v_rms=V_RMS, teff_range=None,
               plume=True, kmax=4.0, nmodes=24, t_s=None):
    """
    Equal-area points with smooth random T_eff' and velocity fields (one toy dump).

    Parameters
    ----------
    npoints: int
        Points (:func:`ppmpy.synspec.sphere.fibonacci_sphere`), at least 2 (ValueError otherwise: the fields are
        standardised to rms 1 over the points).
    seed, dump: int
        The fields are drawn from ``np.random.default_rng([seed, dump, 11])``: other dumps of a run use other
        ``dump`` numbers.
    teff0, teff_rel_rms: float
        Mean T_eff' [K] and its relative rms (defaults: M424-like, 38 230 K and 0.9 %).
    v_rms: float
        Rms of each velocity component [km/s] (default 45).
    teff_range: (float, float), optional
        Map the T_eff' field onto a uniform distribution over (lo, hi) instead (rank transform: the i-th smallest
        value becomes lo + (hi - lo)(i + 1/2) / N; still smooth on the sphere). The toy library dump uses this, so
        that every library bin is equally filled.
    plume: bool
        Add one compact cool downflow (a plume) at a random position, Gaussian in angle with sigma 2.5 deg: T_eff'
        down by 5 rms and u_r down by 3 v_rms (135 km/s) at its centre (not with ``teff_range``). The smooth fields
        alone have lighter tails than a Gaussian (extremes 1.5-2.9 rms per dump); M424's T_eff' has a long cool
        tail (relT down to -7.4 %, ~-8 rms) and a few points per dump beyond the library's node range (<= 0.02 %):
        the plume puts ~0.03 % of the points beyond the toy's +-3 rms nodes.
    kmax, nmodes: float, int
        Largest wavenumber [1/R] and number of plane waves of each field (:func:`_smooth_field`): structures of
        ~R / kmax and larger.
    t_s: float, optional
        Time [s] (default dump x 2835 s, the M424 dump spacing).

    Returns
    -------
    dict
        theta, phi [rad], teff [K], ur, uth, uph [km/s]: float64 (N,); t_s (float), dump (int). Usable wherever
        a sample mapping is accepted (teff, ur, uth, uph).
    """
    # PP 2026-10-01: new (amplitudes of the M424 samples: sphere_sample.py, meta.json of d3200_r4050_N1236544)
    from .sphere import fibonacci_sphere, sphere_xyz
    if int(npoints) < 2:
        # PP 2026-10-01: reviewer: one point gave NaN fields (rms 0 in the standardisation)
        raise ValueError("toy_sphere needs at least 2 points (the fields are standardised over them), got {!r}".format(
            npoints))
    theta, phi = fibonacci_sphere(npoints)
    xyz = sphere_xyz(theta, phi)
    rng = np.random.default_rng([int(seed), int(dump), 11])
    g = [_smooth_field(xyz, rng, kmax, nmodes) for _ in range(4)]
    if plume and teff_range is None:
        c = rng.standard_normal(3)
        c /= np.sqrt((c ** 2).sum())
        cosa = np.clip(xyz[0] * c[0] + xyz[1] * c[1] + xyz[2] * c[2], -1.0, 1.0)
        shape = np.exp(-0.5 * (np.arccos(cosa) / np.radians(_PLUME[0])) ** 2)
        g[0] = g[0] - _PLUME[1] * shape
        g[1] = g[1] - _PLUME[2] * shape
    if teff_range is None:
        teff = teff0 * (1.0 + teff_rel_rms * g[0])
    else:
        lo, hi = float(teff_range[0]), float(teff_range[1])
        if not hi > lo:
            raise ValueError("teff_range must be (lo, hi) with hi > lo, got {!r}".format(teff_range))
        rank = np.empty(theta.size)
        rank[np.argsort(g[0], kind="stable")] = np.arange(theta.size)
        teff = lo + (hi - lo) * (rank + 0.5) / theta.size
    return dict(theta=theta, phi=phi, teff=teff, ur=v_rms * g[1], uth=v_rms * g[2], uph=v_rms * g[3],
                t_s=float(2835.0 * dump if t_s is None else t_s), dump=int(dump))


# ----------------------------------------------------------------------------------------------
# the per-point models
# ----------------------------------------------------------------------------------------------
def _row_stretch(Y, nrow, dy0=_ROW_DY0):
    """s > 0 with Y s / sinh(s) 2 / (nrow - 1) = dy0 (central spacing of Y sinh(s u) / sinh(s)); 0 = uniform rows."""
    target = dy0 * (nrow - 1) / (2.0 * Y)                # = s / sinh(s), decreasing from 1 at s = 0
    if target >= 1.0:
        return 0.0
    lo, hi = 1e-6, 50.0
    for _ in range(100):                                  # bisection, to ~1e-14
        mid = 0.5 * (lo + hi)
        if mid / np.sinh(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _rows(grid, nrow, shift):
    """
    Wavelength rows of the models in velocity: (N, nrow) float64. Y sinh(s u) / sinh(s) (u uniform in [-1, 1],
    Y = vmax - vshift), 2.6 km/s apart at the line centre on any grid (TOY_GRID: s = 3.0, ~25 km/s in the far wings;
    M424 grid: s = 4.6), shifted per point by shift x 0.3 of the central spacing.
    """
    Y = _ymax(grid)
    u = np.linspace(-1.0, 1.0, nrow)
    s = _row_stretch(Y, nrow)
    base = Y * u if s == 0.0 else Y * np.sinh(s * u) / np.sinh(s)
    return base[None, :] + shift[:, None] * _ROW_SHIFT * np.diff(base).min()


def toy_profiles_store(path, sphere, lines=3, grid=None, seed=0, nrow=NROW, noise=0.003, block=20000, meta=True):
    """
    Write the per-point models of a toy dump as an uncompressed profiles.npz-like file.

    Every point gets its own model: the toy lines (:func:`toy_depth`, :func:`toy_continuum`) at its own T_eff' on
    its own wavelength rows (``nrow`` rows spanning +-(vmax - vshift), denser at the line centre, shifted per point
    by up to 0.3 of the smallest row spacing: FASTWIND's frequency grids differ from model to model), with a random
    relative depth scatter ``noise`` per model and line.

    Parameters
    ----------
    path: str or os.PathLike
        Output .npz (written member by member through a hidden temporary, then renamed).
    sphere: mapping
        theta, phi, teff (N,) of the points (:func:`toy_sphere`); ur (optional) is copied as ur_kms.
    lines: int or LineSet
        :func:`toy_lineset`.
    grid: VelocityGrid, optional
        Sets the band and the taper (default :data:`TOY_GRID`).
    seed: int
        Line parameters (:func:`toy_line_params`); the row shifts and the depth scatter are drawn from
        ``default_rng([seed, N, 23])``.
    nrow: int
        Rows per model (FASTWIND: 161).
    noise: float
        Relative rms of the per-model depth scatter (0 for exact models).
    block: int
        Points computed at a time (memory: ~10 block x nl x nrow x 8 bytes; the file does not depend on it).
    meta: bool
        Add a '_meta' member (provenance: kind 'synspec.toy_profiles', the parameters).

    Returns
    -------
    str
        The path. Members in the order of :func:`ppmpy.synspec.fwresults.combine` (the M424 profiles.npz) and with
        its dtypes: idx int32, teff float64, status 'ok' ('<U12', the width of M424's, whose failed points carry
        longer statuses), niter int16, T_tau23 float64, t_pnlte, t_formal float32, lam [A, air], fcont, fnorm
        float32 (N, nl, nrow), r (1.0), theta, phi, x, y, z, ur_kms, relT float64, teff_nudge float32 (0), lines,
        then '_meta'; open it with :meth:`ppmpy.synspec.fwresults.ProfileStore.open`.
    """
    # PP 2026-10-01: new (the members and dtypes of fw_sphere_merge.py --combine, written as fwresults.combine;
    # reviewer: 'lines' after the row members and status '<U12', as M424's profiles.npz)
    from .fwresults import _meta_array, _NpzStream
    from .io import make_meta
    from .sphere import sphere_xyz
    grid = _grid(grid)
    ls = toy_lineset(lines)
    pars = toy_line_params(ls, seed)
    teff = np.asarray(sphere["teff"], dtype=np.float64)
    theta = np.asarray(sphere["theta"], dtype=np.float64)
    phi = np.asarray(sphere["phi"], dtype=np.float64)
    N, nl, nrow = teff.size, len(ls), int(nrow)
    if teff.shape != (N,) or theta.shape != (N,) or phi.shape != (N,) or N == 0:
        raise ValueError("sphere needs theta, phi, teff of one shape (N,), N >= 1")
    if nrow < 4:
        raise ValueError("nrow must be >= 4")
    rng = np.random.default_rng([int(seed), N, 23])
    shift = rng.uniform(-1.0, 1.0, N)
    eps = noise * rng.standard_normal((N, nl))
    _ymax(grid)                                          # grid check: vmax > vshift
    block = max(1, int(block))

    def blocks(kind):
        for i0 in range(0, N, block):
            s = slice(i0, min(N, i0 + block))
            yr = _rows(grid, nrow, shift[s])
            out = np.empty((yr.shape[0], nl, nrow), np.float32)
            for j, p in enumerate(pars):
                if kind == "lam":
                    out[:, j] = p["lref"] * np.exp(yr / C_KMS)
                elif kind == "fcont":
                    out[:, j] = toy_continuum(teff[s], p, yr, grid)
                else:
                    out[:, j] = 1.0 - toy_depth(teff[s], yr, p, grid) * (1.0 + eps[s, j, None])
            yield out

    x, y, z = sphere_xyz(theta, phi)
    ur = np.asarray(sphere["ur"], dtype=np.float64) if "ur" in sphere else np.zeros(N)
    small = [("idx", np.arange(N, dtype=np.int32)), ("teff", teff), ("status", np.full(N, "ok", dtype="<U12")),
             ("niter", np.full(N, 61, np.int16)), ("T_tau23", 1.04 * teff), ("t_pnlte", np.full(N, 150.0, np.float32)),
             ("t_formal", np.full(N, 0.5, np.float32))]
    rest = [("r", np.ones(N)), ("theta", theta), ("phi", phi), ("x", x), ("y", y), ("z", z), ("ur_kms", ur),
            ("relT", teff / TEFF0 - 1.0), ("teff_nudge", np.zeros(N, np.float32)), ("lines", np.array(ls.names))]
    w = _NpzStream(os.fspath(path))
    try:
        for k, v in small:
            w.write_array(k, v)
        for kind in ("lam", "fcont", "fnorm"):
            with w.member(kind, (N, nl, nrow), np.float32) as fid:
                for b in blocks(kind):
                    fid.write(b)
        for k, v in rest:
            w.write_array(k, v)
        if meta:
            w.write_array("_meta", _meta_array(make_meta(
                "synspec.toy_profiles", params=dict(seed=int(seed), npoints=int(N), lines=ls.to_dict(),
                                                    grid=grid.to_dict(), nrow=nrow, noise=float(noise),
                                                    line_params=pars))))
        w.close()
    except BaseException:
        w.abort()
        raise
    return os.fspath(path)


def _write_sample(path, s):
    """A per-dump sample file as sphere_sample.py writes it (float32 teff, ur, uth, uph; t_s)."""
    # PP 2026-10-01: new (the layout of sphere_sample.py: float32 teff, ur, uth, uph and t_s)
    from .io import save_npz
    arrays = {k: np.asarray(s[k], dtype=np.float32) for k in ("teff", "ur", "uth", "uph")}
    arrays["t_s"] = np.float64(s["t_s"])
    return save_npz(path, arrays)


# ----------------------------------------------------------------------------------------------
# a whole toy run
# ----------------------------------------------------------------------------------------------
def toy_run(root, n=20000, nnode=40, seed=0, grid=None, lines=3, ndumps=3, nmin=20, noise=0.003,
            los="thompson2024", block=2500, stride=4, nproc=1, start_method=None, log=None):
    """
    A toy per-point run in a directory, made with the production code.

    The library dump 0 (:func:`toy_sphere` with ``teff_range`` = :func:`toy_teff_range`, the end bins' models moved
    to the bin centres) gets its per-point models (:func:`toy_profiles_store`); its exact sums and library come from
    :func:`ppmpy.synspec.disc.integrate_exact_stream`, the integrator from :func:`ppmpy.synspec.dumps.flux_integrator`
    (``nnode`` nodes while n / nnode >= nmin); every dump 0 .. ndumps (later dumps: :func:`toy_sphere` with the
    plume) is integrated by :func:`ppmpy.synspec.dumps.run_disc_dumps` and collected by
    :func:`ppmpy.synspec.dumps.collect_timeseries`. The sample files hold float32 (as sphere_sample.py writes them).

    Parameters
    ----------
    root: str or os.PathLike
        Directory (created): profiles.npz, samples/dNNNN.npz, disc_los.npz, library.npz, flux/dNNNN.npz,
        flux_timeseries.npz.
    n, nnode, seed: int
        Points, library nodes (:func:`toy_teff_range`), seed of everything.
    grid: VelocityGrid, optional
        Default :data:`TOY_GRID`.
    lines: int or LineSet
        :func:`toy_lineset`.
    ndumps: int
        Later dumps (1 .. ndumps) besides the library dump 0.
    nmin: int
        Models per node (:func:`ppmpy.synspec.library.lib_nodes`).
    noise: float
        Per-model depth scatter of :func:`toy_profiles_store`.
    los: str or array-like
        Lines of sight (default the 8 of Thompson et al. 2024).
    block, stride: int
        :func:`ppmpy.synspec.disc.integrate_exact_stream` (exact sums and library of dump 0).
    nproc, start_method:
        Workers of the exact sums and of :func:`ppmpy.synspec.dumps.run_disc_dumps` (results independent of them).
    log: callable, optional

    Returns
    -------
    dict
        root, profiles, samples (directory), sample0 (path), dumps (0 .. ndumps), later (1 .. ndumps), library,
        exact (disc_los.npz), timeseries (paths); theta, phi (N,), los, grid, lines (LineSet), dT, nmin, teff_range
        (lo, hi), nodes (:class:`ppmpy.synspec.library.LibraryNodes`), integ (the flux integrator,
        :func:`ppmpy.synspec.dumps.flux_integrator`), block, stride, wall (per stage) [s].
    """
    # PP 2026-10-01: new (the M424 sequence: fw_sphere_extract / FASTWIND -> fw_disc_los.py -> fw_disc_dumps.py ->
    # fw_disc_collect.py, with the synspec ports)
    from .disc import integrate_exact_stream, save_disc_los
    from .dumps import _blas_limit, collect_timeseries, flux_integrator, run_disc_dumps
    from .library import FluxLibrary, lib_nodes
    T0 = time.time()
    wall = {}
    grid = _grid(grid)
    ls = toy_lineset(lines)
    root = os.path.abspath(os.fspath(root))
    sdir = os.path.join(root, "samples")
    os.makedirs(sdir, exist_ok=True)
    lo, hi, dT = toy_teff_range(nnode)
    s0 = toy_sphere(n, seed=seed, dump=0, teff_range=(lo, hi))
    # the end bins' models at their bin centres, so that the library dump has no point beyond the end nodes (a
    # uniform distribution would put half of each end bin there, clamped: an error of the toy, not of the method)
    t = s0["teff"]
    t[t < lo + dT] = lo + 0.5 * dT
    t[t >= hi - dT] = hi - 0.5 * dT
    prof = toy_profiles_store(os.path.join(root, "profiles.npz"), s0, ls, grid, seed, noise=noise)
    dumps = list(range(int(ndumps) + 1))
    for d in dumps:
        s = s0 if d == 0 else toy_sphere(n, seed=seed, dump=d)
        _write_sample(os.path.join(sdir, "d{:04d}.npz".format(d)), s)
    wall["data"] = time.time() - T0
    t = time.time()
    ex = integrate_exact_stream(prof, os.path.join(sdir, "d0000.npz"), ls, los=los, grid=grid, dT=dT, checks=False,
                                block=block, stride=stride, nproc=nproc, start_method=start_method, log=log)
    exact = save_disc_los(os.path.join(root, "disc_los.npz"), ex)
    libp = ex["library"].save(os.path.join(root, "library.npz"))
    wall["exact"] = time.time() - t
    t = time.time()
    lib = FluxLibrary.load(libp)
    with _blas_limit(1):
        nodes = lib_nodes(lib, nmin=nmin)
    integ = flux_integrator(libp, nmin=nmin, grid=grid)
    run_disc_dumps(dumps, sdir, root, "flux", flux_integrator, (libp,), s0["theta"], s0["phi"], los, nproc=nproc,
                   start_method=start_method, lref=ls, factory_kwargs=dict(nmin=nmin, grid=grid), log=log)
    ts = collect_timeseries(root, "flux")
    wall["dumps"] = time.time() - t
    return dict(root=root, profiles=prof, samples=sdir, sample0=os.path.join(sdir, "d0000.npz"), dumps=dumps,
                later=dumps[1:], library=libp, exact=exact, timeseries=ts, theta=s0["theta"], phi=s0["phi"], los=los,
                grid=grid, lines=ls, dT=dT, nmin=int(nmin), teff_range=(lo, hi), nodes=nodes, integ=integ,
                block=int(block), stride=int(stride), wall=wall)


# ----------------------------------------------------------------------------------------------
# analytic anchors (truths of the toy that do not pass through the pipeline's shared helpers)
# ----------------------------------------------------------------------------------------------
def library_vs_analytic(library, lines=3, grid=None, seed=0, tolerance=None, fc_tolerance=None, tolerances=None):
    """
    A library built from toy models (:func:`toy_profiles_store`, e.g. by
    :func:`ppmpy.synspec.disc.integrate_exact_stream` as in :func:`toy_run`) against the toy's analytic lines at the
    bins' mean T_eff'.

    Parameters
    ----------
    library: FluxLibrary, str or os.PathLike
        The library (bins without models are skipped).
    lines, grid, seed:
        The toy's lines (:func:`toy_lineset`), velocity grid (default :data:`TOY_GRID`) and line parameters
        (:func:`toy_line_params`), as given to :func:`toy_profiles_store`.
    tolerance, fc_tolerance: float, optional
        Largest |f_lib - f| [continuum] and |F_c,lib / F_c - 1|; default ``tolerances['library']``,
        ``tolerances['library_fc']``, else :data:`TOY_TOLERANCES`.
    tolerances: dict, optional

    Returns
    -------
    list of CheckResult
        'library_vs_analytic': max |prof - (1 - :func:`toy_depth` (tmean))| over the filled bins, lines and the grid
        (the bin means of the models' profiles, interpolated from their rows, vs the analytic line at the mean
        T_eff': row interpolation ~1e-3 at the line centres, plus the depth scatter / sqrt(count));
        'library_fc_vs_analytic' (unit 'relative'): max |fc / :func:`toy_continuum` (tmean) - 1| (the first
        frequency point, ``fcont[:, :, 0]``). Details: per_line, lines, the bin of the maximum, nb.

    Notes
    -----
    Independent of the disc integration: catches a library whose binning, T_eff' labels, row interpolation or
    F_c (e.g. a constant weight in place of fcont[:, :, 0], invisible to the checks that compare the exact sums
    with the library made by the same code) differ from the models.
    """
    # PP 2026-10-01: new (reviewer: the self-test compared the pipeline only with references made by the same shared
    # helpers; this check uses the toy's analytic truth)
    from .library import FluxLibrary
    from .validate import CheckResult
    grid = _grid(grid)
    ls = toy_lineset(lines)
    pars = toy_line_params(ls, seed)
    lib = library if isinstance(library, FluxLibrary) else FluxLibrary.load(os.fspath(library))
    if lib.ny != grid.ny or lib.nl != len(ls):
        raise ValueError("library has {} lines x {} grid points, the toy {} x {}".format(lib.nl, lib.ny, len(ls),
                                                                                        grid.ny))
    tol = dict(TOY_TOLERANCES, **(tolerances or {}))
    tol_p = float(tol["library"] if tolerance is None else tolerance)
    tol_f = float(tol["library_fc"] if fc_tolerance is None else fc_tolerance)
    filled = np.flatnonzero(lib.count > 0)
    if filled.size == 0:
        raise ValueError("library has no filled bin")
    tm = lib.tmean[filled]
    dprof = np.empty((filled.size, len(ls)))
    dfc = np.empty((filled.size, len(ls)))
    for j, p in enumerate(pars):
        ana = 1.0 - toy_depth(tm, grid.y, p, grid)
        dprof[:, j] = np.abs(np.asarray(lib.prof[filled, j], dtype=np.float64) - ana).max(axis=1)
        dfc[:, j] = np.abs(lib.fc[filled, j] / toy_continuum(tm, p, grid=grid) - 1.0)
    names = list(ls.names)
    out = []
    for name, d, t, unit in (("library_vs_analytic", dprof, tol_p, "continuum"),
                             ("library_fc_vs_analytic", dfc, tol_f, "relative")):
        b = int(np.unravel_index(np.argmax(d), d.shape)[0]) if np.all(np.isfinite(d)) else -1
        out.append(CheckResult(name, float(np.max(d)) if np.all(np.isfinite(d)) else np.nan, t, unit=unit,
                               details=dict(lines=names, per_line=d.max(axis=0), nb=int(filled.size),
                                            worst_bin=None if b < 0 else int(filled[b]),
                                            worst_tmean=None if b < 0 else float(tm[b]),
                                            note="library bins vs the toy's analytic {} at the bins' mean T_eff'"
                                            .format("profiles" if unit == "continuum" else "continuum F_c"))))
    return out


def convention_check(integ, mu, tn, pn, theta, phi, los, v0=100.0, teff=TEFF0, teff_far=None, grid=None,
                     tolerance=None, tolerances=None, lref=None):
    """
    Sign and hemisphere conventions of a disc integrator with given projections, against analytic values.

    For every line of sight k, with v from :func:`ppmpy.synspec.sphere.los_velocity` (looked up at call time) and
    the given projections (those of the pipeline under test, :func:`ppmpy.synspec.sphere.project_los`):

    * uniform outflow u_r = v0, u_theta = u_phi = 0, every point at ``teff``: the integrator's weighted mean
      velocity (``vmean``, its third output) must be +2/3 v0 (I(mu) = const, weights mu: int mu^2 / int mu over the
      visible hemisphere), and the centroid of the depth 1 - F must lie 2/3 v0 below that of 1 - F0 (v > 0 towards
      the observer is a blueshift, lambda (1 - v/c); the profile is the rest-frame one convolved with the weighted
      shift distribution, so the centroids differ by its mean, whatever the line shape);
    * the hemisphere: the same flow with the points of the far hemisphere at ``teff_far``, the hemisphere taken
      from mu_true = n . l_k evaluated here from theta, phi and the line of sight (independently of the given
      projections): F and F0 must equal those of the first case (hidden points contribute nothing).

    Parameters
    ----------
    integ: object
        DiscFlux duck type (``pairs``, ``__call__(mu, v, k0, k1, a, novel)`` -> (F, F0, vmean, ...); ``grid``).
    mu, tn, pn: array-like
        (nlos, N) projections of the points.
    theta, phi: array-like
        (N,) the points [rad].
    los: str or array-like
        The lines of sight of the projections (:func:`ppmpy.synspec.sphere.project_los`).
    v0: float
        Outflow speed [km/s] (default 100; must stay below the grid's vshift).
    teff, teff_far: float
        T_eff' of the visible and of the far hemisphere (default TEFF0 and TEFF0 - 1000 K).
    grid: VelocityGrid, optional
        Default ``integ.grid``.
    tolerance: float, optional
        Default ``tolerances['convention']``, else :data:`TOY_TOLERANCES`.
    lref: LineSet, optional
        Line names of the details.

    Returns
    -------
    CheckResult
        'convention' (unit 'relative'): max over lines of sight and lines of |vmean / (2/3 v0) - 1|,
        |dc / (-2/3 v0) - 1| (dc = centroid of 1 - F minus that of 1 - F0) and max |F_split - F|, |F0_split - F0|
        relative to the largest depth of F0. Details: per_line, vmean_rel, centroid_rel, hemisphere (nlos, nl),
        vmean, dc [km/s], expected (+2/3 v0), n_clip.

    Notes
    -----
    Intact (toy, n 3000-20 000, both grids): 2.2e-4-2.9e-4, from the centroid (the shifts rounded to whole grid
    steps, and their second-order Doppler term, -c ln(1 - v/c) ~ v (1 + v / 2c)); vmean ~4e-6 (the Fibonacci
    quadrature of int mu^2 / int mu); the hemisphere part is exactly 0. A flipped velocity sign gives 2, a visible far
    hemisphere (|mu|) or the projections of another line of sight a hemisphere part of 0.1-1 (toy:
    :data:`TOY_TOLERANCES` 'convention' 1e-3).
    """
    # PP 2026-10-01: new (reviewer: faults in the shared helpers project_los / los_velocity passed every check of the
    # self-test; this check compares the pipeline with analytic values)
    from .sphere import _los_vectors, los_velocity
    from .validate import CheckResult
    MU, TN, PN = (np.asarray(p, dtype=np.float64) for p in (mu, tn, pn))
    if MU.ndim == 1:
        MU, TN, PN = MU[None], TN[None], PN[None]
    th, ph = np.asarray(theta, dtype=np.float64), np.asarray(phi, dtype=np.float64)
    L = _los_vectors(los)
    if MU.shape != (L.shape[0], th.size) or TN.shape != MU.shape or PN.shape != MU.shape:
        raise ValueError("projections must be (nlos, N) = ({}, {}), got {}".format(L.shape[0], th.size, MU.shape))
    grid = _grid(getattr(integ, "grid", None) if grid is None else grid)
    v0 = float(v0)
    if not 0.0 < abs(v0) < grid.vshift:
        raise ValueError("v0 must be nonzero and below vshift = {} km/s, got {}".format(grid.vshift, v0))
    teff = float(teff)
    teff_far = teff - 1000.0 if teff_far is None else float(teff_far)
    tol = dict(TOY_TOLERANCES, **(tolerances or {}))
    tol = float(tol["convention"] if tolerance is None else tolerance)
    N = th.size
    st = np.sin(th)
    nx, ny, nz = st * np.cos(ph), st * np.sin(ph), np.cos(th)
    ur, zero = np.full(N, v0), np.zeros(N)
    near = integ.pairs(np.full(N, teff))
    y = grid.y
    vexp = 2.0 * v0 / 3.0
    rel_v, rel_c, hemi, vm_all, dc_all, ncl = [], [], [], [], [], []
    for k in range(MU.shape[0]):
        v = los_velocity(ur, zero, zero, MU[k], TN[k], PN[k])
        out = integ(MU[k], v, *near, novel=True)
        F, F0 = np.asarray(out[0], dtype=np.float64), np.asarray(out[1], dtype=np.float64)
        D, D0 = 1.0 - F, 1.0 - F0
        dc = (D * y).sum(axis=-1) / D.sum(axis=-1) - (D0 * y).sum(axis=-1) / D0.sum(axis=-1)
        vm = np.asarray(out[2], dtype=np.float64) if len(out) > 2 else np.full(F.shape[0], np.nan)
        mtrue = nx * L[k, 0] + ny * L[k, 1] + nz * L[k, 2]
        far = integ.pairs(np.where(mtrue > 0.0, teff, teff_far))
        outs = integ(MU[k], v, *far, novel=True)
        dh = np.maximum(np.abs(np.asarray(outs[0], dtype=np.float64) - F).max(axis=-1),
                        np.abs(np.asarray(outs[1], dtype=np.float64) - F0).max(axis=-1))
        rel_v.append(np.abs(vm / vexp - 1.0))
        rel_c.append(np.abs(dc / -vexp - 1.0))
        hemi.append(dh / D0.max(axis=-1))
        vm_all.append(vm)
        dc_all.append(dc)
        ncl.append(int(out[4]) if len(out) > 4 else -1)
    rel_v, rel_c, hemi = np.array(rel_v), np.array(rel_c), np.array(hemi)
    allv = np.maximum(np.maximum(rel_v, rel_c), hemi)
    # NaN stays NaN (fails): np.maximum propagates it
    value = float(np.max(allv)) if np.all(np.isfinite(allv)) else np.nan
    names = getattr(lref, "names", None) or getattr(integ, "names", None)
    nl = allv.shape[1]
    names = [str(n) for n in names] if names is not None and len(names) == nl else ["line{}".format(j)
                                                                                    for j in range(nl)]
    return CheckResult("convention", value, tol, unit="relative",
                       details=dict(lines=names, per_line=allv.max(axis=0), vmean_rel=rel_v, centroid_rel=rel_c,
                                    hemisphere=hemi, vmean=np.array(vm_all), dc=np.array(dc_all), expected=vexp,
                                    v0=v0, teff=teff, teff_far=teff_far, n_clip=ncl,
                                    note="uniform outflow v0: vmean = +2/3 v0, depth centroid moved by -2/3 v0 "
                                         "(blueshift); far hemisphere (true mu <= 0) at teff_far changes nothing"))


# ----------------------------------------------------------------------------------------------
# mutations of the pipeline under test
# ----------------------------------------------------------------------------------------------
class _Mutant:
    """An integrator wrapper: every attribute of the wrapped integrator (t, fc, grid, nl, lref, node_params, ...)."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, k):
        if k == "_inner":
            raise AttributeError(k)
        return getattr(self._inner, k)

    def pairs(self, teff, mode="clamp"):
        return self._inner.pairs(teff, mode=mode)

    def __call__(self, mu, v, k0, k1, a, novel=True):
        return self._inner(mu, v, k0, k1, a, novel=novel)


class _VSign(_Mutant):
    def __call__(self, mu, v, k0, k1, a, novel=True):
        return self._inner(mu, -np.asarray(v), k0, k1, a, novel=novel)


class _SwapWeights(_Mutant):
    def pairs(self, teff, mode="clamp"):
        k0, k1, a = self._inner.pairs(teff, mode=mode)
        return k0, k1, 1.0 - a


class _FOffset(_Mutant):
    """F of one integrator (offset nodes), F0 and the rest of the other (intact)."""

    def __init__(self, inner, bad):
        _Mutant.__init__(self, inner)
        self._bad = bad

    def __call__(self, mu, v, k0, k1, a, novel=True):
        out = self._inner(mu, v, k0, k1, a, novel=novel)
        return (self._bad(mu, v, k0, k1, a, novel=False)[0],) + tuple(out[1:])


def _offset_nodes(nodes, grid):
    from .library import LibraryNodes
    win = (np.abs(grid.y) <= _ymax(grid)).astype(np.float64)
    return LibraryNodes(nodes.t, nodes.count, nodes.prof + _EPS * win, nodes.fc, groups=nodes.groups,
                        params=nodes.params)


def _drop_node(nodes):
    from .library import LibraryNodes
    keep = np.arange(nodes.nn) != nodes.nn // 2
    return LibraryNodes(nodes.t[keep], nodes.count[keep], nodes.prof[keep], nodes.fc[keep],
                        groups=[g for g, k in zip(nodes.groups, keep) if k], params=nodes.params)


def _mutated(mutate, nodes, grid, lref):
    """(integrator, nodes) of the pipeline under test built from intact nodes, with the fault ``mutate``."""
    # PP 2026-10-01: new (the faults of test_validate.py's sensitivity tests, as a selftest option)
    from .disc import DiscFlux

    def flux(nd):
        f = DiscFlux(nd, grid)
        f.lref = lref
        return f
    good = flux(nodes)
    if mutate is None or mutate in ("los_mixup", "misaligned"):
        return good, nodes
    if mutate == "node_offset":
        nd = _offset_nodes(nodes, grid)
        return flux(nd), nd
    if mutate == "node_missing":
        nd = _drop_node(nodes)
        return flux(nd), nd
    if mutate == "v_sign":
        return _VSign(good), nodes
    if mutate == "weights_swapped":
        return _SwapWeights(good), nodes
    if mutate == "f_offset":
        return _FOffset(good, flux(_offset_nodes(nodes, grid))), nodes
    raise ValueError("unknown mutation {!r} (one of {})".format(mutate, sorted(MUTATIONS)))


def _holdout_factory(mutate, nmin, grid, lref):
    """The hold-out's integrator factory (FluxLibrary -> integrator) with the fault, None for the default."""
    if mutate is None or mutate in ("los_mixup", "misaligned"):
        return None
    from .library import lib_nodes

    def factory(L):
        return _mutated(mutate, lib_nodes(L, nmin=nmin), grid, lref)[0]
    factory.__qualname__ = "toy_factory[{}]".format(mutate)
    return factory


def _series(integ, samples, dumps, proj, grid, lref):
    """The time series of a pipeline in memory: F, F0 (nd, nlos, nl, ny) float32 (as the stored products), Y, LREF."""
    from .sphere import los_velocity
    MU, TN, PN = proj
    F = F0 = None
    for i, d in enumerate(dumps):
        s = samples(d)
        k0, k1, a = integ.pairs(s["teff"])
        for k in range(MU.shape[0]):
            out = integ(MU[k], los_velocity(s["ur"], s["uth"], s["uph"], MU[k], TN[k], PN[k]), k0, k1, a)
            if F is None:
                F = np.zeros((len(dumps), MU.shape[0]) + np.shape(out[0]), np.float32)
                F0 = np.zeros_like(F)
            F[i, k], F0[i, k] = out[0], out[1]
    return dict(F=F, F0=F0, Y=grid.y, LREF=np.asarray(lref.lref))


# ----------------------------------------------------------------------------------------------
# serial = fork = spawn
# ----------------------------------------------------------------------------------------------
def _maxdiff(a, b):
    """max |a - b| for a bit-for-bit comparison: equal values (NaN at the same place, equal infinities) count 0, a NaN
    against a number inf; another shape inf."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return np.inf
    if not a.size:
        return 0.0
    a, b = np.atleast_1d(a), np.atleast_1d(b)                 # 0-d members (dump, t_s, ...)
    with np.errstate(invalid="ignore"):
        d = np.abs(a - b)
    d[(a == b) | (np.isnan(a) & np.isnan(b))] = 0.0
    d[np.isnan(d)] = np.inf
    return float(d.max())


def _numeric(a):
    return a.dtype.kind in "biuf"


def _npz_maxdiff(pa, pb):
    """Largest difference over the numeric members common to two .npz files (a member of another shape: inf), and
    the members compared."""
    out, names = 0.0, []
    with np.load(pa) as a, np.load(pb) as b:
        for k in sorted(set(a.files) & set(b.files)):
            x, y = a[k], b[k]
            if _numeric(x) and _numeric(y):
                out = max(out, _maxdiff(x, y))
                names.append(k)
    return out, names


_LIB_MEMBERS = ("prof", "fc", "count", "tmean", "edges")


def _parallel_checks(run, proj, nproc, start_methods, log=None):
    """
    The results of the parallel paths with 'fork' and 'spawn' vs the serial ones, bit for bit: the exact sums and the
    library (integrate_exact_stream; prof, fc, count, tmean, edges), every numeric member of the per-dump products
    (run_disc_dumps: F, F0, diag_F, vmean_w, sigma_w, n_clip, ...) and the brute force.
    """
    # PP 2026-10-01: new; reviewer: every numeric member of the per-dump products and the library's count, tmean,
    # edges (only F, F0, diag_F and prof, fc were compared)
    import multiprocessing as mp
    from .disc import integrate_exact_stream
    from .dumps import dump_path, flux_integrator, load_sample, run_disc_dumps
    from .library import FluxLibrary
    from .validate import CheckResult, brute_force
    checks = []
    with np.load(run["exact"]) as z:
        Fx, F0x = z["F"], z["F0"]
    L0 = FluxLibrary.load(run["library"])
    d1 = run["later"][0]
    s1 = load_sample(run["samples"], d1)
    Fb, F0b = brute_force(run["nodes"], s1, *proj, run["grid"])
    for m in start_methods:
        if m not in mp.get_all_start_methods():
            checks.append(CheckResult("parallel_" + m, np.nan, None,
                                      details=dict(note="start method {!r} not available here".format(m))))
            continue
        t = time.time()
        ex = integrate_exact_stream(run["profiles"], run["sample0"], run["lines"], los=run["los"], grid=run["grid"],
                                    dT=run["dT"], checks=False, block=run["block"], stride=run["stride"], nproc=nproc,
                                    start_method=m)
        Lp = ex["library"]
        d_ex = max([_maxdiff(ex["F"], Fx), _maxdiff(ex["F0"], F0x)]
                   + [_maxdiff(getattr(Lp, k), getattr(L0, k)) for k in _LIB_MEMBERS])
        name = "flux_" + m
        run_disc_dumps(run["dumps"], run["samples"], run["root"], name, flux_integrator, (run["library"],),
                       run["theta"], run["phi"], run["los"], nproc=nproc, start_method=m, lref=run["lines"],
                       factory_kwargs=dict(nmin=run["nmin"], grid=run["grid"]))
        d_dumps, members = 0.0, set()
        for d in run["dumps"]:
            dd, names = _npz_maxdiff(dump_path(run["root"], "flux", d), dump_path(run["root"], name, d))
            d_dumps = max(d_dumps, dd)
            members.update(names)
        Fp, F0p = brute_force(run["nodes"], s1, *proj, run["grid"], nproc=nproc, start_method=m)
        d_br = max(_maxdiff(Fp, Fb), _maxdiff(F0p, F0b))
        wall = time.time() - t
        if log is not None:
            log("parallel {} ({} workers, {:.1f} s): exact sums {:.1e}, per-dump products {:.1e}, brute force "
                "{:.1e}".format(m, nproc, wall, d_ex, d_dumps, d_br))
        checks.append(CheckResult("parallel_" + m, max(d_ex, d_dumps, d_br), 0.0,
                                  details=dict(nproc=int(nproc), exact=d_ex, dumps=d_dumps, brute=d_br, wall=wall,
                                               dump_members=sorted(members), library_members=list(_LIB_MEMBERS),
                                               note="{} workers vs serial, bit for bit: exact sums and library, "
                                                    "every numeric member of the per-dump products, brute "
                                                    "force".format(m))))
    return checks


# ----------------------------------------------------------------------------------------------
# the self-test
# ----------------------------------------------------------------------------------------------
def toy_tolerances(n=20000, nnode=40, tolerances=None):
    """
    The tolerances of a toy self-test of ``n`` points and ``nnode`` nodes: :data:`TOY_TOLERANCES` (calibrated at
    nnode 40, dT 52 K, and n >= 8000) scaled with the node spacing dT (:func:`toy_teff_range`) and with n, then the
    explicit ``tolerances`` (used as given).

    * :data:`TOY_TOL_INTERP2` (V4, the interpolation error alone, ~dT^2) x max(1, (dT / 52 K)^2);
    * :data:`TOY_TOL_INTERP` (V1, V2, their dEW, V3: interpolation plus the per-model scatter, resp. the
      extrapolation distance beyond the end nodes) x max(1, dT / 52 K);
    * :data:`TOY_TOL_NOISE` (V1, V2, their dEW) x max(1, sqrt(:data:`TOY_N_CAL` / n)) (the scatter averages as
      ~1 / sqrt(n)).

    Measured on intact runs (2026-10-01, quick grid; value / unscaled tolerance): nnode 10 (dT 206 K): V1 1.16, V3
    1.88, V4 1.28 (V4 x 21 against nnode 40, dT x 4); n 3000: V1 up to 1.08, V2_dEW 0.99 (seeds 0-4). With the
    scaling every check stays below 0.7 of its tolerance for nnode 10-100 and n 3000-20 000.

    Returns
    -------
    tol: dict
    scale: dict
        interp (dT / 52, >= 1), interp2 (its square), noise (the n factor); dT [K].
    """
    # PP 2026-10-01: new (reviewer: the tolerances were calibrated at nnode 40 only, and nnode 10 failed V1, V3, V4 at
    # up to 1.9 x tolerance on an intact pipeline; the scaling factors fit the measured growth, a single dT^2 factor
    # would have made V1 at nnode 10 blind to the 1e-3 node faults)
    n, nnode = int(n), int(nnode)
    dT = toy_teff_range(nnode)[2]
    s_i = max(1.0, dT / _DT_CAL)
    s_n = max(1.0, float(np.sqrt(TOY_N_CAL / max(n, 1))))
    tol = dict(TOY_TOLERANCES)
    for k in TOY_TOL_INTERP:
        tol[k] = tol[k] * s_i
    for k in TOY_TOL_INTERP2:
        tol[k] = tol[k] * s_i * s_i
    for k in TOY_TOL_NOISE:
        tol[k] = tol[k] * s_n
    tol.update(tolerances or {})
    return tol, dict(interp=s_i, interp2=s_i * s_i, noise=s_n, dT=dT)


def selftest(n=20000, nnode=40, seed=0, nproc=1, quick=True, mutate=None, tolerances=None,
             start_methods=("fork", "spawn"), workdir=None, keep=False, log=None):
    """
    Build a toy run (:func:`toy_run`) and validate it with :func:`ppmpy.synspec.validate.run_validation`: V1-V6,
    the hold-out test (V2), the brute force (against the integrator and the stored per-dump products), EW
    conservation and the LPV comparison; the analytic anchors (:func:`library_vs_analytic`,
    :func:`convention_check`), which compare the pipeline with the toy's analytic truth instead of references made
    by the same shared helpers; with ``nproc`` > 1 also the equality, bit for bit, of the parallel paths with
    'fork' and 'spawn' and the serial ones (exact sums and library, per-dump products, brute force).

    Parameters
    ----------
    n, nnode, seed: int
        Points, library nodes, seed (:func:`toy_run`). Every node must keep its models: n >= 20 nnode (ValueError
        otherwise; the integrator merges bins below nmin 20). Calibrated and checked for nnode
        :data:`TOY_NNODE_RANGE` (node spacing 21-206 K) and n >= 3000; outside these a UserWarning (the tolerances
        are scaled with the node spacing and n, :func:`toy_tolerances`, but not verified there).
    nproc: int
        > 1: run the parallel checks with this many workers ('parallel_fork', 'parallel_spawn'). The run itself and
        the other checks are serial.
    quick: bool
        True: :data:`TOY_GRID` (2001 points), 3 later dumps, V4 / V6 on dumps 1, 2 (2000-point subsets), the brute
        force on dump 1. False: the M424 grid (5401 points), 6 later dumps, V4 / V6 on dumps 1-4 (5000 points), the
        brute force on dumps 1, 2.
    mutate: str, optional
        Inject a fault of :data:`MUTATIONS` into the pipeline under test; the report must then fail
        (:data:`SELFTEST_EXPECT`). The references (exact sums, library, stored products) stay intact.
    tolerances: dict, optional
        Overrides of the toy tolerances (:func:`toy_tolerances`; used as given, not scaled).
    start_methods: sequence of str
        Start methods of the parallel checks (one not available here gives an informational NaN check).
    workdir: str, optional
        Parent of the temporary run directory (default ``tempfile.gettempdir()``).
    keep: bool
        Keep the run directory (``report.meta['selftest']['root']``), also when the self-test raises; default: removed,
        also on an error (after the local variables of the failed frames are cleared, so that no memory map of the
        run's files stays open; the traceback keeps its lines).
    log: callable, optional

    Returns
    -------
    ValidationReport
        Checks V1_flux, V1_flux_dEW, V2_holdout, V2_holdout_dEW, V2_insample, V2_insample_dEW, V3_extrap,
        V3_leaveout (info), V4, V5, V6, brute_vs_integrator, brute_vs_stored, ew_conservation, library_vs_analytic,
        library_fc_vs_analytic, convention (+ parallel_fork, parallel_spawn); meta: those of run_validation (ran,
        skipped, tolerances, lpv) plus 'selftest' (parameters, nmins of V5, the tolerance scale factors, node
        range, dumps, stage times, peak RSS of this process; 'root' when kept, 'leftover' when the directory could
        not be removed) and 'warnings' (messages recorded during the checks, e.g. of the LPV comparison, which flags
        but does not fail). ``report.passed()`` is the verdict.

    Notes
    -----
    V5 rebuilds the nodes with nmin 1, 5 and 100 and is exactly 0 for the toy while every bin keeps at least that
    many models: only the nmins <= n // nnode are used (``meta['selftest']['nmins']``; V5 is skipped when none is
    left). Measured on a Trillium login node (2026-10-01): default (n 20 000, quick) 10 s, max RSS 0.57 GB;
    nproc=2 17.5 s (workers <= 0.41 GB); quick=False 40 s, 1.35 GB; n 8000 5 s. Writes ~6 kB per point (n 20 000:
    119 MB, of which profiles.npz 113 MB) to the temporary directory, removed at the end (on NFS after closing the
    memory maps of its files; a directory that cannot be removed gives a UserWarning, ``meta['selftest']
    ['leftover']`` and another attempt at exit). With nproc > 1 and 'spawn', a calling script needs the
    ``if __name__ == "__main__":`` guard.
    """
    # PP 2026-10-01: new; reviewer: nmins of V5 limited to n // nnode, tolerances scaled with dT and n, the analytic
    # anchors, cleanup on errors
    T0 = time.time()
    if mutate is not None and mutate not in MUTATIONS:
        raise ValueError("unknown mutation {!r} (one of {})".format(mutate, sorted(MUTATIONS)))
    n, nnode = int(n), int(nnode)
    toy_teff_range(nnode)                                 # ValueError for nnode < 2
    if n // nnode < _NMIN:
        raise ValueError("selftest needs n >= {} nnode (every node keeps at least nmin = {} models), got n {} for "
                         "nnode {}".format(_NMIN, _NMIN, n, nnode))
    if not TOY_NNODE_RANGE[0] <= nnode <= TOY_NNODE_RANGE[1] or n < TOY_N_MIN:
        warnings.warn("selftest: n {} / nnode {} outside the checked range (nnode {}-{}, n >= {}): the scaled "
                      "tolerances are not verified there".format(n, nnode, TOY_NNODE_RANGE[0], TOY_NNODE_RANGE[1],
                                                                 TOY_N_MIN), UserWarning, stacklevel=2)
    grid = TOY_GRID if quick else VelocityGrid()
    ndumps = 3 if quick else 6
    check_dl = [1, 2] if quick else [1, 2, 3, 4]
    brute_dl = [1] if quick else [1, 2]
    nsub = 2000 if quick else 5000
    tol, scale = toy_tolerances(n, nnode, tolerances)
    nmins = [m for m in V5_NMINS if m <= n // nnode]
    cfg = dict(n=n, nnode=nnode, seed=int(seed), nproc=int(nproc), quick=bool(quick), mutate=mutate, grid=grid,
               ndumps=ndumps, check_dumps=check_dl, brute_dumps=brute_dl, nsub=nsub, nmins=nmins, tol_scale=scale)
    root = tempfile.mkdtemp(prefix="synspec_selftest_", dir=workdir)
    try:
        report = _selftest_run(root, cfg, tol, start_methods, log)
    except BaseException as exc:
        if not keep:
            # PP 2026-10-01: reviewer: the frames of the traceback kept the memory maps of the run's files (NFS) open,
            # so rmtree failed silently and the directory stayed behind
            traceback.clear_frames(exc.__traceback__)
            _cleanup(root)
        raise
    leftover = None if keep else _cleanup(root)
    report.meta["selftest"]["root"] = root if keep else None
    report.meta["selftest"]["leftover"] = leftover
    report.meta["selftest"]["wall"] = time.time() - T0
    try:
        import resource
        report.meta["selftest"]["maxrss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except (ImportError, AttributeError):
        pass
    if log is not None:
        log("selftest: {} ({:.1f} s)".format("PASSED" if report.passed() else "FAILED",
                                             report.meta["selftest"]["wall"]))
    return report


def _remove_tree(root, tries=5):
    """Remove the run directory; on NFS a file still memory-mapped somewhere leaves a '.nfs*' entry until it is
    closed, so collect garbage (closing such maps) and retry a few times. True when the directory is gone."""
    import gc
    for i in range(tries):
        shutil.rmtree(root, ignore_errors=True)
        if not os.path.exists(root):
            return True
        gc.collect()
        time.sleep(0.1 * (i + 1))
    return False


def _cleanup(root):
    """Remove the run directory: None when it is gone, else the path, after a UserWarning and with another attempt
    registered for the interpreter's exit."""
    import gc
    gc.collect()
    if _remove_tree(root):
        return None
    warnings.warn("selftest: could not remove the run directory {} (files still open?); it is retried at exit, or "
                  "remove it by hand".format(root), UserWarning, stacklevel=3)
    atexit.register(shutil.rmtree, root, True)
    return root


def _selftest_run(root, cfg, tol, start_methods, log):
    """The body of :func:`selftest` in ``root`` (its local references, e.g. memory maps of the run's files, are
    released when it returns, before the directory is removed)."""
    # PP 2026-10-01: new
    from .dumps import load_sample
    from .sphere import project_los
    from .validate import ValidationReport, run_validation, teff_ranges, v3_select
    grid, mutate, seed = cfg["grid"], cfg["mutate"], cfg["seed"]
    run = toy_run(root, n=cfg["n"], nnode=cfg["nnode"], seed=seed, grid=grid, ndumps=cfg["ndumps"], nmin=_NMIN)
    stage = dict(run["wall"])
    t = time.time()
    ls = run["lines"]
    proj = project_los(run["theta"], run["phi"], run["los"])
    integ, nodes = _mutated(mutate, run["nodes"], grid, ls.lref)
    samples = {d: load_sample(run["samples"], d) for d in run["dumps"]}
    holdout = dict(profiles=run["profiles"], sample=run["sample0"], theta=None, phi=None, los=run["los"], grid=grid,
                   lref=ls, nmin=run["nmin"], dT=run["dT"], block=run["block"], stride=run["stride"])
    factory = _holdout_factory(mutate, run["nmin"], grid, ls.lref)
    if factory is not None:
        holdout["factory"] = factory
    pproj = proj
    if mutate == "los_mixup":
        order = np.arange(proj[0].shape[0])
        order[[0, 1]] = [1, 0]
        pproj = tuple(p[order] for p in proj)
    elif mutate == "misaligned":
        from .fwresults import ProfileStore
        perm = np.random.default_rng([seed, 29]).permutation(run["theta"].size)
        samples = {d: dict(s, **{k: s[k][perm] for k in ("teff", "ur", "uth", "uph")}) for d, s in samples.items()}
        st = ProfileStore.open(run["profiles"])
        holdout.update(profiles=dict({k: st[k] for k in st.keys()}, teff=samples[0]["teff"]), sample=samples[0],
                       check_teff=False)
    ts = run["timeseries"] if mutate is None else _series(integ, samples.__getitem__, run["dumps"], pproj, grid, ls)
    with np.load(run["profiles"]) as z:
        tmod = z["teff"]
    v3 = v3_select(teff_ranges(run["samples"], run["later"], trange=(tmod.min(), tmod.max())))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        rv = run_validation(integ, *pproj, sample=samples[0], exact=run["exact"], library=run["library"], nodes=nodes,
                            samples=samples.__getitem__, dumps=cfg["check_dumps"], v3_dumps=v3, lref=ls, grid=grid,
                            nsub=cfg["nsub"], nmins=tuple(cfg["nmins"]), holdout=holdout,
                            brute_dumps=cfg["brute_dumps"], brute_kw=dict(stored=(run["root"], "flux")),
                            timeseries=ts, tolerances=tol, log=log)
    checks = list(rv.checks)
    # the analytic anchors: the library made by the production code vs the toy's lines, the integrator under test
    # with the pipeline's projections vs Lambert's law and the blueshift convention
    checks += library_vs_analytic(run["library"], ls, grid, seed, tolerances=tol)
    checks.append(convention_check(integ, *pproj, run["theta"], run["phi"], run["los"], grid=grid, tolerances=tol,
                                   lref=ls))
    stage["checks"] = time.time() - t
    if cfg["nproc"] > 1:
        t = time.time()
        checks += _parallel_checks(run, proj, cfg["nproc"], start_methods, log=log)
        stage["parallel"] = time.time() - t
    node_t = run["nodes"].t
    info = dict(cfg, grid=grid.to_dict(), lines=ls.to_dict(), dT=run["dT"], nmin=run["nmin"], nodes=int(node_t.size),
                node_range=[float(node_t[0]), float(node_t[-1])], teff_range=list(run["teff_range"]),
                v3_dumps=[int(d) for d in v3], stage_wall=stage)
    meta = dict(rv.meta, selftest=info, warnings=["{}: {}".format(w.category.__name__, w.message) for w in rec])
    return ValidationReport(checks, meta=meta, arrays=rv.arrays, data=rv.data)


def main(argv=None):
    """Command line: ``python -m ppmpy.synspec.testing [--n N] [--nnode K] [--seed S] [--nproc P] [--full]
    [--mutate NAME] [--keep] [--workdir DIR] [--json FILE]``; prints the report table, exit status 0 if it passed."""
    import argparse
    ap = argparse.ArgumentParser(prog="python -m ppmpy.synspec.testing", description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--nnode", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--full", action="store_true", help="quick=False (the M424 grid, more dumps)")
    ap.add_argument("--mutate", choices=sorted(MUTATIONS))
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--workdir")
    ap.add_argument("--json", help="also write the report as JSON")
    a = ap.parse_args(argv)
    rep = selftest(n=a.n, nnode=a.nnode, seed=a.seed, nproc=a.nproc, quick=not a.full, mutate=a.mutate,
                   workdir=a.workdir, keep=a.keep, log=print)
    print(rep.table())
    if a.json:
        rep.to_json(a.json)
    return 0 if rep.passed() else 1


if __name__ == "__main__":
    raise SystemExit(main())
