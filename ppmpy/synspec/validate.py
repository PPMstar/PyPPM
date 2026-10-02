"""
Validation of the disc integration: the checks V1-V6 of the all-dump method, the hold-out test, a brute-force
per-point sum, EW conservation, and a report that can be printed and saved.

The all-dump method (:class:`ppmpy.synspec.disc.DiscFlux`; the intensity method likewise) replaces every point's own
local model by a linear T_eff' interpolation between library nodes and rounds the Doppler shifts to whole grid steps.
The checks measure what that costs, as max |dF| over the lines of sight and the velocity grid, per line, in units of
the continuum (dEW in Angstrom, EW conservation relative):

========  ===========================  =======================================================================
V1        :func:`v1_exact`             the library's own dump through the pipeline vs the exact per-point sums
                                       (every point's own model; :func:`ppmpy.synspec.disc.integrate_exact_stream`),
                                       with (F) and without (F0) velocities; and the EW difference
V2        :func:`v2_holdout`           the points split at random into halves A, B; the exact sums of each half vs
                                       the library built from the OTHER half (the situation of every later dump:
                                       T_eff' values whose own models were never computed); the in-sample
                                       prediction (library of all points) for reference
V3        :func:`v3_tails`             points beyond the node range (clamped to the end node): clamping vs linear
                                       extrapolation (and, informational, vs leaving them out)
V4        :func:`v4_nearest`           linear T_eff' interpolation vs the nearest library bin
V5        :func:`v5_node_merging`      other node merging (nmin) vs the run's nodes
V6        :func:`v6_rounding`          Doppler shifts rounded to grid steps vs continuous shifts (direct per-point
                                       sum over a random subset of visible points)
brute     :func:`brute_force_check`    the full per-point sum with the same rounding, done directly (each point's
                                       interpolated node profile shifted and added, no FFT) vs the integrator
                                       (float64) and vs stored per-dump products (float32)
brute_imu :func:`brute_force_imu_check`
                                       the same for the intensity method (each point's intensities interpolated in
                                       T_eff' and s, weighted by mu, shifted with the edge values, added), on a
                                       random subset vs the integrator or on all points also vs stored products;
                                       the profiles, the velocity moments [km/s] and the clipped counts
EW        :func:`ew_conservation`      whole-step Doppler shifts conserve the EW on the y grid
========  ===========================  =======================================================================

Every check function returns a :class:`ValidationReport` (one or more :class:`CheckResult`: name, value =
the largest deviation, tolerance, passed, details with the per-line values) whose ``arrays`` hold the numbers under
the member names of the legacy products (validate.npz, holdout.npz), so ``report.to_npz`` writes a superset of them.
:func:`run_validation` runs the checks that the given inputs allow and compares every deviation with the run's own
line-profile variability (LPV residual rms) when a time series is given. A NaN in any compared profile makes the
check's value NaN, which fails; a check with nothing to check (no dumps, no nmin) raises ValueError, and a report
without checks does not pass.

Sensitivity (what each check sees; tests/synspec/test_validate.py injects these faults into a synthetic run and
asserts exactly this table). Every check gets the integrator under test and the nodes it was built from, as for a
real run; V2 gets the faulty integrator as its ``factory``; 'stored' is the brute force vs per-dump products made by
the intact pipeline:

==================================  ====  ====  ====  ====  ====  ====  =====  ======  ====
fault in the pipeline               V1    V2    V3    V4    V5    V6    brute  stored  EW
==================================  ====  ====  ====  ====  ====  ====  =====  ======  ====
node profiles offset by 1e-5        x     x     -     x     x     -     -      x       -
one node missing                    x     x     -     x     x     -     -      x       -
v sign flipped in the integrator    x     x     -     x     x     x     x      -       -
T_eff' weights a, 1 - a swapped     x     x     (c)   x     x     x     x      -       -
lines of sight mixed up             x     (a)   -     -     -     -     -      x       -
points misaligned with the models   x     x     -     -     -     -     -      x       -
F offset by 1e-5, F0 not            x     x     -     x     x     (b)   x      -       x
==================================  ====  ====  ====  ====  ====  ====  =====  ======  ====

(a) V2 projects its own lines of sight for both its exact sums and its predictions, so they cannot be mixed up
there; (b) not when the offset is below V6's own rounding scatter (M424 3e-5, the toy 4e-4); (c) through the points
beyond the nodes, which the faulty weights clamp to the wrong node (the toy: 7 x its intact V3). A corrupted
in-sample library passed to :func:`v2_holdout` (``library``) fails only V2_insample. V1 and V2 compare the pipeline
with sums over the per-point models themselves (at the level of the float32 library, ~1e-8 on the toy). V3-V6
compare the integrator with variants of itself or with its own nodes (they quantify method choices; a fault shared
by both sides cancels): V6 and the brute force see faults of the integration (shifts, weights, histograms, FFT), not
of the nodes; V4 and V5 see a fault only above their own scatter (M424: V4 9e-7, V5 3e-7, V6 3e-5). The brute force
vs stored products sees wrong inputs of a re-run (nodes, samples, projections). EW conservation compares F with F0
of the same integrator, so it sees only faults that treat them differently. V3 is no fault detector: it measures
what the clamped tails may cost, and fails when T_eff' lies far beyond the nodes.

The references are not fully independent of the pipeline: the exact sums
(:func:`ppmpy.synspec.disc.integrate_exact_stream`), the brute force and the pipeline share the projections
(:func:`ppmpy.synspec.sphere.project_los`), the line-of-sight velocity (:func:`ppmpy.synspec.sphere.los_velocity`)
and the shift rounding (:meth:`~ppmpy.synspec.spectral.VelocityGrid.shift_steps`), and V6 and the brute force share
:func:`ppmpy.synspec.library.node_pairs` with DiscFlux, so a sign or convention error there cancels in every
comparison. The test suite checks these conventions analytically (a point at the disc centre moving towards the
observer is blueshifted by its speed; at theta = 60 deg a motion along theta_hat projects with theta_hat . n). On
the toy whose library dump has every model at its bin centre (a = 0 at the nodes) a fault of the interpolation
weights that keeps a = 0 and 1 (e.g. a -> a^2) would not show; a second toy with T_eff' spread within the bins and
profiles linear in T_eff' shows it (V1, V2 and the brute force fail).

Applicability: V4, V5, V6 and the brute force assume a flux integrator DiscFlux(lib_nodes(library, nmin, smooth,
corr)) (``t`` and ``fc`` of its nodes). V4 compares it with the nearest bin of the flux library (a node correction
``corr`` must be passed again; with smoothed nodes V4 measures smoothing and interpolation together and is
informational), V5 rebuilds nodes from the library with the integrator's smoothing (a correction must be passed), V6
and the brute force take the integrator's nodes (checked: the same t, and fc where the integrator has one). An
intensity integrator differs from the flux library by its method (~1e-3): it gets V1 (and V2 through a factory, V3)
and its own brute force (:func:`brute_force_imu_check`); :func:`run_validation` skips V4-V6 and the flux brute force
for integrators without ``fc``.

Conventions
-----------
* Integrators have the duck type of :class:`ppmpy.synspec.disc.DiscFlux`: ``t`` (increasing node T_eff'),
  ``pairs(teff) -> (k0, k1, a)``, ``__call__(mu, v, k0, k1, a, novel=True) -> (F, F0, ...)`` with F, F0 (nl, ny);
  optional ``grid`` (:class:`~ppmpy.synspec.spectral.VelocityGrid`; else pass ``grid``) and ``lines`` / ``lref``
  (else pass ``lref``). :class:`ppmpy.synspec.disc.DiscImu` qualifies (any mode; :func:`ppmpy.synspec.dumps.
  imu_integrator` can attach the lref the legacy intensity library lacks), and so does the frozen legacy
  fw_disc.DiscImu (with grid and lref given).
* Projections mu, tn, pn (nlos, N) come from :func:`ppmpy.synspec.sphere.project_los` ('matmul' for the all-dump
  products, as fw_disc_dumps_validate.py); v = u . n > 0 towards the observer (:func:`los_velocity`).
* Samples: a directory of per-dump sphere samples (:func:`ppmpy.synspec.dumps.load_sample`), a mapping dump ->
  sample (mapping with teff, ur, uth, uph), or a callable dump -> sample; arrays are used as float64 (exact for
  the stored float32).
* Deviations are max over lines of sight and grid, per line (``np.abs(A - B).max(axis=(0, 2))``, the legacy
  dmax); a check's value is the largest over its lines (and over F and F0, halves, dumps), NaN if any is NaN.

Tolerances
----------
:data:`DEFAULT_TOLERANCES` are the recorded M424 values (:data:`RECORDED_M424`) rounded up with a margin of
1.2-3.9 (EW conservation: 6.7, a few float32 roundings; brute force vs stored products: 2.5e-7, 8 x the float32
rounding bound 2^-25 = 3.0e-8 of profile values in [0.5, 1), which is what M424 gives); they are M424-specific (grid,
library density, velocity field) and should be set per run (``tolerances``). The relevant scale is the run's LPV
(:func:`compare_lpv`). M424: residual rms 3.0e-4 / 9.0e-5 / 2.9e-4 (imu, |v| <= 600 km/s; flux within 5 %), EW rms
1.27e-4 / 2.3e-4 / 2.6e-4 A (flux; imu 1.25e-4 / 2.7e-4 / 2.7e-4 A). Largest ratio over the lines (imu residual rms;
EW checks: flux EW rms):

* profiles: V1 flux 1.1 %, V1 imu 0.8 %, V2 hold-out 2.9 % (holdout_B_F0 of lambda4922, 8.4e-6), V2 in-sample 2.4 %,
  V3 0.7 %, V4 0.4 %, V5 0.1 %; V6 9.5 / 3.1 / 11.3 % (20000-point subsets, whose random rounding errors are larger
  than those of the full sum: with all ~618000 visible points V6 is 3.3e-6 - 7.8e-6, 1.4-2.7 % of the rms, for dumps
  3600 and 4000); the informational V3 leave-out 35 %.
* EW (lambda4026; the other lines <= 1.9 %): V1 dEW 5.7 % (7.3e-6 A), V2 dEW 18 % hold-out (2.3e-5 A) and 12 %
  in-sample (1.6e-5 A).

So :func:`compare_lpv` warns on V6, V1_flux_dEW, V2_holdout_dEW and V2_insample_dEW (and flags V3_leaveout). The
lambda4026 EW deviations come from its EW(T_eff') sawtooth (the continuum-sampling artefact of the project, which
contaminates the lambda4026 EW time series by 16 % of its rms): the branch switches every ~170 K are smeared by the
10 K bins and the nodes. The EW time series of lambda4026 carries library errors of 6-18 % of its rms, its profiles
<= 3 % (V6 aside, a subset effect).

Validation
----------
tests/synspec/test_validate.py. Synthetic (M424 grid, 3 lines, 8 lines of sight, toy run of a few thousand points):
v1_exact, v3_tails, v4_nearest, v5_node_merging, v6_rounding equal, bit for bit, the arrays of the frozen
fw_disc_dumps_validate.py (run as a script on the toy files), v2_holdout those of the frozen fw_disc_holdout.py
(serial and 2 workers, 'fork' and 'spawn'; library file or built in the pass), and :func:`exact_continuous` and
:class:`NearestBin` the profiles of the frozen fw_disc_validate.py; general runs (2 lines, 3 lines of sight, another
grid): every check passes on correct inputs and fails exactly on the faults of the table above (V2 through faulty
factories and a replaced DiscFlux), and a second toy with T_eff' within the bins shows faulty interpolation weights;
the sign conventions are checked analytically; the brute force agrees with a naive per-point loop and DiscFlux to
~1e-15 (serial = fork = spawn, bit for bit); NaN inputs fail V1 and EW conservation in any position. Intensity
method (toy intensity libraries): :func:`v1_exact` with :class:`ppmpy.synspec.disc.DiscImu` gives the arrays of
the frozen fw_disc.DiscImu bit for bit; :func:`brute_force_imu` agrees with a naive per-point loop and with DiscImu
(every mode) to rounding, serial = fork = spawn; :func:`brute_force_imu_check` fails for a flipped velocity sign,
swapped T_eff' weights and mixed-up stored products, and refuses another library. M424 (marker m424):
validate.npz (V1 flux, V3, V4, V5, V6; V1 imu with the frozen fw_disc.DiscImu and with
:class:`ppmpy.synspec.disc.DiscImu`) and holdout.npz are reproduced bit for bit; teff_ranges reproduces the first
six columns of teff_ranges_all_dumps.npy (all 1601 dumps, a scratch check; 6 dumps in the test); the brute force
agrees with DiscFlux to 2.1e-15 and with the stored flux products to their float32 rounding, 2^-25 = 2.98e-8 (dumps
3200 and 4800); the intensity brute force agrees with DiscImu to 2.5e-14 (dump 4800, 20000-point subset) and
8.3e-15 (all points; F, F0) and with the stored imu products to their float32 rounding, 2^-25 = 2.98e-8 (dump 4800);
EW conservation and the LPV rms of the flux and imu time series equal :data:`RECORDED_M424`. Under numpy 1.26 on
the AVX512 Trillium nodes (library and sphere module notes for what is hardware-dependent).

Measured (M424, Trillium login node, 2026-10-01): V1 + V3-V6 serially 65-71 s, 1.7 GB; V1 imu with the frozen
DiscImu 13 s with a warm page cache (9.4 GB); v2_holdout with 8 'fork' workers 72 s (workers 452 s CPU, parent
5.9 GB peak RSS, 3.7 GB per worker including inherited pages); brute force 34-38 s per dump with 8 workers (~250 s
CPU, < 1 GB per process); teff_ranges of all 1601 dumps 2 s with 8 workers (warm cache); EW conservation and LPV
rms of one production time series 60-70 s. 2026-10-02: V1 imu with ppmpy's DiscImu 15 s after a 67 s set-up (peak
RSS 9.6 GB; the set-up takes 21-200 s on a login node, almost all system time, 192 s in one run); the intensity brute
force of a 20000-point subset 16 s in one process, of all points of a dump 143 s with 8 'fork' workers (~15 min in
one process: 37 s per line of sight and line; about 1 GB per worker, and 2.5 GB peak RSS for the parent, which holds
the task list, ~0.5 GB, and computes the integrator's profiles: a lazy DiscImu).

Notes
-----
fw_disc_validate.py (the first library-vs-exact check, 2026-09-28) compared the nearest-bin library
(fw_disc.integrate_lib, bins of the models' own T_eff') with fw_disc.integrate_exact: continuous Doppler shifts
(neither rounded nor clipped) on the fw_disc.directions(ndir) lines of sight ('matvec' projections), so its
differences include the 1 km/s rounding. :func:`v1_exact` with :class:`NearestBin` against disc_los8.npz (rounded
shifts, the los8 directions) leaves the rounding out; ``v1_exact(NearestBin(lib, grid), exact_continuous(...),
...)`` with the models' T_eff' in the sample is the old check (bit for bit on a toy, test_validate.py).

r3_out.npz (project/analysis, written 2026-09-29 12:01 during the review workflow; producer not kept): Fl, Fl0
float32 (21, 8, 3, 5401) for dumps 3200-4800 in steps of 80, the 8 lines of sight in the los8 order (the diagonal
matches best), closest to the production flux run of the same dump (F within 6.5e-4 - 9.5e-4, F0 within 3.4e-6 -
2.7e-5; the intensity run differs by 1.9e-3 in F0; the brute force: 8.3e-4 / 1.5e-4 / 7.8e-4 for dump 3200,
9.5e-4 / 1.6e-4 / 8.8e-4 for 4800). It is NOT a same-input re-implementation of the pipeline:
none of these reproduce it -- flipped, scaled (0.9-1.2) or partial velocity components, weights mu without F_c,
the nearest-bin library, neighbouring dumps, strided or random subsets of the points. Its inputs (velocities and
T_eff' of the points) evidently differed. :func:`brute_force_check` compares with it as an informational check
(``reference``); the 2026-09-29 "agree to <= 7.6e-6" check is restored by the brute force against the stored
products (float32 rounding) and against DiscFlux (float64).

PP 2026-10-01: ported from the project's fw_disc_dumps_validate.py (V1, V3-V6), fw_disc_holdout.py (V2),
fw_disc_validate.py (the first library-vs-exact check: :func:`v1_exact` with :class:`NearestBin` and
:func:`exact_continuous`) and fig_disc_dumps_validation.py (the LPV comparison); brute force, EW conservation, T_eff'
ranges and the report are new. See the provenance comments per function.
"""
import datetime
import json
import os
import shutil
import tempfile
import time
import warnings

import numpy as np
from scipy import sparse

from . import parallel as par
from .conventions import C_KMS
from .diagnostics import line_diagnostics
from .disc import (DiscFlux, _ExactStreamer, _check_teff, _fc0, _ordered_window, _profile_arrays, _rows, _shift_add,
                   _spill, _spill_matrix, _stream_init, _stream_task, _velocities, integrate_library_nearest)
from .dumps import SAMPLE_PATTERN, _blas_limit, _integ_grid, _integ_lref, flux_integrator, load_sample, sample_path
from .io import make_meta, save_npz
from .library import FluxLibrary, _share, lib_nodes, node_pairs, teff_bins, teff_edges
from .spectral import VelocityGrid, interp_rows
from .sphere import _los_vectors, disc_weights, los_velocity, project_los

__all__ = ["CheckResult", "ValidationReport", "NearestBin", "DEFAULT_TOLERANCES", "RECORDED_M424",
           "RECORDED_M424_ARRAYS", "LPV_WARN", "exact_continuous", "v1_exact", "v2_holdout", "v3_tails", "v4_nearest",
           "v5_node_merging", "v6_rounding",
           "teff_ranges", "v3_select", "brute_force", "brute_force_check", "brute_force_imu", "brute_force_imu_check",
           "ew_conservation", "lpv_residual_rms", "compare_lpv", "run_validation"]

DEFAULT_TOLERANCES = dict(
    V1=5e-6,           # max |dF|, pipeline vs exact per-point sums (M424: flux 3.2e-6, imu 2.0e-6)
    V1_dEW=2e-5,       # max |dEW| [A] (M424 flux: 7.3e-6)
    V2=1e-5,           # hold-out and in-sample, both halves, F and F0 (M424: 8.4e-6 hold-out, 7.1e-6 in-sample)
    V2_dEW=5e-5,       # [A] (M424: 2.3e-5 hold-out, 1.6e-5 in-sample)
    V3=5e-6,           # clamped vs extrapolated (M424: 1.6e-6); the leave-out comparison is informational
    V4=2e-6,           # interpolation vs nearest bin (M424: 9.0e-7)
    V5=1e-6,           # node merging nmin 1/5/100 vs 20 (M424: 2.6e-7)
    V6=5e-5,           # rounded vs continuous shifts, 20000-point subsets (M424: 3.3e-5)
    brute=2.5e-7,      # brute force vs stored float32 products: 8 x their rounding bound 2^-25 = 2.98e-8 (M424:
                       # 2.98e-8; the 2026-09-29 review's re-implementations agreed to <= 7.6e-6, 'brute_review')
    brute_f64=1e-10,   # brute force vs the integrator, float64 (M424: ~1e-15)
    brute_moments=1e-9,  # [km/s] intensity brute force vs the integrator / the stored float64 velocity moments
                       # vmean_w, sigma_w (M424 dump 4800: 5e-13 vs DiscImu on a subset; all points 1.1e-13 vs DiscImu
                       # and vs the stored imu products)
    brute_n_clip=0,    # clipped shifts per line of sight, intensity brute force vs the integrator / stored (exact)
    ew=1e-6,           # relative |EW(F) - EW(F0)| on the y grid without the Jacobian (M424 float32 products: 1.6e-7)
)
"""Default tolerances: the recorded M424 values (:data:`RECORDED_M424`) with a margin. M424-specific: set them per
run."""

RECORDED_M424 = dict(
    V1_flux=3.211538721514806e-06, V1_flux_dEW=7.272872323937918e-06, V1_imu=2.034146711515916e-06,
    V2_holdout=8.387348581329057e-06, V2_holdout_dEW=2.2978252717242853e-05, V2_insample=7.109697187868136e-06,
    V2_insample_dEW=1.581087167523698e-05, V3_extrap=1.5613700771188732e-06, V3_leaveout=0.00010174720566125117,
    V4=9.032221559568399e-07, V5=2.556791788288493e-07, V6=3.315605358289453e-05, brute_review=7.6e-06,
    brute_stored=2.980219060422229e-08, ew_flux=1.4861438631683864e-07, ew_imu=1.600401820519393e-07,
)
"""The largest value of each M424 check, the regression baselines: V1-V6 as stored (validate.npz, holdout.npz of
2026-09-29); brute_review = the 2026-09-29 review's re-implementations (rounded, as recorded in the project);
brute_stored = :func:`brute_force_check` vs the stored flux products of dumps 3200 and 4800 (2026-10-01; the float32
rounding bound 2^-25); ew_* = :func:`ew_conservation` of the flux / imu time series (2026-10-01). All but
brute_review are the full float64 values (equality regressions). Per-line values: :data:`RECORDED_M424_ARRAYS`."""

RECORDED_M424_ARRAYS = dict(
    V1_flux_F=(1.6484739550071126e-06, 3.9495194881222773e-07, 2.348818084807469e-06),
    V1_flux_F0=(1.748638151055637e-06, 4.402135623804426e-07, 3.211538721514806e-06),
    V1_flux_dEW=(7.272872323937918e-06, 2.6015392112777036e-06, 2.2946320896721772e-06),
    V1_imu_F=(1.9203030946490784e-06, 7.368405136043421e-07, 1.2782090688112646e-06),
    V1_imu_F0=(2.034146711515916e-06, 6.498684712585856e-07, 1.1671614701391775e-06),
    V4=((5.912614206016187e-07, 3.5767729933411374e-07, 4.1710635478864333e-07),
        (9.032221559568399e-07, 2.9428449543900115e-07, 7.344957621002735e-07),
        (5.140756769161925e-07, 2.0392644950462113e-07, 5.28230606366975e-07),
        (6.113172774657727e-07, 3.1117821885917607e-07, 6.847058142689377e-07)),
    V4_dumps=(3600, 4000, 4400, 4800),
    V5_nmin1=(4.199858127940104e-08, 4.492101712827434e-08, 9.740525253043586e-08),
    V5_nmin5=(1.578868080720497e-08, 4.3973031660371475e-08, 3.9101549664799506e-08),
    V5_nmin100=(9.191472960523583e-08, 1.28353950801241e-07, 2.556791788288493e-07),
    V3_dumps=(3334, 4169, 4391),
    V3_leaveout=((8.716388060014957e-05, 2.697679693863808e-05, 0.00010174720566125117),
                 (2.317381322858303e-05, 6.978392441903125e-06, 2.6951279617826174e-05),
                 (1.0377329601518603e-05, 2.5386685208461657e-06, 1.1978961077518946e-05)),
    V3_extrap=((8.143009553318592e-07, 6.619530128482154e-07, 1.5613700771188732e-06),
               (2.0462467542614604e-07, 1.5608179992909754e-07, 3.7361210658559685e-07),
               (2.1034169128686386e-07, 3.928553244936239e-07, 2.1362206892305835e-07)),
    V3_n=((11.0, 350.0), (4.0, 106.0), (16.0, 32.0)),
    V6=((2.305436149907525e-05, 2.256107851650313e-06, 2.6175750392876118e-05),
        (1.5202486659871006e-05, 2.250633177380834e-06, 2.2167888141688685e-05),
        (1.869478003457825e-05, 1.662706122562696e-06, 1.3626003901978656e-05),
        (2.889511756554164e-05, 2.7672542808332423e-06, 3.315605358289453e-05)),
    holdout_A_F=(2.9026737800030844e-06, 7.604579972397829e-07, 3.8567686797552625e-06),
    holdout_A_F0=(4.423439090350811e-06, 9.805427189091276e-07, 4.692240276549242e-06),
    holdout_A_dEW=(2.103669121811258e-05, 3.317642338296345e-06, 4.77385588326662e-06),
    insample_A_F=(2.234223663633017e-06, 5.705070278416713e-07, 3.3713767766396785e-06),
    insample_A_F0=(3.2049175165971278e-06, 6.419881347641265e-07, 3.4694967123716225e-06),
    insample_A_dEW=(1.4214748817575895e-05, 2.5134926403547198e-06, 3.6363504852809925e-06),
    holdout_B_F=(3.03635671616842e-06, 8.303745244742089e-07, 2.861738907178335e-06),
    holdout_B_F0=(4.14984599650392e-06, 1.0290035805660125e-06, 8.387348581329057e-06),
    holdout_B_dEW=(2.2978252717242853e-05, 3.6155342013621805e-06, 4.26343109388716e-06),
    insample_B_F=(2.428126699483002e-06, 6.211232178587878e-07, 2.4707491913522617e-06),
    insample_B_F0=(2.8309187627417742e-06, 7.512939554921161e-07, 7.109697187868136e-06),
    insample_B_dEW=(1.581087167523698e-05, 2.6903632757147022e-06, 2.995734614208434e-06),
    lpv_rms_imu=(0.0003031444282102798, 8.99582551439878e-05, 0.0002932557512540865),
    lpv_rms_flux=(0.00030146169576681546, 8.630965525186246e-05, 0.0002883749993735234),
    lpv_ew_rms_imu=(0.00012526056789546094, 0.0002722065336809854, 0.00026779177558872304),
    lpv_ew_rms_flux=(0.00012718662242087417, 0.00023382483576872505, 0.0002586874770619554),
    ew_flux=(8.252004627583862e-08, 9.914276785678812e-08, 1.4861438631683864e-07),
    ew_imu=(7.633749066550687e-08, 1.1685266365731131e-07, 1.600401820519393e-07),
)
"""Per-line M424 values (HEI4026, HEII4200, HEI4922) under the legacy member names of validate.npz / holdout.npz
(per dump: rows), plus the LPV residual rms (|v| <= 600 km/s) and the EW rms [A] (:func:`lpv_residual_rms` of the
flux / imu time series) and the EW conservation of those time series (:func:`ew_conservation`); full float64
values."""

LPV_WARN = 0.05
"""A check whose deviation exceeds this fraction of the run's LPV residual rms (per line) is flagged and warned."""

_UNITS = ("continuum", "A", "relative", "km/s", "count")


# ----------------------------------------------------------------------------------------------
# results
# ----------------------------------------------------------------------------------------------
def _jsonable(x):
    """x with numpy arrays and scalars converted to lists and numbers (JSON)."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None if np.isnan(x) else ("inf" if x > 0 else "-inf")
    return x


def _float_or_none(x):
    """A JSON number back to float: None -> None, 'inf' / '-inf' / 'nan' strings -> float."""
    return None if x is None else float(x)


def _float_array(x):
    """Per-line values as a float64 array; None (NaN after a JSON round trip) -> NaN."""
    if x is None:
        return None
    return np.asarray(_none_to_nan(x), dtype=np.float64)


def _none_to_nan(x):
    if isinstance(x, (list, tuple)):
        return [_none_to_nan(v) for v in x]
    return np.nan if x is None else x


_FLOAT_DETAILS = ("per_line", "lpv_ratio", "lpv_scale")


def _fmt_max(x):
    """'{:.1%}' of the largest finite ratio ('nan' if none is finite)."""
    a = np.asarray(x, dtype=np.float64)
    f = a[np.isfinite(a)]
    return "nan" if f.size == 0 else "{:.1%}".format(float(f.max()))


class CheckResult:
    """
    One validation check: the largest deviation found, its tolerance and the verdict.

    Parameters
    ----------
    name: str
        Check name (e.g. 'V1_flux', 'V2_holdout').
    value: float
        The largest deviation (NaN fails any tolerance).
    tolerance: float or None
        Largest acceptable value (may be inf, not NaN); None = informational (``passed`` None).
    passed: bool or None
        Default ``value <= tolerance`` (None without a tolerance), so a NaN value fails.
    details: dict, optional
        Per-line values ('per_line'), line names ('lines'), dumps, notes, LPV comparison.
    unit: str
        'continuum' (|dF| in units of the continuum), 'A' (EW), 'relative', 'km/s' (velocity moments) or 'count'
        (e.g. clipped shifts); the LPV comparison (:func:`compare_lpv`) covers the first three.

    Attributes
    ----------
    status: str
        'PASS', 'FAIL' or 'info'.
    """

    def __init__(self, name, value, tolerance=None, passed=None, details=None, unit="continuum"):
        self.name = str(name)
        self.value = float(value)
        self.tolerance = None if tolerance is None else float(tolerance)
        if self.tolerance is not None and np.isnan(self.tolerance):
            raise ValueError("tolerance of {} is NaN: use None for an informational check".format(self.name))
        if passed is None and self.tolerance is not None:
            passed = bool(self.value <= self.tolerance)
        self.passed = None if passed is None else bool(passed)
        self.details = dict(details or {})
        if unit not in _UNITS:
            raise ValueError("unit must be one of {}, got {!r}".format(_UNITS, unit))
        self.unit = unit

    @property
    def status(self):
        return "info" if self.passed is None else ("PASS" if self.passed else "FAIL")

    def to_dict(self):
        """JSON-able record (arrays as lists)."""
        return dict(name=self.name, value=_jsonable(self.value), tolerance=_jsonable(self.tolerance),
                    passed=self.passed, unit=self.unit, details=_jsonable(self.details))

    @classmethod
    def from_dict(cls, d):
        """Inverse of :meth:`to_dict`: value None -> NaN, 'inf' -> inf; details keep their JSON types (lists, not
        arrays) except per_line, lpv_ratio, lpv_scale, which become float64 arrays (None -> NaN)."""
        # PP 2026-10-01: reviewer: NaN per-line values came back as None and broke table(); an inf tolerance did not
        # survive to_json
        v = d["value"]
        v = np.nan if v is None else float(v)
        det = dict(d.get("details") or {})
        for k in _FLOAT_DETAILS:
            if det.get(k) is not None:
                det[k] = _float_array(det[k])
        return cls(d["name"], v, _float_or_none(d.get("tolerance")), d.get("passed"), det,
                   d.get("unit", "continuum"))

    def __repr__(self):
        return "CheckResult({}: {:.3g} {} tol {} {})".format(
            self.name, self.value, self.unit, "-" if self.tolerance is None else "{:.3g}".format(self.tolerance),
            self.status)


class ValidationReport:
    """
    A set of :class:`CheckResult` with provenance and the numbers in the layout of the legacy products.

    Parameters
    ----------
    checks: sequence of CheckResult
    meta: dict, optional
        Parameters, inputs, timing.
    arrays: dict, optional
        name -> array: the legacy members (validate.npz / holdout.npz names) and further arrays to save.
    data: dict, optional
        In-memory results not saved (e.g. the profiles of the pipeline and of the reference).
    """

    def __init__(self, checks=(), meta=None, arrays=None, data=None):
        self.checks = list(checks)
        names = [c.name for c in self.checks]
        if len(set(names)) != len(names):
            raise ValueError("repeated check names: {}".format(names))
        self.meta = dict(meta or {})
        self.arrays = dict(arrays or {})
        self.data = dict(data or {})

    # -- access ------------------------------------------------------------------------------
    def __len__(self):
        return len(self.checks)

    def __iter__(self):
        return iter(self.checks)

    def __getitem__(self, name):
        for c in self.checks:
            if c.name == name:
                return c
        raise KeyError(name)

    def __contains__(self, name):
        return any(c.name == name for c in self.checks)

    def names(self):
        return [c.name for c in self.checks]

    def passed(self):
        """True if there is at least one check and none failed (informational checks do not count)."""
        return bool(self.checks) and all(c.passed is not False for c in self.checks)

    def failed(self):
        """The checks that failed."""
        return [c for c in self.checks if c.passed is False]

    def warnings(self):
        """The checks flagged by :func:`compare_lpv` (deviation above its fraction of the LPV rms)."""
        return [c for c in self.checks if c.details.get("lpv_warn")]

    @classmethod
    def merge(cls, *reports, meta=None):
        """One report from several (checks and arrays concatenated; names must not repeat)."""
        checks, arrays, data, metas = [], {}, {}, {}
        for r in reports:
            if r is None:
                continue
            checks += r.checks
            for k, v in r.arrays.items():
                if k in arrays and k not in ("lines",):
                    raise ValueError("array {!r} in more than one report".format(k))
                arrays[k] = v
            data.update(r.data)
            for c in r.checks:
                metas[c.name] = r.meta
        m = dict(meta or {})
        m.setdefault("parts", _jsonable(metas))
        return cls(checks, meta=m, arrays=arrays, data=data)

    def __repr__(self):
        return "ValidationReport({} checks, {} failed, {} informational)".format(
            len(self.checks), len(self.failed()), sum(c.passed is None for c in self.checks))

    # -- printing ------------------------------------------------------------------------------
    def table(self, per_line=True):
        """
        The checks as a text table: name, value, tolerance, value / tolerance, status, the largest ratio to the LPV
        residual rms (when :func:`compare_lpv` was applied) and, with ``per_line``, the per-line values.

        Returns
        -------
        str
        """
        rows = [("check", "value", "unit", "tolerance", "val/tol", "status", "/LPV", "per line")]
        for c in self.checks:
            tol = "-" if c.tolerance is None else "{:.2e}".format(c.tolerance)
            ratio = "-" if not c.tolerance else "{:.2f}".format(c.value / c.tolerance)
            lpv = c.details.get("lpv_ratio")
            lpv = "-" if lpv is None else "{}{}".format(_fmt_max(_float_array(lpv)),
                                                        " !" if c.details.get("lpv_warn") else "")
            pl = c.details.get("per_line")
            pl = (" / ".join("{:.2e}".format(x) for x in _float_array(pl).ravel())
                  if (per_line and pl is not None) else "")
            rows.append((c.name, "{:.3e}".format(c.value), c.unit, tol, ratio, c.status, lpv, pl))
        w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
        lines = ["  ".join(r[i].ljust(w[i]) for i in range(len(r))).rstrip() for r in rows]
        lines.insert(1, "-" * len(lines[0]))
        lines.append("{} checks: {} passed, {} failed, {} informational".format(
            len(self.checks), sum(c.passed is True for c in self.checks), len(self.failed()),
            sum(c.passed is None for c in self.checks)))
        return "\n".join(lines)

    # -- files ---------------------------------------------------------------------------------
    def _record(self):
        return dict(meta=_jsonable(self.meta), checks=[c.to_dict() for c in self.checks])

    def to_json(self, path):
        """
        Write the report (meta, checks with details, arrays as lists) as JSON, atomically.

        Returns
        -------
        str
            path.
        """
        rec = self._record()
        rec["arrays"] = {k: dict(dtype=str(np.asarray(v).dtype), value=_jsonable(np.asarray(v)))
                         for k, v in self.arrays.items()}
        path = os.fspath(path)
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        tmp = "{}.tmp{}".format(path, os.getpid())
        try:
            with open(tmp, "w") as f:
                json.dump(rec, f, indent=1, sort_keys=False, allow_nan=False)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return path

    @classmethod
    def from_json(cls, path):
        """Read a report written by :meth:`to_json`."""
        with open(os.fspath(path)) as f:
            rec = json.load(f)
        arrays = {k: np.asarray(v["value"], dtype=v["dtype"]) for k, v in rec.get("arrays", {}).items()}
        return cls([CheckResult.from_dict(c) for c in rec["checks"]], meta=rec.get("meta"), arrays=arrays)

    def to_npz(self, path, meta=None):
        """
        Write the arrays (legacy member names, e.g. those of validate.npz) plus check_names, check_values,
        check_tolerances (NaN = informational), check_passed (1, 0, -1 = informational), check_units and '_meta'
        (:func:`ppmpy.synspec.io.make_meta` with the report's meta and the checks with their details), atomically
        (:func:`ppmpy.synspec.io.save_npz`). A caller's ``meta`` (a make_meta record) is written instead, with the
        checks always added as its 'checks' (replacing any) and the report's meta as its 'params' unless it has
        them, so :meth:`from_npz` reads the report back either way.

        Returns
        -------
        str
            path.
        """
        out = dict(self.arrays)
        for k in ("check_names", "check_values", "check_tolerances", "check_passed", "check_units"):
            if k in out:
                raise ValueError("array name {!r} is reserved".format(k))
        out.update(check_names=np.array(self.names()), check_values=np.array([c.value for c in self.checks]),
                   check_tolerances=np.array([np.nan if c.tolerance is None else c.tolerance for c in self.checks]),
                   check_passed=np.array([-1 if c.passed is None else int(c.passed) for c in self.checks], np.int8),
                   check_units=np.array([c.unit for c in self.checks]))
        rec = self._record()
        if meta is None:
            meta = make_meta("synspec.validation", params=rec["meta"], checks=rec["checks"])
        else:
            # PP 2026-10-01: reviewer: a caller's meta dropped the checks (from_npz then read an empty report)
            meta = dict(meta)
            meta["checks"] = rec["checks"]
            meta.setdefault("params", rec["meta"])
        return save_npz(os.fspath(path), out, meta=meta)

    @classmethod
    def from_npz(cls, path):
        """Read a report written by :meth:`to_npz` (details from its '_meta')."""
        from .io import read_meta
        with np.load(os.fspath(path)) as z:
            arrays = {k: z[k] for k in z.files if not k.startswith("check_") and k != "_meta"}
            meta = read_meta(z)
        checks = [CheckResult.from_dict(c) for c in meta.get("checks", [])]
        return cls(checks, meta=meta.get("params", {}), arrays=arrays)


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _tol(name, tolerance, tolerances=None):
    """The tolerance of a check: explicit value, else tolerances[name], else DEFAULT_TOLERANCES[name]."""
    if tolerance is not None:
        return float(tolerance)
    if tolerances is not None and name in tolerances:
        t = tolerances[name]
        return None if t is None else float(t)
    t = DEFAULT_TOLERANCES.get(name)
    return None if t is None else float(t)


def _line_names(integ, lref, nl):
    """Line names: a LineSet's, else the integrator's LineSet, else 'line<j>'."""
    names = getattr(lref, "names", None)
    if names is None and integ is not None:
        names = getattr(getattr(integ, "lines", None), "names", None)
    if names is None or len(names) != nl:
        names = ["line{}".format(j) for j in range(nl)]
    return [str(n) for n in names]


def _sample_dict(s, where=""):
    """teff, ur, uth, uph as float64 (N,) (+ t_s if present) from a mapping or an .npz path."""
    if isinstance(s, (str, os.PathLike)):
        with np.load(os.fspath(s)) as z:
            s = {k: z[k] for k in z.files}
    try:
        out = {k: np.asarray(s[k]).astype(np.float64) for k in ("teff", "ur", "uth", "uph")}
    except (KeyError, TypeError, IndexError):
        raise ValueError("a sample needs teff, ur, uth, uph{}".format(where)) from None
    n = out["teff"].shape
    if len(n) != 1 or any(out[k].shape != n for k in ("ur", "uth", "uph")):
        raise ValueError("teff, ur, uth, uph must be 1-D of one length{}".format(where))
    try:
        if "t_s" in s:
            out["t_s"] = float(np.asarray(s["t_s"]))
    except TypeError:
        pass
    return out


def _get_sample(samples, d, pattern=SAMPLE_PATTERN):
    """The sample of dump d: samples = directory (load_sample), mapping dump -> sample, or callable."""
    # PP 2026-10-01: ported from fw_disc_dumps_validate.py:51-53 (sample(d): .astype(np.float64) of the stored float32)
    if isinstance(samples, (str, os.PathLike)):
        return load_sample(samples, d, pattern=pattern)
    s = samples(d) if callable(samples) else samples[d]
    return _sample_dict(s, " (dump {})".format(d))


def _projections(mu, tn, pn, N=None):
    mu, tn, pn = np.asarray(mu), np.asarray(tn), np.asarray(pn)
    if mu.ndim != 2 or tn.shape != mu.shape or pn.shape != mu.shape or mu.shape[0] == 0:
        raise ValueError("mu, tn, pn must have one shape (nlos, N), got {}, {}, {}".format(mu.shape, tn.shape,
                                                                                          pn.shape))
    if N is not None and mu.shape[1] != N:
        raise ValueError("the projections are for {} points, the sample has {}".format(mu.shape[1], N))
    return mu, tn, pn


def _run(integ, smp, mu, tn, pn, pairs=None, mask=None):
    """
    F, F0 (nlos, nl, ny) float64 of one sample with an integrator; pairs = (k0, k1, a) override; mask: points to
    include (others get mu = 0, i.e. hidden).
    """
    # PP 2026-10-01: ported from fw_disc_dumps_validate.py:56-64 (run(): v = ur * MU[k] + uth * TN[k] + uph * PN[k]
    # via los_velocity, same operation order)
    teff, ur, uth, uph = smp["teff"], smp["ur"], smp["uth"], smp["uph"]
    mu, tn, pn = _projections(mu, tn, pn, teff.size)
    k0, k1, a = integ.pairs(teff) if pairs is None else pairs
    F = F0 = None
    for k in range(mu.shape[0]):
        m = mu[k] if mask is None else np.where(mask, mu[k], 0.0)
        out = integ(m, los_velocity(ur, uth, uph, mu[k], tn[k], pn[k]), k0, k1, a)
        if F is None:
            F = np.zeros((mu.shape[0],) + np.shape(out[0]))
            F0 = np.zeros_like(F)
        F[k], F0[k] = out[0], out[1]
    return F, F0


def _dmax(A, B):
    """max |A - B| over lines of sight and grid, per line (A, B (nlos, nl, ny))."""
    # PP 2026-10-01: ported from fw_disc_dumps_validate.py:67-68 (dmax)
    A, B = np.asarray(A), np.asarray(B)
    if A.shape != B.shape or A.ndim != 3:
        raise ValueError("profiles must have one shape (nlos, nl, ny), got {} and {}".format(A.shape, B.shape))
    return np.abs(A - B).max(axis=(0, 2))


def _dew(F, Fref, y, lr, ew_jacobian=True):
    """max over lines of sight of |EW(F) - EW(Fref)| per line [A] (each profile on its own, as the legacy loop)."""
    # PP 2026-10-01: ported from fw_disc_dumps_validate.py:80-81 and fw_disc_holdout.py:132-133 (fd.diagnostics(F[k, j],
    # j)['ew'] per profile; line_diagnostics gives the same bits for any leading axes)
    kw = dict(keys=("ew",), ew_jacobian=ew_jacobian)
    e = line_diagnostics(np.asarray(F), y, lr[None, :], **kw)["ew"]
    e0 = line_diagnostics(np.asarray(Fref), y, lr[None, :], **kw)["ew"]
    return np.abs(e - e0).max(axis=0)


def _exact_arrays(exact):
    """F, F0 of exact sums: a path (.npz), an np.load()ed file or a mapping."""
    if isinstance(exact, (str, os.PathLike)):
        with np.load(os.fspath(exact)) as z:
            return np.asarray(z["F"]), np.asarray(z["F0"])
    return np.asarray(exact["F"]), np.asarray(exact["F0"])


def _logger(log, T0):
    def _log(msg):
        if log is not None:
            log("[{:7.1f} s] {}".format(time.time() - T0, msg))
    return _log


def _fmt(x):
    return " ".join("{:.1e}".format(float(v)) for v in np.ravel(x))


def _dumps_given(dumps, check):
    """The dumps as a list of int; ValueError if there are none (a check of nothing must not pass)."""
    dl = [int(d) for d in dumps]
    if not dl:
        raise ValueError("{}: no dumps given".format(check))
    return dl


def _node_params(integ):
    return dict(getattr(integ, "node_params", None) or {})


def _check_integ_nodes(integ, nodes, where):
    """
    Raise unless ``nodes`` are the integrator's: the same t (if the integrator has t) and fc (if it has fc). V6 and
    the brute force interpolate ``nodes`` themselves, so nodes of another nmin or library would read as a rounding or
    integration fault.
    """
    # PP 2026-10-01: reviewer: nothing checked that nodes belong to integ (V6 failed at 1.7e-3 with nodes of another
    # nmin; an intensity integrator was compared with the flux nodes)
    if integ is None:
        return
    t = getattr(integ, "t", None)
    if t is not None:
        tn = np.asarray(nodes["t"])
        if np.shape(t) != tn.shape or not np.array_equal(np.asarray(t), tn):
            raise ValueError("{}: the integrator's nodes are not these nodes ({} nodes {:.1f}-{:.1f} K vs {} nodes "
                             "{:.1f}-{:.1f} K): pass the nodes the integrator was built from (another nmin, library or "
                             "method?)".format(where, np.size(t), float(np.min(t)), float(np.max(t)), tn.size,
                                               float(tn.min()), float(tn.max())))
    fc = getattr(integ, "fc", None)
    if fc is not None and not np.array_equal(np.asarray(fc), np.asarray(nodes["fc"])):
        raise ValueError("{}: the integrator's node continuum fluxes fc differ from those of these nodes (another "
                         "library or correction?)".format(where))


# ----------------------------------------------------------------------------------------------
# V1: the library's own dump vs the exact per-point sums
# ----------------------------------------------------------------------------------------------
class NearestBin:
    """
    The nearest-bin library method (:func:`ppmpy.synspec.disc.integrate_library_nearest`, the first library method of
    fw_disc_validate.py) with the integrator duck type, so that :func:`v1_exact` and the other checks accept it:
    ``pairs(teff)`` gives (bin, bin, 0), the call integrates with the weights mu F_c of the bin.

    fw_disc_validate.py's reference was :func:`exact_continuous` (continuous shifts) on fw_disc.directions(ndir), with
    the bins of the models' own T_eff'; v1_exact(NearestBin(...), exact_continuous(...), sample with the models' teff,
    ...) is that check (module notes). Against disc_los8.npz (rounded shifts) it leaves the rounding out.

    Parameters
    ----------
    library: FluxLibrary, str or os.PathLike
    grid: VelocityGrid
        The library's grid.

    Attributes
    ----------
    t: np.ndarray
        Bin centres [K] (increasing); method 'nearest'; grid; nl.
    """

    method = "nearest"

    def __init__(self, library, grid):
        # PP 2026-10-01: ported from fw_disc_validate.py:36-38 (b = digitize bins, integrate_lib with mu F_c[b])
        self.lib = library if isinstance(library, FluxLibrary) else FluxLibrary.load(library)
        if not isinstance(grid, VelocityGrid):
            raise TypeError("grid must be a VelocityGrid")
        if self.lib.ny != grid.ny:
            raise ValueError("the library has {} grid points, the grid {}".format(self.lib.ny, grid.ny))
        self.grid, self.nl = grid, self.lib.nl
        self.t = self.lib.centres

    def pairs(self, teff):
        b = teff_bins(teff, self.lib.edges)
        return b, b, np.zeros(b.shape)

    def __call__(self, mu, v, k0, k1, a, novel=True):
        tc = self.t[k0]                                   # the bin centre selects the same bin
        w = disc_weights(mu, 1.0)[:, None] * self.lib.fc[k0]
        F = integrate_library_nearest(self.lib, tc, v, w, self.grid)
        F0 = integrate_library_nearest(self.lib, tc, np.zeros_like(v), w, self.grid) if novel else None
        vis = mu > 0
        wv = w[vis]
        vm = (wv * v[vis, None]).sum(axis=0) / wv.sum(axis=0)
        sd = np.sqrt((wv * (v[vis, None] - vm) ** 2).sum(axis=0) / wv.sum(axis=0))
        ncl = int(self.grid.shift_steps(v[vis])[1].sum())
        return F, F0, vm, sd, ncl


def exact_continuous(profiles, sample, mu, tn, pn, lref, grid=None, chunk=20000, rows="auto", novel=True, log=None):
    """
    Exact disc integrals with continuous Doppler shifts: every visible point's own model, weight mu F_c (F_c =
    fcont[:, :, 0] of the model), placed on its own shifted abscissa and interpolated onto the grid
    (:func:`ppmpy.synspec.disc.integrate_exact` per line of sight; no rounding or clipping of the shifts). The
    reference of fw_disc_validate.py, for :func:`v1_exact`.

    Parameters
    ----------
    profiles: ProfileStore, str, os.PathLike, np.load()ed .npz or mapping
        The per-point models: lam, fnorm, fcont (N, nl, nrow) (a path is memory-mapped).
    sample: str, os.PathLike, mapping or 3-sequence
        u_r, u_theta, u_phi (N,) [km/s] of the points (as :func:`ppmpy.synspec.disc.integrate_exact_stream`; teff is
        not used).
    mu, tn, pn: array-like
        (nlos, N) projections (fw_disc_validate.py: ``project_los(theta, phi, directions, method='matvec')`` with
        the ndir Fibonacci directions of the legacy fw_disc.directions(ndir)).
    lref: LineSet or array-like
        (nl,) velocity zero points [A].
    grid: VelocityGrid, optional
        Output grid (default the M424 grid).
    chunk, rows:
        :func:`ppmpy.synspec.disc.integrate_exact` (legacy chunk 20000; the last bits depend on it).
    novel: bool
        Also compute F0 (v = 0: the same sums without shifts).
    log: callable, optional

    Returns
    -------
    dict
        F, F0 (nlos, nl, ny) float64 (F0 zeros if not ``novel``).

    Notes
    -----
    Work: (visible points) x nrow interpolations per line of sight and line, twice with ``novel`` (M424: ~3 min
    per line of sight and line in one process). Memory: the rows of one chunk.

    Validation
    ----------
    F equals fw_disc.integrate_exact of the frozen fw_disc_validate.py bit for bit (toy on the M424 grid,
    test_validate.py).
    """
    # PP 2026-10-01: ported from fw_disc_validate.py:29-34 (mu_vlos, fd.integrate_exact(p, v, fd.weights(mu, 1.0)[:,
    # None] * fc) with fc = fcont[:, :, 0] as float64); F0 (v = 0) is new
    from .disc import integrate_exact
    T0 = time.time()
    _log = _logger(log, T0)
    src = _profile_arrays(profiles)
    lam, fn = src["lam"], src["fnorm"]
    if "fcont" not in src:
        raise ValueError("profiles need a member 'fcont'")
    N, nl = lam.shape[:2]
    (ur, uth, uph), _, _ = _velocities(sample)
    for nm, a in (("ur", ur), ("uth", uth), ("uph", uph)):
        if np.shape(a) != (N,):
            raise ValueError("{} must have shape (N,) = ({},), got {}".format(nm, N, np.shape(a)))
    mu, tn, pn = _projections(mu, tn, pn, N)
    fc = _fc0(src["fcont"])
    if grid is None:
        grid = VelocityGrid()
    F = np.zeros((mu.shape[0], nl, grid.ny))
    F0 = np.zeros_like(F)
    for k in range(mu.shape[0]):
        v = los_velocity(ur, uth, uph, mu[k], tn[k], pn[k])
        w = disc_weights(mu[k], 1.0)[:, None] * fc
        F[k] = integrate_exact(lam, fn, lref, v, w, chunk=chunk, grid=grid, rows=rows)
        if novel:
            F0[k] = integrate_exact(lam, fn, lref, np.zeros(N), w, chunk=chunk, grid=grid, rows=rows)
        _log("exact_continuous: line of sight {} of {} ({} visible points)".format(k + 1, mu.shape[0],
                                                                                   int((mu[k] > 0).sum())))
    return dict(F=F, F0=F0)


def v1_exact(integ, exact, sample, mu, tn, pn, label="flux", lref=None, grid=None, ew_jacobian=True, tolerance=None,
             dew_tolerance=None, tolerances=None):
    """
    V1: the dump whose own models built the library (M424: 3200) through the pipeline vs the exact per-point sums
    (port of fw_disc_dumps_validate.py V1, flux and intensity; fw_disc_validate.py did this for the nearest-bin
    library against sums with continuous shifts: :class:`NearestBin` with :func:`exact_continuous`).

    Parameters
    ----------
    integ: object
        The integrator under test (DiscFlux duck type, module notes), e.g. :class:`ppmpy.synspec.disc.DiscFlux`
        of the run's nodes, :class:`ppmpy.synspec.disc.DiscImu` (the intensity method, label 'imu'; e.g.
        :func:`ppmpy.synspec.dumps.imu_integrator`), or the frozen fw_disc.DiscImu.
    exact: str, os.PathLike or mapping
        Exact per-point sums with F, F0 (nlos, nl, ny) for the same lines of sight (M424:
        d3200_r4050_N1236544/disc_los8.npz from :func:`ppmpy.synspec.disc.integrate_exact_stream`, shifts rounded
        as the pipeline's; disc_los8_imu.npz for the intensity method; or :func:`exact_continuous`, continuous
        shifts, whose difference includes the rounding).
    sample: str, os.PathLike or mapping
        teff, ur, uth, uph (N,) of that dump (M424: samples_r4050_N1236544/d3200.npz).
    mu, tn, pn: array-like
        (nlos, N) projections (:func:`ppmpy.synspec.sphere.project_los`, 'matmul' as the legacy script).
    label: str
        Run label in the names: V1_<label> (F and F0), V1_<label>_dEW; arrays V1_<label>_F, _F0, _dEW.
    lref: LineSet or array-like, optional
        (nl,) velocity zero points [A] for the EW (default the integrator's 'lines' / 'lref').
    grid: VelocityGrid, optional
        Default ``integ.grid``.
    ew_jacobian: bool
        EW with lambda/lref (fw_disc.diagnostics since 2026-09-29: validate.npz).
    tolerance, dew_tolerance: float, optional
        Default ``tolerances['V1']`` / ``['V1_dEW']``, else :data:`DEFAULT_TOLERANCES`.
    tolerances: dict, optional
        Per-check tolerances (names as :data:`DEFAULT_TOLERANCES`).

    Returns
    -------
    ValidationReport
        Checks V1_<label> (max over F and F0; NaN if either has a NaN) and V1_<label>_dEW [A]; arrays V1_<label>_F,
        V1_<label>_F0, V1_<label>_dEW (nl,) (the validate.npz members); data F, F0 (the pipeline's profiles).

    Validation
    ----------
    M424: V1_flux_F, _F0, _dEW and V1_imu_F, _F0 of validate.npz bit for bit (test_validate.py), the latter with the
    frozen fw_disc.DiscImu and with :class:`ppmpy.synspec.disc.DiscImu` (default options, built by
    :func:`ppmpy.synspec.dumps.imu_integrator`). Synthetic: the ppmpy DiscImu gives the frozen DiscImu's arrays bit
    for bit.
    """
    # PP 2026-10-01: ported from fw_disc_dumps_validate.py:71-87
    T0 = time.time()
    grid = _integ_grid(integ, grid)
    _, lr = _integ_lref(integ, lref)
    Fx, F0x = _exact_arrays(exact)
    smp = _sample_dict(sample)
    F, F0 = _run(integ, smp, mu, tn, pn)
    if Fx.shape != F.shape or F0x.shape != F0.shape:
        raise ValueError("the exact sums have shape {} / {}, the pipeline gives {}".format(Fx.shape, F0x.shape,
                                                                                         F.shape))
    if lr.size != F.shape[1]:
        raise ValueError("need one reference wavelength per line ({}), got {}".format(F.shape[1], lr.size))
    dF, dF0 = _dmax(F, Fx), _dmax(F0, F0x)
    dEW = _dew(F, Fx, grid.y, lr, ew_jacobian)
    names = _line_names(integ, lref, F.shape[1])
    key = "V1_" + label
    det = dict(lines=names, F=dF, F0=dF0, nlos=F.shape[0], npoints=int(smp["teff"].size))
    # PP 2026-10-01: reviewer: Python max() dropped a NaN of dF0; np.maximum keeps it
    checks = [CheckResult(key, float(np.max(np.maximum(dF, dF0))), _tol("V1", tolerance, tolerances),
                          details=dict(det, per_line=np.maximum(dF, dF0))),
              CheckResult(key + "_dEW", dEW.max(), _tol("V1_dEW", dew_tolerance, tolerances), unit="A",
                          details=dict(lines=names, per_line=dEW, ew_jacobian=bool(ew_jacobian)))]
    arrays = {key + "_F": dF, key + "_F0": dF0, key + "_dEW": dEW}
    meta = dict(check="V1", label=label, wall=time.time() - T0, ew_jacobian=bool(ew_jacobian),
                integrator=type(integ).__name__)
    return ValidationReport(checks, meta=meta, arrays=arrays, data={key: dict(F=F, F0=F0)})


# ----------------------------------------------------------------------------------------------
# V2: hold-out
# ----------------------------------------------------------------------------------------------
def _holdout_matrix(MU, SH, fcj, bins, nb, halves, all_rows):
    """
    The sparse weight matrix of one line for the hold-out test (rows x points): per half and line of sight the
    velocity groups (visible points of the half, weight mu F_c) and one row without velocities, then per half
    one row per T_eff' bin (weight 1: the half's library), then (all_rows) one row per bin over all points (the
    in-sample library).

    Returns
    -------
    MC: scipy.sparse.csc_matrix
    W: np.ndarray
        Row sums (of the CSR matrix, as the legacy code).
    layout: dict
        (X, k, 'v') -> (first row, shifts); (X, k, '0') -> row; (X, 'lib') -> first row; ('all', 'lib').
    """
    # PP 2026-10-01: ported from fw_disc_holdout.py:91-110 (same order of rows, columns, values and dtypes); the
    # in-sample rows are new (the legacy script read library_dT10.npz instead; equal bits, see v2_holdout)
    nlos, N = MU.shape
    rows, cols, vals, layout = [], [], [], {}
    r0 = 0
    for X, m in halves:
        for k in range(nlos):
            w = disc_weights(MU[k], fcj) * m
            vis = np.where(w > 0)[0]
            sv, gi = np.unique(SH[k, vis], return_inverse=True)
            rows.append(r0 + gi)
            cols.append(vis)
            vals.append(w[vis])
            layout[X, k, "v"] = (r0, sv)
            r0 += sv.size
            rows.append(np.full(vis.size, r0))
            cols.append(vis)
            vals.append(w[vis])
            layout[X, k, "0"] = r0
            r0 += 1
        idx = np.where(m)[0]
        rows.append(r0 + bins[idx])
        cols.append(idx)
        vals.append(np.ones(idx.size))
        layout[X, "lib"] = r0
        r0 += nb
    if all_rows:
        rows.append(r0 + bins)
        cols.append(np.arange(N))
        vals.append(np.ones(N))
        layout["all", "lib"] = r0
        r0 += nb
    M = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r0, N))
    W = np.asarray(M.sum(axis=1)).ravel()
    return M.tocsc(), W, layout


def _stream_lines(lam3, fn3, mats, grid, lr, block, stride, rows, nproc, start_method, maxtasksperchild, timeout,
                  tmpdir, finish, progress=None, _log=None):
    """
    Stream the profiles through one sparse weight matrix per line: partial sum q of line j adds M[:, block] @ P_block
    over blocks q, q + stride, ...; the partial sums are added in order (q = 0, 1, ...) and ``finish(j, A)`` gets
    each line's sum. Serial, or ``nproc`` pool workers ('fork' / 'spawn') via the machinery of
    :func:`ppmpy.synspec.disc.integrate_exact_stream`, with the same bits.
    """
    # PP 2026-10-01: the streaming loop of disc.integrate_exact_stream (itself ported from fw_disc_los.py:73-83,
    # 111-112), used for fw_disc_holdout.py:57-68,111 (stream(), A = sum(pool.map(stream, range(nproc))))
    N = lam3.shape[0]
    nl = len(mats)
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
                A, acc[0] = acc[0], None
                if _log is not None:
                    _log("line {}: streamed {} profiles through {} weight rows".format(j, N, A.shape[0]))
                finish(j, A)
                mats[j] = None
                del A

    nproc = max(1, min(int(nproc), len(tasks)))
    if nproc <= 1:
        streamer = _ExactStreamer(lam3, fn3, mats, grid.y, lr, block, stride, rows)
        consume(((j, q), streamer.partial(j, q)) for j, q in tasks)
        return
    method = par.get_context(start_method).get_start_method()
    par.login_node_warning(nproc)
    spill = None
    try:
        if method == "fork":
            spec = dict(lam=("array", lam3), fnorm=("array", fn3), mats=list(mats))
        else:
            spill = tempfile.mkdtemp(prefix="synspec_validate_", dir=tmpdir)
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


def v2_holdout(profiles, sample, theta, phi, los, grid, lref, seed=11, nmin=20, block=5000, nproc=1,
               start_method=None, stride=20, library=None, dT=10.0, rows="auto", ew_jacobian=False,
               project_method="matvec", check_teff=True, pad_tol=None, factory=None, maxtasksperchild=None,
               timeout=900.0, tmpdir=None, tolerance=None, dew_tolerance=None, tolerances=None, progress=None,
               log=None):
    """
    V2, the hold-out test of the T_eff' library (port of fw_disc_holdout.py).

    For later dumps the library predicts profiles for T_eff' values whose own models were never computed. This
    test mimics that on the library's own dump: the points are split at random into halves A and B; for each half
    X the exact disc-integrated profiles of X alone (X's own models, the dump's velocities, weight mu F_c, Doppler
    shifts rounded to grid steps and clipped as :meth:`VelocityGrid.shift_steps`) are compared with the prediction
    of :class:`ppmpy.synspec.disc.DiscFlux` from the library of the OTHER half only (``dT`` bins, nodes with >= nmin
    models, linear T_eff' interpolation) applied to X's points; the in-sample prediction (library of all points)
    is the reference. Another integrator built from a flux library (smoothed or corrected nodes) is tested through
    ``factory``.

    Parameters
    ----------
    profiles: ProfileStore, str, os.PathLike, np.load()ed .npz or mapping
        The per-point models: lam, fnorm, fcont (N, nl, nrow), teff (N,) (M424: profiles.npz; a path is
        memory-mapped, :func:`ppmpy.synspec.disc.integrate_exact_stream`).
    sample: str, os.PathLike, mapping or 3-sequence
        u_r, u_theta, u_phi (N,) of that dump [km/s] (M424: samples_r4050_N1236544/d3200.npz); a 'teff' member is
        checked against the models' (``check_teff``).
    theta, phi: array-like or None
        (N,) point coordinates [rad]; None = the profiles' members.
    los: str or array-like
        Lines of sight (:func:`ppmpy.synspec.sphere.project_los`; M424 'thompson2024').
    grid: VelocityGrid
        The velocity grid (M424: ``VelocityGrid()``).
    lref: LineSet or array-like
        (nl,) velocity zero points [A] (line j of the profiles is placed with lref[j]).
    seed: int
        The halves: ``np.random.default_rng(seed).random(N) < 0.5`` is half A (legacy 11).
    nmin: int
        Models per node (:func:`ppmpy.synspec.library.lib_nodes`).
    block, stride: int
        Profiles per streamed block and number of interleaved partial sums (legacy --block and --nproc: the M424
        holdout.npz was made with 5000 and 20). They set the last bits of the exact sums.
    nproc: int
        Worker processes (results independent of nproc and the start method, bit for bit).
    start_method, maxtasksperchild, timeout, tmpdir:
        Pool options, as :func:`ppmpy.synspec.disc.integrate_exact_stream`.
    library: FluxLibrary, str or os.PathLike, optional
        The in-sample library (M424: library_dT10.npz, as the legacy script); its edges bin the halves. None (default)
        builds it in the same pass from all points (float32 profiles, empty bins filled; bit for bit
        ``FluxLibrary.build(block=block, stride=stride)``, so for M424 the same as library_dT10.npz).
    dT: float
        Bin width [K] when ``library`` is None.
    rows: int, 'auto' or None
        Models interpolated at a time inside a block (memory; same bits).
    ew_jacobian: bool
        EW of the dEW checks with lambda/lref (False reproduces holdout.npz, which was written before
        fw_disc.diagnostics got the factor on 2026-09-29).
    project_method: str
        :func:`ppmpy.synspec.sphere.project_los` method ('matvec' = fw_disc.mu_vlos, as the legacy script).
    check_teff: bool
        Require the sample's T_eff' to match the models' (:func:`ppmpy.synspec.disc.integrate_exact_stream`).
    pad_tol: float, optional
        :class:`ppmpy.synspec.disc.DiscFlux` zero-padding tolerance (default its own).
    factory: callable, optional
        factory(FluxLibrary) -> integrator (DiscFlux duck type, on ``grid``), called for each half's library and the
        in-sample one (under a 1-thread BLAS limit), e.g. ``lambda L: flux_integrator(L, nmin=20, smooth=335)``
        (:func:`ppmpy.synspec.dumps.flux_integrator`) for the flux_sm335 run, or with ``corr`` (per library bin, the
        edges of all halves are the same) for flux_lamfix. Default ``DiscFlux(lib_nodes(L, nmin=nmin), grid,
        pad_tol)``, the legacy method (then ``nmin`` and ``pad_tol`` apply; with a factory they are not used).
    tolerance, dew_tolerance, tolerances:
        Default ``tolerances['V2']`` / ``['V2_dEW']``, else :data:`DEFAULT_TOLERANCES`; used for the hold-out and
        the in-sample checks.
    progress, log: callable, optional
        progress(tasks_done, tasks_total); log(message).

    Returns
    -------
    ValidationReport
        Checks V2_holdout, V2_holdout_dEW, V2_insample, V2_insample_dEW (max over both halves, F and F0); arrays =
        the members of holdout.npz in its order: lines, seed, nmin, then per half X in (A, B): holdout_X_F,
        holdout_X_F0, holdout_X_dEW, insample_X_F, insample_X_F0, insample_X_dEW (nl,), exact_X_F (nlos, nl, ny)
        float32; data: half (bool (N,), True = A), exact (X -> (F, F0) float64), libraries (X -> FluxLibrary,
        'all' -> the in-sample one), predictions ((tag, X) -> (F, F0)).

    Memory
    ------
    As :func:`ppmpy.synspec.disc.integrate_exact_stream` with about twice as many weight rows per line (per half:
    the velocity groups and the library rows; M424 ~6000 rows, 0.26 GB per partial sum); the parent holds at most
    2 nproc partial sums.

    Validation
    ----------
    Synthetic: the arrays of the frozen fw_disc_holdout.py bit for bit (any nproc, fork and spawn). M424: holdout.npz
    bit for bit, both with ``library`` = library_dT10.npz and with the library built in the pass
    (test_validate.py).
    """
    # PP 2026-10-01: ported from fw_disc_holdout.py:39-143 (data, MU/V/SH, bins, halves, per-line weight matrices,
    # streamed sums, exact profiles and half libraries, predictions, R); in-sample library optionally from the pass
    T0 = time.time()
    _log = _logger(log, T0)
    if not isinstance(grid, VelocityGrid):
        raise TypeError("grid must be a VelocityGrid, got {}".format(type(grid).__name__))
    y, ny = grid.y, grid.ny
    skip = [k for k, x in (("theta", theta), ("phi", phi)) if x is not None]
    src = _profile_arrays(profiles, skip=skip)
    lam3, fn3 = src["lam"], src["fnorm"]
    if lam3.ndim != 3 or lam3.shape != fn3.shape:
        raise ValueError("lam and fnorm must have one shape (N, nl, nrow)")
    N, nl = lam3.shape[:2]
    names_l = getattr(lref, "names", None)
    lr = np.atleast_1d(np.asarray(getattr(lref, "lref", lref), dtype=np.float64))
    if lr.shape != (nl,):
        raise ValueError("need one reference wavelength per line ({}), got {}".format(nl, lr.size))
    if names_l is not None and src["lines"] is not None and list(names_l) != src["lines"]:
        raise ValueError("the LineSet's lines {} are not the profiles' lines {} (in this order)".format(
            list(names_l), src["lines"]))
    names = list(names_l) if names_l is not None else (src["lines"] or ["line{}".format(j) for j in range(nl)])
    for k in ("teff", "fcont"):
        if k not in src:
            raise ValueError("profiles need a member {!r}".format(k))
    teff = np.asarray(src["teff"])
    theta = np.asarray(src["theta"] if theta is None else theta)
    phi = np.asarray(src["phi"] if phi is None else phi)
    fc_all = _fc0(src["fcont"])
    (ur, uth, uph), vteff, vpath = _velocities(sample)
    for nm, a in (("teff", teff), ("theta", theta), ("phi", phi), ("ur", ur), ("uth", uth), ("uph", uph)):
        if np.shape(a) != (N,):
            raise ValueError("{} must have shape (N,) = ({},), got {}".format(nm, N, np.shape(a)))
    if check_teff and vteff is not None:
        _check_teff(vteff, teff, src.get("teff_nudge"), vpath)
    block, stride = int(block), int(stride)
    if block < 1 or stride < 1:
        raise ValueError("block and stride must be >= 1")
    rows = _rows(rows, ny)

    # lines of sight, velocities, shifts (fw_disc_holdout.py:45-50)
    L = _los_vectors(los)
    nlos = L.shape[0]
    MU, TN, PN = project_los(theta, phi, L, method=project_method)
    V = los_velocity(ur, uth, uph, MU, TN, PN)
    del TN, PN
    SH = grid.shift_steps(V)[0]
    # bins and halves (fw_disc_holdout.py:51-56)
    if library is not None:
        lib_all = library if isinstance(library, FluxLibrary) else FluxLibrary.load(library)
        edges, dT = lib_all.edges, lib_all.dT
        if lib_all.prof.shape[1:] != (nl, ny):
            raise ValueError("library profiles {} do not match {} lines x {} grid points".format(
                lib_all.prof.shape, nl, ny))
    else:
        lib_all = None
        edges = teff_edges(teff, dT)
    nb = edges.size - 1
    bins = teff_bins(teff, edges)
    half = np.random.default_rng(seed).random(N) < 0.5          # True: half A
    halves = [("A", half), ("B", ~half)]
    cnt = {X: np.bincount(bins[m], minlength=nb).astype(float) for X, m in halves}
    libs = {X: dict(edges=edges, prof=np.zeros((nb, nl, ny)), fc=np.zeros((nb, nl)), count=cnt[X],
                    tmean=np.bincount(bins[m], weights=teff[m], minlength=nb) / np.maximum(np.bincount(bins[m],
                                                                                                    minlength=nb), 1))
            for X, m in halves}
    cnt_all = np.bincount(bins, minlength=nb).astype(float)
    ok_all = cnt_all > 0
    lprof_all, lfc_all = (np.zeros((nb, nl, ny)), np.zeros((nb, nl))) if lib_all is None else (None, None)
    exact = {X: (np.zeros((nlos, nl, ny)), np.zeros((nlos, nl, ny))) for X, _ in halves}
    _log("{} points ({} in half A), {} lines of sight, {} lines, {} bins; v_los {:.1f} .. {:.1f} km/s; {} workers, "
         "stride {}, block {}".format(N, int(half.sum()), nlos, nl, nb, V.min(), V.max(), nproc, stride, block))

    mats, Ws, lays = [], [], []
    for j in range(nl):
        MC, W, lay = _holdout_matrix(MU, SH, fc_all[:, j], bins, nb, halves, lib_all is None)
        mats.append(MC)
        Ws.append(W)
        lays.append(lay)

    def finish(j, A):
        # fw_disc_holdout.py:113-121
        W, lay = Ws[j], lays[j]
        for X, m in halves:
            for k in range(nlos):
                g0, sv = lay[X, k, "v"]
                exact[X][0][k, j] = 1.0 - _shift_add(A[g0:g0 + sv.size], W[g0:g0 + sv.size], sv, ny) \
                    / W[g0:g0 + sv.size].sum()
                r = lay[X, k, "0"]
                exact[X][1][k, j] = A[r] / W[r]
            Lb = libs[X]
            ok = Lb["count"] > 0
            r = lay[X, "lib"]
            Lb["prof"][ok, j] = A[r:r + nb][ok] / Lb["count"][ok, None]
            Lb["fc"][:, j] = np.bincount(bins[m], weights=fc_all[m, j], minlength=nb) / np.maximum(Lb["count"], 1)
        if lib_all is None:
            # disc.integrate_exact_stream finish() (fw_disc_los.py:134-144 library rows)
            r = lay["all", "lib"]
            lprof_all[ok_all, j] = A[r:r + nb][ok_all] / cnt_all[ok_all, None]
            lfc_all[:, j] = np.bincount(bins, weights=fc_all[:, j], minlength=nb) / np.maximum(cnt_all, 1)

    _stream_lines(lam3, fn3, mats, grid, lr, block, stride, rows, nproc, start_method, maxtasksperchild, timeout,
                  tmpdir, finish, progress=progress, _log=_log)
    mats = None
    if lib_all is None:
        # disc.integrate_exact_stream library assembly (fw_disc_los.py:147-155): empty bins copy the nearest filled bin
        filled = np.where(ok_all)[0]
        for i in np.where(~ok_all)[0]:
            kk = filled[np.argmin(np.abs(filled - i))]
            lprof_all[i], lfc_all[i] = lprof_all[kk], lfc_all[kk]
        tsum = np.bincount(bins, weights=teff, minlength=nb)
        tmean = np.where(ok_all, tsum / np.maximum(cnt_all, 1), 0.5 * (edges[:-1] + edges[1:]))
        lib_all = FluxLibrary(edges, tmean, cnt_all, lprof_all.astype(np.float32), lfc_all, dT,
                              params=dict(dT=float(dT), block=block, stride=stride,
                                          source="synspec.validate.v2_holdout"))
        del lprof_all
    hlibs = {X: FluxLibrary(edges, Lb["tmean"], Lb["count"], Lb["prof"], Lb["fc"], dT,
                            params=dict(dT=float(dT), half=X, seed=int(seed), fill_empty=False))
             for X, Lb in libs.items()}

    # predictions (fw_disc_holdout.py:123-141)
    if factory is None:
        dkw = {} if pad_tol is None else dict(pad_tol=pad_tol)

        def factory(L):
            # DiscFlux looked up at call time (module global), so it can be replaced for tests
            return DiscFlux(lib_nodes(L, nmin=nmin), grid=grid, **dkw)
        fname = "DiscFlux(lib_nodes(nmin={}))".format(int(nmin))
    else:
        fname = getattr(factory, "__qualname__", type(factory).__name__)
    with _blas_limit(1):
        fx_all = factory(lib_all)
    R = dict(lines=np.array(names), seed=int(seed), nmin=int(nmin))
    preds = {}
    for X, m in halves:
        other = "B" if X == "A" else "A"
        with _blas_limit(1):
            fx_other = factory(hlibs[other])
        for tag, fx in (("holdout", fx_other), ("insample", fx_all)):
            k0, k1, w = fx.pairs(teff)
            F, F0 = np.zeros((nlos, nl, ny)), np.zeros((nlos, nl, ny))
            for k in range(nlos):
                F[k], F0[k] = fx(np.where(m, MU[k], 0.0), V[k], k0, k1, w)[:2]
            R["{}_{}_F".format(tag, X)] = _dmax(F, exact[X][0])
            R["{}_{}_F0".format(tag, X)] = _dmax(F0, exact[X][1])
            R["{}_{}_dEW".format(tag, X)] = _dew(F, exact[X][0], y, lr, ew_jacobian)
            preds[tag, X] = (F, F0)
            _log("half {} ({} points), library from {}: max|dF| {} (no velocities {}); max|dEW| {} A".format(
                X, int(m.sum()), "half " + other if tag == "holdout" else "all points",
                _fmt(R["{}_{}_F".format(tag, X)]), _fmt(R["{}_{}_F0".format(tag, X)]),
                _fmt(R["{}_{}_dEW".format(tag, X)])))
        R["exact_{}_F".format(X)] = exact[X][0].astype(np.float32)

    checks = []
    for tag, nm in (("holdout", "V2_holdout"), ("insample", "V2_insample")):
        pl = np.max([np.maximum(R["{}_{}_F".format(tag, X)], R["{}_{}_F0".format(tag, X)]) for X, _ in halves], axis=0)
        pe = np.max([R["{}_{}_dEW".format(tag, X)] for X, _ in halves], axis=0)
        checks.append(CheckResult(nm, pl.max(), _tol("V2", tolerance, tolerances),
                                  details=dict(lines=names, per_line=pl, halves=[int(half.sum()), int((~half).sum())],
                                               **{"{}_F".format(X): R["{}_{}_F".format(tag, X)] for X, _ in halves},
                                               **{"{}_F0".format(X): R["{}_{}_F0".format(tag, X)] for X, _ in halves})))
        checks.append(CheckResult(nm + "_dEW", pe.max(), _tol("V2_dEW", dew_tolerance, tolerances), unit="A",
                                  details=dict(lines=names, per_line=pe, ew_jacobian=bool(ew_jacobian))))
    meta = dict(check="V2", seed=int(seed), nmin=int(nmin), block=block, stride=stride, rows=rows, nproc=int(nproc),
                dT=float(dT), library=None if library is None else (library if isinstance(library, (str, os.PathLike))
                                                                     else "FluxLibrary"),
                ew_jacobian=bool(ew_jacobian), project_method=project_method, los=L.tolist(), lref=lr.tolist(),
                profiles=src["path"], velocities=vpath, factory=fname, wall=time.time() - T0)
    _log("done: {:.0f} s".format(meta["wall"]))
    data = dict(half=half, exact=exact, libraries=dict(hlibs, all=lib_all), predictions=preds)
    return ValidationReport(checks, meta=meta, arrays=R, data=data)


# ----------------------------------------------------------------------------------------------
# V3-V6
# ----------------------------------------------------------------------------------------------
def _row_teff(args):
    path, trange = args
    with np.load(path) as z:
        t = z["teff"].astype(np.float64)
    n_lo = np.nan if trange is None else float((t < trange[0]).sum())
    n_hi = np.nan if trange is None else float((t > trange[1]).sum())
    return t.min(), t.max(), n_lo, n_hi, t.std()


def teff_ranges(samples_dir, dumps, trange=None, nproc=1, pattern=SAMPLE_PATTERN, start_method=None, timeout=900.0):
    """
    T_eff' range of every dump's samples (to pick the V3 dumps; :func:`v3_select`).

    Parameters
    ----------
    samples_dir: str or os.PathLike
        Per-dump sample files (only 'teff' is read; M424: 4.9 MB per dump).
    dumps: iterable of int
    trange: (float, float), optional
        n_lo, n_hi count the points below trange[0] / above trange[1] (compared in float64). The legacy table
        (M424 teff_ranges_all_dumps.npy) used the T_eff' range of the library's models (min and max of profiles.npz
        'teff': 35402.107-38905.438 K), not the node range. None: n_lo, n_hi are NaN.
    nproc: int
        Worker processes (each reads whole files; the work is I/O).
    pattern: str
        Sample file name pattern.
    start_method, timeout:
        Pool options (:func:`ppmpy.synspec.parallel.make_pool`, :func:`ppmpy.synspec.parallel.imap_watchdog`).

    Returns
    -------
    np.ndarray
        (ndumps, 6) float64: dump, tmin, tmax, n_lo, n_hi, std (of the float64 T_eff', ddof 0), in the order of
        ``dumps``.

    Validation
    ----------
    The first six columns of the M424 teff_ranges_all_dumps.npy bit for bit (its two further columns are counts
    of unknown definition, not used by V3).
    """
    # PP 2026-10-01: new; reproduces columns 0-5 of /scratch/ppathak/fastwind_sphere/teff_ranges_all_dumps.npy (written
    # 2026-09-29 by an unrecorded command line; read by fw_disc_dumps_validate.py --ranges)
    dl = [int(d) for d in dumps]
    tr = None if trange is None else (float(trange[0]), float(trange[1]))
    args = [(sample_path(samples_dir, d, pattern), tr) for d in dl]
    out = np.zeros((len(dl), 6))
    out[:, 0] = dl
    nproc = max(1, min(int(nproc), len(dl)))
    if nproc <= 1:
        for i, a in enumerate(args):
            out[i, 1:] = _row_teff(a)
        return out
    with par.make_pool(nproc, start_method=start_method) as pool:
        for i, row in par.imap_watchdog(pool, _row_teff_indexed, list(enumerate(args)), timeout=timeout):
            out[i, 1:] = row
    return out


def _row_teff_indexed(item):
    i, a = item
    return i, _row_teff(a)


def v3_select(ranges):
    """
    The V3 dumps from a :func:`teff_ranges` table: the dump with the most points beyond trange (n_lo + n_hi), the
    one with the coolest and the one with the hottest point, sorted, without repeats (fw_disc_dumps_validate.py
    --ranges; M424: 3334, 4169, 4391).

    Raises
    ------
    ValueError
        The table is empty or its n_lo, n_hi (columns 3, 4) or tmin, tmax are not finite (a table made without
        ``trange``).
    """
    # PP 2026-10-01: ported from fw_disc_dumps_validate.py:107-109
    rg = np.asarray(ranges, dtype=np.float64)
    if rg.ndim != 2 or rg.shape[0] == 0 or rg.shape[1] < 5:
        raise ValueError("need a teff_ranges table (ndumps >= 1, >= 5 columns), got shape {}".format(rg.shape))
    if not np.all(np.isfinite(rg[:, 1:5])):
        # PP 2026-10-01: reviewer: argmax over NaN counts silently picked the first dump
        raise ValueError("the teff_ranges table has non-finite tmin / tmax / n_lo / n_hi (made without trange?): "
                         "pass trange to teff_ranges")
    return sorted({int(rg[np.argmax(rg[:, 3] + rg[:, 4]), 0]), int(rg[np.argmin(rg[:, 1]), 0]),
                   int(rg[np.argmax(rg[:, 2]), 0])})


def v3_tails(integ, samples, dumps, mu, tn, pn, margin=300.0, tolerance=None, tolerances=None, lref=None,
             pattern=SAMPLE_PATTERN, log=None):
    """
    V3: points beyond the node range, which the pipeline clamps to the end node. Per dump, the profiles with
    clamping (F) vs (a) leaving these points out (informational: shows that they matter at all) and (b)
    extrapolating them linearly with the slope between the end node and the first node at least ``margin`` inside
    (the check: the uncertainty of the clamped tails).

    Parameters
    ----------
    integ: object
        Integrator (DiscFlux duck type; its pairs may carry weights < 0 or > 1 for (b)).
    samples: str, os.PathLike, mapping or callable
        Per-dump samples (module notes).
    dumps: sequence of int
        Dumps to test (M424: :func:`v3_select` of :func:`teff_ranges`, i.e. 3334, 4169, 4391; the legacy default
        without a table was 4391, 4169).
    mu, tn, pn: array-like
        (nlos, N) projections.
    margin: float
        [K] > 0: the extrapolation slope uses the end node and node searchsorted(t, t[0] + margin) (low end), node
        searchsorted(t, t[-1] - margin) - 1 (high end) (clipped to the nodes; never the end node itself).
    tolerance, tolerances:
        For V3_extrap (default :data:`DEFAULT_TOLERANCES` 'V3'); V3_leaveout is informational.

    Returns
    -------
    ValidationReport
        Checks V3_extrap, V3_leaveout (info); arrays V3_dumps, V3_leaveout (nd, nl), V3_extrap (nd, nl), V3_n (nd, 2)
        (points below / above the nodes; float64 as validate.npz).

    Raises
    ------
    ValueError
        No dumps, or margin not > 0 (the slope would use the end node twice: a division by zero).
    """
    # PP 2026-10-01: ported from fw_disc_dumps_validate.py:105-131 (without the ranges file: dumps are given)
    T0 = time.time()
    _log = _logger(log, T0)
    dl = _dumps_given(dumps, "V3")
    margin = float(margin)
    if not (margin > 0 and np.isfinite(margin)):
        raise ValueError("margin must be > 0 and finite, got {!r}".format(margin))
    tn_ = np.asarray(integ.t, dtype=np.float64)
    nn = tn_.size
    jlo = min(int(np.searchsorted(tn_, tn_[0] + margin)), nn - 1)
    jhi = max(int(np.searchsorted(tn_, tn_[-1] - margin)) - 1, 0)
    if not (0 < jlo and jhi < nn - 1):                               # cannot happen for margin > 0 and nn >= 2
        raise ValueError("no extrapolation node for margin {:g} K (nodes {}, {} of {})".format(margin, jlo, jhi, nn))
    lo_out, ex_out, n_out = None, None, np.zeros((len(dl), 2))
    for i, d in enumerate(dl):
        smp = _get_sample(samples, d, pattern)
        teff = smp["teff"]
        lo, hi = teff < tn_[0], teff > tn_[-1]
        n_out[i] = lo.sum(), hi.sum()
        F, _ = _run(integ, smp, mu, tn, pn)
        with np.errstate(invalid="ignore", divide="ignore"):         # NaN if no point is inside the node range
            Fo, _ = _run(integ, smp, mu, tn, pn, mask=~(lo | hi))
        k0, k1, w = integ.pairs(teff)
        k0, k1, w = np.array(k0, copy=True), np.array(k1, copy=True), np.array(w, dtype=np.float64, copy=True)
        k0[lo], k1[lo], w[lo] = 0, jlo, (teff[lo] - tn_[0]) / (tn_[jlo] - tn_[0])                # w < 0
        k0[hi], k1[hi], w[hi] = jhi, nn - 1, (teff[hi] - tn_[jhi]) / (tn_[-1] - tn_[jhi])        # w > 1
        Fe, _ = _run(integ, smp, mu, tn, pn, pairs=(k0, k1, w))
        if lo_out is None:
            lo_out, ex_out = np.zeros((len(dl), F.shape[1])), np.zeros((len(dl), F.shape[1]))
        lo_out[i], ex_out[i] = _dmax(F, Fo), _dmax(F, Fe)
        _log("V3 dump {}: {} points below {:.0f} K (min {:.0f}), {} above {:.0f} K (max {:.0f}); clamped vs left out: "
             "max|dF| {}; vs extrapolated: {}".format(d, int(lo.sum()), tn_[0], teff.min(), int(hi.sum()), tn_[-1],
                                                     teff.max(), _fmt(lo_out[i]), _fmt(ex_out[i])))
    names = _line_names(integ, lref, ex_out.shape[1])
    checks = [CheckResult("V3_extrap", ex_out.max(), _tol("V3", tolerance, tolerances),
                          details=dict(lines=names, per_line=ex_out.max(axis=0), dumps=dl,
                                       per_dump=ex_out, n_out=n_out, margin=float(margin), nodes=[jlo, jhi])),
              CheckResult("V3_leaveout", lo_out.max(), None,
                          details=dict(lines=names, per_line=lo_out.max(axis=0), dumps=dl,
                                       per_dump=lo_out, n_out=n_out))]
    arrays = dict(V3_dumps=np.array(dl, dtype=np.int64), V3_leaveout=lo_out, V3_extrap=ex_out, V3_n=n_out)
    return ValidationReport(checks, meta=dict(check="V3", dumps=dl, margin=float(margin), wall=time.time() - T0),
                            arrays=arrays)


def _load_corr(corr, corr_key, lib):
    """A per-bin profile correction (nb, nl, ny) from an array or an .npz (member corr_key); None stays None."""
    if corr is None:
        return None
    if isinstance(corr, (str, os.PathLike)):
        with np.load(os.fspath(corr)) as z:
            corr = z[corr_key]
    corr = np.asarray(corr)
    if corr.shape != np.shape(lib["prof"]):
        raise ValueError("corr must have the shape of the library profiles {}, got {}".format(np.shape(lib["prof"]),
                                                                                           corr.shape))
    return corr


def _corr_consistent(integ, corr, check):
    """Raise if the integrator's nodes carry a correction (node_params corr) and none is given, or the reverse."""
    p = _node_params(integ)
    if "corr" not in p:
        return
    if bool(p["corr"]) and corr is None:
        raise ValueError("{}: the integrator's nodes carry a profile correction (node_params corr=True, e.g. "
                         "flux_lamfix): pass the same corr (array or .npz)".format(check))
    if not p["corr"] and corr is not None:
        raise ValueError("{}: a corr was given, but the integrator's nodes carry none (node_params corr=False)"
                         .format(check))


def v4_nearest(integ, library, samples, dumps, mu, tn, pn, grid=None, corr=None, corr_key="corr", tolerance=None,
               tolerances=None, lref=None, pattern=SAMPLE_PATTERN, log=None):
    """
    V4: linear T_eff' interpolation between nodes (the integrator) vs the nearest library bin
    (:func:`ppmpy.synspec.disc.integrate_library_nearest`, weights mu F_c of the bin), per dump.

    Assumes a flux integrator DiscFlux(lib_nodes(library, nmin, smooth, corr)) (module notes: applicability).

    Parameters
    ----------
    integ: object
        Integrator under test (DiscFlux duck type).
    library: FluxLibrary, str or os.PathLike
        The flux library the nodes were built from (M424 library_dT10.npz).
    samples, dumps:
        Per-dump samples and the dumps (M424: 3600, 4000, 4400, 4800).
    mu, tn, pn: array-like
        (nlos, N) projections.
    grid: VelocityGrid, optional
        Default ``integ.grid``.
    corr: np.ndarray, str or os.PathLike, optional
        The per-bin profile correction the integrator's nodes were built with (lib_nodes ``corr``; M424
        flux_lamfix: lamfix_dT10.npz), added to the bin profiles before the nearest-bin sum. Required when the
        integrator's ``node_params`` record a correction, refused when they record none.
    corr_key: str
        Member of a corr .npz.
    tolerance, tolerances:
        Default ``tolerances['V4']``, else :data:`DEFAULT_TOLERANCES`. With smoothed nodes (``node_params`` smooth
        > 0, e.g. flux_sm335) V4 measures smoothing and interpolation together and is informational unless
        ``tolerance`` is given explicitly.

    Returns
    -------
    ValidationReport
        Check V4; arrays V4_dumps, V4 (nd, nl); data V4_F_first: the integrator's F of the first dump (V5's
        reference in the legacy script).

    Raises
    ------
    ValueError
        No dumps; corr missing or superfluous (see ``corr``).
    """
    # PP 2026-10-01: ported from fw_disc_dumps_validate.py:89-101; corr and the smoothed case are new (reviewer:
    # V4 of flux_sm335 / flux_lamfix measured the variant, not the interpolation)
    T0 = time.time()
    _log = _logger(log, T0)
    grid = _integ_grid(integ, grid)
    lib = library if isinstance(library, FluxLibrary) else FluxLibrary.load(library)
    dl = _dumps_given(dumps, "V4")
    corr = _load_corr(corr, corr_key, lib)
    _corr_consistent(integ, corr, "V4")
    smooth = float(_node_params(integ).get("smooth", 0.0) or 0.0)
    nlib = lib if corr is None else dict(edges=lib.edges, prof=np.asarray(lib.prof).astype(np.float64) + corr)
    out, first = None, None
    for i, d in enumerate(dl):
        smp = _get_sample(samples, d, pattern)
        teff, ur, uth, uph = smp["teff"], smp["ur"], smp["uth"], smp["uph"]
        mu_, tn_, pn_ = _projections(mu, tn, pn, teff.size)
        F, _ = _run(integ, smp, mu_, tn_, pn_)
        b = teff_bins(teff, lib.edges)
        Fn = np.array([integrate_library_nearest(nlib, teff, los_velocity(ur, uth, uph, mu_[k], tn_[k], pn_[k]),
                                                 disc_weights(mu_[k], 1.0)[:, None] * lib["fc"][b], grid)
                       for k in range(mu_.shape[0])])
        if out is None:
            out = np.zeros((len(dl), F.shape[1]))
            first = F
        out[i] = _dmax(F, Fn)
        _log("V4 dump {}: T_eff' interpolation vs nearest bin: max|dF| {}".format(d, _fmt(out[i])))
    names = _line_names(integ, lref, out.shape[1])
    det = dict(lines=names, per_line=out.max(axis=0), dumps=dl, per_dump=out, corr=corr is not None, smooth=smooth)
    tol = _tol("V4", tolerance, tolerances)
    if smooth > 0 and tolerance is None:
        tol = None
        det["note"] = ("smoothed nodes (smooth {:g} K): V4 measures the smoothing and the interpolation together; "
                       "informational".format(smooth))
    chk = CheckResult("V4", out.max(), tol, details=det)
    return ValidationReport([chk], meta=dict(check="V4", dumps=dl, corr=corr is not None, smooth=smooth,
                                             wall=time.time() - T0),
                            arrays=dict(V4_dumps=np.array(dl, dtype=np.int64), V4=out),
                            data=dict(V4_F_first=first))


def v5_node_merging(integ, library, samples, dump, mu, tn, pn, nmins=(1, 5, 100), factory=None, grid=None,
                    corr=None, corr_key="corr", tolerance=None, tolerances=None, lref=None, pattern=SAMPLE_PATTERN,
                    log=None):
    """
    V5: nodes merged with other minimum model counts vs the integrator's own nodes, for one dump.

    Assumes a flux integrator DiscFlux(lib_nodes(library, nmin, smooth, corr)) (module notes: applicability).

    Parameters
    ----------
    integ: object
        Integrator under test (M424: nmin 20); its profiles are the reference.
    library: FluxLibrary, str or os.PathLike
        The flux library.
    samples, dump:
        Per-dump samples and the dump (legacy: the first V4 dump, M424 3600).
    mu, tn, pn: array-like
        (nlos, N) projections.
    nmins: sequence of int
        Other node merging (legacy 1, 5, 100; at least one).
    factory: callable, optional
        factory(nmin) -> integrator; default :func:`ppmpy.synspec.dumps.flux_integrator` (library, nmin, grid, the
        integrator's pad_tol and smoothing ``node_params['smooth']``, and ``corr``).
    grid: VelocityGrid, optional
        Default ``integ.grid``.
    corr: np.ndarray, str or os.PathLike, optional
        The per-bin profile correction of the integrator's nodes, for the default factory: required when its
        ``node_params`` record a correction (else ValueError: the nodes rebuilt without it would differ by the
        correction, not by the merging), refused when they record none.
    corr_key: str
        Member of a corr .npz.

    Returns
    -------
    ValidationReport
        Check V5 (max over nmins); arrays V5_nmin<n> (nl,).

    Raises
    ------
    ValueError
        No nmins; corr missing or superfluous (default factory only).
    """
    # PP 2026-10-01: ported from fw_disc_dumps_validate.py:100-103 (DiscFlux(lib_nodes(lib, nmin=nm)) via
    # flux_integrator, which builds the nodes under a 1-thread BLAS limit: the same bits for these unsmoothed nodes);
    # smooth and corr of the integrator's nodes are new (reviewer: V5 of flux_sm335 / flux_lamfix rebuilt the plain
    # nodes and measured the variant)
    T0 = time.time()
    _log = _logger(log, T0)
    grid = _integ_grid(integ, grid)
    nmins = [int(n) for n in nmins]
    if not nmins:
        raise ValueError("V5: no nmins given")
    smooth = float(_node_params(integ).get("smooth", 0.0) or 0.0)
    if factory is None:
        lib = library if isinstance(library, FluxLibrary) else FluxLibrary.load(library)
        corr = _load_corr(corr, corr_key, lib)
        _corr_consistent(integ, corr, "V5")
        kw = {} if getattr(integ, "pad_tol", None) is None else dict(pad_tol=integ.pad_tol)

        def factory(nm):
            return flux_integrator(lib, nmin=nm, smooth=smooth, corr=corr, grid=grid, **kw)
    elif corr is not None:
        raise ValueError("V5: corr applies to the default factory only")
    smp = _get_sample(samples, int(dump), pattern)
    Fref, _ = _run(integ, smp, mu, tn, pn)
    arrays, vals = {}, []
    for nm in nmins:
        F, _ = _run(factory(int(nm)), smp, mu, tn, pn)
        arrays["V5_nmin{}".format(int(nm))] = _dmax(F, Fref)
        vals.append(arrays["V5_nmin{}".format(int(nm))])
        _log("V5 dump {}: nmin {}: max|dF| {}".format(dump, nm, _fmt(vals[-1])))
    vals = np.array(vals)
    names = _line_names(integ, lref, Fref.shape[1])
    chk = CheckResult("V5", vals.max(), _tol("V5", tolerance, tolerances),
                      details=dict(lines=names, per_line=vals.max(axis=0), dump=int(dump), nmins=nmins, per_nmin=vals,
                                   nmin_ref=_node_params(integ).get("nmin"), smooth=smooth))
    return ValidationReport([chk], meta=dict(check="V5", dump=int(dump), nmins=nmins, smooth=smooth,
                                             wall=time.time() - T0), arrays=arrays)


def v6_rounding(integ, nodes, samples, dumps, mu, tn, pn, nsub=20000, seed=7, chunk=2000, grid=None, tolerance=None,
                tolerances=None, lref=None, pattern=SAMPLE_PATTERN, log=None):
    """
    V6: Doppler shifts rounded to whole grid steps (the integrator) vs continuous shifts, on a random subset of the
    visible points of each dump (line of sight i mod nlos for the i-th dump): the subset's profile from the
    integrator (other points hidden) vs a direct per-point sum of the interpolated node profiles, each placed on its
    own continuously shifted abscissa (:func:`ppmpy.synspec.spectral.interp_rows`).

    Parameters
    ----------
    integ: object
        Integrator under test (DiscFlux duck type).
    nodes: LibraryNodes or mapping
        Its nodes (t, prof (nn, nl, ny), fc (nn, nl)); the direct sum interpolates them with
        :func:`ppmpy.synspec.library.node_pairs` of their own t (the integrator's pairs for intact nodes). Checked
        to be the integrator's (the same t, and fc if the integrator has fc; ValueError otherwise).
    samples, dumps:
        Per-dump samples and the dumps (M424: 3600, 4000, 4400, 4800).
    mu, tn, pn: array-like
        (nlos, N) projections.
    nsub: int
        Points per subset (legacy 20000; capped at the visible points). The rounding errors of the points are random,
        so V6 falls with the subset: M424 dumps 3600 / 4000 give 2.6e-5 / 2.2e-5 with 20000 points and 4.2e-6 / 7.8e-6
        with all ~618000 visible points (nsub >= N; 7 min per dump in one process, interpolation of every point).
    seed: int
        One ``np.random.default_rng(seed)`` draws the subsets of all dumps in order (legacy 7).
    chunk: int
        Points per interpolation call (legacy 2000; sets the interpolation offsets, i.e. the last bits).
    grid: VelocityGrid, optional
        Default ``integ.grid``.

    Returns
    -------
    ValidationReport
        Check V6; arrays V6 (nd, nl).

    Raises
    ------
    ValueError
        No dumps, or nodes that are not the integrator's.
    """
    # PP 2026-10-01: ported from fw_disc_dumps_validate.py:133-161
    T0 = time.time()
    _log = _logger(log, T0)
    grid = _integ_grid(integ, grid)
    dl = _dumps_given(dumps, "V6")
    _check_integ_nodes(integ, nodes, "V6")
    y = grid.y
    t = np.asarray(nodes["t"], dtype=np.float64)
    P, FC = np.asarray(nodes["prof"]), np.asarray(nodes["fc"])
    rng = np.random.default_rng(seed)
    nl = P.shape[1]
    out = np.zeros((len(dl), nl))
    mu, tn, pn = _projections(mu, tn, pn)
    chunk = max(1, int(chunk))
    for i, d in enumerate(dl):
        smp = _get_sample(samples, d, pattern)
        teff, ur, uth, uph = smp["teff"], smp["ur"], smp["uth"], smp["uph"]
        k = i % mu.shape[0]
        v = los_velocity(ur, uth, uph, mu[k], tn[k], pn[k])
        visible = np.where(mu[k] > 0)[0]
        sub = np.sort(rng.choice(visible, min(int(nsub), visible.size), replace=False))
        k0, k1, w = node_pairs(t, teff[sub])
        mask = np.zeros(teff.size, bool)
        mask[sub] = True
        Fh = integ(np.where(mask, mu[k], 0.0), v, *integ.pairs(teff))[0]
        s = -C_KMS * np.log(1.0 - v[sub] / C_KMS)
        for j in range(nl):
            fc = (1 - w) * FC[k0, j] + w * FC[k1, j]
            num = np.zeros(y.size)
            for c0 in range(0, sub.size, chunk):
                c = slice(c0, c0 + chunk)
                line = (1 - w[c, None]) * FC[k0[c], j, None] * P[k0[c], j] \
                    + w[c, None] * FC[k1[c], j, None] * P[k1[c], j]
                num += mu[k][sub[c]] @ interp_rows(y[None, :] - s[c, None], line, y)
            Fd = num / np.sum(mu[k][sub] * fc)
            out[i, j] = np.abs(Fh[j] - Fd).max()
        _log("V6 dump {}, los{}, {} points: rounded vs continuous shifts: max|dF| {}".format(d, k + 1, sub.size,
                                                                                            _fmt(out[i])))
    names = _line_names(integ, lref, nl)
    chk = CheckResult("V6", out.max(), _tol("V6", tolerance, tolerances),
                      details=dict(lines=names, per_line=out.max(axis=0), dumps=dl, per_dump=out, nsub=int(nsub),
                                   seed=int(seed)))
    return ValidationReport([chk], meta=dict(check="V6", dumps=dl, nsub=int(nsub), seed=int(seed), chunk=chunk,
                                             wall=time.time() - T0), arrays=dict(V6=out))


# ----------------------------------------------------------------------------------------------
# brute force
# ----------------------------------------------------------------------------------------------
def _brute_line(dep, t, fcj, m, v, teff, dv, nshift, chunk):
    """
    F, F0 (ny,) of one line and line of sight, point by point: each visible point's interpolated node depth
    w0 d_k0 + w1 d_k1 (w0 = mu (1 - a) F_c,k0, w1 = mu a F_c,k1) is shifted by its rounded Doppler shift and added.
    """
    # PP 2026-10-01: new (restores the brute-force check of the 2026-09-29 review); no FFT, no histogram
    ny = dep.shape[1]
    k0, k1, a = node_pairs(t, teff)
    s = np.rint(-C_KMS * np.log(1.0 - v / C_KMS) / dv).astype(np.int64)
    s = np.clip(s, -nshift, nshift)
    w0 = m * (1.0 - a) * fcj[k0]
    w1 = m * a * fcj[k1]
    den = np.sum(w0 + w1)
    order = np.argsort(s, kind="stable")
    D, D0 = np.zeros(ny), np.zeros(ny)
    for c0 in range(0, order.size, chunk):
        ii = order[c0:c0 + chunk]
        rows = w0[ii, None] * dep[k0[ii]] + w1[ii, None] * dep[k1[ii]]
        D0 += rows.sum(axis=0)
        si = s[ii]
        starts = np.concatenate([[0], np.flatnonzero(np.diff(si)) + 1])
        G = np.add.reduceat(rows, starts, axis=0)
        for g, sh in zip(G, si[starts]):
            sh = int(sh)
            if sh >= ny or sh <= -ny:
                continue
            # observed y_i sees the rest profile at y_i + sh (blueshift for sh > 0)
            if sh >= 0:
                D[:ny - sh] += g[sh:]
            else:
                D[-sh:] += g[:ny + sh]
    return 1.0 - D / den, 1.0 - D0 / den


def _brute_init(spec):
    return spec


def _brute_task(task):
    st = par.worker_state()
    j, m, v, teff = task
    return _brute_line(st["dep"][j], st["t"], st["fc"][:, j], m, v, teff, st["dv"], st["nshift"], st["chunk"])


def brute_force(nodes, sample, mu, tn, pn, grid, chunk=512, nproc=1, start_method=None, timeout=900.0):
    """
    Disc-integrated profiles of one sample by a direct per-point sum with the rounding of
    :class:`ppmpy.synspec.disc.DiscFlux` (shifts rint(-c ln(1 - v/c) / dv) clipped to +-nshift; T_eff' interpolated
    linearly between nodes, clamped): every visible point's interpolated node profile (weights mu (1 - a) F_c,k0,
    mu a F_c,k1) is formed, shifted and added; nothing beyond the grid comes in (the zero padding of DiscFlux). No
    FFT and no (node, shift) histogram, so an independent evaluation of the same sum.

    Parameters
    ----------
    nodes: LibraryNodes or mapping
        t (nn,), prof (nn, nl, ny), fc (nn, nl) on ``grid``.
    sample: mapping
        teff, ur, uth, uph (N,).
    mu, tn, pn: array-like
        (nlos, N) projections.
    grid: VelocityGrid
    chunk: int
        Points formed at a time (memory: ~3 chunk x ny x 8 bytes per process).
    nproc: int
        Worker processes over the (line of sight, line) tasks ('fork' or 'spawn'; same bits).

    Returns
    -------
    F, F0: np.ndarray
        (nlos, nl, ny) float64.

    Notes
    -----
    Work: (visible points) x ny multiply-adds per line of sight and line (M424: 3.3e9, ~10 s), i.e. ~4 min per
    dump in one process.
    """
    # PP 2026-10-01: new
    if not isinstance(grid, VelocityGrid):
        raise TypeError("grid must be a VelocityGrid")
    smp = _sample_dict(sample)
    teff, ur, uth, uph = smp["teff"], smp["ur"], smp["uth"], smp["uph"]
    mu, tn, pn = _projections(mu, tn, pn, teff.size)
    prof = np.asarray(nodes["prof"], dtype=np.float64)
    nn, nl, ny = prof.shape
    if ny != grid.ny:
        raise ValueError("node profiles have {} grid points, the grid {}".format(ny, grid.ny))
    dep = np.ascontiguousarray(np.transpose(1.0 - prof, (1, 0, 2)))           # (nl, nn, ny)
    spec = dict(dep=dep, t=np.asarray(nodes["t"], dtype=np.float64), fc=np.asarray(nodes["fc"], dtype=np.float64),
                dv=grid.dv, nshift=grid.nshift, chunk=max(1, int(chunk)))
    nlos = mu.shape[0]
    tasks = []
    for k in range(nlos):
        vis = mu[k] > 0
        v = los_velocity(ur, uth, uph, mu[k], tn[k], pn[k])[vis]
        for j in range(nl):
            tasks.append((j, mu[k][vis], v, teff[vis]))
    F, F0 = np.zeros((nlos, nl, ny)), np.zeros((nlos, nl, ny))
    nproc = max(1, min(int(nproc), len(tasks)))
    if nproc <= 1:
        for i, task in enumerate(tasks):
            j, m, v, te = task
            F[i // nl, j], F0[i // nl, j] = _brute_line(dep[j], spec["t"], spec["fc"][:, j], m, v, te, spec["dv"],
                                                        spec["nshift"], spec["chunk"])
        return F, F0
    par.login_node_warning(nproc)
    with par.make_pool(nproc, initializer=_brute_init, initargs=(spec,), start_method=start_method) as pool:
        for i, res in par.imap_watchdog(pool, _brute_task_indexed, list(enumerate(tasks)), timeout=timeout):
            F[i // nl, tasks[i][0]], F0[i // nl, tasks[i][0]] = res
    return F, F0


def _brute_task_indexed(item):
    i, task = item
    return i, _brute_task(task)


def _stored_profiles(stored, d):
    """(F, F0) of dump d from a per-dump directory (dNNNN.npz), an (outdir, name) pair or a callable."""
    if callable(stored):
        return stored(d)
    if isinstance(stored, (tuple, list)):
        stored = os.path.join(os.fspath(stored[0]), stored[1])
    path = os.path.join(os.fspath(stored), "d{:04d}.npz".format(int(d)))
    with np.load(path) as z:
        return np.asarray(z["F"]), np.asarray(z["F0"])


def _reference_arrays(reference):
    """dumps, F, F0 of a reference set (path or mapping with dumps and Fl, Fl0 (r3_out.npz) or F, F0)."""
    z = np.load(os.fspath(reference)) if isinstance(reference, (str, os.PathLike)) else reference
    try:
        kf, kf0 = ("Fl", "Fl0") if "Fl" in z else ("F", "F0")
        return [int(d) for d in np.asarray(z["dumps"])], z[kf], z[kf0]
    finally:
        if isinstance(reference, (str, os.PathLike)):
            z.close()


def brute_force_check(nodes, samples, mu, tn, pn, grid=None, lref=None, dumps=None, integ=None, stored=None,
                      reference=None, chunk=512, nproc=1, start_method=None, timeout=900.0, tolerance=None,
                      f64_tolerance=None, tolerances=None, pattern=SAMPLE_PATTERN, log=None):
    """
    The full per-point sum done directly (:func:`brute_force`) compared with the integrator (float64), with stored
    per-dump products (float32) and, informationally, with an external reference set.

    Parameters
    ----------
    nodes: LibraryNodes or mapping
        The integrator's nodes.
    samples: str, os.PathLike, mapping, callable or a single sample
        Per-dump samples (module notes); with ``dumps`` None, one sample (mapping or .npz path) labelled dump -1.
    mu, tn, pn: array-like
        (nlos, N) projections ('matmul' for the stored M424 products).
    grid: VelocityGrid, optional
        Default ``integ.grid``, else required.
    lref: LineSet or array-like, optional
        For the line names only.
    dumps: sequence of int, optional
    integ: object, optional
        Integrator to compare with (default :class:`ppmpy.synspec.disc.DiscFlux` (nodes, grid)): a flux integrator
        built from ``nodes`` (checked: the same t, and fc if it has fc; ValueError otherwise).
    stored: str, (outdir, name) or callable, optional
        Per-dump products of the run (dNNNN.npz with F, F0; M424 disc_dumps_r4050_N1236544/flux), compared after
        nothing but their float32 rounding (default tolerance 2.5e-7; M424 2^-25 = 2.98e-8): the 2026-09-29 review
        check (<= 7.6e-6 then).
    reference: str or mapping, optional
        A further set with dumps and F, F0 (or Fl, Fl0: r3_out.npz), compared where the dumps overlap; tolerance
        None (informational; r3_out.npz differs by ~1e-3, module notes).
    chunk, nproc, start_method, timeout:
        :func:`brute_force`.
    tolerance, f64_tolerance, tolerances:
        brute (vs stored) and brute_f64 (vs integ) of :data:`DEFAULT_TOLERANCES` by default.

    Returns
    -------
    ValidationReport
        Checks brute_vs_integrator (F and F0, float64), brute_vs_stored (if stored), brute_vs_reference (info, if
        reference overlaps); arrays brute_dumps, brute_integ (nd, nl), brute_stored, brute_reference; data brute
        (dump -> (F, F0)).

    Raises
    ------
    ValueError
        No grid, an empty ``dumps``, or an integrator not built from ``nodes``.

    Validation
    ----------
    Synthetic: equals a naive per-point Python loop to ~1e-15 and DiscFlux to ~1e-15 (also for shifts beyond
    +-nshift and T_eff' beyond the nodes). M424 dumps 3200 and 4800: DiscFlux to ~1e-15, the stored flux products to
    float32 rounding (test_validate.py).
    """
    # PP 2026-10-01: new (the lost brute-force re-implementation check of the 2026-09-29 review)
    T0 = time.time()
    _log = _logger(log, T0)
    if grid is None:
        grid = _integ_grid(integ, None) if integ is not None else None
    if not isinstance(grid, VelocityGrid):
        raise ValueError("grid (a VelocityGrid) is required")
    if integ is None:
        integ = DiscFlux(nodes, grid=grid)
    else:
        _check_integ_nodes(integ, nodes, "brute force")
    if dumps is None:
        dl, getter = [-1], (lambda d: _sample_dict(samples))
    else:
        dl = _dumps_given(dumps, "brute force")

        def getter(d):
            return _get_sample(samples, d, pattern)
    ref = None
    if reference is not None:
        rd, rF, rF0 = _reference_arrays(reference)
        ref = {d: i for i, d in enumerate(rd)}
    d_int, d_sto, d_ref, data, ref_dumps = [], [], [], {}, []
    nl = None
    for d in dl:
        smp = getter(d)
        t1 = time.time()
        Fb, F0b = brute_force(nodes, smp, mu, tn, pn, grid, chunk=chunk, nproc=nproc, start_method=start_method,
                              timeout=timeout)
        t2 = time.time()
        Fi, F0i = _run(integ, smp, mu, tn, pn)
        nl = Fb.shape[1]
        d_int.append(np.maximum(_dmax(Fb, Fi), _dmax(F0b, F0i)))
        msg = "brute force dump {} ({:.0f} s): vs integrator max|dF| {}".format(d, t2 - t1, _fmt(d_int[-1]))
        if stored is not None:
            Fs, F0s = _stored_profiles(stored, d)
            d_sto.append(np.maximum(_dmax(Fb, np.asarray(Fs, dtype=np.float64)),
                                    _dmax(F0b, np.asarray(F0s, dtype=np.float64))))
            msg += "; vs stored {}".format(_fmt(d_sto[-1]))
        if ref is not None and d in ref:
            i = ref[d]
            d_ref.append(np.maximum(_dmax(Fb, np.asarray(rF[i], dtype=np.float64)),
                                    _dmax(F0b, np.asarray(rF0[i], dtype=np.float64))))
            ref_dumps.append(d)
            msg += "; vs reference {}".format(_fmt(d_ref[-1]))
        _log(msg)
        data[d] = (Fb, F0b)
    names = _line_names(integ, lref, nl)
    arrays = dict(brute_dumps=np.array(dl, dtype=np.int64), brute_integ=np.array(d_int))
    a = np.array(d_int)
    checks = [CheckResult("brute_vs_integrator", a.max(), _tol("brute_f64", f64_tolerance, tolerances),
                          details=dict(lines=names, per_line=a.max(axis=0), dumps=dl, per_dump=a))]
    if stored is not None:
        a = np.array(d_sto)
        arrays["brute_stored"] = a
        checks.append(CheckResult("brute_vs_stored", a.max(), _tol("brute", tolerance, tolerances),
                                  details=dict(lines=names, per_line=a.max(axis=0), dumps=dl, per_dump=a,
                                               note="stored profiles are float32 (rounding <= 6e-8)")))
    if d_ref:
        a = np.array(d_ref)
        arrays["brute_reference"] = a
        arrays["brute_reference_dumps"] = np.array(ref_dumps, dtype=np.int64)
        checks.append(CheckResult("brute_vs_reference", a.max(), None,
                                  details=dict(lines=names, per_line=a.max(axis=0), dumps=ref_dumps, per_dump=a)))
    meta = dict(check="brute", dumps=dl, chunk=int(chunk), nproc=int(nproc), wall=time.time() - T0,
                stored=None if stored is None or callable(stored) else str(stored),
                reference=reference if isinstance(reference, (str, os.PathLike)) else None)
    return ValidationReport(checks, meta=meta, arrays=arrays, data=dict(brute=data))


# ----------------------------------------------------------------------------------------------
# brute force of the intensity method
# ----------------------------------------------------------------------------------------------
def _imu_library_arrays(library):
    """The representatives' arrays of an intensity library (path: Il, Ic memory-mapped; NpzFile; mapping such as
    ImuLibrary): nodes u = unique(src), their T_eff' t, and per node s (nn, nl, K), nnode (nn, nl); Il, Ic as stored
    (nb, nl, K, ny), indexed by bin; grid: the VelocityGrid the library records (ImuLibrary, a file's '_meta'), or
    None."""
    from .disc import _imu_grid, _imu_open
    m, path, _ = _imu_open(library)
    src = np.asarray(m["src"])
    u = np.unique(src)
    t = np.asarray(m["teff_rep"], dtype=np.float64)[u]
    if t.size < 2 or not np.all(np.diff(t) > 0):
        raise ValueError("the intensity library needs at least 2 representatives with increasing T_eff'")
    return dict(u=u, t=t, S=np.asarray(m["s"], dtype=np.float64)[u], NN=np.asarray(m["nnode"])[u], Il=m["Il"],
                Ic=m["Ic"], path=path, grid=_imu_grid(library, m))


def _imu_clamped_pairs(t, teff):
    """(k0, k1, a) of T_eff' linear between the nodes t, clamped at the ends (written out here, not
    library.node_pairs, so that the brute force shares no interpolation code with the integrator)."""
    k0 = np.clip(np.searchsorted(t, teff, side="right") - 1, 0, t.size - 2)
    k1 = k0 + 1
    a = np.clip((teff - t[k0]) / (t[k1] - t[k0]), 0.0, 1.0)
    return k0, k1, a


def _brute_imu_line(L, j, m, v, k0, k1, a, dv, nshift, icentre, chunk):
    """
    F, F0 (ny,), vmean, vsig and the clipped count of one line and line of sight of the intensity method, point by
    point: each visible point's intensities I_l, I_c interpolated linearly in T_eff' (nodes k0, k1, weights 1 - a, a of
    any sign) and in s = sqrt(1 - mu^2) between its nodes' real rays, weighted by mu, shifted by its rounded Doppler
    shift with the edge values beyond the grid, and added; no velocity histogram, no FFT.
    """
    # PP 2026-10-02: new (the intensity method's brute force; its rules are those of disc.DiscImu, coded anew)
    S, NN, Il, Ic, u = L["S"], L["NN"], L["Il"], L["Ic"], L["u"]
    ny = int(np.shape(Il)[-1])
    n = m.size
    s = np.sqrt(np.clip(1.0 - m * m, 0.0, 1.0))
    raw = np.rint(-C_KMS * np.log(1.0 - v / C_KMS) / dv)
    n_clip = int((np.abs(raw) > nshift).sum())
    sh = np.clip(raw, -nshift, nshift).astype(np.int64)
    kk, tt = np.empty((2, n), np.int64), np.empty((2, n))
    for q, kn in enumerate((k0, k1)):
        for node in np.unique(kn):
            sel = kn == node
            nr = int(NN[node, j])
            sn = S[node, j, :nr]
            g = np.clip(np.searchsorted(sn, s[sel], side="right") - 1, 0, nr - 2)
            kk[q, sel] = g
            tt[q, sel] = np.clip((s[sel] - sn[g]) / (sn[g + 1] - sn[g]), 0.0, 1.0)
    w = np.stack([m * (1 - a) * (1 - tt[0]), m * (1 - a) * tt[0], m * a * (1 - tt[1]), m * a * tt[1]])
    b = np.stack([u[k0], u[k0], u[k1], u[k1]])
    r = np.stack([kk[0], kk[0] + 1, kk[1], kk[1] + 1])
    wc = (w * np.asarray(Ic[b, j, r, icentre], dtype=np.float64)).sum(axis=0)      # mu I_c(mu) at the line centre
    vm = np.sum(wc * v) / wc.sum()
    sd = np.sqrt(np.sum(wc * (v - vm) ** 2) / wc.sum())
    order = np.argsort(sh, kind="stable")
    yy = np.arange(ny)
    num, den, n0, d0 = np.zeros(ny), np.zeros(ny), np.zeros(ny), np.zeros(ny)
    for c0 in range(0, n, chunk):
        ii = order[c0:c0 + chunk]
        RL, RC = np.zeros((ii.size, ny)), np.zeros((ii.size, ny))
        for q in range(4):
            wq = w[q, ii, None]
            RL += wq * np.asarray(Il[b[q, ii], j, r[q, ii]], dtype=np.float64)
            RC += wq * np.asarray(Ic[b[q, ii], j, r[q, ii]], dtype=np.float64)
        n0 += RL.sum(axis=0)
        d0 += RC.sum(axis=0)
        si = sh[ii]
        starts = np.concatenate([[0], np.flatnonzero(np.diff(si)) + 1])
        GL, GC = np.add.reduceat(RL, starts, axis=0), np.add.reduceat(RC, starts, axis=0)
        for g, st in enumerate(starts):
            idx = np.clip(yy + si[st], 0, ny - 1)            # observed y sees the rest profile at y + shift
            num += GL[g][idx]
            den += GC[g][idx]
        del RL, RC, GL, GC
    return num / den, n0 / d0, vm, sd, n_clip


def _brute_imu_init(spec):
    lib = spec["library"]
    spec = dict(spec)
    spec["L"] = _imu_library_arrays(lib)
    return spec


def _brute_imu_task(item):
    i, (k, j, m, v, k0, k1, a) = item
    st = par.worker_state()
    return i, _brute_imu_line(st["L"], j, m, v, k0, k1, a, st["dv"], st["nshift"], st["icentre"], st["chunk"])


def brute_force_imu(library, sample, mu, tn, pn, grid, mask=None, pairs=None, lines=None, chunk=256, nproc=1,
                    start_method=None, timeout=900.0):
    """
    Disc-integrated profiles of the intensity method by a direct per-point sum with the rules of
    :class:`ppmpy.synspec.disc.DiscImu`: every visible point's line and continuum intensities, interpolated linearly in
    T_eff' between the representatives (nodes ``unique(src)`` at ``teff_rep``; clamped, or the given ``pairs``) and in
    s = sqrt(1 - mu^2) between the real rays of each node, weighted by mu, shifted by rint(-c ln(1 - v/c) / dv) grid
    steps (clipped to +-nshift) with the edge intensities beyond the grid, and summed: F = sum mu I_l / sum mu I_c;
    F0 the same without shifts; the velocity moments weighted by mu I_c(mu) at the line centre. No velocity histogram,
    no FFT and none of DiscImu's code (its ray search over all nodes at once, its row selection, the library FFTs), so
    an independent evaluation of the same sum.

    Parameters
    ----------
    library: str, os.PathLike or mapping
        The intensity library (as :class:`ppmpy.synspec.disc.DiscImu`; a path is memory-mapped, and is what 'spawn'
        workers should get: a mapping is pickled to them).
    sample: mapping or str
        teff, ur, uth, uph (N,) (teff unused with ``pairs``).
    mu, tn, pn: array-like
        (nlos, N) projections.
    grid: VelocityGrid, dict or None
        The library's grid (dv, nshift, the line centre); a dict is passed to VelocityGrid; None: the grid the
        library records, else the M424 grid (as DiscImu). A library that records its grid (ImuLibrary, a file
        written by ImuLibrary.save) refuses a grid with other points (ny, y[0], y[-1]; vshift is free), as
        DiscImu does; a legacy library is checked on ny only.
    mask: np.ndarray of bool, optional
        (N,) points to include (the others count as hidden), e.g. a random subset.
    pairs: tuple, optional
        (k0, k1, a) (N,) on the library's nodes (k0, k1 integers in [0, nn)), weights of any sign (e.g.
        validate.v3_tails' extrapolation).
    lines: sequence of int, optional
        Lines to compute (default all); the others stay NaN.
    chunk: int
        Points formed at a time (memory: ~6 chunk ny x 8 bytes per process).
    nproc, start_method, timeout:
        Worker processes over the (line of sight, line) tasks ('fork' or 'spawn'; the same bits as serial).

    Returns
    -------
    dict
        F, F0 (nlos, nl, ny) float64; vmean_w, sigma_w (nlos, nl); n_clip (nlos,) int64 (visible points with clipped
        shifts; per line of sight, as DiscImu).

    Notes
    -----
    Work: per (line of sight, line) 8 intensity rows of ny values gathered and added per visible point (M424: ~620 000
    visible points, ~2.7e10 multiply-adds; measured 37 s per line of sight and line in one process, so 15 min per
    dump of 8 lines of sight and 3 lines, 143 s with 8 'fork' workers; a 20000-point subset 16 s); use ``mask`` for
    subsets (:func:`brute_force_imu_check`, ``nsub``). Memory per process: ~6 chunk ny x 8 bytes (M424, chunk 256:
    66 MB), the per-point arrays of one task and the pages of the memory-mapped library it reads (peak RSS about 1 GB
    per worker); the calling process holds the per-point arrays of all tasks (nlos x nl tasks of ~7 x 8 bytes per
    visible point; M424 all points ~0.5 GB).

    Validation
    ----------
    Synthetic (toy libraries, M424 grid and others, 1-3 lines, nnode < K rows, empty bins, clipped shifts, T_eff'
    beyond the nodes, extrapolated pairs): a naive per-point Python loop to rounding (another summation order, <=
    6e-15) and DiscImu (every mode) to rounding; serial = fork = spawn bit for bit (tests/synspec/test_validate.py).
    M424: :func:`brute_force_imu_check`.
    """
    # PP 2026-10-02: new
    # PP 2026-10-02 (reviewer): dict grids, the recorded grid's points checked (another grid of the same ny gave wrong
    # shifts silently), integer node indices in pairs
    from .disc import _same_grid_points
    if isinstance(grid, dict):
        grid = VelocityGrid(**grid)
    L = _imu_library_arrays(library)
    if grid is None:
        grid = L["grid"] if L["grid"] is not None else VelocityGrid()
    if not isinstance(grid, VelocityGrid):
        raise TypeError("grid must be a VelocityGrid or a dict, got {}".format(type(grid).__name__))
    if L["grid"] is not None and not _same_grid_points(grid, L["grid"]):
        raise ValueError("grid {} (ny {}, y {:g}..{:g} km/s) differs from the intensity library's recorded grid {} "
                         "(ny {}, y {:g}..{:g} km/s)".format(grid, grid.ny, grid.y[0], grid.y[-1], L["grid"],
                                                             L["grid"].ny, L["grid"].y[0], L["grid"].y[-1]))
    nb, nl, K, ny = (int(x) for x in np.shape(L["Il"]))
    if ny != grid.ny:
        raise ValueError("the intensity library has {} grid points, the grid {}".format(ny, grid.ny))
    smp = _sample_dict(sample)
    teff, ur, uth, uph = smp["teff"], smp["ur"], smp["uth"], smp["uph"]
    mu, tn, pn = _projections(mu, tn, pn, teff.size)
    if pairs is None:
        k0, k1, a = _imu_clamped_pairs(L["t"], teff)
    else:
        k0, k1, a = (np.asarray(x) for x in pairs)
        a = a.astype(np.float64)
        if any(x.shape != teff.shape for x in (k0, k1, a)):
            raise ValueError("pairs must be (N,) arrays")
        if k0.dtype.kind not in "iu" or k1.dtype.kind not in "iu":
            raise ValueError("pairs: k0, k1 must be integer node indices, got {} and {}".format(k0.dtype, k1.dtype))
        if k0.size and (min(k0.min(), k1.min()) < 0 or max(k0.max(), k1.max()) >= L["t"].size):
            raise ValueError("pairs: node indices must be in [0, {})".format(L["t"].size))
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != teff.shape:
            raise ValueError("mask must have shape (N,) = ({},)".format(teff.size))
    jl = list(range(nl)) if lines is None else sorted({int(x) for x in lines})
    if not jl or jl[0] < 0 or jl[-1] >= nl:
        raise ValueError("lines must be indices in [0, {})".format(nl))
    nlos = mu.shape[0]
    tasks = []
    ncl = np.zeros(nlos, np.int64)
    for k in range(nlos):
        vis = mu[k] > 0
        if mask is not None:
            vis &= mask
        v = los_velocity(ur, uth, uph, mu[k], tn[k], pn[k])[vis]
        for j in jl:
            tasks.append((k, j, mu[k][vis], v, k0[vis], k1[vis], a[vis]))
    F, F0 = np.full((nlos, nl, ny), np.nan), np.full((nlos, nl, ny), np.nan)
    vm, sd = np.full((nlos, nl), np.nan), np.full((nlos, nl), np.nan)
    spec = dict(dv=grid.dv, nshift=grid.nshift, icentre=grid.icentre, chunk=max(1, int(chunk)))

    def put(i, res):
        k, j = tasks[i][:2]
        F[k, j], F0[k, j], vm[k, j], sd[k, j], ncl[k] = res

    nproc = max(1, min(int(nproc), len(tasks)))
    if nproc <= 1:
        for i, task in enumerate(tasks):
            put(i, _brute_imu_line(L, task[1], *task[2:], spec["dv"], spec["nshift"], spec["icentre"],
                                   spec["chunk"]))
        return dict(F=F, F0=F0, vmean_w=vm, sigma_w=sd, n_clip=ncl)
    smethod = par.get_context(start_method).get_start_method()
    spec["library"] = L["path"] if (L["path"] is not None and smethod != "fork") else library
    par.login_node_warning(nproc)
    with par.make_pool(nproc, initializer=_brute_imu_init, initargs=(spec,), start_method=smethod) as pool:
        for i, res in par.imap_watchdog(pool, _brute_imu_task, list(enumerate(tasks)), timeout=timeout):
            put(i, res)
    return dict(F=F, F0=F0, vmean_w=vm, sigma_w=sd, n_clip=ncl)


_STORED_KEYS = ("F", "F0", "vmean_w", "sigma_w", "n_clip")


def _stored_record(stored, d):
    """
    The stored products of dump d for :func:`brute_force_imu_check`: dict with F, F0 and, where the source has
    them, vmean_w, sigma_w, n_clip. ``stored``: a per-dump directory or an (outdir, name) pair (dNNNN.npz, which
    holds all five), or a callable dump -> (F, F0), (F, F0, vmean_w, sigma_w, n_clip) or a mapping with F, F0 and
    any of the others.
    """
    # PP 2026-10-02: new (reviewer: the stored velocity moments and clipped counts, products of record, were not
    # compared)
    if callable(stored):
        r = stored(d)
        if hasattr(r, "keys"):
            out = {k: np.asarray(r[k]) for k in _STORED_KEYS if k in r}
            if "F" not in out or "F0" not in out:
                raise ValueError("stored({}) returned a mapping without F and F0".format(d))
            return out
        r = tuple(r)
        if len(r) not in (2, 5):
            raise ValueError("stored({}) must return (F, F0), (F, F0, vmean_w, sigma_w, n_clip) or a mapping, got {} "
                             "items".format(d, len(r)))
        return {k: np.asarray(x) for k, x in zip(_STORED_KEYS, r)}
    if isinstance(stored, (tuple, list)):
        stored = os.path.join(os.fspath(stored[0]), stored[1])
    path = os.path.join(os.fspath(stored), "d{:04d}.npz".format(int(d)))
    with np.load(path) as z:
        return {k: np.asarray(z[k]) for k in _STORED_KEYS if k in z.files}


def brute_force_imu_check(integ, samples, mu, tn, pn, library=None, dumps=None, nsub=20000, seed=7, stored=None,
                          lref=None, chunk=256, nproc=1, start_method=None, timeout=900.0, tolerance=None,
                          f64_tolerance=None, tolerances=None, pattern=SAMPLE_PATTERN, log=None):
    """
    The intensity method's per-point sum done directly (:func:`brute_force_imu`) compared with an intensity
    integrator (:class:`ppmpy.synspec.disc.DiscImu`, float64) on a random subset of the points, or on all points and
    then also with stored per-dump products (F, F0 float32; vmean_w, sigma_w float64; n_clip): the profiles, the
    line-of-sight velocity moments and the clipped counts.

    Parameters
    ----------
    integ: DiscImu
        The integrator under test (its nodes must be the library's representatives: the same T_eff'). A per-line
        build (``built``, e.g. DiscImu(lines=(1,))) is compared on its built lines only; the brute force computes
        only those.
    samples: str, os.PathLike, mapping, callable or a single sample
        Per-dump samples (module notes); with ``dumps`` None, one sample (mapping or .npz path) labelled dump -1.
    mu, tn, pn: array-like
        (nlos, N) projections ('matmul' for the stored M424 products).
    library: str, os.PathLike or mapping, optional
        The intensity library; default the integrator's file (``integ.path``).
    dumps: sequence of int, optional
    nsub: int or None
        Points of the random subset (one ``np.random.default_rng(seed)`` draws the subsets of all dumps in order, from
        all N points; each line of sight uses the visible ones among them; the integrator gets the others hidden).
        None: all points (M424: ~2-3 min per line of sight and line in one process; use nproc), required for
        ``stored``.
    seed: int
    stored: str, (outdir, name) or callable, optional
        Per-dump products of the run (dNNNN.npz; M424 disc_dumps_r4050_N1236544/imu): F, F0 compared after their
        float32 rounding (default tolerance 'brute' 2.5e-7), and vmean_w, sigma_w, n_clip (the float64 moments to
        'brute_moments', the counts exactly); needs nsub None. A callable dump -> (F, F0), (F, F0, vmean_w,
        sigma_w, n_clip) or a mapping: the moments and counts are compared where it gives them (for every dump or
        for none; ValueError otherwise).
    lref: LineSet or array-like, optional
        For the line names only.
    chunk, nproc, start_method, timeout:
        :func:`brute_force_imu`.
    tolerance, f64_tolerance, tolerances:
        'brute' (vs stored) and 'brute_f64' (vs the integrator) of :data:`DEFAULT_TOLERANCES` by default; the
        moments 'brute_moments' (1e-9 km/s) and the clipped counts 'brute_n_clip' (0) through ``tolerances``.

    Returns
    -------
    ValidationReport
        Checks brute_imu_vs_integrator (F and F0, float64), brute_imu_moments (max |d vmean_w|, |d sigma_w| vs the
        integrator, km/s), brute_imu_n_clip (max |d n_clip| per line of sight vs the integrator, count), and with
        ``stored`` brute_imu_vs_stored, brute_imu_moments_vs_stored and brute_imu_n_clip_vs_stored (the latter two
        where the stored products have them). Details: lines, per_line (the compared lines), line_index (their
        indices), not_computed (names of lines the integrator did not build), dumps, per_dump; the profile check
        also keeps moments and n_clip_equal. Arrays brute_imu_dumps, brute_imu_integ, brute_imu_moments (nd, nl;
        NaN for lines not compared), brute_imu_n_clip (nd, nlos), and the *_stored counterparts; data brute_imu
        (dump -> dict of the brute force).

    Raises
    ------
    ValueError
        No library, an empty ``dumps``, nodes that are not the library's, stored with a subset, stored moments or
        counts given for some dumps only.

    Validation
    ----------
    Synthetic: passes on correct inputs (also for a per-line DiscImu) and fails for a flipped velocity sign,
    swapped T_eff' weights and lines of sight mixed up in stored products; an integrator whose profiles are right
    but whose velocity moments or clipped counts are wrong fails exactly brute_imu_moments / brute_imu_n_clip, and
    stored products with altered moments or counts fail the *_vs_stored checks; refuses the library of other nodes
    (tests/synspec/test_validate.py). M424 dump 4800 (2026-10-02): a 20000-point subset, DiscImu (default) to
    1.1e-14 / 1.5e-14 / 2.5e-14 per line, velocity moments to 5e-13 km/s; all points, DiscImu (lazy) to 7.0e-15 /
    8.3e-15 / 7.3e-15 and the stored imu products to 2^-25 = 2.98e-8 (their float32 rounding), the velocity moments
    to 8.7e-14 / 9.0e-14 / 1.1e-13 km/s (vs DiscImu and vs the stored float64 moments alike), the clipped counts
    equal (160 s with 8 'fork' workers, parent peak RSS 2.3 GB).
    """
    # PP 2026-10-02: new (the intensity counterpart of brute_force_check)
    # PP 2026-10-02 (reviewer): the velocity moments and clipped counts decide checks of their own (they were only
    # stored in details, and an n_clip mismatch turned the profile check's value into NaN); the stored moments and
    # counts are compared; a per-line integrator is compared on its built lines
    T0 = time.time()
    _log = _logger(log, T0)
    if library is None:
        library = getattr(integ, "path", None)
        if library is None:
            raise ValueError("library is required (the integrator has no library file)")
    grid = _integ_grid(integ, None)
    L = _imu_library_arrays(library)
    t = np.asarray(getattr(integ, "t", ()), dtype=np.float64)
    if t.shape != L["t"].shape or not np.array_equal(t, L["t"]):
        raise ValueError("brute force imu: the integrator's nodes ({} nodes) are not this library's representatives "
                         "({} nodes): pass the library the integrator was built from".format(t.size, L["t"].size))
    if stored is not None and nsub is not None:
        raise ValueError("stored products are compared on all points: pass nsub=None")
    nl = int(np.shape(L["Il"])[1])
    built = getattr(integ, "built", None)
    jl = None if built is None else sorted({int(j) for j in built})
    if jl is not None and (not jl or jl[0] < 0 or jl[-1] >= nl):
        raise ValueError("the integrator's built lines {} are not lines of the library ({} lines)".format(jl, nl))
    cols = list(range(nl)) if jl is None else jl
    if dumps is None:
        dl, getter = [-1], (lambda d: _sample_dict(samples))
    else:
        dl = _dumps_given(dumps, "brute force imu")

        def getter(d):
            return _get_sample(samples, d, pattern)
    mu, tn, pn = _projections(mu, tn, pn)
    nlos = mu.shape[0]
    rng = np.random.default_rng(seed)

    def per_line(A, B):
        """(nl,) max |A - B| over the lines of sight per line, NaN for the lines not compared."""
        A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
        if A.shape != B.shape or A.shape != (nlos, nl):
            raise ValueError("velocity moments of shape {} and {}, expected (nlos, nl) = ({}, {})".format(
                A.shape, B.shape, nlos, nl))
        out = np.full(nl, np.nan)
        out[cols] = np.abs(A[:, cols] - B[:, cols]).max(axis=0)
        return out

    def prof(Fa, F0a, Fb, F0b):
        out = np.full(nl, np.nan)
        Fb, F0b = np.asarray(Fb, dtype=np.float64), np.asarray(F0b, dtype=np.float64)
        out[cols] = np.maximum(_dmax(Fa[:, cols], Fb[:, cols]), _dmax(F0a[:, cols], F0b[:, cols]))
        return out

    def counts(a, b):
        b = np.asarray(b)
        if b.shape != (nlos,):
            raise ValueError("clipped counts of shape {}, expected ({},)".format(b.shape, nlos))
        return np.abs(np.asarray(a, dtype=np.int64) - b.astype(np.int64))

    d_int, d_mom, d_ncl, d_sto, d_msto, d_nsto, data = [], [], [], [], [], [], {}
    for d in dl:
        smp = getter(d)
        N = smp["teff"].size
        mask = None
        if nsub is not None:
            mask = np.zeros(N, bool)
            mask[rng.choice(N, min(int(nsub), N), replace=False)] = True
        t1 = time.time()
        b = brute_force_imu(library, smp, mu, tn, pn, grid, mask=mask, lines=jl, chunk=chunk, nproc=nproc,
                            start_method=start_method, timeout=timeout)
        t2 = time.time()
        Fi, F0i, vmi, sdi, nci = _run_imu_full(integ, smp, mu, tn, pn, mask)
        d_int.append(prof(b["F"], b["F0"], Fi, F0i))
        d_mom.append(np.maximum(per_line(b["vmean_w"], vmi), per_line(b["sigma_w"], sdi)))
        d_ncl.append(counts(b["n_clip"], nci))
        msg = ("brute force imu dump {} ({} points, {:.0f} s): vs integrator max|dF| {}; velocity moments {} km/s; "
               "clipped counts differ by {}".format(d, "all" if mask is None else int(mask.sum()), t2 - t1,
                                                    _fmt(d_int[-1][cols]), _fmt(d_mom[-1][cols]),
                                                    int(d_ncl[-1].max())))
        if stored is not None:
            st = _stored_record(stored, d)
            d_sto.append(prof(b["F"], b["F0"], st["F"], st["F0"]))
            msg += "; vs stored {}".format(_fmt(d_sto[-1][cols]))
            if "vmean_w" in st and "sigma_w" in st:
                d_msto.append(np.maximum(per_line(b["vmean_w"], st["vmean_w"]),
                                         per_line(b["sigma_w"], st["sigma_w"])))
                msg += ", moments {} km/s".format(_fmt(d_msto[-1][cols]))
            if "n_clip" in st:
                d_nsto.append(counts(b["n_clip"], st["n_clip"]))
                msg += ", clipped counts differ by {}".format(int(d_nsto[-1].max()))
        _log(msg)
        data[d] = b
    for nm, got in (("velocity moments", d_msto), ("clipped counts", d_nsto)):
        if stored is not None and 0 < len(got) < len(dl):
            raise ValueError("stored gives the {} for {} of {} dumps: give them for every dump or for none".format(
                nm, len(got), len(dl)))
    names = _line_names(integ, lref, nl)
    lines = [names[j] for j in cols]
    skip = [names[j] for j in range(nl) if j not in cols]

    def det(A, **kw):
        A = np.asarray(A, dtype=np.float64)
        out = dict(lines=lines, per_line=A[:, cols].max(axis=0), line_index=list(cols), dumps=dl, per_dump=A)
        if skip:
            out["not_computed"] = skip
        out.update(kw)
        return out

    def vmax(A):
        return float(np.asarray(A, dtype=np.float64)[:, cols].max())

    ncl_ok = all(int(x.max()) == 0 for x in d_ncl)
    a, am, an = np.array(d_int), np.array(d_mom), np.array(d_ncl)
    arrays = dict(brute_imu_dumps=np.array(dl, dtype=np.int64), brute_imu_integ=a, brute_imu_moments=am,
                  brute_imu_n_clip=an)
    nsub_rec = None if nsub is None else int(nsub)
    checks = [CheckResult("brute_imu_vs_integrator", vmax(a), _tol("brute_f64", f64_tolerance, tolerances),
                          details=det(a, moments=am, n_clip_equal=ncl_ok, nsub=nsub_rec)),
              CheckResult("brute_imu_moments", vmax(am), _tol("brute_moments", None, tolerances), unit="km/s",
                          details=det(am, nsub=nsub_rec, note="max |d vmean_w|, |d sigma_w| vs the integrator")),
              CheckResult("brute_imu_n_clip", float(an.max()), _tol("brute_n_clip", None, tolerances), unit="count",
                          details=dict(dumps=dl, per_dump=an, nsub=nsub_rec,
                                       note="|d n_clip| per line of sight vs the integrator"))]
    if stored is not None:
        a = np.array(d_sto)
        arrays["brute_imu_stored"] = a
        checks.append(CheckResult("brute_imu_vs_stored", vmax(a), _tol("brute", tolerance, tolerances),
                                  details=det(a, note="stored profiles are float32 (rounding <= 6e-8)")))
        if d_msto:
            am = np.array(d_msto)
            arrays["brute_imu_moments_stored"] = am
            checks.append(CheckResult("brute_imu_moments_vs_stored", vmax(am), _tol("brute_moments", None, tolerances),
                                      unit="km/s", details=det(am, note="max |d vmean_w|, |d sigma_w| vs the stored "
                                                                        "float64 moments")))
        if d_nsto:
            an = np.array(d_nsto)
            arrays["brute_imu_n_clip_stored"] = an
            checks.append(CheckResult("brute_imu_n_clip_vs_stored", float(an.max()),
                                      _tol("brute_n_clip", None, tolerances), unit="count",
                                      details=dict(dumps=dl, per_dump=an, note="|d n_clip| per line of sight vs the "
                                                                               "stored counts")))
    meta = dict(check="brute_imu", dumps=dl, nsub=nsub_rec, seed=int(seed), chunk=int(chunk), nproc=int(nproc),
                lines=list(cols), wall=time.time() - T0,
                stored=None if stored is None or callable(stored) else str(stored))
    return ValidationReport(checks, meta=meta, arrays=arrays, data=dict(brute_imu=data))


def _run_imu_full(integ, smp, mu, tn, pn, mask):
    """F, F0 (nlos, nl, ny), vmean, vsig (nlos, nl), n_clip (nlos,) of the integrator with the points outside mask
    hidden (as :func:`_run`, keeping the velocity moments and the clipped counts)."""
    teff, ur, uth, uph = smp["teff"], smp["ur"], smp["uth"], smp["uph"]
    k0, k1, a = integ.pairs(teff)
    out = None
    for k in range(mu.shape[0]):
        m = mu[k] if mask is None else np.where(mask, mu[k], 0.0)
        r = integ(m, los_velocity(ur, uth, uph, mu[k], tn[k], pn[k]), k0, k1, a)
        if out is None:
            nl, ny = np.shape(r[0])
            out = [np.zeros((mu.shape[0], nl, ny)), np.zeros((mu.shape[0], nl, ny)), np.zeros((mu.shape[0], nl)),
                   np.zeros((mu.shape[0], nl)), np.zeros(mu.shape[0], np.int64)]
        for x, y in zip(out, r):
            x[k] = y
    return out


# ----------------------------------------------------------------------------------------------
# EW conservation
# ----------------------------------------------------------------------------------------------
def ew_conservation(F, F0=None, y=None, lref=None, ew_jacobian=False, block=64, tolerance=None, tolerances=None,
                    name="ew_conservation"):
    """
    Doppler shifts by whole grid steps conserve the equivalent width on the y grid: EW(F) = EW(F0) up to grid-end
    effects and rounding (the profiles must be back at the continuum within the shifts of the grid ends).

    Parameters
    ----------
    F, F0: array-like, str or mapping
        Profiles with and without Doppler shifts, (..., nl, ny) (e.g. a per-dump product (nlos, nl, ny) or a time
        series (dumps, nlos, nl, ny); memory maps are read ``block`` rows of the first axis at a time). F may be the
        path of such a product or a mapping (with F0 None): its F, F0, and Y, LREF unless given.
    y: array-like
        The velocity grid [km/s].
    lref: LineSet or array-like
        (nl,) velocity zero points [A].
    ew_jacobian: bool
        False (default): EW = lref / c int (1 - F) dy, exactly conserved by whole-step shifts. True: with the factor
        lambda/lref, which a Doppler shift changes by exp(-s/c) (the profile is compressed in wavelength by
        (1 - v/c)): M424 up to 1.1e-5 relative, a physical O(v/c) effect, 3-8 % of the EW variability.
    block: int
        Rows of the first axis per read.
    tolerance, tolerances:
        Default ``tolerances['ew']``, else :data:`DEFAULT_TOLERANCES` (1e-6: the M424 float32 products give 1.6e-7).

    Returns
    -------
    CheckResult
        value = max |EW(F) - EW(F0)| / |EW(F0)|; details: per_line (relative), max_abs [A] and mean_rel per line,
        argmax_flat (per line: flat index over the leading axes of the maximum, or of the first non-finite profile),
        nonfinite (per line: profiles whose relative difference is not finite, e.g. a NaN in F or F0, or EW(F0) =
        0). Any non-finite profile makes its line's value NaN, so the check fails (whatever the block order).
    """
    # PP 2026-10-01: new; NaN handling made sticky after the review (a NaN of an earlier block was overwritten)
    src = None
    if F0 is None:
        src = F
        if isinstance(src, (str, os.PathLike)):
            from .lpv import open_timeseries
            src = open_timeseries(os.fspath(src))
        F, F0 = src["F"], src["F0"]
        if y is None and "Y" in src:
            y = src["Y"]
        if lref is None and "LREF" in src:
            lref = src["LREF"]
    if y is None or lref is None:
        raise ValueError("y and lref are required")
    y = np.asarray(y, dtype=np.float64)
    lr = np.atleast_1d(np.asarray(getattr(lref, "lref", lref), dtype=np.float64))
    shape = np.shape(F)
    if np.shape(F0) != shape or len(shape) < 2 or shape[-1] != y.size or shape[-2] != lr.size or lr.size == 0:
        raise ValueError("F and F0 must have one shape (..., nl, ny) = (..., {}, {}), got {} and {}".format(
            lr.size, y.size, shape, np.shape(F0)))
    kw = dict(keys=("ew",), ew_jacobian=ew_jacobian)
    nl = lr.size
    rel_max, abs_max, rel_sum = np.zeros(nl), np.zeros(nl), np.zeros(nl)
    nonfinite = np.zeros(nl, dtype=np.int64)
    count = 0
    where = [None] * nl                                  # flat index over the leading axes of the largest deviation
    lead = shape[:-2]
    n0 = lead[0] if lead else 1
    per_row = int(np.prod(lead[1:])) if lead else 1      # profiles per index of the first axis (and line)
    block = max(1, int(block))
    for i0 in range(0, n0, block):
        if lead:
            A, B = np.asarray(F[i0:i0 + block]), np.asarray(F0[i0:i0 + block])
        else:
            A, B = np.asarray(F)[None], np.asarray(F0)[None]
        A, B = A.astype(np.float64), B.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            e = line_diagnostics(A, y, lr, **kw)["ew"]
            e0 = line_diagnostics(B, y, lr, **kw)["ew"]
            ab = np.abs(e - e0).reshape(-1, nl)
            r2 = ab / np.abs(e0).reshape(-1, nl)
        bad = ~np.isfinite(r2)
        nonfinite += bad.sum(axis=0)
        for j in range(nl):
            if np.isnan(rel_max[j]):                     # sticky: a non-finite profile seen before
                continue
            if bad[:, j].any():
                rel_max[j], where[j] = np.nan, i0 * per_row + int(np.argmax(bad[:, j]))
                continue
            k = int(np.argmax(r2[:, j]))
            if r2[k, j] > rel_max[j]:
                rel_max[j], where[j] = r2[k, j], i0 * per_row + k
        abs_max = np.maximum(abs_max, ab.max(axis=0))   # NaN propagates
        rel_sum += r2.sum(axis=0)
        count += r2.shape[0]
    names = [str(n) for n in getattr(lref, "names", ["line{}".format(j) for j in range(nl)])]
    det = dict(lines=names, per_line=rel_max, max_abs=abs_max, mean_rel=rel_sum / max(count, 1),
               argmax_flat=where, nonfinite=nonfinite, ew_jacobian=bool(ew_jacobian), shape=list(shape))
    if nonfinite.any():
        det["note"] = "non-finite relative EW differences (NaN profiles or EW(F0) = 0) in {} profiles".format(
            int(nonfinite.sum()))
    return CheckResult(name, float(np.max(rel_max)), _tol("ew", tolerance, tolerances), details=det, unit="relative")


# ----------------------------------------------------------------------------------------------
# LPV comparison and the driver
# ----------------------------------------------------------------------------------------------
def lpv_residual_rms(timeseries, y=None, vwin=600.0):
    """
    The run's line-profile variability: per line, rms and max of R = F - <F>_t within |y| <= vwin (pooled over
    lines of sight, dumps and pixels; as fig_disc_dumps_validation.py), and the rms of the EW time series (absolute
    and relative to the mean EW) when the series has diag_F / diag_keys.

    Parameters
    ----------
    timeseries: str, os.PathLike or mapping
        A time series (:func:`ppmpy.synspec.dumps.collect_timeseries`; F (dumps, nlos, nl, ny), Y; diag_F, diag_keys
        optional). A path is opened with memory maps (:func:`ppmpy.synspec.lpv.open_timeseries`).
    y: array-like, optional
        Default the series' Y.
    vwin: float
        Window [km/s] (600, as the legacy figure and systematics).

    Returns
    -------
    dict
        rms, max (nl,); ew_rms, ew_rel_rms (nl,) or None; vwin.
    """
    # PP 2026-10-01: ported from fig_disc_dumps_validation.py:91-97 (res = F[..., |Y| <= 600] - mean_t; per-line std
    # and max), one line at a time (memory); the EW part is new
    ts = timeseries
    if isinstance(ts, (str, os.PathLike)):
        from .lpv import open_timeseries
        ts = open_timeseries(os.fspath(ts))
    Y = np.asarray(ts["Y"] if y is None else y, dtype=np.float64)
    m = np.abs(Y) <= vwin
    Fts = ts["F"]
    nl = np.shape(Fts)[2]
    rms, mx = np.zeros(nl), np.zeros(nl)
    for j in range(nl):
        X = np.asarray(Fts[:, :, j])[..., m].astype(np.float64)
        X -= X.mean(axis=0, keepdims=True)
        rms[j], mx[j] = X.std(), np.abs(X).max()
        del X
    ew_rms = ew_rel = None
    if "diag_F" in ts and "diag_keys" in ts:
        keys = [str(k) for k in np.asarray(ts["diag_keys"])]
        if "ew" in keys:
            e = np.asarray(ts["diag_F"])[..., keys.index("ew")]
            ew_rms = (e - e.mean(axis=0)).std(axis=(0, 1))
            ew_rel = ew_rms / np.abs(e.mean(axis=(0, 1)))
    return dict(rms=rms, max=mx, ew_rms=ew_rms, ew_rel_rms=ew_rel, vwin=float(vwin))


def compare_lpv(report, lpv, warn=LPV_WARN):
    """
    Put every check's per-line values in relation to the run's LPV (:func:`lpv_residual_rms`): details 'lpv_ratio'
    (per line: value / residual rms for 'continuum' checks, / EW rms for 'A', / relative EW rms for 'relative'; per
    line of 'line_index' where a check gives that; 'km/s' and 'count' checks are not compared) and
    'lpv_warn' (any ratio > warn, or any ratio not finite, e.g. a NaN per-line value; a :class:`UserWarning` for
    checks with a tolerance, informational ones are only flagged).

    Returns
    -------
    ValidationReport
        The same report (modified in place).
    """
    # PP 2026-10-01: new (task: compare each value with the run's own LPV residual rms, warn above 5 %)
    scale = {"continuum": lpv.get("rms"), "A": lpv.get("ew_rms"), "relative": lpv.get("ew_rel_rms")}
    for c in report.checks:
        s = scale.get(c.unit)
        pl = c.details.get("per_line")
        if s is None or pl is None or np.size(pl) == 0:
            continue
        pl, s = np.asarray(pl, dtype=np.float64), np.asarray(s, dtype=np.float64)
        li = c.details.get("line_index")
        if pl.shape != s.shape and li is not None and s.ndim == 1 and pl.shape == (len(li),):
            # PP 2026-10-02: per-line values of some lines only (brute_force_imu_check of a per-line integrator)
            li = np.asarray(li, dtype=np.int64)
            if li.size and li.min() >= 0 and li.max() < s.size:
                s = s[li]
        if pl.shape != s.shape:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = pl / s
        fin = np.isfinite(ratio)
        c.details["lpv_ratio"] = ratio
        c.details["lpv_scale"] = s
        # PP 2026-10-01: reviewer: all-NaN ratios gave lpv_warn False (nanmax NaN > warn); non-finite ones now flag
        c.details["lpv_warn"] = bool((not fin.all()) or ratio.max() > warn)
        if not fin.all():
            c.details["lpv_nonfinite"] = int((~fin).sum())
        if c.details["lpv_warn"] and c.tolerance is not None:          # informational checks are flagged only
            warnings.warn("{}: deviation up to {} of the LPV {} (> {:.0%}{}): {}".format(
                c.name, _fmt_max(ratio), {"continuum": "residual rms", "A": "EW rms",
                                          "relative": "relative EW rms"}[c.unit], warn,
                "" if fin.all() else ", or not finite", " / ".join("{:.1%}".format(r) for r in ratio)),
                UserWarning, stacklevel=2)
    report.meta["lpv"] = _jsonable(lpv)
    report.meta["lpv_warn_fraction"] = float(warn)
    return report


def run_validation(integ=None, mu=None, tn=None, pn=None, sample=None, exact=None, library=None, nodes=None,
                   samples=None, dumps=(), v3_dumps=(), label="flux", lref=None, grid=None, nsub=20000, v6_seed=7,
                   v6_chunk=2000, nmins=(1, 5, 100), margin=300.0, corr=None, corr_key="corr", holdout=None,
                   brute_dumps=(), brute_kw=None, timeseries=None, vwin=600.0, lpv_warn=LPV_WARN, tolerances=None,
                   pattern=SAMPLE_PATTERN, imu_library=None, log=None):
    """
    Run the checks the given inputs allow, for one run (no defaults of a particular star: every path and array is
    given), and relate them to the run's LPV.

    Parameters
    ----------
    integ: object
        The run's integrator (DiscFlux duck type).
    mu, tn, pn: array-like
        (nlos, N) projections of the run's points onto its lines of sight.
    sample, exact:
        V1 (:func:`v1_exact`): the library dump's sample and the exact per-point sums.
    library: FluxLibrary, str or os.PathLike, optional
        The flux library (V4, V5).
    nodes: LibraryNodes or mapping, optional
        The integrator's nodes (V6, brute force); checked against the integrator (the same t, and fc if it has fc;
        ValueError otherwise).
    samples: str, os.PathLike, mapping or callable
        Per-dump samples (V3-V6, brute force).
    dumps: sequence of int
        V4, V6 dumps; V5 uses the first.
    v3_dumps: sequence of int
        V3 dumps (:func:`v3_select`).
    label: str
        Run label of V1.
    lref, grid:
        Defaults: the integrator's.
    nsub, v6_seed, v6_chunk, nmins, margin:
        Options of V6, V5, V3.
    corr, corr_key:
        The per-bin profile correction of the integrator's nodes (V4, V5; :func:`v4_nearest`): required for a
        corrected run (flux_lamfix).
    holdout: dict, optional
        Keyword arguments of :func:`v2_holdout` (profiles, sample, theta, phi, los, grid, lref, factory, ...): runs
        V2.
    brute_dumps: sequence of int
        Dumps of :func:`brute_force_check` (with ``nodes``; flux integrators) or, for an intensity integrator
        (method 'imu', e.g. :class:`ppmpy.synspec.disc.DiscImu`), of :func:`brute_force_imu_check` (the library
        ``imu_library`` or the integrator's file; default a 20000-point subset per dump); ``brute_kw`` further
        arguments of the one that runs (stored, reference, nproc, ...; for the intensity method nsub, seed, stored
        with nsub=None, nproc, ...).
    timeseries: str, os.PathLike or mapping, optional
        The run's time series: EW conservation of its F / F0, and the LPV comparison (:func:`compare_lpv`).
    vwin, lpv_warn:
        LPV window [km/s] and warning fraction.
    tolerances: dict, optional
        Per-check tolerances (keys of :data:`DEFAULT_TOLERANCES`; missing keys take the defaults).
    imu_library: str, os.PathLike or mapping, optional
        The intensity library of an intensity integrator, for its brute force (default the integrator's file,
        ``integ.path``).
    log: callable, optional

    Returns
    -------
    ValidationReport
        All checks run, with ``meta['ran']`` (the checks) and ``meta['skipped']`` (name -> missing inputs or the
        reason). When no check ran, a :class:`UserWarning` and an empty report (``passed()`` False).

    Notes
    -----
    * V4, V5, V6 and the brute force apply to flux integrators DiscFlux(lib_nodes(library, nmin, smooth, corr))
      only (module notes: applicability): for an integrator without ``fc`` (e.g. the intensity method) they are
      skipped ('flux integrator' in ``meta['skipped']``); V1 and V2 (through ``holdout['factory']``) apply to any,
      V3 to any whose call takes pairs with weights of any sign (DiscFlux, DiscImu). With smoothed nodes V4 is
      informational; a corrected run needs ``corr``.
    * Intensity integrators (``method == 'imu'``: :class:`ppmpy.synspec.disc.DiscImu`; M424: V1 imu reproduces
      validate.npz bit for bit) get, instead of the flux brute force, :func:`brute_force_imu_check` ('brute_imu',
      with ``brute_dumps`` and the library); the frozen legacy fw_disc.DiscImu has no ``method`` and gets V1 (and
      V3) only.
    * V1 dEW and V2 dEW use different EW definitions by default, as the legacy products: V1 with the factor
      lambda/lref (``ew_jacobian=True``, validate.npz), V2 without it (``ew_jacobian=False``, holdout.npz was
      written before fw_disc.diagnostics got the factor); pass ``holdout['ew_jacobian']=True`` for one definition.
      The difference is O(v/c) of the EW (M424 <= 1e-5 relative), far below the dEW values.

    Raises
    ------
    ValueError
        ``nodes`` that are not those of a flux integrator.
    """
    # PP 2026-10-01: new driver for fw_disc_dumps_validate.py (V1, V3-V6), fw_disc_holdout.py (V2), the brute force
    # and EW checks, and the LPV comparison of fig_disc_dumps_validation.py
    T0 = time.time()
    parts, skipped = [], {}
    tol = dict(tolerances or {})
    dl = [int(d) for d in dumps]
    have_proj = mu is not None and tn is not None and pn is not None
    flux = None if integ is None else (True if hasattr(integ, "fc") and hasattr(integ, "t") else None)
    if flux and nodes is not None:                     # other integrators do not use the nodes (V6, brute skipped)
        _check_integ_nodes(integ, nodes, "run_validation")

    def need(name, **inputs):
        miss = [k for k, v in inputs.items() if v is None or (isinstance(v, (list, tuple)) and not v)]
        if miss:
            skipped[name] = miss
        return not miss

    proj = mu if have_proj else None
    if need("V1", integ=integ, projections=proj, sample=sample, exact=exact):
        parts.append(v1_exact(integ, exact, sample, mu, tn, pn, label=label, lref=lref, grid=grid, tolerances=tol))
    if need("V2", holdout=holdout):
        kw = dict(holdout)
        kw.setdefault("tolerances", tol)
        kw.setdefault("log", log)
        parts.append(v2_holdout(**kw))
    if need("V3", integ=integ, projections=proj, samples=samples, v3_dumps=list(v3_dumps)):
        parts.append(v3_tails(integ, samples, v3_dumps, mu, tn, pn, margin=margin, tolerances=tol, lref=lref,
                              pattern=pattern, log=log))
    fl = dict(flux_integrator=flux) if integ is not None else {}
    if need("V4", integ=integ, **fl, projections=proj, samples=samples, library=library, dumps=dl):
        parts.append(v4_nearest(integ, library, samples, dl, mu, tn, pn, grid=grid, corr=corr, corr_key=corr_key,
                                tolerances=tol, lref=lref, pattern=pattern, log=log))
    if need("V5", integ=integ, **fl, projections=proj, samples=samples, library=library, dumps=dl,
            nmins=list(nmins)):
        parts.append(v5_node_merging(integ, library, samples, dl[0], mu, tn, pn, nmins=nmins, grid=grid, corr=corr,
                                     corr_key=corr_key, tolerances=tol, lref=lref, pattern=pattern, log=log))
    if need("V6", integ=integ, **fl, projections=proj, samples=samples, nodes=nodes, dumps=dl):
        parts.append(v6_rounding(integ, nodes, samples, dl, mu, tn, pn, nsub=nsub, seed=v6_seed, chunk=v6_chunk,
                                 grid=grid, tolerances=tol, lref=lref, pattern=pattern, log=log))
    if need("brute", **fl, nodes=nodes, projections=proj, samples=samples, brute_dumps=list(brute_dumps)):
        kw = dict(brute_kw or {})
        kw.setdefault("tolerances", tol)
        kw.setdefault("log", log)
        g = grid if grid is not None else (getattr(integ, "grid", None) if integ is not None else None)
        parts.append(brute_force_check(nodes, samples, mu, tn, pn, grid=g, lref=lref, dumps=brute_dumps, integ=integ,
                                       pattern=pattern, **kw))
    # PP 2026-10-02: the brute force of the intensity method (only for intensity integrators: the skipped list of
    # flux runs is unchanged)
    if integ is not None and not flux and getattr(integ, "method", None) == "imu":
        ilib = imu_library if imu_library is not None else getattr(integ, "path", None)
        if need("brute_imu", projections=proj, samples=samples, brute_dumps=list(brute_dumps), library=ilib):
            kw = dict(brute_kw or {})
            kw.setdefault("tolerances", tol)
            kw.setdefault("log", log)
            parts.append(brute_force_imu_check(integ, samples, mu, tn, pn, library=ilib, dumps=brute_dumps, lref=lref,
                                               pattern=pattern, **kw))
    lpv = None
    if need("ew_conservation", timeseries=timeseries):
        ts = timeseries
        if isinstance(ts, (str, os.PathLike)):
            from .lpv import open_timeseries
            ts = open_timeseries(os.fspath(ts))
        y = ts["Y"] if "Y" in ts else (grid.y if grid is not None else None)
        lr = ts["LREF"] if "LREF" in ts else (lref if lref is not None else _integ_lref(integ, None)[1])
        chk = ew_conservation(ts["F"], ts["F0"], y, lr, tolerances=tol)
        chk.details["lines"] = _line_names(integ, lref, np.size(lr))
        parts.append(ValidationReport([chk], meta=dict(check="EW")))
        lpv = lpv_residual_rms(ts, vwin=vwin)
    meta = dict(label=label, ran=None, skipped=skipped, tolerances=dict(DEFAULT_TOLERANCES, **tol),
                created=datetime.datetime.now().isoformat(timespec="seconds"))
    report = ValidationReport.merge(*parts, meta=meta)
    report.meta["ran"] = report.names()
    if not report.checks:
        # PP 2026-10-01: reviewer: an empty report passed
        warnings.warn("run_validation: no check ran (missing inputs: {})".format(
            "; ".join("{}: {}".format(k, ", ".join(v)) for k, v in skipped.items())), UserWarning, stacklevel=2)
    if lpv is not None:
        compare_lpv(report, lpv, warn=lpv_warn)
    report.meta["wall"] = time.time() - T0
    return report
