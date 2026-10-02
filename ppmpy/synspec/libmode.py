"""
Library mode: one FASTWIND model per T_eff' node instead of one per sphere point.

The per-point models of a run differ only in T_eff' (every other INDAT input is fixed; checked on the archived M424
models by :func:`ppmpy.synspec.fwresults.check_indat_premise`), and the disc integration uses them only through a
T_eff' library (:mod:`ppmpy.synspec.library`): node profiles interpolated linearly in T_eff'. So the library can be
computed directly, with one FASTWIND model (or a few replicas) per T_eff' node: e.g. every 10 K over the T_eff'
range of all dumps, a few hundred models instead of one per sphere point (M424: 1 236 544). The runner, the
libraries and the integrators are the same:

1. :func:`plan_teff_nodes`: nodes at ``offset + k dT`` covering the T_eff' range of per-dump samples (e.g.
   :func:`ppmpy.synspec.validate.teff_ranges` of all dumps) plus a margin, ``replicas`` models per node ->
   :class:`NodePlan`. :meth:`NodePlan.write` writes the 'idx teff' table of a per-point run (points.txt format,
   nothing else), so the FASTWIND runner runs it unchanged
   (``python3 -m ppmpy.synspec.fastwind run RUN_DIR --template T --formal F``; with ``--formal-build v10.6_HHe_imu``
   pformalsol also writes the OUT_IMU files of the intensity library); :meth:`NodePlan.write_points_npz` writes the
   points.npz that :func:`ppmpy.synspec.fwresults.merge_task` / ``merge_tasks`` need (idx, teff; no coordinates:
   pass ``copy_keys=()``). Run a plan in its own run directory: the runner skips every idx its results tree already
   holds, whatever the T_eff'. ``plan_teff_nodes(avoid=RUN_DIR)`` starts the indices beyond those of an existing run
   (:func:`existing_indices`), :meth:`NodePlan.check_collisions` checks a plan against one.
2. ``fwresults.merge_tasks`` + ``fwresults.combine`` -> profiles.npz of the node models (missing.txt lists failed
   models with T_eff + 1 K, as for a per-point run). A retry nudge can move a replica across its bin edge (room
   dT / (2R) for replicas spread over the bin; :func:`plan_teff_nodes` warns at <= 1 K): the flux library keeps it in
   its planned node (``nudged='keep'``), the intensity library does not use it as the node's representative.
3. :func:`library_from_models` -> :class:`NodeLibrary`: the flux library with one bin per planned node (the replica
   mean; :func:`flux_library_from_models`, given the plan), its interpolation nodes (``lib_nodes(nmin=1)``), and,
   when the models have OUT_IMU files, the intensity library of the same models (:func:`imu_library_from_models`:
   the flux library's models in its bins, one representative per node, replicas not averaged).
   :meth:`NodeLibrary.save` writes the library files.
4. The dumps as for a per-point run: :func:`ppmpy.synspec.dumps.run_disc_dumps` with
   :func:`ppmpy.synspec.dumps.flux_integrator` (the saved library, ``nmin=1``) or
   :func:`ppmpy.synspec.dumps.imu_integrator` (the saved intensity library).

nmin
----
:func:`ppmpy.synspec.library.lib_nodes` merges consecutive filled bins until a node holds >= nmin models. A
per-point library uses nmin = 20 to merge its sparse tails (M424: 303 filled bins -> 245 nodes). In library mode
every bin is a planned node with ``replicas`` models, so nmin > 1 merges planned nodes: with one model per node,
nmin = 20 would turn every 20 nodes into one node 200 K wide, and nmin = replicas would still merge a node whose
replicas partly failed with its neighbour. Library mode therefore always uses nmin = 1 (:class:`NodeLibrary` does),
and ``dumps.flux_integrator``'s default nmin = 20 must be overridden (``factory_kwargs=dict(nmin=1)`` for
run_disc_dumps).

Replicas
--------
FASTWIND is deterministic (the same INDAT gives the same model: production point 571348 failed twice identically,
and the reruns of the pilot points are byte-identical), so replicas need distinct T_eff': replica r of a node sits at
T_node + (r - (R - 1)/2) replica_step (default dT / R: spread evenly over the node's bin). Their mean T_eff' is the
node's, and the bin mean of their profiles averages the model-to-model artefacts of FASTWIND's output (convergence
noise, M424 0.5 % rms in EW between runs; the per-model frequency grids and the 0.01 A wavelength rounding of the OUT
files; the T_eff' branches of the continuum sampling), as a per-point library bin of 10 K averages ~3500 models.
V8 (M424): replicas within ~0.01 K of each other are practically one model (no gain); replicas spread over the bin
(replica_step = dT / R) roughly halve the EW deviation and the time-variable profile deviation at dT 10-20 K, at
every node phase; replicas 1 K apart help little.

Replicas improve only the flux library. The intensity library (:func:`imu_library_from_models`) keeps one
representative per node, the model closest to the node's mean T_eff'; the replicas' intensities are not averaged
(UserWarning when an intensity library is built from nodes with several models). R replicas therefore pay R models
per node for the flux method alone, and the flux and intensity libraries of such a run differ in kind (replica mean
vs single model). With an even R spread over the bin no replica sits at the node: the two middle replicas are equally
close to the mean and the lower one (smaller idx) is taken. One model per node at dT = 10 K is a library of the kind
of the production intensity library, which keeps one model per 10 K bin.

V8: sparse libraries from existing per-point models
---------------------------------------------------
:func:`sparse_library_test` answers "which node spacing suffices?" without new FASTWIND runs: from the models of a
per-point run (M424: the dump-3200 models) it takes, for every node, the 1 or R models closest to the node T_eff'
(within the node's bin; ties by model index: deterministic), builds the library-mode library from them exactly as
:func:`flux_library_from_models` does for real node models, integrates the dump subset with
:class:`~ppmpy.synspec.disc.DiscFlux` (and, optionally, the intensity method with a lazy
:class:`~ppmpy.synspec.disc.DiscImu` from models that have OUT_IMU files), and compares with the per-dump products of
the full per-point library: max|dF| per line over lines of sight, dumps and grid, max|dF0| (no Doppler shifts), the
time-variable part (residuals about the subset mean) and the EW, each also as a fraction of the run's LPV
(:func:`ppmpy.synspec.validate.lpv_residual_rms`). A 'reference' row rebuilds the full library's integrator and must
reproduce the stored products to their float32 rounding (the comparison machinery is sound). The deviation of a sparse
library depends on where its nodes fall (which single models represent the nodes), so every variant is tested at
several node phases (``phases``: nodes at offset + (phase + k) dT), and the recommendation uses the worst phase. A
selection that is the whole pool (the 303 production intensity representatives at dT 10 K, offset 5) is the pool's own
library: a check row, never recommended.

M424 (2026-10-02; dump subset 3200, 3334, 4000, 4169, 4391, 4800 + 10 drawn with default_rng(5); 8 lines of sight;
nodes at 5 + (phase + k) dT K, phases 0, 1/4, 1/2, 3/4, over the subset's T_eff' 33 536-39 006 K plus dT, filled where
the dump-3200 models reach, 35 402-38 905 K; ratios to the LPV residual rms 3.0e-4 / 8.6e-5 / 2.9e-4 and EW rms
1.3e-4 / 2.3e-4 / 2.6e-4 A of the flux run (imu: its own), largest over the lines, range over the phases that are not
checks; /scratch/ppathak/synspec_shadow/m7/libmode/v8/run_v8_phases.py -> v8p_flux.json, v8p_imu.json; flux 6.4 min
on the login node, imu 12.6 min with one process per dT):

=================  ======  =======  ===========  ============  ===========  ===========
variant            models  nodes    max|dF|/LPV  max|dF0|/LPV  max|dR|/LPV  dEW_t/EWrms
=================  ======  =======  ===========  ============  ===========  ===========
flux reference        -    245      0.03 %       0.03 %        0.05 %       0.00 %
flux dT 10, R 1      551   300-304  6.8-14 %     22-59 %       1.21-1.51 %  3.75-4.81 %
flux dT 10, R 3b    1653   300-304  6.1-8.5 %    19-29 %       0.47-0.77 %  1.50-1.86 %
flux dT 20, R 1      278   156-158  16-20 %      41-117 %      1.40-2.67 %  4.40-5.80 %
flux dT 20, R 3b     834   156-158  5.6-19 %     20-52 %       0.79-0.85 %  2.11-2.35 %
flux dT 50, R 1      114   67-68    47-173 %     183-369 %     3.12-4.52 %  9.6-17 %
flux dT 50, R 3b     342   67-68    34-179 %     88-329 %      2.17-4.63 %  8.4-20 %
flux dT 100, R 1      59   35-36    52-165 %     180-275 %     6.6-10.7 %   12-24 %
imu dT 10            551   223-281  0.3-19 %     0.3-24 %      0.13-1.51 %  0.16-4.04 %
imu dT 20            278   156-158  10-27 %      18-53 %       1.27-1.90 %  4.80-9.57 %
imu dT 50            114   67       36-147 %     58-217 %      3.83-4.69 %  13-22 %
=================  ======  =======  ===========  ============  ===========  ===========

(models: planned nodes x replicas, what a library-mode run covering the subset's range needs; nodes: filled from the
dump-3200 models, over the phases; R 3b: 3 replicas spread over the bin; the 3 models closest to the node give R 1 to
<= 2 %.) The time-variable part, which the residual spectra and EW time series of the LPV analysis see, stays small:
max|dR| <= 2.7 % of the LPV rms for dT <= 20 K and <= 4.7 % for 50 K at every phase. The EW of HEI 4026 limits the
spacing, and it is a single-model artefact (the EW(T_eff') sawtooth of the continuum sampling), not a spacing effect:
with one model per node it depends on which models represent the nodes, i.e. on the node phase. dT 10-20 K with R = 1
gives about 4-6 % of the EW rms for HEI 4026 (flux: 3.75-4.81 % at 10 K, 4.40-5.80 % at 20 K); below 5 % robustly
needs replicas spread over the bin (flux only: 1.50-1.86 % at 10 K, 2.11-2.35 % at 20 K) or a fix of the sawtooth. The
static part (the time-mean profile; larger without Doppler broadening, F0) comes from the single models' own artefacts
(per-model frequency grids, the 0.01 A wavelength rounding, continuum-sampling branches), which a 10 K bin of ~3500
models averages; it does not shrink with denser nodes.

The intensity pool has one model per production 10 K bin (the 303 representatives with OUT_IMU files), so imu dT 10 K
can only be emulated by subsets of the production representatives: offset 5 selects all of them (the production
intensity library, reproduced to its float32 rounding: the check, not in the table's ranges), offsets 7.5 and 2.5 keep
279 and 281 (EW 1.69 and 0.16 %), offset 0 (bins centred on the production bin edges) keeps 223, each within 5 K of
its node, about one per 16 K (EW 4.04 %). These subsets are a pessimistic proxy for one model per 10 K node, which is
the production intensity library's kind. imu dT 20 K fails (EW 4.80-9.57 %), and the intensity method gains nothing
from replicas.

Recommendation for small machines (criterion 'lpv' <= 5 % at the worst phase; ``SparseLibraryTest.recommend``): dT =
10 K with one model per node (551 FASTWIND models for this T_eff' range instead of 1 236 544), for both methods, with a
small margin: flux 0.19 % (EW 4.81 % at offset 5), imu 0.96 % (the subset proxy). A robust margin for the flux method
needs 3 replicas spread over the bin (834 models at dT 20 K: 2.35 %; 1653 at 10 K: 1.86 %). When only the residual
spectra matter (criterion 'residual', max|dR|), dT = 50 K (114 models: 3.1-4.7 % at every phase, both methods)
suffices; 100 K does not (6.6-10.7 %).

Validation
----------
tests/synspec/test_libmode.py. Synthetic: node planning covers the sample range plus the margin, with the node grid,
replica offsets and the 'idx teff' table the runner parses (fastwind.batch.split_table) and merge_task accepts; the
planning safeguards (teff_ranges tables recognised column by column, 'idx teff' arrays refused, implausible spans and
node counts refused unless forced, the retry-nudge room warning, ``avoid`` / check_collisions against a per-point run
directory, the points.npz size warning); select_node_models' validation of replica targets and the row-number
validation of ``select``; the flux library from node models equals the analytic node profiles (models on the grid
points, float64: to the ~1e-12 of interp_rows' row offsets), with replica means, empty bins for failed nodes, the plan
checks and nudged = 'keep' / 'drop' / 'raise'; the offset warning without a plan; nmin = 1 keeps every node (nmin = 20
would merge them); node_representatives (the flux library's models in its bins only: unplanned and nudged models
excluded, ties to the smaller idx); an end-to-end run with the fake FASTWIND (plan -> runner -> merge -> combine, a
failed model retried through missing.txt -> library_from_models with the intensity library from extracted OUT_IMU
files and the replica warning -> DiscFlux and DiscImu -> save -> the dumps factories); V8 on a toy run: its reference
row reproduces the reference products, max|dF| decreases with denser nodes, replicas average the models' scatter, node
phases give per-phase and aggregate rows (worst / best / mean) and the deviation depends on the phase; on a small pool
a whole-pool selection is a check row (exact, reused by dedupe, the same without dedupe) and is never recommended; the
recommendation rules (reference, checks, per-phase rows and imu replicas excluded; 'max' / 'mean' over phases). M424
(slow): V8 on three dumps of the subset (44 s; filled nodes within one bin of the pool's T_eff' range). Real FASTWIND
(fastwind, slow; 2 models at once, 185 s on the login node): a library-mode run at the T_eff' of two production
per-point models reproduces their profiles (lam, fcont, fnorm of profiles.npz) and their intensity-library rows
(imu_library_dT10.npz: Ic, Il, s, rmax, nnode, teff_rep) bit for bit.

PP 2026-10-02: new (M7, library mode). PP 2026-10-02: review fixes: V8 check rows never recommended, node phases with
the worst phase deciding, the intensity library from the flux library's models (replicas not averaged), nudged
models kept in their node, offset warning without a plan, select / offsets / teff_span validation, idx collisions;
V8 rerun with phases (run_v8_phases.py).
"""
import glob
import json
import math
import os
import time
import warnings

import numpy as np

from .fwresults import LEDGER_SUFFIX, POINT_DIR, TEFF_NUDGE, TEFF_NUDGE_MAX, ProfileStore, read_ledger
from .library import (IMU_LAYOUT, FluxLibrary, Representatives, build_imu_library, find_candidates, lib_nodes,
                      teff_bins)
from .spectral import LineSet, VelocityGrid

__all__ = ["NodePlan", "plan_teff_nodes", "teff_span", "read_node_plan", "existing_indices", "select_node_models",
           "node_representatives", "flux_library_from_models", "library_nodes", "imu_library_from_models",
           "library_from_models", "NodeLibrary", "sparse_library_test", "SparseLibraryTest", "V8_DTS", "V8_REPLICAS",
           "V8_PHASES", "REPLICA_STEP", "NUDGED", "MAX_NODES", "MAX_REL_SPAN", "PLAN_KIND", "LIBRARY_FILE",
           "IMU_LIBRARY_FILE", "REPRESENTATIVES_FILE", "REPORT_FILE"]

REPLICA_STEP = "bin"
"""Default T_eff' spacing of the replicas of a node: 'bin' = dT / R, the replicas spread evenly over the node's bin
(FASTWIND is deterministic: replicas need distinct T_eff'; V8 on M424: spread replicas average best)."""
V8_DTS = (10.0, 20.0, 50.0)
"""Node spacings [K] of the V8 table."""
V8_REPLICAS = (1, 3)
"""Models per node of the V8 table."""
V8_PHASES = (0.0, 0.25, 0.5, 0.75)
"""Node phases (fractions of dT added to the node offset) of the M424 V8 table: the deviation of a sparse library
depends on where its nodes fall (single-model artefacts), so the table reports the range over the phases and the
recommendation uses the worst one (``sparse_library_test(phases=...)``)."""
NUDGED = "keep"
"""Default handling of node models whose retry nudge moved them across their bin edge
(:func:`flux_library_from_models`): 'keep' them in their planned node."""
MAX_NODES = 100000
"""Largest plan :func:`plan_teff_nodes` makes without ``force`` (M424 at dT 10 K: 551 nodes)."""
MAX_REL_SPAN = 0.5
"""Largest T_eff' span / T_eff' :func:`plan_teff_nodes` accepts without ``force`` (M424: 0.14)."""
PLAN_KIND = "synspec.libmode.node_plan"
"""'_meta' kind of :meth:`NodePlan.write_points_npz`."""
LIBRARY_FILE = "library_nodes.npz"
IMU_LIBRARY_FILE = "imu_library_nodes.npz"
REPRESENTATIVES_FILE = "representatives.txt"
REPORT_FILE = "library_mode.json"
_TEFF_TOL = 1e-3            # K: T_eff of a model (meta.txt, '%.3f') vs the plan


def _float(x, name, positive=False, nonneg=False):
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError("{} must be a number, got {!r}".format(name, x)) from None
    if not math.isfinite(v) or (positive and not v > 0) or (nonneg and not v >= 0):
        raise ValueError("{} must be a {}finite number, got {!r}".format(
            name, "positive " if positive else ("non-negative " if nonneg else ""), x))
    return v


def _int(x, name, minimum=0):
    try:
        ok = int(x) == x
    except (TypeError, ValueError, OverflowError):
        ok = False
    if not ok or isinstance(x, (bool, np.bool_)) or int(x) < minimum:
        raise ValueError("{} must be an integer >= {}, got {!r}".format(name, minimum, x))
    return int(x)


def _logger(log, T0):
    def _log(msg):
        if log is not None:
            log("[{:7.1f} s] {}".format(time.time() - T0, msg))
    return _log


# ----------------------------------------------------------------------------------------------------------------
# node planning
# ----------------------------------------------------------------------------------------------------------------
def _whole(c):
    return bool(np.all(c == np.round(c)))


def _is_ranges_table(a):
    """
    A :func:`ppmpy.synspec.validate.teff_ranges` table, column by column: (n, 6); unique, whole, non-negative dump
    numbers; finite tmin <= tmax; n_lo, n_hi whole and >= 0, or all NaN (teff_ranges without trange); 0 <= std <=
    (tmax - tmin) / 2 (the standard deviation of values inside [tmin, tmax] cannot exceed half the range). An (n, 6)
    array of T_eff' values fails the last test (std ~ T_eff').
    """
    # PP 2026-10-02: every column checked (reviewer: a (2, 6) T_eff' array with a whole first column was a table)
    if not (a.ndim == 2 and a.shape[0] >= 1 and a.shape[1] == 6 and np.all(np.isfinite(a[:, [0, 1, 2, 5]]))):
        return False
    d, lo, hi, sd = a[:, 0], a[:, 1], a[:, 2], a[:, 5]
    if not (_whole(d) and np.all(d >= 0) and np.unique(d).size == d.size and np.all(lo <= hi)):
        return False
    for c in (a[:, 3], a[:, 4]):
        if not (np.all(np.isnan(c)) or (np.all(np.isfinite(c)) and _whole(c) and np.all(c >= 0))):
            return False
    tol = 1e-9 * np.maximum(1.0, np.abs(hi))
    return bool(np.all(sd >= 0) and np.all(sd <= 0.5 * (hi - lo) + tol))


def _is_idx_table(a):
    """An 'idx teff' table as an array (np.loadtxt of points.txt): (n >= 2, 2) with a whole, unique, non-negative
    first column."""
    if not (a.ndim == 2 and a.shape[0] >= 2 and a.shape[1] == 2 and np.all(np.isfinite(a[:, 0]))):
        return False
    c = a[:, 0]
    return _whole(c) and bool(np.all(c >= 0)) and np.unique(c).size == c.size


def _span_kind(a, kind):
    if kind == "ranges":
        if a.ndim != 2 or a.shape[1] < 3:
            raise ValueError("a teff_ranges table needs >= 3 columns (dump, tmin, tmax, ...), got shape {}".format(
                a.shape))
        return "ranges"
    if kind == "values":
        return "values"
    if _is_ranges_table(a):
        return "ranges"
    if _is_idx_table(a):
        raise ValueError("a ({}, 2) array whose first column holds unique whole numbers looks like an 'idx teff' "
                         "table: pass its T_eff' column (a[:, 1]) or the file path, or kind='values' if every number "
                         "is a T_eff'".format(a.shape[0]))
    return "values"


def _span_arrays(x, kind):
    """Yield (array, 'values' | 'ranges') for every item of a T_eff' source."""
    if isinstance(x, (str, os.PathLike)):
        p = os.fspath(x)
        if p.endswith(".npz"):
            with np.load(p) as z:
                if "teff" not in z.files:
                    raise ValueError("{} has no 'teff' member".format(p))
                yield np.asarray(z["teff"], dtype=np.float64).ravel(), "values"
        else:
            t = np.loadtxt(p, ndmin=2)
            if t.shape[1] != 2:
                raise ValueError("{}: expected 'idx teff' lines, got {} columns".format(p, t.shape[1]))
            yield t[:, 1].astype(np.float64), "values"
        return
    if hasattr(x, "keys") and not isinstance(x, np.ndarray):
        if "teff" not in x:
            raise ValueError("a sample mapping needs 'teff'")
        yield np.asarray(x["teff"], dtype=np.float64).ravel(), "values"
        return
    if isinstance(x, np.ndarray) or np.isscalar(x):
        a = np.asarray(x, dtype=np.float64)
        yield a, _span_kind(a, kind)
        return
    items = list(x)
    if items and all(np.isscalar(v) for v in items):
        a = np.asarray(items, dtype=np.float64)
        yield a, _span_kind(a, "values" if kind == "auto" else kind)
        return
    for v in items:
        for out in _span_arrays(v, kind):
            yield out


def teff_span(teff_samples, kind="auto"):
    """
    The T_eff' range (min, max) of samples.

    Parameters
    ----------
    teff_samples:
        Any of, or a list / tuple of any of: an array of T_eff' values [K] (any shape; a (tmin, tmax) pair works too);
        a :func:`ppmpy.synspec.validate.teff_ranges` table (ndumps, 6: dump, tmin, tmax, n_lo, n_hi, std; recognised
        when every column is consistent: unique whole dump numbers, tmin <= tmax, whole n_lo / n_hi or NaN, 0 <= std
        <= (tmax - tmin) / 2; or forced with ``kind='ranges'``); a sample mapping with 'teff'
        (:func:`ppmpy.synspec.dumps.load_sample`); the path of a sample file (.npz with 'teff') or of an 'idx teff'
        table (points.txt).
    kind: {'auto', 'values', 'ranges'}
        How arrays are read ('values': every number is a T_eff'). 'auto' refuses an (n, 2) array whose first column
        holds unique whole numbers (an 'idx teff' table read with np.loadtxt, whose indices would enter the range).

    Returns
    -------
    (float, float)
        tmin, tmax [K].

    Raises
    ------
    ValueError
        No values, a non-finite one, or (kind 'auto') an array that looks like an 'idx teff' table.
    """
    # PP 2026-10-02: new (M7)
    if kind not in ("auto", "values", "ranges"):
        raise ValueError("kind must be 'auto', 'values' or 'ranges', got {!r}".format(kind))
    lo, hi, n = np.inf, -np.inf, 0
    for a, k in _span_arrays(teff_samples, kind):
        v = a[:, 1:3] if k == "ranges" else a
        if v.size == 0:
            continue
        if not np.all(np.isfinite(v)):
            raise ValueError("T_eff' samples must be finite")
        lo, hi, n = min(lo, float(v.min())), max(hi, float(v.max())), n + v.size
    if n == 0:
        raise ValueError("no T_eff' values given")
    return lo, hi


class NodePlan:
    """
    The models of a library-mode run: ``replicas`` models per T_eff' node (:func:`plan_teff_nodes`).

    Parameters
    ----------
    idx: array-like
        (n,) model (point) indices, unique, >= 0 (the 'idx' of the runner's table).
    teff: array-like
        (n,) T_eff' of every model [K], as written in the table (the value of its text).
    node: array-like
        (n,) node of every model (0 .. nn - 1).
    replica: array-like
        (n,) replica number within the node.
    node_teff: array-like
        (nn,) node T_eff' [K], increasing, spaced by dT.
    dT: float
        Node spacing [K]; node k's bin is [node_teff[k] - dT/2, node_teff[k] + dT/2) (the flux library's bins).
    offset, margin, replicas, replica_step, decimals, start_idx:
        The planning options (:func:`plan_teff_nodes`; replica_step None when unknown, e.g. a plan read from a
        table).
    span: (float, float), optional
        T_eff' range of the samples the plan covers.

    Attributes
    ----------
    n, nn: int
        Models and nodes.
    edges: np.ndarray
        (nn + 1,) bin edges of the nodes.
    texts: list of str
        T_eff' of every model as written (``'%.{decimals}f'``).
    """

    def __init__(self, idx, teff, node, replica, node_teff, dT, offset=0.0, margin=None, replicas=1,
                 replica_step=REPLICA_STEP, decimals=3, start_idx=0, span=None):
        self.idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        self.teff = np.asarray(teff, dtype=np.float64).reshape(-1)
        self.node = np.asarray(node, dtype=np.int64).reshape(-1)
        self.replica = np.asarray(replica, dtype=np.int64).reshape(-1)
        self.node_teff = np.asarray(node_teff, dtype=np.float64).reshape(-1)
        self.dT = _float(dT, "dT", positive=True)
        self.offset = _float(offset, "offset")
        self.margin = None if margin is None else _float(margin, "margin", nonneg=True)
        self.replicas = _int(replicas, "replicas", 1)
        self.replica_step = (None if replica_step is None else self.dT / self.replicas if replica_step == "bin"
                             else _float(replica_step, "replica_step", nonneg=True))
        self.decimals = _int(decimals, "decimals", 0)
        self.start_idx = _int(start_idx, "start_idx", 0)
        self.span = None if span is None else (float(span[0]), float(span[1]))
        n, nn = self.idx.size, self.node_teff.size
        if not (self.teff.shape == self.node.shape == self.replica.shape == (n,)):
            raise ValueError("idx, teff, node and replica must have one length")
        if nn < 1 or not np.all(np.isfinite(self.node_teff)):
            raise ValueError("need at least one finite node T_eff'")
        if nn > 1 and not np.allclose(np.diff(self.node_teff), self.dT, rtol=1e-9, atol=1e-9 * self.dT):
            raise ValueError("node T_eff' must increase in steps of dT = {:g} K".format(self.dT))
        if n and (self.idx.min() < 0 or np.unique(self.idx).size != n):
            raise ValueError("model indices must be unique and >= 0")
        if n and (self.node.min() < 0 or self.node.max() >= nn):
            raise ValueError("node numbers must lie in 0 .. {}".format(nn - 1))
        if n and not np.all(np.isfinite(self.teff) & (self.teff > 0)):
            raise ValueError("model T_eff' must be positive and finite")
        self._order = np.argsort(self.idx, kind="stable")

    @property
    def n(self):
        """Number of models."""
        return self.idx.size

    @property
    def nn(self):
        """Number of nodes."""
        return self.node_teff.size

    def __len__(self):
        return self.n

    @property
    def edges(self):
        """(nn + 1,) bin edges: node_teff[0] - dT/2 + dT k (k = 0 .. nn), the bins of the flux library."""
        return self.node_teff[0] - 0.5 * self.dT + self.dT * np.arange(self.nn + 1)

    @property
    def texts(self):
        """The T_eff' of every model as written in the table."""
        return ["{:.{d}f}".format(t, d=self.decimals) for t in self.teff]

    def __repr__(self):
        return "NodePlan({} models, {} nodes {:.{d}f}-{:.{d}f} K, dT {:g} K, {} replica(s), idx {}..{})".format(
            self.n, self.nn, self.node_teff[0], self.node_teff[-1], self.dT, self.replicas,
            int(self.idx.min()) if self.n else "-", int(self.idx.max()) if self.n else "-",
            d=max(0, min(self.decimals, 3)))

    def position(self, idx):
        """
        Position of model indices in the plan (index into ``idx``, ``teff``, ...), -1 for indices not planned.

        Parameters
        ----------
        idx: array-like of int

        Returns
        -------
        np.ndarray of int64
        """
        q = np.asarray(idx, dtype=np.int64)
        if self.n == 0:
            return np.full(q.shape, -1, np.int64)
        srt = self.idx[self._order]
        k = np.clip(np.searchsorted(srt, q), 0, self.n - 1)
        return np.where(srt[k] == q, self._order[k], -1).astype(np.int64)

    def table_lines(self):
        """The 'idx teff' lines of the table (points.txt format), in plan order (node, replica)."""
        return ["{} {}\n".format(int(i), t) for i, t in zip(self.idx, self.texts)]

    def write(self, path):
        """
        Write the 'idx teff' table (the points.txt / missing.txt format of a per-point run: one line per model,
        nothing else) atomically. The FASTWIND runner (``python3 -m ppmpy.synspec.fastwind run RUN_DIR [LIST]``;
        :func:`ppmpy.synspec.fastwind.batch.run_models`) and :func:`ppmpy.synspec.fwresults.combine` read it
        unchanged.

        Returns
        -------
        str
            path.
        """
        # PP 2026-10-02: new (M7)
        path = os.fspath(path)
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, ".{}.tmp{}".format(os.path.basename(path), os.getpid()))
        try:
            with open(tmp, "w") as f:
                f.write("".join(self.table_lines()))
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return path

    def params(self):
        """The planning options (JSON-able)."""
        return dict(dT=self.dT, offset=self.offset, margin=self.margin, replicas=self.replicas,
                    replica_step=self.replica_step, decimals=self.decimals, start_idx=self.start_idx,
                    span=None if self.span is None else list(self.span), n=int(self.n), nn=int(self.nn),
                    node_range=[float(self.node_teff[0]), float(self.node_teff[-1])])

    def check_collisions(self, avoid, error=True):
        """
        Model indices of the plan that another run already uses.

        A library-mode run belongs in its own run directory: the runner skips every idx its results tree already
        holds (whatever the T_eff'), and merge_task / combine would mix another run's models into the node library
        (caught only when the library is built, after the run). Call this before running a plan next to existing
        results.

        Parameters
        ----------
        avoid:
            Indices in use (:func:`existing_indices`: a run directory, results directory, points.txt / missing.txt,
            profiles.npz / points.npz, ledger, array of indices, or a list of these).
        error: bool
            Raise ValueError on a collision (default) instead of returning the indices.

        Returns
        -------
        np.ndarray of int64
            The colliding indices (sorted; empty if none).
        """
        # PP 2026-10-02: new (reviewer: idx collisions with a per-point run)
        hit = np.intersect1d(self.idx, existing_indices(avoid))
        if hit.size and error:
            raise ValueError("{} planned model indices are already used ({}{}): plan with avoid=... (start_idx beyond "
                             "them) and run the plan in its own run directory".format(
                                 hit.size, hit[:10].tolist(), " ..." if hit.size > 10 else ""))
        return hit

    def points_arrays(self):
        """
        The members of a points.npz for :func:`ppmpy.synspec.fwresults.merge_task` (which needs ``idx`` = row number
        and ``teff``): idx = 0 .. max(idx), teff (NaN for rows not planned), node, replica (-1 for those rows), and
        the node table node_teff. Rows are allocated up to max(idx), so a large ``start_idx`` makes a large file
        (:meth:`write_points_npz` warns).
        """
        m = int(self.idx.max()) + 1 if self.n else 0
        teff = np.full(m, np.nan)
        node = np.full(m, -1, np.int64)
        rep = np.full(m, -1, np.int64)
        teff[self.idx], node[self.idx], rep[self.idx] = self.teff, self.node, self.replica
        return dict(idx=np.arange(m, dtype=np.int64), teff=teff, node=node, replica=rep,
                    node_teff=self.node_teff.copy())

    def write_points_npz(self, path, meta=True):
        """
        Write :meth:`points_arrays` (uncompressed .npz, atomically) with the plan in '_meta' (``params``), for
        ``fwresults.merge_task(results, tag, out, points=path, copy_keys=())`` and :meth:`read`.

        Returns
        -------
        str
            path.

        Warns
        -----
        UserWarning
            The file holds more than max(100 000, 10 n) rows (merge_task indexes points.npz by idx, so rows run up to
            max(idx): a start_idx far beyond the plan's size, e.g. 1e7, writes 1e7 rows).
        """
        # PP 2026-10-02: new (M7); warning on large start_idx (reviewer)
        from .io import make_meta, save_npz
        rows = int(self.idx.max()) + 1 if self.n else 0
        if rows > max(100000, 10 * self.n):
            warnings.warn("points.npz gets {} rows for {} models (merge_task needs idx = row number; start_idx {}): a "
                          "library-mode run in its own run directory can start at idx 0".format(rows, self.n,
                                                                                               self.start_idx))
        m = make_meta(PLAN_KIND, params=self.params()) if meta else None
        return save_npz(os.fspath(path), self.points_arrays(), meta=m)

    @classmethod
    def read(cls, path, dT=None, offset=0.0, decimals=None):
        """
        Read a plan.

        Parameters
        ----------
        path: str or os.PathLike
            A points.npz of :meth:`write_points_npz` (the whole plan; ``dT``, ``offset`` ignored), or an 'idx teff'
            table (:meth:`write`), for which ``dT`` is needed: node k = floor((T - offset) / dT + 1/2), i.e. the bin
            [node - dT/2, node + dT/2) holding T; replicas are numbered in T_eff' order within a node.
        dT, offset: float
            Node grid of a table.
        decimals: int, optional
            Decimals of a table's T_eff' (default: the most found in the file).

        Returns
        -------
        NodePlan
        """
        # PP 2026-10-02: new (M7)
        path = os.fspath(path)
        if path.endswith(".npz"):
            from .io import read_meta
            with np.load(path) as z:
                meta = read_meta(z)
                if meta.get("kind") != PLAN_KIND:
                    raise ValueError("{} is not a node plan (no '_meta' of kind {})".format(path, PLAN_KIND))
                rows = np.flatnonzero(z["node"] >= 0)
                p = meta["params"]
                return cls(z["idx"][rows], z["teff"][rows], z["node"][rows], z["replica"][rows], z["node_teff"],
                           p["dT"], offset=p["offset"], margin=p["margin"], replicas=p["replicas"],
                           replica_step=p["replica_step"], decimals=p["decimals"], start_idx=p["start_idx"],
                           span=p["span"])
        if dT is None:
            raise ValueError("dT is needed to read a plan from an 'idx teff' table")
        dT, offset = _float(dT, "dT", positive=True), _float(offset, "offset")
        idx, txt = [], []
        with open(path) as f:
            for ln in f:
                tok = ln.split()
                if not tok or tok[0].startswith("#"):
                    continue
                if len(tok) != 2:
                    raise ValueError("{}: expected 'idx teff', got {!r}".format(path, ln))
                idx.append(int(tok[0]))
                txt.append(tok[1])
        if not idx:
            raise ValueError("{} lists no models".format(path))
        teff = np.array([float(t) for t in txt])
        if decimals is None:
            decimals = max(len(t.split(".")[1]) if "." in t else 0 for t in txt)
        k = np.floor((teff - offset) / dT + 0.5).astype(np.int64)
        k0 = int(k.min())
        node_teff = offset + dT * np.arange(k0, int(k.max()) + 1)
        node = k - k0
        order = np.lexsort((np.asarray(idx), teff, node))
        rep = np.zeros(teff.size, np.int64)
        for pos in range(1, order.size):
            a, b = order[pos - 1], order[pos]
            rep[b] = rep[a] + 1 if node[a] == node[b] else 0
        return cls(idx, teff, node, rep, node_teff, dT, offset=offset, replicas=int(rep.max()) + 1, replica_step=None,
                   decimals=decimals, start_idx=min(idx))


def read_node_plan(path, dT=None, offset=0.0, decimals=None):
    """:meth:`NodePlan.read`."""
    return NodePlan.read(path, dT=dT, offset=offset, decimals=decimals)


def _first_column(path):
    """The integer first field of every non-empty, non-comment line of a text table."""
    out = []
    with open(path) as f:
        for n, ln in enumerate(f, 1):
            s = ln.split(None, 1)
            if not s or s[0].startswith("#"):
                continue
            try:
                out.append(int(s[0]))
            except ValueError:
                raise ValueError("{}:{}: expected an integer index first, got {!r}".format(path, n, ln.rstrip())) \
                    from None
    return out


def _npz_indices(path):
    with np.load(path) as z:
        if "idx" not in z.files:
            raise ValueError("{} has no 'idx' member".format(path))
        idx = np.asarray(z["idx"], dtype=np.int64).reshape(-1)
        if "node" in z.files:                                   # a library-mode points.npz: the planned rows
            return idx[np.asarray(z["node"]).reshape(-1) >= 0]
        return idx


def existing_indices(src):
    """
    Model (point) indices already used by a run, to keep a library-mode plan clear of them
    (:func:`plan_teff_nodes` ``avoid``, :meth:`NodePlan.check_collisions`).

    Parameters
    ----------
    src:
        A run directory (its points.txt, missing*.txt, profiles.npz, points.npz and the ledgers
        ``results/<tag>/part_*.idx``), a results directory (``<tag>/part_*.idx``), an 'idx teff' table
        (points.txt, missing.txt), a ledger (part_*.idx), a profiles.npz / points.npz ('idx'; a library-mode
        points.npz: its planned rows), an array of indices, or a list / tuple of these.

    Returns
    -------
    np.ndarray of int64
        Sorted unique indices (empty for a directory that holds none of these files).

    Raises
    ------
    FileNotFoundError
        A path that does not exist.
    ValueError
        A table whose first field is not an integer, an .npz without 'idx'.
    """
    # PP 2026-10-02: new (reviewer: idx collisions of a library-mode list with a per-point run)
    if isinstance(src, (str, os.PathLike)):
        p = os.fspath(src)
        if os.path.isdir(p):
            got = []
            for name in ["points.txt"] + sorted(os.path.basename(x)
                                                for x in glob.glob(os.path.join(p, "missing*.txt"))):
                if os.path.isfile(os.path.join(p, name)):
                    got += _first_column(os.path.join(p, name))
            parts = []
            for name in ("profiles.npz", "points.npz"):
                if os.path.isfile(os.path.join(p, name)):
                    parts.append(_npz_indices(os.path.join(p, name)))
            for pat in ("*/part_*" + LEDGER_SUFFIX, "results/*/part_*" + LEDGER_SUFFIX):
                for led in sorted(glob.glob(os.path.join(p, pat))):
                    got += [i for i, _ in read_ledger(led)]
            parts.append(np.asarray(got, dtype=np.int64))
            return np.unique(np.concatenate(parts))
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        if p.endswith(".npz"):
            return np.unique(_npz_indices(p))
        if p.endswith(LEDGER_SUFFIX):
            return np.unique(np.asarray([i for i, _ in read_ledger(p)], dtype=np.int64))
        return np.unique(np.asarray(_first_column(p), dtype=np.int64))
    if isinstance(src, (list, tuple)) and any(isinstance(x, (str, os.PathLike, np.ndarray, list, tuple)) for x in src):
        return np.unique(np.concatenate([existing_indices(x) for x in src] + [np.zeros(0, np.int64)]))
    a = np.asarray(src).reshape(-1)
    if a.size == 0:
        return np.zeros(0, np.int64)
    if not (np.issubdtype(a.dtype, np.integer) or (np.issubdtype(a.dtype, np.floating) and _whole(a))):
        raise ValueError("indices must be integers")
    return np.unique(a.astype(np.int64))


def plan_teff_nodes(teff_samples, dT=10.0, margin=None, replicas=1, start_idx=0, decimals=3, offset=0.0,
                    replica_step=REPLICA_STEP, kind="auto", avoid=None, force=False):
    """
    Plan the FASTWIND models of a library-mode run: T_eff' nodes every dT covering the T_eff' range of the samples
    plus a margin, ``replicas`` models per node.

    Parameters
    ----------
    teff_samples:
        T_eff' of the per-dump samples the library must cover (:func:`teff_span`): e.g. a
        :func:`ppmpy.synspec.validate.teff_ranges` table over the chosen dumps (M424, all 1601 dumps: 2 s with 8
        workers), sample mappings, arrays, sample files, or an (tmin, tmax) pair.
    dT: float
        Node spacing [K] (V8, :func:`sparse_library_test`, gives the error of the spacing; M424 production bins: 10).
    margin: float, optional
        Extra range [K] on both sides (default dT): nodes run from at most tmin - margin to at least tmax + margin, so
        T_eff' a little beyond the planned samples is interpolated, not clamped to the end node.
    replicas: int
        Models per node, at T_node + (r - (R - 1)/2) replica_step (r = 0 .. R - 1; mean T_node). FASTWIND is
        deterministic, so replicas need distinct T_eff'; their mean averages the model-to-model convergence noise.
    start_idx: int
        Index of the first model; model (node k, replica r) gets idx = start_idx + k R + r. Run a plan in its own run
        directory: the runner skips every idx its results tree already holds, and the merge would mix another run's
        models into the library. With ``avoid``, start_idx is raised beyond the indices in use.
    decimals: int
        Decimals of the T_eff' in the table (3: the '%.3f' of points.txt, missing.txt and the INDAT TEFF of the
        per-point runs; meta.txt and the merged profiles then record exactly these values).
    offset: float
        Nodes lie at offset + k dT (k integer). 0 (default): multiples of dT. M424: offset 5 puts nodes at the
        centres of the 10 K bins of the per-point library (35405, 35415, ... K; bins [35400, 35410), ...).
    replica_step: float or 'bin'
        T_eff' spacing of the replicas [K], or 'bin' (default, :data:`REPLICA_STEP`): dT / R, the replicas spread
        evenly over the node's bin (the replica mean then averages FASTWIND's model-to-model artefacts as a per-point
        bin does; V8 on M424 at dT 10-20 K: half the EW and residual deviation of one model per node, while replicas
        1 K apart help little). Replicas improve only the flux library: the intensity library keeps one model per
        node (:func:`imu_library_from_models`). (R - 1) replica_step / 2 must be below dT / 2 (every replica inside
        its node's bin); the top replica keeps dT / (2R) to the bin edge (less the rounding). A retry nudge (+1 K per
        failed attempt) can move it across: :func:`flux_library_from_models` keeps such a model in its planned node
        (nudged='keep'), but the intensity library cannot use it as the node's representative (a warning when the
        room is <= 1 K).
    kind: {'auto', 'values', 'ranges'}
        How arrays in ``teff_samples`` are read (:func:`teff_span`).
    avoid: optional
        Indices in use (:func:`existing_indices`: e.g. the per-point run directory): start_idx is raised to one
        beyond the largest of them.
    force: bool
        Plan even when the plan looks wrong: more than :data:`MAX_NODES` nodes, or a T_eff' span above
        :data:`MAX_REL_SPAN` of T_eff' (wrong samples, e.g. model indices or dump numbers read as T_eff').

    Returns
    -------
    NodePlan
        Models in the order node, replica; T_eff' rounded to ``decimals`` (the value of the table's text).

    Raises
    ------
    ValueError
        Bad options, replicas that would leave their bin or collide after rounding, T_eff' <= 0, an implausible plan
        (unless force).

    Warns
    -----
    UserWarning
        The top replica lies <= 1 K (the retry nudge) below its bin edge.

    Examples
    --------
    >>> from ppmpy.synspec import validate, libmode
    >>> rg = validate.teff_ranges(SAMPLES, range(3200, 4801), nproc=8)           # doctest: +SKIP
    >>> plan = libmode.plan_teff_nodes(rg, dT=10.0)                              # doctest: +SKIP
    >>> plan.write("RUN_DIR/points.txt"); plan.write_points_npz("RUN_DIR/points.npz")   # doctest: +SKIP
    """
    # PP 2026-10-02: new (M7)
    dT = _float(dT, "dT", positive=True)
    margin = dT if margin is None else _float(margin, "margin", nonneg=True)
    R = _int(replicas, "replicas", 1)
    start_idx = _int(start_idx, "start_idx", 0)
    decimals = _int(decimals, "decimals", 0)
    offset = _float(offset, "offset")
    step = dT / R if replica_step == "bin" else _float(replica_step, "replica_step", nonneg=True)
    if R > 1:
        if not step > 0:
            raise ValueError("replica_step must be > 0 for replicas > 1")
        if not 0.5 * (R - 1) * step < 0.5 * dT:
            raise ValueError("{} replicas {:g} K apart span {:g} K: they must stay inside the node's bin of dT = {:g} "
                             "K (lower replica_step)".format(R, step, (R - 1) * step, dT))
    lo, hi = teff_span(teff_samples, kind)
    if not force and hi > 0 and hi - lo > MAX_REL_SPAN * hi:
        raise ValueError("the samples span {:g}-{:g} K, more than {:.0%} of T_eff': wrong samples (model indices or "
                         "dump numbers read as T_eff'?); force=True plans anyway".format(lo, hi, MAX_REL_SPAN))
    k0 = int(math.floor((lo - margin - offset) / dT))
    k1 = int(math.ceil((hi + margin - offset) / dT))
    while offset + k0 * dT > lo - margin:                  # guard against rounding of the quotients
        k0 -= 1
    while offset + k1 * dT < hi + margin:
        k1 += 1
    if not force and k1 - k0 + 1 > MAX_NODES:
        raise ValueError("{} nodes ({:g}-{:g} K at dT {:g} K) exceed MAX_NODES = {}: wrong samples or dT?; force=True "
                         "plans anyway".format(k1 - k0 + 1, lo, hi, dT, MAX_NODES))
    if avoid is not None:
        used = existing_indices(avoid)
        if used.size:
            start_idx = max(start_idx, int(used.max()) + 1)
    node_teff = offset + dT * np.arange(k0, k1 + 1, dtype=np.float64)
    if not node_teff[0] > 0:
        raise ValueError("the nodes reach T_eff' <= 0 ({:g} K)".format(node_teff[0]))
    offs = (np.arange(R) - 0.5 * (R - 1)) * step
    t = (node_teff[:, None] + offs[None, :]).ravel()
    texts = ["{:.{d}f}".format(v, d=decimals) for v in t]
    teff = np.array([float(s) for s in texts])
    if len(set(texts)) != len(texts):
        raise ValueError("model T_eff' collide at {} decimals (replica_step {:g} K): use more decimals".format(
            decimals, step))
    nn = node_teff.size
    node = np.repeat(np.arange(nn, dtype=np.int64), R)
    rep = np.tile(np.arange(R, dtype=np.int64), nn)
    edges = node_teff[0] - 0.5 * dT + dT * np.arange(nn + 1)
    if np.any(teff_bins(teff, edges) != node):
        raise ValueError("rounded replica T_eff' leave their node's bin (decimals {}, replica_step {:g} K)".format(
            decimals, step))
    room = float(np.min(edges[node + 1] - teff))
    if room <= TEFF_NUDGE:
        warnings.warn("the top replica of a node lies {:.3g} K below its bin edge: one retry nudge (+{:g} K) moves it "
                      "into the next bin (flux_library_from_models keeps it in its node, nudged='keep'; it cannot be "
                      "the node's intensity-library representative); fewer replicas or a smaller replica_step leave "
                      "more room".format(room, TEFF_NUDGE))
    return NodePlan(start_idx + np.arange(nn * R, dtype=np.int64), teff, node, rep, node_teff, dT, offset=offset,
                    margin=margin, replicas=R, replica_step=step, decimals=decimals, start_idx=start_idx,
                    span=(lo, hi))


def _as_plan(plan):
    if plan is None or isinstance(plan, NodePlan):
        return plan
    if isinstance(plan, (str, os.PathLike)):
        return NodePlan.read(plan)
    raise ValueError("plan must be a NodePlan or the path of its points.npz, got {!r}".format(type(plan).__name__))


# ----------------------------------------------------------------------------------------------------------------
# node models
# ----------------------------------------------------------------------------------------------------------------
def _edges_of(node_teff, edges=None):
    t = np.asarray(node_teff, dtype=np.float64).reshape(-1)
    if t.size == 0 or not np.all(np.isfinite(t)) or (t.size > 1 and not np.all(np.diff(t) > 0)):
        raise ValueError("node T_eff' must be finite and increasing")
    if edges is not None:
        e = np.asarray(edges, dtype=np.float64).reshape(-1)
        if e.size != t.size + 1 or not np.all(np.diff(e) > 0) or np.any(t < e[:-1]) or np.any(t >= e[1:]):
            raise ValueError("edges must be (nn + 1,) increasing with node k in [edges[k], edges[k + 1])")
        return t, e
    if t.size == 1:
        raise ValueError("a single node needs its edges")
    mid = 0.5 * (t[:-1] + t[1:])
    return t, np.concatenate([[t[0] - (mid[0] - t[0])], mid, [t[-1] + (t[-1] - mid[-1])]])


def _closest(Ts, o, teff, key, lo, hi, target, n):
    """Up to n rows (of the sorted candidates Ts = teff[o], positions lo .. hi - 1) closest to target, ordered by
    (|T - target|, key); all candidates at the n-th distance are considered, so ties go to the smaller key."""
    if hi <= lo:
        return np.zeros(0, np.int64), np.zeros(0)
    p = int(np.searchsorted(Ts, target))
    w0, w1 = max(lo, p - n), min(hi, p + n)
    if w1 <= w0:                        # target outside the window (select_node_models validates the targets)
        return np.zeros(0, np.int64), np.zeros(0)
    d = np.abs(Ts[w0:w1] - target)
    dn = float(np.partition(d, n - 1)[n - 1]) if d.size > n else float(d.max())
    pad = 1e-9 * max(1.0, abs(target))
    c0 = max(lo, int(np.searchsorted(Ts, target - dn - pad, side="left")))
    c1 = min(hi, int(np.searchsorted(Ts, target + dn + pad, side="right")))
    rr = o[c0:c1]
    dd = np.abs(teff[rr] - target)
    keep = dd <= dn
    rr, dd = rr[keep], dd[keep]
    s = np.lexsort((key[rr], dd))[:n]
    return rr[s], dd[s]


def select_node_models(teff, node_teff, replicas=1, edges=None, idx=None, usable=None, offsets=None):
    """
    For every node, the ``replicas`` models closest in T_eff' to the node (or to the node plus each replica's
    offset), among the models inside the node's bin.

    Parameters
    ----------
    teff: array-like
        (N,) T_eff' of the available models [K] (e.g. the per-point models of a run, ``ProfileStore.teff``).
    node_teff: array-like
        (nn,) node T_eff', increasing.
    replicas: int
        Models per node.
    edges: array-like, optional
        (nn + 1,) bin edges (node k's bin is [edges[k], edges[k + 1]), as :func:`ppmpy.synspec.library.teff_bins`).
        Default: midpoints between nodes (end bins symmetric about the end nodes); :attr:`NodePlan.edges` for a plan.
    idx: array-like, optional
        (N,) model indices, the tie-breaker (default the row numbers).
    usable: array-like of bool, optional
        (N,) models that may be chosen (default all).
    offsets: array-like, optional
        (replicas,) T_eff' offsets [K] of the replicas from the node (as :func:`plan_teff_nodes` places them:
        (r - (R - 1)/2) replica_step): replica r is the model closest to node + offsets[r] that no earlier replica of
        the node took. Default: the R models closest to the node itself. Every target node + offsets[r] must lie in
        the node's bin (ValueError otherwise).

    Returns
    -------
    rows: np.ndarray
        (nn, replicas) int64 row numbers of the chosen models (default: closest first; ties: smaller idx first; with
        offsets: in replica order); -1 where a node's bin holds fewer models.
    dist: np.ndarray
        (nn, replicas) |T_eff' - target| [K] (NaN where -1).

    Notes
    -----
    Deterministic: the order of the models in the input does not matter (sorted by (T_eff', idx)). Models outside
    every bin are never chosen. In a dense per-point run (M424: ~350 models per K) the models closest to a node lie
    within ~0.01 K of it; FASTWIND's output barely changes over such a T_eff' step, so such 'replicas' are not
    independent (V8: 3 closest models = 1 model); offsets emulate replicas spread as a library-mode run spreads them.
    """
    # PP 2026-10-02: new (M7, the node models of V8)
    teff = np.asarray(teff, dtype=np.float64).reshape(-1)
    N = teff.size
    R = _int(replicas, "replicas", 1)
    tn, e = _edges_of(node_teff, edges)
    nn = tn.size
    key = np.arange(N, dtype=np.int64) if idx is None else np.asarray(idx, dtype=np.int64).reshape(-1)
    if key.shape != (N,):
        raise ValueError("idx must have the shape of teff")
    ok = np.isfinite(teff) if usable is None else (np.asarray(usable, dtype=bool).reshape(-1) & np.isfinite(teff))
    if ok.shape != (N,):
        raise ValueError("usable must have the shape of teff")
    if offsets is not None:
        offsets = np.asarray(offsets, dtype=np.float64).reshape(-1)
        if offsets.shape != (R,) or not np.all(np.isfinite(offsets)):
            raise ValueError("offsets must be {} finite numbers (one per replica)".format(R))
        tgt = tn[:, None] + offsets[None, :]
        out = (tgt < e[:-1, None]) | (tgt >= e[1:, None])
        if out.any():
            k = int(np.flatnonzero(out.any(axis=1))[0])
            raise ValueError("replica targets node + offsets must lie inside the node's bin [edges[k], edges[k + 1]): "
                             "offsets {} put node {:g} K's targets {} outside [{:g}, {:g}) ({} of {} nodes)".format(
                                 offsets.tolist(), tn[k], tgt[k][out[k]].tolist(), e[k], e[k + 1],
                                 int(out.any(axis=1).sum()), nn))
    cand = np.flatnonzero(ok)
    o = cand[np.lexsort((key[cand], teff[cand]))]
    Ts = teff[o]
    a = np.searchsorted(Ts, e[:-1], side="left")
    b = np.searchsorted(Ts, e[1:], side="left")
    rows = np.full((nn, R), -1, np.int64)
    dist = np.full((nn, R), np.nan)
    for k in range(nn):
        lo, hi = int(a[k]), int(b[k])
        if offsets is None:
            rr, dd = _closest(Ts, o, teff, key, lo, hi, tn[k], R)
            rows[k, :rr.size], dist[k, :rr.size] = rr, dd
            continue
        taken = set()
        for r in range(R):
            rr, dd = _closest(Ts, o, teff, key, lo, hi, tn[k] + offsets[r], R)
            for i, x in zip(rr, dd):
                if int(i) not in taken:
                    taken.add(int(i))
                    rows[k, r], dist[k, r] = i, x
                    break
    return rows, dist


def _as_store(store):
    if isinstance(store, ProfileStore):
        return store
    if isinstance(store, (str, os.PathLike)):
        return ProfileStore.open(os.fspath(store))
    if hasattr(store, "keys"):
        return ProfileStore({k: store[k] for k in store.keys()})
    raise ValueError("store must be a ProfileStore, a profiles.npz path or a mapping of its members")


def _substore(st, rows):
    """A ProfileStore (arrays) of some rows of a store: the members idx, teff, status, niter, teff_nudge (where
    present), lam, fcont, fnorm, lines; fancy indexing reads only those rows of memory maps."""
    rows = np.asarray(rows, dtype=np.int64)
    m = {}
    for k in ("idx", "teff", "status", "niter", "teff_nudge"):
        if k in st:
            m[k] = np.asarray(st[k])[rows]
    for k in ("lam", "fcont", "fnorm"):
        m[k] = np.asarray(st[k][rows])
    m["lines"] = np.asarray(st["lines"])
    if st.meta:
        m["_meta"] = np.array(json.dumps(st.meta))
    return ProfileStore(m)


def _lines_of(st, lref, lines=None):
    """Line names (the store's, checked against a LineSet / ``lines``) and the reference wavelengths."""
    names = list(st.lines)
    if lines is not None:
        given = [lines] if isinstance(lines, str) else list(getattr(lines, "names", lines))
        if [str(x) for x in given] != names:
            raise ValueError("lines {} differ from the store's lines {}".format(given, names))
    if isinstance(lref, LineSet):
        if list(lref.names) != names:
            raise ValueError("the LineSet names {} differ from the store's lines {}".format(lref.names, names))
        lr = lref.lref
    else:
        lr = np.atleast_1d(np.asarray(getattr(lref, "lref", lref), dtype=np.float64))
    if lr.shape != (len(names),):
        raise ValueError("need one reference wavelength per line of the store ({}), got {}".format(len(names),
                                                                                               lr.size))
    return names, lr


def _usable(st, select=None, cap_ok=True, cap=None):
    ok = st.usable(cap_ok=cap_ok, cap=cap)
    if select is not None:
        s = np.asarray(select)
        if s.dtype == bool:
            if s.shape != (st.n,):
                raise ValueError("a bool select must have shape ({},)".format(st.n))
            ok &= s
        else:
            # PP 2026-10-02: row numbers validated (reviewer: the -1 of select_node_models selected the last row)
            s = s.reshape(-1)
            m = np.zeros(st.n, bool)
            if s.size:
                if not np.issubdtype(s.dtype, np.integer):
                    raise ValueError("select must be a bool mask or integer row numbers, got dtype {}".format(s.dtype))
                if s.min() < 0 or s.max() >= st.n:
                    raise ValueError("select row numbers must lie in 0 .. {}, got {} .. {} (drop the -1 entries of "
                                     "select_node_models: rows[rows >= 0])".format(st.n - 1, int(s.min()),
                                                                                   int(s.max())))
                m[s.astype(np.int64)] = True
            ok &= m
    return ok


def _plan_check(plan, st, ok, unplanned, nudged=NUDGED):
    """Models of the store vs the plan: positions, unplanned models, T_eff' (meta.txt) vs the plan, models nudged
    across their bin edge (``nudged``). Returns (planned mask, position in the plan, moved mask)."""
    idx = np.asarray(st.idx, dtype=np.int64)
    pos = plan.position(idx)
    unpl = pos < 0
    if unpl.any():
        msg = "{} models of the store are not in the plan (idx {}{})".format(
            int(unpl.sum()), idx[unpl][:10].tolist(), " ..." if unpl.sum() > 10 else "")
        if unplanned == "raise":
            raise ValueError(msg + ": another run's models? (unplanned='skip' leaves them out)")
        warnings.warn(msg + ": left out")
    teff = np.asarray(st.teff, dtype=np.float64)
    pl = ~unpl
    d = np.full(st.n, np.nan)
    d[pl] = teff[pl] - plan.teff[pos[pl]]
    bad = pl & ((d < -_TEFF_TOL) | (d > TEFF_NUDGE_MAX + _TEFF_TOL))
    if bad.any():
        raise ValueError("{} models have a T_eff' other than planned (plus a retry nudge of 0..{:g} K): idx {}".format(
            int(bad.sum()), TEFF_NUDGE_MAX, idx[bad][:10].tolist()))
    use = ok & pl
    moved = use & (teff_bins(teff, plan.edges) != np.where(pl, plan.node[np.maximum(pos, 0)], -1))
    moved |= use & ((teff < plan.edges[0]) | (teff >= plan.edges[-1]))
    if moved.any():
        msg = "{} models lie outside their planned node's bin (retry nudges across the bin edge: idx {})".format(
            int(moved.sum()), idx[moved][:10].tolist())
        if nudged == "raise":
            raise ValueError(msg + "; nudged='keep' keeps them in their planned node, or plan a smaller replica_step "
                             "or larger dT")
        warnings.warn(msg + (": left out (nudged='drop')" if nudged == "drop" else
                             ": kept in their planned node (nudged='keep'; the node's mean T_eff' includes the nudge)"))
    return pl, pos, moved


def _offset_check(teff, dT, offset):
    """
    Warn when bins at offset + (k +- 1/2) dT cut through the T_eff' clusters of node models (no plan given): the
    models' phases ((T - offset) / dT + 1/2) mod 1 (bin edges at 0) are clustered when their largest circular gap
    is >= 0.05 (library-mode models; a dense per-point run is not checked); then the edges should lie well inside a
    gap: a model within 10 % of the largest gap from an edge means a wrong offset (e.g. replicas of neighbouring
    nodes in one bin). Returns the offset suggested by the largest gap (None when not checked).
    """
    # PP 2026-10-02: new (reviewer: a wrong offset without a plan silently mixed neighbouring nodes' replicas)
    t = np.asarray(teff, dtype=np.float64)
    if t.size < 2:
        return None
    ph = np.sort(np.mod((t - offset) / dT + 0.5, 1.0))
    gaps = np.diff(np.concatenate([ph, [ph[0] + 1.0]]))
    g = int(np.argmax(gaps))
    if gaps[g] < 0.05:
        return None
    centre = math.fmod(ph[g] + 0.5 * gaps[g], 1.0)
    suggested = float(np.mod(offset + centre * dT, dT))
    dist = float(np.min(np.minimum(ph, 1.0 - ph)))
    if dist < 0.1 * gaps[g]:
        warnings.warn("models lie within {:.3g} K of the bin edges of offset {:g} K (dT {:g} K): the bins may mix the "
                      "replicas of neighbouring nodes; the models' T_eff' suggest offset {:.4g} K (mod dT). Pass the "
                      "run's plan (plan=...) to bin by planned node".format(dist * dT, offset, dT, suggested))
    return suggested


def flux_library_from_models(store, grid, lref, plan=None, dT=None, offset=0.0, edges=None, select=None,
                             cap_ok=True, cap=None, prof_dtype=np.float32, block=5000, nproc=1, unplanned="raise",
                             lines=None, nudged=NUDGED):
    """
    The flux library of node models: one bin per node, holding the mean (over the node's replicas) rest-frame profile
    on the velocity grid and the mean continuum flux (:meth:`ppmpy.synspec.library.FluxLibrary.build` with bins
    centred on the nodes).

    Parameters
    ----------
    store: ProfileStore, str or mapping
        The node models (:func:`ppmpy.synspec.fwresults.combine` of the library-mode run, or any profiles store /
        mapping with idx, teff, status, lam, fcont, fnorm, lines). Failed models (status != ok) are left out.
    grid: VelocityGrid or np.ndarray
        Velocity grid of the library.
    lref: LineSet or array-like
        Reference wavelengths of the store's lines (a LineSet's names must equal the store's lines).
    plan: NodePlan or str, optional
        The plan of the run (or its points.npz); recommended for a library-mode run. Bins = the planned nodes
        (``plan.edges``), every model in its planned node's bin: nodes without a usable model are empty bins (count 0;
        :func:`ppmpy.synspec.library.lib_nodes` skips them). Checked: every model is planned (``unplanned``), its
        T_eff' is the planned one plus a retry nudge (0 .. 10 K), and where it lies (``nudged``).
    dT, offset: float, optional
        Without a plan: nodes at offset + k dT, bins from the node of the coolest to that of the hottest usable model
        (node of T = floor((T - offset) / dT + 1/2)). A wrong offset would put the replicas of neighbouring nodes into
        one bin: a UserWarning when models lie at the bin edges (with the offset their T_eff' suggest).
    edges: array-like, optional
        Without a plan: explicit bin edges (spaced by dT; e.g. ``NodePlan.edges`` of a plan whose idx differ from the
        models', as in :func:`sparse_library_test`). Every usable model must lie inside them.
    select: array-like, optional
        Bool mask (N,) or row numbers (0 .. N - 1; ValueError otherwise) of the models to use (further restricted to
        the usable ones).
    cap_ok, cap:
        :meth:`ppmpy.synspec.fwresults.ProfileStore.usable` (default: models at the iteration cap are used).
    prof_dtype: dtype
        np.float32 (default, as the per-point libraries) or np.float64.
    block, nproc:
        :meth:`FluxLibrary.build` (the models are first gathered into compact arrays, so only their rows are read).
    unplanned: {'raise', 'skip'}
        Models of the store that the plan does not list.
    lines: sequence of str, optional
        Expected line names (checked against the store).
    nudged: {'keep', 'drop', 'raise'}
        With a plan: models whose retry nudges (+1 K per failed attempt) moved them across their bin edge. 'keep'
        (default, :data:`NUDGED`): binned with their planned node (the node's mean T_eff' includes the nudge; the
        node T_eff' must stay increasing); 'drop': left out; 'raise': ValueError. 'keep' and 'drop' warn.

    Returns
    -------
    FluxLibrary
        ``params`` as :meth:`FluxLibrary.build` plus mode 'library', offset, replicas (planned; None without a plan),
        n_nodes (bins), n_filled, missing_nodes (T_eff' of the empty bins), model_idx and model_bin (the models used
        and their bins; the intensity library takes its representatives from these), n_nudged_kept;
        ``inputs`` {'profiles': store path} where known.

    Raises
    ------
    ValueError
        No usable model, lines / lref mismatch, plan violations, models outside the given edges, bad ``select``.
    """
    # PP 2026-10-02: new (M7); nudged models, model_bin, offset check without a plan, select validation (reviewer)
    st = _as_store(store)
    names, lr = _lines_of(st, lref, lines)
    if st.teff is None:
        raise ValueError("the store has no 'teff' member")
    if unplanned not in ("raise", "skip"):
        raise ValueError("unplanned must be 'raise' or 'skip', got {!r}".format(unplanned))
    if nudged not in ("keep", "drop", "raise"):
        raise ValueError("nudged must be 'keep', 'drop' or 'raise', got {!r}".format(nudged))
    ok = _usable(st, select, cap_ok, cap)
    teff = np.asarray(st.teff, dtype=np.float64)
    bin_t = teff                                    # the T_eff' that decides a model's bin
    kept = np.zeros(st.n, bool)
    plan = _as_plan(plan)
    if plan is not None:
        pl, pos, moved = _plan_check(plan, st, ok, unplanned, nudged)
        ok &= pl
        if nudged == "drop":
            ok &= ~moved
        elif moved.any():
            kept = moved & ok
            bin_t = teff.copy()
            bin_t[kept] = plan.teff[pos[kept]]      # the planned T_eff': inside the planned node's bin
        dT, e = plan.dT, plan.edges
        offset = plan.offset
    else:
        if dT is None:
            raise ValueError("give a plan, or dT (and offset or edges)")
        dT = _float(dT, "dT", positive=True)
        offset = _float(offset, "offset")
        if not ok.any():
            raise ValueError("no usable models (status ok) in the store")
        if edges is not None:
            e = np.asarray(edges, dtype=np.float64).reshape(-1)
            if e.size < 2 or not np.allclose(np.diff(e), dT, rtol=1e-9, atol=0.0):
                raise ValueError("edges must have >= 2 entries spaced by dT = {:g}".format(dT))
        else:
            _offset_check(teff[ok], dT, offset)
            k = np.floor((teff[ok] - offset) / dT + 0.5)
            e = offset + dT * (np.arange(k.min(), k.max() + 2) - 0.5)
            while teff[ok].min() < e[0]:                 # rounding at an exact edge
                e = np.concatenate([[e[0] - dT], e])
            while teff[ok].max() >= e[-1]:
                e = np.concatenate([e, [e[-1] + dT]])
    if not ok.any():
        raise ValueError("no usable models (status ok) in the store")
    out = ok & ((bin_t < e[0]) | (bin_t >= e[-1]))
    if out.any():
        raise ValueError("{} usable models lie outside the bins {:g}..{:g} K (idx {})".format(
            int(out.sum()), e[0], e[-1], np.asarray(st.idx)[out][:10].tolist()))
    rows = np.flatnonzero(ok)
    sub = st if rows.size == st.n else _substore(st, rows)
    lam, fnorm = sub["lam"], sub["fnorm"]
    fcont0 = np.asarray(sub["fcont"][:, :, 0])
    lib = FluxLibrary.build(bin_t[rows], lam, fnorm, fcont0, grid, lr, dT=dT, edges=e, prof_dtype=prof_dtype,
                            block=block, nproc=nproc)
    mbin = teff_bins(bin_t[rows], e)
    kr = kept[rows]
    if kr.any():                                    # the bins of kept models: mean of the real T_eff'
        tsum = np.bincount(mbin, weights=teff[rows], minlength=lib.nb)
        b = np.unique(mbin[kr])
        lib.tmean[b] = tsum[b] / lib.count[b]
        if np.any(np.diff(lib.tmean[lib.filled]) <= 0):
            raise ValueError("models nudged across their bin edge make the node T_eff' non-increasing (idx {}): use "
                             "nudged='drop'".format(np.asarray(sub.idx)[kr][:10].tolist()))
    empty = ~lib.filled
    lib.params.update(mode="library", offset=float(offset), replicas=None if plan is None else plan.replicas,
                      lines=list(names), n_nodes=int(lib.nb), n_filled=int(lib.filled.sum()),
                      missing_nodes=[float(x) for x in lib.centres[empty]],
                      model_idx=[int(i) for i in np.asarray(sub.idx)], model_bin=[int(b) for b in mbin],
                      n_nudged_kept=int(kr.sum()), plan=None if plan is None else plan.params())
    if st.path is not None:
        lib.inputs = dict(profiles=st.path)
    return lib


def library_nodes(flux):
    """The interpolation nodes of a library-mode flux library: ``lib_nodes(flux, nmin=1)`` (every filled bin is a
    node; see the module notes on nmin), with the loaded BLAS limited to 1 thread (as dumps.flux_integrator)."""
    from .dumps import _blas_limit
    with _blas_limit(1):
        return lib_nodes(flux, nmin=1)


def _imu_names(names, suffix, check):
    need = ["OUT_IMU.{}_{}".format(n, suffix) for n in names]
    if check:
        need += ["OUT.{}_{}".format(n, suffix) for n in names]
    return need


def node_representatives(flux, candidates, allow_missing=False):
    """
    The representative model of every filled node of a library-mode flux library: among the node's candidates, the
    model closest to the node's mean T_eff' (``flux['tmean']``; ties: the smaller idx).

    Parameters
    ----------
    flux: FluxLibrary or mapping
        The node library (edges, tmean, count). When its ``params`` record model_idx and model_bin
        (:func:`flux_library_from_models`), only those models are candidates, each for the node the flux library put
        it in (so the flux and intensity libraries use the same models, e.g. unplanned models left out); otherwise a
        candidate's node is the bin of its T_eff' (:func:`ppmpy.synspec.library.teff_bins`).
    candidates: sequence of tuple
        (idx, teff, model_dir) per model with OUT_IMU files; (idx, teff) gives empty dirs.
    allow_missing: bool
        Filled nodes without an eligible candidate are listed in ``missing`` instead of raising.

    Returns
    -------
    Representatives
        One per filled node with a candidate, nodes ascending.

    Raises
    ------
    ValueError
        Filled nodes without a candidate (unless allow_missing), a non-finite candidate T_eff'.

    Notes
    -----
    A candidate must also lie inside its node's bin by its own T_eff' (:func:`ppmpy.synspec.library.build_imu_library`
    requires it): a replica nudged across the bin edge (kept in its node by the flux library, nudged='keep') is not
    a candidate. Replicas are not averaged: one model per node, as the production intensity library has one model
    per 10 K bin. With an even number of replicas spread over the bin ('bin'), no replica sits at the node: the two
    middle replicas are equally close to the mean and the smaller idx (the lower replica) is taken.
    """
    # PP 2026-10-02: new (reviewer: the intensity library must use the flux library's models and bins; replaces
    # library.select_representatives here, whose bins come from teff_bins and ties from the candidate order)
    edges = np.asarray(flux["edges"], dtype=np.float64)
    tmean = np.asarray(flux["tmean"], dtype=np.float64)
    count = np.asarray(flux["count"])
    nb = tmean.size
    cands = list(candidates)
    idx_c = np.array([int(c[0]) for c in cands], dtype=np.int64)
    t_c = np.array([float(c[1]) for c in cands], dtype=np.float64)
    if not np.all(np.isfinite(t_c)):
        raise ValueError("candidate T_eff' must be finite")
    filled = np.flatnonzero(count > 0)
    sel_b = sel_k = np.zeros(0, np.int64)
    if t_c.size:
        b_t = teff_bins(t_c, edges)
        ok = (t_c >= edges[0]) & (t_c < edges[-1])
        params = getattr(flux, "params", None) or {}
        mi, mb = params.get("model_idx"), params.get("model_bin")
        if mi is not None and mb is not None and len(mi) == len(mb):
            lut = dict(zip((int(i) for i in mi), (int(x) for x in mb)))
            b_c = np.array([lut.get(int(i), -1) for i in idx_c], dtype=np.int64)
            ok &= (b_c >= 0) & (b_c == b_t)
        else:
            b_c = b_t
        ok &= count[np.clip(b_c, 0, nb - 1)] > 0
        use = np.flatnonzero(ok)
        if use.size:
            bu = b_c[use]
            order = np.lexsort((idx_c[use], np.abs(t_c[use] - tmean[bu]), bu))
            first = order[np.r_[True, np.diff(bu[order]) != 0]]
            sel_b, sel_k = bu[first].astype(np.int64), use[first]
    missing = [int(b) for b in np.setdiff1d(filled, sel_b)]
    if missing and not allow_missing:
        raise ValueError("{} of {} filled nodes have no candidate model with OUT_IMU files inside their bin (bins "
                         "{}{}; a replica nudged across its bin edge cannot represent its node): allow_missing=True "
                         "treats them as empty".format(len(missing), filled.size, missing[:20],
                                                " ..." if len(missing) > 20 else ""))
    dirs = [cands[k][2] if len(cands[k]) > 2 else "" for k in sel_k]
    return Representatives(sel_b, idx_c[sel_k], t_c[sel_k], dirs, missing=missing, n_candidates=len(cands))


def imu_library_from_models(flux, store, grid=None, lref=None, imu_runs=None, layout=IMU_LAYOUT, imu_dirs=None,
                            results_dir=None, extract_dir=None, suffix="VTV010", check=True, allow_missing=False,
                            select=None, cap_ok=True, cap=None, lines=None, nproc=1, log=None):
    """
    The intensity library of node models: per node, the emergent intensities of one representative model (the
    model closest to the node's mean T_eff'; :func:`node_representatives`) from its OUT_IMU files
    (:func:`ppmpy.synspec.library.build_imu_library`).

    Replicas improve only the flux library: the intensity library stays one model per node (the replicas' Ic / Il
    are not averaged), as the production intensity library has one model per 10 K bin. A library-mode run with R
    replicas therefore pays R models per node for the flux method alone (a UserWarning says so), and the flux and
    intensity libraries of such a run differ in kind (replica mean vs single model). For the intensity method, one
    model per node (R = 1) at dT = 10 K is a library of the production kind.

    Parameters
    ----------
    flux: FluxLibrary
        The node library of the same models (:func:`flux_library_from_models`): its bins and mean T_eff', and (its
        params model_idx, model_bin) the models it used: only those are candidates, in the node it put them in.
    store: ProfileStore, str or mapping
        The node models (idx, teff, status); T_eff' are taken from here.
    grid: VelocityGrid, optional
        Velocity grid (default the M424 grid; must be the flux library's).
    lref: LineSet or array-like
        Reference wavelengths of the store's lines.
    imu_runs, layout: str, optional
        OUT_IMU (and OUT) files of model idx in ``imu_runs/layout.format(idx=idx)`` (M424 pformalsol reruns:
        /scratch/ppathak/fastwind_imu/runs with :data:`ppmpy.synspec.library.IMU_LAYOUT`).
    imu_dirs: str or sequence of str, optional
        Directories of extracted model directories ``P<idx>/`` with meta.txt and the OUT_IMU files
        (:func:`ppmpy.synspec.library.find_candidates`).
    results_dir, extract_dir: str, optional
        The packed results of the library-mode run (``RUN_DIR/results``, run with the intensity formal build): the
        representatives' OUT_IMU, OUT and meta.txt files are extracted to ``extract_dir/P<idx>/``
        (:func:`ppmpy.synspec.fwresults.extract_points`; present directories are kept) and read from there.
    suffix: str
        OUT file suffix.
    check: bool
        Flux check of :func:`build_imu_library` (the OUT files must be there too).
    allow_missing: bool
        Filled bins without a model with OUT_IMU files are treated like empty bins (default: ValueError).
    select, cap_ok, cap, lines:
        As :func:`flux_library_from_models`.
    nproc: int
        Workers of the extraction.
    log: callable, optional

    Returns
    -------
    imu: ImuLibrary
    checks: dict
        Of :func:`build_imu_library`.
    reps: Representatives
        The representative of every node (directories with the OUT_IMU files).

    Raises
    ------
    ValueError
        No source given, no model with OUT_IMU files (run FASTWIND with the intensity formal build, ``--formal-build
        v10.6_HHe_imu``, or rerun pformalsol: ``python3 -m ppmpy.synspec.fastwind rerun-formal``), filled bins
        without one (unless allow_missing), T_eff' of meta.txt differing from the store's.

    Warns
    -----
    UserWarning
        Nodes with several models (replicas): one representative each.
    """
    # PP 2026-10-02: new (M7); candidates = the flux library's models in its bins, replica warning (reviewer)
    st = _as_store(store)
    names, lr = _lines_of(st, lref, lines)
    grid = VelocityGrid() if grid is None else grid
    ok = _usable(st, select, cap_ok, cap)
    teff = np.asarray(st.teff, dtype=np.float64)
    idx = np.asarray(st.idx, dtype=np.int64)
    e = np.asarray(flux["edges"])
    ok &= (teff >= e[0]) & (teff < e[-1])
    mi = (getattr(flux, "params", None) or {}).get("model_idx")
    if mi is not None:
        ok &= np.isin(idx, np.asarray(mi, dtype=np.int64))
    cnt = np.asarray(flux["count"])
    if np.any(cnt > 1):
        warnings.warn("{} of {} nodes hold several models (up to {}): the intensity library keeps one representative "
                      "per node (closest to the node's mean T_eff'); replicas average only the flux library".format(
                          int((cnt > 1).sum()), int((cnt > 0).sum()), int(cnt.max())), stacklevel=2)
    sources = sum(x is not None for x in (imu_runs, imu_dirs, results_dir))
    if sources != 1:
        raise ValueError("give exactly one source of OUT_IMU files: imu_runs, imu_dirs or results_dir (+ extract_dir)")
    need = _imu_names(names, suffix, check)
    if results_dir is not None:
        if extract_dir is None:
            raise ValueError("results_dir needs extract_dir (where the representatives are extracted)")
        pre = node_representatives(flux, [(int(i), float(t), "") for i, t in zip(idx[ok], teff[ok])],
                                   allow_missing=allow_missing)
        from .fwresults import extract_points
        extract_points(results_dir, [int(i) for i in pre.idx], extract_dir, members=["OUT_IMU.*", "OUT.*"],
                       nproc=nproc, log=log)
        imu_runs, layout = extract_dir, POINT_DIR
    if imu_runs is not None:
        cands = []
        for i, t in zip(idx[ok], teff[ok]):
            d = os.path.join(os.fspath(imu_runs), layout.format(idx=int(i)))
            if all(os.path.exists(os.path.join(d, f)) for f in need):
                cands.append((int(i), float(t), d))
    else:
        known = dict(zip(idx[ok].tolist(), teff[ok].tolist()))
        cands = []
        for i, t, d in find_candidates(imu_dirs, require=("meta.txt",) + tuple(need)):
            if i in known:
                if abs(t - known[i]) > _TEFF_TOL:
                    raise ValueError("{}: meta.txt T_eff' {} differs from the store's {}".format(d, t, known[i]))
                cands.append((i, known[i], d))
    if not cands:
        raise ValueError("no node model has {} files: run FASTWIND with the intensity formal build (--formal-build "
                         "v10.6_HHe_imu) or rerun pformalsol (python3 -m ppmpy.synspec.fastwind rerun-formal)".format(
                             "/".join(sorted({f.split(".")[0] for f in need}))))
    reps = node_representatives(flux, cands, allow_missing=allow_missing)
    imu, checks = build_imu_library(reps, flux, names, grid, lref=lr, runs_dir=None, suffix=suffix, check=check,
                                    allow_missing=allow_missing)
    imu.params.update(mode="library", lines=list(names), replicas_averaged=False,
                      max_models_per_node=int(cnt.max()) if cnt.size else 0)
    return imu, checks, reps


class NodeLibrary:
    """
    The libraries of a library-mode run (:func:`library_from_models`).

    Attributes
    ----------
    flux: FluxLibrary
        One bin per planned node (replica means).
    nodes: LibraryNodes
        ``lib_nodes(flux, nmin=1)``: every filled bin is an interpolation node.
    imu: ImuLibrary or None
        The intensity library of the same models (one representative per node; replicas are not averaged), when
        built.
    imu_checks: dict or None
        Checks of :func:`ppmpy.synspec.library.build_imu_library`.
    representatives: Representatives or None
    plan: NodePlan or None
    grid: VelocityGrid
    lines: LineSet
    report: dict
        n_planned (nodes), n_filled, missing_nodes (T_eff' of nodes without a usable model), node_range, models
        (used), failed (idx, teff, status of models that are not usable), replicas (models per filled node: min,
        max), imu (built or not).
    """

    def __init__(self, flux, nodes, grid, lines, imu=None, imu_checks=None, representatives=None, plan=None,
                 report=None):
        self.flux, self.nodes, self.grid, self.lines = flux, nodes, grid, lines
        self.imu, self.imu_checks, self.representatives, self.plan = imu, imu_checks, representatives, plan
        self.report = dict(report or {})

    def __repr__(self):
        return "NodeLibrary({} nodes {:.0f}-{:.0f} K of {} planned, {} models, imu {})".format(
            self.nodes.nn, self.nodes.t[0], self.nodes.t[-1], self.report.get("n_planned"),
            self.report.get("models"), "yes" if self.imu is not None else "no")

    def flux_integrator(self, grid=None, pad_tol=None):
        """
        :class:`ppmpy.synspec.disc.DiscFlux` of the nodes (:func:`ppmpy.synspec.dumps.flux_integrator` with nmin=1;
        lref attached from the library).
        """
        from .dumps import PAD_TOL, flux_integrator
        return flux_integrator(self.flux, nmin=1, grid=self.grid if grid is None else grid,
                               pad_tol=PAD_TOL if pad_tol is None else pad_tol)

    def imu_integrator(self, **kw):
        """:class:`ppmpy.synspec.disc.DiscImu` of the intensity library (:func:`ppmpy.synspec.dumps.imu_integrator`;
        keywords: fft, dtype, lines, chunk)."""
        if self.imu is None:
            raise ValueError("no intensity library (the node models have no OUT_IMU files, or imu=False)")
        from .dumps import imu_integrator
        kw.setdefault("lref", self.lines)
        return imu_integrator(self.imu, **kw)

    def save(self, outdir, meta=None):
        """
        Write the libraries for :func:`ppmpy.synspec.dumps.run_disc_dumps` (file names keep 'spawn' workers cheap):
        ``outdir/`` :data:`LIBRARY_FILE` (FluxLibrary.save), :data:`IMU_LIBRARY_FILE` (ImuLibrary.save, when built),
        :data:`REPRESENTATIVES_FILE` and :data:`REPORT_FILE` (the report and the plan, JSON).

        Returns
        -------
        dict
            name -> path of the files written ('flux', 'imu', 'representatives', 'report').
        """
        # PP 2026-10-02: new (M7)
        outdir = os.fspath(outdir)
        os.makedirs(outdir, exist_ok=True)
        out = dict(flux=self.flux.save(os.path.join(outdir, LIBRARY_FILE), meta=meta))
        if self.imu is not None:
            out["imu"] = self.imu.save(os.path.join(outdir, IMU_LIBRARY_FILE), meta=meta)
        if self.representatives is not None:
            out["representatives"] = self.representatives.write(os.path.join(outdir, REPRESENTATIVES_FILE))
        rec = dict(report=self.report, plan=None if self.plan is None else self.plan.params(),
                   grid=self.grid.to_dict(), lines=self.lines.to_dict(), nmin=1)
        path = os.path.join(outdir, REPORT_FILE)
        tmp = "{}.tmp{}".format(path, os.getpid())
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=1, sort_keys=True, default=_json_default)
        os.replace(tmp, path)
        out["report"] = path
        return out


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def library_from_models(store, grid, lref, plan=None, dT=None, offset=0.0, edges=None, imu=None, imu_runs=None,
                        imu_layout=IMU_LAYOUT, imu_dirs=None, results_dir=None, extract_dir=None, suffix="VTV010",
                        check=True, allow_missing=False, select=None, cap_ok=True, cap=None, prof_dtype=np.float32,
                        unplanned="raise", lines=None, nproc=1, log=None, nudged=NUDGED):
    """
    The libraries of a library-mode run: flux library with one bin per node (replica mean), its nodes with
    ``lib_nodes(nmin=1)`` (nmin must not merge nodes in library mode: module notes), and the intensity library of
    the same models when OUT_IMU files exist.

    Replicas improve only the flux library: the intensity library keeps one representative per node (the model
    closest to the node's mean T_eff'; :func:`node_representatives`), as the production intensity library keeps one
    model per 10 K bin. A run with R > 1 replicas therefore gets the replica averaging in the flux method only
    (UserWarning when the intensity library is built), and its flux and intensity libraries differ in kind.

    Parameters
    ----------
    store: ProfileStore, str or mapping
        The node models: profiles.npz of :func:`ppmpy.synspec.fwresults.combine` of the library-mode run.
    grid: VelocityGrid
        Velocity grid of the libraries.
    lref: LineSet or array-like
        Reference wavelengths of the store's lines (a LineSet also names them; names must equal the store's).
    plan, dT, offset, edges:
        The nodes (:func:`flux_library_from_models`): the run's :class:`NodePlan` (or its points.npz), else dT and
        offset (or edges).
    imu: bool, optional
        True: build the intensity library (a source must be given); False: do not; None (default): build it when a
        source is given (imu_runs, imu_dirs or results_dir + extract_dir; :func:`imu_library_from_models`).
    imu_runs, imu_layout, imu_dirs, results_dir, extract_dir, suffix, check, allow_missing:
        :func:`imu_library_from_models` (check: its flux check).
    select, cap_ok, cap, prof_dtype, unplanned, lines, nproc, nudged:
        :func:`flux_library_from_models` (the intensity library uses the models the flux library used, in its bins).
    log: callable, optional

    Returns
    -------
    NodeLibrary

    Warns
    -----
    UserWarning
        Planned nodes without a usable model (they are skipped: the neighbours interpolate across them); replicas
        with an intensity library; nudged models (``nudged``).
    """
    # PP 2026-10-02: new (M7); nudged, intensity library from the flux library's models (reviewer)
    T0 = time.time()
    _log = _logger(log, T0)
    st = _as_store(store)
    names, lr = _lines_of(st, lref, lines)
    lineset = lref if isinstance(lref, LineSet) else LineSet(names, lr)
    grid = VelocityGrid() if grid is None else grid
    plan = _as_plan(plan)
    flux = flux_library_from_models(st, grid, lineset, plan=plan, dT=dT, offset=offset, edges=edges, select=select,
                                    cap_ok=cap_ok, cap=cap, prof_dtype=prof_dtype, unplanned=unplanned, nproc=nproc,
                                    nudged=nudged)
    nodes = library_nodes(flux)
    _log("flux library: {} of {} nodes filled, {} models, nodes {:.1f}-{:.1f} K".format(
        int(flux.filled.sum()), flux.nb, int(flux.count.sum()), nodes.t[0], nodes.t[-1]))
    bad = np.flatnonzero(~st.usable(cap_ok=cap_ok, cap=cap))          # failed (or capped, cap_ok=False) models
    st_status = np.asarray(st.status) if st.status is not None else np.full(st.n, "ok")
    failed = [dict(idx=int(np.asarray(st.idx)[i]), teff=float(np.asarray(st.teff)[i]), status=str(st_status[i]))
              for i in bad]
    cnt = flux.count[flux.filled]
    report = dict(n_planned=int(flux.nb), n_filled=int(flux.filled.sum()),
                  missing_nodes=list(flux.params["missing_nodes"]),
                  node_range=[float(nodes.t[0]), float(nodes.t[-1])], models=int(flux.count.sum()),
                  failed=failed, replicas=[int(cnt.min()), int(cnt.max())], dT=float(flux.dT),
                  n_nudged_kept=int(flux.params.get("n_nudged_kept", 0)), imu=False)
    if report["missing_nodes"]:
        warnings.warn("{} planned nodes have no usable model ({}): skipped, their neighbours interpolate across "
                      "them (rerun missing.txt to fill them)".format(len(report["missing_nodes"]),
                                                                    report["missing_nodes"][:10]))
    has_src = any(x is not None for x in (imu_runs, imu_dirs, results_dir))
    if imu and not has_src:
        raise ValueError("imu=True needs imu_runs, imu_dirs or results_dir + extract_dir")
    im = chk = reps = None
    if has_src and imu is not False:
        im, chk, reps = imu_library_from_models(flux, st, grid=grid, lref=lineset, imu_runs=imu_runs,
                                                layout=imu_layout, imu_dirs=imu_dirs, results_dir=results_dir,
                                                extract_dir=extract_dir, suffix=suffix, check=check,
                                                allow_missing=allow_missing, select=select, cap_ok=cap_ok, cap=cap,
                                                nproc=nproc, log=log)
        report["imu"] = True
        report["imu_summary"] = chk.get("summary")
        report["imu_nodes"] = int(np.unique(im.src).size)
        report["imu_replicas_averaged"] = False
        _log("intensity library: {} representatives, K = {}".format(len(reps), im.K))
    return NodeLibrary(flux, nodes, grid, lineset, imu=im, imu_checks=chk, representatives=reps, plan=plan,
                       report=report)


# ----------------------------------------------------------------------------------------------------------------
# V8: sparse libraries from existing per-point models
# ----------------------------------------------------------------------------------------------------------------
class SparseLibraryTest:
    """
    The V8 table (:func:`sparse_library_test`).

    Attributes
    ----------
    rows: list of dict
        One per variant and node phase: method ('flux' / 'imu'), label ('dT<dT>_R<R>', with 's<step>' for spread
        replicas and '_o<offset>' when several node phases are tested; 'reference'), base (the label without the
        phase), dT [K] (None for the reference), replicas, replica_step (None: the R models closest to the node),
        offset [K, mod dT] and phase (fraction of dT) of the nodes, n_phases, check (True: the selection is every
        model of the pool, e.g. the production intensity representatives: the pool's own library, a check of the
        comparison, not a sparse library; never recommended), same_as (an earlier variant with the same node models
        and grouping, i.e. the same integrator, whose comparison is reused), n_planned, n_filled (nodes with models),
        models (FASTWIND models a library-mode run needs: planned nodes x replicas), pool_models (models of the
        per-point run used), max_dist (largest |T_eff' - target| of a chosen model), node_range (planned),
        filled_range (node T_eff' of the integrator), pool_range (T_eff' range of the pool), max_dF, max_dF0 (nl,)
        (max |F - F_ref| over lines of sight, dumps and the whole grid; F0: no Doppler shifts), ratio_F, ratio_F0
        (nl,) (/ the LPV residual rms), dR_rms, dR_max (nl,) (the time-variable part: residuals about the subset mean,
        |y| <= vwin, rms and max), ratio_dR, ratio_dR_max (nl,) (/ the LPV residual rms), max_dEW (nl,) [A],
        dEW_t_rms (nl,) (EW about the subset mean), ratio_EW (nl,) (dEW_t_rms / the LPV EW rms), n_lo, n_hi (most
        points of a dump below / above the nodes), wall [s].
        With several phases, an aggregate row follows the phase rows of each variant: aggregate True, label = base,
        phases, offsets, phase_labels, phase_check, n_checks, check (all phases checks); every per-line array is the
        largest over the phases that are not checks (the worst node placement), with ``<key>_min`` and
        ``<key>_mean``; phase_max {ratio key: per phase, the largest over the lines}.
    lines: list of str
    dumps: list of int
    lpv: dict
        The LPV scales per method (:func:`ppmpy.synspec.validate.lpv_residual_rms`).
    params: dict
    recommend: dict
        Per method: the variant with the fewest models whose criterion ratio (over the phases: ``phase_stat``, 'max'
        = the worst phase) stays <= warn, with ratio, margin (warn - ratio) and ratio_worst; None if none. Check rows,
        the reference, per-phase rows of an aggregated variant, and intensity-method rows with replicas != 1 (the
        intensity library does not average replicas) are never recommended.
    """

    def __init__(self, rows, lines, dumps, lpv, params, recommend):
        self.rows, self.lines, self.dumps = list(rows), list(lines), list(dumps)
        self.lpv, self.params, self.recommend = dict(lpv), dict(params), dict(recommend)

    def __repr__(self):
        return "SparseLibraryTest({} variants, {} dumps, recommend {})".format(
            len(self.rows), len(self.dumps), {k: (v or {}).get("label") for k, v in self.recommend.items()})

    def row(self, label, method="flux"):
        """The row of a variant by label (e.g. 'dT10_R1': with several phases, the aggregate row; 'dT10_R1_o5': one
        phase)."""
        for r in self.rows:
            if r["label"] == label and r["method"] == method:
                return r
        raise KeyError((method, label))

    def table(self, detail=True):
        """
        The table as text: per variant the nodes, models, the largest ratio over the lines of max|dF|, max|dF0|,
        max|dR|, rms(dR) to the LPV residual rms and of the EW deviation to the LPV EW rms, and the per-line max|dF|.
        An aggregate row shows the range over its phases (min-max over the phases that are not checks of the
        largest ratio over the lines). '(check)' marks check rows, '= X' rows that reuse variant X's comparison.

        Parameters
        ----------
        detail: bool
            Also the per-phase rows of aggregated variants (default).
        """
        hdr = ("method", "variant", "nodes", "models", "max|dF|/LPV", "max|dF0|/LPV", "max|dR|/LPV", "dR_rms/LPV",
               "dEW_t/EWrms", "max|dF| per line")
        out = [hdr]
        for r in self.rows:
            if not detail and r.get("n_phases", 1) > 1 and not r.get("aggregate"):
                continue
            name = ("  " if r.get("n_phases", 1) > 1 and not r.get("aggregate") else "") + r["label"]
            if r.get("check"):
                name += " (check)"
            if r.get("same_as"):
                name += " = " + r["same_as"]
            nodes = "{}/{}".format(r["n_filled"], r["n_planned"])
            out.append((r["method"], name, nodes, str(r["models"]), _cell(r, "ratio_F"), _cell(r, "ratio_F0"),
                        _cell(r, "ratio_dR_max"), _cell(r, "ratio_dR"), _cell(r, "ratio_EW"),
                        " / ".join("{:.2e}".format(x) for x in r["max_dF"])))
        w = [max(len(row[i]) for row in out) for i in range(len(hdr))]
        lines = ["  ".join(row[i].ljust(w[i]) for i in range(len(hdr))).rstrip() for row in out]
        lines.insert(1, "-" * len(lines[0]))
        for m, rec in sorted(self.recommend.items()):
            lines.append("recommended ({}, {} <= {:.0%}{}): {}".format(
                m, self.params.get("criterion"), self.params.get("warn"),
                "" if (rec or {}).get("phase_stat") is None else ", {} over {} phases".format(
                    rec["phase_stat"], rec.get("n_phases")),
                "none" if rec is None else "{} ({} models; ratio {:.2%}, margin {:.2%})".format(
                    rec["label"], rec["models"], rec["ratio"], rec.get("margin", np.nan))))
        return "\n".join(lines)

    def as_dict(self):
        return dict(rows=self.rows, lines=self.lines, dumps=self.dumps, lpv=self.lpv, params=self.params,
                    recommend=self.recommend)

    def to_json(self, path):
        """Write the table (JSON) atomically; returns path."""
        path = os.fspath(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = "{}.tmp{}".format(path, os.getpid())
        with open(tmp, "w") as f:
            json.dump(self.as_dict(), f, indent=1, default=_json_default)
        os.replace(tmp, path)
        return path

    @classmethod
    def from_json(cls, path):
        with open(os.fspath(path)) as f:
            d = json.load(f)
        return cls(d["rows"], d["lines"], d["dumps"], d["lpv"], d["params"], d["recommend"])

    def to_report(self):
        """
        A :class:`ppmpy.synspec.validate.ValidationReport` with one informational check per variant
        ('V8_<method>_<label>': value = max|dF| over the lines, per-line values), compared with the LPV
        (:func:`ppmpy.synspec.validate.compare_lpv`: lpv_ratio per line).
        """
        from .validate import CheckResult, ValidationReport, compare_lpv
        reports = []
        for m in sorted({r["method"] for r in self.rows}):
            checks = [CheckResult("V8_{}_{}".format(r["method"], r["label"]), float(np.max(r["max_dF"])),
                                  details=dict(per_line=np.asarray(r["max_dF"]), lines=self.lines, dumps=self.dumps,
                                               models=r["models"], dT=r["dT"], replicas=r["replicas"],
                                               check=bool(r.get("check")), aggregate=bool(r.get("aggregate"))))
                      for r in self.rows if r["method"] == m]
            rep = ValidationReport(checks, meta=dict(kind="synspec.libmode.v8", params=self.params))
            lpv = self.lpv.get(m)
            if lpv is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    compare_lpv(rep, {k: (None if v is None else np.asarray(v)) for k, v in lpv.items()
                                      if k in ("rms", "max", "ew_rms", "ew_rel_rms")}, warn=self.params["warn"])
            reports.append(rep)
        return ValidationReport.merge(*reports, meta=dict(kind="synspec.libmode.v8", params=self.params))


def _pct(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.any(np.isfinite(x)):
        return "-"
    return "{:.2%}".format(float(np.nanmax(x)))


def _cell(r, key):
    """A table cell: the largest ratio over the lines; for an aggregate row its range over the non-check phases."""
    if not r.get("aggregate"):
        return _pct(r.get(key, []))
    pm = r.get("phase_max", {}).get(key, [])
    v = np.asarray([x for x, c in zip(pm, r.get("phase_check", [])) if not c] or pm, dtype=np.float64)
    if v.size == 0 or not np.any(np.isfinite(v)):
        return "-"
    return "{:.2%}-{:.2%}".format(float(np.nanmin(v)), float(np.nanmax(v)))


def _reference_dump(reference, d):
    """F, F0, diag_F (float64) and diag_keys of the reference product of dump d."""
    if isinstance(reference, (str, os.PathLike)):
        from .dumps import DUMP_PATTERN
        with np.load(os.path.join(os.fspath(reference), DUMP_PATTERN.format(dump=int(d)))) as z:
            return dict(F=z["F"].astype(np.float64), F0=z["F0"].astype(np.float64),
                        diag_F=z["diag_F"].astype(np.float64), diag_keys=[str(k) for k in z["diag_keys"]])
    r = reference(d) if callable(reference) else reference[d]
    return dict(F=np.asarray(r["F"], dtype=np.float64), F0=np.asarray(r["F0"], dtype=np.float64),
                diag_F=None if r.get("diag_F") is None else np.asarray(r["diag_F"], dtype=np.float64),
                diag_keys=[str(k) for k in r.get("diag_keys", ())])


def _lpv_of(lpv, vwin, log):
    if lpv is None:
        return None
    if isinstance(lpv, (str, os.PathLike)):
        from .validate import lpv_residual_rms
        t = time.time()
        out = lpv_residual_rms(os.fspath(lpv), vwin=vwin)
        if log is not None:
            log("LPV of {}: {:.0f} s".format(os.fspath(lpv), time.time() - t))
        return out
    return dict(lpv)


class _Acc:
    """Running comparison of one variant with the reference over the dumps (alias: an earlier variant with the same
    integrator, whose comparison is reused)."""

    def __init__(self, method, label, dT, R, integ, info, nl, step=None, alias=None):
        self.method, self.label, self.dT, self.R, self.integ, self.info = method, label, dT, R, integ, info
        self.step, self.alias = step, alias
        self.maxF, self.maxF0, self.maxEW = np.zeros(nl), np.zeros(nl), np.zeros(nl)
        self.Fw, self.ew = [], []
        self.n_lo = self.n_hi = 0
        self.wall = 0.0

    def add(self, r, ref, m, kew):
        F, F0 = r["F"], r["F0"]
        self.maxF = np.maximum(self.maxF, np.abs(F - ref["F"]).max(axis=(0, 2)))
        self.maxF0 = np.maximum(self.maxF0, np.abs(F0 - ref["F0"]).max(axis=(0, 2)))
        self.Fw.append(F[..., m])
        if kew is not None and ref["diag_F"] is not None:
            e = r["diag_F"][..., kew]
            self.ew.append(e)
            self.maxEW = np.maximum(self.maxEW, np.abs(e - ref["diag_F"][..., kew]).max(axis=0))
        self.n_lo, self.n_hi = max(self.n_lo, int(r["n_lo"])), max(self.n_hi, int(r["n_hi"]))


def _finish(acc, Rref, EWref, lpv):
    """The row of a variant: the time-variable comparison and the LPV ratios."""
    if acc.alias is not None:
        row = _finish(acc.alias, Rref, EWref, lpv)
        row.update(method=acc.method, label=acc.label, dT=acc.dT, replicas=acc.R, replica_step=acc.step,
                   wall=acc.wall, same_as=acc.alias.label)
        row.update(acc.info)
        return row
    nl = acc.maxF.size
    A = np.array(acc.Fw)                                             # (nd, nlos, nl, nw)
    dR = (A - A.mean(axis=0)) - Rref
    dR_rms = np.sqrt((dR ** 2).mean(axis=(0, 1, 3)))
    dR_max = np.abs(dR).max(axis=(0, 1, 3))
    if acc.ew and EWref is not None:
        E = np.array(acc.ew)                                         # (nd, nlos, nl)
        dE = (E - E.mean(axis=0)) - EWref
        dEW_t = np.sqrt((dE ** 2).mean(axis=(0, 1)))
    else:
        dEW_t = np.full(nl, np.nan)
    rms = np.full(nl, np.nan) if lpv is None else np.asarray(lpv["rms"], dtype=np.float64)
    ewr = (np.full(nl, np.nan) if lpv is None or lpv.get("ew_rms") is None
           else np.asarray(lpv["ew_rms"], dtype=np.float64))
    row = dict(method=acc.method, label=acc.label, dT=acc.dT, replicas=acc.R, replica_step=acc.step,
               max_dF=acc.maxF.tolist(),
               max_dF0=acc.maxF0.tolist(), ratio_F=(acc.maxF / rms).tolist(), ratio_F0=(acc.maxF0 / rms).tolist(),
               dR_rms=dR_rms.tolist(), dR_max=dR_max.tolist(), ratio_dR=(dR_rms / rms).tolist(),
               ratio_dR_max=(dR_max / rms).tolist(),
               max_dEW=acc.maxEW.tolist(), dEW_t_rms=dEW_t.tolist(), ratio_EW=(dEW_t / ewr).tolist(),
               n_lo=acc.n_lo, n_hi=acc.n_hi, wall=acc.wall, same_as=None)
    row.update(acc.info)
    return row


_ROW_ARRAYS = ("max_dF", "max_dF0", "ratio_F", "ratio_F0", "dR_rms", "dR_max", "ratio_dR", "ratio_dR_max", "max_dEW",
               "dEW_t_rms", "ratio_EW")
"""The per-line arrays of a V8 row (aggregated over the phases)."""
_RATIO_KEYS = ("ratio_F", "ratio_F0", "ratio_dR_max", "ratio_dR", "ratio_EW")


def _aggregate(prs):
    """The aggregate row of one variant's phase rows: per-line largest, smallest and mean over the phases that are
    not checks (all phases if every one is a check)."""
    use = [r for r in prs if not r.get("check")] or prs
    r0 = prs[0]
    fin = [r["max_dist"] for r in prs if r.get("max_dist") is not None]
    agg = dict(method=r0["method"], label=r0["base"], base=r0["base"], dT=r0["dT"], replicas=r0["replicas"],
               replica_step=r0.get("replica_step"), aggregate=True, n_phases=len(prs),
               phases=[r["phase"] for r in prs], offsets=[r["offset"] for r in prs],
               phase_labels=[r["label"] for r in prs], phase_check=[bool(r.get("check")) for r in prs],
               n_checks=int(sum(bool(r.get("check")) for r in prs)), check=all(bool(r.get("check")) for r in prs),
               same_as=None, models=max(r["models"] for r in prs), n_planned=max(r["n_planned"] for r in prs),
               n_filled=max(r["n_filled"] for r in prs), pool_models=max(r["pool_models"] for r in prs),
               max_dist=max(fin) if fin else None,
               node_range=[min(r["node_range"][0] for r in prs), max(r["node_range"][1] for r in prs)],
               filled_range=[min(r["filled_range"][0] for r in prs), max(r["filled_range"][1] for r in prs)],
               pool_range=r0.get("pool_range"), n_lo=max(r["n_lo"] for r in prs), n_hi=max(r["n_hi"] for r in prs),
               wall=float(sum(r["wall"] for r in prs)))
    for k in _ROW_ARRAYS:
        A = np.array([r[k] for r in use], dtype=np.float64)
        agg[k] = A.max(axis=0).tolist()
        agg[k + "_min"] = A.min(axis=0).tolist()
        agg[k + "_mean"] = A.mean(axis=0).tolist()
    agg["phase_max"] = {k: [float(np.max(r[k])) for r in prs] for k in _RATIO_KEYS}
    if "dumps" in r0:
        agg["dumps"] = r0["dumps"]
    return agg


def _with_aggregates(rows):
    """The rows with an aggregate row after the last phase row of every variant tested at several phases."""
    groups = {}
    for r in rows:
        if r.get("n_phases", 1) > 1 and not r.get("aggregate"):
            groups.setdefault((r["method"], r["base"]), []).append(r)
    last = {k: id(v[-1]) for k, v in groups.items()}
    out = []
    for r in rows:
        out.append(r)
        k = (r["method"], r.get("base"))
        if k in last and id(r) == last[k]:
            out.append(_aggregate(groups[k]))
    return out


_CRITERIA = dict(lpv=("ratio_dR_max", "ratio_EW"), residual=("ratio_dR_max",), residual_rms=("ratio_dR",),
                 max=("ratio_F",))
"""The ratios each recommendation criterion requires to stay <= warn (largest over the lines)."""


def _crit(r, keys, stat="max"):
    """The criterion value of a row: the largest over the keys and lines; for an aggregate row over the phases the
    worst phase (stat 'max') or the phase mean (stat 'mean')."""
    suf = "_mean" if stat == "mean" and r.get("aggregate") else ""
    v = []
    for k in keys:
        x = r.get(k + suf, r.get(k))
        v.append(float(np.max(x)) if x is not None and np.size(x) and np.all(np.isfinite(x)) else np.nan)
    return np.nan if np.any(np.isnan(v)) else max(v)


def _recommend(rows, criterion, warn, phase_stat="max"):
    """Per method the row with the fewest models whose criterion stays <= warn (see SparseLibraryTest.recommend)."""
    # PP 2026-10-02: check / reference rows, per-phase rows of aggregated variants and imu replicas excluded (reviewer)
    keys = _CRITERIA[criterion]
    out = {}
    for m in sorted({r["method"] for r in rows}):
        cand = [r for r in rows if r["method"] == m and r.get("dT") is not None and not r.get("check")]
        if m == "imu":
            cand = [r for r in cand if r.get("replicas") == 1]
        cand = [r for r in cand if r.get("aggregate") or r.get("n_phases", 1) == 1]
        ok = [r for r in cand if _crit(r, keys, phase_stat) <= warn]
        if not ok:
            out[m] = None
            continue
        best = min(ok, key=lambda r: (r["models"], _crit(r, keys, phase_stat)))
        ratio = _crit(best, keys, phase_stat)
        out[m] = dict(label=best["label"], dT=best["dT"], replicas=best["replicas"],
                      replica_step=best.get("replica_step"), models=best["models"], ratio=ratio,
                      margin=float(warn - ratio), ratio_worst=_crit(best, keys, "max"),
                      phase_stat=phase_stat if best.get("aggregate") else None, n_phases=best.get("n_phases", 1))
    return out


def _variant_info(plan, rows, R):
    got = rows >= 0
    return dict(n_planned=int(plan.nn), n_filled=int(got.any(axis=1).sum()), models=int(plan.nn * R),
                node_range=[float(plan.node_teff[0]), float(plan.node_teff[-1])],
                pool_models=int(got.sum()))


def _selection_key(rows):
    """The node models and their grouping (one tuple per filled node, in node order): two variants with the same key
    have the same library-mode library and nodes (lib_nodes nmin 1), hence the same integrator."""
    return tuple(tuple(sorted(int(x) for x in r[r >= 0])) for r in rows if (r >= 0).any())


def _phases(phases):
    """Node phases (fractions of dT): None -> (0,), n -> k / n (k = 0 .. n - 1), or a sequence in [0, 1)."""
    if phases is None:
        return [0.0]
    if isinstance(phases, (int, np.integer)) and not isinstance(phases, (bool, np.bool_)):
        n = _int(phases, "phases", 1)
        return [k / n for k in range(n)]
    ph = [_float(x, "phase") for x in phases]
    if not ph or any(not 0.0 <= x < 1.0 for x in ph) or len(set(ph)) != len(ph):
        raise ValueError("phases must be distinct fractions of dT in [0, 1), got {!r}".format(phases))
    return ph


def sparse_library_test(store, samples, dumps, reference, theta=None, phi=None, los="thompson2024", grid=None,
                        lref=None, lpv=None, dTs=V8_DTS, replicas=V8_REPLICAS, offset=0.0, margin=None,
                        reference_library=None, reference_nmin=20, imu=None, vwin=600.0, warn=None,
                        criterion="lpv", pattern=None, tune_malloc=False, log=None, phases=None, phase_stat="max",
                        dedupe=True):
    """
    V8: how accurate is a library-mode run with node spacing dT and R models per node? Sparse libraries from the
    models of an existing per-point run, against that run's full-library products (no new FASTWIND runs).

    For every dT and node phase: nodes at offset + (phase + k) dT covering the T_eff' range of the dumps' samples
    plus ``margin`` (:func:`plan_teff_nodes`); for every R: per node the R models of the per-point run closest to the
    node T_eff' inside its bin (:func:`select_node_models`; deterministic; or, for spread replicas, the models closest
    to the replicas' T_eff' of a library-mode plan), their library (:func:`flux_library_from_models`, bins centred on
    the nodes, nodes with ``lib_nodes(nmin=1)``) and :class:`~ppmpy.synspec.disc.DiscFlux`; every dump is integrated
    (:func:`ppmpy.synspec.dumps.disc_dump`, the products' diagnostics) and compared with the reference product. Nodes
    whose bin holds no model of the per-point run (beyond its T_eff' range) are skipped (the end nodes then clamp, as
    the full library does).

    Parameters
    ----------
    store: ProfileStore or str
        The per-point models (M424: d3200_r4050_N1236544/profiles.npz; memory-mapped, only the chosen rows are read).
    samples: str, mapping or callable
        Per-dump samples (directory of dNNNN.npz, mapping dump -> sample, or callable; as the validation checks).
    dumps: iterable of int
        The dumps to integrate (M424: the subset 3200, 3334, 4000, 4169, 4391, 4800 + 10 drawn with
        default_rng(5)).
    reference: str, mapping or callable
        The full-library products: a per-dump directory (M424: disc_dumps_r4050_N1236544/flux, dNNNN.npz with F, F0,
        diag_F, diag_keys), or dump -> dict(F, F0, diag_F, diag_keys).
    theta, phi: array-like, optional
        Point coordinates (default the store's theta, phi).
    los: str or array-like
        Lines of sight of the reference (default the 8 of Thompson et al. 2024).
    grid: VelocityGrid, optional
        Grid of the reference (default M424).
    lref: LineSet or array-like
        Reference wavelengths of the store's lines.
    lpv: dict or str, optional
        The flux run's LPV scales (:func:`ppmpy.synspec.validate.lpv_residual_rms`) or the path of its time series
        (computed here with ``vwin``; M424 ~60-70 s). None: ratios NaN.
    dTs, replicas:
        Node spacings [K] and models per node (default 10, 20, 50 K and 1, 3). An entry of ``replicas`` is R (the R
        models closest to the node) or (R, step): replicas spread as a library-mode run spreads them, the models
        closest to node + (r - (R - 1)/2) step (step [K], or 'bin' for dT / R: the replicas sample the whole bin).
        M424: the models closest to a node lie within ~0.01 K of it and are practically one model (R 3 = R 1 to
        <= 2 %), so R alone does not show what replicas 1 K apart would average.
    offset, margin:
        Node grid and range (:func:`plan_teff_nodes`; offset 5 puts M424 nodes at the centres of the per-point
        library's 10 K bins).
    reference_library: str or FluxLibrary, optional
        The reference run's library: adds a 'reference' row with ``flux_integrator(reference_library,
        nmin=reference_nmin)``, which must reproduce the stored products to their float32 rounding (a check of the
        comparison itself).
    imu: dict, optional
        The intensity method as well (DiscImu lazy): pool (Representatives or representatives.txt: the models with
        OUT_IMU files), runs (directory of their OUT_IMU files) and layout (default IMU_LAYOUT), reference (per-dump
        directory of the imu run), lpv (dict or the imu time series), dTs (default ``dTs``), dumps (default
        ``dumps``), phases (default ``phases``), fft ('lazy'), dtype ('float64'), check (build_imu_library's flux
        check, default False). One model per node (the pool member closest to the node inside its bin; the intensity
        library does not average replicas). A selection that is the whole pool (M424: at dT 10 K every bin holds
        one production representative, e.g. offset 5) rebuilds the production intensity library: a check row.
    vwin: float
        Window [km/s] of the time-variable comparison and of an LPV computed here (600, as the LPV).
    warn: float, optional
        Largest acceptable ratio to the LPV (default :data:`ppmpy.synspec.validate.LPV_WARN`, 5 %).
    criterion: {'lpv', 'residual', 'residual_rms', 'max'}
        What the recommendation requires to stay <= warn (largest over the lines): 'lpv' (default) the time-variable
        part, i.e. what an LPV analysis of residual spectra and EW sees: max|dR| / LPV residual rms (dR = the
        deviation of the residuals about the subset mean, |y| <= vwin) and the rms of the EW deviation about the
        subset mean / LPV EW rms; 'residual' max|dR| only; 'residual_rms' rms(dR) only; 'max' max|dF| / LPV residual
        rms (static offsets included; conservative: a largest deviation against an rms).
    pattern: str, optional
        Sample file pattern (default 'd{dump:04d}.npz').
    tune_malloc: bool
        Set glibc's malloc thresholds of this process (:data:`ppmpy.synspec.disc.WORKER_MALLOC`, as the pool workers
        of run_disc_dumps): the per-dump temporaries are then reused instead of page-faulted again (M424 on a login
        node: 251 s of the 338 s CPU of the 16-dump flux table were system time without). Changes no result, but
        changes this process's malloc settings (default False).
    log: callable, optional
    phases: int or sequence of float, optional
        Node phases: the nodes of a variant at offset + (phase + k) dT for each phase (fractions of dT in [0, 1);
        an int n gives 0, 1/n, ..., (n - 1)/n; default (0,): ``offset`` only). The deviation of a sparse library
        depends on where its nodes fall (single-model artefacts such as the EW(T_eff') sawtooth): with several phases
        every variant gets one row per phase and an aggregate row (worst, best and mean over the phases that are not
        checks), and the recommendation uses the aggregate. M424: :data:`V8_PHASES`.
    phase_stat: {'max', 'mean'}
        Which aggregate the recommendation uses: 'max' (default) the worst phase, 'mean' the phase mean.
    dedupe: bool
        A variant whose node models and grouping equal an earlier variant's (same integrator; e.g. imu selections that
        are the whole pool at several phases) reuses its comparison (``same_as``) instead of integrating again.

    Returns
    -------
    SparseLibraryTest
        Rows per variant (and phase) and the recommendation per method (the fewest FASTWIND models within ``warn``).

    Notes
    -----
    Memory: the projections (3 nlos N float64; M424 0.24 GB), one dump's samples (4 N float64, 40 MB), the
    integrators (DiscFlux: nl nn (ny + L) x 8 bytes, ~0.1 GB at dT = 10), the windowed profiles of every variant and
    dump (nd nlos nl nw x 8 bytes); imu: one intensity library at a time (nb nl K ny x 4 bytes x 2, 1.9 GB at dT = 10
    with the planned bins beyond the pool's range) plus DiscImu's lazy working set (~1 GB). An imu-only call (dTs=(),
    no reference_library) reads the flux reference products not at all.
    Measured (M424, 16 dumps, Trillium login node, 2026-10-02): flux, 9 variants (reference, dT 10/20/50/100 x R 1/3)
    352 s wall, 87 s user + 251 s system CPU without tune_malloc, peak RSS 2.0 GB; 17 variants (also the spread
    replicas) with tune_malloc 231 s, 164 s user + 55 s system, 2.5 GB; imu (dT 10/20/50, lazy float64 DiscImu)
    782 s, 369 s user + 396 s system, 4.4 GB (the dT 10 set-up reads 909 OUT_IMU files: 13 s). With the four phases
    (run_v8_phases.py, tune_malloc): flux, 33 variants (dT 10/20/50/100 x R 1, 3 spread x 4 phases + reference) 386 s,
    316 s user + 63 s system, 3.6 GB; imu one process per dT, 4 phases each: dT 10 757 s (4.4 GB), dT 20 520 s
    (2.8 GB), dT 50 320 s (1.8 GB).
    """
    # PP 2026-10-02: new (M7, V8); phases + aggregates, check rows, dedupe, imu-only without flux I/O (reviewer)
    from .dumps import SAMPLE_PATTERN, disc_dump, flux_integrator
    from .sphere import project_los
    from .validate import LPV_WARN, _get_sample
    T0 = time.time()
    _log = _logger(log, T0)
    warn = LPV_WARN if warn is None else _float(warn, "warn", positive=True)
    if criterion not in _CRITERIA:
        raise ValueError("criterion must be one of {}, got {!r}".format(sorted(_CRITERIA), criterion))
    if phase_stat not in ("max", "mean"):
        raise ValueError("phase_stat must be 'max' or 'mean', got {!r}".format(phase_stat))
    phs = _phases(phases)
    pattern = SAMPLE_PATTERN if pattern is None else pattern
    st = _as_store(store)
    names, lr = _lines_of(st, lref)
    lineset = lref if isinstance(lref, LineSet) else LineSet(names, lr)
    grid = VelocityGrid() if grid is None else grid
    dl = sorted({int(d) for d in dumps})
    if not dl:
        raise ValueError("no dumps given")
    dTs = [_float(x, "dT", positive=True) for x in dTs]
    Rs = [_replica_spec(x) for x in replicas]
    offset = _float(offset, "offset")
    if tune_malloc:
        from .disc import WORKER_MALLOC, _tune_malloc
        _tune_malloc(**WORKER_MALLOC)
    if theta is None or phi is None:
        if "theta" not in st or "phi" not in st:
            raise ValueError("give theta, phi (the store has no coordinates)")
        theta, phi = st["theta"], st["phi"]
    mu, tn, pn = project_los(theta, phi, los, method="matmul")
    _log("projections {} x {}".format(*mu.shape))
    # T_eff' range of the dumps (the nodes must cover it)
    dl_all = set(dl) | ({int(d) for d in (imu.get("dumps") or ())} if imu else set())
    span = [np.inf, -np.inf]
    for d in sorted(dl_all):
        t = np.asarray(_get_sample(samples, d, pattern)["teff"])
        span = [min(span[0], float(t.min())), max(span[1], float(t.max()))]
    _log("T_eff' of {} dumps: {:.1f}-{:.1f} K".format(len(dl_all), *span))
    pool_t = np.asarray(st.teff, dtype=np.float64)
    pool_idx = np.asarray(st.idx, dtype=np.int64)
    usable = st.usable() & np.isfinite(pool_t)
    pool_rows = np.flatnonzero(usable)
    pool_range = [float(pool_t[pool_rows].min()), float(pool_t[pool_rows].max())] if pool_rows.size else None
    m = np.abs(grid.y) <= vwin
    plans = {}

    def plan_of(dT, ph):
        if (dT, ph) not in plans:
            plans[dT, ph] = plan_teff_nodes(span, dT=dT, margin=margin, offset=offset + ph * dT, kind="values")
        return plans[dT, ph]

    accs = []
    if reference_library is not None:
        t = time.time()
        integ = flux_integrator(reference_library, nmin=reference_nmin, grid=grid)
        accs.append(_Acc("flux", "reference", None, None, integ, dict(
            n_planned=int(integ.nn), n_filled=int(integ.nn), models=None,
            node_range=[float(integ.t[0]), float(integ.t[-1])], pool_models=None, base="reference", n_phases=1,
            offset=None, phase=None, check=False, filled_range=[float(integ.t[0]), float(integ.t[-1])],
            pool_range=pool_range), lr.size))
        accs[-1].wall += time.time() - t
    seen = {}
    for dT in dTs:
        for R, step in Rs:
            stepv = None if step is None else (dT / R if step == "bin" else step)
            if stepv is not None and R > 1 and not 0.5 * (R - 1) * stepv < 0.5 * dT:
                raise ValueError("{} replicas {:g} K apart leave the bin of dT = {:g} K".format(R, stepv, dT))
            offs = None if stepv is None else (np.arange(R) - 0.5 * (R - 1)) * stepv
            base = "dT{:g}_R{}".format(dT, R) + ("" if stepv is None else "s{:.3g}".format(stepv))
            for ph in phs:
                t = time.time()
                plan = plan_of(dT, ph)
                off = float(np.mod(plan.offset, dT))
                rows, dist = select_node_models(pool_t, plan.node_teff, R, edges=plan.edges, idx=pool_idx,
                                                usable=usable, offsets=offs)
                sel = np.sort(rows[rows >= 0])
                info = _variant_info(plan, rows, R)
                info.update(max_dist=float(np.nanmax(dist)) if np.isfinite(dist).any() else None, base=base,
                            n_phases=len(phs), offset=off, phase=ph, pool_range=pool_range,
                            check=bool(sel.size == pool_rows.size and np.array_equal(sel, pool_rows)))
                label = base + ("_o{:g}".format(off) if len(phs) > 1 else "")
                key = _selection_key(rows)
                if dedupe and key in seen:
                    orig = seen[key]
                    info["filled_range"] = orig.info["filled_range"]
                    acc = _Acc("flux", label, dT, R, None, info, lr.size, step=stepv, alias=orig)
                    _log("flux {}: the node models of {} (comparison reused)".format(label, orig.label))
                else:
                    lib = flux_library_from_models(_substore(st, sel), grid, lineset, dT=dT, offset=plan.offset,
                                                   edges=plan.edges)
                    integ = flux_integrator(lib, nmin=1, grid=grid)
                    info["filled_range"] = [float(integ.t[0]), float(integ.t[-1])]
                    acc = _Acc("flux", label, dT, R, integ, info, lr.size, step=stepv)
                    seen[key] = acc
                    _log("flux {}: {} of {} nodes filled ({} models), nodes {:.1f}-{:.1f} K, max |T - target| {:.3f} "
                         "K{}".format(label, info["n_filled"], info["n_planned"], sel.size, integ.t[0], integ.t[-1],
                                      info["max_dist"] or 0.0, " (check: the whole pool)" if info["check"] else ""))
                acc.wall += time.time() - t
                accs.append(acc)
    out_rows, lpvs = [], {}
    if accs:
        active = [a for a in accs if a.alias is None]
        Rref, kew, EWref = [], None, []
        for d in dl:
            s = _get_sample(samples, d, pattern)
            ref = _reference_dump(reference, d)
            if kew is None and "ew" in ref["diag_keys"]:
                kew = ref["diag_keys"].index("ew")
            Rref.append(ref["F"][..., m])
            if kew is not None and ref["diag_F"] is not None:
                EWref.append(ref["diag_F"][..., kew])
            for a in active:
                t = time.time()
                r = disc_dump(a.integ, s, mu, tn, pn, lref=lr, grid=grid)
                a.add(r, ref, m, kew)
                a.wall += time.time() - t
            _log("dump {}: ".format(d) + ", ".join("{} {:.1e}".format(a.label, float(a.maxF.max())) for a in active))
        Rref = np.array(Rref)
        Rref -= Rref.mean(axis=0)
        EWref = None if not EWref else np.array(EWref) - np.mean(EWref, axis=0)
        lpv_f = _lpv_of(lpv, vwin, _log)
        out_rows = [_finish(a, Rref, EWref, lpv_f) for a in accs]
        lpvs["flux"] = lpv_f
    del accs, seen

    if imu:
        out_rows += _sparse_imu(st, samples, dl, imu, mu, tn, pn, grid, lineset, plan_of, dTs, phs, vwin, pattern,
                                lpvs, dedupe, _log)
    out_rows = _with_aggregates(out_rows)
    params = dict(dTs=dTs, replicas=[list(x) for x in Rs], offset=float(offset), margin=margin, vwin=float(vwin),
                  warn=float(warn), criterion=criterion, phases=phs, phase_stat=phase_stat, dedupe=bool(dedupe),
                  span=span, reference=str(reference) if isinstance(reference, (str, os.PathLike)) else None,
                  store=st.path, los=los if isinstance(los, str) else np.asarray(los).tolist(), grid=grid.to_dict(),
                  reference_nmin=int(reference_nmin), pool_range=pool_range, wall=time.time() - T0)
    res = SparseLibraryTest(out_rows, names, dl, {k: _json_lpv(v) for k, v in lpvs.items()}, params,
                            _recommend(out_rows, criterion, warn, phase_stat))
    _log("V8 done\n" + res.table())
    return res


def _replica_spec(x):
    """(R, step) from R or (R, step); step None (closest to the node), a number [K] or 'bin'."""
    if isinstance(x, (tuple, list)):
        if len(x) != 2:
            raise ValueError("a replicas entry must be R or (R, step), got {!r}".format(x))
        R, step = _int(x[0], "replicas", 1), x[1]
        if step is not None and step != "bin":
            step = _float(step, "replica step", positive=True)
        return R, step
    return _int(x, "replicas", 1), None


def _json_lpv(lpv):
    if lpv is None:
        return None
    return {k: (None if v is None else (np.asarray(v).tolist() if np.ndim(v) else float(v))) for k, v in lpv.items()}


def _sparse_imu(st, samples, dl0, cfg, mu, tn, pn, grid, lineset, plan_of, dTs0, phs0, vwin, pattern, lpvs, dedupe,
                _log):
    """The intensity-method rows of V8 (one DiscImu at a time; one model per node)."""
    from .dumps import disc_dump, imu_integrator
    from .validate import _get_sample
    cfg = dict(cfg)
    for k in ("pool", "runs", "reference"):
        if cfg.get(k) is None:
            raise ValueError("imu needs '{}'".format(k))
    pool = cfg["pool"]
    if isinstance(pool, (str, os.PathLike)):
        pool = Representatives.read(pool)
    pidx = np.asarray(pool.idx, dtype=np.int64)
    sidx = np.asarray(st.idx, dtype=np.int64)
    o = np.argsort(sidx, kind="stable")
    k = np.clip(np.searchsorted(sidx[o], pidx), 0, sidx.size - 1)
    if not np.all(sidx[o][k] == pidx):
        raise ValueError("pool models missing from the store: {}".format(pidx[sidx[o][k] != pidx][:10].tolist()))
    prow = o[k]
    nprow = np.unique(prow).size
    pt = np.asarray(st.teff, dtype=np.float64)[prow]
    pool_range = [float(pt.min()), float(pt.max())]
    dTs = [_float(x, "dT", positive=True) for x in (cfg.get("dTs") or dTs0)]
    phs = _phases(cfg["phases"]) if cfg.get("phases") is not None else list(phs0)
    dl = sorted({int(d) for d in (cfg.get("dumps") or dl0)})
    m = np.abs(grid.y) <= vwin
    lpv = _lpv_of(cfg.get("lpv"), vwin, _log)
    lpvs["imu"] = lpv
    rows_out, seen = [], {}
    for dT in dTs:
        base = "dT{:g}_R1".format(dT)
        for ph in phs:
            t = time.time()
            plan = plan_of(dT, ph)
            off = float(np.mod(plan.offset, dT))
            rows, dist = select_node_models(pt, plan.node_teff, 1, edges=plan.edges, idx=pidx)
            sel = np.sort(prow[rows[rows >= 0]])
            info = _variant_info(plan, rows, 1)
            info.update(max_dist=float(np.nanmax(dist)) if np.isfinite(dist).any() else None, base=base,
                        n_phases=len(phs), offset=off, phase=ph, pool_range=pool_range, dumps=dl,
                        check=bool(sel.size == nprow))
            label = base + ("_o{:g}".format(off) if len(phs) > 1 else "")
            key = tuple(sel.tolist())
            if dedupe and key in seen:
                row = dict(seen[key])
                row.update(info)
                row.update(label=label, wall=time.time() - t, same_as=seen[key]["label"],
                           filled_range=seen[key]["filled_range"])
                rows_out.append(row)
                _log("imu {}: the node models of {} (comparison reused)".format(label, row["same_as"]))
                continue
            sub = _substore(st, sel)
            lib = flux_library_from_models(sub, grid, lineset, dT=dT, offset=plan.offset, edges=plan.edges)
            im, _, reps = imu_library_from_models(lib, sub, grid=grid, lref=lineset, imu_runs=cfg["runs"],
                                                  layout=cfg.get("layout", IMU_LAYOUT), check=cfg.get("check", False))
            integ = imu_integrator(im, lref=lineset, fft=cfg.get("fft", "lazy"), dtype=cfg.get("dtype", "float64"))
            info["n_filled"] = int(integ.nn)
            info["filled_range"] = [float(integ.t[0]), float(integ.t[-1])]
            a = _Acc("imu", label, dT, 1, integ, info, lineset.lref.size)
            a.wall += time.time() - t
            _log("imu {}: {} of {} nodes ({} representatives), nodes {:.1f}-{:.1f} K, set-up {:.0f} s{}".format(
                label, integ.nn, plan.nn, len(reps), integ.t[0], integ.t[-1], a.wall,
                " (check: the whole pool)" if info["check"] else ""))
            Rref, EWref, kew = [], [], None
            for d in dl:
                s = _get_sample(samples, d, pattern)
                ref = _reference_dump(cfg["reference"], d)
                if kew is None and "ew" in ref["diag_keys"]:
                    kew = ref["diag_keys"].index("ew")
                Rref.append(ref["F"][..., m])
                if kew is not None and ref["diag_F"] is not None:
                    EWref.append(ref["diag_F"][..., kew])
                t = time.time()
                r = disc_dump(integ, s, mu, tn, pn, lref=lineset.lref, grid=grid, batch=None)
                a.add(r, ref, m, kew)
                a.wall += time.time() - t
            _log("imu {}: max|dF| {:.2e} ({} dumps)".format(label, float(a.maxF.max()), len(dl)))
            Rref = np.array(Rref)
            Rref -= Rref.mean(axis=0)
            EWref = None if not EWref else np.array(EWref) - np.mean(EWref, axis=0)
            row = _finish(a, Rref, EWref, lpv)
            seen[key] = row
            rows_out.append(row)
            del integ, im, a, lib, sub
    return rows_out
