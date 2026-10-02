"""
Tests of ppmpy.synspec.libmode (M7, library mode: one FASTWIND model per T_eff' node).

Synthetic (any machine): node planning (range, grid, replicas, the 'idx teff' table the runner parses and the
points.npz merge_task takes, reading back); planning safeguards (teff_ranges tables recognised column by column, 'idx
teff' arrays refused, implausible spans / node counts, the retry-nudge room warning, avoid= / check_collisions with
existing_indices of a per-point run, the points.npz size warning); the closest-model selection and its validation
(replica targets outside the bin, -1 / out-of-range / float select rows); the flux library of node models against the
analytic node profiles (models on the grid points, float64); the plan checks and nudged='keep' / 'drop' / 'raise'; the
offset warning without a plan; nmin = 1 keeps every node; node_representatives (only the flux library's models in its
bins, nudged replicas excluded, ties to the smaller idx); an end-to-end library-mode run with the fake FASTWIND (plan ->
runner -> merge -> combine -> library_from_models with the intensity library from extracted OUT_IMU files -> DiscFlux,
DiscImu, save / load; the replica warning); V8 (sparse_library_test) on a toy run: its reference row reproduces the
toy's products, its error decreases with denser nodes, node phases give per-phase and aggregate rows; on a small pool a
whole-pool selection is a check row (exact, reused by dedupe) and is never recommended; the recommendation rules.

M424 (marker m424, slow): the V8 table on a few dumps of the subset (time-boxed; the full 16-dump table with node
phases is the M7 scratch run /scratch/ppathak/synspec_shadow/m7/libmode/v8/run_v8_phases.py). Real FASTWIND (markers
fastwind, slow, m424): library mode at the T_eff' of two production per-point models (the intensity-library
representatives P009729, P001652) reproduces their OUT profiles (profiles.npz rows) and their intensity-library rows
(imu_library_dT10.npz) bit for bit.

PP 2026-10-02: new (M7); review fixes (check rows, phases, nudged, select / offsets validation, collisions).
"""
import json
import os
import shutil
import sys
import time
import warnings

import numpy as np
import pytest

from conftest import m424_path
from ppmpy.synspec import dumps as dm
from ppmpy.synspec import libmode as lm
from ppmpy.synspec import library as lb
from ppmpy.synspec import testing as tt
from ppmpy.synspec import validate as va
from ppmpy.synspec.conventions import C_KMS
from ppmpy.synspec.fastwind import batch, fake
from ppmpy.synspec.fastwind.install import FastwindInstall
from ppmpy.synspec.fwresults import ProfileStore, combine, merge_task
from ppmpy.synspec.spectral import LineSet, VelocityGrid

LINES = ["HEI4026", "HEII4200", "HEI4922"]
LREF = np.array([4026.22, 4199.90, 4921.93])
LINESET = LineSet(LINES, LREF)
SHADOW = os.environ.get("PPMPY_SYNSPEC_LIBMODE_SHADOW", "/scratch/ppathak/synspec_shadow/m7/libmode")
SUBSET = (3200, 3334, 4000, 4169, 4391, 4800, 3237, 3287, 3657, 3948, 4022, 4206, 4267, 4481, 4488, 4766)


def _quiet(*a):
    pass


# ------------------------------------------------------------------------------------------------------------------
# synthetic node models
# ------------------------------------------------------------------------------------------------------------------
def _grid_store(idx, teff, grid, ls, pars, status=None):
    """A profiles store whose models sit on the grid points (float64 rows: interpolation is exact), analytic toy
    lines (testing.toy_depth / toy_continuum)."""
    teff = np.asarray(teff, dtype=np.float64)
    n, nl, ny = teff.size, len(ls), grid.ny
    lam = np.empty((n, nl, ny))
    fn = np.empty((n, nl, ny))
    fc = np.empty((n, nl, ny))
    for j, p in enumerate(pars):
        lam[:, j] = ls.lref[j] * np.exp(grid.y / C_KMS)[None, :]
        fn[:, j] = 1.0 - tt.toy_depth(teff, grid.y, p, grid)
        fc[:, j] = tt.toy_continuum(teff, p, np.broadcast_to(grid.y, (n, ny)), grid)
    st = np.full(n, "ok", dtype="<U12") if status is None else np.asarray(status, dtype="<U12")
    bad = st != "ok"
    lam[bad], fn[bad], fc[bad] = np.nan, np.nan, np.nan
    return dict(idx=np.asarray(idx, dtype=np.int32), teff=teff, status=st, niter=np.full(n, 61, np.int16),
                lam=lam, fcont=fc, fnorm=fn, lines=np.array(ls.names), teff_nudge=np.zeros(n, np.float32))


def _expected(teffs, grid, pars):
    """Mean over models (rows of teffs) of the analytic profile and of F_c at the first grid point."""
    t = np.asarray(teffs, dtype=np.float64)
    prof = np.array([np.mean(1.0 - tt.toy_depth(t, grid.y, p, grid), axis=0) for p in pars])
    fc = np.array([np.mean(tt.toy_continuum(t, p, np.full((t.size, 1), grid.y[0]), grid)) for p in pars])
    return prof, fc


# ------------------------------------------------------------------------------------------------------------------
# planning
# ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dT,R,offset,margin,step", [(10.0, 1, 0.0, None, 1.0), (20.0, 3, 5.0, 0.0, 1.0),
                                                     (50.0, 2, 0.0, 100.0, 3.0), (7.5, 4, 2.5, 3.0, 0.5),
                                                     (10.0, 3, 5.0, None, "bin"), (30.0, 2, 0.0, 0.0, None)])
def test_plan_covers_range(tmp_path, dT, R, offset, margin, step):
    """Nodes on offset + k dT from <= tmin - margin to >= tmax + margin (no node more than needed), R replicas per
    node centred on it inside its bin, idx from start_idx; the table parses with the runner's split_table, reads back,
    and the points.npz serves merge_task (idx = row number, teff)."""
    rng = np.random.default_rng(1)
    t = 38230.0 * (1.0 + 0.009 * rng.standard_normal(5000))
    kw = {} if step is None else dict(replica_step=step)
    plan = lm.plan_teff_nodes(t, dT=dT, replicas=R, offset=offset, margin=margin, start_idx=100, **kw)
    sv = dT / R if step in (None, "bin") else step                      # default 'bin': dT / R
    assert plan.replica_step == pytest.approx(sv)
    np.testing.assert_allclose(plan.teff[:R] - plan.node_teff[0], (np.arange(R) - 0.5 * (R - 1)) * sv, atol=5e-4)
    mg = dT if margin is None else margin
    lo, hi = t.min() - mg, t.max() + mg
    assert plan.node_teff[0] <= lo < plan.node_teff[0] + dT and plan.node_teff[-1] - dT < hi <= plan.node_teff[-1]
    np.testing.assert_allclose(np.diff(plan.node_teff), dT, rtol=1e-12)
    k = (plan.node_teff - offset) / dT
    np.testing.assert_allclose(k, np.round(k), atol=1e-9)
    assert plan.n == plan.nn * R and len(plan) == plan.n and plan.replicas == R
    np.testing.assert_array_equal(plan.idx, 100 + np.arange(plan.n))
    np.testing.assert_array_equal(plan.node, np.repeat(np.arange(plan.nn), R))
    np.testing.assert_allclose(plan.teff.reshape(plan.nn, R).mean(axis=1), plan.node_teff, atol=1e-9)
    np.testing.assert_array_equal(lb.teff_bins(plan.teff, plan.edges), plan.node)
    assert plan.span == (t.min(), t.max())
    # the table: the runner's parser, read back
    path = plan.write(str(tmp_path / "points.txt"))
    ent = batch.split_table(path)
    assert [e[1] for e in ent] == plan.idx.tolist() and [e[2] for e in ent] == plan.texts
    assert all(len(s.split(".")[1]) == 3 for s in plan.texts)
    p2 = lm.NodePlan.read(path, dT=dT, offset=offset)
    for k in ("idx", "teff", "node", "replica", "node_teff"):
        np.testing.assert_allclose(getattr(p2, k), getattr(plan, k), rtol=0, atol=1e-9, err_msg=k)
    assert p2.replicas == R and p2.decimals == 3
    # points.npz: merge_task's contract, and the whole plan back
    pz = plan.write_points_npz(str(tmp_path / "points.npz"))
    with np.load(pz) as z:
        assert np.all(z["idx"][plan.idx] == plan.idx) and np.array_equal(z["teff"][plan.idx], plan.teff)
        assert np.isnan(z["teff"][:100]).all()
    p3 = lm.read_node_plan(pz)
    for k in ("idx", "teff", "node", "replica", "node_teff", "edges"):
        np.testing.assert_array_equal(getattr(p3, k), getattr(plan, k), err_msg=k)
    assert p3.params() == plan.params()
    np.testing.assert_array_equal(plan.position([100, 99, plan.idx[-1], 10 ** 6]), [0, -1, plan.n - 1, -1])


def test_plan_sources(tmp_path):
    """teff_span / plan_teff_nodes take arrays, sample mappings and files, (tmin, tmax) pairs, teff_ranges tables and
    lists of these; all give the same nodes."""
    rng = np.random.default_rng(2)
    a = [38000.0 + 300.0 * rng.standard_normal(1000) for _ in range(3)]
    lo, hi = min(x.min() for x in a), max(x.max() for x in a)
    table = np.array([[3200 + i, x.min(), x.max(), 0.0, 0.0, x.std()] for i, x in enumerate(a)])
    np.savez(str(tmp_path / "d0001.npz"), teff=a[0].astype(np.float32))
    for src in (np.concatenate(a), a, [dict(teff=x) for x in a], (lo, hi), table, [table, (lo, hi)]):
        assert lm.teff_span(src) == (lo, hi), type(src)
    assert lm.teff_span(str(tmp_path / "d0001.npz")) == (float(a[0].astype(np.float32).min()),
                                                        float(a[0].astype(np.float32).max()))
    p1, p2 = lm.plan_teff_nodes(table, dT=10.0), lm.plan_teff_nodes(a, dT=10.0)
    np.testing.assert_array_equal(p1.node_teff, p2.node_teff)
    np.testing.assert_array_equal(lm.teff_span(table, kind="values"), (table.min(), table.max()))
    pts = tmp_path / "points.txt"
    pts.write_text("0 38000.000\n1 38100.500\n")
    assert lm.teff_span(str(pts)) == (38000.0, 38100.5)


def test_nodeplan_direct():
    """A NodePlan made directly (arbitrary node T_eff' on a grid of spacing dT, as the real-FASTWIND test does): default
    replica step, repr, bins, positions; inconsistent inputs raise."""
    p = lm.NodePlan([7, 3], [35785.014, 35977.532], [0, 1], [0, 0], [35785.014, 35977.532], 192.518, offset=35785.014)
    assert p.replica_step == pytest.approx(192.518) and p.texts == ["35785.014", "35977.532"] and "2 nodes" in repr(p)
    np.testing.assert_array_equal(lb.teff_bins(p.teff, p.edges), [0, 1])
    np.testing.assert_array_equal(p.position([3, 7, 5]), [1, 0, -1])
    for bad in (dict(idx=[1, 1]), dict(node=[0, 2]), dict(node_teff=[35785.014, 36000.0]), dict(teff=[0.0, 1.0])):
        kw = dict(idx=[7, 3], teff=[35785.014, 35977.532], node=[0, 1], replica=[0, 0],
                  node_teff=[35785.014, 35977.532], dT=192.518)
        kw.update(bad)
        with pytest.raises(ValueError):
            lm.NodePlan(**kw)


@pytest.mark.parametrize("kw", [dict(dT=0.0), dict(dT=np.nan), dict(margin=-1.0), dict(replicas=0),
                                dict(replicas=3, replica_step=5.0), dict(replicas=2, replica_step=0.0),
                                dict(replicas=2, replica_step=0.4, decimals=0), dict(start_idx=-1),
                                dict(replicas=1.5)])
def test_plan_errors(kw):
    with pytest.raises(ValueError):
        lm.plan_teff_nodes((38000.0, 38500.0), **dict(dict(dT=10.0), **kw))


def test_plan_bad_samples():
    for src in ([], np.array([38000.0, np.nan]), [dict(x=1)]):
        with pytest.raises(ValueError):
            lm.plan_teff_nodes(src)


def test_plan_sanity():
    """teff_span: an (n, 6) array of T_eff' with a whole first column is values, not a teff_ranges table (every column
    is checked; teff_ranges' NaN counts still pass); an np.loadtxt 'idx teff' table is refused in 'auto' mode;
    plan_teff_nodes refuses implausible spans and node counts unless forced, and warns when a retry nudge would
    move the top replica out of its bin."""
    a = np.array([[38001.0, 37100.3, 38950.7, 36950.2, 39010.8, 37800.4],
                  [38012.0, 37200.1, 38800.5, 37600.3, 38300.2, 37900.6]])
    assert lm.teff_span(a) == (36950.2, 39010.8)                        # the old heuristic: (37100.3, 38950.7)
    assert lm.teff_span(a, kind="ranges") == (37100.3, 38950.7)
    for c34 in ((np.nan, np.nan), (3.0, 0.0)):                         # teff_ranges without / with trange
        rg = np.array([[3200, 37000.5, 38900.25, c34[0], c34[1], 300.0], [3201, 36900.0, 39000.0, 1.0, 2.0, 310.0]])
        if np.isnan(c34[0]):
            rg[:, 3:5] = np.nan
        assert lm.teff_span(rg) == (36900.0, 39000.0)
    tbl = np.column_stack([np.arange(5), 38000.0 + 10.0 * np.arange(5) + 0.123])
    with pytest.raises(ValueError, match="idx teff"):
        lm.teff_span(tbl)
    assert lm.teff_span(tbl[:, 1]) == (38000.123, 38040.123)
    with pytest.raises(ValueError, match="force"):
        lm.plan_teff_nodes(tbl, kind="values")                          # indices 0..4 taken for T_eff'
    with pytest.raises(ValueError, match="force"):
        lm.plan_teff_nodes((20000.0, 60000.0), dT=1000.0)
    assert lm.plan_teff_nodes((20000.0, 60000.0), dT=1000.0, force=True).nn == 43
    with pytest.raises(ValueError, match="MAX_NODES"):
        lm.plan_teff_nodes((38000.0, 39100.0), dT=0.01)
    with pytest.warns(UserWarning, match="below its bin edge"):
        lm.plan_teff_nodes((38000.0, 38100.0), dT=10.0, replicas=5)     # room dT / (2R) = 1 K
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lm.plan_teff_nodes((38000.0, 38100.0), dT=10.0, replicas=3)     # room 1.67 K


def test_plan_avoid_and_collisions(tmp_path):
    """existing_indices reads a per-point run directory (points.txt, missing*.txt, profiles.npz, the results ledgers),
    a results directory, single files, arrays and lists; plan_teff_nodes(avoid=...) starts beyond those indices;
    NodePlan.check_collisions finds a plan that would collide; a huge start_idx warns when points.npz is written."""
    run = tmp_path / "pp"
    led = run / "results" / "task_0000"
    led.mkdir(parents=True)
    (run / "points.txt").write_text("".join("{} {:.3f}\n".format(i, 38000.0 + i) for i in range(50)))
    (run / "missing.txt").write_text("7 38008.000\n")
    (led / "part_000.idx").write_text("60 38060.000 ok 61 1 1 1\n61 38061.000 pnlte_failed 3 0 0 0\n")
    np.savez(str(run / "profiles.npz"), idx=np.arange(55), teff=np.full(55, 38000.0))
    np.testing.assert_array_equal(lm.existing_indices(str(run)), np.r_[np.arange(55), 60, 61])
    assert lm.existing_indices(str(run / "results")).tolist() == [60, 61]
    assert lm.existing_indices(str(led / "part_000.idx")).tolist() == [60, 61]
    assert lm.existing_indices([str(run / "missing.txt"), np.array([3, 1])]).tolist() == [1, 3, 7]
    (tmp_path / "empty").mkdir()
    assert lm.existing_indices(str(tmp_path / "empty")).size == 0 and lm.existing_indices([]).size == 0
    with pytest.raises(FileNotFoundError):
        lm.existing_indices(str(tmp_path / "nope.txt"))
    with pytest.raises(ValueError):
        lm.existing_indices(np.array([1.5]))
    plan = lm.plan_teff_nodes((38000.0, 38100.0), dT=10.0, avoid=str(run))
    assert plan.start_idx == 62 and plan.idx[0] == 62 and plan.check_collisions(str(run)).size == 0
    p0 = lm.plan_teff_nodes((38000.0, 38100.0), dT=10.0)
    with pytest.raises(ValueError, match="already used"):
        p0.check_collisions(str(run))
    np.testing.assert_array_equal(p0.check_collisions(str(run), error=False), p0.idx[p0.idx < 55])
    pz = plan.write_points_npz(str(tmp_path / "lm" / "points.npz"))
    np.testing.assert_array_equal(lm.existing_indices(pz), plan.idx)         # the planned rows only
    big = lm.plan_teff_nodes((38000.0, 38100.0), dT=10.0, start_idx=10 ** 6)
    with pytest.warns(UserWarning, match="rows for"):
        big.write_points_npz(str(tmp_path / "big.npz"))


# ------------------------------------------------------------------------------------------------------------------
# selection and libraries
# ------------------------------------------------------------------------------------------------------------------
def test_select_node_models():
    """The R closest models inside each node's bin, ties by idx; fewer -> -1; unusable models never chosen; the
    input order does not matter."""
    teff = np.array([99.0, 101.0, 100.0, 103.0, 97.0, 104.9, 105.0, 120.0, 100.0, 95.0])
    idx = np.array([5, 4, 9, 3, 2, 1, 0, 7, 8, 6])
    nodes = np.array([100.0, 110.0, 120.0, 130.0])
    rows, dist = lm.select_node_models(teff, nodes, 3, idx=idx)
    # node 100, bin [95, 105): 100 (idx 9, 8: tie -> idx 8 first), then 99 / 101 at distance 1 (idx 5, 4 -> 4 first)
    np.testing.assert_array_equal(rows[0], [8, 2, 1])
    np.testing.assert_allclose(dist[0], [0.0, 0.0, 1.0])
    # node 110, bin [105, 115): only 105.0 (row 6)
    np.testing.assert_array_equal(rows[1], [6, -1, -1])
    np.testing.assert_array_equal(rows[2], [7, -1, -1])
    assert (rows[3] == -1).all() and np.isnan(dist[3]).all()
    perm = np.random.default_rng(0).permutation(teff.size)
    r2, _ = lm.select_node_models(teff[perm], nodes, 3, idx=idx[perm])
    np.testing.assert_array_equal(np.where(r2 >= 0, perm[np.maximum(r2, 0)], -1), rows)
    usable = np.ones(teff.size, bool)
    usable[8] = False
    r3, _ = lm.select_node_models(teff, nodes, 1, idx=idx, usable=usable)
    assert r3[0, 0] == 2
    with pytest.raises(ValueError):
        lm.select_node_models(teff, nodes[::-1], 1)
    # replicas spread around the node (offsets): each the closest model to its target not taken by an earlier one
    r4, d4 = lm.select_node_models(teff, nodes, 3, idx=idx, offsets=(-1.0, 0.0, 1.0))
    np.testing.assert_array_equal(r4[0], [0, 8, 1])
    np.testing.assert_allclose(d4[0], 0.0)
    np.testing.assert_array_equal(r4[1], [6, -1, -1])
    with pytest.raises(ValueError):
        lm.select_node_models(teff, nodes, 3, offsets=(0.0, 1.0))
    # replica targets outside the node's bin: a clear error (was a numpy reduction error)
    with pytest.raises(ValueError, match="inside the node's bin"):
        lm.select_node_models(teff, nodes, 1, offsets=(7.0,))


def test_select_rows_validated():
    """select as row numbers: the -1 entries of select_node_models, rows beyond the store and float rows raise
    (an unselected model filled a node before); the valid rows and a bool mask work."""
    grid, ls = tt.TOY_GRID, tt.toy_lineset(1)
    pars = tt.toy_line_params(ls, 4)
    plan = lm.plan_teff_nodes((37950.0, 38050.0), dT=20.0, margin=0.0)
    st = _grid_store(plan.idx, plan.teff, grid, ls, pars)
    rows, _ = lm.select_node_models(plan.teff, plan.node_teff, 2, edges=plan.edges)
    assert (rows[:, 1] == -1).all()
    for bad in (rows.ravel(), [0, plan.n], np.array([0.0, 1.0])):
        with pytest.raises(ValueError, match="select"):
            lm.flux_library_from_models(st, grid, ls, plan=plan, select=bad)
    assert lm.flux_library_from_models(st, grid, ls, plan=plan, select=rows[rows >= 0]).count.sum() == plan.n
    m = np.zeros(plan.n, bool)
    m[[0, 2]] = True
    f = lm.flux_library_from_models(st, grid, ls, plan=plan, select=m)
    assert f.count.tolist() == [1, 0, 1] + [0] * (plan.nn - 3)


@pytest.mark.parametrize("R", [1, 3])
def test_library_from_node_models_is_analytic(R):
    """The flux library of node models on the grid points equals the analytic node profiles (mean over the usable
    replicas) to 1e-10 (the row offsets of interp_rows round the abscissae: ~1e-12 here), its F_c, mean T_eff' and
    counts too; a node whose models all failed is an empty bin, skipped by the nodes; the failed models are reported."""
    grid = tt.TOY_GRID
    ls = tt.toy_lineset(3)
    pars = tt.toy_line_params(ls, 0)
    plan = lm.plan_teff_nodes((37600.0, 38800.0), dT=100.0, replicas=R, replica_step=2.0, margin=0.0)
    status = np.full(plan.n, "ok", dtype="<U12")
    status[plan.node == 3] = "pnlte_failed"                       # a whole node fails
    if R > 1:
        status[(plan.node == 5) & (plan.replica == 0)] = "formal_failed"
    store = _grid_store(plan.idx, plan.teff, grid, ls, pars, status=status)
    with pytest.warns(UserWarning, match="1 planned nodes have no usable model"):
        nl = lm.library_from_models(store, grid, ls, plan=plan, prof_dtype=np.float64)
    f = nl.flux
    assert f.nb == plan.nn and f.dT == 100.0
    np.testing.assert_allclose(f.centres, plan.node_teff, atol=1e-9)
    ok = status == "ok"
    for k in range(plan.nn):
        m = (plan.node == k) & ok
        assert f.count[k] == m.sum()
        if not m.any():
            assert k == 3
            continue
        prof, fc = _expected(plan.teff[m], grid, pars)
        np.testing.assert_allclose(f.prof[k], prof, rtol=0, atol=1e-10)     # interp_rows offsets: ~1e-12
        np.testing.assert_allclose(f.fc[k], fc, rtol=1e-12)
        np.testing.assert_allclose(f.tmean[k], plan.teff[m].mean(), rtol=1e-14)
    assert nl.nodes.nn == plan.nn - 1 == f.filled.sum() and 3 not in np.concatenate(nl.nodes.groups)
    assert f.params["missing_nodes"] == [float(plan.node_teff[3])] and nl.report["missing_nodes"] == [plan.node_teff[3]]
    assert sorted(x["idx"] for x in nl.report["failed"]) == sorted(plan.idx[~ok].tolist())
    assert nl.report["models"] == ok.sum() and nl.imu is None and f.params["mode"] == "library"
    assert f.params["model_idx"] == sorted(plan.idx[ok].tolist())
    # the integrator of the nodes: every filled node, nmin 1
    integ = nl.flux_integrator()
    assert integ.nn == plan.nn - 1 and integ.node_params["nmin"] == 1
    np.testing.assert_allclose(integ.lref, ls.lref)
    # without a plan (dT only): the same library on the bins of the filled nodes
    f2 = lm.flux_library_from_models(store, grid, ls, dT=100.0, prof_dtype=np.float64)
    assert f2.nb == plan.nn and np.array_equal(f2.prof[f2.filled], f.prof[f.filled])


def test_library_plan_checks():
    """Models the plan does not list and T_eff' other than planned (beyond a retry nudge) raise; unplanned='skip'
    leaves the extra models out; a nudge across the bin edge keeps the model in its planned node (nudged='keep',
    default; the node mean includes the nudge), leaves it out ('drop') or raises ('raise'); keeping it raises when the
    node T_eff' would no longer increase."""
    grid = tt.TOY_GRID
    ls = tt.toy_lineset(2)
    pars = tt.toy_line_params(ls, 1)
    plan = lm.plan_teff_nodes((37900.0, 38100.0), dT=20.0, replicas=1, margin=0.0)
    base = _grid_store(plan.idx, plan.teff, grid, ls, pars)
    lm.flux_library_from_models(base, grid, ls, plan=plan)
    # one more model, not planned
    extra = _grid_store(np.r_[plan.idx, 999], np.r_[plan.teff, 38000.0], grid, ls, pars)
    with pytest.raises(ValueError, match="not in the plan"):
        lm.flux_library_from_models(extra, grid, ls, plan=plan)
    with pytest.warns(UserWarning, match="left out"):
        f = lm.flux_library_from_models(extra, grid, ls, plan=plan, unplanned="skip")
    assert f.count.sum() == plan.n
    # a retry nudge of +1 K stays in the bin; -5 K is not a nudge
    for dt, err in ((1.0, None), (-5.0, "other than planned")):
        t = plan.teff.copy()
        t[2] += dt
        s = _grid_store(plan.idx, t, grid, ls, pars)
        if err is None:
            lm.flux_library_from_models(s, grid, ls, plan=plan)
        else:
            with pytest.raises(ValueError, match=err):
                lm.flux_library_from_models(s, grid, ls, plan=plan)
    # +10 K (ten retries) reaches the next bin: kept in its node, dropped, or an error
    t = plan.teff.copy()
    t[2] += 10.0
    s = _grid_store(plan.idx, t, grid, ls, pars)
    with pytest.warns(UserWarning, match="kept in their planned node"):
        fk = lm.flux_library_from_models(s, grid, ls, plan=plan, prof_dtype=np.float64)
    assert (fk.count == 1).all() and fk.params["n_nudged_kept"] == 1 and fk.params["model_bin"][2] == 2
    np.testing.assert_allclose(fk.tmean, t, rtol=1e-14)
    np.testing.assert_allclose(fk.prof[2], _expected(t[2:3], grid, pars)[0], rtol=0, atol=1e-10)
    with pytest.warns(UserWarning, match="left out"):
        fd = lm.flux_library_from_models(s, grid, ls, plan=plan, nudged="drop")
    assert fd.count[2] == 0 and fd.count.sum() == plan.n - 1
    with pytest.raises(ValueError, match="outside their planned node's bin"):
        lm.flux_library_from_models(s, grid, ls, plan=plan, nudged="raise")
    with pytest.raises(ValueError, match="nudged must be"):
        lm.flux_library_from_models(s, grid, ls, plan=plan, nudged="move")
    # dT 5 K: a +6 K nudge passes the next node, keeping it would make the node T_eff' decrease
    p5 = lm.plan_teff_nodes((37950.0, 38000.0), dT=5.0, margin=0.0)
    t5 = p5.teff.copy()
    t5[3] += 6.0
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="non-increasing"):
        lm.flux_library_from_models(_grid_store(p5.idx, t5, grid, ls, pars), grid, ls, plan=p5)
    with pytest.raises(ValueError, match="differ"):
        lm.flux_library_from_models(base, grid, LineSet(["a", "b"], ls.lref), plan=plan)
    with pytest.raises(ValueError, match="give a plan"):
        lm.flux_library_from_models(base, grid, ls)


def test_nmin1_keeps_every_node():
    """lib_nodes(nmin=1) keeps every node of a library-mode library; the per-point default nmin=20 would merge 20
    single-model nodes into one (the pitfall the module notes describe)."""
    grid = tt.TOY_GRID
    ls = tt.toy_lineset(1)
    pars = tt.toy_line_params(ls, 2)
    plan = lm.plan_teff_nodes((37000.0, 39000.0), dT=50.0)
    f = lm.flux_library_from_models(_grid_store(plan.idx, plan.teff, grid, ls, pars), grid, ls, plan=plan)
    assert f.filled.sum() == plan.nn == lm.library_nodes(f).nn == lb.lib_nodes(f, nmin=1).nn
    assert lb.lib_nodes(f, nmin=20).nn == plan.nn // 20
    assert dm.flux_integrator(f, nmin=1, grid=grid).nn == plan.nn
    np.testing.assert_allclose(lm.library_nodes(f).t, plan.node_teff, atol=1e-9)


def test_node_representatives():
    """The intensity library's representatives come from the flux library's models in its bins: an unplanned model
    (unplanned='skip') and a replica nudged across its bin edge (kept in the flux node) are never representatives;
    ties (replicas at node -+ dT/4) go to the smaller idx; without the flux library's model list the bins of the
    T_eff' decide; a node without a candidate raises unless allow_missing."""
    grid, ls = tt.TOY_GRID, tt.toy_lineset(1)
    pars = tt.toy_line_params(ls, 3)
    plan = lm.plan_teff_nodes((37900.0, 38100.0), dT=20.0, replicas=2, margin=0.0)       # replicas at node -+ 5 K
    t = plan.teff.copy()
    k = int(np.flatnonzero((plan.node == 2) & (plan.replica == 1))[0])
    t[k] += 6.0                                                                         # node + 11 K: next bin
    idx, teff = np.r_[plan.idx, 999], np.r_[t, plan.node_teff[1]]                      # 999 at node 1's mean
    st = _grid_store(idx, teff, grid, ls, pars)
    with pytest.warns(UserWarning):
        f = lm.flux_library_from_models(st, grid, ls, plan=plan, unplanned="skip", prof_dtype=np.float64)
    assert 999 not in f.params["model_idx"] and f.count[2] == 2 and f.tmean[2] == pytest.approx(plan.node_teff[2] + 3)
    cands = [(int(i), float(x), "") for i, x in zip(idx, teff)]
    reps = lm.node_representatives(f, cands)
    assert len(reps) == plan.nn and 999 not in reps.idx.tolist()
    for b, i, x, _ in reps:
        m = plan.node == b
        assert i == (plan.idx[m & (plan.replica == 0)][0] if b == 2 else plan.idx[m].min()), (b, i)
    bare = dict(edges=f.edges, tmean=f.tmean, count=f.count)
    assert 999 in lm.node_representatives(bare, cands).idx.tolist()
    only = [c for c in cands if c[0] != plan.idx[(plan.node == 2) & (plan.replica == 0)][0]]
    with pytest.raises(ValueError, match="no candidate"):
        lm.node_representatives(f, only)
    r3 = lm.node_representatives(f, only, allow_missing=True)
    assert r3.missing == [2] and len(r3) == plan.nn - 1


def test_no_plan_offset_warning():
    """Without a plan, bins at a wrong offset mix the replicas of neighbouring nodes: a warning (with the offset the
    models suggest); the right offset (or the plan) gives one node per bin and no warning."""
    grid, ls = tt.TOY_GRID, tt.toy_lineset(1)
    pars = tt.toy_line_params(ls, 5)
    plan = lm.plan_teff_nodes((37900.0, 38100.0), dT=10.0, replicas=3, offset=5.0, margin=0.0)
    st = _grid_store(plan.idx, plan.teff, grid, ls, pars)
    with pytest.warns(UserWarning, match="suggest offset"):
        f0 = lm.flux_library_from_models(st, grid, ls, dT=10.0, offset=0.0)
    assert not (f0.count[f0.filled] == 3).all()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f5 = lm.flux_library_from_models(st, grid, ls, dT=10.0, offset=5.0)
        fp = lm.flux_library_from_models(st, grid, ls, plan=plan)
    assert (f5.count == 3).all() and np.array_equal(f5.edges, fp.edges) and np.array_equal(f5.prof, fp.prof)
    p1 = lm.plan_teff_nodes((37900.0, 38100.0), dT=10.0, offset=5.0, margin=0.0)      # one model per node
    with pytest.warns(UserWarning, match="suggest offset 5 K"):
        lm.flux_library_from_models(_grid_store(p1.idx, p1.teff, grid, ls, pars), grid, ls, dT=10.0)


# ------------------------------------------------------------------------------------------------------------------
# end to end with the fake FASTWIND
# ------------------------------------------------------------------------------------------------------------------
def _fake_profile(name, teff, lam):
    """The fake pformalsol's profile (fastwind.fake.profile), on its rows."""
    digits = "".join(ch for ch in name if ch.isdigit())
    lam0 = float(digits)
    depth = 0.3 * (38230.0 / teff) * (1.0 + 0.1 * (len(digits) % 3))
    return 1.0 - depth * np.exp(-((lam - lam0) / 1.5) ** 2), 1.0e-6 * (teff / 38230.0) ** 4


def test_fake_fastwind_library_mode(tmp_path):
    """Plan -> runner (fake FASTWIND, intensity build) -> merge_task -> combine -> library_from_models with the
    intensity library from the extracted OUT_IMU files: the flux library equals the replica means of the fake's
    profiles (to its print precision), a failed model is retried through missing.txt, DiscFlux and DiscImu integrate,
    and the saved libraries load into the dumps factories."""
    inst = fake.install_fake(str(tmp_path / "fw"), python=sys.executable)
    imu_inst = FastwindInstall(inst.root, fake.FAKE_BUILD, formal_build=fake.FAKE_IMU_BUILD)
    tpl, formal = os.path.join(inst.root, "INDAT.template"), os.path.join(inst.root, "FORMAL_INPUT")
    plan = lm.plan_teff_nodes((37400.0, 38600.0), dT=200.0, replicas=2, margin=0.0, start_idx=10, replica_step=1.0)
    assert plan.nn == 7 and plan.n == 14
    fake.set_fake_config(inst.build, fail=[plan.texts[5]])
    run = tmp_path / "run"
    plan.write(str(run / "points.txt"))
    pz = plan.write_points_npz(str(run / "points.npz"))
    t0 = time.time()
    s = batch.run_models(str(run / "points.txt"), imu_inst, tpl, formal, str(run / "results"), nworkers=4,
                         local_root=str(tmp_path / "local"), pack_interval=5, log=_quiet)
    assert s["exit_code"] == 0 and s["status"] == {"ok": 13, "pnlte_failed": 1}, s
    merge_task(str(run / "results"), "task_0000", str(run / "merged" / "task_0000.npz"), pz, copy_keys=())
    combine([str(run / "merged" / "task_0000.npz")], str(run / "points.txt"), str(run))
    miss = np.loadtxt(str(run / "missing.txt"), ndmin=2)
    assert miss.shape == (1, 2) and miss[0, 0] == plan.idx[5] and miss[0, 1] == plan.teff[5] + 1.0
    # the retry round (missing.txt as the list, as for a per-point run); its +1 K stays in the node's bin
    fake.set_fake_config(inst.build)
    s2 = batch.run_models(str(run / "missing.txt"), imu_inst, tpl, formal, str(run / "results"), nworkers=2,
                          local_root=str(tmp_path / "local"), list_name="missing", log=_quiet)
    assert s2["exit_code"] == 0 and s2["status"] == {"ok": 1}
    for tag in ("task_0000", "task_missing_0000"):
        merge_task(str(run / "results"), tag, str(run / "merged" / (tag + ".npz")), pz, copy_keys=())
    combine(sorted(str(p) for p in (run / "merged").glob("task_*.npz")), str(run / "points.txt"), str(run))
    assert os.path.getsize(str(run / "missing.txt")) == 0
    grid = tt.TOY_GRID
    ls = LineSet(LINES, [4026.0, 4200.0, 4922.0])
    with pytest.warns(UserWarning, match="replicas average only the flux library"):
        nl = lm.library_from_models(str(run / "profiles.npz"), grid, ls, plan=pz, results_dir=str(run / "results"),
                                    extract_dir=str(run / "imu_models"))
    print("fake library-mode run: {:.1f} s".format(time.time() - t0))
    f = nl.flux
    assert f.nb == plan.nn and f.filled.all() and (f.count == 2).all() and nl.nodes.nn == plan.nn
    store = ProfileStore.open(str(run / "profiles.npz"))
    tused = np.asarray(store.teff)
    assert np.isclose(tused[np.asarray(store.idx) == plan.idx[5]][0], plan.teff[5] + 1.0)
    for k in range(plan.nn):
        ts = tused[np.isin(np.asarray(store.idx), plan.idx[plan.node == k])]
        np.testing.assert_allclose(f.tmean[k], ts.mean(), rtol=1e-14)
        for j, name in enumerate(LINES):
            lam_rows = float("".join(ch for ch in name if ch.isdigit())) + 0.5 * (np.arange(1, 162) - 81)
            lam_grid = ls.lref[j] * np.exp(grid.y / C_KMS)
            prof = np.mean([np.interp(lam_grid, lam_rows, _fake_profile(name, t, lam_rows)[0]) for t in ts], axis=0)
            np.testing.assert_allclose(f.prof[k, j], prof, rtol=0, atol=1e-5)
            np.testing.assert_allclose(f.fc[k, j], np.mean([_fake_profile(name, t, 0.0)[1] for t in ts]), rtol=1e-6)
    # the intensity library: one representative per node (closest to the node's mean T_eff'; two replicas equally
    # close: the smaller idx), replicas not averaged
    assert nl.imu is not None and nl.report["imu"] and nl.report["imu_nodes"] == plan.nn
    assert len(nl.representatives) == plan.nn and np.isin(nl.representatives.idx, plan.idx).all()
    assert nl.report["imu_replicas_averaged"] is False and nl.imu.params["max_models_per_node"] == 2
    for b, i, t, _ in nl.representatives:
        ts = tused[np.isin(np.asarray(store.idx), plan.idx[plan.node == b])]
        assert abs(t - f.tmean[b]) == pytest.approx(np.min(np.abs(ts - f.tmean[b])), abs=1e-9)
        if b != plan.node[5]:                               # the node of the retried (+1 K) replica
            assert i == plan.idx[plan.node == b].min()
    assert all(os.path.isdir(d) for d in nl.representatives.dirs)
    # integrate a toy dump
    sph = tt.toy_sphere(4000, seed=1, dump=1, teff0=38000.0, teff_rel_rms=0.004, plume=False)
    mu, tn, pn = __import__("ppmpy.synspec.sphere", fromlist=["project_los"]).project_los(
        sph["theta"], sph["phi"], "thompson2024")
    rf = dm.disc_dump(nl.flux_integrator(), sph, mu[:2], tn[:2], pn[:2])
    ri = dm.disc_dump(nl.imu_integrator(fft="lazy"), sph, mu[:2], tn[:2], pn[:2], batch=None)
    for r in (rf, ri):
        assert np.isfinite(r["F"]).all() and r["n_lo"] == r["n_hi"] == 0
    # a uniform star at a node's T_eff' without velocities: I_line / I_cont of the fake does not depend on mu, so the
    # intensity method gives the flux method's node profile up to the replica mean (+-0.5 K) and the print precision
    uni = dict(sph, teff=np.full(sph["teff"].size, plan.node_teff[3]), ur=0 * sph["ur"], uth=0 * sph["ur"],
               uph=0 * sph["ur"])
    uf = dm.disc_dump(nl.flux_integrator(), uni, mu[:2], tn[:2], pn[:2])
    ui = dm.disc_dump(nl.imu_integrator(fft="lazy"), uni, mu[:2], tn[:2], pn[:2], batch=None)
    np.testing.assert_allclose(uf["F"], np.broadcast_to(f.prof[3].astype(np.float64), uf["F"].shape), atol=1e-12)
    assert np.abs(ui["F"] - uf["F"]).max() < 5e-5 and np.abs(ui["F0"] - uf["F0"]).max() < 5e-5
    # save and load: the dumps factories take the files
    paths = nl.save(str(tmp_path / "lib"))
    g = dm.flux_integrator(paths["flux"], nmin=1, grid=grid)
    assert g.nn == plan.nn and np.array_equal(g.t, nl.nodes.t)
    gi = dm.imu_integrator(paths["imu"], fft="lazy")
    assert gi.nn == plan.nn
    rec = json.load(open(paths["report"]))
    assert rec["nmin"] == 1 and rec["plan"]["nn"] == plan.nn and rec["report"]["models"] == plan.n
    assert lb.Representatives.read(paths["representatives"]).idx.tolist() == nl.representatives.idx.tolist()
    # a source of OUT_IMU files that has none
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="no node model has"):
        lm.imu_library_from_models(f, store, grid=grid, lref=ls, imu_runs=str(tmp_path / "nothing"))


# ------------------------------------------------------------------------------------------------------------------
# V8 on a toy run
# ------------------------------------------------------------------------------------------------------------------
def _toy_v8(root, noise, seed=3):
    """A toy per-point run (nnode 40, 6000 points) and three later dumps whose T_eff' lies inside the library's range
    (toy_sphere without the plume, rms 0.4 %: no point is clamped, so V8 sees the node spacing only), with their
    products computed by the toy's own integrator (the reference)."""
    from ppmpy.synspec.sphere import project_los
    run = tt.toy_run(str(root), n=6000, nnode=40, seed=seed, ndumps=1, noise=noise)
    mu, tn, pn = project_los(run["theta"], run["phi"], run["los"])
    smp = {d: tt.toy_sphere(6000, seed=seed, dump=d, teff_rel_rms=0.004, plume=False) for d in (11, 12, 13)}
    lo, hi = run["teff_range"]
    assert all(lo + run["dT"] < s["teff"].min() and s["teff"].max() < hi - run["dT"] for s in smp.values())
    ref = {d: dm.disc_dump(run["integ"], smp[d], mu, tn, pn) for d in smp}
    return dict(run, smp=smp, ref=ref)


@pytest.fixture(scope="module")
def toyrun(tmp_path_factory):
    return _toy_v8(tmp_path_factory.mktemp("v8toy"), noise=0.0)


def test_v8_toy(toyrun, tmp_path):
    """V8 on a toy run without model noise: the reference row (the toy's integrator rebuilt from its library file)
    reproduces the reference products; the sparse libraries' max|dF| decreases with denser nodes (linear T_eff'
    interpolation error ~ dT^2, above the toy library's own: 3.6e-4, 7.6e-5, 1.8e-5 for 400, 200, 100 K); the
    table, report and JSON work. (With the toy's own later dumps, whose plume puts points beyond the library, the
    end nodes dominate: a single model sits at the node, the per-point library's end node at its bin's mean.)"""
    lpv = va.lpv_residual_rms(toyrun["timeseries"])
    res = lm.sparse_library_test(toyrun["profiles"], toyrun["smp"], sorted(toyrun["smp"]), toyrun["ref"],
                                 grid=toyrun["grid"], lref=toyrun["lines"], lpv=lpv, dTs=(400.0, 200.0, 100.0),
                                 replicas=(1, 3), offset=0.0, reference_library=toyrun["library"],
                                 reference_nmin=toyrun["nmin"])
    print(res.table())
    ref = res.row("reference")
    assert max(ref["max_dF"]) <= 2.0 ** -25 and max(ref["max_dF0"]) <= 2.0 ** -25
    e = [max(res.row("dT{:g}_R1".format(d))["max_dF"]) for d in (400.0, 200.0, 100.0)]
    assert e[0] > e[1] > e[2] > 10 * max(ref["max_dF"]) and e[0] > 4 * e[2], e
    for d in (400.0, 200.0, 100.0):
        r1, r3 = res.row("dT{:g}_R1".format(d)), res.row("dT{:g}_R3".format(d))
        assert r3["models"] == 3 * r1["models"] and r1["n_filled"] == r3["n_filled"] == r1["n_planned"]
        assert r1["n_lo"] == r1["n_hi"] == 0
        np.testing.assert_allclose(r1["ratio_F"], np.asarray(r1["max_dF"]) / lpv["rms"])
        assert all(np.isfinite(r1[k]).all() for k in ("ratio_dR", "ratio_EW", "dR_rms"))
    rec = res.recommend["flux"]
    assert rec is not None and rec["ratio"] <= 0.05 and rec["label"] == "dT400_R1"
    rep = res.to_report()
    assert "V8_flux_dT100_R1" in rep.names() and rep["V8_flux_dT100_R1"].details.get("lpv_ratio") is not None
    back = lm.SparseLibraryTest.from_json(res.to_json(str(tmp_path / "v8.json")))
    assert back.table() == res.table() and [r["label"] for r in back.rows] == [r["label"] for r in res.rows]
    assert back.recommend == res.recommend
    # the toy's own dumps through the stored per-dump files (a directory reference; the plume dump 1 is clamped)
    res2 = lm.sparse_library_test(toyrun["profiles"], toyrun["samples"], toyrun["dumps"],
                                  os.path.join(toyrun["root"], "flux"), grid=toyrun["grid"], lref=toyrun["lines"],
                                  dTs=(200.0,), replicas=(1,), reference_library=toyrun["library"],
                                  reference_nmin=toyrun["nmin"])
    assert max(res2.row("reference")["max_dF"]) <= 2.0 ** -25 and res2.row("dT200_R1")["n_lo"] > 0


def test_v8_toy_replicas(tmp_path):
    """V8 on a toy run whose models carry a random depth scatter (0.3 %, FASTWIND's convergence noise was 0.5 % in
    EW): more replicas per node average it (dT 100 K: max|dF| 6.8e-4, 2.3e-4, 1.5e-4 for 1, 3, 9 replicas)."""
    t = _toy_v8(tmp_path, noise=0.003)
    res = lm.sparse_library_test(t["profiles"], t["smp"], sorted(t["smp"]), t["ref"], grid=t["grid"], lref=t["lines"],
                                 dTs=(100.0,), replicas=(1, 3, 9, (3, "bin"), (3, 2.0)))
    print(res.table())
    e = [max(res.row("dT100_R{}".format(r))["max_dF"]) for r in (1, 3, 9)]
    assert e[0] > e[1] > e[2] and e[2] < 0.5 * e[0], e
    rb, r2 = res.row("dT100_R3s33.3"), res.row("dT100_R3s2")
    assert rb["replica_step"] == pytest.approx(100.0 / 3) and r2["replica_step"] == 2.0 and rb["models"] == r2["models"]
    assert rb["max_dist"] < 1.0 and max(rb["max_dF"]) < e[0]
    with pytest.raises(ValueError, match="leave the bin"):
        lm.sparse_library_test(t["profiles"], t["smp"], [11], t["ref"], grid=t["grid"], lref=t["lines"],
                               dTs=(100.0,), replicas=((3, 60.0),))
    assert res.recommend["flux"] is None and np.isnan(res.row("dT100_R1")["ratio_F"]).all()     # no LPV given


def test_v8_errors(toyrun):
    with pytest.raises(ValueError, match="criterion"):
        lm.sparse_library_test(toyrun["profiles"], toyrun["samples"], [1], os.path.join(toyrun["root"], "flux"),
                               grid=toyrun["grid"], lref=toyrun["lines"], criterion="median")
    with pytest.raises(ValueError, match="no dumps"):
        lm.sparse_library_test(toyrun["profiles"], toyrun["samples"], [], os.path.join(toyrun["root"], "flux"),
                               grid=toyrun["grid"], lref=toyrun["lines"])
    with pytest.raises(ValueError, match="phases"):
        lm.sparse_library_test(toyrun["profiles"], toyrun["samples"], [1], os.path.join(toyrun["root"], "flux"),
                               grid=toyrun["grid"], lref=toyrun["lines"], phases=(0.0, 1.0))
    with pytest.raises(ValueError, match="phase_stat"):
        lm.sparse_library_test(toyrun["profiles"], toyrun["samples"], [1], os.path.join(toyrun["root"], "flux"),
                               grid=toyrun["grid"], lref=toyrun["lines"], phase_stat="median")


def test_recommend_rules():
    """The recommendation skips the reference (dT None), check rows, the per-phase rows of an aggregated variant and
    intensity-method rows with replicas; with phases it uses the worst phase ('max') or the phase mean ('mean')."""
    b = dict(method="flux", replicas=1, ratio_dR_max=[0.01, 0.0], ratio_EW=[0.01, 0.02])
    rows = [dict(b, label="reference", dT=None, models=None),
            dict(b, label="check", dT=10.0, models=100, check=True),
            dict(b, label="a", dT=20.0, models=200),
            dict(b, label="c_o0", dT=40.0, models=50, n_phases=2),
            dict(b, label="c_o20", dT=40.0, models=50, n_phases=2, ratio_EW=[0.09, 0.0]),
            dict(b, label="c", dT=40.0, models=50, n_phases=2, aggregate=True, ratio_EW=[0.09, 0.02],
                 ratio_EW_mean=[0.05, 0.02], ratio_dR_max_mean=[0.01, 0.0]),
            dict(b, method="imu", label="i3", dT=50.0, models=30, replicas=3),
            dict(b, method="imu", label="i1", dT=10.0, models=300)]
    r = lm._recommend(rows, "lpv", 0.05)
    assert r["flux"]["label"] == "a" and r["imu"]["label"] == "i1"
    assert r["flux"]["ratio"] == pytest.approx(0.02) and r["flux"]["margin"] == pytest.approx(0.03)
    r2 = lm._recommend(rows, "lpv", 0.05, phase_stat="mean")
    assert r2["flux"]["label"] == "c" and r2["flux"]["ratio"] == pytest.approx(0.05)
    assert r2["flux"]["ratio_worst"] == pytest.approx(0.09) and r2["flux"]["phase_stat"] == "mean"
    assert lm._recommend([rows[1]], "lpv", 0.05) == {"flux": None}


def _small_pool(n=16, dT=20.0, t0=37850.0, seed=7):
    """A per-point pool of n models every dT K from t0 (analytic profiles on the grid points; inside the dumps' T_eff'
    range, as the M424 intensity pool lies inside the planned nodes) and three dumps integrated with the pool's own
    library (one node per model, nmin 1): the reference."""
    from ppmpy.synspec.sphere import project_los
    grid, ls = tt.TOY_GRID, tt.toy_lineset(3)
    pars = tt.toy_line_params(ls, 0)
    store = _grid_store(np.arange(n), t0 + dT * np.arange(n), grid, ls, pars)
    lib = lm.flux_library_from_models(store, grid, ls, dT=dT, offset=float(np.mod(t0, dT)))       # models at nodes
    integ = dm.flux_integrator(lib, nmin=1, grid=grid)
    smp = {d: tt.toy_sphere(3000, seed=seed, dump=d, teff0=38000.0, teff_rel_rms=0.004, plume=False) for d in (1, 2, 3)}
    mu, tn, pn = project_los(smp[1]["theta"], smp[1]["phi"], "thompson2024")
    return dict(store=store, grid=grid, ls=ls, smp=smp, theta=smp[1]["theta"], phi=smp[1]["phi"],
                ref={d: dm.disc_dump(integ, smp[d], mu, tn, pn) for d in smp})


def test_v8_check_rows_not_recommended():
    """A variant whose selection is the whole pool (here dT 20 K = the pool spacing, both phases; M424: the imu dT 10 K
    selection of the production representatives) reproduces the reference exactly, is marked as a check and is never
    recommended, even when it alone passes; the second phase reuses the first one's comparison (same node models),
    and without dedupe gives the same numbers."""
    t = _small_pool()
    kw = dict(theta=t["theta"], phi=t["phi"], grid=t["grid"], lref=t["ls"], dTs=(20.0, 40.0), replicas=(1,),
              phases=(0.0, 0.5))
    tiny = dict(rms=np.full(3, 1e-12), ew_rms=np.full(3, 1e-12))         # only an exact library passes
    res = lm.sparse_library_test(t["store"], t["smp"], [1, 2, 3], t["ref"], lpv=tiny, **kw)
    print(res.table())
    c0, c1, agg = res.row("dT20_R1_o0"), res.row("dT20_R1_o10"), res.row("dT20_R1")
    assert c0["check"] and c1["check"] and agg["check"] and agg["n_checks"] == 2
    assert c0["pool_models"] == 16 and max(c0["max_dF"]) == 0.0 and c0["same_as"] is None
    assert c1["same_as"] == "dT20_R1_o0" and c1["max_dF"] == c0["max_dF"] and c1["offset"] == 10.0
    for lab in ("dT40_R1_o0", "dT40_R1_o20"):
        r = res.row(lab)
        assert not r["check"] and 0 < r["pool_models"] < 16 and max(r["max_dF"]) > 0
    assert res.row("dT40_R1_o0")["max_dF"] != res.row("dT40_R1_o20")["max_dF"]
    assert res.recommend["flux"] is None
    nod = lm.sparse_library_test(t["store"], t["smp"], [1, 2, 3], t["ref"], lpv=tiny, dedupe=False, **kw)
    assert nod.row("dT20_R1_o10")["same_as"] is None and nod.row("dT20_R1_o10")["max_dF"] == c1["max_dF"]
    big = dict(rms=np.full(3, 1.0), ew_rms=np.full(3, 1.0))
    assert lm.sparse_library_test(t["store"], t["smp"], [1, 2, 3], t["ref"], lpv=big,
                                  **kw).recommend["flux"]["label"] == "dT40_R1"


def test_v8_toy_phases(toyrun):
    """Node phases on the toy run: one row per phase and an aggregate row (per line the largest, smallest and mean
    over the phases), the deviation depends on where the nodes fall, the recommendation uses the aggregate (worst
    phase), and the table shows the range over the phases."""
    lpv = va.lpv_residual_rms(toyrun["timeseries"])
    res = lm.sparse_library_test(toyrun["profiles"], toyrun["smp"], sorted(toyrun["smp"]), toyrun["ref"],
                                 grid=toyrun["grid"], lref=toyrun["lines"], lpv=lpv, dTs=(400.0, 200.0),
                                 replicas=(1,), offset=0.0, phases=4)
    print(res.table())
    agg = res.row("dT400_R1")
    ph = [res.row(lab) for lab in agg["phase_labels"]]
    assert agg["aggregate"] and agg["n_phases"] == 4 and agg["offsets"] == [0.0, 100.0, 200.0, 300.0]
    assert [r["label"] for r in ph] == ["dT400_R1_o0", "dT400_R1_o100", "dT400_R1_o200", "dT400_R1_o300"]
    for k in ("max_dF", "ratio_EW", "ratio_dR_max", "ratio_F0"):
        A = np.array([r[k] for r in ph])
        np.testing.assert_allclose(agg[k], A.max(axis=0), rtol=0, atol=0)
        np.testing.assert_allclose(agg[k + "_min"], A.min(axis=0), rtol=0, atol=0)
        np.testing.assert_allclose(agg[k + "_mean"], A.mean(axis=0), rtol=1e-15)
    e = [max(r["max_dF"]) for r in ph]
    print("dT 400 K, max|dF| per phase:", e)
    assert max(e) > 1.05 * min(e)
    rec = res.recommend["flux"]
    assert rec is not None and rec["phase_stat"] == "max" and rec["n_phases"] == 4
    assert rec["label"] in ("dT400_R1", "dT200_R1")
    assert rec["ratio"] == pytest.approx(lm._crit(res.row(rec["label"]), lm._CRITERIA["lpv"], "max"))
    assert "-" in res.table(detail=False).splitlines()[2].split()[4]
    assert len(res.table(detail=False).splitlines()) < len(res.table().splitlines())


# ------------------------------------------------------------------------------------------------------------------
# M424
# ------------------------------------------------------------------------------------------------------------------
def _m424_samples():
    root = os.environ.get("PPMPY_SYNSPEC_M424_SAMPLES", "/scratch/ppathak/fastwind_sphere/samples_r4050_N1236544")
    if not os.path.isdir(root):
        pytest.skip("M424 samples not available: {}".format(root))
    return root


@pytest.mark.m424
@pytest.mark.slow
def test_m424_v8_subset():
    """V8 on three dumps of the subset (time-boxed; the 16-dump table with LPV ratios and node phases is the M7
    scratch run): the reference row (library_dT10.npz, nmin 20) reproduces the stored flux products to their float32
    rounding; dT 10 K with one model per node is closer to the full library than dT 50 K; the planned nodes cover
    more than the pool, the filled ones start and end within one bin of the pool's T_eff' range."""
    prof = m424_path("run", "profiles.npz")
    lib = m424_path("run", "library_dT10.npz")
    ref = m424_path("disc", "flux")
    t0 = time.time()
    res = lm.sparse_library_test(prof, _m424_samples(), (3200, 4000, 4800), ref, lref=LINESET, dTs=(10.0, 50.0),
                                 replicas=(1,), offset=5.0, reference_library=lib, log=print)
    print(res.table(), "\n{:.0f} s".format(time.time() - t0))
    assert max(res.row("reference")["max_dF"]) <= 2.0 ** -25
    e10, e50 = max(res.row("dT10_R1")["max_dF"]), max(res.row("dT50_R1")["max_dF"])
    assert 0 < e10 < e50 < 1e-2
    pool = res.params["pool_range"]
    assert pool == pytest.approx([35402.107, 38905.438], abs=1e-3)
    for lab in ("dT10_R1", "dT50_R1"):
        r = res.row(lab)
        fr, dT = r["filled_range"], r["dT"]
        # the planned nodes reach beyond the pool (the subset's T_eff' plus dT); the filled nodes start and end
        # within one bin of the pool's range
        assert r["node_range"][0] < pool[0] and r["node_range"][1] > pool[1] and r["n_filled"] < r["n_planned"]
        assert pool[0] <= fr[0] < pool[0] + dT and pool[1] - dT < fr[1] <= pool[1] and not r["check"]


def _real():
    root = os.environ.get("PPMPY_FASTWIND_ROOT", "/scratch/ppathak/FW_10.6.4.1")
    if not (os.path.isdir(root) and os.path.isdir("/cvmfs")):
        pytest.skip("FASTWIND install or /cvmfs not available")
    inst = FastwindInstall(root, "v10.6_HHe", formal_build="v10.6_HHe_imu")
    probs = inst.check()
    if probs:
        pytest.skip("FASTWIND not runnable here: " + "; ".join(probs))
    return inst


@pytest.mark.fastwind
@pytest.mark.slow
@pytest.mark.m424
def test_real_fastwind_library_mode():
    """Real FASTWIND (2 models at once, ~5 min): a library-mode plan whose two nodes are the T_eff' of the production
    per-point models 9729 and 1652 (the intensity-library representatives of two 10 K bins), run with the intensity
    formal build, merged and combined, reproduces their production profiles (lam, fcont, fnorm of profiles.npz) bit
    for bit, and library_from_models' intensity library (OUT_IMU extracted from the packed results) equals the rows
    of the production imu_library_dT10.npz (Ic, Il, s, rmax, nnode) bit for bit."""
    inst = _real()
    prof = m424_path("run", "profiles.npz")
    imu_prod = os.environ.get("PPMPY_SYNSPEC_M424_IMU", "/scratch/ppathak/fastwind_imu") + "/imu_library_dT10.npz"
    if not os.path.exists(imu_prod):
        pytest.skip("not available: " + imu_prod)
    project = os.environ.get("PPMPY_SYNSPEC_PROJECT_ANALYSIS",
                             "/home/ppathak/stellar-atmosphere-KU-Leuven/project/analysis")
    tpl = os.path.join(project, "fastwind", "INDAT_M424test.DAT")
    formal = os.path.join(project, "fastwind", "FORMAL_INPUT_He3")
    for p in (tpl, formal):
        if not os.path.exists(p):
            pytest.skip("not available: " + p)
    with open(m424_path("run", "points.txt")) as f:
        txt = {int(t[0]): t[1] for t in (ln.split() for ln in f) if int(t[0]) in (9729, 1652)}
    idx = [9729, 1652]
    teff = [float(txt[i]) for i in idx]
    plan = lm.NodePlan(idx, teff, [0, 1], [0, 0], teff, teff[1] - teff[0], offset=teff[0])
    assert plan.texts == [txt[i] for i in idx]
    w = os.path.join(SHADOW, "pytest_real_{}".format(os.getpid()))
    shutil.rmtree(w, ignore_errors=True)
    os.makedirs(w)
    plan.write(os.path.join(w, "points.txt"))
    pz = plan.write_points_npz(os.path.join(w, "points.npz"))
    t0 = time.time()
    s = batch.run_models(os.path.join(w, "points.txt"), inst, tpl, formal, os.path.join(w, "results"), nworkers=2,
                         local_root=os.path.join(w, "local"), log=print)
    print("2 real models: {:.0f} s".format(time.time() - t0))
    assert s["exit_code"] == 0 and s["status"] == {"ok": 2}, s
    merge_task(os.path.join(w, "results"), "task_0000", os.path.join(w, "merged", "task_0000.npz"), pz, copy_keys=())
    combine([os.path.join(w, "merged", "task_0000.npz")], os.path.join(w, "points.txt"), w)
    new = ProfileStore.open(os.path.join(w, "profiles.npz"))
    old = ProfileStore.open(prof)
    for k in ("lam", "fcont", "fnorm"):
        a, b = np.asarray(new[k]), np.asarray(old[k][np.sort(idx)])
        assert np.array_equal(a, b), k
    nl = lm.library_from_models(new, VelocityGrid(), LINESET, plan=plan, results_dir=os.path.join(w, "results"),
                                extract_dir=os.path.join(w, "imu_models"))
    with np.load(imu_prod) as z:
        for b, i in enumerate(idx):
            # the bin of model i (empty neighbouring bins copy it: idx_rep equal, src pointing to this bin)
            pb = [int(x) for x in np.flatnonzero(z["idx_rep"] == i) if z["src"][x] == x]
            assert len(pb) == 1
            pb = pb[0]
            for k in ("Ic", "Il", "s", "rmax", "nnode"):
                assert np.array_equal(np.asarray(nl.imu[k])[b], z[k][pb], equal_nan=True), (i, k)
            assert nl.imu["teff_rep"][b] == z["teff_rep"][pb]
    for _ in range(5):                  # the runner's watchdog child may still be removing its local root
        shutil.rmtree(w, ignore_errors=True)
        if not os.path.exists(w):
            break
        time.sleep(1)
