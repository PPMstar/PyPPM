"""
Tests of the library extension (ppmpy.synspec.libmode.extend_flux_library, extend_imu_library, flux_seam_check,
imu_seam_check) and of lib_nodes' single bins (ppmpy.synspec.library).

Synthetic (any machine): a toy per-point base library (models on 161 FASTWIND-like rows, sparse tails that nmin merges)
and library-mode node models below, above and just inside its range: the base bins of the extended library are the base
library's bit for bit, the bins outside are the node library's bins (one per node) bit for bit, edges and node T_eff'
increase, the seam rule (whole node bins outside; overlapping nodes left out; gap bins for node bins that do not touch
the seam), lib_nodes merges only the base bins (the base's nodes bit for bit) and keeps every node bin, the saved file
keeps its single bins (dumps.flux_integrator), the seam check (exact for node models equal to the base bin's model,
detects a perturbed node; variant 'nodes' = the integrator's F0 of a uniform star), the errors. Review fixes: an
extension of an extension keeps every node bin of both steps (= the direct extension), a node FluxLibrary is checked
like the base (lref, lines, grid; array-valued lines; dtype warning), node means inside the base range raise (single
nudged models warn), the line order at the seam (continuum flux), stale single-bin records raise, lib_nodes' smoothing
stays on each side of the seam (base nodes bit for bit), a per-bin correction (extend_correction) gives the base's
corrected nodes bit for bit. Intensity library: toy OUT_IMU / OUT files; the extended intensity library has the base
rows bit for bit and the node rows of a library built over all models, DiscImu of both is the same; the chained
extension gives the same intensity library; without a model_idx record (a prebuilt node library) the same library;
the 'nodes' seam variant is exact at a representative.

M424 (marker m424, slow): the production library_dT10.npz shrunk to 37 600-38 700 K (bins 220-329), extended by the
per-point models closest to the 10 K node centres outside (one model per node, as a library-mode run): base bins and
base nodes bit for bit; the disc-integrated profiles of three dumps against the production flux products within the
V8 static floor (single-model nodes); the seam check of six nodes just inside. Intensity: the production
imu_library_dT10.npz shrunk the same way and extended by the production representatives outside reproduces the
production intensity library's nodes bit for bit and the stored imu products of dump 3200 to their float32 rounding.

PP 2026-10-02: new (library extension). PP 2026-10-02: review fixes (chained extensions, node-library checks, nudges
into the base range, line order, single-bin records, smoothing / correction with single bins, seam variant 'nodes';
M424 dF0 bound 3e-5).
"""
import os
import time
import warnings

import numpy as np
import pytest

from conftest import m424_path
from ppmpy.synspec import dumps as dm
from ppmpy.synspec import libmode as lm
from ppmpy.synspec import library as lb
from ppmpy.synspec import testing as tt
from ppmpy.synspec.conventions import C_KMS
from ppmpy.synspec.fwresults import ProfileStore
from ppmpy.synspec.spectral import LineSet, VelocityGrid

LINES = ["HEI4026", "HEII4200", "HEI4922"]
LREF = np.array([4026.22, 4199.90, 4921.93])
LINESET = LineSet(LINES, LREF)
NROW = 161


# ------------------------------------------------------------------------------------------------------------------
# synthetic models
# ------------------------------------------------------------------------------------------------------------------
def _rows_y(ny_half=1000.0):
    u = np.linspace(-1.0, 1.0, NROW)
    return ny_half * np.sign(u) * np.abs(u) ** 1.5


def _store(idx, teff, ls, pars, grid, scale=None):
    """A profiles store of toy models on NROW rows (testing.toy_depth / toy_continuum), float64; ``scale`` (n,)
    multiplies each model's depth (a perturbed model)."""
    teff = np.asarray(teff, dtype=np.float64)
    n, nl = teff.size, len(ls)
    yr = _rows_y()
    lam = np.empty((n, nl, NROW))
    fn = np.empty((n, nl, NROW))
    fc = np.empty((n, nl, NROW))
    sc = np.ones(n) if scale is None else np.asarray(scale, dtype=np.float64)
    for j, p in enumerate(pars):
        lam[:, j] = ls.lref[j] * np.exp(yr / C_KMS)[None, :]
        fn[:, j] = 1.0 - sc[:, None] * tt.toy_depth(teff, yr, p, grid)
        fc[:, j] = tt.toy_continuum(teff, p, np.broadcast_to(yr, (n, NROW)), grid)
    return dict(idx=np.asarray(idx, dtype=np.int32), teff=teff, status=np.full(n, "ok", dtype="<U12"),
                niter=np.full(n, 61, np.int16), lam=lam, fcont=fc, fnorm=fn, lines=np.array(ls.names),
                teff_nudge=np.zeros(n, np.float32))


def _merge(*stores):
    return {k: (np.concatenate([s[k] for s in stores]) if k != "lines" else stores[0][k]) for k in stores[0]}


@pytest.fixture(scope="module")
def toy():
    """A base per-point library (2400 models, 37 800-38 400 K, sparse tails: 10 K bins) and node plans."""
    grid, ls = tt.TOY_GRID, tt.toy_lineset(2)
    pars = tt.toy_line_params(ls, 4)
    rng = np.random.default_rng(3)
    t = 38100.0 + 90.0 * rng.standard_normal(2400)
    t = t[(t >= 37800.0) & (t < 38400.0)]
    t[0], t[1] = 37800.5, 38399.5                                     # the base range: 37 800-38 400 K
    st = _store(np.arange(t.size), t, ls, pars, grid)
    fc0 = st["fcont"][:, :, 0]
    base = lb.FluxLibrary.build(st["teff"], st["lam"], st["fnorm"], fc0, grid, ls, dT=10.0, block=500)
    return dict(grid=grid, ls=ls, pars=pars, base=base, base_store=st)


def _nodes(toy, tails=((37600.0, 37800.0), (38400.0, 38600.0)), seam=(37805.0, 37815.0, 38385.0, 38395.0),
           offset=5.0, dT=10.0, start=10 ** 5):
    """Node models (one per node) at offset + k dT in the tail ranges plus seam nodes inside the base range."""
    tn = []
    for lo, hi in tails:
        k0, k1 = np.ceil((lo - offset) / dT), np.floor((hi - offset) / dT)
        tn.append(offset + dT * np.arange(k0, k1 + 1))
    tn = np.unique(np.concatenate(tn + [np.asarray(seam, dtype=np.float64)]))
    return _store(start + np.arange(tn.size), tn, toy["ls"], toy["pars"], toy["grid"]), tn


def _quiet_extend(*a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lm.extend_flux_library(*a, **k)


# ------------------------------------------------------------------------------------------------------------------
# lib_nodes single
# ------------------------------------------------------------------------------------------------------------------
def test_lib_nodes_single_legacy(toy):
    """Without single bins (None on a library without the declaration, False, an all-False mask) lib_nodes is the
    legacy merging; with single bins every filled single bin is a node of its own, no node spans a single bin and every
    other node holds >= nmin models unless it is the leftover of its stretch."""
    base = toy["base"]
    ref = lb.lib_nodes(base, nmin=20)
    assert "single" not in ref.params and ref.nn < base.filled.sum()
    for s in (None, False, np.zeros(base.nb, bool), []):
        n = lb.lib_nodes(base, nmin=20, single=s)
        assert n.params == ref.params
        for k in ("t", "count", "prof", "fc"):
            assert np.array_equal(n[k], ref[k]), k
    rng = np.random.default_rng(0)
    for trial in range(5):
        mask = rng.uniform(size=base.nb) < 0.15
        n = lb.lib_nodes(base, nmin=20, single=mask)
        assert n.params["single"] == int(mask.sum())
        got = np.concatenate(n.groups)
        assert np.array_equal(np.sort(got), np.flatnonzero(base.count > 0))     # every filled bin in one node
        for g in n.groups:
            if mask[g].any():
                assert g.size == 1 and base.count[g[0]] > 0
            else:
                assert not mask[g[0]:g[-1] + 1].any()                         # no single bin inside a merged node
        assert np.all(np.diff(n.t) > 0)
    with pytest.raises(ValueError):
        lb.lib_nodes(base, single=np.zeros(3, bool))
    with pytest.raises(ValueError):
        lb.lib_nodes(base, single=[base.nb])


# ------------------------------------------------------------------------------------------------------------------
# the extended flux library
# ------------------------------------------------------------------------------------------------------------------
def test_extend_flux_aligned(toy, tmp_path):
    """Nodes at the base bins' centres (offset 5, dT 10): base bins bit for bit, node bins = the node library's bins
    bit for bit, no gap bins, seam nodes left out (warning, recorded), edges and node T_eff' increase, every node
    model's T_eff' falls into its node bin; lib_nodes(nmin 20) gives the base's nodes bit for bit plus one node per
    node bin (single=False would merge them); the saved file keeps the single bins (dumps.flux_integrator)."""
    grid, ls, base = toy["grid"], toy["ls"], toy["base"]
    nst, tn = _nodes(toy)
    with pytest.warns(UserWarning, match="overlap the base library's range"):
        ext = lm.extend_flux_library(base, nst, grid, ls, dT=10.0, offset=5.0)
    rec = ext.params[lm.EXTENSION_KEY]
    b0, b1, t_lo, t_hi = lm.base_range(base)
    assert (t_lo, t_hi) == (37800.0, 38400.0) and rec["base_range"] == [t_lo, t_hi]
    i0, i1 = rec["base_slice"]
    assert rec["gap_bins"] == [] and rec["seam_nodes"] == [37805.0, 37815.0, 38385.0, 38395.0]
    assert rec["n_low"] == 20 and rec["n_high"] == 20 and i0 == 20 and i1 - i0 == b1 - b0
    for k in ("edges", "tmean", "count", "prof", "fc"):
        a, b = getattr(ext, k), getattr(base, k)
        if k == "edges":
            assert np.array_equal(a[i0:i1 + 1], b[b0:b1 + 1])
        else:
            assert np.array_equal(a[i0:i1], b[b0:b1]) and a.dtype == b.dtype, k
    assert np.all(np.diff(ext.edges) > 0) and np.all(np.diff(ext.tmean[ext.filled]) > 0)
    # the node bins: the bins of the node library (flux_library_from_models of all node models)
    nlib = lm.flux_library_from_models(nst, grid, ls, dT=10.0, offset=5.0, prof_dtype=base.prof.dtype)
    out = np.r_[0:i0, i1:ext.nb]
    src = np.asarray(rec["node_source"])
    assert rec["node_bins"] == out.tolist()
    for k in ("tmean", "count", "prof", "fc"):
        assert np.array_equal(getattr(ext, k)[out], getattr(nlib, k)[src]), k
    assert np.array_equal(ext.edges[out], nlib.edges[src])
    outside = (tn < t_lo) | (tn >= t_hi)
    assert np.array_equal(ext.bin_index(tn[outside]), out)
    assert sorted(ext.params["model_idx"]) == sorted(nst["idx"][outside].tolist())
    assert ext.params[lb.SINGLE_BINS_KEY] == out.tolist()
    # nodes: the base's own nodes (nmin 20) bit for bit, and every node bin a node
    bn = lb.lib_nodes(base, nmin=20)
    en = lb.lib_nodes(ext, nmin=20)
    assert en.nn == bn.nn + out.size and en.params["single"] == out.size
    inb = np.array([g[0] >= i0 and g[-1] < i1 for g in en.groups])
    assert inb.sum() == bn.nn
    for k in ("t", "count", "prof", "fc"):
        assert np.array_equal(en[k][inb], bn[k]), k
    assert [g.tolist() for g in np.array(en.groups, dtype=object)[~inb]] == [[i] for i in out]
    assert lb.lib_nodes(ext, nmin=20, single=False).nn < en.nn                 # nmin would merge the node bins
    en2 = lm.extended_nodes(ext, nmin=20)
    assert np.array_equal(en2.t, en.t) and np.array_equal(en2.prof, en.prof)
    # save -> load -> flux_integrator: the same nodes
    path = ext.save(str(tmp_path / "ext.npz"))
    back = lb.FluxLibrary.load(path)
    assert back.params[lb.SINGLE_BINS_KEY] == out.tolist()
    integ = dm.flux_integrator(path, nmin=20, grid=grid)
    assert integ.nn == en.nn and np.array_equal(integ.t, en.t)
    np.testing.assert_allclose(integ.lref, ls.lref)
    with pytest.raises(ValueError, match="no single bins"):
        lm.extended_nodes(base)


def test_extend_flux_misaligned_gaps(toy):
    """Nodes not aligned with the base edges (offset 2): node bins that overlap the seam are left out entirely (also
    their part outside), gap bins (empty) fill the space to the seam, every T_eff' has exactly one bin; the node T_eff'
    still interpolate across the gap (lib_nodes skips the empty gap bins)."""
    grid, ls, base = toy["grid"], toy["ls"], toy["base"]
    nst, tn = _nodes(toy, tails=((37600.0, 37805.0), (38400.0, 38600.0)), offset=2.0, seam=())
    with pytest.warns(UserWarning, match="overlap"):
        ext = lm.extend_flux_library(base, nst, grid, ls, dT=10.0, offset=2.0)
    rec = ext.params[lm.EXTENSION_KEY]
    i0, i1 = rec["base_slice"]
    # node 37 802 K (bin 37 797-37 807 K) overlaps the seam at 37 800 K: left out; gap bin [37 797, 37 800)
    assert rec["seam_nodes"] == [37802.0, 38402.0] and len(rec["gap_bins"]) == 2
    g0, g1 = rec["gap_bins"]
    assert (ext.edges[g0], ext.edges[g0 + 1]) == (37797.0, 37800.0) and g0 == i0 - 1
    assert (ext.edges[g1], ext.edges[g1 + 1]) == (38400.0, 38407.0) and g1 == i1
    assert ext.count[g0] == ext.count[g1] == 0 and ext.tmean[g0] == 37798.5
    assert np.all(np.diff(ext.edges) > 0) and np.all(np.diff(ext.tmean[ext.filled]) > 0)
    used = (tn + 5.0 <= 37800.0) | (tn - 5.0 >= 38400.0)
    assert ext.count[rec["node_bins"]].sum() == used.sum() == len(ext.params["model_idx"])
    en = lb.lib_nodes(ext, nmin=20)
    assert not np.isin([g0, g1], np.concatenate(en.groups)).any()


def test_extend_flux_float64_and_errors(toy):
    """prof_dtype float64 widens the float32 base exactly; lossy dtypes, other lines / grids, no node outside and a
    node nudged across the seam (nudged='keep' would put its T_eff' into the base range) raise."""
    grid, ls, base = toy["grid"], toy["ls"], toy["base"]
    nst, tn = _nodes(toy, seam=())
    e64 = lm.extend_flux_library(base, nst, grid, ls, dT=10.0, offset=5.0, prof_dtype=np.float64)
    i0, i1 = e64.params[lm.EXTENSION_KEY]["base_slice"]
    assert e64.prof.dtype == np.float64 and np.array_equal(e64.prof[i0:i1], base.prof.astype(np.float64))
    b64 = lb.FluxLibrary(base.edges, base.tmean, base.count, base.prof.astype(np.float64), base.fc, base.dT,
                         params=base.params)
    with pytest.raises(ValueError, match="round"):
        lm.extend_flux_library(b64, nst, grid, ls, dT=10.0, offset=5.0, prof_dtype=np.float32)
    with pytest.raises(ValueError, match="lref"):
        lm.extend_flux_library(base, nst, grid, LineSet(ls.names, ls.lref + 1.0), dT=10.0, offset=5.0)
    named = lb.FluxLibrary(base.edges, base.tmean, base.count, base.prof, base.fc, base.dT,
                           params=dict(base.params, lines=["X1", "X2"]))
    with pytest.raises(ValueError, match="lines"):
        lm.extend_flux_library(named, nst, grid, ls, dT=10.0, offset=5.0)
    with pytest.raises(ValueError, match="grid"):
        lm.extend_flux_library(base, nst, VelocityGrid(dv=1.0, vmax=900.0, vshift=300.0), ls, dT=10.0, offset=5.0)
    inside, _ = _nodes(toy, tails=(), seam=(37905.0, 38005.0))
    with pytest.raises(ValueError, match="no node model lies outside"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lm.extend_flux_library(base, inside, grid, ls, dT=10.0, offset=5.0)
    # a plan whose top-node replica below the seam was nudged by +8 K (or +5 K: exactly onto the seam at 37 800 K)
    # into the base range (kept in its node): its node's mean T_eff' lies inside the base range
    plan = lm.plan_teff_nodes((37700.0, 37795.0), dT=10.0, offset=5.0, margin=0.0)
    for nudge in (8.0, 5.0):
        t = plan.teff.copy()
        t[-1] += nudge                                                    # 37 795 -> 37 803 / 37 800 K
        st = _store(plan.idx, t, ls, toy["pars"], grid)
        with pytest.raises(ValueError, match="inside the base library's range"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lm.extend_flux_library(base, st, grid, ls, plan=plan)
        with pytest.warns(UserWarning, match="left out"):
            ed = lm.extend_flux_library(base, st, grid, ls, plan=plan, nudged="drop")
        assert ed.params[lm.EXTENSION_KEY]["n_low"] == plan.nn - 1
    # two replicas per node, the upper one of the last node nudged into the base range (37 797.5 -> 37 800.5 K): the
    # node's mean (37 796.5 K) stays outside, so the node is used, with a warning naming the model
    p2 = lm.plan_teff_nodes((37700.0, 37795.0), dT=10.0, offset=5.0, margin=0.0, replicas=2)
    t = p2.teff.copy()
    t[-1] += 3.0
    st2 = _store(p2.idx, t, ls, toy["pars"], grid)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        e2 = lm.extend_flux_library(base, st2, grid, ls, plan=p2)
    assert any("have T_eff' inside the base library's range" in str(x.message) and "{}/37800.5".format(p2.idx[-1])
               in str(x.message) for x in w)
    i0 = e2.params[lm.EXTENSION_KEY]["base_slice"][0]
    assert e2.tmean[i0 - 1] == 37796.5 and e2.count[i0 - 1] == 2
    # the line order at the seam: a base without recorded lines whose lines are swapped
    sw = lb.FluxLibrary(base.edges, base.tmean, base.count, base.prof[:, ::-1], base.fc[:, ::-1], base.dT)
    with pytest.raises(ValueError, match="line order"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lm.extend_flux_library(sw, nst, grid, ls, dT=10.0, offset=5.0)
    bare = lb.FluxLibrary(base.edges, base.tmean, base.count, base.prof, base.fc, base.dT)
    with pytest.warns(UserWarning, match="records no lines"):
        eb = lm.extend_flux_library(bare, nst, grid, ls, dT=10.0, offset=5.0)
    assert eb.params[lm.EXTENSION_KEY]["line_identity_checked"] is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        eb = lm.extend_flux_library(bare, nst, grid, ls, dT=10.0, offset=5.0, base_lines=ls)
    assert eb.params[lm.EXTENSION_KEY]["line_identity_checked"] is True
    with pytest.raises(ValueError, match="base_lines"):
        lm.extend_flux_library(bare, nst, grid, ls, dT=10.0, offset=5.0, base_lines=ls.names[::-1])


def test_extend_flux_from_node_library(toy):
    """The node models' flux library can be given instead of the store (built once, e.g. for the seam check too): the
    same extended library."""
    grid, ls, base = toy["grid"], toy["ls"], toy["base"]
    nst, _ = _nodes(toy)
    nlib = lm.flux_library_from_models(nst, grid, ls, dT=10.0, offset=5.0)
    a = _quiet_extend(base, nst, grid, ls, dT=10.0, offset=5.0)
    b = _quiet_extend(base, nlib, grid, ls)
    for k in ("edges", "tmean", "count", "prof", "fc"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
    assert a.params[lb.SINGLE_BINS_KEY] == b.params[lb.SINGLE_BINS_KEY]


def test_extend_flux_node_library_checks(toy):
    """A prebuilt node FluxLibrary is checked like the base: another lref, other lines or another grid raise;
    array-valued params['lines'] work; a float64 node library is rounded to the float32 base with a warning (kept bit
    for bit with prof_dtype float64); without a model_idx record the result records None (not [])."""
    grid, ls, base = toy["grid"], toy["ls"], toy["base"]
    nst, _ = _nodes(toy, seam=())
    bad = lm.flux_library_from_models(nst, grid, LineSet(ls.names, ls.lref + 5.0), dT=10.0, offset=5.0)
    with pytest.raises(ValueError, match="node library's lref"):
        lm.extend_flux_library(base, bad, grid, ls)
    nlib = lm.flux_library_from_models(nst, grid, ls, dT=10.0, offset=5.0)
    other = lb.FluxLibrary(nlib.edges, nlib.tmean, nlib.count, nlib.prof, nlib.fc, nlib.dT,
                           params=dict(nlib.params, lines=["X1", "X2"]))
    with pytest.raises(ValueError, match="node library's lines"):
        lm.extend_flux_library(base, other, grid, ls)
    with pytest.raises(ValueError, match="lines"):
        lm.extend_flux_library(base, nlib, grid, ls, lines=["X1", "X2"])
    ogrid = lb.FluxLibrary(nlib.edges, nlib.tmean, nlib.count, nlib.prof, nlib.fc, nlib.dT,
                           params=dict(nlib.params, y0=nlib.params["y0"] + 1.0))
    with pytest.raises(ValueError, match="node library's grid"):
        lm.extend_flux_library(base, ogrid, grid, ls)
    ref = lm.extend_flux_library(base, nlib, grid, ls)
    arr = lb.FluxLibrary(nlib.edges, nlib.tmean, nlib.count, nlib.prof, nlib.fc, nlib.dT,
                         params=dict(nlib.params, lines=np.array(nlib.params["lines"])))
    a = lm.extend_flux_library(base, arr, grid, ls.lref)                # names from the node library (array)
    assert a.params["lines"] == list(ls.names) and np.array_equal(a.prof, ref.prof)
    n64 = lm.flux_library_from_models(nst, grid, ls, dT=10.0, offset=5.0, prof_dtype=np.float64)
    with pytest.warns(UserWarning, match="rounded to float32"):
        e32 = lm.extend_flux_library(base, n64, grid, ls)
    assert e32.prof.dtype == np.float32
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        e64 = lm.extend_flux_library(base, n64, grid, ls, prof_dtype=np.float64)
    nb_ = e64.params[lm.EXTENSION_KEY]["node_bins"]
    assert np.array_equal(e64.prof[nb_], n64.prof[e64.params[lm.EXTENSION_KEY]["node_source"]])
    bare = lb.FluxLibrary(nlib.edges, nlib.tmean, nlib.count, nlib.prof, nlib.fc, nlib.dT,
                          params={k: v for k, v in nlib.params.items() if k not in ("model_idx", "model_bin")})
    eb = lm.extend_flux_library(base, bare, grid, ls)
    assert eb.params["model_idx"] is None and eb.params["model_bin"] is None
    assert np.array_equal(eb.prof, ref.prof)


def test_extend_flux_chained(toy, tmp_path):
    """An extension of an extension (low tail first, high tail later): every node bin of both steps is its own node
    (nodes = base nodes + n_low + n_high), and the library is the direct extension's (bins, arrays, single bins, nodes);
    also after save -> load of the first step."""
    grid, ls, base = toy["grid"], toy["ls"], toy["base"]
    lo, _ = _nodes(toy, tails=((37600.0, 37800.0),), seam=())
    hi, _ = _nodes(toy, tails=((38400.0, 38600.0),), seam=(), start=10 ** 6)
    both = _merge(lo, hi)
    direct = lm.extend_flux_library(base, both, grid, ls, dT=10.0, offset=5.0)
    ext1 = lm.extend_flux_library(base, lo, grid, ls, dT=10.0, offset=5.0)
    bn = lb.lib_nodes(base, nmin=20)
    for e1 in (ext1, lb.FluxLibrary.load(ext1.save(str(tmp_path / "ext1.npz")))):
        ext2 = lm.extend_flux_library(e1, hi, grid, ls, dT=10.0, offset=5.0)
        r1, r2 = ext1.params[lm.EXTENSION_KEY], ext2.params[lm.EXTENSION_KEY]
        assert r1["n_low"] == 20 and r1["n_high"] == 0 and r2["n_low"] == 0 and r2["n_high"] == 20
        assert r2["base_single"] == 20
        n2 = lb.lib_nodes(ext2, nmin=20)
        assert n2.nn == bn.nn + 40 and n2.params["single"] == 40
        for k in ("edges", "tmean", "count", "prof", "fc"):
            assert np.array_equal(getattr(ext2, k), getattr(direct, k)), k
        assert ext2.params[lb.SINGLE_BINS_KEY] == direct.params[lb.SINGLE_BINS_KEY]
        nd = lb.lib_nodes(direct, nmin=20)
        for k in ("t", "count", "prof", "fc"):
            assert np.array_equal(n2[k], nd[k]), k
        single = set(ext2.params[lb.SINGLE_BINS_KEY])
        for g in n2.groups:
            if g[0] in single:
                assert g.size == 1
        assert sorted(int(g[0]) for g in n2.groups if g[0] in single) == sorted(single)


def test_single_bins_record(toy):
    """The single-bin declaration records its bin layout: a library rebuilt with other bins but the same params
    raises (other nb, or the same nb with shifted bins); an explicit single= is not checked; a record without the
    layout keys (older files) is accepted."""
    grid, ls, base = toy["grid"], toy["ls"], toy["base"]
    nst, _ = _nodes(toy, seam=())
    ext = lm.extend_flux_library(base, nst, grid, ls, dT=10.0, offset=5.0)
    assert ext.params[lb.SINGLE_NB_KEY] == ext.nb
    assert ext.params[lb.SINGLE_EDGES_KEY] == [float(ext.edges[i]) for i in ext.params[lb.SINGLE_BINS_KEY]]
    cut = lb.FluxLibrary(ext.edges[5:], ext.tmean[5:], ext.count[5:], ext.prof[5:], ext.fc[5:], ext.dT,
                         params=ext.params)
    with pytest.raises(ValueError, match="declares single bins for"):
        lb.lib_nodes(cut, nmin=20)
    shift = lb.FluxLibrary(ext.edges + 10.0, ext.tmean + 10.0, np.roll(ext.count, 1), np.roll(ext.prof, 1, axis=0),
                           np.roll(ext.fc, 1, axis=0), ext.dT, params=ext.params)
    with pytest.raises(ValueError, match="recorded lower edges"):
        lb.lib_nodes(shift, nmin=20)
    with pytest.raises(ValueError, match="recorded lower edges"):
        lm.extend_flux_library(shift, _nodes(toy, tails=((38600.0, 38700.0),), seam=())[0], grid, ls, dT=10.0,
                               offset=5.0)
    lb.lib_nodes(cut, nmin=20, single=False)
    lb.lib_nodes(cut, nmin=20, single=[0, 1])
    old = {k: v for k, v in ext.params.items() if k not in (lb.SINGLE_NB_KEY, lb.SINGLE_EDGES_KEY)}
    o = lb.FluxLibrary(ext.edges, ext.tmean, ext.count, ext.prof, ext.fc, ext.dT, params=old)
    assert lb.lib_nodes(o, nmin=20).nn == lb.lib_nodes(ext, nmin=20).nn
    with pytest.raises(ValueError):
        lb.single_bins_record([ext.nb], ext.edges)


def _stretch_nodes(lib, nmin, **kw):
    from ppmpy.synspec.dumps import _blas_limit
    with _blas_limit(1):
        return lb.lib_nodes(lib, nmin=nmin, **kw)


def test_lib_nodes_smooth_corr_extended(toy):
    """With single bins the smoothing stays on each side of the seam: the base nodes of the extended library are the
    base's smoothed nodes bit for bit (smooth 100 and 335) and each node run is smoothed over its own nodes (= the
    smoothed nodes of a library of that run alone); ignoring the declaration (single=False) mixes them. A per-bin
    correction assembled by extend_correction gives the base's corrected nodes bit for bit and the node bins their
    own corrections; without node corrections the node bins stay uncorrected (warning)."""
    grid, ls, base = toy["grid"], toy["ls"], toy["base"]
    lo, _ = _nodes(toy, tails=((37600.0, 37800.0),), seam=())
    hi, _ = _nodes(toy, tails=((38400.0, 38600.0),), seam=(), start=10 ** 6)
    nst = _merge(lo, hi)
    ext = lm.extend_flux_library(base, nst, grid, ls, dT=10.0, offset=5.0)
    rec = ext.params[lm.EXTENSION_KEY]
    i0, i1 = rec["base_slice"]
    runs = [lm.flux_library_from_models(x, grid, ls, dT=10.0, offset=5.0) for x in (lo, hi)]
    for sm in (100.0, 335.0):
        bn = _stretch_nodes(base, 20, smooth=sm)
        en = _stretch_nodes(ext, 20, smooth=sm)
        inb = np.array([g[0] >= i0 and g[-1] < i1 for g in en.groups])
        for k in ("t", "count", "prof", "fc"):
            assert np.array_equal(en[k][inb], bn[k]), (sm, k)
        lo_n, hi_n = np.flatnonzero(~inb & (en.t < rec["base_range"][0])), np.flatnonzero(~inb & (en.t > 38000.0))
        for sel, rl in zip((lo_n, hi_n), runs):
            rn = _stretch_nodes(rl, 1, smooth=sm)
            assert rn.nn == sel.size
            for k in ("t", "prof", "fc"):
                assert np.array_equal(en[k][sel], rn[k]), (sm, k)
        assert not np.array_equal(en.prof[lo_n], _stretch_nodes(runs[0], 1).prof)    # the node runs are smoothed
        mixed = _stretch_nodes(ext, 20, smooth=sm, single=False)
        assert mixed.nn < en.nn
    # the per-bin correction
    rng = np.random.default_rng(5)
    bcorr = 1e-4 * rng.standard_normal(base.prof.shape)
    bcorr[base.count == 0] = 0.0
    nlib = lm.flux_library_from_models(nst, grid, ls, dT=10.0, offset=5.0)
    ncorr = 1e-4 * rng.standard_normal(nlib.prof.shape)
    corr = lm.extend_correction(ext, bcorr, ncorr)
    assert np.array_equal(corr[i0:i1], bcorr[rec["base_bins"][0]:rec["base_bins"][1]])
    assert np.array_equal(corr[rec["node_bins"]], ncorr[rec["node_source"]])
    for sm in (0.0, 335.0):
        bn = _stretch_nodes(base, 20, smooth=sm, corr=bcorr)
        en = _stretch_nodes(ext, 20, smooth=sm, corr=corr)
        inb = np.array([g[0] >= i0 and g[-1] < i1 for g in en.groups])
        for k in ("t", "count", "prof", "fc"):
            assert np.array_equal(en[k][inb], bn[k]), (sm, k)
    en = _stretch_nodes(ext, 20, corr=corr)
    nodes = [g[0] for g in en.groups if g[0] < i0 or g[0] >= i1]
    sel = np.flatnonzero([g[0] < i0 or g[0] >= i1 for g in en.groups])
    assert np.array_equal(en.prof[sel], ext.prof[nodes].astype(np.float64) + corr[nodes])
    with pytest.warns(UserWarning, match="uncorrected"):
        c0 = lm.extend_correction(ext, bcorr)
    assert not c0[rec["node_bins"]].any()
    with pytest.raises(ValueError, match="node library's shape"):
        lm.extend_correction(ext, bcorr, ncorr[1:])
    with pytest.raises(ValueError, match="base library's shape"):
        lm.extend_correction(ext, bcorr[1:], ncorr)
    with pytest.raises(ValueError, match="extended FluxLibrary"):
        lm.extend_correction(base, bcorr, ncorr)


def test_flux_seam_check(toy):
    """Seam nodes equal to a base bin's only model give ~0 (the row offsets of the interpolation: ~1e-12); toy nodes
    at the bin centres differ from the base bins by the T_eff' offset within the bin (the interpolated variant is
    closer); a perturbed node (depth x 1.01) shows up as 1 % of the depth."""
    grid, ls, base, pars = toy["grid"], toy["ls"], toy["base"], toy["pars"]
    nst, tn = _nodes(toy)
    chk = lm.flux_seam_check(base, nst, grid, ls, dT=10.0, offset=5.0)
    print(chk["table"])
    s = chk["summary"]
    assert s["n_nodes"] == 4 and [r["t"] for r in chk["nodes"]] == [37805.0, 37815.0, 38385.0, 38395.0]
    assert max(s["max_dF_interp"]) < max(s["max_dF_bin"]) < 2e-3
    # a node model identical to the only model of a base bin (a sparse tail bin)
    bst = toy["base_store"]
    b1 = int(np.flatnonzero(base.count == 1)[0])
    k = int(np.flatnonzero(base.bin_index(bst["teff"]) == b1)[0])
    same = {kk: (v[k:k + 1] if kk != "lines" else v) for kk, v in bst.items()}
    same["idx"] = np.array([7], np.int32)
    c0 = lm.flux_seam_check(base, same, grid, ls, dT=10.0, offset=float(np.mod(bst["teff"][k], 10.0)))
    r = c0["nodes"][0]
    assert r["t"] == bst["teff"][k] and r["base_bin"] == b1 and r["base_count"] == 1
    for lab in ("bin", "interp"):                     # float32 base profiles; the same model otherwise
        assert max(r["max_dF_" + lab]) < 1e-7 and max(np.abs(r["dfc_" + lab])) < 1e-14, lab
    tail, _ = _nodes(toy, seam=(), start=1000)
    pert = _store([5], [37805.0], ls, pars, grid, scale=[1.01])
    c1 = lm.flux_seam_check(base, _merge(pert, tail), grid, ls, dT=10.0, offset=5.0)
    r1 = [x for x in c1["nodes"] if x["t"] == 37805.0][0]
    r0 = [x for x in chk["nodes"] if x["t"] == 37805.0][0]
    depth = [tt.toy_depth(np.array([37805.0]), grid.y, p, grid).max() for p in pars]
    for j in range(len(pars)):
        assert r1["max_dF_bin"][j] == pytest.approx(0.01 * depth[j], rel=0.2, abs=2 * r0["max_dF_bin"][j])
        assert r1["dEW_bin"][j] > 0                                        # deeper node: larger EW
    none = lm.flux_seam_check(base, tail, grid, ls, dT=10.0, offset=5.0)
    assert none["summary"]["n_nodes"] == 0 and np.isnan(none["summary"]["max_dF_bin"]).all()
    # variant 'nodes': the extended library's nodes as the integrator combines them; = DiscFlux's F0 of a uniform star
    # at the seam node's T_eff' (no Doppler shifts), the same with the extended library given
    ext = _quiet_extend(base, nst, grid, ls, dT=10.0, offset=5.0)
    en = lb.lib_nodes(ext, nmin=20)
    integ = dm.flux_integrator(ext, nmin=20, grid=grid)
    sph = tt.toy_sphere(2000, seed=4, dump=1, teff0=38000.0, plume=False)
    from ppmpy.synspec.sphere import project_los
    mu, tn_, pn = project_los(sph["theta"], sph["phi"], "thompson2024")
    chk2 = lm.flux_seam_check(base, nst, grid, ls, dT=10.0, offset=5.0, ext=ext)
    for r, r2 in zip(chk["nodes"], chk2["nodes"]):
        assert r["nodes_t"] == r2["nodes_t"] and r["max_dF_nodes"] == r2["max_dF_nodes"]
        k = int(np.flatnonzero(nst["teff"] == r["t"])[0])
        u = dict(sph, teff=np.full(sph["teff"].size, r["t"]))
        res = dm.disc_dump(integ, u, mu[:1], tn_[:1], pn[:1], batch=None)
        F0 = res["F0"][0]
        nlib = lm.flux_library_from_models({kk: (v[k:k + 1] if kk != "lines" else v) for kk, v in nst.items()}, grid,
                                           ls, dT=10.0, offset=5.0, prof_dtype=np.float64)
        d = np.abs(F0 - nlib.prof[0]).max(axis=1)
        np.testing.assert_allclose(d, r["max_dF_nodes"], rtol=1e-6, atol=1e-12)
        assert r["nodes_t"][0] < r["t"] < r["nodes_t"][1]
    lo = [r for r in chk["nodes"] if r["t"] < 38000.0]
    assert [r["nodes_t"][0] for r in lo] == [37795.0, 37795.0]                     # the last node below the seam
    assert lo[0]["nodes_t"][1] == en.t[np.flatnonzero(en.t > 37800.0)[0]]         # the first (merged) base node
    assert chk["n_interp_nodes"] == en.nn and "interp. nodes" in chk["table"]
    assert max(s["max_dF_nodes"]) < 2e-3
    r1n = [x for x in c1["nodes"] if x["t"] == 37805.0][0]
    for j in range(len(pars)):                                                      # the perturbed node: 1 % depth
        assert r1n["max_dF_nodes"][j] == pytest.approx(0.01 * depth[j], rel=0.2, abs=2 * r0["max_dF_nodes"][j])
    assert np.isnan(c0["summary"]["max_dF_nodes"]).all() and "nodes_t" not in c0["nodes"][0]   # no node outside


# ------------------------------------------------------------------------------------------------------------------
# the extended intensity library (toy OUT_IMU / OUT files)
# ------------------------------------------------------------------------------------------------------------------
IMU_LINE = "HEI4026"
IMU_LS = LineSet([IMU_LINE], [4026.22])


def _write_imu_model(d, idx, teff, nray, rng, h=0.004):
    """meta.txt, OUT_IMU and OUT of one toy model (one line): a limb-darkened continuum and a line whose depth depends
    on T_eff' and mu; OUT holds the flux of the rays (5 decimals) on the 0.01 A wavelengths. Returns (lam, fnorm,
    fcont) of OUT (the profiles-store row)."""
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "meta.txt"), "w") as f:
        f.write("{} {:.3f} ok 60 37000.0 150.0 0.5\n".format(idx, teff))
    u = np.linspace(-1.0, 1.0, NROW)
    yb = 2600.0 * np.sign(u) * np.abs(u) ** 1.6 + 3.0
    lam = 4026.22 * np.exp(yb / C_KMS)
    core = np.linspace(0.0, 1.0, 11)
    env = 1.0 + 1e-4 * np.cumsum(rng.uniform(5.0, 40.0, nray - 11))
    p = np.concatenate([core, env])
    mu = np.sqrt(np.clip(1.0 - p ** 2, 0.0, 1.0))
    ld = np.where(p <= 1.0, 1.0 - 0.5 * (1.0 - mu), 0.5 * np.exp(-np.maximum(p - 1.0, 0.0) / h))
    x = yb / 400.0
    c = 1e-5 * (teff / 38000.0) ** 4 * (1.0 + 0.02 * x / 6.5)[:, None]
    Ic = c * ld[None, :]
    depth = 0.45 * (38000.0 / teff) ** 3 * np.exp(-0.5 * x ** 2)
    Il = Ic * (1.0 - depth[:, None] * (0.6 + 0.4 * mu[None, :]))
    fimu = os.path.join(d, "OUT_IMU.{}_VTV010".format(IMU_LINE))
    with open(fimu, "w") as fh:
        fh.write("# rays NP-1, core rays NC = {:4d} {:4d}\n".format(nray, 10))
        fh.write("# p  " + " ".join("{:15.8E}".format(v) for v in p) + "\n")
        fh.write("# K, lambda, I_cont(p_1..p_NP-1), I_line(p_1..p_NP-1)\n")
        for k in range(lam.size):
            fh.write("{:5d} {:11.4f} ".format(k + 1, lam[k]) + " ".join("{:.6E}".format(v) for v in Ic[k]) + " "
                     + " ".join("{:.6E}".format(v) for v in Il[k]) + "\n")
    from ppmpy.synspec.fwresults import read_out, read_out_imu
    lr, pr, icr, ilr = read_out_imu(fimu)
    fn, fl, fcn = lb.flux_from_rays(pr, icr, ilr, lb.r_outer(pr, icr), full=True)
    f = np.round(fn, 5)
    with open(os.path.join(d, "OUT.{}_VTV010".format(IMU_LINE)), "w") as fh:
        for k in range(lam.size):
            fh.write("{:4d} {:11.5f} {:15.2f} {:19.6E} {:11.5f} {:11.5f}\n".format(k + 1, 1.2 - 0.015 * k, lam[k],
                                                                               fcn[k], f[k], f[k]))
        fh.write("  -1.08387540430798\n")
    out = read_out(os.path.join(d, "OUT.{}_VTV010".format(IMU_LINE)))
    return out["lam"], out["fnorm"], out["fcont"]


def _imu_store(runs, idx, teff, nray, rng, h=None):
    n = len(idx)
    h = np.full(n, 0.004) if h is None else np.asarray(h, dtype=np.float64)
    lam, fn, fc = np.empty((n, 1, NROW)), np.empty((n, 1, NROW)), np.empty((n, 1, NROW))
    for k, (i, t, nr) in enumerate(zip(idx, teff, nray)):
        lam[k, 0], fn[k, 0], fc[k, 0] = _write_imu_model(os.path.join(runs, lb.IMU_LAYOUT.format(idx=int(i))), int(i),
                                                         float(t), int(nr), rng, h=float(h[k]))
    return dict(idx=np.asarray(idx, dtype=np.int32), teff=np.asarray(teff, dtype=np.float64),
                status=np.full(n, "ok", dtype="<U12"), niter=np.full(n, 61, np.int16), lam=lam, fcont=fc, fnorm=fn,
                lines=np.array([IMU_LINE]), teff_nudge=np.zeros(n, np.float32))


@pytest.fixture(scope="module")
def imutoy(tmp_path_factory):
    """Base models: one per 10 K bin from 37 800 to 37 900 K (one bin empty), node models below (37 700-37 800 K, one
    node failing to have OUT_IMU-compatible rays: none) and above (37 900-37 960 K, one with more rays than the base's
    K), and seam copies; the M424 grid."""
    root = str(tmp_path_factory.mktemp("extimu"))
    runs = os.path.join(root, "runs")
    rng = np.random.default_rng(11)
    grid = VelocityGrid()
    tb = np.array([37802.25, 37815.0, 37824.5, 37846.0, 37853.0, 37866.5, 37875.0, 37884.0, 37899.0])  # 37 830 empty
    base_st = _imu_store(runs, np.arange(1, tb.size + 1), tb, rng.integers(24, 30, tb.size), rng)
    tl = 37705.0 + 10.0 * np.arange(10)
    th = 37905.0 + 10.0 * np.arange(6)
    nray = np.r_[rng.integers(24, 30, tl.size), rng.integers(24, 30, th.size)]
    nray[-2] = 44                                                        # K grows beyond the base's
    h = np.full(nray.size, 0.004)
    h[-2] = 0.012                                                        # a wider envelope: more rays inside R_max
    node_st = _imu_store(runs, 100 + np.arange(tl.size + th.size), np.r_[tl, th], nray, rng, h=h)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bflux = lm.flux_library_from_models(base_st, grid, IMU_LS, dT=10.0, offset=5.0, edges=37800.0 + 10.0 *
                                            np.arange(11))
        bimu, _, _ = lm.imu_library_from_models(bflux, base_st, grid=grid, lref=IMU_LS, imu_runs=runs)
    return dict(root=root, runs=runs, grid=grid, base_st=base_st, node_st=node_st, bflux=bflux, bimu=bimu)


def test_extend_imu(imutoy):
    """Base rows bit for bit (padded to the larger K), node rows = the rows of an intensity library built over all
    models (same models, same build), src of empty / gap bins = the nearest bin with a representative; DiscImu of the
    extended library has the nodes, arrays and profiles of the one built over all models."""
    from ppmpy.synspec.sphere import project_los
    t = imutoy
    grid, runs = t["grid"], t["runs"]
    ext = lm.extend_flux_library(t["bflux"], t["node_st"], grid, IMU_LS, dT=10.0, offset=5.0)
    imu, chk, reps = lm.extend_imu_library(t["bimu"], ext, t["node_st"], imu_runs=runs)
    rec = ext.params[lm.EXTENSION_KEY]
    i0, i1 = rec["base_slice"]
    bimu = t["bimu"]
    Kb = bimu.K
    assert len(reps) == 16 and chk["summary"][IMU_LINE]["max_dF"] < 1e-4 and imu.K > Kb
    for k in ("rmax", "nnode", "idx_rep", "teff_rep"):
        assert np.array_equal(imu[k][i0:i1], bimu[k]), k
    assert np.array_equal(imu.src[i0:i1], bimu.src + i0)
    for k in ("Ic", "Il"):
        assert np.array_equal(imu[k][i0:i1, :, :Kb], bimu[k]) and not imu[k][i0:i1, :, Kb:].any(), k
    assert np.array_equal(imu.s[i0:i1, :, :Kb], bimu.s, equal_nan=True) and np.isnan(imu.s[i0:i1, :, Kb:]).all()
    # the library over all models (the same bins: base and node bins on the 10 K grid)
    allst = _merge(t["base_st"], t["node_st"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fall = lm.flux_library_from_models(allst, grid, IMU_LS, dT=10.0, offset=5.0)
        iall, _, _ = lm.imu_library_from_models(fall, allst, grid=grid, lref=IMU_LS, imu_runs=runs)
    assert np.array_equal(fall.edges, ext.edges) and imu.K == iall.K
    u = np.unique(imu.src)
    assert np.array_equal(u, np.unique(iall.src)) and np.array_equal(imu.src, iall.src)
    for k in ("Ic", "Il", "s", "rmax", "nnode", "idx_rep", "teff_rep"):
        assert np.array_equal(imu[k][u], iall[k][u], equal_nan=True), k
    # the integrators and a toy dump
    sph = tt.toy_sphere(3000, seed=2, dump=1, teff0=37830.0, teff_rel_rms=0.002, plume=False)
    mu, tn, pn = project_los(sph["theta"], sph["phi"], "thompson2024")
    a = dm.imu_integrator(imu, fft="lazy", lref=IMU_LS)
    b = dm.imu_integrator(iall, fft="lazy", lref=IMU_LS)
    assert a.nn == b.nn == u.size and np.array_equal(a.t, b.t)
    ra = dm.disc_dump(a, sph, mu[:2], tn[:2], pn[:2], batch=None)
    rb = dm.disc_dump(b, sph, mu[:2], tn[:2], pn[:2], batch=None)
    assert np.array_equal(ra["F"], rb["F"]) and np.array_equal(ra["F0"], rb["F0"])
    assert imu.params["mode"] == "extended" and imu.params[lm.EXTENSION_KEY]["rep_idx"] == reps.idx.tolist()
    with pytest.raises(ValueError, match="extended FluxLibrary"):
        lm.extend_imu_library(bimu, t["bflux"], t["node_st"], imu_runs=runs)
    _imu_same = ("edges", "tmean", "count", "src", "idx_rep", "teff_rep", "rmax", "nnode", "s", "Ic", "Il")
    # chained: the low tail first, the high tail later (the base intensity library of step 2 is step 1's)
    nst = t["node_st"]
    lo = {k: (v[:10] if k != "lines" else v) for k, v in nst.items()}
    hi = {k: (v[10:] if k != "lines" else v) for k, v in nst.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e1 = lm.extend_flux_library(t["bflux"], lo, grid, IMU_LS, dT=10.0, offset=5.0)
        i1, _, _ = lm.extend_imu_library(bimu, e1, lo, imu_runs=runs)
        e2 = lm.extend_flux_library(e1, hi, grid, IMU_LS, dT=10.0, offset=5.0)
        i2, _, r2 = lm.extend_imu_library(i1, e2, hi, imu_runs=runs)
    assert len(r2) == 6 and np.array_equal(e2.edges, ext.edges)
    for k in _imu_same:
        assert np.array_equal(i2[k], imu[k], equal_nan=True), k
    # a node library without a model_idx record: the candidates' bins by T_eff', the same library
    nlib = lm.flux_library_from_models(nst, grid, IMU_LS, dT=10.0, offset=5.0)
    bare = lb.FluxLibrary(nlib.edges, nlib.tmean, nlib.count, nlib.prof, nlib.fc, nlib.dT,
                          params={k: v for k, v in nlib.params.items() if k not in ("model_idx", "model_bin")})
    eb = lm.extend_flux_library(t["bflux"], bare, grid, IMU_LS)
    assert eb.params["model_idx"] is None
    ib, _, _ = lm.extend_imu_library(bimu, eb, nst, imu_runs=runs)
    for k in _imu_same:
        assert np.array_equal(ib[k], imu[k], equal_nan=True), k
    # recorded node models absent from the store: a clear error
    other = dict(nst, idx=nst["idx"] + 10000)
    with pytest.raises(ValueError, match="none of the extended library's"):
        lm.extend_imu_library(bimu, ext, other, imu_runs=runs)


def test_imu_seam_check(imutoy, tmp_path):
    """Seam nodes that are copies of base representatives (same OUT_IMU files) give exactly 0; a perturbed copy (line
    intensities x (1 - 0.01 depth)) shows a nonzero static difference."""
    t = imutoy
    grid = t["grid"]
    runs = str(tmp_path / "runs")
    rng = np.random.default_rng(11)
    base = t["base_st"]
    import shutil
    rows = [0, 4]
    seam = {k: (v[rows] if k != "lines" else v) for k, v in base.items()}
    seam["idx"] = np.array([501, 502], np.int32)
    for i_src, i_new in zip(base["idx"][rows], seam["idx"]):
        shutil.copytree(os.path.join(t["runs"], lb.IMU_LAYOUT.format(idx=int(i_src))),
                        os.path.join(runs, lb.IMU_LAYOUT.format(idx=int(i_new))))
    st = _merge(seam, _imu_store(runs, [601, 602], [37705.0, 37915.0], [26, 26], rng))
    chk = lm.imu_seam_check(t["bimu"], st, grid, IMU_LS, dT=10.0, offset=3.0, imu_runs=runs)
    print(chk["table"])
    assert chk["summary"]["n_nodes"] == 2
    for r in chk["nodes"]:
        assert r["base_idx"] in base["idx"][rows].tolist() and r["max_dF_bin"] == [0.0] and r["dfc_bin"] == [0.0]
    # perturb the line intensities of one copy
    f = os.path.join(runs, lb.IMU_LAYOUT.format(idx=501), "OUT_IMU.{}_VTV010".format(IMU_LINE))
    from ppmpy.synspec.fwresults import read_out_imu
    lam, p, Ic, Il = read_out_imu(f)
    Il2 = Il * (1.0 - 0.01 * (1.0 - Il / Ic))
    with open(f, "w") as fh:
        fh.write("# rays NP-1, core rays NC = {:4d} {:4d}\n".format(p.size, 10))
        fh.write("# p  " + " ".join("{:15.8E}".format(v) for v in p) + "\n")
        fh.write("# K, lambda, I_cont(p_1..p_NP-1), I_line(p_1..p_NP-1)\n")
        for k in range(lam.size):
            fh.write("{:5d} {:11.4f} ".format(k + 1, lam[k]) + " ".join("{:.6E}".format(v) for v in Ic[k]) + " "
                     + " ".join("{:.6E}".format(v) for v in Il2[k]) + "\n")
    c2 = lm.imu_seam_check(t["bimu"], st, grid, IMU_LS, dT=10.0, offset=3.0, imu_runs=runs)
    r = [x for x in c2["nodes"] if x["idx"] == 501][0]
    assert 1e-4 < r["max_dF_bin"][0] < 1e-2 and r["dEW_bin"][0] > 0
    # variant 'nodes' with the extended intensity library: at a seam node that sits on a representative (the copy
    # of base model 1 at 37 802.25 K, unperturbed: idx 502 is the copy of row 4) the interpolant is that node exactly;
    # the perturbed copy differs as for 'bin'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ext = lm.extend_flux_library(t["bflux"], t["node_st"], grid, IMU_LS, dT=10.0, offset=5.0)
        imu, _, _ = lm.extend_imu_library(t["bimu"], ext, t["node_st"], imu_runs=t["runs"])
    c3 = lm.imu_seam_check(t["bimu"], st, grid, IMU_LS, dT=10.0, offset=3.0, imu_runs=runs, ext_imu=imu)
    print(c3["table"])
    r3 = {x["idx"]: x for x in c3["nodes"]}
    assert r3[502]["nodes_a"] == 0.0 and r3[502]["max_dF_nodes"][0] < 1e-14
    assert r3[502]["nodes_t"][0] == r3[502]["t"]
    np.testing.assert_allclose(r3[501]["max_dF_nodes"], r3[501]["max_dF_bin"], rtol=1e-6)
    assert "max_dF_nodes" not in c2["nodes"][0] and np.isnan(c2["summary"]["max_dF_nodes"]).all()


# ------------------------------------------------------------------------------------------------------------------
# M424
# ------------------------------------------------------------------------------------------------------------------
M424_SHRINK = (220, 330)            # bins of the shrunk base: 37 600-38 700 K (5.6 % of the dump-3200 points below,
#                                     2.7 % above)
M424_DUMPS = (3200, 4000, 4800)


def _m424_samples():
    root = os.environ.get("PPMPY_SYNSPEC_M424_SAMPLES", "/scratch/ppathak/fastwind_sphere/samples_r4050_N1236544")
    if not os.path.isdir(root):
        pytest.skip("M424 samples not available: {}".format(root))
    return root


def _shrunk_flux(lib, b0, b1):
    return lb.FluxLibrary(lib.edges[b0:b1 + 1], lib.tmean[b0:b1], lib.count[b0:b1], lib.prof[b0:b1], lib.fc[b0:b1],
                          lib.dT, params=lib.params, inputs=lib.inputs)


def _projections(st):
    from ppmpy.synspec.sphere import project_los
    return project_los(st["theta"], st["phi"], "thompson2024", method="matmul")


@pytest.mark.m424
@pytest.mark.slow
def test_m424_extend_flux():
    """library_dT10.npz shrunk to bins 220-329 (37 600-38 700 K) and extended by one per-point model per 10 K node
    outside (the model closest to the bin centre, as a library-mode run at offset 5 K): the base bins and the base's
    nodes (nmin 20) are the production library's bit for bit; three dumps integrated with it agree with the production
    flux products within the V8 static floor of single-model nodes (V8 at dT 10 K, all nodes single models: max|dF|
    2-4e-5);
    the seam check of six nodes just inside the range is reported."""
    from ppmpy.synspec.validate import _get_sample
    t0 = time.time()
    st = ProfileStore.open(m424_path("run", "profiles.npz"))
    full = lb.FluxLibrary.load(m424_path("run", "library_dT10.npz"))
    ref = m424_path("disc", "flux")
    samples = _m424_samples()
    b0, b1 = M424_SHRINK
    base = _shrunk_flux(full, b0, b1)
    t_lo, t_hi = full.edges[b0], full.edges[b1]
    pool_t = np.asarray(st.teff, dtype=np.float64)
    # nodes at the bin centres outside the shrunk range, plus three seam nodes just inside each end
    c = full.centres
    want = (c < t_lo) | (c > t_hi) | (np.abs(c - t_lo - 15.0) < 11.0) | (np.abs(c - t_hi + 15.0) < 11.0)
    assert int(want.sum()) == full.nb - (b1 - b0) + 6
    # one model per node: closest to the node inside its 10 K bin (the selection on the whole contiguous node grid)
    rows, dist = lm.select_node_models(pool_t, c, 1, edges=full.edges, idx=np.asarray(st.idx), usable=st.usable())
    rows = rows[want, 0]
    sel = np.sort(rows[rows >= 0])
    tn = c[want]
    nst = lm._substore(st, sel)
    print("{} node models of {} nodes ({:.0f} s)".format(sel.size, tn.size, time.time() - t0))
    with pytest.warns(UserWarning, match="overlap"):
        ext = lm.extend_flux_library(base, nst, VelocityGrid(), LINESET, dT=10.0, offset=5.0, base_lines=LINESET)
    rec = ext.params[lm.EXTENSION_KEY]
    i0, i1 = rec["base_slice"]
    assert rec["gap_bins"] == [] and len(rec["seam_nodes"]) == 6 and rec["line_identity_checked"]
    print("{} filled node bins outside ({} below, {} above; {} node models), {} seam nodes ({} models) left out, {} "
          "empty node bins".format(rec["n_low"] + rec["n_high"], rec["n_low"], rec["n_high"], rec["node_models"],
                                   len(rec["seam_nodes"]), rec["seam_models"], len(rec["missing_nodes"])))
    for k in ("tmean", "count", "prof", "fc"):
        assert np.array_equal(getattr(ext, k)[i0:i1], getattr(full, k)[b0:b1]), k
    assert np.array_equal(ext.edges, full.edges)                         # node bins = the production bins here
    fn = lb.lib_nodes(full, nmin=20)
    en = lb.lib_nodes(ext, nmin=20)
    inb = np.array([g[0] >= i0 and g[-1] < i1 for g in en.groups])
    fin = np.array([g[0] >= b0 and g[-1] < b1 for g in fn.groups])
    assert inb.sum() == fin.sum() == b1 - b0
    for k in ("t", "count", "prof", "fc"):
        assert np.array_equal(en[k][inb], fn[k][fin]), k
    assert en.nn == inb.sum() + rec["n_low"] + rec["n_high"]
    # the disc-integrated profiles against the production flux products
    integ = dm.flux_integrator(ext, nmin=20)
    mu, tnn, pn = _projections(st)
    out = {}
    for d in M424_DUMPS:
        s = _get_sample(samples, d, dm.SAMPLE_PATTERN)
        r = dm.disc_dump(integ, s, mu, tnn, pn, lref=LREF)
        with np.load(os.path.join(ref, "d{:04d}.npz".format(d))) as z:
            F, F0 = z["F"].astype(np.float64), z["F0"].astype(np.float64)
        frac = float(((np.asarray(s["teff"]) < t_lo) | (np.asarray(s["teff"]) >= t_hi)).mean())
        out[d] = (np.abs(r["F"] - F).max(axis=(0, 2)), np.abs(r["F0"] - F0).max(axis=(0, 2)), frac)
        print("dump {}: {:.1%} of the points outside the base; max|dF| {} max|dF0| {}".format(
            d, frac, np.array2string(out[d][0], precision=2), np.array2string(out[d][1], precision=2)))
    dF = np.max([o[0] for o in out.values()], axis=0)
    dF0 = np.max([o[1] for o in out.values()], axis=0)
    assert dF.max() < 3e-5 and dF0.max() < 3e-5, (dF, dF0)
    # the seam check (with the extended library: variant 'nodes', the integrator's interpolant)
    chk = lm.flux_seam_check(base, nst, VelocityGrid(), LINESET, dT=10.0, offset=5.0, ext=ext, base_lines=LINESET)
    print(chk["table"])
    sm = chk["summary"]
    print("seam: max|dF| bin {} interp {} nodes {}, |dEW| bin {} interp {} nodes {} A, |dF_c/F_c| bin {} nodes {}"
          .format(*(" / ".join("{:.2e}".format(x) for x in sm[k])
                    for k in ("max_dF_bin", "max_dF_interp", "max_dF_nodes", "max_abs_dEW_bin", "max_abs_dEW_interp",
                              "max_abs_dEW_nodes", "max_abs_dfc_bin", "max_abs_dfc_nodes"))))
    assert chk["summary"]["n_nodes"] == 6 and max(chk["summary"]["max_dF_bin"]) < 5e-3
    assert max(chk["summary"]["max_dF_nodes"]) < 5e-3
    print("{:.0f} s".format(time.time() - t0))


@pytest.mark.m424
@pytest.mark.slow
def test_m424_extend_imu():
    """imu_library_dT10.npz shrunk to bins 220-329 and extended by the production representatives outside (their
    OUT_IMU files in the pformalsol reruns): the extended intensity library has the production library's nodes (the
    303 representatives) with their rows bit for bit, so DiscImu reproduces the stored imu products of dump 3200 to
    their float32 rounding."""
    from ppmpy.synspec.validate import _get_sample
    imu_root = os.environ.get("PPMPY_SYNSPEC_M424_IMU_ROOT", "/scratch/ppathak/fastwind_imu")
    for f in ("imu_library_dT10.npz", "representatives.txt", "runs"):
        if not os.path.exists(os.path.join(imu_root, f)):
            pytest.skip("not available: " + os.path.join(imu_root, f))
    t0 = time.time()
    st = ProfileStore.open(m424_path("run", "profiles.npz"))
    full = lb.FluxLibrary.load(m424_path("run", "library_dT10.npz"))
    prod = lb.ImuLibrary.load(os.path.join(imu_root, "imu_library_dT10.npz"))
    reps = lb.Representatives.read(os.path.join(imu_root, "representatives.txt"))
    b0, b1 = M424_SHRINK
    base = _shrunk_flux(full, b0, b1)
    bimu = lb.ImuLibrary(prod.edges[b0:b1 + 1], prod.tmean[b0:b1], prod.count[b0:b1], prod.src[b0:b1] - b0,
                         prod.idx_rep[b0:b1], prod.teff_rep[b0:b1], prod.rmax[b0:b1], prod.nnode[b0:b1],
                         prod.s[b0:b1], prod.Ic[b0:b1], prod.Il[b0:b1], params=prod.params)
    out = (reps.bins < b0) | (reps.bins >= b1)
    sidx = np.asarray(st.idx, dtype=np.int64)
    o = np.argsort(sidx)
    rows = np.sort(o[np.searchsorted(sidx[o], reps.idx[out])])
    assert np.array_equal(np.sort(sidx[rows]), np.sort(reps.idx[out]))
    nst = lm._substore(st, rows)
    ext = lm.extend_flux_library(base, nst, VelocityGrid(), LINESET, dT=10.0, offset=5.0, base_lines=LINESET)
    imu, chk, nreps = lm.extend_imu_library(bimu, ext, nst, imu_runs=os.path.join(imu_root, "runs"), check=False)
    print("extended intensity library: {} bins, {} node representatives ({:.0f} s)".format(imu.nb, len(nreps),
                                                                                         time.time() - t0))
    up, ue = np.unique(prod.src), np.unique(imu.src)
    assert np.array_equal(imu.edges, prod.edges) and np.array_equal(up, ue)
    assert np.array_equal(imu.teff_rep[ue], prod.teff_rep[up]) and np.array_equal(imu.idx_rep[ue], prod.idx_rep[up])
    for k in ("rmax", "nnode", "s"):
        assert np.array_equal(imu[k][ue], prod[k][up], equal_nan=True), k
    for k in ("Ic", "Il"):
        for b in ue:
            assert np.array_equal(imu[k][b], prod[k][b]), (k, b)
    integ = dm.imu_integrator(imu, fft="lazy", lref=LINESET)
    mu, tnn, pn = _projections(st)
    s = _get_sample(_m424_samples(), 3200, dm.SAMPLE_PATTERN)
    r = dm.disc_dump(integ, s, mu, tnn, pn, lref=LREF, batch=None)
    with np.load(os.path.join(m424_path("disc", "imu"), "d3200.npz")) as z:
        dF = np.abs(r["F"].astype(np.float32).astype(np.float64) - z["F"].astype(np.float64)).max()
        dF0 = np.abs(r["F0"].astype(np.float32).astype(np.float64) - z["F0"].astype(np.float64)).max()
    print("dump 3200 vs the production imu product: max|dF| {:.2e}, max|dF0| {:.2e} ({:.0f} s)".format(
        dF, dF0, time.time() - t0))
    assert dF <= 2.0 ** -23 and dF0 <= 2.0 ** -23
