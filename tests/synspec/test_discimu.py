"""Tests of the intensity method of ppmpy.synspec.disc: DiscImu (T_eff'-interpolated emergent intensities, the all-dump
'imu' run), integrate_imu_nearest and uniform_star_imu (the first intensity method, fw_disc_imu.py).

Synthetic (any machine): toy intensity libraries (a few T_eff' nodes and rays, 1-3 lines, empty bins, rows with
nnode < K): DiscImu bit for bit the frozen fw_disc.DiscImu (M424 grid); the lazy and float32 modes against the
default (lazy F bit for bit on x86-64, to rounding elsewhere: LAZY_BITWISE); a per-point brute-force sum (any grid;
also with extrapolated T_eff' weights, non-adjacent nodes as validate.v3_tails builds them and exactly cancelling
weights, in every mode); intensities independent of mu = DiscFlux; linear limb darkening against the analytic disc
integrals (no velocities: line depth; rigid rotation: Gray's rotation profile and the velocity dispersion); inputs
(dict grids, dtype spellings, the grid and lines recorded in a file's '_meta', bad call inputs), pickling (recipes for
files and memory-mapped ImuLibrary objects), fingerprint; run_disc_dumps serial / fork / spawn;
integrate_imu_nearest bit for bit the frozen fw_disc_imu.py run as a script on a toy star.

M424 (marker m424, slow): DiscImu reproduces the stored imu/dNNNN.npz products of dumps 3200, 4000, 4800 (setup
~1 min, ~9 GB); the lazy and float32 modes against the default (several dumps, 8 lines of sight; F and the
diagnostics) and the memory of the lazy mode (subprocess, VmHWM); the V3 extrapolation pairs on dumps 3334, 4391
(lazy vs default, a direct convolution); integrate_imu_nearest reproduces disc_los8_imu.npz (subprocess, VmHWM).

PPMPY_SYNSPEC_M424_IMU (default /scratch/ppathak/fastwind_imu/imu_library_dT10.npz) and PPMPY_SYNSPEC_M424_IMU_RUNS
(default /scratch/ppathak/fastwind_imu/runs) locate the intensity library and the representatives' model directories;
PPMPY_SYNSPEC_M424_SAMPLES the per-dump sphere samples."""
import json
import os
import pickle
import platform
import resource
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

import conftest
from conftest import ROOT, m424_path
from ppmpy.synspec import disc
from ppmpy.synspec import dumps as dp
from ppmpy.synspec import library as lb
from ppmpy.synspec import parallel as par
from ppmpy.synspec import sphere as sph
from ppmpy.synspec.conventions import C_KMS
from ppmpy.synspec.diagnostics import DIAG_KEYS, diagnostics_array
from ppmpy.synspec.io import npz_member_memmap
from ppmpy.synspec.spectral import LineSet, VelocityGrid, interp_rows

LREF = np.array([4026.22, 4199.90, 4921.93])
LINES = ["HEI4026", "HEII4200", "HEI4922"]
LINESET = LineSet(LINES, LREF)
LEGACY = os.environ.get("PPMPY_SYNSPEC_M424_PROJECT",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy"))
FROZEN_DIAG_KW = dict(vwin=400.0, ew_jacobian=True, inclusive=True)     # fw_disc.diagnostics of the frozen copy
IMU_LIB_M424 = os.environ.get("PPMPY_SYNSPEC_M424_IMU", "/scratch/ppathak/fastwind_imu/imu_library_dT10.npz")
IMU_RUNS_M424 = os.environ.get("PPMPY_SYNSPEC_M424_IMU_RUNS", "/scratch/ppathak/fastwind_imu/runs")
SAMPLES_M424 = os.environ.get("PPMPY_SYNSPEC_M424_SAMPLES", "/scratch/ppathak/fastwind_sphere/samples_r4050_N1236544")
SMALL = VelocityGrid(dv=2.0, vmax=600.0, vshift=150.0)


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _legacy_fw_disc():
    """The frozen fw_disc.py (skips when it is not there)."""
    if not os.path.exists(os.path.join(LEGACY, "fw_disc.py")):
        pytest.skip("legacy fw_disc.py not available in {}".format(LEGACY))
    sys.path.insert(0, LEGACY)
    try:
        import fw_disc as fd
    finally:
        sys.path.remove(LEGACY)
    return fd


def _run_legacy_script(name, argv, replace=(), ns=None):
    """Execute a frozen legacy script as a whole (its argparse sees argv), after text replacements."""
    path = os.path.join(LEGACY, name)
    if not os.path.exists(path):
        pytest.skip("legacy {} not available in {}".format(name, LEGACY))
    src = open(path).read()
    for a, b in replace:
        assert a in src, (name, a)
        src = src.replace(a, b)
    old = sys.argv
    sys.argv = [name] + [str(x) for x in argv]
    try:
        g = dict(ns or {}, __name__="legacy_" + name[:-3])
        exec(compile(src, path, "exec"), g)
    finally:
        sys.argv = old
    return g


def _toy_imu(grid, nl=3, nb=8, K=6, seed=0, empty=(2, 5), short=((4, 1, 4), (6, 0, 3), (0, 2, 2)), dT=150.0,
             t0=37600.0, garbage=7.0, slope=1e-5, sig=(25.0, 60.0, 15.0)):
    """A toy intensity library (legacy imu_library_dT10.npz members): nb bins of width dT from t0, empty bins copying
    the nearest filled bin (lower on a tie; src, as fw_imu_library.py), K rays with s = p / R_max from 0 to 1
    (rows with nnode < K: NaN beyond, intensities 'garbage' there, which must never contribute), float32 intensities
    I_c = c (1 - u (1 - mu)) (1 + slope y / vmax) and I_l = I_c (1 - d(T, mu) g(y)), the line depth, width and centre
    changing with T_eff' and mu, zero beyond |y| ~ 6 sigma."""
    rng = np.random.default_rng(seed)
    y = grid.y
    edges = t0 + dT * np.arange(nb + 1)
    filled = np.array([b for b in range(nb) if b not in empty])
    src = np.array([filled[np.argmin(np.abs(filled - b))] for b in range(nb)])
    teff_rep = (edges[:-1] + rng.uniform(0.2, 0.8, nb) * dT)[src]
    nnode = np.full((nb, nl), K, np.int64)
    for b, j, n in short:
        if b < nb and j < nl:
            nnode[b, j] = n
    nnode = nnode[src]
    s = np.full((nb, nl, K), np.nan)
    Il = np.full((nb, nl, K, y.size), garbage, np.float32)
    Ic = np.full((nb, nl, K, y.size), garbage, np.float32)
    for b in filled:
        x = (teff_rep[b] - t0) / (nb * dT)
        for j in range(nl):
            n = nnode[b, j]
            s[b, j, :n] = np.concatenate([[0.0], np.sort(rng.uniform(0.05, 0.98, n - 2)), [1.0]])
            for k in range(n):
                mu = np.sqrt(1.0 - s[b, j, k] ** 2)
                ic = (1.0 + 0.3 * x + 0.1 * j) * (1.0 - 0.5 * (1.0 - mu)) * (1.0 + slope * y / grid.vmax)
                sg = sig[j % len(sig)] * (1.0 + 0.1 * mu)
                d = (0.3 + 0.1 * j) * (1.0 + 0.2 * x) * (0.6 + 0.4 * mu) * np.exp(
                    -0.5 * ((y - (5.0 * j + 3.0 * x + 2.0 * mu)) / sg) ** 2)
                Ic[b, j, k] = ic
                Il[b, j, k] = ic * (1.0 - d)
    s, Il, Ic = s[src], Il[src], Ic[src]
    count = np.where(np.isin(np.arange(nb), filled), 20.0 + np.arange(nb), 0.0)
    return dict(edges=edges, tmean=0.5 * (edges[:-1] + edges[1:]), count=count, src=src,
                idx_rep=(1000 + np.arange(nb))[src], teff_rep=teff_rep, rmax=np.ones((nb, nl)), nnode=nnode, s=s,
                Ic=Ic, Il=Il)


def _save(path, lib, compressed=False):
    (np.savez_compressed if compressed else np.savez)(str(path), **lib)
    return str(path)


def _toy_points(N, tn, seed=1, vsig=60.0, fast=(450.0, -520.0, 401.0, -400.6, 900.0)):
    """mu, v, teff of N random points: a few visible ones with |v| > vshift (M424 grid: 400 km/s) and T_eff' beyond
    the nodes."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(-1.0, 1.0, N)
    v = vsig * rng.standard_normal(N)
    v[:len(fast)] = fast
    mu[:len(fast)] = 0.5
    mu[len(fast)] = 1.0                                             # disc centre: s = 0
    teff = rng.uniform(tn[0] - 80.0, tn[-1] + 80.0, N)
    return mu, v, teff


def _brute(lib, grid, mu, v, teff, lines, pairs=None):
    """Per-point reference of DiscImu: each visible point's intensities interpolated linearly in T_eff' between the
    representatives (clamped; or the given pairs (k0, k1, a): any nodes, weights 1 - a and a of any sign) and in s
    between its node's real rays, shifted by whole grid steps with the edge values beyond the grid, summed with
    weight mu; mean and rms of v with weight mu I_c(mu) at y = 0."""
    u = np.unique(lib["src"])
    t = lib["teff_rep"][u]
    S, NN = lib["s"][u], lib["nnode"][u]
    Il, Ic = lib["Il"][u].astype(np.float64), lib["Ic"][u].astype(np.float64)
    nl, ny = S.shape[1], grid.ny
    F, F0 = np.full((nl, ny), np.nan), np.full((nl, ny), np.nan)
    vm, sd = np.full(nl, np.nan), np.full(nl, np.nan)
    vis = np.where(mu > 0)[0]
    if pairs is None:
        k0 = np.clip(np.searchsorted(t, teff, side="right") - 1, 0, t.size - 2)
        k1 = k0 + 1
        a = np.clip((teff - t[k0]) / (t[k0 + 1] - t[k0]), 0.0, 1.0)
    else:
        k0, k1, a = pairs
    sh = np.clip(np.rint(-C_KMS * np.log(1.0 - v / C_KMS) / grid.dv).astype(int), -grid.nshift, grid.nshift)
    ic = int(np.argmin(np.abs(grid.y)))
    for j in lines:
        num, den, n0, d0 = np.zeros(ny), np.zeros(ny), np.zeros(ny), np.zeros(ny)
        wv = np.zeros(vis.size)
        for q, i in enumerate(vis):
            si = np.sqrt(max(1.0 - mu[i] ** 2, 0.0))
            rl, rc = np.zeros(ny), np.zeros(ny)
            for node, w in ((k0[i], 1.0 - a[i]), (k1[i], a[i])):
                n = NN[node, j]
                sn = S[node, j, :n]
                kk = min(max(np.searchsorted(sn, si, side="right") - 1, 0), n - 2)
                tt = min(max((si - sn[kk]) / (sn[kk + 1] - sn[kk]), 0.0), 1.0)
                rl += w * ((1 - tt) * Il[node, j, kk] + tt * Il[node, j, kk + 1])
                rc += w * ((1 - tt) * Ic[node, j, kk] + tt * Ic[node, j, kk + 1])
            idx = np.clip(np.arange(ny) + sh[i], 0, ny - 1)
            num += mu[i] * rl[idx]
            den += mu[i] * rc[idx]
            n0 += mu[i] * rl
            d0 += mu[i] * rc
            wv[q] = mu[i] * rc[ic]
        F[j], F0[j] = num / den, n0 / d0
        vm[j] = np.sum(wv * v[vis]) / wv.sum()
        sd[j] = np.sqrt(np.sum(wv * (v[vis] - vm[j]) ** 2) / wv.sum())
    return F, F0, vm, sd, int((np.abs(np.rint(-C_KMS * np.log(1.0 - v[vis] / C_KMS) / grid.dv)) > grid.nshift).sum())


def _eq(a, b):
    np.testing.assert_array_equal(a, b)


LAZY_BITWISE = platform.machine().lower() in ("x86_64", "amd64")
"""Whether fft='lazy' must give F bit for bit the default: it does on x86-64 (measured with scipy 1.13: the vector and
scalar code paths of pocketfft round alike); elsewhere (e.g. aarch64, FMA contraction) a row batched differently
may differ in the last bits, so the lazy F is checked to rounding there."""


def _eq_lazy(a, b, tol=1e-14):
    """Bitwise on x86-64, else equal NaN positions and |a - b| <= tol (lazy vs precomputed F; LAZY_BITWISE)."""
    if LAZY_BITWISE:
        _eq(a, b)
        return
    a, b = np.asarray(a), np.asarray(b)
    _eq(np.isnan(a), np.isnan(b))
    assert np.abs(a - b)[~np.isnan(a)].max(initial=0.0) <= tol


@pytest.fixture(scope="module")
def toy3(tmp_path_factory):
    """Toy library on the M424 grid with 3 lines (the frozen fw_disc.DiscImu needs both), saved uncompressed."""
    d = tmp_path_factory.mktemp("imu3")
    lib = _toy_imu(VelocityGrid())
    return dict(lib=lib, path=_save(d / "imu.npz", lib), dir=str(d))


# ----------------------------------------------------------------------------------------------
# DiscImu: legacy, modes, analytic cases
# ----------------------------------------------------------------------------------------------
def test_discimu_matches_legacy(toy3):
    """Bit for bit the frozen fw_disc.DiscImu (M424 grid; empty bins, nnode < K rows, clipped shifts, T_eff' beyond the
    nodes, the disc centre, novel on / off, line subsets): the arrays of the set-up and every output of a call."""
    fd = _legacy_fw_disc()
    ref = fd.DiscImu(toy3["path"])
    got = disc.DiscImu(toy3["path"])
    assert (got.nn, got.K, got.L, got.nv, got.vs) == (ref.nn, ref.K, ref.L, ref.nv, ref.vs)
    _eq(got.t, ref.t)
    _eq(got.nnode, ref.nnode)
    for j in range(3):
        _eq(got.Sflat[j], ref.Sflat[j])
        for x, y in zip(got.I0[j] + got.Ihat[j], ref.I0[j] + ref.Ihat[j]):
            assert x.dtype == y.dtype
            _eq(x, y)
        _eq(got.ic0[j], ref.ic0[j])
    assert got.grid.icentre == VelocityGrid().ny // 2
    mu, v, teff = _toy_points(6000, got.t)
    for x, y in zip(got.pairs(teff), ref.pairs(teff)):
        _eq(x, y)
    k0, k1, a = got.pairs(teff)
    for novel, lines in ((True, (0, 1, 2)), (False, (0, 1, 2)), (True, (1,)), (True, (0, 2))):
        r = ref(mu, v, k0, k1, a, novel=novel, lines=lines)
        g = got(mu, v, k0, k1, a, novel=novel, lines=lines)
        for x, y in zip(g[:4], r[:4]):
            _eq(x, y)                                              # NaN where legacy has NaN
        assert g[4] == r[4] == 4                                   # 450, -520, 401, 900 clipped; -400.6 -> 400 steps
    assert got.method == "imu" and got.node_params == {} and got.lines is None and got.lref is None


def test_discimu_vs_brute_force():
    """A per-point brute-force sum on another grid with 2 lines (empty bins, nnode < K, garbage beyond nnode, T_eff'
    beyond the nodes, clipped shifts) in every mode: default and lazy to rounding, float32 to its precision."""
    grid = SMALL
    lib = _toy_imu(grid, nl=2, nb=7, K=7, seed=3, slope=0.05)
    mu, v, teff = _toy_points(1500, lib["teff_rep"][np.unique(lib["src"])], seed=4,
                              fast=(200.0, -310.0, 151.0, -149.2, 600.0))
    ref = _brute(lib, grid, mu, v, teff, (0, 1))
    assert ref[4] > 3                                               # the 3 fast ones and the tail of the 60 km/s
    for kw, tol in ((dict(), 2e-14), (dict(fft="lazy"), 2e-14), (dict(dtype="float32"), 2e-7),
                    (dict(fft="lazy", dtype="float32"), 2e-6)):
        D = disc.DiscImu(lib, grid, **kw)
        F, F0, vm, sd, ncl = D(mu, v, *D.pairs(teff))
        assert np.abs(F - ref[0]).max() <= tol, (kw, np.abs(F - ref[0]).max())
        assert np.abs(F0 - ref[1]).max() <= max(tol, 2e-14), (kw, np.abs(F0 - ref[1]).max())
        np.testing.assert_allclose(vm, ref[2], rtol=0, atol=1e-11)
        np.testing.assert_allclose(sd, ref[3], rtol=1e-12)
        assert ncl == ref[4]
    # one line built: the other line is NaN, the built one as before
    D1 = disc.DiscImu(lib, grid, lines=[1], fft="lazy")
    F, F0, vm, sd, _ = D1(mu, v, *D1.pairs(teff))
    assert np.all(np.isnan(F[0])) and np.all(np.isnan(F0[0])) and np.isnan(vm[0])
    assert np.abs(F[1] - ref[0][1]).max() <= 2e-14
    assert D1.built == (1,) and D1.I0 == [None, None] and D1.ic0[0] is None


def _v3_pairs(t, teff, margin):
    """T_eff' pairs as validate.v3_tails builds them: clamped inside the nodes; below the first node k0 = 0, k1 = jlo
    (the first node at least ``margin`` inside) with a < 0, above the last k0 = jhi, k1 = nn - 1 with a > 1 (nodes not
    adjacent)."""
    nn = t.size
    jlo = min(int(np.searchsorted(t, t[0] + margin)), nn - 1)
    jhi = max(int(np.searchsorted(t, t[-1] - margin)) - 1, 0)
    k0, k1, a = lb.node_pairs(t, teff)
    k0, k1, a = k0.copy(), k1.copy(), a.astype(np.float64)
    lo, hi = teff < t[0], teff > t[-1]
    k0[lo], k1[lo], a[lo] = 0, jlo, (teff[lo] - t[0]) / (t[jlo] - t[0])
    k0[hi], k1[hi], a[hi] = jhi, nn - 1, (teff[hi] - t[jhi]) / (t[-1] - t[jhi])
    return k0, k1, a


def _legacy_used_rows(H):
    """The legacy row selection (positive row sums), to show what the tests catch."""
    return np.where(H.sum(axis=1) > 0)[0]


IMU_MODES = (dict(), dict(fft="lazy"), dict(dtype="float32"), dict(fft="lazy", dtype="float32"))
IMU_MODE_TOL = (2e-14, 2e-14, 2e-7, 2e-6)


def test_discimu_extrapolated_pairs(monkeypatch):
    """T_eff' weights < 0 or > 1 (pairs(mode='extrapolate'), and non-adjacent k0 / k1 as validate.v3_tails builds
    them): every histogram row with a nonzero weight enters F, so every mode equals the per-point brute force, F and
    F0, per call and in integrate_los. Cases: all points above the nodes; 10 % far above; the v3_tails pairs with
    points below and above; a row whose weights cancel exactly (row sum 0). The legacy row rule (positive row
    sums) misses rows in each case (checked: deviations > 1e-4)."""
    grid = SMALL
    lib = _toy_imu(grid, nl=2, nb=7, K=7, seed=3, slope=0.05)
    t = lib["teff_rep"][np.unique(lib["src"])]
    rng = np.random.default_rng(51)
    mu, v, _ = _toy_points(1200, t, seed=52, fast=(200.0, -310.0, 600.0))
    teff_all_above = rng.uniform(t[-1] + 20.0, t[-1] + 300.0, mu.size)
    teff_far = rng.uniform(t[0], t[-1], mu.size)
    far = rng.random(mu.size) < 0.1
    teff_far[far] = rng.uniform(t[-1] + 400.0, t[-1] + 800.0, int(far.sum()))
    teff_v3 = rng.uniform(t[0] - 400.0, t[-1] + 400.0, mu.size)
    # exact cancellation: two points at the same mu, pair (0, 1), a = -1 and a = +1, different shifts: the rows of
    # node 1 get -w at one shift and +w at the other (row sum exactly 0), node 0 gets 2w (a = -1) and 0 (a = +1)
    mu_c, v_c = np.array([0.6, 0.6]), np.array([30.0, -50.0])
    pairs_c = (np.array([0, 0]), np.array([1, 1]), np.array([-1.0, 1.0]))
    cases = [("all above", mu, v, teff_all_above, lb.node_pairs(t, teff_all_above, mode="extrapolate")),
             ("10% far above", mu, v, teff_far, lb.node_pairs(t, teff_far, mode="extrapolate")),
             ("v3_tails", mu, v, teff_v3, _v3_pairs(t, teff_v3, 300.0)),
             ("cancel", mu_c, v_c, np.zeros(2), pairs_c)]
    assert np.any(cases[2][4][1] - cases[2][4][0] > 1)                # non-adjacent nodes
    for name, m, vv, teff, pairs in cases:
        ref = _brute(lib, grid, m, vv, teff, (0, 1), pairs=pairs)
        ref2 = _brute(lib, grid, m, -vv, teff, (0, 1), pairs=pairs)
        for kw, tol in zip(IMU_MODES, IMU_MODE_TOL):
            D = disc.DiscImu(lib, grid, **kw)
            g = D(m, vv, *pairs)
            dF, dF0 = np.abs(g[0] - ref[0]).max(), np.abs(g[1] - ref[1]).max()
            assert dF <= tol and dF0 <= max(tol, 2e-14), (name, kw, dF, dF0)
            np.testing.assert_allclose(g[2], ref[2], rtol=0, atol=1e-10)
            np.testing.assert_allclose(g[3], ref[3], rtol=1e-11)
            r = D.integrate_los(np.stack([m, m]), np.stack([vv, -vv]), pairs=pairs)
            for k, rr in enumerate((ref, ref2)):
                assert np.abs(r["F"][k] - rr[0]).max() <= tol, (name, kw, k)
                assert np.abs(r["F0"][k] - rr[1]).max() <= max(tol, 2e-14), (name, kw, k)
    # the legacy rule drops rows here: the test would catch it in every case and mode
    monkeypatch.setattr(disc, "_used_rows", _legacy_used_rows)
    for name, m, vv, teff, pairs in cases:
        ref = _brute(lib, grid, m, vv, teff, (0, 1), pairs=pairs)
        for kw in IMU_MODES:
            D = disc.DiscImu(lib, grid, **kw)
            g = D(m, vv, *pairs)
            assert np.abs(g[0] - ref[0]).max() > 1e-4, (name, kw)


def test_discimu_bad_inputs(toy3):
    """Inputs that would silently give wrong profiles raise: a NaN T_eff' (a = NaN), a NaN or infinite velocity, v >= c,
    a non-finite mu, node indices out of range or not integers, mismatched shapes; per call and in integrate_los.
    Hidden points (mu <= 0) may carry anything."""
    D = disc.DiscImu(toy3["path"], fft="lazy")
    mu, v, teff = _toy_points(500, D.t, seed=61)
    k0, k1, a = D.pairs(teff)
    vis = np.where(mu > 0)[0][7]
    hid = np.where(mu <= 0)[0][3]

    def with_(x, i, val):
        x = x.copy()
        x[i] = val
        return x

    t_nan = with_(teff, vis, np.nan)
    for args, msg in (((mu, v, *D.pairs(t_nan)), "T_eff' weight"),
                      ((mu, with_(v, vis, np.nan), k0, k1, a), "velocity"),
                      ((mu, with_(v, vis, np.inf), k0, k1, a), "velocity"),
                      ((mu, with_(v, vis, C_KMS), k0, k1, a), "velocity"),
                      ((with_(mu, hid, np.nan), v, k0, k1, a), "mu"),
                      ((mu, v, with_(k0, vis, -1), k1, a), "node indices"),
                      ((mu, v, k0, with_(k1, vis, D.nn), a), "node indices"),
                      ((mu, v, k0.astype(float), k1, a), "integer"),
                      ((mu, v[:-1], k0, k1, a), "1-D")):
        with pytest.raises(ValueError, match=msg):
            D(*args)
        if args[1].shape == mu.shape:
            with pytest.raises(ValueError, match=msg):
                D.integrate_los(np.stack([args[0]] * 2), np.stack([args[1]] * 2), pairs=args[2:])
    ref = D(mu, v, k0, k1, a)
    v_h, a_h = with_(v, hid, np.nan), with_(a, hid, np.nan)           # hidden points: ignored
    g = D(mu, v_h, k0, k1, a_h)
    for x, y in zip(g[:4], ref[:4]):
        _eq(x, y)
    assert g[4] == ref[4]


def test_discimu_lazy_bitwise_and_float32(toy3):
    """fft='lazy': F, vmean, vsig bit for bit the default, F0 to rounding, per call and for several lines of sight at
    once (integrate_los, each library row transformed once); integrate_los = per-call results in every mode;
    dtype='float32' within 2e-7 (precomputed) and 1e-6 (lazy, float32 FFTs) of the default."""
    grid = VelocityGrid()
    D = disc.DiscImu(toy3["path"])
    Dl = disc.DiscImu(toy3["path"], fft="lazy")
    D32 = disc.DiscImu(toy3["path"], dtype="float32")
    Dl32 = disc.DiscImu(toy3["path"], dtype="float32", fft="lazy")
    assert Dl.I0 == [None] * 3 and Dl.Ihat == [None] * 3
    assert D32.Ihat[0][0].dtype == np.complex64 and D32.I0[0][0].dtype == np.float32
    rng = np.random.default_rng(8)
    theta, phi = sph.fibonacci_sphere(4000)
    MU, TN, PN = sph.project_los(theta, phi, "thompson2024")
    teff = rng.uniform(D.t[0] - 50.0, D.t[-1] + 50.0, theta.size)
    V = sph.los_velocity(*(60.0 * rng.standard_normal((3, theta.size))), MU, TN, PN)
    pairs = D.pairs(teff)
    ref = D.integrate_los(MU, V, pairs=pairs)
    for k in range(MU.shape[0]):
        g = D(MU[k], V[k], *pairs)
        for key, x in zip(("F", "F0", "vmean_w", "sigma_w"), g[:4]):
            _eq(ref[key][k], x)
        assert ref["n_clip"][k] == g[4]
    lz = Dl.integrate_los(MU, V, pairs=pairs)
    _eq_lazy(lz["F"], ref["F"])
    for key in ("vmean_w", "sigma_w", "n_clip"):
        _eq(lz[key], ref[key])
    assert 0 < np.abs(lz["F0"] - ref["F0"]).max() <= 1e-15 or np.array_equal(lz["F0"], ref["F0"])
    for k in (0, 5):
        g = Dl(MU[k], V[k], *pairs)
        _eq_lazy(g[0], ref["F"][k])
        _eq_lazy(g[0], lz["F"][k])
        assert np.abs(g[1] - ref["F0"][k]).max() <= 1e-15
    assert Dl._last_cache["rows"] <= Dl.nn * Dl.K
    for DD, tol in ((D32, 2e-7), (Dl32, 1e-6)):
        r = DD.integrate_los(MU, V, pairs=pairs)
        dF = max(np.abs(r["F"] - ref["F"]).max(), np.abs(r["F0"] - ref["F0"]).max())
        assert 0 < dF <= tol, (DD, dF)
        _eq(r["vmean_w"], ref["vmean_w"])
        _eq(r["n_clip"], ref["n_clip"])
    # novel=False: F0 NaN (legacy), F unchanged
    g = Dl(MU[1], V[1], *pairs, novel=False)
    assert np.all(np.isnan(g[1]))
    _eq_lazy(g[0], ref["F"][1])


def test_discimu_uniform_intensity_equals_discflux():
    """Intensities independent of mu (I_c = c per node, I_l = c f(y)): DiscImu is the flux method, DiscFlux with
    prof = f and fc = c, to rounding (also the velocity moments: weights mu F_c)."""
    grid = SMALL
    lib = _toy_imu(grid, nl=2, nb=6, K=5, seed=5, empty=(3,), short=((1, 0, 3),))
    u = np.unique(lib["src"])
    rng = np.random.default_rng(6)
    c = rng.uniform(0.8, 1.6, (lib["src"].size, 2))
    f = np.ones((lib["src"].size, 2, grid.ny))
    for b in range(lib["src"].size):
        for j in range(2):
            f[b, j] = 1.0 - (0.3 + 0.2 * rng.random()) * np.exp(-0.5 * ((grid.y - rng.uniform(-20, 20)) / 40.0) ** 2)
    c, f = c[lib["src"]], f[lib["src"]]
    lib["Ic"] = np.broadcast_to(c[:, :, None, None], lib["Ic"].shape).astype(np.float32)
    lib["Il"] = (lib["Ic"] * f[:, :, None, :].astype(np.float32)).astype(np.float32)
    nodes = dict(t=lib["teff_rep"][u], fc=c[u].astype(np.float32).astype(np.float64),
                 prof=(lib["Il"][u, :, 0].astype(np.float64) / lib["Ic"][u, :, 0].astype(np.float64)))
    DF = disc.DiscFlux(nodes, grid)
    mu, v, teff = _toy_points(20000, nodes["t"], seed=7, fast=(200.0, -310.0, 600.0))
    pairs = DF.pairs(teff)
    for fft in ("precomputed", "lazy"):
        DI = disc.DiscImu(lib, grid, fft=fft)
        rf, ri = DF(mu, v, *pairs), DI(mu, v, *pairs)
        assert np.abs(ri[0] - rf[0]).max() <= 2e-14
        assert np.abs(ri[1] - rf[1]).max() <= 2e-14
        np.testing.assert_allclose(ri[2], rf[2], rtol=0, atol=1e-11)
        np.testing.assert_allclose(ri[3], rf[3], rtol=1e-12)
        assert ri[4] == rf[4] > 2


def _ld_library(grid, u_ld, depth, K=181):
    """Two identical nodes, one line, K rays at s = sin(pi/2 k/(K - 1)) (dense towards the limb); I_c = 1 - u (1 - mu)
    (constant in y), I_l = I_c (1 - depth(mu, y))."""
    s = np.sin(0.5 * np.pi * np.arange(K) / (K - 1))
    s[-1] = 1.0
    mu = np.sqrt(np.clip(1.0 - s ** 2, 0.0, 1.0))
    Ic = np.repeat((1.0 - u_ld * (1.0 - mu))[:, None], grid.ny, axis=1)
    Il = Ic * (1.0 - depth(mu[:, None], grid.y[None, :]))
    one = dict(Ic=Ic.astype(np.float32), Il=Il.astype(np.float32))
    return dict(src=np.array([0, 1]), teff_rep=np.array([38000.0, 38100.0]), s=np.tile(s, (2, 1, 1)),
                nnode=np.full((2, 1), K), Il=np.stack([one["Il"][None]] * 2), Ic=np.stack([one["Ic"][None]] * 2))


def test_discimu_linear_limb_darkening_analytic():
    """Linear limb darkening I_c = 1 - u (1 - mu), u = 0.6, 200 000 equal-area points:
    (a) no velocities, line depth D0 mu g(y): F0 = 1 - D0 g(y) [(1 - u)/3 + u/4] / [(1 - u)/2 + u/3];
    (b) rigid rotation (v sin i = 150 km/s, equator on), depth D0 g(y) narrow: F = 1 - D0 (g * G)(y) with Gray's
    rotation profile G for limb darkening u; the velocity dispersion with weights mu I_c(mu) is
    v sin i sqrt(<1 - mu^2> / 2) = v sin i sqrt([(1 - u)/4 + 2u/15] / [2 ((1 - u)/2 + u/3)]), the mean 0."""
    grid = VelocityGrid(dv=1.0, vmax=900.0, vshift=300.0)
    uld, D0 = 0.6, 0.5
    theta, phi = sph.fibonacci_sphere(200000)
    # (a)
    lib = _ld_library(grid, uld, lambda m, y: D0 * m * np.exp(-0.5 * (y / 20.0) ** 2))
    D = disc.DiscImu(lib, grid)
    n = np.array([0.3, 0.5, 0.81]) / np.linalg.norm([0.3, 0.5, 0.81])
    MU, TN, PN = sph.project_los(theta, phi, n[None, :])
    pairs = D.pairs(np.full(theta.size, 38050.0))
    F, F0, vm, sd, _ = D(MU[0], np.zeros(theta.size), *pairs)
    R = ((1 - uld) / 3 + uld / 4) / ((1 - uld) / 2 + uld / 3)
    expect = 1.0 - D0 * R * np.exp(-0.5 * (grid.y / 20.0) ** 2)
    assert np.abs(F0[0] - expect).max() <= 2e-5 * D0                   # measured 4.3e-6 D0
    assert np.abs(F[0] - F0[0]).max() <= 1e-14
    # (b)
    lib = _ld_library(grid, uld, lambda m, y: D0 * np.exp(-0.5 * (y / 3.0) ** 2) + 0.0 * m)
    D = disc.DiscImu(lib, grid)
    vrot = 150.0
    MU, TN, PN = sph.project_los(theta, phi, np.array([[1.0, 0.0, 0.0]]))
    V = sph.los_velocity(np.zeros(theta.size), np.zeros(theta.size), vrot * np.sin(theta), MU, TN, PN)[0]
    F, F0, vm, sd, ncl = D(MU[0], V, *pairs)
    r2 = ((1 - uld) / 4 + 2 * uld / 15) / ((1 - uld) / 2 + uld / 3)
    assert abs(sd[0] / (vrot * np.sqrt(r2 / 2)) - 1.0) <= 2e-5          # measured 2.3e-6
    assert abs(vm[0]) <= 1e-5 and ncl == 0                              # measured 1e-7 km/s
    dv = 0.01
    x = np.arange(-vrot, vrot + dv / 2, dv)
    G = (2 * (1 - uld) * np.sqrt(np.clip(1 - (x / vrot) ** 2, 0, 1)) + 0.5 * np.pi * uld * (1 - (x / vrot) ** 2)) / (
        np.pi * vrot * (1 - uld / 3))
    G /= G.sum() * dv
    yy = grid.y[np.abs(grid.y) <= 250]
    depth = np.array([np.sum(G * D0 * np.exp(-0.5 * ((y0 + x) / 3.0) ** 2)) * dv for y0 in yy])
    got = 1.0 - F[0][np.abs(grid.y) <= 250]
    assert np.abs(got - depth).max() <= 6e-3 * depth.max()            # measured 2.6e-3: 1 km/s shift rounding
    ew_got, ew_exp = got.sum() * grid.dv, depth.sum() * grid.dv
    assert abs(ew_got / ew_exp - 1.0) <= 1e-6                           # measured 2e-8


def test_discimu_inputs(toy3, tmp_path):
    """Path, np.load()ed NpzFile (reopened with memory maps), dict and compressed .npz give the same bits; bad options,
    shapes and libraries raise."""
    lib = toy3["lib"]
    mu, v, teff = _toy_points(3000, lib["teff_rep"][np.unique(lib["src"])], seed=11)
    D = disc.DiscImu(toy3["path"])
    assert isinstance(D._src, type(None)) and D.path == os.path.abspath(toy3["path"])
    k = D.pairs(teff)
    ref = D(mu, v, *k)
    cpath = _save(tmp_path / "c.npz", lib, compressed=True)
    with np.load(toy3["path"]) as z:
        Dz = disc.DiscImu(z, fft="lazy")
        assert isinstance(Dz._src[0], np.memmap)
    for other in (disc.DiscImu(dict(lib)), disc.DiscImu(cpath), Dz, disc.DiscImu(cpath, fft="lazy")):
        g = other(mu, v, *k)
        _eq_lazy(g[0], ref[0])
        for x, y in zip(g[2:4], ref[2:4]):
            _eq(x, y)
        assert np.abs(g[1] - ref[1]).max() <= 1e-15
    assert not isinstance(disc.DiscImu(cpath, fft="lazy")._src[0], np.memmap)
    with pytest.raises(ValueError, match="dtype"):
        disc.DiscImu(lib, dtype="float16")
    with pytest.raises(ValueError, match="fft"):
        disc.DiscImu(lib, fft="eager")
    with pytest.raises(ValueError, match="chunk"):
        disc.DiscImu(lib, chunk=0)
    with pytest.raises(ValueError, match="line"):
        disc.DiscImu(lib, lines=[3])
    with pytest.raises(ValueError, match="unknown line"):
        disc.DiscImu(lib, lines=["HEI4026"])
    with pytest.raises(ValueError, match="shape"):
        disc.DiscImu(lib, grid=SMALL)
    with pytest.raises(KeyError):
        disc.DiscImu({k: x for k, x in lib.items() if k != "nnode"})
    bad = dict(lib, teff_rep=lib["teff_rep"][::-1].copy())
    with pytest.raises(ValueError, match="increase"):
        disc.DiscImu(bad)
    s = lib["s"].copy()
    s[0, 0, [1, 2]] = s[0, 0, [2, 1]]
    with pytest.raises(ValueError, match="not increasing"):
        disc.DiscImu(dict(lib, s=s))
    s = lib["s"].copy()
    s[1, 1, 0] = -0.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        disc.DiscImu(dict(lib, s=s))
    nn = lib["nnode"].copy()
    nn[0, 0] = 1
    with pytest.raises(ValueError, match="nnode"):
        disc.DiscImu(dict(lib, nnode=nn))
    D1 = disc.DiscImu(lib, lines=(0, 2))
    with pytest.raises(ValueError, match="not built"):
        D1(mu, v, *k, lines=(1,))
    # a library that records its lines: names select lines, and dumps uses its lref
    Dn = disc.DiscImu(dict(lib, lref=LREF, lines=np.array(LINES)), lines=["HEII4200"])
    assert Dn.built == (1,) and isinstance(Dn.lines, LineSet) and Dn.lines.names == LINES
    assert dp._integ_lref(Dn)[1].tolist() == LREF.tolist() and dp._integ_nl(Dn) == 3
    # every spelling of the dtypes; a dict grid (as dumps.flux_integrator and factory_kwargs)
    for spelled, name in (("f4", "float32"), ("single", "float32"), (np.float32, "float32"),
                          (np.dtype("float32"), "float32"), ("f8", "float64"), ("double", "float64"),
                          (float, "float64")):
        Dt = disc.DiscImu(lib, dtype=spelled, fft="lazy")
        assert Dt.dtype == name and Dt._init_args["dtype"] == name and Dt.fingerprint()["dtype"] == name
    with pytest.raises(ValueError, match="dtype"):
        disc.DiscImu(lib, dtype="no-such-type")
    Dg = disc.DiscImu(lib, grid=dict(dv=1.0, vmax=2700.0, vshift=400.0))
    assert Dg.grid.to_dict() == VelocityGrid().to_dict()
    g = Dg(mu, v, *k)
    for x, y in zip(g[:4], ref[:4]):
        _eq(x, y)
    with pytest.raises(TypeError):
        disc.DiscImu(lib, grid="M424")


def test_discimu_pickle_and_fingerprint(toy3, tmp_path):
    """A DiscImu built from a file pickles as its recipe (small; rebuilt and identity-checked when unpickled), one built
    from a mapping as its arrays. fingerprint(): the library hash is the same in every mode and for the same arrays
    from a mapping, and changes with one intensity value; the options are recorded."""
    D = disc.DiscImu(toy3["path"], fft="lazy")
    blob = pickle.dumps(D)
    assert len(blob) < 4000
    D2 = pickle.loads(blob)
    mu, v, teff = _toy_points(2000, D.t, seed=12)
    for x, y in zip(D(mu, v, *D.pairs(teff))[:4], D2(mu, v, *D2.pairs(teff))[:4]):
        _eq(x, y)
    Dm = disc.DiscImu(dict(toy3["lib"]))
    assert len(pickle.dumps(Dm)) > Dm.memory()["library"]
    fp = D.fingerprint()
    json.dumps(fp)
    fps = [disc.DiscImu(toy3["path"]).fingerprint(), Dm.fingerprint(), D2.fingerprint(),
           disc.DiscImu(toy3["path"], dtype="float32", chunk=64).fingerprint()]
    assert len({f["library_sha256"] for f in fps + [fp]}) == 1
    assert (fp["fft"], fps[0]["fft"], fps[3]["dtype"], fps[3]["chunk"]) == ("lazy", "precomputed", "float32", 64)
    lib = dict(toy3["lib"])
    lib["Il"] = lib["Il"].copy()
    lib["Il"][3, 1, 2, 2000] += np.float32(1e-6)
    assert disc.DiscImu(lib).fingerprint()["library_sha256"] != fp["library_sha256"]
    assert disc.DiscImu(toy3["path"], lines=[0]).fingerprint()["library_sha256"] != fp["library_sha256"]
    # the file changes: unpickling refuses
    p = _save(tmp_path / "x.npz", toy3["lib"])
    blob = pickle.dumps(disc.DiscImu(p, fft="lazy"))
    os.utime(p, ns=(1, 1))
    with pytest.raises(RuntimeError, match="changed"):
        pickle.loads(blob)
    m = disc.DiscImu(toy3["path"]).memory()
    assert m["library"] > 0 and m["source"] == 0
    ml = D.memory()
    assert ml["source"] == ml["source_mapped"] > 0 and ml["library"] < m["library"] / 10


def test_discimu_imulibrary(toy3, tmp_path):
    """An ImuLibrary (library.py) as the library: the same bits as the file; its recorded LineSet and grid are taken
    (a grid with other points raises, another vshift is used); an integrator over a memory-mapped ImuLibrary pickles
    as a recipe in both modes (small; rebuilt from the file, identity-checked, its in-memory line record kept, its
    small arrays checked); over an ImuLibrary read without memory maps or over a dict it pickles its arrays and still
    works."""
    if not hasattr(lb, "ImuLibrary"):
        pytest.skip("library.ImuLibrary not available")
    L = lb.ImuLibrary.load(toy3["path"])
    mu, v, teff = _toy_points(3000, disc.DiscImu(toy3["path"]).t, seed=41)
    ref = disc.DiscImu(toy3["path"])
    k = ref.pairs(teff)
    r = ref(mu, v, *k)
    for kw in (dict(), dict(fft="lazy")):
        D = disc.DiscImu(L, **kw)
        g = D(mu, v, *k)
        _eq_lazy(g[0], r[0])
        for x, y in zip(g[2:4], r[2:4]):
            _eq(x, y)
        assert np.abs(g[1] - r[1]).max() <= 1e-15
        assert D.fingerprint()["library_sha256"] == ref.fingerprint()["library_sha256"]
    for kw in (dict(fft="lazy"), dict()):
        Dl = disc.DiscImu(L, **kw)
        assert Dl._ident == L._ident and Dl._lib_ref is None
        blob = pickle.dumps(Dl)
        assert len(blob) < 4000, (kw, len(blob))                   # precomputed: was its arrays (21.8 MB here)
        D2 = pickle.loads(blob)
        g = D2(mu, v, *k)
        _eq_lazy(g[0], r[0])
        for x, y in zip(g[2:4], r[2:4]):
            _eq(x, y)
        assert D2.fingerprint() == Dl.fingerprint()
    # an ImuLibrary of the legacy file whose params were set in memory: its lines survive the rebuild from the file
    L2 = lb.ImuLibrary.load(toy3["path"])
    L2.params.update(lines=LINES, lref=LREF.tolist())
    D2 = pickle.loads(pickle.dumps(disc.DiscImu(L2)))
    assert D2.lines.names == LINES and D2.lref.tolist() == LREF.tolist()
    # small arrays changed in memory: the rebuild from the file would differ, so unpickling refuses
    L3 = lb.ImuLibrary.load(toy3["path"])
    L3.teff_rep = L3.teff_rep + 1.0
    blob = pickle.dumps(disc.DiscImu(L3, fft="lazy"))
    with pytest.raises(RuntimeError, match="differ"):
        pickle.loads(blob)
    # the file replaced after the ImuLibrary was read: refused
    p = _save(tmp_path / "x.npz", toy3["lib"])
    blob = pickle.dumps(disc.DiscImu(lb.ImuLibrary.load(p)))
    os.utime(p, ns=(1, 1))
    with pytest.raises(RuntimeError, match="changed"):
        pickle.loads(blob)
    # read without memory maps: the arrays travel (documented size)
    Lr = lb.ImuLibrary.load(toy3["path"], mmap=False)
    Dr = disc.DiscImu(Lr)
    assert Dr._ident is None and len(pickle.dumps(Dr)) > Dr.memory()["library"]
    lib = toy3["lib"]
    keys = ("edges", "tmean", "count", "src", "idx_rep", "teff_rep", "rmax", "nnode", "s", "Ic", "Il")
    g2 = VelocityGrid(dv=1.0, vmax=2700.0, vshift=400.0)
    Lp = lb.ImuLibrary(*[lib[x] for x in keys], params=dict(lines=LINES, lref=LREF.tolist(), grid=g2.to_dict()))
    Dp = disc.DiscImu(Lp)
    assert Dp.grid.to_dict() == g2.to_dict() and Dp.lines.names == LINES
    with pytest.raises(ValueError, match="differs"):
        disc.DiscImu(Lp, grid=VelocityGrid(dv=1.0, vmax=2600.0, vshift=400.0))
    # vshift is an integration option: another one is used (as the same arrays given as a dict)
    g3 = VelocityGrid(dv=1.0, vmax=2700.0, vshift=300.0)
    for grid in (g3, g3.to_dict()):
        D3 = disc.DiscImu(Lp, grid=grid)
        assert D3.grid.to_dict() == g3.to_dict() and D3.vs == 300
        g = D3(mu, v, *k)
        gd = disc.DiscImu(dict(lib), grid=g3)(mu, v, *k)
        for x, y in zip(g[:4], gd[:4]):
            _eq(x, y)
        assert g[4] > r[4]                                           # more shifts clipped at 300 km/s
    Dd = disc.DiscImu(dict(lib), fft="lazy")
    g = pickle.loads(pickle.dumps(Dd))(mu, v, *k)
    _eq_lazy(g[0], r[0])


def test_discimu_recorded_grid_file(tmp_path):
    """A library file written by ImuLibrary.save records its grid and lines in '_meta': DiscImu given the path, the
    np.load()ed file or a dict of its members takes them (as from the ImuLibrary itself) -- a grid on dv 2 km/s with
    the M424 number of points is not mistaken for the M424 grid; a grid with other points raises, another vshift is
    used; pickling (the spawn rebuild) keeps grid and lines; dumps takes the lref from the integrator."""
    g2 = VelocityGrid(dv=2.0, vmax=5400.0, vshift=400.0)
    assert g2.ny == VelocityGrid().ny
    lib = _toy_imu(g2, seed=71)
    keys = ("edges", "tmean", "count", "src", "idx_rep", "teff_rep", "rmax", "nnode", "s", "Ic", "Il")
    Lp = lb.ImuLibrary(*[lib[x] for x in keys], params=dict(lines=LINES, lref=LREF.tolist(), grid=g2.to_dict()))
    p = Lp.save(str(tmp_path / "imu_dv2.npz"))
    mu, v, teff = _toy_points(3000, lib["teff_rep"][np.unique(lib["src"])], seed=72)
    ref = disc.DiscImu(Lp)
    k = ref.pairs(teff)
    r = ref(mu, v, *k)
    with np.load(p) as z:
        zd = {x: z[x] for x in z.files}
        Dz = disc.DiscImu(z, fft="lazy")
    for D in (disc.DiscImu(p), disc.DiscImu(lb.ImuLibrary.load(p)), Dz, disc.DiscImu(zd),
              pickle.loads(pickle.dumps(disc.DiscImu(p, fft="lazy")))):
        assert D.grid.to_dict() == g2.to_dict(), D
        assert isinstance(D.lines, LineSet) and D.lines.names == LINES and D.lref.tolist() == LREF.tolist()
        assert dp._integ_lref(D)[1].tolist() == LREF.tolist()
        g = D(mu, v, *k)
        _eq_lazy(g[0], r[0])
        for x, y in zip(g[2:4], r[2:4]):
            _eq(x, y)
        assert np.abs(g[1] - r[1]).max() <= 1e-15
    assert disc.DiscImu(p, lines=["HEII4200"]).built == (1,)
    for grid in (VelocityGrid(), dict(dv=1.0, vmax=2700.0, vshift=400.0)):
        with pytest.raises(ValueError, match="differs"):
            disc.DiscImu(p, grid=grid)
    D3 = disc.DiscImu(p, grid=dict(dv=2.0, vmax=5400.0, vshift=300.0))
    assert D3.grid.vshift == 300.0 and D3.vs == 150
    # the legacy layout (no '_meta'): nothing recorded, the default grid
    Lp.save(str(tmp_path / "legacy.npz"), meta=False)
    D4 = disc.DiscImu(str(tmp_path / "legacy.npz"))
    assert D4.grid.to_dict() == VelocityGrid().to_dict() and D4.lines is None and D4.lref is None

def _imu_worker_init(D):
    """Pool initializer: the worker state is the integrator itself ('spawn': unpickled = rebuilt from its file)."""
    return D


def _imu_worker_call(args):
    mu, v, teff, batch = args
    D = par.worker_state()
    if batch:
        r = D.integrate_los(mu, v, teff=teff)
        return [r[k] for k in ("F", "F0", "vmean_w", "sigma_w")]
    return list(D(mu, v, *D.pairs(teff))[:4])


@pytest.mark.parametrize("method", ["fork", "spawn"])
@pytest.mark.parametrize("fft", ["precomputed", "lazy"])
def test_discimu_pool_workers(toy3, method, fft):
    """A DiscImu passed to pool workers ('fork': inherited; 'spawn': pickled as its file name and options and rebuilt)
    gives the parent's results bit for bit, per call and for several lines of sight."""
    D = disc.DiscImu(toy3["path"], fft=fft)
    mu, v, teff = _toy_points(3000, D.t, seed=31)
    MU = np.stack([mu, -mu, np.roll(mu, 5)])
    V = np.stack([v, 0.5 * v, -v])
    ref1 = list(D(mu, v, *D.pairs(teff))[:4])
    r = D.integrate_los(MU, V, teff=teff)
    ref2 = [r[k] for k in ("F", "F0", "vmean_w", "sigma_w")]
    with par.make_pool(2, initializer=_imu_worker_init, initargs=(D,), start_method=method) as pool:
        got = pool.map(_imu_worker_call, [(mu, v, teff, False), (MU, V, teff, True)])
    for g, ref in zip(got, (ref1, ref2)):
        for x, y in zip(g, ref):
            _eq(x, y)


def _write_samples(d, theta, dumps, seed=0, nfast=2):
    rng = np.random.default_rng(seed)
    os.makedirs(d, exist_ok=True)
    for dump in dumps:
        N = theta.size
        teff = 38000.0 + 300.0 * rng.standard_normal(N)
        vel = {k: (40.0 * rng.standard_normal(N)).astype(np.float32) for k in ("ur", "uth", "uph")}
        vel["ur"][:nfast] = 600.0
        np.savez(os.path.join(d, "d{:04d}.npz".format(dump)), teff=teff.astype(np.float32), t_s=2835.0 * dump, **vel)


@pytest.mark.parametrize("fft", ["precomputed", "lazy"])
def test_discimu_run_disc_dumps_fork_spawn(toy3, tmp_path, fft):
    """run_disc_dumps with the DiscImu class as the integrator factory: serial, 'fork' and 'spawn' (each spawn worker
    builds its own DiscImu from the file and must match the parent's record, fingerprint included) give the same
    per-dump files byte for byte; the lazy run's F equals the default run's."""
    theta, phi = sph.fibonacci_sphere(3000)
    sdir = str(tmp_path / "samples")
    dl = [3200, 3201, 3202, 3203]
    _write_samples(sdir, theta, dl)
    files = {}
    for nproc, method in ((1, None), (2, "fork"), (2, "spawn")):
        out = str(tmp_path / "out_{}_{}".format(nproc, method))
        r = dp.run_disc_dumps(dl, sdir, out, "imu_" + fft, disc.DiscImu, (toy3["path"],), theta, phi, "thompson2024",
                              nproc=nproc, start_method=method, lref=LINESET, factory_kwargs=dict(fft=fft),
                              maxtasksperchild=2)
        assert sorted(r["done"]) == dl and r["fields"]["method"] == "imu"
        files[method] = [open(dp.dump_path(out, "imu_" + fft, d), "rb").read() for d in dl]
        custom = dp.read_run_record(os.path.join(out, "imu_" + fft))["params"]["integrator"]["custom"]
        assert custom["fft"] == fft and custom["method"] == "imu" and custom["library_sha256"]
    assert files[None] == files["fork"] == files["spawn"]
    if fft == "lazy":
        out_ref = str(tmp_path / "ref")
        dp.run_disc_dumps(dl, sdir, out_ref, "imu", disc.DiscImu, (toy3["path"],), theta, phi, "thompson2024",
                          lref=LINESET)
        for d in dl:
            a = np.load(dp.dump_path(out_ref, "imu", d))
            b = np.load(dp.dump_path(str(tmp_path / "out_1_None"), "imu_lazy", d))
            _eq_lazy(a["F"], b["F"], tol=6e-8)                         # float32 stored
            assert np.abs(a["F0"] - b["F0"]).max() <= 6e-8
            for k in ("vmean_w", "sigma_w", "n_clip", "n_lo", "n_hi", "wout", "node_range"):
                _eq(a[k], b[k])
            assert int(a["n_clip"].sum()) > 0


# ----------------------------------------------------------------------------------------------
# integrate_imu_nearest / uniform_star_imu vs the frozen fw_disc_imu.py
# ----------------------------------------------------------------------------------------------
def _write_rep_files(d, lib, b, grid, nrow=161):
    """OUT.<line>_VTV010 (lambda rounded to 0.01 A) and OUT_IMU.<line>_VTV010 (precise lambda, rays) of the
    representative of bin b: the flux profile is the rays' flux (I linear in s), so the uniform star nearly matches."""
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(b)
    for j, (ln, l0) in enumerate(zip(LINES, LREF)):
        yk = 1500.0 * np.sinh(4.4 * np.linspace(-1.0, 1.0, nrow)) / np.sinh(4.4) + rng.uniform(-0.3, 0.3)
        lam = l0 * np.exp(yk / C_KMS)
        n = int(lib["nnode"][b, j])
        s = lib["s"][b, j, :n]
        Ic = np.array([np.interp(yk, grid.y, lib["Ic"][b, j, k].astype(np.float64)) for k in range(n)])
        Il = np.array([np.interp(yk, grid.y, lib["Il"][b, j, k].astype(np.float64)) for k in range(n)])
        mf = np.linspace(0.0, 1.0, 4001)
        sf = np.sqrt(1.0 - mf ** 2)
        Lf = np.array([np.interp(sf, s, Il[:, i]) for i in range(nrow)])
        Cf = np.array([np.interp(sf, s, Ic[:, i]) for i in range(nrow)])
        fn = np.trapz(Lf * 2 * mf, mf, axis=1) / np.trapz(Cf * 2 * mf, mf, axis=1)
        with open(os.path.join(d, "OUT.{}_VTV010".format(ln)), "w") as fh:
            for k in range(nrow):
                fh.write("{:4d} {:11.5f} {:15.2f} {:19.6E} {:11.5f} {:11.5f}\n".format(k + 1, 1.2 - 0.015 * k, lam[k],
                                                                                   7.4e-7, fn[k], fn[k]))
            fh.write("  -1.08387540430798\n")
        with open(os.path.join(d, "OUT_IMU.{}_VTV010".format(ln)), "w") as fh:
            fh.write("# rays NP-1, core rays NC = {:4d} {:4d}\n".format(n, 1))
            fh.write("# p " + " ".join("{:.8E}".format(x) for x in s) + "\n")
            fh.write("# K, lambda, I_cont(p_1..p_NP-1), I_line(p_1..p_NP-1)\n")
            for k in range(nrow):
                fh.write("{:5d} {:.8f} ".format(k + 1, lam[k]) + " ".join("{:.6E}".format(x) for x in Ic[:, k]) + " "
                         + " ".join("{:.6E}".format(x) for x in Il[:, k]) + "\n")


@pytest.fixture(scope="module")
def legacy_imu_toy(tmp_path_factory):
    """A toy star (3000 equal-area points; profiles.npz with teff, theta, phi; d3200.npz samples, float32, a few
    shifts beyond 400 km/s; a dummy disc_los8.npz for the script's comparison messages), the toy library on the M424
    grid (bins of 150 K covering 38230 K) and the representative's OUT / OUT_IMU files."""
    root = tmp_path_factory.mktemp("legacy_imu")
    run, smp, runs = (str(root / x) for x in ("run", "samples", "runs"))
    for d in (run, smp, runs):
        os.makedirs(d)
    grid = VelocityGrid()
    lib = _toy_imu(grid, nb=8, t0=37500.0, dT=150.0, seed=21)
    lpath = _save(root / "imu.npz", lib)
    rng = np.random.default_rng(22)
    N = 3000
    theta, phi = sph.fibonacci_sphere(N)
    teff = 38100.0 + 350.0 * rng.standard_normal(N)
    np.savez(os.path.join(run, "profiles.npz"), teff=teff, theta=theta, phi=phi, idx=np.arange(N))
    vel = {k: (40.0 * rng.standard_normal(N)).astype(np.float32) for k in ("ur", "uth", "uph")}
    vel["ur"][:3] = 700.0
    np.savez(os.path.join(smp, "d3200.npz"), teff=teff.astype(np.float32), t_s=2835.0 * 3200, **vel)
    z = np.zeros((8, 3, 5))
    np.savez(os.path.join(run, "disc_los8.npz"), F=np.ones((8, 3, grid.ny)), diag_F=z, diag_F0=z, sigma_w=z[:, :, 0])
    bref = int(np.clip(np.digitize(38230.0, lib["edges"]) - 1, 0, lib["src"].size - 1))
    _write_rep_files(os.path.join(runs, "P{0:06d}/P{0:06d}".format(int(lib["idx_rep"][bref]))), lib, bref, grid)
    return dict(root=str(root), run=run, samples=smp, runs=runs, lib=lib, lib_path=lpath, theta=theta, phi=phi,
                teff=teff, bref=bref)


def test_integrate_imu_nearest_matches_legacy(legacy_imu_toy, monkeypatch, tmp_path):
    """The frozen fw_disc_imu.py run as a script (fw_disc.RUN, SAMPLES patched to the toy; the representatives'
    directory replaced) vs integrate_imu_nearest: F, F0, vmean_w, sigma_w, the diagnostics (frozen options) and the
    uniform-star check bit for bit; save_disc_los writes the same members."""
    fd = _legacy_fw_disc()
    T = legacy_imu_toy
    monkeypatch.setattr(fd, "RUN", T["run"])
    monkeypatch.setattr(fd, "SAMPLES", T["samples"])
    _run_legacy_script("fw_disc_imu.py", ["--lib", T["lib_path"]],
                       replace=[('"/scratch/ppathak/fastwind_imu/runs"', repr(T["runs"]))])
    ref = np.load(os.path.join(T["run"], "disc_los8_imu.npz"))
    logs = []
    r = disc.integrate_imu_nearest(T["lib_path"], os.path.join(T["run"], "profiles.npz"),
                                   os.path.join(T["samples"], "d3200.npz"), LINESET, runs=T["runs"],
                                   diag_kw=FROZEN_DIAG_KW, log=logs.append)
    for k in ("F", "F0", "vmean_w", "sigma_w", "diag_F", "diag_F0"):
        _eq(r[k], ref[k])
    _eq(r["y"], ref["Y"])
    _eq(r["los"], ref["los"])
    assert list(r["checks"]) == list(ref["check_keys"]) == ["uniform_" + x for x in LINES]
    _eq(np.array(list(r["checks"].values())), ref["check_vals"])
    assert 0 < max(r["checks"].values()) < 2e-3                       # interpolation of the coarse toy OUT grid
    out = disc.save_disc_los(str(tmp_path / "imu8.npz"), r)
    got = np.load(out)
    assert [k for k in got.files if k != "_meta"] == list(ref.files)
    for k in ref.files:
        _eq(got[k], ref[k])
    assert any("uniform star" in x for x in logs)
    # memory-mapped library, a mapping and explicit arrays: the same bits
    r2 = disc.integrate_imu_nearest(dict(T["lib"]), None, os.path.join(T["samples"], "d3200.npz"), LINESET,
                                    teff=T["teff"], theta=T["theta"], phi=T["phi"], diag_kw=FROZEN_DIAG_KW)
    for k in ("F", "F0", "vmean_w", "sigma_w"):
        _eq(r2[k], r[k])
    assert r2["checks"] == {}
    # samples of another dump are refused
    with pytest.raises(ValueError, match="differ"):
        disc.integrate_imu_nearest(T["lib_path"], dict(teff=T["teff"] + 500.0, theta=T["theta"], phi=T["phi"]),
                                   os.path.join(T["samples"], "d3200.npz"), LINESET)


def test_uniform_star_imu(legacy_imu_toy):
    """The uniform-star check alone: the bin of the reference T_eff', its representative's directory, F vs the
    FASTWIND flux of the rays (small), and errors."""
    T = legacy_imu_toy
    mu = sph.project_los(T["theta"], T["phi"], "thompson2024", method="matvec")[0][0]
    u = disc.uniform_star_imu(T["lib_path"], mu, LINESET, runs=T["runs"])
    assert u["bin"] == T["bref"] and u["idx_rep"] == int(T["lib"]["idx_rep"][T["bref"]])
    assert u["dir"].startswith(T["runs"]) and set(u["checks"]) == {"uniform_" + x for x in LINES}
    assert max(u["checks"].values()) < 2e-3
    u2 = disc.uniform_star_imu(T["lib_path"], mu, LINESET, rep_dir=u["dir"])
    assert u2["checks"] == u["checks"]
    with pytest.raises(ValueError, match="rep_dir"):
        disc.uniform_star_imu(T["lib_path"], mu, LINESET)
    with pytest.raises(TypeError):
        disc.uniform_star_imu(T["lib_path"], mu, LREF, runs=T["runs"])
    # integrate_imu_nearest: the check needs line names (OUT.<line>_VTV010); a plain lref is refused up front
    smp = os.path.join(T["samples"], "d3200.npz")
    for kw in (dict(runs=T["runs"]), dict(rep_dir=u["dir"])):
        with pytest.raises(ValueError, match="LineSet"):
            disc.integrate_imu_nearest(T["lib_path"], None, smp, LREF, teff=T["teff"], theta=T["theta"],
                                       phi=T["phi"], **kw)


# ----------------------------------------------------------------------------------------------
# M424 regressions
# ----------------------------------------------------------------------------------------------
def _m424_imu():
    if not os.path.exists(IMU_LIB_M424):
        pytest.skip("M424 intensity library not available: {}".format(IMU_LIB_M424))
    return IMU_LIB_M424


def _samples(dump):
    p = os.path.join(SAMPLES_M424, "d{:04d}.npz".format(dump))
    if not os.path.exists(p):
        pytest.skip("M424 product not available: {}".format(p))
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def m424_imu():
    """The default DiscImu of the M424 library (as fw_disc_dumps.py --method imu; ~1 min, ~8 GB) and the 'matmul'
    projections of points.npz (fw_disc_dumps.py)."""
    path = _m424_imu()
    pts = m424_path("run", "points.npz")
    th, ph = np.asarray(npz_member_memmap(pts, "theta")), np.asarray(npz_member_memmap(pts, "phi"))
    t0 = time.time()
    with dp._blas_limit(1):                     # as run_disc_dumps (the F0 matrix-vector products)
        D = disc.DiscImu(path)
    print("DiscImu M424 set-up {:.0f} s, max RSS {:.1f} GB".format(
        time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
    return D, sph.project_los(th, ph, "thompson2024")


@pytest.mark.m424
@pytest.mark.slow
@pytest.mark.parametrize("dump", [3200, 4000, 4800])
def test_m424_discimu_dumps(m424_imu, dump):
    """DiscImu (default) through dumps.disc_dump reproduces the stored products of the imu run of fw_disc_dumps.py:
    F, F0 bit for bit after the float32 cast, and every other member (diagnostics, velocity moments, n_clip,
    coverage, T_eff' statistics) exactly."""
    D, (MU, TN, PN) = m424_imu
    out = np.load(m424_path("disc", "imu", "d{:04d}.npz".format(dump)))
    smp = _samples(dump)
    smp["dump"] = dump
    r = dp.disc_dump(D, smp, MU, TN, PN, lref=LINESET)
    arr = dp.dump_arrays(r, dp.run_fields(D))
    assert list(arr) == list(out.files)
    for k in out.files:
        assert np.asarray(arr[k]).dtype == out[k].dtype, k
        _eq(np.asarray(arr[k]), out[k])


@pytest.mark.m424
@pytest.mark.slow
def test_m424_run_disc_dumps_imu(tmp_path):
    """run_disc_dumps with the DiscImu class as factory (default options, 2 'fork' workers) writes the production
    imu/d3200.npz and d3201.npz of fw_disc_dumps.py --method imu byte for byte."""
    path = _m424_imu()
    pts = m424_path("run", "points.npz")
    th, ph = np.asarray(npz_member_memmap(pts, "theta")), np.asarray(npz_member_memmap(pts, "phi"))
    dl = [3200, 3201]
    for d in dl:
        m424_path("disc", "imu", "d{:04d}.npz".format(d))
    r = dp.run_disc_dumps(dl, SAMPLES_M424, str(tmp_path), None, disc.DiscImu, (path,), th, ph, "thompson2024",
                          nproc=2, start_method="fork", lref=LINESET)
    assert r["name"] == "imu" and sorted(r["done"]) == dl
    for d in dl:
        with open(dp.dump_path(str(tmp_path), "imu", d), "rb") as f:
            got = f.read()
        with open(m424_path("disc", "imu", "d{:04d}.npz".format(d)), "rb") as f:
            assert got == f.read()


@pytest.mark.m424
@pytest.mark.slow
def test_m424_discimu_lowmem(m424_imu):
    """The low-memory modes against the default on dumps 3200, 4000, 4800 (8 lines of sight): fft='lazy' (batch over
    the lines of sight) F (bit for bit on x86-64), vmean_w, sigma_w bit for bit, F0 to rounding; dtype='float32'
    (precomputed and lazy) well below the 2e-7 target. The diagnostics of the per-dump files (dumps.disc_dump: moments
    over the whole grid) amplify the float32 error: bounds on their deviations (EW [A], v1, sigma [km/s])."""
    D, (MU, TN, PN) = m424_imu
    modes = dict(lazy=dict(fft="lazy"), f32=dict(dtype="float32"), lazy32=dict(fft="lazy", dtype="float32"))
    with dp._blas_limit(1):
        ints = {name: disc.DiscImu(IMU_LIB_M424, **kw) for name, kw in modes.items()}
    dev = {k: [0.0, 0.0] for k in modes}
    ddev = {k: np.zeros(len(DIAG_KEYS)) for k in modes}
    for name, Dm in ints.items():
        assert Dm.fingerprint()["library_sha256"] == D.fingerprint()["library_sha256"]
    y = D.grid.y
    for dump in (3200, 4000, 4800):
        smp = _samples(dump)
        V = sph.los_velocity(smp["ur"], smp["uth"], smp["uph"], MU, TN, PN)
        pairs = D.pairs(smp["teff"].astype(np.float64))
        ref = D.integrate_los(MU, V, pairs=pairs)
        dref = diagnostics_array(ref["F"], y, LREF, vwin=None)            # as dumps.disc_dump (whole-grid window)
        for name, Dm in ints.items():
            got = Dm.integrate_los(MU, V, pairs=pairs)
            dev[name][0] = max(dev[name][0], np.abs(got["F"] - ref["F"]).max())
            dev[name][1] = max(dev[name][1], np.abs(got["F0"] - ref["F0"]).max())
            dg = diagnostics_array(got["F"], y, LREF, vwin=None)
            ddev[name] = np.maximum(ddev[name], np.abs(dg - dref).max(axis=(0, 1)))
            if name == "lazy":
                _eq_lazy(got["F"], ref["F"])
                for k in ("vmean_w", "sigma_w", "n_clip"):
                    _eq(got[k], ref[k])
            del got
        del ref, V
    print("M424 deviations from the default (max |dF|, max |dF0|):", dev)
    print("diag_F deviations ({}):".format(", ".join(DIAG_KEYS)), {k: x.tolist() for k, x in ddev.items()})
    # measured 2026-10-02: lazy F 0, F0 7.1e-15; float32 precomputed 2.0e-9, lazy 7.1e-9 (F0 7.1e-15); diagnostics
    # (EW [A], v1, sigma [km/s]): lazy 0; float32 precomputed 6.8e-8, 5.0e-5, 1.3e-3; lazy 2.6e-7, 4.8e-4, 6.3e-3
    assert dev["lazy"][0] <= (0.0 if LAZY_BITWISE else 1e-14) and dev["lazy"][1] <= 1e-13
    assert max(dev["f32"]) <= 2e-8 and max(dev["lazy32"]) <= 5e-8
    keys = list(DIAG_KEYS)
    for name, bounds in (("f32", dict(ew=3e-7, v1=2e-4, sigma=4e-3)), ("lazy32", dict(ew=1e-6, v1=2e-3, sigma=2e-2))):
        for k, b in bounds.items():
            assert ddev[name][keys.index(k)] <= b, (name, k, ddev[name])
    if LAZY_BITWISE:
        assert not ddev["lazy"].any()


def _direct_conv(H, I, vs, ny, rows):
    """sum over the given rows r and shift columns c of H[r, c] I_r(y + (c - vs) dv), edge values beyond the grid: the
    disc sum of DiscImu without FFTs (each nonzero column a matrix-vector product)."""
    P = np.pad(np.asarray(I[rows], dtype=np.float64), ((0, 0), (vs, vs)), mode="edge")
    Hr = H[rows]
    out = np.zeros(ny)
    for c in range(Hr.shape[1]):
        sel = np.flatnonzero(Hr[:, c])
        if sel.size:
            out += Hr[sel, c] @ P[sel, c:c + ny]
    return out


@pytest.mark.m424
@pytest.mark.slow
def test_m424_discimu_extrapolated_pairs(m424_imu, monkeypatch):
    """The V3 extrapolation pairs (validate.v3_tails, margin 300 K) on dumps 3334 and 4391 (the M424 dumps with the
    most points beyond the nodes / the hottest point): rows with net weight <= 0 but nonzero weights exist (the
    legacy rule dropped them: 24 and 66 (line of sight, line, row) cases); lazy F equals the default F (bit for bit on
    x86-64) and the lazy F0 (chunked over the used rows) the default F0 (all rows) to rounding; on dump 4391 the
    default F of line of sight 1 equals a direct convolution (no FFT) over every row with a nonzero weight; there the
    legacy rule is off by more than the V3 tolerance (5e-6; 3334: by > 1e-6)."""
    D, (MU, TN, PN) = m424_imu
    with dp._blas_limit(1):
        Dl = disc.DiscImu(IMU_LIB_M424, fft="lazy")
    for dump in (3334, 4391):
        smp = _samples(dump)
        teff = smp["teff"].astype(np.float64)
        pairs = _v3_pairs(D.t, teff, 300.0)
        assert ((pairs[2] < 0) | (pairs[2] > 1)).any()
        V = sph.los_velocity(smp["ur"], smp["uth"], smp["uph"], MU, TN, PN)
        ref = D.integrate_los(MU, V, pairs=pairs)
        lz = Dl.integrate_los(MU, V, pairs=pairs)
        _eq_lazy(lz["F"], ref["F"])
        assert np.abs(lz["F0"] - ref["F0"]).max() <= 1e-13
        del lz
        ndrop = np.zeros((MU.shape[0], D.nl), int)
        for k in range(MU.shape[0]):
            p = D._prepare(MU[k], V[k], *pairs)
            for j in range(D.nl):
                H = D._hist(j, p["m"], p["s"], p["sh"], p["k0"], p["k1"], p["a"])[2]
                ndrop[k, j] = int(((H != 0).any(axis=1) & ~(H.sum(axis=1) > 0)).sum())
                del H
        print("dump {}: rows the legacy rule dropped (los x line): {} (total {})".format(dump, ndrop.tolist(),
                                                                                        int(ndrop.sum())))
        assert ndrop.sum() == {3334: 24, 4391: 66}[dump]
        with monkeypatch.context() as mp:
            mp.setattr(disc, "_used_rows", _legacy_used_rows)
            leg = D.integrate_los(MU, V, pairs=pairs)
        dev = np.abs(leg["F"] - ref["F"]).max(axis=(0, 2))
        print("dump {}: legacy rule, max|dF| per line {}".format(dump, dev))
        # measured 2026-10-02: 3334 2.2e-6, 4391 1.1e-5 (the V3 tolerance is 5e-6)
        assert dev.max() > {3334: 1e-6, 4391: 5e-6}[dump]
        del leg
        if dump == 4391:
            kk = 0
            p = D._prepare(MU[kk], V[kk], *pairs)
            for j in range(D.nl):
                H = D._hist(j, p["m"], p["s"], p["sh"], p["k0"], p["k1"], p["a"])[2]
                rr = np.flatnonzero((H != 0).any(axis=1))
                Fd = _direct_conv(H, D.I0[j][0], D.vs, D.ny, rr) / _direct_conv(H, D.I0[j][1], D.vs, D.ny, rr)
                dd = np.abs(Fd - ref["F"][kk, j]).max()
                print("dump {} los{} line {}: direct convolution over {} rows vs F: {:.1e}".format(dump, kk + 1, j,
                                                                                                 rr.size, dd))
                assert dd <= 1e-13                                     # measured 3.1e-15
                del H
        del ref, V


_STATUS_CODE = """
def status():
    out = {}
    for l in open('/proc/self/status'):
        k = l.split(':')[0]
        if k in ('VmHWM', 'VmRSS', 'RssAnon', 'RssFile'):
            out[k] = int(l.split()[1]) / 1e6
    return out
"""


def _run_measured(code, tmp_path, name):
    """Run code in a fresh Python process (the checkout first on sys.path; ``status()`` gives VmHWM, VmRSS, RssAnon,
    RssFile [GB]); returns the JSON it writes to the file ``OUT``."""
    out = str(tmp_path / (name + ".json"))
    src = "import sys\nsys.path.insert(0, {!r})\nOUT = {!r}\n".format(ROOT, out) + _STATUS_CODE + textwrap.dedent(code)
    subprocess.run([sys.executable, "-c", src], check=True)
    with open(out) as f:
        return json.load(f)


@pytest.mark.m424
@pytest.mark.slow
def test_m424_discimu_lazy_memory(tmp_path):
    """fft='lazy' in a fresh process: set-up, one dump of 8 lines of sight with integrate_los, then the same dump
    through dumps.disc_dump (one call per line of sight), within 3 GB peak RSS (VmHWM; ~1.6 GB of it are clean pages
    of the memory-mapped library, 0.83 GB of them mapped at set-up when I_c at the line centre is read)."""
    path = _m424_imu()
    pts = m424_path("run", "points.npz")
    smp = os.path.join(SAMPLES_M424, "d4000.npz")
    if not os.path.exists(smp):
        pytest.skip("M424 samples not available")
    code = """
        import json, time
        import numpy as np
        from ppmpy.synspec import disc, dumps as dp, sphere as sph
        from ppmpy.synspec.io import npz_member_memmap
        from ppmpy.synspec.spectral import LineSet
        r = dict(base=status())
        t0 = time.time()
        D = disc.DiscImu({path!r}, fft='lazy')
        r['setup'], r['t_setup'] = status(), time.time() - t0
        th, ph = (np.asarray(npz_member_memmap({pts!r}, k)) for k in ('theta', 'phi'))
        MU, TN, PN = sph.project_los(th, ph, 'thompson2024')
        with np.load({smp!r}) as z:
            smp = {{k: z[k] for k in z.files}}
        V = sph.los_velocity(smp['ur'], smp['uth'], smp['uph'], MU, TN, PN)
        r['inputs'] = status()
        t1 = time.time()
        out = D.integrate_los(MU, V, teff=smp['teff'].astype(np.float64))
        r['batch'], r['t_batch'], r['cache_rows'] = status(), time.time() - t1, D._last_cache['rows']
        del out, V
        t1 = time.time()
        dd = dp.disc_dump(D, smp, MU, TN, PN, lref=LineSet({lines!r}, {lref!r}))
        r['disc_dump'], r['t_disc_dump'] = status(), time.time() - t1
        json.dump(r, open(OUT, 'w'))
    """.format(path=path, pts=pts, smp=smp, lines=LINES, lref=LREF.tolist())
    r = _run_measured(code, tmp_path, "lazy")
    print("lazy (GB): imports {base}; set-up {t_setup:.1f} s {setup}; inputs {inputs}; integrate_los "
          "{t_batch:.1f} s {batch} (row cache {cache_rows} rows); disc_dump {t_disc_dump:.1f} s {disc_dump}"
          .format(**r))
    # measured 2026-10-02 (VmHWM): set-up 0.86 GB (RssFile 0.83), integrate_los 2.72 GB; disc_dump alone (fresh
    # process) 2.17 GB, RssAnon 0.37 / RssFile 1.59 GB after it
    assert r["batch"]["VmHWM"] <= 3.0 and r["disc_dump"]["VmHWM"] <= 3.0
    assert r["setup"]["RssAnon"] <= 0.2 and r["disc_dump"]["RssAnon"] <= 1.5


@pytest.mark.m424
@pytest.mark.slow
def test_m424_integrate_imu_nearest(tmp_path):
    """integrate_imu_nearest (dump 3200, profiles.npz T_eff', theta, phi; d3200.npz velocities; the representatives'
    OUT / OUT_IMU files) reproduces disc_los8_imu.npz bit for bit: F, F0, vmean_w, sigma_w, diag_F, diag_F0 (legacy
    diagnostics options) and the uniform-star check values (3.9e-6, 4.8e-6, 6.1e-6). Run in a fresh process: peak RSS
    (VmHWM) within 3 GB."""
    lib = _m424_imu()
    ref = np.load(m424_path("run", "disc_los8_imu.npz"))
    prof = m424_path("run", "profiles.npz")
    smp = os.path.join(SAMPLES_M424, "d3200.npz")
    if not os.path.exists(smp) or not os.path.isdir(IMU_RUNS_M424):
        pytest.skip("M424 samples or representatives not available")
    res = str(tmp_path / "nearest.npz")
    code = """
        import json, time
        import numpy as np
        from ppmpy.synspec import disc
        from ppmpy.synspec.spectral import LineSet
        r = dict(base=status())
        t0 = time.time()
        out = disc.integrate_imu_nearest({lib!r}, {prof!r}, {smp!r}, LineSet({lines!r}, {lref!r}), runs={runs!r},
                                         log=print)
        r['run'], r['t'] = status(), time.time() - t0
        np.savez({res!r}, check_keys=np.array(list(out['checks'])), check_vals=np.array(list(out['checks'].values())),
                 **{{k: out[k] for k in ('F', 'F0', 'vmean_w', 'sigma_w', 'diag_F', 'diag_F0')}})
        json.dump(r, open(OUT, 'w'))
    """.format(lib=lib, prof=prof, smp=smp, lines=LINES, lref=LREF.tolist(), runs=IMU_RUNS_M424, res=res)
    m = _run_measured(code, tmp_path, "nearest")
    print("integrate_imu_nearest M424: {t:.0f} s; memory [GB]: start {base}, end {run}".format(**m))
    r = np.load(res)
    for k in ("F", "F0", "vmean_w", "sigma_w", "diag_F", "diag_F0"):
        _eq(r[k], ref[k])
    _eq(r["check_keys"], ref["check_keys"])
    _eq(r["check_vals"], ref["check_vals"])
    np.testing.assert_allclose(r["check_vals"], [3.9e-6, 4.8e-6, 6.1e-6], rtol=0.03)
    assert m["run"]["VmHWM"] <= 3.0                                    # measured 2.4 GB
