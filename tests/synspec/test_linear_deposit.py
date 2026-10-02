"""Tests of the sub-grid Doppler shifts (deposit='linear') of ppmpy.synspec: VelocityGrid.shift_steps, DiscFlux,
DiscImu, the run drivers (flux_integrator, imu_integrator, run_disc_dumps, run_fields) and
validate.brute_force(continuous=True).

PP 2026-10-02: new. The default deposit 'nearest' rounds every Doppler shift to whole grid steps (the M424 products);
for line-of-sight velocities below the grid step (M487, IGW only: v_los rms 0.45 km/s on the 1 km/s grid) that puts
most points at shift 0. 'linear' splits each point's weight between the two neighbouring steps (linear interpolation
of the shifted profile between grid points).

Synthetic (any machine): shift_steps (split, clipping, 'nearest' unchanged); a single point with v = 0.3 km/s
('linear' = the linearly interpolated shifted profile, 'nearest' = the unshifted one); the error against an
analytically shifted profile falls as dv^2 ('linear') and dv ('nearest'); a toy star with the toy velocities x 0.02,
x 0.01 (M487-like) and x 1: 'linear' equals the continuous brute force to rounding, 'nearest' is off by 6-44 % of the
velocity signal F - F0 at the small velocities; DiscImu 'linear' against a per-point continuous brute force (every
mode) and = DiscFlux for mu-independent intensities; the drivers (run names '_lin', run records, refusal of mixing and
of legacy adoption, serial = spawn).

Reviewer fixes (PP 2026-10-02): with_deposit twins (DiscFlux, DiscImu: shared arrays, bit for bit a fresh build,
pickling); the validation of a 'linear' integrator (V1, V2, V4, V5 on its 'nearest' twin with the nearest
integrator's values, V6 and the brute force checking the deposit, run_validation; the general toy of
test_validate.py with its velocities x 1 and x 0.02); the intensity brute force with continuous shifts (against a
per-point reference and a 'linear' DiscImu, clipped shifts, brute_force_imu_check following the deposit);
brute_force(continuous='cubic') (Catmull-Rom: = linear at whole steps, far closer to a smooth analytic shift) and the
clipping of continuous shifts; the driver's deposit checks (an explicit name with all dumps present, the per-dump
member 'deposit' refusing mixed adoption both ways, a positional deposit, the parameter order).

M424 (marker m424; ~15 s in all): DiscFlux with deposit='nearest' given explicitly reproduces the production flux files
of dumps 3200, 4000 bit for bit; 'linear' and 'nearest' against the continuous brute force on 20 000-point subsets of
every line of sight; M487 dump 3200 (its sphere samples of moms.sample_moms_sphere, slab backend, 3550 Mm;
PPMPY_SYNSPEC_M487_SAMPLES) with the M424 flux library: the same comparison and the signal size. Slow (m424, slow;
~2-3 min, ~10 GB): DiscImu (default) reproduces the production imu/d3200.npz bit for bit, and its 'linear' twin
equals the intensity brute force with continuous shifts on 20 000-point subsets of M424 and M487 dump 3200.
"""
import os
import pickle
import types

import numpy as np
import pytest

from conftest import m424_path
from ppmpy.synspec import disc
from ppmpy.synspec import dumps as dm
from ppmpy.synspec import library as lb
from ppmpy.synspec import sphere as sph
from ppmpy.synspec import testing as ts
from ppmpy.synspec import validate as vd
from ppmpy.synspec.conventions import C_KMS
from ppmpy.synspec.io import npz_member_memmap
from ppmpy.synspec.spectral import DEPOSITS, LineSet, VelocityGrid, check_deposit

LREF = np.array([4026.22, 4199.90, 4921.93])
LINES = ["HEI4026", "HEII4200", "HEI4922"]
LINESET = LineSet(LINES, LREF)
SMALL = VelocityGrid(dv=1.0, vmax=300.0, vshift=40.0)
SAMPLES_M424 = os.environ.get("PPMPY_SYNSPEC_M424_SAMPLES", "/scratch/ppathak/fastwind_sphere/samples_r4050_N1236544")
SAMPLES_M487 = os.environ.get("PPMPY_SYNSPEC_M487_SAMPLES",
                              "/scratch/ppathak/fastwind_sphere/M487_r3550_N1236544/samples")


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _xsteps(v, dv):
    """The exact Doppler shift in grid steps."""
    return -C_KMS * np.log(1.0 - np.asarray(v, dtype=np.float64) / C_KMS) / dv


def _gauss(y, c, s, a):
    return a * np.exp(-0.5 * ((y - c) / s) ** 2)


def _flat_nodes(grid, sig=8.0, amp=0.5, nl=1):
    """Two nodes with the same Gaussian line (T_eff' plays no role), F_c = 1."""
    f = 1.0 - _gauss(grid.y, 0.0, sig, amp)
    prof = np.broadcast_to(f, (2, nl, grid.ny)).copy()
    return dict(t=np.array([38000.0, 38100.0]), fc=np.ones((2, nl)), prof=prof)


def _toy_nodes(grid, nn=12, nl=3, seed=0, sigmas=(20.0, 45.0, 12.0)):
    """Library nodes whose Gaussian lines change with T_eff' (depth exactly 0 near the grid ends)."""
    rng = np.random.default_rng(seed)
    t = 37800.0 + np.cumsum(rng.uniform(20.0, 80.0, nn))
    x = (t - t.mean()) / t.std()
    prof = np.empty((nn, nl, grid.ny))
    fc = np.empty((nn, nl))
    for j in range(nl):
        prof[:, j] = 1.0 - _gauss(grid.y[None, :], (3.0 * x + 2.0 * j)[:, None],
                                  (sigmas[j % 3] * (1.0 + 0.05 * x))[:, None], (0.4 + 0.05 * x * (j + 1))[:, None])
        fc[:, j] = (t / 38000.0) ** 4 * (1.0 + 0.03 * j)
    return dict(t=t, fc=fc, prof=prof)


def _one(D, v, mu=1.0, a=0.0):
    """Profiles of a single visible point."""
    return D(np.array([mu]), np.array([float(v)]), np.array([0]), np.array([1]), np.array([a]))


def _toy_imu(grid, nl=2, nb=5, K=5, seed=0, dT=150.0, t0=37800.0):
    """A small intensity library (members of imu_library_dT10.npz; no empty bins): K rays s = p / R_max in [0, 1],
    float32 intensities I_c = c (1 - 0.5 (1 - mu)) (1 + 0.02 y / vmax) and I_l = I_c (1 - d), d a Gaussian line whose
    depth, width and centre change with T_eff' and mu, zero near the grid ends."""
    rng = np.random.default_rng(seed)
    y = grid.y
    edges = t0 + dT * np.arange(nb + 1)
    src = np.arange(nb)
    teff_rep = edges[:-1] + rng.uniform(0.2, 0.8, nb) * dT
    s = np.empty((nb, nl, K))
    Il = np.empty((nb, nl, K, y.size), np.float32)
    Ic = np.empty((nb, nl, K, y.size), np.float32)
    for b in range(nb):
        x = (teff_rep[b] - t0) / (nb * dT)
        for j in range(nl):
            s[b, j] = np.concatenate([[0.0], np.sort(rng.uniform(0.05, 0.98, K - 2)), [1.0]])
            for k in range(K):
                mu = np.sqrt(1.0 - s[b, j, k] ** 2)
                ic = (1.0 + 0.3 * x + 0.1 * j) * (1.0 - 0.5 * (1.0 - mu)) * (1.0 + 0.02 * y / grid.vmax)
                d = (0.3 + 0.1 * j) * (1.0 + 0.2 * x) * (0.6 + 0.4 * mu) * np.exp(
                    -0.5 * ((y - (5.0 * j + 3.0 * x + 2.0 * mu)) / ((15.0 + 20.0 * j) * (1.0 + 0.1 * mu))) ** 2)
                Ic[b, j, k] = ic
                Il[b, j, k] = ic * (1.0 - d)
    return dict(edges=edges, tmean=0.5 * (edges[:-1] + edges[1:]), count=np.full(nb, 20.0), src=src,
                idx_rep=1000 + src, teff_rep=teff_rep, rmax=np.ones((nb, nl)), nnode=np.full((nb, nl), K, np.int64),
                s=s, Ic=Ic, Il=Il)


def _brute_imu_continuous(lib, grid, mu, v, teff):
    """Per-point reference of DiscImu with continuous shifts: each visible point's intensities interpolated linearly in
    T_eff' (clamped) and in s between its node's rays, evaluated at y + x (x = -c ln(1 - v/c), not rounded) by linear
    interpolation with the edge values beyond the grid (np.interp), summed with weight mu."""
    t = lib["teff_rep"]
    S = lib["s"]
    Il, Ic = lib["Il"].astype(np.float64), lib["Ic"].astype(np.float64)
    nl, ny = S.shape[1], grid.ny
    F, F0 = np.zeros((nl, ny)), np.zeros((nl, ny))
    vis = np.where(mu > 0)[0]
    k0 = np.clip(np.searchsorted(t, teff, side="right") - 1, 0, t.size - 2)
    a = np.clip((teff - t[k0]) / (t[k0 + 1] - t[k0]), 0.0, 1.0)
    x = -C_KMS * np.log(1.0 - v / C_KMS)
    for j in range(nl):
        num, den, n0, d0 = np.zeros(ny), np.zeros(ny), np.zeros(ny), np.zeros(ny)
        for i in vis:
            si = np.sqrt(max(1.0 - mu[i] ** 2, 0.0))
            rl, rc = np.zeros(ny), np.zeros(ny)
            for node, w in ((k0[i], 1.0 - a[i]), (k0[i] + 1, a[i])):
                sn = S[node, j]
                kk = min(max(np.searchsorted(sn, si, side="right") - 1, 0), sn.size - 2)
                tt = min(max((si - sn[kk]) / (sn[kk + 1] - sn[kk]), 0.0), 1.0)
                rl += w * ((1 - tt) * Il[node, j, kk] + tt * Il[node, j, kk + 1])
                rc += w * ((1 - tt) * Ic[node, j, kk] + tt * Ic[node, j, kk + 1])
            num += mu[i] * np.interp(grid.y + x[i], grid.y, rl)
            den += mu[i] * np.interp(grid.y + x[i], grid.y, rc)
            n0 += mu[i] * rl
            d0 += mu[i] * rc
        F[j], F0[j] = num / den, n0 / d0
    return F, F0


def _samples(root, dump):
    p = os.path.join(root, "d{:04d}.npz".format(dump))
    if not os.path.exists(p):
        pytest.skip("samples not available: {}".format(p))
    with np.load(p) as z:
        return {k: z[k].astype(np.float64) for k in ("teff", "ur", "uth", "uph")}


def _subset_mu(MU, nsub, seed):
    """MU with all but nsub random visible points of every line of sight hidden (mu = 0)."""
    rng = np.random.default_rng(seed)
    M = np.zeros_like(MU)
    for k in range(MU.shape[0]):
        vis = np.where(MU[k] > 0)[0]
        sub = rng.choice(vis, min(nsub, vis.size), replace=False)
        M[k, sub] = MU[k, sub]
    return M


# ----------------------------------------------------------------------------------------------
# VelocityGrid.shift_steps
# ----------------------------------------------------------------------------------------------
def test_shift_steps_linear():
    """'linear' gives s0 = floor(x) and w1 = x - s0 inside +-nshift; beyond, all weight at the clipped step; 'nearest'
    is the original expression, bit for bit."""
    g = VelocityGrid(dv=0.5, vmax=100.0, vshift=20.0)
    v = np.concatenate([np.linspace(-30.0, 30.0, 2001), [0.0, 0.3, -0.3, 1e-9, -1e-9]])
    x = _xsteps(v, g.dv)
    s0, w1, clip = g.shift_steps(v, deposit="linear")
    assert s0.dtype == np.int64 and w1.dtype == np.float64 and clip.dtype == bool
    ok = ~clip
    np.testing.assert_array_equal(s0[ok], np.floor(x[ok]).astype(np.int64))
    assert np.abs(s0[ok] + w1[ok] - x[ok]).max() <= 1e-12
    assert np.all((w1 >= 0.0) & (w1 <= 1.0)) and np.all((s0 >= -g.nshift) & (s0 <= g.nshift - 1))
    np.testing.assert_array_equal(clip, np.abs(x) > g.nshift)
    eff = s0 + w1                                             # the effective shift in steps
    np.testing.assert_array_equal(eff[clip], np.clip(x[clip], -g.nshift, g.nshift).round())
    assert np.all(np.abs(eff[clip]) == g.nshift)
    # edges: x exactly +-nshift
    vb = -C_KMS * np.expm1(-np.array([g.nshift, -g.nshift]) * g.dv / C_KMS)
    sb, wb, _ = g.shift_steps(vb, deposit="linear")
    np.testing.assert_allclose(sb + wb, [g.nshift, -g.nshift], atol=1e-9)
    # 'nearest' unchanged: the original code
    s, c = g.shift_steps(v)
    s_ref = np.rint(-C_KMS * np.log(1.0 - v / C_KMS) / g.dv).astype(np.int64)
    np.testing.assert_array_equal(s, np.clip(s_ref, -g.nshift, g.nshift))
    np.testing.assert_array_equal(c, np.abs(s_ref) > g.nshift)
    np.testing.assert_array_equal(g.shift_steps(v, deposit="nearest")[0], s)
    # nshift = 0: everything at step 0
    g0 = VelocityGrid(dv=1.0, vmax=50.0, vshift=0.0)
    s0, w1, clip = g0.shift_steps(np.array([-2.0, -0.3, 0.0, 0.3, 2.0]), deposit="linear")
    np.testing.assert_array_equal(s0, 0)
    np.testing.assert_array_equal(w1, 0.0)
    np.testing.assert_array_equal(clip, [True, True, False, True, True])
    # bad inputs
    with pytest.raises(ValueError, match="non-finite"):
        g.shift_steps(np.array([1.0, np.nan]), deposit="linear")
    with pytest.raises(ValueError, match="non-finite"):
        g.shift_steps(np.array([C_KMS]), deposit="linear")
    with pytest.raises(ValueError, match="deposit must be one of"):
        g.shift_steps(v, deposit="cubic")
    assert DEPOSITS == ("nearest", "linear") and check_deposit("linear") == "linear"
    with pytest.raises(ValueError):
        check_deposit(None)


# ----------------------------------------------------------------------------------------------
# DiscFlux: single point, convergence, toy star
# ----------------------------------------------------------------------------------------------
def test_discflux_single_point():
    """One point with v = 0.3 km/s: 'linear' gives the profile shifted by x = -c ln(1 - v/c) with linear interpolation
    between grid points (np.interp), 'nearest' the unshifted profile; F0, vmean, vsig do not depend on the deposit."""
    for dv in (1.0, 0.5):
        g = VelocityGrid(dv=dv, vmax=200.0, vshift=10.0)
        nd = _flat_nodes(g)
        f = nd["prof"][0, 0]
        x = -C_KMS * np.log(1.0 - 0.3 / C_KMS)
        rn = _one(disc.DiscFlux(nd, g), 0.3)
        rl = _one(disc.DiscFlux(nd, g, deposit="linear"), 0.3)
        want = np.interp(g.y + x, g.y, f)
        assert np.abs(rl[0][0] - want).max() <= 1e-14
        if dv == 1.0:
            assert np.abs(rn[0][0] - f).max() <= 1e-14                # rint(0.3) = 0: no shift at all
            assert np.abs(rl[0][0] - rn[0][0]).max() > 1e-2           # the shift that 'nearest' loses
        np.testing.assert_array_equal(rl[1], rn[1])                  # F0
        assert rl[2] == rn[2] and rl[3] == rn[3] and rl[4] == rn[4] == 0
    # a negative and an exactly-on-grid shift
    g = VelocityGrid(dv=1.0, vmax=200.0, vshift=10.0)
    nd = _flat_nodes(g)
    D = disc.DiscFlux(nd, g, deposit="linear")
    for v in (-0.3, -2.7, 5.0, 9.99):
        x = -C_KMS * np.log(1.0 - v / C_KMS)
        assert np.abs(_one(D, v)[0][0] - np.interp(g.y + x, g.y, nd["prof"][0, 0])).max() <= 1e-14
    # clipped: all weight at the clipped step, counted
    rl, rn = _one(D, 25.0), _one(disc.DiscFlux(nd, g), 25.0)
    assert np.abs(rl[0] - rn[0]).max() <= 1e-14 and rl[4] == rn[4] == 1
    rl = _one(D, 10.2)                                               # x = 10.2 > 10: clipped in 'linear' only
    assert rl[4] == 1 and _one(disc.DiscFlux(nd, g), 10.2)[4] == 0
    assert "deposit=linear" in repr(D) and "deposit" not in repr(disc.DiscFlux(nd, g))
    with pytest.raises(ValueError, match="deposit must be one of"):
        disc.DiscFlux(nd, g, deposit="exact")


def test_discflux_error_scaling_with_dv():
    """Against the analytically shifted profile (a Gaussian line of sigma 8 km/s, depth 0.5) the worst error over
    single-point shifts 0 <= v <= 1 km/s halves with dv for 'nearest' (O(dv): rounding) and quarters for 'linear'
    (O(dv^2): interpolation)."""
    dvs = (1.0, 0.5, 0.25, 0.125)
    err = {}
    for dep in DEPOSITS:
        e = []
        for dv in dvs:
            g = VelocityGrid(dv=dv, vmax=200.0, vshift=10.0)
            D = disc.DiscFlux(_flat_nodes(g), g, deposit=dep)
            worst = 0.0
            for v in np.linspace(0.0, 1.0, 201):
                x = -C_KMS * np.log(1.0 - v / C_KMS)
                worst = max(worst, np.abs(_one(D, v)[0][0] - (1.0 - _gauss(g.y + x, 0.0, 8.0, 0.5))).max())
            e.append(worst)
        err[dep] = np.array(e)
    rn, rl = err["nearest"][:-1] / err["nearest"][1:], err["linear"][:-1] / err["linear"][1:]
    assert np.all((rn > 1.8) & (rn < 2.2)), rn                      # measured 2.00 2.00 2.08
    assert np.all((rl > 3.8) & (rl < 4.2)), rl                      # measured 3.98 3.99 4.01
    assert np.all(err["linear"] < 0.06 * err["nearest"])            # dv = 1: 9.7e-4 vs 1.9e-2


@pytest.fixture(scope="module")
def toy_star():
    """The toy flux library (toy_library, TOY_GRID) and a toy sphere of 4000 points (velocities: 45 km/s rms per
    component and a plume; scaled in the test)."""
    lib = ts.toy_library(nnode=40)
    nodes = lb.lib_nodes(lib, nmin=20)
    s = ts.toy_sphere(4000, seed=3, dump=1)
    MU, TN, PN = sph.project_los(s["theta"], s["phi"], "thompson2024")
    return types.SimpleNamespace(lib=lib, nodes=nodes, sphere=s, proj=(MU, TN, PN), grid=ts.TOY_GRID)


def _scaled(s, f):
    return dict(teff=s["teff"], ur=f * s["ur"], uth=f * s["uth"], uph=f * s["uph"])


def test_toy_star_m487_like(toy_star):
    """Toy velocities x 0.02 (0.9 km/s rms per component) and x 0.01 (0.45 km/s, M487-like: its v_los rms is 0.45
    km/s): 'linear' equals the continuous brute force (validate.brute_force, continuous=True) to rounding (measured
    2026-10-02: <= 2.2e-15), 'nearest' is off by a sizeable part of the velocity signal max|F - F0| (x 0.02: 3.1e-4 of
    5.0e-3, 6 %; x 0.01: 1.1e-3 of 2.5e-3, 44 %); 'nearest' equals the rounded brute force. With the toy's own
    velocities (M424-like) the rounding is 1.0e-4 of 0.29 (4e-4)."""
    MU, TN, PN = toy_star.proj
    nodes, grid = toy_star.nodes, toy_star.grid
    for f, lin_tol, near_min, near_max in ((0.02, 1e-13, 0.04, None), (0.01, 1e-13, 0.2, None),
                                           (1.0, 1e-13, None, 0.01)):
        smp = _scaled(toy_star.sphere, f)
        V = sph.los_velocity(smp["ur"], smp["uth"], smp["uph"], MU, TN, PN)
        out = {}
        for dep in DEPOSITS:
            D = disc.DiscFlux(nodes, grid, deposit=dep)
            out[dep] = D.integrate_los(MU, V, teff=smp["teff"])
        Fc, F0c = vd.brute_force(nodes, smp, MU, TN, PN, grid, continuous=True)
        Fr, _ = vd.brute_force(nodes, smp, MU, TN, PN, grid)
        sig = np.abs(Fc - F0c).max()
        dl = np.abs(out["linear"]["F"] - Fc).max()
        dn = np.abs(out["nearest"]["F"] - Fc).max()
        assert out["linear"]["n_clip"].sum() == 0
        assert dl <= lin_tol, (f, dl)
        assert np.abs(out["nearest"]["F"] - Fr).max() <= 1e-13
        assert np.abs(out["linear"]["F0"] - F0c).max() <= 1e-13
        assert np.abs(out["linear"]["F0"] - out["nearest"]["F0"]).max() <= 1e-15
        np.testing.assert_array_equal(out["linear"]["vmean_w"], out["nearest"]["vmean_w"])
        np.testing.assert_array_equal(out["linear"]["sigma_w"], out["nearest"]["sigma_w"])
        print("toy x{}: signal {:.3g}, linear {:.3g}, nearest {:.3g} vs continuous".format(f, sig, dl, dn))
        if near_min is not None:                                    # M487-like: nearest error ~ the signal
            assert dn >= near_min * sig, (dn, sig)
            assert dl <= 1e-8 * sig
        if near_max is not None:                                    # M424-like: rounding << signal
            assert dn <= near_max * sig, (dn, sig)


def test_brute_force_check_follows_deposit(toy_star):
    """validate.brute_force_check takes continuous shifts for a 'linear' integrator (continuous=None) and passes;
    forcing the rounded brute force makes it fail by the rounding error; a 'nearest' integrator keeps the rounded
    one."""
    MU, TN, PN = toy_star.proj
    smp = _scaled(toy_star.sphere, 0.01)
    Dl = disc.DiscFlux(toy_star.nodes, toy_star.grid, deposit="linear")
    Dn = disc.DiscFlux(toy_star.nodes, toy_star.grid)
    rl = vd.brute_force_check(toy_star.nodes, smp, MU[:2], TN[:2], PN[:2], integ=Dl)
    rn = vd.brute_force_check(toy_star.nodes, smp, MU[:2], TN[:2], PN[:2], integ=Dn)
    bad = vd.brute_force_check(toy_star.nodes, smp, MU[:2], TN[:2], PN[:2], integ=Dl, continuous=False)
    assert rl.passed() and rn.passed() and not bad.passed()
    assert rl.arrays["brute_integ"].max() <= 1e-13 and bad.arrays["brute_integ"].max() > 1e-4


# ----------------------------------------------------------------------------------------------
# DiscImu
# ----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def imu_toy(tmp_path_factory):
    lib = _toy_imu(SMALL)
    d = tmp_path_factory.mktemp("imu_lin")
    path = str(d / "imu.npz")
    np.savez(path, **lib)
    return types.SimpleNamespace(lib=lib, path=path)


def _imu_points(N, t, seed=1, vsig=0.8):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(-1.0, 1.0, N)
    mu[0] = 1.0                                                     # disc centre
    v = vsig * rng.standard_normal(N)
    teff = rng.uniform(t[0] - 30.0, t[-1] + 30.0, N)
    return mu, v, teff


def test_discimu_linear_vs_continuous_brute(imu_toy):
    """DiscImu 'linear' equals a per-point sum of the interpolated intensities shifted continuously (edge values beyond
    the grid) in every mode (float64 to rounding, float32 to its precision); 'nearest' does not (sub-km/s velocities);
    F0 and the velocity moments do not depend on the deposit; integrate_los equals the per-call loop."""
    lib = imu_toy.lib
    mu, v, teff = _imu_points(600, lib["teff_rep"])
    Fb, F0b = _brute_imu_continuous(lib, SMALL, mu, v, teff)
    for kw, tol in ((dict(), 2e-14), (dict(fft="lazy"), 2e-14), (dict(dtype="float32"), 2e-7),
                    (dict(fft="lazy", dtype="float32"), 2e-6)):
        Dl = disc.DiscImu(imu_toy.path, SMALL, deposit="linear", **kw)
        Dn = disc.DiscImu(imu_toy.path, SMALL, **kw)
        pr = Dl.pairs(teff)
        rl, rn = Dl(mu, v, *pr), Dn(mu, v, *pr)
        assert np.abs(rl[0] - Fb).max() <= tol, (kw, np.abs(rl[0] - Fb).max())
        assert np.abs(rl[1] - F0b).max() <= max(tol, 2e-14)
        assert np.abs(rn[0] - Fb).max() > max(10.0 * tol, 1e-5)
        assert np.abs(rl[1] - rn[1]).max() <= 1e-15
        np.testing.assert_array_equal(rl[2], rn[2])
        np.testing.assert_array_equal(rl[3], rn[3])
        assert rl[4] == rn[4] == 0
        MU2, V2 = np.stack([mu, mu[::-1]]), np.stack([v, -v])
        r2 = Dl.integrate_los(MU2, V2, pairs=pr)
        np.testing.assert_array_equal(r2["F"][0], rl[0])
        np.testing.assert_array_equal(r2["vmean_w"][0], rl[2])
        g = Dl(mu[::-1], -v, *pr)
        np.testing.assert_array_equal(r2["F"][1], g[0])
    # a single point at the disc centre with v = 0.3 km/s: the interpolated shifted intensities
    Dl = disc.DiscImu(imu_toy.path, SMALL, deposit="linear")
    k = 2
    r = Dl(np.array([1.0]), np.array([0.3]), np.array([k]), np.array([k + 1]), np.array([0.0]))
    x = -C_KMS * np.log(1.0 - 0.3 / C_KMS)
    want = np.interp(SMALL.y + x, SMALL.y, lib["Il"][k, 0, 0].astype(np.float64)) / np.interp(
        SMALL.y + x, SMALL.y, lib["Ic"][k, 0, 0].astype(np.float64))
    assert np.abs(r[0][0] - want).max() <= 1e-14


def test_discimu_uniform_intensity_equals_discflux_linear(imu_toy):
    """Intensities independent of mu: DiscImu 'linear' = DiscFlux 'linear' (as for 'nearest')."""
    lib = dict(imu_toy.lib)
    nb = lib["src"].size
    rng = np.random.default_rng(6)
    c = rng.uniform(0.8, 1.6, (nb, 2))
    f = np.empty((nb, 2, SMALL.ny))
    for b in range(nb):
        for j in range(2):
            f[b, j] = 1.0 - (0.3 + 0.2 * rng.random()) * np.exp(-0.5 * ((SMALL.y - rng.uniform(-5, 5)) / 20.0) ** 2)
    lib["Ic"] = np.broadcast_to(c[:, :, None, None], lib["Ic"].shape).astype(np.float32)
    lib["Il"] = (lib["Ic"] * f[:, :, None, :].astype(np.float32)).astype(np.float32)
    nodes = dict(t=lib["teff_rep"], fc=c.astype(np.float32).astype(np.float64),
                 prof=lib["Il"][:, :, 0].astype(np.float64) / lib["Ic"][:, :, 0].astype(np.float64))
    mu, v, teff = _imu_points(20000, nodes["t"], seed=7, vsig=3.0)
    DF = disc.DiscFlux(nodes, SMALL, deposit="linear")
    DI = disc.DiscImu(lib, SMALL, deposit="linear")
    pr = DF.pairs(teff)
    rf, ri = DF(mu, v, *pr), DI(mu, v, *pr)
    assert np.abs(ri[0] - rf[0]).max() <= 2e-14
    assert np.abs(ri[1] - rf[1]).max() <= 2e-14
    np.testing.assert_allclose(ri[2], rf[2], rtol=0, atol=1e-11)


def test_discimu_deposit_record_and_pickle(imu_toy):
    """The deposit is in the fingerprint (only when 'linear': the nearest records stay as they were), survives pickling
    (a file recipe and an array mapping) and is checked."""
    Dn = disc.DiscImu(imu_toy.path, SMALL)
    Dl = disc.DiscImu(imu_toy.path, SMALL, deposit="linear", fft="lazy")
    assert "deposit" not in Dn.fingerprint() and Dl.fingerprint()["deposit"] == "linear"
    assert "deposit=linear" in repr(Dl) and "deposit" not in repr(Dn)
    for D in (Dl, disc.DiscImu(imu_toy.lib, SMALL, deposit="linear")):
        E = pickle.loads(pickle.dumps(D))
        assert E.deposit == "linear" and E.fingerprint() == D.fingerprint()
        mu, v, teff = _imu_points(300, D.t)
        np.testing.assert_array_equal(E(mu, v, *E.pairs(teff))[0], D(mu, v, *D.pairs(teff))[0])
    with pytest.raises(ValueError, match="deposit must be one of"):
        disc.DiscImu(imu_toy.path, SMALL, deposit="Linear")


# ----------------------------------------------------------------------------------------------
# drivers: flux_integrator, imu_integrator, run_fields, run_disc_dumps
# ----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def toy_run_inputs(tmp_path_factory):
    """A toy library file and the samples of 3 dumps with M487-like velocities (x 0.02)."""
    root = tmp_path_factory.mktemp("lin_run")
    lib = ts.toy_library(nnode=40)
    path = lib.save(str(root / "library.npz"))
    sdir = root / "samples"
    sdir.mkdir()
    for d in (1, 2, 3):
        s = ts.toy_sphere(4000, seed=5, dump=d)
        s.update(ur=0.02 * s["ur"], uth=0.02 * s["uth"], uph=0.02 * s["uph"])
        ts._write_sample(str(sdir / "d{:04d}.npz".format(d)), s)
    th, ph = sph.fibonacci_sphere(4000)
    return types.SimpleNamespace(root=root, library=path, samples=str(sdir), theta=th, phi=ph,
                                 lines=ts.toy_lineset(3))


def _run(t, out, name=None, deposit=None, kw=None, **extra):
    return dm.run_disc_dumps([1, 2, 3], t.samples, str(out), name, dm.flux_integrator, (t.library,), t.theta, t.phi,
                             "thompson2024", factory_kwargs=dict(dict(nmin=20, grid=ts.TOY_GRID), **(kw or {})),
                             deposit=deposit, **extra)


def test_run_fields_names(toy_run_inputs, imu_toy):
    """Default run names get '_lin' for deposit='linear'; 'nearest' keeps the legacy names."""
    lib = toy_run_inputs.library
    assert dm.run_fields(dm.flux_integrator(lib, grid=ts.TOY_GRID))["name"] == "flux"
    assert dm.run_fields(dm.flux_integrator(lib, grid=ts.TOY_GRID, deposit="linear"))["name"] == "flux_lin"
    assert dm.run_fields(dm.flux_integrator(lib, grid=ts.TOY_GRID, smooth=150.0, deposit="linear"))["name"] == \
        "flux_sm150_lin"
    for kw, want in ((dict(), "imu"), (dict(deposit="linear"), "imu_lin"), (dict(fft="lazy", deposit="linear"),
                                                                               "imu_lazy_lin")):
        integ = dm.imu_integrator(imu_toy.path, grid=SMALL, **kw)
        assert dm.run_fields(integ)["name"] == want
        assert (want == "imu") == (dm._run_mode(integ) == [])
    assert dm._run_mode(dm.flux_integrator(lib, grid=ts.TOY_GRID, deposit="linear")) == [("deposit", "linear")]
    assert dm._integ_deposit(object()) == "nearest"


def test_run_disc_dumps_linear(toy_run_inputs, tmp_path):
    """run_disc_dumps(deposit='linear'): its own directory flux_lin, the deposit in the run record (absent in a nearest
    run's), the files equal disc_dump with a 'linear' DiscFlux, serial = spawn byte for byte; mixing, conflicting
    options and the adoption of a record-less directory are refused."""
    t = toy_run_inputs
    s = _run(t, tmp_path, deposit="linear", lref=t.lines)
    assert s["name"] == "flux_lin" and s["done"] == [1, 2, 3]
    rec = dm.read_run_record(str(tmp_path / "flux_lin"))
    assert rec["params"]["deposit"] == "linear" and rec["params"]["integrator"]["deposit"] == "linear"
    assert rec["params"]["integrator"]["inputs"]["kwargs"]["deposit"] == "linear"
    sn = _run(t, tmp_path, lref=t.lines)
    rn = dm.read_run_record(str(tmp_path / "flux"))
    assert sn["name"] == "flux" and "deposit" not in rn["params"] and "deposit" not in rn["params"]["integrator"]
    assert "deposit" not in rn["params"]["integrator"]["inputs"]["kwargs"]
    # an explicit deposit='nearest' is the same run as the default (the run record matches; recomputed bit for bit)
    before = {d: open(str(tmp_path / "flux" / "d{:04d}.npz".format(d)), "rb").read() for d in (1, 2, 3)}
    assert sorted(_run(t, tmp_path, deposit="nearest", lref=t.lines, overwrite=True)["done"]) == [1, 2, 3]
    assert sorted(_run(t, tmp_path, kw=dict(deposit="nearest"), lref=t.lines, overwrite=True)["done"]) == [1, 2, 3]
    assert dm.read_run_record(str(tmp_path / "flux"))["params"] == rn["params"]
    assert all(open(str(tmp_path / "flux" / "d{:04d}.npz".format(d)), "rb").read() == b for d, b in before.items())
    # the files: disc_dump with a linear DiscFlux
    integ = dm.flux_integrator(t.library, grid=ts.TOY_GRID, deposit="linear")
    MU, TN, PN = sph.project_los(t.theta, t.phi, "thompson2024")
    for d in (1, 3):
        r = dm.disc_dump(integ, dm.load_sample(t.samples, d), MU, TN, PN, lref=t.lines)
        with np.load(str(tmp_path / "flux_lin" / "d{:04d}.npz".format(d))) as z, \
                np.load(str(tmp_path / "flux" / "d{:04d}.npz".format(d))) as zn:
            np.testing.assert_array_equal(z["F"], r["F"].astype(np.float32))
            assert str(z["name"]) == "flux_lin"
            assert np.abs(z["F"] - zn["F"]).max() > 1e-6                    # sub-km/s shifts: nearest loses them
            np.testing.assert_array_equal(z["F0"], zn["F0"])
    # spawn workers build the same integrator (deposit passed to the factory): same bytes
    s2 = _run(t, tmp_path / "spawn", deposit="linear", lref=t.lines, nproc=2, start_method="spawn")
    assert sorted(s2["done"]) == [1, 2, 3]
    for d in (1, 2, 3):
        f = "d{:04d}.npz".format(d)
        with open(str(tmp_path / "flux_lin" / f), "rb") as a, \
                open(str(tmp_path / "spawn" / "flux_lin" / f), "rb") as b:
            assert a.read() == b.read()
    # refused: nearest into the linear directory, conflicting options, a factory ignoring the deposit
    with pytest.raises(ValueError, match="another configuration"):
        _run(t, tmp_path, name="flux_lin", lref=t.lines, overwrite=True)
    with pytest.raises(ValueError, match="conflicts"):
        _run(t, tmp_path / "x", deposit="linear", kw=dict(deposit="nearest"), lref=t.lines)
    with pytest.raises(ValueError, match="deposit must be one of"):
        _run(t, tmp_path / "x", deposit="lin", lref=t.lines)

    def _nearest_only(library, **kw):
        kw.pop("deposit", None)
        return dm.flux_integrator(library, **kw)

    with pytest.raises(ValueError, match="not the requested"):
        dm.run_disc_dumps([1], t.samples, str(tmp_path / "x"), None, _nearest_only, (t.library,), t.theta, t.phi,
                          "thompson2024", factory_kwargs=dict(grid=ts.TOY_GRID), deposit="linear", lref=t.lines)
    # a record-less (legacy) nearest directory is not adopted by a linear run
    os.remove(str(tmp_path / "flux" / dm.RUN_FILE))
    with pytest.raises(ValueError, match="not the legacy ones"):
        _run(t, tmp_path, name="flux", deposit="linear", lref=t.lines, overwrite=True)
    assert not os.path.exists(str(tmp_path / "flux" / dm.RUN_FILE))
    s3 = _run(t, tmp_path, name="flux", lref=t.lines, overwrite=True)               # nearest adopts it
    assert sorted(s3["done"]) == [1, 2, 3] and "deposit" not in dm.read_run_record(str(tmp_path / "flux"))["params"]


def test_run_disc_dumps_imu_linear(toy_run_inputs, imu_toy, tmp_path):
    """imu_integrator(deposit='linear') through run_disc_dumps: name imu_lazy_lin, recorded, files = disc_dump."""
    t = toy_run_inputs
    smp = {d: dm.load_sample(t.samples, d) for d in (1, 2)}
    sdir = tmp_path / "s"
    sdir.mkdir()
    lo, hi = imu_toy.lib["teff_rep"][0], imu_toy.lib["teff_rep"][-1]
    for d, s in smp.items():                                       # T_eff' into the toy imu library's range
        s = dict(s, teff=lo + (hi - lo) * (s["teff"] - s["teff"].min()) / np.ptp(s["teff"]))
        ts._write_sample(str(sdir / "d{:04d}.npz".format(d)), s)
    ls = LineSet(LINES[:2], LREF[:2])
    s = dm.run_disc_dumps([1, 2], str(sdir), str(tmp_path), None, dm.imu_integrator, (imu_toy.path,), t.theta,
                          t.phi, "thompson2024", factory_kwargs=dict(grid=SMALL, fft="lazy"), deposit="linear", lref=ls)
    assert s["name"] == "imu_lazy_lin" and s["done"] == [1, 2]
    rec = dm.read_run_record(str(tmp_path / "imu_lazy_lin"))
    assert rec["params"]["deposit"] == "linear"
    integ = dm.imu_integrator(imu_toy.path, grid=SMALL, fft="lazy", deposit="linear")
    MU, TN, PN = sph.project_los(t.theta, t.phi, "thompson2024")
    r = dm.disc_dump(integ, dm.load_sample(str(sdir), 2), MU, TN, PN, lref=ls, batch=None)
    with np.load(str(tmp_path / "imu_lazy_lin" / "d0002.npz")) as z:
        np.testing.assert_array_equal(z["F"], r["F"].astype(np.float32))


# ----------------------------------------------------------------------------------------------
# M424 and M487
# ----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def m424_lin():
    """The production flux nodes (lib_nodes nmin=20 of library_dT10.npz) with both deposits, and the 'matmul'
    projections of the M424 points (also M487's: the same sphere)."""
    lib = lb.FluxLibrary.load(m424_path("run", "library_dT10.npz"))
    pts = m424_path("run", "points.npz")
    th, ph = np.asarray(npz_member_memmap(pts, "theta")), np.asarray(npz_member_memmap(pts, "phi"))
    with dm._blas_limit(1):
        nodes = lb.lib_nodes(lib, nmin=20)
    return types.SimpleNamespace(nodes=nodes, theta=th, phi=ph, proj=sph.project_los(th, ph, "thompson2024"),
                                 D=dict((dep, disc.DiscFlux(nodes, deposit=dep)) for dep in DEPOSITS))


@pytest.mark.m424
@pytest.mark.parametrize("dump", [3200, 4000])
def test_m424_nearest_bitwise_and_linear_vs_continuous(m424_lin, dump):
    """deposit='nearest' given explicitly reproduces the production flux/dNNNN.npz bit for bit (F, F0 after the float32
    cast; vmean_w, sigma_w, n_clip exactly). On 20 000-point subsets of every line of sight: 'linear' equals the
    continuous brute force to rounding (measured 2026-10-02: <= 4.3e-15), 'nearest' differs by its rounding (2.1e-5 /
    2.2e-6 / 2.9e-5 for dump 3200, 3.5e-5 / 4.5e-6 / 4.9e-5 for 4000; lines 4026 / 4200 / 4922). ~4 s each."""
    MU, TN, PN = m424_lin.proj
    smp = _samples(SAMPLES_M424, dump)
    V = sph.los_velocity(smp["ur"], smp["uth"], smp["uph"], MU, TN, PN)
    r = m424_lin.D["nearest"].integrate_los(MU, V, teff=smp["teff"])
    with np.load(m424_path("disc", "flux", "d{:04d}.npz".format(dump))) as z:
        np.testing.assert_array_equal(r["F"].astype(np.float32), z["F"])
        np.testing.assert_array_equal(r["F0"].astype(np.float32), z["F0"])
        np.testing.assert_array_equal(r["vmean_w"], z["vmean_w"])
        np.testing.assert_array_equal(r["sigma_w"], z["sigma_w"])
        np.testing.assert_array_equal(r["n_clip"], z["n_clip"])
    M = _subset_mu(MU, 20000, seed=dump)
    Fc, F0c = vd.brute_force(m424_lin.nodes, smp, M, TN, PN, m424_lin.D["linear"].grid, continuous=True, nproc=4)
    rl = m424_lin.D["linear"].integrate_los(M, V, teff=smp["teff"])
    rn = m424_lin.D["nearest"].integrate_los(M, V, teff=smp["teff"])
    dl = np.abs(rl["F"] - Fc).max(axis=(0, 2))
    dn = np.abs(rn["F"] - Fc).max(axis=(0, 2))
    print("M424 dump {}: max|dF| vs continuous: linear {}, nearest {}".format(dump, dl, dn))
    assert rl["n_clip"].sum() == 0
    assert dl.max() <= 1e-13
    assert np.all(dn > 5e-7) and dn.max() < 1e-4
    assert np.abs(rl["F0"] - F0c).max() <= 1e-13


@pytest.mark.m424
def test_m487_linear_vs_continuous(m424_lin):
    """M487 dump 3200 (IGW only; samples of moms.sample_moms_sphere, slab backend, 3550 Mm, the M424 sphere) with the
    M424 flux library: 'linear' equals the continuous brute force on a 20 000-point subset of every line of sight to
    rounding; 'nearest' is off by about half the signal (F - F0, the profile without velocities) because 73 % of the
    visible points have |v| < 0.5 km/s and get shift 0 (v_los rms 0.45 km/s). Measured 2026-10-02 (lines 4026 / 4200 /
    4922): full sphere, max|F - F0| 1.3e-3 / 1.7e-4 / 2.5e-3 (rms over |y| <= 600 km/s 1.2e-4 / 2.1e-5 / 1.5e-4), max
    |F_linear - F_nearest| 6.7e-4 / 7.8e-5 / 1.0e-3; subset: linear vs continuous <= 4.9e-15, nearest 6.8e-4 / 8.0e-5
    / 1.0e-3."""
    MU, TN, PN = m424_lin.proj
    smp = _samples(SAMPLES_M487, 3200)
    V = sph.los_velocity(smp["ur"], smp["uth"], smp["uph"], MU, TN, PN)
    rl = m424_lin.D["linear"].integrate_los(MU, V, teff=smp["teff"])
    rn = m424_lin.D["nearest"].integrate_los(MU, V, teff=smp["teff"])
    sig_l = np.abs(rl["F"] - rl["F0"]).max(axis=(0, 2))
    d_ln = np.abs(rl["F"] - rn["F"]).max(axis=(0, 2))
    print("M487 dump 3200 full sphere: signal max|F - F0| {}, linear - nearest {}".format(sig_l, d_ln))
    assert np.all(d_ln > 0.3 * sig_l)
    M = _subset_mu(MU, 20000, seed=11)
    Fc, F0c = vd.brute_force(m424_lin.nodes, smp, M, TN, PN, m424_lin.D["linear"].grid, continuous=True, nproc=4)
    sl = m424_lin.D["linear"].integrate_los(M, V, teff=smp["teff"])
    sn = m424_lin.D["nearest"].integrate_los(M, V, teff=smp["teff"])
    dl = np.abs(sl["F"] - Fc).max(axis=(0, 2))
    dn = np.abs(sn["F"] - Fc).max(axis=(0, 2))
    sig = np.abs(Fc - F0c).max(axis=(0, 2))
    print("M487 subset: linear {}, nearest {}, signal {}".format(dl, dn, sig))
    assert dl.max() <= 1e-13
    assert np.all(dn > 0.3 * sig)


# ----------------------------------------------------------------------------------------------
# reviewer fixes (PP 2026-10-02): nearest twins in the validation, the intensity brute force with continuous shifts,
# the 'cubic' sensitivity, the run drivers' deposit checks
# ----------------------------------------------------------------------------------------------
def test_with_deposit_twins(imu_toy):
    """with_deposit: self for its own deposit, else a copy that shares the arrays and gives bit for bit the results of
    an integrator built with that deposit (DiscFlux, DiscImu in both FFT modes); a DiscImu twin pickles with its
    deposit; the original is unchanged."""
    g = VelocityGrid(dv=1.0, vmax=500.0, vshift=40.0)
    nd = _toy_nodes(g, nl=2)
    Dl = disc.DiscFlux(nd, g, deposit="linear")
    assert Dl.with_deposit("linear") is Dl
    Tn = Dl.with_deposit("nearest")
    assert Tn.deposit == "nearest" and Dl.deposit == "linear" and Tn.P is Dl.P and Tn.Dhat is Dl.Dhat
    rng = np.random.default_rng(2)
    mu, v, teff = rng.uniform(-1, 1, 3000), 0.8 * rng.standard_normal(3000), rng.uniform(nd["t"][0], nd["t"][-1], 3000)
    pr = Dl.pairs(teff)
    for a, b in ((Tn, disc.DiscFlux(nd, g)), (Tn.with_deposit("linear"), Dl)):
        ra, rb = a(mu, v, *pr), b(mu, v, *pr)
        for x, y in zip(ra, rb):
            np.testing.assert_array_equal(x, y)
    with pytest.raises(ValueError, match="deposit must be one of"):
        Dl.with_deposit("cubic")
    mu, v, teff = _imu_points(400, imu_toy.lib["teff_rep"])
    g = SMALL
    for kw in (dict(), dict(fft="lazy")):
        Il = disc.DiscImu(imu_toy.path, g, deposit="linear", **kw)
        In = Il.with_deposit("nearest")
        assert In.deposit == "nearest" and Il.deposit == "linear" and Il._init_args["deposit"] == "linear"
        assert "deposit" not in In.fingerprint() and Il.fingerprint()["deposit"] == "linear"
        pr = Il.pairs(teff)
        ref = disc.DiscImu(imu_toy.path, g, **kw)(mu, v, *pr)
        for x, y in zip(In(mu, v, *pr), ref):
            np.testing.assert_array_equal(x, y)
        E = pickle.loads(pickle.dumps(In))
        assert E.deposit == "nearest"
        np.testing.assert_array_equal(E(mu, v, *pr)[0], ref[0])
        assert np.abs(Il(mu, v, *pr)[0] - ref[0]).max() > 1e-6          # the original keeps its deposit


@pytest.fixture(scope="module")
def gen_toy():
    """The general synthetic run of test_validate.py (2 lines, 3 lines of sight, GEN_GRID, velocities 25 km/s per
    component) and its velocities x 0.02 (0.5 km/s, M487-like)."""
    import test_validate as tv
    g = tv._gen_toy()
    small = {d: dict(s, ur=0.02 * s["ur"], uth=0.02 * s["uth"], uph=0.02 * s["uph"]) for d, s in g["samples"].items()}
    return types.SimpleNamespace(tv=tv, g=g, small=small)


@pytest.mark.parametrize("scale", [1.0, 0.02])
def test_validation_of_a_linear_integrator(gen_toy, scale):
    """V1, V2 (a factory of 'linear' integrators), V4 and V5 of a 'linear' DiscFlux run on its 'nearest' twin: the same
    values as for the 'nearest' integrator, bit for bit, and they pass (before the fix V1 / V4 / V5 failed at 3.7e-5 /
    5.8e-5 / 5.7e-5 with the toy velocities, V4 / V5 at 7.3e-5 / 6.8e-5 with x 0.02); V3 compares the integrator with
    itself; V6 and the brute force (continuous shifts) check the deposit itself and pass at rounding; run_validation
    runs all of them. Comparing the 'linear' integrator itself with the rounded exact sums (nearest_twin=False)
    fails, as it should."""
    tv, g = gen_toy.tv, gen_toy.g
    samples = g["samples"] if scale == 1.0 else gen_toy.small     # V1: dump 0 of the exact sums, unscaled
    MU, TN, PN = g["proj"]
    Dn = g["integ"]
    Dl = disc.DiscFlux(g["nodes"], tv.GEN_GRID, deposit="linear")
    kw = dict(tolerances=tv.GEN_TOL)
    vals = {}
    for dep, integ in (("nearest", Dn), ("linear", Dl)):
        r1 = vd.v1_exact(integ, g["exact"], g["samples"][0], MU, TN, PN, lref=tv.GEN_LS, **kw)
        r4 = vd.v4_nearest(integ, g["lib"], samples, [1, 2], MU, TN, PN, **kw)
        r5 = vd.v5_node_merging(integ, g["lib"], samples, 1, MU, TN, PN, **kw)
        r6 = vd.v6_rounding(integ, g["nodes"], samples, [1, 2], MU, TN, PN, nsub=1500, **kw)
        rb = vd.brute_force_check(g["nodes"], samples, MU, TN, PN, tv.GEN_GRID, dumps=[1], integ=integ, **kw)
        vals[dep] = (r1, r4, r5, r6, rb)
        for r in (r1, r4, r5, r6, rb):
            assert r.passed(), (dep, scale, r.table())
    for k in range(3):                                              # V1, V4, V5: the same numbers
        np.testing.assert_array_equal(vals["linear"][k].arrays[list(vals["linear"][k].arrays)[0]],
                                      vals["nearest"][k].arrays[list(vals["nearest"][k].arrays)[0]])
    r1, r4, r5, r6, rb = vals["linear"]
    assert r1["V1_flux"].details["compared"] == "nearest twin" and r1.meta["nearest_twin"] is True
    assert r4["V4"].details["compared"] == "nearest twin" and r5["V5"].details["deposit"] == "linear"
    assert "compared" not in vals["nearest"][0]["V1_flux"].details
    assert r6["V6"].details["measures"].startswith("sub-grid (linear) deposit") and r6["V6"].value < 1e-11
    assert "measures" not in vals["nearest"][3]["V6"].details
    assert rb.meta["continuous"] is True and rb["brute_vs_integrator"].value < 1e-13
    bad = vd.v1_exact(Dl, g["exact"], g["samples"][0], MU, TN, PN, lref=tv.GEN_LS, nearest_twin=False, **kw)
    assert not bad.passed() and bad["V1_flux"].value > 1e-6
    # V2 through a factory of 'linear' integrators: twins; the same values as the default (nearest) factory
    hk = dict(profiles=g["prof"], sample=samples[0], theta=None, phi=None, los=tv.GEN_LOS, grid=tv.GEN_GRID,
              lref=tv.GEN_LS, block=1000, stride=2, tolerances=tv.GEN_TOL)
    r2n = vd.v2_holdout(**hk)
    r2l = vd.v2_holdout(factory=lambda L: disc.DiscFlux(lb.lib_nodes(L, nmin=20), tv.GEN_GRID, deposit="linear"),
                        **hk)
    assert r2l.passed()
    for k in r2n.arrays:
        if k.startswith(("holdout_", "insample_")):
            np.testing.assert_array_equal(r2l.arrays[k], r2n.arrays[k])
    # the driver
    r = vd.run_validation(Dl, MU, TN, PN, sample=g["samples"][0], exact=g["exact"], library=g["lib"],
                          nodes=g["nodes"],
                          samples=samples, dumps=[1, 2], v3_dumps=[3], lref=tv.GEN_LS, nsub=1000, margin=50.0,
                          brute_dumps=[1], tolerances=tv.GEN_TOL)
    assert r.names() == ["V1_flux", "V1_flux_dEW", "V3_extrap", "V3_leaveout", "V4", "V5", "V6",
                         "brute_vs_integrator"]
    assert r.passed(), r.table()


def test_brute_force_imu_continuous(imu_toy):
    """brute_force_imu(continuous=True) equals the per-point reference _brute_imu_continuous and a 'linear' DiscImu to
    rounding; brute_force_imu_check follows the integrator's deposit (a 'linear' DiscImu passes, at sub-km/s
    velocities; before the fix it failed at 8e-5 against the rounded brute force, which continuous=False still shows);
    a 'nearest' DiscImu keeps the rounded one; clipped shifts are counted and treated as the integrator does."""
    lib = imu_toy.lib
    N = 3000
    rng = np.random.default_rng(4)
    th, ph = sph.fibonacci_sphere(N)
    MU, TN, PN = sph.project_los(th, ph, "thompson2024")
    smp = dict(teff=rng.uniform(lib["teff_rep"][0] - 20.0, lib["teff_rep"][-1] + 20.0, N),
               **{k: 0.8 * rng.standard_normal(N) for k in ("ur", "uth", "uph")})
    Dl = disc.DiscImu(imu_toy.path, SMALL, deposit="linear")
    Dn = disc.DiscImu(imu_toy.path, SMALL)
    b = vd.brute_force_imu(imu_toy.path, smp, MU[:2], TN[:2], PN[:2], SMALL, continuous=True)
    for k in range(2):
        V = sph.los_velocity(smp["ur"], smp["uth"], smp["uph"], MU[k], TN[k], PN[k])
        Fr, F0r = _brute_imu_continuous(lib, SMALL, MU[k], V, smp["teff"])
        assert np.abs(b["F"][k] - Fr).max() <= 1e-14 and np.abs(b["F0"][k] - F0r).max() <= 1e-14
    rl = vd.brute_force_imu_check(Dl, smp, MU, TN, PN, nsub=None)
    rn = vd.brute_force_imu_check(Dn, smp, MU, TN, PN, nsub=None)
    bad = vd.brute_force_imu_check(Dl, smp, MU, TN, PN, nsub=None, continuous=False)
    assert rl.passed() and rn.passed() and not bad.passed(), (rl.table(), bad.table())
    assert rl["brute_imu_vs_integrator"].value <= 1e-13 and rl.meta["continuous"] is True
    assert "continuous" not in rn.meta and bad["brute_imu_vs_integrator"].value > 1e-5
    # clipped shifts (|x| > vshift = 40 km/s for some points): counted and clipped as DiscImu 'linear'
    fast = dict(smp, ur=30.0 * smp["ur"], uth=30.0 * smp["uth"], uph=30.0 * smp["uph"])
    rf = vd.brute_force_imu_check(Dl, fast, MU[:3], TN[:3], PN[:3], nsub=None)
    assert rf.passed() and rf.data["brute_imu"][-1]["n_clip"].sum() > 0, rf.table()
    with pytest.raises(ValueError, match="continuous must be"):
        vd.brute_force_imu_check(Dl, smp, MU, TN, PN, continuous="cubic")


def test_brute_force_cubic_and_clipping():
    """brute_force(continuous='cubic'): Catmull-Rom; at whole-step shifts it equals the 'linear' brute force (the
    profile values themselves), for a smooth line between grid points it is far closer to the analytically shifted
    profile than linear interpolation (the interpolation-model sensitivity it is for). Continuous shifts are clipped
    to +-vshift as the integrators clip them, so 'linear' DiscFlux equals the continuous brute force also with
    clipped points. brute_force_check refuses 'cubic'."""
    g = VelocityGrid(dv=1.0, vmax=200.0, vshift=10.0)
    nd = _flat_nodes(g)
    one = dict(teff=np.array([38050.0, 38050.0]), ur=np.zeros(2), uth=np.zeros(2), uph=np.zeros(2))
    MU, TN, PN = np.array([[1.0, 0.0]]), np.zeros((1, 2)), np.zeros((1, 2))
    err = {}
    for v in (0.3, -0.45, 2.0, 25.0):
        s = dict(one, ur=np.array([v, 0.0]))                          # mu = 1: v_los = u_r
        x = np.clip(-C_KMS * np.log(1.0 - v / C_KMS), -g.vshift, g.vshift)
        exact = 1.0 - _gauss(g.y + x, 0.0, 8.0, 0.5)
        Fl, _ = vd.brute_force(nd, s, MU, TN, PN, g, continuous=True)
        Fc, _ = vd.brute_force(nd, s, MU, TN, PN, g, continuous="cubic")
        D = disc.DiscFlux(nd, g, deposit="linear")
        Fi = D(MU[0], np.array([v, 0.0]), *D.pairs(s["teff"]))[0]
        assert np.abs(Fi - Fl[0]).max() <= 1e-14, v                  # also clipped (v = 25 km/s > vshift)
        err[v] = (np.abs(Fl[0, 0] - exact).max(), np.abs(Fc[0, 0] - exact).max())
        if abs(x - round(x)) < 1e-6:
            assert np.abs(Fc - Fl).max() <= 1e-14
    for v in (0.3, -0.45):
        assert err[v][1] < 0.1 * err[v][0], err
    with pytest.raises(ValueError, match="continuous must be"):
        vd.brute_force(nd, one, MU, TN, PN, g, continuous="spline")
    with pytest.raises(ValueError, match="continuous must be None, True or False"):
        vd.brute_force_check(nd, one, MU, TN, PN, g, integ=disc.DiscFlux(nd, g), continuous="cubic")


def test_run_disc_dumps_deposit_checks(toy_run_inputs, tmp_path):
    """Reviewer fixes of the driver: an explicit name whose dumps are all present is checked for the requested deposit
    (before: returned 'all present' for a 'nearest' directory); a 'linear' run writes 'deposit' into every per-dump
    file (the 'nearest' files have none), so a 'nearest' run refuses a record-less 'linear' directory and the reverse;
    a deposit among the factory's positional arguments is seen by the conflict check; deposit is the last
    parameter (after log)."""
    import inspect
    t = toy_run_inputs
    assert list(inspect.signature(dm.run_disc_dumps).parameters)[-2:] == ["log", "deposit"]
    _run(t, tmp_path, name="m487", lref=t.lines)                          # a complete 'nearest' directory
    with pytest.raises(ValueError, match="computed with deposit 'nearest', not the requested 'linear'"):
        _run(t, tmp_path, name="m487", deposit="linear", lref=t.lines)
    with pytest.raises(ValueError, match="not the requested 'linear'"):
        _run(t, tmp_path, name="m487", kw=dict(deposit="linear"), lref=t.lines)
    assert _run(t, tmp_path, name="m487", lref=t.lines)["skipped"] == [1, 2, 3]        # nearest: fine
    s = _run(t, tmp_path, name="m487c", deposit="linear", lref=t.lines)
    assert s["done"] == [1, 2, 3]
    with np.load(str(tmp_path / "m487c" / "d0001.npz")) as z, np.load(str(tmp_path / "m487" / "d0001.npz")) as zn:
        assert str(z["deposit"]) == "linear" and "deposit" not in zn.files
        assert z.files[:len(dm.DUMP_KEYS)] == list(dm.DUMP_KEYS) and zn.files == list(dm.DUMP_KEYS)
    assert dm._stored_constants(str(tmp_path / "m487c" / "d0001.npz"))["deposit"] == "linear"
    assert _run(t, tmp_path, name="m487c", deposit="linear", lref=t.lines)["skipped"] == [1, 2, 3]
    with pytest.raises(ValueError, match="not the requested 'nearest'"):
        _run(t, tmp_path, name="m487c", lref=t.lines)                    # the factory's default: 'nearest'
    # the record lost, d0002 gone: a 'nearest' run refuses to adopt the 'linear' files (before: wrote d0002)
    os.remove(str(tmp_path / "m487c" / dm.RUN_FILE))
    os.remove(str(tmp_path / "m487c" / "d0002.npz"))
    with pytest.raises(ValueError, match="another configuration.*deposit"):
        _run(t, tmp_path, name="m487c", lref=t.lines)
    assert not os.path.exists(str(tmp_path / "m487c" / "d0002.npz"))
    with pytest.raises(ValueError, match="not the legacy ones"):                    # nor does 'linear' (no record)
        _run(t, tmp_path, name="m487c", deposit="linear", lref=t.lines)
    # positional deposit (8th argument of flux_integrator) vs deposit=: conflict seen; equal: no duplicate keyword
    args = (t.library, 20, 0.0, None, "corr", ts.TOY_GRID, disc.PAD_TOL, "nearest")
    with pytest.raises(ValueError, match="conflicts"):
        dm.run_disc_dumps([1], t.samples, str(tmp_path / "pos"), None, dm.flux_integrator, args, t.theta, t.phi,
                          "thompson2024", deposit="linear", lref=t.lines)
    args = args[:-1] + ("linear",)
    s = dm.run_disc_dumps([1], t.samples, str(tmp_path / "pos"), None, dm.flux_integrator, args, t.theta, t.phi,
                          "thompson2024", deposit="linear", lref=t.lines)
    assert s["name"] == "flux_lin" and s["done"] == [1]
    with open(str(tmp_path / "pos" / "flux_lin" / "d0001.npz"), "rb") as a, \
            open(str(tmp_path / "m487c" / "d0001.npz"), "rb") as b:
        za, zb = np.load(a), np.load(b)
        np.testing.assert_array_equal(za["F"], zb["F"])
    with pytest.raises(ValueError, match="'deposit'"):
        dm._check_extra(dict(deposit="x"))                                     # reserved for the deposit member


@pytest.mark.m424
@pytest.mark.slow
def test_m424_imu_nearest_bitwise_and_linear_brute():
    """The intensity method on M424 (~2 min, ~10 GB): DiscImu with the default deposit reproduces the production
    imu/d3200.npz bit for bit (F, F0 after the float32 cast; vmean_w, sigma_w, n_clip exactly); its 'linear' twin
    (with_deposit) equals the intensity brute force with continuous shifts on 20 000-point subsets of M424 dump 3200
    and M487 dump 3200 (brute_force_imu_check follows the deposit) to rounding."""
    mp = m424_path
    path = os.environ.get("PPMPY_SYNSPEC_M424_IMU", "/scratch/ppathak/fastwind_imu/imu_library_dT10.npz")
    if not os.path.exists(path):
        pytest.skip("M424 intensity library not available: {}".format(path))
    pts = mp("run", "points.npz")
    th, ph = np.asarray(npz_member_memmap(pts, "theta")), np.asarray(npz_member_memmap(pts, "phi"))
    MU, TN, PN = sph.project_los(th, ph, "thompson2024")
    with dm._blas_limit(1):
        Dn = disc.DiscImu(path)
    smp = _samples(SAMPLES_M424, 3200)
    V = sph.los_velocity(smp["ur"], smp["uth"], smp["uph"], MU, TN, PN)
    r = Dn.integrate_los(MU, V, teff=smp["teff"])
    with np.load(mp("disc", "imu", "d3200.npz")) as z:
        np.testing.assert_array_equal(r["F"].astype(np.float32), z["F"])
        np.testing.assert_array_equal(r["F0"].astype(np.float32), z["F0"])
        np.testing.assert_array_equal(r["vmean_w"], z["vmean_w"])
        np.testing.assert_array_equal(r["sigma_w"], z["sigma_w"])
        np.testing.assert_array_equal(r["n_clip"], z["n_clip"])
    Dl = Dn.with_deposit("linear")
    for root, dump in ((SAMPLES_M424, 3200), (SAMPLES_M487, 3200)):
        s = _samples(root, dump)
        rep = vd.brute_force_imu_check(Dl, s, MU[:2], TN[:2], PN[:2], library=path, nsub=20000, seed=dump, nproc=1)
        print("{} dump {}: imu linear vs continuous brute force {}".format(root, dump, rep.table()))
        assert rep.meta["continuous"] is True and rep.passed()
        assert rep["brute_imu_vs_integrator"].value <= 1e-13
