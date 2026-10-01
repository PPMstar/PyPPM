"""Tests of ppmpy.synspec.disc: synthetic / analytic checks (no data), bitwise comparisons with the frozen legacy
fw_disc.py (DiscFlux, integrate_exact, integrate_lib) and fw_disc_los.py (the streamed exact sums, run from their own
source lines on a synthetic star), the parallel paths (fork, spawn, nproc, rows), and regressions (m424) against the
M424 production products (flux/dNNNN.npz of fw_disc_dumps.py; disc_los8.npz and library_dT10.npz of fw_disc_los.py).

PPMPY_SYNSPEC_M424_SAMPLES (default /scratch/ppathak/fastwind_sphere/samples_r4050_N1236544) locates the per-dump sphere
samples. The M424 comparisons are bitwise under numpy 1.26 on the AVX512 Trillium nodes (library and sphere module
notes)."""
import io
import json
import os
import resource
import subprocess
import sys
import textwrap
import time
import types

import numpy as np
import pytest
from scipy import sparse

import conftest
from conftest import ROOT, m424_path
from ppmpy.synspec import disc
from ppmpy.synspec import library as lb
from ppmpy.synspec import parallel as par
from ppmpy.synspec import sphere as sph
from ppmpy.synspec.conventions import C_KMS, los_thompson2024
from ppmpy.synspec.diagnostics import diagnostics_array, line_diagnostics
from ppmpy.synspec.fwresults import ProfileStore
from ppmpy.synspec.io import npz_member_memmap, read_meta, save_npz
from ppmpy.synspec.spectral import LineSet, VelocityGrid, lam_of_y

LREF = np.array([4026.22, 4199.90, 4921.93])
LINES = ["HEI4026", "HEII4200", "HEI4922"]
LINESET = LineSet(LINES, LREF)
LEGACY = os.environ.get("PPMPY_SYNSPEC_M424_PROJECT",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy"))
FROZEN_DIAG_KW = dict(vwin=400.0, ew_jacobian=True, inclusive=True)     # fw_disc.diagnostics of the frozen copy
RESULT_KEYS = ("F", "F0", "vmean_w", "sigma_w", "diag_F", "diag_F0")

# PP 2026-10-01: no transparent-huge-page fixture here (unlike test_library.py). Review measurement on this node: with
# THP off a synthetic star was slower for every rows value (rows 1000: 10.5 s vs 8.8 s); the system time of the M424
# stream came from the page faults of > 32 MB interpolation temporaries, fixed by rows='auto' (16 MiB).


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


def _legacy_exec(name, first, last, ns):
    """Execute the lines of a frozen legacy script from the first line starting with `first` to the next line starting
    with `last` (stripped; dedented) in namespace ns (as test_library.py / test_sphere.py); returns ns."""
    path = os.path.join(LEGACY, name)
    if not os.path.exists(path):
        pytest.skip("legacy {} not available in {}".format(name, LEGACY))
    lines = open(path).read().splitlines()
    i0 = next(i for i, x in enumerate(lines) if x.strip().startswith(first))
    i1 = next(i for i in range(i0, len(lines)) if lines[i].strip().startswith(last))
    exec(textwrap.dedent("\n".join(lines[i0:i1 + 1])), ns)
    return ns


class _FakePool:
    """multiprocessing.Pool stand-in for the legacy scripts: `with Pool(n) as pool: pool.map(...)` -> builtin map."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return types.SimpleNamespace(map=map)

    def __exit__(self, *exc):
        return False


def _samples_path(dump):
    """Path of the per-dump sphere samples of sphere_sample.py (teff, ur, uth, uph float32, t_s); skips when absent.
    Uses conftest.M424['samples'] once conftest defines it (PPMPY_SYNSPEC_M424_SAMPLES)."""
    name = "d{:04d}.npz".format(dump)
    if "samples" in conftest.M424:
        return m424_path("samples", name)
    root = os.environ.get("PPMPY_SYNSPEC_M424_SAMPLES", "/scratch/ppathak/fastwind_sphere/samples_r4050_N1236544")
    p = os.path.join(root, name)
    if not os.path.exists(p):
        pytest.skip("M424 product not available: {}".format(p))
    return p


def _samples(dump):
    """The per-dump sphere samples (dict of arrays)."""
    with np.load(_samples_path(dump)) as z:
        return {k: z[k] for k in z.files}


def _gauss_depth(y, centre, sigma, amp):
    return amp * np.exp(-0.5 * ((y - centre) / sigma) ** 2)


def _toy_nodes(nn, grid, nl=3, seed=0, sigmas=(60.0, 140.0, 40.0)):
    """Synthetic library nodes on `grid`: Gaussian absorption lines whose depth, width and centre change with T_eff';
    depth exactly 0 far from the centre (exp underflows), as the padding check requires."""
    rng = np.random.default_rng(seed)
    t = 36000.0 + np.cumsum(rng.uniform(5.0, 30.0, nn))
    x = (t - t.mean()) / t.std()
    prof = np.empty((nn, nl, grid.ny))
    fc = np.empty((nn, nl))
    for j in range(nl):
        sig = sigmas[j % len(sigmas)] * (1.0 + 0.05 * x)
        amp = 0.4 + 0.05 * x * (j + 1)
        cen = 3.0 * x + 2.0 * j
        prof[:, j] = 1.0 - _gauss_depth(grid.y[None, :], cen[:, None], sig[:, None], amp[:, None])
        fc[:, j] = (t / 38000.0) ** 4 * (1.0 + 0.03 * j)
    return dict(t=t, count=np.full(nn, 20.0), prof=prof, fc=fc)


def _toy_points(N, tn, seed=1, vsig=60.0, nfast=5):
    """mu, v, teff of N random points; a few with |v| > 400 km/s (clipped by DiscFlux) and T_eff' beyond the nodes."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(-1.0, 1.0, N)
    v = vsig * rng.standard_normal(N)
    v[:nfast] = np.array([450.0, -520.0, 401.0, -400.6, 900.0])[:nfast]
    mu[:nfast] = 0.5                                               # visible
    teff = rng.uniform(tn[0] - 50.0, tn[-1] + 50.0, N)
    return mu, v, teff


def _toy_star(N, grid=None, nl=3, seed=0, nrow=161, span=3000.0, vsig=40.0):
    """A synthetic per-point run on the equal-area grid: profiles (lam, fnorm, fcont float32 (N, nl, nrow), like
    profiles.npz), T_eff', coordinates, and float32 velocity samples (ur, uth, uph)."""
    rng = np.random.default_rng(seed)
    theta, phi = sph.fibonacci_sphere(N)
    teff = 38230.0 + 300.0 * rng.standard_normal(N)
    teff[:3] = [37000.0, 39500.0, 38230.0]                         # sparse tails: empty library bins in between
    x = (teff - 38230.0) / 300.0
    yn = np.linspace(-span, span, nrow)
    jit = rng.uniform(-0.4, 0.4, N)
    lref = LREF[:nl]
    lam = (lref[None, :, None] * np.exp((yn[None, None, :] + jit[:, None, None]) / C_KMS)).astype(np.float32)
    fnorm = np.empty((N, nl, nrow), np.float32)
    fcont = np.empty((N, nl, nrow), np.float32)
    sig = (60.0, 140.0, 40.0)
    for j in range(nl):
        depth = _gauss_depth(yn[None, :], 2.0 * x[:, None] + j, sig[j] * (1.0 + 0.03 * x[:, None]),
                             0.4 + 0.05 * (j + 1) * x[:, None])
        fnorm[:, j] = 1.0 - depth
        fcont[:, j] = ((teff / 38230.0) ** 4 * (1.0 + 0.05 * j))[:, None] * (1.0 + 1e-4 * yn / span)[None, :]
    vel = {k: (vsig * rng.standard_normal(N)).astype(np.float32) for k in ("ur", "uth", "uph")}
    prof = dict(lam=lam, fnorm=fnorm, fcont=fcont, teff=teff, theta=theta, phi=phi,
                lines=np.array(LINES[:nl]), idx=np.arange(N, dtype=np.int32))
    return prof, vel


@pytest.fixture(scope="module")
def star():
    return _toy_star(3000)


def _assert_results_equal(r1, r2, keys=RESULT_KEYS):
    for k in keys:
        np.testing.assert_array_equal(r1[k], r2[k], err_msg=k)
    assert r1["checks"] == r2["checks"]
    for k in lb.LIBRARY_KEYS:
        np.testing.assert_array_equal(r1["library"][k], r2["library"][k], err_msg="library " + k)


@pytest.fixture
def spill_log(monkeypatch):
    """Records disc._spill calls in the parent: (name, whether a file was written)."""
    log = []
    orig = disc._spill

    def rec(a, d, name):
        path = os.path.join(d, name + ".npy")
        before = os.path.exists(path)
        out = orig(a, d, name)
        log.append((name, (not before) and os.path.exists(path)))
        return out

    monkeypatch.setattr(disc, "_spill", rec)
    return log


# ----------------------------------------------------------------------------------------------
# module
# ----------------------------------------------------------------------------------------------
def test_no_heavy_imports():
    code = ("import sys; sys.path.insert(0, {!r}); import ppmpy.synspec.disc; "
            "print(sorted(m for m in sys.modules if m.startswith(('matplotlib', 'ppmpy.ppm'))))").format(ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


# ----------------------------------------------------------------------------------------------
# DiscFlux
# ----------------------------------------------------------------------------------------------
def test_discflux_matches_legacy():
    """Bit for bit the frozen fw_disc.DiscFlux (M424 grid; clipped shifts, T_eff' beyond the nodes, novel on/off)."""
    fd = _legacy_fw_disc()
    grid = VelocityGrid()
    nodes = _toy_nodes(37, grid)
    mu, v, teff = _toy_points(40000, nodes["t"])
    ref = fd.DiscFlux(nodes)
    got = disc.DiscFlux(nodes, grid)
    assert got.L == ref.L and got.nv == ref.nv and got.vs == ref.vs
    np.testing.assert_array_equal(got.P, ref.P)
    np.testing.assert_array_equal(got.Dhat, ref.Dhat)
    for a, b in zip(got.pairs(teff), ref.pairs(teff)):
        np.testing.assert_array_equal(a, b)
    k0, k1, a = got.pairs(teff)
    for novel in (True, False):
        r, g = ref(mu, v, k0, k1, a, novel=novel), got(mu, v, k0, k1, a, novel=novel)
        for x, y in zip(g[:4], r[:4]):
            if y is None:
                assert x is None
            else:
                np.testing.assert_array_equal(x, y)
        assert g[4] == r[4] == 4                                   # 450, -520, 401, 900 clipped; -400.6 -> -400
    # integrate_los = the per-LOS loop of fw_disc_dumps.py
    MU = np.stack([mu, -mu, np.roll(mu, 7)])
    V = np.stack([v, 0.5 * v, -v])
    out = got.integrate_los(MU, V, teff=teff)
    for k in range(3):
        F, F0, vm, sd, ncl = ref(MU[k], V[k], k0, k1, a)
        np.testing.assert_array_equal(out["F"][k], F)
        np.testing.assert_array_equal(out["F0"][k], F0)
        np.testing.assert_array_equal(out["vmean_w"][k], vm)
        np.testing.assert_array_equal(out["sigma_w"][k], sd)
        assert out["n_clip"][k] == ncl


def test_discflux_clip_count():
    grid = VelocityGrid()
    got = disc.DiscFlux(_toy_nodes(5, grid), grid)
    mu = np.ones(6)
    v = np.array([0.0, 399.0, 401.0, -401.0, 1000.0, -2.0])
    s = np.rint(-C_KMS * np.log(1.0 - v / C_KMS))
    k0, k1, a = got.pairs(np.full(6, got.t[2]))
    assert got(mu, v, k0, k1, a)[4] == int((np.abs(s) > 400).sum()) == 3
    with np.errstate(invalid="ignore"):                             # nothing visible: NaN profiles
        assert got(-mu, v, k0, k1, a)[4] == 0                      # hidden points are not counted


def test_discflux_uniform_star_no_velocity():
    """Every node the same profile p: F = F0 = p for any visible set, T_eff' and weights."""
    grid = VelocityGrid(dv=1.0, vmax=900.0, vshift=200.0)
    nodes = _toy_nodes(6, grid, nl=2, sigmas=(30.0, 45.0))
    nodes["prof"][:] = nodes["prof"][2]
    rng = np.random.default_rng(3)
    mu, teff = rng.uniform(-1, 1, 5000), rng.uniform(nodes["t"][0], nodes["t"][-1], 5000)
    D = disc.DiscFlux(nodes, grid)
    F, F0, vm, sd, ncl = D(mu, np.zeros(5000), *D.pairs(teff))
    np.testing.assert_allclose(F, nodes["prof"][2], rtol=0, atol=2e-15)
    np.testing.assert_allclose(F0, nodes["prof"][2], rtol=0, atol=2e-15)
    assert np.all(vm == 0) and np.all(sd == 0) and ncl == 0


@pytest.mark.parametrize("dv", [1.0, 0.5, 2.0])
def test_discflux_uniform_velocity(dv):
    """All points moving with v0 towards the observer (an exact number s0 of grid steps): F(y) = p(y + s0 dv), i.e. the
    line is blueshifted by s0 dv; F0 = p; vmean = v0, vsig = 0."""
    grid = VelocityGrid(dv=dv, vmax=1000.0, vshift=150.0)
    nodes = _toy_nodes(4, grid, nl=3, sigmas=(30.0, 45.0, 25.0))
    nodes["prof"][:] = nodes["prof"][1]
    p = nodes["prof"][1]
    D = disc.DiscFlux(nodes, grid)
    for s0 in (37, -23):
        v0 = C_KMS * (1.0 - np.exp(-s0 * dv / C_KMS))              # -c ln(1 - v0/c) = s0 dv
        mu = np.linspace(-0.3, 1.0, 1000)
        F, F0, vm, sd, ncl = D(mu, np.full(1000, v0), *D.pairs(np.full(1000, nodes["t"][1])))
        expect = np.ones_like(p)
        if s0 > 0:
            expect[:, :-s0] = p[:, s0:]
        else:
            expect[:, -s0:] = p[:, :s0]
        np.testing.assert_allclose(F, expect, rtol=0, atol=2e-15)
        np.testing.assert_allclose(F0, p, rtol=0, atol=2e-15)
        np.testing.assert_allclose(vm, v0, rtol=1e-14)
        np.testing.assert_allclose(sd, 0.0, atol=1e-12)
        # sign convention: v > 0 (towards the observer) moves the line to smaller y (blueshift)
        c_p = np.sum(grid.y * (1 - p[0])) / np.sum(1 - p[0])
        c_F = np.sum(grid.y * (1 - F[0])) / np.sum(1 - F[0])
        assert abs((c_F - c_p) + s0 * dv) < 1e-9


def _direct(nodes, grid, mu, v, k0, k1, a):
    """F by explicit shift-and-add over the points (no FFT): the definition of DiscFlux."""
    vis = mu > 0
    s, _ = grid.shift_steps(v[vis])
    m, k0, k1, a = mu[vis], k0[vis], k1[vis], a[vis]
    nl, ny = nodes["prof"].shape[1:]
    num, den = np.zeros((nl, ny)), np.zeros(nl)
    for i in range(m.size):
        for k, wt in ((k0[i], m[i] * (1 - a[i])), (k1[i], m[i] * a[i])):
            d = nodes["fc"][k][:, None] * (1.0 - nodes["prof"][k])
            sh = int(s[i])
            if sh >= 0:
                num[:, :ny - sh] += wt * d[:, sh:]
            else:
                num[:, -sh:] += wt * d[:, :ny + sh]
            den += wt * nodes["fc"][k]
    return 1.0 - num / den[:, None]


@pytest.mark.parametrize("nl,grid", [(1, VelocityGrid(dv=2.0, vmax=900.0, vshift=150.0)),
                                     (2, VelocityGrid(dv=0.5, vmax=500.0, vshift=120.0)),
                                     (4, VelocityGrid(dv=1.0, vmax=700.0, vshift=250.0))])
def test_discflux_general_grid_vs_direct(nl, grid):
    """Any number of lines and any grid: equal to the explicit shift-and-add to rounding."""
    nodes = _toy_nodes(9, grid, nl=nl, sigmas=(30.0, 20.0, 35.0, 25.0))
    mu, v, teff = _toy_points(1500, nodes["t"], vsig=50.0, nfast=0)
    D = disc.DiscFlux(nodes, grid)
    k0, k1, a = D.pairs(teff)
    F = D(mu, v, k0, k1, a)[0]
    assert F.shape == (nl, grid.ny)
    np.testing.assert_allclose(F, _direct(nodes, grid, mu, v, k0, k1, a), rtol=0, atol=5e-15)


def test_discflux_vs_integrate_exact_toy():
    """DiscFlux vs the brute force with continuous shifts on per-point profiles that are exactly the node interpolation:
    equal to rounding for whole-step shifts, and within the 1 km/s rounding bound (0.5 x the largest profile step)
    for arbitrary velocities."""
    grid = VelocityGrid(dv=1.0, vmax=600.0, vshift=150.0)
    nl = 2
    nodes = _toy_nodes(7, grid, nl=nl, sigmas=(30.0, 22.0))
    D = disc.DiscFlux(nodes, grid)
    rng = np.random.default_rng(5)
    N = 1500
    mu = rng.uniform(-1.0, 1.0, N)
    teff = rng.uniform(nodes["t"][0], nodes["t"][-1], N)
    k0, k1, a = D.pairs(teff)
    fc = (1 - a)[:, None] * nodes["fc"][k0] + a[:, None] * nodes["fc"][k1]                     # (N, nl)
    f = ((1 - a)[:, None, None] * nodes["fc"][k0][:, :, None] * nodes["prof"][k0]
         + a[:, None, None] * nodes["fc"][k1][:, :, None] * nodes["prof"][k1]) / fc[:, :, None]
    lam = np.stack([np.broadcast_to(lam_of_y(grid.y, LREF[j]), (N, grid.ny)) for j in range(nl)], axis=1)
    w = np.where(mu > 0, mu, 0.0)[:, None] * fc
    step = 0.5 * np.abs(np.diff(nodes["prof"], axis=-1)).max()
    for whole in (True, False):
        s = rng.integers(-100, 101, N).astype(float) if whole else rng.normal(0.0, 45.0, N)
        v = C_KMS * (1.0 - np.exp(-s / C_KMS))
        F = D(mu, v, k0, k1, a)[0]
        Fx = disc.integrate_exact(lam, f, LREF[:nl], v, w, grid=grid, chunk=100)    # small row offsets
        err = np.abs(F - Fx).max()
        if whole:
            assert err < 1e-9
        else:
            assert 1e-6 < err <= step * (1 + 1e-9) + 1e-12


def test_discflux_inputs():
    grid = VelocityGrid(dv=1.0, vmax=600.0, vshift=150.0)
    nodes = _toy_nodes(5, grid, nl=2, sigmas=(30.0, 22.0))
    D = disc.DiscFlux(nodes, grid)
    assert D.pad_depth == 0.0 and D.pad_tol == disc.PAD_TOL == 1e-12 and D.node_params == {}
    wide = _toy_nodes(5, grid, nl=2, sigmas=(150.0, 22.0))        # depth ~1e-3 beyond |y| = 450
    with pytest.raises(ValueError, match="FFT convolution would cut them"):
        disc.DiscFlux(wide, grid)
    chk = grid.check_zero_padding(1.0 - wide["prof"])
    assert disc.DiscFlux(wide, grid, pad_tol=chk["max_depth"]).pad_depth == chk["max_depth"]
    # a rounding residue (as the smoothed M424 nodes: 1.78e-15) passes the default and is named as such with tol 0
    resid = _toy_nodes(5, grid, nl=2, sigmas=(30.0, 22.0))
    resid["prof"][:, 1, :4] -= 3e-15
    Dr = disc.DiscFlux(resid, grid)
    assert 2e-15 < Dr.pad_depth < 4e-15
    with pytest.raises(ValueError, match="rounding residue"):
        disc.DiscFlux(resid, grid, pad_tol=0.0)
    for bad in (-1.0, float("nan")):
        with pytest.raises(ValueError, match="pad_tol"):
            disc.DiscFlux(nodes, grid, pad_tol=bad)
    bad = dict(nodes, t=nodes["t"][::-1].copy())
    with pytest.raises(ValueError, match="increase"):
        disc.DiscFlux(bad, grid)
    with pytest.raises(ValueError, match="prof"):
        disc.DiscFlux(nodes, VelocityGrid())
    with pytest.raises(ValueError, match="fc"):
        disc.DiscFlux(dict(nodes, fc=nodes["fc"][:, :1]), grid)
    with pytest.raises(TypeError):
        disc.DiscFlux(nodes, grid.y)
    D = disc.DiscFlux(nodes, grid)
    with pytest.raises(ValueError, match="teff or pairs"):
        D.integrate_los(np.ones((2, 3)), np.zeros((2, 3)))
    # LibraryNodes works as well as the legacy dict; its params are kept
    ln = lb.LibraryNodes(nodes["t"], nodes["count"], nodes["prof"], nodes["fc"], params=dict(nmin=7, smooth=0.0))
    Dn = disc.DiscFlux(ln, grid)
    np.testing.assert_array_equal(Dn.Dhat, D.Dhat)
    assert Dn.node_params == dict(nmin=7, smooth=0.0)


def test_discflux_pairs_and_extrapolate():
    """integrate_los(pairs=...) = integrate_los(teff=...); pairs(mode='extrapolate') = library.node_pairs and
    extrapolates a node depth that is linear in T_eff' exactly (clamp holds the end nodes); inside the node range
    both modes agree."""
    grid = VelocityGrid(dv=1.0, vmax=600.0, vshift=150.0)
    nodes = _toy_nodes(6, grid, nl=2, sigmas=(30.0, 22.0))
    t = nodes["t"]
    g = np.stack([_gauss_depth(grid.y, 0.0, 30.0, 1.0), _gauss_depth(grid.y, 5.0, 22.0, 1.0)])
    amp = 0.3 + 1e-4 * (t - t[0])                                   # depth linear in T_eff', F_c = 1
    nodes["prof"] = 1.0 - amp[:, None, None] * g[None]
    nodes["fc"] = np.ones((6, 2))
    D = disc.DiscFlux(nodes, grid)
    T = np.array([t[0] - 40.0, t[-1] + 60.0, 0.5 * (t[1] + t[2])])
    for mode in ("clamp", "extrapolate"):
        for x, y in zip(D.pairs(T, mode=mode), lb.node_pairs(t, T, mode=mode)):
            np.testing.assert_array_equal(x, y)
    k0, k1, a = D.pairs(T, mode="extrapolate")
    assert a[0] < 0 and a[1] > 1 and 0 < a[2] < 1
    np.testing.assert_array_equal(a[2], D.pairs(T)[2][2])
    one, zero = np.ones(1), np.zeros(1)
    for i, Ti in enumerate(T):
        ext = D(one, zero, k0[i:i + 1], k1[i:i + 1], a[i:i + 1])[1]
        np.testing.assert_allclose(ext, 1.0 - (0.3 + 1e-4 * (Ti - t[0])) * g, rtol=0, atol=1e-14)
        cl = D(one, zero, *(p[i:i + 1] for p in D.pairs(T)))[1]
        Tc = np.clip(Ti, t[0], t[-1])
        np.testing.assert_allclose(cl, 1.0 - (0.3 + 1e-4 * (Tc - t[0])) * g, rtol=0, atol=1e-14)
    # several lines of sight with explicit pairs (both modes)
    mu, v, teff = _toy_points(3000, t, vsig=40.0, nfast=0)
    MU, V = np.stack([mu, np.roll(mu, 3)]), np.stack([v, -0.5 * v])
    r_teff = D.integrate_los(MU, V, teff=teff)
    r_pairs = D.integrate_los(MU, V, pairs=D.pairs(teff))
    pe = D.pairs(teff, mode="extrapolate")
    r_ext = D.integrate_los(MU, V, pairs=pe, novel=False)
    assert r_ext["F0"] is None
    for k in range(2):
        for key in ("F", "F0", "vmean_w", "sigma_w", "n_clip"):
            np.testing.assert_array_equal(r_pairs[key][k], r_teff[key][k], err_msg=key)
        F, _, vm, sd, ncl = D(MU[k], V[k], *pe, novel=False)
        np.testing.assert_array_equal(r_ext["F"][k], F)
        np.testing.assert_array_equal(r_ext["vmean_w"][k], vm)
        np.testing.assert_array_equal(r_ext["sigma_w"][k], sd)
    assert not np.array_equal(r_ext["F"], r_teff["F"])              # points beyond the nodes: the modes differ


# ----------------------------------------------------------------------------------------------
# integrate_exact, integrate_library_nearest
# ----------------------------------------------------------------------------------------------
def test_integrate_exact_matches_legacy(star):
    fd = _legacy_fw_disc()
    prof, vel = star
    N = prof["teff"].size
    rng = np.random.default_rng(2)
    v = 50.0 * rng.standard_normal(N)
    mu = rng.uniform(-1, 1, N)
    w = sph.disc_weights(mu, 1.0)[:, None] * prof["fcont"][:, :, 0].astype(np.float64)
    ref = fd.integrate_exact(prof, v, w, chunk=700)
    for rows in (None, "auto", 1000, 37):
        got = disc.integrate_exact(prof["lam"], prof["fnorm"], LINESET, v, w, chunk=700, rows=rows)
        np.testing.assert_array_equal(got, ref)
    # check (a) of fw_disc_los.py: one weight column repeated for the 3 lines, a subset, one line
    idx = np.sort(rng.choice(np.where(mu > 0)[0], 500, replace=False))
    ref1 = fd.integrate_exact(prof, v, np.repeat(w[:, :1], 3, axis=1), idx=idx, lines=(1,))
    got1 = disc.integrate_exact(prof["lam"], prof["fnorm"], LREF, v, w[:, 0], idx=idx, lines=(1,), rows=64)
    np.testing.assert_array_equal(got1, ref1)
    assert np.all(got1[[0, 2]] == 0)


def test_integrate_library_nearest_matches_legacy(star):
    fd = _legacy_fw_disc()
    prof, _ = star
    lib = lb.FluxLibrary.build(prof["teff"], prof["lam"], prof["fnorm"], prof["fcont"][:, :, 0], VelocityGrid(), LREF,
                               block=900)
    rng = np.random.default_rng(4)
    N = prof["teff"].size
    v = 80.0 * rng.standard_normal(N)
    v[:3] = (500.0, -450.0, 0.0)                                   # clipped by both
    mu = rng.uniform(-1, 1, N)
    w2 = sph.disc_weights(mu, 1.0)[:, None] * lib.fc[lib.bin_index(prof["teff"])]
    for w in (w2, w2[:, 0]):
        ref = fd.integrate_lib(lib.as_dict(), prof["teff"], v, w)
        np.testing.assert_array_equal(disc.integrate_library_nearest(lib, prof["teff"], v, w), ref)


# ----------------------------------------------------------------------------------------------
# integrate_exact_stream
# ----------------------------------------------------------------------------------------------
def _legacy_los(prof, vel, block, nproc, nsub, dT=10.0):
    """The frozen fw_disc_los.py run from its own source lines (data -> MU/V/SH, the per-line weight matrices and the
    streamed sums with its stream()/shift_add(), check (a), the library assembly, check (b), diagnostics, vmean/vsig;
    Pool -> builtin map, no files written)."""
    fd = _legacy_fw_disc()
    a = types.SimpleNamespace(nproc=nproc, dT=dT, nsub=nsub, block=block, nmax=0)
    ns = dict(np=np, sparse=sparse, fd=fd, a=a, Pool=_FakePool, log=lambda msg: None, N=prof["teff"].size,
              teff=prof["teff"], theta=prof["theta"], phi=prof["phi"], s=vel, lam_all=prof["lam"],
              fn_all=prof["fnorm"], fc_all=prof["fcont"][:, :, 0].astype(np.float64))
    _legacy_exec("fw_disc_los.py", "rhat, that, phat = fd.unit_vectors(theta, phi)", "del A, M, MC", ns)
    _legacy_exec("fw_disc_los.py", "# library (same format", "count=cnt, prof=lib_prof", ns)
    _legacy_exec("fw_disc_los.py", "wl = fd.weights(MU[0], 1.0)", "vsig[k, j] = np.sqrt", ns)
    return ns


def test_stream_matches_legacy(star):
    """Bit for bit the frozen fw_disc_los.py: F, F0, diagnostics, vmean/vsig, checks (a) and (b) and the library."""
    prof, vel = star
    ns = _legacy_los(prof, vel, block=400, nproc=3, nsub=600)
    got = disc.integrate_exact_stream(prof, vel, LINESET, block=400, stride=3, nsub=600, rows=150,
                                      diag_kw=FROZEN_DIAG_KW)
    np.testing.assert_array_equal(got["los"], ns["los"])
    for k, ref in (("F", ns["F"]), ("F0", ns["F0"]), ("vmean_w", ns["vmean"]), ("sigma_w", ns["vsig"]),
                   ("diag_F", ns["dF"]), ("diag_F0", ns["dF0"])):
        np.testing.assert_array_equal(got[k], ref, err_msg=k)
    assert list(got["checks"]) == list(ns["checks"])                # same keys, same order
    assert got["checks"] == ns["checks"]
    assert all(0 < x for k, x in got["checks"].items() if not k.startswith("lib_dEW"))
    for k in lb.LIBRARY_KEYS:
        np.testing.assert_array_equal(got["library"][k], ns["lib"][k], err_msg=k)
    assert got["library"].prof.dtype == np.float32
    assert not got["library"].filled.all()                          # the empty-bin fill was exercised
    assert got["stats"]["n_check"] == 600 and got["stats"]["N"] == 3000


def test_stream_library_equals_build(star):
    """The library by-product is FluxLibrary.build with the same block and stride, bit for bit."""
    prof, vel = star
    got = disc.integrate_exact_stream(prof, vel, LINESET, block=350, stride=4, checks=False)
    ref = lb.FluxLibrary.build(prof["teff"], prof["lam"], prof["fnorm"], prof["fcont"][:, :, 0], VelocityGrid(), LREF,
                               block=350, stride=4)
    for k in lb.LIBRARY_KEYS:
        np.testing.assert_array_equal(got["library"][k], ref[k], err_msg=k)
    assert got["checks"] == {}


def test_stream_parallel_fork_spawn_rows(star, tmp_path, spill_log):
    """nproc, the start method and rows do not change any bit; stride does (float64 sums), but only at rounding.
    Under spawn the in-memory profiles and the weight matrices go through memory-mapped files in tmpdir, which is
    removed afterwards."""
    prof, vel = star
    kw = dict(block=300, stride=3, nsub=400)
    ref = disc.integrate_exact_stream(prof, vel, LINESET, rows=None, **kw)
    for rows in (41, "auto"):
        _assert_results_equal(disc.integrate_exact_stream(prof, vel, LINESET, rows=rows, exact_rows=rows, **kw), ref)
    fork = disc.integrate_exact_stream(prof, vel, LINESET, nproc=2, start_method="fork", **kw)
    _assert_results_equal(fork, ref)
    assert spill_log == []                                           # fork: nothing written
    spawn = disc.integrate_exact_stream(prof, vel, LINESET, nproc=3, start_method="spawn", maxtasksperchild=2,
                                        tmpdir=str(tmp_path), **kw)
    _assert_results_equal(spawn, ref)
    assert os.listdir(str(tmp_path)) == []
    written = sorted(n for n, w in spill_log if w)
    assert "lam" in written and "fnorm" in written                  # in-memory profiles are spilled at full size
    assert sorted(n for n in written if n.startswith("m")) == sorted(
        "m{}_{}".format(j, k) for j in range(3) for k in ("data", "indices", "indptr"))
    other = disc.integrate_exact_stream(prof, vel, LINESET, block=300, stride=5, nsub=400)
    np.testing.assert_allclose(other["F"], ref["F"], rtol=0, atol=1e-14)
    np.testing.assert_allclose(other["F0"], ref["F0"], rtol=0, atol=1e-14)


def test_stream_profilestore_spawn(star, tmp_path, spill_log):
    """A profiles.npz path (ProfileStore memory maps; reopened by file name in spawn workers, never written to tmpdir)
    gives the same bits as the arrays; velocities from a sample file, recorded with its identity; an np.load()ed
    NpzFile is reopened from its file name."""
    prof, vel = star
    path = save_npz(str(tmp_path / "profiles.npz"), prof)
    spath = save_npz(str(tmp_path / "d0000.npz"), dict(vel, teff=prof["teff"].astype(np.float32)))
    kw = dict(block=500, stride=2, nsub=300)
    ref = disc.integrate_exact_stream(prof, vel, LINESET, **kw)
    got = disc.integrate_exact_stream(path, spath, LINESET, nproc=2, start_method="spawn", **kw)
    _assert_results_equal(got, ref)
    # only the weight matrices were written: lam and fnorm are memory maps of profiles.npz
    assert {"lam", "fnorm"} <= {n for n, _ in spill_log}
    assert sorted(n for n, w in spill_log if w) == sorted(
        "m{}_{}".format(j, k) for j in range(3) for k in ("data", "indices", "indptr"))
    assert got["library"].inputs == {"profiles": os.path.abspath(path)}
    assert got["params"]["velocities"] == os.path.abspath(spath)
    assert got["inputs"]["profiles"]["path"] == os.path.abspath(path)
    assert got["inputs"]["velocities"]["path"] == os.path.abspath(spath)
    assert got["params"]["inputs"] == got["inputs"]
    assert got["stats"]["teff_samples_max_diff"] <= 0.002
    meta = read_meta(disc.save_disc_los(str(tmp_path / "disc_los.npz"), got))
    assert set(meta["inputs"]) == {"profiles", "velocities"}
    assert meta["params"]["inputs"]["velocities"]["size"] == os.path.getsize(spath)
    # names from the profiles' 'lines' member when lref is an array
    got2 = disc.integrate_exact_stream(path, spath, LREF, **kw)
    assert list(got2["checks"])[:3] == ["round_" + n for n in LINES]
    # np.load()ed files (the legacy idiom): reopened by file name, nothing read whole
    with np.load(path) as zp, np.load(spath) as zs:
        got3 = disc.integrate_exact_stream(zp, zs, LINESET, **kw)
    _assert_results_equal(got3, ref)
    assert got3["library"].inputs == {"profiles": os.path.abspath(path)}
    assert got3["params"]["velocities"] == os.path.abspath(spath)
    with open(path, "rb") as f, np.load(io.BytesIO(f.read())) as zb:
        with pytest.raises(ValueError, match="without a file name"):
            disc.integrate_exact_stream(zb, vel, LINESET, **kw)


def test_stream_skips_unneeded_members(star):
    """fcont is not accessed when fc0 is given, nor teff, theta, phi when given."""
    prof, vel = star

    class Spy(dict):
        def __getitem__(self, k):
            self.read.append(k)
            return dict.__getitem__(self, k)

    spy = Spy(prof)
    spy.read = []
    fc0 = prof["fcont"][:, :, 0].astype(np.float64)
    r = disc.integrate_exact_stream(spy, vel, LINESET, fc0=fc0, teff=prof["teff"], theta=prof["theta"],
                                    phi=prof["phi"], block=500, stride=2, nsub=100)
    assert not {"fcont", "teff", "theta", "phi"} & set(spy.read)
    _assert_results_equal(r, disc.integrate_exact_stream(prof, vel, LINESET, block=500, stride=2, nsub=100))


def test_stream_lineset_order(star):
    """A LineSet must list the profiles' lines in their order (line j is placed with lref[j]); also for a
    ProfileStore. An array lref takes the profiles' names."""
    prof, vel = star
    swapped = LineSet([LINES[1], LINES[0], LINES[2]], LREF[[1, 0, 2]])
    kw = dict(block=500, stride=2, checks=False)
    with pytest.raises(ValueError, match="not the profiles' lines"):
        disc.integrate_exact_stream(prof, vel, swapped, **kw)
    with pytest.raises(ValueError, match="not the profiles' lines"):
        disc.integrate_exact_stream(ProfileStore(prof), vel, swapped, **kw)
    renamed = LineSet(["a", "b", "c"], LREF)
    with pytest.raises(ValueError, match="not the profiles' lines"):
        disc.integrate_exact_stream(prof, vel, renamed, **kw)
    noname = {k: v for k, v in prof.items() if k != "lines"}
    r = disc.integrate_exact_stream(noname, vel, renamed, **kw)     # no names to compare with
    assert r["names"] == ["a", "b", "c"]
    assert disc.integrate_exact_stream(ProfileStore(prof), vel, LREF, **kw)["names"] == LINES


def test_stream_velocity_teff_check(star, tmp_path):
    """Samples whose T_eff' are not the models' (another dump) raise unless check_teff=False; the tolerance is the
    largest teff_nudge + 0.01 K."""
    prof, vel = star
    kw = dict(block=500, stride=2, checks=False)
    other = dict(vel, teff=(prof["teff"] + 500.0).astype(np.float32))
    with pytest.raises(ValueError, match="another dump"):
        disc.integrate_exact_stream(prof, other, LREF, **kw)
    spath = save_npz(str(tmp_path / "d9999.npz"), other)
    with pytest.raises(ValueError, match="d9999.npz"):
        disc.integrate_exact_stream(prof, spath, LREF, **kw)
    mix = disc.integrate_exact_stream(prof, spath, LREF, check_teff=False, **kw)
    assert mix["stats"]["teff_samples_max_diff"] is None and mix["params"]["check_teff"] is False
    # a nudged model (T_eff' + 2 K after a failed run): the samples hold the original T_eff'
    nudged = dict(prof, teff=prof["teff"].copy(), teff_nudge=np.zeros(prof["teff"].size, np.float32))
    nudged["teff"][7] += 2.0
    nudged["teff_nudge"][7] = 2.0
    smp = dict(vel, teff=prof["teff"].astype(np.float32))
    r = disc.integrate_exact_stream(nudged, smp, LREF, **kw)
    assert 1.99 < r["stats"]["teff_samples_max_diff"] < 2.01
    with pytest.raises(ValueError, match="another dump"):
        disc.integrate_exact_stream({k: v for k, v in nudged.items() if k != "teff_nudge"}, smp, LREF, **kw)
    with pytest.raises(ValueError, match="teff has shape"):
        disc.integrate_exact_stream(prof, dict(vel, teff=prof["teff"][:10]), LREF, **kw)


def test_stream_default_diagnostics_and_check_b(star):
    """The defaults (LEGACY_DIAG_KW): diag_F, diag_F0 = diagnostics_array(..., **LEGACY_DIAG_KW); check (b) =
    integrate_library_nearest of the returned library vs F[0], with the no-Jacobian EW of the stored disc_los8.npz."""
    prof, vel = star
    r = disc.integrate_exact_stream(prof, vel, LINESET, block=400, stride=3, nsub=300)
    y = r["y"]
    assert r["params"]["diag_kw"] == disc.LEGACY_DIAG_KW
    with np.errstate(invalid="ignore", divide="ignore"):
        np.testing.assert_array_equal(r["diag_F"], diagnostics_array(r["F"], y, LREF, **disc.LEGACY_DIAG_KW))
        np.testing.assert_array_equal(r["diag_F0"], diagnostics_array(r["F0"], y, LREF, **disc.LEGACY_DIAG_KW))
    MU, TN, PN = sph.project_los(prof["theta"], prof["phi"], "thompson2024", method="matvec")
    V = sph.los_velocity(vel["ur"], vel["uth"], vel["uph"], MU, TN, PN)
    lib = r["library"]
    wl = sph.disc_weights(MU[0], 1.0)[:, None] * lib.fc[lib.bin_index(prof["teff"])]
    Fl = disc.integrate_library_nearest(lib, prof["teff"], V[0], wl)

    def ew(F, j, jac):
        return line_diagnostics(F, y, LREF[j], keys=("ew",), ew_jacobian=jac)["ew"]

    differs = []
    for j, n in enumerate(LINES):
        assert r["checks"]["lib_" + n] == float(np.abs(Fl[j] - r["F"][0, j]).max())
        assert r["checks"]["lib_dEW_" + n] == float(ew(Fl[j], j, False) - ew(r["F"][0, j], j, False))
        differs.append(r["checks"]["lib_dEW_" + n] != float(ew(Fl[j], j, True) - ew(r["F"][0, j], j, True)))
    assert any(differs)                                              # the EW option is visible in check (b)


SCRIPT_NO_MAIN_GUARD = """
import sys
sys.path.insert(0, {root!r})
sys.path.insert(0, {tests!r})
import test_disc as T
from ppmpy.synspec import disc
prof, vel = T._toy_star(300)
disc.integrate_exact_stream(prof, vel, T.LINESET, block=100, stride=2, nsub=50, nproc=2, start_method="spawn",
                            timeout=15.0, tmpdir={tmp!r})
"""


def test_stream_spawn_failed_bootstrap_does_not_hang(tmp_path):
    """A 'spawn' caller without the __main__ guard: every worker dies while bootstrapping. With the small payload the
    parent is not blocked in the launcher's pipe write (as it was with the pickled weight matrices); the watchdog
    raises PoolStalled after `timeout`."""
    script = tmp_path / "no_guard.py"
    script.write_text(SCRIPT_NO_MAIN_GUARD.format(root=ROOT, tests=os.path.dirname(os.path.abspath(__file__)),
                                                  tmp=str(tmp_path)))
    t0 = time.time()
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300)
    assert res.returncode != 0
    assert "PoolStalled" in res.stderr
    assert time.time() - t0 < 200


def test_stream_uniform_star():
    """Analytic: every point the same rest profile p. Without velocities F = F0 = p for every line of sight; a uniform
    translation u = v0 n_0 moves LOS 0 by exactly s0 grid steps (blueshift) and gives vmean = v0, sigma = 0 there."""
    grid = VelocityGrid(dv=1.0, vmax=800.0, vshift=200.0)
    prof, _ = _toy_star(1200, nl=2, span=1500.0)
    prof["lam"][:] = prof["lam"][5]
    prof["fnorm"][:] = prof["fnorm"][5]
    N = 1200
    zero = {k: np.zeros(N, np.float32) for k in ("ur", "uth", "uph")}
    p = disc.integrate_exact(prof["lam"][:1], prof["fnorm"][:1], LREF[:2], np.zeros(1), np.ones(1), grid=grid)
    r = disc.integrate_exact_stream(prof, zero, LREF[:2], grid=grid, block=256, stride=2, nsub=100)
    for k in range(8):                                              # interp_rows row offsets: ~1e-11
        np.testing.assert_allclose(r["F"][k], p, rtol=0, atol=1e-9)
        np.testing.assert_allclose(r["F0"][k], p, rtol=0, atol=1e-9)
    assert r["checks"]["round_HEI4026"] < 1e-9                     # row offsets of the blocks: ~1e-11
    s0 = 30
    v0 = C_KMS * (1.0 - np.exp(-s0 / C_KMS))
    n0 = los_thompson2024()[0]
    rhat, that, phat = sph.sphere_basis(prof["theta"], prof["phi"])
    u = {"ur": rhat @ n0 * v0, "uth": that @ n0 * v0, "uph": phat @ n0 * v0}
    r = disc.integrate_exact_stream(prof, u, LREF[:2], grid=grid, block=256, stride=2, nsub=100)
    expect = np.ones_like(p)
    expect[:, :-s0] = p[:, s0:]
    np.testing.assert_allclose(r["F"][0], expect, rtol=0, atol=1e-9)
    np.testing.assert_allclose(r["vmean_w"][0], v0, rtol=1e-12)
    np.testing.assert_allclose(r["sigma_w"][0], 0.0, atol=1e-9)
    np.testing.assert_allclose(r["vmean_w"][4], -v0, rtol=1e-12)    # the opposite line of sight sees a redshift


def test_stream_inputs(star):
    prof, vel = star
    with pytest.raises(ValueError, match="lref"):
        disc.integrate_exact_stream(prof, vel, None)
    with pytest.raises(ValueError, match="one reference wavelength"):
        disc.integrate_exact_stream(prof, vel, LREF[:2])
    with pytest.raises(ValueError, match="velocities"):
        disc.integrate_exact_stream(prof, 3.0, LREF)
    with pytest.raises(ValueError, match="ur must have shape"):
        disc.integrate_exact_stream(prof, {k: v[:10] for k, v in vel.items()}, LREF)
    with pytest.raises(ValueError, match="theta is needed"):
        disc.integrate_exact_stream({k: prof[k] for k in ("lam", "fnorm", "fcont", "teff")}, vel, LREF)
    with pytest.raises(KeyError):
        disc.integrate_exact_stream({"lam": prof["lam"]}, vel, LREF)
    for bad in ("all", 0, 2.5):
        with pytest.raises(ValueError, match="rows"):
            disc.integrate_exact_stream(prof, vel, LREF, rows=bad)
        with pytest.raises(ValueError, match="exact_rows"):
            disc.integrate_exact_stream(prof, vel, LREF, exact_rows=bad)


def test_tune_malloc_subprocess():
    """The pool workers' glibc settings apply on Linux/glibc (in a subprocess: the caller is never changed)."""
    code = ("import sys; sys.path.insert(0, {!r}); from ppmpy.synspec import disc; "
            "print(disc._tune_malloc(**disc.WORKER_MALLOC))").format(ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    import platform
    expect = sys.platform.startswith("linux") and platform.libc_ver()[0] == "glibc"
    assert out == str(expect)


def test_rows_auto():
    """rows='auto': (rows, ny) float64 temporaries of at most INTERP_BYTES (16 MiB, below glibc's 32 MiB mmap
    threshold); M424 grid: 388 rows."""
    assert disc._rows("auto", VelocityGrid().ny) == 388
    assert 388 * VelocityGrid().ny * 8 <= disc.INTERP_BYTES < 389 * VelocityGrid().ny * 8
    assert disc._rows("auto", 10 ** 9) == 1
    assert disc._rows(None, 5) is None and disc._rows(7, 5) == 7 and disc._rows(7.0, 5) == 7


# ----------------------------------------------------------------------------------------------
# the bounded, ordered task window of the stream
# ----------------------------------------------------------------------------------------------
def _window_task(arg):
    i, d, slow = arg
    open(os.path.join(d, "started_{:03d}".format(i)), "w").close()
    if i == 0:
        time.sleep(slow)
    return 10 * i


def _window_fail(i):
    if i == 3:
        raise ValueError("task 3 failed")
    return i


def _window_bad_init():
    raise RuntimeError("no state")


def _window_state(i):
    return par.worker_state()


def test_ordered_window(tmp_path):
    """Results in task order; while the oldest task is slow, at most `window` tasks have been submitted (the parent
    holds at most window - 1 finished results); a stall raises PoolStalled; task and initializer errors re-raise."""
    d = str(tmp_path)
    tasks = [(i, d, 1.5) for i in range(20)]
    with par.make_pool(4, start_method="fork") as pool:
        it = disc._ordered_window(pool, _window_task, tasks, window=3, timeout=60.0)
        first = next(it)
        started = sorted(os.listdir(d))
        rest = list(it)
    assert first == (tasks[0], 0)
    assert started == ["started_000", "started_001", "started_002"]
    assert [r for _, r in rest] == [10 * i for i in range(1, 20)]
    assert [t for t, _ in rest] == tasks[1:]
    with par.make_pool(2, start_method="fork") as pool:
        with pytest.raises(par.PoolStalled) as e:
            list(disc._ordered_window(pool, _window_task, [(i, d, 30.0) for i in range(6)], window=2, timeout=1.0))
    assert e.value.missing[0][0] == 0 and len(e.value.missing) == 6 - 1    # task 1 had arrived
    with par.make_pool(2, start_method="fork") as pool:
        with pytest.raises(ValueError, match="task 3 failed"):
            list(disc._ordered_window(pool, _window_fail, list(range(8)), window=4, timeout=60.0))
    with par.make_pool(2, initializer=_window_bad_init, start_method="fork") as pool:
        with pytest.raises(par.WorkerInitError, match="no state"):
            list(disc._ordered_window(pool, _window_state, list(range(4)), window=2, timeout=60.0))


def test_shift_add_edges():
    ny = 10
    A = np.zeros((4, ny))
    W = np.ones((4, ny))
    depth = disc._shift_add(A, W, np.array([0, 3, -12, 10]), ny)    # |s| >= ny: off the grid
    expect = np.ones(ny)
    expect[:ny - 3] += 1.0
    np.testing.assert_array_equal(depth, expect)


def test_save_disc_los(star, tmp_path):
    prof, vel = star
    r = disc.integrate_exact_stream(prof, vel, LINESET, block=600, stride=2, nsub=200)
    path = disc.save_disc_los(str(tmp_path / "disc_los8.npz"), r)
    with np.load(path) as z:
        assert sorted(z.files) == sorted(disc.DISC_LOS_KEYS + ("_meta",))
        for k in ("F", "F0", "vmean_w", "sigma_w", "diag_F", "diag_F0", "los"):
            np.testing.assert_array_equal(z[k], r[k])
        np.testing.assert_array_equal(z["Y"], r["y"])
        np.testing.assert_array_equal(z["LREF"], LREF)
        assert list(z["check_keys"]) == list(r["checks"])
        np.testing.assert_array_equal(z["check_vals"], list(r["checks"].values()))
        meta = read_meta(z)
    assert meta["kind"] == "synspec.disc_los" and meta["params"]["stride"] == 2
    json.dumps(meta)


# ----------------------------------------------------------------------------------------------
# M424 regressions
# ----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def m424_flux():
    """(get, projections): get(run) is the DiscFlux of the production nodes of a flux run of fw_disc_dumps.py
    (lib_nodes nmin=20 of library_dT10.npz; flux_sm335: smooth=335; flux_lamfix: corr = lamfix_dT10.npz), built once;
    the projections are the 'matmul' ones of the M424 grid (theta, phi of points.npz), as fw_disc_dumps.py."""
    lib = lb.FluxLibrary.load(m424_path("run", "library_dT10.npz"))
    pts = m424_path("run", "points.npz")
    th, ph = np.asarray(npz_member_memmap(pts, "theta")), np.asarray(npz_member_memmap(pts, "phi"))
    cache = {}

    def get(run):
        if run not in cache:
            kw = {}
            if run == "flux_sm335":
                kw = dict(smooth=335.0)
            elif run == "flux_lamfix":
                with np.load(m424_path("disc", "lamfix_dT10.npz")) as z:
                    kw = dict(corr=z["corr"])
            cache[run] = disc.DiscFlux(lb.lib_nodes(lib, nmin=20, **kw))       # default pad_tol (PAD_TOL)
        return cache[run]

    return get, sph.project_los(th, ph, "thompson2024")


@pytest.mark.m424
@pytest.mark.parametrize("run", ["flux", "flux_sm335", "flux_lamfix"])
@pytest.mark.parametrize("dump", [3200, 4000, 4800])
def test_m424_discflux_dumps(m424_flux, run, dump):
    """DiscFlux (+ lib_nodes, project_los 'matmul', los_velocity) reproduces the stored products of the three flux runs
    of fw_disc_dumps.py: F, F0 bitwise after the float32 cast, vmean_w, sigma_w, n_clip exactly. flux_sm335 needs the
    rounding-level default pad_tol (its nodes keep a depth of 1.78e-15 at |y| > 2300 km/s)."""
    out = np.load(m424_path("disc", run, "d{:04d}.npz".format(dump)))
    get, (MU, TN, PN) = m424_flux
    D = get(run)
    assert str(out["name"]) == run and str(out["method"]) == "flux"
    assert D.node_params["nmin"] == int(out["nmin"]) == 20
    assert D.node_params["smooth"] == float(out["smooth"])
    assert D.node_params["corr"] == bool(out["lamfix"])
    assert D.pad_depth <= D.pad_tol
    assert (D.pad_depth > 0) == (run == "flux_sm335")
    np.testing.assert_array_equal([D.t[0], D.t[-1]], out["node_range"])
    smp = _samples(dump)
    V = sph.los_velocity(smp["ur"], smp["uth"], smp["uph"], MU, TN, PN)
    r = D.integrate_los(MU, V, teff=smp["teff"].astype(np.float64))
    np.testing.assert_array_equal(r["F"].astype(np.float32), out["F"])
    np.testing.assert_array_equal(r["F0"].astype(np.float32), out["F0"])
    np.testing.assert_array_equal(r["vmean_w"], out["vmean_w"])
    np.testing.assert_array_equal(r["sigma_w"], out["sigma_w"])
    np.testing.assert_array_equal(r["n_clip"], out["n_clip"])


@pytest.mark.m424
@pytest.mark.slow
def test_m424_exact_stream():
    """integrate_exact_stream on profiles.npz (memory maps) with the dump-3200 samples (the file: its teff is checked
    against the models'), block 5000 and stride 20 (the production run: fw_disc_los.py --nproc 20), 8 workers, the
    default rows='auto': disc_los8.npz and library_dT10.npz bit for bit."""
    prof = m424_path("run", "profiles.npz")
    ref = np.load(m424_path("run", "disc_los8.npz"))
    lib_ref = np.load(m424_path("run", "library_dT10.npz"))
    spath = _samples_path(3200)
    t0, c0 = time.time(), resource.getrusage(resource.RUSAGE_CHILDREN)
    r = disc.integrate_exact_stream(prof, spath, LINESET, block=5000, stride=20, nproc=8, log=print)
    wall = time.time() - t0
    self_ru, child_ru = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    print("integrate_exact_stream M424: {:.0f} s wall, workers {:.0f} s user + {:.0f} s system, parent max RSS {:.2f} "
          "GB, worker max RSS {:.2f} GB".format(wall, child_ru.ru_utime - c0.ru_utime, child_ru.ru_stime - c0.ru_stime,
                                               self_ru.ru_maxrss / 1e6, child_ru.ru_maxrss / 1e6))
    assert r["params"]["rows"] == r["params"]["exact_rows"] == 388
    assert r["params"]["velocities"] == os.path.abspath(spath)
    assert r["stats"]["teff_samples_max_diff"] <= 1.01               # the 2 nudged points (+1 K)
    np.testing.assert_array_equal(r["y"], ref["Y"])
    np.testing.assert_array_equal(r["lref"], ref["LREF"])
    np.testing.assert_array_equal(r["los"], ref["los"])
    for k in RESULT_KEYS:
        np.testing.assert_array_equal(r[k], ref[k], err_msg=k)
    np.testing.assert_array_equal(r["diag_keys"], ref["diag_keys"])
    assert list(r["checks"]) == list(ref["check_keys"])
    np.testing.assert_array_equal(np.array(list(r["checks"].values())), ref["check_vals"])
    for k in lb.LIBRARY_KEYS:
        np.testing.assert_array_equal(r["library"][k], lib_ref[k], err_msg="library " + k)
