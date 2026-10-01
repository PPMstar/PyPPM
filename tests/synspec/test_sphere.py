"""Tests of ppmpy.synspec.sphere: equal-area grid, basis, line-of-sight projections, velocities, weights.

Oracles: the frozen fw_disc.py (imported) and literal source lines of the frozen scripts fw_disc_dumps.py,
fw_disc_los.py and fw_sphere_extract.py (executed by _legacy_exec), see tests/synspec/legacy/README.txt.
"""
import glob
import os
import subprocess
import sys
import textwrap
import types

import numpy as np
import pytest

import conftest
from conftest import ROOT, m424_path
from ppmpy.synspec import conventions as cv
from ppmpy.synspec import sphere as sph

LEGACY = os.environ.get("PPMPY_SYNSPEC_M424_PROJECT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy"))


def _legacy_path(name):
    p = os.path.join(LEGACY, name)
    if not os.path.exists(p):
        pytest.skip("legacy file not available: {}".format(p))
    return p


def _legacy_fw_disc():
    _legacy_path("fw_disc.py")
    sys.path.insert(0, LEGACY)
    try:
        import fw_disc as fd
    finally:
        sys.path.remove(LEGACY)
    return fd


def _legacy_exec(name, first, last, **ns):
    """Execute the source lines of a frozen legacy script from the line starting with `first` to the next line
    starting with `last` (both stripped, dedented) in namespace ns; returns the namespace."""
    lines = open(_legacy_path(name)).read().splitlines()
    i0 = next(i for i, s in enumerate(lines) if s.strip().startswith(first))
    i1 = next(i for i in range(i0, len(lines)) if lines[i].strip().startswith(last))
    exec(textwrap.dedent("\n".join(lines[i0:i1 + 1])), ns)
    return ns


def _samples(dump):
    """Per-dump sphere samples of sphere_sample.py (teff, ur, uth, uph float32, t_s); skips when absent.
    Uses conftest's m424_path('samples', ...) once conftest.M424 has that entry (orchestrator); until then the same
    default location and environment variable here."""
    name = "d{:04d}.npz".format(dump)
    if "samples" in conftest.M424:
        return np.load(m424_path("samples", name))
    root = os.environ.get("PPMPY_SYNSPEC_M424_SAMPLES", "/scratch/ppathak/fastwind_sphere/samples_r4050_N1236544")
    p = os.path.join(root, name)
    if not os.path.exists(p):
        pytest.skip("M424 product not available: {}".format(p))
    return np.load(p)


def _numpy_blas_is_openblas():
    """True when numpy is linked against OpenBLAS (numpy >= 1.25: show_config dicts; else the wheels' bundled
    library)."""
    try:
        name = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
        return "openblas" in str(name).lower()
    except Exception:
        d = os.path.dirname(np.__file__)
        return bool(glob.glob(os.path.join(d, os.pardir, "numpy.libs", "libopenblas*"))
                    or glob.glob(os.path.join(d, ".dylibs", "libopenblas*")))


EPS1 = 2.3e-16          # ~1 ulp at 1: the projections are <= 1 in magnitude
OPENBLAS = _numpy_blas_is_openblas()     # chunks and LOS subsets are checked bit for bit with OpenBLAS, else to EPS1


# ----------------------------------------------------------------------------------------------
# synthetic / analytic
# ----------------------------------------------------------------------------------------------
def test_no_heavy_imports():
    code = ("import sys; import ppmpy.synspec.sphere, ppmpy.synspec.parallel;"
            "bad=[m for m in ('ppmpy.ppm','matplotlib','nugridpy','pyshtools') if m in sys.modules];"
            "print(','.join(bad))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                         env=dict(os.environ, PYTHONPATH=ROOT))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", "heavy modules imported: " + out.stdout


def test_fibonacci_sphere_properties():
    n = 1000
    th, ph = sph.fibonacci_sphere(n)
    assert th.shape == ph.shape == (n,) and th.dtype == ph.dtype == np.float64
    assert np.all((th > 0) & (th < np.pi)) and np.all((ph >= 0) & (ph < 2 * np.pi))
    # equal area: cos(theta) uniformly spaced in (-1, 1)
    np.testing.assert_allclose(np.cos(th), 1.0 - 2.0 * (np.arange(n) + 0.5) / n, rtol=0, atol=1e-14)
    # same spiral as the observer directions of conventions (which do not wrap phi)
    r, _, _ = sph.sphere_basis(th, ph)
    np.testing.assert_allclose(r, cv.fibonacci_directions(n), rtol=0, atol=1e-11)
    assert sph.fibonacci_sphere(1)[0][0] == np.pi / 2
    for bad in (0, -3, 2.5):
        with pytest.raises(ValueError):
            sph.fibonacci_sphere(bad)


def test_fibonacci_sphere_matches_ppm():
    ppm = pytest.importorskip("ppmpy.ppm")
    n, radius = 20000, np.array([4050.0])
    th, ph = sph.fibonacci_sphere(n)
    # the grid method of MomsDataSet does not use the instance except for _get_igrid, which only uses its arguments
    dummy = types.SimpleNamespace(_get_igrid=lambda r, t, p, npts: ppm.MomsDataSet._get_igrid(None, r, t, p, npts))
    igrid, th_p, ph_p = ppm.MomsDataSet._constantArea_spherical_grid(dummy, radius, n)
    np.testing.assert_array_equal(th, th_p)
    np.testing.assert_array_equal(ph, ph_p)
    x, y, z = sph.sphere_xyz(th, ph, radius[0])
    ig = np.asarray(igrid).reshape(-1, 3)                   # columns z, y, x
    np.testing.assert_array_equal(ig[:, 0], z)
    np.testing.assert_array_equal(ig[:, 1], y)
    np.testing.assert_array_equal(ig[:, 2], x)


def test_sphere_xyz_like_extract():
    n = 5000
    th, ph = sph.fibonacci_sphere(n)
    ns = _legacy_exec("fw_sphere_extract.py", "ind = np.arange(a.npoints)", "phi = g - 2.0",
                      np=np, a=types.SimpleNamespace(npoints=n, radius=4050.0))
    np.testing.assert_array_equal(th, ns["theta"])
    np.testing.assert_array_equal(ph, ns["phi"])
    ns = _legacy_exec("fw_sphere_extract.py", "r = np.full(a.npoints", "x, y, z = r", **ns)
    for got, ref in zip(sph.sphere_xyz(th, ph, 4050.0), (ns["x"], ns["y"], ns["z"])):
        np.testing.assert_array_equal(got, ref)
    np.testing.assert_allclose(np.sqrt(sum(c ** 2 for c in sph.sphere_xyz(th, ph, 4050.0))), 4050.0, rtol=1e-15)


def test_sphere_basis_orthonormal_and_legacy():
    fd = _legacy_fw_disc()
    th, ph = sph.fibonacci_sphere(3000)
    r, t, p = sph.sphere_basis(th, ph)
    for a, b in zip((r, t, p), fd.unit_vectors(th, ph)):
        np.testing.assert_array_equal(a, b)
    eye = np.einsum("nia,nja->nij", np.stack([r, t, p], 1), np.stack([r, t, p], 1))
    np.testing.assert_allclose(eye, np.broadcast_to(np.eye(3), eye.shape), atol=1e-15)
    np.testing.assert_allclose(np.cross(r, t), p, atol=1e-15)              # right-handed (r, theta, phi)
    # theta_hat, phi_hat are the derivatives of r_hat
    h = 1e-6
    np.testing.assert_allclose((sph.sphere_basis(th + h, ph)[0] - sph.sphere_basis(th - h, ph)[0]) / (2 * h), t, atol=1e-9)
    st = np.sin(th)[:, None]
    np.testing.assert_allclose((sph.sphere_basis(th, ph + h)[0] - sph.sphere_basis(th, ph - h)[0]) / (2 * h), st * p, atol=1e-9)
    # any shape
    r2, _, _ = sph.sphere_basis(th.reshape(30, 100), ph.reshape(30, 100))
    assert r2.shape == (30, 100, 3)


def test_project_los_matches_legacy_synthetic():
    """'matmul' = the MU, TN, PN lines of fw_disc_dumps.py; 'matvec' = fw_disc.mu_vlos; on any grid (same BLAS)."""
    fd = _legacy_fw_disc()
    n = 20011
    th, ph = sph.fibonacci_sphere(n)
    ns = _legacy_exec("fw_disc_dumps.py", "rhat, that, phat = fd.unit_vectors", "MU, TN, PN = (np.ascontiguousarray",
                      np=np, fd=fd, pts=dict(theta=th, phi=ph))
    los = fd.los8()
    np.testing.assert_array_equal(los, cv.los_thompson2024())
    for spec in (los, "thompson2024"):
        got = sph.project_los(th, ph, spec)
        for a, b in zip(got, (ns["MU"], ns["TN"], ns["PN"])):
            assert a.shape == (8, n) and a.dtype == np.float64 and a.flags.c_contiguous
            np.testing.assert_array_equal(a, b)
    rhat, that, phat = fd.unit_vectors(th, ph)
    mu_v, tn_v, pn_v = sph.project_los(th, ph, los, method="matvec")
    for k in range(8):
        np.testing.assert_array_equal(mu_v[k], rhat @ los[k])
        np.testing.assert_array_equal(tn_v[k], that @ los[k])
        np.testing.assert_array_equal(pn_v[k], phat @ los[k])
    ex = sph.project_los(th, ph, los, method="explicit")
    for a, b in zip(ex, (rhat, that, phat)):
        np.testing.assert_array_equal(a, b[:, 0] * los[:, 0:1] + b[:, 1] * los[:, 1:2] + b[:, 2] * los[:, 2:3])
    # the three methods differ by at most an ulp or two (absolute <= 2 eps, values <= 1)
    for a, b, c in zip(sph.project_los(th, ph, los), (mu_v, tn_v, pn_v), ex):
        assert np.max(np.abs(a - b)) <= 4.5e-16 and np.max(np.abs(a - c)) <= 4.5e-16


def _same_bits_or_ulp(a, b, exact):
    if exact:
        np.testing.assert_array_equal(a, b)
    else:
        assert np.max(np.abs(a - b)) <= EPS1


@pytest.mark.parametrize("method", sph.PROJECT_METHODS)
def test_project_los_chunks_and_subsets(method):
    """Blocks of points and subsets of lines of sight do not change any bit: 'explicit' always (elementwise IEEE),
    'matmul' and 'matvec' with OpenBLAS (thanks to the padding; test_project_los_padding_needed shows that this
    test sees its 1-ulp effects); other BLAS libraries to 1 ulp."""
    exact = method == "explicit" or OPENBLAS
    th, ph = sph.fibonacci_sphere(7777)
    los = cv.los_thompson2024()
    full = sph.project_los(th, ph, los, method=method)
    for chunk in (1, 2, 3, 1000, 7776, 7777, 10 ** 6, np.int64(500), 2000.0):
        got = sph.project_los(th, ph, los, method=method, chunk=chunk)
        for a, b in zip(got, full):
            assert a.flags.c_contiguous
            _same_bits_or_ulp(a, b, exact)
    for sub in (slice(0, 1), slice(2, 3), slice(2, 5), slice(7, 8)):
        for chunk in (None, 1, 999):
            got = sph.project_los(th, ph, los[sub], method=method, chunk=chunk)
            for a, b in zip(got, full):
                assert a.shape == (sub.stop - sub.start, th.size)
                _same_bits_or_ulp(a, b[sub], exact)
    for i in (0, 1, 4000, 7776):
        one = sph.project_los(th[i:i + 1], ph[i:i + 1], los, method=method)
        assert one[0].shape == (8, 1)
        for a, b in zip(one, full):
            _same_bits_or_ulp(a[:, 0], b[:, i], exact)


def test_project_los_padding_needed():
    """Sensitivity of the bitwise test above: without the padding, a single line of sight (numpy uses dgemv) and a
    one-point block (dot) differ from the 8-LOS dgemm in the last bit somewhere, and the padded versions do not."""
    if not OPENBLAS:
        pytest.skip("numpy's BLAS is not OpenBLAS: blocks are checked to 1 ulp only")
    th, ph = sph.fibonacci_sphere(7777)
    los = cv.los_thompson2024()
    r, _, _ = sph.sphere_basis(th, ph)
    mu = sph.project_los(th, ph, los)[0]
    n_los = int(np.sum((r @ los[2:3].T)[:, 0] != mu[2]))                    # unpadded single LOS
    n_one = int(np.sum([np.any((r[i:i + 1] @ los.T)[0] != mu[:, i]) for i in range(500)]))   # unpadded points
    if n_los == 0 and n_one == 0:
        pytest.skip("this BLAS/CPU gives the same bits without padding: the bitwise test cannot see the padding here")
    assert n_los > 0 and n_one > 0, (n_los, n_one)          # Trillium (OpenBLAS 0.3.23, EPYC 9655): 1529 and 271
    np.testing.assert_array_equal(sph.project_los(th, ph, los[2:3])[0][0], mu[2])
    np.testing.assert_array_equal(np.concatenate([sph.project_los(th[i:i + 1], ph[i:i + 1], los)[0] for i in range(500)],
                                                 axis=1), mu[:, :500])


def test_project_los_inputs():
    th, ph = sph.fibonacci_sphere(100)
    mu, tn, pn = sph.project_los(th, ph, [0.0, 0.0, 1.0])               # single (3,) vector -> (1, N)
    assert mu.shape == tn.shape == pn.shape == (1, 100)
    np.testing.assert_allclose(mu[0], np.cos(th), atol=1e-15)
    np.testing.assert_allclose(tn[0], -np.sin(th), atol=1e-15)
    np.testing.assert_allclose(pn[0], 0.0, atol=0)
    with pytest.raises(ValueError, match="unit vectors"):
        sph.project_los(th, ph, [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        sph.project_los(th, ph, np.ones((2, 2)))
    with pytest.raises(ValueError):
        sph.project_los(th, ph, cv.los_thompson2024(), method="gemm")
    with pytest.raises(ValueError):
        sph.project_los(th, ph[:50], cv.los_thompson2024())
    for bad in (0, -5, 2.5, "10"):
        with pytest.raises(ValueError, match="chunk"):
            sph.project_los(th, ph, cv.los_thompson2024(), chunk=bad)
    # a normalised array from los_array is accepted as it is
    L = cv.los_array("fibonacci:5")
    assert sph.project_los(th, ph, L)[0].shape == (5, 100)


def test_float32_coordinates_evaluated_in_float64():
    """float32 theta, phi (e.g. a float32 memory map) are converted to float64 before sin/cos."""
    th, ph = sph.fibonacci_sphere(3001)
    th32, ph32 = th.astype(np.float32), ph.astype(np.float32)
    th64, ph64 = th32.astype(np.float64), ph32.astype(np.float64)
    for got, ref in zip(sph.project_los(th32, ph32, "thompson2024"), sph.project_los(th64, ph64, "thompson2024")):
        assert got.dtype == np.float64
        np.testing.assert_array_equal(got, ref)
    for got, ref in zip(sph.sphere_basis(th32, ph32), sph.sphere_basis(th64, ph64)):
        assert got.dtype == np.float64
        np.testing.assert_array_equal(got, ref)
    for got, ref in zip(sph.sphere_xyz(th32, ph32, 4050.0), sph.sphere_xyz(th64, ph64, 4050.0)):
        np.testing.assert_array_equal(got, ref)
    # float64 input: no copy (memory maps stay memory maps underneath)
    assert np.shares_memory(sph._float64(th), th)


def test_los_velocity_uniform_flow():
    """A uniform velocity U decomposed into (u_r, u_theta, u_phi) has v = U . n at every point."""
    th, ph = sph.fibonacci_sphere(5000)
    r, t, p = sph.sphere_basis(th, ph)
    U = np.array([12.0, -30.0, 7.5])
    ur, uth, uph = r @ U, t @ U, p @ U
    los = cv.los_array("fibonacci:6")
    v = sph.los_velocity(ur, uth, uph, *sph.project_los(th, ph, los))
    assert v.shape == (6, 5000)
    np.testing.assert_allclose(v, np.broadcast_to((los @ U)[:, None], v.shape), rtol=0, atol=1e-13)
    # rigid rotation about z, u = Omega z x r (u_phi = Omega sin theta on the unit sphere): v = 0 seen pole-on and
    # v = -Omega sin(theta) sin(phi) seen from +x
    om, z0 = 2.0, np.zeros_like(th)
    v = sph.los_velocity(z0, z0, om * np.sin(th), *sph.project_los(th, ph, [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))
    np.testing.assert_allclose(v[0], 0.0, atol=1e-15)
    np.testing.assert_allclose(v[1], -om * np.sin(th) * np.sin(ph), rtol=0, atol=1e-15)
    assert sph.los_velocity(ur, uth, uph, *(x[0] for x in sph.project_los(th, ph, los))).shape == (5000,)


def test_los_velocity_legacy_order():
    """Bit for bit fw_disc_dumps.py:94 (float64 samples) and fw_disc.mu_vlos (float32 samples)."""
    fd = _legacy_fw_disc()
    n = 4096
    th, ph = sph.fibonacci_sphere(n)
    rng = np.random.default_rng(3)
    u32 = [(rng.normal(size=n) * 40).astype(np.float32) for _ in range(3)]
    los = fd.los8()
    MU, TN, PN = sph.project_los(th, ph, los)
    v = sph.los_velocity(*u32, MU, TN, PN)
    ur, uth, uph = (u.astype(np.float64) for u in u32)
    for k in range(8):
        ns = _legacy_exec("fw_disc_dumps.py", "v = ur * MU[k]", "v = ur * MU[k]",
                          ur=ur, uth=uth, uph=uph, MU=MU, TN=TN, PN=PN, k=k)
        np.testing.assert_array_equal(v[k], ns["v"])
        np.testing.assert_array_equal(sph.los_velocity(ur, uth, uph, MU[k], TN[k], PN[k]), ns["v"])
    rhat, that, phat = fd.unit_vectors(th, ph)
    mv = sph.project_los(th, ph, los, method="matvec")
    vv = sph.los_velocity(*u32, *mv)
    for k in range(8):
        mu, vref = fd.mu_vlos(los[k], rhat, that, phat, *u32)
        np.testing.assert_array_equal(mv[0][k], mu)
        np.testing.assert_array_equal(vv[k], vref)


def test_disc_weights():
    fd = _legacy_fw_disc()
    th, ph = sph.fibonacci_sphere(200000)
    mu = sph.project_los(th, ph, "thompson2024")[0]
    fc = np.linspace(0.5, 1.5, th.size)
    for uld in (0.0, 0.3):
        np.testing.assert_array_equal(sph.disc_weights(mu, fc[None, :], uld), fd.weights(mu, fc[None, :], uld))
        np.testing.assert_array_equal(sph.disc_weights(mu[3], 1.0, uld), fd.weights(mu[3], 1.0, uld))
    w = sph.disc_weights(mu)
    assert np.all(w[mu <= 0] == 0) and np.all(w[mu > 0] == mu[mu > 0])
    # linear limb darkening: integral of mu I(mu) over the visible hemisphere = pi (1 - u/3)
    dA = 4 * np.pi / th.size
    for u in (0.0, 0.4, 1.0):
        np.testing.assert_allclose(sph.disc_weights(mu, 1.0, u).sum(axis=1) * dA, np.pi * (1 - u / 3), rtol=1e-6)


def test_quadrature_check():
    e1, e2 = sph.quadrature_check(5000, "thompson2024")
    assert e1.shape == e2.shape == (8,)
    assert np.max(np.abs(e1)) == pytest.approx(3.05e-5, rel=0.01)       # value quoted in the docstring
    assert np.max(np.abs(e2)) == pytest.approx(7.39e-6, rel=0.01)
    f1, f2 = sph.quadrature_check(50000, "thompson2024")
    assert np.max(np.abs(f1)) < np.max(np.abs(e1)) / 10 and np.max(np.abs(f2)) < np.max(np.abs(e2)) / 5
    for m in ("matvec", "explicit"):
        np.testing.assert_allclose(sph.quadrature_check(5000, "thompson2024", method=m)[0], e1, rtol=0, atol=1e-13)
    # blocks: the same sums up to rounding, for any block size
    for chunk in (1, 777, 4999, None):
        g1, g2 = sph.quadrature_check(5000, "thompson2024", chunk=chunk)
        np.testing.assert_allclose(g1, e1, rtol=0, atol=1e-14)
        np.testing.assert_allclose(g2, e2, rtol=0, atol=1e-14)
    # the same mu as project_los (directly from the definition, with mu only projected)
    th, ph = sph.fibonacci_sphere(5000)
    mu = sph.project_los(th, ph, "thompson2024")[0]
    dA = 4 * np.pi / 5000
    np.testing.assert_array_equal(e1, sph.disc_weights(mu).sum(axis=1) * dA / np.pi - 1.0)
    for bad in (0, 2.5):
        with pytest.raises(ValueError):
            sph.quadrature_check(bad, "thompson2024")
        with pytest.raises(ValueError):
            sph.quadrature_check(100, "thompson2024", chunk=bad)
    with pytest.raises(ValueError):
        sph.quadrature_check(100, "thompson2024", method="gemm")


# ----------------------------------------------------------------------------------------------
# M424 regressions
# ----------------------------------------------------------------------------------------------
N424 = 1236544


@pytest.fixture(scope="module")
def m424_grid():
    pts = np.load(m424_path("run", "points.npz"))
    th, ph = sph.fibonacci_sphere(N424)
    return th, ph, pts


@pytest.mark.m424
def test_m424_grid_bitwise(m424_grid):
    th, ph, pts = m424_grid
    np.testing.assert_array_equal(th, pts["theta"])
    np.testing.assert_array_equal(ph, pts["phi"])
    for got, k in zip(sph.sphere_xyz(th, ph, 4050.0), "xyz"):
        np.testing.assert_array_equal(got, pts[k])
    from ppmpy.synspec.io import npz_member_memmap
    prof = m424_path("run", "profiles.npz")                     # the coordinates the per-point models were merged with
    np.testing.assert_array_equal(np.asarray(npz_member_memmap(prof, "theta")), th)
    np.testing.assert_array_equal(np.asarray(npz_member_memmap(prof, "phi")), ph)


@pytest.mark.m424
def test_m424_projection_and_velocity_bitwise(m424_grid):
    """MU, TN, PN and v of fw_disc_dumps.py (frozen source lines) for dump 3200, also in chunks and LOS subsets."""
    fd = _legacy_fw_disc()
    th, ph, pts = m424_grid
    ns = _legacy_exec("fw_disc_dumps.py", "rhat, that, phat = fd.unit_vectors", "MU, TN, PN = (np.ascontiguousarray",
                      np=np, fd=fd, pts=pts)
    ref = (ns["MU"], ns["TN"], ns["PN"])
    got = sph.project_los(th, ph, "thompson2024")
    for a, b in zip(got, ref):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(sph.project_los(th, ph, fd.los8(), chunk=100003), ref):     # bounded memory, same bits
        np.testing.assert_array_equal(a, b)
    for sub in (slice(0, 1), slice(5, 8)):                                        # padding keeps dgemm for one LOS
        for a, b in zip(sph.project_los(th, ph, fd.los8()[sub]), ref):
            np.testing.assert_array_equal(a, b[sub])
    smp = _samples(3200)
    ur, uth, uph = (smp[k].astype(np.float64) for k in ("ur", "uth", "uph"))
    v = sph.los_velocity(smp["ur"], smp["uth"], smp["uph"], *got)
    for k in range(8):
        nsv = _legacy_exec("fw_disc_dumps.py", "v = ur * MU[k]", "v = ur * MU[k]",
                           ur=ur, uth=uth, uph=uph, MU=ref[0], TN=ref[1], PN=ref[2], k=k)
        np.testing.assert_array_equal(v[k], nsv["v"])


@pytest.mark.m424
def test_m424_methods_same_doppler_steps(m424_grid):
    """Module notes: the three projection methods differ in the last bit of mu, tn, pn (~30 % of the values), but for
    dump 3200 no rounded Doppler shift (fw_disc.shift_steps) and no visibility (mu > 0) changes, so the F profiles do
    not depend on the method."""
    fd = _legacy_fw_disc()
    th, ph, _ = m424_grid
    smp = _samples(3200)
    ref = None
    for m in sph.PROJECT_METHODS:
        P = sph.project_los(th, ph, "thompson2024", method=m)
        V = sph.los_velocity(smp["ur"], smp["uth"], smp["uph"], *P)
        cur = (P[0], P[0] > 0, fd.shift_steps(V)[0])
        del P, V
        if ref is None:
            ref = cur
            continue
        assert 0.05 < np.mean(cur[0] != ref[0]) < 0.6                   # the bits do differ
        np.testing.assert_array_equal(cur[1], ref[1])
        np.testing.assert_array_equal(cur[2], ref[2])


@pytest.mark.m424
def test_m424_flux_product_d3200(m424_grid):
    """End to end: our geometry + the frozen DiscFlux reproduce the stored dump-3200 flux product."""
    fd = _legacy_fw_disc()
    th, ph, _ = m424_grid
    out = np.load(m424_path("disc", "flux", "d3200.npz"))
    smp = _samples(3200)
    nodes = fd.lib_nodes(np.load(m424_path("run", "library_dT10.npz")), nmin=int(out["nmin"]))
    INT = fd.DiscFlux(nodes)
    teff = smp["teff"].astype(np.float64)
    k0, k1, w1 = INT.pairs(teff)
    lo, hi = teff < INT.t[0], teff > INT.t[-1]
    MU, TN, PN = sph.project_los(th, ph, "thompson2024")
    V = sph.los_velocity(smp["ur"], smp["uth"], smp["uph"], MU, TN, PN)
    for k in range(8):
        F, F0, vm, sd, ncl = INT(MU[k], V[k], k0, k1, w1)
        np.testing.assert_array_equal(F.astype(np.float32), out["F"][k])
        np.testing.assert_array_equal(F0.astype(np.float32), out["F0"][k])
        np.testing.assert_array_equal(vm, out["vmean_w"][k])
        np.testing.assert_array_equal(sd, out["sigma_w"][k])
        assert ncl == out["n_clip"][k]
        vis = MU[k] > 0
        assert MU[k][vis & (lo | hi)].sum() / MU[k][vis].sum() == out["wout"][k]


@pytest.mark.m424
@pytest.mark.slow
def test_m424_matvec_disc_los8(m424_grid):
    """'matvec' reproduces the weighted mean and rms line-of-sight velocities of disc_los8.npz (fw_disc_los.py)."""
    from ppmpy.synspec.io import npz_member_memmap
    fd = _legacy_fw_disc()
    th, ph, _ = m424_grid
    ref = np.load(m424_path("run", "disc_los8.npz"))
    s = _samples(3200)
    fc_all = np.asarray(npz_member_memmap(m424_path("run", "profiles.npz"), "fcont")[:, :, 0]).astype(np.float64)
    MU, TN, PN = sph.project_los(th, ph, "thompson2024", method="matvec")
    for a, b in zip(sph.project_los(th, ph, cv.los_thompson2024()[2:3], method="matvec", chunk=262147), (MU, TN, PN)):
        np.testing.assert_array_equal(a, b[2:3])                              # chunks and LOS subsets: same bits
    V = sph.los_velocity(s["ur"], s["uth"], s["uph"], MU, TN, PN)             # float32 samples, as fw_disc_los.py
    ns = _legacy_exec("fw_disc_los.py", "vmean, vsig = np.zeros", "vsig[k, j] = np.sqrt", np=np, fd=fd, MU=MU, V=V,
                      fc_all=fc_all)
    np.testing.assert_array_equal(ns["vmean"], ref["vmean_w"])
    np.testing.assert_array_equal(ns["vsig"], ref["sigma_w"])
    np.testing.assert_array_equal(ref["los"], cv.los_thompson2024())
