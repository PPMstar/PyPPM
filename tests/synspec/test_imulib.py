"""Tests of the intensity library of ppmpy.synspec.library (r_outer, imu_from_rays, flux_from_p, flux_from_rays,
find_candidates, select_representatives, Representatives, build_imu_library, ImuLibrary): analytic checks on toy rays
(no data), bitwise comparisons with the frozen legacy fw_disc.py and fw_imu_library.py (its own source lines executed
on synthetic raw / runs directories in tmp), and regressions (m424) against the M424 products in
PPMPY_SYNSPEC_M424_IMU (default /scratch/ppathak/fastwind_imu: representatives.txt, runs/, imu_library_dT10.npz).

PPMPY_SYNSPEC_M424_RAW (os.pathsep-separated; default /scratch/ppathak/fastwind_imu/raw and
/scratch/ppathak/fastwind_contfix/d3200_4parts/raw) locates the extracted candidate model directories.

The slow M424 build writes a 1.9 GB file into PPMPY_SYNSPEC_TMP (default $SCRATCH/synspec_tmp; skipped when neither is
set, so that it never lands in a RAM-backed /tmp, or when there is not enough free space) and deletes it afterwards.
Bitwise agreement with the stored M424 product needs numpy's AVX512_SKX np.log (as the production, Trillium);
elsewhere Ic / Il are compared within 1 float32 ulp (library.py module notes)."""
import filecmp
import glob
import json
import os
import pickle
import shutil
import subprocess
import sys
import textwrap
import time
import types

import numpy as np
import pytest

from conftest import ROOT, m424_path
from ppmpy.synspec import library as lb
from ppmpy.synspec import parallel as par
from ppmpy.synspec.conventions import C_KMS
from ppmpy.synspec.spectral import LineSet, VelocityGrid

LREF = np.array([4026.22, 4199.90, 4921.93])
LINES = ["HEI4026", "HEII4200", "HEI4922"]
PROJECT = os.environ.get("PPMPY_SYNSPEC_M424_PROJECT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy"))
IMU_ROOT = os.environ.get("PPMPY_SYNSPEC_M424_IMU", "/scratch/ppathak/fastwind_imu")
RAW_M424 = tuple(x for x in os.environ.get(
    "PPMPY_SYNSPEC_M424_RAW", os.pathsep.join(("/scratch/ppathak/fastwind_imu/raw",
                                               "/scratch/ppathak/fastwind_contfix/d3200_4parts/raw"))).split(os.pathsep)
                 if x)
HW_BITWISE = bool(lb._cpu_features().get("AVX512_SKX", False))


@pytest.fixture(scope="module", autouse=True)
def _no_transparent_huge_pages():
    """Transparent huge pages off for this process while these tests run (speed only; see test_library.py)."""
    libc, ok = None, False
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            ok = libc.prctl(41, 1, 0, 0, 0) == 0
        except (OSError, AttributeError):
            ok = False
    yield
    if ok:
        libc.prctl(41, 0, 0, 0, 0)


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _legacy_module():
    """The frozen fw_disc.py (skips when it is not there)."""
    if not os.path.exists(os.path.join(PROJECT, "fw_disc.py")):
        pytest.skip("legacy fw_disc.py not available in {}".format(PROJECT))
    sys.path.insert(0, PROJECT)
    try:
        import fw_disc as fd
    finally:
        sys.path.remove(PROJECT)
    return fd


def _legacy_exec(name, first, last, ns):
    """Execute the lines of a frozen legacy script from the first line starting with `first` to the next line starting
    with `last` (stripped; dedented) in namespace ns; returns ns."""
    path = os.path.join(PROJECT, name)
    if not os.path.exists(path):
        pytest.skip("legacy {} not available in {}".format(name, PROJECT))
    lines = open(path).read().splitlines()
    i0 = next(i for i, x in enumerate(lines) if x.strip().startswith(first))
    i1 = next(i for i in range(i0, len(lines)) if lines[i].strip().startswith(last))
    exec(textwrap.dedent("\n".join(lines[i0:i1 + 1])), ns)
    return ns


def _legacy_flux(p, Ic, Il, rmax):
    """fw_imu_library.py's flux check (lines 105, 114-116) on the kept rays -> fn."""
    ns = dict(np=np, p=p, Ic=Ic, Il=Il, rmax=rmax)
    _legacy_exec("fw_imu_library.py", "mf = np.linspace(0.0, 1.0, 4001)", "mf = np.linspace(0.0, 1.0, 4001)", ns)
    _legacy_exec("fw_imu_library.py", "sf = np.sqrt(1.0 - mf ** 2)", "fn = np.trapz(", ns)
    return ns["fn"]


def _imu_path(*parts):
    p = os.path.join(IMU_ROOT, *parts)
    if not os.path.exists(p):
        pytest.skip("M424 intensity products not available: {}".format(p))
    return p


def _assert_bitwise(a, b, what=""):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype, (what, a.dtype, b.dtype)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), "{}: max |diff| {}".format(
        what, np.nanmax(np.abs(a.astype(float) - b.astype(float))))


def _assert_m424(a, ref, what=""):
    """Bitwise where numpy's np.log is the production's (AVX512_SKX SVML), else within 1 float32 ulp (Ic, Il)."""
    if HW_BITWISE or np.asarray(ref).dtype != np.float32:
        _assert_bitwise(a, ref, what)
        return
    a, ref = np.asarray(a), np.asarray(ref)
    assert a.dtype == ref.dtype and a.shape == ref.shape, (what, a.dtype, ref.dtype, a.shape, ref.shape)
    assert np.all(np.abs(a.astype(np.float64) - ref) <= np.spacing(np.abs(ref)).astype(np.float64)), what


def _toy_rays(rng, nk=161, nray=30, h=0.003, u=0.5, ncore=11):
    """p (nray,) [core rays 0 .. 1, then an envelope], continuum intensity limb-darkened inside p = 1 and falling off as
    exp(-(p - 1) / h) outside, line intensity with a mu-dependent Gaussian depth; nk wavelength points."""
    core = np.linspace(0.0, 1.0, ncore)
    env = 1.0 + 1e-4 * np.cumsum(rng.uniform(5.0, 40.0, nray - ncore))
    p = np.concatenate([core, env])
    x = np.linspace(-1.0, 1.0, nk)
    c = 1e-5 * (1.0 + 0.05 * x)[:, None]
    mu = np.sqrt(np.clip(1.0 - p ** 2, 0.0, 1.0))
    ld = np.where(p <= 1.0, 1.0 - u * (1.0 - mu), (1.0 - u) * np.exp(-np.maximum(p - 1.0, 0.0) / h))
    Ic = c * ld[None, :]
    depth = 0.5 * np.exp(-0.5 * (x / 0.15) ** 2)
    Il = Ic * (1.0 - depth[:, None] * (0.6 + 0.4 * mu[None, :]))
    return p, Ic, Il


# ----------------------------------------------------------------------------------------------
# ray helpers (analytic and legacy)
# ----------------------------------------------------------------------------------------------
def test_r_outer_rule_and_legacy():
    """First ray whose continuum is below frac x centre at every wavelength point; p[-1] if none; == fw_disc.r_outer."""
    fd = _legacy_module()
    p = np.array([0.0, 0.5, 1.0, 1.01, 1.02, 1.03, 1.5])
    Ic = np.ones((4, p.size))
    Ic[:, 3:] = 1e-4                                   # below 1e-3 from ray 3 on ...
    Ic[2, 3] = 2e-3                                    # ... except at one wavelength point
    assert lb.r_outer(p, Ic) == 1.02 == fd.r_outer(p, Ic)
    assert lb.r_outer(p, Ic, frac=3e-3) == 1.01 == fd.r_outer(p, Ic, frac=3e-3)
    assert lb.r_outer(p, Ic, frac=1e-5) == 1.5 == fd.r_outer(p, Ic, frac=1e-5)       # no ray below: the last one
    Ic2 = Ic.copy()
    Ic2[:, 0] *= 7.0                                   # relative to the disc centre (column 0), per wavelength
    assert lb.r_outer(p, Ic2) == fd.r_outer(p, Ic2) == 1.01
    rng = np.random.default_rng(0)
    for h in (0.001, 0.003, 0.01):
        pp, ic, _ = _toy_rays(rng, h=h)
        assert lb.r_outer(pp, ic) == fd.r_outer(pp, ic)
        assert isinstance(lb.r_outer(pp, ic), float)


def test_imu_from_rays_matches_legacy():
    fd = _legacy_module()
    rng = np.random.default_rng(1)
    p, Ic, Il = _toy_rays(rng)
    rmax = lb.r_outer(p, Ic)
    mu, I = lb.imu_from_rays(p, Il, rmax)
    mu0, I0 = fd.imu_from_rays(p, Il, rmax)
    _assert_bitwise(mu, mu0, "mu")
    _assert_bitwise(I, I0, "I_mu")
    n = int((p <= rmax).sum())
    assert mu.size == n and I.shape == (Il.shape[0], n)
    assert mu[0] == 0.0 and mu[-1] == 1.0 and np.all(np.diff(mu) > 0)
    _assert_bitwise(I[:, -1], Il[:, 0], "disc centre last")


def test_flux_from_p_uniform_and_legacy():
    """int I 2p dp of a uniform intensity = I R^2 (trapezoid exact for a linear integrand); == fw_disc.flux_from_p."""
    fd = _legacy_module()
    p = np.array([0.0, 0.3, 0.7, 0.95, 1.0, 1.2])
    I = np.outer([1.0, 2.0, 0.5], np.ones(p.size))
    np.testing.assert_allclose(lb.flux_from_p(p, I), I[:, 0] * p[-1] ** 2, rtol=1e-14)
    rng = np.random.default_rng(2)
    pp, ic, il = _toy_rays(rng)
    _assert_bitwise(lb.flux_from_p(pp, il), fd.flux_from_p(pp, il), "flux_from_p")


def test_flux_from_rays_uniform_intensity():
    """Uniform intensity (any R_max, any ray spacing) -> F_line = F_cont = I, F/F_c = 1."""
    rng = np.random.default_rng(3)
    p = np.sort(np.concatenate([[0.0], rng.uniform(0.0, 1.013, 25), [1.013, 1.2, 2.0]]))
    I0 = rng.uniform(0.5, 2.0, 7)
    Ic = np.repeat(I0[:, None], p.size, axis=1)
    Il = 0.8 * Ic
    fn, fl, fc = lb.flux_from_rays(p, Ic, Il, 1.013, full=True)
    np.testing.assert_allclose(fc, I0, rtol=1e-13)
    np.testing.assert_allclose(fl, 0.8 * I0, rtol=1e-13)
    np.testing.assert_allclose(fn, 0.8, rtol=1e-13)
    assert lb.flux_from_rays(p, Ic, Ic, 1.013).shape == (7,)


def test_flux_from_rays_linear_limb_darkening_analytic():
    """I_c = 1 - u (1 - mu), I_l = I_c (1 - d mu): F_c = 1 - u/3, F_l = F_c - d ((2 - 2u)/3 + u/2), with rays uniform in
    mu (I linear in s between them), rays beyond R_max ignored."""
    u, d = 0.6, np.array([0.0, 0.2, 0.5])
    mu_n = np.linspace(0.0, 1.0, 801)
    rmax = 1.0125
    p = np.sort(rmax * np.sqrt(1.0 - mu_n ** 2))
    p = np.concatenate([p, [1.02, 1.5]])                                  # beyond R_max: must not matter
    mu = np.sqrt(np.clip(1.0 - (p / rmax) ** 2, 0.0, 1.0))
    Ic = np.repeat((1.0 - u * (1.0 - mu))[None, :], d.size, axis=0)
    Ic[:, -2:] = 1e3
    Il = Ic * (1.0 - d[:, None] * mu[None, :])
    fn, fl, fc = lb.flux_from_rays(p, Ic, Il, rmax, nmu=20001, full=True)
    Fc = 1.0 - u / 3.0
    Fl = Fc - d * ((2.0 - 2.0 * u) / 3.0 + u / 2.0)
    np.testing.assert_allclose(fc, Fc, rtol=2e-6)
    np.testing.assert_allclose(fl, Fl, rtol=2e-6)
    np.testing.assert_allclose(fn, Fl / Fc, rtol=2e-6)
    with pytest.raises(ValueError, match="2 rays"):
        lb.flux_from_rays(p, Ic, Il, 0.0)


def test_flux_from_rays_matches_legacy_check():
    """== the flux check of fw_imu_library.py (its own lines) bit for bit, on kept rays and on all rays."""
    rng = np.random.default_rng(4)
    for h in (0.002, 0.004):
        p, Ic, Il = _toy_rays(rng, h=h)
        rmax = lb.r_outer(p, Ic)
        keep = p <= rmax
        ref = _legacy_flux(p[keep], Ic[:, keep], Il[:, keep], rmax)
        _assert_bitwise(lb.flux_from_rays(p[keep], Ic[:, keep], Il[:, keep], rmax), ref, "kept rays")
        _assert_bitwise(lb.flux_from_rays(p, Ic, Il, rmax), ref, "all rays (masked inside)")


# ----------------------------------------------------------------------------------------------
# synthetic raw / runs directories
# ----------------------------------------------------------------------------------------------
NB, DT, T0 = 12, 10.0, 35400.0
COUNT = np.array([3, 0, 0, 2, 5, 0, 0, 0, 1, 4, 0, 0], float)       # filled bins 0, 3, 4, 8, 9


def _toy_flux_library():
    edges = T0 + DT * np.arange(NB + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    tmean = centres.copy()
    tmean[[0, 3, 4, 8, 9]] = [35404.25, 35435.0, 35443.125, 35487.5, 35491.75]
    return dict(edges=edges, tmean=tmean, count=COUNT.copy())


def _write_model_files(d, rng, nray, h, shift):
    """OUT_IMU.<line>_VTV010 (precise wavelengths, all rays) and OUT.<line>_VTV010 (0.01 A wavelengths, F/F_c from the
    rays rounded to 5 decimals, 161 rows) for the three M424 lines."""
    os.makedirs(d, exist_ok=True)
    u = np.linspace(-1.0, 1.0, 161)
    yb = 3100.0 * np.sign(u) * np.abs(u) ** 1.6 + 15.0 + shift
    for j, (ln, l0) in enumerate(zip(LINES, LREF)):
        hj = h * (1.0 + 0.3 * j) * rng.uniform(0.8, 1.25)
        p, Ic, Il = _toy_rays(rng, nray=nray, h=hj)
        lam = l0 * np.exp((yb + rng.uniform(-0.3, 0.3)) / C_KMS)
        with open(os.path.join(d, "OUT_IMU.{}_VTV010".format(ln)), "w") as fh:
            fh.write("# rays NP-1, core rays NC = {:4d} {:4d}\n".format(nray, 10))
            fh.write("# p  " + " ".join("{:15.8E}".format(x) for x in p) + "\n")
            fh.write("# K, lambda, I_cont(p_1..p_NP-1), I_line(p_1..p_NP-1)\n")
            for k in range(lam.size):
                fh.write("{:5d} {:11.4f} ".format(k + 1, lam[k]) + " ".join("{:.6E}".format(v) for v in Ic[k]) + " "
                         + " ".join("{:.6E}".format(v) for v in Il[k]) + "\n")
        # the file as read back (6 significant digits) gives the 'FASTWIND flux' of the OUT file
        from ppmpy.synspec.fwresults import read_out_imu
        lam_r, p_r, Ic_r, Il_r = read_out_imu(os.path.join(d, "OUT_IMU.{}_VTV010".format(ln)))
        rm = lb.r_outer(p_r, Ic_r)
        f = np.round(lb.flux_from_rays(p_r, Ic_r, Il_r, rm) + rng.normal(0.0, 2e-6, lam.size), 5)
        with open(os.path.join(d, "OUT.{}_VTV010".format(ln)), "w") as fh:
            for k in range(lam.size):
                fh.write("{:4d} {:11.5f} {:15.2f} {:19.6E} {:11.5f} {:11.5f}\n".format(k + 1, 1.2 - 0.015 * k, lam[k],
                                                                                   7.4e-7, f[k], f[k]))
            fh.write("  -1.08387540430798\n")


@pytest.fixture(scope="module")
def synraw(tmp_path_factory):
    """Two raw directories of candidate models (meta.txt + CONT_FORMAL; one without CONT_FORMAL, one far below the
    edges, a tie in bin 3 across the directories, one in an empty bin) and a runs directory with the model files of
    every candidate in the layout runs/P<idx:06d>/P<idx:06d>."""
    root = tmp_path_factory.mktemp("imulib")
    rng = np.random.default_rng(10)
    lib = _toy_flux_library()
    models = []                                         # (raw dir name, idx, teff)
    for b, n in ((0, 2), (4, 3), (8, 1), (9, 2)):
        for _ in range(n):
            models.append(("A" if rng.uniform() < 0.5 else "B", int(rng.integers(1000, 1300000)),
                           round(float(rng.uniform(lib["edges"][b], lib["edges"][b + 1])), 3)))
    models += [("B", 500, 35434.5), ("A", 501, 35435.5),     # bin 3: a tie around tmean = 35435.0, the first wins
               ("A", 7, 35455.0),                            # empty bin 5: ignored
               ("B", 8, 35000.0),                            # below the edges -> bin 0, far from tmean
               ("A", 9, 35491.75)]                           # exactly tmean[9], but no CONT_FORMAL: not a candidate
    for k, (rd, idx, t) in enumerate(models):
        d = root / "raw{}".format(rd) / "P{:06d}".format(idx)
        d.mkdir(parents=True)
        (d / "meta.txt").write_text("{} {:.3f} ok 60 37000.0 150.0 0.5\n".format(idx, t))
        if idx != 9:
            (d / "CONT_FORMAL").write_text("x\n")
        _write_model_files(str(root / "runs" / "P{:06d}".format(idx) / "P{:06d}".format(idx)), rng,
                           nray=int(rng.integers(24, 34)), h=float(rng.uniform(0.002, 0.006)),
                           shift=float(rng.uniform(-20, 20)))
    return dict(root=str(root), raw=[str(root / "rawA"), str(root / "rawB")], runs=str(root / "runs"), lib=lib)


def _legacy_select_and_build(synraw, raw, out):
    """fw_imu_library.py steps 1-3 (its own lines) on the synthetic directories: candidates, representatives, the
    representatives.txt and the library file out/imu_library_dT10.npz. Returns the namespace."""
    fd = _legacy_module()
    lib = synraw["lib"]
    a = types.SimpleNamespace(raw=raw, out=out, stage="select")
    ns = dict(np=np, os=os, glob=__import__("glob"), fd=fd, a=a, flib=lib, RUNS=synraw["runs"], log=lambda msg: None)
    _legacy_exec("fw_imu_library.py", "edges, tmean, count = flib[", "assert not missing", ns)
    _legacy_exec("fw_imu_library.py", "sel = os.path.join(a.out", 'log(f"wrote {sel}', ns)
    _legacy_exec("fw_imu_library.py", "mdir_of = {b: os.path.join(RUNS", "os.replace(out[:-4]", ns)
    return ns


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_select_representatives_matches_legacy(synraw, tmp_path, order):
    """find_candidates + select_representatives + Representatives.write == fw_imu_library.py (candidates, bins,
    closest to tmean, first on a tie, representatives.txt text) for both orders of the raw directories."""
    raw = [synraw["raw"][k] for k in order]
    ns = _legacy_select_and_build(synraw, raw, str(tmp_path / "legacy"))
    cands = lb.find_candidates(raw)
    assert cands == ns["cands"]
    assert all(c[0] != 9 for c in cands)                                      # no CONT_FORMAL
    reps = lb.select_representatives(synraw["lib"], cands)
    assert list(reps.bins) == sorted(ns["rep"]) == [0, 3, 4, 8, 9]
    assert [reps[b][0] for b in reps.bins] == [ns["cands"][ns["rep"][b]][0] for b in sorted(ns["rep"])]
    assert reps[3][0] == (501 if order == (0, 1) else 500)                     # tie: first in candidate order
    assert reps.missing == [] and reps.n_candidates == len(cands)
    path = reps.write(str(tmp_path / "mine.txt"))
    assert open(path).read() == open(os.path.join(str(tmp_path / "legacy"), "representatives.txt")).read()
    back = lb.read_representatives(path)
    _assert_bitwise(back.bins, reps.bins, "bins")
    _assert_bitwise(back.idx, reps.idx, "idx")
    _assert_bitwise(back.teff, reps.teff, "teff (3 decimals in meta.txt)")
    assert back.dirs == reps.dirs
    _assert_bitwise(lb.read_representatives(path, from_meta=True).teff, reps.teff, "teff from meta.txt")
    assert lb.find_candidates(raw[0]) == lb.find_candidates([raw[0]])
    assert len(lb.find_candidates(raw, require=("meta.txt",))) == len(cands) + 1


def test_select_representatives_missing_bins(synraw):
    cands = [c for c in lb.find_candidates(synraw["raw"]) if not 35480.0 <= c[1] < 35490.0]      # drop bin 8
    with pytest.raises(ValueError, match=r"1 of 5 filled bins have no candidate.*\[8\]"):
        lb.select_representatives(synraw["lib"], cands)
    reps = lb.select_representatives(synraw["lib"], cands, allow_missing=True)
    assert reps.missing == [8] and list(reps.bins) == [0, 3, 4, 9]
    none = lb.select_representatives(synraw["lib"], [], allow_missing=True)
    assert len(none) == 0 and none.missing == [0, 3, 4, 8, 9]
    with pytest.raises(ValueError, match="finite"):
        lb.select_representatives(synraw["lib"], [(1, np.nan, "x")])


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_build_matches_legacy_bitwise(synraw, tmp_path, order):
    """build_imu_library == the legacy build lines member for member, and save(meta=False) writes the same bytes as the
    legacy np.savez; the flux checks == the legacy chk rows. The toy models have different ray counts inside R_max, so
    s is NaN-padded and the empty bins (1, 2, 5, 6, 7, 10, 11) are mapped to the nearest filled bin."""
    raw = [synraw["raw"][k] for k in order]
    out = str(tmp_path / "legacy")
    ns = _legacy_select_and_build(synraw, raw, out)
    reps = lb.select_representatives(synraw["lib"], lb.find_candidates(raw))
    imu, chk = lb.build_imu_library(reps, synraw["lib"], LINES, VelocityGrid(), LREF, synraw["runs"])
    ref = np.load(os.path.join(out, "imu_library_dT10.npz"))
    assert ref.files == list(lb.IMU_KEYS)
    for k in lb.IMU_KEYS:
        _assert_bitwise(imu[k], ref[k], k)
    mine = imu.save(str(tmp_path / "mine.npz"), meta=False)
    assert filecmp.cmp(mine, os.path.join(out, "imu_library_dT10.npz"), shallow=False)
    c = ns["chk"].reshape(len(reps), 3, 3)
    _assert_bitwise(c[:, :, 1], chk["max_dF"], "max_dF")
    _assert_bitwise(c[:, :, 2], chk["dEW"], "dEW")
    assert np.all(chk["max_dF"] < 5e-5)
    # NaN padding and nnode
    nn = imu.nnode
    assert nn.min() < imu.K == nn.max()
    for b in range(imu.nb):
        for j in range(3):
            n = nn[b, j]
            assert np.all(np.isnan(imu.s[b, j, n:])) and not np.any(np.isnan(imu.s[b, j, :n]))
            assert imu.s[b, j, 0] == 0.0 and imu.s[b, j, n - 1] == 1.0 and np.all(np.diff(imu.s[b, j, :n]) > 0)
            assert not imu.Ic[b, j, n:].any() and not imu.Il[b, j, n:].any()
    # empty bins: nearest filled bin, the lower one on a tie (bin 6: 4 and 8 both 2 away)
    _assert_bitwise(imu.src, np.array([0, 0, 3, 3, 4, 4, 4, 8, 8, 9, 9, 9]), "src")
    _assert_bitwise(imu.unique_nodes(), np.array([0, 3, 4, 8, 9]), "unique nodes")
    for b in range(imu.nb):
        sb = imu.src[b]
        for k in ("s", "Ic", "Il", "rmax", "nnode"):
            _assert_bitwise(imu[k][b], imu[k][sb], "{} copy of bin {}".format(k, b))
        assert imu.idx_rep[b] == reps[sb][0] and imu.teff_rep[b] == reps[sb][1]
    assert np.all(np.diff(imu.node_teff()) > 0)
    assert set(chk["summary"]) == set(LINES)


def test_build_inputs_and_errors(synraw, tmp_path):
    """reps from representatives.txt, a mapping or Representatives; LineSet; runs_dir=None (the representatives'
    own directories); progress; allow_missing; the errors."""
    lib = synraw["lib"]
    reps = lb.select_representatives(lib, lb.find_candidates(synraw["raw"]))
    ref, _ = lb.build_imu_library(reps, lib, LINES, VelocityGrid(), LREF, synraw["runs"], check=False)
    txt = reps.write(str(tmp_path / "representatives.txt"))
    calls = []
    a, chk = lb.build_imu_library(txt, lib, LineSet(LINES, LREF), VelocityGrid().y, runs_dir=synraw["runs"],
                                  progress=lambda i, n: calls.append((i, n)))
    assert calls[-1] == (2 * len(reps), 2 * len(reps)) and len(calls) == 2 * len(reps)
    assert a.inputs["representatives"] == txt and a.inputs["runs"] == synraw["runs"]
    m = {int(b): (i, t) for b, i, t, _ in reps}
    b_, _ = lb.build_imu_library(m, lib, LINES, VelocityGrid(), LREF, synraw["runs"], check=False)
    own = lb.Representatives(reps.bins, reps.idx, reps.teff,
                             [os.path.join(synraw["runs"], lb.IMU_LAYOUT.format(idx=i)) for i in reps.idx])
    c_, _ = lb.build_imu_library(own, lib, LINES, VelocityGrid(), LREF, runs_dir=None, check=False)
    for x in (a, b_, c_):
        for k in lb.IMU_KEYS:
            _assert_bitwise(x[k], ref[k], k)
    # a filled bin without a representative
    part = lb.Representatives(reps.bins[reps.bins != 8], reps.idx[reps.bins != 8], reps.teff[reps.bins != 8],
                              [d for b, d in zip(reps.bins, reps.dirs) if b != 8])
    with pytest.raises(ValueError, match=r"1 filled bins have no representative: \[8\]"):
        lb.build_imu_library(part, lib, LINES, VelocityGrid(), LREF, synraw["runs"])
    d_, _ = lb.build_imu_library(part, lib, LINES, VelocityGrid(), LREF, synraw["runs"], check=False,
                                 allow_missing=True)
    assert d_.src[8] == 9 and d_.src[7] == 9 and d_.src[6] == 4
    # missing OUT_IMU files (all listed), bins outside the library, lref
    with pytest.raises(ValueError, match=r"5 representatives have no OUT_IMU/OUT files"):
        lb.build_imu_library(reps, lib, LINES, VelocityGrid(), LREF, str(tmp_path / "nowhere"))
    bad = lb.Representatives([0, 12], [1, 2], [1.0, 2.0], ["a", "b"])
    with pytest.raises(ValueError, match="0 .. nb - 1"):
        lb.build_imu_library(bad, lib, LINES, VelocityGrid(), LREF, synraw["runs"], allow_missing=True)
    with pytest.raises(ValueError, match="lref is required"):
        lb.build_imu_library(reps, lib, LINES, VelocityGrid(), runs_dir=synraw["runs"])
    with pytest.raises(ValueError, match="strictly increasing"):
        lb.Representatives([3, 3], [1, 2], [1.0, 2.0], ["a", "b"])


# ----------------------------------------------------------------------------------------------
# ImuLibrary: files, memory maps, pickling, memory estimate
# ----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synimu(synraw, tmp_path_factory):
    reps = lb.select_representatives(synraw["lib"], lb.find_candidates(synraw["raw"]))
    imu, _ = lb.build_imu_library(reps, synraw["lib"], LINES, VelocityGrid(), LREF, synraw["runs"])
    d = tmp_path_factory.mktemp("synimu")
    return dict(imu=imu, legacy=imu.save(str(d / "legacy.npz"), meta=False), dir=d)


def test_imu_library_load_save(synimu, tmp_path):
    imu = synimu["imu"]
    lib = lb.ImuLibrary.load(synimu["legacy"])
    assert lib.meta == {} and lib.inputs == dict(source=synimu["legacy"])
    assert isinstance(lib.Ic, np.memmap) and isinstance(lib.Il, np.memmap) and not isinstance(lib.s, np.memmap)
    for k in lb.IMU_KEYS:
        _assert_bitwise(lib[k], imu[k], k)
    full = lb.ImuLibrary.load(synimu["legacy"], mmap=False)
    assert not isinstance(full.Ic, np.memmap)
    # save with '_meta' -> load keeps provenance; meta=False -> the legacy bytes again (also from memory maps)
    p = lib.save(str(tmp_path / "with_meta.npz"))
    back = lb.ImuLibrary.load(p)
    assert back.meta["kind"] == "synspec.imu_library" and "source_meta" not in back.meta
    third = lb.ImuLibrary.load(back.save(str(tmp_path / "third.npz")))           # keeps the source's '_meta'
    assert third.meta["source_meta"]["kind"] == "synspec.imu_library"
    for k in lb.IMU_KEYS:
        _assert_bitwise(back[k], imu[k], k)
    p2 = back.save(str(tmp_path / "again.npz"), meta=False)
    assert filecmp.cmp(p2, synimu["legacy"], shallow=False)
    assert "Ic" in lib and lib.files == list(lb.IMU_KEYS) and set(lib.as_dict()) == set(lb.IMU_KEYS)
    assert (lib.nb, lib.nl, lib.ny) == (NB, 3, VelocityGrid().ny) and lib.K == imu.nnode.max()
    _assert_bitwise(lib.bin_index([35399.0, 35415.0, 99999.0]), np.array([0, 1, 11]), "bin_index")
    assert "ImuLibrary(nb=12, nodes=5" in repr(lib)
    # compressed members are read, not mapped
    pc = str(tmp_path / "compressed.npz")
    np.savez_compressed(pc, **imu.as_dict())
    cz = lb.ImuLibrary.load(pc)
    assert not isinstance(cz.Ic, np.memmap)
    _assert_bitwise(cz.Il, imu.Il, "compressed Il")
    np.savez(str(tmp_path / "bad.npz"), edges=imu.edges)
    with pytest.raises(ValueError, match="not an intensity library"):
        lb.ImuLibrary.load(str(tmp_path / "bad.npz"))
    with pytest.raises(ValueError, match="src"):
        d = imu.as_dict()
        d["src"] = d["src"].copy()
        d["src"][0] = 1                                 # bin 1 is not its own source
        lb.ImuLibrary(**d)


def test_imu_library_pickle_and_spawn(synimu, tmp_path):
    """A memory-mapped library travels as its path (memory maps again on the other side); a replaced file raises; an
    in-memory library travels with its arrays. 'fork' and 'spawn' workers: same library."""
    path = str(tmp_path / "lib.npz")
    shutil.copy(synimu["legacy"], path)
    lib = lb.ImuLibrary.load(path)
    s = pickle.dumps(lib)
    assert len(s) < 10000                                             # no arrays inside
    back = pickle.loads(s)
    assert isinstance(back.Ic, np.memmap) and back.path == os.path.abspath(path)
    _assert_bitwise(back.Il, lib.Il, "Il")
    mem = pickle.loads(pickle.dumps(synimu["imu"]))
    _assert_bitwise(mem.Ic, synimu["imu"].Ic, "in-memory Ic")
    for method in ("fork", "spawn"):
        with par.make_pool(1, start_method=method) as pool:
            assert pool.apply(repr, (lib,)) == repr(lib), method
    lib.save(path, meta=False)                                         # replaced (new inode)
    with pytest.raises(RuntimeError, match="changed since it was opened"):
        pickle.loads(s)


MEM_KEYS = ("library", "source", "source_mapped", "per_call")


@pytest.mark.parametrize("fft", ["precomputed", "lazy"])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_imu_library_memory_estimate_equals_discimu(synimu, fft, dtype):
    """memory_estimate (without building) == DiscImu(lib, ...).memory() for every fft / dtype mode, line subsets (by
    index and by name), two grids and chunks, an in-memory and a memory-mapped library (source_mapped)."""
    from ppmpy.synspec import disc
    imu = synimu["imu"]
    mapped = lb.ImuLibrary.load(synimu["legacy"])
    for lib in (imu, mapped):
        for lines in (None, [1], (2, 0)):
            for grid, chunk in ((None, 128), (VelocityGrid(vshift=150.0), 7), (VelocityGrid(vmax=2000.0), 7)):
                if grid is not None and grid.ny != lib.ny:
                    # both refuse a grid with other points than the library's (another vshift is accepted)
                    with pytest.raises(ValueError):
                        disc.DiscImu(lib, grid=grid, lines=lines, dtype=dtype, fft=fft, chunk=chunk)
                    with pytest.raises(ValueError):
                        lib.memory_estimate(grid=grid, lines=lines, dtype=dtype, fft=fft, chunk=chunk)
                    continue
                D = disc.DiscImu(lib, grid=grid, lines=lines, dtype=dtype, fft=fft, chunk=chunk)
                est = lib.memory_estimate(grid=grid, lines=lines, dtype=dtype, fft=fft, chunk=chunk)
                assert {k: est[k] for k in MEM_KEYS} == D.memory(), (fft, dtype, lines, chunk)
                assert (est["nn"], est["K"], est["L"], est["nv"], tuple(est["built"])) == (D.nn, D.K, D.L, D.nv, D.built)
                assert est["total"] == est["library"] + est["source"] + est["per_call"]
                assert est["anonymous"] == est["total"] - est["source_mapped"]
                if fft == "lazy":
                    assert est["source"] > 0 and est["source_mapped"] == (est["source"] if lib is mapped else 0)
                else:
                    assert est["source"] == est["source_mapped"] == 0
    by_name = imu.memory_estimate(lines=["HEII4200"], dtype=dtype, fft=fft)
    D = disc.DiscImu(imu, lines=["HEII4200"], dtype=dtype, fft=fft)
    assert {k: by_name[k] for k in MEM_KEYS} == D.memory() == {k: imu.memory_estimate(
        lines=[1], dtype=dtype, fft=fft)[k] for k in MEM_KEYS}


def test_imu_library_memory_estimate_errors(synimu):
    imu = synimu["imu"]
    pre = imu.memory_estimate()
    assert pre == imu.memory_estimate(dtype=np.float64) and pre["nn"] == 5 and pre["nr"] == 5 * imu.K
    with pytest.raises(ValueError, match="fft"):
        imu.memory_estimate(fft="never")
    with pytest.raises(ValueError, match="dtype"):
        imu.memory_estimate(dtype="float16")
    with pytest.raises(ValueError, match="chunk"):
        imu.memory_estimate(chunk=0)
    with pytest.raises(ValueError, match="ny"):
        lb.ImuLibrary.load(synimu["legacy"]).memory_estimate(grid=VelocityGrid(vmax=600.0))
    with pytest.raises(ValueError, match="recorded grid"):
        imu.memory_estimate(grid=VelocityGrid(vmax=600.0))
    assert imu.memory_estimate(grid=VelocityGrid()) == pre
    with pytest.raises(ValueError, match="unknown line"):
        lb.ImuLibrary.load(synimu["legacy"]).memory_estimate(lines=["HEI4026"])      # the legacy file has no names
    with pytest.raises(ValueError, match="line indices"):
        imu.memory_estimate(lines=[3])


def test_imu_library_lines_grid_and_discimu_lref(synraw, synimu, tmp_path):
    """The build records lines, lref and the grid: ImuLibrary.lines / lref / grid, kept by save() + load() ('_meta');
    DiscImu takes lines and lref from the library. The legacy file records none of them."""
    from ppmpy.synspec import disc
    imu = synimu["imu"]
    assert isinstance(imu.lines, LineSet) and imu.lines.names == LINES
    _assert_bitwise(imu.lref, LREF, "lref")
    assert imu.grid.to_dict() == VelocityGrid().to_dict() and imu.params["dv"] == 1.0
    D = disc.DiscImu(imu)
    _assert_bitwise(D.lref, LREF, "DiscImu lref")
    assert D.names == LINES
    back = lb.ImuLibrary.load(imu.save(str(tmp_path / "with_meta.npz")))
    assert back.lines.names == LINES and back.grid.to_dict() == VelocityGrid().to_dict()
    _assert_bitwise(disc.DiscImu(back, fft="lazy").lref, LREF, "DiscImu lref (loaded)")
    legacy = lb.ImuLibrary.load(synimu["legacy"])
    assert legacy.lines is None and legacy.lref is None and legacy.grid is None
    assert disc.DiscImu(legacy, fft="lazy").lref is None
    # a plain array as the grid: dv recorded, no VelocityGrid
    reps = lb.select_representatives(synraw["lib"], lb.find_candidates(synraw["raw"]))
    arr, _ = lb.build_imu_library(reps, synraw["lib"], LINES, VelocityGrid().y, LREF, synraw["runs"], check=False)
    assert arr.grid is None and arr.params["dv"] == 1.0 and arr.params["grid"] is None
    # mapping protocol: iteration yields the member names
    assert list(imu) == list(lb.IMU_KEYS) and set(dict(imu)) == set(lb.IMU_KEYS)
    with pytest.raises(ValueError, match="params"):
        lb.ImuLibrary(params=dict(lines=LINES[:2], lref=LREF[:2].tolist()), **imu.as_dict())


def test_imu_library_load_paths_and_pickle(synimu, tmp_path, monkeypatch):
    """inputs['source'] and path are absolute for a relative path; a compressed file (members read) and mmap=False
    pickle their arrays and keep the path; only memory-mapped libraries travel as their path."""
    imu = synimu["imu"]
    d = tmp_path / "rel"
    d.mkdir()
    shutil.copy(synimu["legacy"], str(d / "legacy.npz"))
    np.savez_compressed(str(d / "compressed.npz"), **imu.as_dict())
    monkeypatch.chdir(str(d))
    rel = lb.ImuLibrary.load("legacy.npz")
    assert rel.inputs["source"] == rel.path == str(d / "legacy.npz") and rel._mmap
    monkeypatch.chdir(str(tmp_path))
    meta = lb.ImuLibrary.load(rel.save(str(tmp_path / "again.npz"))).meta        # saved from elsewhere: source kept
    assert meta["inputs"]["source"]["path"] == str(d / "legacy.npz")
    cz = lb.ImuLibrary.load(str(d / "compressed.npz"))
    assert not cz._mmap and not isinstance(cz.Ic, np.memmap)
    full = lb.ImuLibrary.load(str(d / "legacy.npz"), mmap=False)
    assert not full._mmap
    for x in (cz, full):
        s = pickle.dumps(x)
        assert len(s) > x.Ic.nbytes                                     # the arrays travel
        back = pickle.loads(s)
        assert back.path == x.path and not isinstance(back.Ic, np.memmap)
        _assert_bitwise(back.Il, imu.Il, "Il")
    assert len(pickle.dumps(rel)) < 10000


def test_representative_dirs_from_imu_library_and_representatives(synraw, synimu, tmp_path):
    """representative_dirs and wavelength_rounding_correction accept an ImuLibrary (its idx_rep and edges, as its file)
    and Representatives (dirs of their bins, None elsewhere; T_eff' checked against edges)."""
    imu = synimu["imu"]
    runs = synraw["runs"]
    ref = lb.representative_dirs(synimu["legacy"], imu.nb, runs=runs, edges=imu.edges)
    assert lb.representative_dirs(imu, imu.nb, runs=runs, edges=imu.edges) == ref
    assert lb.representative_dirs(imu.idx_rep, imu.nb, runs=runs) == ref
    with pytest.raises(ValueError, match="bins"):
        lb.representative_dirs(imu, imu.nb, runs=runs, edges=imu.edges + 5.0)
    reps = lb.select_representatives(synraw["lib"], lb.find_candidates(synraw["raw"]))
    rd = lb.representative_dirs(reps, imu.nb, runs=runs, edges=imu.edges)
    assert [rd[b] for b in reps.bins] == [ref[b] for b in reps.bins]
    assert all(rd[b] is None for b in range(imu.nb) if b not in reps)
    assert lb.representative_dirs(reps, imu.nb) == [reps[b][2] if b in reps else None for b in range(imu.nb)]
    shifted = lb.Representatives(reps.bins, reps.idx, reps.teff + DT, reps.dirs)
    with pytest.raises(ValueError, match="outside their bins"):
        lb.representative_dirs(shifted, imu.nb, runs=runs, edges=imu.edges)
    with pytest.raises(ValueError, match="0 .. nb - 1"):
        lb.representative_dirs(reps, 5, runs=runs)
    # the wavelength rounding correction from the library object, its file and the representatives
    flib = dict(synraw["lib"], prof=np.zeros((NB, 3, VelocityGrid().ny)))
    c_file = lb.wavelength_rounding_correction(flib, synimu["legacy"], LINES, VelocityGrid(), LREF, runs=runs)
    assert np.abs(c_file).max() > 1e-4
    _assert_bitwise(lb.wavelength_rounding_correction(flib, imu, LINES, VelocityGrid(), LREF, runs=runs), c_file,
                    "corr from ImuLibrary")
    _assert_bitwise(lb.wavelength_rounding_correction(flib, reps, LineSet(LINES, LREF), VelocityGrid(), runs=runs),
                    c_file, "corr from Representatives")


def test_representatives_read_strips_and_from_meta(synraw, tmp_path):
    """Trailing blanks / tabs / CRLF do not end up in the directories; from_meta True / 'auto' / False."""
    reps = lb.select_representatives(synraw["lib"], lb.find_candidates(synraw["raw"]))
    path = str(tmp_path / "r.txt")
    with open(path, "wb") as f:
        for b, i, t, d in reps:
            f.write("{} {} {:.3f} {} \t \r\n".format(b, i, t, d).encode())
        f.write(b"   \r\n")
    for fm in (False, True, "auto"):
        back = lb.read_representatives(path, from_meta=fm)
        assert back.dirs == reps.dirs, fm
        _assert_bitwise(back.teff, reps.teff, "teff")
    nometa = str(tmp_path / "nometa.txt")
    with open(nometa, "w") as f:
        f.write("0 1 35400.5 {}\n".format(tmp_path / "nowhere"))
    assert lb.read_representatives(nometa, from_meta="auto").teff[0] == 35400.5           # no meta.txt: the file
    with pytest.raises(FileNotFoundError):
        lb.read_representatives(nometa, from_meta=True)
    with pytest.raises(ValueError, match="from_meta"):
        lb.read_representatives(nometa, from_meta="yes")


def test_find_candidates_require_str_and_status(synraw, tmp_path):
    """require as one str == as a tuple; the opt-in status filter (meta.txt 3rd column; no column: no match)."""
    cands = lb.find_candidates(synraw["raw"])
    assert lb.find_candidates(synraw["raw"], require="CONT_FORMAL") == cands
    assert lb.find_candidates(synraw["raw"], status="ok") == lb.find_candidates(synraw["raw"], status=("ok",)) == cands
    raw = str(tmp_path / "raw")
    shutil.copytree(synraw["raw"][0], raw)
    dirs = [c[2] for c in lb.find_candidates(raw)]                         # the first two candidates are changed
    with open(os.path.join(dirs[0], "meta.txt")) as f:
        tok = f.read().split()
    with open(os.path.join(dirs[0], "meta.txt"), "w") as f:
        f.write(" ".join(tok[:2] + ["formal_failed"] + tok[3:]) + "\n")
    with open(os.path.join(dirs[1], "meta.txt")) as f:
        tok = f.read().split()
    with open(os.path.join(dirs[1], "meta.txt"), "w") as f:
        f.write(" ".join(tok[:2]) + "\n")
    every = lb.find_candidates(raw)
    ok = lb.find_candidates(raw, status=("ok",))
    assert [c[2] for c in every[2:]] == [c[2] for c in ok] and len(every) == len(ok) + 2
    assert [c[2] for c in lb.find_candidates(raw, status=("ok", "formal_failed"))] == [dirs[0]] + [c[2] for c in ok]


def test_build_teff_from_meta_matches_legacy(synraw, tmp_path):
    """meta.txt with 4 decimals: the legacy build takes teff_rep from meta.txt (its candidate scan), representatives.txt
    keeps 3 decimals; build_imu_library(representatives.txt) with the default teff_from_meta='auto' reproduces the
    legacy file bit for bit, teff_from_meta=False gives the rounded teff_rep (everything else equal)."""
    raw = []
    for r in synraw["raw"]:
        dst = str(tmp_path / os.path.basename(r))
        shutil.copytree(r, dst)
        for m in glob.glob(os.path.join(dst, "P*", "meta.txt")):
            with open(m) as f:
                tok = f.read().split()
            tok[1] = "{:.4f}".format(float(tok[1]) + 0.0004)
            with open(m, "w") as f:
                f.write(" ".join(tok) + "\n")
        raw.append(dst)
    out = str(tmp_path / "legacy")
    _legacy_select_and_build(synraw, raw, out)
    ref = np.load(os.path.join(out, "imu_library_dT10.npz"))
    txt = os.path.join(out, "representatives.txt")
    imu, _ = lb.build_imu_library(txt, synraw["lib"], LINES, VelocityGrid(), LREF, synraw["runs"])
    for k in lb.IMU_KEYS:
        _assert_bitwise(imu[k], ref[k], k)
    assert np.all(np.round(ref["teff_rep"] * 1e4) % 10 == 4)                 # really 4 decimals
    rounded, _ = lb.build_imu_library(txt, synraw["lib"], LINES, VelocityGrid(), LREF, synraw["runs"], check=False,
                                      teff_from_meta=False)
    assert not np.any(rounded.teff_rep == ref["teff_rep"])
    np.testing.assert_allclose(rounded.teff_rep, ref["teff_rep"], atol=5e-4, rtol=0)
    for k in lb.IMU_KEYS:
        if k != "teff_rep":
            _assert_bitwise(rounded[k], ref[k], k)


def test_build_upfront_checks(synraw, tmp_path):
    """Missing OUT files fail before any file is parsed when check=True (not with check=False); representatives whose
    T_eff' lies in another bin (made for other edges) are refused; T_eff' beyond the edges is clamped to the end bins."""
    lib = synraw["lib"]
    reps = lb.select_representatives(lib, lb.find_candidates(synraw["raw"]))
    ref, _ = lb.build_imu_library(reps, lib, LINES, VelocityGrid(), LREF, synraw["runs"], check=False)
    own = []
    for b, i, t, d in reps:
        src = os.path.join(synraw["runs"], lb.IMU_LAYOUT.format(idx=i))
        dst = str(tmp_path / "imu_only" / "P{:06d}".format(i))
        os.makedirs(dst)
        for f in glob.glob(os.path.join(src, "OUT_IMU.*")):
            os.symlink(f, os.path.join(dst, os.path.basename(f)))
        own.append(dst)
    only = lb.Representatives(reps.bins, reps.idx, reps.teff, own)
    seen = []
    with pytest.raises(ValueError, match=r"5 representatives have no OUT files \(needed by the flux check"):
        lb.build_imu_library(only, lib, LINES, VelocityGrid(), LREF, check=True, progress=lambda i, n: seen.append(i))
    assert seen == []                                                          # nothing parsed
    b_, _ = lb.build_imu_library(only, lib, LINES, VelocityGrid(), LREF, check=False)
    for k in lb.IMU_KEYS:
        _assert_bitwise(b_[k], ref[k], k)
    shifted = lb.Representatives(reps.bins, reps.idx, reps.teff + DT, reps.dirs)
    with pytest.raises(ValueError, match=r"bins \[0, 3, 4, 8, 9\] lie outside their bins"):
        lb.build_imu_library(shifted, lib, LINES, VelocityGrid(), LREF, synraw["runs"])
    t = reps.teff.copy()
    t[0], t[-1] = lib["edges"][0] - 500.0, lib["edges"][-1] + 500.0          # beyond the edges: end bins
    lib2 = dict(lib, count=lib["count"].copy())
    lib2["count"][-1] = 1.0
    clamped = lb.Representatives(np.r_[reps.bins, NB - 1], np.r_[reps.idx, reps.idx[-1]], np.r_[t[:-1], reps.teff[-1],
                                                                                              t[-1]],
                                 reps.dirs + [reps.dirs[-1]])
    c_, _ = lb.build_imu_library(clamped, lib2, LINES, VelocityGrid(), LREF, synraw["runs"], check=False)
    assert c_.teff_rep[0] == t[0] and c_.teff_rep[NB - 1] == t[-1] and c_.src[NB - 1] == NB - 1


# ----------------------------------------------------------------------------------------------
# M424
# ----------------------------------------------------------------------------------------------
@pytest.mark.m424
def test_m424_select_representatives(tmp_path):
    """find_candidates over the raw directories (PPMPY_SYNSPEC_M424_RAW) + select_representatives(library_dT10) ==
    representatives.txt (same text), also with the status filter ('ok'); teff of the file's 3 decimals == the meta.txt
    values. The candidate count (3881 on 2026-10-02) is printed, not asserted: the raw directories are not protected
    products."""
    for r in RAW_M424:
        if not os.path.isdir(r):
            pytest.skip("M424 raw model directories not available: {}".format(r))
    ref_txt = _imu_path("representatives.txt")
    flib = lb.FluxLibrary.load(m424_path("run", "library_dT10.npz"))
    t0 = time.time()
    cands = lb.find_candidates(RAW_M424)
    reps = lb.select_representatives(flib, cands)
    t1 = time.time()
    ok = lb.find_candidates(RAW_M424, status="ok")
    print("\nM424 representatives: {} candidates ({} with status ok), {} bins, {:.1f} s".format(
        len(cands), len(ok), len(reps), t1 - t0))
    assert len(reps) == 303 and reps.missing == []
    ref = open(ref_txt).read()
    assert open(reps.write(str(tmp_path / "r.txt"))).read() == ref
    assert open(lb.select_representatives(flib, ok).write(str(tmp_path / "ok.txt"))).read() == ref
    _assert_bitwise(lb.read_representatives(ref_txt).teff, lb.read_representatives(ref_txt, from_meta=True).teff,
                    "teff")


@pytest.mark.m424
def test_m424_load_legacy_library():
    """The stored library loads with memory maps (Ic, Il), 303 unique nodes, node T_eff' increasing (the legacy
    DiscImu's t), nnode 42, s in [0, 1]. memory_estimate: 'precomputed' float64 = the 7.70 GB of arrays the legacy
    DiscImu held (measured 2026-10-02); the lazy modes equal DiscImu(lib, fft='lazy').memory() (cheap to build: only
    the line-centre continuum is read) with the intensities memory-mapped, not process memory."""
    from ppmpy.synspec import disc
    lib = lb.ImuLibrary.load(_imu_path("imu_library_dT10.npz"))
    assert isinstance(lib.Ic, np.memmap) and (lib.nb, lib.nl, lib.K, lib.ny) == (351, 3, 42, 5401)
    u = lib.unique_nodes()
    assert u.size == 303 and np.all(np.diff(lib.node_teff()) > 0)
    assert np.all(lib.nnode == 42) and np.nanmin(lib.s) == 0.0 and np.nanmax(lib.s) == 1.0
    assert lib.lines is None and lib.lref is None and lib.grid is None             # the legacy file records none
    est = lib.memory_estimate()
    assert (est["nr"], est["L"], est["source"]) == (12726, 7200, 0)
    assert 7.69e9 < est["library"] < 7.71e9
    for dtype in ("float64", "float32"):
        for lines in (None, [1]):
            e = lib.memory_estimate(fft="lazy", dtype=dtype, lines=lines)
            assert {k: e[k] for k in MEM_KEYS} == disc.DiscImu(lib, fft="lazy", dtype=dtype, lines=lines).memory()
            assert e["source"] == e["source_mapped"] == 2 * 12726 * 5401 * 4 * (3 if lines is None else 1)
            assert e["anonymous"] < 0.2e9


@pytest.mark.m424
def test_m424_flux_from_rays_vs_out_subset():
    """flux_from_rays vs the model's own OUT profile for every 10th representative (31 models x 3 lines) <= 1e-5, and
    == the legacy flux check bit for bit."""
    from ppmpy.synspec.fwresults import read_out, read_out_imu
    reps = lb.read_representatives(_imu_path("representatives.txt"))
    worst = 0.0
    for b, i, t, _ in list(reps)[::10]:
        d = os.path.join(_imu_path("runs"), lb.IMU_LAYOUT.format(idx=i))
        for ln in LINES:
            lam, p, Ic, Il = read_out_imu(os.path.join(d, "OUT_IMU.{}_VTV010".format(ln)))
            rmax = lb.r_outer(p, Ic)
            keep = p <= rmax
            fn = lb.flux_from_rays(p, Ic, Il, rmax)
            _assert_bitwise(fn, _legacy_flux(p[keep], Ic[:, keep], Il[:, keep], rmax), "legacy flux check")
            worst = max(worst, float(np.abs(fn - read_out(os.path.join(d, "OUT.{}_VTV010".format(ln)))["fnorm"]).max()))
    print("\nM424 flux from rays vs OUT (every 10th representative): max |dF| {:.2e}".format(worst))
    assert worst <= 1e-5


_M424_BUILD = r"""
import json, os, resource, sys, time
sys.path.insert(0, sys.argv[1])
try:
    import ctypes
    ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)
except Exception:
    pass
import numpy as np
from ppmpy.synspec import library as lb
from ppmpy.synspec.spectral import LineSet, VelocityGrid
flux_lib, imu_root, out = sys.argv[2:5]
flib = lb.FluxLibrary.load(flux_lib)
t0 = time.time()
# representatives.txt as a path: teff_rep from the meta.txt of the raw directories (teff_from_meta='auto')
imu, chk = lb.build_imu_library(os.path.join(imu_root, "representatives.txt"), flib,
                                LineSet(["HEI4026", "HEII4200", "HEI4922"], [4026.22, 4199.90, 4921.93]),
                                VelocityGrid(), runs_dir=os.path.join(imu_root, "runs"))
dt = time.time() - t0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
imu.save(out, meta=False)
print(json.dumps(dict(time=dt, peak_rss=rss, summary=chk["summary"], max_dF=float(np.max(chk["max_dF"])),
                      max_abs_dEW=float(np.max(np.abs(chk["dEW"]))))))
"""


@pytest.mark.m424
@pytest.mark.slow
def test_m424_build_equals_stored_library():
    """build_imu_library(representatives.txt, library_dT10, runs) == imu_library_dT10.npz member for member, bit for
    bit, and save(meta=False) writes the same bytes (in a fresh process: time and peak RSS of the build are printed);
    flux from the rays vs OUT <= 1e-5 for all 303 representatives x 3 lines."""
    stored = _imu_path("imu_library_dT10.npz")
    _imu_path("representatives.txt")
    _imu_path("runs")
    flux_lib = m424_path("run", "library_dT10.npz")
    d = os.environ.get("PPMPY_SYNSPEC_TMP") or (os.path.join(os.environ["SCRATCH"], "synspec_tmp")
                                                 if os.environ.get("SCRATCH") else None)
    if d is None:
        pytest.skip("set PPMPY_SYNSPEC_TMP (or SCRATCH) to a disk directory for the 1.9 GB library (not a RAM /tmp)")
    os.makedirs(d, exist_ok=True)
    if shutil.disk_usage(d).free < 2.5 * os.path.getsize(stored):
        pytest.skip("not enough free space in {} for the 1.9 GB library (set PPMPY_SYNSPEC_TMP)".format(d))
    out = os.path.join(d, "imu_library_test_{}.npz".format(os.getpid()))
    try:
        res = subprocess.run([sys.executable, "-c", _M424_BUILD, ROOT, flux_lib, IMU_ROOT, out], capture_output=True,
                             text=True, timeout=1800)
        assert res.returncode == 0, res.stderr[-3000:]
        info = json.loads(res.stdout.strip().splitlines()[-1])
        print("\nM424 intensity library build: {:.1f} s, peak RSS {:.2f} GB; flux vs OUT max |dF| {:.2e}, max |dEW| "
              "{:.2e} A".format(info["time"], info["peak_rss"] / 1e9, info["max_dF"], info["max_abs_dEW"]))
        assert info["max_dF"] <= 1e-5
        for ln, s in info["summary"].items():
            assert s["nnode_min"] == s["nnode_max"] == 42, (ln, s)
        mine, ref = lb.ImuLibrary.load(out), lb.ImuLibrary.load(stored)
        for k in lb.IMU_KEYS:
            if k in ("Ic", "Il"):
                for b in range(ref.nb):                                 # bin by bin: no 955 MB temporaries
                    _assert_m424(mine[k][b], ref[k][b], "{} bin {}".format(k, b))
            else:
                _assert_bitwise(mine[k], ref[k], k)
        if HW_BITWISE:
            assert filecmp.cmp(out, stored, shallow=False)
    finally:
        if os.path.exists(out):
            os.remove(out)
