"""Tests of ppmpy.synspec.library: synthetic / analytic checks (no data), bitwise comparisons with the frozen legacy
fw_disc.py, fw_disc_los.py and fw_disc_holdout.py on synthetic inputs (float64 bin sums and means, so that summation
order is visible: the float32 cast hides it), the pool watchdog, and regressions (m424) against the M424 production
products (library_dT10.npz, lamfix_dT10.npz, the per-dump outputs of fw_disc_dumps.py).

PPMPY_SYNSPEC_M424_IMU (default /scratch/ppathak/fastwind_imu) locates the intensity library and its model runs.
The M424 comparisons are bitwise only where numpy computes np.log with the AVX512_SKX SVML routines, as for the
production (Trillium); elsewhere they allow 1 float32 ulp / 1e-15 (library.py module notes)."""
import contextlib
import os
import pathlib
import resource
import shutil
import signal
import sys
import textwrap
import time
import types

import numpy as np
import pytest

from scipy import sparse

from conftest import m424_path
from ppmpy.synspec import io as sio
from ppmpy.synspec import library as lb
from ppmpy.synspec import parallel as par
from ppmpy.synspec.conventions import C_KMS
from ppmpy.synspec.spectral import LineSet, VelocityGrid, interp_rows, y_of_lam

LREF = np.array([4026.22, 4199.90, 4921.93])
LINES = ["HEI4026", "HEII4200", "HEI4922"]
# frozen copies of the original project sources (tests/synspec/legacy/README.txt)
PROJECT = os.environ.get("PPMPY_SYNSPEC_M424_PROJECT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy"))
IMU_ROOT = os.environ.get("PPMPY_SYNSPEC_M424_IMU", "/scratch/ppathak/fastwind_imu")
LIB_KEYS = ("edges", "tmean", "count", "prof", "fc", "dT")
NODE_KEYS = ("t", "count", "prof", "fc")
# the stored M424 products were computed with numpy's AVX512_SKX SVML np.log; without it 3 % of the y = c ln(lam/lref)
# values differ by 1 ulp
HW_BITWISE = bool(lb._cpu_features().get("AVX512_SKX", False))


@pytest.fixture(scope="module", autouse=True)
def _no_transparent_huge_pages():
    """Transparent huge pages off for this process (inherited by its fork and spawn workers) while these tests run. On a
    host with THP 'always' and fragmented memory (the Trillium login nodes) a large temporary of the interpolation can
    cost seconds of direct compaction (2026-10-01: interp_rows of 800 rows 1.8 s per call, 94 % system time, against
    0.18 s with THP off; this module between 1.5 and 5.7 min). Linux only (prctl PR_SET_THP_DISABLE = 41); changes
    speed, never results."""
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
    with `last` (stripped; dedented) in namespace ns (as test_sphere.py); returns ns."""
    path = os.path.join(PROJECT, name)
    if not os.path.exists(path):
        pytest.skip("legacy {} not available in {}".format(name, PROJECT))
    lines = open(path).read().splitlines()
    i0 = next(i for i, x in enumerate(lines) if x.strip().startswith(first))
    i1 = next(i for i in range(i0, len(lines)) if lines[i].strip().startswith(last))
    exec(textwrap.dedent("\n".join(lines[i0:i1 + 1])), ns)
    return ns


def _legacy_library64(teff, lam, fnorm, fcont, chunk, dT=10.0, sums=False):
    """fw_disc.library's arithmetic in float64, run from its own source lines (fw_disc.py:70-88, in library()): the bin
    sums of the profiles before the division (sums=True: up to 'prof[:, j] += S @ f') or the float64 bin means (up to
    'fcs[ok] /= cnt[ok, None]'; empty bins still zero). Returns the namespace (edges, b, cnt, tsum, prof, fcs)."""
    fd = _legacy_module()
    ns = dict(np=np, sparse=sparse, interp_rows=fd.interp_rows, Y=fd.Y, C_KMS=fd.C_KMS, LREF=fd.LREF, teff=teff,
              lam=lam, fn=fnorm, fc=fcont[:, :, 0], dT=dT, chunk=chunk)
    _legacy_exec("fw_disc.py", "edges = np.arange(np.floor(teff.min() / dT)",
                 "prof[:, j] += S @ f" if sums else "fcs[ok] /= cnt[ok, None]", ns)
    return ns


def _legacy_los_library(teff, lam, fnorm, fcont, block, nproc, dT=10.0):
    """The library by-product of the frozen fw_disc_los.py (which wrote the M424 library_dT10.npz), run from its own
    source lines: edges/bins/cnt (:63-66), stream() (:73-83, blocks of `block` models strided over `nproc` workers;
    pool.map -> map), the per-line division (:139-140) and the assembly (:147-155). The weight matrix holds only the
    library rows (each row of the original's sparse product is summed independently of the others).

    Returns dict(lib=the legacy dict (prof float32), prof64=lib_prof (float64, after the empty-bin fill), sums=(nb, 3, ny)
    bin sums of the profiles before the division)."""
    fd = _legacy_module()
    a = types.SimpleNamespace(block=block, nproc=nproc, dT=dT)
    ns = dict(np=np, fd=fd, a=a, teff=teff, N=teff.size, lam_all=lam, fn_all=fnorm, ny=fd.Y.size,
              fc_all=fcont[:, :, 0].astype(np.float64), pool=types.SimpleNamespace(map=map))
    _legacy_exec("fw_disc_los.py", "edges = np.arange(np.floor(teff.min() / a.dT)", "cnt = np.bincount(bins", ns)
    _legacy_exec("fw_disc_los.py", "def stream(worker):", "return A", ns)
    nb = ns["nb"]
    ns.update(lib_prof=np.zeros((nb, 3, fd.Y.size)), lib_fc=np.zeros((nb, 3)), r_lib=0, ok=ns["cnt"] > 0)
    sums = np.zeros((nb, 3, fd.Y.size))
    for j in range(3):
        M = sparse.csr_matrix((np.ones(teff.size), (ns["bins"], np.arange(teff.size))), shape=(nb, teff.size))
        ns.update(JLINE=j, MC=M.tocsc(), j=j)
        _legacy_exec("fw_disc_los.py", "A = sum(pool.map(stream", "A = sum(pool.map(stream", ns)
        sums[:, j] = ns["A"][:nb]
        _legacy_exec("fw_disc_los.py", "lib_prof[ok, j] = A[r_lib", "lib_fc[:, j] = np.bincount", ns)
    _legacy_exec("fw_disc_los.py", "# library (same format", "count=cnt, prof=lib_prof", ns)
    return dict(lib=ns["lib"], prof64=ns["lib_prof"], sums=sums)


class _FakePool:
    """multiprocessing.Pool stand-in for the legacy scripts: `with Pool(n) as pool: pool.map(...)` -> builtin map."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return types.SimpleNamespace(map=map)

    def __exit__(self, *exc):
        return False


def _legacy_holdout(teff, lam, fnorm, fcont, edges, block, nproc, seed=11):
    """The hold-out libraries of the frozen fw_disc_holdout.py, run from its own source lines: bins and the random halves
    (:51-54), stream() (:59-68), shift_add() (:71-79) and the per-line loop (:82-121: weight matrix with the velocity
    groups of 8 lines of sight and the library rows of each half, the per-half bin means). MU and the shifts SH are
    random (they only add other rows to the weight matrix). Returns the namespace (libs, HALVES, bins, ...)."""
    fd = _legacy_module()
    N = teff.size
    rng = np.random.default_rng(99)
    a = types.SimpleNamespace(nproc=nproc, block=block, nmin=20, seed=seed)
    ns = dict(np=np, sparse=sparse, fd=fd, a=a, teff=teff, N=N, MU=rng.uniform(-1.0, 1.0, (8, N)),
              SH=rng.integers(-6, 7, (8, N)), lam_all=lam, fn_all=fnorm, fc_all=fcont[:, :, 0].astype(np.float64),
              Pool=_FakePool, log=lambda msg: None, edges=edges)
    _legacy_exec("fw_disc_holdout.py", "nb = edges.size - 1", "HALVES = {", ns)
    ns.update(ny=fd.Y.size, MC=None, JLINE=0)
    _legacy_exec("fw_disc_holdout.py", "def stream(worker):", "return A", ns)
    _legacy_exec("fw_disc_holdout.py", "def shift_add(", "return depth", ns)
    _legacy_exec("fw_disc_holdout.py", "exact = {X:", "del A, M, MC", ns)
    return ns


def _imu_path(*parts):
    p = os.path.join(IMU_ROOT, *parts)
    if not os.path.exists(p):
        pytest.skip("M424 intensity products not available: {}".format(p))
    return p


def _assert_bitwise(a, b, what=""):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype, (what, a.dtype, b.dtype)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    assert np.array_equal(a, b), "{}: max |diff| {}".format(what, np.max(np.abs(a.astype(float) - b.astype(float))))


def _assert_m424(a, ref, what=""):
    """Bitwise where numpy's np.log is the production's (AVX512_SKX SVML), else within 1 float32 ulp / 1e-15."""
    if HW_BITWISE:
        _assert_bitwise(a, ref, what)
        return
    a, ref = np.asarray(a), np.asarray(ref)
    assert a.dtype == ref.dtype and a.shape == ref.shape, (what, a.dtype, ref.dtype, a.shape, ref.shape)
    if a.dtype == np.float32:
        assert np.all(np.abs(a.astype(np.float64) - ref) <= np.spacing(np.abs(ref)).astype(np.float64)), what
    else:
        np.testing.assert_allclose(a, ref, rtol=0, atol=1e-15, err_msg=what)


def _assert_same_library(lib, ref, what="library", keys=LIB_KEYS):
    for k in keys:
        _assert_bitwise(lib[k], ref[k], "{} {}".format(what, k))


def _assert_same_nodes(nodes, ref, what="nodes"):
    for k in NODE_KEYS:
        _assert_bitwise(nodes[k], ref[k], "{} {}".format(what, k))


# ----------------------------------------------------------------------------------------------
# synthetic per-point runs
# ----------------------------------------------------------------------------------------------
def _band(nrow=161):
    """FASTWIND-like band of y [km/s]: denser in the core, spanning more than the +-2700 km/s grid."""
    u = np.linspace(-1.0, 1.0, nrow)
    return 3100.0 * np.sign(u) * np.abs(u) ** 1.6 + 15.0


def _synthetic_run(rng, N, teff_range=(35400.0, 36400.0), nrow=161, jitter=True, gaps=True, linear=False):
    """teff (N,), lam, fnorm (N, 3, nrow) float32 (0.01 A rounded wavelengths -> duplicates), fcont (N, 3, nrow), which
    varies along the frequency axis (column 0 is the library's F_c). linear=True: one wavelength grid for all models and
    F linear in T_eff' (exact interpolation test)."""
    teff = rng.uniform(*teff_range, N)
    if gaps:                                                    # sparse tails and an empty stretch
        teff[: N // 50] = rng.uniform(teff_range[0] - 300.0, teff_range[0] - 30.0, N // 50)
        teff[N // 50: N // 25] = rng.uniform(teff_range[1] + 60.0, teff_range[1] + 400.0, N // 25 - N // 50)
        teff[(teff > teff_range[0] + 200) & (teff < teff_range[0] + 260)] += 80.0
    yb = _band(nrow)
    lam = np.zeros((N, 3, nrow), np.float32)
    fnorm = np.zeros((N, 3, nrow), np.float32)
    for j in range(3):
        sh = rng.uniform(-30.0, 30.0, (N, 1)) if jitter else np.zeros((N, 1))
        lj = np.round(LREF[j] * np.exp((yb[None, :] + sh) / C_KMS), 2)
        lam[:, j] = lj
        yj = C_KMS * np.log(lam[:, j].astype(np.float64) / LREF[j])
        x = (teff[:, None] - 36000.0) / 1000.0
        if linear:
            depth = (0.3 + 0.05 * j) + 0.1 * x
        else:
            depth = (0.3 + 0.05 * j) + 0.1 * x + 0.02 * np.sin(7 * x)
        fnorm[:, j] = 1.0 - depth * np.exp(-0.5 * (yj / (60.0 + 40.0 * j)) ** 2) \
            - 0.05 * np.exp(-0.5 * ((yj - 200.0) / 300.0) ** 2)
    fcont = (1.0 + 1e-4 * (teff[:, None, None] - 36000.0) + np.zeros((1, 3, nrow))).astype(np.float32)
    fcont *= np.array([1.0, 0.9, 0.7], np.float32)[None, :, None]
    fcont *= (1.0 + 1e-3 * np.arange(nrow)).astype(np.float32)[None, None, :]        # factor 1 at the first point
    return teff, lam, fnorm, fcont


def _write_profiles(path, teff, lam, fnorm, fcont):
    np.savez(path, teff=teff, lam=lam, fnorm=fnorm, fcont=fcont)
    return path


@pytest.fixture(scope="module")
def synrun(tmp_path_factory):
    """A synthetic run of 1500 models as profiles.npz (legacy layout) + the legacy fw_disc.library of it."""
    fd = _legacy_module()
    d = tmp_path_factory.mktemp("synrun")
    rng = np.random.default_rng(42)
    teff, lam, fnorm, fcont = _synthetic_run(rng, 1500)
    path = _write_profiles(str(d / "profiles.npz"), teff, lam, fnorm, fcont)
    legacy = fd.library(run=str(d), dT=10.0, chunk=400)      # writes d/library_dT10.npz (its cache) only
    return dict(dir=str(d), path=path, teff=teff, lam=lam, fnorm=fnorm, fcont=fcont, legacy=legacy)


def _build(sr, **kw):
    """FluxLibrary.build on the synthetic run (arrays in memory unless lam/fnorm are given)."""
    lam = kw.pop("lam", sr["lam"])
    fnorm = kw.pop("fnorm", sr["fnorm"])
    return lb.FluxLibrary.build(sr["teff"], lam, fnorm, sr["fcont"][:, :, 0], VelocityGrid().y, LREF, **kw)


# ----------------------------------------------------------------------------------------------
# helpers: interpolation offsets, edges and bins
# ----------------------------------------------------------------------------------------------
def test_interp_rows_row0():
    """spectral.interp_rows(row0=...): sub-blocks with their positions' offsets (int) or selected rows (index array)
    give the whole block's rows bit for bit; without the offsets the rows differ in the last bits."""
    rng = np.random.default_rng(1)
    teff, lam, fnorm, _ = _synthetic_run(rng, 300, gaps=False)
    y = VelocityGrid().y
    L = y_of_lam(lam[:, 0], LREF[0])
    F = fnorm[:, 0]
    full = interp_rows(L, F, y)
    _assert_bitwise(interp_rows(L, F, y, row0=0), full, "row0=0")
    _assert_bitwise(interp_rows(L, F, y, row0=np.arange(300)), full, "row0=arange")
    parts = [interp_rows(L[r0:r0 + 70], F[r0:r0 + 70], y, row0=r0) for r0 in range(0, 300, 70)]
    _assert_bitwise(np.concatenate(parts), full, "sub-blocks")
    sel = np.sort(rng.choice(300, 120, replace=False))
    _assert_bitwise(interp_rows(L[sel], F[sel], y, row0=sel), full[sel], "selected rows")
    plain = np.concatenate([interp_rows(L[r0:r0 + 70], F[r0:r0 + 70], y) for r0 in range(0, 300, 70)])
    assert not np.array_equal(plain, full) and np.max(np.abs(plain - full)) < 1e-9
    for bad in (sel[::-1], sel[:-1], sel.astype(float), np.zeros(120, int)):
        with pytest.raises(ValueError, match="row0"):
            interp_rows(L[sel], F[sel], y, row0=bad)


def test_edges_and_bins():
    teff = np.array([35402.1, 35410.0, 35419.99, 35480.0, 35480.0])
    e = lb.teff_edges(teff, 10.0)
    _assert_bitwise(e, np.arange(np.floor(teff.min() / 10.0) * 10.0, teff.max() + 10.0, 10.0), "edges")
    assert e[0] == 35400.0 and e[-1] >= teff.max()
    b = lb.teff_bins(teff, e)
    assert e.size == 9 and list(b) == [0, 1, 1, 7, 7]                   # max on the last edge: last bin (clip)
    # beyond the edges: end bins
    assert list(lb.teff_bins([35000.0, 40000.0], e)) == [0, e.size - 2]
    with pytest.raises(ValueError):
        lb.teff_edges([], 10.0)


# ----------------------------------------------------------------------------------------------
# FluxLibrary.build: legacy equivalence in float32 and float64, parallel, memory paths
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rows", [None, 400, 1000, 97, 1])
def test_build_matches_legacy_library(synrun, rows):
    """build(block = legacy chunk) == fw_disc.library bit for bit, for every rows (CSR or np.add.at path)."""
    lib = _build(synrun, dT=10.0, block=400, rows=rows)
    _assert_same_library(lib, synrun["legacy"], "rows={}".format(rows))
    assert (lib.count == 0).any() and lib.count.sum() == synrun["teff"].size


@pytest.mark.parametrize("rows", [None, 97, 1])
def test_build_float64_matches_legacy_library(synrun, rows):
    """The float64 bin sums (_accumulate) and bin means (prof_dtype=float64) equal those of fw_disc.library (its own
    source lines, chunk 400) bit for bit: summation order and row offsets are the legacy ones."""
    t, lam, fn, fc = synrun["teff"], synrun["lam"], synrun["fnorm"], synrun["fcont"]
    ref = _legacy_library64(t, lam, fn, fc, chunk=400, sums=True)
    y = VelocityGrid().y
    acc = lb._accumulate(t, lam, fn, fc[:, :, 0], y, LREF, ref["edges"], block=400, stride=1,
                         rows=400 if rows is None else rows)
    _assert_bitwise(acc["sums"], ref["prof"], "float64 bin sums")
    for k, rk in (("cnt", "cnt"), ("tsum", "tsum"), ("fcs", "fcs")):
        _assert_bitwise(acc[k], ref[rk], k)
    ref = _legacy_library64(t, lam, fn, fc, chunk=400)
    lib = _build(synrun, block=400, rows=rows, prof_dtype=np.float64, fill_empty=False)
    assert lib.prof.dtype == np.float64 and lib.params["prof_dtype"] == "float64"
    _assert_bitwise(lib.prof, ref["prof"], "float64 bin means")
    _assert_bitwise(lib.fc, ref["fcs"], "fc")
    # float32 default == the cast of the float64 means (filled bins)
    lib32 = _build(synrun, block=400, rows=rows)
    ok = lib32.filled
    _assert_bitwise(lib32.prof[ok], ref["prof"][ok].astype(np.float32), "float32 cast")


@pytest.mark.parametrize("nproc,start_method", [(1, None), (2, "fork"), (3, "spawn")])
def test_build_matches_legacy_los_library(synrun, nproc, start_method):
    """build(block, stride = P) == the library of fw_disc_los.py --block B --nproc P bit for bit (the arithmetic of the
    M424 library_dT10.npz), in float32 and in float64 (bin sums and means), whatever our own nproc."""
    t, lam, fn, fc = synrun["teff"], synrun["lam"], synrun["fnorm"], synrun["fcont"]
    y = VelocityGrid().y
    for block, stride, rows in ((200, 3, 70), (250, 7, 1000)):           # 7 partial sums > 6 blocks: one empty
        ref = _legacy_los_library(t, lam, fn, fc, block=block, nproc=stride)
        lib = _build(synrun, block=block, stride=stride, rows=rows, nproc=nproc, start_method=start_method)
        _assert_same_library(lib, ref["lib"], "stride {}".format(stride))
        assert lib.params["stride"] == stride
        lib64 = _build(synrun, block=block, stride=stride, rows=rows, nproc=nproc, start_method=start_method,
                       prof_dtype=np.float64)
        _assert_bitwise(lib64.prof, ref["prof64"], "float64 prof, stride {}".format(stride))
        acc = lb._accumulate(t, lam, fn, fc[:, :, 0], y, LREF, lib.edges, block=block, stride=stride, rows=rows,
                             nproc=nproc, start_method=start_method)
        _assert_bitwise(acc["sums"], ref["sums"], "float64 bin sums, stride {}".format(stride))


def test_float64_sees_summation_order(synrun):
    """The synthetic run is sensitive to the float64 summation order (so the float64 tests can tell strides, block sizes
    and block orders apart), while the float32 prof hides it."""
    a = _build(synrun, block=200, stride=1, prof_dtype=np.float64)
    b = _build(synrun, block=200, stride=3, prof_dtype=np.float64)
    c = _build(synrun, block=333, stride=1, prof_dtype=np.float64)
    assert not np.array_equal(a.prof, b.prof) and not np.array_equal(a.prof, c.prof)
    assert np.abs(a.prof - b.prof).max() < 1e-14
    # the sums of the blocks added in reverse order
    y = VelocityGrid().y
    s = lb._BlockSums(synrun["lam"], synrun["fnorm"], lb.teff_bins(synrun["teff"], a.edges), a.nb, y, LREF, 200, 200)
    rev = np.zeros((a.nb, 3, y.size))
    for i0 in range(1400, -1, -200):
        ub, P = s(i0)
        rev[ub] += P
    acc = lb._accumulate(synrun["teff"], synrun["lam"], synrun["fnorm"], synrun["fcont"][:, :, 0], y, LREF, a.edges,
                         block=200)
    assert not np.array_equal(rev, acc["sums"])


def test_build_float64_invariant_to_nproc_rows(synrun):
    """nproc (fork, spawn; arrays and memory maps) and rows never change the float64 bin means."""
    ref = _build(synrun, block=200, stride=3, rows=None, prof_dtype=np.float64)
    lam = sio.npz_member_memmap(synrun["path"], "lam")
    fnorm = sio.npz_member_memmap(synrun["path"], "fnorm")
    for kw in (dict(rows=97), dict(rows=1), dict(rows=70, nproc=2, start_method="fork"),
               dict(rows=70, nproc=3, start_method="spawn"),
               dict(rows=50, nproc=2, start_method="fork", lam=lam, fnorm=fnorm),
               dict(rows=1000, nproc=3, start_method="spawn", lam=lam, fnorm=fnorm, maxtasksperchild=1)):
        lib = _build(synrun, block=200, stride=3, prof_dtype=np.float64, **kw)
        _assert_bitwise(lib.prof, ref.prof, "float64 {}".format({k: v for k, v in kw.items() if k not in ("lam", "fnorm")}))


def test_build_block_changes_only_rounding(synrun):
    """Another block size regroups the float64 partial sums: <= 1 float32 ulp in prof, fc etc. identical."""
    a = _build(synrun, block=400)
    b = _build(synrun, block=333)
    for k in ("edges", "tmean", "count", "fc"):
        _assert_bitwise(a[k], b[k], k)
    ulp = np.spacing(np.abs(a.prof))
    assert np.all(np.abs(a.prof - b.prof) <= ulp)


@pytest.mark.parametrize("start_method", ["fork", "spawn"])
@pytest.mark.parametrize("memmap", [True, False])
def test_build_parallel_equals_serial(synrun, start_method, memmap):
    """nproc > 1 adds the per-block sums in block order: bitwise equal to nproc = 1 (fork: inputs inherited; spawn: whole
    memory maps reopened by the workers, other arrays pickled), in float32 and float64."""
    if memmap:
        lam = sio.npz_member_memmap(synrun["path"], "lam")
        fnorm = sio.npz_member_memmap(synrun["path"], "fnorm")
        assert lb._share(lam, start_method)[0] == ("memmap" if start_method == "spawn" else "array")
    else:
        lam, fnorm = synrun["lam"], synrun["fnorm"]
        assert lb._share(lam, start_method)[0] == "array"
    calls = []
    lib = _build(synrun, lam=lam, fnorm=fnorm, block=400, rows=150, nproc=2, start_method=start_method,
                 maxtasksperchild=1, progress=lambda k, n: calls.append((k, n)))
    _assert_same_library(lib, synrun["legacy"], "nproc=2 {}".format(start_method))
    assert calls == [(k, 4) for k in range(1, 5)]
    ref64 = _build(synrun, block=400, prof_dtype=np.float64)
    lib64 = _build(synrun, lam=lam, fnorm=fnorm, block=400, rows=150, nproc=2, start_method=start_method,
                   prof_dtype=np.float64)
    _assert_bitwise(lib64.prof, ref64.prof, "float64 nproc=2 {}".format(start_method))


def test_share_and_unshare(synrun, tmp_path):
    """fork: the object itself; spawn: a whole memory map by file name + identity (a sliced one is pickled); a reopened
    file that was replaced since raises; a vanished file cannot be shared."""
    p = str(tmp_path / "profiles.npz")
    shutil.copy(synrun["path"], p)
    mm = sio.npz_member_memmap(p, "lam")
    kind, obj = lb._share(mm, "fork")
    assert kind == "array" and obj is mm
    rec = lb._share(mm, "spawn")
    kind, filename, offset, dtype, shape, order, ident = rec
    assert kind == "memmap" and shape == mm.shape and order == "C" and ident == lb._file_identity(p)
    _assert_bitwise(np.asarray(lb._unshare(rec)), np.asarray(mm), "reopened")
    for view in (mm[10:20], mm[:, 1], mm[::2]):
        assert lb._share(view, "spawn")[0] == "array"
    assert lb._share(np.zeros(10), "spawn")[0] == "array"
    q = str(tmp_path / "copy.npz")
    shutil.copy(p, q)
    os.replace(q, p)                                            # same content, another file
    with pytest.raises(RuntimeError, match="changed since the build started"):
        lb._unshare(rec)
    os.remove(p)
    with pytest.raises(ValueError, match="cannot be reopened"):
        lb._share(mm, "spawn")


class _Hung(Exception):
    pass


@contextlib.contextmanager
def _deadline(seconds):
    """Fail (instead of hanging) when the block takes longer than `seconds` (SIGALRM; the main thread's waits on the pool
    are interruptible)."""
    def handler(signum, frame):
        raise _Hung("no result after {} s: the pool watchdog did not fire".format(seconds))
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def test_build_worker_killed_raises_pool_stalled(synrun, monkeypatch):
    """A worker killed by a signal loses its block: the watchdog raises PoolStalled with the blocks not delivered
    (fork: the patched block function is inherited)."""
    orig = lb._BlockSums.__call__

    def dying(self, i0):
        if i0 == 400:
            os.kill(os.getpid(), signal.SIGKILL)
        return orig(self, i0)

    monkeypatch.setattr(lb._BlockSums, "__call__", dying)
    t0 = time.time()
    with _deadline(120), pytest.raises(par.PoolStalled) as ei:
        _build(synrun, block=400, nproc=2, start_method="fork", timeout=5)
    assert ei.value.missing == [400, 800, 1200] and ei.value.timeout == 5
    assert time.time() - t0 < 60


def test_build_worker_init_failure_raises(synrun, monkeypatch):
    """An initializer that raises (here: the reopened memory map is not the file the parent recorded) gives
    WorkerInitError in the parent instead of endless worker restarts (spawn)."""
    lam = sio.npz_member_memmap(synrun["path"], "lam")
    fnorm = sio.npz_member_memmap(synrun["path"], "fnorm")
    monkeypatch.setattr(lb, "_file_identity", lambda path: (0, 0, 0, 0))        # parent only; spawn workers are fresh
    t0 = time.time()
    with _deadline(240), pytest.raises(par.WorkerInitError, match="changed since the build started"):
        _build(synrun, lam=lam, fnorm=fnorm, block=400, nproc=2, start_method="spawn", timeout=300)
    assert time.time() - t0 < 120


def test_from_profiles_npz_and_roundtrip(synrun, tmp_path):
    y = VelocityGrid()
    lib = lb.FluxLibrary.from_profiles_npz(pathlib.Path(synrun["path"]), y, LREF, block=400, rows=200)
    _assert_same_library(lib, synrun["legacy"], "from_profiles_npz")
    p = lib.save(tmp_path / "lib.npz")
    back = lb.FluxLibrary.load(p)
    _assert_same_library(back, lib, "save/load")
    meta = sio.read_meta(p)
    assert meta["kind"] == "synspec.flux_library" and meta["params"]["block"] == 400
    assert meta["inputs"]["profiles"]["path"] == os.path.abspath(synrun["path"])
    assert "source_meta" not in meta and meta["cpu"]["machine"]
    with np.load(p) as z:
        assert sorted(z.files) == sorted(LIB_KEYS + ("_meta",))
        assert z["dT"].shape == () and z["prof"].dtype == np.float32
    # load -> save keeps the provenance: same params, the source file and its '_meta'
    assert back.meta == meta and back.params == meta["params"] and back.inputs == {"source": str(p)}
    q = back.save(str(tmp_path / "lib2.npz"))
    meta2 = sio.read_meta(q)
    assert meta2["params"] == meta["params"] and meta2["source_meta"] == meta
    assert meta2["inputs"]["source"]["path"] == os.path.abspath(str(p))
    _assert_same_library(lb.FluxLibrary.load(q), lib, "load/save/load")
    legacy = lb.FluxLibrary.load(os.path.join(synrun["dir"], "library_dT10.npz"))      # no '_meta'
    assert legacy.meta == {} and legacy.params == {}
    _assert_same_library(legacy, lib, "legacy file")
    assert "source_meta" not in sio.read_meta(legacy.save(str(tmp_path / "lib3.npz")))


def test_from_profiles_npz_detects_replaced_file(synrun, tmp_path):
    """profiles.npz replaced while the library is built from it (e.g. a merge re-run): RuntimeError."""
    p = str(tmp_path / "profiles.npz")
    shutil.copy(synrun["path"], p)

    def replace(k, n):
        if k == 1:
            q = str(tmp_path / "new.npz")
            shutil.copy(p, q)
            os.replace(q, p)

    with pytest.raises(RuntimeError, match="replaced or rewritten"):
        lb.FluxLibrary.from_profiles_npz(p, VelocityGrid(), LREF, block=400, progress=replace)


def test_build_input_checks(synrun):
    y = VelocityGrid().y
    t, lam, fn, fc = synrun["teff"], synrun["lam"], synrun["fnorm"], synrun["fcont"][:, :, 0]
    with pytest.raises(ValueError):
        lb.FluxLibrary.build(t[:-1], lam, fn, fc, y, LREF)
    with pytest.raises(ValueError):
        lb.FluxLibrary.build(t, lam, fn[:, :2], fc, y, LREF)
    with pytest.raises(ValueError):
        lb.FluxLibrary.build(t, lam, fn, fc[:, :2], y, LREF)
    with pytest.raises(ValueError):
        lb.FluxLibrary.build(t, lam, fn, fc, y, LREF, block=0)
    with pytest.raises(ValueError):
        lb.FluxLibrary.build(t, lam, fn, fc, y, LREF, stride=0)
    with pytest.raises(ValueError):
        lb.FluxLibrary.build(t, lam, fn, fc, y, LREF, edges=[1.0])
    with pytest.raises(ValueError, match="spaced"):
        lb.FluxLibrary.build(t, lam, fn, fc, y, LREF, dT=10.0, edges=np.arange(35000.0, 37000.0, 20.0))
    with pytest.raises(ValueError, match="prof_dtype"):
        lb.FluxLibrary.build(t, lam, fn, fc, y, LREF, prof_dtype=np.float16)
    with pytest.raises(ValueError, match="select"):
        lb.FluxLibrary.build(t, lam, fn, fc, y, LREF, select=np.zeros(t.size, bool))
    with pytest.raises(ValueError, match="select"):
        lb.FluxLibrary.build(t, lam, fn, fc, y, LREF, select=np.ones(t.size - 1, bool))
    with pytest.raises(ValueError, match="repeat"):
        lb.FluxLibrary.build(t, lam, fn, fc, y, LREF, select=[1, 2, 2])
    with pytest.raises(ValueError, match="0 .. N - 1"):
        lb.FluxLibrary.build(t, lam, fn, fc, y, LREF, select=[1, t.size])
    with pytest.raises(ValueError, match="lref"):
        lb.FluxLibrary.build(t, lam, fn, fc, y, ["HEI4026", "HEII4200", "HEI4922"])
    # lists and tuples are accepted (same numbers)
    n = 30
    ref = lb.FluxLibrary.build(t[:n], lam[:n], fn[:n], fc[:n], y, LREF, block=16, rows=5)
    lst = lb.FluxLibrary.build(list(t[:n]), lam[:n].tolist(), fn[:n].tolist(), fc[:n].tolist(), list(y), tuple(LREF),
                               block=16, rows=5)
    # tolist() of float32 gives float64 values (exact), so the profiles are the same numbers
    _assert_same_library(lst, ref, "lists")


def test_empty_bin_fill_and_tmean():
    """Empty bins: count 0, tmean = centre, profile and F_c of the nearest filled bin (lower one on a tie); with
    fill_empty=False tmean, prof and fc stay 0 there (fw_disc_holdout.py)."""
    rng = np.random.default_rng(3)
    teff, lam, fnorm, fcont = _synthetic_run(rng, 40, gaps=False)
    teff[:] = np.repeat([35405.0, 35435.0, 35455.0, 35475.0], 10) + rng.uniform(-4, 4, 40)
    lib = lb.FluxLibrary.build(teff, lam, fnorm, fcont[:, :, 0], VelocityGrid().y, LREF, block=16, rows=5)
    assert list(lib.count) == [10, 0, 0, 10, 0, 10, 0, 10]
    filled = lib.filled
    _assert_bitwise(lib.tmean[~filled], lib.centres[~filled], "tmean of empty bins")
    for i, k in ((1, 0), (2, 3), (4, 3), (6, 5)):              # bin 4: tie between 3 and 5 -> 3
        _assert_bitwise(lib.prof[i], lib.prof[k], "prof {}".format(i))
        _assert_bitwise(lib.fc[i], lib.fc[k], "fc {}".format(i))
    for i in np.where(filled)[0]:
        m = lib.bin_index(teff) == i
        assert lib.tmean[i] == pytest.approx(teff[m].mean(), rel=1e-15)
        np.testing.assert_allclose(lib.fc[i], fcont[m, :, 0].astype(np.float64).mean(axis=0), rtol=1e-14)
    raw = lb.FluxLibrary.build(teff, lam, fnorm, fcont[:, :, 0], VelocityGrid().y, LREF, block=16, rows=5,
                               fill_empty=False)
    assert np.all(raw.tmean[~filled] == 0) and np.all(raw.prof[~filled] == 0) and np.all(raw.fc[~filled] == 0)
    for k in ("tmean", "prof", "fc"):
        _assert_bitwise(raw[k][filled], lib[k][filled], k)
    assert lb.lib_nodes(raw, nmin=5).nn == lb.lib_nodes(lib, nmin=5).nn


@pytest.mark.parametrize("rows,nproc,start_method", [(None, 1, None), (37, 1, None), (70, 2, "fork"),
                                                     (70, 2, "spawn")])
def test_select_matches_legacy_holdout(synrun, rows, nproc, start_method):
    """build(select=half, edges of the full library, block, stride = --nproc, prof_dtype=float64, fill_empty=False) ==
    the hold-out libraries of the frozen fw_disc_holdout.py bit for bit (global block positions and row offsets), and
    lib_nodes of them == fw_disc.lib_nodes of the legacy ones."""
    fd = _legacy_module()
    t, lam, fn, fc = synrun["teff"], synrun["lam"], synrun["fnorm"], synrun["fcont"]
    edges = synrun["legacy"]["edges"]
    ns = _legacy_holdout(t, lam, fn, fc, edges, block=200, nproc=3)
    for X, m in ns["HALVES"].items():
        ref = ns["libs"][X]
        sel = m if X == "A" else np.where(m)[0]                 # bool mask or indices
        lib = _build(synrun, block=200, stride=3, rows=rows, nproc=nproc, start_method=start_method, select=sel,
                     edges=edges, prof_dtype=np.float64, fill_empty=False)
        _assert_same_library(lib, ref, "half {}".format(X), keys=("edges", "tmean", "count", "prof", "fc"))
        assert lib.params["select"] and lib.params["n_selected"] == int(m.sum())
        _assert_same_nodes(lb.lib_nodes(lib, nmin=20), fd.lib_nodes(ref, nmin=20), "nodes half {}".format(X))


def test_select_keeps_offsets_and_default_edges(synrun):
    """select keeps the global row positions (unlike fancy-indexing the inputs, which changes the offsets and the
    float64 sums) and the edges of all models; the full selection is the full library."""
    m = synrun["teff"] > 35600.0
    full = synrun["legacy"]
    sub = _build(synrun, block=400, select=m)
    _assert_bitwise(sub.edges, full["edges"], "edges of all models")
    b = lb.teff_bins(synrun["teff"][m], full["edges"])
    _assert_bitwise(sub.count, np.bincount(b, minlength=sub.nb).astype(float), "count")
    assert sub.params["edges_given"] is False and sub.params["n_selected"] == int(m.sum())
    fancy = lb.FluxLibrary.build(synrun["teff"][m], synrun["lam"][m], synrun["fnorm"][m], synrun["fcont"][m, :, 0],
                                 VelocityGrid().y, LREF, block=400, edges=full["edges"])
    assert fancy.params["edges_given"] is True
    for k in ("edges", "tmean", "count", "fc"):
        _assert_bitwise(sub[k], fancy[k], k)
    assert np.all(np.abs(sub.prof - fancy.prof) <= np.spacing(np.abs(fancy.prof)))
    s64 = _build(synrun, block=400, select=m, prof_dtype=np.float64)
    f64 = lb.FluxLibrary.build(synrun["teff"][m], synrun["lam"][m], synrun["fnorm"][m], synrun["fcont"][m, :, 0],
                               VelocityGrid().y, LREF, block=400, edges=full["edges"], prof_dtype=np.float64)
    assert not np.array_equal(s64.prof, f64.prof)              # other offsets
    _assert_same_library(_build(synrun, block=400, select=np.ones(synrun["teff"].size, bool)), full, "select all")


# ----------------------------------------------------------------------------------------------
# nodes, interpolation, coverage
# ----------------------------------------------------------------------------------------------
def test_linear_profiles_interpolate_exactly():
    """Profiles linear in T_eff' (one wavelength grid): bin means, merged nodes and node_pairs reproduce the profile at
    any T_eff' inside the node range to float32 rounding; beyond it the clamped value is the end node's and the
    extrapolated one is exact again."""
    rng = np.random.default_rng(5)
    teff, lam, fnorm, fcont = _synthetic_run(rng, 3000, jitter=False, linear=True)
    y = VelocityGrid().y
    lib = lb.FluxLibrary.build(teff, lam, fnorm, fcont[:, :, 0], y, LREF, block=700, rows=250)
    nodes = lb.lib_nodes(lib, nmin=20)
    assert nodes.nn < int(lib.filled.sum())                     # the tails were merged

    def exact(T):
        lam1 = lam[:1]                                          # same wavelengths for all models
        x = (np.atleast_1d(T)[:, None] - 36000.0) / 1000.0
        out = np.zeros((x.shape[0], 3, y.size))
        for j in range(3):
            yj = C_KMS * np.log(lam1[:, j].astype(np.float64) / LREF[j])
            Fj = 1.0 - ((0.3 + 0.05 * j) + 0.1 * x) * np.exp(-0.5 * (yj / (60.0 + 40.0 * j)) ** 2) \
                - 0.05 * np.exp(-0.5 * ((yj - 200.0) / 300.0) ** 2)
            out[:, j] = interp_rows(np.repeat(yj, x.shape[0], 0), Fj, y)
        return out

    Tq = rng.uniform(nodes.t[0], nodes.t[-1], 50)
    k0, k1, a = nodes.pairs(Tq)
    assert np.all((a >= 0) & (a <= 1)) and np.all(k1 == k0 + 1)
    interp = (1.0 - a)[:, None, None] * nodes.prof[k0] + a[:, None, None] * nodes.prof[k1]
    err = np.abs(interp - exact(Tq)).max()
    signal_ = np.abs(exact(nodes.t[-1]) - exact(nodes.t[0])).max()
    assert err < 3e-7 and signal_ > 1e-2, (err, signal_)
    # F_c is linear in T_eff' too
    fci = (1.0 - a)[:, None] * nodes.fc[k0] + a[:, None] * nodes.fc[k1]
    np.testing.assert_allclose(fci, (1.0 + 1e-4 * (Tq[:, None] - 36000.0)) * np.array([1.0, 0.9, 0.7], np.float32),
                               rtol=2e-7)
    # beyond the range
    Tout = np.array([nodes.t[0] - 150.0, nodes.t[-1] + 150.0])
    k0, k1, a = nodes.pairs(Tout)
    assert list(a) == [0.0, 1.0] and list(k0) == [0, nodes.nn - 2]
    k0, k1, a = nodes.pairs(Tout, mode="extrapolate")
    assert a[0] < 0 and a[1] > 1
    ext = (1.0 - a)[:, None, None] * nodes.prof[k0] + a[:, None, None] * nodes.prof[k1]
    assert np.abs(ext - exact(Tout)).max() < 3e-6


def _toy_library(counts, ny=7, nl=2, seed=0):
    rng = np.random.default_rng(seed)
    nb = len(counts)
    edges = 35000.0 + 10.0 * np.arange(nb + 1)
    counts = np.asarray(counts, dtype=float)
    tmean = np.where(counts > 0, edges[:-1] + rng.uniform(1, 9, nb), 0.5 * (edges[:-1] + edges[1:]))
    prof = rng.uniform(0.5, 1.0, (nb, nl, ny)).astype(np.float32)
    fc = rng.uniform(1.0, 2.0, (nb, nl))
    return lb.FluxLibrary(edges, tmean, counts, prof, fc, 10.0)


def test_node_merging_keeps_count_weighted_means():
    counts = [1, 0, 3, 0, 0, 2, 30, 25, 21, 0, 4, 5, 2, 0, 1]
    lib = _toy_library(counts)
    nodes = lb.lib_nodes(lib, nmin=6)
    groups = [list(g) for g in nodes.groups]
    assert groups == [[0, 2, 5], [6], [7], [8], [10, 11, 12, 14]]       # leftover [12, 14] joined the last node
    assert nodes.count.sum() == sum(counts)
    assert np.all(np.diff(nodes.t) > 0)
    for i, g in enumerate(nodes.groups):
        w = lib.count[g]
        assert nodes.count[i] == w.sum()
        assert nodes.t[i] == pytest.approx(np.average(lib.tmean[g], weights=w), rel=1e-15)
        np.testing.assert_allclose(nodes.prof[i], np.average(lib.prof[g].astype(float), axis=0, weights=w), rtol=1e-14)
        np.testing.assert_allclose(nodes.fc[i], np.average(lib.fc[g], axis=0, weights=w), rtol=1e-14)
    # the overall model mean of T_eff' is kept
    assert np.sum(nodes.count * nodes.t) == pytest.approx(np.sum(lib.count * lib.tmean), rel=1e-14)
    # all bins below nmin: one node; empty library: error
    assert lb.lib_nodes(_toy_library([1, 2, 0, 1]), nmin=20).nn == 1
    with pytest.raises(ValueError):
        lb.lib_nodes(_toy_library([0, 0, 0]), nmin=1)


def test_lib_nodes_corr_and_smooth():
    lib = _toy_library([5] * 40, ny=9, nl=3, seed=2)
    corr = np.random.default_rng(9).normal(0, 1e-3, lib.prof.shape)
    a = lb.lib_nodes(lib, nmin=5)
    b = lb.lib_nodes(lib, nmin=5, corr=corr)
    np.testing.assert_allclose(b.prof - a.prof, corr, atol=1e-15)             # nmin = 5: one bin per node
    assert b.params["corr"] and not a.params["corr"]
    with pytest.raises(ValueError):
        lb.lib_nodes(lib, corr=corr[:, :2])
    # smoothing keeps a linear trend in T_eff' exactly (local-linear fit), and leaves t, count alone
    # (bin centres as T_eff' and values k / 1024 + j / 8: exactly linear and exact in float32)
    tc = lib.centres
    lin = (tc - 35000.0)[:, None, None] / 1024.0 + np.arange(3)[None, :, None] / 8.0 + np.zeros(lib.prof.shape)
    lib2 = lb.FluxLibrary(lib.edges, tc, lib.count, lin.astype(np.float32), lib.fc, 10.0)
    assert np.array_equal(lib2.prof, lin)
    s0, s1 = lb.lib_nodes(lib2, nmin=5), lb.lib_nodes(lib2, nmin=5, smooth=55.0)
    _assert_bitwise(s0.t, s1.t, "t")
    _assert_bitwise(s0.count, s1.count, "count")
    np.testing.assert_allclose(s1.prof, s0.prof, rtol=0, atol=1e-13)
    assert not np.array_equal(lb.lib_nodes(lib, nmin=5, smooth=55.0).prof, a.prof)      # it does smooth
    assert s1.params["smooth"] == 55.0


def test_node_pairs_and_coverage():
    tn = np.array([1.0, 2.0, 4.0, 8.0])
    k0, k1, a = lb.node_pairs(tn, np.array([0.0, 1.0, 1.5, 2.0, 3.0, 8.0, 9.0]))
    assert list(k0) == [0, 0, 0, 1, 1, 2, 2]
    np.testing.assert_array_equal(a, [0.0, 0.0, 0.5, 0.0, 0.5, 1.0, 1.0])
    k0, k1, a = lb.node_pairs(tn, np.array([0.0, 3.0, 12.0]), mode="extrapolate")
    np.testing.assert_array_equal(a, [-1.0, 0.5, 2.0])
    with pytest.raises(ValueError):
        lb.node_pairs(tn, 1.0, mode="nearest")
    with pytest.raises(ValueError):
        lb.node_pairs(tn[:1], 1.0)
    # scalar
    k0, k1, a = lb.node_pairs(tn, 5.0)
    assert (int(k0), int(k1), float(a)) == (2, 3, 0.25)
    # coverage
    teff = np.array([0.5, 1.0, 3.0, 8.0, 8.5, 9.0])
    mu = np.array([[0.5, -0.1, 0.2, 0.3, 0.1, 0.0],
                   [-0.5, 0.4, 0.2, 0.3, -0.1, 0.9]])
    n_lo, n_hi, wout = lb.coverage(tn, teff, mu)
    assert (n_lo, n_hi) == (1, 2)
    np.testing.assert_allclose(wout, [(0.5 + 0.1) / (0.5 + 0.2 + 0.3 + 0.1), 0.9 / (0.4 + 0.2 + 0.3 + 0.9)], rtol=1e-15)
    assert lb.coverage(tn, teff)[2] is None
    w1 = lb.coverage(tn, teff, mu[0])[2]
    assert isinstance(w1, float) and w1 == wout[0]
    with pytest.raises(ValueError):
        lb.coverage(tn, teff, mu[:, :3])


def test_coverage_and_pairs_float32_teff():
    """float32 T_eff' (the per-dump samples) next to a node is compared in float64, as the legacy driver
    (smp['teff'].astype(np.float64)); a float32 comparison would round the node to float32 and miss the point."""
    tn = np.array([35829.1226, 36000.0, 38891.0])
    teff = np.array([35829.12109375, 36500.0, 38891.0], np.float32)
    assert np.float32(tn[0]) == teff[0]                                 # the node rounds onto the point in float32
    mu = np.array([0.3, 0.4, 0.5])
    n_lo, n_hi, wout = lb.coverage(tn, teff, mu)
    assert (n_lo, n_hi) == (1, 0)
    assert wout == pytest.approx(0.3 / 1.2, rel=1e-15)
    assert lb.coverage(tn, teff.astype(np.float64), mu) == (n_lo, n_hi, wout)
    for x, y in zip(lb.node_pairs(tn, teff), lb.node_pairs(tn, teff.astype(np.float64))):
        _assert_bitwise(x, y, "node_pairs float32 == float64")


def test_lib_nodes_and_pairs_match_legacy_synthetic(synrun, monkeypatch):
    """lib_nodes / node_pairs == the frozen fw_disc.lib_nodes / node_pairs bit for bit (legacy lamfix via a patched
    lam_corrections returning the same correction)."""
    fd = _legacy_module()
    lib = lb.FluxLibrary.load(os.path.join(synrun["dir"], "library_dT10.npz"))
    corr = np.random.default_rng(7).normal(0.0, 1e-3, lib.prof.shape)
    monkeypatch.setattr(fd, "lam_corrections", lambda _lib: corr)
    for nmin in (1, 20, 50):
        for smooth in (0.0, 35.0, 335.0):
            for use_corr in (False, True):
                mine = lb.lib_nodes(lib, nmin=nmin, smooth=smooth, corr=corr if use_corr else None)
                ref = fd.lib_nodes(np.load(os.path.join(synrun["dir"], "library_dT10.npz")), nmin=nmin, smooth=smooth,
                                   lamfix=use_corr)
                _assert_same_nodes(mine, ref, "nmin={} smooth={} corr={}".format(nmin, smooth, use_corr))
    # the legacy code accepts a FluxLibrary (dict-like access)
    _assert_same_nodes(fd.lib_nodes(lib, nmin=20), lb.lib_nodes(lib, nmin=20), "FluxLibrary input")
    tq = np.random.default_rng(8).uniform(mine.t[0] - 50, mine.t[-1] + 50, 1000)
    for x, y in zip(lb.node_pairs(mine.t, tq), fd.node_pairs(mine.t, tq)):
        _assert_bitwise(x, y, "node_pairs")


# ----------------------------------------------------------------------------------------------
# wavelength rounding correction (synthetic OUT / OUT_IMU files)
# ----------------------------------------------------------------------------------------------
def _write_model(d, lines, lref, rng, nrow=161, shift=0.0, exact=False, nrow_imu=None):
    """OUT.<line>_VTV010 (lambda rounded to 0.01 A, + a trailing single value) and OUT_IMU.<line>_VTV010 (precise;
    nrow_imu rows, default nrow)."""
    os.makedirs(d, exist_ok=True)
    nrow_imu = nrow if nrow_imu is None else nrow_imu
    yb = _band(max(nrow, nrow_imu))
    for ln, l0 in zip(lines, lref):
        lp = l0 * np.exp((yb + shift + rng.uniform(-0.3, 0.3)) / C_KMS)
        lr = np.round(lp, 2) if not exact else lp
        ylp = C_KMS * np.log(lp / l0)
        f = 1.0 - 0.4 * np.exp(-0.5 * (ylp / 80.0) ** 2)
        with open(os.path.join(d, "OUT.{}_VTV010".format(ln)), "w") as fh:
            for k in range(nrow):
                lam_txt = "{:15.2f}".format(lr[k]) if not exact else "{:15.8f}".format(lr[k])
                fh.write("{:4d} {:11.5f} {} {:19.6E} {:11.5f} {:11.5f}\n".format(k + 1, 1.2 - 0.015 * k, lam_txt,
                                                                               7.4e-7, f[k], f[k]))
            fh.write("  -1.08387540430798\n")
        with open(os.path.join(d, "OUT_IMU.{}_VTV010".format(ln)), "w") as fh:
            fh.write("# rays NP-1, core rays NC =    3    1\n# p    0.0 0.5 1.0\n# K, lambda, I_cont(p_1..p_NP-1), "
                     "I_line(p_1..p_NP-1)\n")
            for k in range(nrow_imu):
                fh.write("{:5d} {:.8f} 1.0E-05 9.0E-06 1.0E-07 {:.6E} 8.0E-06 1.0E-07\n".format(k + 1, lp[k], f[k] * 1e-5))


@pytest.fixture(scope="module")
def synreps(tmp_path_factory, synrun):
    """Representative model directories for the synthetic library (layout runs/P<idx:06d>/P<idx:06d>) and an
    intensity-library-like .npz with edges and idx_rep."""
    d = tmp_path_factory.mktemp("synreps")
    lib = lb.FluxLibrary.load(os.path.join(synrun["dir"], "library_dT10.npz"))
    rng = np.random.default_rng(11)
    idx = np.zeros(lib.nb, np.int64)
    filled = np.where(lib.filled)[0]
    for b in range(lib.nb):
        idx[b] = 1000 + filled[np.argmin(np.abs(filled - b))]
    runs = d / "runs"
    for b in filled:
        _write_model(str(runs / "P{:06d}".format(idx[b]) / "P{:06d}".format(idx[b])), LINES, LREF, rng,
                     shift=rng.uniform(-20, 20))
    npz = str(d / "imu_library_dT10.npz")
    np.savez(npz, edges=lib.edges, idx_rep=idx)
    return dict(dir=str(d), runs=str(runs), npz=npz, idx=idx, lib=lib)


def test_wavelength_rounding_matches_legacy(synreps, tmp_path, monkeypatch):
    """== the frozen fw_disc.lam_corrections bit for bit (its cache redirected to tmp_path); the same from model
    indices, directories, pathlib paths or a LineSet."""
    fd = _legacy_module()
    monkeypatch.setattr(fd, "DISC_DUMPS", str(tmp_path / "disc"))
    lib = synreps["lib"]
    ref = fd.lam_corrections(lib.as_dict(), runs=synreps["runs"], imu_lib=synreps["npz"])
    y = VelocityGrid()
    mine = lb.wavelength_rounding_correction(lib, synreps["npz"], LINES, y, LREF, runs=synreps["runs"])
    _assert_bitwise(mine, ref, "corr")
    assert np.abs(mine).max() > 1e-4 and np.all(mine[~lib.filled] == 0)
    dirs = lb.representative_dirs(synreps["idx"], lib.nb, runs=synreps["runs"])
    _assert_bitwise(lb.wavelength_rounding_correction(lib, dirs, LINES, y, LREF), ref, "corr from dirs")
    _assert_bitwise(lb.wavelength_rounding_correction(lib, synreps["idx"], LINES, y, LREF, runs=synreps["runs"]), ref,
                    "corr from idx")
    _assert_bitwise(lb.wavelength_rounding_correction(lib, dirs, LineSet(LINES, LREF), y), ref, "corr from LineSet")
    _assert_bitwise(lb.wavelength_rounding_correction(lib, pathlib.Path(synreps["npz"]), LINES, y, LREF,
                                                      runs=pathlib.Path(synreps["runs"])), ref, "corr from Paths")
    assert lb.representative_dirs([pathlib.Path("a"), None], 2) == ["a", None]


def test_wavelength_rounding_zero_and_errors(synreps, tmp_path):
    lib = synreps["lib"]
    rng = np.random.default_rng(12)
    d = str(tmp_path / "exact")
    _write_model(d, LINES, LREF, rng, exact=True)
    dirs = [d if c > 0 else None for c in lib.count]
    corr = lb.wavelength_rounding_correction(lib, dirs, LINES, VelocityGrid(), LREF)
    assert np.abs(corr).max() < 1e-12                                  # precise == written wavelengths
    with pytest.raises(ValueError, match="tol"):
        lb.wavelength_rounding_correction(lib, synreps["npz"], LINES, VelocityGrid(), LREF, runs=synreps["runs"],
                                          tol=1e-4)
    with pytest.raises(ValueError, match="lref is required"):
        lb.wavelength_rounding_correction(lib, dirs, LINES, VelocityGrid())
    # OUT_IMU shorter (truncated by a crashed pformalsol) or longer than OUT
    for n_imu in (100, 170):
        dd = str(tmp_path / "imu{}".format(n_imu))
        _write_model(dd, LINES, LREF, rng, nrow_imu=n_imu)
        with pytest.raises(ValueError, match="different numbers of rows"):
            lb.wavelength_rounding_correction(lib, [dd if c > 0 else None for c in lib.count], LINES, VelocityGrid(),
                                              LREF)
    dirs[np.where(lib.filled)[0][0]] = None
    with pytest.raises(ValueError, match="no representative"):
        lb.wavelength_rounding_correction(lib, dirs, LINES, VelocityGrid(), LREF)
    with pytest.raises(ValueError, match="runs"):
        lb.representative_dirs(synreps["idx"], lib.nb)
    with pytest.raises(ValueError, match="one representative per bin"):
        lb.representative_dirs(synreps["idx"][:-1], lib.nb, runs=synreps["runs"])
    bad = str(tmp_path / "bad.npz")
    np.savez(bad, edges=lib.edges + 5.0, idx_rep=synreps["idx"])
    with pytest.raises(ValueError, match="bins"):
        lb.representative_dirs(bad, lib.nb, runs=synreps["runs"], edges=lib.edges)
    noedges = str(tmp_path / "noedges.npz")
    np.savez(noedges, idx_rep=synreps["idx"])
    with pytest.raises(ValueError, match="no 'edges'"):
        lb.representative_dirs(noedges, lib.nb, runs=synreps["runs"], edges=lib.edges)
    assert len(lb.representative_dirs(noedges, lib.nb, runs=synreps["runs"])) == lib.nb       # no check asked for
    with pytest.raises(ValueError, match="no 'edges'"):
        lb.wavelength_rounding_correction(lib, noedges, LINES, VelocityGrid(), LREF, runs=synreps["runs"])
    with pytest.raises(ValueError, match="lines"):
        lb.wavelength_rounding_correction(lib, synreps["npz"], LINES[:2], VelocityGrid(), LREF[:2], runs=synreps["runs"])


# ----------------------------------------------------------------------------------------------
# M424 regressions
# ----------------------------------------------------------------------------------------------
def _rss_gb():
    """Peak RSS [GB] of this process and of the largest finished child (file pages of memory maps included)."""
    return (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6)


@pytest.mark.m424
def test_m424_block_paths_equal():
    """On real profiles (blocks of 2000 models): the legacy CSR path (rows >= block) == the np.add.at path, and with
    select (a random half) == fw_disc_holdout.py's arithmetic (whole block interpolated, CSR over the half's columns)."""
    path = m424_path("run", "profiles.npz")
    lam, fnorm = sio.npz_member_memmap(path, "lam"), sio.npz_member_memmap(path, "fnorm")
    teff = np.array(sio.npz_member_memmap(path, "teff"))
    edges = lb.teff_edges(teff, 10.0)
    b = lb.teff_bins(teff, edges)
    nb = edges.size - 1
    y = VelocityGrid().y
    half = np.random.default_rng(11).random(teff.size) < 0.5
    for i0 in (0, 600000, 1234000):
        ub, csr = lb._BlockSums(lam, fnorm, b, nb, y, LREF, 2000, 2000)(i0)
        ub2, add = lb._BlockSums(lam, fnorm, b, nb, y, LREF, 2000, 300)(i0)
        _assert_bitwise(ub2, ub, "bins {}".format(i0))
        _assert_bitwise(add, csr, "block {}".format(i0))
        i1 = min(i0 + 2000, teff.size)
        m = half[i0:i1]
        S = sparse.csr_matrix((np.ones(int(m.sum())), (b[i0:i1][m], np.flatnonzero(m))), shape=(nb, i1 - i0))
        for rows in (2000, 300):
            us, P = lb._BlockSums(lam, fnorm, b, nb, y, LREF, 2000, rows, sel=half)(i0)
            for j in range(3):
                ref = S @ interp_rows(y_of_lam(np.asarray(lam[i0:i1, j]), LREF[j]), np.asarray(fnorm[i0:i1, j]), y)
                _assert_bitwise(P[:, j], ref[us], "select block {} rows {} line {}".format(i0, rows, j))
                assert np.all(ref[np.setdiff1d(np.arange(nb), us)] == 0)


@pytest.mark.m424
@pytest.mark.slow
@pytest.mark.parametrize("nproc,stride,start_method", [(1, 1, None), (4, 20, "spawn")])
def test_m424_build_equals_production_library(nproc, stride, start_method):
    """FluxLibrary.from_profiles_npz(profiles.npz) (default block 5000) == library_dT10.npz bit for bit; stride 20 =
    the production's own reduction (fw_disc_los.py, 20 workers). 10 min serial, 2.7 min with nproc 4 on the shared
    login node with a warm page cache (2026-10-01; 16 and 6 min at a load average of 76; peak RSS 5.3 GB of the main
    process, mostly file pages of the memory maps, 1.7 GB per worker)."""
    path = m424_path("run", "profiles.npz")
    ref = np.load(m424_path("run", "library_dT10.npz"))
    t0, c0 = time.time(), time.process_time()
    lib = lb.FluxLibrary.from_profiles_npz(path, VelocityGrid(), LREF, dT=10.0, nproc=nproc, stride=stride,
                                           start_method=start_method)
    wall, cpu = time.time() - t0, time.process_time() - c0
    rss, rss_child = _rss_gb()
    print("\nM424 build nproc={} stride={} ({}): {:.0f} s wall, {:.0f} s CPU (main), peak RSS {:.2f} GB (main), "
          "{:.2f} GB (largest worker)".format(nproc, stride, start_method, wall, cpu, rss, rss_child))
    _assert_same_library(lib, ref, "M424 nproc={}".format(nproc), keys=("edges", "tmean", "count", "fc", "dT"))
    _assert_m424(lib.prof, ref["prof"], "M424 prof nproc={}".format(nproc))
    assert lib.nb == 351 and int(lib.filled.sum()) == 303 and lib.count.sum() == 1236544


def _m424_library():
    return np.load(m424_path("run", "library_dT10.npz"))


@pytest.mark.m424
@pytest.mark.parametrize("smooth,lamfix", [(0.0, False), (335.0, False), (0.0, True)])
def test_m424_lib_nodes_match_legacy(monkeypatch, smooth, lamfix):
    """lib_nodes(library_dT10) == frozen fw_disc.lib_nodes bit for bit: 245 nodes, 35829-38891 K."""
    fd = _legacy_module()
    corr = None
    if lamfix:
        corr = np.load(m424_path("disc", "lamfix_dT10.npz"))["corr"]
        monkeypatch.setattr(fd, "lam_corrections", lambda _lib: corr)        # never touch the legacy cache logic
    lib = lb.FluxLibrary.load(m424_path("run", "library_dT10.npz"))
    mine = lb.lib_nodes(lib, nmin=20, smooth=smooth, corr=corr)
    ref = fd.lib_nodes(_m424_library(), nmin=20, smooth=smooth, lamfix=lamfix)
    _assert_same_nodes(mine, ref, "smooth={} lamfix={}".format(smooth, lamfix))
    assert mine.nn == 245 and round(mine.t[0]) == 35829 and round(mine.t[-1]) == 38891


@pytest.mark.m424
@pytest.mark.parametrize("name,smooth,lamfix", [("flux", 0.0, False), ("flux_sm335", 335.0, False),
                                                ("flux_lamfix", 0.0, True)])
def test_m424_coverage_matches_dump_outputs(name, smooth, lamfix):
    """node range, n_lo, n_hi (bit for bit) and wout (bit for bit with the production's np.sin/np.cos SIMD paths) of
    fw_disc_dumps.py (dump 3200 and 4800); the per-dump samples' float32 teff is passed as stored."""
    fd = _legacy_module()
    lib = lb.FluxLibrary.load(m424_path("run", "library_dT10.npz"))
    corr = np.load(m424_path("disc", "lamfix_dT10.npz"))["corr"] if lamfix else None
    nodes = lb.lib_nodes(lib, nmin=20, smooth=smooth, corr=corr)
    pts = np.load(m424_path("run", "points.npz"))
    rhat, _, _ = fd.unit_vectors(pts["theta"], pts["phi"])
    MU = np.ascontiguousarray((rhat @ fd.los8().T).T)                     # as fw_disc_dumps.py:72-74
    for d in (3200, 4800):
        out = np.load(m424_path("disc", name, "d{:04d}.npz".format(d)))
        smp = np.load(os.path.join(os.path.dirname(m424_path("disc")), "samples_r4050_N1236544",
                                   "d{:04d}.npz".format(d)))
        assert smp["teff"].dtype == np.float32
        _assert_bitwise(np.array([nodes.t[0], nodes.t[-1]]), out["node_range"], "node_range")
        n_lo, n_hi, wout = nodes.coverage(smp["teff"], MU)
        assert (n_lo, n_hi) == (int(out["n_lo"]), int(out["n_hi"]))
        _assert_m424(wout, out["wout"], "wout d{}".format(d))


@pytest.mark.m424
def test_m424_wavelength_rounding_equals_lamfix():
    """wavelength_rounding_correction(library_dT10, intensity library's idx_rep, runs) == lamfix_dT10.npz 'corr' bit for
    bit (909 pairs of OUT / OUT_IMU files, ~5 s; bitwise needs the production's np.log, see _assert_m424)."""
    ref = np.load(m424_path("disc", "lamfix_dT10.npz"))["corr"]
    lib = lb.FluxLibrary.load(m424_path("run", "library_dT10.npz"))
    t0 = time.time()
    corr = lb.wavelength_rounding_correction(lib, _imu_path("imu_library_dT10.npz"), LINES, VelocityGrid(), LREF,
                                             runs=_imu_path("runs"))
    print("\nM424 wavelength rounding correction: {:.0f} s".format(time.time() - t0))
    _assert_m424(corr, ref, "lamfix corr")
