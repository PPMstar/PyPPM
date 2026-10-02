"""Tests of ppmpy.synspec.moms: moms block reading, grid, Jacobian, sphere sampling (both backends), products.

Oracles: ppmpy's own MomsData / MomsDataSet (read_moms_cube, _get_cgrid, _get_jacobian, get_spherical_components,
get_spherical_interpolation) on synthetic dumps in the exact block layout, the frozen legacy scripts
fw_sphere_extract.py and sphere_sample.py run as whole scripts on those dumps (a stand-in figstyle module in tmp,
see tests/synspec/legacy/README.txt), and the stored M424 products (slow): points.npz / points.txt / meta.json of
the dump-3200 run and the per-dump samples dNNNN.npz of the dump subset 3200, 3334, 4000, 4169, 4391, 4800 + 10
dumps drawn with default_rng(5) (the full 1601-dump reproduction is not run here).
"""
import inspect
import itertools
import json
import os
import pickle
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import scipy.interpolate

import conftest
from conftest import ROOT
from ppmpy.synspec import moms as mm

LEGACY = os.environ.get("PPMPY_SYNSPEC_M424_PROJECT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy"))
VARS = ["xc", "ux", "uy", "uz", "slot4_unknown", "dUr", "|w|", "T9", "rho", "dT9"]

M424_RUN_DIR = os.environ.get("PPMPY_SYNSPEC_M424_PPMRUN",
                              "/scratch/fherwig/PPMruns/PPMruns_fh/M424a-fullstar-100xhtdiff-1792")
M424_SAMPLES = os.environ.get("PPMPY_SYNSPEC_M424_SAMPLES", "/scratch/ppathak/fastwind_sphere/samples_r4050_N1236544")
M424_NPOINTS, M424_RADIUS, M424_TEFF0 = 1236544, 4050.0, 38230.0


SUBSET_FIXED = (3200, 3334, 4000, 4169, 4391, 4800)


def subset_dumps():
    """The verification subset: the 6 fixed dumps + the literal draw
    ``np.random.default_rng(5).choice(np.arange(3201, 4800), 10, replace=False)`` (= 3237, 3287, 3657, 3948, 4022,
    4206, 4267, 4481, 4488, 4766; none of them fixed)."""
    drawn = np.random.default_rng(5).choice(np.arange(3201, 4800), 10, replace=False)
    return sorted(set(SUBSET_FIXED) | {int(d) for d in drawn})


def test_subset_dumps():
    assert subset_dumps() == sorted([3200, 3334, 4000, 4169, 4391, 4800, 3237, 3287, 3657, 3948, 4022, 4206, 4267,
                                     4481, 4488, 4766])


def _ppm():
    try:
        from ppmpy import ppm
    except Exception as e:                    # pragma: no cover - ppmpy.ppm needs its own dependencies
        pytest.skip("ppmpy.ppm not importable: {}".format(e))
    return ppm


def _legacy_path(name):
    p = os.path.join(LEGACY, name)
    if not os.path.exists(p):
        pytest.skip("legacy file not available: {}".format(p))
    return p


# ----------------------------------------------------------------------------------------------
# synthetic dumps in the PPMstar block layout
# ----------------------------------------------------------------------------------------------
TOY = dict(nb=2, nblock=24, ghost=2, nslots=10, dx=0.3, run_id="toy", A=0.02, Om=0.01, T0=1.0e-3)


def _toy_fields(n, dx, dump, seed=0):
    """(10, n, n, n) float32 fields [z, y, x]: xc, linear T9, radial outflow + rigid rotation, noise elsewhere."""
    c = 4.0 * dx
    g = c * np.arange(n) - (c * (n / 2.) - c / 2.)
    Z, Y, X = np.meshgrid(g, g, g, indexing="ij")
    f = 1.0 + 0.1 * (dump % 7)                 # dumps differ
    rng = np.random.default_rng(seed + dump)
    F = rng.standard_normal((TOY["nslots"], n, n, n)) * 1e-3
    F[0] = X
    A, Om = TOY["A"] * f, TOY["Om"] * f
    F[1] = A * X - Om * Y
    F[2] = A * Y + Om * X
    F[3] = A * Z
    F[7] = TOY["T0"] * (1.0 + 1e-3 * f * X - 2e-3 * Y + 0.5e-3 * Z) + 1e-9 * rng.standard_normal((n, n, n))
    return F.astype(np.float32)


def write_toy_dump(root, dump, nb=2, nblock=24, ghost=2, fields=None, run_id="toy", pattern=mm.DEFAULT_PATTERN):
    """Write one dump as nb^3 block files root/pattern (default NNNN/run_id-BQavNNNN.<ext>; ext letters c0 c1 c2 ->
    cube block (c2, c1, c0), ghost-padded with NaN) such that ppmpy's read_moms_cube reassembles `fields`; returns
    the fields."""
    n = nb * nblock
    F = _toy_fields(n, TOY["dx"], dump) if fields is None else fields
    size = nblock + ghost
    for ext in itertools.product("abcdefgh"[:nb], repeat=3):
        c0, c1, c2 = (ord(e) - 97 for e in ext)
        B = np.full((F.shape[0], size, size, size), np.nan, dtype=np.float32)
        B[:, ghost - 1:ghost - 1 + nblock, ghost - 1:ghost - 1 + nblock, ghost - 1:ghost - 1 + nblock] = \
            F[:, c2 * nblock:(c2 + 1) * nblock, c1 * nblock:(c1 + 1) * nblock, c0 * nblock:(c0 + 1) * nblock]
        path = os.path.join(root, pattern.format(dump=dump, run_id=run_id, ext="".join(ext)))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        B.tofile(path)
    return F


def write_toy_rprof(prfs, dump, t, deex, run_id="toy"):
    """A minimal .rprof: the DUMP line (problem time) and a footer with deex (the parts ppmpy's Rprof parses)."""
    os.makedirs(prfs, exist_ok=True)
    lines = ["", "DUMP {:6d}  at time step # 1,   at problem time =  {:.8E},     and at  now".format(dump, t), "",
             "DATE: today"] + ["text"] * 15 + [
        "    42                          dtinit         1.000000000000000E+00                        nsteps        10",
        "    43                          deex         {:.15E}                        dtinit         4.1E+00".format(deex)]
    with open(os.path.join(prfs, "{}-{:04d}.rprof".format(run_id, dump)), "w") as f:
        f.write("\n".join(lines) + "\n")


def toy_times(dump):
    return 1000.0 * dump + 0.125


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """Three toy dumps (2^3 blocks of 24^3 + ghosts) and their rprofs."""
    root = tmp_path_factory.mktemp("moms_toy")
    mdir, prfs = str(root / "moms"), str(root / "prfs")
    F = {}
    for d in (7, 8, 9):
        F[d] = write_toy_dump(mdir, d)
        write_toy_rprof(prfs, d, toy_times(d), TOY["dx"])
    # ppmpy without an RprofSet builds its grid from slot 0: h = mean(diff(xc[0, 0, :])), i.e. MomsGrid(h / 4)
    h = np.mean(np.diff(F[7][0].astype(np.float64)[0, 0, :]))
    return dict(root=str(root), moms=mdir, prfs=prfs, F=F, dx_eff=h / 4.0, n=TOY["nb"] * TOY["nblock"])


def toy_source(toy, rprof=True, dx="eff"):
    return mm.MomsSource(toy["moms"], VARS, dx=toy["dx_eff"] if dx == "eff" else dx,
                         rprof=toy["prfs"] if rprof else None)


@pytest.fixture(scope="module")
def toy_mds(toy):
    """ppmpy MomsDataSet over the toy dumps (no RprofSet: grid from slot 0)."""
    ppm = _ppm()
    return ppm.MomsDataSet(toy["moms"], init_dump_read=7, dumps_in_mem=1, rprofset=None, var_list=VARS, verbose=0)


def _bits_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# ----------------------------------------------------------------------------------------------
# reader, grid, Jacobian
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("nb", [1, 2])
@pytest.mark.parametrize("mode", ["read", "memmap"])
def test_reader_equals_read_moms_cube(tmp_path, nb, mode):
    ppm = _ppm()
    nblock = 24 if nb == 2 else 20
    F = write_toy_dump(str(tmp_path), 5, nb=nb, nblock=nblock, fields=_toy_fields(nb * nblock, 0.3, 5))
    md = ppm.MomsData(str(tmp_path / "0005" / "toy-BQav0005.aaa"), verbose=0)
    src = mm.MomsSource(str(tmp_path), VARS, dx=0.3)
    assert src.run_id == "toy" and src.dumps() == [5]
    rd = mm.MomsBlockReader(src, 5, mode=mode)
    assert rd.n == nb * nblock == md.ngridpoints and rd.nblock == nblock and rd.size == nblock + 2
    assert rd.dtype == md.var.dtype == (np.float32 if nb == 1 else np.float64)
    rng = np.random.default_rng(1)
    n = rd.n
    for s in range(10):
        cube = md.get(s)
        assert _bits_equal(rd.cube(s), cube)
        assert np.array_equal(rd.cube(s), F[s], equal_nan=False)
        for _ in range(4):
            lo = rng.integers(0, n - 1, 3)
            hi = [rng.integers(a + 1, n + 1) for a in lo]
            box = rd.slab(VARS[s], lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
            assert _bits_equal(box, cube[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        i, j, k = rng.integers(0, n, (3, 500))
        assert _bits_equal(rd.gather(s, i, j, k), cube[i, j, k])
    with pytest.raises(KeyError):
        mm.MomsBlockReader(src, 5, slots=["T9"], mode=mode).slab("ux")
    with pytest.raises(ValueError):
        rd.slab("T9", 0, n + 1)


def test_source_errors(tmp_path):
    write_toy_dump(str(tmp_path), 3)
    src = mm.MomsSource(str(tmp_path), VARS)
    with pytest.raises(ValueError):
        src.dx                                   # neither dx nor rprof
    with pytest.raises(KeyError):
        src.slot("nope")
    os.remove(str(tmp_path / "0003" / "toy-BQav0003.bab"))
    with pytest.raises(FileNotFoundError):
        src.block_files(3)
    with pytest.raises(FileNotFoundError):
        src.block_files(4)
    with pytest.raises(ValueError):
        mm.MomsSource(str(tmp_path), {"T9": 12})


SMALL = dict(nb=2, nblock=6)                     # 12^3 cells: file-layout tests


def _small(dump):
    return _toy_fields(SMALL["nb"] * SMALL["nblock"], 0.3, dump)


@pytest.mark.parametrize("pattern,dumps", [
    ("{dump:04d}/{run_id}-XY{dump:04d}.{ext}", (3, 3200)),      # another prefix: ppmpy's rule would give [0, 3]
    ("{run_id}_{dump:04d}.{ext}", (3, 3200)),                   # '_' separator, flat directory
    ("{run_id}.{dump}.{ext}", (1, 12)),                         # not zero-padded: dump 1 must not see dump 12
    ("{run_id}-{ext}/{dump:04d}.bq", (2, 3)),                   # ext in a directory name
    (mm.DEFAULT_PATTERN, (3, 3200)),
])
def test_source_pattern(tmp_path, pattern, dumps):
    F = {d: write_toy_dump(str(tmp_path), d, fields=_small(d), pattern=pattern, **SMALL) for d in dumps}
    for run_id in (None, "toy"):
        src = mm.MomsSource(str(tmp_path), VARS, run_id=run_id, dx=0.3, pattern=pattern)
        assert src.run_id == "toy" and src.dumps() == sorted(dumps)
        for d in dumps:
            pairs = src.blocks(d)
            assert [e for e, _ in pairs] == ["".join(e) for e in itertools.product("ab", repeat=3)]
            assert src.block_files(d) == [p for _, p in pairs] == sorted(p for _, p in pairs)
            assert all(os.path.exists(p) for _, p in pairs)
            rd = mm.MomsBlockReader(src, d)
            for s in (0, 3, 7):
                assert np.array_equal(rd.cube(s), F[d][s])
    # ppmpy's discovery (default pattern) on files of another pattern: the reviewer's wrong dumps
    if pattern.startswith("{dump:04d}/{run_id}-XY"):
        assert mm.MomsSource(str(tmp_path), VARS, dx=0.3).dumps() == [0, 3]
    assert mm.MomsSource(str(tmp_path), VARS, run_id="other", dx=0.3, pattern=pattern).dumps() == []


def test_source_pattern_errors(tmp_path):
    write_toy_dump(str(tmp_path), 3, fields=_small(3), **SMALL)
    for bad in ("{dump:04d}/{foo}.{ext}", "{dump:04d}/{run_id}.aaa", "{run_id}.{ext}", "{dump:04d}/{run_id"):
        with pytest.raises(ValueError):
            mm.MomsSource(str(tmp_path), VARS, pattern=bad)
    with pytest.raises(ValueError, match="cannot infer run_id"):          # no file matches the pattern
        mm.MomsSource(str(tmp_path), VARS, pattern="{run_id}_{dump:04d}.{ext}")
    # a pattern without run_id needs none
    src = mm.MomsSource(str(tmp_path), VARS, pattern="{dump:04d}/toy-BQav{dump:04d}.{ext}", dx=0.3)
    assert src.run_id is None and src.dumps() == [3] and len(src.block_files(3)) == 8
    # default pattern: an .aaa file without '-' sorted first -> clear error; with run_id it is skipped
    open(str(tmp_path / "0003" / "junk.aaa"), "w").close()
    with pytest.raises(ValueError, match="cannot infer the run id"):
        mm.MomsSource(str(tmp_path), VARS)
    assert mm.MomsSource(str(tmp_path), VARS, run_id="toy").dumps() == [3]
    # blocks that do not form the nb^3 layout; stray files with other extensions are ignored
    open(str(tmp_path / "0003" / "toy-BQav0003.aaa.bak"), "w").close()
    src = mm.MomsSource(str(tmp_path), VARS, run_id="toy", dx=0.3)
    assert len(src.block_files(3)) == 8
    os.rename(str(tmp_path / "0003" / "toy-BQav0003.bbb"), str(tmp_path / "0003" / "toy-BQav0003.ccc"))
    with pytest.raises(ValueError, match="layout"):
        src.block_files(3)


def test_rprof_run_ids(tmp_path):
    prfs = str(tmp_path / "prfs")
    for d in (7, 8):
        write_toy_rprof(prfs, d, toy_times(d), 0.3)
    write_toy_rprof(prfs, 5, 55.5, 0.5, run_id="aaa")                   # another run id, sorted first
    assert list(mm.rprof_files(prfs)) == [5]
    assert list(mm.rprof_files(prfs, "toy")) == [7, 8]
    with pytest.raises(KeyError):
        mm.rprof_files(prfs, "nope")
    write_toy_dump(str(tmp_path / "moms"), 7, fields=_small(7), **SMALL)
    src = mm.MomsSource(str(tmp_path / "moms"), VARS, rprof=prfs)
    assert src.dx == 0.3 and src.time_s(8) == toy_times(8)            # the moms run id, not the first one
    with pytest.raises(KeyError):
        src.time_s(9)
    src = mm.MomsSource(str(tmp_path / "moms"), VARS, rprof=prfs, rprof_run_id="aaa")
    assert src.dx == 0.5 and src.time_s(5) == 55.5
    with pytest.raises(KeyError):
        mm.MomsSource(str(tmp_path / "moms"), VARS, rprof=prfs, rprof_run_id="nope").dx
    assert mm.MomsGrid.from_rprof(prfs, 12, run_id="toy").dx == 0.3
    assert mm.MomsGrid.from_rprof(prfs, 12).dx == 0.5
    # ppmpy's frombqavs naming
    bq = str(tmp_path / "bq")
    os.makedirs(bq)
    shutil.copy(os.path.join(prfs, "toy-0008.rprof"), os.path.join(bq, "toy-BQav0008.rprof"))
    assert mm.rprof_files(bq, "toy", prefix="BQav") == {8: os.path.join(bq, "toy-BQav0008.rprof")}
    with pytest.raises(FileNotFoundError):
        mm.rprof_files(bq)


class _StubRprofSet:
    """Duck-typed ppmpy RprofSet over toy rprof files (ppmpy's Rprof cannot parse them); like the real one it does
    not pickle (a dict_keys attribute)."""

    def __init__(self, directory, run_id="toy", bqav=False):
        self._RprofSet__dir_name = os.path.join(directory, "frombqavs" if bqav else "", "")
        self._RprofSet__bqav = bqav
        self._run_id = run_id
        self._files = mm.rprof_files(self._RprofSet__dir_name, run_id, "BQav" if bqav else "")
        self._keys = {}.keys()

    def get_run_id(self):
        return self._run_id

    def get_dump_list(self):
        return list(self._files)

    def get_dump(self, dump):
        h = mm.read_rprof_header(self._files[dump])
        return type("R", (), dict(get=lambda self_, k: h[k]))()

    def get(self, var, dump):
        return mm.read_rprof_header(self._files[dump])[var]


@pytest.mark.parametrize("bqav", [False, True])
def test_source_pickles_rprofset(toy, tmp_path, bqav):
    prfs = str(tmp_path / "prfs")
    for d in (7, 8, 9):
        write_toy_rprof(os.path.join(prfs, "frombqavs") if bqav else prfs, d, toy_times(d), 0.25)
        if bqav:
            os.rename(os.path.join(prfs, "frombqavs", "toy-{:04d}.rprof".format(d)),
                      os.path.join(prfs, "frombqavs", "toy-BQav{:04d}.rprof".format(d)))
    rp = _StubRprofSet(prfs, bqav=bqav)
    with pytest.raises(TypeError):
        pickle.dumps(rp)
    for dx in (None, toy["dx_eff"]):
        src = mm.MomsSource(toy["moms"], VARS, dx=dx, rprof=rp)
        back = pickle.loads(pickle.dumps(src))
        assert isinstance(back.rprof, str) and back.rprof_run_id == "toy"
        assert back.dx == src.dx == (0.25 if dx is None else dx)
        assert [back.time_s(d) for d in (7, 8, 9)] == [src.time_s(d) for d in (7, 8, 9)] == \
            [toy_times(d) for d in (7, 8, 9)]
        assert src.rprof is rp and back.dumps() == src.dumps() == [7, 8, 9]
    # no directory to fall back on: a clear error
    del rp._RprofSet__dir_name
    with pytest.raises(TypeError, match="rprof directory"):
        pickle.dumps(mm.MomsSource(toy["moms"], VARS, rprof=rp))


def test_grid_and_rprof_header(toy, toy_mds):
    g = mm.MomsGrid(toy["dx_eff"], toy["n"])
    assert _bits_equal(g.coord, toy_mds._unique_coord)
    x, y, z, r = g.box(0, toy["n"], 0, toy["n"], 0, toy["n"])
    for a, b in ((x, toy_mds._xc_view), (y, toy_mds._yc_view), (z, toy_mds._zc_view), (r, toy_mds._radius_view)):
        assert _bits_equal(a, b)
    h = mm.read_rprof_header(os.path.join(toy["prfs"], "toy-0008.rprof"))
    assert h == dict(t=toy_times(8), deex=TOY["dx"])
    assert mm.rprof_files(toy["prfs"]) == {d: os.path.join(toy["prfs"], "toy-{:04d}.rprof".format(d)) for d in (7, 8, 9)}
    assert mm.MomsGrid.from_rprof(toy["prfs"], 48).dx == TOY["dx"]
    src = toy_source(toy)
    assert src.time_s(9) == toy_times(9)
    assert np.isnan(toy_source(toy, rprof=False).time_s(9))
    with pytest.raises(KeyError):
        mm.read_rprof_header(os.path.join(toy["prfs"], "toy-0008.rprof"), ("nope",))


def test_grid_jacobian_port(toy, toy_mds):
    ppm = _ppm()
    n = toy["n"]
    g = mm.MomsGrid(toy["dx_eff"], n)
    ref = ppm.MomsDataSet._get_jacobian(toy_mds, toy_mds._xc_view, toy_mds._yc_view, toy_mds._zc_view,
                                        toy_mds._radius_view)
    full = mm.grid_jacobian(*g.box(0, n, 0, n, 0, n))
    assert _bits_equal(full, ref)
    sub = mm.grid_jacobian(*g.box(5, 17, 3, 40, 20, 21))
    assert _bits_equal(sub, ref[:, 5:17, 3:40, 20:21])
    rng = np.random.default_rng(2)
    i, j, k = rng.integers(0, n, (3, 1000))
    cx, cy, cz = g.coord[k], g.coord[j], g.coord[i]
    cr = np.sqrt(np.power(cx, 2.0) + np.power(cy, 2.0) + np.power(cz, 2.0))
    assert _bits_equal(mm.grid_jacobian(cx, cy, cz, cr, dtype=np.float32), ref[:, i, j, k])
    # ppmpy's 1-d (igrid) case: float64
    ref1 = ppm.MomsDataSet._get_jacobian(toy_mds, cx, cy, cz, cr)
    assert _bits_equal(mm.grid_jacobian(cx, cy, cz, cr), ref1)


# ----------------------------------------------------------------------------------------------
# sampling one dump
# ----------------------------------------------------------------------------------------------
NPT, RAD = 20000, 25.0


@pytest.fixture(scope="module")
def toy_ref(toy, toy_mds):
    """momsdataset backend on toy dump 8 (the legacy code path) and the legacy sphere_sample.py lines on it."""
    src = toy_source(toy)
    ref = mm.sample_moms_sphere(src, 8, RAD, NPT, 38230.0, backend="momsdataset", moms=toy_mds)
    # the frozen sphere_sample.py loop body, executed on the same MomsDataSet
    lines = open(_legacy_path("sphere_sample.py")).read().splitlines()
    i0 = next(i for i, s in enumerate(lines) if s.strip().startswith("T9 = m.get_spherical_interpolation"))
    i1 = next(i for i in range(i0, len(lines)) if lines[i].strip().startswith("teff = a.teff0"))
    ns = dict(np=np, m=toy_mds, d=8, fs=type("fs", (), dict(MOMS_VARS=VARS)),
              a=type("a", (), dict(radius=RAD, npoints=NPT, teff0=38230.0)))
    iu = next(i for i, s in enumerate(lines) if s.strip().startswith("iu = "))      # the velocity slots
    exec(textwrap.dedent(lines[iu]), ns)
    exec(textwrap.dedent("\n".join(lines[i0:i1 + 1])), ns)
    return src, ref, ns


def test_toy_momsdataset_backend_equals_legacy_lines(toy_ref):
    src, ref, ns = toy_ref
    for k, kl in (("T", "T9"), ("relT", "relT"), ("teff", "teff"), ("ur", "ur"), ("uth", "uth"), ("uph", "uph")):
        assert _bits_equal(ref[k], ns[kl]), k


@pytest.mark.parametrize("slab", [1, 3, 16, 64])
@pytest.mark.parametrize("cells", ["corners", "box"])
def test_toy_slab_equals_momsdataset(toy_ref, slab, cells):
    src, ref, _ = toy_ref
    s = mm.sample_moms_sphere(src, 8, RAD, NPT, 38230.0, backend="slab", slab=slab, slab_cells=cells)
    for k in ("theta", "phi", "x", "y", "z", "T", "relT", "teff", "ur", "uth", "uph"):
        assert _bits_equal(s[k], ref[k]), k
    assert s["T_mean"] == ref["T_mean"] and s["t_s"] == ref["t_s"] == toy_times(8)


def test_toy_components_false_and_no_velocity(toy_ref, toy_mds):
    src, ref, _ = toy_ref
    for kw in (dict(backend="slab"), dict(backend="momsdataset", moms=toy_mds)):
        s = mm.sample_moms_sphere(src, 8, RAD, NPT, 38230.0, components=False, **kw)
        assert "uth" not in s and _bits_equal(s["ur"], ref["ur"]) and _bits_equal(s["teff"], ref["teff"])
        s = mm.sample_moms_sphere(src, 8, RAD, NPT, 38230.0, velocity=None, **kw)
        assert "ur" not in s and _bits_equal(s["relT"], ref["relT"])


def test_toy_analytic(toy_ref, toy):
    """Linear T and u = A r_vec + Om z_hat x r_vec: T exact to float32, u_r = A r, u_theta = 0, u_phi = Om r sin(theta)."""
    _, s, _ = toy_ref
    f = 1.0 + 0.1 * (8 % 7)
    A, Om = TOY["A"] * f, TOY["Om"] * f
    T = TOY["T0"] * (1.0 + 1e-3 * f * s["x"] - 2e-3 * s["y"] + 0.5e-3 * s["z"])
    assert np.abs(s["T"] / T - 1).max() < 5e-6
    vs = 1e3
    assert np.abs(s["ur"] / (vs * A * RAD) - 1).max() < 2e-3          # trilinear interpolation of A r (not linear)
    assert np.abs(s["uth"]).max() < 1e-4 * vs * A * RAD
    assert np.abs(s["uph"] / (vs * Om * RAD * np.sin(s["theta"]) + 1e-30) - 1)[np.sin(s["theta"]) > 0.5].max() < 3e-3
    assert s["meta"]["backend"] == "momsdataset"


def test_toy_errors(toy, tmp_path):
    src = toy_source(toy)
    with pytest.raises(ValueError):
        mm.sample_moms_sphere(src, 8, 30.0, 1000, 38230.0, backend="slab")       # outside the grid (|x| <= 28.2)
    with pytest.raises(ValueError):
        mm.sample_moms_sphere(src, 8, RAD, 1000, 38230.0, backend="nope")
    # a NaN cell next to the sphere is reported
    F = _toy_fields(48, 0.3, 8)
    g = mm.MomsGrid(toy["dx_eff"], 48).coord
    i = int(np.argmin(np.abs(g - RAD)))
    F[7, i, 24, 24] = np.nan
    write_toy_dump(str(tmp_path), 8, fields=F)
    bad = mm.MomsSource(str(tmp_path), VARS, dx=toy["dx_eff"])
    with pytest.raises(ValueError, match="non-finite"):
        mm.sample_moms_sphere(bad, 8, RAD, NPT, 38230.0, backend="slab")


def test_defaults(toy_ref):
    """slab is the default backend (sample_moms_sphere, sample_moms_points, sample_moms_dumps); log flushes."""
    src, ref, _ = toy_ref
    for f in (mm.sample_moms_sphere, mm.sample_moms_points, mm.sample_moms_dumps):
        assert inspect.signature(f).parameters["backend"].default == "slab"
    log = inspect.signature(mm.sample_moms_dumps).parameters["log"].default
    assert log.func is print and log.keywords == dict(flush=True)
    s = mm.sample_moms_sphere(src, 8, RAD, NPT, 38230.0)
    assert s["meta"]["backend"] == "slab" and all(_bits_equal(s[k], ref[k]) for k in ("T", "ur", "uth", "uph"))


def test_momsdataset_grid_checks(toy, toy_ref, toy_mds):
    """The momsdataset backend refuses a grid other than the source's (ppmpy's slot-0 grid, another dx) instead
    of silently giving other values; it needs rprof unless a MomsDataSet is passed."""
    _, ref, _ = toy_ref
    pts = (ref["z"][:50], ref["y"][:50], ref["x"][:50])
    # dx only: ppmpy would build its grid from slot 0 (step 1.2000000325 instead of 1.2) -> refused before loading
    src = mm.MomsSource(toy["moms"], VARS, dx=TOY["dx"])
    for f, args in ((mm.sample_moms_sphere, (RAD, NPT, 38230.0)), (mm.sample_moms_points, pts)):
        with pytest.raises(ValueError, match="needs source.rprof"):
            f(src, 8, *args, backend="momsdataset")
        with pytest.raises(ValueError, match="not the source's grid"):          # the slot-0 MomsDataSet
            f(src, 8, *args, backend="momsdataset", moms=toy_mds)
    # rprof whose deex (0.3) is not the dx given: refused before ppmpy reads anything
    with pytest.raises(ValueError, match="not the source's grid"):
        mm.sample_moms_sphere(toy_source(toy), 8, RAD, NPT, 38230.0, backend="momsdataset")
    # rprof only (dx 0.3) against the slot-0 MomsDataSet
    with pytest.raises(ValueError, match="not the source's grid"):
        mm.sample_moms_sphere(toy_source(toy, dx=None), 8, RAD, NPT, 38230.0, backend="momsdataset",
                              moms=toy_mds)
    # neither dx nor rprof: the MomsDataSet's own grid, nothing to check against
    bare = mm.MomsSource(toy["moms"], VARS)
    s = mm.sample_moms_sphere(bare, 8, RAD, NPT, 38230.0, backend="momsdataset", moms=toy_mds)
    assert all(_bits_equal(s[k], ref[k]) for k in ("T", "relT", "ur", "uth", "uph")) and np.isnan(s["t_s"])
    # a MomsDataSet of another run id
    with pytest.raises(ValueError, match="run id"):
        mm.sample_moms_sphere(mm.MomsSource(toy["moms"], VARS, run_id="other", dx=toy["dx_eff"]), 8, RAD, NPT,
                              38230.0, backend="momsdataset", moms=toy_mds)


# points on grid nodes, on slab boundaries +- 1 ulp and on the grid ends, for 1, 2 and 3 blocks per dimension
HARD_NBLOCK = {1: 20, 2: 12, 3: 8}
HARD_SLABS = (1, 5)


@pytest.fixture(scope="module", params=sorted(HARD_NBLOCK))
def hard(request, tmp_path_factory):
    ppm = _ppm()
    nb = request.param
    n = nb * HARD_NBLOCK[nb]
    root = str(tmp_path_factory.mktemp("moms_hard{}".format(nb)))
    write_toy_dump(root, 5, nb=nb, nblock=HARD_NBLOCK[nb], fields=_toy_fields(n, 0.3, 5))
    m = ppm.MomsDataSet(root, init_dump_read=5, dumps_in_mem=1, rprofset=None, var_list=VARS, verbose=0)
    h = np.mean(np.diff(m.get(0)[0, 0, :]))                  # ppmpy's slot-0 spacing (float32 for one block)
    src = mm.MomsSource(root, VARS, dx=float(h) / 4.0)
    g = src.grid(n).coord
    assert _bits_equal(g, m._unique_coord)
    # coordinates: nodes, +- 1 ulp, the ends; slab boundaries (multiples of 5, and every node for slab 1)
    idx = sorted({0, 1, 2, 4, 5, 6, 9, 10, 11, n // 2 - 1, n // 2, n - 3, n - 2, n - 1})
    c = [g[i] for i in idx] + [np.nextafter(g[i], np.inf) for i in idx if i < n - 1] + \
        [np.nextafter(g[i], -np.inf) for i in idx if i > 0]
    c = np.array(sorted(set(c)))
    Z, Y, X = (a.ravel() for a in np.meshgrid(c, c[::3], c[1::3], indexing="ij"))
    rng = np.random.default_rng(3)
    R = rng.uniform(g[0], g[-1], (3, 3000))
    z, y, x = np.concatenate([Z, Y[::-1], R[0]]), np.concatenate([Y, X, R[1]]), np.concatenate([X, Z, R[2]])
    # oracle: RegularGridInterpolator on the full ppmpy cubes (T9 and get_spherical_components of ux, uy, uz)
    xi = np.column_stack([z, y, x])
    cubes = [m.get(7)] + m.get_spherical_components(m.get(1), m.get(2), m.get(3))
    want = [scipy.interpolate.RegularGridInterpolator((g, g, g), v)(xi) for v in cubes]
    want = dict(T=want[0], ur=want[1] * 1e3, uth=want[2] * 1e3, uph=want[3] * 1e3)
    return dict(nb=nb, src=src, m=m, g=g, z=z, y=y, x=x, want=want, dtype=cubes[0].dtype)


@pytest.mark.parametrize("slab", HARD_SLABS)
@pytest.mark.parametrize("cells", ["corners", "box"])
def test_points_hard(hard, slab, cells):
    want = hard["want"]
    assert hard["dtype"] == (np.float32 if hard["nb"] == 1 else np.float64)
    s = mm.sample_moms_points(hard["src"], 5, hard["z"], hard["y"], hard["x"], slab=slab, slab_cells=cells)
    for k in ("T", "ur", "uth", "uph"):
        assert _bits_equal(s[k], want[k]), k
    assert _bits_equal(s["grid"], hard["g"])
    if slab == HARD_SLABS[0] and cells == "corners":
        d = mm.sample_moms_points(hard["src"], 5, hard["z"], hard["y"], hard["x"], backend="momsdataset",
                                  moms=hard["m"])
        for k in ("T", "ur", "uth", "uph"):
            assert _bits_equal(d[k], want[k]), k
        s1 = mm.sample_moms_points(hard["src"], 5, hard["z"], hard["y"], hard["x"], components=False,
                                   velocity_scale=None, slab=slab)
        assert sorted(s1) == ["T", "grid", "ur"] and _bits_equal(s1["ur"] * 1e3, want["ur"])


def test_points_out_of_bounds(hard):
    g = hard["g"]
    for p in (np.nextafter(g[0], -np.inf), np.nextafter(g[-1], np.inf)):
        for axis in range(3):
            zyx = [np.zeros(3), np.zeros(3), np.zeros(3)]
            zyx[axis][1] = p
            with pytest.raises(ValueError):
                mm.sample_moms_points(hard["src"], 5, *zyx)
            with pytest.raises(ValueError):
                mm.sample_moms_points(hard["src"], 5, *zyx, backend="momsdataset", moms=hard["m"])
    with pytest.raises(ValueError):
        mm.sample_moms_points(hard["src"], 5, np.zeros(3), np.zeros(2), np.zeros(3))


# ----------------------------------------------------------------------------------------------
# products vs the frozen legacy scripts (run as scripts on the toy dumps)
# ----------------------------------------------------------------------------------------------
FAKE_FIGSTYLE = '''
import os, sys
sys.path.insert(0, {root!r})
from ppmpy import ppm
MOMS_DIR = {moms!r}
PRFS = {prfs!r}
MOMS_DUMPS = (7, 9)
MOMS_VARS = {vars!r}
def moms(dump=MOMS_DUMPS[0], var_list=None, dumps_in_mem=1):
    return ppm.MomsDataSet(MOMS_DIR, init_dump_read=dump, dumps_in_mem=dumps_in_mem, rprofset=None,
                           var_list=MOMS_VARS if var_list is None else var_list, verbose=0)
def time_s(dump):
    for line in open(os.path.join(PRFS, "toy-%04d.rprof" % dump)):
        if line.startswith("DUMP"):
            return float(line.split("=")[1].split(",")[0])
'''


def _run_legacy(toy, tmp, script, args):
    fake = os.path.join(str(tmp), "fake_figstyle")
    os.makedirs(fake, exist_ok=True)
    with open(os.path.join(fake, "figstyle.py"), "w") as f:
        f.write(FAKE_FIGSTYLE.format(root=ROOT, moms=toy["moms"], prfs=toy["prfs"], vars=VARS))
    env = dict(os.environ, PYTHONPATH=fake + os.pathsep + os.environ.get("PYTHONPATH", ""))
    res = subprocess.run([sys.executable, _legacy_path(script)] + [str(a) for a in args], env=env,
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout + res.stderr
    return res.stdout


def _npz_equal(pa, pb):
    a, b = np.load(pa), np.load(pb)
    assert list(a.files) == list(b.files)
    for k in a.files:
        assert _bits_equal(a[k], b[k]), k


def test_write_points_table_equals_legacy_script(toy, tmp_path):
    out_l = tmp_path / "legacy"
    _run_legacy(toy, tmp_path, "fw_sphere_extract.py", ["--dump", 8, "--radius", RAD, "--npoints", NPT,
                                                        "--teff0", 38230.0, "--outdir", out_l])
    src = toy_source(toy)
    # components False / True; teff0 and radius as int too (recorded as floats, like the legacy argparse values)
    for comps, teff0, radius in ((False, 38230.0, RAD), (True, 38230.0, RAD), (True, 38230, int(RAD))):
        out = tmp_path / "new{}{}".format(int(comps), type(teff0).__name__)
        s = mm.sample_moms_sphere(src, 8, radius, NPT, teff0, backend="slab", components=comps)
        rec = mm.write_points_table(str(out), s, teff0)
        _npz_equal(out / "points.npz", out_l / "points.npz")
        for name in ("points.txt", "meta.json"):
            assert (out / name).read_bytes() == (out_l / name).read_bytes(), name
        assert rec == json.loads((out_l / "meta.json").read_text())
        assert sorted(os.listdir(out)) == ["meta.json", "points.npz", "points.txt"]       # no temporaries left


@pytest.mark.parametrize("nproc,start", [(1, None), (2, "fork"), (2, "spawn")])
def test_sample_moms_dumps_equals_legacy_script(toy, tmp_path, nproc, start):
    out_l = tmp_path / "legacy"
    _run_legacy(toy, tmp_path, "sphere_sample.py", ["--d0", 7, "--d1", 9, "--radius", RAD, "--npoints", NPT,
                                                    "--teff0", 38230.0, "--outdir", out_l])
    src = toy_source(toy)
    out = tmp_path / "new"
    res = mm.sample_moms_dumps(src, [7, 8, 9], RAD, NPT, 38230.0, str(out), nproc=nproc, start_method=start,
                               log=None)
    assert [r["dump"] for r in res] == [7, 8, 9]
    for d in (7, 8, 9):
        _npz_equal(out / "d{:04d}.npz".format(d), out_l / "d{:04d}.npz".format(d))
    # restartable: nothing to do; overwrite redoes; the rank split covers every dump once
    assert mm.sample_moms_dumps(src, [7, 8, 9], RAD, NPT, 38230.0, str(out), log=None) == []
    split = [mm.sample_moms_dumps(src, [7, 8, 9], RAD, NPT, 38230.0, str(tmp_path / "split"), rank=r, nranks=2,
                                  log=None) for r in (0, 1)]
    assert [x["dump"] for x in split[0]] == [8] and [x["dump"] for x in split[1]] == [7, 9]
    for d in (7, 8, 9):
        _npz_equal(tmp_path / "split" / "d{:04d}.npz".format(d), out_l / "d{:04d}.npz".format(d))
    assert sorted(os.listdir(out)) == ["d0007.npz", "d0008.npz", "d0009.npz"]


def test_sample_moms_dumps_check(toy, tmp_path):
    src = toy_source(toy)
    s = mm.sample_moms_sphere(src, 8, RAD, NPT, 38230.0, backend="slab", components=False)
    mm.write_points_table(str(tmp_path / "run"), s, 38230.0)
    res = mm.sample_moms_dumps(src, [8, 9], RAD, NPT, 38230.0, str(tmp_path / "out"), check=str(tmp_path / "run"),
                               log=None)
    assert res[0]["check"] == dict(relT=0.0, teff=0.0, ur_kms=0.0)
    assert res[1]["check"]["teff"] > 0


@pytest.mark.parametrize("start", ["fork", "spawn"])
def test_sample_moms_dumps_rprofset_workers(toy, tmp_path, start):
    """A source holding an RprofSet (here a stand-in that, like ppmpy's, does not pickle) works with spawn workers:
    the same files as with the rprof directory in one process."""
    ref = tmp_path / "ref"
    mm.sample_moms_dumps(toy_source(toy), [7, 8, 9], RAD, NPT, 38230.0, str(ref), log=None)
    src = mm.MomsSource(toy["moms"], VARS, dx=toy["dx_eff"], rprof=_StubRprofSet(toy["prfs"]))
    out = tmp_path / "new"
    res = mm.sample_moms_dumps(src, [7, 8, 9], RAD, NPT, 38230.0, str(out), nproc=2, start_method=start, log=None)
    assert [r["dump"] for r in res] == [7, 8, 9]
    for d in (7, 8, 9):
        _npz_equal(out / "d{:04d}.npz".format(d), ref / "d{:04d}.npz".format(d))


@pytest.fixture(scope="module")
def real_rprofs(tmp_path_factory, toy):
    """Copies of the M424 rprofs of dumps 7-9 named like the toy run (ppmpy's Rprof cannot parse the toy files)."""
    src = os.path.join(M424_RUN_DIR, "prfs")
    if not os.path.isdir(src):
        pytest.skip("M424 rprofs not available: {}".format(src))
    d = str(tmp_path_factory.mktemp("real_rprofs"))
    for dump in (7, 8, 9):
        shutil.copy(os.path.join(src, "M25Z0cnstmu-{:04d}.rprof".format(dump)),
                    os.path.join(d, "toy-{:04d}.rprof".format(dump)))
    return d


@pytest.mark.m424
def test_real_rprofset(toy, real_rprofs, tmp_path, capfd):
    """With a ppmpy RprofSet (M424 rprof files): the momsdataset backend builds its grid from it and equals the slab
    backend, quietly (verbose=0); the source pickles (directory + run id) with the same dx and t_s, and spawn
    workers write the same files."""
    ppm = _ppm()
    rp = ppm.RprofSet(real_rprofs, verbose=0)
    with pytest.raises(TypeError):
        pickle.dumps(rp)
    rad = 300.0                                                   # grid of 4 x 4.57 Mm cells: |x| <= 430 Mm
    for rprof in (real_rprofs, rp):
        src = mm.MomsSource(toy["moms"], VARS, rprof=rprof)
        assert src.dx == rp.get_dump(7).get("deex") == 4.572318077087402
        capfd.readouterr()
        a = mm.sample_moms_sphere(src, 8, rad, NPT, 38230.0, backend="momsdataset")
        assert capfd.readouterr().out == ""
        b = mm.sample_moms_sphere(src, 8, rad, NPT, 38230.0)
        for k in ("T", "relT", "teff", "ur", "uth", "uph"):
            assert _bits_equal(a[k], b[k]), k
        assert a["t_s"] == b["t_s"] == float(rp.get("t", 8)) and a["meta"]["grid_step"] == b["meta"]["grid_step"]
    back = pickle.loads(pickle.dumps(src))
    assert back.rprof == os.path.join(real_rprofs, "") and back.rprof_run_id == "toy" and back.dx == src.dx
    assert [back.time_s(d) for d in (7, 8, 9)] == [float(rp.get("t", d)) for d in (7, 8, 9)]
    ref = tmp_path / "ref"
    mm.sample_moms_dumps(mm.MomsSource(toy["moms"], VARS, rprof=real_rprofs), [7, 8, 9], rad, NPT, 38230.0,
                         str(ref), log=None)
    mm.sample_moms_dumps(src, [7, 8, 9], rad, NPT, 38230.0, str(tmp_path / "new"), nproc=2, start_method="spawn",
                         log=None)
    for d in (7, 8, 9):
        _npz_equal(tmp_path / "new" / "d{:04d}.npz".format(d), ref / "d{:04d}.npz".format(d))


# ----------------------------------------------------------------------------------------------
# M424 (slow): the stored products
# ----------------------------------------------------------------------------------------------
def _m424_source():
    mdir, prfs = os.path.join(M424_RUN_DIR, "moms", "myavsbq"), os.path.join(M424_RUN_DIR, "prfs")
    if not (os.path.isdir(mdir) and os.path.isdir(prfs)):
        pytest.skip("M424 moms/rprofs not available: {}".format(M424_RUN_DIR))
    return mm.MomsSource(mdir, VARS, rprof=prfs)


def _m424_sample_file(dump):
    p = os.path.join(conftest.M424.get("samples", M424_SAMPLES), "d{:04d}.npz".format(dump))
    if not os.path.exists(p):
        pytest.skip("M424 product not available: {}".format(p))
    return p


@pytest.mark.m424
def test_m424_rprof_header():
    ppm = _ppm()
    src = _m424_source()
    path = os.path.join(M424_RUN_DIR, "prfs", "M25Z0cnstmu-0000.rprof")
    rp = ppm.Rprof(path, verbose=0)
    assert mm.read_rprof_header(path) == dict(t=rp.get("t"), deex=rp.get("deex"))
    assert src.dx == 4.572318077087402 and src.time_s(3200) == 9070526.0


@pytest.fixture(scope="module")
def m424_d3200(tmp_path_factory):
    src = _m424_source()
    s = mm.sample_moms_sphere(src, 3200, M424_RADIUS, M424_NPOINTS, M424_TEFF0, backend="slab")
    return src, s


@pytest.mark.m424
@pytest.mark.slow
def test_m424_d3200_slab_equals_points_and_samples(m424_d3200, tmp_path):
    src, s = m424_d3200
    run = conftest.m424_path("run")
    p = np.load(os.path.join(run, "points.npz"))
    for k, v in (("theta", s["theta"]), ("phi", s["phi"]), ("x", s["x"]), ("y", s["y"]), ("z", s["z"]),
                 ("relT", s["relT"]), ("teff", s["teff"]), ("ur_kms", s["ur"]), ("T9", s["T"])):
        assert _bits_equal(p[k], v), k
    mm.write_points_table(str(tmp_path), s, M424_TEFF0)
    _npz_equal(tmp_path / "points.npz", os.path.join(run, "points.npz"))
    for name in ("points.txt", "meta.json"):
        with open(os.path.join(run, name), "rb") as f:
            assert (tmp_path / name).read_bytes() == f.read(), name
    ref = np.load(_m424_sample_file(3200))
    for k in mm.SAMPLE_KEYS:
        assert _bits_equal(s[k].astype(np.float32), ref[k]), k
    assert float(s["T"].mean()) == float(ref["T9_mean"]) and s["t_s"] == float(ref["t_s"])


@pytest.mark.m424
@pytest.mark.slow
def test_m424_d3200_momsdataset_equals_slab(m424_d3200):
    """The legacy path (ppmpy MomsDataSet, ~19 GB) gives the same bits as the slab backend."""
    src, s = m424_d3200
    ref = mm.sample_moms_sphere(src, 3200, M424_RADIUS, M424_NPOINTS, M424_TEFF0, backend="momsdataset")
    for k in ("T", "relT", "teff", "ur", "uth", "uph"):
        assert _bits_equal(ref[k], s[k]), k
    assert ref["meta"]["grid_first"] == s["meta"]["grid_first"]


@pytest.mark.m424
@pytest.mark.slow
def test_m424_subset_dumps_equal_samples(tmp_path):
    """sample_moms_dumps (slab, 4 workers) on the verification subset == the stored dNNNN.npz byte for byte (members)."""
    src = _m424_source()
    dumps = subset_dumps()
    refs = [_m424_sample_file(d) for d in dumps]
    res = mm.sample_moms_dumps(src, dumps, M424_RADIUS, M424_NPOINTS, M424_TEFF0, str(tmp_path), nproc=4,
                               log=None)
    assert [r["dump"] for r in res] == dumps
    for d, ref in zip(dumps, refs):
        _npz_equal(tmp_path / "d{:04d}.npz".format(d), ref)
