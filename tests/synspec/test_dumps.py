"""Tests of ppmpy.synspec.dumps: the all-dump driver (port of fw_disc_dumps.py) and the time-series collect (port of
fw_disc_collect.py).

Synthetic (no data; the M424 velocity grid, 3 lines and 8 lines of sight, so that the frozen legacy code runs next to
ours): a toy flux library and a toy sphere of 3000 points with random per-dump samples. Per-dump files byte for byte
those of the frozen fw_disc_dumps.py process() (its source lines) for the flux, smoothed and lamfix variants; the time
series byte for byte that of the frozen fw_disc_collect.py (run as a script); serial, fork and spawn identical, also in
a subprocess with 4 OpenBLAS threads (the integrator's nodes depend on the thread count; the legacy oracle runs with 1,
as the production); restart, rank split, watchdog, collect options; the run record (inputs fingerprint, node profiles,
points, extra values, corrupted / stale records, exclusive creation), the workers' integrator check, lref and nl checks
before anything is written, the legacy nmin rule for non-flux integrators, canonical dump file names, the second-pass
identity check of the collect. Intensity method (a toy intensity library on the M424 grid): imu_integrator runs write
the files of the frozen fw_disc_dumps.py --method imu (frozen fw_disc.DiscImu) byte for byte (serial, fork, spawn,
batched, lref to the run or the factory), the lazy and float32 modes (serial = fork = spawn = per line of sight; the
run record refuses mixing modes; each mode its own default name; record-less legacy directories adopted by the legacy
mode only), lref attachment (names checked, also against a library that records names only; lref not part of the
integrator's identity, so a restart may move it between the run and the factory), lines by name, an ImuLibrary under
spawn, disc_dump's batch option, the CPU-time warning of in-process runs. M424 (marker
m424): dumps 3200-3209 of the three production flux runs, byte for byte, and their time series rows; the imu run
(slow): dumps 3200-3209 with 4 fork and 2 spawn workers byte for byte, the lazy mode in a fresh process (time, peak
RSS <= 3 GB).

PPMPY_SYNSPEC_M424_SAMPLES (default /scratch/ppathak/fastwind_sphere/samples_r4050_N1236544) locates the per-dump
sphere samples, PPMPY_SYNSPEC_DUMPS_SHADOW (default /scratch/ppathak/synspec_shadow/m2/dumps) the scratch directory of
the M424 runs (a fresh subdirectory per test, removed when the test passes); PPMPY_SYNSPEC_DUMPS_SHADOW_M4 (default
/scratch/ppathak/synspec_shadow/m4/dumps_validate_fwresults) that of the imu runs; PPMPY_SYNSPEC_M424_IMU the intensity
library (the .npz, or a directory holding imu_library_dT10.npz)."""
import hashlib
import json
import os
import platform
import resource
import socket
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import types
import warnings

import numpy as np
import pytest

import conftest
from conftest import ROOT, m424_path
from ppmpy.synspec import dumps as dm
from ppmpy.synspec import library as lb
from ppmpy.synspec import parallel as par
from ppmpy.synspec import sphere as sph
from ppmpy.synspec.conventions import los_thompson2024
from ppmpy.synspec.diagnostics import diagnostics_array
from ppmpy.synspec.io import npz_member_memmap, read_meta
from ppmpy.synspec.spectral import LineSet, VelocityGrid

LREF = np.array([4026.22, 4199.90, 4921.93])
LINES = ["HEI4026", "HEII4200", "HEI4922"]
LINESET = LineSet(LINES, LREF)
LEGACY = os.environ.get("PPMPY_SYNSPEC_M424_PROJECT",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy"))
SHADOW = os.environ.get("PPMPY_SYNSPEC_DUMPS_SHADOW", "/scratch/ppathak/synspec_shadow/m2/dumps")
SHADOW_M4 = os.environ.get("PPMPY_SYNSPEC_DUMPS_SHADOW_M4",
                           "/scratch/ppathak/synspec_shadow/m4/dumps_validate_fwresults")
GRID = VelocityGrid()
N_TOY = 3000
DUMPS = list(range(100, 106))
NMIN = 20
# toy library variants: name -> (factory kwargs except corr, legacy smooth, legacy lamfix)
VARIANTS = {"flux": (dict(), 0.0, False), "flux_sm150": (dict(smooth=150.0), 150.0, False),
            "flux_lamfix": (dict(), 0.0, True)}


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
    with `last` (stripped; dedented) in namespace ns (as test_disc.py); returns ns."""
    path = os.path.join(LEGACY, name)
    if not os.path.exists(path):
        pytest.skip("legacy {} not available in {}".format(name, LEGACY))
    with open(path) as f:
        lines = f.read().splitlines()
    i0 = next(i for i, x in enumerate(lines) if x.strip().startswith(first))
    i1 = next(i for i in range(i0, len(lines)) if lines[i].strip().startswith(last))
    exec(textwrap.dedent("\n".join(lines[i0:i1 + 1])), ns)
    return ns


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _assert_npz_equal(a, b, bytes_too=True):
    """Same members in the same order, dtypes, shapes and values (NaN-aware); optionally the same bytes."""
    with np.load(a) as za, np.load(b) as zb:
        assert za.files == zb.files
        for k in za.files:
            x, y = za[k], zb[k]
            assert x.dtype == y.dtype and x.shape == y.shape, k
            assert np.array_equal(x, y, equal_nan=x.dtype.kind == "f"), k
    if bytes_too:
        assert _sha(a) == _sha(b)


def _toy_library(seed=0, nb=80, t0=36600.0, dT=10.0):
    """A flux library on the M424 grid: Gaussian absorption lines whose depth, width and centre change with T_eff'
    (depth exactly 0 at |y| > 2300 km/s after the float32 cast), sparse tails and an empty bin inside."""
    rng = np.random.default_rng(seed)
    edges = t0 + dT * np.arange(nb + 1)
    count = rng.integers(1, 15, nb).astype(float)
    count[:6] = [0, 1, 0, 2, 0, 3]
    count[-5:] = [2, 0, 1, 0, 1]
    count[30] = 0
    centres = 0.5 * (edges[:-1] + edges[1:])
    tmean = np.where(count > 0, centres + rng.uniform(-4.0, 4.0, nb), centres)
    x = (tmean - 37000.0) / 300.0
    y = GRID.y
    prof = np.empty((nb, 3, GRID.ny))
    fc = np.empty((nb, 3))
    for j, (sig, amp, c0) in enumerate(((60.0, 0.3, 0.0), (140.0, 0.2, 5.0), (40.0, 0.4, -3.0))):
        s = sig * (1.0 + 0.05 * x)[:, None]
        depth = amp * (1.0 + 0.2 * x)[:, None] * np.exp(-0.5 * ((y[None, :] - c0 - 2.0 * x[:, None]) / s) ** 2)
        prof[:, j] = 1.0 - depth
        fc[:, j] = 1e-3 * (1.0 + 0.1 * x + 0.01 * j)
    return lb.FluxLibrary(edges, tmean, count, prof.astype(np.float32), fc, dT)


def _toy_corr(nb, seed=1):
    """An additive per-bin profile correction (the lamfix variant), exactly 0 far from the line centres."""
    rng = np.random.default_rng(seed)
    amp = 1e-3 * rng.standard_normal((nb, 3, 1))
    return amp * np.exp(-0.5 * (GRID.y[None, None, :] / 20.0) ** 2)


def _write_samples(sdir, dumps, n=N_TOY):
    """Per-dump samples like sphere_sample.py's (float32 teff, ur, uth, uph; t_s; relT): T_eff' partly beyond the toy
    nodes, a few |v| > 400 km/s (clipped)."""
    os.makedirs(sdir, exist_ok=True)
    for d in dumps:
        rng = np.random.default_rng(d)
        teff = rng.normal(37000.0, 180.0, n).astype(np.float32)
        u = rng.normal(0.0, 25.0, (3, n)).astype(np.float32)
        u[0, rng.choice(n, 8, replace=False)] = 450.0
        np.savez(os.path.join(sdir, "d{:04d}.npz".format(d)), relT=(teff / 37000.0 - 1.0).astype(np.float32),
                 teff=teff, ur=u[0], uth=u[1], uph=u[2], t_s=np.float64(2835.0 * d))


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """Toy inputs in a module tmp dir: library file, lamfix correction (as the lamfix_dT10.npz cache), samples of
    DUMPS, the sphere (theta, phi)."""
    root = tmp_path_factory.mktemp("dumps_toy")
    lib = _toy_library()
    lib_path = lib.save(str(root / "library_dT10.npz"))
    lamfix_dir = root / "lamfix"
    lamfix_dir.mkdir()
    corr_path = str(lamfix_dir / "lamfix_dT10.npz")
    np.savez(corr_path, corr=_toy_corr(lib.nb))
    sdir = str(root / "samples")
    _write_samples(sdir, DUMPS)
    theta, phi = sph.fibonacci_sphere(N_TOY)
    return types.SimpleNamespace(root=root, library=lib_path, corr=corr_path, lamfix_dir=str(lamfix_dir),
                                 samples=sdir, theta=theta, phi=phi)


def _kw(toy, variant):
    """factory_kwargs of a toy variant."""
    kw = dict(VARIANTS[variant][0], nmin=NMIN)
    if VARIANTS[variant][2]:
        kw["corr"] = toy.corr
    return kw


def _run(toy, outdir, variant="flux", dumps=DUMPS, name=None, **kw):
    """run_disc_dumps on the toy inputs."""
    return dm.run_disc_dumps(dumps, toy.samples, str(outdir), name, dm.flux_integrator, (toy.library,), toy.theta,
                             toy.phi, "thompson2024", lref=LINESET, factory_kwargs=_kw(toy, variant), **kw)


def _legacy_dump_files(toy, outroot, variant, monkeypatch):
    """The per-dump files of the frozen fw_disc_dumps.py for a toy variant: its source lines (NAME, the projections,
    KEYS, process()) executed with the toy library / sphere / samples; returns the output directory. Run with the
    loaded BLAS limited to 1 thread, as the production (container: OMP_NUM_THREADS=1): the legacy lib_nodes with
    smoothing or lamfix depends on the thread count."""
    fd = _legacy_fw_disc()
    _, smooth, lamfix = VARIANTS[variant]
    monkeypatch.setattr(fd, "SAMPLES", toy.samples)
    monkeypatch.setattr(fd, "DISC_DUMPS", toy.lamfix_dir)          # lam_corrections() reads its cache there
    a = types.SimpleNamespace(overwrite=False, method="flux", lamfix=lamfix, smooth=smooth, nmin=NMIN)
    ns = dict(np=np, fd=fd, os=os, time=time, a=a, pts=dict(theta=toy.theta, phi=toy.phi))
    _legacy_exec("fw_disc_dumps.py", "NAME = a.method", "NAME = a.method", ns)
    _legacy_exec("fw_disc_dumps.py", "rhat, that, phat = fd.unit_vectors", "del rhat, that, phat, pts", ns)
    _legacy_exec("fw_disc_dumps.py", "KEYS = [", "KEYS = [", ns)
    with np.load(toy.library) as lib, dm._blas_limit(1):
        ns["INT"] = fd.DiscFlux(fd.lib_nodes(lib, nmin=a.nmin, smooth=a.smooth, lamfix=a.lamfix))
    out = os.path.join(str(outroot), ns["NAME"])
    os.makedirs(out)
    ns["OUT"] = out
    _legacy_exec("fw_disc_dumps.py", "def process(d):", 'f"{time.time() - t0:.1f} s")', ns)
    for d in DUMPS:
        assert ns["process"](d).startswith("dump {}: EW".format(d))
    assert ns["NAME"] == variant
    return out


def _legacy_collect(outroot, name, d0, d1):
    """Run the frozen fw_disc_collect.py as a script (it writes <outroot>/<name>_timeseries.npz); returns the path."""
    script = os.path.join(LEGACY, "fw_disc_collect.py")
    if not os.path.exists(script):
        pytest.skip("legacy fw_disc_collect.py not available in {}".format(LEGACY))
    res = subprocess.run([sys.executable, script, "--name", name, "--d0", str(d0), "--d1", str(d1), "--out",
                          str(outroot)], capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout + res.stderr
    return os.path.join(str(outroot), "{}_timeseries.npz".format(name))


def _files(d):
    return sorted(f for f in os.listdir(str(d)) if f.endswith(".npz"))


# picklable factories / integrators for the pool tests (module level: importable by spawn workers)
def _differs_in_workers(library, **kw):
    """flux_integrator whose nodes differ in pool workers (another nmin): the fingerprint check must catch it."""
    import multiprocessing
    if multiprocessing.current_process().name != "MainProcess":
        kw = dict(kw, nmin=kw.get("nmin", 20) + 7)
    return dm.flux_integrator(library, **kw)


class _Killer:
    """An integrator whose call kills the worker process (like the OOM killer or a CPU-time limit)."""
    method = "flux"

    def __init__(self, base):
        self.t, self.grid, self.node_params = base.t, base.grid, base.node_params

    def pairs(self, teff):
        return lb.node_pairs(self.t, teff)

    def __call__(self, *args):
        os._exit(7)


def _killer_factory(library):
    return _Killer(dm.flux_integrator(library))


def _extra(d, sample, res):
    """Per-dump extra fields (picklable)."""
    return dict(sample_path=np.array(os.path.basename(sample["path"])), ew0=res["diag_F"][:, :, 0].mean(axis=0))


def _perturbed(lib, seed=5):
    """The library with other profiles (the depth of one filled bin changed by 1e-4 relative; the continuum stays 1)
    but the same edges, tmean, counts and fc: the same nodes (t, node_params), other node profiles."""
    prof = np.array(lib.prof, copy=True)
    b = int(np.where(lib.count > 0)[0][seed])
    prof[b] = (1.0 - (1.0 - prof[b].astype(np.float64)) * (1.0 + 1e-4)).astype(prof.dtype)
    assert not np.array_equal(prof, lib.prof)
    return lb.FluxLibrary(lib.edges, lib.tmean, lib.count, prof, lib.fc, lib.dT)


def _prof_differs_in_workers(library, **kw):
    """flux_integrator whose node profiles (not t, not node_params) differ in pool workers: the check of the
    workers' integrator must hash the profiles too."""
    import multiprocessing
    lib = lb.FluxLibrary.load(library)
    if multiprocessing.current_process().name != "MainProcess":
        lib = _perturbed(lib)
    return dm.flux_integrator(lib, **kw)


class _ImuLike:
    """A non-DiscFlux integrator (method 'imu', as the M4 DiscImu will be) whose node_params carry nmin 20; it wraps a
    DiscFlux and identifies itself through fingerprint()."""
    method = "imu"

    def __init__(self, base):
        self.base = base
        self.t, self.grid, self.nl = base.t, base.grid, base.nl
        self.node_params = dict(base.node_params)

    def pairs(self, teff):
        return self.base.pairs(teff)

    def __call__(self, *args):
        return self.base(*args)

    def fingerprint(self):
        return dict(base=dm._integ_info(self.base)["sha256"])


def _imu_like_factory(library, **kw):
    return _ImuLike(dm.flux_integrator(library, **kw))


# ----------------------------------------------------------------------------------------------
# module
# ----------------------------------------------------------------------------------------------
def test_no_heavy_imports():
    code = ("import sys; sys.path.insert(0, {!r}); import ppmpy.synspec.dumps; "
            "print(sorted(m for m in sys.modules if m.startswith(('matplotlib', 'ppmpy.ppm'))))").format(ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_los_and_grid_match_legacy():
    """The default lines of sight and grid are the legacy module constants bit for bit (the time series stores them)."""
    fd = _legacy_fw_disc()
    np.testing.assert_array_equal(los_thompson2024(), fd.los8())
    np.testing.assert_array_equal(GRID.y, fd.Y)
    assert GRID.nshift == fd.VSHIFT


# ----------------------------------------------------------------------------------------------
# one dump
# ----------------------------------------------------------------------------------------------
def test_load_sample(toy):
    s = dm.load_sample(toy.samples, 101)
    for k in ("teff", "ur", "uth", "uph"):
        assert s[k].dtype == np.float64 and s[k].shape == (N_TOY,)
    with np.load(os.path.join(toy.samples, "d0101.npz")) as z:
        np.testing.assert_array_equal(s["teff"], z["teff"].astype(np.float64))
        assert s["t_s"] == float(z["t_s"]) and isinstance(s["t_s"], float)
    assert s["dump"] == 101 and s["path"] == os.path.abspath(os.path.join(toy.samples, "d0101.npz"))
    with pytest.raises(FileNotFoundError):
        dm.load_sample(toy.samples, 99)


def test_disc_dump_parts(toy):
    """disc_dump = DiscFlux.integrate_los + coverage + diagnostics_array (whole grid, recorded as grid.vmax); the toy
    exercises clipped shifts and points beyond both ends of the node range."""
    integ = dm.flux_integrator(toy.library, nmin=NMIN)
    smp = dm.load_sample(toy.samples, 102)
    MU, TN, PN = sph.project_los(toy.theta, toy.phi, "thompson2024")
    r = dm.disc_dump(integ, smp, MU, TN, PN, lref=LINESET)
    V = sph.los_velocity(smp["ur"], smp["uth"], smp["uph"], MU, TN, PN)
    ref = integ.integrate_los(MU, V, teff=smp["teff"])
    for k in ("F", "F0", "vmean_w", "sigma_w", "n_clip"):
        np.testing.assert_array_equal(r[k], ref[k], err_msg=k)
    n_lo, n_hi, wout = lb.coverage(integ.t, smp["teff"], MU)
    assert (r["n_lo"], r["n_hi"]) == (n_lo, n_hi) and n_lo > 0 and n_hi > 0
    np.testing.assert_array_equal(r["wout"], wout)
    assert r["n_clip"].sum() > 0
    np.testing.assert_array_equal(r["diag_F"], diagnostics_array(ref["F"], GRID.y, LREF))
    assert r["diag_vwin"] == 2700.0 and list(r["diag_keys"]) == ["ew", "v1", "sigma", "fwhm", "depth"]
    np.testing.assert_array_equal(r["node_range"], [integ.t[0], integ.t[-1]])
    assert r["dump"] == 102 and r["teff_max"] == smp["teff"].max()
    # a moment window and a key subset
    r2 = dm.disc_dump(integ, smp, MU[:2], TN[:2], PN[:2], lref=LREF, diag_vwin=400.0, diag_keys=("ew", "v1"))
    assert r2["diag_F"].shape == (2, 3, 2) and r2["diag_vwin"] == 400.0
    np.testing.assert_array_equal(r2["F"], r["F"][:2])
    np.testing.assert_array_equal(r2["diag_F"], diagnostics_array(r["F"][:2], GRID.y, LREF, vwin=400.0,
                                                                  keys=("ew", "v1")))
    with pytest.raises(ValueError, match="lref is required"):
        dm.disc_dump(integ, smp, MU, TN, PN)
    with pytest.raises(ValueError, match="points"):
        dm.disc_dump(integ, smp, MU[:, :-1], TN[:, :-1], PN[:, :-1], lref=LREF)
    with pytest.raises(ValueError, match="one reference wavelength per line"):
        dm.disc_dump(integ, smp, MU, TN, PN, lref=LREF[:2])
    with pytest.raises(ValueError, match="diag_keys"):
        dm.disc_dump(integ, smp, MU, TN, PN, lref=LREF, diag_keys=("ew", "nope"))


def test_run_fields_and_dump_arrays(toy):
    f = dm.run_fields(dm.flux_integrator(toy.library, nmin=NMIN))
    assert f == dict(method="flux", name="flux", lamfix=False, smooth=0.0, nmin=NMIN)
    f = dm.run_fields(dm.flux_integrator(toy.library, nmin=NMIN, smooth=335.0, corr=toy.corr))
    assert f == dict(method="flux", name="flux_lamfix_sm335", lamfix=True, smooth=335.0, nmin=NMIN)
    assert dm.run_fields(dm.flux_integrator(toy.library), name="mine")["name"] == "mine"
    with pytest.raises(ValueError, match="plain directory name"):
        dm.run_fields(dm.flux_integrator(toy.library), name="a/b")
    with pytest.raises(ValueError, match="'method'"):
        dm.run_fields(types.SimpleNamespace(t=np.arange(3.0)))
    imu_like = types.SimpleNamespace(method="imu", t=np.arange(3.0))
    assert dm.run_fields(imu_like) == dict(method="imu", name="imu", lamfix=False, smooth=0.0, nmin=0)
    # legacy rule: nmin only for flux (fw_disc_dumps.py: a.nmin if method == "flux" else 0), whatever node_params say
    imu_nodes = types.SimpleNamespace(method="imu", t=np.arange(3.0), node_params=dict(nmin=20, smooth=0.0, corr=False))
    assert dm.run_fields(imu_nodes) == dict(method="imu", name="imu", lamfix=False, smooth=0.0, nmin=0)
    integ = dm.flux_integrator(toy.library, nmin=NMIN)
    smp = dm.load_sample(toy.samples, 100)
    MU, TN, PN = sph.project_los(toy.theta, toy.phi, "thompson2024")
    arrays = dm.dump_arrays(dm.disc_dump(integ, smp, MU, TN, PN, lref=LREF), dm.run_fields(integ),
                            extra=dict(note=np.array("x")))
    assert tuple(arrays)[:len(dm.DUMP_KEYS)] == dm.DUMP_KEYS and list(arrays)[-1] == "note"
    assert arrays["F"].dtype == np.float32 and arrays["n_clip"].dtype == np.int64
    with pytest.raises(ValueError, match="extra field names"):
        dm.dump_arrays(dm.disc_dump(integ, smp, MU, TN, PN, lref=LREF), dm.run_fields(integ), extra=dict(F=1))


# ----------------------------------------------------------------------------------------------
# the driver against the frozen legacy code
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_dump_files_match_legacy(toy, tmp_path, monkeypatch, variant):
    """Per-dump files byte for byte those of the frozen fw_disc_dumps.py process() (flux, smoothed, lamfix nodes;
    the legacy name rule); then the time series byte for byte that of the frozen fw_disc_collect.py."""
    legacy = _legacy_dump_files(toy, tmp_path / "legacy", variant, monkeypatch)
    s = _run(toy, tmp_path / "ours", variant)
    assert s["name"] == variant and s["done"] == DUMPS and s["fields"]["lamfix"] == VARIANTS[variant][2]
    for d in DUMPS:
        _assert_npz_equal(os.path.join(legacy, "d{:04d}.npz".format(d)), dm.dump_path(tmp_path / "ours", variant, d))
    ts_legacy = _legacy_collect(tmp_path / "ours", variant, DUMPS[0], DUMPS[-1])
    ts = dm.collect_timeseries(str(tmp_path / "ours"), variant, dumps=DUMPS, out=str(tmp_path / "ours_ts.npz"))
    _assert_npz_equal(ts_legacy, ts)
    with np.load(ts) as z:
        assert z.files == list(dm.TIMESERIES_KEYS)
        assert z["n_lo"].dtype == np.float64 and z["n_clip"].dtype == np.int64     # legacy dtypes


def test_fork_spawn_serial_identical(toy, tmp_path):
    """Serial, fork (maxtasksperchild 1: workers replaced; workers' malloc untuned) and spawn runs write identical
    files; spawn's temporary projection files are removed."""
    ref = _run(toy, tmp_path / "serial", "flux_sm150", nproc=1)
    fork = _run(toy, tmp_path / "fork", "flux_sm150", nproc=2, start_method="fork", maxtasksperchild=1,
                tune_malloc=False)
    spill = tmp_path / "spill"
    spill.mkdir()
    spawn = _run(toy, tmp_path / "spawn", "flux_sm150", nproc=3, start_method="spawn", tmpdir=str(spill), timeout=300)
    assert os.listdir(str(spill)) == []
    assert fork["start_method"] == "fork" and spawn["start_method"] == "spawn" and ref["start_method"] is None
    for s in (ref, fork, spawn):
        assert sorted(s["done"]) == DUMPS and len(s["results"]) == len(DUMPS)
    for f in _files(tmp_path / "serial" / "flux_sm150"):
        a = str(tmp_path / "serial" / "flux_sm150" / f)
        _assert_npz_equal(a, str(tmp_path / "fork" / "flux_sm150" / f))
        _assert_npz_equal(a, str(tmp_path / "spawn" / "flux_sm150" / f))
    assert {r["pid"] for r in fork["results"]} - {os.getpid()}                    # computed in workers


_BLAS_SCRIPT = r'''
import json, os, sys
sys.path.insert(0, {root!r})
import numpy as np
from ppmpy.synspec import dumps as dm, library as lb, parallel as par, sphere as sph
from ppmpy.synspec.spectral import LineSet


def run(variant, kw, out, **opt):
    theta, phi = sph.fibonacci_sphere({n})
    return dm.run_disc_dumps({dumps}, {samples!r}, out, None, dm.flux_integrator, ({library!r},), theta, phi,
                             "thompson2024", lref=LineSet({lines!r}, {lref!r}), factory_kwargs=kw, **opt)


if __name__ == "__main__":
    res = dict(before=par.blas_threads())
    lib = lb.FluxLibrary.load({library!r})
    corr = np.load({corr!r})["corr"]
    # the problem: lib_nodes' BLAS sums with smoothing / a correction depend on the thread count
    raw = [dm._sha256(n.prof) for n in (lb.lib_nodes(lib, nmin={nmin}, smooth=150.0),
                                         lb.lib_nodes(lib, nmin={nmin}, corr=corr))]
    with dm._blas_limit(1) as old:
        res["inside"] = par.blas_threads()
        res["old"] = old
        one = [dm._sha256(n.prof) for n in (lb.lib_nodes(lib, nmin={nmin}, smooth=150.0),
                                             lb.lib_nodes(lib, nmin={nmin}, corr=corr))]
    res["after_limit"] = par.blas_threads()
    res["sensitive"] = raw != one
    res["integ"] = [dm._integ_info(dm.flux_integrator({library!r}, nmin={nmin}, **kw))["sha256"]
                    for kw in (dict(smooth=150.0), dict(corr={corr!r}))]
    out = sys.argv[1]
    for sub, variant, kw, opt in (("spawn", "flux_sm150", dict(smooth=150.0), dict(nproc=2, start_method="spawn")),
                                  ("spawn", "flux_lamfix", dict(corr={corr!r}), dict(nproc=2, start_method="spawn")),
                                  ("fork", "flux_sm150", dict(smooth=150.0), dict(nproc=2, start_method="fork")),
                                  ("serial", "flux_sm150", dict(smooth=150.0), dict(nproc=1))):
        s = run(variant, dict(kw, nmin={nmin}), os.path.join(out, sub), timeout=300, **opt)
        assert s["name"] == variant and sorted(s["done"]) == {dumps}, s
    res["after_runs"] = par.blas_threads()
    print("RESULT " + json.dumps(res))
'''


def test_blas_threads_do_not_matter(toy, tmp_path):
    """In a process whose OpenBLAS has 4 threads (OMP_NUM_THREADS removed, OPENBLAS_NUM_THREADS=4): the integrator
    equals the single-threaded one, spawn runs of the smoothed and lamfix variants work (the workers' integrator
    matches the parent's), and spawn, fork and serial files equal those of a single-threaded process byte for byte;
    the thread counts are restored afterwards."""
    script = tmp_path / "blas_script.py"
    script.write_text(_BLAS_SCRIPT.format(root=ROOT, n=N_TOY, dumps=DUMPS, samples=toy.samples, library=toy.library,
                                          lines=LINES, lref=LREF.tolist(), corr=toy.corr, nmin=NMIN))
    env = {k: v for k, v in os.environ.items() if k not in par.THREAD_ENV_VARS}
    env.update(OPENBLAS_NUM_THREADS="4")
    t0 = time.time()
    res = subprocess.run([sys.executable, str(script), str(tmp_path / "threads")], capture_output=True, text=True,
                         timeout=600, env=env, cwd=str(tmp_path))
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-5000:]
    r = json.loads(next(x for x in res.stdout.splitlines() if x.startswith("RESULT "))[7:])
    print("BLAS threads", r["before"], "sensitive", r["sensitive"], "{:.1f} s".format(time.time() - t0))
    if not r["before"] or set(r["before"].values()) != {4}:
        pytest.skip("cannot run OpenBLAS with 4 threads here: {}".format(r["before"]))
    assert set(r["inside"].values()) == {1} and r["old"] == r["before"]
    assert r["after_limit"] == r["before"] and r["after_runs"] == r["before"]       # restored
    # the reference: this process (single-threaded in the container; run_disc_dumps limits it anyway)
    with dm._blas_limit(1):
        ref_integ = [dm._integ_info(dm.flux_integrator(toy.library, nmin=NMIN, **kw))["sha256"]
                     for kw in (dict(smooth=150.0), dict(corr=toy.corr))]
    assert r["integ"] == ref_integ
    for variant in ("flux_sm150", "flux_lamfix"):
        ref = _run(toy, tmp_path / "ref", variant)
        assert sorted(ref["done"]) == DUMPS
        subs = ("spawn", "fork", "serial") if variant == "flux_sm150" else ("spawn",)
        for sub in subs:
            for f in _files(tmp_path / "ref" / variant):
                _assert_npz_equal(str(tmp_path / "ref" / variant / f), str(tmp_path / "threads" / sub / variant / f))
        rec = dm.read_run_record(str(tmp_path / "threads" / "spawn" / variant))
        assert rec["params"] == dm.read_run_record(str(tmp_path / "ref" / variant))["params"]


def test_restart_overwrite_and_stale_tmp(toy, tmp_path):
    """Existing files are skipped (untouched), a second run with nothing to do does not build the integrator,
    overwrite recomputes (new file, same bytes); temporaries of killed writers are reported, not used."""
    out = tmp_path / "run"
    s1 = _run(toy, out, dumps=DUMPS[:3], name="flux")
    assert s1["done"] == DUMPS[:3] and s1["skipped"] == []
    st = {d: os.stat(dm.dump_path(out, "flux", d)) for d in DUMPS[:3]}
    sha = {d: _sha(dm.dump_path(out, "flux", d)) for d in DUMPS[:3]}
    open(str(out / "flux" / "d0104.tmp999999.npz"), "wb").close()
    s2 = _run(toy, out)
    assert s2["todo"] == DUMPS[3:] and s2["done"] == DUMPS[3:] and s2["skipped"] == DUMPS[:3]
    assert s2["stale_tmp"] == ["d0104.tmp999999.npz"]
    for d in DUMPS[:3]:
        now = os.stat(dm.dump_path(out, "flux", d))
        assert (now.st_ino, now.st_mtime_ns) == (st[d].st_ino, st[d].st_mtime_ns)
    calls = []

    def factory(*a, **k):
        calls.append(1)
        return dm.flux_integrator(*a, **k)

    s3 = dm.run_disc_dumps(DUMPS, toy.samples, str(out), "flux", factory, (toy.library,), toy.theta, toy.phi,
                           "thompson2024", lref=LINESET, factory_kwargs=dict(nmin=NMIN))
    assert s3["todo"] == [] and s3["skipped"] == DUMPS and calls == []
    assert s3["fields"] is None and s3["nproc"] == 0 and s3["run_file"] is None    # nothing built
    s4 = _run(toy, out, dumps=DUMPS[:2], overwrite=True)
    assert s4["done"] == DUMPS[:2]
    for d in DUMPS[:2]:
        assert os.stat(dm.dump_path(out, "flux", d)).st_ino != st[d].st_ino
        assert _sha(dm.dump_path(out, "flux", d)) == sha[d]
    ts = dm.collect_timeseries(str(out), "flux")                     # the temporary is not a dump
    with np.load(ts) as z:
        np.testing.assert_array_equal(z["dumps"], DUMPS)


def test_run_record_refuses_mixing(toy, tmp_path):
    """A run into a directory of another configuration raises: by the run record, or (legacy directories without
    one) by the constants of an existing per-dump file; other lines of sight or lines are caught by the record."""
    out = tmp_path / "run"
    _run(toy, out, dumps=DUMPS[:2], name="flux")
    rec = dm.read_run_record(str(out / "flux"))
    assert rec["kind"] == "synspec.disc_dumps.run" and rec["params"]["nmin"] == NMIN
    assert rec["params"]["los"] == los_thompson2024().tolist() and rec["info"]["names"] == LINES
    # the integrator by its inputs (file content, not path); the computed arrays' sha256 as information
    ip = rec["params"]["integrator"]
    assert ip["inputs"] == dict(factory="ppmpy.synspec.dumps.flux_integrator",
                                args=[dict(file_sha256=_sha(toy.library))], kwargs=dict(nmin=NMIN))
    assert "sha256" not in ip and len(rec["info"]["integrator_sha256"]) == 64
    assert rec["writer"]["pid"] == os.getpid() and rec["writer"]["host"] == socket.gethostname()
    with pytest.raises(ValueError, match="another configuration.*integrator"):
        _run(toy, out, "flux_sm150", name="flux")
    with pytest.raises(ValueError, match="another configuration.*los"):
        dm.run_disc_dumps(DUMPS, toy.samples, str(out), "flux", dm.flux_integrator, (toy.library,), toy.theta,
                          toy.phi, los_thompson2024()[:4], lref=LINESET, factory_kwargs=dict(nmin=NMIN))
    with pytest.raises(ValueError, match="another configuration.*lref"):
        dm.run_disc_dumps(DUMPS, toy.samples, str(out), "flux", dm.flux_integrator, (toy.library,), toy.theta,
                          toy.phi, "thompson2024", lref=LREF + 0.01, factory_kwargs=dict(nmin=NMIN))
    os.remove(str(out / "flux" / dm.RUN_FILE))
    with pytest.raises(ValueError, match="written by another configuration.*smooth"):
        _run(toy, out, "flux_sm150", name="flux")
    assert not os.path.exists(str(out / "flux" / dm.RUN_FILE))
    s = _run(toy, out, name="flux", log=print)                       # same constants: the directory is adopted
    assert s["done"] == DUMPS[2:] and dm.read_run_record(str(out / "flux"))["params"] == rec["params"]


def test_run_record_identifies_profiles_points_extras(toy, tmp_path):
    """Refused restarts: a library with the same nodes (t, counts, node_params) but other profiles (by the inputs
    fingerprint; and by the computed arrays' sha256 for a factory that cannot be fingerprinted, a local function);
    other points (permuted theta / phi); other values of constant extra fields. Same inputs with other bits (another
    CPU): accepted with a RuntimeWarning."""
    lib = lb.FluxLibrary.load(toy.library)
    other = _perturbed(lib).save(str(tmp_path / "other_library.npz"))
    assert np.array_equal(dm.flux_integrator(other, nmin=NMIN).t, dm.flux_integrator(toy.library, nmin=NMIN).t)
    out = tmp_path / "run"
    _run(toy, out, dumps=DUMPS[:2], name="flux", extra_fields=dict(radius_Mm=4050.0))
    with pytest.raises(ValueError, match="another configuration.*integrator"):
        dm.run_disc_dumps(DUMPS, toy.samples, str(out), "flux", dm.flux_integrator, (other,), toy.theta, toy.phi,
                          "thompson2024", lref=LINESET, factory_kwargs=dict(nmin=NMIN),
                          extra_fields=dict(radius_Mm=4050.0))
    with pytest.raises(ValueError, match=r"another configuration \(differs in \['points'\]\)"):
        dm.run_disc_dumps(DUMPS, toy.samples, str(out), "flux", dm.flux_integrator, (toy.library,), toy.theta[::-1],
                          toy.phi[::-1], "thompson2024", lref=LINESET, factory_kwargs=dict(nmin=NMIN),
                          extra_fields=dict(radius_Mm=4050.0))
    with pytest.raises(ValueError, match=r"another configuration \(differs in \['extra_fields'\]\)"):
        _run(toy, out, extra_fields=dict(radius_Mm=3800.0))
    with pytest.raises(ValueError, match="no objects"):
        _run(toy, tmp_path / "obj", extra_fields=dict(bad=np.array([{}], dtype=object)))
    # a local factory: no inputs fingerprint, the integrator's computed arrays decide
    libs = dict(path=toy.library)

    def local_factory(nmin):
        return dm.flux_integrator(libs["path"], nmin=nmin)

    out2 = tmp_path / "local"
    dm.run_disc_dumps(DUMPS[:2], toy.samples, str(out2), "flux", local_factory, (NMIN,), toy.theta, toy.phi,
                      "thompson2024", lref=LINESET)
    rec = dm.read_run_record(str(out2 / "flux"))
    assert "inputs" not in rec["params"]["integrator"]
    assert rec["params"]["integrator"]["sha256"] == rec["info"]["integrator_sha256"]
    assert rec["params"]["integrator"]["arrays"] == ["t", "fc", "P"]
    libs["path"] = other
    with pytest.raises(ValueError, match="another configuration.*integrator"):
        dm.run_disc_dumps(DUMPS, toy.samples, str(out2), "flux", local_factory, (NMIN,), toy.theta, toy.phi,
                          "thompson2024", lref=LINESET)
    assert _files(out2 / "flux") == ["d0100.npz", "d0101.npz"]
    # same inputs, other bits (as on another CPU): accepted, with a warning
    path = str(out / "flux" / dm.RUN_FILE)
    rec = _read_json(path)
    rec["info"]["integrator_sha256"] = "0" * 64
    with open(path, "w") as f:
        json.dump(rec, f)
    msgs = []
    with pytest.warns(RuntimeWarning, match="other bits"):
        s = _run(toy, out, dumps=DUMPS[:3], extra_fields=dict(radius_Mm=4050.0), log=msgs.append)
    assert s["done"] == [DUMPS[2]] and any("WARNING" in m for m in msgs)


def test_run_record_corrupted_stale_and_race(toy, tmp_path, monkeypatch):
    """A corrupted record: a clear ValueError. A record of another configuration whose run wrote nothing: replaced
    when its writer is gone (this host, process not running), refused (with a hint) when it may still run. Record
    creation is exclusive: a run that finds no record but loses the race compares against the winner's."""
    d = tmp_path / "c" / "flux"
    d.mkdir(parents=True)
    (d / dm.RUN_FILE).write_text("{not json")
    with pytest.raises(ValueError, match="corrupted.*remove it"):
        _run(toy, tmp_path / "c")
    (d / dm.RUN_FILE).write_text("[1, 2]")
    with pytest.raises(ValueError, match="not a run record"):
        dm.read_run_record(str(d))
    # a stale record: a first run of another configuration in this process, its results gone
    out = tmp_path / "stale"
    _run(toy, out, "flux_sm150", name="flux", dumps=DUMPS[:1])
    os.remove(dm.dump_path(out, "flux", DUMPS[0]))
    msgs = []
    s = _run(toy, out, name="flux", dumps=DUMPS[:2], log=msgs.append)
    assert s["done"] == DUMPS[:2] and any("replacing the run record" in m for m in msgs)
    assert dm.read_run_record(str(out / "flux"))["params"]["smooth"] == 0.0
    assert sorted(os.listdir(str(out / "flux"))) == [dm.RUN_FILE, "d0100.npz", "d0101.npz"]
    # writer alive (the parent of this process), or on another host: refused, with the hint
    out = tmp_path / "alive"
    _run(toy, out, "flux_sm150", name="flux", dumps=DUMPS[:1])
    os.remove(dm.dump_path(out, "flux", DUMPS[0]))
    path = str(out / "flux" / dm.RUN_FILE)
    rec = _read_json(path)
    for writer in (dict(host=socket.gethostname(), pid=os.getppid()), dict(host="elsewhere.invalid", pid=1)):
        rec["writer"].update(writer)
        with open(path, "w") as f:
            json.dump(rec, f)
        with pytest.raises(ValueError, match="no per-dump files yet.*remove"):
            _run(toy, out, name="flux")
    # exclusive creation
    p = str(tmp_path / "x.json")
    assert dm._create_json_excl(p, dict(a=1)) is True
    assert dm._create_json_excl(p, dict(a=2)) is False
    assert _read_json(p) == dict(a=1)
    assert not [f for f in os.listdir(str(tmp_path)) if ".tmp." in f]
    # the race: the record of another run (alive, nothing written yet) appears between our read (None) and our
    # creation: we compare with it instead of overwriting it
    real = dm.read_run_record
    calls = []

    def first_none(dirname, **kw):
        calls.append(1)
        return None if len(calls) == 1 else real(dirname, **kw)

    monkeypatch.setattr(dm, "read_run_record", first_none)
    with pytest.raises(ValueError, match="another configuration.*integrator.*no per-dump files yet"):
        _run(toy, out, name="flux")
    assert len(calls) == 2 and _files(out / "flux") == [] and _read_json(path) == rec


def test_rank_split(toy, tmp_path):
    """Dump i of the list goes to rank i % nranks (legacy --rank/--nranks); the union equals the serial run."""
    order = [103, 100, 105, 101, 104, 102]
    s0 = _run(toy, tmp_path / "ranks", dumps=order, nranks=2, rank=0)
    s1 = _run(toy, tmp_path / "ranks", dumps=order, nranks=2, rank=1, nproc=2)
    assert s0["dumps"] == [103, 105, 104] == par.split_items(order, 0, 2) and s0["done"] == [103, 105, 104]
    assert s1["dumps"] == [100, 101, 102] and sorted(s1["done"]) == [100, 101, 102]
    _run(toy, tmp_path / "all")
    for f in _files(tmp_path / "all" / "flux"):
        _assert_npz_equal(str(tmp_path / "all" / "flux" / f), str(tmp_path / "ranks" / "flux" / f))
    with pytest.raises(ValueError, match="rank"):
        _run(toy, tmp_path / "bad", rank=2, nranks=2)


def test_inputs_checked_before_work(toy, tmp_path):
    """Missing sample files, a wrong number of points, repeated dumps: errors before anything is written."""
    with pytest.raises(FileNotFoundError, match="no sample file"):
        _run(toy, tmp_path / "a", dumps=DUMPS + [99])
    with pytest.raises(ValueError, match="points"):
        dm.run_disc_dumps(DUMPS, toy.samples, str(tmp_path / "b"), "flux", dm.flux_integrator, (toy.library,),
                          toy.theta[:-1], toy.phi[:-1], "thompson2024", lref=LINESET)
    with pytest.raises(ValueError, match="repeated"):
        _run(toy, tmp_path / "c", dumps=[100, 100])
    with pytest.raises(ValueError, match="lref is required"):
        dm.run_disc_dumps(DUMPS, toy.samples, str(tmp_path / "d"), "flux", dm.flux_integrator, (toy.library,),
                          toy.theta, toy.phi, "thompson2024")
    with pytest.raises(ValueError, match="extra field names"):
        _run(toy, tmp_path / "e", extra_fields=dict(F0=1))
    with pytest.raises(ValueError, match=r"one reference wavelength per line \(3\), got 2"):
        dm.run_disc_dumps(DUMPS, toy.samples, str(tmp_path / "f"), "flux", dm.flux_integrator, (toy.library,),
                          toy.theta, toy.phi, "thompson2024", lref=LREF[:2], factory_kwargs=dict(nmin=NMIN))
    for d in "abcdef":                     # no results and no run record (a corrected rerun is not refused)
        assert not os.path.exists(str(tmp_path / d / "flux")) or os.listdir(str(tmp_path / d / "flux")) == []
    s = _run(toy, tmp_path / "f", dumps=DUMPS[:1])
    assert s["done"] == DUMPS[:1]


def test_meta_and_extra_fields(toy, tmp_path):
    """meta=True adds '_meta' (run parameters, sample file, timing) after the legacy members and the extras; the
    legacy members are unchanged. A callable extra_fields runs in the (spawn) workers."""
    ref = tmp_path / "ref"
    _run(toy, ref)
    out = tmp_path / "meta"
    msgs = []
    s = _run(toy, out, meta=True, extra_fields=_extra, nproc=2, start_method="spawn", timeout=300, log=msgs.append)
    assert sorted(s["done"]) == DUMPS
    assert any("spawn: each worker start builds the integrator" in m for m in msgs)
    for d in DUMPS:
        p = dm.dump_path(out, "flux", d)
        with np.load(p) as z, np.load(dm.dump_path(ref, "flux", d)) as zr:
            assert z.files == list(dm.DUMP_KEYS) + ["sample_path", "ew0", "_meta"]
            for k in dm.DUMP_KEYS:
                np.testing.assert_array_equal(z[k], zr[k], err_msg=k)
            assert str(z["sample_path"]) == "d{:04d}.npz".format(d)
            np.testing.assert_array_equal(z["ew0"], zr["diag_F"][:, :, 0].mean(axis=0))
        m = read_meta(p)
        assert m["kind"] == "synspec.disc_dump" and m["dump"] == d and m["params"]["name"] == "flux"
        assert m["inputs"]["sample"]["path"] == os.path.abspath(os.path.join(toy.samples, "d{:04d}.npz".format(d)))
        assert m["params"] == dm.read_run_record(str(out / "flux"))["params"]
        assert m["params"]["extra_fields"].startswith("callable ")
    # constant extras
    s = _run(toy, tmp_path / "const", dumps=DUMPS[:1], extra_fields=dict(radius_Mm=4050.0))
    with np.load(dm.dump_path(tmp_path / "const", "flux", DUMPS[0])) as z:
        assert z.files[-1] == "radius_Mm" and float(z["radius_Mm"]) == 4050.0


def test_watchdog_killed_worker(toy, tmp_path):
    """Workers killed while computing (os._exit): PoolStalled after the timeout, listing the dumps not finished."""
    t0 = time.time()
    with pytest.raises(par.PoolStalled) as e:
        dm.run_disc_dumps(DUMPS, toy.samples, str(tmp_path), "killed", _killer_factory, (toy.library,), toy.theta,
                          toy.phi, "thompson2024", lref=LINESET, nproc=2, start_method="fork", timeout=4.0)
    assert sorted(e.value.missing) == DUMPS and time.time() - t0 < 60
    assert _files(tmp_path / "killed") == []


def test_worker_integrator_must_match(toy, tmp_path):
    """A spawn worker whose factory builds another integrator than the parent's (e.g. its input file changed) fails
    in its initializer: WorkerInitError, nothing written; also when only the node profiles differ (same t and
    node_params)."""
    with pytest.raises(par.WorkerInitError, match=r"differs from the parent's \(in \['nn', 'node_params'"):
        dm.run_disc_dumps(DUMPS[:2], toy.samples, str(tmp_path), "flux", _differs_in_workers, (toy.library,),
                          toy.theta, toy.phi, "thompson2024", lref=LINESET, factory_kwargs=dict(nmin=NMIN), nproc=2,
                          start_method="spawn", timeout=300)
    assert _files(tmp_path / "flux") == []
    with pytest.raises(par.WorkerInitError, match=r"differs from the parent's \(in \['sha256'\]"):
        dm.run_disc_dumps(DUMPS[:2], toy.samples, str(tmp_path / "prof"), "flux", _prof_differs_in_workers,
                          (toy.library,), toy.theta, toy.phi, "thompson2024", lref=LINESET,
                          factory_kwargs=dict(nmin=NMIN), nproc=2, start_method="spawn", timeout=300)
    assert _files(tmp_path / "prof" / "flux") == []


def test_integ_info_contract(toy):
    """DiscFlux: t, fc and P hashed. Other integrators: fingerprint() (plus t), else every ndarray attribute, else
    ValueError."""
    base = dm.flux_integrator(toy.library, nmin=NMIN)
    i = dm._integ_info(base)
    assert i["arrays"] == ["t", "fc", "P"] and i["sha256"] == dm._sha256(base.t, base.fc, base.P)
    other = dm.flux_integrator(_perturbed(lb.FluxLibrary.load(toy.library)), nmin=NMIN)
    assert np.array_equal(other.t, base.t) and dm._integ_info(other)["sha256"] != i["sha256"]
    j = dm._integ_info(_ImuLike(base))
    assert j["arrays"] == ["t"] and j["custom"] == dict(base=i["sha256"]) and j["cls"].endswith("._ImuLike")
    plain = types.SimpleNamespace(t=base.t, prof=np.ones(3), grid=base.grid)
    k = dm._integ_info(plain)
    assert k["arrays"] == ["prof", "t"] and "custom" not in k
    with pytest.raises(ValueError, match="fingerprint"):
        dm._integ_info(types.SimpleNamespace(t=[1.0, 2.0], grid=base.grid))


def test_imu_like_integrator(toy, tmp_path):
    """A non-flux integrator with node_params nmin 20 writes nmin 0 (legacy rule; the production imu files), its
    method and name; the profiles equal those of the DiscFlux it wraps."""
    s = dm.run_disc_dumps(DUMPS[:2], toy.samples, str(tmp_path), None, _imu_like_factory, (toy.library,), toy.theta,
                          toy.phi, "thompson2024", lref=LINESET, factory_kwargs=dict(nmin=NMIN), nproc=2,
                          start_method="spawn", timeout=300)
    assert s["name"] == "imu" and s["fields"] == dict(method="imu", name="imu", lamfix=False, smooth=0.0, nmin=0)
    _run(toy, tmp_path / "flux", dumps=DUMPS[:2])
    for d in DUMPS[:2]:
        with np.load(dm.dump_path(tmp_path, "imu", d)) as z, np.load(dm.dump_path(tmp_path / "flux", "flux", d)) as r:
            assert z.files == r.files and int(z["nmin"]) == 0 and str(z["method"]) == "imu" and int(r["nmin"]) == NMIN
            for k in z.files:
                if k not in ("method", "name", "nmin"):
                    np.testing.assert_array_equal(z[k], r[k], err_msg=k)
    assert dm.read_run_record(str(tmp_path / "imu"))["params"]["integrator"]["custom"]["base"]


def test_lref_from_library(toy, tmp_path):
    """A library that records lref and its grid (FluxLibrary.build parameters): flux_integrator attaches lref; runs
    default to it (same files as with the LineSet given); a LineSet in another line order, or another grid, is
    refused before anything is written."""
    lib = lb.FluxLibrary.load(toy.library)
    lib.params = dict(lref=LREF.tolist(), ny=GRID.ny, y0=float(GRID.y[0]), y1=float(GRID.y[-1]))
    path = lib.save(str(tmp_path / "lib_lref.npz"))
    integ = dm.flux_integrator(path, nmin=NMIN)
    np.testing.assert_array_equal(integ.lref, LREF)
    with pytest.raises(ValueError, match="grid"):
        dm.flux_integrator(path, grid=VelocityGrid(dv=1.0, vmax=2600.0))
    smp = dm.load_sample(toy.samples, 100)
    MU, TN, PN = sph.project_los(toy.theta, toy.phi, "thompson2024")
    r = dm.disc_dump(integ, smp, MU, TN, PN)
    np.testing.assert_array_equal(r["diag_F"], dm.disc_dump(integ, smp, MU, TN, PN, lref=LINESET)["diag_F"])
    s = dm.run_disc_dumps(DUMPS[:2], toy.samples, str(tmp_path / "a"), "flux", dm.flux_integrator, (path,),
                          toy.theta, toy.phi, "thompson2024", factory_kwargs=dict(nmin=NMIN))
    assert s["done"] == DUMPS[:2]
    _run(toy, tmp_path / "b", dumps=DUMPS[:2])
    for d in DUMPS[:2]:
        _assert_npz_equal(dm.dump_path(tmp_path / "b", "flux", d), dm.dump_path(tmp_path / "a", "flux", d))
    swapped = LineSet([LINES[1], LINES[0], LINES[2]], LREF[[1, 0, 2]])
    with pytest.raises(ValueError, match="differs from the integrator's"):
        dm.disc_dump(integ, smp, MU, TN, PN, lref=swapped)
    with pytest.raises(ValueError, match="differs from the integrator's"):
        dm.run_disc_dumps(DUMPS, toy.samples, str(tmp_path / "c"), "flux", dm.flux_integrator, (path,), toy.theta,
                          toy.phi, "thompson2024", lref=swapped, factory_kwargs=dict(nmin=NMIN))
    assert not os.path.exists(str(tmp_path / "c" / "flux"))


def test_rebuild_cost_message():
    """Non-fork pools: the integrator builds are logged; a RuntimeWarning when their CPU time is large or a build
    takes more than half the watchdog timeout."""
    msgs = []
    dm._rebuild_cost(0.5, 1601, 8, 20, 900.0, "spawn", msgs.append)
    assert msgs == ["spawn: each worker start builds the integrator (0.5 s here): ~81 builds for 1601 dumps "
                    "(maxtasksperchild 20), ~40 s of CPU"]
    with pytest.warns(RuntimeWarning, match="build the integrator ~81 times"):
        dm._rebuild_cost(120.0, 1601, 8, 20, 900.0, "spawn", msgs.append)
    with pytest.warns(RuntimeWarning, match="half the watchdog timeout"):
        dm._rebuild_cost(500.0, 2, 1, 20, 900.0, "spawn", msgs.append)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dm._rebuild_cost(60.0, 1601, 8, None, 900.0, "spawn", msgs.append)    # one build per worker: 480 s
    assert msgs[-1].endswith("~8 builds for 1601 dumps (maxtasksperchild None), ~480 s of CPU")


# ----------------------------------------------------------------------------------------------
# intensity method: imu_integrator, batched lines of sight
# ----------------------------------------------------------------------------------------------
def _toy_imu_library(seed=0, nb=11, K=5, t0=36650.0, dT=65.0, empty=(3, 8), short=((2, 1, 4), (6, 0, 3))):
    """A toy intensity library (the members of the legacy imu_library_dT10.npz) on the M424 grid with 3 lines: nb bins
    of width dT from t0 (covering most of the toy samples' T_eff', the rest beyond the nodes), empty bins served by the
    nearest filled one (src), K rays s = p / R_max from 0 to 1 (some rows with nnode < K: NaN beyond, garbage
    intensities there), float32 intensities with limb darkening, a line depth / width / centre changing with T_eff'
    and mu (the 'imu' rows of fw_imu_library.py)."""
    rng = np.random.default_rng(seed)
    y = GRID.y
    edges = t0 + dT * np.arange(nb + 1)
    filled = np.array([b for b in range(nb) if b not in empty])
    src = np.array([filled[np.argmin(np.abs(filled - b))] for b in range(nb)])
    teff_rep = (edges[:-1] + rng.uniform(0.2, 0.8, nb) * dT)[src]
    nnode = np.full((nb, 3), K, np.int64)
    for b, j, n in short:
        nnode[b, j] = n
    nnode = nnode[src]
    s = np.full((nb, 3, K), np.nan)
    Il = np.full((nb, 3, K, y.size), 5.0, np.float32)
    Ic = np.full((nb, 3, K, y.size), 5.0, np.float32)
    for b in filled:
        x = (teff_rep[b] - 37000.0) / 300.0
        for j, (sig, amp, c0) in enumerate(((60.0, 0.3, 0.0), (140.0, 0.2, 5.0), (40.0, 0.4, -3.0))):
            n = nnode[src[b], j]
            s[b, j, :n] = np.concatenate([[0.0], np.sort(rng.uniform(0.1, 0.95, n - 2)), [1.0]])
            for k in range(n):
                mu = np.sqrt(1.0 - s[b, j, k] ** 2)
                ic = (1.0 + 0.1 * x + 0.01 * j) * (1.0 - 0.6 * (1.0 - mu)) * (1.0 + 1e-5 * y)
                d = amp * (1.0 + 0.2 * x) * (0.7 + 0.3 * mu) * np.exp(
                    -0.5 * ((y - c0 - 2.0 * x - 3.0 * mu) / (sig * (1.0 + 0.05 * x) * (1.0 + 0.1 * mu))) ** 2)
                Ic[b, j, k] = ic
                Il[b, j, k] = ic * (1.0 - d)
    count = np.where(np.isin(np.arange(nb), filled), 30.0, 0.0)
    return dict(edges=edges, tmean=0.5 * (edges[:-1] + edges[1:]), count=count, src=src,
                idx_rep=(5000 + np.arange(nb))[src], teff_rep=teff_rep, rmax=np.ones((nb, 3)), nnode=nnode,
                s=s[src], Ic=Ic[src], Il=Il[src])


LAZY_BITWISE = platform.machine().lower() in ("x86_64", "amd64")
"""fft='lazy' gives F bit for bit the default on x86-64 (ppmpy.synspec.disc.DiscImu notes); elsewhere to rounding."""


@pytest.fixture(scope="module")
def toy_imu(toy):
    """The toy intensity library saved uncompressed (legacy layout, no '_meta') next to the toy inputs."""
    path = str(toy.root / "imu_library_dT10.npz")
    if not os.path.exists(path):
        np.savez(path, **_toy_imu_library())
    return path


def _legacy_imu_dump_files(toy, imu_path, outroot, monkeypatch):
    """The per-dump files of the frozen fw_disc_dumps.py --method imu for the toy: its source lines (NAME, the
    projections, KEYS, process()) with INT = the frozen fw_disc.DiscImu of the toy intensity library, built and run
    with the loaded BLAS limited to 1 thread (the production container: OMP_NUM_THREADS=1)."""
    fd = _legacy_fw_disc()
    monkeypatch.setattr(fd, "SAMPLES", toy.samples)
    a = types.SimpleNamespace(overwrite=False, method="imu", lamfix=False, smooth=0.0, nmin=NMIN)
    ns = dict(np=np, fd=fd, os=os, time=time, a=a, pts=dict(theta=toy.theta, phi=toy.phi))
    _legacy_exec("fw_disc_dumps.py", "NAME = a.method", "NAME = a.method", ns)
    _legacy_exec("fw_disc_dumps.py", "rhat, that, phat = fd.unit_vectors", "del rhat, that, phat, pts", ns)
    _legacy_exec("fw_disc_dumps.py", "KEYS = [", "KEYS = [", ns)
    with dm._blas_limit(1):
        ns["INT"] = fd.DiscImu(imu_path)
        out = os.path.join(str(outroot), ns["NAME"])
        os.makedirs(out)
        ns["OUT"] = out
        _legacy_exec("fw_disc_dumps.py", "def process(d):", 'f"{time.time() - t0:.1f} s")', ns)
        for d in DUMPS:
            assert ns["process"](d).startswith("dump {}: EW".format(d))
    assert ns["NAME"] == "imu"
    return out


def _run_imu(toy, imu_path, outdir, name=None, dumps=DUMPS, kw=None, lref=LINESET, **opt):
    return dm.run_disc_dumps(dumps, toy.samples, str(outdir), name, dm.imu_integrator, (imu_path,), toy.theta,
                             toy.phi, "thompson2024", lref=lref, factory_kwargs=kw, **opt)


def test_imu_dump_files_match_legacy(toy, toy_imu, tmp_path, monkeypatch):
    """imu_integrator (default options) through run_disc_dumps writes the per-dump files of the frozen
    fw_disc_dumps.py --method imu (frozen fw_disc.DiscImu) byte for byte: serial, 'fork' (shared integrator) and
    'spawn' (each worker builds its own from the file and must match the parent's fingerprint), per line of sight and
    batched (integrate_los); the time series equals the frozen fw_disc_collect.py's."""
    legacy = _legacy_imu_dump_files(toy, toy_imu, tmp_path / "legacy", monkeypatch)
    runs = (("serial", dict(nproc=1)), ("batch", dict(nproc=1, batch=True)),
            ("lref", dict(nproc=1, kw=dict(lref=LINESET), lref=None)),
            ("fork", dict(nproc=2, start_method="fork", maxtasksperchild=2)),
            ("spawn", dict(nproc=2, start_method="spawn", maxtasksperchild=2, timeout=300)))
    for sub, opt in runs:
        s = _run_imu(toy, toy_imu, tmp_path / sub, **opt)
        assert s["name"] == "imu" and sorted(s["done"]) == DUMPS, sub
        assert s["fields"] == dict(method="imu", name="imu", lamfix=False, smooth=0.0, nmin=0)
        for d in DUMPS:
            _assert_npz_equal(os.path.join(legacy, "d{:04d}.npz".format(d)), dm.dump_path(tmp_path / sub, "imu", d))
        with np.load(dm.dump_path(tmp_path / sub, "imu", DUMPS[0])) as z:
            assert int(z["n_clip"].sum()) > 0 and int(z["n_lo"]) > 0 and int(z["n_hi"]) > 0
    ts_legacy = _legacy_collect(tmp_path / "spawn", "imu", DUMPS[0], DUMPS[-1])
    ts = dm.collect_timeseries(str(tmp_path / "spawn"), "imu", out=str(tmp_path / "ts.npz"))
    _assert_npz_equal(ts_legacy, ts)
    rec = dm.read_run_record(str(tmp_path / "spawn" / "imu"))
    integ = rec["params"]["integrator"]
    assert integ["cls"] == "ppmpy.synspec.disc.DiscImu"
    assert integ["inputs"]["factory"] == "ppmpy.synspec.dumps.imu_integrator"
    assert integ["inputs"]["args"] == [dict(file_sha256=_sha(toy_imu))]
    c = integ["custom"]
    assert (c["method"], c["fft"], c["dtype"], c["chunk"], c["lines"]) == ("imu", "precomputed", "float64", 128,
                                                                          [0, 1, 2])
    assert "lref" not in c and len(c["library_sha256"]) == 64             # lref: in the params only (labels the lines)
    assert rec["params"]["lref"] == LREF.tolist() and rec["info"]["names"] == LINES
    # lref given to the factory instead of the run: the same params (the same run; reviewer)
    rec2 = dm.read_run_record(str(tmp_path / "lref" / "imu"))
    assert "lref" not in rec2["params"]["integrator"]["inputs"]["kwargs"] and rec2["params"] == rec["params"]


def test_imu_lazy_runs(toy, toy_imu, tmp_path):
    """fft='lazy' (batched by default) and dtype='float32': serial = fork = spawn byte for byte, batched = per line of
    sight byte for byte; against the default run F to the float32 storage rounding (lazy float64: bit for bit on
    x86-64), the velocity moments and coverage exactly; the run record refuses to mix the modes in one directory."""
    ref = _run_imu(toy, toy_imu, tmp_path / "ref")
    assert ref["done"] == DUMPS
    lazy = dict(fft="lazy")
    out = {}
    for sub, opt in (("serial", dict(nproc=1)), ("perlos", dict(nproc=1, batch=False)),
                     ("fork", dict(nproc=2, start_method="fork")),
                     ("spawn", dict(nproc=2, start_method="spawn", timeout=300))):
        s = _run_imu(toy, toy_imu, tmp_path / sub, name="imu_lazy", kw=lazy, **opt)
        assert sorted(s["done"]) == DUMPS and s["name"] == "imu_lazy"
        out[sub] = [_sha(dm.dump_path(tmp_path / sub, "imu_lazy", d)) for d in DUMPS]
    assert out["serial"] == out["perlos"] == out["fork"] == out["spawn"]
    f32 = _run_imu(toy, toy_imu, tmp_path / "f32", name="imu_f32", kw=dict(fft="lazy", dtype="float32"))
    assert f32["done"] == DUMPS
    for d in DUMPS:
        with np.load(dm.dump_path(tmp_path / "ref", "imu", d)) as a, \
                np.load(dm.dump_path(tmp_path / "serial", "imu_lazy", d)) as b, \
                np.load(dm.dump_path(tmp_path / "f32", "imu_f32", d)) as c:
            assert np.abs(a["F"].astype(float) - b["F"]).max() <= (0.0 if LAZY_BITWISE else 6e-8)
            assert np.abs(a["F0"].astype(float) - b["F0"]).max() <= 6e-8
            assert np.abs(a["F"].astype(float) - c["F"]).max() <= 1e-6
            for k in ("vmean_w", "sigma_w", "n_clip", "n_lo", "n_hi", "wout", "node_range"):
                np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    rec = dm.read_run_record(str(tmp_path / "serial" / "imu_lazy"))["params"]["integrator"]["custom"]
    assert rec["fft"] == "lazy" and rec["dtype"] == "float64"
    with pytest.raises(ValueError, match="another configuration"):
        _run_imu(toy, toy_imu, tmp_path / "serial", name="imu_lazy", kw=dict(fft="precomputed"), overwrite=True)


def test_imu_integrator_lref(toy, toy_imu, tmp_path):
    """lref attached by imu_integrator (the legacy library records none): a LineSet names the lines (labels kept), an
    array does not; it survives pickling (recipe), defaults disc_dump's lref and enters the fingerprint; another
    length, or values / names other than those a library records, are refused."""
    import pickle
    MU, TN, PN = sph.project_los(toy.theta, toy.phi, "thompson2024")
    smp = dm.load_sample(toy.samples, DUMPS[0])
    bare = dm.imu_integrator(toy_imu)
    assert bare.lines is None and bare.lref is None and bare.fingerprint()["lref"] is None
    with pytest.raises(ValueError, match="lref is required"):
        dm.disc_dump(bare, smp, MU, TN, PN)
    labelled = LineSet(LINES, LREF, labels=["a", "b", "c"])
    integ = dm.imu_integrator(toy_imu, lref=labelled)
    assert integ.lines.names == LINES and integ.lines.labels == ["a", "b", "c"] and integ.names == LINES
    np.testing.assert_array_equal(integ.lref, LREF)
    assert integ.fingerprint()["lref"] == LREF.tolist()
    r = dm.disc_dump(integ, smp, MU, TN, PN)
    r_ref = dm.disc_dump(bare, smp, MU, TN, PN, lref=LINESET)
    for k in ("F", "F0", "diag_F", "diag_F0", "vmean_w", "sigma_w"):
        np.testing.assert_array_equal(r[k], r_ref[k], err_msg=k)
    back = pickle.loads(pickle.dumps(integ))
    assert back.names == LINES and np.array_equal(back.lref, LREF) and back.fingerprint() == integ.fingerprint()
    arr = dm.imu_integrator(toy_imu, lref=LREF)
    assert arr.lines is None and arr.names is None and np.array_equal(arr.lref, LREF)
    with pytest.raises(ValueError, match="wavelengths"):
        dm.imu_integrator(toy_imu, lref=LREF[:2])
    # a library that records its lines (members lref / lines): a matching lref is accepted, others are refused
    with np.load(toy_imu) as z:
        rec = {k: z[k] for k in z.files}
    path = str(tmp_path / "imu_lines.npz")
    np.savez(path, lref=LREF, lines=np.array(LINES), **rec)
    own = dm.imu_integrator(path, lref=LREF + 1e-14)
    assert own.names == LINES and np.array_equal(own.lref, LREF)
    with pytest.raises(ValueError, match="differs from the intensity library's"):
        dm.imu_integrator(path, lref=LREF[[1, 0, 2]])
    with pytest.raises(ValueError, match="line names"):
        dm.imu_integrator(path, lref=LineSet(["x", "y", "z"], LREF))
    # a library that records only the names (member lines): a LineSet must have those names in that order (reviewer:
    # other or permuted names relabelled the lines); an array lref is attached under the library's names
    names_only = str(tmp_path / "imu_names.npz")
    np.savez(names_only, lines=np.array(LINES), **rec)
    nm = dm.imu_integrator(names_only, lref=LREF)
    assert nm.names == LINES and nm.lines.names == LINES and np.array_equal(nm.lref, LREF)
    nm = dm.imu_integrator(names_only, lref=labelled)
    assert nm.names == LINES and nm.lines.labels == ["a", "b", "c"]
    for bad in (LineSet(["X", "Y", "Z"], LREF), LineSet([LINES[1], LINES[0], LINES[2]], LREF)):
        with pytest.raises(ValueError, match="line names"):
            dm.imu_integrator(names_only, lref=bad)


def test_imu_integrator_lines_by_name(toy, toy_imu, tmp_path):
    """Lines selected by name: with the legacy library (no names) through the names of a LineSet lref (reviewer: they
    failed before the factory attached the names), with a library that names its lines through its names; unknown
    names, names without a LineSet, a LineSet of another length are refused. A per-line run equals the full run on its
    line."""
    one = dm.imu_integrator(toy_imu, lines=["HEII4200"], lref=LINESET, fft="lazy")
    assert one.built == (1,) and one.names == LINES and np.array_equal(one.lref, LREF)
    assert dm.imu_integrator(toy_imu, lines="HEI4922", lref=LINESET).built == (2,)
    assert dm.imu_integrator(toy_imu, lines=["HEI4922", 0], lref=LINESET, fft="lazy").built == (0, 2)
    with pytest.raises(ValueError, match="unknown line 'HEII9999'"):
        dm.imu_integrator(toy_imu, lines=["HEII9999"], lref=LINESET, fft="lazy")
    with pytest.raises(ValueError, match="records no line names"):
        dm.imu_integrator(toy_imu, lines=["HEII4200"], lref=LREF, fft="lazy")
    with pytest.raises(ValueError, match="names 2 lines"):
        dm.imu_integrator(toy_imu, lines=["HEII4200"], lref=LineSet(LINES[:2], LREF[:2]), fft="lazy")
    with np.load(toy_imu) as z:
        rec = {k: z[k] for k in z.files}
    path = str(tmp_path / "imu_names.npz")
    np.savez(path, lines=np.array(LINES), **rec)
    assert dm.imu_integrator(path, lines=["HEI4922"], fft="lazy").built == (2,)
    with pytest.raises(ValueError, match="line names"):
        dm.imu_integrator(path, lines=["HEI4922"], lref=LineSet(["A", "B", "C"], LREF), fft="lazy")
    MU, TN, PN = sph.project_los(toy.theta, toy.phi, "thompson2024")
    smp = dm.load_sample(toy.samples, DUMPS[0])
    full = dm.disc_dump(dm.imu_integrator(toy_imu, lref=LINESET, fft="lazy"), smp, MU, TN, PN, batch=True)
    part = dm.disc_dump(one, smp, MU, TN, PN, batch=True)
    np.testing.assert_array_equal(part["F"][:, 1], full["F"][:, 1])
    assert np.all(np.isnan(part["F"][:, [0, 2]])) and dm.run_fields(one)["name"] == "imu_lazy_l1"


def test_imu_mode_names_and_legacy_adoption(toy, toy_imu, tmp_path):
    """run_fields gives every DiscImu mode its own default name ('imu' only for the legacy options); a per-dump
    directory without a run record (as the production imu/ of the legacy driver) is adopted by the legacy mode only
    (reviewer: a lazy float32 run with name=None adopted it and mixed its files in), and the refused run writes
    nothing."""
    for kw, want in ((dict(), "imu"), (dict(fft="lazy"), "imu_lazy"), (dict(dtype="float32"), "imu_f32"),
                     (dict(fft="lazy", dtype="f4"), "imu_lazy_f32"), (dict(fft="lazy", chunk=64), "imu_lazy_c64"),
                     (dict(chunk=128, lines=[0, 1, 2]), "imu"), (dict(fft="lazy", lines=[0, 2]), "imu_lazy_l0-2")):
        integ = dm.imu_integrator(toy_imu, **kw)
        assert dm.run_fields(integ)["name"] == want, kw
        assert dm.run_fields(integ, "mine")["name"] == "mine"
        assert (want == "imu") == (dm._imu_mode(integ) == [])
    assert dm._imu_mode(dm.flux_integrator(toy.library, nmin=NMIN)) == []
    # a legacy directory: files of the default mode, no run record
    s = _run_imu(toy, toy_imu, tmp_path, dumps=DUMPS[:2])
    ddir = os.path.join(str(tmp_path), "imu")
    os.remove(os.path.join(ddir, dm.RUN_FILE))
    before = {f: _sha(os.path.join(ddir, f)) for f in os.listdir(ddir)}
    for kw in (dict(fft="lazy", dtype="float32"), dict(fft="lazy"), dict(dtype="float32"), dict(chunk=64),
               dict(lines=[1])):
        with pytest.raises(ValueError, match="not the legacy ones"):
            _run_imu(toy, toy_imu, tmp_path, name="imu", kw=kw)
    assert {f: _sha(os.path.join(ddir, f)) for f in os.listdir(ddir)} == before          # nothing written
    s = _run_imu(toy, toy_imu, tmp_path, kw=dict(fft="lazy", dtype="float32"), dumps=DUMPS[:2])
    assert s["name"] == "imu_lazy_f32" and s["done"] == DUMPS[:2]                        # its own directory
    logs = []
    s = _run_imu(toy, toy_imu, tmp_path, log=logs.append)                                 # the legacy mode adopts it
    assert s["name"] == "imu" and s["done"] == DUMPS[2:] and any("adopting the directory" in x for x in logs)
    assert all(_sha(os.path.join(ddir, f)) == h for f, h in before.items())


def test_imu_lref_restart(toy, toy_imu, tmp_path):
    """lref is not part of the integrator's identity (reviewer): a run with lref given to run_disc_dumps continues
    with lref given to the factory (and the other way round) in the same directory, byte for byte the files of one
    run; lref values other than the recorded ones are still refused (params lref)."""
    _run_imu(toy, toy_imu, tmp_path / "ref")
    a = _run_imu(toy, toy_imu, tmp_path / "mix", dumps=DUMPS[:2])
    b = _run_imu(toy, toy_imu, tmp_path / "mix", dumps=DUMPS[:4], kw=dict(lref=LINESET), lref=None)
    c = _run_imu(toy, toy_imu, tmp_path / "mix", kw=dict(lref=LREF), lref=LINESET)
    assert (a["done"], b["done"], c["done"]) == (DUMPS[:2], DUMPS[2:4], DUMPS[4:])
    for d in DUMPS:
        _assert_npz_equal(dm.dump_path(tmp_path / "ref", "imu", d), dm.dump_path(tmp_path / "mix", "imu", d))
    other = LineSet(LINES, LREF + 0.01)
    with pytest.raises(ValueError, match="another configuration"):
        _run_imu(toy, toy_imu, tmp_path / "mix", kw=dict(lref=other), lref=None, overwrite=True)


def test_cpu_budget_warning(toy, tmp_path, monkeypatch):
    """The CPU-time limit of an in-process run: a RuntimeWarning (and log line) when the CPU used plus the first
    dump's CPU time for every remaining dump exceeds the limit (lazy M424: ~11 s per dump, 3600 s limit: killed after
    ~300 dumps); none within it or without a limit; run_disc_dumps checks once, after its first dump."""
    logs = []
    with pytest.warns(RuntimeWarning, match=r"killed after ~326 more"):
        msg = dm._cpu_budget_warning(11.0, 1600, logs.append, limit=3600.0, used=10.0)
    assert logs == ["WARNING: " + msg] and "nproc >= 2" in msg
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert dm._cpu_budget_warning(11.0, 300, logs.append, limit=3600.0, used=10.0) is None
    calls = []
    monkeypatch.setattr(dm, "_cpu_budget_warning", lambda per, left, _log: calls.append((per, left)))
    s = dm.run_disc_dumps(DUMPS[:3], toy.samples, str(tmp_path), "flux", dm.flux_integrator, (toy.library,),
                          toy.theta, toy.phi, "thompson2024", lref=LINESET, factory_kwargs=dict(nmin=NMIN))
    assert s["done"] == DUMPS[:3] and len(calls) == 1 and calls[0][1] == 2 and calls[0][0] >= 0.0


def test_imu_integrator_imulibrary_spawn(toy, toy_imu, tmp_path):
    """imu_integrator of an ImuLibrary memory-mapped from its file (pickled to 'spawn' workers as its path; the run
    record falls back to the integrator's fingerprint) writes the files of the path run byte for byte."""
    lib = lb.ImuLibrary.load(toy_imu)
    ref = _run_imu(toy, toy_imu, tmp_path / "path", dumps=DUMPS[:3])
    s = dm.run_disc_dumps(DUMPS[:3], toy.samples, str(tmp_path / "obj"), None, dm.imu_integrator, (lib,), toy.theta,
                          toy.phi, "thompson2024", lref=LINESET, nproc=2, start_method="spawn", timeout=300)
    assert ref["done"] == sorted(s["done"]) == DUMPS[:3]                  # done: in the order the workers finish
    for d in DUMPS[:3]:
        _assert_npz_equal(dm.dump_path(tmp_path / "path", "imu", d), dm.dump_path(tmp_path / "obj", "imu", d))
    rec = dm.read_run_record(str(tmp_path / "obj" / "imu"))["params"]["integrator"]
    assert "inputs" not in rec and rec["custom"]["library_sha256"] == dm.read_run_record(
        str(tmp_path / "path" / "imu"))["params"]["integrator"]["custom"]["library_sha256"]


def test_disc_dump_batch(toy):
    """disc_dump(batch=True) equals the per-line-of-sight loop bit for bit for DiscFlux (its integrate_los); the
    default (None) batches only lazy integrators; batch=True without integrate_los is refused."""
    integ = dm.flux_integrator(toy.library, nmin=NMIN)
    MU, TN, PN = sph.project_los(toy.theta, toy.phi, "thompson2024")
    smp = dm.load_sample(toy.samples, DUMPS[1])
    a = dm.disc_dump(integ, smp, MU, TN, PN, lref=LINESET)
    b = dm.disc_dump(integ, smp, MU, TN, PN, lref=LINESET, batch=True)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    assert dm._use_batch(integ, None) is False and dm._use_batch(integ, True) is True
    assert dm._use_batch(types.SimpleNamespace(fft="lazy", integrate_los=lambda *x, **y: None), None) is True
    with pytest.raises(ValueError, match="integrate_los"):
        dm.disc_dump(_ImuLike(integ), smp, MU, TN, PN, lref=LINESET, batch=True)
    with pytest.raises(ValueError, match="integrate_los"):
        dm.run_disc_dumps(DUMPS, toy.samples, str(toy.root / "never"), "x", _imu_like_factory, (toy.library,),
                          toy.theta, toy.phi, "thompson2024", lref=LINESET, batch=True)
    assert not os.path.exists(str(toy.root / "never" / "x"))


# ----------------------------------------------------------------------------------------------
# collect
# ----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def toy_run(toy, tmp_path_factory):
    """A finished toy flux run (meta=False) in a module tmp dir."""
    root = tmp_path_factory.mktemp("dumps_run")
    _run(toy, root)
    return root


def test_collect_rows_and_missing(toy_run, tmp_path):
    """Every member: the per-dump values stacked in dump order; missing dumps refused unless allow_missing; dumps=None
    takes the files present; the output is written atomically (no temporary left)."""
    full = dm.collect_timeseries(str(toy_run), "flux", out=str(tmp_path / "full.npz"))
    sub = [104, 101, 103]
    part = dm.collect_timeseries(str(toy_run), "flux", dumps=sub, out=str(tmp_path / "sub.npz"))
    with np.load(full) as zf, np.load(part) as zp:
        np.testing.assert_array_equal(zf["dumps"], DUMPS)
        np.testing.assert_array_equal(zp["dumps"], sub)
        rows = [DUMPS.index(d) for d in sub]
        for d, i in zip(sub, rows):
            with np.load(dm.dump_path(toy_run, "flux", d)) as r:
                for k in ("F", "F0", "diag_F", "diag_F0", "vmean_w", "sigma_w", "wout", "n_clip", "t_s", "n_lo",
                          "n_hi", "teff_mean", "teff_std", "teff_min", "teff_max"):
                    np.testing.assert_array_equal(zf[k][i], r[k], err_msg=k)
                    np.testing.assert_array_equal(zp[k][sub.index(d)], r[k], err_msg=k)
        for k in ("Y", "LREF", "los", "method", "name", "diag_vwin", "diag_keys", "node_range"):
            np.testing.assert_array_equal(zf[k], zp[k], err_msg=k)
        np.testing.assert_array_equal(zf["Y"], GRID.y)
        np.testing.assert_array_equal(zf["LREF"], LREF)
        np.testing.assert_array_equal(zf["los"], los_thompson2024())
    assert sorted(os.listdir(str(tmp_path))) == ["full.npz", "sub.npz"]
    with pytest.raises(FileNotFoundError, match="missing"):
        dm.collect_timeseries(str(toy_run), "flux", dumps=DUMPS + [106, 107], out=str(tmp_path / "x.npz"))
    ok = dm.collect_timeseries(str(toy_run), "flux", dumps=[99] + DUMPS, out=str(tmp_path / "x.npz"),
                               allow_missing=True)
    assert _sha(ok) == _sha(full)
    with pytest.raises(FileNotFoundError, match="no per-dump files"):
        dm.collect_timeseries(str(toy_run), "flux", dumps=[99], out=str(tmp_path / "y.npz"), allow_missing=True)


def test_collect_axes_sources(toy, toy_run, tmp_path):
    """Y, LREF, los come from the run record or the per-dump '_meta'; without either (legacy directories) they must
    be given; given ones must agree with the recorded ones."""
    with pytest.raises(ValueError, match="differs from the one recorded"):
        dm.collect_timeseries(str(toy_run), "flux", los=-los_thompson2024(), out=str(tmp_path / "a.npz"))
    with pytest.raises(ValueError, match="differs from the one recorded"):
        dm.collect_timeseries(str(toy_run), "flux", grid=VelocityGrid(dv=2.0, vmax=5400.0),
                              out=str(tmp_path / "a.npz"))
    same = dm.collect_timeseries(str(toy_run), "flux", los="thompson2024", grid=GRID, lref=LINESET,
                                 out=str(tmp_path / "same.npz"))
    # a legacy directory: no run record, no '_meta'
    legacy = tmp_path / "legacy"
    shutil.copytree(str(toy_run / "flux"), str(legacy / "flux"))
    os.remove(str(legacy / "flux" / dm.RUN_FILE))
    with pytest.raises(ValueError, match="unknown.*pass grid=, lref= and los="):
        dm.collect_timeseries(str(legacy), "flux", out=str(tmp_path / "b.npz"))
    got = dm.collect_timeseries(str(legacy), "flux", los="thompson2024", grid=GRID, lref=LREF,
                                out=str(tmp_path / "b.npz"))
    assert _sha(got) == _sha(same)
    with pytest.raises(ValueError, match="do not fit the profiles"):
        dm.collect_timeseries(str(legacy), "flux", los="thompson2024", grid=GRID, lref=LREF[:2],
                              out=str(tmp_path / "c.npz"))
    # per-dump '_meta' only (record removed): the axes come from the files; meta=True records the run
    mrun = tmp_path / "metarun"
    _run(toy, mrun, meta=True)
    os.remove(str(mrun / "flux" / dm.RUN_FILE))
    got = dm.collect_timeseries(str(mrun), "flux", out=str(tmp_path / "m.npz"), meta=True)
    with np.load(got) as z, np.load(same) as zs:
        assert z.files == list(dm.TIMESERIES_KEYS) + ["_meta"]
        for k in dm.TIMESERIES_KEYS:
            np.testing.assert_array_equal(z[k], zs[k], err_msg=k)
    m = read_meta(got)
    assert m["kind"] == "synspec.disc_timeseries" and m["params"]["n_dumps"] == len(DUMPS)
    assert m["params"]["run"]["name"] == "flux" and m["params"]["missing"] == []


def test_collect_refuses_mixed_files(toy, toy_run, tmp_path):
    """A file of another configuration under the same name, or of another name / dump number, is refused."""
    other = tmp_path / "other"
    _run(toy, other, "flux_sm150", name="flux", dumps=DUMPS[:1])
    mixed = tmp_path / "mixed"
    shutil.copytree(str(toy_run / "flux"), str(mixed / "flux"))
    shutil.copy(dm.dump_path(other, "flux", DUMPS[0]), dm.dump_path(mixed, "flux", DUMPS[0]))
    with pytest.raises(ValueError, match="mixes runs"):
        dm.collect_timeseries(str(mixed), "flux", out=str(tmp_path / "a.npz"))
    shutil.copy(dm.dump_path(toy_run, "flux", DUMPS[1]), dm.dump_path(mixed, "flux", DUMPS[0]))
    with pytest.raises(ValueError, match="holds dump {}".format(DUMPS[1])):
        dm.collect_timeseries(str(mixed), "flux", out=str(tmp_path / "a.npz"))
    shutil.copy(dm.dump_path(toy_run, "flux", DUMPS[0]), dm.dump_path(mixed, "flux", DUMPS[0]))
    os.rename(str(mixed / "flux"), str(mixed / "renamed"))
    with pytest.raises(ValueError, match="of run 'flux'"):
        dm.collect_timeseries(str(mixed), "renamed", out=str(tmp_path / "a.npz"))
    assert not os.path.exists(str(tmp_path / "a.npz"))


def test_collect_canonical_names_and_grid_arg(toy_run, tmp_path):
    """Only files named exactly dNNNN.npz are dumps (a 'd100.npz' next to 'd0100.npz' is not a second dump 100, and
    alone it is not dump 100); grid may be a VelocityGrid, a dict of its arguments or the 1-D grid."""
    full = dm.collect_timeseries(str(toy_run), "flux", out=str(tmp_path / "full.npz"))
    copy = tmp_path / "copy"
    shutil.copytree(str(toy_run / "flux"), str(copy / "flux"))
    shutil.copy(str(copy / "flux" / "d0100.npz"), str(copy / "flux" / "d100.npz"))
    a = dm.collect_timeseries(str(copy), "flux", out=str(tmp_path / "a.npz"))
    assert _sha(a) == _sha(full)
    os.remove(str(copy / "flux" / "d0100.npz"))
    b = dm.collect_timeseries(str(copy), "flux", out=str(tmp_path / "b.npz"))
    with np.load(b) as z:
        np.testing.assert_array_equal(z["dumps"], DUMPS[1:])
    for g in (GRID, GRID.to_dict(), GRID.y):
        assert _sha(dm.collect_timeseries(str(toy_run), "flux", grid=g, out=str(tmp_path / "g.npz"))) == _sha(full)
    for bad in ("nope", dict(dv=1.0, nope=2), 5.0):
        with pytest.raises(ValueError, match="grid"):
            dm.collect_timeseries(str(toy_run), "flux", grid=bad, out=str(tmp_path / "g.npz"))


def test_collect_file_replaced_between_passes(toy_run, tmp_path, monkeypatch):
    """A per-dump file replaced between the first pass (small members) and the second (profiles): RuntimeError,
    nothing written (no output, no temporary)."""
    copy = tmp_path / "copy"
    shutil.copytree(str(toy_run / "flux"), str(copy / "flux"))
    real = dm._read_profiles
    done = []

    def replace_then_read(path, key, ident, shape):
        if not done:
            done.append(path)
            shutil.copy(path, path + ".new")
            os.replace(path + ".new", path)              # same content, another file
        return real(path, key, ident, shape)

    monkeypatch.setattr(dm, "_read_profiles", replace_then_read)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(RuntimeError, match="changed while the time series was collected"):
        dm.collect_timeseries(str(copy), "flux", out=str(out / "ts.npz"))
    assert done and os.listdir(str(out)) == []


def test_collect_messages(toy_run, tmp_path):
    """The legacy summary lines; the streamed residual rms and max equal the whole-array legacy numbers."""
    msgs = []
    ts = dm.collect_timeseries(str(toy_run), "flux", out=str(tmp_path / "ts.npz"), log=msgs.append)
    with np.load(ts) as z:
        F = z["F"]
        res = F - F.mean(axis=0, dtype=np.float64, keepdims=True)
        for j, name in enumerate(LINES):
            line = next(m for m in msgs if "] {}: EW".format(name) in m)
            want = "residual F - <F>_t: rms {:.1e}, max {:.1e}".format(res[:, :, j].std(),
                                                                        np.abs(res[:, :, j]).max())
            assert line.endswith(want), (line, want)
            assert "FWHM" in line and "<v> rms" in line
    assert any("clipped |v| > 400 km/s:" in m for m in msgs)
    assert any("6 of 6 dumps present" in m for m in msgs) and any(m.endswith("GB)") for m in msgs)
    # a legacy directory (no record, no '_meta'): the names of a LineSet and the vshift of a grid label the lines
    legacy = tmp_path / "legacy"
    shutil.copytree(str(toy_run / "flux"), str(legacy / "flux"))
    os.remove(str(legacy / "flux" / dm.RUN_FILE))
    for grid in (GRID, GRID.to_dict()):
        lm = []
        dm.collect_timeseries(str(legacy), "flux", los="thompson2024", grid=grid, lref=LINESET,
                              out=str(tmp_path / "l.npz"), log=lm.append)
        assert [m.split("] ")[1].split(":")[0] for m in lm if ": EW" in m] == LINES
        assert any("clipped |v| > 400 km/s:" in m for m in lm)
    lm = []
    dm.collect_timeseries(str(legacy), "flux", los="thompson2024", grid=GRID.y, lref=LREF, out=str(tmp_path / "l.npz"),
                          log=lm.append)
    assert any("] line0: EW" in m for m in lm) and any("clipped |v| > vshift km/s:" in m for m in lm)


# ----------------------------------------------------------------------------------------------
# M424 regressions
# ----------------------------------------------------------------------------------------------
M424_DUMPS = list(range(3200, 3210))
M424_RUNS = {"flux": dict(), "flux_sm335": dict(smooth=335.0), "flux_lamfix": dict(corr="lamfix")}


def _m424_samples():
    root = conftest.M424.get("samples") or os.environ.get(
        "PPMPY_SYNSPEC_M424_SAMPLES", "/scratch/ppathak/fastwind_sphere/samples_r4050_N1236544")
    if not os.path.isdir(root):
        pytest.skip("M424 samples not available: {}".format(root))
    return root


@pytest.fixture(scope="module")
def m424():
    lib = m424_path("run", "library_dT10.npz")
    pts = m424_path("run", "points.npz")
    if not os.path.isdir(SHADOW):
        pytest.skip("shadow directory {} not available".format(SHADOW))
    assert not os.path.realpath(SHADOW).startswith("/scratch/ppathak/fastwind_sphere")
    return types.SimpleNamespace(library=lib, theta=npz_member_memmap(pts, "theta"), phi=npz_member_memmap(pts, "phi"),
                                 samples=_m424_samples(), lamfix=m424_path("disc", "lamfix_dT10.npz"))


def _m424_rows(ref_path, dumps):
    ref_dumps = np.load(ref_path)["dumps"]
    rows = np.searchsorted(ref_dumps, dumps)
    np.testing.assert_array_equal(ref_dumps[rows], dumps)
    return rows


def _assert_timeseries_rows(ts, ref_path, dumps):
    """Every member of ts equals the rows of the production time series (whole members for the constants)."""
    rows = _m424_rows(ref_path, dumps)
    with np.load(ts) as z, np.load(ref_path) as zr:
        assert z.files == zr.files == list(dm.TIMESERIES_KEYS)
        for k in z.files:
            if k in ("F", "F0"):
                r = npz_member_memmap(ref_path, k)[rows[0]:rows[-1] + 1]
            elif zr[k].ndim and zr[k].shape[0] == zr["dumps"].shape[0] and k not in ("Y", "LREF", "los"):
                r = zr[k][rows]
            else:
                r = zr[k]
            assert z[k].dtype == r.dtype, k
            np.testing.assert_array_equal(z[k], r, err_msg=k)


@pytest.mark.m424
@pytest.mark.parametrize("run,nproc,start", [("flux", 4, "fork"), ("flux_sm335", 4, "spawn"), ("flux_lamfix", 1, None)])
def test_m424_run_disc_dumps(m424, run, nproc, start):
    """run_disc_dumps (flux_integrator of library_dT10.npz, the points of points.npz, the dump samples) for dumps
    3200-3209: every per-dump file equals the production file of fw_disc_dumps.py byte for byte (fork, spawn and
    serial); collect_timeseries of them equals the rows of the production time series. The serial run tunes this
    process's malloc (tune_malloc=True; without, the system time of the page faults made it 45-100 s here instead of
    ~8 s on the Trillium login node)."""
    kw = dict(M424_RUNS[run], nmin=20)
    if kw.get("corr") == "lamfix":
        kw["corr"] = m424.lamfix
    out = tempfile.mkdtemp(prefix="test_{}_".format(run), dir=SHADOW)
    t0, c0, p0 = time.time(), resource.getrusage(resource.RUSAGE_CHILDREN), resource.getrusage(resource.RUSAGE_SELF)
    s = dm.run_disc_dumps(M424_DUMPS, m424.samples, out, None, dm.flux_integrator, (m424.library,), m424.theta,
                          m424.phi, "thompson2024", lref=LINESET, factory_kwargs=kw, nproc=nproc, start_method=start,
                          tmpdir=out, tune_malloc=True if nproc == 1 else None, log=print)
    wall = time.time() - t0
    ru, rc = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    print("M424 {}: {} dumps in {:.1f} s ({} workers, {}), workers {:.0f} s CPU, parent {:.1f} s user + {:.1f} s "
          "system, max RSS {:.2f} GB".format(run, len(M424_DUMPS), wall, nproc, start,
                                            rc.ru_utime + rc.ru_stime - c0.ru_utime - c0.ru_stime,
                                            ru.ru_utime - p0.ru_utime, ru.ru_stime - p0.ru_stime, ru.ru_maxrss / 1e6))
    assert s["name"] == run and sorted(s["done"]) == M424_DUMPS and s["nn"] == 245
    for d in M424_DUMPS:
        _assert_npz_equal(dm.dump_path(out, run, d), m424_path("disc", run, "d{:04d}.npz".format(d)))
    ts = dm.collect_timeseries(out, run, out=os.path.join(out, "ts.npz"), log=print)
    _assert_timeseries_rows(ts, m424_path("disc", "{}_timeseries.npz".format(run)), M424_DUMPS)
    shutil.rmtree(out)


@pytest.mark.m424
def test_m424_collect_production_files(m424, tmp_path):
    """collect_timeseries of the production per-dump files 3200-3209 of the flux run (legacy directory: grid, lref,
    los given) equals the rows of flux_timeseries.npz, and byte for byte the frozen fw_disc_collect.py run on the same
    files (symlinked into tmp)."""
    src = m424_path("disc", "flux")
    out = dm.collect_timeseries(os.path.dirname(src), "flux", dumps=M424_DUMPS, out=str(tmp_path / "ts.npz"),
                                los="thompson2024", grid=GRID, lref=LINESET)
    _assert_timeseries_rows(out, m424_path("disc", "flux_timeseries.npz"), M424_DUMPS)
    (tmp_path / "legacy" / "flux").mkdir(parents=True)
    for d in M424_DUMPS:
        name = "d{:04d}.npz".format(d)
        os.symlink(os.path.join(src, name), str(tmp_path / "legacy" / "flux" / name))
    ts_legacy = _legacy_collect(tmp_path / "legacy", "flux", M424_DUMPS[0], M424_DUMPS[-1])
    assert _sha(ts_legacy) == _sha(out)


def _m424_imu_lib():
    """The M424 intensity library: PPMPY_SYNSPEC_M424_IMU (the .npz, as test_discimu.py, or a directory holding
    imu_library_dT10.npz, as test_imulib.py), default /scratch/ppathak/fastwind_imu/imu_library_dT10.npz."""
    p = os.environ.get("PPMPY_SYNSPEC_M424_IMU", "/scratch/ppathak/fastwind_imu/imu_library_dT10.npz")
    if os.path.isdir(p):
        p = os.path.join(p, "imu_library_dT10.npz")
    if not os.path.exists(p):
        pytest.skip("M424 intensity library not available: {}".format(p))
    return p


def _shadow_m4():
    if not os.path.isdir(os.path.dirname(SHADOW_M4)):
        pytest.skip("shadow directory {} not available".format(SHADOW_M4))
    os.makedirs(SHADOW_M4, exist_ok=True)
    assert not os.path.realpath(SHADOW_M4).startswith(("/scratch/ppathak/fastwind_sphere",
                                                       "/scratch/ppathak/fastwind_imu"))
    return SHADOW_M4


@pytest.mark.m424
@pytest.mark.slow
@pytest.mark.parametrize("nproc,start", [(4, "fork"), (2, "spawn")])
def test_m424_run_disc_dumps_imu(m424, nproc, start):
    """run_disc_dumps with imu_integrator (default options, the legacy imu_library_dT10.npz, lref = the M424 LineSet)
    for dumps 3200-3209: every per-dump file equals the production imu/dNNNN.npz of fw_disc_dumps.py --method imu
    byte for byte, with 4 'fork' workers (sharing the parent's precomputed library FFTs) and with 2 'spawn' workers
    (each builds its own integrator and must match the parent's fingerprint); collect_timeseries of them equals the
    rows of imu_timeseries.npz. Measured 2026-10-02 (Trillium login node): see the printed line; ~8-10 GB per
    process that builds the integrator."""
    lib = _m424_imu_lib()
    for d in M424_DUMPS:
        m424_path("disc", "imu", "d{:04d}.npz".format(d))
    out = tempfile.mkdtemp(prefix="test_imu_{}_".format(start), dir=_shadow_m4())
    t0, c0, p0 = time.time(), resource.getrusage(resource.RUSAGE_CHILDREN), resource.getrusage(resource.RUSAGE_SELF)
    s = dm.run_disc_dumps(M424_DUMPS, m424.samples, out, None, dm.imu_integrator, (lib,), m424.theta, m424.phi,
                          "thompson2024", lref=LINESET, nproc=nproc, start_method=start, tmpdir=out, log=print)
    wall = time.time() - t0
    ru, rc = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    print("M424 imu: {} dumps in {:.1f} s ({} workers, {}), workers {:.0f} s CPU (max RSS {:.2f} GB), parent {:.1f} s "
          "user + {:.1f} s system, max RSS {:.2f} GB".format(
              len(M424_DUMPS), wall, nproc, start, rc.ru_utime + rc.ru_stime - c0.ru_utime - c0.ru_stime,
              rc.ru_maxrss / 1e6, ru.ru_utime - p0.ru_utime, ru.ru_stime - p0.ru_stime, ru.ru_maxrss / 1e6))
    assert s["name"] == "imu" and sorted(s["done"]) == M424_DUMPS and s["nn"] == 303 and s["start_method"] == start
    for d in M424_DUMPS:
        _assert_npz_equal(dm.dump_path(out, "imu", d), m424_path("disc", "imu", "d{:04d}.npz".format(d)))
    rec = dm.read_run_record(os.path.join(out, "imu"))["params"]["integrator"]
    assert rec["cls"] == "ppmpy.synspec.disc.DiscImu" and rec["custom"]["fft"] == "precomputed"
    assert rec["inputs"]["factory"] == "ppmpy.synspec.dumps.imu_integrator"
    ts = dm.collect_timeseries(out, "imu", out=os.path.join(out, "ts.npz"), log=print)
    _assert_timeseries_rows(ts, m424_path("disc", "imu_timeseries.npz"), M424_DUMPS)
    shutil.rmtree(out)


@pytest.mark.m424
@pytest.mark.slow
def test_m424_run_disc_dumps_imu_lazy(m424):
    """The laptop mode: run_disc_dumps with imu_integrator(fft='lazy') in one fresh process (nproc 1, lines of sight
    batched by default) for dumps 3200 and 4800: F equals the production imu files bit for bit (x86-64; else to the
    float32 rounding), F0 to the float32 rounding, the velocity moments and every other member exactly; batched equals
    per line of sight bit for bit (every member but the name) and is faster; peak RSS (VmHWM) of the whole run,
    projections included, <= 3 GB."""
    lib = _m424_imu_lib()
    dl = [3200, 4800]
    for d in dl:
        m424_path("disc", "imu", "d{:04d}.npz".format(d))
    pts = m424_path("run", "points.npz")
    out = tempfile.mkdtemp(prefix="test_imu_lazy_", dir=_shadow_m4())
    code = textwrap.dedent("""
        import json, sys, time
        sys.path.insert(0, {root!r})
        import numpy as np
        from ppmpy.synspec import dumps as dm
        from ppmpy.synspec.io import npz_member_memmap
        from ppmpy.synspec.spectral import LineSet

        def status():
            out = {{}}
            for l in open('/proc/self/status'):
                k = l.split(':')[0]
                if k in ('VmHWM', 'VmRSS', 'RssAnon', 'RssFile'):
                    out[k] = int(l.split()[1]) / 1e6
            return out

        th, ph = (npz_member_memmap({pts!r}, k) for k in ('theta', 'phi'))
        r = dict(base=status())
        for name, batch in (('imu_lazy', None), ('imu_lazy_perlos', False)):
            t0 = time.time()
            s = dm.run_disc_dumps({dl}, {samples!r}, {out!r}, name, dm.imu_integrator, ({lib!r},), th, ph,
                                  'thompson2024', lref=LineSet({lines!r}, {lref!r}), factory_kwargs=dict(fft='lazy'),
                                  batch=batch, tune_malloc=True)
            r[name] = dict(wall=time.time() - t0, status=status(), done=s['done'])
        print('RESULT ' + json.dumps(r))
    """).format(root=ROOT, pts=pts, dl=dl, samples=m424.samples, out=out, lib=lib, lines=LINES, lref=LREF.tolist())
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=3000)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    r = json.loads(res.stdout.split("RESULT ", 1)[1])
    print("M424 lazy run (GB): {}".format(r))
    assert r["imu_lazy"]["done"] == dl and r["imu_lazy"]["status"]["VmHWM"] <= 3.0
    assert r["imu_lazy"]["wall"] < r["imu_lazy_perlos"]["wall"]
    for d in dl:
        a = dm.dump_path(out, "imu_lazy", d)
        with np.load(a) as z, np.load(dm.dump_path(out, "imu_lazy_perlos", d)) as zp:
            assert z.files == zp.files
            for k in z.files:
                if k != "name":
                    assert z[k].dtype == zp[k].dtype and np.array_equal(z[k], zp[k],
                                                                        equal_nan=z[k].dtype.kind == "f"), k
        with np.load(a) as z, np.load(m424_path("disc", "imu", "d{:04d}.npz".format(d))) as ref:
            assert z.files == ref.files
            for k in z.files:
                if k == "name":
                    continue
                if k in ("F0", "diag_F0") or (k == "F" and not LAZY_BITWISE):
                    tol = 6e-8 if k != "diag_F0" else 1e-6
                    assert np.abs(z[k].astype(float) - ref[k]).max() <= tol, k
                else:
                    np.testing.assert_array_equal(z[k], ref[k], err_msg=k)
    shutil.rmtree(out)


def test_modname_spawn_alias():
    """A class defined in the calling script is '__main__' in the parent and '__mp_main__' in spawn workers; run
    records must treat both as the same module."""
    from ppmpy.synspec import dumps as dm

    class A:
        pass
    A.__module__ = "__mp_main__"
    assert dm._modname(A) == "__main__"
    A.__module__ = "pkg.mod"
    assert dm._modname(A) == "pkg.mod"


def test_imu_fingerprint_defaults_normalised():
    """imu_integrator runs started with explicit defaults and with omitted defaults have the same input record."""
    from ppmpy.synspec import dumps as dm
    a = dm._inputs_fingerprint(dm.imu_integrator, ("lib.npz",), {})
    b = dm._inputs_fingerprint(dm.imu_integrator, ("lib.npz",), dict(fft="precomputed", dtype="float64", lines=None))
    c = dm._inputs_fingerprint(dm.imu_integrator, ("lib.npz",), dict(fft="lazy"))
    assert a == b and a != c
