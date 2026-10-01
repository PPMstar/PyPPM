"""Tests of ppmpy.synspec.fwresults: OUT / OUT_IMU readers, merge, streaming combine, ProfileStore, EW, summary.

Synthetic runs are built in tmp_path with the layout of the M424 per-point run (results/<tag>/part_*.tar.gz
with P<idx>/meta.txt and OUT.* per point) and processed both by fwresults and by the frozen legacy
fw_sphere_merge.py (tests/synspec/legacy, run as a script); the files must agree byte for byte.
"""
import glob
import hashlib
import io
import json
import multiprocessing
import os
import pickle
import re
import shutil
import subprocess
import sys
import tarfile
import tracemalloc
import warnings

import numpy as np
import pytest

from conftest import ROOT, m424_path
from ppmpy.synspec import fwresults as fw
from ppmpy.synspec.io import npz_member_memmap

LEGACY = os.environ.get("PPMPY_SYNSPEC_M424_PROJECT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy"))
LINES = ["HEI4026", "HEII4200", "HEI4922"]
LREF = [4026.22, 4199.90, 4921.93]
NROW = 161
IMU_RUN = os.environ.get("PPMPY_SYNSPEC_M424_IMU", "/scratch/ppathak/fastwind_imu/runs/P001563/P001563")
ITMORE_CHECK = os.environ.get("PPMPY_SYNSPEC_M424_ITMORE", "/scratch/ppathak/fastwind_runs/itmore_check")


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _legacy_file(name):
    p = os.path.join(LEGACY, name)
    if not os.path.exists(p):
        pytest.skip("legacy file not available: {}".format(p))
    return p


def _run_legacy_merge(*args):
    """The frozen fw_sphere_merge.py as a script (it parses sys.argv at import)."""
    out = subprocess.run([sys.executable, _legacy_file("fw_sphere_merge.py")] + list(args), capture_output=True,
                         text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    return out.stdout


def _imu_file(name):
    p = os.path.join(IMU_RUN, name)
    if not os.path.exists(p):
        pytest.skip("M424 FASTWIND run not available: {}".format(p))
    return p


# ----------------------------------------------------------------------------------------------
# synthetic FASTWIND files and runs
# ----------------------------------------------------------------------------------------------
def _profile(seed, j, nrow=NROW):
    """Formatted OUT table of a synthetic line profile; returns (bytes, expected float64 (nrow, 6), ew)."""
    rng = np.random.default_rng(1000 * seed + j)
    lref = LREF[j]
    lam = lref + np.linspace(-40.0, 40.0, nrow)
    lam[nrow // 2 + 1] = lam[nrow // 2] + 0.003             # rounds to the same 0.01 A as its neighbour
    depth = 0.3 + 0.1 * rng.random()
    fn = 1.0 - depth * np.exp(-((lam - lref) / (2.0 + rng.random())) ** 2)
    fc = 7.4e-7 * (1.0 + 0.01 * seed + 0.001 * rng.random())
    rows = []
    for k in range(nrow):
        rows.append("{:4d} {:11.5f} {:15.2f} {:19.6E} {:15.5f} {:15.5f}    ".format(
            k + 1, 1.2 - 2.5 * k / (nrow - 1), lam[k], fc * (1 + 1e-4 * k), fn[k], fn[k] + 1e-5))
    ew = -float(np.round(depth * 3.1, 14))
    text = "\n".join(rows) + "\n  {!r}     \n".format(ew)
    exp = np.array([[float(t) for t in r.split()] for r in rows])
    return text.encode(), exp, ew


def _tar_add(tf, name, data):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    ti.mtime = 1758800000
    tf.addfile(ti, io.BytesIO(data))


def _tar_dir(tf, name):
    ti = tarfile.TarInfo(name)
    ti.type = tarfile.DIRTYPE
    ti.mode = 0o770
    ti.mtime = 1758800000
    tf.addfile(ti)


def _write_part(path, records, extra_top=False, orphan=None):
    """records: list of (idx, teff_str, status, niter, tr23_str, seed); seed selects the profiles."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tarfile.open(path, "w:gz") as tf:
        _tar_dir(tf, "./")
        if extra_top:
            _tar_add(tf, "./README", b"not a point\n")
        for idx, teff, status, niter, tr23, seed in records:
            d = "./P{:06d}".format(idx)
            _tar_dir(tf, d + "/")
            _tar_add(tf, d + "/meta.txt", "{} {} {} {} {} {:.1f} {:.1f}\n".format(
                idx, teff, status, niter, tr23, 150.0 + seed + 0.04, 0.5).encode())
            _tar_add(tf, d + "/MODEL", b"x" * 100)
            if status == "ok":
                for j, ln in enumerate(LINES):
                    _tar_add(tf, d + "/OUT.{}_VTV010".format(ln), _profile(seed, j)[0])
            _tar_add(tf, d + "/INDAT.DAT", b"indat\n")
        if orphan is not None:                            # a directory without meta.txt: dropped
            _tar_add(tf, "./P{:06d}/OUT.HEI4026_VTV010".format(orphan), _profile(99, 0)[0])


NPTS = 12


def _make_run(root):
    """A small per-point run with every merge / combine case; returns (run_dir, expectations)."""
    run = os.path.join(str(root), "run")
    os.makedirs(run)
    rng = np.random.default_rng(5)
    teff = np.round(38000.0 + 600.0 * rng.standard_normal(NPTS), 3)
    pts = dict(idx=np.arange(NPTS), teff=teff, r=np.full(NPTS, 4050.0), theta=rng.random(NPTS) * np.pi,
               phi=rng.random(NPTS) * 2 * np.pi, x=rng.standard_normal(NPTS), y=rng.standard_normal(NPTS),
               z=rng.standard_normal(NPTS), ur_kms=50 * rng.standard_normal(NPTS), relT=0.01 * rng.standard_normal(NPTS))
    np.savez(os.path.join(run, "points.npz"), **pts)
    np.savetxt(os.path.join(run, "points.txt"), np.column_stack([pts["idx"], teff]), fmt=["%d", "%.3f"])
    T = lambda i, nudge=0.0: "{:.3f}".format(teff[i] + nudge)
    res = os.path.join(run, "results")
    # task_0000: two parts; 3 fails then succeeds, 5 succeeds then fails (first ok kept), 6 fails
    _write_part(os.path.join(res, "task_0000", "part_1.tar.gz"),
                [(0, T(0), "ok", 61, "39000.5", 0), (3, T(3), "pnlte_failed", 19, "nan", 3),
                 (5, T(5), "ok", 102, "39100.25", 5), (4, T(4), "ok", 60, "39050.125", 4),
                 (6, T(6), "pnlte_failed", 21, "nan", 6)], orphan=11)
    _write_part(os.path.join(res, "task_0000", "part_2.tar.gz"),
                [(3, T(3), "ok", 63, "39010.0", 33), (5, T(5), "formal_failed", 102, "39100.25", 55),
                 (8, T(8), "ok", 102, "38990.0", 8)])
    # task_0001: only statuses of length 2 and 13; 4 again (ok, other profiles: the task_0000 record is kept)
    _write_part(os.path.join(res, "task_0001", "part_1.tar.gz"),
                [(1, T(1), "ok", 59, "39001.0", 1), (4, T(4), "ok", 60, "39050.125", 44),
                 (7, T(7), "pnlte_failed", 19, "nan", 7), (2, T(2), "pnlte_timeout", 0, "nan", 2),
                 (10, T(10), "ok", 64, "39002.0", 10)])
    # task_missing_0000: retries with T_eff + 1 K; 7 fails again, 6 succeeds; 9 is never run
    _write_part(os.path.join(res, "task_missing_0000", "part_1.tar.gz"),
                [(6, T(6, 1.0), "ok", 62, "39003.0", 66), (7, T(7, 1.0), "pnlte_failed", 19, "nan", 77)])
    exp = dict(
        status={0: "ok", 1: "ok", 2: "pnlte_timeout", 3: "ok", 4: "ok", 5: "ok", 6: "ok", 7: "pnlte_failed",
                8: "ok", 10: "ok"},
        seed={0: 0, 1: 1, 3: 33, 4: 4, 5: 5, 6: 66, 8: 8, 10: 10},
        nudge={6: 1.0},                                  # 7: both records failed, the first (task_0001, +0 K) is kept
        missing={2: teff[2] + 1.0, 7: teff[7] + 2.0, 9: teff[9], 11: teff[11]},
        teff=teff)
    return run, exp


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """The synthetic run, merged and combined by the legacy script and by fwresults."""
    root = tmp_path_factory.mktemp("fwrun")
    run, exp = _make_run(root)
    res = os.path.join(run, "results")
    tags = ["task_0000", "task_0001", "task_missing_0000"]
    logs = {}
    for t in tags:
        out = _run_legacy_merge(run, "--tags", t, "--out", "merged/{}.npz".format(t))
        new = []
        fw.merge_task(res, t, os.path.join(run, "new_merged", t + ".npz"), os.path.join(run, "points.npz"), log=new.append)
        logs[t] = (out, new)
    logs["combine"] = _run_legacy_merge(run, "--combine")
    return dict(run=run, exp=exp, tags=tags, logs=logs)


# ----------------------------------------------------------------------------------------------
# readers
# ----------------------------------------------------------------------------------------------
def test_no_heavy_imports():
    code = ("import sys; import ppmpy.synspec.fwresults;"
            "bad=[m for m in ('ppmpy.ppm','matplotlib','scipy') if m in sys.modules]; print(','.join(bad))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                         env=dict(os.environ, PYTHONPATH=ROOT))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_read_out_synthetic(tmp_path):
    data, exp, ew = _profile(3, 0)
    p = tmp_path / "OUT.HEI4026_VTV010"
    p.write_bytes(data)
    for src in (str(p), data):
        r = fw.read_out(src)
        assert r["nrow"] == NROW and r["ew_fastwind"] == ew
        for c, k in enumerate(("x", "lam", "fcont", "fnorm", "frot")):
            assert r[k].dtype == np.float64
            np.testing.assert_array_equal(r[k], exp[:, c + 1])
        np.testing.assert_array_equal(r["k"], np.arange(1, NROW + 1))
        # 0.01 A rounding: a repeated wavelength is kept as it is
        assert (np.diff(r["lam"]) == 0).sum() == 1
    g = np.genfromtxt(str(p), max_rows=NROW)
    for c, k in enumerate(("x", "lam", "fcont", "fnorm", "frot")):
        assert r[k].tobytes() == g[:, c + 1].tobytes()
    r2 = fw.read_out(data, nrow=50)
    assert r2["nrow"] == 50 and np.isnan(r2["ew_fastwind"])
    np.testing.assert_array_equal(r2["lam"], exp[:50, 2])
    with pytest.raises(ValueError):
        fw.read_out(data, nrow=NROW + 5)
    with pytest.raises(ValueError):
        fw.read_out(b"just text\n")


def test_out_table_matches_genfromtxt_and_fallback():
    data = _profile(7, 2)[0]
    g = np.genfromtxt(io.BytesIO(data), usecols=[2, 3, 4], max_rows=NROW)
    assert fw._out_table(data, NROW, (2, 3, 4)).tobytes() == g.tobytes()
    # a comment line at the top: the fast path declines, genfromtxt skips it -> same numbers
    data2 = b"# header\n" + data
    assert fw._out_table(data2, NROW, (2, 3, 4)).tobytes() == g.tobytes()
    # Fortran-style tokens without 'E' are rejected by float(): genfromtxt's handling (NaN) is kept
    bad = data.replace(b"E-07", b"-007", 1)
    gb = np.genfromtxt(io.BytesIO(bad), usecols=[2, 3, 4], max_rows=NROW)
    np.testing.assert_array_equal(fw._out_table(bad, NROW, (2, 3, 4)), gb)


def _imu_text(nk=7, nray=5, ncore=2, seed=0):
    rng = np.random.default_rng(seed)
    p = np.concatenate([np.linspace(0, 0.99, ncore), 1.0 + np.cumsum(rng.random(nray - ncore)) * 1e-3])
    lam = 4026.0 + np.arange(nk) * 0.123456
    Ic = rng.random((nk, nray))
    Il = Ic * (1 - 0.3 * rng.random((nk, nray)))
    head = "# rays NP-1, core rays NC = {:4d} {:4d}\n# p {}\n# K, lambda, I_cont(p_1..p_NP-1), I_line(p_1..p_NP-1)\n".format(
        nray, ncore, " ".join("{:15.8E}".format(v) for v in p))
    rows = "".join("{:4d} {:.6f} {} {}\n".format(k + 1, lam[k], " ".join("{:.8E}".format(v) for v in Ic[k]),
                                                 " ".join("{:.8E}".format(v) for v in Il[k])) for k in range(nk))
    return head + rows


def test_read_out_imu_synthetic(tmp_path):
    sys.path.insert(0, LEGACY)
    try:
        import fw_disc as legacy
    finally:
        sys.path.remove(LEGACY)
    f = tmp_path / "OUT_IMU.HEI4026_VTV010"
    f.write_text(_imu_text())
    new = fw.read_out_imu(str(f), counts=True)
    old = legacy.read_imu(str(f))
    for a, b in zip(new[:4], old):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    assert new[4] == dict(nray=5, ncore=2)
    assert len(fw.read_out_imu(str(f))) == 4
    f.write_text(_imu_text().replace("=    5", "=    6", 1))
    with pytest.raises(ValueError):
        fw.read_out_imu(str(f))


def test_parse_meta():
    m = fw.parse_meta(b"571348 37299.910 ok 63 38834.0857003107 148.3 0.4\n")
    assert m == dict(idx=571348, teff=37299.91, status="ok", niter=63, T_tau23=38834.0857003107, t_pnlte=148.3,
                     t_formal=0.4)
    assert np.isnan(fw.parse_meta("7 38000.000 pnlte_failed 19 nan 140.0 0.0")["T_tau23"])


def test_npy_header_matches_numpy():
    for shape, dt in [((0,), "<i4"), ((7,), "<f8"), ((5, 3, 161), "<f4"), ((12345, 3, 7), "<f4"),
                      ((0, 3, 161), "<f4"), ((4,), "<U12"), ((3,), "<U8"), ((2, 2), "<i2"), ((), "<U5")]:
        a = np.zeros(shape, dt)
        b = io.BytesIO()
        np.save(b, a)
        ref = b.getvalue()[:len(b.getvalue()) - a.nbytes]
        assert fw._npy_header(shape, dt) == ref
        assert fw._npy_header(shape, np.dtype(dt)) == ref


def test_iter_part_points(tmp_path):
    part = str(tmp_path / "res" / "t" / "part_1.tar.gz")
    _write_part(part, [(3, "38000.000", "ok", 61, "39000.0", 3), (4, "38001.000", "pnlte_failed", 19, "nan", 4)],
                extra_top=True, orphan=9)
    got = list(fw.iter_part_points(part))
    assert [g[0] for g in got] == ["P000003", "P000004"]
    assert sorted(got[0][1]) == ["OUT.HEI4026_VTV010", "OUT.HEI4922_VTV010", "OUT.HEII4200_VTV010", "meta.txt"]
    assert sorted(got[1][1]) == ["meta.txt"]
    assert sorted(dict(fw.iter_part_points(part, want=("meta.txt", "INDAT")))["P000003"]) == ["INDAT.DAT", "meta.txt"]
    if hasattr(tarfile, "data_filter"):
        evil = str(tmp_path / "res" / "t" / "part_2.tar.gz")
        with tarfile.open(evil, "w:gz") as tf:
            _tar_add(tf, "../../P000001/meta.txt", b"1 38000.000 ok 61 39000.0 1.0 0.1\n")
        with pytest.raises(ValueError, match="unsafe member"):
            list(fw.iter_part_points(evil))
        assert len(list(fw.iter_part_points(evil, check_names=False))) == 1


# ----------------------------------------------------------------------------------------------
# merge and combine (synthetic, against the frozen legacy script)
# ----------------------------------------------------------------------------------------------
def test_merge_task_matches_legacy_bytes(synth):
    run = synth["run"]
    for t in synth["tags"]:
        assert _sha(os.path.join(run, "new_merged", t + ".npz")) == _sha(os.path.join(run, "merged", t + ".npz")), t
        old, new = synth["logs"][t]
        old = old.splitlines()
        assert new[:-1] == old[:-1] and new[-1].split()[0] == "wrote" and new[-1].split()[2:] == old[-1].split()[2:]


def test_merge_task_content(synth):
    run, exp = synth["run"], synth["exp"]
    z = np.load(os.path.join(run, "new_merged", "task_0000.npz"))
    assert z.files == ["idx", "teff", "status", "niter", "T_tau23", "t_pnlte", "t_formal", "lam", "fcont", "fnorm",
                       "lines", "r", "theta", "phi", "x", "y", "z", "ur_kms", "relT", "teff_nudge"]
    np.testing.assert_array_equal(z["idx"], [0, 3, 4, 5, 6, 8])
    assert z["idx"].dtype == np.int32 and z["niter"].dtype == np.int16 and z["lam"].dtype == np.float32
    # 3: failed then ok -> ok; 5: ok then failed -> the ok record; 6 failed
    np.testing.assert_array_equal(z["status"], ["ok", "ok", "ok", "ok", "pnlte_failed", "ok"])
    assert z["status"].dtype == np.dtype("<U12")
    np.testing.assert_array_equal(z["niter"], [61, 63, 60, 102, 21, 102])
    for row, i in enumerate(z["idx"]):
        if z["status"][row] != "ok":
            assert np.isnan(z["lam"][row]).all() and np.isnan(z["fnorm"][row]).all()
            continue
        s = {0: 0, 3: 33, 4: 4, 5: 5, 8: 8}[int(i)]
        for j in range(3):
            e = _profile(s, j)[1]
            np.testing.assert_array_equal(z["lam"][row, j], e[:, 2].astype(np.float32))
            np.testing.assert_array_equal(z["fcont"][row, j], e[:, 3].astype(np.float32))
            np.testing.assert_array_equal(z["fnorm"][row, j], e[:, 4].astype(np.float32))
    pts = np.load(os.path.join(run, "points.npz"))
    np.testing.assert_array_equal(z["x"], pts["x"][z["idx"]])
    np.testing.assert_array_equal(z["teff_nudge"], 0.0)
    zm = np.load(os.path.join(run, "new_merged", "task_missing_0000.npz"))
    np.testing.assert_array_equal(zm["teff_nudge"], [1.0, 1.0])


def test_merge_task_points_mapping(synth, tmp_path):
    """points given as a loaded mapping (dict, open NpzFile) gives the same file as the path."""
    run = synth["run"]
    res = os.path.join(run, "results")
    ref = _sha(os.path.join(run, "merged", "task_0000.npz"))
    pts_dict = dict(np.load(os.path.join(run, "points.npz")))
    out = str(tmp_path / "a.npz")
    fw.merge_task(res, "task_0000", out, pts_dict)
    assert _sha(out) == ref
    with np.load(os.path.join(run, "points.npz")) as z:
        fw.merge_task(res, "task_0000", str(tmp_path / "b.npz"), z)
    assert _sha(str(tmp_path / "b.npz")) == ref
    assert sorted(os.listdir(str(tmp_path))) == ["a.npz", "b.npz"]              # no temporaries left


def test_merge_combine_record_niter_cap(synth, tmp_path):
    """niter_cap recorded in '_meta' by merge_task and carried over by combine (only when all files agree)."""
    run = synth["run"]
    res, pts, ptxt = os.path.join(run, "results"), os.path.join(run, "points.npz"), os.path.join(run, "points.txt")
    md = tmp_path / "merged"
    for t in synth["tags"]:
        fw.merge_task(res, t, str(md / (t + ".npz")), pts, niter_cap=102)
    from ppmpy.synspec.io import read_meta
    assert read_meta(str(md / "task_0000.npz")) == dict(niter_cap=102)
    p, _ = fw.combine(str(md / "task_*.npz"), ptxt, str(tmp_path / "c1"))
    s = fw.ProfileStore.open(p)
    assert s.niter_cap == 102 and s.meta == dict(niter_cap=102)
    np.testing.assert_array_equal(s.usable(cap_ok=False), s.usable() & (np.asarray(s.niter) < 102))
    # the arrays are those of the legacy file
    ref = np.load(os.path.join(run, "profiles.npz"))
    with np.load(p) as z:
        assert [k for k in z.files if k != "_meta"] == ref.files
        for k in ref.files:
            assert z[k].tobytes() == ref[k].tobytes()
    # a retry round run with another ITMORE: the caps differ -> nothing recorded, with a warning
    fw.merge_task(res, "task_missing_0000", str(md / "task_missing_0000.npz"), pts, niter_cap=302)
    with pytest.warns(UserWarning, match="iteration caps"):
        p, _ = fw.combine(str(md / "task_*.npz"), ptxt, str(tmp_path / "c2"))
    assert fw.ProfileStore.open(p).niter_cap is None
    # legacy merged files (no '_meta') mixed with recorded ones: also unknown; an explicit cap wins
    shutil.copy(os.path.join(run, "merged", "task_missing_0000.npz"), str(md))
    with pytest.warns(UserWarning, match="iteration caps"):
        fw.combine(str(md / "task_*.npz"), ptxt, str(tmp_path / "c3"))
    p, _ = fw.combine(str(md / "task_*.npz"), ptxt, str(tmp_path / "c4"), niter_cap=102, meta=dict(kind="x"))
    assert fw.ProfileStore.open(p).meta == dict(kind="x", niter_cap=102)


def test_merge_task_rejects_large_nudge(tmp_path):
    run, _ = _make_run(tmp_path)
    teff = np.load(os.path.join(run, "points.npz"))["teff"]
    _write_part(os.path.join(run, "results", "bad", "part_1.tar.gz"),
                [(1, "{:.3f}".format(teff[1] + 25.0), "ok", 60, "39000.0", 1)])
    with pytest.raises(ValueError, match="nudge"):
        fw.merge_task(os.path.join(run, "results"), "bad", os.path.join(run, "bad.npz"), os.path.join(run, "points.npz"))
    assert not os.path.exists(os.path.join(run, "bad.npz"))


@pytest.mark.parametrize("block", [1, 3, 20000])
def test_combine_matches_legacy_bytes(synth, tmp_path, block):
    run = synth["run"]
    p, m = fw.combine(os.path.join(run, "merged", "task_*.npz"), os.path.join(run, "points.txt"), str(tmp_path),
                      block=block)
    assert _sha(p) == _sha(os.path.join(run, "profiles.npz"))
    assert _sha(m) == _sha(os.path.join(run, "missing.txt"))
    assert sorted(os.listdir(str(tmp_path))) == ["missing.txt", "profiles.npz"]      # no temporaries (hidden ones too)
    # the summary lines are the legacy ones (collected while writing)
    lines = []
    fw.combine(os.path.join(run, "merged", "task_*.npz"), os.path.join(run, "points.txt"), str(tmp_path / "log"),
               block=block, log=lines.append)
    assert "\n".join(lines) + "\n" == synth["logs"]["combine"]


def test_combine_content(synth, tmp_path):
    run, exp = synth["run"], synth["exp"]
    p, m = fw.combine(os.path.join(run, "new_merged", "task_*.npz"), os.path.join(run, "points.txt"), str(tmp_path),
                      block=2)
    z = np.load(p)
    assert z.files[-1] == "lines" and z.files[:3] == ["idx", "teff", "status"]
    np.testing.assert_array_equal(z["idx"], sorted(exp["status"]))
    assert z["status"].dtype == np.dtype("<U13")
    assert z["status"].tolist() == [exp["status"][i] for i in sorted(exp["status"])]
    np.testing.assert_array_equal(z["teff_nudge"], [exp["nudge"].get(i, 0.0) for i in sorted(exp["status"])])
    for row, i in enumerate(z["idx"].tolist()):
        if i in exp["seed"]:
            for j in range(3):
                np.testing.assert_array_equal(z["fnorm"][row, j], _profile(exp["seed"][i], j)[1][:, 4].astype(np.float32))
        else:
            assert np.isnan(z["fnorm"][row]).all()
    miss = np.loadtxt(m, ndmin=2)
    np.testing.assert_array_equal(miss[:, 0], sorted(exp["missing"]))
    np.testing.assert_allclose(miss[:, 1], [exp["missing"][i] for i in sorted(exp["missing"])], rtol=0, atol=6e-4)
    # profiles readable as memory maps (stored, zip64 per member)
    np.testing.assert_array_equal(npz_member_memmap(p, "fnorm"), z["fnorm"])


def test_combine_skips_temporaries(synth, tmp_path):
    """Temporaries of interrupted writes in merged/ are skipped by the glob (the legacy glob took them)."""
    run = synth["run"]
    md = tmp_path / "merged"
    shutil.copytree(os.path.join(run, "merged"), str(md))
    (md / "task_0001.tmp4242.npz").write_bytes(b"PK\x03\x04 truncated")        # io.save_npz naming
    shutil.copy(str(md / "task_0001.npz"), str(md / "task_0001.tmp.npz"))       # legacy naming, complete
    (md / ".task_0000.tmp77.npz").write_bytes(b"hidden: never matched")          # fwresults naming
    with pytest.warns(UserWarning, match="temporary files") as rec:
        p, m = fw.combine(str(md / "task_*.npz"), os.path.join(run, "points.txt"), str(tmp_path / "out"))
    msg = " ".join(str(r.message) for r in rec)
    assert "task_0001.tmp4242.npz" in msg and "task_0001.tmp.npz" in msg and ".task_0000" not in msg
    assert _sha(p) == _sha(os.path.join(run, "profiles.npz"))
    assert _sha(m) == _sha(os.path.join(run, "missing.txt"))
    assert sorted(os.listdir(str(tmp_path / "out"))) == ["missing.txt", "profiles.npz"]


def test_combine_empty_merged_files(tmp_path):
    """Tags without points (no parts; a part without points) against the frozen legacy: status widens to <U32."""
    run, exp = _make_run(tmp_path)
    res = os.path.join(run, "results")
    os.makedirs(os.path.join(res, "task_000"))                                  # no parts
    _write_part(os.path.join(res, "task_0002", "part_1.tar.gz"), [])                     # no points
    tags = ["task_000", "task_0000", "task_0001", "task_0002", "task_missing_0000"]
    for t in tags:
        _run_legacy_merge(run, "--tags", t, "--out", "merged/{}.npz".format(t))
        s = fw.merge_task(res, t, os.path.join(run, "new_merged", t + ".npz"), os.path.join(run, "points.npz"))
        assert _sha(os.path.join(run, "new_merged", t + ".npz")) == _sha(os.path.join(run, "merged", t + ".npz")), t
        if t in ("task_000", "task_0002"):
            assert s["npoint"] == 0 and s["status"] == {}
    with np.load(os.path.join(run, "merged", "task_000.npz")) as z:
        assert z["status"].dtype == np.float64 and z["status"].size == 0
    _run_legacy_merge(run, "--combine")
    for block in (2, 20000):
        p, m = fw.combine(os.path.join(run, "new_merged", "task_*.npz"), os.path.join(run, "points.txt"),
                          str(tmp_path / "new{}".format(block)), block=block)
        assert _sha(p) == _sha(os.path.join(run, "profiles.npz"))
        assert _sha(m) == _sha(os.path.join(run, "missing.txt"))
    with np.load(p) as z:
        assert z["status"].dtype == np.dtype("<U32")
        np.testing.assert_array_equal(z["idx"], sorted(exp["status"]))
    # only empty files
    p, m = fw.combine([os.path.join(run, "new_merged", "task_000.npz")], os.path.join(run, "points.txt"),
                      str(tmp_path / "empty"))
    with np.load(p) as z:
        assert z["idx"].size == 0 and z["lam"].shape == (0, 3, NROW)
    assert np.loadtxt(m, ndmin=2).shape == (NPTS, 2)
    assert fw._status_ok(np.zeros(0), 0).dtype == bool


def test_npz_stream_checks_member_bytes(tmp_path):
    out = str(tmp_path / "x.npz")
    for nwrite in (5, 7):                                    # short and overlong data
        w = fw._NpzStream(out)
        assert os.path.basename(w.tmp).startswith(".x.tmp")
        with pytest.raises(ValueError, match="data bytes"):
            try:
                with w.member("a", (6,), np.float32) as fid:
                    fid.write(np.zeros(nwrite, np.float32))
                w.close()
            except BaseException:
                w.abort()
                raise
        assert os.listdir(str(tmp_path)) == []
    w = fw._NpzStream(out)
    with w.member("a", (2, 3), "<U3") as fid:
        fid.write(np.array([["ab", "c", ""]], "<U3"))
        fid.write(np.array([["d", "ef", "ghi"]], "<U3").tobytes())
    w.write_array("b", np.arange(3))
    w.close()
    b = str(tmp_path / "ref.npz")
    np.savez(b, a=np.array([["ab", "c", ""], ["d", "ef", "ghi"]], "<U3"), b=np.arange(3))
    assert _sha(out) == _sha(b)


def test_member_fallback_warns(synth, tmp_path, monkeypatch):
    """A stored member that cannot be memory-mapped is loaded with a FullLoadWarning (strict: the error)."""
    path = os.path.join(synth["run"], "profiles.npz")

    def no_mmap(p, key):
        raise OSError("mmap not supported here")

    monkeypatch.setattr(fw, "npz_member_memmap", no_mmap)
    with pytest.warns(fw.FullLoadWarning, match="cannot be memory-mapped") as rec:
        s = fw.ProfileStore.open(path)
    msg = [str(r.message) for r in rec]
    assert any(re.search(r"member 'fnorm' \(.* GB\) cannot be memory-mapped \(OSError: mmap not supported", x) for x in msg)
    assert len(msg) == len(np.load(path).files)
    assert not isinstance(s["fnorm"], np.memmap)
    np.testing.assert_array_equal(s.fnorm(1), np.load(path)["fnorm"][:, 1])
    with pytest.raises(OSError):
        fw.ProfileStore.open(path, strict=True)
    # combine falls back to loading (with the warning) and still writes the same bytes
    run = synth["run"]
    with pytest.warns(fw.FullLoadWarning):
        p, _ = fw.combine(os.path.join(run, "merged", "task_*.npz"), os.path.join(run, "points.txt"), str(tmp_path))
    assert _sha(p) == _sha(path)


def test_combine_meta_and_errors(synth, tmp_path):
    run = synth["run"]
    files = os.path.join(run, "new_merged", "task_*.npz")
    p, _ = fw.combine(files, os.path.join(run, "points.txt"), str(tmp_path), meta=dict(kind="test"),
                      profiles_name="p_meta.npz", missing_name="m.txt")
    from ppmpy.synspec.io import read_meta
    assert read_meta(p) == dict(kind="test")
    ref = np.load(os.path.join(run, "profiles.npz"))
    with np.load(p) as z:
        for k in ref.files:
            assert z[k].tobytes() == ref[k].tobytes() and z[k].dtype == ref[k].dtype
    with pytest.raises(ValueError):
        fw.combine([], os.path.join(run, "points.txt"), str(tmp_path))


def _merge_one_pid(args):
    """fw._merge_one that also reports its process (module level: picklable for the pool)."""
    s = fw.merge_task(*args[:4], **args[4])
    s["pid"] = os.getpid()
    return s


NEED_FORK = pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="needs the fork start method")


@NEED_FORK
def test_merge_tasks_fresh_process_per_tag(synth, tmp_path, monkeypatch):
    """nproc = 1 with several tags: one worker, but a new process for every tag (CPU-time limit, memory)."""
    run = synth["run"]
    monkeypatch.setattr(fw, "_merge_one", _merge_one_pid)
    s = fw.merge_tasks(os.path.join(run, "results"), str(tmp_path / "m"), os.path.join(run, "points.npz"), nproc=1,
                       start_method="fork")
    pids = [x["pid"] for x in s]
    assert len(pids) == 3 and len(set(pids)) == 3 and os.getpid() not in pids
    # a single tag runs in this process
    s = fw.merge_tasks(os.path.join(run, "results"), str(tmp_path / "m1"), os.path.join(run, "points.npz"),
                       tags=["task_0001"], nproc=4, start_method="fork")
    assert [x["pid"] for x in s] == [os.getpid()]
    for t in synth["tags"]:
        assert _sha(str(tmp_path / "m" / (t + ".npz"))) == _sha(os.path.join(run, "merged", t + ".npz"))


@NEED_FORK
@pytest.mark.skipif(os.name != "posix", reason="stale temporaries are removed on POSIX only")
def test_merge_tasks_failure_removes_stale_temporaries(tmp_path):
    run, _ = _make_run(tmp_path)
    teff = np.load(os.path.join(run, "points.npz"))["teff"]
    _write_part(os.path.join(run, "results", "task_0005", "part_1.tar.gz"),
                [(1, "{:.3f}".format(teff[1] + 25.0), "ok", 60, "39000.0", 1)])          # merge_task raises
    md = tmp_path / "merged"
    md.mkdir()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    stale = md / ".task_0001.tmp{}.npz".format(dead.pid)            # left by a writer that no longer runs
    live = md / ".task_0001.tmp{}.npz".format(os.getpid())          # a writer that still runs: kept
    other = md / ".task_9999.tmp{}.npz".format(dead.pid)            # not a tag of this call: kept
    for f in (stale, live, other):
        f.write_bytes(b"partial")
    with pytest.raises(ValueError, match="nudge"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fw.merge_tasks(os.path.join(run, "results"), str(md), os.path.join(run, "points.npz"), nproc=2,
                       start_method="fork")
    assert not stale.exists() and live.exists() and other.exists()
    assert not glob.glob(str(md / "task_0005*"))


@pytest.mark.parametrize("start_method", ["fork", "spawn"])
def test_merge_tasks_parallel(synth, tmp_path, start_method):
    run = synth["run"]
    out = str(tmp_path / "merged")
    s = fw.merge_tasks(os.path.join(run, "results"), out, os.path.join(run, "points.npz"), nproc=3,
                       start_method=start_method)
    assert [os.path.basename(x["path"]) for x in s] == [t + ".npz" for t in synth["tags"]]
    for t in synth["tags"]:
        assert _sha(os.path.join(out, t + ".npz")) == _sha(os.path.join(run, "merged", t + ".npz"))
    # incremental: nothing newer -> nothing merged; a newer part -> only that tag
    assert fw.merge_tasks(os.path.join(run, "results"), out, os.path.join(run, "points.npz"), nproc=2,
                          start_method=start_method) == []
    part = glob.glob(os.path.join(run, "results", "task_0001", "part_*.tar.gz"))[0]
    st = os.stat(part)
    try:
        os.utime(part, (st.st_atime, os.path.getmtime(os.path.join(out, "task_0001.npz")) + 10))
        s2 = fw.merge_tasks(os.path.join(run, "results"), out, os.path.join(run, "points.npz"), nproc=2,
                            start_method=start_method)
        assert [os.path.basename(x["path"]) for x in s2] == ["task_0001.npz"]
    finally:
        os.utime(part, (st.st_atime, st.st_mtime))


# ----------------------------------------------------------------------------------------------
# ProfileStore, EW, summary (synthetic)
# ----------------------------------------------------------------------------------------------
def test_profile_store(synth):
    path = os.path.join(synth["run"], "profiles.npz")
    s = fw.ProfileStore.open(path)
    ref = np.load(path)
    assert s.n == ref["idx"].size and s.lines == LINES and s.nrow == NROW and s.meta == {}
    for k in ("idx", "teff", "status", "niter", "teff_nudge"):
        a = getattr(s, k)
        assert isinstance(a, np.memmap) and a.dtype == ref[k].dtype
        np.testing.assert_array_equal(a, ref[k])
    assert sorted(s.coordinates) == ["phi", "r", "theta", "x", "y", "z"]
    np.testing.assert_array_equal(s["ur_kms"], ref["ur_kms"])
    for j in (0, "HEII4200", 2):
        jj = s.lines.index(j) if isinstance(j, str) else j
        v = s.lam(j)
        assert v.shape == (s.n, NROW) and np.shares_memory(v, s["lam"])
        np.testing.assert_array_equal(v, ref["lam"][:, jj])
        np.testing.assert_array_equal(s.fnorm(j), ref["fnorm"][:, jj])
        np.testing.assert_array_equal(s.fcont(j), ref["fcont"][:, jj])
        f = s.fc(j)
        assert f.dtype == np.float64 and f.shape == (s.n,)
        np.testing.assert_array_equal(f, ref["fcont"][:, jj, 0])
    np.testing.assert_array_equal(s.fc(None), ref["fcont"][:, :, 0])
    # blocks: all points, a row list and a mask
    L = np.concatenate([b[2] for b in s.iter_blocks(1, block=3)])
    assert [b[:2] for b in s.iter_blocks(1, block=4)] == [(0, 4), (4, 8), (8, 10)]
    np.testing.assert_array_equal(L, ref["lam"][:, 1].astype(np.float64))
    rows = np.array([1, 2, 5, 9])
    got = list(s.iter_blocks("HEI4922", block=3, rows=rows))
    assert [g[:2] for g in got] == [(0, 3), (3, 4)]
    assert got[0][2].dtype == np.float64 and got[0][3].dtype == np.float32
    np.testing.assert_array_equal(np.concatenate([g[3] for g in got]), ref["fnorm"][rows, 2])
    mask = np.zeros(s.n, bool)
    mask[rows] = True
    for a, b in zip(got, s.iter_blocks(2, block=3, rows=mask)):
        np.testing.assert_array_equal(a[2], b[2])
    # usable: ok; the cap is not recorded in a legacy file and never inferred from max(niter)
    ok = ref["status"] == "ok"
    np.testing.assert_array_equal(s.usable(), ok)
    assert s.niter_cap is None
    with pytest.raises(ValueError, match="cap"):
        s.usable(cap_ok=False)
    np.testing.assert_array_equal(s.usable(cap_ok=False, cap=102), ok & (ref["niter"] < 102))
    np.testing.assert_array_equal(s.usable(finite=True, block=3), ok)
    # pickling re-opens the file (no copy of the arrays), with memory maps also for a store opened with mmap=False
    s2 = pickle.loads(pickle.dumps(s))
    assert isinstance(s2["fnorm"], np.memmap) and s2.path == s.path
    s3 = fw.ProfileStore.open(path, mmap=False)
    assert not isinstance(s3["fnorm"], np.memmap)
    np.testing.assert_array_equal(s3.fnorm(0), s.fnorm(0))
    s4 = pickle.loads(pickle.dumps(s3))
    assert isinstance(s4["fnorm"], np.memmap) and s4.path == s.path
    np.testing.assert_array_equal(s4.fnorm(2), s.fnorm(2))


def test_profile_store_rows_checked(synth):
    s = fw.ProfileStore.open(os.path.join(synth["run"], "profiles.npz"))
    good = np.zeros(s.n, bool)
    good[[0, 4]] = True
    assert [b[:2] for b in s.iter_blocks(0, rows=good)] == [(0, 2)]
    assert list(s.iter_blocks(0, rows=[])) == []
    # errors are raised by the call itself, not at the first block
    for bad, err in [(np.ones(3, bool), ValueError), (np.ones(s.n + 1, bool), ValueError), ([0, s.n], ValueError),
                     ([-1, 2], ValueError), ([0.0, 1.0], TypeError), (np.zeros((2, 2), int), ValueError)]:
        with pytest.raises(err):
            s.iter_blocks(0, rows=bad)


def test_profile_store_pickle_from_arrays():
    """A store built from arrays sends its arrays and its '_meta' (niter_cap included)."""
    n, nrow = 4, 5
    m = dict(idx=np.arange(n, dtype=np.int32), lam=np.ones((n, 2, nrow), np.float32), fcont=np.ones((n, 2, nrow), np.float32),
             fnorm=np.ones((n, 2, nrow), np.float32), lines=np.array(["A", "B"]), niter=np.array([57, 58, 61, 61], np.int16),
             status=np.array(["ok"] * n), _meta=np.array(json.dumps(dict(kind="test", niter_cap=102))))
    s = fw.ProfileStore(m)
    assert s.meta == dict(kind="test", niter_cap=102) and s.niter_cap == 102 and "_meta" not in s
    s2 = pickle.loads(pickle.dumps(s))
    assert s2.meta == s.meta and s2.niter_cap == 102 and s2.path is None
    np.testing.assert_array_equal(s2["niter"], m["niter"])
    s3 = pickle.loads(pickle.dumps(fw.ProfileStore({k: v for k, v in m.items() if k != "_meta"})))
    assert s3.meta == {} and s3.niter_cap is None


def test_profile_store_compressed(synth, tmp_path):
    ref = np.load(os.path.join(synth["run"], "profiles.npz"))
    p = str(tmp_path / "c.npz")
    np.savez_compressed(p, **{k: ref[k] for k in ref.files})
    s = fw.ProfileStore.open(p)
    assert not isinstance(s["lam"], np.memmap)
    np.testing.assert_array_equal(s.lam(2), ref["lam"][:, 2])


def _legacy_ew(p, n, chunk, cache):
    """The EW block of the frozen fig_fw_sphere_ew.py, executed as it stands."""
    with open(_legacy_file("fig_fw_sphere_ew.py")) as fh:
        src = fh.read().splitlines()
    i0 = next(i for i, l in enumerate(src) if l.strip().startswith('lam0 = p["lam"][0]'))
    i1 = next(i for i, l in enumerate(src) if l.strip() == 'print("wrote", cache)')
    code = "\n".join(l[4:] for l in src[i0:i1 + 1])
    ns = dict(np=np, os=os, p=p, n=n, CHUNK=chunk, cache=cache, idx=p["idx"], teff=p["teff"])
    exec(compile(code, "fig_fw_sphere_ew.py", "exec"), ns)
    return ns["ew"]


def test_ew_per_point_matches_legacy(synth, tmp_path):
    path = os.path.join(synth["run"], "profiles.npz")
    p = np.load(path)
    cache = str(tmp_path / "ew_legacy.npz")
    old = _legacy_ew(p, p["idx"].size, 4, cache)
    s = fw.ProfileStore.open(path)
    for block in (1, 3, 100000):
        new = fw.ew_per_point(s, block=block)
        assert new.dtype == np.float64 and new.tobytes() == old.tobytes()
    out = str(tmp_path / "ew.npz")
    new, chk = fw.ew_per_point(path, out=out, checks=True)
    assert _sha(out) == _sha(cache)
    nbad = int((~np.isfinite(p["fnorm"])).any(axis=(1, 2)).sum())
    assert chk["nbad"] == nbad == 2
    ok = p["status"] == "ok"
    # analytic check: EW of the Gaussian lines ~ depth sqrt(pi) w (trapezoid on the 0.5 A grid)
    assert np.all(new[ok] > 0.5) and np.all(np.isnan(new[~ok]))


def test_status_summary(synth):
    path = os.path.join(synth["run"], "profiles.npz")
    s = fw.status_summary(path)
    assert s["n"] == 10 and not s["complete"]
    assert s["status"] == {"ok": 8, "pnlte_failed": 1, "pnlte_timeout": 1}
    # iterations of the 8 successful models only (7 failed at 19, 2 timed out at 0); no cap given -> none reported
    ni = s["niter"]
    assert ni["n"] == 8 and ni["min"] == 59 and ni["max"] == 102 and ni["n_at_max"] == 2 and ni["hist"][102] == 2
    assert 19 not in ni["hist"] and 0 not in ni["hist"]
    assert "cap" not in ni and "n_at_cap" not in ni and "median_at_cap" not in s["t_pnlte"]
    assert s["teff_nudge"] == dict(n=1, max=1.0, idx=[6])
    tp = np.load(path)["t_pnlte"]
    assert s["cpu_hours"] == pytest.approx((tp.astype(np.float64).sum() + 10 * 0.5) / 3600, rel=1e-6)
    s = fw.status_summary(path, cap=102)
    assert s["niter"]["cap"] == 102 and s["niter"]["n_at_cap"] == 2 and s["niter"]["n_above_cap"] == 0
    assert s["t_pnlte"]["median_at_cap"] == 156.5                 # points 5 and 8: 155.0 and 158.0 s
    ok_below = [150.0 + x for x in (0, 1, 33, 4, 66, 10)]          # seeds of the ok models below the cap
    assert s["t_pnlte"]["median_below_cap"] == np.median(np.float32(ok_below))


def test_niter_cap_not_inferred():
    """A run in which no model reaches the cap (as the ITMORE = 300 reruns, 57-61 iterations): nothing is dropped."""
    n = 6
    niter = np.array([57, 58, 59, 60, 61, 61], np.int16)
    m = dict(idx=np.arange(n, dtype=np.int32), lam=np.ones((n, 1, 3), np.float32), fcont=np.ones((n, 1, 3), np.float32),
             fnorm=np.ones((n, 1, 3), np.float32), lines=np.array(["A"]), niter=niter, status=np.array(["ok"] * n),
             t_pnlte=np.arange(n, dtype=np.float32) + 170)
    s = fw.ProfileStore(m)
    with pytest.raises(ValueError, match="cap"):
        s.usable(cap_ok=False)
    assert s.usable(cap_ok=False, cap=302).all()                    # both converged models at 61 are kept
    assert s.usable(cap_ok=False, cap=fw.NITER_CAP_M424).all()
    with pytest.warns(UserWarning, match="niter > cap"):
        u = s.usable(cap_ok=False, cap=60)                           # a cap below max(niter) belongs to another run
    np.testing.assert_array_equal(u, niter < 60)
    st = fw.status_summary(s)
    assert st["niter"]["max"] == 61 and st["niter"]["n_at_max"] == 2 and "n_at_cap" not in st["niter"]
    st = fw.status_summary(s, cap=302)
    assert st["niter"]["n_at_cap"] == 0 and st["niter"]["n_above_cap"] == 0
    assert np.isnan(st["t_pnlte"]["median_at_cap"]) and st["t_pnlte"]["median_below_cap"] == 172.5
    # a cap recorded in '_meta' is the default
    m["_meta"] = np.array(json.dumps(dict(niter_cap=302)))
    s = fw.ProfileStore(m)
    assert s.niter_cap == 302 and s.usable(cap_ok=False).all()
    assert fw.status_summary(s)["niter"]["cap"] == 302


INDAT_M424 = """M424test                                       CATALOG
T  T   0   100                                 OPTNEUPDATE,HE_ONE,ITSTART,ITMORE
0.                                             OPTMIXED
38230.0,     4.25000,    6.2100                TEFF, LOG G, RSTAR
"""


def test_niter_cap_from_indat(tmp_path):
    assert fw.niter_cap_from_indat(INDAT_M424) == 102 == fw.NITER_CAP_M424
    f = tmp_path / "INDAT.DAT"
    f.write_text(INDAT_M424.replace("0   100", "0   300"))
    assert fw.niter_cap_from_indat(str(f)) == 302
    assert fw.niter_cap_from_indat(INDAT_M424.replace("T  T   0   100", "T,T,0,100"), extra=0) == 100
    with pytest.raises(ValueError, match="ITSTART"):
        fw.niter_cap_from_indat(INDAT_M424.replace("0   100", "40   100"))
    with pytest.raises(ValueError, match="line 2"):
        fw.niter_cap_from_indat("name\nT T\n")


# ----------------------------------------------------------------------------------------------
# bounded memory of the streaming combine and the block readers
# ----------------------------------------------------------------------------------------------
NFILE_MEM, NREC_MEM = 20, 40000


@pytest.fixture(scope="module")
def big_merged(tmp_path_factory):
    """NFILE_MEM merged files with NREC_MEM records in all (point i in file i % NFILE_MEM), written with np.savez."""
    d = tmp_path_factory.mktemp("bigmerged")
    rng = np.random.default_rng(11)
    for t in range(NFILE_MEM):
        i = np.arange(t, NREC_MEM, NFILE_MEM, dtype=np.int32)
        n = i.size
        prof = {k: (rng.random((n, 3, NROW), np.float32) + off) for k, off in (("lam", 4000.0), ("fcont", 0.0), ("fnorm", 0.5))}
        prof["lam"] = np.sort(prof["lam"], axis=2)
        arrs = dict(idx=i, teff=38000.0 + rng.random(n), status=np.array(["ok"] * n), niter=np.full(n, 60, np.int16),
                    T_tau23=39000.0 + rng.random(n), t_pnlte=np.full(n, 150.0, np.float32), t_formal=np.full(n, 0.5, np.float32),
                    lines=np.array(LINES))
        arrs.update(prof)
        for k in COORDS_ALL:
            arrs[k] = rng.random(n)
        arrs["teff_nudge"] = np.zeros(n, np.float32)
        np.savez(str(d / "task_{:04d}.npz".format(t)), **arrs)
    np.savetxt(str(d / "points.txt"), np.column_stack([np.arange(NREC_MEM + 7), np.full(NREC_MEM + 7, 38000.0)]),
               fmt=["%d", "%.3f"])
    return d


COORDS_ALL = ("r", "theta", "phi", "x", "y", "z", "ur_kms", "relT")


def _traced_peak(func, *args, **kw):
    tracemalloc.start()
    try:
        out = func(*args, **kw)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return out, peak


def test_streaming_memory_bounded(big_merged, tmp_path):
    """combine, ew_per_point and usable(finite) hold O(block) of the profiles, not whole members (tracemalloc)."""
    d = big_merged
    prof_bytes = NREC_MEM * 3 * NROW * 4 * 3                           # 232 MB of lam, fcont, fnorm
    (p, m), peak_c = _traced_peak(fw.combine, str(d / "task_*.npz"), str(d / "points.txt"), str(tmp_path), block=500)
    s = fw.ProfileStore.open(p)
    ew, peak_e = _traced_peak(fw.ew_per_point, s, block=500)
    ok, peak_u = _traced_peak(s.usable, finite=True, block=500)
    # measured 2026-10-01: ~7 / 11 / 0.3 MB (the selection arrays, O(records), dominate the combine)
    assert peak_c < prof_bytes / 20 and peak_e < prof_bytes / 15 and peak_u < prof_bytes / 50, (peak_c, peak_e, peak_u)
    assert s.n == NREC_MEM and ok.all() and np.isfinite(ew).all()
    # the content, against a direct gather
    with np.load(str(d / "task_0003.npz")) as z:
        np.testing.assert_array_equal(np.asarray(s["fnorm"][3::NFILE_MEM]), z["fnorm"])
    assert np.loadtxt(m, ndmin=2)[:, 0].tolist() == list(range(NREC_MEM, NREC_MEM + 7))


_RSS_SCRIPT = """
import os, sys
sys.path.insert(0, {root!r})
import numpy as np
from ppmpy.synspec import fwresults as fw
def hwm():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1]) * 1024
fw.combine({first!r}, {ptxt!r}, {out0!r}, block=500)          # warm-up on one file (imports, small arrays)
h0 = hwm()
fw.combine({files!r}, {ptxt!r}, {out!r}, block=500)
print(h0, hwm())
"""


@pytest.mark.skipif(not os.path.exists("/proc/self/status"), reason="needs /proc (Linux)")
def test_combine_resident_memory_bounded(big_merged, tmp_path):
    """No memory maps in the gather: the peak resident set (file pages included) does not grow with the files
    (measured 2026-10-01: +0 MB; the first version, with memory maps of every file, +72 MB ~ the total size of the
    largest member, 77 MB)."""
    d = big_merged
    code = _RSS_SCRIPT.format(root=ROOT, first=[str(d / "task_0000.npz")], files=str(d / "task_*.npz"),
                              ptxt=str(d / "points.txt"), out0=str(tmp_path / "w"), out=str(tmp_path / "o"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    h0, h1 = map(int, r.stdout.split())
    largest_member = NREC_MEM * 3 * NROW * 4
    assert h1 - h0 < largest_member / 4, (h0, h1)


# ----------------------------------------------------------------------------------------------
# M424 regressions
# ----------------------------------------------------------------------------------------------
@pytest.mark.m424
def test_m424_read_out():
    f = _imu_file("OUT.HEI4026_VTV010")
    r = fw.read_out(f)
    assert r["nrow"] == NROW and r["ew_fastwind"] < 0
    g = np.genfromtxt(f, max_rows=NROW)
    for c, k in enumerate(("x", "lam", "fcont", "fnorm", "frot")):
        assert r[k].tobytes() == g[:, c + 1].tobytes()
    with open(f) as fh:
        assert float(fh.read().split("\n")[NROW]) == r["ew_fastwind"]


@pytest.mark.m424
def test_m424_read_out_imu():
    sys.path.insert(0, LEGACY)
    try:
        import fw_disc as legacy
    finally:
        sys.path.remove(LEGACY)
    for ln in LINES:
        f = _imu_file("OUT_IMU.{}_VTV010".format(ln))
        new = fw.read_out_imu(f, counts=True)
        old = legacy.read_imu(f)
        for a, b in zip(new[:4], old):
            assert a.tobytes() == b.tobytes() and a.shape == b.shape
        assert new[4] == dict(nray=79, ncore=10)


@pytest.mark.m424
def test_m424_merge_task_bitwise(tmp_path):
    res = m424_path("run", "results", "task_missing_0000")
    ref = m424_path("run", "merged", "task_missing_0000.npz")
    out = str(tmp_path / "task_missing_0000.npz")
    s = fw.merge_task(os.path.dirname(res), "task_missing_0000", out, m424_path("run", "points.npz"))
    assert s["npoint"] == 2 and s["nnudge"] == 2 and s["status"] == {"ok": 2}
    assert _sha(out) == _sha(ref)


@pytest.mark.m424
def test_m424_combine_matches_legacy(tmp_path):
    """combine of task_missing_0000 alone == the frozen legacy --combine on a copy of the same file."""
    run = tmp_path / "run"
    (run / "merged").mkdir(parents=True)
    shutil.copy(m424_path("run", "merged", "task_missing_0000.npz"), str(run / "merged"))
    os.symlink(m424_path("run", "points.txt"), str(run / "points.txt"))
    _run_legacy_merge(str(run), "--combine")
    p, m = fw.combine(str(run / "merged" / "task_*.npz"), str(run / "points.txt"), str(tmp_path / "new"))
    assert _sha(p) == _sha(str(run / "profiles.npz"))
    assert _sha(m) == _sha(str(run / "missing.txt"))
    with open(m) as fh:
        assert sum(1 for _ in fh) == 1236544 - 2


@pytest.mark.m424
def test_m424_part_parse_matches_merged():
    """The first 40 points of a production part, parsed by fwresults, equal merged/task_0000.npz (and genfromtxt)."""
    part = sorted(glob.glob(os.path.join(m424_path("run", "results", "task_0000"), "part_*.tar.gz")))[0]
    merged = m424_path("run", "merged", "task_0000.npz")
    idx = npz_member_memmap(merged, "idx")
    lam, fn, fc = (npz_member_memmap(merged, k) for k in ("lam", "fnorm", "fcont"))
    n = 0
    for pdir, files in fw.iter_part_points(part):
        m = fw.parse_meta(files["meta.txt"])
        row = int(np.searchsorted(idx, m["idx"]))
        assert idx[row] == m["idx"]
        for j, ln in enumerate(LINES):
            b = files["OUT.{}_VTV010".format(ln)]
            t = fw._out_table(b, NROW, (2, 3, 4))
            assert t.tobytes() == np.genfromtxt(io.BytesIO(b), usecols=[2, 3, 4], max_rows=NROW).tobytes()
            for c, A in enumerate((lam, fc, fn)):
                assert t[:, c].astype(np.float32).tobytes() == np.asarray(A[row, j]).tobytes()
        n += 1
        if n == 40:
            break
    assert n == 40


@pytest.mark.m424
def test_m424_profile_store():
    path = m424_path("run", "profiles.npz")
    s = fw.ProfileStore.open(path)
    assert s.n == 1236544 and s.lines == LINES and s.nrow == NROW
    with np.load(path) as z:
        for k in ("idx", "teff", "niter"):
            ref = z[k]
            a = getattr(s, k)
            assert isinstance(a, np.memmap) and a.dtype == ref.dtype and a.tobytes() == ref.tobytes()
    # a few rows of lam/fnorm against the merged task file they came from (point i of points.txt -> task i % 40)
    merged = m424_path("run", "merged", "task_0007.npz")
    with np.load(merged) as z:
        mi, ml, mf = z["idx"], z["lam"], z["fnorm"]
    pick = [0, 1, 777, mi.size - 1]
    rows = mi[pick]
    for j in range(3):
        assert np.asarray(s.lam(j)[rows]).tobytes() == ml[pick, j].tobytes()
        assert np.asarray(s.fnorm(j)[rows]).tobytes() == mf[pick, j].tobytes()
        got = list(s.iter_blocks(j, block=2, rows=rows))
        np.testing.assert_array_equal(np.concatenate([g[3] for g in got]), mf[pick, j])
    assert s.usable().sum() == 1236544
    assert s.niter_cap is None                                   # legacy file: the cap is not recorded
    assert s.usable(cap_ok=False, cap=fw.NITER_CAP_M424).sum() == 1236544 - 230649


@pytest.mark.m424
def test_m424_status_summary():
    s = fw.status_summary(m424_path("run", "profiles.npz"))
    assert s["n"] == 1236544 and s["complete"] and s["status"] == {"ok": 1236544}
    assert s["niter"]["max"] == 102 and s["niter"]["n_at_max"] == 230649 and "cap" not in s["niter"]
    s = fw.status_summary(m424_path("run", "profiles.npz"), cap=fw.NITER_CAP_M424)
    assert s["niter"]["cap"] == 102 and s["niter"]["n_at_cap"] == 230649 and s["niter"]["n_above_cap"] == 0
    assert s["teff_nudge"]["n"] == 2 and s["teff_nudge"]["idx"] == [571348, 850281] and s["teff_nudge"]["max"] == 1.0
    # project log 2026-09-30: median pnlte time 352 s at the cap vs 179 s (exactly 351.6 and 178.5 s)
    assert s["t_pnlte"]["median_at_cap"] == pytest.approx(351.6, abs=1e-4)
    assert s["t_pnlte"]["median_below_cap"] == 178.5


@pytest.mark.m424
def test_m424_niter_cap_from_indat():
    """ITMORE + 2 = the niter of models that hit the cap: the archived M424 INDATs (ITMORE 100 -> 102 = max niter
    of the run) and the ITMORE = 300 reruns of the ITMORE check (pnlte.log of unconverged models: 302 lines)."""
    if not os.path.isdir(os.path.join(ITMORE_CHECK, "runs")):
        pytest.skip("ITMORE check not available: {}".format(ITMORE_CHECK))
    lst = np.loadtxt(os.path.join(ITMORE_CHECK, "list.txt"), ndmin=2).astype(np.int64)   # idx teff niter capped
    nrun = ncap = 0
    for idx, _, niter_old, capped in lst:
        name = "P{:06d}".format(idx)
        assert fw.niter_cap_from_indat(os.path.join(ITMORE_CHECK, "orig", name, "INDAT.DAT")) == fw.NITER_CAP_M424
        assert (niter_old == fw.NITER_CAP_M424) == bool(capped)
        log = os.path.join(ITMORE_CHECK, "runs", name, "pnlte.log")
        if not os.path.exists(log):
            continue
        cap = fw.niter_cap_from_indat(os.path.join(ITMORE_CHECK, "inputs", name + ".DAT"))
        with open(log, errors="replace") as f:
            n = sum("ITERATION NO" in ln for ln in f)                  # as fw_sphere_point.sh: grep -c
        assert cap == 302 and n <= cap
        nrun += 1
        ncap += n == cap
    assert nrun >= 40 and ncap >= 30                               # 38 of 40 capped models stop at 300 again
    assert int(np.max(fw.ProfileStore.open(m424_path("run", "profiles.npz")).niter)) == fw.NITER_CAP_M424


@pytest.mark.m424
@pytest.mark.slow
def test_m424_combine_subset_matches_legacy(tmp_path):
    """Two full production tags and the retry tag (links, not copies): combine == the frozen legacy --combine."""
    run = tmp_path / "run"
    (run / "merged").mkdir(parents=True)
    for t in ("task_0000", "task_0001", "task_missing_0000"):
        os.symlink(m424_path("run", "merged", t + ".npz"), str(run / "merged" / (t + ".npz")))
    os.symlink(m424_path("run", "points.txt"), str(run / "points.txt"))
    _run_legacy_merge(str(run), "--combine")
    p, m = fw.combine(str(run / "merged" / "task_*.npz"), str(run / "points.txt"), str(tmp_path / "new"))
    assert _sha(p) == _sha(str(run / "profiles.npz"))
    assert _sha(m) == _sha(str(run / "missing.txt"))
    want = np.unique(np.concatenate([npz_member_memmap(str(f), "idx") for f in sorted((run / "merged").iterdir())]))
    with np.load(p) as z:
        np.testing.assert_array_equal(z["idx"], want)
        assert (z["status"] == "ok").all() and z["status"].dtype == np.dtype("<U12")    # 850281: the retry is kept


@pytest.mark.m424
@pytest.mark.slow
def test_m424_ew_per_point(tmp_path):
    ref = m424_path("run", "ew.npz")
    out = str(tmp_path / "ew.npz")
    ew = fw.ew_per_point(m424_path("run", "profiles.npz"), out=out)
    with np.load(ref) as z:
        assert ew.tobytes() == z["ew"].tobytes()
    assert _sha(out) == _sha(ref)
