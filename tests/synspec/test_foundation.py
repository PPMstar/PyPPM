"""Tests of conventions, spectral and io."""
import ast
import glob
import os

import numpy as np
import pytest

from conftest import ROOT, m424_path
from ppmpy.synspec import conventions as cv
from ppmpy.synspec import io as sio
from ppmpy.synspec import spectral as sp


def test_python39_syntax():
    for f in glob.glob(os.path.join(ROOT, "ppmpy", "synspec", "*.py")):
        ast.parse(open(f).read(), filename=f, feature_version=(3, 9))


def test_no_heavy_imports():
    import subprocess
    import sys
    code = ("import sys; import ppmpy.synspec.conventions, ppmpy.synspec.spectral, ppmpy.synspec.io;"
            "bad=[m for m in ('ppmpy.ppm','matplotlib','nugridpy','pyshtools') if m in sys.modules];"
            "print(','.join(bad))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                         env=dict(os.environ, PYTHONPATH=ROOT))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", "heavy modules imported: " + out.stdout


def test_units():
    assert cv.muhz_to_cpd(1.0) == pytest.approx(0.0864)
    assert cv.cpd_to_muhz(1.0) == pytest.approx(11.574074, rel=1e-6)


def test_los_sets():
    t = cv.los_thompson2024()
    f = cv.los_ppmstar_fortran()
    assert t.shape == f.shape == (8, 3)
    np.testing.assert_allclose(np.linalg.norm(t, axis=1), 1.0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(f, axis=1), 1.0, rtol=0, atol=1e-15)
    # documented relation between the two sets
    dots = np.einsum("ij,ij->i", t, f)
    np.testing.assert_allclose(dots, [1, -1, -1, -1 / 3, 1, -1, -1, -1 / 3], atol=1e-12)
    np.testing.assert_array_equal(cv.los_array("thompson2024"), t)
    np.testing.assert_allclose(np.linalg.norm(cv.los_array("fibonacci:10"), axis=1), 1.0, atol=1e-15)
    with pytest.raises(ValueError):
        cv.los_array("default")


def test_los_fortran_matches_ppm():
    ppm = pytest.importorskip("ppmpy.ppm")
    ref = np.array(ppm.make_los_vectors_fortran())
    np.testing.assert_array_equal(cv.los_ppmstar_fortran(), ref)


@pytest.mark.m424
def test_los_thompson_matches_products():
    d = np.load(m424_path("run", "disc_los8.npz"))
    np.testing.assert_array_equal(cv.los_thompson2024(), d["los"])


def test_velocity_grid_m424_defaults():
    g = sp.VelocityGrid()
    assert g.ny == 5401 and g.nshift == 400 and g.y[g.icentre] == 0.0
    np.testing.assert_array_equal(g.y, np.arange(-2700.0, 2700.0 + 0.5, 1.0))


def test_shift_steps():
    g = sp.VelocityGrid(dv=0.5, vmax=100.0, vshift=20.0)
    v = np.linspace(-30, 30, 1001)
    s, clipped = g.shift_steps(v)
    exact = -cv.C_KMS * np.log(1.0 - v / cv.C_KMS) / g.dv
    ok = ~clipped
    assert np.all(np.abs(s[ok] - exact[ok]) <= 0.5 + 1e-12)
    assert np.all(np.abs(s) <= g.nshift) and clipped.sum() > 0
    assert np.all(np.abs(exact[clipped]) > g.nshift - 0.5)


def test_lam_y_roundtrip():
    lam = np.linspace(4000, 4050, 7)
    np.testing.assert_allclose(sp.lam_of_y(sp.y_of_lam(lam, 4026.22), 4026.22), lam, rtol=1e-14)
    # a Doppler shift by v moves a line by -c ln(1 - v/c) on the y grid
    v = 123.4
    dy = sp.y_of_lam(sp.doppler_lambda(4026.22, v), 4026.22)
    assert dy == pytest.approx(cv.C_KMS * np.log(1 - v / cv.C_KMS), rel=1e-12)


def test_interp_rows_matches_np_interp():
    rng = np.random.default_rng(1)
    L = np.sort(rng.uniform(-50, 50, size=(20, 31)), axis=1)
    F = rng.standard_normal((20, 31))
    g = np.linspace(-60, 60, 241)
    out = sp.interp_rows(L, F, g)
    ref = np.array([np.interp(g, L[i], F[i]) for i in range(20)])
    # rows are offset by 1e5 i, so abscissae carry ~ n 1e5 eps of rounding (documented in interp_rows)
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-8)


def test_lineset():
    ls = sp.LineSet(["HEI4026", "HEII4200"], [4026.22, 4199.90])
    assert len(ls) == 2 and ls.index("HEII4200") == 1 and ls.labels == ls.names
    with pytest.raises(ValueError):
        sp.LineSet(["A"], [1.0, 2.0])


def test_save_npz_and_meta(tmp_path):
    p = str(tmp_path / "x.npz")
    meta = sio.make_meta("synspec.test", params=dict(a=1), inputs=dict(me=__file__))
    sio.save_npz(p, dict(a=np.arange(5), b=np.ones((2, 3))), meta=meta)
    z = np.load(p)
    np.testing.assert_array_equal(z["a"], np.arange(5))
    m = sio.read_meta(p)
    assert m["kind"] == "synspec.test" and m["params"] == dict(a=1) and "me" in m["inputs"]
    assert not [f for f in os.listdir(tmp_path) if "tmp" in f]
    np.testing.assert_array_equal(sio.npz_member_memmap(p, "b"), np.ones((2, 3)))


def test_npz_member_memmap_compressed(tmp_path):
    p = str(tmp_path / "c.npz")
    np.savez_compressed(p, a=np.arange(1000))
    with pytest.raises(ValueError):
        sio.npz_member_memmap(p, "a")


@pytest.mark.m424
def test_npz_member_memmap_profiles():
    p = m424_path("run", "profiles.npz")
    mm = sio.npz_member_memmap(p, "teff")
    np.testing.assert_array_equal(np.asarray(mm), np.load(p)["teff"])


def test_make_meta_non_path_inputs(tmp_path):
    """Non-path input values (lists of directories, labels) are recorded instead of raising."""
    m = sio.make_meta("synspec.test", inputs=dict(raw=[str(tmp_path), "/nonexistent"], me=__file__, label="x"))
    assert m["inputs"]["raw"][0]["size"] >= 0 and m["inputs"]["raw"][1] == "/nonexistent"
    assert m["inputs"]["label"] == "x" and "size" in m["inputs"]["me"]
