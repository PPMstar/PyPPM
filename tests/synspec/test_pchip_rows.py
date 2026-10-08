"""pchip_rows and build_imu_library(interp=) (PP 2026-10-08)."""
import numpy as np
import pytest

from ppmpy.synspec.library import IMU_INTERP, pchip_rows


def test_pchip_rows_smooth_function():
    x = np.linspace(-50.0, 50.0, 101)                     # 1 km/s
    V = np.vstack([1 - 0.4 * np.exp(-(x / 12.0) ** 2), 1 - 0.2 * np.exp(-((x - 5) / 20.0) ** 2)])
    g = np.linspace(-60.0, 60.0, 481)                     # 0.25 km/s, beyond both ends
    out = pchip_rows(x, V, g)
    exact = np.vstack([1 - 0.4 * np.exp(-(np.clip(g, -50, 50) / 12.0) ** 2),
                       1 - 0.2 * np.exp(-((np.clip(g, -50, 50) - 5) / 20.0) ** 2)])
    assert out.shape == (2, g.size)
    assert np.abs(out - exact).max() < 2e-4
    assert np.all(out[:, g < -50] == V[:, :1]) and np.all(out[:, g > 50] == V[:, -1:])   # constant beyond the ends
    # continuous derivative: on a fine grid the slope changes little between steps, unlike linear interpolation,
    # whose slope jumps at every knot
    gf = np.linspace(-40.0, 40.0, 1601)                    # 0.05 km/s
    sp = np.abs(np.diff(np.diff(pchip_rows(x, V[:1], gf)[0]))).max()
    sl = np.abs(np.diff(np.diff(np.interp(gf, x, V[0])))).max()
    assert sp < 0.2 * sl


def test_pchip_rows_duplicates_averaged():
    x = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
    V = np.array([[0.0, 1.0, 3.0, 2.0, 3.0]])
    out = pchip_rows(x, V, np.array([1.0]))
    assert out[0, 0] == pytest.approx(2.0)


def test_pchip_rows_rejects_decreasing():
    with pytest.raises(ValueError):
        pchip_rows(np.array([0.0, 2.0, 1.0]), np.zeros((1, 3)), np.array([0.5]))


def test_interp_options():
    assert IMU_INTERP == ("linear", "pchip")
