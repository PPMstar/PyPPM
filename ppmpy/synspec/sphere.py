"""
Points on a sphere: the equal-area (Fibonacci) grid of the per-point FASTWIND models, the local
basis vectors, projections onto lines of sight, line-of-sight velocities and disc weights.

Conventions
-----------
* Spherical coordinates follow the physics convention of the moms grid: theta from +z,
  phi from +x in the x-y plane, phi in [0, 2 pi).
* Local basis (:func:`sphere_basis`): r_hat, theta_hat (towards +theta, i.e. southwards),
  phi_hat (towards +phi). A velocity u has components u_r, u_theta, u_phi along them
  (ppmpy ``get_spherical_components``; M424 samples ``ur``, ``uth``, ``uph``).
* Lines of sight n point from the star towards the observer (:mod:`ppmpy.synspec.conventions`).
  mu = r_hat . n > 0 on the visible hemisphere; v = u . n > 0 towards the observer (blueshift).
* Equal-area grid: every point represents dA = 4 pi / N of the unit sphere, so a disc integral
  is sum over visible points of mu I dA (Lambert's cosine law, I(mu) = const gives pi I).

Reproducibility
---------------
The projections mu = r_hat . n etc. are three-term dot products. The M424 production computed
them with BLAS through numpy's ``@``, whose result depends in the last bit (1 ulp, ~1e-16) on the
kernel BLAS picks:

* fw_disc_dumps.py (all-dump time series): one matrix product ``rhat @ LOS.T`` for the 8 lines of
  sight (dgemm) -> ``method='matmul'`` (default);
* fw_disc_los.py (exact per-point sums, disc_los8.npz, library_dT10.npz): one ``rhat @ n`` per
  line of sight (dgemv, fw_disc.mu_vlos) -> ``method='matvec'``;
* ``method='explicit'``: the plain sum r_x n_x + r_y n_y + r_z n_z in IEEE arithmetic, independent
  of BLAS and chunking (opt-in; not the production bits).

Between the methods 10-40 % of the M424 values differ in the last bit. numpy evaluates a product with
a single row or column by dgemv or a dot product instead of dgemm, so one point (a block of one
point) and, for 'matmul', a single line of sight are padded to two; with that, the result for a
point and line of sight does not depend on which other points (chunks) and lines of sight are
computed with it (checked bit for bit with numpy's OpenBLAS 0.3.23 on the Trillium AMD EPYC 9655
nodes, where the M424 products were made). Another BLAS may differ by 1 ulp in mu, tn, pn. That
moves v = u . n by at most |u| x 1.1e-16 ~ 1e-14 km/s, so a Doppler shift crosses a 1 km/s
rounding boundary with a probability of ~1e-14 per point: none does for dump 3200, where the
rounded shifts and the visibility masks (mu > 0), hence the F profiles, are identical for the three
methods (the closest Doppler shift lies 3e-8 km/s from a boundary); only vmean_w, sigma_w change
in the last bits.

All methods share the grid and the basis, whose bits depend on numpy's transcendental functions:
numpy 1.26 evaluates ``arccos`` (theta of :func:`fibonacci_sphere`) with AVX512 SIMD code that
differs from libm ``acos`` in the last bit for ~30 % of the grid values, and sin, cos come from libm,
which also differs between platforms. So theta, phi and the basis -- and with them every
projection -- are bit for bit portable only between numpy builds and CPUs with the same functions:
:func:`fibonacci_sphere` reproduces the M424 points.npz bit for bit with numpy 1.26 on the AVX512
(AVX512_SKX dispatch) Trillium nodes. Elsewhere, for bitwise work, read theta and phi from
points.npz or profiles.npz (:func:`ppmpy.synspec.io.npz_member_memmap`) instead of recomputing them
(the basis then still depends on libm sin, cos).

PP 2026-10-01: ported from fw_sphere_extract.py (grid, x/y/z), fw_disc.py (unit_vectors, mu_vlos,
weights) and fw_disc_dumps.py (MU, TN, PN, v); the moms sampling functions follow in M5 (section
at the end of this module).
"""
import numpy as np

from .conventions import los_array

__all__ = ["fibonacci_sphere", "sphere_xyz", "sphere_basis", "project_los", "los_velocity",
           "disc_weights", "quadrature_check", "PROJECT_METHODS"]

PROJECT_METHODS = ("matmul", "matvec", "explicit")


# ----------------------------------------------------------------------------------------------
# the grid
# ----------------------------------------------------------------------------------------------
def fibonacci_sphere(npoints):
    """
    Equal-area (golden-spiral) grid of ``npoints`` points on the sphere, as used by ppmpy's
    ``MomsDataSet._constantArea_spherical_grid`` for spherical interpolations.

    Parameters
    ----------
    npoints: int
        Number of points N (M424 per-point run: 1 236 544 = 2 x 618 272).

    Returns
    -------
    theta, phi: np.ndarray
        (N,) float64, physics convention [rad]; theta_i = arccos(1 - 2 (i + 1/2) / N),
        phi_i = pi (1 + sqrt 5)(i + 1/2) mod 2 pi.

    Validation
    ----------
    Bit for bit equal to ``MomsDataSet._constantArea_spherical_grid`` (same expressions) and,
    with numpy 1.26 on the AVX512 Trillium nodes, to theta, phi of the M424 points.npz
    (tests/synspec/test_sphere.py). The arccos bits depend on numpy's SIMD dispatch (module
    notes): off Trillium, load theta, phi from points.npz or profiles.npz
    (:func:`ppmpy.synspec.io.npz_member_memmap`) where bit equality with the M424 products
    matters.

    Notes
    -----
    :func:`ppmpy.synspec.conventions.fibonacci_directions` is the same spiral without the mod 2 pi
    (observer directions); the unit vectors agree to rounding.
    """
    npoints = _positive_int(npoints, "npoints")
    return _fibonacci_range(0, npoints, npoints)


def _positive_int(x, name):
    """int(x) for an integral x >= 1 (also numpy integers and integral floats), else ValueError."""
    try:
        ok = int(x) == x and x >= 1
    except (TypeError, ValueError, OverflowError):
        ok = False
    if not ok:
        raise ValueError("{} must be a positive integer, got {!r}".format(name, x))
    return int(x)


def _fibonacci_range(i0, i1, npoints):
    """Points i0 <= i < i1 of fibonacci_sphere(npoints) (elementwise, so the same values as a slice of it)."""
    # PP 2026-10-01: ported from fw_sphere_extract.py:44-47 (= ppm.py MomsDataSet._constantArea_spherical_grid);
    # np.arange(0, n) + 0.5 is np.arange(n) + 0.5 of the legacy code
    ind = np.arange(i0, i1) + 0.5
    theta = np.arccos(1.0 - 2.0 * ind / npoints)
    g = np.pi * (1.0 + 5 ** 0.5) * ind
    phi = g - 2.0 * np.pi * np.floor(g / (2.0 * np.pi))
    return theta, phi


def sphere_xyz(theta, phi, r=1.0):
    """
    Cartesian coordinates of points (r, theta, phi) in the simulation frame.

    Parameters
    ----------
    theta, phi: array-like
        Physics convention [rad]; converted to float64 (no-op for the float64 grid).
    r: float or array-like
        Radius (M424: 4050 Mm).

    Returns
    -------
    x, y, z: np.ndarray
        r sin(theta) cos(phi), r sin(theta) sin(phi), r cos(theta), evaluated left to right.

    Validation
    ----------
    Bit for bit equal to x, y, z of the M424 points.npz (r = 4050).
    """
    # PP 2026-10-01: ported from fw_sphere_extract.py:52-53 (same operation order)
    theta, phi = _float64(theta), _float64(phi)
    return r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)


def _float64(x):
    """x as a float64 array (no copy for float64 arrays and memory maps)."""
    return np.asarray(x, dtype=np.float64)


def sphere_basis(theta, phi):
    """
    Local orthonormal basis of points on the sphere.

    Parameters
    ----------
    theta, phi: array-like
        Physics convention [rad], same shape S; converted to float64 (float32 coordinates would
        otherwise be evaluated in float32; no-op for the float64 grid).

    Returns
    -------
    rhat, that, phat: np.ndarray
        (S..., 3) float64 unit vectors r_hat = (sin t cos p, sin t sin p, cos t),
        theta_hat = (cos t cos p, cos t sin p, -sin t), phi_hat = (-sin p, cos p, 0).

    Validation
    ----------
    Bit for bit fw_disc.unit_vectors (float64 coordinates).
    """
    return _basis(_float64(theta), _float64(phi))


def _basis(theta, phi, rhat_only=False):
    """sphere_basis of float64 arrays; rhat_only: (rhat,) only (the same bits)."""
    # PP 2026-10-01: ported from fw_disc.py:104-109 (unit_vectors)
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    rhat = np.stack([st * cp, st * sp, ct], axis=-1)
    if rhat_only:
        return (rhat,)
    that = np.stack([ct * cp, ct * sp, -st], axis=-1)
    phat = np.stack([-sp, cp, np.zeros_like(phi)], axis=-1)
    return rhat, that, phat


# ----------------------------------------------------------------------------------------------
# lines of sight
# ----------------------------------------------------------------------------------------------
def _los_vectors(los):
    """(nlos, 3) C-contiguous float64 unit vectors. A string goes through conventions.los_array; an
    array is used as it is (not renormalised, which could change its last bits) after a check."""
    if isinstance(los, str):
        L = los_array(los)
    else:
        L = np.asarray(los, dtype=np.float64)
        if L.ndim == 1:
            L = L[None, :]
        if L.ndim != 2 or L.shape[1] != 3 or L.shape[0] == 0:
            raise ValueError("lines of sight must be (nlos, 3) or (3,), got shape {}".format(np.shape(los)))
        norm = np.sqrt(np.sum(L * L, axis=1))
        if not np.all(np.abs(norm - 1.0) <= 1e-12):
            raise ValueError("lines of sight must be unit vectors (|n| - 1 up to {:.1e}); normalise them with "
                             "ppmpy.synspec.conventions.los_array".format(float(np.max(np.abs(norm - 1.0)))))
    return np.ascontiguousarray(L)


def _gemm(A, B):
    """A @ B through dgemm also when B has one column (numpy would use dgemv there and the last bit could
    differ): B is padded with a copy of its column, then the result is cut out. (One-row A: _project_block.)"""
    if B.shape[1] >= 2:
        return A @ B
    return (A @ np.concatenate([B, B], axis=1))[:, :1]


def _project_block(theta, phi, L, method, rhat_only=False):
    """(mu, tn, pn), each (nlos, n), for one block of float64 points; rhat_only: (mu,) only (the same bits:
    every basis vector is projected by its own product)."""
    if theta.size == 1:
        # PP 2026-10-01: a one-point block takes numpy's dot path instead of dgemm/dgemv (last bit can differ):
        # compute it as two copies of the point
        return tuple(np.ascontiguousarray(x[:, :1])
                     for x in _project_block(np.repeat(theta, 2), np.repeat(phi, 2), L, method, rhat_only))
    basis = _basis(theta, phi, rhat_only)
    if method == "matmul":
        # PP 2026-10-01: ported from fw_disc_dumps.py:71-74 (one dgemm for all lines of sight, then transposed copies)
        return tuple(np.ascontiguousarray(_gemm(b, L.T).T) for b in basis)
    if method == "matvec":
        # PP 2026-10-01: ported from fw_disc.py:134-137 (mu_vlos: one dgemv per line of sight), as fw_disc_los.py:58-60
        out = tuple(np.empty((L.shape[0], theta.size)) for _ in basis)
        for k in range(L.shape[0]):
            nvec = L[k]
            for o, b in zip(out, basis):
                o[k] = b @ nvec
        return out
    # PP 2026-10-01: new (opt-in): r_x n_x + r_y n_y + r_z n_z in IEEE double, left to right
    return tuple(b[:, 0] * L[:, 0:1] + b[:, 1] * L[:, 1:2] + b[:, 2] * L[:, 2:3] for b in basis)


def project_los(theta, phi, los, method="matmul", chunk=None):
    """
    Projections of the local basis of every point onto the lines of sight.

    Parameters
    ----------
    theta, phi: array-like
        (N,) point coordinates, physics convention [rad] (:func:`fibonacci_sphere`); converted to
        float64 (no copy for float64 arrays or memory maps; float32 coordinates would otherwise be
        evaluated in float32, |dmu| ~ 1e-7).
    los: str or array-like
        Lines of sight: a :func:`ppmpy.synspec.conventions.los_array` name (e.g. 'thompson2024')
        or unit vectors (nlos, 3) or (3,) from the star towards the observer. Arrays are not
        renormalised (that could change their last bit); non-unit vectors raise ValueError.
    method: {'matmul', 'matvec', 'explicit'}
        How the dot products are evaluated (see the module notes): 'matmul' = fw_disc_dumps.py
        (default), 'matvec' = fw_disc.mu_vlos / fw_disc_los.py, 'explicit' = portable plain sums.
    chunk: int, optional
        Process the points in blocks of this many (a positive integer; bounds the temporaries to
        ~ chunk (9 + 6 nlos) x 8 bytes; the outputs, 3 nlos N x 8 bytes, are allocated once).
        None = all at once, as the legacy scripts did. Blocks do not change any bit (dot products
        are per point; numpy's sin/cos are elementwise; a one-point block and, for 'matmul', a
        single line of sight are padded, see the module notes; checked bit for bit with numpy's
        OpenBLAS, tests/synspec/test_sphere.py).

    Returns
    -------
    mu, tn, pn: np.ndarray
        (nlos, N) C-contiguous float64: r_hat . n, theta_hat . n and phi_hat . n.

    Conventions
    -----------
    mu > 0 marks the visible hemisphere of line of sight n; the line-of-sight velocity is
    v = u_r mu + u_theta tn + u_phi pn (:func:`los_velocity`).

    Validation
    ----------
    'matmul' is bit for bit the MU, TN, PN of fw_disc_dumps.py and reproduces the stored
    dump-3200 flux product; 'matvec' reproduces vmean_w, sigma_w of disc_los8.npz bit for bit
    (tests/synspec/test_sphere.py, marker m424).
    """
    if method not in PROJECT_METHODS:
        raise ValueError("method must be one of {}, got {!r}".format(PROJECT_METHODS, method))
    theta, phi = _float64(theta), _float64(phi)
    if theta.ndim != 1 or theta.shape != phi.shape:
        raise ValueError("theta and phi must be 1-D arrays of the same length")
    if chunk is not None:
        chunk = _positive_int(chunk, "chunk")
    L = _los_vectors(los)
    n = theta.size
    if chunk is None or chunk >= n:
        return _project_block(theta, phi, L, method)
    out = tuple(np.empty((L.shape[0], n)) for _ in range(3))
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        for o, b in zip(out, _project_block(theta[i0:i1], phi[i0:i1], L, method)):
            o[:, i0:i1] = b
    return out


def los_velocity(ur, uth, uph, mu, tn, pn):
    """
    Line-of-sight velocity v = u . n of every point.

    Parameters
    ----------
    ur, uth, uph: array-like
        Velocity components along r_hat, theta_hat, phi_hat (N,) [any unit; M424 km/s]. Converted
        to float64 first (exact for float32 samples, as the legacy ``.astype(np.float64)``).
    mu, tn, pn: array-like
        Projections from :func:`project_los`, (nlos, N) or (N,).

    Returns
    -------
    np.ndarray
        v, float64, broadcast shape ((nlos, N) or (N,)); > 0 towards the observer (blueshift,
        lambda_obs = lambda (1 - v/c)).

    Validation
    ----------
    Evaluated as (ur mu + uth tn) + uph pn with every product and sum rounded separately, i.e. bit
    for bit ``ur * MU[k] + uth * TN[k] + uph * PN[k]`` of fw_disc_dumps.py and the v of
    fw_disc.mu_vlos. Two temporaries of the output size.
    """
    # PP 2026-10-01: ported from fw_disc_dumps.py:94 and fw_disc.py:137 (same operation order; every product and
    # sum rounded once, as in the expression ur * mu + uth * tn + uph * pn)
    ur, uth, uph = (np.asarray(u, dtype=np.float64) for u in (ur, uth, uph))
    mu, tn, pn = np.asarray(mu), np.asarray(tn), np.asarray(pn)
    shape = np.broadcast(ur, uth, uph, mu, tn, pn).shape
    v, t = np.empty(shape), np.empty(shape)
    np.multiply(ur, mu, out=v)
    np.multiply(uth, tn, out=t)
    v += t
    np.multiply(uph, pn, out=t)
    v += t
    return v


def disc_weights(mu, fc=1.0, uld=0.0):
    """
    Disc-integration weight of every point: mu [1 - uld (1 - mu)] F_c on the visible hemisphere
    (mu > 0), 0 elsewhere.

    Parameters
    ----------
    mu: array-like
        r_hat . n (:func:`project_los`).
    fc: float or array-like
        Continuum flux of each point (broadcast against mu), or 1.
    uld: float
        Linear limb-darkening coefficient (0 = I(mu) constant, the M424 flux method).

    Returns
    -------
    np.ndarray
        Weights; multiply by dA = 4 pi R^2 / N for a disc integral.

    Validation
    ----------
    Bit for bit fw_disc.weights.
    """
    # PP 2026-10-01: ported from fw_disc.py:141-143 (weights)
    return np.where(mu > 0, mu * (1.0 - uld * (1.0 - mu)), 0.0) * fc


def quadrature_check(npoints, los, method="matmul", chunk=1 << 18):
    """
    Accuracy of disc integrals on the equal-area grid: relative errors of the visible-hemisphere
    sums sum(mu dA) and sum(mu^2 dA), dA = 4 pi / N, against their exact values pi and 2 pi / 3.

    Parameters
    ----------
    npoints: int
        Grid size N (:func:`fibonacci_sphere`).
    los: str or array-like
        Lines of sight (:func:`project_los`).
    method: str
        Projection method (:func:`project_los`).
    chunk: int or None
        Grid points per block: the grid, mu and the sums are built block by block, so the memory is
        ~ chunk (16 + 4 nlos) x 8 bytes (default 262 144 points: ~100 MB for 8 lines of sight;
        measured 106 MB for N = 1e7, 0.7 s) for any N. None = the whole grid at once (N = 1e7:
        2.1 GB, 5.8 s).

    Returns
    -------
    err_mu, err_mu2: np.ndarray
        (nlos,) signed relative errors sum/exact - 1.

    Notes
    -----
    Thompson et al. (2024) lines of sight: max |err_mu| = 3.1e-5 and max |err_mu2| = 7.4e-6 for
    N = 5000; 3.7e-8 for N = 618 272 and 1.5e-8 for N = 1 236 544 (err_mu), i.e. far below the
    1e-4 line-profile variability of M424. The block sums change the result only by rounding
    (~1e-16 relative).
    """
    # PP 2026-10-01: new (the check behind the equal-area weights of fw_disc_los.py / fw_disc_dumps.py); only mu is
    # projected (r_hat; the same values as project_los(...)[0]), block by block
    npoints = _positive_int(npoints, "npoints")
    chunk = npoints if chunk is None else _positive_int(chunk, "chunk")
    L = _los_vectors(los)
    if method not in PROJECT_METHODS:
        raise ValueError("method must be one of {}, got {!r}".format(PROJECT_METHODS, method))
    s1, s2 = np.zeros(L.shape[0]), np.zeros(L.shape[0])
    for i0 in range(0, npoints, chunk):
        theta, phi = _fibonacci_range(i0, min(i0 + chunk, npoints), npoints)
        mu = _project_block(theta, phi, L, method, rhat_only=True)[0]
        w = disc_weights(mu)
        s1 += w.sum(axis=1)
        w *= mu
        s2 += w.sum(axis=1)
    dA = 4.0 * np.pi / npoints
    err_mu = s1 * dA / np.pi - 1.0
    err_mu2 = s2 * dA / (2.0 * np.pi / 3.0) - 1.0
    return err_mu, err_mu2


# ----------------------------------------------------------------------------------------------
# moms sampling (milestone M5; not ported yet)
# ----------------------------------------------------------------------------------------------
# To come here (ppmpy.ppm imported inside the functions only):
#   * sampling one moms dump on fibonacci_sphere(N) at radius r with MomsDataSet.get_spherical_interpolation
#     (trilinear on the cell-centre grid): relT = (T9 - <T9>) / <T9>, T_eff' = teff0 (1 + relT), and
#     u_r, u_theta, u_phi [km/s] from get_spherical_components (slots 1-3), as sphere_sample.py
#     (samples_r4050_N1236544/dNNNN.npz, float32) and fw_sphere_extract.py (points.npz, float64);
#   * the check of theta, phi and x, y, z (sphere_xyz) against MomsDataSet._constantArea_spherical_grid
#     and its interpolation grid (igrid columns z, y, x);
#   * splitting dumps over workers (sphere_sample.py assigns dump d to worker d % n; see
#     ppmpy.synspec.parallel.split_items for the index rule of fw_disc_dumps.py).
# Bitwise reproduction of the M424 products off Trillium (M3 validation, M5 sampling): take theta, phi from
# points.npz / profiles.npz (io.npz_member_memmap) instead of fibonacci_sphere (arccos bits depend on numpy's
# SIMD dispatch, see the module notes).
