"""
Constants, sign conventions and line-of-sight sets used throughout ppmpy.synspec.

Conventions
-----------
* Velocity grid: y = c ln(lambda / lambda_ref) in km/s, on air wavelengths in Angstrom.
* Doppler shift: v = u . n is positive towards the observer (blueshift);
  lambda_obs = lambda (1 - v/c). A centroid on the y grid is therefore about -<v>.
* Lines of sight are unit vectors pointing from the star to the observer, in the
  simulation frame (x, y, z of the moms grid). Spherical coordinates follow the
  physics convention: theta from +z, phi from +x in the x-y plane.
* Time in s, frequency in microHz; 1 d^-1 = 11.574 microHz. A relative fluctuation
  squared times 1e12 is ppm^2.

PP 2026-10-01: ported from the project's fw_disc.py (C_KMS, los8, directions).
"""
import numpy as np

C_KMS = 299792.458                     # speed of light [km/s]
SECONDS_PER_DAY = 86400.0
CPD_PER_MUHZ = SECONDS_PER_DAY * 1e-6  # 1 microHz = 0.0864 d^-1
MUHZ_PER_CPD = 1.0 / CPD_PER_MUHZ      # 1 d^-1 = 11.574 microHz
PPM2_PER_REL2 = 1e12                   # (relative fluctuation)^2 -> ppm^2


def muhz_to_cpd(f):
    """Frequency in microHz -> cycles per day."""
    return np.asarray(f) * CPD_PER_MUHZ


def cpd_to_muhz(f):
    """Frequency in cycles per day -> microHz."""
    return np.asarray(f) * MUHZ_PER_CPD


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def los_thompson2024():
    """
    The 8 lines of sight of Thompson et al. (2024, PPMstar M107; arXiv:2303.06125,
    Sect. 2.2), as used for all M424 line-profile products.

    los1 = (1,1,1), los2 = los1 x (0,0,1), los3 = los1 x los2,
    los4 = los1 + los2 + los3, each normalised before it is used for the next;
    los5..8 = -los1..-los4.

    Returns
    -------
    np.ndarray
        (8, 3) unit vectors from the star towards the observer.

    Notes
    -----
    Not the PPMstar Fortran lum1..8 set (:func:`los_ppmstar_fortran`): only LOS 1
    and 5 agree; 2, 3, 6, 7 are reversed and 4, 8 differ. ``ppm.make_los_vectors``
    agrees except for LOS 4 (13 degrees apart; it does not normalise before summing).
    """
    l1 = _unit([1.0, 1.0, 1.0])
    l2 = _unit(np.cross(l1, [0.0, 0.0, 1.0]))
    l3 = _unit(np.cross(l1, l2))
    l4 = _unit(l1 + l2 + l3)
    first = np.array([l1, l2, l3, l4])
    return np.concatenate([first, -first])


def los_ppmstar_fortran(xxlos=0.3, yylos=0.3, zzlos=0.3):
    """
    The 8 lines of sight of the PPMstar Fortran (rprof lum1..lum8), as
    ``ppm.make_los_vectors_fortran`` (PPM2F-12-12-21-O2.F lines 9594-9631).

    Parameters
    ----------
    xxlos, yylos, zzlos: float
        Primary line of sight from the flags file (M424: 0.3, 0.3, 0.3).

    Returns
    -------
    np.ndarray
        (8, 3) unit vectors.
    """
    rrlos = np.sqrt(xxlos * xxlos + yylos * yylos + zzlos * zzlos)
    los = np.zeros((8, 3))
    if rrlos == 0.0:
        los[0] = (1.0, 0.0, 0.0)
    else:
        los[0] = (xxlos / rrlos, yylos / rrlos, zzlos / rrlos)
    if (xxlos == 0.0) and (yylos == 0.0) and (zzlos != 0.0):
        los[1] = (zzlos / abs(zzlos), 0.0, 0.0)
    else:
        rrlos = np.sqrt(los[0, 0] ** 2 + los[0, 1] ** 2)
        los[1] = (-los[0, 1] / rrlos, los[0, 0] / rrlos, 0.0)
    los[2, 0] = los[0, 1] * los[1, 2] - los[0, 2] * los[1, 1]
    los[2, 1] = -los[0, 0] * los[1, 2] + los[0, 2] * los[1, 0]
    los[2, 2] = los[0, 0] * los[1, 1] - los[0, 1] * los[1, 0]
    los[3] = los[0] + los[1] + los[2]
    los[3] /= np.linalg.norm(los[3])
    los[4:] = -los[:4]
    return los


def fibonacci_directions(n):
    """
    n observer directions spread evenly over the sphere (golden spiral, offset 0.5).

    Returns
    -------
    np.ndarray
        (n, 3) unit vectors.
    """
    k = np.arange(n) + 0.5
    th = np.arccos(1.0 - 2.0 * k / n)
    ph = np.pi * (1.0 + 5 ** 0.5) * k
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1)


def los_array(spec):
    """
    Resolve a line-of-sight specification to unit vectors.

    Parameters
    ----------
    spec: str or array-like
        'thompson2024', 'ppmstar_fortran', 'fibonacci:N', or an (n, 3) array of
        vectors (normalised here). There is deliberately no default set.

    Returns
    -------
    np.ndarray
        (n, 3) unit vectors.
    """
    if isinstance(spec, str):
        if spec == "thompson2024":
            return los_thompson2024()
        if spec == "ppmstar_fortran":
            return los_ppmstar_fortran()
        if spec.startswith("fibonacci:"):
            return fibonacci_directions(int(spec.split(":", 1)[1]))
        raise ValueError("unknown line-of-sight set {!r} (use 'thompson2024', 'ppmstar_fortran', "
                         "'fibonacci:N' or an array)".format(spec))
    v = np.atleast_2d(np.asarray(spec, dtype=np.float64))
    if v.shape[-1] != 3:
        raise ValueError("line-of-sight vectors must have 3 components, got shape {}".format(v.shape))
    return v / np.linalg.norm(v, axis=1, keepdims=True)
