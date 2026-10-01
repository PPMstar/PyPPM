"""
Spectral grids: line sets, the logarithmic velocity grid, row interpolation and Doppler steps.

All profiles are handled on a uniform velocity grid y = c ln(lambda / lambda_ref) with step dv
(km/s). A Doppler shift by v (km/s, > 0 towards the observer) moves a profile by
-c ln(1 - v/c) on this grid exactly; it is applied as a whole number of steps.

PP 2026-10-01: ported from the project's fw_disc.py (Y, LREF, interp_rows, lam_of_y,
shift_steps); the defaults reproduce the M424 grid (dv 1 km/s, |y| <= 2700 km/s,
shifts up to 400 km/s).
"""
import numpy as np

from .conventions import C_KMS


class LineSet:
    """
    The spectral lines of a run: FORMAL_INPUT names, rest air wavelengths used as
    velocity zero points [Angstrom], and plot labels.

    Parameters
    ----------
    names: sequence of str
        Line names as in FASTWIND's FORMAL_INPUT (e.g. 'HEI4026').
    lref: sequence of float
        Velocity zero point of each line [Angstrom, air].
    labels: sequence of str, optional
        Plot labels; default the names.
    """

    def __init__(self, names, lref, labels=None):
        self.names = [str(n) for n in names]
        self.lref = np.asarray(lref, dtype=np.float64)
        if self.lref.shape != (len(self.names),):
            raise ValueError("need one reference wavelength per line")
        self.labels = list(labels) if labels is not None else list(self.names)
        if len(self.labels) != len(self.names):
            raise ValueError("need one label per line")

    def __len__(self):
        return len(self.names)

    def __repr__(self):
        return "LineSet({})".format(", ".join("{} {:.2f} A".format(n, l) for n, l in zip(self.names, self.lref)))

    def index(self, name):
        """Position of a line by name."""
        return self.names.index(name)

    def to_dict(self):
        return dict(names=list(self.names), lref=self.lref.tolist(), labels=list(self.labels))


class VelocityGrid:
    """
    Uniform velocity grid y = -vmax .. vmax in steps of dv [km/s], and Doppler shifts
    of up to +-vshift [km/s] applied as whole grid steps.

    Parameters
    ----------
    dv: float
        Grid step [km/s] (M424: 1).
    vmax: float
        Half-width of the grid [km/s] (M424: 2700).
    vshift: float
        Largest |line-of-sight velocity| handled [km/s] (M424: 400). Larger shifts
        are clipped and counted.

    Attributes
    ----------
    y: np.ndarray
        The grid, ``np.arange(-vmax, vmax + dv / 2, dv)``.
    nshift: int
        vshift in grid steps.
    """

    def __init__(self, dv=1.0, vmax=2700.0, vshift=400.0):
        self.dv = float(dv)
        self.vmax = float(vmax)
        self.vshift = float(vshift)
        self.y = np.arange(-self.vmax, self.vmax + self.dv / 2, self.dv)
        self.ny = self.y.size
        self.nshift = int(round(self.vshift / self.dv))
        self.icentre = int(np.argmin(np.abs(self.y)))

    def __repr__(self):
        return "VelocityGrid(dv={:g}, vmax={:g}, vshift={:g})".format(self.dv, self.vmax, self.vshift)

    def lam(self, lref):
        """Wavelengths of the grid for a line with zero point lref [Angstrom]."""
        return lam_of_y(self.y, lref)

    def shift_steps(self, v):
        """
        Doppler shift of v [km/s, > 0 towards the observer] in grid steps.

        Returns
        -------
        s: np.ndarray of int64
            ``rint(-c ln(1 - v/c) / dv)``, clipped to +-nshift.
        clipped: np.ndarray of bool
            True where the shift exceeded nshift.
        """
        s = np.rint(-C_KMS * np.log(1.0 - np.asarray(v) / C_KMS) / self.dv).astype(np.int64)
        return np.clip(s, -self.nshift, self.nshift), np.abs(s) > self.nshift

    def check_zero_padding(self, depth, tol=0.0):
        """
        The FFT disc integration assumes that every rest profile has zero depth
        (1 - F = 0) beyond |y| > vmax - vshift. Returns the largest |depth| there and
        whether it is <= tol.

        Parameters
        ----------
        depth: np.ndarray
            Absorption depth 1 - F on this grid, last axis = y.
        """
        out = np.abs(self.y) > self.vmax - self.vshift
        worst = float(np.max(np.abs(np.asarray(depth)[..., out]))) if out.any() else 0.0
        return dict(max_depth=worst, ok=worst <= tol)

    def to_dict(self):
        return dict(dv=self.dv, vmax=self.vmax, vshift=self.vshift)


def lam_of_y(y, lref):
    """Wavelength [Angstrom] of velocity y [km/s] for zero point lref: lref exp(y/c)."""
    return lref * np.exp(np.asarray(y) / C_KMS)


def y_of_lam(lam, lref):
    """Velocity coordinate [km/s] of wavelength lam: c ln(lam / lref)."""
    return C_KMS * np.log(np.asarray(lam, dtype=np.float64) / lref)


def doppler_lambda(lam, v):
    """Observed wavelength of rest wavelength lam for velocity v [km/s, > 0 towards the observer]."""
    return np.asarray(lam) * (1.0 - np.asarray(v) / C_KMS)


def interp_rows(L, F, g):
    """
    Linear interpolation of every row (L[i], F[i]) onto the common abscissa g;
    constant beyond the ends of each row.

    Rows of L must be increasing and span less than 1e5 (the rows are separated
    by offsets of 1e5 so that one searchsorted call handles all of them; this is
    the exact algorithm of the M424 production, so results are bit-identical).
    The offsets cost precision: abscissae are rounded to about n * 1e5 * 2.2e-16
    (4e-10 for 20 rows, 4e-7 km/s for 20 000 rows), i.e. an error of about
    |dF/dx| times that, ~1e-9 for line profiles on a 1 km/s grid.

    Parameters
    ----------
    L, F: np.ndarray
        (n, k) abscissae and values.
    g: np.ndarray
        (m,) common abscissa.

    Returns
    -------
    np.ndarray
        (n, m) float64.
    """
    n, k = L.shape
    off = (np.arange(n) * 1.0e5)[:, None]
    Lf = (L.astype(np.float64) + off).ravel()
    q = (g[None, :] + off).ravel()
    i = np.searchsorted(Lf, q)
    base = np.repeat(np.arange(n) * k, g.size)
    i = np.clip(i, base + 1, base + k - 1)
    x0, x1 = Lf[i - 1], Lf[i]
    y0, y1 = F.ravel()[i - 1].astype(np.float64), F.ravel()[i].astype(np.float64)
    w = np.clip((q - x0) / np.where(x1 > x0, x1 - x0, 1.0), 0.0, 1.0)
    return (y0 + w * (y1 - y0)).reshape(n, g.size)
