"""
Sampling of PPMstar 2.0 moms dumps (briquette-averaged cubes) on the equal-area sphere: the temperature
perturbation and T_eff' = teff0 (1 + relT) of every point, and the spherical velocity components u_r, u_theta,
u_phi (port of the project's fw_sphere_extract.py and sphere_sample.py).

* :class:`MomsGrid`: the cell-centre coordinates of the moms grid (``MomsDataSet._get_cgrid`` arithmetic).
* :class:`MomsSource`: a moms directory: file pattern, slot map, run id, dumps, block files, grid spacing, rprof
  times.
* :class:`MomsBlockReader`: one dump read from its block files (plain reads of plane ranges, or memory maps);
  :meth:`MomsBlockReader.slab` / :meth:`MomsBlockReader.gather` give any box / any cells of a slot exactly as
  ``MomsData.read_moms_cube`` assembles the whole cube.
* :func:`grid_jacobian`: the Cartesian -> spherical unit-vector matrix of ``MomsDataSet._get_jacobian``.
* :func:`sample_moms_points`: the temperature slot and the velocity components of one dump at any points.
* :func:`sample_moms_sphere`: one dump on the sphere, box by box from the block files (``backend='slab'``,
  default; 0.3 GB and ~2-6 s per M424 dump) or through a ppmpy ``MomsDataSet`` (``backend='momsdataset'``, the
  legacy code path and the oracle of the slab backend; 18 GB and ~50 s per M424 dump). Same bits.
* :func:`write_points_table`: points.npz, points.txt and meta.json of fw_sphere_extract.py.
* :func:`sample_moms_dumps`: the per-dump files dNNNN.npz of sphere_sample.py (restartable, rank split, worker
  processes).

Conventions
-----------
* Points: :func:`ppmpy.synspec.sphere.fibonacci_sphere` (theta from +z, phi from +x) on the sphere of radius
  ``radius``; x, y, z = :func:`ppmpy.synspec.sphere.sphere_xyz` (the values ppmpy interpolates at).
* The moms cube is indexed [z, y, x] (ppmpy); the interpolation is scipy's RegularGridInterpolator (trilinear) on
  the cell centres, exactly as ``MomsDataSet.get_spherical_interpolation`` with ``method='trilinear'``.
* The grid: :class:`MomsGrid` of the run spacing dx ('deex' of the rprof header, or given). ppmpy builds the same
  grid from an RprofSet; without one it derives the spacing from slot 0 (mean spacing of xc[0, 0, :]), which is
  NOT valid for M424 (slot 0 is a radius-like field there, zero along that row, so the grid collapses). The
  momsdataset backend therefore needs ``rprof`` (or a MomsDataSet passed as ``moms``) and checks that ppmpy's
  grid is the source's grid.
* relT = (T - <T>) / <T> with <T> the plain mean over the points (equal-area grid), teff = teff0 (1 + relT).
* Velocities: ``get_spherical_components`` of the slots ``velocity`` on the cube (float32 Jacobian, see
  :func:`grid_jacobian`), each component then interpolated, times ``velocity_scale`` (M424: Mm/s -> km/s, 1e3).
  They are the moms values (M424: not density-corrected).

Bitwise equality of the slab backend
------------------------------------
Every step of the legacy path is elementwise per cell or per point, so it can be done on a box of the cube:
the Jacobian and the components (same ufuncs, dtypes and operation order as ppmpy on contiguous arrays) and the
trilinear interpolation (RegularGridInterpolator finds the interval i with g[i] <= x < g[i+1], clamped to
n - 2, and combines the 8 corner values with weights computed from g[i], g[i+1] only). A point whose interval
indices are (i, j, k) is evaluated in a box that contains the cells i..i+1, j..j+1, k..k+1, so its local
intervals are the global ones minus the box offset and the arithmetic is the same. The interpolation reads only
the 8 corner cells of each point, so by default (``slab_cells='corners'``) the fields and the Jacobian are
computed at those cells only (~2 % of the M424 cube), the rest of the box array is left unset; ``'box'`` computes
them on every cell of the box. The T field and the three components are stacked into one (nz, ny, nx, 4) array
(RegularGridInterpolator's trailing dimension), which multiplies and adds the same numbers per member. The cube
dtype is that of ``read_moms_cube``: float64 for a dump of several blocks (it assembles into ``np.zeros``),
float32 for a single block (a view of the file), so the component products round as in ppmpy in both cases.

Validated (tests/synspec/test_moms.py): slab == momsdataset == stored M424 products bit for bit (points.npz of
dump 3200 and samples dNNNN.npz of the subset dumps), slab sizes 1..n identical, toy dumps in the block layout
with 1, 2 and 3 blocks per dimension, points on grid nodes, on slab boundaries +- 1 ulp and on the grid ends.

Memory and time (M424 dump 3200, 1 236 544 points, Trillium login node, 2026-10-02, one process): momsdataset
backend 52 s (28 s with the files in the page cache) and 18.1 GB peak RSS; slab backend 6.2 s (15 s on a cold
Lustre read, 5 s warm) and 0.33-0.44 GB (2.0 s user, the rest system time: reading 4 of the 10 slots). The 16
verification-subset dumps with 4 slab workers: 9.2 s in total, 0.35 GB per process. With memory maps instead of
reads the slab backend took 18.5 s (17 s system time: page faults on Lustre). Memory of the slab backend grows
with ``slab`` (see :func:`sample_moms_points`): slab=1000 (one box) or ``slab_cells='box'`` peak at 3.4 GB,
'box' also takes ~16 s.

PP 2026-10-02: ported from fw_sphere_extract.py (project analysis, commit 67e042f) and sphere_sample.py
(same commit), and from ppmpy ppm.py MomsData.read_moms_cube, MomsDataSet._find_dumps, _get_cgrid,
_get_jacobian, get_spherical_components, get_interpolation and get_spherical_interpolation (the slab backend).
"""
import functools
import glob
import itertools
import json
import os
import re
import string
import time

import numpy as np
import scipy.interpolate

from . import io as sio
from . import parallel as par
from . import sphere as sph

__all__ = ["MomsGrid", "MomsSource", "MomsBlockReader", "grid_jacobian", "sample_moms_points",
           "sample_moms_sphere", "write_points_table", "sample_moms_dumps", "sample_path", "read_rprof_header",
           "rprof_files", "BACKENDS", "DEFAULT_PATTERN", "SAMPLE_KEYS", "POINTS_NOTE"]

BACKENDS = ("momsdataset", "slab")
DEFAULT_PATTERN = "{dump:04d}/{run_id}-BQav{dump:04d}.{ext}"      # ppmpy MomsDataSet._get_dump
SAMPLE_KEYS = ("relT", "teff", "ur", "uth", "uph")                  # float32 members of dNNNN.npz (legacy order)
POINTS_NOTE = "ur from moms (not density-corrected); Doppler shift applied after FASTWIND"
PATTERN_FIELDS = ("dump", "run_id", "ext")
_MARK = "\x00"                                                      # placeholder of the block extension


# ----------------------------------------------------------------------------------------------
# rprof headers (no ppmpy.ppm needed)
# ----------------------------------------------------------------------------------------------
def _header_value(s):
    # PP 2026-10-02: ported from ppm.py Rprof footer parsing ('.' -> float, else int)
    return float(s) if "." in s else int(s)


def read_rprof_header(path, names=("t", "deex")):
    """
    Header values of one .rprof file, parsed as ppmpy's ``Rprof`` does.

    Parameters
    ----------
    path: str
        The .rprof file.
    names: sequence of str
        't' (problem time [s], from the 'DUMP' line) and/or footer parameters (e.g. 'deex', the grid
        spacing of the run [Mm]); the last occurrence of a footer parameter wins (ppmpy).

    Returns
    -------
    dict
        name -> value (float, or int for values without a '.'); KeyError for a name not found.
    """
    # PP 2026-10-02: ported from ppm.py Rprof._read_rprof (t from the 'DUMP' line, footer: 'DATE:' + 16 lines, then
    # lines 'index name value name value')
    with open(path) as f:
        lines = f.readlines()
    out = {}
    names = list(names)
    footer = None
    for i, line in enumerate(lines):
        if "t" in names and "t" not in out and line.startswith("DUMP"):
            out["t"] = float(line.split("=")[1].split(",")[0])
        s = line.split()
        if footer is None and s and s[0] == "DATE:":
            footer = i + 16
    if footer is not None:
        for line in lines[footer:]:
            s = line.split()
            if len(s) == 5:
                for j in (1, 3):
                    if s[j] in names:
                        out[s[j]] = _header_value(s[j + 1])
    missing = [n for n in names if n not in out]
    if missing:
        raise KeyError("{} not found in the header of {}".format(missing, path))
    return out


def rprof_files(prfs_dir, run_id=None, prefix=""):
    """
    The rprof files ``<run_id>-<prefix>NNNN.rprof`` of a directory.

    Parameters
    ----------
    prfs_dir: str
    run_id: str, optional
        Run id of the files; KeyError if the directory has none of it. Default: the only run id of the
        directory, or the first in sorted order if there are several.
    prefix: str
        Text between the '-' and the dump number ('' for the rprofs written by PPMstar, 'BQav' for ppmpy's
        ``RprofSet(..., bqav=True)`` files in frombqavs/).

    Returns
    -------
    dict
        dump -> path, in dump order.
    """
    pat = re.compile(r"^(.*)-" + re.escape(prefix) + r"(\d{4,})\.rprof$")
    found = {}
    for name in sorted(os.listdir(prfs_dir)):
        m = pat.match(name)
        if m:
            found.setdefault(m.group(1), {})[int(m.group(2))] = os.path.join(prfs_dir, name)
    if not found:
        raise FileNotFoundError("no <run_id>-{}NNNN.rprof files in {}".format(prefix, prfs_dir))
    if run_id is None:
        rid = sorted(found)[0]
    elif run_id in found:
        rid = run_id
    else:
        raise KeyError("no rprof files of run id {!r} in {} (run ids: {})".format(run_id, prfs_dir, sorted(found)))
    return dict(sorted(found[rid].items()))


def _is_rprofset(obj):
    return hasattr(obj, "get_dump_list") and hasattr(obj, "get_dump") and hasattr(obj, "get")


def _rprof_dx(rprof, run_id=None, prefix=""):
    """'deex' of the first rprof dump: ppmpy's expression for an RprofSet, the header of the first file of
    ``run_id`` (:func:`rprof_files`) for a directory."""
    if _is_rprofset(rprof):
        return rprof.get_dump(rprof.get_dump_list()[0]).get("deex")
    files = rprof_files(rprof, run_id, prefix)
    return read_rprof_header(next(iter(files.values())), ("deex",))["deex"]


# ----------------------------------------------------------------------------------------------
# the grid
# ----------------------------------------------------------------------------------------------
class MomsGrid:
    """
    Cell-centre coordinates of a moms grid of n^3 cells, each ``coarsen`` run cells of size dx wide.

    Parameters
    ----------
    dx: float
        Run grid spacing (rprof header 'deex' [Mm]; M424: 4.572318077087402). Used as given (ppmpy uses the
        Python float of the rprof header).
    n: int
        Moms cells per dimension (M424: 448).
    coarsen: int
        Run cells per moms cell (PPMstar briquettes: 4).

    Attributes
    ----------
    coord: np.ndarray
        (n,) float64 cell centres, identical in x, y and z; bit for bit ppmpy's ``MomsDataSet._unique_coord``
        when constructed with the dx of its RprofSet: c dx arange(n) - (c dx (n/2) - c (dx/2)).
    """

    def __init__(self, dx, n, coarsen=4):
        # PP 2026-10-02: ported from ppm.py MomsDataSet._get_cgrid (rprofset branch); 4. -> float(coarsen)
        self.dx = dx
        self.n = sph._positive_int(n, "n")
        self.coarsen = coarsen
        c = float(coarsen)
        right_xcbound = c * dx * (self.n / 2.) - c * (dx / 2.)
        left_xcbound = -right_xcbound
        grid_values = c * dx * np.arange(0, self.n) + left_xcbound
        # ppmpy: xc_array = np.ones((n, n, n)) * grid_values; 1.0 * g == g
        self.coord = np.asarray(grid_values, dtype=np.float64)

    @classmethod
    def from_rprof(cls, rprof, n, coarsen=4, run_id=None):
        """
        The grid of ``MomsDataSet(..., rprofset=rprof)``: dx = 'deex' of the first rprof dump.

        Parameters
        ----------
        rprof: str or RprofSet
            An rprof directory (its header is parsed here, :func:`read_rprof_header`) or a ppmpy RprofSet
            (``rprof.get_dump(rprof.get_dump_list()[0]).get('deex')``, as ppmpy).
        n, coarsen:
            See :class:`MomsGrid`.
        run_id: str, optional
            Directory: run id of the rprof files (:func:`rprof_files`).
        """
        return cls(_rprof_dx(rprof, run_id), n, coarsen)

    def box(self, i0, i1, j0, j1, k0, k1):
        """
        x, y, z, r of the cells [i0:i1, j0:j1, k0:k1] (index order z, y, x) as contiguous float64 arrays,
        r = sqrt(x^2 + y^2 + z^2) evaluated as ppmpy's ``_radius``.
        """
        # PP 2026-10-02: ported from ppm.py MomsDataSet._get_cgrid (xc/yc/zc views, _radius); contiguous like ppmpy's
        # raveled arrays, so np.power takes the same (SIMD) loop
        g = self.coord
        shape = (i1 - i0, j1 - j0, k1 - k0)
        x = np.ones(shape) * g[k0:k1]
        y = np.ones(shape) * g[j0:j1][:, None]
        z = np.ones(shape) * g[i0:i1][:, None, None]
        r = np.sqrt(np.power(x, 2.0) + np.power(y, 2.0) + np.power(z, 2.0))
        return x, y, z, r

    def intervals(self, p):
        """
        Interval index i of each coordinate, g[i] <= p < g[i+1], clamped to [0, n-2] (scipy's
        ``find_interval_ascending`` as used by RegularGridInterpolator). ValueError for p outside [g[0], g[-1]]
        (RegularGridInterpolator with bounds_error=True).
        """
        g = self.coord
        p = np.asarray(p, dtype=float)
        if not (np.all(g[0] <= p) and np.all(p <= g[-1])):
            raise ValueError("points outside the moms grid [{}, {}]".format(g[0], g[-1]))
        i = np.searchsorted(g, p, side="right") - 1
        return np.clip(i, 0, self.n - 2)


# ----------------------------------------------------------------------------------------------
# the data
# ----------------------------------------------------------------------------------------------
def _pattern_fields(pattern):
    """The replacement fields of a file pattern; ValueError unless they are dump, ext and optionally run_id."""
    try:
        fields = [f for _, f, _, _ in string.Formatter().parse(pattern) if f is not None]
    except ValueError as e:
        raise ValueError("bad file pattern {!r}: {}".format(pattern, e)) from None
    bad = sorted(set(fields) - set(PATTERN_FIELDS))
    if bad or "dump" not in fields or "ext" not in fields:
        raise ValueError("file pattern {!r} must use the fields {{dump}} and {{ext}} (and optionally {{run_id}}), "
                         "found {}".format(pattern, fields))
    return fields


def _pattern_regex(pattern, run_id=None):
    """
    Regular expression of the .aaa files of a pattern, matched against paths relative to the moms directory
    ('/'-separated): dump -> (?P<dump>\\d+) (later occurrences \\d+), ext -> 'aaa', run_id -> the given run id or
    (?P<run_id>[^/]+?) (later occurrences a back reference). A match is a candidate only; the caller confirms it
    by formatting the pattern with the dump and run id found.
    """
    out, seen = [], set()
    for literal, field, _, _ in string.Formatter().parse(pattern):
        out.append(re.escape(literal))
        if field is None:
            continue
        if field == "dump":
            out.append(r"\d+" if "dump" in seen else r"(?P<dump>\d+)")
        elif field == "ext":
            out.append("aaa")
        elif run_id is not None:
            out.append(re.escape(run_id))
        else:
            out.append("(?P=run_id)" if "run_id" in seen else r"(?P<run_id>[^/]+?)")
        seen.add(field)
    return re.compile("^" + "".join(out) + "$")


class MomsSource:
    """
    A directory of decompressed moms dumps (``moms/myavsbq``) and how to read it.

    Parameters
    ----------
    moms_dir: str
        Root directory; block ``ext`` ('aaa', 'aab', ...) of dump d is ``moms_dir/pattern``.
    slots: dict or sequence
        Slot map: name -> 0-based slot index, or a sequence of names whose positions are the slots
        (M424: ``['xc', 'ux', 'uy', 'uz', 'slot4_unknown', 'dUr', '|w|', 'T9', 'rho', 'dT9']``).
    run_id: str, optional
        File prefix; default: inferred from the first .aaa file (sorted walk of moms_dir). For the default
        pattern ppmpy's rule (the file name up to its last '-'); for another pattern the {run_id} part of the
        first file matching it (ValueError if none does). Not needed by a pattern without {run_id}.
    dx: float, optional
        Run grid spacing [Mm]; default: 'deex' of ``rprof``.
    rprof: str or RprofSet, optional
        rprof directory or ppmpy RprofSet: dx (unless given), the dump times t_s, and the grid of the
        'momsdataset' backend (ppmpy builds it from the RprofSet; without one ppmpy uses slot 0, which is not
        valid for M424, so that backend needs rprof or a MomsDataSet passed in).
    nslots: int
        Variables per block file (ppmpy: 10).
    ghost: int
        Ghost cells per dimension of a block (ppmpy: 2; the slice [ghost-1 : ghost-1+n_block] is kept).
    pattern: str
        File pattern relative to moms_dir, a format string with the fields dump, ext and optionally run_id
        (default: ppmpy's ``{dump:04d}/{run_id}-BQav{dump:04d}.{ext}``).
    rprof_run_id: str, optional
        Run id of the rprof files in an rprof directory (default: ``run_id``); KeyError when the directory has
        no files of it.

    Notes
    -----
    Pickling (worker processes started with 'spawn') replaces an RprofSet by its directory and run id, with dx
    resolved first: the workers then read t_s from the rprof headers (:func:`read_rprof_header`, the same
    values as ``RprofSet.get('t', dump)``).
    """

    def __init__(self, moms_dir, slots, run_id=None, dx=None, rprof=None, nslots=10, ghost=2,
                 pattern=DEFAULT_PATTERN, rprof_run_id=None):
        self.moms_dir = os.fspath(moms_dir)
        if not os.path.isdir(self.moms_dir):
            raise FileNotFoundError("moms directory {} does not exist".format(self.moms_dir))
        if isinstance(slots, dict):
            self.slots = {str(k): int(v) for k, v in slots.items()}
        else:
            self.slots = {str(k): i for i, k in enumerate(slots) if k is not None}
        self.nslots = sph._positive_int(nslots, "nslots")
        bad = {k: v for k, v in self.slots.items() if not 0 <= v < self.nslots}
        if bad:
            raise ValueError("slots outside 0..{}: {}".format(self.nslots - 1, bad))
        self.ghost = int(ghost)
        if self.ghost < 1:
            raise ValueError("ghost must be >= 1 (ppmpy keeps [ghost-1 : ghost-1+n])")
        self.pattern = pattern
        self._fields = _pattern_fields(pattern)
        self.rprof = rprof
        self.rprof_run_id = rprof_run_id
        self._rprof_prefix = ""
        self._dx = dx
        self._rprof_files = None
        self._dumps = None
        self.run_id = run_id if run_id is not None else self._find_run_id()

    def __getstate__(self):
        st = dict(self.__dict__)
        st["_dumps"] = None
        if self.rprof is not None and _is_rprofset(self.rprof):
            # an RprofSet does not pickle: keep its dx, directory and run id (t_s then from the headers)
            directory = getattr(self.rprof, "_RprofSet__dir_name", None)
            if not directory:
                raise TypeError("cannot pickle a MomsSource whose rprof object has no directory "
                                "(_RprofSet__dir_name); pass the rprof directory instead")
            st["_dx"] = self.dx
            st["rprof"] = directory
            get_run_id = getattr(self.rprof, "get_run_id", None)
            st["rprof_run_id"] = str(get_run_id()) if get_run_id is not None else self._rprof_run_id()
            st["_rprof_prefix"] = "BQav" if getattr(self.rprof, "_RprofSet__bqav", False) else ""
            st["_rprof_files"] = None
        return st

    def _rprof_run_id(self):
        return self.rprof_run_id if self.rprof_run_id is not None else self.run_id

    def _walk(self):
        """Paths of the files under moms_dir relative to it ('/'-separated), in sorted walk order."""
        for dirpath, dirnames, filenames in os.walk(self.moms_dir):
            dirnames.sort()
            rel = os.path.relpath(dirpath, self.moms_dir)
            for f in sorted(filenames):
                yield f if rel == "." else "/".join(rel.split(os.sep) + [f])

    def _confirmed(self, rel, m, run_id):
        """dump of a regex match whose formatted pattern gives the path back, else None."""
        d = int(m.group("dump"))
        try:
            ok = self.pattern.format(dump=d, run_id=run_id, ext="aaa") == rel
        except (ValueError, TypeError):
            ok = False
        return d if ok else None

    def _find_run_id(self):
        if "run_id" not in self._fields:
            return None
        if self.pattern == DEFAULT_PATTERN:
            # PP 2026-10-02: ported from ppm.py MomsDataSet._find_dumps (run id = file name up to its last '-';
            # ppmpy takes the first file of an unsorted walk, here the sorted walk)
            for rel in self._walk():
                if rel.endswith(".aaa"):
                    name = rel.split("/")[-1]
                    if "-" not in name:
                        raise ValueError("cannot infer the run id from {} (ppmpy's rule: the name up to its last "
                                         "'-'); pass run_id or a pattern".format(os.path.join(self.moms_dir, rel)))
                    return name[:name.rindex("-")]
            raise FileNotFoundError("no .aaa files in {}".format(self.moms_dir))
        rx = _pattern_regex(self.pattern)
        for rel in self._walk():
            m = rx.match(rel)
            if m and self._confirmed(rel, m, m.group("run_id")) is not None:
                return m.group("run_id")
        raise ValueError("no file under {} matches the pattern {!r} (block 'aaa'): cannot infer run_id".format(
            self.moms_dir, self.pattern))

    def slot(self, name):
        """Slot index of a variable name (or an int slot)."""
        if isinstance(name, (int, np.integer)):
            if not 0 <= int(name) < self.nslots:
                raise ValueError("slot {} outside 0..{}".format(name, self.nslots - 1))
            return int(name)
        try:
            return self.slots[name]
        except KeyError:
            raise KeyError("unknown moms variable {!r}; slots: {}".format(name, self.slots)) from None

    def dumps(self):
        """
        Sorted dump numbers with an .aaa file of this run id. Default pattern: ppmpy's discovery (any .aaa file
        under moms_dir named <run_id>-XXXXNNNN.aaa, the dump after 4 characters); another pattern: the files
        whose path relative to moms_dir is the pattern formatted with their dump number (ext 'aaa').
        """
        if self._dumps is None:
            found = set()
            if self.pattern == DEFAULT_PATTERN:
                # PP 2026-10-02: ported from ppm.py MomsDataSet._find_dumps (files without '-' skipped: not of the run)
                for rel in self._walk():
                    f = rel.split("/")[-1]
                    if f.endswith(".aaa") and "-" in f and f[:f.rindex("-")] == self.run_id:
                        num = f[f.rindex("-") + 1:].split(".")[0][4:]
                        if num.isnumeric():
                            found.add(int(num))
            else:
                rx = _pattern_regex(self.pattern, self.run_id if "run_id" in self._fields else None)
                for rel in self._walk():
                    m = rx.match(rel)
                    if m:
                        d = self._confirmed(rel, m, self.run_id)
                        if d is not None:
                            found.add(d)
            self._dumps = sorted(found)
        return list(self._dumps)

    def path(self, dump, ext="aaa"):
        """The file of block ``ext`` of a dump."""
        return os.path.join(self.moms_dir, self.pattern.format(dump=int(dump), run_id=self.run_id, ext=ext))

    def blocks(self, dump):
        """
        The blocks of a dump as (ext, path) pairs in extension order ('aaa', 'aab', ..., the nb^3 extensions of
        the letters a.. for nb blocks per dimension).

        Raises
        ------
        FileNotFoundError
            No .aaa file, or the number of blocks is not a cube (ppmpy: "Missing at least one moms file").
        ValueError
            Block files that do not form the nb^3 layout (e.g. 'aaa'..'aac' with 8 files).
        """
        # PP 2026-10-02: ported from ppm.py MomsData.read_moms_cube (glob, sort, nbq_per_dim); the glob is that of
        # the pattern with ext = three letters, so only this dump's blocks match whatever the pattern
        aaa = self.path(dump, "aaa")
        if not os.path.exists(aaa):
            raise FileNotFoundError("dump {}: {} does not exist".format(dump, aaa))
        files = glob.glob(glob.escape(self.path(dump, _MARK)).replace(_MARK, "[a-z][a-z][a-z]"))
        nb = int(round(len(files) ** (1. / 3.)))
        if nb ** 3 != len(files):
            raise FileNotFoundError("dump {}: {} block files, not a cube (missing blocks?)".format(dump, len(files)))
        exts = ["".join(e) for e in itertools.product(string.ascii_lowercase[:nb], repeat=3)]
        pairs = [(e, self.path(dump, e)) for e in exts]
        if {os.path.normpath(p) for _, p in pairs} != {os.path.normpath(f) for f in files}:
            raise ValueError("dump {}: block files do not form the {}^3 layout {}..{}: {}".format(
                dump, nb, exts[0], exts[-1], sorted(files)))
        return pairs

    def block_files(self, dump):
        """The block files of a dump in extension order (:meth:`blocks`; ppmpy's sorted glob)."""
        return [p for _, p in self.blocks(dump)]

    @property
    def dx(self):
        """Run grid spacing: the ``dx`` given, else 'deex' of the first rprof dump (run id ``rprof_run_id``)."""
        if self._dx is None:
            if self.rprof is None:
                raise ValueError("MomsSource needs dx or rprof for the grid")
            self._dx = _rprof_dx(self.rprof, self._rprof_run_id(), self._rprof_prefix)
        return self._dx

    def grid(self, n, coarsen=4):
        """:class:`MomsGrid` of n cells per dimension with this source's dx."""
        return MomsGrid(self.dx, n, coarsen)

    def time_s(self, dump):
        """
        Problem time of a dump [s] from the rprof (``float(rprofset.get('t', dump))`` as figstyle.time_s, or the
        header of ``<rprof_run_id>-NNNN.rprof``); NaN without rprof.
        """
        if self.rprof is None:
            return float("nan")
        if _is_rprofset(self.rprof):
            return float(self.rprof.get("t", int(dump)))
        if self._rprof_files is None:
            self._rprof_files = rprof_files(self.rprof, self._rprof_run_id(), self._rprof_prefix)
        try:
            path = self._rprof_files[int(dump)]
        except KeyError:
            raise KeyError("no rprof file of dump {} in {}".format(dump, self.rprof)) from None
        return float(read_rprof_header(path, ("t",))["t"])


class MomsBlockReader:
    """
    One moms dump read from its block files (float32), for the slots requested: by plain reads of contiguous plane
    ranges (``mode='read'``, default) or through read-only memory maps (``mode='memmap'``).

    Parameters
    ----------
    source: MomsSource
    dump: int
    slots: sequence, optional
        Variable names or slot indices to read (default: all of the source's slot map).
    mode: {'read', 'memmap'}
        'read': ``np.fromfile`` of the planes a request needs (one contiguous read per block and slot);
        'memmap': ``np.memmap`` of each requested slot of each block file. Same values. On Lustre (/scratch)
        page faults of memory maps cost much kernel time even for cached data (M424 dump 3200, 1 236 544
        points: 17 s of system time with memmap vs ~1 s with reads), so 'read' is the default.

    Attributes
    ----------
    n: int
        Cells per dimension of the assembled cube (blocks per dimension x cells per block).
    nb, nblock, size: int
        Blocks per dimension, cells per block (without ghosts), cells per block with ghosts.
    dtype: np.dtype
        dtype of the cube ``read_moms_cube`` assembles: float64 for several blocks, float32 for one.

    Notes
    -----
    Block 'abc' holds cube[c2 nblock : ..., c1 nblock : ..., c0 nblock : ...] (index order z, y, x; ppmpy maps
    extension letters 3, 2, 1 to the axes 1, 2, 3 of its var array) from its ghost-padded array
    [slot, g-1 : g-1+nblock, ...] (g = ghost).
    """

    def __init__(self, source, dump, slots=None, mode="read"):
        # PP 2026-10-02: ported from ppm.py MomsData.read_moms_cube (sizes from the file length, ghost slice, block
        # placement by extension, float64 assembly)
        if mode not in ("read", "memmap"):
            raise ValueError("mode must be 'read' or 'memmap', got {!r}".format(mode))
        self.source = source
        self.dump = int(dump)
        self.mode = mode
        pairs = source.blocks(self.dump)
        self.files = [f for _, f in pairs]
        fsize = os.path.getsize(self.files[0])
        size = int(round((fsize // 4 // source.nslots) ** (1. / 3.)))
        if size ** 3 * source.nslots * 4 != fsize:
            raise ValueError("{}: size {} B is not {} slots of a float32 cube".format(self.files[0], fsize,
                                                                                     source.nslots))
        if size <= source.ghost:
            raise ValueError("{}: blocks of {}^3 cells cannot hold {} ghost cells".format(self.files[0], size,
                                                                                       source.ghost))
        self.size = size
        self.nblock = size - source.ghost
        self.nb = int(round(len(self.files) ** (1. / 3.)))
        self.n = self.nb * self.nblock
        self.dtype = np.dtype(np.float32 if self.nb == 1 else np.float64)
        names = list(source.slots) if slots is None else list(slots)
        self.slots = sorted({source.slot(s) for s in names})
        self.blocks = []
        for ext, f in pairs:
            if os.path.getsize(f) != fsize:
                raise ValueError("block files of dump {} differ in size".format(self.dump))
            pos = (ord(ext[2]) - 97, ord(ext[1]) - 97, ord(ext[0]) - 97)
            maps = None
            if mode == "memmap":
                maps = {s: np.memmap(f, dtype=np.float32, mode="r", offset=s * size ** 3 * 4,
                                     shape=(size, size, size)) for s in self.slots}
            self.blocks.append((pos, f, maps))

    def _check_slot(self, slot):
        s = self.source.slot(slot)
        if s not in self.slots:
            raise KeyError("slot {} not read by this reader (slots: {})".format(slot, self.slots))
        return s

    def _planes(self, f, maps, s, a0, a1):
        """Ghost-padded planes a0:a1 of slot s of block file f, shape (a1-a0, size, size), float32."""
        if maps is not None:
            return maps[s][a0:a1]
        size = self.size
        arr = np.fromfile(f, dtype=np.float32, count=(a1 - a0) * size * size, offset=(s * size + a0) * size * size * 4)
        if arr.size != (a1 - a0) * size * size:
            raise IOError("{}: short read".format(f))
        return arr.reshape(a1 - a0, size, size)

    def slab(self, slot, i0=0, i1=None, j0=0, j1=None, k0=0, k1=None):
        """
        The box [i0:i1, j0:j1, k0:k1] (index order z, y, x) of a slot of the cube, in :attr:`dtype`: the values
        ``MomsDataSet.get(slot)[i0:i1, j0:j1, k0:k1]`` would hold.
        """
        s = self._check_slot(slot)
        n = self.n
        lo = (i0, j0, k0)
        hi = tuple(n if v is None else v for v in (i1, j1, k1))
        if not all(0 <= a <= b <= n for a, b in zip(lo, hi)):
            raise ValueError("box {} outside 0..{}".format(list(zip(lo, hi)), n))
        out = np.empty(tuple(b - a for a, b in zip(lo, hi)), dtype=self.dtype)
        nbk, g0 = self.nblock, self.source.ghost - 1
        filled = 0
        for pos, f, maps in self.blocks:
            src, dst = [], []
            for p, a, b in zip(pos, lo, hi):
                c0, c1 = max(a, p * nbk), min(b, (p + 1) * nbk)
                if c0 >= c1:
                    break
                src.append(slice(g0 + c0 - p * nbk, g0 + c1 - p * nbk))
                dst.append(slice(c0 - a, c1 - a))
            else:
                planes = self._planes(f, maps, s, src[0].start, src[0].stop)
                out[tuple(dst)] = planes[:, src[1], src[2]]
                filled += int(np.prod([d.stop - d.start for d in dst]))
        if filled != out.size:
            raise RuntimeError("box not covered by the blocks of dump {}".format(self.dump))
        return out

    def gather(self, slot, i, j, k):
        """
        Values of a slot at the cells (i, j, k) (index order z, y, x; 1-d integer arrays), in :attr:`dtype`:
        ``MomsDataSet.get(slot)[i, j, k]``. Reads the planes min(i)..max(i) of the blocks that hold cells.
        """
        s = self._check_slot(slot)
        i, j, k = (np.asarray(a, dtype=np.intp) for a in (i, j, k))
        n = self.n
        if i.size and (min(i.min(), j.min(), k.min()) < 0 or max(i.max(), j.max(), k.max()) >= n):
            raise ValueError("cell indices outside 0..{}".format(n - 1))
        nbk, g0 = self.nblock, self.source.ghost - 1
        bi, bj, bk = i // nbk, j // nbk, k // nbk
        out = np.empty(i.shape, dtype=self.dtype)
        for pos, f, maps in self.blocks:
            m = (bi == pos[0]) & (bj == pos[1]) & (bk == pos[2])
            if m.any():
                li = g0 + i[m] - pos[0] * nbk
                a0 = int(li.min())
                planes = self._planes(f, maps, s, a0, int(li.max()) + 1)
                out[m] = planes[li - a0, g0 + j[m] - pos[1] * nbk, g0 + k[m] - pos[2] * nbk]
        return out

    def cube(self, slot):
        """The whole cube of a slot (n^3 values in :attr:`dtype`; tests and small dumps)."""
        return self.slab(slot)

    def close(self):
        """Drop the memory maps."""
        self.blocks = [(pos, f, None) for pos, f, maps in self.blocks]


# ----------------------------------------------------------------------------------------------
# spherical components
# ----------------------------------------------------------------------------------------------
def grid_jacobian(x, y, z, r, dtype=None):
    """
    Unit vectors r_hat, theta_hat, phi_hat in Cartesian components at the cells (ppmpy
    ``MomsDataSet._get_jacobian``).

    Parameters
    ----------
    x, y, z, r: np.ndarray
        Coordinates of the cells (same shape; M424: float64, contiguous as from :meth:`MomsGrid.box`).
    dtype: np.dtype, optional
        Output dtype; None: ppmpy's rule below. The slab backend passes float32 for its 1-d cell lists (ppmpy's
        grid case, the cells gathered instead of a box: elementwise the same numbers).

    Returns
    -------
    np.ndarray
        (8,) + x.shape: float32 for arrays of 2 or more dimensions (ppmpy's grid case), float64 for 1-d arrays
        (ppmpy's igrid case). Rows: r_hat . (x, y, z), theta_hat . (x, y, z), phi_hat . (x, y) (phi_hat . z = 0).
        The theta_hat rows are float32 quotients of float32-rounded products (ppmpy's ``out=`` placeholders).

    Notes
    -----
    ppmpy allocates (8, x.shape[0], y.shape[0], z.shape[0]), which is (8,) + x.shape for its cubes; here
    (8,) + x.shape for any box.
    """
    # PP 2026-10-02: ported from ppm.py MomsDataSet._get_jacobian (verbatim apart from the allocation shape)
    if dtype is not None:
        jacobian = np.zeros((8,) + x.shape, dtype=dtype)
    elif len(x.shape) > 1:
        jacobian = np.zeros((8,) + x.shape, dtype='float32')
    else:
        jacobian = np.zeros((8, x.shape[0]))

    rcyl = np.sqrt(np.power(x, 2.0) + np.power(y, 2.0))

    np.divide(x, r, out=jacobian[0])
    np.divide(y, r, out=jacobian[1])
    np.divide(z, r, out=jacobian[2])

    np.divide(np.multiply(x, z, out=jacobian[3]), np.multiply(r, rcyl, out=jacobian[4]),
              out=jacobian[3])
    np.divide(np.multiply(y, z, out=jacobian[4]), np.multiply(r, rcyl, out=jacobian[5]),
              out=jacobian[4])
    np.divide(-rcyl, r, out=jacobian[5])

    np.divide(-y, rcyl, out=jacobian[6])
    np.divide(x, rcyl, out=jacobian[7])
    return jacobian


def _components(ux, uy, uz, jacobian, components=True):
    # PP 2026-10-02: ported from ppm.py MomsDataSet.get_spherical_components (same expressions and order)
    ur = ux * jacobian[0] + uy * jacobian[1] + uz * jacobian[2]
    if not components:
        return [ur]
    utheta = ux * jacobian[3] + uy * jacobian[4] + uz * jacobian[5]
    uphi = ux * jacobian[6] + uy * jacobian[7]
    return [ur, utheta, uphi]


# ----------------------------------------------------------------------------------------------
# sampling one dump
# ----------------------------------------------------------------------------------------------
def _grid_mismatch(coord, source, what):
    """ValueError text when ppmpy's grid `coord` is not the source's grid (None if equal or nothing to compare)."""
    if source._dx is None and source.rprof is None:
        return None
    ref = source.grid(len(coord)).coord
    if np.array_equal(coord, ref):
        return None
    return ("{}: ppmpy's grid (first {!r}, step {!r}) is not the source's grid (dx {!r}: first {!r}, step {!r}); "
            "ppmpy builds its grid from the RprofSet ('deex' of its first dump) or, without one, from slot 0 (not "
            "valid for M424)".format(what, float(coord[0]), float(coord[1] - coord[0]), source.dx, float(ref[0]),
                                     float(ref[1] - ref[0])))


def _momsdataset(source, dump):
    """A ppmpy MomsDataSet over the source (one dump in memory), as figstyle.moms(), after checking that its grid
    (from the rprof) will be the source's grid."""
    if source.pattern != DEFAULT_PATTERN:
        raise ValueError("the momsdataset backend reads ppmpy's file pattern only; use backend='slab'")
    if source.nslots != 10:
        raise ValueError("ppmpy's MomsData has 10 slots; use backend='slab' for nslots={}".format(source.nslots))
    if source.rprof is None:
        raise ValueError("backend='momsdataset' needs source.rprof (without an RprofSet ppmpy builds the grid from "
                         "slot 0, which is not valid for M424 and ignores dx): pass rprof, a MomsDataSet as moms, or "
                         "use backend='slab'")
    if source._dx is not None:
        # before loading the dump (18 GB for M424): the rprof's dx must give the source's grid
        n = MomsBlockReader(source, dump, slots=[]).n
        msg = _grid_mismatch(MomsGrid(_rprof_dx(source.rprof, source._rprof_run_id(), source._rprof_prefix), n).coord,
                             source, "dump {}".format(dump))
        if msg:
            raise ValueError(msg)
    from ppmpy import ppm
    rp = source.rprof
    if not _is_rprofset(rp):
        rp = ppm.RprofSet(os.fspath(rp), verbose=0)
    names = [None] * source.nslots
    for k, v in source.slots.items():
        names[v] = k
    var_list = names if all(nm is not None for nm in names) else []
    return ppm.MomsDataSet(source.moms_dir, init_dump_read=int(dump), dumps_in_mem=1, rprofset=rp,
                           var_list=var_list, verbose=0)


def _check_momsdataset(m, source, dump):
    """The MomsDataSet must read the source's files (run id) on the source's grid."""
    coord = getattr(m, "_unique_coord", None)
    if coord is None:
        raise ValueError("dump {}: the ppmpy MomsDataSet is not valid (no grid; dump not found?)".format(dump))
    rid = getattr(m, "_run_id", source.run_id)
    if rid != source.run_id:
        raise ValueError("dump {}: the MomsDataSet reads run id {!r}, the source {!r}".format(dump, rid,
                                                                                            source.run_id))
    msg = _grid_mismatch(np.asarray(coord), source, "dump {}".format(dump))
    if msg:
        raise ValueError(msg)
    return np.asarray(coord)


def _sample_momsdataset(m, dump, radius, npoints, tslot, vslots, velocity_scale, components, igrid=None):
    """
    The legacy path (fw_sphere_extract.py / sphere_sample.py) through a MomsDataSet m: the sphere of ``radius``
    with ``npoints`` points (``get_spherical_interpolation``), or the points ``igrid`` ([z, y, x] rows,
    ``get_interpolation``).
    """
    # PP 2026-10-02: ported from sphere_sample.py:47-53 and fw_sphere_extract.py:38-41
    if igrid is None:
        def interp(v):
            return m.get_spherical_interpolation(v, radius, fname=dump, npoints=npoints)
    else:
        def interp(v):
            return m.get_interpolation(v, igrid, fname=dump)
    T = interp(tslot)
    vel = {}
    if vslots is not None:
        ux, uy, uz = (m.get(i, fname=dump) for i in vslots)
        comps = m.get_spherical_components(ux, uy, uz)
        del ux, uy, uz
        keys = ("ur", "uth", "uph") if components else ("ur",)
        for k, c in zip(keys, comps):
            v = interp(c)
            vel[k] = v * velocity_scale if velocity_scale is not None else v
        del comps
    return T, vel


def _sample_slab(source, dump, z, y, x, tslot, vslots, velocity_scale, components, slab, cells="corners"):
    """
    The slab backend: the points are grouped by boxes of `slab` z intervals; per box the T field and the velocity
    components are computed at the corner cells of its points (cells='corners') or on all its cells ('box'), then
    interpolated with RegularGridInterpolator on the box. Only the 8 corner values of a point enter its result, so
    the values elsewhere in the box array are never read.
    """
    reader = MomsBlockReader(source, dump, [tslot] + (list(vslots) if vslots is not None else []))
    try:
        grid = source.grid(reader.n)
        g = grid.coord
        # interval indices as RegularGridInterpolator finds them (bounds checked like bounds_error=True)
        iz, iy, ix = grid.intervals(z), grid.intervals(y), grid.intervals(x)
        npt = z.shape[0]
        nf = 1 + (0 if vslots is None else (3 if components else 1))
        res = np.empty((npt, nf))
        kslab = iz // int(slab)
        order = np.argsort(kslab, kind="stable")
        bounds = np.searchsorted(kslab[order], np.arange(kslab.max() + 2))
        xi = np.empty((npt, 3))
        xi[:, 0], xi[:, 1], xi[:, 2] = z, y, x               # ppmpy igrid order [z, y, x]
        corner = np.array([(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)])
        for kk in range(len(bounds) - 1):
            sel = order[bounds[kk]:bounds[kk + 1]]
            if sel.size == 0:
                continue
            i0, i1 = int(iz[sel].min()), int(iz[sel].max()) + 2
            j0, j1 = int(iy[sel].min()), int(iy[sel].max()) + 2
            k0, k1 = int(ix[sel].min()), int(ix[sel].max()) + 2
            shape = (i1 - i0, j1 - j0, k1 - k0)
            if cells == "box":
                fields = [reader.slab(tslot, i0, i1, j0, j1, k0, k1)]
                if vslots is not None:
                    u = [reader.slab(s, i0, i1, j0, j1, k0, k1) for s in vslots]
                    bx, by, bz, br = grid.box(i0, i1, j0, j1, k0, k1)
                    jac = grid_jacobian(bx, by, bz, br)
                    del bx, by, bz, br
                    fields += _components(u[0], u[1], u[2], jac, components)
                    del u, jac
                vals = np.stack(fields, axis=-1)
                del fields
            else:
                # the distinct corner cells of the points (box-local flat index)
                li = (iz[sel] - i0)[:, None] + corner[:, 0]
                lj = (iy[sel] - j0)[:, None] + corner[:, 1]
                lk = (ix[sel] - k0)[:, None] + corner[:, 2]
                flat = np.unique(np.ravel_multi_index((li.ravel(), lj.ravel(), lk.ravel()), shape))
                del li, lj, lk
                ci, cj, ck = np.unravel_index(flat, shape)
                gi, gj, gk = ci + i0, cj + j0, ck + k0
                fields = [reader.gather(tslot, gi, gj, gk)]
                if vslots is not None:
                    u = [reader.gather(s, gi, gj, gk) for s in vslots]
                    # cell coordinates as MomsGrid.box (1.0 * g == g), r as ppmpy's _radius, float32 Jacobian
                    cx, cy, cz = g[gk], g[gj], g[gi]
                    cr = np.sqrt(np.power(cx, 2.0) + np.power(cy, 2.0) + np.power(cz, 2.0))
                    jac = grid_jacobian(cx, cy, cz, cr, dtype=np.float32)
                    del cx, cy, cz, cr
                    fields += _components(u[0], u[1], u[2], jac, components)
                    del u, jac
                vals = np.empty(shape + (nf,), dtype=np.result_type(*fields))
                vals.reshape(-1, nf)[flat] = np.stack(fields, axis=-1)
                del fields, flat, ci, cj, ck, gi, gj, gk
            rgi = scipy.interpolate.RegularGridInterpolator((g[i0:i1], g[j0:j1], g[k0:k1]), vals)
            res[sel] = rgi(xi[sel])
            del rgi, vals
    finally:
        reader.close()
    T = res[:, 0].copy()
    vel = {}
    if vslots is not None:
        keys = ("ur", "uth", "uph") if components else ("ur",)
        for j, k in enumerate(keys):
            v = res[:, 1 + j].copy()
            vel[k] = v * velocity_scale if velocity_scale is not None else v
    return T, vel, g


def _check_args(backend, velocity, slab, slab_cells):
    if backend not in BACKENDS:
        raise ValueError("backend must be one of {}, got {!r}".format(BACKENDS, backend))
    if velocity is not None and len(velocity) != 3:
        raise ValueError("velocity must name 3 slots (ux, uy, uz) or be None")
    if slab_cells not in ("corners", "box"):
        raise ValueError("slab_cells must be 'corners' or 'box', got {!r}".format(slab_cells))
    return sph._positive_int(slab, "slab")


def _check_finite(dump, arrays, what):
    bad = [k for k, v in arrays.items() if not np.all(np.isfinite(v))]
    if bad:
        raise ValueError("dump {}: non-finite sampled values in {} ({})".format(dump, bad, what))


def sample_moms_points(source, dump, z, y, x, temperature="T9", velocity=("ux", "uy", "uz"), velocity_scale=1e3,
                       backend="slab", slab=16, slab_cells="corners", moms=None, components=True):
    """
    The temperature slot and the spherical velocity components of one dump at any points, interpolated
    trilinearly on the moms cell centres (ppmpy ``MomsDataSet.get_interpolation``).

    Parameters
    ----------
    source: MomsSource
    dump: int
    z, y, x: array-like
        (N,) coordinates of the points [Mm] (converted to float64), inside [g[0], g[-1]] of the grid in every
        coordinate (ValueError otherwise; RegularGridInterpolator's bounds_error).
    temperature: str or int
        Temperature slot (M424: 'T9').
    velocity: sequence of 3 or None
        Slots of u_x, u_y, u_z; None: no velocities.
    velocity_scale: float or None
        Factor applied to the interpolated components (M424: 1e3, Mm/s -> km/s); None: none.
    backend: {'slab', 'momsdataset'}
        'slab' (default): block-file reads, box by box, the grid from the source's dx. 'momsdataset': ppmpy
        MomsDataSet (``moms`` or one built from the source, which needs ``source.rprof``; its grid is checked
        against the source's grid when the source has dx or rprof); the oracle of the slab backend.
    slab: int
        z intervals per box of the slab backend. Results do not depend on it; memory does: the box array holds
        (slab + 1) x n_y x n_x cells x (1 + 3 velocity components) x 8 B, with n_y, n_x the extent of the box's
        points in cells (M424 at 4050 Mm: up to the whole 448^2 plane, so ~115 MB for slab = 16 and ~2.9 GB for a
        single box, slab >= n).
    slab_cells: {'corners', 'box'}
        Slab backend: compute the fields at the corner cells of the points only (default, fast) or on every cell
        of each box (the dense variant; same bits; on top of the box array ~100 B per box cell for the reads, the
        coordinates and the float32 Jacobian, and ~10x slower for M424: 16 s against 1.5 s per dump).
    moms: ppmpy MomsDataSet, optional
        momsdataset backend: use this one; it must read the source's run id on the source's grid (checked).
    components: bool
        True: ur, uth, uph; False: ur only (the same ur bits).

    Returns
    -------
    dict
        T (N,) float64 the interpolated temperature slot; ur[, uth, uph] (N,) float64 [velocity units x
        velocity_scale]; grid (n,) float64 the cell-centre coordinates used.

    Raises
    ------
    ValueError
        Unknown backend, points outside the grid, a MomsDataSet on another grid, or non-finite sampled values.
    """
    slab = _check_args(backend, velocity, slab, slab_cells)
    dump = int(dump)
    z, y, x = (np.asarray(a, dtype=np.float64) for a in (z, y, x))
    if z.ndim != 1 or not z.shape == y.shape == x.shape:
        raise ValueError("z, y, x must be 1-d arrays of equal length")
    tslot = source.slot(temperature)
    vslots = None if velocity is None else [source.slot(v) for v in velocity]
    if backend == "momsdataset":
        m = _momsdataset(source, dump) if moms is None else moms
        coord = _check_momsdataset(m, source, dump)
        igrid = np.empty((z.shape[0], 3))
        igrid[:, 0], igrid[:, 1], igrid[:, 2] = z, y, x       # ppmpy igrid order [z, y, x]
        T, vel = _sample_momsdataset(m, dump, None, None, tslot, vslots, velocity_scale, components, igrid=igrid)
        del m
    else:
        T, vel, coord = _sample_slab(source, dump, z, y, x, tslot, vslots, velocity_scale, components, slab,
                                     slab_cells)
    out = dict(T=T)
    out.update(vel)
    _check_finite(dump, out, "outer cells NaN?")
    out["grid"] = coord
    return out


def sample_moms_sphere(source, dump, radius, npoints, teff0, temperature="T9", velocity=("ux", "uy", "uz"),
                       velocity_scale=1e3, backend="slab", slab=16, moms=None, components=True,
                       slab_cells="corners"):
    """
    One moms dump on the equal-area sphere: temperature perturbation, T_eff' and velocity components per point.

    Parameters
    ----------
    source: MomsSource
    dump: int
    radius: float
        Sphere radius [Mm] (M424: 4050).
    npoints: int
        Points of :func:`ppmpy.synspec.sphere.fibonacci_sphere` (M424: 1 236 544).
    teff0: float
        T_eff of the unperturbed model [K] (M424: 38230): teff = teff0 (1 + relT).
    temperature, velocity, velocity_scale, slab, slab_cells, components:
        See :func:`sample_moms_points` (components False: ur only, as fw_sphere_extract.py).
    backend: {'slab', 'momsdataset'}
        'slab' (default; 0.3 GB and ~2-6 s per M424 dump): :func:`sample_moms_points` at the sphere's x, y, z.
        'momsdataset': the legacy code path, ppmpy ``get_spherical_interpolation`` / ``get_spherical_components``
        on a MomsDataSet (``moms``, or one built from the source with its rprof: 18 GB and ~50 s per M424 dump).
        Same bits.
    moms: ppmpy MomsDataSet, optional
        momsdataset backend: use this one (checked: run id and grid of the source).

    Returns
    -------
    dict
        theta, phi, x, y, z (N,) float64 (x, y, z: the interpolation points [Mm]); T (N,) the interpolated
        temperature slot, T_mean its mean, relT, teff (N,) float64; ur[, uth, uph] (N,) float64 [velocity
        units x velocity_scale]; t_s (problem time, NaN without rprof); meta (dict: dump, radius, npoints, teff0,
        temperature, velocity, backend, seconds, grid start and step).

    Raises
    ------
    ValueError
        Unknown backend, points outside the grid, a MomsDataSet on another grid, or non-finite sampled values.
    """
    slab = _check_args(backend, velocity, slab, slab_cells)
    t0 = time.time()
    dump = int(dump)
    radius = float(radius)
    npoints = sph._positive_int(npoints, "npoints")
    tslot = source.slot(temperature)
    vslots = None if velocity is None else [source.slot(v) for v in velocity]
    # PP 2026-10-02: ported from fw_sphere_extract.py:44-53 (grid and x, y, z: the expressions of ppmpy's igrid)
    theta, phi = sph.fibonacci_sphere(npoints)
    x, y, z = sph.sphere_xyz(theta, phi, radius)
    if backend == "momsdataset":
        m = _momsdataset(source, dump) if moms is None else moms
        coord = _check_momsdataset(m, source, dump)
        T, vel = _sample_momsdataset(m, dump, radius, npoints, tslot, vslots, velocity_scale, components)
        del m
    else:
        p = sample_moms_points(source, dump, z, y, x, temperature=tslot, velocity=vslots,
                               velocity_scale=velocity_scale, backend="slab", slab=slab, slab_cells=slab_cells,
                               components=components)
        coord = p.pop("grid")
        T = p.pop("T")
        vel = p
    # PP 2026-10-02: ported from fw_sphere_extract.py:58-59 / sphere_sample.py:54-55
    T_mean = T.mean()
    relT = (T - T_mean) / T_mean
    teff = teff0 * (1.0 + relT)
    out = dict(theta=theta, phi=phi, x=x, y=y, z=z, T=T, T_mean=float(T_mean), relT=relT, teff=teff)
    out.update(vel)
    _check_finite(dump, {k: out[k] for k in ("T", "relT", "teff") + tuple(vel)},
                  "outer cells NaN? radius {} Mm".format(radius))
    out["t_s"] = source.time_s(dump)
    out["meta"] = dict(dump=dump, radius=radius, npoints=npoints, teff0=teff0, temperature=str(temperature),
                       velocity=None if velocity is None else [str(v) for v in velocity],
                       velocity_scale=velocity_scale, backend=backend, slab=slab if backend == "slab" else None,
                       slab_cells=slab_cells if backend == "slab" else None,
                       components=bool(components), grid_first=float(coord[0]), grid_step=float(coord[1] - coord[0]),
                       seconds=time.time() - t0)
    return out


# ----------------------------------------------------------------------------------------------
# products
# ----------------------------------------------------------------------------------------------
def _atomic(path, write):
    base, ext = os.path.splitext(path)
    tmp = "{}.tmp{}{}".format(base, os.getpid(), ext)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def write_points_table(outdir, sample, teff0, fmt=("%d", "%.3f"), meta=None):
    """
    The products of fw_sphere_extract.py for a sample: points.npz, points.txt and meta.json (atomic writes).

    Parameters
    ----------
    outdir: str
        Directory (created).
    sample: dict
        From :func:`sample_moms_sphere` (needs ur; the temperature key in points.npz is the sample's
        ``meta['temperature']``, M424 'T9').
    teff0: float
        Recorded in meta.json (as a float, like the legacy argparse value).
    fmt: tuple
        Formats of the 'idx teff' lines of points.txt.
    meta: dict, optional
        Entries added to (or replacing) the legacy meta.json record.

    Returns
    -------
    dict
        The meta.json record.

    Notes
    -----
    points.npz: idx (int32), r, theta, phi, x, y, z, relT, teff, ur_kms, <temperature> (legacy order);
    points.txt equals the legacy file byte for byte, meta.json too for the legacy record (json, indent=1).
    """
    # PP 2026-10-02: ported from fw_sphere_extract.py:57-81
    os.makedirs(outdir, exist_ok=True)
    tname = sample["meta"]["temperature"]
    T9, relT, teff, ur_kms = sample["T"], sample["relT"], sample["teff"], sample["ur"]
    npoints = relT.shape[0]
    radius = float(sample["meta"]["radius"])
    r = np.full(npoints, radius)
    idx = np.arange(npoints, dtype=np.int32)
    arrays = dict(idx=idx, r=r, theta=sample["theta"], phi=sample["phi"], x=sample["x"], y=sample["y"],
                  z=sample["z"], relT=relT, teff=teff, ur_kms=ur_kms)
    arrays[tname] = T9
    _atomic(os.path.join(outdir, "points.npz"), lambda p: np.savez(p, **arrays))
    _atomic(os.path.join(outdir, "points.txt"),
            lambda p: np.savetxt(p, np.column_stack([idx, teff]), fmt=list(fmt)))
    rec = dict(dump=int(sample["meta"]["dump"]), radius_Mm=radius, npoints=int(npoints), teff0=float(teff0),
               t_days=sample["t_s"] / 86400)
    rec[tname + "_mean"] = float(T9.mean())
    rec.update(relT_min=float(relT.min()), relT_max=float(relT.max()), relT_std=float(relT.std()),
               teff_min=float(teff.min()), teff_max=float(teff.max()), ur_min=float(ur_kms.min()),
               ur_max=float(ur_kms.max()), ur_std=float(ur_kms.std()), note=POINTS_NOTE)
    if meta:
        rec.update(meta)

    def _json(p):
        with open(p, "w") as f:
            json.dump(rec, f, indent=1)
    _atomic(os.path.join(outdir, "meta.json"), _json)
    return rec


def sample_path(outdir, dump):
    """Per-dump sample file ``outdir/dNNNN.npz``."""
    return os.path.join(outdir, "d{:04d}.npz".format(int(dump)))


def _dumps_init(source, kw):
    return dict(source=source, kw=kw)


def _dumps_task(dump):
    st = par.worker_state()
    return _sample_one(st["source"], dump, **st["kw"])


def _sample_one(source, dump, radius, npoints, teff0, outdir, backend, check, temperature, velocity,
                velocity_scale, slab, meta):
    # PP 2026-10-02: ported from sphere_sample.py:47-65
    t0 = time.time()
    s = sample_moms_sphere(source, dump, radius, npoints, teff0, temperature=temperature, velocity=velocity,
                           velocity_scale=velocity_scale, backend=backend, slab=slab, components=True)
    tname = s["meta"]["temperature"]
    arrays = dict(relT=s["relT"].astype(np.float32), teff=s["teff"].astype(np.float32))
    for k in ("ur", "uth", "uph"):
        if k in s:
            arrays[k] = s[k].astype(np.float32)
    arrays[tname + "_mean"] = float(s["T"].mean())
    arrays["t_s"] = s["t_s"]
    rec = None
    if meta:
        rec = sio.make_meta("synspec.moms_sample", params=dict(s["meta"], seconds=None),
                            inputs=dict(moms=source.path(dump)))
    sio.save_npz(sample_path(outdir, dump), arrays, meta=rec)
    info = dict(dump=int(dump), seconds=time.time() - t0, teff_min=float(s["teff"].min()),
                teff_max=float(s["teff"].max()))
    if "ur" in s:
        info.update(ur_min=float(s["ur"].min()), ur_max=float(s["ur"].max()))
    if "uth" in s:
        info["ut_rms"] = float(np.sqrt(np.mean(s["uth"] ** 2 + s["uph"] ** 2)))
    if check:
        p = np.load(os.path.join(check, "points.npz"))
        info["check"] = {k: float(np.abs(p[k] - v).max())
                         for k, v in (("relT", s["relT"]), ("teff", s["teff"]), ("ur_kms", s.get("ur")))
                         if v is not None}
    return info


def sample_moms_dumps(source, dumps, radius, npoints, teff0, outdir, rank=0, nranks=1, nproc=1, backend="slab",
                      overwrite=False, check=None, start_method=None, temperature="T9", velocity=("ux", "uy", "uz"),
                      velocity_scale=1e3, slab=16, meta=False, timeout=1800.0, maxtasksperchild=50,
                      log=functools.partial(print, flush=True)):
    """
    Per-dump sphere samples (sphere_sample.py): ``outdir/dNNNN.npz`` with relT, teff, ur, uth, uph (float32,
    points in idx order), ``<temperature>_mean`` and t_s.

    Parameters
    ----------
    source: MomsSource
        With 'spawn' workers it is pickled: an RprofSet is replaced by its directory (see :class:`MomsSource`).
    dumps: iterable of int
        Dumps to sample; dump d goes to rank d % nranks (legacy --worker/--nworkers).
    radius, npoints, teff0:
        See :func:`sample_moms_sphere`.
    outdir: str
        Output directory (created). Existing files are skipped unless ``overwrite`` (restartable); files are
        written to a temporary name and renamed.
    rank, nranks: int
        This rank and the number of ranks (nodes or independent runs).
    nproc: int
        Worker processes (1: in this process). Memory per worker: 18 GB (momsdataset backend, M424), 0.3 GB
        (slab). On a login node keep nproc small (one heavy job at a time).
    backend: {'slab', 'momsdataset'}
    overwrite: bool
    check: str, optional
        Run directory with a points.npz: report max |diff| of relT, teff, ur_kms against it per dump (legacy
        --check; zero for the dump of that run).
    start_method: str, optional
        'fork' or 'spawn' (:func:`ppmpy.synspec.parallel.get_context`).
    temperature, velocity, velocity_scale, slab:
        See :func:`sample_moms_sphere`.
    meta: bool
        Add a '_meta' provenance member (the legacy files have none).
    timeout: float
        Watchdog [s] for a killed worker (:func:`ppmpy.synspec.parallel.imap_watchdog`).
    maxtasksperchild: int or None
        Dumps per worker before it is replaced (restarts the CPU-time count of ``ulimit -t``).
    log: callable or None
        Progress messages (default: print, flushed, as the legacy script).

    Returns
    -------
    list of dict
        Per sampled dump: dump, seconds, T_eff' and u_r ranges, |u_t| rms, and the check differences.
    """
    if backend not in BACKENDS:
        raise ValueError("backend must be one of {}, got {!r}".format(BACKENDS, backend))
    if int(nranks) != nranks or nranks < 1 or int(rank) != rank or not 0 <= rank < nranks:
        raise ValueError("need integers 0 <= rank < nranks, got rank={!r}, nranks={!r}".format(rank, nranks))
    os.makedirs(outdir, exist_ok=True)
    # PP 2026-10-02: ported from sphere_sample.py:41-45 (dump d -> worker d % n; existing files skipped)
    mine = [int(d) for d in dumps if int(d) % nranks == rank]
    todo = [d for d in mine if overwrite or not os.path.exists(sample_path(outdir, d))]
    if log:
        log("rank {}/{}: {} dumps, {} to do -> {}".format(rank, nranks, len(mine), len(todo), outdir))
    kw = dict(radius=radius, npoints=npoints, teff0=teff0, outdir=outdir, backend=backend, check=check,
              temperature=temperature, velocity=velocity, velocity_scale=velocity_scale, slab=slab, meta=meta)
    results = []
    if not todo:
        return results

    def _report(info):
        results.append(info)
        if log:
            msg = "dump {}: Teff' {:.0f}-{:.0f} K".format(info["dump"], info["teff_min"], info["teff_max"])
            if "ur_min" in info:
                msg += ", u_r {:.1f}..{:.1f}".format(info["ur_min"], info["ur_max"])
            if "ut_rms" in info:
                msg += ", |u_t| rms {:.1f} km/s".format(info["ut_rms"])
            msg += "  {:.1f} s".format(info["seconds"])
            if "check" in info:
                msg += "  check max|diff| " + ", ".join("{} {:.3e}".format(k, v) for k, v in info["check"].items())
            log(msg)

    if nproc == 1:
        for d in todo:
            _report(_sample_one(source, d, **kw))
    else:
        with par.make_pool(min(int(nproc), len(todo)), initializer=_dumps_init, initargs=(source, kw),
                           maxtasksperchild=maxtasksperchild, start_method=start_method) as pool:
            for info in par.imap_watchdog(pool, _dumps_task, todo, timeout=timeout):
                _report(info)
    results.sort(key=lambda r: r["dump"])
    return results
