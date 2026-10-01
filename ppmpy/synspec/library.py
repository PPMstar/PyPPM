"""
T_eff' libraries of local flux profiles and their interpolation nodes.

The local models of a per-point FASTWIND run differ only in T_eff' (every other input is fixed), so
the run's continuum-normalised flux profiles form a library in T_eff':

* :class:`FluxLibrary`: mean rest-frame profile (on the velocity grid) and mean continuum flux per
  T_eff' bin of width dT (M424: 10 K, 351 bins of which 303 hold models). :meth:`FluxLibrary.build`
  streams the profiles block by block (e.g. memory maps of profiles.npz), optionally in parallel and
  optionally from a subset of the models (hold-out tests); :meth:`FluxLibrary.load` /
  :meth:`FluxLibrary.save` read and write the legacy ``library_dT10.npz``.
* :func:`lib_nodes` -> :class:`LibraryNodes`: T_eff' interpolation nodes, consecutive filled bins
  merged until a node holds >= nmin models (M424: 245 nodes, 35829-38891 K), optionally with an
  additive per-bin profile correction and a local-linear smoothing in T_eff'.
* :func:`node_pairs`: linear interpolation in T_eff' between nodes (clamped at the ends);
  :func:`coverage`: points beyond the node range and their visible weight.
* :func:`wavelength_rounding_correction`: per-bin correction for the 0.01 A rounding of the
  wavelengths in FASTWIND's OUT files (the 'lamfix' library variant).

Conventions
-----------
* Profiles are interpolated from each model's native wavelengths onto the common grid
  y = c ln(lambda / lref) with :func:`ppmpy.synspec.spectral.interp_rows` (constant beyond the ends
  of the model's band), with the row offsets of the model's position in its block.
* Bins: ``edges = arange(floor(min T / dT) dT, max T + dT, dT)`` and the bin of T is
  ``clip(digitize(T, edges) - 1, 0, nb - 1)`` (:func:`teff_edges`, :func:`teff_bins`), so the
  highest T_eff' falls into the last bin even when it equals an edge.
* Empty bins (sparse tails) get count 0, T = bin centre, and a copy of the profile and F_c of the
  nearest filled bin (the lower one on a tie); :func:`lib_nodes` skips them. ``fill_empty=False``
  leaves them zero instead (the hold-out libraries of fw_disc_holdout.py).
* A node's T_eff', profile and F_c are count-weighted means over its bins, i.e. means over its
  models: merging never changes the mean of a linear function of T_eff'.

Validation
----------
tests/synspec/test_library.py. Synthetic runs (any machine; the legacy code runs next to ours): the
float64 bin sums and means of :meth:`FluxLibrary.build` equal, bit for bit, those of the frozen
fw_disc.library (``block`` = its chunk), of the library by-product of fw_disc_los.py (``stride`` =
its --nproc) and of the hold-out libraries of fw_disc_holdout.py (``select``), for any ``nproc``
(fork and spawn) and ``rows``. M424: ``build`` on profiles.npz equals the production
library_dT10.npz, :func:`lib_nodes` the frozen ``fw_disc.lib_nodes`` for (nmin=20), (nmin=20,
smooth=335) and (nmin=20, lamfix), :func:`coverage` the n_lo, n_hi, wout and node_range of the
fw_disc_dumps.py outputs, and :func:`wavelength_rounding_correction` ``lamfix_dT10.npz``, all bit for
bit. Hardware: the bitwise agreement with the stored M424 products assumes the same numpy build and
CPU features as the production (numpy 1.26 computes ``np.log`` with the AVX512_SKX SVML routines on
Trillium; without AVX512, 3 % of the y = c ln(lambda / lref) values change by 1 ulp, so the products
agree only to ~1e-16 in float64 or 1 float32 ulp there). :meth:`FluxLibrary.save` records the
relevant CPU features in '_meta'.

Notes
-----
The M424 library_dT10.npz was written by fw_disc_los.py (library by-product of the exact disc sums,
blocks of 5000 models, 20 workers), not by fw_disc.library (chunks of 20000), although it is that
function's cache file. The two differ only in the row offsets of the interpolation (the block size):
``build(block=20000)`` (fw_disc.library's arithmetic) differs from the file in 198 of the 4.9 million
profile values of the filled bins, by 1 float32 ulp (245 of the 5.7 million values of all bins,
counting the copies in empty bins). ``build(block=5000)`` (default) reproduces the file.

PP 2026-10-01: ported from the project's fw_disc.py (library, lam_corrections, lib_nodes,
node_pairs), fw_disc_los.py (its library by-product), fw_disc_holdout.py (hold-out libraries) and
fw_disc_dumps.py (n_lo, n_hi, wout); see the provenance comments per function.
"""
import mmap
import multiprocessing
import os
import platform

import numpy as np
from scipy import sparse

from . import parallel as par
from .spectral import interp_rows, y_of_lam

LIBRARY_KEYS = ("edges", "tmean", "count", "prof", "fc", "dT")
"""Members of a flux-library file (legacy ``library_dT10.npz``; :meth:`FluxLibrary.save` adds '_meta')."""

NODE_KEYS = ("t", "count", "prof", "fc")
"""Keys of the legacy ``fw_disc.lib_nodes`` dict (:class:`LibraryNodes` supports ``nodes[key]``)."""


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------
def _grid_y(grid):
    """The velocity grid of a :class:`~ppmpy.synspec.spectral.VelocityGrid` or a 1-D array."""
    y = getattr(grid, "y", grid)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or y.size < 2:
        raise ValueError("grid must be a VelocityGrid or a 1-D array of velocities")
    return y


def _lref(lref):
    """Reference wavelengths of a :class:`~ppmpy.synspec.spectral.LineSet` or an array (n_lines,)."""
    lr = getattr(lref, "lref", lref)
    try:
        lr = np.atleast_1d(np.asarray(lr, dtype=np.float64))
    except (TypeError, ValueError):
        raise ValueError("lref must be a LineSet or numbers (one wavelength per line), got {!r}".format(lref)) from None
    if lr.ndim != 1:
        raise ValueError("lref must be 1-D (one wavelength per line)")
    return lr


def _line_names(lines):
    """Line names of a LineSet or a sequence of str."""
    names = getattr(lines, "names", lines)
    if isinstance(names, str):
        names = [names]
    return [str(n) for n in names]


def _as_array(a):
    """np.asarray for inputs without an ndim (lists, tuples); arrays and np.memmap objects are kept as they are
    (np.asarray would drop the memmap subclass, which the parallel build reopens by file name)."""
    return a if hasattr(a, "ndim") and hasattr(a, "shape") else np.asarray(a)


def _file_identity(path):
    """(st_dev, st_ino, st_size, st_mtime_ns) of a file: changes when the file is replaced or rewritten."""
    st = os.stat(path)
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def _cpu_features():
    """CPU features that decide numpy's SIMD code paths (np.log: AVX512_SKX SVML), for '_meta'."""
    feats = None
    for mod in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
        try:
            feats = __import__(mod, fromlist=["__cpu_features__"]).__cpu_features__
            break
        except (ImportError, AttributeError):
            continue
    out = dict(machine=platform.machine())
    if feats:
        out.update({k: bool(feats[k]) for k in ("AVX2", "FMA3", "AVX512F", "AVX512_SKX") if k in feats})
    return out


def teff_edges(teff, dT=10.0):
    """
    Bin edges of a T_eff' library.

    Parameters
    ----------
    teff: array-like
        T_eff' of all models [K].
    dT: float
        Bin width [K].

    Returns
    -------
    np.ndarray
        ``np.arange(np.floor(min / dT) * dT, max + dT, dT)`` (float64), as fw_disc.library.
    """
    # PP 2026-10-01: ported from fw_disc.py:70
    teff = np.asarray(teff)
    if teff.size == 0:
        raise ValueError("no models")
    return np.arange(np.floor(teff.min() / dT) * dT, teff.max() + dT, dT)


def teff_bins(teff, edges):
    """
    Library bin of each T_eff'.

    Parameters
    ----------
    teff: float or array-like
        T_eff' [K].
    edges: array-like
        (nb + 1,) increasing bin edges [K] (:func:`teff_edges`).

    Returns
    -------
    np.ndarray of int
        ``clip(digitize(teff, edges) - 1, 0, nb - 1)``, as fw_disc.library: values beyond the edges go to
        the end bins, and a value equal to the last edge to the last bin.
    """
    # PP 2026-10-01: ported from fw_disc.py:71
    edges = np.asarray(edges)
    return np.clip(np.digitize(teff, edges) - 1, 0, edges.size - 2)


def _selection(select, N):
    """A bool mask (N,) from select (None, bool mask or integer indices); None for 'all models'."""
    if select is None:
        return None
    s = np.asarray(select)
    if s.dtype == bool:
        if s.shape != (N,):
            raise ValueError("a bool select must have shape (N,) = ({},), got {}".format(N, s.shape))
        m = s.copy()
    elif s.dtype.kind in "iu" and s.ndim == 1:
        if s.size and (s.min() < 0 or s.max() >= N):
            raise ValueError("select indices must lie in 0 .. N - 1 = {}".format(N - 1))
        m = np.zeros(N, bool)
        m[s] = True
        if int(m.sum()) != s.size:
            raise ValueError("select indices must not repeat")
    else:
        raise ValueError("select must be None, a bool mask (N,) or 1-D integer indices, got dtype {} shape {}".format(
            s.dtype, s.shape))
    if not m.any():
        raise ValueError("select holds no models")
    return m


# ----------------------------------------------------------------------------------------------
# streaming the profiles: per-block bin sums (shared by the serial and the parallel build)
# ----------------------------------------------------------------------------------------------
class _BlockSums:
    """
    Per-bin sums of the interpolated profiles of one block of models.

    ``self(i0)`` returns (ub, P): ub = the sorted bins that the block's (selected) models fall into, and
    P (ub.size, nl, ny) float64 with P[u, j] = sum over those models of bin ub[u] (in model order,
    starting from 0) of profile j on the grid. Every model is interpolated with the row offsets of its
    position in the block (blocks start at multiples of ``block`` over all N models, selected or not).
    ``rows`` < block interpolates and accumulates ``rows`` models at a time (less memory; same numbers:
    same row offsets, same summation order).
    """

    def __init__(self, lam, fnorm, bins, nb, y, lref, block, rows, sel=None):
        self.lam, self.fnorm, self.bins, self.sel = lam, fnorm, bins, sel
        self.nb, self.y, self.lref = int(nb), y, lref
        self.block, self.rows = int(block), int(rows)
        self.n = bins.size

    def __call__(self, i0):
        # PP 2026-10-01: ported from fw_disc.py:80-85 (chunk loop of library()), fw_disc_los.py:73-83 and
        # fw_disc_holdout.py:59-68 (stream(): only the selected models' columns in the weight matrix); the legacy loops
        # run over the lines outside the chunks, which only changes the order in which independent sums are made
        i1 = min(i0 + self.block, self.n)
        nl, ny = self.lref.size, self.y.size
        bb = np.asarray(self.bins[i0:i1])
        loc = None
        if self.sel is not None:
            loc = np.flatnonzero(self.sel[i0:i1])
            if loc.size == 0:
                return np.zeros(0, np.intp), np.zeros((0, nl, ny))
            if loc.size == bb.size:
                loc = None
        L = np.asarray(self.lam[i0:i1])
        F = np.asarray(self.fnorm[i0:i1])
        if loc is not None:
            L, F, bb = L[loc], F[loc], bb[loc]
        ub, inv = np.unique(bb, return_inverse=True)
        P = np.zeros((ub.size, nl, ny))
        if self.rows >= bb.size:
            # legacy: sparse (bins x block) indicator matrix times the block's profiles (rows of the bins in use only;
            # every row's sum is made independently of the others)
            S = sparse.csr_matrix((np.ones(bb.size), (inv, np.arange(bb.size))), shape=(ub.size, bb.size))
            for j in range(nl):
                f = interp_rows(y_of_lam(L[:, j], self.lref[j]), F[:, j], self.y, row0=0 if loc is None else loc)
                P[:, j] = S @ f
        else:
            # same sums, `rows` models at a time: np.add.at adds model by model in order, starting from 0,
            # exactly as the CSR product (scipy csr_matvecs: y[b] += 1.0 * x[i] for i ascending)
            for j in range(nl):
                Pj = np.zeros((ub.size, ny))
                for r0 in range(0, bb.size, self.rows):
                    r1 = min(r0 + self.rows, bb.size)
                    f = interp_rows(y_of_lam(L[r0:r1, j], self.lref[j]), F[r0:r1, j], self.y,
                                    row0=r0 if loc is None else loc[r0:r1])
                    np.add.at(Pj, inv[r0:r1], f)
                P[:, j] = Pj
        return ub, P


def _share(a, start_method):
    """
    How a pool worker gets an input array. 'fork': the object itself (inherited, memory maps keep their
    mapping, nothing is pickled). Other start methods: a whole np.memmap is reopened by file name, with the
    file's identity (device, inode, size, mtime) recorded here and checked in the worker; anything else is
    pickled into every worker.
    """
    if start_method != "fork" and isinstance(a, np.memmap) and isinstance(getattr(a, "base", None), mmap.mmap) \
            and a.filename:
        order = "F" if (a.flags.f_contiguous and not a.flags.c_contiguous) else "C"
        try:
            ident = _file_identity(a.filename)
        except OSError as e:
            raise ValueError("the memory-mapped file {} cannot be reopened by the workers ({}); use start_method "
                             "'fork' or pass an array".format(a.filename, e)) from None
        return ("memmap", a.filename, int(a.offset), a.dtype, a.shape, order, ident)
    return ("array", a)


def _unshare(s):
    """The array of a :func:`_share` record (in a worker); raises when a reopened file is not the parent's."""
    if s[0] != "memmap":
        return s[1]
    _, filename, offset, dtype, shape, order, ident = s
    with open(filename, "rb") as f:
        st = os.fstat(f.fileno())
        now = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        if now != ident:
            raise RuntimeError("{} changed since the build started (device, inode, size, mtime {} -> {}): the workers "
                               "would read other data than the parent".format(filename, ident, now))
        return np.memmap(f, dtype=dtype, mode="r", shape=shape, order=order, offset=offset)


def _pool_init(spec):
    """Pool initializer (:func:`ppmpy.synspec.parallel.make_pool`): the worker's own _BlockSums (fork and spawn)."""
    return _BlockSums(lam=_unshare(spec["lam"]), fnorm=_unshare(spec["fnorm"]), bins=spec["bins"], nb=spec["nb"],
                      y=spec["y"], lref=spec["lref"], block=spec["block"], rows=spec["rows"], sel=spec["sel"])


def _pool_block(i0):
    return par.worker_state()(i0)


def _ordered_block_sums(summer, starts, nproc, start_method, maxtasksperchild, spec, timeout):
    """Yield the block sums in block order: serial, or from a pool via imap (which keeps the order) with a watchdog."""
    if nproc <= 1:
        for i0 in starts:
            yield summer(i0)
        return
    with par.make_pool(nproc, initializer=_pool_init, initargs=(spec,), maxtasksperchild=maxtasksperchild,
                       start_method=start_method) as pool:
        it = pool.imap(_pool_block, starts)
        for k in range(len(starts)):
            try:
                res = it.next(timeout=timeout)
            except multiprocessing.TimeoutError:
                # a worker killed by a signal (out of memory, CPU-time limit) never returns its block
                raise par.PoolStalled(starts[k:], timeout) from None
            yield res


def _accumulate(teff, lam, fnorm, fcont0, y, lref, edges, block=5000, stride=1, rows=1000, nproc=1, sel=None,
                start_method=None, maxtasksperchild=None, timeout=900.0, progress=None):
    """
    The float64 bin sums of :meth:`FluxLibrary.build` before the division (inputs already checked).

    Returns
    -------
    dict
        cnt (nb,) float64 (models per bin), tsum (nb,) (sum of T_eff'), fcs (nb, nl) (sum of F_c), sums (nb, nl, ny)
        (sum of the profiles on the grid), all over the selected models.
    """
    # PP 2026-10-01: ported from fw_disc.py:70-85, fw_disc_los.py:63-66,73-83,131-132 and fw_disc_holdout.py:52,83-85,
    # 108-109,120 (bin counts and sums; blocks strided over the partial sums of the workers)
    N, nl = teff.size, lref.size
    nb = edges.size - 1
    b = teff_bins(teff, edges)
    bs, ts = (b, teff) if sel is None else (b[sel], teff[sel])
    cnt = np.bincount(bs, minlength=nb).astype(float)
    tsum = np.bincount(bs, weights=ts, minlength=nb)
    fcs = np.zeros((nb, nl))
    for j in range(nl):
        fcs[:, j] = np.bincount(bs, weights=fcont0[:, j] if sel is None else fcont0[sel, j], minlength=nb)
    del bs, ts

    starts = list(range(0, N, block))                   # over all N models: the row offsets of the full run
    acc = [np.zeros((nb, nl, y.size)) for _ in range(min(stride, len(starts)))]
    nproc = min(nproc, len(starts))
    summer, spec = None, None
    if nproc <= 1:
        summer = _BlockSums(lam, fnorm, b, nb, y, lref, block, rows, sel=sel)
    else:
        method = par.get_context(start_method).get_start_method()
        par.login_node_warning(nproc)
        spec = dict(lam=_share(lam, method), fnorm=_share(fnorm, method), bins=b, nb=nb, y=y, lref=lref, block=block,
                    rows=rows, sel=sel)
        start_method = method
    for k, (ub, P) in enumerate(_ordered_block_sums(summer, starts, nproc, start_method, maxtasksperchild, spec,
                                                    timeout)):
        # fw_disc.library: prof[:, j] += S @ f; fw_disc_los: A += blk @ P in worker k % nproc. Bins a block does not
        # touch would get + 0.0, which changes nothing.
        acc[k % len(acc)][ub] += P
        if progress is not None:
            progress(k + 1, len(starts))
    sums = acc[0]
    for A in acc[1:]:                                   # fw_disc_los: sum(pool.map(stream, range(nproc)))
        sums += A
    return dict(cnt=cnt, tsum=tsum, fcs=fcs, sums=sums)


# ----------------------------------------------------------------------------------------------
# the flux library
# ----------------------------------------------------------------------------------------------
class FluxLibrary:
    """
    Mean rest-frame flux profile and continuum flux per T_eff' bin.

    Parameters
    ----------
    edges: np.ndarray
        (nb + 1,) bin edges [K].
    tmean: np.ndarray
        (nb,) mean T_eff' of each bin's models [K] (bin centre for empty bins; 0 with ``fill_empty=False``).
    count: np.ndarray
        (nb,) number of models per bin (float64, as the legacy file).
    prof: np.ndarray
        (nb, nl, ny) mean continuum-normalised profile on the velocity grid (float32 in the legacy file;
        float64 from ``build(prof_dtype=np.float64)``).
    fc: np.ndarray
        (nb, nl) float64 mean continuum flux F_c (first frequency point of each line's band).
    dT: float
        Bin width [K].
    meta: dict, optional
        Provenance record of the file the library was read from ({} for the legacy file or a new build).
    params, inputs: dict, optional
        Build parameters and input files (name -> path), written to '_meta' by :meth:`save`.

    Notes
    -----
    ``lib[key]`` returns the arrays by their legacy names (:data:`LIBRARY_KEYS`), so a FluxLibrary can be
    passed wherever the legacy code took the dict of fw_disc.library or the NpzFile of library_dT10.npz.
    """

    def __init__(self, edges, tmean, count, prof, fc, dT, meta=None, params=None, inputs=None):
        self.edges = np.asarray(edges, dtype=np.float64)
        self.tmean = np.asarray(tmean, dtype=np.float64)
        self.count = np.asarray(count, dtype=np.float64)
        self.prof = np.asarray(prof)
        self.fc = np.asarray(fc, dtype=np.float64)
        self.dT = float(dT)
        self.meta = dict(meta or {})
        self.params = dict(params or {})
        self.inputs = dict(inputs or {})
        nb = self.edges.size - 1
        if not (self.tmean.shape == self.count.shape == (nb,) and self.prof.ndim == 3 and self.prof.shape[0] == nb
                and self.fc.shape == self.prof.shape[:2]):
            raise ValueError("inconsistent library shapes: edges {}, tmean {}, count {}, prof {}, fc {}".format(
                self.edges.shape, self.tmean.shape, self.count.shape, self.prof.shape, self.fc.shape))

    # -- shape and access ----------------------------------------------------------------------
    @property
    def nb(self):
        """Number of bins."""
        return self.count.size

    @property
    def nl(self):
        """Number of lines."""
        return self.prof.shape[1]

    @property
    def ny(self):
        """Number of velocity-grid points."""
        return self.prof.shape[2]

    @property
    def centres(self):
        """Bin centres [K]."""
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def filled(self):
        """Bins that hold models (count > 0)."""
        return self.count > 0

    def bin_index(self, teff):
        """Library bin of each T_eff' (:func:`teff_bins`)."""
        return teff_bins(teff, self.edges)

    def __getitem__(self, key):
        if key not in LIBRARY_KEYS:
            raise KeyError(key)
        return np.float64(self.dT) if key == "dT" else getattr(self, key)

    def keys(self):
        return list(LIBRARY_KEYS)

    def as_dict(self):
        """The legacy dict (edges, tmean, count, prof, fc, dT); arrays are not copied."""
        return {k: self[k] for k in LIBRARY_KEYS}

    def __repr__(self):
        return "FluxLibrary(nb={}, filled={}, models={:.0f}, nl={}, ny={}, dT={:g} K, T {:.0f}-{:.0f} K)".format(
            self.nb, int(self.filled.sum()), self.count.sum(), self.nl, self.ny, self.dT, self.edges[0], self.edges[-1])

    # -- files ---------------------------------------------------------------------------------
    @classmethod
    def load(cls, path):
        """
        Read a library file (legacy library_dT10.npz, or one written by :meth:`save`).

        Parameters
        ----------
        path: str or os.PathLike
            The .npz (members edges, tmean, count, prof, fc, dT; '_meta' optional).

        Returns
        -------
        FluxLibrary
            ``meta`` holds the file's '_meta' record ({} for the legacy file), ``params`` its build
            parameters (meta['params'], or {}), ``inputs`` {'source': path}.

        Raises
        ------
        ValueError
            If a member is missing.
        """
        from .io import read_meta
        path = os.fspath(path)
        with np.load(path) as z:
            missing = [k for k in LIBRARY_KEYS if k not in z.files]
            if missing:
                raise ValueError("{} is not a flux library (missing {})".format(path, missing))
            meta = read_meta(z)
            return cls(z["edges"], z["tmean"], z["count"], z["prof"], z["fc"], float(z["dT"]), meta=meta,
                       params=meta.get("params", {}), inputs=dict(source=path))

    def save(self, path, meta=None):
        """
        Write the library atomically as an uncompressed .npz with the legacy members (edges, tmean,
        count, prof, fc, dT 0-d float64) plus '_meta'.

        Parameters
        ----------
        path: str or os.PathLike
            Target .npz.
        meta: dict, optional
            Provenance record. Default: :func:`ppmpy.synspec.io.make_meta` ('synspec.flux_library') with
            ``params``, ``inputs`` (recorded with size and mtime), the CPU features that decide numpy's
            SIMD paths ('cpu'), and, for a library read from a file with a '_meta' record, that record
            as 'source_meta' (so load -> save keeps the provenance).

        Returns
        -------
        str
            path.
        """
        from .io import make_meta, save_npz
        if meta is None:
            extra = dict(cpu=_cpu_features())
            if self.meta:
                extra["source_meta"] = self.meta
            meta = make_meta("synspec.flux_library", params=self.params, inputs=self.inputs, **extra)
        arrays = dict(edges=self.edges, tmean=self.tmean, count=self.count, prof=self.prof, fc=self.fc,
                      dT=np.float64(self.dT))
        return save_npz(os.fspath(path), arrays, meta=meta)

    # -- building --------------------------------------------------------------------------------
    @classmethod
    def build(cls, teff, lam, fnorm, fcont0, grid, lref, dT=10.0, block=5000, stride=1, nproc=1, rows=1000,
              edges=None, select=None, prof_dtype=np.float32, fill_empty=True, start_method=None,
              maxtasksperchild=None, timeout=900.0, progress=None):
        """
        Bin means of the profiles of a per-point run (port of fw_disc.library, of the library by-product
        of fw_disc_los.py, which wrote the M424 library_dT10.npz, and of the hold-out libraries of
        fw_disc_holdout.py).

        Parameters
        ----------
        teff: array-like
            (N,) T_eff' of every model [K].
        lam, fnorm: array-like
            (N, nl, nrow) wavelengths [A, air; increasing along the last axis] and continuum-normalised
            flux of every model and line (float32 in the M424 product). Memory maps are read block by
            block (:func:`ppmpy.synspec.io.npz_member_memmap` of profiles.npz 'lam', 'fnorm').
        fcont0: array-like
            (N, nl) continuum flux of every model and line at the first frequency point
            (``fcont[:, :, 0]``).
        grid: VelocityGrid or np.ndarray
            Common velocity grid y [km/s].
        lref: LineSet or array-like
            (nl,) velocity zero points [A].
        dT: float
            Bin width [K].
        block: int
            Models per block (over all N models, from model 0). Each block's models are interpolated with
            the row offsets of their position in the block (:func:`ppmpy.synspec.spectral.interp_rows`) and
            summed per bin in model order, starting from 0. 5000 (fw_disc_los.py and fw_disc_holdout.py;
            the M424 library_dT10.npz) reproduces that file bit for bit; 20000 is fw_disc.library's chunk
            (that function's numbers, bit for bit; on M424, 198 prof values of filled bins then differ from
            the file by 1 float32 ulp). Other values change the float64 sums in the last bits, and prof
            by at most 1 float32 ulp; edges, tmean, count, fc never change.
        stride: int
            Number of interleaved partial sums: block k is added to partial sum k mod stride, and the
            partial sums are added in order. 1 = fw_disc.library (all blocks in order); P = fw_disc_los.py
            or fw_disc_holdout.py run with --nproc P (worker w summed blocks w, w + P, ...; the M424
            library_dT10.npz: 20). Changes the float64 sums in the last bits; the M424 float32 prof
            happens to be the same for every stride tried (1-40, 48, 64). Memory: stride x (nb, nl, ny)
            float64.
        nproc: int
            Worker processes. Workers return the per-block sums, which are added in block order, so the
            result does not depend on nproc, bit for bit (also in float64).
        rows: int or None
            Models interpolated at a time inside a block (memory and speed). Does not change the result
            (same row offsets and summation order). rows >= block (or None) runs the legacy code
            (one interpolation per block and line, sparse indicator-matrix product); smaller values
            accumulate with np.add.at. Default 1000 (see Notes).
        edges: array-like, optional
            Bin edges to use instead of ``teff_edges(teff, dT)`` (all N models, also with ``select``);
            T_eff' beyond them goes to the end bins. Their spacing must be dT (rtol 1e-9).
        select: array-like, optional
            Bool mask (N,) or integer indices of the models to use (default all), e.g. one half of the
            models for a hold-out test. Unselected models are skipped in every bin sum but keep their row
            positions, so blocks and interpolation offsets stay those of the full run (as
            fw_disc_holdout.py; fancy-indexing the inputs instead would copy them and change the
            offsets). Only the selected rows are interpolated.
        prof_dtype: dtype
            np.float32 (default; the legacy files) or np.float64 (the bin means before the cast, e.g.
            the hold-out libraries of fw_disc_holdout.py).
        fill_empty: bool
            True (default, fw_disc.library and fw_disc_los.py): empty bins get the bin centre as tmean
            and the profile and F_c of the nearest filled bin. False (fw_disc_holdout.py): empty bins
            keep tmean = 0 and zero profile and F_c.
        start_method: str, optional
            'fork', 'spawn' or 'forkserver' for nproc > 1 (default :func:`ppmpy.synspec.parallel.get_context`:
            PPMPY_SYNSPEC_START_METHOD, else 'fork' on Linux). With 'fork' the workers inherit the inputs;
            otherwise whole memory maps are reopened by file name (a file replaced since the start of the
            build raises :class:`~ppmpy.synspec.parallel.WorkerInitError`) and other arrays are pickled into
            every worker.
        maxtasksperchild: int, optional
            Blocks per worker process before it is replaced (CPU-time limits).
        timeout: float or None
            Raise :class:`~ppmpy.synspec.parallel.PoolStalled` (``missing`` = the block starts not yet
            delivered) when the next block in order has not arrived for this long [s] (a worker killed by
            the out-of-memory killer or a CPU-time limit loses its block); None waits for ever. Must exceed
            the worker start-up plus the longest block (M424: a few s per block of 5000 with a warm page
            cache, ~15 s cold).
        progress: callable, optional
            Called as progress(blocks_done, blocks_total) after every block.

        Returns
        -------
        FluxLibrary

        Notes
        -----
        Memory and time: the inputs are read one block at a time ((block, nl, nrow) of lam and fnorm);
        interpolating ``rows`` models needs ~10 temporaries of (rows x ny) x 8 bytes: peak RSS ~0.7 GB for
        rows = 1000 and ~9.6 GB for rows = 20000 on the M424 grid (ny = 5401). The sums are stride x (nb, nl,
        ny) float64 during the build (45 MB each for M424), plus one block sum per worker in flight, which
        holds only the bins the block touches (M424, blocks of 5000: 158-212 of 351 bins, i.e. ~25 MB per
        block instead of 45 MB through the pool); memory maps add their file pages to the RSS.

        Timing (Trillium login node, 2026-10-01): the whole M424 build (1.24 million models, default block
        and rows) takes 10 min serially (597 s CPU) and 2.7 min with nproc = 4 (spawn); nproc = 8 (fork)
        130 s, workers peaking at ~1.2-1.7 GB (with file pages); at a load average of 76 the serial and the
        nproc = 4 builds took 16 and 6 min. These times assume a warm page cache:
        reading the 4.8 GB of lam and fnorm cold from /scratch made an nproc = 8 build take 412 s instead
        of 114 s. Large temporaries are slow where page faults are expensive: with transparent huge pages
        'always' and fragmented memory (these login nodes) each large allocation can trigger direct
        compaction, mostly system time (interp_rows of 800 rows: 1.8 s per call instead of 0.18 s; 20000
        models took 118 s with rows = 5000, 110 s of it system time, and ~5 min with rows = 20000). Keep
        rows <= ~1000, or disable THP for the process before the build (Linux:
        ``ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)``, PR_SET_THP_DISABLE, inherited by the workers).

        Conventions / legacy quirks: each line's continuum flux is that of the first frequency point of
        its band; the mean profile of an empty bin is copied from the nearest filled bin (the lower one on
        a tie), its tmean is the bin centre and its count 0; prof is stored as float32 (bin means in
        float64, cast once).
        """
        # PP 2026-10-01: ported from fw_disc.py:64-101 (library), fw_disc_los.py:63-66,73-83,132-155 (library
        # by-product: blocks of 5000, strided over the workers) and fw_disc_holdout.py:50-52,83-85,117-120 (hold-out
        # libraries); the cache in RUN/ is gone (the caller saves)
        y = _grid_y(grid)
        lr = _lref(lref)
        nl = lr.size
        teff, lam, fnorm, fcont0 = (_as_array(a) for a in (teff, lam, fnorm, fcont0))
        N = teff.size
        if N == 0:
            raise ValueError("no models")
        if teff.shape != (N,):
            raise ValueError("teff must be 1-D")
        for name, a in (("lam", lam), ("fnorm", fnorm)):
            if a.ndim != 3 or a.shape[:2] != (N, nl):
                raise ValueError("{} must have shape (N, nl, nrow) = ({}, {}, ...), got {}".format(
                    name, N, nl, a.shape))
        if lam.shape != fnorm.shape:
            raise ValueError("lam and fnorm differ in shape: {} vs {}".format(lam.shape, fnorm.shape))
        if fcont0.shape != (N, nl):
            raise ValueError("fcont0 must have shape (N, nl) = ({}, {}), got {}".format(N, nl, fcont0.shape))
        block = int(block)
        if block < 1:
            raise ValueError("block must be >= 1")
        rows = block if rows is None else min(int(rows), block)
        if rows < 1:
            raise ValueError("rows must be >= 1")
        stride = int(stride)
        if stride < 1:
            raise ValueError("stride must be >= 1")
        nproc = max(1, int(nproc))
        prof_dtype = np.dtype(prof_dtype)
        if prof_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("prof_dtype must be float32 or float64, got {}".format(prof_dtype))
        sel = _selection(select, N)

        edges_given = edges is not None
        edges = teff_edges(teff, dT) if edges is None else np.asarray(edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2 or not np.allclose(np.diff(edges), dT, rtol=1e-9, atol=0.0):
            raise ValueError("edges must have at least 2 entries spaced by dT = {:g}".format(dT))

        acc = _accumulate(teff, lam, fnorm, fcont0, y, lr, edges, block=block, stride=stride, rows=rows, nproc=nproc,
                          sel=sel, start_method=start_method, maxtasksperchild=maxtasksperchild, timeout=timeout,
                          progress=progress)
        cnt, tsum, fcs, prof = acc["cnt"], acc["tsum"], acc["fcs"], acc["sums"]
        ok = cnt > 0
        prof[ok] /= cnt[ok, None, None]
        fcs[ok] /= cnt[ok, None]
        if fill_empty:
            tmean = np.where(ok, tsum / np.maximum(cnt, 1), 0.5 * (edges[:-1] + edges[1:]))
        else:
            tmean = tsum / np.maximum(cnt, 1)           # fw_disc_holdout.py:84: 0 for empty bins
        prof = prof.astype(prof_dtype, copy=False)
        # empty bins (sparse tails): take the nearest filled bin
        if fill_empty and not ok.all():
            filled = np.where(ok)[0]
            for i in np.where(~ok)[0]:
                k = filled[np.argmin(np.abs(filled - i))]
                prof[i], fcs[i] = prof[k], fcs[k]
        params = dict(dT=float(dT), block=block, stride=stride, rows=rows, n_models=int(N),
                      n_selected=int(N if sel is None else sel.sum()), select=sel is not None, lref=lr.tolist(),
                      ny=int(y.size), y0=float(y[0]), y1=float(y[-1]), edges_given=edges_given,
                      prof_dtype=prof_dtype.name, fill_empty=bool(fill_empty))
        return cls(edges, tmean, cnt, prof, fcs, dT, params=params)

    @classmethod
    def from_profiles_npz(cls, path, grid, lref, dT=10.0, **build_kw):
        """
        :meth:`build` from a merged per-point product (M424: ``profiles.npz`` with teff, lam, fnorm,
        fcont), reading lam and fnorm through memory maps and fcont[:, :, 0] block by block.

        Parameters
        ----------
        path: str or os.PathLike
            Uncompressed .npz with members teff (N,), lam, fnorm, fcont (N, nl, nrow).
        grid, lref, dT:
            As :meth:`build` (lref: one entry per line of the file).
        **build_kw:
            Further :meth:`build` arguments (block, nproc, rows, select, prof_dtype, ...).

        Returns
        -------
        FluxLibrary
            ``inputs`` = {'profiles': path}.

        Raises
        ------
        RuntimeError
            If the file was replaced or rewritten while the library was built from it.
        """
        from .io import npz_member_memmap
        path = os.fspath(path)
        ident = _file_identity(path)
        teff = np.array(npz_member_memmap(path, "teff"))
        lam = npz_member_memmap(path, "lam")
        fnorm = npz_member_memmap(path, "fnorm")
        fcont = npz_member_memmap(path, "fcont")
        N, nl = fcont.shape[:2]
        fcont0 = np.empty((N, nl), dtype=fcont.dtype)
        step = int(build_kw.get("block", 5000))
        for i0 in range(0, N, step):
            fcont0[i0:i0 + step] = fcont[i0:i0 + step, :, 0]
        del fcont
        lib = cls.build(teff, lam, fnorm, fcont0, grid, lref, dT=dT, **build_kw)
        if _file_identity(path) != ident:
            raise RuntimeError("{} was replaced or rewritten during the build; build again".format(path))
        lib.inputs = dict(profiles=path)
        return lib


# ----------------------------------------------------------------------------------------------
# interpolation nodes
# ----------------------------------------------------------------------------------------------
class LibraryNodes:
    """
    T_eff' interpolation nodes of a flux library (:func:`lib_nodes`).

    Attributes
    ----------
    t: np.ndarray
        (nn,) node T_eff' [K], increasing (count-weighted mean of the bins' tmean).
    count: np.ndarray
        (nn,) number of models per node.
    prof: np.ndarray
        (nn, nl, ny) float64 node profile (count-weighted mean of the bins').
    fc: np.ndarray
        (nn, nl) node continuum flux.
    groups: list of np.ndarray
        Library bins of each node.
    params: dict
        nmin, smooth, corr (whether a correction was added).

    Notes
    -----
    ``nodes[key]`` returns t, count, prof, fc (the legacy dict of fw_disc.lib_nodes, as taken by
    fw_disc.DiscFlux).
    """

    def __init__(self, t, count, prof, fc, groups=None, params=None):
        self.t = np.asarray(t, dtype=np.float64)
        self.count = np.asarray(count, dtype=np.float64)
        self.prof = np.asarray(prof)
        self.fc = np.asarray(fc)
        self.groups = list(groups) if groups is not None else []
        self.params = dict(params or {})

    @property
    def nn(self):
        """Number of nodes."""
        return self.t.size

    def __getitem__(self, key):
        if key not in NODE_KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def keys(self):
        return list(NODE_KEYS)

    def as_dict(self):
        """The legacy dict (t, count, prof, fc); arrays are not copied."""
        return {k: getattr(self, k) for k in NODE_KEYS}

    def pairs(self, teff, mode="clamp"):
        """:func:`node_pairs` with these nodes."""
        return node_pairs(self.t, teff, mode=mode)

    def coverage(self, teff, mu=None):
        """:func:`coverage` with these nodes."""
        return coverage(self.t, teff, mu=mu)

    def __repr__(self):
        return "LibraryNodes(nn={}, T {:.0f}-{:.0f} K, models={:.0f}, {})".format(
            self.nn, self.t[0], self.t[-1], self.count.sum(), self.params)


def lib_nodes(lib, nmin=20, smooth=0.0, corr=None):
    """
    T_eff' interpolation nodes from a flux library (port of fw_disc.lib_nodes).

    Consecutive filled bins are merged until a node holds >= nmin models (only the sparse tails are
    affected); a leftover at the hot end joins the last node. Node T_eff' = mean T_eff' of its models;
    node profile and F_c = means over its models.

    Parameters
    ----------
    lib: FluxLibrary or mapping
        Flux library (:class:`FluxLibrary`, or the legacy dict / NpzFile with count, tmean, prof, fc).
    nmin: int
        Minimum number of models per node.
    smooth: float
        W > 0 [K]: replace every node's profile and F_c by a local-linear fit in T_eff' over the nodes
        within +-W/2 (weights = model counts); nodes with fewer than 3 neighbours, or a degenerate fit,
        keep their values. W = 335 K (one period of the M424 EW(T_eff') sawtooth of the continuum
        sampling) removes the sawtooth while keeping linear trends.
    corr: np.ndarray, optional
        (nb, nl, ny) additive per-bin profile correction, added to the bin profiles (in float64) before
        merging, e.g. :func:`wavelength_rounding_correction` (the legacy ``lamfix=True``).

    Returns
    -------
    LibraryNodes

    Validation
    ----------
    M424 library_dT10.npz, nmin=20: 245 nodes, 35829-38891 K; equal bit for bit to the frozen legacy
    code for (nmin=20), (nmin=20, smooth=335) and (nmin=20, corr=lamfix) (test_library.py).
    """
    # PP 2026-10-01: ported from fw_disc.py:299-349 (lib_nodes); lamfix=True -> corr=lam_corrections(lib)
    cnt = np.asarray(lib["count"])
    tmean = np.asarray(lib["tmean"])
    bfc = np.asarray(lib["fc"])
    bprof = np.asarray(lib["prof"]).astype(np.float64)
    if corr is not None:
        corr = np.asarray(corr)
        if corr.shape != bprof.shape:
            raise ValueError("corr must have the shape of the library profiles {}, got {}".format(
                bprof.shape, corr.shape))
        bprof = bprof + corr
    groups, cur, c = [], [], 0.0
    for b in np.where(cnt > 0)[0]:
        cur.append(b)
        c += cnt[b]
        if c >= nmin:
            groups.append(cur)
            cur, c = [], 0.0
    if cur:
        if groups:
            groups[-1] = groups[-1] + cur
        else:
            groups.append(cur)
    if not groups:
        raise ValueError("the library holds no models")
    nn = len(groups)
    nl, ny = bprof.shape[1:]
    t, n, fc = np.zeros(nn), np.zeros(nn), np.zeros((nn, nl))
    prof = np.zeros((nn, nl, ny))
    for i, g in enumerate(groups):
        w = cnt[g]
        n[i] = w.sum()
        t[i] = np.sum(w * tmean[g]) / n[i]
        prof[i] = np.tensordot(w, bprof[g], axes=1) / n[i]
        fc[i] = w @ bfc[g] / n[i]
    if smooth > 0:
        ps, fs_ = np.empty_like(prof), np.empty_like(fc)
        for i in range(nn):
            k = np.where(np.abs(t - t[i]) <= smooth / 2)[0]
            w, dt = n[k], t[k] - t[i]
            S0, S1, S2 = w.sum(), w @ dt, w @ dt ** 2
            det = S0 * S2 - S1 ** 2
            if k.size < 3 or det <= 1e-12 * S0 * S2:
                ps[i], fs_[i] = prof[i], fc[i]
                continue
            c0, c1 = S2 / det, -S1 / det                 # local-linear value at t_i: sum_k (c0 + c1 dt_k) w_k y_k
            wk = w * (c0 + c1 * dt)
            ps[i] = np.tensordot(wk, prof[k], axes=1)
            fs_[i] = wk @ fc[k]
        prof, fc = ps, fs_
    params = dict(nmin=nmin, smooth=float(smooth), corr=corr is not None)
    return LibraryNodes(t, n, prof, fc, groups=[np.asarray(g, dtype=np.int64) for g in groups], params=params)


def node_pairs(tn, teff, mode="clamp"):
    """
    Linear interpolation in T_eff' between nodes (port of fw_disc.node_pairs).

    Parameters
    ----------
    tn: array-like
        (nn,) node T_eff', increasing, nn >= 2.
    teff: float or array-like
        T_eff' to interpolate to (evaluated in float64, as the legacy driver's
        ``smp['teff'].astype(np.float64)``).
    mode: str
        'clamp' (default, legacy): beyond the end nodes the end node is used (a in [0, 1]).
        'extrapolate': linear extrapolation from the end segments (a < 0 below the first node, > 1
        above the last; the weights 1 - a, a can be negative). Identical to 'clamp' inside the range.

    Returns
    -------
    k0, k1: np.ndarray of int
        Lower and upper node (k1 = k0 + 1).
    a: np.ndarray
        Weight of k1 (the value is (1 - a) y[k0] + a y[k1]).
    """
    # PP 2026-10-01: ported from fw_disc.py:352-357; mode='extrapolate' is new. The float64 cast changes nothing for
    # arrays (float32 - float64 array is computed in float64 anyway)
    tn = np.asarray(tn, dtype=np.float64)
    teff = np.asarray(teff, dtype=np.float64)
    if tn.ndim != 1 or tn.size < 2:
        raise ValueError("need at least 2 nodes")
    if mode not in ("clamp", "extrapolate"):
        raise ValueError("mode must be 'clamp' or 'extrapolate', got {!r}".format(mode))
    k0 = np.clip(np.searchsorted(tn, teff, side="right") - 1, 0, tn.size - 2)
    a = (teff - tn[k0]) / (tn[k0 + 1] - tn[k0])
    if mode == "clamp":
        a = np.clip(a, 0.0, 1.0)
    return k0, k0 + 1, a


def coverage(tn, teff, mu=None):
    """
    Points outside the node range and their share of the visible weight (as fw_disc_dumps.py).

    Parameters
    ----------
    tn: array-like
        (nn,) node T_eff', increasing.
    teff: array-like
        (N,) T_eff' of the points; compared in float64 (as the legacy driver: the per-dump samples store
        float32, and under NumPy 1.x a float32 array compared with a float64 scalar is compared in float32).
    mu: array-like, optional
        (N,) or (nlos, N) mu = r.n of the points for one or several lines of sight.

    Returns
    -------
    n_lo, n_hi: int
        Points below the first / above the last node.
    wout: float, np.ndarray or None
        Fraction of the visible weight sum(mu, mu > 0) carried by those points, per line of sight
        (float for 1-D mu, (nlos,) for 2-D, None without mu).
    """
    # PP 2026-10-01: ported from fw_disc_dumps.py:85,95-96 (n_lo, n_hi, wout; teff.astype(np.float64) of :84)
    tn = np.asarray(tn, dtype=np.float64)
    teff = np.asarray(teff, dtype=np.float64)
    lo, hi = teff < tn[0], teff > tn[-1]
    n_lo, n_hi = int(lo.sum()), int(hi.sum())
    if mu is None:
        return n_lo, n_hi, None
    m = np.asarray(mu)
    one = m.ndim == 1
    m2 = m[None] if one else m
    if m2.ndim != 2 or m2.shape[1] != teff.size:
        raise ValueError("mu must have shape (N,) or (nlos, N) with N = {}, got {}".format(teff.size, m.shape))
    out = lo | hi
    wout = np.zeros(m2.shape[0])
    for k in range(m2.shape[0]):
        mk = m2[k]
        vis = mk > 0
        wout[k] = mk[vis & out].sum() / mk[vis].sum()
    return n_lo, n_hi, (float(wout[0]) if one else wout)


# ----------------------------------------------------------------------------------------------
# wavelength rounding of the OUT files
# ----------------------------------------------------------------------------------------------
def representative_dirs(reps, nb, runs=None, layout="P{idx:06d}/P{idx:06d}", edges=None):
    """
    Model directory of each library bin's representative model.

    Parameters
    ----------
    reps: str, os.PathLike or array-like
        * path of an .npz with 'idx_rep' (nb,) (the intensity library, M424
          /scratch/ppathak/fastwind_imu/imu_library_dT10.npz; with ``edges``, its 'edges' must be there
          and equal them);
        * (nb,) integer model indices;
        * (nb,) directories (str, or None for bins without one).
    nb: int
        Number of library bins.
    runs: str or os.PathLike, optional
        Directory holding the model directories (M424 /scratch/ppathak/fastwind_imu/runs); needed for
        model indices.
    layout: str
        Model directory below ``runs`` for model index idx (M424: 'P{idx:06d}/P{idx:06d}').
    edges: array-like, optional
        Library edges to check an .npz's 'edges' against (np.allclose, as the legacy assert).

    Returns
    -------
    list
        nb directories (or None).

    Raises
    ------
    ValueError
        If the .npz has no 'idx_rep', or ``edges`` is given and its 'edges' are missing or differ; if
        the number of representatives is not nb; for a non-integer index or a missing ``runs``.
    """
    # PP 2026-10-01: ported from fw_disc.py:280-283 (the model directory of each bin in lam_corrections)
    if isinstance(reps, (str, os.PathLike)):
        path = os.fspath(reps)
        with np.load(path) as z:
            if edges is not None:
                if "edges" not in z.files:
                    raise ValueError("{} has no 'edges' to check against the library's".format(path))
                e = z["edges"]
                if e.shape != np.shape(edges) or not np.allclose(e, edges):
                    raise ValueError("the bins of {} differ from the library's".format(path))
            if "idx_rep" not in z.files:
                raise ValueError("{} has no 'idx_rep' (representative model of each bin)".format(path))
            reps = z["idx_rep"]
    reps = list(reps) if not isinstance(reps, np.ndarray) else reps
    if len(reps) != nb:
        raise ValueError("need one representative per bin ({}), got {}".format(nb, len(reps)))
    out = []
    for r in reps:
        if r is None or isinstance(r, (str, os.PathLike)):
            out.append(None if r is None else os.fspath(r))
            continue
        if isinstance(r, (bool, np.bool_)) or not float(r).is_integer():
            raise ValueError("model index must be an integer, got {!r}".format(r))
        if runs is None:
            raise ValueError("runs (directory of the model directories) is needed for model indices")
        out.append(os.path.join(os.fspath(runs), layout.format(idx=int(r))))
    return out


def wavelength_rounding_correction(lib, reps, lines, grid, lref=None, runs=None, suffix="VTV010", tol=0.0051,
                                   layout="P{idx:06d}/P{idx:06d}"):
    """
    Per-bin correction of a flux library for the 0.01 A rounding of the wavelengths in FASTWIND's
    OUT files (port of fw_disc.lam_corrections; the 'lamfix' variant).

    FASTWIND's OUT.<line> files print lambda to 0.01 A only; profiles.npz and the flux library were built
    from them. The modified pformalsol also writes OUT_IMU.<line> with the precise wavelengths of the same
    frequency points. For each filled bin, the bin's representative model gives
    corr = F_OUT placed at the precise wavelengths - F_OUT placed at the rounded wavelengths (both
    interpolated linearly onto the grid). The frequency grid is the same for all models of a bin (the M424
    core points agree to <= 0.001 A), so every model of the bin shares this error; add corr to the bin
    profiles (``lib_nodes(lib, corr=corr)``).

    Parameters
    ----------
    lib: FluxLibrary or mapping
        Flux library (count, edges, prof).
    reps: str, os.PathLike or array-like
        Representative model of each bin (:func:`representative_dirs`), e.g. the intensity library's
        .npz (its 'idx_rep'; its 'edges' must equal the library's) with ``runs``.
    lines: LineSet or sequence of str
        Line names of the OUT files (e.g. 'HEI4026'), in the order of the library's lines.
    grid: VelocityGrid or np.ndarray
        The library's velocity grid y [km/s].
    lref: array-like, optional
        (nl,) velocity zero points [A]; required unless ``lines`` is a LineSet (default ``lines.lref``).
    runs: str, optional
        Directory of the model directories (for model indices).
    suffix: str
        File suffix: OUT.<line>_<suffix>, OUT_IMU.<line>_<suffix>.
    tol: float
        Largest allowed |precise - rounded| wavelength [A] (0.005 + reading slack); larger means the
        files do not belong together.
    layout: str
        Model directory below runs (:func:`representative_dirs`).

    Returns
    -------
    np.ndarray
        (nb, nl, ny) float64; zero for empty bins. Nothing is cached: the caller saves it.

    Raises
    ------
    ValueError
        If OUT and OUT_IMU of a line have different numbers of rows (e.g. a truncated file of a crashed
        pformalsol) or wavelengths further apart than tol, or a filled bin has no representative.

    Validation
    ----------
    Reproduces the M424 lamfix_dT10.npz 'corr' bit for bit (test_library.py; with the numpy build and
    CPU features of the production, see the module notes: np.log).
    """
    # PP 2026-10-01: ported from fw_disc.py:273-296 (lam_corrections). Files are read with fwresults.read_out (every
    # six-column row; legacy: genfromtxt, 161 rows) and read_out_imu (= fw_disc.read_imu), the same numbers bit for bit;
    # new: OUT and OUT_IMU must have the same number of rows (legacy failed only when OUT_IMU was the longer file)
    from .fwresults import read_out, read_out_imu
    y = _grid_y(grid)
    names = _line_names(lines)
    if lref is None and not hasattr(lines, "lref"):
        raise ValueError("lref is required unless lines is a LineSet")
    lr = _lref(lines if lref is None else lref)
    if lr.size != len(names):
        raise ValueError("need one reference wavelength per line")
    count = np.asarray(lib["count"])
    nb = count.size
    if np.shape(lib["prof"])[1:] != (len(names), y.size):
        raise ValueError("library profiles {} do not match {} lines x {} grid points".format(
            np.shape(lib["prof"]), len(names), y.size))
    dirs = representative_dirs(reps, nb, runs=runs, layout=layout, edges=np.asarray(lib["edges"]))
    corr = np.zeros((nb, len(names), y.size))
    for b in np.where(count > 0)[0]:
        d = dirs[b]
        if not d:
            raise ValueError("bin {} holds models but has no representative model directory".format(b))
        for j, ln in enumerate(names):
            lp = read_out_imu(os.path.join(d, "OUT_IMU.{}_{}".format(ln, suffix)))[0]
            out = read_out(os.path.join(d, "OUT.{}_{}".format(ln, suffix)))
            lr_out, f = out["lam"], out["fnorm"]
            if lr_out.shape != lp.shape:
                raise ValueError("{}: OUT and OUT_IMU of {} have different numbers of rows ({} vs {})".format(
                    d, ln, lr_out.size, lp.size))
            dmax = np.abs(lp - lr_out).max()
            if not dmax <= tol:
                raise ValueError("{}: OUT and OUT_IMU wavelengths of {} differ by {:.4g} A > tol {}".format(
                    d, ln, dmax, tol))
            yr, yp = y_of_lam(lr_out, lr[j]), y_of_lam(lp, lr[j])
            corr[b, j] = np.interp(y, yp, f) - np.interp(y, yr, f)
    return corr
