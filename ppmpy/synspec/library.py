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
* Intensity library (the SPAMMS approach): :class:`ImuLibrary` holds, per T_eff' bin, the emergent
  continuum and line intensities I(y, mu) of one representative model (OUT_IMU files of the modified
  pformalsol) on the velocity grid, at the rays inside R_max (:func:`r_outer`) with node coordinate
  s = p / R_max = sqrt(1 - mu^2); :func:`find_candidates`, :func:`select_representatives`
  (:class:`Representatives`, representatives.txt) and :func:`build_imu_library` make it (legacy
  ``imu_library_dT10.npz``, reproduced bit for bit); :func:`imu_from_rays`, :func:`flux_from_rays` and
  :func:`flux_from_p` are the ray helpers (flux with I linear in s reproduces FASTWIND's flux).

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
tests/synspec/test_imulib.py (intensity library): synthetic raw / runs directories (the frozen
fw_imu_library.py lines run next to ours: candidates, representatives.txt, the build and the flux check, bit for
bit, and the same file bytes from :meth:`ImuLibrary.save` with meta=False), analytic ray fluxes,
:meth:`ImuLibrary.memory_estimate` == :meth:`ppmpy.synspec.disc.DiscImu.memory` (every mode); M424: the selection
reproduces representatives.txt and the build imu_library_dT10.npz, byte for byte.

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
PP 2026-10-02: intensity library ported from fw_imu_library.py (candidates, representatives, build,
flux check) and fw_disc.py (r_outer, flux_from_p, imu_from_rays); see the section 'intensity library'.
PP 2026-10-02: :func:`lib_nodes` ``single`` (bins that are nodes of their own; declared by the extended libraries of
:func:`ppmpy.synspec.libmode.extend_flux_library`), opt-in: libraries without the declaration merge as before.
PP 2026-10-02: review fixes: the smoothing of lib_nodes stays within the stretches of single / merged nodes; the
declaration records its bin layout (:func:`single_bins_record`) and lib_nodes checks it.
"""
import glob
import mmap
import multiprocessing
import os
import platform
import zipfile

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


SINGLE_BINS_KEY = "single_bins"
"""``FluxLibrary.params`` key of the bins that are interpolation nodes of their own (:func:`lib_nodes` ``single``;
written by :func:`ppmpy.synspec.libmode.extend_flux_library` for the bins of library-mode node models)."""
SINGLE_NB_KEY = "single_bins_nb"
"""``FluxLibrary.params`` key of the number of bins the declared single bins belong to (:data:`SINGLE_BINS_KEY`)."""
SINGLE_EDGES_KEY = "single_bins_edges"
"""``FluxLibrary.params`` key of the lower edges [K] of the declared single bins (:data:`SINGLE_BINS_KEY`): with
:data:`SINGLE_NB_KEY` the record of the bin layout the indices belong to (checked by :func:`lib_nodes`)."""


def single_bins_record(bins, edges):
    """
    The params entries that declare single bins (:data:`SINGLE_BINS_KEY`) with the record of their bin layout
    (:data:`SINGLE_NB_KEY`, :data:`SINGLE_EDGES_KEY`), so that a library rebuilt with other bins but the same params
    cannot apply the indices to the wrong bins (:func:`lib_nodes` raises).

    Parameters
    ----------
    bins: array-like of int
        Single bin indices.
    edges: array-like
        (nb + 1,) bin edges of the library they belong to [K].

    Returns
    -------
    dict
    """
    # PP 2026-10-02: new (reviewer: single_bins carried no record of their bin layout)
    e = np.asarray(edges, dtype=np.float64)
    b = sorted(int(i) for i in np.asarray(bins, dtype=np.int64).reshape(-1))
    if b and (b[0] < 0 or b[-1] >= e.size - 1):
        raise ValueError("single bins must lie in 0 .. {}".format(e.size - 2))
    return {SINGLE_BINS_KEY: b, SINGLE_NB_KEY: int(e.size - 1), SINGLE_EDGES_KEY: [float(e[i]) for i in b]}


def _lib_edges(lib):
    e = getattr(lib, "edges", None)
    if e is None:
        try:
            e = lib["edges"]
        except (KeyError, TypeError, ValueError):
            return None
    return np.asarray(e, dtype=np.float64)


def _single_mask(lib, single, nb):
    """The (nb,) bool mask of the bins that form nodes of their own, or None (legacy merging). A declaration in
    ``lib.params`` is checked against its layout record (:func:`single_bins_record`) where it has one."""
    # PP 2026-10-02: new (library extension: nmin applies to the per-point bins only)
    # PP 2026-10-02: the declaration's layout record (number of bins, lower edges) is checked (reviewer)
    if single is False:
        return None
    if single is None:
        params = getattr(lib, "params", None)
        single = params.get(SINGLE_BINS_KEY) if isinstance(params, dict) else None
        if single is None:
            return None
        rnb, redges = params.get(SINGLE_NB_KEY), params.get(SINGLE_EDGES_KEY)
        if rnb is not None and int(rnb) != nb:
            raise ValueError("the library declares single bins for {} bins but has {} (params copied to a library "
                             "with other bins? pass single= explicitly or drop params['{}'])".format(
                                 int(rnb), nb, SINGLE_BINS_KEY))
        if redges is not None:
            e = _lib_edges(lib)
            sb = np.asarray(single, dtype=np.int64).reshape(-1)
            if (e is not None and (len(redges) != sb.size or (sb.size and (sb.min() < 0 or sb.max() >= nb)) or
                                   not np.array_equal(e[sb], np.asarray(redges, dtype=np.float64)))):
                raise ValueError("the library's declared single bins do not have their recorded lower edges (params "
                                 "copied to a library with other bins? pass single= explicitly)")
    s = np.asarray(single)
    if s.dtype == bool:
        if s.shape != (nb,):
            raise ValueError("a bool single must have shape (nb,) = ({},), got {}".format(nb, s.shape))
        m = s.copy()
    else:
        s = s.reshape(-1)
        if s.size and (s.dtype.kind not in "iu" or s.min() < 0 or s.max() >= nb):
            raise ValueError("single must be a bool mask (nb,) or bin indices in 0 .. nb - 1 = {}".format(nb - 1))
        m = np.zeros(nb, bool)
        m[s.astype(np.int64)] = True
    return m if m.any() else None


def _node_groups(cnt, nmin, single=None):
    """
    The library bins of every node (:func:`lib_nodes`). single None: the legacy rule (fw_disc.lib_nodes). With a mask:
    every filled single bin is a node of its own; the other bins are merged as by the legacy rule within each run of
    consecutive non-single bins (a run's leftover joins the run's last node, or is a node of its own), never across a
    single bin.
    """
    if single is None:
        # PP 2026-10-01: ported from fw_disc.py:299-349 (lib_nodes; the legacy merging, unchanged)
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
        return groups
    # PP 2026-10-02: new (library extension)
    groups, run, cur, c = [], [], [], 0.0

    def close():
        if cur:
            if run:
                run[-1] = run[-1] + cur
            else:
                run.append(list(cur))
        groups.extend(run)

    for b in range(cnt.size):
        if single[b]:
            close()
            run, cur, c = [], [], 0.0
            if cnt[b] > 0:
                groups.append([b])
            continue
        if not cnt[b] > 0:
            continue
        cur.append(b)
        c += cnt[b]
        if c >= nmin:
            run.append(cur)
            cur, c = [], 0.0
    close()
    return groups


def lib_nodes(lib, nmin=20, smooth=0.0, corr=None, single=None):
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
        sampling) removes the sawtooth while keeping linear trends. With single bins in force the window never
        crosses from the nodes of single bins to the merged nodes or back: the fit of a node uses only the nodes of
        its stretch (maximal run of consecutive nodes of one kind). So the merged nodes of an extended library's
        base bins are the base library's smoothed nodes bit for bit (one-sided at the seam, as at the base's own
        ends), and each run of library-mode nodes is smoothed over its own nodes.
    corr: np.ndarray, optional
        (nb, nl, ny) additive per-bin profile correction, added to the bin profiles (in float64) before
        merging, e.g. :func:`wavelength_rounding_correction` (the legacy ``lamfix=True``). It acts per bin, never
        across bins: for an extended library the base rows must be the base library's correction (then the base
        nodes are its corrected nodes bit for bit) and the node rows the node models' own corrections
        (:func:`ppmpy.synspec.libmode.extend_correction`).
    single: None, False, bool mask (nb,) or bin indices, optional
        Bins that are interpolation nodes of their own: never merged with a neighbour, and the merging of the
        other bins (nmin) runs separately in every stretch between them (a leftover joins the stretch's last
        node); the smoothing (``smooth``) stays within the stretches too. None (default): the library's own
        declaration, ``lib.params['single_bins']`` (:data:`SINGLE_BINS_KEY`; written by
        :func:`ppmpy.synspec.libmode.extend_flux_library` for the library-mode node bins, whose single models nmin
        must not merge; ValueError when its layout record, :func:`single_bins_record`, does not match the library's
        bins), else none; libraries without it (every per-point library, the M424 library_dT10.npz, legacy dicts and
        NpzFiles) are merged and smoothed exactly as before. False: ignore a declaration (legacy merging and
        smoothing). An explicit mask or index list is not checked against a record.

    Returns
    -------
    LibraryNodes
        ``params`` gain 'single' (number of single bins) only when single bins are in force.

    Validation
    ----------
    M424 library_dT10.npz, nmin=20: 245 nodes, 35829-38891 K; equal bit for bit to the frozen legacy
    code for (nmin=20), (nmin=20, smooth=335) and (nmin=20, corr=lamfix) (test_library.py). single:
    tests/synspec/test_library_extend.py (the base bins of an extended library give the base library's nodes bit
    for bit, every single bin is a node).
    """
    # PP 2026-10-01: ported from fw_disc.py:299-349 (lib_nodes); lamfix=True -> corr=lam_corrections(lib)
    # PP 2026-10-02: single (library extension; the grouping moved to _node_groups, its legacy branch unchanged)
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
    smask = _single_mask(lib, single, cnt.size)
    groups = _node_groups(cnt, nmin, smask)
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
        # PP 2026-10-02: with single bins in force the window stays within a stretch (reviewer: smoothing across the
        # seam of an extended library changed the base nodes); without them one stretch, the legacy loop unchanged
        if smask is None:
            stretch = None
        else:
            kind = np.array([bool(smask[g[0]]) for g in groups], dtype=np.int8)
            stretch = np.split(np.arange(nn), np.flatnonzero(np.diff(kind) != 0) + 1)
            stretch = {int(i): st for st in stretch for i in st}
        ps, fs_ = np.empty_like(prof), np.empty_like(fc)
        for i in range(nn):
            if stretch is None:
                k = np.where(np.abs(t - t[i]) <= smooth / 2)[0]
            else:
                st = stretch[i]
                k = st[np.abs(t[st] - t[i]) <= smooth / 2]
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
    if smask is not None:
        params["single"] = int(smask.sum())
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
    reps: str, os.PathLike, ImuLibrary, Representatives or array-like
        * path of an .npz with 'idx_rep' (nb,) (the intensity library, M424
          /scratch/ppathak/fastwind_imu/imu_library_dT10.npz; with ``edges``, its 'edges' must be there
          and equal them);
        * an object with 'idx_rep' (and 'edges'), e.g. an :class:`ImuLibrary` (the same as its file);
        * :class:`Representatives`: the directory of each bin with a representative
          (:meth:`Representatives.model_dirs`: ``runs/layout`` with ``runs``, else the representatives' own
          directories), None for the other bins (not the nearest bin's model, unlike an intensity library's
          idx_rep; a filled bin without one then fails in :func:`wavelength_rounding_correction`); with
          ``edges``, every representative's T_eff' must lie in its bin (:func:`teff_bins`);
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
        the number of representatives is not nb; for a non-integer index or a missing ``runs``; for
        Representatives with bins outside 0 .. nb - 1 or (with ``edges``) T_eff' outside their bins.
    """
    # PP 2026-10-01: ported from fw_disc.py:280-283 (the model directory of each bin in lam_corrections)
    # PP 2026-10-02: objects with idx_rep (ImuLibrary) and Representatives accepted
    if isinstance(reps, Representatives):
        if len(reps) and (reps.bins.min() < 0 or reps.bins.max() >= nb):
            raise ValueError("representative bins must lie in 0 .. nb - 1 = {}".format(nb - 1))
        if edges is not None and len(reps):
            wrong = reps.bins[teff_bins(reps.teff, np.asarray(edges)) != reps.bins]
            if wrong.size:
                raise ValueError("the T_eff' of the representatives of bins {}{} lie outside their bins (made for "
                                 "other bin edges?)".format(wrong[:20].tolist(), " ..." if wrong.size > 20 else ""))
        md = reps.model_dirs(runs, layout=layout)
        return [md.get(b) for b in range(nb)]
    if hasattr(reps, "idx_rep") and not isinstance(reps, (str, os.PathLike, np.ndarray)):
        if edges is not None:
            e = getattr(reps, "edges", None)
            if e is None:
                raise ValueError("{!r} has no 'edges' to check against the library's".format(type(reps).__name__))
            e = np.asarray(e)
            if e.shape != np.shape(edges) or not np.allclose(e, edges):
                raise ValueError("the bins of the {} differ from the library's".format(type(reps).__name__))
        reps = np.asarray(reps.idx_rep)
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


# ----------------------------------------------------------------------------------------------
# intensity library: emergent intensities I(lambda, mu) of one representative model per bin
# ----------------------------------------------------------------------------------------------
IMU_KEYS = ("edges", "tmean", "count", "src", "idx_rep", "teff_rep", "rmax", "nnode", "s", "Ic", "Il")
"""Members of an intensity-library file (legacy ``imu_library_dT10.npz``, in this order; :meth:`ImuLibrary.save`
adds '_meta' unless meta=False)."""

IMU_LAYOUT = "P{idx:06d}/P{idx:06d}"
"""Model directory of model idx below the runs directory of the modified pformalsol (fw_imu_run.sh)."""

_trapz = getattr(np, "trapezoid", None) or np.trapz


def r_outer(p, Ic, frac=1e-3):
    """
    Outer radius of the emitting atmosphere in the rays of FASTWIND's formal solution.

    Parameters
    ----------
    p: array-like
        (nray,) impact parameters [inner-boundary radius], increasing, p[0] = 0 (disc centre).
    Ic: array-like
        (nk, nray) emergent continuum intensity of every wavelength point and ray.
    frac: float
        Threshold relative to the disc-centre intensity of the same wavelength point.

    Returns
    -------
    float
        p of the first ray (in p order) whose continuum intensity is below ``frac`` times the
        disc-centre value at every wavelength point; p[-1] if no ray is. Rays with p <= R_max are
        mapped to the disc (the ray at R_max itself becomes mu = 0).

    Notes
    -----
    The legacy docstring says 'stays below'; the code (kept) takes the first such ray. M424
    (frac = 1e-3): R_max = 1.0119-1.0140, 42 of 79 rays kept for every model and line.
    """
    # PP 2026-10-02: ported from fw_disc.py:244-249 (r_outer)
    p = np.asarray(p)
    Ic = np.asarray(Ic)
    rel = (Ic / Ic[:, :1]).max(axis=0)
    beyond = np.where(rel < frac)[0]
    return float(p[beyond[0]]) if beyond.size else float(p[-1])


def imu_from_rays(p, I, rmax):
    """
    Intensity as a function of mu for the rays with p <= rmax.

    Parameters
    ----------
    p: array-like
        (nray,) impact parameters, increasing.
    I: array-like
        (nk, nray) intensities.
    rmax: float
        Outer radius (:func:`r_outer`).

    Returns
    -------
    mu: np.ndarray
        (nmu,) ``sqrt(1 - (p / rmax)^2)`` of the rays inside rmax, in increasing order (mu = 0 at
        p = rmax).
    I_mu: np.ndarray
        (nk, nmu) the rays' intensities in the same order.

    Notes
    -----
    With this mapping int I 2 mu dmu = int_0^rmax I 2 p dp / rmax^2, so the flux of the 1-D model is
    kept exactly when I is interpolated linearly in s = p / rmax (:func:`flux_from_rays`).
    """
    # PP 2026-10-02: ported from fw_disc.py:257-264 (imu_from_rays)
    p = np.asarray(p)
    I = np.asarray(I)
    inside = p <= rmax
    mu = np.sqrt(np.clip(1.0 - (p[inside] / rmax) ** 2, 0.0, 1.0))[::-1]
    return mu, I[:, inside][:, ::-1]


def flux_from_p(p, I):
    """
    Flux-like integral int I 2p dp over all rays (what FASTWIND does), trapezoidal.

    Parameters
    ----------
    p: array-like
        (nray,) impact parameters.
    I: array-like
        (nk, nray) intensities.

    Returns
    -------
    np.ndarray
        (nk,). M424: the ratio line / continuum of this integral over all 79 rays reproduces FASTWIND's
        flux profile to 2.6e-4 (project log 2026-09-28); :func:`flux_from_rays` to <= 8e-6.
    """
    # PP 2026-10-02: ported from fw_disc.py:252-254 (flux_from_p)
    p = np.asarray(p)
    return _trapz(np.asarray(I) * 2.0 * p, p, axis=1)


def flux_from_rays(p, Ic, Il, rmax, nmu=4001, full=False):
    """
    Continuum-normalised flux profile from the rays with I linear in s = p / R_max (the flux check of
    fw_imu_library.py and the intensity method's interpolation).

    F = int_0^1 I(mu) 2 mu dmu on ``nmu`` equidistant mu points (trapezoidal), with I(mu) interpolated
    linearly in s = sqrt(1 - mu^2) between the rays p <= rmax (np.interp per wavelength point).

    Parameters
    ----------
    p: array-like
        (nray,) impact parameters, increasing; only the rays with p <= rmax are used.
    Ic, Il: array-like
        (nk, nray) continuum and line intensity.
    rmax: float
        Outer radius (:func:`r_outer`); s = p / rmax.
    nmu: int
        Number of mu points (legacy 4001).
    full: bool
        Also return the two fluxes.

    Returns
    -------
    fn: np.ndarray
        (nk,) F_line / F_cont.
    fl, fc: np.ndarray
        (nk,) F_line and F_cont (only with ``full=True``; F = I for a uniform intensity).

    Validation
    ----------
    Bit for bit the arithmetic of fw_imu_library.py:105,114-116 (test_imulib.py). M424: max |fn - F/F_c
    of OUT| <= 8e-6 over the 303 representatives and 3 lines.
    """
    # PP 2026-10-02: ported from fw_imu_library.py:105,114-116 (step 4: flux check); the mask p <= rmax is new (the
    # legacy code passed the kept rays; applying it again to them changes nothing)
    p = np.asarray(p)
    inside = p <= rmax
    pk = p[inside]
    Ic = np.asarray(Ic)[:, inside]
    Il = np.asarray(Il)[:, inside]
    if pk.size < 2:
        raise ValueError("need at least 2 rays with p <= rmax")
    mf = np.linspace(0.0, 1.0, int(nmu))
    sf = np.sqrt(1.0 - mf ** 2)
    Lf = np.array([np.interp(sf, pk / rmax, row) for row in Il])
    Cf = np.array([np.interp(sf, pk / rmax, row) for row in Ic])
    fl = _trapz(Lf * 2 * mf, mf, axis=1)
    fc = _trapz(Cf * 2 * mf, mf, axis=1)
    fn = fl / fc
    return (fn, fl, fc) if full else fn


def find_candidates(raw_dirs, pattern="P*", require=("meta.txt", "CONT_FORMAL"), status=None):
    """
    Saved per-point models that can serve as representatives: model directories ``<raw>/P<idx>/`` with a
    ``meta.txt`` (``idx teff status ...``) and the model files (CONT_FORMAL) that pformalsol needs.

    Parameters
    ----------
    raw_dirs: str, os.PathLike or sequence of them
        Directories holding extracted model directories (fw_imu_extract.sh; M424
        /scratch/ppathak/fastwind_imu/raw and /scratch/ppathak/fastwind_contfix/d3200_4parts/raw).
    pattern: str
        Glob pattern of the model directories.
    require: str or sequence of str
        File(s) a model directory must hold (meta.txt is always needed).
    status: str or sequence of str, optional
        Keep only models whose meta.txt status (3rd column, as :func:`ppmpy.synspec.fwresults.parse_meta`)
        is one of these, e.g. ('ok',); a meta.txt without a status column does not match. None (default,
        legacy): the status is ignored. M424: all 3881 candidates are 'ok' (fw_sphere_point.sh keeps the
        model files of 'ok' models only), so ('ok',) selects the same representatives.

    Returns
    -------
    list of tuple
        (idx int, teff float [K, from meta.txt], model_dir str), in the order of raw_dirs and, within
        each, of the sorted directory names (the legacy order, which decides ties in
        :func:`select_representatives`).
    """
    # PP 2026-10-02: ported from fw_imu_library.py:55-60 (meta.txt is read with a context manager); new: a str
    # require, the opt-in status filter
    if isinstance(raw_dirs, (str, os.PathLike)):
        raw_dirs = [raw_dirs]
    require = (require,) if isinstance(require, str) else tuple(require)
    need = require if "meta.txt" in require else ("meta.txt",) + require
    if isinstance(status, str):
        status = (status,)
    status = None if status is None else tuple(str(s) for s in status)
    cands = []
    for r in raw_dirs:
        for d in sorted(glob.glob(os.path.join(os.fspath(r), pattern))):
            if all(os.path.exists(os.path.join(d, f)) for f in need):
                with open(os.path.join(d, "meta.txt")) as fh:
                    f = fh.read().split()
                if status is not None and (len(f) < 3 or f[2] not in status):
                    continue
                cands.append((int(f[0]), float(f[1]), d))
    return cands


class Representatives:
    """
    The representative model of each library bin (:func:`select_representatives`,
    :meth:`read`).

    Parameters
    ----------
    bins: array-like
        (n,) library bins, strictly increasing.
    idx: array-like
        (n,) model indices.
    teff: array-like
        (n,) the models' T_eff' [K].
    dirs: sequence of str
        (n,) model directories (where the model files are; for the build the OUT_IMU files are read from
        the runs directory of the modified pformalsol unless ``runs_dir`` is None).
    missing: sequence of int, optional
        Filled bins without a candidate model (:func:`select_representatives`).
    n_candidates: int, optional
        Number of candidate models the selection saw.

    Notes
    -----
    Iterating yields (bin, idx, teff, dir); ``reps[b]`` gives (idx, teff, dir) of bin b.
    """

    def __init__(self, bins, idx, teff, dirs, missing=(), n_candidates=None):
        self.bins = np.asarray(bins, dtype=np.int64).reshape(-1)
        self.idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        self.teff = np.asarray(teff, dtype=np.float64).reshape(-1)
        self.dirs = [os.fspath(d) for d in dirs]
        self.missing = [int(b) for b in missing]
        self.n_candidates = None if n_candidates is None else int(n_candidates)
        n = self.bins.size
        if not (self.idx.size == self.teff.size == len(self.dirs) == n):
            raise ValueError("bins, idx, teff and dirs must have the same length")
        if n > 1 and not np.all(np.diff(self.bins) > 0):
            raise ValueError("bins must be strictly increasing (one representative per bin)")

    def __len__(self):
        return self.bins.size

    def __iter__(self):
        for k in range(self.bins.size):
            yield int(self.bins[k]), int(self.idx[k]), float(self.teff[k]), self.dirs[k]

    def __getitem__(self, b):
        k = int(np.searchsorted(self.bins, b))
        if k >= self.bins.size or self.bins[k] != b:
            raise KeyError(b)
        return int(self.idx[k]), float(self.teff[k]), self.dirs[k]

    def __contains__(self, b):
        k = int(np.searchsorted(self.bins, b))
        return k < self.bins.size and self.bins[k] == b

    def __repr__(self):
        return "Representatives(n={}, missing={}, T {}, candidates={})".format(
            len(self), len(self.missing),
            "{:.0f}-{:.0f} K".format(self.teff.min(), self.teff.max()) if len(self) else "-", self.n_candidates)

    def as_dict(self):
        """{bin: (idx, teff, dir)}."""
        return {b: (i, t, d) for b, i, t, d in self}

    def model_dirs(self, runs_dir=None, layout=IMU_LAYOUT):
        """
        {bin: directory with the OUT_IMU files}: ``runs_dir/layout.format(idx=idx)`` (the reruns of the
        modified pformalsol, fw_imu_run.sh), or the representatives' own directories for runs_dir None.
        """
        if runs_dir is None:
            return {b: d for b, _, _, d in self}
        return {b: os.path.join(os.fspath(runs_dir), layout.format(idx=i)) for b, i, _, _ in self}

    def write(self, path):
        """
        Write representatives.txt in the legacy format, one line ``bin idx teff dir`` per bin (teff with
        3 decimals, as fw_imu_library.py --stage select; fw_imu_run.sh reads the 4th column), atomically.

        Returns
        -------
        str
            path.
        """
        # PP 2026-10-02: ported from fw_imu_library.py:76-79 (temporary name + os.replace is new)
        path = os.fspath(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = "{}.tmp{}".format(path, os.getpid())
        try:
            with open(tmp, "w") as f:
                for b, i, t, d in self:
                    f.write("{} {} {:.3f} {}\n".format(b, i, t, d))
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return path

    @classmethod
    def read(cls, path, from_meta=False):
        """
        Read representatives.txt (legacy format ``bin idx teff dir``).

        Parameters
        ----------
        path: str or os.PathLike
            The file.
        from_meta: bool or 'auto'
            True: take teff from ``<dir>/meta.txt`` (full precision) instead of the file's 3 decimals; the
            model index there must agree. 'auto': the same where ``<dir>/meta.txt`` exists, else the file's
            value. False (default): the file's value. The legacy build used the meta.txt value
            (:func:`build_imu_library` reads a representatives.txt with 'auto' by default); both agree
            whenever meta.txt has at most 3 decimals (M424: all 3881 candidates).

        Returns
        -------
        Representatives

        Notes
        -----
        Each line is stripped before it is split, so trailing blanks (or the CR of a CRLF file) do not end up
        in the directory name; a directory name ending in blanks cannot be represented.
        """
        # PP 2026-10-02: ported from fw_imu_library.py:76-79 (the format); new: from_meta, 'auto', stripped lines
        if isinstance(from_meta, str) and from_meta != "auto":
            raise ValueError("from_meta must be False, True or 'auto', got {!r}".format(from_meta))
        mode = "auto" if isinstance(from_meta, str) else bool(from_meta)
        bins, idx, teff, dirs = [], [], [], []
        with open(os.fspath(path)) as f:
            for ln in f:
                if not ln.strip():
                    continue
                tok = ln.strip().split(None, 3)
                if len(tok) != 4:
                    raise ValueError("{}: expected 'bin idx teff dir', got {!r}".format(path, ln))
                b, i, t, d = int(tok[0]), int(tok[1]), float(tok[2]), tok[3]
                meta = os.path.join(d, "meta.txt")
                if mode is True or (mode == "auto" and os.path.isfile(meta)):
                    with open(meta) as fh:
                        m = fh.read().split()
                    if int(m[0]) != i:
                        raise ValueError("{}/meta.txt is model {}, representatives.txt says {}".format(d, m[0], i))
                    t = float(m[1])
                bins.append(b)
                idx.append(i)
                teff.append(t)
                dirs.append(d)
        return cls(bins, idx, teff, dirs)


def select_representatives(lib, candidates, allow_missing=False):
    """
    Pick the representative model of every filled library bin: among the candidates in the bin, the
    one whose T_eff' is closest to the bin's mean T_eff' (first in candidate order on a tie).

    Parameters
    ----------
    lib: FluxLibrary or mapping
        Flux library (edges, tmean, count), e.g. library_dT10.npz.
    candidates: sequence of tuple
        (idx, teff, model_dir) per saved model (:func:`find_candidates`); (idx, teff) gives empty dirs.
    allow_missing: bool
        False (default, legacy): raise ValueError naming the filled bins without a candidate. True:
        return the bins that have one and list the others in ``missing``.

    Returns
    -------
    Representatives
        One entry per filled bin with a candidate, bins ascending.

    Raises
    ------
    ValueError
        For bins without a candidate (unless allow_missing), or a non-finite candidate T_eff'.

    Notes
    -----
    Bins: :func:`teff_bins` (candidates beyond the edges go to the end bins). The legacy loop over bins
    (argmin per bin) is replaced by one lexsort with the same tie rule; M424 (3881 candidates from
    the two raw directories) reproduces /scratch/ppathak/fastwind_imu/representatives.txt.
    """
    # PP 2026-10-02: ported from fw_imu_library.py:52-71 (bins of the candidates, closest to tmean, missing bins)
    edges = np.asarray(lib["edges"])
    tmean = np.asarray(lib["tmean"])
    count = np.asarray(lib["count"])
    cands = list(candidates)
    idx_c = np.array([int(c[0]) for c in cands], dtype=np.int64)
    t_c = np.array([float(c[1]) for c in cands], dtype=np.float64)
    if not np.all(np.isfinite(t_c)):
        raise ValueError("candidate T_eff' must be finite")
    filled = np.flatnonzero(count > 0)
    sel_b = np.zeros(0, np.int64)
    sel_k = np.zeros(0, np.int64)
    if t_c.size:
        b_c = teff_bins(t_c, edges)
        use = np.flatnonzero(count[b_c] > 0)
        if use.size:
            bu = b_c[use]
            dt = np.abs(t_c[use] - tmean[bu])
            order = np.lexsort((use, dt, bu))               # bin, then |T - tmean|, then candidate order
            first = order[np.r_[True, np.diff(bu[order]) != 0]]
            sel_b, sel_k = bu[first].astype(np.int64), use[first]
    missing = [int(b) for b in np.setdiff1d(filled, sel_b)]
    if missing and not allow_missing:
        raise ValueError("{} of {} filled bins have no candidate model (extract models for them first): {}{}".format(
            len(missing), filled.size, missing[:20], " ..." if len(missing) > 20 else ""))
    dirs = [cands[k][2] if len(cands[k]) > 2 else "" for k in sel_k]
    return Representatives(sel_b, idx_c[sel_k], t_c[sel_k], dirs, missing=missing, n_candidates=len(cands))


def read_representatives(path, from_meta=False):
    """:meth:`Representatives.read`."""
    return Representatives.read(path, from_meta=from_meta)


def write_representatives(reps, path):
    """:meth:`Representatives.write`."""
    return reps.write(path)


def _as_representatives(reps, from_meta=False):
    """A Representatives from itself, a representatives.txt path (read with ``from_meta``), or a mapping
    {bin: (idx, teff[, dir])}."""
    if isinstance(reps, Representatives):
        return reps
    if isinstance(reps, (str, os.PathLike)):
        return Representatives.read(reps, from_meta=from_meta)
    if hasattr(reps, "items"):
        bins = sorted(int(b) for b in reps)
        vals = [tuple(reps[b]) for b in bins]
        return Representatives(bins, [v[0] for v in vals], [v[1] for v in vals],
                               [v[2] if len(v) > 2 else "" for v in vals])
    raise ValueError("reps must be Representatives, a representatives.txt path or a mapping bin -> (idx, teff[, dir])")


class ImuLibrary:
    """
    Emergent-intensity library: per T_eff' bin, the continuum and line intensities of one representative
    model at its rays inside R_max, on the common velocity grid.

    Parameters are the attributes below (the legacy members :data:`IMU_KEYS`, then meta, params, inputs,
    path); arrays keep their dtypes (memory maps stay memory maps).

    Attributes
    ----------
    edges: np.ndarray
        (nb + 1,) float64 bin edges [K] (of the flux library).
    tmean, count: np.ndarray
        (nb,) float64 bin mean T_eff' and number of models (of the flux library).
    src: np.ndarray
        (nb,) int64 bin whose representative a bin uses: itself for a bin with a representative, else
        the nearest such bin (the lower one on a tie).
    idx_rep, teff_rep: np.ndarray
        (nb,) int64 / float64 model index and T_eff' of bin src[b]'s representative.
    rmax: np.ndarray
        (nb, nl) float64 R_max (:func:`r_outer`) per line.
    nnode: np.ndarray
        (nb, nl) int64 rays kept (p <= R_max).
    s: np.ndarray
        (nb, nl, K) float64 ray nodes s = p / R_max (increasing, 0 .. 1), NaN beyond nnode.
    Ic, Il: np.ndarray or np.memmap
        (nb, nl, K, ny) float32 continuum and line intensity of the kept rays on the grid, 0 beyond
        nnode. Bins with src != b hold copies of bin src's arrays (as the legacy file).
    meta, params, inputs: dict
        '_meta' record of the file read ({} for the legacy file), build parameters and input paths.
    path: str or None
        File the library was read from (absolute).
    lines: LineSet or None
        The lines (names and velocity zero points) recorded by :func:`build_imu_library` (``params['lines']``
        and ``params['lref']``; kept in the '_meta' of :meth:`save`); None for the legacy file.
        :class:`ppmpy.synspec.disc.DiscImu` takes its ``lines`` / ``lref`` from it.
    lref: np.ndarray or None
        (nl,) float64 velocity zero points [A] (``params['lref']``), or None.
    grid: VelocityGrid or None
        The velocity grid recorded by :func:`build_imu_library` (``params['grid']``; None for the legacy file
        and for a library built on a plain array of velocities).

    Notes
    -----
    ``lib[key]`` returns the arrays by their legacy names and iterating yields the names (as a mapping), so an
    ImuLibrary can be passed where the legacy code took the NpzFile of imu_library_dT10.npz. M424
    (imu_library_dT10.npz): nb = 351 (303 with a representative), nl = 3, K = 42 (nnode = 42 everywhere),
    ny = 5401; Ic and Il 955 MB each.
    """

    MMAP_MIN_BYTES = 1 << 20
    """:meth:`load` memory-maps the members of at least this size (M424: Ic, Il); smaller ones are read."""

    def __init__(self, edges, tmean, count, src, idx_rep, teff_rep, rmax, nnode, s, Ic, Il, meta=None, params=None,
                 inputs=None, path=None):
        # PP 2026-10-02: new (the legacy code used the NpzFile); dtypes are kept as given (bitwise save)
        self.edges, self.tmean, self.count = np.asarray(edges), np.asarray(tmean), np.asarray(count)
        self.src, self.idx_rep, self.teff_rep = np.asarray(src), np.asarray(idx_rep), np.asarray(teff_rep)
        self.rmax, self.nnode, self.s = np.asarray(rmax), np.asarray(nnode), np.asarray(s)
        self.Ic = Ic if isinstance(Ic, np.ndarray) else np.asarray(Ic)
        self.Il = Il if isinstance(Il, np.ndarray) else np.asarray(Il)
        self.meta = dict(meta or {})
        self.params = dict(params or {})
        self.inputs = dict(inputs or {})
        self.path = None if path is None else os.fspath(path)
        self._ident = None
        self._mmap = False
        nb = self.edges.size - 1
        if self.edges.ndim != 1 or nb < 1:
            raise ValueError("edges must be 1-D with at least 2 entries")
        for k in ("tmean", "count", "src", "idx_rep", "teff_rep"):
            if getattr(self, k).shape != (nb,):
                raise ValueError("{} must have shape (nb,) = ({},), got {}".format(k, nb, getattr(self, k).shape))
        if self.Ic.ndim != 4 or self.Ic.shape != self.Il.shape or self.Ic.shape[0] != nb:
            raise ValueError("Ic and Il must have the same shape (nb, nl, K, ny), got {} and {}".format(
                self.Ic.shape, self.Il.shape))
        nl, K = self.Ic.shape[1:3]
        if self.s.shape != (nb, nl, K):
            raise ValueError("s must have shape (nb, nl, K) = {}, got {}".format((nb, nl, K), self.s.shape))
        for k in ("rmax", "nnode"):
            if getattr(self, k).shape != (nb, nl):
                raise ValueError("{} must have shape (nb, nl) = {}, got {}".format(k, (nb, nl), getattr(self, k).shape))
        if self.src.dtype.kind not in "iu" or self.src.min() < 0 or self.src.max() >= nb \
                or not np.array_equal(self.src[self.src], self.src):
            raise ValueError("src must map every bin to a bin that is its own source")
        for k in ("lines", "lref"):
            if self.params.get(k) is not None and len(self.params[k]) != nl:
                raise ValueError("params[{!r}] has {} entries, the library {} lines".format(k, len(self.params[k]), nl))

    # -- shape and access ----------------------------------------------------------------------
    @property
    def nb(self):
        """Number of bins."""
        return self.count.size

    @property
    def nl(self):
        """Number of lines."""
        return self.Ic.shape[1]

    @property
    def K(self):
        """Ray nodes per (bin, line), padded (max of nnode)."""
        return self.Ic.shape[2]

    @property
    def ny(self):
        """Number of velocity-grid points."""
        return self.Ic.shape[3]

    @property
    def filled(self):
        """Bins that hold models in the flux library (count > 0)."""
        return self.count > 0

    @property
    def nbytes(self):
        """Bytes of all arrays (memory maps count with their full size)."""
        return int(sum(getattr(self, k).nbytes for k in IMU_KEYS))

    @property
    def lines(self):
        """LineSet of the recorded line names and lref (``params``), or None (legacy file)."""
        # PP 2026-10-02: new (picked up by disc.DiscImu through its LineSet attribute 'lines')
        from .spectral import LineSet
        names, lref = self.params.get("lines"), self.params.get("lref")
        if names is None or lref is None:
            return None
        return LineSet(names, lref)

    @property
    def lref(self):
        """(nl,) float64 recorded velocity zero points [A] (``params['lref']``), or None."""
        lref = self.params.get("lref")
        return None if lref is None else np.asarray(lref, dtype=np.float64)

    @property
    def grid(self):
        """The recorded VelocityGrid (``params['grid']``), or None."""
        from .spectral import VelocityGrid
        g = self.params.get("grid")
        return None if g is None else VelocityGrid(**g)

    def bin_index(self, teff):
        """Library bin of each T_eff' (:func:`teff_bins`)."""
        return teff_bins(teff, self.edges)

    def unique_nodes(self):
        """Bins with their own representative model, ascending (``np.unique(src)``; M424: 303): the T_eff' nodes of
        the intensity method (legacy fw_disc.DiscImu)."""
        # PP 2026-10-02: ported from fw_disc.py:415 (DiscImu.__init__)
        return np.unique(self.src)

    def node_teff(self):
        """T_eff' of the representatives of :meth:`unique_nodes` [K] (increasing for a library built here)."""
        return self.teff_rep[self.unique_nodes()]

    def __getitem__(self, key):
        if key not in IMU_KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key):
        return key in IMU_KEYS

    def __iter__(self):
        return iter(IMU_KEYS)

    def keys(self):
        return list(IMU_KEYS)

    @property
    def files(self):
        """The member names (as NpzFile.files)."""
        return list(IMU_KEYS)

    def as_dict(self):
        """The legacy members as a dict; arrays are not copied."""
        return {k: getattr(self, k) for k in IMU_KEYS}

    def __repr__(self):
        return "ImuLibrary(nb={}, nodes={}, nl={}, K={}, ny={}, T_rep {:.0f}-{:.0f} K{})".format(
            self.nb, self.unique_nodes().size, self.nl, self.K, self.ny, float(np.min(self.teff_rep)),
            float(np.max(self.teff_rep)), ", " + self.path if self.path else "")

    # -- files ---------------------------------------------------------------------------------
    @classmethod
    def load(cls, path, mmap=True):
        """
        Read an intensity library (legacy imu_library_dT10.npz, or one written by :meth:`save`).

        Parameters
        ----------
        path: str or os.PathLike
            The .npz (members :data:`IMU_KEYS`; '_meta' optional).
        mmap: bool
            True (default): members of at least :attr:`MMAP_MIN_BYTES` stored uncompressed are read-only
            memory maps (:func:`ppmpy.synspec.io.npz_member_memmap`; zero copy, opening reads only the
            headers and the small members, ~0.4 MB for M424); compressed members are read. False: every
            member is read (M424: 1.9 GB).

        Returns
        -------
        ImuLibrary
            ``inputs`` = {'source': absolute path}; ``meta`` = the file's '_meta' ({} for the legacy file).

        Notes
        -----
        Pickling (e.g. to a 'spawn' worker): a library with memory-mapped members (mmap=True and an
        uncompressed file) is re-opened from its path in the receiving process (memory maps again, nothing
        copied); the file must not have been replaced in between (device, inode, size, mtime are checked).
        A library without memory maps (mmap=False, or a compressed file, whose members are read) pickles its
        arrays (M424: 1.9 GB per worker; a 'fork' pool inherits them instead), keeping ``path``.
        """
        from .io import npz_member_memmap, read_meta
        path = os.fspath(path)
        ident = _file_identity(path)
        with zipfile.ZipFile(path) as zf:
            infos = {i.filename[:-4]: i for i in zf.infolist() if i.filename.endswith(".npy")}
        missing = [k for k in IMU_KEYS if k not in infos]
        if missing:
            raise ValueError("{} is not an intensity library (missing {})".format(path, missing))
        arrays = {}
        with np.load(path) as z:
            meta = read_meta(z)
            for k in IMU_KEYS:
                info = infos[k]
                if mmap and info.file_size >= cls.MMAP_MIN_BYTES and info.compress_type == zipfile.ZIP_STORED:
                    arrays[k] = npz_member_memmap(path, k)
                else:
                    arrays[k] = z[k]
        if _file_identity(path) != ident:
            raise RuntimeError("{} was replaced while it was read; load again".format(path))
        apath = os.path.abspath(path)
        lib = cls(meta=meta, params=meta.get("params", {}), inputs=dict(source=apath), path=apath, **arrays)
        lib._ident = ident
        lib._mmap = any(isinstance(a, np.memmap) for a in arrays.values())
        return lib

    def save(self, path, meta=None):
        """
        Write the library atomically as an uncompressed .npz with the legacy members in the legacy order,
        plus '_meta'.

        Parameters
        ----------
        path: str or os.PathLike
            Target .npz (M424: 1.91 GB).
        meta: dict, False or None
            None (default): :func:`ppmpy.synspec.io.make_meta` ('synspec.imu_library') with ``params``,
            ``inputs``, the CPU features ('cpu') and, for a library read from a file with '_meta', that
            record as 'source_meta'. False: no '_meta', i.e. the legacy file byte for byte (np.savez writes
            fixed zip dates). A dict: that record.

        Returns
        -------
        str
            path.

        Notes
        -----
        Memory maps are written in 16 MiB pieces by np.savez, so saving a memory-mapped library needs no
        copy in memory. Saving onto the file the library is mapped from replaces it (os.replace; the
        maps keep the old data), after which this object can no longer be pickled by path.
        """
        from .io import make_meta, save_npz
        if meta is None:
            extra = dict(cpu=_cpu_features())
            if self.meta:
                extra["source_meta"] = self.meta
            meta = make_meta("synspec.imu_library", params=self.params, inputs=self.inputs, **extra)
        return save_npz(os.fspath(path), self.as_dict(), meta=meta if meta is not False else None)

    def __reduce__(self):
        # PP 2026-10-02: a memory-mapped library travels as its path (as fwresults.ProfileStore)
        if self.path is not None and self._mmap and self._ident is not None:
            return (_reopen_imu_library, (self.path, self._ident))
        return (_imu_library_from_state, (self.as_dict(), self.meta, self.params, self.inputs, self.path))

    # -- memory of the disc integration ----------------------------------------------------------
    def memory_estimate(self, grid=None, lines=None, dtype="float64", fft="precomputed", chunk=128):
        """
        Memory of a :class:`ppmpy.synspec.disc.DiscImu` built on this library (``DiscImu(lib, grid, lines, dtype,
        fft, chunk)``), computed without building it: the terms of :meth:`DiscImu.memory`, which it equals
        (test_imulib.py, all fft / dtype modes and line subsets).

        Parameters
        ----------
        grid: VelocityGrid, optional
            The DiscImu grid, as DiscImu: default the grid recorded by the library (:attr:`grid`), else the
            M424 grid; a grid other than the recorded one is refused; it must have ``ny`` points. Sets the
            histogram width nv = 2 nshift + 1 and the FFT length L = next_fast_len(ny + 2 nshift + nv - 1).
        lines: sequence of int or str, optional
            The lines DiscImu builds (default all; names need :attr:`lines`).
        dtype: {'float64', 'float32'}
            Precision of the DiscImu library spectra.
        fft: {'precomputed', 'lazy'}
            DiscImu FFT mode.
        chunk: int
            Histogram rows per FFT block (DiscImu default 128).

        Returns
        -------
        dict
            Bytes, as :meth:`DiscImu.memory`: ``library`` (the integrator's own arrays: 'precomputed' the
            float64 / float32 intensity rows I0 and their FFTs Ihat of every built line; both modes the ray
            nodes Sflat of all lines and the line-centre continuum ic0 of the built lines), ``source`` ('lazy':
            this library's Il, Ic of the built lines' representatives, referenced, not copied), ``source_mapped``
            (the part of source that is memory-mapped file pages: clean page cache, shared by all processes
            and evictable; this library's Il and Ic are memory maps after :meth:`load` of an uncompressed
            file) and ``per_call`` (rough largest temporaries of one call for one line of sight: histogram, its
            FFT block, the lazy row cache; without the per-point arrays). Derived: ``total`` = library + source
            + per_call, ``anonymous`` = total - source_mapped (process memory that cannot be dropped), and
            ``total_gb``, ``anonymous_gb``. Shapes: ``nn`` (nodes, ``np.unique(src)``), ``K``, ``nr`` = nn K
            rows per line, ``ny``, ``nv``, ``L``, ``nf`` = L / 2 + 1, ``built`` (line indices).

        Raises
        ------
        ValueError
            Bad options, a grid other than the recorded one or with another number of points, unknown lines.

        Notes
        -----
        M424 (imu_library_dT10.npz memory-mapped; nn 303, K 42, nr 12726, ny 5401, L 7200). Estimate (GB; equal
        to DiscImu(ImuLibrary.load(...), ...).memory()) and measurement (login node 2026-10-02, a fresh process
        per mode, warm page cache; peak RSS = VmHWM, which includes the touched file pages of the memory maps:
        all 1.68 GB of Il and Ic for a 'precomputed' build and for a full-disc call; one call = one line of sight
        of 1 236 544 random points, all built lines, per-point arrays ~0.1 GB included):

        ====================  =======  ===============  ========  =========  ===============  ===============
        fft, dtype, lines     library  source (mapped)  per_call  anonymous  build: s, peak   call: s, peak
        ====================  =======  ===============  ========  =========  ===============  ===============
        precomputed f64, all  7.699    0                0.105     7.80       5.7 s, 9.48 GB   1.3 s, 9.72 GB
        precomputed f32, all  3.850    0                0.105     3.96       4.6 s, 5.64 GB   1.4 s, 5.87 GB
        precomputed f32, one  1.283    0                0.105     1.39       1.5 s, 2.02 GB   0.6 s, 2.24 GB
        lazy f64, all         0.001    1.650 (1.650)    0.186     0.19       0.03 s, 0.88 GB  2.8 s, 1.79 GB
        lazy f32, all         0.001    1.650 (1.650)    0.157     0.16       0.03 s, 0.88 GB  2.1 s, 1.77 GB
        lazy f32, one         0.0004   0.550 (0.550)    0.157     0.16       0.01 s, 0.35 GB  0.7 s, 0.82 GB
        ====================  =======  ===============  ========  =========  ===============  ===============

        Peak process memory is therefore about ``anonymous`` (measured: precomputed f64 7.8 GB, f32 4.0 GB, f32
        one line 1.4 GB; lazy ~0.3 GB with the per-point arrays) plus the clean, evictable file pages of the
        memory maps; per_call is rough (measured 0.1-0.2 GB beyond it with the points of one call). The legacy
        fw_disc.DiscImu ('precomputed', float64, all lines) held the same 7.70 GB. Building 'precomputed' copies
        one line's representatives' float32 intensities at a time (2 nr ny x 4 bytes, M424 0.55 GB) and
        transforms IMU_FFT_BLOCK rows at a time; 'lazy' reads only ic0 (the measured 0.85 GB of the build are
        file pages around those values). A laptop with <= 3 GB: 'lazy' (any dtype) or 'precomputed' float32 per
        line.
        """
        # PP 2026-10-02: new; the terms of disc.DiscImu.memory (one model of the memory; the legacy fw_disc.DiscImu
        # layout is 'precomputed', 'float64', all lines)
        from scipy import fft as sfft
        from .spectral import VelocityGrid
        if fft not in ("precomputed", "lazy"):
            raise ValueError("fft must be 'precomputed' or 'lazy', got {!r}".format(fft))
        try:
            dt = np.dtype(dtype)
        except TypeError:
            dt = None
        if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("dtype must be 'float64' or 'float32', got {!r}".format(dtype))
        try:
            ok = int(chunk) == chunk and chunk >= 1
        except (TypeError, ValueError, OverflowError):
            ok = False
        if not ok:
            raise ValueError("chunk must be a positive integer, got {!r}".format(chunk))
        chunk = int(chunk)
        lgrid = self.grid                                       # as DiscImu: the recorded grid is the default
        if lgrid is not None:
            if grid is None:
                grid = lgrid
            elif not (hasattr(grid, "y") and grid.ny == lgrid.ny and grid.y[0] == lgrid.y[0]
                      and grid.y[-1] == lgrid.y[-1]):
                # PP 2026-10-02: same rule as disc.DiscImu (the grid points must agree; vshift may differ)
                raise ValueError("grid {} differs from the library's recorded grid {}".format(grid, lgrid))
        grid = VelocityGrid() if grid is None else grid
        if not hasattr(grid, "nshift") or grid.ny != self.ny:
            raise ValueError("grid must be a VelocityGrid with ny = {} points (the library's), got {!r}".format(
                self.ny, grid))
        built = self._line_indices(lines)
        nn, K, nl, ny = int(self.unique_nodes().size), int(self.K), int(self.nl), int(self.ny)
        nr, vs = nn * K, int(grid.nshift)
        nv = 2 * vs + 1
        L = int(sfft.next_fast_len(ny + 2 * vs + nv - 1, real=True))
        nf = L // 2 + 1
        own = nl * nr * 8 + len(built) * nr * 8                  # Sflat (all lines), ic0 (built lines)
        src = mapped = 0
        if fft == "precomputed":
            csz = 16 if dt == np.dtype(np.float64) else 8
            own += len(built) * (2 * nr * ny * dt.itemsize + 2 * nr * nf * csz)
        else:
            src = len(built) * 2 * nr * ny * np.dtype(getattr(self.Il, "dtype", np.float64)).itemsize
            if isinstance(self.Il, np.memmap) and isinstance(self.Ic, np.memmap):
                mapped = src
        per_call = nr * nv * 8 + chunk * (nv + 3 * nf) * 16
        if fft == "lazy":
            csz = 16 if dt == np.dtype(np.float64) else 8
            per_call += 4 * chunk * (2 * nf * csz + 2 * ny * 4)
        total = own + src + per_call
        return dict(library=int(own), source=int(src), source_mapped=int(mapped), per_call=int(per_call),
                    total=int(total), anonymous=int(total - mapped), total_gb=total / 1e9,
                    anonymous_gb=(total - mapped) / 1e9, nn=nn, K=K, nr=nr, ny=ny, nv=nv, L=L, nf=nf,
                    built=list(built))

    def _line_indices(self, lines):
        """Sorted unique line indices of a lines argument (ints or recorded names; default all), as DiscImu."""
        if lines is None:
            return tuple(range(self.nl))
        if isinstance(lines, (str, int, np.integer)):
            lines = [lines]
        names = None if self.lines is None else self.lines.names
        out = []
        for x in lines:
            if isinstance(x, str):
                if names is None or x not in names:
                    raise ValueError("unknown line {!r} (library lines: {})".format(x, names))
                x = names.index(x)
            try:
                ok = int(x) == x and 0 <= int(x) < self.nl
            except (TypeError, ValueError, OverflowError):
                ok = False
            if not ok:
                raise ValueError("line indices must be integers in [0, {}), got {!r}".format(self.nl, x))
            out.append(int(x))
        out = tuple(sorted(set(out)))
        if not out:
            raise ValueError("no lines given")
        return out


def _imu_library_from_state(arrays, meta, params, inputs, path=None):
    return ImuLibrary(meta=meta, params=params, inputs=inputs, path=path, **arrays)


def _reopen_imu_library(path, ident):
    """Re-open a memory-mapped library in another process; raises when the file is not the sender's."""
    now = _file_identity(path)
    if now != tuple(ident):
        raise RuntimeError("{} changed since it was opened (device, inode, size, mtime {} -> {})".format(
            path, tuple(ident), now))
    return ImuLibrary.load(path, mmap=True)


def build_imu_library(reps, lib, lines, grid, lref=None, runs_dir=None, suffix="VTV010", frac=1e-3, check=True,
                      layout=IMU_LAYOUT, nmu=4001, allow_missing=False, progress=None, teff_from_meta="auto"):
    """
    Intensity library from the OUT_IMU files of the representatives (port of fw_imu_library.py
    --stage build).

    For every bin with a representative and every line: read OUT_IMU.<line>_<suffix>, keep the rays with
    p <= R_max (:func:`r_outer`), store s = p / R_max and interpolate each ray's continuum and line
    intensity from the model's wavelengths onto the grid (y = c ln(lambda / lref), :func:`~ppmpy.synspec.
    spectral.interp_rows`, constant beyond the band; cast to float32). Bins without a representative (empty
    bins) get src = the nearest bin with one (the lower one on a tie) and copies of its arrays.

    Parameters
    ----------
    reps: Representatives, str or mapping
        Representatives (:func:`select_representatives`), a representatives.txt (read with
        ``teff_from_meta``), or {bin: (idx, teff[, dir])}. Every representative's T_eff' must lie in its bin
        of ``lib`` (:func:`teff_bins`, which puts T_eff' beyond the edges into the end bins).
    lib: FluxLibrary or mapping
        The flux library whose bins are used (edges, tmean, count are copied into the result).
    lines: LineSet or sequence of str
        Line names of the OUT_IMU files, in library order.
    grid: VelocityGrid or np.ndarray
        Velocity grid y [km/s].
    lref: array-like, optional
        (nl,) velocity zero points [A]; default ``lines.lref`` for a LineSet.
    runs_dir: str, optional
        Directory of the reruns of the modified pformalsol (M424 /scratch/ppathak/fastwind_imu/runs); the
        files of model idx are in ``runs_dir/layout.format(idx=idx)``. None: the representatives' own
        directories.
    suffix: str
        File suffix (OUT_IMU.<line>_<suffix>, OUT.<line>_<suffix>).
    frac: float
        :func:`r_outer` threshold.
    check: bool
        Flux check per model and line (fw_imu_library.py step 4): :func:`flux_from_rays` vs the model's
        own OUT profile (F/F_c, on the OUT wavelengths), max |dF| and dEW = int (1 - F) dlambda difference
        [A] with the OUT wavelengths.
    layout: str
        Model directory below runs_dir (:data:`IMU_LAYOUT`).
    nmu: int
        mu points of the flux check.
    allow_missing: bool
        False (default): every filled bin of lib needs a representative (legacy assert). True: filled
        bins without one are treated like empty bins (src = nearest bin with a representative).
    progress: callable, optional
        progress(bins_done, bins_total) after each representative.
    teff_from_meta: bool or 'auto'
        For reps given as a representatives.txt: where teff_rep comes from (:meth:`Representatives.read`
        ``from_meta``). 'auto' (default): the full-precision value of ``<dir>/meta.txt`` where it exists (as
        the legacy build, which took it from its candidate scan), else the file's 3 decimals. False: the
        file's value.

    Returns
    -------
    ImuLibrary
        With ``params`` (frac, suffix, layout, nmu, lines, lref, grid ends ``y0``, ``y1``, ``ny`` and step
        ``dv``, ``grid`` = VelocityGrid.to_dict() for a VelocityGrid (else None), K, ...; :attr:`ImuLibrary.lines`,
        :attr:`ImuLibrary.lref` and :attr:`ImuLibrary.grid` come from them) and ``inputs`` (runs_dir,
        representatives file, flux library file where known; absolute paths).
    checks: dict
        ``lines``, ``bins`` (n,) bins with a representative, ``nnode``, ``rmax`` (n, nl), ``max_dF``, ``dEW``
        (n, nl; None without check), and ``summary`` {line: dict(max_dF, max_abs_dEW, nnode_min,
        nnode_max, rmax_min, rmax_max)} (the legacy log line).

    Raises
    ------
    ValueError
        Filled bins without a representative (unless allow_missing), representatives outside the bins or
        whose T_eff' lies in another bin (e.g. a representatives.txt made for other edges), missing OUT_IMU
        files and, with check, OUT files (all checked before any file is parsed), OUT and OUT_IMU with
        different numbers of rows (check).

    Memory and time
    ---------------
    One (bin, line) file is parsed at a time; only its kept rays are held until the output arrays are
    filled (M424: 909 files of 0.36 MB, kept rays ~0.1 GB in float64), then the output arrays
    (nb x nl x K x ny float32 twice: 1.91 GB for M424) plus one (K, ny) interpolation at a time.
    M424 on the Trillium login node (2026-10-02, warm page cache): 19.5-21.5 s, peak RSS 2.04-2.09 GB (fresh
    process); ``save`` of the 1.91 GB file a few s more. Serial: 909 small files, no pool needed.

    Validation
    ----------
    M424: build(representatives.txt, library_dT10.npz, runs) equals
    /scratch/ppathak/fastwind_imu/imu_library_dT10.npz member for member, bit for bit, and ``save(meta=False)``
    writes the same bytes; synthetic runs: equal to the frozen fw_imu_library.py build lines, also for meta.txt
    values with more than 3 decimals (teff_from_meta='auto'; test_imulib.py). Bitwise agreement with the stored
    product assumes the production's np.log (AVX512_SKX SVML; module notes).
    """
    # PP 2026-10-02: ported from fw_imu_library.py:84-134 (step 2 file check, step 3 library, step 4 flux check, empty
    # bins, idx_rep/teff_rep); new: per-(bin, line) parsing with only the kept rays held, explicit errors instead of
    # asserts, allow_missing, checks returned instead of logged, up-front checks of the OUT files (check=True) and of
    # the representatives' bins, teff_from_meta for a representatives.txt (legacy: meta.txt values)
    from .fwresults import read_out, read_out_imu
    y = _grid_y(grid)
    names = _line_names(lines)
    if lref is None and not hasattr(lines, "lref"):
        raise ValueError("lref is required unless lines is a LineSet")
    lr = _lref(lines if lref is None else lref)
    if lr.size != len(names):
        raise ValueError("need one reference wavelength per line")
    nl = len(names)
    edges = np.asarray(lib["edges"])
    tmean = np.asarray(lib["tmean"])
    count = np.asarray(lib["count"])
    nb = edges.size - 1
    if tmean.shape != (nb,) or count.shape != (nb,):
        raise ValueError("library edges, tmean and count do not match")
    reps_path = os.path.abspath(os.fspath(reps)) if isinstance(reps, (str, os.PathLike)) else None
    reps = _as_representatives(reps, from_meta=teff_from_meta)
    if len(reps) == 0:
        raise ValueError("no representatives")
    if reps.bins.min() < 0 or reps.bins.max() >= nb:
        raise ValueError("representative bins must lie in 0 .. nb - 1 = {}".format(nb - 1))
    wrong = reps.bins[teff_bins(reps.teff, edges) != reps.bins]
    if wrong.size:
        raise ValueError("the T_eff' of the representatives of bins {}{} lie outside their bins (representatives "
                         "made for other bin edges?)".format(wrong[:20].tolist(), " ..." if wrong.size > 20 else ""))
    lost = [int(b) for b in np.setdiff1d(np.flatnonzero(count > 0), reps.bins)]
    if lost and not allow_missing:
        raise ValueError("{} filled bins have no representative: {}{}".format(len(lost), lost[:20],
                                                                           " ..." if len(lost) > 20 else ""))
    mdir = reps.model_dirs(runs_dir, layout=layout)
    need = ["OUT_IMU.{}_{}".format(ln, suffix) for ln in names]
    if check:
        need += ["OUT.{}_{}".format(ln, suffix) for ln in names]
    lack = {int(b): [f for f in need if not os.path.exists(os.path.join(mdir[b], f))] for b in reps.bins}
    bad = [b for b in lack if lack[b]]
    if bad:
        kinds = [k for k in ("OUT_IMU", "OUT") if any(f.split(".")[0] == k for b in bad for f in lack[b])]
        raise ValueError("{} representatives have no {} files ({}): bins {}{}, e.g. {} lacks {}".format(
            len(bad), "/".join(kinds), "run the modified pformalsol first" if "OUT_IMU" in kinds else
            "needed by the flux check; check=False skips it", bad[:20], " ..." if len(bad) > 20 else "",
            mdir[bad[0]], lack[bad[0]]))

    # pass 1: the kept rays of every (bin, line), and the flux check
    nrep = len(reps)
    per = {}
    K = 0
    nn_r, rm_r = np.zeros((nrep, nl), np.int64), np.zeros((nrep, nl))
    dF_r = np.full((nrep, nl), np.nan) if check else None
    dEW_r = np.full((nrep, nl), np.nan) if check else None
    for r, b in enumerate(reps.bins):
        b = int(b)
        m = mdir[b]
        for j, ln in enumerate(names):
            lam, p, Ic, Il = read_out_imu(os.path.join(m, "OUT_IMU.{}_{}".format(ln, suffix)))
            rmax = r_outer(p, Ic, frac)
            keep = p <= rmax
            n = int(keep.sum())
            if n < 2:
                raise ValueError("{}: fewer than 2 rays inside R_max for {}".format(m, ln))
            pk, Ick, Ilk = p[keep], Ic[:, keep], Il[:, keep]
            per[b, j] = (lam.copy(), pk, rmax, Ick, Ilk)
            K = max(K, n)
            nn_r[r, j], rm_r[r, j] = n, rmax
            if check:
                fn = flux_from_rays(pk, Ick, Ilk, rmax, nmu=nmu)
                out = read_out(os.path.join(m, "OUT.{}_{}".format(ln, suffix)))
                w0, f0 = out["lam"], out["fnorm"]
                if w0.size != lam.size:
                    raise ValueError("{}: OUT and OUT_IMU of {} have different numbers of rows ({} vs {})".format(
                        m, ln, w0.size, lam.size))
                dF_r[r, j] = np.abs(fn - f0).max()
                dEW_r[r, j] = _trapz(1 - fn, w0) - _trapz(1 - f0, w0)
            del lam, p, Ic, Il
        if progress is not None:
            progress(r + 1, 2 * nrep)

    # pass 2: the arrays (one (K, ny) interpolation at a time)
    ny = y.size
    S = np.full((nb, nl, K), np.nan)
    IC = np.zeros((nb, nl, K, ny), np.float32)
    IL = np.zeros((nb, nl, K, ny), np.float32)
    RM = np.zeros((nb, nl))
    NN = np.zeros((nb, nl), np.int64)
    for r, b in enumerate(reps.bins):
        b = int(b)
        for j in range(nl):
            lam, p, rmax, Ic, Il = per.pop((b, j))
            yl = y_of_lam(lam, lr[j])
            n = p.size
            S[b, j, :n] = p / rmax
            RM[b, j], NN[b, j] = rmax, n
            IC[b, j, :n] = interp_rows(np.repeat(yl[None, :], n, 0), Ic.T, y)
            IL[b, j, :n] = interp_rows(np.repeat(yl[None, :], n, 0), Il.T, y)
        if progress is not None:
            progress(nrep + r + 1, 2 * nrep)

    # bins without a representative: the nearest bin with one
    filled = reps.bins
    src = np.array([filled[np.argmin(np.abs(filled - b))] for b in range(nb)], dtype=np.int64)
    for b in range(nb):
        if src[b] != b:
            S[b], IC[b], IL[b], RM[b], NN[b] = S[src[b]], IC[src[b]], IL[src[b]], RM[src[b]], NN[src[b]]
    pos = np.searchsorted(filled, src)
    idx_rep = reps.idx[pos].astype(np.int64)
    teff_rep = reps.teff[pos].astype(np.float64)

    summary = {}
    for j, ln in enumerate(names):
        summary[ln] = dict(nnode_min=int(nn_r[:, j].min()), nnode_max=int(nn_r[:, j].max()),
                           rmax_min=float(rm_r[:, j].min()), rmax_max=float(rm_r[:, j].max()),
                           max_dF=float(dF_r[:, j].max()) if check else None,
                           max_abs_dEW=float(np.abs(dEW_r[:, j]).max()) if check else None)
    checks = dict(lines=list(names), bins=filled.copy(), nnode=nn_r, rmax=rm_r, max_dF=dF_r, dEW=dEW_r,
                  summary=summary)
    params = dict(frac=float(frac), suffix=suffix, layout=layout, nmu=int(nmu), lines=list(names), lref=lr.tolist(),
                  ny=int(ny), y0=float(y[0]), y1=float(y[-1]), dv=float(y[1] - y[0]),
                  grid=grid.to_dict() if hasattr(grid, "to_dict") else None, K=int(K), n_rep=int(nrep),
                  allow_missing=bool(allow_missing))
    inputs = {}
    if runs_dir is not None:
        inputs["runs"] = os.path.abspath(os.fspath(runs_dir))
    if reps_path is not None:
        inputs["representatives"] = reps_path
    src_lib = getattr(lib, "inputs", None) or {}
    for k in ("source", "profiles"):
        if src_lib.get(k):
            inputs["flux_library_" + k] = src_lib[k]
    imu = ImuLibrary(edges, tmean, count, src, idx_rep, teff_rep, RM, NN, S, IC, IL, params=params, inputs=inputs)
    return imu, checks
