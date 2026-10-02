"""
All dumps: disc-integrated line profiles of every moms dump for a set of lines of sight, and their assembly into
one time series (port of the project's fw_disc_dumps.py and fw_disc_collect.py).

The per-point models of one dump (M424: dump 3200) differ only in T_eff', so they are a T_eff' library for every
dump: each dump contributes its own T_eff' (interpolated between library nodes) and its own velocities (Doppler
shifts) on the same sphere of points. The per-dump sphere samples (sphere_sample.py: ``teff``, ``ur``, ``uth``,
``uph`` per point, float32, and ``t_s``) are the only per-dump input; no new FASTWIND runs.

* :func:`load_sample`: the samples of one dump, as float64.
* :func:`disc_dump`: one dump with a disc integrator (:class:`ppmpy.synspec.disc.DiscFlux`, the flux method, or
  :class:`ppmpy.synspec.disc.DiscImu`, the intensity method): profiles with and without Doppler shifts,
  diagnostics, weighted velocity moments, points beyond the node range and their visible weight, clipped shifts,
  T_eff' statistics.
* :func:`run_disc_dumps`: all dumps: restartable, one atomic .npz per dump, worker processes ('fork' or 'spawn'),
  a watchdog for killed workers, dumps split over ranks (nodes).
* :func:`collect_timeseries`: the per-dump files -> one ``<name>_timeseries.npz``, streamed dump by dump.
* :func:`flux_integrator`: the integrator factory of the flux runs (library file -> nodes -> DiscFlux).
* :func:`imu_integrator`: the integrator factory of the intensity runs (intensity library file -> DiscImu; the
  default options are the production 'imu' run, fft='lazy' / dtype='float32' the low-memory modes).

Files
-----
``<outdir>/<name>/dNNNN.npz``
    One per dump: the members :data:`DUMP_KEYS` in this order with the legacy dtypes (F, F0 float32), then the
    ``extra_fields``, then '_meta' with ``meta=True``. Without extras and '_meta' (the default) the file equals
    the one fw_disc_dumps.py writes for the same numbers byte for byte. Written to ``dNNNN.tmp<pid>.npz`` and
    renamed (atomic); a worker killed while writing leaves that temporary behind (listed, never read).
``<outdir>/<name>/_run.json`` (:data:`RUN_FILE`)
    The parameters that determine the results (integrator, grid, lines, lines of sight, points, samples,
    diagnostics, extra fields; ``params``), the provenance of the run that created the directory (``info``) and
    its writer (host, pid). A later run into the same directory (restart, another rank) must have the same
    ``params``, else ValueError: restarts never mix two configurations. The integrator enters ``params`` through
    a fingerprint of its inputs (factory name, arguments, sha256 of the files and arrays among them) where the
    arguments allow one, else through the sha256 of its computed arrays; the latter is always kept in ``info``
    (a restart whose integrator has the same inputs but other bits, e.g. on another CPU, is accepted with a
    warning). The record is created exclusively (temporary file, then a hard link), so of two runs starting at
    once only one creates it and the other compares against it. A record whose run wrote nothing (no per-dump
    files or temporaries) and whose writer is known to be gone (this host, process not running) is replaced by a
    run of another configuration. Directories without a record (the legacy outputs) are checked against the run
    constants stored in an existing dump file instead, and adopted only by integrators with the legacy options (a
    :class:`ppmpy.synspec.disc.DiscImu` in another mode, e.g. lazy or float32, stores the same constants but
    rounds differently, so it is refused there; :func:`run_fields` gives each mode its own default name).
``<outdir>/<name>_timeseries.npz``
    The members :data:`TIMESERIES_KEYS` in this order (legacy dtypes, including the float64 n_lo, n_hi), then
    '_meta' with ``meta=True``. Byte for byte the file of fw_disc_collect.py for the same per-dump files.

Conventions
-----------
* Lines of sight point from the star to the observer; v = u . n > 0 towards the observer (blueshift),
  lambda_obs = lambda (1 - v/c) (:mod:`ppmpy.synspec.conventions`); mu = r_hat . n > 0 is the visible hemisphere.
* Projections mu, theta_hat . n, phi_hat . n of the points: :func:`ppmpy.synspec.sphere.project_los` with
  ``method='matmul'`` (the dgemm of fw_disc_dumps.py); v: :func:`ppmpy.synspec.sphere.los_velocity`.
* Diagnostics: :func:`ppmpy.synspec.diagnostics.diagnostics_array` with the current fw_disc.diagnostics options
  (EW with the factor lambda/lref, window |y| <= vwin); the default window is the whole grid (legacy
  ``vwin = VY``, stored as diag_vwin = grid.vmax): the default 400 km/s of the dump-3200 figures would cut the
  Stark wings of lambda4026/4200.
* Points beyond the node range take the end node (clamped T_eff' interpolation) and are counted in n_lo, n_hi;
  wout is their share of the visible weight sum(mu). Shifts beyond +-vshift are clipped and counted (n_clip).
* BLAS threads: the node profiles of :func:`ppmpy.synspec.library.lib_nodes` with smoothing or a correction
  are sums through BLAS (dgemv) whose last bits depend on the number of BLAS threads; the M424 products were
  made with 1. :func:`flux_integrator` therefore builds the nodes with the loaded BLAS limited to 1 thread
  (restored afterwards), and :func:`run_disc_dumps` builds the integrator and computes in this process under
  the same limit, the thread count of its pool workers; so the results do not depend on OMP_NUM_THREADS /
  OPENBLAS_NUM_THREADS, on nproc or on the start method. The limit reaches the OpenBLAS of the numpy and scipy
  wheels (or, with threadpoolctl installed, every BLAS it knows); another BLAS (e.g. conda's MKL without
  threadpoolctl) follows its environment variables, so set them to 1 there for the bitwise M424 results. The
  per-dump computation (DiscFlux call, diagnostics, 'matmul' projections) does not depend on the thread count
  (checked with 1 and 8 OpenBLAS threads).

Validation
----------
tests/synspec/test_dumps.py. Synthetic (M424 grid, toy library and sphere, 6 dumps): the per-dump files equal,
byte for byte, those of the frozen fw_disc_dumps.py process() (its source lines) for the flux, smoothed and
lamfix variants; the time series equals, byte for byte, that of the frozen fw_disc_collect.py (run as a script);
serial, 'fork' and 'spawn' runs give identical files, also in a process with 4 OpenBLAS threads; restarts skip,
ranks split as the legacy driver. Intensity method (toy intensity library on the M424 grid): the files of
:func:`imu_integrator` runs are byte for byte those of the frozen fw_disc_dumps.py process() with the frozen
fw_disc.DiscImu, serial = 'fork' = 'spawn' (each spawn worker builds its DiscImu and must match the parent's
fingerprint), for the default and the lazy modes, with and without batched lines of sight. M424
(marker m424): dumps 3200-3209 of the flux, flux_sm335 and flux_lamfix runs equal the production files of
fw_disc_dumps.py byte for byte, and their time series the corresponding rows of the production
``*_timeseries.npz``; so do dumps 3200-3209 of the imu run (:func:`imu_integrator`, 4 'fork' and 2 'spawn'
workers; slow). Once at full scale (2026-10-01, scratch check, not a test): all 1601 dumps of the flux run
equal the production files byte for byte, and their collected time series equals the production
flux_timeseries.npz byte for byte (sha256), with the same printed summary. All of this under numpy 1.26 on the
AVX512 Trillium nodes; see the library and sphere module notes for what is hardware-dependent.

PP 2026-10-01: ported from the project's fw_disc_dumps.py (process(), the driver) and fw_disc_collect.py; see the
provenance comments per function.
"""
import contextlib
import datetime
import errno
import hashlib
import json
import math
import os
import re
import shutil
import socket
import tempfile
import time
import warnings

import numpy as np

from . import parallel as par
from .diagnostics import DIAG_KEYS, diagnostics_array
from .disc import PAD_TOL, WORKER_MALLOC, DiscFlux, DiscImu, _spill, _tune_malloc
from .fwresults import _NpzStream, _meta_array, _npz_layout, _pid_alive
from .io import file_identity, make_meta, npz_member_memmap, save_npz
from .library import FluxLibrary, _share, _unshare, coverage, lib_nodes
from .spectral import LineSet, VelocityGrid
from .sphere import _los_vectors, los_velocity, project_los

__all__ = ["load_sample", "sample_path", "dump_path", "disc_dump", "dump_arrays", "run_fields", "flux_integrator",
           "imu_integrator", "run_disc_dumps", "collect_timeseries", "read_run_record", "DUMP_KEYS", "TIMESERIES_KEYS",
           "RUN_FILE", "SAMPLE_PATTERN", "DUMP_PATTERN", "PROJECT_CHUNK", "LREF_TOL", "REBUILD_WARN"]

DUMP_KEYS = ("dump", "t_s", "method", "name", "lamfix", "smooth", "F", "F0", "diag_keys", "diag_vwin", "diag_F",
             "diag_F0", "vmean_w", "sigma_w", "n_lo", "n_hi", "wout", "n_clip", "teff_mean", "teff_std", "teff_min",
             "teff_max", "nmin", "node_range")
"""Members of a per-dump file, in the order of fw_disc_dumps.py's np.savez call (:func:`dump_arrays`)."""

TIMESERIES_KEYS = ("dumps", "t_s", "Y", "LREF", "los", "method", "name", "diag_vwin", "F", "F0", "diag_keys", "diag_F",
                   "diag_F0", "vmean_w", "sigma_w", "n_lo", "n_hi", "wout", "n_clip", "teff_mean", "teff_std",
                   "teff_min", "teff_max", "node_range")
"""Members of a time-series file, in the order of fw_disc_collect.py's np.savez call (:func:`collect_timeseries`)."""

SAMPLE_PATTERN = "d{dump:04d}.npz"
"""File name of a dump's sphere samples in the samples directory (sphere_sample.py)."""

DUMP_PATTERN = "d{dump:04d}.npz"
"""File name of a dump's disc-integrated profiles in <outdir>/<name> (fw_disc_dumps.py)."""

RUN_FILE = "_run.json"
"""Run record in <outdir>/<name> (module notes)."""

PROJECT_CHUNK = 1 << 18
"""Points per block of :func:`ppmpy.synspec.sphere.project_los` (bounds its temporaries; changes no bit)."""

_DUMP_RE = re.compile(r"^d(\d+)\.npz$")
_TMP_RE = re.compile(r"^d\d+\.tmp\d+\.npz$")
_RUN_CONST = ("method", "name", "lamfix", "smooth", "nmin", "diag_vwin", "diag_keys", "node_range")
_RUN_KIND = "synspec.disc_dumps.run"
_ACTIVE_DIRS = {}             # per-dump directory -> run_disc_dumps calls of this process working on it
LREF_TOL = 1e-12
"""Largest difference [A] between a given lref and the integrator's own (as Y / LREF / los in the collect)."""


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------
def _logger(log, T0):
    def _log(msg):
        if log is not None:
            log("[{:7.1f} s] {}".format(time.time() - T0, msg))
    return _log



def _modname(obj, default="?"):
    """Module name of obj for run records, with the spawn alias '__mp_main__' mapped to '__main__' (a class defined in
    the calling script is '__main__' in the parent but '__mp_main__' in spawn workers)."""
    # PP 2026-10-01: migration finding: integrators defined in a script failed the spawn worker check ('differs in [cls]')
    mod = getattr(obj, "__module__", None) or default
    return "__main__" if mod == "__mp_main__" else mod

def _norm(x):
    """JSON round trip (tuples -> lists, numpy scalars -> numbers): the form in which records are compared."""
    return json.loads(json.dumps(x, sort_keys=True, default=_json_default))


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _dump_list(dumps):
    """Dump numbers as a list of int (order kept); repeated dumps raise."""
    out = []
    for d in dumps:
        try:
            ok = int(d) == d and d >= 0
        except (TypeError, ValueError, OverflowError):
            ok = False
        if not ok:
            raise ValueError("dump numbers must be non-negative integers, got {!r}".format(d))
        out.append(int(d))
    if len(set(out)) != len(out):
        raise ValueError("repeated dump numbers in {}".format(out[:20]))
    return out


def _check_name(name):
    bad_sep = os.sep in name or (os.altsep and os.altsep in name) if isinstance(name, str) else True
    if not isinstance(name, str) or not name or name in (".", "..") or bad_sep:
        raise ValueError("name must be a plain directory name, got {!r}".format(name))
    return name


def _diag_keys(keys):
    keys = tuple(str(k) for k in keys)
    bad = [k for k in keys if k not in DIAG_KEYS]
    if bad or not keys:
        raise ValueError("diag_keys must be a non-empty subset of {}, got {}".format(DIAG_KEYS, keys))
    return keys


def _integ_grid(integ, grid=None):
    """The VelocityGrid of the profiles: ``grid`` if given, else the integrator's ``grid`` attribute."""
    g = grid if grid is not None else getattr(integ, "grid", None)
    if isinstance(g, dict):
        g = VelocityGrid(**g)
    if not isinstance(g, VelocityGrid):
        raise ValueError("the velocity grid is unknown: the integrator has no 'grid' (VelocityGrid) attribute and no "
                         "grid was given")
    return g


def _lref_of(src):
    """(names or None, lref float64 (nl,)) of a LineSet or an array of reference wavelengths."""
    names = getattr(src, "names", None)
    lr = np.atleast_1d(np.asarray(getattr(src, "lref", src), dtype=np.float64))
    if lr.ndim != 1 or lr.size == 0:
        raise ValueError("lref must be 1-D (one wavelength per line), got shape {}".format(lr.shape))
    return (list(names) if names is not None else None), lr


def _integ_lref(integ, lref=None):
    """
    (names or None, lref float64 (nl,)) of a run: ``lref`` (LineSet or array) if given, else the integrator's own
    ('lines' or 'lref' attribute, e.g. set by :func:`flux_integrator` from the library's build parameters). When
    both exist they must agree to :data:`LREF_TOL` (catches e.g. a LineSet in another line order); the given one
    is used, with the integrator's names if it has none.
    """
    # PP 2026-10-01: reviewer: lref could be neither defaulted nor checked (a LineSet in another order was accepted)
    own = None
    if integ is not None:
        own = getattr(integ, "lines", None)
        if own is None:
            own = getattr(integ, "lref", None)
    if lref is None and own is None:
        raise ValueError("lref is required (a LineSet or one reference wavelength per line): the EW needs it and the "
                         "integrator does not carry it")
    names, lr = _lref_of(lref if lref is not None else own)
    if lref is not None and own is not None:
        onames, olr = _lref_of(own)
        if olr.shape != lr.shape or not np.allclose(olr, lr, rtol=0.0, atol=LREF_TOL):
            raise ValueError("lref {} differs from the integrator's {} (another line order or other lines?)".format(
                lr.tolist(), olr.tolist()))
        if names is None:
            names = onames
    return names, lr


def _integ_nl(integ):
    """Number of lines of an integrator (``nl`` attribute, else the second axis of a 2-D ``fc``), or None."""
    nl = getattr(integ, "nl", None)
    if isinstance(nl, (int, np.integer)) and not isinstance(nl, bool):
        return int(nl)
    fc = getattr(integ, "fc", None)
    if fc is not None and np.ndim(fc) == 2:
        return int(np.shape(fc)[1])
    return None


def _use_batch(integ, batch):
    """Whether :func:`disc_dump` integrates all lines of sight in one ``integrate_los`` call: ``batch`` if given (True
    needs the method), else True for integrators whose ``fft`` attribute is 'lazy' (a lazy DiscImu transforms each
    library row once per call)."""
    has = callable(getattr(integ, "integrate_los", None))
    if batch is None:
        return has and getattr(integ, "fft", None) == "lazy"
    if batch and not has:
        raise ValueError("batch=True needs an integrator with an integrate_los method ({} has none)".format(
            type(integ).__name__))
    return bool(batch)


def _sha256(*arrays):
    """sha256 of the dtype, shape and bytes of C-contiguous copies of arrays."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        if a.dtype.hasobject:
            raise ValueError("cannot hash an object array")
        h.update("{}:{}".format(a.dtype.str, a.shape).encode())
        h.update(memoryview(a.reshape(-1).view(np.uint8)))
    return h.hexdigest()


def _file_sha256(path, block=1 << 24):
    """sha256 of a file's content (read in blocks of 16 MiB)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _integ_info(integ):
    """
    What identifies an integrator's results: class, nodes (number, T_eff' range), node_params, a ``fingerprint()``
    record if the integrator defines one ('custom'), and the sha256 of its arrays: t, fc and P for a
    :class:`ppmpy.synspec.disc.DiscFlux`; for other integrators t with a ``fingerprint()``, else every ndarray
    attribute (``vars(integ)``, by name). An integrator with neither raises ValueError. The 'lref' entry of a
    :class:`ppmpy.synspec.disc.DiscImu` fingerprint is left out: it labels the lines without changing the profiles,
    and the run's params record the lref in force.
    """
    # PP 2026-10-01: reviewer: a non-DiscFlux integrator (the planned DiscImu) was identified by its node T_eff' only
    t = np.asarray(integ.t, dtype=np.float64)
    cls = type(integ)
    custom = getattr(integ, "fingerprint", None)
    if isinstance(integ, DiscFlux):
        arrays = [("t", integ.t), ("fc", integ.fc), ("P", integ.P)]
    elif callable(custom):
        arrays = [("t", integ.t)]
    else:
        try:
            attrs = vars(integ)
        except TypeError:
            attrs = {}
        arrays = sorted(((k, v) for k, v in attrs.items() if isinstance(v, np.ndarray)), key=lambda kv: kv[0])
        if not arrays:
            raise ValueError("cannot identify the integrator {}: define a fingerprint() method (a JSON-able record of "
                             "everything that determines its results) or keep its state in ndarray attributes".format(
                                 cls.__name__))
    info = dict(cls="{}.{}".format(_modname(cls), cls.__qualname__), nn=int(t.size),
                t_range=[float(t[0]), float(t[-1])], node_params=_norm(dict(getattr(integ, "node_params", None) or {})),
                arrays=[k for k, _ in arrays], sha256=_sha256(*(np.asarray(v) for _, v in arrays)))
    if callable(custom):
        rec = _norm(custom())
        if isinstance(integ, DiscImu) and isinstance(rec, dict):
            # PP 2026-10-02: reviewer: lref attached by imu_integrator entered the identity, so moving it between
            # run_disc_dumps(lref=...) and factory_kwargs refused a restart of byte-identical files
            rec.pop("lref", None)
        info["custom"] = rec
    return info


class _NoFingerprint(Exception):
    """A factory argument whose content cannot be fingerprinted (the integrator's arrays are hashed instead)."""


def _fingerprint_value(x):
    """JSON-able fingerprint of a factory argument: scalars and plain strings as they are, files by the sha256 of
    their content (not their path), arrays / VelocityGrid / LineSet / FluxLibrary by their values, lists, tuples and
    dicts with str keys element by element. Anything else (a directory, an object) raises _NoFingerprint."""
    if x is None or isinstance(x, (bool, int, float)):
        return x
    if isinstance(x, (str, os.PathLike)):
        p = os.fspath(x)
        if not isinstance(p, str):
            raise _NoFingerprint("bytes path")
        if os.path.isfile(p):
            return dict(file_sha256=_file_sha256(p))
        if os.path.exists(p):
            raise _NoFingerprint("directory {}".format(p))
        return str(p)
    if isinstance(x, np.generic) and not isinstance(x, (np.void, np.object_)):
        return _norm(x)
    if isinstance(x, np.ndarray):
        if x.dtype.hasobject:
            raise _NoFingerprint("object array")
        return dict(array_sha256=_sha256(x))
    if isinstance(x, VelocityGrid):
        return dict(VelocityGrid=x.to_dict())
    if isinstance(x, LineSet):
        return dict(LineSet=dict(names=list(x.names), lref=x.lref.tolist()))
    if isinstance(x, FluxLibrary):
        return dict(FluxLibrary=_sha256(x.edges, x.tmean, x.count, x.prof, x.fc), dT=x.dT)
    if isinstance(x, (list, tuple)):
        return [_fingerprint_value(v) for v in x]
    if isinstance(x, dict) and all(isinstance(k, str) for k in x):
        return {k: _fingerprint_value(x[k]) for k in sorted(x)}
    raise _NoFingerprint(type(x).__name__)


def _inputs_fingerprint(factory, args, kwargs):
    """
    Fingerprint of an integrator's inputs: factory (module.qualname) and its arguments (:func:`_fingerprint_value`),
    or None when the factory is a local function or lambda (its captured state is unknown) or an argument cannot be
    fingerprinted. Unlike the sha256 of the integrator's computed arrays it does not depend on the CPU or the BLAS.
    The ``lref`` argument of :func:`imu_integrator` (the 7th) is left out (it does not change the profiles; the
    run's params record the lref in force).
    """
    # PP 2026-10-01: reviewer: identify a run by its inputs, so that a restart on another CPU / BLAS is not refused
    qual = getattr(factory, "__qualname__", None)
    mod = _modname(factory, None) if getattr(factory, "__module__", None) else None
    if not qual or not mod or "<" in qual:
        return None
    if factory is imu_integrator:
        # PP 2026-10-02: reviewer: lref only labels the lines (the run's params record it), so giving it to the factory
        # or to run_disc_dumps is the same run
        args, kwargs = tuple(args)[:6], {k: v for k, v in dict(kwargs).items() if k != "lref"}
        # PP 2026-10-02: migration finding: an explicit default (fft='precomputed') and an omitted one are the same run:
        # bind to the signature with the defaults filled in (lref removed above)
        import inspect
        try:
            sig = inspect.signature(imu_integrator)
            b = sig.bind_partial(*args, **kwargs)
            kw = dict(b.arguments)
            kw.pop("lref", None)
            lib = kw.pop("library", None)
            simple = (str, int, float, bool, type(None))
            kw = {k: v for k, v in kw.items()                   # drop explicit defaults (simple values only)
                  if not (isinstance(v, simple) and isinstance(sig.parameters[k].default, simple)
                          and v == sig.parameters[k].default)}
            args, kwargs = ((lib,) if "library" in b.arguments else ()), kw
        except TypeError:
            pass
    try:
        return _norm(dict(factory="{}.{}".format(mod, qual), args=_fingerprint_value(list(args)),
                          kwargs=_fingerprint_value(dict(kwargs))))
    except _NoFingerprint:
        return None


def _extra_desc(extra_fields):
    """The record of extra_fields: a callable by name; a dict by name, dtype, shape and sha256 of every value, in
    order (so a restart with other values is refused)."""
    # PP 2026-10-01: reviewer: only the names of constant extras were recorded, so other values were mixed in
    if extra_fields is None:
        return None
    if callable(extra_fields):
        return "callable {}.{}".format(_modname(extra_fields),
                                       getattr(extra_fields, "__qualname__", type(extra_fields).__name__))
    out = []
    for k, v in extra_fields.items():
        a = np.asarray(v)
        out.append([k, a.dtype.str, list(a.shape), _sha256(a)])
    return out


def _check_extra(extra):
    """Validate extra fields (a dict, or the dict returned by a callable): names, and no object arrays (np.savez
    would pickle them)."""
    if extra is None:
        return {}
    if not isinstance(extra, dict):
        raise TypeError("extra_fields must be a dict or a callable returning a dict, got {}".format(
            type(extra).__name__))
    clash = [k for k in extra if k in DUMP_KEYS or k == "_meta" or not isinstance(k, str) or not k]
    if clash:
        raise ValueError("extra field names must be non-empty strings other than the legacy members and '_meta': {}"
                         .format(clash))
    obj = [k for k, v in extra.items() if np.asarray(v).dtype.hasobject]
    if obj:
        raise ValueError("extra fields must be numeric, boolean or string arrays (no objects, which np.savez would "
                         "pickle): {}".format(obj))
    return extra


_NO_LINK_ERRNO = tuple(getattr(errno, k) for k in ("EPERM", "ENOTSUP", "EOPNOTSUPP", "EXDEV", "EMLINK", "ENOSYS")
                       if hasattr(errno, k))


def _tmp_name(path):
    """A temporary next to path, unique over hosts (shared file systems) and processes."""
    return "{}.tmp.{}.{}".format(path, socket.gethostname(), os.getpid())


def _dump_json(obj, f):
    json.dump(obj, f, indent=1, sort_keys=True, default=_json_default)
    f.write("\n")
    f.flush()


def _write_json_atomic(path, obj):
    """Write (or replace) path atomically: temporary, then os.replace."""
    tmp = _tmp_name(path)
    try:
        with open(tmp, "w") as f:
            _dump_json(obj, f)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _create_json_excl(path, obj):
    """
    Create path holding obj unless it exists; True if created. The complete content is written to a temporary and
    hard-linked to path (fails if path exists; atomic, also on NFS), so a reader never sees a partial file. Where
    hard links are not supported, path is created with O_CREAT | O_EXCL and written (a reader may then see it
    empty for a moment: :func:`read_run_record` retries).
    """
    # PP 2026-10-01: reviewer: check-then-write of the record was not atomic (two runs could both write theirs)
    tmp = _tmp_name(path)
    try:
        with open(tmp, "w") as f:
            _dump_json(obj, f)
        try:
            os.link(tmp, path)
            return True
        except FileExistsError:
            return False
        except OSError as e:
            if e.errno not in _NO_LINK_ERRNO:
                raise
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f:
            _dump_json(obj, f)
        return True
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


@contextlib.contextmanager
def _blas_limit(n=1):
    """
    Limit the BLAS libraries loaded in this process to n threads inside the block and restore each one's previous
    thread count afterwards (also on an exception). Through threadpoolctl when it is installed, else the OpenBLAS
    bundled with the numpy and scipy wheels (ctypes, :func:`ppmpy.synspec.parallel.limit_blas_threads`). Yields
    {library path: previous threads}; empty when no library could be controlled (e.g. MKL without threadpoolctl:
    it then follows MKL_NUM_THREADS / OMP_NUM_THREADS). A BLAS first loaded inside the block (e.g. scipy's
    OpenBLAS on the first import of scipy.linalg) follows its environment variables.
    """
    # PP 2026-10-01: reviewer: lib_nodes' BLAS sums (smooth > 0, corr) depend on the thread count, so an integrator
    # built in a multi-threaded parent differed from the single-threaded workers' (spawn check failed, last bits of
    # the products changed). Kept here (private) because parallel.py belongs to another module owner.
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        threadpool_limits = None
    if threadpool_limits is not None:
        old = par.blas_threads()
        with threadpool_limits(limits=n, user_api="blas"):
            yield old
        return
    libs = par._loaded_bundled_openblas()
    old = {p: int(get()) for p, (get, _) in libs.items()}
    try:
        for p, (_, set_) in libs.items():
            if old[p] != n:
                set_(n)
        yield old
    finally:
        for p, (get, set_) in libs.items():
            if int(get()) != old[p]:
                set_(old[p])


# ----------------------------------------------------------------------------------------------
# one dump
# ----------------------------------------------------------------------------------------------
def sample_path(samples_dir, dump, pattern=SAMPLE_PATTERN):
    """Path of the sphere samples of a dump: ``samples_dir / pattern.format(dump=dump)``."""
    return os.path.join(os.fspath(samples_dir), pattern.format(dump=int(dump)))


def dump_path(outdir, name, dump):
    """Path of the disc-integrated profiles of a dump: ``outdir / name / DUMP_PATTERN.format(dump=dump)``."""
    return os.path.join(os.fspath(outdir), name, DUMP_PATTERN.format(dump=int(dump)))


def load_sample(samples_dir, dump, pattern=SAMPLE_PATTERN):
    """
    The sphere samples of one dump (sphere_sample.py), as fw_disc_dumps.py reads them.

    Parameters
    ----------
    samples_dir: str or os.PathLike
        Directory of the per-dump sample files (M424: samples_r4050_N1236544).
    dump: int
        Dump number.
    pattern: str
        File name, formatted with ``dump`` (default :data:`SAMPLE_PATTERN`, 'd{dump:04d}.npz').

    Returns
    -------
    dict
        teff [K], ur, uth, uph [km/s] as float64 (N,) (``.astype(np.float64)`` of the stored float32, exact),
        t_s [s] (float), dump (int), path (absolute).

    Raises
    ------
    KeyError
        A member is missing.
    ValueError
        The arrays are not 1-D of one length.
    """
    # PP 2026-10-01: ported from fw_disc_dumps.py:84-86 (np.load, .astype(np.float64)) and :102 (float(smp["t_s"]))
    path = sample_path(samples_dir, dump, pattern)
    with np.load(path) as z:
        missing = [k for k in ("teff", "ur", "uth", "uph", "t_s") if k not in z.files]
        if missing:
            raise KeyError("{} has no member(s) {}".format(path, missing))
        out = {k: z[k].astype(np.float64) for k in ("teff", "ur", "uth", "uph")}
        t_s = float(z["t_s"])
    n = out["teff"].shape
    if len(n) != 1 or any(out[k].shape != n for k in ("ur", "uth", "uph")):
        raise ValueError("{}: teff, ur, uth, uph must be 1-D of one length, got {}".format(
            path, {k: out[k].shape for k in out}))
    out.update(t_s=t_s, dump=int(dump), path=os.path.abspath(path))
    return out


def disc_dump(integ, sample, mu, tn, pn, diag_vwin=None, diag_keys=DIAG_KEYS, lref=None, grid=None, batch=False):
    """
    Disc-integrated profiles of one dump for several lines of sight (the per-dump computation of fw_disc_dumps.py).

    For every line of sight k: v = u_r mu_k + u_theta tn_k + u_phi pn_k (:func:`ppmpy.synspec.sphere.los_velocity`)
    and ``integ(mu_k, v, k0, k1, a)`` with the T_eff' interpolation pairs (k0, k1, a) = ``integ.pairs(teff)`` of the
    dump; then the diagnostics of every profile, the points beyond the node range and their visible weight.

    Parameters
    ----------
    integ: object
        A disc integrator: :class:`ppmpy.synspec.disc.DiscFlux`, or any object with ``t`` (increasing node
        T_eff' [K]), ``pairs(teff) -> (k0, k1, a)``, ``__call__(mu, v, k0, k1, a) -> (F (nl, ny), F0 (nl, ny),
        vmean (nl,), vsig (nl,), n_clip)`` and ``grid`` (VelocityGrid; or pass ``grid``). Optional: ``nl``
        (number of lines; lref is then checked before anything is computed), ``lines`` (LineSet) or ``lref``
        (the default lref, and a check of a given one). :func:`run_disc_dumps` needs more (see there).
    sample: mapping
        teff, ur, uth, uph (N,) (converted to float64; exact for float32) and t_s (:func:`load_sample`).
    mu, tn, pn: array-like
        (nlos, N) projections r_hat . n, theta_hat . n, phi_hat . n of the points
        (:func:`ppmpy.synspec.sphere.project_los`, 'matmul' for the M424 bits).
    diag_vwin: float, optional
        Moment window |y| <= diag_vwin [km/s] for v1 and sigma; None (default) = the whole grid, recorded as
        grid.vmax (the legacy ``vwin = VY``).
    diag_keys: sequence of str
        Diagnostics (:data:`ppmpy.synspec.diagnostics.DIAG_KEYS`, the legacy set and order).
    lref: LineSet or array-like, optional
        (nl,) velocity zero points [A] for the EW; default the integrator's 'lines' or 'lref' attribute (set by
        :func:`flux_integrator` when the library records its lref; the legacy library_dT10.npz does not, so the
        M424 runs need it). When both are there they must agree to :data:`LREF_TOL` (ValueError otherwise, e.g.
        for the lines in another order).
    grid: VelocityGrid, optional
        Default ``integ.grid``.
    batch: bool or None
        False (default, legacy): one ``integ(...)`` call per line of sight. True: one
        ``integ.integrate_los(mu, v, pairs=(k0, k1, a))`` call for all lines of sight (the integrator must have
        it; ValueError otherwise), with the velocities of all lines of sight at once (nlos N x 8 bytes, M424
        79 MB). None: True for integrators whose ``fft`` attribute is 'lazy' (a lazy
        :class:`ppmpy.synspec.disc.DiscImu` then transforms each library row once per dump instead of once per
        line of sight: M424 11.1 instead of 19.4 s per dump; the anonymous memory peak grows from ~0.6 to ~1.1 GB),
        else False. The results are the same bit for bit (DiscFlux and DiscImu: integrate_los equals the per-call
        loop; checked on toys and on M424 dumps, tests/synspec/test_dumps.py).

    Returns
    -------
    dict
        dump (sample['dump'] or None), t_s (float); F, F0 (nlos, nl, ny) float64 with / without Doppler shifts;
        diag_keys (array of str), diag_vwin (float), diag_F, diag_F0 (nlos, nl, nkeys); vmean_w, sigma_w (nlos, nl)
        (weighted mean and rms of v, the integrator's weights; DiscFlux: mu F_c); n_lo, n_hi (int: points below /
        above the node range); wout (nlos,) (their share of the visible weight sum(mu)); n_clip (nlos,) int64
        (visible points with clipped shifts); teff_mean, teff_std, teff_min, teff_max (float64); node_range (2,)
        (first and last node T_eff').

    Validation
    ----------
    Bit for bit fw_disc_dumps.py process() before its np.savez (synthetic, and M424 dumps 3200-3209 through
    :func:`run_disc_dumps`): same pairs, velocities (every product and sum rounded as ``ur * MU[k] + uth * TN[k] +
    uph * PN[k]``), integrator calls, wout (:func:`ppmpy.synspec.library.coverage`) and diagnostics (whole-grid
    window = the legacy vwin = VY, EW with the Jacobian). Independent of the number of BLAS threads (the
    integrator's nodes are not: build it with :func:`flux_integrator`, see the module notes).

    Notes
    -----
    Memory: F and F0 (2 nlos nl ny x 8 bytes; M424 2.1 MB), the float64 samples (4 N x 8 bytes), one velocity
    array and the integrator's temporaries for one line of sight (``batch``: the velocities of all lines of sight
    and the integrator's temporaries of its integrate_los).
    """
    # PP 2026-10-01: ported from fw_disc_dumps.py:84-100 (process(): pairs, lo/hi, the loop over the lines of sight,
    # wout, the diagnostics) and :102-105 (t_s, the T_eff' statistics, node_range)
    grid = _integ_grid(integ, grid)
    _, lr = _integ_lref(integ, lref)
    nl_integ = _integ_nl(integ)
    if nl_integ is not None and lr.size != nl_integ:
        raise ValueError("need one reference wavelength per line ({}), got {}".format(nl_integ, lr.size))
    keys = _diag_keys(diag_keys)
    teff = np.asarray(sample["teff"], dtype=np.float64)
    ur, uth, uph = (np.asarray(sample[k], dtype=np.float64) for k in ("ur", "uth", "uph"))
    mu, tn, pn = np.asarray(mu), np.asarray(tn), np.asarray(pn)
    if teff.ndim != 1 or any(u.shape != teff.shape for u in (ur, uth, uph)):
        raise ValueError("teff, ur, uth, uph must be 1-D of one length")
    if mu.ndim != 2 or mu.shape[0] == 0 or tn.shape != mu.shape or pn.shape != mu.shape:
        raise ValueError("mu, tn, pn must have one shape (nlos, N) with nlos >= 1, got {}, {}, {}".format(
            mu.shape, tn.shape, pn.shape))
    if mu.shape[1] != teff.size:
        raise ValueError("the projections are for {} points, the sample has {}".format(mu.shape[1], teff.size))
    tnodes = np.asarray(integ.t, dtype=np.float64)
    batch = _use_batch(integ, batch)
    k0, k1, a = integ.pairs(teff)
    n_lo, n_hi, wout = coverage(tnodes, teff, mu)
    nlos, ny = mu.shape[0], grid.ny
    F = F0 = vm = sd = None
    ncl = np.zeros(nlos, np.int64)
    if batch:
        # PP 2026-10-02: new (one integrate_los call: a lazy DiscImu transforms each library row once per dump)
        V = np.empty(mu.shape)
        for k in range(nlos):
            V[k] = los_velocity(ur, uth, uph, mu[k], tn[k], pn[k])
        r = integ.integrate_los(mu, V, pairs=(k0, k1, a))
        del V
        F, F0 = np.asarray(r["F"], dtype=np.float64), r["F0"]
        nl = F.shape[1] if F.ndim == 3 else -1
        if F.shape != (nlos, nl, ny) or F0 is None or np.shape(F0) != F.shape:
            raise ValueError("the integrator's integrate_los returned profiles of shape {} / {}, expected (nlos, nl, "
                             "{}) = ({}, nl, {})".format(F.shape, None if F0 is None else np.shape(F0), ny, nlos, ny))
        F0 = np.asarray(F0, dtype=np.float64)
        vm, sd = np.asarray(r["vmean_w"], dtype=np.float64), np.asarray(r["sigma_w"], dtype=np.float64)
        ncl[:] = np.asarray(r["n_clip"])
        del r
    for k in range(0 if batch else nlos):
        v = los_velocity(ur, uth, uph, mu[k], tn[k], pn[k])
        Fk, F0k, vmk, sdk, nck = integ(mu[k], v, k0, k1, a)
        if F is None:
            nl = np.shape(Fk)[0]
            if np.shape(Fk) != (nl, ny) or np.shape(F0k) != (nl, ny):
                raise ValueError("the integrator returned profiles of shape {} / {}, expected (nl, {})".format(
                    np.shape(Fk), np.shape(F0k), ny))
            F, F0 = np.zeros((nlos, nl, ny)), np.zeros((nlos, nl, ny))
            vm, sd = np.zeros((nlos, nl)), np.zeros((nlos, nl))
        F[k], F0[k], vm[k], sd[k], ncl[k] = Fk, F0k, vmk, sdk, nck
        del v, Fk, F0k
    if lr.size != F.shape[1]:
        raise ValueError("need one reference wavelength per line ({}), got {}".format(F.shape[1], lr.size))
    with np.errstate(invalid="ignore", divide="ignore"):
        dF = diagnostics_array(F, grid.y, lr, vwin=diag_vwin, keys=keys)
        dF0 = diagnostics_array(F0, grid.y, lr, vwin=diag_vwin, keys=keys)
    dump = sample.get("dump") if hasattr(sample, "get") else None
    return dict(dump=None if dump is None else int(dump), t_s=float(sample["t_s"]), F=F, F0=F0,
                diag_keys=np.array(list(keys)), diag_vwin=float(grid.vmax) if diag_vwin is None else float(diag_vwin),
                diag_F=dF, diag_F0=dF0, vmean_w=vm, sigma_w=sd, n_lo=n_lo, n_hi=n_hi, wout=np.asarray(wout),
                n_clip=ncl, teff_mean=teff.mean(), teff_std=teff.std(), teff_min=teff.min(), teff_max=teff.max(),
                node_range=np.array([tnodes[0], tnodes[-1]]))


def run_fields(integ, name=None):
    """
    The run constants stored in every per-dump file: method, name, lamfix, smooth, nmin.

    Parameters
    ----------
    integ: object
        The integrator. :class:`ppmpy.synspec.disc.DiscFlux` is method 'flux'; any other integrator must have a
        string attribute ``method`` (e.g. 'imu'). lamfix, smooth come from ``integ.node_params`` (corr, smooth of
        :func:`ppmpy.synspec.library.lib_nodes`; False, 0.0 where absent); nmin too for method 'flux' (0 where
        absent), and 0 for every other method whatever its node_params say (legacy rule: the production imu files
        have nmin 0).
    name: str, optional
        Run name (the output subdirectory). Default the legacy rule: method + '_lamfix' (with a profile
        correction) + '_sm<smooth:g>' (with smoothing), e.g. 'flux', 'flux_lamfix', 'flux_sm335'; for a
        :class:`ppmpy.synspec.disc.DiscImu` whose options differ from the legacy integrator's (:func:`_imu_mode`)
        further '_lazy' (fft='lazy'), '_f32' (dtype='float32'), '_c<chunk>' (chunk != 128) and '_l<j>-<k>' (only
        lines j, k built), e.g. 'imu_lazy', 'imu_lazy_f32'; the default options keep 'imu'.

    Returns
    -------
    dict
        method (str), name (str), lamfix (bool), smooth (float), nmin (int).
    """
    # PP 2026-10-01: ported from fw_disc_dumps.py:55 (NAME) and :102-105 (method, lamfix, smooth, nmin of np.savez;
    # nmin = a.nmin if method == "flux" else 0)
    if isinstance(integ, DiscFlux):
        method = "flux"
    else:
        method = getattr(integ, "method", None)
        if not isinstance(method, str) or not method:
            raise ValueError("the integrator ({}) has no 'method' string attribute (e.g. 'imu')".format(
                type(integ).__name__))
    p = dict(getattr(integ, "node_params", None) or {})
    lamfix, smooth = bool(p.get("corr", False)), float(p.get("smooth", 0.0))
    nmin = int(p.get("nmin", 0)) if method == "flux" else 0
    if name is None:
        name = method + ("_lamfix" if lamfix else "") + ("_sm{:g}".format(smooth) if smooth else "")
        # PP 2026-10-02: reviewer: every DiscImu mode defaulted to 'imu' (the legacy directory)
        name += "".join("_" + _MODE_TAG[k](v) for k, v in _imu_mode(integ))
    return dict(method=method, name=_check_name(name), lamfix=lamfix, smooth=smooth, nmin=nmin)


_MODE_TAG = dict(fft=str, dtype=lambda v: "f32" if v == "float32" else str(v), chunk="c{}".format,
                 lines=lambda v: "l" + "-".join(str(j) for j in v))


def _imu_mode(integ):
    """
    The options of a :class:`ppmpy.synspec.disc.DiscImu` that differ from those of the legacy integrator (the
    production imu run: fft 'precomputed', dtype 'float64', chunk 128, all lines built), as a list of (option,
    value) in the order fft, dtype, chunk, lines; [] for the legacy options and for any other integrator.
    """
    # PP 2026-10-02: new (reviewer: run names and the adoption of record-less directories by DiscImu mode)
    if not isinstance(integ, DiscImu):
        return []
    out = []
    if integ.fft != "precomputed":
        out.append(("fft", integ.fft))
    if integ.dtype != "float64":
        out.append(("dtype", integ.dtype))
    if int(integ.chunk) != DiscImu.CHUNK:
        out.append(("chunk", int(integ.chunk)))
    if tuple(integ.built) != tuple(range(integ.nl)):
        out.append(("lines", [int(j) for j in integ.built]))
    return out


def dump_arrays(result, fields, extra=None):
    """
    The members of a per-dump file in the legacy order and dtypes (:data:`DUMP_KEYS`), then ``extra``.

    Parameters
    ----------
    result: dict
        From :func:`disc_dump` (with ``dump`` set).
    fields: dict
        From :func:`run_fields`.
    extra: dict, optional
        Further members (names other than the legacy ones and '_meta').

    Returns
    -------
    dict
        name -> value for :func:`ppmpy.synspec.io.save_npz`: dump, n_lo, n_hi, nmin int64; t_s, smooth,
        diag_vwin float; method, name str; lamfix bool; F, F0 float32; the rest as computed (float64, n_clip
        int64).
    """
    # PP 2026-10-01: ported from fw_disc_dumps.py:102-105 (the members, order and dtypes of np.savez)
    if result.get("dump") is None:
        raise ValueError("result has no dump number")
    out = dict(dump=np.int64(result["dump"]), t_s=float(result["t_s"]), method=str(fields["method"]),
               name=str(fields["name"]), lamfix=bool(fields["lamfix"]), smooth=float(fields["smooth"]),
               F=np.asarray(result["F"]).astype(np.float32), F0=np.asarray(result["F0"]).astype(np.float32),
               diag_keys=np.asarray(result["diag_keys"]), diag_vwin=float(result["diag_vwin"]),
               diag_F=result["diag_F"], diag_F0=result["diag_F0"], vmean_w=result["vmean_w"],
               sigma_w=result["sigma_w"], n_lo=np.int64(result["n_lo"]), n_hi=np.int64(result["n_hi"]),
               wout=result["wout"], n_clip=np.asarray(result["n_clip"], dtype=np.int64), teff_mean=result["teff_mean"],
               teff_std=result["teff_std"], teff_min=result["teff_min"], teff_max=result["teff_max"],
               nmin=np.int64(fields["nmin"]), node_range=result["node_range"])
    out.update(_check_extra(extra))
    return out


def flux_integrator(library, nmin=20, smooth=0.0, corr=None, corr_key="corr", grid=None, pad_tol=PAD_TOL):
    """
    The integrator of a flux run: library -> :func:`ppmpy.synspec.library.lib_nodes` ->
    :class:`ppmpy.synspec.disc.DiscFlux`. A module-level factory for :func:`run_disc_dumps` (picklable, so
    'spawn' workers build their own integrator from file names).

    Parameters
    ----------
    library: str, os.PathLike or FluxLibrary
        Flux library (M424: library_dT10.npz).
    nmin, smooth:
        Options of :func:`ppmpy.synspec.library.lib_nodes` (M424 runs: nmin 20; flux_sm335: smooth 335).
    corr: str, os.PathLike or np.ndarray, optional
        Additive per-bin profile correction (nb, nl, ny), or a .npz holding it as ``corr_key`` (M424 flux_lamfix:
        lamfix_dT10.npz, the cache of fw_disc.lam_corrections).
    grid: VelocityGrid or dict, optional
        Grid of the library profiles (default the M424 grid); a dict is passed to VelocityGrid. Checked against
        the grid the library records in its build parameters (ny, y0, y1 of
        :meth:`ppmpy.synspec.library.FluxLibrary.build`; the legacy library_dT10.npz records none).
    pad_tol: float
        :class:`ppmpy.synspec.disc.DiscFlux` zero-padding tolerance.

    Returns
    -------
    DiscFlux
        Its ``node_params`` give the run fields (:func:`run_fields`): flux, flux_lamfix, flux_sm335, ... When the
        library records its reference wavelengths (``params['lref']``, libraries built by
        :meth:`ppmpy.synspec.library.FluxLibrary.build`), they are attached as ``lref`` (float64 (nl,)):
        :func:`disc_dump` and :func:`run_disc_dumps` then default to them and check a given lref against them.

    Raises
    ------
    ValueError
        The grid does not match the library's recorded grid, or the library's lref does not fit its lines.

    Validation
    ----------
    M424: library_dT10.npz with (nmin=20), (nmin=20, smooth=335), (nmin=20, corr=lamfix_dT10.npz) reproduces the
    three production flux runs bit for bit (test_dumps.py, test_disc.py).

    Notes
    -----
    The nodes are built with the loaded BLAS limited to 1 thread (restored afterwards): with smooth > 0 or a
    correction, :func:`ppmpy.synspec.library.lib_nodes` sums through BLAS, whose last bits depend on the thread
    count, and the M424 products were made with 1 thread. So the integrator is the same in a multi-threaded
    parent and in single-threaded pool workers (module notes).
    """
    # PP 2026-10-01: ported from fw_disc_dumps.py:65-67 (NODES = fd.lib_nodes(np.load(library_dT10.npz), nmin, smooth,
    # lamfix); INT = fd.DiscFlux(NODES)); lamfix -> corr = the lamfix_dT10.npz cache of fw_disc.lam_corrections
    # PP 2026-10-01: lib_nodes under a 1-thread BLAS limit; lref / grid from the library's build parameters (reviewer)
    lib = library if isinstance(library, FluxLibrary) else FluxLibrary.load(library)
    if isinstance(corr, (str, os.PathLike)):
        with np.load(os.fspath(corr)) as z:
            corr = z[corr_key]
    if isinstance(grid, dict):
        grid = VelocityGrid(**grid)
    lp = lib.params or {}
    if all(k in lp for k in ("ny", "y0", "y1")):
        g = grid if grid is not None else VelocityGrid()
        if (g.ny, float(g.y[0]), float(g.y[-1])) != (int(lp["ny"]), float(lp["y0"]), float(lp["y1"])):
            raise ValueError("the library's profiles are on a grid of {} points {:g}..{:g} km/s, the integrator's grid "
                             "{} has {} points {:g}..{:g} km/s: pass grid=".format(
                                 lp["ny"], lp["y0"], lp["y1"], g, g.ny, g.y[0], g.y[-1]))
    with _blas_limit(1):
        nodes = lib_nodes(lib, nmin=nmin, smooth=smooth, corr=corr)
    integ = DiscFlux(nodes, grid=grid, pad_tol=pad_tol)
    if lp.get("lref") is not None:
        lr = np.atleast_1d(np.asarray(lp["lref"], dtype=np.float64))
        if lr.shape != (integ.nl,):
            raise ValueError("the library records {} reference wavelengths for {} lines".format(lr.size, integ.nl))
        integ.lref = lr
    return integ


def _imu_line_indices(library, lines, lref):
    """
    ``lines`` of :func:`imu_integrator` with line names resolved through the LineSet ``lref`` when the intensity
    library records no names (the legacy imu_library_dT10.npz); unchanged otherwise (DiscImu resolves names the
    library records, and the factory then checks that the LineSet's names agree). Reads only the library's small
    members (``s`` and the recorded lines).

    Raises
    ------
    ValueError
        Names with neither the library nor a LineSet naming the lines, a LineSet of another number of lines than
        the library, or a name the LineSet does not have.
    """
    # PP 2026-10-02: reviewer: lines=['HEII4200'] with the legacy library failed although lref named the lines
    if lines is None:
        return None
    seq = [lines] if isinstance(lines, (str, int, np.integer)) else list(lines)
    if not any(isinstance(x, str) for x in seq):
        return lines
    from .disc import _imu_lines, _imu_open
    m = _imu_open(library, need=("s",), optional=("lref", "lines"))[0]
    nl = int(np.shape(m["s"])[1])
    if _imu_lines(library, m, nl)[2] is not None:
        return lines                                    # the library names its lines: DiscImu resolves them
    if not isinstance(lref, LineSet):
        raise ValueError("lines {} are given by name, but the intensity library records no line names: pass lref as "
                         "a LineSet (e.g. the M424 LineSet) or give line indices".format(seq))
    if len(lref) != nl:
        raise ValueError("lref names {} lines, the intensity library has {}".format(len(lref), nl))
    out = []
    for x in seq:
        if isinstance(x, str):
            if x not in lref.names:
                raise ValueError("unknown line {!r} (lines of lref: {})".format(x, list(lref.names)))
            x = lref.names.index(x)
        out.append(x)
    return out


def imu_integrator(library, grid=None, lines=None, dtype="float64", fft="precomputed", chunk=DiscImu.CHUNK, lref=None):
    """
    The integrator of an intensity ('imu') run: :class:`ppmpy.synspec.disc.DiscImu` of an intensity library. A
    module-level factory for :func:`run_disc_dumps` (as :func:`flux_integrator`): 'fork' workers share the
    integrator built in the parent (the precomputed library FFTs, copy-on-write); 'spawn' / 'forkserver' workers
    call it in their initializer with the same (picklable) arguments, and their integrator must have the parent's
    record (class, nodes, :meth:`~ppmpy.synspec.disc.DiscImu.fingerprint`).

    Parameters
    ----------
    library: str, os.PathLike or mapping
        The intensity library: a file (legacy imu_library_dT10.npz, or one written by
        :meth:`ppmpy.synspec.library.ImuLibrary.save`; Il and Ic memory-mapped), or an
        :class:`ppmpy.synspec.library.ImuLibrary` / mapping. Prefer the file name for runs: 'spawn' workers
        receive the arguments pickled (a memory-mapped ImuLibrary pickles as its path, a mapping as its arrays,
        1.9 GB for M424), and the run record fingerprints a file by the sha256 of its content (a mapping by the
        integrator's :meth:`~ppmpy.synspec.disc.DiscImu.fingerprint`, which hashes the library content it uses).
    grid, lines, dtype, fft, chunk:
        :class:`ppmpy.synspec.disc.DiscImu` options (defaults = the legacy integrator of fw_disc_dumps.py --method
        imu: the library's recorded grid or the M424 grid, all lines, 'float64', 'precomputed', 128 rows).
        fft='lazy' (and dtype='float32') are the low-memory modes. ``lines`` may name lines (e.g. ['HEII4200']):
        by the library's recorded names, or, for a library that records none (the legacy file), by the names of
        ``lref`` given as a LineSet. Runs of other modes get their own default name in :func:`run_disc_dumps`
        (:func:`run_fields`: 'imu_lazy', 'imu_f32', 'imu_lazy_f32', '_c<chunk>', '_l<lines>'), and the run record
        refuses to mix modes in one directory: their F0 (and with float32 F) differ from the default's at the
        rounding level. A per-dump directory without a run record (the production imu/ of the legacy driver) is
        adopted only by the default mode.
    lref: LineSet or array-like, optional
        (nl,) reference wavelengths [A] of the library's lines (a LineSet also names them), attached to the
        integrator as ``lref`` (and ``lines``, ``names`` for a LineSet) when the library records none (the legacy
        imu_library_dT10.npz: pass the M424 LineSet). Where the library records wavelengths they must agree to
        :data:`LREF_TOL`, and where it records names a LineSet's names must equal them, in order (ValueError
        otherwise); a LineSet then only adds what the library lacks. Run records, :func:`disc_dump` and the collect
        messages then take the lines from the integrator. lref does not change the profiles: the run record keeps it
        in its params (lref) only, not in the integrator's identity, so giving it here or to :func:`run_disc_dumps`
        is the same run.

    Returns
    -------
    DiscImu
        method 'imu', node_params {} (:func:`run_fields`: name 'imu' for the default options, lamfix False, smooth
        0, nmin 0 as the production files).

    Raises
    ------
    ValueError
        Bad DiscImu options, a grid other than the library's recorded one, lref of another length, other values or
        other names than the library's, line names that neither the library nor a LineSet lref defines.

    Validation
    ----------
    Synthetic (toy intensity library on the M424 grid): the per-dump files equal those of the frozen
    fw_disc_dumps.py --method imu (frozen fw_disc.DiscImu) byte for byte, serial, 'fork' and 'spawn', per line of
    sight and batched, with lref given to the run or to this factory. M424: through :func:`run_disc_dumps`
    (default options, the legacy library and the M424 LineSet) dumps 3200-3209 equal the production imu/dNNNN.npz
    of fw_disc_dumps.py --method imu byte for byte, with 4 'fork' and with 2 'spawn' workers; fft='lazy' (batched)
    gives F bit for bit (x86-64) and F0 to the float32 rounding of the files (tests/synspec/test_dumps.py, slow).

    Notes
    -----
    Built with the loaded BLAS limited to 1 thread (as :func:`flux_integrator`; the set-up has no BLAS sums, so
    this changes nothing but keeps the two factories alike). Memory and time of the modes:
    :class:`ppmpy.synspec.disc.DiscImu` (M424 default 7.7 GB and 21-200 s of set-up per process that builds it,
    almost all system time for the first touch of the spectra, so it depends on the login node's state: the parent,
    and every 'spawn' worker start; lazy ~1 MB plus the mapped library, < 0.1 s). Measured through
    :func:`run_disc_dumps` (M424 dumps 3200-3209, Trillium login node, 2026-10-02, three runs): 4 'fork' workers
    116-181 s (the parent's set-up 85-147 s; then 3.2-3.3 s per dump wall), parent peak RSS 9.3 GB, workers 8.1 GB
    each counting the shared pages; 2 'spawn' workers 273 s (the parent's set-up 108 s, then each worker's ~100 s
    and ~13 s per dump), 9.3 GB per process. A 'spawn' parent builds the integrator only to fix the run record and
    releases it before the workers start: with the default mode prefer 'fork', or fft='lazy'. fft='lazy', one
    process (dumps 3200 and 4800, lines of sight batched): 26-28 s including the set-up and the run record's
    fingerprints (41-47 s per line of sight), peak RSS 2.84 GB of which ~1.6 GB are clean pages of the
    memory-mapped library.

    CPU-time limits: a lazy dump takes ~11 s of CPU (batched), so all 1601 M424 dumps in one process (nproc=1)
    would need ~5 h of CPU and pass the login node's ``ulimit -t`` of 3600 s after roughly 300 dumps (the process
    is killed; finished dumps stay). On the login node run lazy modes with nproc >= 2 (pool workers, replaced after
    ``maxtasksperchild`` dumps, each with a fresh CPU-time count; the default 20 dumps are ~4 min of CPU), or
    restart the in-process run in a loop (it skips the dumps present). :func:`run_disc_dumps` warns when the
    projected CPU time of an in-process run exceeds the process's limit.
    """
    # PP 2026-10-02: new (the factory of the imu run; fw_disc_dumps.py:68: INT = fd.DiscImu(), lref = fd.LREF)
    lines = _imu_line_indices(library, lines, lref)
    with _blas_limit(1):
        integ = DiscImu(library, grid=grid, lines=lines, dtype=dtype, fft=fft, chunk=chunk)
    if lref is not None:
        names, lr = _lref_of(lref)
        if lr.shape != (integ.nl,):
            raise ValueError("lref has {} wavelengths, the intensity library {} lines".format(lr.size, integ.nl))
        own = integ.lines if integ.lines is not None else integ.lref
        if own is not None:
            onames, olr = _lref_of(own)
            if olr.shape != lr.shape or not np.allclose(olr, lr, rtol=0.0, atol=LREF_TOL):
                raise ValueError("lref {} differs from the intensity library's {} (another line order or other "
                                 "lines?)".format(lr.tolist(), olr.tolist()))
            lr = olr
            if onames is not None:
                if names is not None and list(names) != list(onames):
                    raise ValueError("the line names {} differ from the intensity library's {}".format(
                        list(names), list(onames)))
                names = onames
        elif integ.names is not None:
            # PP 2026-10-02: reviewer: names were checked only where the library also records lref, so a LineSet
            # with other (or permuted) names relabelled the lines of a library that records names only
            if names is not None and list(names) != list(integ.names):
                raise ValueError("the line names {} differ from the intensity library's {}".format(
                    list(names), list(integ.names)))
            names = list(integ.names)
        labels = getattr(lref, "labels", None) if isinstance(lref, LineSet) and list(lref.names) == list(
            names or []) else None
        integ.lref = np.array(lr, dtype=np.float64)
        integ.names = None if names is None else [str(x) for x in names]
        integ.lines = None if names is None else LineSet(integ.names, integ.lref, labels)
    return integ


# ----------------------------------------------------------------------------------------------
# all dumps
# ----------------------------------------------------------------------------------------------
class _DumpWorker:
    """Per-process state of :func:`run_disc_dumps`: the integrator, the projections and the run configuration."""

    def __init__(self, integ, proj, cfg):
        self.integ = integ
        self.mu, self.tn, self.pn = proj
        self.cfg = cfg

    def process(self, d):
        # PP 2026-10-01: ported from fw_disc_dumps.py:79-110 (process(): skip existing, compute, atomic np.savez, the
        # message); the message is formatted by the parent from the returned summary
        cfg = self.cfg
        path = dump_path(cfg["outdir"], cfg["fields"]["name"], d)
        if os.path.exists(path) and not cfg["overwrite"]:
            return dict(dump=int(d), skipped=True)            # written meanwhile (another rank)
        t0 = time.time()
        smp = load_sample(cfg["samples_dir"], d, cfg["pattern"])
        res = disc_dump(self.integ, smp, self.mu, self.tn, self.pn, diag_vwin=cfg["diag_vwin"],
                        diag_keys=cfg["diag_keys"], lref=cfg["lref"], batch=cfg.get("batch", False))
        extra = cfg["extra_fields"]
        if callable(extra):
            extra = extra(d, smp, res)
        arrays = dump_arrays(res, cfg["fields"], extra)
        meta = None
        if cfg["meta"] is not None:
            meta = dict(cfg["meta"])
            meta.update(dump=int(d), inputs=dict(sample=file_identity(smp["path"])), pid=os.getpid(),
                        wall=time.time() - t0, written=datetime.datetime.now().isoformat(timespec="seconds"))
        save_npz(path, arrays, meta=meta)
        return _dump_summary(d, res, cfg["diag_keys"], time.time() - t0)


def _dump_summary(d, res, keys, wall):
    """The numbers of the legacy per-dump message (mean over the lines of sight)."""
    dF, sd, wout = res["diag_F"], res["sigma_w"], res["wout"]
    out = dict(dump=int(d), wall=float(wall), pid=os.getpid(), sigma_v=float(sd[:, 0].mean()), n_lo=int(res["n_lo"]),
               n_hi=int(res["n_hi"]), wout_max=float(np.max(wout)), n_clip=int(np.sum(res["n_clip"])))
    for q in ("ew", "fwhm"):
        if q in keys:
            out[q] = [float(x) for x in dF[:, :, keys.index(q)].mean(axis=0)]
    return out


def _format_summary(s):
    """fw_disc_dumps.py:107-110 (the per-dump message), for any number of lines."""
    parts = ["dump {}:".format(s["dump"])]
    if "ew" in s:
        parts.append("EW {} A,".format(" ".join("{:.4f}".format(x) for x in s["ew"])))
    if "fwhm" in s:
        parts.append("FWHM {} km/s,".format(" ".join("{:.0f}".format(x) for x in s["fwhm"])))
    parts.append("sigma_v {:.1f} km/s, out of range {}/{} (vis. weight <= {:.1e}), clipped {}; {:.1f} s".format(
        s["sigma_v"], s["n_lo"], s["n_hi"], s["wout_max"], s["n_clip"], s["wall"]))
    return " ".join(parts)


def _pool_init(spec):
    """Pool initializer: the worker's _DumpWorker. 'fork': the parent's integrator and projections (inherited, shared
    copy-on-write, as the legacy module globals); otherwise the integrator is built by the factory (its record,
    :func:`_integ_info`, including the sha256 of its arrays, must equal the parent's; the pool limits the worker's
    BLAS to 1 thread, as the parent's build) and the projections are memory-mapped from the parent's temporary
    files. First the worker's glibc malloc thresholds (disc.WORKER_MALLOC; changes no result; unless
    tune_malloc=False)."""
    # PP 2026-10-01: _tune_malloc: the per-dump temporaries (~10-40 MB per line of sight) were mapped and page-faulted
    # afresh on every call (M424, one process: 0.07-3 s -> 0.01 s system time per dump, Trillium login node)
    if spec.get("tune", True):
        _tune_malloc(**WORKER_MALLOC)
    src = spec["integ"]
    if src[0] == "object":
        integ = src[1]
    else:
        _, factory, args, kwargs = src
        with _blas_limit(1):
            integ = factory(*args, **kwargs)
        got, want = _norm(_integ_info(integ)), spec["integ_info"]
        if got != want:
            diff = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))
            raise RuntimeError("the integrator built in worker {} differs from the parent's (in {}; sha256 {} vs {}): "
                               "did the factory's input files change?".format(os.getpid(), diff, got.get("sha256"),
                                                                              want.get("sha256")))
    proj = tuple(np.asarray(_unshare(r)) for r in spec["proj"])
    return _DumpWorker(integ, proj, spec["cfg"])


def _pool_task(d):
    return par.worker_state().process(d)


def read_run_record(dirname, retries=3, wait=0.2):
    """
    The run record (:data:`RUN_FILE`) of a per-dump directory, or None if it has none.

    Parameters
    ----------
    dirname: str or os.PathLike
        The per-dump directory (``outdir/name``).
    retries, wait:
        Rereads of a record that does not parse, ``wait`` seconds apart (a record being created where hard links
        are not supported can be empty for a moment).

    Raises
    ------
    ValueError
        The record is not valid JSON, or not a run record (no 'params'); the message says how to proceed.
    """
    # PP 2026-10-01: reviewer: a corrupted record gave a bare JSONDecodeError
    path = os.path.join(os.fspath(dirname), RUN_FILE)
    for i in range(int(retries) + 1):
        try:
            with open(path) as f:
                rec = json.load(f)
            break
        except FileNotFoundError:
            return None
        except ValueError as e:
            if i == int(retries):
                raise ValueError("{} is corrupted ({}): if no run is writing into {}, remove it (or the directory) and "
                                 "rerun".format(path, e, os.fspath(dirname))) from None
            time.sleep(wait)
    if not isinstance(rec, dict) or not isinstance(rec.get("params"), dict):
        raise ValueError("{} is not a run record of ppmpy.synspec.dumps (no 'params'): if no run is writing into {}, "
                         "remove it and rerun".format(path, os.fspath(dirname)))
    return rec


def _present_dumps(src):
    """Dump numbers of the per-dump files in a directory, sorted: only files named exactly
    DUMP_PATTERN.format(dump) (e.g. d0100.npz, not d100.npz; temporaries excluded)."""
    # PP 2026-10-01: reviewer: 'd100.npz' next to 'd0100.npz' counted dump 100 twice
    if not os.path.isdir(src):
        return []
    out = set()
    for f in os.listdir(src):
        m = _DUMP_RE.match(f)
        if m and f == DUMP_PATTERN.format(dump=int(m.group(1))):
            out.add(int(m.group(1)))
    return sorted(out)


def _present_tmp(src):
    """Temporaries of per-dump writes (dNNNN.tmp<pid>.npz) in a directory, sorted."""
    if not os.path.isdir(src):
        return []
    return sorted(f for f in os.listdir(src) if _TMP_RE.match(f))


def _stored_constants(path):
    """The run constants (_RUN_CONST) and the profile shape (nlos, nl, ny) stored in a per-dump file."""
    with np.load(path) as r:
        c = dict(method=str(r["method"]), name=str(r["name"]), lamfix=bool(r["lamfix"]), smooth=float(r["smooth"]),
                 nmin=int(r["nmin"]), diag_vwin=float(r["diag_vwin"]), diag_keys=[str(k) for k in r["diag_keys"]],
                 node_range=[float(x) for x in r["node_range"]])
    c["shape"] = list(_npz_layout(path)["F"][0])
    return c


def _writer_gone(rec, ddir):
    """True if the run that wrote a record is known to be gone: on this host, and its process is not running (or is
    this process, which has no active run on ddir). False where it cannot be told (another host, no writer)."""
    w = rec.get("writer") or {}
    try:
        pid = int(w.get("pid"))
    except (TypeError, ValueError):
        return False
    if w.get("host") != socket.gethostname() or os.name != "posix":
        return False
    if pid == os.getpid():
        return ddir not in _ACTIVE_DIRS
    return not _pid_alive(pid)


def _move_stale_record(path, rec):
    """Move a stale record out of the way (atomic rename to a unique name, then remove) if it is still the record
    judged stale; if the rename caught a record written meanwhile by another run, put that one back. Returns True
    when the stale record is gone."""
    grave = "{}.stale.{}.{}".format(path, socket.gethostname(), os.getpid())
    try:
        os.rename(path, grave)
    except FileNotFoundError:
        return True
    try:
        with open(grave) as f:
            moved = json.load(f)
    except ValueError:
        moved = None
    if moved != rec:
        try:
            os.link(grave, path)
        except OSError:
            pass
        os.remove(grave)
        return False
    os.remove(grave)
    return True


def _check_run_record(ddir, params, info, expect, _log, mode=None):
    """
    Refuse to mix two configurations in one directory. With a run record: its params must equal this run's;
    a mismatching record of a run that wrote nothing and whose writer is gone (:func:`_writer_gone`) is replaced.
    Without one (legacy directories): an integrator with non-legacy options (``mode``, :func:`_imu_mode`, e.g. a lazy
    DiscImu, whose files would differ at the rounding level from the legacy driver's although the stored constants
    agree) is refused; otherwise the constants of an existing per-dump file must match. Then the record is created
    exclusively (:func:`_create_json_excl`); if another run created one meanwhile, compare with that. Returns (record
    path, the record in force).
    """
    # PP 2026-10-01: reviewer: exclusive creation (two runs starting at once), stale records of runs that wrote
    # nothing, integrator bits in 'info' only when its inputs are fingerprinted
    path = os.path.join(ddir, RUN_FILE)
    p = _norm(params)
    new = dict(kind=_RUN_KIND, params=p, info=_norm(info),
               writer=dict(host=socket.gethostname(), pid=os.getpid(),
                           time=datetime.datetime.now().isoformat(timespec="seconds")))
    checked_legacy = False
    for _ in range(5):
        rec = read_run_record(ddir)
        if rec is None:
            have = _present_dumps(ddir)
            if have and not checked_legacy:
                if mode:
                    # PP 2026-10-02: reviewer: the stored constants are the same for every DiscImu mode, so a lazy /
                    # float32 run adopted the record-less production imu/ directory and mixed its files in
                    raise ValueError("{} holds {} per-dump files without {} (a run of the legacy driver, e.g. the "
                                     "production imu run, whose integrator had the legacy options), but this "
                                     "integrator's options are not the legacy ones ({}): its files would differ at "
                                     "the rounding level; use another name (name=None gives each mode its own) or "
                                     "output directory".format(ddir, len(have), RUN_FILE, ", ".join(
                                         "{}={!r}".format(k, v) for k, v in mode)))
                f = dump_path(os.path.dirname(ddir), os.path.basename(ddir), have[0])
                got = _stored_constants(f)
                want = _norm(expect)
                diff = sorted(k for k in want if got.get(k) != want[k])
                if diff:
                    raise ValueError("{} was written by another configuration (differs in {}: {} vs {}): use another "
                                     "name or output directory".format(f, diff, {k: got.get(k) for k in diff},
                                                                       {k: want[k] for k in diff}))
                _log("{}: {} existing dump files without {} match this run's constants; adopting the directory".format(
                    ddir, len(have), RUN_FILE))
                checked_legacy = True
            if _create_json_excl(path, new):
                return path, new
            continue                                        # created meanwhile by another run: compare with it
        old = rec["params"]
        if old == p:
            mine, theirs = info.get("integrator_sha256"), (rec.get("info") or {}).get("integrator_sha256")
            if theirs is not None and mine != theirs:
                msg = ("{}: the integrator built now has the same inputs as the run that created the directory but "
                       "other bits (sha256 {} vs {}; another CPU, BLAS or numpy?): the dumps computed now may differ "
                       "from those present at the rounding level".format(ddir, mine, theirs))
                warnings.warn(msg, RuntimeWarning, stacklevel=3)
                _log("WARNING: " + msg)
            return path, rec
        diff = sorted(k for k in set(old) | set(p) if old.get(k) != p.get(k))
        empty = not _present_dumps(ddir) and not _present_tmp(ddir)
        if empty and _writer_gone(rec, ddir):
            if _move_stale_record(path, rec):
                _log("{}: replacing the run record of a run of another configuration (differs in {}) that wrote no "
                     "results and is no longer running".format(ddir, diff))
            continue
        if empty:
            raise ValueError("{} has the run record of another configuration (differs in {}) but no per-dump files "
                             "yet (a run that is starting, or one that failed before writing results): if no run is "
                             "writing into it, remove {} and rerun; else use another name or output directory".format(
                                 ddir, diff, path))
        raise ValueError("{} holds the results of another configuration (differs in {}): use another name or output "
                         "directory, or remove the directory".format(ddir, diff))
    raise RuntimeError("{}: could not create or read a stable run record (runs of different configurations starting "
                       "at once?)".format(path))


def run_disc_dumps(dumps, samples_dir, outdir, name, integ_factory, factory_args, theta, phi, los, nproc=1, rank=0,
                   nranks=1, maxtasksperchild=20, timeout=900.0, overwrite=False, start_method=None, extra_fields=None,
                   lref=None, factory_kwargs=None, pattern=SAMPLE_PATTERN, project_method="matmul", diag_vwin=None,
                   diag_keys=DIAG_KEYS, meta=False, tmpdir=None, tune_malloc=None, batch=None, log=None):
    """
    Disc-integrated profiles of many dumps (port of fw_disc_dumps.py): one file per dump in ``outdir/name``.

    Parameters
    ----------
    dumps: iterable of int
        Dump numbers (each needs ``samples_dir/pattern``); this rank takes dumps[i] with i % nranks == rank.
    samples_dir: str or os.PathLike
        Per-dump sphere samples (:func:`load_sample`).
    outdir: str or os.PathLike
        Output root: the files go to ``outdir/name/dNNNN.npz`` (:func:`dump_path`).
    name: str or None
        Run name; None = the legacy name from the integrator (:func:`run_fields`: flux, flux_lamfix, flux_sm335,
        imu; for DiscImu modes other than the legacy one e.g. imu_lazy, imu_f32, imu_lazy_f32).
    integ_factory: callable
        ``integ_factory(*factory_args, **factory_kwargs)`` builds the integrator (e.g. :func:`flux_integrator`,
        :func:`imu_integrator`).
        Called once in this process, with the loaded BLAS limited to 1 thread (the pool workers' count; module
        notes); 'spawn' / 'forkserver' workers call it again (module-level function and picklable arguments:
        pass file names, not arrays), and their integrator must have the same record (:func:`_integ_info`:
        class, nodes, node_params, fingerprint() and the sha256 of its arrays), else
        :class:`ppmpy.synspec.parallel.WorkerInitError`. The integrator: as for :func:`disc_dump`, plus a
        ``method`` string unless it is a DiscFlux (:func:`run_fields`), and, unless it is a DiscFlux, either a
        ``fingerprint()`` method returning a JSON-able record of everything that determines its results, or its
        state in ndarray attributes (all of them are hashed); ValueError otherwise.
    factory_args: tuple
        Its positional arguments.
    theta, phi: array-like
        (N,) coordinates of the points [rad] (M424: points.npz, e.g. via
        :func:`ppmpy.synspec.io.npz_member_memmap`; bitwise M424 results need these, not recomputed ones).
    los: str or array-like
        Lines of sight (:func:`ppmpy.synspec.sphere.project_los`; M424: 'thompson2024').
    nproc: int
        Worker processes (1 = in this process). The results do not depend on nproc, the start method or the
        BLAS thread settings of the environment (module notes: BLAS threads). In this process the CPU time of all
        dumps adds up against its ``ulimit -t`` (3600 s on the Trillium login nodes): after the first dump a
        RuntimeWarning says when the projected total exceeds the limit (e.g. a lazy DiscImu, ~11 s of CPU per M424
        dump, is killed after ~300 dumps); use nproc >= 2 there (workers recycled by ``maxtasksperchild``) or
        restart the run in a loop.
    rank, nranks: int
        Split of the dumps over independent runs (nodes): dump i of the list goes to rank i % nranks
        (:func:`ppmpy.synspec.parallel.split_items`, as the legacy --rank/--nranks).
    maxtasksperchild: int or None
        Dumps per worker before it is replaced (legacy 20: restarts the CPU-time count of a ``ulimit -t`` limit,
        3600 s on the Trillium login nodes; None: never). With 'fork' a replacement costs nothing (it inherits
        the integrator). With 'spawn' / 'forkserver' every worker start, replacements included, builds the
        integrator: about max(nproc, len(todo) / maxtasksperchild) builds, e.g. 80 for 1601 dumps. An M424
        DiscFlux takes 0.05-3 s (login node; negligible); an integrator with a slow setup (the intensity method's
        took ~2 min on a login node) would spend ~2.7 h of CPU on 80 builds: raise maxtasksperchild there as far
        as the CPU-time limit allows (each worker's builds plus its dumps must stay below it), or use 'fork'. The
        build time and the number of builds are logged, and a RuntimeWarning is issued above
        :data:`REBUILD_WARN` s.
    timeout: float or None
        Raise :class:`ppmpy.synspec.parallel.PoolStalled` when no dump finishes for this long [s] (a killed
        worker); finished dumps are kept, so a rerun continues. Worker start-up (spawn: building the integrator)
        counts against it: workers started together are also replaced together, so it must exceed the build
        time plus one dump (a RuntimeWarning when the build takes more than half of it).
    overwrite: bool
        Recompute dumps whose file exists (default: skip them; restartable).
    start_method: str, optional
        :func:`ppmpy.synspec.parallel.get_context` (default 'fork' on Linux). 'fork' workers share the parent's
        integrator and projections (copy-on-write, as the legacy module globals); 'spawn' workers build the
        integrator themselves and memory-map the projections, which the parent writes to a temporary directory
        under ``tmpdir`` (3 nlos N x 8 bytes; M424 237 MB; removed at the end). Scripts using 'spawn' need the
        ``if __name__ == "__main__":`` guard.
    extra_fields: dict or callable, optional
        Further members of every per-dump file, after the legacy ones: a dict of constant values (recorded with
        the sha256 of each value: a restart with other values is refused), or ``extra_fields(dump, sample,
        result) -> dict`` (called in the workers; picklable for 'spawn'; recorded by its name). No object arrays.
    lref: LineSet or array-like, optional
        (nl,) velocity zero points [A] of the lines (for the EW); default the integrator's 'lines' / 'lref'
        attribute (:func:`flux_integrator` sets it from libraries that record it). Required when the integrator
        has none, as for the M424 library_dT10.npz (M424: LineSet(['HEI4026', 'HEII4200', 'HEI4922'], [4026.22,
        4199.90, 4921.93]); a LineSet's names label the collect messages). Checked before anything is written:
        one per line of the integrator, and equal to the integrator's own to :data:`LREF_TOL` where it has one.
    factory_kwargs: dict, optional
        Keyword arguments of ``integ_factory``.
    pattern: str
        Sample file name pattern (:data:`SAMPLE_PATTERN`).
    project_method: str
        :func:`ppmpy.synspec.sphere.project_los` method ('matmul' = fw_disc_dumps.py).
    diag_vwin, diag_keys:
        Diagnostics options (:func:`disc_dump`; default whole grid, the legacy keys).
    meta: bool
        Add a '_meta' member (provenance: the run parameters, the sample file, timing) to every per-dump file.
        Default False: the files then equal the legacy ones byte for byte.
    tmpdir: str, optional
        Parent directory of the temporary projection files of non-fork pools (default
        ``tempfile.gettempdir()``).
    tune_malloc: bool or None
        glibc malloc settings :data:`ppmpy.synspec.disc.WORKER_MALLOC` (allocations below 32 MiB from the heap, up
        to 512 MiB of freed heap kept), so that the per-dump temporaries (10-40 MB per line of sight) are not mapped
        and page-faulted afresh on every call. None (default): in the pool workers only (as
        :func:`ppmpy.synspec.disc.integrate_exact_stream`); True: also in this process when it computes the dumps
        itself (nproc 1); the setting is process-wide and permanent, so prefer it in scripts; False: nowhere.
        Changes no result. Measured on a Trillium login node (transparent huge pages 'always'), M424, one dump in
        one process: 0.6 s user plus 0.07-3 s system without (varying with the node's state), 0.01 s system with.
        No effect outside glibc / Linux.
    batch: bool or None
        :func:`disc_dump`: all lines of sight of a dump in one ``integrate_los`` call (True), one call per line of
        sight (False, legacy), or None (default): batched for integrators with ``fft == 'lazy'`` (a lazy
        :class:`ppmpy.synspec.disc.DiscImu`: each library row transformed once per dump; M424 11 instead of 19 s per
        dump), else per line of sight. The files are the same byte for byte either way, so it is not part of the
        run record.
    log: callable, optional
        log(message) for progress messages (e.g. print), with the legacy per-dump lines.

    Returns
    -------
    dict
        name, dir (the per-dump directory), fields (:func:`run_fields`), dumps (this rank's), todo (not present at
        the start, or all with ``overwrite``), done (computed here), skipped (present), results (per-dump summaries:
        EW, FWHM, sigma_v, n_lo, n_hi, wout_max, n_clip, wall, pid), stale_tmp (temporaries of killed writers found
        in the directory; not removed), run_file, nn, node_range, nproc (workers used; 1 = in this process),
        start_method (None without a pool), wall [s]. When nothing is to do the integrator is not built
        (with ``name`` given): fields, run_file, nn, node_range are None then, nproc 0.

    Raises
    ------
    FileNotFoundError
        Sample files of dumps to do are missing (checked before anything is computed).
    ValueError
        Inconsistent inputs (checked before the run record is written), ``outdir/name`` holds results of another
        configuration or its run record is corrupted (module notes).
    ppmpy.synspec.parallel.PoolStalled
        No dump finished within ``timeout``.
    ppmpy.synspec.parallel.WorkerInitError
        A 'spawn' worker's integrator differs from this process's (the factory's input files changed?).

    Memory
    ------
    Parent: the projections (3 nlos N x 8 bytes; M424 237 MB, plus ~0.1 GB of temporaries while they are made,
    in blocks of :data:`PROJECT_CHUNK` points) and the integrator (DiscFlux: nl nn (ny + L + 2) x 8 bytes, M424
    69 MB); with 'spawn' both are released once the workers can build / map their own. Each worker: one dump
    (:func:`disc_dump`: the samples, 4 N x 8 bytes, and the integrator's temporaries) on top of the shared
    integrator and projections ('fork') or its own integrator and the mapped projections ('spawn').

    Intensity runs (:func:`imu_integrator`; the default mode holds 7.7 GB per process that builds it, 21-200 s of
    set-up on a login node): see there for the measured time and memory of 'fork', 'spawn' and the lazy mode, and
    for the CPU-time limit of lazy runs in one process.

    Measured (M424 flux run, all 1601 dumps, 8 'fork' workers, Trillium login node, 2026-10-01; every file byte for
    byte the production one): 154 s wall (0.10 s per dump; 0.6-0.7 s per dump and worker), workers 997 s user +
    204 s system, 0.58 GB max RSS per worker (shared pages counted in each), parent 0.49 GB. Before the workers'
    malloc settings (:data:`ppmpy.synspec.disc.WORKER_MALLOC`): 176 s, 1018 + 333 s. Dumps 3200-3209: 'fork' 4
    workers 3.3 s, 'spawn' 4 workers 7.6 s (each worker builds its DiscFlux), serial 7.2 s. Legacy production
    (compute node): 0.7 s per dump alone, ~67 dumps/s with 96 workers.

    Notes
    -----
    Legacy behaviour kept: skip existing files (also re-checked by the worker), atomic writes through
    ``dNNNN.tmp<pid>.npz``, ``maxtasksperchild`` 20, the 900 s watchdog, the rank split, the per-dump message.
    New: the run record (no silent mixing of configurations on restart), the up-front check of the sample files,
    the number of points and the lines, explicit lines of sight, lines and grid instead of module constants, the
    'spawn' start method, optional '_meta' and extra fields, the pool workers' glibc malloc settings (as
    :func:`ppmpy.synspec.disc.integrate_exact_stream`; changes no result), the 1-thread BLAS limit of this
    process while it builds and computes (restored at the end).
    """
    # PP 2026-10-01: ported from fw_disc_dumps.py:52-75 (NAME, OUT, the integrator, the projections) and :113-139 (the
    # driver: dump list, rank split, todo, Pool(maxtasksperchild) with the imap_unordered watchdog, the messages)
    T0 = time.time()
    _log = _logger(log, T0)
    dumps = _dump_list(dumps)
    mine = par.split_items(dumps, rank, nranks)
    if not callable(integ_factory):
        raise TypeError("integ_factory must be callable")
    factory_args = tuple(factory_args or ())
    factory_kwargs = dict(factory_kwargs or {})
    try:
        nproc_ok = int(nproc) == nproc and nproc >= 1
    except (TypeError, ValueError):
        nproc_ok = False
    if not nproc_ok:
        raise ValueError("nproc must be a positive integer, got {!r}".format(nproc))
    nproc = int(nproc)
    if extra_fields is not None and not callable(extra_fields):
        _check_extra(extra_fields)
    keys = _diag_keys(diag_keys)
    samples_dir = os.path.realpath(os.fspath(samples_dir))
    outdir = os.path.abspath(os.fspath(outdir))

    def _todo(nm):
        return [d for d in mine if overwrite or not os.path.exists(dump_path(outdir, nm, d))]

    def _summary(nm, todo, **kw):
        tset = set(todo)
        out = dict(name=nm, dir=os.path.join(outdir, nm), fields=None, dumps=mine, todo=todo, done=[],
                   skipped=[d for d in mine if d not in tset], results=[], stale_tmp=[], run_file=None, nn=None,
                   node_range=None, nproc=0, start_method=None, wall=time.time() - T0)
        out.update(kw)
        return out

    if name is not None:
        _check_name(name)
        todo = _todo(name)
        if not todo:
            _log("{}: {} dumps (rank {}/{}), all present in {}".format(name, len(mine), rank, nranks,
                                                                      os.path.join(outdir, name)))
            return _summary(name, todo)
    # everything computed in this process with the BLAS of the pool workers (1 thread): the integrator's nodes
    # depend on it (module notes)
    with contextlib.ExitStack() as stack:
        stack.enter_context(_blas_limit(1))
        # the integrator (fw_disc_dumps.py:65-69); 'fork' workers inherit it
        tb = time.time()
        integ = integ_factory(*factory_args, **factory_kwargs)
        tb = time.time() - tb
        fields = run_fields(integ, name)
        name = fields["name"]
        use_batch = _use_batch(integ, batch)                # ValueError for batch=True without integrate_los
        todo = _todo(name)
        if not todo:
            _log("{}: {} dumps (rank {}/{}), all present".format(name, len(mine), rank, nranks))
            return _summary(name, todo, fields=fields)
        gone = [d for d in todo if not os.path.exists(sample_path(samples_dir, d, pattern))]
        if gone:
            raise FileNotFoundError("{} of {} dumps to do have no sample file in {} ({}), e.g. {}".format(
                len(gone), len(todo), samples_dir, pattern, gone[:10]))
        grid = _integ_grid(integ)
        names, lr = _integ_lref(integ, lref)
        nl_integ = _integ_nl(integ)
        if nl_integ is not None and lr.size != nl_integ:
            raise ValueError("need one reference wavelength per line ({}), got {}".format(nl_integ, lr.size))
        L = _los_vectors(los)
        theta64, phi64 = np.asarray(theta, dtype=np.float64), np.asarray(phi, dtype=np.float64)
        if theta64.ndim != 1 or theta64.shape != phi64.shape:
            raise ValueError("theta and phi must be 1-D arrays of one length")
        N = theta64.size
        try:
            ns = npz_member_memmap(sample_path(samples_dir, todo[0], pattern), "teff").shape
        except (ValueError, OSError, KeyError):
            ns = None                                   # compressed or unusual: disc_dump checks it
        if ns is not None and ns != (N,):
            raise ValueError("the samples have {} points, theta and phi {}".format(ns, N))
        tnodes = np.asarray(integ.t, dtype=np.float64)
        vwin_rec = float(grid.vmax) if diag_vwin is None else float(diag_vwin)
        integ_info = _norm(_integ_info(integ))
        inputs = _inputs_fingerprint(integ_factory, factory_args, factory_kwargs)
        integ_par = {k: v for k, v in integ_info.items() if k not in ("sha256", "arrays")}
        if inputs is not None:
            integ_par["inputs"] = inputs                # the computed arrays' sha256: info only (another CPU / BLAS)
        else:
            integ_par.update(sha256=integ_info["sha256"], arrays=integ_info["arrays"])
        params = dict(name=name, method=fields["method"], lamfix=fields["lamfix"], smooth=fields["smooth"],
                      nmin=fields["nmin"], grid=grid.to_dict(), lref=lr.tolist(), los=L.tolist(),
                      project_method=project_method, diag_vwin=vwin_rec, diag_keys=list(keys),
                      samples=dict(dir=samples_dir, pattern=pattern),
                      points=dict(n=int(N), sha256=_sha256(theta64, phi64)), integrator=integ_par,
                      extra_fields=_extra_desc(extra_fields))
        factory = "{}.{}".format(_modname(integ_factory),
                                 getattr(integ_factory, "__qualname__", type(integ_factory).__name__))
        info = dict(names=names, factory=factory,
                    factory_args=[a if isinstance(a, (str, int, float, bool)) or a is None
                                  else (os.fspath(a) if isinstance(a, os.PathLike) else type(a).__name__)
                                  for a in factory_args],
                    factory_kwargs={k: (v if isinstance(v, (str, int, float, bool)) or v is None
                                        else (os.fspath(v) if isinstance(v, os.PathLike) else type(v).__name__))
                                    for k, v in factory_kwargs.items()},
                    integrator_sha256=integ_info["sha256"], integrator_arrays=integ_info["arrays"])
        ddir = os.path.join(outdir, name)
        os.makedirs(ddir, exist_ok=True)
        expect = dict(method=fields["method"], name=name, lamfix=fields["lamfix"], smooth=fields["smooth"],
                      nmin=fields["nmin"], diag_vwin=vwin_rec, diag_keys=list(keys),
                      node_range=[float(tnodes[0]), float(tnodes[-1])], shape=[L.shape[0], lr.size, grid.ny])
        run_file, _ = _check_run_record(ddir, params, info, expect, _log, mode=_imu_mode(integ))
        _ACTIVE_DIRS[ddir] = _ACTIVE_DIRS.get(ddir, 0) + 1
        stack.callback(_release_dir, ddir)
        stale = _present_tmp(ddir)
        if stale:
            _log("{} temporaries of interrupted writes in {} (not used; remove them when no run writes here): {}"
                 .format(len(stale), ddir, stale[:5]))

        # projections (fw_disc_dumps.py:70-75): 'matmul' = rhat @ LOS.T, then C-contiguous (nlos, N)
        MU, TN, PN = project_los(theta64, phi64, L, method=project_method, chunk=PROJECT_CHUNK)
        del theta64, phi64
        base_meta = None
        if meta:
            base_meta = make_meta("synspec.disc_dump", params=params, names=names, run=info)
        cfg = dict(outdir=outdir, fields=fields, overwrite=bool(overwrite), samples_dir=samples_dir, pattern=pattern,
                   lref=lr, diag_vwin=diag_vwin, diag_keys=keys, extra_fields=extra_fields, meta=base_meta,
                   batch=use_batch)
        _log("{}: {} dumps (rank {}/{}), {} to do, {} workers -> {}; T_eff' nodes {} ({:.0f}-{:.0f} K){}".format(
            name, len(mine), rank, nranks, len(todo), nproc, ddir, tnodes.size, tnodes[0], tnodes[-1],
            "; lines of sight batched" if use_batch else ""))

        done, skipped_late, results = [], [], []
        t1 = time.time()

        def _handle(n, s):
            if s.get("skipped"):
                skipped_late.append(s["dump"])
                return
            done.append(s["dump"])
            results.append(s)
            _log("[{}/{}] {}".format(n, len(todo), _format_summary(s)))

        nproc_eff = min(nproc, len(todo))
        smethod = None
        if nproc_eff <= 1:
            if tune_malloc:
                _tune_malloc(**WORKER_MALLOC)
            worker = _DumpWorker(integ, (MU, TN, PN), cfg)
            c0 = _cpu_seconds()
            for n, d in enumerate(todo, 1):
                _handle(n, worker.process(d))
                if n == 1 and len(todo) > 1 and c0 is not None:
                    _cpu_budget_warning(_cpu_seconds() - c0, len(todo) - 1, _log)
        else:
            smethod = par.get_context(start_method).get_start_method()
            par.login_node_warning(nproc_eff)
            if smethod == "fork":
                spec = dict(integ=("object", integ), proj=tuple(("array", x) for x in (MU, TN, PN)), cfg=cfg,
                            tune=tune_malloc is not False)
            else:
                _rebuild_cost(tb, len(todo), nproc_eff, maxtasksperchild, timeout, smethod, _log)
                spill = tempfile.mkdtemp(prefix="synspec_dumps_", dir=tmpdir)
                stack.callback(shutil.rmtree, spill, ignore_errors=True)
                proj = tuple(_share(_spill(x, spill, k), smethod) for x, k in zip((MU, TN, PN), ("mu", "tn", "pn")))
                spec = dict(integ=("factory", integ_factory, factory_args, factory_kwargs), proj=proj, cfg=cfg,
                            integ_info=integ_info, tune=tune_malloc is not False)
                integ = MU = TN = PN = proj = None          # the workers build / map their own
            with par.make_pool(nproc_eff, initializer=_pool_init, initargs=(spec,), maxtasksperchild=maxtasksperchild,
                               start_method=smethod) as pool:
                try:
                    for n, s in enumerate(par.imap_watchdog(pool, _pool_task, todo, timeout=timeout), 1):
                        _handle(n, s)
                except par.PoolStalled as e:
                    left = [d for d in todo if not os.path.exists(dump_path(outdir, name, d))]
                    _log("ABORT: no dump finished for {:.0f} s (worker killed?); {} dumps missing, e.g. {}; rerun to "
                         "finish (existing outputs are kept)".format(e.timeout, len(left), left[:10]))
                    raise
            del spec
    dt = time.time() - t1
    _log("done: {} dumps in {:.0f} s ({:.2f} s per dump wall, {} workers)".format(len(done), dt, dt / max(len(done), 1),
                                                                               nproc_eff))
    out = _summary(name, todo, fields=fields, done=done, results=results, stale_tmp=stale, run_file=run_file,
                   nn=int(tnodes.size), node_range=[float(tnodes[0]), float(tnodes[-1])], nproc=nproc_eff,
                   start_method=smethod)
    out["skipped"] = out["skipped"] + skipped_late
    return out


def _release_dir(ddir):
    """End of a run_disc_dumps call on ddir (_ACTIVE_DIRS)."""
    n = _ACTIVE_DIRS.get(ddir, 0) - 1
    if n > 0:
        _ACTIVE_DIRS[ddir] = n
    else:
        _ACTIVE_DIRS.pop(ddir, None)


REBUILD_WARN = 600.0
"""Estimated CPU time [s] of the integrator rebuilds of non-fork pool workers above which :func:`run_disc_dumps`
warns (each worker start, including replacements after ``maxtasksperchild`` dumps, builds the integrator)."""


def _rebuild_cost(tb, ntodo, nproc, maxtasksperchild, timeout, smethod, _log):
    """Log (and warn when large) the cost of the integrator builds in non-fork workers: one per worker start,
    i.e. about max(nproc, ceil(ntodo / maxtasksperchild)) builds of ~tb seconds (the parent's build time)."""
    # PP 2026-10-01: reviewer: with spawn every replacement worker rebuilds the integrator (DiscImu ~2 min)
    starts = nproc if not maxtasksperchild else max(nproc, int(math.ceil(ntodo / float(maxtasksperchild))))
    cost = tb * starts
    _log("{}: each worker start builds the integrator ({:.1f} s here): ~{} builds for {} dumps (maxtasksperchild {}), "
         "~{:.0f} s of CPU".format(smethod, tb, starts, ntodo, maxtasksperchild, cost))
    if cost > REBUILD_WARN:
        warnings.warn("the {} workers will build the integrator ~{} times ({:.1f} s each, ~{:.0f} s of CPU): raise "
                      "maxtasksperchild (or use None) as far as the per-process CPU-time limit allows, or use 'fork'"
                      .format(smethod, starts, tb, cost), RuntimeWarning, stacklevel=4)
    if timeout is not None and maxtasksperchild and tb > 0.5 * timeout:
        warnings.warn("building the integrator takes {:.0f} s, more than half the watchdog timeout ({:.0f} s): "
                      "replacement workers may trip it; raise timeout".format(tb, timeout), RuntimeWarning,
                      stacklevel=4)


def _cpu_seconds():
    """CPU time (user + system) of this process [s], or None where the resource module is missing."""
    try:
        import resource
    except ImportError:
        return None
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def _cpu_budget_warning(per_dump, left, _log, limit=None, used=None):
    """
    Warn (RuntimeWarning and log) when a run in this process (nproc 1) would pass the process's soft CPU-time limit
    (``ulimit -t``; 3600 s on the Trillium login nodes, where the process is then killed): the CPU used so far plus
    ``per_dump`` x ``left``. ``limit`` / ``used`` override the process's values (tests). Returns the message, or None
    (no limit, or within it).
    """
    # PP 2026-10-02: reviewer: a lazy DiscImu takes ~11 s of CPU per dump, so an in-process run of all 1601 dumps
    # would be killed after ~300 dumps on the login node
    if limit is None or used is None:
        try:
            import resource
            soft = resource.getrlimit(resource.RLIMIT_CPU)[0]
        except (ImportError, OSError, ValueError, AttributeError):
            return None
        if soft == resource.RLIM_INFINITY or soft < 0:
            return None
        limit = soft if limit is None else limit
        used = _cpu_seconds() if used is None else used
    need = used + per_dump * left
    if need <= limit:
        return None
    more = int(max(limit - used, 0.0) // max(per_dump, 1e-9))
    msg = ("this process may use {:.0f} s of CPU (ulimit -t) and has used {:.0f} s; the remaining {} dumps need "
           "~{:.0f} s more ({:.1f} s each), so it will be killed after ~{} more (finished dumps stay; rerun to "
           "continue): use nproc >= 2 (workers are replaced every maxtasksperchild dumps) or restart the run in a "
           "loop".format(limit, used, left, per_dump * left, per_dump, more))
    warnings.warn(msg, RuntimeWarning, stacklevel=3)
    _log("WARNING: " + msg)
    return msg


# ----------------------------------------------------------------------------------------------
# the time series
# ----------------------------------------------------------------------------------------------
def _axes_from_params(p):
    """(Y, LREF, los) of a run's params (run record or per-dump '_meta')."""
    return (VelocityGrid(**p["grid"]).y, np.asarray(p["lref"], dtype=np.float64),
            np.asarray(p["los"], dtype=np.float64))


def _grid_arg(grid):
    """(Y float64 (ny,), vshift or None) of a grid argument: VelocityGrid, dict of VelocityGrid arguments, or the
    1-D grid itself."""
    # PP 2026-10-01: reviewer: a dict raised a TypeError deep in numpy
    if isinstance(grid, dict):
        try:
            grid = VelocityGrid(**grid)
        except TypeError as e:
            raise ValueError("grid: a dict must hold VelocityGrid arguments (dv, vmax, vshift): {}".format(e)) from None
    if isinstance(grid, VelocityGrid):
        return np.asarray(grid.y, dtype=np.float64), float(grid.vshift)
    try:
        y = np.asarray(grid, dtype=np.float64)
    except (TypeError, ValueError):
        y = None
    if y is None or y.ndim != 1 or y.size < 2:
        raise ValueError("grid must be a VelocityGrid, a dict of its arguments or the 1-D velocity grid, got {!r}"
                         .format(type(grid).__name__))
    return y, None


def _resolve_axes(src, metas, los, grid, lref, fshape):
    """Y, LREF, los of the time series, the run params (or None) and the labels of the messages (line names or None,
    vshift or None): from the run record and the per-dump '_meta' (they must agree exactly), checked against the
    arguments (Y exactly, LREF and los to 1e-12), else from the arguments; their sizes must match the profiles
    (nlos, nl, ny). Names: the record's / '_meta''s, else those of a LineSet given as lref; vshift: the recorded
    grid's, else that of a VelocityGrid (or dict) given as grid."""
    rec = read_run_record(src)
    recorded = []
    names = None
    if rec is not None:
        recorded.append((os.path.join(src, RUN_FILE), _norm(rec["params"])))
        names = (rec.get("info") or {}).get("names")
    for d, m in metas:
        p = m.get("params")
        if p is not None:
            recorded.append(("'_meta' of dump {}".format(d), _norm(p)))
        if names is None:
            names = m.get("names")
    params = None
    for where, p in recorded:
        if params is None:
            params = p
        elif p != params:
            diff = sorted(k for k in set(p) | set(params) if p.get(k) != params.get(k))
            raise ValueError("{} and {} record different runs (differ in {}): the directory mixes "
                             "configurations".format(recorded[0][0], where, diff))
    labels = ("Y", "LREF", "los")
    rec_ax = _axes_from_params(params) if params is not None else (None, None, None)
    arg_ax = [None, None, None]
    vshift = None
    if params is not None and isinstance(params.get("grid"), dict):
        vshift = params["grid"].get("vshift")
    if grid is not None:
        arg_ax[0], vs_arg = _grid_arg(grid)
        if vshift is None:
            vshift = vs_arg
    if lref is not None:
        lnames, arg_ax[1] = _lref_of(lref)
        if names is None:
            names = lnames
    if los is not None:
        arg_ax[2] = _los_vectors(los)
    out = []
    for k, (r, a) in enumerate(zip(rec_ax, arg_ax)):
        if r is not None and a is not None:
            same = np.array_equal(r, a) if k == 0 else (r.shape == a.shape and np.allclose(r, a, rtol=0.0, atol=1e-12))
            if not same:
                raise ValueError("the {} given differs from the one recorded for the run in {}".format(labels[k], src))
        v = r if r is not None else a
        if v is None:
            raise ValueError("{} of {} is unknown: the per-dump files have no '_meta' and the directory no {} (legacy "
                             "outputs); pass grid=, lref= and los=".format(labels[k], src, RUN_FILE))
        out.append(v)
    Y, LREF, LOS = out
    nlos, nl, ny = fshape
    if Y.shape != (ny,) or LREF.shape != (nl,) or LOS.shape != (nlos, 3):
        raise ValueError("Y {}, LREF {}, los {} do not fit the profiles (nlos, nl, ny) = {}".format(
            Y.shape, LREF.shape, LOS.shape, fshape))
    if names is not None and len(names) != nl:
        names = None
    return Y, LREF, LOS, params, (names, vshift)


def _read_profiles(path, key, ident, shape):
    """Member F or F0 of a per-dump file as float32, after checking that the file is the one read in the first pass."""
    with open(path, "rb") as fh:
        st = os.fstat(fh.fileno())
        if (st.st_ino, st.st_size, st.st_mtime_ns) != ident:
            raise RuntimeError("{} changed while the time series was collected (a run writing into the directory?)"
                               .format(path))
        with np.load(fh) as z:
            a = z[key]
    if a.shape != shape:
        raise ValueError("{}: {} has shape {}, the first dump {}".format(path, key, a.shape, shape))
    return np.ascontiguousarray(a, dtype=np.float32)


def collect_timeseries(outdir, name, dumps=None, out=None, allow_missing=False, los=None, grid=None, lref=None,
                       meta=False, log=None):
    """
    Assemble the per-dump files of a run into one time series (port of fw_disc_collect.py), streamed.

    Parameters
    ----------
    outdir: str or os.PathLike
        Output root of :func:`run_disc_dumps`; the per-dump files are read from ``outdir/name``.
    name: str
        Run name (every file's ``name`` must equal it).
    dumps: iterable of int, optional
        Dumps to collect, in this order (legacy: range(d0, d1 + 1)). Default: every per-dump file present,
        sorted (files named exactly as :data:`DUMP_PATTERN`; e.g. a 'd100.npz' is ignored).
    out: str or os.PathLike, optional
        Output file (default ``outdir/<name>_timeseries.npz``, the legacy location). Written atomically
        (hidden temporary in the target directory, then renamed).
    allow_missing: bool
        Collect the dumps present when some of ``dumps`` are missing (default: FileNotFoundError, as the legacy
        script, which refused).
    los, grid, lref:
        Lines of sight, velocity grid and reference wavelengths for the members los, Y, LREF. Taken from the run
        record (:data:`RUN_FILE`) or the files' '_meta' when present (the arguments, if given, must agree:
        Y exactly, LREF and los to 1e-12); required for legacy directories that have neither. grid: a
        VelocityGrid, a dict of its arguments, or the 1-D grid (ValueError otherwise); lref: a LineSet or the
        wavelengths. Without a record, the names of a LineSet and the vshift of a VelocityGrid (or dict) label the
        messages (as the legacy HEI4026, ..., 'clipped |v| > 400 km/s').
    meta: bool
        Add a '_meta' member (provenance: source directory, dumps, missing dumps, the run parameters). Default
        False: the file equals the legacy one byte for byte.
    log: callable, optional
        log(message): the legacy messages (present / missing dumps, per-line EW, centroid and FWHM ranges,
        residual rms and maximum, out-of-range points), at the cost of one more pass over the written profiles.

    Returns
    -------
    str
        The path written.

    Raises
    ------
    FileNotFoundError
        Missing dumps (without ``allow_missing``), or none present.
    ValueError
        A file of another run (dump number, name, method, lamfix, smooth, nmin, diag_vwin, diag_keys, node_range,
        shapes, recorded run parameters differ), or Y / LREF / los unknown or inconsistent.
    RuntimeError
        A per-dump file was replaced between the two passes.

    Notes
    -----
    Members (:data:`TIMESERIES_KEYS`, legacy order and dtypes): dumps int64; t_s float64; Y, LREF, los float64;
    method, name str; diag_vwin float64; F, F0 (nd, nlos, nl, ny) float32; diag_keys; diag_F, diag_F0
    (nd, nlos, nl, nkeys); vmean_w, sigma_w (nd, nlos, nl); n_lo, n_hi (nd,) float64 (legacy quirk: int in the
    per-dump files); wout (nd, nlos) float64; n_clip (nd, nlos) int64; teff_mean, teff_std, teff_min, teff_max
    (nd,); node_range (2,). The legacy script took method from the first file and diag_keys, node_range from the
    last without comparing them; here all files must agree.

    Memory: the small members of all dumps (M424: ~4 MB) and one dump's profiles at a time; the output's F and
    F0 are written member by member as they are read (two passes over the per-dump files, the second one
    reading only F and F0); with ``log`` a third pass reads the written F one dump at a time (explicit reads, no
    memory map). Measured (M424 flux, 1601 dumps, 1.66 GB output, login node, warm page cache): 4.4 s wall,
    VmHWM 59 MB (33 MB anonymous); with ``log`` 5.5 s, 63 MB.

    Validation
    ----------
    Byte for byte the file of the frozen fw_disc_collect.py for the same per-dump files (synthetic, M424 grid;
    M424 production files of dumps 3200-3209); M424 dumps 3200-3209: every member equals the corresponding rows of
    the production ``*_timeseries.npz``; all 1601 dumps of the flux run (scratch check): byte for byte the
    production flux_timeseries.npz, and the same messages as its production log.
    """
    # PP 2026-10-01: ported from fw_disc_collect.py:26-53 (present / missing dumps, the per-dump reads and checks, the
    # np.savez call; now streamed with fwresults._NpzStream, byte for byte the same) and :54-60 (the messages)
    T0 = time.time()
    _log = _logger(log, T0)
    _check_name(name)
    src = os.path.join(os.fspath(outdir), name)
    present = _present_dumps(src)
    want = present if dumps is None else _dump_list(dumps)
    pset = set(present)
    have = [d for d in want if d in pset]
    missing = [d for d in want if d not in pset]
    _log("{}: {} of {} dumps present{}".format(name, len(have), len(want), "; missing {}{}".format(
        missing[:20], " ..." if len(missing) > 20 else "") if missing else ""))
    if missing and not allow_missing:
        raise FileNotFoundError("{} of {} dumps missing in {}, e.g. {} (allow_missing=True collects the others)".format(
            len(missing), len(want), src, missing[:20]))
    if not have:
        raise FileNotFoundError("no per-dump files to collect in {}".format(src))
    paths = [dump_path(outdir, name, d) for d in have]
    lay = _npz_layout(paths[0])
    fshape = tuple(lay["F"][0])
    if len(fshape) != 3:
        raise ValueError("{}: F has shape {}, expected (nlos, nl, ny)".format(paths[0], fshape))
    nlos, nl, ny = fshape
    nk = lay["diag_F"][0][-1]
    nd = len(have)
    small = dict(t_s=((), np.float64), diag_F=((nlos, nl, nk), np.float64), diag_F0=((nlos, nl, nk), np.float64),
                 vmean_w=((nlos, nl), np.float64), sigma_w=((nlos, nl), np.float64), n_lo=((), np.float64),
                 n_hi=((), np.float64), wout=((nlos,), np.float64), n_clip=((nlos,), np.int64),
                 teff_mean=((), np.float64), teff_std=((), np.float64), teff_min=((), np.float64),
                 teff_max=((), np.float64))
    arr = {k: np.zeros((nd,) + s, dt) for k, (s, dt) in small.items()}
    first, idents, metas = None, [], []
    for i, (d, p) in enumerate(zip(have, paths)):
        st = os.stat(p)
        idents.append((st.st_ino, st.st_size, st.st_mtime_ns))
        with np.load(p) as r:
            if int(r["dump"]) != d or str(r["name"]) != name:
                raise ValueError("{} holds dump {} of run {!r}, expected dump {} of {!r}".format(
                    p, int(r["dump"]), str(r["name"]), d, name))
            const = dict(method=str(r["method"]), lamfix=bool(r["lamfix"]), smooth=float(r["smooth"]),
                         nmin=int(r["nmin"]), diag_vwin=float(r["diag_vwin"]), diag_keys=r["diag_keys"],
                         node_range=r["node_range"])
            if first is None:
                first = const
            else:
                diff = [k for k in const if not np.array_equal(const[k], first[k])]
                if diff:
                    raise ValueError("{} differs from {} in {}: the directory mixes runs".format(p, paths[0], diff))
            for k, (s, _) in small.items():
                v = r[k]
                if np.shape(v) != s:
                    raise ValueError("{}: {} has shape {}, expected {}".format(p, k, np.shape(v), s))
                arr[k][i] = v
            if "_meta" in r.files:
                metas.append((d, json.loads(str(r["_meta"]))))
    Y, LREF, LOS, run_params, msg_labels = _resolve_axes(src, metas, los, grid, lref, fshape)
    del metas

    out = os.fspath(out) if out is not None else os.path.join(os.fspath(outdir), "{}_timeseries.npz".format(name))
    acc = np.zeros(fshape) if log is not None else None
    stream = _NpzStream(out)
    try:
        for k, v in (("dumps", np.array(have, dtype=np.int64)), ("t_s", arr["t_s"]), ("Y", Y), ("LREF", LREF),
                     ("los", LOS), ("method", np.array(first["method"])), ("name", np.array(name)),
                     ("diag_vwin", np.float64(first["diag_vwin"]))):
            stream.write_array(k, v)
        for key in ("F", "F0"):
            with stream.member(key, (nd,) + fshape, np.float32) as w:
                for i, p in enumerate(paths):
                    a = _read_profiles(p, key, idents[i], fshape)
                    if acc is not None and key == "F":
                        acc += a
                    w.write(a)
                    del a
        stream.write_array("diag_keys", first["diag_keys"])
        for k in ("diag_F", "diag_F0", "vmean_w", "sigma_w", "n_lo", "n_hi", "wout", "n_clip", "teff_mean",
                  "teff_std", "teff_min", "teff_max"):
            stream.write_array(k, arr[k])
        stream.write_array("node_range", first["node_range"])
        if meta:
            m = make_meta("synspec.disc_timeseries", params=dict(name=name, source=os.path.abspath(src), n_dumps=nd,
                                                                 dumps=[have[0], have[-1]], missing=missing,
                                                                 run=run_params))
            stream.write_array("_meta", _meta_array(m))
        stream.close()
    except BaseException:
        stream.abort()
        raise
    if log is not None:
        _collect_messages(out, arr, first["diag_keys"], acc / nd, msg_labels, _log)
    _log("wrote {} ({:.2f} GB)".format(out, os.path.getsize(out) / 1e9))
    return out


def _collect_messages(out, arr, keys, mean, labels, _log):
    """fw_disc_collect.py:54-60: per line EW mean and rms, centroid rms, FWHM range, residual F - <F>_t rms and max
    (streamed over the written F: rms = sqrt(<r^2> - <r>^2), the legacy std up to rounding); out-of-range points.
    labels = (line names or None, vshift or None) from _resolve_axes."""
    # PP 2026-10-01: ported from fw_disc_collect.py:54-60; the residuals are accumulated dump by dump
    keys = [str(k) for k in np.asarray(keys)]
    nl = mean.shape[1]
    names, vs = labels
    names = names or ["line{}".format(j) for j in range(nl)]
    # explicit reads of one dump at a time (a memory map would leave every page of F resident: 0.83 GB for M424)
    mm = npz_member_memmap(out, "F")
    shape, offset = mm.shape, int(mm.offset)
    del mm
    s1, s2, mx = np.zeros(nl), np.zeros(nl), np.zeros(nl)
    buf = np.empty(shape[1:], np.float32)
    view = buf.reshape(-1).view(np.uint8)
    with open(out, "rb", buffering=0) as f:
        f.seek(offset)
        for i in range(shape[0]):
            got = 0
            while got < view.size:
                k = f.readinto(view[got:])
                if not k:
                    raise IOError("{}: short read of member F".format(out))
                got += k
            r = buf.astype(np.float64) - mean
            s1 += r.sum(axis=(0, 2))
            s2 += (r * r).sum(axis=(0, 2))
            mx = np.maximum(mx, np.abs(r).max(axis=(0, 2)))
    n = shape[0] * shape[1] * shape[3]
    rms = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0.0))
    dF = arr["diag_F"]
    for j in range(nl):
        parts = ["{}:".format(names[j])]
        if "ew" in keys:
            e = dF[:, :, j, keys.index("ew")]
            parts.append("EW {:.4f} A (rms over dumps and LOS {:.1e});".format(e.mean(), e.std()))
        if "v1" in keys:
            parts.append("<v> rms {:.2f} km/s;".format(dF[:, :, j, keys.index("v1")].std()))
        if "fwhm" in keys:
            f = dF[:, :, j, keys.index("fwhm")]
            parts.append("FWHM {:.0f}-{:.0f} km/s;".format(f.min(), f.max()))
        parts.append("residual F - <F>_t: rms {:.1e}, max {:.1e}".format(rms[j], mx[j]))
        _log(" ".join(parts))
    _log("out-of-range points per dump: max {} below / {} above, visible weight <= {:.1e}; clipped |v| > {} km/s: {}"
         .format(int(arr["n_lo"].max()), int(arr["n_hi"].max()), arr["wout"].max(),
                 "vshift" if vs is None else "{:g}".format(vs), int(arr["n_clip"].sum())))
