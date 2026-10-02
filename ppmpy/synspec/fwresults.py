"""
FASTWIND results on the numpy side: the OUT / OUT_IMU readers, the merge of the packed per-point
results of a sphere run into one profile file, the streaming combine of the per-task files, a
zero-copy store of the merged profiles, per-point equivalent widths and run statistics.

Layout of a per-point run (M424: ``/scratch/ppathak/fastwind_sphere/d3200_r4050_N1236544``)
-----------------------------------------------------------------------------------------
``results/<tag>/part_*.tar.gz``
    Packed results, one directory ``P<idx>/`` per point with ``meta.txt``
    (``idx teff status niter T_tau23 t_pnlte t_formal``), ``OUT.<LINE>_<suffix>`` (one per line)
    and, optionally, the model files. A point's files are contiguous in a part.
``merged/<tag>.npz``
    :func:`merge_task` of one tag (in parallel: :func:`merge_tasks`).
``profiles.npz``, ``missing.txt``
    :func:`combine` of all merged files; ``missing.txt`` lists the points of ``points.txt``
    without a successful model (failed ones with T_eff raised by 1 K per failed attempt).

Members of a merged / combined file (dtypes as written by the legacy pipeline)
-----------------------------------------------------------------------------
``idx`` int32, ``teff`` float64 [K], ``status`` str (``ok``, ``formal_failed``, ``pnlte_failed``,
``pnlte_timeout``), ``niter`` int16, ``T_tau23`` float64 [K], ``t_pnlte``, ``t_formal`` float32 [s];
``lam`` (air wavelength [Angstrom]), ``fcont`` (continuum flux), ``fnorm`` (F / F_cont): float32
(npoint, nline, nrow), NaN for points with status != ok; ``lines`` str; ``r theta phi x y z ur_kms relT``
copied from ``points.npz``; ``teff_nudge`` float32 (T_eff used minus T_eff of points.npz, 0 or +1 K per
retry of a failed model).

Conventions
-----------
* Profiles are at rest. FASTWIND's OUT files print lambda to 0.01 Angstrom only, so a row of ``lam``
  can contain the same value twice (seen in HEI4026 / HEII4200): compare profiles row by row, and do not
  use ``np.interp`` on a row as if it were strictly increasing without care.
* Numbers are parsed with Python's ``float`` (correctly rounded), exactly as ``np.genfromtxt`` did in the
  legacy scripts; a file that does not look like a plain FASTWIND table falls back to ``np.genfromtxt``.
* Iteration cap: ``niter`` is the number of 'ITERATION NO' lines in pnlte.log (fw_sphere_point.sh), i.e. the
  NLTE iterations 1..n plus two start lines, so a model that never converged has niter = ITMORE + 2 for a
  fresh start (ITSTART = 0; M424: ITMORE = 100, cap 102 = :data:`NITER_CAP_M424`). The cap is a property of
  the run configuration and is never inferred from the data (in a run where no model hits it, max(niter)
  is a converged model): pass it explicitly, get it with :func:`niter_cap_from_indat`, or record it in the
  '_meta' of the merged files (``niter_cap`` of :func:`merge_task` / :func:`combine`).
* Temporary files: outputs are written to a hidden name in the target directory
  (``.<name>.tmp<pid>.npz``) and moved into place; glob patterns given to :func:`combine` also skip the
  visible temporary names of other writers (``*.tmp.npz``, ``*.tmp<pid>.npz``, e.g. the legacy script's).

Kept legacy quirks
------------------
* A merged file without points (a tag directory without parts, or parts without points) stores ``teff``,
  ``status`` and ``T_tau23`` as empty float64 arrays (``np.array([])``); concatenated with string status
  arrays this widens the combined ``status`` dtype to ``<U32`` (np.concatenate's rule), exactly as the legacy
  ``--combine``.
* A point packed twice keeps the first record unless a later one is successful (merge), and the combine
  keeps per idx the first successful record in file order.

Validation
----------
* :func:`merge_task` writes the same file, byte for byte, as the project's ``fw_sphere_merge.py --tags``
  (``numpy.savez`` writes every zip entry with the fixed date 1980-01-01, so the files are deterministic);
  checked on synthetic parts and on the M424 tag ``task_missing_0000``.
* :func:`combine` writes the same ``profiles.npz`` (byte for byte) and ``missing.txt`` as
  ``fw_sphere_merge.py --combine``, without holding the concatenated profiles in memory: synthetic runs
  (including empty merged files) and M424 subsets in the tests; the full 41-tag M424 combine reproduces the
  production ``profiles.npz`` and ``missing.txt`` byte for byte (verified 2026-10-01 on the login node: first
  version with memory maps 66 s, VmHWM 2.5 GB of mostly file pages; current version with explicit reads
  11 s with a warm page cache, peak RSS 0.18 GB, of which 0.17 GB anonymous).
* :func:`ew_per_point` reproduces the M424 ``ew.npz`` bit for bit.

PP 2026-10-01: ported from the project scripts fw_sphere_merge.py (merge, --combine), fw_disc.py
(read_imu, library inputs), fig_fw_sphere_ew.py (ew.npz) and the OUT readers of fw_itmore_check.py /
fig_fastwind_*.py; see the provenance comments per function.
"""
import contextlib
import glob
import io
import json
import os
import re
import tarfile
import warnings
import zipfile

import numpy as np

from .io import npz_member_memmap

LINES_M424 = ("HEI4026", "HEII4200", "HEI4922")
NROW_M424 = 161
COPY_KEYS = ("r", "theta", "phi", "x", "y", "z", "ur_kms", "relT")
TEFF_NUDGE = 1.0          # K added to T_eff for each retry of a failed model
TEFF_NUDGE_MAX = 10.0     # largest nudge accepted by merge_task
NITER_EXTRA = 2           # 'ITERATION NO' lines of pnlte.log besides the NLTE iterations (hydro model, iteration 0)
NITER_CAP_M424 = 102      # niter of the M424 models that stopped at the cap: ITMORE 100 + NITER_EXTRA
_OUT_NCOL = 6             # index, x, lambda, F_cont, F/F_cont, rotated flux
_TMP_RE = re.compile(r"\.tmp\d*\.npz$")     # temporary names of interrupted writes (legacy, io.save_npz)
_MAX_OPEN = 256           # combine keeps the merged files open (one handle each) up to this many files

_trapz = getattr(np, "trapezoid", None) or np.trapz


class FullLoadWarning(UserWarning):
    """An uncompressed .npz member could not be memory-mapped and was loaded into memory as a whole."""


def _status_ok(status, n):
    """status == 'ok' as a bool (n,) array; also for the empty float64 status of a merge without points
    (numpy < 1.25 compares an empty float array with a string to the scalar False)."""
    if status is None:
        return np.ones(n, bool)
    st = np.asarray(status)
    if st.size == 0:
        return np.zeros(st.shape, bool)
    return st == "ok"


def niter_cap_from_indat(indat, extra=NITER_EXTRA):
    """
    The iteration cap of a run in units of ``niter`` (meta.txt) from its INDAT.DAT.

    Parameters
    ----------
    indat: str
        INDAT.DAT file name, or its text (anything containing a newline).
    extra: int
        'ITERATION NO' lines of pnlte.log besides the NLTE iterations 1..ITMORE (default 2: 'FINAL
        ITERATION NO . 7 FOR HYDRO MODEL' and 'ITERATION NO 0'), which fw_sphere_point.sh counts too.

    Returns
    -------
    int
        ITMORE + extra (M424 template: ITMORE = 100 -> 102 = :data:`NITER_CAP_M424`).

    Raises
    ------
    ValueError
        If the second line does not hold OPTNEUPDATE, HE_ONE, ITSTART, ITMORE, or ITSTART != 0 (a restart
        prints a different set of start lines, so the offset is not known).

    Notes
    -----
    pnlte stops at iteration ITSTART + ITMORE (nlte.f90:2240, 2274) when it has not converged. Checked on the
    M424 ITMORE test: the reruns with ITMORE = 300 that did not converge have 302 'ITERATION NO' lines.
    """
    # PP 2026-10-01: new (reviewer: the cap must come from the run configuration, not from max(niter))
    if "\n" in indat:
        text = indat
    else:
        with open(indat) as f:
            text = f.read()
    lines = text.splitlines()
    tok = re.split(r"[,\s]+", lines[1].strip()) if len(lines) > 1 else []
    try:
        itstart, itmore = int(tok[2]), int(tok[3])
    except (IndexError, ValueError):
        raise ValueError("INDAT line 2 is not 'OPTNEUPDATE HE_ONE ITSTART ITMORE': {!r}".format(
            lines[1] if len(lines) > 1 else "")) from None
    if itstart != 0:
        raise ValueError("ITSTART = {} (restart): the number of start lines in pnlte.log is not known; pass "
                         "the cap explicitly".format(itstart))
    return itmore + int(extra)


# ----------------------------------------------------------------------------------------------
# single FASTWIND output files
# ----------------------------------------------------------------------------------------------
def _read_bytes(src):
    if isinstance(src, (bytes, bytearray, memoryview)):
        return bytes(src)
    with open(src, "rb") as f:
        return f.read()


def _out_table(data, nrow, cols):
    """
    Columns ``cols`` of the first ``nrow`` rows of an OUT table, float64 (nrow, len(cols)); bit-identical
    to ``np.genfromtxt(BytesIO(data), usecols=cols, max_rows=nrow)``.

    The fast path splits the lines and converts every token with Python's float (what genfromtxt's
    float converter does); anything unusual (blank or comment lines, a wrong number of columns, a token
    float() rejects) falls back to genfromtxt itself.
    """
    # PP 2026-10-01: replaces np.genfromtxt(io.BytesIO(...), usecols=[2, 3, 4], max_rows=NROW) of
    # fw_sphere_merge.py:107 (2x faster; same numbers, checked bitwise on real files in the tests)
    lines = data.split(b"\n", nrow)[:nrow]
    if len(lines) == nrow:
        rows = [ln.split() for ln in lines]
        if all(len(r) == _OUT_NCOL for r in rows):
            try:
                vals = np.array([float(t) for r in rows for t in r], dtype=np.float64)
            except ValueError:
                vals = None
            if vals is not None:
                return vals.reshape(nrow, _OUT_NCOL)[:, list(cols)]
    arr = np.genfromtxt(io.BytesIO(data), usecols=list(cols), max_rows=nrow)
    return np.asarray(arr, dtype=np.float64).reshape(nrow, len(cols))


def read_out(path, nrow=None):
    """
    Read one FASTWIND line-profile file ``OUT.<LINE>_<suffix>`` (from pformalsol).

    Parameters
    ----------
    path: str or bytes
        File name, or the file content (e.g. read from a tar archive).
    nrow: int, optional
        Number of table rows (M424: 161). Default: detected, i.e. the leading lines with six columns.

    Returns
    -------
    dict
        ``k`` int64 (row index), ``x`` (FASTWIND's frequency variable), ``lam`` (air wavelength
        [Angstrom], printed to 0.01 Angstrom, so it can repeat), ``fcont`` (continuum flux),
        ``fnorm`` (F / F_cont), ``frot`` (FASTWIND's 'rotated' flux; unreliable): float64 (nrow,);
        ``ew_fastwind``: the number on the trailer line, FASTWIND's own equivalent width [Angstrom],
        negative for absorption (NaN if there is no trailer; which column FASTWIND integrates is not
        checked here, and frot == fnorm in the M424 files); ``nrow``.

    Notes
    -----
    Numbers equal ``np.genfromtxt(path, usecols=..., max_rows=nrow)`` bit for bit (the legacy readers).
    """
    # PP 2026-10-01: generalises the readers of fw_itmore_check.py:42-44, fig_fastwind_variations.py:46-49,
    # fig_fastwind_mdot.py:25-28 (genfromtxt, usecols=[2, 4], max_rows=161)
    data = _read_bytes(path)
    lines = data.split(b"\n")
    if nrow is None:
        nrow = 0
        for ln in lines:
            if len(ln.split()) != _OUT_NCOL:
                break
            nrow += 1
        if nrow == 0:
            raise ValueError("no six-column table at the start of the OUT file")
    nrow = int(nrow)
    if nrow < 1 or len(lines) < nrow:
        raise ValueError("OUT file has fewer than {} rows".format(nrow))
    t = _out_table(data, nrow, range(_OUT_NCOL))
    ew = np.nan
    for ln in lines[nrow:]:
        tok = ln.split()
        if tok:
            if len(tok) == 1:
                try:
                    ew = float(tok[0])
                except ValueError:
                    pass
            break
    return dict(k=np.rint(t[:, 0]).astype(np.int64), x=t[:, 1].copy(), lam=t[:, 2].copy(), fcont=t[:, 3].copy(),
                fnorm=t[:, 4].copy(), frot=t[:, 5].copy(), ew_fastwind=ew, nrow=nrow)


def read_out_imu(path, counts=False):
    """
    Read ``OUT_IMU.<LINE>_<suffix>`` of the modified pformalsol (patch formalsol_imu.patch of the
    M424 project): emergent continuum and line intensity of every ray of FASTWIND's formal solution.

    Parameters
    ----------
    path: str
        File name.
    counts: bool
        Also return the ray counts of the header line ``# rays NP-1, core rays NC = <nray> <ncore>``.

    Returns
    -------
    lam: np.ndarray
        (nk,) wavelengths [Angstrom, air], printed to more digits than in OUT.
    p: np.ndarray
        (nray,) impact parameters in units of the inner-boundary radius.
    Ic, Il: np.ndarray
        (nk, nray) continuum and line intensity.
    counts: dict
        ``nray``, ``ncore`` (only with ``counts=True``).

    Raises
    ------
    ValueError
        If the header counts and the number of p values or table columns disagree.

    Notes
    -----
    The arrays are those of the legacy ``fw_disc.read_imu`` (same parsing, bit for bit).
    """
    # PP 2026-10-01: ported from fw_disc.py:233-241 (read_imu); header counts and checks are new
    with open(path) as f:
        h1 = f.readline()
        h2 = f.readline()
    p = np.array(h2.split()[2:], dtype=float)
    nray = ncore = None
    if "=" in h1:
        tail = h1.split("=", 1)[1].split()
        if len(tail) >= 2:
            nray, ncore = int(tail[0]), int(tail[1])
    d = np.loadtxt(path, comments="#", ndmin=2)
    n = p.size
    if nray is not None and nray != n:
        raise ValueError("{}: header says {} rays, p has {} values".format(path, nray, n))
    if d.shape[1] != 2 + 2 * n:
        raise ValueError("{}: {} columns, expected 2 + 2 x {}".format(path, d.shape[1], n))
    out = (d[:, 1], p, d[:, 2:2 + n], d[:, 2 + n:2 + 2 * n])
    if counts:
        return out + (dict(nray=nray, ncore=ncore),)
    return out


def parse_meta(text):
    """
    One ``meta.txt`` of a per-point model: ``idx teff status niter T_tau23 t_pnlte t_formal``.

    Returns
    -------
    dict
        idx (int), teff (float, K), status (str), niter (int), T_tau23 (float, K; NaN when pnlte
        did not print it), t_pnlte, t_formal (float, s).
    """
    # PP 2026-10-01: ported from fw_sphere_merge.py:102-103
    if isinstance(text, (bytes, bytearray)):
        text = text.decode()
    f = text.split()
    return dict(idx=int(f[0]), teff=float(f[1]), status=f[2], niter=int(f[3]), T_tau23=float(f[4]),
                t_pnlte=float(f[5]), t_formal=float(f[6]))


# ----------------------------------------------------------------------------------------------
# packed results -> merged file per tag
# ----------------------------------------------------------------------------------------------
def iter_part_points(part, want=("meta.txt", "OUT."), check_names=True):
    """
    Stream one packed part (``part_*.tar.gz``) once, in order, and yield every point's files.

    Parameters
    ----------
    part: str
        Archive (any compression tarfile reads in stream mode).
    want: tuple of str
        File-name prefixes to read (others are skipped without being kept).
    check_names: bool
        Validate every file member with ``tarfile.data_filter`` where Python has it (3.9.17+; the
        'data' extraction filter): a name that would leave the archive's directory ('..') raises
        ValueError. Nothing is ever written to disk (the files are read into memory), so this only
        guards against corrupt or hostile archives.

    Yields
    ------
    pdir: str
        The point's directory, e.g. ``P001563``.
    files: dict
        file name -> bytes, for the wanted files of that directory.

    Notes
    -----
    Same grouping as the legacy merge: the files of a directory are collected until the directory
    changes (random access in a .tar.gz would decompress from the start); a group without
    ``meta.txt`` is dropped. Top-level files (no directory) are ignored.
    """
    # PP 2026-10-01: ported from fw_sphere_merge.py:114-132
    filt = getattr(tarfile, "data_filter", None) if check_names else None
    ferr = getattr(tarfile, "FilterError", ValueError)
    dest = os.path.dirname(os.path.abspath(part))
    want = tuple(want)
    cur, files = None, {}
    with tarfile.open(part, mode="r|*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            if filt is not None:
                try:
                    filt(m, dest)
                except ferr as e:
                    raise ValueError("unsafe member {!r} in {}: {}".format(m.name, part, e)) from e
            name = m.name.lstrip("./")
            if "/" not in name:
                continue
            pdir, fname = name.rsplit("/", 1)
            if pdir != cur:
                if cur is not None and "meta.txt" in files:
                    yield cur, files
                cur, files = pdir, {}
            if fname.startswith(want):
                files[fname] = tf.extractfile(m).read()
    if cur is not None and "meta.txt" in files:
        yield cur, files


def _parts_of(results_dir, tag):
    return sorted(glob.glob(os.path.join(results_dir, tag, "part_*.tar.gz")))


def merge_task(results_dir, tag, out, points, lines=LINES_M424, suffix="VTV010", nrow=NROW_M424,
               copy_keys=COPY_KEYS, teff_nudge_max=TEFF_NUDGE_MAX, meta=None, niter_cap=None, log=None):
    """
    Merge the packed results of one tag (``results_dir/<tag>/part_*.tar.gz``) into one .npz.

    Parameters
    ----------
    results_dir: str
        ``RUN_DIR/results``.
    tag: str
        Task directory (``task_0007``, ``task_missing_0000``) or a glob pattern (``task_*``). A pattern
        merges all its tags here, in this process: memory O(all their points) and the CPU time of all
        of them in one process (M424: ~520 s CPU per tag, so the 41 tags would exceed a 3600 s ``ulimit -t``
        of a login node); use :func:`merge_tasks` for many tags.
    out: str
        Output .npz (written atomically, through the hidden temporary ``.<name>.tmp<pid>.npz``).
    points: str or mapping
        ``points.npz`` of the run (path, or a mapping such as ``dict(np.load(...))`` or an open
        NpzFile): ``idx`` (= row number), ``teff`` and ``copy_keys``.
    lines: sequence of str
        Line names; their OUT files are read in this order.
    suffix: str
        OUT file suffix (``VTV010``: vturb 10 km/s).
    nrow: int
        Rows of every OUT table (161).
    copy_keys: sequence of str
        Per-point members copied from ``points``.
    teff_nudge_max: float
        Largest accepted T_eff(meta.txt) - T_eff(points) [K]; larger differences raise ValueError.
    meta: dict, optional
        Provenance record stored as '_meta' (:func:`ppmpy.synspec.io.make_meta`); default none, which
        keeps the file identical to the legacy one.
    niter_cap: int, optional
        The run's iteration cap in units of niter (:func:`niter_cap_from_indat`; M424:
        :data:`NITER_CAP_M424`), recorded as ``niter_cap`` in '_meta' (which is then written even without
        ``meta``) for :func:`combine` and :meth:`ProfileStore.usable`.
    log: callable, optional
        Receives the legacy summary lines (e.g. ``print``).

    Returns
    -------
    dict
        path, parts, npoint, status (name -> count), nnudge.

    Notes
    -----
    Each part is streamed once (:func:`iter_part_points`). A point packed twice keeps the first
    record, unless a later one is successful and the kept one is not. Output keys, dtypes and member
    order are those of ``fw_sphere_merge.py --tags``; with ``meta=None`` and ``niter_cap=None`` the file is
    byte-identical.
    Peak memory is O(points of the tag): ~12 KB per point for three lines (the records and the arrays
    built from them). Run time is set by the decompression of the parts. M424 task_0039 (34 parts,
    105 GB packed with the model files, 30 913 points): 733 s, peak anonymous memory 0.43 GB on the
    login node, output byte-identical to the production merged/task_0039.npz.
    """
    # PP 2026-10-01: ported from fw_sphere_merge.py:97-158 (per-tag merge)
    lines = list(lines)
    nl = len(lines)
    rec = {}
    parts = _parts_of(results_dir, tag)
    for part in parts:
        for pdir, files in iter_part_points(part):
            m = parse_meta(files["meta.txt"])
            prof = np.full((3, nl, nrow), np.nan, np.float32)
            if m["status"] == "ok":
                for j, ln in enumerate(lines):
                    name = "OUT.{}_{}".format(ln, suffix)
                    if name not in files:
                        raise KeyError("{}: point {} has status ok but no {}".format(part, pdir, name))
                    prof[:, j, :] = _out_table(files[name], nrow, (2, 3, 4)).T
            old = rec.get(m["idx"])
            if old is None or (old[2] != "ok" and m["status"] == "ok"):
                rec[m["idx"]] = (m["idx"], m["teff"], m["status"], m["niter"], m["T_tau23"], m["t_pnlte"],
                                 m["t_formal"], prof)
    idx = np.array(sorted(rec), dtype=np.int32)
    r = [rec[i] for i in idx]
    rec = None
    prof = np.array([x[7] for x in r], np.float32) if r else np.zeros((0, 3, nl, nrow), np.float32)
    arrays = dict(idx=idx, teff=np.array([x[1] for x in r]), status=np.array([x[2] for x in r]),
                  niter=np.array([x[3] for x in r], np.int16), T_tau23=np.array([x[4] for x in r]),
                  t_pnlte=np.array([x[5] for x in r], np.float32), t_formal=np.array([x[6] for x in r], np.float32),
                  lam=prof[:, 0], fcont=prof[:, 1], fnorm=prof[:, 2], lines=np.array(lines))
    with contextlib.ExitStack() as stack:
        pts = stack.enter_context(np.load(points)) if isinstance(points, (str, os.PathLike)) else points
        if not np.all(pts["idx"][idx] == idx):
            raise ValueError("points index mismatch: points['idx'][i] != i for merged points")
        for k in copy_keys:
            arrays[k] = pts[k][idx]
        dteff = arrays["teff"] - pts["teff"][idx]        # points.txt has T_eff to 1e-3 K
    arrays["teff_nudge"] = np.where(np.abs(dteff) <= 1e-3, 0.0, np.round(dteff, 3)).astype(np.float32)
    if not np.all((arrays["teff_nudge"] >= 0) & (arrays["teff_nudge"] <= teff_nudge_max)):
        raise ValueError("T_eff in the results differs from points by more than a retry nudge "
                         "(0..{} K)".format(teff_nudge_max))
    if niter_cap is not None:
        meta = dict(meta or {}, niter_cap=int(niter_cap))
    if meta is not None:
        arrays["_meta"] = _meta_array(meta)
    # PP 2026-10-01: written with the zipfile calls of np.savez (as io.save_npz, same bytes) but through a hidden
    # temporary name, which a 'task_*.npz' glob of combine cannot pick up after an interrupted write
    w = _NpzStream(out)
    try:
        for k, v in arrays.items():
            w.write_array(k, v)
        w.close()
    except BaseException:
        w.abort()
        raise
    st, cnt = np.unique(arrays["status"], return_counts=True)
    nnud = int((arrays["teff_nudge"] != 0).sum())
    summary = dict(path=out, parts=parts, npoint=int(idx.size),
                   status={str(s): int(c) for s, c in zip(st, cnt)}, nnudge=nnud)
    if log is not None:
        log("{} parts, {} points: ".format(len(parts), idx.size) + ", ".join("{} {}".format(s, c) for s, c in zip(st, cnt))
            + ("; {} with a Teff nudge".format(nnud) if nnud else ""))
        ok = _status_ok(arrays["status"], idx.size)
        if ok.any():
            log("pnlte time: median {:.0f} s, max {:.0f} s; iterations median {}, max {}".format(
                np.median(arrays["t_pnlte"][ok]), arrays["t_pnlte"][ok].max(), int(np.median(arrays["niter"][ok])),
                arrays["niter"][ok].max()))
        log("wrote {} ({:.1f} MB)".format(out, os.path.getsize(out) / 1e6))
    return summary


def _merge_one(args):
    results_dir, tag, out, points, kw = args
    return merge_task(results_dir, tag, out, points, **kw)


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _remove_stale_tmp(merged_dir, tags):
    """Remove the hidden temporaries ``.<tag>.tmp<pid>.npz`` of these tags left by writers that no longer run
    (e.g. pool workers terminated in the middle of a write); POSIX only (liveness check with signal 0)."""
    if os.name != "posix":
        return []
    removed = []
    for t in tags:
        for f in glob.glob(os.path.join(glob.escape(merged_dir), "." + glob.escape(t) + ".tmp*.npz")):
            m = re.search(r"\.tmp(\d+)\.npz$", f)
            if m is None or _pid_alive(int(m.group(1))):
                continue
            try:
                os.remove(f)
                removed.append(f)
            except FileNotFoundError:
                pass
    return removed


def merge_tasks(results_dir, merged_dir, points, tags="task_*", nproc=1, start_method=None, incremental=True,
                timeout=7200.0, log=None, **kw):
    """
    :func:`merge_task` for many tags, in parallel: ``results_dir/<tag>`` -> ``merged_dir/<tag>.npz``.

    Parameters
    ----------
    results_dir, merged_dir: str
        ``RUN_DIR/results`` and ``RUN_DIR/merged``.
    points: str
        Path of ``points.npz`` (a path, so that workers do not receive copies of the arrays).
    tags: str or sequence of str
        Glob pattern of task directories, or the tags themselves.
    nproc: int
        Worker processes. With more than one tag to merge, every tag runs in a fresh worker process
        (:func:`ppmpy.synspec.parallel.make_pool`, ``maxtasksperchild=1``), also for nproc = 1: memory
        is that of one tag per worker (~0.2 GB for 30 000 points; M424 task_0039 0.43 GB) and every tag
        starts a new CPU-time count (~520 s CPU per M424 tag, below the 3600 s ``ulimit -t`` of the
        Trillium login nodes). A single tag is merged in this process.
    start_method: str, optional
        'fork', 'spawn' or 'forkserver' (default: :func:`ppmpy.synspec.parallel.get_context`). Results do
        not depend on it.
    incremental: bool
        Skip a tag whose merged file is newer than all its parts (as fw_sphere_merge.sbatch).
    timeout: float or None
        Watchdog (:func:`ppmpy.synspec.parallel.imap_watchdog`): raise
        :class:`~ppmpy.synspec.parallel.PoolStalled` when no tag finishes for this long [s] (a worker
        killed by the out-of-memory killer or a CPU-time limit never returns its tag). An M424 tag takes
        ~12 min on a login node; raise it for much larger tags. None waits for ever.
    log: callable, optional
        Receives one line per merged tag.
    **kw:
        Passed to :func:`merge_task` (lines, suffix, nrow, copy_keys, teff_nudge_max, meta, niter_cap).

    Returns
    -------
    list of dict
        The summaries of the merged tags, in tag order (skipped tags are not listed).

    Notes
    -----
    If a merge fails (an exception in a worker, the watchdog, an interrupt), the pool is terminated, which
    can kill other workers in the middle of a write; the hidden temporaries of the tags of this call whose
    writer no longer runs are then removed before the exception propagates. Merged files already written
    stay (they are complete).
    """
    # PP 2026-10-01: ported from fw_sphere_merge.sbatch (incremental, xargs -P over the task directories)
    from . import parallel as par
    if isinstance(tags, str):
        tags = sorted(os.path.basename(d) for d in glob.glob(os.path.join(results_dir, tags)) if os.path.isdir(d))
    jobs = []
    for t in tags:
        out = os.path.join(merged_dir, t + ".npz")
        parts = _parts_of(results_dir, t)
        if incremental and os.path.exists(out):
            mt = os.path.getmtime(out)
            if not any(os.path.getmtime(p) > mt for p in parts):
                continue
        jobs.append((results_dir, t, out, points, kw))
    if not jobs:
        return []
    os.makedirs(merged_dir, exist_ok=True)
    res = []

    def _log(s):
        if log is not None:
            log("{}: {} points {}".format(os.path.basename(s["path"])[:-4], s["npoint"], s["status"]))

    if len(jobs) == 1:
        res.append(_merge_one(jobs[0]))
        _log(res[-1])
    else:
        # PP 2026-10-01: a fresh process per tag even for nproc = 1 (reviewer: 41 M424 tags in one process exceed the
        # 3600 s CPU limit of a login node); stale temporaries of terminated workers are removed after a failure
        try:
            with par.make_pool(min(nproc, len(jobs)), maxtasksperchild=1, start_method=start_method) as pool:
                for s in par.imap_watchdog(pool, _merge_one, jobs, timeout=timeout):
                    res.append(s)
                    _log(s)
        except BaseException:
            gone = _remove_stale_tmp(merged_dir, [j[1] for j in jobs])
            if gone:
                warnings.warn("merge_tasks failed; removed the temporaries of interrupted writes: {}".format(gone))
            raise
    order = {j[2]: i for i, j in enumerate(jobs)}
    return sorted(res, key=lambda s: order[s["path"]])


# ----------------------------------------------------------------------------------------------
# streaming .npz writer (byte-identical to numpy.savez)
# ----------------------------------------------------------------------------------------------
def _npy_header(shape, dtype):
    """The .npy header numpy.lib.format.write_array writes for a C-ordered array of this shape and dtype."""
    d = dict(descr=np.lib.format.dtype_to_descr(np.dtype(dtype)), fortran_order=False,
             shape=tuple(int(s) for s in shape))
    buf = io.BytesIO()
    try:
        np.lib.format.write_array_header_1_0(buf, d)
    except ValueError:                                   # header too long for format 1.0, as write_array
        buf = io.BytesIO()
        np.lib.format.write_array_header_2_0(buf, d)
    return buf.getvalue()


def _meta_array(meta):
    """The '_meta' member as io.save_npz writes it."""
    return np.array(json.dumps(meta, sort_keys=True, default=str))


class _CountingWriter:
    """Write-only file wrapper that counts the bytes passed through."""

    def __init__(self, fid):
        self.fid = fid
        self.nbytes = 0

    def write(self, data):
        n = data.nbytes if isinstance(data, np.ndarray) else memoryview(data).nbytes
        if isinstance(data, np.ndarray):
            data = data.reshape(-1).view(np.uint8)     # 1-D bytes: len() = nbytes, also for older zipfile
        self.fid.write(data)
        self.nbytes += n
        return n


class _NpzStream:
    """
    Uncompressed .npz written member by member, with the zipfile calls of numpy.savez (fixed entry
    dates, zip64 forced per member), so the file equals np.savez of the same arrays byte for byte.
    Written to the hidden temporary ``.<name>.tmp<pid>.npz`` in the target directory (no ``*.npz`` glob
    matches it) and moved into place by :meth:`close`; :meth:`abort` removes it.
    """

    def __init__(self, path):
        path = os.fspath(path)
        if not path.endswith(".npz"):
            raise ValueError("path must end in .npz: {}".format(path))
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        self.path = path
        self.tmp = os.path.join(d, ".{}.tmp{}.npz".format(os.path.basename(path)[:-4], os.getpid()))
        self.zf = zipfile.ZipFile(self.tmp, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True)

    def write_array(self, key, arr):
        with self.zf.open(key + ".npy", "w", force_zip64=True) as fid:
            np.lib.format.write_array(fid, np.asanyarray(arr), allow_pickle=False)

    @contextlib.contextmanager
    def member(self, key, shape, dtype):
        """
        File object for the data of a C-ordered member. The bytes written are counted: anything but
        prod(shape) x itemsize raises ValueError when the block exits (the caller then aborts).
        """
        # PP 2026-10-01: byte count checked (reviewer: a short block would give a valid zip with a truncated .npy)
        expect = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        with self.zf.open(key + ".npy", "w", force_zip64=True) as fid:
            fid.write(_npy_header(shape, dtype))
            cw = _CountingWriter(fid)
            yield cw
            if cw.nbytes != expect:
                raise ValueError("member {}: wrote {} data bytes, the header promises {}".format(key, cw.nbytes, expect))

    def close(self):
        self.zf.close()
        os.replace(self.tmp, self.path)

    def abort(self):
        try:
            self.zf.close()
        finally:
            if os.path.exists(self.tmp):
                os.remove(self.tmp)


def _npz_layout(path):
    """Member name -> (shape, dtype, stored, fortran_order) of an .npz, in file order, from the .npy headers only."""
    out = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with zf.open(info) as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                else:
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            out[info.filename[:-4]] = (tuple(shape), dtype, info.compress_type == zipfile.ZIP_STORED, fortran)
    return out


def _member(path, key, layout=None, strict=False):
    """
    One member: a read-only memory map when it is stored uncompressed (and not empty), else loaded.
    A stored member that cannot be mapped is loaded with a :class:`FullLoadWarning` naming its size
    (strict=True re-raises the error instead).
    """
    lay = (layout or _npz_layout(path))[key]
    shape, dtype, stored, _ = lay
    if stored and len(shape) > 0 and int(np.prod(shape)) > 0 and not dtype.hasobject:
        try:
            return npz_member_memmap(path, key)
        except (ValueError, OSError) as e:
            if strict:
                raise
            # PP 2026-10-01: no silent full load (reviewer: 2.4 GB per member of the M424 profiles.npz)
            warnings.warn("{}: member {!r} ({:.3g} GB) cannot be memory-mapped ({}: {}); loading it into memory".format(
                path, key, int(np.prod(shape, dtype=np.int64)) * dtype.itemsize / 1e9, type(e).__name__, e),
                FullLoadWarning, stacklevel=2)
    with np.load(path) as z:
        return z[key]


def _expand(files):
    """Merged files: a glob pattern (sorted; temporaries of interrupted writes are skipped) or a list."""
    if isinstance(files, (str, os.PathLike)):
        found = sorted(glob.glob(os.fspath(files)))
        tmp = [f for f in found if _TMP_RE.search(f)]
        if tmp:
            # PP 2026-10-01: the legacy glob took them (BadZipFile, or duplicate records of a finished write)
            warnings.warn("ignoring temporary files of interrupted writes: {}".format(tmp))
            found = [f for f in found if not _TMP_RE.search(f)]
        return found
    return [os.fspath(f) for f in files]


class _RowSource:
    """
    Rows of one member of one merged file for the combine. An uncompressed C-ordered member is read
    with explicit reads (``readinto`` at offset + row0 x row bytes) of the row range asked for, so
    neither anonymous nor file-backed memory grows with the file (a memory map would leave every page
    it touched resident: ~ the total size of a member over all files). Anything else (compressed,
    Fortran order) is loaded once (:func:`_member`). With ``keep_open`` the file stays open between
    reads (an open() on a network file system costs ~0.3 ms) until :meth:`close`.
    """

    def __init__(self, path, key, layout, strict=False, keep_open=True):
        shape, dtype, stored, fortran = layout[key]
        self.path, self.key, self.dtype, self.tail = path, key, dtype, tuple(shape[1:])
        self.rowbytes = int(np.prod(self.tail, dtype=np.int64)) * dtype.itemsize
        self.keep_open = keep_open
        self.fh = None
        self.offset = self.arr = None
        if stored and not fortran and not dtype.hasobject and len(shape) > 0 and shape[0] > 0 and self.rowbytes > 0:
            try:
                self.offset = int(npz_member_memmap(path, key).offset)     # header only, nothing is read
            except (ValueError, OSError):
                pass
        if self.offset is None:
            self.arr = _member(path, key, layout, strict)

    def rows(self, loc):
        """Rows ``loc`` (int array, small spread: the rows of one gather block) in the file's dtype."""
        if self.arr is not None:
            return self.arr[loc]
        lo, hi = int(loc.min()), int(loc.max()) + 1
        buf = np.empty((hi - lo,) + self.tail, self.dtype)
        f = self.fh if self.fh is not None else open(self.path, "rb", buffering=0)
        try:
            f.seek(self.offset + lo * self.rowbytes)
            want, got = buf.nbytes, 0
            view = buf.reshape(-1).view(np.uint8)
            while got < want:                                       # raw reads may return less than asked
                n = f.readinto(view[got:])
                if not n:
                    break
                got += n
        finally:
            if self.keep_open:
                self.fh = f
            else:
                f.close()
        if got != want:
            raise IOError("{}: short read of member {} ({} of {} bytes)".format(self.path, self.key, got, want))
        return buf[loc - lo]

    def close(self):
        if self.fh is not None:
            self.fh.close()
            self.fh = None


# ----------------------------------------------------------------------------------------------
# merged files -> profiles.npz + missing.txt
# ----------------------------------------------------------------------------------------------
def combine(merged_files, points_txt, out_dir, teff_nudge=TEFF_NUDGE, block=20000, profiles_name="profiles.npz",
            missing_name="missing.txt", meta=None, niter_cap=None, log=None):
    """
    Combine merged per-tag files into ``profiles.npz`` and list the points without a successful model
    in ``missing.txt``, streaming (the concatenated profiles are never held in memory).

    Parameters
    ----------
    merged_files: str or sequence of str
        The merged files in the legacy order (``sorted(glob(RUN_DIR/merged/task_*.npz))``); a string is
        taken as a glob pattern and sorted, skipping temporaries of interrupted writes (``*.tmp.npz``,
        ``*.tmp<pid>.npz``) with a warning. The order decides which of two equal records is kept.
    points_txt: str
        ``points.txt`` of the run (``idx teff`` per line).
    out_dir: str
        Directory of the two outputs (both written atomically).
    teff_nudge: float
        K added to the T_eff of a failed point per failed attempt (the largest nudge already tried plus
        this) in ``missing.txt``; points that were never run keep their T_eff.
    block: int
        Rows per gather step: memory ~ 2 x block x row size of the largest member (lam: 3 x 161 x 4 B),
        i.e. ~4 MB per 1000 rows for three lines (77 MB for the default). The output does not depend on it.
    profiles_name, missing_name: str
        Output file names.
    meta: dict, optional
        Provenance record stored as '_meta'; default none (byte-identical to the legacy file).
    niter_cap: int, optional
        Iteration cap recorded as ``niter_cap`` in '_meta' (written then even without ``meta``). Default:
        the ``niter_cap`` in the '_meta' of the merged files (:func:`merge_task`) when all of them hold
        the same value; if only some hold one, or they differ (e.g. retries run with another ITMORE),
        nothing is recorded and a warning says so. Legacy merged files hold none: nothing is recorded.
    log: callable, optional
        Receives the legacy summary lines.

    Returns
    -------
    (str, str)
        Paths of the profiles file and of the missing list.

    Notes
    -----
    Same result as ``fw_sphere_merge.py --combine``: all records concatenated in file order, sorted by
    idx (stable), and per idx the first successful record kept (else the first one); member order and
    dtypes of np.concatenate (e.g. the widest status string; an empty merged file widens it to <U32, as
    in the legacy). The output is written member by member in blocks of rows read from the merged files
    with explicit reads (:class:`_RowSource`; no memory maps, so no file pages stay mapped), so the
    memory is O(number of records) for the selection arrays (~50 B per record) plus O(block) for the
    profiles, instead of 15-22 GB. M424 (41 files, 1 236 544 points, block 20 000): 11 s with a warm page
    cache, peak RSS 0.18 GB (selection ~0.07 GB, gather +0.08 GB); the merged files stay open (one handle
    each, up to 256 files) while a member is written.
    ``missing.txt`` uses np.savetxt with the legacy formats ('%d', '%.3f').
    """
    # PP 2026-10-01: ported from fw_sphere_merge.py:57-95 (--combine), rewritten as a streaming gather
    files = _expand(merged_files)
    if not files:
        raise ValueError("no merged files given")
    layouts = [_npz_layout(f) for f in files]
    keys = [k for k in layouts[0] if k not in ("lines", "_meta")]
    for f, lay in zip(files, layouts):
        miss_k = [k for k in keys if k not in lay]
        if miss_k:
            raise KeyError("{} lacks members {}".format(f, miss_k))
    # output dtype and shape per member, as np.concatenate would give them (dtype rules ignore values)
    spec = {}
    for k in keys:
        empties = [np.empty((0,) + lay[k][0][1:], lay[k][1]) for lay in layouts]
        spec[k] = np.concatenate(empties)
    sizes = np.array([lay["idx"][0][0] for lay in layouts], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    # selection from idx and status only
    idx_l, ok_l, fi_l, fn_l, caps = [], [], [], [], []
    for f in files:
        with np.load(f) as z:
            i, st, nud = z["idx"], z["status"], z["teff_nudge"]
            # PP 2026-10-01: an empty merge stores status as empty float64 (numpy < 1.25: '== "ok"' -> scalar False)
            ok = _status_ok(st, i.size)
            idx_l.append(i)
            ok_l.append(ok)
            fi_l.append(i[~ok])
            fn_l.append(nud[~ok])
            caps.append(json.loads(str(z["_meta"])).get("niter_cap") if "_meta" in z.files else None)
    if niter_cap is None and any(c is not None for c in caps):
        if all(c == caps[0] for c in caps):
            niter_cap = caps[0]
        else:
            warnings.warn("the merged files record different or missing iteration caps ({}); none is recorded in {} "
                          "(pass niter_cap, or a cap to ProfileStore.usable)".format(sorted(set(map(str, caps))),
                                                                                    profiles_name))
    if niter_cap is not None:
        meta = dict(meta or {}, niter_cap=int(niter_cap))
    with np.load(files[0]) as z:
        lines_arr = z["lines"]
    for f in files[1:]:
        with np.load(f) as z:
            if not np.array_equal(z["lines"], lines_arr):
                warnings.warn("{}: lines {} differ from those of {} (the first file's are kept, as before)".format(
                    f, z["lines"].tolist(), files[0]))
    idx_cat = np.concatenate(idx_l)
    ok_cat = np.concatenate(ok_l)
    # largest T_eff nudge already tried for each point that has failed (before duplicates are dropped)
    fi = np.concatenate(fi_l).astype(np.int64)
    fn = np.concatenate(fn_l).astype(np.float64)
    tried_idx, inv = np.unique(fi, return_inverse=True)
    tried_val = np.full(tried_idx.size, -np.inf)
    np.maximum.at(tried_val, inv, fn)
    idx_l = ok_l = fi_l = fn_l = fi = fn = inv = None
    # sort (stable) and keep, per idx, the first successful record, else the first one
    o = np.argsort(idx_cat, kind="stable")
    M = o.size
    if M:
        _, first = np.unique(idx_cat[o], return_index=True)
        pos = np.arange(M, dtype=np.int64)
        key = np.where(ok_cat[o], pos, M + pos)
        mn = np.minimum.reduceat(key, first)
        chosen = np.where(mn >= M, mn - M, mn)
        sel = o[chosen]
    else:
        sel = np.zeros(0, np.int64)
    o = key = pos = None
    nout = sel.size
    pid = np.searchsorted(offsets[1:], sel, side="right")
    loc = sel - offsets[pid]
    out_idx = idx_cat[sel]
    out_ok = ok_cat[sel]
    idx_cat = ok_cat = None
    os.makedirs(out_dir, exist_ok=True)
    ppath = os.path.join(out_dir, profiles_name)
    block = max(int(block), 1)
    keep_open = len(files) <= _MAX_OPEN
    st_count, nud_n, nud_max = {}, 0, -np.inf          # for the log, collected while writing (no reload)
    w = _NpzStream(ppath)
    try:
        for k in keys:
            dt, tail = spec[k].dtype, spec[k].shape[1:]
            srcs = [None] * len(files)
            try:
                with w.member(k, (nout,) + tail, dt) as fid:
                    for b0 in range(0, nout, block):
                        b1 = min(nout, b0 + block)
                        p_b, l_b = pid[b0:b1], loc[b0:b1]
                        buf = np.empty((b1 - b0,) + tail, dt)
                        for t in np.unique(p_b):
                            if srcs[t] is None:
                                srcs[t] = _RowSource(files[t], k, layouts[t], keep_open=keep_open)
                            m = p_b == t
                            buf[m] = srcs[t].rows(l_b[m])
                        fid.write(buf)
                        if log is not None and k == "status":
                            for a, c in zip(*np.unique(buf, return_counts=True)):
                                st_count[a] = st_count.get(a, 0) + int(c)
                        elif log is not None and k == "teff_nudge":
                            nud_n += int((buf != 0).sum())
                            nud_max = max(nud_max, float(buf.max()))
            finally:
                for src in srcs:
                    if src is not None:
                        src.close()
            srcs = None
        w.write_array("lines", lines_arr)
        if meta is not None:
            w.write_array("_meta", _meta_array(meta))
        w.close()
    except BaseException:
        w.abort()
        raise
    # missing.txt: points of points.txt without a successful model
    allidx = np.loadtxt(points_txt, ndmin=2)
    good = np.unique(out_idx[out_ok]).astype(np.int64)
    miss = allidx[~np.isin(allidx[:, 0].astype(np.int64), good)]
    mi = miss[:, 0].astype(np.int64)
    j = np.searchsorted(tried_idx, mi)
    hit = j < tried_idx.size
    hit[hit] = tried_idx[j[hit]] == mi[hit]
    miss[hit, 1] = miss[hit, 1] + (tried_val[j[hit]] + teff_nudge)
    nfail = int(hit.sum())
    mpath = os.path.join(out_dir, missing_name)
    tmp = os.path.join(out_dir, ".{}.tmp{}".format(missing_name, os.getpid()))
    try:
        np.savetxt(tmp, miss, fmt=["%d", "%.3f"])
        os.replace(tmp, mpath)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    if log is not None:
        log("combined {} task files: {} points (".format(len(files), nout)
            + ", ".join("{} {}".format(a, st_count[a]) for a in sorted(st_count))
            + "); {} of {} points without a successful model -> {}".format(miss.shape[0], allidx.shape[0], missing_name)
            + " ({} failed models listed with Teff + {:g} K per failed attempt, {} never run)".format(
                nfail, teff_nudge, miss.shape[0] - nfail))
        if nud_n:
            log("{} points in {} were computed with a Teff nudge (max {:g} K)".format(nud_n, profiles_name, nud_max))
    return ppath, mpath


# ----------------------------------------------------------------------------------------------
# reading the merged profiles
# ----------------------------------------------------------------------------------------------
class ProfileStore:
    """
    The per-point profiles of a run (``profiles.npz`` or a merged per-task file) with zero-copy access.

    Open with :meth:`open`; members stored uncompressed are read-only memory maps
    (:func:`ppmpy.synspec.io.npz_member_memmap`), so opening the 7.35 GB M424 file reads nothing but
    the headers. Members that are compressed are loaded into memory; a stored member that cannot be
    mapped is loaded with a :class:`FullLoadWarning` (``strict=True``: the error is raised instead).

    Memory: reading through the memory maps keeps the anonymous memory O(block), but the file pages a
    process has touched count in its resident set (clean pages, which the kernel drops under memory
    pressure, but Slurm/cgroup accounting sees them): a pass over all profiles of one member, e.g.
    :func:`ew_per_point` or ``usable(finite=True)``, can show up to the member's size (M424: 2.4 GB per
    member) in RSS.

    Pickling (e.g. to a 'spawn' worker): a store opened from a file is re-opened from its path in the
    receiving process, always with memory maps (``mmap=True``, whatever the sender used), so no array is
    copied; a store built from arrays sends its arrays and its '_meta'.

    ``ProfileStore(members, path=None, mmap=True, strict=False)`` builds a store from a mapping name ->
    array (at least idx, lam, fcont, fnorm, lines; '_meta' optional, JSON text as written by
    :func:`ppmpy.synspec.io.save_npz`).

    Attributes
    ----------
    path: str or None
        The file (None for a store built from arrays).
    n: int
        Number of points.
    lines: list of str
        Line names (second axis of lam/fcont/fnorm).
    nrow: int
        Rows of every profile.
    idx, teff, status, niter, teff_nudge: np.ndarray
        Per-point members (None when absent).
    coordinates: dict
        Those of r, theta, phi, x, y, z that are present.
    meta: dict
        The '_meta' record ({} for the legacy files).
    niter_cap: int or None
        The iteration cap recorded in '_meta' (``niter_cap`` of :func:`merge_task` / :func:`combine`);
        None when unknown (legacy files: M424 used :data:`NITER_CAP_M424`).
    """

    PROFILE_KEYS = ("lam", "fcont", "fnorm")
    COORD_KEYS = ("r", "theta", "phi", "x", "y", "z")

    def __init__(self, members, path=None, mmap=True, strict=False):
        # PP 2026-10-01: new (replaces np.load(profiles.npz) in fw_disc.load_run and the figure scripts)
        self._m = {k: v for k, v in members.items() if k != "_meta"}
        self.meta = {}
        if "_meta" in members:
            self.meta = json.loads(str(np.asarray(members["_meta"])))
        for k in ("idx", "lam", "fcont", "fnorm", "lines"):
            if k not in self._m:
                raise KeyError("profile store needs member {!r}".format(k))
        self.path = path
        self._mmap = mmap
        self._strict = strict
        cap = self.meta.get("niter_cap")
        self.niter_cap = None if cap is None else int(cap)
        self.n = int(self._m["idx"].shape[0])
        self.lines = [str(s) for s in np.asarray(self._m["lines"]).tolist()]
        self.nrow = int(self._m["lam"].shape[2])
        self.idx = self._m["idx"]
        self.teff = self._m.get("teff")
        self.status = self._m.get("status")
        self.niter = self._m.get("niter")
        self.teff_nudge = self._m.get("teff_nudge")
        self.coordinates = {k: self._m[k] for k in self.COORD_KEYS if k in self._m}

    @classmethod
    def open(cls, path, mmap=True, strict=False):
        """
        Open a profiles file.

        Parameters
        ----------
        path: str
            ``profiles.npz`` (legacy, no '_meta') or any .npz with the same members.
        mmap: bool
            Memory-map the uncompressed members (default); False loads every member (7.35 GB for M424).
        strict: bool
            Raise instead of loading (with a :class:`FullLoadWarning`) a stored member that cannot be
            memory-mapped.
        """
        lay = _npz_layout(path)
        members = {}
        if mmap:
            for k in lay:
                members[k] = _member(path, k, lay, strict=strict)
        else:
            with np.load(path) as z:
                members = {k: z[k] for k in z.files}
        return cls(members, path=os.path.abspath(path), mmap=mmap, strict=strict)

    def __reduce__(self):
        # PP 2026-10-01: a store opened from a file is re-opened from its path (always memory-mapped: a store opened
        # with mmap=False would otherwise load the whole file in every worker); one built from arrays keeps its '_meta'
        if self.path is not None:
            return (ProfileStore.open, (self.path, True, self._strict))
        m = dict(self._m)
        if self.meta:
            m["_meta"] = _meta_array(self.meta)
        return (ProfileStore, (m, None, self._mmap, self._strict))

    def __repr__(self):
        return "ProfileStore({}, n={}, lines={}, nrow={})".format(self.path, self.n, self.lines, self.nrow)

    def __contains__(self, key):
        return key in self._m

    def __getitem__(self, key):
        """Any member by name (memory map or array)."""
        return self._m[key]

    def keys(self):
        return list(self._m)

    def _j(self, j):
        return self.lines.index(j) if isinstance(j, str) else int(j)

    def lam(self, j):
        """Wavelengths [Angstrom] of line j (index or name): (n, nrow) float32 view, no copy."""
        return self._m["lam"][:, self._j(j), :]

    def fnorm(self, j):
        """F / F_cont of line j: (n, nrow) float32 view."""
        return self._m["fnorm"][:, self._j(j), :]

    def fcont(self, j):
        """Continuum flux of line j: (n, nrow) float32 view."""
        return self._m["fcont"][:, self._j(j), :]

    def fc(self, j, col=0):
        """
        Continuum flux of line j at profile row ``col`` as float64 (n,) (the disc-integration weight
        F_c of the legacy library: ``fcont[:, :, 0]``). j=None gives all lines, (n, nline).
        """
        if j is None:
            return np.array(self._m["fcont"][:, :, col], dtype=np.float64)
        return np.array(self._m["fcont"][:, self._j(j), col], dtype=np.float64)

    def iter_blocks(self, j, block=20000, rows=None):
        """
        Iterate over the profiles of line j in blocks of points.

        Parameters
        ----------
        j: int or str
            Line.
        block: int
            Points per block.
        rows: array of int or bool, optional
            Subset of points: row numbers in 0..n-1 (increasing for efficient reads), or a boolean mask
            of length n; default all. Checked when iter_blocks is called (ValueError, TypeError).

        Yields
        ------
        i0, i1: int
            Block bounds, positions in ``rows`` (in 0..n without rows).
        lam: np.ndarray
            (i1 - i0, nrow) float64 (``lam.astype(np.float64)``, as the legacy library).
        fnorm: np.ndarray
            (i1 - i0, nrow) in the stored dtype (float32).
        """
        # PP 2026-10-01: block reads of fw_disc.py:81-83 (library) and :172-175 (integrate_exact)
        jj = self._j(j)
        block = max(int(block), 1)
        rows = None if rows is None else self._check_rows(rows)
        return self._iter_blocks(jj, block, rows)

    def _check_rows(self, rows):
        # PP 2026-10-01: reviewer: a mask of the wrong length was accepted and selected the wrong points
        rows = np.asarray(rows)
        if rows.ndim != 1:
            raise ValueError("rows must be one-dimensional, got shape {}".format(rows.shape))
        if rows.dtype == bool:
            if rows.shape != (self.n,):
                raise ValueError("boolean rows mask has length {}, the store has n = {}".format(rows.size, self.n))
            return np.flatnonzero(rows)
        if rows.size == 0:
            return rows.astype(np.intp)
        if rows.dtype.kind not in "iu":
            raise TypeError("rows must be integers or a boolean mask, got dtype {}".format(rows.dtype))
        if rows.min() < 0 or rows.max() >= self.n:
            raise ValueError("rows must lie in 0..{} (got {}..{})".format(self.n - 1, rows.min(), rows.max()))
        return rows

    def _iter_blocks(self, jj, block, rows):
        L, F = self._m["lam"], self._m["fnorm"]
        if rows is None:
            for i0 in range(0, self.n, block):
                i1 = min(self.n, i0 + block)
                yield i0, i1, np.asarray(L[i0:i1, jj]).astype(np.float64), np.array(F[i0:i1, jj])
            return
        for i0 in range(0, rows.size, block):
            ii = rows[i0:i0 + block]
            yield i0, i0 + ii.size, np.asarray(L[ii, jj]).astype(np.float64), np.asarray(F[ii, jj])

    def usable(self, cap_ok=True, cap=None, finite=False, block=20000):
        """
        Mask of the points to use: status ok, and (cap_ok=False) below the iteration cap.

        Parameters
        ----------
        cap_ok: bool
            Accept models that stopped at the iteration cap (default; M424: 230 649 models at niter =
            102, formally unconverged; their profiles differ by <= 2.6e-4 in EW from ITMORE = 300 reruns).
        cap: int, optional
            The iteration cap in units of niter, from the run configuration (:func:`niter_cap_from_indat`;
            M424: :data:`NITER_CAP_M424`). Default: :attr:`niter_cap` recorded in the file; with
            cap_ok=False and neither, ValueError (the cap is not inferred from max(niter), which in a run
            where no model reached the cap is a converged model).
        finite: bool
            Also require finite lam/fcont/fnorm (reads the profiles in blocks).
        block: int
            Points per block for ``finite``.

        Returns
        -------
        np.ndarray
            (n,) bool.

        Warns
        -----
        UserWarning
            If some niter exceed the cap (the cap does not belong to this run).
        """
        # PP 2026-10-01: cap explicit (reviewer: max(niter) dropped converged models of uncapped runs)
        ok = _status_ok(self.status, self.n)
        if not cap_ok:
            cap = self.niter_cap if cap is None else cap
            if cap is None:
                raise ValueError("the iteration cap is not known: pass cap= (niter_cap_from_indat(INDAT); M424: "
                                 "NITER_CAP_M424 = {}) or record niter_cap when merging".format(NITER_CAP_M424))
            if self.niter is None:
                raise KeyError("the store has no niter member")
            ni = np.asarray(self.niter)
            nabove = int((ni[ok] > cap).sum())
            if nabove:
                warnings.warn("{} usable points have niter > cap = {}: is the cap that of this run?".format(nabove, cap))
            ok &= ni < cap
        if finite:
            block = max(int(block), 1)
            for k in self.PROFILE_KEYS:
                A = self._m[k]
                for i0 in range(0, self.n, block):
                    ok[i0:i0 + block] &= np.isfinite(np.asarray(A[i0:i0 + block])).all(axis=(1, 2))
        return ok


def _store(store):
    return ProfileStore.open(store) if isinstance(store, (str, os.PathLike)) else store


def ew_per_point(store, block=20000, out=None, checks=False):
    """
    Rest-frame equivalent width of every line of every point, EW = int (1 - F/F_cont) dlambda
    (trapezoidal on the model's own wavelengths, no Doppler shift, no broadening) [Angstrom].

    Parameters
    ----------
    store: ProfileStore or str
        The profiles.
    block: int
        Points per block. Peak anonymous memory ~50 bytes per profile element (block x nline x nrow;
        float64 copies and temporaries of the trapezoid), i.e. ~25 MB per 1000 points for 3 x 161: measured
        0.44 GB for the default 20 000 (M424: 2.8 s with a warm page cache; the legacy CHUNK of 100 000
        needs 2.0 GB and 14.5 s). The file pages read through the memory maps also count in RSS (see
        :class:`ProfileStore`). The result does not depend on it.
    out: str, optional
        Also write ``idx, teff, ew, lines`` to this .npz (the legacy ``ew.npz``, byte-identical).
    checks: bool
        Also return the legacy checks: number of points with non-finite profiles and the largest
        deviation of each line's wavelength grid from that of point 0 [Angstrom].

    Returns
    -------
    ew: np.ndarray
        (n, nline) float64.
    checks: dict
        nbad, dlam_max (only with checks=True).

    Notes
    -----
    Reproduces the M424 ``ew.npz`` bit for bit (each row's sum is independent of the block size).
    """
    # PP 2026-10-01: ported from fig_fw_sphere_ew.py:53-67
    store = _store(store)
    L, F = store["lam"], store["fnorm"]
    n, nl = store.n, len(store.lines)
    ew = np.empty((n, nl))
    lam0 = np.asarray(L[0]) if n else None
    dlam_max = np.zeros(nl)
    nbad = 0
    block = max(int(block), 1)
    for i0 in range(0, n, block):
        lam = np.asarray(L[i0:i0 + block]).astype(np.float64)
        fn = np.asarray(F[i0:i0 + block]).astype(np.float64)
        if checks:
            nbad += int((~np.isfinite(fn)).any(axis=(1, 2)).sum())
            dlam_max = np.maximum(dlam_max, np.abs(lam - lam0).max(axis=(0, 2)))
        ew[i0:i0 + block] = _trapz(1.0 - fn, lam, axis=2)
    if out is not None:
        from .io import save_npz
        save_npz(out, dict(idx=store.idx, teff=store.teff, ew=ew, lines=store["lines"]))
    if checks:
        return ew, dict(nbad=nbad, dlam_max=dlam_max)
    return ew


def status_summary(store, cap=None):
    """
    Statistics of a run: model status, iterations, T_eff nudges and run times.

    Parameters
    ----------
    store: ProfileStore or str
    cap: int, optional
        Iteration cap in units of niter (:func:`niter_cap_from_indat`; M424: :data:`NITER_CAP_M424`);
        default the store's recorded :attr:`ProfileStore.niter_cap`. Without one, nothing is reported
        about the cap (it is not inferred from max(niter)).

    Returns
    -------
    dict
        n; complete (idx == 0..n-1); status (name -> count); niter (successful models, status ok): n,
        min, median, mean, max, n_at_max (models with niter == max), hist (niter -> count), and with a
        cap: cap, n_at_cap, n_above_cap (> 0 means the cap is not this run's); teff_nudge: n, max, idx
        (nudged points); t_pnlte (status ok): median, mean, max, and with a cap median_at_cap,
        median_below_cap; t_formal: median (status ok); cpu_hours (sum of pnlte and formal times of all
        models, float64).

    Notes
    -----
    All iteration and timing statistics use the same models (status ok); failed models' iteration
    counts are in ``store.niter``. M424 dump 3200: 1 236 544 points, all ok; 230 649 at the cap
    niter = 102; 2 nudged by +1 K.
    """
    # PP 2026-10-01: ported from the printed statistics of fig_fw_sphere_ew.py:36-47 and fw_sphere_merge.py:151-157;
    # cap explicit, one status filter (reviewer)
    store = _store(store)
    n = store.n
    idx = np.asarray(store.idx)
    st = np.asarray(store.status) if store.status is not None else np.full(n, "ok")
    s, c = np.unique(st, return_counts=True)
    out = dict(n=n, complete=bool(idx.size == n and np.array_equal(idx, np.arange(n))),
               status={str(a): int(b) for a, b in zip(s, c)})
    ok = _status_ok(store.status, n)
    cap = store.niter_cap if cap is None else cap
    cap = None if cap is None else int(cap)
    if store.niter is not None and n:
        ni = np.asarray(store.niter)[ok]
        d = dict(n=int(ni.size))
        if ni.size:
            u, cu = np.unique(ni, return_counts=True)
            d.update(min=int(ni.min()), median=float(np.median(ni)), mean=float(ni.mean(dtype=np.float64)),
                     max=int(ni.max()), n_at_max=int((ni == ni.max()).sum()), hist={int(a): int(b) for a, b in zip(u, cu)})
        if cap is not None:
            d.update(cap=cap, n_at_cap=int((ni == cap).sum()), n_above_cap=int((ni > cap).sum()))
        out["niter"] = d
    if store.teff_nudge is not None:
        nud = np.asarray(store.teff_nudge)
        on = nud != 0
        out["teff_nudge"] = dict(n=int(on.sum()), max=float(nud.max()) if n else 0.0, idx=idx[on].tolist())
    if "t_pnlte" in store and n:
        tp = np.asarray(store["t_pnlte"])
        d = dict(median=float(np.median(tp[ok])) if ok.any() else np.nan,
                 mean=float(tp[ok].mean(dtype=np.float64)) if ok.any() else np.nan,
                 max=float(tp[ok].max()) if ok.any() else np.nan)
        if "niter" in out and cap is not None:
            atc = ok & (np.asarray(store.niter) == cap)
            blc = ok & (np.asarray(store.niter) < cap)
            d["median_at_cap"] = float(np.median(tp[atc])) if atc.any() else np.nan
            d["median_below_cap"] = float(np.median(tp[blc])) if blc.any() else np.nan
        out["t_pnlte"] = d
        cpu = tp.sum(dtype=np.float64)
        if "t_formal" in store:
            tf = np.asarray(store["t_formal"])
            out["t_formal"] = dict(median=float(np.median(tf[ok])) if ok.any() else np.nan)
            cpu += tf.sum(dtype=np.float64)
        out["cpu_hours"] = float(cpu / 3600.0)
    return out


# ----------------------------------------------------------------------------------------------
# the premise of the T_eff' library: every model's INDAT.DAT differs only in MODNAM and TEFF
# ----------------------------------------------------------------------------------------------
INDAT_SCHEMA = (
    ("MODNAM",),
    ("OPTNEUPDATE", "HE_ONE", "ITSTART", "ITMORE"),
    ("OPTMIXED",),
    ("TEFF", "LOGG", "RSTAR"),
    ("RMAX", "TMIN"),
    ("MDOT", "VMIN", "VINF", "BETA", "VDIV"),
    ("YHE", "IHE"),
    ("OPTMOD", "OPTTLUCY", "MEGAS", "ACCEL", "OPTCMF"),
    ("VTURB", "METALLICITY", "LINES", "LINES_IN_MODEL"),
    ("ENATCOR", "EXPANSION", "SET_FIRST", "SET_STEP"),
)
"""Field names of the fixed first ten lines of a FASTWIND v10 INDAT.DAT, one tuple per line: the list-directed
READ statements of nlte.f90 (v10.6.4.1, lines 1511-1520), named as in the comments of the M424 template
(INDAT_M424test.DAT; 'LOG G' -> LOGG, 'VMIN(START)' -> VMIN, 'IHE(START)' -> IHE, METALLICITY = XMET). The lines that
follow (clumping, optional Hopf parameters, abundances, X-rays) have no fixed layout and are compared whole, as
fields 'LINE<n>' (see :func:`check_indat_premise`)."""

INDAT_SERIAL_MAX = 50
"""Parts that :func:`check_indat_premise` with nproc=1 reads in the calling process; more go through one recycled
worker process (CPU-time limit per process of the login nodes)."""

_INDAT_SPLIT = re.compile(r"[,\s]+")
_INDAT_LOGICAL = re.compile(r"\.?(T|F|TRUE|FALSE)\.?")
_INDAT_STRING_FIELDS = ("MODNAM",)


def _indat_value(tok):
    """A list-directed INDAT token: float (Fortran D exponents too), bool (T, F, .TRUE., ...) or the string itself."""
    try:
        return float(tok.replace("D", "E").replace("d", "e"))
    except ValueError:
        pass
    if _INDAT_LOGICAL.fullmatch(tok.upper()):
        return tok.upper().strip(".").startswith("T")
    return tok


def _indat_fields(text, schema=INDAT_SCHEMA):
    """
    name -> (value, raw text) of an INDAT.DAT: the first len(names) tokens of each schema line (the rest of the line
    is FASTWIND's comment and ignored; a missing token or line gives (None, None)); every further non-blank line as
    'LINE<n>' (1-based) with the tuple of all its token values, comment words included. MODNAM stays a string.
    """
    if isinstance(text, (bytes, bytearray)):
        text = text.decode("latin-1")
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    out = {}
    for i, names in enumerate(schema):
        tok = [t for t in _INDAT_SPLIT.split(lines[i].strip()) if t] if i < len(lines) else []
        for k, nm in enumerate(names):
            if k >= len(tok):
                out[nm] = (None, None)
            else:
                out[nm] = (tok[k] if nm in _INDAT_STRING_FIELDS else _indat_value(tok[k]), tok[k])
    for i in range(len(schema), len(lines)):
        tok = [t for t in _INDAT_SPLIT.split(lines[i].strip()) if t]
        out["LINE{}".format(i + 1)] = (tuple(_indat_value(t) for t in tok), lines[i].strip())
    return out


def _indat_same(a, b):
    """Equal INDAT values: same type and value (1. == 1.0, but 1.0 != T); tuples element by element; NaN == NaN."""
    if isinstance(a, tuple) or isinstance(b, tuple):
        return (isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b)
                and all(_indat_same(x, y) for x, y in zip(a, b)))
    if type(a) is not type(b):
        return False
    if isinstance(a, float) and a != a and b != b:
        return True
    return a == b


def _indat_diff(ref, fields):
    """Names of the fields whose values differ between two _indat_fields records (missing = (None, None))."""
    names = list(ref) + [k for k in fields if k not in ref]
    miss = (None, None)
    return [k for k in names if not _indat_same(ref.get(k, miss)[0], fields.get(k, miss)[0])]


def _indat_parts(results_dir, tag):
    """The parts of one tag, sorted; results_dir and tag taken literally (glob metacharacters such as [ ] escaped,
    unlike :func:`_parts_of`)."""
    # PP 2026-10-01: reviewer: a results_dir with [ ] found no parts through _parts_of (left unchanged)
    return sorted(glob.glob(os.path.join(glob.escape(results_dir), glob.escape(tag), "part_*.tar.gz")))


def _indat_tags(results_dir, tags):
    """Tag directories: None = every subdirectory with parts (sorted); a str = glob pattern; else the given tags."""
    if tags is None:
        tags = "*"
    if isinstance(tags, str):
        return sorted(os.path.basename(d) for d in glob.glob(os.path.join(glob.escape(results_dir), tags))
                      if os.path.isdir(d) and _indat_parts(results_dir, os.path.basename(d)))
    return [str(t) for t in tags]


def _indat_part(args):
    """Compare every INDAT.DAT of one part with the reference record (a check_indat_premise task)."""
    part, tag, ref, allowed, schema, max_report = args
    allowed = set(allowed)
    r = dict(part=part, tag=tag, n_points=0, n_no_indat=0, n_offending=0, differ={}, offending=[], status={},
             modnam_mismatch=0, teff_meta_mismatch=0, teff_meta_maxdiff=0.0, teff_min=np.inf, teff_max=-np.inf,
             idx=[])
    for pdir, files in iter_part_points(part, want=("meta.txt", "INDAT.DAT")):
        m = parse_meta(files["meta.txt"])
        r["status"][m["status"]] = r["status"].get(m["status"], 0) + 1
        if "INDAT.DAT" not in files:
            r["n_no_indat"] += 1
            continue
        f = _indat_fields(files["INDAT.DAT"], schema)
        r["n_points"] += 1
        r["idx"].append(m["idx"])
        diff = _indat_diff(ref, f)
        for k in diff:
            r["differ"][k] = r["differ"].get(k, 0) + 1
        bad = [k for k in diff if k not in allowed]
        if bad:
            r["n_offending"] += 1
            if len(r["offending"]) < max_report:
                r["offending"].append(dict(idx=m["idx"], pdir=pdir, part=part, tag=tag,
                                           fields={k: [ref.get(k, (None, None))[1], f.get(k, (None, None))[1]]
                                                   for k in bad}))
        # PP 2026-10-01: reviewer: a missing MODNAM and a missing, non-numeric or non-finite TEFF count as
        # mismatches (they were skipped)
        modnam = f.get("MODNAM", (None, None))[0]
        if modnam != pdir:
            r["modnam_mismatch"] += 1
        teff = f.get("TEFF", (None, None))[0]
        if not (isinstance(teff, float) and np.isfinite(teff)):
            r["teff_meta_mismatch"] += 1
            continue
        r["teff_min"], r["teff_max"] = min(r["teff_min"], teff), max(r["teff_max"], teff)
        d = abs(teff - m["teff"])
        if np.isfinite(d):
            r["teff_meta_maxdiff"] = max(r["teff_meta_maxdiff"], d)
        # INDAT's TEFF is the '%.3f' rounding of meta.txt's T_eff (printed verbatim, awk '%s'), and round(x, 3) is
        # the same correctly rounded decimal, so the two agree to float parsing
        if not abs(teff - round(m["teff"], 3)) <= 1e-6:
            r["teff_meta_mismatch"] += 1
    return r


def check_indat_premise(results_dir, tags=None, max_parts=None, template=None, fields_allowed=("MODNAM", "TEFF"),
                        schema=INDAT_SCHEMA, max_report=10, nproc=1, start_method=None, timeout=3600.0,
                        require_indat=True, log=None):
    """
    Verify the premise of the T_eff' library on the archived models: every model's INDAT.DAT differs from a
    reference only in the allowed fields (default MODNAM and TEFF), i.e. the per-point models differ only in T_eff'.

    The packed parts are streamed (:func:`iter_part_points` with ``want=('meta.txt', 'INDAT.DAT')``; nothing is
    extracted to disk) and each INDAT.DAT is compared field by field with the reference.

    Parameters
    ----------
    results_dir: str
        ``RUN_DIR/results`` (tag directories with ``part_*.tar.gz``).
    tags: None, str or sequence of str
        Tag directories to read: None = every subdirectory that holds parts (sorted); a str is a glob pattern
        (``'task_00[0-3]?'``; a plain tag name matches itself); a sequence lists the tags (a tag without parts reads
        nothing and is listed in ``empty_tags``).
    max_parts: int, optional
        At most this many parts per tag (the first ones in sorted order; default all). M424 parts hold ~800 points
        each, 2.5-3.4 GB packed with the model files, ~20-40 s each to stream.
    template: str, optional
        The reference INDAT.DAT: a file name, or its text (anything containing a newline; e.g. the template the run
        was made from, M424 project/analysis/fastwind/INDAT_M424test.DAT). Default: the first INDAT.DAT found (the
        first point of the first part read).
    fields_allowed: sequence of str
        Fields that may differ (:data:`INDAT_SCHEMA` names, or 'LINE<n>' for line n > 10).
    schema: sequence of tuple of str
        Field names of the fixed leading lines (default :data:`INDAT_SCHEMA`, FASTWIND v10). Tokens beyond a line's
        names are FASTWIND's comment and ignored; every further line is compared whole (its tokens, comment
        words included: a changed comment there is reported, which errs on the safe side).
    max_report: int
        Offending points reported in detail (the first ones in part order).
    nproc: int
        Worker processes, one part per task (:func:`ppmpy.synspec.parallel.make_pool` with maxtasksperchild=8;
        'fork' or 'spawn' via ``start_method``). The result does not depend on nproc. nproc=1 reads the parts in
        this process only up to :data:`INDAT_SERIAL_MAX` (50) parts; more go through one worker process that is
        replaced every 8 parts: an M424 part takes 20-40 s of CPU, so one process streaming ~1500 parts would pass
        the 3600 s CPU-time limit per process of the Trillium login nodes (``ulimit -t``) after ~100-150 parts and be
        killed.
    timeout: float or None
        Watchdog of the pool [s] per part (:func:`ppmpy.synspec.parallel.imap_watchdog`).
    require_indat: bool
        A point with meta.txt but no INDAT.DAT fails the check (default; every point of fw_sphere_point.sh gets its
        INDAT.DAT, failed ones included). False: such points are only counted (``n_no_indat``).
    log: callable, optional
        One line per part read.

    Returns
    -------
    dict
        passed: at least one model read, none offending, no consistency mismatch (MODNAM, TEFF) and, with
        ``require_indat``, no point without INDAT.DAT; complete (no point without INDAT.DAT); results_dir; tags;
        empty_tags; parts (read, in order); reference (source 'template' or 'first', point and part for 'first',
        fields: name -> text); fields_allowed; require_indat; n_points (models with an INDAT.DAT), n_unique
        (distinct idx), n_no_indat (points with meta.txt but no INDAT.DAT), n_offending (models differing in a
        field not allowed), n_premise_ok; differ (field -> number of models where it differs from the reference,
        allowed fields included); offending (up to max_report: idx, pdir, part, tag, fields: name -> [reference
        text, model text]); status (meta.txt status -> count); consistency: modnam_mismatch (MODNAM missing or not
        the point's directory name: the INDAT of another point), teff_meta_mismatch (TEFF missing, not a finite
        number, or not the '%.3f' rounding of meta.txt's T_eff, to 1e-6 K: the model would sit in the library under
        another T_eff' than its own, since :func:`merge_task` takes the label from meta.txt) and teff_meta_maxdiff
        (largest |TEFF - meta.txt T_eff| [K], <= 5e-4 from the rounding), teff_range (min, max of TEFF) [K];
        wall [s].

    Notes
    -----
    Values are compared as FASTWIND reads them (list-directed): numbers as floats (``1.`` equals ``1.0``, Fortran
    ``D`` exponents allowed), logicals (T, F, .TRUE., ...) as booleans, anything else as text; MODNAM as text.
    Note that :func:`iter_part_points` drops a point directory without meta.txt, so ``want`` must include it (with
    ``want=('INDAT.DAT',)`` alone nothing would be yielded).

    Validation: synthetic parts in tests/synspec/test_testing.py (a changed LOGG or VINF, an extra line, a missing
    INDAT.DAT, a MODNAM other than the directory or missing, a TEFF other than meta.txt's, not a number or NaN, each
    of these alone failing the verdict; meta.txt T_eff with more decimals and exact '%.3f' ties; equal values
    written differently; glob metacharacters in the paths; serial = 2 workers = one recycled worker). M424 (marker
    m424, slow; 2026-10-01): the first part of task_0000, task_0039 and
    task_missing_0000 (1503 models, T_eff 36 669-38 861 K): every INDAT.DAT differs from the template
    (INDAT_M424test.DAT) in MODNAM and TEFF only, MODNAM is the directory name and TEFF the T_eff of meta.txt; 16 s
    with 3 workers, 24 s serially (warm page cache), 64 MB.
    """
    # PP 2026-10-01: new (task: verify that the archived INDATs differ only in MODNAM and TEFF, the premise of the
    # T_eff' library of the all-dump method; fw_sphere_point.sh writes INDAT.DAT from the template by replacing line 1
    # and the first value of line 4)
    import time
    T0 = time.time()
    results_dir = os.fspath(results_dir)
    tag_list = _indat_tags(results_dir, tags)
    parts, empty = [], []
    for t in tag_list:
        p = _indat_parts(results_dir, t)
        if max_parts is not None:
            p = p[:max(int(max_parts), 0)]
        if not p:
            empty.append(t)
        parts += [(x, t) for x in p]
    max_report = max(int(max_report), 0)
    if template is not None:
        template = os.fspath(template)
        if "\n" in template:
            text, path = template, None
        else:
            path = os.path.abspath(template)
            with open(path, "rb") as fh:
                text = fh.read()
        ref = _indat_fields(text, schema)
        reference = dict(source="template", path=path)
    else:
        ref, reference = None, dict(source="first")
        for part, _ in parts:
            gen = iter_part_points(part, want=("meta.txt", "INDAT.DAT"))
            try:
                for pdir, files in gen:
                    if "INDAT.DAT" in files:
                        ref = _indat_fields(files["INDAT.DAT"], schema)
                        reference.update(part=part, pdir=pdir, idx=parse_meta(files["meta.txt"])["idx"])
                        break
            finally:
                gen.close()                               # stops the stream (and closes the archive) at once
            if ref is not None:
                break
    allowed = tuple(str(k) for k in fields_allowed)
    out = dict(passed=False, complete=False, results_dir=results_dir, tags=tag_list, empty_tags=empty,
               parts=[p for p, _ in parts], reference=reference, fields_allowed=list(allowed),
               require_indat=bool(require_indat), n_points=0, n_unique=0, n_no_indat=0, n_offending=0,
               n_premise_ok=0, differ={}, offending=[], status={},
               consistency=dict(modnam_mismatch=0, teff_meta_mismatch=0, teff_meta_maxdiff=0.0, teff_range=None),
               wall=0.0)
    if ref is None:
        out["reference"]["fields"] = None
        out["wall"] = time.time() - T0
        return out
    reference["fields"] = {k: v[1] for k, v in ref.items()}
    tasks = [(p, t, ref, allowed, tuple(tuple(x) for x in schema), max_report) for p, t in parts]
    res = [None] * len(tasks)

    def _done(i, r):
        res[i] = r
        if log is not None:
            log("{} {}: {} models, {} offending{}".format(r["tag"], os.path.basename(r["part"]), r["n_points"],
                                                         r["n_offending"], "; differ: {}".format(r["differ"])
                                                         if r["differ"] else ""))

    nproc = max(1, min(int(nproc), len(tasks)))
    if nproc <= 1 and len(tasks) <= INDAT_SERIAL_MAX:
        for i, a in enumerate(tasks):
            _done(i, _indat_part(a))
    else:                                                 # PP 2026-10-01: reviewer: nproc=1 with many parts too
        from . import parallel as par
        par.login_node_warning(nproc)
        with par.make_pool(nproc, maxtasksperchild=8, start_method=start_method) as pool:
            for i, r in par.imap_watchdog(pool, _indat_part_indexed, list(enumerate(tasks)), timeout=timeout):
                _done(i, r)
    idx, tmin, tmax = [], np.inf, -np.inf
    names = list(ref)
    for r in res:
        for k in ("n_points", "n_no_indat", "n_offending"):
            out[k] += r[k]
        for k, c in r["differ"].items():
            out["differ"][k] = out["differ"].get(k, 0) + c
            if k not in names:
                names.append(k)
        for k, c in r["status"].items():
            out["status"][k] = out["status"].get(k, 0) + c
        if len(out["offending"]) < max_report:
            out["offending"] += r["offending"][:max_report - len(out["offending"])]
        c = out["consistency"]
        c["modnam_mismatch"] += r["modnam_mismatch"]
        c["teff_meta_mismatch"] += r["teff_meta_mismatch"]
        c["teff_meta_maxdiff"] = max(c["teff_meta_maxdiff"], r["teff_meta_maxdiff"])
        tmin, tmax = min(tmin, r["teff_min"]), max(tmax, r["teff_max"])
        idx += r["idx"]
    out["differ"] = {k: out["differ"][k] for k in names if k in out["differ"]}
    out["n_unique"] = int(np.unique(np.asarray(idx, dtype=np.int64)).size)
    out["n_premise_ok"] = out["n_points"] - out["n_offending"]
    if np.isfinite(tmin):
        out["consistency"]["teff_range"] = [float(tmin), float(tmax)]
    c = out["consistency"]
    out["complete"] = bool(out["n_no_indat"] == 0)
    # PP 2026-10-01: reviewer: the verdict ignored the consistency counts and the missing INDATs
    out["passed"] = bool(out["n_points"] > 0 and out["n_offending"] == 0 and c["teff_meta_mismatch"] == 0
                         and c["modnam_mismatch"] == 0 and (out["complete"] or not require_indat))
    out["wall"] = time.time() - T0
    return out


def _indat_part_indexed(item):
    i, a = item
    return i, _indat_part(a)
