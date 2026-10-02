"""
Packed per-point results of a FASTWIND sphere run: the packer (finished result directories -> one part
``part_*.tar.gz`` plus its ledger ``part_*.idx``), ledger readers, and crash recovery (standard library only).

Layout (fw_sphere_task.sh; read unchanged by ``ppmpy.synspec.fwresults``: merge_task, read_ledger, locate_points,
extract_points, check_indat_premise)::

    <results>/<tag>/part_<name>.tar.gz     the point directories ./P<idx>/... (GNU tar format, gzip)
    <results>/<tag>/part_<name>.idx        the ledger: the meta.txt lines of the part's points

A part is written as ``part_<name>.tar.gz.tmp`` and renamed when complete; its ledger is written afterwards (as
``part_<name>.idx.tmp``, renamed), so a ledger line always means "this point is in a complete part". The union of
the ledgers ``<tag>/*.idx`` is the done list of a restarted task (:func:`done_set`, the legacy ``cat $OUT/*.idx``).

Part names: ``part_<UTC YYYYmmdd_HHMMSS>Z_<host>_<pid>_<seq>`` (:func:`part_name`; pid of the runner, seq its pack
counter), so the names of one tag sort in time order, also next to the legacy ``part_<local date>_<RANDOM>``.

Archive members, as ``tar -czf part -C STAGE .`` of the legacy packer: ``./`` then per point ``./P<idx>`` and its
files; here the points and their files are in sorted order (GNU tar used directory order), so a point's files are
contiguous (fwresults.iter_part_points) and the ledger lines are in the same order as the archive (the legacy
``cat $STAGE/*/meta.txt``, also sorted).

Crash safety of :func:`pack`: results are moved from the result directory to the stage, the name of the part being
written is recorded in the hidden marker ``<stage>/.pack``, the archive and then the ledger are written through
temporary names, and the stage is emptied. A pack interrupted at any point is completed by the next :func:`pack` with
the same stage (from the marker: a complete archive gets its ledger, an incomplete one is removed and its points are
packed again), so no point is lost or packed twice. Across runner restarts (a new stage), :func:`recover` repairs the
tag directory: an archive without ledger ("orphan", e.g. the packer was killed between the two renames) gets its
ledger rebuilt from the archive; a corrupt orphan is renamed ``*.corrupt``; a ledger without archive is renamed
``*.orphan`` (its points would otherwise count as done); stale temporaries are removed.

PP 2026-10-02: new (M6); ported from fw_sphere_task.sh:54-63 (pack()) and :46-48 (done ledger) of the project
stellar-atmosphere-KU-Leuven.
"""
import argparse
import collections
import glob
import gzip
import itertools
import json
import os
import re
import shutil
import signal
import socket
import sys
import tarfile
import time
import zlib

LEDGER_SUFFIX = ".idx"
PART_SUFFIX = ".tar.gz"
TMP_SUFFIX = ".tmp"
CORRUPT_SUFFIX = ".corrupt"
ORPHAN_SUFFIX = ".orphan"
MARKER = ".pack"
"""Hidden file in the stage naming the part being written (crash recovery of :func:`pack`)."""
META_FILE = "meta.txt"
COMPRESSLEVEL = 6
"""gzip level of the parts (GNU tar -z runs gzip with its default, 6)."""

CHILD_IGNORED = tuple(n for n in ("SIGUSR1", "SIGTERM", "SIGINT", "SIGHUP") if hasattr(signal, n))
"""Signals the runner's packer child ignores (``pack --ignore-signals``; the runner blocks them around the fork)."""
CORRUPT_ERRORS = (tarfile.TarError, EOFError, zlib.error, gzip.BadGzipFile)
"""Errors of reading an archive that mean it is corrupt or truncated (other errors, e.g. EIO or EACCES on scratch,
are not taken for corruption: :func:`recover` raises them)."""

_PART_RE = re.compile(r"^part_(?P<when>\d{8}_\d{6}Z?)_(?:(?P<host>[A-Za-z0-9.-]+)_(?P<pid>\d+)_(?P<seq>\d+)|\d+)")
_seq = itertools.count(1)

MetaRecord = collections.namedtuple("MetaRecord", "idx teff status niter T_tau23 t_pnlte t_formal")
MetaRecord.__doc__ = """One meta.txt / ledger line: idx (int), teff (str, as written), status (str), niter (int),
T_tau23 (str, as written, 'nan' when pnlte printed none), t_pnlte, t_formal (float, s)."""


# ----------------------------------------------------------------------------------------------------------------
# names, ledgers
# ----------------------------------------------------------------------------------------------------------------
def short_host():
    """The short host name with characters other than letters, digits, '.' and '-' replaced by '-'."""
    h = socket.gethostname().split(".")[0] or "host"
    return re.sub(r"[^A-Za-z0-9.-]", "-", h)


def part_name(when=None, host=None, pid=None, seq=None):
    """
    A part name ``part_<UTC YYYYmmdd_HHMMSS>Z_<host>_<pid>_<seq:04d>`` (no suffix).

    Parameters
    ----------
    when: float, optional
        Epoch seconds (default now).
    host: str, optional
        Default :func:`short_host`.
    pid: int, optional
        Default this process's pid (the runner passes its own pid to its packer children).
    seq: int, optional
        Default a per-process counter.
    """
    # PP 2026-10-02: replaces part_$(date +%Y%m%d_%H%M%S)_$RANDOM (fw_sphere_task.sh:58), which could repeat
    t = time.gmtime(time.time() if when is None else when)
    return "part_{}Z_{}_{}_{:04d}".format(time.strftime("%Y%m%d_%H%M%S", t), host or short_host(),
                                         os.getpid() if pid is None else int(pid), next(_seq) if seq is None else seq)


def parse_part_name(name):
    """dict(when, host, pid, seq) of a part (or temporary) file name; host / pid / seq None for legacy names; None
    for a name that is not a part name."""
    m = _PART_RE.match(os.path.basename(name))
    if not m:
        return None
    d = m.groupdict()
    return dict(when=d["when"], host=d["host"], pid=int(d["pid"]) if d["pid"] else None,
                seq=int(d["seq"]) if d["seq"] else None)


def parse_meta_line(line):
    """
    A :class:`MetaRecord` of one meta.txt / ledger line (``"%d %s %s %d %s %.1f %.1f"``, fw_sphere_point.sh).

    Raises
    ------
    ValueError
        Not seven fields, or idx / niter not integers, or the times not numbers.
    """
    # PP 2026-10-02: ported from fw_sphere_point.sh:52-53 (the meta.txt format); stdlib twin of fwresults.parse_meta
    if isinstance(line, (bytes, bytearray)):
        line = bytes(line).decode()
    f = line.split()
    if len(f) != 7:
        raise ValueError("meta line needs 7 fields, got {}: {!r}".format(len(f), line.rstrip("\n")))
    return MetaRecord(int(f[0]), f[1], f[2], int(f[3]), f[4], float(f[5]), float(f[6]))


def read_index(path):
    """
    The records of one ledger (``part_*.idx``), in file order (blank lines skipped).

    Returns
    -------
    list of MetaRecord

    Raises
    ------
    ValueError
        A malformed line (file and line number in the message).
    """
    # PP 2026-10-02: new; the ledger is written by fw_sphere_task.sh:60 (cat $STAGE/*/meta.txt)
    out = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(parse_meta_line(line))
            except ValueError as e:
                raise ValueError("{}:{}: {}".format(path, n, e)) from None
    return out


def ledger_paths(out_dir):
    """The ledgers of a tag directory: ``<out_dir>/*.idx`` (the legacy ``cat $OUT/*.idx``; no dotfiles), sorted."""
    return sorted(glob.glob(os.path.join(glob.escape(out_dir), "*" + LEDGER_SUFFIX)))


def done_set(out_dirs):
    """
    The point indices listed in the ledgers of one or several tag directories (the done list of fw_sphere_task.sh:46,
    any status). Only the first field of each line is read (as the legacy awk filter).

    Raises
    ------
    ValueError
        A ledger line whose first field is not an integer.
    """
    # PP 2026-10-02: ported from fw_sphere_task.sh:46-48 (cat $OUT/*.idx; awk filter on $1)
    if isinstance(out_dirs, (str, os.PathLike)):
        out_dirs = [out_dirs]
    done = set()
    for d in out_dirs:
        for p in ledger_paths(os.fspath(d)):
            with open(p) as f:
                for n, line in enumerate(f, 1):
                    tok = line.split(None, 1)
                    if not tok:
                        continue
                    try:
                        done.add(int(tok[0]))
                    except ValueError:
                        raise ValueError("{}:{}: not a meta.txt line: {!r}".format(p, n, line.rstrip())) from None
    return done


# ----------------------------------------------------------------------------------------------------------------
# pack
# ----------------------------------------------------------------------------------------------------------------
def _is_tmp_name(name):
    return name.startswith(".") or name.endswith(TMP_SUFFIX)


def finished_results(res_dir):
    """
    Names of the finished result directories in ``res_dir``: directories, not hidden (a result being replaced is
    renamed to a hidden name), not ``*.tmp`` (being written) - the legacy ``ls $RES | grep -v '\\.tmp$'``. Sorted.
    """
    try:
        names = os.listdir(res_dir)
    except FileNotFoundError:
        return []
    return sorted(n for n in names if not _is_tmp_name(n) and os.path.isdir(os.path.join(res_dir, n))
                  and not os.path.islink(os.path.join(res_dir, n)))


def _stage_points(stage_dir):
    return finished_results(stage_dir)


def _fsync_path(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_tar(path, stage_dir, points, compresslevel=COMPRESSLEVEL, fsync=True):
    """Write the archive of ``points`` (directories in ``stage_dir``) to ``path`` (must not exist)."""
    with open(path, "xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=compresslevel, fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as tf:
                tf.add(stage_dir, arcname=".", recursive=False)
                for p in points:
                    tf.add(os.path.join(stage_dir, p), arcname="./" + p, recursive=True)
        raw.flush()
        if fsync:
            os.fsync(raw.fileno())


def _ledger_bytes_from_stage(stage_dir, points):
    out = []
    for p in points:
        try:
            with open(os.path.join(stage_dir, p, META_FILE), "rb") as f:
                out.append(f.read())
        except FileNotFoundError:
            pass                                   # cat $STAGE/*/meta.txt skips a directory without one
    return b"".join(out)


def ledger_bytes_from_part(part):
    """
    The ledger of an archive as :func:`pack` writes it: the ``meta.txt`` of every point directory, ordered by the
    directory name (the legacy ``cat $STAGE/*/meta.txt``). Reads the whole archive.

    Raises
    ------
    tarfile.TarError, EOFError, zlib.error, gzip.BadGzipFile
        A truncated or corrupt archive (:data:`CORRUPT_ERRORS`).
    OSError
        The file cannot be read.
    """
    # PP 2026-10-02: new; the inverse of fw_sphere_task.sh:59-60 for an archive whose ledger is missing.
    # tarfile's stream mode takes a truncated gzip for the end of the archive (no error), so a gzip archive is read
    # through GzipFile and drained to its end, which checks the end-of-stream marker, CRC and length.
    metas = {}

    def collect(tf):
        for m in tf:
            if not m.isfile():
                continue
            name = m.name
            while name.startswith("./"):
                name = name[2:]
            parts = name.split("/")
            if len(parts) == 2 and parts[1] == META_FILE:
                metas[parts[0]] = tf.extractfile(m).read()

    with open(part, "rb") as raw:
        gzipped = raw.read(2) == b"\x1f\x8b"
        raw.seek(0)
        if gzipped:
            with gzip.GzipFile(fileobj=raw, mode="rb") as gz:
                with tarfile.open(fileobj=gz, mode="r|") as tf:
                    collect(tf)
                while gz.read(1 << 20):
                    pass
        else:
            with tarfile.open(fileobj=raw, mode="r|*") as tf:
                collect(tf)
    return b"".join(metas[k] for k in sorted(metas))


def _write_ledger(out_dir, part, data, fsync=True, unique=False):
    final = os.path.join(out_dir, part + LEDGER_SUFFIX)
    tmp = final + TMP_SUFFIX + (".{}".format(os.getpid()) if unique else "")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        if fsync:
            os.fsync(f.fileno())
    os.replace(tmp, final)
    return final


def _clear_stage(stage_dir, points):
    for p in points:
        shutil.rmtree(os.path.join(stage_dir, p), ignore_errors=True)


def _read_marker(stage_dir):
    try:
        with open(os.path.join(stage_dir, MARKER)) as f:
            return f.read().strip() or None
    except FileNotFoundError:
        return None


def _write_marker(stage_dir, part):
    tmp = os.path.join(stage_dir, MARKER + TMP_SUFFIX)
    with open(tmp, "w") as f:
        f.write(part + "\n")
    os.replace(tmp, os.path.join(stage_dir, MARKER))


def _remove_marker(stage_dir):
    try:
        os.unlink(os.path.join(stage_dir, MARKER))
    except FileNotFoundError:
        pass


def _finish_interrupted(stage_dir, out_dir, fsync=True, log=None):
    """Complete a pack interrupted with this stage (see the module notes). Returns a description or None."""
    prev = _read_marker(stage_dir)
    if prev is None:
        return None
    tar = os.path.join(out_dir, prev + PART_SUFFIX)
    points = _stage_points(stage_dir)
    if os.path.isfile(tar):
        led = os.path.join(out_dir, prev + LEDGER_SUFFIX)
        if not os.path.isfile(led):
            _write_ledger(out_dir, prev, ledger_bytes_from_part(tar), fsync=fsync, unique=True)
        _clear_stage(stage_dir, points)
        what = "completed"
    else:
        for t in (tar + TMP_SUFFIX, os.path.join(out_dir, prev + LEDGER_SUFFIX + TMP_SUFFIX)):
            if os.path.lexists(t):
                os.unlink(t)
        what = "discarded"                     # its points stay in the stage and are packed now
    _remove_marker(stage_dir)
    if log is not None:
        log("interrupted pack {} {} ({} points in the stage)".format(prev, what, len(points)))
    return dict(part=prev, action=what, npoint=len(points))


def pack(res_dir, stage_dir, out_dir, part=None, compresslevel=COMPRESSLEVEL, fsync=True, log=None):
    """
    Pack the finished results of ``res_dir`` into one part in ``out_dir`` (fw_sphere_task.sh's pack()).

    Parameters
    ----------
    res_dir: str
        Node-local result directory (finished points appear there as ``P<idx>/`` by atomic renames).
    stage_dir: str
        Node-local staging directory on the same file system (results are moved there, then archived). Points left
        there by an interrupted pack are packed too.
    out_dir: str
        The tag directory (``RUN_DIR/results/<tag>``); created.
    part: str, optional
        Part name without suffix (default :func:`part_name`).
    compresslevel: int
        gzip level (6, as GNU tar -z).
    fsync: bool
        fsync the archive and the ledger before their renames.
    log: callable, optional
        Receives one summary line.

    Returns
    -------
    dict
        part (None when nothing was packed), npoint, points (directory names), archive, ledger, bytes, seconds,
        resumed (what was done about an interrupted pack, or None).
    """
    # PP 2026-10-02: ported from fw_sphere_task.sh:54-63
    t0 = time.monotonic()
    os.makedirs(stage_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    resumed = _finish_interrupted(stage_dir, out_dir, fsync=fsync, log=log)
    staged = set(_stage_points(stage_dir))
    for name in finished_results(res_dir):
        if name in staged:                     # a result of the same name waits in the stage: next pack
            continue
        try:
            os.rename(os.path.join(res_dir, name), os.path.join(stage_dir, name))
        except FileNotFoundError:
            continue
    points = _stage_points(stage_dir)
    out = dict(part=None, npoint=0, points=[], archive=None, ledger=None, bytes=0, seconds=0.0, resumed=resumed)
    if not points:
        out["seconds"] = time.monotonic() - t0
        return out
    part = part or part_name()
    if os.path.lexists(os.path.join(out_dir, part + PART_SUFFIX)):
        raise FileExistsError("part {} exists in {}".format(part, out_dir))
    _write_marker(stage_dir, part)
    tar = os.path.join(out_dir, part + PART_SUFFIX)
    tmp = tar + TMP_SUFFIX
    if os.path.lexists(tmp):
        os.unlink(tmp)
    try:
        _write_tar(tmp, stage_dir, points, compresslevel=compresslevel, fsync=fsync)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        _remove_marker(stage_dir)              # nothing archived: the stage is packed again next time
        raise
    os.replace(tmp, tar)
    led = _write_ledger(out_dir, part, _ledger_bytes_from_stage(stage_dir, points), fsync=fsync)
    _clear_stage(stage_dir, points)
    _remove_marker(stage_dir)
    out.update(part=part, npoint=len(points), points=points, archive=tar, ledger=led, bytes=os.path.getsize(tar),
               seconds=time.monotonic() - t0)
    if log is not None:
        log("packed {} points -> {} ({:.1f} MB, {:.1f} s)".format(len(points), part, out["bytes"] / 1e6,
                                                                  out["seconds"]))
    return out


# ----------------------------------------------------------------------------------------------------------------
# scan, recover
# ----------------------------------------------------------------------------------------------------------------
def scan(out_dir):
    """
    The state of a tag directory.

    Returns
    -------
    dict
        parts (archive names without suffix), ledgers (ledger file names), orphan_parts (archives without ledger),
        orphan_ledgers (part ledgers without archive), tmp (temporary files of the packer), corrupt, orphaned
        (files renamed by :func:`recover`), runner_records (runner_*.json). Lists of names, sorted.
    """
    # PP 2026-10-02: new
    try:
        names = sorted(os.listdir(out_dir))
    except FileNotFoundError:
        names = []
    parts = [n[:-len(PART_SUFFIX)] for n in names if n.startswith("part_") and n.endswith(PART_SUFFIX)]
    ledgers = [n for n in names if n.endswith(LEDGER_SUFFIX) and not n.startswith(".")]
    pset = set(parts)
    lset = {n[:-len(LEDGER_SUFFIX)] for n in ledgers}
    return dict(parts=parts, ledgers=ledgers,
                orphan_parts=[p for p in parts if p not in lset],
                orphan_ledgers=[n for n in ledgers if n.startswith("part_") and n[:-len(LEDGER_SUFFIX)] not in pset],
                tmp=[n for n in names if n.startswith("part_") and TMP_SUFFIX in n[5:]
                     and not n.endswith((CORRUPT_SUFFIX, ORPHAN_SUFFIX))],
                corrupt=[n for n in names if n.endswith(CORRUPT_SUFFIX)],
                orphaned=[n for n in names if n.endswith(ORPHAN_SUFFIX)],
                runner_records=[n for n in names if n.startswith("runner_") and n.endswith(".json")])


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open("/proc/{}/stat".format(pid)) as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def _tmp_active(path, min_age, now):
    """True if a temporary file may still be written: younger than ``min_age`` s, or named after a runner that is
    alive on this host."""
    try:
        age = now - os.lstat(path).st_mtime
    except FileNotFoundError:
        return True                                 # gone meanwhile: nothing to do
    if age < min_age:
        return True
    info = parse_part_name(path)
    return bool(info and info["host"] == short_host() and info["pid"] and _pid_alive(info["pid"]))


def recover(out_dir, min_age=0.0, fsync=True, log=None):
    """
    Repair a tag directory after a crash (see the module notes): rebuild the ledger of every archive without one,
    rename a corrupt orphan archive to ``*.corrupt`` and a part ledger without archive to ``*.orphan``, and remove
    stale temporaries of the packer.

    Parameters
    ----------
    out_dir: str
        The tag directory.
    min_age: float
        Temporaries younger than this (s) are kept; so are those named after a runner alive on this host.
    fsync: bool
        fsync rebuilt ledgers.
    log: callable, optional
        Receives one line per action.

    Returns
    -------
    dict
        rebuilt (part names), corrupt, orphan_ledgers, removed_tmp, kept_tmp (lists of names).

    Raises
    ------
    OSError, MemoryError
        Reading an orphan archive failed for another reason than corruption (:data:`CORRUPT_ERRORS`); nothing is
        renamed, the next recover tries again.

    Notes
    -----
    Run it when no packer writes into ``out_dir`` (one runner per tag, as the legacy task). A rebuilt ledger holds
    the archive's meta.txt files in directory-name order, byte-identical to the ledger :func:`pack` would have
    written (and, line by line, to the legacy one).
    """
    # PP 2026-10-02: new (the legacy task had no recovery: an archive without .idx was packed again on restart)
    say = log or (lambda *a: None)
    st = scan(out_dir)
    out = dict(rebuilt=[], corrupt=[], orphan_ledgers=[], removed_tmp=[], kept_tmp=[])
    now = time.time()
    for n in st["tmp"]:
        p = os.path.join(out_dir, n)
        if _tmp_active(p, min_age, now):
            out["kept_tmp"].append(n)
            continue
        try:
            os.unlink(p)
            out["removed_tmp"].append(n)
            say("removed stale temporary {}".format(n))
        except FileNotFoundError:
            pass
    for part in st["orphan_parts"]:
        tar = os.path.join(out_dir, part + PART_SUFFIX)
        try:
            data = ledger_bytes_from_part(tar)
        except CORRUPT_ERRORS as e:                                  # truncated / corrupt gzip or tar
            # PP 2026-10-02: reviewer: only these; a transient OSError (EIO, EACCES) or MemoryError propagates, so
            # a valid archive is never set aside (its points would be run again)
            os.replace(tar, tar + CORRUPT_SUFFIX)
            out["corrupt"].append(part)
            say("corrupt orphan archive {} ({!r}) -> {}".format(part, e, part + PART_SUFFIX + CORRUPT_SUFFIX))
            continue
        _write_ledger(out_dir, part, data, fsync=fsync, unique=True)
        out["rebuilt"].append(part)
        say("rebuilt ledger of {} ({} points)".format(part, data.count(b"\n")))
    for n in st["orphan_ledgers"]:
        p = os.path.join(out_dir, n)
        os.replace(p, p + ORPHAN_SUFFIX)
        out["orphan_ledgers"].append(n)
        say("ledger without archive {} -> {}".format(n, n + ORPHAN_SUFFIX))
    return out


# ----------------------------------------------------------------------------------------------------------------
# CLI: python -m ppmpy.synspec.fastwind.archive {pack, recover, scan, ledger}
# ----------------------------------------------------------------------------------------------------------------
def _stamp(msg):
    sys.stdout.write("{} {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S"), msg))
    sys.stdout.flush()


def ignore_signals(names=CHILD_IGNORED):
    """
    SIG_IGN for the named signals, then unblock them (a child started by the runner with them blocked: a signal
    that arrived in between is discarded instead of killing the child during its start-up).
    """
    # PP 2026-10-02: new (reviewer: a step-wide Slurm signal in the packer's first ~10-30 ms killed it)
    sigs = [getattr(signal, n) for n in names]
    for s in sigs:
        signal.signal(s, signal.SIG_IGN)
    if hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, sigs)


def _write_report(path, obj):
    if not path:
        return
    tmp = path + TMP_SUFFIX
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.write("\n")
    os.replace(tmp, path)


def main(argv=None):
    """Command line: pack (the packer child of the runner), recover, scan, ledger. Returns the exit status."""
    # PP 2026-10-02: new (M6); the packer runs as a child process of batch.run_models
    ap = argparse.ArgumentParser(prog="python -m ppmpy.synspec.fastwind.archive",
                                 description="Pack / recover the per-point parts of a FASTWIND sphere run.")
    sub = ap.add_subparsers(dest="cmd")
    sub.required = True
    p = sub.add_parser("pack", help="pack the finished results of RES_DIR into one part in OUT_DIR")
    p.add_argument("res_dir")
    p.add_argument("stage_dir")
    p.add_argument("out_dir")
    p.add_argument("--part", help="part name without suffix (default: generated)")
    p.add_argument("--compresslevel", type=int, default=COMPRESSLEVEL)
    p.add_argument("--no-fsync", action="store_true")
    p.add_argument("--report", help="write the result as JSON to this file")
    p.add_argument("--ignore-signals", action="store_true",
                   help="ignore SIGINT / SIGTERM / SIGUSR1 / SIGHUP (the runner's packer always finishes its part)")
    p = sub.add_parser("recover", help="rebuild missing ledgers, drop stale temporaries in tag directories")
    p.add_argument("out_dir", nargs="+")
    p.add_argument("--min-age", type=float, default=0.0)
    p = sub.add_parser("scan", help="print the state of tag directories (JSON)")
    p.add_argument("out_dir", nargs="+")
    p = sub.add_parser("ledger", help="print the ledger of an archive (rebuilt from its meta.txt files)")
    p.add_argument("part")
    a = ap.parse_args(argv)
    if a.cmd == "pack":
        if a.ignore_signals:
            ignore_signals()
        elif hasattr(signal, "pthread_sigmask"):
            signal.pthread_sigmask(signal.SIG_UNBLOCK, [getattr(signal, n) for n in CHILD_IGNORED])
        try:
            r = pack(a.res_dir, a.stage_dir, a.out_dir, part=a.part, compresslevel=a.compresslevel,
                     fsync=not a.no_fsync, log=_stamp)
        except Exception as e:
            _write_report(a.report, dict(ok=False, error=repr(e)))
            raise
        r = dict(r, ok=True)
        _write_report(a.report, r)
        if r["part"] is None:
            _stamp("nothing to pack")
        return 0
    if a.cmd == "recover":
        for d in a.out_dir:
            r = recover(d, min_age=a.min_age, log=_stamp)
            print(json.dumps(dict(r, out_dir=d)))
        return 0
    if a.cmd == "scan":
        for d in a.out_dir:
            print(json.dumps(dict(scan(d), out_dir=d)))
        return 0
    sys.stdout.buffer.write(ledger_bytes_from_part(a.part))
    return 0


if __name__ == "__main__":
    sys.exit(main())
