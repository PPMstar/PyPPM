"""
File helpers: atomic writes, a small provenance record ('_meta') stored with every product,
and zero-copy access to members of uncompressed .npz files.

PP 2026-10-01: new. Products are plain .npz files (uncompressed, no pickles); readers
accept files without '_meta' (the M424 production predates it).
"""
import datetime
import json
import os
import platform
import socket
import subprocess
import sys
import zipfile

import numpy as np


def _ppmpy_info():
    """Path, git commit and dirty flag of the imported ppmpy (None where unknown)."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    info = dict(path=root, commit=None, dirty=None)
    try:
        info["commit"] = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                                        text=True, timeout=10).stdout.strip() or None
        st = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                            capture_output=True, text=True, timeout=10)
        info["dirty"] = bool(st.stdout.strip()) if st.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def file_identity(path):
    """Path, size and modification time of an input file (cheap; no hashing)."""
    st = os.stat(path)
    return dict(path=os.path.abspath(path), size=st.st_size,
                mtime=datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"))


def _input_record(v):
    """file_identity for an existing path; other values (lists of paths, labels) are recorded as given."""
    # PP 2026-10-02: migration finding: a list of raw directories in ImuLibrary.inputs raised TypeError
    if isinstance(v, (str, bytes, os.PathLike)) and os.path.exists(v):
        return file_identity(v)
    if isinstance(v, (list, tuple)):
        return [_input_record(x) for x in v]
    return v if isinstance(v, (str, int, float, bool)) else str(v)


def make_meta(kind, params=None, inputs=None, **extra):
    """
    Provenance record for a product.

    Parameters
    ----------
    kind: str
        Product type, e.g. 'synspec.zerocross'.
    params: dict, optional
        The parameters that produced it (JSON-serialisable).
    inputs: dict, optional
        name -> path of input files (recorded with size and mtime).
    **extra:
        Further JSON-serialisable entries.
    """
    import scipy
    from . import API_VERSION
    meta = dict(kind=kind, api=API_VERSION, created=datetime.datetime.now().isoformat(timespec="seconds"),
                host=socket.gethostname(), argv=list(sys.argv), python=platform.python_version(),
                numpy=np.__version__, scipy=scipy.__version__, ppmpy=_ppmpy_info(),
                slurm_job=os.environ.get("SLURM_JOB_ID"), params=params or {},
                inputs={k: _input_record(v) for k, v in (inputs or {}).items() if v is not None})
    meta.update(extra)
    return meta


def save_npz(path, arrays, meta=None):
    """
    Write an uncompressed .npz atomically (temporary name + os.replace), with an
    optional '_meta' member holding the provenance record as JSON text.

    Parameters
    ----------
    path: str
        Target file (must end in .npz).
    arrays: dict
        name -> array-like.
    meta: dict, optional
        Provenance record (:func:`make_meta`).
    """
    if not path.endswith(".npz"):
        raise ValueError("path must end in .npz: {}".format(path))
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    out = dict(arrays)
    if meta is not None:
        out["_meta"] = np.array(json.dumps(meta, sort_keys=True, default=str))
    tmp = "{}.tmp{}.npz".format(path[:-4], os.getpid())
    try:
        np.savez(tmp, **out)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def read_meta(npz):
    """The '_meta' record of an .npz (path or open NpzFile); {} for files written before it existed."""
    z = np.load(npz) if isinstance(npz, str) else npz
    if "_meta" not in z.files:
        return {}
    return json.loads(str(z["_meta"]))


def npz_member_memmap(path, key):
    """
    Read-only memory map of one member of an uncompressed .npz, without reading the
    file (e.g. one array of the 7.35 GB M424 profiles.npz).

    Raises
    ------
    ValueError
        If the member is compressed.
    """
    name = key if key.endswith(".npy") else key + ".npy"
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(name)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("member {} of {} is compressed; cannot memory-map it".format(key, path))
        with open(path, "rb") as f:
            f.seek(info.header_offset)
            local = f.read(30)
            nlen = int.from_bytes(local[26:28], "little")
            xlen = int.from_bytes(local[28:30], "little")
            start = info.header_offset + 30 + nlen + xlen
            f.seek(start)
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            else:
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            offset = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, order="F" if fortran else "C", offset=offset)
