"""
A FASTWIND installation: where the code and its data are, checks, fingerprints, and staging of a minimal run root
(standard library only).

FASTWIND is not part of ppmpy; the user gives its location. The layout (FASTWIND v10.6.4.1 as built for M424 in
``/scratch/ppathak/FW_10.6.4.1``)::

    <root>/inicalc/{DATA,OP_DATA_NEW,ATOMDAT_NEW,RaymondSmith}     data directories read by the codes
    <root>/inicalc/HOPFPARA_ALL_{HHe,met}                          Hopf-parameter tables
    <root>/<build>/{pnlte_<tag>.eo, pformalsol_<tag>.eo, ATOM_FILE, <tag>.dat}

``<tag>`` is the model atom: the first line of ATOM_FILE without blanks and without '.dat' (A10HHe.dat ->
A10HHe; fastwind_run.sh). The codes open their data relative to the working directory (``../inicalc/...``,
``../HOPFPARA_ALL_*``) and write scratch files into it, so every model runs in its own directory one level below a
*run root* that holds those links or copies: :meth:`FastwindInstall.stage` builds one (the node-local copy of
fw_sphere_task.sh, or links). ``inicalc/inicalc`` is a link to inicalc itself in the M424 install; nothing here
follows or copies it.

The binaries of the M424 build need the ELF interpreter of the cluster software stack
(``/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/lib64/ld-linux-x86-64.so.2``): they run on the host, not in
the Python container. :meth:`FastwindInstall.check` reports an interpreter that is absent on this machine.

PP 2026-10-02: new (M6); ported from fastwind_run.sh:22-42 (atom tag, root links) and fw_sphere_task.sh:37-41
(node-local copy) of the project stellar-atmosphere-KU-Leuven.
"""
import hashlib
import itertools
import os
import shutil
import struct
import threading

DATA_DIRS = ("DATA", "OP_DATA_NEW", "ATOMDAT_NEW", "RaymondSmith")
"""inicalc subdirectories a model needs (fw_sphere_task.sh:38)."""

HOPF_FILES = ("HOPFPARA_ALL_HHe", "HOPFPARA_ALL_met")
"""Hopf-parameter tables, read from the parent of the run directory (CLAUDE.md pitfall 1)."""

ENV_ROOT = "PPMPY_FASTWIND_ROOT"
ENV_BUILD = "PPMPY_FASTWIND_BUILD"
ENV_FORMAL_BUILD = "PPMPY_FASTWIND_FORMAL_BUILD"
ENV_LAUNCHER = "PPMPY_FASTWIND_LAUNCHER"

STAGE_MANIFEST = "FASTWIND_STAGE.txt"
"""key = value file written into a staged run root (install, builds, tag, mode)."""

_PT_INTERP = 3


_seq = itertools.count()


def tmp_path(path, kind="tmp"):
    """
    A hidden temporary name next to ``path`` that is unique per call: ``.<name>.<kind><pid>.<thread id>.<n>``.

    Unique across processes (pid), threads of one process (thread id) and calls of one thread (a counter), so
    concurrent writers never unlink or rename each other's temporaries. Hidden (leading '.'), so ``ls``-based
    packers and :meth:`StagedRoot.bin_files` skip it.
    """
    # PP 2026-10-02: new (M6); reviewer: names with only os.getpid() collided between threads of one process
    d = os.path.dirname(os.path.abspath(path))
    return os.path.join(d, ".{}.{}{}.{}.{}".format(os.path.basename(path), kind, os.getpid(), threading.get_ident(),
                                                   next(_seq)))


def _unlink_quiet(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def atomic_write(path, data):
    """
    Write bytes (or str, UTF-8) to ``path`` through a unique hidden temporary name in the same directory, then
    rename (:func:`tmp_path`; safe from many threads and processes at once; the file mode follows the umask).
    """
    # PP 2026-10-02: new (M6)
    if isinstance(data, str):
        data = data.encode()
    tmp = tmp_path(path, "tmp")
    try:
        with open(tmp, "xb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        _unlink_quiet(tmp)
        raise
    return path


def atomic_symlink(target, path):
    """
    Make ``path`` a symbolic link to ``target``, atomically: link to a unique temporary name, then rename over
    ``path`` (a plain symlink on an existing directory link fails, and concurrent launches race; fastwind_run.sh:35).
    An existing link with the same target is left alone; when the rename fails (e.g. ``path`` is a real directory)
    but ``path`` is meanwhile a link to ``target`` (another thread or process won), that is accepted.
    """
    # PP 2026-10-02: ported from fastwind_run.sh:35 (link(): ln -s tmp + mv -Tf)
    try:
        if os.readlink(path) == target:
            return path
    except OSError:
        pass
    tmp = tmp_path(path, "lnk")
    os.symlink(target, tmp)
    try:
        os.replace(tmp, path)
    except OSError:
        _unlink_quiet(tmp)
        try:
            if os.readlink(path) == target:
                return path
        except OSError:
            pass
        raise
    return path


def elf_interpreter(path):
    """
    The program interpreter (PT_INTERP) of an ELF executable, or None for a file that is not ELF or has none
    (static, script).

    Reads the ELF and program headers with :mod:`struct` (32/64 bit, either byte order).
    """
    # PP 2026-10-02: new (M6)
    with open(path, "rb") as f:
        ident = f.read(16)
        if len(ident) < 16 or ident[:4] != b"\x7fELF":
            return None
        cls, data = ident[4], ident[5]
        end = "<" if data == 1 else ">"
        if cls == 2:
            hdr = f.read(48)
            if len(hdr) < 48:
                return None
            phoff = struct.unpack(end + "Q", hdr[16:24])[0]
            phentsize, phnum = struct.unpack(end + "HH", hdr[38:42])
            fmt, size = end + "IIQQQQQQ", 56
        elif cls == 1:
            hdr = f.read(36)
            if len(hdr) < 36:
                return None
            phoff = struct.unpack(end + "I", hdr[12:16])[0]
            phentsize, phnum = struct.unpack(end + "HH", hdr[26:30])
            fmt, size = end + "IIIIIIII", 32
        else:
            return None
        if phentsize < size:
            return None
        for i in range(phnum):
            f.seek(phoff + i * phentsize)
            ph = f.read(size)
            if len(ph) < size:
                return None
            v = struct.unpack(fmt, ph)
            if v[0] != _PT_INTERP:
                continue
            if cls == 2:
                off, filesz = v[2], v[5]
            else:
                off, filesz = v[1], v[4]
            f.seek(off)
            return f.read(filesz).split(b"\0", 1)[0].decode("latin-1")
    return None


def has_imu_patch(path):
    """True if the executable ``path`` contains the bytes 'OUT_IMU' (the intensity patch formalsol_imu.patch)."""
    # PP 2026-10-02: new (M6); the patch opens OUT_IMU.<line>_<suffix>
    needle = b"OUT_IMU"
    prev = b""
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                return False
            if needle in prev[-len(needle):] + b:
                return True
            prev = b


def _sha256(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_atom_tag(build_dir):
    """
    (tag, atom file name) from ``<build>/ATOM_FILE``: first line without blanks (A10HHe.dat) and that name without
    '.dat' (A10HHe), as fastwind_run.sh:28-29.
    """
    # PP 2026-10-02: ported from fastwind_run.sh:28-29 (head -1 ATOM_FILE | tr -d ' '; ${ATOM%.dat})
    with open(os.path.join(build_dir, "ATOM_FILE"), "rb") as f:
        first = f.readline().decode("latin-1").rstrip("\r\n")
    atom = first.replace(" ", "")
    if not atom:
        raise ValueError("{}/ATOM_FILE: empty first line".format(build_dir))
    tag = atom[:-4] if atom.endswith(".dat") else atom
    return tag, atom


class StagedRoot:
    """
    A staged run root: ``inicalc/<data dirs>``, ``HOPFPARA_ALL_*`` and ``bin/`` (the executables, ATOM_FILE and the
    atom file) in one directory; model run directories go directly below it.

    Attributes
    ----------
    root: str
        The directory.
    tag, atom: str
        Model atom tag and file name (A10HHe, A10HHe.dat).
    launcher: tuple of str
        Command prefix for the executables (e.g. ``('taskset', '-c', '5')``); empty by default.
    mode: str
        'link' or 'copy'.
    """

    # PP 2026-10-02: new (M6)
    def __init__(self, root, tag, atom, launcher=(), mode="link", has_imu=None):
        self.root = os.path.abspath(root)
        self.tag = tag
        self.atom = atom
        self.launcher = tuple(launcher)
        self.mode = mode
        self.has_imu = has_imu

    @property
    def bin_dir(self):
        return os.path.join(self.root, "bin")

    @property
    def pnlte(self):
        return "pnlte_{}.eo".format(self.tag)

    @property
    def pformalsol(self):
        return "pformalsol_{}.eo".format(self.tag)

    def bin_files(self):
        """
        Names in bin/ that every run directory links (fw_sphere_point.sh: ``ln -s $FWL/bin/* .``, which skips
        dotfiles: hidden temporaries of a crashed or concurrent staging are never linked).
        """
        return sorted(f for f in os.listdir(self.bin_dir) if not f.startswith("."))

    def check(self):
        """Problems of the staged root (missing parts), as a list of strings."""
        probs = []
        for d in DATA_DIRS:
            if not os.path.isdir(os.path.join(self.root, "inicalc", d)):
                probs.append("staged root: inicalc/{} missing".format(d))
        for h in HOPF_FILES:
            if not os.path.isfile(os.path.join(self.root, h)):
                probs.append("staged root: {} missing".format(h))
        for b in (self.pnlte, self.pformalsol, "ATOM_FILE", self.atom):
            if not os.path.isfile(os.path.join(self.bin_dir, b)):
                probs.append("staged root: bin/{} missing".format(b))
        return probs

    @classmethod
    def open(cls, root, launcher=None):
        """Re-open a root staged earlier (reads :data:`STAGE_MANIFEST`)."""
        man = read_manifest(os.path.join(root, STAGE_MANIFEST))
        if launcher is None:
            launcher = tuple(t for t in man.get("launcher", "").split("\t") if t)
        imu = man.get("has_imu")
        return cls(root, man["tag"], man["atom"], launcher=launcher, mode=man.get("mode", "link"),
                   has_imu=None if imu in (None, "None") else imu == "True")

    def __repr__(self):
        return "StagedRoot({!r}, tag={!r}, mode={!r})".format(self.root, self.tag, self.mode)


def read_manifest(path):
    """dict of a key = value text file (:data:`STAGE_MANIFEST`)."""
    out = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                out[k.strip()] = v.strip()
    return out


class FastwindInstall:
    """
    A FASTWIND installation (code + data), located by the user.

    Parameters
    ----------
    root: str
        Top directory with ``inicalc/`` (M424: ``/scratch/ppathak/FW_10.6.4.1``).
    build: str
        Build directory with pnlte / pformalsol / ATOM_FILE / atom file: a path, or a name below ``root``
        (M424: ``v10.6_HHe``).
    atom: str, optional
        Model atom tag (A10HHe); default from ``<build>/ATOM_FILE``. Must agree with it.
    formal_build: str, optional
        Build whose pformalsol is used instead (M424: ``v10.6_HHe_imu``, the patched pformalsol that also writes
        OUT_IMU.*); default ``build``.
    launcher: sequence of str
        Command prefix for the executables (default none).

    Notes
    -----
    Nothing is read at construction except ATOM_FILE (when ``atom`` is not given); :meth:`check` lists problems.
    """

    # PP 2026-10-02: new (M6)
    def __init__(self, root, build, atom=None, formal_build=None, launcher=()):
        self.root = os.path.abspath(root)
        self.build = self._resolve(build)
        self.formal_build = self._resolve(formal_build) if formal_build else self.build
        self.launcher = tuple(launcher) if not isinstance(launcher, str) else tuple(launcher.split())
        self._atom_given = atom
        if atom is None:
            self.tag, self.atom = read_atom_tag(self.build)
        else:
            self.tag = atom[:-4] if atom.endswith(".dat") else atom
            self.atom = self.tag + ".dat"

    def _resolve(self, b):
        return os.path.abspath(b if os.path.isabs(b) else os.path.join(self.root, b))

    @classmethod
    def from_env(cls, environ=None, **kw):
        """
        From the environment: :data:`ENV_ROOT` (required), :data:`ENV_BUILD` (required), :data:`ENV_FORMAL_BUILD`,
        :data:`ENV_LAUNCHER` (split on blanks). Keywords override.

        Raises
        ------
        KeyError
            When the root or build is not set.
        """
        env = os.environ if environ is None else environ
        args = dict(root=env.get(ENV_ROOT), build=env.get(ENV_BUILD), formal_build=env.get(ENV_FORMAL_BUILD) or None,
                    launcher=tuple(env.get(ENV_LAUNCHER, "").split()))
        args.update(kw)
        for k, e in (("root", ENV_ROOT), ("build", ENV_BUILD)):
            if not args.get(k):
                raise KeyError("FASTWIND location not set: {} (or pass {}=...)".format(e, k))
        return cls(**args)

    # -- files ----------------------------------------------------------------------------------------
    @property
    def pnlte(self):
        """Path of pnlte_<tag>.eo of the build."""
        return os.path.join(self.build, "pnlte_{}.eo".format(self.tag))

    @property
    def pformalsol(self):
        """Path of pformalsol_<tag>.eo of the formal build (the build by default)."""
        return os.path.join(self.formal_build, "pformalsol_{}.eo".format(self.tag))

    def data_dir(self, name):
        return os.path.join(self.root, "inicalc", name)

    def hopf_file(self, name):
        return os.path.join(self.root, "inicalc", name)

    def bin_sources(self):
        """name -> source path of the files a run root's bin/ needs (fw_sphere_task.sh:40)."""
        return {os.path.basename(self.pnlte): self.pnlte, os.path.basename(self.pformalsol): self.pformalsol,
                "ATOM_FILE": os.path.join(self.build, "ATOM_FILE"), self.atom: os.path.join(self.build, self.atom)}

    # -- checks ---------------------------------------------------------------------------------------
    def check(self):
        """
        Problems that would stop a model, as a list of strings (empty: the install looks usable here).

        Missing root, data directories, Hopf tables, build directories, executables (or not executable),
        ATOM_FILE / atom file, an ATOM_FILE that disagrees with ``atom``, and an ELF interpreter of the
        executables that does not exist on this machine (e.g. the /cvmfs loader inside a container). Non-ELF
        executables (scripts) skip the interpreter check.
        """
        probs = []
        if not os.path.isdir(self.root):
            return ["FASTWIND root {} does not exist".format(self.root)]
        for d in DATA_DIRS:
            if not os.path.isdir(self.data_dir(d)):
                probs.append("data directory missing: {}".format(self.data_dir(d)))
        for h in HOPF_FILES:
            if not os.path.isfile(self.hopf_file(h)):
                probs.append("Hopf table missing: {}".format(self.hopf_file(h)))
        for b in sorted({self.build, self.formal_build}):
            if not os.path.isdir(b):
                probs.append("build directory missing: {}".format(b))
        af = os.path.join(self.build, "ATOM_FILE")
        if os.path.isfile(af):
            try:
                tag, _ = read_atom_tag(self.build)
                if tag != self.tag:
                    probs.append("ATOM_FILE names {} but atom {} was given".format(tag, self.tag))
            except (OSError, ValueError) as e:
                probs.append("ATOM_FILE unreadable: {}".format(e))
        for name, src in self.bin_sources().items():
            if not os.path.isfile(src):
                probs.append("missing: {}".format(src))
        seen = set()
        for exe in (self.pnlte, self.pformalsol):
            if exe in seen or not os.path.isfile(exe):
                continue
            seen.add(exe)
            if not os.access(exe, os.X_OK):
                probs.append("not executable: {}".format(exe))
            try:
                interp = elf_interpreter(exe)
            except OSError as e:
                probs.append("cannot read {}: {}".format(exe, e))
                continue
            if interp and not os.path.exists(interp):
                probs.append("ELF interpreter {} of {} does not exist on this machine (run on the host, not in "
                             "the container)".format(interp, exe))
        return probs

    def has_imu_patch(self):
        """True if the pformalsol used (formal build) contains 'OUT_IMU', i.e. has the intensity patch."""
        # PP 2026-10-02: new (M6); the patch formalsol_imu.patch opens OUT_IMU.<line>_<suffix>
        return has_imu_patch(self.pformalsol)

    def fingerprint(self):
        """
        dict: root, build, formal_build, tag, and the sha256 of pnlte, pformalsol, ATOM_FILE and the atom file
        (``sha256:<name>`` keys; a missing file gives None).
        """
        out = dict(root=self.root, build=self.build, formal_build=self.formal_build, tag=self.tag)
        for name, src in sorted(self.bin_sources().items()):
            out["sha256:" + name] = _sha256(src) if os.path.isfile(src) else None
        return out

    # -- staging --------------------------------------------------------------------------------------
    def stage(self, dest, mode="link"):
        """
        Build a run root in ``dest``: ``inicalc/<DATA_DIRS>``, ``HOPFPARA_ALL_{HHe,met}``, ``bin/`` (pnlte, the
        formal build's pformalsol, ATOM_FILE, atom file) and :data:`STAGE_MANIFEST`.

        Parameters
        ----------
        dest: str
            Directory (created). M424 production: node memory, ``/dev/shm/...`` with mode 'copy'.
        mode: str
            'link': symbolic links to the install (cheap; shared-filesystem reads at run time);
            'copy': real copies (fw_sphere_task.sh's node-local install, ~0.4 GB).

        Returns
        -------
        StagedRoot
            ``has_imu`` is read from the staged ``bin/pformalsol`` (not from the install).

        Raises
        ------
        FileNotFoundError
            The install is incomplete (:meth:`check`, the ELF interpreter excepted).
        ValueError
            ``dest`` holds copied data directories staged from another install (they would be kept, stale).

        Notes
        -----
        Safe to run concurrently on the same ``dest`` from many threads and processes, and to repeat: every
        temporary name is unique per call (:func:`tmp_path`); links are made under a temporary name and renamed into
        place; copies are made into a hidden temporary directory or file and renamed into place. A data directory
        that already exists as a real directory (an earlier 'copy' staging of the same install) is kept, in both
        modes; a link from an earlier 'link' staging is replaced by a copy in 'copy' mode. The small files (bin/ and
        the Hopf tables) are compared with the source (size and sha256) and replaced when they differ, so staging a
        rebuilt or another build into the same ``dest`` never leaves a stale binary. Only the four data directories
        are copied (the install's ``inicalc/inicalc`` self link is never followed); copies keep symbolic links inside
        the data directories as links. Staging *different* builds into one ``dest`` at the same time gives a mixture;
        don't.
        """
        # PP 2026-10-02: ported from fw_sphere_task.sh:37-41 (copy mode) and fastwind_run.sh:31-42 (link mode)
        if mode not in ("link", "copy"):
            raise ValueError("mode must be 'link' or 'copy', got {!r}".format(mode))
        probs = [p for p in self.check() if "ELF interpreter" not in p]
        if probs:
            raise FileNotFoundError("cannot stage FASTWIND: " + "; ".join(probs))
        dest = os.path.abspath(dest)
        man_path = os.path.join(dest, STAGE_MANIFEST)
        if os.path.isfile(man_path):
            old = read_manifest(man_path).get("install")
            copied = [d for d in DATA_DIRS if os.path.isdir(os.path.join(dest, "inicalc", d))
                      and not os.path.islink(os.path.join(dest, "inicalc", d))]
            if old and old != self.root and copied:
                raise ValueError("{} holds data directories copied from the install {} (not {}); stage into a new "
                                 "directory".format(dest, old, self.root))
        for d in (dest, os.path.join(dest, "inicalc"), os.path.join(dest, "bin")):
            os.makedirs(d, exist_ok=True)
        items = [(self.data_dir(d), os.path.join(dest, "inicalc", d)) for d in DATA_DIRS]
        items += [(self.hopf_file(h), os.path.join(dest, h)) for h in HOPF_FILES]
        items += [(src, os.path.join(dest, "bin", name)) for name, src in sorted(self.bin_sources().items())]
        for src, dst in items:
            if mode == "link":
                _link_into_place(src, dst)
            else:
                _copy_into_place(src, dst)
        st = StagedRoot(dest, self.tag, self.atom, launcher=self.launcher, mode=mode)
        try:
            imu = has_imu_patch(os.path.join(st.bin_dir, st.pformalsol))
        except OSError:
            imu = None
        st.has_imu = imu
        man = "".join("{} = {}\n".format(k, v) for k, v in (
            ("install", self.root), ("build", self.build), ("formal_build", self.formal_build), ("tag", self.tag),
            ("atom", self.atom), ("mode", mode), ("has_imu", imu), ("launcher", "\t".join(self.launcher))))
        atomic_write(man_path, man)
        return st

    def __repr__(self):
        return "FastwindInstall({!r}, build={!r}, formal_build={!r}, tag={!r})".format(
            self.root, self.build, self.formal_build, self.tag)


def _same_file(a, b):
    """True if the regular files ``a`` and ``b`` have the same size and sha256."""
    try:
        if os.path.getsize(a) != os.path.getsize(b):
            return False
    except OSError:
        return False
    return _sha256(a) == _sha256(b)


def _link_into_place(src, dst):
    """'link' staging of one item: a real directory already there (an earlier 'copy' staging) is kept."""
    if os.path.isdir(dst) and not os.path.islink(dst):
        return dst
    return atomic_symlink(src, dst)


def _copy_into_place(src, dst):
    """
    'copy' staging of one item through a unique hidden temporary name (:func:`tmp_path`).

    A directory is kept when ``dst`` is already a real directory (another thread or process may have won the race);
    a link left by a 'link' staging is replaced. A file is kept when ``dst`` is a regular file with the same size
    and sha256 as ``src``, else replaced (a rebuilt or other build never leaves a stale binary).
    """
    # PP 2026-10-02: ported from fw_sphere_task.sh:39-41 (cp -r); unique temporaries and the file comparison are new
    if os.path.isdir(src):
        if os.path.isdir(dst) and not os.path.islink(dst):
            return dst
        tmp = tmp_path(dst, "cp")
        try:
            shutil.copytree(src, tmp, symlinks=True)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        if os.path.islink(dst):             # a link from an earlier 'link' staging: replace it
            _unlink_quiet(dst)              # (gone already, or a directory meanwhile: the rename decides)
        try:
            os.rename(tmp, dst)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
            if os.path.isdir(dst) and not os.path.islink(dst):     # another thread or process won the race
                return dst
            raise
        return dst
    if os.path.isfile(dst) and not os.path.islink(dst) and _same_file(src, dst):
        return dst
    tmp = tmp_path(dst, "cp")
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        _unlink_quiet(tmp)
        raise
    return dst
