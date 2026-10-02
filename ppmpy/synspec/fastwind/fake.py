"""
A fake FASTWIND for tests: executable standard-library Python stand-ins for pnlte and pformalsol that honour the
file contract of the real codes, and :func:`selftest` (standard library only).

The contract (what :mod:`.model` and the readers rely on), as the real v10.6.4.1 codes:

* run in a directory one level below a root with ``../inicalc/{DATA,OP_DATA_NEW,ATOMDAT_NEW,RaymondSmith}`` and
  ``../HOPFPARA_ALL_{HHe,met}``; read ``ATOM_FILE`` (its first line names the atom file, which must be present) and
  ``INDAT.DAT``; need the catalogue directory ``<MODNAM>/``; write scratch files (control.dat, OX) into the run
  directory; a missing file stops them like a Fortran runtime error (``forrtl: severe (29)``, exit 29);
* pnlte prints 'FINAL ITERATION NO . 7 FOR HYDRO MODEL', 'ITERATION NO 0', 'ITERATION NO <n>' per NLTE iteration,
  'T(TAUROSS=2/3) = <T>' lines, the end summary and '! ESTO ES EL ACABOSE !' (exit 0); writes ``<MODNAM>/`` model
  files (MODEL, NLTE_POP, ... CONT_FORMAL_ALL), CONVERG ('IIT EMAX MEANERR') and MAXTCORR.dat ('NCOR EMAXTC IIT')
  that obey nlte.f90's convergence rules (temperature converged at iteration 23, NLTE converged at
  ``converge_at``, one more iteration, or the cap ITSTART + ITMORE);
* pformalsol reads the model name, the turbulence answer and IESCAT from stdin, FORMAL_INPUT (free-field, ':'
  comments), the model files, and writes ``<MODNAM>/OUT.<line>_<suffix>`` (formalsol.f90's suffix rules) with 161
  rows 'index x lambda F_cont F/F_cont F_rot' and the trailer EW; the intensity variant also writes OUT_IMU.* in
  the layout of formalsol_imu.patch (and is the only one whose file contains that name). Like formalsol.f90, it
  treats the lines one after the other and opens a line's OUT (and OUT_IMU) file *before* computing it, writing
  the tables only at the end, so a pformalsol killed or crashed during a line leaves that file empty or truncated
  under its final name.

Behaviour is programmable per T_eff (INDAT TEFF formatted '%.3f') through ``<build>/fake_config.json``
(:func:`set_fake_config`): ``fail`` (deterministic 'error in ne -- nlteopt' after ``fail_after`` iterations, exit
0, no ACABOSE), ``crash`` (Fortran-like SIGSEGV message, exit 174), ``hang`` (prints a few iterations, starts a
helper child in the same process group writing ``hang_child.pid``, then sleeps), ``cap`` (never converges: stops at
the cap), ``formal_fail`` (pformalsol stops before any OUT file, exit 0), ``formal_hang`` (hangs inside the last
line, whose OUT file is open and empty; the earlier lines are complete), ``formal_partial`` (the last line's table
is cut after ``partial_rows`` rows (default 20), then exit 174 like a Fortran crash), ``formal_partial_hang`` (the
same cut, then hang); ``converge_at`` (default 30) and ``sleep`` (seconds per pnlte run). The environment
variable PPMPY_FAKE_FASTWIND_MODE=hang makes a fake hang at once (a decoy in tests).

PP 2026-10-02: new (M6).
"""
import json
import os
import stat
import subprocess
import sys
import tempfile
import time

from .install import DATA_DIRS, HOPF_FILES, FastwindInstall

FAKE_TAG = "A10HHe"
FAKE_BUILD = "v10.6_HHe"
FAKE_IMU_BUILD = "v10.6_HHe_imu"
CONFIG_NAME = "fake_config.json"
ENV_MODE = "PPMPY_FAKE_FASTWIND_MODE"
TEFF_LIST_KEYS = ("fail", "crash", "hang", "cap", "formal_fail", "formal_hang", "formal_partial",
                  "formal_partial_hang")
"""Behaviour keys of :func:`set_fake_config` that take lists of T_eff."""

_COMMON = r'''
import json, math, os, re, subprocess, sys, time

CONFIG = {config!r}


def cfg():
    try:
        with open(CONFIG) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {{}}


def out(*a):
    print(*a)
    sys.stdout.flush()


def fortran_missing(path, unit=10):
    out("forrtl: severe (29): file not found, unit {{}}, file {{}}".format(unit, os.path.abspath(path)))
    sys.exit(29)


def need(path, isdir=False):
    if not (os.path.isdir(path) if isdir else os.path.isfile(path)):
        fortran_missing(path)


def check_root():
    for d in {data_dirs!r}:
        need(os.path.join("..", "inicalc", d), isdir=True)
    for h in {hopf!r}:
        need(os.path.join("..", h))
    need("ATOM_FILE")
    with open("ATOM_FILE") as f:
        atom = f.readline().strip().replace(" ", "")
    need(atom)


def tokens(line):
    return [t for t in re.split(r"[,\s]+", line.strip()) if t]


def read_indat(path="INDAT.DAT"):
    need(path)
    with open(path, encoding="latin-1") as f:
        lines = [ln.rstrip("\r") for ln in f.read().split("\n")]
    t2 = tokens(lines[1])
    return dict(modnam=tokens(lines[0])[0], itstart=int(t2[2]), itmore=int(t2[3]),
                teff=float(tokens(lines[3])[0].replace("D", "E")))


def key(teff):
    return "%.3f" % teff


def hang_forever(tag):
    child = subprocess.Popen([sys.executable, "-c", "import time\nwhile True: time.sleep(1)"])
    with open("hang_child.pid", "w") as f:
        f.write("%d\n" % child.pid)
    out("fake {{}}: hanging (child {{}})".format(tag, child.pid))
    while True:
        time.sleep(1)
'''

_PNLTE = r'''
def main():
    if os.environ.get({env!r}) == "hang":
        hang_forever("pnlte")
    c = cfg()
    check_root()
    ind = read_indat()
    mod = ind["modnam"]
    need(mod, isdir=True)
    with open("INDAT.DAT", "rb") as f, open(os.path.join(mod, "INDAT.DAT"), "wb") as g:
        g.write(f.read())
    for scratch in ("control.dat", "OX"):
        with open(scratch, "w") as f:
            f.write("fake scratch\n")
    k = key(ind["teff"])
    time.sleep(float(c.get("sleep", 0)))
    cap = ind["itstart"] + ind["itmore"]
    mode = "ok"
    for m in ("fail", "crash", "hang", "cap"):
        if k in c.get(m, []):
            mode = m
    conv_at = max(int(c.get("converge_at", 30)), 24)      # NLTE convergence counts after T converged (23)
    out(" >>> VTURB =   10.0000000000000       km/s")
    out("             FINAL ITERATION NO .            7  FOR HYDRO MODEL")
    out("                +     ITERATION NO 0     +")
    tconv = 23
    last = cap if (mode == "cap" or conv_at + 1 > cap) else conv_at + 1
    if mode in ("fail", "crash", "hang"):
        last = min(int(c.get("fail_after", 17)), cap)
    conv = open(os.path.join(mod, "CONVERG"), "w")
    mtc = open(os.path.join(mod, "MAXTCORR.dat"), "w")
    ncor = 0
    ttau = ind["teff"] * 1.0356
    for it in range(1, last + 1):
        out("                ++++++++++++++++++++++++++")
        out("                +  ITERATION NO %12d   +" % it)
        out("                ++++++++++++++++++++++++++")
        if it % 2 == 1 and (mode == "cap" or it <= tconv):
            e = 0.0029 if (it == tconv and mode != "cap") else 0.004 + 0.05 * max(0, tconv - it) / tconv
            mtc.write(" %3d %12.5E %3d\n" % (ncor, e, it))
            ncor += 1
            if it == tconv and mode != "cap":
                out("  TEMPERATURE CONVERGED")
        if mode == "cap" or it < conv_at:
            emax, meanerr = 3.0e-3 * (1.0 + max(1, conv_at - it)), -2.0 - 2.0 * min(it, conv_at - 1) / conv_at
        else:
            emax, meanerr = 2.9e-3 * (0.9 ** (it - conv_at)), -4.0 - 0.1 * (it - conv_at)
        conv.write("%12d %24.15E %s %24.15E\n" % (it, emax, " ", meanerr))
        conv.flush()
        ttau *= 0.99999
        out("  T(TAUROSS=2/3) = %24.13f" % ttau)
        out("  CORR. MAX: %24.15E" % emax)
        if mode == "fail" and it == last:
            out(" error in ne -- nlteopt")
            sys.exit(0)
        if mode == "crash" and it == last:
            out("forrtl: severe (174): SIGSEGV, segmentation fault occurred")
            sys.exit(174)
        if mode == "hang" and it == last:
            hang_forever("pnlte")
    conv.close()
    mtc.close()
    for name in {model_files!r}:
        with open(os.path.join(mod, name), "w") as f:
            f.write("fake {{}} of {{}} TEFF {{}}\n".format(name, mod, k))
    out("  TOTAL NUMBER OF APPLIED T CORRECTION(S) %12d" % ncor)
    out("  Maximum relative correction in the last TC  %.15E" % (0.0029 if mode != "cap" else 0.004))
    out("  CPU time:   %.6f" % 1.5)
    out("  ALL LEVELS OK!!!")
    out("! ESTO ES EL ACABOSE !")


main()
'''

_FORMAL = r'''
NROW = 161


def ffr_items(text):
    items = []
    for raw in text.split("\n"):
        line = raw.rstrip("\r")[:80]
        if line.startswith(":"):
            continue
        for p in line.split(":")[0::2]:
            items.extend(t for t in re.split(r"[ \t\r,]+", p) if t)
    return items


def read_lines(path="FORMAL_INPUT"):
    need(path)
    with open(path, encoding="latin-1") as f:
        it = ffr_items(f.read())
    pos, names = 1, []
    while pos < len(it):
        nco = int(float(it[pos + 1]))
        names.append(it[pos])
        pos += 2 + 4 * nco
    return names


def suffix(vt, iescat):
    tok = vt.replace(",", " ").split()
    vtmi = float(tok[0])
    vtma = float(tok[1]) if len(tok) > 1 else vtmi
    v1 = (vtmi * 1.0e5) * 1.0e-5
    if v1 == 0.0:
        return "ESC" if iescat == 1 else ""
    i = int(v1)
    if iescat == 1:
        return "ESC_VT%03d" % i
    return ("VT%03d" if vtmi == vtma else "VTV%03d") % i


def profile(n, teff):
    digits = "".join(ch for ch in n if ch.isdigit())
    lam0 = float(digits) if digits else 5000.0
    depth = 0.3 * (38230.0 / teff) * (1.0 + 0.1 * (len(digits) % 3))
    fc = 1.0e-6 * (teff / 38230.0) ** 4
    return lam0, depth, fc


def main():
    c = cfg()
    model = sys.stdin.readline().strip()
    vt = sys.stdin.readline().strip()
    iescat = int(sys.stdin.readline().strip() or 0)
    out("  INPUT CATALOGUE NAME FOR FORMAL CALC.")
    if os.environ.get({env!r}) == "hang":
        hang_forever("pformalsol")
    check_root()
    names = read_lines()
    for f in {model_files!r}:
        need(os.path.join(model, f))
    with open(os.path.join(model, "MODEL")) as f:
        k = f.read().split()[-1]
    teff = float(k)
    out("  LINES TO BE TREATED")
    for n in names:
        out(" " + n.ljust(20))
    out(" END OF FORMAL-INPUT, NO MORE LINES ")
    if k in c.get("formal_fail", []):
        out(" SOMETHING WRONG WITH NO OF COMPONENTS")
        sys.exit(0)
    suf = suffix(vt, iescat)
    cut = int(c.get("partial_rows", 20))
    for j, n in enumerate(names):
        last = j == len(names) - 1
        # formalsol.f90 opens OUT.<line> (and the intensity file of the patch) before the formal integral of the line
        f = open(os.path.join(model, "OUT." + n + ("_" + suf if suf else "")), "w")
        g = open_imu(model, n, suf)
        if last and k in c.get("formal_hang", []):
            hang_forever("pformalsol")
        lam0, depth, fc = profile(n, teff)
        rows, ew = [], 0.0
        for i in range(1, NROW + 1):
            lam = lam0 + (i - 81) * 0.5
            x = (81 - i) / 66.0
            fn = 1.0 - depth * math.exp(-((lam - lam0) / 1.5) ** 2)
            ew += (1.0 - fn) * 0.5
            rows.append(" %3d %11.5f %15.2f %19.6E %11.5f %15.5f    \n" % (i, x, lam, fc, fn, fn))
        if g is not None:
            write_imu(g, lam0, depth, fc)
            g.close()
        if last and (k in c.get("formal_partial", []) or k in c.get("formal_partial_hang", [])):
            f.write("".join(rows[:cut]))
            f.flush()
            if k in c.get("formal_partial", []):
                out("forrtl: severe (174): SIGSEGV, segmentation fault occurred")
                sys.exit(174)
            hang_forever("pformalsol")
        f.write("".join(rows))
        f.write("  %.14f     \n" % (-ew))
        f.close()
    out(" DETAIL FILE IS OVER. SUCCESSFUL!! ")


main()
'''

_NO_IMU_CODE = r'''
def open_imu(model, n, suf):
    return None


def write_imu(g, lam0, depth, fc):
    pass
'''

_IMU_CODE = r'''
def open_imu(model, n, suf):
    return open(os.path.join(model, "OUT_IMU." + n + ("_" + suf if suf else "")), "w")


def write_imu(g, lam0, depth, fc):
    p = [0.0, 0.5, 0.9, 1.0, 1.008]
    ncore = 4
    g.write("%s%5d%5d\n" % ("# rays NP-1, core rays NC =", len(p), ncore))
    g.write("# p  " + "".join("%16.8E" % v for v in p) + "\n")
    g.write("# K, lambda, I_cont(p_1..p_NP-1), I_line(p_1..p_NP-1)\n")
    for i in range(1, NROW + 1):
        lam = lam0 + (i - 81) * 0.5
        mus = [math.sqrt(max(0.0, 1.0 - (v / 1.008) ** 2)) for v in p]
        ic = [fc * (0.4 + 0.6 * mu) for mu in mus]
        il = [ic[j] * (1.0 - depth * math.exp(-((lam - lam0) / 1.5) ** 2)) for j in range(len(p))]
        g.write("%5d%12.4f" % (i, lam) + "".join("%14.6E" % v for v in ic + il) + "\n")
'''

_FAKE_INDAT = """FAKEMOD                                        CATALOG
T  T   0   100                                 OPTNEUPDATE,HE_ONE,ITSTART,ITMORE
0.                                             OPTMIXED
38230.0,     4.25000,    6.2100                TEFF, LOG G, RSTAR
120.,  0.6                                     RMAX, TMIN
1.0e-10,  0.1,  2500.00,   1.00000,  0.1       MDOT, VMIN(START), VINF, BETA, VDIV
0.100000,  2.00000                             YHE, IHE(START)
F T F T T                                      OPTMOD,OPTTLUCY,MEGAS,ACCEL,OPTCMF
10., 1.0, T  T                                 VTURB,METALLICITY,LINES,LINES_IN_MODEL
T F 1 2                                        ENATCOR, EXPANSION,SET_FIRST, SET_STEP
1., 0.1, 0.2                                   CLF, VCLSTART, VCLMAX
"""
"""An INDAT template in the layout of the M424 template (INDAT_M424test.DAT, with ITMORE = 100)."""

_FAKE_FORMAL = """:T VSINI
0.

:T LINES TO BE SOLVED, NUMBER OF COMPONENTS, LEVELS, LINE-NUMBER
:T AND STARK BROADENING OPTION
:T He I 4026.22 (2p3P-5d3D) blended with He II 4025.67 (n=4-13)
HEI4026   2  HE12P3 HE15D3 0  1   HE24 HE213 0  1
:T He II 4199.90 (n=4-11)
HEII4200  1  HE24   HE211 0  1
:T He I 4921.93 (2p1P-4d1D)
HEI4922   1  HE12P1 HE14D1 0  1
"""
"""A FORMAL_INPUT in the layout of the M424 line list (FORMAL_INPUT_He3)."""


def fake_indat_text():
    """The INDAT template text of the fake install (M424 layout)."""
    return _FAKE_INDAT


def fake_formal_text():
    """The FORMAL_INPUT text of the fake install (the three M424 lines)."""
    return _FAKE_FORMAL


def _script(python, body, config, imu_code=""):
    from .model import MODEL_FILES
    common = _COMMON.format(config=config, data_dirs=DATA_DIRS, hopf=HOPF_FILES)
    text = body.format(env=ENV_MODE, model_files=MODEL_FILES)
    return "#!{}\n# fake FASTWIND executable (ppmpy.synspec.fastwind.fake)\n{}\n{}\n{}".format(python, common,
                                                                                            imu_code, text)


def _write_exec(path, text):
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def set_fake_config(build_dir, **cfg):
    """
    Write the behaviour of a fake build (``<build>/fake_config.json``), replacing the previous one.

    Keywords: fail, crash, hang, cap, formal_fail, formal_hang, formal_partial, formal_partial_hang (lists of T_eff,
    numbers or '%.3f' strings), fail_after (int), converge_at (int), partial_rows (int), sleep (float s).
    """
    # PP 2026-10-02: new (M6)
    out = {}
    for k, v in cfg.items():
        if k in TEFF_LIST_KEYS:
            v = [x if isinstance(x, str) else "%.3f" % float(x) for x in v]
        out[k] = v
    path = os.path.join(build_dir, CONFIG_NAME)
    with open(path + ".tmp", "w") as f:
        json.dump(out, f)
    os.replace(path + ".tmp", path)
    return path


def install_fake(dest, imu=True, python=None, self_link=True, **config):
    """
    Write a fake FASTWIND install in ``dest``: ``inicalc/{DATA,...}`` (one small file each), the two Hopf tables, the
    ``inicalc/inicalc`` self link of the real install, a build ``v10.6_HHe`` (pnlte_A10HHe.eo,
    pformalsol_A10HHe.eo, ATOM_FILE, A10HHe.dat, fake_config.json) and, with ``imu``, ``v10.6_HHe_imu`` whose
    pformalsol also writes OUT_IMU.*; plus ``INDAT.template`` and ``FORMAL_INPUT`` in the M424 layout.

    Parameters
    ----------
    dest: str
    imu: bool
        Also write the intensity build.
    python: str, optional
        Interpreter of the executables' '#!' line (default ``sys.executable``).
    self_link: bool
        Create ``inicalc/inicalc -> inicalc`` (as the M424 install).
    **config:
        Initial behaviour (:func:`set_fake_config`), written to both builds.

    Returns
    -------
    FastwindInstall
        The standard build (``FastwindInstall(dest, FAKE_BUILD, formal_build=FAKE_IMU_BUILD)`` uses the intensity
        pformalsol).
    """
    # PP 2026-10-02: new (M6)
    python = python or sys.executable
    dest = os.path.abspath(dest)
    ini = os.path.join(dest, "inicalc")
    os.makedirs(ini, exist_ok=True)
    for d in DATA_DIRS:
        os.makedirs(os.path.join(ini, d), exist_ok=True)
        with open(os.path.join(ini, d, "README"), "w") as f:
            f.write("fake {} data\n".format(d))
    for h in HOPF_FILES:
        with open(os.path.join(ini, h), "w") as f:
            f.write("fake Hopf table {}\n".format(h))
    if self_link and not os.path.lexists(os.path.join(ini, "inicalc")):
        os.symlink(ini, os.path.join(ini, "inicalc"))
    builds = [(FAKE_BUILD, _NO_IMU_CODE)] + ([(FAKE_IMU_BUILD, _IMU_CODE)] if imu else [])
    for build, imu_code in builds:
        b = os.path.join(dest, build)
        os.makedirs(b, exist_ok=True)
        cfgpath = os.path.join(b, CONFIG_NAME)
        _write_exec(os.path.join(b, "pnlte_{}.eo".format(FAKE_TAG)), _script(python, _PNLTE, cfgpath))
        _write_exec(os.path.join(b, "pformalsol_{}.eo".format(FAKE_TAG)), _script(python, _FORMAL, cfgpath,
                                                                                 imu_code))
        with open(os.path.join(b, "ATOM_FILE"), "w") as f:
            f.write("{}.dat\nthom_new.dat\nLINES_CnewNOSi_new_coll.dat\n".format(FAKE_TAG))
        with open(os.path.join(b, FAKE_TAG + ".dat"), "w") as f:
            f.write("fake model atom\n")
        set_fake_config(b, **config)
    with open(os.path.join(dest, "INDAT.template"), "w") as f:
        f.write(_FAKE_INDAT)
    with open(os.path.join(dest, "FORMAL_INPUT"), "w") as f:
        f.write(_FAKE_FORMAL)
    return FastwindInstall(dest, FAKE_BUILD)


# ----------------------------------------------------------------------------------------------------------------
# self test (for the host python without pytest)
# ----------------------------------------------------------------------------------------------------------------
BLOCK_NUMPY = r"""
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('numpy', 'scipy', 'matplotlib'):
            raise ImportError('blocked: ' + name)
        return None
    find_module = None
sys.meta_path.insert(0, _Block())
"""
"""Code that makes numpy / scipy / matplotlib imports fail (prepended in :func:`check_stdlib_only`)."""

MODULES = ("ppmpy.synspec.fastwind", "ppmpy.synspec.fastwind.install", "ppmpy.synspec.fastwind.indat",
           "ppmpy.synspec.fastwind.formal", "ppmpy.synspec.fastwind.logs", "ppmpy.synspec.fastwind.model",
           "ppmpy.synspec.fastwind.fake", "ppmpy.synspec.fastwind.archive", "ppmpy.synspec.fastwind.batch",
           "ppmpy.synspec.fastwind.__main__")      # PP 2026-10-02: all 10 modules (review)


def check_stdlib_only(python=None, modules=MODULES):
    """
    Import every module of this subpackage in a fresh interpreter with numpy, scipy and matplotlib blocked.

    Returns
    -------
    (ok, output): (bool, str)
    """
    # PP 2026-10-02: new (M6)
    python = python or sys.executable
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    code = BLOCK_NUMPY + "sys.path.insert(0, {!r})\nimport importlib\n".format(root)
    code += "".join("importlib.import_module({!r})\n".format(m) for m in modules)
    code += "print('imported', len({!r}), 'modules without numpy; python', sys.version.split()[0])\n".format(modules)
    p = subprocess.run([python, "-c", code], capture_output=True, text=True)
    return p.returncode == 0, (p.stdout + p.stderr).strip()


AWK_EDIT = ('NR == 1 {printf "%-47sCATALOG\\n", n; next}\n'
            '     NR == 4 {sub(/^[^,]*,/, sprintf("%.3f,", t))} {print}')
"""The awk program of fw_sphere_point.sh:23-24, verbatim (run with -v n=<name> -v t=<teff>)."""


def awk_edit(template_path, name, teff_text, awk="awk"):
    """The legacy INDAT edit of one point: awk run as a subprocess (bytes)."""
    # PP 2026-10-02: ported from fw_sphere_point.sh:23-24 (the awk command itself)
    return subprocess.run([awk, "-v", "n=" + name, "-v", "t=" + teff_text, AWK_EDIT, template_path],
                          capture_output=True, check=True).stdout


def awk_edit_batch(template_path, pairs, awk="awk", chunk=500):
    """
    The legacy edit for many (name, teff text) pairs with few awk processes: the same program with NR -> FNR, the
    template given once per pair and the values as command-line assignments before it. Returns a list of bytes.
    """
    # PP 2026-10-02: new (M6); fw_sphere_point.sh:23-24 with NR -> FNR, for 10^4 values in few processes
    prog = 'FNR == 1 {printf "\\001"}\n' + AWK_EDIT.replace("NR ==", "FNR ==")     # POSIX awk (gawk, mawk)
    outs = []
    for i in range(0, len(pairs), chunk):
        args = []
        for n, t in pairs[i:i + chunk]:
            args += ["n=" + n, "t=" + t, template_path]
        r = subprocess.run([awk, prog] + args, capture_output=True, check=True).stdout
        parts = r.split(b"\x01")
        if parts[0] != b"":
            raise RuntimeError("unexpected awk output before the first template")
        outs.extend(parts[1:])
    return outs


def selftest(workdir=None, n_awk=2000, verbose=True):
    """
    Checks that need no numpy and no pytest (run with the host python): stdlib-only imports, the INDAT edit against
    awk (``n_awk`` random T_eff), suffix rules, and fake-FASTWIND runs (ok, deterministic failure, formal failure,
    timeout that kills the group but not a same-named decoy, rerun_formal with OUT_IMU). Raises AssertionError on a
    failure; returns a dict of timings.
    """
    # PP 2026-10-02: new (M6)
    import random
    import shutil
    import signal
    from . import formal as fo
    from . import indat as ind
    from . import model as mo
    t0 = time.time()
    rep = {}

    def say(*a):
        if verbose:
            print(*a)
            sys.stdout.flush()

    ok, msg = check_stdlib_only()
    assert ok, msg
    say("stdlib-only:", msg)
    own = workdir is None
    work = workdir or tempfile.mkdtemp(prefix="fwfake_")
    try:
        inst = install_fake(os.path.join(work, "fw"))
        tpl = os.path.join(inst.root, "INDAT.template")
        if shutil.which("awk"):
            rng = random.Random(7)
            vals = ["%.3f" % rng.uniform(30000, 45000) for _ in range(n_awk // 2)]
            vals += [repr(rng.uniform(30000, 45000)) for _ in range(n_awk - len(vals))]
            vals += ["37298.0625", "37298.9995", "38230", "1e4"]
            pairs = [("P%06d" % i, t) for i, t in enumerate(vals)]
            got = awk_edit_batch(tpl, pairs)
            text = open(tpl).read()
            bad = [p for p, g in zip(pairs, got) if ind.edit_like_awk(text, p[0], p[1]).encode() != g]
            assert len(got) == len(pairs) and not bad, bad[:3]
            assert awk_edit(tpl, *pairs[0]) == got[0]
            say("INDAT edit == awk for", len(pairs), "values")
        assert fo.formal_suffix((10, 0.1), 0) == "VTV010" and fo.formal_suffix(10) == "VT010"
        assert fo.formalsol_stdin("P000001") == b"P000001\n10 0.1\n0\n"
        st = inst.stage(os.path.join(work, "stage"))
        res = os.path.join(work, "res")
        r = mo.run_model(st, (1, "38230.000"), res, tpl, os.path.join(inst.root, "FORMAL_INPUT"), keep="model")
        assert r["status"] == "ok" and r["niter"] == 33, r
        assert open(os.path.join(res, "P000001", "meta.txt")).read().split()[:4] == ["1", "38230.000", "ok", "33"]
        set_fake_config(inst.build, fail=[37000.0], formal_fail=[37100.0], hang=[37200.0], formal_partial=[37300.0])
        r = mo.run_model(st, (2, 37000.0), res, tpl, os.path.join(inst.root, "FORMAL_INPUT"))
        assert r["status"] == "pnlte_failed" and r["niter"] == 19, r
        r = mo.run_model(st, (3, 37100.0), res, tpl, os.path.join(inst.root, "FORMAL_INPUT"))
        assert r["status"] == "formal_failed", r
        r = mo.run_model(st, (5, 37300.0), res, tpl, os.path.join(inst.root, "FORMAL_INPUT"))
        assert r["status"] == "formal_failed" and "formal_incomplete" in r["flags"], r
        decoy_dir = os.path.join(work, "decoy")
        os.makedirs(decoy_dir)
        env = dict(os.environ)
        env[ENV_MODE] = "hang"
        decoy = subprocess.Popen([inst.pnlte], cwd=decoy_dir, env=env, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            t = time.time()
            r = mo.run_model(st, (4, 37200.0), res, tpl, os.path.join(inst.root, "FORMAL_INPUT"), pnlte_timeout=3,
                             grace=1, keep_run=True)
            assert r["status"] == "pnlte_timeout" and time.time() - t < 15, r
            pid = int(open(os.path.join(st.root, "P000004", "hang_child.pid")).read())
            time.sleep(0.2)
            assert not _alive(pid), "helper child of the timed-out pnlte survived"
            assert decoy.poll() is None, "decoy was killed"
        finally:
            os.killpg(decoy.pid, signal.SIGKILL)
            decoy.wait()
        inst2 = FastwindInstall(inst.root, FAKE_BUILD, formal_build=FAKE_IMU_BUILD)
        assert inst2.has_imu_patch() and not inst.has_imu_patch()
        st2 = inst2.stage(os.path.join(work, "stage_imu"))
        rr = mo.rerun_formal(st2, os.path.join(res, "P000001"), os.path.join(inst.root, "FORMAL_INPUT"))
        assert rr["status"] == "ok", rr
        say("fake runs ok")
    finally:
        if own:
            shutil.rmtree(work, ignore_errors=True)
    rep["seconds"] = time.time() - t0
    say("selftest passed in %.1f s" % rep["seconds"])
    return rep


def _alive(pid):
    """True if ``pid`` runs and is not a zombie."""
    try:
        with open("/proc/{}/stat".format(pid)) as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False
