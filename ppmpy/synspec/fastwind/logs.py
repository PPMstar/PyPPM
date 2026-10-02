"""
FASTWIND logs and convergence files: pnlte.log, CONVERG, MAXTCORR.dat; the convergence verdict with nlte.f90's
own criteria, and the classification of a finished model (standard library only).

What 'finished' means
---------------------
pnlte ends every regular run with ``STOP '! ESTO ES EL ACABOSE !'`` and exit status 0, converged or not; it also
exits 0 on its error STOPs (e.g. ``' error in ne -- nlteopt'``). The legacy status (fw_sphere_point.sh) is
therefore: 'ok' when pnlte.log contains 'ESTO ES EL ACABOSE' and pformalsol wrote every OUT file,
'formal_failed' when it did not, 'pnlte_timeout' when pnlte was killed at the time limit, else 'pnlte_failed'.
Here "wrote every OUT file" is stricter than the legacy ``ls OUT.* | wc -l``: pformalsol must exit 0 within its time
limit and every expected file must be complete (``formal.out_problem``), because pformalsol opens each OUT file
before computing it and a killed or crashed pformalsol leaves empty or truncated files behind.

Convergence (nlte.f90 v10.6.4.1, 1650-1680, 2240-2274, 2424-2434, 2482-2502)
---------------------------------------------------------------------------
* With ENATCOR = T the temperature is corrected every SET_STEP iterations from ITSTART + SET_FIRST; each correction
  appends 'NCOR EMAXTC IIT' to MAXTCORR.dat. The temperature has converged at the first correction with
  EMAXTC < 3e-3 and IIT > 21 (no convergence inside the Sobolev cycle); from then on ENATCOR = F.
* Every iteration appends 'IIT EMAX MEANERR' to CONVERG. When ENATCOR = F (and no temperature correction in that
  iteration), EMAX < 3e-3 or MEANERR <= -4.5 marks the model converged; pnlte then runs one more iteration (the
  complete output) and stops. The correction iteration in which the temperature converged is not checked.
* Without convergence pnlte stops after iteration ITSTART + ITMORE, again with 'ESTO ES EL ACABOSE'.

:func:`convergence` applies these rules to CONVERG / MAXTCORR.dat: ``converged`` is True when a row before the last
meets the criterion after the temperature converged (the code's own trigger). The legacy count ``niter`` (lines with
'ITERATION NO' in pnlte.log) is the number of NLTE iterations + 2 for a fresh start (the hydro line 'FINAL ITERATION
NO . 7 FOR HYDRO MODEL' and 'ITERATION NO 0').

PP 2026-10-02: new (M6); niter / T(TAUROSS=2/3) / status from fw_sphere_point.sh:27-43; convergence rules from
nlte.f90 and fw_itmore_check.py:118-123 (project stellar-atmosphere-KU-Leuven).
"""
import math
import os

ACABOSE = b"ESTO ES EL ACABOSE"
ITERATION_MARK = b"ITERATION NO"
TTAU_MARK = b"T(TAUROSS=2/3)"
EMAX_CONV = 3.0e-3
"""EMAX below which the NLTE iteration has converged (nlte.f90:2501)."""
MEANERR_CONV = -4.5
"""log MEANERR at or below which the NLTE iteration has converged (nlte.f90:2501, 'JO Sept 2021')."""
EMAXTC_CONV = 3.0e-3
"""Largest relative temperature correction below which the temperature has converged (nlte.f90:2428)."""
TCONV_MIN_IT = 21
"""The temperature convergence counts only for IIT > 21 (nlte.f90:2428, 'prevent convergence in Sobo cycle')."""
NITER_EXTRA = 2
"""'ITERATION NO' lines besides the NLTE iterations for a fresh start (as fwresults.NITER_EXTRA)."""

ERROR_PATTERNS = (b"error in", b"forrtl", b"severe", b"Segmentation", b"SIGSEGV", b"NOT FOUND", b"NOT CONVERGED IN",
                  b"STOP", b"Killed", b"Abort", b"not allowed", b"NOT ALLOWED", b"WRONG", b"PROBLEMS WITH LEVEL")
"""Byte patterns of lines reported as errors by :func:`parse_pnlte_log` (case-sensitive substrings; the final
'ESTO ES EL ACABOSE' line is not an error)."""

STATUS_WORDS = ("ok", "formal_failed", "pnlte_failed", "pnlte_timeout")
"""The legacy status words of meta.txt."""


def _lines(src):
    """Byte lines of a file name or of bytes / str content."""
    if isinstance(src, (bytes, bytearray)):
        return bytes(src).split(b"\n")
    if isinstance(src, str) and "\n" in src:
        return src.encode("latin-1").split(b"\n")
    with open(src, "rb") as f:
        return f.read().split(b"\n")


def parse_pnlte_log(src, tail=40):
    """
    Digest of a pnlte log (stdout + stderr of pnlte).

    Parameters
    ----------
    src: str or bytes
        File name, or the log content.
    tail: int
        Number of final lines returned in ``tail`` (fw_sphere_point.sh keeps 40 in pnlte_tail.log).

    Returns
    -------
    dict
        acabose (bool: 'ESTO ES EL ACABOSE' present), niter (lines containing 'ITERATION NO', as ``grep -c``),
        T_tau23 (str: the last field of the last 'T(TAUROSS=2/3)' line, verbatim as awk's ``$NF``; None if
        absent), T_tau23_value (float, NaN if absent), last_iteration (int, the largest 'ITERATION NO <n>'; None),
        temp_converged (bool: 'TEMPERATURE CONVERGED' printed), n_tcorr (int from 'TOTAL NUMBER OF APPLIED T
        CORRECTION(S)'; None), emaxtc_last (float from 'Maximum relative correction in the last TC'; None),
        all_levels_ok (bool), cpu_time (float, s; None), errors (list of str: lines matching
        :data:`ERROR_PATTERNS`, in order, at most 50), last_line (str: the last non-blank line), tail (list of str).
    """
    # PP 2026-10-02: ported from fw_sphere_point.sh:27-43 (grep ACABOSE, grep -c 'ITERATION NO', T(TAUROSS=2/3)
    # | tail -1 | awk '{print $NF}'); the other items are new
    lines = _lines(src)
    if lines and lines[-1] == b"":
        lines = lines[:-1]                  # the split after a final newline (grep counts lines, not separators)
    out = dict(acabose=False, niter=0, T_tau23=None, T_tau23_value=math.nan, last_iteration=None,
               temp_converged=False, n_tcorr=None, emaxtc_last=None, all_levels_ok=False, cpu_time=None,
               errors=[], last_line="", tail=[])
    for ln in lines:
        if ITERATION_MARK in ln:
            out["niter"] += 1
            tok = ln.split(ITERATION_MARK, 1)[1].replace(b"+", b" ").split()
            if tok and tok[0].isdigit():
                n = int(tok[0])
                if out["last_iteration"] is None or n > out["last_iteration"]:
                    out["last_iteration"] = n
            continue
        if ACABOSE in ln:
            out["acabose"] = True
            continue
        if TTAU_MARK in ln:
            tok = ln.split()
            if tok:
                out["T_tau23"] = tok[-1].decode("latin-1")
            continue
        if b"TEMPERATURE CONVERGED" in ln:
            out["temp_converged"] = True
        elif b"TOTAL NUMBER OF APPLIED T CORRECTION" in ln:
            tok = ln.split()
            try:
                out["n_tcorr"] = int(tok[-1])
            except (IndexError, ValueError):
                pass
        elif b"Maximum relative correction in the last TC" in ln:
            try:
                out["emaxtc_last"] = float(ln.split()[-1])
            except (IndexError, ValueError):
                pass
        elif b"ALL LEVELS OK" in ln:
            out["all_levels_ok"] = True
        elif b"CPU time:" in ln:
            try:
                out["cpu_time"] = float(ln.split()[-1])
            except (IndexError, ValueError):
                pass
        if len(out["errors"]) < 50 and any(p in ln for p in ERROR_PATTERNS):
            out["errors"].append(ln.decode("latin-1").strip())
    if out["T_tau23"] is not None:
        try:
            out["T_tau23_value"] = float(out["T_tau23"])
        except ValueError:
            pass
    for ln in reversed(lines):
        if ln.strip():
            out["last_line"] = ln.decode("latin-1").strip()
            break
    out["tail"] = [ln.decode("latin-1") for ln in lines[-tail:]] if tail else []
    return out


def tail_bytes(path, n=40):
    """The last ``n`` lines of a file as bytes, like ``tail -n`` (a final line without newline counts)."""
    # PP 2026-10-02: ported from fw_sphere_point.sh:51 (tail -40 pnlte.log > pnlte_tail.log)
    with open(path, "rb") as f:
        data = f.read()
    if not data:
        return b""
    body = data[:-1] if data.endswith(b"\n") else data
    parts = body.split(b"\n")
    return b"\n".join(parts[-n:]) + (b"\n" if data.endswith(b"\n") else b"")


def _float(tok):
    try:
        return float(tok.replace(b"D", b"E").replace(b"d", b"e"))
    except ValueError:
        return math.nan


def parse_converg(path):
    """
    Rows of CONVERG (``WRITE (8, FMT=*) IIT, EMAX, ' ', MEANERR``, nlte.f90:2482): list of (iit int, emax float,
    log meanerr float). Lines that are not three numbers are skipped; unreadable numbers give NaN.
    """
    # PP 2026-10-02: new (M6); fw_itmore_check.py:118 read it with np.loadtxt
    rows = []
    for ln in _lines(path):
        tok = ln.split()
        if len(tok) != 3:
            continue
        try:
            iit = int(tok[0])
        except ValueError:
            continue
        rows.append((iit, _float(tok[1]), _float(tok[2])))
    return rows


def parse_maxtcorr(path):
    """
    Rows of MAXTCORR.dat (``'(1X,I3,1X,G12.5,1X,I3)') NCOR, EMAXTC, IIT``, nlte.f90:2427): list of (ncor int,
    emaxtc float, iit int or None). An I3 overflow ('***', IIT > 999) gives None.
    """
    # PP 2026-10-02: new (M6)
    rows = []
    for ln in _lines(path):
        tok = ln.split()
        if len(tok) < 2:
            continue
        try:
            ncor = int(tok[0])
        except ValueError:
            continue
        iit = None
        if len(tok) >= 3:
            try:
                iit = int(tok[2])
            except ValueError:
                iit = None
        rows.append((ncor, _float(tok[1]), iit))
    return rows


def convergence(model_dir, itstart=0, itmore=None, enatcor=True):
    """
    The convergence verdict of one pnlte model from its CONVERG and MAXTCORR.dat (nlte.f90's criteria, see the module
    notes).

    Parameters
    ----------
    model_dir: str
        The model (catalogue) directory, ``<run>/<MODNAM>``, or a directory holding copies of the two files.
    itstart, itmore: int
        ITSTART and ITMORE of the INDAT (the cap is ITSTART + ITMORE); ``itmore`` None: cap unknown.
    enatcor: bool
        ENATCOR of the INDAT (temperature correction on). With False the temperature criterion is skipped.

    Returns
    -------
    dict
        converged (bool), converged_it (the iteration whose EMAX / MEANERR triggered the stop; None),
        temp_converged_it (first MAXTCORR row with EMAXTC < 3e-3 and IIT > 21; None; 0 when ``enatcor`` is False),
        n_iter (last IIT in CONVERG; 0 if empty), emax_last, meanerr_last (last row; NaN if empty), emaxtc_last
        (last MAXTCORR row; NaN), n_tcorr (MAXTCORR rows), cap (ITSTART + ITMORE or None), at_cap (n_iter == cap),
        criterion_last (the last row meets the criterion: e.g. a model that converged exactly in the capped
        iteration, which pnlte does not test), consistent (False when the rows contradict the rules: a converged
        model whose last iteration is not converged_it + 1, or a model that stopped early without converging),
        has_converg, has_maxtcorr (files found).

    Notes
    -----
    Checked against the legacy verdict of fw_itmore_check.py (temperature converged and the last row meeting the
    criterion) on the 48 ITMORE-check models, and on the M424 reference model (T converged at 41, converged at 58,
    59 iterations).
    """
    # PP 2026-10-02: new (M6); rules of nlte.f90:2428-2434, 2482-2502 (see the module notes)
    cpath = os.path.join(model_dir, "CONVERG")
    mpath = os.path.join(model_dir, "MAXTCORR.dat")
    conv = parse_converg(cpath) if os.path.isfile(cpath) else []
    mtc = parse_maxtcorr(mpath) if os.path.isfile(mpath) else []
    cap = None if itmore is None else int(itstart) + int(itmore)
    if not enatcor:
        tconv = 0
    else:
        tconv = None
        for ncor, emaxtc, iit in mtc:
            if iit is not None and emaxtc < EMAXTC_CONV and iit > TCONV_MIN_IT:
                tconv = iit
                break

    def crit(row):
        return row[1] < EMAX_CONV or row[2] <= MEANERR_CONV

    conv_it = None
    if tconv is not None:
        for row in conv[:-1]:
            if row[0] > tconv and crit(row):
                conv_it = row[0]
                break
    n_iter = conv[-1][0] if conv else 0
    consistent = True
    if conv_it is not None and n_iter != conv_it + 1:
        consistent = False
    if conv_it is None and cap is not None and conv and n_iter != cap:
        consistent = False
    return dict(converged=conv_it is not None, converged_it=conv_it, temp_converged_it=tconv, n_iter=n_iter,
                emax_last=conv[-1][1] if conv else math.nan, meanerr_last=conv[-1][2] if conv else math.nan,
                emaxtc_last=mtc[-1][1] if mtc else math.nan, n_tcorr=len(mtc), cap=cap,
                at_cap=cap is not None and n_iter == cap, criterion_last=bool(conv) and crit(conv[-1]),
                consistent=consistent, has_converg=os.path.isfile(cpath), has_maxtcorr=os.path.isfile(mpath))


def classify(log, formal_ok=None, timed_out=False, returncode=0, conv=None, formal_timed_out=False,
             formal_returncode=None, formal_problems=None):
    """
    The legacy status word and extra flags of one model.

    Parameters
    ----------
    log: dict
        :func:`parse_pnlte_log` of pnlte.log.
    formal_ok: bool or None
        pformalsol succeeded: exit status 0, no time limit, every expected output complete (None: not run).
    timed_out: bool
        pnlte was killed at its time limit.
    returncode: int
        pnlte's exit status (negative: killed by that signal).
    conv: dict, optional
        :func:`convergence` of the model directory.
    formal_timed_out: bool
        pformalsol was killed at its time limit.
    formal_returncode: int, optional
        pformalsol's exit status (None: not run).
    formal_problems: dict, optional
        Output name -> why it is not complete (``formal.out_problem``), for the incomplete ones.

    Returns
    -------
    status: str
        'ok', 'formal_failed', 'pnlte_timeout' or 'pnlte_failed', as fw_sphere_point.sh: ACABOSE first (then
        pformalsol decides), then the time limit, else failed.
    flags: tuple of str
        Sorted, from: 'not_converged' (finished at the cap without convergence), 'temp_not_converged',
        'converged_at_cap' (criterion met only in the capped iteration), 'inconsistent_converg', 'no_converg',
        'error' (error lines in the log), 'levels_not_ok' (no 'ALL LEVELS OK'), 'nonzero_exit', 'signal',
        'timeout', 'formal_not_run', 'formal_timeout', 'formal_nonzero_exit', 'formal_signal', 'formal_missing'
        (an expected output absent), 'formal_incomplete' (present but empty or truncated).
    """
    # PP 2026-10-02: ported from fw_sphere_point.sh:29-37 (status); flags new (M6)
    flags = set()
    if log["acabose"]:
        if formal_ok is None:
            status = "formal_failed"
            flags.add("formal_not_run")
        else:
            status = "ok" if formal_ok else "formal_failed"
    elif timed_out:
        status = "pnlte_timeout"
    else:
        status = "pnlte_failed"
    if timed_out:
        flags.add("timeout")
    if returncode is not None and returncode < 0:
        flags.add("signal")
    elif returncode:
        flags.add("nonzero_exit")
    if formal_timed_out:
        flags.add("formal_timeout")
    if formal_returncode is not None and formal_returncode < 0:
        flags.add("formal_signal")
    elif formal_returncode:
        flags.add("formal_nonzero_exit")
    for why in (formal_problems or {}).values():
        flags.add("formal_missing" if why == "missing" else "formal_incomplete")
    if log.get("errors"):
        flags.add("error")
    if log["acabose"] and not log.get("all_levels_ok"):
        flags.add("levels_not_ok")
    if conv is not None and log["acabose"]:
        if not conv["has_converg"]:
            flags.add("no_converg")
        else:
            if not conv["converged"]:
                flags.add("not_converged")
                if conv["criterion_last"]:
                    flags.add("converged_at_cap")
            if conv["temp_converged_it"] is None:
                flags.add("temp_not_converged")
            if not conv["consistent"]:
                flags.add("inconsistent_converg")
    return status, tuple(sorted(flags))
