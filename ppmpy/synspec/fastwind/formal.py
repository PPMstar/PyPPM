"""
pformalsol's inputs and output names: the line list FORMAL_INPUT, the answers on standard input, and the
OUT / OUT_IMU file names (standard library only).

FORMAL_INPUT is read by FASTWIND's free-field reader (ffr.f90, called from formalsol.f90:384-465): lines of at
most 80 columns; a line starting with ':' is a control line (':T ...' in the templates, i.e. a comment); the
comment delimiter is ':' (``FFRCDC(':T')``), so text from a ':' to the next ':' or the end of the line is skipped;
items are separated by blanks. The items are: VSINI, then per line a name, the number of components and, per
component, lower level, upper level, transition number (0 for H / He) and Stark option. Line names are taken from
the file, never assumed.

pformalsol asks on standard input for the catalogue (model) name, the turbulence ('10 0.1': v_turb,min = 10 km/s,
v_turb,max = 0.1 v_inf) and IESCAT (0 / 1); :func:`formalsol_stdin` writes these answers as the legacy scripts
(``printf "%s\\n10 0.1\\n0\\n"``).

Output names (formalsol.f90:550-582): ``OUT.<line>`` when v_turb,min = 0 without electron scattering,
``OUT.<line>_ESC`` with it; otherwise ``OUT.<line>_<suffix>`` with suffix ``VT<iii>`` (constant v_turb),
``VTV<iii>`` (depth-dependent) or ``ESC_VT<iii>``, iii = int(v_turb,min) in three digits (VTV010 for '10 0.1').
The intensity patch (formalsol_imu.patch) writes ``OUT_IMU.`` + the same rest.

Complete outputs
----------------
formalsol.f90 opens OUT.<line>_<suffix> (and the patch OUT_IMU.<...>) *before* the formal integral of that line
(~557-587, CALL FORMAL ~693) and closes it afterwards (~701), writing the tables at the end of FORMAL: a pformalsol
killed at a time limit, crashed or signalled leaves an empty or truncated file under the final name. Existence is
therefore no test of success; :func:`out_problem` checks that a file is complete (OUT: the table of
:data:`OUT_NROW` six-column rows and the one-number EW trailer; OUT_IMU: the header and :data:`OUT_NROW` rows of
2 + 2 x nray columns).

PP 2026-10-02: new (M6); rules from formalsol.f90 (v10.6.4.1) and ffr.f90; stdin answers from fastwind_run.sh:51 /
fw_sphere_point.sh:31 (project stellar-atmosphere-KU-Leuven).
"""
import re

from .install import atomic_write

FFR_COLUMNS = 80
"""ffr.f90 reads FORMAL_INPUT lines into CHARACTER*80: columns beyond 80 are ignored by pformalsol."""

LINE_NAME_MAX = 19
"""formalsol.f90 keeps line names in CHARACTER*20 and cuts the file name at the first blank: at most 19 characters."""

MODEL_NAME_MAX = 60
"""``CHARACTER FILE*60`` in formalsol.f90: the catalogue name read from stdin."""

VTURB_M424 = (10, 0.1)
"""The M424 turbulence answer '10 0.1' (10 km/s, rising to 0.1 v_inf): suffix VTV010."""

OUT_NROW = 161
"""Rows of every OUT / OUT_IMU table of the M424 lines (NFOBS; ``fwresults.NROW_M424``)."""

OUT_NCOL = 6
"""Columns of an OUT table row: index, x, lambda, F_cont, F/F_cont, F_rot (formalsol.f90 FORMAT 9000)."""

_ITEM_SEP = re.compile(r"[ \t\r,]+")


class FormalLine:
    """
    One line of FORMAL_INPUT: name and components.

    Attributes
    ----------
    name: str
        Line name (output files OUT.<name>_<suffix>).
    components: list of tuple
        (lower level, upper level, transition number, Stark option) per component; levels are str, the two
        numbers int.
    """

    # PP 2026-10-02: new (M6)
    def __init__(self, name, components):
        self.name = str(name)
        self.components = [(str(a), str(b), int(c), int(d)) for a, b, c, d in components]

    def __eq__(self, other):
        return isinstance(other, FormalLine) and (self.name, self.components) == (other.name, other.components)

    def __repr__(self):
        return "FormalLine({!r}, {!r})".format(self.name, self.components)


def _ffr_items(text):
    """
    The items of a free-field file as pformalsol reads them (see the module notes). Records end at '\n' only (a
    '\r' before it belongs to the ending); items are separated by blanks, tabs and commas (other control bytes such
    as \x0c or \x85 in a comment neither end a record nor separate items).
    """
    # PP 2026-10-02: reviewer: str.splitlines also split at \x0b, \x0c, \x1c-\x1e, \x85
    items = []
    for raw in text.split("\n"):
        if raw.endswith("\r"):
            raw = raw[:-1]
        line = raw[:FFR_COLUMNS]
        if line.startswith(":"):
            continue
        parts = line.split(":")
        kept = parts[0::2]                  # text outside ':' ... ':' comments
        for p in kept:
            items.extend(t for t in _ITEM_SEP.split(p) if t)
    return items


def _num(tok, what):
    try:
        return float(tok.replace("D", "E").replace("d", "e"))
    except ValueError:
        raise ValueError("FORMAL_INPUT: {} expected, got {!r}".format(what, tok)) from None


def _int(tok, what):
    v = _num(tok, what)
    if v != int(v):
        raise ValueError("FORMAL_INPUT: integer {} expected, got {!r}".format(what, tok))
    return int(v)


class FormalInput:
    """
    The line list of pformalsol (FORMAL_INPUT).

    Parameters
    ----------
    vsini: float
        Projected rotation velocity [km/s] (M424: 0; rotation is applied later).
    lines: sequence of FormalLine or (name, components) pairs
    text: str, optional
        The original file text; kept and written back unchanged by :meth:`to_text` as long as it parses to the
        same content.
    """

    # PP 2026-10-02: new (M6)
    def __init__(self, vsini, lines, text=None):
        self.vsini = float(vsini)
        self.lines = [ln if isinstance(ln, FormalLine) else FormalLine(*ln) for ln in lines]
        self._text = text
        for ln in self.lines:
            if not ln.name or len(ln.name) > LINE_NAME_MAX or any(c.isspace() for c in ln.name):
                raise ValueError("line name {!r}: 1-{} characters without blanks".format(ln.name, LINE_NAME_MAX))
            if not ln.components:
                raise ValueError("line {} has no components".format(ln.name))
            if len(ln.components) > 10:
                raise ValueError("line {}: more than 10 components (pformalsol: 'TOO MANY COMPONENTS')".format(
                    ln.name))
        names = self.names
        if len(set(names)) != len(names):
            raise ValueError("duplicate line names: {}".format(names))

    @property
    def names(self):
        """Line names in file order."""
        return [ln.name for ln in self.lines]

    @classmethod
    def from_text(cls, text):
        """Parse FORMAL_INPUT text."""
        if isinstance(text, (bytes, bytearray)):
            text = bytes(text).decode("latin-1")
        it = _ffr_items(text)
        if not it:
            raise ValueError("FORMAL_INPUT: no VSINI")
        vsini = _num(it[0], "VSINI")
        pos, lines = 1, []
        while pos < len(it):
            name = it[pos]
            if pos + 1 >= len(it):
                raise ValueError("FORMAL_INPUT: line {} without the number of components".format(name))
            nco = _int(it[pos + 1], "number of components of " + name)
            comp = it[pos + 2:pos + 2 + 4 * nco]
            if len(comp) != 4 * nco:
                raise ValueError("FORMAL_INPUT: line {}: {} components need {} items, found {}".format(
                    name, nco, 4 * nco, len(comp)))
            lines.append(FormalLine(name, [(comp[4 * i], comp[4 * i + 1], _int(comp[4 * i + 2], "transition no."),
                                            _int(comp[4 * i + 3], "Stark option")) for i in range(nco)]))
            pos += 2 + 4 * nco
        return cls(vsini, lines, text=text)

    @classmethod
    def read(cls, path):
        """Read a FORMAL_INPUT file."""
        with open(path, "rb") as f:
            return cls.from_text(f.read())

    def _same(self, other):
        return self.vsini == other.vsini and self.lines == other.lines

    def to_text(self):
        """
        The file text: the original text when there is one and it still describes this object; otherwise a
        generated file in the template layout (':T' comment lines, VSINI, one line per profile).
        """
        if self._text is not None:
            try:
                if self._same(FormalInput.from_text(self._text)):
                    return self._text
            except ValueError:
                pass
        out = [":T VSINI", "{!r}".format(self.vsini), "",
               ":T LINES TO BE SOLVED, NUMBER OF COMPONENTS, LEVELS, LINE-NUMBER",
               ":T AND STARK BROADENING OPTION"]
        for ln in self.lines:
            row = "{:<9s} {:2d}".format(ln.name, len(ln.components))
            for a, b, c, d in ln.components:
                row += "  {} {} {} {}".format(a, b, c, d)
            out.append(row)
        for row in out:
            if len(row) > FFR_COLUMNS:
                raise ValueError("FORMAL_INPUT line longer than {} columns: {!r}".format(FFR_COLUMNS, row))
        return "\n".join(out) + "\n"

    def write(self, path):
        """Write the file atomically (unique temporary name, then rename; safe from many threads); returns ``path``."""
        return atomic_write(path, self.to_text().encode("latin-1"))

    def subset(self, names):
        """A FormalInput with only the given lines (in the given order)."""
        by = {ln.name: ln for ln in self.lines}
        return FormalInput(self.vsini, [by[n] for n in names])

    def __eq__(self, other):
        return isinstance(other, FormalInput) and self._same(other)

    def __repr__(self):
        return "FormalInput(vsini={!r}, lines={!r})".format(self.vsini, self.names)


def parse_vturb(vturb):
    """
    (vtmin, vtmax) as pformalsol reads its turbulence answer: a number (constant), a pair, or the answer text
    ('10 0.1'; one value means vtmax = vtmin, formalsol.f90:333-341).
    """
    # PP 2026-10-02: ported from formalsol.f90:333-341 (VTURB_STR, one or two values)
    if isinstance(vturb, str):
        tok = vturb.replace(",", " ").split()
        if not tok:
            raise ValueError("empty turbulence answer")
        vals = [float(t.replace("D", "E").replace("d", "e")) for t in tok[:2]]
    elif isinstance(vturb, (tuple, list)):
        vals = [float(v) for v in vturb]
    else:
        vals = [float(vturb)]
    if not 1 <= len(vals) <= 2:
        raise ValueError("turbulence: one or two values, got {!r}".format(vturb))
    return (vals[0], vals[1] if len(vals) == 2 else vals[0])


def formal_suffix(vturb=VTURB_M424, iescat=0):
    """
    The suffix of pformalsol's output files for a turbulence answer and IESCAT (formalsol.f90:550-582).

    Parameters
    ----------
    vturb: float, pair or str
        v_turb,min [km/s] alone (constant), (v_turb,min, v_turb,max), or the answer text ('10 0.1').
    iescat: int
        0 (no electron scattering) or 1.

    Returns
    -------
    str
        '' (no suffix: ``OUT.<line>``), 'ESC', 'VT<iii>', 'VTV<iii>' or 'ESC_VT<iii>' (iii = int(v_turb,min)).

    Raises
    ------
    ValueError
        Combinations pformalsol stops on: v_turb,min = 0 with v_turb,max != 0; IESCAT = 1 with depth-dependent
        turbulence; IESCAT not 0 / 1; v_turb >= 1000 km/s; negative v_turb.
    """
    # PP 2026-10-02: ported from formalsol.f90:340-349 (checks) and 550-582 (names)
    vtmi, vtma = parse_vturb(vturb)
    if iescat not in (0, 1):
        raise ValueError("IESCAT must be 0 or 1 (pformalsol: 'ERROR IN ESCAT OPTION'), got {!r}".format(iescat))
    if vtmi == 0.0 and vtma != 0.0:
        raise ValueError("VTURBMIN = 0 AND VTURBMAX NE 0! NOT ALLOWED (pformalsol stops)")
    if iescat == 1 and vtmi != vtma:
        raise ValueError("DEPTH DEPENDENT TURBULENCE ONLY FOR NO ESCAT (pformalsol stops)")
    if vtmi < 0 or vtma < 0:
        raise ValueError("negative turbulence {!r}".format(vturb))
    vturb1 = (vtmi * 1.0e5) * 1.0e-5          # VTURB = VTMI*1.D5 (cm/s); VTURB1 = VTURB*1.D-5
    if vturb1 == 0.0:
        return "ESC" if iescat == 1 else ""
    iturb = int(vturb1)
    if iturb >= 1000:
        raise ValueError("VTURB > 1000 KM/S (pformalsol stops)")
    if iescat == 1:
        return "ESC_VT{:03d}".format(iturb)
    return ("VT{:03d}" if vtmi == vtma else "VTV{:03d}").format(iturb)


def _fmt_answer(v):
    if isinstance(v, int) and not isinstance(v, bool):
        return str(v)
    return repr(float(v))


def vturb_answer(vturb=VTURB_M424):
    """The turbulence answer text: a str verbatim; numbers as written by Python (10 -> '10', 0.1 -> '0.1')."""
    # PP 2026-10-02: ported from fw_sphere_point.sh:31 (the answer "10 0.1")
    if isinstance(vturb, str):
        return vturb
    if isinstance(vturb, (tuple, list)):
        return " ".join(_fmt_answer(v) for v in vturb)
    return _fmt_answer(vturb)


def formalsol_stdin(model, vturb=VTURB_M424, iescat=0):
    """
    The standard input of pformalsol: model name, turbulence answer, IESCAT, one per line (bytes).

    ``formalsol_stdin('P000001') == b'P000001\\n10 0.1\\n0\\n'`` (fw_sphere_point.sh: ``printf "%s\\n10 0.1\\n0\\n"``).
    """
    # PP 2026-10-02: ported from fw_sphere_point.sh:31 / fastwind_run.sh:51
    model = str(model)
    if not model or len(model) > MODEL_NAME_MAX or "\n" in model:
        raise ValueError("model name {!r}: 1-{} characters, one line".format(model, MODEL_NAME_MAX))
    formal_suffix(vturb, iescat)                     # refuse what pformalsol would stop on
    return "{}\n{}\n{}\n".format(model, vturb_answer(vturb), int(iescat)).encode()


def out_name(line, suffix, kind="OUT"):
    """
    Output file name of one line: ``<kind>.<line>`` for an empty suffix, else ``<kind>.<line>_<suffix>``.

    Parameters
    ----------
    line: str
        Line name (from FORMAL_INPUT).
    suffix: str
        :func:`formal_suffix`.
    kind: str
        'OUT' (flux profile) or 'OUT_IMU' (intensities, patched pformalsol).
    """
    # PP 2026-10-02: ported from formalsol.f90:563-581 and formalsol_imu.patch (OUT_IMU)
    if kind not in ("OUT", "OUT_IMU"):
        raise ValueError("kind must be 'OUT' or 'OUT_IMU', got {!r}".format(kind))
    return "{}.{}".format(kind, line) if not suffix else "{}.{}_{}".format(kind, line, suffix)


def out_names(formal, vturb=VTURB_M424, iescat=0, kind="OUT"):
    """Output file names of every line of ``formal`` (FormalInput, path or names), in file order."""
    # PP 2026-10-02: new (M6)
    if isinstance(formal, str):
        formal = FormalInput.read(formal)
    names = formal.names if isinstance(formal, FormalInput) else list(formal)
    suf = formal_suffix(vturb, iescat)
    return [out_name(n, suf, kind) for n in names]


def _is_number(tok):
    try:
        float(tok.replace(b"D", b"E").replace(b"d", b"e"))
        return True
    except ValueError:
        return False


def out_problem(path, kind="OUT", nrow=OUT_NROW):
    """
    Why an output file of pformalsol is not complete, or None when it is.

    Parameters
    ----------
    path: str
        ``OUT.<line>_<suffix>`` or ``OUT_IMU.<line>_<suffix>``.
    kind: str
        'OUT' or 'OUT_IMU'.
    nrow: int or None
        Rows the table must have at least (:data:`OUT_NROW`); None: at least one.

    Returns
    -------
    str or None
        'missing', 'empty', 'no final newline', 'N table rows < nrow', 'no EW trailer' (OUT), 'bad header' /
        'last row has N columns, expected M' (OUT_IMU); None for a complete file.

    Notes
    -----
    OUT: the leading lines with six columns are the table (as ``fwresults.read_out``), followed by the trailer line
    with one number (``WRITE (2,*) AEQUIT``, the last thing formalsol.f90 writes before closing the file).
    OUT_IMU: '#' header lines (the first one ends with ``= <nray> <ncore>``) and the rows; a cheap line count plus
    the column count of the last row (a killed writer leaves at most one partial line, the last).
    """
    # PP 2026-10-02: new (M6); reviewer: a pformalsol killed at its time limit left empty / truncated OUT files that
    # passed the legacy existence test
    if kind not in ("OUT", "OUT_IMU"):
        raise ValueError("kind must be 'OUT' or 'OUT_IMU', got {!r}".format(kind))
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return "missing"
    except IsADirectoryError:
        return "not a file"
    if not data.strip():
        return "empty"
    if not data.endswith(b"\n"):
        return "no final newline"
    lines = data[:-1].split(b"\n")
    need = 1 if nrow is None else int(nrow)
    if kind == "OUT":
        n = 0
        for ln in lines:
            if len(ln.split()) != OUT_NCOL:
                break
            n += 1
        if n < need:
            return "{} table rows < {}".format(n, need)
        rest = [ln.split() for ln in lines[n:] if ln.strip()]
        if not rest or len(rest[0]) != 1 or not _is_number(rest[0][0]):
            return "no EW trailer"
        return None
    head = [ln for ln in lines if ln.startswith(b"#")]
    rows = [ln for ln in lines if ln.strip() and not ln.startswith(b"#")]
    nray = None
    if head and b"=" in head[0]:
        tail = head[0].split(b"=", 1)[1].split()
        if len(tail) >= 2 and tail[0].isdigit():
            nray = int(tail[0])
    if nray is None:
        return "bad header"
    if len(rows) < need:
        return "{} table rows < {}".format(len(rows), need)
    ncol = len(rows[-1].split())
    if ncol != 2 + 2 * nray:
        return "last row has {} columns, expected {}".format(ncol, 2 + 2 * nray)
    return None
