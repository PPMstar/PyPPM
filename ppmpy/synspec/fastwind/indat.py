"""
A format-preserving editor of FASTWIND's main input file INDAT.DAT (standard library only).

FASTWIND v10 (nlte.f90, v10.6.4.1 lines 1511-1520) reads the first ten lines of INDAT.DAT with list-directed
READ statements, one statement per line; whatever follows the values on a line is ignored, which the templates
use for a comment naming the fields. :class:`Indat` names these values after the template comments
(:data:`INDAT_SCHEMA`, identical to ``ppmpy.synspec.fwresults.INDAT_SCHEMA``) and edits them in place, keeping
every other byte of the file: comments, blanks, the lines after the tenth (clumping, Hopf parameters,
abundances, X-rays), line endings.

Byte compatibility with the legacy runner
------------------------------------------
``Indat.read(template).set(MODNAM=name, TEFF=t).to_text()`` is byte-identical to the awk edit of
fw_sphere_point.sh::

    awk -v n="$name" -v t="$teff" 'NR == 1 {printf "%-47sCATALOG\\n", n; next}
         NR == 4 {sub(/^[^,]*,/, sprintf("%.3f,", t))} {print}' INDAT.template

for templates in FASTWIND's own layout (line 1: the model name with its comment 'CATALOG' in column 48; line 4:
the T_eff token at the start of the line, directly followed by a comma; a final newline): every INDAT template
of the M424 project, checked in the tests on 10^4 random T_eff values against awk run as a subprocess. The
rules that give this:

* MODNAM: line 1 becomes the name left-justified in 47 columns, then the template's comment (awk writes
  'CATALOG' whatever the template has). A name of 47 or more characters is followed by one blank (awk glues the
  comment to it, which FASTWIND would read as part of the name); names longer than 50 characters
  (``CHARACTER*50 MODNAM``) are refused.
* TEFF given as a number is written ``'%.3f'`` (awk's ``sprintf("%.3f", t)``; Python's and C's '%.3f' are both
  correctly rounded, ties to even); a string is written verbatim.
* any other value replaces its token only; when the token is followed by blanks and more text on the line, the
  blank run is shortened or lengthened (to at least one blank) so that the following text keeps its column. A
  token followed directly by a comma (TEFF on line 4) leaves the rest of the line as it is, as awk's ``sub``.

Differences from awk, all outside FASTWIND's layout: awk also removes blanks before the first value of line 4
(and between that value and its comma), writes 'CATALOG' instead of another line-1 comment, adds a final
newline that the template lacks, and keeps a carriage return only on lines it does not rewrite.

Lines are split at '\n' only, as Fortran and awk do (a '\r' before it counts as part of the line ending); tokens
are separated by blanks, tabs and commas only. Other control bytes (form feed, 0x85 = a cp1252 ellipsis read as
latin-1, ...) are ordinary characters, so they cannot shift the line numbers of the fields (str.splitlines would).

PP 2026-10-02: new (M6); replaces the awk edit of fw_sphere_point.sh:23-24 (project stellar-atmosphere-KU-Leuven).
"""
import math
import re

from .install import atomic_write

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
"""Field names of the first ten lines of INDAT.DAT, one tuple per line, in the order of nlte.f90's READ
statements (v10.6.4.1, 1511-1520: MODNAM; OPTNEUPDATE, HELONE, ITLAST, ITMORE; OPTMIXED; TEFF, GGRAV, SRNOM; RMAX,
TMIN; XMLOSS, VMIN, VMAX, BETA, VDIV; YHEIN, HEI; OPTMOD, OPTLUCY, MEGAS, ACCEL, OPTCMF1; VTURB, XMET, LINES,
LINES_IN_MODEL; ENATCOR, EXPANSION, SET_FIRST, SET_STEP), named as in the template comments. Same tuple as
``ppmpy.synspec.fwresults.INDAT_SCHEMA`` (checked in the tests; duplicated because this subpackage must not import
numpy)."""

LOGICAL_FIELDS = frozenset(("OPTNEUPDATE", "HE_ONE", "OPTMOD", "OPTTLUCY", "MEGAS", "ACCEL", "OPTCMF", "LINES",
                            "LINES_IN_MODEL", "ENATCOR", "EXPANSION"))
"""Fields FASTWIND reads as LOGICAL (written T / F)."""

INTEGER_FIELDS = frozenset(("ITSTART", "ITMORE", "SET_FIRST", "SET_STEP"))
"""Fields FASTWIND reads as INTEGER."""

STRING_FIELDS = frozenset(("MODNAM",))

MODNAM_WIDTH = 47
"""Column width of the model name on line 1 (fw_sphere_point.sh: ``printf "%-47sCATALOG"``)."""

MODNAM_MAXLEN = 50
"""``CHARACTER*50 :: MODNAM`` (nlte.f90:1107)."""

TEFF_FORMAT = "%.3f"
"""Format of a numeric TEFF (fw_sphere_point.sh: ``sprintf("%.3f,", t)``)."""

FIELD_LINE = {name: i for i, names in enumerate(INDAT_SCHEMA) for name in names}
FIELD_POS = {name: k for names in INDAT_SCHEMA for k, name in enumerate(names)}
FIELDS = tuple(name for names in INDAT_SCHEMA for name in names)

_TOKEN = re.compile(r"[^ \t\r\n,]+")
"""A list-directed value: separated by blanks, tabs and commas (ASCII only; other bytes belong to the token)."""
_BLANKS = " \t"
_LOGICAL = re.compile(r"\.?(T|F|TRUE|FALSE)\.?", re.IGNORECASE)
_SEPARATORS = re.compile(r"[\s,/']")


def split_lines(text):
    """
    Lines of ``text`` with their endings, split at '\n' only (Fortran and awk records; unlike str.splitlines,
    which also splits at \x0b, \x0c, \x1c-\x1e and \x85). ``''.join(split_lines(t)) == t``.
    """
    # PP 2026-10-02: new (M6); reviewer: str.splitlines shifted the INDAT line numbers at a 0x85 byte
    parts = text.split("\n")
    out = [p + "\n" for p in parts[:-1]]
    if parts[-1]:
        out.append(parts[-1])
    return out


def _split_ending(line):
    """(content, line ending) of one line of :func:`split_lines`: the ending is '\r\n', '\n' or ''."""
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    return line, ""


def parse_value(name, tok):
    """
    The typed value of one INDAT token.

    Parameters
    ----------
    name: str
        Field name (:data:`FIELDS`).
    tok: str
        The token text.

    Returns
    -------
    str, bool, int or float
        MODNAM: the string; logical fields: bool (T, F, .TRUE., ...); integer fields: int; others: float
        (Fortran D exponents accepted).

    Raises
    ------
    ValueError
        A token that does not read as the field's type.
    """
    # PP 2026-10-02: new (M6)
    if name in STRING_FIELDS:
        return tok
    if name in LOGICAL_FIELDS:
        if not _LOGICAL.fullmatch(tok):
            raise ValueError("{} = {!r} is not a Fortran logical".format(name, tok))
        return tok.strip(".").upper().startswith("T")
    if name in INTEGER_FIELDS:
        try:
            return int(tok)
        except ValueError:
            raise ValueError("{} = {!r} is not an integer".format(name, tok)) from None
    try:
        return float(tok.replace("D", "E").replace("d", "e"))
    except ValueError:
        raise ValueError("{} = {!r} is not a number".format(name, tok)) from None


def format_value(name, value):
    """
    The token text written for a value of field ``name``.

    A str is written verbatim (after checking that it is one list-directed token); bool -> 'T' / 'F'; int fields
    take integral numbers; TEFF numbers are written with :data:`TEFF_FORMAT`; other numbers with ``repr`` (the
    shortest text that reads back as the same double; Fortran reads '1e-10' and '2500.0').

    Raises
    ------
    ValueError
        A value of the wrong type, non-finite, or text that is not one token.
    """
    # PP 2026-10-02: new (M6)
    if isinstance(value, str):
        if not value or _SEPARATORS.search(value):
            raise ValueError("{}: {!r} is not one list-directed token".format(name, value))
        if name in STRING_FIELDS:
            return value
        parse_value(name, value)            # type check only; the text is kept as given
        return value
    if name in STRING_FIELDS:
        raise ValueError("{} must be a str".format(name))
    if name in LOGICAL_FIELDS:
        if not isinstance(value, bool):
            raise ValueError("{} must be a bool (or 'T' / 'F')".format(name))
        return "T" if value else "F"
    if isinstance(value, bool):
        raise ValueError("{} is not a logical field".format(name))
    if name in INTEGER_FIELDS:
        if int(value) != value:
            raise ValueError("{} must be an integer, got {!r}".format(name, value))
        return str(int(value))
    v = float(value)
    if not math.isfinite(v):
        raise ValueError("{} = {!r} is not finite".format(name, value))
    if name == "TEFF":
        return TEFF_FORMAT % v
    return repr(v)


class Indat:
    """
    INDAT.DAT as text, with its fields read and written in place.

    Parameters
    ----------
    text: str
        The file content (``Indat.read`` decodes bytes as latin-1, so any byte survives a round trip).

    Notes
    -----
    ``to_text()`` of an unedited object is the input text. Lines beyond the ten of :data:`INDAT_SCHEMA` are kept
    verbatim (``extra_lines()``). Methods that edit return the object, so edits chain:
    ``Indat.read(path).set(MODNAM='P000001', TEFF=37298.91).write('INDAT.DAT')``.
    """

    # PP 2026-10-02: new (M6)
    def __init__(self, text):
        if isinstance(text, (bytes, bytearray)):
            text = bytes(text).decode("latin-1")
        self._lines = split_lines(text)

    @classmethod
    def read(cls, path):
        """Read an INDAT.DAT file (bytes decoded as latin-1)."""
        with open(path, "rb") as f:
            return cls(f.read())

    def copy(self):
        """An independent copy."""
        return Indat(self.to_text())

    def to_text(self):
        """The file content (str)."""
        return "".join(self._lines)

    def to_bytes(self):
        """The file content as bytes (latin-1, the inverse of :meth:`read`)."""
        return self.to_text().encode("latin-1")

    def write(self, path):
        """
        Write the file atomically (a temporary name unique per call in the same directory, then rename; safe from
        many threads); returns ``path``.
        """
        return atomic_write(path, self.to_bytes())

    # -- reading ---------------------------------------------------------------------------------------
    def _tokens(self, i):
        """[(start, end, text)] of the tokens of line i (empty when the file is shorter)."""
        if i >= len(self._lines):
            return []
        content, _ = _split_ending(self._lines[i])
        return [(m.start(), m.end(), m.group()) for m in _TOKEN.finditer(content)]

    def _span(self, name):
        if name not in FIELD_LINE:
            raise KeyError("unknown INDAT field {!r}; known: {}".format(name, ", ".join(FIELDS)))
        i, k = FIELD_LINE[name], FIELD_POS[name]
        tok = self._tokens(i)
        if k >= len(tok):
            return i, None
        return i, tok[k]

    def raw(self, name):
        """The token text of field ``name`` (None if the line has no such token)."""
        _, t = self._span(name)
        return None if t is None else t[2]

    def get(self, name):
        """The typed value of field ``name`` (:func:`parse_value`); KeyError if the token is missing."""
        t = self.raw(name)
        if t is None:
            raise KeyError("INDAT field {} is missing (line {})".format(name, FIELD_LINE[name] + 1))
        return parse_value(name, t)

    def fields(self):
        """dict field name -> typed value, in READ order, for every field present (unparsable ones as raw text)."""
        out = {}
        for name in FIELDS:
            t = self.raw(name)
            if t is None:
                continue
            try:
                out[name] = parse_value(name, t)
            except ValueError:
                out[name] = t
        return out

    def extra_lines(self):
        """The lines after the ten of :data:`INDAT_SCHEMA`, without line endings."""
        return [_split_ending(ln)[0] for ln in self._lines[len(INDAT_SCHEMA):]]

    # -- editing ---------------------------------------------------------------------------------------
    def set(self, **values):
        """
        Set fields in place (keyword = field name); returns self.

        Values are formatted with :func:`format_value`. MODNAM rewrites line 1 in the template layout (see the
        module notes); any other field replaces its token, keeping the column of the text after it when the token
        is followed by blanks.

        Raises
        ------
        KeyError
            Unknown field, or a field whose line or token is missing in the file.
        ValueError
            A value that cannot be written for the field.
        """
        for name, value in values.items():
            text = format_value(name, value)
            i, t = self._span(name)
            if t is None:
                raise KeyError("INDAT field {} is missing (line {}): cannot set it".format(name, i + 1))
            content, end = _split_ending(self._lines[i])
            if name == "MODNAM":
                if len(text) > MODNAM_MAXLEN:
                    raise ValueError("MODNAM {!r} longer than {} characters".format(text, MODNAM_MAXLEN))
                rest = content[t[1]:].lstrip(_BLANKS)
                pad = text.ljust(MODNAM_WIDTH)
                if rest and len(text) >= MODNAM_WIDTH:
                    pad += " "
                self._lines[i] = pad + rest + end
                continue
            s, e, old = t
            after = content[e:]
            ws = len(after) - len(after.lstrip(" "))
            if 0 < ws < len(after) and after[ws] not in _BLANKS + "\r":   # blanks, then more text: keep its column
                after = " " * max(1, ws - (len(text) - len(old))) + after[ws:]
            self._lines[i] = content[:s] + text + after + end
        return self

    # -- checks ----------------------------------------------------------------------------------------
    def validate(self):
        """
        Problems that would stop or mislead FASTWIND, as a list of strings (empty: none found).

        Checked: every field of :data:`INDAT_SCHEMA` present and of its type; MODNAM a single token of at most 50
        characters; TEFF, LOGG, RSTAR, VINF positive; MDOT > 0 (FASTWIND needs a wind); ITSTART >= 0 and ITMORE >= 1;
        YHE >= 0; a clumping line (line 11) whose first value is >= 1 (nlte.f90: 'FIRST CLUMPING PARAMETER < 1');
        OPTTLUCY = F needs an OPTHOPF line (line 12: with F the Hopf parameters are looked up in
        HOPFPARA_ALL_* and must match T_eff, log g, Y_He exactly, CLAUDE.md pitfall 2); METALLICITY = 0 with
        LINES = T (FASTWIND stops); LINES = F with LINES_IN_MODEL = T (FASTWIND stops).
        """
        # PP 2026-10-02: new (M6); checks taken from nlte.f90:1511-1600 and the project's FASTWIND pitfalls
        probs = []
        vals = {}
        for name in FIELDS:
            t = self.raw(name)
            if t is None:
                probs.append("{}: missing (line {})".format(name, FIELD_LINE[name] + 1))
                continue
            try:
                vals[name] = parse_value(name, t)
            except ValueError as e:
                probs.append(str(e))
        m = vals.get("MODNAM")
        if m is not None and (len(m) > MODNAM_MAXLEN or "/" in m):
            probs.append("MODNAM {!r}: more than {} characters or a '/'".format(m, MODNAM_MAXLEN))
        for name in ("TEFF", "LOGG", "RSTAR", "VINF", "MDOT"):
            v = vals.get(name)
            if isinstance(v, float) and not v > 0:
                probs.append("{} = {} must be > 0".format(name, v))
        if isinstance(vals.get("ITSTART"), int) and vals["ITSTART"] < 0:
            probs.append("ITSTART = {} < 0".format(vals["ITSTART"]))
        if isinstance(vals.get("ITMORE"), int) and vals["ITMORE"] < 1:
            probs.append("ITMORE = {} < 1".format(vals["ITMORE"]))
        if isinstance(vals.get("YHE"), float) and vals["YHE"] < 0:
            probs.append("YHE = {} < 0".format(vals["YHE"]))
        extra = self.extra_lines()
        clf = extra[0].strip() if extra else ""
        if clf.upper() == "THICK":
            clf = extra[1].strip() if len(extra) > 1 else ""
        tok = [t for t in re.split(r"[,\s]+", clf) if t]
        try:
            if not tok or float(tok[0].replace("D", "E").replace("d", "e")) < 1.0:
                probs.append("clumping line (line 11) missing or first clumping factor < 1: {!r}".format(clf))
        except ValueError:
            probs.append("clumping line (line 11) does not start with a number: {!r}".format(clf))
        if vals.get("OPTTLUCY") is False:
            if len(extra) < 2 or not _LOGICAL.fullmatch((re.split(r"[,\s]+", extra[1].strip()) + [""])[0]):
                probs.append("OPTTLUCY = F needs an OPTHOPF line (line 12)")
            else:
                probs.append("OPTTLUCY = F: the Hopf parameters must exist for exactly this (TEFF, LOGG, YHE) in "
                             "HOPFPARA_ALL_* (else 'HOPF-PARAMETERS NOT FOUND'); use T for arbitrary T_eff")
        if vals.get("METALLICITY") == 0.0 and vals.get("LINES") is True:
            probs.append("METALLICITY = 0 but LINES = T (FASTWIND stops: 'XMET = 0., BUT LINE-BLOCKING REQUIRED')")
        if vals.get("LINES") is False and vals.get("LINES_IN_MODEL") is True:
            probs.append("LINES = F but LINES_IN_MODEL = T (FASTWIND stops)")
        return probs

    def __repr__(self):
        f = self.fields()
        return "Indat(MODNAM={!r}, TEFF={!r}, {} lines)".format(f.get("MODNAM"), f.get("TEFF"), len(self._lines))


def edit_like_awk(template_text, name, teff):
    """
    ``Indat(template_text).set(MODNAM=name, TEFF=teff).to_text()``: the INDAT.DAT of one sphere point as written by
    fw_sphere_point.sh (``teff`` a number, or the verbatim text of points.txt, which is parsed and formatted '%.3f'
    exactly as awk's ``sprintf("%.3f", t)`` does with a string).
    """
    # PP 2026-10-02: ported from fw_sphere_point.sh:23-24 (awk edit)
    t = float(teff) if isinstance(teff, str) else teff
    return Indat(template_text).set(MODNAM=name, TEFF=t).to_text()
