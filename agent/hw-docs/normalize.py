"""Normalization rules for the docling corpus (issue #28).

THE one place that defines every text transform shared by the corpus
consumers: the renderer (`build.py`, which writes `md/`), the semantic
search entry (`search.py`), and the structural citation grader (#29). Written
against docling 2.126 output only — nothing here is inherited from the
pymupdf4llm implementations (issue #28 hard wall).

Rules, in application order:

1. Escape folding — docling's markdown export escapes underscores that
   markdown could read as emphasis, so prose identifiers arrive as
   ``M\\_TTCAN``. ``\\_`` is the only escape docling emits anywhere in
   this corpus (verified across every #27 spike output); fold it to ``_``.

2. Whitespace — paragraphs export as single long lines, but narrow table
   cells flatten with embedded spaces. For *matching* (`canonical`) runs
   of whitespace collapse to one space; the md *rendering* keeps docling's
   line and table structure untouched.

3. Wrapped-token reconstruction — a narrow PDF cell wraps a long
   identifier mid-token and docling renders the pieces space-separated:
   ``GP70_UART1_T XD_BOOT2_STR AP``. Two tiers:

   a. local (always applied — the wrap is unambiguous from the pieces
      alone): one fragment *ends* with ``_`` (``FORCE_ RECOVERY``), or
      one side is an all-caps run of at most two characters while the
      other side carries an underscore (``PEX1_CLKRE Q``,
      ``..._STR AP``).
   b. corpus-confirmed (applied via a wrap table derived by `build.py`):
      locally ambiguous — ``GP70_UART1_T XD_BOOT2_STR`` looks exactly
      like two real pin names (``I2C2_SDA I2C2_SCL``). `build.py`
      derives the table by confirming that the *joined* form occurs as a
      standalone token elsewhere in the corpus; ``GP70_UART1_TXD_BOOT2_
      STRAP`` does (pin-function-names guide), ``I2C2_SDAI2C2_SCL``
      occurs nowhere, so only the first pair ever joins. The table lives
      in ``md/index/wrap_table.json`` and is applied at consumption time
      (search, grading) as well as at render time.

Consumers:
    render(text, table)    -> md writing (line structure preserved)
    canonical(text, table) -> matching/compare form (single-spaced)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# A fragment of a (possibly wrapped) identifier: uppercase letters,
# digits and underscores, containing at least one letter. Pure digits
# ("239") and mixed-case words ("Cathode") are not fragments — that is
# what keeps "UART 0 Transmit" and "LED Cathode" out of every rule below.
_CORE = r"[A-Z0-9_]*[A-Z][A-Z0-9_]*"
FRAGMENT_RE = re.compile(r"\b" + _CORE + r"\b")

# Runs of fragments separated by single spaces; only interesting when at
# least one fragment carries an underscore (otherwise there is no
# identifier evidence to reconstruct).
_RUN_RE = re.compile(
    r"(?<![A-Za-z0-9_])"          # not mid-token on the left
    r"(" + _CORE + r")"           # first fragment
    r"(?: (" + _CORE + r"))+"     # space-separated further fragments
    r"(?![A-Za-z0-9_])"           # not mid-token on the right
)

_WS_RE = re.compile(r"\s+")


def fold_escapes(text: str) -> str:
    """Rule 1: ``M\\_TTCAN`` -> ``M_TTCAN`` (the sole docling escape)."""
    return text.replace("\\_", "_")


def collapse_ws(text: str) -> str:
    """Rule 2: whitespace runs -> single space (matching form only)."""
    return _WS_RE.sub(" ", text).strip()


def _joins_locally(left: str, right: str) -> bool:
    """Local, unambiguous wrap evidence between adjacent fragments."""
    if left.endswith("_"):
        return True  # no valid identifier ends with an underscore
    if len(left) <= 2 and "_" in right:
        return True
    if len(right) <= 2 and "_" in left:
        return True
    return False


def join_local(text: str) -> str:
    """Rule 3a: merge fragment pairs with local wrap evidence."""
    def _sub(m: re.Match) -> str:
        parts = m.group(0).split(" ")
        if "_" not in m.group(0):
            return m.group(0)  # no identifier evidence — not a wrap run
        out = [parts[0]]
        for p in parts[1:]:
            if _joins_locally(out[-1], p):
                out[-1] += p
            else:
                out.append(p)
        return " ".join(out)

    return _RUN_RE.sub(_sub, text)


def _prep_raw(text: str) -> str:
    """Raw docling md -> pre-join form (fold + collapse only). Token
    counting happens here: counting joined text would let a join confirm
    itself circularly ("T J" supplying the count for "TJ")."""
    return collapse_ws(fold_escapes(text))


def _prep(text: str) -> str:
    """Raw docling md -> form the wrap rules operate on."""
    return join_local(_prep_raw(text))


def token_counts(texts) -> dict[str, int]:
    """Occurrences of every standalone fragment token across `texts`."""
    counts: dict[str, int] = {}
    for text in texts:
        for tok in FRAGMENT_RE.findall(_prep_raw(text)):
            counts[tok] = counts.get(tok, 0) + 1
    return counts


def derive_wrap_table(texts) -> list[dict]:
    """Rule 3b: corpus-confirmed joins.

    A fragment run joins when its fully-joined form occurs as a
    standalone token somewhere in the corpus — wrapped pieces are rare as
    standalone tokens, real adjacent names joined never occur.
    Returns entries sorted longest-wrapped-first (application order).
    """
    counts = token_counts(texts)
    table: dict[str, str] = {}
    for text in texts:
        for m in _RUN_RE.finditer(_prep(text)):
            wrapped = m.group(0)
            if "_" not in wrapped:
                continue  # no identifier evidence — not a wrap run
            joined = wrapped.replace(" ", "")
            if counts.get(joined, 0) >= 1:
                table[wrapped] = joined
    return [
        {"wrapped": w, "joined": j}
        for w, j in sorted(table.items(), key=lambda kv: -len(kv[0]))
    ]


def load_wrap_table(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def apply_wraps(text: str, table) -> str:
    """Rule 3b application; entries are already longest-first. Boundary
    guards mirror the derivation side so a table entry can never fire
    mid-token (the wrapped form inside a longer identifier)."""
    for entry in table or ():
        text = re.sub(
            r"(?<![A-Za-z0-9_])" + re.escape(entry["wrapped"]) + r"(?![A-Za-z0-9_])",
            entry["joined"].replace("\\", "\\\\"), text)
    return text


def render(text: str, table=None) -> str:
    """md-writing form: escapes folded and wraps reconstructed, docling's
    line/table layout preserved."""
    return apply_wraps(join_local(fold_escapes(text)), table)


def canonical(text: str, table=None) -> str:
    """matching form: fold, collapse, reconstruct — used by search and
    the grader on BOTH sides of any comparison."""
    return apply_wraps(join_local(collapse_ws(fold_escapes(text))), table)
