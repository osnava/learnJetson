#!/usr/bin/env python3
"""Structural citation grader over the docling corpus (issue #29).

Is `doc §section (p. N) + quote` real, or invented? The human-facing
surface is unchanged from the v1 grader (#22); every corpus check is new
and structural — resolved against the docling JSON source of truth
(`md/<doc>.json`, per-object `prov.page_no`), never against text-stream
anchors or span heuristics. Nothing here is inherited from the
pymupdf-tuned v1 machinery (the #27/#28 hard wall); the answer-*parsing*
side is the surface contract, carried over verbatim in shape.

Per citation:

  1. document resolves to a corpus stem          else DOC_UNKNOWN
     (a name that is no corpus stem can never be
     fetched, so it fails like an invention)
  2. fetched-ness                                else DOC_MISSING
     (corpus absent, or a canonical stem whose
     files are not on this machine — never the
     agent's fault)
  3. the cited section's heading object exists    else SECTION_NOT_FOUND
     in the JSON — by number (§3.4, Ch. 2), by
     caption (Table 3-4, Figure 3-1), or by name
     (§Encode — search.py prints unnumbered
     headings too)
  4. cited page is in the section's span, i.e.    else PAGE_OUTSIDE_SECTION
     the page set of the body items between that
     heading and the heading that closes it
  5. the quoted line occurs on the cited page —   else QUOTE_NOT_AT_PAGE
     in that page's provenance-ordered items

Structural rules (what replaces v1's span heuristics):

- Document order is the body tree walk of the JSON; items outside the
  `body` content layer (running headers/footers) are dropped everywhere.
  That alone is v1's MIN_SPAN_CHARS trick: a page carrying only furniture
  never joins a section's span, and a page with a heading + one line
  always does.
- A numbered section (§3.4, depth = dotted parts) is closed by the next
  numbered/chapter heading of depth <= its own; unnumbered headings
  ("Notes:") never close a numbered section. A named section (§Encode)
  is closed by any next heading. Chapters are depth 1.
- The span of a `Table N-M` citation is the captioned table object's own
  prov pages (plus the caption's) — tables are objects, not headings;
  nothing to close. `Figure N-M` resolves the same way against the
  caption object (plus the picture's pages — a drawing may sit on the
  page before its caption): a figure is ADDRESSED — caption + page —
  never described. The caption text is the citable quote; the drawing
  itself is the human's to see in the cached PDF
  (`pdf/<doc>.pdf#page=N`).

Quote comparison — one normalization, defined on docling output and
shared with the renderer and search (normalize.py, issue #28): both the
quote and the page's item text go through `canonical()` (escape fold,
whitespace collapse, wrapped-token reconstruction, corpus-confirmed wrap
table), prefixed by a symmetric NFKC + curly-punctuation fold of the
answer side. The final comparison strips ALL whitespace from both sides
(the *skeleton*). That one rule absorbs every way docling output can
differ from how an answer spells a real line:

  - a cell wrapped mid-cell ("...error text / continues on this wrapped
    cell") — quoted with a plain space;
  - a token the PDF wrapped in a narrow cell, which canonical already
    reconstructed ("GP70_UART1_T XD_BOOT2_STR AP");
  - dehyphenation: a line wrapped after a hyphen arrives from docling as
    `Auto-Power- On` (hyphen + space); the answer writes `Auto-Power-On`.
    The hyphen stays literal on both sides — only the wrapped space
    differs, and the skeleton deletes spaces. No hyphen surgery, ever:
    joining or dropping hyphens would also fuse genuinely hyphenated
    adjacent words.

Tables serialize for matching as rows of space-joined cells; a quote
copied from the md rendering instead spells cell boundaries as pipes
(`VDD_3V3_SYS|40-pin header|3.3|0.1`), so `|` folds to a space on both
sides before the skeleton — a quote may span the cell boundaries of one
row however the answer spells them.

A fetched document without a JSON (the TRM's slab mode, the schematics:
md and chunks only) cannot be graded structurally — verdict
NO_PROVENANCE, soft like DOC_MISSING: never a pass, never the agent's
fault.

Exit codes (the deliberate inherited surface — search.py mirrors it):
  0  every citation OK (an answer with no citations at all also exits 0,
     with a warning on stderr — judging citation *rate* is #24/#25's job)
  1  at least one citation fails verification
  2  only DOC_MISSING / NO_PROVENANCE — an unfetchable-or-ungradable
     corpus is NOT a hallucinating agent
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import normalize  # noqa: E402

# --- corpus vocabulary (stems pinned in INDEX.md; keep both in step) -----

CANONICAL_STEMS = [
    "datasheet",
    "devkit-carrier-schematics",
    "devkit-carrier-spec",
    "orin-nx-nano-design-guide",
    "orin-pin-function-names",
    "orin-pinmux",
    "orin-thermal-design-guide",
    "orin-trm",
]

# Prose shorthand sessions actually used -> corpus stem. Full stems are
# canonical; these exist so hand-written answers need no exact filenames.
ALIASES = {
    "data sheet": "datasheet",
    "carrier board spec": "devkit-carrier-spec",
    "carrier spec": "devkit-carrier-spec",
    "carrier schematics": "devkit-carrier-schematics",
    "schematics": "devkit-carrier-schematics",
    "design guide": "orin-nx-nano-design-guide",
    "thermal design guide": "orin-thermal-design-guide",
    "thermal guide": "orin-thermal-design-guide",
    "trm": "orin-trm",
    "pin function names guide": "orin-pin-function-names",
    "pinmux": "orin-pinmux",
}

_DOC_NAMES = sorted(
    {f"{s}.md" for s in CANONICAL_STEMS} | set(CANONICAL_STEMS)
    | set(ALIASES) | set(ALIASES.values()),
    key=len, reverse=True,  # longest-first so `devkit-carrier-spec.md` beats sub-words
)
DOC_RE = re.compile(
    r"(?<![\w.-])(" + "|".join(re.escape(n) for n in _DOC_NAMES)
    + r"|[A-Za-z0-9_][\w.-]*\.md)(?![\w-])",
    re.IGNORECASE,
)

# Section-token patterns; kind decides how a token maps to JSON headings.
SEC_PATTERNS = [
    # §3.4 / §2.1.1 / §3.1-3.8 (en dash or hyphen range)
    ("sec", re.compile(r"§\s*(?P<t>\d+(?:\.\d+)*(?:\s*[–-]\s*\d+(?:\.\d+)*)?)")),
    # Ch. 1 / Chapter 3 / ch3
    ("chapter", re.compile(r"\b[Cc]h(?:apter)?\.?\s*(?P<t>\d+)\b\.?")),
    # Table 3-4
    ("table", re.compile(r"\bTable\s+(?P<t>\d+\s*-\s*\d+)\b")),
    # Figure 3-1 — a figure addressed by its caption object (figures are
    # cited, never described; the caption is the citable part)
    ("figure", re.compile(r"\bFigure\s+(?P<t>\d+\s*-\s*\d+)\b")),
    # §Encode — an unnumbered heading cited by name (search.py prints
    # them). Stops at the first character that cannot belong to a
    # heading name; the numbered pattern above owns digit-first tokens.
    ("name", re.compile(
        r"§\s*(?P<t>[A-Za-z](?:[A-Za-z0-9 &/+'-]*[A-Za-z0-9&/+-])?)")),
]
PAGE_PAREN_RE = re.compile(r"\(\s*p\.\s*(\d+)\s*\)")
# The bare form must also fire inside "(devkit-carrier-spec.md §3.3 …, p. 26)"
# — answers wrap the whole citation in parens, so a following ")" means
# nothing. The lookbehind already stops it from re-matching inside "(p. N)".
PAGE_BARE_RE = re.compile(r"(?<![\w(])p\.\s*(\d+)\b")
QUOTE_RE = re.compile(r'"([^"]{2,400}?)"', re.S)
PARA_SPLIT_RE = re.compile(r"\n\s*\n")

# How far a §section / doc token may sit from its (p. N) and still belong
# to it. Generous enough for "§3.4, Table 3-4 (p. 28)"; tight enough that a
# §token from a previous sentence is not kidnapped.
SEC_ATTACH_CHARS = 120
DOC_ATTACH_CHARS = 250

FAIL_VERDICTS = {
    "SECTION_NOT_FOUND", "PAGE_OUTSIDE_SECTION", "QUOTE_NOT_AT_PAGE", "NO_QUOTE",
    "DOC_UNKNOWN",
}
# NO_QUOTE is not literally an invention — but it is unverifiable, and the
# protocol requires the quote, so it fails the citation either way.
SOFT_VERDICTS = {"DOC_MISSING", "NO_PROVENANCE"}

# Symmetric unicode hygiene for the comparison: NFKC plus curly quotes and
# dash variants folded to their ASCII forms, and `|` folded to a space —
# the md rendering delimits table cells with pipes while the JSON substrate
# joins them with spaces, and a cell boundary is spelling, not quote
# content. Applied to BOTH the answer quote and the corpus text: it is not
# a docling rule, just punctuation neutrality, so neither side wins.
_FOLD = str.maketrans(
    {"“": '"', "”": '"', "„": '"', "‟": '"', "‘": "'", "’": "'",
     "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-", "‒": "-",
     "|": " "},
)
_WS_RE = re.compile(r"\s+")


# --- answer parsing (the human-facing surface — unchanged from v1) --------

@dataclass
class Citation:
    doc_token: str | None          # as written in the answer (None = inherited/absent)
    stem: str | None               # resolved corpus stem
    secs: list[tuple[str, str]]    # (token, kind) as written, primary first
    page: int
    quote: str | None = None
    span: tuple[int, int] = field(default=(0, 0))  # citation extent in the paragraph


def sections_in(text: str):
    """Yield (start, end, token, kind) for every section-like token."""
    found = []
    for kind, pat in SEC_PATTERNS:
        found.extend((m.start(), m.end(), m.group("t"), kind)
                     for m in pat.finditer(text))
    found.sort()
    return found


def page_cites(text: str):
    """Every page marker — `(p. N)` and the bare `p. N` (which also fires
    inside "(doc.md §3.3 …, p. 26)", where the paren wraps the whole
    citation). The bare form's lookbehind stops it re-matching inside
    `(p. N)` itself."""
    return sorted(list(PAGE_PAREN_RE.finditer(text)) + list(PAGE_BARE_RE.finditer(text)),
                  key=lambda m: m.start())


def claim_sections(pages, secs):
    """Attach each section token to the nearest page cite that follows it
    within SEC_ATTACH_CHARS. Returns one claim list per page cite."""
    claimed = [[] for _ in pages]
    used = set()
    for s in secs:
        for i, pm in enumerate(pages):
            if s[0] in used or s[1] > pm.start():
                continue
            if pm.start() - s[1] <= SEC_ATTACH_CHARS:
                claimed[i].append(s)
                used.add(s[0])
                break
    return claimed


def parse_answer(text: str) -> list[Citation]:
    text = unicodedata.normalize("NFKC", text).translate(_FOLD)
    citations: list[Citation] = []
    for para in PARA_SPLIT_RE.split(text):
        pages = page_cites(para)
        if not pages:
            continue

        secs = sections_in(para)
        docs = list(DOC_RE.finditer(para))
        quotes = list(QUOTE_RE.finditer(para))

        # Attach sections: a section token belongs to the nearest (p. N)
        # that follows it within SEC_ATTACH_CHARS.
        claimed = claim_sections(pages, secs)

        # Attach documents: rightmost doc token before the citation's first
        # section; inherit the previous citation's doc when none (supports
        # "docA §1 (p. 2), §2 (p. 5)").
        inherited = (None, None)
        para_cits: list[Citation] = []
        for i, pm in enumerate(pages):
            if not claimed[i]:
                continue  # a bare (p. N) with no §section is not a gradeable citation
            first_sec_start = claimed[i][0][0]
            doc_m = None
            for m in docs:
                if m.end() <= first_sec_start and first_sec_start - m.end() <= DOC_ATTACH_CHARS:
                    doc_m = m  # keep the rightmost candidate
            if doc_m is not None:
                pair = (doc_m.group(0), _resolve_stem(doc_m.group(0)))
            else:
                pair = inherited
            inherited = pair
            para_cits.append(Citation(
                doc_token=pair[0], stem=pair[1],
                secs=[(tok, kind) for _, _, tok, kind in claimed[i]],
                page=int(pm.group(1)),
                span=(doc_m.start() if doc_m else pm.start(), pm.end()),
            ))

        # Pair quotes: each citation takes the nearest unclaimed quote in
        # the paragraph, ignoring quotes inside any citation's own extent
        # (a quoted section label like `"HD Video → Encode"` is part of the
        # citation, not the load-bearing line).
        inner = [(c.span[0], c.span[1]) for c in para_cits]
        free = [q for q in quotes
                if not any(a <= q.start() and q.end() <= b for a, b in inner)]
        for c in para_cits:
            best, best_d = None, None
            for q in free:
                d = q.start() - c.span[1] if q.start() >= c.span[1] else c.span[0] - q.end()
                if best_d is None or d < best_d:
                    best, best_d = q, d
            if best is not None:
                c.quote = best.group(1).strip()
                free.remove(best)

        citations.extend(para_cits)
    return citations


def _resolve_stem(token: str) -> str:
    t = token.lower().removesuffix(".md")
    if t in CANONICAL_STEMS:
        return t
    return ALIASES.get(t, t)


def token_lookups(token: str, kind: str) -> list[tuple[str, str]]:
    """Section token -> the (kind, label) lookups it requires.

    A range §3.1-3.8 requires both endpoints; the citing page may sit
    anywhere in the combined span, which the caller assembles.
    """
    if kind == "name":
        return [("name", token.replace("–", "-").strip())]  # spaces are significant
    token = _label_key(token)
    if kind in ("table", "figure"):
        return [(kind, token)]
    if "-" in token:  # §3.1-3.8
        a, b = token.split("-", 1)
        return [("num", a), ("num", b)]
    if "." in token:
        return [("num", token)]
    return [("chapter", token)]  # a bare §3 refers to the chapter


# --- docling JSON substrate ------------------------------------------------

_CHAPTER_HEAD_RE = re.compile(r"Chapter\s+(\d+)\b")
_NUM_HEAD_RE = re.compile(r"(\d+(?:\.\d+)*)\s")
_TABLE_CAP_RE = re.compile(r"Table\s+(\d+)\s*-\s*(\d+)\b")
_FIGURE_CAP_RE = re.compile(r"Figure\s+(\d+)\s*-\s*(\d+)\b")


@dataclass
class Heading:
    kind: str      # "chapter" | "num" | "name"
    label: str     # "3" | "3.4" | "Button Header"
    depth: int     # chapter 1, num = dotted parts; names never close numbered
    index: int     # position of the heading item in DocIndex.items
    title: str     # heading text as written, for notes


@dataclass
class DocIndex:
    """One parsed md/<doc>.json: body items in document order plus the
    heading and caption registries the grader resolves against."""
    items: list[dict]                       # body-layer items, document order
    headings: list[Heading]                 # section_header items, document order
    tables: dict[str, tuple[set[int], str]]  # "3-4" -> (pages, caption text)
    figures: dict[str, tuple[set[int], str]]  # "3-1" -> (pages, caption text)


def item_pages(item: dict) -> set[int]:
    """Page set of one JSON item's provenance — the linter (#30) sums these
    to compare JSON provenance coverage against a source PDF."""
    return {p.get("page_no") for p in (item.get("prov") or []) if p.get("page_no")}


def resolve_token(dox: DocIndex, token: str, kind: str) -> Span | None:
    """One section token (as written in an answer or INDEX row) resolved to
    its span — the public seam the linter (#30) checks routing rows
    through, so it never walks this module's private resolvers."""
    lookups = token_lookups(token, kind)
    if len(lookups) == 2:  # §3.1-3.8: one structural range
        return _range_span(dox, lookups[0][1], lookups[1][1])
    return _resolve(dox, *lookups[0])


def _deref(doc: dict, ref: str):
    """#/texts/12 -> doc['texts'][12] (collections are already plural)."""
    _, kind, idx = ref.split("/")
    return doc.get(kind, [])[int(idx)]


def _item_text(item: dict) -> str:
    """Matching-form text of one JSON item: headings/texts/captions carry
    `text`; tables serialize as rows of space-joined cells (pipes are md
    rendering syntax, not substrate). Pictures and groups contribute
    nothing."""
    text = item.get("text")
    if text is not None:
        return text
    if item.get("label") == "table":
        cells = sorted(item.get("data", {}).get("table_cells", []),
                       key=lambda c: (c.get("start_row_offset_idx", 0),
                                      c.get("start_col_offset_idx", 0)))
        rows: list[list[str]] = []
        cur: int | None = None
        for c in cells:
            r = c.get("start_row_offset_idx", 0)
            if cur is None or r != cur:
                rows.append([])
                cur = r
            rows[-1].append(c.get("text", ""))
        return "\n".join(" ".join(row) for row in rows)
    return ""


def load_doc(path: Path) -> DocIndex | None:
    doc = json.loads(path.read_text(encoding="utf-8", errors="replace"))

    # Document order = the body tree walk; furniture (running headers and
    # footers, content_layer != "body") is dropped everywhere — that is the
    # whole span-exclusion story, no content thresholds.
    items: list[dict] = []
    seen: set[int] = set()

    def walk(node: dict) -> None:
        for ch in node.get("children", []):
            ref = ch.get("$ref")
            if not ref:
                continue
            try:
                it = _deref(doc, ref)
            except (ValueError, IndexError, KeyError):
                continue
            if id(it) in seen:
                continue
            seen.add(id(it))
            items.append(it)
            walk(it)

    walk(doc.get("body", {}))
    body = [i for i in items if i.get("content_layer", "body") == "body"]

    headings: list[Heading] = []
    tables: dict[str, tuple[set[int], str]] = {}
    figures: dict[str, tuple[set[int], str]] = {}
    for idx, it in enumerate(body):
        label = it.get("label")
        text = it.get("text") or ""
        if label == "section_header":
            m = _CHAPTER_HEAD_RE.match(text)
            if m:
                headings.append(Heading("chapter", m.group(1), 1, idx, text))
                continue
            m = _NUM_HEAD_RE.match(text)
            if m:
                num = m.group(1)
                headings.append(Heading("num", num, num.count(".") + 1, idx, text))
                continue
            # unnumbered: citable as §name, never closes a numbered section
            name = normalize.canonical(text).rstrip(":").strip()
            headings.append(Heading("name", name, 99, idx, text))
        elif label == "caption":
            # a caption's span is its own pages plus the parent object's —
            # the figure drawing may sit on the page before its caption
            pages = item_pages(it)
            parent = (it.get("parent") or {}).get("$ref", "")
            if parent.startswith(("#/tables/", "#/pictures/")):
                try:
                    pages |= item_pages(_deref(doc, parent))
                except (ValueError, IndexError, KeyError):
                    pass
            m = _TABLE_CAP_RE.match(text)
            if m:
                tables[f"{m.group(1)}-{m.group(2)}"] = (pages, text)
                continue
            m = _FIGURE_CAP_RE.match(text)
            if m:
                figures[f"{m.group(1)}-{m.group(2)}"] = (pages, text)
    return DocIndex(body, headings, tables, figures)


def _section_item_range(dox: DocIndex, hpos: int) -> tuple[int, int | None]:
    """items[a:b] belonging to the section headed by headings[hpos]: from
    the heading until the heading that closes it (None = end of document)."""
    h = dox.headings[hpos]
    for j in range(hpos + 1, len(dox.headings)):
        n = dox.headings[j]
        if h.kind == "name":
            return h.index, n.index           # any next heading closes a name
        if n.kind != "name" and n.depth <= h.depth:
            return h.index, n.index           # shallower-or-equal numbered ends it
    return h.index, None


def _range_pages(dox: DocIndex, a: int, b: int | None) -> set[int]:
    pages: set[int] = set()
    for it in dox.items[a: b if b is not None else len(dox.items)]:
        pages |= item_pages(it)
    return pages


@dataclass
class Span:
    pages: set[int]
    title: str


def _label_key(label: str) -> str:
    """Whitespace-free label key: '3. 4' and '3 - 4' must hash like '3.4'
    and '3-4'. Names never pass through here — their spaces are content."""
    return re.sub(r"\s*", "", label).replace("–", "-")


def _find_heading(dox: DocIndex, kind: str, label: str) -> int | None:
    """Position in headings[] of the first heading of this kind+label."""
    lab = _label_key(label)
    for j, h in enumerate(dox.headings):
        if h.kind == kind and h.label == lab:
            return j
    return None


def _range_span(dox: DocIndex, a_label: str, b_label: str) -> Span | None:
    """§3.1-3.8: the items from the first endpoint's heading through the
    end of the last endpoint's section — the true structural range, holes
    included (a furniture-only interior page stays out)."""
    ja, jb = _find_heading(dox, "num", a_label), _find_heading(dox, "num", b_label)
    if ja is None or jb is None:
        return None
    if dox.headings[jb].index < dox.headings[ja].index:
        ja, jb = jb, ja
    _, b_end = _section_item_range(dox, jb)
    return Span(_range_pages(dox, dox.headings[ja].index, b_end),
                dox.headings[ja].title)


def _heading_name(h: Heading) -> str | None:
    """The citable name of a heading: the label itself for unnumbered
    headings, the title minus the leading `Chapter N.` / `3.4` prefix for
    numbered ones (so §Widget Overview finds the '1.1 Widget Overview'
    object), None when the numbered heading carries no name."""
    if h.kind == "name":
        return h.label
    return normalize.canonical(
        re.sub(r"^(Chapter\s+\d+\.?|\d+(?:\.\d+)*)\s+", "", h.title)
    ).rstrip(":").strip() or None


def _resolve(dox: DocIndex, kind: str, label: str) -> Span | None:
    """One section token resolved against the JSON registries."""
    if kind == "table":
        entry = dox.tables.get(_label_key(label))
        return Span(entry[0], entry[1]) if entry else None
    if kind == "figure":
        entry = dox.figures.get(_label_key(label))
        return Span(entry[0], entry[1]) if entry else None
    if kind == "name":
        want = normalize.canonical(label).rstrip(":").strip()
        matches = [j for j, h in enumerate(dox.headings)
                   if (name := _heading_name(h)) is not None
                   and (name == want or name.endswith(" " + want))]
        if not matches:
            return None
        # duplicate names (two 'Description' headings): any of their spans
        pages: set[int] = set()
        for j in matches:
            a, b = _section_item_range(dox, j)
            pages |= _range_pages(dox, a, b)
        return Span(pages, dox.headings[matches[0]].title)
    j = _find_heading(dox, kind, label)
    if j is None:
        return None
    a, b = _section_item_range(dox, j)
    return Span(_range_pages(dox, a, b), dox.headings[j].title)


def _skeleton(text: str, wrap_table) -> str:
    """The comparison form: symmetric NFKC/punctuation fold, the shared
    canonical() from normalize.py, then ALL whitespace deleted — the one
    rule that absorbs cell wraps, wrapped tokens and dehyphenation
    (`Auto-Power- On` -> `Auto-Power-On`; hyphens stay literal)."""
    s = unicodedata.normalize("NFKC", text).translate(_FOLD)
    return _WS_RE.sub("", normalize.canonical(s, wrap_table))


def _page_skeleton(dox: DocIndex, page: int, wrap_table) -> str:
    parts = [_item_text(it) for it in dox.items if page in item_pages(it)]
    return _skeleton("\n".join(p for p in parts if p), wrap_table)


def quote_needle(quote: str, wrap_table) -> str:
    """A quote folded to its skeleton, minus the answer's own terminal
    punctuation ("…nominal." for a corpus line that continues)."""
    n = _skeleton(quote, wrap_table).strip(".,;:!?…")
    return n or _skeleton(quote, wrap_table)


# --- grading -----------------------------------------------------------------

@dataclass
class Result:
    citation: Citation
    verdict: str
    notes: list[str] = field(default_factory=list)


def _fmt_pages(pages: set[int]) -> str:
    ps = sorted(pages)
    if len(ps) == 1:
        return f"page {ps[0]}"
    if ps == list(range(ps[0], ps[-1] + 1)):
        return f"pages {ps[0]}-{ps[-1]}"
    return "pages " + ", ".join(map(str, ps))  # gappy span: show the holes


def grade(citations: list[Citation], corpus_dir: Path) -> list[Result]:
    corpus_present = (corpus_dir.is_dir()
                      and (any(corpus_dir.glob("*.json"))
                           or any(corpus_dir.glob("*.md"))))
    wrap_table = normalize.load_wrap_table(corpus_dir / "index" / "wrap_table.json")
    cache: dict[str, DocIndex | None] = {}

    def load(stem: str) -> DocIndex | None:
        if stem not in cache:
            json_f = corpus_dir / f"{stem}.json"
            cache[stem] = load_doc(json_f) if json_f.is_file() else None
        return cache[stem]

    results = []
    for c in citations:
        if not corpus_present:
            results.append(Result(c, "DOC_MISSING",
                                  ["corpus not fetched — run hw-docs/fetch.sh"]))
            continue
        if c.stem is None:
            results.append(Result(c, "DOC_MISSING", ["no document named in citation"]))
            continue
        dox = load(c.stem)
        if dox is None:
            if not (corpus_dir / f"{c.stem}.md").is_file() and c.stem not in CANONICAL_STEMS:
                # a name that is not a corpus stem can never be fetched —
                # citing it is a wrong-document citation, not a fetch gap
                results.append(Result(c, "DOC_UNKNOWN",
                                      [f"'{c.doc_token}' is not a corpus document "
                                       f"(see INDEX.md for the pinned list)"]))
            elif (corpus_dir / f"{c.stem}.md").is_file():
                # fetched, but slab-mode/md-only: no JSON, no structural
                # grading — never a pass, never the agent's fault
                results.append(Result(c, "NO_PROVENANCE",
                                      [f"{c.stem}.md is fetched but carries no docling "
                                       f"JSON (md-only document — the TRM converts in "
                                       "slabs, a full JSON would be ~970 MB); citations "
                                       "to it cannot be verified structurally"]))
            else:
                results.append(Result(c, "DOC_MISSING",
                                      [f"{c.stem}.md not in {corpus_dir} "
                                       "(fetch.sh --full adds the rest)"]))
            continue

        # every section token must resolve; each contributes its page set;
        # the primary (first) token's pages are what the cited page must
        # belong to
        missing, page_sets = [], []
        for tok, kind in c.secs:
            sp = resolve_token(dox, tok, kind)
            if sp is not None:
                page_sets.append(sp)
            else:
                missing.append(tok)

        verdict, notes = "OK", []
        if missing:
            verdict = "SECTION_NOT_FOUND"
            notes.append(f"no {' or '.join(missing)} heading in {c.stem}.json")
        else:
            span = page_sets[0]
            span_txt = _fmt_pages(span.pages)
            if c.page not in span.pages:
                verdict = "PAGE_OUTSIDE_SECTION"
                notes.append(f"§{c.secs[0][0]} '{span.title}' spans {span_txt}, "
                             f"cited p. {c.page}")
            if c.quote is None:
                notes.append("no adjacent quote")
                if verdict == "OK":
                    verdict = "NO_QUOTE"
            else:
                if quote_needle(c.quote, wrap_table) not in _page_skeleton(
                        dox, c.page, wrap_table):
                    notes.append("quote does not occur on the cited page")
                    if verdict == "OK":
                        verdict = "QUOTE_NOT_AT_PAGE"
            if verdict == "OK":
                notes.append(f"§{c.secs[0][0]} '{span.title}' spans {span_txt}")
        results.append(Result(c, verdict, notes))
    return results


def exit_code(results: list[Result]) -> int:
    if any(r.verdict in FAIL_VERDICTS for r in results):
        return 1
    if any(r.verdict in SOFT_VERDICTS for r in results):
        return 2
    return 0


# --- CLI ----------------------------------------------------------------------

def _describe(c: Citation) -> str:
    doc = c.doc_token or c.stem or "<unnamed>"
    return f"{doc} {' '.join(tok for tok, _ in c.secs)} (p. {c.page})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Verify that citations in an agent answer really resolve "
                    "in the docling JSON source of truth (issue #29).",
        usage="%(prog)s [answer.md] [--corpus DIR] [--json]   (stdin when no file)",
    )
    ap.add_argument("answer", nargs="?", default="-",
                    help="file containing the agent answer (default: stdin)")
    ap.add_argument("--corpus", type=Path,
                    default=Path(__file__).resolve().parent / "md",
                    help="corpus directory of md/<doc>.json sources (default: md/ beside this file)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    if args.answer in ("-", ""):
        text = sys.stdin.read()
    else:
        text = Path(args.answer).read_text(encoding="utf-8", errors="replace")

    citations = parse_answer(text)
    results = grade(citations, args.corpus)
    code = exit_code(results)

    if args.json:
        print(json.dumps({
            "corpus": str(args.corpus),
            "citations": [
                {"citation": _describe(r.citation), "quote": r.citation.quote,
                 "verdict": r.verdict, "notes": r.notes}
                for r in results
            ],
            "exit": code,
        }, ensure_ascii=False, indent=2))
    else:
        if not results:
            print("no citations found — nothing to grade "
                  "(citation *rate* is the golden set's job, not this one)",
                  file=sys.stderr)
        for i, r in enumerate(results, 1):
            print(f"[{i}] {_describe(r.citation)}")
            if r.citation.quote:
                q = r.citation.quote if len(r.citation.quote) <= 72 else r.citation.quote[:69] + "…"
                print(f'    quote: "{q}"')
            print(f"    {r.verdict}" + (f" — {'; '.join(r.notes)}" if r.notes else ""))
    return code


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
