#!/usr/bin/env python3
"""Citation grader: is `doc §section (p. N) + quote` real, or invented?

Issue #22. Reads an agent answer (file argument or stdin), finds every
citation of the form `doc §section (p. N)` together with an adjacent quoted
line, and verifies it against the fetched corpus in md/ (gitignored;
hw-docs/fetch.sh materializes it — this script runs on the operator PC,
never on the Jetson):

  1. document resolves to md/<doc>.md               else DOC_MISSING
     (a name that is no corpus stem at all         else DOC_UNKNOWN —
      can never be fetched, so it fails like an invention)
  2. every cited section heading exists in the doc  else SECTION_NOT_FOUND
  3. cited page falls inside the section's span     else PAGE_OUTSIDE_SECTION
  4. quoted line really occurs on the cited page    else QUOTE_NOT_AT_PAGE

Check 3 is what catches a right-quote-wrong-section citation; check 4 what
catches a right-quote-wrong-page one (the p. 29 vs p. 28 bug from #20,
found by hand once — never again). A citation with no adjacent quote is
NO_QUOTE: the protocol in AGENTS.md/INDEX.md requires quoting the
load-bearing line, and an unquotable citation is unverifiable.

Section spans: a section covers the page its heading sits on, plus any
later page that still carries section content. pymupdf4llm emits a
running-header line at the top of each page, so a section whose *next*
section starts near a page top would otherwise claim that page too — a
page is only included when ≥ MIN_SPAN_CHARS of normalized text sit between
its anchor and the next heading (§3.4 spans pages 28-28, not 28-29).

Quote comparison normalizes both sides: NFKC, curly quotes/dashes folded,
`<br>` cell-wrap artifacts and markdown emphasis/link syntax removed, ALL
whitespace collapsed. pymupdf4llm breaks table cells mid-token
(`GP70_UART1_T<br>XD_BOOT2_STR<br>AP`) and mid-word-with-space
(`System<br>Sleep/Wake`), so no single whitespace treatment survives;
collapsing everything is the only comparison that stays honest on both.

Exit codes:
  0  every citation OK (an answer with no citations at all also exits 0,
     with a warning on stderr — judging citation *rate* is #24/#25's job)
  1  at least one citation fails verification: wrong section, wrong page,
     quote not at the page, no quote, or an unknown document
  2  only DOC_MISSING / NO_ANCHORS — a corpus that was never fetched (or
     converted without page anchors) is NOT a hallucinating agent;
     deliberately distinct from failure
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

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

# Section-token patterns; kind decides how a token maps to corpus headings.
SEC_PATTERNS = [
    # §3.4 / §2.1.1 / §3.1-3.8 (en dash or hyphen range)
    ("sec", re.compile(r"§\s*(?P<t>\d+(?:\.\d+)*(?:\s*[–-]\s*\d+(?:\.\d+)*)?)")),
    # Ch. 1 / Chapter 3 / ch3
    ("chapter", re.compile(r"\b[Cc]h(?:apter)?\.?\s*(?P<t>\d+)\b\.?")),
    # Table 3-4
    ("table", re.compile(r"\bTable\s+(?P<t>\d+\s*-\s*\d+)\b")),
]
PAGE_PAREN_RE = re.compile(r"\(\s*p\.\s*(\d+)\s*\)")
# The bare form must also fire inside "(devkit-carrier-spec.md §3.3 …, p. 26)"
# — answers wrap the whole citation in parens, so a following ")" means
# nothing. The lookbehind already stops it from re-matching inside "(p. N)".
PAGE_BARE_RE = re.compile(r"(?<![\w(])p\.\s*(\d+)\b")
QUOTE_RE = re.compile(r'"([^"]{2,400}?)"', re.S)
ANCHOR_RE = re.compile(r"<!--\s*p\.(\d+)\s*-->")
HEADING_RE = re.compile(r"(?m)^(#{1,6})\s+(.+?)[ \t]*$")
TABLE_LINE_RE = re.compile(r"(?m)^Table\s+(\d+)\s*-\s*(\d+)\s*[.:]")
PARA_SPLIT_RE = re.compile(r"\n\s*\n")

# How far a §section / doc token may sit from its (p. N) and still belong
# to it. Generous enough for "§3.4, Table 3-4 (p. 28)"; tight enough that a
# §token from a previous sentence is not kidnapped.
SEC_ATTACH_CHARS = 120
DOC_ATTACH_CHARS = 250
# Normalized chars of section content a page must carry (past its running
# header) to count as part of a section's span.
MIN_SPAN_CHARS = 100

HALLUCINATION_VERDICTS = {
    "SECTION_NOT_FOUND", "PAGE_OUTSIDE_SECTION", "QUOTE_NOT_AT_PAGE", "NO_QUOTE",
}
# NO_QUOTE is not literally an invention — but it is unverifiable, and the
# protocol requires the quote, so it fails the citation either way.
FAIL_VERDICTS = HALLUCINATION_VERDICTS | {"DOC_UNKNOWN"}
SOFT_VERDICTS = {"DOC_MISSING", "NO_ANCHORS"}


# --- text normalization ---------------------------------------------------

_FOLD = str.maketrans(
    {"“": '"', "”": '"', "„": '"', "‟": '"', "‘": "'", "’": "'",
     "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-", "‒": "-"},
)
_BR_RE = re.compile(r"<br\s*/?>", re.I)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_LINK_RE = re.compile(r"!?\[[^\]]*\]\([^)]*\)")


def normalize(s: str) -> str:
    """Fold a quote or a corpus span to its comparison skeleton."""
    s = unicodedata.normalize("NFKC", s).translate(_FOLD)
    s = _BR_RE.sub(" ", s)
    s = _COMMENT_RE.sub(" ", s)      # page anchors, picture-text markers
    s = _LINK_RE.sub(" ", s)         # keep link text, drop URLs
    s = s.replace("\\", "").replace("*", "").replace("`", "")
    return re.sub(r"\s+", "", s)


# --- answer parsing ---------------------------------------------------------

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
    `(p. N)` itself. Shared by the grader and the corpus linter."""
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


def quote_needle(quote: str) -> str:
    """A quote folded to its comparison skeleton, minus the answer's own
    terminal punctuation ("…nominal." for a corpus line that continues)."""
    n = normalize(quote).strip(".,;:!?…")
    return n or normalize(quote)


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


# --- corpus inspection ------------------------------------------------------

@dataclass
class Heading:
    level: int                      # 1-6 markdown rank; plain table captions rank 4
    kind: str                       # "num" | "chapter" | "table"
    label: str                      # "3.4" | "1" | "3-4"
    pos: int
    title: str


def headings(text: str) -> list[Heading]:
    """Headings with an identifiable number; unnumbered ones can neither be
    cited as §sections nor end another section's span."""
    out: list[Heading] = []
    for m in HEADING_RE.finditer(text):
        body = m.group(2).strip().strip("*").strip()
        cm = re.match(r"[Cc]hapter\s+(\d+)\b", body)
        tm = re.match(r"Table\s+(\d+)\s*-\s*(\d+)\b", body)
        nm = re.match(r"(\d+(?:\.\d+)*)\s", body + " ")
        if cm:
            out.append(Heading(len(m.group(1)), "chapter", cm.group(1), m.start(), body))
        elif tm:
            out.append(Heading(len(m.group(1)), "table", f"{tm.group(1)}-{tm.group(2)}", m.start(), body))
        elif nm:
            out.append(Heading(len(m.group(1)), "num", nm.group(1), m.start(), body))
    # Table captions the converter emitted as plain lines, not headings
    # (Table 3-3 in the carrier spec, Table 2-5 in the data sheet).
    for m in TABLE_LINE_RE.finditer(text):
        out.append(Heading(4, "table", f"{m.group(1)}-{m.group(2)}",
                           m.start(), text[m.start():m.end()].rstrip(".:")))
    out.sort(key=lambda h: h.pos)
    return out


def token_lookups(token: str, kind: str) -> list[tuple[str, str]]:
    """Section token -> the heading (kind, label) lookups it requires.

    A range §3.1-3.8 requires both endpoints; the citing page may sit
    anywhere in the combined span, which the caller assembles.
    """
    token = re.sub(r"\s*", "", token).replace("–", "-")
    if kind == "table":
        return [("table", token)]
    if kind == "chapter":
        return [("chapter", token)]
    if "-" in token:  # §3.1-3.8
        a, b = token.split("-", 1)
        return [("num", a), ("num", b)]
    if "." in token:
        return [("num", token)]
    return [("chapter", token)]  # a bare §3 refers to the chapter


def _section_span(text: str, anchors: list[tuple[int, int]], headings: list[Heading],
                  kind: str, label: str):
    """(pages, heading) for one heading, or None. `pages` is a set — an
    interior page that carries no section content (a running-header-only
    page inside the section) is not silently covered by min/max."""
    mine = [h for h in headings if h.kind == kind and h.label == label]
    if not mine or not anchors:
        return None
    h = mine[0]
    later = [x for x in headings if x.pos > h.pos and x.level <= h.level]
    end = later[0].pos if later else len(text)
    first = max((p for p, pos in anchors if pos <= h.pos), default=anchors[0][0])
    pages = {first}
    for p, pos in anchors:
        if h.pos < pos < end:
            # only the part of the page that lies inside this section:
            # bounded by the next anchor, else an empty interior page would
            # inherit the content of every later page. A page counts when it
            # carries real content — enough text, or any heading (a page
            # holding just "## 3.5 …" + one line is content; a page holding
            # only the running header is not; the next same-level heading is
            # already bounded out, so headings found here are this
            # section's own subsections).
            nxt = next((q for _, q in anchors if q > pos), len(text))
            sliver = text[pos:min(end, nxt)]
            if (HEADING_RE.search(sliver) or TABLE_LINE_RE.search(sliver)
                    or len(normalize(sliver)) >= MIN_SPAN_CHARS):
                pages.add(p)
    return pages, h


def page_text(text: str, anchors: list[tuple[int, int]], page: int) -> str | None:
    starts = [pos for p, pos in anchors if p == page]
    if not starts:
        return None
    s = starts[0]
    e = next((pos for _, pos in anchors if pos > s), len(text))
    return text[s:e]


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
    corpus_present = corpus_dir.is_dir() and any(corpus_dir.glob("*.md"))
    cache: dict[str, str | None] = {}

    def load(stem: str) -> str | None:
        if stem not in cache:
            cache[stem] = next(
                (c.read_text(encoding="utf-8", errors="replace")
                 for c in (corpus_dir / f"{stem}.md", corpus_dir / stem) if c.is_file()),
                None)
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
        text = load(c.stem)
        if text is None:
            if c.stem in CANONICAL_STEMS:
                results.append(Result(c, "DOC_MISSING",
                                      [f"{c.stem}.md not in {corpus_dir} "
                                       "(fetch.sh --full adds the rest)"]))
            else:
                # a name that is not a corpus stem can never be fetched —
                # citing it is a wrong-document citation, not a fetch gap
                results.append(Result(c, "DOC_UNKNOWN",
                                      [f"'{c.doc_token}' is not a corpus document "
                                       f"(see INDEX.md for the pinned list)"]))
            continue
        anchors = [(int(m.group(1)), m.start()) for m in ANCHOR_RE.finditer(text)]
        if not anchors:
            results.append(Result(c, "NO_ANCHORS",
                                  [f"{c.stem}.md has no <!-- p.N --> page anchors — "
                                   "converted with the pdftotext fallback?"]))
            continue
        heads = headings(text)

        # every section token must resolve; each contributes its page set,
        # merged across range endpoints; the primary (first) token's pages
        # are what the cited page must belong to
        missing, page_sets = [], []
        for tok, kind in c.secs:
            lookups = token_lookups(tok, kind)
            got = []
            for lk_kind, lk_label in lookups:
                sp = _section_span(text, anchors, heads, lk_kind, lk_label)
                if sp is None:
                    missing.append(tok)
                    break
                got.append(sp)
            if len(got) == len(lookups):
                if len(got) == 1:
                    page_sets.append((set(got[0][0]), got[0][1]))
                else:  # §3.1–3.8: everything between the endpoints counts
                    lo = min(p for pages, _ in got for p in pages)
                    hi = max(p for pages, _ in got for p in pages)
                    page_sets.append((set(range(lo, hi + 1)), got[0][1]))

        verdict, notes = "OK", []
        if missing:
            verdict = "SECTION_NOT_FOUND"
            notes.append(f"no {' or '.join(missing)} heading in {c.stem}.md")
        else:
            pages, h = page_sets[0]
            span_txt = _fmt_pages(pages)
            if c.page not in pages:
                verdict = "PAGE_OUTSIDE_SECTION"
                notes.append(f"§{c.secs[0][0]} '{h.title}' spans {span_txt}, "
                             f"cited p. {c.page}")
            if c.quote is None:
                notes.append("no adjacent quote")
                if verdict == "OK":
                    verdict = "NO_QUOTE"
            else:
                ptext = page_text(text, anchors, c.page)
                needle = quote_needle(c.quote)
                if ptext is None:
                    quote_bad = f"{c.stem}.md has no p.{c.page} anchor"
                elif needle not in normalize(ptext):
                    quote_bad = "quote does not occur on the cited page"
                else:
                    quote_bad = None
                if quote_bad:
                    notes.append(quote_bad)
                    if verdict == "OK":
                        verdict = "QUOTE_NOT_AT_PAGE"
            if verdict == "OK":
                notes.append(f"§{c.secs[0][0]} '{h.title}' spans {span_txt}")
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
                    "in the fetched hardware corpus (issue #22).",
        usage="%(prog)s [answer.md] [--corpus DIR] [--json]   (stdin when no file)",
    )
    ap.add_argument("answer", nargs="?", default="-",
                    help="file containing the agent answer (default: stdin)")
    ap.add_argument("--corpus", type=Path,
                    default=Path(__file__).resolve().parent / "md",
                    help="corpus directory of converted documents (default: md/ beside me)")
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
