#!/usr/bin/env python3
"""Provenance-based corpus linter over the v2 docling substrate (issue #30).

Successor to the v1 pymupdf linter (check.py, #23, deleted in this sweep).
The question is unchanged — does the hand-maintained map in INDEX.md still
match the fetched corpus, which rots on a schedule NVIDIA controls? — but
every corpus check is written from scratch against docling output (the
#27/#28 hard wall: nothing adapted from the anchor-counting machinery):

  1. routing     every §/Ch./Table token in the routing table resolves
                 against the cited document's JSON heading/table registries
                 (grade.py's structural resolution — no md-text regexes)
  2. memorized   the "worth memorizing" bullets grade OK through the v2
                 grader: section object exists, page inside its span, quote
                 at that page
  3. versions    the version string each docling rendering carries vs the
                 INDEX pin (pattern table below, derived from the v2 md)
  4. provenance  JSON page provenance vs the source PDF: body-item page set
                 == PDF page count (core docs); md-only docs (TRM,
                 schematics) tile their slab shards 1..N with N == PDF
                 pages; heading/size/chunk floors; search shard present;
                 no v1-era `<!-- p.N -->` remnant anywhere in md/
  5. urls        fetch.sh manifest HEADs — HTML on a direct document is the
                 stale-URL signature; the login-gated datasheet must answer
                 its login page

Verdicts PASS / FAIL / SKIP. A document that is not fetched — or a check
this machine cannot run (source PDF not cached, no page-counter library,
TRM grind in progress) — is SKIP, never PASS: a corpus that was never
fetched must not paint the run green. Exit 0 when nothing failed (skips
allowed), 1 on any FAIL.

Run:  python v2/lint.py               (operator-side, after fetch.sh)
      python v2/lint.py --urls-only   (the corpus-independent CI tier)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import grade  # noqa: E402  (v2 structural grader: resolution + parse/grade)

HERE = Path(__file__).resolve().parent          # v2/
ROOT = HERE.parent                              # hw-docs/

# The one thing that can never come from docling: v1's page anchors. Any
# file in md/ still carrying them is a leftover pymupdf rendering that
# every other check would silently verify against — fail it by name.
V1_ANCHOR_RE = re.compile(r"<!--\s*p\.\s*\d+\s*-->")

# A routing table with fewer rows than this has lost content, not rows.
ROUTING_ROW_FLOOR = 20

# Per-document facts, as data (tests inject their own table).
#
#   pdf          source PDF under pdf/ (schematics convert from the
#                reference-design zip, not pdf/<stem>.pdf)
#   md_only      no docling JSON by design (TRM: a full JSON would be
#                ~970 MB; schematics: md + chunks only) — pages verified
#                through shard provenance instead
#   heading_floor  section_header objects in the JSON (core docs) or
#                  `#`-headings in the md (md-only), at ~2/3 of the counts
#                  measured on the healthy 2026-09-08 corpus — far above
#                  zero, below honest re-conversion variance
#   size_kb      md rendering size floor (smallest healthy core conversion
#                today is 51 KB)
#   chunk_floor  total chunks across shards, md-only docs
#
# The TRM heading floor: 12,288 `#`-headings in the raw parts through
# p.6250 (1.97/page) extrapolate to ~17k over the 8,783-page document;
# 8,000 is its "structure survived" bar, set before the grind completed.
DOC_SPECS: dict[str, dict] = {
    "datasheet": {"pdf": "datasheet.pdf", "heading_floor": 75, "size_kb": 100},
    "devkit-carrier-spec": {"pdf": "devkit-carrier-spec.pdf",
                            "heading_floor": 27, "size_kb": 70},
    "orin-nx-nano-design-guide": {"pdf": "orin-nx-nano-design-guide.pdf",
                                  "heading_floor": 85, "size_kb": 280},
    "orin-pin-function-names": {"pdf": "orin-pin-function-names.pdf",
                                "heading_floor": 15, "size_kb": 30},
    "orin-thermal-design-guide": {"pdf": "orin-thermal-design-guide.pdf",
                                  "heading_floor": 57, "size_kb": 70},
    "orin-trm": {"pdf": "orin-trm.pdf", "md_only": True,
                 "heading_floor": 8000, "size_kb": 8192},
    "devkit-carrier-schematics": {
        "pdf": "devkit-carrier-reference-design/P3768_A04_Concept_schematics.pdf",
        "md_only": True, "size_kb": 60, "chunk_floor": 120},
}

# Where the pinned version lives in each docling rendering. Written from
# scratch against the v2 md (2026-09-08, grepping each rendering for its
# document number): document-number footers survive docling conversion as
# furniture text in the md rendering, so six patterns land on the same
# strings the PDFs themselves carry — convergent with the v1 era, not
# carried over from it. The one true re-derivation is the TRM: pymupdf
# rendered its title page as `**Version: 1.2**`, docling writes
# `Version:\xa01.2` (no-break space — `\s` covers it, `**` gone).
VERSION_PATTERNS = {
    "datasheet": (r"DS-11105-001_v(\d+\.\d+)", "data-sheet footer"),
    "devkit-carrier-spec": (r"SP-11324-001_v(\d+\.\d+)", "spec footer"),
    "orin-nx-nano-design-guide": (r"DG-10931-001_v(\d+\.\d+)", "design-guide footer"),
    "orin-thermal-design-guide": (r"TDG-11127-001_v(\d+\.\d+)", "thermal-guide footer"),
    "orin-pin-function-names": (r"DA-11434-001_v?(\d+\.\d+)", "title block"),
    "orin-trm": (r"Version:\s*(\d+\.\d+)", "TRM title page"),
    # the pinmux "document" is the workbook: its Customer-Readme sheet
    # opens with `1.2,Date,Revision,Description`
    "orin-pinmux": (r"(?m)^(\d+\.\d+),Date,Revision", "Customer-Readme sheet"),
}

# Pinned-versions table Document cell -> corpus stem (more specific keys
# first — two of the rows are "… Design Guide"). The reference-design row
# pins a board rev (`A04`), not a semver; its evidence is the rev string
# inside the converted schematics.
PIN_SYNONYMS = [
    ("Thermal Design Guide", "orin-thermal-design-guide"),
    ("Design Guide", "orin-nx-nano-design-guide"),
    ("Data Sheet", "datasheet"),
    ("Carrier Board Spec", "devkit-carrier-spec"),
    ("Pinmux", "orin-pinmux"),
    ("Function Names", "orin-pin-function-names"),
    ("TRM", "orin-trm"),
    ("reference design", "devkit-carrier-reference-design"),
]

URL_TIMEOUT_S = 15


class Report:
    def __init__(self) -> None:
        self.passed = self.failed = self.skipped = 0
        self.lines: list[str] = []

    def record(self, verdict: str, label: str, note: str = "") -> None:
        if verdict == "PASS":
            self.passed += 1
        elif verdict == "FAIL":
            self.failed += 1
        elif verdict == "SKIP":
            self.skipped += 1
        else:
            # an unknown verdict must crash, not silently count as a skip —
            # "a skip is never a pass" only works if skips are deliberate
            raise ValueError(f"unknown verdict {verdict!r} for {label!r}")
        self.lines.append(f"{verdict}  {label}" + (f" — {note}" if note else ""))

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0


# --- INDEX.md parsing --------------------------------------------------------

def block_after(text: str, heading: str) -> str:
    """Content under a `## heading` up to the next `## ` heading."""
    m = re.search(rf"(?m)^{re.escape(heading)}[^\n]*\n(.*?)(?=^## |\Z)", text, re.S)
    return m.group(1) if m else ""


def data_rows(block: str) -> list[list[str]]:
    out: list[list[str]] = []
    for line in block.splitlines():
        if line.startswith("|") and not re.match(r"^\|[-\s|:]+\|$", line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not out or len(cells) == len(out[0]):
                out.append(cells)
    return out[1:]  # header dropped; separator rows already filtered


def memorized_bullets(idx: str) -> list[str]:
    m = re.search(r"(?m)^\w+ answers worth memorizing[^\n]*\n(.*?)(?=^## |\Z)",
                  idx, re.S)
    if not m:
        return []
    bullets = [ln[2:].strip() for ln in re.split(r"(?m)^(?=- )", m.group(1))
               if ln.startswith("- ")]
    return [" ".join(b.split()) for b in bullets if b]


def index_pins(idx: str) -> dict[str, str]:
    """Pinned-versions table rows keyed by corpus stem (full pin token,
    letter suffix included — `1.2p`, `A04`)."""
    pins: dict[str, str] = {}
    for row in data_rows(block_after(idx, "## Pinned versions")):
        doc_cell, ver_cell = row[0], row[1]
        stem = next((s for key, s in PIN_SYNONYMS if key.lower() in doc_cell.lower()),
                    None)
        m = re.match(r"[\dA-Z][\d.]*[a-zA-Z]*", ver_cell)
        if stem and m:
            pins[stem] = m.group(0)
    return pins


# --- 1. routing table --------------------------------------------------------

def doc_cell_docs(cell: str) -> list[tuple[str, bool, str | None]]:
    """Doc cell -> [(token, is_full_tier, stem-or-None)]. Shapes: `stem.md`,
    `stem.md` (--full), two docs middot-joined, `stem.*.csv` glob,
    `pdf/dir/` directory (stem None for glob/dir — presence only)."""
    out = []
    for part in cell.split("·"):
        m = re.search(r"`([^`]+)`", part)
        if not m:
            continue
        token = m.group(1)
        if "*" in token or "/" in token:
            out.append((token, "--full" in part, None))
        else:
            stem = token.removesuffix(".md")
            out.append((stem, "--full" in part, stem))
    return out


def is_section_cell(cell: str) -> bool:
    c = cell.strip(" —–")
    return not (c.startswith(("whole doc", "grep", "sheets")) or c == "")


def lint_routing(idx: str, corpus: Path, root: Path, rep: Report) -> None:
    block = block_after(idx, "## Routing table")
    if not block:
        rep.record("FAIL", "routing table", "no `## Routing table` block in INDEX.md")
        return
    rows = data_rows(block)
    if len(rows) < ROUTING_ROW_FLOOR:
        rep.record("FAIL", "routing table",
                   f"{len(rows)} data rows, floor is {ROUTING_ROW_FLOOR} — rows lost?")
    else:
        rep.record("PASS", f"routing table ({len(rows)} data rows)")

    dox_cache: dict[str, grade.DocIndex | None] = {}

    def doc(stem: str):
        if stem not in dox_cache:
            dox_cache[stem] = grade.load_doc(corpus / f"{stem}.json")
        return dox_cache[stem]

    def resolves(stem: str, tok: str, kind: str) -> bool:
        return grade.resolve_token(doc(stem), tok, kind) is not None

    for row in rows:
        q, doc_cell, sec_cell = row[0], row[1], row[2]
        docs = doc_cell_docs(doc_cell)
        present: list[tuple[str, str] | None] = []   # (stem, state); None = absent
        for token, full, stem in docs:
            label = f"{q[:38]}: {token}" + (" (--full)" if full else "")
            if stem is None and "*" in token:            # glob artifact
                if list(corpus.glob(token)):
                    rep.record("PASS", f"{label} present")
                else:
                    rep.record("SKIP", label, "not fetched (fetch.sh [--full])")
                present.append(None)
                continue
            if stem is None:                             # pdf/…/ directory
                d = root / token
                if d.is_dir():
                    rep.record("PASS", f"{label} present")
                else:
                    rep.record("SKIP", label, "not fetched (fetch.sh [--full])")
                present.append(None)
                continue
            state = ("json" if (corpus / f"{stem}.json").is_file()
                     else "md" if (corpus / f"{stem}.md").is_file() else "absent")
            if state == "absent":
                rep.record("SKIP", label, "not fetched (fetch.sh [--full])")
                present.append(None)
            else:
                rep.record("PASS", f"{label} present")
                present.append((stem, state))

        if not is_section_cell(sec_cell):
            rep.record("SKIP", f"{q[:38]}: sections", "non-section cell")
            continue
        secs = grade.sections_in(sec_cell)
        if not secs:
            rep.record("SKIP", f"{q[:38]}: sections", "no §/Ch./Table tokens")
            continue
        # middot rows pair docs and sections positionally; a section whose
        # paired doc is absent is SKIPped — never checked against the row's
        # *other* document, which would validate it by accident
        if len(present) == len(secs):
            pairs = list(zip(present, secs))
        else:
            pairs = [(slot, sec) for slot in present for sec in secs]
        checked = 0
        for slot, (_, _, tok, kind) in pairs:
            if slot is None:
                rep.record("SKIP", f"{q[:38]}: {tok}",
                           "paired document not fetched" if len(present) > 1
                           else "document not fetched")
                continue
            stem, state = slot
            if state != "json":
                rep.record("SKIP", f"{q[:38]}: {stem} {tok}",
                           "md-only document — no JSON provenance to resolve against")
                continue
            checked += 1
            hit = resolves(stem, tok, kind)
            rep.record("PASS" if hit else "FAIL", f"{q[:38]}: {stem} {tok}",
                       "" if hit else f"no {tok} heading object in {stem}.json")
        if not checked:
            rep.record("SKIP", f"{q[:38]}: sections",
                       "no JSON-bearing document to resolve against")


# --- 2. memorized answers ----------------------------------------------------

def lint_memorized(idx: str, corpus: Path, rep: Report) -> None:
    bullets = memorized_bullets(idx)
    if not bullets:
        rep.record("FAIL", "memorized answers", "the block is gone from INDEX.md")
        return
    for n, bullet in enumerate(bullets, 1):
        label = f"memorized #{n}"
        citations = grade.parse_answer(bullet)
        if not citations:
            rep.record("FAIL", label,
                       "no gradeable citation — the bullet must carry "
                       "`doc §section (p. N)` + a \"load-bearing quote\"")
            continue
        results = grade.grade(citations, corpus)
        soft = [r for r in results if r.verdict in grade.SOFT_VERDICTS]
        hard = [r for r in results if r.verdict not in grade.SOFT_VERDICTS
                and r.verdict != "OK"]
        if hard:
            for r in hard:
                rep.record("FAIL", label,
                           f"{r.citation.secs[0][0]} (p. {r.citation.page}) "
                           f"— {r.verdict}: {'; '.join(r.notes)}")
        elif soft:
            rep.record("SKIP", label,
                       f"{len(soft)}/{len(results)} citation(s) unfetchable — "
                       + "; ".join(r.notes[0] for r in soft if r.notes))
        else:
            rep.record("PASS", label,
                       f"{len(results)} citation(s) verified (section, page, quote)")


# --- 3. pinned versions ------------------------------------------------------

def lint_versions(idx: str, corpus: Path, rep: Report,
                  patterns: dict = VERSION_PATTERNS) -> None:
    pins = index_pins(idx)
    if not pins:
        rep.record("FAIL", "pinned versions", "no pins parsed from INDEX.md")
        return

    # the reference-design pin is a board rev; its evidence is the rev
    # string inside the converted schematics
    if "devkit-carrier-reference-design" in pins:
        label = "pin devkit-carrier-reference-design"
        rev = pins["devkit-carrier-reference-design"]
        sch = corpus / "devkit-carrier-schematics.md"
        if not sch.is_file():
            rep.record("SKIP", label, "schematics not fetched (fetch.sh --full)")
        elif re.search(rev.replace("-", "[-_]"),
                       sch.read_text(encoding="utf-8", errors="replace"), re.I):
            rep.record("PASS", label, f"{rev} pinned, present in schematics")
        else:
            rep.record("FAIL", label, f"{rev} not found in the converted schematics")

    for stem, (pat, where) in patterns.items():
        label = f"pin {stem}"
        if stem == "orin-pinmux":
            csvs = sorted(corpus.glob("orin-pinmux.*Readme*.csv"))
            path = csvs[0] if csvs else None
        else:
            path = corpus / f"{stem}.md"
        if path is None or not path.is_file():
            if list(corpus.glob(f"{stem}.p*.raw.md")):
                rep.record("SKIP", label, "slab grind in progress — "
                           "pin verified when the md is assembled")
            else:
                rep.record("SKIP", label, "not fetched — cannot verify the pin")
            continue
        m = re.search(pat, path.read_text(encoding="utf-8", errors="replace"))
        if not m:
            rep.record("FAIL", label,
                       f"version pattern not found in {where} — document changed "
                       "shape, or a different version landed")
            continue
        extracted = m.group(1)
        pin = pins.get(stem)
        if pin is None:
            rep.record("FAIL", label, "stem missing from the pinned-versions table")
        elif pin == extracted or pin.startswith(extracted):
            rep.record("PASS", label, f"{pin} pinned, {extracted} in {where}")
        else:
            rep.record("FAIL", label,
                       f"INDEX pins {pin} but the document on disk is {extracted} "
                       "(NVIDIA bumped it — update INDEX.md + fetch.sh together)")


# --- 4. provenance -----------------------------------------------------------

def pdf_page_count(src: Path) -> int | None:
    """Pages of a cached source PDF, or None when no counting library is
    importable (pypdfium2 ships with docling; pypdf is the pure-python
    fallback). The caller SKIPs on None — never a pass."""
    try:
        import pypdfium2 as pdfium
        return len(pdfium.PdfDocument(str(src)))
    except ImportError:
        try:
            from pypdf import PdfReader
            return len(PdfReader(str(src)).pages)
        except ImportError:
            return None


def _md_headings(path: Path) -> int:
    return len(re.findall(r"(?m)^#{1,6} ",
                          path.read_text(encoding="utf-8", errors="replace")))


def lint_purity(corpus: Path, rep: Report) -> None:
    """No v1-era rendering may sit in md/ — a pymupdf leftover would
    silently satisfy every md-based check below."""
    stale = [p.name for p in sorted(corpus.glob("*.md"))
             if V1_ANCHOR_RE.search(p.read_text(encoding="utf-8", errors="replace"))]
    if stale:
        for name in stale:
            rep.record("FAIL", f"purity {name}",
                       "v1 pymupdf rendering (page anchors) still on disk — "
                       "v2 checks must not verify against it; re-run fetch.sh")
    else:
        rep.record("PASS", "purity", "no v1-era renderings in md/")


def _shard_spans(corpus: Path, stem: str) -> list[tuple[int, int]]:
    """Sorted (first, last) page spans of a doc's shards, from meta files."""
    spans = []
    for meta_f in sorted((corpus / "index").glob(f"{stem}*.meta.json")):
        try:
            a, b = json.loads(meta_f.read_text(encoding="utf-8"))["pages"].split("-")
        except (ValueError, KeyError):
            continue
        spans.append((int(a), int(b)))
    return sorted(spans)


def lint_provenance(corpus: Path, pdf_dir: Path, rep: Report,
                    page_count=pdf_page_count,
                    specs: dict | None = None) -> None:
    specs = specs if specs is not None else DOC_SPECS
    lint_purity(corpus, rep)

    for stem, spec in specs.items():
        label = f"provenance {stem}"
        md = corpus / f"{stem}.md"
        src = pdf_dir / spec["pdf"]
        # md-only docs grind in slabs; leftover *.raw.md parts mean it is
        # still going — that state is checked before "not fetched", so a
        # resumable grind reports progress, not absence
        parts = (list(corpus.glob(f"{stem}.p*.raw.md"))
                 if spec.get("md_only") else [])

        if stem == "orin-pinmux":  # the workbook: md + per-sheet CSVs
            if not md.is_file():
                rep.record("SKIP", "provenance orin-pinmux", "not fetched")
            elif not list(corpus.glob("orin-pinmux.*.csv")):
                rep.record("FAIL", "provenance orin-pinmux",
                           "md present but sheet CSVs missing — re-run fetch.sh")
            else:
                rep.record("PASS", "provenance orin-pinmux", "md + sheet CSVs")
            continue

        if parts:
            spans = _shard_spans(corpus, stem)
            done = spans[-1][1] if spans else 0
            total = page_count(src) if src.is_file() else None
            rep.record("SKIP", label,
                       f"slab grind in progress — shards through p.{done}"
                       + (f" of {total}" if total else ""))
            continue
        if not md.is_file():
            rep.record("SKIP", label, "not fetched")
            continue
        size_kb = md.stat().st_size // 1024
        if spec.get("size_kb") and size_kb < spec["size_kb"]:
            rep.record("FAIL", label, f"{size_kb} KB md, floor {spec['size_kb']} KB — "
                       "conversion came up empty; re-run fetch.sh")
            continue

        if not spec.get("md_only"):
            # core doc: the JSON is the truth — its body-item page set must
            # cover the PDF exactly, and the search shard must exist
            json_f = corpus / f"{stem}.json"
            dox = grade.load_doc(json_f) if json_f.is_file() else None
            if dox is None:
                rep.record("FAIL", label, "md present but no docling JSON — "
                           "re-run fetch.sh (the JSON is the source of truth)")
                continue
            if not (corpus / "index" / f"{stem}.chunks.jsonl").is_file():
                rep.record("FAIL", label, "fetched without a search shard — "
                           "route→search→declare breaks at step 2; re-run fetch.sh")
                continue
            heads = len(dox.headings)
            if spec.get("heading_floor") and heads < spec["heading_floor"]:
                rep.record("FAIL", label,
                           f"{heads} heading objects, floor {spec['heading_floor']} — "
                           "conversion lost structure; re-run fetch.sh")
                continue
            if not src.is_file():
                rep.record("SKIP", f"pages {stem}", "source PDF not cached")
                rep.record("PASS", label, f"{heads} headings, {size_kb} KB md")
                continue
            n_pdf = page_count(src)
            if n_pdf is None:
                rep.record("SKIP", f"pages {stem}",
                           "no page counter (pypdfium2/pypdf) installed")
                rep.record("PASS", label, f"{heads} headings, {size_kb} KB md")
                continue
            json_pages: set[int] = set()
            for it in dox.items:
                json_pages |= grade.item_pages(it)
            if len(json_pages) == n_pdf:
                rep.record("PASS", f"pages {stem}",
                           f"JSON provenance covers {n_pdf}/{n_pdf} PDF pages")
                rep.record("PASS", label, f"{heads} headings, {size_kb} KB md")
            else:
                rep.record("FAIL", f"pages {stem}",
                           f"JSON provenance covers {len(json_pages)} of {n_pdf} PDF "
                           "pages — re-run fetch.sh and eyeball the diff")
            continue

        # md-only doc: pages verified through shard provenance. Complete =
        # md written (build.py only assembles it from a full slab set) AND
        # no leftover parts (the in-progress case SKIPped above).
        spans = _shard_spans(corpus, stem)
        if not spans:
            rep.record("FAIL", label, "md present but no index shards — "
                       "re-run fetch.sh")
            continue
        gaps = [(spans[i][1] + 1, spans[i + 1][0] - 1)
                for i in range(len(spans) - 1) if spans[i + 1][0] != spans[i][1] + 1]
        if spans[0][0] != 1 or gaps:
            rep.record("FAIL", f"pages {stem}",
                       f"shard page spans do not tile from p.1 (gaps: {gaps})")
            continue
        covered = spans[-1][1]
        if not src.is_file():
            rep.record("SKIP", f"pages {stem}", "source PDF not cached")
        else:
            n_pdf = page_count(src)
            if n_pdf is None:
                rep.record("SKIP", f"pages {stem}",
                           "no page counter (pypdfium2/pypdf) installed")
            elif covered != n_pdf:
                rep.record("FAIL", f"pages {stem}",
                           f"shards tile 1-{covered} but the PDF has {n_pdf} pages")
                continue
            else:
                rep.record("PASS", f"pages {stem}",
                           f"shards tile 1-{covered} of {n_pdf} PDF pages")
        chunks = 0
        for meta_f in (corpus / "index").glob(f"{stem}*.meta.json"):
            chunks += json.loads(meta_f.read_text(encoding="utf-8")).get("chunks", 0)
        if spec.get("chunk_floor") and chunks < spec["chunk_floor"]:
            rep.record("FAIL", label, f"{chunks} chunks, floor {spec['chunk_floor']}")
            continue
        if spec.get("heading_floor"):
            heads = _md_headings(md)
            if heads < spec["heading_floor"]:
                rep.record("FAIL", label,
                           f"{heads} md headings, floor {spec['heading_floor']} — "
                           "conversion lost structure")
                continue
        rep.record("PASS", label, f"{size_kb} KB md, {chunks} chunks")


# --- 5. URL HEAD checks ------------------------------------------------------

def parse_manifest(fetch_sh: Path) -> list[tuple[str, str, bool]]:
    """fetch.sh's ITEMS rows as (name, url, gated), plus the reference-design
    zip URL that lives outside ITEMS, in the --full block."""
    text = fetch_sh.read_text(encoding="utf-8")
    out = [(m.group(1), m.group(2), m.group(3) == "login")
           for m in re.finditer(r'^\s*"([^|]+)\|([^|]+)\|[^|]+\|[^|]+\|([^"]*)"\s*$',
                                text, re.M)]
    zip_m = re.search(r'"(https://[^"]*reference_design[^"]*)"', text)
    if zip_m:
        out.append(("devkit-carrier-reference-design", zip_m.group(1), False))
    return out


def classify_url(status: int, ctype: str, gated: bool) -> tuple[str, str]:
    """(verdict, note) for one URL response: HTML on a direct document is
    the stale-URL signature; the gated item's login page is expected."""
    is_html = ctype.lower().startswith("text/html")
    if status != 200:
        return "FAIL", f"HTTP {status}"
    if is_html and not gated:
        return "FAIL", "HTML on a direct document — stale URL"
    if gated and not is_html:
        return "FAIL", "gated item stopped returning the login page — check the gate"
    return "PASS", "login page (expected)" if gated else (ctype or "200")


def lint_urls(fetch_sh: Path, rep: Report) -> None:
    import urllib.request
    items = parse_manifest(fetch_sh)
    if not items:
        rep.record("FAIL", "urls", "could not parse the fetch.sh manifest")
        return
    for name, url, gated in items:
        try:
            req = urllib.request.Request(url, method="HEAD",
                                         headers={"User-Agent": "learnJetson-corpus-lint"})
            with urllib.request.urlopen(req, timeout=URL_TIMEOUT_S) as r:
                status, ctype = r.status, r.headers.get("Content-Type", "")
            if status in (403, 405):  # some CDNs dislike HEAD — poke 1 byte
                raise OSError("head rejected")
        except OSError:
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "learnJetson-corpus-lint", "Range": "bytes=0-99"})
                with urllib.request.urlopen(req, timeout=URL_TIMEOUT_S) as r:
                    status, ctype = r.status, r.headers.get("Content-Type", "")
            except OSError as e:
                rep.record("FAIL", f"url {name}", f"{e}")
                continue
        verdict, note = classify_url(status, ctype, gated)
        rep.record(verdict, f"url {name}", note)


# --- CLI ---------------------------------------------------------------------

def main(argv: list[str] | None = None, page_count=pdf_page_count) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", type=Path, default=ROOT / "INDEX.md")
    ap.add_argument("--corpus", type=Path, default=ROOT / "md")
    ap.add_argument("--pdf", type=Path, default=ROOT / "pdf")
    ap.add_argument("--fetch-sh", type=Path, default=ROOT / "fetch.sh")
    ap.add_argument("--offline", action="store_true", help="skip the URL HEAD checks")
    ap.add_argument("--urls-only", action="store_true",
                    help="run only the URL HEAD checks (the CI tier)")
    args = ap.parse_args(argv)

    rep = Report()
    if args.urls_only:
        lint_urls(args.fetch_sh, rep)
    else:
        if not args.index.is_file():
            print(f"FAIL  INDEX.md not found at {args.index}", file=sys.stderr)
            return 1
        idx = args.index.read_text(encoding="utf-8", errors="replace")
        lint_routing(idx, args.corpus, args.pdf.parent, rep)
        lint_memorized(idx, args.corpus, rep)
        lint_versions(idx, args.corpus, rep)
        lint_provenance(args.corpus, args.pdf, rep, page_count=page_count)
        if args.offline:
            rep.record("SKIP", "urls", "--offline")
        else:
            lint_urls(args.fetch_sh, rep)

    print("\n".join(rep.lines))
    print(f"\nsummary: {rep.passed} passed, {rep.failed} failed, {rep.skipped} skipped "
          f"(skip = not fetched / not verifiable here — never a pass)")
    return rep.exit_code


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
