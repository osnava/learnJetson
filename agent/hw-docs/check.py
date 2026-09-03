#!/usr/bin/env python3
"""Static corpus linter — does the hand-maintained map still match reality?

Issue #23. INDEX.md is a hand-written map over a corpus that rots on a
schedule NVIDIA controls: URLs go stale, versions bump per JetPack
release, section numbers move, and a pymupdf upgrade can change table
extraction or drop page anchors. Three routing rows were wrong on the
first pass and caught by a human; this runs the same checks in seconds,
on every push, at 2am too.

Five checks, all deterministic (no model in the loop):

  1. routing table   every §section / Ch. / Table reference in the
                     `## Routing table` block resolves to a real heading
                     in the cited document (reuses grade.py's heading
                     machinery). Only that block is parsed — the pinned-
                     versions table is not routing rows.
  2. pinned versions the version string embedded in each on-disk
                     document (per-document pattern table below) matches
                     the pin in INDEX.md — size never tells you that.
                     The TRM pin is `1.2p`, its text says `1.2`; a pin
                     may carry a letter suffix past the extracted core.
  3. memorized answers  the "worth memorizing" bullets: italic *"…"*
                     strings must occur at the cited page; non-path
                     backticked tokens likewise. Bare "…" quotes are
                     section paths, `*.md` / `../*` backticks are
                     filenames — neither is evidence (verified traps).
  4. conversion smoke  `<!-- p.N -->` anchor count == source PDF page
                     count (exact today, incl. the 8,783-page TRM);
                     heading count above a per-document floor (catches a
                     silent pdftotext downgrade); no document mostly empty.
  5. URL HEAD        every fetch.sh manifest URL answers 200; non-gated
                     items must NOT be HTML (that is the stale-URL
                     signature), the login-gated datasheet MUST be (it
                     redirects to a login page until fetched by hand).

Verdicts are PASS / FAIL / SKIP. An absent document is SKIP, never PASS —
a corpus that was never fetched must not paint the build green. Exit 0
when nothing failed (skips allowed), 1 on any FAIL. `--offline` skips
check 5; everything else is local.

Run: python check.py   (or ./check.sh)   — see check.sh for the CI path.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grade  # noqa: E402  (heading/anchor/normalize machinery, shared)

HERE = Path(__file__).resolve().parent

# Per-document version patterns: how the pinned version is embedded in the
# converted text (the seven documents do not share one regex — verified
# 2026-09). Group 1 extracts the version core.
VERSION_PATTERNS = {
    "datasheet": (r"DS-11105-001_v(\d+\.\d+)", "data-sheet footer"),
    "devkit-carrier-spec": (r"SP-11324-001_v(\d+\.\d+)", "spec footer"),
    "orin-nx-nano-design-guide": (r"DG-10931-001_v(\d+\.\d+)", "design-guide footer"),
    "orin-pin-function-names": (r"DA-11434-001_v?(\d+\.\d+)", "title block"),
    "orin-thermal-design-guide": (r"TDG-11127-001_v(\d+\.\d+)", "thermal-guide footer"),
    "orin-trm": (r"\*\*Version:\s*(\d+\.\d+)\*\*", "TRM title page"),
    # the pinmux "document" is the workbook: its Customer-Readme sheet
    # opens with `,Revision History` then `1.2,Date,Revision,Description`
    "orin-pinmux": (r"(?m)^(\d+\.\d+),Date,Revision", "Customer-Readme CSV"),
}

# Pinned-versions table row -> corpus stem (matched against the Document
# cell; more specific keys first — two of the rows are "… Design Guide").
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

# Conversion smoke floors. Headings: ~2/3 of today's counts, far above
# zero (the pdftotext fallback emits none). Size: the smallest healthy
# conversion today is 33 KB; 15 KB separates "converted" from "empty".
HEADING_FLOORS = {
    "datasheet": 60,
    "devkit-carrier-spec": 25,
    "orin-nx-nano-design-guide": 80,
    "orin-pin-function-names": 15,
    "orin-thermal-design-guide": 40,
    "orin-trm": 6000,
    "devkit-carrier-schematics": 4,
}
SIZE_FLOOR_KB = 15
# Schematics convert from the reference-design zip, not pdf/<stem>.pdf.
SPECIAL_SOURCES = {
    "devkit-carrier-schematics": "devkit-carrier-reference-design/P3768_A04_Concept_schematics.pdf",
}
# A routing table with fewer rows than this has lost content, not rows.
ROUTING_ROW_FLOOR = 20
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
        else:
            self.skipped += 1
        self.lines.append(f"{verdict}  {label}" + (f" — {note}" if note else ""))

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0


# --- shared parsing helpers -------------------------------------------------

def block_under(text: str, heading_prefix: str) -> str:
    """The text after a `## heading` line, up to the next `## ` heading."""
    m = re.search(rf"(?m)^{re.escape(heading_prefix)}[^\n]*\n(.*?)(?=^## |\Z)", text, re.S)
    return m.group(1) if m else ""


def data_rows(block: str) -> list[list[str]]:
    out = []
    for line in block.splitlines():
        if line.startswith("|") and not re.match(r"^\|[-\s|:]+\|$", line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not out or len(cells) == len(out[0]):
                out.append(cells)
    return out[1:]  # drop the header row; separator rows already filtered


# --- 1. routing table -------------------------------------------------------

def doc_cell_artifacts(cell: str, corpus: Path, root: Path) -> list[tuple[str, str | None, list[Path]]]:
    """Document cell -> [(label, stem-or-None, artifacts)]. Shapes: plain
    `stem.md`, tier suffix `stem.md` (`--full`), two docs middot-joined,
    glob `stem.*.csv`, directory `pdf/…/` (resolved against `root`)."""
    out = []
    for part in cell.split("·"):
        m = re.search(r"`([^`]+)`", part)
        if not m:
            continue
        token = m.group(1)
        tier = " (--full)" if "--full" in part else ""
        if "*" in token:                                   # glob
            out.append((f"{token}{tier}", None, sorted(corpus.glob(token))))
        elif "/" in token:                                # directory, e.g. pdf/<name>/
            d = root / token
            out.append((f"{token.rstrip('/')}{tier}", None, [d] if d.is_dir() else []))
        else:                                              # stem[.md]
            stem = token.removesuffix(".md")
            hit = [p for p in (corpus / f"{stem}.md", corpus / stem) if p.is_file()]
            out.append((f"{stem}{tier}", stem, hit))
    return out


def is_section_cell(cell: str) -> bool:
    c = cell.strip(" —–")
    return not (c.startswith("whole doc") or c.startswith("grep")
                or c.startswith("sheets") or c == "")


def lint_routing(idx: str, corpus: Path, pdf_root: Path, rep: Report) -> None:
    block = block_under(idx, "## Routing table")
    rows = data_rows(block)
    if not block:
        rep.record("FAIL", "routing table", "no `## Routing table` block in INDEX.md")
        return
    if len(rows) < ROUTING_ROW_FLOOR:
        rep.record("FAIL", "routing table",
                   f"{len(rows)} data rows, floor is {ROUTING_ROW_FLOOR} — rows lost?")
    else:
        rep.record("PASS", f"routing table ({len(rows)} data rows)")

    texts: dict[str, str] = {}
    headings: dict[str, list] = {}

    def loaded(stem: str):
        if stem not in texts:
            p = corpus / f"{stem}.md"
            texts[stem] = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
            headings[stem] = grade.headings(texts[stem]) if texts[stem] else []
        return texts[stem]

    def resolves(stem: str, claim) -> bool:
        loaded(stem)
        return all(any(h.kind == k and h.label == l for h in headings[stem])
                   for _, _, tok, kind in claim
                   for k, l in grade.token_lookups(tok, kind))

    for row in rows:
        q, doc_cell, sec_cell = row[0], row[1], row[2]
        artifacts = doc_cell_artifacts(doc_cell, corpus, pdf_root)
        present: list[str | None] = []   # positional doc slots; None = absent
        for label, stem, hits in artifacts:
            if hits:
                rep.record("PASS", f"{q[:38]}: {label} present")
                if stem and (corpus / f"{stem}.md").is_file():
                    present.append(stem)
            else:
                rep.record("SKIP", f"{q[:38]}: {label}", "not fetched (fetch.sh [--full])")
                present.append(None)
        if not is_section_cell(sec_cell):
            rep.record("SKIP", f"{q[:38]}: sections", "non-section cell")
            continue
        secs = grade.sections_in(sec_cell)
        if not secs:
            rep.record("SKIP", f"{q[:38]}: sections", "no §-tokens to resolve")
            continue
        # middot rows pair docs and sections positionally; a section whose
        # paired doc is absent is SKIPPed — never checked against the row's
        # *other* document, which would validate it by accident
        if len(artifacts) == len(secs):
            pairs = list(zip(present, secs))
        else:
            pairs = [(stem, sec) for stem in present for sec in secs]
        checked = 0
        for stem, (_, _, tok, kind) in pairs:
            if stem is None:
                rep.record("SKIP", f"{q[:38]}: {tok}",
                           "document not fetched" if len(artifacts) == 1
                           else "paired document not fetched")
                continue
            checked += 1
            hit = resolves(stem, [(0, 0, tok, kind)])
            rep.record("PASS" if hit else "FAIL",
                       f"{q[:38]}: {stem} {tok}",
                       "" if hit else f"no {tok} heading in {stem}.md")
        if not checked:
            rep.record("SKIP", f"{q[:38]}: sections", "no fetched document to resolve against")


# --- 2. pinned versions -----------------------------------------------------

def lint_versions(idx: str, corpus: Path, rep: Report) -> None:
    block = block_under(idx, "## Pinned versions")
    if not block:
        rep.record("FAIL", "pinned versions", "no `## Pinned versions` block in INDEX.md")
        return
    # the pin keeps its full token (`1.2p`, `A04`) so a letter suffix is
    # visible in the comparison, not truncated away before it
    pins: dict[str, str] = {}
    for row in data_rows(block):
        doc_cell, ver_cell = row[0], row[1]
        stem = next((s for key, s in PIN_SYNONYMS if key.lower() in doc_cell.lower()), None)
        if stem:
            m = re.match(r"[\dA-Z][\d.]*[a-zA-Z]*", ver_cell)
            if m:
                pins[stem] = m.group(0)
    # the reference design pins a board rev (`A04`), not a semver: its
    # evidence is the rev string inside the converted schematics
    if "devkit-carrier-reference-design" in pins:
        label = "pin devkit-carrier-reference-design"
        rev = pins["devkit-carrier-reference-design"]
        sch = corpus / "devkit-carrier-schematics.md"
        if not sch.is_file():
            rep.record("SKIP", label, "schematics not fetched (--full)")
        elif re.search(rev.replace("-", "[-_]"), sch.read_text(encoding="utf-8",
                                                               errors="replace"), re.I):
            rep.record("PASS", label, f"{rev} pinned, present in schematics")
        else:
            rep.record("FAIL", label, f"{rev} not found in the converted schematics")
    for stem, (pat, where) in VERSION_PATTERNS.items():
        label = f"pin {stem}"
        if stem == "orin-pinmux":
            csvs = sorted(corpus.glob("orin-pinmux.*Readme*.csv"))
            path = csvs[0] if csvs else None
        else:
            path = corpus / f"{stem}.md"
        if path is None or not path.is_file():
            rep.record("SKIP", label, "not fetched — cannot verify the pin")
            continue
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        m = re.search(pat, text)
        if not m:
            rep.record("FAIL", label, f"version pattern not found in {where} — document "
                                      "changed shape, or a different version landed")
            continue
        extracted = m.group(1)
        pin = pins.get(stem)
        if pin is None:
            rep.record("FAIL", label, f"stem not found in the pinned-versions table")
        elif pin == extracted or pin.startswith(extracted):
            rep.record("PASS", label, f"{pin} pinned, {extracted} in {where}"
                      + (f" (pin carries {''.join(ch for ch in pin if ch.isalpha())} "
                         "past the document's version)" if pin != extracted else ""))
        else:
            rep.record("FAIL", label,
                       f"INDEX pins {pin} but the document on disk is {extracted} "
                       "(NVIDIA bumped it — update INDEX.md + fetch.sh together)")


# --- 3. memorized answers -----------------------------------------------------

ITALIC_QUOTE_RE = re.compile(r'\*"([^"]+)"\*')
BACKTICK_RE = re.compile(r"`([^`]+)`")


def _path_shaped(tok: str) -> bool:
    return "/" in tok or tok.endswith((".md", ".csv"))


def lint_memorized(idx: str, corpus: Path, rep: Report) -> None:
    m = re.search(r"(?m)^Two answers worth memorizing[^\n]*\n(.*?)(?=^## |\Z)", idx, re.S)
    if not m:
        rep.record("FAIL", "memorized answers", "the block is gone from INDEX.md")
        return
    bullets = [ln[2:].strip()
               for ln in re.split(r"(?m)^(?=- )", m.group(1)) if ln.startswith("- ")]
    bullets = [" ".join(b.split()) for b in bullets if b]
    if len(bullets) < 2:
        rep.record("FAIL", "memorized answers", f"{len(bullets)} bullet(s), expected both")
        return

    corpus_stems = [p.stem for p in sorted(corpus.glob("*.md"))]
    for n, bullet in enumerate(bullets, 1):
        label = f"memorized #{n}"
        pages = grade.page_cites(bullet)
        secs = grade.sections_in(bullet)
        italics = ITALIC_QUOTE_RE.findall(bullet)
        tokens = [t for t in BACKTICK_RE.findall(bullet) if not _path_shaped(t)]
        if not pages or not secs:
            rep.record("SKIP", label, "no citation to verify")
            continue
        evidence = [grade.normalize(q).strip(".,;:!?…") for q in italics] + \
                   [grade.normalize(t).strip("*") for t in tokens]
        if not evidence:
            rep.record("SKIP", label, "no italic quote, no non-path tokens — nothing checkable")
            continue
        # explicit doc = first doc token that names a real document; a
        # path backtick (`../inventory.md`) is a filename, not a source
        explicit_stem = None
        for dm in grade.DOC_RE.finditer(bullet):
            s = grade._resolve_stem(dm.group(0))
            if s in grade.CANONICAL_STEMS or s in corpus_stems:
                explicit_stem = s
                break
        for pm, claim in zip(pages, grade.claim_sections(pages, secs)):
            if not claim:
                continue
            page = int(pm.group(1))
            cited = ", ".join(tok for _, _, tok, _ in claim)
            if explicit_stem and not (corpus / f"{explicit_stem}.md").is_file():
                rep.record("SKIP", f"{label} ({cited}, p. {page})",
                           f"{explicit_stem}.md not fetched")
                continue
            candidates = corpus_stems
            if explicit_stem:
                # the named doc only constrains the search when the cited
                # sections live in it — a "see also `other.md`" backtick is
                # a pointer, not the source
                etext = (corpus / f"{explicit_stem}.md").read_text(encoding="utf-8",
                                                                  errors="replace")
                if all(any(h.kind == k and h.label == l for h in grade.headings(etext))
                       for _, _, tok, kind in claim
                       for k, l in grade.token_lookups(tok, kind)):
                    candidates = [explicit_stem]
            resolving = []
            for stem in dict.fromkeys(candidates):
                p = corpus / f"{stem}.md"
                if not p.is_file():
                    continue
                text = p.read_text(encoding="utf-8", errors="replace")
                anchors = [(int(a.group(1)), a.start()) for a in grade.ANCHOR_RE.finditer(text)]
                if all(any(h.kind == k and h.label == l for h in grade.headings(text))
                       for _, _, tok, kind in claim
                       for k, l in grade.token_lookups(tok, kind)):
                    page_text = grade.page_text(text, anchors, page)
                    if page_text is not None and all(
                            ev in grade.normalize(page_text) for ev in evidence):
                        resolving.append(stem)
            if len(resolving) == 1:
                rep.record("PASS", f"{label} ({cited}, p. {page})",
                           f"evidence verified in {resolving[0]}.md")
            elif not resolving:
                rep.record("FAIL", f"{label} ({cited}, p. {page})",
                           "evidence not found at the cited page in any corpus document")
            else:
                rep.record("FAIL", f"{label} ({cited}, p. {page})",
                           f"ambiguous — evidence matches {resolving}; name the document")


# --- 4. conversion smoke -------------------------------------------------------

def lint_conversion(corpus: Path, pdf_dir: Path, rep: Report) -> None:
    for stem, floor in HEADING_FLOORS.items():
        md = corpus / f"{stem}.md"
        if not md.is_file():
            rep.record("SKIP", f"smoke {stem}", "not fetched")
            continue
        text = md.read_text(encoding="utf-8", errors="replace")
        if md.stat().st_size < SIZE_FLOOR_KB * 1024:
            rep.record("FAIL", f"smoke {stem}", "document is mostly empty")
            continue
        heads = len(grade.HEADING_RE.findall(text))
        if heads < floor:
            rep.record("FAIL", f"smoke {stem}",
                       f"{heads} headings, floor {floor} — converted with the "
                       "pdftotext fallback? (pip install pymupdf4llm, re-run fetch.sh)")
            continue
        rep.record("PASS", f"smoke {stem}", f"{heads} headings, "
                     f"{md.stat().st_size // 1024} KB")
        src = pdf_dir / SPECIAL_SOURCES.get(stem, f"{stem}.pdf")
        if not src.is_file():
            rep.record("SKIP", f"pages {stem}", "source PDF not cached — cannot count")
            continue
        try:
            import pymupdf
        except ImportError:
            rep.record("SKIP", f"pages {stem}", "pymupdf not installed")
            continue
        n_pages = pymupdf.open(str(src)).page_count
        n_anchors = len(grade.ANCHOR_RE.findall(text))
        if n_anchors == n_pages:
            rep.record("PASS", f"pages {stem}", f"{n_anchors} anchors == {n_pages} PDF pages")
        else:
            rep.record("FAIL", f"pages {stem}",
                       f"{n_anchors} anchors vs {n_pages} PDF pages — either a converter "
                       "regression dropped anchors, or the PDF gained a text-empty page "
                       "(convert.py skips blank pages); re-run fetch.sh and eyeball the diff")


# --- 5. URL HEAD checks ---------------------------------------------------------

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


def classify_url(name: str, status: int, ctype: str, gated: bool) -> tuple[str, str]:
    """(verdict, note) for one URL response — HTML on a non-gated item is
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
            if status in (403, 405):  # some CDNs dislike HEAD — poke with 1 byte
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
        verdict, note = classify_url(name, status, ctype, gated)
        rep.record(verdict, f"url {name}", note)


# --- CLI ------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", type=Path, default=HERE / "INDEX.md")
    ap.add_argument("--corpus", type=Path, default=HERE / "md")
    ap.add_argument("--pdf", type=Path, default=HERE / "pdf")
    ap.add_argument("--fetch-sh", type=Path, default=HERE / "fetch.sh")
    ap.add_argument("--offline", action="store_true", help="skip the URL HEAD checks")
    args = ap.parse_args(argv)

    idx = args.index.read_text(encoding="utf-8", errors="replace") if args.index.is_file() else ""
    if not idx:
        print(f"FAIL  INDEX.md not found at {args.index}", file=sys.stderr)
        return 1

    rep = Report()
    lint_routing(idx, args.corpus, args.pdf.parent, rep)
    lint_versions(idx, args.corpus, rep)
    lint_memorized(idx, args.corpus, rep)
    lint_conversion(args.corpus, args.pdf, rep)
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
