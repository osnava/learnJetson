"""Tests for the v2 provenance linter (v2/lint.py, issue #30).

Two tiers, like v2/test_grade.py: a synthetic tier (a temp corpus built
from the committed docling fixture plus a synthetic INDEX) that runs
anywhere and carries the linter's contract in CI, and a real-corpus tier
that lints the actual INDEX.md against the fetched v2 corpus — skips
cleanly when unfetched, a skip is never a pass. URL checks are exercised
as pure functions here; CI runs them live via `v2/lint.py --urls-only`.

Run: python agent/hw-docs/v2/test_lint.py
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lint  # noqa: E402

FIXTURES = HERE / "fixtures"
REAL = HERE.parent / "md"

# Every routing shape INDEX.md uses: single §, range, §+Table, chapter,
# §name, middot pair, --full tier, glob, directory, and the non-section
# cells (whole doc / grep / sheets / em-dash).
SYNTH_INDEX = """# Synthetic INDEX

## Routing table

| Question about | Where (in `md/`) | Section |
|---|---|---|
| Single section | `demo-doc.md` | §1.1 |
| Range | `demo-doc.md` | §1.1–1.3 |
| Section plus table | `demo-doc.md` | §1.2 + Table 1-1 |
| Chapter form | `demo-doc.md` | Ch. 1 (§1.1–1.3) |
| Named section | `demo-doc.md` | §Encode |
| Two docs paired | `demo-doc.md` · `demo2.md` | §1.2 · §2.1 |
| Tier suffix | `demo2.md` (`--full`) | §2.1 |
| Pinmux glob | `demo-pinmux.*.csv` | sheets `…Pinmux_DP` / `…Pinmux_HDMI` |
| Whole doc | `demo-doc.md` | whole doc |
| TRM grep | `demo2.md` (`--full`) | grep — 7,000+ pages |
| Schematics grep | `demo2.md` | grep by net or refdes |
| Reference dir | `pdf/demo-reference/` (`--full`) | — |

Two answers worth memorizing (both verified against the corpus):

- Widget rail — `demo-doc.md` §1.1 (p. 1) states it plainly: "The widget
  supply rail is 3.3 V nominal."
- Widget modes — `demo-doc.md` §1.2 Table 1-1 (p. 2):
  "1|WIDGET_ERR|WIDGET_ERR: Multi word error text continues on this
  wrapped cell" is the error row.

## Pinned versions (URLs verified 2026-09-08)

| Document | Version | Source |
|---|---|---|
| Demo document | 0.9 | synthetic |
"""

# demo-doc.json's body items carry pages {1,2,3,5,6,7} — six of them; the
# synthetic "PDF" agrees. demo2 is md-only, tiled by two slab shards 1..4.
SYNTH_SPECS = {
    "demo-doc": {"pdf": "demo-doc.pdf", "heading_floor": 3},
    "demo2": {"pdf": "demo2.pdf", "md_only": True, "chunk_floor": 5},
}
SYNTH_PAGES = {"demo-doc.pdf": 6, "demo2.pdf": 4}


def synth_page_count(src: Path) -> int | None:
    return SYNTH_PAGES.get(Path(src).name)


def build_corpus(root: Path) -> Path:
    corpus = root / "md"
    (corpus / "index").mkdir(parents=True)
    shutil.copy(FIXTURES / "demo-doc.json", corpus / "demo-doc.json")
    (corpus / "demo-doc.md").write_text(
        "# Demo\n\n## 1.1 Widget Overview\n\nThe widget supply rail is "
        "3.3 V nominal.\n\n## 1.2 Status Registers\n\n| bit | meaning |\n"
        "\nSP-000_v0.9 | 4\n", encoding="utf-8")
    (corpus / "index" / "demo-doc.chunks.jsonl").write_text(
        json.dumps({"doc": "demo-doc"}) + "\n", encoding="utf-8")
    (corpus / "index" / "demo-doc.meta.json").write_text(
        json.dumps({"pages": "1-7", "chunks": 8}), encoding="utf-8")
    # demo2: md-only, tiled by two slab shards
    (corpus / "demo2.md").write_text(
        "# Demo2\n\n## 2.1 Header\n\nGrep-friendly net names.\n", encoding="utf-8")
    for shard, span, chunks in [("demo2.p1-2", "1-2", 3), ("demo2.p3-4", "3-4", 3)]:
        (corpus / "index" / f"{shard}.meta.json").write_text(
            json.dumps({"pages": span, "chunks": chunks}), encoding="utf-8")
        (corpus / "index" / f"{shard}.chunks.jsonl").write_text(
            json.dumps({"doc": "demo2"}) + "\n", encoding="utf-8")
    (corpus / "demo-pinmux.Sheet1.csv").write_text(",pin,ball\n", encoding="utf-8")
    return corpus


class CorpusCase(unittest.TestCase):
    """A temp corpus + helpers to run lint.main in-process."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.corpus = build_corpus(self.root)
        self.pdf = self.root / "pdf"
        self.pdf.mkdir()
        # placeholder "PDFs" — page counting is stubbed per file name
        for name in SYNTH_PAGES:
            (self.pdf / name).write_bytes(b"%PDF-1.4 synthetic")
        self.idx = self.root / "INDEX.md"
        self.idx.write_text(SYNTH_INDEX, encoding="utf-8")
        self.idx_text = SYNTH_INDEX

    def run_lint(self, index_text: str | None = None, specs=SYNTH_SPECS,
                 page_count=synth_page_count, corpus: Path | None = None):
        """lint.main over (optionally edited) INDEX text; returns (code, out)."""
        if index_text is not None and index_text != self.idx_text:
            self.idx.write_text(index_text, encoding="utf-8")
            self.idx_text = index_text
        out = io.StringIO()
        # the row floor guards the real INDEX against mass loss; the
        # synthetic table is small on purpose. Pin synonyms map the
        # synthetic table's Document cell to the demo stem.
        with contextlib.redirect_stdout(out), unittest.mock.patch.object(
                lint, "ROUTING_ROW_FLOOR", 8), unittest.mock.patch.object(
                lint, "DOC_SPECS", specs), unittest.mock.patch.object(
                lint, "PIN_SYNONYMS", [("Demo document", "demo-doc")]):
            code = lint.main(["--index", str(self.idx),
                              "--corpus", str(corpus or self.corpus),
                              "--pdf", str(self.pdf), "--offline"],
                             page_count=page_count)
        return code, out.getvalue()


class RoutingTest(CorpusCase):
    def test_clean_index_exits_zero_and_covers_every_shape(self):
        code, out = self.run_lint()
        self.assertEqual(code, 0, out)
        for needle in ("demo-doc 1.1", "demo-doc 1-1", "demo-doc 1",
                       "demo-doc Encode", "demo-pinmux.*.csv present"):
            self.assertIn(needle, out)
        # md-only docs are present but carry no JSON to resolve against
        self.assertIn("SKIP  Two docs paired: demo2 2.1 — md-only document", out)
        self.assertIn("SKIP  Tier suffix: demo2 2.1 — md-only document", out)
        # non-section cells skip, never fail
        self.assertEqual(out.count("non-section cell"), 5, out)

    def test_broken_section_reference_fails(self):
        code, out = self.run_lint(SYNTH_INDEX.replace("§1.2 + Table 1-1",
                                                      "§9.9 + Table 1-1"))
        self.assertEqual(code, 1)
        self.assertIn("FAIL  Section plus table: demo-doc 9.9", out)

    def test_broken_table_reference_fails(self):
        code, out = self.run_lint(SYNTH_INDEX.replace("§1.2 + Table 1-1",
                                                      "§1.2 + Table 9-9"))
        self.assertEqual(code, 1)
        self.assertIn("FAIL  Section plus table: demo-doc 9-9", out)

    def test_absent_doc_skips_never_passes(self):
        (self.corpus / "demo2.md").unlink()
        code, out = self.run_lint()
        self.assertEqual(code, 0, out)
        self.assertIn("SKIP  Tier suffix: demo2 (--full) — not fetched", out)

    def test_middot_row_with_one_doc_absent_never_validates_against_the_other(self):
        # the pairing must stay positional: the surviving doc must not
        # validate the absent doc's section by accident
        (self.corpus / "demo2.md").unlink()
        code, out = self.run_lint()
        self.assertIn("SKIP  Two docs paired: 2.1 — paired document not fetched", out)
        self.assertIn("demo-doc 1.2", out)   # the surviving half still checks
        self.assertNotIn("demo-doc 2.1", out)

    def test_directory_row_presence(self):
        code, out = self.run_lint()
        self.assertIn("SKIP  Reference dir: pdf/demo-reference/ (--full) — not fetched",
                      out)
        (self.pdf / "demo-reference").mkdir()
        code, out = self.run_lint()
        self.assertIn("PASS  Reference dir: pdf/demo-reference/ (--full) present", out)


class MemorizedTest(CorpusCase):
    def test_both_bullets_verified(self):
        code, out = self.run_lint()
        self.assertEqual(code, 0, out)
        self.assertIn("PASS  memorized #1 — 1 citation(s) verified", out)
        self.assertIn("PASS  memorized #2 — 1 citation(s) verified", out)

    def test_page_shifted_by_one_fails(self):
        bad = SYNTH_INDEX.replace("§1.1 (p. 1) states", "§1.1 (p. 2) states")
        code, out = self.run_lint(bad)
        self.assertEqual(code, 1)
        self.assertIn("FAIL  memorized #1", out)
        self.assertIn("PAGE_OUTSIDE_SECTION", out)

    def test_missing_block_fails(self):
        code, out = self.run_lint(re.sub(r"(?m)^Two answers.*(?=## Pinned)",
                                         "", SYNTH_INDEX, flags=re.S))
        self.assertEqual(code, 1)
        self.assertIn("FAIL  memorized answers — the block is gone", out)

    def test_bullet_without_citation_fails(self):
        bad = SYNTH_INDEX.replace(
            "- Widget rail — `demo-doc.md` §1.1 (p. 1) states it plainly:",
            "- Widget rail — the docs say the rail is 3.3 V nominal:")
        code, out = self.run_lint(bad)
        self.assertEqual(code, 1)
        self.assertIn("no gradeable citation", out)


class VersionsTest(CorpusCase):
    PATTERNS = {"demo-doc": (r"SP-000_v(\d+\.\d+)", "demo footer")}

    def run_versions(self, pin: str):
        idx = SYNTH_INDEX.replace("| Demo document | 0.9 | synthetic |",
                                  f"| Demo document | {pin} | synthetic |")
        rep = lint.Report()
        with unittest.mock.patch.object(lint, "PIN_SYNONYMS",
                                        [("Demo document", "demo-doc")]):
            lint.lint_versions(idx, self.corpus, rep, patterns=self.PATTERNS)
        return rep

    def test_pin_matches_rendering(self):
        rep = self.run_versions("0.9")
        self.assertEqual(rep.failed, 0, rep.lines)
        self.assertIn("PASS  pin demo-doc — 0.9 pinned, 0.9 in demo footer",
                      rep.lines)

    def test_pin_with_letter_suffix_matches_core(self):
        rep = self.run_versions("0.9p")
        self.assertEqual(rep.failed, 0, rep.lines)
        self.assertIn("0.9p pinned, 0.9 in demo footer", rep.lines[0])

    def test_version_bump_fails(self):
        rep = self.run_versions("1.1")
        self.assertEqual(rep.failed, 1)
        self.assertIn("INDEX pins 1.1 but the document on disk is 0.9",
                      rep.lines[0])

    def test_document_shape_change_fails(self):
        (self.corpus / "demo-doc.md").write_text("# Demo\n\nno footer\n",
                                                 encoding="utf-8")
        rep = self.run_versions("0.9")
        self.assertEqual(rep.failed, 1)
        self.assertIn("version pattern not found", rep.lines[0])

    def test_unfetched_document_skips(self):
        (self.corpus / "demo-doc.md").unlink()
        rep = self.run_versions("0.9")
        self.assertEqual((rep.passed, rep.failed, rep.skipped), (0, 0, 1))
        self.assertIn("not fetched", rep.lines[0])


class ProvenanceTest(CorpusCase):
    def test_clean_corpus_pages_and_shards(self):
        code, out = self.run_lint()
        self.assertEqual(code, 0, out)
        self.assertIn("PASS  pages demo-doc — JSON provenance covers 6/6 PDF pages",
                      out)
        self.assertIn("PASS  pages demo2 — shards tile 1-4 of 4 PDF pages", out)
        self.assertIn("PASS  purity — no v1-era renderings in md/", out)

    def test_json_page_shortfall_fails(self):
        specs = {**SYNTH_SPECS}
        code, out = self.run_lint(specs=specs,
                                  page_count=lambda src: 8)  # PDF grew 2 pages
        self.assertEqual(code, 1)
        self.assertIn("FAIL  pages demo-doc — JSON provenance covers 6 of 8", out)

    def test_missing_search_shard_fails(self):
        (self.corpus / "index" / "demo-doc.chunks.jsonl").unlink()
        code, out = self.run_lint()
        self.assertEqual(code, 1)
        self.assertIn("fetched without a search shard", out)

    def test_md_without_json_fails(self):
        (self.corpus / "demo-doc.json").unlink()
        code, out = self.run_lint()
        self.assertEqual(code, 1)
        self.assertIn("md present but no docling JSON", out)

    def test_heading_floor_breach_fails(self):
        specs = {**SYNTH_SPECS, "demo-doc": {**SYNTH_SPECS["demo-doc"],
                                             "heading_floor": 99}}
        code, out = self.run_lint(specs=specs)
        self.assertEqual(code, 1)
        self.assertIn("8 heading objects, floor 99", out)

    def test_size_floor_breach_fails(self):
        specs = {**SYNTH_SPECS, "demo-doc": {**SYNTH_SPECS["demo-doc"],
                                             "size_kb": 500}}
        code, out = self.run_lint(specs=specs)
        self.assertEqual(code, 1)
        self.assertIn("floor 500 KB", out)

    def test_shard_gap_fails(self):
        (self.corpus / "index" / "demo2.p1-2.meta.json").unlink()
        code, out = self.run_lint()
        self.assertEqual(code, 1)
        self.assertIn("do not tile from p.1", out)

    def test_shard_coverage_short_of_pdf_fails(self):
        code, out = self.run_lint(page_count=lambda src: 5)  # PDF has 5, shards 4
        self.assertEqual(code, 1)
        self.assertIn("shards tile 1-4 but the PDF has 5 pages", out)

    def test_chunk_floor_breach_fails(self):
        specs = {**SYNTH_SPECS, "demo2": {**SYNTH_SPECS["demo2"],
                                          "chunk_floor": 99}}
        code, out = self.run_lint(specs=specs)
        self.assertEqual(code, 1)
        self.assertIn("6 chunks, floor 99", out)

    def test_mid_grind_slabs_skip_with_progress(self):
        specs = {**SYNTH_SPECS,
                 "demo3": {"pdf": "demo3.pdf", "md_only": True}}
        (self.pdf / "demo3.pdf").write_bytes(b"%PDF-1.4 synthetic")
        (self.corpus / "demo3.p1-2.raw.md").write_text("# raw slab\n",
                                                       encoding="utf-8")
        (self.corpus / "index" / "demo3.p1-2.meta.json").write_text(
            json.dumps({"pages": "1-2", "chunks": 2}), encoding="utf-8")
        code, out = self.run_lint(specs=specs,
                                  page_count=lambda src: {"demo3.pdf": 6}.get(
                                      Path(src).name, synth_page_count(src)))
        self.assertEqual(code, 0, out)
        self.assertIn("SKIP  provenance demo3 — slab grind in progress — "
                      "shards through p.2 of 6", out)

    def test_unfetched_doc_skips(self):
        specs = {**SYNTH_SPECS, "ghost": {"pdf": "ghost.pdf"}}
        code, out = self.run_lint(specs=specs)
        self.assertEqual(code, 0, out)
        self.assertIn("SKIP  provenance ghost — not fetched", out)

    def test_v1_anchor_remnant_fails(self):
        (self.corpus / "demo-doc.md").write_text(
            "# Demo\n\n<!-- p. 1 -->\n\nold pymupdf rendering\n", encoding="utf-8")
        code, out = self.run_lint()
        self.assertEqual(code, 1)
        self.assertIn("FAIL  purity demo-doc.md — v1 pymupdf rendering", out)

    def test_no_page_counter_skips_never_passes(self):
        code, out = self.run_lint(page_count=lambda src: None)
        self.assertEqual(code, 0, out)
        self.assertIn("SKIP  pages demo-doc — no page counter", out)
        self.assertIn("SKIP  pages demo2 — no page counter", out)


class UrlLogicTest(unittest.TestCase):
    """Pure-function tier — CI runs the live HEADs via --urls-only."""

    def test_classify_url(self):
        self.assertEqual(lint.classify_url(200, "application/pdf", False),
                         ("PASS", "application/pdf"))
        self.assertEqual(lint.classify_url(200, "text/html", True),
                         ("PASS", "login page (expected)"))
        self.assertEqual(lint.classify_url(200, "text/html", False)[0], "FAIL")
        self.assertEqual(lint.classify_url(503, "application/pdf", False)[0], "FAIL")
        # the gated item handing out the PDF directly means the gate moved
        self.assertEqual(lint.classify_url(200, "application/pdf", True)[0], "FAIL")

    def test_manifest_parse(self):
        items = lint.parse_manifest(HERE.parent / "fetch.sh")
        self.assertEqual(len(items), 8)  # 7 ITEMS rows + the reference-design zip
        gates = {n: g for n, _, g in items}
        self.assertTrue(gates["datasheet"])
        self.assertFalse(any(g for n, g in gates.items() if n != "datasheet"))
        self.assertEqual(items[-1][0], "devkit-carrier-reference-design")
        self.assertTrue(items[-1][1].endswith(".zip"))

    def test_urls_only_mode_uses_the_manifest(self):
        with unittest.mock.patch.object(lint, "lint_urls") as lu:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = lint.main(["--urls-only"])
        self.assertEqual(code, 0)
        lu.assert_called_once()
        self.assertIn("summary:", out.getvalue())


def needs(*stems: str):
    return unittest.skipUnless(
        all((REAL / f"{s}.json").is_file() for s in stems),
        "v2 JSON not fetched: " + ", ".join(
            s for s in stems if not (REAL / f"{s}.json").is_file()))


real_corpus = unittest.skipUnless(
    any(REAL.glob("*.json")),
    "v2 corpus (docling JSON) not fetched — run agent/hw-docs/fetch.sh")


@real_corpus
class RealCorpusTest(unittest.TestCase):
    """The actual INDEX.md against the fetched v2 corpus (issue #30's
    acceptance: the map re-verified row by row against docling provenance).

    The TRM grinds in slabs for hours — while it runs, its checks SKIP
    (progress noted), which keeps the run green without faking coverage.
    """

    def lint_real(self, *edit):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = lint.main(["--offline"] + list(edit))
        return code, out.getvalue()

    def test_real_index_lints_clean(self):
        code, out = self.lint_real()
        self.assertEqual(code, 0, out)
        self.assertIn("0 failed", out)
        self.assertGreaterEqual(out.count("\nPASS  ") + out.count("PASS  "), 60)

    @needs("datasheet", "devkit-carrier-spec", "orin-nx-nano-design-guide",
           "orin-thermal-design-guide", "orin-pin-function-names")
    def test_every_routing_row_resolves(self):
        code, out = self.lint_real()
        self.assertIn("PASS  routing table (27 data rows)", out)
        fails = [l for l in out.splitlines() if l.startswith("FAIL")]
        self.assertEqual(fails, [], "\n".join(fails))

    @needs("datasheet", "devkit-carrier-spec")
    def test_memorized_answers_verified_at_their_pages(self):
        code, out = self.lint_real()
        self.assertIn("PASS  memorized #1 — 1 citation(s) verified", out)
        self.assertIn("PASS  memorized #2 — 1 citation(s) verified", out)

    @needs("datasheet", "devkit-carrier-spec", "orin-nx-nano-design-guide",
           "orin-thermal-design-guide", "orin-pin-function-names")
    def test_json_provenance_equals_pdf_pages(self):
        code, out = self.lint_real()
        for stem, pages in [("datasheet", "54/54"), ("devkit-carrier-spec", "38/38"),
                            ("orin-nx-nano-design-guide", "97/97"),
                            ("orin-pin-function-names", "21/21"),
                            ("orin-thermal-design-guide", "42/42")]:
            self.assertIn(f"PASS  pages {stem} — JSON provenance covers {pages} "
                          "PDF pages", out)

    def test_page_shifted_memorized_citation_is_caught(self):
        idx = (HERE.parent / "INDEX.md").read_text(encoding="utf-8")
        self.assertIn("§3.4 Table 3-4 (p. 28):", idx)
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "INDEX.md"
            bad.write_text(idx.replace("§3.4 Table 3-4 (p. 28):",
                                       "§3.4 Table 3-4 (p. 29):"), encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = lint.main(["--offline", "--index", str(bad)])
        self.assertEqual(code, 1)
        self.assertIn("FAIL  memorized #2", out.getvalue())
        self.assertIn("p. 29", out.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
