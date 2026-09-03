"""Tests for check.py — issue #23 acceptance criteria.

Two tiers, like test_grade.py: a synthetic tier (fixtures/md + a synthetic
INDEX built in-memory) that runs anywhere and exercises every routing-table
shape, and a real-corpus tier that lints the actual INDEX.md against the
fetched corpus, including the deliberately-broken variants the issue asks
for. URL checks are never exercised here (--offline) — CI runs them live.
"""
from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check  # noqa: E402

FIXTURES = HERE / "fixtures"
SYNTH = FIXTURES / "md"
REAL = HERE / "md"

real_corpus = unittest.skipUnless(
    any(REAL.glob("*.md")), "corpus not fetched (run agent/hw-docs/fetch.sh)")

# Every shape from the issue: 5 document-cell forms, 8 section-cell forms,
# 5 non-section cells (sheets / whole doc / two greps / em-dash).
SYNTH_INDEX = """# Synthetic INDEX

## Routing table

| Question about | Where (in `md/`) | Section |
|---|---|---|
| Single section | `demo-doc.md` | §1.1 |
| Range | `demo-doc.md` | §1.1–1.3 |
| Out-of-order list | `demo-doc.md` | §1.3, §1.1 |
| Mixed list and range | `demo-doc.md` | §1.1, §1.2–1.3 |
| Section plus table | `demo-doc.md` | §1.2 + Table 1-1 |
| Chapter form | `demo-doc.md` | Ch. 1 (§1.1–1.3) |
| Prose-annotated | `demo-doc.md` | §1.3 (Table 1-1); pull-ups §1.1–1.2 |
| Tier suffix | `demo2.md` (`--full`) | §2.1 |
| Two docs paired | `demo-doc.md` · `demo2.md` | §1.2 · §2.1 |
| Pinmux glob | `demo-pinmux.*.csv` | sheets `…Pinmux_DP` / `…Pinmux_HDMI` |
| Whole doc | `demo-doc.md` | whole doc |
| TRM grep | `demo2.md` (`--full`) | grep — 7,000+ pages |
| Schematics grep | `demo-doc.md` | grep by net or refdes |
| Reference dir | `pdf/demo-reference/` (`--full`) | — |

Two answers worth memorizing (both verified against the corpus):

- Widget rail — `demo-doc.md` Ch. 1 (p. 1) states it plainly: *"The widget
  supply rail is 3.3 V nominal."* The section label "Widget Basics" is a
  path, not evidence.
- Widget modes — §1.2 Table 1-1 (p. 2): `WIDGET_RDY` and `WIDGET_ERR` are
  the mode pins; see also `demo2.md` and `../inventory.md`.

## Pinned versions (URLs verified 2026-09-03)

| Document | Version | Source |
|---|---|---|
| Demo document | 1.0 | synthetic |
"""


def run_lint(index_text: str, corpus: Path, pdf_dir: Path) -> tuple[int, str]:
    """check.main in-process against a temp INDEX; return (exit, output)."""
    with tempfile.TemporaryDirectory() as tmp:
        idx = Path(tmp) / "INDEX.md"
        idx.write_text(index_text, encoding="utf-8")
        out = io.StringIO()
        # the row floor guards the real INDEX against mass loss; synthetic
        # tables are small on purpose
        with contextlib.redirect_stdout(out), unittest.mock.patch.object(
                check, "ROUTING_ROW_FLOOR", 5):
            code = check.main(["--index", str(idx), "--corpus", str(corpus),
                               "--pdf", str(pdf_dir), "--offline"])
    return code, out.getvalue()


def lint_real(index_text: str | None = None) -> tuple[int, str]:
    """The real INDEX.md (optionally edited) against the fetched corpus."""
    text = index_text if index_text is not None else         (HERE / "INDEX.md").read_text(encoding="utf-8")
    return run_lint(text, REAL, HERE / "pdf")


class SyntheticTier(unittest.TestCase):
    """Deterministic shape coverage against fixtures/md."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.pdf = self.root / "pdf"
        self.pdf.mkdir()
        self.addCleanup(self.tmp.cleanup)

    def test_clean_run_passes_every_shape(self):
        code, out = run_lint(SYNTH_INDEX, SYNTH, self.pdf)
        self.assertEqual(code, 0, out)
        self.assertNotIn("FAIL", out)
        # the shape pairings the issue calls out
        self.assertIn("demo-doc 1.2", out)          # middot pairing, first doc
        self.assertIn("demo2 2.1", out)             # middot pairing, second doc
        self.assertIn("demo-doc 1", out)            # chapter form
        self.assertIn("demo-doc 1.3", out)          # prose-annotated multi-section
        self.assertIn("demo-doc 1-1", out)          # table token
        self.assertIn("demo-pinmux.*.csv present", out)
        # non-section cells skip, never fail
        self.assertEqual(out.count("non-section cell"), 5, out)
        self.assertIn("summary:", out)

    def test_absent_directory_row_skips_never_passes(self):
        code, out = run_lint(SYNTH_INDEX, SYNTH, self.pdf)
        self.assertIn("SKIP  Reference dir: pdf/demo-reference (--full)", out)
        (self.pdf / "demo-reference").mkdir()
        code, out = run_lint(SYNTH_INDEX, SYNTH, self.pdf)
        self.assertIn("PASS  Reference dir: pdf/demo-reference (--full) present", out)

    def test_broken_section_reference_fails(self):
        bad = SYNTH_INDEX.replace("§1.2 + Table 1-1", "§9.9 + Table 1-1")
        code, out = run_lint(bad, SYNTH, self.pdf)
        self.assertEqual(code, 1)
        self.assertIn("FAIL  Section plus table: demo-doc 9.9", out)

    def test_memorized_page_shifted_by_one_fails(self):
        bad = SYNTH_INDEX.replace("Table 1-1 (p. 2):", "Table 1-1 (p. 3):")
        code, out = run_lint(bad, SYNTH, self.pdf)
        self.assertEqual(code, 1)
        self.assertIn("FAIL  memorized #2", out)
        self.assertIn("p. 3", out)

    def test_memorized_evidence_and_traps(self):
        code, out = run_lint(SYNTH_INDEX, SYNTH, self.pdf)
        self.assertEqual(code, 0, out)
        # italic quote verified in the named doc; token rule verified by
        # inference in the unnamed one; see-also/path backticks no-ops
        self.assertIn("memorized #1 (1, p. 1)", out)
        self.assertIn("memorized #2 (1.2, 1-1, p. 2)", out)
        self.assertIn("evidence verified in demo-doc.md", out)
        self.assertNotIn("FAIL", out)

    def test_manifest_parse(self):
        items = check.parse_manifest(HERE / "fetch.sh")
        self.assertEqual(len(items), 8)  # 7 ITEMS rows + the reference-design zip
        self.assertEqual(dict((n, g) for n, _, g in items)["datasheet"], True)
        self.assertTrue(all(not g for n, _, g in items if n != "datasheet"))
        self.assertEqual(items[-1][0], "devkit-carrier-reference-design")
        self.assertTrue(items[-1][1].endswith(".zip"))

    def test_classify_url(self):
        # HTML on a direct document is the stale-URL signature; the gated
        # item's login page is expected
        self.assertEqual(check.classify_url("d", 200, "application/pdf", False)[0], "PASS")
        self.assertEqual(check.classify_url("d", 200, "text/html", True),
                         ("PASS", "login page (expected)"))
        self.assertEqual(check.classify_url("d", 200, "text/html", False)[0], "FAIL")
        self.assertEqual(check.classify_url("d", 503, "application/pdf", False)[0], "FAIL")
        self.assertEqual(check.classify_url("d", 200, "application/pdf", True)[0], "FAIL")

    def test_middot_row_with_one_doc_absent_never_validates_against_the_other(self):
        # regression (found in review): the pairing used to degrade to a
        # cross-product, letting the surviving doc validate the missing
        # doc's section by accident
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp)
            for f in SYNTH.iterdir():
                if f.name.startswith("demo-doc"):   # demo-doc only, no demo2
                    (corpus / f.name).write_text(f.read_text(encoding="utf-8"),
                                                 encoding="utf-8")
            code, out = run_lint(SYNTH_INDEX, corpus, self.pdf)
            self.assertIn("SKIP  Two docs paired: 2.1 — paired document not fetched", out)
            self.assertNotIn("demo-doc 2.1", out.replace("1.2", ""), out)
            self.assertIn("demo-doc 1.2", out)   # the surviving half still checks


class CliSmoke(unittest.TestCase):
    @unittest.skipIf(sys.platform == "win32",
                     "Windows python resolves 'bash' to WSL's, which cannot "
                     "see C:/ paths; CI (linux) runs check.sh end-to-end")
    def test_check_sh_wrapper(self):
        done = subprocess.run(
            ["bash", (HERE / "check.sh").as_posix(), "--offline"],
            capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("summary:", done.stdout)


@real_corpus
class RealCorpusTier(unittest.TestCase):
    """The actual INDEX.md against the fetched corpus, plus the two
    deliberately-broken variants from the acceptance criteria."""

    def test_clean_index_lints_with_zero_failures(self):
        code, out = lint_real()
        self.assertEqual(code, 0, out)
        self.assertIn("routing table (27 data rows)", out)
        self.assertIn("1.2p pinned, 1.2 in TRM title page", out)
        self.assertIn("0 failed", out)

    def test_broken_section_reference_fails(self):
        idx = (HERE / "INDEX.md").read_text(encoding="utf-8")
        code, out = lint_real(idx.replace("| `devkit-carrier-spec.md` | §3.4, Table 3-4 |",
                                          "| `devkit-carrier-spec.md` | §3.4, Table 9-9 |"))
        self.assertEqual(code, 1)
        self.assertIn("FAIL  Button header", out)
        self.assertIn("9-9", out)

    def test_memorized_page_shifted_by_one_fails(self):
        idx = (HERE / "INDEX.md").read_text(encoding="utf-8")
        self.assertIn("§3.4 Table 3-4 (p. 28):", idx)
        code, out = lint_real(idx.replace("§3.4 Table 3-4 (p. 28):",
                                          "§3.4 Table 3-4 (p. 29):"))
        self.assertEqual(code, 1)
        self.assertIn("FAIL  memorized #2", out)
        self.assertIn("p. 29", out)

    def test_false_positive_traps_never_fail(self):
        code, out = lint_real()
        self.assertEqual(code, 0, out)
        fails = [l for l in out.splitlines() if l.startswith("FAIL")]
        self.assertEqual(fails, [])
        for trap in ("HD Video → Encode", "`datasheet.md`", "`../inventory.md`"):
            self.assertNotIn(trap, "".join(fails))


if __name__ == "__main__":
    unittest.main(verbosity=2)
