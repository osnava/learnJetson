"""Tests for the structural citation grader (grade.py, issue #29).

Two tiers:
  - synthetic (fixtures/demo-doc.json, committed): a hand-written docling
    JSON exercising every structural rule — runs anywhere, no docling
  - real (md/*.json, gitignored): the fetched docling corpus; skips cleanly
    when unfetched — a skip is never a pass

Run: python agent/hw-docs/test_grade.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grade  # noqa: E402

FIXTURES = HERE / "fixtures"
SYNTH = FIXTURES                      # demo-doc.json + orin-trm.md live here
REAL = HERE / "md"


def needs(*stems: str):
    """Skip unless these docs' JSON sources of truth are fetched — CI runs
    --core, where the login-gated datasheet and the --full giants are
    deliberately absent."""
    return unittest.skipUnless(
        all((REAL / f"{s}.json").is_file() for s in stems),
        "corpus JSON not fetched: " + ", ".join(
            s for s in stems if not (REAL / f"{s}.json").is_file()))


def run_cli(answer: str | None, corpus: Path, *extra: str,
            stdin: str = "") -> subprocess.CompletedProcess:
    """Run grade.py as a subprocess; answer=None means stdin."""
    cmd = [sys.executable, str(HERE / "grade.py")]
    if answer is not None:
        cmd.append(answer)
    cmd += ["--corpus", str(corpus), *extra]
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", input=stdin if answer is None else None)


def one(text: str, corpus: Path = SYNTH) -> grade.Result:
    """Grade a one-citation answer against the given corpus."""
    results = grade.grade(grade.parse_answer(text), corpus)
    assert len(results) == 1, f"expected 1 citation, got {len(results)}: {results}"
    return results[0]


class ParseTest(unittest.TestCase):
    """The human-facing surface `doc §section (p. N) + quote` — unchanged
    from the v1 grader (issue #29 keeps the surface, rebuilds the checks)."""

    def test_doc_section_table_page_and_quote_are_extracted(self):
        cits = grade.parse_answer(
            'Per `devkit-carrier-spec.md` §3.4 Table 3-4, p. 28: pin 3 is '
            '"UART2_RXD (DEBUG)", Input, 3.3V.')
        self.assertEqual(len(cits), 1)
        c = cits[0]
        self.assertEqual(c.stem, "devkit-carrier-spec")
        self.assertEqual(c.page, 28)
        self.assertEqual([tok for tok, _ in c.secs], ["3.4", "3-4"])
        self.assertEqual([kind for _, kind in c.secs], ["sec", "table"])
        self.assertEqual(c.quote, "UART2_RXD (DEBUG)")

    def test_paren_and_bare_page_forms(self):
        self.assertEqual(grade.parse_answer("doc §1.1 (p. 2) x")[0].page, 2)
        self.assertEqual(grade.parse_answer("doc §1.1 p. 2 x")[0].page, 2)

    def test_second_citation_inherits_the_doc(self):
        cits = grade.parse_answer(
            "demo-doc.md §1.1 (p. 1) holds; §1.2 (p. 2) too, quoted \"WIDGET_RDY\".")
        self.assertEqual([c.stem for c in cits], ["demo-doc", "demo-doc"])

    def test_quote_inside_the_citation_is_not_the_load_bearing_quote(self):
        cits = grade.parse_answer(
            'No encoder — `datasheet.md` Ch. 1 Overview, "HD Video → Encode" '
            '(p. 7), states: "1080p30 Supported via CPU Cores with Software."')
        self.assertEqual(len(cits), 1)
        self.assertEqual(cits[0].quote, "1080p30 Supported via CPU Cores with Software.")

    def test_section_name_after_the_number_is_not_a_second_token(self):
        # search.py prints §3.4 Button Header — the words after the number
        # belong to the numbered token, not a name token of their own
        cits = grade.parse_answer("doc.md §3.4 Button Header (p. 28) — \"x y\".")
        self.assertEqual([t for t, _ in cits[0].secs], ["3.4"])

    def test_named_section_token_is_extracted(self):
        # search.py also emits §<heading name> for unnumbered headings
        cits = grade.parse_answer("datasheet.md §Encode (p. 7) — \"1080p30 x\".")
        self.assertEqual([(t, k) for t, k in cits[0].secs], [("Encode", "name")])


class SyntheticCorpusTest(unittest.TestCase):
    """Structural acceptance criteria against fixtures/demo-doc.json —
    the docling JSON shapes every rule is written against."""

    def test_ok_citation(self):
        r = one('demo-doc.md §1.1 (p. 1) — "The widget supply rail is 3.3 V nominal."')
        self.assertEqual(r.verdict, "OK")
        self.assertIn("spans page 1", r.notes[0])

    def test_page_shifted_by_one_is_caught(self):
        # the p. 29 vs p. 28 bug from #20: same section, same quote, page off
        # by one. §1.1's body items all carry prov page 1 — 2 is not in the
        # span by construction, no content threshold needed.
        r = one('demo-doc.md §1.1 (p. 2) — "The widget supply rail is 3.3 V nominal."')
        self.assertEqual(r.verdict, "PAGE_OUTSIDE_SECTION")
        self.assertIn("quote does not occur", " ".join(r.notes))

    def test_right_quote_wrong_page_only(self):
        # section spans the page, but the quote lives on a different one
        r = one('demo-doc.md Ch. 1 (p. 1) — "Absolute maximum is 125 degrees C."')
        self.assertEqual(r.verdict, "QUOTE_NOT_AT_PAGE")

    def test_chapter_spans_its_short_heading_pages(self):
        # p.3 carries a heading + one line — body content, so part of the
        # chapter; only furniture (running header/footer) is span-excluded
        r = one('demo-doc.md Ch. 1 (p. 3) — "Absolute maximum is 125 degrees C."')
        self.assertEqual(r.verdict, "OK")

    def test_unnumbered_heading_does_not_close_numbered_section(self):
        # 'Notes:' sits between §1.2's table and a trailing text item — an
        # unnumbered heading must never end a numbered section's span
        r = one('demo-doc.md §1.2 (p. 2) — "Bits are read-only after self-test."')
        self.assertEqual(r.verdict, "OK")

    def test_interior_furniture_only_page_is_not_in_the_span(self):
        # p.4 sits inside §1.3's heading-to-close extent but carries only
        # running header/footer (content_layer furniture) — a page set, not
        # a min/max range: §1.3 spans {3, 5}
        r = one('demo-doc.md §1.3 (p. 4) — "Absolute maximum is 125 degrees C."')
        self.assertEqual(r.verdict, "PAGE_OUTSIDE_SECTION")
        self.assertIn("pages 3, 5", " ".join(r.notes))

    def test_section_not_found(self):
        r = one('demo-doc.md §9.9 (p. 1) — "The widget supply rail is 3.3 V nominal."')
        self.assertEqual(r.verdict, "SECTION_NOT_FOUND")

    def test_page_outside_range_span(self):
        # the range §1.1–1.3 runs to the end of §1.3's section — which,
        # per the unnumbered-heading rule, extends through the name
        # headings up to Chapter 2 (pages {1,2,3,5,6}); p. 7 is outside it
        r = one('demo-doc.md §1.1–1.3 (p. 7) — "Absolute maximum is 125 degrees C."')
        self.assertEqual(r.verdict, "PAGE_OUTSIDE_SECTION")

    def test_no_quote(self):
        r = one("demo-doc.md §1.1 (p. 1) says the rail is 3.3 V nominal.")
        self.assertEqual(r.verdict, "NO_QUOTE")

    def test_quote_from_table_cell_wrapped_mid_cell(self):
        # the cell text wraps mid-cell ('...error text\ncontinues on this
        # wrapped cell') — quoted with a plain space
        r = one('demo-doc.md §1.2 Table 1-1 (p. 2) — "WIDGET_ERR: Multi word '
                'error text continues on this wrapped cell"')
        self.assertEqual(r.verdict, "OK")

    def test_dehyphenated_quote_resolves(self):
        # the cell reads 'Auto-Power- On' (hyphen + space, the PDF's line
        # wrap) — quoted as 'Auto-Power-On'. The hyphen stays literal on
        # both sides; only the wrapped space differs.
        r = one('demo-doc.md §1.2 Table 1-1 (p. 2) — "Self-test runs when '
                'Auto-Power-On is disabled."')
        self.assertEqual(r.verdict, "OK")

    def test_quote_spelled_with_md_pipes_matches(self):
        # a quote copied from the md rendering spells cell boundaries as
        # pipes ('|1|WIDGET_ERR|...|') — the same row, same cells as the
        # space-joined substrate, so it must match
        r = one('demo-doc.md §1.2 Table 1-1 (p. 2) — "1|WIDGET_ERR|WIDGET_ERR: '
                'Multi word error text continues on this wrapped cell"')
        self.assertEqual(r.verdict, "OK")

    def test_named_section_resolves_against_heading_text(self):
        # unnumbered headings are citable as §name — search.py emits them
        r = one('demo-doc.md §Encode (p. 6) — "1080p30 Supported via CPU '
                'Cores with Software"')
        self.assertEqual(r.verdict, "OK")

    def test_named_section_matches_numbered_heading_by_suffix(self):
        # §Widget Overview -> the '1.1 Widget Overview' heading object
        r = one('demo-doc.md §Widget Overview (p. 1) — "The widget supply '
                'rail is 3.3 V nominal."')
        self.assertEqual(r.verdict, "OK")

    def test_unknown_named_section_fails(self):
        r = one('demo-doc.md §Nonexistent Mode (p. 6) — "Rails sequenced down."')
        self.assertEqual(r.verdict, "SECTION_NOT_FOUND")

    def test_bare_and_paren_citations_in_one_paragraph(self):
        # both page forms side by side must yield two graded citations —
        # a paragraph-level either/or silently dropped the bare one once
        results = grade.grade(grade.parse_answer(
            '"The widget supply rail is 3.3 V nominal" holds (demo-doc.md '
            '§1.1 p. 1), and §1.2 (p. 2) says "WIDGET_RDY".'), SYNTH)
        self.assertEqual([r.verdict for r in results], ["OK", "OK"], results)

    def test_unknown_document_fails_like_an_invention(self):
        # citing INDEX.md (or any non-corpus .md) as a source is a
        # wrong-document citation — it can never be fetched, unlike a
        # canonical stem whose doc is simply not on disk yet
        r = one('INDEX.md §1.1 (p. 1) — "whatever"')
        self.assertEqual(r.verdict, "DOC_UNKNOWN")

    def test_doc_missing_and_exit_code_distinct_from_hallucination(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "corpus"  # present but empty: fetched-less
            empty.mkdir()
            r = one('ghost-doc.md §1.1 (p. 1) — "anything at all, invented."', empty)
            self.assertEqual(r.verdict, "DOC_MISSING")
            answer = Path(tmp) / "answer.md"  # kept out of the corpus dir
            answer.write_text('ghost-doc.md §1.1 (p. 1) — "anything"', encoding="utf-8")
            done = run_cli(str(answer), empty)
            self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
            self.assertIn("DOC_MISSING", done.stdout)

    def test_known_doc_not_fetched_is_also_doc_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = one('orin-trm.md §1.1 (p. 3) — "anything"', Path(tmp))
            self.assertEqual(r.verdict, "DOC_MISSING")
            self.assertIn("fetch.sh", " ".join(r.notes))

    def test_fetched_doc_without_json_is_no_provenance_not_a_pass(self):
        # the TRM converts md-only (a full JSON would be ~970 MB): the doc
        # IS fetched, but structural grading needs the JSON — a soft
        # verdict, never a pass, and never the agent's fault
        r = one('orin-trm.md §1.1 (p. 3) — "anything at all, invented."', SYNTH)
        self.assertEqual(r.verdict, "NO_PROVENANCE")
        self.assertIn("JSON", " ".join(r.notes))
        self.assertEqual(grade.exit_code([r]), 2)


class CliTest(unittest.TestCase):
    def test_clean_answer_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.md"
            a.write_text('demo-doc.md §1.1 (p. 1) — "The widget supply rail is 3.3 V nominal."',
                         encoding="utf-8")
            done = run_cli(str(a), SYNTH)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertIn("OK", done.stdout)

    def test_bad_citation_exits_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.md"
            a.write_text('demo-doc.md §1.1 (p. 2) — "The widget supply rail is 3.3 V nominal."',
                         encoding="utf-8")
            done = run_cli(str(a), SYNTH)
            self.assertEqual(done.returncode, 1)
            self.assertIn("PAGE_OUTSIDE_SECTION", done.stdout)

    def test_no_provenance_exits_two(self):
        done = run_cli(None, SYNTH, stdin='orin-trm.md §1.1 (p. 3) — "anything"')
        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
        self.assertIn("NO_PROVENANCE", done.stdout)

    def test_stdin(self):
        done = run_cli(None, SYNTH,
                       stdin='demo-doc.md §1.1 (p. 1) — "The widget supply rail is 3.3 V nominal."')
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_json_output(self):
        done = run_cli(None, SYNTH, "--json",
                       stdin='demo-doc.md §1.1 (p. 2) — "The widget supply rail is 3.3 V nominal."')
        payload = json.loads(done.stdout)
        self.assertEqual(payload["exit"], 1)
        self.assertEqual(payload["citations"][0]["verdict"], "PAGE_OUTSIDE_SECTION")

    def test_answer_without_citations_warns_but_passes(self):
        done = run_cli(None, SYNTH)
        self.assertEqual(done.returncode, 0)
        self.assertIn("no citations", done.stderr)


real_corpus = unittest.skipUnless(
    any(REAL.glob("*.json")),
    "docling corpus (JSON) not fetched — run agent/hw-docs/fetch.sh")


@real_corpus
class RealCorpusTest(unittest.TestCase):
    """The fetched docling corpus: the #29 known-good probes, the UART-session
    answer (issue #20), and INDEX's memorized answers."""

    @needs("devkit-carrier-spec")
    def test_button_table_probe(self):
        # the issue's probe: button table = devkit-carrier-spec §3.4 (p. 28)
        r = one('devkit-carrier-spec.md §3.4 (p. 28) — "SYS_RESET*"', REAL)
        self.assertEqual(r.verdict, "OK", r.notes)

    @needs("devkit-carrier-spec")
    def test_p29_vs_p28_trap_is_caught(self):
        # the historical bug class: same section, same quote, page + 1 —
        # §3.4's heading object sits on p.28 and its last body item ends
        # before §3.5 starts on p.29, so 29 is outside by construction
        r = one('devkit-carrier-spec.md §3.4 (p. 29) — "SYS_RESET*"', REAL)
        self.assertEqual(r.verdict, "PAGE_OUTSIDE_SECTION")

    @needs("devkit-carrier-spec")
    def test_dehyphenated_cell_quote_on_p28(self):
        # 'Auto-Power- On' in the PDF's wrapped cell, quoted as one word
        r = one('devkit-carrier-spec.md §3.4 (p. 28) — "Connect pins 11 and 12 '
                'to initiate power-on if Auto-Power-On disabled"', REAL)
        self.assertEqual(r.verdict, "OK", r.notes)

    @needs("datasheet")
    def test_encoder_probe(self):
        # the issue's probe: encoder line = datasheet §Encode (p. 7), an
        # unnumbered heading object followed by its one-line text item
        r = one('datasheet.md §Encode (p. 7) — "1080p30 Supported via CPU '
                'Cores with Software"', REAL)
        self.assertEqual(r.verdict, "OK", r.notes)

    @needs("devkit-carrier-spec")
    def test_uart_session_answer_grades_clean(self):
        results = grade.grade(grade.parse_answer(
            (HERE / "fixtures" / "uart-session-answer.md").read_text(
                encoding="utf-8")), REAL)
        # two citations in the fixture: bare "…Table 3-4, p. 28:" and the
        # parenthesized "§3.4 (p. 28)" — both must grade, both must pass
        self.assertEqual([r.verdict for r in results], ["OK", "OK"], results)

    @needs("devkit-carrier-spec")
    def test_uart_session_answer_page_shifted_by_one_is_caught(self):
        # the historical bug, mechanically: the same answer citing p. 29
        done = run_cli(str(HERE / "fixtures" /
                           "uart-session-answer.corrupt-page.md"), REAL)
        self.assertEqual(done.returncode, 1)
        self.assertIn("PAGE_OUTSIDE_SECTION", done.stdout)
        self.assertNotIn("OK —", done.stdout)

    def test_corrupt_fixture_differs_only_in_the_page(self):
        # the twins must never drift: the corrupt one is exactly the clean
        # answer with p. 28 shifted to p. 29, or the test above proves
        # nothing about page-shift detection
        clean = (HERE / "fixtures" / "uart-session-answer.md").read_text(
            encoding="utf-8")
        corrupt = (HERE / "fixtures" /
                   "uart-session-answer.corrupt-page.md").read_text(encoding="utf-8")
        self.assertEqual(clean.replace("p. 28", "p. 29"), corrupt)

    @needs("datasheet")
    def test_index_memorized_datasheet_answer(self):
        r = one('No hardware video encoder on Orin Nano — `datasheet.md` '
                'Ch. 1 Overview, "HD Video → Encode" (p. 7), states it '
                'affirmatively: "1080p30 Supported via CPU Cores with Software."',
                REAL)
        self.assertEqual(r.verdict, "OK", r.notes)

    @needs("devkit-carrier-spec")
    def test_uart1_wrapped_identifier_resolves(self):
        # the ball-name cell is 'GP70_UART1_T XD_BOOT2_STR AP' in the JSON
        # (spaces where the PDF wrapped the token) — canonical() joins it;
        # quoted here as the joined form
        r = one('J12 pin 8 is UART1_TXD (devkit-carrier-spec.md §3.3 Table 3-3, '
                'p. 26) — ball name "GP70_UART1_TXD_BOOT2_STRAP", Output/Bidir.',
                REAL)
        self.assertEqual(r.verdict, "OK", r.notes)

    @needs("datasheet")
    def test_datasheet_decode_table_citation(self):
        # §2.9 + Table 2-5 (INDEX routing row): the table opens on p. 20 —
        # hand-written citations assuming p. 21 (the TOC's printed page + a
        # wrong offset guess) are exactly what this grader exists to catch
        r = one('Decode silicon (datasheet.md §2.9 Table 2-5, p. 20) covers '
                'H.264 "Baseline, Main, High" up to 4K30.', REAL)
        self.assertEqual(r.verdict, "OK", r.notes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
