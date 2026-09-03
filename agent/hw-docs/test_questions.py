"""Tests for eval/questions.yaml — issue #24 acceptance criteria.

Two tiers, like test_grade.py / test_check.py:

  - structure tier: the golden set's own invariants — ~25 items, ~40%
    answerable:false, >=5 prior-divergent items with a note on which way
    the wrong prior pulls, >=6 routing categories including the four added
    late in #20 (fan drive type, serial-bus electrical, boot straps,
    DP/HDMI), and a well-formed schema. Runs anywhere (needs PyYAML).
    - real-corpus tier: every answerable item's expected_citation is
    assembled into a gradeable answer and must pass grade.py against the
    fetched corpus — a question whose own ground truth does not grade
    clean is a broken question. Citations into documents that are not
    fetched count as skips (CI fetches --core; the datasheet is
    login-gated) — a skip is never a pass, and no citation may ever FAIL.

Run: python agent/hw-docs/test_questions.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grade  # noqa: E402  (citation machinery shared with the runner)

import yaml  # noqa: E402  (pip install pyyaml — in the CI pip line)

REAL = HERE / "md"
QUESTIONS = HERE / "eval" / "questions.yaml"

# The four routing categories added late in #20 — the golden set must
# exercise them or the coverage claim of the knowledge layer is hollow.
REQUIRED_LATE_CATEGORIES = {"fan-drive-type", "serial-bus-electrical",
                            "boot-straps", "dp-hdmi"}
MIN_ROUTING_CATEGORIES = 6
MIN_ITEMS = 24          # "~25 items" with room to breathe
MIN_DIVERGENT = 5       # "at least 5 documented prior-doc divergent"
# "~40% unanswerable" — the band the runner's unanswerable-rate metric
# will be judged against, so the set must stay inside it
UNANSWERABLE_BAND = (0.35, 0.45)


def load_questions() -> list[dict]:
    with open(QUESTIONS, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, list) and data, "questions.yaml must be a non-empty list"
    return data


def citation_answer(item: dict) -> str:
    """The item's expected_citations as one gradeable answer: one citation
    + its load-bearing quote per paragraph, so quote pairing cannot
    mis-fire no matter how many citations the item carries."""
    paras = [f'{c["doc"]}.md {c["section"]} (p. {c["page"]}) — "{c["quote"]}"'
             for c in item["expected_citation"]]
    return "\n\n".join(paras)


def doc_fetched(stem: str) -> bool:
    return (REAL / f"{stem}.md").is_file()


class StructureTier(unittest.TestCase):
    """The golden set's invariants — the acceptance criteria of #24."""

    @classmethod
    def setUpClass(cls):
        cls.items = load_questions()

    def test_size_and_unanswerable_share(self):
        self.assertGreaterEqual(len(self.items), MIN_ITEMS)
        unanswerable = [i for i in self.items if not i["answerable"]]
        share = len(unanswerable) / len(self.items)
        lo, hi = UNANSWERABLE_BAND
        self.assertTrue(lo <= share <= hi,
                        f"{len(unanswerable)}/{len(self.items)} = {share:.0%} "
                        f"unanswerable, must stay in {lo:.0%}-{hi:.0%}")

    def test_ids_unique_and_fields_present(self):
        ids = [i["id"] for i in self.items]
        self.assertEqual(len(ids), len(set(ids)), "duplicate ids")
        for i in self.items:
            for field in ("id", "question", "category", "answerable",
                          "expected_answer"):
                self.assertIn(field, i, f"{i.get('id')}: missing {field}")
                self.assertTrue(str(i[field]).strip(), f"{i['id']}: empty {field}")
            self.assertIsInstance(i["answerable"], bool)

    def test_answerable_items_carry_wellformed_citations(self):
        for i in self.items:
            if not i["answerable"]:
                continue
            self.assertIsInstance(i["expected_citation"], list)
            self.assertGreaterEqual(len(i["expected_citation"]), 1, i["id"])
            self.assertNotIn("expected_redirect", i,
                             f"{i['id']}: answerable, so no redirect")
            for c in i["expected_citation"]:
                self.assertIn(c["doc"], grade.CANONICAL_STEMS,
                              f"{i['id']}: {c['doc']} is not a corpus stem")
                self.assertTrue(grade.sections_in(c["section"]),
                                f"{i['id']}: {c['section']!r} has no §/Ch./Table token")
                self.assertIsInstance(c["page"], int)
                self.assertGreater(c["page"], 0)
                # QUOTE_RE is `[^"]{2,400}` — outside that the grader drops
                # the quote and the citation becomes NO_QUOTE
                self.assertTrue(2 <= len(c["quote"]) <= 400,
                                f"{i['id']}: quote length {len(c['quote'])}")

    def test_unanswerable_items_carry_a_redirect_and_no_citation(self):
        for i in self.items:
            if i["answerable"]:
                continue
            self.assertTrue(i.get("expected_redirect", "").strip(),
                            f"{i['id']}: unanswerable without a redirect")
            self.assertNotIn("expected_citation", i,
                             f"{i['id']}: unanswerable, so no citation")

    def test_at_least_five_prior_divergent_items(self):
        divergent = [i for i in self.items if i.get("divergence", "").strip()]
        self.assertGreaterEqual(len(divergent), MIN_DIVERGENT)
        for i in divergent:
            self.assertGreater(len(i["divergence"].split()), 5,
                               f"{i['id']}: divergence note too thin to act on")

    def test_category_coverage_spans_the_late_routing_rows(self):
        answerable_cats = {i["category"] for i in self.items if i["answerable"]}
        missing = REQUIRED_LATE_CATEGORIES - answerable_cats
        self.assertEqual(missing, set(),
                         f"routing categories added in #20 not covered: {missing}")
        self.assertGreaterEqual(len(answerable_cats), MIN_ROUTING_CATEGORIES)


real_corpus = unittest.skipUnless(any(REAL.glob("*.md")),
                                  "corpus not fetched (run agent/hw-docs/fetch.sh)")


@real_corpus
class RealCorpusTier(unittest.TestCase):
    """Ground truth must grade clean — the anchor of the whole epic."""

    @classmethod
    def setUpClass(cls):
        cls.items = [i for i in load_questions() if i["answerable"]]

    def test_every_answerable_citation_grades_ok(self):
        graded = skipped = 0
        failures = []
        for item in self.items:
            results = grade.grade(grade.parse_answer(citation_answer(item)), REAL)
            self.assertEqual(len(results), len(item["expected_citation"]),
                             f"{item['id']}: a citation did not parse into a "
                             "gradeable claim — check doc/section/page syntax")
            for r in results:
                if r.verdict in ("DOC_MISSING", "NO_ANCHORS"):
                    skipped += 1  # corpus gap, not a broken question (CI: --core)
                else:
                    graded += 1
                    if r.verdict != "OK":
                        failures.append(
                            f"{item['id']}: {r.citation.doc_token} "
                            f"{r.citation.secs[0][0]} (p. {r.citation.page}) "
                            f"— {r.verdict}: {'; '.join(r.notes)}")
        self.assertGreater(graded, 0, "every citation skipped — corpus empty?")
        self.assertEqual(failures, [], "\n".join(failures))

    @unittest.skipUnless(doc_fetched("devkit-carrier-spec"),
                         "devkit-carrier-spec not fetched")
    def test_page_shifted_by_one_is_caught(self):
        # the historical bug class from #20, on this set's own ground
        # truth: button-header-pin12's p. 28 moved to p. 29 must fail —
        # §3.4 spans pages 28-28
        item = next((i for i in self.items if i["id"] == "button-header-pin12"), None)
        self.assertIsNotNone(item, "button-header-pin12 renamed or removed — "
                                   "update this test with another §x.y (p. N) "
                                   "citation whose section spans one page")
        c = dict(item["expected_citation"][0])
        c["page"] += 1
        text = citation_answer({"expected_citation": [c]})
        r = grade.grade(grade.parse_answer(text), REAL)[0]
        self.assertNotEqual(r.verdict, "OK", r.notes)
        self.assertEqual(r.verdict, "PAGE_OUTSIDE_SECTION", r.notes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
