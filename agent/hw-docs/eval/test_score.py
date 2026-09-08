"""Synthetic tests for eval/score.py — issues #25/#26 (runner + baseline).

Everything here runs anywhere: transcripts are hand-built NDJSON in the
claude -p --output-format stream-json shape, questions are a two-item
synthetic set, and citation grading runs against an empty directory
(exit 2 = unverifiable — the deterministic path with no corpus). The
judge tier injects fake judge stdout; no claude, no network.
"""
import json
import tempfile
import unittest
from pathlib import Path

import sys
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # grade.py lives flat in hw-docs/

import score  # noqa: E402


def nd(*events) -> str:
    return "\n".join(json.dumps(e) for e in events) + "\n"


def init_event(**over):
    e = {"type": "system", "subtype": "init", "model": "test-model",
         "session_id": "s-1", "cwd": "C:\\x"}
    e.update(over)
    return e


def assistant(*blocks):
    return {"type": "assistant",
            "message": {"role": "assistant", "content": list(blocks)}}


def tool_use(name, **inp):
    return {"type": "tool_use", "name": name, "input": inp}


def text_block(t):
    return {"type": "text", "text": t}


def result_event(text="final answer", **over):
    e = {"type": "result", "subtype": "success", "is_error": False,
         "result": text, "num_turns": 3, "total_cost_usd": 0.01,
         "duration_ms": 1234}
    e.update(over)
    return e


class ParseTranscriptTest(unittest.TestCase):
    def test_extracts_metadata_tools_and_answer(self):
        t = nd(
            init_event(),
            assistant(tool_use("Read", file_path="agent/hw-docs/INDEX.md"),
                      text_block("thinking")),
            assistant(tool_use("Bash", command="python hw-docs/search.py q")),
            assistant(text_block("The answer is no. datasheet.md §2.10.5 (p. 31)")),
            result_event(),
        )
        p = score.parse_transcript(t)
        self.assertEqual(p["model"], "test-model")
        self.assertEqual(p["final_answer"], "final answer")
        self.assertEqual(p["num_turns"], 3)
        self.assertEqual(p["cost_usd"], 0.01)
        self.assertEqual([u["tool"] for u in p["tool_uses"]],
                         ["Read", "Bash"])
        self.assertTrue(p["ok"])

    def test_auth_failure_is_not_ok(self):
        t = nd(init_event(),
               result_event(text="Not logged in · Please run /login",
                            is_error=True))
        p = score.parse_transcript(t)
        self.assertFalse(p["ok"])
        self.assertIn("Not logged in", p["error"])

    def test_malformed_lines_skipped_missing_result_means_no_answer(self):
        t = "not json\n" + nd(init_event(), assistant(text_block("hi")))
        p = score.parse_transcript(t)
        self.assertIsNone(p["final_answer"])
        self.assertFalse(p["ok"])

    def test_windows_backslash_paths(self):
        t = nd(assistant(tool_use("Read",
                                  file_path="C:\\repo\\agent\\hw-docs\\INDEX.md")),
               result_event())
        self.assertTrue(score.routing_signals(
            score.parse_transcript(t)["tool_uses"])["routed"])


class RoutingSignalsTest(unittest.TestCase):
    def signals(self, *tools):
        return score.routing_signals(
            [{"tool": n, "input": i} for n, i in tools])

    def test_routed_via_read(self):
        self.assertTrue(self.signals(
            ("Read", {"file_path": "hw-docs/INDEX.md"}))["routed"])

    def test_routed_via_grep_path(self):
        self.assertTrue(self.signals(
            ("Grep", {"pattern": "shutdown", "path": "agent/hw-docs/INDEX.md"}))["routed"])

    def test_routed_via_bash_cat(self):
        self.assertTrue(self.signals(
            ("Bash", {"command": "cat agent/hw-docs/INDEX.md | head"}))["routed"])

    def test_corpus_read_alone_is_not_routed(self):
        s = self.signals(("Read", {"file_path": "agent/hw-docs/md/datasheet.md"}))
        self.assertFalse(s["routed"])
        self.assertFalse(s["searched"])

    def test_searched_via_bash_or_powershell(self):
        for tool in ("Bash", "PowerShell"):
            self.assertTrue(self.signals(
                (tool, {"command": "python agent/hw-docs/search.py fan"}))["searched"])

    def test_answer_key_read_is_contamination(self):
        s = self.signals(("Read", {"file_path": "agent/hw-docs/eval/questions.yaml"}))
        self.assertTrue(s["contaminated"])


class CitationSignalsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_cited(self):
        a = 'No. datasheet.md §2.10.5 (p. 31) — "One independent CAN port"'
        s = score.citation_signals(a, self.tmp)
        self.assertTrue(s["cited"])
        self.assertEqual(s["grade_exit"], 2)  # empty corpus dir → unverifiable

    def test_plain_prose_not_cited(self):
        s = score.citation_signals("Yes, probably H.265.", self.tmp)
        self.assertFalse(s["cited"])
        self.assertIsNone(s["grade_exit"])


QUESTIONS = [
    {"id": "q1", "question": "how many CAN?", "category": "module-interfaces",
     "answerable": True, "expected_answer": "exactly one CAN",
     "divergence": "prior says several"},
    {"id": "q2", "question": "what IP?", "category": "machine-facts",
     "answerable": False, "expected_answer": "not sourceable",
     "expected_redirect": "agent/inventory.md"},
    {"id": "q3", "question": "shutdown temp?", "category": "thermal-limits",
     "answerable": True, "expected_answer": "105 C"},
]


class GroupingTest(unittest.TestCase):
    def test_groups(self):
        self.assertEqual(score.item_grouping(QUESTIONS[0]), "divergent")
        self.assertEqual(score.item_grouping(QUESTIONS[1]), "unanswerable")
        self.assertEqual(score.item_grouping(QUESTIONS[2]), "already-knows")


class JudgeTest(unittest.TestCase):
    def test_answerable_prompt_carries_ground_truth_and_rubric(self):
        p = score.judge_prompt(QUESTIONS[0], "candidate answer text")
        self.assertIn("exactly one CAN", p)
        self.assertIn("candidate answer text", p)
        self.assertIn("CORRECT", p)
        self.assertNotIn("ABSTAINED", p)

    def test_unanswerable_prompt_uses_abstain_rubric(self):
        p = score.judge_prompt(QUESTIONS[1], "not sourceable, see inventory")
        self.assertIn("ABSTAINED", p)
        self.assertIn("agent/inventory.md", p)

    def test_parse_judge_bare_and_fenced(self):
        bare = score.parse_judge('{"verdict": "CORRECT", "forum_only": false, "note": "ok"}')
        self.assertEqual(bare["verdict"], "CORRECT")
        fenced = score.parse_judge('```json\n{"verdict": "GUESSED", "forum_only": true, "note": "x"}\n```')
        self.assertEqual(fenced["verdict"], "GUESSED")
        self.assertTrue(fenced["forum_only"])

    def test_parse_judge_garbage(self):
        self.assertEqual(score.parse_judge("no json here")["verdict"],
                         "PARSE_ERROR")


def row(qid, **over):
    r = {"id": qid, "routed": False, "searched": False, "contaminated": False,
         "cited": False, "grade_exit": None, "verdict": None,
         "forum_only": False, "ok": True, "cost_usd": 0.0, "num_turns": 1}
    q = next(q for q in QUESTIONS if q["id"] == qid)
    r.update({"category": q["category"], "answerable": q["answerable"],
              "group": score.item_grouping(q)})
    r.update(over)
    return r


class ScoreboardTest(unittest.TestCase):
    def test_rates_by_slice(self):
        rows = [
            row("q1", routed=True, cited=True, grade_exit=0, verdict="CORRECT"),
            row("q3", routed=False, cited=False, verdict="WRONG"),
            row("q2", verdict="ABSTAINED"),
        ]
        sb = score.scoreboard(rows)
        self.assertEqual(sb["all"]["n"], 3)
        self.assertAlmostEqual(sb["all"]["routed"], 1 / 3)
        self.assertEqual(sb["module-interfaces"]["correct"], 1.0)
        self.assertEqual(sb["thermal-limits"]["correct"], 0.0)
        self.assertEqual(sb["unanswerable"]["abstained"], 1.0)
        self.assertEqual(sb["divergent"]["valid"], 1.0)

    def test_errors_and_unscored_counted(self):
        rows = [row("q1", ok=False), row("q3", verdict=None)]
        sb = score.scoreboard(rows)
        self.assertEqual(sb["all"]["errors"], 1)
        self.assertEqual(sb["all"]["unscored"], 1)


class DeltaTest(unittest.TestCase):
    def test_delta_renders_markdown_with_both_signs(self):
        with_rows = [row("q1", routed=True, verdict="CORRECT"),
                     row("q3", routed=True, verdict="CORRECT"),
                     row("q2", verdict="ABSTAINED")]
        without_rows = [row("q1", routed=False, verdict="WRONG"),
                        row("q3", routed=False, verdict="CORRECT"),
                        row("q2", verdict="GUESSED")]
        md = score.render_delta(with_rows, without_rows, QUESTIONS)
        self.assertIn("| metric | with | without | delta |", md)
        self.assertIn("module-interfaces", md)
        self.assertIn("+100.0", md)   # correct 0→1
        self.assertIn("±0.0", md)     # already-knows correct unchanged


class FilterTranscriptTest(unittest.TestCase):
    def test_tool_result_bodies_truncated_structure_kept(self):
        big = "x" * 5000
        t = nd(
            init_event(),
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "content": big}]}},
            result_event(),
        )
        out = score.filter_transcript(t, keep=500)
        self.assertLess(len(out), len(t))
        kept = [json.loads(l) for l in out.splitlines()]
        self.assertEqual(kept[1]["type"], "user")
        self.assertLessEqual(
            len(kept[1]["message"]["content"][0]["content"]), 600)
        self.assertTrue(kept[1]["message"]["content"][0]["content"].endswith("…"))


if __name__ == "__main__":
    unittest.main()
