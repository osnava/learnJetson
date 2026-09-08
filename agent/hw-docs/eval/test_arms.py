"""Synthetic tests for eval/arms.py — the #26 baseline-control arm builder.

The strip transforms are exercised against synthetic kit files that carry
the exact anchor fragments of the real ones (reading-order arrow chain,
the `## Hardware questions` section, tooling-table rows, the SETUP corpus
block and checklist row, the README index link). The invariant under
test: after the Without strip, the strings `hw-docs` / `INDEX.md` appear
nowhere in the kit docs a cold session reads, while INDEX.md itself stays
on disk (present but unreferenced — the issue #26 definition of the
Without arm).
"""
import tempfile
import unittest
from pathlib import Path

import sys
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import arms  # noqa: E402


AGENTS_MD = """# AGENTS.md — Operating this Jetson with an AI agent

Rulebook text.

**The kit, in reading order for a fresh session:** this rulebook →
[`inventory.md`](inventory.md) (machine facts, gitignored) →
[SETUP.md](SETUP.md) if you're bringing up a **new** Jetson with this kit →
[FIELD_NOTES.md](FIELD_NOTES.md) for the *why* behind these rules, with
sources → [`hw-docs/INDEX.md`](hw-docs/INDEX.md) the moment a question
touches hardware → the [runbooks](../docs/troubleshooting/README.md) when
something breaks.

## Hardware questions — answer from primary docs

Anything about pins, voltage domains, connectors: route it through
[`hw-docs/INDEX.md`](hw-docs/INDEX.md) and answer from the fetched
markdown in `hw-docs/md/`, citing `doc §section (p. N)`.

1. **Route** through INDEX.md; read/grep the routed section.
2. Routing misses? **Search**: `python hw-docs/search.py "question"`.

- Corpus not fetched yet? Run `hw-docs/fetch.sh`.

## Discovery — always in this order

1. `./jetson.sh find` — sweeps the LAN subnet.

## Tooling in this folder

| File | Purpose |
|---|---|
| `jetson.sh` | find / ssh / status |
| `hw-docs/INDEX.md` | routing table |
| `hw-docs/
fetch.sh` | materialize the corpus |
"""

README_MD = """# learnJetson

- [AI agent guide](agent/AGENTS.md) — [Setup guide (new Jetson)](agent/SETUP.md) · [Field notes & lessons](agent/FIELD_NOTES.md) · [Hardware docs index](agent/hw-docs/INDEX.md) (datasheet/TRM/carrier spec as greppable markdown, fetched on clone)
- [License](#license)
"""

SETUP_MD = """# Setup guide

Prose before.

```bash
uv tool install docling             # corpus converter (fetch.sh locates it)
agent/hw-docs/fetch.sh              # ~9 MB; --full adds TRM + schematics
```

`fetch.sh` resolves the docling-capable Python automatically. The routing
table from question to doc-section lives in
[`agent/hw-docs/INDEX.md`](hw-docs/INDEX.md); when routing misses,
`python agent/hw-docs/search.py "question"` searches the corpus
semantically with page provenance.

**Verify:**

| ☐ | YOLO engine inference verified in-container |
| ☐ | Hardware corpus fetched — `ls agent/hw-docs/md/` shows the converted docs; INDEX.md routes |
"""


class StripAgentsTest(unittest.TestCase):
    def test_without_removes_every_hw_docs_trace(self):
        out = arms.strip_agents_md(AGENTS_MD, without=True)
        self.assertNotIn("hw-docs", out)
        self.assertNotIn("INDEX.md", out)
        # reading order survives and still chains to the runbooks
        self.assertIn("FIELD_NOTES.md", out)
        self.assertIn("runbooks", out)
        # neighbouring sections survive
        self.assertIn("## Discovery", out)
        self.assertIn("jetson.sh find", out)

    def test_with_keeps_layer_drops_only_the_answer_key_row(self):
        # wrapped rows too: the `| `hw-docs/` opener sits alone on its line
        out = arms.strip_agents_md(AGENTS_MD + """
| `hw-docs/INDEX.md` | routing table |
| `hw-docs/
fetch.sh` | materialize the corpus |
| `hw-docs/eval/questions.yaml` | golden question set (issue #24) |
""", without=False)
        self.assertIn("hw-docs/INDEX.md", out)          # layer stays
        self.assertIn("fetch.sh", out)                  # layer stays
        self.assertNotIn("eval/questions.yaml", out)    # answer key row goes
        self.assertNotIn("golden question set", out)    # …and its remainder
        self.assertIn("Hardware questions", out)


class StripReadmeTest(unittest.TestCase):
    def test_without_drops_the_index_link(self):
        out = arms.strip_readme(README_MD)
        self.assertNotIn("hw-docs", out)
        self.assertIn("Field notes & lessons", out)


class StripSetupTest(unittest.TestCase):
    def test_without_removes_corpus_block_and_checklist_row(self):
        out = arms.strip_setup(SETUP_MD)
        self.assertNotIn("hw-docs", out)
        self.assertNotIn("fetch.sh", out)
        self.assertNotIn("docling", out)
        self.assertIn("YOLO engine", out)   # checklist keeps its other rows
        self.assertIn("**Verify:**", out)


class BuildArmTest(unittest.TestCase):
    def _kit(self, root: Path):
        (root / "agent").mkdir(parents=True)
        (root / "agent" / "AGENTS.md").write_text(AGENTS_MD, encoding="utf-8")
        (root / "README.md").write_text(README_MD, encoding="utf-8")
        (root / "agent" / "SETUP.md").write_text(SETUP_MD, encoding="utf-8")
        hw = root / "agent" / "hw-docs"
        hw.mkdir()
        (hw / "INDEX.md").write_text("# routing table\n", encoding="utf-8")
        (hw / "search.py").write_text("print()\n", encoding="utf-8")
        (hw / "README.md").write_text("layer docs\n", encoding="utf-8")
        (hw / "eval").mkdir()
        (hw / "eval" / "questions.yaml").write_text("[]\n", encoding="utf-8")
        (hw / "explore").mkdir()
        (hw / "explore" / "FINDINGS.md").write_text("spike\n", encoding="utf-8")
        (hw / "md").mkdir()
        (hw / "md" / "datasheet.md").write_text("doc\n", encoding="utf-8")
        (hw / "md" / "index").mkdir()
        (hw / "md" / "index" / "datasheet.chunks.jsonl").write_text("{}\n",
                                                                   encoding="utf-8")

    def test_build_without_leaves_index_unreferenced_and_corpus_absent(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src"
            dst = Path(td) / "without"
            src.mkdir()
            self._kit(src)
            arms.build_without(src, dst)
            hw = dst / "agent" / "hw-docs"
            self.assertEqual([p.name for p in hw.iterdir()], ["INDEX.md"])
            for f in ("agent/AGENTS.md", "README.md", "agent/SETUP.md"):
                self.assertNotIn("hw-docs",
                                 (dst / f).read_text(encoding="utf-8"), f)
            problems = arms.verify_without(dst)
            self.assertEqual(problems, [])

    def test_build_with_keeps_corpus_drops_answer_keys(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src"
            dst = Path(td) / "with"
            src.mkdir()
            self._kit(src)
            arms.build_with(src, dst)
            hw = dst / "agent" / "hw-docs"
            self.assertFalse((hw / "eval").exists())
            self.assertFalse((hw / "explore").exists())
            self.assertTrue((hw / "INDEX.md").exists())
            self.assertTrue((hw / "search.py").exists())
            agents_md = (dst / "agent" / "AGENTS.md").read_text(encoding="utf-8")
            self.assertIn("hw-docs/INDEX.md", agents_md)
            self.assertNotIn("eval/questions.yaml", agents_md)
            self.assertEqual(arms.verify_with(dst), [])


if __name__ == "__main__":
    unittest.main()
