"""Synthetic tests for build.py — the torch/docling-free parts.

The chunker itself (docling_core) and the embedder (torch) are exercised
operator-side by the real fetch.sh run; here we pin the pure orchestration:
shard writing/reading round-trip, wrap-table merge, md finalization, and
the xlsx path. docling_core-dependent tiers SKIP loudly when absent —
a skip is never a pass.
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import build  # noqa: E402


class WriteShardRoundTrip(unittest.TestCase):
    def test_atomic_shard_files(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        recs = [{"doc": "d", "headings": ["1 Intro"], "pages": [2],
                 "text": "hello"}]
        emb = np.array([[0.5, 0.5]], dtype="float32")
        build.write_shard(tmp, "d", recs, emb, {"doc": "d", "model": "m", "dim": 2})
        self.assertTrue((tmp / "d.chunks.jsonl").is_file())
        self.assertTrue((tmp / "d.emb.npy").is_file())
        self.assertEqual(len(list(tmp.glob("*.tmp"))), 0)  # no strays
        back = [json.loads(l) for l in
                (tmp / "d.chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(back, recs)
        np.testing.assert_array_equal(np.load(tmp / "d.emb.npy"), emb)
        self.assertEqual(json.loads((tmp / "d.meta.json").read_text())["dim"], 2)


class WrapMerge(unittest.TestCase):
    def test_merge_keeps_prior_entries(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        (tmp / "wrap_table.json").write_text(json.dumps(
            [{"wrapped": "A_B C_D", "joined": "A_BC_D"}]), encoding="utf-8")
        merged = build.merge_wrap_table(
            tmp, ["X_Y Z_W is a wrap.", "X_YZ_W occurs whole."])
        pairs = {(e["wrapped"], e["joined"]) for e in merged}
        self.assertIn(("A_B C_D", "A_BC_D"), pairs)
        self.assertIn(("X_Y Z_W", "X_YZ_W"), pairs)
        # longest-first application order
        lens = [len(e["wrapped"]) for e in merged]
        self.assertEqual(lens, sorted(lens, reverse=True))


class FinalizeMd(unittest.TestCase):
    def test_parts_concatenate_and_normalize(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        p1 = tmp / "a.p1-2.raw.md"
        p2 = tmp / "a.p3-4.raw.md"
        p1.write_text("# part one\n\nM\\_TTCAN intro\n", encoding="utf-8")
        p2.write_text("part two\n", encoding="utf-8")
        build.finalize_md(tmp, tmp, "a", [p1, p2],
                          [{"wrapped": "X Y_Z", "joined": "XY_Z"}])
        out = (tmp / "a.md").read_text(encoding="utf-8")
        self.assertEqual(out, "# part one\n\nM_TTCAN intro\npart two\n")
        self.assertFalse(p1.exists())  # parts cleaned up


class ChunkerTier(unittest.TestCase):
    """Provenance contract of chunk_records, exercised against the real
    #27 spike JSON where present (operator machine); elsewhere SKIP is
    loud — a skip is never a pass."""

    SPIKE = Path(__file__).parent / "explore" / "devkit-carrier-spec.json"

    def test_button_table_chunk_carries_page_28(self):
        try:
            from docling_core.types.doc.document import DoclingDocument
        except ImportError:
            self.skipTest("docling_core not installed (operator machine only)")
        if not self.SPIKE.is_file():
            self.skipTest(f"spike JSON not fetched: {self.SPIKE}")
        doc = DoclingDocument.load_from_json(self.SPIKE)
        records = build.chunk_records(doc, "devkit-carrier-spec")
        self.assertTrue(records)
        hits = [r for r in records if "SYS_RESET" in r["text"]]
        self.assertTrue(hits, "button-table row not chunked")
        self.assertEqual(hits[0]["pages"], [28])  # golden-set ground truth
        self.assertTrue(hits[0]["headings"])
        self.assertIn("3.4", hits[0]["headings"][-1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
