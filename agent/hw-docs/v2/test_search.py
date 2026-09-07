"""Synthetic tests for v2/search.py — run anywhere with numpy.

Builds a tiny fake index (hand-made vectors, no torch), injects a fake
embedder, and checks the provenance-carrying output contract.
"""
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import search  # noqa: E402


def make_index(tmp: Path):
    """Three chunks; vectors are one-hot-ish so a chosen query vector
    deterministically selects the expected chunk."""
    recs = [
        {"doc": "devkit-carrier-spec", "headings": ["3.4 Button Header"],
         "pages": [28], "text": "SYS_RESET* reset button pin 8"},
        {"doc": "devkit-carrier-spec", "headings": ["3.3 Expansion Header"],
         "pages": [26], "text": "GP70_UART1_T XD_BOOT2_STRAP"},
        {"doc": "orin-thermal-design-guide",
         "headings": ["2. Scope", "2.1 Airflow"], "pages": [4, 5],
         "text": "heatsink airflow CFM"},
    ]
    vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
    for i, (r, v) in enumerate(zip(recs, vecs)):
        shard = tmp / f"doc{i}"
        with open(shard.with_suffix(".chunks.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
        np.save(shard.with_suffix(".emb.npy"), v.reshape(1, -1))
    # wrap table: joined strap name must be reconstructed at load time
    (tmp / "wrap_table.json").write_text(json.dumps(
        [{"wrapped": "GP70_UART1_T XD_BOOT2_STRAP",
          "joined": "GP70_UART1_TXD_BOOT2_STRAP"}]), encoding="utf-8")
    return recs


class SearchTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())
        make_index(self.tmp)

    def run_search(self, *argv, embed):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = search.main([*argv], embed_fn=embed)
        return code, buf.getvalue()

    def test_hit_carries_section_and_page(self):
        code, out = self.run_search("reset button", "--index-dir", str(self.tmp),
                                    "--k", "2", embed=lambda q: np.array([1, 0, 0], "float32"))
        self.assertEqual(code, 0)
        self.assertIn("devkit-carrier-spec §3.4 Button Header (p. 28)", out)
        self.assertIn("SYS_RESET*", out)

    def test_wrap_table_applied_at_load(self):
        code, out = self.run_search("strap", "--index-dir", str(self.tmp),
                                    embed=lambda q: np.array([0, 1, 0], "float32"))
        self.assertEqual(code, 0)
        self.assertIn("GP70_UART1_TXD_BOOT2_STRAP", out)
        self.assertNotIn("XD_BOOT2_STRAP ", out)

    def test_json_mode_provenance_fields(self):
        code, out = self.run_search("air", "--index-dir", str(self.tmp), "--json",
                                    embed=lambda q: np.array([0, 0, 1], "float32"))
        self.assertEqual(code, 0)
        hit = json.loads(out)[0]
        self.assertEqual(hit["doc"], "orin-thermal-design-guide")
        self.assertEqual(hit["section"], "2.1 Airflow")
        self.assertEqual(hit["breadcrumb"], "2. Scope > 2.1 Airflow")
        self.assertEqual(hit["pages"], [4, 5])

    def test_doc_filter_and_unknown_doc(self):
        code, out = self.run_search("reset", "--index-dir", str(self.tmp),
                                    "--doc", "no-such-doc",
                                    embed=lambda q: np.array([1, 0, 0], "float32"))
        self.assertEqual(code, 1)  # index exists, doc unknown to it
        code, out = self.run_search("reset", "--index-dir", str(self.tmp),
                                    "--doc", "devkit-carrier-spec",
                                    embed=lambda q: np.array([1, 0, 0], "float32"))
        self.assertEqual(code, 0)

    def test_no_index_exit_2(self):
        code, out = self.run_search("x", "--index-dir", str(self.tmp / "nope"),
                                    embed=lambda q: None)
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
