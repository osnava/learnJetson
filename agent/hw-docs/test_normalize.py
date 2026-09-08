"""Synthetic tests for normalize.py — run anywhere (pure stdlib).

Every case is pinned to a real artifact documented in the #27 spike
(FINDINGS.md) or to a counter-example verified in the spike outputs.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import normalize as N  # noqa: E402


class FoldEscapes(unittest.TestCase):
    def test_underscore_escape_folds(self):
        # TRM prose: M\_TTCAN (329 occurrences across the spike TRM md)
        self.assertEqual(N.fold_escapes(r"M\_TTCAN0"), "M_TTCAN0")

    def test_no_other_escapes_in_this_corpus(self):
        # verified across all spike outputs: only \_ ever appears; \z is
        # left alone (folding it is markdown-destructive, and it never
        # occurs in this corpus anyway)
        self.assertEqual(N.fold_escapes("\\_\\_x\\z"), "__x\\z")

    def test_clean_text_untouched(self):
        self.assertEqual(N.fold_escapes("GP32_UART2_TXD"), "GP32_UART2_TXD")


class Collapse(unittest.TestCase):
    def test_runs_collapse(self):
        self.assertEqual(N.collapse_ws("a \n\t b   c"), "a b c")

    def test_canonical_folds_and_collapses(self):
        self.assertEqual(N.canonical("M\\_TTCAN  \n 1"), "M_TTCAN 1")


class LocalJoins(unittest.TestCase):
    def test_trailing_underscore(self):
        # design-guide spike output: "FORCE_ RECOVERY" -> "FORCE_RECOVERY"
        self.assertEqual(N.join_local("FORCE_ RECOVERY"), "FORCE_RECOVERY")

    def test_short_tail_fragment(self):
        # carrier Table 3-3 spike output: "..._STR AP" tail
        self.assertEqual(N.join_local("GP70_UART1_T XD_BOOT2_STR AP"),
                         "GP70_UART1_T XD_BOOT2_STRAP")
        # design-guide spike output: "PEX1_CLKRE Q" -> "PEX1_CLKREQ"
        self.assertEqual(N.join_local("PEX1_CLKRE Q"), "PEX1_CLKREQ")

    def test_two_real_pin_names_never_join_locally(self):
        # ambiguous locally: both are valid names — needs the wrap table
        self.assertEqual(N.join_local("I2C2_SDA I2C2_SCL"),
                         "I2C2_SDA I2C2_SCL")

    def test_english_and_numbers_untouched(self):
        self.assertEqual(N.join_local("UART 0 Transmit"), "UART 0 Transmit")
        self.assertEqual(N.join_local("LED Cathode"), "LED Cathode")
        self.assertEqual(N.join_local("USB VBUS_A"), "USB VBUS_A")
        self.assertEqual(N.join_local("pins 7 and 8"), "pins 7 and 8")
        self.assertEqual(N.join_local("CMOS - 1.8V"), "CMOS - 1.8V")

    def test_subscript_splits_untouched(self):
        # thermal-guide spike output renders T<sub>J</sub> as "T J"; with
        # no underscore in the run there is no identifier evidence, and
        # self-supplied counts ("T J" elsewhere confirming "TJ") are
        # excluded by counting raw text only
        self.assertEqual(N.join_local("T J max"), "T J max")
        self.assertEqual(N.join_local("AC CAP value"), "AC CAP value")
        self.assertNotIn("T J", {e["wrapped"] for e in
                                 N.derive_wrap_table(["T J = 1", "T J = 2"])})
        self.assertNotIn("AC CAP", {e["wrapped"] for e in
                                    N.derive_wrap_table(["AC CAP", "AC CAP"])})


class WrapTable(unittest.TestCase):
    DOCS = [
        "| 3 | GP70_UART1_T XD_BOOT2_STR AP | strap pin |",
        "GP70_UART1_TXD_BOOT2_STRAP is a configuration strap.",
        "| 9 | I2C2_SDA | data |",
        "| 10 | I2C2_SCL | clock |",
        "I2C2_SDA and I2C2_SCL are open-drain.",
        "FORCE_ RECOVERY straps",
        "FORCE_RECOVERY is pin 10.",
    ]

    def test_derivation(self):
        table = N.derive_wrap_table(self.DOCS)
        joined = {e["wrapped"]: e["joined"] for e in table}
        self.assertEqual(joined.get("GP70_UART1_T XD_BOOT2_STRAP"),
                         "GP70_UART1_TXD_BOOT2_STRAP")
        self.assertNotIn("I2C2_SDA I2C2_SCL", joined)   # joined occurs nowhere
        self.assertNotIn("FORCE_ RECOVERY", joined)     # tier-1 already merged

    def test_apply_wraps(self):
        table = [{"wrapped": "GP70_UART1_T XD_BOOT2_STRAP",
                  "joined": "GP70_UART1_TXD_BOOT2_STRAP"}]
        self.assertEqual(
            N.canonical("strap: GP70_UART1_T XD_BOOT2_STRAP yes", table),
            "strap: GP70_UART1_TXD_BOOT2_STRAP yes")

    def test_canonical_with_table_finds_spelled_out_name(self):
        # the grader/search side: answer spells the real name, corpus has
        # the wrapped rendering — canonical() must make them equal
        table = N.derive_wrap_table(self.DOCS)
        a = N.canonical("GP70_UART1_T XD_BOOT2_STR AP", table)
        b = N.canonical("GP70_UART1_TXD_BOOT2_STRAP", table)
        self.assertEqual(a, b)

    def test_render_preserves_line_structure(self):
        table = [{"wrapped": "PEX1_CLKRE Q", "joined": "PEX1_CLKREQ"}]
        out = N.render("line one\nPEX1_CLKRE Q\n| cell |  keeps   pads |", table)
        self.assertEqual(out, "line one\nPEX1_CLKREQ\n| cell |  keeps   pads |")


if __name__ == "__main__":
    unittest.main(verbosity=2)
