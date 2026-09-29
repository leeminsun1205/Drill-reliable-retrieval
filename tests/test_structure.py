import unittest

from drill_ir.structure import parse_clause_spans


class ClauseParserTest(unittest.TestCase):
    def test_parses_sequential_line_markers_and_preserves_lead_in(self) -> None:
        text = "Phạm vi áp dụng\n1. Khoản một\n2. Khoản hai\n3. Khoản ba"
        parsed = parse_clause_spans(text)
        self.assertEqual(parsed.status, "parsed")
        self.assertEqual(parsed.detected_numbers, (1, 2, 3))
        self.assertEqual([text[span.char_start : span.char_end] for span in parsed.spans], [
            "Phạm vi áp dụng\n1. Khoản một\n",
            "2. Khoản hai\n",
            "3. Khoản ba",
        ])

    def test_accepts_sequence_starting_above_one(self) -> None:
        parsed = parse_clause_spans("2. Hai\n3. Ba")
        self.assertEqual(parsed.status, "parsed")
        self.assertEqual(parsed.detected_numbers, (2, 3))

    def test_restarted_or_gapped_sequence_is_ambiguous(self) -> None:
        for text in ("1. Một\n3. Ba", "1. Một\n2. Hai\n2. Hai lại\n3. Ba"):
            with self.subTest(text=text):
                self.assertEqual(parse_clause_spans(text).status, "ambiguous_sequence")

    def test_decimal_and_glued_markers_are_not_forced_into_structure(self) -> None:
        text = "4.1 Phụ lục\n4.2 Nội dung\n1.Không có khoảng trắng"
        self.assertEqual(parse_clause_spans(text).status, "no_structure")


if __name__ == "__main__":
    unittest.main()
