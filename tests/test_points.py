import unittest

from drill_ir.structure import parse_point_spans


class PointParserTest(unittest.TestCase):
    def test_parses_vietnamese_sequence_and_lead_in(self) -> None:
        text = "1. Dẫn nhập:\t\ta) Điểm a;\t\tb) Điểm b;\t\tc) Điểm c"
        parsed = parse_point_spans(text)
        self.assertEqual(parsed.status, "parsed")
        self.assertEqual(parsed.detected_letters, ("a", "b", "c"))
        self.assertEqual(text[: parsed.lead_in_end], "1. Dẫn nhập:\t\t")
        self.assertEqual([text[s.char_start:s.char_end] for s in parsed.spans], [
            "a) Điểm a;\t\t",
            "b) Điểm b;\t\t",
            "c) Điểm c",
        ])

    def test_accepts_d_then_d_stroke_sequence(self) -> None:
        parsed = parse_point_spans("Dẫn:\td) Bốn;đ) Năm;e) Sáu")
        self.assertEqual(parsed.status, "parsed")
        self.assertEqual(parsed.detected_letters, ("d", "đ", "e"))

    def test_restarted_or_gapped_sequence_is_ambiguous(self) -> None:
        self.assertEqual(
            parse_point_spans("Dẫn:\ta) Một;c) Ba").status,
            "ambiguous_sequence",
        )
        self.assertEqual(
            parse_point_spans("Dẫn:\ta) Một;b) Hai;a) Lại").status,
            "ambiguous_sequence",
        )

    def test_parenthetical_prose_and_single_marker_are_not_forced(self) -> None:
        self.assertEqual(parse_point_spans("ký hiệu a) trong câu").status, "no_structure")
        self.assertEqual(parse_point_spans("Dẫn:\ta) Một").status, "no_structure")


if __name__ == "__main__":
    unittest.main()
