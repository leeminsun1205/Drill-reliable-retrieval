import unittest

from drill_ir.structure import build_article_number_index, extract_reference_records


class ReferenceExtractionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            {"law_id": "10/2020/QH14", "content": [
                {"aid": 10, "content_Article": "Theo Điều 2 của Luật này."},
                {"aid": 20, "content_Article": "Nội dung hai"},
            ]},
            {"law_id": "20/2021/NĐ-CP", "content": [
                {"aid": 30, "content_Article": "Theo Điều 1 Luật số 10/2020/QH14."},
            ]},
        ]

    def test_maps_raw_position_and_resolves_local_and_cross_document_edges(self) -> None:
        index, mappings = build_article_number_index(self.documents)
        self.assertEqual(index[("10/2020/QH14", 2)].aid, 20)
        self.assertEqual(len(mappings), 3)
        records = extract_reference_records(self.documents, index)
        resolved = [record for record in records if record.target_aid is not None]
        self.assertEqual([(record.source_aid, record.target_aid) for record in resolved], [(10, 20), (30, 10)])

    def test_respects_non_article_and_missing_body_overrides(self) -> None:
        index, _ = build_article_number_index(
            self.documents,
            numbering_schemes={"20/2021/NĐ-CP": "sections"},
            excluded_article_numbers={"10/2020/QH14": [2]},
            missing_body_aids=[10],
        )
        self.assertNotIn(("20/2021/NĐ-CP", 1), index)
        self.assertNotIn(("10/2020/QH14", 2), index)
        self.assertEqual(index[("10/2020/QH14", 1)].status, "missing_body")
        self.assertFalse(index[("10/2020/QH14", 1)].candidate_eligible)

    def test_nearby_local_marker_wins_over_later_external_law_id(self) -> None:
        documents = [{
            "law_id": "14/2022/TT-NHNN",
            "content": [
                {"aid": 1, "content_Article": "Điều 2 Thông tư này thay thế Thông tư số 12/2019/TT-NHNN."},
                {"aid": 2, "content_Article": "Nội dung"},
            ],
        }]
        index, _ = build_article_number_index(documents)
        records = extract_reference_records(documents, index)
        resolved = [record for record in records if record.target_aid is not None]
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].scope, "same_document")
        self.assertEqual(resolved[0].target_aid, 2)

    def test_named_external_instrument_blocks_later_local_marker(self) -> None:
        documents = [
            {"law_id": "14/2016/TT-BNNPTNT", "content": [
                {"aid": 1, "content_Article": (
                    "Các Điều 14, Điều 15, khoản 1 Điều 19 của Luật thú y "
                    "và các quy định tại Thông tư này."
                )},
            ]},
        ]
        index, _ = build_article_number_index(documents)
        records = extract_reference_records(documents, index)
        article_19 = [record for record in records if record.article_number == 19]
        self.assertEqual(len(article_19), 1)
        self.assertEqual(article_19[0].status, "named_target_without_law_id")
        self.assertIsNone(article_19[0].target_aid)

    def test_nested_explicit_reference_does_not_capture_outer_amendment_target(self) -> None:
        documents = [
            {"law_id": "24/2022/NĐ-CP", "content": [
                {"aid": 1, "content_Article": (
                    "Sửa đổi Điều 19 (đã được sửa theo khoản 9 Điều 5 "
                    "Nghị định số 140/2018/NĐ-CP)."
                )},
            ]},
            {"law_id": "140/2018/NĐ-CP", "content": [
                {"aid": aid, "content_Article": ""} for aid in range(10, 15)
            ]},
        ]
        index, _ = build_article_number_index(documents)
        records = extract_reference_records(documents, index)
        by_number = {record.article_number: record for record in records}
        self.assertIsNone(by_number[19].target_aid)
        self.assertEqual(by_number[19].status, "amendment_target_without_law_id")
        self.assertEqual(by_number[5].target_aid, 14)

    def test_later_instrument_id_does_not_capture_named_law_reference(self) -> None:
        documents = [
            {"law_id": "134/2016/NĐ-CP", "content": [
                {"aid": 1, "content_Article": (
                    "Hàng hóa tại khoản 3 Điều 2 Luật thuế xuất khẩu, thuế nhập khẩu "
                    "thực hiện theo quy định tại Nghị định số 08/2015/NĐ-CP."
                )},
            ]},
            {"law_id": "08/2015/NĐ-CP", "content": [
                {"aid": 10, "content_Article": ""},
                {"aid": 11, "content_Article": ""},
            ]},
        ]
        index, _ = build_article_number_index(documents)
        record = extract_reference_records(documents, index)[0]
        self.assertEqual(record.status, "named_target_without_law_id")
        self.assertIsNone(record.target_aid)

    def test_quoted_article_heading_is_not_a_reference_edge(self) -> None:
        documents = [{"law_id": "24/2022/NĐ-CP", "content": [
            {"aid": 1, "content_Article": "Sửa đổi như sau:“Điều 19. Hồ sơ đăng ký."},
        ]}]
        index, _ = build_article_number_index(documents)
        record = extract_reference_records(documents, index)[0]
        self.assertEqual(record.status, "quoted_article_heading")
        self.assertIsNone(record.target_aid)

    def test_preceding_law_id_does_not_capture_later_named_law(self) -> None:
        documents = [
            {"law_id": "04/2020/TT-BTP", "content": [
                {"aid": 1, "content_Article": (
                    "Theo khoản 2 Điều 44 Nghị định số 123/2015/NĐ-CP, "
                    "quan hệ được xác định tại khoản 2 Điều 11 Luật hôn nhân và gia đình."
                )},
            ]},
            {"law_id": "123/2015/NĐ-CP", "content": [
                {"aid": aid, "content_Article": ""} for aid in range(10, 54)
            ]},
        ]
        index, _ = build_article_number_index(documents)
        records = extract_reference_records(documents, index)
        by_number = {record.article_number: record for record in records}
        self.assertEqual(by_number[44].target_aid, 53)
        self.assertEqual(by_number[11].status, "named_target_without_law_id")
        self.assertIsNone(by_number[11].target_aid)

    def test_expands_compact_same_document_article_lists(self) -> None:
        documents = [{
            "law_id": "10/2020/QH14",
            "content": [
                {"aid": 10, "content_Article": (
                    "Thực hiện theo Điều 2, 3 và 4 của Luật này."
                )},
                {"aid": 20, "content_Article": "Hai"},
                {"aid": 30, "content_Article": "Ba"},
                {"aid": 40, "content_Article": "Bốn"},
            ],
        }]
        index, _ = build_article_number_index(documents)
        records = extract_reference_records(documents, index)
        self.assertEqual(
            [(record.article_number, record.target_aid) for record in records],
            [(2, 20), (3, 30), (4, 40)],
        )

    def test_expands_compact_cross_document_article_lists(self) -> None:
        documents = [
            {"law_id": "20/2021/NĐ-CP", "content": [
                {"aid": 1, "content_Article": (
                    "Theo Điều 1, 2, 3 Nghị định số 10/2020/NĐ-CP."
                )},
            ]},
            {"law_id": "10/2020/NĐ-CP", "content": [
                {"aid": 10, "content_Article": "Một"},
                {"aid": 20, "content_Article": "Hai"},
                {"aid": 30, "content_Article": "Ba"},
            ]},
        ]
        index, _ = build_article_number_index(documents)
        records = extract_reference_records(documents, index)
        self.assertEqual(
            [(record.article_number, record.target_aid) for record in records],
            [(1, 10), (2, 20), (3, 30)],
        )

    def test_explicit_current_law_id_is_same_document_scope(self) -> None:
        documents = [{
            "law_id": "06/2019/TT-NHNN",
            "content": [
                {"aid": 10, "content_Article": (
                    "Theo Điều 2 Thông tư 06/2019/TT-NHNN."
                )},
                {"aid": 20, "content_Article": "Hai"},
            ],
        }]
        index, _ = build_article_number_index(documents)
        record = extract_reference_records(documents, index)[0]
        self.assertEqual(record.target_aid, 20)
        self.assertEqual(record.scope, "same_document")


if __name__ == "__main__":
    unittest.main()
