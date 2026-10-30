import unittest

from drill_ir.text_features import article_static_features, chunk_pair_features, normalized_tokens


class TextFeatureTest(unittest.TestCase):
    def test_normalizes_and_measures_legal_overlap(self) -> None:
        query = "Theo Điều 3, người lao động được gì?"
        chunk = "Điều 3 quy định quyền của người lao động."

        features = chunk_pair_features(query, chunk)

        self.assertIn("điều", normalized_tokens(query))
        self.assertGreater(features["query_chunk_overlap_count"], 0)
        self.assertGreater(features["query_token_coverage"], 0.0)
        self.assertEqual(features["legal_citation_overlap_count"], 1)

    def test_builds_static_article_structure_features(self) -> None:
        features = article_static_features("Điều 3\n1. Quy định\na) Nội dung 2025")

        self.assertEqual(features["article_clause_marker_count"], 1)
        self.assertEqual(features["article_point_marker_count"], 1)
        self.assertGreater(features["article_digit_token_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
