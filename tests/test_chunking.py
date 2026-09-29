import unittest

from drill_ir.data import Article
from drill_ir.preprocessing import (
    ArticleTruncationChunker,
    DiemChunker,
    KhoanChunker,
    RecursiveTokenChunker,
    RecursiveTokenChunkerConfig,
    TokenWindowChunker,
    TokenWindowChunkerConfig,
)


class FakeFastTokenizer:
    is_fast = True

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
        truncation: bool = False,
        verbose: bool = True,
    ) -> dict[str, list]:
        del truncation, verbose
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for token in text.split():
            start = text.index(token, cursor)
            end = start + len(token)
            offsets.append((start, end))
            cursor = end
        token_count = len(offsets) + (2 if add_special_tokens else 0)
        result: dict[str, list] = {"input_ids": list(range(token_count))}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        self.last_pair = pair
        return 2


class ChunkingTest(unittest.TestCase):
    def test_article_control_preserves_raw_text_and_mapping(self) -> None:
        article = Article(aid=7, law_id="law", text="nguyên văn")
        chunks = ArticleTruncationChunker().chunk_articles([article])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].aid, 7)
        self.assertEqual(chunks[0].text, article.text)
        self.assertEqual(chunks[0].source_spans, ((0, len(article.text)),))

    def test_token_windows_respect_total_budget_overlap_and_offsets(self) -> None:
        article = Article(aid=9, law_id="law", text="aa bb cc dd ee")
        chunker = TokenWindowChunker(
            TokenWindowChunkerConfig(
                model_name="fake",
                max_tokens=5,
                overlap_tokens=1,
                document_prefix="passage ",
            ),
            tokenizer=FakeFastTokenizer(),
        )
        chunks = chunker.chunk_articles([article])
        # Five total tokens minus one prefix and two special tokens leaves two
        # content tokens per window, with one-token overlap.
        self.assertEqual([chunk.text for chunk in chunks], ["aa bb", "bb cc", "cc dd", "dd ee"])
        self.assertEqual([chunk.chunk_index for chunk in chunks], [0, 1, 2, 3])
        for chunk in chunks:
            self.assertEqual(article.text[chunk.char_start : chunk.char_end], chunk.text)
            self.assertEqual(chunk.source_spans, ((chunk.char_start, chunk.char_end),))
            self.assertLessEqual(chunk.token_count, 5)
        self.assertEqual(chunker.last_stats["content_token_budget"], 2)
        self.assertEqual(chunker.last_stats["max_encoded_tokens_observed"], 5)

    def test_overlap_must_be_smaller_than_content_budget(self) -> None:
        chunker = TokenWindowChunker(
            TokenWindowChunkerConfig(
                model_name="fake",
                max_tokens=4,
                overlap_tokens=1,
                document_prefix="passage ",
            ),
            tokenizer=FakeFastTokenizer(),
        )
        with self.assertRaisesRegex(ValueError, "overlap_tokens"):
            chunker.chunk_articles([Article(1, "law", "aa bb")])

    def test_full_encoder_input_is_rechecked_and_shrunk(self) -> None:
        class BoundarySensitiveTokenizer(FakeFastTokenizer):
            def __call__(self, text: str, **kwargs):
                result = super().__call__(text, **kwargs)
                if kwargs["add_special_tokens"] and text.startswith("passage "):
                    result["input_ids"].append(999)
                return result

        chunker = TokenWindowChunker(
            TokenWindowChunkerConfig(
                model_name="fake",
                max_tokens=5,
                overlap_tokens=0,
                document_prefix="passage ",
            ),
            tokenizer=BoundarySensitiveTokenizer(),
        )
        chunks = chunker.chunk_articles([Article(1, "law", "aa bb cc")])
        self.assertEqual([chunk.text for chunk in chunks], ["aa", "bb", "cc"])
        self.assertTrue(all(chunk.token_count <= 5 for chunk in chunks))

    def test_recursive_mode_prefers_paragraph_boundary(self) -> None:
        article = Article(1, "law", "aa bb\n\ncc dd ee ff")
        chunker = RecursiveTokenChunker(
            RecursiveTokenChunkerConfig(
                model_name="fake",
                max_tokens=7,
                overlap_tokens=1,
                document_prefix="passage ",
                minimum_fill_ratio=0.5,
            ),
            tokenizer=FakeFastTokenizer(),
        )
        chunks = chunker.chunk_articles([article])
        self.assertEqual(chunks[0].text, "aa bb")
        self.assertEqual(chunks[0].chunk_type, "recursive_token")
        self.assertGreaterEqual(chunker.last_stats["boundary_counts"]["paragraph"], 1)
        self.assertTrue(all(chunk.token_count <= 7 for chunk in chunks))
        for chunk in chunks:
            self.assertEqual(article.text[chunk.char_start : chunk.char_end], chunk.text)

    def test_recursive_mode_rejects_invalid_minimum_fill(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimum_fill_ratio"):
            RecursiveTokenChunker(
                RecursiveTokenChunkerConfig(
                    model_name="fake",
                    minimum_fill_ratio=0.0,
                ),
                tokenizer=FakeFastTokenizer(),
            )

    def test_khoan_mode_emits_structured_chunks(self) -> None:
        article = Article(7, "law", "Dẫn nhập\n1. Khoản một\n2. Khoản hai")
        chunker = KhoanChunker(
            TokenWindowChunkerConfig(
                model_name="fake",
                max_tokens=12,
                overlap_tokens=1,
                document_prefix="passage ",
            ),
            tokenizer=FakeFastTokenizer(),
        )
        chunks = chunker.chunk_articles([article])
        self.assertEqual([chunk.structure_path for chunk in chunks], [("khoan:1",), ("khoan:2",)])
        self.assertEqual([chunk.chunk_type for chunk in chunks], ["khoan", "khoan"])
        self.assertEqual(chunker.last_stats["parse_status_counts"], {"parsed": 1})
        for chunk in chunks:
            self.assertEqual(article.text[chunk.char_start : chunk.char_end], chunk.text)
            self.assertLessEqual(chunk.token_count, 12)

    def test_khoan_mode_falls_back_for_ambiguous_sequence(self) -> None:
        article = Article(8, "law", "1. Một\n2. Hai\n2. Hai lại\n3. Ba")
        chunker = KhoanChunker(
            TokenWindowChunkerConfig(
                model_name="fake",
                max_tokens=8,
                overlap_tokens=1,
                document_prefix="passage ",
            ),
            tokenizer=FakeFastTokenizer(),
        )
        chunks = chunker.chunk_articles([article])
        self.assertTrue(all(chunk.chunk_type == "article_fallback_window" for chunk in chunks))
        self.assertTrue(all(chunk.parse_status == "ambiguous_sequence" for chunk in chunks))
        self.assertEqual(chunker.last_stats["parse_status_counts"], {"ambiguous_sequence": 1})

    def test_diem_mode_repeats_khoan_lead_in_and_tracks_source_spans(self) -> None:
        article = Article(
            9,
            "law",
            "1. Dẫn nhập:\t\ta) Điểm một;\t\tb) Điểm hai\n2. Khoản không có điểm",
        )
        chunker = DiemChunker(
            TokenWindowChunkerConfig(
                model_name="fake",
                max_tokens=12,
                overlap_tokens=1,
                document_prefix="passage ",
            ),
            tokenizer=FakeFastTokenizer(),
        )
        chunks = chunker.chunk_articles([article])
        point_chunks = [chunk for chunk in chunks if chunk.chunk_type == "diem"]
        self.assertEqual(
            [chunk.structure_path for chunk in point_chunks],
            [("khoan:1", "diem:a"), ("khoan:1", "diem:b")],
        )
        self.assertTrue(all(chunk.text.startswith("1. Dẫn nhập:\t\t") for chunk in point_chunks))
        self.assertTrue(all(len(chunk.source_spans) == 2 for chunk in point_chunks))
        self.assertEqual(chunks[-1].chunk_type, "khoan_fallback")
        self.assertLessEqual(chunker.last_stats["max_encoded_tokens_observed"], 12)

    def test_diem_mode_falls_back_for_restarted_point_sequence(self) -> None:
        article = Article(
            10,
            "law",
            "1. Dẫn:\t\ta) Một;\t\tb) Hai;\t\ta) Lại\n2. Khoản hai",
        )
        chunker = DiemChunker(
            TokenWindowChunkerConfig(
                model_name="fake",
                max_tokens=20,
                overlap_tokens=1,
                document_prefix="passage ",
            ),
            tokenizer=FakeFastTokenizer(),
        )
        chunks = chunker.chunk_articles([article])
        self.assertEqual(chunks[0].chunk_type, "khoan_fallback")
        self.assertEqual(chunks[0].parse_status, "point_ambiguous_sequence")

    def test_diem_mode_windows_long_points_with_repeated_lead_in(self) -> None:
        article = Article(
            11,
            "law",
            "1. Dẫn:\t\ta) aa bb cc dd;\t\tb) ee ff\n2. Khoản hai",
        )
        chunker = DiemChunker(
            TokenWindowChunkerConfig(
                model_name="fake",
                max_tokens=7,
                overlap_tokens=1,
                document_prefix="passage ",
            ),
            tokenizer=FakeFastTokenizer(),
        )
        chunks = chunker.chunk_articles([article])
        point_chunks = [chunk for chunk in chunks if chunk.chunk_type == "diem_token_window"]
        self.assertGreater(len(point_chunks), 1)
        self.assertTrue(all(chunk.text.startswith("1. Dẫn:\t\t") for chunk in point_chunks))
        self.assertTrue(all((chunk.token_count or 0) <= 7 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
