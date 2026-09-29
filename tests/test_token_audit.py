import unittest

from drill_ir.preprocessing.token_audit import audit_token_lengths


class FakeAuditTokenizer:
    def __call__(
        self,
        value,
        *,
        add_special_tokens: bool,
        truncation: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ):
        del truncation, verbose
        special = 2 if add_special_tokens else 0
        if isinstance(value, str):
            return {"input_ids": value.split() + ([0, 1] if special else [])}
        ids = [text.split() + ([0, 1] if special else []) for text in value]
        result = {"input_ids": ids}
        if return_length:
            result["length"] = [len(tokens) for tokens in ids]
        return result

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        return 2 if not pair else 3


class TokenAuditTest(unittest.TestCase):
    def test_lengths_include_prefix_and_special_tokens(self) -> None:
        report = audit_token_lengths(
            ["a b", "a b c d"],
            tokenizer=FakeAuditTokenizer(),
            prefix="passage prefix ",
            thresholds=[5, 7],
            batch_size=1,
        )
        self.assertEqual(report["min"], 6)
        self.assertEqual(report["max"], 8)
        self.assertEqual(report["thresholds"]["5"]["over_count"], 2)
        self.assertEqual(report["thresholds"]["7"]["over_count"], 1)


if __name__ == "__main__":
    unittest.main()
