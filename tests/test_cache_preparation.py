import unittest

from drill_ir.retrieval.cache_preparation import configured_cache_components


class CachePreparationTest(unittest.TestCase):
    def test_lists_sparse_and_named_multi_dense_components(self) -> None:
        config = {
            "bm25": {},
            "dense": {},
            "hybrid": {
                "dense_components": [
                    {"name": "bge_m3", "profile": "bge_m3"},
                    {"name": "challenger", "profile": "challenger"},
                ]
            },
        }
        self.assertEqual(
            configured_cache_components(config),
            ["sparse", "bge_m3", "challenger"],
        )

    def test_single_dense_config_does_not_invent_sparse_component(self) -> None:
        self.assertEqual(configured_cache_components({"dense": {}}), ["dense"])


if __name__ == "__main__":
    unittest.main()
