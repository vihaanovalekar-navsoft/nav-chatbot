import unittest

from navai.retrieval import (
    BM25Index,
    build_search_query,
    reciprocal_rank_fusion,
    rerank,
)


class LengthReranker:
    """Scores each pair by the character length of the passage."""

    def predict(self, pairs):
        return [float(len(text)) for _, text in pairs]


class FakeCollection:
    def get(self, include):
        self.include = include
        return {
            "ids": ["a", "b"],
            "documents": ["NavAI analytics platform", "pricing plans"],
            "metadatas": [{"doc": "A"}, {"doc": "B"}],
        }


def chunk(chunk_id, text, distance=None):
    return {
        "id": chunk_id,
        "text": text,
        "metadata": {"doc": chunk_id.upper()},
        "distance": distance,
    }


class SearchQueryTest(unittest.TestCase):
    def test_builds_search_query_from_previous_user_turn(self):
        history = [
            {"role": "user", "content": "How does NavAI help retailers?"},
            {"role": "assistant", "content": "It helps with analytics."},
        ]

        self.assertEqual(
            build_search_query("What about pricing?", history),
            "What about pricing?\nHow does NavAI help retailers?",
        )

    def test_builds_search_query_without_user_history(self):
        history = [{"role": "assistant", "content": "Hello"}]

        self.assertEqual(
            build_search_query("What is NavAI?", history), "What is NavAI?"
        )


class BM25Test(unittest.TestCase):
    def test_builds_from_collection(self):
        collection = FakeCollection()

        index = BM25Index.from_collection(collection)

        self.assertEqual(collection.include, ["documents", "metadatas"])
        self.assertEqual(
            [record["id"] for record in index.records], ["a", "b"]
        )

    def test_ranks_matching_chunk_first(self):
        index = BM25Index(
            ["a", "b"],
            ["NavAI analytics platform", "pricing plans"],
            [{"doc": "A"}, {"doc": "B"}],
        )

        results = index.search("pricing", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "b")
        self.assertGreater(results[0]["bm25_score"], 0)
        self.assertIsNone(results[0]["distance"])

    def test_returns_no_chunks_without_a_match(self):
        index = BM25Index(
            ["a"], ["NavAI analytics platform"], [{"doc": "A"}]
        )

        self.assertEqual(index.search("unrelated", top_k=4), [])

    def test_empty_index_returns_no_chunks(self):
        index = BM25Index([], [], [])

        self.assertEqual(index.search("pricing", top_k=4), [])


class FusionTest(unittest.TestCase):
    def test_fuses_rankings_by_reciprocal_rank(self):
        fused = reciprocal_rank_fusion(
            [
                [chunk("a", "a", 0.1), chunk("b", "b", 0.2)],
                [chunk("b", "b"), chunk("c", "c")],
            ],
            top_k=3,
        )

        self.assertEqual([item["id"] for item in fused], ["b", "a", "c"])
        self.assertEqual(fused[0]["distance"], 0.2)
        self.assertGreater(fused[0]["rrf"], fused[1]["rrf"])

    def test_keeps_top_k(self):
        fused = reciprocal_rank_fusion(
            [[chunk("a", "a"), chunk("b", "b")]], top_k=1
        )

        self.assertEqual([item["id"] for item in fused], ["a"])


class RerankTest(unittest.TestCase):
    def test_orders_by_score_and_keeps_top_k(self):
        chunks = rerank(
            "question",
            [
                chunk("a", "short", 0.1),
                chunk("b", "longest", 0.2),
                chunk("c", "medium", 0.3),
            ],
            LengthReranker(),
            top_k=2,
        )
        self.assertEqual(
            [item["text"] for item in chunks], ["longest", "medium"]
        )
        self.assertEqual(chunks[0]["metadata"]["doc"], "B")
        self.assertEqual(chunks[0]["distance"], 0.2)
        self.assertEqual(chunks[0]["score"], 7.0)

    def test_returns_all_when_top_k_exceeds_candidates(self):
        chunks = rerank(
            "question",
            [chunk("a", "a"), chunk("b", "bb")],
            LengthReranker(),
            top_k=5,
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual([item["text"] for item in chunks], ["bb", "a"])

    def test_no_candidates(self):
        self.assertEqual(
            rerank("question", [], LengthReranker(), top_k=4), []
        )


if __name__ == "__main__":
    unittest.main()
