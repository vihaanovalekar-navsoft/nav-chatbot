import unittest

from navai.answers import build_result, sources_for


def sample_chunks():
    return [
        {
            "text": "a",
            "metadata": {"doc": "A", "chunk_index": 0},
            "distance": 0.1,
            "score": 0.9,
        },
        {
            "text": "b",
            "metadata": {"doc": "B", "chunk_index": 1},
            "distance": 0.2,
            "score": 0.8,
        },
    ]


class AnswerTest(unittest.TestCase):
    def test_hides_sources_without_citation(self):
        self.assertEqual(sources_for("No citation.", sample_chunks()), [])

    def test_shows_sources_with_citation(self):
        sources = sources_for("Answer [1].", sample_chunks())

        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["doc"], "A")
        self.assertEqual(sources[0]["rerank"], 0.9)

    def test_shows_sources_with_grouped_citation(self):
        sources = sources_for("Answer [1, 2].", sample_chunks())

        self.assertEqual(len(sources), 2)

    def test_shows_sources_with_tag_citation(self):
        sources = sources_for("Answer <1>.", sample_chunks())

        self.assertEqual(len(sources), 2)

    def test_keeps_missing_distance_as_none(self):
        chunks = sample_chunks()
        chunks[0]["distance"] = None

        sources = sources_for("Answer [1].", chunks)

        self.assertIsNone(sources[0]["distance"])

    def test_empty_text_gets_fallback(self):
        result = build_result(None, sample_chunks())

        self.assertEqual(
            result["answer"],
            "I could not generate an answer. Please rephrase the question.",
        )
        self.assertEqual(result["sources"], [])

    def test_strips_answer_text(self):
        result = build_result("  Answer [1].  ", sample_chunks())

        self.assertEqual(result["answer"], "Answer [1].")


if __name__ == "__main__":
    unittest.main()
