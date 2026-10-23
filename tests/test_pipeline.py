import unittest
from pathlib import Path
from unittest.mock import patch

import pipeline
from prompts import SYSTEM_INSTRUCTION
from settings import OPENROUTER_MODEL


def sample_chunks():
    return [
        {
            "text": "a",
            "metadata": {"doc": "A", "chunk_index": 0},
            "distance": 0.1,
            "score": 0.9,
        }
    ]


class PipelineConfigTest(unittest.TestCase):
    def test_uses_config_values(self):
        config = {
            "chunk_size": 128,
            "retrieval_k": 7,
            "crossencoder_k": 2,
            "temperature": 0.1,
        }

        with patch("pipeline.load_config", return_value=config), patch(
            "pipeline.SentenceTransformer"
        ), patch("pipeline.CrossEncoder"), patch(
            "pipeline.chromadb.PersistentClient"
        ) as client, patch("pipeline.OpenAI"), patch(
            "pipeline.BM25Index.from_collection"
        ), patch("pipeline.load_dotenv"), patch.dict(
            "os.environ", {"OPEN_API_KEY": "test"}
        ):
            bot = pipeline.NavAIChat()

        self.assertEqual(bot.model, OPENROUTER_MODEL)
        self.assertEqual(bot.reasoning_effort, "default")
        self.assertEqual(bot.retrieval_k, 7)
        self.assertEqual(bot.crossencoder_k, 2)
        self.assertEqual(bot.temperature, 0.1)
        self.assertEqual(Path(client.call_args.kwargs["path"]).name, "chroma_128")

        bot.groq.chat.completions.create.return_value.choices[
            0
        ].message.content = "Answer [1]."
        with patch(
            "pipeline.retrieve_hybrid", return_value=[]
        ) as retrieve, patch(
            "pipeline.rerank", return_value=sample_chunks()
        ) as rerank:
            prepared = bot.prepare("question")
            result = bot.answer("question")

        self.assertIn("<1>\nA (chunk 0)\na\n</1>", prepared["context"])
        self.assertEqual(
            prepared["prompt"],
            "Context:\n<1>\nA (chunk 0)\na\n</1>\n\nQuestion: question",
        )
        self.assertEqual(retrieve.call_args.args[4], 7)
        self.assertIs(retrieve.call_args.args[3], bot.bm25)
        self.assertEqual(rerank.call_args.args[3], 2)
        generation = bot.groq.chat.completions.create.call_args
        self.assertEqual(generation.kwargs["model"], OPENROUTER_MODEL)
        self.assertNotIn("reasoning_effort", generation.kwargs)
        self.assertEqual(generation.kwargs["temperature"], 0.1)
        self.assertEqual(
            generation.kwargs["messages"][0],
            {"role": "system", "content": SYSTEM_INSTRUCTION},
        )
        self.assertEqual(
            generation.kwargs["messages"][1],
            {"role": "user", "content": prepared["prompt"]},
        )
        self.assertEqual(result["answer"], "Answer [1].")


if __name__ == "__main__":
    unittest.main()
