import unittest
from pathlib import Path
from unittest.mock import patch

import navai.pipeline as pipeline
from navai.prompts import SYSTEM_INSTRUCTION
from navai.settings import GEMINI_MODEL, GROQ_MODEL, OPENROUTER_MODEL

CONFIG = {
    "chunk_size": 128,
    "retrieval_k": 7,
    "crossencoder_k": 2,
    "temperature": 0.1,
}


def sample_chunks():
    return [
        {
            "text": "a",
            "metadata": {"doc": "A", "chunk_index": 0},
            "distance": 0.1,
            "score": 0.9,
        }
    ]


class PipelineTest(unittest.TestCase):
    def build_bot(self, model=None, env=None):
        with patch("navai.pipeline.load_config", return_value=CONFIG), patch(
            "navai.pipeline.SentenceTransformer"
        ), patch("navai.pipeline.CrossEncoder"), patch(
            "navai.pipeline.chromadb.PersistentClient"
        ) as client, patch("navai.pipeline.BM25Index.from_collection"), patch(
            "navai.pipeline.load_dotenv"
        ), patch("navai.pipeline.genai.Client") as gemini_client, patch(
            "navai.pipeline.OpenAI"
        ) as openai_client, patch.dict("os.environ", env or {}, clear=True):
            bot = pipeline.NavAIChat(model=model)
        return bot, client, gemini_client, openai_client

    def test_defaults_to_gemini(self):
        bot, client, gemini_client, openai_client = self.build_bot(
            env={"GOOGLE_API_KEY": "test"}
        )

        self.assertEqual(bot.model, GEMINI_MODEL)
        self.assertEqual(bot.provider, "gemini")
        self.assertEqual(bot.reasoning_effort, "default")
        self.assertEqual(bot.retrieval_k, 7)
        self.assertEqual(bot.crossencoder_k, 2)
        self.assertEqual(bot.temperature, 0.1)
        self.assertEqual(Path(client.call_args.kwargs["path"]).name, "chroma_128")
        gemini_client.assert_called_once_with(api_key="test")
        openai_client.assert_not_called()
        self.assertIsNone(bot.groq)

        bot.gemini.models.generate_content.return_value.text = "Answer [1]."
        with patch("navai.pipeline.retrieve_hybrid", return_value=[]) as retrieve, patch(
            "navai.pipeline.rerank", return_value=sample_chunks()
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
        generation = bot.gemini.models.generate_content.call_args
        self.assertEqual(generation.kwargs["model"], GEMINI_MODEL)
        self.assertEqual(generation.kwargs["contents"], prepared["prompt"])
        config = generation.kwargs["config"]
        self.assertEqual(config.system_instruction, SYSTEM_INSTRUCTION)
        self.assertEqual(config.temperature, 1.0)
        self.assertEqual(config.max_output_tokens, 2048)
        self.assertEqual(result["answer"], "Answer [1].")

    def test_empty_gemini_text_uses_fallback(self):
        bot, *_ = self.build_bot(env={"GOOGLE_API_KEY": "test"})
        bot.gemini.models.generate_content.return_value.text = None
        with patch("navai.pipeline.retrieve_hybrid", return_value=[]), patch(
            "navai.pipeline.rerank", return_value=sample_chunks()
        ):
            result = bot.answer("question")

        self.assertEqual(
            result["answer"],
            "I could not generate an answer. Please rephrase the question.",
        )

    def test_gemini_requires_google_key(self):
        with self.assertRaisesRegex(RuntimeError, "GOOGLE_API_KEY"):
            self.build_bot(env={})

    def test_openrouter_model_uses_openai_client(self):
        bot, _, gemini_client, openai_client = self.build_bot(
            model=OPENROUTER_MODEL, env={"OPEN_API_KEY": "test"}
        )

        self.assertEqual(bot.provider, "openrouter")
        gemini_client.assert_not_called()
        openai_client.assert_called_once()
        self.assertIsNone(bot.gemini)

        bot.groq.chat.completions.create.return_value.choices[
            0
        ].message.content = "Answer [1]."
        with patch("navai.pipeline.retrieve_hybrid", return_value=[]), patch(
            "navai.pipeline.rerank", return_value=sample_chunks()
        ):
            prepared = bot.prepare("question")
            result = bot.answer("question")

        generation = bot.groq.chat.completions.create.call_args
        self.assertEqual(generation.kwargs["model"], OPENROUTER_MODEL)
        self.assertNotIn("reasoning_effort", generation.kwargs)
        self.assertEqual(generation.kwargs["max_tokens"], 2048)
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

    def test_groq_model_uses_groq_client(self):
        with patch("navai.pipeline.Groq") as groq_client:
            bot, _, gemini_client, openai_client = self.build_bot(
                model=GROQ_MODEL, env={"GROQ_API_KEY": "test"}
            )

        self.assertEqual(bot.provider, "groq")
        gemini_client.assert_not_called()
        openai_client.assert_not_called()
        groq_client.assert_called_once_with(api_key="test")
        self.assertIsNone(bot.gemini)

        bot.groq.chat.completions.create.return_value.choices[
            0
        ].message.content = "Answer [1]."
        with patch("navai.pipeline.retrieve_hybrid", return_value=[]), patch(
            "navai.pipeline.rerank", return_value=sample_chunks()
        ):
            result = bot.answer("question")

        generation = bot.groq.chat.completions.create.call_args
        self.assertEqual(generation.kwargs["reasoning_effort"], "default")
        self.assertEqual(generation.kwargs["max_completion_tokens"], 2048)
        self.assertEqual(result["answer"], "Answer [1].")

    def test_groq_requires_groq_key(self):
        with self.assertRaisesRegex(RuntimeError, "GROQ_API_KEY"):
            self.build_bot(model=GROQ_MODEL, env={})


if __name__ == "__main__":
    unittest.main()
