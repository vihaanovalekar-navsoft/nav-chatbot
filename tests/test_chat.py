import contextlib
import io
import sys
import unittest
from unittest.mock import patch

from chat import main


class FakeChat:
    def answer(self, question):
        return {
            "answer": "Test answer [1].",
            "sources": [
                {
                    "label": 1,
                    "doc": "Test Doc",
                    "chunk_index": 2,
                    "rerank": 1.2345,
                    "distance": 0.5,
                }
            ],
        }


class CliTest(unittest.TestCase):
    def test_prints_answer_and_sources(self):
        output = io.StringIO()

        with patch.object(sys, "argv", ["chat.py", "What is NavAI?"]), patch(
            "chat.NavAIChat", FakeChat
        ), contextlib.redirect_stdout(output):
            main()

        text = output.getvalue()
        self.assertIn("Test answer [1].", text)
        self.assertIn("Sources:", text)
        self.assertIn("[1] Test Doc | chunk 2", text)


if __name__ == "__main__":
    unittest.main()
