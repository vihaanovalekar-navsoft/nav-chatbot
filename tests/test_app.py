import unittest

import os

from navai import pipeline
from navai.settings import ROOT
from streamlit.testing.v1 import AppTest

APP_PATH = str(ROOT / "app.py")


class FakeChat:
    def __init__(self):
        self.history = None

    def answer(self, question, history=None):
        self.history = history
        return {"answer": f"Echo: {question}", "sources": []}


class AppTestSuite(unittest.TestCase):
    def setUp(self):
        self.patcher = None
        self._original_chat = pipeline.NavAIChat
        pipeline.NavAIChat = FakeChat

    def tearDown(self):
        pipeline.NavAIChat = self._original_chat

    def _run_app(self):
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()
        return at

    def test_serves_chat_page(self):
        at = self._run_app()

        self.assertEqual(at.exception, [])
        self.assertTrue(any(t.value == "NavAI Assistant" for t in at.title))
        self.assertEqual(len(at.chat_input), 1)

    def test_answers_chat_message(self):
        at = self._run_app()
        at.chat_input[0].set_value("hello").run()

        self.assertEqual(at.exception, [])
        self.assertTrue(any("Echo: hello" in m.value for m in at.markdown))

    def test_passes_history_to_pipeline(self):
        at = self._run_app()
        at.chat_input[0].set_value("first").run()
        at.chat_input[0].set_value("second").run()

        # The FakeChat instance is stored in session state; its last history
        # should include the prior user and assistant turns.
        # Re-run a fresh app and inspect via a capturing fake.
        captured = {}

        class CapturingChat:
            def __init__(self):
                self.history = None

            def answer(self, question, history=None):
                captured["history"] = history
                return {"answer": f"Echo: {question}", "sources": []}

        pipeline.NavAIChat = CapturingChat
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()
        at.chat_input[0].set_value("first").run()
        at.chat_input[0].set_value("second").run()

        self.assertEqual(
            captured["history"],
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "Echo: first"},
            ],
        )

    def test_rejects_blank_message(self):
        at = self._run_app()
        at.chat_input[0].set_value("   ").run()

        self.assertTrue(len(at.error) > 0 or at.exception != [])

    def test_rejects_too_long_message(self):
        at = self._run_app()
        at.chat_input[0].set_value("x" * 2001).run()

        self.assertTrue(len(at.error) > 0 or at.exception != [])


if __name__ == "__main__":
    unittest.main()
