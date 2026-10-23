import unittest
from types import SimpleNamespace

import httpx
from groq import BadRequestError

from eval_judge import (
    JUDGE_INSTRUCTIONS,
    JUDGE_MODEL,
    JUDGE_VERSION,
    judge_answer,
)
from prompts import prompt_fingerprint


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, content):
        self.chat = SimpleNamespace(completions=FakeCompletions(content))


class RetryCompletions:
    def __init__(self, error):
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise self.error
        message = SimpleNamespace(
            content='{"score": "1", "reason": "correct"}'
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class JudgeTest(unittest.TestCase):
    def setUp(self):
        self.case = {
            "answerable": True,
            "question": "What is NavAI?",
            "ideal_answer": "An analytics platform.",
            "key_points": ["analytics"],
        }

    def test_pins_judge_version_to_prompt_digest(self):
        self.assertEqual(
            (JUDGE_VERSION, prompt_fingerprint(JUDGE_INSTRUCTIONS)),
            ("j1", "a1ba796c17b1"),
        )

    def test_returns_score_and_reason(self):
        client = FakeClient('{"score": "1", "reason": "  correct  "}')

        score, reason = judge_answer(client, self.case, "A platform.")

        self.assertEqual(score, 1)
        self.assertEqual(reason, "correct")
        call = client.chat.completions.kwargs
        self.assertEqual(call["model"], JUDGE_MODEL)
        self.assertEqual(call["messages"][0]["content"], JUDGE_INSTRUCTIONS)
        self.assertIn("Question: What is NavAI?", call["messages"][1]["content"])
        self.assertIn(
            "Generated answer:\nA platform.", call["messages"][1]["content"]
        )
        self.assertEqual(call["temperature"], 0)
        self.assertEqual(call["max_completion_tokens"], 512)
        self.assertEqual(call["response_format"], {"type": "json_object"})

    def test_accepts_integer_score(self):
        client = FakeClient('{"score": 1, "reason": "correct"}')

        score, reason = judge_answer(client, self.case, "A platform.")

        self.assertEqual(score, 1)
        self.assertEqual(reason, "correct")

    def test_retries_without_json_mode_on_validation_failure(self):
        request = httpx.Request("POST", "https://api.groq.com")
        response = httpx.Response(400, request=request)
        error = BadRequestError(
            "json_validate_failed",
            response=response,
            body={"code": "json_validate_failed"},
        )
        completions = RetryCompletions(error)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

        score, reason = judge_answer(client, self.case, "A platform.")

        self.assertEqual(score, 1)
        self.assertEqual(reason, "correct")
        self.assertIn("response_format", completions.calls[0])
        self.assertNotIn("response_format", completions.calls[1])

    def test_passes_model_and_reasoning_effort(self):
        client = FakeClient('{"score": "1", "reason": "correct"}')

        judge_answer(
            client,
            self.case,
            "A platform.",
            model="openai/gpt-oss-120b",
            reasoning_effort="low",
        )

        call = client.chat.completions.kwargs
        self.assertEqual(call["model"], "openai/gpt-oss-120b")
        self.assertEqual(call["reasoning_effort"], "low")

    def test_rejects_missing_structured_result(self):
        client = FakeClient(None)

        with self.assertRaises(ValueError):
            judge_answer(client, self.case, "A platform.")

    def test_rejects_invalid_structured_result(self):
        client = FakeClient('{"score": "2", "reason": "bad score"}')

        with self.assertRaises(ValueError):
            judge_answer(client, self.case, "A platform.")


if __name__ == "__main__":
    unittest.main()
