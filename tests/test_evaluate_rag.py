import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import navai.evaluate_rag as evaluate_rag
from navai.eval_judge import JUDGE_INSTRUCTIONS
from navai.prompts import PROMPT_VERSION, SYSTEM_INSTRUCTION


class EvaluateRagTest(unittest.TestCase):
    def test_main_embeds_prompts_and_retrieval_metadata(self):
        case = {
            "question": "What is NavAI?",
            "answerable": True,
            "ideal_answer": "An analytics platform.",
            "key_points": ["analytics"],
        }
        row = {
            "score": 1,
            "context_tokens": 5,
            "answer_words": 2,
        }
        captured = {}

        def capture(output, report):
            captured["output"] = output
            captured["report"] = report

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            checkpoint = Path(directory) / "checkpoint.json"
            with patch.object(
                evaluate_rag, "NavAIChat"
            ) as chat_class, patch.object(
                evaluate_rag.genai, "Client"
            ) as client_class, patch.object(
                evaluate_rag, "load_cases", return_value=[case]
            ), patch.object(
                evaluate_rag, "count_tokens", return_value=10
            ), patch.object(
                evaluate_rag, "run_case", return_value=row
            ) as run_case, patch.object(
                evaluate_rag, "write_report", side_effect=capture
            ), patch.object(
                sys,
                "argv",
                [
                    "evaluate_rag.py",
                    "--output",
                    str(output),
                    "--checkpoint",
                    str(checkpoint),
                ],
            ), patch.dict(
                "os.environ", {"GOOGLE_API_KEY": "test"}
            ), contextlib.redirect_stdout(io.StringIO()):
                evaluate_rag.main()
            saved_rows = json.loads(checkpoint.read_text())

        report = captured["report"]
        self.assertEqual(report["prompt"], SYSTEM_INSTRUCTION)
        self.assertEqual(report["judge_prompt"], JUDGE_INSTRUCTIONS)
        self.assertEqual(report["config"]["prompt_version"], PROMPT_VERSION)
        self.assertEqual(report["config"]["judge_version"], "j1")
        self.assertEqual(report["config"]["retrieval_mode"], "dense+bm25_rrf")
        self.assertEqual(report["config"]["reasoning_effort"], "default")
        self.assertEqual(report["config"]["answer_model"], "qwen/qwen3.8-27b")
        self.assertEqual(report["config"]["judge_model"], "qwen/qwen3.8-27b")
        self.assertEqual(report["config"]["rpm"], 30)
        self.assertEqual(
            report["config"]["tokenizer_model"], "gemini-3.5-flash-lite"
        )
        self.assertIs(run_case.call_args.args[4], client_class.return_value)
        self.assertIs(run_case.call_args.args[0], chat_class.return_value)
        self.assertEqual(captured["output"], output)
        self.assertEqual(saved_rows, [row])

    def test_resume_skips_saved_rows(self):
        case = {
            "question": "What is NavAI?",
            "answerable": True,
            "ideal_answer": "An analytics platform.",
            "key_points": ["analytics"],
        }
        row = {
            "number": 1,
            "score": 1,
            "context_tokens": 5,
            "answer_words": 2,
        }
        captured = {}

        def capture(output, report):
            captured["report"] = report

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            checkpoint.write_text(json.dumps([row]), encoding="utf-8")
            output = Path(directory) / "report.html"
            with patch.object(
                evaluate_rag, "NavAIChat"
            ), patch.object(
                evaluate_rag.genai, "Client"
            ), patch.object(
                evaluate_rag, "load_cases", return_value=[case]
            ), patch.object(
                evaluate_rag, "count_tokens", return_value=10
            ), patch.object(
                evaluate_rag, "run_case"
            ) as run_case, patch.object(
                evaluate_rag, "write_report", side_effect=capture
            ), patch.object(
                sys,
                "argv",
                [
                    "evaluate_rag.py",
                    "--output",
                    str(output),
                    "--checkpoint",
                    str(checkpoint),
                    "--resume",
                ],
            ), patch.dict(
                "os.environ", {"GOOGLE_API_KEY": "test"}
            ), contextlib.redirect_stdout(io.StringIO()):
                evaluate_rag.main()

        run_case.assert_not_called()
        self.assertEqual(captured["report"]["results"], [row])

    def test_default_reasoning_effort_for_gpt_oss(self):
        self.assertEqual(
            evaluate_rag.default_reasoning_effort("openai/gpt-oss-120b"),
            "low",
        )
        self.assertEqual(
            evaluate_rag.default_reasoning_effort("qwen/qwen3.8-27b"),
            "default",
        )


if __name__ == "__main__":
    unittest.main()
