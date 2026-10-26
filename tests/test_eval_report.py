import tempfile
import unittest
from pathlib import Path

from navai.eval_report import render_report, write_report


def report_data(score):
    return {
        "generated_at": "2026-01-01 00:00:00 UTC",
        "prompt": "Prompt <body> text.",
        "judge_prompt": "Judge <body> text.",
        "config": {
            "chunk_size": 256,
            "retrieval_mode": "dense+bm25_rrf",
            "retrieval_k": 20,
            "crossencoder_k": 4,
            "rrf_k": 60,
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
            "temperature": 0.1,
            "judge_temperature": 0,
            "reasoning_effort": "default",
            "rpm": 30,
            "prompt_version": "v3",
            "prompt_fingerprint": "abc123def456",
            "judge_version": "j1",
            "judge_fingerprint": "def456abc123",
            "answer_model": "qwen/qwen3.8-27b",
            "judge_model": "qwen/qwen3.8-27b",
            "tokenizer_model": "gemini-3.5-flash-lite",
            "embedding_model": "all-MiniLM-L6-v2",
            "reranker_model": "ms-marco-MiniLM-L-6-v2",
            "chroma_path": "data/chroma",
            "cases_file": "cases.json",
        },
        "system_instruction_tokens": 200,
        "results": [
            {
                "number": 1,
                "question": "What is <NavAI>?",
                "answerable": True,
                "score": score,
                "reason": "Covers <script>alert(1)</script>",
                "answer": "<script>alert(1)</script>",
                "ideal_answer": "NavAI",
                "key_points": ["NavAI"],
                "answer_words": 2,
                "answer_chars": 20,
                "context_tokens": 900,
                "prompt_tokens": 950,
                "sources": ["Doc <A> (chunk 0)"],
            }
        ],
    }


class EvalReportTest(unittest.TestCase):
    def test_renders_accuracy_configs_lengths_and_tokens(self):
        html = render_report(report_data(1))

        self.assertIn("Overall accuracy", html)
        self.assertIn("100.0%", html)
        self.assertIn("256 tokens", html)
        self.assertIn("20 chunks", html)
        self.assertIn("2 w / 20 c", html)
        self.assertIn("900", html)
        self.assertIn("1,150", html)
        self.assertIn("qwen/qwen3.8-27b", html)
        self.assertIn("gemini-3.5-flash-lite", html)
        self.assertIn("30 requests/min", html)
        self.assertIn("dense+bm25_rrf", html)
        self.assertIn("<th>RRF k</th><td>60</td>", html)
        self.assertIn("<th>BM25 k1</th><td>1.5</td>", html)
        self.assertIn("<th>BM25 b</th><td>0.75</td>", html)
        self.assertIn("<th>Reasoning effort</th><td>default</td>", html)
        self.assertIn("Judge requests are excluded.", html)

    def test_renders_prompt_version_and_escaped_prompt(self):
        html = render_report(report_data(1))

        self.assertIn("System Prompt", html)
        self.assertIn("v3", html)
        self.assertIn("abc123def456", html)
        self.assertIn("Prompt &lt;body&gt; text.", html)
        self.assertNotIn("Prompt <body> text.", html)

    def test_renders_judge_prompt_version_and_escaped_prompt(self):
        html = render_report(report_data(1))

        self.assertIn("Judge Prompt", html)
        self.assertIn("j1", html)
        self.assertIn("def456abc123", html)
        self.assertIn("Judge &lt;body&gt; text.", html)
        self.assertNotIn("Judge <body> text.", html)

    def test_escapes_generated_content(self):
        html = render_report(report_data(1))

        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&lt;NavAI&gt;", html)

    def test_renders_zero_score(self):
        html = render_report(report_data(0))

        self.assertIn("0.0%", html)

    def test_renders_answerable_and_unanswerable_splits(self):
        data = report_data(1)
        data["results"].append(
            {
                "number": 2,
                "question": "What are NavAI's pricing plans?",
                "answerable": False,
                "score": 0,
                "reason": "It does not state the information.",
                "answer": "I do not have that information.",
                "ideal_answer": "Not stated.",
                "key_points": [],
                "answer_words": 6,
                "answer_chars": 30,
                "context_tokens": 900,
                "prompt_tokens": 950,
                "sources": [],
            }
        )

        html = render_report(data)

        self.assertIn("50.0%", html)
        self.assertIn("100.0% (1 / 1)", html)
        self.assertIn("0.0% (0 / 1)", html)

    def test_writes_report_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.html"

            written = write_report(path, report_data(1))

            self.assertEqual(written, path)
            self.assertIn("NavBot RAG Evaluation", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
