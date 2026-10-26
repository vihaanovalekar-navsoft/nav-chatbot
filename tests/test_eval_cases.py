import json
import tempfile
import unittest
from pathlib import Path

from navai.eval_cases import load_cases
from navai.settings import ROOT


class EvalCasesTest(unittest.TestCase):
    def test_loads_project_cases(self):
        cases = load_cases(ROOT / "navai_rag_eval.json")

        self.assertGreater(len(cases), 0)
        self.assertTrue(any(case["answerable"] for case in cases))
        self.assertTrue(any(not case["answerable"] for case in cases))

    def test_rejects_missing_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(
                json.dumps([{"answerable": True}]),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_cases(path)

    def test_rejects_invalid_key_points(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "answerable": True,
                            "question": "Q",
                            "ideal_answer": "A",
                            "key_points": "not a list",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_cases(path)

    def test_rejects_empty_array(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text("[]", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_cases(path)


if __name__ == "__main__":
    unittest.main()
