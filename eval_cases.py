"""Load and validate RAG evaluation cases."""

import json
from pathlib import Path

REQUIRED_FIELDS = ("answerable", "question", "ideal_answer", "key_points")


def load_cases(path):
    """Return validated evaluation cases from the JSON file at `path`."""
    with Path(path).open(encoding="utf-8") as handle:
        cases = json.load(handle)

    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation file must be a non-empty JSON array.")

    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Case {index} must be an object.")
        missing = [field for field in REQUIRED_FIELDS if field not in case]
        if missing:
            raise ValueError(
                f"Case {index} is missing: {', '.join(missing)}."
            )
        if not isinstance(case["answerable"], bool):
            raise ValueError(f"Case {index} answerable must be a boolean.")
        if not isinstance(case["question"], str) or not case["question"].strip():
            raise ValueError(f"Case {index} question must be non-empty text.")
        if not isinstance(case["ideal_answer"], str):
            raise ValueError(f"Case {index} ideal_answer must be text.")
        if not isinstance(case["key_points"], list) or not all(
            isinstance(point, str) for point in case["key_points"]
        ):
            raise ValueError(
                f"Case {index} key_points must be a list of strings."
            )

    return cases
