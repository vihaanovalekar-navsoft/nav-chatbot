"""Run AI-judged RAG evaluation and write an HTML report."""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from google import genai

from .answers import build_result
from .eval_cases import load_cases
from .eval_judge import (
    JUDGE_INSTRUCTIONS,
    JUDGE_TEMPERATURE,
    JUDGE_VERSION,
    judge_answer,
)
from .eval_report import write_report
from .pipeline import NavBotChat
from .prompts import (
    PROMPT_VERSION,
    SYSTEM_INSTRUCTION,
    prompt_fingerprint,
)
from .retrieval import BM25_B, BM25_K1, RRF_K
from .settings import (
    EMBEDDING_MODEL,
    GROQ_MODEL,
    GROQ_RPM,
    RERANKER_MODEL,
    TOKEN_COUNT_MODEL,
    ROOT,
    database_path,
    load_config,
)


def count_tokens(client, text):
    """Return Gemini's token count for `text`."""
    response = client.models.count_tokens(
        model=TOKEN_COUNT_MODEL, contents=text
    )
    return response.total_tokens


class RequestPacer:
    """Enforce a minimum interval between Groq chat calls."""

    def __init__(self, requests_per_minute):
        self.interval = 60 / requests_per_minute
        self.last = None

    def wait(self):
        """Sleep until the next generate call is allowed."""
        if self.last is not None:
            delay = self.interval - (time.monotonic() - self.last)
            if delay > 0:
                time.sleep(delay)
        self.last = time.monotonic()


def run_case(bot, case, number, pacer, token_client):
    """Answer and judge one case, returning a report row."""
    prepared = bot.prepare(case["question"])
    context_tokens = count_tokens(token_client, prepared["context"])
    prompt_tokens = count_tokens(token_client, prepared["prompt"])
    pacer.wait()
    result = build_result(
        bot.generate(prepared["prompt"]), prepared["chunks"]
    )
    pacer.wait()
    score, reason = judge_answer(
        bot.groq,
        case,
        result["answer"],
        model=bot.model,
        reasoning_effort=bot.reasoning_effort,
    )
    return {
        "number": number,
        "question": case["question"],
        "answerable": case["answerable"],
        "score": score,
        "reason": reason,
        "answer": result["answer"],
        "ideal_answer": case["ideal_answer"],
        "key_points": list(case["key_points"]),
        "answer_words": len(result["answer"].split()),
        "answer_chars": len(result["answer"]),
        "context_tokens": context_tokens,
        "prompt_tokens": prompt_tokens,
        "sources": [
            f"{source['doc']} (chunk {source['chunk_index']})"
            for source in result["sources"]
        ],
    }


def default_reasoning_effort(model):
    """Return the effort value accepted by `model`."""
    if model.startswith("openai/gpt-oss"):
        return "low"
    if ":" in model:
        return None
    return "default"


def parse_args():
    """Return validated command-line options."""
    parser = argparse.ArgumentParser(
        description="Run the NavBot RAG evaluation with a Groq judge."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "navai_rag_eval.json",
        help="Evaluation case JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "HTML report path. Defaults to "
            "reports/rag_eval_cs<chunk>_ce<top>_<version>_<timestamp>.html."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N cases.",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=GROQ_RPM,
        help=(
            "Groq requests per minute to allow. Defaults to 30 "
            "for the free tier."
        ),
    )
    parser.add_argument(
        "--model",
        default=GROQ_MODEL,
        help="Groq model for answers and judging. Defaults to qwen.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help=(
            "Groq reasoning effort. Defaults to low for GPT-OSS "
            "models and default for other models."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="JSON checkpoint path. Rows are saved after each case.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load existing rows from --checkpoint and skip them.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    if args.rpm <= 0:
        parser.error("--rpm must be a positive integer")
    if args.resume and args.checkpoint is None:
        parser.error("--resume requires --checkpoint")
    return args


def main():
    """Run all evaluation cases and write the HTML report."""
    args = parse_args()
    cases = load_cases(args.cases)
    if args.limit is not None:
        cases = cases[:args.limit]

    config = load_config()
    reasoning_effort = (
        args.reasoning_effort or default_reasoning_effort(args.model)
    )
    bot = NavBotChat(
        model=args.model, reasoning_effort=reasoning_effort
    )
    token_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    system_tokens = count_tokens(token_client, SYSTEM_INSTRUCTION)
    pacer = RequestPacer(args.rpm)
    rows = []
    if args.resume:
        rows = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("Checkpoint must contain a JSON list.")
        rows = [row for row in rows if row["number"] <= len(cases)]
        rows.sort(key=lambda row: row["number"])

    print(
        f"Evaluating {len(cases)} cases with {args.model} "
        f"(judge {args.model}, dense+BM25 RRF, "
        f"reasoning {reasoning_effort})."
    )
    done = {row["number"] for row in rows}
    for number, case in enumerate(cases, start=1):
        if number in done:
            continue
        row = run_case(bot, case, number, pacer, token_client)
        rows.append(row)
        if args.checkpoint:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            args.checkpoint.write_text(
                json.dumps(rows, indent=2), encoding="utf-8"
            )
        print(
            f"[{number}/{len(cases)}] score={row['score']} "
            f"context={row['context_tokens']} tokens "
            f"answer={row['answer_words']} words | "
            f"{case['question'][:60]}",
            flush=True,
        )

    generated_at = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = args.output or (
        ROOT
        / "reports"
        / (
            f"rag_eval_cs{config['chunk_size']}_"
            f"ce{config['crossencoder_k']}_{PROMPT_VERSION}_{stamp}.html"
        )
    )
    report = {
        "generated_at": generated_at,
        "config": {
            "chunk_size": config["chunk_size"],
            "retrieval_mode": "dense+bm25_rrf",
            "retrieval_k": config["retrieval_k"],
            "crossencoder_k": config["crossencoder_k"],
            "rrf_k": RRF_K,
            "bm25_k1": BM25_K1,
            "bm25_b": BM25_B,
            "temperature": config["temperature"],
            "rpm": args.rpm,
            "prompt_version": PROMPT_VERSION,
            "prompt_fingerprint": prompt_fingerprint(),
            "answer_model": args.model,
            "judge_model": args.model,
            "reasoning_effort": reasoning_effort,
            "judge_version": JUDGE_VERSION,
            "judge_fingerprint": prompt_fingerprint(JUDGE_INSTRUCTIONS),
            "judge_temperature": JUDGE_TEMPERATURE,
            "tokenizer_model": TOKEN_COUNT_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": RERANKER_MODEL,
            "chroma_path": str(database_path(config["chunk_size"])),
            "cases_file": str(args.cases.resolve()),
        },
        "system_instruction_tokens": system_tokens,
        "prompt": SYSTEM_INSTRUCTION,
        "judge_prompt": JUDGE_INSTRUCTIONS,
        "results": rows,
    }
    write_report(output, report)

    correct = sum(row["score"] for row in rows)
    print(f"Accuracy: {correct}/{len(rows)} ({correct / len(rows):.1%})")
    print(f"Report: {output.resolve()}")


if __name__ == "__main__":
    main()
