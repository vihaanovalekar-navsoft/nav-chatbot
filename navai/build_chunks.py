"""Build fixed-token windows from Docx.zip into data/chunks*.jsonl."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from .chunk_report import print_stats
from .documents import build_rows
from .settings import EMBEDDING_MODEL, ROOT, chunk_file, load_config

ARCHIVE = ROOT / "Docx.zip"
SPECIAL_TOKENS = 2
OVERLAP_TOKENS = 32


def parse_args():
    """Return validated command-line options."""
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=config["chunk_size"],
        help=(
            "Maximum model input tokens per chunk, including special tokens. "
            "Defaults to chunk_size in config.json."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "JSONL output path. Defaults to data/chunks.jsonl for 256 and "
            "data/chunks_<size>.jsonl otherwise."
        ),
    )
    args = parser.parse_args()
    if args.chunk_size - SPECIAL_TOKENS <= OVERLAP_TOKENS:
        parser.error("chunk size must be greater than 34")
    return args


def main():
    """Build chunks from the archive and write them to JSONL."""
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL, local_files_only=True
    )
    # The tokenizer warns above its 512-token default; windows are capped below.
    tokenizer.model_max_length = 1_000_000
    rows = list(
        build_rows(
            ARCHIVE,
            tokenizer,
            args.chunk_size,
            overlap=OVERLAP_TOKENS,
            special_tokens=SPECIAL_TOKENS,
        )
    )

    output = args.output or chunk_file(args.chunk_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {output}")
    print_stats(rows, args.chunk_size, SPECIAL_TOKENS)


if __name__ == "__main__":
    main()
