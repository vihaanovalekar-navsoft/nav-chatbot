"""Print chunk statistics for the build command."""

import statistics


def group_by_doc(rows):
    """Group chunk records by document, preserving first-seen order."""
    groups = {}
    for row in rows:
        groups.setdefault(row["doc"], []).append(row)
    return groups.items()


def print_stats(rows, chunk_size, special_tokens=2):
    """Print token, character, and per-document statistics for `rows`."""
    tokens = [row["token_count"] for row in rows]
    chars = [row["char_count"] for row in rows]
    over_limit = sum(
        1 for value in tokens if value + special_tokens > chunk_size
    )

    print(f"Chunks: {len(rows)} from {len({row['doc'] for row in rows})} documents")
    print(
        f"Chunk size: {chunk_size} input tokens "
        f"({chunk_size - special_tokens} text + {special_tokens} special)"
    )
    print(
        f"Text tokens: min {min(tokens)} | avg {statistics.mean(tokens):.1f} | "
        f"max {max(tokens)}"
    )
    print(
        f"Model input with {special_tokens} special tokens: "
        f"max {max(tokens) + special_tokens} of {chunk_size}"
    )
    print(
        f"Characters: min {min(chars)} | avg {statistics.mean(chars):.1f} | "
        f"max {max(chars)}"
    )
    print(
        f"Chunks over {chunk_size} input tokens: {over_limit} "
        f"({over_limit / len(rows):.1%})"
    )
    print()

    print(f"{'Document':44s} {'Chunks':>6s} {'Avg tok':>7s} {'Max tok':>7s}")
    for title, doc_rows in group_by_doc(rows):
        doc_tokens = [row["token_count"] for row in doc_rows]
        print(
            f"{title[:44]:44s} {len(doc_rows):6d} "
            f"{statistics.mean(doc_tokens):7.1f} {max(doc_tokens):7d}"
        )
