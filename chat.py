"""Command-line interface for the NavAI assistant."""

import sys

from pipeline import NavAIChat


def main():
    """Answer one question from the command line and print its sources."""
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input("Question: ").strip()
    if not question:
        raise SystemExit("A question is required.")

    result = NavAIChat().answer(question)
    print(result["answer"])
    if result["sources"]:
        print("\nSources:")
        for source in result["sources"]:
            distance = source["distance"]
            distance_text = "n/a" if distance is None else f"{distance:.4f}"
            print(
                f"[{source['label']}] {source['doc']} | "
                f"chunk {source['chunk_index']} | "
                f"rerank {source['rerank']:.4f} | "
                f"distance {distance_text}"
            )


if __name__ == "__main__":
    main()
