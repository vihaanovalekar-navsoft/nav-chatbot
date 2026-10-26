"""Command-line interface for the NavBot assistant."""

import sys

from .answers import format_source
from .pipeline import NavBotChat


def main():
    """Answer one question from the command line and print its sources."""
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input("Question: ").strip()
    if not question:
        raise SystemExit("A question is required.")

    result = NavBotChat().answer(question)
    print(result["answer"])
    if result["sources"]:
        print("\nSources:")
        for source in result["sources"]:
            print(format_source(source))


if __name__ == "__main__":
    main()
