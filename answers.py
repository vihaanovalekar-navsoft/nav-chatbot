"""Citation checks and final answer construction."""

import re

CITATION = re.compile(
    r"(?:\[\s*\d+(?:\s*,\s*\d+)*\s*\]|<\s*\d+\s*>)"
)


def sources_for(answer, chunks):
    """Return chunk sources when `answer` cites a chunk, otherwise []."""
    if not CITATION.search(answer):
        return []
    return [
        {
            "label": number,
            "doc": chunk["metadata"]["doc"],
            "chunk_index": chunk["metadata"]["chunk_index"],
            "rerank": round(chunk["score"], 4),
            "distance": (
                None
                if chunk["distance"] is None
                else round(chunk["distance"], 4)
            ),
        }
        for number, chunk in enumerate(chunks, start=1)
    ]


def format_source(source):
    """Return one citation source as a readable status line."""
    distance = source["distance"]
    distance_text = "n/a" if distance is None else f"{distance:.4f}"
    return (
        f"[{source['label']}] {source['doc']} | "
        f"chunk {source['chunk_index']} | "
        f"rerank {source['rerank']:.4f} | "
        f"distance {distance_text}"
    )


def build_result(text, chunks):
    """Return the final answer and its citation-gated sources."""
    answer = (text or "").strip()
    if not answer:
        answer = "I could not generate an answer. Please rephrase the question."
    return {"answer": answer, "sources": sources_for(answer, chunks)}
