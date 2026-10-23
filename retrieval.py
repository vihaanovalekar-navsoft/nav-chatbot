"""Hybrid retrieve and rerank NavAI chunks for a question."""

import math
import re
from collections import Counter, defaultdict

RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Return lowercase word tokens for BM25 scoring."""
    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """In-memory BM25 index over every chunk in a collection."""

    def __init__(self, ids, documents, metadatas):
        self.records = [
            {
                "id": chunk_id,
                "text": text,
                "metadata": metadata or {},
                "distance": None,
            }
            for chunk_id, text, metadata in zip(ids, documents, metadatas)
        ]
        self.postings = defaultdict(list)
        self.lengths = []
        for index, record in enumerate(self.records):
            tokens = tokenize(record["text"])
            self.lengths.append(len(tokens))
            for token, count in Counter(tokens).items():
                self.postings[token].append((index, count))
        self.average_length = (
            sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        )
        document_count = len(self.records)
        self.idf = {
            token: math.log(
                1
                + (document_count - len(entries) + 0.5)
                / (len(entries) + 0.5)
            )
            for token, entries in self.postings.items()
        }

    @classmethod
    def from_collection(cls, collection):
        """Build an index from every chunk in `collection`."""
        data = collection.get(include=["documents", "metadatas"])
        return cls(
            data.get("ids") or [],
            data.get("documents") or [],
            data.get("metadatas") or [],
        )

    def search(self, question, top_k):
        """Return the `top_k` highest BM25-scoring chunks for `question`."""
        if not self.records or self.average_length == 0:
            return []
        scores = defaultdict(float)
        for token, query_count in Counter(tokenize(question)).items():
            idf = self.idf.get(token)
            if idf is None:
                continue
            for index, frequency in self.postings[token]:
                length = self.lengths[index]
                denominator = frequency + BM25_K1 * (
                    1 - BM25_B + BM25_B * length / self.average_length
                )
                scores[index] += (
                    query_count * idf * frequency * (BM25_K1 + 1) / denominator
                )
        order = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [
            {**self.records[index], "bm25_score": float(scores[index])}
            for index in order
        ]


def build_search_query(question, history):
    """Return the current question plus the last user turn, newest first."""
    previous = next(
        (
            message["content"]
            for message in reversed(history)
            if message["role"] == "user"
        ),
        "",
    )
    return f"{question}\n{previous}" if previous else question


def retrieve(question, embedder, collection, top_k):
    """Return the `top_k` nearest chunks for `question`."""
    vector = embedder.encode([question], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=vector,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "id": chunk_id,
            "text": document,
            "metadata": metadata,
            "distance": distance,
        }
        for chunk_id, document, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def reciprocal_rank_fusion(rankings, top_k, rrf_k=RRF_K):
    """Fuse ranked chunk lists into one ranking with reciprocal ranks."""
    fused = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            entry = fused.get(chunk["id"])
            if entry is None:
                entry = {"chunk": dict(chunk), "score": 0.0}
                fused[chunk["id"]] = entry
            entry["score"] += 1.0 / (rrf_k + rank)
            if (
                entry["chunk"].get("distance") is None
                and chunk.get("distance") is not None
            ):
                entry["chunk"]["distance"] = chunk["distance"]
            if chunk.get("bm25_score") is not None:
                entry["chunk"]["bm25_score"] = chunk["bm25_score"]
    ordered = sorted(
        fused.values(), key=lambda entry: entry["score"], reverse=True
    )[:top_k]
    return [{**entry["chunk"], "rrf": entry["score"]} for entry in ordered]


def retrieve_hybrid(question, embedder, collection, bm25, top_k):
    """Return `top_k` chunks fused from dense and BM25 rankings."""
    dense = retrieve(question, embedder, collection, top_k)
    sparse = bm25.search(question, top_k)
    return reciprocal_rank_fusion([dense, sparse], top_k)


def rerank(question, chunks, reranker, top_k):
    """Return the top `top_k` chunks ordered by cross-encoder score."""
    if not chunks:
        return []
    scores = reranker.predict(
        [(question, chunk["text"]) for chunk in chunks]
    )
    order = sorted(
        range(len(chunks)), key=lambda index: scores[index], reverse=True
    )[:top_k]
    return [
        {**chunks[index], "score": float(scores[index])} for index in order
    ]
