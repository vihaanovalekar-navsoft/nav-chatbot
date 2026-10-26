"""Embed a chunk JSONL file into a persistent Chroma collection."""

import argparse
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .settings import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    chunk_file,
    database_path,
    load_config,
)


def load_chunks(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def parse_args():
    config = load_config()
    chunk_size = config["chunk_size"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=chunk_file(chunk_size))
    parser.add_argument("--db", type=Path, default=database_path(chunk_size))
    parser.add_argument("--collection", default=COLLECTION_NAME)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_chunks(args.chunks)
    model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    embeddings = model.encode(
        [row["text"] for row in rows],
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    client = chromadb.PersistentClient(
        path=str(args.db),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )
    collection.upsert(
        ids=[row["id"] for row in rows],
        embeddings=embeddings.tolist(),
        documents=[row["text"] for row in rows],
        metadatas=[
            {
                "doc": row["doc"],
                "chunk_index": row["chunk_index"],
                "token_start": row["token_start"],
                "token_count": row["token_count"],
                "char_count": row["char_count"],
            }
            for row in rows
        ],
    )

    print(f"Embedded {len(rows)} chunks into {args.db}")
    print(f"Collection: {args.collection}")
    print(f"Collection count: {collection.count()}")
    print(f"Embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
