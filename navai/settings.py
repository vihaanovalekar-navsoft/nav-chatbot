"""Configuration, shared constants, and data paths."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
DEFAULT_CHUNK_SIZE = 256
COLLECTION_NAME = "navai_chunks"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Gemini answers chat questions and supplies eval token counts.
GEMINI_MODEL = "gemini-3.5-flash-lite"
TOKEN_COUNT_MODEL = GEMINI_MODEL
GROQ_MODEL = "qwen/qwen3.8-27b"
OPENROUTER_MODEL = "thinkingmachines/inkling-small:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_RPM = 30


def load_config():
    """Return validated settings from config.json."""
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)

    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object.")

    for key in ("chunk_size", "retrieval_k", "crossencoder_k"):
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    temperature = config.get("temperature")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise ValueError("temperature must be a number.")
    if not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2.")
    if config["crossencoder_k"] > config["retrieval_k"]:
        raise ValueError("crossencoder_k must not exceed retrieval_k.")

    return config


def chunk_file(chunk_size):
    """Return the chunk JSONL path for `chunk_size` input tokens."""
    name = (
        "chunks.jsonl"
        if chunk_size == DEFAULT_CHUNK_SIZE
        else f"chunks_{chunk_size}.jsonl"
    )
    return ROOT / "data" / name


def database_path(chunk_size):
    """Return the Chroma directory for `chunk_size` input tokens."""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise ValueError("Chunk size must be an integer.")
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")
    name = (
        "chroma"
        if chunk_size == DEFAULT_CHUNK_SIZE
        else f"chroma_{chunk_size}"
    )
    return ROOT / "data" / name
