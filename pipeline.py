"""Load the NavAI retrieval models and generate answers with Groq."""

import os
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings
from dotenv import load_dotenv
from groq import Groq
from huggingface_hub import hf_hub_download
from openai import OpenAI
from tokenizers import Tokenizer
import onnxruntime as ort

from answers import build_result
from prompts import SYSTEM_INSTRUCTION, build_prompt, format_context
from retrieval import BM25Index, build_search_query, rerank, retrieve_hybrid
from settings import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GROQ_MODEL,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    RERANKER_MODEL,
    ROOT,
    database_path,
    load_config,
)

ONNX_EMBEDDING_FILE = "onnx/model_quint8_avx2.onnx"
ONNX_RERANKER_FILE = "onnx/model_quint8_avx2.onnx"


class SentenceTransformer:
    """Lightweight ONNX Runtime embedding model."""

    def __init__(
        self,
        model_name_or_path=EMBEDDING_MODEL,
        backend=None,
        model_kwargs=None,
        local_files_only=False,
        **kwargs,
    ):
        model_kwargs = model_kwargs or {}
        file_name = model_kwargs.get("file_name", ONNX_EMBEDDING_FILE)
        self.model_name = model_name_or_path
        self.onnx_path = hf_hub_download(
            model_name_or_path, file_name, local_files_only=local_files_only
        )
        self.tokenizer = Tokenizer.from_file(
            hf_hub_download(
                model_name_or_path, "tokenizer.json", local_files_only=local_files_only
            )
        )
        self.session = ort.InferenceSession(
            self.onnx_path, providers=["CPUExecutionProvider"]
        )
        self.pad_token_id = self.tokenizer.get_vocab().get("[PAD]", 0)

    def _tokenize(self, texts):
        encodings = self.tokenizer.encode_batch(texts)
        max_length = max(len(encoding.ids) for encoding in encodings)
        input_ids = []
        attention_masks = []
        token_type_ids = []
        for encoding in encodings:
            padding = max_length - len(encoding.ids)
            input_ids.append(encoding.ids + [self.pad_token_id] * padding)
            attention_masks.append(
                encoding.attention_mask + [0] * padding
            )
            token_type_ids.append(encoding.type_ids + [0] * padding)
        return {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "attention_mask": np.array(attention_masks, dtype=np.int64),
            "token_type_ids": np.array(token_type_ids, dtype=np.int64),
        }

    def encode(
        self,
        sentences,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
        **kwargs,
    ):
        if isinstance(sentences, str):
            sentences = [sentences]
        if not sentences:
            return np.zeros((0, 384), dtype=np.float32)
        embeddings = []
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            inputs = self._tokenize(batch)
            hidden_state = self.session.run(None, inputs)[0]
            mask = inputs["attention_mask"][..., None].astype(np.float32)
            pooled = (hidden_state * mask).sum(axis=1) / mask.sum(axis=1)
            if normalize_embeddings:
                norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                pooled = pooled / np.clip(norms, 1e-12, None)
            embeddings.append(pooled.astype(np.float32))
        return np.vstack(embeddings)


class CrossEncoder:
    """Lightweight ONNX Runtime cross-encoder."""

    def __init__(
        self,
        model_name_or_path=RERANKER_MODEL,
        backend=None,
        model_kwargs=None,
        local_files_only=False,
        **kwargs,
    ):
        model_kwargs = model_kwargs or {}
        file_name = model_kwargs.get("file_name", ONNX_RERANKER_FILE)
        self.model_name = model_name_or_path
        self.onnx_path = hf_hub_download(
            model_name_or_path, file_name, local_files_only=local_files_only
        )
        self.tokenizer = Tokenizer.from_file(
            hf_hub_download(
                model_name_or_path, "tokenizer.json", local_files_only=local_files_only
            )
        )
        self.session = ort.InferenceSession(
            self.onnx_path, providers=["CPUExecutionProvider"]
        )

    def predict(self, pairs):
        scores = []
        for query, passage in pairs:
            encoding = self.tokenizer.encode(query, passage)
            inputs = {
                "input_ids": np.array([encoding.ids], dtype=np.int64),
                "attention_mask": np.array(
                    [encoding.attention_mask], dtype=np.int64
                ),
                "token_type_ids": np.array(
                    [encoding.type_ids], dtype=np.int64
                ),
            }
            output = self.session.run(None, inputs)[0]
            scores.append(float(np.squeeze(output)))
        return np.array(scores)


class NavAIChat:
    """Loaded retrieval and generation pipeline."""

    def __init__(self, model=None, reasoning_effort="default"):
        load_dotenv(ROOT / ".env")
        config = load_config()
        self.model = model or OPENROUTER_MODEL
        self.reasoning_effort = reasoning_effort
        self.uses_openai = ":" in self.model or self.model == OPENROUTER_MODEL
        self.retrieval_k = config["retrieval_k"]
        self.crossencoder_k = config["crossencoder_k"]
        self.temperature = config["temperature"]
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = CrossEncoder(RERANKER_MODEL)
        chroma = chromadb.PersistentClient(
            path=str(database_path(config["chunk_size"])),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = chroma.get_collection(COLLECTION_NAME)
        self.bm25 = BM25Index.from_collection(self.collection)
        openrouter_key = (
            os.getenv("OPEN_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.groq = None
        if self.uses_openai:
            if not openrouter_key:
                raise RuntimeError("OPEN_API_KEY is required for OpenRouter models.")
            self.groq = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=openrouter_key,
                default_headers={
                    "HTTP-Referer": "https://opencode.ai",
                    "X-Title": "OpenCode",
                },
            )
        elif os.getenv("GROQ_API_KEY"):
            self.groq = Groq(api_key=os.environ["GROQ_API_KEY"])

    def prepare(self, question, history=None):
        """Return the assembled prompt, context, and reranked chunks."""
        history = history or []
        search_query = build_search_query(question, history)
        candidates = retrieve_hybrid(
            search_query,
            self.embedder,
            self.collection,
            self.bm25,
            self.retrieval_k,
        )
        chunks = rerank(
            search_query, candidates, self.reranker, self.crossencoder_k
        )
        context = format_context(chunks)
        return {
            "prompt": build_prompt(question, context, history),
            "context": context,
            "chunks": chunks,
        }

    def generate(self, prompt):
        """Return Groq's answer text for `prompt`."""
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "top_p": 0.95,
        }
        if self.reasoning_effort != "default":
            request["reasoning_effort"] = self.reasoning_effort
        elif not self.uses_openai:
            request["reasoning_effort"] = "default"
        if self.uses_openai:
            request["max_tokens"] = 2048
        else:
            request["max_completion_tokens"] = 2048
        response = self.groq.chat.completions.create(**request)
        return response.choices[0].message.content

    def answer(self, question, history=None):
        """Return an answer and its sources for `question`."""
        prepared = self.prepare(question, history)
        return build_result(self.generate(prepared["prompt"]), prepared["chunks"])