"""Load the NavAI retrieval models and generate answers with Groq."""

import os

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer

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
        self.embedder = SentenceTransformer(
            EMBEDDING_MODEL, local_files_only=True
        )
        self.reranker = CrossEncoder(RERANKER_MODEL, local_files_only=True)
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
