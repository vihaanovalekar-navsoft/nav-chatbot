# NavAI RAG Assistant

NavAI is a retrieval-augmented assistant for the NavAI website. It answers
questions from the NavAI documentation using hybrid retrieval, then generates a
short, citation-grounded answer.

## Features

- FastAPI web app and command-line interface.
- Hybrid retrieval: dense embeddings + in-memory BM25, fused with reciprocal
  rank fusion, then reranked with a cross-encoder.
- Persistent ChromaDB vector store, one collection per chunk size.
- Citation-gated answers and source display.
- Evaluation harness with per-case checkpointing and resume.
- One-page methodology PDF for chunking and retrieval.

## Project Layout

- `app.py` - FastAPI application and `/api/chat` endpoint.
- `server.py` - Uvicorn server startup and browser launch.
- `chat.py` - Command-line interface.
- `pipeline.py` - Retrieval, reranking, and answer generation.
- `retrieval.py` - Dense retrieval, BM25, RRF, and reranking.
- `prompts.py` - System and judge prompts.
- `answers.py` - Citation parsing and answer construction.
- `evaluate_rag.py` - Evaluation runner.
- `eval_judge.py` - LLM judge.
- `eval_report.py` - HTML report generation.
- `build_chunks.py` / `chunking.py` - Fixed-token chunking.
- `embed_chunks.py` - ChromaDB embedding.
- `tests/` - Unit tests.
- `reports/NavAI_RAG_Methodology.pdf` - Methodology summary.

## Setup

Create a `.env` file with:

```text
GOOGLE_API_KEY=... # required by the web app
GROQ_API_KEY=...   # required by the evaluation CLI default model
OPEN_API_KEY=...   # required for OpenRouter models
```

Install dependencies:

```bash
pip install fastapi uvicorn chromadb sentence-transformers groq openai \
  google-genai python-dotenv pydantic psutil transformers
```

Build chunks and embeddings:

```bash
python build_chunks.py
python embed_chunks.py
```

## Run

FastAPI web app:

```bash
python app.py
```

Command line:

```bash
python chat.py "What is NavAI?"
```

## Evaluation

Run with checkpointing:

```bash
python -u evaluate_rag.py --model recursive/flint-v1.0:free --checkpoint reports/eval_checkpoint.json
```

Resume after a failure:

```bash
python -u evaluate_rag.py --model recursive/flint-v1.0:free --checkpoint reports/eval_checkpoint.json --resume
```

## Tests

```bash
python -m unittest discover -s tests
```


## Deployment

Copy `.env.example` to `.env` and fill in the required keys. For Streamlit Cloud,
add the same keys to the app secrets.

Run locally:

```bash
python app.py
```

Run with Docker:

```bash
docker build -t navai-rag .
docker run -p 8000:8000 --env-file .env navai-rag
```

`Procfile` and `runtime.txt` are included for platform-as-a-service deployments.

## Methodology

See `reports/NavAI_RAG_Methodology.pdf`.

