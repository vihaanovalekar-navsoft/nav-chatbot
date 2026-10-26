"""Read DOCX files and build chunk records."""

import io
import re
import zipfile
from pathlib import Path

from docx import Document

from .chunking import chunk_text


def iter_documents(archive):
    """Yield `(title, text)` for every DOCX in the archive."""
    with zipfile.ZipFile(archive) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".docx") or Path(name).name.startswith("~$"):
                continue
            document = Document(io.BytesIO(zf.read(name)))
            lines = [
                line.strip()
                for paragraph in document.paragraphs
                for line in paragraph.text.splitlines()
                if line.strip()
            ]
            yield Path(name).stem, "\n".join(lines)


def slug(text):
    """Return a filesystem-safe identifier for `text`."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def build_rows(archive, tokenizer, chunk_size, overlap, special_tokens=2):
    """Yield chunk records for each document in `archive`."""
    chunk_tokens = chunk_size - special_tokens
    for title, text in iter_documents(archive):
        slugged = slug(title)
        for index, (token_start, token_count, chunk) in enumerate(
            chunk_text(text, tokenizer, chunk_tokens, overlap)
        ):
            yield {
                "id": f"{slugged}-{index:04d}",
                "doc": title,
                "chunk_index": index,
                "token_start": token_start,
                "token_count": token_count,
                "char_count": len(chunk),
                "text": chunk,
            }
