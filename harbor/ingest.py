"""Load uploaded files and pull out whatever text can be read locally."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
TEXT_SUFFIXES = {".md", ".txt"}


@dataclass(frozen=True)
class SourceDocument:
    name: str
    data: bytes
    media_type: str
    text: str | None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def _pdf_text(data: bytes) -> str | None:
    """Text and tables from a PDF; None for a scanned PDF with no text layer."""
    parts: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
            for table in page.extract_tables():
                for row in table:
                    parts.append(" | ".join(cell or "" for cell in row))
    text = "\n".join(p for p in parts if p).strip()
    return text or None


def load_document(name: str, data: bytes) -> SourceDocument:
    suffix = Path(name).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return SourceDocument(name, data, "text/plain", data.decode("utf-8", errors="replace"))
    if suffix == ".pdf":
        return SourceDocument(name, data, "application/pdf", _pdf_text(data))
    if suffix in IMAGE_TYPES:
        return SourceDocument(name, data, IMAGE_TYPES[suffix], None)
    raise ValueError(f"Unsupported file type: {name}")


def load_path(path: Path) -> SourceDocument:
    return load_document(path.name, path.read_bytes())
