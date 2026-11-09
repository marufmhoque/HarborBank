"""Version-tagged policy clauses: exact lookup for citations, FAISS for free-text search.

Tiers and citations use exact lookup only. Vector search is an assistive tool for
reviewers, and it searches one policy version at a time so versions can't mix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol, Sequence

import faiss
import numpy as np

from harbor.policy_registry import POLICY_VERSIONS, PolicyVersion

EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class PolicyClause:
    clause_id: str
    policy_version: str
    effective_date: date
    section_title: str
    text: str


def parse_clauses(policy: PolicyVersion, text: str) -> list[PolicyClause]:
    """Number each substantive line within its section, in document order."""
    clauses: list[PolicyClause] = []
    section: str | None = None
    title = ""
    count = 0
    for raw in text.splitlines():
        line = raw.strip()
        if heading := re.match(r"##\s+(\d+)\.\s+(.*)", line):
            section, title, count = heading[1], heading[2], 0
            continue
        if not line or section is None:
            continue
        if line.startswith("|") and not re.match(r"\|\s*\d", line):
            continue  # table header or separator
        count += 1
        body = re.sub(r"^(-|\d+\.)\s+", "", line)
        clauses.append(PolicyClause(f"{section}.{count}", policy.version, policy.effective_date, title, body))
    return clauses


class ClauseStore:
    def __init__(self, data_dir: Path) -> None:
        self.clauses: list[PolicyClause] = []
        for policy in POLICY_VERSIONS:
            self.clauses += parse_clauses(policy, (data_dir / policy.source_file).read_text(encoding="utf-8"))
        self._by_key = {(c.policy_version, c.clause_id): c for c in self.clauses}

    def get(self, version: str, clause_id: str) -> PolicyClause | None:
        return self._by_key.get((version, clause_id))

    def for_version(self, version: str) -> list[PolicyClause]:
        return [c for c in self.clauses if c.policy_version == version]

    def mentions(self, version: str, term: str) -> list[PolicyClause]:
        return [c for c in self.for_version(version) if term.lower() in c.text.lower()]


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class OpenAIEmbedder:
    def __init__(self, client) -> None:
        self._client = client

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=EMBEDDING_MODEL, input=list(texts))
        return np.array([item.embedding for item in response.data], dtype="float32")


class PolicyVectorIndex:
    """One FAISS index per version: filtering happens before any similarity search."""

    def __init__(self, store: ClauseStore, embedder: Embedder) -> None:
        self._embedder = embedder
        self._indexes: dict[str, tuple[faiss.Index, list[PolicyClause]]] = {}
        for policy in POLICY_VERSIONS:
            clauses = store.for_version(policy.version)
            vectors = embedder.embed([c.text for c in clauses])
            faiss.normalize_L2(vectors)
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            self._indexes[policy.version] = (index, clauses)

    def search(self, query: str, version: str, k: int = 3) -> list[tuple[PolicyClause, float]]:
        index, clauses = self._indexes[version]
        vector = self._embedder.embed([query])
        faiss.normalize_L2(vector)
        scores, ids = index.search(vector, min(k, len(clauses)))
        return [(clauses[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]
