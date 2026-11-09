"""Runs the three agents in order, off the UI thread, reporting progress."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date
from typing import Callable, Sequence

from harbor.agent1 import DocumentParser, run_agent1
from harbor.agent2 import PolicyAuditReport, run_agent2
from harbor.agent3 import LetterDraft, run_agent3
from harbor.ingest import SourceDocument
from harbor.pend_timer import PendStore
from harbor.policy_index import ClauseStore
from harbor.schemas import ApplicationPayload


@dataclass(frozen=True)
class PipelineResult:
    payload: ApplicationPayload
    report: PolicyAuditReport
    letter: LetterDraft


async def run_pipeline(
    documents: Sequence[SourceDocument],
    customers: Sequence[dict],
    review_date: date,
    clause_store: ClauseStore,
    pend_store: PendStore,
    llm: DocumentParser | None = None,
    related_documents: Sequence[SourceDocument] = (),
    on_step: Callable[[str], None] = lambda _: None,
) -> PipelineResult:
    """Each agent starts only after the previous one succeeds. A failure stops the run."""
    on_step("Agent 1: parsing documents and separating claimed from verified data")
    payload = await asyncio.to_thread(run_agent1, documents, customers, review_date, llm, related_documents)

    on_step("Agent 2: testing the file against policy and calculating terms")
    report = await asyncio.to_thread(run_agent2, payload, clause_store, pend_store)

    on_step("Agent 3: drafting the applicant letter")
    letter = await asyncio.to_thread(run_agent3, payload, report)

    return PipelineResult(payload, report, letter)
