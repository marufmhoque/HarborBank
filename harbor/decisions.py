"""Credit officer decisions. The agents recommend; only this log records a decision.

Policy §4: on a compliance hold, underwriting stops, so the only action allowed is
an explicit override and referral with a written reason.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from harbor.pend_timer import PEND_LIMIT_DAYS

Decision = Literal["APPROVE", "REJECT", "PEND", "OVERRIDE & REFER"]
HOLD_TIER = "HOLD FOR COMPLIANCE"

_ALIGNED: dict[str, set[str]] = {
    "APPROVE": {"MEETS CRITERIA"},
    "REJECT": {"DECLINE REVIEW"},
    "PEND": {"PEND (Missing Info)", "STALE"},
}


class DecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    application_id: str
    decision: Decision
    agent_recommendation: str
    officer: str
    note: str
    departs_from_recommendation: bool
    pend_due: date | None
    decided_at: datetime


def is_allowed(decision: Decision, tier: str) -> bool:
    return tier != HOLD_TIER or decision == "OVERRIDE & REFER"


def departs(decision: Decision, tier: str) -> bool:
    return tier not in _ALIGNED.get(decision, set())


def check_decision(decision: Decision, tier: str, officer: str, note: str) -> str | None:
    """Return why this decision can't be recorded, or None if it can."""
    if not is_allowed(decision, tier):
        return "This file is on compliance hold. Only Override & Refer is available."
    if not officer.strip():
        return "Enter the credit officer's name."
    if departs(decision, tier) and not note.strip():
        return "A written reason is required when the decision differs from the recommendation."
    return None


class DecisionLog:
    """Append-only JSON Lines file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def record(self, application_id: str, decision: Decision, tier: str, officer: str, note: str, review_date: date) -> DecisionRecord:
        problem = check_decision(decision, tier, officer, note)
        if problem:
            raise ValueError(problem)
        entry = DecisionRecord(
            application_id=application_id, decision=decision, agent_recommendation=tier,
            officer=officer.strip(), note=note.strip(), departs_from_recommendation=departs(decision, tier),
            pend_due=review_date + timedelta(days=PEND_LIMIT_DAYS) if decision == "PEND" else None,
            decided_at=datetime.now(timezone.utc),
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.model_dump(mode="json")) + "\n")
        return entry

    def latest(self, application_id: str) -> DecisionRecord | None:
        if not self._path.exists():
            return None
        found = None
        for line in self._path.read_text(encoding="utf-8").splitlines():
            entry = DecisionRecord.model_validate_json(line)
            if entry.application_id == application_id:
                found = entry
        return found
