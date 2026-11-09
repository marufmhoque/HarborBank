"""14-day PEND timer. Persists to a JSON file so it survives app restarts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

PEND_LIMIT_DAYS = 14


@dataclass(frozen=True)
class PendStatus:
    pended_on: date
    expires_on: date
    days_elapsed: int
    expired: bool


class PendStore:
    """Maps application ID to the date pended and a fingerprint of its documents."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._data: dict[str, dict] = {}
        if path and path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    def check(self, application_id: str, fingerprint: str, review_date: date) -> PendStatus:
        """Start the timer on first PEND, and restart it when the documents change."""
        entry = self._data.get(application_id)
        if entry is None or entry["fingerprint"] != fingerprint:
            entry = {"pended_on": review_date.isoformat(), "fingerprint": fingerprint}
            self._data[application_id] = entry
            self._save()
        pended_on = date.fromisoformat(entry["pended_on"])
        elapsed = (review_date - pended_on).days
        return PendStatus(pended_on, pended_on + timedelta(days=PEND_LIMIT_DAYS), elapsed, elapsed > PEND_LIMIT_DAYS)

    def clear(self, application_id: str) -> None:
        if self._data.pop(application_id, None) is not None:
            self._save()

    def _save(self) -> None:
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
