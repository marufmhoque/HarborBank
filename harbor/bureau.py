"""Reads existing bureau data from customers.json.

This module must never contact a credit bureau: there is deliberately no client,
network import or pull function here, and tests enforce that.
"""

from __future__ import annotations

from datetime import date

from harbor.schemas import BureauMetadataStub, BureauSnapshot


def read_bureau_snapshot(record: dict | None) -> BureauSnapshot:
    if record is None:
        return BureauSnapshot()
    pull_date = record.get("bureau_pull_date")
    return BureauSnapshot(
        credit_score=record.get("credit_score"),
        pull_date=date.fromisoformat(pull_date) if pull_date else None,
        bureau_metadata_stub=BureauMetadataStub(**record.get("bureau_metadata_stub", {})),
    )
