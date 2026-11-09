import asyncio
from datetime import date

import pytest

import harbor.pipeline as pipeline
from harbor.ingest import load_document
from harbor.pend_timer import PendStore
from tests.test_agent2 import APPS, CUSTOMERS, REVIEW, STORE, _customers


def _workflow_text():
    """APP-1004 with an amount inside the unsecured cap, so nothing fails on its face."""
    return (APPS / "app_1004.md").read_text(encoding="utf-8").replace("$150,000", "$90,000")


def run(text, customers=CUSTOMERS, **kwargs):
    doc = load_document("app_1004.md", text.encode())
    return asyncio.run(pipeline.run_pipeline([doc], customers, REVIEW, STORE, PendStore(), **kwargs))


def test_documented_workflow_ends_in_pend_with_bankruptcy_surfaced():
    bureau = _customers(bureau_metadata_stub={"bankruptcy_flag": True, "bankruptcy_date": "2024-03-15"})
    result = run(_workflow_text(), customers=bureau)

    assert result.payload.claimed.claimed_income.value == 95000
    assert result.payload.verified.verified_noi is None

    report = result.report
    assert report.tier == "PEND (Missing Info)"
    assert report.pend.expires_on == date(2026, 11, 22)
    assert report.risk_flags[0].fact == "Bankruptcy on record"
    assert report.risk_flags[0].date == date(2024, 3, 15)
    assert "has no clause" in report.risk_flags[0].policy_basis

    letter = result.letter
    assert letter.validation.passed, letter.validation.errors
    assert "2025 business tax return" in letter.rendered_text
    assert "size of the loan payment" in letter.rendered_text
    assert letter.rendered_text.count("size of the loan payment") == 1
    assert "November 22, 2026" in letter.rendered_text
    assert "bankruptcy" not in letter.rendered_text.lower()


def test_agents_run_in_order_and_report_each_step():
    steps = []
    run(_workflow_text(), on_step=steps.append)
    assert [s.split(":")[0] for s in steps] == ["Agent 1", "Agent 2", "Agent 3"]


def test_failure_stops_later_agents(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("agent 2 failed")

    def must_not_run(*args, **kwargs):
        raise AssertionError("Agent 3 ran after Agent 2 failed")

    monkeypatch.setattr(pipeline, "run_agent2", boom)
    monkeypatch.setattr(pipeline, "run_agent3", must_not_run)
    with pytest.raises(RuntimeError, match="agent 2 failed"):
        run(_workflow_text())
