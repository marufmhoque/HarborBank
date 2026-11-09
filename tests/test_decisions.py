from datetime import date

import pytest

from harbor.decisions import DecisionLog, check_decision, is_allowed

REVIEW = date(2026, 11, 8)


def test_hold_allows_only_override():
    assert not is_allowed("APPROVE", "HOLD FOR COMPLIANCE")
    assert not is_allowed("REJECT", "HOLD FOR COMPLIANCE")
    assert not is_allowed("PEND", "HOLD FOR COMPLIANCE")
    assert is_allowed("OVERRIDE & REFER", "HOLD FOR COMPLIANCE")


def test_aligned_decision_needs_only_a_name():
    assert check_decision("APPROVE", "MEETS CRITERIA", "A. Officer", "") is None
    assert check_decision("APPROVE", "MEETS CRITERIA", "  ", "") is not None


@pytest.mark.parametrize(
    "decision,tier",
    [("APPROVE", "DECLINE REVIEW"), ("REJECT", "MEETS CRITERIA"), ("PEND", "REFER (Policy)"), ("OVERRIDE & REFER", "MEETS CRITERIA")],
)
def test_departing_decision_needs_a_reason(decision, tier):
    assert "reason" in check_decision(decision, tier, "A. Officer", "")
    assert check_decision(decision, tier, "A. Officer", "Reviewed the file") is None


def test_log_records_and_returns_latest(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl")
    assert log.latest("APP-1") is None
    log.record("APP-1", "PEND", "PEND (Missing Info)", "A. Officer", "", REVIEW)
    entry = log.record("APP-1", "REJECT", "PEND (Missing Info)", "A. Officer", "Applicant withdrew", REVIEW)
    assert log.latest("APP-1") == entry
    assert entry.departs_from_recommendation is True
    assert log.latest("APP-2") is None


def test_pend_gets_a_14_day_due_date(tmp_path):
    entry = DecisionLog(tmp_path / "d.jsonl").record("APP-1", "PEND", "STALE", "A. Officer", "", REVIEW)
    assert entry.pend_due == date(2026, 11, 22)


def test_blocked_decision_is_not_written(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl")
    with pytest.raises(ValueError):
        log.record("APP-1", "APPROVE", "HOLD FOR COMPLIANCE", "A. Officer", "note", REVIEW)
    assert log.latest("APP-1") is None
