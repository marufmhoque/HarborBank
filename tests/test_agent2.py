import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from harbor.agent1 import run_agent1
from harbor.agent2 import SENSITIVITY_TAG, run_agent2, run_sensitivity
from harbor.credit_rules import CREDIT_RULES
from harbor.financials import dscr, loan_terms, monthly_payment
from harbor.ingest import load_document, load_path
from harbor.pend_timer import PendStore
from harbor.policy_index import ClauseStore, PolicyVectorIndex

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
APPS = DATA / "applications"
CUSTOMERS = json.loads((DATA / "customers.json").read_text(encoding="utf-8"))
REVIEW = date(2026, 11, 8)
STORE = ClauseStore(DATA)


def payload(app_id, related=True, text_edit=None, customers=CUSTOMERS, review=REVIEW):
    target = APPS / f"app_{app_id}.md"
    others = [load_path(p) for p in sorted(APPS.glob("*.md")) if p != target] if related else []
    doc = load_path(target)
    if text_edit:
        doc = load_document(doc.name, text_edit(doc.text).encode())
    return run_agent1([doc], customers, review, related_documents=others)


def criterion(report, name):
    return next(c for c in report.criteria if c.name == name)


# --- Calculator ---


@pytest.mark.parametrize(
    "principal,rate,months,expected",
    [("80000", "8.50", 60, "1641.32"), ("100000", "6", 360, "599.55"), ("1200", "0", 12, "100.00")],
)
def test_monthly_payment(principal, rate, months, expected):
    assert monthly_payment(Decimal(principal), Decimal(rate), months) == Decimal(expected)


def test_loan_terms_interest_and_fee():
    terms = loan_terms(Decimal(80000), Decimal("8.50"), 60, Decimal("1.0"))
    assert terms.origination_fee == Decimal("800.00")
    assert terms.total_interest == Decimal("1641.32") * 60 - Decimal(80000)


def test_dscr_is_not_rounded_up_into_passing():
    assert dscr(Decimal("1249.9"), Decimal(1000)) < CREDIT_RULES["2026.2"].dscr_meets


# --- Rules match the policy documents ---


@pytest.mark.parametrize("version", ["2025.1", "2026.2"])
def test_rule_values_appear_in_cited_clauses(version):
    rules = CREDIT_RULES[version]

    def text(key):
        return STORE.get(version, rules.clauses[key]).text

    assert f"{rules.min_months} months" in text("time_in_business")
    assert f"${rules.min_amount:,.0f}" in text("amount") and f"${rules.max_amount:,.0f}" in text("amount")
    assert f"${rules.unsecured_cap:,.0f}" in text("unsecured_cap")
    assert f"{rules.max_term} months" in text("term")
    assert str(rules.score_meets) in text("score") and str(rules.score_refer) in text("score")
    assert str(rules.dscr_meets) in text("dscr_meets") and str(rules.dscr_refer) in text("dscr_refer")
    assert "1.0%" in text("fee")
    for band in rules.rate_bands:
        assert f"{band.rate_pct}%" in STORE.get(version, band.clause).text


# --- The four sample applications ---


def test_app_1001_held_for_cross_application_income_conflict():
    r = run_agent2(payload("1001"), STORE)
    assert r.tier == "HOLD FOR COMPLIANCE"
    assert r.financials is None
    assert r.routing_citation == "SB-100 v2026.2 §4.3"


def test_app_1001_alone_meets_criteria_with_exact_math():
    r = run_agent2(payload("1001", related=False), STORE)
    assert r.tier == "MEETS CRITERIA"
    f = r.financials
    assert (f.annual_rate_pct, f.monthly_payment, f.origination_fee) == (Decimal("8.50"), Decimal("1641.32"), Decimal("800.00"))
    assert f.dscr == Decimal(62000) / (Decimal(18000) + Decimal("1641.32") * 12)
    assert criterion(r, "DSCR").status == "PASS"
    assert criterion(r, "DSCR").citation == "SB-100 v2026.2 §2.3"
    assert f.rate_citation == "SB-100 v2026.2 §5.1"


def test_app_1002_hard_fail_goes_to_decline_review_not_pend():
    r = run_agent2(payload("1002"), STORE)
    assert r.tier == "DECLINE REVIEW"
    assert criterion(r, "Time in business").status == "FAIL"
    assert criterion(r, "Time in business").citation == "SB-100 v2026.2 §1.1"
    assert r.pend is None
    assert any("Business license" in n for n in r.also_noted)


def test_app_1003_held_and_injection_does_not_alter_outcome():
    r = run_agent2(payload("1003"), STORE)
    assert r.tier == "HOLD FOR COMPLIANCE"
    assert r.financials is None
    assert criterion(r, "DSCR").status == "NOT EVALUATED"
    assert any("EIN" in reason for reason in r.reasons)


def test_app_1004_held_with_hard_failure_still_noted():
    r = run_agent2(payload("1004"), STORE)
    assert r.tier == "HOLD FOR COMPLIANCE"
    assert criterion(r, "Unsecured cap").status == "FAIL"
    assert any("Unsecured cap" in n for n in r.also_noted)


# --- Routing branches the samples don't reach ---


def _drop_license(text):
    return text.replace("business license; ", "")


def test_pend_reports_impact_and_goes_stale_after_14_days(tmp_path):
    store = PendStore(tmp_path / "t.json")
    r = run_agent2(payload("1001", related=False, text_edit=_drop_license), STORE, store)
    assert r.tier == "PEND (Missing Info)"
    assert r.pend.expires_on == REVIEW + timedelta(days=14)
    assert r.missing_impacts[0].impacts == ("Required documents",)

    day14 = payload("1001", related=False, text_edit=_drop_license, review=date(2026, 11, 22))
    assert run_agent2(day14, STORE, PendStore(tmp_path / "t.json")).tier == "PEND (Missing Info)"
    day15 = payload("1001", related=False, text_edit=_drop_license, review=date(2026, 11, 23))
    assert run_agent2(day15, STORE, PendStore(tmp_path / "t.json")).tier == "STALE"


def test_document_update_restarts_pend_timer(tmp_path):
    store = PendStore(tmp_path / "t.json")
    run_agent2(payload("1001", related=False, text_edit=_drop_license), STORE, store)
    changed = lambda t: _drop_license(t).replace("second oven", "new oven")
    later = payload("1001", related=False, text_edit=changed, review=date(2026, 11, 30))
    r = run_agent2(later, STORE, store)
    assert r.tier == "PEND (Missing Info)"
    assert r.pend.pended_on == date(2026, 11, 30)


def test_missing_income_reports_income_needed():
    def no_return(text):
        lines = [l for l in text.splitlines() if not l.startswith("From 2025")]
        return "\n".join(lines)

    r = run_agent2(payload("1001", related=False, text_edit=no_return), STORE)
    assert r.tier == "PEND (Missing Info)"
    assert r.required_noi_to_meet_dscr == Decimal("1.25") * (Decimal(18000) + Decimal("1641.32") * 12)


def _customers(**changes):
    return [{**c, **changes} if c["customer_id"] == "C2001" else c for c in CUSTOMERS]


def test_score_in_referral_band_is_refer_policy():
    r = run_agent2(payload("1001", related=False, customers=_customers(credit_score=660)), STORE)
    assert r.tier == "REFER (Policy)"
    assert r.financials.annual_rate_pct == Decimal("11.00")


def test_low_score_is_decline_review():
    assert run_agent2(payload("1001", related=False, customers=_customers(credit_score=600)), STORE).tier == "DECLINE REVIEW"


def test_2025_referral_score_has_no_price_but_still_refers():
    earlier = lambda t: t.replace("Signed 2026-10-27", "Signed 2026-06-01")
    r = run_agent2(payload("1001", related=False, text_edit=earlier, customers=_customers(credit_score=640)), STORE)
    assert r.policy_version == "2025.1"
    assert r.tier == "REFER (Policy)"
    assert r.financials is None


def test_bankruptcy_surfaces_fact_without_inventing_a_rule():
    stub = {"bankruptcy_flag": True, "bankruptcy_date": "2023-05-01"}
    r = run_agent2(payload("1001", related=False, customers=_customers(bureau_metadata_stub=stub)), STORE)
    assert r.tier == "REFER (Risk/History)"
    flag = r.risk_flags[0]
    assert flag.date == date(2023, 5, 1)
    assert flag.source == "customers.json bureau_metadata_stub"
    assert "has no clause" in flag.policy_basis


def test_combined_exposure_is_included():
    p = payload("1001")
    from harbor.agent2 import _Ctx, _debt_service

    lines, unpriced = _debt_service(_Ctx(p, CREDIT_RULES["2026.2"], None), Decimal("8.50"))
    assert [l.label for l in lines][-1] == "Open application APP-1004"
    assert unpriced == []


# --- Sensitivity ---


def test_sensitivity_is_tagged_and_never_changes_tier():
    p = payload("1004")
    before = run_agent2(p, STORE).tier
    result = run_sensitivity(p)
    assert result.tag == SENSITIVITY_TAG == "[Unverified Claim - Sensitivity Test Only]"
    assert result.noi_tested == Decimal("95000.0")
    assert run_agent2(p, STORE).tier == before


# --- Vector index ---


class FakeEmbedder:
    def embed(self, texts):
        return np.array([[1.0, float(len(t) % 7), float("bank" in t.lower())] for t in texts], dtype="float32")


def test_vector_search_returns_only_the_requested_version():
    index = PolicyVectorIndex(STORE, FakeEmbedder())
    for version in ("2025.1", "2026.2"):
        hits = index.search("bank statements", version, k=50)
        assert hits and {c.policy_version for c, _ in hits} == {version}
