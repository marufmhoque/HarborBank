"""Agent 2: credit and policy audit. Deterministic: no model is involved.

Routing order: compliance hold, hard-eligibility failure, missing information,
then the credit and cash-flow tests. Every output is a recommendation for a
credit officer (Policy SB-100 v2026.2 §4).
"""

from __future__ import annotations

import hashlib
from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from harbor.agent1 import _months_between
from harbor.credit_rules import CREDIT_RULES, CreditRules
from harbor.financials import dscr as compute_dscr, loan_terms, monthly_payment, to_cents
from harbor.pend_timer import PEND_LIMIT_DAYS, PendStore
from harbor.policy_index import ClauseStore
from harbor.schemas import ApplicationPayload

Tier = Literal[
    "MEETS CRITERIA",
    "REFER (Policy)",
    "REFER (Risk/History)",
    "PEND (Missing Info)",
    "STALE",
    "DECLINE REVIEW",
    "HOLD FOR COMPLIANCE",
]
Status = Literal["PASS", "REFER", "FAIL", "MISSING", "NOT EVALUATED"]
DataBasis = Literal["application_request", "application_face", "verified", "verified_with_claimed_debt", "none"]

SENSITIVITY_TAG = "[Unverified Claim - Sensitivity Test Only]"

# Agent 1 audit codes that mean "identity, EIN, name or income facts conflict,
# or the file shows signs of fraud or manipulation."
HOLD_CODES = frozenset(
    {
        "EIN_MISMATCH",
        "BUSINESS_NAME_MISMATCH",
        "OWNER_NAME_MISMATCH",
        "INCOME_MISMATCH",
        "TAX_FIGURE_CONFLICT",
        "CROSS_APPLICATION_CONFLICT",
        "CROSS_APPLICATION_INCOME_CONFLICT",
        "SUSPECTED_PROMPT_INJECTION",
    }
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Criterion(_Strict):
    name: str
    status: Status
    observed: str
    threshold: str
    data_basis: DataBasis
    citation: str | None
    clause_text: str | None
    note: str | None = None


class ExposureLine(_Strict):
    label: str
    annual_debt_service: Decimal
    basis: str


class Financials(_Strict):
    principal: Decimal
    annual_rate_pct: Decimal
    term_months: int
    monthly_payment: Decimal
    origination_fee: Decimal
    total_interest: Decimal
    total_of_payments: Decimal
    rate_citation: str
    fee_citation: str
    dscr: Decimal | None = None
    noi_used: Decimal | None = None
    noi_tax_year: int | None = None
    annual_debt_service: Decimal | None = None
    exposure: tuple[ExposureLine, ...] = ()


class RiskFlag(_Strict):
    """A fact for the officer. Agent 2 draws no credit conclusion from it."""

    fact: str
    date: date | None
    source: str
    is_mock: bool
    policy_basis: str


class MissingImpact(_Strict):
    item: str
    detail: str
    policy_citation: str | None
    impacts: tuple[str, ...]


class PendInfo(_Strict):
    pended_on: date
    expires_on: date
    days_elapsed: int
    limit_days: int = PEND_LIMIT_DAYS


class PolicyAuditReport(_Strict):
    schema_version: Literal["1.0"] = "1.0"
    application_id: str | None
    review_date: date
    policy_version: str | None
    tier: Tier
    reasons: tuple[str, ...]
    routing_citation: str | None
    criteria: tuple[Criterion, ...]
    financials: Financials | None
    risk_flags: tuple[RiskFlag, ...] = ()
    missing_impacts: tuple[MissingImpact, ...] = ()
    required_noi_to_meet_dscr: Decimal | None = None
    also_noted: tuple[str, ...] = ()
    pend: PendInfo | None = None
    officer_notice: str = Field(default="Recommendation only. A credit officer makes every decision.")


class SensitivityResult(_Strict):
    tag: Literal["[Unverified Claim - Sensitivity Test Only]"] = SENSITIVITY_TAG
    noi_tested: Decimal
    dscr: Decimal
    outcome_if_verified: Literal["meets test", "refer", "decline review"]
    annual_debt_service: Decimal


def _money(value: float | Decimal) -> str:
    return f"${Decimal(str(value)):,.0f}"


class _Ctx:
    def __init__(self, payload: ApplicationPayload, rules: CreditRules, store: ClauseStore) -> None:
        self.payload = payload
        self.rules = rules
        self.store = store

    def citation(self, key: str) -> str | None:
        clause_id = self.rules.clauses.get(key)
        return f"SB-100 v{self.rules.version} §{clause_id}" if clause_id else None

    def clause_text(self, key: str) -> str | None:
        clause_id = self.rules.clauses.get(key)
        clause = self.store.get(self.rules.version, clause_id) if clause_id else None
        return clause.text if clause else None

    def criterion(self, name, status, observed, threshold, basis, key, note=None) -> Criterion:
        return Criterion(
            name=name, status=status, observed=observed, threshold=threshold, data_basis=basis,
            citation=self.citation(key), clause_text=self.clause_text(key), note=note,
        )


def _hard_criteria(ctx: _Ctx) -> list[Criterion]:
    claimed, rules = ctx.payload.claimed, ctx.rules
    out: list[Criterion] = []

    start = claimed.business_start_date
    as_of = claimed.submitted_date or claimed.signed_date
    if start and as_of:
        months = _months_between(start.value, as_of.value)
        out.append(ctx.criterion("Time in business", "PASS" if months >= rules.min_months else "FAIL", f"{months} months at {as_of.value}", f"At least {rules.min_months} months", "application_face", "time_in_business"))
    else:
        out.append(ctx.criterion("Time in business", "NOT EVALUATED", "Dates not available", f"At least {rules.min_months} months", "none", "time_in_business"))

    amount = Decimal(str(claimed.requested_amount.value)) if claimed.requested_amount else None
    if amount is not None:
        ok = rules.min_amount <= amount <= rules.max_amount
        out.append(ctx.criterion("Loan amount", "PASS" if ok else "FAIL", _money(amount), f"{_money(rules.min_amount)} to {_money(rules.max_amount)}", "application_request", "amount"))
    else:
        out.append(ctx.criterion("Loan amount", "NOT EVALUATED", "Not stated", f"{_money(rules.min_amount)} to {_money(rules.max_amount)}", "none", "amount"))

    if amount is not None and claimed.secured is not None:
        secured = claimed.secured.value
        ok = secured or amount <= rules.unsecured_cap
        observed = f"{_money(amount)} {'secured' if secured else 'unsecured'}"
        out.append(ctx.criterion("Unsecured cap", "PASS" if ok else "FAIL", observed, f"Unsecured up to {_money(rules.unsecured_cap)}", "application_request", "unsecured_cap"))
    else:
        out.append(ctx.criterion("Unsecured cap", "NOT EVALUATED", "Not stated", f"Unsecured up to {_money(rules.unsecured_cap)}", "none", "unsecured_cap"))

    if claimed.industry:
        excluded = any(term in claimed.industry.value.lower() for term in rules.excluded_industries)
        out.append(ctx.criterion("Industry", "FAIL" if excluded else "PASS", claimed.industry.value, "Not an excluded industry", "application_face", "industry"))
    else:
        out.append(ctx.criterion("Industry", "NOT EVALUATED", "Not stated", "Not an excluded industry", "none", "industry"))

    if claimed.term_months:
        ok = claimed.term_months.value <= rules.max_term
        out.append(ctx.criterion("Term", "PASS" if ok else "FAIL", f"{claimed.term_months.value} months", f"Up to {rules.max_term} months", "application_request", "term"))
    else:
        out.append(ctx.criterion("Term", "NOT EVALUATED", "Not stated", f"Up to {rules.max_term} months", "none", "term"))
    return out


def _score_status(score: int, rules: CreditRules) -> Status:
    if score >= rules.score_meets:
        return "PASS"
    return "REFER" if score >= rules.score_refer else "FAIL"


def _dscr_status(value: Decimal, rules: CreditRules) -> Status:
    if value >= rules.dscr_meets:
        return "PASS"
    return "REFER" if value >= rules.dscr_refer else "FAIL"


def _dscr_key(status: Status) -> str:
    return {"PASS": "dscr_meets", "REFER": "dscr_refer", "FAIL": "dscr_decline"}[status]


def _debt_service(ctx: _Ctx, rate_pct: Decimal) -> tuple[list[ExposureLine], list[str]]:
    """Annual debt service lines, plus the IDs of related applications that can't be priced."""
    claimed = ctx.payload.claimed
    principal = Decimal(str(claimed.requested_amount.value))
    months = claimed.term_months.value
    lines = [
        ExposureLine(
            label="Existing business debt",
            annual_debt_service=to_cents(Decimal(str(claimed.claimed_existing_monthly_debt.value)) * 12),
            basis="Stated on application; not independently verified",
        ),
        ExposureLine(
            label="Proposed loan",
            annual_debt_service=monthly_payment(principal, rate_pct, months) * 12,
            basis=f"{rate_pct}% over {months} months",
        ),
    ]
    unpriced: list[str] = []
    if ctx.rules.combined_exposure_required:
        for related in ctx.payload.related_applications:
            if related.requested_amount and related.term_months:
                payment = monthly_payment(Decimal(str(related.requested_amount)), rate_pct, related.term_months)
                lines.append(
                    ExposureLine(
                        label=f"Open application {related.application_id}",
                        annual_debt_service=payment * 12,
                        basis=f"{_money(related.requested_amount)} over {related.term_months} months at {rate_pct}%, the rate for this score",
                    )
                )
            else:
                unpriced.append(related.application_id)
    return lines, unpriced


def _risk_flags(payload: ApplicationPayload, store: ClauseStore) -> list[RiskFlag]:
    stub = payload.verified.bureau.bureau_metadata_stub
    version = payload.policy.version if payload.policy else None
    facts: list[tuple[str, str, date | None]] = []
    if stub.bankruptcy_flag:
        facts.append(("Bankruptcy on record", "bankruptcy", stub.bankruptcy_date))
    if stub.public_records_count:
        facts.append((f"{stub.public_records_count} public record(s)", "public record", None))
    if stub.tax_lien_flag:
        facts.append(("Tax lien on record", "lien", None))
    if stub.judgment_flag:
        facts.append(("Judgment on record", "judgment", None))

    flags: list[RiskFlag] = []
    for fact, term, when in facts:
        clauses = store.mentions(version, term) if version else []
        basis = (
            "; ".join(f"SB-100 v{version} §{c.clause_id}" for c in clauses)
            if clauses
            else f"SB-100 v{version} has no clause on this. Surfaced for officer judgment; no rule applied."
        )
        flags.append(
            RiskFlag(
                fact=fact, date=when or payload.verified.bureau.pull_date,
                source="customers.json bureau_metadata_stub", is_mock=stub.is_mock, policy_basis=basis,
            )
        )
    return flags


def _impacts(item: str) -> tuple[str, ...]:
    lowered = item.lower()
    if "tax return" in lowered or "net operating income" in lowered:
        return ("DSCR test cannot run on verified income", "Required documents")
    if "credit score" in lowered:
        return ("Credit score test", "Pricing", "DSCR test")
    if "existing monthly debt" in lowered or "requested amount" in lowered or lowered == "term":
        return ("DSCR test", "Monthly payment and pricing")
    return ("Required documents",)


def _fingerprint(payload: ApplicationPayload) -> str:
    joined = "|".join(sorted(d.sha256 for d in payload.source_documents))
    return hashlib.sha256(joined.encode()).hexdigest()


def run_agent2(payload: ApplicationPayload, store: ClauseStore, pend_store: PendStore | None = None) -> PolicyAuditReport:
    pend_store = pend_store if pend_store is not None else PendStore()
    common = {"application_id": payload.application_id, "review_date": payload.review_date}
    missing_impacts = tuple(
        MissingImpact(item=m.item, detail=m.detail, policy_citation=m.policy_citation, impacts=_impacts(m.item))
        for m in payload.missing.items
    )

    if payload.policy is None:
        return PolicyAuditReport(
            **common, policy_version=None, tier="PEND (Missing Info)",
            reasons=("No policy version could be selected because the signature date is missing or predates all versions.",),
            routing_citation=None, criteria=(), financials=None, missing_impacts=missing_impacts,
        )

    rules = CREDIT_RULES[payload.policy.version]
    ctx = _Ctx(payload, rules, store)
    common["policy_version"] = rules.version
    risk_flags = tuple(_risk_flags(payload, store))
    hard = _hard_criteria(ctx)
    hard_failures = [c for c in hard if c.status == "FAIL"]
    gaps = tuple(f"{m.item}: {m.detail}" for m in payload.missing.items)
    doc_items = [m for m in payload.missing.items if m.kind != "field"]
    documents = ctx.criterion(
        "Required documents", "MISSING" if doc_items else "PASS",
        "; ".join(m.item for m in doc_items) or "All received", "Listed in Section 3", "verified", "documents",
    )

    # 1. Compliance hold stops underwriting.
    holds = [e for e in payload.audit_log if e.code in HOLD_CODES]
    if holds:
        stopped = [
            ctx.criterion(name, "NOT EVALUATED", "Underwriting stopped by compliance hold", "", "none", key)
            for name, key in (("Credit score", "score"), ("DSCR", "dscr_meets"))
        ]
        noted = tuple(f"Hard eligibility issue: {c.name}: {c.observed} ({c.citation})" for c in hard_failures) + gaps
        return PolicyAuditReport(
            **common, tier="HOLD FOR COMPLIANCE", reasons=tuple(dict.fromkeys(e.message for e in holds)),
            routing_citation=ctx.citation("hold_route"), criteria=(*hard, documents, *stopped), financials=None,
            risk_flags=risk_flags, missing_impacts=missing_impacts, also_noted=noted,
        )

    # Everything needed for the credit tests that exists on this file.
    claimed = payload.claimed
    score = payload.verified.bureau.credit_score
    band = rules.rate_for(score) if score is not None else None
    can_price = band is not None and claimed.requested_amount and claimed.term_months
    financials = None
    criteria = [*hard, documents]
    if score is not None:
        criteria.append(ctx.criterion("Credit score", _score_status(score, rules), str(score), f"{rules.score_meets}+ meets; {rules.score_refer}-{rules.score_meets - 1} refer", "verified", "score"))
    else:
        criteria.append(ctx.criterion("Credit score", "NOT EVALUATED", "Not available", f"{rules.score_meets}+ meets", "none", "score"))

    required_noi = None
    dscr_status: Status = "NOT EVALUATED"
    unpriced: list[str] = []
    if can_price:
        principal = Decimal(str(claimed.requested_amount.value))
        terms = loan_terms(principal, band.rate_pct, claimed.term_months.value, rules.fee_pct)
        lines: list[ExposureLine] = []
        total = None
        if claimed.claimed_existing_monthly_debt:
            lines, unpriced = _debt_service(ctx, band.rate_pct)
            total = sum((line.annual_debt_service for line in lines), Decimal(0))
            required_noi = to_cents(rules.dscr_meets * total)
        noi = payload.verified.verified_noi
        ratio = None
        if noi and total and not unpriced:
            ratio = compute_dscr(Decimal(str(noi.amount)), total)
            dscr_status = _dscr_status(ratio, rules)
            criteria.append(
                ctx.criterion(
                    "DSCR", dscr_status, f"{ratio:.2f} ({_money(noi.amount)} / {_money(total)})",
                    f"{rules.dscr_meets}+ meets; {rules.dscr_refer} to below {rules.dscr_meets} refer",
                    "verified_with_claimed_debt", _dscr_key(dscr_status),
                    note="Income is verified. Existing debt payments are as stated on the application.",
                )
            )
        else:
            detail = f"Can't price open application(s): {', '.join(unpriced)}" if unpriced else "Needs verified income and existing debt"
            criteria.append(ctx.criterion("DSCR", "NOT EVALUATED", detail, f"{rules.dscr_meets}+ meets", "none", "dscr_meets"))
        financials = Financials(
            principal=principal, annual_rate_pct=band.rate_pct, term_months=claimed.term_months.value,
            monthly_payment=terms.monthly_payment, origination_fee=terms.origination_fee, total_interest=terms.total_interest,
            total_of_payments=terms.total_of_payments, rate_citation=f"SB-100 v{rules.version} §{band.clause}",
            fee_citation=ctx.citation("fee"), dscr=ratio, noi_used=Decimal(str(noi.amount)) if noi else None,
            noi_tax_year=noi.tax_year if noi else None, annual_debt_service=total, exposure=tuple(lines),
        )
    else:
        criteria.append(ctx.criterion("DSCR", "NOT EVALUATED", "No rate band or loan terms available", f"{rules.dscr_meets}+ meets", "none", "dscr_meets"))

    result = dict(common, criteria=tuple(criteria), financials=financials, risk_flags=risk_flags, missing_impacts=missing_impacts, required_noi_to_meet_dscr=required_noi)

    # 2. Hard failure goes to decline review without pending for documents.
    if hard_failures:
        pend_store.clear(payload.application_id)
        return PolicyAuditReport(
            **result, tier="DECLINE REVIEW", reasons=tuple(f"{c.name}: {c.observed}. Requires {c.threshold.lower()} ({c.citation})." for c in hard_failures),
            routing_citation=ctx.citation("hard_fail_route"), also_noted=gaps,
        )

    # 3. Missing information pends, then goes stale after 14 days.
    # A score outside every rate band is a result, not missing information.
    unpriceable = score is not None and band is None
    unresolved = [c for c in criteria if c.status == "NOT EVALUATED" and not (c.name == "DSCR" and unpriceable)]
    if payload.missing.items or unresolved:
        reasons = [f"{m.item}: {m.detail}" for m in payload.missing.items] + [f"{c.name}: {c.observed}." for c in unresolved if not payload.missing.items]
        status = pend_store.check(payload.application_id or "unknown", _fingerprint(payload), payload.review_date)
        pend = PendInfo(pended_on=status.pended_on, expires_on=status.expires_on, days_elapsed=status.days_elapsed)
        if status.expired:
            reasons.insert(0, f"Pended {status.pended_on} with no document update for {status.days_elapsed} days, past the {PEND_LIMIT_DAYS}-day limit.")
        return PolicyAuditReport(
            **result, tier="STALE" if status.expired else "PEND (Missing Info)", reasons=tuple(reasons),
            routing_citation=ctx.citation("pend_route"), pend=pend,
        )

    pend_store.clear(payload.application_id)

    # 4. Credit and cash-flow tests.
    tested = [c for c in criteria if c.name in ("Credit score", "DSCR")]
    failed = [c for c in tested if c.status == "FAIL"]
    if failed:
        return PolicyAuditReport(**result, tier="DECLINE REVIEW", reasons=tuple(f"{c.name} {c.observed} is below the referral range ({c.citation})." for c in failed), routing_citation=failed[0].citation)
    if risk_flags:
        return PolicyAuditReport(**result, tier="REFER (Risk/History)", reasons=tuple(f"{f.fact} (dated {f.date}, source: {f.source})." for f in risk_flags), routing_citation=None)
    referred = [c for c in tested if c.status == "REFER"]
    if referred:
        return PolicyAuditReport(**result, tier="REFER (Policy)", reasons=tuple(f"{c.name} {c.observed} is in the referral range ({c.citation})." for c in referred), routing_citation=referred[0].citation)
    return PolicyAuditReport(**result, tier="MEETS CRITERIA", reasons=("All policy criteria are met using verified figures.",), routing_citation=None)


def run_sensitivity(payload: ApplicationPayload, noi: float | None = None) -> SensitivityResult | None:
    """What-if DSCR on a claimed income figure. Never feeds the tier."""
    claimed = payload.claimed
    tested = noi if noi is not None else (claimed.claimed_income.value if claimed.claimed_income else None)
    score = payload.verified.bureau.credit_score
    if payload.policy is None or tested is None or score is None or not (claimed.requested_amount and claimed.term_months and claimed.claimed_existing_monthly_debt):
        return None
    rules = CREDIT_RULES[payload.policy.version]
    band = rules.rate_for(score)
    if band is None:
        return None
    lines, unpriced = _debt_service(_Ctx(payload, rules, None), band.rate_pct)
    if unpriced:
        return None
    total = sum((line.annual_debt_service for line in lines), Decimal(0))
    ratio = compute_dscr(Decimal(str(tested)), total)
    outcome = {"PASS": "meets test", "REFER": "refer", "FAIL": "decline review"}[_dscr_status(ratio, rules)]
    return SensitivityResult(noi_tested=Decimal(str(tested)), dscr=ratio, outcome_if_verified=outcome, annual_debt_service=total)
