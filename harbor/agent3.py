"""Agent 3: draft applicant letters from the audit report.

Code builds every statement from PolicyAuditReport fields. Claude may only reword
those statements, and a reworded letter is kept only if it passes the validator.
Applicant-written notes are never used. Every letter is a draft for a credit officer.
"""

from __future__ import annotations

import json
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, ConfigDict

from harbor.agent2 import Criterion, PolicyAuditReport
from harbor.credit_rules import CREDIT_RULES
from harbor.llm_parser import MODEL
from harbor.schemas import ApplicationPayload

EQUAL_OPPORTUNITY = "Harbor Ridge Bank provides credit on an equal opportunity basis."
CONTACT_PLACEHOLDER = "[Credit officer name, phone and email]"
ADDRESS_PLACEHOLDER = "[Applicant mailing address]"
MAX_REASONS = 4
MAX_GRADE = 10.0

Section = Literal["summary", "reason", "term", "request", "next"]

_FORBIDDEN = re.compile(
    r"\b(approved|approval|approve|denied|denial|declined|fraud\w*|manipulat\w*|injection|KYC|compliance hold|"
    r"DSCR|EIN_MISMATCH|[A-Z]+_[A-Z_]+)\b",
    re.IGNORECASE,
)
_LONG_DATE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}), (\d{4})\b"
)
_MONTHS = "January February March April May June July August September October November December".split()
_SECTION = re.compile(r"section ([\d.]*\d(?:\(\d+\))?)", re.IGNORECASE)
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LetterStatement(_Strict):
    id: str
    section: Section
    text: str
    grounding: tuple[str, ...]


class GroundingResult(_Strict):
    passed: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class LetterDraft(_Strict):
    application_id: str | None
    letter_type: str
    tier: str
    statements: tuple[LetterStatement, ...]
    rendered_text: str
    method: Literal["template", "template_reworded_by_llm"]
    validation: GroundingResult
    officer_notes: tuple[str, ...] = ()


class PolishedStatement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    text: str


class PolishedLetter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    statements: list[PolishedStatement]


def _long_date(value: date) -> str:
    return f"{value:%B} {value.day}, {value.year}"


def _money(value) -> str:
    return f"${Decimal(str(value)):,.2f}"


def _whole(value) -> str:
    return f"${Decimal(str(value)):,.0f}"


def _cite(citation: str | None) -> str:
    if not citation:
        return ""
    head, section = citation.split(" §")
    policy, version = head.split(" v")
    return f" (Policy {policy}, version {version}, section {section})"


def _criterion(report: PolicyAuditReport, name: str) -> Criterion | None:
    return next((c for c in report.criteria if c.name == name), None)


def _reason_for(c: Criterion, report: PolicyAuditReport) -> str:
    rules = CREDIT_RULES[report.policy_version]
    cite = _cite(c.citation)
    if c.name == "Time in business":
        months = re.match(r"\d+", c.observed)[0]
        return f"Your business had been operating for {months} months when you applied. Our policy requires at least {rules.min_months} months{cite}."
    if c.name == "Loan amount":
        return f"You asked for {c.observed}. Our policy allows loans from {_whole(rules.min_amount)} to {_whole(rules.max_amount)}{cite}."
    if c.name == "Unsecured cap":
        amount = c.observed.split()[0]
        return f"You asked for {amount} without collateral, which means nothing pledged to back the loan. Our policy limits these loans to {_whole(rules.unsecured_cap)}{cite}."
    if c.name == "Industry":
        return f"Your business is in an industry our policy does not lend to{cite}."
    if c.name == "Term":
        return f"You asked to repay over {c.observed}. Our policy allows up to {rules.max_term} months{cite}."
    if c.name == "Credit score":
        if c.status == "PASS":
            return f"Your credit score on file is {c.observed}. Our policy looks for {rules.score_meets} or higher{cite}."
        if c.status == "REFER":
            return f"Your credit score on file is {c.observed}. Our policy sends scores from {rules.score_refer} to {rules.score_meets - 1} to a credit officer for review{cite}."
        return f"Your credit score on file is {c.observed}. Our policy sends scores below {rules.score_refer} for further review{cite}."
    if c.name == "DSCR":
        f = report.financials
        ratio = f"{f.dscr:.2f}"
        intro = (
            f"Your {f.noi_tax_year} business tax return shows income of {_whole(f.noi_used)}. "
            f"We compared that with your yearly loan payments of {_whole(f.annual_debt_service)}, including this loan, and got a coverage ratio of {ratio}"
        )
        if c.status == "PASS":
            return f"{intro}. Our policy looks for {rules.dscr_meets} or higher{cite}."
        if c.status == "REFER":
            return f"{intro}. Our policy sends ratios from {rules.dscr_refer} up to {rules.dscr_meets} to a credit officer for review{cite}."
        return f"{intro}. Our policy sends ratios below {rules.dscr_refer} for further review{cite}."
    raise ValueError(f"No letter wording for criterion {c.name}")


def _named_in_reasons(c: Criterion, report: PolicyAuditReport) -> bool:
    return any(r.startswith(c.name) for r in report.reasons)


INCOME_IMPACT = (
    "The income on your filed tax returns helps decide the size of the loan payment your business can support, "
    "so we need it to review the amount you asked for. We cannot use income that is only stated on the application."
)


def _request_text(item: str, detail: str, kind: str, impacts: tuple[str, ...], citation: str | None, explained: set[str]) -> str:
    label = item
    if item == "Verified net operating income":
        label = f"{item} (the business income shown on a filed tax return)"
    if kind == "stale_document":
        limit = re.search(r"limit is (\d+) days", detail)[1]
        need = f"Please sign and date your application again. Our policy needs a signature from within the last {limit} days"
    elif match := re.match(r"(\d+) of (\d+) months", detail):
        need = f"We have {match[1]} of the {match[2]} months we need"
    else:
        need = "This is not in your file"
    if any("income" in i for i in impacts):
        why = INCOME_IMPACT
    elif any("DSCR" in i for i in impacts):
        why = "We need this to confirm your business income can cover the loan payments."
    elif any("Credit score" in i for i in impacts):
        why = "We need this to review your credit and set your rate."
    else:
        why = "Our policy requires this to complete your file."
    text = f"{label}: {need}{_cite(citation)}."
    if why in explained and why == INCOME_IMPACT:
        return text
    explained.add(why)
    return f"{text} {why}"


def _build_statements(report: PolicyAuditReport) -> tuple[str, str, list[LetterStatement], list[str]]:
    tier = report.tier
    out: list[LetterStatement] = []
    notes: list[str] = []

    def add(section: Section, text: str, *grounding: str) -> None:
        out.append(LetterStatement(id=f"{section}_{sum(s.section == section for s in out) + 1}", section=section, text=text, grounding=grounding))

    def add_reasons(criteria: list[Criterion]) -> None:
        if len(criteria) > MAX_REASONS:
            notes.append(f"{len(criteria)} reasons apply. Policy allows four in the letter; the rest were left out.")
        for c in criteria[:MAX_REASONS]:
            add("reason", _reason_for(c, report), f"criteria:{c.name}")

    if tier == "MEETS CRITERIA":
        f = report.financials
        add("summary", "Thank you for applying. Our review found that your application meets our lending criteria. A credit officer will make the final decision.", "tier")
        add_reasons([c for c in (_criterion(report, "Credit score"), _criterion(report, "DSCR")) if c])
        add("term", f"Loan amount: {_whole(f.principal)} over {f.term_months} months", "financials.principal", "financials.term_months")
        add("term", f"Fixed annual rate: {f.annual_rate_pct}%{_cite(f.rate_citation)}", "financials.annual_rate_pct", "financials.rate_citation")
        add("term", f"Monthly payment: {_money(f.monthly_payment)}", "financials.monthly_payment")
        add("term", f"Origination fee, paid at closing: {_money(f.origination_fee)}{_cite(f.fee_citation)}", "financials.origination_fee", "financials.fee_citation")
        add("term", f"Total interest over the life of the loan: {_money(f.total_interest)}", "financials.total_interest")
        add("next", "These terms are not final until a credit officer signs off. If the loan moves forward, your credit officer will contact you about the closing steps.", "tier")
        return "Proposed Terms Letter", "Why your application meets our criteria:", out, notes

    if tier == "REFER (Policy)":
        add("summary", "Thank you for applying. Your application needs review by a credit officer before we can give you an answer.", "tier")
        add_reasons([c for c in report.criteria if c.status == "REFER"])
        add("next", "Your credit officer will contact you after that review.", "tier")
        return "Referral Notice", "Why it needs this review:", out, notes

    if tier == "REFER (Risk/History)":
        add("summary", "Thank you for applying. Your application needs review by a credit officer before we can give you an answer.", "tier")
        for flag in report.risk_flags[:MAX_REASONS]:
            add("reason", f"Our file shows this item from your credit record: {flag.fact}, dated {_long_date(flag.date)}. A credit officer will look at this.", f"risk:{flag.fact}")
        if any(f.is_mock for f in report.risk_flags):
            notes.append("The risk flag comes from mock bureau data. Confirm it against a real source before sending.")
        add("next", "Your credit officer will contact you after that review. If this item is wrong, please let us know.", "tier")
        return "Referral Notice", "What your file shows:", out, notes

    if tier == "DECLINE REVIEW":
        add("summary", "Thank you for applying. Our review found that your application does not meet part of our lending policy. We have sent it to a credit officer, who will make the final decision.", "tier")
        add_reasons([c for c in report.criteria if c.status == "FAIL" and _named_in_reasons(c, report)])
        add("next", "Your credit officer will contact you with the final decision.", "tier")
        notes.append("If this becomes a final adverse decision, legal or compliance should add any required adverse action notice content. This draft doesn't include it.")
        return "Adverse Action / Referral Notice", "The specific reasons:", out, notes

    if tier in ("PEND (Missing Info)", "STALE"):
        stale = tier == "STALE"
        deadline = _long_date(report.pend.expires_on) if report.pend else None
        if stale:
            add("summary", f"We asked for more information about your application, and we did not receive it by {deadline}. Our review is on pause because your file is out of date.", "tier", "pend.expires_on")
        else:
            add("summary", "Thank you for applying. We cannot finish reviewing your application until we receive the items below. No decision has been made.", "tier")
        explained: set[str] = set()
        for m in report.missing_impacts:
            kind = "stale_document" if "limit is" in m.detail else "document"
            add("request", _request_text(m.item, m.detail, kind, m.impacts, m.policy_citation, explained), f"missing:{m.item}")
        if stale:
            add("next", "To continue, please send everything listed above. Your credit officer can tell you what happens next.", "tier")
        elif report.pend:
            add("next", f"Please send these items by {deadline}, which is {report.pend.limit_days} days from when we paused your file. If we do not have them by then, your file will be marked out of date.", "pend.expires_on", "pend.limit_days")
        else:
            add("next", "Please send these items as soon as you can.", "tier")
            notes.append("No PEND timer is running because no policy version could be selected.")
        return ("Stale File Notice" if stale else "Information Request"), "", out, notes

    if tier == "HOLD FOR COMPLIANCE":
        add("summary", "Thank you for applying. We need more time to finish reviewing your application. No decision has been made.", "tier")
        add("next", "A member of our team will contact you if we need anything else from you.", "tier")
        notes.append("Letter is neutral so it doesn't disclose a compliance review. Policy §6 asks for specific reasons, so Compliance should confirm this wording before anything is sent.")
        notes += [f"Hold reason (internal only): {r}" for r in report.reasons]
        return "Status Notice", "", out, notes

    raise ValueError(f"No letter for tier {tier}")


# --- Grounding validator ---


def _normalize_dates(text: str) -> str:
    return _LONG_DATE.sub(lambda m: f"{m[3]}-{_MONTHS.index(m[1]) + 1:02d}-{int(m[2]):02d}", text)


def _figures(text: str) -> set[str | Decimal]:
    """Dates and numbers in the text, normalized so $1,500.00 equals 1500."""
    text = _normalize_dates(text)
    found: set[str | Decimal] = set(_ISO_DATE.findall(text))
    for raw in _NUMBER.findall(_ISO_DATE.sub(" ", text)):
        try:
            found.add(Decimal(raw.rstrip(",").replace(",", "")))
        except InvalidOperation:
            pass
    return found


def _report_evidence(report: PolicyAuditReport) -> tuple[set[str | Decimal], set[str]]:
    dumped = report.model_dump(
        mode="json",
        include={"application_id", "criteria", "financials", "risk_flags", "missing_impacts", "pend", "routing_citation"},
    )
    text = json.dumps(dumped, ensure_ascii=False)
    figures = _figures(text)
    if report.financials and report.financials.dscr is not None:
        figures.add(report.financials.dscr.quantize(Decimal("0.01")))
    return figures, set(re.findall(r"§([\d.]*\d(?:\(\d+\))?)", text))


def _resolve(report: PolicyAuditReport, ref: str) -> bool:
    if ":" in ref:
        kind, name = ref.split(":", 1)
        items = {
            "criteria": [c.name for c in report.criteria],
            "missing": [m.item for m in report.missing_impacts],
            "risk": [r.fact for r in report.risk_flags],
        }[kind]
        return name in items
    value = report
    for part in ref.split("."):
        value = getattr(value, part, None)
        if value is None:
            return False
    return True


def _grade_level(text: str) -> float:
    sentences = max(len(re.findall(r"[.!?]+(?:\s|$)", text)), 1)
    words = re.findall(r"[A-Za-z']+", text)
    syllables = sum(max(len(re.findall(r"[aeiouy]+", w.lower())) - (w.lower().endswith("e") and len(w) > 3), 1) for w in words)
    return 0.39 * len(words) / sentences + 11.8 * syllables / max(len(words), 1) - 15.59


def validate_letter(text: str, report: PolicyAuditReport, known_names: tuple[str, ...] = ()) -> GroundingResult:
    """Check a letter, including an officer-edited one, against the audit report."""
    errors: list[str] = []
    warnings: list[str] = []
    allowed_figures, allowed_sections = _report_evidence(report)

    for required, label in (
        (EQUAL_OPPORTUNITY, "equal opportunity sentence"),
        ("DRAFT", "draft marking"),
        ("Questions?", "contact line"),
        (f"version {report.policy_version}", "policy version"),
        ("SB-100", "policy number"),
    ):
        if report.policy_version is None and label == "policy version":
            continue
        if required not in text:
            errors.append(f"Missing required {label}.")

    for match in _FORBIDDEN.finditer(text):
        errors.append(f"Term not allowed in an applicant letter: {match[0]}")

    for section in _SECTION.findall(text):
        if section not in allowed_sections:
            errors.append(f"Section {section} is not cited in the audit report.")

    scan = _SECTION.sub(" ", text)
    for name in known_names:
        scan = scan.replace(name, " ")
    scan = scan.replace("SB-100", " ")
    if report.policy_version:
        scan = scan.replace(report.policy_version, " ")
    letter_date = _long_date(report.review_date)
    scan = scan.replace(letter_date, " ")
    for figure in _figures(scan) - allowed_figures:
        errors.append(f"Figure or date not in the audit report: {figure}")

    lowered = text.lower()
    f = report.financials
    if report.tier == "MEETS CRITERIA":
        for label, value in (("monthly payment", _money(f.monthly_payment)), ("origination fee", _money(f.origination_fee)), ("total interest", _money(f.total_interest)), ("rate", f"{f.annual_rate_pct}%")):
            if value not in text:
                errors.append(f"Missing exact {label}: {value}")
    if report.tier in ("PEND (Missing Info)", "STALE"):
        for m in report.missing_impacts:
            if m.item.lower() not in lowered:
                errors.append(f"Missing required item: {m.item}")
        if report.pend and _long_date(report.pend.expires_on) not in text:
            errors.append("Missing the 14-day deadline date.")
    if report.tier == "HOLD FOR COMPLIANCE" and "$" in text:
        errors.append("A compliance hold letter must not contain loan figures.")

    if CONTACT_PLACEHOLDER in text:
        warnings.append("Fill in the credit officer contact line.")
    if ADDRESS_PLACEHOLDER in text:
        warnings.append("Fill in the applicant mailing address.")
    grade = _grade_level(text)
    if grade > MAX_GRADE:
        warnings.append(f"Estimated reading level is grade {grade:.0f}; policy asks for about grade 8.")
    return GroundingResult(passed=not errors, errors=tuple(dict.fromkeys(errors)), warnings=tuple(warnings))


# --- Rendering ---

_ENV = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"), undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True
)


def _names(payload: ApplicationPayload) -> tuple[str, str]:
    record = payload.verified.customer_record
    owner = (record.owner if record else None) or (payload.claimed.owner_name.value if payload.claimed.owner_name else "Applicant")
    business = (record.business if record else None) or (payload.claimed.business_name.value if payload.claimed.business_name else "")
    return owner, business


def _render(payload: ApplicationPayload, report: PolicyAuditReport, statements, heading: str) -> str:
    owner, business = _names(payload)
    by = lambda section: [s for s in statements if s.section == section]
    return _ENV.get_template("letter.txt.j2").render(
        letter_date=_long_date(report.review_date), addressee=owner, business=business,
        application_id=report.application_id or "", summary=by("summary"), reasons=by("reason"),
        reasons_heading=heading, terms=by("term"), requests=by("request"), next_steps=by("next"),
        policy_id="SB-100", policy_version=report.policy_version or "",
    )


def known_names(payload: ApplicationPayload, report: PolicyAuditReport) -> tuple[str, ...]:
    return (*_names(payload), report.application_id or "applicant")


def run_agent3(payload: ApplicationPayload, report: PolicyAuditReport) -> LetterDraft:
    letter_type, heading, statements, notes = _build_statements(report)
    unresolved = [f"{s.id}: {r}" for s in statements for r in s.grounding if not _resolve(report, r)]
    text = _render(payload, report, statements, heading)
    result = validate_letter(text, report, known_names(payload, report))
    if unresolved:
        result = GroundingResult(passed=False, errors=(*result.errors, *(f"Unresolved source: {u}" for u in unresolved)), warnings=result.warnings)
    return LetterDraft(
        application_id=report.application_id, letter_type=letter_type, tier=report.tier, statements=tuple(statements),
        rendered_text=text, method="template", validation=result, officer_notes=tuple(notes),
    )


POLISH_SYSTEM = """You improve the wording of statements in a bank's letter to a loan applicant.
Rewrite each statement in warm, clear language at about an 8th-grade reading level.
Keep every number, date, dollar amount, percentage and "(Policy ... section ...)" reference exactly.
Add no facts, reasons or promises. Remove none. Keep each statement's id.
Never use the words approved, approval, denied, declined or DSCR."""


def polish_letter(draft: LetterDraft, payload: ApplicationPayload, report: PolicyAuditReport, client) -> tuple[LetterDraft, str | None]:
    """Reword with Claude. Returns the original draft and a reason if any check fails."""
    if draft.tier == "HOLD FOR COMPLIANCE":
        return draft, "Hold letters are not reworded."
    request = json.dumps([{"id": s.id, "text": s.text} for s in draft.statements], ensure_ascii=False)
    response = client.beta.messages.parse(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "high"},
        system=POLISH_SYSTEM,
        messages=[{"role": "user", "content": request}],
        output_format=PolishedLetter,
    )
    if response.stop_reason != "end_turn" or response.parsed_output is None:
        return draft, f"No usable rewording (stop_reason={response.stop_reason})."
    rewritten = {s.id: s.text for s in response.parsed_output.statements}
    if set(rewritten) != {s.id for s in draft.statements}:
        return draft, "Rewording added or dropped a statement."
    for original in draft.statements:
        new = rewritten[original.id]
        if _figures(new) != _figures(original.text) or _SECTION.findall(new) != _SECTION.findall(original.text):
            return draft, f"Rewording changed a figure or citation in {original.id}."
    statements = tuple(s.model_copy(update={"text": rewritten[s.id]}) for s in draft.statements)
    heading = _build_statements(report)[1]
    text = _render(payload, report, statements, heading)
    result = validate_letter(text, report, known_names(payload, report))
    if not result.passed:
        return draft, "Rewording failed validation: " + "; ".join(result.errors)
    return draft.model_copy(update={"statements": statements, "rendered_text": text, "method": "template_reworded_by_llm", "validation": result}), None
