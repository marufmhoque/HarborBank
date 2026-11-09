"""Agent 1: document parsing, data partitioning and cross-checks.

Agent 1 reports what the file says and where it conflicts. Applying the decision
rules (PEND, HOLD, DECLINE REVIEW and so on) is left to the next agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol, Sequence

from harbor.bureau import read_bureau_snapshot
from harbor.form_parser import is_application_form, parse_application_form
from harbor.ingest import SourceDocument
from harbor.policy_registry import PolicyVersion, select_policy
from harbor.schemas import (
    ApplicationPayload,
    AppliedPolicy,
    AuditEvent,
    Claimed,
    ClaimedData,
    ClaimedSource,
    CustomerRecord,
    DocExtraction,
    DocumentReceipt,
    MissingData,
    MissingItem,
    RelatedApplication,
    SourceDocumentRecord,
    TaxFigure,
    TaxReturnSource,
    UntrustedNote,
    VerifiedData,
    VerifiedEIN,
    VerifiedNOI,
)

INCOME_TOLERANCE = 0.05

_INJECTION_PATTERNS = [
    r"ignore (all |any )?(prior|previous|above|earlier) instructions",
    r"system (note|prompt|message)",
    r"\bAI (reviewer|agent|model|system|assistant)\b",
    r"\bpre-?approved\b",
    r"mark (this|the) (application|loan)[^.]*\bapproved\b",
    r"skip [^.]*\b(check|checks|verification|review)\b",
    r"do not mention this",
]
_INJECTION = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)
_DOCS_BY_REFERENCE = re.compile(
    r"(provided|submitted|attached|sent|on file)\b.*\b(recent|previous|prior|earlier|last|other)\b", re.IGNORECASE
)


class DocumentParser(Protocol):
    def extract(self, doc: SourceDocument) -> DocExtraction: ...


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION.search(text))


def _months_between(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + end.month - start.month - (end.day < start.day)


def _same(a: str, b: str) -> bool:
    return " ".join(a.split()).casefold() == " ".join(b.split()).casefold()


@dataclass
class _Run:
    audit: list[AuditEvent] = field(default_factory=list)
    missing: list[MissingItem] = field(default_factory=list)

    def log(self, code: str, severity: str, message: str, **kwargs) -> None:
        self.audit.append(AuditEvent(code=code, severity=severity, message=message, **kwargs))


def _extract_document(
    doc: SourceDocument, llm: DocumentParser | None, run: _Run
) -> tuple[DocExtraction | None, str | None]:
    if is_application_form(doc.text):
        return parse_application_form(doc.text), "rule_parser"
    if llm is None:
        run.log(
            "EXTRACTION_UNAVAILABLE",
            "warning",
            f"{doc.name} needs the LLM parser, which is not configured. It was not used.",
            documents=(doc.name,),
        )
        return None, None
    try:
        return llm.extract(doc), "llm_parser"
    except Exception as exc:  # A failed read must leave the document uncounted.
        run.log(
            "EXTRACTION_FAILED",
            "warning",
            f"{doc.name} could not be parsed ({type(exc).__name__}). It was not used.",
            documents=(doc.name,),
        )
        return None, None


def _parse_date(value: str | None, name: str, document: str, run: _Run) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        run.log("INVALID_DATE", "warning", f"Could not read {name} value {value!r}.", field=name, documents=(document,))
        return None


def _build_claimed(form: DocExtraction | None, document: str, method: str | None, run: _Run) -> ClaimedData:
    if form is None:
        return ClaimedData()
    evidence = {e.field: e.text for e in form.evidence}

    def claim(source_field: str, value=None):
        value = getattr(form, source_field) if value is None else value
        if value is None:
            return None
        source = ClaimedSource(document=document, evidence=evidence.get(source_field, ""), method=method)
        return Claimed(value=value, source=source)

    def claim_date(source_field: str):
        return claim(source_field, _parse_date(getattr(form, source_field), source_field, document, run))

    return ClaimedData(
        customer_id=claim("customer_id"),
        business_name=claim("business_name"),
        ein=claim("ein"),
        owner_name=claim("owner_name"),
        ownership_pct=claim("ownership_pct"),
        business_start_date=claim_date("business_start_date"),
        industry=claim("industry"),
        requested_amount=claim("requested_amount"),
        term_months=claim("term_months"),
        secured=claim("secured"),
        collateral_description=claim("collateral_description"),
        collateral_value=claim("collateral_value"),
        purpose=claim("purpose"),
        claimed_income=claim("stated_income"),
        claimed_existing_monthly_debt=claim("existing_monthly_debt"),
        signed_date=claim_date("signed_date"),
        submitted_date=claim_date("submitted_date"),
    )


def _collect_receipts(parsed: Sequence[tuple[SourceDocument, DocExtraction, str]]) -> list[DocumentReceipt]:
    receipts: list[DocumentReceipt] = []
    for doc, extraction, _ in parsed:
        if extraction.document_type == "application_form":
            receipts += [
                DocumentReceipt(
                    kind=i.kind,
                    tax_year=i.tax_year,
                    months=tuple(i.months),
                    basis="intake_log",
                    source_document=doc.name,
                    evidence=i.evidence,
                )
                for i in extraction.documents_received
            ]
            continue
        if extraction.document_type == "tax_return":
            receipts += [
                DocumentReceipt(
                    kind="tax_return", tax_year=f.tax_year, basis="primary_document", source_document=doc.name, evidence=f.evidence
                )
                for f in extraction.tax_figures
            ]
        elif extraction.document_type == "bank_statement":
            months = tuple(m for i in extraction.documents_received for m in i.months)
            receipts.append(
                DocumentReceipt(
                    kind="bank_statement", months=months, basis="primary_document", source_document=doc.name, evidence=doc.name
                )
            )
        elif extraction.document_type in ("photo_id", "business_license"):
            receipts.append(
                DocumentReceipt(
                    kind=extraction.document_type, basis="primary_document", source_document=doc.name, evidence=doc.name
                )
            )
    return receipts


def _verify_tax_figures(
    parsed: Sequence[tuple[SourceDocument, DocExtraction, str]], receipts: Sequence[DocumentReceipt], run: _Run
) -> tuple[list[VerifiedNOI], list[VerifiedEIN]]:
    """Promote tax figures to verified, preferring the return itself over an intake abstract."""
    received_years = {r.tax_year for r in receipts if r.kind == "tax_return"}
    by_year: dict[int, VerifiedNOI] = {}
    eins: list[VerifiedEIN] = []

    ordered = sorted(parsed, key=lambda p: p[1].document_type == "tax_return", reverse=True)
    for doc, extraction, method in ordered:
        primary = extraction.document_type == "tax_return"
        if not primary and extraction.document_type != "application_form":
            continue
        for figure in extraction.tax_figures:
            if figure.tax_year not in received_years:
                run.log(
                    "TAX_FIGURE_WITHOUT_RETURN",
                    "warning",
                    f"A {figure.tax_year} tax figure is quoted but no {figure.tax_year} return is logged. Not treated as verified.",
                    field="verified_noi",
                    documents=(doc.name,),
                )
                continue
            source = TaxReturnSource(
                document=doc.name, evidence=figure.evidence, method=method, primary_document_reviewed=primary
            )
            if figure.ein:
                eins.append(VerifiedEIN(tax_year=figure.tax_year, ein=figure.ein, source=source))
            if figure.net_operating_income is None:
                continue
            existing = by_year.get(figure.tax_year)
            if existing is None:
                by_year[figure.tax_year] = VerifiedNOI(
                    tax_year=figure.tax_year, amount=figure.net_operating_income, source=source
                )
            elif existing.amount != figure.net_operating_income:
                run.log(
                    "TAX_FIGURE_CONFLICT",
                    "critical",
                    f"Tax year {figure.tax_year} NOI differs between documents.",
                    field="verified_noi",
                    values={existing.source.document: str(existing.amount), doc.name: str(figure.net_operating_income)},
                    documents=(existing.source.document, doc.name),
                )
    return sorted(by_year.values(), key=lambda n: n.tax_year, reverse=True), eins


def _find_missing_documents(
    policy: PolicyVersion, claimed: ClaimedData, receipts: Sequence[DocumentReceipt], review_date: date, run: _Run
) -> None:
    signed = claimed.signed_date.value
    start = claimed.business_start_date.value if claimed.business_start_date else None
    application_date = claimed.submitted_date.value if claimed.submitted_date else signed

    latest_year = signed.year - 1
    required_years = [latest_year, latest_year - 1]
    if policy.one_tax_year_if_under_24_months and start and _months_between(start, application_date) < 24:
        required_years = [latest_year]
    received_years = {r.tax_year for r in receipts if r.kind == "tax_return"}
    for year in required_years:
        if year not in received_years:
            run.missing.append(
                MissingItem(
                    item=f"{year} business tax return", kind="document",
                    detail=f"Required return for tax year {year} is not in the file.",
                    policy_citation=policy.cite("3", 1),
                )
            )

    months = sorted({m for r in receipts if r.kind == "bank_statement" for m in r.months})
    if len(months) < policy.bank_statement_months:
        covered = f" Received: {months[0]} to {months[-1]}." if months else ""
        run.missing.append(
            MissingItem(
                item="Business bank statements", kind="document",
                detail=f"{len(months)} of {policy.bank_statement_months} months received.{covered}",
                policy_citation=policy.cite("3", 2),
            )
        )

    kinds = {r.kind for r in receipts}
    if "photo_id" not in kinds:
        run.missing.append(
            MissingItem(item="Government-issued photo ID", kind="document", detail="Not in the file.", policy_citation=policy.cite("3", 3))
        )
    if "business_license" not in kinds:
        run.missing.append(
            MissingItem(item="Business license", kind="document", detail="Not in the file.", policy_citation=policy.cite("3", 4))
        )
    if "signed_application" not in kinds:
        run.missing.append(
            MissingItem(item="Signed application", kind="document", detail="Not in the file.", policy_citation=policy.cite("3", 5))
        )
    elif policy.signature_max_age_days is not None:
        age = (review_date - signed).days
        if age > policy.signature_max_age_days:
            citation = policy.cite("3", 5)
            detail = f"Signed {signed} is {age} days before the {review_date} review; limit is {policy.signature_max_age_days} days."
            run.missing.append(
                MissingItem(item="Signed application (re-signature)", kind="stale_document", detail=detail, policy_citation=citation)
            )
            run.log("SIGNATURE_STALE", "warning", detail, field="signed_date", policy_citation=citation)


_REQUIRED_CLAIMS = {
    "customer_id": "Customer ID",
    "business_name": "Business name",
    "ein": "EIN",
    "owner_name": "Owner name",
    "business_start_date": "Business start date",
    "industry": "Industry",
    "requested_amount": "Requested amount",
    "term_months": "Term",
    "secured": "Secured or unsecured",
    "claimed_existing_monthly_debt": "Existing monthly debt payments",
    "signed_date": "Signature date",
    "submitted_date": "Submission date",
}


def _cross_check(
    claimed: ClaimedData,
    verified: VerifiedData,
    policy: PolicyVersion | None,
    form_name: str,
    run: _Run,
) -> None:
    citation = policy.cite("4") if policy else None
    record = verified.customer_record

    eins = {}
    if claimed.ein:
        eins["application"] = claimed.ein.value
    if record and record.ein_on_file:
        eins["customer_record"] = record.ein_on_file
    for item in verified.tax_return_eins:
        eins[f"{item.tax_year}_tax_return"] = item.ein
    if len(set(eins.values())) > 1:
        run.log(
            "EIN_MISMATCH", "critical", "The EIN differs across the file.",
            field="ein", values=eins, documents=(form_name,), policy_citation=citation,
        )

    if record:
        for label, stated, on_file in (
            ("business_name", claimed.business_name, record.business),
            ("owner_name", claimed.owner_name, record.owner),
        ):
            if stated and on_file and not _same(stated.value, on_file):
                run.log(
                    f"{label.upper()}_MISMATCH", "critical", f"{label} differs from the customer record.",
                    field=label, values={"application": stated.value, "customer_record": on_file},
                    documents=(form_name,), policy_citation=citation,
                )

    income = claimed.claimed_income
    noi = verified.verified_noi
    if income and noi and abs(income.value - noi.amount) > INCOME_TOLERANCE * noi.amount:
        run.log(
            "INCOME_MISMATCH", "critical", "Stated income differs from the filed tax return.",
            field="claimed_income",
            values={"claimed_income": str(income.value), f"verified_noi_{noi.tax_year}": str(noi.amount)},
            documents=(form_name, noi.source.document), policy_citation=citation,
        )
    if income and not noi:
        run.log(
            "CLAIMED_INCOME_UNVERIFIED", "warning",
            "Stated income has no filed tax return behind it and cannot be used for cash-flow tests.",
            field="claimed_income", values={"claimed_income": str(income.value)}, documents=(form_name,),
            policy_citation=policy.cite("2") if policy else None,
        )

    if claimed.secured and claimed.secured.value and claimed.collateral_value:
        run.log(
            "COLLATERAL_VALUE_UNVERIFIED", "info", "The appraisal value is stated on the form; no appraisal is in the file.",
            field="collateral_value", values={"stated": str(claimed.collateral_value.value)}, documents=(form_name,),
        )

    if claimed.signed_date and claimed.submitted_date and claimed.signed_date.value > claimed.submitted_date.value:
        run.log("SIGNED_AFTER_SUBMISSION", "warning", "The signature date is after the submission date.", field="signed_date", documents=(form_name,))


def _cross_check_related(
    claimed: ClaimedData,
    verified: VerifiedData,
    this_id: str | None,
    related_forms: Sequence[tuple[str, DocExtraction]],
    policy: PolicyVersion | None,
    run: _Run,
) -> list[RelatedApplication]:
    record = verified.customer_record
    if record is None:
        return []
    citation = policy.cite("4") if policy else None
    intake = {f.application_id: (name, f) for name, f in related_forms if f.customer_id == record.customer_id and f.application_id != this_id}
    on_record = {a for a in record.open_applications if a != this_id}
    related: list[RelatedApplication] = []

    for app_id in sorted(on_record | set(intake)):
        name, form = intake.get(app_id, (None, None))
        figures = sorted((f for f in form.tax_figures if f.net_operating_income is not None), key=lambda f: f.tax_year, reverse=True) if form else []
        latest: TaxFigure | None = figures[0] if figures else None
        related.append(
            RelatedApplication(
                application_id=app_id,
                on_customer_record=app_id in on_record,
                found_in_intake=form is not None,
                requested_amount=form.requested_amount if form else None,
                term_months=form.term_months if form else None,
                stated_income=form.stated_income if form else None,
                tax_return_noi=latest.net_operating_income if latest else None,
                tax_return_year=latest.tax_year if latest else None,
            )
        )
        if form is None:
            run.log("RELATED_APPLICATION_NOT_FOUND", "warning", f"{app_id} is open on the customer record but its file was not found.", policy_citation=citation)
            continue
        if app_id not in on_record:
            run.log("OPEN_APPLICATION_NOT_ON_RECORD", "warning", f"{app_id} is in intake for this customer but not in the customer record's open applications.", documents=(name,), policy_citation=citation)

        for label, mine, theirs in (
            ("ein", claimed.ein.value if claimed.ein else None, form.ein),
            ("owner_name", claimed.owner_name.value if claimed.owner_name else None, form.owner_name),
            ("business_name", claimed.business_name.value if claimed.business_name else None, form.business_name),
        ):
            if mine and theirs and not _same(mine, theirs):
                run.log("CROSS_APPLICATION_CONFLICT", "critical", f"{label} differs from {app_id}.", field=label, values={"this_application": mine, app_id: theirs}, documents=(name,), policy_citation=citation)

        my_income = claimed.claimed_income.value if claimed.claimed_income else None
        my_noi = verified.verified_noi.amount if verified.verified_noi else None
        pairs = [(my_income, latest.net_operating_income if latest else None, "stated income here vs. tax return on"), (my_noi, form.stated_income, "tax return here vs. stated income on")]
        for a, b, description in pairs:
            if a is not None and b is not None and abs(a - b) > INCOME_TOLERANCE * min(a, b):
                run.log(
                    "CROSS_APPLICATION_INCOME_CONFLICT", "critical", f"Income conflict: {description} {app_id}.",
                    field="income", values={"this_application": str(a), app_id: str(b)}, documents=(name,), policy_citation=citation,
                )
    return related


def run_agent1(
    documents: Sequence[SourceDocument],
    customers: Sequence[dict],
    review_date: date,
    llm: DocumentParser | None = None,
    related_documents: Sequence[SourceDocument] = (),
) -> ApplicationPayload:
    run = _Run()
    parsed: list[tuple[SourceDocument, DocExtraction, str]] = []
    records: list[SourceDocumentRecord] = []
    notes: list[UntrustedNote] = []

    for doc in documents:
        extraction, method = _extract_document(doc, llm, run)
        records.append(
            SourceDocumentRecord(name=doc.name, sha256=doc.sha256, document_type=extraction.document_type if extraction else None, method=method)
        )
        if extraction is None:
            continue
        parsed.append((doc, extraction, method))
        for line in extraction.unparsed_lines:
            run.log("UNPARSED_LINE", "warning", f"Could not interpret: {line}", documents=(doc.name,))
        for text in extraction.applicant_notes:
            notes.append(UntrustedNote(text=text, document=doc.name, suspected_injection=looks_like_injection(text)))
        for text in extraction.embedded_instructions_found:
            notes.append(UntrustedNote(text=text, document=doc.name, suspected_injection=True))
        if doc.text and looks_like_injection(doc.text) and not any(n.suspected_injection and n.document == doc.name for n in notes):
            notes.append(UntrustedNote(text=doc.text, document=doc.name, suspected_injection=True))

    forms = [p for p in parsed if p[1].document_type == "application_form"]
    form_doc, form, form_method = forms[0] if forms else (None, None, None)
    form_name = form_doc.name if form_doc else ""
    if form is None:
        run.log("NO_APPLICATION_FORM", "critical", "No application form could be read from the submitted files.")
    for extra_doc, _, _ in forms[1:]:
        run.log("MULTIPLE_APPLICATION_FORMS", "warning", f"Only the first application form was used; {extra_doc.name} was not.", documents=(extra_doc.name,))

    claimed = _build_claimed(form, form_name, form_method, run)

    policy = select_policy(claimed.signed_date.value) if claimed.signed_date else None
    if policy is None:
        run.log("POLICY_NOT_DETERMINED", "critical", "No policy version can be selected without a signature date on or after the first effective date.", field="signed_date")
    policy_citation_4 = policy.cite("4") if policy else None

    customer_id = claimed.customer_id.value if claimed.customer_id else None
    raw_record = next((c for c in customers if c.get("customer_id") == customer_id), None)
    customer_record = None
    if raw_record:
        customer_record = CustomerRecord(
            customer_id=raw_record["customer_id"], business=raw_record.get("business"), owner=raw_record.get("owner"),
            ein_on_file=raw_record.get("ein_on_file"), open_applications=tuple(raw_record.get("open_applications", [])),
        )
    elif form is not None:
        run.log("CUSTOMER_NOT_FOUND", "critical", "The applicant's customer ID is not in customers.json.", field="customer_id", values={"customer_id": str(customer_id)})

    bureau = read_bureau_snapshot(raw_record)
    if raw_record:
        run.log("BUREAU_READ_FROM_FILE", "info", "Credit score read from customers.json. No external credit pull was made.", field="credit_score")

    receipts = _collect_receipts(parsed)
    noi_history, eins = _verify_tax_figures(parsed, receipts, run)
    verified = VerifiedData(
        verified_noi=noi_history[0] if noi_history else None,
        noi_history=tuple(noi_history),
        tax_return_eins=tuple(eins),
        customer_record=customer_record,
        bureau=bureau,
        documents_received=tuple(receipts),
    )

    if form is not None:
        for attr, label in _REQUIRED_CLAIMS.items():
            if getattr(claimed, attr) is None:
                run.missing.append(MissingItem(item=label, kind="field", detail="Not stated on the application form."))
    if policy and claimed.signed_date:
        _find_missing_documents(policy, claimed, receipts, review_date, run)
    if verified.verified_noi is None:
        run.missing.append(
            MissingItem(
                item="Verified net operating income", kind="field",
                detail="No net operating income from a filed tax return is in this file.",
                policy_citation=policy.cite("2") if policy else None,
            )
        )
    if bureau.credit_score is None:
        run.missing.append(MissingItem(item="Credit score", kind="field", detail="No score in customers.json. A credit pull will not be initiated."))

    for note in notes:
        if note.suspected_injection:
            run.log(
                "SUSPECTED_PROMPT_INJECTION", "critical",
                "Applicant-supplied text appears to address a reviewer or AI. It was treated as data and not followed.",
                values={"text": note.text}, documents=(note.document,), policy_citation=policy_citation_4,
            )
        elif _DOCS_BY_REFERENCE.search(note.text):
            run.log(
                "DOCUMENTS_BY_REFERENCE", "warning",
                "The applicant says documents were provided elsewhere. Documents from another application were not carried over.",
                values={"text": note.text}, documents=(note.document,),
            )

    if form is not None:
        _cross_check(claimed, verified, policy, form_name, run)

    application_id = form.application_id if form else None
    related_forms = [(d.name, parse_application_form(d.text)) for d in related_documents if is_application_form(d.text)]
    related = _cross_check_related(claimed, verified, application_id, related_forms, policy, run)

    return ApplicationPayload(
        application_id=application_id,
        review_date=review_date,
        policy=AppliedPolicy(policy_id=policy.policy_id, version=policy.version, effective_date=policy.effective_date, source_file=policy.source_file) if policy else None,
        claimed=claimed,
        verified=verified,
        missing=MissingData(items=tuple(run.missing)),
        related_applications=tuple(related),
        untrusted_notes=tuple(notes),
        source_documents=tuple(records),
        audit_log=tuple(run.audit),
    )
