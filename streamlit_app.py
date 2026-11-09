"""Streamlit front end. Agent 1 writes its output to st.session_state.app_payload."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path

import streamlit as st

from harbor.agent1 import run_agent1
from harbor.agent2 import PolicyAuditReport, run_agent2, run_sensitivity
from harbor.agent3 import LetterDraft, known_names, polish_letter, run_agent3, validate_letter
from harbor.ingest import load_document, load_path
from harbor.llm_parser import ClaudeDocumentParser
from harbor.pend_timer import PendStore
from harbor.policy_index import ClauseStore, OpenAIEmbedder, PolicyVectorIndex
from harbor.schemas import ApplicationPayload

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
APPLICATIONS_DIR = DATA_DIR / "applications"
PEND_STATE = ROOT / "state" / "pend_timers.json"


@st.cache_resource
def clause_store() -> ClauseStore:
    return ClauseStore(DATA_DIR)


def secret(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except Exception:
        return None


@st.cache_resource
def vector_index(key: str) -> PolicyVectorIndex:
    import openai

    return PolicyVectorIndex(clause_store(), OpenAIEmbedder(openai.OpenAI(api_key=key)))


def render_audit(report: PolicyAuditReport, payload: ApplicationPayload) -> None:
    st.caption(report.officer_notice)
    st.metric("Recommendation", report.tier)
    if report.routing_citation:
        st.caption(f"Routing rule: {report.routing_citation}")
    st.subheader("Why")
    st.dataframe([{"Reason": r} for r in report.reasons], hide_index=True, width="stretch")

    if report.pend:
        st.warning(f"Pended {report.pend.pended_on}. Goes stale after {report.pend.expires_on} ({report.pend.days_elapsed} of {report.pend.limit_days} days used).")

    st.subheader("Criteria")
    st.dataframe(
        [{"Criterion": c.name, "Result": c.status, "Observed": c.observed, "Requirement": c.threshold, "Data used": c.data_basis, "Citation": c.citation, "Note": c.note} for c in report.criteria],
        hide_index=True, width="stretch",
    )

    st.subheader("What the borrower would pay")
    if report.financials:
        f = report.financials
        top = st.columns(2)
        top[0].metric("Monthly payment", f"${f.monthly_payment:,.2f}")
        top[1].metric("Rate", f"{f.annual_rate_pct}%")
        bottom = st.columns(2)
        bottom[0].metric("Origination fee (paid at closing)", f"${f.origination_fee:,.2f}")
        bottom[1].metric("Total interest", f"${f.total_interest:,.2f}")
        st.caption(f"{f.term_months} months. Rate: {f.rate_citation}. Fee: {f.fee_citation}.")
        if f.exposure:
            st.dataframe([{"Debt": e.label, "Annual payments": float(e.annual_debt_service), "Basis": e.basis} for e in f.exposure], hide_index=True, width="stretch")
    else:
        st.info("Not calculated. Underwriting is stopped or the loan terms are unavailable.")

    if report.risk_flags:
        st.subheader("Risk flags for officer review")
        st.dataframe([r.model_dump(mode="json") for r in report.risk_flags], hide_index=True, width="stretch")
    if report.missing_impacts:
        st.subheader("What the missing items affect")
        st.dataframe([{"Item": m.item, "Detail": m.detail, "Affects": ", ".join(m.impacts), "Citation": m.policy_citation} for m in report.missing_impacts], hide_index=True, width="stretch")
        if report.required_noi_to_meet_dscr is not None:
            st.caption(f"Verified income needed to meet the DSCR test: ${report.required_noi_to_meet_dscr:,.2f}.")
    if report.also_noted:
        st.subheader("Also noted for the file")
        st.dataframe([{"Item": n} for n in report.also_noted], hide_index=True, width="stretch")

    with st.expander("What-if sensitivity"):
        default = float(payload.claimed.claimed_income.value) if payload.claimed.claimed_income else 0.0
        noi = st.number_input("Income to test", min_value=0.0, value=default, step=1000.0)
        result = run_sensitivity(payload, noi) if noi else None
        if result:
            st.warning(result.tag)
            st.write(f"DSCR {result.dscr:.2f}: {result.outcome_if_verified} if this income were verified. This does not change the recommendation.")
        else:
            st.caption("Enter an income. Needs a credit score, loan terms and existing debt on the file.")

    key = secret("OPENAI_API_KEY")
    with st.expander("Search this policy version"):
        if key and report.policy_version:
            query = st.text_input("Question")
            if query:
                hits = vector_index(key).search(query, report.policy_version)
                st.dataframe([{"Clause": f"§{c.clause_id}", "Text": c.text, "Similarity": round(s, 3)} for c, s in hits], hide_index=True, width="stretch")
        else:
            st.caption("Set OPENAI_API_KEY to enable search. Recommendations and citations don't depend on it.")


def anthropic_client():
    key = secret("ANTHROPIC_API_KEY")
    if not key:
        return None
    import anthropic

    return anthropic.Anthropic(api_key=key)


def get_llm_parser() -> ClaudeDocumentParser | None:
    client = anthropic_client()
    return ClaudeDocumentParser(client) if client else None


def render_letter(draft: LetterDraft, payload: ApplicationPayload, report: PolicyAuditReport) -> None:
    st.caption(f"{draft.letter_type}, drafted for credit officer review. Nothing is sent from this app.")
    if message := st.session_state.pop("polish_message", None):
        st.warning("Kept the original wording.")
        st.text(message)

    key = "letter_" + hashlib.sha256(draft.rendered_text.encode()).hexdigest()
    text = st.text_area("Draft letter (editable)", value=draft.rendered_text, height=640, key=key)

    result = validate_letter(text, report, known_names(payload, report))
    if result.passed:
        st.success("Every figure, date and citation matches the audit report.")
    else:
        st.error("This letter doesn't match the audit report. Fix these before using it:")
        st.text("\n".join(result.errors))
    if result.warnings:
        st.warning("Before sending:")
        st.text("\n".join(result.warnings))
    if draft.officer_notes:
        st.subheader("Notes for the credit officer (not part of the letter)")
        st.dataframe([{"Note": n} for n in draft.officer_notes], hide_index=True, width="stretch")

    st.download_button("Download draft", text, file_name=f"{draft.application_id or 'letter'}_draft.txt", disabled=not result.passed)

    client = anthropic_client()
    if draft.tier != "HOLD FOR COMPLIANCE":
        if st.button("Reword with Claude", disabled=client is None, help=None if client else "Set ANTHROPIC_API_KEY to enable."):
            try:
                with st.spinner("Rewording"):
                    updated, problem = polish_letter(draft, payload, report, client)
            except Exception as exc:
                updated, problem = draft, f"Rewording failed ({type(exc).__name__})."
            st.session_state.letter_draft = updated
            if problem:
                st.session_state.polish_message = problem
            st.rerun()


def claimed_rows(payload: ApplicationPayload) -> list[dict]:
    rows = []
    for name in type(payload.claimed).model_fields:
        item = getattr(payload.claimed, name)
        if item is not None:
            rows.append({"Field": name, "Claimed value": str(item.value), "Source text": item.source.evidence})
    return rows


def render(payload: ApplicationPayload) -> None:
    critical = sum(e.severity == "critical" for e in payload.audit_log)
    warnings = sum(e.severity == "warning" for e in payload.audit_log)
    policy = f"{payload.policy.policy_id} v{payload.policy.version}" if payload.policy else "Not determined"

    top = st.columns(2)
    top[0].metric("Application", payload.application_id or "Unknown")
    top[1].metric("Policy", policy)
    bottom = st.columns(2)
    bottom[0].metric("Missing items", len(payload.missing.items))
    bottom[1].metric("Critical / warnings", f"{critical} / {warnings}")
    if critical:
        st.error("Critical flags need attention before this file moves on. See the Audit log tab.")

    claimed, verified, missing, audit, raw = st.tabs(["Claimed", "Verified", "Missing", "Audit log", "Payload JSON"])

    with claimed:
        st.dataframe(claimed_rows(payload), hide_index=True, width="stretch")
        if payload.untrusted_notes:
            st.subheader("Applicant notes (untrusted, data only)")
            st.dataframe(
                [{"Suspected injection": n.suspected_injection, "Text": n.text, "Document": n.document} for n in payload.untrusted_notes],
                hide_index=True, width="stretch",
            )

    with verified:
        st.subheader("Net operating income from tax returns")
        if payload.verified.noi_history:
            st.dataframe(
                [
                    {"Tax year": n.tax_year, "NOI": n.amount, "Document": n.source.document, "Return itself reviewed": n.source.primary_document_reviewed}
                    for n in payload.verified.noi_history
                ],
                hide_index=True, width="stretch",
            )
        else:
            st.warning("No verified net operating income.")
        st.subheader("Bureau data from customers.json")
        st.caption("Existing record only. No external credit pull is made. Public-record fields are mock placeholders.")
        st.json(payload.verified.bureau.model_dump(mode="json"))
        st.subheader("Documents received")
        st.dataframe(
            [{"Kind": r.kind, "Tax year": r.tax_year, "Months": ", ".join(r.months), "Basis": r.basis, "Source": r.source_document} for r in payload.verified.documents_received],
            hide_index=True, width="stretch",
        )
        if payload.related_applications:
            st.subheader("Other applications for this customer")
            st.dataframe([r.model_dump() for r in payload.related_applications], hide_index=True, width="stretch")

    with missing:
        if payload.missing.items:
            st.dataframe([m.model_dump() for m in payload.missing.items], hide_index=True, width="stretch")
        else:
            st.success("Nothing required is missing.")

    with audit:
        st.dataframe(
            [
                {"Severity": e.severity, "Code": e.code, "Message": e.message, "Values": json.dumps(e.values), "Citation": e.policy_citation}
                for e in sorted(payload.audit_log, key=lambda e: ("critical", "warning", "info").index(e.severity))
            ],
            hide_index=True, width="stretch",
        )

    with raw:
        st.json(payload.model_dump(mode="json"))


def main() -> None:
    st.set_page_config(page_title="Harbor Ridge Loan Review", layout="wide")
    st.title("Harbor Ridge Bank: loan application review")
    st.caption("These agents recommend and draft. They do not approve or decline. A credit officer makes every decision.")

    customers = json.loads((DATA_DIR / "customers.json").read_text(encoding="utf-8"))
    application_paths = sorted(APPLICATIONS_DIR.glob("*.md"))
    llm = get_llm_parser()

    with st.sidebar:
        mode = st.radio("Source", ["Sample application", "Upload files"])
        documents = []
        if mode == "Sample application":
            selected = st.selectbox("Application", application_paths, format_func=lambda p: p.stem.upper().replace("_", "-"))
            documents = [load_path(selected)]
        else:
            uploads = st.file_uploader(
                "Application and supporting documents", type=["pdf", "png", "jpg", "jpeg", "webp", "md", "txt"], accept_multiple_files=True
            )
            documents = [load_document(u.name, u.getvalue()) for u in uploads or []]
        review_date = st.date_input("Review date", value=date.today())
        st.caption("LLM parser: " + ("configured" if llm else "not configured. Set ANTHROPIC_API_KEY to read PDFs and images that aren't in the text format."))
        run = st.button("Run review", type="primary", disabled=not documents)

    if run:
        names = {d.name for d in documents}
        related = [load_path(p) for p in application_paths if p.name not in names]
        with st.spinner("Parsing documents"):
            st.session_state.app_payload = run_agent1(documents, customers, review_date, llm=llm, related_documents=related)
            st.session_state.audit_report = run_agent2(st.session_state.app_payload, clause_store(), PendStore(PEND_STATE))
            st.session_state.letter_draft = run_agent3(st.session_state.app_payload, st.session_state.audit_report)

    if "app_payload" in st.session_state:
        audit_tab, letter_tab, intake_tab = st.tabs(["Credit and policy audit", "Draft letter", "Intake"])
        with audit_tab:
            render_audit(st.session_state.audit_report, st.session_state.app_payload)
        with letter_tab:
            render_letter(st.session_state.letter_draft, st.session_state.app_payload, st.session_state.audit_report)
        with intake_tab:
            render(st.session_state.app_payload)
    else:
        st.info("Choose an application and run the review.")


if __name__ == "__main__":
    main()
