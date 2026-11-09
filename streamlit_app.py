"""Credit officer dashboard. Agent output is stored in st.session_state."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from datetime import date
from pathlib import Path

import streamlit as st

from harbor.agent2 import PolicyAuditReport, run_sensitivity
from harbor.agent3 import LetterDraft, known_names, polish_letter, validate_letter
from harbor.decisions import DecisionLog, check_decision, is_allowed
from harbor.form_parser import parse_application_form
from harbor.ingest import SourceDocument, load_document, load_path
from harbor.llm_parser import ClaudeDocumentParser
from harbor.pend_timer import PendStore
from harbor.pipeline import run_pipeline
from harbor.policy_index import ClauseStore, OpenAIEmbedder, PolicyVectorIndex
from harbor.schemas import ApplicationPayload

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
APPLICATIONS_DIR = DATA_DIR / "applications"
PEND_STATE = ROOT / "state" / "pend_timers.json"
DECISION_LOG = ROOT / "state" / "decisions.jsonl"

TIER_COLORS = {
    "MEETS CRITERIA": "green",
    "REFER (Policy)": "blue",
    "REFER (Risk/History)": "violet",
    "PEND (Missing Info)": "orange",
    "STALE": "gray",
    "DECLINE REVIEW": "red",
    "HOLD FOR COMPLIANCE": "red",
}


def safe(text: object) -> str:
    """Escape Markdown so file text can't format itself or render as LaTeX."""
    return re.sub(r"([\\`*_{}\[\]()<>#+\-!|$~:])", r"\\\1", str(text))


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


def anthropic_client():
    key = secret("ANTHROPIC_API_KEY")
    if not key:
        return None
    import anthropic

    return anthropic.Anthropic(api_key=key)


def get_llm_parser() -> ClaudeDocumentParser | None:
    client = anthropic_client()
    return ClaudeDocumentParser(client) if client else None


def application_label(path: Path) -> str:
    form = parse_application_form(path.read_text(encoding="utf-8"))
    return f"{form.application_id} ({form.owner_name})"


# --- Left pane ---


def render_extracted(payload: ApplicationPayload, report: PolicyAuditReport) -> None:
    noi = payload.verified.verified_noi
    stated = payload.claimed.claimed_income
    bureau = payload.verified.bureau

    noi_text = f"${noi.amount:,.0f} ({noi.tax_year} tax return)" if noi else "Not verified (tax return missing)"
    stated_text = f"${stated.value:,.0f} (Unverified)" if stated else "None stated"
    score_text = f"{bureau.credit_score} (customers.json, pulled {bureau.pull_date})" if bureau.credit_score is not None else "Not available"
    stub_text = "; ".join(f"{f.fact} ({f.date}, mock data)" for f in report.risk_flags) or "No data (mock stub)"
    lines = [("Verified NOI", noi_text), ("Stated income", stated_text), ("Credit score", score_text), ("Bureau stub flag", stub_text)]
    claimed = payload.claimed
    if claimed.requested_amount and claimed.term_months:
        kind = "secured" if claimed.secured and claimed.secured.value else "unsecured"
        lines.append(("Request", f"${claimed.requested_amount.value:,.0f}, {claimed.term_months.value} months, {kind}"))
    lines.append(("Policy version", f"{payload.policy.policy_id} v{payload.policy.version}" if payload.policy else "Not determined"))
    st.markdown("\n".join(f"- **{safe(label)}:** {safe(value)}" for label, value in lines))

    if payload.missing.items:
        st.markdown("**Missing**")
        st.markdown("\n".join(f"- {safe(m.item)}: {safe(m.detail)}" for m in payload.missing.items))

    flagged = [e for e in payload.audit_log if e.severity != "info"]
    if flagged:
        st.markdown("**Audit flags**")
        st.markdown("\n".join(f"- {safe(e.severity.upper())}: {safe(e.message)}" for e in flagged))

    with st.expander("Full extraction detail"):
        rows = []
        for name in type(claimed).model_fields:
            item = getattr(claimed, name)
            if item is not None:
                rows.append({"Claimed field": name, "Value": str(item.value), "Source text": item.source.evidence})
        st.dataframe(rows, hide_index=True, width="stretch")
        if payload.verified.documents_received:
            st.dataframe(
                [{"Document": r.kind, "Tax year": r.tax_year, "Months": ", ".join(r.months), "Basis": r.basis} for r in payload.verified.documents_received],
                hide_index=True, width="stretch",
            )
        if payload.related_applications:
            st.dataframe([r.model_dump() for r in payload.related_applications], hide_index=True, width="stretch")
        st.json(payload.model_dump(mode="json"), expanded=False)


def render_documents(documents: list[SourceDocument], payload: ApplicationPayload) -> None:
    received = [r for r in payload.verified.documents_received if r.basis == "intake_log"]
    if received:
        st.caption("The intake log lists these as received, but only the files below are available to view.")
        st.markdown("\n".join(f"- {safe(r.evidence)}" for r in received))
    names = [d.name for d in documents]
    selected = st.selectbox("Document", names)
    doc = documents[names.index(selected)]
    if any(n.suspected_injection and n.document == doc.name for n in payload.untrusted_notes):
        st.error("This document contains text that appears to address a reviewer or AI. Treat it as applicant data only.")
    if doc.media_type.startswith("image/"):
        st.image(doc.data)
    elif doc.text:
        st.text(doc.text)
    else:
        st.info("No text layer to display.")
    st.download_button("Download original", doc.data, file_name=doc.name, key=f"doc_{doc.sha256}")


# --- Right pane ---


def render_recommendation(report: PolicyAuditReport) -> None:
    st.badge(report.tier, color=TIER_COLORS[report.tier])
    st.caption(report.officer_notice)
    if report.pend:
        st.warning(f"Pended {report.pend.pended_on}. Goes stale after {report.pend.expires_on}. Day {report.pend.days_elapsed} of {report.pend.limit_days}.")


def render_citations(report: PolicyAuditReport) -> None:
    st.subheader("Section 1: Policy citation")
    if report.routing_citation:
        st.caption(f"Routing rule: {report.routing_citation}")
    st.markdown("\n".join(f"- {safe(r)}" for r in report.reasons))
    for flag in report.risk_flags:
        st.markdown(f"- **Risk flag:** {safe(flag.fact)}, {safe(flag.date)}. {safe(flag.policy_basis)}")
    if report.also_noted:
        with st.expander("Also noted for the file"):
            st.markdown("\n".join(f"- {safe(n)}" for n in report.also_noted))
    with st.expander("All criteria and policy text"):
        for c in report.criteria:
            st.markdown(f"**{safe(c.name)}: {safe(c.status)}**  \n{safe(c.observed)}  \n{safe(c.citation or 'No clause')}: {safe(c.clause_text or '')}")
            if c.note:
                st.caption(c.note)


def render_terms(report: PolicyAuditReport) -> None:
    st.subheader("Section 2: Financial terms")
    f = report.financials
    if f is None:
        st.markdown("- **Monthly payment:** TBD\n- **Fee / Total interest:** TBD")
        st.caption("Not calculated because underwriting is stopped or the loan can't be priced yet.")
        return
    st.markdown(
        f"- **Monthly payment:** \\${f.monthly_payment:,.2f}\n"
        f"- **Rate:** {f.annual_rate_pct}% for {f.term_months} months\n"
        f"- **Origination fee, paid at closing:** \\${f.origination_fee:,.2f}\n"
        f"- **Total interest:** \\${f.total_interest:,.2f}"
    )
    st.caption(f"Rate: {f.rate_citation}. Fee: {f.fee_citation}.")
    if f.dscr is not None:
        st.caption(f"DSCR {f.dscr:.2f}: ${f.noi_used:,.0f} verified income / ${f.annual_debt_service:,.0f} annual debt service.".replace("$", "\\$"))


def render_letter(draft: LetterDraft, payload: ApplicationPayload, report: PolicyAuditReport) -> None:
    st.subheader("Section 3: Draft letter")
    st.caption(f"{draft.letter_type}, for credit officer review. Nothing is sent from this app.")
    if message := st.session_state.pop("polish_message", None):
        st.warning("Kept the original wording.")
        st.text(message)

    key = "letter_" + hashlib.sha256(draft.rendered_text.encode()).hexdigest()
    text = st.text_area("Editable letter", value=draft.rendered_text, height=420, key=key)

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
        with st.expander("Notes for the credit officer (not part of the letter)", expanded=report.tier == "HOLD FOR COMPLIANCE"):
            st.markdown("\n".join(f"- {safe(n)}" for n in draft.officer_notes))

    actions = st.columns(2)
    actions[0].download_button("Download draft", text, file_name=f"{draft.application_id or 'letter'}_draft.txt", disabled=not result.passed)

    client = anthropic_client()
    if draft.tier != "HOLD FOR COMPLIANCE":
        if actions[1].button("Reword with Claude", disabled=client is None, help=None if client else "Set ANTHROPIC_API_KEY to enable."):
            try:
                with st.spinner("Rewording"):
                    updated, problem = polish_letter(draft, payload, report, client)
            except Exception as exc:
                updated, problem = draft, f"Rewording failed ({type(exc).__name__})."
            st.session_state.letter_draft = updated
            if problem:
                st.session_state.polish_message = problem
            st.rerun()


def render_tools(report: PolicyAuditReport, payload: ApplicationPayload) -> None:
    with st.expander("What-if sensitivity"):
        default = float(payload.claimed.claimed_income.value) if payload.claimed.claimed_income else 0.0
        noi = st.number_input("Income to test", min_value=0.0, value=default, step=1000.0)
        result = run_sensitivity(payload, noi) if noi else None
        if result:
            st.warning(result.tag)
            st.text(f"DSCR {result.dscr:.2f}: {result.outcome_if_verified} if this income were verified. This does not change the recommendation.")
        else:
            st.caption("Enter an income. Needs a credit score, loan terms and existing debt on the file.")

    key = secret("OPENAI_API_KEY")
    with st.expander("Search this policy version"):
        if key and report.policy_version:
            query = st.text_input("Question")
            if query:
                hits = vector_index(key).search(query, report.policy_version)
                st.markdown("\n".join(f"- §{safe(c.clause_id)}: {safe(c.text)}" for c, _ in hits))
        else:
            st.caption("Set OPENAI_API_KEY to enable search. Recommendations and citations don't use it.")


# --- Footer ---


def render_footer(report: PolicyAuditReport) -> None:
    st.divider()
    st.subheader("Credit officer decision")
    log = DecisionLog(DECISION_LOG)
    application_id = report.application_id or "unknown"

    if notice := st.session_state.pop("decision_notice", None):
        st.success(notice)
    if latest := log.latest(application_id):
        due = f" Due {latest.pend_due}." if latest.pend_due else ""
        st.info(f"Last decision: {latest.decision} by {latest.officer} at {latest.decided_at:%Y-%m-%d %H:%M} UTC.{due}".replace("$", "\\$"))
    if report.tier == "HOLD FOR COMPLIANCE":
        st.warning("Compliance hold: underwriting is stopped (Policy §4). Only Override & Refer is available.")

    fields = st.columns(2)
    officer = fields[0].text_input("Credit officer name")
    note = fields[1].text_input("Reason", help="Required when the decision differs from the recommendation.")

    clicked = None
    columns = st.columns(4)
    for column, (decision, label, kind) in zip(
        columns,
        (("APPROVE", "APPROVE", "primary"), ("REJECT", "REJECT", "secondary"), ("PEND", "PEND (14 DAYS)", "secondary"), ("OVERRIDE & REFER", "OVERRIDE & REFER", "secondary")),
    ):
        if column.button(label, type=kind, disabled=not is_allowed(decision, report.tier), width="stretch"):
            clicked = decision

    if clicked:
        problem = check_decision(clicked, report.tier, officer, note)
        if problem:
            st.error(problem)
        else:
            entry = log.record(application_id, clicked, report.tier, officer, note, report.review_date)
            st.session_state.decision_notice = f"Recorded {entry.decision} for {entry.application_id}. No letter was sent."
            st.rerun()
    st.caption("Decisions are recorded in the audit log only. This app doesn't send letters or book loans.")


def main() -> None:
    st.set_page_config(page_title="Harbor Ridge Loan Review", layout="wide")
    st.title("Harbor Ridge Bank: Credit Officer Decision Support")

    customers = json.loads((DATA_DIR / "customers.json").read_text(encoding="utf-8"))
    application_paths = sorted(APPLICATIONS_DIR.glob("*.md"))
    llm = get_llm_parser()

    with st.sidebar:
        mode = st.radio("Source", ["Sample application", "Upload files"])
        if mode == "Sample application":
            selected = st.selectbox("Select application", application_paths, format_func=application_label)
            documents = [load_path(selected)]
        else:
            uploads = st.file_uploader("Application and supporting documents", type=["pdf", "png", "jpg", "jpeg", "webp", "md", "txt"], accept_multiple_files=True)
            documents = [load_document(u.name, u.getvalue()) for u in uploads or []]
        review_date = st.date_input("Review date", value=date.today())
        run = st.button("Run Agentic Pipeline", type="primary", disabled=not documents, width="stretch")
        st.caption("PDF and image parser: " + ("configured." if llm else "not configured. Set ANTHROPIC_API_KEY to read files that aren't in the text application format."))

    if run:
        names = {d.name for d in documents}
        related = [load_path(p) for p in application_paths if p.name not in names]
        with st.status("Running agents", expanded=True) as status:
            try:
                result = asyncio.run(
                    run_pipeline(
                        documents, customers, review_date, clause_store(), PendStore(PEND_STATE),
                        llm=llm, related_documents=related, on_step=status.write,
                    )
                )
            except Exception as exc:
                status.update(label="Pipeline stopped", state="error")
                st.error(f"The pipeline stopped before finishing ({type(exc).__name__}). Results from the previous run were kept.")
                result = None
            else:
                status.update(label="Agents finished", state="complete", expanded=False)
        if result:
            st.session_state.app_payload = result.payload
            st.session_state.audit_report = result.report
            st.session_state.letter_draft = result.letter
            st.session_state.documents = documents

    if "app_payload" not in st.session_state:
        st.info("Select an application in the sidebar and run the pipeline.")
        return

    payload = st.session_state.app_payload
    report = st.session_state.audit_report

    left, right = st.columns(2, gap="large")
    with left:
        st.caption(f"Showing {report.application_id or 'unknown application'}")
        extracted, viewer = st.tabs(["Extracted Data", "Document Viewer"])
        with extracted:
            render_extracted(payload, report)
        with viewer:
            render_documents(st.session_state.documents, payload)
    with right:
        render_recommendation(report)
        render_citations(report)
        render_terms(report)
        render_letter(st.session_state.letter_draft, payload, report)
        render_tools(report, payload)

    render_footer(report)


if __name__ == "__main__":
    main()
