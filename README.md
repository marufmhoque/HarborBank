# HarborBank

Multi-agent small-business loan review for Harbor Ridge Bank, hosted on Streamlit.
The system recommends and drafts only. A credit officer makes every decision.

## Agent 1: Document parsing and extraction

Turns application files and `customers.json` into an `ApplicationPayload`, stored in
`st.session_state.app_payload`, with three separate partitions:

- **Claimed:** what the applicant stated on the application form.
- **Verified:** tax-return figures, bank-held records and bureau data from `customers.json`.
- **Missing:** required documents and fields that are absent or stale, cited to the policy version.

Mismatches go to the audit log. Agent 1 never makes a credit pull and has no recommendation field.

| Module | Purpose |
|---|---|
| `harbor/schemas.py` | Pydantic contracts that enforce the partitions |
| `harbor/agent1.py` | Orchestration, partitioning and cross-checks |
| `harbor/form_parser.py` | Rule parser for the bank's text application format |
| `harbor/ingest.py` | File loading and `pdfplumber` text extraction |
| `harbor/llm_parser.py` | Claude parser for PDFs and images the rules can't read |
| `harbor/bureau.py` | Reads bureau data from `customers.json` only |
| `harbor/policy_registry.py` | Document rules per SB-100 version, selected by signature date |

## Agent 2: Credit and policy audit

Takes the `ApplicationPayload` and returns a `PolicyAuditReport`. It is fully deterministic.

Routing order: HOLD FOR COMPLIANCE, then DECLINE REVIEW for hard-eligibility failures, then
PEND (STALE after 14 days without a document update), then the credit and cash-flow tests
(DECLINE REVIEW, REFER (Risk/History), REFER (Policy), MEETS CRITERIA).

| Module | Purpose |
|---|---|
| `harbor/agent2.py` | Rule engine, tier router and what-if sensitivity |
| `harbor/financials.py` | Payment, interest, fee and DSCR math in `Decimal` |
| `harbor/credit_rules.py` | Thresholds per policy version, tied to clause IDs |
| `harbor/policy_index.py` | Clause lookup for citations; FAISS search per version |
| `harbor/pend_timer.py` | 14-day timer stored in `state/pend_timers.json` |

Set `OPENAI_API_KEY` to enable the optional policy search. Tiers and citations don't use it.

## Agent 3: Communication and letters

Turns the audit report into an editable draft letter for a credit officer.

Code builds every statement from report fields and Jinja2 lays out the letter. Claude can optionally
reword it, but a rewording is kept only if every figure, date and citation is unchanged. The validator
in `harbor/agent3.py` checks the draft and re-checks each officer edit against the report.

## Run

```bash
py -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
.venv/Scripts/python -m pytest
.venv/Scripts/python -m streamlit run streamlit_app.py
```

Set `ANTHROPIC_API_KEY`, as an environment variable or in Streamlit secrets, to read PDFs and
images that aren't in the text application format. Without it those files are flagged and not counted.
