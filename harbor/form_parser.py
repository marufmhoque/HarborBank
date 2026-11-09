"""Deterministic parser for the bank's text loan-application format."""

from __future__ import annotations

import re

from harbor.schemas import DocExtraction, FieldEvidence, ReceivedItem, TaxFigure

_MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()
_DATE = r"\d{4}-\d{2}-\d{2}"
_MONEY = r"\$([\d,]+(?:\.\d+)?)"


def _money(text: str) -> float:
    return float(text.replace(",", ""))


def is_application_form(text: str | None) -> bool:
    return bool(text and re.search(r"LOAN APPLICATION\s+APP-\d+", text))


def _month_range(start: str, end: str, year: int) -> list[str]:
    first = _MONTHS.index(start[:3].lower()) + 1
    last = _MONTHS.index(end[:3].lower()) + 1
    return [f"{year}-{m:02d}" for m in range(first, last + 1)]


def _parse_received(segment: str) -> tuple[list[ReceivedItem], list[str]]:
    items: list[ReceivedItem] = []
    unparsed: list[str] = []
    for raw in [s.strip(" .") for s in segment.split(";") if s.strip(" .")]:
        lowered = raw.lower()
        if "tax return" in lowered:
            years = [int(y) for y in re.findall(r"\b(20\d{2})\b", raw)]
            if not years:
                unparsed.append(raw)
            items += [ReceivedItem(kind="tax_return", tax_year=y, months=[], evidence=raw) for y in years]
        elif "bank statement" in lowered:
            match = re.search(r"([A-Za-z]+)-([A-Za-z]+) (\d{4})", raw)
            if match:
                months = _month_range(match[1], match[2], int(match[3]))
                items.append(ReceivedItem(kind="bank_statement", tax_year=None, months=months, evidence=raw))
            else:
                unparsed.append(raw)
        elif "photo id" in lowered:
            items.append(ReceivedItem(kind="photo_id", tax_year=None, months=[], evidence=raw))
        elif "business license" in lowered:
            items.append(ReceivedItem(kind="business_license", tax_year=None, months=[], evidence=raw))
        elif "signed application" in lowered:
            items.append(ReceivedItem(kind="signed_application", tax_year=None, months=[], evidence=raw))
        else:
            unparsed.append(raw)
    return items, unparsed


def parse_application_form(text: str) -> DocExtraction:
    out = DocExtraction(document_type="application_form")
    evidence: dict[str, str] = {}

    def record(line: str, **values) -> None:
        for field, value in values.items():
            setattr(out, field, value)
            evidence[field] = line

    for line in (ln.strip() for ln in text.splitlines()):
        if not line:
            continue
        if m := re.match(rf"LOAN APPLICATION\s+(APP-\d+)\s+Submitted ({_DATE})\s+Signed ({_DATE})", line):
            record(line, application_id=m[1], submitted_date=m[2], signed_date=m[3])
        elif m := re.match(r"Borrower: (.+?) \(Customer (C\d+)\)\s+EIN(?: on application)?:? ([\d-]+)", line):
            record(line, business_name=m[1], customer_id=m[2], ein=m[3])
        elif m := re.match(r"Owner: (.+?) \((\d+(?:\.\d+)?)%\)", line):
            record(line, owner_name=m[1], ownership_pct=float(m[2]))
        elif m := re.match(rf"Business start date: ({_DATE})\s+Industry: (.+)", line):
            record(line, business_start_date=m[1], industry=m[2])
        elif m := re.match(rf"Request: {_MONEY} for (\d+) months, (.+)", line):
            record(line, requested_amount=_money(m[1]), term_months=int(m[2]))
            remainder = m[3]
            if remainder.lower().startswith("unsecured"):
                record(line, secured=False)
            elif c := re.match(rf"secured by (.+?), appraised at {_MONEY}", remainder, re.I):
                record(line, secured=True, collateral_description=c[1], collateral_value=_money(c[2]))
            else:
                out.unparsed_lines.append(line)
            if p := re.search(r"Purpose: (.+?)\.?$", remainder):
                record(line, purpose=p[1])
        elif m := re.match(
            rf"From (\d{{4}}) business tax return(?: \(([^)]*)\))?: net operating income {_MONEY}"
            rf"(?: \((\d{{4}}): {_MONEY}\))?",
            line,
        ):
            ein = re.search(r"EIN ([\d-]+)", m[2] or "")
            out.tax_figures.append(
                TaxFigure(tax_year=int(m[1]), net_operating_income=_money(m[3]), ein=ein[1] if ein else None, evidence=line)
            )
            if m[4]:
                out.tax_figures.append(
                    TaxFigure(tax_year=int(m[4]), net_operating_income=_money(m[5]), ein=None, evidence=line)
                )
        elif m := re.match(r'Net operating income: "([^"]*)"', line):
            amount = re.search(_MONEY, m[1])
            if amount:
                record(line, stated_income=_money(amount[1]))
            else:
                out.unparsed_lines.append(line)
        elif m := re.match(rf"Existing business debt payments: {_MONEY} per month", line):
            record(line, existing_monthly_debt=_money(m[1]))
        elif m := re.match(r"Documents received: (.*)", line):
            segment, _, note = m[1].partition(" Note:")
            items, unparsed = _parse_received(segment)
            out.documents_received += items
            out.unparsed_lines += unparsed
            if note.strip():
                out.applicant_notes.append(note.strip().strip('"'))
        elif m := re.match(r"Not received: (.*)", line):
            out.documents_not_received += [s.strip(" .") for s in m[1].split(";") if s.strip(" .")]
        elif m := re.match(r"Applicant notes: (.*)", line):
            out.applicant_notes.append(m[1].strip().strip('"'))
        else:
            out.unparsed_lines.append(line)

    out.evidence = [FieldEvidence(field=f, text=t) for f, t in evidence.items()]
    return out
