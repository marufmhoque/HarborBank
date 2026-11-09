"""Versioned document requirements from Policy SB-100, selected by signature date."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class PolicyVersion:
    policy_id: str
    version: str
    effective_date: date
    source_file: str
    bank_statement_months: int
    one_tax_year_if_under_24_months: bool
    signature_max_age_days: int | None
    numbered_doc_list: bool

    def cite(self, section: str, item: int | None = None) -> str:
        suffix = f"({item})" if item is not None and self.numbered_doc_list else ""
        return f"{self.policy_id} v{self.version} §{section}{suffix}"


POLICY_VERSIONS: tuple[PolicyVersion, ...] = (
    PolicyVersion(
        policy_id="SB-100",
        version="2025.1",
        effective_date=date(2025, 1, 1),
        source_file="policy_SB100_2025.md",
        bank_statement_months=6,
        one_tax_year_if_under_24_months=False,
        signature_max_age_days=None,
        numbered_doc_list=False,
    ),
    PolicyVersion(
        policy_id="SB-100",
        version="2026.2",
        effective_date=date(2026, 7, 1),
        source_file="policy_SB100_2026.md",
        bank_statement_months=6,
        one_tax_year_if_under_24_months=True,
        signature_max_age_days=30,
        numbered_doc_list=True,
    ),
)


def select_policy(signed: date) -> PolicyVersion | None:
    """Return the latest version effective on the signature date, if any."""
    applicable = [p for p in POLICY_VERSIONS if p.effective_date <= signed]
    return max(applicable, key=lambda p: p.effective_date, default=None)
