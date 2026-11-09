"""SB-100 thresholds per version, each tied to the clause it comes from.

tests/test_agent2.py checks every value against the policy text, so this table
cannot drift from the policy documents unnoticed.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class RateBand:
    min_score: int
    max_score: int | None
    rate_pct: Decimal
    clause: str


@dataclass(frozen=True)
class CreditRules:
    version: str
    min_months: int
    min_amount: Decimal
    max_amount: Decimal
    unsecured_cap: Decimal
    excluded_industries: tuple[str, ...]
    max_term: int
    score_meets: int
    score_refer: int
    dscr_meets: Decimal
    dscr_refer: Decimal
    rate_bands: tuple[RateBand, ...]
    fee_pct: Decimal
    combined_exposure_required: bool
    clauses: dict[str, str]

    def rate_for(self, score: int) -> RateBand | None:
        for band in self.rate_bands:
            if score >= band.min_score and (band.max_score is None or score <= band.max_score):
                return band
        return None


_EXCLUDED = ("cannabis", "gambling", "adult entertainment")

CREDIT_RULES: dict[str, CreditRules] = {
    "2025.1": CreditRules(
        version="2025.1",
        min_months=18,
        min_amount=Decimal(25000),
        max_amount=Decimal(250000),
        unsecured_cap=Decimal(150000),
        excluded_industries=_EXCLUDED,
        max_term=84,
        score_meets=660,
        score_refer=620,
        dscr_meets=Decimal("1.20"),
        dscr_refer=Decimal("1.10"),
        rate_bands=(
            RateBand(700, None, Decimal("8.75"), "5.1"),
            RateBand(680, 699, Decimal("9.75"), "5.1"),
            RateBand(660, 679, Decimal("11.25"), "5.1"),
        ),
        fee_pct=Decimal("1.0"),
        combined_exposure_required=False,
        clauses={
            "time_in_business": "1.1",
            "amount": "1.2",
            "unsecured_cap": "1.3",
            "industry": "1.4",
            "term": "1.5",
            "score": "2.1",
            "dscr_meets": "2.2",
            "dscr_refer": "2.2",
            "dscr_decline": "2.2",
            "documents": "3.1",
            "fee": "5.1",
        },
    ),
    "2026.2": CreditRules(
        version="2026.2",
        min_months=24,
        min_amount=Decimal(25000),
        max_amount=Decimal(250000),
        unsecured_cap=Decimal(100000),
        excluded_industries=_EXCLUDED,
        max_term=84,
        score_meets=680,
        score_refer=640,
        dscr_meets=Decimal("1.25"),
        dscr_refer=Decimal("1.15"),
        rate_bands=(
            RateBand(700, None, Decimal("8.50"), "5.1"),
            RateBand(680, 699, Decimal("9.50"), "5.2"),
            RateBand(640, 679, Decimal("11.00"), "5.3"),
        ),
        fee_pct=Decimal("1.0"),
        combined_exposure_required=True,
        clauses={
            "time_in_business": "1.1",
            "amount": "1.2",
            "unsecured_cap": "1.3",
            "industry": "1.4",
            "term": "1.5",
            "score": "2.1",
            "dscr_meets": "2.3",
            "dscr_refer": "2.4",
            "dscr_decline": "2.5",
            "documents": "3.1",
            "hard_fail_route": "4.1",
            "pend_route": "4.2",
            "hold_route": "4.3",
            "combined_exposure": "4.5",
            "fee": "5.4",
        },
    ),
}
