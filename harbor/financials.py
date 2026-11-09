"""Deterministic loan math. No model ever performs these calculations."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from dataclasses import dataclass

CENT = Decimal("0.01")


def to_cents(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def monthly_payment(principal: Decimal, annual_rate_pct: Decimal, months: int) -> Decimal:
    """M = P * r(1+r)^n / ((1+r)^n - 1), rounded to the cent."""
    if principal <= 0 or months <= 0:
        raise ValueError("Principal and term must be positive")
    r = annual_rate_pct / Decimal(100) / Decimal(12)
    if r == 0:
        return to_cents(principal / months)
    growth = (1 + r) ** months
    return to_cents(principal * r * growth / (growth - 1))


def dscr(noi: Decimal, annual_debt_service: Decimal) -> Decimal:
    """Unrounded ratio. Thresholds are compared against this exact value."""
    if annual_debt_service <= 0:
        raise ValueError("Annual debt service must be positive")
    return noi / annual_debt_service


@dataclass(frozen=True)
class LoanTerms:
    monthly_payment: Decimal
    total_interest: Decimal
    origination_fee: Decimal
    total_of_payments: Decimal


def loan_terms(principal: Decimal, annual_rate_pct: Decimal, months: int, fee_pct: Decimal) -> LoanTerms:
    payment = monthly_payment(principal, annual_rate_pct, months)
    total = payment * months
    return LoanTerms(
        monthly_payment=payment,
        total_interest=to_cents(total - principal),
        origination_fee=to_cents(principal * fee_pct / Decimal(100)),
        total_of_payments=to_cents(total),
    )
