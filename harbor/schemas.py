"""Pydantic contracts for Agent 1.

Separation of claimed and verified data is enforced by the types:
- Every model forbids extra fields, so `claimed_income` cannot be placed on
  VerifiedData and `verified_noi` cannot be placed on ClaimedData.
- A claimed value's source must be the application form.
- A verified NOI's source must be a tax return.
- ApplicationPayload has no decision field. Agent 1 extracts; it never recommends.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")

DocumentType = Literal[
    "application_form",
    "tax_return",
    "bank_statement",
    "photo_id",
    "business_license",
    "other",
]
ExtractionMethod = Literal["rule_parser", "llm_parser"]
DocumentKind = Literal[
    "tax_return", "bank_statement", "photo_id", "business_license", "signed_application"
]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- Raw per-document extraction. Also the structured-output format for the LLM. ---


class FieldEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    text: str


class TaxFigure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tax_year: int
    net_operating_income: float | None
    ein: str | None
    evidence: str


class ReceivedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: DocumentKind
    tax_year: int | None
    months: list[str]  # YYYY-MM
    evidence: str


class DocExtraction(BaseModel):
    """What one document says. No field here can express a credit decision."""

    model_config = ConfigDict(extra="forbid")
    document_type: DocumentType
    application_id: str | None = None
    customer_id: str | None = None
    business_name: str | None = None
    ein: str | None = None
    owner_name: str | None = None
    ownership_pct: float | None = None
    business_start_date: str | None = None
    industry: str | None = None
    requested_amount: float | None = None
    term_months: int | None = None
    secured: bool | None = None
    collateral_description: str | None = None
    collateral_value: float | None = None
    purpose: str | None = None
    stated_income: float | None = None
    existing_monthly_debt: float | None = None
    signed_date: str | None = None
    submitted_date: str | None = None
    tax_figures: list[TaxFigure] = Field(default_factory=list)
    documents_received: list[ReceivedItem] = Field(default_factory=list)
    documents_not_received: list[str] = Field(default_factory=list)
    applicant_notes: list[str] = Field(default_factory=list)
    embedded_instructions_found: list[str] = Field(default_factory=list)
    evidence: list[FieldEvidence] = Field(default_factory=list)
    unparsed_lines: list[str] = Field(default_factory=list)


# --- Provenance ---


class ClaimedSource(Strict):
    document: str
    document_type: Literal["application_form"] = "application_form"
    evidence: str
    method: ExtractionMethod


class Claimed(Strict, Generic[T]):
    value: T
    source: ClaimedSource


class TaxReturnSource(Strict):
    document: str
    document_type: Literal["tax_return"] = "tax_return"
    evidence: str
    method: ExtractionMethod
    primary_document_reviewed: bool = Field(
        description="False when the figure comes from the intake abstract of a "
        "return logged as received, rather than the return itself."
    )


# --- Partitions ---


class ClaimedData(Strict):
    """Stated by the applicant on the application form."""

    customer_id: Claimed[str] | None = None
    business_name: Claimed[str] | None = None
    ein: Claimed[str] | None = None
    owner_name: Claimed[str] | None = None
    ownership_pct: Claimed[float] | None = None
    business_start_date: Claimed[date] | None = None
    industry: Claimed[str] | None = None
    requested_amount: Claimed[float] | None = None
    term_months: Claimed[int] | None = None
    secured: Claimed[bool] | None = None
    collateral_description: Claimed[str] | None = None
    collateral_value: Claimed[float] | None = None
    purpose: Claimed[str] | None = None
    claimed_income: Claimed[float] | None = None
    claimed_existing_monthly_debt: Claimed[float] | None = None
    signed_date: Claimed[date] | None = None
    submitted_date: Claimed[date] | None = None


class VerifiedNOI(Strict):
    tax_year: int
    amount: float
    source: TaxReturnSource


class VerifiedEIN(Strict):
    tax_year: int
    ein: str
    source: TaxReturnSource


class DocumentReceipt(Strict):
    kind: DocumentKind
    tax_year: int | None = None
    months: tuple[str, ...] = ()
    basis: Literal["intake_log", "primary_document"]
    source_document: str
    evidence: str


class CustomerRecord(Strict):
    customer_id: str
    business: str | None = None
    owner: str | None = None
    ein_on_file: str | None = None
    open_applications: tuple[str, ...] = ()
    source_file: Literal["customers.json"] = "customers.json"


class BureauMetadataStub(Strict):
    """Mock placeholder. None means no data, never a clean result."""

    is_mock: Literal[True] = True
    public_records_count: int | None = None
    bankruptcy_flag: bool | None = None
    bankruptcy_date: date | None = None
    tax_lien_flag: bool | None = None
    judgment_flag: bool | None = None


class BureauSnapshot(Strict):
    credit_score: int | None = None
    pull_date: date | None = None
    source: Literal["customers.json"] = "customers.json"
    external_pull_performed: Literal[False] = False
    bureau_metadata_stub: BureauMetadataStub = BureauMetadataStub()


class VerifiedData(Strict):
    """From filed documents and bank-held records only."""

    verified_noi: VerifiedNOI | None = None
    noi_history: tuple[VerifiedNOI, ...] = ()
    tax_return_eins: tuple[VerifiedEIN, ...] = ()
    customer_record: CustomerRecord | None = None
    bureau: BureauSnapshot = BureauSnapshot()
    documents_received: tuple[DocumentReceipt, ...] = ()


class MissingItem(Strict):
    item: str
    kind: Literal["document", "stale_document", "field"]
    detail: str
    policy_citation: str | None = None


class MissingData(Strict):
    items: tuple[MissingItem, ...] = ()


# --- Everything else on the payload ---


class AppliedPolicy(Strict):
    policy_id: str
    version: str
    effective_date: date
    selected_by: Literal["signed_date"] = "signed_date"
    source_file: str


class UntrustedNote(Strict):
    """Applicant free text. Data only: never instructions, never letter content."""

    text: str
    document: str
    suspected_injection: bool


class RelatedApplication(Strict):
    application_id: str
    on_customer_record: bool
    found_in_intake: bool
    requested_amount: float | None = None
    term_months: int | None = None
    stated_income: float | None = None
    tax_return_noi: float | None = None
    tax_return_year: int | None = None


class AuditEvent(Strict):
    code: str
    severity: Literal["info", "warning", "critical"]
    message: str
    field: str | None = None
    values: dict[str, str] = Field(default_factory=dict)
    documents: tuple[str, ...] = ()
    policy_citation: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SourceDocumentRecord(Strict):
    name: str
    sha256: str
    document_type: DocumentType | None
    method: ExtractionMethod | None


class ApplicationPayload(Strict):
    schema_version: Literal["1.0"] = "1.0"
    application_id: str | None
    review_date: date
    policy: AppliedPolicy | None
    claimed: ClaimedData
    verified: VerifiedData
    missing: MissingData
    related_applications: tuple[RelatedApplication, ...] = ()
    untrusted_notes: tuple[UntrustedNote, ...] = ()
    source_documents: tuple[SourceDocumentRecord, ...] = ()
    audit_log: tuple[AuditEvent, ...] = ()
