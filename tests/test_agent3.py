from datetime import date
from types import SimpleNamespace

import pytest

from harbor.agent2 import run_agent2
from harbor.agent3 import EQUAL_OPPORTUNITY, PolishedLetter, PolishedStatement, polish_letter, run_agent3, validate_letter
from harbor.pend_timer import PendStore
from tests.test_agent2 import STORE, _customers, _drop_license, payload


def build(app_id, pend_store=None, **kwargs):
    p = payload(app_id, **kwargs)
    report = run_agent2(p, STORE, pend_store)
    return p, report, run_agent3(p, report)


def check(text, p, r):
    from harbor.agent3 import known_names

    return validate_letter(text, r, known_names(p, r))


CASES = {
    "meets": dict(app_id="1001", related=False),
    "refer_policy": dict(app_id="1001", related=False, customers=_customers(credit_score=660)),
    "refer_risk": dict(app_id="1001", related=False, customers=_customers(bureau_metadata_stub={"bankruptcy_flag": True, "bankruptcy_date": "2023-05-01"})),
    "decline": dict(app_id="1002"),
    "pend": dict(app_id="1001", related=False, text_edit=_drop_license),
    "hold_ein": dict(app_id="1003"),
    "hold_income": dict(app_id="1004"),
}


@pytest.mark.parametrize("case", CASES)
def test_every_tier_produces_a_grounded_draft(case):
    _, _, letter = build(**CASES[case])
    assert letter.validation.passed, letter.validation.errors
    assert letter.rendered_text.startswith("DRAFT")
    assert EQUAL_OPPORTUNITY in letter.rendered_text
    assert "version 2026.2" in letter.rendered_text


def test_proposed_terms_letter_has_exact_figures_and_no_approval_language():
    _, _, letter = build(**CASES["meets"])
    text = letter.rendered_text
    for figure in ("$1,641.32", "$800.00", "$18,479.20", "8.50%"):
        assert figure in text
    assert "approved" not in text.lower()
    assert "credit officer will make the final decision" in text


def test_pend_letter_states_item_impact_and_deadline():
    _, _, letter = build(**CASES["pend"])
    text = letter.rendered_text
    assert letter.letter_type == "Information Request"
    assert "Business license" in text and "November 22, 2026" in text
    assert "requires this to complete your file" in text


def test_stale_letter(tmp_path):
    store = PendStore(tmp_path / "t.json")
    build(pend_store=store, **CASES["pend"])
    _, report, letter = build(pend_store=store, review=date(2026, 11, 23), **CASES["pend"])
    assert report.tier == "STALE"
    assert letter.validation.passed, letter.validation.errors
    assert "did not receive it by November 22, 2026" in letter.rendered_text


def test_decline_letter_cites_only_the_actual_reason():
    _, _, letter = build(**CASES["decline"])
    text = letter.rendered_text
    assert "20 months" in text and "at least 24 months" in text and "section 1.1" in text
    assert "credit score" not in text.lower()
    assert "Business license" not in text


def test_bankruptcy_letter_states_file_fact_and_does_not_blame_score():
    _, _, letter = build(**CASES["refer_risk"])
    text = letter.rendered_text
    assert "Bankruptcy on record, dated May 1, 2023" in text
    assert "credit score" not in text.lower()
    assert any("mock" in n for n in letter.officer_notes)


def test_hold_letter_does_not_tip_off_or_repeat_injected_text():
    _, _, letter = build(**CASES["hold_ein"])
    lowered = letter.rendered_text.lower()
    for term in ("ein", "fraud", "compliance", "pre-approved", "system note", "manager"):
        assert not __import__("re").search(rf"\b{term}\b", lowered), term
    assert any("EIN" in n for n in letter.officer_notes)


def test_validator_rejects_edits_that_break_grounding():
    p, r, letter = build(**CASES["meets"])
    text = letter.rendered_text
    assert not check(text.replace("$1,641.32", "$1,500.00"), p, r).passed
    assert not check(text.replace("section 2.1", "section 9.9"), p, r).passed
    assert not check(text + "\nYour loan is approved.", p, r).passed
    assert not check(text.replace(EQUAL_OPPORTUNITY, ""), p, r).passed


def test_validator_catches_an_invented_figure():
    p, r, letter = build(**CASES["decline"])
    assert any("$5,000" in e or "5000" in e for e in check(letter.rendered_text + "\nYou owe $5,000.", p, r).errors)


def test_placeholders_are_warnings():
    _, _, letter = build(**CASES["meets"])
    assert any("contact" in w for w in letter.validation.warnings)
    assert any("address" in w for w in letter.validation.warnings)


class FakeClient:
    def __init__(self, transform):
        self.transform = transform
        self.calls = 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self.parse))

    def parse(self, **kwargs):
        self.calls += 1
        import json

        items = json.loads(kwargs["messages"][0]["content"])
        output = PolishedLetter(statements=[PolishedStatement(id=i["id"], text=self.transform(i["text"])) for i in items])
        return SimpleNamespace(stop_reason="end_turn", parsed_output=output)


def test_good_rewording_is_kept():
    p, r, letter = build(**CASES["meets"])
    updated, problem = polish_letter(letter, p, r, FakeClient(lambda t: t.replace("Thank you for applying.", "Thanks for applying.")))
    assert problem is None
    assert updated.method == "template_reworded_by_llm"
    assert "Thanks for applying." in updated.rendered_text


@pytest.mark.parametrize(
    "transform",
    [lambda t: t.replace("1,641.32", "1,500.00"), lambda t: t + " Your loan is approved."],
)
def test_bad_rewording_falls_back_to_template(transform):
    p, r, letter = build(**CASES["meets"])
    updated, problem = polish_letter(letter, p, r, FakeClient(transform))
    assert problem
    assert updated == letter


def test_hold_letters_are_never_sent_to_the_model():
    p, r, letter = build(**CASES["hold_ein"])
    client = FakeClient(lambda t: t)
    updated, problem = polish_letter(letter, p, r, client)
    assert client.calls == 0 and updated == letter and problem
