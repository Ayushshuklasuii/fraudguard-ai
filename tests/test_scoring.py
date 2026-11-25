"""Tests for score_transaction(): good input, bad input, risk levels, parity."""
import pytest
from pydantic import ValidationError

from src.scoring import risk_level, score_transaction
from tests.conftest import THRESHOLDS


def make_txn(rows, i):
    """A payment dict for row i, with its earlier same-card payments as history."""
    r = rows[i]
    txn = {k: r[k] for k in ["TransactionID", "TransactionDT", "TransactionAmt",
                             "has_identity", "card1", "ProductCD", "card2", "card4",
                             "card6", "addr1", "P_emaildomain", "DeviceType"]}
    txn["card_history"] = [{"TransactionDT": x["TransactionDT"], "TransactionAmt": x["TransactionAmt"]}
                           for x in rows if x["card1"] == r["card1"]
                           and x["TransactionDT"] < r["TransactionDT"]]
    return txn


# ---- 1. Training-serving parity: the most important test ----
def test_live_score_equals_batch_score(demo):
    rows, X, model, art = demo["rows"], demo["X"], demo["model"], demo["artifacts"]
    batch_scores = model.predict_proba(X)[:, 1]
    for i in range(0, len(rows), 25):                      # 120 payments
        live = score_transaction(make_txn(rows, i), artifacts=art)
        assert abs(live["fraud_score"] - batch_scores[i]) < 1e-6


# ---- 2. Output shape ----
def test_result_has_expected_fields(demo):
    result = score_transaction(make_txn(demo["rows"], 500), artifacts=demo["artifacts"])
    assert set(result) == {"transaction_id", "fraud_score", "risk_level",
                           "model_version", "velocity"}
    assert 0.0 <= result["fraud_score"] <= 1.0
    assert result["risk_level"] in {"low", "medium", "high"}
    assert result["model_version"] == "test"


def test_payment_with_no_history_and_missing_fields_still_scores(demo):
    txn = {"TransactionID": 1, "TransactionDT": 5000, "TransactionAmt": 20.0,
           "has_identity": 0, "card1": 99999}               # optional fields left out
    result = score_transaction(txn, artifacts=demo["artifacts"])
    assert 0.0 <= result["fraud_score"] <= 1.0


# ---- 3. Bad input is rejected ----
GOOD = {"TransactionID": 1, "TransactionDT": 5000, "TransactionAmt": 20.0,
        "has_identity": 0, "card1": 1234}


@pytest.mark.parametrize("change", [
    {"TransactionAmt": -5.0},          # negative amount
    {"TransactionAmt": 0},             # zero amount
    {"TransactionAmt": "abc"},         # wrong type
    {"has_identity": 2},               # must be 0 or 1
    {"TransactionDT": -1},             # time cannot be negative
    {"Amount": 10.0},                  # unknown field name (typo)
    {"card_history": [{"TransactionDT": 10, "TransactionAmt": -1}]},  # bad history item
])
def test_bad_input_is_rejected(demo, change):
    with pytest.raises(ValidationError):
        score_transaction({**GOOD, **change}, artifacts=demo["artifacts"])


@pytest.mark.parametrize("missing", ["TransactionID", "TransactionDT", "TransactionAmt",
                                     "has_identity", "card1"])
def test_missing_required_field_is_rejected(demo, missing):
    bad = {k: v for k, v in GOOD.items() if k != missing}
    with pytest.raises(ValidationError):
        score_transaction(bad, artifacts=demo["artifacts"])


# ---- 4. Risk levels ----
@pytest.mark.parametrize("score, expected", [
    (0.99, "high"), (0.70, "high"),        # at the high cutoff = high
    (0.69, "medium"), (0.35, "medium"),    # at the medium cutoff = medium
    (0.34, "low"), (0.0, "low"),
])
def test_risk_levels(score, expected):
    assert risk_level(score, THRESHOLDS) == expected
