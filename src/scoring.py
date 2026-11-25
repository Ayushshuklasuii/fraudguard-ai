"""FraudGuard AI v2.0 - src/scoring.py

ONE function that scores ONE payment: score_transaction().
Kafka (Phase 4) and FastAPI (Phase 5) will both call it, so they always
give the same answer for the same payment.

What it does:
  1. Check the input with Pydantic (right fields, right types, sane values)
  2. Build the velocity features from the card's earlier payments
  3. Encode the row exactly like training did
  4. Score it with the saved XGBoost model
  5. Turn the score into low / medium / high using the saved thresholds
"""
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import joblib
import json
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

try:                                   # imported as a package (tests, API)
    from src.features import (FEATURES, WINDOW_1H, WINDOW_24H, encode,
                              live_velocity, make_feature_row)
except ModuleNotFoundError:            # run as a plain script from src/
    from features import (FEATURES, WINDOW_1H, WINDOW_24H, encode,
                          live_velocity, make_feature_row)

DEFAULT_ART_DIR = Path(__file__).resolve().parent.parent / "artifacts"


# ----------------------------------------------------------------------------
# Input check (Pydantic)
# ----------------------------------------------------------------------------
class HistoryItem(BaseModel):
    """One earlier payment of the same card."""
    TransactionDT: float = Field(ge=0)
    TransactionAmt: float = Field(gt=0)


class Transaction(BaseModel):
    """One payment to score. Unknown field names are rejected."""
    model_config = ConfigDict(extra="forbid")

    TransactionID: int
    TransactionDT: float = Field(ge=0)        # seconds offset, like the dataset
    TransactionAmt: float = Field(gt=0)       # dollars
    has_identity: int = Field(ge=0, le=1)     # 1 if an identity record exists
    card1: int
    ProductCD: Optional[str] = None
    card2: Optional[float] = None
    card4: Optional[str] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    P_emaildomain: Optional[str] = None
    DeviceType: Optional[str] = None
    # Earlier payments of this card. In Phase 3 this will come from Redis.
    card_history: List[HistoryItem] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# Load the saved model once
# ----------------------------------------------------------------------------
@lru_cache(maxsize=4)
def load_artifacts(version: str = "v1", art_dir: Optional[str] = None) -> dict:
    d = Path(art_dir) if art_dir else DEFAULT_ART_DIR
    model = joblib.load(d / f"model_{version}.joblib")
    meta = json.loads((d / f"model_{version}_meta.json").read_text())

    # Safety guard against training-serving skew: if the code and the saved
    # model disagree about the features or the windows, stop right here.
    if meta["features"] != FEATURES:
        raise ValueError("Feature list in the model file differs from features.py")
    if meta["velocity_windows_sec"] != {"1h": WINDOW_1H, "24h": WINDOW_24H}:
        raise ValueError("Velocity windows in the model file differ from features.py")
    return {"model": model, "meta": meta}


# ----------------------------------------------------------------------------
# Risk level
# ----------------------------------------------------------------------------
def risk_level(score: float, thresholds: dict) -> str:
    if score >= thresholds["high"]:
        return "high"
    if score >= thresholds["medium"]:
        return "medium"
    return "low"


# ----------------------------------------------------------------------------
# The one function
# ----------------------------------------------------------------------------
def score_transaction(txn, artifacts: Optional[dict] = None) -> dict:
    """Score one payment. `txn` is a dict (or a Transaction).

    Raises pydantic.ValidationError if the input is bad.
    Note: fraud_score is a model score between 0 and 1, NOT a true probability.
    """
    art = artifacts or load_artifacts()
    t = txn if isinstance(txn, Transaction) else Transaction.model_validate(txn)

    history = [(h.TransactionDT, h.TransactionAmt) for h in t.card_history]
    velocity = live_velocity(history, t.TransactionDT)

    row = make_feature_row(t.model_dump(exclude={"card_history"}), velocity)
    X = encode(pd.DataFrame([row]), art["meta"]["category_maps"])

    score = float(art["model"].predict_proba(X)[:, 1][0])
    return {
        "transaction_id": t.TransactionID,
        "fraud_score": score,
        "risk_level": risk_level(score, art["meta"]["thresholds"]),
        "model_version": art["meta"]["version"],
        "velocity": velocity,
    }
