"""FraudGuard AI v2.0 - src/features.py

The feature code lives HERE, once. Training (train.py), live scoring
(scoring.py), Kafka and the API all use it. That is how we avoid
training-serving skew: the model never sees a feature that was built
differently from the way it was built during training.

Two ways to build the velocity features, same rules:
  * add_velocity_features(df)  - batch, for the whole training table
  * live_velocity(history, t)  - live, for ONE payment (history comes
                                 from Redis later, in Phase 3)
Rule: look only at EARLIER payments of the same card, in the window
[t - window, t). The current payment is never counted.
"""
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Settings shared by training and serving
# ----------------------------------------------------------------------------
CAT_COLS = ["ProductCD", "card4", "card6", "P_emaildomain", "DeviceType"]
BASE_FEATURES = ["TransactionAmt", "hour", "has_identity", "ProductCD",
                 "card1", "card2", "card4", "card6", "addr1",
                 "P_emaildomain", "DeviceType"]
VELOCITY_FEATURES = ["cnt_1h", "cnt_24h", "amt_sum_24h", "secs_since_last"]
FEATURES = BASE_FEATURES + VELOCITY_FEATURES

CARD_KEY = "card1"    # IEEE-CIS has no real card ID; card1 is our best guess
WINDOW_1H = 3600      # seconds
WINDOW_24H = 86400    # seconds


# ----------------------------------------------------------------------------
# Velocity, batch version (whole table at once, used in training)
# ----------------------------------------------------------------------------
def add_velocity_features(df: pd.DataFrame, key: str = CARD_KEY) -> pd.DataFrame:
    n = len(df)
    cnt_1h = np.zeros(n)
    cnt_24h = np.zeros(n)
    amt_24h = np.zeros(n)
    since_last = np.full(n, np.nan)

    t_all = df["TransactionDT"].to_numpy(dtype="float64")
    a_all = df["TransactionAmt"].to_numpy(dtype="float64")

    for idx in df.groupby(key, sort=False).indices.values():
        idx = np.sort(idx)  # df is time-sorted, so this stays in order
        t, a = t_all[idx], a_all[idx]
        cum = np.concatenate(([0.0], np.cumsum(a)))

        right = np.searchsorted(t, t, side="left")                 # strictly earlier rows
        left_1h = np.searchsorted(t, t - WINDOW_1H, side="left")
        left_24h = np.searchsorted(t, t - WINDOW_24H, side="left")

        cnt_1h[idx] = right - left_1h
        cnt_24h[idx] = right - left_24h
        amt_24h[idx] = cum[right] - cum[left_24h]

        has_prev = right > 0
        prev_t = t[np.maximum(right - 1, 0)]
        since_last[idx[has_prev]] = (t - prev_t)[has_prev]

    out = df.copy()
    out["cnt_1h"] = cnt_1h
    out["cnt_24h"] = cnt_24h
    out["amt_sum_24h"] = amt_24h
    out["secs_since_last"] = since_last  # NaN = first time we see this card
    return out


# ----------------------------------------------------------------------------
# Velocity, live version (ONE payment, history given by the caller)
# ----------------------------------------------------------------------------
def live_velocity(history, t: float) -> dict:
    """Velocity features for one payment made at time t.

    history = list of (TransactionDT, TransactionAmt) for EARLIER payments of
    the same card. Payments at the same time as t, or later, are ignored,
    exactly like in the batch version.
    """
    cnt_1h = 0
    cnt_24h = 0
    amt_24h = 0.0
    last = None
    for h_t, h_amt in history:
        if h_t >= t:                      # not strictly earlier: skip
            continue
        if h_t >= t - WINDOW_1H:
            cnt_1h += 1
        if h_t >= t - WINDOW_24H:
            cnt_24h += 1
            amt_24h += h_amt
        if last is None or h_t > last:
            last = h_t
    return {"cnt_1h": float(cnt_1h),
            "cnt_24h": float(cnt_24h),
            "amt_sum_24h": float(amt_24h),
            "secs_since_last": float(t - last) if last is not None else np.nan}


# ----------------------------------------------------------------------------
# Encoding (learned from TRAIN rows only, saved for serving)
# ----------------------------------------------------------------------------
def fit_category_maps(train_df: pd.DataFrame) -> dict:
    return {c: {str(v): i for i, v in
                enumerate(sorted(train_df[c].dropna().astype(str).unique()))}
            for c in CAT_COLS}


def encode(df: pd.DataFrame, maps: dict) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in CAT_COLS:
        X[c] = X[c].map(maps[c]).fillna(-1).astype("int32")  # unseen or missing = -1
    return X


# ----------------------------------------------------------------------------
# One live payment -> one row of raw features (before encoding)
# ----------------------------------------------------------------------------
def _num(v):
    return np.nan if v is None else v


def make_feature_row(txn: dict, velocity: dict) -> dict:
    """Build the raw feature row for ONE payment. Same columns as training."""
    return {
        "TransactionAmt": txn["TransactionAmt"],
        "hour": int(txn["TransactionDT"] // 3600) % 24,   # same rule as train.py
        "has_identity": txn["has_identity"],
        "ProductCD": txn.get("ProductCD"),
        "card1": txn["card1"],
        "card2": _num(txn.get("card2")),
        "card4": txn.get("card4"),
        "card6": txn.get("card6"),
        "addr1": _num(txn.get("addr1")),
        "P_emaildomain": txn.get("P_emaildomain"),
        "DeviceType": txn.get("DeviceType"),
        **velocity,
    }
