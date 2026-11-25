"""Shared test setup: builds a tiny fake dataset and a tiny trained model.

The tests use their own small model in a temporary folder, so they never
touch your real artifacts/ folder and run in a few seconds.
"""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.features import (CAT_COLS, CARD_KEY, FEATURES, WINDOW_1H, WINDOW_24H,
                          add_velocity_features, encode, fit_category_maps)
from src.scoring import load_artifacts

THRESHOLDS = {"high": 0.70, "medium": 0.35}


def make_rows(n=3000, seed=7):
    """Fake payments, sorted by time, many payments per card."""
    rng = np.random.default_rng(seed)
    times = np.sort(rng.integers(0, 400_000, n))
    rows = []
    for i in range(n):
        rows.append({
            "TransactionID": 1_000_000 + i,
            "TransactionDT": int(times[i]),
            "TransactionAmt": float(round(rng.uniform(1, 500), 2)),
            "has_identity": int(rng.random() < 0.25),
            "card1": int(rng.integers(0, 15)),
            "ProductCD": str(rng.choice(["W", "C", "H", "S", "R"])),
            "card2": None if rng.random() < 0.05 else float(rng.integers(100, 600)),
            "card4": None if rng.random() < 0.05 else str(rng.choice(["visa", "mastercard"])),
            "card6": str(rng.choice(["debit", "credit"])),
            "addr1": None if rng.random() < 0.1 else float(rng.integers(100, 500)),
            "P_emaildomain": None if rng.random() < 0.15 else str(rng.choice(["gmail.com", "yahoo.com"])),
            "DeviceType": None if rng.random() < 0.7 else str(rng.choice(["desktop", "mobile"])),
            "isFraud": int(rng.random() < 0.06),
        })
    return rows


@pytest.fixture(scope="session")
def demo(tmp_path_factory):
    rows = make_rows()
    df = pd.DataFrame(rows)
    df["hour"] = (df["TransactionDT"] // 3600) % 24
    df = add_velocity_features(df)

    maps = fit_category_maps(df)
    X = encode(df, maps)
    model = XGBClassifier(n_estimators=30, max_depth=3, random_state=0, n_jobs=1)
    model.fit(X, df["isFraud"].to_numpy())

    art_dir = tmp_path_factory.mktemp("artifacts")
    joblib.dump(model, art_dir / "model_test.joblib")
    meta = {"version": "test", "features": FEATURES, "category_columns": CAT_COLS,
            "category_maps": maps, "card_key": CARD_KEY,
            "velocity_windows_sec": {"1h": WINDOW_1H, "24h": WINDOW_24H},
            "thresholds": THRESHOLDS}
    (art_dir / "model_test_meta.json").write_text(json.dumps(meta))

    artifacts = load_artifacts("test", str(art_dir))
    return {"rows": rows, "df": df, "X": X, "model": model, "maps": maps,
            "artifacts": artifacts}
