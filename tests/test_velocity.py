"""Checks the fast (batch) velocity code against a slow, obvious version."""
import numpy as np
import pandas as pd

from src.features import add_velocity_features


def brute_force(df, i):
    """Slow, obvious version: look at earlier rows of the same card only."""
    row = df.iloc[i]
    same = df[(df["card1"] == row["card1"]) & (df["TransactionDT"] < row["TransactionDT"])]
    in_1h = same[same["TransactionDT"] >= row["TransactionDT"] - 3600]
    in_24h = same[same["TransactionDT"] >= row["TransactionDT"] - 86400]
    last = (row["TransactionDT"] - same["TransactionDT"].max()) if len(same) else np.nan
    return len(in_1h), len(in_24h), in_24h["TransactionAmt"].sum(), last


def test_velocity_matches_brute_force():
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({
        "TransactionDT": np.sort(rng.integers(0, 200_000, n)),
        "TransactionAmt": rng.uniform(1, 500, n).round(2),
        "card1": rng.integers(0, 12, n),
    })
    out = add_velocity_features(df)
    for i in range(n):
        c1, c24, amt24, last = brute_force(df, i)
        assert out.loc[i, "cnt_1h"] == c1
        assert out.loc[i, "cnt_24h"] == c24
        assert abs(out.loc[i, "amt_sum_24h"] - amt24) < 1e-6
        if np.isnan(last):
            assert np.isnan(out.loc[i, "secs_since_last"])
        else:
            assert out.loc[i, "secs_since_last"] == last


def test_current_transaction_not_counted():
    df = pd.DataFrame({"TransactionDT": [100, 100, 200],
                       "TransactionAmt": [10.0, 20.0, 30.0],
                       "card1": [1, 1, 1]})
    out = add_velocity_features(df)
    assert out.loc[0, "cnt_1h"] == 0   # same-time rows are not "earlier"
    assert out.loc[1, "cnt_1h"] == 0
    assert out.loc[2, "cnt_1h"] == 2
