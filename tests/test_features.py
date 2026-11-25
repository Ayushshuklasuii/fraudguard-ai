"""PARITY: the live velocity code must give the same numbers as the batch code."""
import numpy as np
import pandas as pd

from src.features import FEATURES, add_velocity_features, encode, live_velocity


def history_for(rows, i):
    """Earlier payments of the same card, as (time, amount) pairs."""
    r = rows[i]
    return [(x["TransactionDT"], x["TransactionAmt"]) for x in rows
            if x["card1"] == r["card1"] and x["TransactionDT"] < r["TransactionDT"]]


def test_live_velocity_matches_batch(demo):
    rows, df = demo["rows"], demo["df"]
    for i in range(0, len(rows), 7):                      # every 7th payment
        live = live_velocity(history_for(rows, i), rows[i]["TransactionDT"])
        assert live["cnt_1h"] == df.loc[i, "cnt_1h"]
        assert live["cnt_24h"] == df.loc[i, "cnt_24h"]
        assert abs(live["amt_sum_24h"] - df.loc[i, "amt_sum_24h"]) < 1e-6
        if np.isnan(df.loc[i, "secs_since_last"]):
            assert np.isnan(live["secs_since_last"])
        else:
            assert live["secs_since_last"] == df.loc[i, "secs_since_last"]


def test_live_ignores_same_time_and_future_payments():
    history = [(100, 10.0), (100, 20.0), (500, 99.0)]    # same time and a future one
    v = live_velocity(history, 100)
    assert v["cnt_1h"] == 0 and v["cnt_24h"] == 0 and v["amt_sum_24h"] == 0
    assert np.isnan(v["secs_since_last"])


def test_live_window_edges():
    t = 100_000
    history = [(t - 3600, 5.0),     # exactly 1 hour ago: inside the window
               (t - 3601, 7.0),     # just outside 1 hour, inside 24 hours
               (t - 86400, 1.0),    # exactly 24 hours ago: inside
               (t - 86401, 50.0)]   # just outside 24 hours
    v = live_velocity(history, t)
    assert v["cnt_1h"] == 1
    assert v["cnt_24h"] == 3
    assert v["amt_sum_24h"] == 13.0
    assert v["secs_since_last"] == 3600


def test_unseen_and_missing_categories_become_minus_one(demo):
    row = {c: 0 for c in FEATURES}
    row.update({"ProductCD": "NEW_PRODUCT", "card4": None, "card6": "debit",
                "P_emaildomain": "never-seen.com", "DeviceType": None})
    X = encode(pd.DataFrame([row]), demo["maps"])
    assert X.loc[0, "ProductCD"] == -1
    assert X.loc[0, "card4"] == -1
    assert X.loc[0, "P_emaildomain"] == -1
    assert X.loc[0, "DeviceType"] == -1
    assert X.loc[0, "card6"] != -1
