"""Try the scoring function on two sample payments.

Run from the project folder:   python try_scoring.py
Needs artifacts/model_v1.joblib and artifacts/model_v1_meta.json (made by train.py).
"""
from src.scoring import DEFAULT_ART_DIR, score_transaction

if not (DEFAULT_ART_DIR / "model_v1.joblib").exists():
    raise SystemExit("model_v1.joblib not found in artifacts/. Run train.py first.")

# A payment from a card with NO earlier payments
fresh_card = {
    "TransactionID": 3000001,
    "TransactionDT": 1_000_000,
    "TransactionAmt": 59.0,
    "has_identity": 0,
    "card1": 9500, "card2": 514.0,
    "ProductCD": "W", "card4": "visa", "card6": "debit",
    "addr1": 315.0, "P_emaildomain": "gmail.com", "DeviceType": None,
}

# The same kind of payment, but this card was used 4 times in the last hour
busy_card = {**fresh_card, "TransactionID": 3000002,
             "card_history": [{"TransactionDT": 1_000_000 - s, "TransactionAmt": a}
                              for s, a in [(120, 150.0), (400, 200.0), (900, 99.0), (1500, 320.0)]]}

for name, txn in [("fresh card", fresh_card), ("busy card ", busy_card)]:
    r = score_transaction(txn)
    v = r["velocity"]
    print(f"{name}: score {r['fraud_score']:.4f} | risk {r['risk_level']:6s} | "
          f"cnt_1h {v['cnt_1h']:.0f} | model {r['model_version']}")
