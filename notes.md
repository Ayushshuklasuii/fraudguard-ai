# FraudGuard AI - Phase 1 results (train.py v1)

Test split: last 15% by time, 88,581 payments, 3,083 fraud.

- XGBoost test PR-AUC: 0.1975
- Baseline (Logistic Regression) test PR-AUC: 0.0905
- Precision: 0.0696
- Recall: 0.8362
- Recall at 1% false alarms: 0.1589
- Fraud caught: $387,447 / missed: $82,161
- Review cost (assumed $5 per alert): $185,165
- Single-row latency (model only): p50 5.77 ms, p95 7.22 ms

Top features: ProductCD, card6, DeviceType, TransactionAmt, P_emaildomain.

Only secs_since_last (velocity) reached the top 10.

Limits:

- card1 is a rough guess for "same card", so velocity features are noisy.
- The $5 review cost is an assumption. It makes the model flag about 42% of payments.
- Only 11 of 400+ columns are used.

# FraudGuard AI - Phase 2 results (features.py + scoring.py)

- `src/features.py` now owns loading, velocity features, encoding and the
  single-row feature builder. `src/train.py` imports it, so there is no second
  copy of the feature logic.
- `src/scoring.py` exposes the one `score_transaction()` path: Pydantic-validated
  payment in, fraud score + risk tier + model version out.
- The scorer refuses to load an artifact whose feature list or category columns
  differ from `src/features.py` (catches stale models).
- Tests: 28 passed (`python -m pytest tests -q`).
- Batch-vs-live velocity parity: exact equality on all 4 velocity fields for 300
  randomly sampled payments across the full 590,540-row dataset.
- Sample payment (gmail / visa / debit / mobile, $129.99): score 0.593 -> high
  with model v1 thresholds (high 0.2897, medium 0.1448).
