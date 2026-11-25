"""FraudGuard AI v2.0 - src/train.py
Trains a fraud model on the IEEE-CIS Fraud Detection data.

Feature code now lives in src/features.py (shared with scoring.py).

Steps:
  1. Load only the columns we need and join transaction + identity tables
  2. Sort by time and add point-in-time card velocity features (past only)
  3. Split by time: 70% train / 15% validation / 15% test
  4. Train a Logistic Regression baseline and an XGBoost model
  5. Pick the alert threshold on VALIDATION data using a dollar-cost rule
  6. Evaluate on the TEST split exactly once
  7. Save the model, encoders, thresholds and metrics for scoring.py

Run:  python src/train.py --data-dir data --version v1
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Shared feature code (also used by scoring.py). Works both when this file is
# run as a script (python src/train.py) and when it is imported (tests).
try:
    from src.features import (CAT_COLS, CARD_KEY, FEATURES, WINDOW_1H,
                              WINDOW_24H, add_velocity_features, encode,
                              fit_category_maps)
except ModuleNotFoundError:
    from features import (CAT_COLS, CARD_KEY, FEATURES, WINDOW_1H,
                          WINDOW_24H, add_velocity_features, encode,
                          fit_category_maps)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
TX_COLS = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
           "ProductCD", "card1", "card2", "card4", "card6", "addr1",
           "P_emaildomain"]
ID_COLS = ["TransactionID", "DeviceType"]
REVIEW_COST = 5.0     # assumed dollar cost of an analyst reviewing one alert
SPLIT = (0.70, 0.85)  # train ends at 70%, validation ends at 85%


# ----------------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------------
def load_data(data_dir: Path) -> pd.DataFrame:
    tx = pd.read_csv(data_dir / "train_transaction.csv", usecols=TX_COLS)
    ident = pd.read_csv(data_dir / "train_identity.csv", usecols=ID_COLS)
    ident = ident.drop_duplicates("TransactionID")
    df = tx.merge(ident, on="TransactionID", how="left")

    # Real "has identity record" flag (not guessed from DeviceType)
    df["has_identity"] = df["TransactionID"].isin(ident["TransactionID"]).astype(int)

    # Strict time order. TransactionID breaks ties so the order is repeatable.
    df = df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    df["hour"] = (df["TransactionDT"] // 3600) % 24  # approximate hour of day
    return df


# ----------------------------------------------------------------------------
# 4. Threshold and metrics
# ----------------------------------------------------------------------------
def best_cost_threshold(y, score, amt, review_cost=REVIEW_COST):
    """Pick the score cutoff with the lowest total cost on validation data.

    cost = dollars of fraud we miss + review_cost for each false alarm.
    A missed fraud costs its transaction amount. Returns (threshold, cost).
    """
    order = np.argsort(-score, kind="stable")
    s, yy, aa = score[order], y[order], amt[order]
    total_fraud = float((aa * yy).sum())
    missed = total_fraud - np.cumsum(aa * yy)   # fraud $ still missed
    false_alarms = np.cumsum(1 - yy)            # normal rows flagged
    cost = np.concatenate(([total_fraud], missed + review_cost * false_alarms))
    k = int(cost.argmin())
    if k == 0:  # best choice: flag nothing
        return float(s[0]) + 1e-6, float(cost[0])
    return float(s[k - 1]), float(cost[k])


def evaluate(y, score, amt, threshold):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y, score)
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "recall_at_1pct_fpr": float(tpr[fpr <= 0.01].max()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "fraud_usd_caught": float(amt[(pred == 1) & (y == 1)].sum()),
        "fraud_usd_missed": float(amt[(pred == 0) & (y == 1)].sum()),
        "review_cost_usd": float(REVIEW_COST * (tp + fp)),
    }


def single_row_latency(model, X, n=200):
    """Rough single-transaction scoring time (model only, no Kafka or DB)."""
    ms = []
    for i in range(min(n, len(X))):
        t0 = time.perf_counter()
        model.predict_proba(X.iloc[[i]])
        ms.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(ms, 50)),
            "p95_ms": float(np.percentile(ms, 95))}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data(Path(args.data_dir))
    print(f"  rows: {len(df):,}   fraud rate: {df['isFraud'].mean():.2%}")

    print("Adding velocity features...")
    df = add_velocity_features(df)

    n = len(df)
    a, b = int(SPLIT[0] * n), int(SPLIT[1] * n)
    train, val, test = df.iloc[:a], df.iloc[a:b], df.iloc[b:]
    print(f"  train {len(train):,} | val {len(val):,} | test {len(test):,}")
    print(f"  fraud rate train {train['isFraud'].mean():.2%} | "
          f"val {val['isFraud'].mean():.2%} | test {test['isFraud'].mean():.2%}")

    maps = fit_category_maps(train)
    X_train, X_val, X_test = (encode(d, maps) for d in (train, val, test))
    y_train, y_val, y_test = (d["isFraud"].to_numpy() for d in (train, val, test))
    amt_val = val["TransactionAmt"].to_numpy()
    amt_test = test["TransactionAmt"].to_numpy()

    # --- Baseline: Logistic Regression ---
    print("Training baseline (Logistic Regression)...")
    base = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=1000))
    base.fit(X_train, y_train)
    base_val_ap = average_precision_score(y_val, base.predict_proba(X_val)[:, 1])

    # --- Main model: XGBoost ---
    print("Training XGBoost...")
    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    params = dict(n_estimators=600, max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                  eval_metric="aucpr", early_stopping_rounds=30,
                  random_state=42, n_jobs=-1)
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_scores = model.predict_proba(X_val)[:, 1]
    xgb_val_ap = average_precision_score(y_val, val_scores)
    print(f"  validation PR-AUC   baseline {base_val_ap:.4f}   xgboost {xgb_val_ap:.4f}")

    # --- Threshold from VALIDATION data only ---
    high_thr, val_cost = best_cost_threshold(y_val, val_scores, amt_val)
    medium_thr = high_thr * 0.5  # starting rule, tune later in risk.py
    print(f"  high-risk threshold {high_thr:.4f} | medium {medium_thr:.4f} "
          f"(val cost ${val_cost:,.0f})")

    # --- Final TEST evaluation (once) ---
    print("Evaluating on the test split (once)...")
    test_scores = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_scores, amt_test, high_thr)
    base_test_ap = float(average_precision_score(y_test,
                                                 base.predict_proba(X_test)[:, 1]))
    latency = single_row_latency(model, X_test)
    importance = sorted(zip(FEATURES, model.feature_importances_),
                        key=lambda x: -x[1])[:10]

    print("\n=== TEST RESULTS (xgboost) ===")
    for k, v in test_metrics.items():
        print(f"  {k:22s} {v:,.4f}" if isinstance(v, float) else f"  {k:22s} {v:,}")
    print(f"  baseline test PR-AUC   {base_test_ap:.4f}")
    print(f"  single-row latency     p50 {latency['p50_ms']:.2f} ms | "
          f"p95 {latency['p95_ms']:.2f} ms")
    print("\nTop features:")
    for name, imp in importance:
        print(f"  {name:18s} {imp:.4f}")

    # --- Save everything scoring.py will need ---
    model_path = out_dir / f"model_{args.version}.joblib"
    meta_path = out_dir / f"model_{args.version}_meta.json"
    joblib.dump(model, model_path)
    meta = {
        "version": args.version,
        "features": FEATURES,
        "category_columns": CAT_COLS,
        "category_maps": maps,
        "card_key": CARD_KEY,
        "velocity_windows_sec": {"1h": WINDOW_1H, "24h": WINDOW_24H},
        "thresholds": {"high": high_thr, "medium": medium_thr},
        "review_cost_usd": REVIEW_COST,
        "xgb_params": {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                       for k, v in params.items()},
        "best_iteration": int(model.best_iteration),
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "time_ranges": {
            "train": [float(train["TransactionDT"].min()), float(train["TransactionDT"].max())],
            "val": [float(val["TransactionDT"].min()), float(val["TransactionDT"].max())],
            "test": [float(test["TransactionDT"].min()), float(test["TransactionDT"].max())],
        },
        "val_pr_auc": {"baseline": float(base_val_ap), "xgboost": float(xgb_val_ap)},
        "test_metrics": test_metrics,
        "test_pr_auc_baseline": base_test_ap,
        "latency_single_row": latency,
        "top_features": [[k, float(v)] for k, v in importance],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nSaved: {model_path}\nSaved: {meta_path}")

    # --- Optional MLflow logging ---
    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("fraudguard-ai")
            with mlflow.start_run(run_name=f"xgboost_{args.version}"):
                mlflow.log_params({k: v for k, v in params.items()})
                mlflow.log_metrics({
                    "val_pr_auc": float(xgb_val_ap),
                    "test_pr_auc": test_metrics["pr_auc"],
                    "test_recall": test_metrics["recall"],
                    "test_precision": test_metrics["precision"],
                    "test_recall_at_1pct_fpr": test_metrics["recall_at_1pct_fpr"],
                    "threshold_high": high_thr,
                    "latency_p95_ms": latency["p95_ms"],
                })
                mlflow.log_artifact(str(meta_path))
                mlflow.log_artifact(str(model_path))
            print("Logged run to MLflow (run `mlflow ui` to view it).")
        except ImportError:
            print("mlflow not installed - skipped.")


if __name__ == "__main__":
    main()
