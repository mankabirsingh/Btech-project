#!/usr/bin/env python3
"""
scripts/p10/juliet_leak_check.py
================================
Part A2.3: Leak check model.
Trains a logistic regression on character 3-grams of processed code with NO semantic model.
Evaluates on held-out test splits (S-flow, S-cwe, S-rand).
Reports top driving tokens (highest positive and negative coefficients).
Writes summary to results/p10/juliet_leak_check.json.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_P10_DIR = PROJECT_ROOT / "data" / "p10"
RESULTS_P10_DIR = PROJECT_ROOT / "results" / "p10"
RESULTS_P10_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RESULTS_P10_DIR / "juliet_leak_check.json"


def main():
    print("=" * 80)
    print("PART A2.3: JULIET CHARACTER 3-GRAM LEAK CHECK")
    print("=" * 80)
    t0 = time.time()

    parquet_path = DATA_P10_DIR / "juliet_extracted_40k.parquet"
    df = pd.read_parquet(parquet_path)
    print(f"[1] Loaded {len(df)} Juliet functions from {parquet_path.name}")

    splits = ["split_flow", "split_cwe", "split_rand"]
    results = {}

    for s_name in splits:
        print(f"\n[2] Evaluating Leak Check on {s_name}...")
        train_mask = (df[s_name] == "train")
        val_mask = (df[s_name] == "val")
        test_mask = (df[s_name] == "test")

        vec = CountVectorizer(analyzer="char", ngram_range=(3, 3), max_features=5000)
        X_train = vec.fit_transform(df["code"][train_mask])
        X_val = vec.transform(df["code"][val_mask])
        X_test = vec.transform(df["code"][test_mask])

        y_train = df["vuln_label"][train_mask].values
        y_val = df["vuln_label"][val_mask].values
        y_test = df["vuln_label"][test_mask].values

        clf = LogisticRegression(C=0.1, max_iter=500)
        clf.fit(X_train, y_train)

        preds_val = clf.predict(X_val)
        probs_val = clf.predict_proba(X_val)[:, 1]
        acc_val = float(accuracy_score(y_val, preds_val))
        bal_val = float(balanced_accuracy_score(y_val, preds_val))
        auc_val = float(roc_auc_score(y_val, probs_val))

        preds_test = clf.predict(X_test)
        probs_test = clf.predict_proba(X_test)[:, 1]
        acc_test = float(accuracy_score(y_test, preds_test))
        bal_test = float(balanced_accuracy_score(y_test, preds_test))
        auc_test = float(roc_auc_score(y_test, probs_test))
        f1_test = float(f1_score(y_test, preds_test))

        feature_names = np.array(vec.get_feature_names_out())
        coef = clf.coef_[0]

        top_pos_idx = np.argsort(coef)[-15:][::-1]
        top_neg_idx = np.argsort(coef)[:15]

        top_pos_ngrams = [{"ngram": feature_names[i], "coef": float(coef[i])} for i in top_pos_idx]
        top_neg_ngrams = [{"ngram": feature_names[i], "coef": float(coef[i])} for i in top_neg_idx]

        majority_baseline = float(max((y_test == 0).mean(), (y_test == 1).mean()))

        print(f"    Val Accuracy:   {acc_val:.4f} (Balanced: {bal_val:.4f}, AUROC: {auc_val:.4f})")
        print(f"    Test Accuracy:  {acc_test:.4f} (Balanced: {bal_test:.4f}, AUROC: {auc_test:.4f}, F1: {f1_test:.4f})")
        print(f"    Majority Class Baseline: {majority_baseline:.4f}")
        print("    Top positive driving ngrams (vuln):", [d["ngram"] for d in top_pos_ngrams[:5]])
        print("    Top negative driving ngrams (safe):", [d["ngram"] for d in top_neg_ngrams[:5]])

        results[s_name] = {
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_test": int(test_mask.sum()),
            "majority_baseline": majority_baseline,
            "val_accuracy": acc_val,
            "val_balanced_accuracy": bal_val,
            "val_auroc": auc_val,
            "test_accuracy": acc_test,
            "test_balanced_accuracy": bal_test,
            "test_auroc": auc_test,
            "test_f1": f1_test,
            "top_positive_ngrams": top_pos_ngrams,
            "top_negative_ngrams": top_neg_ngrams
        }

    report = {
        "status": "complete",
        "leak_check_splits": results,
        "runtime_s": float(time.time() - t0),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[3] Leak check summary report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
