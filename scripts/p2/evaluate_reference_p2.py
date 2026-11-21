"""Evaluation of reference Champion P2 recipe on held-out dev and length-bucketed test analysis."""

import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

from src.eval.locked_test import compute_file_sha256, load_locked_test
from src.eval.stats import bootstrap_ci, metrics_full, threshold_for_prevalence

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
P2_SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "p2"
RESULTS_P2_DIR = PROJECT_ROOT / "results" / "p2"


def main():
    t_start = time.time()
    RESULTS_P2_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("STEP 7: REFERENCE EVALUATION (CHAMPION P2 & LENGTH-BUCKETED TEST)")
    print("=" * 80)

    # 1. Load Parquet Datasets & SHAs
    train_parquet_path = P2_SPLITS_DIR / "train_minus_dev.parquet"
    dev_parquet_path = P2_SPLITS_DIR / "dev_heldout.parquet"

    train_df = pd.read_parquet(train_parquet_path)
    dev_df = pd.read_parquet(dev_parquet_path)
    test_df = load_locked_test(reason="Step 7: Reference evaluation of length-bucketed predictions")

    train_sha = compute_file_sha256(train_parquet_path)
    dev_sha = compute_file_sha256(dev_parquet_path)
    test_sha = "ebd95dfc837f12264dd0fd8126563d00ede27908bca38fe46611e25d7bb45bbc"

    print(f"Train samples: {len(train_df):,} | SHA-256: {train_sha[:16]}...")
    print(f"Dev samples:   {len(dev_df):,} | SHA-256: {dev_sha[:16]}...")
    print(f"Test samples:  {len(test_df):,} | SHA-256: {test_sha[:16]}...")

    # 2. Load Cached Embeddings & 28-dim Structural Features
    print("\n[1] Loading Cached Embeddings & 28-dim Features...")
    emb_train = np.load(CACHE_DIR / f"emb_codebert_256_{train_sha}.npy")
    feat28_train = np.load(CACHE_DIR / f"feat_cxx28_{train_sha}.npy")

    emb_dev = np.load(CACHE_DIR / f"emb_codebert_256_{dev_sha}.npy")
    feat28_dev = np.load(CACHE_DIR / f"feat_cxx28_{dev_sha}.npy")

    emb_test = np.load(CACHE_DIR / f"emb_codebert_256_{test_sha}.npy")
    feat28_test = np.load(CACHE_DIR / f"feat_cxx28_{test_sha}.npy")

    y_train = train_df["vuln_label"].values.astype(int)
    y_dev = dev_df["vuln_label"].values.astype(int)
    y_test = test_df["vuln_label"].values.astype(int)

    # 3. Train Champion P2 Model (Exact Phase 15 Recipe: C=0.01, balanced, scaler fit on train)
    print("\n[2] Training Reference Model 'champion_p2' on train_minus_dev...")
    scaler = StandardScaler()
    s_feat_train = scaler.fit_transform(feat28_train)
    X_train_p2 = np.hstack([emb_train, s_feat_train])

    s_feat_dev = scaler.transform(feat28_dev)
    X_dev_p2 = np.hstack([emb_dev, s_feat_dev])

    clf_p2 = LogisticRegression(C=0.01, class_weight="balanced", max_iter=1000, random_state=42)
    t0 = time.time()
    clf_p2.fit(X_train_p2, y_train)
    print(f"  Fitted Champion P2 model in {time.time()-t0:.2f}s")

    # Save reference checkpoint without overwriting existing champion
    ckpt_p2_path = MODELS_DIR / "phase15_champion_p2_reference.pkl"
    with open(ckpt_p2_path, "wb") as f:
        pickle.dump({
            "clf": clf_p2,
            "scaler": scaler,
            "feature_dim": X_train_p2.shape[1],
            "c_val": 0.01,
            "train_samples": len(X_train_p2),
            "train_sha256": train_sha,
        }, f)
    print(f"  Saved reference checkpoint to {ckpt_p2_path}")

    # 4. Predict on dev_heldout
    print("\n[3] Evaluating Champion P2 on dev_heldout...")
    prob_dev = clf_p2.predict_proba(X_dev_p2)[:, 1]

    # Threshold selection at target prevalence 0.30
    thr_info = threshold_for_prevalence(prob_dev, y_dev, target_prevalence=0.30, metric="f1")
    opt_thr = thr_info["optimal_threshold"]
    print(f"  Dev Prevalence: {y_dev.mean()*100:.2f}% | Target: 30.0% | Adapted Threshold: {opt_thr:.4f}")

    # Cohort Masks
    # C++ vs C
    is_cpp_dev = dev_df["is_cpp"].values.astype(bool) if "is_cpp" in dev_df.columns else np.array(["std::" in c or "::" in c for c in dev_df["code"]])

    # Production vs test-file code
    gtest_regex = re.compile(r"\b(EXPECT_CALL|ON_CALL|TEST_F|TEST_P|TEST|MOCK_METHOD|ASSERT_TRUE|ASSERT_FALSE|EXPECT_TRUE|EXPECT_FALSE|EXPECT_EQ|ASSERT_EQ)\b")
    path_regex = re.compile(r"(test|mock|fake)", re.IGNORECASE)
    is_test_file = np.array([
        bool(gtest_regex.search(c)) or bool(path_regex.search(str(s)))
        for c, s in zip(dev_df["code"].values, dev_df["sample_id"].values)
    ])

    cohorts = {
        "Overall": np.ones(len(y_dev), dtype=bool),
        "C++ Subset": is_cpp_dev,
        "C Subset": ~is_cpp_dev,
        "Production Code": ~is_test_file,
        "Test/Mock Code": is_test_file,
    }

    dev_results = {}
    for c_name, mask in cohorts.items():
        y_c = y_dev[mask]
        p_c = prob_dev[mask]

        m_050 = metrics_full(y_c, p_c, threshold=0.50)
        ci_050 = bootstrap_ci(y_c, p_c, threshold=0.50, n_resamples=1000, seed=42)

        m_opt = metrics_full(y_c, p_c, threshold=opt_thr)
        ci_opt = bootstrap_ci(y_c, p_c, threshold=opt_thr, n_resamples=1000, seed=42)

        dev_results[c_name] = {
            "n": int(len(y_c)),
            "n_positive": int(np.sum(y_c)),
            "prevalence": float(np.mean(y_c)),
            "threshold_050": {**m_050, **ci_050},
            "threshold_adapted_030": {**m_opt, **ci_opt},
        }

    # Print Dev Table
    print("\n" + "=" * 90)
    print("CHAMPION P2 PERFORMANCE ON HELD-OUT DEV SET (dev_heldout.parquet)")
    print("=" * 90)
    print(f"{'Cohort':<18} | {'N':<6} {'Prev':<7} | {'AUROC [95% CI]':<26} | {'Acc (t=0.50)':<14} {'FPR':<8} {'F1':<8}")
    print("-" * 90)
    for c_name, res in dev_results.items():
        n = res["n"]
        prev = res["prevalence"] * 100
        auc = res["threshold_050"]["auroc"]
        auc_ci = res["threshold_050"]["auroc_ci_95"]
        acc = res["threshold_050"]["accuracy"] * 100
        fpr = res["threshold_050"]["fpr"] * 100
        f1 = res["threshold_050"]["f1"]
        print(f"{c_name:<18} | {n:<6} {prev:>5.1f}% | {auc:.4f} [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}] | {acc:>6.2f}%       {fpr:>5.2f}%   {f1:>5.3f}")

    # 5. Length-Bucketed Analysis of Existing Phase 15 Champion on Clean Test
    print("\n[4] Computing Length-Bucketed Performance for Existing Champion on Test Set...")
    with open(MODELS_DIR / "phase15_codebert_cxx_hybrid_best.pkl", "rb") as f:
        ckpt_p15 = pickle.load(f)

    s_feat_test = ckpt_p15["scaler"].transform(feat28_test)
    X_test_p15 = np.hstack([emb_test, s_feat_test])
    prob_champ_test = ckpt_p15["clf"].predict_proba(X_test_p15)[:, 1]

    # Compute CodeBERT token lengths for all test snippets
    print("  Tokenizing test code to get exact token lengths...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    test_token_lens = []
    for c in test_df["code"].values:
        toks = tokenizer.encode(str(c), add_special_tokens=False, truncation=False)
        test_token_lens.append(len(toks))
    test_token_lens = np.array(test_token_lens)

    length_buckets = {
        "<= 256": (test_token_lens <= 256),
        "257 - 512": (test_token_lens >= 257) & (test_token_lens <= 512),
        "513 - 1024": (test_token_lens >= 513) & (test_token_lens <= 1024),
        "> 1024": (test_token_lens > 1024),
    }

    bucket_results = {}
    print("\n" + "=" * 90)
    print("PHASE 15 CHAMPION HELD-OUT TEST PERFORMANCE BY FUNCTION TOKEN LENGTH")
    print("=" * 90)
    print(f"{'Length Bucket':<14} | {'N':<6} {'% Test':<8} {'Prev':<7} | {'AUROC [95% CI]':<26} | {'Accuracy':<10} {'FPR':<8} {'F1':<8}")
    print("-" * 90)

    for b_name, mask in length_buckets.items():
        y_b = y_test[mask]
        p_b = prob_champ_test[mask]
        n_b = len(y_b)
        pct_b = float(n_b / len(y_test) * 100.0)
        prev_b = float(np.mean(y_b))

        m_b = metrics_full(y_b, p_b, threshold=0.50)
        ci_b = bootstrap_ci(y_b, p_b, threshold=0.50, n_resamples=1000, seed=42)

        auc = m_b["auroc"]
        auc_ci = ci_b["auroc_ci_95"]
        acc = m_b["accuracy"] * 100
        fpr = m_b["fpr"] * 100
        f1 = m_b["f1"]

        print(f"{b_name:<14} | {n_b:<6} {pct_b:>6.1f}%  {prev_b*100:>5.1f}% | {auc:.4f} [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}] | {acc:>6.2f}%    {fpr:>5.2f}%   {f1:>5.3f}")

        bucket_results[b_name] = {
            "n": n_b,
            "pct_of_test": pct_b,
            "n_positive": int(np.sum(y_b)),
            "prevalence": prev_b,
            "metrics": {**m_b, **ci_b},
        }

    # 6. Save Complete Reference Results
    output_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "runtime_seconds": round(time.time() - t_start, 2),
            "dev_sha256": dev_sha,
            "train_sha256": train_sha,
            "test_sha256": test_sha,
        },
        "threshold_tuning": thr_info,
        "champion_p2_on_dev": dev_results,
        "champion_on_test_by_token_length": bucket_results,
    }

    ref_json_path = RESULTS_P2_DIR / "reference_numbers.json"
    with open(ref_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved reference results to {ref_json_path}")


if __name__ == "__main__":
    main()
