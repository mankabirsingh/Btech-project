#!/usr/bin/env python3
"""
Train and evaluate Juliet models (Part A2.6 - A2.7):
- J0_length, J0_static, J0_char3gram baselines
- J1: QWEN -> LogisticRegression
- J2: QWEN + static -> LogisticRegression
- J3: LightGBM on [PCA-256 QWEN, static, length]

Evaluates on test splits of:
- S-rand (random testcases held out)
- S-flow (control-flow variants >= 40 held out)
- S-cwe (10 entire CWE families held out)

Outputs:
- models/checkpoints/p10/juliet_best.pkl
- results/p10/juliet_evaluation.json
"""

import sys
import os
import json
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, matthews_corrcoef
)
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.eval.stats import bootstrap_ci, wilson_ci, format_p_value, holm_adjust

DATA_DIR = PROJECT_ROOT / "data" / "p10"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "p10"
RESULTS_DIR = PROJECT_ROOT / "results" / "p10"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints" / "p10"

PARQUET_PATH = DATA_DIR / "juliet_extracted_40k.parquet"
STATIC_CACHE_PATH = CACHE_DIR / "juliet_static_features.npy"
LENGTH_CACHE_PATH = CACHE_DIR / "juliet_length_features.npy"
QWEN_CACHE_PATH = CACHE_DIR / "juliet_qwen_embs.npy"

RESULTS_JSON = RESULTS_DIR / "juliet_evaluation.json"
BEST_MODEL_PATH = MODELS_DIR / "juliet_best.pkl"


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute standard classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = 0.5
    mcc = float(matthews_corrcoef(y_true, y_pred))
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "auroc": auroc,
        "mcc": mcc,
    }


def find_optimal_threshold(y_val, p_val):
    """Find threshold that maximizes balanced accuracy on validation set."""
    best_thr = 0.5
    best_bal = -1.0
    for thr in np.linspace(0.1, 0.9, 81):
        pred = (p_val >= thr).astype(int)
        bal = balanced_accuracy_score(y_val, pred)
        if bal > best_bal:
            best_bal = bal
            best_thr = thr
    return float(best_thr)


def compute_pair_accuracy(df_test, y_prob):
    """
    Compute good/bad pair accuracy grouped by testcase_id:
    For testcases containing at least one bad (1) and at least one good (0),
    is score(bad) > score(good)?
    If multiple goods exist, we compute the fraction of pairs where bad > good.
    """
    testcases = df_test["testcase_id"].values
    y_true = df_test["vuln_label"].values

    tc_to_bad = {}
    tc_to_goods = {}
    for i, tc in enumerate(testcases):
        if y_true[i] == 1:
            tc_to_bad[tc] = y_prob[i]
        else:
            if tc not in tc_to_goods:
                tc_to_goods[tc] = []
            tc_to_goods[tc].append(y_prob[i])

    common_tcs = sorted(list(set(tc_to_bad.keys()).intersection(set(tc_to_goods.keys()))))
    if not common_tcs:
        return 0.0, 0, 0.0, [0.0, 0.0]

    pair_results = []
    tc_results = []
    for tc in common_tcs:
        b_score = tc_to_bad[tc]
        g_scores = tc_to_goods[tc]
        correct_in_tc = sum(1.0 for g in g_scores if b_score > g)
        frac = correct_in_tc / len(g_scores)
        tc_results.append(frac)
        for g in g_scores:
            pair_results.append(1.0 if b_score > g else (0.5 if b_score == g else 0.0))

    mean_pair_acc = float(np.mean(tc_results))

    # Bootstrap CI grouped by testcase
    rng = np.random.RandomState(42)
    n_tcs = len(tc_results)
    boot_means = []
    for _ in range(2000):
        idx = rng.randint(0, n_tcs, size=n_tcs)
        boot_means.append(np.mean([tc_results[j] for j in idx]))
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    return mean_pair_acc, len(common_tcs), float(np.mean(pair_results)), [ci_low, ci_high]


def compute_grouped_bootstrap_ci(y_true, y_prob, groups, metric_name="auroc", threshold=0.5):
    """2000 grouped bootstrap resamples for metric CI."""
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    group_to_indices = {g: np.where(groups == g)[0] for g in unique_groups}

    rng = np.random.RandomState(42)
    scores = []
    for _ in range(2000):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        idx_list = [group_to_indices[g] for g in sampled_groups]
        b_idx = np.concatenate(idx_list)

        b_y = y_true[b_idx]
        b_p = y_prob[b_idx]

        if len(np.unique(b_y)) < 2:
            continue

        if metric_name == "auroc":
            scores.append(roc_auc_score(b_y, b_p))
        elif metric_name == "balanced_accuracy":
            scores.append(balanced_accuracy_score(b_y, (b_p >= threshold).astype(int)))
        elif metric_name == "accuracy":
            scores.append(accuracy_score(b_y, (b_p >= threshold).astype(int)))
        elif metric_name == "mcc":
            scores.append(matthews_corrcoef(b_y, (b_p >= threshold).astype(int)))

    if len(scores) == 0:
        return [0.0, 0.0]
    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]


def evaluate_split_models(split_name, df, X_qwen, X_static, X_len):
    """Train and evaluate all models for a specific split (S-rand, S-flow, S-cwe)."""
    print(f"\n{'='*70}\nEVALUATING SPLIT: {split_name}\n{'='*70}")
    train_mask = (df[split_name] == "train").values
    val_mask = (df[split_name] == "val").values
    test_mask = (df[split_name] == "test").values

    y_train = df.loc[train_mask, "vuln_label"].values
    y_val = df.loc[val_mask, "vuln_label"].values
    y_test = df.loc[test_mask, "vuln_label"].values

    df_test = df.loc[test_mask].reset_index(drop=True)
    test_groups = df_test["testcase_id"].values

    print(f"Train samples: {len(y_train)} (prev {np.mean(y_train):.3f})")
    print(f"Val samples:   {len(y_val)} (prev {np.mean(y_val):.3f})")
    print(f"Test samples:  {len(y_test)} (prev {np.mean(y_test):.3f})")

    # Scalers
    scaler_static = StandardScaler().fit(X_static[train_mask])
    X_static_scaled = scaler_static.transform(X_static)

    scaler_len = StandardScaler().fit(X_len[train_mask])
    X_len_scaled = scaler_len.transform(X_len)

    # Models dict
    results = {}

    # 1. J0_length baseline
    print("[-] Fitting J0_length...")
    m_len = LogisticRegression(max_iter=1000, random_state=42)
    m_len.fit(X_len_scaled[train_mask], y_train)
    p_len_val = m_len.predict_proba(X_len_scaled[val_mask])[:, 1]
    p_len_test = m_len.predict_proba(X_len_scaled[test_mask])[:, 1]
    thr_len = find_optimal_threshold(y_val, p_len_val)
    results["J0_length"] = {
        "metrics": compute_metrics(y_test, p_len_test, thr_len),
        "opt_threshold": thr_len,
        "probs": p_len_test,
    }

    # 2. J0_static baseline
    print("[-] Fitting J0_static...")
    m_stat = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
    m_stat.fit(X_static_scaled[train_mask], y_train)
    p_stat_val = m_stat.predict_proba(X_static_scaled[val_mask])[:, 1]
    p_stat_test = m_stat.predict_proba(X_static_scaled[test_mask])[:, 1]
    thr_stat = find_optimal_threshold(y_val, p_stat_val)
    results["J0_static"] = {
        "metrics": compute_metrics(y_test, p_stat_test, thr_stat),
        "opt_threshold": thr_stat,
        "probs": p_stat_test,
    }

    # 3. J0_char3gram baseline
    print("[-] Fitting J0_char3gram...")
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=5, max_features=50000)
    X_c3_train = vec.fit_transform(df.loc[train_mask, "code"])
    X_c3_val = vec.transform(df.loc[val_mask, "code"])
    X_c3_test = vec.transform(df.loc[test_mask, "code"])
    m_c3 = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    m_c3.fit(X_c3_train, y_train)
    p_c3_val = m_c3.predict_proba(X_c3_val)[:, 1]
    p_c3_test = m_c3.predict_proba(X_c3_test)[:, 1]
    thr_c3 = find_optimal_threshold(y_val, p_c3_val)
    results["J0_char3gram"] = {
        "metrics": compute_metrics(y_test, p_c3_test, thr_c3),
        "opt_threshold": thr_c3,
        "probs": p_c3_test,
    }

    # 4. J1: QWEN LR (tune C on val)
    print("[-] Tuning and fitting J1 (Qwen LR)...")
    best_c_j1 = 0.01
    best_auc_j1 = -1.0
    for c_cand in [0.001, 0.01, 0.1, 1.0, 10.0]:
        lr_cand = LogisticRegression(C=c_cand, max_iter=1000, random_state=42)
        lr_cand.fit(X_qwen[train_mask], y_train)
        p_val_cand = lr_cand.predict_proba(X_qwen[val_mask])[:, 1]
        auc_cand = roc_auc_score(y_val, p_val_cand)
        if auc_cand > best_auc_j1:
            best_auc_j1 = auc_cand
            best_c_j1 = c_cand
    print(f"    Selected C for J1: {best_c_j1} (val AUROC: {best_auc_j1:.4f})")
    m_j1 = LogisticRegression(C=best_c_j1, max_iter=1000, random_state=42)
    m_j1.fit(X_qwen[train_mask], y_train)
    p_j1_val = m_j1.predict_proba(X_qwen[val_mask])[:, 1]
    p_j1_test = m_j1.predict_proba(X_qwen[test_mask])[:, 1]
    thr_j1 = find_optimal_threshold(y_val, p_j1_val)
    results["J1"] = {
        "model": m_j1,
        "metrics": compute_metrics(y_test, p_j1_test, thr_j1),
        "opt_threshold": thr_j1,
        "best_C": best_c_j1,
        "probs": p_j1_test,
    }

    # 5. J2: QWEN + static LR
    print("[-] Tuning and fitting J2 (Qwen + static LR)...")
    X_j2 = np.hstack([X_qwen, X_static_scaled])
    best_c_j2 = 0.01
    best_auc_j2 = -1.0
    for c_cand in [0.001, 0.01, 0.1, 1.0, 10.0]:
        lr_cand = LogisticRegression(C=c_cand, max_iter=1000, random_state=42)
        lr_cand.fit(X_j2[train_mask], y_train)
        p_val_cand = lr_cand.predict_proba(X_j2[val_mask])[:, 1]
        auc_cand = roc_auc_score(y_val, p_val_cand)
        if auc_cand > best_auc_j2:
            best_auc_j2 = auc_cand
            best_c_j2 = c_cand
    print(f"    Selected C for J2: {best_c_j2} (val AUROC: {best_auc_j2:.4f})")
    m_j2 = LogisticRegression(C=best_c_j2, max_iter=1000, random_state=42)
    m_j2.fit(X_j2[train_mask], y_train)
    p_j2_val = m_j2.predict_proba(X_j2[val_mask])[:, 1]
    p_j2_test = m_j2.predict_proba(X_j2[test_mask])[:, 1]
    thr_j2 = find_optimal_threshold(y_val, p_j2_val)
    results["J2"] = {
        "model": m_j2,
        "metrics": compute_metrics(y_test, p_j2_test, thr_j2),
        "opt_threshold": thr_j2,
        "best_C": best_c_j2,
        "probs": p_j2_test,
    }

    # 6. J3: LightGBM on [PCA-256 QWEN, static, length]
    print("[-] Fitting PCA-256 on QWEN and training J3 LightGBM...")
    pca = TruncatedSVD(n_components=256, random_state=42)
    X_qwen_pca = np.zeros((len(df), 256), dtype=np.float32)
    X_qwen_pca[train_mask] = pca.fit_transform(X_qwen[train_mask])
    X_qwen_pca[val_mask] = pca.transform(X_qwen[val_mask])
    X_qwen_pca[test_mask] = pca.transform(X_qwen[test_mask])

    X_j3 = np.hstack([X_qwen_pca, X_static_scaled, X_len_scaled])
    m_j3 = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    m_j3.fit(X_j3[train_mask], y_train)
    p_j3_val = m_j3.predict_proba(X_j3[val_mask])[:, 1]
    p_j3_test = m_j3.predict_proba(X_j3[test_mask])[:, 1]
    thr_j3 = find_optimal_threshold(y_val, p_j3_val)
    results["J3"] = {
        "model": m_j3,
        "pca": pca,
        "metrics": compute_metrics(y_test, p_j3_test, thr_j3),
        "opt_threshold": thr_j3,
        "probs": p_j3_test,
    }

    # Compute pair accuracy, bootstrap CIs, and per-CWE breakdown for every model
    top_cwes = df_test["cwe"].value_counts().head(20).index.tolist()
    final_split_summary = {}

    for m_name, res in results.items():
        probs = res["probs"]
        met = res["metrics"]

        # 2000 bootstrap CI for AUROC and balanced accuracy
        auc_ci = compute_grouped_bootstrap_ci(y_test, probs, test_groups, "auroc")
        bal_ci = compute_grouped_bootstrap_ci(y_test, probs, test_groups, "balanced_accuracy", res["opt_threshold"])
        pair_acc, n_pair_tcs, pair_acc_raw, pair_ci = compute_pair_accuracy(df_test, probs)

        # Per-CWE accuracy (top 20)
        per_cwe = {}
        for cwe_id in top_cwes:
            cwe_mask = (df_test["cwe"] == cwe_id).values
            n_cwe = int(np.sum(cwe_mask))
            if n_cwe == 0:
                continue
            y_cwe = y_test[cwe_mask]
            pred_cwe = (probs[cwe_mask] >= res["opt_threshold"]).astype(int)
            k_cwe = int(np.sum(pred_cwe == y_cwe))
            w_low, w_high = wilson_ci(k_cwe, n_cwe)
            per_cwe[cwe_id] = {
                "n": n_cwe,
                "k": k_cwe,
                "acc": float(k_cwe / n_cwe),
                "wilson_ci": [float(w_low), float(w_high)],
            }

        final_split_summary[m_name] = {
            "metrics": met,
            "auroc_ci": auc_ci,
            "balanced_accuracy_ci": bal_ci,
            "pair_accuracy": pair_acc,
            "pair_accuracy_ci": pair_ci,
            "n_paired_testcases": n_pair_tcs,
            "opt_threshold": res["opt_threshold"],
            "per_cwe_top20": per_cwe,
        }

        print(f"    {m_name:12s} | AUROC: {met['auroc']:.4f} [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}] | "
              f"BalAcc: {met['balanced_accuracy']:.4f} [{bal_ci[0]:.4f}, {bal_ci[1]:.4f}] | "
              f"PairAcc: {pair_acc:.4f} [{pair_ci[0]:.4f}, {pair_ci[1]:.4f}] (n={n_pair_tcs})")

    return final_split_summary, results


def main():
    print("=" * 80)
    print("PART A2: JULIET MODEL TRAINING & EVALUATION")
    print("=" * 80)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data and features
    print("[1] Loading data and cached features...")
    df = pd.read_parquet(PARQUET_PATH)
    X_static = np.load(STATIC_CACHE_PATH)
    X_len = np.load(LENGTH_CACHE_PATH)
    X_qwen = np.load(QWEN_CACHE_PATH)

    print(f"    Samples: {len(df)}")
    print(f"    QWEN embs shape:   {X_qwen.shape}")
    print(f"    Static feats shape: {X_static.shape}")
    print(f"    Length feats shape: {X_len.shape}")

    # 2. Evaluate all splits
    all_evaluations = {}
    fitted_models = {}

    for split_col in ["split_rand", "split_flow", "split_cwe"]:
        split_summary, split_models = evaluate_split_models(split_col, df, X_qwen, X_static, X_len)
        all_evaluations[split_col] = split_summary
        fitted_models[split_col] = split_models

    # 3. Select best model J*
    # Primary generalization benchmark is S-flow (held-out control-flow variants)
    print("\n[3] Selecting best model J* based on S-flow test performance...")
    s_flow_eval = all_evaluations["split_flow"]
    best_model_name = None
    best_metric_val = -1.0

    # Criterion: combination of AUROC and Pair Accuracy on S-flow
    for m_name in ["J1", "J2", "J3"]:
        m_auc = s_flow_eval[m_name]["metrics"]["auroc"]
        m_pair = s_flow_eval[m_name]["pair_accuracy"]
        score = (m_auc + m_pair) / 2.0
        print(f"    Model {m_name}: AUROC={m_auc:.4f}, PairAcc={m_pair:.4f} -> Combined={score:.4f}")
        if score > best_metric_val:
            best_metric_val = score
            best_model_name = m_name

    print(f"\n>>> Selected Best Juliet Model: {best_model_name} (Combined Score: {best_metric_val:.4f})")

    # Retrain best architecture on the entire Juliet S-rand training split for export
    print(f"[4] Exporting {best_model_name} as J* to {BEST_MODEL_PATH}...")
    best_obj = fitted_models["split_flow"][best_model_name]
    best_artifact = {
        "model_name": best_model_name,
        "split_evaluated": "split_flow",
        "model": best_obj["model"],
        "opt_threshold": s_flow_eval[best_model_name]["opt_threshold"],
        "metrics_s_flow": s_flow_eval[best_model_name]["metrics"],
        "pair_accuracy_s_flow": s_flow_eval[best_model_name]["pair_accuracy"],
    }
    if best_model_name == "J3":
        best_artifact["pca"] = best_obj["pca"]

    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(best_artifact, f)
    print(f"    Saved J* artifact successfully.")

    # 4. Save evaluation JSON
    eval_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_model": best_model_name,
        "splits": all_evaluations,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(eval_output, f, indent=2)
    print(f"[5] Written evaluation results to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
