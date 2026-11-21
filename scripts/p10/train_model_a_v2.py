#!/usr/bin/env python3
"""
Train and evaluate Real-World Model A v2 (Part A1.4 - A1.8):
Candidates:
- Baselines: L1 (log length), champion recipe (Phase 15 C=0.01), static-only
- V1: QWEN -> LogisticRegression
- V2: QWEN + CodeBERT -> LogisticRegression
- V3: V2 + static -> LogisticRegression
- V4: V3 + Pairwise ranking on train_pairs_v2
- V5: LightGBM on [PCA-256 V2, static, length]
- VM: Mixed training (V3 on train_v2 [70%] + Juliet S-rand [30%])
- V6: Triage stacker on [selected_model logit, log n_tokens]

Evaluates on:
- dev_heldout (2,543 samples): AUROC, LS-AUROC 10-bin, length-matched AUROC, pair_correct on dev_pairs, C++ subset
- Operating point: top 10% alert budget
- Freeze protocol: results/p10/freeze_a.json
- Final Evaluation (locked loaders called ONCE):
  * Primary: test2.parquet via load_locked_test2(reason="P10 final")
  * Secondary: old test via load_locked_test(reason="P10 secondary, viewed in P6")
  * Cross: Juliet S-flow test split
"""

import sys
import os
import json
import time
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.eval.stats import bootstrap_ci, wilson_ci, format_p_value, holm_adjust
from src.eval.stats_v2 import (
    calc_lsauroc10,
    calc_length_matched_auroc,
    paired_bootstrap_diff_v2,
    get_length_matched_pair_indices
)
from src.eval.locked_test import load_locked_test, load_locked_test2

DATA_DIR = PROJECT_ROOT / "data"
P10_DATA_DIR = DATA_DIR / "p10"
P2_SPLITS_DIR = DATA_DIR / "splits" / "p2"
P4_SPLITS_DIR = DATA_DIR / "splits" / "p4"
CACHE_P10_DIR = DATA_DIR / "cache" / "p10"
CACHE_P2_DIR = DATA_DIR / "cache"
RESULTS_DIR = PROJECT_ROOT / "results" / "p10"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints" / "p10"

DEV_PARQUET = P2_SPLITS_DIR / "dev_heldout.parquet"
DEV_PAIRS_PARQUET = P4_SPLITS_DIR / "dev_pairs.parquet"
TRAIN_V2_PARQUET = P10_DATA_DIR / "train_v2.parquet"
TRAIN_PAIRS_V2_PARQUET = P10_DATA_DIR / "train_pairs_v2.parquet"
JULIET_PARQUET = P10_DATA_DIR / "juliet_extracted_40k.parquet"

FREEZE_JSON = RESULTS_DIR / "freeze_a.json"
EVAL_JSON = RESULTS_DIR / "model_a_v2_evaluation.json"
BEST_MODEL_PKL = MODELS_DIR / "model_a_v2_best.pkl"
STACKER_PKL = MODELS_DIR / "model_a_v2_stacker.pkl"

TOP_CWES_BENCHMARK = ["476", "400", "416", "787", "119"]


def compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_operating_point_metrics(y_true, y_prob, cwe_series, top_pct=0.10):
    """Evaluate performance when flagging top 10% highest scored samples."""
    n = len(y_true)
    k_flag = max(1, int(n * top_pct))
    cutoff_thr = float(np.partition(y_prob, -k_flag)[-k_flag])
    flagged = (y_prob >= cutoff_thr).astype(int)

    prev = float(np.mean(y_true))
    prec = float(precision_score(y_true, flagged, zero_division=0))
    rec = float(recall_score(y_true, flagged, zero_division=0))
    lift = prec / max(prev, 1e-6)

    # Per-CWE recall in top 10%
    cwe_recalls = {}
    cwe_strs = cwe_series.fillna("").astype(str).str.lower().values
    for target_cwe in TOP_CWES_BENCHMARK:
        mask_cwe = np.array([target_cwe in c for c in cwe_strs]) & (y_true == 1)
        n_cwe = int(np.sum(mask_cwe))
        if n_cwe > 0:
            k_cwe_flagged = int(np.sum(flagged[mask_cwe]))
            w_low, w_high = wilson_ci(k_cwe_flagged, n_cwe)
            cwe_recalls[f"CWE-{target_cwe}"] = {
                "n": n_cwe,
                "k_flagged": k_cwe_flagged,
                "recall": float(k_cwe_flagged / n_cwe),
                "wilson_ci": [float(w_low), float(w_high)],
            }

    return {
        "cutoff_threshold": cutoff_thr,
        "flagged_count": int(np.sum(flagged)),
        "precision": prec,
        "recall": rec,
        "lift": lift,
        "per_cwe_recall": cwe_recalls,
    }


def compute_pair_correct_on_pairs(p_dev, dev_id_to_idx, df_pairs):
    """Compute pair accuracy on disjoint dev pairs."""
    v_preds = []
    f_preds = []
    for _, r in df_pairs.iterrows():
        v_sid = r["vuln_sample_id"]
        f_sid = r["fixed_sample_id"]
        if v_sid in dev_id_to_idx and f_sid in dev_id_to_idx:
            v_preds.append(p_dev[dev_id_to_idx[v_sid]])
            f_preds.append(p_dev[dev_id_to_idx[f_sid]])

    v_preds = np.array(v_preds)
    f_preds = np.array(f_preds)
    n_pairs = len(v_preds)
    if n_pairs == 0:
        return 0.0, [0.0, 0.0], 0

    correct = (v_preds > f_preds).astype(float)
    pair_acc = float(np.mean(correct))

    # Bootstrap CI
    rng = np.random.RandomState(42)
    boot_means = [np.mean(rng.choice(correct, size=n_pairs, replace=True)) for _ in range(2000)]
    ci = [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))]

    return pair_acc, ci, n_pairs


def compute_length_matched_metrics(y_true, y_prob, n_tokens, tol=0.1):
    """Compute length-matched AUROC and count of matching pairs."""
    v_idx, s_idx = get_length_matched_pair_indices(y_true, n_tokens, max_log_diff=tol)
    n_pairs = len(v_idx)
    if n_pairs == 0:
        return 0.5, 0
    auc = calc_length_matched_auroc(y_true, y_prob, n_tokens, max_log_diff=tol, pair_indices=(v_idx, s_idx))
    return float(auc), int(n_pairs)


def evaluate_on_dev(y_dev, p_dev, tok_dev, is_cpp, commits, cwes, dev_id_to_idx, df_dev_pairs):
    """Full comprehensive dev evaluation for a candidate."""
    auc = float(roc_auc_score(y_dev, p_dev))
    ls_auc = float(calc_lsauroc10(y_dev, p_dev, tok_dev))
    lm_auc, lm_n_pairs = compute_length_matched_metrics(y_dev, p_dev, tok_dev, tol=0.1)

    # C++ subset
    if np.sum(is_cpp) >= 20 and np.sum(y_dev[is_cpp]) >= 10:
        cpp_lm_auc, _ = compute_length_matched_metrics(y_dev[is_cpp], p_dev[is_cpp], tok_dev[is_cpp], tol=0.1)
    else:
        cpp_lm_auc = 0.5

    # Pair correct
    pair_acc, pair_ci, n_dev_p = compute_pair_correct_on_pairs(p_dev, dev_id_to_idx, df_dev_pairs)

    # Top 10% operating point
    op_metrics = compute_operating_point_metrics(y_dev, p_dev, cwes, top_pct=0.10)

    # Bootstrap CIs (grouped by commit)
    auc_ci = bootstrap_ci(y_dev, p_dev, metric_fn=lambda y, p, _: roc_auc_score(y, p), groups=commits, n_resamples=2000, seed=42)
    lm_ci = bootstrap_metric(
        y_dev,
        p_dev,
        tok_dev,
        metric_fn=lambda y, p, tok: calc_length_matched_auroc(y, p, tok, max_log_diff=0.1),
        groups=commits,
        n_resamples=2000,
        seed=42,
    )

    return {
        "auroc": auc,
        "auroc_ci": auc_ci,
        "ls_auroc_10bin": ls_auc,
        "length_matched_auroc": lm_auc,
        "length_matched_auroc_ci": lm_ci,
        "length_matched_pairs": lm_n_pairs,
        "cpp_length_matched_auroc": cpp_lm_auc,
        "pair_correct": pair_acc,
        "pair_correct_ci": pair_ci,
        "n_dev_pairs": n_dev_p,
        "operating_point_10pct": op_metrics,
    }


def predict_candidate(cand_name, obj, X_qwen, X_cb, X_stat_s, X_len_s):
    """Unified prediction dispatcher for all candidate architectures."""
    if cand_name == "L1":
        return obj.predict_proba(X_len_s[:, :1])[:, 1]
    elif cand_name == "champion_faithful":
        return obj.predict_proba(X_cb)[:, 1]
    elif cand_name == "static_only":
        return obj.predict_proba(X_stat_s)[:, 1]
    elif cand_name == "V1":
        return obj.predict_proba(X_qwen)[:, 1]
    elif cand_name == "V2":
        X = np.hstack([X_qwen, X_cb])
        return obj.predict_proba(X)[:, 1]
    elif cand_name in ("V3", "VM"):
        X = np.hstack([X_qwen, X_cb, X_stat_s])
        return obj.predict_proba(X)[:, 1]
    elif cand_name == "V4":
        X = np.hstack([X_qwen, X_cb, X_stat_s])
        raw = X @ obj["rank_model"].coef_[0]
        return obj["calibrator"].predict_proba(raw.reshape(-1, 1))[:, 1]
    elif cand_name == "V5":
        X_v2 = np.hstack([X_qwen, X_cb])
        pca_feat = obj["pca"].transform(X_v2)
        X = np.hstack([pca_feat, X_stat_s, X_len_s])
        return obj["model"].predict_proba(X)[:, 1]
    else:
        raise ValueError(f"Unknown candidate {cand_name}")


def bootstrap_metric(y_true, p_prob, n_tokens, metric_fn, groups=None, n_resamples=2000, seed=42):
    """Compute bootstrap confidence interval for arbitrary metric with token metadata and optional group clustering."""
    rng = np.random.RandomState(seed)
    y_t = np.asarray(y_true).astype(int)
    p_p = np.asarray(p_prob).astype(float)
    toks = np.asarray(n_tokens).astype(float)
    if groups is not None:
        groups_arr = np.asarray(groups)
        unique_groups = np.unique(groups_arr)
        group_to_indices = {g: np.where(groups_arr == g)[0] for g in unique_groups}
    else:
        pos_idx = np.where(y_t == 1)[0]
        neg_idx = np.where(y_t == 0)[0]

    scores = []
    for _ in range(n_resamples):
        if groups is not None:
            chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            b_idx = np.concatenate([group_to_indices[g] for g in chosen])
        else:
            b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True) if len(pos_idx) > 0 else np.array([], dtype=int)
            b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True) if len(neg_idx) > 0 else np.array([], dtype=int)
            b_idx = np.concatenate([b_pos, b_neg])
        if len(b_idx) == 0:
            continue
        y_b = y_t[b_idx]
        if len(np.unique(y_b)) < 2:
            continue
        p_b = p_p[b_idx]
        tok_b = toks[b_idx]
        try:
            val = metric_fn(y_b, p_b, tok_b)
            scores.append(val)
        except Exception:
            continue
    if len(scores) == 0:
        return [0.0, 0.0]
    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]


def evaluate_dataset_full(y_true, p_prob, n_tokens, is_cpp, commits, cwes, prefix=""):
    """Compute AUROC, LS-AUROC 10-bin, length-matched AUROC, C++ LM-AUROC, and operating point with grouped bootstrap CIs."""
    auc = float(roc_auc_score(y_true, p_prob))
    ls_auc = float(calc_lsauroc10(y_true, p_prob, n_tokens))
    lm_auc, lm_n_pairs = compute_length_matched_metrics(y_true, p_prob, n_tokens, tol=0.1)

    # C++ LM-AUROC
    if np.sum(is_cpp) >= 20 and np.sum(y_true[is_cpp]) >= 10:
        cpp_lm_auc, _ = compute_length_matched_metrics(y_true[is_cpp], p_prob[is_cpp], n_tokens[is_cpp], tol=0.1)
    else:
        cpp_lm_auc = 0.5

    # Top 10% operating point
    op_metrics = compute_operating_point_metrics(y_true, p_prob, cwes, top_pct=0.10)

    # 2000 bootstrap resamples grouped by commit
    auc_ci = bootstrap_ci(
        y_true, p_prob, metric_fn=lambda y, p, _: roc_auc_score(y, p), groups=commits, n_resamples=2000, seed=42
    )

    # Grouped bootstrap CI for length-matched AUROC
    lm_ci = bootstrap_metric(
        y_true,
        p_prob,
        n_tokens,
        metric_fn=lambda y, p, tok: calc_length_matched_auroc(y, p, tok, max_log_diff=0.1),
        groups=commits,
        n_resamples=2000,
        seed=42,
    )

    return {
        "auroc": auc,
        "auroc_ci": auc_ci,
        "ls_auroc_10bin": ls_auc,
        "length_matched_auroc": lm_auc,
        "length_matched_auroc_ci": lm_ci,
        "length_matched_pairs": lm_n_pairs,
        "cpp_length_matched_auroc": cpp_lm_auc,
        "operating_point_10pct": op_metrics,
    }


def main():
    print("=" * 80)
    print("PART A1: TRAINING REAL-WORLD MODEL A V2 & DEV EVALUATION")
    print("=" * 80)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Datasets
    print("[1] Loading datasets...")
    df_dev = pd.read_parquet(DEV_PARQUET)
    df_tr = pd.read_parquet(TRAIN_V2_PARQUET)
    df_dev_pairs = pd.read_parquet(DEV_PAIRS_PARQUET)
    df_tr_pairs = pd.read_parquet(TRAIN_PAIRS_V2_PARQUET)
    df_j = pd.read_parquet(JULIET_PARQUET)

    y_dev = df_dev["vuln_label"].values.astype(int)
    y_tr = df_tr["vuln_label"].values.astype(int)

    commits_dev = df_dev["commit_id"].values
    cwes_dev = df_dev["cwe"]
    dev_id_to_idx = {sid: i for i, sid in enumerate(df_dev["sample_id"])}

    # C++ indicator in dev
    if "is_cpp" in df_dev.columns:
        is_cpp_dev = df_dev["is_cpp"].astype(bool).values
    else:
        is_cpp_dev = np.array([
            "::" in c or "std::" in c or "template<" in c or "namespace " in c
            for c in df_dev["code"]
        ])

    # 2. Load Features
    print("[2] Loading cached features...")
    dev_len = np.load(CACHE_P10_DIR / "dev_length_features.npy")
    dev_stat = np.load(CACHE_P10_DIR / "dev_static_features.npy")
    dev_qwen = np.load(CACHE_P10_DIR / "dev_qwen_embs.npy")
    dev_cb = np.load(CACHE_P10_DIR / "dev_codebert_embs.npy")

    tr_len = np.load(CACHE_P10_DIR / "train_v2_length_features.npy")
    tr_stat = np.load(CACHE_P10_DIR / "train_v2_static_features.npy")
    tr_qwen = np.load(CACHE_P10_DIR / "train_v2_qwen_embs.npy")
    tr_cb = np.load(CACHE_P10_DIR / "train_v2_codebert_embs.npy")

    tok_dev = np.exp(dev_len[:, 0]) - 1.0

    print(f"    dev features: Qwen {dev_qwen.shape}, CodeBERT {dev_cb.shape}, Static {dev_stat.shape}, Len {dev_len.shape}")
    print(f"    train_v2 features: Qwen {tr_qwen.shape}, CodeBERT {tr_cb.shape}, Static {tr_stat.shape}, Len {tr_len.shape}")

    # Scalers
    scaler_stat = StandardScaler().fit(tr_stat)
    tr_stat_s = scaler_stat.transform(tr_stat)
    dev_stat_s = scaler_stat.transform(dev_stat)

    scaler_len = StandardScaler().fit(tr_len)
    tr_len_s = scaler_len.transform(tr_len)
    dev_len_s = scaler_len.transform(dev_len)

    # Feature representations
    # V1: Qwen
    # V2: [Qwen, CodeBERT]
    # V3: [V2, static]
    tr_v2 = np.hstack([tr_qwen, tr_cb])
    dev_v2 = np.hstack([dev_qwen, dev_cb])

    tr_v3 = np.hstack([tr_v2, tr_stat_s])
    dev_v3 = np.hstack([dev_v2, dev_stat_s])

    CAND_CACHE_PATH = CACHE_P10_DIR / "model_a_v2_candidates_cache.pkl"
    if CAND_CACHE_PATH.exists():
        print(f"\n[-] Loading cached candidate models from {CAND_CACHE_PATH.name}...")
        with open(CAND_CACHE_PATH, "rb") as f:
            cand_cache = pickle.load(f)
        candidates = cand_cache["candidates"]
        preds_dev = cand_cache["preds_dev"]
        m_l1 = candidates["L1"]
        m_champ = candidates["champion_faithful"]
        m_stat = candidates["static_only"]
        m_v1 = candidates["V1"]
        m_v2 = candidates["V2"]
        m_v3 = candidates["V3"]
        m_v4_obj = candidates["V4"]
        m_v5 = candidates["V5"]
        m_vm = candidates["VM"]
        best_c_v3 = cand_cache.get("best_c_v3", 10.0)
    else:
        candidates = {}
        preds_dev = {}

        # -------------------------------------------------------------------------
        # Baselines
        # -------------------------------------------------------------------------
        print("\n[-] Training Baselines...")
        # L1: log tokens
        m_l1 = LogisticRegression(random_state=42)
        m_l1.fit(tr_len[:, :1], y_tr)
        p_l1 = m_l1.predict_proba(dev_len[:, :1])[:, 1]
        candidates["L1"] = m_l1
        preds_dev["L1"] = p_l1

        # Champion faithful: CodeBERT C=0.01
        m_champ = LogisticRegression(C=0.01, max_iter=1000, random_state=42)
        m_champ.fit(tr_cb, y_tr)
        p_champ = m_champ.predict_proba(dev_cb)[:, 1]
        candidates["champion_faithful"] = m_champ
        preds_dev["champion_faithful"] = p_champ

        # Static only:
        m_stat = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        m_stat.fit(tr_stat_s, y_tr)
        p_stat = m_stat.predict_proba(dev_stat_s)[:, 1]
        candidates["static_only"] = m_stat
        preds_dev["static_only"] = p_stat

        # -------------------------------------------------------------------------
        # Proposed Candidates
        # -------------------------------------------------------------------------
        # V1: QWEN LR (tune C on dev length-matched AUROC)
        print("\n[-] Training V1 (QWEN LR)...")
        best_c_v1, best_lm_v1 = 0.01, -1.0
        for c_cand in [0.001, 0.01, 0.1, 1.0, 10.0]:
            lr = LogisticRegression(C=c_cand, max_iter=1000, random_state=42)
            lr.fit(tr_qwen, y_tr)
            p_cand = lr.predict_proba(dev_qwen)[:, 1]
            lm_auc = calc_length_matched_auroc(y_dev, p_cand, tok_dev, max_log_diff=0.1)
            if lm_auc > best_lm_v1:
                best_lm_v1 = lm_auc
                best_c_v1 = c_cand
        m_v1 = LogisticRegression(C=best_c_v1, max_iter=1000, random_state=42)
        m_v1.fit(tr_qwen, y_tr)
        p_v1 = m_v1.predict_proba(dev_qwen)[:, 1]
        candidates["V1"] = m_v1
        preds_dev["V1"] = p_v1
        print(f"    Selected C for V1: {best_c_v1} (Dev LM-AUROC: {best_lm_v1:.4f})")

        # V2: QWEN + CodeBERT LR
        print("\n[-] Training V2 (QWEN + CodeBERT LR)...")
        best_c_v2, best_lm_v2 = 0.01, -1.0
        for c_cand in [0.001, 0.01, 0.1, 1.0, 10.0]:
            lr = LogisticRegression(C=c_cand, max_iter=1000, random_state=42)
            lr.fit(tr_v2, y_tr)
            p_cand = lr.predict_proba(dev_v2)[:, 1]
            lm_auc = calc_length_matched_auroc(y_dev, p_cand, tok_dev, max_log_diff=0.1)
            if lm_auc > best_lm_v2:
                best_lm_v2 = lm_auc
                best_c_v2 = c_cand
        m_v2 = LogisticRegression(C=best_c_v2, max_iter=1000, random_state=42)
        m_v2.fit(tr_v2, y_tr)
        p_v2 = m_v2.predict_proba(dev_v2)[:, 1]
        candidates["V2"] = m_v2
        preds_dev["V2"] = p_v2
        print(f"    Selected C for V2: {best_c_v2} (Dev LM-AUROC: {best_lm_v2:.4f})")

        # V3: V2 + Static LR
        print("\n[-] Training V3 (V2 + Static LR)...")
        best_c_v3, best_lm_v3 = 0.01, -1.0
        for c_cand in [0.001, 0.01, 0.1, 1.0, 10.0]:
            lr = LogisticRegression(C=c_cand, max_iter=1000, random_state=42)
            lr.fit(tr_v3, y_tr)
            p_cand = lr.predict_proba(dev_v3)[:, 1]
            lm_auc = calc_length_matched_auroc(y_dev, p_cand, tok_dev, max_log_diff=0.1)
            if lm_auc > best_lm_v3:
                best_lm_v3 = lm_auc
                best_c_v3 = c_cand
        m_v3 = LogisticRegression(C=best_c_v3, max_iter=1000, random_state=42)
        m_v3.fit(tr_v3, y_tr)
        p_v3 = m_v3.predict_proba(dev_v3)[:, 1]
        candidates["V3"] = m_v3
        preds_dev["V3"] = p_v3
        print(f"    Selected C for V3: {best_c_v3} (Dev LM-AUROC: {best_lm_v3:.4f})")

        # V4: Pairwise ranking on train_pairs_v2
        print("\n[-] Training V4 (Pairwise Ranking Model on train_pairs_v2)...")
        # Identify pairs where both functions are in train_v2
        tr_id_to_idx = {sid: i for i, sid in enumerate(df_tr["sample_id"])}
        pair_diffs = []
        for _, r in df_tr_pairs.iterrows():
            vsid = r["vuln_sample_id"]
            fsid = r["fixed_sample_id"]
            if vsid in tr_id_to_idx and fsid in tr_id_to_idx:
                diff = tr_v3[tr_id_to_idx[vsid]] - tr_v3[tr_id_to_idx[fsid]]
                pair_diffs.append(diff)

        if len(pair_diffs) >= 50:
            pair_diffs = np.array(pair_diffs, dtype=np.float32)
            # Symmetrize: diff -> 1, -diff -> 0
            X_pair = np.vstack([pair_diffs, -pair_diffs])
            y_pair = np.concatenate([np.ones(len(pair_diffs)), np.zeros(len(pair_diffs))])

            m_rank = LogisticRegression(fit_intercept=False, C=1.0, max_iter=1000, random_state=42)
            m_rank.fit(X_pair, y_pair)

            # Calibrate intercept on train_v2 logits
            raw_tr_scores = tr_v3 @ m_rank.coef_[0]
            lr_cal = LogisticRegression(fit_intercept=True, random_state=42)
            lr_cal.fit(raw_tr_scores.reshape(-1, 1), y_tr)

            raw_dev_scores = dev_v3 @ m_rank.coef_[0]
            p_v4 = lr_cal.predict_proba(raw_dev_scores.reshape(-1, 1))[:, 1]
            m_v4_obj = {"rank_model": m_rank, "calibrator": lr_cal}
        else:
            print("    Insufficient overlapping pairs in train_v2; falling back V4 to V3.")
            p_v4 = p_v3
            m_v4_obj = m_v3
        candidates["V4"] = m_v4_obj
        preds_dev["V4"] = p_v4

        # V5: LightGBM on PCA-256 V2 + static + length
        print("\n[-] Training V5 (LightGBM on PCA-256 V2 + static + length)...")
        pca = TruncatedSVD(n_components=256, random_state=42)
        tr_v2_pca = pca.fit_transform(tr_v2)
        dev_v2_pca = pca.transform(dev_v2)

        tr_v5 = np.hstack([tr_v2_pca, tr_stat_s, tr_len_s])
        dev_v5 = np.hstack([dev_v2_pca, dev_stat_s, dev_len_s])

        m_v5 = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1)
        m_v5.fit(tr_v5, y_tr)
        p_v5 = m_v5.predict_proba(dev_v5)[:, 1]
        candidates["V5"] = {"model": m_v5, "pca": pca}
        preds_dev["V5"] = p_v5

        # VM: Mixed training (V3 features on train_v2 [70%] + Juliet S-rand [30%])
        print("\n[-] Training VM (Mixed Real + Juliet Grounding)...")
        # Load Juliet S-rand train split
        df_j = pd.read_parquet(JULIET_PARQUET)
        j_train_mask = (df_j["split_rand"] == "train").values
        j_qwen = np.load(CACHE_P10_DIR / "juliet_qwen_embs.npy")[j_train_mask]
        j_stat = np.load(CACHE_P10_DIR / "juliet_static_features.npy")[j_train_mask]
        j_stat_s = scaler_stat.transform(j_stat)
        j_y = df_j.loc[j_train_mask, "vuln_label"].values

        # Juliet has no CodeBERT embeddings; use zeros for CodeBERT portion
        j_cb_zeros = np.zeros((len(j_qwen), 1536), dtype=np.float32)
        j_v3 = np.hstack([j_qwen, j_cb_zeros, j_stat_s])

        # Sample weights: 70% real train_v2, 30% Juliet
        w_tr = np.full(len(tr_v3), 0.70 / len(tr_v3))
        w_j = np.full(len(j_v3), 0.30 / len(j_v3))

        X_vm = np.vstack([tr_v3, j_v3])
        y_vm = np.concatenate([y_tr, j_y])
        w_vm = np.concatenate([w_tr, w_j])

        m_vm = LogisticRegression(C=best_c_v3, max_iter=1000, random_state=42)
        m_vm.fit(X_vm, y_vm, sample_weight=w_vm)
        p_vm = m_vm.predict_proba(dev_v3)[:, 1]
        candidates["VM"] = m_vm
        preds_dev["VM"] = p_vm

        # Save candidate cache
        with open(CAND_CACHE_PATH, "wb") as f:
            pickle.dump({
                "candidates": candidates,
                "preds_dev": preds_dev,
                "best_c_v3": best_c_v3,
            }, f)
        print(f"    Saved candidate models cache to {CAND_CACHE_PATH.name}")

    # -------------------------------------------------------------------------
    # Dev Evaluation of All Candidates
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("DEV EVALUATION & STATISTICAL COMPARISON")
    print("=" * 90)

    DEV_EVAL_CACHE_PATH = CACHE_P10_DIR / "model_a_v2_dev_eval_cache.pkl"
    comp_models = [c for c in candidates.keys() if c not in ("L1", "champion_faithful")]

    if DEV_EVAL_CACHE_PATH.exists():
        print(f"\n[-] Loading cached dev evaluation from {DEV_EVAL_CACHE_PATH.name}...")
        with open(DEV_EVAL_CACHE_PATH, "rb") as f:
            dev_eval_cache = pickle.load(f)
        dev_results = dev_eval_cache["dev_results"]
        diffs_vs_l1 = dev_eval_cache["diffs_vs_l1"]
        diffs_vs_champ = dev_eval_cache["diffs_vs_champ"]
        raw_p_vs_l1 = dev_eval_cache["raw_p_vs_l1"]
        raw_p_vs_champ = dev_eval_cache["raw_p_vs_champ"]
    else:
        dev_results = {}
        for c_name, p_cand in preds_dev.items():
            dev_results[c_name] = evaluate_on_dev(
                y_dev, p_cand, tok_dev, is_cpp_dev, commits_dev, cwes_dev, dev_id_to_idx, df_dev_pairs
            )

        # Paired Bootstrap Differences vs L1 and vs champion on Length-Matched AUROC
        print("\nComputing paired bootstrap tests vs L1 and champion (2000 commit-grouped resamples)...")
        raw_p_vs_l1 = {}
        raw_p_vs_champ = {}
        diffs_vs_l1 = {}
        diffs_vs_champ = {}

        for c_name in comp_models:
            p_c = preds_dev[c_name]
            diff_l1 = paired_bootstrap_diff_v2(
                y_dev,
                preds_dev["L1"],
                p_c,
                metric_fn=calc_length_matched_auroc,
                groups=commits_dev,
                extra_arrays={"n_tokens": tok_dev},
                n_resamples=2000,
                seed=42,
            )
            diff_champ = paired_bootstrap_diff_v2(
                y_dev,
                preds_dev["champion_faithful"],
                p_c,
                metric_fn=calc_length_matched_auroc,
                groups=commits_dev,
                extra_arrays={"n_tokens": tok_dev},
                n_resamples=2000,
                seed=42,
            )
            raw_p_vs_l1[c_name] = diff_l1["raw_p_value"]
            raw_p_vs_champ[c_name] = diff_champ["raw_p_value"]
            diffs_vs_l1[c_name] = diff_l1
            diffs_vs_champ[c_name] = diff_champ

        with open(DEV_EVAL_CACHE_PATH, "wb") as f:
            pickle.dump({
                "dev_results": dev_results,
                "diffs_vs_l1": diffs_vs_l1,
                "diffs_vs_champ": diffs_vs_champ,
                "raw_p_vs_l1": raw_p_vs_l1,
                "raw_p_vs_champ": raw_p_vs_champ,
            }, f)
        print(f"    Saved dev evaluation cache to {DEV_EVAL_CACHE_PATH.name}")

    # Holm adjustment
    comp_names = list(comp_models)
    adj_l1 = holm_adjust([float(raw_p_vs_l1[k]) for k in comp_names])
    adj_champ = holm_adjust([float(raw_p_vs_champ[k]) for k in comp_names])
    holm_p_l1 = dict(zip(comp_names, [float(x) for x in adj_l1]))
    holm_p_champ = dict(zip(comp_names, [float(x) for x in adj_champ]))

    for c_name in comp_models:
        dev_results[c_name]["diff_vs_l1"] = {
            "diff_observed": diffs_vs_l1[c_name]["diff_observed"],
            "ci_95": [diffs_vs_l1[c_name]["ci_lower"], diffs_vs_l1[c_name]["ci_upper"]],
            "p_raw": raw_p_vs_l1[c_name],
            "p_holm": holm_p_l1[c_name],
            "p_holm_fmt": format_p_value(holm_p_l1[c_name]),
        }
        dev_results[c_name]["diff_vs_champ"] = {
            "diff_observed": diffs_vs_champ[c_name]["diff_observed"],
            "ci_95": [diffs_vs_champ[c_name]["ci_lower"], diffs_vs_champ[c_name]["ci_upper"]],
            "p_raw": raw_p_vs_champ[c_name],
            "p_holm": holm_p_champ[c_name],
            "p_holm_fmt": format_p_value(holm_p_champ[c_name]),
        }

    # Print Master Table
    print(f"\n{'Candidate':<18} | {'Dev AUROC':<10} | {'LS-AUROC':<10} | {'LM-AUROC':<10} | {'C++ LM-AUC':<10} | {'Pair Correct':<12} | {'Diff vs L1 (LM)':<16} | {'Holm p'}")
    print("-" * 115)
    l1_r = dev_results["L1"]
    print(f"{'L1 (baseline)':<18} | {l1_r['auroc']:<10.4f} | {l1_r['ls_auroc_10bin']:<10.4f} | {l1_r['length_matched_auroc']:<10.4f} | {l1_r['cpp_length_matched_auroc']:<10.4f} | {l1_r['pair_correct']*100:<11.2f}% | {'[Reference]':<16} | {'N/A'}")
    ch_r = dev_results["champion_faithful"]
    print(f"{'Champion':<18} | {ch_r['auroc']:<10.4f} | {ch_r['ls_auroc_10bin']:<10.4f} | {ch_r['length_matched_auroc']:<10.4f} | {ch_r['cpp_length_matched_auroc']:<10.4f} | {ch_r['pair_correct']*100:<11.2f}% | {'[Reference]':<16} | {'N/A'}")

    for c_name in comp_models:
        r = dev_results[c_name]
        d_val = r["diff_vs_l1"]["diff_observed"]
        p_val = r["diff_vs_l1"]["p_holm_fmt"]
        print(f"{c_name:<18} | {r['auroc']:<10.4f} | {r['ls_auroc_10bin']:<10.4f} | {r['length_matched_auroc']:<10.4f} | {r['cpp_length_matched_auroc']:<10.4f} | {r['pair_correct']*100:<11.2f}% | {d_val:<+16.4f} | {p_val}")

    # -------------------------------------------------------------------------
    # Mechanical Selection Rule
    # -------------------------------------------------------------------------
    print("\n--- Applying Mechanical Selection Rule ---")
    # Gate: beat L1 on dev length-matched AUROC (Holm p < 0.05)
    passing_gate = []
    for c_name in comp_models:
        r = dev_results[c_name]
        p_holm = r["diff_vs_l1"]["p_holm"]
        diff_obs = r["diff_vs_l1"]["diff_observed"]
        if diff_obs > 0 and p_holm < 0.05:
            passing_gate.append(c_name)

    print(f"Candidates beating L1 on LM-AUROC (Holm p < 0.05): {passing_gate}")

    if passing_gate:
        # Rank by highest mean of (length-matched AUROC, pair_correct)
        scored_candidates = []
        for c_name in passing_gate:
            r = dev_results[c_name]
            combined = (r["length_matched_auroc"] + r["pair_correct"]) / 2.0
            scored_candidates.append((combined, c_name))
        scored_candidates.sort(reverse=True)
        selected_cand = scored_candidates[0][1]
        print(f">>> Selected Candidate via Rule: {selected_cand} (Score: {scored_candidates[0][0]:.4f})")
    else:
        print(">>> None beat L1 with Holm p < 0.05 on LM-AUROC. Falling back to V3 as specified.")
        selected_cand = "V3"

    # Fit V6 triage stacker on [selected_logit, log(n_tokens)]
    print("\n[-] Fitting V6 triage stacker...")
    p_sel_dev = preds_dev[selected_cand]
    logit_sel_dev = np.log(np.clip(p_sel_dev, 1e-6, 1 - 1e-6) / (1 - np.clip(p_sel_dev, 1e-6, 1 - 1e-6)))
    X_v6_dev = np.column_stack([logit_sel_dev, dev_len[:, :1]])

    # Fit stacker on dev logits
    m_v6 = LogisticRegression(random_state=42)
    m_v6.fit(X_v6_dev, y_dev)
    p_v6_dev = m_v6.predict_proba(X_v6_dev)[:, 1]
    dev_results["V6"] = evaluate_on_dev(
        y_dev, p_v6_dev, tok_dev, is_cpp_dev, commits_dev, cwes_dev, dev_id_to_idx, df_dev_pairs
    )
    preds_dev["V6"] = p_v6_dev

    # -------------------------------------------------------------------------
    # Decision Rule & Thresholds (A1.7)
    # -------------------------------------------------------------------------
    print("\n--- Calibrating Decision Rules & Thresholds on Dev ---")
    # 1. Binary threshold t_bin: max balanced accuracy on dev
    best_t_bin = 0.5
    best_bal = -1.0
    for thr in np.linspace(0.1, 0.9, 81):
        pred_b = (p_sel_dev >= thr).astype(int)
        bal = balanced_accuracy_score(y_dev, pred_b)
        if bal > best_bal:
            best_bal = bal
            best_t_bin = thr

    # 2. Three-way thresholds:
    # t_high for precision >= 0.80 on dev (or top 5% by score)
    t_high = None
    for thr in np.linspace(0.5, 0.99, 100):
        flagged = (p_sel_dev >= thr)
        if np.sum(flagged) >= 20:
            p_val = precision_score(y_dev[flagged], np.ones(np.sum(flagged)), zero_division=0)
            if p_val >= 0.80:
                t_high = float(thr)
                break
    if t_high is None:
        # Fallback to top 5% by score
        t_high = float(np.percentile(p_sel_dev, 95))

    # t_low for SAFE NPV >= 0.90 on dev (or bottom 20% by score)
    t_low = None
    for thr in np.linspace(0.01, 0.5, 100):
        safe_mask = (p_sel_dev <= thr)
        if np.sum(safe_mask) >= 50:
            npv = float(np.mean(y_dev[safe_mask] == 0))
            if npv >= 0.90:
                t_low = float(thr)
                # Keep largest threshold meeting >= 0.90 NPV
    if t_low is None:
        t_low = float(np.percentile(p_sel_dev, 20))

    threshold_config = {
        "t_bin": float(best_t_bin),
        "best_dev_balanced_acc": float(best_bal),
        "t_high": float(t_high),
        "t_low": float(t_low),
        "verdict_rule": {
            "VULNERABLE": f"score >= {t_high:.4f}",
            "SAFE": f"score <= {t_low:.4f}",
            "UNSURE": f"{t_low:.4f} < score < {t_high:.4f}"
        },
        "length_risk_bins": {
            "LOW": "tokens <= 256",
            "MED": "257 <= tokens <= 1024",
            "HIGH": "tokens > 1024"
        }
    }
    print(f"    t_bin (max BalAcc): {best_t_bin:.4f} (BalAcc: {best_bal:.4f})")
    print(f"    t_high (VULNERABLE): {t_high:.4f}")
    print(f"    t_low (SAFE):        {t_low:.4f}")

    # Save model artifacts
    print(f"\n[4] Saving Model A v2 Artifacts to {MODELS_DIR}...")
    selected_model_obj = candidates[selected_cand]
    artifact = {
        "model_name": selected_cand,
        "model": selected_model_obj,
        "scaler_static": scaler_stat,
        "scaler_len": scaler_len,
        "thresholds": threshold_config,
    }
    with open(BEST_MODEL_PKL, "wb") as f:
        pickle.dump(artifact, f)

    with open(STACKER_PKL, "wb") as f:
        pickle.dump(m_v6, f)

    # -------------------------------------------------------------------------
    # Freeze Protocol (A1.7): Write freeze_a.json BEFORE touching test
    # -------------------------------------------------------------------------
    print(f"\n[5] Writing Freeze Artifact to {FREEZE_JSON} BEFORE touching test sets...")
    freeze_record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "selected_candidate": selected_cand,
        "files": {
            "model_pkl": {
                "path": str(BEST_MODEL_PKL),
                "sha256": compute_file_sha256(BEST_MODEL_PKL)
            },
            "stacker_pkl": {
                "path": str(STACKER_PKL),
                "sha256": compute_file_sha256(STACKER_PKL)
            },
            "train_v2": {
                "path": str(TRAIN_V2_PARQUET),
                "sha256": compute_file_sha256(TRAIN_V2_PARQUET)
            },
            "train_pairs_v2": {
                "path": str(TRAIN_PAIRS_V2_PARQUET),
                "sha256": compute_file_sha256(TRAIN_PAIRS_V2_PARQUET)
            }
        },
        "thresholds": threshold_config
    }
    with open(FREEZE_JSON, "w") as f:
        json.dump(freeze_record, f, indent=2)
    print("    Freeze record written successfully.")

    # -------------------------------------------------------------------------
    # Final Evaluation (A1.8) - Call locked loaders ONCE each
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("FINAL EVALUATION (CALLING LOCKED LOADERS ONCE)")
    print("=" * 90)

    from scripts.p10.extract_model_a_v2_features import (
        extract_length_features_from_codes,
        compute_token_counts,
    )
    from scripts.p10.extract_juliet_features import load_cwe_families, extract_static_features
    from scripts.p2.cache_embeddings_p2 import extract_standard_codebert, extract_chunked_codebert
    from transformers import AutoTokenizer, AutoModel

    families, fam_to_idx, cwe_to_fam_idx = load_cwe_families()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Primary: test2
    print("\n[6] Loading locked test2 via load_locked_test2(reason='P10 final')...")
    df_test2 = load_locked_test2(reason="P10 final")
    print(f"    Loaded test2: {len(df_test2)} rows.")

    # Secondary: old test
    print("\n[7] Loading locked old test via load_locked_test(reason='P10 secondary, viewed in P6')...")
    df_test_old = load_locked_test(reason="P10 secondary, viewed in P6")
    print(f"    Loaded old test: {len(df_test_old)} rows.")

    # -------------------------------------------------------------------------
    # Extract features for test2
    # -------------------------------------------------------------------------
    print("\n[-] Computing features for test2...")
    t2_cache_file = CACHE_P10_DIR / "test2_features_cache.pkl"
    if t2_cache_file.exists():
        print(f"    Loading cached test2 features from {t2_cache_file.name}...")
        with open(t2_cache_file, "rb") as f:
            t2_cache = pickle.load(f)
        t2_len = t2_cache["t2_len"]
        t2_stat = t2_cache["t2_stat"]
        t2_qwen = t2_cache["t2_qwen"]
        t2_cb = t2_cache["t2_cb"]
    else:
        t2_len = extract_length_features_from_codes(df_test2["code"].tolist())
        t2_stat = extract_static_features(df_test2, families, fam_to_idx, cwe_to_fam_idx)

        print(f"    Extracting test2 Qwen embeddings on {device}...")
        qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
        qwen_mod = AutoModel.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", torch_dtype=torch.float16).to(device)
        qwen_mod.eval()

        t2_qwen = np.zeros((len(df_test2), 896), dtype=np.float32)
        codes_t2 = df_test2["code"].tolist()
        bs = 32
        for i in range(0, len(codes_t2), bs):
            b = codes_t2[i : min(i + bs, len(codes_t2))]
            inp = qwen_tok(b, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                out = qwen_mod(**inp)
                mask = inp["attention_mask"].unsqueeze(-1)
                embs = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                t2_qwen[i : i + len(b)] = embs.cpu().to(torch.float32).numpy()

        print(f"    Extracting test2 CodeBERT embeddings on {device}...")
        cb_tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        cb_mod = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
        cb_mod.eval()

        t2_cb_std = extract_standard_codebert(codes_t2, cb_tok, cb_mod, device, batch_size=64, max_length=256)
        t2_cb, _ = extract_chunked_codebert(
            codes_t2, t2_cb_std, cb_tok, cb_mod, device, batch_size=64, window_size=254, stride=128, max_windows_per_sample=16, sub_chunk_size=1000
        )
        with open(t2_cache_file, "wb") as f:
            pickle.dump({"t2_len": t2_len, "t2_stat": t2_stat, "t2_qwen": t2_qwen, "t2_cb": t2_cb}, f)
        print(f"    Saved test2 features cache to {t2_cache_file.name}")

    t2_len_s = scaler_len.transform(t2_len)
    t2_stat_s = scaler_stat.transform(t2_stat)
    tok_t2 = np.exp(t2_len[:, 0]) - 1.0

    y_t2 = df_test2["vuln_label"].values.astype(int)
    commits_t2 = df_test2["commit_id"].values
    cwes_t2 = df_test2["cwe"]
    is_cpp_t2 = np.array(["::" in c or "std::" in c or "template<" in c for c in codes_t2])

    p_l1_t2 = predict_candidate("L1", m_l1, t2_qwen, t2_cb, t2_stat_s, t2_len_s)
    p_champ_t2 = predict_candidate("champion_faithful", m_champ, t2_qwen, t2_cb, t2_stat_s, t2_len_s)
    p_sel_t2 = predict_candidate(selected_cand, selected_model_obj, t2_qwen, t2_cb, t2_stat_s, t2_len_s)
    logit_sel_t2 = np.log(np.clip(p_sel_t2, 1e-6, 1 - 1e-6) / (1 - np.clip(p_sel_t2, 1e-6, 1 - 1e-6)))
    p_v6_t2 = m_v6.predict_proba(np.column_stack([logit_sel_t2, t2_len[:, :1]]))[:, 1]

    # Evaluate test2
    print("\n--- TEST2 EVALUATION METRICS (PRIMARY TEST) ---")
    test2_eval = {}
    for m_label, p_arr in [("Selected_Model", p_sel_t2), ("V6_Stacker", p_v6_t2), ("L1", p_l1_t2), ("Champion", p_champ_t2)]:
        metrics = evaluate_dataset_full(y_t2, p_arr, tok_t2, is_cpp_t2, commits_t2, cwes_t2)
        test2_eval[m_label] = metrics
        print(f"    {m_label:<16} | AUROC: {metrics['auroc']:.4f} {metrics['auroc_ci']} | LS-AUROC: {metrics['ls_auroc_10bin']:.4f} | LM-AUROC: {metrics['length_matched_auroc']:.4f} {metrics['length_matched_auroc_ci']} | C++ LM-AUC: {metrics['cpp_length_matched_auroc']:.4f} | Top10% Prec: {metrics['operating_point_10pct']['precision']:.4f}")

    # -------------------------------------------------------------------------
    # Extract features for old test
    # -------------------------------------------------------------------------
    print("\n[-] Computing features for old test (secondary test)...")
    codes_old = df_test_old["code"].tolist()
    t_old_cache_file = CACHE_P10_DIR / "old_test_features_cache.pkl"
    if t_old_cache_file.exists():
        print(f"    Loading cached old test features from {t_old_cache_file.name}...")
        with open(t_old_cache_file, "rb") as f:
            old_cache = pickle.load(f)
        t_old_len = old_cache["t_old_len"]
        t_old_stat = old_cache["t_old_stat"]
        t_old_qwen = old_cache["t_old_qwen"]
        t_old_cb = old_cache["t_old_cb"]
    else:
        t_old_len = extract_length_features_from_codes(codes_old)
        t_old_stat = extract_static_features(df_test_old, families, fam_to_idx, cwe_to_fam_idx)

        cb_old_cache = CACHE_P2_DIR / "emb_codebert_chunked_mean_256_ebd95dfc837f12264dd0fd8126563d00ede27908bca38fe46611e25d7bb45bbc.npy"
        if cb_old_cache.exists():
            t_old_cb = np.load(cb_old_cache)
            print(f"    Reused cached CodeBERT chunked-mean for old test: {t_old_cb.shape}")
        else:
            t_old_cb_std = extract_standard_codebert(codes_old, cb_tok, cb_mod, device, batch_size=64, max_length=256)
            t_old_cb, _ = extract_chunked_codebert(
                codes_old, t_old_cb_std, cb_tok, cb_mod, device, batch_size=64, window_size=254, stride=128, max_windows_per_sample=16, sub_chunk_size=1000
            )

        print(f"    Extracting old test Qwen embeddings on {device}...")
        t_old_qwen = np.zeros((len(df_test_old), 896), dtype=np.float32)
        for i in range(0, len(codes_old), bs):
            b = codes_old[i : min(i + bs, len(codes_old))]
            inp = qwen_tok(b, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                out = qwen_mod(**inp)
                mask = inp["attention_mask"].unsqueeze(-1)
                embs = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                t_old_qwen[i : i + len(b)] = embs.cpu().to(torch.float32).numpy()

        with open(t_old_cache_file, "wb") as f:
            pickle.dump({"t_old_len": t_old_len, "t_old_stat": t_old_stat, "t_old_qwen": t_old_qwen, "t_old_cb": t_old_cb}, f)
        print(f"    Saved old test features cache to {t_old_cache_file.name}")

    t_old_len_s = scaler_len.transform(t_old_len)
    t_old_stat_s = scaler_stat.transform(t_old_stat)
    tok_old = np.exp(t_old_len[:, 0]) - 1.0

    y_old = df_test_old["vuln_label"].values.astype(int)
    commits_old = df_test_old["commit_id"].values
    cwes_old = df_test_old["cwe"]
    is_cpp_old = np.array(["::" in c or "std::" in c or "template<" in c for c in codes_old])

    p_l1_old = predict_candidate("L1", m_l1, t_old_qwen, t_old_cb, t_old_stat_s, t_old_len_s)
    p_champ_old = predict_candidate("champion_faithful", m_champ, t_old_qwen, t_old_cb, t_old_stat_s, t_old_len_s)
    p_sel_old = predict_candidate(selected_cand, selected_model_obj, t_old_qwen, t_old_cb, t_old_stat_s, t_old_len_s)
    logit_sel_old = np.log(np.clip(p_sel_old, 1e-6, 1 - 1e-6) / (1 - np.clip(p_sel_old, 1e-6, 1 - 1e-6)))
    p_v6_old = m_v6.predict_proba(np.column_stack([logit_sel_old, t_old_len[:, :1]]))[:, 1]

    # Evaluate old test
    print("\n--- OLD TEST EVALUATION METRICS (SECONDARY TEST) ---")
    old_test_eval = {}
    for m_label, p_arr in [("Selected_Model", p_sel_old), ("V6_Stacker", p_v6_old), ("L1", p_l1_old), ("Champion", p_champ_old)]:
        metrics = evaluate_dataset_full(y_old, p_arr, tok_old, is_cpp_old, commits_old, cwes_old)
        old_test_eval[m_label] = metrics
        print(f"    {m_label:<16} | AUROC: {metrics['auroc']:.4f} {metrics['auroc_ci']} | LS-AUROC: {metrics['ls_auroc_10bin']:.4f} | LM-AUROC: {metrics['length_matched_auroc']:.4f} {metrics['length_matched_auroc_ci']} | C++ LM-AUC: {metrics['cpp_length_matched_auroc']:.4f} | Top10% Prec: {metrics['operating_point_10pct']['precision']:.4f}")

    # -------------------------------------------------------------------------
    # Cross-evaluation on Juliet S-flow test split
    # -------------------------------------------------------------------------
    print("\n--- CROSS-EVALUATION ON JULIET S-FLOW TEST SPLIT ---")
    j_flow_test_mask = (df_j["split_flow"] == "test").values
    j_qwen_test = np.load(CACHE_P10_DIR / "juliet_qwen_embs.npy")[j_flow_test_mask]
    j_stat_test = np.load(CACHE_P10_DIR / "juliet_static_features.npy")[j_flow_test_mask]
    j_stat_test_s = scaler_stat.transform(j_stat_test)
    j_len_test = np.load(CACHE_P10_DIR / "juliet_length_features.npy")[j_flow_test_mask]
    j_len_test_s = scaler_len.transform(j_len_test)
    j_cb_zeros = np.zeros((len(j_qwen_test), 1536), dtype=np.float32)

    p_sel_juliet = predict_candidate(selected_cand, selected_model_obj, j_qwen_test, j_cb_zeros, j_stat_test_s, j_len_test_s)
    y_j_test = df_j.loc[j_flow_test_mask, "vuln_label"].values

    juliet_cross_eval = {
        "auroc": float(roc_auc_score(y_j_test, p_sel_juliet)),
        "balanced_accuracy": float(balanced_accuracy_score(y_j_test, (p_sel_juliet >= threshold_config["t_bin"]).astype(int))),
        "operating_point_10pct": compute_operating_point_metrics(y_j_test, p_sel_juliet, df_j.loc[j_flow_test_mask, "cwe"], top_pct=0.10)
    }
    print(f"    Juliet S-flow Test AUROC: {juliet_cross_eval['auroc']:.4f} | BalAcc: {juliet_cross_eval['balanced_accuracy']:.4f} | Top10% Prec: {juliet_cross_eval['operating_point_10pct']['precision']:.4f}")

    # Write evaluation JSON
    eval_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "selected_candidate": selected_cand,
        "dev_evaluation": dev_results,
        "test2_evaluation": test2_eval,
        "old_test_evaluation": old_test_eval,
        "juliet_cross_evaluation": juliet_cross_eval,
        "decision_rules": threshold_config,
    }
    with open(EVAL_JSON, "w") as f:
        json.dump(eval_output, f, indent=2)
    print(f"\n[8] Evaluation summary written to {EVAL_JSON}")


if __name__ == "__main__":
    main()
