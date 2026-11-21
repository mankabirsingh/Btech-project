"""Evaluation protocol for Phase 16 Prompt 3 experiments."""

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.eval.stats import (
    calc_length_stratified_auroc,
    holm_adjust,
    length_stratified_auroc,
    metrics_full,
    paired_bootstrap_diff,
    threshold_at_fpr,
)

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
P2_SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "p2"
RESULTS_P3_DIR = PROJECT_ROOT / "results" / "p3"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load train_minus_dev, dev_heldout, and precomputed token lengths."""
    df_train = pd.read_parquet(P2_SPLITS_DIR / "train_minus_dev.parquet")
    df_dev = pd.read_parquet(P2_SPLITS_DIR / "dev_heldout.parquet")

    tok_train = np.load(CACHE_DIR / "train_token_lens.npy")
    tok_dev = np.load(CACHE_DIR / "dev_token_lens.npy")

    return df_train, df_dev, tok_train, tok_dev


def get_cached_features() -> Dict[str, Dict[str, np.ndarray]]:
    """Loads all precomputed features from data/cache/ for train and dev."""
    # Read cache manifest to get exact filenames
    with open(CACHE_DIR / "cache_manifest.json", "r") as f:
        manifest = json.load(f)

    train_files = manifest["datasets"]["train_minus_dev"]["files"]
    dev_files = manifest["datasets"]["dev_heldout"]["files"]

    features = {
        "train": {
            "emb_codebert_256": np.load(PROJECT_ROOT / train_files["emb_codebert_256"]["path"]),
            "emb_chunked_mean": np.load(PROJECT_ROOT / train_files["emb_codebert_chunked_mean_256"]["path"]),
            "emb_chunked_max": np.load(PROJECT_ROOT / train_files["emb_codebert_chunked_max_256"]["path"]),
            "cxx28": np.load(PROJECT_ROOT / train_files["feat_cxx28"]["path"]),
            "cxx38": np.load(PROJECT_ROOT / train_files["feat_cxx38"]["path"]),
        },
        "dev": {
            "emb_codebert_256": np.load(PROJECT_ROOT / dev_files["emb_codebert_256"]["path"]),
            "emb_chunked_mean": np.load(PROJECT_ROOT / dev_files["emb_codebert_chunked_mean_256"]["path"]),
            "emb_chunked_max": np.load(PROJECT_ROOT / dev_files["emb_codebert_chunked_max_256"]["path"]),
            "cxx28": np.load(PROJECT_ROOT / dev_files["feat_cxx28"]["path"]),
            "cxx38": np.load(PROJECT_ROOT / dev_files["feat_cxx38"]["path"]),
        },
    }
    return features


def get_champion_p2_reference(
    df_train: pd.DataFrame,
    df_dev: pd.DataFrame,
    tok_dev: np.ndarray,
    features: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    """
    Computes GroupKFold OOF predictions on train_minus_dev and predictions on dev
    for the champion_p2 reference model.
    """
    ref_cache_path = RESULTS_P3_DIR / "champion_p2_reference_eval.json"
    if ref_cache_path.exists():
        with open(ref_cache_path, "r") as f:
            return json.load(f)

    print("\nComputing Champion P2 Reference 5-Fold GroupKFold OOF & Dev metrics...")
    y_train = df_train["vuln_label"].values.astype(int)
    y_dev = df_dev["vuln_label"].values.astype(int)
    kfolds = df_train["kfold"].values

    X_train_raw = np.hstack([features["train"]["emb_codebert_256"], features["train"]["cxx28"]])
    X_dev_raw = np.hstack([features["dev"]["emb_codebert_256"], features["dev"]["cxx28"]])

    oof_probs = np.zeros(len(y_train), dtype=np.float32)
    fold_aurocs = []

    for fold in range(5):
        tr_idx = np.where(kfolds != fold)[0]
        val_idx = np.where(kfolds == fold)[0]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_raw[tr_idx])
        X_val = scaler.transform(X_train_raw[val_idx])

        clf = LogisticRegression(C=0.01, class_weight="balanced", max_iter=1000, random_state=42)
        clf.fit(X_tr, y_train[tr_idx])
        p_val = clf.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = p_val
        fold_auc = float(roc_auc_score(y_train[val_idx], p_val))
        fold_aurocs.append(fold_auc)

    # Threshold on OOF reweighted to 30% prevalence at FPR <= 10%
    thr_info = threshold_at_fpr(y_train, oof_probs, target_fpr=0.10, target_prevalence=0.30)
    opt_thr = thr_info["threshold"]

    # Full fit
    scaler_full = StandardScaler()
    X_train_scaled = scaler_full.fit_transform(X_train_raw)
    X_dev_scaled = scaler_full.transform(X_dev_raw)

    clf_full = LogisticRegression(C=0.01, class_weight="balanced", max_iter=1000, random_state=42)
    clf_full.fit(X_train_scaled, y_train)
    p_dev = clf_full.predict_proba(X_dev_scaled)[:, 1]

    # Dev evaluation
    m_dev = metrics_full(y_dev, p_dev, threshold=opt_thr)
    ls_dev = length_stratified_auroc(y_dev, p_dev, tok_dev, n_resamples=1000, seed=42)

    is_cpp = df_dev["is_cpp"].values.astype(bool)
    cpp_auc = float(roc_auc_score(y_dev[is_cpp], p_dev[is_cpp])) if len(np.unique(y_dev[is_cpp])) > 1 else 0.5
    cpp_ci = length_stratified_auroc(y_dev[is_cpp], p_dev[is_cpp], tok_dev[is_cpp], n_resamples=1000, seed=42)["ci_95"]

    ref_data = {
        "candidate_id": "champion_p2",
        "description": "Champion P2 reference (CodeBERT 256 + cxx28, C=0.01, balanced)",
        "cv_fold_aurocs": fold_aurocs,
        "cv_mean_auroc": float(np.mean(fold_aurocs)),
        "cv_std_auroc": float(np.std(fold_aurocs)),
        "oof_threshold_info": thr_info,
        "oof_threshold": opt_thr,
        "dev_auroc": m_dev["auroc"],
        "dev_auroc_ci_95": ls_dev["ci_95"],  # standard auroc CI
        "dev_length_stratified_auroc": ls_dev["length_stratified_auroc"],
        "dev_length_stratified_ci_95": ls_dev["ci_95"],
        "dev_cpp_auroc": cpp_auc,
        "dev_cpp_auroc_ci_95": cpp_ci,
        "dev_auprc": m_dev["auprc"],
        "dev_recall_at_10fpr": m_dev["recall"],
        "dev_precision_at_10fpr": m_dev["precision"],
        "dev_fpr_at_10fpr": m_dev["fpr"],
        "p_dev": p_dev.tolist(),
        "p_oof": oof_probs.tolist(),
    }

    # Also compute standard AUROC CI on dev
    from src.eval.stats import bootstrap_ci
    std_ci = bootstrap_ci(y_dev, p_dev, threshold=opt_thr, n_resamples=1000, seed=42)
    ref_data["dev_auroc_ci_95"] = std_ci["auroc_ci_95"]

    with open(ref_cache_path, "w") as f:
        json.dump(ref_data, f, indent=2)

    return ref_data


def evaluate_candidate(
    candidate_id: str,
    description: str,
    category: str,
    X_train_raw: np.ndarray,
    X_dev_raw: np.ndarray,
    df_train: pd.DataFrame,
    df_dev: pd.DataFrame,
    tok_dev: np.ndarray,
    champ_ref: Dict[str, Any],
    clf_factory: Optional[Callable[[float], Any]] = None,
    C_val: float = 0.01,
    custom_fit_predict: Optional[Callable[..., Tuple[np.ndarray, np.ndarray, Any]]] = None,
) -> Dict[str, Any]:
    """
    Standard evaluation pipeline for any candidate on train_minus_dev and dev_heldout.
    Enforces Step 0 rules strictly.
    """
    t0 = time.time()
    y_train = df_train["vuln_label"].values.astype(int)
    y_dev = df_dev["vuln_label"].values.astype(int)
    kfolds = df_train["kfold"].values
    dev_commits = df_dev["commit_id"].values
    is_cpp = df_dev["is_cpp"].values.astype(bool)

    p_champ_dev = np.array(champ_ref["p_dev"])

    if custom_fit_predict is not None:
        oof_probs, fold_aurocs, p_dev, fitted_model = custom_fit_predict(
            X_train_raw, y_train, kfolds, X_dev_raw, y_dev
        )
    else:
        # Standard logistic regression with fold-wise StandardScaler
        oof_probs = np.zeros(len(y_train), dtype=np.float32)
        fold_aurocs = []

        for fold in range(5):
            tr_idx = np.where(kfolds != fold)[0]
            val_idx = np.where(kfolds == fold)[0]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train_raw[tr_idx])
            X_val = scaler.transform(X_train_raw[val_idx])

            clf = clf_factory(C_val) if clf_factory else LogisticRegression(
                C=C_val, class_weight="balanced", max_iter=1000, random_state=42
            )
            clf.fit(X_tr, y_train[tr_idx])
            p_val = clf.predict_proba(X_val)[:, 1]
            oof_probs[val_idx] = p_val
            fold_auc = float(roc_auc_score(y_train[val_idx], p_val))
            fold_aurocs.append(fold_auc)

        # Full fit
        scaler_full = StandardScaler()
        X_train_scaled = scaler_full.fit_transform(X_train_raw)
        X_dev_scaled = scaler_full.transform(X_dev_raw)

        clf_full = clf_factory(C_val) if clf_factory else LogisticRegression(
            C=C_val, class_weight="balanced", max_iter=1000, random_state=42
        )
        clf_full.fit(X_train_scaled, y_train)
        p_dev = clf_full.predict_proba(X_dev_scaled)[:, 1]
        fitted_model = {"clf": clf_full, "scaler": scaler_full}

    # Step 0.3: Threshold on OOF reweighted to 30% prevalence at FPR <= 10%
    thr_info = threshold_at_fpr(y_train, oof_probs, target_fpr=0.10, target_prevalence=0.30)
    opt_thr = thr_info["threshold"]

    # Step 0.2: Dev metrics
    m_dev = metrics_full(y_dev, p_dev, threshold=opt_thr)

    from src.eval.stats import bootstrap_ci
    ci_dev = bootstrap_ci(y_dev, p_dev, threshold=opt_thr, n_resamples=1000, seed=42)
    ls_dev = length_stratified_auroc(y_dev, p_dev, tok_dev, n_resamples=1000, seed=42)

    # C++ subset
    y_cpp = y_dev[is_cpp]
    p_cpp = p_dev[is_cpp]
    tok_cpp = tok_dev[is_cpp]
    cpp_auc = float(roc_auc_score(y_cpp, p_cpp)) if len(np.unique(y_cpp)) > 1 else 0.5
    cpp_ci = bootstrap_ci(y_cpp, p_cpp, threshold=opt_thr, n_resamples=1000, seed=42)["auroc_ci_95"]

    # Step 0.2f: Paired bootstrap difference vs champion_p2 grouped by commit_id
    def _auc_fn(y_sub, p_sub, _):
        return roc_auc_score(y_sub, p_sub) if len(np.unique(y_sub)) > 1 else 0.5

    def _ls_auc_fn(y_sub, p_sub, _, **extra):
        t_sub = extra.get("n_tokens", None)
        if t_sub is None:
            return roc_auc_score(y_sub, p_sub) if len(np.unique(y_sub)) > 1 else 0.5
        return calc_length_stratified_auroc(y_sub, p_sub, t_sub)

    diff_overall = paired_bootstrap_diff(
        y_dev, p_champ_dev, p_dev, _auc_fn, groups=dev_commits, n_resamples=1000, seed=42
    )

    diff_ls = paired_bootstrap_diff(
        y_dev,
        p_champ_dev,
        p_dev,
        _ls_auc_fn,
        groups=dev_commits,
        extra_arrays={"n_tokens": tok_dev},
        n_resamples=1000,
        seed=42,
    )

    diff_cpp = paired_bootstrap_diff(
        y_cpp,
        p_champ_dev[is_cpp],
        p_cpp,
        _auc_fn,
        groups=dev_commits[is_cpp],
        n_resamples=1000,
        seed=42,
    )

    result = {
        "candidate_id": candidate_id,
        "description": description,
        "category": category,
        "C_chosen": C_val,
        "cv_fold_aurocs": fold_aurocs,
        "cv_mean_auroc": float(np.mean(fold_aurocs)),
        "cv_std_auroc": float(np.std(fold_aurocs)),
        "oof_threshold_info": thr_info,
        "oof_threshold": opt_thr,
        "dev_auroc": m_dev["auroc"],
        "dev_auroc_ci_95": ci_dev["auroc_ci_95"],
        "dev_length_stratified_auroc": ls_dev["length_stratified_auroc"],
        "dev_length_stratified_ci_95": ls_dev["ci_95"],
        "dev_cpp_auroc": cpp_auc,
        "dev_cpp_auroc_ci_95": cpp_ci,
        "dev_auprc": m_dev["auprc"],
        "dev_recall_at_10fpr": m_dev["recall"],
        "dev_precision_at_10fpr": m_dev["precision"],
        "dev_fpr_at_10fpr": m_dev["fpr"],
        "diff_vs_champion": {
            "overall_auroc": diff_overall,
            "length_stratified_auroc": diff_ls,
            "cpp_auroc": diff_cpp,
        },
        "runtime_seconds": round(time.time() - t0, 2),
        "p_dev": p_dev.tolist(),
        "p_oof": oof_probs.tolist(),
    }
    return result
