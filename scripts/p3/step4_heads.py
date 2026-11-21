"""STEP 4: Classifier Head Optimization.

Evaluates H1 (Logistic Regression), H2 (LightGBM), and H3 (2-layer MLP, 5 seeds)
on the best representation identified in Step 3.
Reports all Step 0 metrics and std across seeds for H3.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
torch.set_num_threads(1)

from scripts.p3.eval_protocol import (
    evaluate_candidate,
    get_cached_features,
    get_champion_p2_reference,
    load_datasets,
)
from src.eval.stats import (
    bootstrap_ci,
    calc_length_stratified_auroc,
    length_stratified_auroc,
    metrics_full,
    paired_bootstrap_diff,
    threshold_at_fpr,
)

RESULTS_P3_DIR = PROJECT_ROOT / "results" / "p3"
RESULTS_P3_DIR.mkdir(parents=True, exist_ok=True)


def extract_cyclomatic_proxy(codes: pd.Series) -> np.ndarray:
    pattern = re.compile(r"\b(if|for|while|case|catch)\b|&&|\|\|")
    counts = [len(pattern.findall(str(c))) for c in codes]
    return np.array(counts, dtype=np.float32)


class TwoLayerMLP(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
    max_epochs: int = 35,
    patience: int = 8,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Tuple[np.ndarray, int]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_dim = X_tr.shape[1]
    model = TwoLayerMLP(in_dim, hidden_dim=256, dropout=0.3).to(device)

    # Class weighting for BCE
    pos_weight = torch.tensor([(len(y_tr) - np.sum(y_tr)) / max(1, np.sum(y_tr))], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    X_tr_t = torch.from_numpy(np.ascontiguousarray(X_tr, dtype=np.float32)).to(device)
    y_tr_t = torch.from_numpy(np.ascontiguousarray(y_tr, dtype=np.float32)).to(device)
    t_val_x = torch.from_numpy(np.ascontiguousarray(X_val, dtype=np.float32)).to(device)
    t_val_y = torch.from_numpy(np.ascontiguousarray(y_val, dtype=np.float32)).to(device)
    n_samples = len(X_tr)

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            bx, by = X_tr_t[idx], y_tr_t[idx]
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(t_val_x)
            val_loss = float(criterion(val_out, t_val_y).item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(t_val_x)).cpu().numpy()

    return val_probs, best_epoch


def train_mlp_full(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    epochs: int,
    seed: int = 42,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_dim = X_tr.shape[1]
    model = TwoLayerMLP(in_dim, hidden_dim=256, dropout=0.3).to(device)

    pos_weight = torch.tensor([(len(y_tr) - np.sum(y_tr)) / max(1, np.sum(y_tr))], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    X_tr_t = torch.from_numpy(np.ascontiguousarray(X_tr, dtype=np.float32)).to(device)
    y_tr_t = torch.from_numpy(np.ascontiguousarray(y_tr, dtype=np.float32)).to(device)
    n_samples = len(X_tr)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            bx, by = X_tr_t[idx], y_tr_t[idx]
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def main():
    print("=" * 80)
    print("STEP 4: CLASSIFIER HEAD OPTIMIZATION (H1, H2, H3)")
    print("=" * 80)
    t_start = time.time()

    df_train, df_dev, tok_train, tok_dev = load_datasets()
    features = get_cached_features()
    champ_ref = get_champion_p2_reference(df_train, df_dev, tok_dev, features)

    y_train = df_train["vuln_label"].values.astype(int)
    y_dev = df_dev["vuln_label"].values.astype(int)
    kfolds = df_train["kfold"].values
    dev_commits = df_dev["commit_id"].values
    is_cpp = df_dev["is_cpp"].values.astype(bool)

    # 1. Load Step 3 Results to Identify Best Representation
    step3_json = RESULTS_P3_DIR / "step3_representation.json"
    if not step3_json.exists():
        raise FileNotFoundError("Step 3 results not found. Run step3_representation.py first.")

    with open(step3_json, "r") as f:
        step3_data = json.load(f)

    best_rep_id = step3_data["best_representation"]
    best_rep_info = step3_data["representations"][best_rep_id]
    print(f"Best Representation from Step 3: {best_rep_id} ({best_rep_info['description']})")

    # Construct the exact feature matrix for the best representation
    log_tok_train = np.log1p(tok_train).reshape(-1, 1).astype(np.float32)
    log_tok_dev = np.log1p(tok_dev).reshape(-1, 1).astype(np.float32)
    lines_train = np.array([str(c).count("\n") + 1 for c in df_train["code"]], dtype=np.float32)
    lines_dev = np.array([str(c).count("\n") + 1 for c in df_dev["code"]], dtype=np.float32)
    cyc_train = np.log1p(extract_cyclomatic_proxy(df_train["code"])).reshape(-1, 1)
    cyc_dev = np.log1p(extract_cyclomatic_proxy(df_dev["code"])).reshape(-1, 1)
    L2_train = np.hstack([log_tok_train, np.log1p(lines_train).reshape(-1, 1), cyc_train])
    L2_dev = np.hstack([log_tok_dev, np.log1p(lines_dev).reshape(-1, 1), cyc_dev])

    rep_dict_tr = {
        "R1": np.hstack([features["train"]["emb_codebert_256"], features["train"]["cxx28"]]),
        "R2": np.hstack([features["train"]["emb_chunked_mean"], features["train"]["cxx28"]]),
        "R3": np.hstack([features["train"]["emb_chunked_max"], features["train"]["cxx28"]]),
        "R4": np.hstack([features["train"]["emb_codebert_256"], features["train"]["emb_chunked_mean"], features["train"]["emb_chunked_max"], features["train"]["cxx28"]]),
        "R5": np.hstack([features["train"]["emb_codebert_256"], features["train"]["emb_chunked_mean"], features["train"]["emb_chunked_max"], features["train"]["cxx38"]]),
        "R6": np.hstack([features["train"]["emb_codebert_256"], features["train"]["emb_chunked_mean"], features["train"]["emb_chunked_max"], features["train"]["cxx28"], L2_train]),
    }

    rep_dict_dv = {
        "R1": np.hstack([features["dev"]["emb_codebert_256"], features["dev"]["cxx28"]]),
        "R2": np.hstack([features["dev"]["emb_chunked_mean"], features["dev"]["cxx28"]]),
        "R3": np.hstack([features["dev"]["emb_chunked_max"], features["dev"]["cxx28"]]),
        "R4": np.hstack([features["dev"]["emb_codebert_256"], features["dev"]["emb_chunked_mean"], features["dev"]["emb_chunked_max"], features["dev"]["cxx28"]]),
        "R5": np.hstack([features["dev"]["emb_codebert_256"], features["dev"]["emb_chunked_mean"], features["dev"]["emb_chunked_max"], features["dev"]["cxx38"]]),
        "R6": np.hstack([features["dev"]["emb_codebert_256"], features["dev"]["emb_chunked_mean"], features["dev"]["emb_chunked_max"], features["dev"]["cxx28"], L2_dev]),
    }

    X_train_best = rep_dict_tr[best_rep_id]
    X_dev_best = rep_dict_dv[best_rep_id]

    # --- H1: Logistic Regression ---
    print("\n[2] Setting up H1: Logistic Regression (inherited from Step 3)...")
    res_H1 = {
        "candidate_id": "H1",
        "description": f"Logistic Regression on {best_rep_id} (C*={best_rep_info['C_chosen']})",
        "classifier": "LogisticRegression",
        **{k: v for k, v in best_rep_info.items() if k not in ["candidate_id", "description"]},
    }

    # --- H2: LightGBM ---
    print("\n[3] Tuning and Evaluating H2: LightGBM...")
    leaves_grid = [15, 31]
    best_nl = 15
    best_lgb_macro = -1.0
    lgb_tuning_results = {}

    for nl in leaves_grid:
        fold_aucs = []
        fold_best_iters = []
        oof_lgb_cand = np.zeros(len(y_train), dtype=np.float32)

        for fold in range(5):
            tr_idx = np.where(kfolds != fold)[0]
            val_idx = np.where(kfolds == fold)[0]

            clf_lgb = lgb.LGBMClassifier(
                num_leaves=nl,
                learning_rate=0.05,
                n_estimators=500,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
                verbose=-1,
            )
            clf_lgb.fit(
                X_train_best[tr_idx],
                y_train[tr_idx],
                eval_set=[(X_train_best[val_idx], y_train[val_idx])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            )
            p_val = clf_lgb.predict_proba(X_train_best[val_idx])[:, 1]
            oof_lgb_cand[val_idx] = p_val
            fold_auc = float(roc_auc_score(y_train[val_idx], p_val))
            fold_aucs.append(fold_auc)
            fold_best_iters.append(clf_lgb.best_iteration_)

        macro_auc = float(np.mean(fold_aucs))
        lgb_tuning_results[nl] = {
            "macro_mean_auroc": macro_auc,
            "std_auroc": float(np.std(fold_aucs)),
            "fold_aurocs": fold_aucs,
            "mean_best_iteration": int(np.mean(fold_best_iters)),
        }
        if macro_auc > best_lgb_macro:
            best_lgb_macro = macro_auc
            best_nl = nl
            oof_lgb_best = oof_lgb_cand
            best_lgb_iters = int(np.mean(fold_best_iters))

    print(f"  Selected num_leaves = {best_nl} (Macro AUROC: {best_lgb_macro:.4f}, Mean Iter: {best_lgb_iters})")

    # Fit final LightGBM on full train
    clf_lgb_full = lgb.LGBMClassifier(
        num_leaves=best_nl,
        learning_rate=0.05,
        n_estimators=best_lgb_iters,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        verbose=-1,
    )
    clf_lgb_full.fit(X_train_best, y_train)
    p_dev_lgb = clf_lgb_full.predict_proba(X_dev_best)[:, 1]

    # Threshold on OOF
    thr_info_lgb = threshold_at_fpr(y_train, oof_lgb_best, target_fpr=0.10, target_prevalence=0.30)
    opt_thr_lgb = thr_info_lgb["threshold"]

    # Metrics on dev
    m_dev_lgb = metrics_full(y_dev, p_dev_lgb, threshold=opt_thr_lgb)
    ci_dev_lgb = bootstrap_ci(y_dev, p_dev_lgb, threshold=opt_thr_lgb, n_resamples=1000, seed=42)
    ls_dev_lgb = length_stratified_auroc(y_dev, p_dev_lgb, tok_dev, n_resamples=1000, seed=42)

    y_cpp = y_dev[is_cpp]
    p_cpp_lgb = p_dev_lgb[is_cpp]
    cpp_auc_lgb = float(roc_auc_score(y_cpp, p_cpp_lgb))
    cpp_ci_lgb = bootstrap_ci(y_cpp, p_cpp_lgb, threshold=opt_thr_lgb, n_resamples=1000, seed=42)["auroc_ci_95"]

    p_champ_dev = np.array(champ_ref["p_dev"])
    diff_overall_lgb = paired_bootstrap_diff(
        y_dev, p_champ_dev, p_dev_lgb, lambda y, p, _: roc_auc_score(y, p), groups=dev_commits, n_resamples=1000, seed=42
    )
    diff_ls_lgb = paired_bootstrap_diff(
        y_dev, p_champ_dev, p_dev_lgb,
        lambda y, p, _, **extra: calc_length_stratified_auroc(y, p, extra["n_tokens"]),
        groups=dev_commits, extra_arrays={"n_tokens": tok_dev}, n_resamples=1000, seed=42
    )
    diff_cpp_lgb = paired_bootstrap_diff(
        y_cpp, p_champ_dev[is_cpp], p_cpp_lgb, lambda y, p, _: roc_auc_score(y, p),
        groups=dev_commits[is_cpp], n_resamples=1000, seed=42
    )

    res_H2 = {
        "candidate_id": "H2",
        "description": f"LightGBM on {best_rep_id} (num_leaves={best_nl}, lr=0.05, n_est={best_lgb_iters})",
        "category": "Classifier Head",
        "classifier": "LightGBM",
        "tuning": lgb_tuning_results,
        "cv_mean_auroc": best_lgb_macro,
        "cv_std_auroc": lgb_tuning_results[best_nl]["std_auroc"],
        "oof_threshold_info": thr_info_lgb,
        "oof_threshold": opt_thr_lgb,
        "dev_auroc": m_dev_lgb["auroc"],
        "dev_auroc_ci_95": ci_dev_lgb["auroc_ci_95"],
        "dev_length_stratified_auroc": ls_dev_lgb["length_stratified_auroc"],
        "dev_length_stratified_ci_95": ls_dev_lgb["ci_95"],
        "dev_cpp_auroc": cpp_auc_lgb,
        "dev_cpp_auroc_ci_95": cpp_ci_lgb,
        "dev_auprc": m_dev_lgb["auprc"],
        "dev_recall_at_10fpr": m_dev_lgb["recall"],
        "dev_precision_at_10fpr": m_dev_lgb["precision"],
        "dev_fpr_at_10fpr": m_dev_lgb["fpr"],
        "diff_vs_champion": {
            "overall_auroc": diff_overall_lgb,
            "length_stratified_auroc": diff_ls_lgb,
            "cpp_auroc": diff_cpp_lgb,
        },
        "p_dev": p_dev_lgb.tolist(),
        "p_oof": oof_lgb_best.tolist(),
    }
    print(f"  H2 Dev AUROC: {m_dev_lgb['auroc']:.4f} {ci_dev_lgb['auroc_ci_95']}")
    print(f"  H2 Dev Length-Stratified AUROC: {ls_dev_lgb['length_stratified_auroc']:.4f} {ls_dev_lgb['ci_95']}")

    # --- H3: 2-layer MLP (5 Seeds) ---
    print("\n[4] Training H3: 2-layer MLP across 5 seeds...")
    scaler_mlp = StandardScaler()
    X_tr_mlp = scaler_mlp.fit_transform(X_train_best)
    X_dv_mlp = scaler_mlp.transform(X_dev_best)

    seeds = [42, 43, 44, 45, 46]
    seed_dev_preds = []
    seed_oof_preds = []
    seed_dev_aurocs = []
    seed_dev_ls_aurocs = []
    seed_dev_cpp_aurocs = []

    for s_idx, s in enumerate(seeds):
        print(f"  Training Seed {s} ({s_idx+1}/5)...")
        oof_s = np.zeros(len(y_train), dtype=np.float32)
        best_epochs = []

        for fold in range(5):
            tr_idx = np.where(kfolds != fold)[0]
            val_idx = np.where(kfolds == fold)[0]

            sc_fold = StandardScaler()
            X_tr_f = sc_fold.fit_transform(X_train_best[tr_idx])
            X_val_f = sc_fold.transform(X_train_best[val_idx])

            p_val, b_ep = train_mlp_fold(
                X_tr_f, y_train[tr_idx], X_val_f, y_train[val_idx],
                seed=s + fold * 10, max_epochs=35, patience=8, lr=1e-3, device="cpu"
            )
            oof_s[val_idx] = p_val
            best_epochs.append(b_ep)

        # Full fit for this seed
        mean_ep = max(5, int(np.mean(best_epochs)))
        model_full = train_mlp_full(
            X_tr_mlp, y_train, epochs=mean_ep,
            seed=s, batch_size=256, lr=1e-3, device="cpu"
        )
        with torch.no_grad():
            p_dev_s = torch.sigmoid(model_full(torch.from_numpy(np.ascontiguousarray(X_dv_mlp, dtype=np.float32)))).numpy()
        seed_dev_preds.append(p_dev_s)
        seed_oof_preds.append(oof_s)

        s_auc = float(roc_auc_score(y_dev, p_dev_s))
        s_ls = float(calc_length_stratified_auroc(y_dev, p_dev_s, tok_dev))
        s_cpp = float(roc_auc_score(y_cpp, p_dev_s[is_cpp]))

        seed_dev_aurocs.append(s_auc)
        seed_dev_ls_aurocs.append(s_ls)
        seed_dev_cpp_aurocs.append(s_cpp)

    # Ensemble: mean of 5 seeds
    p_dev_mlp_ens = np.mean(seed_dev_preds, axis=0)
    p_oof_mlp_ens = np.mean(seed_oof_preds, axis=0)

    # Threshold on ensemble OOF
    thr_info_mlp = threshold_at_fpr(y_train, p_oof_mlp_ens, target_fpr=0.10, target_prevalence=0.30)
    opt_thr_mlp = thr_info_mlp["threshold"]

    m_dev_mlp = metrics_full(y_dev, p_dev_mlp_ens, threshold=opt_thr_mlp)
    ci_dev_mlp = bootstrap_ci(y_dev, p_dev_mlp_ens, threshold=opt_thr_mlp, n_resamples=1000, seed=42)
    ls_dev_mlp = length_stratified_auroc(y_dev, p_dev_mlp_ens, tok_dev, n_resamples=1000, seed=42)

    p_cpp_mlp = p_dev_mlp_ens[is_cpp]
    cpp_auc_mlp = float(roc_auc_score(y_cpp, p_cpp_mlp))
    cpp_ci_mlp = bootstrap_ci(y_cpp, p_cpp_mlp, threshold=opt_thr_mlp, n_resamples=1000, seed=42)["auroc_ci_95"]

    diff_overall_mlp = paired_bootstrap_diff(
        y_dev, p_champ_dev, p_dev_mlp_ens, lambda y, p, _: roc_auc_score(y, p), groups=dev_commits, n_resamples=1000, seed=42
    )
    diff_ls_mlp = paired_bootstrap_diff(
        y_dev, p_champ_dev, p_dev_mlp_ens,
        lambda y, p, _, **extra: calc_length_stratified_auroc(y, p, extra["n_tokens"]),
        groups=dev_commits, extra_arrays={"n_tokens": tok_dev}, n_resamples=1000, seed=42
    )
    diff_cpp_mlp = paired_bootstrap_diff(
        y_cpp, p_champ_dev[is_cpp], p_cpp_mlp, lambda y, p, _: roc_auc_score(y, p),
        groups=dev_commits[is_cpp], n_resamples=1000, seed=42
    )

    res_H3 = {
        "candidate_id": "H3",
        "description": f"2-layer MLP on {best_rep_id} (hidden=256, dropout=0.3, ensemble of 5 seeds)",
        "category": "Classifier Head",
        "classifier": "TwoLayerMLP",
        "seeds": seeds,
        "seed_stats": {
            "dev_auroc_mean": float(np.mean(seed_dev_aurocs)),
            "dev_auroc_std": float(np.std(seed_dev_aurocs)),
            "dev_ls_auroc_mean": float(np.mean(seed_dev_ls_aurocs)),
            "dev_ls_auroc_std": float(np.std(seed_dev_ls_aurocs)),
            "dev_cpp_auroc_mean": float(np.mean(seed_dev_cpp_aurocs)),
            "dev_cpp_auroc_std": float(np.std(seed_dev_cpp_aurocs)),
        },
        "oof_threshold_info": thr_info_mlp,
        "oof_threshold": opt_thr_mlp,
        "dev_auroc": m_dev_mlp["auroc"],
        "dev_auroc_ci_95": ci_dev_mlp["auroc_ci_95"],
        "dev_length_stratified_auroc": ls_dev_mlp["length_stratified_auroc"],
        "dev_length_stratified_ci_95": ls_dev_mlp["ci_95"],
        "dev_cpp_auroc": cpp_auc_mlp,
        "dev_cpp_auroc_ci_95": cpp_ci_mlp,
        "dev_auprc": m_dev_mlp["auprc"],
        "dev_recall_at_10fpr": m_dev_mlp["recall"],
        "dev_precision_at_10fpr": m_dev_mlp["precision"],
        "dev_fpr_at_10fpr": m_dev_mlp["fpr"],
        "diff_vs_champion": {
            "overall_auroc": diff_overall_mlp,
            "length_stratified_auroc": diff_ls_mlp,
            "cpp_auroc": diff_cpp_mlp,
        },
        "p_dev": p_dev_mlp_ens.tolist(),
        "p_oof": p_oof_mlp_ens.tolist(),
    }
    print(f"  H3 Dev AUROC: {m_dev_mlp['auroc']:.4f} {ci_dev_mlp['auroc_ci_95']} (Across seeds: {res_H3['seed_stats']['dev_auroc_mean']:.4f} +/- {res_H3['seed_stats']['dev_auroc_std']:.4f})")
    print(f"  H3 Dev Length-Stratified AUROC: {ls_dev_mlp['length_stratified_auroc']:.4f} {ls_dev_mlp['ci_95']}")

    output_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "runtime_seconds": round(time.time() - t_start, 2),
            "representation_used": best_rep_id,
        },
        "heads": {
            "H1": {k: v for k, v in res_H1.items() if k not in ["p_dev", "p_oof"]},
            "H2": {k: v for k, v in res_H2.items() if k not in ["p_dev", "p_oof"]},
            "H3": {k: v for k, v in res_H3.items() if k not in ["p_dev", "p_oof"]},
        },
    }

    out_json = RESULTS_P3_DIR / "step4_heads.json"
    with open(out_json, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nStep 4 successfully saved to {out_json} in {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
