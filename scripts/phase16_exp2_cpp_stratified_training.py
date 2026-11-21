#!/usr/bin/env python3
"""
Phase 16 — Experiment 2: C++ and Cross-Project Stratified 35k Training
=====================================================================
Addresses the critical finding that the Phase 15 25k training split contained
only 1 Envoy sample and less than 6% C++.

Constructs and evaluates a C++-enriched, project-balanced 35,000-sample training dataset:
- Exactly 50.0% vulnerable / 50.0% safe (17,500 vuln, 17,500 safe)
- 20.0%+ modern C++ representation (Chrome, TensorFlow, Poppler, C++ templates/classes)
- Strict exclusion of Envoy Proxy (0 Envoy samples in training to guarantee pure unseen zero-shot evaluation)
- Combined with Phase 16 38-dimensional Omission-Aware feature extractor
- C tuned strictly on 3,000-sample validation set
- Checkpoint saved as models/checkpoints/phase16_codebert_cxx_stratified_35k.pkl
"""

import sys
import os
import re
import json
import time
import pickle
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_P16_DIR = PROJECT_ROOT / "results" / "phase16"
RESULTS_P15_DIR = PROJECT_ROOT / "results" / "phase15"
RESULTS_P14_DIR = PROJECT_ROOT / "results" / "phase14"
RESULTS_P16_DIR.mkdir(parents=True, exist_ok=True)

from scripts.phase16_omission_extractor import Phase16OmissionAwareCxxExtractor


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_codebert_embeddings(code_list, tokenizer, model, device, batch_size=64):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(code_list), batch_size):
            batch_codes = [str(c)[:2048] for c in code_list[i:i + batch_size]]
            inputs = tokenizer(batch_codes, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state  # (B, L, 768)
            cls_emb = last_hidden[:, 0, :].cpu().numpy()  # (B, 768)
            mean_emb = last_hidden.mean(dim=1).cpu().numpy()  # (B, 768)
            combined = np.hstack([cls_emb, mean_emb])  # (B, 1536)
            embeddings.append(combined)
    return np.vstack(embeddings)


def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = 0.50
    try:
        prauc = float(average_precision_score(y_true, y_prob))
    except Exception:
        prauc = float(np.mean(y_true))

    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "auroc": auroc,
        "pr_auc": prauc,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "fpr": fpr,
        "fnr": fnr
    }


def main():
    print("=" * 80)
    print("PHASE 16 — EXPERIMENT 2: C++ & CROSS-PROJECT STRATIFIED 35K TRAINING")
    print("=" * 80)

    # 1. Load Parquet Pools
    print("\n[1] Constructing 35,000 C++-Enriched Stratified Training Split...")
    train_pool = pd.read_parquet(SPLITS_DIR / "vulnerability_train_expanded.parquet")

    # Strictly exclude Envoy from training to preserve unseen evaluation purity
    train_pool_no_envoy = train_pool[train_pool['project'].str.lower() != 'envoy'].copy()

    # Identify C++ samples
    cpp_kw = r'\b(class|public:|private:|protected:|namespace|std::|template|typename|unique_ptr|shared_ptr|static_cast|reinterpret_cast)\b'
    is_cpp = train_pool_no_envoy['code'].str.contains(cpp_kw, regex=True) | train_pool_no_envoy['project'].isin(['Chrome', 'tensorflow', 'poppler'])

    cpp_pool_vuln = train_pool_no_envoy[is_cpp & (train_pool_no_envoy['vuln_label'] == 1)]
    cpp_pool_safe = train_pool_no_envoy[is_cpp & (train_pool_no_envoy['vuln_label'] == 0)]
    c_pool_vuln = train_pool_no_envoy[~is_cpp & (train_pool_no_envoy['vuln_label'] == 1)]
    c_pool_safe = train_pool_no_envoy[~is_cpp & (train_pool_no_envoy['vuln_label'] == 0)]

    print(f"Available Pool (excluding Envoy):")
    print(f"  C++ Vuln: {len(cpp_pool_vuln):,} | C++ Safe: {len(cpp_pool_safe):,}")
    print(f"  C Vuln:   {len(c_pool_vuln):,} | C Safe:   {len(c_pool_safe):,}")

    # Sample 3,500 C++ vuln + 3,500 C++ safe = 7,000 C++ samples (20.0%)
    # Sample 14,000 C vuln + 14,000 C safe = 28,000 C samples (80.0%)
    # Total = 35,000 balanced samples (17,500 vuln, 17,500 safe)
    n_cpp_per_class = min(3500, len(cpp_pool_vuln), len(cpp_pool_safe))
    n_c_per_class = 17500 - n_cpp_per_class

    train_35k = pd.concat([
        cpp_pool_vuln.sample(n=n_cpp_per_class, random_state=42),
        cpp_pool_safe.sample(n=n_cpp_per_class, random_state=42),
        c_pool_vuln.sample(n=n_c_per_class, random_state=42),
        c_pool_safe.sample(n=n_c_per_class, random_state=42)
    ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    X_train_35k = train_35k['code'].values
    y_train_35k = train_35k['vuln_label'].values.astype(int)

    print(f"\nFinal 35,000 Stratified Training Set:")
    print(f"  Total Samples:    {len(X_train_35k)}")
    print(f"  Vulnerability:    {y_train_35k.mean()*100:.1f}% ({y_train_35k.sum()}/{len(y_train_35k)})")
    print(f"  Modern C++ Share: {2 * n_cpp_per_class / len(X_train_35k)*100:.1f}% ({2*n_cpp_per_class} samples)")
    print(f"  Envoy Samples:    0 (Strictly Excluded)")

    # 2. Embeddings Cache Check / Extraction
    emb_35k_path = RESULTS_P16_DIR / "codebert_train_35k_emb.npy"
    if emb_35k_path.exists():
        print(f"\n[2] Loading existing 35k CodeBERT embeddings: {emb_35k_path.name}")
        emb_train_35k = np.load(emb_35k_path)
    else:
        print(f"\n[2] Extracting CodeBERT embeddings for 35,000 samples on {get_device()}...")
        device = get_device()
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        cb_model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
        t0 = time.time()
        emb_train_35k = extract_codebert_embeddings(X_train_35k, tokenizer, cb_model, device, batch_size=64)
        np.save(emb_35k_path, emb_train_35k)
        print(f"  35k extraction completed and saved in {time.time()-t0:.1f}s")

    # 3. Load Validation and Test Datasets
    val_df = pd.read_parquet(SPLITS_DIR / "vulnerability_val_expanded.parquet")
    val_3k = val_df.sample(n=3000, random_state=42).reset_index(drop=True)
    X_val = val_3k['code'].values
    y_val = val_3k['vuln_label'].values.astype(int)

    test_df = pd.read_parquet(SPLITS_DIR / "vulnerability_test_heldout_projects_clean.parquet")
    X_test = test_df['code'].values
    y_test = test_df['vuln_label'].values.astype(int)
    projects = test_df['project'].values

    emb_val = np.load(RESULTS_P14_DIR / "codebert_val_emb.npy")
    emb_test = np.load(RESULTS_P14_DIR / "codebert_test_clean_emb.npy")

    # 4. Extract 38-dim Omission-Aware Features
    print("\n[3] Extracting 38-dim Omission-Aware Features across 35k Train, Val, Test...")
    extractor = Phase16OmissionAwareCxxExtractor()
    scaler = StandardScaler()

    s_train_38 = scaler.fit_transform(extractor.transform(X_train_35k))
    s_val_38 = scaler.transform(extractor.transform(X_val))
    s_test_38 = scaler.transform(extractor.transform(X_test))

    X_tr_cand2 = np.hstack([emb_train_35k, s_train_38])
    X_val_cand2 = np.hstack([emb_val, s_val_38])
    X_te_cand2 = np.hstack([emb_test, s_test_38])

    # 5. Validation Tuning of Regularization C
    print("\n[4] Tuning Regularization C on Validation Set...")
    best_c_2 = 0.05
    best_val_auc_2 = 0.0
    for c_cand in [0.005, 0.01, 0.05, 0.1, 0.2, 0.5]:
        clf_temp = LogisticRegression(C=c_cand, class_weight='balanced', max_iter=1000, random_state=42)
        clf_temp.fit(X_tr_cand2, y_train_35k)
        p_val = clf_temp.predict_proba(X_val_cand2)[:, 1]
        auc_v = roc_auc_score(y_val, p_val)
        if auc_v > best_val_auc_2:
            best_val_auc_2 = auc_v
            best_c_2 = c_cand
        print(f"    C = {c_cand:5.3f} -> Val AUROC = {auc_v:.4f}")

    print(f"  Selected C = {best_c_2} (Val AUROC = {best_val_auc_2:.4f})")

    # 6. Fit Candidate 2 Checkpoint
    clf_cand2 = LogisticRegression(C=best_c_2, class_weight='balanced', max_iter=1000, random_state=42)
    clf_cand2.fit(X_tr_cand2, y_train_35k)
    prob_cand2 = clf_cand2.predict_proba(X_te_cand2)[:, 1]
    pred_cand2 = (prob_cand2 >= 0.50).astype(int)

    m_cand2 = compute_metrics(y_test, pred_cand2, prob_cand2)

    print(f"\nCandidate 2 Performance (35k C++ Stratified + 38-dim Features):")
    print(f"  Accuracy:          {m_cand2['accuracy']*100:.2f}%")
    print(f"  AUROC:             {m_cand2['auroc']:.4f}")
    print(f"  F1 Score:          {m_cand2['f1']:.4f}")
    print(f"  Precision:         {m_cand2['precision']*100:.2f}%")
    print(f"  Recall:            {m_cand2['recall']*100:.2f}%")
    print(f"  FPR:               {m_cand2['fpr']*100:.2f}%")

    # Save Checkpoint
    ckpt_cand2_path = MODELS_DIR / "phase16_codebert_cxx_stratified_35k.pkl"
    with open(ckpt_cand2_path, "wb") as f:
        pickle.dump({
            "clf": clf_cand2,
            "scaler": scaler,
            "feature_names": extractor.get_feature_names(),
            "best_c": best_c_2,
            "val_score": float(best_val_auc_2),
            "train_size": len(X_train_35k),
            "cpp_share": 2 * n_cpp_per_class / len(X_train_35k),
            "model_id": "phase16_codebert_cxx_stratified_35k"
        }, f)
    print(f"  Saved Candidate 2 Checkpoint to: {ckpt_cand2_path}")

    # Project Breakdown
    print(f"\nProject-wise Breakdown for Candidate 2:")
    proj_cand2 = {}
    for p in sorted(np.unique(projects)):
        p_mask = (projects == p)
        m_p = compute_metrics(y_test[p_mask], pred_cand2[p_mask], prob_cand2[p_mask])
        proj_cand2[p] = m_p
        print(f"  • {p:10}: Acc = {m_p['accuracy']*100:.2f}%, AUROC = {m_p['auroc']:.4f}, FPR = {m_p['fpr']*100:.1f}%, Rec = {m_p['recall']*100:.1f}%")

    # Save Experiment 2 results
    out_exp2_path = RESULTS_P16_DIR / "exp2_cpp_stratified_training.json"
    with open(out_exp2_path, "w") as f:
        json.dump({
            "candidate2_overall": m_cand2,
            "candidate2_by_project": proj_cand2,
            "training_specs": {
                "total_samples": len(X_train_35k),
                "cpp_samples": 2 * n_cpp_per_class,
                "cpp_share": 2 * n_cpp_per_class / len(X_train_35k),
                "c_samples": 2 * n_c_per_class,
                "best_c": best_c_2,
                "val_auroc": best_val_auc_2
            }
        }, f, indent=2)

    print(f"\n[✓] Experiment 2 results saved to: {out_exp2_path}")


if __name__ == "__main__":
    main()
