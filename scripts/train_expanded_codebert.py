#!/usr/bin/env python3
"""
Expanded Multi-Project CodeBERT Hybrid Model Training & Evaluation (Phase 4 & 5)
===============================================================================
Extracts deep contextual embeddings using CodeBERT (microsoft/codebert-base)
with hybrid representation:
  - Token context: 384 tokens
  - Hidden representation: [CLS] + Token Mean Pooling (1,536 dims)
  - Security Structural Feature Fusion (+12 static security tokens)
  - Regularized Classifier with Balanced Class Weighting

Evaluates strictly across:
  1. Expanded Multi-Project Validation Set (N=2,000, 829 projects)
  2. Untouched Original Test Set (N=5,412, Devign & PrimeVul)
  3. Novel Held-Out Projects Test Set (N=3,325, Vim, OpenSSL, Envoy, GPAC)
  4. Synthetic NIST Juliet Test Set (N=4,000, Clean synthetic CWEs)
"""

import sys
import os
import time
import json
import logging
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.search_classical_models import SecurityStructuralFeatureExtractor
from src.security.static_analysis import analyze_code

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("expanded_codebert")

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def extract_codebert_embeddings(codes, tokenizer, model, device, batch_size=64, max_length=384):
    """Extract CLS + Mean-Pooled embeddings on MPS device."""
    model.eval()
    embeddings = []
    
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(codes), batch_size):
            batch_codes = [str(c) if c is not None else "" for c in codes[i:i + batch_size]]
            inputs = tokenizer(
                batch_codes,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)
            
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state  # [B, L, 768]
            
            # [CLS] representation
            cls_repr = hidden[:, 0, :]  # [B, 768]
            
            # Masked mean pooling across non-padding tokens
            mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden.size()).float()
            sum_embeddings = torch.sum(hidden * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_repr = sum_embeddings / sum_mask  # [B, 768]
            
            combined = torch.cat([cls_repr, mean_repr], dim=-1).cpu().numpy()  # [B, 1536]
            embeddings.append(combined)
            
    res = np.vstack(embeddings)
    elapsed = time.time() - t0
    logger.info(f"Extracted {len(codes)} embeddings in {elapsed:.1f}s ({len(codes)/elapsed:.1f} samples/s)")
    return res


def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "positive_rate": float(np.mean(y_pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "fpr": fpr,
        "fnr": fnr
    }


def find_optimal_threshold(y_true, y_prob, metric="acc"):
    best_t = 0.50
    best_val = -1.0
    for t in np.arange(0.20, 0.81, 0.02):
        pred = (y_prob >= t).astype(int)
        if metric == "acc":
            score = accuracy_score(y_true, pred)
        elif metric == "mcc":
            score = matthews_corrcoef(y_true, pred)
        else:
            score = balanced_accuracy_score(y_true, pred)
            
        if score > best_val:
            best_val = score
            best_t = float(t)
    return best_t


def main():
    device = get_device()
    logger.info(f"Using compute device: {device}")
    
    # Load model and tokenizer
    model_name = "microsoft/codebert-base"
    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    # Load splits
    train_df = pd.read_parquet(SPLITS_DIR / "vulnerability_train_expanded.parquet")
    val_df = pd.read_parquet(SPLITS_DIR / "vulnerability_val_expanded.parquet")
    
    test_orig_df = pd.read_parquet(SPLITS_DIR / "vulnerability_test.parquet")
    test_heldout_df = pd.read_parquet(SPLITS_DIR / "vulnerability_test_heldout_projects.parquet")
    test_juliet_df = pd.read_parquet(SPLITS_DIR / "vulnerability_test_synthetic_juliet.parquet")
    
    # Create diverse multi-project training slice (12,000 balanced samples for CodeBERT fine-head)
    vuln_train = train_df[train_df['vuln_label'] == 1]
    benign_train = train_df[train_df['vuln_label'] == 0]
    
    sub_train = pd.concat([
        vuln_train.sample(n=6000, random_state=42),
        benign_train.sample(n=6000, random_state=42)
    ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    val_sub = val_df.sample(n=2000, random_state=42).reset_index(drop=True)
    
    logger.info("Extracting structural security features...")
    extractor = SecurityStructuralFeatureExtractor()
    scaler = StandardScaler()
    
    struct_train = scaler.fit_transform(extractor.fit_transform(sub_train['code'].values))
    struct_val = scaler.transform(extractor.transform(val_sub['code'].values))
    struct_test_orig = scaler.transform(extractor.transform(test_orig_df['code'].values))
    struct_test_heldout = scaler.transform(extractor.transform(test_heldout_df['code'].values))
    struct_test_juliet = scaler.transform(extractor.transform(test_juliet_df['code'].values))
    
    logger.info("Extracting CodeBERT embeddings...")
    emb_train = extract_codebert_embeddings(sub_train['code'].values, tokenizer, model, device)
    emb_val = extract_codebert_embeddings(val_sub['code'].values, tokenizer, model, device)
    emb_test_orig = extract_codebert_embeddings(test_orig_df['code'].values, tokenizer, model, device)
    emb_test_heldout = extract_codebert_embeddings(test_heldout_df['code'].values, tokenizer, model, device)
    emb_test_juliet = extract_codebert_embeddings(test_juliet_df['code'].values, tokenizer, model, device)
    
    # Fuse deep contextual embeddings + static security features (1,548 dims)
    X_train = np.hstack([emb_train, struct_train])
    y_train = sub_train['vuln_label'].values.astype(int)
    
    X_val = np.hstack([emb_val, struct_val])
    y_val = val_sub['vuln_label'].values.astype(int)
    
    X_test_orig = np.hstack([emb_test_orig, struct_test_orig])
    y_test_orig = test_orig_df['vuln_label'].values.astype(int)
    
    X_test_heldout = np.hstack([emb_test_heldout, struct_test_heldout])
    y_test_heldout = test_heldout_df['vuln_label'].values.astype(int)
    
    X_test_juliet = np.hstack([emb_test_juliet, struct_test_juliet])
    y_test_juliet = test_juliet_df['vuln_label'].values.astype(int)
    
    logger.info(f"Fused feature dimension: {X_train.shape[1]}")
    
    # Train regularized logistic classifier with balanced class weights
    clf = LogisticRegression(C=0.01, class_weight='balanced', max_iter=1000, solver='lbfgs', random_state=42)
    t0 = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - t0
    logger.info(f"Classifier fit completed in {fit_time:.2f}s")
    
    # Validation evaluation & threshold tuning
    val_probs = clf.predict_proba(X_val)[:, 1]
    t_opt_acc = find_optimal_threshold(y_val, val_probs, metric="acc")
    
    val_m_50 = compute_metrics(y_val, (val_probs >= 0.50).astype(int), val_probs)
    val_m_opt = compute_metrics(y_val, (val_probs >= t_opt_acc).astype(int), val_probs)
    
    logger.info(f"[Expanded CodeBERT] Val (t=0.50): Acc={val_m_50['accuracy']*100:.2f}%, AUROC={val_m_50['auroc']:.4f}, MCC={val_m_50['mcc']:+.4f}")
    logger.info(f"[Expanded CodeBERT] Val (t={t_opt_acc:.2f}): Acc={val_m_opt['accuracy']*100:.2f}%, AUROC={val_m_opt['auroc']:.4f}, MCC={val_m_opt['mcc']:+.4f}")
    
    # Evaluate across the 3 test benchmarks
    test_evals = {}
    benchmarks = {
        "untouched_original_test": (X_test_orig, y_test_orig),
        "novel_projects_test": (X_test_heldout, y_test_heldout),
        "synthetic_juliet_test": (X_test_juliet, y_test_juliet)
    }
    
    for b_name, (X_t, y_t) in benchmarks.items():
        t_infer = time.time()
        probs = clf.predict_proba(X_t)[:, 1]
        infer_elapsed = time.time() - t_infer
        
        m_50 = compute_metrics(y_t, (probs >= 0.50).astype(int), probs)
        m_opt = compute_metrics(y_t, (probs >= t_opt_acc).astype(int), probs)
        
        logger.info(f"[Expanded CodeBERT -> {b_name}] Acc={m_50['accuracy']*100:.2f}%, AUROC={m_50['auroc']:.4f}, MCC={m_50['mcc']:+.4f} (t=0.50)")
        logger.info(f"[Expanded CodeBERT -> {b_name}] Acc={m_opt['accuracy']*100:.2f}%, AUROC={m_opt['auroc']:.4f}, MCC={m_opt['mcc']:+.4f} (t={t_opt_acc:.2f})")
        
        test_evals[b_name] = {
            "metrics_default_t50": m_50,
            "metrics_calibrated_topt": m_opt,
            "inference_time_sec": infer_elapsed,
            "n_samples": len(X_t)
        }
        
    # Save checkpoint
    with open(MODELS_DIR / "expanded_codebert_hybrid_best.pkl", "wb") as f:
        pickle.dump({
            "clf": clf,
            "scaler": scaler,
            "val_threshold_acc": t_opt_acc,
            "val_metrics": val_m_opt,
            "test_evaluations": test_evals
        }, f)
    logger.info("Saved expanded CodeBERT hybrid checkpoint to models/checkpoints/expanded_codebert_hybrid_best.pkl")
    
    # Save results json
    results_path = RESULTS_DIR / "expanded_codebert_evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "CodeBERT Expanded Hybrid (384-tok, MeanPool+CLS+SecurityStruct)",
            "training_samples": len(X_train),
            "val_threshold_acc": t_opt_acc,
            "val_metrics": val_m_opt,
            "test_evaluations": test_evals
        }, f, indent=2)
    logger.info(f"Saved expanded CodeBERT results to {results_path}")


if __name__ == "__main__":
    main()
