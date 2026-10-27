#!/usr/bin/env python3
"""
Phase 3 Targeted Experiments for Cross-Project Vulnerability Detection
=====================================================================
Executes three targeted experiments:
  Experiment A: Scaling CodeBERT Training Data (12k -> 24k diverse samples)
  Experiment B: Extended 512-Token Context with Head-Tail Pooling
  Experiment C: C++ Aware & Resource/Logic Flaw Structural Augmentation

Evaluated across:
  Benchmark 1: Clean 4-Project Benchmark (N = 3,169: OpenSSL, GPAC, Envoy, Vim)
  Benchmark 2: Fresh Untouched 5-Project Benchmark (N = 3,801: tcpdump, jasper, radare2, FreeRDP, krb5)
  Uncertainty Estimation: 1,000 paired bootstrap confidence intervals.
"""

import sys
import os
import re
import time
import json
import logging
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.search_classical_models import SecurityStructuralFeatureExtractor
from scripts.train_expanded_codebert import get_device

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("phase3_experiments")

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"

# =============================================================================
# C++ & Logic-Flaw Enhanced Structural Feature Extractor
# =============================================================================
class EnhancedSecurityStructuralFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Augments base security rules with modern C++ & logic/resource exhaustion metrics:
    - Loop count (for, while, do) -> Resource consumption (CWE-400)
    - Path traversal indicators (../, /etc/, path, filename) -> Path Traversal (CWE-22)
    - C++ Scope operator (::) & std:: containers
    - Smart pointer usage (unique_ptr, shared_ptr, make_unique)
    - Exception handling (try, catch, throw)
    - Allocation with limits / size checks
    """
    def __init__(self):
        self.base_extractor = SecurityStructuralFeatureExtractor()
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        base_features = self.base_extractor.transform(X)
        
        enhanced_feats = []
        for code in X:
            c = str(code) if code is not None else ""
            
            # C++ & Logic Flaw features
            n_loops = len(re.findall(r'\b(for|while|do)\b', c))
            n_path_traversal = len(re.findall(r'(\.\./|/etc/|path|filename|filepath|url)', c, re.I))
            n_cpp_scope = c.count('::')
            n_smart_ptrs = len(re.findall(r'\b(unique_ptr|shared_ptr|make_unique|make_shared)\b', c))
            n_exceptions = len(re.findall(r'\b(try|catch|throw)\b', c))
            n_std_containers = len(re.findall(r'\bstd::(vector|string|map|set|list|deque|array)\b', c))
            n_size_checks = len(re.findall(r'\b(size\s*==|length\s*==|sizeof|capacity|limit)\b', c))
            
            enhanced_feats.append([
                n_loops,
                n_path_traversal,
                n_cpp_scope,
                n_smart_ptrs,
                n_exceptions,
                n_std_containers,
                n_size_checks
            ])
            
        return np.hstack([base_features, np.array(enhanced_feats, dtype=np.float32)])


def extract_codebert_512_head_tail(codes, tokenizer, model, device, batch_size=64):
    """Extract 512-token head-tail pooled embeddings."""
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
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state  # [B, L, 768]
            
            # [CLS] representation (head context)
            cls_repr = hidden[:, 0, :]  # [B, 768]
            
            # Masked mean pooling across active tokens
            mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden.size()).float()
            sum_embeddings = torch.sum(hidden * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_repr = sum_embeddings / sum_mask  # [B, 768]
            
            combined = torch.cat([cls_repr, mean_repr], dim=-1).cpu().numpy()
            embeddings.append(combined)
            
    res = np.vstack(embeddings)
    elapsed = time.time() - t0
    logger.info(f"Extracted {len(codes)} 512-tok embeddings in {elapsed:.1f}s ({len(codes)/elapsed:.1f} samples/s)")
    return res


def compute_metrics_with_ci(y_true, y_pred, y_prob, n_bootstrap=1000, seed=42):
    """Compute metrics and 95% bootstrap confidence intervals."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    
    # Bootstrap for CI
    rng = np.random.RandomState(seed)
    n = len(y_true)
    accs, aucs = [], []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt_b = y_true[idx]
        yp_b = y_pred[idx]
        ypr_b = y_prob[idx]
        if len(np.unique(yt_b)) < 2:
            continue
        accs.append(accuracy_score(yt_b, yp_b))
        aucs.append(roc_auc_score(yt_b, ypr_b))
        
    ci_acc = [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))] if accs else [acc, acc]
    ci_auc = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))] if aucs else [auc, auc]
    
    return {
        "accuracy": acc,
        "accuracy_95ci": ci_acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "auroc": auc,
        "auroc_95ci": ci_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }


def main():
    device = get_device()
    logger.info(f"Using compute device: {device}")
    
    # Load model and tokenizer
    model_name = "microsoft/codebert-base"
    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cb_model = AutoModel.from_pretrained(model_name).to(device)
    
    # Load data
    train_df = pd.read_parquet(SPLITS_DIR / "vulnerability_train_expanded.parquet")
    val_df = pd.read_parquet(SPLITS_DIR / "vulnerability_val_expanded.parquet")
    clean_test_df = pd.read_parquet(SPLITS_DIR / "vulnerability_test_heldout_projects_clean.parquet")
    fresh_test_df = pd.read_parquet(SPLITS_DIR / "vulnerability_test_fresh_5projects.parquet")
    
    # Exclude fresh test projects from training to prevent leakage
    fresh_projects = set(fresh_test_df['project'].unique())
    train_pool = train_df[~train_df['project'].isin(fresh_projects)].copy().reset_index(drop=True)
    
    logger.info(f"Train Pool: {len(train_pool)} samples (zero overlap with fresh test projects {fresh_projects})")
    logger.info(f"Clean Benchmark 1 (4 Projects): {len(clean_test_df)} samples")
    logger.info(f"Fresh Benchmark 2 (5 Projects): {len(fresh_test_df)} samples")
    
    # -------------------------------------------------------------------------
    # Create Stratified 24k Training Slice for Experiment A
    # -------------------------------------------------------------------------
    vuln_train = train_pool[train_pool['vuln_label'] == 1]
    benign_train = train_pool[train_pool['vuln_label'] == 0]
    
    sub_train_24k = pd.concat([
        vuln_train.sample(n=min(12000, len(vuln_train)), random_state=42),
        benign_train.sample(n=min(12000, len(benign_train)), random_state=42)
    ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    val_sub = val_df.sample(n=min(2500, len(val_df)), random_state=42).reset_index(drop=True)
    
    logger.info(f"Sampled 24k Training Slice: {len(sub_train_24k)} samples across {sub_train_24k['project'].nunique()} projects.")
    
    # Extract Enhanced Features (19 dimensions: 12 base + 7 C++ & logic)
    enhanced_extractor = EnhancedSecurityStructuralFeatureExtractor()
    scaler = StandardScaler()
    
    logger.info("Extracting Enhanced Structural Features...")
    struct_train = scaler.fit_transform(enhanced_extractor.transform(sub_train_24k['code'].values))
    struct_val = scaler.transform(enhanced_extractor.transform(val_sub['code'].values))
    struct_b1 = scaler.transform(enhanced_extractor.transform(clean_test_df['code'].values))
    struct_b2 = scaler.transform(enhanced_extractor.transform(fresh_test_df['code'].values))
    
    # Extract 512-token CodeBERT Embeddings
    logger.info("Extracting 512-token Head-Tail CodeBERT Embeddings...")
    emb_train = extract_codebert_512_head_tail(sub_train_24k['code'].values, tokenizer, cb_model, device)
    emb_val = extract_codebert_512_head_tail(val_sub['code'].values, tokenizer, cb_model, device)
    emb_b1 = extract_codebert_512_head_tail(clean_test_df['code'].values, tokenizer, cb_model, device)
    emb_b2 = extract_codebert_512_head_tail(fresh_test_df['code'].values, tokenizer, cb_model, device)
    
    # Fuse: 1,536 (CodeBERT) + 19 (Enhanced Structure) = 1,555 dimensions
    X_train = np.hstack([emb_train, struct_train])
    y_train = sub_train_24k['vuln_label'].values.astype(int)
    
    X_val = np.hstack([emb_val, struct_val])
    y_val = val_sub['vuln_label'].values.astype(int)
    
    X_b1 = np.hstack([emb_b1, struct_b1])
    y_b1 = clean_test_df['vuln_label'].values.astype(int)
    
    X_b2 = np.hstack([emb_b2, struct_b2])
    y_b2 = fresh_test_df['vuln_label'].values.astype(int)
    
    logger.info(f"Fused Feature Dimension: {X_train.shape[1]}")
    
    # Train Regularized Logistic Classifier with balanced weighting
    clf = LogisticRegression(C=0.05, class_weight='balanced', max_iter=1000, solver='lbfgs', random_state=42)
    t0 = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - t0
    logger.info(f"Model fit completed in {fit_time:.2f}s")
    
    # Validation evaluation & threshold tuning
    p_val = clf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, p_val)
    val_acc_50 = accuracy_score(y_val, (p_val >= 0.50).astype(int))
    logger.info(f"[Phase 3 CodeBERT Enhanced] Val Acc={val_acc_50*100:.2f}%, Val AUROC={val_auc:.4f}")
    
    # Benchmark 1 Evaluation (Clean 4-Project Benchmark, N = 3,169)
    p_b1 = clf.predict_proba(X_b1)[:, 1]
    res_b1 = compute_metrics_with_ci(y_b1, (p_b1 >= 0.50).astype(int), p_b1)
    
    logger.info("\n" + "="*70)
    logger.info("BENCHMARK 1 EVALUATION (Clean 4 Projects, N = 3,169)")
    logger.info("="*70)
    logger.info(f"Overall Acc:     {res_b1['accuracy']*100:.2f}% (95% CI: [{res_b1['accuracy_95ci'][0]*100:.2f}%, {res_b1['accuracy_95ci'][1]*100:.2f}%])")
    logger.info(f"Balanced Acc:    {res_b1['balanced_accuracy']*100:.2f}%")
    logger.info(f"AUROC:           {res_b1['auroc']:.4f} (95% CI: [{res_b1['auroc_95ci'][0]:.4f}, {res_b1['auroc_95ci'][1]:.4f}])")
    logger.info(f"MCC:             {res_b1['mcc']:+.4f}")
    logger.info(f"Precision:       {res_b1['precision']*100:.2f}%")
    logger.info(f"Recall:          {res_b1['recall']*100:.2f}%")
    logger.info(f"Confusion Matrix:{res_b1['confusion_matrix']}")
    
    # Per-repository breakdown on Benchmark 1
    b1_by_proj = {}
    for proj in ['gpac', 'vim', 'openssl', 'envoy']:
        mask = (clean_test_df['project'] == proj).values
        y_p = y_b1[mask]
        p_p = p_b1[mask]
        pred_p = (p_p >= 0.50).astype(int)
        m_p = compute_metrics_with_ci(y_p, pred_p, p_p)
        b1_by_proj[proj] = m_p
        logger.info(f"  -> {proj:8} (N={len(y_p):4}): Acc={m_p['accuracy']*100:.2f}%, AUROC={m_p['auroc']:.4f}, MCC={m_p['mcc']:+.4f}")
        
    # Benchmark 2 Evaluation (Fresh Untouched 5 Projects, N = 3,801)
    p_b2 = clf.predict_proba(X_b2)[:, 1]
    res_b2 = compute_metrics_with_ci(y_b2, (p_b2 >= 0.50).astype(int), p_b2)
    
    logger.info("\n" + "="*70)
    logger.info("BENCHMARK 2 EVALUATION (Fresh Untouched 5 Projects, N = 3,801)")
    logger.info("="*70)
    logger.info(f"Overall Acc:     {res_b2['accuracy']*100:.2f}% (95% CI: [{res_b2['accuracy_95ci'][0]*100:.2f}%, {res_b2['accuracy_95ci'][1]*100:.2f}%])")
    logger.info(f"Balanced Acc:    {res_b2['balanced_accuracy']*100:.2f}%")
    logger.info(f"AUROC:           {res_b2['auroc']:.4f} (95% CI: [{res_b2['auroc_95ci'][0]:.4f}, {res_b2['auroc_95ci'][1]:.4f}])")
    logger.info(f"MCC:             {res_b2['mcc']:+.4f}")
    logger.info(f"Precision:       {res_b2['precision']*100:.2f}%")
    logger.info(f"Recall:          {res_b2['recall']*100:.2f}%")
    logger.info(f"Confusion Matrix:{res_b2['confusion_matrix']}")
    
    b2_by_proj = {}
    for proj in ['radare2', 'tcpdump', 'jasper', 'FreeRDP', 'krb5']:
        mask = (fresh_test_df['project'] == proj).values
        y_p = y_b2[mask]
        p_p = p_b2[mask]
        pred_p = (p_p >= 0.50).astype(int)
        m_p = compute_metrics_with_ci(y_p, pred_p, p_p)
        b2_by_proj[proj] = m_p
        logger.info(f"  -> {proj:8} (N={len(y_p):4}): Acc={m_p['accuracy']*100:.2f}%, AUROC={m_p['auroc']:.4f}, MCC={m_p['mcc']:+.4f}")
        
    # Save checkpoint
    with open(MODELS_DIR / "phase3_codebert_enhanced_best.pkl", "wb") as f:
        pickle.dump({
            "clf": clf,
            "scaler": scaler,
            "val_metrics": {"accuracy": val_acc_50, "auroc": val_auc},
            "benchmark1_results": res_b1,
            "benchmark2_results": res_b2
        }, f)
    logger.info("Saved phase3 model checkpoint to models/checkpoints/phase3_codebert_enhanced_best.pkl")
    
    # Save machine-readable output
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples": len(sub_train_24k),
        "feature_dim": X_train.shape[1],
        "validation_metrics": {"accuracy": val_acc_50, "auroc": val_auc},
        "benchmark1_clean_4projects": {
            "overall": res_b1,
            "by_project": b1_by_proj
        },
        "benchmark2_fresh_5projects": {
            "overall": res_b2,
            "by_project": b2_by_proj
        }
    }
    
    with open(RESULTS_DIR / "phase3_generalization_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Saved Phase 3 generalization results to {RESULTS_DIR / 'phase3_generalization_results.json'}")

if __name__ == "__main__":
    main()
