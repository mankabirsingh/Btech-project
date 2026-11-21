#!/usr/bin/env python3
"""
Phase 4: Envoy Vulnerability Detection Rescue & C++ Domain Adaptation
====================================================================
Implements:
  1. Complete Metric Audit & Discrepancy Resolution on Envoy
  2. Experiment A: Threshold & Probability Calibration Analysis (Validation-only)
  3. Experiment B: C++-Aware Vulnerability Training (Balanced C/C++ Corpus)
  4. Experiment C: Patch-Aware Commit Ranking & Classification Evaluation
  5. Validation-Driven Model Selection
  6. Single Final Evaluation on Locked Held-Out Benchmark & Envoy

Author: Research Team (DTU B.Tech Project-I)
"""

import sys
import os
import time
import json
import logging
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss
)
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import clean_code
from scripts.train_expanded_codebert import get_device, extract_codebert_embeddings
from scripts.run_phase3_experiments import (
    extract_codebert_512_head_tail,
    EnhancedSecurityStructuralFeatureExtractor,
    compute_metrics_with_ci
)
from scripts.search_classical_models import SecurityStructuralFeatureExtractor, c_tokenize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("phase4_rescue")

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def is_cpp_function(code_series: pd.Series) -> pd.Series:
    """Identify C++ functions by language keywords and syntactic tokens."""
    return (
        code_series.str.contains('::', regex=False) |
        code_series.str.contains('std::', regex=False) |
        code_series.str.contains('nullptr', regex=False) |
        code_series.str.contains(r'\bclass\b', regex=True)
    )


def compute_comprehensive_metrics(y_true, y_prob, threshold=0.50):
    """Compute full metric suite at a specified threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    
    # Review budget metrics: Recall at top 20%, 30%, 40% flagged
    n = len(y_true)
    sorted_indices = np.argsort(y_prob)[::-1]
    y_sorted = y_true[sorted_indices]
    total_pos = int(np.sum(y_true))
    
    rec_at_20 = float(np.sum(y_sorted[:int(n * 0.20)]) / total_pos) if total_pos > 0 else 0.0
    rec_at_30 = float(np.sum(y_sorted[:int(n * 0.30)]) / total_pos) if total_pos > 0 else 0.0
    rec_at_40 = float(np.sum(y_sorted[:int(n * 0.40)]) / total_pos) if total_pos > 0 else 0.0
    
    return {
        "threshold": float(threshold),
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
        "flagged_rate": float(np.mean(y_pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "fpr": fpr,
        "fnr": fnr,
        "recall_at_20_budget": rec_at_20,
        "recall_at_30_budget": rec_at_30,
        "recall_at_40_budget": rec_at_40
    }


def main():
    device = get_device()
    logger.info(f"Phase 4 Execution using device: {device}")
    
    # -------------------------------------------------------------------------
    # 1. Load Data Partitions
    # -------------------------------------------------------------------------
    logger.info("Loading dataset splits...")
    train_df = pd.read_parquet(SPLITS_DIR / "vulnerability_train_expanded.parquet")
    val_df = pd.read_parquet(SPLITS_DIR / "vulnerability_val_expanded.parquet")
    clean_test_df = pd.read_parquet(SPLITS_DIR / "vulnerability_test_heldout_projects_clean.parquet")
    fresh_test_df = pd.read_parquet(SPLITS_DIR / "vulnerability_test_fresh_5projects.parquet")
    
    # Locked Envoy test slice
    envoy_test_df = clean_test_df[clean_test_df['project'] == 'envoy'].reset_index(drop=True)
    y_test_envoy = envoy_test_df['vuln_label'].values.astype(int)
    
    logger.info(f"Loaded train: {len(train_df)}, val: {len(val_df)}, clean test: {len(clean_test_df)} (Envoy: {len(envoy_test_df)})")
    
    # Verify Envoy test statistics
    n_envoy = len(envoy_test_df)
    n_envoy_pos = int(np.sum(y_test_envoy))
    n_envoy_neg = n_envoy - n_envoy_pos
    envoy_vuln_rate = float(n_envoy_pos / n_envoy)
    logger.info(f"Envoy Test Set: N={n_envoy}, Pos(Vuln)={n_envoy_pos}, Neg(Benign)={n_envoy_neg}, Rate={envoy_vuln_rate*100:.2f}%")
    
    # Construct Repository-Separated C++ Validation Partition (Zero Envoy overlap)
    val_cpp_mask = is_cpp_function(val_df['code'])
    val_cpp_all = val_df[val_cpp_mask].copy().reset_index(drop=True)
    logger.info(f"Total C++ Validation Pool: {len(val_cpp_all)} samples across {val_cpp_all['project'].nunique()} repositories.")
    
    # Stratified validation slice (N = 2,000: 500 vulnerable + 1,500 benign to mirror ~25% base rate)
    val_cpp_vuln = val_cpp_all[val_cpp_all['vuln_label'] == 1]
    val_cpp_benign = val_cpp_all[val_cpp_all['vuln_label'] == 0]
    val_cpp_sub = pd.concat([
        val_cpp_vuln.sample(n=min(500, len(val_cpp_vuln)), random_state=42),
        val_cpp_benign.sample(n=min(1500, len(val_cpp_benign)), random_state=42)
    ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    X_val_cpp = val_cpp_sub['code'].values
    y_val_cpp = val_cpp_sub['vuln_label'].values.astype(int)
    logger.info(f"Sampled C++ Validation Set: {len(val_cpp_sub)} samples ({y_val_cpp.mean()*100:.1f}% vuln).")
    
    # Load CodeBERT Tokenizer and Model
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cb_model = AutoModel.from_pretrained(model_name).to(device)
    
    # -------------------------------------------------------------------------
    # 2. Extract Features for C++ Validation Set
    # -------------------------------------------------------------------------
    logger.info("Extracting features for C++ Validation Set...")
    extractor = EnhancedSecurityStructuralFeatureExtractor()
    scaler = StandardScaler()
    
    emb_val_cpp = extract_codebert_512_head_tail(X_val_cpp, tokenizer, cb_model, device)
    struct_val_cpp = extractor.transform(X_val_cpp)
    
    # Load Phase 3 Best Checkpoint as Baseline
    with open(MODELS_DIR / "phase3_codebert_enhanced_best.pkl", "rb") as f:
        p3_ckpt = pickle.load(f)
    p3_clf = p3_ckpt['clf']
    p3_scaler = p3_ckpt['scaler']
    
    struct_val_cpp_scaled = p3_scaler.transform(struct_val_cpp)
    X_val_cpp_fused = np.hstack([emb_val_cpp, struct_val_cpp_scaled])
    
    # Baseline predictions on C++ Validation Set
    prob_val_cpp_p3 = p3_clf.predict_proba(X_val_cpp_fused)[:, 1]
    metrics_val_p3_50 = compute_comprehensive_metrics(y_val_cpp, prob_val_cpp_p3, threshold=0.50)
    logger.info(f"[Baseline Phase 3 on C++ Val] Acc={metrics_val_p3_50['accuracy']*100:.2f}%, AUROC={metrics_val_p3_50['auroc']:.4f}, Prec={metrics_val_p3_50['precision']*100:.2f}%, Rec={metrics_val_p3_50['recall']*100:.2f}%, BalAcc={metrics_val_p3_50['balanced_accuracy']*100:.2f}%")
    
    # -------------------------------------------------------------------------
    # 3. Experiment A: Threshold & Calibration Analysis (Validation Only)
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT A: THRESHOLD & CALIBRATION ANALYSIS (VALIDATION ONLY)")
    logger.info("="*70)
    
    # Sweep thresholds on validation
    threshold_sweep_results = []
    for t in np.arange(0.10, 0.95, 0.05):
        m = compute_comprehensive_metrics(y_val_cpp, prob_val_cpp_p3, threshold=t)
        threshold_sweep_results.append(m)
        logger.info(f"  T={t:.2f} | Acc={m['accuracy']*100:.1f}% | BalAcc={m['balanced_accuracy']*100:.1f}% | Prec={m['precision']*100:.1f}% | Rec={m['recall']*100:.1f}% | FPR={m['fpr']*100:.1f}% | Flagged={m['flagged_rate']*100:.1f}%")
        
    # Calibration evaluation: Isotonic Regression on validation
    iso_calibrator = IsotonicRegression(out_of_bounds='clip')
    iso_calibrator.fit(prob_val_cpp_p3, y_val_cpp)
    prob_val_cpp_cal = iso_calibrator.predict(prob_val_cpp_p3)
    
    brier_uncal = float(brier_score_loss(y_val_cpp, prob_val_cpp_p3))
    brier_cal = float(brier_score_loss(y_val_cpp, prob_val_cpp_cal))
    logger.info(f"Validation Calibration: Brier Uncalibrated={brier_uncal:.4f} -> Calibrated={brier_cal:.4f} (Reliability Improvement: {(brier_uncal-brier_cal)/brier_uncal*100:.1f}%)")
    
    # Select Operating Threshold based on Security-Review Objective:
    # Objective: Flag at least 70% of true vulnerabilities while keeping review workload <= 40% of codebase.
    target_candidates = [m for m in threshold_sweep_results if m['recall'] >= 0.70 and m['flagged_rate'] <= 0.40]
    if target_candidates:
        selected_thresh_m = max(target_candidates, key=lambda x: x['balanced_accuracy'])
    else:
        # Fallback to maximizing balanced accuracy
        selected_thresh_m = max(threshold_sweep_results, key=lambda x: x['balanced_accuracy'])
        
    selected_threshold = selected_thresh_m['threshold']
    logger.info(f"\nSelected Operating Threshold on Validation: T* = {selected_threshold:.2f}")
    logger.info(f"Validation Metrics at T*={selected_threshold:.2f}: Acc={selected_thresh_m['accuracy']*100:.2f}%, Recall={selected_thresh_m['recall']*100:.2f}%, Precision={selected_thresh_m['precision']*100:.2f}%, FPR={selected_thresh_m['fpr']*100:.2f}%, Flagged Rate={selected_thresh_m['flagged_rate']*100:.2f}%")
    
    # -------------------------------------------------------------------------
    # 4. Experiment B: C++-Aware Vulnerability Training
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT B: C++-AWARE VULNERABILITY TRAINING")
    logger.info("="*70)
    
    # Identify C++ and C training pools
    train_cpp_mask = is_cpp_function(train_df['code'])
    train_cpp_pool = train_df[train_cpp_mask].copy().reset_index(drop=True)
    train_c_pool = train_df[~train_cpp_mask].copy().reset_index(drop=True)
    
    # Prevent fresh test project leakage
    fresh_projects = set(fresh_test_df['project'].unique())
    train_cpp_pool = train_cpp_pool[~train_cpp_pool['project'].isin(fresh_projects)].reset_index(drop=True)
    train_c_pool = train_c_pool[~train_c_pool['project'].isin(fresh_projects)].reset_index(drop=True)
    
    logger.info(f"Available C++ training samples: {len(train_cpp_pool)} across {train_cpp_pool['project'].nunique()} repositories.")
    logger.info(f"Available C training samples:   {len(train_c_pool)} across {train_c_pool['project'].nunique()} repositories.")
    
    # Stratified C++ Sampling (6,000 samples: 3,000 vuln + 3,000 benign)
    # Give priority to logic and resource exhaustion CWEs (CWE-400, CWE-22, CWE-476, CWE-416, CWE-770)
    cpp_vuln = train_cpp_pool[train_cpp_pool['vuln_label'] == 1]
    cpp_benign = train_cpp_pool[train_cpp_pool['vuln_label'] == 0]
    
    rare_cwes = ['CWE-400', 'CWE-22', 'CWE-476', 'CWE-416', 'CWE-770', 'CWE-835']
    rare_cpp_vuln = cpp_vuln[cpp_vuln['cwe'].isin(rare_cwes)]
    other_cpp_vuln = cpp_vuln[~cpp_vuln['cwe'].isin(rare_cwes)]
    
    n_rare = len(rare_cpp_vuln)
    n_other = min(3000 - n_rare, len(other_cpp_vuln))
    
    sampled_cpp_vuln = pd.concat([rare_cpp_vuln, other_cpp_vuln.sample(n=n_other, random_state=42)], ignore_index=True)
    sampled_cpp_benign = cpp_benign.sample(n=len(sampled_cpp_vuln), random_state=42)
    
    # Sample balanced C functions (6,000 samples: 3,000 vuln + 3,000 benign)
    c_vuln = train_c_pool[train_c_pool['vuln_label'] == 1].sample(n=3000, random_state=42)
    c_benign = train_c_pool[train_c_pool['vuln_label'] == 0].sample(n=3000, random_state=42)
    
    train_cpp_adapted = pd.concat([
        sampled_cpp_vuln, sampled_cpp_benign, c_vuln, c_benign
    ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    logger.info(f"Constructed Balanced C/C++ Training Corpus: {len(train_cpp_adapted)} samples across {train_cpp_adapted['project'].nunique()} repositories.")
    logger.info(f"  C++ Samples: {is_cpp_function(train_cpp_adapted['code']).sum()} ({(is_cpp_function(train_cpp_adapted['code']).mean()*100):.1f}%)")
    logger.info(f"  Vulnerable:  {(train_cpp_adapted['vuln_label'] == 1).sum()} (50.0%)")
    
    # Feature extraction for C++ adapted training set
    logger.info("Extracting features for C++ adapted training set...")
    t0 = time.time()
    emb_train_adapted = extract_codebert_512_head_tail(train_cpp_adapted['code'].values, tokenizer, cb_model, device)
    scaler_adapted = StandardScaler()
    struct_train_adapted = scaler_adapted.fit_transform(extractor.transform(train_cpp_adapted['code'].values))
    X_train_adapted = np.hstack([emb_train_adapted, struct_train_adapted])
    y_train_adapted = train_cpp_adapted['vuln_label'].values.astype(int)
    logger.info(f"Feature extraction completed in {time.time()-t0:.1f}s. Feature dim: {X_train_adapted.shape[1]}")
    
    # Train C++ Adapted Hybrid Classifier
    clf_adapted = LogisticRegression(C=0.05, class_weight='balanced', max_iter=1000, solver='lbfgs', random_state=42)
    t0 = time.time()
    clf_adapted.fit(X_train_adapted, y_train_adapted)
    logger.info(f"C++ Adapted Classifier fit completed in {time.time()-t0:.2f}s")
    
    # Evaluate C++ Adapted Model on C++ Validation Set
    struct_val_cpp_adapted_scaled = scaler_adapted.transform(struct_val_cpp)
    X_val_cpp_adapted_fused = np.hstack([emb_val_cpp, struct_val_cpp_adapted_scaled])
    prob_val_cpp_adapted = clf_adapted.predict_proba(X_val_cpp_adapted_fused)[:, 1]
    
    metrics_val_adapted_50 = compute_comprehensive_metrics(y_val_cpp, prob_val_cpp_adapted, threshold=0.50)
    logger.info(f"[C++ Adapted Model on C++ Val (T=0.50)] Acc={metrics_val_adapted_50['accuracy']*100:.2f}%, AUROC={metrics_val_adapted_50['auroc']:.4f}, Prec={metrics_val_adapted_50['precision']*100:.2f}%, Rec={metrics_val_adapted_50['recall']*100:.2f}%, BalAcc={metrics_val_adapted_50['balanced_accuracy']*100:.2f}%")
    
    # Sweep thresholds for adapted model on validation
    adapted_thresh_sweep = []
    for t in np.arange(0.10, 0.95, 0.05):
        m = compute_comprehensive_metrics(y_val_cpp, prob_val_cpp_adapted, threshold=t)
        adapted_thresh_sweep.append(m)
        
    adapted_target_candidates = [m for m in adapted_thresh_sweep if m['recall'] >= 0.70 and m['flagged_rate'] <= 0.40]
    if adapted_target_candidates:
        selected_adapted_thresh_m = max(adapted_target_candidates, key=lambda x: x['balanced_accuracy'])
    else:
        selected_adapted_thresh_m = max(adapted_thresh_sweep, key=lambda x: x['balanced_accuracy'])
        
    selected_adapted_threshold = selected_adapted_thresh_m['threshold']
    logger.info(f"[C++ Adapted Model on C++ Val (T*={selected_adapted_threshold:.2f})] Acc={selected_adapted_thresh_m['accuracy']*100:.2f}%, Recall={selected_adapted_thresh_m['recall']*100:.2f}%, Precision={selected_adapted_thresh_m['precision']*100:.2f}%, FPR={selected_adapted_thresh_m['fpr']*100:.2f}%")
    
    # -------------------------------------------------------------------------
    # 5. Experiment C: Patch-Aware Commit Ranking & Classification
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT C: PATCH-AWARE MODELING EVALUATION")
    logger.info("="*70)
    with open(RESULTS_DIR / "diff_aware_experiment_results.json") as f:
        diff_results = json.load(f)
    diff_whole = diff_results['results']['whole_function_baseline']
    diff_patch = diff_results['results']['changed_line_diff_aware']
    logger.info(f"Whole-Function Ranking Acc: {diff_whole['pairwise_ranking_accuracy']*100:.2f}%, Inverted Error: {diff_whole['inverted_error_rate']*100:.2f}%")
    logger.info(f"Changed-Line Patch Ranking Acc: {diff_patch['pairwise_ranking_accuracy']*100:.2f}%, Inverted Error: {diff_patch['inverted_error_rate']*100:.2f}%")
    logger.info(f"Gain from Diff-Aware Modeling: Ranking Acc +{diff_results['results']['relative_gains']['ranking_acc_delta']*100:.2f}%, Inverted Error -{diff_results['results']['relative_gains']['inverted_error_reduction']*100:.2f}%")
    
    # -------------------------------------------------------------------------
    # 6. Model Selection on Validation Data
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*70)
    logger.info("VALIDATION-BASED MODEL SELECTION")
    logger.info("="*70)
    logger.info(f"Candidate 1 (Baseline P3 CodeBERT):  Val AUROC={metrics_val_p3_50['auroc']:.4f}, Val PR-AUC={metrics_val_p3_50['pr_auc']:.4f}, Rec@30%={metrics_val_p3_50['recall_at_30_budget']*100:.1f}%, Rec@40%={metrics_val_p3_50['recall_at_40_budget']*100:.1f}%")
    logger.info(f"Candidate 2 (C++ Adapted Hybrid):     Val AUROC={metrics_val_adapted_50['auroc']:.4f}, Val PR-AUC={metrics_val_adapted_50['pr_auc']:.4f}, Rec@30%={metrics_val_adapted_50['recall_at_30_budget']*100:.1f}%, Rec@40%={metrics_val_adapted_50['recall_at_40_budget']*100:.1f}%")
    
    # Selection criteria: Higher AUROC and better recall at review budget
    if metrics_val_adapted_50['auroc'] >= metrics_val_p3_50['auroc']:
        best_candidate_name = "C++ Adapted Hybrid"
        winning_clf = clf_adapted
        winning_scaler = scaler_adapted
        winning_threshold = selected_adapted_threshold
        winning_val_metrics = metrics_val_adapted_50
    else:
        best_candidate_name = "Phase 3 Enhanced CodeBERT"
        winning_clf = p3_clf
        winning_scaler = p3_scaler
        winning_threshold = selected_threshold
        winning_val_metrics = metrics_val_p3_50
        
    logger.info(f"Selected Best Candidate on Validation: {best_candidate_name} with Operating Threshold T* = {winning_threshold:.2f}")
    
    # -------------------------------------------------------------------------
    # 7. Final Single Evaluation on Locked Benchmarks (Clean 4-Projects & Envoy)
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*70)
    logger.info("EVALUATION ON LOCKED BENCHMARKS (CLEAN 4-PROJECTS & ENVOY)")
    logger.info("="*70)
    
    # Extract features for Benchmark 1 (Clean 4-Project Benchmark, N = 3,169)
    logger.info("Extracting features for Clean 4-Project Benchmark...")
    emb_b1 = extract_codebert_512_head_tail(clean_test_df['code'].values, tokenizer, cb_model, device)
    struct_b1 = winning_scaler.transform(extractor.transform(clean_test_df['code'].values))
    X_b1 = np.hstack([emb_b1, struct_b1])
    y_b1 = clean_test_df['vuln_label'].values.astype(int)
    
    prob_b1 = winning_clf.predict_proba(X_b1)[:, 1]
    
    # Benchmark 1 Overall at Default T=0.50
    m_b1_t50 = compute_comprehensive_metrics(y_b1, prob_b1, threshold=0.50)
    # Benchmark 1 Overall at Selected T*
    m_b1_topt = compute_comprehensive_metrics(y_b1, prob_b1, threshold=winning_threshold)
    
    logger.info(f"[Benchmark 1 Overall T=0.50] Acc={m_b1_t50['accuracy']*100:.2f}%, AUROC={m_b1_t50['auroc']:.4f}, Prec={m_b1_t50['precision']*100:.2f}%, Rec={m_b1_t50['recall']*100:.2f}%, F1={m_b1_t50['f1']*100:.2f}%")
    logger.info(f"[Benchmark 1 Overall T*={winning_threshold:.2f}] Acc={m_b1_topt['accuracy']*100:.2f}%, AUROC={m_b1_topt['auroc']:.4f}, Prec={m_b1_topt['precision']*100:.2f}%, Rec={m_b1_topt['recall']*100:.2f}%, F1={m_b1_topt['f1']*100:.2f}%, Flagged={m_b1_topt['flagged_rate']*100:.2f}%")
    
    # Per-repository breakdown on Benchmark 1
    b1_by_project = {}
    for proj in ['gpac', 'vim', 'openssl', 'envoy']:
        mask_p = (clean_test_df['project'] == proj).values
        y_p = y_b1[mask_p]
        p_p = prob_b1[mask_p]
        
        m_p_50 = compute_comprehensive_metrics(y_p, p_p, threshold=0.50)
        m_p_opt = compute_comprehensive_metrics(y_p, p_p, threshold=winning_threshold)
        
        b1_by_project[proj] = {
            "default_t50": m_p_50,
            "selected_topt": m_p_opt
        }
        logger.info(f"  -> {proj:8} | T=0.50: Acc={m_p_50['accuracy']*100:.2f}%, Rec={m_p_50['recall']*100:.2f}%, Prec={m_p_50['precision']*100:.2f}%, AUROC={m_p_50['auroc']:.4f}")
        logger.info(f"             | T*={winning_threshold:.2f}: Acc={m_p_opt['accuracy']*100:.2f}%, Rec={m_p_opt['recall']*100:.2f}%, Prec={m_p_opt['precision']*100:.2f}%, FPR={m_p_opt['fpr']*100:.2f}%, Flagged={m_p_opt['flagged_rate']*100:.2f}%")
        
    # Evaluate on Fresh 5-Project Benchmark (N = 3,801)
    logger.info("\nExtracting features for Fresh 5-Project Benchmark...")
    emb_b2 = extract_codebert_512_head_tail(fresh_test_df['code'].values, tokenizer, cb_model, device)
    struct_b2 = winning_scaler.transform(extractor.transform(fresh_test_df['code'].values))
    X_b2 = np.hstack([emb_b2, struct_b2])
    y_b2 = fresh_test_df['vuln_label'].values.astype(int)
    prob_b2 = winning_clf.predict_proba(X_b2)[:, 1]
    
    m_b2_t50 = compute_comprehensive_metrics(y_b2, prob_b2, threshold=0.50)
    m_b2_topt = compute_comprehensive_metrics(y_b2, prob_b2, threshold=winning_threshold)
    logger.info(f"[Benchmark 2 Fresh Overall T=0.50] Acc={m_b2_t50['accuracy']*100:.2f}%, AUROC={m_b2_t50['auroc']:.4f}, Prec={m_b2_t50['precision']*100:.2f}%, Rec={m_b2_t50['recall']*100:.2f}%")
    logger.info(f"[Benchmark 2 Fresh Overall T*={winning_threshold:.2f}] Acc={m_b2_topt['accuracy']*100:.2f}%, Rec={m_b2_topt['recall']*100:.2f}%, Prec={m_b2_topt['precision']*100:.2f}%")
    
    # Save Model Checkpoint
    checkpoint_path = MODELS_DIR / "phase4_envoy_adapted_best.pkl"
    with open(checkpoint_path, "wb") as f:
        pickle.dump({
            "model_name": best_candidate_name,
            "clf": winning_clf,
            "scaler": winning_scaler,
            "selected_threshold": winning_threshold,
            "val_metrics": winning_val_metrics,
            "benchmark1_results": m_b1_t50,
            "benchmark1_results_topt": m_b1_topt,
            "benchmark2_results": m_b2_t50
        }, f)
    logger.info(f"Saved Phase 4 winning model checkpoint to {checkpoint_path}")
    
    # Save machine-readable results
    output_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_model_name": best_candidate_name,
        "selected_threshold": winning_threshold,
        "experiment_a_threshold_and_calibration": {
            "validation_cpp_n": len(val_cpp_sub),
            "uncalibrated_brier": brier_uncal,
            "calibrated_brier": brier_cal,
            "threshold_sweep": threshold_sweep_results,
            "selected_operating_point": selected_thresh_m
        },
        "experiment_b_cpp_adaptation": {
            "training_samples": len(train_cpp_adapted),
            "cpp_samples_in_train": int(is_cpp_function(train_cpp_adapted['code']).sum()),
            "c_samples_in_train": int((~is_cpp_function(train_cpp_adapted['code'])).sum()),
            "validation_cpp_metrics_t50": metrics_val_adapted_50,
            "validation_cpp_metrics_topt": selected_adapted_thresh_m
        },
        "experiment_c_patch_aware": diff_results,
        "locked_evaluations": {
            "benchmark1_clean_4projects": {
                "overall_t50": m_b1_t50,
                "overall_topt": m_b1_topt,
                "by_project": b1_by_project
            },
            "benchmark2_fresh_5projects": {
                "overall_t50": m_b2_t50,
                "overall_topt": m_b2_topt
            }
        }
    }
    
    output_json_path = RESULTS_DIR / "phase4_envoy_results.json"
    with open(output_json_path, "w") as f:
        json.dump(output_results, f, indent=2)
    logger.info(f"Saved machine-readable results to {output_json_path}")
    
    # Save updated Envoy test predictions with winning model
    envoy_mask_clean = (clean_test_df['project'] == 'envoy').values
    envoy_prob_winning = prob_b1[envoy_mask_clean]
    envoy_test_df['prob_phase4_adapted'] = envoy_prob_winning
    envoy_test_df['pred_phase4_adapted_t50'] = (envoy_prob_winning >= 0.50).astype(int)
    envoy_test_df['pred_phase4_adapted_topt'] = (envoy_prob_winning >= winning_threshold).astype(int)
    envoy_test_df.to_parquet(RESULTS_DIR / "envoy_phase4_predictions.parquet", index=False)
    logger.info(f"Saved updated Envoy predictions to {RESULTS_DIR / 'envoy_phase4_predictions.parquet'}")

if __name__ == "__main__":
    main()
