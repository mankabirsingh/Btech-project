#!/usr/bin/env python3
"""
Generate and Save Test Predictions for All Models (N=5,412)
===========================================================
Extracts/computes predictions from all 5 evaluated models on the identical
held-out test set (data/splits/vulnerability_test.parquet):
1. Char N-gram Baseline (baseline_char.pkl)
2. Word N-gram Baseline (baseline_word.pkl)
3. CodeBERT Original (codebert_classifier.pkl)
4. Tri-Union Best Model (baseline_tri_union_best.pkl)
5. CodeBERT Improved Hybrid (codebert_improved_hybrid.pkl)

Saves consolidated predictions and ground truth to:
results/experiments/test_predictions_all_models.parquet
"""

import sys
import time
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.search_classical_models import c_tokenize, SecurityStructuralFeatureExtractor
from scripts.improve_codebert_experiments import extract_embeddings_cls_and_mean, get_device

def main():
    print("=" * 80)
    print("GENERATING CONSOLIDATED TEST PREDICTIONS (N=5,412)")
    print("=" * 80)
    
    device = get_device()
    print(f"Device: {device}")
    
    test_df = pd.read_parquet(PROJECT_ROOT / "data" / "splits" / "vulnerability_test.parquet")
    X_test_text = test_df['code'].tolist()
    y_test = test_df['vuln_label'].values
    
    results_df = test_df[[
        'sample_id', 'source_dataset', 'vuln_label', 'cwe', 'severity', 
        'project', 'commit_id', 'group_id'
    ]].copy()
    
    # -------------------------------------------------------------
    # 1. Baseline Char N-gram
    # -------------------------------------------------------------
    print("\n[1/5] Evaluating Baseline Char N-gram...")
    with open(PROJECT_ROOT / "models" / "checkpoints" / "baseline_char.pkl", "rb") as f:
        char_data = pickle.load(f)
    pipe_char = char_data['pipeline']
    prob_char = pipe_char.predict_proba(X_test_text)[:, 1]
    pred_char = (prob_char >= 0.50).astype(int)
    results_df['prob_char'] = prob_char
    results_df['pred_char'] = pred_char
    print(f"  Acc: {accuracy_score(y_test, pred_char):.4f}, AUROC: {roc_auc_score(y_test, prob_char):.4f}")

    # -------------------------------------------------------------
    # 2. Baseline Word N-gram
    # -------------------------------------------------------------
    print("\n[2/5] Evaluating Baseline Word N-gram...")
    with open(PROJECT_ROOT / "models" / "checkpoints" / "baseline_word.pkl", "rb") as f:
        word_data = pickle.load(f)
    pipe_word = word_data['pipeline']
    prob_word = pipe_word.predict_proba(X_test_text)[:, 1]
    pred_word = (prob_word >= 0.50).astype(int)
    results_df['prob_word'] = prob_word
    results_df['pred_word'] = pred_word
    print(f"  Acc: {accuracy_score(y_test, pred_word):.4f}, AUROC: {roc_auc_score(y_test, prob_word):.4f}")

    # -------------------------------------------------------------
    # 3. Tri-Union Best Model
    # -------------------------------------------------------------
    print("\n[3/5] Evaluating Tri-Union Best Model...")
    with open(PROJECT_ROOT / "models" / "checkpoints" / "baseline_tri_union_best.pkl", "rb") as f:
        tri_data = pickle.load(f)
    pipe_tri = tri_data['pipeline']
    prob_tri = pipe_tri.predict_proba(X_test_text)[:, 1]
    pred_tri = (prob_tri >= 0.50).astype(int)
    results_df['prob_tri'] = prob_tri
    results_df['pred_tri'] = pred_tri
    print(f"  Acc: {accuracy_score(y_test, pred_tri):.4f}, AUROC: {roc_auc_score(y_test, prob_tri):.4f}")

    # -------------------------------------------------------------
    # 4. CodeBERT Original ([CLS] 256)
    # -------------------------------------------------------------
    print("\n[4/5] Evaluating CodeBERT Original ([CLS] len=256)...")
    model_name = 'microsoft/codebert-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    encoder.eval()
    
    with open(PROJECT_ROOT / "models" / "checkpoints" / "codebert_classifier.pkl", "rb") as f:
        cb_orig_data = pickle.load(f)
    clf_cb_orig = cb_orig_data['clf']
    
    emb_cb_256 = extract_embeddings_cls_and_mean(X_test_text, tokenizer, encoder, device, batch_size=32, max_length=256, use_mean_pool=False)
    prob_cb_orig = clf_cb_orig.predict_proba(emb_cb_256)[:, 1]
    pred_cb_orig = (prob_cb_orig >= 0.50).astype(int)
    results_df['prob_codebert_orig'] = prob_cb_orig
    results_df['pred_codebert_orig'] = pred_cb_orig
    print(f"  Acc: {accuracy_score(y_test, pred_cb_orig):.4f}, AUROC: {roc_auc_score(y_test, prob_cb_orig):.4f}")

    # -------------------------------------------------------------
    # 5. CodeBERT Improved Hybrid (len=384, [CLS+MeanPool]+SecurityStruct)
    # -------------------------------------------------------------
    print("\n[5/5] Evaluating CodeBERT Improved Hybrid...")
    with open(PROJECT_ROOT / "models" / "checkpoints" / "codebert_improved_hybrid.pkl", "rb") as f:
        cb_imp_data = pickle.load(f)
    clf_cb_imp = cb_imp_data['clf']
    scaler = cb_imp_data['scaler']
    t_star = cb_imp_data.get('threshold', 0.52)
    
    emb_cb_384 = extract_embeddings_cls_and_mean(X_test_text, tokenizer, encoder, device, batch_size=32, max_length=384, use_mean_pool=True)
    struct_ext = SecurityStructuralFeatureExtractor()
    struct_te = scaler.transform(struct_ext.transform(X_test_text))
    X_fused_te = np.hstack([emb_cb_384, struct_te])
    
    prob_cb_imp = clf_cb_imp.predict_proba(X_fused_te)[:, 1]
    pred_cb_imp = (prob_cb_imp >= t_star).astype(int)
    results_df['prob_codebert_imp'] = prob_cb_imp
    results_df['pred_codebert_imp'] = pred_cb_imp
    print(f"  Acc: {accuracy_score(y_test, pred_cb_imp):.4f}, AUROC: {roc_auc_score(y_test, prob_cb_imp):.4f}")

    # Save consolidated dataframe
    out_dir = PROJECT_ROOT / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_predictions_all_models.parquet"
    results_df.to_parquet(out_path, index=False)
    print(f"\nSaved consolidated test predictions to {out_path} ({len(results_df)} rows, {len(results_df.columns)} cols)")

if __name__ == "__main__":
    main()
