#!/usr/bin/env python3
"""
Evaluate Improved CodeBERT on the Untouched Held-Out Test Set (N=5,412)
======================================================================
Model: CodeBERT with max_length=384, [CLS + MeanPool] Token Pooling, and Security Structural Fusion.
Trained on: 8,000 balanced training samples
Evaluated on: Full held-out test split (5,412 samples)
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.search_classical_models import SecurityStructuralFeatureExtractor
from scripts.improve_codebert_experiments import extract_embeddings_cls_and_mean, get_device
from src.evaluation.evaluate import compute_classification_metrics, save_metrics, generate_plots, compare_models
from scripts.train_improved_models import bootstrap_ci

def main():
    print("=" * 80)
    print("EVALUATING IMPROVED CODEBERT (len=384, [CLS+MeanPool]+SecurityStruct) ON TEST SET")
    print("=" * 80)
    
    device = get_device()
    print(f"Device: {device}")
    
    train_df = pd.read_parquet(PROJECT_ROOT / "data" / "splits" / "vulnerability_train.parquet")
    val_df = pd.read_parquet(PROJECT_ROOT / "data" / "splits" / "vulnerability_val.parquet")
    test_df = pd.read_parquet(PROJECT_ROOT / "data" / "splits" / "vulnerability_test.parquet")
    
    SEED = 42
    N_TRAIN = 8000
    sub_train = train_df.sample(n=N_TRAIN, random_state=SEED).reset_index(drop=True)
    
    X_train_text = sub_train['code'].tolist()
    y_train = sub_train['vuln_label'].values
    
    X_val_text = val_df['code'].tolist()
    y_val = val_df['vuln_label'].values
    
    X_test_text = test_df['code'].tolist()
    y_test = test_df['vuln_label'].values
    
    print(f"Train samples: {len(X_train_text)}")
    print(f"Val samples:   {len(X_val_text)}")
    print(f"Test samples:  {len(X_test_text)}")
    
    model_name = 'microsoft/codebert-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    encoder.eval()
    
    print("\nExtracting [CLS + MeanPool] embeddings (max_length=384)...")
    emb_tr = extract_embeddings_cls_and_mean(X_train_text, tokenizer, encoder, device, batch_size=32, max_length=384, use_mean_pool=True)
    emb_va = extract_embeddings_cls_and_mean(X_val_text, tokenizer, encoder, device, batch_size=32, max_length=384, use_mean_pool=True)
    emb_te = extract_embeddings_cls_and_mean(X_test_text, tokenizer, encoder, device, batch_size=32, max_length=384, use_mean_pool=True)
    
    # Extract structural features
    print("Extracting security structural features...")
    struct_ext = SecurityStructuralFeatureExtractor()
    scaler = StandardScaler()
    struct_tr = scaler.fit_transform(struct_ext.transform(X_train_text))
    struct_va = scaler.transform(struct_ext.transform(X_val_text))
    struct_te = scaler.transform(struct_ext.transform(X_test_text))
    
    X_tr_fused = np.hstack([emb_tr, struct_tr])
    X_va_fused = np.hstack([emb_va, struct_va])
    X_te_fused = np.hstack([emb_te, struct_te])
    
    print(f"Fused representations shape: {X_tr_fused.shape}")
    
    # Train classifier head with C=0.01 (selected on validation set)
    clf = LogisticRegression(C=0.01, class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs')
    clf.fit(X_tr_fused, y_train)
    
    # Validation threshold tuning
    val_probs = clf.predict_proba(X_va_fused)[:, 1]
    best_t = 0.50
    from sklearn.metrics import matthews_corrcoef
    best_mcc = matthews_corrcoef(y_val, (val_probs >= 0.5).astype(int))
    for t in np.arange(0.30, 0.70, 0.02):
        t = round(float(t), 2)
        m = matthews_corrcoef(y_val, (val_probs >= t).astype(int))
        if m > best_mcc:
            best_mcc = m
            best_t = t
    print(f"Validation threshold selected: t*={best_t:.2f} (Val MCC: {best_mcc:.4f})")
    
    # Evaluate strictly on test set
    test_probs = clf.predict_proba(X_te_fused)[:, 1]
    test_preds = (test_probs >= best_t).astype(int)
    
    metrics = compute_classification_metrics(y_test, test_preds, np.column_stack([1 - test_probs, test_probs]))
    metrics['brier_score'] = float(np.mean((test_probs - y_test) ** 2))
    metrics['positive_prediction_rate'] = float(np.mean(test_preds))
    metrics['decision_threshold'] = float(best_t)
    
    # Bootstrap CI
    ci_dict = bootstrap_ci(y_test, test_preds, test_probs, n_bootstraps=1000)
    metrics['bootstrap_95ci'] = ci_dict
    
    print("\n--- FINAL TEST METRICS (IMPROVED CODEBERT HYBRID) ---")
    print(f"Accuracy:                 {metrics['accuracy']:.4f} (95% CI: [{ci_dict['accuracy_95ci'][0]:.4f}, {ci_dict['accuracy_95ci'][1]:.4f}])")
    print(f"Precision:                {metrics['precision']:.4f}")
    print(f"Recall:                   {metrics['recall']:.4f}")
    print(f"F1 Score:                 {metrics['f1']:.4f} (95% CI: [{ci_dict['f1_95ci'][0]:.4f}, {ci_dict['f1_95ci'][1]:.4f}])")
    print(f"Macro F1:                 {metrics['f1_macro']:.4f}")
    print(f"MCC:                      {metrics['mcc']:+.4f} (95% CI: [{ci_dict['mcc_95ci'][0]:+.4f}, {ci_dict['mcc_95ci'][1]:+.4f}])")
    print(f"AUROC:                    {metrics['auroc']:.4f} (95% CI: [{ci_dict['auroc_95ci'][0]:.4f}, {ci_dict['auroc_95ci'][1]:.4f}])")
    print(f"Average Precision (PR):   {metrics['average_precision']:.4f}")
    print(f"Brier Score:              {metrics['brier_score']:.4f}")
    print(f"Positive Prediction Rate: {metrics['positive_prediction_rate']:.4f}")
    print(f"Confusion Matrix:         {metrics['confusion_matrix']}")
    
    # Save checkpoint
    CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints"
    with open(CHECKPOINTS_DIR / "codebert_improved_hybrid.pkl", "wb") as f:
        pickle.dump({
            'clf': clf,
            'scaler': scaler,
            'threshold': best_t,
            'max_length': 384,
            'use_mean_pool': True,
            'model_name': 'codebert_improved_hybrid'
        }, f)
        
    RESULTS_DIR = PROJECT_ROOT / "results"
    save_metrics(metrics, RESULTS_DIR / "metrics" / "codebert_improved_hybrid_metrics.json", "codebert_improved_hybrid")
    generate_plots(metrics, y_test, test_preds, np.column_stack([1 - test_probs, test_probs]),
                   RESULTS_DIR / "plots", "codebert_improved_hybrid")
                   
    # Re-run overall comparison
    all_models = {}
    for mf in sorted((RESULTS_DIR / "metrics").glob("*.json")):
        with open(mf) as f:
            data = json.load(f)
            m_name = data.get('model_name', mf.stem)
            m_metrics = data.get('metrics', data)
            all_models[m_name] = m_metrics
    if all_models:
        compare_models(all_models, RESULTS_DIR / "plots")
        
    print("\nSaved improved CodeBERT artifacts and updated comparison plots!")

if __name__ == "__main__":
    main()
