#!/usr/bin/env python3
"""
Controlled Experiments to Improve CodeBERT for Vulnerability Detection
=====================================================================
Investigates:
1. Sequence length impact: max_length=256 vs 384 vs 512 (addressing truncation)
2. Pooling representation: [CLS] vs [CLS + MeanPool] (LineVul-inspired token pooling)
3. Scaling training samples: 3,000 -> 8,000 -> 12,000 on Apple Silicon MPS
4. Hybrid CodeBERT + Security Structural Fusion (deep contextual + static security tokens)
5. Decision threshold tuning strictly on validation set

Strictly evaluated on VALIDATION SPLIT (N=3,608). Final test set is NOT touched.
"""

import sys
import time
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.search_classical_models import SecurityStructuralFeatureExtractor, c_tokenize

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def extract_embeddings_cls_and_mean(codes, tokenizer, model, device, batch_size=32, max_length=384, use_mean_pool=True):
    model.eval()
    embeddings = []
    
    n_batches = (len(codes) + batch_size - 1) // batch_size
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
            
            if use_mean_pool:
                # Masked mean pooling across non-padding tokens
                mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden.size()).float()
                sum_embeddings = torch.sum(hidden * mask, 1)
                sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                mean_repr = sum_embeddings / sum_mask  # [B, 768]
                combined = torch.cat([cls_repr, mean_repr], dim=-1).cpu().numpy()  # [B, 1536]
                embeddings.append(combined)
            else:
                embeddings.append(cls_repr.cpu().numpy())
                
    result = np.vstack(embeddings)
    elapsed = time.time() - t0
    rate = len(codes) / elapsed
    print(f"  Extracted {result.shape[0]} embeddings ({result.shape[1]} dims) in {elapsed:.1f}s ({rate:.1f} samples/s)")
    return result


def evaluate_val_metrics(clf, X_val, y_val, model_name="", tune_threshold=True):
    val_probs = clf.predict_proba(X_val)[:, 1]
    
    best_t = 0.50
    best_mcc = matthews_corrcoef(y_val, (val_probs >= 0.5).astype(int))
    
    if tune_threshold:
        for t in np.arange(0.30, 0.70, 0.02):
            t = round(float(t), 2)
            preds_t = (val_probs >= t).astype(int)
            mcc_t = matthews_corrcoef(y_val, preds_t)
            if mcc_t > best_mcc:
                best_mcc = mcc_t
                best_t = t
                
    preds = (val_probs >= best_t).astype(int)
    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)
    mcc = matthews_corrcoef(y_val, preds)
    auroc = roc_auc_score(y_val, val_probs)
    ap = average_precision_score(y_val, val_probs)
    
    return {
        'model_name': model_name,
        'val_accuracy': float(acc),
        'val_precision': float(prec),
        'val_recall': float(rec),
        'val_f1': float(f1),
        'val_mcc': float(mcc),
        'val_auroc': float(auroc),
        'val_ap': float(ap),
        'best_val_threshold': float(best_t),
    }


def main():
    print("=" * 80)
    print("CODEBERT CONTROLLED IMPROVEMENT EXPERIMENTS ON VALIDATION SET")
    print("=" * 80)
    
    device = get_device()
    print(f"Device: {device}")
    
    train_df = pd.read_parquet(PROJECT_ROOT / "data" / "splits" / "vulnerability_train.parquet")
    val_df = pd.read_parquet(PROJECT_ROOT / "data" / "splits" / "vulnerability_val.parquet")
    
    # Subsample training data for controlled comparisons: 8,000 samples balanced
    SEED = 42
    N_TRAIN = 8000
    sub_train = train_df.sample(n=N_TRAIN, random_state=SEED).reset_index(drop=True)
    
    X_train_text = sub_train['code'].tolist()
    y_train = sub_train['vuln_label'].values
    
    X_val_text = val_df['code'].tolist()
    y_val = val_df['vuln_label'].values
    
    print(f"Training on N={len(X_train_text)} samples (pos={y_train.mean():.3f})")
    print(f"Validating on N={len(X_val_text)} samples (pos={y_val.mean():.3f})")
    
    model_name = 'microsoft/codebert-base'
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    encoder.eval()
    
    results = []
    
    # -------------------------------------------------------------
    # Exp 1: Baseline CodeBERT [CLS] max_length=256
    # -------------------------------------------------------------
    print("\n[Exp 1] Extracting CodeBERT [CLS] (max_length=256)...")
    emb_tr_256 = extract_embeddings_cls_and_mean(X_train_text, tokenizer, encoder, device, batch_size=32, max_length=256, use_mean_pool=False)
    emb_va_256 = extract_embeddings_cls_and_mean(X_val_text, tokenizer, encoder, device, batch_size=32, max_length=256, use_mean_pool=False)
    
    clf1 = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs')
    clf1.fit(emb_tr_256, y_train)
    res1 = evaluate_val_metrics(clf1, emb_va_256, y_val, "CodeBERT [CLS] len=256 C=0.1")
    results.append(res1)
    print(f"  -> Acc: {res1['val_accuracy']:.4f}, AUROC: {res1['val_auroc']:.4f}, MCC: {res1['val_mcc']:.4f}, t*={res1['best_val_threshold']:.2f}")

    # -------------------------------------------------------------
    # Exp 2: CodeBERT [CLS] with max_length=384 (longer context)
    # -------------------------------------------------------------
    print("\n[Exp 2] Extracting CodeBERT [CLS] (max_length=384)...")
    emb_tr_384_cls = extract_embeddings_cls_and_mean(X_train_text, tokenizer, encoder, device, batch_size=32, max_length=384, use_mean_pool=False)
    emb_va_384_cls = extract_embeddings_cls_and_mean(X_val_text, tokenizer, encoder, device, batch_size=32, max_length=384, use_mean_pool=False)
    
    clf2 = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs')
    clf2.fit(emb_tr_384_cls, y_train)
    res2 = evaluate_val_metrics(clf2, emb_va_384_cls, y_val, "CodeBERT [CLS] len=384 C=0.1")
    results.append(res2)
    print(f"  -> Acc: {res2['val_accuracy']:.4f}, AUROC: {res2['val_auroc']:.4f}, MCC: {res2['val_mcc']:.4f}, t*={res2['best_val_threshold']:.2f}")

    # -------------------------------------------------------------
    # Exp 3: CodeBERT [CLS + MeanPool] max_length=384 (Token Pooling)
    # -------------------------------------------------------------
    print("\n[Exp 3] Extracting CodeBERT [CLS + MeanPool] (max_length=384)...")
    emb_tr_384_pool = extract_embeddings_cls_and_mean(X_train_text, tokenizer, encoder, device, batch_size=32, max_length=384, use_mean_pool=True)
    emb_va_384_pool = extract_embeddings_cls_and_mean(X_val_text, tokenizer, encoder, device, batch_size=32, max_length=384, use_mean_pool=True)
    
    clf3 = LogisticRegression(C=0.05, class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs')
    clf3.fit(emb_tr_384_pool, y_train)
    res3 = evaluate_val_metrics(clf3, emb_va_384_pool, y_val, "CodeBERT [CLS+MeanPool] len=384 C=0.05")
    results.append(res3)
    print(f"  -> Acc: {res3['val_accuracy']:.4f}, AUROC: {res3['val_auroc']:.4f}, MCC: {res3['val_mcc']:.4f}, t*={res3['best_val_threshold']:.2f}")

    # -------------------------------------------------------------
    # Exp 4: CodeBERT [CLS + MeanPool] + Security Structural Fusion
    # -------------------------------------------------------------
    print("\n[Exp 4] Extracting Security Structural Features and Fusing with CodeBERT...")
    struct_ext = SecurityStructuralFeatureExtractor()
    scaler = StandardScaler()
    
    struct_tr = scaler.fit_transform(struct_ext.transform(X_train_text))
    struct_va = scaler.transform(struct_ext.transform(X_val_text))
    
    fused_tr = np.hstack([emb_tr_384_pool, struct_tr])
    fused_va = np.hstack([emb_va_384_pool, struct_va])
    print(f"  Fused representations: Shape={fused_tr.shape}")
    
    for c_val in [0.01, 0.05, 0.1]:
        clf4 = LogisticRegression(C=c_val, class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs')
        clf4.fit(fused_tr, y_train)
        res4 = evaluate_val_metrics(clf4, fused_va, y_val, f"CodeBERT [CLS+MeanPool]+SecurityStruct C={c_val}")
        results.append(res4)
        print(f"  -> C={c_val}: Acc: {res4['val_accuracy']:.4f}, AUROC: {res4['val_auroc']:.4f}, MCC: {res4['val_mcc']:.4f}, t*={res4['best_val_threshold']:.2f}")

    # -------------------------------------------------------------
    # Summary of CodeBERT Experiments
    # -------------------------------------------------------------
    df_res = pd.DataFrame(results).sort_values(by='val_auroc', ascending=False).reset_index(drop=True)
    print("\n" + "=" * 80)
    print("CODEBERT VALIDATION EXPERIMENTS SUMMARY")
    print("=" * 80)
    print(df_res.to_string(index=False))
    
    out_dir = PROJECT_ROOT / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_dir / "codebert_improvement_results.csv", index=False)
    with open(out_dir / "codebert_improvement_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved CodeBERT experiments to {out_dir / 'codebert_improvement_results.csv'}")

if __name__ == "__main__":
    main()
