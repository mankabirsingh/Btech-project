#!/usr/bin/env python3
"""
Systematic Classical Baseline Search for Vulnerability Detection
================================================================
Investigates:
1. Character n-gram ranges (1-4, 2-5, 3-6) and vocabulary sizes (20k, 30k, 50k)
2. C/C++-aware token n-grams (keywords, operators, identifiers, API calls)
3. Combined FeatureUnion (character + C-token + structural security features)
4. Classifiers: Logistic Regression (tuned C, class_weight), Linear SVM (Calibrated), SGDClassifier
Strictly evaluated on VALIDATION SPLIT (N=3,608). Final test set is NOT touched.
"""

import sys
import os
import re
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# C/C++ tokenizer
C_TOKEN_PATTERN = re.compile(
    r'[a-zA-Z_][a-zA-Z0-9_]*'  # Identifiers / Keywords
    r'|->|::|\+\+|--|==|!=|<=|>=|&&|\|\||<<|>>'  # Multi-char operators
    r'|[&*=!~^|/%+\-<>?:]'  # Single-char operators
    r'|[(){}\[\];,]'  # Delimiters
    r'|0x[0-9a-fA-F]+|\d+'  # Numbers
)

def c_tokenize(code_str):
    if not isinstance(code_str, str):
        return []
    return C_TOKEN_PATTERN.findall(code_str)

# Structural security feature extractor
DANGEROUS_C_FUNCS = set([
    'strcpy', 'strncpy', 'strcat', 'strncat', 'sprintf', 'snprintf', 'vsprintf', 'vsnprintf',
    'gets', 'fgets', 'memcpy', 'memmove', 'memset', 'malloc', 'calloc', 'realloc', 'free',
    'alloca', 'system', 'popen', 'execve', 'execl', 'execvp', 'scanf', 'fscanf', 'sscanf',
    'open', 'read', 'write', 'close', 'chown', 'chmod', 'unlink', 'realpath', 'getenv',
    'setuid', 'seteuid', 'setgid', 'setegid'
])

CONTROL_KEYWORDS = set(['if', 'else', 'while', 'for', 'switch', 'case', 'goto', 'return', 'break', 'continue'])

class SecurityStructuralFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        features = []
        for code in X:
            if not isinstance(code, str):
                code = ""
            tokens = c_tokenize(code)
            n_tokens = len(tokens) + 1e-6
            n_chars = len(code) + 1.0
            n_lines = code.count('\n') + 1.0
            
            # Dangerous API count
            n_dang = sum(1 for t in tokens if t in DANGEROUS_C_FUNCS)
            # Control keyword ratio
            n_ctrl = sum(1 for t in tokens if t in CONTROL_KEYWORDS)
            # Pointer dereferences / arrow
            n_ptrs = sum(1 for t in tokens if t in ('->', '*'))
            # Memory calls
            n_mem = sum(1 for t in tokens if t in ('malloc', 'calloc', 'realloc', 'free', 'alloca'))
            # Format string calls
            n_fmt = sum(1 for t in tokens if t in ('printf', 'sprintf', 'snprintf', 'fprintf', 'scanf', 'sscanf'))
            # Buffer copy calls
            n_cpy = sum(1 for t in tokens if t in ('strcpy', 'strncpy', 'strcat', 'strncat', 'memcpy', 'memmove'))
            # Max brace depth (heuristic for nesting complexity)
            depth, max_depth = 0, 0
            for t in tokens:
                if t == '{':
                    depth += 1
                    if depth > max_depth:
                        max_depth = depth
                elif t == '}':
                    depth = max(0, depth - 1)
            # Comment density
            n_comments = len(re.findall(r'(//|/\*|#)', code))
            
            feat_row = [
                np.log1p(n_chars),
                np.log1p(n_lines),
                np.log1p(len(tokens)),
                n_dang / n_tokens,
                n_ctrl / n_tokens,
                n_ptrs / n_tokens,
                n_mem,
                n_fmt,
                n_cpy,
                max_depth,
                n_comments / n_lines,
                n_dang,
            ]
            features.append(feat_row)
        return np.array(features, dtype=np.float32)


def evaluate_on_val(pipe, X_train, y_train, X_val, y_val, model_name=""):
    t0 = time.time()
    pipe.fit(X_train, y_train)
    fit_time = time.time() - t0
    
    val_preds = pipe.predict(X_val)
    if hasattr(pipe, "predict_proba"):
        val_probs = pipe.predict_proba(X_val)[:, 1]
    elif hasattr(pipe, "decision_function"):
        df_vals = pipe.decision_function(X_val)
        val_probs = 1.0 / (1.0 + np.exp(-df_vals))
    else:
        val_probs = val_preds.astype(float)
        
    acc = accuracy_score(y_val, val_preds)
    prec = precision_score(y_val, val_preds, zero_division=0)
    rec = recall_score(y_val, val_preds, zero_division=0)
    f1 = f1_score(y_val, val_preds, zero_division=0)
    mcc = matthews_corrcoef(y_val, val_preds)
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
        'training_time_s': float(fit_time),
    }


def main():
    print("=" * 80)
    print("SYSTEMATIC CLASSICAL MODEL EXPLORATION ON VALIDATION SET")
    print("=" * 80)
    
    train_df = pd.read_parquet(PROJECT_ROOT / "data" / "splits" / "vulnerability_train.parquet")
    val_df = pd.read_parquet(PROJECT_ROOT / "data" / "splits" / "vulnerability_val.parquet")
    
    X_train, y_train = train_df['code'].values, train_df['vuln_label'].values
    X_val, y_val = val_df['code'].values, val_df['vuln_label'].values
    
    print(f"Loaded: Train={len(X_train)} samples, Val={len(X_val)} samples")
    
    results = []
    
    # -------------------------------------------------------------
    # 1. Baseline: Char N-gram (1, 4) C=1.0 (Existing)
    # -------------------------------------------------------------
    print("\n[1/10] Evaluating Baseline: Char (1, 4) max_features=30k, C=1.0...")
    pipe_base_char = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=30000, ngram_range=(1, 4), analyzer='char_wb', sublinear_tf=True)),
        ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs'))
    ])
    res = evaluate_on_val(pipe_base_char, X_train, y_train, X_val, y_val, "Baseline Char (1-4) C=1.0")
    results.append(res)
    print(f"  -> Acc: {res['val_accuracy']:.4f}, AUROC: {res['val_auroc']:.4f}, MCC: {res['val_mcc']:.4f}, Time: {res['training_time_s']:.1f}s")

    # -------------------------------------------------------------
    # 2. Baseline: Word N-gram (1, 2) C=1.0 (Existing)
    # -------------------------------------------------------------
    print("\n[2/10] Evaluating Baseline: Word (1, 2) max_features=20k, C=1.0...")
    pipe_base_word = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2), analyzer='word', sublinear_tf=True)),
        ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs'))
    ])
    res = evaluate_on_val(pipe_base_word, X_train, y_train, X_val, y_val, "Baseline Word (1-2) C=1.0")
    results.append(res)
    print(f"  -> Acc: {res['val_accuracy']:.4f}, AUROC: {res['val_auroc']:.4f}, MCC: {res['val_mcc']:.4f}, Time: {res['training_time_s']:.1f}s")

    # -------------------------------------------------------------
    # 3. Char N-gram tuned: (2, 5) max_features=40k, C=0.5
    # -------------------------------------------------------------
    print("\n[3/10] Evaluating Char (2, 5) max_features=40k, C=0.5...")
    pipe_char_25 = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=40000, ngram_range=(2, 5), analyzer='char_wb', sublinear_tf=True)),
        ('clf', LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs'))
    ])
    res = evaluate_on_val(pipe_char_25, X_train, y_train, X_val, y_val, "Char (2-5) 40k C=0.5")
    results.append(res)
    print(f"  -> Acc: {res['val_accuracy']:.4f}, AUROC: {res['val_auroc']:.4f}, MCC: {res['val_mcc']:.4f}, Time: {res['training_time_s']:.1f}s")

    # -------------------------------------------------------------
    # 4. Char N-gram tuned: (1, 5) max_features=50k, C=0.2
    # -------------------------------------------------------------
    print("\n[4/10] Evaluating Char (1, 5) max_features=50k, C=0.2...")
    pipe_char_15 = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 5), analyzer='char_wb', sublinear_tf=True)),
        ('clf', LogisticRegression(C=0.2, class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs'))
    ])
    res = evaluate_on_val(pipe_char_15, X_train, y_train, X_val, y_val, "Char (1-5) 50k C=0.2")
    results.append(res)
    print(f"  -> Acc: {res['val_accuracy']:.4f}, AUROC: {res['val_auroc']:.4f}, MCC: {res['val_mcc']:.4f}, Time: {res['training_time_s']:.1f}s")

    # -------------------------------------------------------------
    # 5. C-Token N-grams (1, 2) max_features=30k, C=0.5
    # -------------------------------------------------------------
    print("\n[5/10] Evaluating C-Token N-grams (1, 2) max_features=30k, C=0.5...")
    pipe_ctoken = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=c_tokenize, token_pattern=None, max_features=30000, ngram_range=(1, 2), sublinear_tf=True)),
        ('clf', LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs'))
    ])
    res = evaluate_on_val(pipe_ctoken, X_train, y_train, X_val, y_val, "C-Token (1-2) 30k C=0.5")
    results.append(res)
    print(f"  -> Acc: {res['val_accuracy']:.4f}, AUROC: {res['val_auroc']:.4f}, MCC: {res['val_mcc']:.4f}, Time: {res['training_time_s']:.1f}s")

    # -------------------------------------------------------------
    # 6. Combined FeatureUnion: Char (2-5) + C-Token (1-2), C=0.5
    # -------------------------------------------------------------
    print("\n[6/10] Evaluating FeatureUnion: Char (2-5) 30k + C-Token (1-2) 20k, C=0.5...")
    union_features = FeatureUnion([
        ('char', TfidfVectorizer(max_features=30000, ngram_range=(2, 5), analyzer='char_wb', sublinear_tf=True)),
        ('ctoken', TfidfVectorizer(tokenizer=c_tokenize, token_pattern=None, max_features=20000, ngram_range=(1, 2), sublinear_tf=True))
    ])
    pipe_union = Pipeline([
        ('union', union_features),
        ('clf', LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs'))
    ])
    res = evaluate_on_val(pipe_union, X_train, y_train, X_val, y_val, "Union: Char(2-5)+CToken(1-2) C=0.5")
    results.append(res)
    print(f"  -> Acc: {res['val_accuracy']:.4f}, AUROC: {res['val_auroc']:.4f}, MCC: {res['val_mcc']:.4f}, Time: {res['training_time_s']:.1f}s")

    # -------------------------------------------------------------
    # 7. Combined FeatureUnion: Char (2-5) + C-Token (1-2), C=0.1
    # -------------------------------------------------------------
    print("\n[7/10] Evaluating FeatureUnion: Char (2-5) + C-Token (1-2), C=0.1 (stronger reg)...")
    pipe_union_c01 = Pipeline([
        ('union', union_features),
        ('clf', LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs'))
    ])
    res = evaluate_on_val(pipe_union_c01, X_train, y_train, X_val, y_val, "Union: Char(2-5)+CToken(1-2) C=0.1")
    results.append(res)
    print(f"  -> Acc: {res['val_accuracy']:.4f}, AUROC: {res['val_auroc']:.4f}, MCC: {res['val_mcc']:.4f}, Time: {res['training_time_s']:.1f}s")

    # -------------------------------------------------------------
    # 8. Tri-Union: Char + C-Token + Security Structural Metrics
    # -------------------------------------------------------------
    print("\n[8/10] Evaluating Tri-Union: Char + C-Token + Security Structural Features, C=0.2...")
    tri_union = FeatureUnion([
        ('char', TfidfVectorizer(max_features=25000, ngram_range=(2, 5), analyzer='char_wb', sublinear_tf=True)),
        ('ctoken', TfidfVectorizer(tokenizer=c_tokenize, token_pattern=None, max_features=15000, ngram_range=(1, 2), sublinear_tf=True)),
        ('struct', Pipeline([
            ('extract', SecurityStructuralFeatureExtractor()),
            ('scale', StandardScaler())
        ]))
    ])
    pipe_tri = Pipeline([
        ('union', tri_union),
        ('clf', LogisticRegression(C=0.2, class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs'))
    ])
    res = evaluate_on_val(pipe_tri, X_train, y_train, X_val, y_val, "Tri-Union: Char+Token+SecurityStruct C=0.2")
    results.append(res)
    print(f"  -> Acc: {res['val_accuracy']:.4f}, AUROC: {res['val_auroc']:.4f}, MCC: {res['val_mcc']:.4f}, Time: {res['training_time_s']:.1f}s")

    # -------------------------------------------------------------
    # 9. Calibrated Linear SVM (SGDClassifier modified_huber) on Tri-Union
    # -------------------------------------------------------------
    print("\n[9/10] Evaluating Calibrated Linear SVM on Tri-Union...")
    pipe_svm = Pipeline([
        ('union', tri_union),
        ('clf', SGDClassifier(loss='modified_huber', alpha=1e-4, penalty='l2', max_iter=1000, random_state=42, class_weight='balanced'))
    ])
    res = evaluate_on_val(pipe_svm, X_train, y_train, X_val, y_val, "Linear SVM (modified_huber) Tri-Union")
    results.append(res)
    print(f"  -> Acc: {res['val_accuracy']:.4f}, AUROC: {res['val_auroc']:.4f}, MCC: {res['val_mcc']:.4f}, Time: {res['training_time_s']:.1f}s")

    # -------------------------------------------------------------
    # 10. Tri-Union with C=0.5
    # -------------------------------------------------------------
    print("\n[10/10] Evaluating Tri-Union with C=0.5...")
    pipe_tri_c05 = Pipeline([
        ('union', tri_union),
        ('clf', LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs'))
    ])
    res = evaluate_on_val(pipe_tri_c05, X_train, y_train, X_val, y_val, "Tri-Union: Char+Token+SecurityStruct C=0.5")
    results.append(res)
    print(f"  -> Acc: {res['val_accuracy']:.4f}, AUROC: {res['val_auroc']:.4f}, MCC: {res['val_mcc']:.4f}, Time: {res['training_time_s']:.1f}s")

    # Summarize all results in DataFrame
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by='val_auroc', ascending=False).reset_index(drop=True)
    
    print("\n" + "=" * 80)
    print("RANKED VALIDATION PERFORMANCE SUMMARY (SORTED BY AUROC)")
    print("=" * 80)
    print(df_res.to_string(index=False))
    
    out_dir = PROJECT_ROOT / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_dir / "classical_model_search_results.csv", index=False)
    with open(out_dir / "classical_model_search_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved search results to {out_dir / 'classical_model_search_results.csv'}")

if __name__ == "__main__":
    main()
