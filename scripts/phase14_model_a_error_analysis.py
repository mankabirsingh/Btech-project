#!/usr/bin/env python3
"""
Phase 14 Model A Deep Error Analysis Script
===========================================
Performs comprehensive empirical diagnostics on Model A's predictions across:
- Envoy Proxy (Modern C++17)
- OpenSSL (C Cryptography)
- GPAC (C Multimedia Demuxer)
- Vim (C Text Editor & Parser)

Analyzes:
1. False Positives (FP) and False Negatives (FN) breakdown
2. Function length effects (lines, characters) on accuracy
3. C++ construct prevalence (templates, namespaces, virtual functions, smart pointers, GoogleTest)
4. Tokenization fragmentation and truncation (>384 tokens)
5. Static security findings & CWE correlation
6. Concrete sample extracts with causal failure mechanisms
"""

import sys
import os
import re
import json
import time
import pickle
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.security.static_analysis import analyze_code

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase14"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


CPP_PATTERNS = {
    "template_usage": r"\btemplate\s*<",
    "namespace_usage": r"\bnamespace\s+\w+",
    "class_definition": r"\bclass\s+\w+",
    "virtual_dispatch": r"\bvirtual\s+\w+|\boverride\b",
    "smart_pointers": r"\bstd::(?:unique_ptr|shared_ptr|weak_ptr|make_unique|make_shared)\b",
    "cpp_containers": r"\bstd::(?:vector|string|map|unordered_map|set|array)\b",
    "cpp_casts": r"\b(?:static_cast|dynamic_cast|reinterpret_cast|const_cast)\s*<",
    "googletest_macros": r"\b(?:EXPECT_|ASSERT_|MOCK_METHOD|TEST_F|TEST_P)\b",
    "raii_locks": r"\bstd::(?:lock_guard|unique_lock|scoped_lock)\b",
    "lambda_functions": r"\[[^\]]*\]\s*\([^\)]*\)\s*(?:->\s*[^\{]+)?\s*\{",
    "exception_handling": r"\btry\s*\{|\bcatch\s*\(|\bthrow\b",
}


def analyze_code_features(code: str) -> Dict[str, Any]:
    lines = code.splitlines()
    loc = len(lines)
    char_count = len(code)
    
    features = {
        "loc": loc,
        "char_count": char_count,
    }
    for feat_name, pattern in CPP_PATTERNS.items():
        features[feat_name] = bool(re.search(pattern, code))
        
    return features


def main():
    print("=" * 80)
    print("PHASE 14 — STEP 2: MODEL A DEEP ERROR ANALYSIS")
    print("=" * 80)

    # 1. Load clean test dataset
    clean_test_path = SPLITS_DIR / "vulnerability_test_heldout_projects_clean.parquet"
    df = pd.read_parquet(clean_test_path)
    print(f"Loaded {len(df)} clean test samples across projects: {df['project'].value_counts().to_dict()}")

    # 2. Load cached predictions or compute them
    base_results_path = RESULTS_DIR / "baseline_reproduction.json"
    if not base_results_path.exists():
        print("Waiting for baseline reproduction to finish...")
        return

    with open(base_results_path, "r") as f:
        base_data = json.load(f)

    # Load CodeBERT test embeddings and model
    emb_cache_path = RESULTS_DIR / "codebert_test_clean_emb.npy"
    if not emb_cache_path.exists():
        print("CodeBERT embeddings cache missing!")
        return

    emb_test_clean = np.load(emb_cache_path)
    with open(MODELS_DIR / "expanded_codebert_hybrid_best.pkl", "rb") as f:
        cb_checkpoint = pickle.load(f)
    cb_clf = cb_checkpoint['clf']
    cb_scaler = cb_checkpoint['scaler']

    from scripts.search_classical_models import SecurityStructuralFeatureExtractor
    extractor = SecurityStructuralFeatureExtractor()
    X_test_clean = df['code'].values
    y_test_clean = df['vuln_label'].values.astype(int)
    projects = df['project'].values

    struct_test_clean = cb_scaler.transform(extractor.transform(X_test_clean))
    X_cb_test_clean = np.hstack([emb_test_clean, struct_test_clean])
    prob_cb = cb_clf.predict_proba(X_cb_test_clean)[:, 1]
    pred_cb = (prob_cb >= 0.50).astype(int)

    # Load Tri-Union model for comparison
    with open(MODELS_DIR / "expanded_tri_union_best.pkl", "rb") as f:
        tri_pipeline = pickle.load(f)['pipeline']
    prob_tri = tri_pipeline.predict_proba(X_test_clean)[:, 1]
    pred_tri = (prob_tri >= 0.50).astype(int)

    # 3. Tokenizer analysis
    print("Analyzing CodeBERT tokenization fragmentation...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    token_counts = []
    truncation_flags = []
    for code in X_test_clean:
        toks = tokenizer.tokenize(str(code))
        token_counts.append(len(toks))
        truncation_flags.append(len(toks) > 384)

    df['token_count'] = token_counts
    df['truncated_384'] = truncation_flags
    df['prob_cb'] = prob_cb
    df['pred_cb'] = pred_cb
    df['correct_cb'] = (df['pred_cb'] == df['vuln_label'])
    df['prob_tri'] = prob_tri
    df['pred_tri'] = pred_tri
    df['correct_tri'] = (df['pred_tri'] == df['vuln_label'])

    # 4. Extract C++ syntactic features
    print("Extracting syntactic and structural C++ patterns...")
    syntax_records = [analyze_code_features(c) for c in X_test_clean]
    syntax_df = pd.DataFrame(syntax_records)
    for col in syntax_df.columns:
        df[col] = syntax_df[col].values

    # 5. Project-Specific Deep Diagnostics
    project_diagnostics = {}
    
    for proj in ['envoy', 'openssl', 'gpac', 'vim']:
        pdf = df[df['project'] == proj].copy()
        n = len(pdf)
        n_vuln = int(pdf['vuln_label'].sum())
        n_benign = n - n_vuln

        # Confusion breakdown CodeBERT
        tp = int(((pdf['pred_cb'] == 1) & (pdf['vuln_label'] == 1)).sum())
        fp = int(((pdf['pred_cb'] == 1) & (pdf['vuln_label'] == 0)).sum())
        tn = int(((pdf['pred_cb'] == 0) & (pdf['vuln_label'] == 0)).sum())
        fn = int(((pdf['pred_cb'] == 0) & (pdf['vuln_label'] == 1)).sum())
        acc_cb = float(pdf['correct_cb'].mean())
        acc_tri = float(pdf['correct_tri'].mean())

        # Error groups
        fps = pdf[(pdf['pred_cb'] == 1) & (pdf['vuln_label'] == 0)]
        fns = pdf[(pdf['pred_cb'] == 0) & (pdf['vuln_label'] == 1)]

        # Length diagnostics
        median_loc_all = float(pdf['loc'].median())
        median_loc_correct = float(pdf[pdf['correct_cb']]['loc'].median()) if pdf['correct_cb'].sum() > 0 else 0.0
        median_loc_incorrect = float(pdf[~pdf['correct_cb']]['loc'].median()) if (~pdf['correct_cb']).sum() > 0 else 0.0

        # Tokenization diagnostics
        median_tokens = float(pdf['token_count'].median())
        pct_truncated = float(pdf['truncated_384'].mean() * 100)

        # C++ construct breakdown
        cpp_prevalence = {}
        cpp_accuracy_cb = {}
        cpp_accuracy_tri = {}
        for feat in CPP_PATTERNS.keys():
            count = int(pdf[feat].sum())
            prev = float(count / n * 100)
            cpp_prevalence[feat] = {"count": count, "prevalence_pct": round(prev, 1)}
            if count >= 10:
                subset = pdf[pdf[feat]]
                cpp_accuracy_cb[feat] = round(float(subset['correct_cb'].mean() * 100), 2)
                cpp_accuracy_tri[feat] = round(float(subset['correct_tri'].mean() * 100), 2)

        # Representative Failure Cases
        rep_fp = []
        for _, row in fps.head(3).iterrows():
            rep_fp.append({
                "loc": int(row['loc']),
                "prob_cb": round(float(row['prob_cb']), 4),
                "prob_tri": round(float(row['prob_tri']), 4),
                "snippet": row['code'][:350]
            })

        rep_fn = []
        for _, row in fns.head(3).iterrows():
            rep_fn.append({
                "loc": int(row['loc']),
                "prob_cb": round(float(row['prob_cb']), 4),
                "prob_tri": round(float(row['prob_tri']), 4),
                "snippet": row['code'][:350]
            })

        project_diagnostics[proj] = {
            "n_samples": n,
            "n_vuln": n_vuln,
            "n_benign": n_benign,
            "codebert_accuracy": round(acc_cb * 100, 2),
            "tri_union_accuracy": round(acc_tri * 100, 2),
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "fpr": round(float(fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0, 2),
            "fnr": round(float(fn / (fn + tp) * 100) if (fn + tp) > 0 else 0.0, 2),
            "median_loc": median_loc_all,
            "median_loc_correct": median_loc_correct,
            "median_loc_incorrect": median_loc_incorrect,
            "median_tokens": median_tokens,
            "pct_truncated_384": round(pct_truncated, 2),
            "cpp_construct_prevalence": cpp_prevalence,
            "accuracy_on_cpp_constructs_codebert": cpp_accuracy_cb,
            "accuracy_on_cpp_constructs_tri_union": cpp_accuracy_tri,
            "representative_false_positives": rep_fp,
            "representative_false_negatives": rep_fn,
        }

        print(f"\n--- Project: {proj.upper()} (N = {n}, Vuln = {n_vuln}, Benign = {n_benign}) ---")
        print(f"  CodeBERT Acc: {acc_cb*100:.2f}% (TP={tp}, FP={fp}, TN={tn}, FN={fn}) | FPR={fp/(fp+tn)*100:.1f}%, FNR={fn/(fn+tp)*100:.1f}%")
        print(f"  Tri-Union Acc: {acc_tri*100:.2f}%")
        print(f"  Median LOC: {median_loc_all:.1f} (Correct: {median_loc_correct:.1f}, Incorrect: {median_loc_incorrect:.1f})")
        print(f"  Median Tokens: {median_tokens:.1f} | Truncated (>384): {pct_truncated:.1f}%")
        if proj == "envoy":
            print("  Top Envoy C++ Construct Accuracies (CodeBERT vs Tri-Union):")
            for feat in cpp_accuracy_cb:
                print(f"    • {feat:20}: CodeBERT={cpp_accuracy_cb[feat]:.1f}% vs Tri-Union={cpp_accuracy_tri[feat]:.1f}% (N={cpp_prevalence[feat]['count']})")

    # Save to json
    output_path = RESULTS_DIR / "model_a_error_analysis.json"
    with open(output_path, "w") as f:
        json.dump(project_diagnostics, f, indent=2)

    print(f"\n[✓] Error analysis saved to: {output_path}")


if __name__ == "__main__":
    main()
