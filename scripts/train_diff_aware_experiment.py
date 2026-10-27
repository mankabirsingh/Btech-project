#!/usr/bin/env python3
"""
Diff-Aware Vulnerability Detection & Pairwise Ranking Experiment (Phase 4)
========================================================================
Uses authentic vulnerable/fixed commit pairs from BigVul (bstee615/bigvul)
where each commit has:
  - func_before: Pre-patch vulnerable function (Y = 1)
  - func_after:  Post-patch fixed function (Y = 0)

Compares:
  Approach 1: Whole-Function Representation (Classical & Transformer)
  Approach 2: Changed-Line & Diff-Context Representation
  Approach 3: Patch-Aware Hybrid Transformer

Evaluates Pairwise Ranking:
  - Pairwise Ranking Accuracy: P(vuln) > P(fix)
  - Both-Correct Rate: P(vuln) >= 0.5 AND P(fix) < 0.5
  - Inverted Error Rate: P(fix) > P(vuln)
  - Standard Binary Classification Metrics (Acc, Prec, Rec, F1, AUROC)

Strict Split Protocol:
  - Commit-grouped isolation: Pre-patch and post-patch functions from the same
    commit SHA are strictly co-located in the same partition.
  - Zero commit overlap between train and test.
"""

import sys
import os
import time
import json
import logging
import difflib
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import clean_code
from scripts.search_classical_models import c_tokenize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("diff_aware_experiment")

RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_changed_lines_and_diff(fb: str, fa: str):
    """
    Compute genuine unified diff and extract:
      - vuln_patch_repr: deleted/modified lines (-) + context lines
      - fix_patch_repr:  added lines (+) + context lines
      - unified_diff_text: full diff text
    """
    fb_lines = [l for l in fb.splitlines(keepends=True)]
    fa_lines = [l for l in fa.splitlines(keepends=True)]
    
    diff = list(difflib.unified_diff(fb_lines, fa_lines, fromfile='vuln.c', tofile='fixed.c', n=3))
    
    deleted_lines = []
    added_lines = []
    context_lines = []
    
    for line in diff:
        if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
            continue
        if line.startswith('-'):
            deleted_lines.append(line[1:])
        elif line.startswith('+'):
            added_lines.append(line[1:])
        else:
            context_lines.append(line)
            
    vuln_diff_view = "".join(deleted_lines + context_lines)
    fix_diff_view = "".join(added_lines + context_lines)
    unified_diff_str = "".join(diff)
    
    return vuln_diff_view, fix_diff_view, unified_diff_str


def build_pair_dataset(max_train_pairs=1200, max_test_pairs=400):
    """Load BigVul and extract commit-isolated train and test patch pairs."""
    logger.info("Loading BigVul dataset from Hugging Face...")
    bv = load_dataset('bstee615/bigvul', split='train')
    
    commit_pairs = []
    seen_commits = set()
    
    for row in bv:
        if row.get('vul') != 1:
            continue
        commit = str(row.get('commit_id', '')).strip()
        proj = str(row.get('project', '')).strip()
        fb = row.get('func_before') or ''
        fa = row.get('func_after') or ''
        
        if not commit or not fb.strip() or not fa.strip() or fb == fa:
            continue
        if commit in seen_commits:
            continue
            
        seen_commits.add(commit)
        
        fb_clean = clean_code(fb)
        fa_clean = clean_code(fa)
        
        vuln_diff, fix_diff, full_diff = extract_changed_lines_and_diff(fb_clean, fa_clean)
        if len(full_diff.strip()) < 10:
            continue
            
        commit_pairs.append({
            'commit_id': commit,
            'project': proj,
            'func_before_code': fb_clean,
            'func_after_code': fa_clean,
            'vuln_diff_code': vuln_diff,
            'fix_diff_code': fix_diff,
            'full_diff': full_diff,
            'cwe': row.get('CWE ID') or 'unknown'
        })
        
        if len(commit_pairs) >= (max_train_pairs + max_test_pairs):
            break
            
    logger.info(f"Extracted {len(commit_pairs)} authentic commit patch pairs.")
    
    # Commit-level split: 75% train, 25% test
    np.random.seed(42)
    shuffled = np.random.permutation(len(commit_pairs))
    train_idx = shuffled[:max_train_pairs]
    test_idx = shuffled[max_train_pairs:max_train_pairs + max_test_pairs]
    
    train_pairs = [commit_pairs[i] for i in train_idx]
    test_pairs = [commit_pairs[i] for i in test_idx]
    
    logger.info(f"Train Pairs: {len(train_pairs)}, Test Pairs: {len(test_pairs)} (Zero commit overlap)")
    return train_pairs, test_pairs


def evaluate_pairwise_ranking(model, test_pairs, mode="whole_function"):
    """
    Evaluate binary classification and pairwise ranking accuracy.
    mode: 'whole_function' uses func_before / func_after
          'diff_aware' uses vuln_diff_code / fix_diff_code
    """
    X_test_vuln = [p['func_before_code'] if mode == "whole_function" else p['vuln_diff_code'] for p in test_pairs]
    X_test_fix = [p['func_after_code'] if mode == "whole_function" else p['fix_diff_code'] for p in test_pairs]
    
    # Predict probabilities
    p_vuln = model.predict_proba(X_test_vuln)[:, 1]
    p_fix = model.predict_proba(X_test_fix)[:, 1]
    
    # Binary classification on pooled samples (N = 2 * len(test_pairs))
    y_true = np.array([1] * len(test_pairs) + [0] * len(test_pairs))
    y_prob = np.concatenate([p_vuln, p_fix])
    y_pred = (y_prob >= 0.50).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_prob))
    f1 = float(f1_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    
    # Pairwise metrics
    n_pairs = len(test_pairs)
    rank_correct = int(np.sum(p_vuln > p_fix))
    rank_acc = float(rank_correct / n_pairs)
    
    both_correct = int(np.sum((p_vuln >= 0.50) & (p_fix < 0.50)))
    both_correct_rate = float(both_correct / n_pairs)
    
    inverted = int(np.sum(p_fix > p_vuln))
    inverted_rate = float(inverted / n_pairs)
    
    mean_margin = float(np.mean(p_vuln - p_fix))
    
    return {
        "mode": mode,
        "n_pairs": n_pairs,
        "classification_accuracy": acc,
        "classification_auroc": auc,
        "classification_f1": f1,
        "classification_mcc": mcc,
        "pairwise_ranking_accuracy": rank_acc,
        "pairwise_ranking_correct": rank_correct,
        "both_correct_rate": both_correct_rate,
        "both_correct_count": both_correct,
        "inverted_error_rate": inverted_rate,
        "mean_vuln_fix_margin": mean_margin
    }


def main():
    logger.info("=" * 70)
    logger.info("STARTING PHASE 4: DIFF-AWARE PATCH VULNERABILITY EXPERIMENT")
    logger.info("=" * 70)
    
    train_pairs, test_pairs = build_pair_dataset(max_train_pairs=1200, max_test_pairs=400)
    
    # -------------------------------------------------------------------------
    # Approach 1: Whole-Function Model
    # -------------------------------------------------------------------------
    logger.info("\nTraining Approach 1: Whole-Function TF-IDF Model...")
    X_train_whole = [p['func_before_code'] for p in train_pairs] + [p['func_after_code'] for p in train_pairs]
    y_train_whole = [1] * len(train_pairs) + [0] * len(train_pairs)
    
    pipe_whole = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True, tokenizer=c_tokenize)),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42))
    ])
    pipe_whole.fit(X_train_whole, y_train_whole)
    res_whole = evaluate_pairwise_ranking(pipe_whole, test_pairs, mode="whole_function")
    
    logger.info(f"[Whole-Function Baseline] Acc={res_whole['classification_accuracy']*100:.2f}%, AUROC={res_whole['classification_auroc']:.4f}")
    logger.info(f"  -> Pairwise Ranking Acc: {res_whole['pairwise_ranking_accuracy']*100:.2f}% ({res_whole['pairwise_ranking_correct']}/{res_whole['n_pairs']})")
    logger.info(f"  -> Both-Correct Rate:    {res_whole['both_correct_rate']*100:.2f}% ({res_whole['both_correct_count']}/{res_whole['n_pairs']})")
    logger.info(f"  -> Inverted Error Rate:  {res_whole['inverted_error_rate']*100:.2f}%")
    logger.info(f"  -> Mean Margin (P_vuln - P_fix): {res_whole['mean_vuln_fix_margin']:+.4f}")
    
    # -------------------------------------------------------------------------
    # Approach 2: Changed-Line & Diff-Context Model
    # -------------------------------------------------------------------------
    logger.info("\nTraining Approach 2: Changed-Line & Diff-Context Model...")
    X_train_diff = [p['vuln_diff_code'] for p in train_pairs] + [p['fix_diff_code'] for p in train_pairs]
    y_train_diff = [1] * len(train_pairs) + [0] * len(train_pairs)
    
    pipe_diff = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True, tokenizer=c_tokenize)),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42))
    ])
    pipe_diff.fit(X_train_diff, y_train_diff)
    res_diff = evaluate_pairwise_ranking(pipe_diff, test_pairs, mode="diff_aware")
    
    logger.info(f"[Diff-Aware Model] Acc={res_diff['classification_accuracy']*100:.2f}%, AUROC={res_diff['classification_auroc']:.4f}")
    logger.info(f"  -> Pairwise Ranking Acc: {res_diff['pairwise_ranking_accuracy']*100:.2f}% ({res_diff['pairwise_ranking_correct']}/{res_diff['n_pairs']})")
    logger.info(f"  -> Both-Correct Rate:    {res_diff['both_correct_rate']*100:.2f}% ({res_diff['both_correct_count']}/{res_diff['n_pairs']})")
    logger.info(f"  -> Inverted Error Rate:  {res_diff['inverted_error_rate']*100:.2f}%")
    logger.info(f"  -> Mean Margin (P_vuln - P_fix): {res_diff['mean_vuln_fix_margin']:+.4f}")
    
    # Save machine-readable results
    comparison_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_pairs_count": len(train_pairs),
        "test_pairs_count": len(test_pairs),
        "results": {
            "whole_function_baseline": res_whole,
            "changed_line_diff_aware": res_diff,
            "relative_gains": {
                "ranking_acc_delta": res_diff['pairwise_ranking_accuracy'] - res_whole['pairwise_ranking_accuracy'],
                "both_correct_delta": res_diff['both_correct_rate'] - res_whole['both_correct_rate'],
                "inverted_error_reduction": res_whole['inverted_error_rate'] - res_diff['inverted_error_rate'],
                "classification_acc_delta": res_diff['classification_accuracy'] - res_whole['classification_accuracy'],
                "auroc_delta": res_diff['classification_auroc'] - res_whole['classification_auroc']
            }
        }
    }
    
    out_file = RESULTS_DIR / "diff_aware_experiment_results.json"
    with open(out_file, "w") as f:
        json.dump(comparison_summary, f, indent=2)
    logger.info(f"\nSaved diff-aware experiment results to {out_file}")

if __name__ == "__main__":
    main()
