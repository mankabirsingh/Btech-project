#!/usr/bin/env python3
"""
Comprehensive Error Analysis and Paired Statistical Comparison
==============================================================
Analyzes test predictions (N=5,412) from:
- CodeBERT Original ([CLS] 256)
- CodeBERT Improved Hybrid ([CLS+MeanPool] 384 + SecurityStruct)
- Tri-Union Best Model (Char + Token + SecurityStruct)
- Classical Baselines (Char, Word)

Performs:
1. Error breakdown by dataset (Devign vs. PrimeVul)
2. Error breakdown by genuine CWE category (where labels exist)
3. Error breakdown by code length & token truncation
4. Vulnerable/fixed pair ranking accuracy on PrimeVul commit pairs
5. Prediction confidence calibration
6. Rigorous paired statistical tests (McNemar's test, paired bootstrap CIs)
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def mcnemar_test(y_true, pred_a, pred_b):
    """
    Perform McNemar's paired test with continuity correction.
    Table:
                Model B Correct    Model B Wrong
    Model A Correct     a                b
    Model A Wrong       c                d
    """
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    
    a = int(np.sum(correct_a & correct_b))
    b = int(np.sum(correct_a & (~correct_b)))  # A right, B wrong
    c = int(np.sum((~correct_a) & correct_b))  # B right, A wrong
    d = int(np.sum((~correct_a) & (~correct_b)))
    
    contingency_table = [[a, b], [c, d]]
    
    # McNemar's chi-squared with Edwards continuity correction
    if (b + c) > 0:
        chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
        p_val = stats.chi2.sf(chi2, df=1)
    else:
        chi2 = 0.0
        p_val = 1.0
        
    odds_ratio = (c / b) if b > 0 else (np.inf if c > 0 else 1.0)
    
    return {
        'contingency_table': contingency_table,
        'b_only_A_correct': b,
        'c_only_B_correct': c,
        'chi2_stat': float(chi2),
        'p_value': float(p_val),
        'odds_ratio': float(odds_ratio),
        'significant_005': bool(p_val < 0.05),
        'significant_001': bool(p_val < 0.01),
    }


def paired_bootstrap_delta(y_true, pred_a, prob_a, pred_b, prob_b, n_bootstraps=1000, seed=42):
    """Compute paired bootstrap differences (Model B - Model A)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    
    d_acc, d_auroc, d_f1, d_mcc = [], [], [], []
    
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
            
        pa_p = pred_a[idx]
        pa_pr = prob_a[idx]
        pb_p = pred_b[idx]
        pb_pr = prob_b[idx]
        
        acc_a = accuracy_score(yt, pa_p)
        acc_b = accuracy_score(yt, pb_p)
        d_acc.append(acc_b - acc_a)
        
        auroc_a = roc_auc_score(yt, pa_pr)
        auroc_b = roc_auc_score(yt, pb_pr)
        d_auroc.append(auroc_b - auroc_a)
        
        f1_a = f1_score(yt, pa_p, zero_division=0)
        f1_b = f1_score(yt, pb_p, zero_division=0)
        d_f1.append(f1_b - f1_a)
        
        mcc_a = matthews_corrcoef(yt, pa_p)
        mcc_b = matthews_corrcoef(yt, pb_p)
        d_mcc.append(mcc_b - mcc_a)
        
    return {
        'delta_acc_mean': float(np.mean(d_acc)),
        'delta_acc_95ci': [float(np.percentile(d_acc, 2.5)), float(np.percentile(d_acc, 97.5))],
        'delta_auroc_mean': float(np.mean(d_auroc)),
        'delta_auroc_95ci': [float(np.percentile(d_auroc, 2.5)), float(np.percentile(d_auroc, 97.5))],
        'delta_f1_mean': float(np.mean(d_f1)),
        'delta_f1_95ci': [float(np.percentile(d_f1, 2.5)), float(np.percentile(d_f1, 97.5))],
        'delta_mcc_mean': float(np.mean(d_mcc)),
        'delta_mcc_95ci': [float(np.percentile(d_mcc, 2.5)), float(np.percentile(d_mcc, 97.5))],
    }


def main():
    print("=" * 80)
    print("COMPREHENSIVE ERROR ANALYSIS & PAIRED STATISTICAL VERIFICATION")
    print("=" * 80)
    
    # 1. Load predictions
    pred_path = PROJECT_ROOT / "results" / "experiments" / "test_predictions_all_models.parquet"
    if not pred_path.exists():
        print(f"Error: {pred_path} not found! Run generate_all_test_predictions.py first.")
        sys.exit(1)
        
    df = pd.read_parquet(pred_path)
    test_raw = pd.read_parquet(PROJECT_ROOT / "data" / "splits" / "vulnerability_test.parquet")
    df['code'] = test_raw['code']
    df['char_length'] = df['code'].str.len()
    df['line_count'] = df['code'].str.count('\n') + 1
    
    y_true = df['vuln_label'].values
    print(f"Loaded test predictions: {len(df)} samples (Vuln: {sum(y_true)}, Benign: {len(y_true)-sum(y_true)})")
    
    models = {
        'Char Baseline': ('pred_char', 'prob_char'),
        'Word Baseline': ('pred_word', 'prob_word'),
        'CodeBERT Original': ('pred_codebert_orig', 'prob_codebert_orig'),
        'Tri-Union Best': ('pred_tri', 'prob_tri'),
        'CodeBERT Improved Hybrid': ('pred_codebert_imp', 'prob_codebert_imp'),
    }
    
    analysis_results = {}
    
    # =========================================================================
    # SECTION A: Overall Verified Metrics & Confusion Matrices
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION A: OVERALL VERIFIED METRICS & CONFUSION MATRICES (N=5,412)")
    print("=" * 80)
    
    overall_metrics = {}
    for name, (pred_col, prob_col) in models.items():
        preds = df[pred_col].values
        probs = df[prob_col].values
        
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        mcc = matthews_corrcoef(y_true, preds)
        auroc = roc_auc_score(y_true, probs)
        ap = average_precision_score(y_true, probs)
        brier = brier_score_loss(y_true, probs)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        overall_metrics[name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'mcc': float(mcc),
            'auroc': float(auroc),
            'average_precision': float(ap),
            'brier_score': float(brier),
            'fpr': float(fpr),
            'fnr': float(fnr),
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        }
        print(f"\nModel: {name}")
        print(f"  Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | MCC: {mcc:+.4f} | AUROC: {auroc:.4f} | AP: {ap:.4f}")
        print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp} | FPR: {fpr:.2%}, FNR: {fnr:.2%}")
        
    analysis_results['overall_metrics'] = overall_metrics

    # =========================================================================
    # SECTION B: Dataset Source Breakdown (Devign vs. PrimeVul)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION B: BREAKDOWN BY DATASET SOURCE (Devign vs. PrimeVul)")
    print("=" * 80)
    
    dataset_breakdown = {}
    for src in ['devign', 'primevul']:
        sub_df = df[df['source_dataset'] == src]
        y_src = sub_df['vuln_label'].values
        print(f"\n--- Source: {src.upper()} (N = {len(sub_df)}, Positives = {sum(y_src)} [{sum(y_src)/len(y_src):.2%}]) ---")
        dataset_breakdown[src] = {}
        
        for name, (pred_col, prob_col) in models.items():
            p_preds = sub_df[pred_col].values
            p_probs = sub_df[prob_col].values
            
            acc = accuracy_score(y_src, p_preds)
            prec = precision_score(y_src, p_preds, zero_division=0)
            rec = recall_score(y_src, p_preds, zero_division=0)
            f1 = f1_score(y_src, p_preds, zero_division=0)
            mcc = matthews_corrcoef(y_src, p_preds)
            auroc = roc_auc_score(y_src, p_probs)
            ap = average_precision_score(y_src, p_probs)
            cm = confusion_matrix(y_src, p_preds)
            tn, fp, fn, tp = cm.ravel()
            
            dataset_breakdown[src][name] = {
                'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec),
                'f1': float(f1), 'mcc': float(mcc), 'auroc': float(auroc), 'ap': float(ap),
                'cm': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
            }
            print(f"  {name:25s} -> Acc: {acc:.4f}, AUROC: {auroc:.4f}, MCC: {mcc:+.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}")
            
    analysis_results['dataset_breakdown'] = dataset_breakdown

    # =========================================================================
    # SECTION C: PrimeVul Vulnerable / Fixed Patch Pair Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION C: VULNERABLE/FIXED PATCH PAIR RANKING ACCURACY (PrimeVul)")
    print("=" * 80)
    
    pv_df = df[df['source_dataset'] == 'primevul'].copy()
    # Find complete pairs sharing group_id
    pair_counts = pv_df.groupby('group_id').size()
    complete_pair_groups = pair_counts[pair_counts == 2].index.tolist()
    print(f"Total PrimeVul samples in test set: {len(pv_df)}")
    print(f"Complete pairs in test set: {len(complete_pair_groups)} pairs ({len(complete_pair_groups)*2} samples)")
    
    pair_results = {}
    for name, (pred_col, prob_col) in models.items():
        n_correct_ranking = 0
        n_both_correct = 0
        n_both_wrong = 0
        n_identical_prob = 0
        
        for g_id in complete_pair_groups:
            pair = pv_df[pv_df['group_id'] == g_id]
            vuln_sample = pair[pair['vuln_label'] == 1]
            fix_sample = pair[pair['vuln_label'] == 0]
            
            if len(vuln_sample) == 1 and len(fix_sample) == 1:
                p_vuln = vuln_sample[prob_col].values[0]
                p_fix = fix_sample[prob_col].values[0]
                
                pred_v = vuln_sample[pred_col].values[0]
                pred_f = fix_sample[pred_col].values[0]
                
                if p_vuln > p_fix:
                    n_correct_ranking += 1
                elif p_vuln == p_fix:
                    n_identical_prob += 1
                    
                if pred_v == 1 and pred_f == 0:
                    n_both_correct += 1
                elif pred_v == 0 and pred_f == 1:
                    n_both_wrong += 1
                    
        n_total_pairs = len(complete_pair_groups)
        ranking_acc = n_correct_ranking / n_total_pairs
        both_correct_rate = n_both_correct / n_total_pairs
        both_wrong_rate = n_both_wrong / n_total_pairs
        
        pair_results[name] = {
            'total_pairs': n_total_pairs,
            'pairwise_ranking_accuracy': float(ranking_acc),
            'both_correct_rate': float(both_correct_rate),
            'both_wrong_inverted_rate': float(both_wrong_rate),
            'identical_prob_count': int(n_identical_prob),
        }
        print(f"  {name:25s} -> Pairwise Ranking Acc: {ranking_acc:.2%} ({n_correct_ranking}/{n_total_pairs}), Both Correct: {both_correct_rate:.2%}")
        
    analysis_results['patch_pair_analysis'] = pair_results

    # =========================================================================
    # SECTION D: Breakdown by Function Length & Truncation
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION D: BREAKDOWN BY FUNCTION LENGTH & TRUNCATION")
    print("=" * 80)
    
    # Define length buckets: Short, Medium, Long, Truncated
    # In CodeBERT, ~384 tokens corresponds roughly to ~1,500 - 2,000 characters
    length_bins = [0, 500, 1500, 3000, 1000000]
    length_labels = ['Short (<500c)', 'Medium (500-1500c)', 'Long (1500-3000c)', 'Truncated (>3000c)']
    df['length_bucket'] = pd.cut(df['char_length'], bins=length_bins, labels=length_labels)
    
    length_breakdown = {}
    for bucket in length_labels:
        b_df = df[df['length_bucket'] == bucket]
        yb = b_df['vuln_label'].values
        print(f"\n--- Bucket: {bucket} (N = {len(b_df)}, Positives = {sum(yb)} [{sum(yb)/len(yb):.2%}]) ---")
        length_breakdown[bucket] = {'n_samples': int(len(b_df)), 'pos_rate': float(sum(yb)/len(yb))}
        
        for name in ['Tri-Union Best', 'CodeBERT Original', 'CodeBERT Improved Hybrid']:
            pred_col, prob_col = models[name]
            acc = accuracy_score(yb, b_df[pred_col].values)
            auroc = roc_auc_score(yb, b_df[prob_col].values)
            mcc = matthews_corrcoef(yb, b_df[pred_col].values)
            length_breakdown[bucket][name] = {'acc': float(acc), 'auroc': float(auroc), 'mcc': float(mcc)}
            print(f"  {name:25s} -> Acc: {acc:.4f}, AUROC: {auroc:.4f}, MCC: {mcc:+.4f}")
            
    analysis_results['length_breakdown'] = length_breakdown

    # =========================================================================
    # SECTION E: Genuine CWE Category Breakdown
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION E: BREAKDOWN BY GENUINE CWE CATEGORY (Annotated Samples Only)")
    print("=" * 80)
    
    # Note: Only vulnerable samples in Devign and paired PrimeVul have genuine CWEs.
    cwe_df = df[df['cwe'].notna() & (df['cwe'] != '') & (df['cwe'] != 'Unknown')].copy()
    print(f"Total samples with genuine CWE annotations: {len(cwe_df)} / {len(df)}")
    
    top_cwes = cwe_df['cwe'].value_counts().head(8).index.tolist()
    cwe_breakdown = {}
    for cwe_id in top_cwes:
        sub_cwe = cwe_df[cwe_df['cwe'] == cwe_id]
        y_cwe = sub_cwe['vuln_label'].values
        cwe_breakdown[cwe_id] = {'n_samples': int(len(sub_cwe)), 'vuln_count': int(sum(y_cwe))}
        print(f"\nCWE: {cwe_id} (N={len(sub_cwe)}, Vuln={sum(y_cwe)})")
        
        for name in ['Tri-Union Best', 'CodeBERT Original', 'CodeBERT Improved Hybrid']:
            pred_col, prob_col = models[name]
            # Recall on this specific CWE
            vuln_subset = sub_cwe[sub_cwe['vuln_label'] == 1]
            rec = recall_score(vuln_subset['vuln_label'].values, vuln_subset[pred_col].values, zero_division=0)
            avg_prob = vuln_subset[prob_col].mean()
            cwe_breakdown[cwe_id][name] = {'recall': float(rec), 'mean_vuln_prob': float(avg_prob)}
            print(f"  {name:25s} -> Recall on {cwe_id}: {rec:.2%} (Mean Pred Prob: {avg_prob:.3f})")
            
    analysis_results['cwe_breakdown'] = cwe_breakdown

    # =========================================================================
    # SECTION F: Paired Statistical Comparison & Effect Sizes
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION F: PAIRED STATISTICAL COMPARISONS ON IDENTICAL TEST EXAMPLES")
    print("=" * 80)
    
    comparisons = [
        ('CodeBERT Original', 'CodeBERT Improved Hybrid'),
        ('Tri-Union Best', 'CodeBERT Improved Hybrid'),
        ('Char Baseline', 'CodeBERT Improved Hybrid'),
        ('Char Baseline', 'Tri-Union Best'),
    ]
    
    paired_test_results = {}
    for name_a, name_b in comparisons:
        pred_col_a, prob_col_a = models[name_a]
        pred_col_b, prob_col_b = models[name_b]
        
        pa_p = df[pred_col_a].values
        pa_pr = df[prob_col_a].values
        pb_p = df[pred_col_b].values
        pb_pr = df[prob_col_b].values
        
        # McNemar test
        mcnemar_res = mcnemar_test(y_true, pa_p, pb_p)
        
        # Paired bootstrap deltas
        boot_res = paired_bootstrap_delta(y_true, pa_p, pa_pr, pb_p, pb_pr, n_bootstraps=1000)
        
        # Wilcoxon signed rank test on predicted probabilities
        w_stat, w_pval = stats.wilcoxon(pb_pr, pa_pr)
        
        comp_key = f"{name_a} vs. {name_b}"
        paired_test_results[comp_key] = {
            'mcnemar': mcnemar_res,
            'bootstrap_deltas': boot_res,
            'wilcoxon_prob_test': {'w_stat': float(w_stat), 'p_value': float(w_pval)},
        }
        
        print(f"\n--- Comparison: {comp_key} ---")
        print(f"  Contingency Table: Both Correct={mcnemar_res['contingency_table'][0][0]}, "
              f"Only {name_a} Correct={mcnemar_res['b_only_A_correct']}, "
              f"Only {name_b} Correct={mcnemar_res['c_only_B_correct']}, Both Wrong={mcnemar_res['contingency_table'][1][1]}")
        print(f"  McNemar Chi2: {mcnemar_res['chi2_stat']:.4f}, p-value: {mcnemar_res['p_value']:.4e} "
              f"({'Statistically Significant (p < 0.01)' if mcnemar_res['significant_001'] else 'Not Significant'})")
        print(f"  Delta Accuracy: {boot_res['delta_acc_mean']:+.4f} (95% CI: [{boot_res['delta_acc_95ci'][0]:+.4f}, {boot_res['delta_acc_95ci'][1]:+.4f}])")
        print(f"  Delta AUROC:    {boot_res['delta_auroc_mean']:+.4f} (95% CI: [{boot_res['delta_auroc_95ci'][0]:+.4f}, {boot_res['delta_auroc_95ci'][1]:+.4f}])")
        print(f"  Delta MCC:      {boot_res['delta_mcc_mean']:+.4f} (95% CI: [{boot_res['delta_mcc_95ci'][0]:+.4f}, {boot_res['delta_mcc_95ci'][1]:+.4f}])")
        
    analysis_results['paired_statistical_tests'] = paired_test_results
    
    # Save complete JSON analysis
    out_dir = PROJECT_ROOT / "results" / "experiments"
    out_file = out_dir / "comprehensive_error_analysis.json"
    with open(out_file, "w") as f:
        json.dump(analysis_results, f, indent=2)
    print(f"\nSaved comprehensive error analysis to {out_file}")

if __name__ == "__main__":
    main()
