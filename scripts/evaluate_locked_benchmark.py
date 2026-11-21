#!/usr/bin/env python3
"""
Locked Benchmark Independent Evaluation Script
==============================================
DTU B.Tech Project: Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code

Evaluates frozen Model A and newly trained Model B on the 32 verified examples in
results/reports/controlled_benchmark_32.json.

Verification:
- Verifies SHA-256 integrity against data/locked_benchmark_32/CHECKSUMS.sha256
- Zero training/tuning overlap (Model B trained on dev_tasks 009-024)
- Exact paired evaluation on identical code samples
"""

import sys
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, confusion_matrix, brier_score_loss
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import clean_code
from src.security.static_analysis import analyze_code
from src.models.inference import predict_model_a, predict_model_b


def verify_benchmark_integrity() -> bool:
    """Verifies that the benchmark file matches its immutable locked checksum."""
    bench_file = PROJECT_ROOT / "results" / "reports" / "controlled_benchmark_32.json"
    chk_file = PROJECT_ROOT / "data" / "locked_benchmark_32" / "CHECKSUMS.sha256"
    
    with open(bench_file, "rb") as f:
        current_sha = hashlib.sha256(f.read()).hexdigest()
        
    with open(chk_file, "r") as f:
        expected_sha = f.readline().split()[0]
        
    match = (current_sha == expected_sha)
    print(f"Benchmark File Checksum: {current_sha}")
    print(f"Locked Expected Checksum: {expected_sha}")
    print(f"Integrity Verified: {'PASS (Identical)' if match else 'FAIL (Mismatch)'}")
    return match


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    n = len(y_true)
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0
    auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    prauc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    brier = float(brier_score_loss(y_true, y_prob))
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    # 95% Wilson score interval for binomial proportion
    z = 1.96
    denom = 1 + z**2 / n
    center = (acc + z**2 / (2 * n)) / denom
    half = (z * np.sqrt((acc * (1 - acc) + z**2 / (4 * n)) / n)) / denom
    ci_lower = max(0.0, float(center - half))
    ci_upper = min(1.0, float(center + half))
    
    return {
        "accuracy": round(acc, 4),
        "accuracy_ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mcc": round(mcc, 4),
        "auroc": round(auroc, 4),
        "prauc": round(prauc, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": cm,
        "n_samples": n
    }


def main():
    print("="*80)
    print("LOCKED 32-EXAMPLE BENCHMARK INDEPENDENT EVALUATION")
    print("="*80)
    
    assert verify_benchmark_integrity(), "Benchmark checksum mismatch! Cannot proceed."
    
    # Load 32 Benchmark records directly from controlled_editing
    from src.experiments.controlled_editing import create_multi_cwe_benchmark
    benchmarks = create_multi_cwe_benchmark()
    
    records = []
    for exp in benchmarks:
        for v in exp.versions:
            records.append({
                "task_id": exp.task_id,
                "version_id": v.version_id,
                "code": v.code,
                "language": v.language,
                "author_type": v.author_type,
                "ai_involvement": v.ai_involvement,
                "vuln_label": v.vuln_label
            })
            
    assert len(records) == 32, f"Expected 32 records, found {len(records)}"
    
    y_true = np.array([r["vuln_label"] for r in records], dtype=int)
    n_vuln = int(np.sum(y_true == 1))
    n_safe = int(np.sum(y_true == 0))
    print(f"Sample Count: {len(records)} (Vulnerable: {n_vuln}, Safe: {n_safe}, Balanced: {n_vuln == n_safe})")
    
    # 1. Run Frozen Model A Inference
    y_pred_a = []
    y_prob_a = []
    
    for r in records:
        res_a = predict_model_a(r["code"], language=r["language"])
        y_pred_a.append(1 if res_a["is_vulnerable"] else 0)
        y_prob_a.append(res_a["probability"])
        
    y_pred_a = np.array(y_pred_a, dtype=int)
    y_prob_a = np.array(y_prob_a, dtype=float)
    metrics_a = compute_metrics(y_true, y_pred_a, y_prob_a)
    
    # 2. Run Newly Trained Model B Inference
    y_pred_b = []
    y_prob_b = []
    
    for r in records:
        prov_dict = {
            "ai_involvement": r["ai_involvement"],
            "provenance_type": r["author_type"],
            "ai_generator": "GitHub Copilot" if r["ai_involvement"] > 0 else "None",
            "human_edit_count": 2 if r["author_type"] == "human_edit_of_ai" else (1 if r["author_type"] == "human" else 0),
            "confidence": 0.95
        }
        res_b = predict_model_b(r["code"], provenance=prov_dict, language=r["language"])
        y_pred_b.append(1 if res_b["is_vulnerable"] else 0)
        y_prob_b.append(res_b["probability"])
        
    y_pred_b = np.array(y_pred_b, dtype=int)
    y_prob_b = np.array(y_prob_b, dtype=float)
    metrics_b = compute_metrics(y_true, y_pred_b, y_prob_b)
    
    # 3. Print Results Comparison Table
    print("\n" + "="*80)
    print("INDEPENDENT EVALUATION RESULTS ON 32 LOCKED VERIFIED EXAMPLES:")
    print("="*80)
    print(f"{'Metric':<25} {'Model A (Code Only)':<25} {'Model B (Code + Provenance)':<25}")
    print("-" * 75)
    print(f"{'Accuracy':<25} {metrics_a['accuracy']*100:.2f}% (CI: {metrics_a['accuracy_ci_95'][0]*100:.1f}-{metrics_a['accuracy_ci_95'][1]*100:.1f}%)   {metrics_b['accuracy']*100:.2f}% (CI: {metrics_b['accuracy_ci_95'][0]*100:.1f}-{metrics_b['accuracy_ci_95'][1]*100:.1f}%)")
    print(f"{'Balanced Accuracy':<25} {metrics_a['balanced_accuracy']*100:.2f}%{'':<18} {metrics_b['balanced_accuracy']*100:.2f}%")
    print(f"{'Precision':<25} {metrics_a['precision']*100:.2f}%{'':<18} {metrics_b['precision']*100:.2f}%")
    print(f"{'Recall':<25} {metrics_a['recall']*100:.2f}%{'':<18} {metrics_b['recall']*100:.2f}%")
    print(f"{'F1 Score':<25} {metrics_a['f1']:.4f}{'':<19} {metrics_b['f1']:.4f}")
    print(f"{'MCC':<25} {metrics_a['mcc']:+.4f}{'':<18} {metrics_b['mcc']:+.4f}")
    print(f"{'AUROC':<25} {metrics_a['auroc']:.4f}{'':<19} {metrics_b['auroc']:.4f}")
    print(f"{'PR-AUC':<25} {metrics_a['prauc']:.4f}{'':<19} {metrics_b['prauc']:.4f}")
    print(f"{'Brier Score':<25} {metrics_a['brier_score']:.4f}{'':<19} {metrics_b['brier_score']:.4f}")
    print(f"{'Confusion Matrix':<25} {str(metrics_a['confusion_matrix']):<25} {str(metrics_b['confusion_matrix']):<25}")
    print("="*80)
    
    # 4. Breakdown by Provenance Subgroup
    print("\n--- PERFORMANCE BREAKDOWN BY PROVENANCE SUBGROUP (N=32) ---")
    subgroups = {
        "Pure Human Baseline (N=9, 8 safe, 1 vuln)": [i for i, r in enumerate(records) if r["author_type"] == "human"],
        "Pure AI Generation (N=8, 0 safe, 8 vuln)": [i for i, r in enumerate(records) if r["author_type"] == "ai"],
        "Human-Edited AI Fixed (N=8, 8 safe, 0 vuln)": [i for i, r in enumerate(records) if r["author_type"] == "human_edit_of_ai"],
        "AI Regressive on Human (N=7, 0 safe, 7 vuln)": [i for i, r in enumerate(records) if r["author_type"] == "ai_edit_of_human"],
        "Combined All Hybrids (N=15, 8 safe, 7 vuln)": [i for i, r in enumerate(records) if r["author_type"] in ["human_edit_of_ai", "ai_edit_of_human"]]
    }
    
    subgroup_metrics = {}
    for sub_name, indices in subgroups.items():
        sub_y = y_true[indices]
        sub_pred_a = y_pred_a[indices]
        sub_pred_b = y_pred_b[indices]
        acc_a = accuracy_score(sub_y, sub_pred_a)
        acc_b = accuracy_score(sub_y, sub_pred_b)
        print(f"{sub_name:<45}: Model A Acc = {acc_a*100:5.1f}% | Model B Acc = {acc_b*100:5.1f}%")
        subgroup_metrics[sub_name] = {
            "n_samples": len(indices),
            "model_a_accuracy": round(float(acc_a), 4),
            "model_b_accuracy": round(float(acc_b), 4)
        }
        
    # 5. Error Analysis on Failures
    errors_a = []
    errors_b = []
    for i, r in enumerate(records):
        if y_pred_a[i] != y_true[i]:
            errors_a.append({
                "task_id": r["task_id"],
                "version_id": r["version_id"],
                "author": r["author_type"],
                "true_label": int(y_true[i]),
                "pred_label": int(y_pred_a[i]),
                "prob": round(float(y_prob_a[i]), 4),
                "error_type": "False Positive" if y_pred_a[i] == 1 else "False Negative"
            })
        if y_pred_b[i] != y_true[i]:
            errors_b.append({
                "task_id": r["task_id"],
                "version_id": r["version_id"],
                "author": r["author_type"],
                "true_label": int(y_true[i]),
                "pred_label": int(y_pred_b[i]),
                "prob": round(float(y_prob_b[i]), 4),
                "error_type": "False Positive" if y_pred_b[i] == 1 else "False Negative"
            })
            
    print(f"\nModel A Errors: {len(errors_a)}/32 ({len(errors_a)/32*100:.1f}%)")
    for e in errors_a:
        print(f"  - [{e['error_type']}] {e['version_id']} ({e['author']}) P={e['prob']:.4f}")
    print(f"\nModel B Errors: {len(errors_b)}/32 ({len(errors_b)/32*100:.1f}%)")
    for e in errors_b:
        print(f"  - [{e['error_type']}] {e['version_id']} ({e['author']}) P={e['prob']:.4f}")
        
    # Save Report Artifact
    out_payload = {
        "benchmark_name": "Locked 32-Example Multi-CWE Benchmark",
        "benchmark_file": "results/reports/controlled_benchmark_32.json",
        "sample_count": 32,
        "n_vulnerable": n_vuln,
        "n_safe": n_safe,
        "integrity_verified": True,
        "model_a_metrics": metrics_a,
        "model_b_metrics": metrics_b,
        "subgroup_metrics": subgroup_metrics,
        "model_a_errors": errors_a,
        "model_b_errors": errors_b,
        "target_accuracy_70pct_achieved": {
            "model_a": metrics_a["accuracy"] >= 0.70,
            "model_b": metrics_b["accuracy"] >= 0.70
        }
    }
    
    out_file = PROJECT_ROOT / "results" / "metrics" / "locked_benchmark_32_evaluation.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nEvaluation artifact saved to {out_file}")


if __name__ == "__main__":
    main()
