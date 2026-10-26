#!/usr/bin/env python3
"""
Controlled Editing Benchmark Runner (Multi-CWE Suite)
=====================================================
Executes the 8-task, 32-version controlled code evolution benchmark.
Evaluates:
- Static analysis findings (Flawfinder + Pattern heuristics)
- CodeBERT and N-gram ML predictions
- Version transitions (v1 -> v2 -> v3 -> v4)
- Review Prioritization ranking (Code-only vs Provenance-Aware)

Usage:
    python scripts/run_controlled_experiment.py
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.controlled_editing import create_multi_cwe_benchmark, analyze_experiment_security
from src.security.static_analysis import analyze_code
from src.models.baseline import VulnerabilityBaseline

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("RUNNING MULTI-CWE CONTROLLED EDITING BENCHMARK (32 VERSIONS)")
    logger.info("=" * 60)
    
    # 1. Create 8-task benchmark suite
    benchmarks = create_multi_cwe_benchmark()
    logger.info(f"Loaded {len(benchmarks)} tasks with 4 versions each (Total: {len(benchmarks)*4} code units)")
    
    # 2. Load models for inference
    char_model_path = PROJECT_ROOT / "models" / "checkpoints" / "baseline_char.pkl"
    word_model_path = PROJECT_ROOT / "models" / "checkpoints" / "baseline_word.pkl"
    codebert_path = PROJECT_ROOT / "models" / "checkpoints" / "codebert_classifier.pkl"
    
    char_model = VulnerabilityBaseline.load(char_model_path) if char_model_path.exists() else None
    word_model = VulnerabilityBaseline.load(word_model_path) if word_model_path.exists() else None
    
    codebert_clf = None
    codebert_tokenizer = None
    codebert_encoder = None
    device = 'cpu'
    
    if codebert_path.exists():
        import pickle
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        with open(codebert_path, 'rb') as f:
            cb_data = pickle.load(f)
            codebert_clf = cb_data['clf']
            
        device = 'mps' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 'cpu'
        codebert_tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        codebert_encoder = AutoModel.from_pretrained('microsoft/codebert-base').to(device)
        codebert_encoder.eval()
        logger.info(f"Loaded CodeBERT on {device} for benchmark inference")
    
    # 3. Analyze each task and version
    all_version_records = []
    task_reports = []
    
    for exp in benchmarks:
        logger.info(f"Processing task: {exp.task_id} ({exp.metadata.get('category', '')})")
        analysis = analyze_experiment_security(exp)
        task_reports.append(analysis)
        
        for v in exp.versions:
            # Model inference
            ml_preds = {}
            code_arr = np.array([v.code])
            
            if char_model:
                prob = float(char_model.predict_proba(code_arr)[0, 1])
                pred = int(char_model.predict(code_arr)[0])
                ml_preds['char_ngram'] = {'vuln_prob': prob, 'pred_label': pred}
                
            if word_model:
                prob = float(word_model.predict_proba(code_arr)[0, 1])
                pred = int(word_model.predict(code_arr)[0])
                ml_preds['word_ngram'] = {'vuln_prob': prob, 'pred_label': pred}
                
            cb_prob = 0.5
            if codebert_clf is not None and codebert_encoder is not None:
                import torch
                with torch.no_grad():
                    inputs = codebert_tokenizer([v.code], padding=True, truncation=True,
                                                max_length=256, return_tensors='pt').to(device)
                    emb = codebert_encoder(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
                    cb_prob = float(codebert_clf.predict_proba(emb)[0, 1])
                    cb_pred = int(cb_prob >= 0.5)
                    ml_preds['codebert'] = {'vuln_prob': cb_prob, 'pred_label': cb_pred}
            
            findings = v.security_findings.get('findings', []) if v.security_findings else []
            n_findings = len(findings)
            risk_score = v.security_findings.get('risk_score', 0) if v.security_findings else 0
            cwes = v.security_findings.get('cwes', []) if v.security_findings else []
            
            # Provenance-aware adjusted score
            # Combines code model probability with provenance risk modifier
            prov_adjusted_score = float(cb_prob * (0.6 + 0.8 * v.ai_involvement))
            prov_adjusted_score = min(max(prov_adjusted_score, 0.0), 1.0)
            
            record = {
                'task_id': exp.task_id,
                'category': exp.metadata.get('category', ''),
                'target_cwe': exp.metadata.get('cwe', ''),
                'version_id': v.version_id,
                'author_type': v.author_type,
                'ai_involvement': v.ai_involvement,
                'ground_truth_vuln': v.vuln_label,
                'n_findings': n_findings,
                'risk_score': risk_score,
                'detected_cwes': cwes,
                'char_vuln_prob': ml_preds.get('char_ngram', {}).get('vuln_prob', 0.5),
                'word_vuln_prob': ml_preds.get('word_ngram', {}).get('vuln_prob', 0.5),
                'codebert_vuln_prob': cb_prob,
                'prov_adjusted_score': prov_adjusted_score,
            }
            all_version_records.append(record)
            
    df = pd.DataFrame(all_version_records)
    logger.info(f"\nGenerated benchmark dataset with {len(df)} samples: "
                f"Vuln={df['ground_truth_vuln'].sum()}, Benign={len(df)-df['ground_truth_vuln'].sum()}")
                
    # 4. Compute Review Prioritization Metrics (Phase 8)
    logger.info("\n--- Computing Review Prioritization Metrics ---")
    
    # Sort by CodeBERT-only vs Provenance-Aware vs Static Analysis
    df_sorted_cb = df.sort_values(by='codebert_vuln_prob', ascending=False).reset_index(drop=True)
    df_sorted_prov = df.sort_values(by='prov_adjusted_score', ascending=False).reset_index(drop=True)
    df_sorted_static = df.sort_values(by='risk_score', ascending=False).reset_index(drop=True)
    
    total_vuln = df['ground_truth_vuln'].sum()
    k_values = [4, 8, 12, 16]  # 12.5%, 25%, 37.5%, 50%
    
    prioritization_results = {}
    for k in k_values:
        cb_top_k = df_sorted_cb.head(k)
        prov_top_k = df_sorted_prov.head(k)
        static_top_k = df_sorted_static.head(k)
        
        prioritization_results[f"top_{k}"] = {
            'k': k,
            'budget_pct': (k / len(df)) * 100,
            'code_only': {
                'precision': float(cb_top_k['ground_truth_vuln'].mean()),
                'recall': float(cb_top_k['ground_truth_vuln'].sum() / total_vuln),
                'vuln_found': int(cb_top_k['ground_truth_vuln'].sum()),
            },
            'provenance_aware': {
                'precision': float(prov_top_k['ground_truth_vuln'].mean()),
                'recall': float(prov_top_k['ground_truth_vuln'].sum() / total_vuln),
                'vuln_found': int(prov_top_k['ground_truth_vuln'].sum()),
            },
            'static_analysis': {
                'precision': float(static_top_k['ground_truth_vuln'].mean()),
                'recall': float(static_top_k['ground_truth_vuln'].sum() / total_vuln),
                'vuln_found': int(static_top_k['ground_truth_vuln'].sum()),
            }
        }
        
    for k_name, res in prioritization_results.items():
        k = res['k']
        pct = res['budget_pct']
        logger.info(f"Top {k} ({pct:.1f}% review budget):")
        logger.info(f"  Code-Only:         Precision={res['code_only']['precision']:.2f}, Recall={res['code_only']['recall']:.2f} ({res['code_only']['vuln_found']}/{total_vuln})")
        logger.info(f"  Provenance-Aware:  Precision={res['provenance_aware']['precision']:.2f}, Recall={res['provenance_aware']['recall']:.2f} ({res['provenance_aware']['vuln_found']}/{total_vuln})")
        logger.info(f"  Static Analysis:   Precision={res['static_analysis']['precision']:.2f}, Recall={res['static_analysis']['recall']:.2f} ({res['static_analysis']['vuln_found']}/{total_vuln})")

    # 5. Save benchmark results
    REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
    PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    full_output = {
        'n_tasks': len(benchmarks),
        'n_versions': len(df),
        'task_summaries': task_reports,
        'records': all_version_records,
        'prioritization_metrics': prioritization_results,
    }
    
    with open(REPORTS_DIR / "controlled_benchmark_32.json", 'w') as f:
        json.dump(full_output, f, indent=2, default=str)
    logger.info(f"Full benchmark results saved to {REPORTS_DIR / 'controlled_benchmark_32.json'}")
    
    # 6. Generate Figures
    # Figure 1: Code Evolution Lineage across Stages (v1 -> v2 -> v3 -> v4)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    stage_groups = df.groupby('author_type').agg({
        'n_findings': 'mean',
        'risk_score': 'mean',
        'codebert_vuln_prob': 'mean',
        'ground_truth_vuln': 'mean'
    }).reindex(['human', 'ai', 'human_edit_of_ai', 'ai_edit_of_human'])
    
    stages = ['v1: Human Secure', 'v2: AI Naive', 'v3: Human Hybrid', 'v4: AI Regressive']
    x = np.arange(len(stages))
    width = 0.35
    
    axes[0].bar(x - width/2, stage_groups['n_findings'], width, label='Static Findings (Avg)', color='#ff6b6b')
    axes[0].bar(x + width/2, stage_groups['risk_score'], width, label='Risk Score (Avg)', color='#feca57')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stages, rotation=15, ha='right')
    axes[0].set_ylabel('Score / Count')
    axes[0].set_title('Security Findings Across Evolution Stages')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].plot(x, stage_groups['ground_truth_vuln'] * 100, 'o-', color='#ee5253', linewidth=2.5, label='Actual Vulnerable (%)')
    axes[1].plot(x, stage_groups['codebert_vuln_prob'] * 100, 's--', color='#54a0ff', linewidth=2, label='CodeBERT Prob (%)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stages, rotation=15, ha='right')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_title('Ground Truth Vulnerability vs CodeBERT Probability')
    axes[1].set_ylim(-5, 110)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "controlled_benchmark_lineage.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Review Prioritization (Recall @ Budget)
    fig, ax = plt.subplots(figsize=(8, 5))
    budgets = [res['budget_pct'] for res in prioritization_results.values()]
    cb_recalls = [res['code_only']['recall'] * 100 for res in prioritization_results.values()]
    prov_recalls = [res['provenance_aware']['recall'] * 100 for res in prioritization_results.values()]
    static_recalls = [res['static_analysis']['recall'] * 100 for res in prioritization_results.values()]
    
    ax.plot(budgets, prov_recalls, 'o-', color='#10ac84', linewidth=2.5, label='Provenance-Aware Prioritization')
    ax.plot(budgets, static_recalls, '^--', color='#ee5253', linewidth=2, label='Static Analysis Prioritization')
    ax.plot(budgets, cb_recalls, 's:', color='#54a0ff', linewidth=2, label='CodeBERT Code-Only Prioritization')
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Random Triage (Baseline)')
    
    ax.set_xlabel('Code Review Budget (% of Total Pull Requests Inspected)')
    ax.set_ylabel('Vulnerabilities Discovered (Recall %)')
    ax.set_title('Code Review Prioritization: Provenance-Aware vs Code-Only')
    ax.set_xlim(5, 55)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "review_prioritization_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("Controlled editing experiment and review prioritization complete!")


if __name__ == "__main__":
    main()
