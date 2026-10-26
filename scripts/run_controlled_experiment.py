#!/usr/bin/env python3
"""
Controlled Editing Experiment Runner
====================================
Runs the controlled editing pilot experiment, analyzes security across code versions,
evaluates ML baseline predictions on each version, and generates reports and plots.
"""

import sys
import json
import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    from src.experiments.controlled_editing import create_pilot_experiment, analyze_experiment_security
    from src.models.baseline import VulnerabilityBaseline
    
    logger.info("=" * 60)
    logger.info("RUNNING CONTROLLED EDITING EXPERIMENT")
    logger.info("=" * 60)
    
    # 1. Create pilot experiment
    experiment = create_pilot_experiment()
    
    # 2. Run security analysis
    logger.info("Analyzing security properties across code versions...")
    analysis = analyze_experiment_security(experiment)
    
    # 3. Load baseline model and evaluate on each version
    char_model_path = PROJECT_ROOT / "models" / "checkpoints" / "baseline_char.pkl"
    word_model_path = PROJECT_ROOT / "models" / "checkpoints" / "baseline_word.pkl"
    
    char_model = VulnerabilityBaseline.load(char_model_path) if char_model_path.exists() else None
    word_model = VulnerabilityBaseline.load(word_model_path) if word_model_path.exists() else None
    
    for v in experiment.versions:
        ml_preds = {}
        if char_model:
            prob = char_model.predict_proba(np.array([v.code]))[0, 1]
            pred = int(char_model.predict(np.array([v.code]))[0])
            ml_preds['char_ngram'] = {'vuln_prob': float(prob), 'pred_label': pred}
        if word_model:
            prob = word_model.predict_proba(np.array([v.code]))[0, 1]
            pred = int(word_model.predict(np.array([v.code]))[0])
            ml_preds['word_ngram'] = {'vuln_prob': float(prob), 'pred_label': pred}
        v.metadata = getattr(v, 'metadata', {})
        v.metadata['ml_predictions'] = ml_preds
    
    # Save experiment
    EXP_DIR = PROJECT_ROOT / "results" / "reports"
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    exp_path = EXP_DIR / "controlled_experiment_pilot.json"
    experiment.save(exp_path)
    
    # Save full analysis
    full_report = {
        'experiment': experiment.to_dict(),
        'security_analysis': analysis,
    }
    with open(EXP_DIR / "controlled_experiment_analysis.json", 'w') as f:
        json.dump(full_report, f, indent=2, default=str)
    
    logger.info(f"Experiment analysis saved to {EXP_DIR / 'controlled_experiment_analysis.json'}")
    
    # 4. Generate comparison plots
    PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    version_names = [v.version_id.replace("pilot_001_", "") for v in experiment.versions]
    ai_involvements = [v.ai_involvement * 100 for v in experiment.versions]
    findings_counts = [len(v.security_findings.get('findings', [])) if v.security_findings else 0 for v in experiment.versions]
    risk_scores = [v.security_findings.get('risk_score', 0) if v.security_findings else 0 for v in experiment.versions]
    
    x = np.arange(len(version_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, findings_counts, width, label='Static Security Findings', color='#ff6b6b')
    bars2 = ax1.bar(x + width/2, risk_scores, width, label='Static Risk Score', color='#feca57')
    ax1.set_xlabel('Code Version')
    ax1.set_ylabel('Count / Score')
    ax1.set_title('Security Findings and Risk Score by Version')
    ax1.set_xticks(x)
    ax1.set_xticklabels(version_names, rotation=25, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ML predicted probability vs AI involvement
    char_probs = []
    for v in experiment.versions:
        ml = v.metadata.get('ml_predictions', {})
        char_probs.append(ml.get('char_ngram', {}).get('vuln_prob', 0.5) * 100)
    
    ax2.plot(x, ai_involvements, 'o--', color='#54a0ff', linewidth=2, markersize=8, label='AI Involvement (%)')
    ax2.plot(x, char_probs, 's-', color='#ee5253', linewidth=2, markersize=8, label='ML Vulnerability Prob (%)')
    ax2.set_xlabel('Code Version')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('AI Involvement vs ML Vulnerability Prediction')
    ax2.set_xticks(x)
    ax2.set_xticklabels(version_names, rotation=25, ha='right')
    ax2.set_ylim(-5, 105)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / "controlled_experiment_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plot saved to {plot_path}")
    logger.info("Controlled editing experiment completed successfully!")

if __name__ == "__main__":
    main()
