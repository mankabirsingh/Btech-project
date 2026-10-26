"""
Evaluation Module for Provenance-Aware Security Risk Assessment
================================================================
Computes classification metrics, generates plots, and saves reports.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    label_names: List[str] = ['benign', 'vulnerable'],
) -> Dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        label_names: Names for each class
    
    Returns:
        Dictionary with all metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score,
        precision_recall_curve, average_precision_score, matthews_corrcoef,
    )
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['f1_micro'] = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
    metrics['mcc'] = float(matthews_corrcoef(y_true, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=label_names, 
                                     output_dict=True, zero_division=0)
    metrics['classification_report'] = report
    
    # AUROC
    if y_prob is not None:
        try:
            if y_prob.ndim == 2:
                auroc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auroc = roc_auc_score(y_true, y_prob)
            metrics['auroc'] = float(auroc)
        except ValueError as e:
            logger.warning(f"Could not compute AUROC: {e}")
            metrics['auroc'] = None
        
        try:
            if y_prob.ndim == 2:
                ap = average_precision_score(y_true, y_prob[:, 1])
            else:
                ap = average_precision_score(y_true, y_prob)
            metrics['average_precision'] = float(ap)
        except ValueError:
            metrics['average_precision'] = None
    
    # Sample counts
    metrics['n_samples'] = int(len(y_true))
    metrics['n_positive'] = int(sum(y_true))
    metrics['n_negative'] = int(len(y_true) - sum(y_true))
    metrics['positive_rate'] = float(sum(y_true) / len(y_true)) if len(y_true) > 0 else 0
    
    return metrics


def print_metrics(metrics: Dict, model_name: str = "Model"):
    """Pretty-print classification metrics."""
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"Samples: {metrics['n_samples']} "
          f"(positive={metrics['n_positive']}, negative={metrics['n_negative']})")
    print(f"Positive rate: {metrics['positive_rate']:.2%}")
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"F1 Macro:  {metrics['f1_macro']:.4f}")
    print(f"MCC:       {metrics['mcc']:.4f}")
    
    if metrics.get('auroc') is not None:
        print(f"AUROC:     {metrics['auroc']:.4f}")
    if metrics.get('average_precision') is not None:
        print(f"Avg Prec:  {metrics['average_precision']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"  {'':>12} Pred Benign  Pred Vuln")
    print(f"  True Benign   {cm[0][0]:>8d}   {cm[0][1]:>8d}")
    print(f"  True Vuln     {cm[1][0]:>8d}   {cm[1][1]:>8d}")
    print(f"{'='*60}\n")


def save_metrics(metrics: Dict, path: Path, model_name: str = "model"):
    """Save metrics to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        'model_name': model_name,
        'metrics': metrics,
    }
    
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    logger.info(f"Metrics saved to {path}")


def generate_plots(
    metrics: Dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    output_dir: Path = None,
    model_name: str = "model",
):
    """
    Generate and save evaluation plots.
    
    Generates:
    - Confusion matrix heatmap
    - ROC curve (if probabilities available)
    - Precision-Recall curve
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, precision_recall_curve
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    cm = np.array(metrics['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Vulnerable'],
                yticklabels=['Benign', 'Vulnerable'],
                ax=ax)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    
    if output_dir:
        cm_path = output_dir / f"confusion_matrix_{model_name}.png"
        fig.savefig(cm_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {cm_path}")
    plt.close()
    
    # 2. ROC Curve
    if y_prob is not None:
        try:
            if y_prob.ndim == 2:
                probs = y_prob[:, 1]
            else:
                probs = y_prob
            
            fpr, tpr, _ = roc_curve(y_true, probs)
            auroc = metrics.get('auroc', 0)
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'{model_name} (AUROC={auroc:.4f})')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {model_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if output_dir:
                roc_path = output_dir / f"roc_curve_{model_name}.png"
                fig.savefig(roc_path, dpi=150, bbox_inches='tight')
                logger.info(f"ROC curve saved to {roc_path}")
            plt.close()
            
            # 3. Precision-Recall Curve
            precision_arr, recall_arr, _ = precision_recall_curve(y_true, probs)
            ap = metrics.get('average_precision', 0)
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.plot(recall_arr, precision_arr, 'r-', linewidth=2,
                    label=f'{model_name} (AP={ap:.4f})')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'Precision-Recall Curve - {model_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if output_dir:
                pr_path = output_dir / f"pr_curve_{model_name}.png"
                fig.savefig(pr_path, dpi=150, bbox_inches='tight')
                logger.info(f"PR curve saved to {pr_path}")
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not generate ROC/PR plots: {e}")


def compare_models(metrics_dict: Dict[str, Dict], output_dir: Optional[Path] = None):
    """
    Create comparison table and chart for multiple models.
    
    Args:
        metrics_dict: {model_name: metrics_dict}
        output_dir: Where to save comparison files
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    rows = []
    for name, m in metrics_dict.items():
        row = {
            'Model': name,
            'Accuracy': m.get('accuracy', 0),
            'Precision': m.get('precision', 0),
            'Recall': m.get('recall', 0),
            'F1': m.get('f1', 0),
            'F1 Macro': m.get('f1_macro', 0),
            'MCC': m.get('mcc', 0),
            'AUROC': m.get('auroc', '-'),
            'N Samples': m.get('n_samples', 0),
        }
        rows.append(row)
    
    comparison_df = pd.DataFrame(rows)
    print("\nMODEL COMPARISON:")
    print(comparison_df.to_string(index=False))
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save table
        comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
        
        # Bar chart
        metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'F1 Macro', 'MCC']
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metric_cols))
        width = 0.8 / len(metrics_dict)
        
        for i, (name, m) in enumerate(metrics_dict.items()):
            values = [m.get(mc.lower().replace(' ', '_'), 0) for mc in metric_cols]
            ax.bar(x + i * width, values, width, label=name)
        
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x + width * (len(metrics_dict) - 1) / 2)
        ax.set_xticklabels(metric_cols, rotation=30)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        fig.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches='tight')
        logger.info(f"Comparison chart saved to {output_dir / 'model_comparison.png'}")
        plt.close()
    
    return comparison_df
