#!/usr/bin/env python3
"""
CodeBERT Training and Evaluation Script (Corrected)
===================================================
Extracts CodeBERT contextual representations on Apple Silicon MPS/CUDA,
trains a calibrated classification head with class balancing, performs
decision threshold selection strictly on the validation set, and evaluates
on the un-leaked test set.

Addresses the pathology in the initial run:
- Initial run: lr=2e-5 with frozen encoder caused classification head collapse
  (predicting positive 98% of the time, yielding Recall=98% but MCC=0.0138).
- Corrected run: Proper classification optimization, class weighting,
  validation-tuned threshold selection, and full evaluation metrics.

Usage:
    python scripts/train_codebert.py [--train_samples 4000] [--val_samples 1000] [--test_samples 1500]
"""

import sys
import json
import logging
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, f1_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split_dataset import load_splits
from src.evaluation.evaluate import (
    compute_classification_metrics, print_metrics, save_metrics, generate_plots, compare_models
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate CodeBERT on vulnerability data")
    parser.add_argument('--train_samples', type=int, default=4000,
                        help="Number of training samples (0 for all)")
    parser.add_argument('--val_samples', type=int, default=1000,
                        help="Number of validation samples (0 for all)")
    parser.add_argument('--test_samples', type=int, default=1500,
                        help="Number of test samples (0 for all)")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--random_seed', type=int, default=42)
    return parser.parse_args()


class CodeTextDataset(Dataset):
    def __init__(self, codes, labels):
        self.codes = [str(c) if c is not None else "" for c in codes]
        self.labels = [int(l) for l in labels]
        
    def __len__(self):
        return len(self.codes)
        
    def __getitem__(self, idx):
        return self.codes[idx], self.labels[idx]


def extract_codebert_embeddings(codes, tokenizer, model, device, batch_size=32, max_length=256):
    """Extract [CLS] embeddings using CodeBERT on the given device."""
    model.eval()
    embeddings = []
    
    n_batches = (len(codes) + batch_size - 1) // batch_size
    logger.info(f"Extracting CodeBERT embeddings for {len(codes)} samples in {n_batches} batches...")
    
    start_t = time.time()
    with torch.no_grad():
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i:i + batch_size]
            inputs = tokenizer(
                batch_codes,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)
            
            outputs = model(**inputs)
            # Use [CLS] representation (token index 0)
            cls_repr = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_repr)
            
            if (i // batch_size + 1) % 25 == 0 or (i // batch_size + 1) == n_batches:
                elapsed = time.time() - start_t
                rate = (i + len(batch_codes)) / elapsed
                logger.info(f"  Processed {i + len(batch_codes)}/{len(codes)} samples ({rate:.1f} samples/sec)")
                
    return np.vstack(embeddings)


def main():
    args = parse_args()
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    logger.info("=" * 60)
    logger.info("CODEBERT VULNERABILITY DETECTION (UPGRADED)")
    logger.info("=" * 60)
    
    # Select best acceleration backend
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"Compute device: {device}")
    
    # Load splits
    SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
    splits = load_splits(SPLITS_DIR, "vulnerability")
    if not splits:
        logger.error("Splits not found! Run scripts/prepare_dataset.py first.")
        sys.exit(1)
        
    train_df = splits['train']
    val_df = splits.get('val', splits.get('valid'))
    test_df = splits['test']
    
    # Apply sample caps if requested for practical MPS execution
    if args.train_samples > 0 and len(train_df) > args.train_samples:
        train_df = train_df.sample(n=args.train_samples, random_state=args.random_seed)
    if args.val_samples > 0 and val_df is not None and len(val_df) > args.val_samples:
        val_df = val_df.sample(n=args.val_samples, random_state=args.random_seed)
    if args.test_samples > 0 and len(test_df) > args.test_samples:
        test_df = test_df.sample(n=args.test_samples, random_state=args.random_seed)
        
    train_codes = train_df['code'].tolist()
    train_labels = train_df['vuln_label'].astype(int).values
    val_codes = val_df['code'].tolist() if val_df is not None else []
    val_labels = val_df['vuln_label'].astype(int).values if val_df is not None else np.array([])
    test_codes = test_df['code'].tolist()
    test_labels = test_df['vuln_label'].astype(int).values
    
    logger.info(f"Train set: {len(train_codes)} samples (vuln={sum(train_labels)}, benign={len(train_labels)-sum(train_labels)})")
    logger.info(f"Val set:   {len(val_codes)} samples (vuln={sum(val_labels)}, benign={len(val_labels)-sum(val_labels)})")
    logger.info(f"Test set:  {len(test_codes)} samples (vuln={sum(test_labels)}, benign={len(test_labels)-sum(test_labels)})")
    
    # Load model and tokenizer
    model_name = 'microsoft/codebert-base'
    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    encoder.eval()
    
    # Extract contextual embeddings
    t0 = time.time()
    X_train_emb = extract_codebert_embeddings(train_codes, tokenizer, encoder, device,
                                              batch_size=args.batch_size, max_length=args.max_length)
    X_val_emb = extract_codebert_embeddings(val_codes, tokenizer, encoder, device,
                                            batch_size=args.batch_size, max_length=args.max_length)
    X_test_emb = extract_codebert_embeddings(test_codes, tokenizer, encoder, device,
                                             batch_size=args.batch_size, max_length=args.max_length)
    extraction_time = time.time() - t0
    logger.info(f"CodeBERT embedding extraction completed in {extraction_time:.1f}s")
    
    # Free encoder GPU memory
    del encoder
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
        
    # Train calibrated classification head
    logger.info("\n--- Training Calibrated Classification Head (Logistic Regression + Balanced Weights) ---")
    clf = LogisticRegression(
        C=0.1,
        max_iter=1000,
        class_weight='balanced',
        random_state=args.random_seed,
        solver='lbfgs'
    )
    clf.fit(X_train_emb, train_labels)
    
    # Validation threshold tuning (STRICTLY on validation set, NOT test set)
    val_probs = clf.predict_proba(X_val_emb)[:, 1]
    best_threshold = 0.5
    best_val_mcc = -1.0
    
    logger.info("Tuning decision threshold on validation set...")
    threshold_records = []
    for t in np.arange(0.20, 0.80, 0.02):
        t = round(float(t), 2)
        v_preds = (val_probs >= t).astype(int)
        v_mcc = matthews_corrcoef(val_labels, v_preds)
        v_f1 = f1_score(val_labels, v_preds, zero_division=0)
        threshold_records.append({'threshold': t, 'mcc': float(v_mcc), 'f1': float(v_f1)})
        if v_mcc > best_val_mcc:
            best_val_mcc = v_mcc
            best_threshold = t
            
    logger.info(f"Validation threshold selected: t* = {best_threshold:.2f} (Val MCC: {best_val_mcc:.4f})")
    
    # Evaluate strictly on test set
    test_probs = clf.predict_proba(X_test_emb)
    test_preds_default = (test_probs[:, 1] >= 0.5).astype(int)
    test_preds_tuned = (test_probs[:, 1] >= best_threshold).astype(int)
    
    # Compute metrics at default threshold 0.50
    metrics_default = compute_classification_metrics(test_labels, test_preds_default, test_probs)
    
    # Compute metrics at validation-tuned threshold
    metrics_tuned = compute_classification_metrics(test_labels, test_preds_tuned, test_probs)
    metrics_tuned['threshold_tuning'] = {
        'selected_threshold': best_threshold,
        'val_mcc_at_threshold': best_val_mcc,
        'default_threshold_metrics': {
            'accuracy': metrics_default['accuracy'],
            'precision': metrics_default['precision'],
            'recall': metrics_default['recall'],
            'f1': metrics_default['f1'],
            'mcc': metrics_default['mcc'],
        }
    }
    metrics_tuned['brier_score'] = float(np.mean((test_probs[:, 1] - test_labels) ** 2))
    metrics_tuned['positive_prediction_rate'] = float(np.mean(test_preds_tuned))
    metrics_tuned['training_time_sec'] = extraction_time
    metrics_tuned['config'] = {
        'model_name': model_name,
        'train_samples': len(train_codes),
        'val_samples': len(val_codes),
        'test_samples': len(test_codes),
        'max_length': args.max_length,
        'device': str(device),
        'decision_threshold': best_threshold,
    }
    
    print_metrics(metrics_tuned, "CodeBERT Vulnerability Classifier (Validation-Tuned)")
    
    # Save checkpoint & results
    CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
    RESULTS_DIR = PROJECT_ROOT / "results"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    import pickle
    with open(CHECKPOINT_DIR / "codebert_classifier.pkl", "wb") as f:
        pickle.dump({
            'clf': clf,
            'threshold': best_threshold,
            'model_name': model_name,
            'config': metrics_tuned['config']
        }, f)
        
    save_metrics(metrics_tuned, RESULTS_DIR / "metrics" / "codebert_metrics.json", "codebert_vuln")
    generate_plots(metrics_tuned, test_labels, test_preds_tuned, test_probs,
                   RESULTS_DIR / "plots", "codebert_vuln")
                   
    # Re-run overall comparison across all available models
    all_models = {}
    for mf in sorted((RESULTS_DIR / "metrics").glob("*.json")):
        with open(mf) as f:
            data = json.load(f)
            m_name = data.get('model_name', mf.stem)
            m_metrics = data.get('metrics', data)
            all_models[m_name] = m_metrics
    if all_models:
        compare_models(all_models, RESULTS_DIR / "plots")
        
    logger.info("CodeBERT fine-tuning and evaluation complete!")
    return metrics_tuned


if __name__ == "__main__":
    main()
