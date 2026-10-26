#!/usr/bin/env python3
"""
CodeBERT Fine-tuning Script
============================
Fine-tunes CodeBERT for vulnerability detection.
Supports MPS (Apple Silicon), CUDA, and CPU.

Usage:
    python scripts/train_codebert.py [--epochs 3] [--batch_size 8] [--subset 0]
"""

import sys
import json
import logging
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--subset', type=int, default=0,
                        help='If >0, use only this many samples per split for pilot')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze CodeBERT weights (faster, less memory)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    import torch
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    
    from src.data.split_dataset import load_splits
    from src.models.code_encoder import (
        CodeBERTClassifier, CodeDataset, get_device, train_codebert
    )
    from src.evaluation.evaluate import (
        compute_classification_metrics, print_metrics, save_metrics, generate_plots
    )
    import numpy as np
    
    logger.info("=" * 60)
    logger.info("CODEBERT FINE-TUNING")
    logger.info("=" * 60)
    
    device = get_device()
    logger.info(f"Device: {device}")
    
    # Load splits
    SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
    splits = load_splits(SPLITS_DIR, "vulnerability")
    
    if not splits:
        logger.error("No splits found! Run prepare_dataset.py first.")
        sys.exit(1)
    
    train_df = splits['train']
    val_df = splits.get('val', splits.get('valid'))
    test_df = splits['test']
    
    # Subset for pilot experiment
    if args.subset > 0:
        logger.info(f"Using subset of {args.subset} samples per split")
        train_df = train_df.head(args.subset)
        if val_df is not None:
            val_df = val_df.head(min(args.subset // 2, len(val_df)))
        test_df = test_df.head(min(args.subset // 2, len(test_df)))
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df) if val_df is not None else 0}, "
                f"Test: {len(test_df)}")
    
    # Load tokenizer
    model_name = 'microsoft/codebert-base'
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = CodeDataset(
        train_df['code'].tolist(),
        train_df['vuln_label'].astype(int).tolist(),
        tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = CodeDataset(
        val_df['code'].tolist(),
        val_df['vuln_label'].astype(int).tolist(),
        tokenizer,
        max_length=args.max_length
    ) if val_df is not None else None
    
    test_dataset = CodeDataset(
        test_df['code'].tolist(),
        test_df['vuln_label'].astype(int).tolist(),
        tokenizer,
        max_length=args.max_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Create model
    logger.info(f"Creating CodeBERT classifier (freeze={args.freeze_encoder})")
    model = CodeBERTClassifier(
        model_name=model_name,
        num_labels=2,
        dropout=0.1,
        freeze_encoder=args.freeze_encoder,
    )
    
    # Train
    CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
    start = time.time()
    
    history = train_codebert(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader if val_loader else train_loader,
        device=device,
        n_epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=CHECKPOINT_DIR,
        model_name='codebert_vuln',
    )
    
    elapsed = time.time() - start
    logger.info(f"Training completed in {elapsed:.1f}s")
    
    # Load best checkpoint if available
    best_ckpt = CHECKPOINT_DIR / "codebert_vuln_best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info(f"Loaded best checkpoint from {best_ckpt} (val_acc={ckpt.get('val_acc', 0):.4f})")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    metrics['training_time_sec'] = elapsed
    metrics['config'] = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'max_length': args.max_length,
        'freeze_encoder': args.freeze_encoder,
        'subset': args.subset,
        'device': str(device),
    }
    
    print_metrics(metrics, "CodeBERT Vulnerability Classifier")
    
    # Save
    RESULTS_DIR = PROJECT_ROOT / "results"
    save_metrics(metrics, RESULTS_DIR / "metrics" / "codebert_metrics.json", "codebert_vuln")
    generate_plots(metrics, y_true, y_pred, y_prob, RESULTS_DIR / "plots", "codebert_vuln")
    
    # Save training history
    with open(RESULTS_DIR / "reports" / "codebert_training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Re-run overall comparison
    from src.evaluation.evaluate import compare_models
    all_models = {}
    for mf in (RESULTS_DIR / "metrics").glob("*.json"):
        with open(mf) as f:
            data = json.load(f)
            m_name = data.get('model_name', mf.stem)
            m_metrics = data.get('metrics', data)
            all_models[m_name] = m_metrics
    if all_models:
        compare_models(all_models, RESULTS_DIR / "plots")
    
    logger.info("CodeBERT fine-tuning and model comparison complete!")


if __name__ == "__main__":
    main()
