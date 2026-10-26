"""
Dataset Splitting for Provenance-Aware Security Risk Assessment
================================================================
Handles train/validation/test splitting with leakage prevention.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_vulnerability_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.10,
    random_seed: int = 42,
    stratify_col: str = 'vuln_label',
    group_col: Optional[str] = 'project',
) -> Dict[str, pd.DataFrame]:
    """
    Split vulnerability dataset into train/val/test with leakage prevention.
    
    Strategy:
    1. If group_col (e.g., 'project') exists with multiple values:
       Split by project to prevent same-project samples in train and test.
    2. Otherwise: stratified random split.
    
    Args:
        df: Processed DataFrame with vuln_label
        test_size: Fraction for test
        val_size: Fraction for validation
        random_seed: Fixed seed for reproducibility
        stratify_col: Column to stratify by
        group_col: Column for group-based splitting
    
    Returns:
        Dict with 'train', 'val', 'test' DataFrames
    """
    np.random.seed(random_seed)
    
    # Remove samples without labels
    df = df[df[stratify_col].notna()].reset_index(drop=True)
    
    if len(df) == 0:
        raise ValueError("No samples with valid labels for splitting")
    
    # Try project-based splitting
    if group_col and group_col in df.columns and df[group_col].notna().sum() > 0:
        splits = _group_based_split(df, group_col, test_size, val_size, random_seed, stratify_col)
        if splits is not None:
            return splits
    
    # Fallback: stratified random split
    logger.info("Using stratified random split")
    return _stratified_split(df, test_size, val_size, random_seed, stratify_col)


def _group_based_split(
    df: pd.DataFrame,
    group_col: str,
    test_size: float,
    val_size: float,
    random_seed: int,
    stratify_col: str,
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Split by groups (e.g., projects) to prevent leakage.
    Ensures no group appears in both train and test.
    """
    valid_groups = df[df[group_col].notna()][group_col].unique()
    
    if len(valid_groups) < 5:
        logger.warning(f"Only {len(valid_groups)} groups, falling back to stratified split")
        return None
    
    np.random.shuffle(valid_groups)
    
    n_test = max(1, int(len(valid_groups) * test_size))
    n_val = max(1, int(len(valid_groups) * val_size))
    
    test_groups = set(valid_groups[:n_test])
    val_groups = set(valid_groups[n_test:n_test + n_val])
    train_groups = set(valid_groups[n_test + n_val:])
    
    test_df = df[df[group_col].isin(test_groups)].reset_index(drop=True)
    val_df = df[df[group_col].isin(val_groups)].reset_index(drop=True)
    train_df = df[df[group_col].isin(train_groups)].reset_index(drop=True)
    
    # Include samples with no group in training
    no_group = df[df[group_col].isna()]
    if len(no_group) > 0:
        train_df = pd.concat([train_df, no_group], ignore_index=True)
    
    if train_df[stratify_col].nunique() < 2 or test_df[stratify_col].nunique() < 2:
        logger.warning("Group-based split resulted in single-class train or test; falling back to stratified split")
        return None
    
    logger.info(f"Group-based split: train={len(train_df)} ({len(train_groups)} groups), "
                f"val={len(val_df)} ({len(val_groups)} groups), "
                f"test={len(test_df)} ({len(test_groups)} groups)")
    
    return {'train': train_df, 'val': val_df, 'test': test_df}


def _stratified_split(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_seed: int,
    stratify_col: str,
) -> Dict[str, pd.DataFrame]:
    """Standard stratified split."""
    # First split: train+val vs test
    stratify = df[stratify_col].values if df[stratify_col].nunique() > 1 else None
    
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_seed,
        stratify=stratify
    )
    
    # Second split: train vs val
    relative_val_size = val_size / (1 - test_size)
    stratify_tv = train_val_df[stratify_col].values if train_val_df[stratify_col].nunique() > 1 else None
    
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_size, random_state=random_seed,
        stratify=stratify_tv
    )
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    logger.info(f"Stratified split: train={len(train_df)}, "
                f"val={len(val_df)}, test={len(test_df)}")
    
    return {'train': train_df, 'val': val_df, 'test': test_df}


def save_splits(splits: Dict[str, pd.DataFrame], output_dir: Path, name: str):
    """Save train/val/test splits to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, df in splits.items():
        path = output_dir / f"{name}_{split_name}.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"Saved {split_name}: {len(df)} samples -> {path}")
    
    # Save split summary
    summary = {}
    for split_name, df in splits.items():
        summary[split_name] = {
            'n_samples': len(df),
            'label_dist': df['vuln_label'].value_counts().to_dict() if 'vuln_label' in df.columns else {},
        }
    
    import json
    summary_path = output_dir / f"{name}_split_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Split summary saved to {summary_path}")


def load_splits(splits_dir: Path, name: str) -> Dict[str, pd.DataFrame]:
    """Load saved splits from disk."""
    splits_dir = Path(splits_dir)
    splits = {}
    for split_name in ['train', 'val', 'test']:
        path = splits_dir / f"{name}_{split_name}.parquet"
        if path.exists():
            splits[split_name] = pd.read_parquet(path)
            logger.info(f"Loaded {split_name}: {len(splits[split_name])} samples")
    return splits


def verify_no_leakage(splits: Dict[str, pd.DataFrame], code_col: str = 'code') -> bool:
    """
    Verify there's no data leakage between splits.
    Checks for exact code duplicates across train/val/test.
    """
    if 'train' not in splits or 'test' not in splits:
        logger.warning("Cannot verify leakage: missing train or test split")
        return True
    
    train_codes = set(splits['train'][code_col].values)
    test_codes = set(splits['test'][code_col].values)
    
    overlap = train_codes & test_codes
    if overlap:
        logger.error(f"DATA LEAKAGE DETECTED: {len(overlap)} identical samples in train and test!")
        return False
    
    if 'val' in splits:
        val_codes = set(splits['val'][code_col].values)
        overlap_tv = train_codes & val_codes
        if overlap_tv:
            logger.error(f"DATA LEAKAGE: {len(overlap_tv)} identical samples in train and val!")
            return False
        overlap_vt = val_codes & test_codes
        if overlap_vt:
            logger.error(f"DATA LEAKAGE: {len(overlap_vt)} identical samples in val and test!")
            return False
    
    logger.info("No data leakage detected between splits ✓")
    return True
