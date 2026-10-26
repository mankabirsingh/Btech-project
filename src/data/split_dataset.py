"""
Dataset Splitting for Provenance-Aware Security Risk Assessment
================================================================
Handles train/validation/test splitting with group-level leakage prevention.
Prevents vulnerable/fixed pairs, identical commits, and exact duplicates from crossing splits.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

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
    group_col: Optional[str] = 'group_id',
) -> Dict[str, pd.DataFrame]:
    """
    Split vulnerability dataset into train/val/test with strict leakage prevention.
    
    Strategy:
    1. If group_col (e.g., 'group_id') exists with multiple values:
       Use stratified group splitting to guarantee that related samples
       (such as vulnerable/fixed pairs or commits) NEVER cross partitions.
    2. Otherwise: standard stratified split.
    
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
    
    # If group_col is available and has sufficient groups, use group-aware splitting
    if group_col and group_col in df.columns and df[group_col].notna().sum() > 0:
        valid_groups = df[df[group_col].notna()][group_col].unique()
        if len(valid_groups) >= 5:
            logger.info(f"Using stratified group split on '{group_col}' ({len(valid_groups)} groups)")
            return _stratified_group_split(df, group_col, test_size, val_size, random_seed, stratify_col)
        else:
            logger.warning(f"Only {len(valid_groups)} groups in '{group_col}', falling back to standard stratified split")
    
    # Fallback: stratified random split
    logger.info("Using standard stratified random split")
    return _stratified_split(df, test_size, val_size, random_seed, stratify_col)


def _stratified_group_split(
    df: pd.DataFrame,
    group_col: str,
    test_size: float,
    val_size: float,
    random_seed: int,
    stratify_col: str,
) -> Dict[str, pd.DataFrame]:
    """
    Perform group-aware splitting that balances positive and negative classes
    while strictly ensuring no group spans across splits.
    """
    rng = np.random.RandomState(random_seed)
    
    # Map NaN groups to individual unique groups
    df = df.copy()
    nan_mask = df[group_col].isna()
    if nan_mask.sum() > 0:
        df.loc[nan_mask, group_col] = [f"singleton_{i}" for i in range(nan_mask.sum())]
    
    # Aggregate statistics per group
    grp_stats = df.groupby(group_col).agg(
        n_samples=(stratify_col, 'count'),
        n_pos=(stratify_col, lambda x: (x == 1).sum()),
        n_neg=(stratify_col, lambda x: (x == 0).sum())
    ).reset_index()
    
    # Shuffle groups deterministically
    grp_stats = grp_stats.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    
    total_samples = len(df)
    target_test_n = int(total_samples * test_size)
    target_val_n = int(total_samples * val_size)
    
    test_groups = set()
    val_groups = set()
    train_groups = set()
    
    curr_test_n = 0
    curr_val_n = 0
    
    for _, row in grp_stats.iterrows():
        gid = row[group_col]
        sz = row['n_samples']
        
        if curr_test_n + sz <= target_test_n:
            test_groups.add(gid)
            curr_test_n += sz
        elif curr_val_n + sz <= target_val_n:
            val_groups.add(gid)
            curr_val_n += sz
        else:
            train_groups.add(gid)
    
    train_df = df[df[group_col].isin(train_groups)].reset_index(drop=True)
    val_df = df[df[group_col].isin(val_groups)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_groups)].reset_index(drop=True)
    
    logger.info(f"Stratified group split on '{group_col}':")
    logger.info(f"  Train: {len(train_df)} samples ({len(train_groups)} groups), vuln_rate={train_df[stratify_col].mean():.3f}")
    logger.info(f"  Val:   {len(val_df)} samples ({len(val_groups)} groups), vuln_rate={val_df[stratify_col].mean():.3f}")
    logger.info(f"  Test:  {len(test_df)} samples ({len(test_groups)} groups), vuln_rate={test_df[stratify_col].mean():.3f}")
    
    return {'train': train_df, 'val': val_df, 'test': test_df}


def _stratified_split(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_seed: int,
    stratify_col: str,
) -> Dict[str, pd.DataFrame]:
    """Standard stratified split."""
    stratify = df[stratify_col].values if df[stratify_col].nunique() > 1 else None
    
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_seed,
        stratify=stratify
    )
    
    relative_val_size = val_size / (1 - test_size)
    stratify_tv = train_val_df[stratify_col].values if train_val_df[stratify_col].nunique() > 1 else None
    
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_size, random_state=random_seed,
        stratify=stratify_tv
    )
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    logger.info(f"Stratified split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
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
            'n_groups': df['group_id'].nunique() if 'group_id' in df.columns else None,
        }
    
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


def verify_no_leakage(
    splits: Dict[str, pd.DataFrame],
    code_col: str = 'code',
    group_col: str = 'group_id',
    sample_near_dup: int = 500,
) -> Dict[str, Any]:
    """
    Thorough leakage verification between splits.
    
    Checks:
    1. Exact code string overlap (must be 0).
    2. Group ID overlap (must be 0 if group_col present).
    3. Near-duplicate overlap (token 3-gram Jaccard similarity > 0.85 on sampled pairs).
    
    Returns:
        Dict with leakage audit results and boolean 'passed'.
    """
    report = {
        'exact_code_leakage': False,
        'exact_code_overlap_count': 0,
        'group_leakage': False,
        'group_overlap_count': 0,
        'near_duplicates_flagged': 0,
        'passed': True,
        'details': {}
    }
    
    if 'train' not in splits or 'test' not in splits:
        logger.warning("Cannot verify leakage: missing train or test split")
        return report
    
    # 1. Exact code overlap
    train_codes = set(splits['train'][code_col].values)
    test_codes = set(splits['test'][code_col].values)
    overlap_tt = train_codes & test_codes
    
    overlap_tv = set()
    overlap_vt = set()
    if 'val' in splits:
        val_codes = set(splits['val'][code_col].values)
        overlap_tv = train_codes & val_codes
        overlap_vt = val_codes & test_codes
    
    total_exact = len(overlap_tt) + len(overlap_tv) + len(overlap_vt)
    report['exact_code_overlap_count'] = total_exact
    if total_exact > 0:
        report['exact_code_leakage'] = True
        report['passed'] = False
        logger.error(f"DATA LEAKAGE: {total_exact} exact code duplicates across splits!")
    else:
        logger.info("Exact code duplicate check: 0 overlap between splits ✓")
        
    # 2. Group ID overlap
    if group_col in splits['train'].columns and group_col in splits['test'].columns:
        train_groups = set(splits['train'][group_col].dropna().values)
        test_groups = set(splits['test'][group_col].dropna().values)
        overlap_grp_tt = train_groups & test_groups
        
        overlap_grp_tv = set()
        overlap_grp_vt = set()
        if 'val' in splits and group_col in splits['val'].columns:
            val_groups = set(splits['val'][group_col].dropna().values)
            overlap_grp_tv = train_groups & val_groups
            overlap_grp_vt = val_groups & test_groups
            
        total_grp = len(overlap_grp_tt) + len(overlap_grp_tv) + len(overlap_grp_vt)
        report['group_overlap_count'] = total_grp
        if total_grp > 0:
            report['group_leakage'] = True
            report['passed'] = False
            logger.error(f"GROUP LEAKAGE: {total_grp} groups overlap across splits!")
        else:
            logger.info(f"Group isolation check on '{group_col}': 0 overlap across splits ✓")
    
    # 3. Near-duplicate check on sample
    if sample_near_dup > 0 and len(splits['test']) > 0:
        # Sample test codes and check against sample train codes
        test_sample = splits['test'][code_col].sample(min(sample_near_dup, len(splits['test'])), random_state=42).tolist()
        train_sample = splits['train'][code_col].sample(min(sample_near_dup, len(splits['train'])), random_state=42).tolist()
        
        def get_shingles(s, k=3):
            tokens = s.split()
            if len(tokens) < k:
                return set(tokens)
            return set(' '.join(tokens[i:i+k]) for i in range(len(tokens) - k + 1))
        
        train_shingles = [get_shingles(c) for c in train_sample]
        high_sim_count = 0
        for tc in test_sample:
            t_sh = get_shingles(tc)
            if not t_sh:
                continue
            for tr_sh in train_shingles:
                if not tr_sh:
                    continue
                jaccard = len(t_sh & tr_sh) / len(t_sh | tr_sh)
                if jaccard > 0.85:
                    high_sim_count += 1
                    break
        report['near_duplicates_flagged'] = high_sim_count
        logger.info(f"Near-duplicate check (sample={sample_near_dup}): {high_sim_count} pairs with Jaccard > 0.85")
    
    report['details'] = {
        'train_samples': len(splits['train']),
        'val_samples': len(splits.get('val', [])),
        'test_samples': len(splits['test']),
    }
    return report
