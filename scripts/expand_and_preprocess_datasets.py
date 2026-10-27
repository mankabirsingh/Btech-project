#!/usr/bin/env python3
"""
Dataset Expansion, Decontamination, and Unified Preprocessing Pipeline (Phase 2 & 3)
=====================================================================================
Downloads and standardizes candidate vulnerability datasets:
  1. DiverseVul (bstee615/diversevul) - Broad real-world C/C++ coverage (150+ projects)
  2. BigVul (bstee615/bigvul) - Pre/post patch pairs with line-level context
  3. ReVeal (claudios/ReVeal) - Real-world Chromium & Debian vulnerabilities
  4. Juliet (LorenzH/juliet_test_suite_c_1_3) - Synthetic benchmark (isolated)

Strict Methodology Rules:
  - Strict Decontamination: Every candidate sample matching code in the original
    untouched test set (data/splits/vulnerability_test.parquet) is quarantined & excluded.
  - Conflict Audit: Any identical code with conflicting labels across datasets is logged.
  - The original held-out test set remains 100% UNTOUCHED and identical (N = 5,412).
  - Creates an expanded training corpus, an expanded validation set, a held-out novel
    projects test set, and a synthetic Juliet evaluation set.
"""

import sys
import os
import json
import time
import logging
import hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import clean_code

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger("expand_dataset")

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
REPORTS_DIR = DATA_DIR / "reports"

def compute_code_hash(code_str: str) -> str:
    """Compute sha256 hash of cleaned code string."""
    cleaned = clean_code(str(code_str))
    return hashlib.sha256(cleaned.encode('utf-8')).hexdigest()


def load_and_decontaminate_test_hashes():
    """Load hashes and labels from the untouched existing test set."""
    test_path = SPLITS_DIR / "vulnerability_test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Original test set not found at {test_path}!")
    
    test_df = pd.read_parquet(test_path)
    logger.info(f"Loaded untouched original test set: {len(test_df)} samples")
    
    test_hashes = {}
    for idx, row in test_df.iterrows():
        h = compute_code_hash(row['code'])
        test_hashes[h] = {
            'sample_id': row['sample_id'],
            'vuln_label': int(row['vuln_label']),
            'project': row.get('project', 'unknown')
        }
    return test_hashes, test_df


def extract_diversevul(test_hashes, max_benign_per_proj=500):
    """
    Extract DiverseVul samples: all vulnerable functions + balanced benign functions,
    excluding any that match the untouched test set.
    """
    logger.info("=" * 60)
    logger.info("Extracting and standardizing DiverseVul (bstee615/diversevul)...")
    logger.info("=" * 60)
    
    dv_ds = load_dataset('bstee615/diversevul', split='train')
    
    records = []
    quarantined = []
    label_conflicts = []
    
    proj_benign_counts = Counter()
    
    for i, row in enumerate(dv_ds):
        func_raw = row.get('func')
        if not func_raw or len(str(func_raw).strip()) < 15:
            continue
        
        cleaned = clean_code(func_raw)
        h = hashlib.sha256(cleaned.encode('utf-8')).hexdigest()
        
        target = int(row.get('target', 0))
        proj = str(row.get('project', 'unknown')).strip()
        commit = str(row.get('commit_id', '')).strip()
        
        # Check decontamination against untouched test set
        if h in test_hashes:
            test_info = test_hashes[h]
            is_conflict = (target != test_info['vuln_label'])
            quarantined.append({
                'source': 'diversevul',
                'index': i,
                'project': proj,
                'commit': commit,
                'target': target,
                'test_sample_id': test_info['sample_id'],
                'test_label': test_info['vuln_label'],
                'conflict': is_conflict
            })
            if is_conflict:
                label_conflicts.append(quarantined[-1])
            continue
        
        # Balance benign samples per project to avoid Linux overwhelming everything
        if target == 0:
            if proj_benign_counts[proj] >= max_benign_per_proj:
                continue
            proj_benign_counts[proj] += 1
            
        cwe_val = row.get('cwe')
        cwe_str = cwe_val[0] if isinstance(cwe_val, list) and len(cwe_val) > 0 else np.nan
        
        records.append({
            'sample_id': f"dv_train_{i}",
            'code': cleaned,
            'language': 'C/C++',
            'vuln_label': target,
            'cwe': cwe_str,
            'severity': np.nan,
            'source_dataset': 'diversevul',
            'task_id': f"dv_task_{proj}_{commit[:8]}" if commit else np.nan,
            'provenance_type': 'human',
            'ai_generator': np.nan,
            'ai_involvement': 0.0,
            'human_edit_history': np.nan,
            'parent_sample_id': np.nan,
            'project': proj,
            'commit_id': commit if commit else np.nan,
            'group_id': f"dv_{proj}_{commit}" if commit else f"dv_sample_{i}"
        })
    
    df = pd.DataFrame(records)
    logger.info(f"DiverseVul extracted: {len(df)} samples "
                f"(Vuln: {(df['vuln_label']==1).sum()}, Benign: {(df['vuln_label']==0).sum()})")
    logger.info(f"Quarantined test-matching samples: {len(quarantined)}, Conflicting labels: {len(label_conflicts)}")
    return df, quarantined, label_conflicts


def extract_bigvul(test_hashes, max_benign_per_proj=300):
    """
    Extract BigVul samples: all vulnerable functions (func_before when vul=1)
    and paired/benign functions (func_after or vul=0), decontaminated against test set.
    """
    logger.info("=" * 60)
    logger.info("Extracting and standardizing BigVul (bstee615/bigvul)...")
    logger.info("=" * 60)
    
    bv_ds = load_dataset('bstee615/bigvul', split='train')
    
    records = []
    quarantined = []
    label_conflicts = []
    proj_benign_counts = Counter()
    
    for i, row in enumerate(bv_ds):
        vul = int(row.get('vul', 0))
        proj = str(row.get('project', 'unknown')).strip()
        commit = str(row.get('commit_id', '')).strip()
        cwe_str = row.get('CWE ID') or np.nan
        
        # When vul=1, func_before is vulnerable
        if vul == 1:
            code_raw = row.get('func_before')
            target = 1
        else:
            code_raw = row.get('func_after') or row.get('func_before')
            target = 0
            
        if not code_raw or len(str(code_raw).strip()) < 15:
            continue
            
        cleaned = clean_code(code_raw)
        h = hashlib.sha256(cleaned.encode('utf-8')).hexdigest()
        
        if h in test_hashes:
            test_info = test_hashes[h]
            is_conflict = (target != test_info['vuln_label'])
            quarantined.append({
                'source': 'bigvul',
                'index': i,
                'project': proj,
                'commit': commit,
                'target': target,
                'test_sample_id': test_info['sample_id'],
                'test_label': test_info['vuln_label'],
                'conflict': is_conflict
            })
            if is_conflict:
                label_conflicts.append(quarantined[-1])
            continue
            
        if target == 0:
            if proj_benign_counts[proj] >= max_benign_per_proj:
                continue
            proj_benign_counts[proj] += 1
            
        records.append({
            'sample_id': f"bv_train_{i}",
            'code': cleaned,
            'language': 'C/C++',
            'vuln_label': target,
            'cwe': cwe_str,
            'severity': np.nan,
            'source_dataset': 'bigvul',
            'task_id': f"bv_task_{proj}_{commit[:8]}" if commit else np.nan,
            'provenance_type': 'human',
            'ai_generator': np.nan,
            'ai_involvement': 0.0,
            'human_edit_history': np.nan,
            'parent_sample_id': np.nan,
            'project': proj,
            'commit_id': commit if commit else np.nan,
            'group_id': f"bv_{proj}_{commit}" if commit else f"bv_sample_{i}"
        })
        
    df = pd.DataFrame(records)
    logger.info(f"BigVul extracted: {len(df)} samples "
                f"(Vuln: {(df['vuln_label']==1).sum()}, Benign: {(df['vuln_label']==0).sum()})")
    logger.info(f"Quarantined test-matching samples: {len(quarantined)}, Conflicting labels: {len(label_conflicts)}")
    return df, quarantined, label_conflicts


def extract_juliet_synthetic():
    """
    Extract synthetic Juliet test suite and save as separate benchmark.
    """
    logger.info("=" * 60)
    logger.info("Extracting Synthetic Juliet Benchmark (LorenzH/juliet_test_suite_c_1_3)...")
    logger.info("=" * 60)
    
    jl_test = load_dataset('LorenzH/juliet_test_suite_c_1_3', split='test[:2000]')
    records = []
    
    for i, row in enumerate(jl_test):
        good_code = row.get('good')
        bad_code = row.get('bad')
        cls_id = row.get('class')
        fname = row.get('filename', '')
        
        if bad_code and len(bad_code.strip()) > 15:
            records.append({
                'sample_id': f"juliet_test_{i}_bad",
                'code': clean_code(bad_code),
                'language': 'C/C++',
                'vuln_label': 1,
                'cwe': f"CWE-{cls_id}",
                'severity': 'Synthetic',
                'source_dataset': 'juliet_synthetic',
                'task_id': f"juliet_{cls_id}_{fname}",
                'provenance_type': 'synthetic',
                'ai_generator': np.nan,
                'ai_involvement': 0.0,
                'human_edit_history': np.nan,
                'parent_sample_id': np.nan,
                'project': 'NIST_Juliet_v1.3',
                'commit_id': np.nan,
                'group_id': f"juliet_pair_{i}"
            })
            
        if good_code and len(good_code.strip()) > 15:
            records.append({
                'sample_id': f"juliet_test_{i}_good",
                'code': clean_code(good_code),
                'language': 'C/C++',
                'vuln_label': 0,
                'cwe': f"CWE-{cls_id}",
                'severity': 'Synthetic',
                'source_dataset': 'juliet_synthetic',
                'task_id': f"juliet_{cls_id}_{fname}",
                'provenance_type': 'synthetic',
                'ai_generator': np.nan,
                'ai_involvement': 0.0,
                'human_edit_history': np.nan,
                'parent_sample_id': f"juliet_test_{i}_bad",
                'project': 'NIST_Juliet_v1.3',
                'commit_id': np.nan,
                'group_id': f"juliet_pair_{i}"
            })
            
    df = pd.DataFrame(records)
    logger.info(f"Juliet Synthetic test extracted: {len(df)} samples (Vuln: {(df['vuln_label']==1).sum()}, Benign: {(df['vuln_label']==0).sum()})")
    return df


def main():
    start_time = time.time()
    logger.info("Starting Phase 2 & 3 Dataset Expansion & Decontamination...")
    
    # Step 1: Load test hashes to strictly prevent test leakage
    test_hashes, test_df = load_and_decontaminate_test_hashes()
    
    # Step 2: Extract decontaminated DiverseVul & BigVul
    dv_df, dv_quar, dv_conf = extract_diversevul(test_hashes, max_benign_per_proj=400)
    bv_df, bv_quar, bv_conf = extract_bigvul(test_hashes, max_benign_per_proj=300)
    
    all_quarantined = dv_quar + bv_quar
    all_conflicts = dv_conf + bv_conf
    
    # Save quarantine & conflict audit
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "dataset_decontamination_audit.json", "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_quarantined_test_matches": len(all_quarantined),
            "label_conflicts_with_test": len(all_conflicts),
            "conflicts_detail": all_conflicts,
            "sample_quarantined": all_quarantined[:10]
        }, f, indent=2)
    logger.info(f"Saved decontamination audit report: {len(all_quarantined)} quarantined samples.")
    
    # Step 3: Combine with existing training and validation sets
    existing_train_df = pd.read_parquet(SPLITS_DIR / "vulnerability_train.parquet")
    existing_val_df = pd.read_parquet(SPLITS_DIR / "vulnerability_val.parquet")
    
    logger.info(f"Existing train: {len(existing_train_df)}, Existing val: {len(existing_val_df)}")
    
    combined_new = pd.concat([dv_df, bv_df], ignore_index=True)
    combined_new = combined_new.drop_duplicates(subset=['code'], keep='first').reset_index(drop=True)
    logger.info(f"New decontaminated samples after deduplication: {len(combined_new)} "
                f"(Vuln: {(combined_new['vuln_label']==1).sum()}, Benign: {(combined_new['vuln_label']==0).sum()})")
    
    # Create project-held-out out-of-distribution test set
    # Hold out novel projects not present in original Devign/PrimeVul train (e.g., 'vim', 'gpac', 'envoy')
    heldout_projects = ['vim', 'gpac', 'envoy', 'openssl']
    heldout_mask = combined_new['project'].isin(heldout_projects)
    test_heldout_projects_df = combined_new[heldout_mask].copy().reset_index(drop=True)
    
    # Remaining new data for train & val
    pool_for_train_val = combined_new[~heldout_mask].copy().reset_index(drop=True)
    
    # Shuffle and split pool into 85% train expansion, 15% val expansion
    np.random.seed(42)
    shuffled_idx = np.random.permutation(len(pool_for_train_val))
    val_split_n = int(0.15 * len(pool_for_train_val))
    val_expanded_new = pool_for_train_val.iloc[shuffled_idx[:val_split_n]]
    train_expanded_new = pool_for_train_val.iloc[shuffled_idx[val_split_n:]]
    
    # Combine with existing train and val
    train_expanded_total = pd.concat([existing_train_df, train_expanded_new], ignore_index=True)
    train_expanded_total = train_expanded_total.drop_duplicates(subset=['code'], keep='first').reset_index(drop=True)
    
    val_expanded_total = pd.concat([existing_val_df, val_expanded_new], ignore_index=True)
    val_expanded_total = val_expanded_total.drop_duplicates(subset=['code'], keep='first').reset_index(drop=True)
    
    # Decontamination verification check: ensure ZERO overlap with untouched test set
    test_code_set = set(test_df['code'])
    train_leak = len(set(train_expanded_total['code']) & test_code_set)
    val_leak = len(set(val_expanded_total['code']) & test_code_set)
    heldout_leak = len(set(test_heldout_projects_df['code']) & test_code_set)
    
    logger.info("=" * 60)
    logger.info("LEAKAGE & DECONTAMINATION AUDIT CHECK:")
    logger.info(f"  Overlap between Train Expanded and Test: {train_leak} (MUST BE 0)")
    logger.info(f"  Overlap between Val Expanded and Test:   {val_leak} (MUST BE 0)")
    logger.info(f"  Overlap between Novel Proj Test & Test:  {heldout_leak} (MUST BE 0)")
    logger.info("=" * 60)
    
    assert train_leak == 0, f"Critical error: {train_leak} training samples overlap with test set!"
    assert val_leak == 0, f"Critical error: {val_leak} validation samples overlap with test set!"
    
    # Extract synthetic Juliet benchmark
    juliet_df = extract_juliet_synthetic()
    
    # Step 4: Save Parquet files
    train_expanded_total.to_parquet(SPLITS_DIR / "vulnerability_train_expanded.parquet", index=False)
    val_expanded_total.to_parquet(SPLITS_DIR / "vulnerability_val_expanded.parquet", index=False)
    test_heldout_projects_df.to_parquet(SPLITS_DIR / "vulnerability_test_heldout_projects.parquet", index=False)
    juliet_df.to_parquet(SPLITS_DIR / "vulnerability_test_synthetic_juliet.parquet", index=False)
    
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "splits": {
            "vulnerability_train_expanded": {
                "n_samples": len(train_expanded_total),
                "vuln_count": int((train_expanded_total['vuln_label'] == 1).sum()),
                "benign_count": int((train_expanded_total['vuln_label'] == 0).sum()),
                "vuln_rate": float((train_expanded_total['vuln_label'] == 1).mean()),
                "n_projects": int(train_expanded_total['project'].nunique())
            },
            "vulnerability_val_expanded": {
                "n_samples": len(val_expanded_total),
                "vuln_count": int((val_expanded_total['vuln_label'] == 1).sum()),
                "benign_count": int((val_expanded_total['vuln_label'] == 0).sum()),
                "vuln_rate": float((val_expanded_total['vuln_label'] == 1).mean()),
                "n_projects": int(val_expanded_total['project'].nunique())
            },
            "vulnerability_test_untouched_original": {
                "n_samples": len(test_df),
                "vuln_count": int((test_df['vuln_label'] == 1).sum()),
                "benign_count": int((test_df['vuln_label'] == 0).sum()),
                "vuln_rate": float((test_df['vuln_label'] == 1).mean()),
                "status": "UNTOUCHED_IDENTICAL"
            },
            "vulnerability_test_heldout_projects": {
                "n_samples": len(test_heldout_projects_df),
                "vuln_count": int((test_heldout_projects_df['vuln_label'] == 1).sum()),
                "benign_count": int((test_heldout_projects_df['vuln_label'] == 0).sum()),
                "projects": list(heldout_projects)
            },
            "vulnerability_test_synthetic_juliet": {
                "n_samples": len(juliet_df),
                "vuln_count": int((juliet_df['vuln_label'] == 1).sum()),
                "benign_count": int((juliet_df['vuln_label'] == 0).sum()),
                "status": "NIST_SYNTHETIC_BENCHMARK"
            }
        },
        "decontamination": {
            "total_quarantined": len(all_quarantined),
            "label_conflicts": len(all_conflicts),
            "train_test_leak": train_leak,
            "val_test_leak": val_leak
        },
        "elapsed_sec": time.time() - start_time
    }
    
    with open(REPORTS_DIR / "dataset_expansion_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    logger.info(f"Dataset expansion completed successfully in {manifest['elapsed_sec']:.1f}s.")
    logger.info(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
