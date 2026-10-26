#!/usr/bin/env python3
"""
Dataset Download, Inspection, and Preparation Pipeline (v2)
============================================================
Downloads PrimeVul, Devign, and HumanVsAICode, inspects schemas,
standardizes data, creates reproducible splits.

Usage:
    python scripts/prepare_dataset.py
"""

import sys
import os
import json
import logging
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"


def inspect_and_report(df, name):
    """Run inspection and print report."""
    from src.data.dataset_loader import inspect_dataset, print_inspection_report
    report = inspect_dataset(df, name)
    print_inspection_report(report)
    return report


def load_devign():
    """Load Devign dataset — well-established C/C++ vulnerability detection."""
    from datasets import load_dataset
    logger.info("=" * 60)
    logger.info("Loading Devign Dataset (DetectVul/devign)")
    logger.info("=" * 60)
    
    ds = load_dataset("DetectVul/devign")
    all_dfs = {}
    for split_name in ds:
        df = ds[split_name].to_pandas()
        all_dfs[split_name] = df
        inspect_and_report(df, f"devign_{split_name}")
    
    return all_dfs


def load_primevul_dataset():
    """Load PrimeVul dataset with vulnerability+fix pairs."""
    from datasets import load_dataset
    logger.info("=" * 60)
    logger.info("Loading PrimeVul Dataset (adefu020/primevul)")
    logger.info("=" * 60)
    
    ds = load_dataset("adefu020/primevul")
    all_dfs = {}
    for split_name in ds:
        df = ds[split_name].to_pandas()
        all_dfs[split_name] = df
        inspect_and_report(df, f"primevul_{split_name}")
    
    return all_dfs


def load_humanvsaicode_dataset():
    """Load HumanVsAICode — paired human/AI code samples."""
    from datasets import load_dataset
    logger.info("=" * 60)
    logger.info("Loading HumanVsAICode Dataset")
    logger.info("=" * 60)
    
    ds = load_dataset("OSS-forge/HumanVsAICode")
    all_dfs = {}
    for split_name in ds:
        df = ds[split_name].to_pandas()
        all_dfs[split_name] = df
        inspect_and_report(df, f"humanvsaicode_{split_name}")
    
    return all_dfs


def standardize_devign(all_dfs):
    """
    Standardize Devign to unified schema.
    Devign columns: id, func, target, project, commit_id, ...
    target: True/False (vulnerability label)
    """
    import pandas as pd
    import numpy as np
    from src.data.preprocessing import clean_code
    
    frames = []
    for split_name, df in all_dfs.items():
        result = pd.DataFrame()
        result['sample_id'] = df['id'].apply(lambda x: f"devign_{split_name}_{x}")
        result['code'] = df['func'].apply(clean_code)
        result['language'] = 'C/C++'
        
        # target is bool in Devign
        result['vuln_label'] = df['target'].astype(int)
        result['cwe'] = np.nan
        result['severity'] = np.nan
        result['source_dataset'] = 'devign'
        result['task_id'] = np.nan
        result['provenance_type'] = 'human'
        result['ai_generator'] = np.nan
        result['ai_involvement'] = 0.0
        result['human_edit_history'] = np.nan
        result['parent_sample_id'] = np.nan
        result['project'] = df['project'] if 'project' in df.columns else np.nan
        result['commit_id'] = df['commit_id'] if 'commit_id' in df.columns else np.nan
        result['_original_split'] = split_name
        
        # Group ID: group by commit_id to keep functions from the same commit in the same split
        commit_col = df['commit_id'] if 'commit_id' in df.columns else pd.Series(index=df.index, dtype=str)
        result['group_id'] = [
            f"devign_commit_{c}" if pd.notna(c) and str(c).strip() != '' else f"devign_sample_{sid}"
            for c, sid in zip(commit_col, df['id'])
        ]
        
        # Filter empty code
        result = result[result['code'].str.len() > 10]
        frames.append(result)
    
    combined = pd.concat(frames, ignore_index=True)
    
    # Remove exact duplicates
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=['code'], keep='first').reset_index(drop=True)
    logger.info(f"Devign standardized: {n_before} -> {len(combined)} samples "
                f"(removed {n_before-len(combined)} duplicates)")
    logger.info(f"  Vulnerable: {(combined['vuln_label']==1).sum()}, "
                f"Benign: {(combined['vuln_label']==0).sum()}")
    
    return combined


def standardize_primevul(all_dfs):
    """
    Standardize PrimeVul to unified schema.
    Uses vulnerable_function for vulnerable rows and fixed_function for benign rows.
    Assigns a shared group_id to ensure vulnerable-fixed pairs never cross splits.
    """
    import pandas as pd
    import numpy as np
    from src.data.preprocessing import clean_code
    
    frames = []
    for split_name, df in all_dfs.items():
        result = pd.DataFrame()
        is_vuln = df['is_vulnerable'].astype(bool)
        
        # Correctly assign vulnerable_function when is_vuln=True, and fixed_function when is_vuln=False
        raw_code = np.where(is_vuln, df['vulnerable_function'], df['fixed_function'])
        result['code'] = [clean_code(c) for c in raw_code]
        
        rec_ids = df['source_record_id'].astype(str)
        result['sample_id'] = [
            f"primevul_{split_name}_{rec}_{'vuln' if v else 'fixed'}"
            for rec, v in zip(rec_ids, is_vuln)
        ]
        result['language'] = 'C/C++'
        result['vuln_label'] = is_vuln.astype(int)
        
        # CWE info: present for vulnerable version
        if 'cwe_ids' in df.columns:
            result['cwe'] = [
                (cwe_list[0] if isinstance(cwe_list, list) and len(cwe_list) > 0 else np.nan)
                if v else np.nan
                for cwe_list, v in zip(df['cwe_ids'], is_vuln)
            ]
        else:
            result['cwe'] = np.nan
        
        result['severity'] = np.nan
        result['source_dataset'] = 'primevul'
        result['task_id'] = rec_ids.apply(lambda x: f"pv_task_{x}")
        result['provenance_type'] = 'human'
        result['ai_generator'] = np.nan
        result['ai_involvement'] = 0.0
        result['human_edit_history'] = np.nan
        
        # Link fixed version to vulnerable version as parent
        result['parent_sample_id'] = [
            f"primevul_{split_name}_{rec}_vuln" if not v else np.nan
            for rec, v in zip(rec_ids, is_vuln)
        ]
        result['project'] = df['project'] if 'project' in df.columns else np.nan
        result['commit_id'] = df.get('fixing_commit_sha', pd.Series(dtype=str))
        result['_original_split'] = split_name
        
        # Group ID: pair-level grouping to keep both vulnerable and fixed functions in the same split!
        result['group_id'] = rec_ids.apply(lambda x: f"pv_pair_{x}")
        
        result = result[result['code'].str.len() > 10]
        frames.append(result)
    
    combined = pd.concat(frames, ignore_index=True)
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=['code'], keep='first').reset_index(drop=True)
    logger.info(f"PrimeVul standardized: {n_before} -> {len(combined)} samples")
    logger.info(f"  Vulnerable: {(combined['vuln_label']==1).sum()}, "
                f"Benign: {(combined['vuln_label']==0).sum()}")
    
    return combined


def standardize_humanvsaicode(all_dfs):
    """
    Standardize HumanVsAICode to unified provenance schema.
    
    Columns: hm_index, docstring, human_code, chatgpt_code, dsc_code, qwen_code
    
    This dataset has PAIRED implementations: same task implemented by human
    and by multiple AI models. This is ideal for provenance experiments.
    
    We "unpivot" it: each code sample becomes a row with its provenance.
    """
    import pandas as pd
    import numpy as np
    from src.data.preprocessing import clean_code
    
    frames = []
    
    for split_name, df in all_dfs.items():
        # Each row has human_code + chatgpt_code + dsc_code + qwen_code
        # We unpivot to create separate samples
        
        code_columns = {
            'human_code': ('human', None, 0.0),
            'chatgpt_code': ('ai', 'ChatGPT', 1.0),
            'dsc_code': ('ai', 'DeepSeek-Coder', 1.0),
            'qwen_code': ('ai', 'Qwen', 1.0),
        }
        
        for col_name, (prov_type, generator, ai_inv) in code_columns.items():
            if col_name not in df.columns:
                continue
            
            result = pd.DataFrame()
            result['sample_id'] = df['hm_index'].apply(
                lambda x: f"hvai_{split_name}_{x}_{col_name}"
            )
            result['code'] = df[col_name].apply(clean_code)
            
            # Detect language from content
            # Dataset has Java and Python subsets based on file names
            result['language'] = 'Python'  # Will refine below
            
            result['vuln_label'] = np.nan  # No vulnerability labels!
            result['cwe'] = np.nan
            result['severity'] = np.nan
            result['source_dataset'] = 'humanvsaicode'
            
            # Task ID from hm_index + docstring
            result['task_id'] = df['hm_index'].apply(lambda x: f"task_{x}")
            
            result['provenance_type'] = prov_type
            result['ai_generator'] = generator if generator else np.nan
            result['ai_involvement'] = ai_inv
            result['human_edit_history'] = np.nan
            
            # Link AI versions to human version as parent
            if prov_type == 'ai':
                result['parent_sample_id'] = df['hm_index'].apply(
                    lambda x: f"hvai_{split_name}_{x}_human_code"
                )
            else:
                result['parent_sample_id'] = np.nan
            
            result['project'] = np.nan
            result['commit_id'] = np.nan
            result['_original_split'] = split_name
            result['docstring'] = df['docstring'] if 'docstring' in df.columns else np.nan
            
            result = result[result['code'].str.len() > 10]
            frames.append(result)
    
    combined = pd.concat(frames, ignore_index=True)
    
    # Detect language heuristically
    def detect_lang(code):
        if 'import java.' in code or 'public class' in code or 'public static void main' in code:
            return 'Java'
        return 'Python'
    
    combined['language'] = combined['code'].apply(detect_lang)
    
    logger.info(f"HumanVsAICode standardized: {len(combined)} samples")
    logger.info(f"  Provenance: {combined['provenance_type'].value_counts().to_dict()}")
    logger.info(f"  AI generators: {combined['ai_generator'].value_counts().to_dict()}")
    logger.info(f"  Languages: {combined['language'].value_counts().to_dict()}")
    
    return combined


def create_splits(vuln_df):
    """Create train/val/test splits for vulnerability data using group-level isolation."""
    from src.data.split_dataset import split_vulnerability_dataset, save_splits, verify_no_leakage
    
    logger.info("=" * 60)
    logger.info("Creating Train/Val/Test Splits (Group Isolation on 'group_id')")
    logger.info("=" * 60)
    
    splits = split_vulnerability_dataset(vuln_df, group_col='group_id', test_size=0.15, val_size=0.10, random_seed=42)
    save_splits(splits, SPLITS_DIR, "vulnerability")
    
    leakage_report = verify_no_leakage(splits, group_col='group_id')
    if not leakage_report['passed']:
        logger.error("DATA LEAKAGE DETECTED!")
    
    # Save leakage report
    with open(REPORTS_DIR / "leakage_audit_report.json", 'w') as f:
        json.dump(leakage_report, f, indent=2, default=str)
    
    for split_name, split_df in splits.items():
        n_vuln = (split_df['vuln_label'] == 1).sum()
        n_benign = (split_df['vuln_label'] == 0).sum()
        logger.info(f"  {split_name}: {len(split_df)} samples "
                    f"(vuln={n_vuln}, benign={n_benign}, "
                    f"vuln_rate={n_vuln/len(split_df)*100:.1f}%)")
    
    return splits


def main():
    logger.info("Starting Dataset Preparation Pipeline v2")
    start_time = time.time()
    
    # ============================
    # Step 1: Load Devign
    # ============================
    try:
        devign_dfs = load_devign()
        devign_std = standardize_devign(devign_dfs)
    except Exception as e:
        logger.error(f"Devign loading failed: {e}")
        devign_std = None
    
    # ============================
    # Step 2: Try PrimeVul  
    # ============================
    primevul_std = None
    try:
        primevul_dfs = load_primevul_dataset()
        primevul_std = standardize_primevul(primevul_dfs)
    except Exception as e:
        logger.warning(f"PrimeVul loading failed (non-critical): {e}")
    
    # ============================
    # Step 3: Load HumanVsAICode
    # ============================
    hvai_path = PROCESSED_DIR / "provenance_data.parquet"
    if hvai_path.exists():
        import pandas as pd
        logger.info(f"Provenance data already exists at {hvai_path}. Reusing existing file.")
        hvai_std = pd.read_parquet(hvai_path)
        logger.info(f"Loaded existing provenance data: {len(hvai_std)} samples")
    else:
        try:
            hvai_dfs = load_humanvsaicode_dataset()
            hvai_std = standardize_humanvsaicode(hvai_dfs)
        except Exception as e:
            logger.warning(f"HumanVsAICode loading failed: {e}")
            hvai_std = None
    
    # ============================
    # Step 4: Combine vulnerability data & create splits
    # ============================
    import pandas as pd
    
    vuln_frames = []
    if devign_std is not None:
        vuln_frames.append(devign_std)
    if primevul_std is not None:
        vuln_frames.append(primevul_std)
    
    if vuln_frames:
        vuln_combined = pd.concat(vuln_frames, ignore_index=True)
        # Remove cross-dataset duplicates
        n_before = len(vuln_combined)
        vuln_combined = vuln_combined.drop_duplicates(subset=['code'], keep='first').reset_index(drop=True)
        logger.info(f"Combined vulnerability data: {n_before} -> {len(vuln_combined)} "
                    f"(removed {n_before - len(vuln_combined)} cross-dataset duplicates)")
        
        # Save processed
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        vuln_combined.to_parquet(PROCESSED_DIR / "vulnerability_data.parquet", index=False)
        logger.info(f"Saved vulnerability data: {len(vuln_combined)} samples")
        
        # Create splits
        splits = create_splits(vuln_combined)
    else:
        logger.error("No vulnerability data available!")
        splits = None
    
    # Save provenance data
    if hvai_std is not None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        hvai_std.to_parquet(PROCESSED_DIR / "provenance_data.parquet", index=False)
        logger.info(f"Saved provenance data: {len(hvai_std)} samples")
    
    # ============================
    # Step 5: Generate summary report
    # ============================
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": {
            "devign": {
                "loaded": devign_std is not None,
                "n_samples": len(devign_std) if devign_std is not None else 0,
                "source": "DetectVul/devign",
            },
            "primevul": {
                "loaded": primevul_std is not None,
                "n_samples": len(primevul_std) if primevul_std is not None else 0,
                "source": "adefu020/primevul",
            },
            "humanvsaicode": {
                "loaded": hvai_std is not None,
                "n_samples": len(hvai_std) if hvai_std is not None else 0,
                "source": "OSS-forge/HumanVsAICode",
            },
        },
        "vulnerability_data": {
            "total": len(vuln_combined) if vuln_frames else 0,
            "vuln": int((vuln_combined['vuln_label']==1).sum()) if vuln_frames else 0,
            "benign": int((vuln_combined['vuln_label']==0).sum()) if vuln_frames else 0,
        } if vuln_frames else {},
        "splits": {
            k: len(v) for k, v in splits.items()
        } if splits else {},
        "elapsed_sec": time.time() - start_time,
    }
    
    with open(REPORTS_DIR / "dataset_preparation_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset preparation completed in {report['elapsed_sec']:.1f}s")
    logger.info(json.dumps(report, indent=2, default=str))
    
    return report


if __name__ == "__main__":
    main()
