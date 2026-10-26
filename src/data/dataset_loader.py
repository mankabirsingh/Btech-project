"""
Dataset Loader for Provenance-Aware Security Risk Assessment
============================================================
Handles loading PrimeVul (vulnerability detection) and 
HumanVsAICode (provenance/AI-involvement) datasets.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
REPORTS_DIR = DATA_DIR / "reports"


def load_primevul(split: str = "train") -> pd.DataFrame:
    """
    Load PrimeVul dataset from HuggingFace datasets library.
    
    PrimeVul contains C/C++ functions with vulnerability labels.
    It provides binary vulnerability labels and CWE categories.
    
    Args:
        split: One of 'train', 'valid', 'test'
    
    Returns:
        DataFrame with columns: func, target, cwe, project, commit_id, ...
    """
    try:
        from datasets import load_dataset
        logger.info(f"Loading PrimeVul split: {split}")
        
        # PrimeVul is available on HuggingFace
        ds = load_dataset("ART-Group/PrimeVul", split=split, trust_remote_code=True)
        df = ds.to_pandas()
        
        logger.info(f"PrimeVul {split}: {len(df)} samples loaded")
        return df
        
    except Exception as e:
        logger.warning(f"Could not load PrimeVul from HuggingFace: {e}")
        # Try loading from local raw files
        return _load_primevul_local(split)


def _load_primevul_local(split: str) -> pd.DataFrame:
    """
    Load PrimeVul from local JSON/CSV files if downloaded via git clone.
    """
    primevul_dir = RAW_DIR / "primevul"
    
    # Try JSON format (common in PrimeVul repo)
    json_path = primevul_dir / f"{split}.jsonl"
    if json_path.exists():
        df = pd.read_json(json_path, lines=True)
        logger.info(f"Loaded PrimeVul {split} from local JSONL: {len(df)} samples")
        return df
    
    json_path2 = primevul_dir / f"{split}.json"
    if json_path2.exists():
        with open(json_path2, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        logger.info(f"Loaded PrimeVul {split} from local JSON: {len(df)} samples")
        return df
    
    csv_path = primevul_dir / f"{split}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded PrimeVul {split} from local CSV: {len(df)} samples")
        return df
    
    raise FileNotFoundError(
        f"PrimeVul {split} not found. Expected at {primevul_dir}. "
        "Run scripts/download_datasets.py first."
    )


def load_humanvsaicode() -> pd.DataFrame:
    """
    Load HumanVsAICode dataset from HuggingFace.
    
    This dataset contains human-written and AI-generated code samples.
    It does NOT necessarily contain vulnerability labels.
    
    Returns:
        DataFrame with code samples and provenance metadata.
    """
    try:
        from datasets import load_dataset
        logger.info("Loading HumanVsAICode dataset...")
        
        ds = load_dataset("OSS-forge/HumanVsAICode", trust_remote_code=True)
        
        # It may have splits or be a single dataset
        if isinstance(ds, dict):
            # Combine all splits for inspection
            dfs = []
            for split_name, split_ds in ds.items():
                df_split = split_ds.to_pandas()
                df_split['_split'] = split_name
                dfs.append(df_split)
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = ds.to_pandas()
        
        logger.info(f"HumanVsAICode: {len(df)} samples loaded")
        return df
        
    except Exception as e:
        logger.error(f"Could not load HumanVsAICode: {e}")
        raise


def inspect_dataset(df: pd.DataFrame, name: str, save_report: bool = True) -> Dict:
    """
    Comprehensive dataset inspection.
    
    Generates statistics about columns, types, missing values,
    label distributions, languages, and provenance information.
    
    Args:
        df: Dataset DataFrame
        name: Dataset identifier for the report
        save_report: Whether to save the report to disk
    
    Returns:
        Dictionary with inspection results
    """
    report = {
        "dataset_name": name,
        "num_samples": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
    }
    
    # Check for common vulnerability label columns
    vuln_cols = [c for c in df.columns if any(
        kw in c.lower() for kw in ['vuln', 'target', 'label', 'vulnerable', 'security', 'cwe']
    )]
    report["vulnerability_related_columns"] = vuln_cols
    
    for col in vuln_cols:
        try:
            sample_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            if isinstance(sample_val, (int, float, bool, str, np.number, np.bool_)):
                report[f"label_dist_{col}"] = df[col].value_counts().head(20).to_dict()
            else:
                report[f"label_dist_{col}"] = f"[{type(sample_val).__name__} non-scalar - omitted]"
        except Exception as e:
            report[f"label_dist_{col}"] = f"[Omitted: {e}]"
    
    # Check for language columns
    lang_cols = [c for c in df.columns if any(
        kw in c.lower() for kw in ['lang', 'language', 'extension']
    )]
    report["language_columns"] = lang_cols
    for col in lang_cols:
        try:
            sample_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            if isinstance(sample_val, (int, float, bool, str, np.number, np.bool_)):
                report[f"lang_dist_{col}"] = df[col].value_counts().head(20).to_dict()
        except Exception:
            pass
    
    # Check for code columns
    code_cols = [c for c in df.columns if any(
        kw in c.lower() for kw in ['code', 'func', 'function', 'snippet', 'source', 'content']
    )]
    report["code_columns"] = code_cols
    
    if code_cols:
        main_code_col = code_cols[0]
        code_lengths = df[main_code_col].astype(str).str.len()
        report["code_length_stats"] = {
            "mean": float(code_lengths.mean()),
            "median": float(code_lengths.median()),
            "min": int(code_lengths.min()),
            "max": int(code_lengths.max()),
            "std": float(code_lengths.std()),
        }
    
    # Check for provenance columns
    prov_cols = [c for c in df.columns if any(
        kw in c.lower() for kw in ['ai', 'human', 'model', 'generator', 'author', 
                                     'provenance', 'origin', 'source', 'generated']
    )]
    report["provenance_columns"] = prov_cols
    for col in prov_cols:
        try:
            sample_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            if isinstance(sample_val, (int, float, bool, str, np.number, np.bool_)):
                vc = df[col].value_counts()
                if len(vc) <= 30:
                    report[f"prov_dist_{col}"] = vc.to_dict()
        except Exception:
            pass
    
    # Check for task/project columns
    task_cols = [c for c in df.columns if any(
        kw in c.lower() for kw in ['task', 'project', 'problem', 'challenge', 'commit']
    )]
    report["task_columns"] = task_cols
    
    # Duplicates check on code column
    if code_cols:
        n_dup = df[code_cols[0]].duplicated().sum()
        report["duplicate_code_samples"] = int(n_dup)
        report["duplicate_pct"] = round(n_dup / len(df) * 100, 2) if len(df) > 0 else 0
    
    # Sample rows
    sample_size = min(3, len(df))
    if sample_size > 0:
        samples = df.head(sample_size).to_dict(orient='records')
        # Truncate long strings for readability
        for s in samples:
            for k, v in s.items():
                if isinstance(v, str) and len(v) > 300:
                    s[k] = v[:300] + "..."
        report["sample_rows"] = samples
    
    if save_report:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORTS_DIR / f"inspection_{name}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Inspection report saved to {report_path}")
        report["report_path"] = str(report_path)
    
    return report


def print_inspection_report(report: Dict):
    """Pretty-print a dataset inspection report."""
    print(f"\n{'='*60}")
    print(f"DATASET INSPECTION: {report['dataset_name']}")
    print(f"{'='*60}")
    print(f"Samples: {report['num_samples']}")
    print(f"Columns ({len(report['columns'])}): {report['columns']}")
    
    print(f"\nMissing Values:")
    for col, count in report['missing_values'].items():
        if count > 0:
            print(f"  {col}: {count} ({report['missing_pct'][col]}%)")
    
    if report.get('vulnerability_related_columns'):
        print(f"\nVulnerability Columns: {report['vulnerability_related_columns']}")
        for col in report['vulnerability_related_columns']:
            key = f"label_dist_{col}"
            if key in report:
                print(f"  {col} distribution: {report[key]}")
    
    if report.get('language_columns'):
        print(f"\nLanguage Columns: {report['language_columns']}")
    
    if report.get('provenance_columns'):
        print(f"\nProvenance Columns: {report['provenance_columns']}")
        for col in report['provenance_columns']:
            key = f"prov_dist_{col}"
            if key in report:
                print(f"  {col} distribution: {report[key]}")
    
    if report.get('code_length_stats'):
        stats = report['code_length_stats']
        print(f"\nCode Length Stats: mean={stats['mean']:.0f}, "
              f"median={stats['median']:.0f}, "
              f"min={stats['min']}, max={stats['max']}")
    
    if report.get('duplicate_code_samples') is not None:
        print(f"\nDuplicate code samples: {report['duplicate_code_samples']} "
              f"({report.get('duplicate_pct', 0)}%)")
    
    print(f"{'='*60}\n")
