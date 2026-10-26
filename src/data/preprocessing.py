"""
Data Preprocessing for Provenance-Aware Security Risk Assessment
=================================================================
Standardizes datasets into a unified schema and handles cleaning.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Unified schema for processed data
UNIFIED_COLUMNS = [
    'sample_id',           # Unique identifier
    'code',                # Source code text
    'language',            # Programming language
    'vuln_label',          # Binary: 1=vulnerable, 0=benign
    'cwe',                 # CWE identifier (if available)
    'severity',            # Severity level (if available)
    'source_dataset',      # Origin dataset name
    'task_id',             # Programming task identifier (if available)
    'provenance_type',     # 'human', 'ai', 'mixed', 'unknown'
    'ai_generator',        # AI model name (if known)
    'ai_involvement',      # Float 0.0-1.0 (if known), NaN if unknown
    'human_edit_history',  # JSON string of edit history (if available)
    'parent_sample_id',    # ID of parent version (if applicable)
    'project',             # Source project (if available)
    'commit_id',           # Commit hash (if available)
]


def clean_code(code: str) -> str:
    """
    Basic code cleaning without altering semantics.
    
    - Strip trailing whitespace from lines
    - Normalize line endings
    - Remove null bytes
    """
    if not isinstance(code, str):
        return ""
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    code = code.replace('\x00', '')
    lines = [line.rstrip() for line in code.split('\n')]
    # Remove trailing empty lines
    while lines and lines[-1] == '':
        lines.pop()
    return '\n'.join(lines)


def standardize_primevul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert PrimeVul DataFrame to unified schema.
    
    PrimeVul typically has columns:
    - func: code function
    - target: 1=vulnerable, 0=benign  
    - project: source project
    - commit_id: git commit
    - cwe: CWE identifiers (may be list or string)
    """
    logger.info(f"Standardizing PrimeVul: {len(df)} samples")
    
    # Map column names
    code_col = None
    for candidate in ['func', 'function', 'code', 'processed_func']:
        if candidate in df.columns:
            code_col = candidate
            break
    
    if code_col is None:
        raise ValueError(f"No code column found in PrimeVul. Columns: {list(df.columns)}")
    
    label_col = None
    for candidate in ['target', 'label', 'vulnerable']:
        if candidate in df.columns:
            label_col = candidate
            break
    
    if label_col is None:
        raise ValueError(f"No label column found in PrimeVul. Columns: {list(df.columns)}")
    
    result = pd.DataFrame()
    result['sample_id'] = [f"primevul_{i}" for i in range(len(df))]
    result['code'] = df[code_col].apply(clean_code).values
    result['language'] = 'C/C++'
    result['vuln_label'] = df[label_col].astype(int).values
    
    # CWE
    if 'cwe' in df.columns:
        result['cwe'] = df['cwe'].values
    else:
        result['cwe'] = np.nan
    
    result['severity'] = np.nan  # Not typically in PrimeVul
    result['source_dataset'] = 'primevul'
    result['task_id'] = np.nan
    
    # PrimeVul is real-world open-source code => provenance is 'human' (pre-AI era mostly)
    result['provenance_type'] = 'human'
    result['ai_generator'] = np.nan
    result['ai_involvement'] = 0.0  # Real open-source code
    result['human_edit_history'] = np.nan
    result['parent_sample_id'] = np.nan
    
    if 'project' in df.columns:
        result['project'] = df['project'].values
    else:
        result['project'] = np.nan
    
    if 'commit_id' in df.columns:
        result['commit_id'] = df['commit_id'].values
    else:
        result['commit_id'] = np.nan
    
    # Filter out empty code
    result = result[result['code'].str.len() > 10].reset_index(drop=True)
    # Update sample_ids after filtering
    result['sample_id'] = [f"primevul_{i}" for i in range(len(result))]
    
    logger.info(f"PrimeVul standardized: {len(result)} samples, "
                f"vulnerable={result['vuln_label'].sum()}, "
                f"benign={(result['vuln_label']==0).sum()}")
    
    return result


def standardize_humanvsaicode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert HumanVsAICode DataFrame to unified schema.
    
    NOTE: This dataset typically does NOT have vulnerability labels.
    We set vuln_label to NaN and use it only for provenance experiments.
    """
    logger.info(f"Standardizing HumanVsAICode: {len(df)} samples")
    
    result = pd.DataFrame()
    result['sample_id'] = [f"hvai_{i}" for i in range(len(df))]
    
    # Find code column
    code_col = None
    for candidate in ['code', 'content', 'source_code', 'snippet', 'solution', 'func']:
        if candidate in df.columns:
            code_col = candidate
            break
    
    if code_col is None:
        # Use first string column with substantial length
        for col in df.columns:
            if df[col].dtype == object:
                avg_len = df[col].astype(str).str.len().mean()
                if avg_len > 50:
                    code_col = col
                    break
    
    if code_col:
        result['code'] = df[code_col].apply(clean_code).values
    else:
        logger.warning("No code column identified in HumanVsAICode")
        result['code'] = ''
    
    # Language
    lang_col = None
    for candidate in ['language', 'lang', 'programming_language']:
        if candidate in df.columns:
            lang_col = candidate
            break
    result['language'] = df[lang_col].values if lang_col else 'unknown'
    
    # No vulnerability labels in this dataset
    result['vuln_label'] = np.nan
    result['cwe'] = np.nan
    result['severity'] = np.nan
    result['source_dataset'] = 'humanvsaicode'
    
    # Task ID
    task_col = None
    for candidate in ['task', 'task_id', 'problem', 'problem_id', 'challenge']:
        if candidate in df.columns:
            task_col = candidate
            break
    result['task_id'] = df[task_col].values if task_col else np.nan
    
    # Provenance - key information from this dataset
    # Check for AI/human labels
    origin_col = None
    for candidate in ['source', 'origin', 'author', 'generated_by', 'model', 
                       'is_ai', 'is_human', 'label', 'type']:
        if candidate in df.columns:
            origin_col = candidate
            break
    
    if origin_col:
        result['_raw_origin'] = df[origin_col].values
        # Map to provenance_type
        result['provenance_type'] = df[origin_col].apply(_map_provenance).values
    else:
        result['provenance_type'] = 'unknown'
    
    # AI generator
    gen_col = None
    for candidate in ['model', 'generator', 'ai_model', 'model_name', 'generated_by']:
        if candidate in df.columns:
            gen_col = candidate
            break
    result['ai_generator'] = df[gen_col].values if gen_col else np.nan
    
    # AI involvement
    result['ai_involvement'] = result['provenance_type'].map({
        'human': 0.0,
        'ai': 1.0,
        'mixed': np.nan,
        'unknown': np.nan
    })
    
    result['human_edit_history'] = np.nan
    result['parent_sample_id'] = np.nan
    result['project'] = np.nan
    result['commit_id'] = np.nan
    
    # Drop _raw_origin if it exists
    if '_raw_origin' in result.columns:
        result = result.drop(columns=['_raw_origin'])
    
    # Filter empty
    result = result[result['code'].str.len() > 10].reset_index(drop=True)
    result['sample_id'] = [f"hvai_{i}" for i in range(len(result))]
    
    logger.info(f"HumanVsAICode standardized: {len(result)} samples, "
                f"provenance distribution: {result['provenance_type'].value_counts().to_dict()}")
    
    return result


def _map_provenance(value) -> str:
    """Map raw provenance labels to standardized categories."""
    if pd.isna(value):
        return 'unknown'
    val = str(value).lower().strip()
    
    human_indicators = ['human', 'manual', 'original', 'handwritten', 'real', '0']
    ai_indicators = ['ai', 'gpt', 'copilot', 'codex', 'chatgpt', 'llm', 'generated',
                     'model', 'claude', 'gemini', 'deepseek', 'starcoder', '1']
    
    for ind in ai_indicators:
        if ind in val:
            return 'ai'
    for ind in human_indicators:
        if ind in val:
            return 'human'
    
    return 'unknown'


def remove_duplicates(df: pd.DataFrame, code_col: str = 'code') -> pd.DataFrame:
    """Remove exact duplicate code samples, keeping the first occurrence."""
    n_before = len(df)
    df = df.drop_duplicates(subset=[code_col], keep='first').reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info(f"Removed {n_removed} duplicate samples ({n_removed/n_before*100:.1f}%)")
    return df


def filter_by_length(df: pd.DataFrame, min_len: int = 20, max_len: int = 50000,
                     code_col: str = 'code') -> pd.DataFrame:
    """Filter out extremely short or long code samples."""
    lengths = df[code_col].str.len()
    mask = (lengths >= min_len) & (lengths <= max_len)
    n_removed = (~mask).sum()
    if n_removed > 0:
        logger.info(f"Filtered {n_removed} samples by length [{min_len}, {max_len}]")
    return df[mask].reset_index(drop=True)


def create_processed_dataset(
    primevul_df: Optional[pd.DataFrame] = None,
    humanvsaicode_df: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Create processed datasets from raw data.
    
    Returns separate DataFrames:
    - 'vulnerability': Samples with vulnerability labels (for security models)
    - 'provenance': Samples with provenance info (for provenance experiments)
    - 'combined': Outer join for exploration (not for direct training)
    
    Does NOT merge incompatible datasets into one training set.
    """
    datasets = {}
    
    if primevul_df is not None:
        vuln_df = standardize_primevul(primevul_df)
        vuln_df = remove_duplicates(vuln_df)
        vuln_df = filter_by_length(vuln_df)
        datasets['vulnerability'] = vuln_df
        logger.info(f"Vulnerability dataset: {len(vuln_df)} samples")
    
    if humanvsaicode_df is not None:
        prov_df = standardize_humanvsaicode(humanvsaicode_df)
        prov_df = remove_duplicates(prov_df)
        prov_df = filter_by_length(prov_df)
        datasets['provenance'] = prov_df
        logger.info(f"Provenance dataset: {len(prov_df)} samples")
    
    return datasets
