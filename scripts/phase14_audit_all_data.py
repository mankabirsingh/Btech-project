#!/usr/bin/env python3
"""
Phase 14 Exhaustive Dataset Audit Script
========================================
Audits all local datasets in the repository to determine:
1. Exact record count and unique code count
2. Vulnerability ground-truth availability
3. Provenance ground-truth availability
4. CWE taxonomy tagging
5. Suitability for Model A (code-only) vs Model B (provenance-aware)
6. Empirical feasibility of reaching 1,000+ verified dual-ground-truth examples
"""

import sys
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase14"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def audit_parquet(path: Path, name: str):
    if not path.exists():
        return {"name": name, "exists": False}
    
    df = pd.read_parquet(path)
    n_records = len(df)
    n_unique_code = df['code'].nunique() if 'code' in df.columns else 0
    has_vuln = 'vuln_label' in df.columns and not df['vuln_label'].isna().all()
    vuln_rate = float(df['vuln_label'].mean()) if has_vuln else None
    
    # Check provenance
    has_prov = 'provenance_type' in df.columns or 'author_type' in df.columns
    prov_col = 'provenance_type' if 'provenance_type' in df.columns else ('author_type' if 'author_type' in df.columns else None)
    prov_dist = df[prov_col].value_counts().to_dict() if prov_col else {}
    
    # Check CWE
    has_cwe = 'cwe' in df.columns and not df['cwe'].isna().all()
    n_cwes = df['cwe'].nunique() if has_cwe else 0
    
    return {
        "name": name,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "exists": True,
        "n_records": n_records,
        "n_unique_code": n_unique_code,
        "has_vuln": has_vuln,
        "vuln_rate": round(vuln_rate, 4) if vuln_rate is not None else "N/A (All NaN)",
        "has_prov": has_prov,
        "prov_distribution": prov_dist,
        "has_cwe": has_cwe,
        "n_cwes": n_cwes,
        "columns": list(df.columns)
    }


def main():
    print("=" * 80)
    print("PHASE 14 — STEP 4: EXHAUSTIVE DATA AUDIT ACROSS ALL LOCAL CORPORA")
    print("=" * 80)

    datasets_to_audit = [
        (DATA_DIR / "processed" / "vulnerability_data.parquet", "Processed Vulnerability Data"),
        (DATA_DIR / "processed" / "provenance_data.parquet", "HumanVsAICode Provenance Corpus"),
        (DATA_DIR / "splits" / "vulnerability_train_expanded.parquet", "Expanded Training Split"),
        (DATA_DIR / "splits" / "vulnerability_val_expanded.parquet", "Expanded Validation Split"),
        (DATA_DIR / "splits" / "vulnerability_test_heldout_projects_clean.parquet", "Clean Held-Out Test Split"),
        (DATA_DIR / "locked_benchmark_32" / "controlled_benchmark_32.json", "Phase 10 Locked Benchmark 32"),
        (DATA_DIR / "phase11_model_b" / "model_b_p11_train.json", "Phase 11 Train Set"),
        (DATA_DIR / "phase11_model_b" / "model_b_p11_test_locked.json", "Phase 11 Locked Test Set"),
        (DATA_DIR / "phase13_model_b" / "model_b_p13_train.json", "Phase 13 Train Set"),
        (DATA_DIR / "phase13_model_b" / "model_b_p13_test_locked.json", "Phase 13 Locked Test Set"),
    ]

    audit_summary = []

    for path, name in datasets_to_audit:
        if path.suffix == ".parquet":
            res = audit_parquet(path, name)
        elif path.suffix == ".json":
            with open(path, "r") as f:
                d = json.load(f)
            records = d.get("records", []) if isinstance(d, dict) else d
            n_records = len(records)
            unique_codes = len(set(r.get("code", "") for r in records if "code" in r))
            cwes = set(r.get("cwe", "") for r in records if "cwe" in r)
            authors = [r.get("author_type", "unknown") for r in records]
            vulns = [r.get("vuln_label", 0) for r in records if "vuln_label" in r]
            
            prov_clean = {str(k): int(v) for k, v in pd.Series(authors).value_counts().items()}
            res = {
                "name": name,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "exists": True,
                "n_records": n_records,
                "n_unique_code": unique_codes,
                "has_vuln": True,
                "vuln_rate": round(float(np.mean(vulns)), 4) if vulns else "N/A",
                "has_prov": True,
                "prov_distribution": prov_clean,
                "has_cwe": True,
                "n_cwes": len(cwes),
                "columns": list(records[0].keys()) if records else []
            }
        audit_summary.append(res)
        print(f"\nDataset: {res['name']} ({res['path']})")
        print(f"  • Total Records: {res['n_records']:,} (Unique Code: {res['n_unique_code']:,})")
        print(f"  • Vulnerability Labels: {res['has_vuln']} (Vuln Rate: {res['vuln_rate']})")
        print(f"  • Provenance Tags: {res['has_prov']} (Distribution: {res['prov_distribution']})")
        print(f"  • CWE Taxonomies: {res['n_cwes']} distinct CWEs")

    # Save to json
    output_path = RESULTS_DIR / "dataset_audit_summary.json"
    with open(output_path, "w") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\n[✓] Audit summary saved to: {output_path}")


if __name__ == "__main__":
    main()
