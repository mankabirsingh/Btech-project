#!/usr/bin/env python3
"""
Phase 15 — Step 6: Combined System Evaluation & Comprehensive Benchmarking
==========================================================================
Produces unified, rigorous comparative evaluation across:
1. Model A Progression:
   - Historical Production Baseline (12k expanded tri-union)
   - Phase 14 Modern C++ Hybrid (small-scale)
   - Phase 15 Modern C++ 25k CodeBERT Hybrid Champion
2. Model B Progression:
   - Phase 13 Provenance Fusion (Benchmark 32)
   - Phase 14 Interaction Fusion (Test 32)
   - Phase 15 Champion Fusion (Test 92 & 10-Fold CV across 150 Tasks / 600 Units)
3. Model A vs Model B Direct Comparison on Test Split (N = 92):
   - Task 1: Binary Vulnerability Classification (Acc, AUROC, F1, MCC, Brier)
   - Task 2: Review-Priority Ranking & Triage (P@10%, P@20%, Recall@20%, Recall@30%, Recall@50%, AP)
4. Full 7-Regime Ablation Suite Analysis (Contribution of Code vs Provenance)
5. End-to-End Latency & Resource Footprint Profiling
"""

import sys
import os
import json
import time
import pickle
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_P15_DIR = PROJECT_ROOT / "results" / "phase15"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"


def measure_inference_latencies() -> Dict[str, Any]:
    """Measures end-to-end inference latencies for Model A and Model B."""
    from src.models.inference import predict_model_a, predict_model_b, _get_model_a_pipeline, clean_code
    
    sample_c = """
    int process_buffer(const char *input, size_t len) {
        char dest[64];
        if (len >= sizeof(dest)) return -1;
        strncpy(dest, input, sizeof(dest) - 1);
        dest[sizeof(dest) - 1] = '\\0';
        return 0;
    }
    """
    sample_prov = {
        "author_type": "human_edit_of_ai",
        "ai_involvement": 0.65,
        "provenance_confidence": 0.95
    }

    # Warmup
    predict_model_a(sample_c)
    predict_model_b(sample_c, sample_prov)

    # 1. Model A ML Pipeline alone
    pipe = _get_model_a_pipeline()
    cleaned = clean_code(sample_c)
    t0 = time.time()
    n_runs = 50
    for _ in range(n_runs):
        _ = pipe.predict_proba([cleaned])[0, 1]
    ml_alone_ms = (time.time() - t0) / n_runs * 1000.0

    # 2. Model A End-to-End (ML + Flawfinder/Pattern Static Analysis)
    t0 = time.time()
    n_e2e = 20
    for _ in range(n_e2e):
        _ = predict_model_a(sample_c)
    model_a_e2e_ms = (time.time() - t0) / n_e2e * 1000.0

    # 3. Model B End-to-End
    t0 = time.time()
    for _ in range(n_e2e):
        _ = predict_model_b(sample_c, sample_prov)
    model_b_e2e_ms = (time.time() - t0) / n_e2e * 1000.0

    # Checkpoint file sizes
    sizes = {}
    for fname in [
        "expanded_tri_union_best.pkl",
        "expanded_codebert_hybrid_best.pkl",
        "phase14_codebert_cxx_hybrid.pkl",
        "phase15_codebert_cxx_hybrid_best.pkl",
        "model_b_provenance_fusion.pkl",
        "phase14_model_b_fusion_best.pkl",
        "phase15_model_b_fusion_best.pkl"
    ]:
        p = MODELS_DIR / fname
        if p.exists():
            sizes[fname] = {
                "size_bytes": p.stat().st_size,
                "size_mb": round(p.stat().st_size / (1024 * 1024), 2)
            }

    return {
        "latencies": {
            "model_a_ml_pipeline_ms": round(ml_alone_ms, 2),
            "model_a_e2e_ms": round(model_a_e2e_ms, 2),
            "model_b_e2e_ms": round(model_b_e2e_ms, 2)
        },
        "checkpoint_sizes": sizes
    }


def main():
    print("=" * 80)
    print("PHASE 15 — STEP 6: COMBINED SYSTEM EVALUATION & SYNTHESIS")
    print("=" * 80)

    # 1. Load Model A Comparison Results & Baselines
    with open(RESULTS_P15_DIR / "phase15_model_a_comparison.json") as f:
        model_a_data = json.load(f)
    with open(RESULTS_P15_DIR / "baseline_reproduction.json") as f:
        baseline_data = json.load(f)

    # 2. Load Model B Evaluation Results
    with open(RESULTS_P15_DIR / "phase15_model_b_evaluation.json") as f:
        model_b_data = json.load(f)

    # 3. Load Real-World Provenance Audit
    with open(RESULTS_P15_DIR / "real_world_provenance_audit.json") as f:
        rw_audit = json.load(f)

    # 4. Measure System Latencies & Memory Footprint
    print("\n[1] Benchmarking System Latencies and Checkpoint Footprints...")
    sys_profile = measure_inference_latencies()
    print(f"  Model A ML Pipeline: {sys_profile['latencies']['model_a_ml_pipeline_ms']} ms/call")
    print(f"  Model A E2E (with static): {sys_profile['latencies']['model_a_e2e_ms']} ms/call")
    print(f"  Model B E2E: {sys_profile['latencies']['model_b_e2e_ms']} ms/call")

    # 5. Compile Model A Progression Table
    p15_a = model_a_data["Candidate 4: Phase 15 Modern C++ Hybrid (25k Balanced)"]
    p14_a = baseline_data["phase14_codebert_cxx_hybrid"]
    base_a = baseline_data["expanded_codebert_hybrid_best"]

    model_a_progression = {
        "metrics_clean_heldout_n3169": {
            "Baseline (CodeBERT 12k)": {
                "accuracy": base_a["overall"]["accuracy"],
                "auroc": base_a["overall"]["auroc"],
                "f1": base_a["overall"]["f1"],
                "fpr": base_a["overall"]["fpr"],
                "envoy_overall": base_a["by_project"]["envoy"]["accuracy"],
                "envoy_production": base_a["envoy_subcohorts"]["production"]["accuracy"],
                "envoy_mocks": base_a["envoy_subcohorts"]["gtest"]["accuracy"],
                "gpac": base_a["by_project"]["gpac"]["accuracy"],
                "vim": base_a["by_project"]["vim"]["accuracy"],
                "openssl": base_a["by_project"]["openssl"]["accuracy"]
            },
            "Phase 14 (CodeBERT C++ 12k)": {
                "accuracy": p14_a["overall"]["accuracy"],
                "auroc": p14_a["overall"]["auroc"],
                "f1": p14_a["overall"]["f1"],
                "fpr": p14_a["overall"]["fpr"],
                "envoy_overall": p14_a["by_project"]["envoy"]["accuracy"],
                "envoy_production": p14_a["envoy_subcohorts"]["production"]["accuracy"],
                "envoy_mocks": p14_a["envoy_subcohorts"]["gtest"]["accuracy"],
                "gpac": p14_a["by_project"]["gpac"]["accuracy"],
                "vim": p14_a["by_project"]["vim"]["accuracy"],
                "openssl": p14_a["by_project"]["openssl"]["accuracy"]
            },
            "Phase 15 Champion (CodeBERT C++ 25k)": {
                "accuracy": p15_a["overall"]["accuracy"],
                "auroc": p15_a["overall"]["auroc"],
                "f1": p15_a["overall"]["f1"],
                "fpr": p15_a["overall"]["fpr"],
                "envoy_overall": p15_a["by_project"]["envoy"]["accuracy"],
                "envoy_production": p15_a["envoy_subcohorts"]["production_code"]["accuracy"],
                "envoy_mocks": p15_a["envoy_subcohorts"]["gtest_scaffolding"]["accuracy"],
                "gpac": p15_a["by_project"]["gpac"]["accuracy"],
                "vim": p15_a["by_project"]["vim"]["accuracy"],
                "openssl": p15_a["by_project"]["openssl"]["accuracy"]
            }
        }
    }

    # 6. Compile Model B Progression & Comparison Table
    abl_test = model_b_data["ablations_test_92"]
    cv_sum = model_b_data["grouped_cv_summary_150_tasks"]

    combined_synthesis = {
        "model_a_progression": model_a_progression,
        "model_b_ablations_test_92": abl_test,
        "model_b_grouped_cv_150_tasks": cv_sum,
        "triage_comparison_test_92": {
            "Model A Code Only": abl_test["Model A: Code Only"]["triage"],
            "Model B Full Fusion": abl_test["Model B: Full (Code + Prov)"]["triage"]
        },
        "system_profile": sys_profile,
        "real_world_provenance_audit_summary": {
            "mined_repositories": len(rw_audit.get("sampled_repositories", [])),
            "verified_records_held_out": rw_audit.get("verified_real_world_records_held_out", 0),
            "finding": rw_audit.get("finding", "")
        }
    }

    out_comb_path = RESULTS_P15_DIR / "combined_evaluation.json"
    with open(out_comb_path, "w") as f:
        json.dump(combined_synthesis, f, indent=2)

    print(f"\n[✓] Combined evaluation saved to: {out_comb_path}")


if __name__ == "__main__":
    main()
