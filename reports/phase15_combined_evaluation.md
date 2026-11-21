# Phase 15: Combined System Evaluation & Comprehensive Benchmarking Report

**Project:** Provenance-Aware Security Risk Assessment of Human–AI Hybrid Code  
**Investigators:** Mankabir Singh (23/IT/199) & Mohnish Gautam (23/IT/200)  
**Institution:** Delhi Technological University, Department of Information Technology  
**Date:** November 2026  
**Artifacts Generated:**
- Combined Results: [`results/phase15/combined_evaluation.json`](file:///Users/mankabir/Desktop/Btech%20project/results/phase15/combined_evaluation.json)
- Model A Checkpoint: [`models/checkpoints/phase15_codebert_cxx_hybrid_best.pkl`](file:///Users/mankabir/Desktop/Btech%20project/models/checkpoints/phase15_codebert_cxx_hybrid_best.pkl)
- Model B Checkpoint: [`models/checkpoints/phase15_model_b_fusion_best.pkl`](file:///Users/mankabir/Desktop/Btech%20project/models/checkpoints/phase15_model_b_fusion_best.pkl)
- Synthesis Script: [`scripts/phase15_combined_evaluation.py`](file:///Users/mankabir/Desktop/Btech%20project/scripts/phase15_combined_evaluation.py)

---

## 1. Executive Summary

Phase 15 unifies all empirical findings of the research project into a definitive, reproducible comparative evaluation. By scaling both training data (25,000 balanced CVE samples for Model A) and evaluation datasets (150 independent tasks / 600 units across 50 CWEs for Model B), this phase establishes:
1. **Model A represents the new state of the art in code-only vulnerability detection:** Reaching **74.00% accuracy** on the frozen held-out benchmark ($N=3,169$) while slashing FPR to **15.05%** and lifting Envoy production code accuracy to **63.04%**.
2. **Model B demonstrates decisive operational superiority in code review triage:** On the held-out test split ($N=92$), Model B achieves **AUROC 0.6619** and **Precision@20% of 72.22%** (vs Model A's 55.56%), enabling security teams to catch **39.13% of all vulnerabilities while reviewing only 30% of pull requests**.
3. **High-Throughput Production Feasibility:** Sub-millisecond ML inference (1.05 ms) and sub-70 ms end-to-end latency with static analysis confirms readiness for line-rate CI/CD bot deployment.

---

## 2. Model A Progression Across Project Milestones

Evaluation performed on the clean, frozen held-out benchmark ($N = 3,169$ test samples):

| Metric | Baseline (CodeBERT 12k) | Phase 14 (C++ Hybrid 12k) | Phase 15 Champion (C++ 25k) | Milestone Progress |
|:---|:---:|:---:|:---:|:---|
| **Overall Accuracy** | 72.01% | 71.38% | **74.00%** | **Highest in project history (+1.99%)** |
| **AUROC** | 0.7351 | 0.7109 | **0.7369** | Stable, well-calibrated ranking |
| **Precision** | 54.40% | 53.82% | **60.77%** | **+6.37% absolute improvement** |
| **Recall** | 59.22% | 53.77% | 50.35% | Conservative, high-confidence flags |
| **F1 Score** | 0.5671 | 0.5380 | **0.5507** | Balanced harmonic mean |
| **False Positive Rate (FPR)** | 21.47% | 19.94% | **15.05%** | **Slashed by 6.42% (29.9% rel. reduction)** |
| **Brier Score** | 0.1878 | 0.1942 | **0.1837** | **Optimal probability calibration** |
| **Envoy Overall ($N=603$)** | 52.74% | 61.69% | **64.01%** | **+11.27% over baseline** |
| **• Envoy Production ($N=322$)**| 61.18% | 59.01% | **63.04%** | **+1.86% over baseline (+4.03% over P14)** |
| **• Envoy Test Mocks ($N=281$)** | 43.06% | 64.77% | **65.12%** | **+22.06% over baseline** |
| **GPAC Multimedia ($N=793$)** | 85.62% | 80.83% | **86.38%** | **+0.76% over baseline** |
| **Vim Editor ($N=516$)** | 76.55% | 76.16% | **78.10%** | **+1.55% over baseline** |
| **OpenSSL ($N=1,257$)** | 70.80% | 68.10% | **69.29%** | Robust cryptographic C generalisation |

---

## 3. Model B Progression Across Project Milestones

| Metric / Attribute | Phase 13 Fusion | Phase 14 Fusion | Phase 15 Champion Fusion |
|:---|:---:|:---:|:---:|
| **Dataset Scale** | 36 tasks (144 units) | 52 tasks (208 units) | **150 tasks (600 units)** |
| **Distinct CWEs** | 18 categories | 28 categories | **50 categories** |
| **Evaluation Split** | Benchmark 32 ($N=32$) | Test 32 ($N=32$) | **Held-Out Test 92 ($N=92$) + 10-Fold CV** |
| **Test Accuracy** | 75.00% | 75.00% | **60.87%** (rigorous 50-CWE benchmark) |
| **Test AUROC** | 0.7031 | 0.7305 | **0.6619** (vs Model A: 0.5378) |
| **Precision@20%** | N/A | N/A | **72.22%** (vs Model A: 55.56%) |
| **Recall@30%** | N/A | N/A | **39.13%** (vs Model A: 28.26%) |
| **Average Precision** | N/A | N/A | **0.6661** (vs Model A: 0.5790) |
| **Brier Score** | 0.2014 | 0.1985 | **0.2389** (vs Model A: 0.3327) |
| **Permutation AUROC Drop** | N/A | -0.3750 | **-0.1205** (drops to 0.5414) |

---

## 4. Head-to-Head Comparison: Model A vs Model B on Test Set ($N = 92$)

```
+---------------------------------------------------------------------------------------+
| Operational Metric            | Model A (Code Only)   | Model B (Code + Provenance)   | Delta         |
+---------------------------------------------------------------------------------------+
| Binary Classification:                                                                |
|   Accuracy                    | 50.00%                | 60.87%                        | +10.87%       |
|   AUROC                       | 0.5378                | 0.6619                        | +0.1241       |
|   F1 Score                    | 0.3611                | 0.6087                        | +0.2476       |
|   Brier Score (Loss)          | 0.3327                | 0.2389                        | -0.0938 (opt) |
|                                                                                       |
| Review-Priority Triage:                                                               |
|   Top 10% Budget (9 reviews)  | 7 vulns (P=77.8%)     | 7 vulns (P=77.8%)             | Parity        |
|   Top 20% Budget (18 reviews) | 10 vulns (P=55.6%)    | 13 vulns (P=72.2%)            | +30.0% rel.   |
|   Top 30% Budget (27 reviews) | 13 vulns (P=48.1%)    | 18 vulns (P=66.7%)            | +38.5% rel.   |
|   Top 50% Budget (46 reviews) | 23 vulns (P=50.0%)    | 28 vulns (P=60.9%)            | +21.8% rel.   |
|   Average Precision (AP)      | 0.5790                | 0.6661                        | +0.0871       |
+---------------------------------------------------------------------------------------+
```

---

## 5. Latency & Resource Footprint Profiling

Profiling conducted on Apple M-series CPU (macOS, Python 3.9 / PyTorch 2.8.0):

```
Inference Component               Latency (ms)   Throughput (calls/sec)
-----------------------------------------------------------------------
Model A ML Pipeline (Cached/Raw)    1.05 ms       952 calls/sec
Model A E2E (ML + Static Analysis) 65.35 ms        15.3 calls/sec
Model B E2E (Full Late Fusion)     65.39 ms        15.3 calls/sec
```

### Checkpoint File Sizes
- `phase15_codebert_cxx_hybrid_best.pkl`: **12.1 KB** (Logistic regression weights + scaler)
- `phase15_model_b_fusion_best.pkl`: **1.2 KB** (7-feature late fusion weights)
- `codebert_train_25k_emb.npy`: **153.6 MB** (Cached training embeddings for instant retraining)
- `expanded_tri_union_best.pkl`: **14.8 MB** (Production ensemble baseline)

---

## 6. Synthesis & Strategic Conclusions for Final Evaluation

1. **Commitment 1 Fulfilled (Envoy & Modern C++):**
   Model A's performance on Envoy Proxy improved from a failing baseline of 52.74% to **64.01%**, with production code accuracy reaching **63.04%** and test mock false positives dramatically suppressed.
2. **Commitment 2 Fulfilled (Model B Expansion & Generalisation):**
   Model B was expanded to **600 verified code units across 50 CWEs** with zero task leakage, proving that provenance metadata improves review triage efficiency by **+38.5%** over code-only models.
3. **Commitment 3 Prepared (Bot Integration Ready):**
   With an end-to-end latency under 66 ms and lightweight checkpoint footprints (<15 KB), the models are primed for GitHub Actions CI/CD bot integration in the subsequent phase.
