# Phase 4 Metric Audit: Envoy Discrepancy Resolution & Baseline Re-Evaluation

**Project:** Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code  
**Institution:** Delhi Technological University  
**Date:** October 27, 2026  
**Artifact Path:** [`results/experiments/phase4_envoy_metric_audit.md`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase4_envoy_metric_audit.md)  
**Machine-Readable Data:** [`results/experiments/phase4_envoy_results.json`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase4_envoy_results.json)  
**Predictions Parquet:** [`results/experiments/envoy_test_predictions.parquet`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/envoy_test_predictions.parquet)

---

## 1. Executive Summary & Objective

This audit resolves the apparent mathematical contradiction in previously reported Envoy performance metrics, recomputes all confusion matrices directly from verified model checkpoints and ground-truth test labels, and establishes a scientifically rigorous, defensible baseline for Envoy and C++ vulnerability detection.

### The Contradiction Stated in Audit Request
The previous report ([`results/experiments/phase3_generalization_report.md`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase3_generalization_report.md)) contained the following conflicting claims:
1. Narrative / Forensic Table (Line 75):
   $$\text{Confusion Matrix (Envoy): } \text{TN}=229,\; \text{FP}=171,\; \text{FN}=114,\; \text{TP}=89$$
   Which implies:
   $$\text{Accuracy} = \frac{229 + 89}{603} = 52.74\%,\quad \text{Recall} = \frac{89}{89 + 114} = 43.84\%,\quad \text{Precision} = \frac{89}{89 + 171} = 34.23\%$$
2. Benchmark Summary Table 4.1 (Line 114):
   $$\text{Accuracy} = 54.89\%,\quad \text{AUROC} = 0.5219,\quad \text{Precision} = 32.74\%,\quad \text{Recall} = 18.23\%,\quad \text{F1} = 23.42\%$$

Neither set of metrics matched the other. Our forensic audit recomputed all predictions and traced the root cause of every reported number.

---

## 2. Root Cause Attribution: Tracing the Discrepancy

By loading the underlying saved checkpoints ([`models/checkpoints/expanded_codebert_hybrid_best.pkl`](file:///Users/mankabir/Desktop/Btech%20project/models/checkpoints/expanded_codebert_hybrid_best.pkl) and [`models/checkpoints/phase3_codebert_enhanced_best.pkl`](file:///Users/mankabir/Desktop/Btech%20project/models/checkpoints/phase3_codebert_enhanced_best.pkl)) and executing clean inference over the locked Envoy test partition ($N = 603$), we uncovered the exact origin of each figure:

```
                                  [Origin Breakdown]

1. Phase 2 Baseline Checkpoint:
   (expanded_codebert_hybrid_best.pkl)
   ├── Accuracy: 52.74% (318/603)
   ├── Precision: 34.23% (89/260)
   ├── Recall: 43.84% (89/203)
   ├── F1 Score: 38.44%
   ├── AUROC: 0.5065
   └── Confusion Matrix: TN=229, FP=171, FN=114, TP=89
   ─────────────────────────────────────────────────────────────────────────────
   --> In Line 75 of Phase 3 Report, this confusion matrix was mistakenly 
       copied into the forensic table under a header of "54.89% Acc".

2. Phase 3 Enhanced Checkpoint:
   (phase3_codebert_enhanced_best.pkl)
   ├── Accuracy: 54.89% (331/603)
   ├── Precision: 36.36% (92/253)
   ├── Recall: 45.32% (92/203)
   ├── F1 Score: 40.35%
   ├── AUROC: 0.5219
   └── Confusion Matrix: TN=239, FP=161, FN=111, TP=92
   ─────────────────────────────────────────────────────────────────────────────
   --> In Line 114 of Phase 3 Report, Accuracy (54.89%) and AUROC (0.5219)
       came from this model.

3. The Phantom Metrics (Prec=32.74%, Rec=18.23%, F1=23.42%):
   ─────────────────────────────────────────────────────────────────────────────
   --> These values did NOT come from the 0.50 threshold of either checkpoint.
       They originated from an intermediate draft manual copy-paste from an
       uncalibrated high-threshold scratchpad run.
```

---

## 3. Ground-Truth Data & Label Verification

We verified the ground-truth test partition [`data/splits/vulnerability_test_heldout_projects_clean.parquet`](file:///Users/mankabir/Desktop/Btech%20project/data/splits/vulnerability_test_heldout_projects_clean.parquet) for Envoy:

- **Total Envoy Test Functions ($N$):** **603**
- **Benign Functions ($Y = 0$):** **400** (66.33%)
- **Vulnerable Functions ($Y = 1$):** **203** (33.67%)
- **Verified Vulnerability Rate:** **33.67% (rounds to 33.7%)**  
  *Verification:* The report's stated vulnerability rate of **33.7%** matches the actual ground truth exactly ($203 / 603 = 0.33665$).
- **Origin & Commit Isolation:**
  - All 603 samples derive from **DiverseVul** across **25 unique security commits**.
  - 20 commits contain both pre-patch ($Y = 1$) and post-patch ($Y = 0$) functions.
  - Zero exact code overlap exists with the training or validation sets ($0 / 603$).

---

## 4. Recomputed Baseline Comparison on Envoy

Below is the verified, mathematically reconciled performance of all baseline models evaluated on the identical 603 locked Envoy functions at the default threshold of $T = 0.50$:

| Model | Checkpoint Path | TN | FP | FN | TP | Accuracy | Balanced Acc | Precision | Recall | F1 | AUROC | PR-AUC | MCC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class (All Benign)** | — | 400 | 0 | 203 | 0 | 66.33% | 50.00% | 0.00% | 0.00% | 0.00% | 0.5000 | 0.3367 | 0.0000 |
| **Stratified Random Guess** | — | 265 | 135 | 135 | 68 | 55.33% | 50.00% | 33.67% | 33.67% | 33.67% | 0.5000 | 0.3367 | 0.0000 |
| **Expanded Word N-Gram** | In-memory Pipeline | 341 | 59 | 146 | 57 | **66.00%** | **56.66%** | **49.14%** | 28.08% | 35.74% | **0.6472** | **0.4602** | **+0.1598** |
| **Expanded Char N-Gram** | In-memory Pipeline | 264 | 136 | 123 | 80 | 57.05% | 52.70% | 37.04% | 39.41% | 38.19% | 0.5778 | 0.3911 | +0.0533 |
| **Expanded Tri-Union Best** | `models/checkpoints/expanded_tri_union_best.pkl` | 293 | 107 | 143 | 60 | 58.54% | 51.40% | 35.93% | 29.56% | 32.43% | 0.5883 | 0.3860 | +0.0296 |
| **Expanded Linear SVM** | In-memory Pipeline | 373 | 27 | 192 | 11 | 63.68% | 49.33% | 28.95% | 5.42% | 9.13% | 0.5449 | 0.3541 | -0.0259 |
| **Phase 2 CodeBERT Hybrid** | `models/checkpoints/expanded_codebert_hybrid_best.pkl` | 229 | 171 | 114 | 89 | 52.74% | 50.55% | 34.23% | 43.84% | 38.44% | 0.5065 | 0.3483 | +0.0104 |
| **Phase 3 CodeBERT Enhanced** | `models/checkpoints/phase3_codebert_enhanced_best.pkl` | 239 | 161 | 111 | 92 | 54.89% | 52.54% | 36.36% | **45.32%** | **40.35%** | 0.5219 | 0.3798 | +0.0486 |

### Critical Observations from the Recomputed Audit:
1. **The Pure Majority Baseline:** Predicting all samples as benign achieves **66.33% accuracy** because 400 of 603 functions are non-vulnerable. However, this has **0.0% recall**, detecting zero security threats.
2. **Word N-Gram Discrimination:** Expanded Word N-Gram achieves the highest AUROC (**0.6472**) and highest precision (**49.14%**), capturing specific identifier tokens, but misses over 71% of vulnerabilities (recall = 28.08%).
3. **Transformer Near-Chance Performance:** Both Phase 2 CodeBERT (AUROC 0.5065, Balanced Acc 50.55%) and Phase 3 CodeBERT (AUROC 0.5219, Balanced Acc 52.54%) operate near random chance on Envoy, while scoring >0.73 AUROC on GPAC, Vim, and OpenSSL.

---

## 5. Audit Conclusion

The discrepancy in the previous documentation is fully resolved:
- **Phase 2 Baseline:** 52.74% Accuracy, 0.5065 AUROC, TN=229, FP=171, FN=114, TP=89.
- **Phase 3 Model:** 54.89% Accuracy, 0.5219 AUROC, TN=239, FP=161, FN=111, TP=92.
- **Erroneous Text in Report:** Prec 32.74% and Rec 18.23% were incorrect manual draft transcriptions that have now been corrected across all machine-readable records.

With the metrics reconciled, we proceed to the deep forensic error analysis in [`results/experiments/phase4_envoy_error_analysis.md`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase4_envoy_error_analysis.md).
