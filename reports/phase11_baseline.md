# Phase 11 Baseline Reproduction Report

**Project Title:** Provenance-Aware Security Risk Assessment of Human–AI Hybrid Code  
**Institution:** Delhi Technological University, Department of Information Technology  
**Authors:** Mankabir Singh (23/IT/199) and Mohnish Gautam (23/IT/200)  
**Date:** October 27, 2026  
**Status:** Cryptographically Verified and Empirically Reproduced  

---

## 1. Executive Summary

This report establishes the baseline evaluation metrics and records all active architectures, feature schemas, checkpoint hashes, dataset splits, and inference behaviors prior to executing Phase 11 model improvements.

All baseline evaluations were executed directly on disk using the active production checkpoints and genuine split manifests without hardcoding, manual rule override, or dataset manipulation.

### Key Baseline Findings:
1. **Model A (Code-Only):**
   - **Cross-Project Generalisation Test ($N=3,169$, strictly clean & decontaminated across OpenSSL, GPAC, Envoy, Vim):** Exactly **64.31% Accuracy**, **0.6852 AUROC**, **57.23% Recall**, **44.98% Precision**, **+0.2354 MCC**.
   - **In-Distribution Validation Split ($N=23,648$):** **76.44% Accuracy** (at standard threshold 0.50, matching 75.84% at calibrated threshold), **0.8326 AUROC**, **75.29% Recall**, **46.03% Precision**, **+0.4459 MCC**.
   - **The Domain Gap:** The 12.13 percentage point drop from in-distribution validation (76.44%) to cross-project generalization (64.31%) reflects genuine domain shift across unseen codebases.
2. **Model B (Code + Provenance):**
   - **Development Validation Split ($N=16$, disjoint from locked benchmark):** **100.00% Accuracy**, **1.0000 AUROC**, **100.00% Precision**, **100.00% Recall**.
   - **Locked Benchmark ($N=32$):** **96.88% Accuracy**, **0.9844 AUROC**, **100.00% Precision**, **93.75% Recall**, **+0.9393 MCC**.
   - **Provenance-Unknown Fallback:** Yields identical probabilities ($|P_B - P_A| = 0.000000$) and 56.25% accuracy, matching Model A exactly.
   - **Audit Limitation Confirmed:** As established in Phase 10.1, Model B's high accuracy on the 32-sample benchmark is driven by the 100% correlation between AI-origin code and vulnerability in the benchmark's construction.

---

## 2. Active Model Inventory & Cryptographic Checksum Verification

| Component | Model Description | Checkpoint Path | SHA-256 Checksum | Feature Dimension / Schema |
| :--- | :--- | :--- | :--- | :--- |
| **Model A** | Tri-Union Classical Security Classifier | [`models/checkpoints/expanded_tri_union_best.pkl`](file:///Users/mankabir/Desktop/Btech%20project/models/checkpoints/expanded_tri_union_best.pkl) | `a5c62fe77f1b5a9fd42aed1843f664a2d4ba59dcf31f27cb1857afd1266994db` | Word TF-IDF (12,000) + Char TF-IDF (18,000) + Structural Flaw AST (10) = 30,010 dims |
| **Model B** | Calibrated Logistic Provenance Fusion | [`models/checkpoints/model_b_provenance_fusion.pkl`](file:///Users/mankabir/Desktop/Btech%20project/models/checkpoints/model_b_provenance_fusion.pkl) | `e0e9d712fa3dc6a41ff01cdc869ad64b28aa4d7a0833611dbbeb99a1789bdbbd` | `[logit_p_code, ai_involvement, is_human_edited, is_pure_ai]` |
| **Deep Baseline** | CodeBERT Token Classifier | [`models/checkpoints/codebert_vuln_best.pt`](file:///Users/mankabir/Desktop/Btech%20project/models/checkpoints/codebert_vuln_best.pt) | `76d62e1d7f956b7be70fe49b7e358236e75b068e5dca8b76909ebab625c2b3ca` | 768-dim CLS embeddings |
| **Hybrid Baseline** | Expanded CodeBERT + AST Hybrid | [`models/checkpoints/expanded_codebert_hybrid_best.pkl`](file:///Users/mankabir/Desktop/Btech%20project/models/checkpoints/expanded_codebert_hybrid_best.pkl) | `df74f91cec7d924c666803b13ae9eb7e8cb0414d83aaa0c770aa0e4dd0e585d0` | 768-dim CodeBERT + 10-dim Structural AST = 778 dims |

---

## 3. Detailed Model A Baseline Metrics

### 3.1 Held-Out Cross-Project Evaluation ($N=3,169$)
Evaluated on [`data/splits/vulnerability_test_heldout_projects_clean.parquet`](file:///Users/mankabir/Desktop/Btech%20project/data/splits/vulnerability_test_heldout_projects_clean.parquet), strictly decontaminated with 0 code overlap against training and validation partitions:
- **Total Samples:** 3,169
- **Safe Functions ($y=0$):** 2,166 (68.35%)
- **Vulnerable Functions ($y=1$):** 1,003 (31.65%)
- **Majority-Class Baseline:** 68.35% (predicting all safe)

| Metric | Measured Value | Standard Error / Notes |
| :--- | :---: | :--- |
| **Accuracy** | **64.31%** (2,038 / 3,169) | $95\%\text{ CI: } [62.6\%, 66.0\%]$ |
| **Balanced Accuracy** | **62.41%** | $\frac{1}{2}(\text{TPR} + \text{TNR})$ |
| **Precision** | **44.98%** (574 / 1,276) | Low precision due to domain false positives |
| **Recall (TPR)** | **57.23%** (574 / 1,003) | Misses 429 vulnerabilities |
| **F1 Score** | **0.5037** | Harmonic mean of precision and recall |
| **Matthews Correlation (MCC)**| **+0.2354** | Non-trivial positive correlation |
| **AUROC** | **0.6852** | Discriminative power across thresholds |
| **PR-AUC (Average Precision)** | **0.4918** | Benchmark baseline: 0.3165 |
| **Brier Score** | **0.2203** | Mean squared error of probabilities |
| **Confusion Matrix** | $\begin{bmatrix} 1464 & 702 \\ 429 & 574 \end{bmatrix}$ | $TN=1464, FP=702, FN=429, TP=574$ |

#### Per-Repository Breakdown:
- **OpenSSL ($N=1,257$):** 67.70% Accuracy | 65.92% Recall | 0.7394 AUROC (High C code consistency)
- **Vim ($N=516$):** 67.44% Accuracy | 47.50% Recall | 0.6812 AUROC (String manipulation patterns)
- **GPAC ($N=793$):** 61.29% Accuracy | 74.75% Recall | 0.7061 AUROC (Multimedia parser buffer flows)
- **Envoy ($N=603$):** 58.54% Accuracy | 29.56% Recall | 0.5883 AUROC (Modern C++ domain divergence)

---

### 3.2 In-Distribution Validation Evaluation ($N=23,648$)
Evaluated on [`data/splits/vulnerability_val_expanded.parquet`](file:///Users/mankabir/Desktop/Btech%20project/data/splits/vulnerability_val_expanded.parquet), partitioned by commit-grouping from the 138k expanded corpus:
- **Total Samples:** 23,648
- **Safe Functions ($y=0$):** 18,718 (79.15%)
- **Vulnerable Functions ($y=1$):** 4,930 (20.85%)

| Metric | Measured Value | Standard Error / Notes |
| :--- | :---: | :--- |
| **Accuracy** | **76.44%** (18,077 / 23,648) | Matches 75.84% reported under threshold calibration |
| **Balanced Accuracy** | **76.02%** | High balance across imbalanced distribution |
| **Precision** | **46.03%** (3,712 / 8,065) | Flags 4,353 false positives out of 18,718 safe |
| **Recall (TPR)** | **75.29%** (3,712 / 4,930) | Detects over 75% of known vulnerabilities |
| **F1 Score** | **0.5713** | Substantially higher than cross-project F1 |
| **Matthews Correlation (MCC)**| **+0.4459** | Strong positive correlation |
| **AUROC** | **0.8326** | Strong discriminative ability in-distribution |
| **PR-AUC** | **0.5609** | Baseline: 0.2085 |
| **Brier Score** | **0.1642** | Well-calibrated probabilities |
| **Confusion Matrix** | $\begin{bmatrix} 14365 & 4353 \\ 1218 & 3712 \end{bmatrix}$ | $TN=14365, FP=4353, FN=1218, TP=3712$ |

---

## 4. Model B Baseline Metrics

### 4.1 Development Validation Split ($N=16$)
Evaluated on [`data/model_b_dev/model_b_dev_val.json`](file:///Users/mankabir/Desktop/Btech%20project/data/model_b_dev/model_b_dev_val.json):
- **Total Samples:** 16 (8 Safe, 8 Vulnerable across 4 disjoint tasks `dev_task_021` to `dev_task_024`)
- **Model A Alone:** 68.75% Accuracy (11/16) | 37.50% Recall (3/8) | 100.00% Precision (3/3) | CM: `[[8, 0], [5, 3]]`
- **Model B (Code + Provenance):** **100.00% Accuracy** (16/16) | **100.00% Recall** (8/8) | **100.00% Precision** (8/8) | **1.0000 AUROC** | CM: `[[8, 0], [0, 8]]`

### 4.2 Locked Benchmark ($N=32$)
Evaluated on [`data/locked_benchmark_32/controlled_benchmark_32.json`](file:///Users/mankabir/Desktop/Btech%20project/data/locked_benchmark_32/controlled_benchmark_32.json):
- **Total Samples:** 32 (16 Safe, 16 Vulnerable across 8 tasks)
- **Model A Alone:** 56.25% Accuracy (18/32) | 87.50% Recall (14/16) | 53.85% Precision (14/26) | CM: `[[4, 12], [2, 14]]`
- **Model B (Code + Provenance):** **96.88% Accuracy** (31/32) | **93.75% Recall** (15/16) | **100.00% Precision** (15/15) | **0.9844 AUROC** | CM: `[[16, 0], [1, 15]]`
- **Model B (Unknown Fallback):** **56.25% Accuracy** (18/32) | Matches Model A with 0 discrepancy ($|P_B - P_A| = 0.000000$).

---

## 5. Audit Baseline Summary & Goals for Phase 11

| Evaluation Target | Current Baseline | Phase 11 Target | Status / Gap |
| :--- | :---: | :---: | :--- |
| **Model A (Held-Out Cross-Project Test)** | 64.31% | $\ge 70.0\%$ | Needs $+5.69$ percentage points improvement on unseen projects |
| **Model B (Genuinely Held-Out Evaluation Set)** | 96.88% (on correlated benchmark) | $\ge 70.0\%$ (on balanced lineage) | Model B must maintain $\ge 70\%$ when AI code is NOT 100% buggy and human code is NOT 100% safe |
| **Benchmark Integrity** | SHA-256 Verified | Immutable | Zero training, tuning, or leakage on `data/locked_benchmark_32/` |
