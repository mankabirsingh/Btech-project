# Phase 16 — Experiment 2: C++ & Cross-Project Stratified 35k Training Report

**Project:** Provenance-Aware Security Risk Assessment of Human–AI Hybrid Code  
**Team:** Mankabir Singh (23/IT/199) & Mohnish Gautam (23/IT/200)  
**Institution:** Delhi Technological University, Department of Information Technology  
**Date:** November 19, 2026  
**Artifact Path:** `reports/phase16_stage2_cpp_enrichment.md`  
**Checkpoint:** `models/checkpoints/phase16_codebert_cxx_stratified_35k.pkl`  
**Results Data:** `results/phase16/exp2_cpp_stratified_training.json`  

---

## 1. Executive Summary & Objective

In Phase 15, error analysis revealed that the uniform random 25,000-sample training partition contained **only 1 single Envoy code snippet (<0.005%)** and less than 6% C++ code overall. Remarkably, Model A's 64.01% Envoy accuracy was achieved almost purely through zero-shot cross-project transfer from general C repositories.

The objective of **Experiment 2** was to evaluate whether intentionally rebalancing the training distribution with 20.0% modern C++ (7,000 samples) and expanding training scale from 25,000 to 35,000 balanced samples improves zero-shot cross-project transfer to Envoy and other modern C++ projects, while strictly maintaining **zero Envoy samples** in the training partition.

---

## 2. Experimental Setup & Stratification Methodology

### 2.1 Training Pool Partitioning & Stratification
From `vulnerability_train_expanded.parquet` (138,408 samples), Envoy Proxy code was strictly quarantined:
- Available pool excluding Envoy:
  - C++ Vulnerable: 4,333
  - C++ Safe: 6,801
  - C Vulnerable: 25,027
  - C Safe: 102,247
- Stratified 35,000 sample cohort construction:
  - Modern C++ Samples: 7,000 (3,500 vulnerable + 3,500 safe from Chrome, TensorFlow, Poppler, etc.)
  - Traditional C Samples: 28,000 (14,000 vulnerable + 14,000 safe)
  - Balanced Class Distribution: 50.0% vulnerable (17,500 / 35,000)
  - **Envoy Samples in Training: Exactly 0** (Absolute isolation guaranteed)

### 2.2 Representation & Architecture
- **Semantic Backbone:** CodeBERT Base (`microsoft/codebert-base`), generating $768 + 768 = 1,536$-dimensional $[CLS] + \text{Mean}$ pooled embeddings extracted on Apple Silicon MPS.
- **Structural Features:** 38-dimensional Phase 16 Omission-Aware C/C++ structural feature vector (StandardScaler normalized).
- **Total Input Dimensionality:** 1,574 dimensions.
- **Classifier:** Regularized Logistic Regression with balanced weighting.

---

## 3. Validation Tuning & Test Results

### 3.1 Validation Parameter Selection
A parameter grid search was conducted on the untouched validation set ($N = 3,000$):

| Regularization $C$ | Validation AUROC | Convergence |
| :--- | :---: | :---: |
| $C = 0.005$ | 0.7478 | Converged |
| $C = 0.010$ | 0.7538 | Converged |
| $C = 0.050$ | 0.7607 | Converged |
| $C = 0.100$ | 0.7621 | Converged |
| $C = 0.200$ | 0.7647 | Converged |
| **$C = 0.500$ (Selected)** | **0.7672** | **Converged** |

*Note: Validation AUROC peaked at $C = 0.500$ (0.7672 vs 0.7538 at $C = 0.01$).*

---

### 3.2 Held-Out Clean Benchmark Performance ($N = 3,169$)

The selected model was evaluated against the clean held-out benchmark:

| Metric | Phase 15 Baseline (25k) | Exp 2 Stratified (35k) | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **74.00%** | 71.63% | -2.37% |
| **AUROC** | **0.7369** | 0.6762 | -0.0607 |
| **F1 Score** | **0.5507** | 0.5069 | -0.0438 |
| **Precision** | **60.77%** | 56.34% | -4.43% |
| **Recall** | **50.35%** | 46.06% | -4.29% |
| **FPR** | **15.05%** | 16.53% | +1.48% |
| **FNR** | **49.65%** | 53.94% | +4.29% |
| **Brier Score** | **0.1837** | 0.1985 | +0.0148 |

---

### 3.3 Project-Wise Performance Breakdown

| Project | Samples | Metric | Phase 15 Baseline | Exp 2 (35k Stratified) | Shift |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Envoy Proxy** | 603 | Accuracy <br> AUROC <br> FPR <br> Recall | **64.01%** <br> **0.5411** <br> 15.75% <br> **24.14%** | 62.02% <br> 0.4761 <br> **14.25%** <br> 15.27% | -1.99% <br> -0.0650 <br> **-1.50%** <br> -8.87% |
| **OpenSSL** | 1,257 | Accuracy <br> AUROC <br> FPR <br> Recall | **69.29%** <br> **0.7479** <br> **25.30%** <br> **62.99%** | 66.98% <br> 0.7135 <br> 27.22% <br> 60.24% | -2.31% <br> -0.0344 <br> +1.92% <br> -2.75% |
| **Vim** | 516 | Accuracy <br> AUROC <br> FPR <br> Recall | 77.13% <br> **0.7573** <br> 12.33% <br> **42.20%** | 76.36% <br> 0.6992 <br> **11.89%** <br> 37.50% | -0.77% <br> -0.0581 <br> **-0.44%** <br> -4.70% |
| **GPAC** | 793 | Accuracy <br> AUROC <br> FPR <br> Recall | **86.38%** <br> **0.8103** <br> **7.06%** <br> **40.40%** | 83.23% <br> 0.6873 <br> 10.09% <br> 36.36% | -3.15% <br> -0.1230 <br> +3.03% <br> -4.04% |

---

## 4. In-Depth Scientific Analysis & Failure Diagnosis

### 4.1 Why Did Validation AUROC Rise while Held-Out Test Generalisation Fell?
1. **Covariate Shift in Modern C++:**
   In the 35k training cohort, the 7,000 C++ samples were predominantly sourced from Chromium and TensorFlow. Chromium's C++ codebase heavily utilizes project-specific abstractions (e.g., `base::BindOnce`, `scoped_refptr`, IPC macros, Blink DOM garbage collection handles). Envoy Proxy, while also written in modern C++, uses an entirely different design pattern (Google Test mocks, `Buffer::Instance`, asynchronous event loops, libevent dispatchers).
2. **Over-fitting to Source Repositories at Higher $C$:**
   Because validation data was drawn uniformly from the multi-project corpus, the classifier rewarded larger weights ($C = 0.500$) for project-specific token patterns that appeared in both train and val pools. However, on genuinely held-out, unseen projects (Envoy, OpenSSL, GPAC, Vim), these large weights resulted in brittle decision boundaries.
3. **The Regularization Paradox:**
   In Phase 15, strong regularization ($C = 0.010$) prevented the linear head from memorizing specific lexical cues, forcing the classifier to rely on universal structural and semantic features. When $C$ was relaxed to $0.500$ to maximize in-distribution validation AUROC, cross-project generalisation was impaired.

---

## 5. Scientific Conclusion & Research Recommendation

1. **Defensible Negative Result:**
   Experiment 2 provides a critical scientific finding for our B.Tech evaluation: **naive volume expansion (25k to 35k) and unconstrained C++ rebalancing without domain-adaptation techniques induces covariate shift and degrades cross-project generalisation.**
2. **Preservation of Baseline:**
   The Phase 15 baseline ($C = 0.01$, 25k balanced split) remains superior to Candidate 2 in both overall accuracy (74.00% vs 71.63%) and Envoy zero-shot transfer (64.01% vs 62.02%).
3. **Deployment Recommendation:**
   Do **not** replace the Phase 15 checkpoint with Candidate 2. Checkpoint `phase16_codebert_cxx_stratified_35k.pkl` is preserved in `models/checkpoints/` as an audited experimental baseline demonstrating the limits of unguided data scaling.
