# Phase 11 Model A Evaluation Report: Cross-Project Generalisation and Architecture Comparison

**Project Title:** Provenance-Aware Security Risk Assessment of Human–AI Hybrid Code  
**Institution:** Delhi Technological University, Department of Information Technology  
**Authors:** Mankabir Singh (23/IT/199) and Mohnish Gautam (23/IT/200)  
**Date:** October 27, 2026  
**Status:** Complete Empirical Evaluation across 902 Repositories and 4 Held-Out Projects  

---

## 1. Executive Summary & Core Results

The primary objective for **Model A** (the code-only vulnerability detector) in Phase 11 is to evaluate cross-project generalization under strictly decontaminated, held-out project conditions and establish whether the **$\ge 70\%$ accuracy target** can be achieved without data manipulation, threshold overfitting, or test-set leakage.

### Key Conclusions:
1. **Classical Model Baseline (`expanded_tri_union_best.pkl`):**
   - Cross-project test accuracy ($N=3,169$ across OpenSSL, GPAC, Envoy, Vim): **64.31%**, AUROC: **0.6852**, MCC: **+0.2354**.
   - While robust, fast (latency $<15$ ms), and reliable, the classical Tri-Union model does not achieve the 70% cross-project target.
2. **Deep Semantic Hybrid Candidate (`expanded_codebert_hybrid_best.pkl`):**
   - Combining CodeBERT [CLS] + mean pooling (1,536 dims) with 10-dim structural AST security features and balanced logistic regression achieves **72.01% Accuracy** on the clean held-out cross-project benchmark ($N=3,169$).
   - AUROC: **0.7351** | F1: **0.5671** | MCC: **+0.3606** | Precision: **55.54%** | Recall: **57.93%**.
   - **Target Status:** The **$\ge 70.0\%$ target is achieved (72.01%)** on the genuine held-out cross-project benchmark.
3. **Repository Transfer Performance:**
   - **GPAC:** 85.62% Accuracy (0.8073 AUROC)
   - **Vim:** 76.55% Accuracy (0.7559 AUROC)
   - **OpenSSL:** 70.80% Accuracy (0.7408 AUROC)
   - **Envoy:** 52.74% Accuracy (0.5065 AUROC) — Envoy remains a hard challenge due to modern C++ stateful proxies and virtual dispatch, as diagnosed in Phase 4–8.

---

## 2. Dataset Availability, Lineage, and Preprocessing Audit

An exhaustive audit of vulnerability datasets in the repository was conducted:

| Dataset | Total Usable Samples | Vulnerability Rate | Sourced Tasks / Projects | Original Purpose & Lineage |
| :--- | :---: | :---: | :---: | :--- |
| **DiverseVul** | 93,103 | 20.9% | 794 projects (git commits 2002–2022) | Real-world C/C++ commit diffs with security advisory citations |
| **BigVul (MSR 2020)** | 21,418 | 25.1% | 348 projects (CVE-backed commits) | Paired vulnerable/fixed C/C++ function snippets |
| **Devign** | 20,452 | 45.8% | 2 major projects (QEMU, FFmpeg) | Graph and token baseline benchmark |
| **PrimeVul** | 6,608 | 21.4% | Curated low-noise vulnerability dataset | Decontaminated, verified C/C++ ground truth |
| **Clean Held-Out Test** | **3,169** | **31.65%** | **4 completely unseen projects** (OpenSSL, GPAC, Envoy, Vim) | Strictly decontaminated out-of-distribution evaluation split |

### Data Hygiene & Leakage Prevention:
1. **Semantic Standardization:** All labels standardized strictly to binary integers ($0 = \text{Safe/Fixed}$, $1 = \text{Confirmed Vulnerable}$).
2. **Deduplication:** 800 exact and near-duplicate snippets matching test code were identified and permanently quarantined in `data/reports/dataset_decontamination_audit.json`.
3. **Commit & Lineage Grouping:** Related functions (e.g. vulnerable pre-patch and safe post-patch variants) are grouped by `group_id` (commit hash + vulnerability ID), ensuring variants never cross train/validation/test partitions.
4. **Validation-Only Tuning:** Feature scaling, TF-IDF vocabularies, and decision thresholds were strictly fitted on training and validation splits.

---

## 3. Comprehensive Model A Comparison on the Held-Out Benchmark

All models were evaluated on the identical clean held-out partition ([`data/splits/vulnerability_test_heldout_projects_clean.parquet`](file:///Users/mankabir/Desktop/Btech%20project/data/splits/vulnerability_test_heldout_projects_clean.parquet), $N=3,169$):

| Architecture / Model Candidate | Accuracy | Balanced Accuracy | Precision | Recall | F1 Score | MCC | AUROC | PR-AUC | Latency (CPU/MPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority-Class Baseline** (All Safe) | 68.35% | 50.00% | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.5000 | 0.3165 | 0.0 ms |
| **Model A1: Expanded Tri-Union Best** (TF-IDF + AST) | 64.31% | 62.41% | 44.98% | 57.23% | 0.5037 | +0.2354 | 0.6852 | 0.4918 | **12 ms** |
| **Model A2: Expanded Char N-gram** (3–5 grams) | 66.74% | 64.48% | 47.91% | 58.33% | 0.5261 | +0.2768 | 0.7106 | 0.5107 | 15 ms |
| **Model A3: Expanded Word N-gram** (1–2 grams) | 69.17% | 64.87% | 51.25% | 53.14% | 0.5218 | +0.2945 | 0.7059 | 0.4982 | 14 ms |
| **Model A4: Calibrated Linear SVM** (SGD Huber) | 69.36% | 60.54% | 52.29% | 36.49% | 0.4298 | +0.2362 | 0.6588 | 0.4619 | 16 ms |
| **Model A5: Validation Probability Ensemble** (CB+Tri) | 65.41% | 63.22% | 46.25% | 57.23% | 0.5116 | +0.2519 | 0.6976 | 0.5012 | 265 ms |
| **Model A6: Expanded CodeBERT Hybrid** (Winning Model) | **72.01%** | **68.23%** | **55.54%** | **57.93%** | **0.5671** | **+0.3606** | **0.7351** | **0.5361** | 260 ms |

---

## 4. Per-Project Performance Breakdown for the Winning Model

### Expanded CodeBERT Hybrid (`expanded_codebert_hybrid_best.pkl`):

| Project | Sample Size ($N$) | Ground Truth Vuln Rate | Accuracy | Recall | Precision | F1 Score | AUROC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OpenSSL** | 1,257 | 46.2% | **70.80%** | 64.54% | 69.96% | 0.6714 | 0.7408 | 0.7142 |
| **GPAC** | 793 | 12.5% | **85.62%** | 51.52% | 43.59% | 0.4725 | 0.8073 | 0.4519 |
| **Vim** | 516 | 23.3% | **76.55%** | 55.00% | 49.62% | 0.5218 | 0.7559 | 0.5011 |
| **Envoy** | 603 | 33.7% | **52.74%** | 43.84% | 34.23% | 0.3842 | 0.5065 | 0.3541 |
| **Combined**| **3,169** | **31.65%** | **72.01%** | **57.93%** | **55.54%** | **0.5671** | **0.7351** | **0.5361** |

### Confusion Matrix (Winning Model on $N=3,169$):
$$\begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix} = \begin{bmatrix} 1701 & 465 \\ 422 & 581 \end{bmatrix}$$
- **True Negatives:** 1,701 safe functions correctly identified.
- **False Positives:** 465 safe functions incorrectly flagged ($FPR = 21.47\%$, down from $32.41\%$ in Tri-Union).
- **False Negatives:** 422 vulnerable functions missed ($FNR = 42.07\%$).
- **True Positives:** 581 vulnerable functions caught.

---

## 5. Deployment Recommendation for Model A

- **Checkpoint Path:** [`models/checkpoints/expanded_codebert_hybrid_best.pkl`](file:///Users/mankabir/Desktop/Btech%20project/models/checkpoints/expanded_codebert_hybrid_best.pkl)
- **SHA-256 Hash:** `df74f91cec7d924c666803b13ae9eb7e8cb0414d83aaa0c770aa0e4dd0e585d0`
- **Fast Classical Fallback Checkpoint:** [`models/checkpoints/expanded_tri_union_best.pkl`](file:///Users/mankabir/Desktop/Btech%20project/models/checkpoints/expanded_tri_union_best.pkl) (`a5c62fe77f1b5a9fd42aed1843f664a2d4ba59dcf31f27cb1857afd1266994db`)
- **Strategy:** Retain the classical Tri-Union checkpoint for low-latency interactive keystroke analysis in the UI ($<15$ ms), while exposing the Expanded CodeBERT Hybrid (72.01% cross-project accuracy) as the benchmark-verified Model A deep detector.
