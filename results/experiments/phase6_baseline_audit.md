# Phase 6 Baseline Audit & Evaluation Freeze

**Project:** Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code  
**Institution:** Delhi Technological University  
**Date:** October 27, 2026  
**Artifact Path:** [`results/experiments/phase6_baseline_audit.md`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase6_baseline_audit.md)  
**Machine-Readable Data:** [`results/experiments/phase5_results.json`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase5_results.json)  
**Predictions Parquet:** [`results/experiments/phase5_validation_predictions.parquet`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase5_validation_predictions.parquet)  
**Winning Checkpoint:** [`models/checkpoints/phase5_patch_hybrid_best.pkl`](file:///Users/mankabir/Desktop/Btech%20project/models/checkpoints/phase5_patch_hybrid_best.pkl)  
**Split Manifest:** [`data/splits/phase5_patch_splits.json`](file:///Users/mankabir/Desktop/Btech%20project/data/splits/phase5_patch_splits.json)  

---

## 1. Executive Summary & Audit Purpose

Phase 6 addresses the core diagnostic finding from Phase 5: while inter-procedural context and diff representations yielded high pairwise patch-ranking accuracy (**87.20% on unseen repositories**) and elevated vulnerable recall on Envoy to **73.89%** (catching 150 of 203 vulnerabilities), deploying the model on isolated single-function snapshots caused the false-positive rate to surge to **71.75% (287 / 400 benign functions flagged)**, keeping AUROC near chance at **0.5172**.

Before developing inter-commit call-graph slicing and false-positive reduction mechanisms, this audit:
1. Independently verifies and reproduces all reported Phase 5 baseline figures directly from saved checkpoints, predictions, manifests, and evaluation scripts.
2. Re-audits data split isolation and establishes an explicit evaluation freeze to prevent test-set leakage.
3. Formally discloses all prior diagnostic exposures of the Envoy benchmark.
4. Enforces the strict methodological boundary between pairwise patch ranking and isolated-function classification.

---

## 2. Independent Verification of Phase 5 Baseline Results

We verified all Phase 5 metrics by loading the saved checkpoint [`phase5_patch_hybrid_best.pkl`](file:///Users/mankabir/Desktop/Btech%20project/models/checkpoints/phase5_patch_hybrid_best.pkl), the machine-readable output [`phase5_results.json`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase5_results.json), and the validation predictions parquet [`phase5_validation_predictions.parquet`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase5_validation_predictions.parquet).

### 2.1 Locked Envoy Benchmark Verification ($N = 603$)
The locked Envoy benchmark contains 603 functions (203 ground-truth vulnerable, 400 ground-truth benign, 33.67% base vulnerability rate):

| Metric | Phase 5 Reported Value | Phase 6 Independently Recomputed | Match Status | Verification Method |
| :--- | :---: | :---: | :---: | :--- |
| **Total Functions ($N$)** | 603 | 603 | **EXACT MATCH** | Recomputed from test dataframe |
| **Vulnerable Functions ($P$)** | 203 | 203 | **EXACT MATCH** | Ground-truth labels `y_true == 1` |
| **Benign Functions ($N$)** | 400 | 400 | **EXACT MATCH** | Ground-truth labels `y_true == 0` |
| **Accuracy** | 43.62% | 43.6153% (263 / 603) | **EXACT MATCH** | $(TN + TP) / N = (113 + 150) / 603$ |
| **Balanced Accuracy** | 51.07% | 51.0708% | **EXACT MATCH** | $0.5 \times (TPR + TNR) = 0.5 \times (0.7389 + 0.2825)$ |
| **Vulnerable Recall (TPR)** | 73.89% | 73.8916% (150 / 203) | **EXACT MATCH** | $TP / (TP + FN) = 150 / 203$ |
| **Precision** | 34.32% | 34.3249% (150 / 437) | **EXACT MATCH** | $TP / (TP + FP) = 150 / (150 + 287)$ |
| **False Positives (FP)** | 287 | 287 | **EXACT MATCH** | Flagged benign functions ($FP = 287$) |
| **False Positive Rate (FPR)**| 71.75% | 71.7500% (287 / 400) | **EXACT MATCH** | $FP / (FP + TN) = 287 / 400$ |
| **F1 Score** | 0.4688 | 0.468750 | **EXACT MATCH** | $2 \times Prec \times Rec / (Prec + Rec)$ |
| **Matthews Corr. Coef. (MCC)**| +0.0227 | +0.022658 | **EXACT MATCH** | Recomputed from confusion matrix |
| **AUROC** | 0.5172 | 0.517174 | **EXACT MATCH** | Mann-Whitney U rank statistic |
| **PR-AUC** | 0.3492 | 0.349174 | **EXACT MATCH** | Average Precision score |

**Confusion Matrix Verification:**
$$\begin{bmatrix} TN = 113 & FP = 287 \\ FN = 53 & TP = 150 \end{bmatrix}$$
- Total Flagged for Review: $TP + FP = 150 + 287 = 437$ out of 603 (**72.47% review workload**).
- Missed Vulnerabilities: $FN = 53$ out of 203 (**26.11% false negative rate**).

### 2.2 Held-Out Test Pairs Verification ($N = 250$ Pairs, 500 Functions)
Evaluated on 250 commit pairs from held-out repositories strictly quarantined from training:

| Metric | Phase 5 Reported Value | Phase 6 Recomputed Value | Match Status |
| :--- | :---: | :---: | :---: |
| **Pairwise Ranking Accuracy** | 87.20% | 87.2000% (218 / 250) | **EXACT MATCH** |
| **Both-Correct Rate ($P_v \ge 0.5 \land P_f < 0.5$)** | 62.80% | 62.8000% (157 / 250) | **EXACT MATCH** |
| **Inverted Error Rate ($P_f > P_v$)** | 12.40% | 12.4000% (31 / 250) | **EXACT MATCH** |
| **Mean Probability Margin ($P_v - P_f$)** | +0.4220 | +0.421972 | **EXACT MATCH** |
| **Pooled Classification AUROC** | 0.8644 | 0.864408 | **EXACT MATCH** |
| **Pooled Classification PR-AUC** | 0.8231 | 0.823092 | **EXACT MATCH** |
| **Pooled Classification Accuracy** | 81.40% | 81.4000% (407 / 500) | **EXACT MATCH** |

### 2.3 Validation Controlled Experiments Verification ($N = 500$ Pairs, 1,000 Functions)
From [`phase5_validation_predictions.parquet`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase5_validation_predictions.parquet):
- **Whole-Function Baseline:** Ranking Accuracy = **50.00% (250/500)**, AUROC = **0.5832**, Margin = **+0.0184**.
- **Experiment A (Patch-Only):** Ranking Accuracy = **72.80% (364/500)**, AUROC = **0.6649**, Margin = **+0.0748**.
- **Experiment B (Patch + Lightweight Context):** Ranking Accuracy = **84.20% (421/500)**, AUROC = **0.8538**, Margin = **+0.3613**.
- **Experiment C (Hybrid WF + Patch + Context):** Ranking Accuracy = **85.40% (427/500)**, AUROC = **0.8563**, Margin = **+0.3705**.

---

## 3. Methodological Distinction: Ranking vs. Classification

A critical finding from the audit is that **pairwise ranking accuracy on diff pairs cannot be conflated with single-function classification accuracy**:
1. **Pairwise Patch Ranking:** The model evaluates two functions simultaneously ($f_{before}$ and $f_{after}$ from the same commit) and tests whether $P(f_{before}) > P(f_{after})$. Because invariant code is cancelled out, ranking achieves **85.40% on validation** and **87.20% on held-out test pairs**.
2. **Isolated Single-Function Classification:** When a function is evaluated alone without a comparative patch, the model must make an absolute prediction based on an absolute threshold ($\tau = 0.50$).
3. **The Feature Asymmetry Driver:** In Phase 5, the model was trained on pairs where the vulnerable version lacked patch additions (`added_guards = 0`, `added_frees = 0`), which received large negative linear coefficients ($\beta_{guard} = -3.1011$, $\beta_{free} = -0.8412$). When deployed on Envoy as single functions, setting these to 0 artificially added $+3.1$ log-odds toward vulnerability, creating 287 false alarms on benign functions.

---

## 4. Evaluation Freeze & Data Leakage Audit

To preserve strict evaluation integrity, we verified data boundaries:

### 4.1 Split Isolation Verification
We inspected [`phase5_patch_splits.json`](file:///Users/mankabir/Desktop/Btech%20project/data/splits/phase5_patch_splits.json):
- **Train Set:** 1,800 commit pairs across 212 clean repositories.
- **Validation Set:** 500 commit pairs across 56 clean repositories.
- **Held-Out Test Set:** 250 commit pairs across quarantined repositories (`gpac`, `vim`, `openssl`, `radare2`, `tcpdump`, `jasper`, `FreeRDP`, `krb5`).
- **Commit ID Overlap:**
  - $\text{Train} \cap \text{Val} = \emptyset$ (0 overlap).
  - $\text{Train} \cap \text{Test} = \emptyset$ (0 overlap).
  - $\text{Val} \cap \text{Test} = \emptyset$ (0 overlap).
- **Exact Code Duplication:** Verified 0 duplicate functions across train, val, and held-out test splits.
- **Label & Metadata Leakage:** Commit messages, CVE descriptions, issue links, and repository labels remain strictly stripped from model inputs.

### 4.2 Disclosure of Prior Test Set Exposures
In accordance with scientific integrity guidelines:
- The locked Envoy benchmark ($N=603$) has been evaluated as an unseen diagnostic target at the conclusion of **Phase 2** (52.74% Acc), **Phase 3** (54.89% Acc), **Phase 4** (53.57% Acc), and **Phase 5** (43.62% Acc, 73.89% Rec).
- **Protocol Commitment:** The Envoy test partition remains locked. All feature definitions, slicing algorithms, context vectors, and classification thresholds in Phase 6 will be developed, calibrated, and chosen **strictly on the repository-separated validation set**. Envoy will be evaluated exactly once after all model decisions are finalized.
