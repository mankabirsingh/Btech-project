# Phase 7 Baseline Audit & Metric Reconciliation

**Project:** Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code  
**Institution:** Delhi Technological University  
**Date:** October 27, 2026  
**Artifact Path:** [`results/experiments/phase7_baseline_audit.md`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase7_baseline_audit.md)  
**Predictions Source:** [`results/experiments/phase6_envoy_predictions.parquet`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase6_envoy_predictions.parquet)  
**Validation Predictions:** [`results/experiments/phase6_validation_predictions.parquet`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase6_validation_predictions.parquet)  
**Machine-Readable Data:** [`results/experiments/phase6_results.json`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase6_results.json)  
**Winning Checkpoint:** [`models/checkpoints/phase6_context_hybrid_best.pkl`](file:///Users/mankabir/Desktop/Btech%20project/models/checkpoints/phase6_context_hybrid_best.pkl)  
**Split Manifest:** [`data/splits/phase5_patch_splits.json`](file:///Users/mankabir/Desktop/Btech%20project/data/splits/phase5_patch_splits.json)  

---

## 1. Executive Summary & Audit Scope

Phase 7 begins with an exhaustive audit of all previously reported experimental results across Phase 5 and Phase 6. In accordance with strict experimental integrity guidelines:
- We recompute all Phase 6 metrics directly from saved parquet prediction files and ground-truth labels.
- We reconcile the reported discrepancy between the **87.20%** and **91.20%** pairwise ranking figures on the held-out test pairs.
- We inspect the exact feature generation logic, split isolation manifests, and verify that test-set labels were never exposed during model training or threshold selection.

---

## 2. Independent Metric Recomputation from Saved Predictions

We independently recomputed every metric by loading [`phase6_envoy_predictions.parquet`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase6_envoy_predictions.parquet), [`phase6_validation_predictions.parquet`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase6_validation_predictions.parquet), and [`phase6_context_hybrid_best.pkl`](file:///Users/mankabir/Desktop/Btech%20project/models/checkpoints/phase6_context_hybrid_best.pkl).

### 2.1 Locked Envoy Benchmark Verification ($N = 603$)
- **Sample Population:** 603 functions from the locked Envoy test partition ([`vulnerability_test_heldout_projects_clean.parquet`](file:///Users/mankabir/Desktop/Btech%20project/data/splits/vulnerability_test_heldout_projects_clean.parquet)).
- **Class Balance:** Exactly 203 ground-truth vulnerable ($Y = 1$) and 400 ground-truth benign ($Y = 0$) functions (33.67% base rate).
- **Evaluated Model:** Experiment C Hybrid Context-Aware Model (`clf_hybrid` in [`phase6_context_hybrid_best.pkl`](file:///Users/mankabir/Desktop/Btech%20project/models/checkpoints/phase6_context_hybrid_best.pkl)).
- **Operating Threshold:** $\tau = 0.50$ (selected purely on validation data).

| Metric | Phase 6 Reported Value | Phase 7 Independently Recomputed Value | Mathematical Formula | Audit Status |
| :--- | :---: | :---: | :---: | :---: |
| **Total Functions ($N$)** | 603 | 603 | $N = TN + FP + FN + TP$ | **EXACT MATCH** |
| **True Negatives ($TN$)** | 186 | 186 | $y = 0 \land \hat{y} = 0$ | **EXACT MATCH** |
| **False Positives ($FP$)** | 214 | 214 | $y = 0 \land \hat{y} = 1$ | **EXACT MATCH** |
| **False Negatives ($FN$)** | 111 | 111 | $y = 1 \land \hat{y} = 0$ | **EXACT MATCH** |
| **True Positives ($TP$)** | 92 | 92 | $y = 1 \land \hat{y} = 1$ | **EXACT MATCH** |
| **Accuracy** | 46.10% | 46.1028% ($278 / 603$) | $(TN + TP) / N$ | **EXACT MATCH** |
| **Balanced Accuracy** | 45.91% | 45.9101% | $0.5 \times (TPR + TNR)$ | **EXACT MATCH** |
| **Vulnerable Recall (TPR)** | 45.32% | 45.3202% ($92 / 203$) | $TP / (TP + FN)$ | **EXACT MATCH** |
| **False Positive Rate (FPR)**| 53.50% | 53.5000% ($214 / 400$) | $FP / (FP + TN)$ | **EXACT MATCH** |
| **Precision** | 30.07% | 30.0654% ($92 / 306$) | $TP / (TP + FP)$ | **EXACT MATCH** |
| **F1 Score** | 0.3615 | 0.361493 | $2TP / (2TP + FP + FN)$ | **EXACT MATCH** |
| **Matthews Corr. Coef. (MCC)**| -0.0773 | -0.077318 | Recomputed from matrix | **EXACT MATCH** |
| **AUROC** | 0.4452 | 0.445166 | Mann-Whitney U rank stat | **EXACT MATCH** |
| **PR-AUC** | 0.3205 | 0.320471 | Average Precision score | **EXACT MATCH** |
| **Total Flagged for Review**| 306 (50.75%) | 306 ($92 + 214$) | $TP + FP$ | **EXACT MATCH** |

**Verified Confusion Matrix:**
$$\begin{bmatrix} TN = 186 & FP = 214 \\ FN = 111 & TP = 92 \end{bmatrix}$$

---

### 2.2 Reconciliation of the 87.20% vs. 91.20% Pairwise-Ranking Discrepancy

The audit inspected both [`phase5_results.json`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase5_results.json) and [`phase6_results.json`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase6_results.json) to reconcile the two held-out ranking accuracy figures:

| Evaluation Dimension | Phase 5 Winning Model (`phase5_patch_hybrid_best.pkl`) | Phase 6 Winning Model (`phase6_context_hybrid_best.pkl`) | Reconciled Finding |
| :--- | :---: | :---: | :--- |
| **Architecture** | Whole Function + Patch + 7-dim Vector | Whole Function + Patch + 16-dim Context Vector | Distinct architectures |
| **Input Features** | Incomplete diff delta (asymmetrical `added_guards`) | Intra-commit call slicing + RAII + GoogleTest flags | Enriched slicing context |
| **Held-Out Test Pairs ($N$)**| 250 commit pairs (500 functions) | Exactly identical 250 commit pairs | Identical evaluation split |
| **Pairwise Correct Count** | **218 / 250 pairs** | **228 / 250 pairs** | **+10 additional pairs resolved** |
| **Pairwise Ranking Accuracy**| **87.20%** | **91.20%** | **+4.00% absolute ranking gain** |
| **Inverted Error Rate** | **12.40%** (31 pairs inverted) | **8.80%** (22 pairs inverted) | **Inverted error cut by 29.0%** |
| **Both-Correct Rate** | 62.80% | 38.80% | $\tau=0.50$ calibration difference |
| **Mean Separation Margin** | +0.4220 | +0.1542 | Symmetrical feature scale |

**Audit Conclusion:** The numbers **do not conflict**. 
- **87.20%** was the held-out test accuracy of the **Phase 5 model**.
- **91.20%** is the held-out test accuracy of the **Phase 6 model**. 
When Phase 6 introduced intra-commit call-graph slicing, guard dominance, callee delegation, and RAII lifecycle features, performance on the identical 250 held-out commit pairs improved from **87.20% to 91.20%**, cutting inverted errors from 31 down to 22.

---

### 2.3 Repository-Separated Validation Audit ($N = 500$ Pairs, 1,000 Functions)
From [`phase6_validation_predictions.parquet`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase6_validation_predictions.parquet):
- **Phase 6 Hybrid Pairwise Ranking Accuracy:** **93.40%** ($467 / 500$ correct).
- **Inverted Error Rate:** **6.60%** ($33 / 500$ pairs inverted).
- **Classification AUROC:** **0.7641** (PR-AUC = **0.7615**).
- **Performance at Selected Threshold $\tau = 0.50$:**
  - Recall: **71.20%** ($356 / 500$).
  - FPR: **32.80%** ($164 / 500$).
  - Precision: **68.46%** ($356 / 520$).
  - Balanced Accuracy: **69.20%**.

---

## 3. Data Split Isolation & Partition Integrity Verification

We audited [`phase5_patch_splits.json`](file:///Users/mankabir/Desktop/Btech%20project/data/splits/phase5_patch_splits.json):
1. **Commit-Isolated Partitions:**
   - **Training Set:** 1,800 commit pairs (3,600 functions) across 212 clean repositories.
   - **Validation Set:** 500 commit pairs (1,000 functions) across 56 clean repositories.
   - **Held-Out Test Pairs:** 250 commit pairs (500 functions) across quarantined test repositories (`gpac`, `vim`, `openssl`, `radare2`, `tcpdump`, `jasper`, `FreeRDP`, `krb5`).
2. **Disjoint Commit Verification:**
   $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
   Verified zero commit overlap across all three splits.
3. **No Pair Crossover:**
   Every commit pair $(f_{before}, f_{after})$ is strictly contained within exactly one partition. Neither vulnerable nor fixed counterparts ever cross splits.
4. **Test Label Integrity Check:**
   We verified that no test labels from locked Envoy ($N=603$) or GPAC ($N=793$) were used during training, feature selection, or threshold calibration. Operating threshold $\tau = 0.50$ was selected strictly from the 500 validation commit pairs.
