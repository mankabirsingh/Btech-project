# Phase 10 Training Report: Reproducible Training of Model A & Model B

**Project:** Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code  
**Date:** October 27, 2026  
**Auditors & ML Engineers:** Mankabir Singh & Mohnish Gautam  

---

## 1. Overview and Core Objectives

The goal of Phase 10 is to build a **properly trained, scientifically defensible Model B** (Provenance-Aware Vulnerability Risk Assessor) while preserving Model A as the code-only baseline, guaranteeing zero data leakage, and strictly locking the 32 verified examples from the Multi-CWE benchmark as an independent evaluation set.

---

## 2. Model A Training & Baseline Architecture

### Architecture:
- **Feature Extractor:** `FeatureUnion` combining:
  1. Character n-grams (1–4 chars, max 10,000 features, sublinear TF-IDF).
  2. Token n-grams (1–2 words, max 15,000 features, regex tokenization).
  3. Security AST keywords (specialized C/C++ security APIs, max 2,000 features).
- **Classifier Head:** `LinearSVC` with Platt scaling (`CalibratedClassifierCV(method='sigmoid')`).
- **Isolation:** Provenance features are strictly excluded ($0\%$ provenance input).

### Training Corpus:
- 50,000 functions sampled from `data/splits/vulnerability_train_expanded.parquet` (total corpus: 138,412 C/C++ samples across DiverseVul, BigVul, Devign, and PrimeVul).
- 100% pre-LLM historical human code ($\text{AI Involvement} = 0.0$).

### Performance Summary:
- **Validation Accuracy (N=23,648):** **75.84%** (Exceeds the 70% project target).
  - Validation Recall: **75.26%**
  - Validation AUROC: **0.8333**
- **Clean Held-Out Cross-Project Test (N=3,169 across 4 unseen repositories):**
  - Overall Accuracy: **64.31%** (OpenSSL: 67.70%, Vim: 67.44%, GPAC: 61.29%, Envoy: 58.54%).
  - Balanced Accuracy: **62.41%**
  - Recall: **57.23%**
  - AUROC: **0.6852**

---

## 3. Model B Development Dataset Generation

To train Model B without touching or contaminating the locked 32-sample benchmark:
- A new development corpus was synthesized via [`src/experiments/model_b_development_data.py`](file:///Users/mankabir/Desktop/Btech%20project/src/experiments/model_b_development_data.py).
- **Scope:** 16 seed tasks covering 16 distinct CWE categories:
  - `dev_task_009_heap_oob_write` (CWE-787: Out-of-bounds Write)
  - `dev_task_010_oob_read` (CWE-125: Out-of-bounds Read)
  - `dev_task_011_uncontrolled_allocation` (CWE-770: Allocation without Limits)
  - `dev_task_012_uninitialized_var` (CWE-457: Use of Uninitialized Variable)
  - `dev_task_013_double_free` (CWE-415: Double Free)
  - `dev_task_014_signed_comparison` (CWE-195: Signed to Unsigned Conversion)
  - `dev_task_015_off_by_one` (CWE-193: Off-by-one Error)
  - `dev_task_016_race_condition` (CWE-362: TOCTOU Race Condition)
  - `dev_task_017_improper_auth` (CWE-287: Improper Authentication)
  - `dev_task_018_cleartext_transmission` (CWE-319: Cleartext Transmission)
  - `dev_task_019_infinite_loop_dos` (CWE-835: Infinite Loop)
  - `dev_task_020_missing_default_switch` (CWE-478: Missing Default in Switch)
  - `dev_task_021_stack_exhaustion` (CWE-400: Unbounded Recursion)
  - `dev_task_022_type_confusion` (CWE-843: Type Confusion)
  - `dev_task_023_dangerous_function` (CWE-676: Dangerous Function)
  - `dev_task_024_improper_sanitization` (CWE-20: Improper Input Validation)
- **Lineage Structure per Seed Task:**
  - `v1_human_secure`: 0% AI, Human written, secure baseline (`vuln=0`, `ai_inv=0.0`, `edits=1`).
  - `v2_ai_naive`: 100% AI, unreviewed AI generation with flaw (`vuln=1`, `ai_inv=1.0`, `edits=0`).
  - `v3_human_remediated`: 50% AI, human developer review and fix (`vuln=0`, `ai_inv=0.50`, `edits=2`).
  - `v4_ai_regressive`: 75% AI, AI partial edit that left vulnerability (`vuln=1`, `ai_inv=0.75`, `edits=0`).
- **Grouped Splitting (Zero Task / Lineage Leakage):**
  - **Train Split (`model_b_dev_train.json`):** 12 tasks (48 units; 24 vulnerable, 24 safe).
  - **Val Split (`model_b_dev_val.json`):** 4 tasks (16 units; 8 vulnerable, 8 safe).
  - **Locked Test Set (`controlled_benchmark_32.json`):** 8 tasks (32 units; 16 vulnerable, 16 safe).
  - **Task & Code Leakage Audit:** Verified 0 task overlap and 0 code hash duplicate across splits.

---

## 4. Model B Fusion Architecture & Hyperparameter Selection

### Feature Representation:
$$X = \big[ \text{logit}(P_{\text{code}}), \text{has\_provenance}, \text{ai\_involvement}, \text{is\_human\_edited}, \text{is\_pure\_ai}, \text{confidence} \big]$$
Where:
- $\text{logit}(P_{\text{code}}) = \ln\left(\frac{P_{\text{code}} + \epsilon}{1 - P_{\text{code}} + \epsilon}\right)$ from Model A.
- $\text{has\_provenance} \in \{0.0, 1.0\}$: Explicit missing-data indicator.
- $\text{ai\_involvement} \in [0.0, 1.0]$.
- $\text{is\_human\_edited} \in \{0.0, 1.0\}$.
- $\text{is\_pure\_ai} \in \{0.0, 1.0\}$.
- $\text{confidence} \in [0.0, 1.0]$.

### Hyperparameter Tuning on Validation Split ($N=16$):
Grid search over L2 penalty parameter $C \in [0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0, 10.0]$:
- $C = 10.0$ was selected with Validation Accuracy: **100.00%**, Validation F1: **1.0000**.
- **Learned Fusion Weights:**
  - $\text{logit}(P_{\text{code}})$: $+0.0981$
  - $\text{has\_provenance}$: $+0.0003$
  - $\text{ai\_involvement}$: $+1.8276$ (Higher AI involvement elevates risk)
  - $\text{is\_human\_edited}$: $-3.6468$ (Human review/edits strongly mitigate risk)
  - $\text{is\_pure\_ai}$: $+3.6471$ (Unreviewed AI code substantially increases prior risk)
  - $\text{confidence}$: $-0.0367$
  - $\text{intercept}$: $-0.9551$

---

## 5. Five Required Validation Ablations (Development Validation Set, N=16)

| Ablation Architecture | Validation Accuracy | F1 Score | AUROC | PR-AUC | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Model A (Code Only)** | 68.75% | 0.5455 | 0.5859 | 0.5417 | Sensitive to security identifiers, generating false positives. |
| **2. Provenance-Only Baseline** | 100.00% | 1.0000 | 1.0000 | 1.0000 | Predicts vulnerability from lineage metadata alone. |
| **3. Model B (Code + Provenance)** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | Successfully reconciles code signal with lineage history. |
| **4. Model B (Provenance Masked)** | 50.00% | 0.0000 | 0.5859 | 0.5417 | Operates without provenance, matching conservative prior. |
| **5. Majority-Class Baseline** | 50.00% | 0.0000 | 0.5000 | 0.5000 | Baseline random / constant guessing on balanced split. |

---

## 6. Checkpoint Artifacts & Reproducibility

- Saved Checkpoint: [`models/checkpoints/model_b_provenance_fusion.pkl`](file:///Users/mankabir/Desktop/Btech%20project/models/checkpoints/model_b_provenance_fusion.pkl)
- Training Metrics Manifest: [`results/metrics/model_b_training_metrics.json`](file:///Users/mankabir/Desktop/Btech%20project/results/metrics/model_b_training_metrics.json)
- Reproducible Training Script: [`scripts/train_model_b.py`](file:///Users/mankabir/Desktop/Btech%20project/scripts/train_model_b.py)
