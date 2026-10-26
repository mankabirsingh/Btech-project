# Phase 1: Comprehensive Repository & Scientific Audit
**B.Tech Project-I — Delhi Technological University**
**Project Title:** Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code  
**Investigators:** Mankabir Singh (23/IT/199) & Mohnish Gautam (23/IT/200)  
**Date of Audit:** October 26, 2026  

---

## 1. Executive Assessment

An exhaustive structural, technical, and scientific audit was performed across all 60 files in the repository. The project has laid strong groundwork with working modules, modern tooling, and an active Streamlit application. However, our technical audit identified **five fundamental scientific and implementation defects** that undermine validity if left unaddressed.

---

## 2. File-by-File Inventory & Status

| Path | Purpose | Audit Status | Key Issues Identified |
| :--- | :--- | :--- | :--- |
| `scripts/prepare_dataset.py` | Dataset ingestion & preparation | **CRITICAL BUG** | Benign PrimeVul samples assigned vulnerable code; all 4,416 benign samples dropped as duplicates; PrimeVul became 100% vulnerable. |
| `src/data/dataset_loader.py` | HuggingFace dataset loading | **FIXED** | Previously hung on non-scalar columns (`vul_lines`, `label`). Fixed by checking for scalar primitives before `value_counts`. |
| `src/data/preprocessing.py` | Code normalization & schemas | **NEEDS WORK** | Preprocessing is sound for cleaning, but `standardize_primevul` does not differentiate `vulnerable_function` vs `fixed_function`. |
| `src/data/split_dataset.py` | Splitting & leakage checking | **HIGH RISK** | Fallback to stratified random split split vulnerable/fixed pairs between train and test (near-duplicate leakage). Needs grouped splitting by commit/record. |
| `src/models/baseline.py` | TF-IDF + Logistic Regression | **FUNCTIONAL** | Clean implementation of Char (1,4) and Word (1,2) baselines. Correctly trains and persists. |
| `src/models/code_encoder.py` | CodeBERT & Provenance Classifier | **FLAW IDENTIFIED** | `extract_provenance_features` includes `has_cwe` (label leak). Provenance classifier defined but was never trained on experimental data. |
| `scripts/train_baseline.py` | Baseline training pipeline | **FUNCTIONAL** | Trained on 23,754 samples, evaluated on 4,752 samples. Generated valid plots and metrics. |
| `scripts/train_codebert.py` | CodeBERT training script | **DEFECTIVE RUN** | Used `--freeze_encoder` with `lr=2e-5` for 3 epochs on 1,500 samples. Linear head barely moved from init; predicted class 1 for 97.9% of samples (Acc 54.7%, Recall 98%, MCC 0.0138). |
| `src/experiments/controlled_editing.py`| Controlled evolution pilot | **INSUFFICIENT** | Only 4 synthetic code samples. Valuable as a proof-of-concept, but statistically insufficient to support research claims without expansion. |
| `scripts/run_controlled_experiment.py`| Runner for pilot experiment | **FUNCTIONAL** | Successfully executes pilot, runs static analysis, and computes model probabilities. |
| `src/security/static_analysis.py` | Rule-based security analysis | **FUNCTIONAL** | Patterns for CWE-120, 78, 134, 190, 94. Strips single-line comments. Rules are heuristics, not ground truth CVEs. |
| `src/evaluation/evaluate.py` | Metrics & visualization | **FUNCTIONAL** | Comprehensive metrics (Acc, Prec, Rec, F1, MCC, AUROC, AP). Generates confusion matrices and curves. |
| `app.py` | Streamlit dashboard | **FUNCTIONAL** | 5 pages verified live. Needs to be connected to genuine provenance-comparison experiments and accurate model evaluations. |
| `tests/test_pipeline.py` | Automated test suite | **NEEDS EXPANSION** | 7 tests pass. Needs tests for grouped splitting, near-duplicate leakage, provenance features, and review prioritization. |

---

## 3. Deep-Dive Audit: Five Critical Flaws

### Flaw 1: PrimeVul Benign Sample Erasure & Label Inversion
- **Root Cause:** In `scripts/prepare_dataset.py` (lines 155–158):
  ```python
  result['code'] = df['vulnerable_function'].apply(clean_code)
  result['vuln_label'] = df['is_vulnerable'].astype(int)
  ```
- **Consequence:** `adefu020/primevul` contains pairs of rows: one with `is_vulnerable=True` and one with `is_vulnerable=False`. Both rows list the vulnerable code under `vulnerable_function` and the patched code under `fixed_function`. By unconditionally extracting `vulnerable_function`, the benign rows were given vulnerable code. When `drop_duplicates(subset=['code'])` was called, all 4,416 benign rows were dropped as exact duplicates of the vulnerable rows.
- **Impact:** PrimeVul contributed **4,416 vulnerable samples and 0 benign samples**. The vulnerability dataset was distorted.

### Flaw 2: Vulnerable-Fixed Pair Data Leakage
- **Root Cause:** PrimeVul contains before-and-after pairs for identical functions. When split randomly without grouping by `source_record_id`, the vulnerable version could enter `train` while the fixed version entered `test`.
- **Consequence:** Because the before and after functions share >95% identical tokens (only the bug fix line differs), a model evaluated on such a test set exhibits inflated performance (near-duplicate leakage).
- **Remedy:** Group-based splitting strictly keyed on `source_record_id` / `project` to ensure all variants of a function reside exclusively in either train, validation, or test.

### Flaw 3: CodeBERT Classifier Collapse
- **Root Cause:** When `freeze_encoder=True` was used, CodeBERT's 125M transformer parameters were frozen. The newly initialized linear classification head (`Linear(768, 2)`) was trained with `lr=2e-5` for only 3 epochs on 1,500 samples.
- **Consequence:** `2e-5` is an appropriate fine-tuning learning rate for transformers, but far too small to optimize a randomly initialized linear layer from scratch. The classifier bias was `[-0.0313, +0.0366]`, causing the model to default to predicting class 1 for 97.9% of samples.
- **Impact:** Resulted in the reported metrics: Accuracy 54.67%, Recall 98.05%, Precision 54.77%, MCC 0.0138. The AUROC was 0.6642, proving that continuous ranking signal existed, but the uncalibrated head and default 0.50 threshold produced severe collapse.

### Flaw 4: Disconnect from the Core Research Question
- **Root Cause:** `vulnerability_data.parquet` had 100% human provenance (`provenance_type='human'`, `ai_involvement=0.0`). The models trained on it were evaluated purely as code-only models.
- **Consequence:** The core research question—*"Can provenance-aware analysis improve security-risk prediction compared with provenance-agnostic detection?"*—was never tested experimentally.
- **Remedy:** Implement an explicit, rigorous comparative study between **Model A (Code-only)** and **Model B (Code + Provenance)** on a unified, verified benchmark dataset with documented provenance metadata.

### Flaw 5: Label Leakage in Feature Extraction
- **Root Cause:** In `src/models/code_encoder.py` (lines 231–233):
  ```python
  has_cwe = df['cwe'].notna().astype(float).values
  features.append(has_cwe.reshape(-1, 1))
  ```
- **Consequence:** CWE (Common Weakness Enumeration) is a vulnerability classification taxonomy. Passing `has_cwe` into the model is target leakage (if a sample has a CWE, it is by definition vulnerable).
- **Remedy:** Remove all vulnerability/security fields from provenance feature extraction. Genuine provenance features must only reflect author identity, AI generator, AI involvement percentage, and edit history.

---

## 4. Audit Conclusion & Correction Plan

The audit confirms that the repository has a solid structural foundation, but requires immediate scientific remediation:
1. Fix PrimeVul extraction to preserve paired benign and vulnerable samples.
2. Implement grouped splitting to prevent vulnerable/fixed pair leakage.
3. Remove label-leaking features from provenance encoding.
4. Correct CodeBERT head training and calibration.
5. Execute the actual Model A (code-only) vs. Model B (provenance-aware) comparative experiment.
6. Expand the controlled evolution benchmark across multiple documented CWE scenarios.
7. Update all metrics, documentation, and the Streamlit app with verified findings.
