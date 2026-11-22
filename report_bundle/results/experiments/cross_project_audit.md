# Cross-Project Generalization Audit & Partition Integrity Report (Phase 1)

**Project Title:** Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code  
**Institution:** Delhi Technological University (DTU) — Department of Information Technology  
**Researchers:** Mankabir Singh (23/IT/199) & Mohnish Gautam (23/IT/200)  
**Role:** Senior Machine Learning & Cybersecurity Research Scientist  
**Date:** October 27, 2026  
**Status:** Empirically Audited, Traced, and Verified from Physical Data Partitions  

---

## 1. Executive Summary of Audit Findings

| Audit Check Item | Verification Status | Empirical Finding & Evidence |
| :--- | :---: | :--- |
| **1. Training Sample Counts** | **VERIFIED** | Classical models were trained on a balanced **50,000-sample subset** (25,000 vuln, 25,000 benign) across 888 projects sampled from `vulnerability_train_expanded.parquet` ($N=138,412$). CodeBERT Hybrid was trained on a **12,000-sample subset** (6,000 vuln, 6,000 benign) due to transformer embedding extraction time on MPS. |
| **2. Partition Consistency** | **PARTIAL** | All models were evaluated on the identical test partitions ($N=5,412$ original, $N=3,325$ novel projects, $N=4,000$ Juliet). However, classical models trained on 50,000 samples while CodeBERT trained on 12,000 samples, and validation subsets differed (5,000 vs. 2,000). |
| **3. Project Absence in Train/Val** | **FAILED / LEAKAGE IDENTIFIED** | While the four held-out projects (`vim`, `gpac`, `envoy`, `openssl`) were excluded from the newly ingested DiverseVul/BigVul records, **360 historical samples** (170 OpenSSL, 106 Vim, 80 GPAC, 4 Envoy) already existed in `vulnerability_train.parquet` from PrimeVul and were concatenated into `vulnerability_train_expanded.parquet`. |
| **4. Exact Function Duplicates Across Splits** | **FAILED / 140 OVERLAPS DETECTED** | **140 unique code functions** in `vulnerability_test_heldout_projects.parquet` (4.2% of the 3,325 samples) appeared identically in the expanded training set, and **16 functions** appeared in validation. |
| **5. Metric Reproduction (71.82% & 79.10%)** | **VERIFIED** | CodeBERT's 71.82% test accuracy on novel projects and Calibrated Linear SVM's 79.10% validation accuracy reproduce bit-for-bit from [`results/experiments/expanded_codebert_evaluation_results.json`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/expanded_codebert_evaluation_results.json) and [`results/experiments/expanded_model_evaluation_results.json`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/expanded_model_evaluation_results.json). |
| **6. Threshold Selection Protocol** | **VERIFIED** | Thresholds were tuned strictly on validation labels ($t=0.50, 0.70, 0.72, 0.74, 0.80$) and frozen prior to test evaluation. |
| **7. Target Leakage from Metadata** | **VERIFIED CLEAN** | Feature extractors strictly processed the raw source code text (`code`). `project` and `source_dataset` were excluded from the model feature matrices, preventing direct label leakage via metadata. |

---

## 2. Detailed Verification of the 7 Key Audit Points

### 2.1 Exact Training Examples and Sample Counts Used
- **Expanded Training Corpus Size:** `vulnerability_train_expanded.parquet` contains **138,412 total samples** (29,362 vulnerable [21.21%], 109,050 benign [78.79%]) across 902 open-source repositories.
- **Classical Models:** `scripts/train_expanded_models.py` drew a balanced random sample of 25,000 vulnerable and 25,000 benign functions ($N = 50,000$ samples) across 888 projects (random seed 42).
- **CodeBERT Hybrid:** `scripts/train_expanded_codebert.py` drew a balanced sample of 6,000 vulnerable and 6,000 benign functions ($N = 12,000$ samples). A total of **26,737 representations** were extracted across train, validation, and the three test partitions.

### 2.2 Partition Consistency Across Models
- All reported models were evaluated on the exact same 3 test partitions:
  1. `vulnerability_test.parquet` ($N = 5,412$ samples)
  2. `vulnerability_test_heldout_projects.parquet` ($N = 3,325$ samples)
  3. `vulnerability_test_synthetic_juliet.parquet` ($N = 4,000$ samples)
- **Discrepancy:** The training set sample count differed between classical models ($N = 50,000$) and CodeBERT ($N = 12,000$) due to compute limits on Apple Silicon MPS. Validation tuning sample sizes also differed (5,000 samples for classical vs. 2,000 samples for CodeBERT).

### 2.3 The Project Overlap Discrepancy (Vim, OpenSSL, Envoy, GPAC)
When the dataset expansion script (`scripts/expand_and_preprocess_datasets.py`) was executed:
```python
heldout_projects = ['vim', 'gpac', 'envoy', 'openssl']
heldout_mask = combined_new['project'].isin(heldout_projects)
test_heldout_projects_df = combined_new[heldout_mask].copy().reset_index(drop=True)
pool_for_train_val = combined_new[~heldout_mask].copy().reset_index(drop=True)
...
train_expanded_total = pd.concat([existing_train_df, train_expanded_new], ignore_index=True)
```
While `combined_new` (DiverseVul + BigVul) correctly filtered out the four projects, `existing_train_df` (which came from PrimeVul and Devign) **already contained functions from those four projects**:
- `openssl`: 170 samples in train, 21 in val
- `vim`: 106 samples in train, 16 in val
- `gpac`: 80 samples in train, 10 in val
- `envoy`: 4 samples in train, 0 in val
- **Total:** **360 historical PrimeVul functions from the held-out projects were inadvertently concatenated into `vulnerability_train_expanded.parquet`**.

### 2.4 Duplicate Code Functions Across Partition Boundaries
A strict SHA-256 code string comparison between `vulnerability_test_heldout_projects.parquet` ($N = 3,325$) and `vulnerability_train_expanded.parquet` revealed:
- **140 code functions** in the held-out test set were identical to functions in the training set.
- **16 code functions** were identical to functions in the validation set.

#### Empirical Impact of the 140 Overlapping Samples (Tri-Union Model):
- On all 3,325 test samples: **64.99% Accuracy, 0.7012 AUROC**
- On the clean, strictly non-overlapping 3,185 samples: **64.43% Accuracy, 0.6881 AUROC**
- On the 140 overlapping samples: **77.86% Accuracy, 0.6400 AUROC**
The 140 overlapping samples had an inflated accuracy (+13.43% higher than clean samples) due to memorization of PrimeVul training code.

### 2.5 Reproduction of Reported Metrics (71.82% and 79.10%)
We re-calculated the metrics directly from the machine-readable prediction records:
- **Expanded CodeBERT Hybrid on Novel Projects:**  
  $\text{Accuracy} = 0.718195$ (**71.82%**), $\text{AUROC} = 0.745413$, $\text{MCC} = +0.377844$ ($t = 0.50$). Verified exact match.
- **Expanded Calibrated Linear SVM on Validation Set:**  
  $\text{Accuracy} = 0.791000$ (**79.10%**), $\text{AUROC} = 0.802812$, $\text{MCC} = +0.404219$ ($t = 0.50$). Verified exact match.
- **Expanded Tri-Union Best on Validation Set:**  
  $\text{Accuracy} = 0.758400$ (**75.84%**), $\text{AUROC} = 0.833315$, $\text{MCC} = +0.440921$ ($t = 0.50$). Verified exact match.

### 2.6 Threshold Selection Protocol
In both training scripts (`scripts/train_expanded_models.py` and `scripts/train_expanded_codebert.py`), the function `find_optimal_threshold(y_val, val_probs)` swept thresholds $t \in [0.20, 0.80]$ strictly on validation labels to select $t^*$. Test predictions were subsequently generated using $t^*$. No test labels were used during threshold selection.

### 2.7 Target Leakage Investigation
Neither `project` nor `source_dataset` was passed into any vectorizer or feature union. The input matrices $X$ consisted solely of:
1. TF-IDF word n-grams (1-2) from code text.
2. TF-IDF char n-grams (3-5) from code text.
3. 12 AST/regex structural security counts (buffer ops, pointer depth, memory allocators, control flow complexity).
4. CodeBERT 768-dim [CLS] and 768-dim masked mean-pooling token embeddings.
Therefore, no target label leakage occurred via metadata channels.

---

## 3. Discrepancy Resolution & Protocol for Phase 2–5

To ensure scientific defensibility:
1. **Strict Project Isolation:** We must purge all samples belonging to the held-out repositories (`vim`, `gpac`, `envoy`, `openssl`) from both the training and validation sets, ensuring that the held-out projects are **0% present** in training or validation across ALL datasets.
2. **Decontamination Verification:** Ensure that 0 code strings overlap between the training corpus and the held-out project test set.
3. **Clean Benchmark Creation:**
   - `vulnerability_test_heldout_projects_clean.parquet`: 3,185 strictly decontaminated functions across Vim, OpenSSL, Envoy, and GPAC.
   - Separate novel project holdout: We will additionally construct an untouched test partition from a separate set of repositories (e.g. `php-src`, `redis`, `wireshark`) for unbiased validation.

---

## 4. Exact Reproduction Commands for the Audit

```bash
# 1. Reproduce project presence check in training and validation
python -c "
import pandas as pd
train = pd.read_parquet('data/splits/vulnerability_train_expanded.parquet')
test = pd.read_parquet('data/splits/vulnerability_test_heldout_projects.parquet')
heldout_projects = set(test['project'].unique())
print('Heldout projects found in train:', heldout_projects & set(train['project'].unique()))
print('Exact code overlap:', len(set(test['code']) & set(train['code'])))
"

# 2. Reproduce the 71.82% and 79.10% reported metrics
python -c "
import json
with open('results/experiments/expanded_codebert_evaluation_results.json') as f:
    cb = json.load(f)['test_evaluations']['novel_projects_test']['metrics_default_t50']
    print(f'CodeBERT Test Acc: {cb[\"accuracy\"]*100:.2f}%, AUROC: {cb[\"auroc\"]:.4f}')

with open('results/experiments/expanded_model_evaluation_results.json') as f:
    for exp in json.load(f)['experiments']:
        m = exp['val_metrics_default']
        print(f'{exp[\"model_name\"]}: Val Acc={m[\"accuracy\"]*100:.2f}%, AUROC={m[\"auroc\"]:.4f}')
"
```
