# Phase 10.1 Independent Audit Report: Verification, Leakage Analysis, and Scientific Validity of Model B

**Project Title:** Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code  
**Date:** October 27, 2026  
**Audit Type:** Independent Scientific & Methodological Verification  
**Auditors:** Senior Research Engineer & ML Audit Team  

---

## 1. Executive Summary & Audit Verdict

### Final Audit Verdict:
> **VERDICT: REPRODUCED BUT LIMITED TO THE CONTROLLED BENCHMARK**
>
> 1. **Cryptographic & Metric Reproducibility (PASS):** The locked 32-sample benchmark was cryptographically verified against its recorded SHA-256 checksum with zero alterations. All reported overall metrics (Model A: 56.25%, Model B: 96.88%) are reproduced directly from exported predictions with 100% mathematical precision.
> 2. **Information Leakage (PASS with respect to data; CAVEAT with respect to task design):** There is **zero exact code leakage, zero near-duplicate code, zero task ID overlap, and zero CWE overlap** between Model B's development dataset and the locked benchmark.
> 3. **The Core Scientific Caveat (CRITICAL DISCLOSURE):** Model B's remarkable 96.88% accuracy is primarily driven by the **shared structural semantics of the controlled benchmark paradigm**. Because the benchmark was constructed such that raw AI code is almost uniformly vulnerable ($15/15 = 100\%$) and human-remediated/safe code is almost uniformly benign ($16/17 = 94.1\%$), provenance features act as a dominant proxy for the benchmark's construction rules. When provenance is inverted, Model B's accuracy collapses to **28.12%**. Therefore, these results validate the triage framework within controlled multi-stage editing paradigms, but cannot be extrapolated as a general-purpose replacement for code analysis in real-world open source.

---

## 2. Verification of the Locked Benchmark

### Cryptographic Checksum Audit:
- **Benchmark Path:** [`results/reports/controlled_benchmark_32.json`](file:///Users/mankabir/Desktop/Btech%20project/results/reports/controlled_benchmark_32.json)
- **Immutable Backup Path:** [`data/locked_benchmark_32/controlled_benchmark_32.json`](file:///Users/mankabir/Desktop/Btech%20project/data/locked_benchmark_32/controlled_benchmark_32.json)
- **Source Module:** [`src/experiments/controlled_editing.py`](file:///Users/mankabir/Desktop/Btech%20project/src/experiments/controlled_editing.py)
- **Checksum Verification:**
  - Current File SHA-256: `d53e20033a0239e6156a9dc9e4934deac85956e1adde32d72777ccc8c234fca7`
  - Recorded Locked SHA-256: `d53e20033a0239e6156a9dc9e4934deac85956e1adde32d72777ccc8c234fca7`
  - Checksum Status: **MATCH (Identical)**

### Class and Category Distribution ($N=32$):
- **Vulnerability Distribution:**
  - Safe (`vuln_label = 0`): Exactly 16 units (50.0%)
  - Vulnerable (`vuln_label = 1`): Exactly 16 units (50.0%)
  - Balance: **Perfect 50:50 balance**
- **Actual Provenance Categories in Benchmark:**
  - `human`: Exactly **9 units** (8 safe, 1 vulnerable: `pilot_001_v4_human_insecure`)
  - `ai`: Exactly **8 units** (0 safe, 8 vulnerable: all naive AI implementations)
  - `human_edit_of_ai`: Exactly **8 units** (8 safe, 0 vulnerable: all human-refactored fixes)
  - `ai_edit_of_human`: Exactly **7 units** (0 safe, 7 vulnerable: all regressive AI partial fixes)
  - Total: $9 + 8 + 8 + 7 = 32$ units across 8 security tasks.

---

## 3. Recomputed Evaluation Metrics from Raw Predictions

All metrics were recomputed from the exported predictions artifact [`results/phase10_1_locked_predictions.csv`](file:///Users/mankabir/Desktop/Btech%20project/results/phase10_1_locked_predictions.csv) using [`scripts/run_phase10_1_audit.py`](file:///Users/mankabir/Desktop/Btech%20project/scripts/run_phase10_1_audit.py):

| Evaluation Metric | Model A (Code Only) | Model B (Code + Provenance) | Model B (Unknown Fallback) | Discrepancy vs Phase 10 |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 56.25% (18/32) | **96.88%** (31/32) | 56.25% (18/32) | Exact Match ($\pm 0.00\%$) |
| **95% Wilson Confidence Interval** | $[39.3\%, 71.8\%]$ | $[84.3\%, 99.5\%]$ | $[39.3\%, 71.8\%]$ | Exact Match |
| **Balanced Accuracy** | 56.25% | **96.88%** | 56.25% | Exact Match |
| **Precision** | 53.85% (14/26) | **100.00%** (15/15) | 53.85% (14/26) | Exact Match |
| **Recall** | 87.50% (14/16) | **93.75%** (15/16) | 87.50% (14/16) | Exact Match |
| **F1 Score** | 0.6667 | **0.9677** | 0.6667 | Exact Match |
| **Matthews Correlation (MCC)** | +0.1601 | **+0.9393** | +0.1601 | Exact Match |
| **AUROC** | 0.5938 | **0.9844** | 0.5938 | Exact Match |
| **PR-AUC (Average Precision)** | 0.5491 | **0.9851** | 0.5491 | Exact Match |
| **Brier Score** | 0.2951 | **0.0294** | 0.2951 | Exact Match |
| **Confusion Matrix** | $\begin{bmatrix} 4 & 12 \\ 2 & 14 \end{bmatrix}$ | $\begin{bmatrix} 16 & 0 \\ 1 & 15 \end{bmatrix}$ | $\begin{bmatrix} 4 & 12 \\ 2 & 14 \end{bmatrix}$ | Exact Match |

---

## 4. Subgroup Metric Reconciliation & Denominator Investigation

In the Phase 10 report table, three subgroup accuracy percentages were listed:
- Pure Human: $33.3\%$ (Model A) and $88.9\%$ (Model B)
- Pure AI: $87.5\%$ (Model A) and $100.0\%$ (Model B)
- Human-AI Hybrid: $53.3\%$ (Model A) and $100.0\%$ (Model B)

### Investigation of Denominators:
1. **Pure Human ($N=9$, NOT $N=16$):**
   - In the benchmark, `author_type == 'human'` appears in exactly 9 units (the 8 `v1_human_secure` baselines + `pilot_001_v4_human_insecure`).
   - On these 9 units:
     - Model A correctly predicted 3 units $\implies 3/9 = \mathbf{33.33\%}$.
     - Model B correctly predicted 8 units $\implies 8/9 = \mathbf{88.89\%}$.
   - *Finding:* The Phase 10 text mistakenly typed $(N=16)$ in the label header, but the math was executed on the 9 samples matching `author_type == 'human'`.
2. **Human-AI Hybrid ($N=15$, NOT $N=8$):**
   - The filter `author_type in ['human_edit_of_ai', 'ai_edit_of_human']` matched both the 8 remediated versions AND the 7 regressive AI edits ($8 + 7 = 15$ units).
   - On these 15 units:
     - Model A correctly predicted 8 units $\implies 8/15 = \mathbf{53.33\%}$.
     - Model B correctly predicted 15 units $\implies 15/15 = \mathbf{100.00\%}$.
   - *Finding:* The Phase 10 text mistakenly typed $(N=8)$ in the label header, but the math was executed on all 15 hybrid units.
3. **Pure AI ($N=8$, Exact Match):**
   - `author_type == 'ai'` matched exactly 8 units.
   - Model A got 7 correct out of 8 $\implies 7/8 = \mathbf{87.50\%}$.
   - Model B got 8 correct out of 8 $\implies 8/8 = \mathbf{100.00\%}$.

The metric reconciliation document ([`reports/phase10_1_metric_reconciliation.md`](file:///Users/mankabir/Desktop/Btech%20project/reports/phase10_1_metric_reconciliation.md)) provides the corrected, fully disaggregated breakdown for all 4 distinct author types.

---

## 5. Training and Information Leakage Audit

A comprehensive leakage audit was conducted comparing the 64 development units (`data/model_b_dev/`) against the 32 locked benchmark units (`data/locked_benchmark_32/`):

1. **Exact Code Matches:** **0 found.**
2. **Near-Duplicate Code Matches (difflib ratio $> 0.60$):** **0 found.** The maximum observed string similarity between any development snippet and any locked benchmark snippet was below $0.51$.
3. **Task ID Overlap:** **0 found.** Development tasks are named `dev_task_009` through `dev_task_024`; locked benchmark tasks are named `pilot_001` and `task_002` through `task_008`.
4. **CWE Overlap:** **0 found.** The development corpus covers 16 CWEs (CWE-787, 125, 770, 457, 415, 195, 193, 362, 287, 319, 835, 478, 400, 843, 676, 20), none of which overlap with the 8 CWEs in the locked benchmark (CWE-120, 134, 190, 78, 416, 22, 327, 476).
5. **No Shared Model Selection:** The training script [`scripts/train_model_b.py`](file:///Users/mankabir/Desktop/Btech%20project/scripts/train_model_b.py) loads only `data/model_b_dev/model_b_dev_train.json` and `data/model_b_dev/model_b_dev_val.json`. The locked benchmark file was never imported, read, or referenced during training or tuning.

### Indirect Encoding Audit (The Construction Paradigm):
While code and tasks were completely disjoint, **both datasets share the same underlying experimental generation schema**:
- When code is generated by AI (`ai` or `ai_edit_of_human`), it is almost exclusively vulnerable ($100\%$ in dev, $100\%$ in locked).
- When code is reviewed/remediated by humans (`human_edit_of_ai`), it is almost exclusively secure ($100\%$ in dev, $100\%$ in locked).
- When code is pure human baseline (`v1`), it is almost exclusively secure ($100\%$ in dev, $88.9\%$ in locked).

*Auditor's Conclusion on Leakage:* There is **no data leakage in the traditional machine learning sense** (no shared samples, tasks, or code). However, there is **structural task association**: Model B learned that the benchmark's lineage categories strongly correlate with vulnerability, which holds true across both sets because both were constructed under the same controlled experiment methodology.

---

## 6. Provenance Value & Diagnostic Ablation Experiments

To test whether Model B's predictions are genuinely sensitive to provenance or merely reflecting code features, we executed four controlled diagnostic tests on the identical 32 locked benchmark samples:

| Diagnostic Regime | Tested Configuration | Accuracy | F1 Score | Confusion Matrix | Implication |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Baseline (Code Only)** | Model A inference pipeline | 56.25% | 0.6667 | `[[4, 12], [2, 14]]` | 12 false positives on safe code |
| **Full Model B** | Code + Actual Recorded Provenance | **96.88%** | **0.9677** | `[[16, 0], [1, 15]]` | 0 false positives, 1 false negative |
| **Unknown Provenance Fallback** | `provenance=None` (Missing metadata) | **56.25%** | **0.6667** | `[[4, 12], [2, 14]]` | **Exact 100% replication of Model A** (zero bias added) |
| **Inverted Provenance** | Human labeled as AI 100%; AI labeled as Human | **28.12%** | **0.0800** | `[[8, 8], [15, 1]]` | **Performance collapses by 68.76%** |
| **All Assumed Pure AI** | `ai_inv=1.0, is_pure_ai=1.0` on all 32 | 50.00% | 0.6667 | `[[0, 16], [0, 16]]` | Predicts all 32 as vulnerable |
| **All Assumed Pure Human** | `ai_inv=0.0, edits=2` on all 32 | 50.00% | 0.0000 | `[[16, 0], [16, 0]]` | Predicts all 32 as safe |

### Scientific Interpretation of Diagnostics:
1. **Fallback Integrity:** When provenance is unknown, Model B produces predictions that are mathematically identical to Model A ($P_{\text{final}} = P_{\text{code}}$, max absolute difference $= 0.000000$). This proves the missing-data indicator functions properly and does not inject spurious assumptions.
2. **Causal Sensitivity:** Inverting provenance crashes accuracy from $96.88\%$ to $28.12\%$. This proves the model is genuinely responding to provenance signals rather than relying solely on Model A's base probabilities.
3. **The Reliance Caveat:** The fact that assuming "All Pure AI" predicts 32/32 vulnerable and assuming "All Pure Human" predicts 32/32 safe demonstrates that the linear fusion head places high weight on provenance relative to code logits ($\text{weight}_{\text{pure\_ai}} = +3.65$, $\text{weight}_{\text{human\_edited}} = -3.65$, vs $\text{weight}_{\text{logit\_code}} = +0.098$).

---

## 7. Verification of Model A's Evaluations

We reproduced Model A's evaluations across both reported benchmarks:

1. **Clean Held-Out Cross-Project Test ($N=3,169$):**
   - Checkpoint: `expanded_tri_union_best.pkl`
   - Overall Accuracy: **64.31%** (2,038 / 3,169)
   - Balanced Accuracy: **62.41%**
   - Recall: **57.23%** (574 / 1,003) | Precision: **44.98%** (574 / 1,276)
   - AUROC: **0.6852**
   - Confusion Matrix: `[[1464, 702], [429, 574]]`
   - Repository Breakdown:
     - OpenSSL ($N=1,257$): 67.70% Acc, 65.92% Rec, 0.7394 AUROC
     - Vim ($N=516$): 67.44% Acc, 47.50% Rec, 0.6812 AUROC
     - GPAC ($N=793$): 61.29% Acc, 74.75% Rec, 0.7061 AUROC
     - Envoy ($N=603$): 58.54% Acc, 29.56% Rec, 0.5883 AUROC
2. **Expanded Validation Corpus ($N=23,648$):**
   - Validation Accuracy: **75.84%**
   - Validation Recall: **75.26%**
   - Validation AUROC: **0.8333**
3. **Scientific Explanation of the Gap ($75.84\%$ vs $64.31\%$):**
   - The validation set was partitioned from the training corpus using stratified commit grouping (zero commit overlap). However, it originated from the same collection of repositories (DiverseVul, BigVul, Devign, PrimeVul), sharing macro conventions, utility libraries, and coding styles.
   - The clean cross-project test set ($N=3,169$) evaluates four completely distinct, unseen repositories (OpenSSL, GPAC, Envoy, Vim). The drop from $75.84\%$ to $64.31\%$ reflects genuine **cross-project domain shift**, particularly on modern C++ proxy code (Envoy: 58.54%) where function-level token statistics underperform.

---

## 8. Summary of Discrepancies and Concrete Resolutions

| Discrepancy Found | Root Cause | Impact | Resolution Applied |
| :--- | :--- | :--- | :--- |
| **Subgroup Denominators in Phase 10 Report** | Label text displayed $(N=16)$ and $(N=8)$, but code ran on actual category matches ($N=9$ and $N=15$). | Caused confusion about whether $33.3\%$ or $53.3\%$ were valid fractions on 16/8 samples. | Fully reconciled in [`reports/phase10_1_metric_reconciliation.md`](file:///Users/mankabir/Desktop/Btech%20project/reports/phase10_1_metric_reconciliation.md) with exact 4-category disaggregation. |
| **Benchmark Paradigm Correlation** | Controlled benchmarks inherently associate AI generation with vulnerability and human remediation with safety. | Model B's high accuracy (96.88%) is partially an artifact of benchmark design rather than deep code understanding. | Disclosed prominently as a fundamental research boundary in audit reports. |
| **Single False Negative in Model B** | `pilot_001_v4_human_insecure` is pure human code ($0\%$ AI) that is vulnerable. | Model B predicted safe ($P=0.0596$) because human provenance offset weak code-only signals. | Error documented and used as evidence that provenance weighting must be complemented by dynamic fuzzing. |
