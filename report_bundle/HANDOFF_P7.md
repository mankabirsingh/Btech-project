# Phase 16 Stage 2: Model B Independent Oracle Rebuild and Evaluation Report

**Date:** November 2026  
**Dataset:** `data/phase15_model_b/` (600 code units, 150 tasks, 50 CWE categories)  
**Oracle Dataset File:** [`data/p7/model_b_oracle_labels.json`](file:///Users/mankabir/Desktop/Btech%20project/data/p7/model_b_oracle_labels.json)  
**CWE Family Specification:** [`data/p7/cwe_families.json`](file:///Users/mankabir/Desktop/Btech%20project/data/p7/cwe_families.json)  
**Results Files:** [`results/p7/part_a_oracle_report.json`](file:///Users/mankabir/Desktop/Btech%20project/results/p7/part_a_oracle_report.json), [`results/p7/part_b_real_llm.json`](file:///Users/mankabir/Desktop/Btech%20project/results/p7/part_b_real_llm.json), [`results/p7/part_c_experiments.json`](file:///Users/mankabir/Desktop/Btech%20project/results/p7/part_c_experiments.json)  
**Model Checkpoint:** [`models/checkpoints/p7/model_b_offset_fusion.pkl`](file:///Users/mankabir/Desktop/Btech%20project/models/checkpoints/p7/model_b_offset_fusion.pkl)  

---

## 1. Part A: Independent Oracle Labeling Diagnostics

Static analysis tools (`cppcheck 2.22.0`, `flawfinder 2.0.20`, `clang --analyze 17.0.0`) and `clang -fsyntax-only` were executed across all 600 units. A unit received `oracle_label = 1` if a sanitizer triggered or if a static analysis finding matched the task's CWE family defined in `data/p7/cwe_families.json`. All 600 units are library functions without `main()`; no external executable harnesses exist in the repository.

### Table A1: Syntax and Label Agreement Diagnostics

| Metric | Value | Proportion / Detail |
| :--- | :--- | :--- |
| Total Code Units | 600 | 150 tasks x 4 variants |
| Compilation Passed (`clang -fsyntax-only`) | 287 | 47.8% |
| Compilation Failed (`clang -fsyntax-only`) | 313 | 52.2% |
| Original Generator Vuln Rate | 300 | 50.0% |
| Independent Oracle Vuln Rate | 131 | 21.8% |
| Agreement Accuracy | 337 / 600 | 56.2% |
| Cohen's Kappa ($\kappa$) | 0.1233 | Near chance agreement |
| Confusion Matrix (TN, FP, FN, TP) | [253, 47, 216, 84] | Rows: Orig [0, 1]; Cols: Oracle [0, 1] |

### Table A2: Vulnerability Rate by Provenance Role

Human edits in this dataset are simulated via template-based insertion/patch rules.

| Author Role | N Units | Original Label Vuln Rate | Oracle Label Vuln Rate | Delta |
| :--- | :--- | :--- | :--- | :--- |
| `ai` | 150 | 58.0% (87/150) | 20.0% (30/150) | -38.0 pp |
| `ai_edit_of_human` | 150 | 58.0% (87/150) | 20.0% (30/150) | -38.0 pp |
| `human` | 150 | 42.0% (63/150) | 24.0% (36/150) | -18.0 pp |
| `human_edit_of_ai` | 150 | 42.0% (63/150) | 23.3% (35/150) | -18.7 pp |

### Table A3: Static Tool Coverage

| Analysis Tool | Units with Findings | Units with CWE Family Match |
| :--- | :--- | :--- |
| `flawfinder` | 202 | 103 |
| `cppcheck` | 66 | 30 |
| `clang --analyze` | 10 | 10 |
| Any Tool (Union) | 248 | 131 |

---

## 2. Part B: Real LLM Generation Execution

Environment variables `GEMINI_API_KEY` and `OPENAI_API_KEY` were evaluated.

| Field | Setting / Output |
| :--- | :--- |
| Status | Skipped |
| `GEMINI_API_KEY` present | False |
| `OPENAI_API_KEY` present | False |
| Tasks Attempted | 0 |
| Code Units Generated | 0 |
| Output File | `results/p7/part_b_real_llm.json` |

---

## 3. Part C: Model B Evaluation and Offset Fusion Experiments

Evaluation was conducted using 5-fold GroupKFold by task, repeated across 5 seeds (42, 43, 44, 45, 46). CIs denote 95% bootstrap intervals from 2000 resamples grouped by task.

- **M1:** $P_{\text{code}}$ from `models/checkpoints/p5/model_a_final/predictor.py`
- **M2:** Provenance features only (logistic regression)
- **M3:** Offset fusion: $\text{logit}(P_{\text{risk}}) = \text{logit}(P_{\text{code}}) + w \cdot \text{provenance}$ (fixed offset, no intercept)
- **M4:** Negative control: M3 with provenance shuffled within each task
- **M5:** Static tool finding count only

**Unit Test:** Offset model verified with $|P_{\text{risk}} - P_{\text{code}}| = 2.15 \times 10^{-8} \le 10^{-6}$ when provenance is zero.

### Table C1: Primary Setting — Independent Oracle Labels (N=600, Prevalence=21.8%)

| Model | AUROC [95% CI] | AUPRC [95% CI] | Precision@20% [95% CI] |
| :--- | :--- | :--- | :--- |
| M1: $P_{\text{code}}$ | 0.5815 [0.4891, 0.6662] | 0.2660 [0.1842, 0.3850] | 0.2583 [0.1167, 0.3833] |
| M2: Provenance only | 0.5184 [0.4813, 0.5607] | 0.2178 [0.1620, 0.2820] | 0.2000 [0.1250, 0.2750] |
| M3: Offset fusion | 0.5722 [0.4935, 0.6478] | 0.2539 [0.1779, 0.3601] | 0.2417 [0.1417, 0.3669] |
| M4: Shuffled provenance | 0.5597 [0.4690, 0.6440] | 0.2429 [0.1690, 0.3504] | 0.2167 [0.1083, 0.3583] |
| M5: Static finding count | 0.8373 [0.7743, 0.8937] | 0.4325 [0.3228, 0.5773] | 0.3750 [0.2167, 0.6167] |

### Table C2: Repeat A — Generator Labels (Circular Baseline, N=600, Prevalence=50.0%)

| Model | AUROC [95% CI] | AUPRC [95% CI] | Precision@20% [95% CI] |
| :--- | :--- | :--- | :--- |
| M1: $P_{\text{code}}$ | 0.5507 [0.5309, 0.5733] | 0.5302 [0.5159, 0.5532] | 0.5417 [0.5000, 0.5917] |
| M2: Provenance only | 0.5746 [0.4929, 0.6571] | 0.5522 [0.4906, 0.6314] | 0.5583 [0.4500, 0.6667] |
| M3: Offset fusion | 0.6055 [0.5501, 0.6621] | 0.6092 [0.5512, 0.6790] | 0.6750 [0.5833, 0.7835] |
| M4: Shuffled provenance | 0.5439 [0.5226, 0.5663] | 0.5221 [0.5071, 0.5474] | 0.5417 [0.4667, 0.5833] |
| M5: Static finding count | 0.5258 [0.4912, 0.5616] | 0.5319 [0.5043, 0.5708] | 0.5667 [0.5083, 0.6250] |

### Table C3: Repeat B — Oracle Labels without `human_edit_of_ai` (N=450, Prevalence=21.3%)

Human edits in this dataset are simulated. Evaluated on units from roles `human`, `ai`, and `ai_edit_of_human`.

| Model | AUROC [95% CI] | AUPRC [95% CI] | Precision@20% [95% CI] |
| :--- | :--- | :--- | :--- |
| M1: $P_{\text{code}}$ | 0.6131 [0.5224, 0.6959] | 0.2793 [0.1924, 0.4067] | 0.2667 [0.1333, 0.4222] |
| M2: Provenance only | 0.4955 [0.4412, 0.5535] | 0.2054 [0.1529, 0.2705] | 0.2000 [0.1222, 0.2778] |
| M3: Offset fusion | 0.5897 [0.5026, 0.6739] | 0.2480 [0.1737, 0.3550] | 0.2333 [0.1222, 0.3778] |
| M4: Shuffled provenance | 0.5928 [0.5057, 0.6791] | 0.2509 [0.1763, 0.3635] | 0.2222 [0.1000, 0.3778] |
| M5: Static finding count | 0.8321 [0.7688, 0.8895] | 0.4152 [0.3097, 0.5592] | 0.3556 [0.1889, 0.6000] |

### Table C4: Paired Task-Grouped Bootstrap Differences (2000 Resamples)

| Setting | Comparison | Metric | Observed Difference | 95% Bootstrap CI | Raw p-value | Holm p-value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Primary Oracle | M3 vs M1 | AUROC | -0.0093 | [-0.0313, +0.0138] | 0.4560 | 0.4560 |
| Primary Oracle | M3 vs M4 | AUROC | +0.0125 | [-0.0062, +0.0336] | 0.1830 | 0.3660 |
| Primary Oracle | M1 vs M5 | AUROC | -0.2558 | [-0.3577, -0.1585] | <0.0005 | <0.0005 |
| Repeat A (Orig) | M3 vs M1 | AUROC | +0.0549 | [-0.0001, +0.1079] | 0.0510 | 0.1020 |
| Repeat A (Orig) | M3 vs M4 | AUROC | +0.0616 | [+0.0064, +0.1147] | 0.0260 | 0.0780 |
| Repeat A (Orig) | M1 vs M5 | AUROC | +0.0248 | [-0.0144, +0.0647] | 0.2050 | 0.2050 |
| Repeat B (No Edits) | M3 vs M1 | AUROC | -0.0235 | [-0.0394, -0.0075] | 0.0040 | 0.0080 |
| Repeat B (No Edits) | M3 vs M4 | AUROC | -0.0031 | [-0.0173, +0.0115] | 0.7050 | 0.7050 |
| Repeat B (No Edits) | M1 vs M5 | AUROC | -0.2189 | [-0.3150, -0.1278] | <0.0005 | <0.0005 |

---

## 4. Surprises and Failures

1. In `clang -fsyntax-only` compilation, 313 of 600 units (52.2%) failed due to omitted standard library headers (Tasks 61 to 158 lacked `#include <stdint.h>`, `<stdlib.h>`, and `<string.h>`).
2. The 58.0% vs 42.0% vulnerability discrepancy between AI and human roles present in the generator labels was absent under oracle labeling: rates were 20.0% (`ai`), 20.0% (`ai_edit_of_human`), 24.0% (`human`), and 23.3% (`human_edit_of_ai`).
3. Cohen's kappa between original generator labels and independent oracle labels was 0.1233, indicating agreement near chance level.
4. Offset fusion (M3) yielded no statistically significant AUROC improvement over $P_{\text{code}}$ alone (M1) under oracle labels (difference = -0.0093, Holm $p = 0.4560$), and was statistically indistinguishable from within-task shuffled provenance (M4, difference = +0.0125, Holm $p = 0.3660$).
5. Removing simulated human edits (Repeat B) reduced provenance-only performance (M2) to AUROC 0.4955 [0.4412, 0.5535], while offset fusion (M3) showed a statistically significant performance reduction relative to $P_{\text{code}}$ alone (difference = -0.0235, Holm $p = 0.0080$).
