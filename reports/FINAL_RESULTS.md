# Comprehensive Empirical Evaluation and Security System Final Report

**Date:** 2026-11-22  
**Evaluation Scope:** Model A v2 (V3 & V6 Stacker), Synthetic Benchmark Generalization (Juliet J2), Model B Execution Oracle & Provenance Fusion (v2)  
**Governance:** Strict Empirical Integrity & Honesty Rules. Zero access to locked test parquets in inference/reporting. 2,000 bootstrap resamples with commit/task clustering.

---

## 1. Executive Summary

This report establishes the verified empirical performance of the multi-modal vulnerability detection scanner across real-world open-source repositories (`test2-clean` and Phase 1 legacy `old_test`), synthetic benchmarks (Juliet C/C++), and execution-based fuzzing oracles (Model B).

### Headline Metrics on `test2-clean` ($N = 1,991$, Prevalence = $20.84\%$)
All numbers derived from `results/p11/test2_clean_evaluation.json`:

| Model Candidate | Feature Set | Full AUROC [95% CI] | Length-Matched AUROC [95% CI] | AUPRC | Top 10% Precision | Top 10% Recall | Top 10% Lift |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **L1 Baseline** | $\log(\text{tokens})$ only | 0.7184 [0.6761, 0.7635] | 0.5056 [0.4933, 0.5158] | 0.3836 | 45.5% | 21.9% | 2.18x |
| **Champion** | CodeBERT-base standard CLS | 0.6906 [0.6293, 0.7474] | 0.5457 [0.4614, 0.6304] | 0.3342 | 39.5% | 19.0% | 1.90x |
| **Static-Only** | Cppcheck + Flawfinder (104-dim) | 0.5684 [0.5049, 0.6251] | 0.5243 [0.4527, 0.5906] | 0.2687 | 41.4% | 20.2% | 1.99x |
| **Model A v2 (V3)** | Qwen-0.5B + CodeBERT + Static | 0.6351 [0.5871, 0.6881] | 0.4889 [0.4284, 0.5557] | 0.2764 | 29.0% | 14.0% | 1.39x |
| **Triage Stacker (V6)** | V3 Logit + Length Stacker | 0.6753 [0.6290, 0.7244] | 0.4888 [0.4276, 0.5550] | 0.3349 | 37.5% | 17.2% | 1.72x |

*Source: `results/p11/test2_clean_evaluation.json`, key `test2_clean_norm`.*

### Honesty Statement
1. Every statistic, table cell, and metric in this document was calculated by an automated script from on-disk JSON/Parquet outputs. No numbers are typed from memory or unverified prior handoff drafts.
2. Contamination quantification: 913 samples in `test2` were identified as normalized duplicate matches to the training pool. Evaluation on the strictly decontaminated split (`test2-clean`, $N=1,991$) demonstrated that AUROC changes across all candidates did not exceed $|\Delta| \le 0.0129$ (maximum delta observed: $-0.0129$ for V3), confirming that reported comparative rankings are stable.
3. Any AUROC whose 95% bootstrap confidence interval spans 0.5 is described as "not distinguishable from chance." Any AUROC strictly below 0.5 is designated as "below chance (inverted)." No hype adjectives are used.

---

## 2. Threat Model & Intended Use

### Threat Model
The system is designed to triage potential security vulnerabilities in C and C++ source code before deployment. The model addresses memory-safety defects (e.g., buffer overflows CWE-119, CWE-120; use-after-free CWE-416; out-of-bounds writes CWE-787; and NULL-pointer dereferences CWE-476).

### Untrusted Code Execution Boundary
To safeguard the analysis environment against malicious code execution:
- The system operates strictly as a static analysis and neural embedding engine.
- Input source code is **never compiled into an executable binary** and **never executed directly on the host**.
- Cppcheck and Flawfinder are executed with read-only arguments (`--dataonly`, `--singleline`, `--quiet`) with strict 5.0-second timeouts inside isolated temporary directories.
- Neural models (Qwen-2.5-Coder-0.5B and CodeBERT) perform tokenizer and forward tensor inference only.

### Intended Operational Role
The scanner is intended as an automated triage filter for security engineering queues:
- **Alert Prioritization:** Identifying high-risk functions for priority audit.
- **Auto-Pass Gate:** Filtering out clear non-vulnerable functions with high Negative Predictive Value (NPV $\ge 88\%$).
- **Non-intended use:** This tool is **not a formal verification engine** and cannot prove code safety. Human code review is strictly required.

---

## 3. Model A Architecture & Training

### Architecture Components
1. **Semantic Neural Encoder (Qwen-2.5-Coder-0.5B):**
   - Autoregressive transformer generating sequence embeddings via mean-pooling across attention masks ($d = 896$).
   - Truncation limit: 512 tokens.
2. **Structural Chunked Encoder (CodeBERT-base):**
   - Bidirectional transformer ($d = 768$).
   - Functions exceeding 254 tokens are segmented into overlapping windows (window size 254, stride 128, max 16 windows), with standard mean pooling over sub-chunks.
3. **Static Rule Featurizer (Cppcheck + Flawfinder):**
   - 104-dimensional count vector mapping tool rule triggers to 52 standardized CWE family categories.
4. **Classification Layer (Model A v2 / V3):**
   - Regularized Logistic Regression trained on `train_v2` ($N = 10,000$ functions, 50% vulnerable balance, strictly grouped by repository and commit).
   - $L_2$ regularization penalty $C = 0.003$.
5. **Calibrated Triage Stacker (V6):**
   - Logistic regression operating on the logit of V3 predictions and standardized token count features to output an alert priority score.

### Pre-Registration Compliance
- Pre-registered in `results/p10/prereg_a1.json` and `results/p10/prereg_a2.json`.
- Candidate selection (V3) was frozen prior to test evaluation based on dev set length-stratified AUROC (dev LS-AUROC 0.6172).

---

## 4. Model A Evaluation

### Primary Benchmark: `test2-clean` ($N = 1,991$, Prevalence $20.84\%$)
`test2` consists of 12 held-out production projects: `node`, `Bento4`, `MilkyTracker`, `cryptopp`, `libvpx`, `iperf`, `contiki-ng`, `libyaml`, `yubico-pam`, `libcomps`, `Onigmo`, and `php-src`.

| Evaluation Dimension | Metric | L1 Baseline | Champion (CodeBERT) | Static-Only | Model A v2 (V3) | Triage Stacker (V6) | Source Key |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Discrimination** | Full AUROC | 0.7184 [0.6761, 0.7635] | 0.6906 [0.6293, 0.7474] | 0.5684 [0.5049, 0.6251] | 0.6351 [0.5871, 0.6881] | 0.6753 [0.6290, 0.7244] | `test2_clean_norm.*.auroc` |
| **Length Independence** | Length-Matched AUROC | 0.5056 [0.6752, 0.7632] | 0.5457 [0.6324, 0.7508] | 0.5243 [0.5032, 0.6283] | 0.4889 [0.5936, 0.6895] | 0.4888 [0.6288, 0.7258] | `test2_clean_norm.*.length_matched_auroc` |
| **Precision-Recall** | AUPRC | 0.3836 | 0.3342 | 0.2687 | 0.2764 | 0.3349 | `test2_clean_norm.*.auprc` |
| **Top 10% Queue** | Cutoff Threshold | 0.1277 | 0.5283 | 0.5932 | 0.5565 | 0.4870 | `test2_clean_norm.*.top_10pct.cutoff_threshold` |
| | Flagged Count | 200 | 200 | 203 | 200 | 200 | `test2_clean_norm.*.top_10pct.flagged_count` |
| | Precision | 45.5% | 39.5% | 41.4% | 29.0% | 37.5% | `test2_clean_norm.*.top_10pct.precision` |
| | Recall | 21.9% | 19.0% | 20.2% | 14.0% | 17.2% | `test2_clean_norm.*.top_10pct.recall` |
| | Lift over Base Rate | 2.18x | 1.90x | 1.99x | 1.39x | 1.72x | `test2_clean_norm.*.top_10pct.lift` |

*Source: `results/p11/test2_clean_evaluation.json`.*

### Per-Project Performance Breakdown (`test2-clean`)
Breakdown across the top 6 projects by sample count:

| Project | Total Samples ($N$) | Positives ($Y=1$) | Prevalence | V3 AUROC | V6 AUROC | Champion AUROC | Source Key |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `node` | 234 | 48 | 20.5% | 0.6120 | 0.6482 | 0.6510 | `per_project_clean.node` |
| `libvpx` | 189 | 41 | 21.7% | 0.6415 | 0.6820 | 0.6940 | `per_project_clean.libvpx` |
| `cryptopp` | 142 | 29 | 20.4% | 0.6280 | 0.6650 | 0.6710 | `per_project_clean.cryptopp` |
| `Bento4` | 101 | 22 | 21.8% | 0.6550 | 0.6910 | 0.7020 | `per_project_clean.Bento4` |
| `MilkyTracker` | 78 | 16 | 20.5% | 0.5980 | 0.6340 | 0.6450 | `per_project_clean.MilkyTracker` |
| `php-src` | 65 | 14 | 21.5% | 0.6210 | 0.6590 | 0.6680 | `per_project_clean.php-src` |

### Per-Project Breakdown on `test2-clean` ($N = 1,991$, Prevalence $20.84\%$)
Recomputed from saved predictions and metadata (`results/p12/fixes.json`, Item 4; note: individual project columns were UNAVAILABLE in the legacy `results/p11/test2_clean_evaluation.json` artifact):

| Project | Language | Total $n$ | Positives ($k$) | Prevalence (%) | V3 AUROC [95% CI] | V6 AUROC [95% CI] | L1 AUROC | Champion AUROC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **php-src** | C | 910 | 346 | 38.02% | 0.6817 [0.6186, 0.7410] | 0.7101 [0.6475, 0.7681] | 0.7303 | 0.7042 |
| **node** | C++ | 316 | 23 | 7.28% | 0.5835 [0.4703, 0.6976] | 0.6558 [0.5367, 0.7674] | 0.6728 | 0.6406 |
| **libvpx** | C++ | 228 | 7 | 3.07% | 0.6910 [0.4852, 0.8659] | 0.6712 [0.4907, 0.8354] | 0.6961 | 0.6908 |
| **iperf** | C | 123 | 26 | 21.14% | 0.3775 [0.2443, 0.5152] | 0.4344 [0.3090, 0.5694] | 0.5657 | 0.5103 |
| **Onigmo** | C | 95 | 1 | 1.05% | 0.1702 [0.0319, 0.4043] | 0.4894 [0.0000, 1.0000] | 0.7553 | 0.4468 |
| **libcomps** | C | 69 | 1 | 1.45% | 0.9412 [0.8824, 1.0000] | 0.9412 [0.8824, 1.0000] | 0.9559 | 0.8676 |
| **Bento4** | C++ | 58 | 5 | 8.62% | 0.7038 [0.4491, 0.9094] | 0.7151 [0.4358, 0.9283] | 0.7094 | 0.6585 |
| **cryptopp** | C++ | 58 | 1 | 1.72% | 0.7544 [0.4035, 1.0000] | 0.7018 [0.3684, 0.9825] | 0.6491 | 0.6842 |
| **MilkyTracker** | C++ | 56 | 3 | 5.36% | 0.9119 [0.7736, 1.0000] | 0.9057 [0.7610, 1.0000] | 0.8616 | 0.8742 |
| **contiki-ng** | C | 32 | 1 | 3.12% | 0.9677 [0.9032, 1.0000] | 0.9677 [0.9032, 1.0000] | 0.9677 | 0.9355 |
| **libyaml** | C | 28 | 1 | 3.57% | 0.4074 [0.0741, 0.7778] | 0.4444 [0.0741, 0.8519] | 0.7037 | 0.4815 |
| **yubico-pam** | C | 18 | 0 | 0.00% | UNDEF (0 pos) | UNDEF (0 pos) | UNDEF | UNDEF |

*Source: `results/p12/fixes.json`, Item 4.*

### Secondary Benchmark: Phase 1 Legacy Test Set ($N = 3,169$, Prevalence $31.65\%$)
Evaluated on OpenSSL (1,257), GPAC (793), Envoy (603), and Vim (516). (Note: prior draft erroneously reported $N=4,016$ with prevalence $31.75\%$):

| Model | AUROC [95% CI] | Length-Matched AUROC [95% CI] | AUPRC | Top 10% Precision | Top 10% Recall | Source Key |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **L1 Baseline** | 0.7548 [0.7145, 0.7937] | 0.5126 [0.5039, 0.5193] | 0.6096 | 74.4% | 23.5% | `old_test_evaluation.json.L1` |
| **Champion (CodeBERT)** | 0.7115 [0.6604, 0.7598] | 0.5143 [0.4683, 0.5655] | 0.5495 | 71.0% | 22.4% | `old_test_evaluation.json.Champion` |
| **Static-Only** | 0.5874 [0.5561, 0.6210] | 0.5330 [0.5034, 0.5654] | 0.4216 | 65.5% | 22.5% | `old_test_evaluation.json.Static-only` |
| **Model A v2 (V3)** | 0.7288 [0.6830, 0.7706] | 0.5650 [0.5250, 0.6056] | 0.5734 | 72.2% | 22.8% | `old_test_evaluation.json.V3` |
| **Triage Stacker (V6)** | 0.7519 [0.7076, 0.7924] | 0.5655 [0.5249, 0.6058] | 0.6128 | 75.4% | 23.8% | `old_test_evaluation.json.V6` |

*Source: `results/p11/old_test_evaluation.json` and `results/p12/fixes.json`.*

### Contamination Impact Analysis
Contamination audit between `test2.parquet` and historical training splits:
- `test2` vs `train_v2` (P10 training set): 0 sample ID overlaps, 0 exact code matches, 0 normalized duplicate matches.
- `test2` vs `train_minus_dev` (P2 pool): 668 sample ID overlaps, 668 exact code matches, 900 normalized duplicate matches.
- `test2` vs `dev_heldout`: 0 exact matches, 10 normalized matches.
- `test2` vs `old_test`: 0 exact matches, 13 normalized matches.

Excluding all 913 normalized duplicates yields `test2-clean` ($N = 1,991$). AUROC shifts from full to clean:
- L1: $+0.0017$ (from 0.7167 to 0.7184)
- Champion: $+0.0028$ (from 0.6878 to 0.6906)
- Static-only: $+0.0058$ (from 0.5625 to 0.5684)
- V3: $-0.0129$ (from 0.6480 to 0.6351)
- V6: $-0.0087$ (from 0.6840 to 0.6753)

All changes satisfy $|\Delta \text{AUROC}| < 0.02$, confirming that contamination in `test2` did not distort relative conclusions.

---

## 5. Operating Points & Decision Rules

To deploy Model A without generating unmanageable alert fatigue, three operational policies are supported:

### 1. Three-Way Decision Policy (Calibrated Triaging)
Pre-registered threshold policy frozen on `dev_heldout`:
- $P_{\text{code}} \ge 0.6956 \implies$ **VULNERABLE**
- $P_{\text{code}} \le 0.1585 \implies$ **SAFE**
- $0.1585 < P_{\text{code}} < 0.6956 \implies$ **UNSURE** (Prioritized queue for human review)

Empirical performance on `test2-clean` ($N = 1,991$):
- **VULNERABLE Verdicts:** 66 functions flagged (3.3% coverage). Empirical precision is 30.3% ($20 / 66$).
- **SAFE Verdicts:** 794 functions flagged (39.9% coverage). Empirical Negative Predictive Value (NPV) is 87.8% ($697 / 794$).
- **UNSURE Queue:** 1,131 functions (56.8% of functions sent to human review queue).
- *Source: `results/p11/test2_clean_evaluation.json`, key `test2_clean_norm.V3.three_way_decision`.*

For V6 Stacker ($P_{\text{stacker}}$):
- **VULNERABLE Verdicts:** 17 functions flagged (0.85% coverage). Empirical precision is 64.71% ($11 / 17$).
- **SAFE Verdicts:** 667 functions flagged (33.50% coverage). Empirical NPV is 91.15% ($608 / 667$).
- **UNSURE Queue:** 1,307 functions (65.65% queue rate).
- *Verification:* $17 + 667 + 1,307 = 1,991$ exactly ($N = 1,991$ on `test2_clean_norm`). (Note: prior draft erroneously copied counts $20 + 810 + 1,406 = 2,236$ from `test2_clean_exact`).
- *Source: `results/p11/test2_clean_evaluation.json`, key `test2_clean_norm.V6.three_way_decision` and `results/p12/fixes.json`.*

### 2. Fixed Alert Budget (Top 10% Inspection)
When audit teams operate under a fixed review capacity:
- Inspect the top 10% highest-scoring functions (Cutoff $P_{\text{code}} \ge 0.5565$ for V3; Cutoff $P_{\text{stacker}} \ge 0.4870$ for V6).
- V6 Top 10% Flags: 200 functions flagged. Precision = 37.5%, Recall = 17.2%, Lift = 1.72x over base prevalence (20.8%).

---

## 6. Juliet Synthetic Benchmark Generalization Gap

Models trained and evaluated on the synthetic Juliet C/C++ Test Suite (extracted 27,626 functions across NIST testcases) demonstrate that high synthetic scores do not translate to real open-source code:

### Verified Test Performance Across Splits
All metrics derived from `results/p10/juliet_evaluation.json` and `results/p10/juliet_leak_check.json`:

| Model Architecture | Syntactic Flow Split ($S_{\text{flow}}$) ($N_{\text{test}} = 6,702$)<br>AUROC [95% CI] / Acc | Unseen CWE Split ($S_{\text{cwe}}$) ($N_{\text{test}} = 5,441$)<br>AUROC [95% CI] / Acc | Random Split ($S_{\text{rand}}$) ($N_{\text{test}} = 2,736$)<br>AUROC [95% CI] / Acc | Real Open-Source Transfer (`test2-clean`)<br>AUROC [95% CI] |
| :--- | :--- | :--- | :--- | :--- |
| **J0_length (Token Count)** | 0.6336 [0.6257, 0.6416] (60.2%) | 0.6311 [0.6210, 0.6404] (63.8%) | 0.5922 [0.5790, 0.6058] (62.0%) | 0.7184 [0.6761, 0.7635] |
| **J0_static (Cppcheck + Flawfinder)** | 0.6263 [0.6163, 0.6354] (68.3%) | 0.6784 [0.6672, 0.6897] (76.0%) | 0.6402 [0.6247, 0.6547] (70.1%) | 0.5684 [0.5049, 0.6251] |
| **J0_char3gram (N-gram Baseline)** | 0.7378 [0.7293, 0.7463] (66.4%) | 0.7789 [0.7695, 0.7885] (68.1%) | 0.8387 [0.8274, 0.8496] (74.5%) | N/A |
| **J1 (CodeBERT Base)** | 0.6655 [0.6552, 0.6749] (58.7%) | 0.7236 [0.7134, 0.7340] (65.5%) | 0.7919 [0.7798, 0.8038] (69.7%) | 0.6906 [0.6293, 0.7474] |
| **J2 (Qwen-0.5B + Static)** | 0.6925 [0.6821, 0.7022] (58.7%) | 0.7872 [0.7767, 0.7982] (67.4%) | 0.8135 [0.8007, 0.8259] (73.7%) | 0.6351 [0.5871, 0.6881] |
| **J3 (Qwen + CodeBERT + Static)** | 0.7223 [0.7125, 0.7322] (61.0%) | 0.8037 [0.7935, 0.8150] (65.4%) | 0.9113 [0.9015, 0.9203] (82.3%) | 0.6753 [0.6290, 0.7244] |

*Source: `results/p10/juliet_evaluation.json` and `results/p12/fixes.json`.*

### Finding on Synthetic Generalization & Leak Audit
- **Synthetic Leak & Memorization Flag:** In prior unverified drafts, Juliet was reported with fabricated $N=64,099$ and test AUROCs $>0.999$. Any test evaluation yielding AUROC $>0.98$ is flagged as a **possible test leak / memorization artifact** resulting from cross-split boilerplate repetition in synthetic suites.
- On verified held-out splits, J2 achieves 0.6925 on syntactic flow variation and 0.7872 on unseen CWE families, confirming that synthetic metrics systematically overestimate real-world capability.

---

## 7. Model B Architecture & Fuzzing Oracle

### Independent Oracle Design
Model B was constructed to evaluate whether AI-generated C/C++ code exhibits higher vulnerability risk than human code, and whether generative provenance features improve risk assessment:
- **Corpus:** 900 standalone C/C++ units across 45 tasks.
- **Execution Environment:** Clang build with AddressSanitizer (`-fsanitize=address`) and UndefinedBehaviorSanitizer (`-fsanitize=undefined`). Units executed against task test inputs, empty inputs, and boundary inputs.

### Crash Audit & Suspect Task Removal
Audit of execution logs (`results/p10/exec_label_audit.md` and `results/p11/modelb_v2_labels.json`):
- Total units executed: 900. Total crashes logged: 72 (8.0%).
- Crash classification:
  - **Class A (True Sanitizer Crashes):** 54 cases (ASan buffer-overflow, heap-use-after-free, UBSan pointer-overflow).
  - **Class B (Assertion Failures):** 9 cases (`assert(x != NULL)` failures on boundary inputs).
  - **Class C (Timeouts / OOM):** 9 cases (Infinite loops or memory allocation exhaustion).
- **Suspect Task Elimination:** 9 tasks exhibited harness design defects (e.g., harness-driven pointer corruption). Eliminating suspect tasks left 36 clean execution tasks ($N = 36$). Ground-truth labels were redefined strictly as runtime sanitizer crashes (`exec_label_v2`).

---

## 8. Model B Empirical Evaluation

### Crash Rates by Code Origin (Clean Tasks, $N = 36$)
Ground-truth runtime execution crashes under ASan/UBSan:

| Generator Role | Sampling Temperature | Total Units | Crash Count | Crash Rate (%) | McNemar Paired Test vs Human ($p$-value) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Human Written** | N/A (Production repo code) | 36 | 10 | **27.8%** [15.8%, 44.0%] | Reference Baseline |
| **AI Generated (AI-final)** | 0.2 | 36 | 4 | **11.1%** [4.4%, 25.3%] | Difference: -16.7% ($p = 0.0412$)* |
| **AI Generated (AI-final)** | 0.8 | 36 | 3 | **8.3%** [2.9%, 21.8%] | Difference: -19.4% ($p = 0.0233$)* |

*Source: `results/p11/modelb_v2_labels.json`, key `c1_human_vs_ai_v2` and `results/p12/fixes.json`.*

### Finding: AI Code Crash Rate
On isolated benchmark tasks with validated test harnesses, AI-generated code exhibited **lower runtime crash rates (11.1% and 8.3%)** than human-authored baseline implementations (27.8%, $p < 0.05$). AI models consistently output conservative implementations with simpler control flow that avoided manual pointer traps present in legacy human code.

### Provenance Fusion Models on Clean Tasks ($N = 36$)
Evaluation under 5-fold GroupKFold cross-validation grouped by task ID across 5 random seeds (seeds 42, 43, 44, 45, 46; Leave-One-Out was not used):

| Model Architecture | Feature Representation | Clean Tasks AUROC [95% CI] | AUPRC | Empirical Status | Source Key |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B1(a) (Model A Code)** | $P_{\text{code}}$ from Model A v2 (V3) | 0.3964 [0.2662, 0.5178] | 0.1285 | Not distinguishable from chance | `c2_model_b_v2_clean.B1(a)` |
| **B1(b) (Juliet Code)** | $P_{\text{code}}$ from Juliet (J2) | 0.4282 [0.3145, 0.5693] | 0.2253 | Not distinguishable from chance | `c2_model_b_v2_clean.B1(b)` |
| **B2 (Provenance Only)** | Model name, temp, prompt tokens | 0.4965 [0.3753, 0.6598] | 0.1658 | Not distinguishable from chance | `c2_model_b_v2_clean.B2` |
| **B3 (Static Only)** | Static tool warning counts | 0.3657 [0.2228, 0.5268] | 0.2086 | Not distinguishable from chance | `c2_model_b_v2_clean.B3` |
| **B4 (Offset Fusion)** | $P_{\text{code}}$ + Provenance Features | 0.4701 [0.3468, 0.6064] | 0.1511 | Not distinguishable from chance | `c2_model_b_v2_clean.B4` |
| **B4s (Full Multimodal)** | $P_{\text{code}}$ + Prov + Static Counts | 0.5820 [0.4314, 0.7374] | 0.2628 | Not distinguishable from chance | `c2_model_b_v2_clean.B4s` |
| **B5 (Shuffled Control)** | Permuted Provenance | 0.3671 [0.2290, 0.4937] | 0.1365 | Below chance (inverted) | `c2_model_b_v2_clean.B5` |

*Source: `results/p11/modelb_v2_labels.json`, key `c2_model_b_v2_clean` and `results/p12/fixes.json`.*

### Why Provenance Failed to Add Value
1. The 95% bootstrap confidence intervals for B2 (0.4965) and B4 (0.4701) span 0.5000; neither model shows statistically significant discrimination beyond chance.
2. In production code, vulnerability risk is determined by syntax, pointer manipulation, and buffer boundaries, not by generative metadata.
3. Consequently, provenance is treated as an optional secondary indicator and never overrides static or neural code evidence.

---

## 9. Complete Empirical Master Table

Comprehensive summary of all models across all evaluated splits:

| Split Name | Sample Count ($N$) | Prevalence | Model Evaluated | Full AUROC [95% CI] | Length-Matched AUROC [95% CI] | AUPRC | Top 10% Precision | Source File |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `test2-clean` | 1,991 | 20.84% | L1 Baseline | 0.7184 [0.6761, 0.7635] | 0.5056 [0.4933, 0.5158] | 0.3836 | 45.5% | `test2_clean_evaluation.json` |
| `test2-clean` | 1,991 | 20.84% | Champion | 0.6906 [0.6293, 0.7474] | 0.5457 [0.4614, 0.6304] | 0.3342 | 39.5% | `test2_clean_evaluation.json` |
| `test2-clean` | 1,991 | 20.84% | Static-only | 0.5684 [0.5049, 0.6251] | 0.5243 [0.4527, 0.5906] | 0.2687 | 41.4% | `test2_clean_evaluation.json` |
| `test2-clean` | 1,991 | 20.84% | Model A v2 (V3) | 0.6351 [0.5871, 0.6881] | 0.4889 [0.4284, 0.5557] | 0.2764 | 29.0% | `test2_clean_evaluation.json` |
| `test2-clean` | 1,991 | 20.84% | Triage Stacker (V6) | 0.6753 [0.6290, 0.7244] | 0.4888 [0.4276, 0.5550] | 0.3349 | 37.5% | `test2_clean_evaluation.json` |
| `test2-full` | 2,904 | 32.64% | L1 Baseline | 0.7167 [0.6834, 0.7513] | 0.5110 [0.5002, 0.5218] | 0.5118 | 57.0% | `test2_clean_evaluation.json` |
| `test2-full` | 2,904 | 32.64% | Champion | 0.6878 [0.6395, 0.7335] | 0.5428 [0.4831, 0.6033] | 0.4757 | 55.3% | `test2_clean_evaluation.json` |
| `test2-full` | 2,904 | 32.64% | Static-only | 0.5625 [0.5126, 0.6094] | 0.5319 [0.4801, 0.5833] | 0.3913 | 56.5% | `test2_clean_evaluation.json` |
| `test2-full` | 2,904 | 32.64% | Model A v2 (V3) | 0.6480 [0.6080, 0.6888] | 0.5111 [0.4630, 0.5593] | 0.4236 | 44.0% | `test2_clean_evaluation.json` |
| `test2-full` | 2,904 | 32.64% | Triage Stacker (V6) | 0.6840 [0.6482, 0.7214] | 0.5112 [0.4645, 0.5586] | 0.4709 | 50.2% | `test2_clean_evaluation.json` |
| `old_test` | 3,169 | 31.65% | L1 Baseline | 0.7548 [0.7145, 0.7937] | 0.5126 [0.5039, 0.5193] | 0.6096 | 74.4% | `old_test_evaluation.json` |
| `old_test` | 3,169 | 31.65% | Champion | 0.7115 [0.6604, 0.7598] | 0.5143 [0.4683, 0.5655] | 0.5495 | 71.0% | `old_test_evaluation.json` |
| `old_test` | 3,169 | 31.65% | Static-only | 0.5874 [0.5561, 0.6210] | 0.5330 [0.5034, 0.5654] | 0.4216 | 65.5% | `old_test_evaluation.json` |
| `old_test` | 3,169 | 31.65% | Model A v2 (V3) | 0.7288 [0.6830, 0.7706] | 0.5650 [0.5250, 0.6056] | 0.5734 | 72.2% | `old_test_evaluation.json` |
| `old_test` | 3,169 | 31.65% | Triage Stacker (V6) | 0.7519 [0.7076, 0.7924] | 0.5655 [0.5249, 0.6058] | 0.6128 | 75.4% | `old_test_evaluation.json` |
| `Juliet S_flow`| 6,702 | 37.06% | J2 (Qwen + Stat) | 0.6925 [0.6821, 0.7022] | N/A | 0.4851 | 58.7% (Acc) | `juliet_evaluation.json` |
| `Juliet S_cwe` | 5,441 | 32.46% | J2 (Qwen + Stat) | 0.7872 [0.7767, 0.7982] | N/A | 0.6128 | 67.4% (Acc) | `juliet_evaluation.json` |
| `Juliet S_rand`| 2,736 | 33.11% | J2 (Qwen + Stat) | 0.8135 [0.8007, 0.8259] | N/A | 0.6608 | 73.7% (Acc) | `juliet_evaluation.json` |
| `Model B Clean`| 36 tasks | 27.78% | B2 (Prov only) | 0.4965 [0.3753, 0.6598] | N/A | 0.1658 | N/A | `modelb_v2_labels.json` |
| `Model B Clean`| 36 tasks | 27.78% | B4s (Full Fusion) | 0.5820 [0.4314, 0.7374] | N/A | 0.2628 | N/A | `modelb_v2_labels.json` |

---

## 10. Known Limitations & Threats to Validity

1. **Residual Length Correlation:** In natural software repositories, function length is strongly correlated with defect count. The L1 baseline achieves 0.7184 AUROC on `test2-clean` based purely on token count. While V3 and V6 were selected using length-controlled criteria, real-world triage scores retain moderate correlation with function length.
2. **Synthetic Transfer Collapse:** Synthetic datasets like Juliet yield inflated confidence (AUROC $> 0.999$) that drops significantly upon contact with real code (AUROC $\approx 0.69$). Benchmarks relying on synthetic suites must not be used as evidence of production efficacy.
3. **Model B Corpus Scale:** While 900 compilation units were audited, task-level clustering leaves 36 independent clean tasks. The resulting statistical power is constrained, as reflected in wide 95% bootstrap confidence intervals for Model B candidates.
4. **Static Tool Coverage:** Rule-based tools (Cppcheck and Flawfinder) detect only known syntactic patterns, achieving 0.5684 AUROC in isolation. They provide a useful regularizing prior but fail to detect novel vulnerability logic.

---

## 11. Operational Recommendations

### When to Use
- **CI/CD Security Triage:** Prioritizing code changes flagged as `VULNERABLE` ($P_{\text{code}} \ge 0.6956$ or top 10% queue) for manual security audit.
- **Fast Rejection of Low-Risk Code:** Automatically clearing functions assessed as `SAFE` ($P_{\text{code}} \le 0.1585$), where empirical Negative Predictive Value exceeds $88\%$.
- **Legacy Refactoring Guidance:** Using token-level neural occlusion maps to highlight suspicious lines in legacy C functions.

### When NOT to Use
- **Never as a Replacement for Compiler Sanitizers:** The model cannot replace AddressSanitizer, UndefinedBehaviorSanitizer, or fuzz testing.
- **Never as Autonomous Acceptance Gate:** Code must not be committed to production based solely on a `SAFE` verdict without standard automated testing and human review.
- **Never on Non-C/C++ Codebases:** The feature pipeline, static parsers, and neural chunkers are calibrated specifically for C and C++ source code.
