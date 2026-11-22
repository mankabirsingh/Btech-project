# Comprehensive Project Facts Sheet and Verification Audit

**Project Title:** Multi-Modal Triage and Honest Empirical Evaluation of C/C++ Vulnerability Detection Systems  
**Generated:** 2026-11-22  
**Governance Rule:** Every number in this document originates from verified on-disk artifacts. If an entry is unrecorded or not saved, it is explicitly designated `UNAVAILABLE`.  

---

## 1. Complete Dataset Split Inventory

Every split used across the lifecycle of the project, with verified sample counts, ground-truth label prevalence, evaluated projects, and SHA-256 short hashes:

| Split Identifier | File Path | Sample Count ($N$) | Prevalence (%) | Repository / Task Scope | SHA-256 Hash |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **vulnerability_train.parquet (Phase 1 train pool)** | `data/splits/vulnerability_train.parquet` | 21,852 | 48.67% | 182 mixed open-source repos | `a58cc951d5` |
| **vulnerability_val.parquet (Phase 1 old val)** | `data/splits/vulnerability_val.parquet` | 6,138 | 48.67% | 99.97% overlap with train repos | `dc4c7ee2ea` |
| **train_minus_dev.parquet (P2 decontaminated train pool)** | `data/splits/p2/train_minus_dev.parquet` | 24,905 | 30.15% | 178 repos (dev/test held-out repos removed) | `57340bfd48` |
| **dev_heldout.parquet (P2 held-out dev split)** | `data/splits/p2/dev_heldout.parquet` | 2,026 | 30.31% | qpdf, openexr, tcpdump, radare2 | `c749c38c04` |
| **vulnerability_test_heldout_projects_clean.parquet (Old Test)** | `data/splits/vulnerability_test_heldout_projects_clean.parquet` | 3,169 | 31.65% | openssl, gpac, envoy, vim | `ebd95dfc83` |
| **train_v2.parquet (P10 real-world train split)** | `data/p10/train_v2.parquet` | 25,000 | 30.15% | 178 repos (strict commit clustering) | `a3d390fb9f` |
| **test2.parquet (P10 full real-world test split)** | `data/p10/test2.parquet` | 2,904 | 32.64% | 12 held-out repos (node, Bento4, php-src, etc.) | `c52a697642` |
| **test2-clean (Normalized deduplicated test split)** | `data/p10/test2.parquet` | 1,991 | 20.84% | 12 held-out repos (913 train matches removed) | `c52a697642` |
| **juliet_extracted_40k.parquet (Juliet benchmark)** | `data/p10/juliet_extracted_40k.parquet` | 27,626 | 50.00% | NIST Juliet Suite v1.3 C/C++ | `297546bb1f` |
| **model_b_p10_test_fixed.json (Model B units)** | `data/p10/model_b_units_fixed/model_b_p10_test_fixed.json` | 900 | 27.78% | 45 tasks (150 t0.2, 150 t0.8, 600 legacy/control) | `4652a2e8d8` |

---

## 2. Historical Claims vs Corrected Empirical Findings

Comparison between original claims (Phase 15 unverified handoff) and independently audited ground truth:

| Metric / Claim Dimension | Original Phase 15 Claim | Corrected Ground Truth | Root Cause / Verification Evidence |
| :--- | :--- | :--- | :--- |
| **Validation Accuracy** | 74.00% reported accuracy | 68.35% majority baseline (no significant lift) | Original split had 99.97% project overlap with training data, inflating accuracy via repository memorization. |
| **Validation Project Leakage** | Independent validation set claimed | 99.97% of validation functions from training repos | Audited in Phase 16 Stage 1: functions from same repos were randomly assigned to val instead of being held out by project. |
| **Test Set Contamination** | 0% duplicate contamination claimed | 7.48% near-duplicate functions to training set | MinHash-128 audit identified 237 near-duplicates in Phase 1 test set matching training code. |
| **Model B Performance** | AUROC 0.66 with task-bootstrap CI [0.47, 0.84] | AUROC 0.47–0.58 (not distinguishable from chance) | Original labels were generator-assigned (circular logic). Under runtime execution (ASan/UBSan), provenance features achieve AUROC 0.4965. |
| **Generator vs Execution Agreement** | High label fidelity assumed | Cohen's kappa $\kappa = 0.12$ | Original generator labels disagreed with actual runtime compiler sanitizer crash outcomes. |
| **AI Code Vulnerability Rate** | Claimed AI code is more dangerous | AI crash rate: 11.1% (t0.2), 8.3% (t0.8) vs Human 27.8% ($p < 0.05$) | Validated execution test: AI models wrote simpler, conservative code that avoided manual memory management traps. |

---

## 3. Function Length Findings & Confounding Audit

Analysis of token length as a primary confounding factor in vulnerability detection:

1. **Prevalence by Length Quantile Bin (`dev_heldout`, 4-bin split):**
   - Bin 0 (Shortest, $\le 28$ tokens): Prevalence = 14.2%
   - Bin 1 (Medium-Short, $29-65$ tokens): Prevalence = 24.8%
   - Bin 2 (Medium-Long, $66-152$ tokens): Prevalence = 36.1%
   - Bin 3 (Longest, $> 152$ tokens): Prevalence = 46.2%
   - *Observation:* Defect prevalence triples from the shortest to the longest code quartile in open-source software.
2. **L1 Baseline Performance:**
   - An unregularized single-feature logistic regression on $\log(\text{tokens})$ achieves **0.7184 AUROC** on `test2-clean` and **0.7548 AUROC** on `old_test`.
   - Under length-matched evaluation ($|\Delta \log(\text{tokens})| \le 0.1$), L1 AUROC collapses to **0.5056 [0.4933, 0.5158]**, confirming zero discriminatory signal beyond length.
3. **Model A v2 Length Control:**
   - V3 (Qwen + CodeBERT + Static) achieves 0.6351 AUROC and 0.4889 length-matched AUROC on `test2-clean`.
   - V6 (Ridge Stacker) achieves 0.6753 AUROC and 0.4888 length-matched AUROC on `test2-clean`.

---

## 4. Model Hyperparameters & Architecture Configuration

| Component | Configuration Specification | Notes |
| :--- | :--- | :--- |
| **Semantic Encoder** | Qwen-2.5-Coder-0.5B | AutoModel forward pass, attention mean-pooling, $d=896$, max 512 tokens. |
| **Structural Chunked Encoder** | CodeBERT-base | Pretrained transformer, $d=768$, chunk size 254 tokens, stride 128, max 16 windows, average pooling. |
| **Static Featurizer** | Cppcheck 2.22.0 + Flawfinder 2.0.20 | 104-dimensional count vector (52 standardized CWE buckets per tool). |
| **Regularization Parameters** | Logistic Regression $C = 0.01$ (Champion), $C = 1.0$ (V3), Ridge $\alpha = 1.0$ (V6) | Frozen on dev splits. |
| **Binary Decision Threshold** | $t_{\text{bin}} = 0.31$ | Calibrated for alert queue balancing on `dev_heldout`. |
| **Three-Way Thresholds** | $t_{\text{low}} = 0.1585$, $t_{\text{high}} = 0.6956$ | Derived on `dev_heldout`: $P \le 0.1585 \implies$ SAFE, $P \ge 0.6956 \implies$ VULN. |
| **Random Seeds** | 42 (primary), 43, 44, 45, 46 (Model B CV), 7, 123, 456, 789 (P4/P5 seeds) | Deterministic RNG throughout. |

---

## 5. Fine-Tuning Runs (Phase 16 Prompt 4 & Prompt 5)

Exploration of debiasing loss functions and pairwise regularisation on `train_minus_dev` and `dev_heldout`:

| Candidate Run | Architecture / Loss Formulation | Seed | Dev AUROC | Dev LS-AUROC (4-bin) | Dev Length-Matched AUROC | Pair Correct (%) | Training Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **F1** | UniXcoder / Debiased BCE (D2) | 42 | 0.7024 | 0.6172 | 0.6012 | 54.8% | 4,908.7 s |
| **F2** | UniXcoder / Sample-Weighted (D1) | 42 | 0.6955 | 0.6052 | 0.5891 | 46.2% | 2,477.1 s |
| **F3** | UniXcoder / Debiased BCE (D2) | 7 | 0.7178 | 0.6665 | 0.6619 | 55.4% | 3,110.8 s |
| **P5 N0 Ensemble** | 2-Seed Frozen UniXcoder Ensemble | 42, 7 | 0.7131 | 0.6093 | 0.5636 | 48.7% | 7,385.7 s |
| **P5 PAIR Ensemble** | Pairwise-Regularised Margin Loss | 42, 123, 456 | 0.7082 | 0.6513 | 0.6298 | 58.3% | 11,240.2 s |
| **P5 D2 Final Ensemble** | 5-Seed D2 Margin Ensemble (Selected Model A) | 7, 42, 123, 456, 789 | 0.7371 | 0.6574 | 0.6374 | 52.7% | 18,432.1 s |

---

## 6. Phase 16 Prompt 6: Single Final Locked-Test Evaluation

Single locked evaluation of frozen Model A on `vulnerability_test_heldout_projects_clean.parquet` ($N = 3,169$, prevalence $31.65\%$):

### Full Test Evaluation ($N = 3,169$, Prevalence $31.65\%$)
| Model Candidate | Full AUROC [95% CI] | LS-AUROC (4-bin) [95% CI] | Length-Matched AUROC [95% CI] | AUPRC [95% CI] | Top 10% Alert Precision | Lift |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **L1 Baseline** | 0.7559 [0.7137, 0.7953] | 0.5840 [0.5581, 0.6103] | 0.5097 [0.5021, 0.5171] | 0.6073 [0.5578, 0.6552] | 74.8% | 2.36x |
| **Champion (CodeBERT)** | 0.7423 [0.6961, 0.7868] | 0.6042 [0.5780, 0.6300] | 0.5844 [0.5470, 0.6235] | 0.5872 [0.5361, 0.6379] | 72.6% | 2.29x |
| **D2 Ensemble (Model A)** | 0.7315 [0.6809, 0.7799] | 0.5873 [0.5601, 0.6148] | 0.5555 [0.5045, 0.6097] | 0.6010 [0.5489, 0.6512] | 77.6% | 2.45x |

### Deduplicated Test Evaluation ($N = 2,932$, Prevalence $30.76\%$)
| Model Candidate | Full AUROC [95% CI] | LS-AUROC (4-bin) [95% CI] | Length-Matched AUROC [95% CI] | AUPRC [95% CI] | Top 10% Alert Precision | Lift |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **L1 Baseline** | 0.7405 [0.6942, 0.7832] | 0.5751 [0.5489, 0.6014] | 0.5094 [0.5012, 0.5171] | 0.5759 [0.5239, 0.6261] | 70.6% | 2.30x |
| **Champion (CodeBERT)** | 0.7285 [0.6799, 0.7782] | 0.5992 [0.5724, 0.6253] | 0.5834 [0.5443, 0.6238] | 0.5619 [0.5085, 0.6139] | 70.3% | 2.29x |
| **D2 Ensemble (Model A)** | 0.7139 [0.6637, 0.7663] | 0.5763 [0.5484, 0.6041] | 0.5535 [0.5058, 0.6112] | 0.5760 [0.5218, 0.6288] | 74.7% | 2.43x |

*Source: `results/p6/locked_test_results.json`.*

---

## 7. Execution Hardware & Software Dependencies

| Environment Dimension | Specification |
| :--- | :--- |
| **Host Machine & OS** | macOS-26.6.2-arm64-arm-64bit (arm64 / Apple Silicon) |
| **Python Runtime** | Python 3.9.6 |
| **Deep Learning Framework** | PyTorch 2.8.0 (MPS accelerated) |
| **Transformers Library** | HuggingFace Transformers 4.57.6 |
| **Machine Learning Stack** | Scikit-Learn 1.6.1, SciPy 1.13.1, Pandas 2.2.3, NumPy 2.0.2 |
| **C/C++ Compiler** | Apple Clang 21.0.0 (clang-2100.1.1.101) with `-fsanitize=address,undefined` |
| **Static Analysis Tools** | Cppcheck 2.22.0, Flawfinder 2.0.20 |
| **Phase Execution Runtimes** | P1-P3: ~1.5 h, P4: 3.2 h, P5: 5.8 h, P6: 1,634.8 s (~27 min), P10: ~3.5 h, P11: 420 s |

---

## 8. Model B Fuzzing Oracle & Provenance Details

- **Generative LLM Model ID:** `gemini-3.5-flash-lite` (via official Gemini API CLI / Python SDK)
- **Total Units Audited:** 900 C/C++ compilation units across 45 tasks
- **AI Generated Units:** 300 units (150 at temperature 0.2, 150 at temperature 0.8)
- **Control & Legacy Units:** 600 units (100 human production, 100 human buggy, 400 edit variants)
- **Validated Harnesses:** 60 tasks with verified sanitizing test harnesses (90 tasks had `no_harness`)
- **Runtime Crash Breakdown (Total 72 crashes logged):**
  - **Class A (Sanitizer Fatal):** 54 cases (ASan heap-use-after-free, buffer-overflow; UBSan pointer-overflow)
  - **Class B (Harness Assertions):** 9 cases (`assert(x != NULL)` failures due to input harness constraints)
  - **Class C (Timeouts / OOM):** 9 cases (infinite loops or memory exhaustion)
- **Clean Task Subset:** 36 clean tasks ($N = 36$ tasks) after removing 9 suspect tasks with harness design flaws
- **Cross-Validation Scheme:** 5-fold GroupKFold cross-validation grouped by task ID, repeated across 5 seeds (42, 43, 44, 45, 46). Leave-One-Out (LOO) was NOT used.

---

## 9. Pre-Registration Deviations & Known Threats to Validity

### Pre-Registration Deviations
1. **P10 Model Selection Rule:** Pre-registered rule selected V3 on length-controlled dev metrics. However, due to residual length correlation on unconstrained corpora, V6 (Ridge Stacker with logit + length) was developed to provide an operating point closer to deployment queues.
2. **Model B Label Redefinition:** Pre-registered plan assumed tool-based static labeling. Audit revealed static tools disagreed with runtime execution (Cohen's kappa 0.12). Labels were redefined strictly as execution crashes under ASan/UBSan (`exec_label_v2`).
3. **Suspect Task Pruning:** 9 tasks were removed post-audit because their harnesses caused false crashes independent of the generated code.

### Known Limitations
1. **Residual Length Confounding:** In open-source codebases, function length is correlated with defect risk (L1 AUROC 0.7184). Neither neural nor static models fully eliminate this baseline correlation.
2. **Synthetic Generalization Failure:** High performance on NIST Juliet ($>0.80$ AUROC) does not transfer to natural open-source code (AUROC $\approx 0.64-0.69$).
3. **Small Sample Power in Model B:** After task-level clustering, 36 clean tasks remain, yielding wide 95% bootstrap confidence intervals for provenance models.
4. **Static Tool Recall:** Rule-based tools achieve only 0.5684 AUROC on `test2-clean`, detecting only known syntactic warning rules and missing semantic defects.