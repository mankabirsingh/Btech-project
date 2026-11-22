# Phase 16 — Prompt 5: Multi-Seed Confirmation, Pairwise Regularisation, and Model A Selection

## 1. Executive Summary

This report documents the resolution of the evaluation discrepancies identified at the conclusion of P4, the execution of multi-seed confirmation across 5 random seeds, the evaluation of pairwise-regularised fine-tuning (PAIR recipe), and the mechanical selection and packaging of the final Model A production candidate.

All evaluations strictly adhere to the project boundaries: training was conducted solely on `train_minus_dev` (plus clean source pool pairs), validation and model selection were performed strictly on `dev_heldout` and `dev_pairs`, and the final held-out test parquet remains locked and untouched.

### Key Metrics Across Milestones

| Milestone / Candidate | Dev AUROC [95% CI] | Dev LS-AUROC (4-bin) [95% CI] | Dev LS-AUROC (10-bin) [95% CI] | Length-Matched AUROC (46,251 pairs) [95% CI] | Dev C++ AUROC | Pair Correct (186 pairs) [95% CI] | Top 10% Alert Precision (Lift) |
|---|---|---|---|---|---|---|---|
| **L1 (Length-Only Baseline)** | 0.7153 [0.6945, 0.7361] | 0.5899 [0.5631, 0.6170] | 0.5314 [0.5018, 0.5571] | 0.5101 [0.5056, 0.5147] | 0.7077 | 21.2% [16.1%, 27.2%] | 53.9% (1.78x) |
| **Phase 15 / P2 Faithful Champion** | 0.7109 [0.6903, 0.7319] | 0.6074 [0.5809, 0.6333] | 0.5801 [0.5503, 0.6069] | 0.5633 [0.5587, 0.5681] | 0.6895 | 47.3% [40.3%, 54.3%] | 57.5% (1.90x) |
| **P4 Single-Seed Best (F3_max)** | 0.7178 [0.6976, 0.7383] | 0.6665 [0.6393, 0.6943] | 0.6536 [0.6238, 0.6828] | 0.6619 [0.6578, 0.6663] | 0.7225 | 55.4% [48.4%, 62.4%] | 68.9% (2.27x) |
| **P5 N0 Frozen Ensemble (2 seeds)** | 0.7131 [0.6931, 0.7337] | 0.6093 [0.5832, 0.6351] | 0.5826 [0.5531, 0.6092] | 0.5636 [0.5591, 0.5682] | 0.6978 | 48.7% [41.7%, 55.6%] | 55.5% (1.83x) |
| **P5 PAIR Ensemble (3 seeds)** | 0.7082 [0.6873, 0.7302] | 0.6513 [0.6263, 0.6769] | 0.6315 [0.6029, 0.6590] | 0.6298 [0.6252, 0.6343] | 0.6873 | 58.3% [51.3%, 65.1%] | 66.1% (2.18x) |
| **P5 D2 Winning Ensemble (5 seeds)** | **0.7371** [0.7161, 0.7581] | **0.6574** [0.6322, 0.6838] | **0.6381** [0.6113, 0.6653] | **0.6374** [0.6330, 0.6418] | **0.7144** | **52.7%** [45.7%, 60.2%] | **72.0% (2.38x)** |

---

## 2. Resolution of Part A Evaluation Checks (A1–A5)

### A1: Root Cause of F3 vs L1 vs Champion P-Value Discrepancy
In P4, F3 vs L1 exhibited $p < 0.0005$ whereas F3 vs Champion yielded an unadjusted $p = 0.0530$ and Holm-adjusted $p = 0.5300$. Two distinct root causes were identified:
1. **Hyperparameter Inconsistency in P4 Script:** In `scripts/p4/step4_master_selection.py:80`, Champion P2 was evaluated using `LogisticRegression(C=0.003)` (the hyperparameter tuned for R2 in P3) rather than the faithful `C=0.01` specified in `scripts/p2/evaluate_reference_p2.py:77`. Under the faithful recipe ($C=0.01$), the observed difference on dev LS-AUROC is $\Delta = +0.0590$ with 95% bootstrap CI `[+0.0047, +0.0994]`, yielding an empirical two-sided $p = 0.0300 < 0.05$.
2. **Cluster Bootstrap Variance from Skewed Commit Sizes:** `dev_heldout` comprises 244 distinct commit clusters, but two dominant commits (`qpdf` with 372 samples and `openexr` with 282 samples) account for 25.7% of all samples. Clustered resampling by commit introduces substantial variance across bootstrap draws. In contrast, L1 uses an entirely orthogonal feature space (token length only), such that F3 achieved higher LS-AUROC than L1 in every single one of the 2,000 bootstrap resamples ($p < 0.0005$).

### A2: Number Consistency Across Milestones
- **Champion C++ AUROC (0.6895 vs 0.6995):** In P2 (`scripts/p2/evaluate_reference_p2.py:77`), $C=0.01$ was used, giving C++ AUROC = **0.6895**. In P4 (`scripts/p4/step4_master_selection.py:80`), $C=0.003$ was accidentally used, giving **0.6995**. The faithful number is **0.6895** ($C=0.01$).
- **R2 C++ AUROC (0.6769 vs 0.6997):** In P3 (`scripts/p3/step3_representation.py:62-64` and `scripts/p3/eval_protocol.py:121`), `StandardScaler` was applied globally across all 1,564 features, standardizing the dense 1,536 CodeBERT dimensions and distorting the unit-sphere embedding geometry, which degraded C++ AUROC to **0.6769**. In P4 (`scripts/p4/step4_master_selection.py:77-87`), `StandardScaler` was applied selectively to `cxx28` only (as established in Phase 15 and P2), preserving embedding structure and restoring C++ AUROC to **0.6997**.

### A3: Quantile Binning Sensitivity (4-Bin vs 10-Quantile LS-AUROC)
Recomputing LS-AUROC using 10 equal-sample quantile bins (edges: 0, 46, 87, 149, 226, 340, 499, 766, 1174, 2355, inf) confirmed stability:
- The relative ranking of the top-performing models remained stable: F3_max (0.6536) and F3_mean (0.6482) lead under both 4-bin and 10-bin definitions.
- L1 LS-AUROC degraded further under finer quantile stratification: from **0.5899** (4-bin) to **0.5314** (10-bin), demonstrating that length-only features lose discriminative power as length bins narrow.

### A4: Alert-Budget Operating Evaluation (Top 10%)
Flagging the top 10% most suspicious functions on `dev_heldout` (threshold $\tau = 0.6874$ for D2_ensemble) yielded:
- **Overall:** Precision = **72.05%**, Recall = **23.74%**, Lift = **2.38x** over the 30.32% baseline prevalence.
- **Per-Project Breakdown:**
  - `qpdf` (prevalence 38.49%): Precision = **79.63%**, Recall = **20.57%**, Lift = **2.07x**.
  - `openexr` (prevalence 12.31%): Precision = **43.59%**, Recall = **35.42%**, Lift = **3.54x**.
  - `tcpdump` (prevalence 38.54%): Precision = **80.52%**, Recall = **20.95%**, Lift = **2.09x**.
  - `radare2` (prevalence 25.89%): Precision = **71.43%**, Recall = **27.52%**, Lift = **2.76x**.
- **Paired Operating Partition on `dev_pairs` (186 pairs):**
  - P-C (vulnerable flagged, fixed safe): **6.99%**
  - P-V (both flagged — high-risk commit): **26.34%**
  - P-B (neither flagged): **63.98%**
  - P-R (reversed alarm — fixed flagged, vuln missed): **2.69%**
  - Alert operating ratio $\frac{\text{P-C}}{\text{P-C} + \text{P-R}} = \frac{6.99}{6.99 + 2.69} = \mathbf{72.2\%}$, demonstrating that when the alert budget differentiates between pair halves, it correctly flags the vulnerable function in over 70% of cases.

### A5: Length-Matched Pair Evaluation
Evaluating all 46,251 (vulnerable, safe) pairs in `dev_heldout` with $|\Delta \log(\text{tokens})| \le 0.1$ revealed:
- **L1 Baseline:** AUROC = **0.5101** [95% CI: 0.5056, 0.5147] (degrades to chance 0.50 by construction).
- **Faithful Champion P2:** AUROC = **0.5633** [95% CI: 0.5587, 0.5681].
- **D2 Ensemble:** AUROC = **0.6374** [95% CI: 0.6330, 0.6418].
This demonstrates that the D2 fine-tuning recipe learns genuine vulnerability features independent of sequence length.

---

## 3. Pre-Registration Verification
- **Pre-registration artifact:** `results/p5/preregistration.json` (created 2026-11-20 16:55:00 UTC).
- **Earliest training run timestamp:** `results/p5/D2_seed1_log.csv` (started 2026-11-20 22:42:00 UTC).
- **Verification:** The pre-registration document was committed and locked prior to initiating any model training. The planned runs (C1, C2, C3), ensemble aggregation formula (unweighted average of logits), Gate 1 criterion (length-matched AUROC vs L1 with Holm $p < 0.05$), Gate 2 criterion (dev LS-AUROC vs Champion), and parsimony tiebreaker were established before observing multi-seed outcomes.

---

## 4. Multi-Seed Training Results

### Seed Statistics Across Model Families (Mean ± Std)

| Recipe Family | Seeds Evaluated | Dev AUROC | Dev LS-AUROC (4-bin) | Dev LS-AUROC (10-bin) | Length-Matched AUROC | Dev C++ AUROC | Pair Correct |
|---|---|---|---|---|---|---|---|
| **D2 (Propensity-Weighted)** | 5 seeds (42, 7, 1, 2, 3) | 0.7146 ± 0.0287 | 0.6434 ± 0.0221 | 0.6244 ± 0.0263 | 0.6249 ± 0.0238 | 0.6998 ± 0.0201 | 54.8% ± 2.0% |
| **PAIR (Pairwise-Regularised)** | 3 seeds (42, 7, 1) | 0.6928 ± 0.0324 | 0.6405 ± 0.0112 | 0.6252 ± 0.0070 | 0.6224 ± 0.0073 | 0.6649 ± 0.0794 | 59.5% ± 2.0% |
| **N0 (Frozen CodeBERT Control)** | 2 seeds (42, 7) | 0.7126 ± 0.0019 | 0.6082 ± 0.0020 | 0.5811 ± 0.0024 | 0.5630 ± 0.0013 | 0.6973 ± 0.0094 | 48.3% ± 1.0% |

### Per-Seed Detail for D2 Family
- **Seed 42 (F1):** Dev AUROC = 0.6656, LS-AUROC = 0.6170, Length-Matched = 0.6066, Pair Correct = 56.2%.
- **Seed 7 (F3):** Dev AUROC = 0.7178, LS-AUROC = 0.6665, Length-Matched = 0.6619, Pair Correct = 55.9%.
- **Seed 1:** Dev AUROC = 0.7406, LS-AUROC = 0.6606, Length-Matched = 0.6285, Pair Correct = 51.6%.
- **Seed 2:** Dev AUROC = 0.7267, LS-AUROC = 0.6235, Length-Matched = 0.6014, Pair Correct = 56.5%.
- **Seed 3:** Dev AUROC = 0.7222, LS-AUROC = 0.6493, Length-Matched = 0.6261, Pair Correct = 54.0%.

---

## 5. Recipe Ensembling and Paired Comparisons

Ensembles were constructed strictly according to the pre-registered specification by taking the unweighted arithmetic mean of predicted logits across all constituent seeds, followed by the sigmoid transform.

### Paired Hypothesis Testing (2,000 Clustered Bootstrap Resamples by Commit)

| Comparison | Metric | Observed Difference ($\Delta$) | 95% Cluster Bootstrap CI | Raw $p$-value | Holm-Adjusted $p$-value |
|---|---|---|---|---|---|
| **D2_ensemble vs L1** | Length-Matched AUROC | +0.1273 | [+0.0985, +0.1808] | <0.0005 | **<0.0005** |
| **D2_ensemble vs L1** | Dev LS-AUROC | +0.0675 | [+0.0354, +0.1240] | <0.0005 | **<0.0005** |
| **D2_ensemble vs Champion** | Length-Matched AUROC | +0.0741 | [+0.0203, +0.1136] | 0.0050 | **0.0150** |
| **D2_ensemble vs Champion** | Dev LS-AUROC | +0.0500 | [+0.0003, +0.0880] | 0.0500 | **0.1500** |
| **PAIR_ensemble vs L1** | Length-Matched AUROC | +0.1197 | [+0.0868, +0.1797] | <0.0005 | **<0.0005** |
| **PAIR_ensemble vs Champion** | Dev LS-AUROC | +0.0438 | [-0.0036, +0.0806] | 0.0740 | **0.1500** |
| **PAIR_ens vs D2_ens** | Length-Matched AUROC | -0.0076 | [-0.0233, +0.0131] | 0.4260 | — |
| **PAIR_ens vs D2_ens** | Dev LS-AUROC | -0.0061 | [-0.0221, +0.0135] | 0.4510 | — |
| **PAIR_ens vs D2_ens** | Pair Correct | +0.0565 | [-0.0189, +0.1371] | 0.1410 | — |

---

## 6. Winning Model Selection and Packaging Confirmation

### Application of Pre-Registered Selection Protocol
1. **Gate 1 (Length-Matched Discrimination vs L1):** Both `D2_ensemble` ($\Delta = +0.1273$, Holm $p < 0.0005$) and `PAIR_ensemble` ($\Delta = +0.1197$, Holm $p < 0.0005$) passed Gate 1.
2. **Gate 2 (Non-Degradation vs Champion):** Both `D2_ensemble` ($\Delta = +0.0500$, $p = 0.0500$) and `PAIR_ensemble` ($\Delta = +0.0438$, $p = 0.0740$) passed Gate 2.
3. **Composite Scoring:** `PAIR_ensemble` scored 0.6066 and `D2_ensemble` scored 0.5821.
4. **Pre-Registered Tie-Breaker Execution:** On the primary metric `length_matched_auroc`, `PAIR_ensemble` (0.6298) and `D2_ensemble` (0.6374) are statistically indistinguishable (difference = -0.0076, 95% CI `[-0.0233, +0.0131]`, $p = 0.4260$). Per the pre-registered parsimony protocol, `D2_ensemble` is selected as the winning model to avoid unnecessary pairwise training overhead.

### Packaging Details
The final production Model A candidate is packaged in `models/checkpoints/p5/model_a_final/`:
- **Weight Checkpoints:** All 5 seed checkpoints (`d2_seed42.pt`, `d2_seed7.pt`, `d2_seed1.pt`, `d2_seed2.pt`, `d2_seed3.pt`) with verified SHA-256 digests.
- **Inference Interface:** `predictor.py` exposing a standalone `predict(code_list: List[str]) -> np.ndarray` entry point supporting batched sliding-window inference and logit averaging across all 5 constituent seeds.
- **Manifest:** Complete configuration and evaluation metadata saved in `models/checkpoints/p5/model_a_final/manifest.json`.
- **Unit Test:** Verified via `tests/test_model_a_final.py` (`Ran 1 test in 6.176s — OK`).

---

## 7. Remaining Limitations and Risk Analysis

While Model A exhibits measurable discrimination beyond function length, the empirical findings identify several concrete limitations:
1. **Pair-Correct Accuracy Remains Moderate:** Across the 186 commit-paired functions in `dev_pairs`, the winning ensemble achieves **52.69%** accuracy (PAIR ensemble reaches 58.33%). Detecting fine-grained vulnerabilities where a commit modifies only 1–3 lines remains challenging.
2. **False Alarms Under Operating Budgets:** At the top-10% alert operating point, **26.34%** of pairs have both the vulnerable and fixed versions flagged (P-V), indicating that commit context still influences scores even when individual function lengths are controlled.
3. **Cross-Project Generalization Variance:** While lift over prevalence is consistently positive across all evaluated dev projects (ranging from 2.07x on `qpdf` to 3.54x on `openexr`), absolute precision is bounded by base rate prevalence (43.59% on `openexr` vs 80.52% on `tcpdump`).
4. **Computational Cost:** Evaluating an ensemble of 5 UniXcoder models with sliding windows requires approximately 1.2 seconds per function on GPU/MPS and is computationally heavier than the frozen logistic regression baseline.
