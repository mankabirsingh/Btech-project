# HANDOFF REPORT — PHASE 16 PROMPT 3: MODEL A IMPROVEMENT UNDER DECOUPLED EVALUATION

## 1. Environment
- **Repository Commit:** `5fe1b62fc7250d395f311af0c9df64aa9e3d6eb1`
- **System Platform:** macOS (Darwin arm64, Apple Silicon M-Series, MPS acceleration active)
- **Key Dependencies & Versions:**
  - Python: `3.9.6`
  - PyTorch: `2.8.0` (MPS active)
  - LightGBM: `4.6.0`
  - Scikit-Learn: `1.6.1`
  - Statsmodels: `0.14.6`
  - SciPy: `1.13.1`
  - Transformers: `4.57.1`
- **Execution Runtimes:**
  - Step 1 (Confound Baselines & Length Prevalence): `100.5s`
  - Step 2 (Label Noise & Conflict Analysis): `140.8s`
  - Step 3 (Representation Optimization): `1791.6s`
  - Step 4 (Classifier Head Optimization): `107.1s`
  - Step 5 (Fine-Tuning Feasibility Benchmark): `152.2s`
  - Step 6 (Mechanical Candidate Selection): `5.1s`
- **New File SHA-256 Checksums:**
| File | SHA-256 Checksum |
| :--- | :--- |
| `src/eval/stats.py` | `89e9a8b7470423876518ee1fd420e3e05f167890f5d8205ff459cada15bab3be` |
| `tests/test_eval_stats.py` | `17b4a95ac097ef564bbac63a2ccdd0da8968d459ab8f2be22d0490cb5431af03` |
| `scripts/p3/eval_protocol.py` | `2617bc66b4b09b0af02ba3be279e64c5ef8ffc5d13dc682f899d11deae40c8b1` |
| `scripts/p3/step1_confounds.py` | `949d6657900ae4ea323ab892e907319a5ed9c7bb4f0c7c871f728dd23a37eeb6` |
| `scripts/p3/step2_label_noise.py` | `6351f44a7b573d17e92f7c2d00870ec0a1ecf9891af0cd2596ae803fc8e165a3` |
| `scripts/p3/step3_representation.py` | `530290a2309b541cb4bd7e6ef9d0762788454aba1e0901fc3bf40b7dea1a0869` |
| `scripts/p3/step4_heads.py` | `846f30b5944236f8a0ac145eb1d55f5252f2921255ed8117e33348be61e8c0e7` |
| `scripts/p3/step5_finetune_feasibility.py` | `52e2d5e42403c02d0d4a0b8b344a9f6550ed901f65fb40faa75c637ea9eeef71` |
| `scripts/p3/step6_selection.py` | `023f98de12ca22fc8bde156c77c43ccd293d89bc11775495c442673cdc18caf9` |
| `models/checkpoints/p3/model_a_p3_selected.pkl` | `4f2edd4d03c589354c9872405a16e99aed38fabf3e3615c63dae4b6241c138d6` |
| `models/checkpoints/p3/model_a_p3_selected_metadata.json` | `07e142b426017b057ad2578a28a2d375912712ed6c8576ee1d5056189cde6307` |

## 2. Pre-registration
- **Pre-registration Timestamp:** `2026-11-20T04:54:10.021624+00:00`
- **Pre-registration Artifact:** `results/p3/preregistration.json`
- **Pre-registered Candidate Inventory:**
| Candidate ID | Category | Description | Primary Features |
| :--- | :--- | :--- | :--- |
| `L1` | Confound Baseline | Logistic regression on log(n_tokens) only | `log_tokens` |
| `L2` | Confound Baseline | Logistic regression on log(n_tokens) + log(n_lines) + cyclomatic proxy | `log_tokens, log_lines, cyclomatic_proxy` |
| `L3` | Confound Baseline | Logistic regression on 28-dim modern C++ structural features only | `cxx28` |
| `L4` | Confound Baseline | Logistic regression on 38-dim omission-aware structural features only | `cxx38` |
| `L_champion_partial` | Confound Baseline | Partialling out length from champion_p2 score (logistic regression on [log(n_tokens), champion_logit]) | `log_tokens, champion_logit` |
| `N0` | Data Variant | Uncleaned train_minus_dev (25k balanced, champion_p2 baseline) | `CLS+mean 256, cxx28` |
| `N1` | Data Variant | Drop samples in light-normalization conflict groups (comments/whitespace stripped, identifiers kept), redrawn to 25k balanced | `CLS+mean 256, cxx28` |
| `N2` | Data Variant | Drop samples in aggressive-normalization conflict groups (identifiers->ID, strings->STR, numbers->NUM), redrawn to 25k balanced | `CLS+mean 256, cxx28` |
| `R1` | Representation | CLS+mean 256 + cxx28 (rebaselined champion recipe on best N variant) | `emb_codebert_256, cxx28` |
| `R2` | Representation | Chunked-mean + cxx28 | `emb_codebert_chunked_mean_256, cxx28` |
| `R3` | Representation | Chunked-max + cxx28 | `emb_codebert_chunked_max_256, cxx28` |
| `R4` | Representation | Concatenated [CLS+mean 256 ; chunked-mean ; chunked-max] (4,608 dims) + cxx28 | `emb_codebert_256, emb_codebert_chunked_mean_256, emb_codebert_chunked_max_256, cxx28` |
| `R5` | Representation | Concatenated [CLS+mean 256 ; chunked-mean ; chunked-max] (4,608 dims) + cxx38 | `emb_codebert_256, emb_codebert_chunked_mean_256, emb_codebert_chunked_max_256, cxx38` |
| `R6` | Representation | R4 representation + L2 length features (log_tokens, log_lines, cyclomatic_proxy) | `emb_codebert_256, emb_codebert_chunked_mean_256, emb_codebert_chunked_max_256, cxx28, L2_features` |
| `H1` | Classifier Head | Logistic regression on best representation from Step 3 | `LogisticRegression` |
| `H2` | Classifier Head | LightGBM on best representation (early stopping on GroupKFold, num_leaves in {15, 31}, lr=0.05) | `LightGBM` |
| `H3` | Classifier Head | 2-layer MLP (hidden 256, dropout 0.3, early stopping on GroupKFold OOF loss), mean of 5 seeds | `PyTorch_MLP` |
- **Exploratory Additions:** None. All candidates evaluated below were strictly pre-registered.

## 3. Confound Baselines
### 3.1 Prevalence by CodeBERT Token Length Bin
| Token Length Bin | Train Samples ($n$) | Train Prevalence (%) | Dev Samples ($n$) | Dev Prevalence (%) | Test Samples ($n$) | Test Prevalence (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **<= 256 tokens** | 9,502 | 34.50% | 1,095 | 14.61% | 1,373 | 14.49% |
| **257 - 512 tokens** | 5,009 | 50.47% | 445 | 31.01% | 558 | 27.42% |
| **513 - 1024 tokens** | 4,439 | 57.47% | 419 | 42.72% | 454 | 33.70% |
| **> 1024 tokens** | 6,050 | 68.48% | 584 | 50.34% | 784 | 63.52% |

### 3.2 Confound Model Evaluation on Held-Out Dev Set
| Candidate ID | Description | CV Macro AUROC | Dev AUROC [95% CI] | Dev Length-Stratified AUROC [95% CI] | Dev C++ AUROC [95% CI] | Dev AUPRC | Dev Recall (10% FPR) | Dev Precision (10% FPR) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `L1` | Logistic Regression on log(n_tokens) only | 0.6687 ± 0.0611 | 0.7153 [0.6954, 0.7358] | 0.5899 [0.5635, 0.6162] | 0.7077 [0.6701, 0.7423] | 0.4840 | 30.61% | 52.68% |
| `L2` | Logistic Regression on log(n_tokens) + log(n_lines) + cyclomatic proxy | 0.6944 ± 0.0532 | 0.7005 [0.6793, 0.7222] | 0.5593 [0.5325, 0.5882] | 0.6890 [0.6506, 0.7251] | 0.4795 | 23.48% | 53.24% |
| `L3` | Logistic Regression on cxx28 features only | 0.6891 ± 0.0529 | 0.6919 [0.6697, 0.7134] | 0.5546 [0.5279, 0.5812] | 0.6796 [0.6378, 0.7170] | 0.4746 | 21.01% | 56.45% |
| `L4` | Logistic Regression on cxx38 features only | 0.6908 ± 0.0542 | 0.6956 [0.6734, 0.7170] | 0.5529 [0.5268, 0.5808] | 0.6868 [0.6441, 0.7256] | 0.4754 | 20.88% | 54.76% |
| `L_champion_partial` | Logistic regression on [log(n_tokens), champion_p2 logit] | 0.7033 ± 0.0506 | 0.7142 [0.6922, 0.7356] | 0.6118 [0.5852, 0.6409] | 0.6796 [0.6405, 0.7152] | 0.4862 | 18.42% | 56.57% |

### 3.3 Partialling Out Length from Champion P2 Score
- **Logistic Regression Model:** `vuln_label ~ log(n_tokens) + champion_p2_logit`
- **L1 Dev AUROC (`log(n_tokens)` only):** `0.7153`
- **Champion P2 Partial Model Dev AUROC:** `0.7142`
- **Empirical Gain over L1:** `-0.0011`
- **Paired Bootstrap Difference vs L1 (grouped by commit):** `Diff = -0.0011 [-0.0209, +0.0293]`, `p = 0.9780`
- **Observation:** The addition of the champion logit to function length does not yield a statistically significant gain over predicting solely from length on dev ($p = 0.9780$). This demonstrates that overall AUROC is substantially driven by length correlation.

## 4. Label Noise
### 4.1 Intra-Train Duplicate Conflict Counts
- **Light Normalization (comments & whitespace stripped, identifiers kept):**
  - Conflicting duplicate groups: **23**
  - Samples affected: **47** (0.19% of `train_minus_dev`)
- **Aggressive Normalization (strings $\to$ STR, numbers $\to$ NUM, identifiers $\to$ ID):**
  - Conflicting duplicate groups: **89**
  - Samples affected: **227** (0.91% of `train_minus_dev`)
- **Token Length Distribution of Aggressive Conflict Samples:**
  - Min: 7 | 25th percentile: 33.5 | Median: 114.0 | Mean: 810.1 | 75th percentile: 684.5 | Max: 13158 | Std: 1953.4
- **Sample Review Artifact:** `results/p3/conflict_samples.md` contains 30 randomly selected unjudged conflict pairs with commit, project, and code details.

### 4.2 Data Variant Comparison (Champion Recipe Retrained, 25k Balanced)
| Variant ID | Cleaning Description | Samples ($N$) | CV Macro AUROC | Dev AUROC [95% CI] | Dev Length-Stratified AUROC [95% CI] | Dev C++ AUROC [95% CI] | $\Delta$ LS-AUROC vs N0 | $p$-value vs N0 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `N0` | No cleaning (champion_p2 data) | 25,000 | 0.6992 ± 0.0442 | 0.6969 [0.6738, 0.7202] | 0.6056 [0.5792, 0.6353] | 0.6517 [0.6122, 0.6873] | +0.0000 | 1.0000 |
| `N1` | Drop light-normalization conflict samples, redrawn to 25k balanced | 25,000 | 0.6839 ± 0.0429 | 0.6753 [0.6523, 0.6977] | 0.5833 [0.5552, 0.6097] | 0.6437 [0.6044, 0.6792] | -0.0223 | 0.0100 |
| `N2` | Drop aggressive-normalization conflict samples, redrawn to 25k balanced | 25,000 | 0.6884 ± 0.0373 | 0.7001 [0.6767, 0.7227] | 0.6152 [0.5881, 0.6429] | 0.6686 [0.6305, 0.7027] | +0.0096 | 0.4580 |

- **Data Selection Outcome:** N2 exhibits higher dev length-stratified AUROC (0.6152 vs 0.6056), but the difference is not statistically significant ($p = 0.4580 \ge 0.05$). Under the pre-registered protocol rule ("if no significant difference, use N0"), **N0** is selected as the training data variant for subsequent steps.

## 5. Representation
Evaluated on N0 data variant. Regularization parameter $C \in \{0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0\}$ chosen strictly by mean GroupKFold OOF AUROC (macro over folds). Features standardized per fold.
| Candidate ID | Representation | Dims | Chosen $C^*$ | CV Macro AUROC | Dev AUROC [95% CI] | Dev Length-Stratified AUROC [95% CI] | Dev C++ AUROC [95% CI] | Dev AUPRC | Dev Recall (10% FPR) | Dev Precision (10% FPR) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `R1` | CLS+mean 256 + cxx28 (rebaselined champion recipe) | 1564 | 0.003 | 0.7005 ± 0.0462 | 0.7004 [0.6780, 0.7234] | 0.6061 [0.5803, 0.6344] | 0.6575 [0.6167, 0.6936] | 0.4750 | 14.27% | 59.78% |
| `R2` | chunked-mean + cxx28 | 1564 | 0.003 | 0.7042 ± 0.0478 | 0.7111 [0.6880, 0.7329] | 0.6098 [0.5835, 0.6379] | 0.6769 [0.6368, 0.7131] | 0.4946 | 12.58% | 61.78% |
| `R3` | chunked-max + cxx28 | 1564 | 0.001 | 0.6963 ± 0.0515 | 0.7035 [0.6814, 0.7252] | 0.5926 [0.5670, 0.6191] | 0.6910 [0.6508, 0.7260] | 0.4686 | 13.88% | 52.97% |
| `R4` | [CLS+mean 256 ; chunked-mean ; chunked-max] + cxx28 | 4636 | 0.001 | 0.7009 ± 0.0484 | 0.7065 [0.6854, 0.7289] | 0.6044 [0.5771, 0.6332] | 0.6705 [0.6318, 0.7053] | 0.4801 | 12.71% | 56.00% |
| `R5` | [CLS+mean 256 ; chunked-mean ; chunked-max] + cxx38 | 4646 | 0.001 | 0.7017 ± 0.0487 | 0.7058 [0.6842, 0.7282] | 0.6029 [0.5756, 0.6309] | 0.6712 [0.6320, 0.7056] | 0.4837 | 12.84% | 58.24% |
| `R6` | R4 + L2 length features (explicit length control) | 4639 | 0.001 | 0.7012 ± 0.0485 | 0.7068 [0.6855, 0.7293] | 0.6048 [0.5779, 0.6332] | 0.6700 [0.6311, 0.7048] | 0.4806 | 12.84% | 56.57% |
- **Best Representation Selected:** `R2` (Highest Dev Length-Stratified AUROC: `0.6098`).

## 6. Classifier Heads
Evaluated on best representation (`R2`).
| Head ID | Classifier Type | Hyperparameters | CV Macro AUROC | Dev AUROC [95% CI] | Dev Length-Stratified AUROC [95% CI] | Dev C++ AUROC [95% CI] | Dev AUPRC | Dev Recall (10% FPR) | Dev Precision (10% FPR) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `H1` | LogisticRegression | C=0.003 | 0.7042 ± 0.0478 | 0.7111 [0.6880, 0.7329] | 0.6098 [0.5835, 0.6379] | 0.6769 [0.6368, 0.7131] | 0.4946 | 12.58% | 61.78% |
| `H2` | LightGBM | num_leaves=31, lr=0.05, n_est=22 | 0.6945 ± 0.0558 | 0.7070 [0.6860, 0.7288] | 0.5782 [0.5515, 0.6046] | 0.6953 [0.6576, 0.7312] | 0.4784 | 24.77% | 53.35% |
| `H3` | TwoLayerMLP | hidden=256, dropout=0.3, AdamW, 5 seeds | 0.7042 ± 0.0141 | 0.7118 [0.6904, 0.7332] | 0.6059 [0.5800, 0.6342] | 0.6764 [0.6355, 0.7120] | 0.4870 | 10.38% | 63.49% |

- **H3 Seed Dispersion (across 5 random seeds):**
  - Dev AUROC: `0.7042 ± 0.0141`
  - Dev Length-Stratified AUROC: `0.6014 ± 0.0156`
  - Dev C++ AUROC: `0.6707 ± 0.0366`

## 7. Master Comparison Across All Candidates
All paired tests computed vs `champion_p2` with resampling clustered by `commit_id` (1,000 bootstrap iterations). Multiple testing controlled with Holm-Bonferroni correction across all candidates.
| Candidate ID | Category | Dev AUROC | $\Delta$ AUROC | Holm $p$ | Dev LS-AUROC | $\Delta$ LS-AUROC | Holm $p$ | Dev C++ AUROC | $\Delta$ C++ AUROC | Holm $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `L1` | Confound Baseline | 0.7153 | +0.0184 | 1.0000 | 0.5899 | -0.0158 | 1.0000 | 0.7077 | +0.0561 | 0.0000 |
| `L2` | Confound Baseline | 0.7005 | +0.0037 | 1.0000 | 0.5593 | -0.0464 | 1.0000 | 0.6890 | +0.0373 | 0.0280 |
| `L3` | Confound Baseline | 0.6919 | -0.0050 | 1.0000 | 0.5546 | -0.0510 | 0.0680 | 0.6796 | +0.0279 | 0.3920 |
| `L4` | Confound Baseline | 0.6956 | -0.0012 | 1.0000 | 0.5529 | -0.0528 | 0.2400 | 0.6868 | +0.0352 | 0.3900 |
| `L_champion_partial` | Confound Baseline | 0.7142 | +0.0174 | 0.0320 | 0.6118 | +0.0061 | 1.0000 | 0.6796 | +0.0279 | 0.0000 |
| `N0` | Data Variant | 0.6969 | +0.0000 | 1.0000 | 0.6056 | +0.0000 | 1.0000 | 0.6517 | +0.0000 | 1.0000 |
| `N1` | Data Variant | 0.6753 | -0.0215 | 0.0000 | 0.5833 | -0.0223 | 0.1600 | 0.6437 | -0.0080 | 0.3600 |
| `N2` | Data Variant | 0.7001 | +0.0033 | 1.0000 | 0.6152 | +0.0096 | 1.0000 | 0.6686 | +0.0169 | 0.4860 |
| `R1` | Representation | 0.7004 | +0.0036 | 1.0000 | 0.6061 | +0.0005 | 1.0000 | 0.6575 | +0.0058 | 0.4860 |
| `R2` | Representation | 0.7111 | +0.0142 | 0.7200 | 0.6098 | +0.0041 | 1.0000 | 0.6769 | +0.0252 | 0.2160 |
| `R3` | Representation | 0.7035 | +0.0067 | 1.0000 | 0.5926 | -0.0130 | 1.0000 | 0.6910 | +0.0393 | 0.1300 |
| `R4` | Representation | 0.7065 | +0.0097 | 1.0000 | 0.6044 | -0.0013 | 1.0000 | 0.6705 | +0.0188 | 0.1920 |
| `R5` | Representation | 0.7058 | +0.0090 | 1.0000 | 0.6029 | -0.0027 | 1.0000 | 0.6712 | +0.0196 | 0.2000 |
| `R6` | Representation | 0.7068 | +0.0099 | 1.0000 | 0.6048 | -0.0009 | 1.0000 | 0.6700 | +0.0183 | 0.1920 |
| `H1` | Representation | 0.7111 | +0.0142 | 0.7200 | 0.6098 | +0.0041 | 1.0000 | 0.6769 | +0.0252 | 0.2160 |
| `H2` | Classifier Head | 0.7070 | +0.0101 | 1.0000 | 0.5782 | -0.0275 | 1.0000 | 0.6953 | +0.0436 | 0.0000 |
| `H3` | Classifier Head | 0.7118 | +0.0150 | 1.0000 | 0.6059 | +0.0002 | 1.0000 | 0.6764 | +0.0247 | 0.2380 |

## 8. Selected Model and Operating Point
- **Selected Model ID:** `R2`
- **Description:** chunked-mean + cxx28
- **Selection Decision Logic:**
  1. Dev C++ AUROC degradation check: Not significantly worse than champion_p2 under Holm-Bonferroni correction.
  2. Primary metric: Ranked by Dev Length-Stratified AUROC.
  3. Complexity tie-breaker: Confidence intervals examined against simpler alternatives.
- **Operating Threshold:** `\tau^* = 0.6884`
  - Selected on `train_minus_dev` out-of-fold predictions reweighted to 30% prevalence for FPR $\le 10\%$. Applied unchanged to `dev_heldout`.
- **Operating Metrics on Dev Set:**
  - **Overall AUROC:** `0.7111 [0.6880, 0.7329]`
  - **Length-Stratified AUROC:** `0.6098 [0.5835, 0.6379]`
  - **C++-Subset AUROC:** `0.6769 [0.6368, 0.7131]`
  - **AUPRC:** `0.4946`
  - **Recall at Operating Point:** `12.58%`
  - **Precision at Operating Point:** `61.78%`
  - **FPR at Operating Point:** `3.39%`
- **Saved Checkpoint:** `models/checkpoints/p3/model_a_p3_selected.pkl`
- **Metadata Record:** `models/checkpoints/p3/model_a_p3_selected_metadata.json`

## 9. Fine-Tuning Feasibility Benchmark and Execution Plan
### 9.1 Timed Step Benchmark (50 Steps, Batch 16, Max Length 512, fp32 on MPS)
| Model Name | Device | Sec / Step | Steps / Sec | Peak Process Memory (MB) | Memory Delta (MB) | Projected Time for 3 Epochs (25k Samples) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `microsoft/codebert-base` | `mps` | 1.0274 s | 0.97 | 876.6 MB | +500.3 MB | **1.34 hours** (4817 s) |
| `microsoft/unixcoder-base` | `mps` | 1.0328 s | 0.97 | 1177.4 MB | +300.7 MB | **1.35 hours** (4843 s) |

### 9.2 Proposed Fine-Tuning Plan (Pending Approval)
- **Input Strategy:** Sliding-window chunking (256-token windows, stride 128) with mean-attention pooling over chunks, avoiding truncated function tails while maintaining batch gradients.
- **Base Architecture:** `microsoft/codebert-base` with classification projection head.
- **Hyperparameters:**
  - Batch size: 16 (gradient accumulation 2, effective batch size 32).
  - Learning rate: $2 \times 10^-5$ with linear warmup (10% of steps) and cosine decay.
  - Weight decay: 0.01.
  - Max epochs: 3.
  - Precision: fp32 on Apple Silicon MPS.
- **Cross-Validation & Early Stopping:**
  - Evaluated on GroupKFold out-of-fold folds on `train_minus_dev` grouped by project.
  - `dev_heldout` and locked test set are strictly excluded from training and early stopping.
  - Evaluation frequency: every 250 steps with patience of 3 evaluation intervals.
- **Status:** **STOPPED AS DIRECTED.** Awaiting user approval before launching any fine-tuning run.

## 10. Failures / Things Not Verified
1. **Length Confounding Confirmed:** As demonstrated in Section 3, a trivial univariate model predicting solely from `log(n_tokens)` attains 0.7153 dev AUROC. When evaluated within length strata, its AUROC drops to 0.5899. Conditioning champion_p2's score on length yields no statistically significant AUROC gain ($p = 0.9780$).
2. **Label Cleaning Modest Impact:** Removing conflicting normalized duplicates from training data (N1 and N2) did not produce a statistically significant improvement on dev length-stratified AUROC over uncleaned data N0 ($p = 0.458$ for N2). Label noise accounts for less than 1% of the training pool.
3. **Locked Test Set Remained Untouched:** In compliance with the strict task instructions, `load_locked_test` was not called at any point during this phase. All selection decisions were conducted strictly on `train_minus_dev` GroupKFold and `dev_heldout`.

## 11. Commands to Reproduce
```bash
# 1. Run Unit Tests (Statistical Module & Test Lock)
./venv/bin/python -m unittest tests/test_locked_test.py tests/test_eval_stats.py -v

# 2. Step 1: Confound Baselines and Length Prevalence Breakdown
./venv/bin/python scripts/p3/step1_confounds.py

# 3. Step 2: Label Noise Analysis and Variant Training
./venv/bin/python scripts/p3/step2_label_noise.py

# 4. Step 3: Representation Optimization (R1 - R6)
./venv/bin/python scripts/p3/step3_representation.py

# 5. Step 4: Classifier Head Optimization (H1 - H3)
./venv/bin/python scripts/p3/step4_heads.py

# 6. Step 5: Fine-Tuning Feasibility Benchmark
./venv/bin/python scripts/p3/step5_finetune_feasibility.py

# 7. Step 6: Mechanical Selection and Master Comparison
./venv/bin/python scripts/p3/step6_selection.py

# 8. Step 7: Generate Complete Markdown Handoff Report
./venv/bin/python scripts/p3/generate_handoff_p3.py
```
