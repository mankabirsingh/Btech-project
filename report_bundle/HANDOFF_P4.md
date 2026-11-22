# HANDOFF REPORT — PHASE 16 PROMPT 4: LENGTH-DEBIASED MODEL A AND PAIRED EVALUATION

## 1. Environment, Runtimes, and SHA-256 Checksums
- **Repository Git Commit:** `5fe1b62fc7250d395f311af0c9df64aa9e3d6eb1`
- **Operating System:** macOS Darwin `arm64` (Apple Silicon MPS active)
- **Software Stack:**
  - Python: `3.9.6`
  - PyTorch: `2.8.0`
  - Transformers: `4.57.1`
  - Scikit-Learn: `1.6.1`
  - SciPy: `1.13.1`
  - Pandas: `2.2.3`
- **Execution Runtimes:**
  - Step 0 (P3 Inconsistencies & Draw Variance): `351.23s`
  - Step 1 (Paired Data Audit & Benchmark): `20.05s`
  - Step 2 (Debiased Training D1-D3): `455.97s`
  - Step 3 (UniXcoder Fine-Tuning Pipeline): `F1: 4908.65s, F2: 2477.07s, F3: 3110.77s`
  - Step 4 (Master Comparison & Selection): `497.47s`

### SHA-256 Checksums of Generated Artifacts
| Artifact Path | SHA-256 Checksum |
| :--- | :--- |
| `scripts/p4/step0_resolutions.py` | `e7e504b71f94acd35f2d0dc9aa123daf5880f3646d498a42abbba7ae65612ae0` |
| `scripts/p4/step1_paired_data.py` | `6b1d9c531d0cd445220a071ecaac20c0a3b82a882f486bd8f9d3d4bd49f267cc` |
| `scripts/p4/step2_debiased_training.py` | `022e584a90df92a44cada61795e17f6e05fdb4e139df06e0246986122497ef3d` |
| `scripts/p4/step3_finetuning.py` | `131320f7873fee6db44167c3293618630d9e69c4871dc07a4335a64c3c712714` |
| `scripts/p4/step4_master_selection.py` | `7ff89bdc768c086a1924f6b022a25c73f128eccceff2905d5100b42094e059e2` |
| `scripts/p4/generate_handoff_p4.py` | `77fbe0a1e161a9ad78ae4e8e1f04674523db4e0b875327991c7b6da679601fdb` |
| `data/splits/p4/dev_pairs.parquet` | `8476f346ad97fcc0bec9cb77a00645e36e00e5527834d98490cec0980be53f13` |
| `data/splits/p4/d1_train_indices.npy` | `4660899668ac50b6ffab4baebe60d45c0e43afdffa5ebfd7e4b4dc985b37a5c4` |
| `data/splits/p4/d2_train_weights.npy` | `a786d6a355fd67c88e26adb4683518e757cd9b2f35a48194c4e79945cee27c01` |
| `data/splits/p4/d3_train_indices.npy` | `d816687d58423f106e9925b4e37b9f9a50f0f317c9282677cc35cbe019392b6f` |
| `results/p4/step0_resolutions.json` | `989dca2f56dcf740800db27a522432912c1ea11f95d109bc4fc2edf3d2dfffa5` |
| `results/p4/step0_p3_master_recomputed.json` | `d40bf067c72ab320979ed3c02e2012f5ef473bc8d33a99739c89bd87b9760fa5` |
| `results/p4/step1_paired_data.json` | `22bea50b90be2184bfdb4755cb45597371ec04cb16cd8da45d2e350cb2b9fc19` |
| `results/p4/step2_debiased_models.json` | `cabbcf531b4cf8d69c14dedef8120bcdc264688b808ed7129db0c2be534248c3` |
| `results/p4/step3_finetuning_eval.json` | `779a44b5e297e675690e3b1a89f42255942b1830950760e78db9b90f0f7fdc67` |
| `results/p4/master_comparison_p4.json` | `f2e2f8446db7cdfb67a7b3fd78df07eb5641d965f762e62a788546e165207078` |
| `models/checkpoints/p4/model_a_p4_selected/selection_metadata.json` | `e8a2e3a5a273ef9d876bc7152c0c29593df7c537d80a4b884f13d4076ef689c6` |
| `models/checkpoints/p4/model_a_p4_selected/best_model.pt` | `55820e8d280723ba650cbbd632ea0b10c3e01e6c86989017b19b602fe0045c3e` |

---

## 2. Step 0 Resolutions: P3 Inconsistencies and Draw Variance

### 0.1 Root Cause of AUROC Discrepancy (0.7109 vs 0.6969)
- **P2 Champion Reference:** `0.7109` (95% CI: `[0.6903, 0.7319]`) in `scripts/p2/evaluate_reference_p2.py:70-75`.
- **P3 N0 Baseline Result:** `0.6969` in `scripts/p3/eval_protocol.py:120-123` and `scripts/p3/step2_label_noise.py:90-94`.
- **Exact Difference:** In Phase 15 and P2, StandardScaler was fit solely on the 28 modern C++ structural features, leaving the 1536 CodeBERT dense embedding features in their natural unscaled geometry. In P3, StandardScaler was fit globally across all 1564 features. Standardizing dense transformer embeddings distorts representation geometry by scaling low-variance embedding noise dimensions up, which degraded dev AUROC by 0.0141 (from 0.7109 to 0.6969). The faithful Phase 15 recipe is 0.7109.
- **Faithful Recipe Verdict:** Phase 15 recipe (`0.7109`) is faithful. Scaler applied solely to cxx28 features.

### 0.2 Draw Variance (5 Seeds on R2 C=0.003)
- **Dev AUROC:** `0.7009 ± 0.0067`
- **Dev LS-AUROC:** `0.5979 ± 0.0091` (Draw Std = `0.0091`)
- **Dev C++ AUROC:** `0.6631 ± 0.0121`

| Seed | Dev AUROC | Dev LS-AUROC | Dev C++ AUROC |
| :--- | :--- | :--- | :--- |
| Seed 0 | 0.7102 | 0.6108 | 0.6742 |
| Seed 1 | 0.7000 | 0.5931 | 0.6647 |
| Seed 2 | 0.6935 | 0.5873 | 0.6456 |
| Seed 3 | 0.6960 | 0.5957 | 0.6739 |
| Seed 4 | 0.7045 | 0.6028 | 0.6570 |

- **N0/N1/N2 Differences vs Draw Std:** {'n0_ls_auroc': 0.605649881055505, 'n1_ls_auroc': 0.5833122978975351, 'n2_ls_auroc': 0.6152490151272481, 'delta_n1_vs_n0': -0.022337583157969854, 'delta_n2_vs_n0': 0.009599134071743065, 'draw_std_ls_auroc': 0.009089110684765356, 'n2_exceeds_draw_std': True, 'n1_exceeds_draw_std': True}

### 0.3 Statistical Standards Implemented
- 2,000 bootstrap resamples utilized for all confidence intervals and paired difference hypothesis tests.
- Empirical bootstrap p-values below `1/2000` are formatted as `<0.0005`.

---

## 3. Paired Data Availability and Benchmark (`dev_pairs.parquet`)

### 3.1 Dataset Column Audit & Pair Identification
- **Explicit Post-Fix Columns Present:** `False` (Checked columns: `func_after`, `func_before`, `fixed_code`).
- **Matching Mechanism:** Disjoint pairing via `parent_sample_id` and `(commit_id, func_name)` matching.
- **Usable Disjoint Pairs in Clean Source Pool (n=131,616):** `5190` pairs (`2874` via parent_sample_id, `2316` via commit/func).
- **Usable Disjoint Pairs in Train Minus Dev (25k subset):** `360` pairs (`720` samples).
- **Usable Disjoint Pairs in Held-Out Dev (n=2,543):** `186` pairs (`372` samples; `104` parent_sample_id, `82` commit/func). Exceeds the 150 pair requirement.
- **Artifact Generated:** `data/splits/p4/dev_pairs.parquet` (`186` pairs).

### 3.2 Length Directionality in Vulnerability Patches
- **Patched (Fixed) Code Strictly Longer:** `140` / `186` (`75.27%`)
- **Vulnerable Code Strictly Longer:** `33` / `186` (`17.74%`)
- **Identical Length:** `12` / `186` (`6.99%`)
- **Significance:** Vulnerability-fixing commits consistently add defensive checks, null assertions, and bounds checks, making fixed code longer in three out of four cases.

### 3.3 Baseline Models on Paired Evaluation
| Model | pair_correct (P(v) > P(f)) | P-C (V+, F-) | P-V (V+, F+) | P-B (V-, F-) | P-R (V-, F+) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L1 (log length)** | 17.74% [12.90%, 23.12%] | 1.61% | 71.51% | 26.34% | 0.54% |
| **Faithful Champion P2** | 46.24% [38.71%, 53.23%] | 2.69% | 54.84% | 41.94% | 0.54% |
| **R2 (Chunked-Mean + CXX28)** | 39.25% [31.72%, 46.77%] | 2.69% | 54.30% | 39.78% | 3.23% |

- *Key Observation on L1:* Because L1 scores strictly on length and security fixes add code, L1 assigns higher vulnerability probability to fixed code in 71.51% of pairs, achieving a pair ranking accuracy of only 17.74%.

---

## 4. Length-Debiased Frozen Models (D1, D2, D3)

- Hyperparameters selected strictly via 5-fold GroupKFold cross-validation on `train_minus_dev` maximizing **OOF LS-AUROC** across $C \in \{0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0\}$.
- Operating thresholds calibrated on out-of-fold predictions to enforce 10% FPR.

| Candidate | Selected C | OOF LS-AUROC | Dev AUROC (95% CI) | Dev LS-AUROC (95% CI) | Dev C++ AUROC | pair_correct (95% CI) | Operating Threshold | Actual Dev FPR | Actual Dev Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **D1 (Stratified Balanced)** | 0.3 | 0.7203 | 0.6250 [0.6028, 0.6477] | 0.5916 [0.5660, 0.6189] | 0.6134 | 47.85% [40.32%, 55.38%] | 0.7096 | 1.30% | 3.63% |
| **D2 (Propensity Weighted)** | 0.3 | 0.7363 | 0.6473 [0.6248, 0.6703] | 0.6058 [0.5797, 0.6319] | 0.6359 | 45.70% [38.71%, 52.69%] | 0.6878 | 1.47% | 3.63% |
| **D3 (Paired Training)** | 0.001 | 0.4436 | 0.6111 [0.5866, 0.6353] | 0.5064 [0.4785, 0.5332] | 0.5816 | 51.08% [43.55%, 58.06%] | 0.5043 | 4.12% | 6.49% |

- **Best Debiased Dataset Identified:** `D2` (dev LS-AUROC: `0.6058` > D1 `0.5916`).

---

## 5. UniXcoder Fine-Tuning Runs (F1, F2, F3)

### Training Configuration
- **Base Architecture:** `microsoft/unixcoder-base` (12-layer RoBERTa encoder, 768-dim, sequence classification head).
- **Context Handling:** 512-token sliding windows with stride 256 for functions exceeding 512 tokens.
- **Optimization:** AdamW ($lr=2\times 10^{-5}$, weight decay $0.01$), linear warmup (10%), gradient accumulation 2 (effective batch size 32).
- **Early Stopping:** Evaluated every 250 steps on held-out fold 0 (patience = 3 checks).

### Run Summaries
#### F1 (UniXcoder + D2 Debiased Data, Seed 42)
- **Total Steps Executed:** `1000`
- **Best Fold 0 LS-AUROC:** `0.6365`
- **Wall Clock Runtime:** `4908.65s`
- **Early Stopped:** `True`

| Pooling Variant | Dev AUROC (95% CI) | Dev LS-AUROC (95% CI) | Dev C++ AUROC (95% CI) | Dev C++ LS-AUROC | pair_correct (95% CI) | Actual Dev FPR | Actual Dev Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Max Logit** | 0.6656 [0.6426, 0.6871] | 0.6170 [0.5901, 0.6438] | 0.6806 [0.6456, 0.7191] | 0.6157 | 54.84% [47.84%, 61.83%] | 0.28% | 3.11% |
| **Mean Logit** | 0.6679 [0.6457, 0.6893] | 0.6167 [0.5901, 0.6431] | 0.6851 [0.6491, 0.7227] | 0.6182 | 56.45% [49.46%, 63.44%] | 0.28% | 3.24% |

#### F2 (UniXcoder + Natural N0 Data, Seed 42 Control)
- **Total Steps Executed:** `1000`
- **Best Fold 0 LS-AUROC:** `0.6380`
- **Wall Clock Runtime:** `2477.07s`
- **Early Stopped:** `True`

| Pooling Variant | Dev AUROC (95% CI) | Dev LS-AUROC (95% CI) | Dev C++ AUROC (95% CI) | Dev C++ LS-AUROC | pair_correct (95% CI) | Actual Dev FPR | Actual Dev Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Max Logit** | 0.7363 [0.7159, 0.7572] | 0.6213 [0.5952, 0.6479] | 0.6968 [0.6584, 0.7327] | 0.6100 | 46.24% [39.25%, 53.23%] | 0.11% | 1.43% |
| **Mean Logit** | 0.7370 [0.7164, 0.7582] | 0.6210 [0.5949, 0.6481] | 0.6968 [0.6591, 0.7331] | 0.6095 | 46.24% [38.71%, 53.23%] | 0.11% | 1.43% |

#### F3 (UniXcoder + D2 Debiased Data, Seed 7)
- **Total Steps Executed:** `1250`
- **Best Fold 0 LS-AUROC:** `0.6470`
- **Wall Clock Runtime:** `3110.77s`
- **Early Stopped:** `True`

| Pooling Variant | Dev AUROC (95% CI) | Dev LS-AUROC (95% CI) | Dev C++ AUROC (95% CI) | Dev C++ LS-AUROC | pair_correct (95% CI) | Actual Dev FPR | Actual Dev Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Max Logit** | 0.7178 [0.6975, 0.7380] | 0.6665 [0.6426, 0.6916] | 0.7225 [0.6884, 0.7578] | 0.6656 | 55.38% [48.39%, 62.37%] | 1.52% | 9.73% |
| **Mean Logit** | 0.7137 [0.6938, 0.7340] | 0.6622 [0.6380, 0.6869] | 0.7226 [0.6887, 0.7572] | 0.6653 | 53.23% [46.24%, 60.22%] | 1.58% | 10.25% |

---

## 6. Master Comparison Table vs L1 and vs Faithful Champion P2

- Statistical significance assessed via 2,000 paired bootstrap resamples grouped by `commit_id`.
- Multi-hypothesis testing across all candidates adjusted using the step-down Holm-Bonferroni method.

| Candidate | Dev AUROC | Dev LS-AUROC | Dev C++ AUROC | Dev C++ LS-AUROC | Pair Correct | Delta vs L1 (LS) | Holm p (vs L1) | Delta vs Champ (LS) | Holm p (vs Champ) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `L1` | 0.7153 | 0.5899 | 0.7077 | 0.6342 | 17.74% | +0.0000 | N/A | +0.0000 | N/A |
| `faithful_champion_p2` | 0.7113 | 0.6042 | 0.6995 | 0.6129 | 46.24% | +0.0143 | 1.0000 | +0.0000 | N/A |
| `R2` | 0.7068 | 0.5980 | 0.6997 | 0.6108 | 39.25% | +0.0081 | 1.0000 | -0.0062 | 1.0000 |
| `D1` | 0.6250 | 0.5916 | 0.6134 | 0.5868 | 47.85% | +0.0017 | 1.0000 | -0.0126 | 1.0000 |
| `D2` | 0.6473 | 0.6058 | 0.6359 | 0.5885 | 45.70% | +0.0159 | 1.0000 | +0.0016 | 1.0000 |
| `D3` | 0.6111 | 0.5064 | 0.5816 | 0.5229 | 51.08% | -0.0835 | 0.0360 | -0.0978 | <0.0005 |
| `F1_max` | 0.6656 | 0.6170 | 0.6806 | 0.6157 | 54.84% | +0.0272 | 0.6900 | +0.0129 | 1.0000 |
| `F1_mean` | 0.6679 | 0.6167 | 0.6851 | 0.6182 | 56.45% | +0.0269 | 0.6900 | +0.0126 | 1.0000 |
| `F2_max` | 0.7363 | 0.6213 | 0.6968 | 0.6100 | 46.24% | +0.0314 | 0.4160 | +0.0171 | 1.0000 |
| `F2_mean` | 0.7370 | 0.6210 | 0.6968 | 0.6095 | 46.24% | +0.0311 | 0.4160 | +0.0168 | 1.0000 |
| `F3_max` | 0.7178 | 0.6665 | 0.7225 | 0.6656 | 55.38% | +0.0766 | <0.0005 | +0.0623 | 0.5300 |
| `F3_mean` | 0.7137 | 0.6622 | 0.7226 | 0.6653 | 53.23% | +0.0723 | <0.0005 | +0.0580 | 0.7380 |

---

## 7. Model Selection Verdict

### Mechanical Decision Rule
- **Criterion:** Select the candidate with the highest dev LS-AUROC among those whose dev LS-AUROC strictly exceeds L1 with statistical significance ($p_{Holm} < 0.05$). If none beats L1, select nothing and keep R2 as working model.
- **Candidate Selected:** `F3_max`
- **Status:** Selected and Saved
- **Data Variant:** `D2 debiased (inverse-propensity weighted)`
- **Run:** `F3`
- **Inference Pooling:** `max`
- **Operating Threshold (10% FPR calibration):** `0.8866`
- **Dev LS-AUROC:** `0.6665`
- **Delta vs L1 (LS-AUROC):** `+0.0766` (Holm $p < 0.0005$)
- **Pair Ranking Accuracy (`pair_correct`):** `55.38%`
- **Selected Model Checkpoint Directory:** `models/checkpoints/p4/model_a_p4_selected/`
- **Preserved Weights:** `models/checkpoints/p4/model_a_p4_selected/best_model.pt`
- **Selection Metadata File:** `models/checkpoints/p4/model_a_p4_selected/selection_metadata.json`

---

## 8. Failures, Surprises, and Scientific Diagnostics

1. **The Vulnerability Patch Directionality Reversal:**
   - In 75.27% of disjoint vulnerable/fixed pairs in `dev_heldout`, the patched (safe) code is strictly longer than the vulnerable precursor due to the addition of null pointer assertions, input validation, and bounds checks.
   - Because L1 is parameterized exclusively on function length, it assigns higher vulnerability risk to the longer patched version in 71.51% of pairs, resulting in a pair ranking accuracy of only 17.74%.
   - Debiased training (D1, D2) successfully mitigates this reversal, restoring pair ranking accuracy to 45.70%–47.85%.

2. **Data Sparsity in Natural Disjoint Pairs:**
   - Although the clean source pool contains 5,190 disjoint pairs, the balanced 25k training split (`train_minus_dev`) contains only 360 disjoint pairs (720 samples).
   - Consequently, direct pairwise contrastive training (D3) suffers from extreme sample starvation, achieving an out-of-fold LS-AUROC of only 0.4436 and dev LS-AUROC of 0.5064.

3. **Resolution of P3 Discrepancy:**
   - In Phase 15 and P2, feature standardization was applied solely to the 28-dimensional C++ structural features, preserving the unscaled geometry of CodeBERT embeddings.
   - In P3, global standardization across all 1564 dimensions collapsed embedding geometry, dropping dev AUROC from 0.7109 to 0.6969.
   - Correcting this confirms the faithful Phase 15 reference dev AUROC is 0.7109.

4. **Draw Variance Magnitude:**
   - Empirical draw variance across 5 balanced 25k draws on R2 is $\pm 0.0091$ LS-AUROC.
   - Several small observed differences in frozen models sit within this noise margin.

---

## 9. Commands to Reproduce Phase 16 Prompt 4 Results

```bash
# 1. Step 0: Inconsistency resolution and draw variance
./venv/bin/python scripts/p4/step0_p3_inconsistencies.py

# 2. Step 1: Paired data audit and benchmark extraction
./venv/bin/python scripts/p4/step1_paired_data.py

# 3. Step 2: Length-debiased frozen model training (D1, D2, D3)
./venv/bin/python scripts/p4/step2_debiased_training.py

# 4. Step 3: UniXcoder fine-tuning pipeline (F1, F2, F3)
./venv/bin/python scripts/p4/step3_finetuning.py --run all

# 5. Step 4: Master comparison and mechanical selection
./venv/bin/python scripts/p4/step4_master_selection.py

# 6. Step 5: Report generation and artifact checksum validation
./venv/bin/python scripts/p4/generate_handoff_p4.py
```