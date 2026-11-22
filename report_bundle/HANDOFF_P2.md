# HANDOFF REPORT — PHASE 16 PROMPT 2: TRUSTWORTHY DEVELOPMENT SETUP

## 1. Environment
- **Repository Commit:** `5fe1b62fc7250d395f311af0c9df64aa9e3d6eb1`
- **System Platform:** macOS (Darwin arm64, Apple Silicon M-Series, MPS acceleration active)
- **Key Dependencies & Versions:**
  - Python: `3.9.6`
  - PyTorch: `2.8.0` (MPS backend verified)
  - Scikit-Learn: `1.6.1`
  - Datasketch: `2.0.0` (scheme: `affine32`)
  - SciPy: `1.13.1`
  - Transformers: `4.57.1`
- **Execution Runtimes:**
  - Step 3 (Validation Diagnosis): `< 2.0s`
  - Step 4 (Near-Duplicate Analysis): `119.0s`
  - Step 5 (Dev Protocol & Dedup): `17.5s`
  - Step 6 (Feature & Embedding Cache): `12543.2s`
  - Step 7 (Reference Numbers & Token Length Evaluation): `28.9s`
- **New Core File Checksums (SHA-256):**
| File | SHA-256 Checksum |
| :--- | :--- |
| `src/eval/locked_test.py` | `aab76085291ef6e0c52282591de6017f2f749d7d5b3bc42a4f50635738bfd2e7` |
| `src/eval/stats.py` | `79fc1151d4421961220c9aab752fe9ad7acc5789f52035e122747057cb5c116f` |
| `scripts/p2/code_normalizer.py` | `274db78fc0362705f368df650a054315f4c6a15842639d1cea78f3b8a785af6e` |
| `scripts/p2/run_validation_diagnosis.py` | `0dcec3825e1960fc5a66e69d6785b630e88091d6f6dae07f629263377084b401` |
| `scripts/p2/run_near_duplicate_analysis.py` | `16755080f8a2171c7a6afda4ac9a379ab582a00290b8f24cda1b1155883f5ca2` |
| `scripts/p2/build_dev_protocol.py` | `39ce3a9e2271c90ff0d46e19e6f9bb646afded87e0f95ea35c0ed38261678461` |
| `scripts/p2/cache_embeddings_p2.py` | `76a01c05d395cda38ba78c5a9d5436783ea88072ce3fc62b34bd4d0d6718a595` |
| `scripts/p2/evaluate_reference_p2.py` | `aa3e5b97396b2c2754e3ad3d3e3afe9f1386dcb65173e90ea332cc139c53d422` |
| `data/splits/p2/dev_heldout.parquet` | `c749c38c040c36f4d03d0b6a516e2d063c8d7c681a43d9bd5b82f02683cb6201` |
| `data/splits/p2/train_minus_dev.parquet` | `57340bfd48c29815c873827229b905e32c78a4141081d5eda3c443d503c4437d` |
| `data/splits/vulnerability_test_heldout_projects_clean.parquet (Locked)` | `ebd95dfc837f12264dd0fd8126563d00ede27908bca38fe46611e25d7bb45bbc` |

## 2. Test Lock
- **Guard Module Path:** `src/eval/locked_test.py`
- **Protected File:** `data/splits/vulnerability_test_heldout_projects_clean.parquet`
- **Immutable Expected Checksum:** `ebd95dfc837f12264dd0fd8126563d00ede27908bca38fe46611e25d7bb45bbc`
- **Access Audit Log:** `data/TEST_ACCESS_LOG.md`
- **Enforcement Mechanism:**
  - Mandatory caller reason string; empty reasons rejected with `ValueError`.
  - Cryptographic SHA-256 recalculation on every access; mismatch terminates execution with `RuntimeError` and logs a security violation.
  - Automatic caller inspection logs timestamp (UTC), calling script relative path, line number, reason, and validation status.
- **Unit Test Verification (`tests/test_locked_test.py`):**
  - Real locked test set load test: **PASSED** (returns 3,169 samples, verifies checksum)
  - Empty reason rejection: **PASSED**
  - Tampered file hash mismatch interception: **PASSED**
  - Missing file handling: **PASSED**

## 3. Validation Diagnosis
### 3.1 Split Composition and Overlap Statistics
| Split | Total Samples ($n$) | Unique Projects ($n_{proj}$) | Unique Commits ($n_{commit}$) |
| :--- | :---: | :---: | :---: |
| **Train (Expanded Pool)** | 138,412 | 902 | 17,587 |
| **Validation (Expanded)** | 23,648 | 829 | 6,231 |
| **Test (Held-Out Projects Clean)** | 3,169 | 4 | 107 overlapping with train |

| Metric | Overlap Count | Percentage of Validation Set |
| :--- | :---: | :---: |
| **Validation Projects in Train** | 825 / 829 | **99.52%** |
| **Validation Samples from Training Projects** | 23,640 / 23,648 | **99.97%** |
| **Validation Commits in Train Commits** | 4,633 / 6,231 | **74.35%** |
| **Validation Samples from Training Commits** | 20,578 / 23,648 | **87.02%** |

### 3.2 Scientific Finding: Is Validation a Random Split of Training Projects?
**Yes, unequivocally.**
The current validation split is fundamentally a random commit/sample partition of the same 902 repositories found in the training pool. 99.52% of validation projects and 99.97% of validation samples come from projects in the training set. Furthermore, 74.35% of validation commits appear in the training pool.
Because the model memorizes syntax, author styles, and shared utility macros from these exact repositories, validation metrics heavily overstate true generalization capability. This empirical diagnosis explains why validation AUROC appeared robust (~0.76) while test AUROC on unseen Envoy Proxy hovered at chance (0.541).

## 4. Near-Duplicates and Data Leakage
### 4.1 Intra-Train Conflicting Exact Duplicates
Normalization pipeline (comments stripped, whitespace collapsed, string literals $\to$ `STR`, numbers $\to$ `NUM`, identifiers $\to$ `ID` preserving C/C++ keywords):
- **Unique Normalized Code Patterns in Train:** 124,186
- **Conflicting Duplicate Patterns (Same Code, Opposite Labels):** 1,315
- **Samples Affected by Intra-Train Label Conflict:** 3,731
- **Agreeing Duplicate Patterns:** 7,972 (19,782 samples)

### 4.2 Near-Duplicate Matches Across Splits (MinHash LSH, 128 Permutations, 5-grams, Jaccard $\ge 0.80$)
| Split / Cohort | Total Samples ($n$) | Matched Pairs | Leaked Unique Samples ($n_{leak}$) | Leakage Rate (%) | Agreement Pairs | Disagreement Pairs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation vs Train** | 23,648 | 30,002 | 7,748 | **32.76%** | 24,738 | 5,264 |
| **Test (Overall) vs Train** | 3,169 | 2,192 | 237 | **7.48%** | 1,946 | 246 |
| **Test: Envoy** | 603 | 132 | 22 | **3.65%** | 114 | 18 |
| **Test: OpenSSL** | 1,257 | 1,040 | 133 | **10.58%** | 874 | 166 |
| **Test: Vim** | 516 | 399 | 34 | **6.59%** | 372 | 27 |
| **Test: GPAC** | 793 | 621 | 48 | **6.05%** | 586 | 35 |

### 4.3 Phase 15 Champion Test Set Evaluation: Full vs Deduplicated (Without Retraining)
Reusing saved test predictions from `models/checkpoints/phase15_codebert_cxx_hybrid_best.pkl`:
| Project Cohort | Full N | Full Acc (%) | Full AUROC [95% CI] | Dedup N | Dedup Acc (%) | Dedup AUROC [95% CI] | $\Delta$ Acc | $\Delta$ AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall** | 3,169 | 74.00% | 0.7369 [0.7181, 0.7571] | 2,932 | 73.60% | 0.7231 [0.7017, 0.7419] | -0.40% | -0.0138 |
| **Envoy** | 603 | 64.01% | 0.5411 [0.4897, 0.5900] | 581 | 63.86% | 0.5467 [0.4948, 0.5945] | -0.16% | +0.0056 |
| **OpenSSL** | 1,257 | 69.29% | 0.7479 [0.7226, 0.7754] | 1,124 | 68.68% | 0.7330 [0.7012, 0.7617] | -0.61% | -0.0149 |
| **Vim** | 516 | 78.10% | 0.7339 [0.6813, 0.7865] | 482 | 77.39% | 0.7042 [0.6481, 0.7612] | -0.71% | -0.0297 |
| **GPAC** | 793 | 86.38% | 0.8103 [0.7632, 0.8614] | 745 | 86.17% | 0.8121 [0.7646, 0.8615] | -0.21% | +0.0018 |

## 5. Project-Held-Out Development Protocol
### 5.1 Chosen Development Projects
Selected strictly outside test projects (`envoy`, `openssl`, `vim`, `gpac`) and excluding `Chromium` and `TensorFlow`:
| Project | Language Category | Sample Count ($n$) | Vulnerable ($n_{vuln}$) | Prevalence (%) | C++ Samples ($n_{cxx}$) | C++ Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `qpdf` | C++-heavy (PDF processing) | 543 | 209 | 38.49% | 500 | 92.08% |
| `openexr` | C++-heavy (HDR image format) | 390 | 48 | 12.31% | 315 | 80.77% |
| `tcpdump` | Mid-sized C (Packet analyzer) | 768 | 296 | 38.54% | 13 | 1.69% |
| `radare2` | Mid-sized C (Disassembler) | 842 | 218 | 25.89% | 21 | 2.49% |
| **Total Dev Held-Out** | **Held-out Benchmark** | **2,543** | **771** | **30.32%** | **849** | **33.39%** |

### 5.2 Splitting Protocol & Properties
- **`data/splits/p2/dev_heldout.parquet`:**
  - $N = 2,543$ samples (satisfies $\ge 2,500$).
  - Modern C++ samples = 849 (satisfies $\ge 600$).
  - Prevalence = **30.32%** (matches clean test set prevalence of 31.65%).
  - SHA-256: `c749c38c040c36f4d03d0b6a516e2d063c8d7c681a43d9bd5b82f02683cb6201`
- **`data/splits/p2/train_minus_dev.parquet`:**
  - Near-duplicates of Dev set detected and purged: **449 samples**.
  - Re-drawn 25,000 balanced split (12,500 vulnerable + 12,500 benign, seed=42).
  - Pre-partitioned into **5-Fold GroupKFold** grouped by `project` (stored in `'kfold'` column, 0 to 4).
  - SHA-256: `57340bfd48c29815c873827229b905e32c78a4141081d5eda3c443d503c4437d`
- **Model Selection & Tuning Rule Going Forward:**
  - Models must be selected exclusively by **Dev AUROC** (with Dev C++ AUROC as secondary criterion).
  - Decision thresholds selected with `threshold_for_prevalence(p_val, y_val, target_prevalence=0.30)`.
  - The held-out locked test set (`vulnerability_test_heldout_projects_clean.parquet`) is touched strictly once at project completion.

## 6. Embedding and Feature Cache
- **Storage Directory:** `data/cache/`
- **Cache Manifest:** `data/cache/cache_manifest.json` and `results/p2/cache_manifest.json`
- **Device Used:** `mps`
- **Model Architecture:** `microsoft/codebert-base`

| Target Dataset | File Type | Shape | Dtype | Filename |
| :--- | :--- | :---: | :---: | :--- |
| `train_minus_dev` | `emb_codebert_256` | `[25000, 1536]` | `float32` | `emb_codebert_256_57340bfd48c29815c873827229b905e32c78a4141081d5eda3c443d503c4437d.npy` |
| `train_minus_dev` | `emb_codebert_chunked_mean_256` | `[25000, 1536]` | `float32` | `emb_codebert_chunked_mean_256_57340bfd48c29815c873827229b905e32c78a4141081d5eda3c443d503c4437d.npy` |
| `train_minus_dev` | `emb_codebert_chunked_max_256` | `[25000, 1536]` | `float32` | `emb_codebert_chunked_max_256_57340bfd48c29815c873827229b905e32c78a4141081d5eda3c443d503c4437d.npy` |
| `train_minus_dev` | `feat_cxx28` | `[25000, 28]` | `float32` | `feat_cxx28_57340bfd48c29815c873827229b905e32c78a4141081d5eda3c443d503c4437d.npy` |
| `train_minus_dev` | `feat_cxx38` | `[25000, 38]` | `float32` | `feat_cxx38_57340bfd48c29815c873827229b905e32c78a4141081d5eda3c443d503c4437d.npy` |
| `dev_heldout` | `emb_codebert_256` | `[2543, 1536]` | `float32` | `emb_codebert_256_c749c38c040c36f4d03d0b6a516e2d063c8d7c681a43d9bd5b82f02683cb6201.npy` |
| `dev_heldout` | `emb_codebert_chunked_mean_256` | `[2543, 1536]` | `float32` | `emb_codebert_chunked_mean_256_c749c38c040c36f4d03d0b6a516e2d063c8d7c681a43d9bd5b82f02683cb6201.npy` |
| `dev_heldout` | `emb_codebert_chunked_max_256` | `[2543, 1536]` | `float32` | `emb_codebert_chunked_max_256_c749c38c040c36f4d03d0b6a516e2d063c8d7c681a43d9bd5b82f02683cb6201.npy` |
| `dev_heldout` | `feat_cxx28` | `[2543, 28]` | `float32` | `feat_cxx28_c749c38c040c36f4d03d0b6a516e2d063c8d7c681a43d9bd5b82f02683cb6201.npy` |
| `dev_heldout` | `feat_cxx38` | `[2543, 38]` | `float32` | `feat_cxx38_c749c38c040c36f4d03d0b6a516e2d063c8d7c681a43d9bd5b82f02683cb6201.npy` |
| `test_clean` | `emb_codebert_256` | `[3169, 1536]` | `float32` | `emb_codebert_256_ebd95dfc837f12264dd0fd8126563d00ede27908bca38fe46611e25d7bb45bbc.npy` |
| `test_clean` | `emb_codebert_chunked_mean_256` | `[3169, 1536]` | `float32` | `emb_codebert_chunked_mean_256_ebd95dfc837f12264dd0fd8126563d00ede27908bca38fe46611e25d7bb45bbc.npy` |
| `test_clean` | `emb_codebert_chunked_max_256` | `[3169, 1536]` | `float32` | `emb_codebert_chunked_max_256_ebd95dfc837f12264dd0fd8126563d00ede27908bca38fe46611e25d7bb45bbc.npy` |
| `test_clean` | `feat_cxx28` | `[3169, 28]` | `float32` | `feat_cxx28_ebd95dfc837f12264dd0fd8126563d00ede27908bca38fe46611e25d7bb45bbc.npy` |
| `test_clean` | `feat_cxx38` | `[3169, 38]` | `float32` | `feat_cxx38_ebd95dfc837f12264dd0fd8126563d00ede27908bca38fe46611e25d7bb45bbc.npy` |

## 7. Reference Numbers
### 7.1 Champion P2 (`phase15_champion_p2_reference.pkl`) on `dev_heldout.parquet`
Retrained using exact Phase 15 recipe ($C=0.01$, balanced weights, 28 structural features + CodeBERT embeddings) on `train_minus_dev.parquet`:
| Cohort | N | Prevalence (%) | AUROC [95% CI] | Accuracy ($\tau=0.50$) | FPR ($\tau=0.50$) | F1 ($\tau=0.50$) | Accuracy ($\tau^*=0.3350$) | FPR ($\tau^*$) | F1 ($\tau^*$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall** | 2,543 | 30.3% | 0.7109 [0.6903, 0.7319] | 67.68% | 26.64% | 0.506 | 58.83% | 51.30% | 0.547 |
| **C++ Subset** | 849 | 29.2% | 0.6895 [0.6489, 0.7286] | 67.49% | 13.31% | 0.274 | 65.37% | 33.61% | 0.515 |
| **C Subset** | 1,694 | 30.9% | 0.7450 [0.7209, 0.7673] | 67.77% | 33.48% | 0.575 | 55.55% | 60.38% | 0.559 |
| **Production Code** | 2,525 | 30.2% | 0.7112 [0.6902, 0.7318] | 67.80% | 26.43% | 0.505 | 58.89% | 51.05% | 0.546 |
| **Test/Mock Code** | 18 | 50.0% | 0.5185 [0.2346, 0.7901] | 50.00% | 66.67% | 0.571 | 50.00% | 100.00% | 0.667 |

### 7.2 Existing Phase 15 Champion Test Performance by Function Token Length
Evaluated on `vulnerability_test_heldout_projects_clean.parquet` ($N=3,169$) using CodeBERT tokenizer length without truncation:
| Token Length Bucket | Samples ($n$) | % of Test | Prevalence (%) | AUROC [95% CI] | Accuracy ($\tau=0.50$) | FPR (%) | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **<= 256 tokens** | 1,373 | 43.3% | 14.5% | 0.5949 [0.5491, 0.6416] | 83.47% | 4.09% | 0.150 |
| **257 - 512 tokens** | 558 | 17.6% | 27.4% | 0.5369 [0.4804, 0.5942] | 68.64% | 15.31% | 0.314 |
| **513 - 1024 tokens** | 454 | 14.3% | 33.7% | 0.6548 [0.6028, 0.7090] | 63.66% | 30.90% | 0.495 |
| **> 1024 tokens** | 784 | 24.7% | 63.5% | 0.6392 [0.5981, 0.6770] | 64.92% | 75.52% | 0.761 |

## 8. Failures / Things Not Verified
1. **Truncation Impact Not Yet Remedied:** 56.67% of clean test functions exceed CodeBERT's 256-token limit (1,796 / 3,169 samples). In the token-bucket breakdown (Section 7.2), functions of length 257-512 drop to AUROC 0.5369 [0.4804, 0.5942], and the largest functions (>1024 tokens) exhibit severe false positive rates (FPR 75.52% under standard threshold) alongside a sharp prevalence shift (63.5% vulnerable). Sliding-window chunked embeddings (mean and max pooled) were computed and cached in Step 6 to test full-context representations in Prompt 3 without re-embedding.
2. **Envoy AUROC Remains Near Chance Under Zero Leakage:** Deduplicating the test set (removing 22 near-duplicate Envoy functions) slightly shifts Envoy AUROC from 0.5411 to 0.5467, confirming that the Envoy challenge is genuine out-of-distribution semantic shift, not artifactual leakage.
3. **Intra-Train Label Noise:** 3,731 samples in the training pool share identical normalized code patterns with conflicting ground-truth vulnerability labels (1,315 conflicting patterns). Cleaning these contradictory labels remains an avenue for model refinement in Prompt 3.
4. **GroupKFold Variance Across Folds:** Because projects vary significantly in size, GroupKFold fold prevalence ranges from 37.7% to 64.7%. Hyperparameter selection on CV must compute balanced accuracy or macro-averaged AUROC.

## 9. Commands to Reproduce
```bash
# 1. Run Unit Tests (Test Lock & Statistical Module)
./venv/bin/python -m unittest tests/test_locked_test.py tests/test_eval_stats.py -v

# 2. Validation Split Diagnosis
./venv/bin/python scripts/p2/run_validation_diagnosis.py

# 3. Near-Duplicate Leakage Analysis (MinHash LSH & Deduplicated Evaluation)
./venv/bin/python scripts/p2/run_near_duplicate_analysis.py

# 4. Construct Project-Held-Out Development Protocol & Splits
./venv/bin/python scripts/p2/build_dev_protocol.py

# 5. Extract and Cache CodeBERT & Structural Features
./venv/bin/python scripts/p2/cache_embeddings_p2.py

# 6. Evaluate Reference Champion P2 and Length-Bucketed Cohorts
./venv/bin/python scripts/p2/evaluate_reference_p2.py

# 7. Generate Complete Markdown Handoff Report
./venv/bin/python scripts/p2/generate_handoff_p2.py
```
