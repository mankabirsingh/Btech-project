# Baseline Audit & Expansion Environment Verification (Phase 1)

**Project Title:** Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code  
**Institution:** Delhi Technological University (DTU) — Department of Information Technology  
**Researchers:** Mankabir Singh (23/IT/199) & Mohnish Gautam (23/IT/200)  
**Role:** Senior Machine Learning & Cybersecurity Research Scientist  
**Date:** October 27, 2026  
**Status:** Reproducibly Audited from Raw Checkpoints and Physical System State  

---

## 1. Environment & Hardware Inventory

| Resource Property | Physical Specification | Verification Method |
| :--- | :--- | :--- |
| **Operating System** | macOS 15.6 (Darwin Kernel 25.6.0 arm64) | `uname -a` |
| **CPU Architecture** | Apple Silicon M1 Max (10 physical cores) | `sysctl -n hw.ncpu` |
| **System Memory (RAM)**| 32.00 GB Unified Memory (34,359,738,368 bytes) | `sysctl -n hw.memsize` |
| **Disk Space Available**| 845 GiB Available on `/System/Volumes/Data` | `df -h .` |
| **PyTorch Accelerator** | Apple Metal Performance Shaders (`torch.backends.mps`) | Verified `torch.backends.mps.is_available() == True` |
| **CUDA Acceleration** | Not Applicable (macOS environment) | `torch.cuda.is_available() == False` |
| **Core Libraries** | `transformers 4.57.6`, `datasets 4.5.0`, `sklearn 1.6.1`, `torch 2.8.0` | Verified via virtualenv runtime inspection |
| **Estimated Compute Cost** | **$0.00** (Full local execution on DTU developer workstation) | Local MPS GPU & 10-core parallel worker pools |

---

## 2. Classification Task Definition & Label Conventions

- **Exact Classification Task:** Binary Function-Level Vulnerability Detection.
  Given source code $C$ representing a single C/C++ function, predict the probability $P(Y = 1 \mid C)$ that the function contains a known security flaw or vulnerable pattern.
- **Language Scope:** C and C++ source code functions (`language = 'C/C++'`).
- **Label Semantics:**
  - `vuln_label = 1` (**Positive Class**): Vulnerable function (pre-patch version, vulnerable commit, or known vulnerability signature).
  - `vuln_label = 0` (**Negative Class**): Secure or benign function (post-patch fixed version or verified safe commit).
- **Positive-Class Convention:** All detection metrics (Precision, Recall, F1, PR-AUC / Average Precision) evaluate the model's ability to identify the **vulnerable functions** (`vuln_label = 1`).

---

## 3. Existing Datasets, Preprocessing, and Split Properties

### 3.1 Datasets Currently Integrated:
1. **Devign (`DetectVul/devign`):** 27,318 original C/C++ functions collected from large C codebases (QEMU, FFmpeg, Wireshark). Filtered for syntax cleanliness and deduplicated.
2. **PrimeVul (`adefu020/primevul`):** 229,484 rows from ~6,000 open-source repositories; filtered to C/C++ paired commit functions (`vulnerable_function` vs. `fixed_function`). Grouped by `source_record_id` (`pv_pair_{id}`).
3. **HumanVsAICode (`OSS-forge/HumanVsAICode`):** Paired multi-model code implementations (ChatGPT, DeepSeek-Coder, Qwen vs. Human) across Python and Java. Contains zero vulnerability labels; utilized exclusively for provenance feature extraction and controlled editing benchmarks.

### 3.2 Existing Split Properties (`data/splits/`):
- **Total Samples:** 36,080 processed C/C++ functions ([`data/processed/vulnerability_data.parquet`](file:///Users/mankabir/Desktop/Btech%20project/data/processed/vulnerability_data.parquet)).
- **Train Set:** 27,060 samples (12,586 Vulnerable [46.51%], 14,474 Benign [53.49%]), 12,376 commit groups.
- **Validation Set:** 3,608 samples (1,620 Vulnerable [44.90%], 1,988 Benign [55.10%]), 1,781 commit groups.
- **Held-Out Test Set:** 5,412 samples (2,634 Vulnerable [48.67%], 2,778 Benign [51.33%]), 2,552 commit groups.

### 3.3 Leakage & Partition Audit:
- **Balance:** Test set is balanced (48.67% vulnerable vs. 51.33% benign).
- **Commit Separation:** Verified **0 group overlap** across train, validation, and test sets. Pre-patch and post-patch functions sharing a commit ID or pair ID are strictly co-located within the same partition.
- **Project Separation:** **Not project-separated.** QEMU and FFmpeg appear across train, validation, and test sets because splits were grouped on commit hashes rather than repository boundaries.
- **Deduplication:** Exact duplicate code strings were stripped across the entire corpus. However, near-duplicate commit pairs (differing by only 1–2 lines of patch code) exist within the partitions.

---

## 4. Independent Reproduction of Baseline Metrics

Directly evaluated from saved predictions in [`results/experiments/test_predictions_all_models.parquet`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/test_predictions_all_models.parquet) on the identical 5,412 test examples:

| Model | Acc | Prec | Rec | F1 | MCC | AUROC | AP (PR-AUC) | Brier Score | Positive Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Word N-gram Baseline** | 50.72% | 49.43% | 54.40% | 51.80% | +0.0164 | 0.5193 | 0.5021 | 0.2638 | 53.57% |
| **Char N-gram Baseline** | 52.75% | 51.31% | **57.18%** | **54.09%** | +0.0575 | 0.5474 | 0.5285 | 0.2566 | 54.23% |
| **CodeBERT Original ([CLS] 256)** | 52.22% | 50.87% | 53.38% | 52.09% | +0.0449 | 0.5425 | 0.5340 | 0.2490 | 51.07% |
| **Tri-Union Best Classical** | 52.73% | 51.45% | 51.21% | 51.33% | +0.0539 | 0.5418 | 0.5367 | 0.2480 | **48.45%** |
| **CodeBERT Imp Hybrid ($t^*=0.52$)** | **54.47%** | **54.20%** | 41.65% | 47.10% | **+0.0855** | **0.5617** | **0.5512** | **0.2461** | 37.40% |
| **CodeBERT Imp Hybrid ($t^*=0.50$)** | 54.25% | 52.99% | 53.15% | 53.07% | +0.0844 | **0.5617** | **0.5512** | **0.2461** | 48.82% |

All metrics match the historical verification reports bit-for-bit.

---

## 5. Candidate Expansion Datasets Available for Integration

Through inspection of Hugging Face APIs and open-source repositories, five candidate corpora were identified:

| Priority | Candidate Dataset | Official Hugging Face Source | Total Samples | Available Features | Suitability & Role |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | **DiverseVul** | `bstee615/diversevul` | 330,492 samples | `func`, `target`, `cwe`, `project`, `commit_id`, `message` | **High Priority:** 150+ real C/C++ projects, CVE-verified flaws, rich commit and CWE metadata. |
| **2** | **BigVul** | `bstee615/bigvul` | 217,007 samples | `func_before`, `func_after`, `vul`, `CWE ID`, `project`, `commit_id` | **High Priority:** Pre/post patch pairs, C/C++ coverage, explicit line additions/deletions. |
| **3** | **ReVeal** | `claudios/ReVeal` | 22,734 samples | `functionSource`, `label`, `project`, `hash` | **Medium Priority:** Real-world Chromium and Debian C/C++ vulnerabilities. |
| **4** | **CVEFixes** | `DetectVul/CVEFixes` | 5,730 samples | `lines`, `raw_lines`, `label`, `type` | **High Value for Diff/Line Models:** Line-level supervision and commit diffs. |
| **5** | **Juliet C/C++** | `LorenzH/juliet_test_suite_c_1_3` | 100,883 samples | `good`, `bad`, `class`, `filename` | **Synthetic Benchmark:** NIST synthetic test suite; kept strictly separate from real-world evaluations. |

---

## 6. Execution Plan for Phases 2–6

1. **Phase 2 (Dataset Integration):** Download and preprocess DiverseVul and BigVul, check label definitions, deduplicate against existing Devign/PrimeVul corpus, and preserve commit/project metadata.
2. **Phase 3 (Leakage-Resistant Splits):** Implement repository/project-aware grouping on the expanded dataset. Preserve the untouched original test set ($N=5,412$) to evaluate whether out-of-distribution expansion improves generalization on the original benchmark, while creating a distinct expanded test partition.
3. **Phase 4 & 5 (Model Training & Selection):** Train classical ensembles, enhanced code encoders, patch/diff representations, and context windowing.
4. **Phase 6 (Target Verification):** Evaluate whether 75% test accuracy is scientifically achievable or bounded by the inherent noise of function-level vulnerability labeling.
