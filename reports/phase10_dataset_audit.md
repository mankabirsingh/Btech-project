# Phase 10 Dataset Audit: Verified Schemas, Provenance Fidelity, and Label Integrity

**Project:** Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code  
**Date:** October 27, 2026  
**Auditors:** Senior ML Engineer & Research Team  

---

## 1. Executive Summary

This audit establishes the empirical ground truth for all datasets locally stored in the repository. We rigorously distinguish:
1. **Vulnerability-labelled datasets** (historical open-source CVE commits).
2. **Provenance-labelled datasets** (authorship attribution benchmarks).
3. **Dual-labelled datasets** (datasets containing both verified vulnerability labels and verified human–AI provenance).

---

## 2. Inventory of Available Datasets

| Dataset Name | Total Usable Samples | Vulnerability Label | Provenance Label | Class Distribution | Provenance Definition & Fidelity | Eligible Target Model |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **DiverseVul** | 106,874 (in train+val) | Yes (`vuln_label` $\in \{0, 1\}$) | None (100% Human) | 21.2% Vulnerable, 78.8% Safe | Historical pre-2023 C/C++ git commits. Zero AI provenance. | **Model A Only** |
| **BigVul** | 24,518 (in train+val) | Yes (Vuln / Fix) | None (100% Human) | 21.3% Vulnerable, 78.7% Safe | Historical commit diffs and functions. Zero AI provenance. | **Model A Only** |
| **Devign** | 23,151 (in train+val) | Yes (`vuln_label` $\in \{0, 1\}$) | None (100% Human) | 46.5% Vulnerable, 53.5% Safe | Curated QEMU / FFmpeg commits (2019). Zero AI provenance. | **Model A Only** |
| **PrimeVul** | 7,517 (in train+val) | Yes (`vuln_label` $\in \{0, 1\}$) | None (100% Human) | 24.8% Vulnerable, 75.2% Safe | Curated multi-repository CVEs. Zero AI provenance. | **Model A Only** |
| **Held-Out Repos (Clean)** | 3,169 | Yes (`vuln_label` $\in \{0, 1\}$) | None (100% Human) | 31.6% Vulnerable, 68.4% Safe | OpenSSL (1,257), GPAC (793), Envoy (603), Vim (516). | **Model A Test Only** |
| **HumanVsAICode** | 1,986,283 | **None (100% Missing / NaN)** | Yes (`provenance_type`) | 74.5% AI (1.48M), 25.5% Human (507k) | Synthetic code generation from competitive programming / docstrings. No CVE or security assessment. | **Authorship Learning Only** |
| **Controlled Multi-CWE Benchmark** | 32 | Yes (Ground Truth) | **Yes (Verified Lineage)** | 50.0% Vulnerable (16), 50.0% Safe (16) | 8 security tasks with 4-stage verified lineage (Human Safe, Human Buggy, AI Buggy, Human-Refactored AI Safe). | **Model B Locked Evaluation** |

---

## 3. Deep-Dive Findings by Dataset Category

### A. Vulnerability-Labelled Datasets (Model A Training Corpus)
- **Total Expanded Volume:** 138,412 training samples, 23,648 validation samples, and 3,169 held-out cross-project test samples.
- **Labels:** High-confidence CVE vulnerability indicators (`vuln_label = 1` for vulnerable pre-patch functions; `vuln_label = 0` for post-patch or non-vulnerable functions).
- **Critical Finding on Provenance:**
  - In `vulnerability_train_expanded.parquet`, all 138,412 rows have `provenance_type = 'human'` and `ai_involvement = 0.0`.
  - These corpora were mined from Git repositories prior to 2022 (before the emergence of modern LLM code generators such as ChatGPT or GitHub Copilot).
  - **Rule Enforcement:** We strictly refuse to assign synthetic AI provenance labels to these historical samples. Presenting ordinary historical vulnerabilities as "AI-generated" would be scientifically fraudulent.

### B. HumanVsAICode (Authorship Attribution Dataset)
- **Total Records:** 1,986,283 samples stored in `data/processed/provenance_data.parquet`.
- **Columns:** `sample_id`, `code`, `language`, `vuln_label`, `provenance_type`, `ai_generator`, `ai_involvement`.
- **Label Analysis:**
  - `vuln_label`: Exactly 1,986,283 NaN values (100% null).
  - `provenance_type`: 1,479,238 `ai` and 507,045 `human`.
- **Critical Finding:**
  - `HumanVsAICode` contains zero security ground truth.
  - While it can train a stylistic authorship classifier (distinguishing human code from LLM code), authorship does not equal security risk.
  - Training a vulnerability classifier on `HumanVsAICode` would confound code style with vulnerability, violating basic research integrity.

### C. Controlled Multi-CWE Benchmark (The 32 Verified Examples)
- **Origin:** Created in `src/experiments/controlled_editing.py` and saved in `results/reports/controlled_benchmark_32.json`.
- **Tasks (8 Seed Tasks):**
  1. `task_buffer_overflow` (CWE-119 / CWE-120: Buffer Copy)
  2. `task_format_string` (CWE-134: Uncontrolled Format String)
  3. `task_integer_overflow` (CWE-190: Integer Overflow in Allocation)
  4. `task_command_injection` (CWE-78: OS Command Injection via system/popen)
  5. `task_use_after_free` (CWE-416: Use After Free in Pointer Lifecycles)
  6. `task_path_traversal` (CWE-22: Path Traversal via Relative Path)
  7. `task_insecure_crypto` (CWE-327: Broken Crypto Algorithm / ECB Mode)
  8. `task_null_pointer` (CWE-476: NULL Pointer Dereference)
- **Lineage Structure per Task:**
  - `v1_human_base`: Human written, safe baseline (AI=0%, Vuln=0).
  - `v2_human_buggy`: Human written, vulnerable flaw (AI=0%, Vuln=1).
  - `v3_ai_raw`: AI generated from prompt, vulnerable flaw (AI=100%, Vuln=1).
  - `v4_human_edited_ai`: Human developer review and remediation of AI code, safe (AI=50%, Vuln=0).
- **Total Units:** 32 verified examples (16 vulnerable, 16 safe).
- **Validation Role:** Preserved completely as a **locked held-out test benchmark**.

---

## 4. Methodological Conclusion for Phase 10

To train a genuinely generalizable Model B without contaminating the 32 locked benchmark examples:
1. **The 32 benchmark examples are locked as an immutable test set** with cryptographic SHA-256 verification.
2. A separate, non-overlapping **development dataset of seed programs** must be established to train Model B's fusion layer.
3. Model B must learn the multi-factor interplay of code representations and provenance metadata without having seen any of the 8 locked benchmark tasks.
