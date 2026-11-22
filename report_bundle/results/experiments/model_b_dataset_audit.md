# Model B Dataset Audit: Provenance Labels, Vulnerability Ground Truth & Training Feasibility

**Project:** Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code  
**Date:** October 2026  
**Roles:** Senior ML Engineer, Full-Stack Developer, Research Engineer  
**Scope:** Rigorous investigation of all datasets in the repository to determine their legitimacy for training Model A (code-only) versus Model B (code + provenance).

---

## 1. Executive Summary

A critical scientific requirement of this project is to avoid conflating ordinary source code with AI-generated code. We conducted a comprehensive schema and label audit across all available datasets:

### Key Audit Conclusions:
1. **Public Vulnerability Corpora Are 100% Pre-AI Human Code:**
   Datasets mined from open-source repositories (Devign, PrimeVul, BigVul, DiverseVul) contain authentic CVE vulnerability labels, but **0% AI-generated code**. All records represent human-authored commits from 2002–2021, prior to the adoption of LLM coding assistants. They are fully legitimate for **Model A**, but cannot be used to train **Model B** without inventing artificial provenance labels.
2. **AI Code Authoring Datasets Contain Zero Vulnerability Labels:**
   The `HumanVsAICode` dataset ([`data/processed/provenance_data.parquet`](file:///Users/mankabir/Desktop/Btech%20project/data/processed/provenance_data.parquet), 1,986,283 samples) contains genuine human versus AI authorship labels (ChatGPT, Qwen, DeepSeek), but has **100% missing (`NaN`) vulnerability labels**. It is designed for authorship attribution, not vulnerability detection.
3. **The Multi-CWE Controlled Editing Benchmark Is the Sole Legitimate Ground Truth:**
   The controlled editing benchmark ([`src/experiments/controlled_editing.py`](file:///Users/mankabir/Desktop/Btech%20project/src/experiments/controlled_editing.py), 32 samples across 8 CWE tasks) is the only dataset in the repository that simultaneously possesses **genuine human/AI editing histories, verified AI-involvement fractions ($0.0 \le \text{AI} \le 1.0$), and ground-truth vulnerability annotations**.
4. **Transparent Presentation Policy for Model B:**
   Because large-scale real-world AI vulnerability datasets do not exist in the open research community, Model B’s fusion layer is trained on the controlled benchmark and uses the pre-trained code encoder. It is honestly presented in the UI and reports as a **Preliminary Provenance-Aware Prototype**, avoiding false claims of large-scale statistical validation.

---

## 2. Dataset-by-Dataset Audit Matrix

| Dataset | Local Samples | Vuln Labels ($Y \in \{0, 1\}$) | Provenance Labels Present? | Provenance Trustworthiness | Can Train Model A? | Can Train Model B? |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: |
| **Devign** | 27,258 | Yes (CWE-verified) | No (100% Human, AI=0.0) | Mined from Linux, QEMU, FFmpeg (Pre-LLM) | **Yes** | **No** (Zero AI variance) |
| **PrimeVul** | 8,822 | Yes (CVE-verified) | No (100% Human, AI=0.0) | Historical Git commits (Pre-LLM) | **Yes** | **No** (Zero AI variance) |
| **BigVul** | 20,752 | Yes (Paired diffs) | No (100% Human, AI=0.0) | Historical Git commits (Pre-LLM) | **Yes** | **No** (Zero AI variance) |
| **DiverseVul** | 90,600 | Yes (Security commits)| No (100% Human, AI=0.0) | Historical Git commits (Pre-LLM) | **Yes** | **No** (Zero AI variance) |
| **Juliet (NIST/NSA)** | 4,000 | Yes (Balanced 50/50)| Synthetic (AI=0.0) | Hand-crafted C/C++ test suites | **Yes** (Eval) | **No** (Not Generative AI) |
| **HumanVsAICode** | 1,986,283 | **No (100% NaN)** | Yes (ChatGPT, Qwen, DeepSeek) | Genuine AI vs Human Solutions | **No** (No vuln labels) | **Authorship Only** |
| **Controlled Multi-CWE Benchmark**| 32 | Yes (16 vuln, 16 safe)| **Yes (Full Edit Lineage)** | **Verified controlled multi-version experiment** | **Yes** (Eval) | **Yes (Prototype Fitting)** |

---

## 3. Deep Dive into Specific Datasets

### 3.1 Historical Vulnerability Corpora (Devign, PrimeVul, BigVul, DiverseVul)
- **Usable Records:** 138,412 training records, 36,080 initial baseline records.
- **Provenance Columns:** `provenance_type = 'human'`, `ai_involvement = 0.0`, `ai_generator = None`, `human_edit_history = None`.
- **Scientific Audit Finding:** Ordinary Git commit metadata (author names, timestamps, commit messages) does not constitute proof of AI generation. Assigning synthetic AI labels to these historical functions would corrupt the research validity. These datasets strictly train **Model A**.

### 3.2 HumanVsAICode (`data/processed/provenance_data.parquet`)
- **Usable Records:** 1,986,283 samples (507,045 Human, 507,045 ChatGPT, 503,835 Qwen, 468,358 DeepSeek-Coder).
- **Vulnerability Column:** `vuln_label` is completely missing (`NaN` for all rows).
- **Scientific Audit Finding:** Mined from LeetCode and Codeforces solutions. While ideal for training code stylometry and AI authorship classifiers, it cannot train vulnerability prediction without hallucinating security labels.

### 3.3 Controlled Multi-CWE Benchmark (`src/experiments/controlled_editing.py`)
- **Usable Records:** 32 code units organized into 8 tasks and 4 evolutionary versions per task:
  1. `v1_human_insecure`: Human-written vulnerable code (e.g. CWE-120 buffer overflow, CWE-22 traversal).
  2. `v2_ai_insecure`: Generative AI completion reproducing the insecure pattern (`ai_involvement = 1.0`).
  3. `v3_human_edited_ai`: Human developer remediation of the AI code (`ai_involvement = 0.5`).
  4. `v4_ai_repaired`: Fully secure AI or human-repaired version (`vuln_label = 0`).
- **Class Balance:** Exactly 16 vulnerable ($Y=1$) and 16 secure ($Y=0$).
- **Scientific Audit Finding:** This is the **only legitimate dataset** in the repository supporting Model B's code-plus-provenance objective.

---

## 4. Implementation Protocol for Model B

In accordance with strict pairing rules:
1. **Model A:** Evaluates source code alone using the 50k-sample expanded Tri-Union model and CodeBERT representations.
2. **Model B:** Evaluates identical source code, augmented with user-supplied or IDE-extracted provenance features:
   - `ai_involvement` ($0.0 \le x \le 1.0$)
   - `provenance_type` (Human, AI, Mixed)
   - `ai_generator` (ChatGPT, Copilot, Claude, None)
   - `human_edit_count`
3. **Honest Presentation Status:** Model B will display:
   `Status: Preliminary Provenance-Aware Prototype (Trained on Controlled Benchmark; Zero Synthetic Provenance on Open Corpora)`.
