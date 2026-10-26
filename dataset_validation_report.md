# Dataset Validation Report
**B.Tech Project-I — Delhi Technological University**  
**Project Title:** Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code  
**Investigators:** Mankabir Singh (23/IT/199) & Mohnish Gautam (23/IT/200)  
**Date:** October 26, 2026  

---

## 1. Overview & Dataset Sources

This report provides a formal validation of all dataset sources used in the research project, documenting raw vs. processed sample counts, schema standardization, label distributions, and scientific suitability for vulnerability and provenance tasks.

```
+----------------------------------------------------------------------------------------------------+
|                                    DATASET PORTFOLIO OVERVIEW                                      |
+--------------------------+-----------------------+---------------------+---------------------------+
| Dataset Identifier       | Domain / Task         | Language Scope      | Primary Utility           |
+--------------------------+-----------------------+---------------------+---------------------------+
| DetectVul/devign         | Vulnerability Det.    | C/C++ (Qemu, FFmpeg)| Code Vulnerability Ground |
| adefu020/primevul        | Vuln + Fix Pairs      | C/C++ (Diverse OSS) | Paired Vuln & Patched Code|
| OSS-forge/HumanVsAICode  | Paired Provenance     | Python & Java       | Cross-Generator Provenance|
+--------------------------+-----------------------+---------------------+---------------------------+
```

---

## 2. In-Depth Dataset Breakdown

### 2.1 `DetectVul/devign`
- **Domain:** Function-level vulnerability classification on real-world C projects (QEMU, FFmpeg).
- **Raw Samples:** 27,318 functions across train, validation, and test splits.
- **Cleaned & Processed:** 27,258 functions (60 empty or unparseable functions filtered).
- **Label Distribution:**
  - Benign (`target = 0`): 14,834 (54.4%)
  - Vulnerable (`target = 1`): 12,424 (45.6%)
- **Commit Isolation:** 10,533 unique `commit_id` groups mapped to prevent cross-function commit leakage.
- **Provenance Status:** 100% human-authored open-source commits (pre-2019). Supports vulnerability detection; constant human provenance.

### 2.2 `adefu020/primevul` (Corrected Implementation)
- **Domain:** Curated C/C++ function pairs containing confirmed CVE vulnerabilities and their corresponding security fixes.
- **Raw Samples:** 8,832 rows (4,416 paired commits).
- **Previous Defect Discovered in Audit:** Earlier pipeline extracted `vulnerable_function` for both positive and negative rows, which incorrectly erased all 4,416 benign fixed functions during deduplication.
- **Corrected Ingestion:**
  - For `is_vulnerable == True`: extracted cleaned `vulnerable_function`.
  - For `is_vulnerable == False`: extracted cleaned `fixed_function`.
- **Cleaned & Processed:** 8,822 functions (10 duplicates removed).
  - Vulnerable (`vuln_label = 1`): 4,416 (50.06%)
  - Benign (`vuln_label = 0`): 4,406 (49.94%)
- **Pair Grouping:** Each function pair is bound to `group_id = pv_pair_{source_record_id}`, guaranteeing that the vulnerable function and its fixed variant remain in the same partition.

### 2.3 `OSS-forge/HumanVsAICode`
- **Domain:** Parallel task implementations across multiple author identities.
- **Raw Samples:** 1,986,283 code samples across 507,045 unique programming tasks.
- **Provenance Distribution:**
  - Human: 507,045 samples (25.5%)
  - ChatGPT: 507,045 samples (25.5%)
  - Qwen: 503,835 samples (25.4%)
  - DeepSeek-Coder: 468,358 samples (23.6%)
- **Critical Scientific Rule Observed:** `HumanVsAICode` **does not** contain verified ground-truth vulnerability labels. In accordance with Rule 4 and Rule 6, authorship labels are strictly treated as **provenance metadata**, not vulnerability labels.

---

## 3. Unified Vulnerability Dataset Summary

The unified vulnerability dataset combines `DetectVul/devign` and `adefu020/primevul`:

```
================================================================================
UNIFIED VULNERABILITY DATASET STATISTICS
================================================================================
Total Processed Samples:     36,080
Total Unique Code Functions: 36,080 (0 exact duplicates)
Vulnerable Functions:        16,840 (46.67%)
Benign Functions:            19,240 (53.33%)
Unique Disjoint Groups:      16,709 (commit- and pair-level groups)
Primary Languages:           C / C++ (100%)
================================================================================
```

### Schema Attributes (16 Columns)
1. `sample_id`: Unique identifier across datasets
2. `code`: Cleaned source code (LF normalized, trailing spaces removed)
3. `language`: Canonical programming language (`C/C++`)
4. `vuln_label`: Binary integer ground-truth (0 = benign, 1 = vulnerable)
5. `cwe`: Primary CWE identifier (where verified in CVE commits, NaN otherwise)
6. `severity`: Severity rating (where verified)
7. `source_dataset`: Origin (`devign` or `primevul`)
8. `task_id`: Benchmark task identifier
9. `provenance_type`: Author category (`human`, `ai`, `mixed`)
10. `ai_generator`: Model name if generated (`ChatGPT`, `DeepSeek-Coder`, etc.)
11. `ai_involvement`: Float ratio in `[0.0, 1.0]` (0.0 = human, 1.0 = pure AI)
12. `human_edit_history`: JSON tracking edit iterations
13. `parent_sample_id`: Links fixed function to vulnerable parent function
14. `project`: Origin repository (e.g. `qemu`, `FFmpeg`, `unzip`, `openssl`)
15. `commit_id`: Git commit hash
16. `group_id`: Group partition key for leakage-free splitting

---

## 4. Verification Check

- [x] All raw sample counts verified against HuggingFace datasets.
- [x] Benign samples in PrimeVul correctly restored and paired with fixes.
- [x] Missing value counts profiled and cataloged in `data/reports/`.
- [x] No artificial or synthetic labels injected into real-world datasets.
