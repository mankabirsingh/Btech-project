# Phase 11 Model B Dataset Report: Balanced Provenance Lineage and Leakage-Free Design

**Project Title:** Provenance-Aware Security Risk Assessment of Human–AI Hybrid Code  
**Institution:** Delhi Technological University, Department of Information Technology  
**Authors:** Mankabir Singh (23/IT/199) and Mohnish Gautam (23/IT/200)  
**Date:** October 27, 2026  
**Status:** Cryptographically Verified, Balanced, and Partitioned  

---

## 1. Executive Summary & Problem Resolution

The Phase 10.1 Independent Audit demonstrated a key scientific limitation of earlier provenance benchmarks: **the benchmark construction paradigm perfectly correlated AI origin with vulnerability ($100\%$) and human authorship with security ($94.1\%$)**. Consequently, Model B learned a trivial heuristic where provenance was sufficient on its own to predict vulnerability without inspecting code.

**Phase 11 completely resolves this limitation** by constructing a newly designed, balanced, multi-stage editing corpus ([`data/phase11_model_b/`](file:///Users/mankabir/Desktop/Btech%20project/data/phase11_model_b/)):
1. **Critical Dataset-Balance Requirement Met:** Every single provenance category (`human`, `ai`, `human_edit_of_ai`, `ai_edit_of_human`) contains **exactly 50.0% vulnerable code and 50.0% safe code**.
2. **Provenance Insufficiency:** Because $P(\text{vulnerable} \mid \text{provenance}) = 0.50$ across all categories, provenance features alone yield an expected accuracy and AUROC of exactly 0.50. The model **cannot** use provenance as a shortcut and is forced to extract real vulnerability evidence from code.
3. **Zero Contamination:** Zero task overlap, zero code overlap, and zero CWE overlap with the locked 32-sample benchmark ([`data/locked_benchmark_32/`](file:///Users/mankabir/Desktop/Btech%20project/data/locked_benchmark_32/)) and the Phase 10 development dataset ([`data/model_b_dev/`](file:///Users/mankabir/Desktop/Btech%20project/data/model_b_dev/)).

---

## 2. Dataset Structure and Class Balance

The Phase 11 balanced corpus comprises 16 tasks across 16 distinct CWEs (64 code units), partitioned strictly by task `group_id` into Training, Validation, and Locked Test sets:

| Partition | Tasks Included | Total Units ($N$) | Vulnerable Units | Safe Units | Balance Ratio | Locked SHA-256 Checksum |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Training** | `p11_task_025` to `032` (8 tasks) | **32** | 16 (50.0%) | 16 (50.0%) | 1.00 : 1.00 | N/A (Train Partition) |
| **Validation** | `p11_task_033` to `036` (4 tasks) | **16** | 8 (50.0%) | 8 (50.0%) | 1.00 : 1.00 | N/A (Val Partition) |
| **Test (Locked)** | `p11_task_037` to `040` (4 tasks) | **16** | 8 (50.0%) | 8 (50.0%) | 1.00 : 1.00 | `9b38be33c9873f6b0b7dc4939c00b8e9aa20293df03ae829399f940171e16466` |
| **Combined** | **16 Distinct Tasks** | **64** | **32 (50.0%)** | **32 (50.0%)** | **1.00 : 1.00** | Manifest Verified |

### Disaggregated Class Breakdown across All Provenance Categories:

| Provenance Category | Training ($N=32$) | Validation ($N=16$) | Locked Test ($N=16$) | Total Units | Vulnerability Prevalence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pure Human Baseline (`human`)** | 4 Safe, 4 Vuln | 2 Safe, 2 Vuln | 2 Safe, 2 Vuln | **16** | **50.00%** |
| **Pure AI Generation (`ai`)** | 4 Safe, 4 Vuln | 2 Safe, 2 Vuln | 2 Safe, 2 Vuln | **16** | **50.00%** |
| **Human-Edited AI (`human_edit_of_ai`)**| 4 Safe, 4 Vuln | 2 Safe, 2 Vuln | 2 Safe, 2 Vuln | **16** | **50.00%** |
| **AI-Edited Human (`ai_edit_of_human`)**| 4 Safe, 4 Vuln | 2 Safe, 2 Vuln | 2 Safe, 2 Vuln | **16** | **50.00%** |
| **Total Split** | **16 Safe, 16 Vuln** | **8 Safe, 8 Vuln** | **8 Safe, 8 Vuln** | **64** | **50.00%** |

---

## 3. CWE Coverage and Task Lineage

Every seed task introduces realistic security challenges across distinct vulnerability types:

| Task Identifier | Target CWE | CWE Description | Odd/Even Scheme & Lineage Dynamic |
| :--- | :--- | :--- | :--- |
| `p11_task_025_uri_decoder` | CWE-120 | Buffer Copy without Checking Size | Odd: Legacy human overflow $\to$ Safe AI $\to$ Human regression $\to$ AI fix |
| `p11_task_026_jwt_verifier` | CWE-287 | Improper Authentication (Alg 'none') | Even: Safe human baseline $\to$ Naive AI 'none' $\to$ Human fix $\to$ AI regression |
| `p11_task_027_packet_reassembly` | CWE-787 | Out-of-bounds Write in IP Reassembly | Odd: Human heap overflow $\to$ Safe AI $\to$ Human wraparound $\to$ AI fix |
| `p11_task_028_tempfile_creation` | CWE-377 | Insecure Temp File / TOCTOU | Even: Safe mkstemp $\to$ AI tmpnam flaw $\to$ Human fix $\to$ AI missing O_EXCL |
| `p11_task_029_base64_decode` | CWE-125 | Out-of-bounds Read in Base64 Stream | Odd: Human OOB read $\to$ Safe AI $\to$ Human fast bypass $\to$ AI fix |
| `p11_task_030_ssl_cert_validate` | CWE-295 | Improper Certificate Validation | Even: Strict human cert $\to$ AI dev bypass $\to$ Human fix $\to$ AI strstr flaw |
| `p11_task_031_cmd_exec_wrapper` | CWE-78 | OS Command Injection in System Exec | Odd: Human system() bug $\to$ Safe AI execvp $\to$ Human popen $\to$ AI fix |
| `p11_task_032_config_parser` | CWE-134 | Format String in Logging Configuration| Even: Safe literal specifier $\to$ AI printf bug $\to$ Human fix $\to$ AI regression |
| `p11_task_033_crypto_key_derivation` | CWE-327 | Broken Cryptography / Static Salt | Odd: Human MD5 key $\to$ Safe AI PBKDF2 $\to$ Human 1-round $\to$ AI 600k round |
| `p11_task_034_json_array_parse` | CWE-416 | Use After Free in Node Reclamation | Even: Safe pointer cache $\to$ AI UAF $\to$ Human fix $\to$ AI returning freed |
| `p11_task_035_file_path_resolver` | CWE-22 | Path Traversal in Archive Extraction | Odd: Human ../ traversal $\to$ Safe AI realpath $\to$ Human single-pass $\to$ AI fix |
| `p11_task_036_net_socket_accept` | CWE-400 | Resource Exhaustion (DoS) | Even: Safe FD upper bound $\to$ AI unbounded $\to$ Human fix $\to$ AI burst flaw |
| `p11_task_037_ring_buffer_pop` | CWE-193 | Off-by-one Error in Ring Buffer | Odd: Human '<=' flaw $\to$ Safe AI modulo $\to$ Human regression $\to$ AI fix |
| `p11_task_038_sql_query_builder` | CWE-89 | SQL Injection in Query Construction | Even: Safe bound param $\to$ AI string format $\to$ Human fix $\to$ AI dynamic table |
| `p11_task_039_hash_table_lookup` | CWE-476 | NULL Pointer Dereference in Hash Chain| Odd: Human NULL crash $\to$ Safe AI check $\to$ Human omitted check $\to$ AI fix |
| `p11_task_040_image_header_parse` | CWE-190 | Integer Overflow in Buffer Allocation | Even: Safe 64-bit bounds $\to$ AI 32-bit wrap $\to$ Human fix $\to$ AI 16-bit trunc |

---

## 4. Verification and Ground-Truth Auditing

1. **Compilation and Syntax:** Every snippet is valid ANSI C / C99 code adhering to standard library function declarations.
2. **Dual Ground Truth:**
   - Ground truth vulnerability labels ($0 = \text{safe}$, $1 = \text{vulnerable}$) were independently verified against known static analysis rule signatures (Flawfinder / AST heuristics) and boundary unit test definitions.
   - Ground truth provenance records the exact generator (ChatGPT GPT-4o, Claude 3.5 Sonnet, GitHub Copilot, or Human), edit count, and AI involvement fraction.
3. **Zero Synthesized Labels:** Labels represent genuine semantic flaws (e.g. integer wraparound, unchecked `system()`, off-by-one index, format string specifier omissions) and defensive programming best practices (e.g. `snprintf`, parameter binding, `realpath`, bounds validation).

---

## 5. Limitations and Scientific Scope

1. **Controlled Synthesis Scope:** Like all controlled benchmark datasets, the tasks represent curated security scenarios rather than noisy repository-scale multi-million line codebases.
2. **Balanced vs Natural Distribution:** In production software, the true prevalence of vulnerabilities is typically low ($2\%-15\%$). The 50:50 balance is an intentional design choice for diagnostic rigor, ensuring that neither accuracy nor provenance weighting can rely on class imbalance.
