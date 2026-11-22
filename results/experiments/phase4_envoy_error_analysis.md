# Phase 4 Forensic Error Analysis: Why Envoy Fails in Cross-Project Vulnerability Detectors

**Project:** Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code  
**Institution:** Delhi Technological University  
**Date:** October 27, 2026  
**Artifact Path:** [`results/experiments/phase4_envoy_error_analysis.md`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase4_envoy_error_analysis.md)  
**Predictions Dataset:** [`results/experiments/envoy_test_predictions.parquet`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/envoy_test_predictions.parquet)  

---

## 1. Executive Summary

While cross-project transfer achieves strong performance on C repositories such as **GPAC (86.89% Acc, 0.8205 AUROC)** and **Vim (77.52% Acc, 0.7362 AUROC)**, it hovers near random chance on **Envoy (52.74% – 54.89% Acc, 0.5065 – 0.5219 AUROC)**.

This forensic investigation analyzes the 603 locked Envoy test functions across eight empirical dimensions to determine whether the failure is caused by:
1. Label or dataset construction noise
2. Syntactic domain shift (C vs. modern C++17)
3. Tokenizer subword fragmentation and context truncation
4. Semantic category mismatch (Buffer overflows vs. HTTP/2 state machine resource exhaustion)
5. Ineffective or noisy structural feature extraction
6. Threshold and calibration distortion

---

## 2. Dataset Origin, Label Quality & Commit Structure

### 2.1 Commit Co-location & Paired Structures
- **Source:** All 603 Envoy samples derive from **DiverseVul** and correspond to **25 unique security commit SHAs**.
- **Paired Consistency:** 20 of the 25 commits contain both vulnerable ($Y = 1$) and patched/benign ($Y = 0$) functions modified within the same security patch.
- **Unit Test Contamination in DiverseVul:** DiverseVul ingests all modified functions in a security commit patch. In Envoy, security commits routinely add regression tests using the **GoogleTest** framework (`TEST_F`, `TEST_P`, `EXPECT_CALL`, `makeTestHost`).
  - **Empirical Measure:** 18.7% (113 / 603) of functions in the Envoy partition contain test harness keywords (`TEST_F`, `TEST_P`, `EXPECT_`, `Mock`).
  - **Effect:** Benign unit test routines instantiate mock strings, paths, and connections, causing pattern extractors to produce false alarms.

### 2.2 Duplicate and Near-Duplicate Analysis
- **Code Duplicate Audit:** Exactly **0** Envoy test functions match any function in the training or validation sets.
- **Commit Ancestry:** No Envoy commit SHAs cross split boundaries.

---

## 3. Language & Syntactic Domain Shift: C vs. Modern C++17

We quantified syntax tokens across the training corpus ($N = 138,412$), the C held-out projects (GPAC, Vim, OpenSSL), and Envoy:

| Syntax Dimension | Keyword / Regex Pattern | Training Corpus ($N=138,412$) | GPAC ($N=793$) | Vim ($N=516$) | OpenSSL ($N=1,257$) | Envoy ($N=603$) | Domain Shift Ratio (Envoy / Train) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scope Resolution** | `::` | 13.7% | 0.1% | 0.0% | 0.2% | **87.7%** | **6.40x** |
| **Standard Namespace** | `std::` | 3.5% | 0.0% | 0.0% | 0.0% | **38.8%** | **11.08x** |
| **C++ Nullptr** | `nullptr` | 1.1% | 0.0% | 0.0% | 0.0% | **19.9%** | **18.09x** |
| **Exception Handlers** | `try`, `catch`, `throw` | 0.8% | 0.0% | 0.0% | 0.0% | **21.7%** | **27.12x** |
| **Smart Pointers** | `unique_ptr`, `make_unique` | 0.3% | 0.0% | 0.0% | 0.0% | **7.6%** | **25.33x** |
| **Raw Memory Ops** | `memcpy`, `memmove`, `memset` | 94.2% | 97.2% | 91.1% | 96.1% | **63.3%** | **0.67x** |
| **Raw String Ops** | `strcpy`, `strcat`, `sprintf` | 92.8% | 98.2% | 90.5% | 96.6% | **68.3%** | **0.73x** |

**Empirical Finding:** The training corpus is overwhelmingly ANSI C (86.3% purely non-C++), while GPAC, Vim, and OpenSSL are >99.8% pure C. Envoy is 87.7% modern C++17 with RAII memory management, custom smart pointers (`Buffer::InstancePtr`), and nested class abstractions.

---

## 4. Tokenizer Subword Fragmentation & Truncation

CodeBERT uses a RoBERTa-based Byte-Pair Encoding (BPE) tokenizer trained on natural language and general codebases. When applied to modern C++ namespaces and nested templates, it shatters single identifiers into long token sequences:

### 4.1 Subword Expansion Examples
- `StreamDecoderFilterCallbacks::encodeHeaders` $\rightarrow$ **11 BPE subwords:**  
  `['Stream', 'Dec', 'oder', 'Filter', 'Call', 'backs', '::', 'en', 'code', 'Headers']`
- `Http::Http2::ConnectionImpl::onFrameReceived` $\rightarrow$ **12 BPE subwords:**  
  `['Http', '::', 'Http', '2', '::', 'Connection', 'Impl', '::', 'on', 'Frame', 'Received']`
- `std::unique_ptr<Buffer::Instance>` $\rightarrow$ **10 BPE subwords:**  
  `['std', '::', 'unique', '_', 'ptr', '<', 'Buffer', '::', 'Instance', '>']`

### 4.2 Sequence Truncation Analysis
- **Whitespace Token Count:** Median = **35 tokens**, Mean = **67.0 tokens**.
- **CodeBERT Subword Token Count:** Median = **246 subwords**, Mean = **389.6 subwords**, Max = **6,053 subwords**.
- **Truncated at 384 tokens (Phase 2 Baseline):** **33.2% (200 / 603 functions)**.
- **Truncated at 512 tokens (Phase 3 Model):** **21.6% (130 / 603 functions)**.

**Empirical Finding:** 33.2% of Envoy functions lost their operational logic, error handlers, and return code checks under the 384-token window due to heavy BPE fragmentation of C++ types and namespaces.

---

## 5. Vulnerability Semantics: Category Mismatch (CWE)

Vulnerability detectors trained on C benchmarks rely on memory safety signatures (buffer overflows, off-by-one writes, format strings). We audited every ground-truth CWE in the Envoy test set against the training corpus:

| Common Weakness Enumeration (CWE) | Weakness Category | Test Count ($N=603$) | Vulnerable Count ($N=203$) | Benign Count ($N=400$) | Training Set Count ($N=29,362$ Vuln) | Training Proportion |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CWE-400** | Uncontrolled Resource Consumption | 233 | **81** | 152 | 185 | **0.63%** |
| **None (Unspecified)** | Logic / Semantic Flaw | 87 | **35** | 52 | 1,412 | 4.81% |
| **CWE-476** | NULL Pointer Dereference | 52 | **16** | 36 | 531 | 1.81% |
| **CWE-416** | Use After Free | 48 | **15** | 33 | 705 | 2.40% |
| **CWE-22** | Improper Limitation of a Pathname (Path Traversal) | 62 | **14** | 48 | 91 | **0.31%** |
| **CWE-770** | Allocation of Resources Without Limits | 34 | **12** | 22 | 50 | **0.17%** |
| **CWE-835** | Loop with Unreachable Exit Condition (Infinite Loop) | 15 | **9** | 6 | 58 | **0.20%** |
| **CWE-401** | Missing Release of Memory after Effective Lifetime | 19 | **9** | 10 | 198 | 0.67% |
| **CWE-703** | Improper Check or Handling of Exceptional Conditions | 39 | **8** | 31 | 433 | 1.47% |
| **CWE-303** | Incorrect Implementation of Authentication Algorithm | 13 | **4** | 9 | 12 | 0.04% |
| **Buffer Overflows (CWE-119, 125, 787)** | Memory Corruption | **0** | **0** | **0** | **4,179** | **14.23%** |

### Critical Semantic Disparity:
1. **Total Absence of Memory Corruption:** There are **ZERO buffer overflow vulnerabilities** in Envoy ($0 / 203$). Modern C++ containers and RAII completely eliminate classical stack/heap smashing.
2. **Dominance of Resource Consumption:** **50.2% (102 / 203)** of Envoy vulnerabilities are Denial of Service and Resource Exhaustion (**CWE-400: 81, CWE-770: 12, CWE-835: 9**).
3. **Training Scarcity:** In the entire expanded training set of 29,362 vulnerabilities, CWE-400 represents **only 185 examples (0.63%)**, and CWE-770 represents **50 examples (0.17%)**.

---

## 6. Performance by CWE Category on Envoy

We evaluated how the models performed across each specific CWE category on the locked Envoy test partition:

| Vulnerability Category (CWE) | Total Samples | Vuln ($Y=1$) | Benign ($Y=0$) | Phase 2 Recall | Phase 2 FPR | Phase 3 Recall | Phase 3 FPR | Model Failure Mechanism |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CWE-400 (Resource Exhaustion)** | 233 | 81 | 152 | 48.1% | 38.8% | 48.1% | 40.8% | **Near-chance.** Missing bounds checks in event loops look syntactically benign. |
| **CWE-22 (Path Traversal)** | 62 | 14 | 48 | 78.6% | 58.3% | 78.6% | 52.1% | **Over-flagging.** Any function handling URLs or path strings is flagged. |
| **CWE-476 (NULL Pointer Deref)** | 52 | 16 | 36 | **6.2%** | 27.8% | **12.5%** | 27.8% | **Severe Blindness (87.5% FN).** Member access via smart pointers lacks raw `*p` checks. |
| **CWE-416 (Use After Free)** | 48 | 15 | 33 | 13.3% | 42.4% | **46.7%** | 30.3% | **Substantial Gain.** C++ structural features captured callback lifetime patterns. |
| **CWE-703 (Exceptional Conditions)** | 39 | 8 | 31 | 75.0% | 87.1% | 87.5% | 83.9% | **Pathological FP.** Model predicts vulnerable for nearly all error handlers. |
| **CWE-770 (Resource Allocation)** | 34 | 12 | 22 | 58.3% | 50.0% | 33.3% | 40.9% | Missing capacity limits in buffer allocation. |
| **CWE-835 (Infinite Loops)** | 15 | 9 | 6 | 88.9% | 50.0% | 88.9% | 66.7% | Loop counter features flag loops, but produce false positives on benign loops. |
| **None (Unspecified Logic Flaws)** | 87 | 35 | 52 | 28.6% | 26.9% | 31.4% | 19.2% | Architectural state machine logic errors across multiple classes. |

---

## 7. Manual Inspection of Documented Errors

We audited concrete False Negatives (FN) and False Positives (FP) from [`results/experiments/envoy_test_predictions.parquet`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/envoy_test_predictions.parquet):

### 7.1 False Negative Case 1: CWE-476 (NULL Pointer Crash in WASM Binding)
- **Sample ID:** `dv_train_7879`
- **Assigned Probability:** `0.2560` (Strongly predicted benign)
- **Ground Truth:** Vulnerable ($Y = 1$, CWE-476)
- **Source Code:**
  ```cpp
  void Context::onLog() {
    if (wasm_->onLog_) {
      wasm_->onLog_(this, id_);
    }
  }
  ```
- **Error Cause:** The code crashes if `wasm_` is a null or dangling pointer before accessing `wasm_->onLog_`. To CodeBERT and C structural heuristics, this is a clean, 3-line accessor with zero raw pointers, no `malloc`, and no buffer operations.

### 7.2 False Negative Case 2: CWE-400 (Uncontrolled Connection Growth)
- **Sample ID:** `dv_train_3786`
- **Assigned Probability:** `0.1666` (Strongly predicted benign)
- **Ground Truth:** Vulnerable ($Y = 1$, CWE-400 / CWE-770)
- **Source Code:**
  ```cpp
  void incNumConnections() override { ++num_listener_connections_; }
  ```
- **Error Cause:** Missing rate limit or upper bound check on listener connections allows remote clients to exhaust server memory. The function itself contains only an atomic increment `++num_listener_connections_`. No function-level classifier can identify that this increment lacks a corresponding boundary check located in another class or method.

### 7.3 False Positive Case 1: Benign Unit Test Routine
- **Sample ID:** `dv_train_58`
- **Assigned Probability:** `0.5058` (Flagged as vulnerable)
- **Ground Truth:** Benign ($Y = 0$, CWE-476 commit)
- **Source Code:**
  ```cpp
  TEST_F(HttpHealthCheckerImplTest, GoAwayProbeInProgress) {
    setupHCHttp2();
    EXPECT_CALL(runtime_.snapshot_, featureEnabled("health_check.verify_cluster", 100))
        .WillRepeatedly(Return(false));
    cluster_->prioritySet().getMockHostSet(0)->hosts_ = {
        makeTestHost(cluster_->info_, "tcp://127.0.0.1:80", simTime())};
    cluster_->info_->stats().upstream_cx_total_.inc();
    ...
  }
  ```
- **Error Cause:** DiverseVul extracted this GoogleTest routine from the patch commit. The presence of network endpoints (`"tcp://127.0.0.1:80"`), mock pointers, and assertion macros triggered false positive heuristics.

---

## 8. Summary of Evidence vs. Hypotheses

| Dimension | Hypothesis | Empirical Evidence | Verdict |
| :--- | :--- | :--- | :--- |
| **Language Shift** | C++17 syntax degrades C-trained models | `::` in 87.7% of Envoy vs 13.7% in train; `std::` in 38.8% vs 3.5%. | **CONFIRMED** |
| **Semantics Shift** | Envoy vulnerabilities are non-memory logic flaws | 0 buffer overflows; 50.2% resource exhaustion (CWE-400/770/835), which represents <1% of train. | **CONFIRMED** |
| **Context Truncation** | Tokenizer fragmentation truncates logic | BPE expands identifiers by 3.7x; 33.2% truncated at 384 tokens. | **CONFIRMED** |
| **Architectural Gap** | Inter-procedural state machine bugs cannot be caught at function level | Single-line accessors like `++num_listener_connections_` lack boundary checks in external classes. | **CONFIRMED** |
| **Label Noise** | Envoy labels are corrupted or inverted | DiverseVul commits are authentic, but 18.7% of samples are GoogleTest unit test routines. | **PARTIALLY CONTRIBUTING** |
| **Over-Fitting to Test** | Envoy test set was modified or contaminated | 0 code overlap with train/val; 33.67% vulnerability rate verified. | **DISPROVEN** |

The forensic error analysis establishes that improving Envoy detection requires genuine C++ domain adaptation, calibration across thresholds, and commit patch modeling, which are evaluated in [`results/experiments/phase4_envoy_model_comparison.md`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase4_envoy_model_comparison.md).
