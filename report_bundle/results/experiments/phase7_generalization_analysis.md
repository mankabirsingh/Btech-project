# Phase 7 Generalization Analysis: Why Validation Performance Does Not Transfer to Envoy

**Project:** Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code  
**Institution:** Delhi Technological University  
**Date:** October 27, 2026  
**Artifact Path:** [`results/experiments/phase7_generalization_analysis.md`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase7_generalization_analysis.md)  
**Baseline Audit:** [`results/experiments/phase7_baseline_audit.md`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase7_baseline_audit.md)  
**Compiler Feasibility:** [`results/experiments/phase7_compiler_feasibility.md`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase7_compiler_feasibility.md)  
**Program Analysis:** [`results/experiments/phase7_program_analysis_report.md`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase7_program_analysis_report.md)  

---

## 1. Executive Summary

A central paradox in our experimental findings across Phases 5 and 6 is the stark divergence between validation and test performance:
- On **Repository-Separated Validation Commit Pairs**, our models achieve **93.40% pairwise ranking accuracy** and **0.7641 AUROC**.
- On **Held-Out Test Commit Pairs**, models achieve **91.20% pairwise ranking accuracy**.
- Yet, when evaluated on **Isolated Envoy Functions ($N = 603$)**, accuracy drops to **46.10%** and single-function AUROC falls near chance to **0.4452**.

This report investigates the root causes of this generalization gap. We separate **confirmed empirical findings** from **working hypotheses** across eight software engineering and machine learning dimensions.

---

## 2. Confirmed Empirical Findings vs. Hypotheses

```
                     [Validation vs. Envoy Domain Shift]
                     
  Dimension               Validation Corpus (BigVul)       Envoy Benchmark
  ---------------------------------------------------------------------------------
  Primary Language        Procedural C (90%+)             Modern C++17 / Object-Oriented
  Top Vulnerability       CWE-20 (Validation), CWE-119    CWE-400 (Resource Exhaustion)
  Test Fixture Ratio      1.0% (13 / 1,338 records)       46.1% (278 / 603 functions)
  Guard Presence          High (Local branch checks)       Low (Only 30.0% have local 'if')
  Dynamic Dispatches      Rare (<2% function pointers)     Prevalent (17.1% virtual calls)
  Deployment Context      Paired Diffs (Before vs. After) Isolated Single-Function Snapshots
```

---

### Confirmed Finding 1: Severe Vulnerability Category Shift (CWE-119 vs. CWE-400)
- **Validation Corpus (BigVul):** Dominated by classic procedural C buffer handling flaws:
  - **CWE-20** (Improper Input Validation): 241 samples.
  - **CWE-119** (Memory Buffer Bounds / Overflow): 180 samples.
  - **CWE-200** (Information Disclosure): 101 samples.
  - **CWE-189** (Numeric / Integer Overflow): 69 samples.
  *Detection Signal:* Lexical tokens like `memcpy`, `strcpy`, buffer indexing `buf[i]`, and integer casts.
- **Envoy Benchmark:** Dominated by architectural and stateful network flaws:
  - **CWE-400** (Uncontrolled Resource Consumption / Flood Exhaustion): **233 / 603 (38.6%)**.
  - **CWE-22** (Path Traversal via HTTP/2 URL decoding): 62 samples.
  - **CWE-476** (NULL Pointer Dereference in asynchronous callbacks): 52 samples.
  - **CWE-416** (Use-After-Free across event loops): 48 samples.
  *Detection Failure:* CWE-400 vulnerabilities do not contain unsafe C library calls. They stem from missing stream quotas, unthrottled RST_STREAM packets, or unbounded buffering, which are lexically invisible in a single function body.

---

### Confirmed Finding 2: Massive Test-Fixture Contamination (1.0% vs. 46.1%)
- **Validation Corpus:** Only **1.0% (13 / 1,338)** of validation functions are unit test cases.
- **Envoy Benchmark:** **46.1% (278 / 603)** of all benchmark functions are GoogleTest fixtures (`TEST_P`, `TEST_F`, `ASSERT_`, `EXPECT_`).
- **Mechanism of Failure:** When DiverseVul mined Envoy's security commits, it scraped both the implementation functions and the accompanying unit tests added by Envoy engineers to reproduce the vulnerability. Test fixtures intentionally allocate malformed frames and mock sockets without defensive error returns. A model trained on 99% production code misidentifies these test assertions as unguarded production vulnerabilities.

---

### Confirmed Finding 3: The Evaluation Paradigm Gap (Paired Diffs vs. Isolated Functions)
- **Validation Evaluation:** Evaluates comparative commit pairs: $f_{before}$ vs. $f_{after}$.
  - The model only needs to determine which of two nearly identical functions is *more* vulnerable.
  - Invariant code cancels out, allowing diff-aware models to hit **93.40% ranking accuracy**.
- **Envoy Benchmark:** Evaluates isolated function snapshots against an absolute classification threshold ($\tau = 0.50$).
  - The model has no comparative reference point.
  - Without a comparative diff, the detector must infer safety from raw source tokens alone.

---

### Confirmed Finding 4: Inverted Guard Responsibility (70% Guard Absence)
- In the validation corpus, vulnerable functions frequently contain the flawed check (e.g. `if (len > 1024)` instead of `if (len > 512)`).
- In Envoy, **70.0% of functions (422 / 603) contain zero local `if (...)` statements**.
- As demonstrated in our program analysis, Envoy enforces preconditions in upstream caller state machines (e.g., `ActiveStream::maybeContinueDecoding`) or delegates validation to downstream codec helpers. An isolated function detector sees no local `if` and triggers false alarms on 53.5% of benign code.

---

### Confirmed Finding 5: Virtual Dispatch & Modern C++ RAII
- **Dynamic Dispatches:** In Envoy, **1,208 call sites (17.1%)** invoke abstract interface methods through virtual pointers (`ptr->method()`). Static token embeddings cannot resolve the runtime receiver, leaving the detector blind to whether the callee enforces a rate limit or handles an exception.
- **RAII Lifecycles:** 92.8% of benign Envoy functions are memory-safe via `std::unique_ptr`, `Buffer::Instance`, and `ScopeGuard`. Detectors trained on C expect raw `free()` calls, misinterpreting smart pointer ownership as resource leakage.

---

## 3. Methodological Implications for Phase 7

1. **Do Not Tune on Envoy:** Tuning hyperparameters or thresholds against Envoy's 278 test fixtures would overfit to dataset curation artifacts rather than solving general vulnerability detection.
2. **Context Parity:** If a model relies on patch diffs or inter-commit call graphs during training, evaluating it on single isolated functions without comparative context constitutes an evaluation mismatch.
3. **Explicit Uncertainty Handling:** Functions with unresolved dynamic dispatches or external callers must be assigned explicit uncertainty scores rather than forcing high-confidence binary predictions.
