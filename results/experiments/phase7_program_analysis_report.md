# Phase 7 Program Analysis Report: Control Flow, Guard Dominance & Dynamic Dispatch

**Project:** Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code  
**Institution:** Delhi Technological University  
**Date:** October 27, 2026  
**Artifact Path:** [`results/experiments/phase7_program_analysis_report.md`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase7_program_analysis_report.md)  
**Baseline Audit:** [`results/experiments/phase7_baseline_audit.md`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase7_baseline_audit.md)  
**Compiler Feasibility:** [`results/experiments/phase7_compiler_feasibility.md`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase7_compiler_feasibility.md)  

---

## 1. Executive Summary & Architecture

Phase 7 establishes a reproducible, scientifically defensible static program analysis pipeline tailored to extracted function-level codebases. Given that whole-program compiler bitcode generation is infeasible without full repository build systems, this analysis implements:
1. Lexical and syntactic call-site classification.
2. Control-Flow Graph (CFG) basic-block decomposition.
3. Guard dominance analysis across branch conditions.
4. Explicit relationship categorization that separates statically inferred edges from unresolved dynamic dispatches.

---

## 2. Formal Categorization of Program Relationships

To prevent false confidence and avoid treating unobserved context as safety or vulnerability, every relationship extracted by our analysis pipeline is partitioned into one of four formal categories:

```
                            [Program Relationship Taxonomy]
                            
   +-------------------------------------------------------------------------+
   | 1. Compiler-Verified: Exact token and syntax confirmed by Clang parser   |
   +-------------------------------------------------------------------------+
   | 2. Statically Inferred: Intra-commit caller/callee AST edges             |
   +-------------------------------------------------------------------------+
   | 3. Dynamic Virtual Dispatch: Abstract interface calls (ptr->method())    |
   +-------------------------------------------------------------------------+
   | 4. Unresolved Context: Callee body located in external, unmodified file  |
   +-------------------------------------------------------------------------+
```

1. **Compiler-Verified Relationships:** Tokens, syntax structures, and keywords validated directly by Clang's lexical front-end preprocessor (`clang -Xclang -dump-tokens`).
2. **Statically Inferred Relationships:** Intra-commit caller/callee invocation edges where the calling function and called function are both present within the commit snapshot.
3. **Dynamic Virtual Interface Dispatches:** Polymorphic calls through class pointers or references (`ptr->method(...)`), where the concrete receiver depends on dynamic dispatch tables (vtables) and cannot be resolved statically without link-time devirtualization.
4. **Unresolved Relationships:** Function calls whose names are syntactically parsed, but whose function bodies reside in external, un-modified compilation units outside the commit slice.

> [!IMPORTANT]
> **Soundness Rule:** The absence of a detected caller guard check must **never** be equated with proof that no guard exists. If a relationship is *Unresolved* or *Dynamically Dispatched*, the analysis assigns `context_resolved = 0` to preserve uncertainty rather than penalizing the function.

---

## 3. Empirical Analysis of Envoy Benchmark ($N = 603$)

We ran the program analysis engine across all 603 functions in the locked Envoy benchmark:

| Program Analysis Dimension | Count / Prevalence | Percentage of Total | Security & False-Positive Implication |
| :--- | :---: | :---: | :--- |
| **Total Call Sites Analyzed** | **7,075 call sites** | 100.0% | Comprehensive coverage of all invocation sites across 603 functions. |
| **Direct Call Sites (Statically Inferred)** | **5,867 call sites** | **82.93%** | Function invocations with direct syntactic name bindings. |
| **Dynamic Virtual Dispatches (`ptr->method`)** | **1,208 call sites** | **17.07%** | Polymorphic filter/connection calls (`filter->decodeHeaders()`). Unresolvable statically without Whole-Program Devirtualization. |
| **Local Guard Checks (`if (...)`)** | **181 functions** | **30.02%** | Only 30.0% of Envoy functions check local bounds; 70.0% rely on upstream caller contracts! |
| **Functions with RAII Smart Pointers** | **90 functions** | **14.93%** | Utilize `std::unique_ptr`, `Buffer::Instance`, or `ScopeGuard` for memory safety. |
| **GoogleTest Assertions / Mock Calls** | **276 functions** | **45.77%** | Unit test fixtures lacking runtime defensive error codes by design. |

---

## 4. Key Program Analysis Insights

### 4.1 The Virtual Method Blindness Problem
In modern C++ network proxies like Envoy, data processing is organized as a pipeline of pluggable filters:
```cpp
// Common Envoy pattern
Http::FilterHeadersStatus FilterManager::decodeHeaders(...) {
  return active_filter_->decodeHeaders(headers, end_stream);
}
```
- `active_filter_` is an abstract interface pointer of type `StreamDecoderFilter*`.
- At static analysis time, `active_filter_->decodeHeaders` could dispatch to any of dozens of concrete filter classes (`CorsFilter`, `RateLimitFilter`, `Router::Filter`).
- Without link-time whole-program points-to analysis, a static detector cannot determine whether the callee enforces a rate limit, drops malformed packets, or closes the connection.
- In our taxonomy, this is classified as **Dynamic Virtual Dispatch (17.07% of all calls)** and must be treated as unresolved rather than insecure.

### 4.2 The 70% Guard Absence Phenomenon
Only **30.02% (181 / 603)** of Envoy functions contain local `if (...)` statements. In a distributed network proxy, functions are designed with strict separation of concerns:
- Low-level codecs parse wire frames.
- Stream managers track buffer capacities.
- Worker threads execute event loops.
- Individual worker methods expect callers to validate state invariants before calling them.

When an isolated function detector inspects one of the **70% of functions that lack local guards**, it mistakenly infers a "missing security check" (CWE-400), creating the 214–287 false positives documented in Phase 5 and Phase 6.
