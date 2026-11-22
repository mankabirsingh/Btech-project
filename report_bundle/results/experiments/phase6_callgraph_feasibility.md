# Phase 6 Call-Graph Feasibility & Inter-Commit Slicing Audit

**Project:** Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code  
**Institution:** Delhi Technological University  
**Date:** October 27, 2026  
**Artifact Path:** [`results/experiments/phase6_callgraph_feasibility.md`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase6_callgraph_feasibility.md)  
**Baseline Audit:** [`results/experiments/phase6_baseline_audit.md`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase6_baseline_audit.md)  

---

## 1. Executive Summary

Phase 6 investigates whether **inter-commit call-graph slicing** can resolve the severe false-positive penalty observed in Phase 5 (71.75% false-positive rate on isolated Envoy functions). In Phase 5, the model was overly sensitive to missing guard checks and raw resource-allocation patterns because it evaluated each function snapshot in isolation.

Before implementing context-aware models, this audit:
1. Inspects source repositories, commit structures, and metadata across BigVul and the held-out benchmarks.
2. Identifies the exact availability of multi-function commit co-location and call relationships.
3. Defines the strongest defensible form of call-graph slicing supported by actual repository artifacts.
4. Documents explicit boundaries and a formal fallback mechanism to avoid data fabrication.

---

## 2. Dataset Commit Structure & Multi-Function Availability

We conducted an empirical audit of the commit topologies in BigVul (`bstee615/bigvul`) and the held-out test partitions:

| Partition / Corpus | Total Samples | Unique Commits | Commits with Multiple Functions | % Multi-Function Commits | Inter-Function Relationship Availability |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **BigVul Entire Corpus** | 150,908 | 4,005 | 3,707 | **92.56%** | High: callers, callees, and helpers co-located in same commit. |
| **Phase 5 Train Split** | 1,800 pairs | 1,800 | 1,642 | **91.22%** | High: co-modified siblings and intra-commit call edges present. |
| **Phase 5 Val Split** | 500 pairs | 500 | 470 | **94.00%** | High: 22,087 records span these 500 commits in BigVul. |
| **Held-Out Test Pairs** | 250 pairs | 250 | 231 | **92.40%** | High: caller/callee context verifiable across commit siblings. |
| **Envoy Locked Benchmark** | 603 | 25 | 23 | **92.00%** | Very High: 100% of functions belong to 25 large commit groups (up to 93 funcs/commit). |

### Key Architectural Discovery:
In function-level datasets, vulnerabilities are typically curated as isolated function strings. However, by querying the underlying `commit_id`, **over 92% of vulnerability-fixing patches modify multiple functions within the same commit**. This authentic co-location makes intra-commit call-graph slicing feasible without fabricating external repository state.

---

## 3. The Inter-Commit Call-Graph Slicing Architecture

We implement an inter-commit call-graph slicing engine that reconstructs the local call graph and extracts boundary conditions across five semantic dimensions:

```
                          [Intra-Commit Directed Graph G = (V, E)]
                          
        [Caller C1 (Guard Checked)]            [Caller C2 (Unchecked)]
             |                                              |
             | (Edge e1: invokes f)                         | (Edge e2: invokes f)
             v                                              v
      +------------------------------------------------------------+
      |                 Target Function f (Under Review)           |
      | - Local Allocations: alloc_words, sockets, locks           |
      | - Modern RAII: std::unique_ptr, Buffer::Instance           |
      | - Local Guards: if (len > MAX), if (!ptr)                  |
      +------------------------------------------------------------+
             |                                              |
             | (Edge e3: delegates validation)              | (Edge e4: delegates cleanup)
             v                                              v
      [Callee K1: validateHeader()]                 [Callee K2: safe_free()]
```

### 3.1 Five Slicing Dimensions Supported by Real Data:
1. **Intra-Commit Call Graph Construction:**
   For any commit with function set $V = \{f_1, f_2, \dots, f_k\}$, we parse function signatures and identifier call sites to build directed edges $E = \{(u, v) \mid u \text{ invokes } v\}$.
2. **Upstream Caller Guard Dominance:**
   If a function $f$ lacks local boundary or null checks, we slice backwards to its intra-commit callers $C = \{u \in V \mid (u, f) \in E\}$. If callers contain preconditions (e.g., `if (size > MAX_BUF) return;`, `if (state != OPEN) return;`), the check is **dominated upstream**. Function $f$ is an internal helper whose preconditions are satisfied, not an unguarded vulnerability.
3. **Downstream Callee Delegation:**
   If function $f$ invokes callees $K = \{v \in V \mid (f, v) \in E\}$ whose semantic prefixes indicate validation (`validate_`, `check_`, `verify_`, `assert_`, `handleCodecError`) or resource cleanup (`clean_`, `free_`, `destroy_`, `reset_`, `close_`), validation and cleanup are **delegated downstream**.
4. **Resource Lifecycle Tracking & Modern C++ RAII:**
   C++ network proxies like Envoy manage resources via **Resource Acquisition Is Initialization (RAII)** using `std::unique_ptr`, `std::shared_ptr`, `Buffer::Instance`, and `ScopeGuard`. Flagging an allocation as a resource leak simply because `free()` is absent is a C-centric flaw. We explicitly model smart pointer ownership and lifecycle balance:
   $$\text{ResourceBalanced} = (\text{alloc} == 0) \lor (\text{free} \ge \text{alloc}) \lor \text{uses\_raii} \lor \text{delegated\_cleanup}$$
5. **Test Fixture & Assertion Discrimination:**
   Unit tests (`TEST`, `TEST_F`, `TEST_P`, `ASSERT_`, `EXPECT_`) deliberately invoke error conditions without defensive returns. In Envoy, 276 of 603 functions (45.7%) are test fixtures. Slicing explicitly flags test harnesses so the classifier does not mistake test setup code for missing runtime checks.

---

## 4. Legitimate Boundaries & Documented Fallback

### 4.1 What Is Missing / Cannot Be Fabricated
1. **Global Inter-File Call Graphs:** Unmodified external source files (files not touched by the commit) are not stored in function-level vulnerability datasets. Inferring external caller edges by name matching across arbitrary external libraries would be unscientific and introduce hallucinated edges.
2. **Whole-Program Compiler ASTs:** Compiling each historical commit with its build system (e.g. Bazel for Envoy) is outside the boundaries of static function corpora.

### 4.2 Documented Fallback & Uncertainty Preservation
When a function has no intra-commit callers ($|C| = 0$):
- We do **not** assume caller checks are missing.
- We mark `caller_context = UNRESOLVED` and set `has_upstream = 0`.
- The model receives an explicit context availability flag (`ctx_available`), allowing the classifier to preserve uncertainty rather than predicting vulnerability based on the absence of unobserved context.
