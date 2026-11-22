# Phase 5 Data Feasibility & Inter-Procedural Audit

**Project:** Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code  
**Institution:** Delhi Technological University  
**Date:** October 27, 2026  
**Artifact Path:** [`results/experiments/phase5_data_feasibility.md`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase5_data_feasibility.md)  

---

## 1. Executive Summary & Objective

Phase 5 investigates whether **patch-level representations** (added/deleted lines from unified commit diffs) augmented with **lightweight inter-procedural context** (callees, resource allocations, guards, limit checks) can resolve the function-level blindness observed on complex architectural vulnerabilities such as Envoy's HTTP/2 state machine flaws.

Before designing models or executing experiments, this audit inspects the existing repository, dataset schemas, commit metadata, and available graph structures to determine **what data legitimately exists versus what is missing**, ensuring complete research integrity without data fabrication.

---

## 2. Dataset Inventory & Availability Audit

We conducted an exhaustive audit of all data sources in the local workspace and remote mirrors:

| Data Source | Total Records | Vulnerable Records | Format & Storage | Available Fields | Missing / Unavailable Fields | Feasibility for Inter-Procedural Modeling |
| :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| **BigVul (`bstee615/bigvul`)** | **150,908** | **8,714** (8,700 unique valid pairs) | Cached Hugging Face Dataset | `func_before`, `func_after`, `commit_id`, `project`, `CWE ID`, `commit_message`, `codeLink`, `CVE ID` | Entire repository `.git` tree, cross-file caller source bodies, global call graphs | **FEASIBLE:** Supports exact unified diffs (`difflib`), callee extraction, intra-commit multi-function co-location, and patch semantics. |
| **CVEFixes** | — | — | Not present in local storage | — | Entire database not downloaded locally | **UNAVAILABLE:** Excluded from local training; BigVul is used as the authentic commit pair source. |
| **DiverseVul** | 330,492 | 18,945 | `vulnerability_train_expanded.parquet` ($N=90,600$), `vulnerability_test_heldout_projects_clean.parquet` ($N=3,169$) | Single function `code`, `vuln_label`, `project`, `commit_id`, `cwe` | Separate `func_before`/`func_after` fields (only isolated function bodies), unified diff strings | **PARTIALLY FEASIBLE:** 20 commits in Envoy contain paired $Y=1$ and $Y=0$ functions; all others are standalone function extractions. |
| **PrimeVul** | 24,000 | 3,100 | `vulnerability_train.parquet`, `vulnerability_data.parquet` | `code`, `vuln_label`, `project`, `commit_id`, `group_id` | Diff representations, caller/callee metadata | **LIMITED:** Paired commits exist via `group_id`, but functions are single-body snapshots. |
| **Devign** | 27,258 | 12,484 | `vulnerability_data.parquet` | `code`, `vuln_label`, `project`, `commit_id` | Diff representations, CWE IDs, caller contexts | **UNUSABLE FOR PATCHES:** Devign contains single functions without paired post-fix code. |

---

## 3. Legitimate Inter-Procedural Context vs. Missing Information

### 3.1 What Is Legitimately Available in the Data
1. **Authentic Function-Level Code Pairs:** BigVul provides 8,700 valid pairs ($N = 17,400$ functions) where $\text{func\_before} \neq \text{func\_after}$ with verified commit hashes across 276 projects.
2. **Exact Unified Diffs:** Using Python's standard `difflib.unified_diff`, we extract:
   - Added lines (`+`): Security guards, resource limits, buffer capacity checks, cleanup calls.
   - Deleted lines (`-`): Vulnerable logic, unsafe function calls, missing error branches.
3. **Direct Callee Identification:** Static AST / regex extraction across C/C++ tokens extracts all function calls made *inside* the target function (e.g., `gss_init_sec_context()`, `make_pair()`, `Buffer::Instance`).
4. **Resource & State Management Operations:** Explicit detection of memory/resource allocators (`malloc`, `new`, `open`, `acquire`), releases (`free`, `delete`, `close`, `release`), bounds checks, and exception branches within the function body and diff context.
5. **Intra-Commit Multi-Function Co-Location:** For commits that modify multiple functions, functions sharing the same `commit_id` can be analyzed together as a co-modified group.

### 3.2 What Is Missing / Cannot Be Fabricated
1. **Global Inter-Procedural Call Graphs (External Callers):** The datasets do **not** contain the surrounding unmodified source files of the target repositories. Callers located in external files (e.g., an HTTP/2 connection manager calling a stream decoder) cannot be resolved without cloning and compiling every repository at that exact historical commit SHA.
2. **Inter-Procedural Data-Flow Across Compilation Units:** Cross-translation-unit taint tracking is unavailable because header files, build configurations, and third-party libraries are not stored in function-level vulnerability corpora.
3. **Commit Messages & CVE Descriptions as Features:** While present in BigVul, commit messages and CVE descriptions (e.g. *"Fix buffer overflow in gss_get_mic"*) are **strictly excluded** from model inputs to prevent severe label leakage.

---

## 4. Methodological Scope & Feasible Implementation Plan

Based on this empirical audit, we define the valid, scientifically defensible scope for Phase 5:

```
                            [Valid Phase 5 Architecture]

    +--------------------------------------------------------------------------+
    |                        Target Function (Before / After)                  |
    +--------------------------------------------------------------------------+
                                         |
                                         v
         +-------------------------------+-------------------------------+
         |                               |                               |
         v                               v                               v
  [1. Diff-Aware Engine]     [2. Callee Context Engine]    [3. State & Resource Engine]
  - Added lines (+)          - Extracted direct callees   - Resource operations (alloc/free)
  - Deleted lines (-)        - Co-modified intra-commit   - Guard checks (if (!ptr) return)
  - Invariant context lines    functions                  - Limit checks (len > MAX)
         |                               |                               |
         +-------------------------------+-------------------------------+
                                         |
                                         v
                         [Fused Patch-Context Transformer]
                         - Pairwise Ranking: P(vuln) > P(fix)
                         - Binary Vulnerability Classification
```

### Key Commitments:
1. **No Data Fabrication:** We will not simulate artificial call graphs or synthesize fake external caller functions.
2. **Context Boundedness:** Lightweight context will consist of verified direct callees, detected resource lifecycles, and added/deleted guard logic.
3. **Split Integrity:** Splits are partitioned strictly by `commit_id` and `project` to prevent paired-commit leakage.
4. **Evaluation Independence:** The locked test benchmarks (Clean 4-Projects, Envoy $N=603$, Fresh 5-Projects) remain strictly isolated for final evaluation.

This establishes a solid, reproducible foundation for building the patch-level models in Phase 2 and 3.
