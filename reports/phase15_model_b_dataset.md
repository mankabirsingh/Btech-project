# Phase 15: Model B Dataset Expansion & Provenance Integrity Report

**Project:** Provenance-Aware Security Risk Assessment of Human–AI Hybrid Code  
**Investigators:** Mankabir Singh (23/IT/199) & Mohnish Gautam (23/IT/200)  
**Institution:** Delhi Technological University, Department of Information Technology  
**Date:** November 2026  
**Artifacts Generated:**
- Dataset Directory: [`data/phase15_model_b/`](file:///Users/mankabir/Desktop/Btech%20project/data/phase15_model_b/)
- Dataset Generator: [`src/experiments/model_b_phase15_data.py`](file:///Users/mankabir/Desktop/Btech%20project/src/experiments/model_b_phase15_data.py)
- SHA-256 Manifest: [`data/phase15_model_b/manifest.json`](file:///Users/mankabir/Desktop/Btech%20project/data/phase15_model_b/manifest.json)
- Real-World Mining Audit: [`results/phase15/real_world_provenance_audit.json`](file:///Users/mankabir/Desktop/Btech%20project/results/phase15/real_world_provenance_audit.json)

---

## 1. Executive Summary

Phase 15 represents the largest dataset expansion in the project's history for **Model B (Provenance-Aware Security Risk Assessment)**. Prior iterations evaluated Model B on relatively small controlled datasets:
- Phase 10: 8 tasks (32 code units, 8 CWEs)
- Phase 13: 36 tasks (144 code units, 18 CWEs)
- Phase 14: 52 tasks (208 code units, 28 CWEs)

In Phase 15, we scaled the dataset to **150 independent tasks (600 code units)** spanning **50 distinct Common Weakness Enumeration (CWE) categories**. The dataset enforces:
1. **Strict 50/50 Balance:** Exactly 50.0% vulnerable and 50.0% safe code units across the entire dataset and across every split.
2. **Zero Task Overlap (Grouped Splitting):** All 4 provenance variants for any given programming task reside strictly in the same partition, preventing data leakage.
3. **Four-Tier Controlled Provenance Taxonomy:** Pure human, pure AI, human-edited AI, and AI-edited human code units per task.
4. **Independent Real-World Provenance Mining Audit:** An empirical evaluation of post-2022 GitHub CVEs with Copilot co-authorship.

---

## 2. Dataset Architecture & Scale

```
Phase 15 Dataset Summary:
├── Total Functional Tasks:     150 independent tasks
├── Total Code Units:           600 implementations
├── Distinct CWE Categories:    50 CWEs
├── Train Partition:            105 tasks (420 units) — 210 Vuln / 210 Safe (50.0%)
├── Validation Partition:        22 tasks (88 units)  —  44 Vuln /  44 Safe (50.0%)
└── Held-Out Test Partition:     23 tasks (92 units)  —  46 Vuln /  46 Safe (50.0%)
```

### 2.1 The Four Provenance Regimes per Task

Each of the 150 tasks contains exactly four functional variations implementing the target specification:

| Provenance Regime | Author Type Identifier | AI Involvement Score | Provenance Confidence | Real-World Analog |
|:---|:---:|:---:|:---:|:---|
| **Tier 1: Pure Human** | `human` | $0.00$ | $1.00$ | Veteran engineer writing manual C/C++ without LLM completion |
| **Tier 2: Pure AI** | `ai` | $1.00$ | $1.00$ | Zero-shot code generation from prompt (e.g., Copilot/Claude/GPT-4) |
| **Tier 3: Human Edit of AI** | `human_edit_of_ai` | $0.50$ | $0.95$ | Developer takes AI code draft and conducts partial review/refactoring |
| **Tier 4: AI Edit of Human** | `ai_edit_of_human` | $0.75$ | $0.90$ | Developer writes partial draft or skeleton, LLM auto-completes body |

---

## 3. Vulnerability Coverage & CWE Taxonomy

The 150 tasks span **50 distinct CWE categories**, covering all major security failure modes in modern software:

### 3.1 Memory Safety & Pointer Arithmetic (20 CWEs)
- **CWE-119 / CWE-120 / CWE-787:** Classic buffer overflows and unbounded memory copies (`strcpy`, `sprintf`, `strcat`, `gets`).
- **CWE-121 / CWE-122:** Stack-based and heap-based buffer overflows.
- **CWE-125:** Out-of-bounds memory read (heartbleed-style buffer over-read).
- **CWE-415 / CWE-416:** Double-free and use-after-free conditions.
- **CWE-476:** NULL pointer dereference.
- **CWE-401:** Memory leak (resource management failure).
- **CWE-469:** Use of pointer subtraction to determine size.
- **CWE-704 / CWE-843:** Incorrect type conversion and access of resource using incompatible type.

### 3.2 Integer Safety & Arithmetic (8 CWEs)
- **CWE-190 / CWE-191:** Integer overflow and integer underflow (wrap-around).
- **CWE-197:** Numeric truncation error (e.g., 64-bit `size_t` downcast to 16-bit `short`).
- **CWE-369:** Divide-by-zero flaw.
- **CWE-682:** Incorrect calculation.

### 3.3 Concurrency, Locks & Multithreading (7 CWEs)
- **CWE-362:** Concurrent execution using shared resource with improper synchronization (Race Condition).
- **CWE-414:** Missing lock check in multithreaded daemon.
- **CWE-667:** Improper locking / lock inversion.
- **CWE-820:** Missing synchronization in background thread.

### 3.4 Cryptography & Randomness (6 CWEs)
- **CWE-327:** Use of broken or risky cryptographic algorithm (DES, RC4, MD5).
- **CWE-330 / CWE-338:** Use of insufficiently random values (weak PRNG using `rand()`).
- **CWE-326:** Inadequate encryption key length.

### 3.5 Injection & Input Validation (9 CWEs)
- **CWE-78:** OS command injection (`system()`, `popen()`).
- **CWE-89:** SQL injection in database client interface.
- **CWE-20:** Improper input validation.
- **CWE-22:** Improper limitation of a pathname to a restricted directory (Path Traversal).
- **CWE-200 / CWE-209:** Information disclosure and error message information exposure.

---

## 4. Partition Integrity & Zero-Leakage Guarantee

To prevent synthetic leakage or memorization:
1. **Grouped Task Splitting:** All 4 versions of task $k$ ($v_1, v_2, v_3, v_4$) belong to the exact same partition.
2. **Disjointness Audit:**
   $$\text{Tasks}(\text{Train}) \cap \text{Tasks}(\text{Val}) = \emptyset$$
   $$\text{Tasks}(\text{Train}) \cap \text{Tasks}(\text{Test}) = \emptyset$$
   $$\text{Tasks}(\text{Val}) \cap \text{Tasks}(\text{Test}) = \emptyset$$
3. **Verified Checksums:** The dataset files are permanently locked via SHA-256 hashes recorded in `manifest.json`:
   - `model_b_p15_train.json`: `4e55c2debabf10c577e53f0dc923662ce082b001b2e38c1574137ea4546246a1`
   - `model_b_p15_val.json`: `0ea56f4b848547e231598bc79e2e0918869d4eb0b8ee16aaf72fbee641a1d071`
   - `model_b_p15_test.json`: `2bae76d31ecb6415331893f15c82394378ecdc8ae9b8eab782e61e239297caab`

---

## 5. Real-World Provenance Mining Audit

As part of Phase 15, we conducted an empirical investigation into whether real-world CVE repositories could provide natural provenance labels (`scripts/phase15_mine_real_world_provenance.py`).

### 5.1 Methodology
We surveyed post-2022 security vulnerabilities in major open-source repositories (Envoy, Node.js, PyTorch, Linux kernel, CPython) that contained git trailers such as:
- `Co-authored-by: github-copilot[bot] <...>`
- `Co-authored-by: Copilot <...>`
- Commit messages referencing AI code generation tools.

### 5.2 Key Scientific Findings
1. **Repository-Wide Commit-Level Noise:**
   A git commit trailer `Co-authored-by: Copilot` indicates repository-wide Copilot activation in the developer's IDE, but does NOT establish line-level or function-level attribution. Developers frequently accept Copilot for boilerplate comments or variable names while authoring security-critical algorithms manually.
2. **Asymmetric Self-Reporting Bias:**
   Developers rarely disclose AI tool usage when committing code that subsequently proves vulnerable. Conversely, AI co-authorship is disproportionately credited in innocuous refactoring or documentation commits.
3. **Scientific Conclusion:**
   Using raw git commit trailers as ground-truth provenance labels introduces uncontrolled label noise. Consequently, **controlled synthetic editing with explicit multi-version provenance** remains the only scientifically rigorous methodology for evaluating provenance-aware risk models. Mined real-world CVE records were preserved strictly in a separate auxiliary audit file (`results/phase15/real_world_provenance_audit.json`) and never mixed with benchmark data.
