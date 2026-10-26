# Provenance-Aware Security Risk Assessment & Review Prioritization Report

**B.Tech Project-I (Mid-Semester Evaluation Milestone)**  
**Department of Information Technology, Delhi Technological University (DTU)**  
**Students:** Mankabir Singh (23/IT/199), Mohnish Gautam (23/IT/200)  
**Date:** October 27, 2026  

---

## 1. Research Question & Hypothesis

### Primary Research Question
> *"Can provenance-aware analysis improve security-risk prediction and review prioritisation for human-AI hybrid code compared with provenance-agnostic vulnerability detection?"*

### Hypotheses Evaluated
1. **Hypothesis 1 (Binary Classification on Homogeneous Data):** In legacy open-source vulnerability datasets (e.g., Devign, PrimeVul), author provenance is homogeneous ($100\%$ human-written, $\text{AI Involvement} = 0.0$). Adding provenance features will yield **zero or negligible empirical improvement** because there is no author variance.
2. **Hypothesis 2 (Triage & Review Prioritization on Hybrid Code):** In human-AI hybrid workflows where code modifications originate from mixed human and generative AI sources, **provenance awareness significantly improves review prioritization** (higher Recall@K under restricted audit budgets) because raw AI generations exhibit higher empirical defect frequency than human peer-reviewed code.
3. **Hypothesis 3 (Nuance & Inductive Bias):** Relying on provenance as a deterministic vulnerability signal causes false positives on remediated human-AI hybrid code. Provenance must serve as a **prioritization weight**, guarded by contextual static and data-flow analysis.

---

## 2. Experimental Setup & Methodology

### 2.1 Provenance Feature Formulation (Zero Label Leakage)
In our Phase 1 audit, we discovered that earlier prototype iterations inadvertently included target security indicators (`has_cwe`) within the feature extraction pipeline, causing synthetic label leakage. 

We eliminated this flaw completely. Provenance features are strictly restricted to verifiable origin and structural metadata:
- **$\text{AI Involvement} \in [0.0, 1.0]$**: Documented proportion of AI generation vs. human modification.
- **Author Category One-Hot**: `[is_human, is_ai, is_human_edit_of_ai, is_ai_edit_of_human]`.
- **Code Length & Volume Proxies**: Normalized $\ln(1 + \text{chars})$ and $\ln(1 + \text{lines})$.
- **Comment Density**: $\frac{\text{comment tokens}}{\text{total characters}}$, capturing distinct AI documentation density without leaking security target labels.

### 2.2 Model Architectures Compared
- **Model A (Code-Only Baseline):** Contextual CodeBERT representations ($\mathbb{R}^{768}$) or token $n$-gram representations trained strictly on code text without author provenance.
- **Model B (Provenance-Aware Classifier):** Multimodal fusion combining code representations with the 10-dimensional provenance vector $\mathbf{p} \in \mathbb{R}^{10}$.
- **Prioritization Scoring Function:**
  $$S(c, \mathbf{p}) = P(\text{vuln} \mid c) \times \left(1 + \alpha \cdot \text{ai\_involvement}\right)$$
  ranking candidate pull requests for human security audit.

---

## 3. Experiment 1: Real-World Open-Source Benchmark (Devign + PrimeVul)

We evaluated Model A vs. Model B across the corrected, grouped-split test set of **5,412 unseen C/C++ functions** (2,634 vulnerable, 2,778 benign) from Devign and PrimeVul.

### Empirical Results Table

| Model | Accuracy | Precision | Recall | F1 Score | MCC | AUROC | PR-AUC (AP) | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Code-Only)** | 49.70% | 49.03% | 51.67% | 50.31% | -0.0046 | 0.5095 | 0.4932 | 0.2503 |
| **Model B (Code + Provenance)** | 50.39% | 49.68% | 52.81% | 51.19% | +0.0094 | 0.5129 | 0.4965 | 0.2498 |
| **Delta ($\Delta$)** | **+0.69%** | **+0.65%** | **+1.14%** | **+0.88%** | **+0.0140** | **+0.0034** | **+0.0033** | **-0.0005** |

### Scientific Finding
**Provenance features do not meaningfully improve vulnerability prediction on legacy open-source datasets.**  
Because open-source projects (QEMU, Linux, FFmpeg) curated prior to 2021 are authored entirely by human developers ($\text{ai\_involvement} = 0.0$ across all 36,080 samples), provenance features provide zero discriminant variance. Claiming that provenance "solves" general open-source vulnerability detection is empirically invalid.

---

## 4. Experiment 2: Multi-CWE Controlled Evolution Benchmark (32 Units)

To rigorously evaluate human-AI hybrid code, we constructed an 8-task, 32-version controlled benchmark covering 8 distinct Common Weakness Enumerations (CWEs):
1. **Task 1 (CWE-120):** Buffer Overflow (`strcpy`/`gets` vs. `strncpy`/`fgets`)
2. **Task 2 (CWE-134):** Format String (`printf(user_str)` vs. static format literal)
3. **Task 3 (CWE-190):** Integer Overflow to Buffer Allocation (`count * sizeof(int)` with `__builtin_mul_overflow`)
4. **Task 4 (CWE-78):** OS Command Injection (`system()` vs. `execve` whitelist)
5. **Task 5 (CWE-416):** Use-After-Free (dangling pointer access vs. immediate NULL-zeroing)
6. **Task 6 (CWE-22):** Path Traversal (`realpath` validation vs. naive single-pass replacement)
7. **Task 7 (CWE-327):** Broken Cryptography (DES/ECB mode vs. AES-256-GCM authenticated cipher)
8. **Task 8 (CWE-476):** NULL Pointer Dereference (unconditional lookup dereference vs. NULL check)

Each task tracks a 4-stage evolutionary lineage:
- **v1: Human Secure Baseline** ($\text{AI} = 0\%$, $\text{Vuln} = 0$)
- **v2: AI-Generated Naive Implementation** ($\text{AI} = 100\%$, $\text{Vuln} = 1$)
- **v3: Human-Remediated Hybrid** ($\text{AI} = 50\%$, $\text{Vuln} = 0$)
- **v4: AI-Refactored Regressive / Incomplete Fix** ($\text{AI} = 75\%$, $\text{Vuln} = 1$)

Total: 32 code units (16 vulnerable, 16 benign, perfectly balanced).

### Binary Classification Performance

| Model | Accuracy | Precision | Recall | F1 Score | MCC | AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (CodeBERT-Only)** | 59.38% | 60.00% | 56.25% | 58.06% | +0.1879 | 0.5664 |
| **Model B (CodeBERT + Provenance)** | **84.38%** | **82.35%** | **87.50%** | **84.85%** | **+0.6999** | **0.9375** |

---

## 5. Review Prioritization Performance (Audit Budget Simulation)

In enterprise DevSecOps workflows, security teams cannot review 100% of pull requests and must triage based on predicted risk. We simulated review budgets ranging from 12.5% to 50.0% of all submitted PRs:

| Review Budget (% Inspected) | Code-Only Recall | Code-Only Precision | Provenance-Aware Recall | Provenance-Aware Precision | Static Analysis Recall | Static Analysis Precision |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **12.5% (Top 4 PRs)** | 12.5% (2/16) | 50.0% | **25.0% (4/16)** | **100.0%** | 25.0% (4/16) | 100.0% |
| **25.0% (Top 8 PRs)** | 25.0% (4/16) | 50.0% | **50.0% (8/16)** | **100.0%** | 43.8% (7/16) | 87.5% |
| **37.5% (Top 12 PRs)** | 43.8% (7/16) | 58.3% | **75.0% (12/16)** | **100.0%** | 50.0% (8/16) | 66.7% |
| **50.0% (Top 16 PRs)** | 56.2% (9/16) | 56.2% | **87.5% (14/16)** | **87.5%** | 56.2% (9/16) | 56.2% |

### Key Findings
1. **2x Recall at 25% Audit Budget:** With budget to audit only 8 of 32 pull requests, provenance awareness catches **8 out of 16 vulnerabilities (50.0% recall)** with **100% precision**, compared to only 4 out of 16 (25.0% recall) for provenance-agnostic models.
2. **Superiority over Static Analysis Alone:** At 37.5% budget, Provenance-Aware Prioritization discovers **12 of 16 vulnerabilities (75% recall)**, outperforming raw static analysis (8 of 16, 50% recall) because static analysis rules miss incomplete regex bypasses and subtle logic flaws present in `v4`.
3. **The Inductive Bias Caveat:** Because `v3` code is hybrid ($\text{AI}=50\%$) but secure, a model that relies exclusively on provenance without verifying AST/static bounds would produce false positives on human-remediated code.

---

## 6. Generated Artifacts & Visualizations

The following verified visualizations were generated and saved to `results/plots/`:
- `results/plots/review_prioritization_comparison.png`: Recall curves across review budgets comparing Provenance-Aware, Static Analysis, and CodeBERT-Only prioritization against random triage.
- `results/plots/controlled_benchmark_lineage.png`: Mean static findings, risk score, actual vulnerability rate, and model probabilities across evolutionary stages ($v_1 \to v_2 \to v_3 \to v_4$).
- `results/reports/controlled_benchmark_32.json`: Full machine-readable lineage tracking, Flawfinder findings, and prediction probabilities.
- `results/reports/provenance_experiment_results.json`: Complete comparative metrics for Experiment 1 and Experiment 2.
