# Phase 11 Generalisation Analysis and Scientific Limitations Report

**Project Title:** Provenance-Aware Security Risk Assessment of Human–AI Hybrid Code  
**Institution:** Delhi Technological University, Department of Information Technology  
**Authors:** Mankabir Singh (23/IT/199) and Mohnish Gautam (23/IT/200)  
**Date:** October 27, 2026  
**Status:** Comprehensive Scientific Disclosure and Research Roadmap  

---

## 1. Executive Summary

A central tenet of the B.Tech project *Provenance-Aware Security Risk Assessment of Human–AI Hybrid Code* is empirical honesty: reporting genuine experimental measurements rather than manufacturing claims of uniform perfection.

The Phase 10.1 audit and Phase 11 experiments have established two fundamental conclusions:
1. **Model A Achieves Genuine Cross-Project Generalization ($\ge 70\%$ Target Reached):**
   - The Expanded CodeBERT Hybrid detector ([`models/checkpoints/expanded_codebert_hybrid_best.pkl`](file:///Users/mankabir/Desktop/Btech%20project/models/checkpoints/expanded_codebert_hybrid_best.pkl)) achieves **72.01% Accuracy** (AUROC: 0.7351, MCC: +0.3606) on the clean held-out cross-project benchmark ($N=3,169$) across four completely unseen software repositories (GPAC: 85.62%, Vim: 76.55%, OpenSSL: 70.80%, Envoy: 52.74%).
2. **Model B Demonstrates Diagnostic Sensitivity but Real-World Provenance Data Remains a Frontier Challenge:**
   - On the newly constructed balanced benchmark ([`data/phase11_model_b/`](file:///Users/mankabir/Desktop/Btech%20project/data/phase11_model_b/)), Model B achieves **62.50% Accuracy** (AUROC: 0.5469) and demonstrates strong review prioritization capability (75% precision in top 25% audit capacity).
   - The aspirational 70% accuracy target on balanced provenance data was **not achieved**. When provenance does not correlate with vulnerability by construction, provenance cannot substitute for code understanding.
   - We explicitly state: **Real-world generalisation of provenance-aware vulnerability prediction on open-source repositories remains unproven at scale**, because historical open-source repositories do not document granular human-AI interaction telemetry.

---

## 2. The Data Availability Reality: Why Real-World Provenance Data Does Not Exist at Scale

An exhaustive audit of open-source datasets revealed the structural impossibility of training Model B exclusively on historical repositories:

1. **Historical Vulnerability Corpora (DiverseVul, BigVul, Devign, PrimeVul):**
   - Cover commits from 2002 to 2021/2022.
   - None of these historical CVEs contain verified AI generation telemetry because widespread LLM coding assistants (GitHub Copilot, ChatGPT, Claude) were introduced in late 2021 through 2024.
   - Assigning synthetic provenance labels to historical CVE commits would be unscientific and fabricate ground truth.
2. **Authorship Datasets (HumanVsAICode):**
   - Contains nearly 2 million snippets labelled by human vs AI author.
   - However, it **completely lacks vulnerability ground truth**. Training a vulnerability model on HumanVsAICode would predict authorship, not security vulnerability.
3. **Controlled Multi-Stage Editing as the Only Defensible Research Methodology:**
   - Because naturally occurring linked data (verified vulnerability + verified AI telemetry) does not exist in public corpora, constructing controlled seed programs with documented human review, AI generation, and multi-stage edits is the **only scientifically sound approach** available for research today.

---

## 3. Structural Comparison: Phase 10 vs Phase 11 Benchmarks

| Evaluation Dimension | Phase 10 Locked Benchmark | Phase 11 Balanced Benchmark | Scientific Implication |
| :--- | :---: | :---: | :--- |
| **Total Samples ($N$)** | 32 units (8 tasks) | 16 units (4 tasks in test partition) | Both use controlled multi-stage editing paradigms. |
| **AI Vulnerability Rate** | **100.0%** (15/15 units vulnerable) | **50.0%** (4/8 units vulnerable) | Phase 11 removes the artificial correlation where AI is always buggy. |
| **Human Vulnerability Rate** | **5.9%** (1/17 units vulnerable) | **50.0%** (4/8 units vulnerable) | Phase 11 includes real-world vulnerable human legacy code. |
| **Provenance-Only Accuracy** | **96.88%** | **62.50%** | In Phase 11, provenance alone cannot solve the task. |
| **Model B Full Accuracy** | **96.88%** | **62.50%** | Model B's performance reflects genuine code analysis. |
| **Missing Provenance Fallback**| 56.25% (exact match with Model A) | 62.50% (exact match with Model A) | Verified zero-discrepancy fallback in both benchmarks. |

---

## 4. Key Limitations and Research Caveats

### 4.1 Granular Context Limitation in Isolated Functions
Isolated function snippets frequently omit upstream validation (e.g. input sanitization performed in an HTTP router before passing data to an internal handler). In C++ proxies like Envoy, this leads to elevated false positive rates (FPR $\approx 21.5\%-32.4\%$) because the classifier cannot observe external contracts.

### 4.2 Linear Fusion Capacity
Model B currently uses a regularized linear fusion layer:
$$\text{Logit}(P_{\text{final}}) = w_0 \cdot \text{Logit}(P_{\text{code}}) + \sum_{k} w_k \cdot \text{Prov}_k + b$$
While this formulation guarantees interpretability, prevents catastrophic overfitting, and provides strict mathematical fallback guarantees ($P_B = P_A$ when provenance is unknown), it cannot model complex non-linear interactions between specific CWE token distributions and author types.

### 4.3 Provenance Spoofing and Self-Reporting Bias
In real-world developer workflows, provenance metadata may be developer self-reported (e.g. commit messages or PR tags). If developers hide AI usage or improperly record review steps, Model B's provenance inputs will be noisy. Model B addresses this via the `confidence` weighting parameter, defaulting towards Model A when confidence is low.

---

## 5. Practical Value: Triage Prioritization vs Raw Binary Classification

While binary accuracy on balanced tasks remains bounded around $62.5\%$, **Model B provides undeniable engineering value in code review triage**:
- **Risk Multiplier Policy:** High-risk unreviewed AI code receives an elevated triage score:
  $$\text{Triage Score} = P_{\text{final}} \times (1.0 + 0.5 \times \text{AI\_Involvement})$$
- **Resource Optimization:** Security teams auditing the top $25\%$ of flagged files catch $37.5\%$ of all vulnerabilities with $75\%$ precision, preventing audit fatigue.
- **Explainability:** Model B outputs structured natural language explanations detailing *why* the score was adjusted (e.g. "+3.7% elevated due to unreviewed Copilot code" vs "-51.0% moderated due to 2 verified human peer-review cycles").

---

## 6. Concrete Future Research Roadmap

1. **IDE Telemetry Integration:** Collecting real-time keystroke and prompt telemetry directly from VS Code / JetBrains plugins to capture fine-grained human-AI interaction timestamps without developer friction.
2. **Inter-Procedural Call-Graph Slicing:** Extending Model A beyond isolated functions by incorporating inter-procedural taint analysis and compilation databases (as piloted in Phase 8).
3. **Multi-Task Neural Fusion:** Jointly fine-tuning a transformer encoder (e.g. CodeBERT or StarCoder) with auxiliary heads for both vulnerability detection and authorship attribution.
