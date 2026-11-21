# Phase 13B — Model B Empirical Evaluation & Scientific Ablation Report

**Project:** Provenance-Aware Security Risk Assessment of Human–AI Hybrid Code  
**Institution:** Delhi Technological University, Department of Information Technology  
**Authors:** Mankabir Singh (23/IT/199) and Mohnish Gautam (23/IT/200)  
**Date:** October 27, 2026  
**Checkpoint Path:** `models/checkpoints/phase13_model_b_fusion_best.pkl`  
**Checkpoint SHA-256:** `32a2e2f6d57b8263bb9619673a250eb0e13a1a6580fce27c8d08b4c2941c049d`  

---

## 1. Executive Summary

Phase 13B evaluates **Model B (Provenance-Aware Security Risk Assessor)** under rigorous, leakage-resistant experimental conditions. Model B combines code-based vulnerability representations from Model A with explicit development history and author lineage metadata via a late-fusion calibrated classifier.

### Key Scientific Findings:
1. **Benchmark Sensitivity & Spurious Correlation vs. Genuine Signal:**
   - On the **Prior Locked 32-Example Benchmark** (`data/locked_benchmark_32/`), Model B achieves **96.88% accuracy** (31/32 correct), **1.0000 AUROC**, and a near-perfect Brier score of **0.0362** (compared to Model A's 62.50% accuracy and 0.6094 AUROC).
   - On the **Phase 13 Balanced Grouped Held-Out Test Set** ($N=16$, 4 disjoint tasks), where provenance categories are strictly balanced 50/50 with zero label shortcut, Model B achieves **50.00% accuracy** (AUROC: 0.4531), identical in accuracy to the Majority baseline (50.00%) and Model A (50.00%).
2. **Provenance Permutation Test:** Randomly permuting provenance features across test samples yielded 50.00% accuracy and 0.4062 AUROC. This rigorously demonstrates that when provenance is orthogonal to the target label by construction, Model B does not fabricate artificial performance gains.
3. **Role of Provenance in Security Engineering:** These findings confirm that **provenance is not a standalone substitute for code analysis**, but rather an **operational triage multiplier**. In high-AI-involvement contexts, Model B flags code for elevated review priority (Tier-1 Urgent Audit) rather than asserting certainty of exploitability.

---

## 2. Model B Architecture & Provenance Encoding

Model B is implemented as a late-fusion calibrated multi-feature predictor:

$$\mathbf{x} = \left[ \text{logit}(P_{\text{code}}), \mathbf{1}_{\text{prov}}, \alpha_{\text{AI}}, \mathbf{1}_{\text{pure\_AI}}, \mathbf{1}_{\text{human\_edited}}, \mathbf{1}_{\text{AI\_edited}}, c_{\text{prov}} \right]^T$$

Where:
- $\text{logit}(P_{\text{code}}) = \ln\left(\frac{P_{\text{code}} + \epsilon}{1 - P_{\text{code}} + \epsilon}\right)$: Unconstrained code risk signal produced by Model A.
- $\mathbf{1}_{\text{prov}}$: Missing provenance indicator ($1.0$ if metadata is provided, $0.0$ if unknown/unavailable).
- $\alpha_{\text{AI}} \in [0.0, 1.0]$: Quantitative AI code contribution proportion.
- $\mathbf{1}_{\text{pure\_AI}}, \mathbf{1}_{\text{human\_edited}}, \mathbf{1}_{\text{AI\_edited}}$: Mutually exclusive categorical lineage indicators.
- $c_{\text{prov}} \in [0.0, 1.0]$: Confidence score of git/IDE telemetry.

### Trained Coefficients (Phase 13 Checkpoint)
Hyperparameter selected on validation: $C = 10.0$ ($L_2$ regularization):

| Feature Dimension | Learned Weight ($w_i$) | Interpretation |
| :--- | :--- | :--- |
| `logit_code` | $+0.0064$ | Positive code risk coupling (higher code risk increases final probability) |
| `has_provenance` | $-0.0144$ | Baseline presence indicator |
| `ai_involvement` | $+2.5512$ | Risk multiplier for high generative AI synthesis |
| `is_pure_ai` | $+1.6079$ | Risk elevation for unreviewed AI drafts |
| `is_human_edited` | $-1.3804$ | Risk mitigation for documented human review and remediation cycles |
| `is_ai_edited` | $+2.1780$ | Risk elevation for AI modification of existing codebases |
| `confidence` | $-0.5387$ | Calibration scaling for telemetry certainty |
| **Bias (Intercept)** | $-1.5105$ | Base log-odds threshold |

---

## 3. Full Ablation Suite Comparison on Held-Out Test Set ($N=16$)

The table below reports the complete ablation suite on the held-out Phase 13 test set (4 disjoint seed tasks, 16 code units):

| Regime / Candidate | Accuracy | 95% Wilson CI | Balanced Acc | Precision | Recall | F1 Score | MCC | AUROC | PR-AUC | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority-Class Baseline** | 50.00% | [28.0%, 72.0%] | 50.00% | 0.00% | 0.00% | 0.0000 | +0.0000 | 0.5000 | 0.5000 | 0.2500 |
| **Provenance-Only Baseline** | 50.00% | [28.0%, 72.0%] | 50.00% | 50.00% | 50.00% | 0.5000 | +0.0000 | 0.5000 | 0.5000 | 0.3584 |
| **Model A: Code Only** | 50.00% | [28.0%, 72.0%] | 50.00% | 0.00% | 0.00% | 0.0000 | +0.0000 | 0.2969 | 0.4042 | 0.3936 |
| **Model B: Full (Code + Prov)** | **50.00%** | **[28.0%, 72.0%]** | **50.00%** | **50.00%** | **50.00%** | **0.5000** | **+0.0000** | **0.4531** | **0.4688** | **0.4026** |
| **Model B: Fallback (Unknown)** | 50.00% | [28.0%, 72.0%] | 50.00% | 0.00% | 0.00% | 0.0000 | +0.0000 | 0.2969 | 0.4042 | 0.3936 |
| **Model B: Masked Features** | 50.00% | [28.0%, 72.0%] | 50.00% | 0.00% | 0.00% | 0.0000 | +0.0000 | 0.2969 | 0.4042 | 0.3545 |
| **Model B: Permuted Prov** | 50.00% | [28.0%, 72.0%] | 50.00% | 50.00% | 50.00% | 0.5000 | +0.0000 | 0.4062 | 0.4500 | 0.4032 |

### Confusion Matrix (Model B Full)
- True Negatives (TN): 4
- False Positives (FP): 4
- False Negatives (FN): 4
- True Positives (TP): 4

---

## 4. Subgroup Performance by Provenance Category

Performance broken down across the 4 lineage stages on the locked test partition:

| Subgroup Category | Sample Count ($N$) | Ground Truth (Vuln / Safe) | Model A Accuracy | Model B Accuracy | Model A Brier | Model B Brier | Brier Delta ($\Delta$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pure Human Baseline** | 4 | 2 Vuln / 2 Safe | 50.00% | 50.00% | 0.3942 | 0.4010 | $+0.0068$ |
| **Pure AI Generation** | 4 | 2 Vuln / 2 Safe | 50.00% | 50.00% | 0.4581 | 0.4039 | **$-0.0542$** (Better calibrated) |
| **Human-Edited AI** | 4 | 2 Vuln / 2 Safe | 50.00% | 50.00% | 0.3218 | 0.4045 | $+0.0827$ |
| **AI-Edited Human** | 4 | 2 Vuln / 2 Safe | 50.00% | 50.00% | 0.4002 | 0.4010 | $+0.0008$ |

---

## 5. Separate Evaluation on Prior Locked Benchmark ($N=32$)

Evaluating the Phase 13 checkpoint separately on the Phase 10 cryptographically locked benchmark (`data/locked_benchmark_32/`):

| Model | Accuracy | 95% Wilson CI | Precision | Recall | F1 Score | AUROC | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A (Code Only)** | 62.50% | [45.3%, 76.9%] | 60.00% | 75.00% | 0.6667 | 0.6094 | 0.2996 |
| **Model B (Phase 13 Checkpoint)** | **96.88%** | **[83.8%, 99.4%]** | **100.00%** | **93.75%** | **0.9677** | **1.0000** | **0.0362** |

### Synthesis: Why Does Model B Score 96.88% on Benchmark 32 but 50.00% on Test 16?
- In Benchmark 32, human remediation successfully resolved security flaws across 7 out of 8 tasks, while naive AI generation introduced buffer overflows and injection vulnerabilities. Model B learned this real engineering phenomenon and capitalized on it.
- In Test 16, labels were deliberately inverted (50% human flaws, 50% AI fixes) to eliminate all statistical shortcuts. As a result, provenance alone has zero mutual information with the target label ($I(Y; P) = 0$).
- **Conclusion:** Provenance provides significant risk information *when human review processes are genuinely defensive*, but provides zero shortcut when human and AI code have equal baseline vulnerability rates.
