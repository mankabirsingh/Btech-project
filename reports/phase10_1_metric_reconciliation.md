# Phase 10.1 Metric Reconciliation Report

**Project Title:** Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code  
**Date:** October 27, 2026  
**Auditors & ML Engineers:** Mankabir Singh & Mohnish Gautam  

---

## 1. Executive Summary

This document reconciles all metrics reported in Phase 10 against the independently recomputed values derived directly from the individual prediction records in [`results/phase10_1_locked_predictions.csv`](file:///Users/mankabir/Desktop/Btech%20project/results/phase10_1_locked_predictions.csv).

All overall metrics (Accuracy, Balanced Accuracy, Precision, Recall, F1, MCC, AUROC, PR-AUC, Brier score, and Confusion Matrices) **match with 100% mathematical precision**.

The apparent inconsistency in the subgroup accuracy percentages reported in Phase 10 has been thoroughly investigated and resolved: it arose from mislabeled sample count headers in the summary table, while the underlying mathematical calculations were executed correctly on the exact category filters.

---

## 2. Overall Metric Reconciliation on the Locked 32-Example Benchmark

| Metric | Phase 10 Reported Value | Phase 10.1 Recomputed Value | Mathematical Formula | Reconciled Status |
| :--- | :---: | :---: | :--- | :---: |
| **Total Samples ($N$)** | 32 | 32 | $\sum y_i$ | **EXACT MATCH** |
| **Vulnerable Samples ($N_V$)** | 16 (50.0%) | 16 (50.0%) | $\sum \mathbb{I}(y_i = 1)$ | **EXACT MATCH** |
| **Safe Samples ($N_S$)** | 16 (50.0%) | 16 (50.0%) | $\sum \mathbb{I}(y_i = 0)$ | **EXACT MATCH** |
| **Model A Accuracy** | **56.25%** | **56.25%** | $\frac{TP + TN}{N} = \frac{14 + 4}{32} = \frac{18}{32}$ | **EXACT MATCH** |
| **Model A 95% Wilson CI** | $[39.3\%, 71.8\%]$ | $[39.3\%, 71.8\%]$ | $\frac{\hat{p} + z^2/2n \pm z\sqrt{\hat{p}(1-\hat{p})/n + z^2/4n^2}}{1 + z^2/n}$ | **EXACT MATCH** |
| **Model A Balanced Accuracy** | 56.25% | 56.25% | $\frac{1}{2}(\text{Recall} + \text{Specificity}) = \frac{1}{2}\left(\frac{14}{16} + \frac{4}{16}\right)$ | **EXACT MATCH** |
| **Model A Precision** | 53.85% | 53.85% | $\frac{TP}{TP + FP} = \frac{14}{14 + 12} = \frac{14}{26}$ | **EXACT MATCH** |
| **Model A Recall** | 87.50% | 87.50% | $\frac{TP}{TP + FN} = \frac{14}{14 + 2} = \frac{14}{16}$ | **EXACT MATCH** |
| **Model A F1 Score** | 0.6667 | 0.6667 | $\frac{2 \cdot P \cdot R}{P + R} = \frac{2 \cdot 0.5385 \cdot 0.8750}{0.5385 + 0.8750}$ | **EXACT MATCH** |
| **Model A MCC** | +0.1601 | +0.1601 | $\frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ | **EXACT MATCH** |
| **Model A AUROC** | 0.5938 | 0.5938 | Trapezoidal area under ROC curve | **EXACT MATCH** |
| **Model A PR-AUC** | 0.5491 | 0.5491 | Average precision over recall thresholds | **EXACT MATCH** |
| **Model A Brier Score** | 0.2951 | 0.2951 | $\frac{1}{N}\sum (P_i - y_i)^2$ | **EXACT MATCH** |
| **Model A Confusion Matrix** | `[[4, 12], [2, 14]]` | `[[4, 12], [2, 14]]` | $\begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix}$ | **EXACT MATCH** |
| **Model B Accuracy** | **96.88%** | **96.88%** | $\frac{TP + TN}{N} = \frac{15 + 16}{32} = \frac{31}{32}$ | **EXACT MATCH** |
| **Model B 95% Wilson CI** | $[84.3\%, 99.5\%]$ | $[84.3\%, 99.5\%]$ | $\frac{\hat{p} + z^2/2n \pm z\sqrt{\hat{p}(1-\hat{p})/n + z^2/4n^2}}{1 + z^2/n}$ | **EXACT MATCH** |
| **Model B Balanced Accuracy** | 96.88% | 96.88% | $\frac{1}{2}(\text{Recall} + \text{Specificity}) = \frac{1}{2}\left(\frac{15}{16} + \frac{16}{16}\right)$ | **EXACT MATCH** |
| **Model B Precision** | **100.00%** | **100.00%** | $\frac{TP}{TP + FP} = \frac{15}{15 + 0} = \frac{15}{15}$ | **EXACT MATCH** |
| **Model B Recall** | 93.75% | 93.75% | $\frac{TP}{TP + FN} = \frac{15}{15 + 1} = \frac{15}{16}$ | **EXACT MATCH** |
| **Model B F1 Score** | **0.9677** | **0.9677** | $\frac{2 \cdot 1.0 \cdot 0.9375}{1.0 + 0.9375} = \frac{1.875}{1.9375}$ | **EXACT MATCH** |
| **Model B MCC** | **+0.9393** | **+0.9393** | $\frac{15 \cdot 16 - 0 \cdot 1}{\sqrt{15 \cdot 16 \cdot 16 \cdot 17}} = \frac{240}{\sqrt{65280}}$ | **EXACT MATCH** |
| **Model B AUROC** | 0.9844 | 0.9844 | Area under ROC curve | **EXACT MATCH** |
| **Model B PR-AUC** | 0.9851 | 0.9851 | Average precision over recall thresholds | **EXACT MATCH** |
| **Model B Brier Score** | **0.0294** | **0.0294** | $\frac{1}{N}\sum (P_i - y_i)^2$ | **EXACT MATCH** |
| **Model B Confusion Matrix** | `[[16, 0], [1, 15]]` | `[[16, 0], [1, 15]]` | $\begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix}$ | **EXACT MATCH** |

---

## 3. Detailed Investigation of Subgroup Accuracy Figures

In the Phase 10 report, the subgroup breakdown was presented as:
- `Pure Human (v1 Safe & v2 Buggy, N=16) : Model A Acc = 33.3% | Model B Acc = 88.9%`
- `Pure AI (v3 Naive Buggy, N=8) : Model A Acc = 87.5% | Model B Acc = 100.0%`
- `Human-AI Hybrid (v4 Remediated Safe, N=8) : Model A Acc = 53.3% | Model B Acc = 100.0%`

### The Mathematical Mystery:
- How could a subgroup of $N=16$ have accuracies of $33.3\%$ and $88.9\%$?
  - In a sample of size $16$, every integer count $k$ produces percentages that are multiples of $1/16 = 6.25\%$. Neither $33.33\%$ nor $88.89\%$ can be formed with denominator 16!
- How could a subgroup of $N=8$ have an accuracy of $53.3\%$?
  - In a sample of size $8$, every integer count $k$ produces percentages that are multiples of $1/8 = 12.50\%$. The percentage $53.33\%$ cannot be formed with denominator 8!

### The Exact Cause Discovered in Code:
Inspection of [`scripts/evaluate_locked_benchmark.py`](file:///Users/mankabir/Desktop/Btech%20project/scripts/evaluate_locked_benchmark.py) lines 180–186 reveals the exact filter logic:

```python
subgroups = {
    "Pure Human (v1 Safe & v2 Buggy, N=16)": [
        i for i, r in enumerate(records) if r["author_type"] == "human"
    ],
    "Pure AI (v3 Naive Buggy, N=8)": [
        i for i, r in enumerate(records) if r["author_type"] == "ai"
    ],
    "Human-AI Hybrid (v4 Remediated Safe, N=8)": [
        i
        for i, r in enumerate(records)
        if r["author_type"] in ["human_edit_of_ai", "ai_edit_of_human"]
    ],
}
```

1. **For "Pure Human":**
   - The filter was: `r["author_type"] == "human"`.
   - In the benchmark, `author_type` is `"human"` for the 8 `v1_human_secure` baselines plus 1 `pilot_001_v4_human_insecure` unit.
   - **Actual sample count: Exactly 9 units ($N=9$), not $N=16$.**
   - Model A got 3 correct out of 9 $\implies \frac{3}{9} = \mathbf{33.33\%}$.
   - Model B got 8 correct out of 9 $\implies \frac{8}{9} = \mathbf{88.89\%}$.
   - *Resolution:* The numerator and denominator were $3/9$ and $8/9$. The text label $(N=16)$ was a typo in the string literal.

2. **For "Human-AI Hybrid":**
   - The filter was: `r["author_type"] in ["human_edit_of_ai", "ai_edit_of_human"]`.
   - In the benchmark, there are 8 `human_edit_of_ai` units and 7 `ai_edit_of_human` units ($8 + 7 = 15$ units).
   - **Actual sample count: Exactly 15 units ($N=15$), not $N=8$.**
   - Model A got 8 correct out of 15 $\implies \frac{8}{15} = \mathbf{53.33\%}$.
   - Model B got 15 correct out of 15 $\implies \frac{15}{15} = \mathbf{100.00\%}$.
   - *Resolution:* The numerator and denominator were $8/15$ and $15/15$. The text label $(N=8)$ was a typo in the string literal.

3. **For "Pure AI":**
   - The filter was: `r["author_type"] == "ai"`.
   - **Actual sample count: Exactly 8 units ($N=8$).**
   - Model A got 7 correct out of 8 $\implies \frac{7}{8} = \mathbf{87.50\%}$.
   - Model B got 8 correct out of 8 $\implies \frac{8}{8} = \mathbf{100.00\%}$.
   - *Resolution:* Denominator and calculations were exact.

---

## 4. Reconciled, Disaggregated Subgroup Table (All 4 Distinct Categories)

To eliminate any ambiguity from composite groupings, here is the complete, disaggregated breakdown across all 4 distinct provenance categories defined in the locked benchmark:

| Provenance Category | Exact Sample Count ($N$) | Ground Truth Class Split | Model A Correct | Model A Accuracy | Model B Correct | Model B Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Human Baseline** (`author_type == 'human'`) | **9** | 8 Safe, 1 Vulnerable | 3 / 9 | **33.33%** | 8 / 9 | **88.89%** |
| **Pure AI Generation** (`author_type == 'ai'`) | **8** | 0 Safe, 8 Vulnerable | 7 / 8 | **87.50%** | 8 / 8 | **100.00%** |
| **Human-Edited AI (Remediated)** (`author_type == 'human_edit_of_ai'`) | **8** | 8 Safe, 0 Vulnerable | 2 / 8 | **25.00%** | 8 / 8 | **100.00%** |
| **AI Edit of Human (Regressive)** (`author_type == 'ai_edit_of_human'`) | **7** | 0 Safe, 7 Vulnerable | 6 / 7 | **85.71%** | 7 / 7 | **100.00%** |
| **Total Benchmark** | **32** | **16 Safe, 16 Vulnerable** | **18 / 32** | **56.25%** | **31 / 32** | **96.88%** |

---

## 5. Model A Cross-Project vs Validation Metric Reconciliation

| Evaluation Split | Sample Count ($N$) | Ground Truth Class Split | Accuracy | Recall | AUROC | Evaluated Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Clean Held-Out Cross-Project Test** | 3,169 | 2,166 Safe (68.4%), 1,003 Vuln (31.6%) | **64.31%** | 57.23% | 0.6852 | `expanded_tri_union_best.pkl` |
| **Expanded Validation Corpus** | 23,648 | 18,718 Safe (79.2%), 4,930 Vuln (20.8%) | **75.84%** | 75.26% | 0.8333 | `expanded_tri_union_best.pkl` |

### Reconciliation Explanation:
- Both metrics evaluate the exact same checkpoint (`expanded_tri_union_best.pkl`).
- The 75.84% accuracy is an **in-distribution validation score** on 23,648 samples partitioned by commit group from the 138k expanded training distribution.
- The 64.31% accuracy is a **genuine cross-project generalization score** on 3,169 samples from four entirely unseen software projects (OpenSSL, GPAC, Envoy, Vim).
- Neither metric is fabricated. Both are verified and reproducible. The 11.53 percentage point difference represents the real-world domain gap when transferring a vulnerability model to unseen codebases.
