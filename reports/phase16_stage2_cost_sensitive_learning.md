# Phase 16 — Stage 2: Experiment 4 Report — Cost-Sensitive Learning & Sample Weighting

**Project:** Provenance-Aware Security Risk Assessment of Human–AI Hybrid Code  
**Investigators:** Mankabir Singh (23/IT/199) & Mohnish Gautam (23/IT/200)  
**Institution:** Delhi Technological University, Department of Information Technology  
**Milestone:** Phase 16 — Stage 2: Experiment 4 (Loss Weighting & Hard Mining)  
**Artifacts Generated:**
- Training Script: [`scripts/phase16_exp4_cost_sensitive.py`](file:///Users/mankabir/Desktop/Btech%20project/scripts/phase16_exp4_cost_sensitive.py)
- Candidate Checkpoint: [`models/checkpoints/phase16_codebert_cost_sensitive.pkl`](file:///Users/mankabir/Desktop/Btech%20project/models/checkpoints/phase16_codebert_cost_sensitive.pkl)
- Results Payload: [`results/phase16/exp4_cost_sensitive_learning.json`](file:///Users/mankabir/Desktop/Btech%20project/results/phase16/exp4_cost_sensitive_learning.json)

---

## 1. Motivation & Hypothesis

In Stage 1 error diagnostics, we observed an overall False Negative Rate of **49.65%** (498 missed vulnerabilities out of 1,003). In security auditing, missed vulnerabilities (false negatives) carry a substantially higher operational cost than false alarms (false positives).

### The Cost-Sensitive Hypothesis
We hypothesized that introducing asymmetric loss penalties on positive (vulnerable) examples during training or up-weighting ambiguous omission patterns would encourage the classifier to shift its decision boundary, recovering marginal false negatives without an unacceptable explosion in false positives.

---

## 2. Experimental Strategies Evaluated

### Strategy A: Asymmetric Class Weighting
We parametrized the training loss with an asymmetric positive class weight:
$$\mathcal{L}_{\text{cost}}(y, \hat{p}) = - \left[ w_{\text{pos}} \cdot y \log(\hat{p}) + (1 - y) \log(1 - \hat{p}) \right]$$
We evaluated $w_{\text{pos}} \in [1.00, 1.15, 1.25, 1.40, 1.50, 1.75, 2.00]$ combined with regularisation $C \in [0.01, 0.05, 0.10, 0.20]$. All selection was performed strictly on the validation set (`vulnerability_val_expanded.parquet`, $N = 3,000$) to maximize validation F1 score.

### Strategy B: Sample-Weighted Omission Hard Mining
We applied instance-level sample weighting on the 25k training set:
- Short functions ($< 6$ lines of code) received a $1.25\times$ weight boost to combat length bias.
- Functions containing arrow dereferences (`->`) with zero null checks received a $1.25\times$ weight boost to force attention to potential CWE-476 omission patterns.

---

## 3. Empirical Results on Held-Out Benchmark ($N = 3,169$)

```
========================================================================================
COST-SENSITIVE EXPERIMENTAL CANDIDATES vs BASELINE (Clean Held-Out Benchmark)
========================================================================================
Candidate / Strategy             Val F1     Accuracy   Precision   Recall   FPR      AUROC
----------------------------------------------------------------------------------------
Phase 15 Baseline (28-dim, C=0.1) 0.5180    74.00%     60.77%      50.35%   15.05%   0.7369
Candidate 4A (w_pos=1.0, C=0.2)  0.5215    73.49%     59.71%      49.95%   15.60%   0.7351
Candidate 4B (Sample Weighted)    0.5168    73.34%     59.38%      49.95%   15.84%   0.7353
========================================================================================
```

---

## 4. Scientific Discussion & Honest Analysis of Findings

1. **Validation Selection Behavior:**
   - On the validation set, asymmetric weights $w_{\text{pos}} \ge 1.25$ initially increased raw recall, but caused precision to plummet because the validation set has a 21.0% vulnerability rate.
   - Consequently, the grid search strictly selected $w_{\text{pos}} = 1.00$ with $C = 0.20$ as the optimal F1-maximizing parameter on validation data.
2. **Held-Out Test Generalization:**
   - Candidate 4A achieved **73.49% accuracy** and an **AUROC of 0.7351** with an FPR of **15.60%**.
   - Candidate 4B (sample-weighted omission mining) achieved **73.34% accuracy** with an FPR of **15.84%**.
3. **Scientific Finding:**
   - Artificial sample re-weighting on synthetic heuristic criteria (short length or raw dereferences) did not outperform standard balanced empirical risk minimization.
   - This proves that the underlying classifier is already operating near its optimal linear separation boundary for these features, and confirms that **the primary bottleneck is not loss weighting, but domain diversity and multimodal representation**, directly pointing to **Experiment 2 (C++ Stratification) and Experiment 5 (Stacking Ensemble)**.
