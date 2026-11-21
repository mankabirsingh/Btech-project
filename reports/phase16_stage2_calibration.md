# Phase 16 — Stage 2: Experiment 3 Report — Calibration & Threshold Optimisation

**Project:** Provenance-Aware Security Risk Assessment of Human–AI Hybrid Code  
**Investigators:** Mankabir Singh (23/IT/199) & Mohnish Gautam (23/IT/200)  
**Institution:** Delhi Technological University, Department of Information Technology  
**Milestone:** Phase 16 — Stage 2: Experiment 3 (Calibration & Decision Thresholds)  
**Artifacts Generated:**
- Calibration Script: [`scripts/phase16_exp3_calibration_and_thresholding.py`](file:///Users/mankabir/Desktop/Btech%20project/scripts/phase16_exp3_calibration_and_thresholding.py)
- Results Payload: [`results/phase16/exp3_calibration_and_thresholding.json`](file:///Users/mankabir/Desktop/Btech%20project/results/phase16/exp3_calibration_and_thresholding.json)

---

## 1. Motivation & Scientific Integrity Rules

In Stage 1 error diagnostics, we observed that:
- 64.9% of False Negatives fall into the marginal band ($0.25 \le P < 0.50$).
- 90.2% of False Positives fall into the marginal band ($0.50 \le P < 0.75$).

Over 75% of classification errors reside in the boundary corridor. This experiment investigates whether post-hoc probability calibration and decision threshold selection can optimize the trade-off between false alarms and missed flaws.

### Strict Methodological Constraint
In accordance with non-negotiable rules:
- **Zero Test Tuning:** All temperatures, Platt parameters, and optimal thresholds $\tau^*$ were fit strictly on the 3,000-sample validation split (`vulnerability_val_expanded.parquet`).
- **No AUROC Manipulation:** Post-hoc monotonic calibration (Platt/temperature scaling) does not alter rank ordering and therefore does not change AUROC. We evaluate calibration using **Expected Calibration Error (ECE)** and **Brier Score**.

---

## 2. Techniques Evaluated

1. **Validation Threshold Optimisation ($\tau^*$):**
   - Grid search across $\tau \in [0.20, 0.80]$ to maximize F1 score on the validation split.
2. **Temperature Scaling ($T$):**
   - Optimising temperature parameter $T > 0$ via Negative Log Likelihood (NLL) minimization on validation logits:
     $$\hat{P}_{\text{cal}} = \sigma\left(\frac{z}{T}\right)$$
3. **Platt Scaling (Logistic Calibration):**
   - Fitting univariate logistic regression on validation logits:
     $$\hat{P}_{\text{platt}} = \sigma(a \cdot z + b)$$
4. **Combined Calibration + Threshold Selection:**
   - Jointly calibrating and selecting $\tau^*$ on validation data.

---

## 3. Empirical Results on Held-Out Benchmark ($N = 3,169$)

```
========================================================================================
CALIBRATION & THRESHOLD EXPERIMENT RESULTS (Frozen Held-Out Benchmark, N = 3,169)
========================================================================================
Configuration                    Threshold (Tau)  Accuracy   Precision   Recall   FPR      ECE     Brier
----------------------------------------------------------------------------------------
1. Raw Default Model             Tau = 0.500      72.86%     58.0%       51.6%    17.3%    0.0641  0.1865
2. Validation Threshold (Tau*)   Tau* = 0.560     72.99%     60.8%       41.9%    12.6%    0.0641  0.1865
3. Temperature Scaling (T=1.08)  Tau = 0.500      72.86%     58.0%       51.6%    17.3%    0.0644  0.1865
4. Platt Scaling                 Tau = 0.500      69.80%     76.7%       10.5%     1.3%    0.1496  0.2069
5. Combined Platt + Tau*         Tau* = 0.260     73.05%     60.8%       41.8%    12.5%    0.1496  0.2069
========================================================================================
```

---

## 4. Scientific Discussion & Strategic Insights

1. **The Validation Threshold Effect ($\tau^* = 0.560$):**
   - The validation set has a 21.0% vulnerability base rate (vs ~31.7% in the test set). Because the validation set has fewer vulnerabilities, tuning $\tau^*$ on validation F1 naturally favors a slightly more conservative threshold ($\tau^* = 0.560$).
   - On the held-out test set, applying $\tau^* = 0.560$ **slashes the false positive rate from 17.27% to 12.60% (a 27.0% relative drop in false alarms)** while lifting overall accuracy to **72.99%**.
   - However, this comes at the expense of sensitivity: recall falls from 51.55% to 41.87%.
2. **Low Raw Temperature Drift ($T = 1.079$):**
   - The optimal temperature on validation is $T = 1.079$, very close to $1.000$. The raw expected calibration error on test data is already exceptionally low ($ECE = 0.0641$, or 6.4%). This confirms that CodeBERT logits scaled by logistic regression are already well-calibrated.
3. **Platt Scaling Distribution Shift:**
   - Fitting unregularized Platt scaling on validation logits pushed probabilities aggressively toward zero because the validation base rate is lower. Without threshold adjustment, Platt scaling caused severe under-prediction (recall dropped to 10.5%).
   - Re-adjusting the threshold to $\tau^* = 0.260$ restored balance (73.05% Acc, 12.5% FPR).
4. **Recommendation:**
   - In production CI/CD deployments where **developer trust and low false positive rate** are paramount, using $\tau^* = 0.560$ achieves a highly desirable **12.6% FPR** with 60.8% precision.
   - For security auditing where catching every vulnerability is the priority, the default calibrated threshold $\tau = 0.500$ remains optimal.
