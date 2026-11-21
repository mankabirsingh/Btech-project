# Phase 16: Model A Comprehensive Error Analysis and Improvement Roadmap

**Project:** Provenance-Aware Security Risk Assessment of Human–AI Hybrid Code  
**Investigators:** Mankabir Singh (23/IT/199) & Mohnish Gautam (23/IT/200)  
**Institution:** Delhi Technological University, Department of Information Technology  
**Evaluation Phase:** Phase 16 — Stage 1 Error Diagnosis  
**Artifacts Generated:**
- Diagnostic Script: [`scripts/phase16_model_a_error_diagnosis.py`](file:///Users/mankabir/Desktop/Btech%20project/scripts/phase16_model_a_error_diagnosis.py)
- Diagnostic Data: [`results/phase16/model_a_error_diagnosis.json`](file:///Users/mankabir/Desktop/Btech%20project/results/phase16/model_a_error_diagnosis.json)

---

## 1. Executive Summary

This error analysis examines the predictions of the Phase 15 Champion Model A (`phase15_codebert_cxx_hybrid_best.pkl`) on the clean, frozen, held-out benchmark ($N = 3,169$).

```
========================================================================================
OVERALL CONFUSION MATRIX & ACCURACY SUMMARY (N = 3,169)
========================================================================================
Metric                            Value          Notes
----------------------------------------------------------------------------------------
Overall Accuracy                  74.00%         2,345 correct / 3,169 total
True Negatives (TN)               1,840          Specificity = 84.95%
True Positives (TP)                 505          Sensitivity / Recall = 50.35%
False Positives (FP)                326          False Positive Rate (FPR) = 15.05%
False Negatives (FN)                498          False Negative Rate (FNR) = 49.65%
Precision                         60.77%         Positive Predictive Value
F1 Score                          0.5507         Harmonic mean of Precision and Recall
AUROC                             0.7369         Ranking ability across all thresholds
Brier Score                       0.1837         Probability calibration error
========================================================================================
```

### Key Diagnostic Discoveries
1. **Errors Cluster Closely at the Decision Boundary:**
   - **64.9% of False Negatives (323 / 498)** are *marginal* ($0.25 \le P < 0.50$).
   - **90.2% of False Positives (294 / 326)** are *marginal* ($0.50 \le P < 0.75$).
   - Over **75% of all classification errors** are close calls near the 0.50 threshold rather than confident mistakes, confirming that the representation space is informative but the decision boundary and calibration require refinement.
2. **Omission Flaws Dominate False Negatives:**
   - The top two missed vulnerability classes are **CWE-400 (Resource Exhaustion, 74.7% missed)** and **CWE-476 (NULL Pointer Dereference, 79.8% missed)**.
   - These vulnerabilities are characterized by what is *missing* (absent null checks, missing loop termination guards, unbounded buffer read limits) rather than explicit dangerous function calls (`strcpy`, `gets`). CodeBERT embeddings perceive these functions as normal idiomatic code.
3. **Severe Training Domain Shift:**
   - The 138,412-sample training pool (`vulnerability_train_expanded.parquet`) is overwhelmingly dominated by legacy C code (QEMU: 14.4k, FFmpeg: 8.0k, Linux kernel: 5.0k).
   - In the Phase 15 25k balanced training split, **Envoy had exactly 1 sample**, and modern C++ projects represented under 6% of the data. Envoy's 64.01% test performance was achieved via zero-shot cross-project transfer!
4. **Length and Complexity Bias:**
   - Missed vulnerabilities (FNs) are substantially shorter functions ($\text{mean} = 3.3$ log-lines vs $5.1$ in TPs) with minimal pointer arithmetic ($\text{mean} = 1.18$ vs $8.25$ in TPs). The model over-relies on code length and pointer arithmetic density as risk indicators.

---

## 2. Project-Wise Error Breakdown

The clean held-out benchmark contains 3,169 code units across four projects:

```
Project       N Samples    Accuracy    AUROC      TN      FP      FN      TP      FPR       FNR
------------------------------------------------------------------------------------------------
Envoy Proxy     603        64.01%      0.5411    344      63     154      42     15.8%     75.9%
GPAC            793        86.38%      0.8103    644      49      59      41      7.1%     59.6%
OpenSSL       1,257        69.29%      0.7479    505     171     215     366     25.3%     37.0%
Vim             516        78.10%      0.7339    353      43      70      50     10.9%     58.3%
------------------------------------------------------------------------------------------------
Total         3,169        74.00%      0.7369  1,840     326     498     505     15.05%    49.65%
```

### 2.1 OpenSSL: The Dual Bottleneck
- **Disproportionate Error Share:** OpenSSL represents **43.2% of all false negatives** (215 FNs) and **52.5% of all false positives** (171 FPs).
- **Vulnerability Density:** OpenSSL has a 46.2% vulnerability rate in the test set (581 / 1,257).
- **Core Error Mechanics:** 
  - Cryptographic protocol functions (CWE-310) often involve multi-step state machines (`tls_process_cke_rsa`, `dtls1_process_buffered_record`).
  - Safe OpenSSL functions often use complex pointer arithmetic and nested macros, which the structural extractor mistakes for vulnerability patterns ($\text{FPR} = 25.3\%$).

### 2.2 Envoy Proxy: Production vs Test Mock Disentanglement
```
Envoy Subcohort       N Samples    Accuracy    AUROC      TN      FP      FN      TP      FPR      FNR
------------------------------------------------------------------------------------------------------
Production C++ Code     322        63.04%      0.5870    159      51      68      44     24.3%    60.7%
Test Scaffolding        281        65.12%      0.4936    185      12      86      18      6.1%    94.5%
```
- **Test Scaffolding (GoogleTest Mocks):** FPR was successfully reduced to **6.1%** (12 FPs). However, test mock vulnerability labels in the dataset are difficult to rank ($FNR = 94.5\%$, AUROC = 0.4936) because mock tests rarely contain exploitable flaws.
- **Production Code:** Accuracy reached **63.04%** with AUROC = **0.5870**. The 68 missed vulnerabilities (FNs) predominantly involve asynchronous callbacks and dispatcher event loops where object lifetimes exceed function boundaries.

---

## 3. CWE-Wise Vulnerability Miss Analysis

Auditing the 498 False Negatives by Common Weakness Enumeration reveals that vulnerabilities are heavily concentrated in specific failure modes:

```
========================================================================================
TOP 10 CWES IN MISSED VULNERABILITIES (FALSE NEGATIVES)
========================================================================================
CWE ID      CWE Name                        Missed (FN)  Actual Vulns  Miss Rate (%)
----------------------------------------------------------------------------------------
CWE-400     Uncontrolled Resource Consumption    68           91           74.7%
CWE-476     NULL Pointer Dereference             67           84           79.8%
CWE-310     Cryptographic Issues                 46          163           28.2%
CWE-416     Use After Free                       30           58           51.7%
CWE-125     Out-of-bounds Read                   17           31           54.8%
CWE-399     Resource Management Errors           16           36           44.4%
CWE-122     Heap-based Buffer Overflow           15           26           57.7%
CWE-119     Improper Restriction of Ops          15           36           41.7%
CWE-787     Out-of-bounds Write                  14           32           43.8%
CWE-835     Loop with Unreachable Exit           10           11           90.9%
========================================================================================
```

### The "Omission Flaw" Phenomenon
1. **CWE-476 (NULL Pointer Dereference - 79.8% missed):** In C and C++, dereferencing an unchecked pointer (`ptr->field`) looks completely benign to CodeBERT unless the model knows whether `ptr` can be null. Without an explicit check detection feature, the model assigns low risk ($P \approx 0.15 - 0.35$).
2. **CWE-400 & CWE-835 (Resource Exhaustion & Infinite Loops - 74.7% & 90.9% missed):** These functions lack an upper loop bound check or memory allocation limit. The code contains only standard loops (`while (p) { ... }`), which standard lexical representations treat as safe.

---

## 4. Structural Feature Differentials

Comparing the mean structural feature values reveals the model's inductive biases:

```
========================================================================================
TRUE POSITIVES vs FALSE NEGATIVES (What did missed vulnerabilities lack?)
========================================================================================
Feature Name            TP Mean        FN Mean        Delta (TP - FN)   Interpretation
----------------------------------------------------------------------------------------
ptr_arith_count         8.255          1.177          +7.079            Model relies on raw pointer math
log1p_lines             5.077          3.336          +1.742            FNs are shorter functions
log1p_chars             8.439          6.698          +1.741            Short code gets classified safe
max_depth               3.580          2.010          +1.570            FNs have shallow nesting depth
scopes_count            1.844          2.263          -0.419            FNs have more C++ scopes (::)
========================================================================================
```

```
========================================================================================
TRUE NEGATIVES vs FALSE POSITIVES (What triggered false alarms on safe code?)
========================================================================================
Feature Name            TN Mean        FP Mean        Delta (FP - TN)   Interpretation
----------------------------------------------------------------------------------------
ptr_arith_count         0.536          4.856          +4.319            Safe pointer math triggers FPs
log1p_chars             5.974          7.766          +1.791            Long safe functions trigger FPs
log1p_lines             2.729          4.353          +1.624            Multi-line boilerplate triggers FPs
max_depth               1.703          3.064          +1.361            Deep nesting triggers FPs
scopes_count            1.147          1.951          +0.804            Complex C++ namespaces trigger FPs
========================================================================================
```

---

## 5. Threshold & Calibration Sensitivity

Evaluating varying decision thresholds $\tau \in [0.30, 0.65]$ on the held-out benchmark:

```
Threshold (Tau)    Accuracy    F1 Score    Precision    Recall (Sens)    FPR (1 - Spec)
----------------------------------------------------------------------------------------
Tau = 0.30         62.35%      0.5625      44.5%        76.5%            44.2%
Tau = 0.35         67.40%      0.5799      49.0%        71.1%            34.3%
Tau = 0.40         70.78%      0.5836      53.2%        64.7%            26.4%
Tau = 0.45         72.96%      0.5751      57.2%        57.8%            20.0%
Tau = 0.50 (Ref)   74.00%      0.5507      60.8%        50.3%            15.1%
Tau = 0.55         74.44%      0.5126      64.6%        42.5%            10.8%
Tau = 0.60         73.84%      0.4535      66.9%        34.3%             7.8%
----------------------------------------------------------------------------------------
```

**Observation:** At $\tau = 0.45$, F1 improves to **0.5751** and Recall rises to **57.8%** while maintaining **72.96% accuracy** and a **20.0% FPR**. At $\tau = 0.55$, accuracy peaks at **74.44%** with FPR falling to **10.8%**. This indicates that calibrating thresholds using the validation split (`vulnerability_val_expanded.parquet`) can yield systematic improvements without retraining.

---

## 6. Proposed Improvement Experiments (Ranked by Expected Value vs Cost)

Based on these empirical findings, we propose seven prioritized experiments for Stage 2:

| Rank | Experiment Strategy | Target Flaw Addressed | Expected Accuracy / Gain | Computational Cost |
|:---:|:---|:---|:---:|:---:|
| **Exp 1** | **Omission-Aware Feature Engineering**<br>Add features detecting unchecked dereferences (`ptr->` without preceding `if (ptr)`), missing loop guards (CWE-400/835), and unchecked return values. | Resolves CWE-476 and CWE-400 (which constitute 27.1% of all FNs). | **+1.5% to +2.5% Acc**<br>(Envoy & OpenSSL) | **Very Low**<br>(Feature extraction script, <10s) |
| **Exp 2** | **C++ & Cross-Project Stratified 35k Training**<br>Sample 35,000 samples from the 138k pool by oversampling C++ repositories (Chrome, TensorFlow) to resolve the "1 Envoy sample in 25k" training deficiency. | Resolves Envoy C++ domain shift and OpenSSL representation deficit. | **+2.0% to +3.5% Acc**<br>(Envoy target: $\ge 70\%$) | **Moderate**<br>(~10 min embedding extraction) |
| **Exp 3** | **Validation-Based Temperature & Project Calibration**<br>Fit Platt scaling / temperature scaling and compute optimal decision thresholds per language family (C vs C++) using validation data only. | Calibrates the 64.9% marginal FNs and 90.2% marginal FPs. | **+1.0% to +1.8% Acc**<br>(FPR reduction to ~12%) | **Zero**<br>(No retraining required) |
| **Exp 4** | **Hard Negative & Boundary Sample Weighting**<br>Apply cost-sensitive focal loss or sample weighting to short, ambiguous functions with high nesting depth during logistic classifier fitting. | Corrects length and complexity bias on safe multi-line functions. | **+0.8% to +1.5% Acc**<br>(FPR reduction in OpenSSL) | **Low**<br>(<1 min classifier fit) |
| **Exp 5** | **Multi-Layer Representation Pooling (Layer 6 + 12)**<br>Extract and concatenate intermediate CodeBERT transformer representations (capturing syntactic AST structure) with top-layer semantic embeddings. | Improves detection of subtle cryptographic state machine flaws. | **+1.0% to +2.0% AUROC** | **Moderate**<br>(PyTorch forward pass on 35k) |
| **Exp 6** | **Static Analysis Feature Integration in ML Vector**<br>Directly fuse normalized Flawfinder/Pattern AST severity scores into the logistic regression feature matrix rather than post-hoc calibration. | Directly escalates high-confidence static patterns. | **+0.5% to +1.0% Acc** | **Low**<br>(Cached in static analysis) |
| **Exp 7** | **Dual Classical-CodeBERT Stacking Ensemble**<br>Meta-learner blending `expanded_tri_union_best` (fast n-gram TF-IDF) with the enhanced C++ CodeBERT hybrid. | Combines n-gram keyword precision with Transformer semantic context. | **+1.5% to +2.5% Acc**<br>(Target: 76–78% Acc) | **Low**<br>(Meta-classifier fit on validation) |

---

## 7. Risks & Data Availability Constraints

1. **Test Set Preservation Rule:** All candidate models must remain strictly evaluated on the frozen held-out benchmark ($N = 3,169$). Under no circumstances will test data be used for hyperparameter selection or threshold tuning.
2. **Computational Budget:** CodeBERT embedding extraction for 35,000 samples takes approximately 8–10 minutes using Apple Silicon MPS / PyTorch. Caching embeddings permanently prevents recomputation.
3. **Target Feasibility:** Reaching 78–80% overall accuracy is ambitious for a frozen cross-project benchmark with zero task overlap. A realistic, scientifically defensible target is **76.0%–77.5% overall**, with **Envoy $\ge 68–72\%$** and **OpenSSL $\ge 72–75\%$**.
