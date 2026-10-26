# Data Leakage & Split Audit Report
**B.Tech Project-I — Delhi Technological University**  
**Project Title:** Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code  
**Investigators:** Mankabir Singh (23/IT/199) & Mohnish Gautam (23/IT/200)  
**Date:** October 26, 2026  

---

## 1. Executive Summary

Data leakage in machine learning for vulnerability detection is a widely recognized threat to scientific validity. Standard random or stratified splitting frequently allows:
1. **Exact duplicate code** between train and test.
2. **Vulnerable/fixed pairs** of the same function to cross partitions, inflating test scores through near-identity recall.
3. **Commit-level contamination**, where functions edited in the exact same commit leak syntactic context.

To prevent these failure modes, we designed and implemented a **Group-Isolated Stratified Split** algorithm in `src/data/split_dataset.py` operating over 16,709 unique groups.

---

## 2. Partition Summary & Label Balance

The unified 36,080-sample vulnerability dataset was partitioned into Train (75%), Validation (10%), and Test (15%) splits:

```
================================================================================
PARTITION BREAKDOWN & CLASS BALANCE
================================================================================
Partition    Samples    Unique Groups    Vulnerable    Benign    Vuln Rate
--------------------------------------------------------------------------------
Train         27,060       12,376          12,586      14,474      46.51%
Validation     3,608        1,781           1,620       1,988      44.90%
Test           5,412        2,552           2,634       2,778      48.67%
--------------------------------------------------------------------------------
Total         36,080       16,709          16,840      19,240      46.67%
================================================================================
```

---

## 3. Four-Tier Leakage Verification

### Tier 1: Exact Duplicate Code Overlap
- **Methodology:** Generated SHA-256 hashes of all normalized code strings in Train, Validation, and Test. Calculated set intersections.
- **Train vs. Test Overlap:** **0 samples (0.00%)**
- **Train vs. Val Overlap:** **0 samples (0.00%)**
- **Val vs. Test Overlap:** **0 samples (0.00%)**
- **Status:** **PASSED (Verified Zero Exact Duplicates) ✓**

### Tier 2: Group ID & Vulnerable-Fixed Pair Isolation
- **Methodology:** PrimeVul vulnerable/fixed pairs share `group_id = pv_pair_{source_record_id}`. Devign functions from the same commit share `group_id = devign_commit_{commit_id}`. Calculated group set intersections across splits.
- **Train Groups & Test Groups Overlap:** **0 groups (0.00%)**
- **Train Groups & Val Groups Overlap:** **0 groups (0.00%)**
- **Val Groups & Test Groups Overlap:** **0 groups (0.00%)**
- **Status:** **PASSED (Verified Strict Group Disjointness) ✓**
- **Guarantee:** No vulnerable function in `train` has its fixed counterpart in `test`.

### Tier 3: Near-Duplicate Similarity Audit (Token 3-Gram Jaccard)
- **Methodology:** Sampled 500 test functions and 500 train functions. Computed pairwise Jaccard similarity across token 3-grams (`shingles`).
- **High-Similarity Threshold (>0.85 Jaccard):** Only 1 sample pair flagged out of 250,000 comparisons (0.0004%), corresponding to standard minimal boilerplate (`main(void) { return 0; }`).
- **Status:** **PASSED (Negligible Near-Duplicate Similarity) ✓**

### Tier 4: Preprocessing & Threshold Leakage Prevention
- **Methodology:**
  - Feature extraction vectorizers (TF-IDF) fit **strictly on `train`** and only transform `val` and `test`.
  - Decision threshold tuning and hyperparameter selection occur **strictly on `val`**.
  - The `test` partition is held out and evaluated only once at the final benchmark stage.
- **Status:** **PASSED ✓**

---

## 4. Audit Verdict

```
+--------------------------------------------------------------------+
|                      LEAKAGE AUDIT VERDICT                         |
+--------------------------------------------------------------------+
|  Exact Code Leakage:              NONE (0 matches)                 |
|  Vulnerable/Fixed Pair Leakage:   NONE (0 matches)                 |
|  Group-Level Contamination:       NONE (0 matches)                 |
|  Target Preprocessing Leakage:    NONE                             |
|  OVERALL AUDIT OUTCOME:           COMPLIANT & LEAKAGE-FREE (PASS)  |
+--------------------------------------------------------------------+
```
