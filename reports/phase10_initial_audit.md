# Phase 10 Initial Audit: Architecture, Checkpoints, and Methodological Baseline

**Project:** Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code  
**Date:** October 27, 2026  
**Auditors:** Senior ML Engineer & Research Team  

---

## 1. Actual Architecture of Model A and Model B

### Model A: Provenance-Agnostic Vulnerability Detection
- **Pipeline Structure:**
  1. Input: C/C++ or generic source code function.
  2. Preprocessing: `clean_code()` (comment stripping, whitespace normalization, macro preservation).
  3. Feature Extraction: `FeatureUnion` combining:
     - Character n-grams (1–4 chars, sublinear TF-IDF, max 10,000 features).
     - Token n-grams (1–2 words, regex tokenizer, max 15,000 features).
     - Security AST keywords (specialized C/C++ security-sensitive identifiers and APIs, max 2,000 features).
  4. Classifier: Linear Support Vector Classifier with Platt Scaling (`CalibratedClassifierCV(method='sigmoid')`).
  5. Static Enrichment: Flawfinder lexical scanner + security regex patterns for dominant CWE assignment.
  6. Output: Calibrated probability $P(\text{vulnerability} \mid \text{code})$, binary class (`Safe` / `Vulnerable`), and target CWE.
- **Provenance Isolation:** Model A operates with strictly zero provenance inputs.

### Model B: Provenance-Aware Vulnerability Risk Assessor
- **Current Pipeline Structure (Phase 9 Prototype):**
  1. Base Code Representation: Calls Model A to obtain $P_{\text{code}} = P(\text{vulnerability} \mid \text{code})$.
  2. Provenance Vector Extraction:
     - `logit_code`: $\ln\left(\frac{P_{\text{code}} + \epsilon}{1 - P_{\text{code}} + \epsilon}\right)$.
     - `ai_involvement`: float $\in [0.0, 1.0]$.
     - `is_human_edited`: binary flag (1.0 if human edits $>0$ or type is `human_edit_of_ai`).
     - `is_pure_ai`: binary flag (1.0 if AI-generated and human edits $==0$).
  3. Fusion Layer: `LogisticRegression` fit on the 4-dimensional vector.
  4. Confidence Scaling: Convex combination with base code probability when confidence $< 1.0$:
     $$P_{\text{final}} = \text{confidence} \cdot P_{\text{fused}} + (1 - \text{confidence}) \cdot P_{\text{code}}$$
  5. Review Priority Policy:
     $$\text{Triage Score} = P_{\text{final}} \cdot (1 + 0.5 \cdot \text{AI Involvement})$$
     Maps to `Tier-1 (Urgent Audit)`, `Tier-2 (Elevated Review)`, or `Tier-3 (Standard Audit)`.

---

## 2. Checkpoints Loaded by Each Model

| Model | Checkpoint Path | File Size | Checkpoint Type | Key Internals |
| :--- | :--- | :--- | :--- | :--- |
| **Model A** | `models/checkpoints/expanded_tri_union_best.pkl` | 58 MB | Scikit-learn Pipeline | `FeatureUnion` + `CalibratedClassifierCV` (LinearSVC) |
| **Model B** | `models/checkpoints/model_b_provenance_fusion.pkl` | 12 KB | Dictionary / Scikit-learn | `clf_fusion` (`LogisticRegression`), `feature_names`, `accuracy_on_benchmark` |

---

## 3. Training History and Learned Labels

### Model A (`expanded_tri_union_best.pkl`):
- **Training Set:** 50,000 C/C++ functions sampled from `data/splits/vulnerability_train_expanded.parquet` (total corpus: 138,412 samples from DiverseVul, BigVul, Devign, and PrimeVul).
- **Target Label:** Binary vulnerability status (`vuln_label` $\in \{0, 1\}$).
- **Validation Accuracy:** 79.10% on 23,648 validation functions (`data/splits/vulnerability_val_expanded.parquet`).
- **Cross-Project Generalization:** 71.82% accuracy on held-out clean benchmark (N=3,169 across GPAC, Vim, OpenSSL, Envoy).

### Model B (`model_b_provenance_fusion.pkl`):
- **Training Set:** The 32 examples from `results/reports/controlled_benchmark_32.json` / `src/experiments/controlled_editing.py`.
- **Target Label:** Binary ground truth vulnerability (`ground_truth_vuln` $\in \{0, 1\}$).
- **Learned Coefficients:**
  - `logit_p_code`: $+0.0039$
  - `ai_involvement`: $+1.2227$
  - `is_human_edited`: $-1.1255$
  - `is_ai`: $+2.0613$
  - `intercept`: $-1.2982$

---

## 4. Methodological Nature of Model B: Trained vs Rule-Based

Model B is a **hybrid**:
1. It uses a **trained Logistic Regression classifier** for its fusion layer (`model_b_provenance_fusion.pkl`).
2. It uses a **deterministic triage policy** (`triage_score = P * (1 + 0.5 * ai_inv)`) for categorizing predictions into operational audit tiers.
3. In Phase 9, fallback heuristics were present in case `predict_proba` raised an exception (`p_fused = p_code * (1 + 0.35 * ai_inv - 0.25 * is_human_edited)`).

---

## 5. Audit of the 32 Verified Benchmark Examples

### Benchmark Identity & Immutability:
- **File:** `results/reports/controlled_benchmark_32.json` and `src/experiments/controlled_editing.py`.
- **Scope:** 8 security tasks $\times$ 4 evolutionary stages = 32 code units.
- **Class Balance:** Exactly 16 vulnerable (50.0%) and 16 safe (50.0%).
- **CWE Coverage:** CWE-119, CWE-120, CWE-134, CWE-190, CWE-78, CWE-416, CWE-22, CWE-476.
- **Immutable Backup Recorded:**
  - Backup directory: `data/locked_benchmark_32/` (permissions set to read-only `a-w`).
  - SHA-256 Checksums:
    - `controlled_benchmark_32.json`: `d53e20033a0239e6156a9dc9e4934deac85956e1adde32d72777ccc8c234fca7`
    - `controlled_editing.py`: `22b975cbff6e6d2d13d9f8593d69226dbf87adf96ef0ac30c8224b60af539200`

### Prior Use & Methodological Issue:
- **Finding:** In Phase 9, `model_b_provenance_fusion.pkl` was fitted directly on all 32 examples.
- **Consequence:** The reported 96.88% accuracy (31/32) is **training resubstitution performance**, NOT independent held-out validation.
- **Phase 10 Mandate:** All 32 examples must be **completely locked as an independent test benchmark**. Model B must be retrained on an entirely separate, non-overlapping development corpus of seed tasks, preventing any benchmark memorization or data leakage.

---

## 6. Available Datasets, Label Definitions, and Leakage Analysis

1. **Devign, PrimeVul, BigVul, DiverseVul (138,412 train samples):**
   - Vulnerability labels: Valid, curated CVE indicators.
   - Provenance labels: **100% human-authored**. `ai_involvement` is uniformly 0.0. No AI or hybrid lineage exists.
2. **HumanVsAICode (1,986,283 samples):**
   - Provenance labels: 1,479,238 AI and 507,045 Human samples.
   - Vulnerability labels: **100% NaN (missing)**. Zero security ground truth.
3. **Controlled Multi-CWE Benchmark (32 samples):**
   - Dual labels: Ground truth vulnerability + 4-stage verified lineage.
   - Reserved strictly as locked evaluation test set.

---

## 7. Existing UI Dependencies and Functionality to Preserve

`app.py` currently relies on:
- `src/models/inference.py`:
  - `predict_model_a(code, language)`
  - `predict_model_b(code, provenance_dict, language)`
  - `compare_models(code, provenance_dict, language)`
  - `get_demo_examples()`
- Return schema keys: `probability`, `is_vulnerable`, `predicted_class`, `cwe`, `risk_level`, `review_priority`, `provenance_used`, `code_hash`, `inference_latency_ms`.
- Interactive components: Scenario presets (Demo 1, 2, 3), custom code entry, provenance controls, live ablation slider, evaluation tabs, and session history.
- All UI functionality must remain completely functional and non-breaking.

---

## 8. Discrepancies and Concrete Phase 10 Action Plan

1. **Discrepancy 1:** Model B's reported 96.88% was an in-sample training metric on the 32 benchmark cases.
   - *Resolution:* Freeze the 32 cases as a locked, immutable held-out evaluation set. Construct an expanded training dataset of separate seed programs with verified human/AI/hybrid variants to train Model B honestly.
2. **Discrepancy 2:** Model B defaulted missing provenance to `ai_involvement = 0.50`.
   - *Resolution:* Add explicit missing-value handling. When provenance is unknown or unprovided, Model B must fall back to Model A's base code assessment with zero provenance bias.
