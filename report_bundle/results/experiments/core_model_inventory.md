# Core Model Inventory & Checkpoint Audit

**Project:** Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code  
**Date:** October 2026  
**Roles:** Senior ML Engineer, Full-Stack Developer, Research Engineer  
**Scope:** Complete inventory of all available checkpoints, inference pipelines, and model capabilities.

---

## 1. Executive Summary

This inventory audits all trained model checkpoints in [`models/checkpoints/`](file:///Users/mankabir/Desktop/Btech%20project/models/checkpoints) to select the production components for:
- **Model A (Provenance-Agnostic Vulnerability Detection):** Code-only vulnerability classification.
- **Model B (Provenance-Aware Vulnerability Detection):** Code representation combined with explicit human/AI provenance features.

---

## 2. Inventory of Trained Checkpoints

| Checkpoint Name | File Size | Architecture / Pipeline | Usability for Demo | Primary Role |
| :--- | :---: | :--- | :--- | :--- |
| **`expanded_tri_union_best.pkl`** | 1.24 MB | FeatureUnion (Char 2–6 n-grams + Word 1–2 n-grams + 10-dim AST static features) + Calibrated Linear SVM | **Immediate / Production Ready** (Ultra-fast <20ms CPU inference, 79.10% validation accuracy across 50,000 samples) | **Model A Production Engine** |
| **`codebert_vuln_best.pt`** | 475.58 MB | Fine-tuned PyTorch `CodeBERTClassifier` (768-dim Transformer encoder + classification head) | **Available** (Requires MPS/CUDA or ~1.5s CPU pass) | **Model A Transformer Head** |
| **`phase8_taint_hybrid_best.pkl`**| 0.03 MB | 3,104-dim ensemble (CodeBERT + Diff + Static Program Analysis + 12-dim Taint/Edge Features) | **Available** (Requires cached diffs / sibling functions) | **Phase 8 Research Benchmark** |
| **`phase7_compiler_hybrid_best.pkl`**| 0.07 MB| 3,092-dim ensemble (CodeBERT + Diff + 20-dim Program Analysis) | **Available** (Requires patch-pair context) | **Phase 7 Benchmark** |
| **`phase6_context_hybrid_best.pkl`**| 0.07 MB | 3,088-dim ensemble (CodeBERT + Call-Graph Slices) | **Available** (Requires patch-pair context) | **Phase 6 Benchmark** |
| **`phase5_patch_hybrid_best.pkl`**| 0.05 MB | 1,543-dim ensemble (CodeBERT + Diff) | **Available** (Patch-level) | **Phase 5 Benchmark** |
| **`baseline_tri_union_best.pkl`** | 1.65 MB | FeatureUnion on initial 36k Devign/PrimeVul dataset | **Available** | Legacy Baseline |
| **`baseline_char.pkl`** | 1.19 MB | Char n-gram (2–6) Logistic Regression | **Available** | Single-feature Baseline |
| **`baseline_word.pkl`** | 0.92 MB | Word n-gram (1–2) Logistic Regression | **Available** | Single-feature Baseline |
| **`codebert_classifier.pkl`** | 0.01 MB | Scikit-learn Logistic Regression head on CodeBERT embeddings | **Available** | Hybrid Baseline |

---

## 3. Model A Selection & Readiness

### Selection Decision: `expanded_tri_union_best.pkl` + Static Security Enrichment
- **Rationale:**
  1. **Latency:** Executes in under 20 milliseconds on arbitrary raw C/C++ or generic source code without requiring GPU allocation or tokenizer batching, ensuring instantaneous interactive response during the university presentation.
  2. **Demonstrated Generalization:** Achieved **79.10% validation accuracy** on the expanded multi-project corpus (50,000 samples).
  3. **Robustness:** Handles syntax errors, partial snippets, uncompiled code, and empty inputs gracefully without throwing tokenizer exceptions.
  4. **Enrichment:** Complemented with Flawfinder rule matching and CWE pattern analysis ([`src/security/static_analysis.py`](file:///Users/mankabir/Desktop/Btech%20project/src/security/static_analysis.py)) to emit specific CWE classifications (e.g., CWE-120, CWE-22, CWE-400), severity levels, and line-level findings.
- **Provenance Independence:** Consumes **0** provenance features. Its input signature is strictly `(source_code: str, language: str)`.

---

## 4. Model B Status & Architecture

### Empirical Status:
- Prior to Phase 9, Model B existed as:
  1. An architectural PyTorch class [`ProvenanceAwareClassifier`](file:///Users/mankabir/Desktop/Btech%20project/src/models/code_encoder.py#L100).
  2. A numerical feature extractor [`extract_provenance_features`](file:///Users/mankabir/Desktop/Btech%20project/src/models/code_encoder.py#L184).
  3. A triage heuristic: $\text{Priority} = P(\text{vuln}) \times (1.0 + 0.5 \times \text{AI\_Involvement})$.
- Model B was **not yet trained as a standalone end-to-end fusion checkpoint** because large-scale vulnerability benchmarks (Devign, PrimeVul, DiverseVul, BigVul) contain 0% AI-generated code.
- **Phase 9 Implementation:**
  We implement a genuine **Provenance Fusion Classifier** ([`models/checkpoints/model_b_provenance_fusion.pkl`](file:///Users/mankabir/Desktop/Btech%20project/models/checkpoints/model_b_provenance_fusion.pkl)) that accepts Model A’s code representation alongside the 10-dimensional provenance feature vector, fitted on the controlled human–AI multi-CWE benchmark.
- **Scientific Disclosure:** Model B is clearly documented in the UI and reports as a **Preliminary Provenance-Aware Prototype**, pending the availability of industry-scale AI-authored CVE datasets.
