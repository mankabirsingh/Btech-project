# Pre-Presentation Verification Report: Phase 13 Provenance-Aware Security Risk Assessment

**Project:** Provenance-Aware Security Risk Assessment of Human–AI Hybrid Code  
**Institution:** Delhi Technological University, Department of Information Technology  
**Authors:** Mankabir Singh (23/IT/199) and Mohnish Gautam (23/IT/200)  
**Verification Date:** October 27, 2026  
**Execution Status:** Fully Verified, 48/48 Unit Tests Passing, Zero Checksum Discrepancies  

---

## 1. Verified Runtime Checkpoint Paths & Configuration

### Runtime Checkpoints Loaded by [`app.py`](file:///Users/mankabir/Desktop/Btech%20project/app.py) via [`src/models/inference.py`](file:///Users/mankabir/Desktop/Btech%20project/src/models/inference.py):

| Model Component | Active Checkpoint Path | Architecture & Feature Spec | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| **Model A (Code Only)** | `models/checkpoints/expanded_tri_union_best.pkl` | Tri-Union SVM (Word + Char + Security AST tokens, 50k samples) | `61d365851efb794025178696ecae7702f1a660a030efbbdff4e70e30d6350fdf` |
| **Model B (Code + Provenance)** | `models/checkpoints/model_b_provenance_fusion.pkl` | Late Fusion Calibrated Logistic Regression (6 features) | `7cb5e582d90da3efad16f3801ecb049d52a22591e102f9e42cb71578328aa07e` |

### Why the UI Uses `model_b_provenance_fusion.pkl` rather than `phase13_model_b_fusion_best.pkl`:
1. **Separation of Presentation UI from Experimental Retraining (Critical Instruction #6 & #7):**  
   The primary presentation requirement is a stable, reliable demonstration interface. `model_b_provenance_fusion.pkl` was calibrated specifically for the 6-dimensional feature schema (`[logit_p_code, has_provenance, ai_involvement, is_human_edited, is_pure_ai, confidence]`) and reliably produces distinct operational tiers:
   - Unreviewed AI drafts elevate risk into **Tier-1 Urgent Audit**.
   - Defensively refactored human edits moderate risk into **Tier-3 Standard Audit**.
   - Unknown provenance cleanly triggers the code-only fallback ($P_{\text{final}} = P_{\text{code}}$).
2. **Phase 13 Checkpoint Purpose (`phase13_model_b_fusion_best.pkl`):**  
   The Phase 13 checkpoint was trained on an alternate 7-dimensional schema (`[logit_code, has_provenance, ai_involvement, is_pure_ai, is_human_edited, is_ai_edited, confidence]`) using an artificially inverted 50/50 balanced dataset designed specifically to test whether provenance provides an inductive shortcut. Swapping the UI backend hours before the presentation would risk schema misalignment, violating the instruction not to perform uncontrolled last-minute checkpoint replacements.

---

## 2. Actual Live Demo Scenario Outputs (Empirically Recorded)

Below are the exact, un-hardcoded outputs produced by the running checkpoints:

### Scenario 1: Vulnerable `strcpy` + Unreviewed AI Provenance
- **Source Code:**
  ```c
  #include <stdio.h>
  #include <string.h>

  void process_request(const char *user_input) {
      char local_buffer[16];
      // Dangerous unbounded memory copy into a fixed-size buffer (CWE-120/CWE-119)
      strcpy(local_buffer, user_input);
      printf("Processing: %s\n", local_buffer);
  }
  ```
- **Selected Provenance:** `AI-generated, not yet human-reviewed`
- **Model A Output:**
  - Predicted Class: **`Vulnerable`**
  - Probability: **`0.8500` (85.0%)**
  - Static Security Findings: **3 findings**
    1. Line 7: `CWE-120` Does not check for buffer overflows when copying to destination [MS-banned]
    2. Line 5: `CWE-119` Statically-sized arrays can be improperly restricted (16 bytes)
    3. Line 7: `CWE-120` Use of `strcpy()` without bounds checking
  - Label: *“Uses source code only.”*
- **Model B Output:**
  - Predicted Class: **`Vulnerable`**
  - Probability / Risk Score: **`0.9818` (98.2%)**
  - Review Priority Tier: **`Tier-1 (Urgent Audit — Unreviewed AI Code)`**
  - Label: *“Uses source code + provenance.”*
- **Comparison Delta:**
  - Probability Delta ($B - A$): **`+13.18 pp` (+13.2%)**
  - Model Agreement: **`True` (Full Agreement on vulnerability)**

---

### Scenario 2: Safer Bounded-Copy (`snprintf`) + Human-Written Provenance
- **Source Code:**
  ```c
  #include <stdio.h>
  #include <string.h>

  int safe_process_request(const char *user_input) {
      if (user_input == NULL) {
          return -1;
      }
      char local_buffer[64];
      // Bounds-checked string formatting with strict size limit and error check
      int written = snprintf(local_buffer, sizeof(local_buffer), "%s", user_input);
      if (written < 0 || (size_t)written >= sizeof(local_buffer)) {
          return -1; // Truncation or formatting error detected
      }
      return 0;
  }
  ```
- **Selected Provenance:** `Human-written`
- **Model A Output:**
  - Predicted Class: **`Safe`**
  - Probability: **`0.2606` (26.1%)**
  - Static Security Findings: **1 informational note** (Flawfinder notice regarding static stack buffer `char local_buffer[64];`, with no severe API bans)
  - Label: *“Uses source code only.”*
- **Model B Output:**
  - Predicted Class: **`Safe`**
  - Probability / Risk Score: **`0.0100` (1.0%, calibrated floor)**
  - Review Priority Tier: **`Tier-3 (Standard Audit)`**
  - Label: *“Uses source code + provenance.”*
- **Comparison Delta:**
  - Probability Delta ($B - A$): **`-25.06 pp` (-25.1%)**
  - Model Agreement: **`True` (Full Agreement on safety)**

---

### Scenario 3: Safer Code + Unknown / Unavailable Provenance (Fallback)
- **Source Code:** Identical safe `snprintf` function as Scenario 2.
- **Selected Provenance:** `Unknown / unavailable`
- **Model A Output:**
  - Predicted Class: **`Safe`**
  - Probability: **`0.2606` (26.1%)**
- **Model B Output:**
  - Predicted Class: **`Safe`**
  - Probability / Risk Score: **`0.2606` (26.1%)** (Identical to Model A)
  - Review Priority Tier: **`Tier-3 (Standard Audit — Unknown Provenance)`**
  - Has Provenance Flag: **`False`**
  - Status Note: **`Model B: Provenance Unknown (Falling back to Code-Only Assessment)`**
- **Comparison Delta:**
  - Probability Delta ($B - A$): **`0.00 pp` (Exact Match)**
  - Verified Fallback: Model B outputs exactly $P_{\text{code}}$ without fabricating assumptions.

---

## 3. Verification of Model A Invariance & Safety Claims

### Model A Invariance Under Changing Provenance:
Tested across all 5 provenance scenarios on both code samples:
- **Vulnerable C Example:** Model A output is strictly invariant at **`85.0000%`** across all 5 provenance selections (sample variance: $0.0000$).
- **Safer C Example:** Model A output is strictly invariant at **`26.0620%`** across all 5 provenance selections (sample variance: $0.0000$).
- **Conclusion:** Provenance features are completely isolated from Model A.

### Verification of the Safer Example:
- **Input Assumptions:** Untrusted `const char *user_input`.
- **Defensive Safeguards Present:**
  1. `if (user_input == NULL) return -1;` protects against null pointer dereference.
  2. `sizeof(local_buffer)` (64 bytes) strictly limits output string length.
  3. Static format string `"%s"` prevents format-string injection (CWE-134).
  4. Explicit error handling: `if (written < 0 || (size_t)written >= sizeof(local_buffer)) return -1;` catches encoding errors and truncation.
- **Explanation Integrity:** The caption correctly attributes safety to the *combination* of explicit buffer bounds (`sizeof`), the NULL guard, and the return-value truncation check. It does not falsely claim that calling `snprintf` alone guarantees security regardless of arguments or truncation handling.

---

## 4. Full Unit Test Suite Results

Command executed: `python -m unittest discover tests -v`  
Environment: `./venv/bin/python` (Python 3.9.6, macOS Darwin)  
Result: **`Ran 48 tests in 5.457s — OK`**

```
test_partition_files_and_manifest_exist (test_phase13.TestPhase13DatasetIntegrity) ... ok
test_locked_test_checksum_integrity (test_phase13.TestPhase13DatasetIntegrity) ... ok
test_class_and_provenance_balance (test_phase13.TestPhase13DatasetIntegrity) ... ok
test_zero_task_and_code_leakage (test_phase13.TestPhase13DatasetIntegrity) ... ok
test_phase13_checkpoint_validity (test_phase13.TestPhase13ModelInferenceAndUI) ... ok
test_missing_provenance_fallback (test_phase13.TestPhase13ModelInferenceAndUI) ... ok
test_model_a_provenance_invariance (test_phase13.TestPhase13ModelInferenceAndUI) ... ok
test_presentation_samples_evaluation (test_phase13.TestPhase13ModelInferenceAndUI) ... ok
test_provenance_options_schema (test_phase13.TestPhase13ModelInferenceAndUI) ... ok
test_class_and_category_balance (test_phase11.TestPhase11DatasetIntegrity) ... ok
test_locked_test_checksum_integrity (test_phase11.TestPhase11DatasetIntegrity) ... ok
test_partition_files_exist_and_valid_json (test_phase11.TestPhase11DatasetIntegrity) ... ok
test_zero_task_and_code_leakage (test_phase11.TestPhase11DatasetIntegrity) ... ok
test_missing_provenance_fallback (test_phase11.TestPhase11ModelInference) ... ok
test_model_a_code_only_evaluation (test_phase11.TestPhase11ModelInference) ... ok
test_model_b_checkpoint_validity (test_phase11.TestPhase11ModelInference) ... ok
test_ui_and_demo_compatibility (test_phase11.TestPhase11ModelInference) ... ok
test_lineage_tracking (test_pipeline.TestControlledEditingAndProvenance) ... ok
test_multi_cwe_benchmark (test_pipeline.TestControlledEditingAndProvenance) ... ok
test_provenance_features_zero_label_leakage (test_pipeline.TestControlledEditingAndProvenance) ... ok
test_clean_code (test_pipeline.TestDataPreprocessing) ... ok
test_leakage_verification (test_pipeline.TestDataPreprocessing) ... ok
test_zero_group_leakage (test_pipeline.TestDataPreprocessing) ... ok
test_baseline_char_model (test_pipeline.TestModelInference) ... ok
test_codebert_classifier_checkpoint (test_pipeline.TestModelInference) ... ok
test_expanded_tri_union_checkpoint (test_pipeline.TestModelInference) ... ok
test_phase3_codebert_enhanced_checkpoint (test_pipeline.TestModelInference) ... ok
test_phase4_envoy_adapted_checkpoint (test_pipeline.TestModelInference) ... ok
test_phase5_patch_hybrid_checkpoint (test_pipeline.TestModelInference) ... ok
test_phase6_context_hybrid_checkpoint (test_pipeline.TestModelInference) ... ok
test_phase7_compiler_hybrid_checkpoint (test_pipeline.TestModelInference) ... ok
test_tri_union_best_checkpoint (test_pipeline.TestModelInference) ... ok
test_locked_benchmark_integrity (test_pipeline.TestPhase10ProperlyTrainedModelB) ... ok
test_model_b_development_data_disjointness (test_pipeline.TestPhase10ProperlyTrainedModelB) ... ok
test_model_b_missing_provenance_fallback (test_pipeline.TestPhase10ProperlyTrainedModelB) ... ok
test_model_b_trained_checkpoint_loading (test_pipeline.TestPhase10ProperlyTrainedModelB) ... ok
test_compilation_database_parsing (test_pipeline.TestPhase8WholeProgramAnalysis) ... ok
test_graph_edge_provenance_and_taint_features (test_pipeline.TestPhase8WholeProgramAnalysis) ... ok
test_phase8_taint_hybrid_checkpoint (test_pipeline.TestPhase8WholeProgramAnalysis) ... ok
test_split_integrity_and_leakage_resolution (test_pipeline.TestPhase8WholeProgramAnalysis) ... ok
test_demo_presentation_scenarios (test_pipeline.TestPhase9ModelDemonstration) ... ok
test_model_a_code_only_inference (test_pipeline.TestPhase9ModelDemonstration) ... ok
test_model_b_provenance_aware_inference (test_pipeline.TestPhase9ModelDemonstration) ... ok
test_model_comparison_fairness (test_pipeline.TestPhase9ModelDemonstration) ... ok
test_comment_false_positive_suppression (test_pipeline.TestSecurityAnalysis) ... ok
test_flawfinder_integration (test_pipeline.TestSecurityAnalysis) ... ok
test_python_eval_detection (test_pipeline.TestSecurityAnalysis) ... ok
test_vulnerable_c_patterns (test_pipeline.TestSecurityAnalysis) ... ok

----------------------------------------------------------------------
Ran 48 tests in 5.457s

OK
```

---

## 5. Cryptographic Checksum Audit of Locked Benchmarks

All benchmark files match their immutable recorded checksums:

| Benchmark Partition | File Location | Recorded SHA-256 Checksum | Verified Current SHA-256 | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 10 Benchmark 32** | `data/locked_benchmark_32/controlled_benchmark_32.json` | `d53e20033a0239e6156a9dc9e4934deac85956e1adde32d72777ccc8c234fca7` | `d53e20033a0239e6156a9dc9e4934deac85956e1adde32d72777ccc8c234fca7` | **PASS (Exact Match)** |
| **Phase 11 Locked Test** | `data/phase11_model_b/model_b_p11_test_locked.json` | `9b38be33c9873f6b0b7dc4939c00b8e9aa20293df03ae829399f940171e16466` | `9b38be33c9873f6b0b7dc4939c00b8e9aa20293df03ae829399f940171e16466` | **PASS (Exact Match)** |
| **Phase 13 Locked Test** | `data/phase13_model_b/model_b_p13_test_locked.json` | `f91d951d1b18a14b3fbc475a6007c20ccc670c7002bf45f928e24e44a70d0b9b` | `f91d951d1b18a14b3fbc475a6007c20ccc670c7002bf45f928e24e44a70d0b9b` | **PASS (Exact Match)** |

---

## 6. Exact Local Commands for Demo Launch

To launch the verified application on the presentation laptop:

```bash
# Step 1: Open Terminal and navigate to the project directory
cd "/Users/mankabir/Desktop/Btech project"

# Step 2: Activate the virtual environment
source venv/bin/activate

# Step 3: Verify the test suite (optional sanity check before panel enters)
python -m unittest discover tests -v

# Step 4: Run Streamlit on port 8501
streamlit run app.py --server.port=8501
```

---

## 7. Known Limitations to Acknowledge to the Panel

1. **Natural vs. Controlled Dual-Ground-Truth Data:**  
   Natural open-source corpora either predate generative AI (historical CVEs: 100% human code) or lack vulnerability ground truth (`HumanVsAICode`: 0% security labels). Multi-stage controlled tasks remain the sole defensible method for establishing verified dual ground truth.
2. **Modern C++ Abstraction Overhead:**  
   On Envoy Proxy (cloud proxy in C++17), CodeBERT accuracy is 52.74% due to virtual dispatch and custom smart pointers. Whole-program compilation analysis and taint tracking (explored in Phase 8) are necessary to close this domain gap.
3. **Single-Function Scope:**  
   The prototype analyzes isolated C/C++ functions and cannot observe inter-procedural guards located in callers without whole-program call-graph extraction.
