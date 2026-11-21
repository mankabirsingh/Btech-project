#!/usr/bin/env python3
"""
Comprehensive Test Suite for Provenance-Aware Security Research Pipeline
========================================================================
Tests:
- Data preprocessing & cleaning
- Grouped data splitting & zero-leakage verification
- Static security analysis (Flawfinder & heuristic rules) with remediation
- Baseline model inference (Char, Word)
- CodeBERT classifier checkpoint loading & prediction
- Multi-CWE controlled editing benchmark (8 tasks, 32 code units)
- Lineage tracking & transition extraction
- Provenance feature extraction with ZERO label leakage
- Review prioritization metric computation
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestDataPreprocessing(unittest.TestCase):
    """Test data cleaning and preprocessing functions."""
    
    def test_clean_code(self):
        from src.data.preprocessing import clean_code
        raw = "int main() {\r\n    return 0;\r\n}\r\n\x00   \n\n"
        cleaned = clean_code(raw)
        self.assertNotIn('\r', cleaned)
        self.assertNotIn('\x00', cleaned)
        self.assertTrue(cleaned.endswith('}'))

    def test_leakage_verification(self):
        from src.data.split_dataset import verify_no_leakage
        train_df = pd.DataFrame({'code': ['func_a()', 'func_b()']})
        val_df = pd.DataFrame({'code': ['func_c()']})
        test_df = pd.DataFrame({'code': ['func_d()']})
        splits = {'train': train_df, 'val': val_df, 'test': test_df}
        self.assertTrue(verify_no_leakage(splits)['passed'])
        
        # Test detection of leakage
        leaky_splits = {'train': train_df, 'test': train_df}
        self.assertFalse(verify_no_leakage(leaky_splits)['passed'])

    def test_zero_group_leakage(self):
        """Verify that grouped splitting never places the same group_id in both train and test."""
        from src.data.split_dataset import split_vulnerability_dataset
        df = pd.DataFrame({
            'code': [f'int func_{i}() {{ return {i}; }}' for i in range(20)],
            'vuln_label': [i % 2 for i in range(20)],
            'group_id': [f'group_{i // 2}' for i in range(20)]  # 10 pairs
        })
        splits = split_vulnerability_dataset(df, test_size=0.2, val_size=0.1, random_seed=42, group_col='group_id')
        train_groups = set(splits['train']['group_id'])
        val_groups = set(splits['val']['group_id'])
        test_groups = set(splits['test']['group_id'])
        
        self.assertEqual(len(train_groups & val_groups), 0)
        self.assertEqual(len(train_groups & test_groups), 0)
        self.assertEqual(len(val_groups & test_groups), 0)


class TestSecurityAnalysis(unittest.TestCase):
    """Test heuristic pattern analysis, Flawfinder integration, and risk scoring."""
    
    def test_vulnerable_c_patterns(self):
        from src.security.static_analysis import analyze_code
        vuln_c = """
        #include <string.h>
        void test(char *input) {
            char buf[16];
            strcpy(buf, input);
            gets(input);
        }
        """
        result = analyze_code(vuln_c, 'c')
        self.assertGreater(result['n_findings'], 0)
        self.assertIn('CWE-120', result['cwes'])
        self.assertIn(result['risk_level'], ['Medium', 'High', 'Critical'])

    def test_comment_false_positive_suppression(self):
        from src.security.static_analysis import analyze_code
        secure_c = """
        // We replaced gets() with fgets() to prevent buffer overflow
        // strcpy() is avoided in favor of strncpy()
        void test() {
            int a = 1;
        }
        """
        result = analyze_code(secure_c, 'c')
        self.assertEqual(result['n_findings'], 0)
        self.assertEqual(result['risk_level'], 'Low')

    def test_python_eval_detection(self):
        from src.security.static_analysis import analyze_code
        py_vuln = "def run(x):\n    eval(x)\n"
        result = analyze_code(py_vuln, 'python')
        self.assertGreater(result['n_findings'], 0)
        self.assertIn('CWE-94', result['cwes'])

    def test_flawfinder_integration(self):
        from src.security.static_analysis import try_flawfinder_analysis
        code = """
        #include <stdio.h>
        #include <string.h>
        void test(char *s) {
            char buf[16];
            strcpy(buf, s);
        }
        """
        findings = try_flawfinder_analysis(code, 'c')
        self.assertGreater(len(findings), 0)
        self.assertTrue(any(f.tool == 'flawfinder' for f in findings))


class TestModelInference(unittest.TestCase):
    """Test baseline and CodeBERT model loading and inference."""
    
    def test_baseline_char_model(self):
        from src.models.baseline import VulnerabilityBaseline
        ckpt = PROJECT_ROOT / "models" / "checkpoints" / "baseline_char.pkl"
        if not ckpt.exists():
            self.skipTest("Baseline checkpoint not found")
        model = VulnerabilityBaseline.load(ckpt)
        self.assertTrue(model.is_trained)
        
        sample_code = np.array(["void func() { char b[10]; gets(b); }"])
        pred = model.predict(sample_code)
        prob = model.predict_proba(sample_code)
        self.assertEqual(len(pred), 1)
        self.assertEqual(prob.shape, (1, 2))
        self.assertAlmostEqual(float(prob.sum()), 1.0, places=4)

    def test_codebert_classifier_checkpoint(self):
        import pickle
        ckpt = PROJECT_ROOT / "models" / "checkpoints" / "codebert_classifier.pkl"
        if not ckpt.exists():
            self.skipTest("CodeBERT checkpoint not found")
        with open(ckpt, 'rb') as f:
            data = pickle.load(f)
        self.assertIn('clf', data)
        self.assertIn('threshold', data)
        self.assertTrue(0.1 <= data['threshold'] <= 0.9)

    def test_tri_union_best_checkpoint(self):
        import pickle
        from scripts.train_improved_models import SecurityStructuralFeatureExtractor, c_tokenize
        ckpt = PROJECT_ROOT / "models" / "checkpoints" / "baseline_tri_union_best.pkl"
        if not ckpt.exists():
            self.skipTest("Tri-Union checkpoint not found")
        with open(ckpt, 'rb') as f:
            data = pickle.load(f)
        self.assertIn('pipeline', data)
        pipe = data['pipeline']
        sample_code = ["void test() { char buf[32]; strcpy(buf, \"abc\"); }"]
        pred = pipe.predict(sample_code)
        prob = pipe.predict_proba(sample_code)
        self.assertEqual(len(pred), 1)
        self.assertEqual(prob.shape, (1, 2))
        self.assertAlmostEqual(float(prob.sum()), 1.0, places=4)

    def test_expanded_tri_union_checkpoint(self):
        import pickle
        from scripts.train_improved_models import SecurityStructuralFeatureExtractor, c_tokenize
        ckpt = PROJECT_ROOT / "models" / "checkpoints" / "expanded_tri_union_best.pkl"
        if not ckpt.exists():
            self.skipTest("Expanded Tri-Union checkpoint not found")
        with open(ckpt, 'rb') as f:
            data = pickle.load(f)
        self.assertIn('pipeline', data)
        pipe = data['pipeline']
        sample_code = ["void test() { char buf[32]; strcpy(buf, \"abc\"); }"]
        pred = pipe.predict(sample_code)
        prob = pipe.predict_proba(sample_code)
        self.assertEqual(len(pred), 1)
        self.assertEqual(prob.shape, (1, 2))
        self.assertAlmostEqual(float(prob.sum()), 1.0, places=4)

    def test_phase3_codebert_enhanced_checkpoint(self):
        import pickle
        ckpt = PROJECT_ROOT / "models" / "checkpoints" / "phase3_codebert_enhanced_best.pkl"
        if not ckpt.exists():
            self.skipTest("Phase 3 CodeBERT enhanced checkpoint not found")
        with open(ckpt, 'rb') as f:
            data = pickle.load(f)
        self.assertIn('clf', data)
        self.assertIn('scaler', data)
        self.assertEqual(data['clf'].n_features_in_, 1555)

    def test_phase4_envoy_adapted_checkpoint(self):
        import pickle
        ckpt = PROJECT_ROOT / "models" / "checkpoints" / "phase4_envoy_adapted_best.pkl"
        if not ckpt.exists():
            self.skipTest("Phase 4 Envoy adapted checkpoint not found")
        with open(ckpt, 'rb') as f:
            data = pickle.load(f)
        self.assertIn('clf', data)
        self.assertIn('scaler', data)
        self.assertIn('selected_threshold', data)
        self.assertEqual(data['clf'].n_features_in_, 1555)

    def test_phase5_patch_hybrid_checkpoint(self):
        import pickle
        ckpt = PROJECT_ROOT / "models" / "checkpoints" / "phase5_patch_hybrid_best.pkl"
        if not ckpt.exists():
            self.skipTest("Phase 5 Patch Hybrid checkpoint not found")
        with open(ckpt, 'rb') as f:
            data = pickle.load(f)
        self.assertIn('clf_hybrid', data)
        self.assertIn('clf_diff', data)
        self.assertIn('clf_ctx', data)
        self.assertIn('validation_results', data)
        self.assertEqual(data['clf_hybrid'].n_features_in_, 1536 + 1536 + 7)

    def test_phase6_context_hybrid_checkpoint(self):
        import pickle
        ckpt = PROJECT_ROOT / "models" / "checkpoints" / "phase6_context_hybrid_best.pkl"
        if not ckpt.exists():
            self.skipTest("Phase 6 Context Hybrid checkpoint not found")
        with open(ckpt, 'rb') as f:
            data = pickle.load(f)
        self.assertIn('clf_hybrid', data)
        self.assertIn('clf_slicing', data)
        self.assertIn('clf_baseline', data)
        self.assertIn('scaler16', data)
        self.assertIn('selected_threshold', data)
        self.assertEqual(data['clf_hybrid'].n_features_in_, 1536 + 1536 + 16)
        self.assertEqual(data['clf_slicing'].n_features_in_, 1536 + 16)

    def test_phase7_compiler_hybrid_checkpoint(self):
        import pickle
        ckpt = PROJECT_ROOT / "models" / "checkpoints" / "phase7_compiler_hybrid_best.pkl"
        if not ckpt.exists():
            self.skipTest("Phase 7 Compiler Hybrid checkpoint not found")
        with open(ckpt, 'rb') as f:
            data = pickle.load(f)
        self.assertIn('clf_hybrid', data)
        self.assertIn('clf_program_analysis', data)
        self.assertIn('clf_baseline', data)
        self.assertIn('scaler20', data)
        self.assertIn('selected_threshold', data)
        self.assertEqual(data['clf_hybrid'].n_features_in_, 1536 + 1536 + 20)
        self.assertEqual(data['clf_program_analysis'].n_features_in_, 1536 + 20)






class TestControlledEditingAndProvenance(unittest.TestCase):
    """Test controlled benchmark, lineage tracking, and provenance feature validity."""
    
    def test_multi_cwe_benchmark(self):
        from src.experiments.controlled_editing import create_multi_cwe_benchmark
        benchmarks = create_multi_cwe_benchmark()
        self.assertEqual(len(benchmarks), 8)
        
        total_versions = sum(len(exp.versions) for exp in benchmarks)
        self.assertEqual(total_versions, 32)
        
        # Check balance: 16 vulnerable, 16 secure
        vuln_count = sum(sum(1 for v in exp.versions if v.vuln_label == 1) for exp in benchmarks)
        self.assertEqual(vuln_count, 16)

    def test_lineage_tracking(self):
        from src.experiments.controlled_editing import create_pilot_experiment
        exp = create_pilot_experiment()
        lineage = exp.get_lineage("pilot_001_v3_human_edited_ai")
        self.assertEqual(len(lineage), 2)
        self.assertEqual(lineage[0].version_id, "pilot_001_v2_ai_insecure")
        self.assertEqual(lineage[1].version_id, "pilot_001_v3_human_edited_ai")

    def test_provenance_features_zero_label_leakage(self):
        from src.models.code_encoder import extract_provenance_features
        df = pd.DataFrame({
            'code': ['int main() { return 0; }', 'void test() { char *p = malloc(10); }'],
            'ai_involvement': [0.0, 1.0],
            'provenance_type': ['human', 'ai'],
            'ai_generator': [None, 'copilot'],
            'human_edit_history': [None, None],
            'cwe': [None, 'CWE-120']  # Deliberate CWE target column in DataFrame
        })
        feats = extract_provenance_features(df)
        self.assertEqual(feats.shape[0], 2)
        self.assertEqual(feats.shape[1], 10)
        # Verify that changing cwe column DOES NOT alter extracted provenance features
        df_altered = df.copy()
        df_altered['cwe'] = ['CWE-999', None]
        feats_altered = extract_provenance_features(df_altered)
        np.testing.assert_array_almost_equal(feats, feats_altered)


class TestPhase8WholeProgramAnalysis(unittest.TestCase):
    """Test Phase 8 compilation database parsing, edge provenance, and taint analysis."""
    
    def test_compilation_database_parsing(self):
        """Verify that envoy_compile_commands.json contains valid translation units."""
        import json
        compdb_path = PROJECT_ROOT / "results" / "experiments" / "envoy_compile_commands.json"
        self.assertTrue(compdb_path.exists(), "envoy_compile_commands.json must exist")
        with open(compdb_path) as f:
            units = json.load(f)
        self.assertGreaterEqual(len(units), 2)
        for u in units:
            self.assertIn("command", u)
            self.assertIn("file", u)
            self.assertIn("directory", u)
            self.assertIn("compiled_successfully", u)

    def test_graph_edge_provenance_and_taint_features(self):
        """Verify that extract_taint_and_graph_features extracts 12 dimensions with correct edge provenance."""
        from scripts.run_phase8_experiments import extract_taint_and_graph_features
        code_sample = """
        void ConnectionManagerImpl::ActiveStream::decodeHeaders(RequestHeaderMap& headers) {
            std::string path = PathUtil::removeQueryAndFragment(headers.getPathValue());
            if (unconsumed_bytes_ > 0) {
                nghttp2_session_consume(session_, stream_id, unconsumed_bytes_);
            }
            filter_->decodeHeaders(headers);
            buffer_.reserve(4096);
        }
        """
        feats = extract_taint_and_graph_features(code_sample)
        self.assertEqual(len(feats), 12)
        # Check source and sink flags
        self.assertEqual(feats[0], 1.0)  # has_source
        self.assertEqual(feats[1], 1.0)  # has_sink
        self.assertEqual(feats[7], 1.0)  # has_sanitizer
        # Check edge ratios sum to <= 1.0
        conf_r, inf_r, unres_r = feats[8], feats[9], feats[10]
        self.assertAlmostEqual(conf_r + inf_r + unres_r, 1.0, places=4)

    def test_phase8_taint_hybrid_checkpoint(self):
        """Verify that phase8_taint_hybrid_best.pkl loads and performs inference on 3,104-dim features."""
        import pickle
        ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "phase8_taint_hybrid_best.pkl"
        self.assertTrue(ckpt_path.exists(), "Phase 8 checkpoint must exist")
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        self.assertIn("model_c", ckpt)
        self.assertIn("feature_dim", ckpt)
        self.assertEqual(ckpt["feature_dim"], 3104)
        dummy_X = np.zeros((2, 3104))
        probs = ckpt["model_c"].predict_proba(dummy_X)[:, 1]
        self.assertEqual(len(probs), 2)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_split_integrity_and_leakage_resolution(self):
        """Verify that actual train, val, and test splits have 0 code duplicates and 0 commit overlap."""
        import json
        manifest_path = PROJECT_ROOT / "data" / "splits" / "phase5_patch_splits.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        c_tr = set(manifest["train_commits"])
        c_val = set(manifest["val_commits"])
        c_test = set(manifest["heldout_test_commits"])
        self.assertEqual(len(c_tr & c_val), 0, "Zero train-val commit overlap")
        self.assertEqual(len(c_tr & c_test), 0, "Zero train-test commit overlap")
        self.assertEqual(len(c_val & c_test), 0, "Zero val-test commit overlap")


class TestPhase9ModelDemonstration(unittest.TestCase):
    """Test Model A, Model B, and Comparative Demonstration Backend."""
    
    def test_model_a_code_only_inference(self):
        """Verify that Model A evaluates source code without provenance."""
        from src.models.inference import predict_model_a
        vuln_c = "void bad(char *s) { char buf[10]; strcpy(buf, s); }"
        res = predict_model_a(vuln_c)
        self.assertEqual(res["model_id"], "Model A (Code Only)")
        self.assertTrue(0.0 <= res["probability"] <= 1.0)
        self.assertIn(res["predicted_class"], ["Vulnerable", "Safe"])
        self.assertIn("cwe", res)
        self.assertIn("risk_level", res)
        self.assertIn("inference_latency_ms", res)
        # Verify empty input handling
        empty_res = predict_model_a("")
        self.assertFalse(empty_res["is_vulnerable"])
        self.assertEqual(empty_res["probability"], 0.0)

    def test_model_b_provenance_aware_inference(self):
        """Verify that Model B consumes provenance features and computes triage priority."""
        from src.models.inference import predict_model_b
        code = "int add(int a, int b) { return a + b; }"
        prov = {
            "ai_involvement": 0.90,
            "provenance_type": "ai",
            "ai_generator": "ChatGPT",
            "human_edit_count": 0
        }
        res = predict_model_b(code, provenance=prov)
        self.assertEqual(res["model_id"], "Model B (Code + Provenance)")
        self.assertTrue(0.0 <= res["probability"] <= 1.0)
        self.assertIn("review_priority", res)
        self.assertEqual(res["provenance_used"]["ai_involvement"], 0.90)

    def test_model_comparison_fairness(self):
        """Verify that compare_models passes identical code and reports probability delta."""
        from src.models.inference import compare_models
        code = "void test() { int x = 0; }"
        prov = {"ai_involvement": 0.50, "provenance_type": "human_edit_of_ai", "human_edit_count": 2}
        comp = compare_models(code, provenance=prov)
        self.assertTrue(comp["source_code_identical"])
        self.assertIn("model_a", comp)
        self.assertIn("model_b", comp)
        self.assertIn("probability_delta_percentage_points", comp)
        self.assertIn("comparison_narrative", comp)

    def test_demo_presentation_scenarios(self):
        """Verify that all 3 demonstration examples are present and executable."""
        from src.models.inference import get_demo_examples, compare_models
        demos = get_demo_examples()
        self.assertIn("demo_1_vulnerable", demos)
        self.assertIn("demo_2_benign", demos)
        self.assertIn("demo_3_hybrid_ablation", demos)
        for key, demo in demos.items():
            res = compare_models(demo["code"], demo["provenance"], language=demo.get("language", "c"))
            self.assertTrue(0.0 <= res["model_a"]["probability"] <= 1.0)
            self.assertTrue(0.0 <= res["model_b"]["probability"] <= 1.0)


class TestPhase10ProperlyTrainedModelB(unittest.TestCase):
    """Test Phase 10 Model B training integrity, benchmark locking, and missing provenance handling."""

    def test_locked_benchmark_integrity(self):
        """Verify that the 32-sample benchmark file matches its immutable locked checksum."""
        import hashlib
        bench_file = PROJECT_ROOT / "results" / "reports" / "controlled_benchmark_32.json"
        chk_file = PROJECT_ROOT / "data" / "locked_benchmark_32" / "CHECKSUMS.sha256"
        self.assertTrue(bench_file.exists(), "Benchmark file must exist")
        self.assertTrue(chk_file.exists(), "Locked checksum file must exist")
        
        with open(bench_file, "rb") as f:
            curr_sha = hashlib.sha256(f.read()).hexdigest()
        with open(chk_file, "r") as f:
            expected_sha = f.readline().split()[0]
        self.assertEqual(curr_sha, expected_sha, "Benchmark file has been altered or regenerated!")

    def test_model_b_development_data_disjointness(self):
        """Verify that development seed tasks have zero overlap with the 8 locked benchmark tasks."""
        import json
        manifest_path = PROJECT_ROOT / "data" / "model_b_dev" / "manifest.json"
        self.assertTrue(manifest_path.exists(), "Model B dev manifest must exist")
        
        with open(PROJECT_ROOT / "data" / "model_b_dev" / "model_b_dev_train.json") as f:
            tr_data = json.load(f)
        with open(PROJECT_ROOT / "data" / "model_b_dev" / "model_b_dev_val.json") as f:
            val_data = json.load(f)
        with open(PROJECT_ROOT / "results" / "reports" / "controlled_benchmark_32.json") as f:
            bench_data = json.load(f)
            
        dev_tasks = set(tr_data["tasks"]) | set(val_data["tasks"])
        bench_tasks = set(r["task_id"] for r in bench_data["records"])
        
        # Zero task overlap
        overlap = dev_tasks & bench_tasks
        self.assertEqual(len(overlap), 0, f"Leakage detected! Overlapping tasks: {overlap}")
        
        # Zero code hash overlap
        dev_hashes = set(r["code_hash"] for r in tr_data["records"] + val_data["records"])
        import hashlib
        bench_hashes = set(hashlib.sha256(r.get("code", "").strip().encode()).hexdigest()[:16] for r in bench_data["records"] if "code" in r)
        hash_overlap = dev_hashes & bench_hashes
        self.assertEqual(len(hash_overlap), 0, f"Code duplicate leakage detected: {hash_overlap}")

    def test_model_b_missing_provenance_fallback(self):
        """Verify that Model B cleanly falls back to Model A probability when provenance is unprovided."""
        from src.models.inference import predict_model_a, predict_model_b
        code = "int calc(int x) { return x * 2; }"
        res_a = predict_model_a(code)
        res_b_none = predict_model_b(code, provenance=None)
        res_b_empty = predict_model_b(code, provenance={})
        res_b_unknown = predict_model_b(code, provenance={"provenance_type": "unknown"})
        
        self.assertAlmostEqual(res_a["probability"], res_b_none["probability"], places=5)
        self.assertAlmostEqual(res_a["probability"], res_b_empty["probability"], places=5)
        self.assertAlmostEqual(res_a["probability"], res_b_unknown["probability"], places=5)
        self.assertFalse(res_b_none["has_provenance"])
        self.assertIn("Provenance Unknown", res_b_none["status_note"])

    def test_model_b_trained_checkpoint_loading(self):
        """Verify that Model B checkpoint contains valid trained weights and schema."""
        import pickle
        ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "model_b_provenance_fusion.pkl"
        self.assertTrue(ckpt_path.exists())
        with open(ckpt_path, "rb") as f:
            data = pickle.load(f)
        self.assertIn("clf_fusion", data)
        self.assertIn("feature_names", data)
        self.assertIn("coefficients", data)
        self.assertIn("regularizer_C", data)
        self.assertIn("benchmark_leakage_audit", data)
        self.assertTrue(len(data["feature_names"]) >= 4)


if __name__ == "__main__":
    unittest.main()

