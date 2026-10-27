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


if __name__ == "__main__":
    unittest.main()
