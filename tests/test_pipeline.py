#!/usr/bin/env python3
"""
Comprehensive Test Suite for Research Pipeline
===============================================
Tests data preprocessing, splitting, static security analysis,
baseline model inference, and controlled editing experiments.
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
        self.assertTrue(verify_no_leakage(splits))
        
        # Test detection of leakage
        leaky_splits = {'train': train_df, 'test': train_df}
        self.assertFalse(verify_no_leakage(leaky_splits))


class TestSecurityAnalysis(unittest.TestCase):
    """Test heuristic pattern analysis and risk scoring."""
    
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


class TestModelInference(unittest.TestCase):
    """Test baseline model loading and inference."""
    
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


class TestControlledEditing(unittest.TestCase):
    """Test controlled editing experiment and lineage tracking."""
    
    def test_pilot_experiment(self):
        from src.experiments.controlled_editing import create_pilot_experiment, analyze_experiment_security
        exp = create_pilot_experiment()
        self.assertEqual(exp.task_id, "pilot_001_buffer_handling")
        self.assertEqual(len(exp.versions), 4)
        
        lineage = exp.get_lineage("pilot_001_v3_human_edited_ai")
        self.assertEqual(len(lineage), 2)
        self.assertEqual(lineage[0].version_id, "pilot_001_v2_ai_insecure")
        self.assertEqual(lineage[1].version_id, "pilot_001_v3_human_edited_ai")
        
        analysis = analyze_experiment_security(exp)
        self.assertEqual(analysis['n_versions'], 4)
        self.assertGreater(len(analysis['transitions']), 0)


if __name__ == "__main__":
    unittest.main()
