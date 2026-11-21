"""
Phase 11 Real-World Generalization & Robust Provenance Detection Tests
======================================================================
DTU B.Tech Project: Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code
Authors: Mankabir Singh (23/IT/199) and Mohnish Gautam (23/IT/200)

Comprehensive unit tests verifying:
1. Phase 11 balanced dataset schema, integrity, and category balance.
2. Complete task and code isolation between train, val, test, and locked benchmarks.
3. Checkpoint validity and feature schemas for both Model A and Model B.
4. Provenance-unknown fallback precision.
5. Review prioritization and UI compatibility.
"""

import unittest
import json
import pickle
import hashlib
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "phase11_model_b"
LOCKED_DIR = PROJECT_ROOT / "data" / "locked_benchmark_32"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"

from src.models.inference import (
    predict_model_a,
    predict_model_b,
    compare_models,
    get_demo_examples,
)


class TestPhase11DatasetIntegrity(unittest.TestCase):
    """Verifies Phase 11 dataset structure, balance, and isolation."""

    def test_partition_files_exist_and_valid_json(self):
        """Verify train, val, and locked test files exist and are valid JSON."""
        for split_file in ["model_b_p11_train.json", "model_b_p11_val.json", "model_b_p11_test_locked.json", "manifest.json"]:
            path = DATA_DIR / split_file
            self.assertTrue(path.exists(), f"Missing dataset file: {path}")
            with open(path, "r") as f:
                data = json.load(f)
            self.assertIsInstance(data, dict)

    def test_locked_test_checksum_integrity(self):
        """Verify the test set matches its recorded cryptographic SHA-256 checksum."""
        test_file = DATA_DIR / "model_b_p11_test_locked.json"
        with open(test_file, "rb") as f:
            computed_sha = hashlib.sha256(f.read()).hexdigest()
        
        checksum_file = DATA_DIR / "CHECKSUMS.sha256"
        with open(checksum_file, "r") as f:
            recorded_sha = f.read().split()[0]
            
        self.assertEqual(computed_sha, recorded_sha, "Locked Phase 11 test checksum mismatch!")

    def test_class_and_category_balance(self):
        """Verify that every partition has 50% vulnerable code across every provenance category."""
        for split_file in ["model_b_p11_train.json", "model_b_p11_val.json", "model_b_p11_test_locked.json"]:
            with open(DATA_DIR / split_file, "r") as f:
                d = json.load(f)
            recs = d["records"]
            
            # Overall 50:50 balance
            n_vuln = sum(1 for r in recs if r["vuln_label"] == 1)
            n_safe = sum(1 for r in recs if r["vuln_label"] == 0)
            self.assertEqual(n_vuln, n_safe, f"Overall class imbalance in {split_file}")
            
            # Per-category 50:50 balance
            for auth in ["human", "ai", "human_edit_of_ai", "ai_edit_of_human"]:
                auth_recs = [r for r in recs if r["author_type"] == auth]
                self.assertGreater(len(auth_recs), 0, f"Missing category {auth} in {split_file}")
                auth_vuln = sum(1 for r in auth_recs if r["vuln_label"] == 1)
                auth_safe = sum(1 for r in auth_recs if r["vuln_label"] == 0)
                self.assertEqual(auth_vuln, auth_safe, f"Category {auth} imbalance in {split_file}: {auth_vuln}V vs {auth_safe}S")

    def test_zero_task_and_code_leakage(self):
        """Verify complete task and code isolation between all partitions and locked benchmarks."""
        with open(DATA_DIR / "model_b_p11_train.json") as f:
            train_recs = json.load(f)["records"]
        with open(DATA_DIR / "model_b_p11_val.json") as f:
            val_recs = json.load(f)["records"]
        with open(DATA_DIR / "model_b_p11_test_locked.json") as f:
            test_recs = json.load(f)["records"]
            
        train_tasks = set(r["task_id"] for r in train_recs)
        val_tasks = set(r["task_id"] for r in val_recs)
        test_tasks = set(r["task_id"] for r in test_recs)
        
        # Zero task overlap within Phase 11
        self.assertEqual(len(train_tasks & val_tasks), 0)
        self.assertEqual(len(train_tasks & test_tasks), 0)
        self.assertEqual(len(val_tasks & test_tasks), 0)
        
        # Zero code overlap
        train_codes = set(r["code"].strip() for r in train_recs)
        val_codes = set(r["code"].strip() for r in val_recs)
        test_codes = set(r["code"].strip() for r in test_recs)
        
        self.assertEqual(len(train_codes & test_codes), 0)
        self.assertEqual(len(val_codes & test_codes), 0)
        
        # Zero overlap with Phase 10 locked 32 benchmark
        with open(LOCKED_DIR / "controlled_benchmark_32.json") as f:
            locked_recs = json.load(f)["records"]
        locked_tasks = set(r["task_id"] for r in locked_recs)
        p11_all_tasks = train_tasks | val_tasks | test_tasks
        self.assertEqual(len(p11_all_tasks & locked_tasks), 0, "Phase 11 tasks leak into locked 32 benchmark!")


class TestPhase11ModelInference(unittest.TestCase):
    """Verifies Model A and Model B inference, fallback, and prioritization."""

    def test_model_a_code_only_evaluation(self):
        """Verify Model A operates without requiring provenance and flags buffer overflow."""
        vuln_c = "void bad(char *s) { char buf[8]; strcpy(buf, s); }"
        res = predict_model_a(vuln_c)
        self.assertIn("probability", res)
        self.assertIn("is_vulnerable", res)
        self.assertIn(res["cwe"], ["CWE-119", "CWE-120"])

    def test_model_b_checkpoint_validity(self):
        """Verify Phase 11 Model B checkpoint loads and contains valid coefficients."""
        ckpt_path = MODELS_DIR / "phase11_model_b_fusion_best.pkl"
        self.assertTrue(ckpt_path.exists(), "Phase 11 Model B checkpoint missing")
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        self.assertIn("clf_fusion", ckpt)
        self.assertIn("weights", ckpt)
        self.assertIn("bias", ckpt)
        # Logit weight must be positive (higher code risk increases final probability)
        self.assertGreater(ckpt["weights"]["logit_code"], 0.0)

    def test_missing_provenance_fallback(self):
        """Verify Model B exactly matches Model A when provenance is omitted."""
        code = "int safe_add(int a, int b) { return a + b; }"
        res_a = predict_model_a(code)
        res_b = predict_model_b(code, provenance=None)
        self.assertAlmostEqual(res_a["probability"], res_b["probability"], places=5)
        self.assertEqual(res_a["is_vulnerable"], res_b["is_vulnerable"])

    def test_ui_and_demo_compatibility(self):
        """Verify that presentation demos and side-by-side comparison execute without error."""
        demos = get_demo_examples()
        self.assertGreaterEqual(len(demos), 3)
        for name, d in demos.items():
            cmp_res = compare_models(d["code"], provenance=d.get("provenance", {}))
            self.assertIn("probability_delta_percentage_points", cmp_res)
            self.assertIn("comparison_narrative", cmp_res)
            self.assertTrue(cmp_res["source_code_identical"])


if __name__ == "__main__":
    unittest.main()
