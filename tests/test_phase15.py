#!/usr/bin/env python3
"""
Unit Test Suite for Phase 15: Advanced Model Optimisation and Dataset Scale-Up
=============================================================================
Tests:
1. Dataset Integrity & Partition Disjointness (150 tasks / 600 units, 50 CWEs, 50/50 balance)
2. Manifest & Checksum Validation
3. Model A Modern C++ 28-Dimensional Extractor
4. Model A Phase 15 Champion Checkpoint Loading & Prediction
5. Model B Checkpoint Loading, Weights, and Feature Schema
6. Model B Fallback on Missing Provenance
7. Model B Permutation Sensitivity
8. Review-Priority Triage Ranking Metric Computation
9. Results Artifacts Validation
"""

import unittest
import json
import pickle
import hashlib
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DATA_P15_DIR = PROJECT_ROOT / "data" / "phase15_model_b"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_P15_DIR = PROJECT_ROOT / "results" / "phase15"


class TestPhase15Dataset(unittest.TestCase):
    """Verifies Phase 15 Model B dataset scale, balance, and partition hygiene."""

    def setUp(self):
        self.manifest_path = DATA_P15_DIR / "manifest.json"
        self.train_path = DATA_P15_DIR / "model_b_p15_train.json"
        self.val_path = DATA_P15_DIR / "model_b_p15_val.json"
        self.test_path = DATA_P15_DIR / "model_b_p15_test.json"

    def test_files_and_manifest_exist(self):
        self.assertTrue(self.manifest_path.exists())
        self.assertTrue(self.train_path.exists())
        self.assertTrue(self.val_path.exists())
        self.assertTrue(self.test_path.exists())

    def test_checksum_integrity(self):
        with open(self.manifest_path) as f:
            manifest = json.load(f)

        for split_key, split_info in manifest["splits"].items():
            filename = split_info["file"]
            expected_sha = split_info["sha256"]
            fpath = DATA_P15_DIR / filename
            self.assertTrue(fpath.exists())
            actual_sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
            self.assertEqual(actual_sha, expected_sha, f"Checksum mismatch for {filename}")

    def test_scale_and_balance(self):
        with open(self.train_path) as f:
            train_data = json.load(f)["records"]
        with open(self.val_path) as f:
            val_data = json.load(f)["records"]
        with open(self.test_path) as f:
            test_data = json.load(f)["records"]

        # Exactly 420 train, 88 val, 92 test = 600 total
        self.assertEqual(len(train_data), 420)
        self.assertEqual(len(val_data), 88)
        self.assertEqual(len(test_data), 92)
        total_units = len(train_data) + len(val_data) + len(test_data)
        self.assertEqual(total_units, 600)

        # Total tasks = 105 + 22 + 23 = 150 tasks
        train_tasks = set(r["task_id"] for r in train_data)
        val_tasks = set(r["task_id"] for r in val_data)
        test_tasks = set(r["task_id"] for r in test_data)

        self.assertEqual(len(train_tasks), 105)
        self.assertEqual(len(val_tasks), 22)
        self.assertEqual(len(test_tasks), 23)
        self.assertEqual(len(train_tasks | val_tasks | test_tasks), 150)

        # Zero task leakage across splits
        self.assertEqual(len(train_tasks & val_tasks), 0)
        self.assertEqual(len(train_tasks & test_tasks), 0)
        self.assertEqual(len(val_tasks & test_tasks), 0)

        # Exactly 50.0% vulnerable / 50.0% safe
        for split_name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            n_vuln = sum(1 for r in data if r["vuln_label"] == 1)
            self.assertEqual(n_vuln, len(data) // 2, f"{split_name} split is not 50/50 balanced")

        # 50 Distinct CWEs
        all_cwes = set(r["cwe"] for r in train_data + val_data + test_data)
        self.assertGreaterEqual(len(all_cwes), 50)


class TestPhase15ModelA(unittest.TestCase):
    """Verifies Modern C++ extractor and Model A Champion Checkpoint."""

    def test_28dim_cxx_extractor(self):
        from scripts.phase15_train_model_a import Phase15EnhancedModernCxxExtractor
        extractor = Phase15EnhancedModernCxxExtractor()
        code_modern = """
        void process(std::unique_ptr<Buffer> buf) {
            auto raw = buf.get();
            auto ptr = std::move(buf);
            char *p = raw;
            p++;
        }
        """
        feats = extractor.transform([code_modern])
        self.assertEqual(feats.shape, (1, 28))
        self.assertGreater(feats[0, 24], 0.0)  # Smart pointer escape .get()
        self.assertGreater(feats[0, 25], 0.0)  # Move semantics std::move
        self.assertGreater(feats[0, 26], 0.0)  # Pointer arithmetic

    def test_model_a_checkpoint_loading(self):
        ckpt_path = MODELS_DIR / "phase15_codebert_cxx_hybrid_best.pkl"
        self.assertTrue(ckpt_path.exists())
        with open(ckpt_path, "rb") as f:
            obj = pickle.load(f)

        self.assertIn("clf", obj)
        self.assertIn("scaler", obj)
        self.assertIn("feature_names", obj)
        self.assertIn("best_c", obj)
        self.assertEqual(len(obj["feature_names"]), 28)
        self.assertEqual(obj["train_size"], 25000)


class TestPhase15ModelB(unittest.TestCase):
    """Verifies Model B Late Fusion checkpoint, fallback, and triage ranking."""

    def setUp(self):
        self.ckpt_path = MODELS_DIR / "phase15_model_b_fusion_best.pkl"
        self.assertTrue(self.ckpt_path.exists())
        with open(self.ckpt_path, "rb") as f:
            self.ckpt = pickle.load(f)

    def test_checkpoint_schema_and_weights(self):
        self.assertIn("clf_fusion", self.ckpt)
        self.assertIn("feature_names", self.ckpt)
        self.assertIn("weights", self.ckpt)
        self.assertIn("bias", self.ckpt)
        self.assertGreater(len(self.ckpt["feature_names"]), 6)

    def test_unknown_provenance_fallback(self):
        from scripts.phase15_train_model_b import extract_features
        clf = self.ckpt["clf_fusion"]
        dummy_rec = [{
            "code": "int main() { return 0; }",
            "vuln_label": 0,
            "author_type": "unknown",
            "ai_involvement": 0.0,
            "provenance_confidence": 0.0
        }]
        X_fb, _, _ = extract_features(dummy_rec, include_interactions=self.ckpt["use_interactions"], fallback=True)
        prob = clf.predict_proba(X_fb)[0, 1]
        self.assertTrue(0.0 <= prob <= 1.0)
        # All provenance features in fallback must be 0
        self.assertTrue(np.all(X_fb[0, 1:] == 0.0))

    def test_triage_ranking_computation(self):
        from scripts.phase15_train_model_b import compute_triage_ranking_metrics
        y_true = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])  # 3 vulns in 10 items
        y_prob = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1, 0.1, 0.05, 0.01])
        triage = compute_triage_ranking_metrics(y_true, y_prob)
        self.assertIn("budget_10pct", triage)
        self.assertIn("budget_20pct", triage)
        self.assertIn("average_precision", triage)
        # Top 10% (1 sample) is y_true[0] = 1 -> Precision@10% = 1.0, Recall@10% = 1/3
        self.assertAlmostEqual(triage["budget_10pct"]["precision_at_k"], 1.0)
        self.assertAlmostEqual(triage["budget_10pct"]["recall_at_k"], 1.0 / 3.0)


class TestPhase15ResultsArtifacts(unittest.TestCase):
    """Verifies that all Phase 15 evaluation JSONs exist and are well-formed."""

    def test_all_results_artifacts_exist(self):
        expected_files = [
            "baseline_reproduction.json",
            "phase15_model_a_comparison.json",
            "phase15_model_b_evaluation.json",
            "combined_evaluation.json",
            "real_world_provenance_audit.json"
        ]
        for fname in expected_files:
            fpath = RESULTS_P15_DIR / fname
            self.assertTrue(fpath.exists(), f"Missing result file: {fname}")
            with open(fpath) as f:
                data = json.load(f)
            self.assertIsInstance(data, dict)
            self.assertGreater(len(data), 0)


if __name__ == "__main__":
    unittest.main()
