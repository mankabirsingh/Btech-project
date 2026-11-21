"""
Phase 14 Test Suite
===================
Tests for Phase 14 Model A (Modern C++ Structural Generalization) and
Model B (52-task / 208-unit Expanded Dataset and Interaction Fusion).
"""

import sys
import os
import json
import pickle
import hashlib
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.inference import (
    predict_model_a,
    predict_model_b,
    compare_models,
    set_active_checkpoints,
    get_active_checkpoints,
)
from src.models.cxx_features import EnhancedModernCxxStructuralExtractor, cxx_tokenize

DATA_DIR = PROJECT_ROOT / "data" / "phase14_model_b"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase14"


class TestPhase14DatasetIntegrity(unittest.TestCase):
    """Validates the Phase 14 expanded dataset integrity, balance, and zero leakage."""

    def test_partition_files_exist_and_valid_json(self):
        for split in ["train", "val", "test"]:
            path = DATA_DIR / f"model_b_p14_{split}.json"
            self.assertTrue(path.exists(), f"Missing split file: {path}")
            with open(path) as f:
                d = json.load(f)
            self.assertIn("records", d)
            self.assertGreater(len(d["records"]), 0)

    def test_sample_counts_and_balance(self):
        with open(DATA_DIR / "model_b_p14_train.json") as f:
            tr = json.load(f)["records"]
        with open(DATA_DIR / "model_b_p14_val.json") as f:
            val = json.load(f)["records"]
        with open(DATA_DIR / "model_b_p14_test.json") as f:
            te = json.load(f)["records"]

        self.assertEqual(len(tr), 144, "Train should have exactly 144 units (36 tasks)")
        self.assertEqual(len(val), 32, "Val should have exactly 32 units (8 tasks)")
        self.assertEqual(len(te), 32, "Test should have exactly 32 units (8 tasks)")
        total_units = len(tr) + len(val) + len(te)
        self.assertEqual(total_units, 208, "Total units must equal 208")

        for split_name, recs in [("train", tr), ("val", val), ("test", te)]:
            vuln_c = sum(1 for r in recs if r["vuln_label"] == 1)
            safe_c = sum(1 for r in recs if r["vuln_label"] == 0)
            self.assertEqual(vuln_c, safe_c, f"{split_name} must have exact 50/50 balance")

    def test_zero_task_leakage_across_partitions(self):
        with open(DATA_DIR / "model_b_p14_train.json") as f:
            tr_tasks = set(r["task_id"] for r in json.load(f)["records"])
        with open(DATA_DIR / "model_b_p14_val.json") as f:
            val_tasks = set(r["task_id"] for r in json.load(f)["records"])
        with open(DATA_DIR / "model_b_p14_test.json") as f:
            te_tasks = set(r["task_id"] for r in json.load(f)["records"])

        self.assertEqual(len(tr_tasks & val_tasks), 0, "Train and Val tasks must be disjoint")
        self.assertEqual(len(tr_tasks & te_tasks), 0, "Train and Test tasks must be disjoint")
        self.assertEqual(len(val_tasks & te_tasks), 0, "Val and Test tasks must be disjoint")

    def test_manifest_checksum_integrity(self):
        manifest_path = DATA_DIR / "manifest.json"
        self.assertTrue(manifest_path.exists())
        with open(manifest_path) as f:
            manifest = json.load(f)

        for split, info in manifest["splits"].items():
            file_path = DATA_DIR / info["file"]
            with open(file_path, "rb") as f:
                actual_sha = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(actual_sha, info["sha256"], f"SHA-256 mismatch on {split}")


class TestPhase14ModelInference(unittest.TestCase):
    """Validates Model A and Model B inference, C++ structural features, and fallbacks."""

    def test_cxx_structural_feature_extractor(self):
        extractor = EnhancedModernCxxStructuralExtractor()
        code_cpp = """
        #include <iostream>
        #include <memory>
        namespace envoy {
            class Connection {
                virtual void close() override;
                std::unique_ptr<Buffer> buf_;
            };
        }
        """
        feats = extractor.transform([code_cpp])
        self.assertEqual(feats.shape, (1, 24), "Must produce exactly 24 dimensions")
        # is_cpp should be 1.0
        self.assertEqual(feats[0, 22], 1.0, "is_cpp flag must be 1.0 for C++ code")

    def test_phase14_checkpoints_exist_and_loadable(self):
        ckpts = [
            "phase14_cxx_structural_svm.pkl",
            "phase14_codebert_cxx_hybrid.pkl",
            "phase14_model_b_fusion_best.pkl"
        ]
        for c in ckpts:
            path = MODELS_DIR / c
            self.assertTrue(path.exists(), f"Missing checkpoint: {path}")
            with open(path, "rb") as f:
                obj = pickle.load(f)
            self.assertIsInstance(obj, dict)

    def test_model_a_provenance_invariance(self):
        sample_code = """
        #include <string.h>
        void process(const char *in) {
            char buf[16];
            strcpy(buf, in);
        }
        """
        res_default = predict_model_a(sample_code)
        
        # Provenance variation must NOT affect Model A
        for prov_type in ["human", "ai", "human_edit_of_ai", "ai_edit_of_human", "unknown"]:
            prov_dict = {"provenance_type": prov_type, "ai_involvement": 0.8}
            comp = compare_models(sample_code, provenance=prov_dict)
            self.assertEqual(
                comp["model_a"]["probability"],
                res_default["probability"],
                "Model A must be strictly invariant to provenance inputs"
            )

    def test_model_b_unknown_provenance_fallback(self):
        sample_code = """
        #include <stdio.h>
        int safe_add(int a, int b) {
            return a + b;
        }
        """
        res_a = predict_model_a(sample_code)
        res_b_fallback = predict_model_b(sample_code, provenance={"provenance_type": "unknown"})
        
        self.assertFalse(res_b_fallback["has_provenance"])
        self.assertAlmostEqual(
            res_b_fallback["probability"],
            res_a["probability"],
            places=5,
            msg="Model B must cleanly fall back to Model A probability when provenance is unknown"
        )

    def test_dynamic_checkpoint_switching(self):
        # Verify switching to Phase 14 Model B checkpoint
        set_active_checkpoints(model_b_name="phase14_model_b_fusion_best.pkl")
        active = get_active_checkpoints()
        self.assertEqual(active["model_b"], "phase14_model_b_fusion_best.pkl")

        # Test inference with Phase 14 checkpoint
        sample_code = "void test() { int x = 0; }"
        res_b = predict_model_b(sample_code, provenance={"provenance_type": "ai", "ai_involvement": 0.9})
        self.assertIn("probability", res_b)
        self.assertIn("review_priority", res_b)

        # Restore production default
        set_active_checkpoints(model_b_name="model_b_provenance_fusion.pkl")
        active_restored = get_active_checkpoints()
        self.assertEqual(active_restored["model_b"], "model_b_provenance_fusion.pkl")


if __name__ == "__main__":
    unittest.main()
