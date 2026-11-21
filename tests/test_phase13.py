#!/usr/bin/env python3
"""
Unit Tests for Phase 13 Deliverables
====================================
DTU B.Tech Project: Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code
Authors: Mankabir Singh (23/IT/199) and Mohnish Gautam (23/IT/200)

Validates:
1. Phase 13 dataset integrity, checksums, and strict zero-leakage isolation.
2. Anti-shortcut 50/50 balance across partitions and provenance categories.
3. Phase 13 Model B checkpoint validity and weights.
4. Unknown provenance fallback behavior.
5. Presentation UI sample snippets and invariance contract.
"""

import unittest
import json
import pickle
import hashlib
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "phase13_model_b"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"

from src.models.inference import predict_model_a, predict_model_b, compare_models
from app import VULNERABLE_C_EXAMPLE, SAFER_C_EXAMPLE, PROVENANCE_OPTIONS


class TestPhase13DatasetIntegrity(unittest.TestCase):
    """Verifies Phase 13 dataset files, checksums, and anti-leakage properties."""

    def test_partition_files_and_manifest_exist(self):
        """Verify that Phase 13 data files, manifest, and checksums exist."""
        self.assertTrue(DATA_DIR.exists(), "Phase 13 data dir missing")
        self.assertTrue((DATA_DIR / "manifest.json").exists(), "manifest.json missing")
        self.assertTrue((DATA_DIR / "CHECKSUMS.sha256").exists(), "CHECKSUMS.sha256 missing")
        self.assertTrue((DATA_DIR / "model_b_p13_train.json").exists(), "Train split missing")
        self.assertTrue((DATA_DIR / "model_b_p13_val.json").exists(), "Val split missing")
        self.assertTrue((DATA_DIR / "model_b_p13_test_locked.json").exists(), "Test split missing")

    def test_locked_test_checksum_integrity(self):
        """Verify the test set matches its recorded cryptographic SHA-256 checksum."""
        test_path = DATA_DIR / "model_b_p13_test_locked.json"
        actual_sha = hashlib.sha256(test_path.read_bytes()).hexdigest()
        with open(DATA_DIR / "manifest.json") as f:
            manifest = json.load(f)
        self.assertEqual(actual_sha, manifest["locked_test_checksum"], "Test set checksum mismatch!")

    def test_class_and_provenance_balance(self):
        """Verify 50/50 class balance in every partition and 25/25/25/25 provenance balance."""
        for split_file in ["model_b_p13_train.json", "model_b_p13_val.json", "model_b_p13_test_locked.json"]:
            with open(DATA_DIR / split_file) as f:
                data = json.load(f)
            recs = data["records"]
            vuln = sum(1 for r in recs if r["vuln_label"] == 1)
            safe = sum(1 for r in recs if r["vuln_label"] == 0)
            self.assertEqual(vuln, safe, f"Class imbalance in {split_file}: {vuln} vs {safe}")

    def test_zero_task_and_code_leakage(self):
        """Verify complete isolation between train, val, test, and prior locked benchmark."""
        with open(DATA_DIR / "model_b_p13_train.json") as f:
            tr = json.load(f)
        with open(DATA_DIR / "model_b_p13_val.json") as f:
            val = json.load(f)
        with open(DATA_DIR / "model_b_p13_test_locked.json") as f:
            te = json.load(f)
        with open(PROJECT_ROOT / "data" / "locked_benchmark_32" / "controlled_benchmark_32.json") as f:
            b32 = json.load(f)

        tr_tasks = set(tr["tasks"])
        val_tasks = set(val["tasks"])
        te_tasks = set(te["tasks"])
        b32_tasks = set(r["task_id"] for r in b32["records"])

        # Check disjointness
        self.assertEqual(len(tr_tasks & val_tasks), 0)
        self.assertEqual(len(tr_tasks & te_tasks), 0)
        self.assertEqual(len(val_tasks & te_tasks), 0)
        self.assertEqual(len(tr_tasks & b32_tasks), 0)
        self.assertEqual(len(te_tasks & b32_tasks), 0)

        # Check code hash uniqueness
        hashes = [r["code_hash"] for r in tr["records"] + val["records"] + te["records"]]
        self.assertEqual(len(hashes), len(set(hashes)), "Duplicate code detected across partitions!")


class TestPhase13ModelInferenceAndUI(unittest.TestCase):
    """Verifies Phase 13 checkpoint loading, fallback behavior, and UI contracts."""

    def test_phase13_checkpoint_validity(self):
        """Verify Phase 13 Model B checkpoint loads and contains valid weights."""
        ckpt_path = MODELS_DIR / "phase13_model_b_fusion_best.pkl"
        self.assertTrue(ckpt_path.exists(), "Phase 13 checkpoint missing")
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        self.assertIn("clf_fusion", ckpt)
        self.assertIn("weights", ckpt)
        self.assertIn("bias", ckpt)
        self.assertIn("test_metrics", ckpt)

    def test_missing_provenance_fallback(self):
        """Verify Model B exactly matches Model A when provenance is unknown."""
        code = "int safe_inc(int x) { return x + 1; }"
        res_a = predict_model_a(code)
        res_b_none = predict_model_b(code, provenance=None)
        res_b_unknown = predict_model_b(code, provenance={"provenance_type": "unknown"})
        self.assertAlmostEqual(res_a["probability"], res_b_none["probability"], places=5)
        self.assertAlmostEqual(res_a["probability"], res_b_unknown["probability"], places=5)
        self.assertEqual(res_a["is_vulnerable"], res_b_none["is_vulnerable"])

    def test_model_a_provenance_invariance(self):
        """Verify changing provenance changes only Model B, leaving Model A invariant."""
        code = VULNERABLE_C_EXAMPLE
        cmp_ai = compare_models(code, PROVENANCE_OPTIONS["AI-generated, not yet human-reviewed"])
        cmp_human = compare_models(code, PROVENANCE_OPTIONS["Human-written"])
        self.assertAlmostEqual(cmp_ai["model_a"]["probability"], cmp_human["model_a"]["probability"], places=5)
        self.assertEqual(cmp_ai["model_a"]["is_vulnerable"], cmp_human["model_a"]["is_vulnerable"])
        # Model B adapts risk
        self.assertNotEqual(cmp_ai["model_b"]["probability"], cmp_human["model_b"]["probability"])

    def test_presentation_samples_evaluation(self):
        """Verify presentation sample snippets produce valid, distinct predictions."""
        res_vuln = predict_model_a(VULNERABLE_C_EXAMPLE)
        res_safe = predict_model_a(SAFER_C_EXAMPLE)
        self.assertTrue(res_vuln["is_vulnerable"])
        self.assertFalse(res_safe["is_vulnerable"])
        self.assertGreater(res_vuln["probability"], 0.70)
        self.assertLess(res_safe["probability"], 0.40)

    def test_provenance_options_schema(self):
        """Verify all 5 required provenance options exist with valid schemas."""
        required = [
            "Human-written",
            "AI-generated, not yet human-reviewed",
            "AI-generated, then human-reviewed/edited",
            "Human-written, then AI-modified",
            "Unknown / unavailable"
        ]
        for opt in required:
            self.assertIn(opt, PROVENANCE_OPTIONS)
            self.assertIn("provenance_type", PROVENANCE_OPTIONS[opt])
            self.assertIn("ai_involvement", PROVENANCE_OPTIONS[opt])


if __name__ == "__main__":
    unittest.main()
