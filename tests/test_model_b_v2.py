"""Unit test for Model B v2 predict(code, provenance=None)."""

import unittest
from pathlib import Path
import numpy as np


class TestModelBv2(unittest.TestCase):
    def setUp(self):
        self.sample_code = """
        #include <stdio.h>
        #include <string.h>
        void copy_data(const char *src) {
            char buf[64];
            strcpy(buf, src);
        }
        """

    def test_missing_provenance_equals_p_code(self):
        """When provenance is None, P_risk must equal P_code within 1e-6."""
        ckpt_file = Path("models/checkpoints/p10/model_b_v2/model_b_v2.pkl")
        if not ckpt_file.exists():
            self.skipTest("Model B v2 checkpoint not yet generated.")

        from models.checkpoints.p10.model_b_v2 import ModelBv2Predictor
        pred = ModelBv2Predictor(ckpt_file)
        feats = pred.compute_code_features(self.sample_code)
        p_code = pred.predict_code_risk(feats)
        p_risk = pred.predict(self.sample_code, provenance=None)

        self.assertAlmostEqual(p_risk, p_code, delta=1e-6)

    def test_with_provenance_returns_valid_prob(self):
        """When provenance is provided, P_risk must be a valid probability in [0, 1]."""
        ckpt_file = Path("models/checkpoints/p10/model_b_v2/model_b_v2.pkl")
        if not ckpt_file.exists():
            self.skipTest("Model B v2 checkpoint not yet generated.")

        from models.checkpoints.p10.model_b_v2 import ModelBv2Predictor
        pred = ModelBv2Predictor(ckpt_file)
        prov = {
            "author_type": "ai",
            "ai_involvement": 1.0,
            "temperature": 0.8,
            "prompt_tokens": 150,
            "candidate_tokens": 300,
        }
        p_risk = pred.predict(self.sample_code, provenance=prov)
        self.assertGreaterEqual(p_risk, 0.0)
        self.assertLessEqual(p_risk, 1.0)


if __name__ == "__main__":
    unittest.main()
