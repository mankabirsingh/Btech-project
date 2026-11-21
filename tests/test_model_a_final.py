"""Unit tests for Model A Final packaged ensemble."""

import unittest
import numpy as np
from models.checkpoints.p5.model_a_final.predictor import predict, ModelAFinalEnsemble


class TestModelAFinal(unittest.TestCase):
    def setUp(self):
        self.sample_codes = [
            "int safe_add(int a, int b) { return a + b; }",
            "void vuln_buffer(char *input) { char buf[16]; strcpy(buf, input); }",
            "int factorial(int n) { if (n <= 1) return 1; return n * factorial(n - 1); }",
            "void null_deref(int *ptr) { *ptr = 42; }",
            "bool is_even(int x) { return (x % 2 == 0); }",
        ]

    def test_predict_single_and_batch(self):
        """Verify predict() accepts 5 sample functions and returns valid probabilities."""
        probs = predict(self.sample_codes, batch_size=8)
        self.assertEqual(len(probs), len(self.sample_codes))
        self.assertIsInstance(probs, np.ndarray)
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))

        # Check vulnerability ranking: buffer overflow should score higher than trivial safe_add
        print(f"Sample predictions: {probs}")
        self.assertGreater(probs[1], probs[0], "Buffer overflow should score higher than safe_add")


if __name__ == "__main__":
    unittest.main()
