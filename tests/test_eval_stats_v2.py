"""Unit tests for src/eval/stats_v2.py."""

import unittest
import numpy as np
from src.eval.stats_v2 import (
    calc_lsauroc10,
    get_length_matched_pair_indices,
    calc_length_matched_auroc,
    alert_budget_metrics,
    paired_metrics_at_cutoff,
    paired_bootstrap_diff_v2,
)


class TestEvalStatsV2(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # 200 samples: 50 positive, 150 negative
        self.n = 200
        self.y_true = np.array([1] * 50 + [0] * 150)
        self.n_tokens = np.random.randint(50, 2000, size=self.n)
        # Probabilities correlated with ground truth
        self.y_prob_a = np.where(
            self.y_true == 1,
            np.random.uniform(0.4, 0.9, size=self.n),
            np.random.uniform(0.1, 0.6, size=self.n),
        )
        # Better model b
        self.y_prob_b = np.where(
            self.y_true == 1,
            np.random.uniform(0.6, 0.99, size=self.n),
            np.random.uniform(0.01, 0.4, size=self.n),
        )
        self.groups = np.random.choice(["grp1", "grp2", "grp3", "grp4"], size=self.n)

    def test_calc_lsauroc10(self):
        """Test 10-quantile length-stratified AUROC."""
        score = calc_lsauroc10(self.y_true, self.y_prob_a, self.n_tokens, min_pos=2, min_neg=2)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        # Model B should score higher than Model A
        score_b = calc_lsauroc10(self.y_true, self.y_prob_b, self.n_tokens, min_pos=2, min_neg=2)
        self.assertGreater(score_b, score)

    def test_length_matched_auroc(self):
        """Test length-matched pair AUROC calculation."""
        v_idx, s_idx = get_length_matched_pair_indices(self.y_true, self.n_tokens, max_log_diff=0.1)
        self.assertGreater(len(v_idx), 0)
        self.assertEqual(len(v_idx), len(s_idx))
        # Ensure all v_idx are true vulnerable, s_idx are safe
        self.assertTrue(np.all(self.y_true[v_idx] == 1))
        self.assertTrue(np.all(self.y_true[s_idx] == 0))
        # Ensure log token diff is <= 0.1
        log_v = np.log(self.n_tokens[v_idx])
        log_s = np.log(self.n_tokens[s_idx])
        self.assertTrue(np.all(np.abs(log_v - log_s) <= 0.10001))

        # Test calc_length_matched_auroc
        auc_m = calc_length_matched_auroc(self.y_true, self.y_prob_a, self.n_tokens, max_log_diff=0.1)
        self.assertGreaterEqual(auc_m, 0.0)
        self.assertLessEqual(auc_m, 1.0)

        auc_m_b = calc_length_matched_auroc(self.y_true, self.y_prob_b, self.n_tokens, max_log_diff=0.1)
        self.assertGreater(auc_m_b, auc_m)

    def test_alert_budget_metrics(self):
        """Test top 10% alert budget metrics."""
        res = alert_budget_metrics(self.y_true, self.y_prob_a, budget_fraction=0.10)
        self.assertEqual(res["budget_fraction"], 0.10)
        self.assertEqual(res["n_total"], 200)
        self.assertEqual(res["n_flagged"], 20)
        self.assertEqual(res["tp"] + res["fp"], 20)
        self.assertEqual(res["prevalence"], 50 / 200)
        self.assertGreaterEqual(res["precision"], 0.0)
        self.assertLessEqual(res["precision"], 1.0)
        self.assertGreaterEqual(res["recall"], 0.0)
        self.assertLessEqual(res["recall"], 1.0)
        if res["prevalence"] > 0:
            self.assertAlmostEqual(res["lift"], res["precision"] / res["prevalence"])

    def test_paired_metrics_at_cutoff(self):
        """Test P-C, P-V, P-B, P-R partition."""
        p_vuln = np.array([0.9, 0.8, 0.3, 0.2])
        p_fixed = np.array([0.2, 0.85, 0.1, 0.7])
        tau = 0.5
        # Pair 0: vuln=0.9 >= 0.5, fixed=0.2 < 0.5 -> P-C
        # Pair 1: vuln=0.8 >= 0.5, fixed=0.85 >= 0.5 -> P-V
        # Pair 2: vuln=0.3 < 0.5, fixed=0.1 < 0.5 -> P-B
        # Pair 3: vuln=0.2 < 0.5, fixed=0.7 >= 0.5 -> P-R
        res = paired_metrics_at_cutoff(p_vuln, p_fixed, tau=tau)
        self.assertAlmostEqual(res["P-C"], 0.25)
        self.assertAlmostEqual(res["P-V"], 0.25)
        self.assertAlmostEqual(res["P-B"], 0.25)
        self.assertAlmostEqual(res["P-R"], 0.25)
        self.assertAlmostEqual(res["P-C"] + res["P-V"] + res["P-B"] + res["P-R"], 1.0)
        # Pair correct:
        # pair 0: 0.9 > 0.2 (win 1.0)
        # pair 1: 0.8 < 0.85 (loss 0.0)
        # pair 2: 0.3 > 0.1 (win 1.0)
        # pair 3: 0.2 < 0.7 (loss 0.0)
        # total = 2 / 4 = 0.5
        self.assertAlmostEqual(res["pair_correct"], 0.5)

    def test_paired_bootstrap_diff_v2(self):
        """Test paired bootstrap difference."""
        from sklearn.metrics import roc_auc_score

        res = paired_bootstrap_diff_v2(
            self.y_true,
            self.y_prob_a,
            self.y_prob_b,
            metric_fn=roc_auc_score,
            groups=self.groups,
            n_resamples=200,
            seed=42,
        )
        self.assertIn("score_a", res)
        self.assertIn("score_b", res)
        self.assertIn("diff_observed", res)
        self.assertGreater(res["diff_observed"], 0.0)
        self.assertIn("ci_lower", res)
        self.assertIn("ci_upper", res)
        self.assertIn("raw_p_value", res)
        self.assertIn("p_value_formatted", res)
        self.assertIn("significant_p05", res)
        self.assertLessEqual(res["raw_p_value"], 1.0)
        self.assertGreaterEqual(res["raw_p_value"], 0.0)


if __name__ == "__main__":
    unittest.main()
