"""Unit tests for src/eval/stats.py."""

import unittest
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.eval.stats import (
    wilson_ci,
    metrics_full,
    bootstrap_ci,
    paired_bootstrap_diff,
    mcnemar,
    delong,
    threshold_for_prevalence,
    calc_length_stratified_auroc,
    length_stratified_auroc,
    threshold_at_fpr,
    holm_adjust,
)


class TestEvalStats(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # 100 samples with balanced classes
        self.y_true = np.array([1] * 35 + [0] * 65)
        # Probabilities correlated with ground truth
        self.y_prob = np.where(self.y_true == 1, np.random.uniform(0.4, 0.95, size=len(self.y_true)),
                                               np.random.uniform(0.05, 0.6, size=len(self.y_true)))
        self.y_pred = (self.y_prob >= 0.5).astype(int)

    def test_wilson_ci(self):
        """Test Wilson score CI calculation."""
        # Known case: 50 / 100 -> ~ 0.4038 to 0.5962
        low, high = wilson_ci(50, 100, 0.95)
        self.assertAlmostEqual(low, 0.4038, places=3)
        self.assertAlmostEqual(high, 0.5962, places=3)

        # Boundary checks
        low0, high0 = wilson_ci(0, 10, 0.95)
        self.assertEqual(low0, 0.0)
        self.assertGreater(high0, 0.0)

        low1, high1 = wilson_ci(10, 10, 0.95)
        self.assertLess(low1, 1.0)
        self.assertAlmostEqual(high1, 1.0, places=5)

        # n = 0
        self.assertEqual(wilson_ci(0, 0), (0.0, 0.0))

    def test_metrics_full_matches_sklearn(self):
        """Verify metrics_full returns exact results identical to scikit-learn."""
        m = metrics_full(self.y_true, self.y_prob, threshold=0.5)

        self.assertEqual(m["n"], len(self.y_true))
        self.assertEqual(m["n_positive"], 35)
        self.assertEqual(m["n_negative"], 65)
        self.assertAlmostEqual(m["prevalence"], 0.35)
        self.assertAlmostEqual(m["majority_class_baseline"], 0.65)

        expected_acc = accuracy_score(self.y_true, self.y_pred)
        self.assertAlmostEqual(m["accuracy"], expected_acc)

        expected_bal_acc = balanced_accuracy_score(self.y_true, self.y_pred)
        self.assertAlmostEqual(m["balanced_accuracy"], expected_bal_acc)

        expected_prec = precision_score(self.y_true, self.y_pred, zero_division=0)
        self.assertAlmostEqual(m["precision"], expected_prec)

        expected_rec = recall_score(self.y_true, self.y_pred, zero_division=0)
        self.assertAlmostEqual(m["recall"], expected_rec)

        expected_f1 = f1_score(self.y_true, self.y_pred, zero_division=0)
        self.assertAlmostEqual(m["f1"], expected_f1)

        expected_mcc = matthews_corrcoef(self.y_true, self.y_pred)
        self.assertAlmostEqual(m["mcc"], expected_mcc)

        expected_auroc = roc_auc_score(self.y_true, self.y_prob)
        self.assertAlmostEqual(m["auroc"], expected_auroc)

        expected_auprc = average_precision_score(self.y_true, self.y_prob)
        self.assertAlmostEqual(m["auprc"], expected_auprc)

        expected_brier = brier_score_loss(self.y_true, self.y_prob)
        self.assertAlmostEqual(m["brier_score"], expected_brier)

    def test_bootstrap_ci_stratified(self):
        """Verify stratified bootstrap CI computation."""
        ci_dict = bootstrap_ci(self.y_true, self.y_prob, n_resamples=200, seed=42)
        for metric in ["accuracy_ci_95", "mcc_ci_95", "auroc_ci_95", "auprc_ci_95"]:
            self.assertIn(metric, ci_dict)
            low, high = ci_dict[metric]
            self.assertLessEqual(low, high)
            self.assertGreaterEqual(low, -1.0)
            self.assertLessEqual(high, 1.0)

    def test_bootstrap_ci_grouped(self):
        """Verify grouped bootstrap CI computation."""
        # Assign 10 groups of 10 samples each
        groups = np.repeat(np.arange(10), 10)
        ci_dict = bootstrap_ci(self.y_true, self.y_prob, groups=groups, n_resamples=200, seed=42)
        self.assertIn("auroc_ci_95", ci_dict)
        low, high = ci_dict["auroc_ci_95"]
        self.assertLessEqual(low, high)

    def test_paired_bootstrap_diff(self):
        """Verify paired bootstrap difference between two models."""
        prob_a = self.y_prob
        # Model B is slightly improved
        prob_b = np.clip(self.y_prob + np.where(self.y_true == 1, 0.1, -0.1), 0.0, 1.0)

        def acc_metric(y, p, thr):
            return accuracy_score(y, (p >= thr).astype(int))

        res = paired_bootstrap_diff(self.y_true, prob_a, prob_b, acc_metric, n_resamples=200, seed=42)
        self.assertIn("diff_observed", res)
        self.assertIn("ci_lower", res)
        self.assertIn("ci_upper", res)
        self.assertIn("p_value", res)
        self.assertLessEqual(res["ci_lower"], res["ci_upper"])

    def test_mcnemar_identical(self):
        """Identical predictions should give zero chi2 and p-value 1.0."""
        res = mcnemar(self.y_true, self.y_pred, self.y_pred)
        self.assertEqual(res["chi2"], 0.0)
        self.assertEqual(res["p_value"], 1.0)
        self.assertFalse(res["significant_p05"])

    def test_delong_identical(self):
        """Identical probabilities should give delta 0.0 and p-value 1.0."""
        res = delong(self.y_true, self.y_prob, self.y_prob)
        self.assertEqual(res["delta"], 0.0)
        self.assertEqual(res["z_score"], 0.0)
        self.assertEqual(res["p_value"], 1.0)

    def test_threshold_for_prevalence(self):
        """Verify threshold adaptation to target prevalence."""
        # When target prevalence is higher than validation, optimal threshold shifts
        res = threshold_for_prevalence(self.y_prob, self.y_true, target_prevalence=0.50)
        self.assertIn("optimal_threshold", res)
        self.assertIn("bayes_threshold", res)
        self.assertGreater(res["optimal_threshold"], 0.0)
        self.assertLess(res["optimal_threshold"], 1.0)

    def test_length_stratified_auroc(self):
        """Verify length-stratified AUROC calculation and confidence intervals."""
        np.random.seed(42)
        n = 200
        # Create 2 bins each with 50 pos and 50 neg
        # Bin 1: tokens 100-200 (<= 256)
        # Bin 2: tokens 300-400 (257-512)
        y = np.array([1]*50 + [0]*50 + [1]*50 + [0]*50)
        tokens = np.array([150]*100 + [350]*100)
        # In Bin 1: strong model
        p1 = np.where(y[:100] == 1, 0.8, 0.2)
        # In Bin 2: moderate model
        p2 = np.where(y[100:] == 1, 0.6, 0.4)
        p = np.concatenate([p1, p2])

        # Point estimate
        scalar_auc = calc_length_stratified_auroc(y, p, tokens, min_pos=20, min_neg=20)
        self.assertGreater(scalar_auc, 0.90)

        # Full dict with bootstrap CI
        res = length_stratified_auroc(y, p, tokens, n_resamples=100, seed=42)
        self.assertIn("length_stratified_auroc", res)
        self.assertIn("ci_95", res)
        self.assertIn("bin_details", res)
        self.assertEqual(res["n_valid_samples"], 200)
        self.assertLessEqual(res["ci_95"][0], res["ci_95"][1])

    def test_threshold_at_fpr(self):
        """Verify threshold selection at target FPR and prevalence."""
        np.random.seed(42)
        y = np.array([1] * 500 + [0] * 500)
        p = np.where(y == 1, np.random.beta(5, 2, size=len(y)), np.random.beta(2, 5, size=len(y)))
        res = threshold_at_fpr(y, p, target_fpr=0.10, target_prevalence=0.30)
        self.assertIn("threshold", res)
        self.assertIn("actual_fpr", res)
        self.assertIn("recall_at_target_fpr", res)
        self.assertIn("precision_at_target_fpr", res)
        self.assertLessEqual(res["actual_fpr"], 0.105)
        self.assertGreater(res["recall_at_target_fpr"], 0.5)

    def test_holm_adjust(self):
        """Verify Holm-Bonferroni p-value adjustment against statsmodels."""
        from statsmodels.stats.multitest import multipletests
        pvals = [0.001, 0.01, 0.03, 0.04, 0.05, 0.10, 0.50]
        expected = multipletests(pvals, method="holm")[1]
        actual = holm_adjust(pvals)
        np.testing.assert_allclose(actual, expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
