"""Statistical evaluation, hypothesis testing, confidence intervals, and metric calculation."""

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes exact Wilson score confidence interval for a binomial proportion k/n.

    Args:
        k: Number of successes / positive observations.
        n: Total number of observations.
        confidence: Confidence level (default 0.95).

    Returns:
        (lower_bound, upper_bound)
    """
    if n == 0:
        return 0.0, 0.0
    k = max(0, min(k, n))
    z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    p_hat = k / n
    denom = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2.0 * n)) / denom
    half_width = (z * np.sqrt((p_hat * (1.0 - p_hat) / n) + ((z**2) / (4.0 * (n**2))))) / denom
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def metrics_full(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, List[float]],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes a comprehensive set of classification and calibration metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probabilities for class 1.
        threshold: Decision threshold for converting probabilities to binary predictions.

    Returns:
        Dict with n, n_positive, n_negative, prevalence, majority_class_baseline,
        tp, fp, tn, fn, accuracy, balanced_accuracy, precision, recall, f1, mcc,
        auroc, auprc, fpr, fnr, brier_score, threshold.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_prob).astype(float)
    y_pred = (y_p >= threshold).astype(int)

    n = len(y_t)
    n_pos = int(np.sum(y_t == 1))
    n_neg = int(np.sum(y_t == 0))
    prevalence = float(n_pos / n) if n > 0 else 0.0
    majority_baseline = float(max(n_pos, n_neg) / n) if n > 0 else 0.0

    cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    acc = float(accuracy_score(y_t, y_pred)) if n > 0 else 0.0
    bal_acc = float(balanced_accuracy_score(y_t, y_pred)) if n > 0 else 0.0
    prec = float(precision_score(y_t, y_pred, zero_division=0))
    rec = float(recall_score(y_t, y_pred, zero_division=0))
    f1 = float(f1_score(y_t, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_t, y_pred)) if (len(np.unique(y_pred)) > 1 and len(np.unique(y_t)) > 1) else 0.0

    try:
        auroc = float(roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 0.50
    except Exception:
        auroc = 0.50

    try:
        auprc = float(average_precision_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else prevalence
    except Exception:
        auprc = prevalence

    brier = float(brier_score_loss(y_t, y_p)) if n > 0 else 0.0

    return {
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "prevalence": prevalence,
        "majority_class_baseline": majority_baseline,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc,
        "auroc": auroc,
        "auprc": auprc,
        "fpr": fpr,
        "fnr": fnr,
        "brier_score": brier,
        "threshold": float(threshold),
    }


def format_p_value(p: float, min_p: float = 0.0005) -> str:
    """Format p-value, representing values below resolution as '<min_p'."""
    if p < min_p:
        return f"<{min_p}"
    return f"{p:.4f}"


def bootstrap_ci(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, List[float]],
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray, float], float]] = None,
    threshold: float = 0.5,
    groups: Optional[Union[np.ndarray, List[Any]]] = None,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Union[Tuple[float, float], Dict[str, List[float]]]:
    """
    Computes bootstrap confidence intervals.
    If groups is None, uses stratified resampling by label.
    If groups is provided, uses cluster/group resampling by group ID.

    If metric_fn is None, computes 95% CIs for accuracy, mcc, auroc, and auprc.
    If metric_fn is given, computes (lower, upper) for that metric.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_prob).astype(float)
    n = len(y_t)
    rng = np.random.RandomState(seed)

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    # Pre-index groups or classes
    if groups is not None:
        groups_arr = np.asarray(groups)
        unique_groups = np.unique(groups_arr)
        group_to_indices = {g: np.where(groups_arr == g)[0] for g in unique_groups}
    else:
        pos_idx = np.where(y_t == 1)[0]
        neg_idx = np.where(y_t == 0)[0]

    def _sample_indices() -> np.ndarray:
        if groups is not None:
            chosen_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            sampled = np.concatenate([group_to_indices[g] for g in chosen_groups])
            return sampled
        else:
            b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True) if len(pos_idx) > 0 else np.array([], dtype=int)
            b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True) if len(neg_idx) > 0 else np.array([], dtype=int)
            return np.concatenate([b_pos, b_neg])

    if metric_fn is not None:
        scores = []
        for _ in range(n_resamples):
            b_idx = _sample_indices()
            if len(b_idx) == 0:
                continue
            y_b = y_t[b_idx]
            p_b = y_p[b_idx]
            try:
                score = metric_fn(y_b, p_b, threshold)
            except Exception:
                continue
            scores.append(score)
        if len(scores) == 0:
            return 0.0, 0.0
        return float(np.percentile(scores, lower_pct)), float(np.percentile(scores, upper_pct))

    # Default multi-metric CI dict
    accs, mccs, aurocs, auprcs = [], [], [], []
    for _ in range(n_resamples):
        b_idx = _sample_indices()
        if len(b_idx) == 0:
            continue
        y_b = y_t[b_idx]
        p_b = y_p[b_idx]
        pred_b = (p_b >= threshold).astype(int)

        accs.append(accuracy_score(y_b, pred_b))
        mccs.append(matthews_corrcoef(y_b, pred_b) if (len(np.unique(pred_b)) > 1 and len(np.unique(y_b)) > 1) else 0.0)
        aurocs.append(roc_auc_score(y_b, p_b) if len(np.unique(y_b)) > 1 else 0.5)
        auprcs.append(average_precision_score(y_b, p_b) if len(np.unique(y_b)) > 1 else float(np.mean(y_b)))

    return {
        "accuracy_ci_95": [float(np.percentile(accs, lower_pct)), float(np.percentile(accs, upper_pct))],
        "mcc_ci_95": [float(np.percentile(mccs, lower_pct)), float(np.percentile(mccs, upper_pct))],
        "auroc_ci_95": [float(np.percentile(aurocs, lower_pct)), float(np.percentile(aurocs, upper_pct))],
        "auprc_ci_95": [float(np.percentile(auprcs, lower_pct)), float(np.percentile(auprcs, upper_pct))],
    }


def paired_bootstrap_diff(
    y_true: Union[np.ndarray, List[int]],
    prob_a: Union[np.ndarray, List[float]],
    prob_b: Union[np.ndarray, List[float]],
    metric_fn: Callable[..., float],
    threshold_a: float = 0.5,
    threshold_b: float = 0.5,
    groups: Optional[Union[np.ndarray, List[Any]]] = None,
    extra_arrays: Optional[Dict[str, np.ndarray]] = None,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes paired bootstrap confidence interval and two-sided p-value for the difference:
    Delta = metric(Model B) - metric(Model A).

    Args:
        y_true: Ground truth binary labels.
        prob_a: Predicted probabilities for Model A.
        prob_b: Predicted probabilities for Model B.
        metric_fn: Function taking (y_sub, prob_sub, threshold, **extra) returning scalar float metric.
        threshold_a: Threshold for Model A.
        threshold_b: Threshold for Model B.
        groups: Cluster identifiers (e.g. task_id or commit_id).
        extra_arrays: Optional dictionary of additional aligned arrays (e.g. n_tokens).
        n_resamples: Number of bootstrap iterations.
        ci: Confidence level (default 0.95).
        seed: Random seed.

    Returns:
        Dict with score_a, score_b, diff_observed, ci_lower, ci_upper, p_value, significant_p05.
    """
    y_t = np.asarray(y_true).astype(int)
    p_a = np.asarray(prob_a).astype(float)
    p_b = np.asarray(prob_b).astype(float)

    extra_kwargs = {k: np.asarray(v) for k, v in extra_arrays.items()} if extra_arrays is not None else {}

    try:
        obs_a = float(metric_fn(y_t, p_a, threshold_a, **extra_kwargs))
        obs_b = float(metric_fn(y_t, p_b, threshold_b, **extra_kwargs))
    except TypeError:
        obs_a = float(metric_fn(y_t, p_a, threshold_a))
        obs_b = float(metric_fn(y_t, p_b, threshold_b))
    diff_obs = float(obs_b - obs_a)

    rng = np.random.RandomState(seed)
    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    if groups is not None:
        groups_arr = np.asarray(groups)
        unique_groups = np.unique(groups_arr)
        group_to_indices = {g: np.where(groups_arr == g)[0] for g in unique_groups}
    else:
        pos_idx = np.where(y_t == 1)[0]
        neg_idx = np.where(y_t == 0)[0]

    def _sample_indices() -> np.ndarray:
        if groups is not None:
            chosen_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            return np.concatenate([group_to_indices[g] for g in chosen_groups])
        else:
            b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True) if len(pos_idx) > 0 else np.array([], dtype=int)
            b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True) if len(neg_idx) > 0 else np.array([], dtype=int)
            return np.concatenate([b_pos, b_neg])

    diffs = []
    for _ in range(n_resamples):
        b_idx = _sample_indices()
        if len(b_idx) == 0:
            continue
        y_sub = y_t[b_idx]
        if len(np.unique(y_sub)) < 2:
            continue
        p_a_sub = p_a[b_idx]
        p_b_sub = p_b[b_idx]
        sub_extra = {k: v[b_idx] for k, v in extra_kwargs.items()} if extra_kwargs else {}
        try:
            m_a = metric_fn(y_sub, p_a_sub, threshold_a, **sub_extra)
            m_b = metric_fn(y_sub, p_b_sub, threshold_b, **sub_extra)
            diffs.append(m_b - m_a)
        except TypeError:
            try:
                m_a = metric_fn(y_sub, p_a_sub, threshold_a)
                m_b = metric_fn(y_sub, p_b_sub, threshold_b)
                diffs.append(m_b - m_a)
            except Exception:
                continue
        except Exception:
            continue

    if len(diffs) == 0:
        return {
            "diff_observed": diff_obs,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "p_value": 1.0,
            "significant_p05": False,
        }

    diffs_arr = np.asarray(diffs)
    ci_lower = float(np.percentile(diffs_arr, lower_pct))
    ci_upper = float(np.percentile(diffs_arr, upper_pct))

    # Two-sided empirical p-value for testing H0: diff == 0
    # Proportion of resamples on or across zero relative to the observed shift
    frac_le_zero = np.mean(diffs_arr <= 0)
    frac_ge_zero = np.mean(diffs_arr >= 0)
    p_val = float(min(1.0, 2.0 * min(frac_le_zero, frac_ge_zero)))

    return {
        "score_a": obs_a,
        "score_b": obs_b,
        "diff_observed": diff_obs,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
        "p_value_formatted": format_p_value(p_val, 1.0 / n_resamples),
        "significant_p05": bool(p_val < 0.05),
    }


def mcnemar(
    y_true: Union[np.ndarray, List[int]],
    pred_a: Union[np.ndarray, List[int]],
    pred_b: Union[np.ndarray, List[int]],
) -> Dict[str, Any]:
    """
    Computes McNemar test with Edwards continuity correction.

    Args:
        y_true: Ground truth binary labels.
        pred_a: Binary predictions from Model A.
        pred_b: Binary predictions from Model B.

    Returns:
        Dict with contingency_table, chi2, p_value, significant_p05.
    """
    y_t = np.asarray(y_true).astype(int)
    p_a = np.asarray(pred_a).astype(int)
    p_b = np.asarray(pred_b).astype(int)

    corr_a = (p_a == y_t)
    corr_b = (p_b == y_t)

    a = int(np.sum(corr_a & corr_b))
    b = int(np.sum(corr_a & (~corr_b)))  # Model A correct, Model B wrong
    c = int(np.sum((~corr_a) & corr_b))  # Model B correct, Model A wrong
    d = int(np.sum((~corr_a) & (~corr_b)))

    total_discordant = b + c
    if total_discordant > 0:
        # Edwards continuity correction
        chi2 = float(((abs(b - c) - 1.0) ** 2) / total_discordant)
        p_val = float(stats.chi2.sf(chi2, df=1))
    else:
        chi2 = 0.0
        p_val = 1.0

    return {
        "contingency_table": {
            "a_both_correct": a,
            "b_model_a_only": b,
            "c_model_b_only": c,
            "d_both_wrong": d,
        },
        "b": b,
        "c": c,
        "chi2": chi2,
        "p_value": p_val,
        "significant_p05": bool(p_val < 0.05),
    }


def delong(
    y_true: Union[np.ndarray, List[int]],
    prob_a: Union[np.ndarray, List[float]],
    prob_b: Union[np.ndarray, List[float]],
) -> Dict[str, Any]:
    """
    Computes paired DeLong test for two AUROC estimates on the same cohort.

    Args:
        y_true: Ground truth binary labels.
        prob_a: Predicted probabilities from Model A.
        prob_b: Predicted probabilities from Model B.

    Returns:
        Dict with auc_a, auc_b, delta, z_score, p_value, significant_p05.
    """
    y_t = np.asarray(y_true).astype(int)
    p_a = np.asarray(prob_a).astype(float)
    p_b = np.asarray(prob_b).astype(float)

    pos_idx = np.where(y_t == 1)[0]
    neg_idx = np.where(y_t == 0)[0]
    m = len(pos_idx)
    n = len(neg_idx)

    if m == 0 or n == 0:
        return {
            "auc_a": 0.5,
            "auc_b": 0.5,
            "delta": 0.0,
            "z_score": 0.0,
            "p_value": 1.0,
            "significant_p05": False,
        }

    # Vectorized placement values
    d1 = p_a[pos_idx][:, None] - p_a[neg_idx][None, :]
    psi1 = (d1 > 0).astype(float) + 0.5 * (d1 == 0).astype(float)
    v10 = np.mean(psi1, axis=1)
    v01 = np.mean(psi1, axis=0)
    auc1 = float(np.mean(v10))

    d2 = p_b[pos_idx][:, None] - p_b[neg_idx][None, :]
    psi2 = (d2 > 0).astype(float) + 0.5 * (d2 == 0).astype(float)
    v20 = np.mean(psi2, axis=1)
    v02 = np.mean(psi2, axis=0)
    auc2 = float(np.mean(v20))

    s10 = np.cov(np.vstack([v10, v20]))
    s01 = np.cov(np.vstack([v01, v02]))
    s = s10 / m + s01 / n

    var_diff = s[0, 0] + s[1, 1] - 2.0 * s[0, 1]
    diff = auc2 - auc1

    if var_diff <= 0:
        z = 0.0
        p_val = 1.0
    else:
        z = diff / np.sqrt(var_diff)
        p_val = float(2.0 * stats.norm.sf(abs(z)))

    return {
        "auc_a": auc1,
        "auc_b": auc2,
        "delta": float(diff),
        "z_score": float(z),
        "p_value": float(p_val),
        "significant_p05": bool(p_val < 0.05),
    }


def threshold_for_prevalence(
    p_val: Union[np.ndarray, List[float]],
    y_val: Union[np.ndarray, List[int]],
    target_prevalence: float = 0.30,
    metric: str = "f1",
) -> Dict[str, Any]:
    """
    Selects decision threshold by reweighting validation data to a target prevalence.

    Assigns weights to positive and negative samples such that:
      w_pos * n_pos / (w_pos * n_pos + w_neg * n_neg) = target_prevalence
    Then evaluates weighted candidate metrics to choose the optimal threshold.
    Also computes the Bayesian log-odds shift analytical threshold.

    Args:
        p_val: Predicted probabilities on validation set.
        y_val: Ground truth labels on validation set.
        target_prevalence: Desired prevalence (default 0.30).
        metric: Optimization criterion ('f1' or 'balanced_acc').

    Returns:
        Dict with optimal_threshold, bayes_threshold, validation_prevalence,
        target_prevalence, weighted_f1, weighted_accuracy.
    """
    y = np.asarray(y_val).astype(int)
    p = np.asarray(p_val).astype(float)

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    n_total = len(y)
    if n_pos == 0 or n_neg == 0:
        return {
            "optimal_threshold": 0.5,
            "bayes_threshold": 0.5,
            "validation_prevalence": float(n_pos / n_total) if n_total > 0 else 0.0,
            "target_prevalence": target_prevalence,
            "weighted_f1": 0.0,
            "weighted_accuracy": 0.0,
        }

    pi_val = n_pos / n_total
    pi_target = float(target_prevalence)

    # Weights so effective prevalence equals pi_target
    # w_neg = 1.0, w_pos = (pi_target / (1 - pi_target)) * (n_neg / n_pos)
    w_pos = (pi_target / (1.0 - pi_target)) * (n_neg / n_pos)
    weights = np.where(y == 1, w_pos, 1.0)
    w_total = np.sum(weights)

    # Analytical Bayesian threshold shift from 0.5
    # logit(p_target) = logit(p_val) + log(pi_target/(1-pi_target)) - log(pi_val/(1-pi_val))
    # Threshold where p_target = 0.5:
    log_odds_shift = np.log(pi_target / (1.0 - pi_target)) - np.log(pi_val / (1.0 - pi_val))
    bayes_threshold = float(1.0 / (1.0 + np.exp(log_odds_shift)))

    # Sweep candidate thresholds
    candidates = np.linspace(0.05, 0.95, 181)
    best_thr = 0.5
    best_score = -1.0
    best_f1 = 0.0
    best_acc = 0.0

    for thr in candidates:
        pred = (p >= thr).astype(int)
        tp_w = np.sum(weights * (y == 1) * (pred == 1))
        fp_w = np.sum(weights * (y == 0) * (pred == 1))
        fn_w = np.sum(weights * (y == 1) * (pred == 0))
        tn_w = np.sum(weights * (y == 0) * (pred == 0))

        prec_w = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else 0.0
        rec_w = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0.0
        f1_w = (2.0 * prec_w * rec_w) / (prec_w + rec_w) if (prec_w + rec_w) > 0 else 0.0
        acc_w = (tp_w + tn_w) / w_total if w_total > 0 else 0.0
        tpr_w = rec_w
        tnr_w = tn_w / (tn_w + fp_w) if (tn_w + fp_w) > 0 else 0.0
        bal_acc_w = 0.5 * (tpr_w + tnr_w)

        score = f1_w if metric == "f1" else bal_acc_w
        if score > best_score:
            best_score = score
            best_thr = thr
            best_f1 = f1_w
            best_acc = acc_w

    return {
        "optimal_threshold": float(round(best_thr, 4)),
        "bayes_threshold": float(round(bayes_threshold, 4)),
        "validation_prevalence": float(round(pi_val, 4)),
        "target_prevalence": float(target_prevalence),
        "weighted_f1": float(round(best_f1, 4)),
        "weighted_accuracy": float(round(best_acc, 4)),
        "optimization_metric": metric,
    }


def calc_length_stratified_auroc(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, List[float]],
    n_tokens: Union[np.ndarray, List[int]],
    bins: Optional[List[float]] = None,
    min_pos: int = 20,
    min_neg: int = 20,
    threshold: float = 0.5,
    **kwargs: Any,
) -> float:
    """
    Computes sample-weighted mean of within-bin AUROCs, skipping bins with <min_pos positives
    or <min_neg negatives. Returns float scalar.
    Accepts arbitrary kwargs / threshold for direct drop-in compatibility with paired_bootstrap_diff.
    """
    if bins is None:
        bins = [0.0, 256.0, 512.0, 1024.0, np.inf]
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_prob).astype(float)
    toks = np.asarray(n_tokens).astype(float)

    total_valid_samples = 0
    weighted_auc_sum = 0.0

    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        if i == 0:
            mask = (toks >= low) & (toks <= high)
        else:
            mask = (toks > low) & (toks <= high)

        y_b = y_t[mask]
        p_b = y_p[mask]
        n_pos = int(np.sum(y_b == 1))
        n_neg = int(np.sum(y_b == 0))
        n_b = len(y_b)

        if n_pos >= min_pos and n_neg >= min_neg:
            try:
                auc_b = float(roc_auc_score(y_b, p_b))
                weighted_auc_sum += auc_b * n_b
                total_valid_samples += n_b
            except Exception:
                pass

    if total_valid_samples == 0:
        if len(np.unique(y_t)) > 1:
            try:
                return float(roc_auc_score(y_t, y_p))
            except Exception:
                return 0.5
        return 0.5

    return float(weighted_auc_sum / total_valid_samples)


def length_stratified_auroc(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, List[float]],
    n_tokens: Union[np.ndarray, List[int]],
    bins: Optional[List[float]] = None,
    min_pos: int = 20,
    min_neg: int = 20,
    n_resamples: int = 2000,
    ci: float = 0.95,
    groups: Optional[Union[np.ndarray, List[Any]]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes sample-weighted mean of within-bin AUROCs, skipping bins with <20 positives
    or <20 negatives, alongside a bootstrap confidence interval.
    """
    if bins is None:
        bins = [0.0, 256.0, 512.0, 1024.0, np.inf]
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_prob).astype(float)
    toks = np.asarray(n_tokens).astype(float)

    bin_details = {}
    total_valid_samples = 0
    weighted_auc_sum = 0.0

    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        b_label = f"<= {int(high)}" if i == 0 else (f"> {int(low)}" if np.isinf(high) else f"{int(low)+1}-{int(high)}")
        if i == 0:
            mask = (toks >= low) & (toks <= high)
        else:
            mask = (toks > low) & (toks <= high)

        y_b = y_t[mask]
        p_b = y_p[mask]
        n_b = len(y_b)
        n_pos = int(np.sum(y_b == 1))
        n_neg = int(np.sum(y_b == 0))
        is_valid = (n_pos >= min_pos and n_neg >= min_neg)
        auc_b = None
        if is_valid:
            try:
                auc_b = float(roc_auc_score(y_b, p_b))
                weighted_auc_sum += auc_b * n_b
                total_valid_samples += n_b
            except Exception:
                is_valid = False

        bin_details[b_label] = {
            "n": n_b,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "prevalence": float(n_pos / n_b) if n_b > 0 else 0.0,
            "auroc": auc_b,
            "included": is_valid,
        }

    pt_est = float(weighted_auc_sum / total_valid_samples) if total_valid_samples > 0 else (
        float(roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 0.5
    )

    rng = np.random.RandomState(seed)
    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    if groups is not None:
        groups_arr = np.asarray(groups)
        unique_groups = np.unique(groups_arr)
        group_to_indices = {g: np.where(groups_arr == g)[0] for g in unique_groups}
    else:
        pos_idx = np.where(y_t == 1)[0]
        neg_idx = np.where(y_t == 0)[0]

    boot_scores = []
    for _ in range(n_resamples):
        if groups is not None:
            chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            b_idx = np.concatenate([group_to_indices[g] for g in chosen])
        else:
            b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True) if len(pos_idx) > 0 else np.array([], dtype=int)
            b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True) if len(neg_idx) > 0 else np.array([], dtype=int)
            b_idx = np.concatenate([b_pos, b_neg])

        if len(b_idx) == 0:
            continue
        val = calc_length_stratified_auroc(y_t[b_idx], y_p[b_idx], toks[b_idx], bins=bins, min_pos=min_pos, min_neg=min_neg)
        boot_scores.append(val)

    if len(boot_scores) == 0:
        ci_95 = [pt_est, pt_est]
    else:
        ci_95 = [float(np.percentile(boot_scores, lower_pct)), float(np.percentile(boot_scores, upper_pct))]

    return {
        "length_stratified_auroc": pt_est,
        "ci_95": ci_95,
        "bin_details": bin_details,
        "n_valid_samples": total_valid_samples,
    }


def threshold_at_fpr(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, List[float]],
    target_fpr: float = 0.10,
    target_prevalence: float = 0.30,
) -> Dict[str, Any]:
    """
    Selects decision threshold on predictions reweighted to target prevalence
    such that FPR <= target_fpr (giving highest recall under FPR constraint).
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    w_pos = (target_prevalence / (1.0 - target_prevalence)) * (n_neg / n_pos) if n_pos > 0 else 1.0
    weights = np.where(y == 1, w_pos, 1.0)

    neg_probs = p[y == 0]
    if len(neg_probs) > 0:
        quantile_thr = float(np.percentile(neg_probs, 100.0 * (1.0 - target_fpr)))
    else:
        quantile_thr = 0.5

    candidates = np.sort(np.unique(np.concatenate([
        np.linspace(0.01, 0.99, 990),
        [quantile_thr],
        np.percentile(neg_probs, np.linspace(85, 95, 101)) if len(neg_probs) > 0 else []
    ])))

    valid_thrs = []
    for thr in candidates:
        pred = (p >= thr).astype(int)
        fp = np.sum((y == 0) & (pred == 1))
        fpr = fp / n_neg if n_neg > 0 else 0.0
        if fpr <= target_fpr + 1e-9:
            valid_thrs.append((thr, fpr))

    if len(valid_thrs) > 0:
        max_valid_fpr = max(v[1] for v in valid_thrs)
        tied_thrs = [v[0] for v in valid_thrs if abs(v[1] - max_valid_fpr) < 1e-6]
        selected_thr = float(max(tied_thrs))
    else:
        selected_thr = float(quantile_thr)

    pred = (p >= selected_thr).astype(int)
    tp_w = np.sum(weights * (y == 1) * (pred == 1))
    fp_w = np.sum(weights * (y == 0) * (pred == 1))
    fn_w = np.sum(weights * (y == 1) * (pred == 0))
    tn_w = np.sum(weights * (y == 0) * (pred == 0))

    prec_w = float(tp_w / (tp_w + fp_w)) if (tp_w + fp_w) > 0 else 0.0
    rec_w = float(tp_w / (tp_w + fn_w)) if (tp_w + fn_w) > 0 else 0.0
    fpr_w = float(fp_w / (fp_w + tn_w)) if (fp_w + tn_w) > 0 else 0.0

    return {
        "threshold": float(round(selected_thr, 4)),
        "target_fpr": float(target_fpr),
        "target_prevalence": float(target_prevalence),
        "actual_fpr": float(round(fpr_w, 4)),
        "recall_at_target_fpr": float(round(rec_w, 4)),
        "precision_at_target_fpr": float(round(prec_w, 4)),
    }


def holm_adjust(p_values: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Computes Holm-Bonferroni family-wise error rate adjusted p-values.

    Args:
        p_values: List or 1D array of unadjusted p-values.

    Returns:
        Array of adjusted p-values in original order, clipped to [0, 1].
    """
    p_arr = np.asarray(p_values, dtype=float)
    m = len(p_arr)
    if m <= 1:
        return np.clip(p_arr, 0.0, 1.0)
    order = np.argsort(p_arr)
    sorted_p = p_arr[order]
    adj_sorted = np.empty(m, dtype=float)
    curr_max = 0.0
    for i in range(m):
        val = (m - i) * sorted_p[i]
        curr_max = max(curr_max, val)
        adj_sorted[i] = min(1.0, curr_max)
    adj_p = np.empty(m, dtype=float)
    adj_p[order] = adj_sorted
    return np.clip(adj_p, 0.0, 1.0)

