"""Stats V2 module: Extended metrics and statistical testing without modifying stats.py.

Includes:
- lsauroc10: Length-stratified AUROC with 10 length-quantile bins (computed on dev).
- length_matched_auroc: AUROC over (vulnerable, safe) pairs whose |log n_tokens diff| <= 0.1.
- alert_budget_metrics: Top-10% alert budget evaluation (precision, recall, lift).
- paired_metrics_at_budget: P-C, P-V, P-B, P-R evaluated at alert-budget operating point.
- paired_bootstrap_diff_v2: Paired difference hypothesis testing returning raw p-values and CIs.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from src.eval.stats import format_p_value, holm_adjust, bootstrap_ci, calc_length_stratified_auroc


def calc_lsauroc10(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, List[float]],
    n_tokens: Union[np.ndarray, List[int]],
    quantile_edges: Optional[np.ndarray] = None,
    min_pos: int = 10,
    min_neg: int = 10,
    **kwargs: Any,
) -> float:
    """Computes sample-weighted mean of within-bin AUROCs across 10 quantile bins."""
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_prob).astype(float)
    toks = np.asarray(n_tokens).astype(float)

    if quantile_edges is None:
        edges = np.quantile(toks, np.linspace(0, 1, 11))
        edges[0] = 0.0
        edges[-1] = np.inf
    else:
        edges = np.asarray(quantile_edges)

    total_valid_samples = 0
    weighted_auc_sum = 0.0

    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        mask = (toks >= low) & (toks <= high) if i == 0 else (toks > low) & (toks <= high)

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


def get_length_matched_pair_indices(
    y_true: np.ndarray,
    n_tokens: np.ndarray,
    max_log_diff: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds all (vulnerable, safe) index pairs (i, j) where |log(n_tokens[i]) - log(n_tokens[j])| <= max_log_diff.
    Returns:
        v_indices, s_indices (1D integer arrays of equal length)
    """
    v_idx = np.where(y_true == 1)[0]
    s_idx = np.where(y_true == 0)[0]

    log_tok = np.log(np.maximum(1.0, n_tokens.astype(float)))
    v_log = log_tok[v_idx]
    s_log = log_tok[s_idx]

    # Pairwise absolute difference matrix
    diff_mat = np.abs(v_log[:, None] - s_log[None, :])
    v_match, s_match = np.where(diff_mat <= max_log_diff)

    return v_idx[v_match], s_idx[s_match]


def calc_length_matched_auroc(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, List[float]],
    n_tokens: Union[np.ndarray, List[int]],
    max_log_diff: float = 0.1,
    pair_indices: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    **kwargs: Any,
) -> float:
    """
    Computes fraction where vulnerable scores higher than safe over all length-matched pairs
    whose |log(n_tokens_v) - log(n_tokens_s)| <= max_log_diff. Ties count as 0.5.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_prob).astype(float)
    toks = np.asarray(n_tokens).astype(float)

    if pair_indices is None:
        v_idx, s_idx = get_length_matched_pair_indices(y_t, toks, max_log_diff=max_log_diff)
    else:
        v_idx, s_idx = pair_indices

    n_pairs = len(v_idx)
    if n_pairs == 0:
        return 0.5

    p_v = y_p[v_idx]
    p_s = y_p[s_idx]

    wins = np.sum(p_v > p_s)
    ties = np.sum(p_v == p_s)

    return float((wins + 0.5 * ties) / n_pairs)


def alert_budget_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, List[float]],
    budget_fraction: float = 0.10,
) -> Dict[str, Any]:
    """
    Evaluates top-budget% alert operating point.
    Flags top budget_fraction (default 10%) of functions by predicted score.
    Returns precision, recall, lift over prevalence, threshold, and confusion counts.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_prob).astype(float)
    n = len(y_t)
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "lift": 0.0, "prevalence": 0.0}

    # Cutoff threshold to flag top fraction
    k_flag = max(1, int(np.round(n * budget_fraction)))
    sorted_probs = np.sort(y_p)
    tau = float(sorted_probs[-k_flag])

    y_pred = (y_p >= tau).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_t == 1)))
    fp = int(np.sum((y_pred == 1) & (y_t == 0)))
    fn = int(np.sum((y_pred == 0) & (y_t == 1)))
    tn = int(np.sum((y_pred == 0) & (y_t == 0)))

    n_pos = int(np.sum(y_t == 1))
    prevalence = float(n_pos / n) if n > 0 else 0.0

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / n_pos) if n_pos > 0 else 0.0
    lift = float(precision / prevalence) if prevalence > 0 else 0.0

    return {
        "budget_fraction": budget_fraction,
        "operating_threshold": tau,
        "n_total": n,
        "n_flagged": tp + fp,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "prevalence": prevalence,
        "precision": precision,
        "recall": recall,
        "lift": lift,
    }


def paired_metrics_at_cutoff(
    p_vuln: np.ndarray,
    p_fixed: np.ndarray,
    tau: float,
) -> Dict[str, float]:
    """Computes pair_correct, P-C, P-V, P-B, P-R given an operating cutoff threshold."""
    is_correct = np.mean(p_vuln > p_fixed) + 0.5 * np.mean(p_vuln == p_fixed)
    is_pc = np.mean((p_vuln >= tau) & (p_fixed < tau))
    is_pv = np.mean((p_vuln >= tau) & (p_fixed >= tau))
    is_pb = np.mean((p_vuln < tau) & (p_fixed < tau))
    is_pr = np.mean((p_vuln < tau) & (p_fixed >= tau))

    return {
        "pair_correct": float(is_correct),
        "P-C": float(is_pc),
        "P-V": float(is_pv),
        "P-B": float(is_pb),
        "P-R": float(is_pr),
        "threshold": float(tau),
    }


def paired_bootstrap_diff_v2(
    y_true: Union[np.ndarray, List[int]],
    prob_a: Union[np.ndarray, List[float]],
    prob_b: Union[np.ndarray, List[float]],
    metric_fn: Callable[..., float],
    groups: Optional[Union[np.ndarray, List[Any]]] = None,
    extra_arrays: Optional[Dict[str, np.ndarray]] = None,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes paired bootstrap confidence interval and two-sided p-value for:
    Delta = metric(Model B) - metric(Model A).
    """
    y_t = np.asarray(y_true).astype(int)
    p_a = np.asarray(prob_a).astype(float)
    p_b = np.asarray(prob_b).astype(float)

    extra_kwargs = {k: np.asarray(v) for k, v in extra_arrays.items()} if extra_arrays is not None else {}

    try:
        obs_a = float(metric_fn(y_t, p_a, **extra_kwargs))
        obs_b = float(metric_fn(y_t, p_b, **extra_kwargs))
    except TypeError:
        obs_a = float(metric_fn(y_t, p_a))
        obs_b = float(metric_fn(y_t, p_b))
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
            m_a = metric_fn(y_sub, p_a_sub, **sub_extra)
            m_b = metric_fn(y_sub, p_b_sub, **sub_extra)
            diffs.append(m_b - m_a)
        except TypeError:
            try:
                m_a = metric_fn(y_sub, p_a_sub)
                m_b = metric_fn(y_sub, p_b_sub)
                diffs.append(m_b - m_a)
            except Exception:
                continue
        except Exception:
            continue

    if len(diffs) == 0:
        return {
            "score_a": obs_a,
            "score_b": obs_b,
            "diff_observed": diff_obs,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "raw_p_value": 1.0,
            "p_value_formatted": "1.0000",
            "significant_p05": False,
        }

    diffs_arr = np.asarray(diffs)
    ci_lower = float(np.percentile(diffs_arr, lower_pct))
    ci_upper = float(np.percentile(diffs_arr, upper_pct))

    # Two-sided empirical p-value for testing H0: diff == 0
    frac_le_zero = np.mean(diffs_arr <= 0)
    frac_ge_zero = np.mean(diffs_arr >= 0)
    p_val = float(min(1.0, 2.0 * min(frac_le_zero, frac_ge_zero)))

    return {
        "score_a": obs_a,
        "score_b": obs_b,
        "diff_observed": diff_obs,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "raw_p_value": p_val,
        "p_value_formatted": format_p_value(p_val, 1.0 / n_resamples),
        "significant_p05": bool(p_val < 0.05),
    }
