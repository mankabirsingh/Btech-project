"""Step 2: Length-Debiased Training with Frozen R2 Features."""

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from scripts.p3.eval_protocol import get_cached_features, load_datasets
from scripts.p4.step1_paired_data import compute_paired_metrics, extract_disjoint_pairs
from src.eval.stats import (
    bootstrap_ci,
    calc_length_stratified_auroc,
    length_stratified_auroc,
    metrics_full,
    threshold_at_fpr,
)

P4_SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "p4"
P4_MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints" / "p4"
RESULTS_P4_DIR = PROJECT_ROOT / "results" / "p4"

P4_MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_P4_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model_on_dev(
    clf: Any,
    scaler_cxx: StandardScaler,
    feats: Dict[str, Dict[str, np.ndarray]],
    df_dev: pd.DataFrame,
    tok_dev: np.ndarray,
    df_pairs: pd.DataFrame,
    tau_10fpr: float,
    candidate_name: str,
    feature_type: str = "chunked_mean",
) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics on dev_heldout with 2000-resample bootstrap CIs."""
    # Build dev feature matrix
    sc_cxx_dv = scaler_cxx.transform(feats["dev"]["cxx28"])
    emb_dv = feats["dev"][f"emb_{feature_type}"]
    X_dv = np.hstack([emb_dv, sc_cxx_dv])
    y_dv = df_dev["vuln_label"].values
    is_cpp = df_dev["is_cpp"].astype(bool).values

    # Predict probabilities
    p_dv = clf.predict_proba(X_dv)[:, 1]

    # Overall metrics
    dev_auroc = float(roc_auc_score(y_dv, p_dv))
    dev_ls_auroc = float(calc_length_stratified_auroc(y_dv, p_dv, tok_dev))
    dev_cpp_auroc = float(roc_auc_score(y_dv[is_cpp], p_dv[is_cpp])) if np.sum(is_cpp) > 0 else 0.5
    dev_cpp_ls_auroc = float(calc_length_stratified_auroc(y_dv[is_cpp], p_dv[is_cpp], tok_dev[is_cpp])) if np.sum(is_cpp) > 0 else 0.5
    dev_auprc = float(average_precision_score(y_dv, p_dv))

    # Bootstrap CIs (2000 resamples)
    ci_ov = bootstrap_ci(y_dv, p_dv, metric_fn=lambda y, p, _: roc_auc_score(y, p), n_resamples=2000, seed=42)
    ci_ls = length_stratified_auroc(y_dv, p_dv, tok_dev, n_resamples=2000, seed=42)["ci_95"]
    ci_cpp = bootstrap_ci(y_dv[is_cpp], p_dv[is_cpp], metric_fn=lambda y, p, _: roc_auc_score(y, p), n_resamples=2000, seed=42)
    ci_cpp_ls = length_stratified_auroc(y_dv[is_cpp], p_dv[is_cpp], tok_dev[is_cpp], n_resamples=2000, seed=42)["ci_95"]
    ci_auprc = bootstrap_ci(y_dv, p_dv, metric_fn=lambda y, p, _: average_precision_score(y, p), n_resamples=2000, seed=42)

    # Threshold metrics at fold-0 10% FPR operating point
    metrics_at_tau = metrics_full(y_dv, p_dv, threshold=tau_10fpr)

    # Paired metrics on dev_pairs
    dev_id_to_idx = {sid: i for i, sid in enumerate(df_dev["sample_id"])}
    v_idx = [dev_id_to_idx[sid] for sid in df_pairs["vuln_sample_id"]]
    f_idx = [dev_id_to_idx[sid] for sid in df_pairs["fixed_sample_id"]]
    paired_eval = compute_paired_metrics(p_dv[v_idx], p_dv[f_idx], threshold=tau_10fpr, n_resamples=2000, seed=42)

    return {
        "candidate": candidate_name,
        "operating_threshold_10fpr": float(tau_10fpr),
        "dev_auroc": dev_auroc,
        "dev_auroc_ci_95": ci_ov,
        "dev_length_stratified_auroc": dev_ls_auroc,
        "dev_length_stratified_ci_95": ci_ls,
        "dev_cpp_auroc": dev_cpp_auroc,
        "dev_cpp_auroc_ci_95": ci_cpp,
        "dev_cpp_length_stratified_auroc": dev_cpp_ls_auroc,
        "dev_cpp_length_stratified_ci_95": ci_cpp_ls,
        "dev_auprc": dev_auprc,
        "dev_auprc_ci_95": ci_auprc,
        "metrics_at_10fpr_threshold": metrics_at_tau,
        "paired_metrics": paired_eval,
        "predictions": p_dv,
    }


def main():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("STEP 2: LENGTH-DEBIASED TRAINING (FROZEN R2 FEATURES)", flush=True)
    print("=" * 80, flush=True)

    # Load Step 0 draw variance std to quote alongside metrics
    with open(RESULTS_P4_DIR / "step0_resolutions.json", "r") as f:
        step0_res = json.load(f)
    draw_std_ls = step0_res["draw_variance_0_2"]["dev_ls_auroc_std"]
    print(f"Step 0.2 Draw Variance Std on LS-AUROC: {draw_std_ls:.4f}", flush=True)

    # Load data splits and cached features
    df_train, df_dev, tok_train, tok_dev = load_datasets()
    feats = get_cached_features()

    # Load dev pairs
    df_pairs = pd.read_parquet(P4_SPLITS_DIR / "dev_pairs.parquet")
    print(f"Loaded {len(df_pairs)} dev pairs from {P4_SPLITS_DIR / 'dev_pairs.parquet'}", flush=True)

    # Feature preparation for R2 (chunked-mean + cxx28, scaler solely on cxx28)
    sc_cxx = StandardScaler()
    sc_cxx_tr = sc_cxx.fit_transform(feats["train"]["cxx28"])
    X_tr_r2 = np.hstack([feats["train"]["emb_chunked_mean"], sc_cxx_tr])
    y_tr = df_train["vuln_label"].values
    groups_tr = df_train["group_id"].values

    bins = [0, 256, 512, 1024, np.inf]
    bin_idx = np.digitize(tok_train, bins) - 1

    # -------------------------------------------------------------------------
    # D1: LENGTH-STRATIFIED BALANCED SAMPLING
    # -------------------------------------------------------------------------
    print("\n--- Constructing D1: Length-Stratified Balanced Sampling ---", flush=True)
    # Target 25,000 samples -> 6,250 per bin -> 3,125 vuln, 3,125 safe per bin
    rng = np.random.RandomState(42)
    d1_indices = []
    d1_bin_counts = {}
    for b in range(4):
        in_b = bin_idx == b
        v_idx = np.where(in_b & (y_tr == 1))[0]
        s_idx = np.where(in_b & (y_tr == 0))[0]
        c_v = rng.choice(v_idx, size=3125, replace=(len(v_idx) < 3125))
        c_s = rng.choice(s_idx, size=3125, replace=(len(s_idx) < 3125))
        d1_indices.extend(c_v)
        d1_indices.extend(c_s)
        low, high = bins[b], bins[b+1]
        d1_bin_counts[f"bin_{b}_[{low},{high}]"] = {
            "vulnerable": int(len(c_v)),
            "safe": int(len(c_s)),
            "total": int(len(c_v) + len(c_s)),
        }
    d1_indices = np.array(d1_indices)
    np.save(P4_SPLITS_DIR / "d1_train_indices.npy", d1_indices)
    print(f"  D1 Total Samples: {len(d1_indices)}", flush=True)
    for b_name, b_c in d1_bin_counts.items():
        print(f"    {b_name}: vuln={b_c['vulnerable']}, safe={b_c['safe']}, total={b_c['total']}", flush=True)

    # -------------------------------------------------------------------------
    # D2: NATURAL 25k DRAW + INVERSE-PROPENSITY SAMPLE WEIGHTS
    # -------------------------------------------------------------------------
    print("\n--- Constructing D2: Inverse-Propensity Sample Weights ---", flush=True)
    weights_d2 = np.zeros(len(y_tr), dtype=np.float32)
    d2_bin_weights = {}
    for b in range(4):
        in_b = bin_idx == b
        v_mask = in_b & (y_tr == 1)
        s_mask = in_b & (y_tr == 0)
        n_v = np.sum(v_mask)
        n_s = np.sum(s_mask)
        target_sum = (n_v + n_s) / 2.0
        w_v = target_sum / n_v
        w_s = target_sum / n_s
        weights_d2[v_mask] = w_v
        weights_d2[s_mask] = w_s
        low, high = bins[b], bins[b+1]
        d2_bin_weights[f"bin_{b}_[{low},{high}]"] = {
            "n_vuln": int(n_v),
            "n_safe": int(n_s),
            "weight_vuln": float(w_v),
            "weight_safe": float(w_s),
            "weighted_vuln_sum": float(np.sum(weights_d2[v_mask])),
            "weighted_safe_sum": float(np.sum(weights_d2[s_mask])),
        }
    np.save(P4_SPLITS_DIR / "d2_train_weights.npy", weights_d2)
    print(f"  D2 Total Samples: {len(weights_d2)}", flush=True)
    print(f"  D2 Total Weighted Mass: vuln={np.sum(weights_d2[y_tr == 1]):.1f}, safe={np.sum(weights_d2[y_tr == 0]):.1f}", flush=True)

    # -------------------------------------------------------------------------
    # D3: PAIRED TRAINING (USING IDENTIFIED DISJOINT PAIRS)
    # -------------------------------------------------------------------------
    print("\n--- Constructing D3: Paired Training ---", flush=True)
    train_pairs = extract_disjoint_pairs(df_train)
    id_to_idx = {sid: i for i, sid in enumerate(df_train["sample_id"])}
    d3_indices = []
    for p in train_pairs:
        d3_indices.append(id_to_idx[p["vuln_sample_id"]])
        d3_indices.append(id_to_idx[p["fixed_sample_id"]])
    d3_indices = np.array(d3_indices)
    np.save(P4_SPLITS_DIR / "d3_train_indices.npy", d3_indices)
    print(f"  D3 Unique Disjoint Pairs in train_minus_dev: {len(train_pairs)} (Total Samples: {len(d3_indices)})", flush=True)

    # -------------------------------------------------------------------------
    # HYPERPARAMETER TUNING: SELECT C BY GROUPKFOLD OOF LS-AUROC
    # -------------------------------------------------------------------------
    print("\n--- Selecting Regularization C by GroupKFold OOF LS-AUROC ---", flush=True)
    C_grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    gkf = GroupKFold(n_splits=5)

    candidates_setup = {
        "D1": {
            "name": "D1_length_stratified_balanced",
            "X": X_tr_r2[d1_indices],
            "y": y_tr[d1_indices],
            "weights": None,
            "groups": groups_tr[d1_indices],
            "tok": tok_train[d1_indices],
            "class_weight": "balanced",
        },
        "D2": {
            "name": "D2_inverse_propensity_weighted",
            "X": X_tr_r2,
            "y": y_tr,
            "weights": weights_d2,
            "groups": groups_tr,
            "tok": tok_train,
            "class_weight": None,  # weights already balance classes per bin
        },
        "D3": {
            "name": "D3_paired_training",
            "X": X_tr_r2[d3_indices],
            "y": y_tr[d3_indices],
            "weights": None,
            "groups": groups_tr[d3_indices],
            "tok": tok_train[d3_indices],
            "class_weight": "balanced",
        },
    }

    best_hyperparams = {}
    cv_search_records = {}

    for cand_key, cfg in candidates_setup.items():
        print(f"\nTuning {cand_key} ({cfg['name']}) on C in {C_grid}...", flush=True)
        best_c = None
        best_oof_ls = -1.0
        best_oof_preds = None
        cand_search = []

        for c_val in C_grid:
            oof_p = np.zeros(len(cfg["y"]))
            for tr_idx, val_idx in gkf.split(cfg["X"], cfg["y"], groups=cfg["groups"]):
                clf_cv = LogisticRegression(
                    C=c_val,
                    class_weight=cfg["class_weight"],
                    max_iter=500,
                    tol=1e-3,
                    random_state=42,
                )
                w_tr = cfg["weights"][tr_idx] if cfg["weights"] is not None else None
                clf_cv.fit(cfg["X"][tr_idx], cfg["y"][tr_idx], sample_weight=w_tr)
                oof_p[val_idx] = clf_cv.predict_proba(cfg["X"][val_idx])[:, 1]

            oof_auc = float(roc_auc_score(cfg["y"], oof_p))
            oof_ls_auc = float(calc_length_stratified_auroc(cfg["y"], oof_p, cfg["tok"]))
            cand_search.append({
                "C": c_val,
                "oof_auroc": oof_auc,
                "oof_length_stratified_auroc": oof_ls_auc,
            })
            print(f"  {cand_key} C={c_val:<6}: OOF LS-AUROC = {oof_ls_auc:.4f} (OOF AUROC = {oof_auc:.4f})", flush=True)

            if oof_ls_auc > best_oof_ls:
                best_oof_ls = oof_ls_auc
                best_c = c_val
                best_oof_preds = oof_p

        # Determine 10% FPR threshold on fold-0 predictions for this candidate
        # Using fold 0 out-of-fold validation slice
        fold0_val_idx = list(gkf.split(cfg["X"], cfg["y"], groups=cfg["groups"]))[0][1]
        y_f0 = cfg["y"][fold0_val_idx]
        p_f0 = best_oof_preds[fold0_val_idx]
        res_thr = threshold_at_fpr(y_f0, p_f0, target_fpr=0.10)
        tau_10fpr = float(res_thr["threshold"])

        best_hyperparams[cand_key] = {
            "best_C": best_c,
            "best_oof_ls_auroc": best_oof_ls,
            "operating_threshold_10fpr": tau_10fpr,
            "fold0_val_size": len(fold0_val_idx),
        }
        cv_search_records[cand_key] = cand_search
        print(f"--> {cand_key} Selected C={best_c} (OOF LS-AUROC={best_oof_ls:.4f}, tau_10fpr={tau_10fpr:.4f})", flush=True)

    # -------------------------------------------------------------------------
    # FINAL MODEL TRAINING AND HELD-OUT DEV EVALUATION
    # -------------------------------------------------------------------------
    print("\n--- Training Final Models on Full Debiased Splits & Evaluating on dev_heldout ---", flush=True)
    eval_results = {}
    fitted_models = {}

    for cand_key, cfg in candidates_setup.items():
        best_c = best_hyperparams[cand_key]["best_C"]
        tau_10 = best_hyperparams[cand_key]["operating_threshold_10fpr"]

        clf = LogisticRegression(
            C=best_c,
            class_weight=cfg["class_weight"],
            max_iter=1000,
            random_state=42,
        )
        clf.fit(cfg["X"], cfg["y"], sample_weight=cfg["weights"])
        fitted_models[cand_key] = clf

        # Save checkpoint
        ckpt_path = P4_MODELS_DIR / f"{cand_key.lower()}_{cfg['name'].lower()}.pkl"
        with open(ckpt_path, "wb") as f:
            pickle.dump({
                "model": clf,
                "scaler_cxx": sc_cxx,
                "candidate": cand_key,
                "name": cfg["name"],
                "best_C": best_c,
                "threshold_10fpr": tau_10,
                "feature_type": "emb_chunked_mean + cxx28",
            }, f)
        print(f"Saved {cand_key} checkpoint to {ckpt_path}", flush=True)

        # Comprehensive evaluation on dev_heldout
        res = evaluate_model_on_dev(
            clf=clf,
            scaler_cxx=sc_cxx,
            feats=feats,
            df_dev=df_dev,
            tok_dev=tok_dev,
            df_pairs=df_pairs,
            tau_10fpr=tau_10,
            candidate_name=cand_key,
            feature_type="chunked_mean",
        )
        res["best_C"] = best_c
        res["oof_ls_auroc"] = best_hyperparams[cand_key]["best_oof_ls_auroc"]
        res["draw_std_ls_auroc"] = draw_std_ls
        res["delta_ls_vs_std"] = (res["dev_length_stratified_auroc"] - 0.5979) / draw_std_ls

        # Save predictions array separately to avoid massive json
        p_dv_cand = res.pop("predictions")
        np.save(RESULTS_P4_DIR / f"p_dev_{cand_key.lower()}.npy", p_dv_cand)

        eval_results[cand_key] = res

        print(f"\n  {cand_key} Results on dev_heldout:")
        print(f"    Dev AUROC:                   {res['dev_auroc']:.4f} [95% CI: {res['dev_auroc_ci_95'][0]:.4f}, {res['dev_auroc_ci_95'][1]:.4f}]")
        print(f"    Dev Length-Stratified AUROC: {res['dev_length_stratified_auroc']:.4f} [95% CI: {res['dev_length_stratified_ci_95'][0]:.4f}, {res['dev_length_stratified_ci_95'][1]:.4f}] (Draw std: {draw_std_ls:.4f})")
        print(f"    Dev C++ AUROC:               {res['dev_cpp_auroc']:.4f} [95% CI: {res['dev_cpp_auroc_ci_95'][0]:.4f}, {res['dev_cpp_auroc_ci_95'][1]:.4f}]")
        print(f"    Dev AUPRC:                   {res['dev_auprc']:.4f} [95% CI: {res['dev_auprc_ci_95'][0]:.4f}, {res['dev_auprc_ci_95'][1]:.4f}]")
        print(f"    pair_correct:                {res['paired_metrics']['pair_correct']*100:.2f}% [95% CI: {res['paired_metrics']['pair_correct_ci_95'][0]*100:.2f}%, {res['paired_metrics']['pair_correct_ci_95'][1]*100:.2f}%]")
        print(f"    P-C: {res['paired_metrics']['P-C']*100:.2f}%, P-V: {res['paired_metrics']['P-V']*100:.2f}%, P-B: {res['paired_metrics']['P-B']*100:.2f}%, P-R: {res['paired_metrics']['P-R']*100:.2f}%")
        print(f"    At 10% FPR Threshold ({tau_10:.4f}): Actual Dev FPR = {res['metrics_at_10fpr_threshold']['fpr']*100:.2f}%, Recall = {res['metrics_at_10fpr_threshold']['recall']*100:.2f}%", flush=True)

    # -------------------------------------------------------------------------
    # IDENTIFY BEST DEBIASED DATASET FOR STEP 3 FINE-TUNING
    # -------------------------------------------------------------------------
    best_debiased = max(eval_results.keys(), key=lambda k: eval_results[k]["dev_length_stratified_auroc"])
    print("\n" + "=" * 80, flush=True)
    print(f"BEST DEBIASED DATASET BY DEV LS-AUROC: {best_debiased} ({eval_results[best_debiased]['dev_length_stratified_auroc']:.4f})", flush=True)
    print(f"Selected for Step 3 UniXcoder fine-tuning (F1 run)", flush=True)
    print("=" * 80, flush=True)

    # Save output artifact
    step2_out = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "runtime_seconds": round(time.time() - t_start, 2),
            "step0_draw_std_ls_auroc": draw_std_ls,
            "best_debiased_dataset": best_debiased,
        },
        "d1_bin_counts": d1_bin_counts,
        "d2_bin_weights": d2_bin_weights,
        "cv_search_records": cv_search_records,
        "best_hyperparams": best_hyperparams,
        "eval_results": eval_results,
    }

    out_file = RESULTS_P4_DIR / "step2_debiased_models.json"
    with open(out_file, "w") as f:
        json.dump(step2_out, f, indent=2)

    print(f"\nSaved Step 2 results to {out_file}", flush=True)
    print(f"Step 2 finished in {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
