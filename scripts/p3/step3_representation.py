"""STEP 3: Representation Optimization.

Evaluates R1 through R6 on N0 data variant.
Tunes C in {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1} strictly by mean GroupKFold OOF AUROC (macro over folds).
Standardizes features inside each fold.
Evaluates all Step 0 metrics on dev_heldout.
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from scripts.p3.eval_protocol import (
    evaluate_candidate,
    get_cached_features,
    get_champion_p2_reference,
    load_datasets,
)

RESULTS_P3_DIR = PROJECT_ROOT / "results" / "p3"
RESULTS_P3_DIR.mkdir(parents=True, exist_ok=True)


def extract_cyclomatic_proxy(codes: pd.Series) -> np.ndarray:
    pattern = re.compile(r"\b(if|for|while|case|catch)\b|&&|\|\|")
    counts = [len(pattern.findall(str(c))) for c in codes]
    return np.array(counts, dtype=np.float32)


from joblib import Parallel, delayed

def tune_c_on_groupkfold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kfolds: np.ndarray,
    c_grid: List[float],
) -> Tuple[float, Dict[float, Dict[str, float]]]:
    """Tunes C strictly on GroupKFold out-of-fold folds by macro-mean AUROC."""
    tuning_results = {}
    best_c = c_grid[0]
    best_macro_auc = -1.0

    # Pre-scale folds ONCE
    scaled_folds = []
    for fold in range(5):
        tr_idx = np.where(kfolds != fold)[0]
        val_idx = np.where(kfolds == fold)[0]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train[tr_idx])
        X_val = scaler.transform(X_train[val_idx])
        scaled_folds.append((tr_idx, val_idx, X_tr, X_val))

    def _eval_fold(fold_idx: int, c_val: float) -> float:
        tr_idx, val_idx, X_tr, X_val = scaled_folds[fold_idx]
        clf = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, random_state=42)
        clf.fit(X_tr, y_train[tr_idx])
        p_val = clf.predict_proba(X_val)[:, 1]
        return float(roc_auc_score(y_train[val_idx], p_val))

    for c in c_grid:
        fold_aucs = Parallel(n_jobs=5)(
            delayed(_eval_fold)(fold, c) for fold in range(5)
        )
        macro_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        tuning_results[c] = {
            "macro_mean_auroc": macro_auc,
            "std_auroc": std_auc,
            "fold_aurocs": fold_aucs,
        }
        if macro_auc > best_macro_auc:
            best_macro_auc = macro_auc
            best_c = c

    return best_c, tuning_results


def main():
    print("=" * 80)
    print("STEP 3: REPRESENTATION OPTIMIZATION (R1 - R6)")
    print("=" * 80)
    t_start = time.time()

    df_train, df_dev, tok_train, tok_dev = load_datasets()
    features = get_cached_features()
    champ_ref = get_champion_p2_reference(df_train, df_dev, tok_dev, features)

    y_train = df_train["vuln_label"].values.astype(int)
    kfolds = df_train["kfold"].values

    # Extract L2 length features
    log_tok_train = np.log1p(tok_train).reshape(-1, 1).astype(np.float32)
    log_tok_dev = np.log1p(tok_dev).reshape(-1, 1).astype(np.float32)

    lines_train = np.array([str(c).count("\n") + 1 for c in df_train["code"]], dtype=np.float32)
    lines_dev = np.array([str(c).count("\n") + 1 for c in df_dev["code"]], dtype=np.float32)
    log_lines_train = np.log1p(lines_train).reshape(-1, 1)
    log_lines_dev = np.log1p(lines_dev).reshape(-1, 1)

    cyc_train = np.log1p(extract_cyclomatic_proxy(df_train["code"])).reshape(-1, 1)
    cyc_dev = np.log1p(extract_cyclomatic_proxy(df_dev["code"])).reshape(-1, 1)

    L2_train = np.hstack([log_tok_train, log_lines_train, cyc_train])
    L2_dev = np.hstack([log_tok_dev, log_lines_dev, cyc_dev])

    # Build representations
    print("\n[1] Constructing Feature Matrices for R1 - R6...")
    emb_std_tr = features["train"]["emb_codebert_256"]
    emb_std_dv = features["dev"]["emb_codebert_256"]

    emb_mean_tr = features["train"]["emb_chunked_mean"]
    emb_mean_dv = features["dev"]["emb_chunked_mean"]

    emb_max_tr = features["train"]["emb_chunked_max"]
    emb_max_dv = features["dev"]["emb_chunked_max"]

    cxx28_tr = features["train"]["cxx28"]
    cxx28_dv = features["dev"]["cxx28"]

    cxx38_tr = features["train"]["cxx38"]
    cxx38_dv = features["dev"]["cxx38"]

    rep_configs = [
        ("R1", "CLS+mean 256 + cxx28 (rebaselined champion recipe)",
         np.hstack([emb_std_tr, cxx28_tr]),
         np.hstack([emb_std_dv, cxx28_dv])),
        ("R2", "chunked-mean + cxx28",
         np.hstack([emb_mean_tr, cxx28_tr]),
         np.hstack([emb_mean_dv, cxx28_dv])),
        ("R3", "chunked-max + cxx28",
         np.hstack([emb_max_tr, cxx28_tr]),
         np.hstack([emb_max_dv, cxx28_dv])),
        ("R4", "[CLS+mean 256 ; chunked-mean ; chunked-max] + cxx28",
         np.hstack([emb_std_tr, emb_mean_tr, emb_max_tr, cxx28_tr]),
         np.hstack([emb_std_dv, emb_mean_dv, emb_max_dv, cxx28_dv])),
        ("R5", "[CLS+mean 256 ; chunked-mean ; chunked-max] + cxx38",
         np.hstack([emb_std_tr, emb_mean_tr, emb_max_tr, cxx38_tr]),
         np.hstack([emb_std_dv, emb_mean_dv, emb_max_dv, cxx38_dv])),
        ("R6", "R4 + L2 length features (explicit length control)",
         np.hstack([emb_std_tr, emb_mean_tr, emb_max_tr, cxx28_tr, L2_train]),
         np.hstack([emb_std_dv, emb_mean_dv, emb_max_dv, cxx28_dv, L2_dev])),
    ]

    c_grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    results_representations = {}

    print("\n[2] Tuning C and Evaluating Representations on Dev...")
    for rid, desc, x_tr, x_dv in rep_configs:
        t_r0 = time.time()
        print(f"\n--- Tuning & Evaluating {rid}: {desc} (Dims: {x_tr.shape[1]}) ---")
        best_c, tuning_dict = tune_c_on_groupkfold(x_tr, y_train, kfolds, c_grid)
        best_macro_auc = tuning_dict[best_c]["macro_mean_auroc"]
        print(f"  Selected C* = {best_c} via GroupKFold CV (Macro AUROC: {best_macro_auc:.4f} +/- {tuning_dict[best_c]['std_auroc']:.4f})")

        res = evaluate_candidate(
            candidate_id=rid,
            description=desc,
            category="Representation",
            X_train_raw=x_tr,
            X_dev_raw=x_dv,
            df_train=df_train,
            df_dev=df_dev,
            tok_dev=tok_dev,
            champ_ref=champ_ref,
            C_val=best_c,
        )
        res["c_tuning"] = {str(k): v for k, v in tuning_dict.items()}
        res["feature_dim"] = int(x_tr.shape[1])

        print(f"  Dev AUROC: {res['dev_auroc']:.4f} {res['dev_auroc_ci_95']}")
        print(f"  Dev Length-Stratified AUROC: {res['dev_length_stratified_auroc']:.4f} {res['dev_length_stratified_ci_95']}")
        print(f"  Dev C++ AUROC: {res['dev_cpp_auroc']:.4f} {res['dev_cpp_auroc_ci_95']}")
        print(f"  Completed {rid} in {time.time()-t_r0:.1f}s")
        results_representations[rid] = res

    # 3. Identify Best Representation
    # By dev length-stratified AUROC (with dev C++ AUROC not significantly worse)
    best_rid = "R1"
    best_ls_auc = results_representations["R1"]["dev_length_stratified_auroc"]

    for rid in ["R2", "R3", "R4", "R5", "R6"]:
        ls_auc = results_representations[rid]["dev_length_stratified_auroc"]
        if ls_auc > best_ls_auc:
            # Check C++ difference
            cpp_diff = results_representations[rid]["diff_vs_champion"]["cpp_auroc"]
            if not (cpp_diff["diff_observed"] < 0 and cpp_diff["p_value"] < 0.05):
                best_rid = rid
                best_ls_auc = ls_auc

    print(f"\n================================================================================")
    print(f"Best Representation Selected: {best_rid} (Dev Length-Stratified AUROC = {best_ls_auc:.4f})")
    print(f"================================================================================")

    output_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "runtime_seconds": round(time.time() - t_start, 2),
            "data_variant": "N0",
            "c_grid": c_grid,
        },
        "representations": {
            k: {sk: sv for sk, sv in v.items() if sk not in ["p_dev", "p_oof"]}
            for k, v in results_representations.items()
        },
        "best_representation": best_rid,
    }

    out_json = RESULTS_P3_DIR / "step3_representation.json"
    with open(out_json, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nStep 3 successfully saved to {out_json} in {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
