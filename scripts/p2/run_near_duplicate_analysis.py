"""Script for comprehensive near-duplicate leakage analysis using normalized code and MinHash LSH."""

import csv
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH

from scripts.p2.code_normalizer import compute_minhash_slice
from scripts.phase15_train_model_a import Phase15EnhancedModernCxxExtractor
from src.eval.locked_test import load_locked_test
from src.eval.stats import metrics_full, bootstrap_ci

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
RESULTS_P2_DIR = PROJECT_ROOT / "results" / "p2"


def extract_dataset_minhashes(
    codes: List[str], dataset_name: str, cache_dir: Path, num_workers: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes and caches MinHash 128-dim hashvalues and MD5 strings for normalized code."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_hval_path = cache_dir / f"{dataset_name}_minhash_128.npy"
    cache_md5_path = cache_dir / f"{dataset_name}_norm_md5.npy"

    if cache_hval_path.exists() and cache_md5_path.exists():
        print(f"  Loading cached MinHashes and MD5s for {dataset_name}...")
        return np.load(cache_hval_path), np.load(cache_md5_path)

    t0 = time.time()
    n = len(codes)
    chunk_size = int(np.ceil(n / num_workers))
    tasks = []
    tmp_dir = cache_dir / f"tmp_{dataset_name}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for p in range(num_workers):
        start = p * chunk_size
        end = min(n, (p + 1) * chunk_size)
        if start >= n:
            break
        tasks.append((p, codes[start:end], str(tmp_dir / f"part_{p}.npy"), str(tmp_dir / f"md5_{p}.npy")))

    with mp.Pool(len(tasks)) as pool:
        results = pool.map(compute_minhash_slice, tasks)

    hvals = np.vstack([np.load(r[0]) for r in results])
    md5s = np.concatenate([np.load(r[1]) for r in results])

    np.save(cache_hval_path, hvals)
    np.save(cache_md5_path, md5s)

    # Clean temporary files
    for r in results:
        Path(r[0]).unlink(missing_ok=True)
        Path(r[1]).unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except Exception:
        pass

    print(f"  Extracted and cached {len(codes):,} {dataset_name} signatures in {time.time()-t0:.2f}s")
    return hvals, md5s


def analyze_conflicts_inside_train(train_df: pd.DataFrame, train_md5s: np.ndarray) -> Dict[str, int]:
    """Finds exact-normalized duplicate functions inside train with conflicting labels."""
    print("  Analyzing intra-training label conflicts on exact-normalized code...")
    df_temp = pd.DataFrame({
        "norm_md5": train_md5s,
        "vuln_label": train_df["vuln_label"].values.astype(int),
        "sample_id": train_df["sample_id"].values,
    })

    grouped = df_temp.groupby("norm_md5")["vuln_label"].agg(["count", "nunique", "sum"])
    conflicting_groups = grouped[grouped["nunique"] > 1]
    agreeing_dup_groups = grouped[(grouped["count"] > 1) & (grouped["nunique"] == 1)]

    total_conflicting_samples = int(conflicting_groups["count"].sum())
    total_conflicting_patterns = len(conflicting_groups)
    total_agreeing_dup_samples = int(agreeing_dup_groups["count"].sum())

    return {
        "unique_normalized_patterns": int(len(grouped)),
        "conflicting_duplicate_patterns": total_conflicting_patterns,
        "conflicting_duplicate_samples": total_conflicting_samples,
        "agreeing_duplicate_patterns": int(len(agreeing_dup_groups)),
        "agreeing_duplicate_samples": total_agreeing_dup_samples,
    }


def main():
    t_start = time.time()
    RESULTS_P2_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("STEP 4: NEAR-DUPLICATE LEAKAGE AND DEDUPLICATED EVALUATION")
    print("=" * 80)

    # 1. Load Datasets
    print("\n[1] Loading Datasets...")
    train_df = pd.read_parquet(SPLITS_DIR / "vulnerability_train_expanded.parquet")
    val_df = pd.read_parquet(SPLITS_DIR / "vulnerability_val_expanded.parquet")
    test_df = load_locked_test(reason="Step 4: Near-duplicate leakage check on clean test set")

    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")

    # 2. Extract MinHashes and Normalized MD5s
    print("\n[2] Extracting MinHashes and Normalized Signatures (8 parallel workers)...")
    train_hvals, train_md5s = extract_dataset_minhashes(train_df["code"].tolist(), "train", CACHE_DIR, 8)
    test_hvals, test_md5s = extract_dataset_minhashes(test_df["code"].tolist(), "test", CACHE_DIR, 8)
    val_hvals, val_md5s = extract_dataset_minhashes(val_df["code"].tolist(), "val", CACHE_DIR, 8)

    # 3. Intra-Train Conflicting Exact Duplicates
    print("\n[3] Intra-Train Duplicate Conflict Check...")
    train_conflict_stats = analyze_conflicts_inside_train(train_df, train_md5s)
    print(f"  Conflicting duplicate patterns: {train_conflict_stats['conflicting_duplicate_patterns']:,}")
    print(f"  Conflicting duplicate samples:  {train_conflict_stats['conflicting_duplicate_samples']:,}")
    print(f"  Agreeing duplicate patterns:    {train_conflict_stats['agreeing_duplicate_patterns']:,}")
    print(f"  Agreeing duplicate samples:     {train_conflict_stats['agreeing_duplicate_samples']:,}")

    # 4. Build MinHash LSH Index on Train Pool
    print("\n[4] Building MinHash LSH Index on Train Pool (threshold=0.80)...")
    t0 = time.time()
    lsh = MinHashLSH(threshold=0.80, num_perm=128)
    for idx in range(len(train_hvals)):
        mh = MinHash(num_perm=128, hashvalues=train_hvals[idx], scheme="affine32")
        lsh.insert(idx, mh)
    print(f"  Indexed {len(train_hvals):,} samples into LSH in {time.time()-t0:.2f}s")

    # 5. Query Test vs Full Train Pool
    print("\n[5] Querying Locked Test Set against Train Pool...")
    t0 = time.time()
    test_matches = []
    test_ids_matched = set()

    for t_idx in range(len(test_hvals)):
        t_mh = MinHash(num_perm=128, hashvalues=test_hvals[t_idx], scheme="affine32")
        candidates = lsh.query(t_mh)
        for tr_idx in candidates:
            tr_mh = MinHash(num_perm=128, hashvalues=train_hvals[tr_idx], scheme="affine32")
            sim = t_mh.jaccard(tr_mh)
            if sim >= 0.80:
                t_sid = test_df["sample_id"].iloc[t_idx]
                t_proj = test_df["project"].iloc[t_idx]
                t_label = int(test_df["vuln_label"].iloc[t_idx])
                tr_sid = train_df["sample_id"].iloc[tr_idx]
                tr_proj = train_df["project"].iloc[tr_idx]
                tr_label = int(train_df["vuln_label"].iloc[tr_idx])
                agree = bool(t_label == tr_label)

                test_matches.append({
                    "test_sample_id": t_sid,
                    "test_project": t_proj,
                    "test_vuln_label": t_label,
                    "train_sample_id": tr_sid,
                    "train_project": tr_proj,
                    "train_vuln_label": tr_label,
                    "jaccard": round(float(sim), 4),
                    "label_agreement": agree,
                })
                test_ids_matched.add(t_sid)

    print(f"  Test queries complete in {time.time()-t0:.2f}s. Found {len(test_matches)} matched pairs across {len(test_ids_matched)} unique test samples.")

    # Save test-train matches to CSV
    test_csv_path = RESULTS_P2_DIR / "near_dups_test_train.csv"
    with open(test_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "test_sample_id", "test_project", "test_vuln_label",
            "train_sample_id", "train_project", "train_vuln_label",
            "jaccard", "label_agreement"
        ])
        writer.writeheader()
        writer.writerows(test_matches)
    print(f"  Saved test matches to {test_csv_path}")

    # Per-project test breakdown
    test_proj_breakdown = {}
    test_df["has_near_dup"] = test_df["sample_id"].isin(test_ids_matched)
    for proj in ["overall", "envoy", "openssl", "vim", "gpac"]:
        sub_df = test_df if proj == "overall" else test_df[test_df["project"].str.lower() == proj.lower()]
        matched_pairs_proj = [m for m in test_matches if (proj == "overall" or m["test_project"].lower() == proj.lower())]
        unique_samples = len(set(m["test_sample_id"] for m in matched_pairs_proj))
        agree_pairs = sum(1 for m in matched_pairs_proj if m["label_agreement"])
        disagree_pairs = sum(1 for m in matched_pairs_proj if not m["label_agreement"])

        test_proj_breakdown[proj] = {
            "total_test_samples": len(sub_df),
            "matched_pairs": len(matched_pairs_proj),
            "unique_test_samples_matched": unique_samples,
            "pct_samples_matched": float(unique_samples / len(sub_df) * 100.0) if len(sub_df) > 0 else 0.0,
            "label_agreement_pairs": agree_pairs,
            "label_disagreement_pairs": disagree_pairs,
        }

    # 6. Query Val vs Full Train Pool
    print("\n[6] Querying Validation Set against Train Pool...")
    t0 = time.time()
    val_matches = []
    val_ids_matched = set()

    for v_idx in range(len(val_hvals)):
        v_mh = MinHash(num_perm=128, hashvalues=val_hvals[v_idx], scheme="affine32")
        candidates = lsh.query(v_mh)
        for tr_idx in candidates:
            tr_mh = MinHash(num_perm=128, hashvalues=train_hvals[tr_idx], scheme="affine32")
            sim = v_mh.jaccard(tr_mh)
            if sim >= 0.80:
                v_sid = val_df["sample_id"].iloc[v_idx]
                v_proj = val_df["project"].iloc[v_idx]
                v_label = int(val_df["vuln_label"].iloc[v_idx])
                tr_sid = train_df["sample_id"].iloc[tr_idx]
                tr_proj = train_df["project"].iloc[tr_idx]
                tr_label = int(train_df["vuln_label"].iloc[tr_idx])
                agree = bool(v_label == tr_label)

                val_matches.append({
                    "val_sample_id": v_sid,
                    "val_project": v_proj,
                    "val_vuln_label": v_label,
                    "train_sample_id": tr_sid,
                    "train_project": tr_proj,
                    "train_vuln_label": tr_label,
                    "jaccard": round(float(sim), 4),
                    "label_agreement": agree,
                })
                val_ids_matched.add(v_sid)

    print(f"  Val queries complete in {time.time()-t0:.2f}s. Found {len(val_matches)} matched pairs across {len(val_ids_matched)} unique val samples.")

    val_csv_path = RESULTS_P2_DIR / "near_dups_val_train.csv"
    with open(val_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "val_sample_id", "val_project", "val_vuln_label",
            "train_sample_id", "train_project", "train_vuln_label",
            "jaccard", "label_agreement"
        ])
        writer.writeheader()
        writer.writerows(val_matches)
    print(f"  Saved val matches to {val_csv_path}")

    val_summary = {
        "total_val_samples": len(val_df),
        "matched_pairs": len(val_matches),
        "unique_val_samples_matched": len(val_ids_matched),
        "pct_samples_matched": float(len(val_ids_matched) / len(val_df) * 100.0),
        "label_agreement_pairs": sum(1 for m in val_matches if m["label_agreement"]),
        "label_disagreement_pairs": sum(1 for m in val_matches if not m["label_agreement"]),
    }

    # 7. Evaluate Phase 15 Champion on Full vs Deduped Test Set (WITHOUT Retraining)
    print("\n[7] Computing Phase 15 Champion Predictions on Full vs Deduped Test Set...")
    ckpt_path = MODELS_DIR / "phase15_codebert_cxx_hybrid_best.pkl"
    with open(ckpt_path, "rb") as f:
        ckpt_p15 = pickle.load(f)

    # Load cached CodeBERT clean test embeddings
    emb_test_path = PROJECT_ROOT / "results" / "phase14" / "codebert_test_clean_emb.npy"
    emb_test = np.load(emb_test_path)

    # Transform 28-dim features
    ext_28 = Phase15EnhancedModernCxxExtractor()
    feat_28 = ext_28.transform(test_df["code"].values)
    feat_scaled = ckpt_p15["scaler"].transform(feat_28)
    X_test_p15 = np.hstack([emb_test, feat_scaled])

    prob_test = ckpt_p15["clf"].predict_proba(X_test_p15)[:, 1]
    y_test = test_df["vuln_label"].values.astype(int)

    # Dedup mask: True if NOT matched as near-duplicate of train
    dedup_mask = ~test_df["sample_id"].isin(test_ids_matched).values
    print(f"  Full Test Samples:    {len(y_test):,}")
    print(f"  Deduped Test Samples: {int(np.sum(dedup_mask)):,} ({int(np.sum(~dedup_mask))} excluded, {np.mean(~dedup_mask)*100:.2f}%)")

    cohorts = ["Overall", "Envoy", "OpenSSL", "Vim", "GPAC"]
    comparison_metrics = {}

    for cohort in cohorts:
        if cohort == "Overall":
            c_mask_full = np.ones(len(y_test), dtype=bool)
        else:
            c_mask_full = (test_df["project"].str.lower() == cohort.lower()).values

        c_mask_dedup = c_mask_full & dedup_mask

        # Metrics on full
        y_f = y_test[c_mask_full]
        p_f = prob_test[c_mask_full]
        m_full = metrics_full(y_f, p_f, threshold=0.50)
        ci_f = bootstrap_ci(y_f, p_f, threshold=0.50, n_resamples=1000, seed=42)

        # Metrics on deduped
        y_d = y_test[c_mask_dedup]
        p_d = prob_test[c_mask_dedup]
        m_dedup = metrics_full(y_d, p_d, threshold=0.50)
        ci_d = bootstrap_ci(y_d, p_d, threshold=0.50, n_resamples=1000, seed=42)

        comparison_metrics[cohort] = {
            "full": {**m_full, **ci_f},
            "deduped": {**m_dedup, **ci_d},
            "deltas": {
                "delta_accuracy": float(m_dedup["accuracy"] - m_full["accuracy"]),
                "delta_auroc": float(m_dedup["auroc"] - m_full["auroc"]),
                "delta_fpr": float(m_dedup["fpr"] - m_full["fpr"]),
                "delta_f1": float(m_dedup["f1"] - m_full["f1"]),
                "samples_removed": int(len(y_f) - len(y_d)),
            }
        }

    # 8. Save Complete Results
    output_summary = {
        "metadata": {
            "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "runtime_seconds": round(time.time() - t_start, 2),
            "jaccard_threshold": 0.80,
            "minhash_num_perm": 128,
        },
        "train_intra_duplicates": train_conflict_stats,
        "test_vs_train_breakdown": test_proj_breakdown,
        "val_vs_train_summary": val_summary,
        "champion_full_vs_deduped_metrics": comparison_metrics,
    }

    summary_json_path = RESULTS_P2_DIR / "near_duplicate_analysis.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)

    print(f"\nSaved near-duplicate analysis summary to {summary_json_path}")
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON: PHASE 15 CHAMPION FULL TEST VS DEDUPED TEST")
    print("=" * 80)
    print(f"{'Cohort':<10} | {'Full N':<7} {'Full Acc':<10} {'Full AUROC':<10} | {'Dedup N':<7} {'Dedup Acc':<10} {'Dedup AUROC':<10} | {'Delta Acc':<10} {'Delta AUROC':<10}")
    print("-" * 95)
    for cohort in cohorts:
        c_res = comparison_metrics[cohort]
        f_n = c_res["full"]["n"]
        f_acc = c_res["full"]["accuracy"] * 100
        f_auc = c_res["full"]["auroc"]
        d_n = c_res["deduped"]["n"]
        d_acc = c_res["deduped"]["accuracy"] * 100
        d_auc = c_res["deduped"]["auroc"]
        delta_acc = c_res["deltas"]["delta_accuracy"] * 100
        delta_auc = c_res["deltas"]["delta_auroc"]
        print(f"{cohort:<10} | {f_n:<7} {f_acc:>6.2f}%    {f_auc:>6.4f}     | {d_n:<7} {d_acc:>6.2f}%    {d_auc:>6.4f}     | {delta_acc:>+6.2f}%     {delta_auc:>+6.4f}")


if __name__ == "__main__":
    main()
