"""Computes and caches frozen CodeBERT embeddings, chunked embeddings, and structural features."""

from collections import defaultdict
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from scripts.phase15_train_model_a import Phase15EnhancedModernCxxExtractor
from scripts.phase16_omission_extractor import Phase16OmissionAwareCxxExtractor
from src.eval.locked_test import compute_file_sha256, load_locked_test

# Suppress tokenizer sequence length warnings during full-sequence encoding
warnings.filterwarnings("ignore")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
P2_SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "p2"
RESULTS_P2_DIR = PROJECT_ROOT / "results" / "p2"


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_standard_codebert(
    codes: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 256,
) -> np.ndarray:
    """Extract standard frozen CodeBERT embeddings: CLS + masked mean (1,536 dimensions)."""
    model.eval()
    embeddings = []
    t0 = time.time()

    with torch.no_grad():
        for i in range(0, len(codes), batch_size):
            batch_codes = [str(c) if c is not None else "" for c in codes[i : i + batch_size]]
            inputs = tokenizer(
                batch_codes,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            hidden = outputs.last_hidden_state  # [B, L, 768]
            cls_repr = hidden[:, 0, :]  # [B, 768]

            mask = inputs["attention_mask"].unsqueeze(-1)
            sum_embeddings = torch.sum(hidden * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_repr = sum_embeddings / sum_mask  # [B, 768]

            combined = torch.cat([cls_repr, mean_repr], dim=-1).cpu().numpy().astype(np.float32)
            embeddings.append(combined)

    res = np.vstack(embeddings)
    elapsed = time.time() - t0
    print(f"    Standard {max_length} tokens: {len(codes):,} samples in {elapsed:.1f}s ({len(codes)/elapsed:.1f} samp/s)")
    return res


def extract_chunked_codebert(
    codes: List[str],
    emb_std: np.ndarray,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 64,
    window_size: int = 254,
    stride: int = 128,
    max_windows_per_sample: int = 16,
    sub_chunk_size: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract sliding window chunked embeddings (window 254 tokens + CLS/SEP, stride 128).
    Captures up to max_windows_per_sample windows (up to 2,176 tokens per function).
    Directly reuses emb_std for single-window functions (length <= 254 tokens).
    Returns (mean_pooled_matrix, max_pooled_matrix), each of shape (N, 1536).
    """
    model.eval()
    t0 = time.time()
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    n = len(codes)
    mean_pooled = np.zeros((n, 1536), dtype=np.float32)
    max_pooled = np.zeros((n, 1536), dtype=np.float32)

    total_windows_processed = 0

    for chunk_start in range(0, n, sub_chunk_size):
        chunk_end = min(n, chunk_start + sub_chunk_size)
        slice_codes = codes[chunk_start:chunk_end]

        window_batch = []
        window_meta = []  # (abs_s_idx, chunk_idx)

        for rel_s_idx, c in enumerate(slice_codes):
            abs_s_idx = chunk_start + rel_s_idx
            code_str = str(c) if c is not None else ""
            toks = tokenizer.encode(code_str, add_special_tokens=False, truncation=False)

            if len(toks) <= window_size:
                # Single window: exactly identical to standard 256 embedding
                mean_pooled[abs_s_idx] = emb_std[abs_s_idx]
                max_pooled[abs_s_idx] = emb_std[abs_s_idx]
            else:
                w_count = 0
                for start in range(0, len(toks), stride):
                    w = toks[start : start + window_size]
                    if not w:
                        break
                    window_batch.append([cls_id] + w + [sep_id])
                    window_meta.append((abs_s_idx, w_count))
                    w_count += 1
                    if w_count >= max_windows_per_sample:
                        break

        total_windows_processed += len(window_batch)

        if len(window_batch) > 0:
            slice_chunk_embs = []
            with torch.no_grad():
                for b_start in range(0, len(window_batch), batch_size):
                    b_chunks = window_batch[b_start : b_start + batch_size]
                    max_l = max(len(ch) for ch in b_chunks)

                    b_ids = [ch + [pad_id] * (max_l - len(ch)) for ch in b_chunks]
                    b_masks = [[1] * len(ch) + [0] * (max_l - len(ch)) for ch in b_chunks]

                    t_ids = torch.tensor(b_ids, dtype=torch.long, device=device)
                    t_mask = torch.tensor(b_masks, dtype=torch.long, device=device)

                    outputs = model(input_ids=t_ids, attention_mask=t_mask)
                    hidden = outputs.last_hidden_state
                    cls_repr = hidden[:, 0, :]
                    mask_f = t_mask.unsqueeze(-1)
                    mean_repr = torch.sum(hidden * mask_f, 1) / torch.clamp(mask_f.sum(1), min=1e-9)

                    comb = torch.cat([cls_repr, mean_repr], dim=-1).cpu().numpy().astype(np.float32)
                    slice_chunk_embs.append(comb)

            all_slice_embs = np.vstack(slice_chunk_embs)

            # Aggregate per sample
            sample_to_embs = defaultdict(list)
            for idx, (abs_s_idx, _) in enumerate(window_meta):
                sample_to_embs[abs_s_idx].append(idx)

            for abs_s_idx, idxs in sample_to_embs.items():
                sub = all_slice_embs[idxs]
                mean_pooled[abs_s_idx] = np.mean(sub, axis=0)
                max_pooled[abs_s_idx] = np.max(sub, axis=0)

        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        elapsed = time.time() - t0
        print(f"    Chunking progress: {chunk_end:,}/{n:,} samples ({total_windows_processed:,} windows embedded in {elapsed:.1f}s, {chunk_end/elapsed:.1f} samp/s)")

    return mean_pooled, max_pooled


def main():
    t_start = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_P2_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 6: EMBEDDING & STRUCTURAL FEATURE CACHING")
    print("=" * 80)

    device = get_torch_device()
    print(f"PyTorch Device: {device}")

    # 1. Initialize Datasets
    print("\n[1] Loading Target Datasets...")
    train_parquet_path = P2_SPLITS_DIR / "train_minus_dev.parquet"
    dev_parquet_path = P2_SPLITS_DIR / "dev_heldout.parquet"

    df_train = pd.read_parquet(train_parquet_path)
    df_dev = pd.read_parquet(dev_parquet_path)
    df_test = load_locked_test(reason="Step 6: Compute frozen embeddings and structural features for locked test cache")

    datasets = [
        ("train_minus_dev", df_train, train_parquet_path, compute_file_sha256(train_parquet_path)),
        ("dev_heldout", df_dev, dev_parquet_path, compute_file_sha256(dev_parquet_path)),
        ("test_clean", df_test, PROJECT_ROOT / "data" / "splits" / "vulnerability_test_heldout_projects_clean.parquet", "ebd95dfc837f12264dd0fd8126563d00ede27908bca38fe46611e25d7bb45bbc"),
    ]

    for name, df, path, sha in datasets:
        print(f"  {name:<16}: {len(df):,} samples | SHA-256: {sha[:16]}...")

    # 2. Extract Structural Features (28-dim Phase 15 and 38-dim Phase 16)
    print("\n[2] Extracting Structural C/C++ Features...")
    ext_28 = Phase15EnhancedModernCxxExtractor()
    ext_38 = Phase16OmissionAwareCxxExtractor()

    for name, df, path, sha in datasets:
        f28_path = CACHE_DIR / f"feat_cxx28_{sha}.npy"
        f38_path = CACHE_DIR / f"feat_cxx38_{sha}.npy"

        if f28_path.exists() and f38_path.exists():
            print(f"  Structural features for {name} already cached in {CACHE_DIR.name}/.")
            continue

        print(f"\n  Processing structural features for {name} ({len(df):,} samples)...")
        codes = df["code"].tolist()

        if not f28_path.exists():
            t0 = time.time()
            f28_mat = ext_28.transform(codes).astype(np.float32)
            np.save(f28_path, f28_mat)
            print(f"    feat_cxx28: shape {f28_mat.shape} saved to {f28_path.name} in {time.time()-t0:.2f}s")

        if not f38_path.exists():
            t0 = time.time()
            f38_mat = ext_38.transform(codes).astype(np.float32)
            np.save(f38_path, f38_mat)
            print(f"    feat_cxx38: shape {f38_mat.shape} saved to {f38_path.name} in {time.time()-t0:.2f}s")

    # 3. Load CodeBERT Model & Tokenizer
    print("\n[3] Loading CodeBERT Model & Tokenizer from HuggingFace...")
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    # 4. Extract Standard & Chunked CodeBERT Embeddings
    print("\n[4] Computing CodeBERT Standard and Chunked Embeddings...")
    cache_records = {}

    for name, df, path, sha in datasets:
        print(f"\n  Processing embeddings for {name} ({len(df):,} samples)...")
        codes = df["code"].tolist()
        dataset_records = {}

        # a) Standard CodeBERT max 256
        std_emb_path = CACHE_DIR / f"emb_codebert_256_{sha}.npy"
        std_emb_short = CACHE_DIR / f"emb_codebert_256_{sha[:16]}.npy"
        t0 = time.time()

        if std_emb_path.exists():
            print(f"    Standard 256 embeddings already exist: {std_emb_path.name}")
            emb_std = np.load(std_emb_path)
            t_std = 0.1
        else:
            emb_std = extract_standard_codebert(codes, tokenizer, model, device, batch_size=64, max_length=256)
            np.save(std_emb_path, emb_std)
            np.save(std_emb_short, emb_std)
            t_std = round(time.time() - t0, 2)

        dataset_records["emb_codebert_256"] = {
            "path": str(std_emb_path.relative_to(PROJECT_ROOT)),
            "shape": list(emb_std.shape),
            "dtype": str(emb_std.dtype),
            "runtime_s": t_std,
        }

        # b) Chunked CodeBERT mean and max pooled
        chk_mean_path = CACHE_DIR / f"emb_codebert_chunked_mean_256_{sha}.npy"
        chk_mean_short = CACHE_DIR / f"emb_codebert_chunked_mean_256_{sha[:16]}.npy"
        chk_max_path = CACHE_DIR / f"emb_codebert_chunked_max_256_{sha}.npy"
        chk_max_short = CACHE_DIR / f"emb_codebert_chunked_max_256_{sha[:16]}.npy"
        t0 = time.time()

        if chk_mean_path.exists() and chk_max_path.exists():
            print(f"    Chunked embeddings already exist: {chk_mean_path.name}")
            emb_mean = np.load(chk_mean_path)
            emb_max = np.load(chk_max_path)
            t_chk = 0.1
        else:
            emb_mean, emb_max = extract_chunked_codebert(
                codes, emb_std, tokenizer, model, device, batch_size=64, window_size=254, stride=128, max_windows_per_sample=16, sub_chunk_size=1000
            )
            np.save(chk_mean_path, emb_mean)
            np.save(chk_mean_short, emb_mean)
            np.save(chk_max_path, emb_max)
            np.save(chk_max_short, emb_max)
            t_chk = round(time.time() - t0, 2)

        dataset_records["emb_codebert_chunked_mean_256"] = {
            "path": str(chk_mean_path.relative_to(PROJECT_ROOT)),
            "shape": list(emb_mean.shape),
            "dtype": str(emb_mean.dtype),
            "runtime_s": t_chk,
        }
        dataset_records["emb_codebert_chunked_max_256"] = {
            "path": str(chk_max_path.relative_to(PROJECT_ROOT)),
            "shape": list(emb_max.shape),
            "dtype": str(emb_max.dtype),
            "runtime_s": t_chk,
        }
        dataset_records["feat_cxx28"] = {
            "path": f"data/cache/feat_cxx28_{sha}.npy",
            "shape": [len(df), 28],
            "dtype": "float32",
        }
        dataset_records["feat_cxx38"] = {
            "path": f"data/cache/feat_cxx38_{sha}.npy",
            "shape": [len(df), 38],
            "dtype": "float32",
        }

        cache_records[name] = {
            "dataset_sha256": sha,
            "sample_count": len(df),
            "files": dataset_records,
        }

    # 5. Save Cache Manifest
    manifest = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_runtime_seconds": round(time.time() - t_start, 2),
            "device": str(device),
            "model": model_name,
        },
        "datasets": cache_records,
    }

    manifest_path = CACHE_DIR / "cache_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    summary_path = RESULTS_P2_DIR / "cache_manifest.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved cache manifest to {manifest_path} and {summary_path}")
    print(f"Total Step 6 runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
