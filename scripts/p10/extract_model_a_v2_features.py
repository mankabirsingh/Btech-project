#!/usr/bin/env python3
"""
Extract features for Real-World Model A v2 (Part A1.3):
1. Length features: log(n_tokens + 1), log(n_lines + 1)
2. Static features: Flawfinder + Cppcheck per-CWE-family counts + totals (104-d)
3. QWEN embeddings: Qwen/Qwen2.5-Coder-0.5B (max 512 tokens, mean-pooled, L2 normalized, 896-d)
4. CodeBERT chunked-mean (1536-d): reuse existing P2 cache where samples overlap, extract remaining.

Processes:
- dev_heldout (data/splits/p2/dev_heldout.parquet, N=2543)
- train_v2 (data/p10/train_v2.parquet, N=25000)

Caches to data/cache/p10/:
- dev_length_features.npy
- dev_static_features.npy
- dev_qwen_embs.npy
- dev_codebert_embs.npy
- train_v2_length_features.npy
- train_v2_static_features.npy
- train_v2_qwen_embs.npy
- train_v2_codebert_embs.npy
"""

import os
import sys
import json
import time
import shutil
import tempfile
import subprocess
import csv
import io
import re
from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from scripts.p10.extract_juliet_features import load_cwe_families, extract_static_features
from scripts.p2.cache_embeddings_p2 import extract_standard_codebert, extract_chunked_codebert

DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
P2_SPLITS_DIR = SPLITS_DIR / "p2"
P10_DATA_DIR = DATA_DIR / "p10"
CACHE_P2_DIR = DATA_DIR / "cache"
CACHE_P10_DIR = DATA_DIR / "cache" / "p10"

DEV_PARQUET = P2_SPLITS_DIR / "dev_heldout.parquet"
TRAIN_V2_PARQUET = P10_DATA_DIR / "train_v2.parquet"

# Cached CodeBERT files from P2
DEV_CODEBERT_P2_CACHE = CACHE_P2_DIR / "emb_codebert_chunked_mean_256_c749c38c040c36f4d03d0b6a516e2d063c8d7c681a43d9bd5b82f02683cb6201.npy"
TRAIN_CODEBERT_P2_CACHE = CACHE_P2_DIR / "emb_codebert_chunked_mean_256_57340bfd48c29815c873827229b905e32c78a4141081d5eda3c443d503c4437d.npy"
TRAIN_MINUS_DEV_PARQUET = P2_SPLITS_DIR / "train_minus_dev.parquet"


def compute_token_counts(codes: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Simple regex word token counts and line counts."""
    n_tokens = []
    n_lines = []
    for c in codes:
        code_str = str(c) if c is not None else ""
        toks = len(re.findall(r"\w+|[^\w\s]", code_str))
        lines = len(code_str.splitlines())
        n_tokens.append(max(1, toks))
        n_lines.append(max(1, lines))
    return np.array(n_tokens, dtype=np.float32), np.array(n_lines, dtype=np.float32)


def extract_length_features_from_codes(codes: List[str]) -> np.ndarray:
    n_tok, n_lin = compute_token_counts(codes)
    return np.column_stack([np.log(n_tok + 1.0), np.log(n_lin + 1.0)]).astype(np.float32)


def extract_qwen_for_dataset(df: pd.DataFrame, out_path: Path, ckpt_path: Path, batch_size: int = 32) -> np.ndarray:
    """Extract QWEN embeddings with checkpointing."""
    if out_path.exists():
        print(f"[*] QWEN embeddings already cached at {out_path.name}")
        return np.load(out_path)

    n_samples = len(df)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[*] Extracting QWEN embeddings for {out_path.name} on {device} (N={n_samples}, batch_size={batch_size})...")

    model_name = "Qwen/Qwen2.5-Coder-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()

    hidden_size = model.config.hidden_size
    all_embs = np.zeros((n_samples, hidden_size), dtype=np.float32)

    start_idx = 0
    if ckpt_path.exists():
        ckpt = np.load(ckpt_path)
        if ckpt.shape[0] <= n_samples and ckpt.shape[1] == hidden_size:
            start_idx = len(ckpt)
            all_embs[:start_idx] = ckpt
            print(f"    Resuming from checkpoint at sample {start_idx}/{n_samples}...")

    codes = df["code"].tolist()
    t0 = time.time()

    for i in range(start_idx, n_samples, batch_size):
        batch = codes[i : min(i + batch_size, n_samples)]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1)
            embs = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embs[i : i + len(batch)] = embs.cpu().to(torch.float32).numpy()

        if (i + len(batch)) % 1000 < batch_size or (i + len(batch)) == n_samples:
            elapsed = time.time() - t0
            done = (i + len(batch)) - start_idx
            rate = done / max(elapsed, 1e-5)
            rem = (n_samples - (i + len(batch))) / max(rate, 1e-5)
            print(f"    Progress: {i + len(batch)}/{n_samples} ({100.0*(i+len(batch))/n_samples:.1f}%) | Rate: {rate:.1f} s/s | Est. remaining: {rem/60.0:.1f} min")

        if (i + len(batch)) % 4000 < batch_size:
            np.save(ckpt_path, all_embs[: i + len(batch)])

    if ckpt_path.exists():
        ckpt_path.unlink()

    np.save(out_path, all_embs)
    print(f"[*] Saved {out_path.name} in {time.time() - t0:.2f}s.")
    return all_embs


def extract_codebert_chunked_for_train_v2(df_train_v2: pd.DataFrame, out_path: Path) -> np.ndarray:
    """Extract or reuse CodeBERT chunked-mean embeddings for train_v2."""
    if out_path.exists():
        print(f"[*] CodeBERT embeddings already cached at {out_path.name}")
        return np.load(out_path)

    n_samples = len(df_train_v2)
    print(f"[*] Building CodeBERT chunked-mean embeddings for train_v2 (N={n_samples})...")
    t0 = time.time()

    all_embs = np.zeros((n_samples, 1536), dtype=np.float32)

    # 1. Reuse existing P2 cached embeddings where sample_id matches train_minus_dev
    if TRAIN_CODEBERT_P2_CACHE.exists() and TRAIN_MINUS_DEV_PARQUET.exists():
        print("    Loading P2 train_minus_dev CodeBERT cache for overlap lookup...")
        df_tmd = pd.read_parquet(TRAIN_MINUS_DEV_PARQUET)
        tmd_embs = np.load(TRAIN_CODEBERT_P2_CACHE)
        id_to_tmd_idx = {sid: i for i, sid in enumerate(df_tmd["sample_id"])}

        matched = 0
        missing_indices = []
        for i, sid in enumerate(df_train_v2["sample_id"]):
            if sid in id_to_tmd_idx:
                all_embs[i] = tmd_embs[id_to_tmd_idx[sid]]
                matched += 1
            else:
                missing_indices.append(i)
        print(f"    Matched {matched}/{n_samples} ({matched/n_samples*100:.1f}%) from P2 cache.")
    else:
        missing_indices = list(range(n_samples))

    # 2. Extract remaining missing samples with CodeBERT
    if missing_indices:
        print(f"    Extracting CodeBERT chunked-mean for {len(missing_indices)} new samples on MPS...")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model_name = "microsoft/codebert-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

        sub_codes = [df_train_v2.iloc[idx]["code"] for idx in missing_indices]
        emb_std_sub = extract_standard_codebert(sub_codes, tokenizer, model, device, batch_size=64, max_length=256)
        emb_mean_sub, _ = extract_chunked_codebert(
            sub_codes, emb_std_sub, tokenizer, model, device, batch_size=64, window_size=254, stride=128, max_windows_per_sample=16, sub_chunk_size=1000
        )
        for rel_i, orig_i in enumerate(missing_indices):
            all_embs[orig_i] = emb_mean_sub[rel_i]
        print(f"    New CodeBERT embeddings extracted in {time.time() - t0:.2f}s.")

    np.save(out_path, all_embs)
    print(f"[*] Saved train_v2_codebert_embs.npy: {all_embs.shape} in {time.time() - t0:.2f}s.")
    return all_embs


def main():
    print("=" * 80)
    print("PART A1.3: EXTRACTING FEATURES FOR REAL-WORLD MODEL A V2")
    print("=" * 80)
    CACHE_P10_DIR.mkdir(parents=True, exist_ok=True)

    families, fam_to_idx, cwe_to_fam_idx = load_cwe_families()

    # ==========================
    # 1. DEV_HELDOUT FEATURES
    # ==========================
    print("\n--- [A] DEV_HELDOUT FEATURES (N=2,543) ---")
    df_dev = pd.read_parquet(DEV_PARQUET)
    print(f"Loaded dev_heldout: {len(df_dev)} rows.")

    dev_len_path = CACHE_P10_DIR / "dev_length_features.npy"
    if dev_len_path.exists():
        print(f"[*] dev length features already cached at {dev_len_path.name}")
    else:
        dev_len = extract_length_features_from_codes(df_dev["code"].tolist())
        np.save(dev_len_path, dev_len)
        print(f"[*] Saved {dev_len_path.name}: {dev_len.shape}")

    dev_static_path = CACHE_P10_DIR / "dev_static_features.npy"
    if dev_static_path.exists():
        print(f"[*] dev static features already cached at {dev_static_path.name}")
    else:
        dev_stat = extract_static_features(df_dev, families, fam_to_idx, cwe_to_fam_idx)
        np.save(dev_static_path, dev_stat)
        print(f"[*] Saved {dev_static_path.name}: {dev_stat.shape}")

    dev_cb_path = CACHE_P10_DIR / "dev_codebert_embs.npy"
    if dev_cb_path.exists():
        print(f"[*] dev CodeBERT embeddings already cached at {dev_cb_path.name}")
    else:
        assert DEV_CODEBERT_P2_CACHE.exists(), f"Missing {DEV_CODEBERT_P2_CACHE}!"
        shutil.copy(DEV_CODEBERT_P2_CACHE, dev_cb_path)
        print(f"[*] Reused P2 dev CodeBERT embeddings -> {dev_cb_path.name}")

    dev_qwen_path = CACHE_P10_DIR / "dev_qwen_embs.npy"
    dev_qwen_ckpt = CACHE_P10_DIR / "dev_qwen_embs_checkpoint.npy"
    extract_qwen_for_dataset(df_dev, dev_qwen_path, dev_qwen_ckpt, batch_size=32)

    # ==========================
    # 2. TRAIN_V2 FEATURES
    # ==========================
    print("\n--- [B] TRAIN_V2 FEATURES (N=25,000) ---")
    df_train_v2 = pd.read_parquet(TRAIN_V2_PARQUET)
    print(f"Loaded train_v2: {len(df_train_v2)} rows.")

    tr_len_path = CACHE_P10_DIR / "train_v2_length_features.npy"
    if tr_len_path.exists():
        print(f"[*] train_v2 length features already cached at {tr_len_path.name}")
    else:
        tr_len = extract_length_features_from_codes(df_train_v2["code"].tolist())
        np.save(tr_len_path, tr_len)
        print(f"[*] Saved {tr_len_path.name}: {tr_len.shape}")

    tr_static_path = CACHE_P10_DIR / "train_v2_static_features.npy"
    if tr_static_path.exists():
        print(f"[*] train_v2 static features already cached at {tr_static_path.name}")
    else:
        tr_stat = extract_static_features(df_train_v2, families, fam_to_idx, cwe_to_fam_idx)
        np.save(tr_static_path, tr_stat)
        print(f"[*] Saved {tr_static_path.name}: {tr_stat.shape}")

    tr_cb_path = CACHE_P10_DIR / "train_v2_codebert_embs.npy"
    extract_codebert_chunked_for_train_v2(df_train_v2, tr_cb_path)

    tr_qwen_path = CACHE_P10_DIR / "train_v2_qwen_embs.npy"
    tr_qwen_ckpt = CACHE_P10_DIR / "train_v2_qwen_embs_checkpoint.npy"
    extract_qwen_for_dataset(df_train_v2, tr_qwen_path, tr_qwen_ckpt, batch_size=32)

    print("\nALL REAL-WORLD MODEL A V2 FEATURES EXTRACTED AND CACHED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
