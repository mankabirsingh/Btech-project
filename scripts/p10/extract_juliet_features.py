#!/usr/bin/env python3
"""
Extract features for Juliet dataset (Part A2.5):
1. Static features: Flawfinder + Cppcheck per-CWE-family counts + totals.
2. Length features: log(n_tokens + 1), log(n_lines + 1).
3. QWEN embeddings: Qwen/Qwen2.5-Coder-0.5B mean-pooled (512 tokens), L2 normalized.

Caches to data/cache/p10/:
- juliet_static_features.npy
- juliet_length_features.npy
- juliet_qwen_embs.npy
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
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "p10"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "p10"
CWE_FAMILIES_FILE = PROJECT_ROOT / "data" / "p7" / "cwe_families.json"

PARQUET_PATH = DATA_DIR / "juliet_extracted_40k.parquet"
STATIC_CACHE_PATH = CACHE_DIR / "juliet_static_features.npy"
LENGTH_CACHE_PATH = CACHE_DIR / "juliet_length_features.npy"
QWEN_CACHE_PATH = CACHE_DIR / "juliet_qwen_embs.npy"
QWEN_CHECKPOINT_PATH = CACHE_DIR / "juliet_qwen_embs_checkpoint.npy"

# Known fallback CWE mappings for common cppcheck IDs
CPPCHECK_ID_TO_CWE = {
    "nullPointer": "476",
    "uninitvar": "457",
    "uninitStructMember": "457",
    "uninitDerivedMember": "457",
    "variableScope": "398",
    "redundantAssignment": "563",
    "unreadVariable": "563",
    "unusedFunction": "561",
    "memleak": "401",
    "resourceLeak": "775",
    "deallocDealloc": "415",
    "doubleFree": "415",
    "deallocuse": "416",
    "bufferAccessOutOfBounds": "119",
    "outOfBounds": "119",
    "arrayIndexOutOfBounds": "129",
    "zerodiv": "369",
    "shiftNegativeLHS": "758",
    "integerOverflow": "190",
}


def load_cwe_families():
    with open(CWE_FAMILIES_FILE, "r") as f:
        fam_dict = json.load(f)
    
    # Sorted list of families
    families = sorted(list(fam_dict.keys()))
    fam_to_idx = {fam: i for i, fam in enumerate(families)}
    
    # Reverse lookup from cwe string/number to family index
    cwe_to_fam_idx = {}
    for fam, cwes in fam_dict.items():
        f_idx = fam_to_idx[fam]
        for c in cwes:
            clean_c = c.lower().replace("cwe-", "").replace("cwe", "").strip()
            cwe_to_fam_idx[clean_c] = f_idx
            
    return families, fam_to_idx, cwe_to_fam_idx


def extract_static_features(df: pd.DataFrame, families, fam_to_idx, cwe_to_fam_idx) -> np.ndarray:
    """Run flawfinder and cppcheck on all code snippets and build feature matrix."""
    n_samples = len(df)
    n_fams = len(families)
    # Features per sample:
    # [0:n_fams] flawfinder family counts
    # [n_fams] flawfinder total hits
    # [n_fams+1] flawfinder max level
    # [n_fams+2 : 2*n_fams+2] cppcheck family counts
    # [2*n_fams+2] cppcheck total hits
    # [2*n_fams+3] cppcheck warning/error count
    total_dim = 2 * n_fams + 4
    feat_matrix = np.zeros((n_samples, total_dim), dtype=np.float32)

    print(f"[*] Extracting static features for {n_samples} samples across {n_fams} CWE families...")
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        print(f"    Dumping snippets to temporary directory {tmp_path}...")
        for i, code in enumerate(df["code"]):
            with open(tmp_path / f"{i}.c", "w") as f:
                f.write(code)
        print(f"    Dump completed in {time.time() - t0:.2f}s.")

        # 1. Run Flawfinder
        print("    Running flawfinder in batch mode...")
        t_ff = time.time()
        flawfinder_bin = str(PROJECT_ROOT / "venv" / "bin" / "flawfinder")
        res_ff = subprocess.run(
            [flawfinder_bin, "--csv", "--quiet", str(tmp_path)],
            capture_output=True,
            text=True
        )
        print(f"    Flawfinder completed in {time.time() - t_ff:.2f}s.")

        reader = csv.DictReader(io.StringIO(res_ff.stdout))
        for row in reader:
            fpath = row.get("File", "")
            fname = Path(fpath).name
            if not fname.endswith(".c"):
                continue
            idx_str = fname[:-2]
            if not idx_str.isdigit():
                continue
            idx = int(idx_str)
            if idx >= n_samples:
                continue

            # Total hit count
            feat_matrix[idx, n_fams] += 1.0

            # Level / severity
            lvl = float(row.get("Level", 1))
            if lvl > feat_matrix[idx, n_fams + 1]:
                feat_matrix[idx, n_fams + 1] = lvl

            # CWEs
            cwe_str = row.get("CWEs", "")
            matches = re.findall(r"\d+", cwe_str)
            for m in matches:
                if m in cwe_to_fam_idx:
                    f_idx = cwe_to_fam_idx[m]
                    feat_matrix[idx, f_idx] += 1.0

        # 2. Run Cppcheck
        print("    Running cppcheck in batch mode (-j8)...")
        t_cpp = time.time()
        cppcheck_bin = "/opt/homebrew/bin/cppcheck"
        res_cpp = subprocess.run(
            [
                cppcheck_bin,
                "--template={file}:{line}:{severity}:{id}:{cwe}",
                "--enable=warning,portability,performance",
                "--inconclusive",
                "-j8",
                "--quiet",
                str(tmp_path)
            ],
            capture_output=True,
            text=True
        )
        print(f"    Cppcheck completed in {time.time() - t_cpp:.2f}s.")

        for line in res_cpp.stderr.splitlines():
            parts = line.strip().split(":")
            if len(parts) < 5:
                continue
            fpath = parts[0]
            fname = Path(fpath).name
            if not fname.endswith(".c"):
                continue
            idx_str = fname[:-2]
            if not idx_str.isdigit():
                continue
            idx = int(idx_str)
            if idx >= n_samples:
                continue

            severity = parts[2]
            err_id = parts[3]
            cwe_raw = parts[4]

            if err_id in ("checkersReport", "syntaxError"):
                continue

            # Total cppcheck hit
            feat_matrix[idx, 2 * n_fams + 2] += 1.0
            if severity in ("error", "warning"):
                feat_matrix[idx, 2 * n_fams + 3] += 1.0

            # Map CWE
            target_f_idx = None
            if cwe_raw and cwe_raw != "0" and cwe_raw in cwe_to_fam_idx:
                target_f_idx = cwe_to_fam_idx[cwe_raw]
            elif err_id in CPPCHECK_ID_TO_CWE:
                c_fallback = CPPCHECK_ID_TO_CWE[err_id]
                if c_fallback in cwe_to_fam_idx:
                    target_f_idx = cwe_to_fam_idx[c_fallback]

            if target_f_idx is not None:
                feat_matrix[idx, n_fams + 2 + target_f_idx] += 1.0

    print(f"[*] Static feature extraction completed in {time.time() - t0:.2f}s. Shape: {feat_matrix.shape}")
    return feat_matrix


def extract_length_features(df: pd.DataFrame) -> np.ndarray:
    """Extract log(n_tokens + 1) and log(n_lines + 1)."""
    n_tokens = df["n_tokens"].values.astype(np.float32)
    n_lines = df["n_lines"].values.astype(np.float32)
    feats = np.column_stack([
        np.log(n_tokens + 1.0),
        np.log(n_lines + 1.0)
    ]).astype(np.float32)
    return feats


def extract_qwen_embeddings(df: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
    """Extract QWEN embeddings with checkpointing and resume."""
    n_samples = len(df)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[*] Extracting QWEN embeddings on {device} (batch_size={batch_size}, max_length=512)...")

    model_name = "Qwen/Qwen2.5-Coder-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()

    hidden_size = model.config.hidden_size
    all_embs = np.zeros((n_samples, hidden_size), dtype=np.float32)

    start_idx = 0
    if QWEN_CHECKPOINT_PATH.exists():
        ckpt = np.load(QWEN_CHECKPOINT_PATH)
        if ckpt.shape[0] <= n_samples and ckpt.shape[1] == hidden_size:
            start_idx = len(ckpt)
            all_embs[:start_idx] = ckpt
            print(f"    Resuming from checkpoint at sample {start_idx}/{n_samples}...")

    codes = df["code"].tolist()
    t0 = time.time()
    last_print = t0

    for i in range(start_idx, n_samples, batch_size):
        batch = codes[i : min(i + batch_size, n_samples)]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1)
            embs = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embs[i : i + len(batch)] = embs.cpu().to(torch.float32).numpy()

        # Print progress every 1000 samples
        if (i + len(batch)) % 1000 < batch_size or (i + len(batch)) == n_samples:
            elapsed = time.time() - t0
            done = (i + len(batch)) - start_idx
            rate = done / max(elapsed, 1e-5)
            rem = (n_samples - (i + len(batch))) / max(rate, 1e-5)
            print(f"    Progress: {i + len(batch)}/{n_samples} ({100.0*(i+len(batch))/n_samples:.1f}%) | Rate: {rate:.1f} samples/s | Est. remaining: {rem/60.0:.1f} min")

        # Checkpoint every 4000 samples
        if (i + len(batch)) % 4000 < batch_size:
            np.save(QWEN_CHECKPOINT_PATH, all_embs[: i + len(batch)])

    # Final save
    if QWEN_CHECKPOINT_PATH.exists():
        QWEN_CHECKPOINT_PATH.unlink()

    print(f"[*] QWEN embedding extraction completed in {time.time() - t0:.2f}s.")
    return all_embs


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded Juliet dataset: {len(df)} samples.")

    # 1. Length features
    if LENGTH_CACHE_PATH.exists():
        print(f"[*] Length features already cached at {LENGTH_CACHE_PATH}.")
    else:
        length_feats = extract_length_features(df)
        np.save(LENGTH_CACHE_PATH, length_feats)
        print(f"[*] Saved length features: {length_feats.shape} to {LENGTH_CACHE_PATH}")

    # 2. Static features
    if STATIC_CACHE_PATH.exists():
        print(f"[*] Static features already cached at {STATIC_CACHE_PATH}.")
    else:
        families, fam_to_idx, cwe_to_fam_idx = load_cwe_families()
        static_feats = extract_static_features(df, families, fam_to_idx, cwe_to_fam_idx)
        np.save(STATIC_CACHE_PATH, static_feats)
        print(f"[*] Saved static features: {static_feats.shape} to {STATIC_CACHE_PATH}")

    # 3. QWEN embeddings
    if QWEN_CACHE_PATH.exists():
        print(f"[*] QWEN embeddings already cached at {QWEN_CACHE_PATH}.")
    else:
        qwen_embs = extract_qwen_embeddings(df, batch_size=32)
        np.save(QWEN_CACHE_PATH, qwen_embs)
        print(f"[*] Saved QWEN embeddings: {qwen_embs.shape} to {QWEN_CACHE_PATH}")

    print("ALL JULIET FEATURES EXTRACTED AND CACHED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
