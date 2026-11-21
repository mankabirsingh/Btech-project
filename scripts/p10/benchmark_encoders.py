#!/usr/bin/env python3
"""
scripts/p10/benchmark_encoders.py
=================================
Part 0.2: Feature-extractor choice benchmark.
Times Qwen/Qwen2.5-Coder-0.5B and Qwen/Qwen2.5-Coder-1.5B on 100 functions:
- fp16 on MPS
- max 1024 tokens
- mean-pool last hidden state over non-pad tokens
- projects total embedding time across ~74,500 functions (40k Juliet + 33.6k real-world + 0.9k Model B)
- decides between 1.5B (if <= 90 min), 0.5B (if <= 90 min), or 0.5B at 512 tokens.
Writes choice to results/p10/encoder_choice.json.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_P10_DIR = PROJECT_ROOT / "results" / "p10"
RESULTS_P10_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RESULTS_P10_DIR / "encoder_choice.json"

TOTAL_PROJECTED_FUNCTIONS = 74500  # 40k Juliet + 25k train + 3.1k dev + 3.2k old test + 2.5k test2 + 0.9k Model B


def benchmark_model(model_name: str, sample_codes: list, max_len: int = 1024, batch_size: int = 16) -> dict:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nBenchmarking {model_name} (max_len={max_len}, batch_size={batch_size}) on {len(sample_codes)} functions on {device}...")

    t_load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()
    load_time = time.time() - t_load_start
    print(f"  Model loaded in {load_time:.2f}s")

    # Warmup
    warmup_batch = sample_codes[:4]
    inputs_w = tokenizer(warmup_batch, max_length=max_len, truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**inputs_w)
    if device.type == "mps":
        torch.mps.synchronize()

    # Time inference on 100 functions
    t_inf_start = time.time()
    embeddings = []
    for i in range(0, len(sample_codes), batch_size):
        batch = sample_codes[i:i + batch_size]
        inputs = tokenizer(batch, max_length=max_len, truncation=True, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state  # [B, L, D]
            mask = inputs["attention_mask"].unsqueeze(-1)  # [B, L, 1]
            sum_h = (last_hidden * mask).sum(dim=1)
            sum_m = mask.sum(dim=1).clamp(min=1)
            emb = (sum_h / sum_m).cpu().numpy()
            embeddings.append(emb)

    if device.type == "mps":
        torch.mps.synchronize()
    inf_time = time.time() - t_inf_start

    all_embs = np.concatenate(embeddings, axis=0)
    time_per_func_s = inf_time / len(sample_codes)
    projected_total_s = time_per_func_s * TOTAL_PROJECTED_FUNCTIONS
    projected_total_min = projected_total_s / 60.0

    print(f"  Inference time for {len(sample_codes)} functions: {inf_time:.2f}s ({time_per_func_s*1000:.1f} ms/func)")
    print(f"  Projected total embedding time ({TOTAL_PROJECTED_FUNCTIONS} funcs): {projected_total_min:.1f} min ({projected_total_s/3600:.2f} hours)")
    print(f"  Output embedding shape: {all_embs.shape}")

    del model
    del tokenizer
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "model_name": model_name,
        "max_len": max_len,
        "batch_size": batch_size,
        "sample_count": len(sample_codes),
        "inference_time_s": float(inf_time),
        "ms_per_function": float(time_per_func_s * 1000),
        "projected_total_functions": TOTAL_PROJECTED_FUNCTIONS,
        "projected_total_min": float(projected_total_min),
        "projected_total_hours": float(projected_total_s / 3600.0),
        "hidden_dim": int(all_embs.shape[1])
    }


def main():
    print("=" * 80)
    print("PART 0.2: ENCODER SELECTION BENCHMARK")
    print("=" * 80)

    # Load 100 functions from train_minus_dev
    df = pd.read_parquet("data/splits/p2/train_minus_dev.parquet")
    sample_codes = df["code"].iloc[:100].tolist()
    print(f"Loaded {len(sample_codes)} sample functions from train_minus_dev.parquet.")

    # 1. Benchmark Qwen 0.5B at 1024
    res_05b_1024 = benchmark_model("Qwen/Qwen2.5-Coder-0.5B", sample_codes, max_len=1024, batch_size=16)

    # 2. Benchmark Qwen 1.5B at 1024
    # Note: If 0.5B alone is already > 90 min, 1.5B will be even slower
    try_15b = (res_05b_1024["projected_total_min"] < 60.0)
    res_15b = None
    if try_15b:
        print("\nQwen 0.5B projected time is under 60 min; benchmarking Qwen 1.5B...")
        try:
            res_15b = benchmark_model("Qwen/Qwen2.5-Coder-1.5B", sample_codes, max_len=1024, batch_size=8)
        except Exception as e:
            print(f"Error benchmarking Qwen 1.5B: {e}")
    else:
        print("\nSkipping Qwen 1.5B because Qwen 0.5B projected time is already >= 60 min.")

    # 3. Check rule:
    # Use 1.5B only if total projected embedding time for all Parts <= 90 min; else 0.5B; if 0.5B > 90 min, 0.5B at 512 tokens.
    if res_15b is not None and res_15b["projected_total_min"] <= 90.0:
        chosen_model = "Qwen/Qwen2.5-Coder-1.5B"
        chosen_max_len = 1024
        chosen_res = res_15b
        decision_reason = "Qwen 1.5B projected embedding time <= 90 min"
    elif res_05b_1024["projected_total_min"] <= 90.0:
        chosen_model = "Qwen/Qwen2.5-Coder-0.5B"
        chosen_max_len = 1024
        chosen_res = res_05b_1024
        decision_reason = "Qwen 1.5B projected time > 90 min or skipped; Qwen 0.5B projected time <= 90 min"
    else:
        # Benchmark 0.5B at 512 tokens
        print("\nQwen 0.5B at 1024 tokens exceeds 90 min; benchmarking Qwen 0.5B at 512 tokens...")
        res_05b_512 = benchmark_model("Qwen/Qwen2.5-Coder-0.5B", sample_codes, max_len=512, batch_size=16)
        chosen_model = "Qwen/Qwen2.5-Coder-0.5B"
        chosen_max_len = 512
        chosen_res = res_05b_512
        decision_reason = "Qwen 0.5B at 1024 tokens > 90 min; fallback to 0.5B at 512 tokens"

    result = {
        "chosen_model": chosen_model,
        "chosen_max_length": chosen_max_len,
        "decision_reason": decision_reason,
        "benchmark_0_5b_1024": res_05b_1024,
        "benchmark_1_5b_1024": res_15b,
        "benchmark_chosen": chosen_res,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 80)
    print(f"ENCODER CHOICE: {chosen_model} (max_length={chosen_max_len})")
    print(f"Reason: {decision_reason}")
    print(f"Projected total embedding time: {chosen_res['projected_total_min']:.1f} min ({chosen_res['projected_total_hours']:.2f} hours)")
    print(f"Written to {OUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
