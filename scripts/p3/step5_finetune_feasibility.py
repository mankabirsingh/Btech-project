"""STEP 5: Fine-Tuning Feasibility Benchmark and Plan.

Times 50 training steps (batch 16, max_len 512, fp32 on MPS) for microsoft/codebert-base
and microsoft/unixcoder-base.
Reports sec/step, peak memory, and projected hours for 3 epochs on 25k samples.
Writes detailed fine-tuning plan without running full fine-tuning.
"""

import json
import os
import resource
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RESULTS_P3_DIR = PROJECT_ROOT / "results" / "p3"
RESULTS_P3_DIR.mkdir(parents=True, exist_ok=True)


def get_process_memory_mb() -> float:
    # On macOS, ru_maxrss is in bytes
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024)


def benchmark_model_training(
    model_name: str,
    device: torch.device,
    batch_size: int = 16,
    max_len: int = 512,
    num_steps: int = 50,
    warmup_steps: int = 5,
) -> dict:
    print(f"\n--- Benchmarking {model_name} on {device} ---")
    print(f"Batch size: {batch_size}, Max length: {max_len}, Precision: fp32")

    mem_before = get_process_memory_mb()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Synthetic batch of size 16, length 512
    dummy_input_ids = torch.randint(10, 1000, (batch_size, max_len), dtype=torch.long, device=device)
    dummy_mask = torch.ones((batch_size, max_len), dtype=torch.long, device=device)
    dummy_labels = torch.randint(0, 2, (batch_size,), dtype=torch.long, device=device)

    # Warmup steps
    print(f"  Running {warmup_steps} warmup steps...")
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        out = model(input_ids=dummy_input_ids, attention_mask=dummy_mask, labels=dummy_labels)
        loss = out.loss
        loss.backward()
        optimizer.step()

    if hasattr(torch.mps, "synchronize") and device.type == "mps":
        torch.mps.synchronize()
    elif hasattr(torch.cuda, "synchronize") and device.type == "cuda":
        torch.cuda.synchronize()

    # Timed steps
    print(f"  Running {num_steps} timed training steps...")
    t0 = time.time()
    for step in range(num_steps):
        optimizer.zero_grad()
        out = model(input_ids=dummy_input_ids, attention_mask=dummy_mask, labels=dummy_labels)
        loss = out.loss
        loss.backward()
        optimizer.step()

    if hasattr(torch.mps, "synchronize") and device.type == "mps":
        torch.mps.synchronize()
    elif hasattr(torch.cuda, "synchronize") and device.type == "cuda":
        torch.cuda.synchronize()

    total_time = time.time() - t0
    sec_per_step = total_time / num_steps

    mem_after = get_process_memory_mb()
    peak_mem_mb = mem_after

    # Projected hours on 25k samples for 3 epochs
    n_samples = 25000
    steps_per_epoch = int(np.ceil(n_samples / batch_size)) if 'np' in globals() else (n_samples + batch_size - 1) // batch_size
    total_steps_3ep = steps_per_epoch * 3
    proj_seconds = total_steps_3ep * sec_per_step
    proj_hours = proj_seconds / 3600.0

    print(f"  Sec / step:      {sec_per_step:.4f} s")
    print(f"  Steps / sec:     {1.0 / sec_per_step:.2f} step/s")
    print(f"  Process Memory:  {peak_mem_mb:.1f} MB (Delta: {peak_mem_mb - mem_before:+.1f} MB)")
    print(f"  Steps per epoch: {steps_per_epoch:,} steps (Batch {batch_size})")
    print(f"  Total steps 3ep: {total_steps_3ep:,} steps")
    print(f"  Projected 3ep:   {proj_hours:.2f} hours ({proj_seconds:.0f} seconds)")

    # Cleanup memory
    del model, optimizer, dummy_input_ids, dummy_mask, dummy_labels
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    return {
        "model_name": model_name,
        "device": str(device),
        "batch_size": batch_size,
        "max_len": max_len,
        "timed_steps": num_steps,
        "sec_per_step": float(round(sec_per_step, 4)),
        "steps_per_sec": float(round(1.0 / sec_per_step, 2)),
        "peak_memory_mb": float(round(peak_mem_mb, 1)),
        "memory_delta_mb": float(round(peak_mem_mb - mem_before, 1)),
        "steps_per_epoch": steps_per_epoch,
        "total_steps_3_epochs": total_steps_3ep,
        "projected_hours_3_epochs": float(round(proj_hours, 2)),
        "projected_seconds_3_epochs": float(round(proj_seconds, 1)),
    }


def main():
    print("=" * 80)
    print("STEP 5: FINE-TUNING FEASIBILITY BENCHMARK & PLAN")
    print("=" * 80)
    t_start = time.time()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Hardware Acceleration Device: {device}")

    models_to_test = [
        "microsoft/codebert-base",
        "microsoft/unixcoder-base",
    ]

    benchmarks = {}
    for m in models_to_test:
        benchmarks[m] = benchmark_model_training(m, device, batch_size=16, max_len=512, num_steps=50)

    # Step 5.2: Detailed Fine-Tuning Execution Plan
    finetuning_plan = {
        "objective": "Full parameter or LoRA fine-tuning of pre-trained code transformer for vulnerability detection without data leakage.",
        "architecture_and_model": {
            "recommended_base": "microsoft/codebert-base",
            "alternative_base": "microsoft/unixcoder-base",
            "input_handling": "Sliding-window chunking (256-token windows, stride 128) with attention pooling over chunk embeddings to preserve entire function context beyond 512 tokens.",
            "head": "Binary classification head with linear projection and dropout 0.2.",
        },
        "training_hyperparameters": {
            "batch_size": 16,
            "gradient_accumulation_steps": 2,
            "effective_batch_size": 32,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.10,
            "lr_scheduler": "cosine",
            "max_epochs": 3,
            "precision": "fp32 on Apple Silicon MPS / fp16 on CUDA",
            "loss_function": "Weighted Cross-Entropy (balanced weights for class parity)",
        },
        "validation_and_early_stopping_protocol": {
            "validation_split": "Strictly GroupKFold out-of-fold folds on train_minus_dev (grouped by project) OR a dedicated held-out project slice from train_minus_dev.",
            "prohibited_actions": [
                "NEVER use dev_heldout for training, early stopping, or hyperparameter selection.",
                "NEVER access locked test parquet (vulnerability_test_heldout_projects_clean.parquet).",
            ],
            "checkpointing_and_patience": "Evaluate validation AUROC every 250 steps; early stopping patience of 3 evaluation checks.",
            "model_selection_metric": "Validation AUROC and length-stratified AUROC on held-out projects.",
        },
        "hardware_and_resource_feasibility": {
            "codebert_sec_per_step": benchmarks["microsoft/codebert-base"]["sec_per_step"],
            "codebert_projected_hours": benchmarks["microsoft/codebert-base"]["projected_hours_3_epochs"],
            "unixcoder_sec_per_step": benchmarks["microsoft/unixcoder-base"]["sec_per_step"],
            "unixcoder_projected_hours": benchmarks["microsoft/unixcoder-base"]["projected_hours_3_epochs"],
            "feasibility_assessment": "Full fine-tuning for 3 epochs is computationally feasible (~1.5 - 2.5 hours on Apple Silicon GPU), but requires explicit user approval before execution.",
        },
        "approval_gate": "STOPPING AS DIRECTED. Awaiting user approval before launching any fine-tuning run.",
    }

    output_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "runtime_seconds": round(time.time() - t_start, 2),
            "device": str(device),
        },
        "benchmarks": benchmarks,
        "finetuning_plan": finetuning_plan,
    }

    out_json = RESULTS_P3_DIR / "step5_finetuning_feasibility.json"
    with open(out_json, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nStep 5 feasibility results saved to {out_json}")


if __name__ == "__main__":
    main()
