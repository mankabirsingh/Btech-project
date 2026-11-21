"""Step 3: Fine-Tuning UniXcoder on Length-Debiased and Control Data."""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import scipy.special
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from scripts.p3.eval_protocol import load_datasets
from scripts.p4.step1_paired_data import compute_paired_metrics
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


def get_device() -> torch.device:
    """Detect Apple Silicon MPS, CUDA, or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TokenizedDataset(torch.utils.data.Dataset):
    """Memory-efficient pre-tokenized sequence dataset with optional sample weights."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.weights = weights if weights is not None else torch.ones_like(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "weight": self.weights[idx],
        }


def build_pretokenized_dataset(
    texts: List[str],
    labels: np.ndarray,
    tokenizer: AutoTokenizer,
    weights: Optional[np.ndarray] = None,
    max_length: int = 512,
    batch_size: int = 1000,
) -> TokenizedDataset:
    """Pre-tokenizes raw source codes into fixed-length tensors in batches."""
    all_input_ids = []
    all_attention_mask = []
    n = len(texts)

    for i in range(0, n, batch_size):
        batch_texts = [str(c) if c is not None else "" for c in texts[i : i + batch_size]]
        enc = tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        all_input_ids.append(enc["input_ids"])
        all_attention_mask.append(enc["attention_mask"])

    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_mask, dim=0)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    w_tensor = torch.tensor(weights, dtype=torch.float32) if weights is not None else torch.ones(len(labels), dtype=torch.float32)

    return TokenizedDataset(input_ids, attention_mask, y_tensor, w_tensor)


def predict_sliding_windows(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    codes: List[str],
    device: torch.device,
    window_size: int = 512,
    stride: int = 256,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inference with 512-token windows and stride 256 for sequences > 512 tokens.
    Batched across all windows for high throughput on GPU/MPS.
    Returns:
      max_logits: shape (N, 2)
      mean_logits: shape (N, 2)
    """
    model.eval()
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    all_window_tokens = []
    window_to_code_idx = []

    for idx, code in enumerate(codes):
        code_str = str(code) if code is not None else ""
        toks = tokenizer.encode(code_str, add_special_tokens=False, truncation=False)

        if len(toks) <= window_size - 2:
            all_window_tokens.append([cls_id] + toks + [sep_id])
            window_to_code_idx.append(idx)
        else:
            for start in range(0, len(toks), stride):
                w = toks[start : start + window_size - 2]
                if not w:
                    break
                all_window_tokens.append([cls_id] + w + [sep_id])
                window_to_code_idx.append(idx)
                if start + window_size - 2 >= len(toks):
                    break

    window_to_code_idx = np.array(window_to_code_idx)
    num_windows = len(all_window_tokens)

    all_logits = []
    with torch.no_grad():
        for i in range(0, num_windows, batch_size):
            batch_tokens = all_window_tokens[i : i + batch_size]
            max_len = max(len(w) for w in batch_tokens)
            padded = [w + [pad_id] * (max_len - len(w)) for w in batch_tokens]
            masks = [[1] * len(w) + [0] * (max_len - len(w)) for w in batch_tokens]

            b_ids = torch.tensor(padded, dtype=torch.long, device=device)
            b_mask = torch.tensor(masks, dtype=torch.long, device=device)

            logits = model(b_ids, attention_mask=b_mask).logits
            all_logits.append(logits.detach().cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)

    num_codes = len(codes)
    max_logits = np.full((num_codes, 2), -1e9, dtype=np.float32)
    mean_logits = np.zeros((num_codes, 2), dtype=np.float32)

    for idx in range(num_codes):
        w_mask = (window_to_code_idx == idx)
        code_w_logits = all_logits[w_mask]
        max_logits[idx] = np.max(code_w_logits, axis=0)
        mean_logits[idx] = np.mean(code_w_logits, axis=0)

    return max_logits, mean_logits


def evaluate_on_fold0(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    y_val: np.ndarray,
    tok_val: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """Fast validation on fold 0 early stopping set using first 512 tokens."""
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in val_loader:
            b_ids = batch["input_ids"].to(device)
            b_mask = batch["attention_mask"].to(device)
            out = model(b_ids, attention_mask=b_mask)
            all_logits.append(out.logits.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    probs = scipy.special.softmax(logits, axis=1)[:, 1]

    val_auroc = float(roc_auc_score(y_val, probs))
    val_ls_auroc = float(calc_length_stratified_auroc(y_val, probs, tok_val))

    return val_ls_auroc, val_auroc, probs


def train_unixcoder_run(
    run_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tok_val: np.ndarray,
    train_weights: Optional[np.ndarray] = None,
    seed: int = 42,
    max_epochs: int = 3,
    batch_size: int = 16,
    grad_accum: int = 2,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    patience: int = 3,
    eval_every_steps: int = 250,
    save_every_steps: int = 500,
) -> Dict[str, Any]:
    """Trains UniXcoder with resumable checkpointing, linear warmup, and early stopping on fold 0."""
    print("\n" + "=" * 80, flush=True)
    print(f"STARTING FINE-TUNING RUN: {run_name} (Seed: {seed})", flush=True)
    print("=" * 80, flush=True)

    device = get_device()
    print(f"Compute Device: {device}", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    ckpt_dir = P4_MODELS_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_csv = RESULTS_P4_DIR / f"{run_name}_log.csv"

    # Load Tokenizer
    model_name = "microsoft/unixcoder-base"
    print(f"Loading tokenizer: {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare datasets
    print(f"Building pre-tokenized training set ({len(train_df)} samples)...", flush=True)
    train_dataset = build_pretokenized_dataset(
        texts=train_df["code"].tolist(),
        labels=train_df["vuln_label"].values,
        tokenizer=tokenizer,
        weights=train_weights,
        max_length=512,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    print(f"Building pre-tokenized early-stopping validation set ({len(val_df)} samples)...", flush=True)
    val_dataset = build_pretokenized_dataset(
        texts=val_df["code"].tolist(),
        labels=val_df["vuln_label"].values,
        tokenizer=tokenizer,
        weights=None,
        max_length=512,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        drop_last=False,
    )

    # Initialize model
    print(f"Initializing {model_name} sequence classification head...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # Total optimization steps
    steps_per_epoch = int(np.ceil(len(train_loader) / grad_accum))
    total_opt_steps = steps_per_epoch * max_epochs
    warmup_steps = int(0.10 * total_opt_steps)
    print(f"Optimization Steps: {total_opt_steps} (Warmup: {warmup_steps}, Steps/epoch: {steps_per_epoch})", flush=True)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps,
    )

    criterion = nn.CrossEntropyLoss(reduction="none")

    # Resumable checkpoint check
    trainer_state_file = ckpt_dir / "trainer_state.pt"
    start_step = 0
    start_epoch = 0
    best_val_ls = -1.0
    patience_counter = 0

    if trainer_state_file.exists():
        print(f"Found existing trainer state at {trainer_state_file}. Resuming...", flush=True)
        checkpoint = torch.load(trainer_state_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = checkpoint["step"]
        start_epoch = checkpoint["epoch"]
        best_val_ls = checkpoint["best_val_ls"]
        patience_counter = checkpoint["patience_counter"]
        print(f"Resumed from step {start_step}, epoch {start_epoch}, best LS-AUROC: {best_val_ls:.4f}", flush=True)
    else:
        # Initialize log CSV
        with open(log_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "train_loss", "val_ls_auroc", "val_auroc", "is_best", "elapsed_s"])

    # Training Loop
    global_step = start_step
    opt_step = start_step
    running_loss = 0.0
    loss_count = 0
    t_train_start = time.time()
    t_step_100_start = time.time()

    stop_early = False

    for epoch in range(start_epoch, max_epochs):
        if stop_early:
            break
        model.train()
        print(f"\n--- Epoch {epoch + 1}/{max_epochs} ---", flush=True)

        for batch_idx, batch in enumerate(train_loader):
            b_ids = batch["input_ids"].to(device)
            b_mask = batch["attention_mask"].to(device)
            b_labels = batch["labels"].to(device)
            b_weights = batch["weight"].to(device)

            outputs = model(b_ids, attention_mask=b_mask)
            unreduced_loss = criterion(outputs.logits, b_labels)
            loss = (unreduced_loss * b_weights).mean() / grad_accum
            loss.backward()

            running_loss += loss.item() * grad_accum
            loss_count += 1

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                # Time projection after 100 optimization steps
                if opt_step == 100:
                    elapsed_100 = time.time() - t_step_100_start
                    time_per_step = elapsed_100 / 100.0
                    proj_total_hours = (time_per_step * total_opt_steps) / 3600.0
                    print(f"\n>>> 100 Steps Benchmark: {time_per_step:.2f} s/step. Projected total training time: {proj_total_hours:.2f} hours.", flush=True)
                    if proj_total_hours > 4.0:
                        print(f"WARNING: Projected training time ({proj_total_hours:.2f}h) exceeds 4.0 hour budget! Stopping run.", flush=True)
                        stop_early = True
                        break

                # Periodic evaluation on early-stopping fold 0
                if opt_step > 0 and opt_step % eval_every_steps == 0:
                    val_ls, val_auc, _ = evaluate_on_fold0(
                        model=model,
                        val_loader=val_loader,
                        device=device,
                        y_val=val_df["vuln_label"].values,
                        tok_val=tok_val,
                    )
                    avg_train_loss = running_loss / max(1, loss_count)
                    running_loss = 0.0
                    loss_count = 0

                    is_best = bool(val_ls > best_val_ls)
                    if is_best:
                        best_val_ls = val_ls
                        patience_counter = 0
                        torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
                        best_marker = " [NEW BEST]"
                    else:
                        patience_counter += 1
                        best_marker = f" (Patience: {patience_counter}/{patience})"

                    elapsed_total = round(time.time() - t_train_start, 1)
                    print(
                        f"Step {opt_step}/{total_opt_steps} | Loss: {avg_train_loss:.4f} | "
                        f"Fold0 Val LS-AUROC: {val_ls:.4f}{best_marker} | Val AUROC: {val_auc:.4f} | Time: {elapsed_total}s",
                        flush=True,
                    )

                    # Log to CSV
                    with open(log_csv, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([opt_step, epoch + 1, round(avg_train_loss, 4), round(val_ls, 4), round(val_auc, 4), is_best, elapsed_total])

                    model.train()

                    # Early stopping condition
                    if patience_counter >= patience:
                        print(f"\nEarly stopping triggered at step {opt_step}! (No validation LS-AUROC gain for {patience} checks).", flush=True)
                        stop_early = True
                        break

                # Periodic checkpointing
                if opt_step > 0 and opt_step % save_every_steps == 0:
                    state = {
                        "step": opt_step,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_ls": best_val_ls,
                        "patience_counter": patience_counter,
                    }
                    torch.save(state, trainer_state_file)
                    print(f"Saved trainer state checkpoint to {trainer_state_file}", flush=True)

    # Load best checkpoint for inference
    best_model_file = ckpt_dir / "best_model.pt"
    if best_model_file.exists():
        print(f"\nLoading best model checkpoint from {best_model_file} (Val LS-AUROC: {best_val_ls:.4f})...", flush=True)
        model.load_state_dict(torch.load(best_model_file, map_location=device))
    else:
        print("\nNo separate best model file found; using final epoch weights.", flush=True)

    training_summary = {
        "run_name": run_name,
        "seed": seed,
        "total_steps": opt_step,
        "best_fold0_ls_auroc": best_val_ls,
        "wall_time_seconds": round(time.time() - t_train_start, 2),
        "early_stopped": stop_early,
    }

    return training_summary, model, tokenizer


def evaluate_unixcoder_on_dev(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    df_val_f0: pd.DataFrame,
    df_dev: pd.DataFrame,
    tok_dev: np.ndarray,
    df_pairs: pd.DataFrame,
    run_name: str,
) -> Dict[str, Any]:
    """Runs sliding-window inference on fold 0 (for 10% FPR threshold) and held-out dev."""
    device = get_device()
    print(f"\n--- Running Sliding Window Inference for {run_name} on fold 0 & dev_heldout ---", flush=True)

    # 1. Fold 0 inference to select 10% FPR threshold
    print("  Evaluating fold 0 with sliding windows to set 10% FPR operating threshold...", flush=True)
    f0_max_log, f0_mean_log = predict_sliding_windows(model, tokenizer, df_val_f0["code"].tolist(), device)
    p_f0_max = scipy.special.softmax(f0_max_log, axis=1)[:, 1]
    p_f0_mean = scipy.special.softmax(f0_mean_log, axis=1)[:, 1]

    tau_max = float(threshold_at_fpr(df_val_f0["vuln_label"].values, p_f0_max, target_fpr=0.10)["threshold"])
    tau_mean = float(threshold_at_fpr(df_val_f0["vuln_label"].values, p_f0_mean, target_fpr=0.10)["threshold"])
    print(f"  Operating Threshold at 10% FPR: max_logit tau={tau_max:.4f}, mean_logit tau={tau_mean:.4f}", flush=True)

    # 2. dev_heldout inference
    print("  Evaluating dev_heldout with sliding windows (512 tokens, stride 256)...", flush=True)
    dev_max_log, dev_mean_log = predict_sliding_windows(model, tokenizer, df_dev["code"].tolist(), device)
    p_dev_max = scipy.special.softmax(dev_max_log, axis=1)[:, 1]
    p_dev_mean = scipy.special.softmax(dev_mean_log, axis=1)[:, 1]

    # Save dev predictions
    np.save(RESULTS_P4_DIR / f"p_dev_{run_name.lower()}_max.npy", p_dev_max)
    np.save(RESULTS_P4_DIR / f"p_dev_{run_name.lower()}_mean.npy", p_dev_mean)

    y_dev = df_dev["vuln_label"].values
    is_cpp = df_dev["is_cpp"].astype(bool).values

    results_by_pooling = {}

    for pool_name, p_dv, tau in [("max_logit", p_dev_max, tau_max), ("mean_logit", p_dev_mean, tau_mean)]:
        dev_auroc = float(roc_auc_score(y_dev, p_dv))
        dev_ls_auroc = float(calc_length_stratified_auroc(y_dev, p_dv, tok_dev))
        dev_cpp_auroc = float(roc_auc_score(y_dev[is_cpp], p_dv[is_cpp])) if np.sum(is_cpp) > 0 else 0.5
        dev_cpp_ls_auroc = float(calc_length_stratified_auroc(y_dev[is_cpp], p_dv[is_cpp], tok_dev[is_cpp])) if np.sum(is_cpp) > 0 else 0.5
        dev_auprc = float(average_precision_score(y_dev, p_dv))

        ci_ov = bootstrap_ci(y_dev, p_dv, metric_fn=lambda y, p, _: roc_auc_score(y, p), n_resamples=2000, seed=42)
        ci_ls = length_stratified_auroc(y_dev, p_dv, tok_dev, n_resamples=2000, seed=42)["ci_95"]
        ci_cpp = bootstrap_ci(y_dev[is_cpp], p_dv[is_cpp], metric_fn=lambda y, p, _: roc_auc_score(y, p), n_resamples=2000, seed=42)
        ci_cpp_ls = length_stratified_auroc(y_dev[is_cpp], p_dv[is_cpp], tok_dev[is_cpp], n_resamples=2000, seed=42)["ci_95"]
        ci_auprc = bootstrap_ci(y_dev, p_dv, metric_fn=lambda y, p, _: average_precision_score(y, p), n_resamples=2000, seed=42)

        metrics_at_tau = metrics_full(y_dev, p_dv, threshold=tau)

        # Paired metrics on dev_pairs
        dev_id_to_idx = {sid: i for i, sid in enumerate(df_dev["sample_id"])}
        v_idx = [dev_id_to_idx[sid] for sid in df_pairs["vuln_sample_id"]]
        f_idx = [dev_id_to_idx[sid] for sid in df_pairs["fixed_sample_id"]]
        paired_eval = compute_paired_metrics(p_dv[v_idx], p_dv[f_idx], threshold=tau, n_resamples=2000, seed=42)

        results_by_pooling[pool_name] = {
            "operating_threshold_10fpr": float(tau),
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
        }

        print(f"\n  {run_name} [{pool_name}] on dev_heldout:")
        print(f"    Dev AUROC:                   {dev_auroc:.4f} [95% CI: {ci_ov[0]:.4f}, {ci_ov[1]:.4f}]")
        print(f"    Dev Length-Stratified AUROC: {dev_ls_auroc:.4f} [95% CI: {ci_ls[0]:.4f}, {ci_ls[1]:.4f}]")
        print(f"    Dev C++ AUROC:               {dev_cpp_auroc:.4f} [95% CI: {ci_cpp[0]:.4f}, {ci_cpp[1]:.4f}]")
        print(f"    pair_correct:                {paired_eval['pair_correct']*100:.2f}% [95% CI: {paired_eval['pair_correct_ci_95'][0]*100:.2f}%, {paired_eval['pair_correct_ci_95'][1]*100:.2f}%]")
        print(f"    At 10% FPR Threshold ({tau:.4f}): Actual Dev FPR = {metrics_at_tau['fpr']*100:.2f}%, Recall = {metrics_at_tau['recall']*100:.2f}%", flush=True)

    return results_by_pooling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, choices=["all", "F1", "F2", "F3"], default="all")
    args = parser.parse_args()

    t_all_start = time.time()
    print("=" * 80, flush=True)
    print("STEP 3: UNIXCODER FINE-TUNING PIPELINE", flush=True)
    print("=" * 80, flush=True)

    # Load Step 0 draw variance std
    with open(RESULTS_P4_DIR / "step0_resolutions.json", "r") as f:
        step0_res = json.load(f)
    draw_std_ls = step0_res["draw_variance_0_2"]["dev_ls_auroc_std"]

    # Load Step 2 debiased model results to identify best debiased dataset
    with open(RESULTS_P4_DIR / "step2_debiased_models.json", "r") as f:
        step2_res = json.load(f)
    best_debiased = step2_res["metadata"]["best_debiased_dataset"]
    print(f"Best debiased dataset identified from Step 2: {best_debiased}", flush=True)

    # Load datasets
    df_train, df_dev, tok_train, tok_dev = load_datasets()
    df_pairs = pd.read_parquet(P4_SPLITS_DIR / "dev_pairs.parquet")

    # Early stopping fold 0 slice
    fold0_mask = (df_train["kfold"] == 0).values
    val_df_f0 = df_train[fold0_mask].copy().reset_index(drop=True)
    tok_val_f0 = tok_train[fold0_mask]

    # Training candidate pools (excluding fold 0)
    train_pool_df = df_train[~fold0_mask].copy().reset_index(drop=True)

    all_finetune_results = {}
    if (RESULTS_P4_DIR / "step3_finetuning_eval.json").exists():
        try:
            with open(RESULTS_P4_DIR / "step3_finetuning_eval.json", "r") as f:
                all_finetune_results = json.load(f)
        except Exception:
            all_finetune_results = {}

    # Prepare Debiased Training Pool & Weights (for F1 and F3)
    f1_weights = None
    if best_debiased == "D1":
        # Length-stratified balanced sampling on train_pool_df
        bins = [0, 256, 512, 1024, np.inf]
        tok_pool = tok_train[~fold0_mask]
        bin_idx = np.digitize(tok_pool, bins) - 1
        y_pool = train_pool_df["vuln_label"].values

        rng = np.random.RandomState(42)
        f1_indices = []
        for b in range(4):
            in_b = bin_idx == b
            v_idx = np.where(in_b & (y_pool == 1))[0]
            s_idx = np.where(in_b & (y_pool == 0))[0]
            c_v = rng.choice(v_idx, size=2500, replace=(len(v_idx) < 2500))
            c_s = rng.choice(s_idx, size=2500, replace=(len(s_idx) < 2500))
            f1_indices.extend(c_v)
            f1_indices.extend(c_s)
        f1_train_df = train_pool_df.iloc[f1_indices].copy().reset_index(drop=True)
    elif best_debiased == "D2":
        # Inverse-propensity weights on train_pool_df
        tok_pool = tok_train[~fold0_mask]
        bins = [0, 256, 512, 1024, np.inf]
        bin_idx = np.digitize(tok_pool, bins) - 1
        y_pool = train_pool_df["vuln_label"].values

        f1_weights = np.zeros(len(y_pool), dtype=np.float32)
        for b in range(4):
            in_b = bin_idx == b
            v_mask = in_b & (y_pool == 1)
            s_mask = in_b & (y_pool == 0)
            n_v = np.sum(v_mask)
            n_s = np.sum(s_mask)
            target_sum = (n_v + n_s) / 2.0
            f1_weights[v_mask] = target_sum / max(1, n_v)
            f1_weights[s_mask] = target_sum / max(1, n_s)
        # Normalize so mean weight is 1.0
        f1_weights = f1_weights / np.mean(f1_weights)
        f1_train_df = train_pool_df
    else:
        # If D3, use paired training
        f1_train_df = train_pool_df

    # -------------------------------------------------------------------------
    # RUN F1: UniXcoder + Best Debiased Dataset (Seed 42)
    # -------------------------------------------------------------------------
    if args.run in ["all", "F1"]:
        summary_f1, model_f1, tok_f1 = train_unixcoder_run(
            run_name="F1_unixcoder_debiased",
            train_df=f1_train_df,
            val_df=val_df_f0,
            tok_val=tok_val_f0,
            train_weights=f1_weights,
            seed=42,
        )
        eval_f1 = evaluate_unixcoder_on_dev(
            model=model_f1,
            tokenizer=tok_f1,
            df_val_f0=val_df_f0,
            df_dev=df_dev,
            tok_dev=tok_dev,
            df_pairs=df_pairs,
            run_name="F1_unixcoder_debiased",
        )
        all_finetune_results["F1"] = {
            "summary": summary_f1,
            "evaluation": eval_f1,
        }
        with open(RESULTS_P4_DIR / "step3_finetuning_eval.json", "w") as f:
            json.dump(all_finetune_results, f, indent=2)

    # -------------------------------------------------------------------------
    # RUN F2: UniXcoder + Natural N0 Data (Seed 42, Control)
    # -------------------------------------------------------------------------
    if args.run in ["all", "F2"]:
        # Natural balanced training set (N0 control on the 4 training folds)
        v_idx = np.where(train_pool_df["vuln_label"] == 1)[0]
        s_idx = np.where(train_pool_df["vuln_label"] == 0)[0]
        rng = np.random.RandomState(42)
        c_v = rng.choice(v_idx, size=10000, replace=True)
        c_s = rng.choice(s_idx, size=10000, replace=True)
        f2_train_df = train_pool_df.iloc[np.concatenate([c_v, c_s])].copy().reset_index(drop=True)

        summary_f2, model_f2, tok_f2 = train_unixcoder_run(
            run_name="F2_unixcoder_natural_control",
            train_df=f2_train_df,
            val_df=val_df_f0,
            tok_val=tok_val_f0,
            train_weights=None,
            seed=42,
        )
        eval_f2 = evaluate_unixcoder_on_dev(
            model=model_f2,
            tokenizer=tok_f2,
            df_val_f0=val_df_f0,
            df_dev=df_dev,
            tok_dev=tok_dev,
            df_pairs=df_pairs,
            run_name="F2_unixcoder_natural_control",
        )
        all_finetune_results["F2"] = {
            "summary": summary_f2,
            "evaluation": eval_f2,
        }
        with open(RESULTS_P4_DIR / "step3_finetuning_eval.json", "w") as f:
            json.dump(all_finetune_results, f, indent=2)

    # -------------------------------------------------------------------------
    # RUN F3: Repeat F1 with seed 7 (ONLY IF F1 beats L1 on dev LS-AUROC > draw_std)
    # -------------------------------------------------------------------------
    # L1 dev LS-AUROC is 0.5878 (from Step 0 / P3)
    # Check condition strictly
    l1_ls_auroc = 0.5878
    f1_max_ls = all_finetune_results.get("F1", {}).get("evaluation", {}).get("max_logit", {}).get("dev_length_stratified_auroc", 0.0)
    delta_f1_l1 = f1_max_ls - l1_ls_auroc

    print("\n--- Checking F3 Trigger Condition ---", flush=True)
    print(f"  F1 dev LS-AUROC: {f1_max_ls:.4f}", flush=True)
    print(f"  L1 dev LS-AUROC: {l1_ls_auroc:.4f}", flush=True)
    print(f"  Delta(F1 - L1): {delta_f1_l1:+.4f} vs Draw Std: {draw_std_ls:.4f}", flush=True)

    if delta_f1_l1 > draw_std_ls:
        print(f"  Condition MET: Delta ({delta_f1_l1:.4f}) > draw std ({draw_std_ls:.4f}). Executing F3 (seed 7)...", flush=True)
        if args.run in ["all", "F3"]:
            summary_f3, model_f3, tok_f3 = train_unixcoder_run(
                run_name="F3_unixcoder_debiased_seed7",
                train_df=f1_train_df,
                val_df=val_df_f0,
                tok_val=tok_val_f0,
                train_weights=f1_weights,
                seed=7,
            )
            eval_f3 = evaluate_unixcoder_on_dev(
                model=model_f3,
                tokenizer=tok_f3,
                df_val_f0=val_df_f0,
                df_dev=df_dev,
                tok_dev=tok_dev,
                df_pairs=df_pairs,
                run_name="F3_unixcoder_debiased_seed7",
            )
            all_finetune_results["F3"] = {
                "summary": summary_f3,
                "evaluation": eval_f3,
            }
            with open(RESULTS_P4_DIR / "step3_finetuning_eval.json", "w") as f:
                json.dump(all_finetune_results, f, indent=2)
    else:
        print(f"  Condition NOT MET: Delta ({delta_f1_l1:.4f}) <= draw std ({draw_std_ls:.4f}). Skipping F3.", flush=True)
        all_finetune_results["F3"] = {
            "status": "skipped",
            "reason": f"F1 dev LS-AUROC ({f1_max_ls:.4f}) did not beat L1 ({l1_ls_auroc:.4f}) by more than draw std ({draw_std_ls:.4f})",
        }
        with open(RESULTS_P4_DIR / "step3_finetuning_eval.json", "w") as f:
            json.dump(all_finetune_results, f, indent=2)

    print(f"\nStep 3 completed in {time.time() - t_all_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
