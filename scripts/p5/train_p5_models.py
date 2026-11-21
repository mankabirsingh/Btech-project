"""Step 3: Training Pipeline for Model A Multi-Seed and Pairwise Models.

Executes:
- C1: D2 recipe (UniXcoder + inverse-propensity weights) on seeds 1, 2, 3.
- C2: PAIR recipe (UniXcoder + 12 single + 2 pairs per step, margin loss) on seeds 42, 7, 1.
- C3: N0 recipe (Frozen CodeBERT CLS+mean + cxx28, C=0.01) on seed 7.

Saves:
- models/checkpoints/p5/<run_name>/best_model.pt
- results/p5/<run_name>_log.csv
- results/p5/p_dev_<run_name>_max.npy
- results/p5/p_dev_<run_name>_mean.npy
- results/p5/logits_dev_<run_name>_max.npy
- results/p5/train_summary_<run_name>.json
"""

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
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from scripts.p3.eval_protocol import get_cached_features, load_datasets
from src.eval.stats import calc_length_stratified_auroc

DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
CACHE_DIR = DATA_DIR / "cache"
P2_SPLITS_DIR = SPLITS_DIR / "p2"
P4_SPLITS_DIR = SPLITS_DIR / "p4"
P5_SPLITS_DIR = SPLITS_DIR / "p5"
RESULTS_P4_DIR = PROJECT_ROOT / "results" / "p4"
P5_MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints" / "p5"
RESULTS_P5_DIR = PROJECT_ROOT / "results" / "p5"

P5_MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_P5_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pretokenize_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    batch_size: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    return input_ids, attention_mask


def predict_sliding_windows(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    codes: List[str],
    device: torch.device,
    window_size: int = 512,
    stride: int = 256,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inference with 512-token windows and stride 256 for sequences > 512 tokens.
    Returns:
      max_probs: shape (N,)
      mean_probs: shape (N,)
      max_logits: shape (N,) (logit diff s = logit[1] - logit[0])
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

    max_logits_diff = np.full(num_codes, -1e9, dtype=np.float32)
    max_probs = np.zeros(num_codes, dtype=np.float32)
    mean_probs = np.zeros(num_codes, dtype=np.float32)

    for idx in range(num_codes):
        w_mask = (window_to_code_idx == idx)
        code_w_logits = all_logits[w_mask]
        # Softmax per window
        code_w_probs = scipy.special.softmax(code_w_logits, axis=1)[:, 1]
        code_w_diff = code_w_logits[:, 1] - code_w_logits[:, 0]

        max_probs[idx] = np.max(code_w_probs)
        mean_probs[idx] = np.mean(code_w_probs)
        max_logits_diff[idx] = np.max(code_w_diff)

    return max_probs, mean_probs, max_logits_diff


def evaluate_on_fold0(
    model: nn.Module,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    y_val: np.ndarray,
    tok_val: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[float, float]:
    """Fast validation on fold 0 early stopping set using first 512 tokens."""
    model.eval()
    all_logits = []
    n = len(val_ids)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            b_ids = val_ids[i : i + batch_size].to(device)
            b_mask = val_mask[i : i + batch_size].to(device)
            out = model(b_ids, attention_mask=b_mask)
            all_logits.append(out.logits.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    probs = scipy.special.softmax(logits, axis=1)[:, 1]

    val_auroc = float(roc_auc_score(y_val, probs))
    val_ls_auroc = float(calc_length_stratified_auroc(y_val, probs, tok_val))

    return val_ls_auroc, val_auroc


def evaluate_fold0_pairs(
    model: nn.Module,
    v_ids: torch.Tensor,
    v_mask: torch.Tensor,
    f_ids: torch.Tensor,
    f_mask: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """Evaluates pair_correct on fold0 pairs."""
    model.eval()
    n = len(v_ids)
    v_logits, f_logits = [], []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            out_v = model(v_ids[i : i + batch_size].to(device), attention_mask=v_mask[i : i + batch_size].to(device))
            out_f = model(f_ids[i : i + batch_size].to(device), attention_mask=f_mask[i : i + batch_size].to(device))
            v_logits.append(out_v.logits.detach().cpu().numpy())
            f_logits.append(out_f.logits.detach().cpu().numpy())

    v_log = np.concatenate(v_logits, axis=0)
    f_log = np.concatenate(f_logits, axis=0)

    p_v = scipy.special.softmax(v_log, axis=1)[:, 1]
    p_f = scipy.special.softmax(f_log, axis=1)[:, 1]

    pair_corr = float(np.mean(p_v > p_f) + 0.5 * np.mean(p_v == p_f))
    return pair_corr


# ==============================================================================
# C1: D2 Training Run (Seeds 1, 2, 3)
# ==============================================================================
def train_d2_run(
    seed: int,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    train_labels: torch.Tensor,
    train_weights: torch.Tensor,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    y_val: np.ndarray,
    tok_val: np.ndarray,
    df_dev: pd.DataFrame,
    device: torch.device,
    tokenizer: AutoTokenizer,
) -> None:
    run_name = f"D2_seed{seed}"
    ckpt_dir = P5_MODELS_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_model_file = ckpt_dir / "best_model.pt"

    pred_max_file = RESULTS_P5_DIR / f"p_dev_{run_name}_max.npy"
    if best_model_file.exists() and pred_max_file.exists():
        print(f"\n[SKIP] {run_name} already completed and evaluated.", flush=True)
        return

    print("\n" + "=" * 80, flush=True)
    print(f"RUNNING C1: D2 RECIPE (Seed: {seed})", flush=True)
    print("=" * 80, flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model_name = "microsoft/unixcoder-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    batch_size = 16
    grad_accum = 2
    lr = 2e-5
    weight_decay = 0.01
    max_epochs = 3
    eval_every = 200
    patience = 3

    n_samples = len(train_ids)
    steps_per_epoch = int(np.ceil((n_samples / batch_size) / grad_accum))
    total_opt_steps = steps_per_epoch * max_epochs
    warmup_steps = int(0.10 * total_opt_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_opt_steps)
    criterion = nn.CrossEntropyLoss(reduction="none")

    log_csv = RESULTS_P5_DIR / f"{run_name}_log.csv"
    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "train_loss", "val_ls_auroc", "val_auroc", "is_best", "elapsed_s"])

    t_start = time.time()
    opt_step = 0
    best_val_ls = -1.0
    patience_counter = 0
    running_loss = 0.0
    loss_count = 0
    stop_early = False

    perm_rng = np.random.RandomState(seed)

    for epoch in range(max_epochs):
        if stop_early:
            break
        model.train()
        indices = perm_rng.permutation(n_samples)

        for b_start in range(0, n_samples, batch_size):
            b_idx = indices[b_start : b_start + batch_size]
            b_ids = train_ids[b_idx].to(device)
            b_mask = train_mask[b_idx].to(device)
            b_lbl = train_labels[b_idx].to(device)
            b_wt = train_weights[b_idx].to(device)

            out = model(b_ids, attention_mask=b_mask)
            unred_loss = criterion(out.logits, b_lbl)
            loss = (unred_loss * b_wt).mean() / grad_accum
            loss.backward()

            running_loss += loss.item() * grad_accum
            loss_count += 1

            if (b_start // batch_size + 1) % grad_accum == 0 or (b_start + batch_size) >= n_samples:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                if opt_step > 0 and opt_step % eval_every == 0:
                    val_ls, val_auc = evaluate_on_fold0(model, val_ids, val_mask, y_val, tok_val, device)
                    avg_loss = running_loss / max(1, loss_count)
                    running_loss, loss_count = 0.0, 0

                    is_best = bool(val_ls > best_val_ls)
                    if is_best:
                        best_val_ls = val_ls
                        patience_counter = 0
                        torch.save(model.state_dict(), best_model_file)
                        marker = " [NEW BEST]"
                    else:
                        patience_counter += 1
                        marker = f" (Patience {patience_counter}/{patience})"

                    elapsed = round(time.time() - t_start, 1)
                    print(f"Step {opt_step}/{total_opt_steps} | Loss: {avg_loss:.4f} | Val LS-AUROC: {val_ls:.4f}{marker} | Time: {elapsed}s", flush=True)

                    with open(log_csv, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([opt_step, epoch + 1, round(avg_loss, 4), round(val_ls, 4), round(val_auc, 4), is_best, elapsed])

                    model.train()
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at step {opt_step}!", flush=True)
                        stop_early = True
                        break

    print(f"\nLoading best checkpoint for {run_name} (Val LS-AUROC: {best_val_ls:.4f})...", flush=True)
    model.load_state_dict(torch.load(best_model_file, map_location=device))

    print(f"Running dev_heldout sliding window inference...", flush=True)
    p_max, p_mean, logit_max = predict_sliding_windows(model, tokenizer, df_dev["code"].tolist(), device)

    np.save(RESULTS_P5_DIR / f"p_dev_{run_name}_max.npy", p_max)
    np.save(RESULTS_P5_DIR / f"p_dev_{run_name}_mean.npy", p_mean)
    np.save(RESULTS_P5_DIR / f"logits_dev_{run_name}_max.npy", logit_max)

    y_dev = df_dev["vuln_label"].values
    tok_dev = np.load(CACHE_DIR / "dev_token_lens.npy")
    dev_ls = float(calc_length_stratified_auroc(y_dev, p_max, tok_dev))
    dev_auc = float(roc_auc_score(y_dev, p_max))

    summary = {
        "run_name": run_name,
        "seed": seed,
        "total_opt_steps": opt_step,
        "best_fold0_ls_auroc": best_val_ls,
        "dev_ls_auroc": dev_ls,
        "dev_auroc": dev_auc,
        "wall_time_s": round(time.time() - t_start, 2),
    }
    with open(P5_MODELS_DIR / run_name / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{run_name} finished! Dev LS-AUROC = {dev_ls:.4f}, Dev AUROC = {dev_auc:.4f}", flush=True)


# ==============================================================================
# C2: PAIR Training Run (Seeds 42, 7, 1)
# ==============================================================================
def train_pair_run(
    seed: int,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    train_labels: torch.Tensor,
    train_weights: torch.Tensor,
    pair_v_ids: torch.Tensor,
    pair_v_mask: torch.Tensor,
    pair_f_ids: torch.Tensor,
    pair_f_mask: torch.Tensor,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    y_val: np.ndarray,
    tok_val: np.ndarray,
    f0_pair_v_ids: torch.Tensor,
    f0_pair_v_mask: torch.Tensor,
    f0_pair_f_ids: torch.Tensor,
    f0_pair_f_mask: torch.Tensor,
    df_dev: pd.DataFrame,
    device: torch.device,
    tokenizer: AutoTokenizer,
) -> None:
    run_name = f"PAIR_seed{seed}"
    ckpt_dir = P5_MODELS_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_model_file = ckpt_dir / "best_model.pt"

    pred_max_file = RESULTS_P5_DIR / f"p_dev_{run_name}_max.npy"
    if best_model_file.exists() and pred_max_file.exists():
        print(f"\n[SKIP] {run_name} already completed and evaluated.", flush=True)
        return

    print("\n" + "=" * 80, flush=True)
    print(f"RUNNING C2: PAIR RECIPE (Seed: {seed})", flush=True)
    print("=" * 80, flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model_name = "microsoft/unixcoder-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # Batch composition: 12 single samples + 2 pairs (4 functions) = 16 functions per step
    single_per_step = 12
    pairs_per_step = 2
    lr = 2e-5
    weight_decay = 0.01
    max_epochs = 3
    eval_every = 200
    patience = 3

    n_single = len(train_ids)
    n_pairs = len(pair_v_ids)
    steps_per_epoch = int(np.ceil(n_single / single_per_step))
    total_opt_steps = steps_per_epoch * max_epochs
    warmup_steps = int(0.10 * total_opt_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_opt_steps)
    criterion = nn.CrossEntropyLoss(reduction="none")

    log_csv = RESULTS_P5_DIR / f"{run_name}_log.csv"
    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "total_loss", "single_loss", "pair_loss", "val_metric", "val_ls_auroc", "val_pair_corr", "is_best", "elapsed_s"])

    t_start = time.time()
    opt_step = 0
    best_val_metric = -1.0
    patience_counter = 0
    running_loss = 0.0
    running_s_loss = 0.0
    running_p_loss = 0.0
    loss_count = 0
    stop_early = False

    perm_rng = np.random.RandomState(seed)

    for epoch in range(max_epochs):
        if stop_early:
            break
        model.train()
        single_indices = perm_rng.permutation(n_single)
        pair_indices = perm_rng.permutation(n_pairs)

        pair_ptr = 0

        for b_start in range(0, n_single, single_per_step):
            s_idx = single_indices[b_start : b_start + single_per_step]
            if len(s_idx) < single_per_step:
                continue

            # Pick 2 pairs
            if pair_ptr + pairs_per_step > n_pairs:
                pair_indices = perm_rng.permutation(n_pairs)
                pair_ptr = 0
            p_idx = pair_indices[pair_ptr : pair_ptr + pairs_per_step]
            pair_ptr += pairs_per_step

            # Batch tensors: 12 single + 2 vuln + 2 fixed = 16
            b_ids = torch.cat([train_ids[s_idx], pair_v_ids[p_idx], pair_f_ids[p_idx]], dim=0).to(device)
            b_mask = torch.cat([train_mask[s_idx], pair_v_mask[p_idx], pair_f_mask[p_idx]], dim=0).to(device)
            s_lbl = train_labels[s_idx].to(device)
            s_wt = train_weights[s_idx].to(device)

            out = model(b_ids, attention_mask=b_mask)
            logits = out.logits

            logits_single = logits[:single_per_step]
            logits_v = logits[single_per_step : single_per_step + pairs_per_step]
            logits_f = logits[single_per_step + pairs_per_step :]

            # Single weighted loss
            unred_loss = criterion(logits_single, s_lbl)
            loss_single = (unred_loss * s_wt).mean()

            # Pair ranking softplus loss: margin logit_v - logit_f
            diff_v = logits_v[:, 1] - logits_v[:, 0]
            diff_f = logits_f[:, 1] - logits_f[:, 0]
            loss_pair = torch.mean(torch.nn.functional.softplus(-(diff_v - diff_f)))

            total_loss = loss_single + 1.0 * loss_pair
            total_loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            opt_step += 1

            running_loss += total_loss.item()
            running_s_loss += loss_single.item()
            running_p_loss += loss_pair.item()
            loss_count += 1

            if opt_step > 0 and opt_step % eval_every == 0:
                val_ls, val_auc = evaluate_on_fold0(model, val_ids, val_mask, y_val, tok_val, device)
                val_pair_corr = evaluate_fold0_pairs(
                    model, f0_pair_v_ids, f0_pair_v_mask, f0_pair_f_ids, f0_pair_f_mask, device
                )
                composite_val = 0.5 * (val_ls + val_pair_corr)

                avg_tot = running_loss / max(1, loss_count)
                avg_s = running_s_loss / max(1, loss_count)
                avg_p = running_p_loss / max(1, loss_count)
                running_loss, running_s_loss, running_p_loss, loss_count = 0.0, 0.0, 0.0, 0

                is_best = bool(composite_val > best_val_metric)
                if is_best:
                    best_val_metric = composite_val
                    patience_counter = 0
                    torch.save(model.state_dict(), best_model_file)
                    marker = " [NEW BEST]"
                else:
                    patience_counter += 1
                    marker = f" (Patience {patience_counter}/{patience})"

                elapsed = round(time.time() - t_start, 1)
                print(
                    f"Step {opt_step}/{total_opt_steps} | Tot Loss: {avg_tot:.4f} (S: {avg_s:.4f}, P: {avg_p:.4f}) | "
                    f"Val Comp: {composite_val:.4f}{marker} (LS: {val_ls:.4f}, PairCorr: {val_pair_corr:.4f}) | Time: {elapsed}s",
                    flush=True,
                )

                with open(log_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        opt_step, epoch + 1, round(avg_tot, 4), round(avg_s, 4), round(avg_p, 4),
                        round(composite_val, 4), round(val_ls, 4), round(val_pair_corr, 4), is_best, elapsed
                    ])

                model.train()
                if patience_counter >= patience:
                    print(f"Early stopping triggered at step {opt_step}!", flush=True)
                    stop_early = True
                    break

    print(f"\nLoading best checkpoint for {run_name} (Val Comp: {best_val_metric:.4f})...", flush=True)
    model.load_state_dict(torch.load(best_model_file, map_location=device))

    print(f"Running dev_heldout sliding window inference...", flush=True)
    p_max, p_mean, logit_max = predict_sliding_windows(model, tokenizer, df_dev["code"].tolist(), device)

    np.save(RESULTS_P5_DIR / f"p_dev_{run_name}_max.npy", p_max)
    np.save(RESULTS_P5_DIR / f"p_dev_{run_name}_mean.npy", p_mean)
    np.save(RESULTS_P5_DIR / f"logits_dev_{run_name}_max.npy", logit_max)

    y_dev = df_dev["vuln_label"].values
    tok_dev = np.load(CACHE_DIR / "dev_token_lens.npy")
    dev_ls = float(calc_length_stratified_auroc(y_dev, p_max, tok_dev))
    dev_auc = float(roc_auc_score(y_dev, p_max))

    summary = {
        "run_name": run_name,
        "seed": seed,
        "total_opt_steps": opt_step,
        "best_composite_val": best_val_metric,
        "dev_ls_auroc": dev_ls,
        "dev_auroc": dev_auc,
        "wall_time_s": round(time.time() - t_start, 2),
    }
    with open(P5_MODELS_DIR / run_name / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{run_name} finished! Dev LS-AUROC = {dev_ls:.4f}, Dev AUROC = {dev_auc:.4f}", flush=True)


# ==============================================================================
# C3: N0 Control (Seed 7)
# ==============================================================================
def train_n0_seed7(df_train: pd.DataFrame, df_dev: pd.DataFrame, tok_dev: np.ndarray) -> None:
    print("\n" + "=" * 80)
    print("RUNNING C3: N0 CONTROL (Seed: 7)")
    print("=" * 80)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    y_train = df_train["vuln_label"].values.astype(int)
    y_dev = df_dev["vuln_label"].values.astype(int)
    feats = get_cached_features()

    vuln_idx = np.where(y_train == 1)[0]
    safe_idx = np.where(y_train == 0)[0]
    n_class = len(vuln_idx)

    rng = np.random.RandomState(7)
    draw_v = rng.choice(vuln_idx, size=n_class, replace=True)
    draw_s = rng.choice(safe_idx, size=n_class, replace=True)
    draw_idx = np.concatenate([draw_v, draw_s])
    rng.shuffle(draw_idx)

    sc_cxx = StandardScaler()
    X_tr_cxx = sc_cxx.fit_transform(feats["train"]["cxx28"][draw_idx])
    X_dv_cxx = sc_cxx.transform(feats["dev"]["cxx28"])

    X_train_n0 = np.hstack([feats["train"]["emb_codebert_256"][draw_idx], X_tr_cxx])
    X_dev_n0 = np.hstack([feats["dev"]["emb_codebert_256"], X_dv_cxx])

    clf_n0 = LogisticRegression(C=0.01, class_weight="balanced", max_iter=1000, random_state=7)
    clf_n0.fit(X_train_n0, y_train[draw_idx])

    p_dev_n0 = clf_n0.predict_proba(X_dev_n0)[:, 1]
    logits_dev_n0 = clf_n0.decision_function(X_dev_n0)

    np.save(RESULTS_P5_DIR / "p_dev_N0_seed7_max.npy", p_dev_n0)
    np.save(RESULTS_P5_DIR / "logits_dev_N0_seed7_max.npy", logits_dev_n0)

    auc = float(roc_auc_score(y_dev, p_dev_n0))
    ls_auc = float(calc_length_stratified_auroc(y_dev, p_dev_n0, tok_dev))

    print(f"N0 Seed 7 Finished! Dev AUROC = {auc:.4f}, LS-AUROC = {ls_auc:.4f}")

    # Also make sure N0 seed 42 logits are saved
    p_dev_n0_42 = clf_n0 = LogisticRegression(C=0.01, class_weight="balanced", max_iter=1000, random_state=42)
    clf_n0.fit(np.hstack([feats["train"]["emb_codebert_256"], sc_cxx.fit_transform(feats["train"]["cxx28"])]), y_train)
    np.save(RESULTS_P5_DIR / "p_dev_N0_seed42_max.npy", clf_n0.predict_proba(X_dev_n0)[:, 1])
    np.save(RESULTS_P5_DIR / "logits_dev_N0_seed42_max.npy", clf_n0.decision_function(X_dev_n0))


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="all", help="all, C1, C2, C3, or specific name like C1_seed1")
    args = parser.parse_args()

    t_all_start = time.time()
    device = get_device()
    print(f"Device: {device}", flush=True)

    # 1. Load Datasets
    df_train, df_dev, tok_train, tok_dev = load_datasets()
    print(f"Loaded train_minus_dev: {len(df_train)} samples, dev_heldout: {len(df_dev)} samples", flush=True)

    # Split train_minus_dev: folds 1-4 for train, fold 0 for validation
    train_df = df_train[df_train["kfold"] != 0].copy().reset_index(drop=True)
    val_df = df_train[df_train["kfold"] == 0].copy().reset_index(drop=True)
    tok_val = tok_train[df_train["kfold"] == 0]

    # Load D2 weights
    d2_weights_all = np.load(P4_SPLITS_DIR / "d2_train_weights.npy")
    d2_weights_train = d2_weights_all[df_train["kfold"] != 0]

    # Load Pairs
    df_train_pairs = pd.read_parquet(P5_SPLITS_DIR / "train_pairs.parquet")
    df_val_pairs = pd.read_parquet(P5_SPLITS_DIR / "fold0_pairs.parquet")
    print(f"Loaded train pairs: {len(df_train_pairs)}, fold0 val pairs: {len(df_val_pairs)}", flush=True)

    # Pre-tokenize all data once
    tokenizer_name = "microsoft/unixcoder-base"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    cache_tokenized_file = CACHE_DIR / "p5_pretokenized_tensors.pt"
    if cache_tokenized_file.exists():
        print(f"Loading cached pre-tokenized tensors from {cache_tokenized_file}...", flush=True)
        cached = torch.load(cache_tokenized_file)
        train_ids = cached["train_ids"]
        train_mask = cached["train_mask"]
        val_ids = cached["val_ids"]
        val_mask = cached["val_mask"]
        pair_v_ids = cached["pair_v_ids"]
        pair_v_mask = cached["pair_v_mask"]
        pair_f_ids = cached["pair_f_ids"]
        pair_f_mask = cached["pair_f_mask"]
        f0_pair_v_ids = cached["f0_pair_v_ids"]
        f0_pair_v_mask = cached["f0_pair_v_mask"]
        f0_pair_f_ids = cached["f0_pair_f_ids"]
        f0_pair_f_mask = cached["f0_pair_f_mask"]
    else:
        print("Pre-tokenizing training sequences...", flush=True)
        train_ids, train_mask = pretokenize_texts(train_df["code"].tolist(), tokenizer, max_length=512)
        print("Pre-tokenizing validation sequences...", flush=True)
        val_ids, val_mask = pretokenize_texts(val_df["code"].tolist(), tokenizer, max_length=512)
        print("Pre-tokenizing training pairs...", flush=True)
        pair_v_ids, pair_v_mask = pretokenize_texts(df_train_pairs["vuln_code"].tolist(), tokenizer, max_length=512)
        pair_f_ids, pair_f_mask = pretokenize_texts(df_train_pairs["fixed_code"].tolist(), tokenizer, max_length=512)
        print("Pre-tokenizing fold0 pairs...", flush=True)
        f0_pair_v_ids, f0_pair_v_mask = pretokenize_texts(df_val_pairs["vuln_code"].tolist(), tokenizer, max_length=512)
        f0_pair_f_ids, f0_pair_f_mask = pretokenize_texts(df_val_pairs["fixed_code"].tolist(), tokenizer, max_length=512)

        torch.save({
            "train_ids": train_ids, "train_mask": train_mask,
            "val_ids": val_ids, "val_mask": val_mask,
            "pair_v_ids": pair_v_ids, "pair_v_mask": pair_v_mask,
            "pair_f_ids": pair_f_ids, "pair_f_mask": pair_f_mask,
            "f0_pair_v_ids": f0_pair_v_ids, "f0_pair_v_mask": f0_pair_v_mask,
            "f0_pair_f_ids": f0_pair_f_ids, "f0_pair_f_mask": f0_pair_f_mask,
        }, cache_tokenized_file)
        print(f"Saved pre-tokenized tensors to {cache_tokenized_file}", flush=True)

    train_labels = torch.tensor(train_df["vuln_label"].values, dtype=torch.long)
    train_weights = torch.tensor(d2_weights_train, dtype=torch.float32)
    y_val = val_df["vuln_label"].values

    # Run C3: N0 Control
    if args.run in ["all", "C3", "C3_seed7"]:
        train_n0_seed7(df_train, df_dev, tok_dev)

    # Run C1: D2 Seeds 1, 2, 3
    if args.run in ["all", "C1"]:
        for s in [1, 2, 3]:
            train_d2_run(s, train_ids, train_mask, train_labels, train_weights, val_ids, val_mask, y_val, tok_val, df_dev, device, tokenizer)
    elif args.run.startswith("C1_seed"):
        s = int(args.run.replace("C1_seed", ""))
        train_d2_run(s, train_ids, train_mask, train_labels, train_weights, val_ids, val_mask, y_val, tok_val, df_dev, device, tokenizer)

    # Run C2: PAIR Seeds 42, 7, 1
    if args.run in ["all", "C2"]:
        for s in [42, 7, 1]:
            train_pair_run(
                s, train_ids, train_mask, train_labels, train_weights,
                pair_v_ids, pair_v_mask, pair_f_ids, pair_f_mask,
                val_ids, val_mask, y_val, tok_val,
                f0_pair_v_ids, f0_pair_v_mask, f0_pair_f_ids, f0_pair_f_mask,
                df_dev, device, tokenizer,
            )
    elif args.run.startswith("C2_seed"):
        s = int(args.run.replace("C2_seed", ""))
        train_pair_run(
            s, train_ids, train_mask, train_labels, train_weights,
            pair_v_ids, pair_v_mask, pair_f_ids, pair_f_mask,
            val_ids, val_mask, y_val, tok_val,
            f0_pair_v_ids, f0_pair_v_mask, f0_pair_f_ids, f0_pair_f_mask,
            df_dev, device, tokenizer,
        )

    # Also link F1 (seed 42) and F3 (seed 7) predictions to D2_seed42 and D2_seed7
    print("\nEnsuring D2 seed 42 and seed 7 prediction files are mapped...", flush=True)
    for p_src, p_dst in [
        (RESULTS_P4_DIR / "p_dev_f1_unixcoder_debiased_max.npy", RESULTS_P5_DIR / "p_dev_D2_seed42_max.npy"),
        (RESULTS_P4_DIR / "p_dev_f1_unixcoder_debiased_mean.npy", RESULTS_P5_DIR / "p_dev_D2_seed42_mean.npy"),
        (RESULTS_P4_DIR / "p_dev_f3_unixcoder_debiased_seed7_max.npy", RESULTS_P5_DIR / "p_dev_D2_seed7_max.npy"),
        (RESULTS_P4_DIR / "p_dev_f3_unixcoder_debiased_seed7_mean.npy", RESULTS_P5_DIR / "p_dev_D2_seed7_mean.npy"),
    ]:
        if p_src.exists() and not p_dst.exists():
            import shutil
            shutil.copy(p_src, p_dst)
            # Compute logits
            probs = np.load(p_src)
            probs_clip = np.clip(probs, 1e-6, 1.0 - 1e-6)
            logits = scipy.special.logit(probs_clip)
            l_dst = str(p_dst).replace("p_dev_", "logits_dev_")
            np.save(l_dst, logits)

    print(f"\nALL TRAINING PIPELINE OPERATIONS COMPLETED IN {time.time() - t_all_start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
