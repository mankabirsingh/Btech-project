"""Step 5: Packaging Model A Final (D2 Ensemble).

Creates models/checkpoints/p5/model_a_final/ with:
- weights/ (all 5 seed checkpoints)
- predictor.py (standalone prediction module with predict(code_list) entry point)
- __init__.py (exposing predict)
- manifest.json (SHA-256 checksums, metadata, and dev metrics)
- tests/test_model_a_final.py (unit test verifying predict() on 5 sample functions)
"""

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FINAL_DIR = PROJECT_ROOT / "models" / "checkpoints" / "p5" / "model_a_final"
WEIGHTS_DIR = FINAL_DIR / "weights"
RESULTS_P5_DIR = PROJECT_ROOT / "results" / "p5"

FINAL_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("PACKAGING MODEL A FINAL (D2 ENSEMBLE)")
    print("=" * 80)

    # 1. Source weight files
    source_weights = {
        "d2_seed42.pt": PROJECT_ROOT / "models" / "checkpoints" / "p4" / "F1_unixcoder_debiased" / "best_model.pt",
        "d2_seed7.pt": PROJECT_ROOT / "models" / "checkpoints" / "p4" / "F3_unixcoder_debiased_seed7" / "best_model.pt",
        "d2_seed1.pt": PROJECT_ROOT / "models" / "checkpoints" / "p5" / "D2_seed1" / "best_model.pt",
        "d2_seed2.pt": PROJECT_ROOT / "models" / "checkpoints" / "p5" / "D2_seed2" / "best_model.pt",
        "d2_seed3.pt": PROJECT_ROOT / "models" / "checkpoints" / "p5" / "D2_seed3" / "best_model.pt",
    }

    checksums = {}
    for dst_name, src_path in source_weights.items():
        dst_path = WEIGHTS_DIR / dst_name
        if not dst_path.exists():
            print(f"Copying {src_path.name} -> {dst_name}...")
            shutil.copy(src_path, dst_path)
        else:
            print(f"Weight file {dst_name} already exists.")
        print(f"Computing SHA-256 for {dst_name}...")
        checksums[dst_name] = sha256_file(dst_path)
        print(f"  {dst_name}: {checksums[dst_name]}")

    # 2. Write predictor.py
    predictor_code = '''"""Model A Final (D2 UniXcoder 5-Seed Ensemble) Inference Module."""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import scipy.special
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PACKAGE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = PACKAGE_DIR / "weights"


def get_default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ModelAFinalEnsemble:
    """UniXcoder 5-seed debiased ensemble predictor."""

    def __init__(
        self,
        weights_dir: Optional[Path] = None,
        device: Optional[Union[str, torch.device]] = None,
        lazy_load: bool = True,
    ):
        self.weights_dir = Path(weights_dir) if weights_dir else WEIGHTS_DIR
        if device is None:
            self.device = get_default_device()
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        self.tokenizer_name = "microsoft/unixcoder-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.seed_files = [
            self.weights_dir / "d2_seed42.pt",
            self.weights_dir / "d2_seed7.pt",
            self.weights_dir / "d2_seed1.pt",
            self.weights_dir / "d2_seed2.pt",
            self.weights_dir / "d2_seed3.pt",
        ]
        self.models: List[nn.Module] = []
        if not lazy_load:
            self._load_models()

    def _load_models(self):
        if self.models:
            return
        for wf in self.seed_files:
            if not wf.exists():
                raise FileNotFoundError(f"Missing weight file: {wf}")
            m = AutoModelForSequenceClassification.from_pretrained(self.tokenizer_name, num_labels=2)
            m.load_state_dict(torch.load(wf, map_location=self.device))
            m.to(self.device)
            m.eval()
            self.models.append(m)

    def predict_sliding_windows_for_model(
        self,
        model: nn.Module,
        codes: List[str],
        window_size: int = 512,
        stride: int = 256,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Computes max logit difference (logit[1] - logit[0]) across sliding windows for one model."""
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        all_window_tokens = []
        window_to_code_idx = []

        for idx, code in enumerate(codes):
            code_str = str(code) if code is not None else ""
            toks = self.tokenizer.encode(code_str, add_special_tokens=False, truncation=False)

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

                b_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
                b_mask = torch.tensor(masks, dtype=torch.long, device=self.device)

                logits = model(b_ids, attention_mask=b_mask).logits
                all_logits.append(logits.detach().cpu().numpy())

        all_logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, 2), dtype=np.float32)
        num_codes = len(codes)
        max_logits_diff = np.full(num_codes, -1e9, dtype=np.float32)

        for idx in range(num_codes):
            w_mask = (window_to_code_idx == idx)
            code_w_logits = all_logits[w_mask]
            if len(code_w_logits) > 0:
                code_w_diff = code_w_logits[:, 1] - code_w_logits[:, 0]
                max_logits_diff[idx] = np.max(code_w_diff)
            else:
                max_logits_diff[idx] = 0.0

        return max_logits_diff

    def predict(
        self,
        code_list: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Main entry point for Model A vulnerability scoring.
        Accepts a list of code strings and returns predicted vulnerability probabilities in [0, 1].
        """
        if not code_list:
            return np.array([], dtype=np.float32)

        self._load_models()

        seed_logits = []
        for model in self.models:
            l_diff = self.predict_sliding_windows_for_model(
                model=model,
                codes=code_list,
                batch_size=batch_size,
            )
            seed_logits.append(l_diff)

        avg_logits = np.mean(seed_logits, axis=0)
        probs = scipy.special.expit(avg_logits)
        return probs.astype(np.float32)


# Global singleton predictor instance
_PREDICTOR: Optional[ModelAFinalEnsemble] = None


def predict(code_list: List[str], batch_size: int = 32, device: Optional[str] = None) -> np.ndarray:
    """
    Standard single-function API for Model A vulnerability scoring.
    Args:
        code_list: List of C/C++ source code function strings.
        batch_size: Inference batch size for sliding window chunks.
        device: Optional device override ('cpu', 'mps', 'cuda').
    Returns:
        np.ndarray of shape (len(code_list),) with probabilities in [0.0, 1.0].
    """
    global _PREDICTOR
    if _PREDICTOR is None or (device is not None and str(_PREDICTOR.device) != device):
        _PREDICTOR = ModelAFinalEnsemble(device=device, lazy_load=False)
    return _PREDICTOR.predict(code_list, batch_size=batch_size)
'''
    with open(FINAL_DIR / "predictor.py", "w") as f:
        f.write(predictor_code)
    print("Wrote predictor.py")

    # 3. Write __init__.py
    init_code = '''"""Model A Final: 5-Seed Debiased UniXcoder Ensemble."""

from .predictor import ModelAFinalEnsemble, predict

__all__ = ["ModelAFinalEnsemble", "predict"]
'''
    with open(FINAL_DIR / "__init__.py", "w") as f:
        f.write(init_code)
    print("Wrote __init__.py")

    # 4. Load dev metrics from master evaluation
    with open(RESULTS_P5_DIR / "p5_master_evaluation.json", "r") as f:
        master_eval = json.load(f)

    ens_metrics = master_eval["ensemble_metrics"]["D2_ensemble"]

    manifest = {
        "model_name": "Model A Final (D2 UniXcoder 5-Seed Ensemble)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "architecture": "microsoft/unixcoder-base (12-layer Transformer, 768 hidden, 12 heads)",
        "recipe": "D2: UniXcoder fine-tuning with inverse-propensity token length debiasing weights",
        "ensemble_rule": "Arithmetic average of predicted logits across 5 seeds: sigmoid(mean([logit_k for k in [42, 7, 1, 2, 3]]))",
        "seeds": [42, 7, 1, 2, 3],
        "inference_strategy": "Sliding window (512 token window, 256 stride, max-pooling per function)",
        "operating_threshold_top10": ens_metrics["operating_threshold_top10"],
        "dev_heldout_metrics": {
            "dev_auroc": ens_metrics["dev_auroc"],
            "dev_auroc_ci_95": ens_metrics["dev_auroc_ci_95"],
            "dev_ls_auroc": ens_metrics["dev_ls_auroc"],
            "dev_ls_auroc_ci_95": ens_metrics["dev_ls_auroc_ci_95"],
            "dev_lsauroc10": ens_metrics["dev_lsauroc10"],
            "dev_lsauroc10_ci_95": ens_metrics["dev_lsauroc10_ci_95"],
            "length_matched_auroc": ens_metrics["length_matched_auroc"],
            "length_matched_ci_95": ens_metrics["length_matched_ci_95"],
            "dev_cpp_auroc": ens_metrics["dev_cpp_auroc"],
            "pair_correct": ens_metrics["pair_correct"],
            "pair_correct_ci_95": ens_metrics["pair_correct_ci_95"],
            "alert_budget_top10_overall": ens_metrics["alert_overall"],
            "alert_budget_top10_per_project": ens_metrics["alert_per_project"],
            "paired_alert_dev_pairs": ens_metrics["paired_alert_dev_pairs"],
        },
        "comparisons_vs_baselines": {
            "delta_vs_l1_length_matched_auroc": master_eval["paired_tests"]["D2_ensemble"]["vs_L1"]["length_matched_auroc"]["diff_observed"],
            "p_holm_vs_l1_length_matched_auroc": master_eval["paired_tests"]["D2_ensemble"]["vs_L1"]["length_matched_auroc"]["p_value_holm_formatted"],
            "delta_vs_champion_ls_auroc": master_eval["paired_tests"]["D2_ensemble"]["vs_champion"]["ls_auroc"]["diff_observed"],
            "p_raw_vs_champion_ls_auroc": master_eval["paired_tests"]["D2_ensemble"]["vs_champion"]["ls_auroc"]["p_value_formatted"],
        },
        "weight_files": checksums,
    }

    with open(FINAL_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Wrote manifest.json")

    # 5. Write unit test tests/test_model_a_final.py
    test_code = '''"""Unit tests for Model A Final packaged ensemble."""

import unittest
import numpy as np
from models.checkpoints.p5.model_a_final.predictor import predict, ModelAFinalEnsemble


class TestModelAFinal(unittest.TestCase):
    def setUp(self):
        self.sample_codes = [
            "int safe_add(int a, int b) { return a + b; }",
            "void vuln_buffer(char *input) { char buf[16]; strcpy(buf, input); }",
            "int factorial(int n) { if (n <= 1) return 1; return n * factorial(n - 1); }",
            "void null_deref(int *ptr) { *ptr = 42; }",
            "bool is_even(int x) { return (x % 2 == 0); }",
        ]

    def test_predict_single_and_batch(self):
        """Verify predict() accepts 5 sample functions and returns valid probabilities."""
        probs = predict(self.sample_codes, batch_size=8)
        self.assertEqual(len(probs), len(self.sample_codes))
        self.assertIsInstance(probs, np.ndarray)
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))

        # Check vulnerability ranking: buffer overflow should score higher than trivial safe_add
        print(f"Sample predictions: {probs}")
        self.assertGreater(probs[1], probs[0], "Buffer overflow should score higher than safe_add")


if __name__ == "__main__":
    unittest.main()
'''
    test_path = PROJECT_ROOT / "tests" / "test_model_a_final.py"
    with open(test_path, "w") as f:
        f.write(test_code)
    print(f"Wrote unit test to {test_path}")

    print("\nPACKAGING COMPLETE!")


if __name__ == "__main__":
    main()
