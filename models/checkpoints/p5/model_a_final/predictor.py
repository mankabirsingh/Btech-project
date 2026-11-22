"""Model A Final (D2 UniXcoder 5-Seed Ensemble) Inference Module."""

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
