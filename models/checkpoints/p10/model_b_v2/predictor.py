"""Model B v2 Predictor: Offset fusion of Model A code risk score with provenance metadata."""

import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
MODEL_DIR = Path(__file__).resolve().parent
MODEL_PKL_PATH = MODEL_DIR / "model_b_v2.pkl"


class OffsetFusionPredictor:
    """Offset fusion model predictor."""
    def __init__(self, w: np.ndarray, b: float):
        self.w = w
        self.b = b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logit_p = X[:, 0]
        prov_s = X[:, 1:]
        z = logit_p + prov_s @ self.w + self.b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return np.column_stack([1.0 - p, p])


class _CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "OffsetFusionPredictor":
            return OffsetFusionPredictor
        return super().find_class(module, name)


class ModelBv2Predictor:
    def __init__(self, checkpoint_path: Optional[Path] = None):
        ckpt = checkpoint_path or MODEL_PKL_PATH
        if not ckpt.exists():
            raise FileNotFoundError(f"Model B v2 checkpoint not found at {ckpt}")

        with open(ckpt, "rb") as f:
            data = _CustomUnpickler(f).load()

        self.fusion_model = data.get("fusion_model")
        self.scaler_provenance = data.get("scaler_provenance")
        self.model_a_artifact = data.get("model_a_artifact", {})
        self.model_a_name = self.model_a_artifact.get("model_name")
        self.model_a = self.model_a_artifact.get("model")
        self.scaler_stat = self.model_a_artifact.get("scaler_static")
        self.scaler_len = self.model_a_artifact.get("scaler_len")

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._qwen_tok = None
        self._qwen_mod = None
        self._cb_tok = None
        self._cb_mod = None

    def _get_qwen(self):
        if self._qwen_tok is None:
            self._qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
            self._qwen_mod = AutoModel.from_pretrained(
                "Qwen/Qwen2.5-Coder-0.5B", torch_dtype=torch.float16
            ).to(self.device)
            self._qwen_mod.eval()
        return self._qwen_tok, self._qwen_mod

    def _get_cb(self):
        if self._cb_tok is None:
            self._cb_tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            self._cb_mod = AutoModel.from_pretrained("microsoft/codebert-base").to(self.device)
            self._cb_mod.eval()
        return self._cb_tok, self._cb_mod

    def compute_code_features(self, code: str) -> Dict[str, np.ndarray]:
        """Compute Qwen, CodeBERT, static, and length features for a single code snippet."""
        from scripts.p10.extract_model_a_v2_features import extract_length_features_from_codes
        from scripts.p10.extract_juliet_features import load_cwe_families, extract_static_features
        from scripts.p2.cache_embeddings_p2 import extract_standard_codebert, extract_chunked_codebert
        import pandas as pd

        # Length
        len_feat = extract_length_features_from_codes([code])
        len_feat_s = self.scaler_len.transform(len_feat)

        # Static
        families, fam_to_idx, cwe_to_fam_idx = load_cwe_families()
        df_tmp = pd.DataFrame([{"code": code, "cwe": "Unknown"}])
        stat_feat = extract_static_features(df_tmp, families, fam_to_idx, cwe_to_fam_idx)
        stat_feat_s = self.scaler_stat.transform(stat_feat)

        # Qwen
        q_tok, q_mod = self._get_qwen()
        inp = q_tok([code], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = q_mod(**inp)
            mask = inp["attention_mask"].unsqueeze(-1)
            embs = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            qwen_emb = embs.cpu().to(torch.float32).numpy()

        # CodeBERT
        cb_tok, cb_mod = self._get_cb()
        cb_std = extract_standard_codebert([code], cb_tok, cb_mod, self.device, batch_size=1, max_length=256)
        cb_emb, _ = extract_chunked_codebert(
            [code], cb_std, cb_tok, cb_mod, self.device, batch_size=1, window_size=254, stride=128, max_windows_per_sample=16, sub_chunk_size=1
        )

        return {
            "qwen": qwen_emb,
            "cb": cb_emb,
            "stat_s": stat_feat_s,
            "len_s": len_feat_s,
        }

    def predict_code_risk(self, code_features: Dict[str, np.ndarray]) -> float:
        """Compute P_code using frozen Model A v2."""
        from scripts.p10.train_model_a_v2 import predict_candidate
        p_c = predict_candidate(
            self.model_a_name,
            self.model_a,
            code_features["qwen"],
            code_features["cb"],
            code_features["stat_s"],
            code_features["len_s"],
        )
        return float(p_c[0])

    def predict(self, code: str, provenance: Optional[Dict[str, Any]] = None) -> float:
        """
        Predict vulnerability risk for a code snippet.
        
        Args:
            code: C/C++ source code string.
            provenance: Optional dict with keys:
                - author_type: 'human' or 'ai'
                - ai_involvement: float in [0, 1]
                - temperature: float (e.g. 0.2, 0.8)
                - prompt_tokens: int
                - candidate_tokens: int
                
        Returns:
            P_risk (float between 0 and 1).
            When provenance is None, returns P_code directly (satisfying |P_risk - P_code| < 1e-6).
        """
        feats = self.compute_code_features(code)
        p_code = self.predict_code_risk(feats)

        if provenance is None or len(provenance) == 0:
            return float(p_code)

        # Extract provenance vector: [is_ai, ai_involvement, temp, ...]
        is_ai = 1.0 if (provenance.get("author_type") == "ai" or provenance.get("is_ai") == 1.0 or "ai" in str(provenance.get("role", ""))) else 0.0
        ai_inv = float(provenance.get("ai_involvement", 1.0 if is_ai else 0.0))
        temp_val = float(provenance.get("temperature", 0.0))
        p_len = float(np.log(max(1, provenance.get("prompt_tokens", 10))))
        c_len = float(np.log(max(1, provenance.get("candidate_tokens", 100))))

        n_dim = self.scaler_provenance.mean_.shape[0]
        if n_dim == 3:
            prov_vec = np.array([[is_ai, ai_inv, temp_val]], dtype=np.float32)
        elif n_dim == 5:
            prov_vec = np.array([[is_ai, ai_inv, temp_val, p_len, c_len]], dtype=np.float32)
        else:
            prov_vec = np.zeros((1, n_dim), dtype=np.float32)
            prov_vec[0, :min(3, n_dim)] = [is_ai, ai_inv, temp_val][:min(3, n_dim)]
        prov_vec_s = self.scaler_provenance.transform(prov_vec)

        # Offset fusion
        logit_p = np.log(np.clip(p_code, 1e-6, 1.0 - 1e-6) / (1.0 - np.clip(p_code, 1e-6, 1.0 - 1e-6)))
        X_fusion = np.column_stack([[logit_p], prov_vec_s])
        p_risk = float(self.fusion_model.predict_proba(X_fusion)[0, 1])
        return p_risk


_GLOBAL_PREDICTOR = None


def predict(code: str, provenance: Optional[Dict[str, Any]] = None) -> float:
    """Convenience top-level prediction function."""
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None:
        _GLOBAL_PREDICTOR = ModelBv2Predictor()
    return _GLOBAL_PREDICTOR.predict(code, provenance=provenance)
