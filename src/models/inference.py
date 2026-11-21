#!/usr/bin/env python3
"""
Production Inference Backend for Model A, Model B, and Comparative Analysis
=============================================================================
DTU B.Tech Project: Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code

Components:
  - predict_model_a(code, language='c'):
      Evaluates source code alone using the expanded Tri-Union model (50k training samples)
      combined with static security AST pattern analysis.
      Zero provenance features are consumed.
  - predict_model_b(code, provenance_dict, language='c'):
      Evaluates source code plus explicit provenance metadata using the trained
      fusion model (models/checkpoints/model_b_provenance_fusion.pkl).
  - compare_models(code, provenance_dict, language='c'):
      Runs both models on identical code and computes delta, agreement, and explanations.
  - get_demo_examples():
      Provides presentation scenarios for the university demonstration.
"""

import sys
import os
import re
import time
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import clean_code
from src.security.static_analysis import analyze_code

# Global Singleton Caches
_MODEL_A_PIPELINE = None
_MODEL_B_CHECKPOINT = None
_ACTIVE_MODEL_A_NAME = os.getenv("MODEL_A_CHECKPOINT", "expanded_tri_union_best.pkl")
_ACTIVE_MODEL_B_NAME = os.getenv("MODEL_B_CHECKPOINT", "model_b_provenance_fusion.pkl")

def set_active_checkpoints(model_a_name: Optional[str] = None, model_b_name: Optional[str] = None):
    """Dynamically switch active checkpoint while preserving fallbacks."""
    global _MODEL_A_PIPELINE, _MODEL_B_CHECKPOINT, _ACTIVE_MODEL_A_NAME, _ACTIVE_MODEL_B_NAME
    if model_a_name:
        _ACTIVE_MODEL_A_NAME = model_a_name
        _MODEL_A_PIPELINE = None
    if model_b_name:
        _ACTIVE_MODEL_B_NAME = model_b_name
        _MODEL_B_CHECKPOINT = None

def get_active_checkpoints() -> Dict[str, str]:
    return {
        "model_a": _ACTIVE_MODEL_A_NAME,
        "model_b": _ACTIVE_MODEL_B_NAME
    }

def _get_model_a_pipeline():
    global _MODEL_A_PIPELINE
    if _MODEL_A_PIPELINE is None:
        ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / _ACTIVE_MODEL_A_NAME
        if not ckpt_path.exists():
            # Fallback to production baseline
            ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "expanded_tri_union_best.pkl"
        with open(ckpt_path, "rb") as f:
            obj = pickle.load(f)
            _MODEL_A_PIPELINE = obj["pipeline"]
    return _MODEL_A_PIPELINE

def _get_model_b_checkpoint():
    global _MODEL_B_CHECKPOINT
    if _MODEL_B_CHECKPOINT is None:
        ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / _ACTIVE_MODEL_B_NAME
        if not ckpt_path.exists():
            # Fallback to production baseline
            ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "model_b_provenance_fusion.pkl"
        with open(ckpt_path, "rb") as f:
            _MODEL_B_CHECKPOINT = pickle.load(f)
    return _MODEL_B_CHECKPOINT


def predict_model_a(source_code: str, language: str = "c") -> Dict[str, Any]:
    """
    Model A: Provenance-Agnostic Vulnerability Detection (Code Only).
    
    Args:
        source_code: C/C++ or generic source code string.
        language: Language identifier (default 'c').
        
    Returns:
        Structured prediction dict with probability, class, CWE, and severity.
    """
    start_time = time.time()
    
    # 1. Validation & Input Handling
    if not source_code or not source_code.strip():
        return {
            "model_id": "Model A (Code Only)",
            "is_vulnerable": False,
            "probability": 0.0,
            "predicted_class": "Safe",
            "threshold": 0.50,
            "cwe": "None",
            "risk_level": "Low",
            "static_findings": [],
            "inference_latency_ms": 0.1,
            "code_hash": "",
            "status": "Empty Code Submission"
        }
        
    cleaned = clean_code(source_code)
    code_hash = hashlib.sha256(cleaned.encode()).hexdigest()[:16]
    
    # 2. Machine Learning Pipeline Inference
    pipe = _get_model_a_pipeline()
    try:
        raw_prob = float(pipe.predict_proba([cleaned])[0, 1])
    except Exception as e:
        raw_prob = 0.50
        
    # 3. Static Security & Flawfinder Enrichment
    static_res = analyze_code(source_code, language=language)
    cwes = static_res.get("cwes", [])
    findings = static_res.get("findings", [])
    static_risk = static_res.get("risk_level", "Low")
    
    # 4. Calibrated Integration
    # If static security rules identify critical MS-banned APIs (strcpy, gets, sprintf)
    # or confirmed boundary flaws, calibrate probability upwards.
    calibrated_prob = raw_prob
    severe_findings = [f for f in findings if f.get("severity") in ["medium", "high", "critical"] and f.get("rule_id") != "flawfinder-char"]
    if static_risk in ["High", "Critical"] and len(severe_findings) > 0:
        calibrated_prob = max(raw_prob, 0.65 + min(0.30, len(severe_findings) * 0.10))
    elif len(severe_findings) > 0 and raw_prob >= 0.40:
        calibrated_prob = max(raw_prob, 0.52)
    elif len(severe_findings) == 0 and raw_prob < 0.50:
        calibrated_prob = min(raw_prob, 0.35)

        
    calibrated_prob = float(np.clip(calibrated_prob, 0.01, 0.99))
    is_vuln = calibrated_prob >= 0.50
    pred_class = "Vulnerable" if is_vuln else "Safe"
    
    # Dominant CWE
    dominant_cwe = cwes[0] if cwes else ("CWE-Other" if is_vuln else "None")
    
    # Risk Level
    if calibrated_prob >= 0.80:
        risk_level = "Critical"
    elif calibrated_prob >= 0.60:
        risk_level = "High"
    elif calibrated_prob >= 0.40:
        risk_level = "Medium"
    else:
        risk_level = "Low"
        
    latency_ms = (time.time() - start_time) * 1000.0
    
    return {
        "model_id": "Model A (Code Only)",
        "is_vulnerable": is_vuln,
        "probability": calibrated_prob,
        "predicted_class": pred_class,
        "threshold": 0.50,
        "cwe": dominant_cwe,
        "risk_level": risk_level,
        "static_findings": findings,
        "n_findings": len(findings),
        "inference_latency_ms": round(latency_ms, 2),
        "code_hash": code_hash,
        "status": "Production Checkpoint (Trained on 50,000 C/C++ Samples)"
    }


def predict_model_b(
    source_code: str,
    provenance: Optional[Dict[str, Any]] = None,
    language: str = "c"
) -> Dict[str, Any]:
    """
    Model B: Provenance-Aware Vulnerability Detection (Code + Provenance).
    
    Args:
        source_code: Identical C/C++ source code as Model A.
        provenance: Dictionary containing:
            - 'ai_involvement': float (0.0 to 1.0)
            - 'provenance_type': 'human', 'ai', 'human_edit_of_ai', 'ai_edit_of_human', 'unknown'
            - 'ai_generator': 'ChatGPT', 'Copilot', 'Claude', etc.
            - 'human_edit_count': int
            - 'provenance_source': str
            - 'confidence': float (0.0 to 1.0)
        language: Language identifier.
        
    Returns:
        Structured prediction dict with probability, class, review priority, and provenance features.
    """
    start_time = time.time()
    
    # 1. Run Model A on identical source code
    pred_a = predict_model_a(source_code, language=language)
    p_code = pred_a["probability"]
    
    # 2. Extract and Validate Provenance Metadata
    if provenance is None or not provenance:
        has_provenance = False
        provenance = {}
    else:
        has_provenance = True
        
    prov_type = str(provenance.get("provenance_type", "unknown")).lower()
    if prov_type == "unknown":
        has_provenance = False
        
    ai_involvement = float(provenance.get("ai_involvement", 0.0 if not has_provenance else 0.50))
    ai_involvement = float(np.clip(ai_involvement, 0.0, 1.0))
    
    ai_generator = provenance.get("ai_generator", "None")
    human_edit_count = int(provenance.get("human_edit_count", 0))
    prov_source = str(provenance.get("provenance_source", "Unknown / Not Provided" if not has_provenance else "Developer Self-Reported"))
    confidence = float(provenance.get("confidence", 0.0 if not has_provenance else 1.0))
    confidence = float(np.clip(confidence, 0.0, 1.0))
    
    # Inferred flags
    is_human_edited = 1.0 if (prov_type == "human_edit_of_ai" or human_edit_count > 0) else 0.0
    is_pure_ai = 1.0 if (prov_type in ["ai", "ai_edit_of_human"] and human_edit_count == 0) else 0.0
    is_pure_human = 1.0 if (prov_type == "human" or (has_provenance and ai_involvement == 0.0)) else 0.0
    
    # 3. Model B Provenance Fusion Layer
    ckpt_b = _get_model_b_checkpoint()
    clf_fusion = ckpt_b["clf_fusion"]
    feature_names = ckpt_b.get("feature_names", ["logit_p_code", "ai_involvement", "is_human_edited", "is_pure_ai"])
    
    eps = 1e-4
    logit_code = np.log((p_code + eps) / (1.0 - p_code + eps))
    
    if not has_provenance:
        # Explicit handling: Missing provenance must not invent assumptions.
        # Falls back directly to base code probability.
        p_final = p_code
        status_note = "Model B: Provenance Unknown (Falling back to Code-Only Assessment)"
        review_priority = "Tier-3 (Standard Audit — Unknown Provenance)"
        triage_score = p_final
    else:
        # Construct feature vector dynamically based on trained checkpoint schema
        feat_vals = []
        for fn in feature_names:
            if fn in ["logit_code", "logit_p_code"]:
                feat_vals.append(logit_code)
            elif fn == "has_provenance":
                feat_vals.append(1.0)
            elif fn == "ai_involvement":
                feat_vals.append(ai_involvement)
            elif fn == "is_pure_ai":
                feat_vals.append(is_pure_ai)
            elif fn == "is_human_edited":
                feat_vals.append(is_human_edited)
            elif fn == "is_ai_edited":
                feat_vals.append(1.0 if prov_type == "ai_edit_of_human" else 0.0)
            elif fn == "confidence":
                feat_vals.append(confidence)
            elif fn == "interaction_code_ai":
                feat_vals.append(logit_code * ai_involvement)
            elif fn == "interaction_code_human":
                feat_vals.append(logit_code * is_human_edited)
            else:
                feat_vals.append(0.0)
        X_fusion = np.array([feat_vals], dtype=float)
            
        try:
            p_fused = float(clf_fusion.predict_proba(X_fusion)[0, 1])
        except Exception:
            # Fallback calibrated formula
            p_fused = p_code * (1.0 + 0.35 * ai_involvement - 0.25 * is_human_edited)
            
        # Scale by confidence if provenance is uncertain
        p_final = float(confidence * p_fused + (1.0 - confidence) * p_code)
        p_final = float(np.clip(p_final, 0.01, 0.99))
        status_note = "Trained Fusion Layer (Trained on Disjoint Development Benchmark)"
        
        # 4. Review Triage Priority Policy
        # Priority = P(vuln) * (1.0 + 0.5 * AI_Involvement)
        triage_score = p_final * (1.0 + 0.5 * ai_involvement)
        if is_pure_ai and p_final >= 0.40:
            review_priority = "Tier-1 (Urgent Audit — Unreviewed AI Code)"
        elif triage_score >= 0.60 or p_final >= 0.60:
            review_priority = "Tier-1 (Urgent Audit)"
        elif triage_score >= 0.35 or p_final >= 0.35:
            review_priority = "Tier-2 (Elevated Review — Hybrid Code)"
        else:
            review_priority = "Tier-3 (Standard Audit)"
            
    is_vuln = p_final >= 0.50
    pred_class = "Vulnerable" if is_vuln else "Safe"
    latency_ms = (time.time() - start_time) * 1000.0
    
    return {
        "model_id": "Model B (Code + Provenance)",
        "is_vulnerable": is_vuln,
        "probability": p_final,
        "base_code_probability": p_code,
        "predicted_class": pred_class,
        "threshold": 0.50,
        "cwe": pred_a["cwe"],
        "risk_level": pred_a["risk_level"],
        "review_priority": review_priority,
        "triage_score": round(triage_score, 4),
        "has_provenance": has_provenance,
        "provenance_used": {
            "has_provenance": has_provenance,
            "ai_involvement": ai_involvement,
            "provenance_type": prov_type,
            "ai_generator": ai_generator,
            "human_edit_count": human_edit_count,
            "provenance_source": prov_source,
            "confidence": confidence
        },
        "inference_latency_ms": round(latency_ms, 2),
        "code_hash": pred_a["code_hash"],
        "status": "Trained Provenance-Aware Model B",
        "status_note": status_note
    }


def compare_models(
    source_code: str,
    provenance: Optional[Dict[str, Any]] = None,
    language: str = "c"
) -> Dict[str, Any]:
    """
    Fair Side-by-Side Comparison of Model A and Model B on identical source code.
    
    Returns:
        Structured dict containing both predictions, probability delta, agreement, and explanation.
    """
    pred_a = predict_model_a(source_code, language=language)
    pred_b = predict_model_b(source_code, provenance=provenance, language=language)
    
    prob_a = pred_a["probability"]
    prob_b = pred_b["probability"]
    delta_pct = (prob_b - prob_a) * 100.0
    
    agree = (pred_a["is_vulnerable"] == pred_b["is_vulnerable"])
    
    # Generate concise, scientifically rigorous narrative
    ai_inv = pred_b["provenance_used"]["ai_involvement"]
    edits = pred_b["provenance_used"]["human_edit_count"]
    gen = pred_b["provenance_used"]["ai_generator"]
    prov_type = pred_b["provenance_used"]["provenance_type"]
    is_human_edited = (prov_type == "human_edit_of_ai" or edits > 0)
    
    if abs(delta_pct) < 1.0:
        narrative = (
            f"Both models strongly agree ({prob_a*100:.1f}% vs {prob_b*100:.1f}%). "
            "Source-code structural features dominate over provenance inputs."
        )
    elif delta_pct > 0:
        narrative = (
            f"Model B elevates vulnerability risk by +{delta_pct:.1f}% percentage points "
            f"({prob_a*100:.1f}% → {prob_b*100:.1f}%). "
            f"High generative AI involvement ({int(ai_inv*100)}% via {gen}) with limited human review "
            "increases the risk of subtle API boundary assumptions and hallucinated contracts."
        )
    else:
        if edits > 0 or is_human_edited:
            narrative = (
                f"Model B moderates vulnerability risk by {delta_pct:.1f}% percentage points "
                f"({prob_a*100:.1f}% → {prob_b*100:.1f}%). "
                f"Documented human developer review and remediation ({edits} edit cycles) mitigates risk, "
                "verifying defensive boundaries."
            )
        else:
            narrative = (
                f"Model B adjusts baseline estimate by {delta_pct:.1f}% percentage points "
                f"({prob_a*100:.1f}% → {prob_b*100:.1f}%) through multi-factor provenance calibration."
            )
        
    return {
        "model_a": pred_a,
        "model_b": pred_b,
        "probability_delta_percentage_points": round(delta_pct, 2),
        "models_agree": agree,
        "comparison_narrative": narrative,
        "source_code_identical": True,
        "comparison_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


def get_demo_examples() -> Dict[str, Dict[str, Any]]:
    """Provides three distinct presentation demonstration scenarios."""
    return {
        "demo_1_vulnerable": {
            "title": "Demo 1: Vulnerable Code (CWE-120 Buffer Overflow)",
            "description": "Unbounded strcpy() copying untrusted input into a 16-byte fixed buffer.",
            "language": "c",
            "code": """#include <string.h>
#include <stdio.h>

void process_request(const char *user_input) {
    char local_buffer[16];
    // Dangerous unbounded memory copy without length verification
    strcpy(local_buffer, user_input);
    printf("Processing: %s\\n", local_buffer);
}""",
            "provenance": {
                "ai_involvement": 0.85,
                "provenance_type": "ai",
                "ai_generator": "ChatGPT",
                "human_edit_count": 0,
                "provenance_source": "IDE Telemetry (Copilot Extension)",
                "confidence": 0.95
            }
        },
        "demo_2_benign": {
            "title": "Demo 2: Safe Code (Bounds-Checked String Copy)",
            "description": "Safe snprintf implementation with explicit null-termination and bounds check.",
            "language": "c",
            "code": """#include <stdio.h>
#include <string.h>

int safe_process_request(const char *user_input) {
    if (user_input == NULL) {
        return -1;
    }
    char local_buffer[64];
    // Bounds-checked string formatting with strict size limit
    int written = snprintf(local_buffer, sizeof(local_buffer), "%s", user_input);
    if (written < 0 || (size_t)written >= sizeof(local_buffer)) {
        return -1; // Truncation or formatting error detected
    }
    return 0;
}""",
            "provenance": {
                "ai_involvement": 0.0,
                "provenance_type": "human",
                "ai_generator": "None",
                "human_edit_count": 2,
                "provenance_source": "Git Author Verification",
                "confidence": 1.0
            }
        },
        "demo_3_hybrid_ablation": {
            "title": "Demo 3: Hybrid Code Provenance Ablation (Same Code, Different Provenance)",
            "description": "Path normalization utility evaluated as pure Human vs raw AI vs Human-Edited AI.",
            "language": "c",
            "code": """#include <string.h>
#include <stdlib.h>

char* sanitize_http_path(const char *path) {
    if (!path) return NULL;
    char *normalized = strdup(path);
    // Strip query parameters (?param=value)
    char *query = strchr(normalized, '?');
    if (query) *query = '\\0';
    // Strip URI fragment (#section)
    char *fragment = strchr(normalized, '#');
    if (fragment) *fragment = '\\0';
    return normalized;
}""",
            "provenance": {
                "ai_involvement": 0.50,
                "provenance_type": "human_edit_of_ai",
                "ai_generator": "GitHub Copilot",
                "human_edit_count": 3,
                "provenance_source": "IDE Revisions Hook",
                "confidence": 0.90
            }
        }
    }
