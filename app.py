#!/usr/bin/env python3
"""
Provenance-Aware Security Risk Assessment — Presentation & Comparison Application
=================================================================================
Interactive Research Prototype for DTU B.Tech Project-I
Authors: Mankabir Singh (23/IT/199) & Mohnish Gautam (23/IT/200)

Features:
1. 🔬 Model A vs Model B Live Comparison (The Primary Showcase)
   - Side-by-side comparative inference on identical source code
   - Model A: Provenance-Agnostic Code-Only Classifier (Trained on 50,000 samples)
   - Model B: Provenance-Aware Fusion Predictor (Code + Provenance Metadata)
   - Preset Scenarios: Demo 1 (Vulnerable Buffer Overflow), Demo 2 (Safe Bounds Check), Demo 3 (Provenance Sensitivity)
   - Real-time Provenance Sensitivity Ablation Slider
   - Review Triage Priority Policy & Flawfinder Static Analysis
2. 🧪 Interactive Provenance Ablation Lab
   - Vary AI involvement and review status across identical code to observe delta response
3. Model Evaluation Dashboard
   - Verified evaluation metrics (Tri-Union, CodeBERT, Cross-Project, Controlled Benchmark)
4. Dataset & Provenance Audit Explorer
   - Ground-truth analysis of Devign, PrimeVul, BigVul, DiverseVul, HumanVsAICode, and the 32-sample benchmark
"""

import sys
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Backend Inference Module
from src.models.inference import (
    predict_model_a,
    predict_model_b,
    compare_models,
    get_demo_examples,
)
from src.security.static_analysis import analyze_code

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================
# Streamlit Page Config
# ========================
st.set_page_config(
    page_title="Provenance-Aware Security Risk Assessment",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# ========================
# Custom CSS Design System
# ========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Main Hero Header */
    .main-header {
        background: linear-gradient(135deg, #0b0c10 0%, #1f2833 50%, #111a24 100%);
        padding: 2rem 2.2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    
    .main-header h1 {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #66fcf1, #45a29e, #9b5de5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #c5c6c7;
        font-size: 1.0rem;
        margin-top: 0.6rem;
        line-height: 1.5;
    }
    
    .badge-sub {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(102, 252, 241, 0.15);
        color: #66fcf1;
        border: 1px solid rgba(102, 252, 241, 0.3);
        margin-bottom: 0.6rem;
    }

    /* Model Cards */
    .model-card-a {
        background: linear-gradient(145deg, #101924, #152232);
        border: 1px solid rgba(0, 210, 255, 0.3);
        border-radius: 14px;
        padding: 1.4rem;
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.08);
        margin-bottom: 1rem;
    }
    
    .model-card-b {
        background: linear-gradient(145deg, #1b162b, #251e3e);
        border: 1px solid rgba(155, 93, 229, 0.4);
        border-radius: 14px;
        padding: 1.4rem;
        box-shadow: 0 6px 20px rgba(155, 93, 229, 0.12);
        margin-bottom: 1rem;
    }

    .card-title-a {
        color: #00d2ff;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .card-title-b {
        color: #c77dff;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .card-subtitle {
        color: #a0aec0;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }

    /* Comparison Delta Banner */
    .delta-banner {
        background: linear-gradient(135deg, #141b2d, #1f2c4c);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1.2rem 0;
    }

    /* Status Badges */
    .badge-vuln {
        background: rgba(255, 71, 87, 0.2);
        color: #ff4757;
        border: 1px solid #ff4757;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }

    .badge-safe {
        background: rgba(46, 213, 115, 0.2);
        color: #2ed573;
        border: 1px solid #2ed573;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .badge-tier1 {
        background: rgba(255, 71, 87, 0.25);
        color: #ff4757;
        border: 1px solid #ff4757;
        padding: 0.3rem 0.7rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    
    .badge-tier2 {
        background: rgba(255, 165, 2, 0.25);
        color: #ffa502;
        border: 1px solid #ffa502;
        padding: 0.3rem 0.7rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    
    .badge-tier3 {
        background: rgba(46, 213, 115, 0.25);
        color: #2ed573;
        border: 1px solid #2ed573;
        padding: 0.3rem 0.7rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    
    .status-tag {
        font-size: 0.75rem;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        background: rgba(255,255,255,0.06);
        color: #a0aec0;
        display: inline-block;
        margin-top: 0.4rem;
    }

    /* Architectural Diagram Box */
    .arch-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px dashed rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090a0f 0%, #141722 100%);
    }
</style>
""", unsafe_allow_html=True)


# ========================
# Cached Baseline Loader
# ========================
@st.cache_data
def get_demos():
    return get_demo_examples()


# ========================
# Sidebar Navigation
# ========================
# ========================
# Sample Snippet Definitions
# ========================
VULNERABLE_C_EXAMPLE = """#include <stdio.h>
#include <string.h>

void process_request(const char *user_input) {
    char local_buffer[16];
    // Dangerous unbounded memory copy into a fixed-size buffer (CWE-120/CWE-119)
    strcpy(local_buffer, user_input);
    printf("Processing: %s\\n", local_buffer);
}"""

SAFER_C_EXAMPLE = """#include <stdio.h>
#include <string.h>

int safe_process_request(const char *user_input) {
    if (user_input == NULL) {
        return -1;
    }
    char local_buffer[64];
    // Bounds-checked string formatting with strict size limit and error check
    int written = snprintf(local_buffer, sizeof(local_buffer), "%s", user_input);
    if (written < 0 || (size_t)written >= sizeof(local_buffer)) {
        return -1; // Truncation or formatting error detected
    }
    return 0;
}"""

CUSTOM_C_EXAMPLE = """#include <stdio.h>
#include <stdlib.h>

// Enter custom C or C++ function to evaluate with both models
void handle_input(const char *data, size_t len) {
    char *buf = (char *)malloc(len);
    if (!buf) return;
    // Inspect memory handling
    free(buf);
}"""

PROVENANCE_OPTIONS = {
    "Human-written": {
        "provenance_type": "human",
        "ai_involvement": 0.0,
        "human_edit_count": 2,
        "ai_generator": "None",
        "provenance_source": "Git Author Verification",
        "confidence": 1.0,
        "description": "Authored entirely by a human engineer with documented commit history."
    },
    "AI-generated, not yet human-reviewed": {
        "provenance_type": "ai",
        "ai_involvement": 0.90,
        "human_edit_count": 0,
        "ai_generator": "ChatGPT",
        "provenance_source": "IDE Telemetry (Copilot Extension)",
        "confidence": 0.95,
        "description": "Synthesized directly by generative LLM without subsequent manual developer review."
    },
    "AI-generated, then human-reviewed/edited": {
        "provenance_type": "human_edit_of_ai",
        "ai_involvement": 0.50,
        "human_edit_count": 2,
        "ai_generator": "GitHub Copilot",
        "provenance_source": "IDE Revisions Hook",
        "confidence": 0.90,
        "description": "Initial AI scaffold followed by developer inspection, defensive check addition, and refactoring."
    },
    "Human-written, then AI-modified": {
        "provenance_type": "ai_edit_of_human",
        "ai_involvement": 0.50,
        "human_edit_count": 0,
        "ai_generator": "Claude 3.5 Sonnet",
        "provenance_source": "PR Automation Hook",
        "confidence": 0.85,
        "description": "Existing human codebase modified by an AI coding assistant without subsequent human audit."
    },
    "Unknown / unavailable": {
        "provenance_type": "unknown",
        "ai_involvement": 0.0,
        "human_edit_count": 0,
        "ai_generator": "None",
        "provenance_source": "Unavailable",
        "confidence": 0.0,
        "description": "Provenance metadata missing or unavailable. Triggers verified code-only fallback in Model B."
    }
}

# ========================
# Sidebar Navigation
# ========================
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "Select Page",
        [
            "Code Security Comparison",
            "Provenance Ablation Lab",
            "Model Evaluation & Benchmarks",
            "Dataset & Provenance Audit",
        ],
        index=0,
        label_visibility="collapsed",
    )
    
    st.markdown("---")
    st.markdown("### Academic Context")
    st.markdown("""
    **B.Tech Project-I**  
    *Delhi Technological University*  
    Department of Information Technology
    
    **Investigators:**  
    • **Mankabir Singh** (23/IT/199)  
    • **Mohnish Gautam** (23/IT/200)
    
    **Research Question:**  
    *Can provenance-aware analysis improve vulnerability detection & review triage for human-AI hybrid code?*
    """)


# =========================================================================================
# PAGE 1: Code Security Comparison (Minimal Presentation Landing Page)
# =========================================================================================
if page == "Code Security Comparison":
    # Section A: Header
    st.markdown('''
    <div class="main-header">
        <h1>Provenance-Aware Code Security Assessment</h1>
        <p>Compare code-only vulnerability detection with provenance-aware risk assessment.</p>
    </div>
    ''', unsafe_allow_html=True)

    # Initialize snippet state
    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = "🔴 Vulnerable C Example"
        st.session_state.editor_code = VULNERABLE_C_EXAMPLE

    # Section B: Code input panel
    st.markdown("### 1. Source Code Input")
    
    col_sample, col_lang = st.columns([3, 1])
    with col_sample:
        sample_choice = st.radio(
            "Select Code Example or Custom Mode:",
            ["🔴 Vulnerable C Example", "🟢 Safer C Example", "Custom Code"],
            horizontal=True,
            index=["🔴 Vulnerable C Example", "🟢 Safer C Example", "Custom Code"].index(st.session_state.selected_sample if st.session_state.selected_sample in ["🔴 Vulnerable C Example", "🟢 Safer C Example", "Custom Code"] else "🔴 Vulnerable C Example"),
            label_visibility="collapsed"
        )
    with col_lang:
        lang_choice = st.selectbox("Language", ["C", "C++"], index=0)

    # Update editor content when sample choice changes
    if sample_choice != st.session_state.selected_sample:
        st.session_state.selected_sample = sample_choice
        if sample_choice == "🔴 Vulnerable C Example":
            st.session_state.editor_code = VULNERABLE_C_EXAMPLE
        elif sample_choice == "🟢 Safer C Example":
            st.session_state.editor_code = SAFER_C_EXAMPLE
        else:
            st.session_state.editor_code = CUSTOM_C_EXAMPLE
        st.rerun()

    if sample_choice == "🔴 Vulnerable C Example":
        st.caption("⚠️ **Vulnerability Context:** Demonstrates unsafe, unbounded `strcpy()` copying untrusted input into a 16-byte fixed buffer without bounds checking (CWE-120 / CWE-119 buffer overflow).")
    elif sample_choice == "🟢 Safer C Example":
        st.caption("🛡️ **Defensive Context:** Uses bounded `snprintf()` with explicit buffer size limit (`sizeof(local_buffer)`), `NULL` check, and return-value truncation verification.")
    else:
        st.caption("**Custom Code Mode:** Enter any C/C++ function below to evaluate both models on your exact input.")

    code_input = st.text_area(
        "Source Code (submitted identically to both models):",
        value=st.session_state.editor_code,
        height=220,
        help="Both Model A and Model B receive this exact source code string without alteration."
    )

    # Section C: Model B Provenance Selector
    st.markdown("### 2. Development History — Model B Only")
    st.caption("Model A is strictly blind to this metadata and evaluates source code alone.")

    prov_choice = st.selectbox(
        "Select Provenance Scenario for Model B:",
        list(PROVENANCE_OPTIONS.keys()),
        index=1,
        help="Choose the development history scenario to provide to Model B."
    )

    selected_prov = PROVENANCE_OPTIONS[prov_choice]
    
    # Visible labeling of hypothetical scenario and fallback handling
    if prov_choice == "Unknown / unavailable":
        st.info("ℹ️ **Fallback Active:** Unknown provenance triggers Model B's verified code-only fallback. Model B matches Model A base code assessment.")
    else:
        st.markdown(f"🏷️ **Illustrative provenance scenario:** *{selected_prov['description']}* (Not independently verified from git telemetry).")

    # Action Button
    st.markdown("---")
    analyze_btn = st.button("🔍 Analyze Code", type="primary", use_container_width=True)

    # Section D: Side-by-Side Results
    if analyze_btn or ("last_cmp" in st.session_state and st.session_state.get("last_code") == code_input and st.session_state.get("last_prov") == prov_choice):
        if not code_input.strip():
            st.warning("⚠️ Please provide non-empty source code to execute analysis.")
        else:
            prov_payload = {
                "provenance_type": selected_prov["provenance_type"],
                "ai_involvement": selected_prov["ai_involvement"],
                "ai_generator": selected_prov["ai_generator"],
                "human_edit_count": selected_prov["human_edit_count"],
                "provenance_source": selected_prov["provenance_source"],
                "confidence": selected_prov["confidence"],
            }

            with st.spinner("Executing Model A (Code-Only) and Model B (Provenance-Aware)..."):
                try:
                    result = compare_models(
                        source_code=code_input,
                        provenance=prov_payload,
                        language=lang_choice.lower()
                    )
                    st.session_state["last_cmp"] = result
                    st.session_state["last_code"] = code_input
                    st.session_state["last_prov"] = prov_choice
                except Exception as e:
                    st.error(f"❌ Inference Error: {str(e)}")
                    result = None

            if result is not None:
                res_a = result["model_a"]
                res_b = result["model_b"]

                st.markdown("### 3. Side-by-Side Comparison Results")

                col_a, col_b = st.columns(2)

                # LEFT PANEL: MODEL A
                with col_a:
                    is_vuln_a = res_a["is_vulnerable"]
                    prob_a = res_a["probability"]
                    badge_a = "badge-vuln" if is_vuln_a else "badge-safe"
                    label_a = "🔴 VULNERABLE" if is_vuln_a else "🟢 NON-VULNERABLE"
                    dominant_cwe = res_a.get("cwe", "None")

                    st.markdown(f'''
                    <div class="model-card-a">
                        <div class="card-title-a">🔷 Model A: Code-Only Detector</div>
                        <div class="card-subtitle"><b>Uses source code only.</b> Strictly blind to development history.</div>
                        <div style="margin: 0.8rem 0;">
                            <span class="{badge_a}">{label_a}</span>
                            <span style="font-size: 1.7rem; font-weight: 800; color: #00d2ff; margin-left: 0.8rem;">
                                {prob_a * 100:.1f}%
                            </span>
                            <span style="color: #a0aec0; font-size: 0.85rem;">probability of vulnerability</span>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)

                    st.progress(prob_a)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Predicted Class", "Vulnerable" if is_vuln_a else "Safe")
                    m2.metric("Detected CWE", dominant_cwe if is_vuln_a else "None")
                    m3.metric("Latency", f"{res_a['inference_latency_ms']} ms")

                    if res_a.get("static_findings"):
                        st.markdown(f"**Static Security Findings ({len(res_a['static_findings'])}):**")
                        for f in res_a["static_findings"]:
                            st.caption(f"• `Line {f.get('line', '?')}`: {f.get('cwe', '')} — {f.get('message', '')}")
                    else:
                        st.caption("Static pattern analysis: No known dangerous C API calls flagged.")

                # RIGHT PANEL: MODEL B
                with col_b:
                    is_vuln_b = res_b["is_vulnerable"]
                    prob_b = res_b["probability"]
                    badge_b = "badge-vuln" if is_vuln_b else "badge-safe"
                    label_b = "🔴 VULNERABLE" if is_vuln_b else "🟢 NON-VULNERABLE"
                    tier_str = res_b.get("review_priority", "Tier-3 (Standard Audit)")
                    
                    if "Tier-1" in tier_str:
                        tier_badge = "badge-tier1"
                    elif "Tier-2" in tier_str:
                        tier_badge = "badge-tier2"
                    else:
                        tier_badge = "badge-tier3"

                    st.markdown(f'''
                    <div class="model-card-b">
                        <div class="card-title-b">🟣 Model B: Provenance-Aware Assessor</div>
                        <div class="card-subtitle"><b>Uses source code + provenance.</b> Fusion layer with development metadata.</div>
                        <div style="margin: 0.8rem 0;">
                            <span class="{badge_b}">{label_b}</span>
                            <span style="font-size: 1.7rem; font-weight: 800; color: #c77dff; margin-left: 0.8rem;">
                                {prob_b * 100:.1f}%
                            </span>
                            <span style="color: #a0aec0; font-size: 0.85rem;">provenance-adjusted risk</span>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)

                    st.progress(prob_b)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Predicted Class", "Vulnerable" if is_vuln_b else "Safe")
                    m2.metric("Base Code Prob", f"{res_b['base_code_probability'] * 100:.1f}%")
                    m3.metric("Latency", f"{res_b['inference_latency_ms']} ms")

                    st.markdown(f"**Review Priority:** <span class='{tier_badge}'>{tier_str}</span>", unsafe_allow_html=True)
                    st.caption(f"Provenance Scenario: *{prov_choice}*")
                    st.caption(f"Status: {res_b.get('status_note', '')}")

                # SCORE DELTA & EXPLANATION
                st.markdown("---")
                delta_val = result["probability_delta_percentage_points"]
                agree = result["models_agree"]

                c_d1, c_d2 = st.columns([1, 3])
                with c_d1:
                    st.metric(
                        label="Score Delta (Model B − Model A)",
                        value=f"{delta_val:+.1f}%",
                        delta=f"{delta_val:+.1f} pp",
                        delta_color="inverse" if delta_val > 0 else "normal",
                        help="Difference in calibrated vulnerability probability between Model B and Model A."
                    )
                    agree_text = "✅ Full Agreement" if agree else "⚠️ Risk Adjusted"
                    st.markdown(f"**Model Agreement:** `{agree_text}`")

                with c_d2:
                    st.markdown(f'''
                    <div class="delta-banner">
                        <div style="font-weight: 700; font-size: 1.0rem; color: #66fcf1; margin-bottom: 0.3rem;">
                            🔬 Prediction & Review Triage Analysis
                        </div>
                        <p style="margin: 0; color: #e2e8f0; font-size: 0.92rem; line-height: 1.45;">
                            {result['comparison_narrative']}
                        </p>
                    </div>
                    ''', unsafe_allow_html=True)

    # Section E: Interpretation Note
    st.markdown("---")
    st.info(
        "💡 **Interpretation Note:** Provenance is an additional risk-assessment signal, not proof of vulnerability "
        "or safety. Model predictions require security review and should not be treated as a substitute for code analysis."
    )


# =========================================================================================
# PAGE 2: Provenance Ablation Lab
# =========================================================================================
elif page == "Provenance Ablation Lab":
    st.markdown("""
    <div class="main-header">
        <span class="badge-sub">CONTROLLED EXPERIMENTATION</span>
        <h1>Interactive Provenance Sensitivity Ablation</h1>
        <p>Systematically vary human-versus-AI provenance parameters on fixed source code to evaluate 
        how Model B adjusts risk estimates, sensitivity coefficients, and review tiers.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 1. Select Code Snippet to Test")
    demos = {k: v for k, v in get_demos().items() if k != "demo_3_hybrid_ablation"}
    sample_key = st.selectbox(
        "Benchmark Code Sample:",
        list(demos.keys()),
        format_func=lambda k: demos[k]["title"]
    )
    test_code = demos[sample_key]["code"]
    st.code(test_code, language="c")
    
    st.markdown("### 2. Multi-Regime Provenance Comparison")
    st.markdown("Evaluating the exact same code across three standard operational regimes:")
    
    regimes = [
        {"name": "Regime 1: Pure Human Developer", "prov": {"ai_involvement": 0.0, "provenance_type": "human", "ai_generator": "None", "human_edit_count": 3, "confidence": 1.0}},
        {"name": "Regime 2: Human-Reviewed AI Code (Hybrid)", "prov": {"ai_involvement": 0.5, "provenance_type": "human_edit_of_ai", "ai_generator": "GitHub Copilot", "human_edit_count": 2, "confidence": 0.90}},
        {"name": "Regime 3: Raw Unreviewed AI Generation", "prov": {"ai_involvement": 1.0, "provenance_type": "ai", "ai_generator": "ChatGPT", "human_edit_count": 0, "confidence": 0.95}}
    ]
    
    regime_results = []
    pred_a_fixed = predict_model_a(test_code)
    
    for reg in regimes:
        pred_b_reg = predict_model_b(test_code, provenance=reg["prov"])
        regime_results.append({
            "Operational Regime": reg["name"],
            "Model A (Code Only)": f"{pred_a_fixed['probability']*100:.1f}%",
            "Model B (Fused)": f"{pred_b_reg['probability']*100:.1f}%",
            "Risk Shift": f"{(pred_b_reg['probability'] - pred_a_fixed['probability'])*100:+.1f}%",
            "Predicted Class": pred_b_reg["predicted_class"],
            "Triage Priority": pred_b_reg["review_priority"]
        })
        
    st.dataframe(pd.DataFrame(regime_results), use_container_width=True, hide_index=True)
    
    st.markdown("""
    > [!NOTE]
    > **Ablation Interpretation:**
    > When code is held constant, any shift in Model B's output is purely an effect of the provenance feature vector.
    > In Regime 3 (Raw AI), the lack of human review cycles triggers Tier-1 urgent audit flags, reflecting real-world engineering policy for automated code generation.
    """)


# =========================================================================================
# PAGE 3: Model Evaluation Dashboard
# =========================================================================================
elif page == "Model Evaluation & Benchmarks":
    st.markdown("""
    <div class="main-header">
        <span class="badge-sub">EMPIRICAL VALIDATION</span>
        <h1>Comprehensive Model Evaluation Dashboard</h1>
        <p>Verified, reproducible evaluation metrics across cross-project generalization benchmarks 
        and controlled human-AI provenance experiments.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs([
        "Model A Benchmarks (Cross-Project)",
        "Model B Controlled Benchmark (N=32)",
    ])
    
    with tab1:
        st.markdown("### Model A: Tri-Union SVM & CodeBERT Performance")
        st.markdown("**Evaluated on 50,000 real-world C/C++ samples and 3,325 unseen repository functions:**")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Validation Accuracy", "79.10%", "+24.63% vs Baseline")
        c2.metric("Cross-Project AUROC", "0.7454", "Held-out repos")
        c3.metric("F1 Score", "0.7890", "Macro F1")
        c4.metric("Inference Latency", "< 20 ms", "CPU real-time")
        
        st.markdown("#### Repository-Level Generalization Breakdown:")
        repo_data = [
            {"Repository": "GPAC (Multimedia Engine)", "Language": "C", "Samples": 128, "Accuracy": "85.62%", "AUROC": "0.8912", "Status": "Strong Generalization"},
            {"Repository": "Vim (Text Editor)", "Language": "C", "Samples": 780, "Accuracy": "76.55%", "AUROC": "0.7820", "Status": "Strong Generalization"},
            {"Repository": "OpenSSL (Cryptography)", "Language": "C", "Samples": 1420, "Accuracy": "70.80%", "AUROC": "0.7240", "Status": "Moderate Generalization"},
            {"Repository": "Envoy Proxy (Cloud Native)", "Language": "C++", "Samples": 603, "Accuracy": "54.89%", "AUROC": "0.5219", "Status": "Domain Gap (Modern C++)"},
        ]
        st.dataframe(pd.DataFrame(repo_data), use_container_width=True, hide_index=True)
        st.caption("Data source: results/experiments/cross_project_optimization_report.md and expanded_tri_union_metrics.json")
        
        with st.expander("📖 Model A Glossary & Metric Guide", expanded=False):
            st.markdown("""
            - **Tri-Union SVM:** A fast machine-learning classifier that scans 3 types of text patterns at once (character chunks, full words, and code syntax) to spot security flaws.
            - **CodeBERT:** A deep-learning neural network pretrained on millions of code functions that understands the logical structure and semantics of programming languages.
            - **Validation Accuracy:** The percentage of unseen test functions the model correctly classified as either safe or vulnerable.
            - **AUROC (Area Under ROC Curve):** A score from 0.5 (random guess) to 1.0 (perfect) measuring how cleanly the model separates safe code from dangerous code.
            - **F1 Score:** A balanced score between 0 and 1 that rewards catching real vulnerabilities while penalizing false alarms.
            - **Latency:** The speed (in milliseconds) it takes for the model to analyze a function on a standard computer CPU.
            - **GPAC:** An open-source multimedia and video streaming software engine written in C, used as an external test codebase.
            - **Vim:** The famous open-source terminal text editor written in C, used as an external benchmark for complex utility code.
            - **OpenSSL:** A critical open-source cryptography and secure communications library written in C, used to test detection on security-sensitive code.
            - **Envoy Proxy:** A modern, high-performance cloud networking proxy written in C++, used to test if models trained on C can generalize to modern C++.
            """)
        
    with tab2:
        st.markdown("### Model B vs Model A: Locked 32-Example Independent Benchmark")
        st.caption("Cryptographically verified evaluation on 8 security tasks (32 units; 16 vulnerable, 16 safe) completely held out from Model B training.")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model B Accuracy", "96.88%", "+40.63% vs Model A (56.25%)")
        c2.metric("Model B Recall", "93.75%", "15/16 vulnerabilities caught")
        c3.metric("Model B Precision", "100.00%", "0 False Positives (vs 12 for Model A)")
        c4.metric("Model B Macro F1", "0.9677", "MCC: +0.9393 | AUROC: 0.9844")
        
        st.markdown("#### Head-to-Head Comparison on Identical 32 Units:")
        comp_df = pd.DataFrame([
            {"Metric": "Accuracy", "Model A (Code Only)": "56.25% (18/32)", "Model B (Code + Prov)": "96.88% (31/32)", "Delta (B − A)": "+40.63%"},
            {"Metric": "95% Wilson CI", "Model A (Code Only)": "[39.3%, 71.8%]", "Model B (Code + Prov)": "[84.3%, 99.5%]", "Delta (B − A)": "Significant"},
            {"Metric": "Precision", "Model A (Code Only)": "53.85% (14/26)", "Model B (Code + Prov)": "100.00% (15/15)", "Delta (B − A)": "+46.15%"},
            {"Metric": "Recall", "Model A (Code Only)": "87.50% (14/16)", "Model B (Code + Prov)": "93.75% (15/16)", "Delta (B − A)": "+6.25%"},
            {"Metric": "False Positives", "Model A (Code Only)": "12 / 16 safe code units", "Model B (Code + Prov)": "0 / 16 safe code units", "Delta (B − A)": "-12 False Positives"},
            {"Metric": "AUROC", "Model A (Code Only)": "0.5938", "Model B (Code + Prov)": "0.9844", "Delta (B − A)": "+0.3906"},
            {"Metric": "Brier Score", "Model A (Code Only)": "0.2951", "Model B (Code + Prov)": "0.0294", "Delta (B − A)": "-0.2657 (calibrated)"},
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        > [!NOTE]
        > **Methodological Guarantee:**
        > Model B was trained strictly on an independent 16-task development corpus (`data/model_b_dev/`, 64 units) with **zero task or code overlap** with these 32 benchmark examples. The 32 benchmark examples are cryptographically locked in `data/locked_benchmark_32/` (SHA-256: `d53e2003...`) ensuring strictly independent test evaluation.
        """)
        
        with st.expander("📖 Model B Metric & Statistical Guide", expanded=False):
            st.markdown("""
            - **Accuracy:** The overall percentage of benchmark samples the model got completely right (31 out of 32 = 96.88%).
            - **Recall:** The percentage of real vulnerabilities the model successfully detected without missing them (15 out of 16 caught = 93.75%).
            - **Precision:** The percentage of flagged alerts that were actually real vulnerabilities, measuring freedom from false alarms (100% means zero wasted audits).
            - **Macro F1:** The balanced score across both safe and vulnerable classes, ensuring the model performs equally well on both.
            - **95% Wilson Confidence Interval (CI):** A rigorous statistical range proving with 95% certainty where the true accuracy lies (between 84.3% and 99.5%).
            - **False Positives:** Safe, benign code snippets mistakenly flagged as dangerous (Model B had 0 false positives, eliminating Model A's 12 false alarms).
            - **AUROC:** The probability that the model ranks a randomly chosen vulnerable snippet higher in risk than a randomly chosen safe snippet (0.9844 is near-perfect).
            - **Brier Score:** Measures how accurate and calibrated the predicted probabilities are (lower is better; 0 is perfection, 0.0294 means the probabilities closely match real risk).
            """)



# =========================================================================================
# PAGE 4: Dataset & Provenance Audit Explorer
# =========================================================================================
elif page == "Dataset & Provenance Audit":
    st.markdown("""
    <div class="main-header">
        <span class="badge-sub">DATASET AUDIT</span>
        <h1>Comprehensive Dataset & Provenance Audit</h1>
        <p>Empirical audit of all 6 local vulnerability datasets, verifying provenance integrity,
        class balance, and suitability for training Model A and Model B.</p>
    </div>
    """, unsafe_allow_html=True)
    
    audit_data = [
        {
            "Dataset": "Devign",
            "Usable Samples": "27,318",
            "Vulnerability Label": "Yes ('target')",
            "Provenance Labels": "No (100% Pre-LLM Human)",
            "Trustworthiness": "High (Manual verification)",
            "Eligible For": "Model A Only",
            "Notes": "C code from QEMU/FFmpeg (2019)"
        },
        {
            "Dataset": "PrimeVul",
            "Usable Samples": "100,000+",
            "Vulnerability Label": "Yes ('target', 'cwe')",
            "Provenance Labels": "No (100% Pre-LLM Human)",
            "Trustworthiness": "High (Curated CVEs)",
            "Eligible For": "Model A Only",
            "Notes": "Multi-language historical benchmark"
        },
        {
            "Dataset": "BigVul",
            "Usable Samples": "188,636",
            "Vulnerability Label": "Yes (Vulnerable/Fixed)",
            "Provenance Labels": "No (100% Pre-LLM Human)",
            "Trustworthiness": "High (Git commit pairs)",
            "Eligible For": "Model A Only",
            "Notes": "Used for diff-aware ranking experiments"
        },
        {
            "Dataset": "DiverseVul",
            "Usable Samples": "330,492",
            "Vulnerability Label": "Yes ('target')",
            "Provenance Labels": "No (100% Pre-LLM Human)",
            "Trustworthiness": "High (18,945 CVEs)",
            "Eligible For": "Model A Only",
            "Notes": "Massive modern vulnerability corpus"
        },
        {
            "Dataset": "HumanVsAICode",
            "Usable Samples": "1,980,000",
            "Vulnerability Label": "No (0% Security Labels)",
            "Provenance Labels": "Yes ('source': AI vs Human)",
            "Trustworthiness": "Medium (Generated code)",
            "Eligible For": "Neither (No vuln ground truth)",
            "Notes": "Authorship attribution only; cannot train risk models"
        },
        {
            "Dataset": "Controlled Multi-CWE Benchmark",
            "Usable Samples": "32",
            "Vulnerability Label": "Yes (16 Vuln, 16 Safe)",
            "Provenance Labels": "Yes (Verified 4-stage lineage)",
            "Trustworthiness": "Gold Standard (Controlled)",
            "Eligible For": "Model A & Model B",
            "Notes": "8 tasks across CWE-119, 120, 190, 476, 787"
        }
    ]
    st.dataframe(pd.DataFrame(audit_data), use_container_width=True, hide_index=True)


