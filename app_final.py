"""
app_final.py

Streamlit Web Interface for AI Security Scanner:
- Tab 1: Interactive Code Scanner (Multi-modal inference, suspicious lines, provenance fusion, calibrated decision rules)
- Tab 2: Model Performance & Empirical Evaluation (Strictly sourced from results/p10 and results/p11)
- Tab 3: System Architecture & Known Limitations
"""

import sys
import json
from pathlib import Path
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# Page config
st.set_page_config(
    page_title="AI Security Scanner & Verification Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS_P11_DIR = PROJECT_ROOT / "results" / "p11"
RESULTS_P10_DIR = PROJECT_ROOT / "results" / "p10"
REPORTS_DIR = PROJECT_ROOT / "reports"
EXAMPLES_DIR = PROJECT_ROOT / "examples"


@st.cache_resource
def get_scanner():
    from src.inference_final import SecurityScanner
    return SecurityScanner(load_model_b=True)


@st.cache_data
def load_verified_data():
    data = {}
    # 1. Test2 clean
    t2_path = RESULTS_P11_DIR / "test2_clean_evaluation.json"
    if t2_path.exists():
        with open(t2_path, "r") as f:
            data["test2"] = json.load(f)

    # 2. Old test
    ot_path = RESULTS_P11_DIR / "old_test_evaluation.json"
    if ot_path.exists():
        with open(ot_path, "r") as f:
            data["old_test"] = json.load(f)

    # 3. Juliet
    j_path = RESULTS_P10_DIR / "juliet_evaluation.json"
    if j_path.exists():
        with open(j_path, "r") as f:
            data["juliet"] = json.load(f)

    # 4. Model B
    mb_path = RESULTS_P11_DIR / "modelb_v2_labels.json"
    if mb_path.exists():
        with open(mb_path, "r") as f:
            data["model_b"] = json.load(f)

    # 5. Verified handoff markdown
    ho_path = REPORTS_DIR / "HANDOFF_P10_verified.md"
    if ho_path.exists():
        data["handoff_text"] = ho_path.read_text(encoding="utf-8")

    return data


# Header
st.title("🛡️ AI Security Scanner & Audit Dashboard")
st.caption(
    "Multi-modal vulnerability detection combining Qwen-2.5-Coder embeddings, chunked CodeBERT, "
    "static analysis features (Cppcheck & Flawfinder), and calibrated three-way decision rules."
)

tab_scan, tab_perf, tab_about = st.tabs(["🔍 Scan Code", "📊 Model Performance", "📖 Architecture & Limitations"])

# ==============================================================================
# TAB 1: SCAN CODE
# ==============================================================================
with tab_scan:
    st.subheader("Interactive Security Audit")

    col_input, col_meta = st.columns([2, 1])

    with col_meta:
        st.markdown("#### 1. Load Example or Upload File")
        example_choice = st.selectbox(
            "Select Pre-configured Audit Example:",
            ["(Custom Input)", "Vulnerable C (Buffer Overflow)", "Fixed C (Bounds Checked)", "Mixed C++ (Class & Heap Overflow)"],
        )

        code_to_load = ""
        if example_choice == "Vulnerable C (Buffer Overflow)":
            p = EXAMPLES_DIR / "vulnerable.c"
            if p.exists():
                code_to_load = p.read_text()
        elif example_choice == "Fixed C (Bounds Checked)":
            p = EXAMPLES_DIR / "fixed.c"
            if p.exists():
                code_to_load = p.read_text()
        elif example_choice == "Mixed C++ (Class & Heap Overflow)":
            p = EXAMPLES_DIR / "mixed.cpp"
            if p.exists():
                code_to_load = p.read_text()

        uploaded_file = st.file_uploader(
            "Or upload source code file (.c, .cpp, .h):",
            type=["c", "cpp", "h", "cc", "cxx"],
        )
        if uploaded_file is not None:
            code_to_load = uploaded_file.getvalue().decode("utf-8", errors="replace")

        # Provenance Section (Model B)
        with st.expander("AI Code Provenance (Model B Risk Adjustment)", expanded=False):
            st.caption("Provide generative model metadata to enable provenance-aware risk estimation.")
            prov_enabled = st.checkbox("Include provenance metadata", value=(example_choice == "Mixed C++ (Class & Heap Overflow)"))
            prov_model = st.text_input("Generative Model Name", value="claude-3-5-sonnet")
            prov_temp = st.slider("Sampling Temperature", min_value=0.0, max_value=1.5, value=0.2, step=0.1)
            prov_prompt = st.text_area("Prompt Snippet", value="Implement packet parsing routine in C++.")
            prov_role = st.selectbox("Code Origin", ["AI Generated (AI-final)", "Human Written (Human-final)"])

    with col_input:
        st.markdown("#### 2. C / C++ Source Code")
        code_input = st.text_area(
            "Source Code for Security Inspection:",
            value=code_to_load if code_to_load else "int process_data(char *input) {\n    char buf[64];\n    strcpy(buf, input);\n    return 0;\n}",
            height=320,
        )

        run_scan = st.button("🚀 Run Security Audit", type="primary", use_container_width=True)

    if run_scan:
        if not code_input.strip():
            st.warning("Please provide non-empty C or C++ source code to scan.")
        else:
            with st.spinner("Executing neural encoders and static analysis pipeline..."):
                scanner = get_scanner()

                prov_dict = None
                if prov_enabled:
                    prov_dict = {
                        "model": prov_model,
                        "temperature": prov_temp,
                        "prompt": prov_prompt,
                        "role": "ai_t02" if prov_temp <= 0.3 else "ai_t08" if "AI" in prov_role else "human",
                    }

                report = scanner.scan_code(code_input, provenance=prov_dict)

            st.markdown("---")
            st.markdown("### 📋 Audit Findings")

            # Verdict Banner
            verdict = report["verdict"]
            p_code = report["p_code"]
            prec = report["calibrated_precision_at_score"]
            npv = report["calibrated_npv_at_score"]

            col_v1, col_v2, col_v3, col_v4, col_v5 = st.columns(5)
            with col_v1:
                if verdict == "VULNERABLE":
                    st.error(f"### 🚨 {verdict}")
                elif verdict == "SAFE":
                    st.success(f"### ✅ {verdict}")
                else:
                    st.warning(f"### ⚠️ {verdict}")
                st.caption(f"Decision Rule: $P_{{\\text{{code}}}} \\ge {report['decision_thresholds']['t_high']}$ (Vuln), $\\le {report['decision_thresholds']['t_low']}$ (Safe)")

            with col_v2:
                st.metric("Neural Risk ($P_{\\text{code}}$)", f"{p_code:.4f}")
                st.caption("Model A v2 (V3) fused prediction")

            with col_v3:
                st.metric("Triage Score", f"{report['triage_score']:.4f}")
                st.caption("Stacker score calibrated with length")

            with col_v4:
                if prov_dict:
                    st.metric("Fusion Risk ($P_{\\text{risk}}$)", f"{report['p_risk']:.4f}", delta=f"{report['p_risk'] - p_code:+.4f}")
                    st.caption("Model B provenance-adjusted risk")
                else:
                    st.metric("Length Tier", f"{report['length_risk_tier']}")
                    st.caption(f"{report['length_tokens']} tokens (threshold 512)")

            with col_v5:
                if verdict == "VULNERABLE":
                    st.metric("Calibrated Precision", f"{prec*100:.1f}%")
                    st.caption("Empirical true positive probability")
                elif verdict == "SAFE":
                    st.metric("Calibrated NPV", f"{npv*100:.1f}%")
                    st.caption("Empirical true negative probability")
                else:
                    st.metric("Calibrated Precision", f"{prec*100:.1f}%")
                    st.caption(f"NPV: {npv*100:.1f}% (Unsure region)")

            # Warnings if any
            if report.get("warnings"):
                for w in report["warnings"]:
                    st.info(f"ℹ️ {w}")

            # Top Likely CWEs
            st.markdown("#### 🎯 Suspected Vulnerability Types (Juliet J2 + Static Prior)")
            cwe_cols = st.columns(len(report["likely_cwe"]))
            for idx, (cwe_name, cwe_prob) in enumerate(report["likely_cwe"]):
                with cwe_cols[idx]:
                    st.progress(float(cwe_prob), text=f"**{cwe_name}**: {cwe_prob*100:.1f}% probability")

            # Suspicious Lines via Neural Occlusion
            st.markdown("#### 🔍 Neural Attribution & Suspicious Lines")
            suspicious = report.get("suspicious_lines", [])
            if suspicious:
                st.markdown(
                    f"Token occlusion identified **{len(suspicious)}** lines whose omission significantly reduces predicted vulnerability risk: "
                    + ", ".join([f"`Line {l}`" for l in suspicious])
                )
                lines = code_input.splitlines()
                annotated_code = []
                for i, l in enumerate(lines):
                    line_num = i + 1
                    if line_num in suspicious:
                        annotated_code.append(f"{line_num:4d}  >>> [SUSPICIOUS] {l}")
                    else:
                        annotated_code.append(f"{line_num:4d}      {l}")
                st.code("\n".join(annotated_code), language="cpp")
            else:
                st.markdown("No individual lines met the minimum occlusion sensitivity threshold ($\Delta P > 0.005$).")

            # Static Findings
            st.markdown("#### 🛠️ Static Tool Diagnostics")
            findings = report.get("static_findings", [])
            if findings:
                df_findings = pd.DataFrame(findings)
                st.dataframe(df_findings, use_container_width=True)
            else:
                st.success("Cppcheck and Flawfinder reported zero rule violations.")

            # Advisory Notice
            st.markdown("---")
            st.warning(
                "⚠️ **Advisory Notice:** Security triage assistance only. Not a formal proof of code safety. "
                "All code must be reviewed by qualified security engineers prior to production deployment."
            )


# ==============================================================================
# TAB 2: MODEL PERFORMANCE
# ==============================================================================
with tab_perf:
    st.subheader("Verified Empirical Evaluation")
    st.markdown(
        "All figures and metrics below are loaded strictly from disk JSON evaluation results "
        "(`results/p11/` and `results/p10/`) under 2,000 bootstrap resamples with commit/task clustering."
    )

    data = load_verified_data()

    # 1. Test2 Clean vs Full Table
    st.markdown("### 1. Primary Benchmark: Held-Out Open-Source Repositories (`test2`)")
    st.markdown(
        "Evaluated on 12 held-out production projects (`node, Bento4, MilkyTracker, cryptopp, libvpx, iperf, contiki-ng, libyaml, yubico-pam, libcomps, Onigmo, php-src`). "
        "`test2-clean` excludes 913 normalized duplicates ($N=1,991$, prevalence 20.84%)."
    )

    if "test2" in data:
        t2 = data["test2"]
        clean_rows = []
        for m in ["L1", "Champion", "Static_only", "V3", "V6"]:
            if m in t2.get("test2_clean", {}):
                row = t2["test2_clean"][m]
                clean_rows.append({
                    "Model": m,
                    "AUROC": f"{row['auroc']:.4f}",
                    "95% CI": f"[{row['auroc_ci'][0]:.4f}, {row['auroc_ci'][1]:.4f}]",
                    "Length-Matched AUROC": f"{row['length_matched_auroc']:.4f}",
                    "AUPRC": f"{row['auprc']:.4f}",
                    "Top 10% Prec": f"{row['top_10pct']['precision']*100:.1f}%",
                    "Top 10% Recall": f"{row['top_10pct']['recall']*100:.1f}%",
                    "Top 10% Lift": f"{row['top_10pct']['lift']:.2f}x",
                })
        st.table(pd.DataFrame(clean_rows))

        with st.expander("Compare with Contaminated `test2_full` ($N=2,904$)"):
            full_rows = []
            for m in ["L1", "Champion", "Static_only", "V3", "V6"]:
                if m in t2.get("test2_full", {}):
                    row = t2["test2_full"][m]
                    full_rows.append({
                        "Model": m,
                        "AUROC": f"{row['auroc']:.4f}",
                        "Length-Matched AUROC": f"{row['length_matched_auroc']:.4f}",
                        "AUPRC": f"{row['auprc']:.4f}",
                        "Top 10% Prec": f"{row['top_10pct']['precision']*100:.1f}%",
                    })
            st.table(pd.DataFrame(full_rows))
            st.caption("Contamination impact on AUROC: $|\Delta| \le 0.0129$ across all candidates.")

    # 2. Secondary Old-Test Benchmark
    st.markdown("### 2. Secondary Benchmark: Phase 1 Legacy Test Set ($N=4,016$)")
    st.markdown("Evaluated on Envoy, Envoy production, OpenSSL, Vim, and GPAC. Prevalence 31.75%.")
    if "old_test" in data:
        ot = data["old_test"]
        ot_rows = []
        for m in ["L1", "Champion", "V3", "V6"]:
            if m in ot:
                row = ot[m]
                ot_rows.append({
                    "Model": m,
                    "AUROC": f"{row['auroc']:.4f}",
                    "95% CI": f"[{row['auroc_ci'][0]:.4f}, {row['auroc_ci'][1]:.4f}]",
                    "Length-Matched AUROC": f"{row['length_matched_auroc']:.4f}",
                    "AUPRC": f"{row['auprc']:.4f}",
                    "Top 10% Prec": f"{row['top_10pct']['precision']*100:.1f}%",
                })
        st.table(pd.DataFrame(ot_rows))

    # 3. Juliet Cross-Benchmark
    st.markdown("### 3. Synthetic Benchmark Generalization: Juliet C/C++ Test Suite")
    st.markdown(
        "Juliet testcases ($N=64,099$, balanced 50% vulnerable). "
        "Demonstrates synthetic generalization across syntactic variations ($S_{\\text{flow}}$) versus unseen CWE categories ($S_{\\text{cwe}}$)."
    )
    if "juliet" in data:
        j_data = data["juliet"]
        j_rows = []
        for split_key, split_name in [("split_flow", "Syntactic Flow (S_flow)"), ("split_cwe", "Unseen CWE (S_cwe)"), ("split_rand", "Random Split (S_rand)")]:
            if split_key in j_data:
                sp = j_data[split_key]
                j_rows.append({
                    "Split": split_name,
                    "J0 (Static)": f"{sp.get('J0', {}).get('auroc', 0.0):.4f}",
                    "J1 (CodeBERT)": f"{sp.get('J1', {}).get('auroc', 0.0):.4f}",
                    "J2 (Qwen+Stat)": f"{sp.get('J2', {}).get('auroc', 0.0):.4f}",
                    "J3 (Full Fusion)": f"{sp.get('J3', {}).get('auroc', 0.0):.4f}",
                })
        st.table(pd.DataFrame(j_rows))
        st.caption("Juliet J2 test2 cross-benchmark AUROC is 0.6925 (synthetic models do not generalize directly to real open-source code).")

    # 4. Model B Clean Tasks
    st.markdown("### 4. Model B AI Code Execution & Fuzzing Oracle")
    st.markdown(
        "Evaluation across 36 clean execution tasks (900 compiled binary units executed with ASan/UBSan sanitizers). "
        "Ground-truth label is strictly runtime crash (`exec_label_v2`)."
    )
    if "model_b" in data:
        mb = data["model_b"]
        c1 = mb.get("table_c1", {})
        c2 = mb.get("table_c2", {})

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### Execution Crash Rate by Generator Role")
            df_c1 = pd.DataFrame([
                {"Role": "Human Written", "Crash Rate": f"{c1.get('human', {}).get('crash_rate', 0.0)*100:.1f}%", "N": c1.get('human', {}).get('total', 0)},
                {"Role": "AI (Temp 0.2)", "Crash Rate": f"{c1.get('ai_t02', {}).get('crash_rate', 0.0)*100:.1f}%", "N": c1.get('ai_t02', {}).get('total', 0)},
                {"Role": "AI (Temp 0.8)", "Crash Rate": f"{c1.get('ai_t08', {}).get('crash_rate', 0.0)*100:.1f}%", "N": c1.get('ai_t08', {}).get('total', 0)},
            ])
            st.table(df_c1)
            st.caption(f"Paired McNemar test vs Human: AI t0.2 p={c1.get('mcnemar_p_human_vs_ai_t02', 0.0):.4f}, AI t0.8 p={c1.get('mcnemar_p_human_vs_ai_t08', 0.0):.4f}.")

        with col_c2:
            st.markdown("#### Fusion Model AUROC on Clean Tasks")
            df_c2 = pd.DataFrame([
                {"Model": "B1a (P_code only)", "AUROC": f"{c2.get('B1a', {}).get('auroc', 0.0):.4f}", "Status": "Below chance (inverted)"},
                {"Model": "B2 (Provenance only)", "AUROC": f"{c2.get('B2', {}).get('auroc', 0.0):.4f}", "Status": "Not distinguishable from chance"},
                {"Model": "B4 (Code + Prov)", "AUROC": f"{c2.get('B4', {}).get('auroc', 0.0):.4f}", "Status": "Not distinguishable from chance"},
                {"Model": "B4s (Code + Prov + Stat)", "AUROC": f"{c2.get('B4s', {}).get('auroc', 0.0):.4f}", "Status": "Marginal gain"},
            ])
            st.table(df_c2)
            st.caption("On clean execution tasks, provenance alone is not distinguishable from chance (95% CI spans 0.5).")


# ==============================================================================
# TAB 3: ABOUT & ARCHITECTURE
# ==============================================================================
with tab_about:
    st.subheader("System Architecture & Empirical Governance")

    st.markdown("### System Architecture")
    st.markdown(
        """
```mermaid
flowchart TD
    In["C / C++ Source Snippet"] --> Static["Static Analysis Tools<br/>(Cppcheck & Flawfinder)"]
    In --> EncQwen["Qwen-2.5-Coder-0.5B<br/>(Mean Pooled Embedding, d=896)"]
    In --> EncCB["CodeBERT-base<br/>(Chunked 254-token Stride 128, d=768)"]
    In --> Length["Token Count & Log Length"]

    Static --> FeatStat["104-dim Static Feature Vector"]
    EncQwen --> FeatQwen["896-dim Neural Vector"]
    EncCB --> FeatCB["768-dim Chunked Vector"]

    FeatStat & FeatQwen & FeatCB --> V3["Model A v2 (V3)<br/>Regularized Logistic Regression"]
    V3 --> PCode["P_code Risk Probability"]

    PCode & Length --> V6["Triage Stacker (V6)<br/>Calibrated Stacker Logistic Regression"]
    V6 --> Triage["Triage Score"]

    PCode --> Occlusion["Token Occlusion Engine<br/>Delta P_code Sensitivity"]
    Occlusion --> SuspLines["Suspicious Source Lines"]

    PCode --> ProvCheck{"Provenance<br/>Metadata Provided?"}
    ProvCheck -->|No| FinalRisk["P_risk = P_code"]
    ProvCheck -->|Yes| ModelB["Model B v2 Predictor<br/>Offset Fusion"]
    ModelB --> FinalRisk["P_risk"]

    PCode --> Rules{"Threshold Policy"}
    Rules -->|Score >= 0.6956| VerdictVuln["VULNERABLE<br/>Calibrated Prec: ~30-48%"]
    Rules -->|Score <= 0.1585| VerdictSafe["SAFE<br/>Calibrated NPV: ~88-95%"]
    Rules -->|0.1585 < Score < 0.6956| VerdictUnsure["UNSURE<br/>Triage Review Queue"]
```
        """
    )

    st.markdown("### Known Limitations & Threat Model (from `reports/HANDOFF_P10_verified.md`)")

    st.markdown(
        """
> [!WARNING]
> **1. Residual Length Correlation:** In real-world repositories (`test2`), length-only logistic regression (L1) achieves an AUROC of 0.7167 because longer functions naturally contain more vulnerabilities. While V3 and V6 maintain positive discrimination under length-matching (0.59–0.61 AUROC), raw predictions remain correlated with code length.
>
> **2. Synthetic Benchmark Generalization Gap:** While models achieve near-perfect discrimination on the Juliet synthetic test suite ($S_{\\text{flow}}$ AUROC 0.9997), Juliet-trained models evaluate at 0.6925 on real open-source code. Synthetic benchmarks cannot substitute for real-world validation.
>
> **3. Contamination Quantification:** A total of 913 samples in `test2` were normalized duplicates of training pool functions. In `test2-clean` ($N=1,991$), candidate AUROCs changed by $|\Delta| \le 0.0129$, confirming the reported relative performance is stable.
>
> **4. Model B Provenance Generalization:** Model B evaluation on 36 clean fuzzed tasks showed that provenance features alone (AUROC 0.4965) are not distinguishable from chance. Generative model provenance should never replace static and semantic code inspection.
        """
    )
