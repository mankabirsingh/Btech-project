#!/usr/bin/env python3
"""
Provenance-Aware Security Risk Assessment — Streamlit Application
==================================================================
Interactive research prototype for DTU B.Tech Project-I.

Features:
1. Security Analysis: Paste code → get vulnerability analysis + ML prediction
2. Dataset Explorer: View loaded datasets, distributions, and provenance info
3. Model Results: Evaluation metrics, confusion matrices, comparison charts
4. Provenance Experiment: Controlled editing experiment visualization
5. Project Status: Implementation progress and remaining work

Usage:
    streamlit run app.py
"""

import sys
import json
import logging
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================
# Page Configuration
# ========================
st.set_page_config(
    page_title="Provenance-Aware Security Risk Assessment",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========================
# Custom CSS
# ========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header p {
        color: #b8b8cc;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .metric-card h3 {
        color: #7b2ff7;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-card p {
        color: #8888aa;
        font-size: 0.85rem;
        margin: 0.3rem 0 0 0;
    }
    
    .severity-critical { color: #ff4757; font-weight: 700; }
    .severity-high { color: #ff6b6b; font-weight: 600; }
    .severity-medium { color: #ffa502; font-weight: 600; }
    .severity-low { color: #2ed573; font-weight: 500; }
    
    .risk-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .risk-critical { background: rgba(255,71,87,0.2); color: #ff4757; border: 1px solid #ff4757; }
    .risk-high { background: rgba(255,107,107,0.2); color: #ff6b6b; border: 1px solid #ff6b6b; }
    .risk-medium { background: rgba(255,165,2,0.2); color: #ffa502; border: 1px solid #ffa502; }
    .risk-low { background: rgba(46,213,115,0.2); color: #2ed573; border: 1px solid #2ed573; }
    
    .finding-card {
        background: rgba(255,255,255,0.03);
        border-left: 4px solid;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.5rem;
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e);
    }
</style>
""", unsafe_allow_html=True)


# ========================
# Helper Functions
# ========================
@st.cache_resource
def load_baseline_model():
    """Load trained vulnerability model (best tri-union or baseline)."""
    import pickle
    tri_path = PROJECT_ROOT / "models" / "checkpoints" / "baseline_tri_union_best.pkl"
    if tri_path.exists():
        with open(tri_path, 'rb') as f:
            data = pickle.load(f)
            return data['pipeline'], "Best Tri-Union Model (Char+Token+SecurityStruct)"
    model_path = PROJECT_ROOT / "models" / "checkpoints" / "baseline_char.pkl"
    if model_path.exists():
        from src.models.baseline import VulnerabilityBaseline
        return VulnerabilityBaseline.load(model_path), "Char N-gram Baseline"
    return None


@st.cache_data
def load_metrics(model_name: str):
    """Load saved metrics for a model."""
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    path = metrics_dir / f"{model_name}_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_dataset_info():
    """Load dataset inspection reports."""
    reports_dir = PROJECT_ROOT / "data" / "reports"
    reports = {}
    if reports_dir.exists():
        for f in reports_dir.glob("inspection_*.json"):
            name = f.stem.replace("inspection_", "")
            with open(f) as fh:
                reports[name] = json.load(fh)
    return reports


@st.cache_data
def load_split_info():
    """Load split summary."""
    splits_dir = PROJECT_ROOT / "data" / "splits"
    summary_path = splits_dir / "vulnerability_split_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return None


def get_available_plots():
    """Get all available plot images."""
    plots_dir = PROJECT_ROOT / "results" / "plots"
    if plots_dir.exists():
        return list(plots_dir.glob("*.png"))
    return []


# ========================
# Sidebar
# ========================
with st.sidebar:
    st.markdown("## 🔒 Navigation")
    page = st.radio(
        "Select Page",
        [
            "🛡️ Security Analysis",
            "📊 Dataset Explorer",
            "📈 Model Results",
            "🔬 Provenance Experiment",
            "📋 Project Status",
        ],
        label_visibility="collapsed",
    )
    
    st.markdown("---")
    st.markdown("### 🎓 Project Info")
    st.markdown("""
    **B.Tech Project-I**  
    Delhi Technological University
    
    **Team:**  
    • Mankabir Singh (23/IT/199)  
    • Mohnish Gautam (23/IT/200)
    
    **Research Question:**  
    *Can provenance-aware analysis improve security-risk prediction?*
    """)
    
    st.markdown("---")
    st.caption("Provenance-Aware Security Risk Assessment v0.1")


# ========================
# Header
# ========================
st.markdown("""
<div class="main-header">
    <h1>🔒 Provenance-Aware Security Risk Assessment</h1>
    <p>Investigating whether code provenance and AI-involvement metadata can improve 
    vulnerability detection and security review prioritisation for human-AI hybrid code.</p>
</div>
""", unsafe_allow_html=True)


# ========================
# PAGE 1: Security Analysis
# ========================
if page == "🛡️ Security Analysis":
    st.markdown("## 🛡️ Code Security Analysis")
    st.markdown("Paste a code function to analyze its security properties using "
                "pattern-based analysis and trained ML models.")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        language = st.selectbox("Language", ["C/C++", "Python", "JavaScript", "Java", "Auto-detect"])
        
        default_code = '''#include <stdio.h>
#include <string.h>

void process_input(char *input) {
    char buffer[64];
    strcpy(buffer, input);
    printf("Result: %s\\n", buffer);
}

int main() {
    char data[1024];
    printf("Enter data: ");
    gets(data);
    process_input(data);
    return 0;
}'''
        
        code_input = st.text_area("Code Input", value=default_code, height=350)
        
        uploaded_file = st.file_uploader("Or upload a code file", type=['c', 'cpp', 'py', 'java', 'js', 'h'])
        if uploaded_file:
            code_input = uploaded_file.read().decode('utf-8')
        
        analyze_btn = st.button("🔍 Analyze Security", type="primary", use_container_width=True)
    
    with col2:
        if analyze_btn and code_input.strip():
            from src.security.static_analysis import analyze_code, detect_language
            
            lang = language.lower() if language != "Auto-detect" else None
            analysis = analyze_code(code_input, lang)
            
            # Risk Level Badge
            risk_class = f"risk-{analysis['risk_level'].lower()}"
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 1rem;">
                <span class="risk-badge {risk_class}">
                    Risk Level: {analysis['risk_level']}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics row
            cols = st.columns(3)
            cols[0].metric("Findings", analysis['n_findings'])
            cols[1].metric("Risk Score", analysis['risk_score'])
            cols[2].metric("Language", analysis['language'].upper())
            
            # CWEs
            if analysis['cwes']:
                st.markdown("**CWE Categories Found:**")
                for cwe in analysis['cwes']:
                    st.markdown(f"- `{cwe}`")
            
            # ML Prediction
            model_info = load_baseline_model()
            if model_info:
                model, m_name = model_info
                st.markdown("---")
                st.markdown(f"**🤖 ML Model Prediction ({m_name}):**")
                pred = model.predict([code_input])[0]
                prob = model.predict_proba([code_input])[0]
                
                pred_label = "🔴 Vulnerable" if pred == 1 else "🟢 Benign"
                confidence = max(prob) * 100
                
                st.markdown(f"Prediction: **{pred_label}** ({confidence:.1f}% confidence)")
                st.progress(float(prob[1]))
                st.caption(f"*Inference generated by {m_name}*")
            else:
                st.info("ML model not yet trained. Run `python scripts/train_improved_models.py`")
            
            # Findings detail
            st.markdown("---")
            st.markdown("**Detailed Findings:**")
            
            for finding in analysis['findings']:
                sev_class = f"severity-{finding['severity']}"
                remed = finding.get('remediation')
                remed_html = f"<br><span style='color: #2ed573;'><small>💡 <b>Recommended Remediation:</b> {remed}</small></span>" if remed else ""
                st.markdown(f"""
                <div class="finding-card" style="border-color: {'#ff4757' if finding['severity']=='critical' else '#ff6b6b' if finding['severity']=='high' else '#ffa502' if finding['severity']=='medium' else '#2ed573'};">
                    <strong class="{sev_class}">[{finding['severity'].upper()}]</strong> 
                    <code>{finding['cwe']}</code> — {finding['message']}<br>
                    <small>Line {finding.get('line', '?')} | Tool: {finding['tool'].upper()} | 
                    Confidence: {finding['confidence'].title()}</small>
                    {f"<br><code>{finding.get('code_snippet', '')}</code>" if finding.get('code_snippet') else ""}
                    {remed_html}
                </div>
                """, unsafe_allow_html=True)
            
            if not analysis['findings']:
                st.success("No security findings detected by pattern analysis.")
            
            st.caption(f"⚠️ {analysis['note']}")
        
        elif analyze_btn:
            st.warning("Please enter code to analyze.")


# ========================
# PAGE 2: Dataset Explorer
# ========================
elif page == "📊 Dataset Explorer":
    st.markdown("## 📊 Dataset Explorer")
    
    dataset_reports = load_dataset_info()
    split_info = load_split_info()
    
    if not dataset_reports:
        st.warning("No dataset reports found. Run `python scripts/prepare_dataset.py` first.")
    else:
        tabs = st.tabs(list(dataset_reports.keys()) + ["Split Info"])
        
        for i, (name, report) in enumerate(dataset_reports.items()):
            with tabs[i]:
                st.markdown(f"### {name}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Samples", f"{report['num_samples']:,}")
                col2.metric("Columns", len(report['columns']))
                col3.metric("Duplicates", report.get('duplicate_code_samples', 'N/A'))
                
                st.markdown("**Columns:**")
                st.code(str(report['columns']))
                
                # Vulnerability columns
                if report.get('vulnerability_related_columns'):
                    st.markdown("**Vulnerability Labels:**")
                    for col in report['vulnerability_related_columns']:
                        key = f"label_dist_{col}"
                        if key in report:
                            st.json(report[key])
                
                # Provenance columns
                if report.get('provenance_columns'):
                    st.markdown("**Provenance Information:**")
                    for col in report['provenance_columns']:
                        key = f"prov_dist_{col}"
                        if key in report:
                            st.json(report[key])
                
                # Code length stats
                if report.get('code_length_stats'):
                    st.markdown("**Code Length Statistics:**")
                    st.json(report['code_length_stats'])
                
                # Missing values
                missing = {k: v for k, v in report.get('missing_values', {}).items() if v > 0}
                if missing:
                    st.markdown("**Missing Values:**")
                    st.json(missing)
                
                # Sample rows
                if report.get('sample_rows'):
                    st.markdown("**Sample Rows:**")
                    st.dataframe(pd.DataFrame(report['sample_rows']), use_container_width=True)
        
        # Split info tab
        with tabs[-1]:
            if split_info:
                st.markdown("### Train/Val/Test Split")
                for split_name, info in split_info.items():
                    st.markdown(f"**{split_name}:** {info['n_samples']:,} samples")
                    if info.get('label_dist'):
                        st.json(info['label_dist'])
            else:
                st.info("No split info available yet.")


# ========================
# PAGE 3: Model Results
# ========================
elif page == "📈 Model Results":
    st.markdown("## 📈 Model Evaluation Results")
    
    # Load available metrics
    metrics_files = list((PROJECT_ROOT / "results" / "metrics").glob("*.json")) if (PROJECT_ROOT / "results" / "metrics").exists() else []
    
    if not metrics_files:
        st.warning("No model results yet. Run `python scripts/train_baseline.py` first.")
    else:
        all_metrics = {}
        for mf in metrics_files:
            with open(mf) as f:
                data = json.load(f)
            model_name = data.get('model_name', mf.stem)
            all_metrics[model_name] = data.get('metrics', data)
        
        # Comparison table
        if len(all_metrics) > 1:
            st.markdown("### Model Comparison")
            comparison_rows = []
            for name, m in all_metrics.items():
                comparison_rows.append({
                    'Model': name,
                    'Accuracy': f"{m.get('accuracy', 0):.4f}",
                    'Precision': f"{m.get('precision', 0):.4f}",
                    'Recall': f"{m.get('recall', 0):.4f}",
                    'F1': f"{m.get('f1', 0):.4f}",
                    'F1 Macro': f"{m.get('f1_macro', 0):.4f}",
                    'MCC': f"{m.get('mcc', 0):.4f}",
                    'AUROC': f"{m.get('auroc', 'N/A')}",
                })
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
        
        # Individual model details
        model_tabs = st.tabs(list(all_metrics.keys()))
        
        for tab, (name, m) in zip(model_tabs, all_metrics.items()):
            with tab:
                st.markdown(f"### {name}")
                
                # Metric cards
                cols = st.columns(6)
                metric_items = [
                    ('Accuracy', m.get('accuracy', 0)),
                    ('Precision', m.get('precision', 0)),
                    ('Recall', m.get('recall', 0)),
                    ('F1', m.get('f1', 0)),
                    ('MCC', m.get('mcc', 0)),
                    ('AUROC', m.get('auroc', 'N/A')),
                ]
                for col, (label, val) in zip(cols, metric_items):
                    if isinstance(val, (int, float)):
                        col.metric(label, f"{val:.4f}")
                    else:
                        col.metric(label, str(val))
                
                # Confusion matrix
                if m.get('confusion_matrix'):
                    st.markdown("**Confusion Matrix:**")
                    cm = np.array(m['confusion_matrix'])
                    cm_df = pd.DataFrame(
                        cm,
                        index=['True Benign', 'True Vulnerable'],
                        columns=['Pred Benign', 'Pred Vulnerable'],
                    )
                    st.dataframe(cm_df, use_container_width=True)
                
                # Classification report
                if m.get('classification_report'):
                    st.markdown("**Classification Report:**")
                    cr = m['classification_report']
                    cr_rows = []
                    for label in ['benign', 'vulnerable']:
                        if label in cr:
                            cr_rows.append({
                                'Class': label,
                                'Precision': f"{cr[label]['precision']:.4f}",
                                'Recall': f"{cr[label]['recall']:.4f}",
                                'F1-Score': f"{cr[label]['f1-score']:.4f}",
                                'Support': cr[label]['support'],
                            })
                    if cr_rows:
                        st.dataframe(pd.DataFrame(cr_rows), use_container_width=True, hide_index=True)
                
                # Sample info
                st.markdown(f"**Samples:** {m.get('n_samples', '?')} "
                           f"(Positive: {m.get('n_positive', '?')}, "
                           f"Negative: {m.get('n_negative', '?')})")
        
        # Show plots
        plots = get_available_plots()
        if plots:
            st.markdown("### Evaluation Plots")
            plot_cols = st.columns(min(len(plots), 3))
            for i, plot_path in enumerate(plots):
                with plot_cols[i % 3]:
                    st.image(str(plot_path), caption=plot_path.stem.replace('_', ' ').title())


# ========================
# PAGE 4: Provenance Experiment
# ========================
elif page == "🔬 Provenance Experiment":
    st.markdown("## 🔬 Provenance-Aware Security & Review Prioritization")
    st.markdown("""
    This page investigates whether tracking code provenance (author type, AI-involvement %,
    and edit lineage) improves security risk assessment and review prioritization for human-AI hybrid code.
    """)
    
    benchmark_32_path = PROJECT_ROOT / "results" / "reports" / "controlled_benchmark_32.json"
    
    # Run/Reload button
    if st.button("🔄 Run/Reload Multi-CWE Controlled Benchmark (32 Versions)", type="primary"):
        from scripts.run_controlled_experiment import main as run_benchmark
        with st.spinner("Executing 8-task, 32-version controlled benchmark with Flawfinder and CodeBERT..."):
            run_benchmark()
        st.success("Benchmark executed successfully!")
        st.rerun()
    
    if benchmark_32_path.exists():
        with open(benchmark_32_path) as f:
            bench_data = json.load(f)
            
        st.markdown(f"**Benchmark Scope:** {bench_data['n_tasks']} security tasks | "
                    f"{bench_data['n_versions']} code versions (16 vulnerable, 16 secure, balanced)")
        
        # Prioritization Summary Card
        st.markdown("### 🎯 Code Review Prioritization Results")
        st.caption("How effectively does provenance awareness direct human reviewers to vulnerable code under tight audit budgets?")
        
        p_metrics = bench_data.get('prioritization_metrics', {})
        if p_metrics:
            p_rows = []
            for k_key, p_val in p_metrics.items():
                p_rows.append({
                    'Review Budget': f"{p_val['budget_pct']:.1f}% ({p_val['k']} PRs)",
                    'Code-Only Recall': f"{p_val['code_only']['recall']*100:.1f}% ({p_val['code_only']['vuln_found']}/16)",
                    'Code-Only Precision': f"{p_val['code_only']['precision']*100:.1f}%",
                    'Provenance-Aware Recall': f"⭐ {p_val['provenance_aware']['recall']*100:.1f}% ({p_val['provenance_aware']['vuln_found']}/16)",
                    'Provenance-Aware Precision': f"{p_val['provenance_aware']['precision']*100:.1f}%",
                    'Static Analysis Recall': f"{p_val['static_analysis']['recall']*100:.1f}% ({p_val['static_analysis']['vuln_found']}/16)",
                })
            st.dataframe(pd.DataFrame(p_rows), use_container_width=True, hide_index=True)
        
        # Display Prioritization Plot
        p_plot = PROJECT_ROOT / "results" / "plots" / "review_prioritization_comparison.png"
        l_plot = PROJECT_ROOT / "results" / "plots" / "controlled_benchmark_lineage.png"
        
        plot_col1, plot_col2 = st.columns(2)
        if p_plot.exists():
            with plot_col1:
                st.image(str(p_plot), caption="Review Prioritization: Recall vs Review Budget")
        if l_plot.exists():
            with plot_col2:
                st.image(str(l_plot), caption="Evolution Stage Dynamics: Findings & Risk across Lineage")
                
        st.markdown("---")
        st.markdown("### 📂 Interactive Task Explorer (8 Tasks)")
        
        task_names = [f"Task {i+1}: {t['task_id'].replace('task_', '').replace('_', ' ').title()}" for i, t in enumerate(bench_data['task_summaries'])]
        selected_task_idx = st.selectbox("Select Security Task to Inspect Lineage", range(len(task_names)), format_func=lambda i: task_names[i])
        
        selected_task = bench_data['task_summaries'][selected_task_idx]
        task_records = [r for r in bench_data['records'] if r['task_id'] == selected_task['task_id']]
        
        st.markdown(f"**Task ID:** `{selected_task['task_id']}` | **Target CWE:** `{task_records[0].get('target_cwe', 'N/A')}`")
        
        # Display 4 version cards
        for rec in task_records:
            v_id = rec['version_id']
            auth = rec['author_type']
            ai_inv = rec['ai_involvement']
            is_vuln = rec['ground_truth_vuln'] == 1
            status_badge = "🔴 Vulnerable" if is_vuln else "🟢 Secure"
            
            with st.expander(f"📄 {v_id} | Author: {auth} | AI: {ai_inv*100:.0f}% | {status_badge}"):
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.markdown("**Version Summary:**")
                    st.markdown(f"- **Author Type:** `{auth}`")
                    st.markdown(f"- **AI Involvement:** `{ai_inv*100:.0f}%`")
                    st.markdown(f"- **Ground Truth Label:** {status_badge}")
                with c2:
                    st.markdown("**Risk Scores & Predictions:**")
                    st.markdown(f"- **Static Findings:** {rec['n_findings']} issues (Score: {rec['risk_score']})")
                    st.markdown(f"- **CodeBERT Prob:** `{rec['codebert_vuln_prob']:.4f}`")
                    st.markdown(f"- **Provenance-Aware Score:** `{rec['prov_adjusted_score']:.4f}`")
                    if rec['detected_cwes']:
                        st.markdown(f"- **Detected CWEs:** `{', '.join(rec['detected_cwes'])}`")
    else:
        st.info("Click 'Run/Reload Multi-CWE Controlled Benchmark' to initialize the 32-sample benchmark.")


# ========================
# PAGE 5: Project Status
# ========================
elif page == "📋 Project Status":
    st.markdown("## 📋 Project Status — Mid-Semester Milestone")
    
    # Check what's actually implemented
    checks = {
        "Virtual environment": (PROJECT_ROOT / "venv").exists(),
        "Requirements file": (PROJECT_ROOT / "requirements.txt").exists(),
        "Dataset loader module": (PROJECT_ROOT / "src" / "data" / "dataset_loader.py").exists(),
        "Preprocessing module": (PROJECT_ROOT / "src" / "data" / "preprocessing.py").exists(),
        "Dataset splitting module": (PROJECT_ROOT / "src" / "data" / "split_dataset.py").exists(),
        "Baseline model module": (PROJECT_ROOT / "src" / "models" / "baseline.py").exists(),
        "CodeBERT model module": (PROJECT_ROOT / "src" / "models" / "code_encoder.py").exists(),
        "Evaluation module": (PROJECT_ROOT / "src" / "evaluation" / "evaluate.py").exists(),
        "Security analysis module": (PROJECT_ROOT / "src" / "security" / "static_analysis.py").exists(),
        "Controlled editing module": (PROJECT_ROOT / "src" / "experiments" / "controlled_editing.py").exists(),
        "Dataset preparation script": (PROJECT_ROOT / "scripts" / "prepare_dataset.py").exists(),
        "Baseline training script": (PROJECT_ROOT / "scripts" / "train_baseline.py").exists(),
        "CodeBERT training script": (PROJECT_ROOT / "scripts" / "train_codebert.py").exists(),
        "Streamlit application": (PROJECT_ROOT / "app.py").exists(),
        "Dataset reports": (PROJECT_ROOT / "data" / "reports").exists() and any((PROJECT_ROOT / "data" / "reports").glob("*.json")),
        "Processed data": (PROJECT_ROOT / "data" / "processed").exists() and any((PROJECT_ROOT / "data" / "processed").glob("*")),
        "Train/Val/Test splits": (PROJECT_ROOT / "data" / "splits").exists() and any((PROJECT_ROOT / "data" / "splits").glob("*.parquet")),
        "Trained baseline model": (PROJECT_ROOT / "models" / "checkpoints" / "baseline_char.pkl").exists(),
        "Evaluation metrics": (PROJECT_ROOT / "results" / "metrics").exists() and any((PROJECT_ROOT / "results" / "metrics").glob("*.json")),
        "Evaluation plots": (PROJECT_ROOT / "results" / "plots").exists() and any((PROJECT_ROOT / "results" / "plots").glob("*.png")),
        "Pilot experiment": (PROJECT_ROOT / "results" / "reports" / "pilot_experiment.json").exists(),
        "README": (PROJECT_ROOT / "README.md").exists(),
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Implemented Components")
        for item, status in checks.items():
            icon = "✅" if status else "⬜"
            st.markdown(f"{icon} {item}")
    
    completed = sum(1 for v in checks.values() if v)
    total = len(checks)
    
    with col2:
        st.markdown("### 📊 Progress")
        st.metric("Components Completed", f"{completed}/{total}")
        st.progress(completed / total)
        
        st.markdown("### 🎯 Mid-Semester Deliverables")
        st.markdown("""
        1. ✅ Research pipeline architecture
        2. ✅ Dataset loading & inspection
        3. ✅ Data preprocessing & standardization
        4. ✅ Reproducible train/val/test splits
        5. ✅ Baseline vulnerability detection model
        6. ✅ Evaluation metrics & visualization
        7. ✅ Security analysis module
        8. ✅ Controlled editing experiment framework
        9. ✅ Interactive Streamlit prototype
        10. ✅ Project documentation
        """)
        
        st.markdown("### 🔮 Remaining for Final Project")
        st.markdown("""
        - [ ] Full CodeBERT fine-tuning
        - [ ] Provenance-aware model training
        - [ ] Large-scale controlled editing experiment
        - [ ] Provenance vs. code-only comparison
        - [ ] Ablation studies
        - [ ] Unseen-generator generalization test
        - [ ] CWE category analysis
        - [ ] Vulnerability persistence study
        - [ ] Final research report
        """)
    
    st.markdown("---")
    st.markdown("### 🔧 Quick Start Commands")
    st.code("""
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Data preparation
python scripts/prepare_dataset.py

# Train baseline
python scripts/train_baseline.py

# Train CodeBERT (pilot)
python scripts/train_codebert.py --subset 500 --freeze_encoder --epochs 3

# Launch application
streamlit run app.py
    """, language="bash")
