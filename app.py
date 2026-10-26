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
    """Load trained baseline model."""
    model_path = PROJECT_ROOT / "models" / "checkpoints" / "baseline_char.pkl"
    if model_path.exists():
        from src.models.baseline import VulnerabilityBaseline
        return VulnerabilityBaseline.load(model_path)
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
            model = load_baseline_model()
            if model:
                st.markdown("---")
                st.markdown("**🤖 ML Model Prediction:**")
                pred = model.predict([code_input])[0]
                prob = model.predict_proba([code_input])[0]
                
                pred_label = "🔴 Vulnerable" if pred == 1 else "🟢 Benign"
                confidence = max(prob) * 100
                
                st.markdown(f"Prediction: **{pred_label}** ({confidence:.1f}% confidence)")
                st.progress(prob[1])
                st.caption("*ML prediction from TF-IDF + Logistic Regression baseline*")
            else:
                st.info("ML model not yet trained. Run `python scripts/train_baseline.py`")
            
            # Findings detail
            st.markdown("---")
            st.markdown("**Detailed Findings:**")
            
            for finding in analysis['findings']:
                sev_class = f"severity-{finding['severity']}"
                st.markdown(f"""
                <div class="finding-card" style="border-color: {'#ff4757' if finding['severity']=='critical' else '#ff6b6b' if finding['severity']=='high' else '#ffa502' if finding['severity']=='medium' else '#2ed573'};">
                    <strong class="{sev_class}">[{finding['severity'].upper()}]</strong> 
                    <code>{finding['cwe']}</code> — {finding['message']}<br>
                    <small>Line {finding.get('line', '?')} | Tool: {finding['tool']} | 
                    Confidence: {finding['confidence']}</small>
                    {f"<br><code>{finding.get('code_snippet', '')}</code>" if finding.get('code_snippet') else ""}
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
    st.markdown("## 🔬 Provenance-Aware Experiment")
    st.markdown("""
    This page demonstrates the controlled editing experiment framework.
    It shows how code versions with different levels of AI involvement
    are tracked and analyzed for security changes.
    """)
    
    # Load or create experiment
    experiment_path = PROJECT_ROOT / "results" / "reports" / "pilot_experiment.json"
    
    if st.button("🔄 Run/Reload Pilot Experiment"):
        from src.experiments.controlled_editing import (
            create_pilot_experiment, analyze_experiment_security
        )
        experiment = create_pilot_experiment()
        security_results = analyze_experiment_security(experiment)
        experiment.save(experiment_path)
        
        # Save security results
        with open(PROJECT_ROOT / "results" / "reports" / "pilot_security_results.json", 'w') as f:
            json.dump(security_results, f, indent=2)
        
        st.success("Pilot experiment executed!")
        st.rerun()
    
    if experiment_path.exists():
        with open(experiment_path) as f:
            exp_data = json.load(f)
        
        st.markdown(f"**Task:** {exp_data['task_description']}")
        st.markdown(f"**Language:** {exp_data['language']}")
        st.info(f"⚠️ {exp_data['metadata'].get('note', 'These are controlled examples.')}")
        
        # Version cards
        for version in exp_data['versions']:
            with st.expander(
                f"📄 {version['version_id']} | "
                f"Author: {version['author_type']} | "
                f"AI: {version['ai_involvement']*100:.0f}% | "
                f"Vuln: {'Yes' if version.get('vuln_label')==1 else 'No' if version.get('vuln_label')==0 else 'Unknown'}"
            ):
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.code(version['code'], language='c')
                
                with col2:
                    st.markdown(f"**Author Type:** {version['author_type']}")
                    st.markdown(f"**AI Involvement:** {version['ai_involvement']*100:.0f}%")
                    if version.get('ai_generator'):
                        st.markdown(f"**AI Generator:** {version['ai_generator']}")
                    if version.get('parent_version_id'):
                        st.markdown(f"**Parent:** {version['parent_version_id']}")
                    st.markdown(f"**Edit:** {version.get('edit_description', 'N/A')}")
                    
                    if version.get('security_findings'):
                        sf = version['security_findings']
                        st.markdown(f"**Security Findings:** {sf['n_findings']}")
                        st.markdown(f"**Risk Level:** {sf['risk_level']}")
                        if sf.get('cwes'):
                            st.markdown(f"**CWEs:** {', '.join(sf['cwes'])}")
        
        # Security comparison
        results_path = PROJECT_ROOT / "results" / "reports" / "pilot_security_results.json"
        if results_path.exists():
            with open(results_path) as f:
                sec_results = json.load(f)
            
            st.markdown("### Security Comparison Across Versions")
            
            comp_rows = []
            for vr in sec_results['version_results']:
                comp_rows.append({
                    'Version': vr['version_id'].replace('pilot_001_', ''),
                    'Author': vr['author_type'],
                    'AI %': f"{vr['ai_involvement']*100:.0f}%",
                    'Findings': vr['n_findings'],
                    'Risk': vr['risk_level'],
                    'CWEs': ', '.join(vr['cwes']) if vr['cwes'] else 'None',
                    'Known Vuln': '✅ Yes' if vr.get('known_vuln_label')==1 else '❌ No' if vr.get('known_vuln_label')==0 else '?',
                })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
            
            # Transitions
            if sec_results.get('transitions'):
                st.markdown("### Edit Transitions")
                for t in sec_results['transitions']:
                    direction = "✅ Fixed" if t.get('vuln_fixed') else ("🔴 Still Vulnerable" if t.get('vuln_persisted') else "➡️ Changed")
                    st.markdown(f"""
                    **{t['from_version'].replace('pilot_001_', '')}** → 
                    **{t['to_version'].replace('pilot_001_', '')}** | 
                    AI: {t['from_ai_involvement']*100:.0f}% → {t['to_ai_involvement']*100:.0f}% | 
                    Risk Change: {t['risk_change']:+d} | 
                    {direction}
                    """)
                    if t.get('cwes_fixed'):
                        st.markdown(f"  CWEs Fixed: {', '.join(t['cwes_fixed'])}")
                    if t.get('cwes_introduced'):
                        st.markdown(f"  CWEs Introduced: {', '.join(t['cwes_introduced'])}")
    else:
        st.info("Click 'Run/Reload Pilot Experiment' to generate controlled examples.")


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
