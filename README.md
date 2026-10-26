# Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code

**B.Tech Project-I — Delhi Technological University (DTU)**  
**Department of Information Technology**

**Team:**
- Mankabir Singh (Roll No: 23/IT/199)
- Mohnish Gautam (Roll No: 23/IT/200)

---

## 1. Research Question

> *"Can provenance-aware analysis improve security-risk prediction and review prioritisation for human-AI hybrid code compared with provenance-agnostic vulnerability detection?"*

---

## 2. Research Audit & Verified Findings Summary

This repository hosts a scientifically defensible research prototype evaluating how code provenance (author type, AI-involvement %, and edit lineage) affects security vulnerability prediction and code review prioritization.

### Key Scientific Findings
1. **Provenance in Legacy Open Source:** On traditional open-source benchmarks (Devign + PrimeVul, 36,080 samples), author provenance is homogeneous ($100\%$ human-authored, $\text{AI Involvement} = 0.0$). Adding provenance features yields a negligible delta ($\Delta \text{Acc} = +0.69\%$, $\Delta \text{F1} = +0.88\%$). Claiming provenance solves open-source vulnerability detection is empirically invalid.
2. **Review Prioritization in Hybrid Repositories:** In human-AI hybrid codebases, provenance awareness provides dramatic benefits for **code review triage and prioritization**:
   - At a **25% Review Budget**, Provenance-Aware Prioritization catches **50.0% of vulnerabilities (8/16)** with **100% precision**, compared to only **25.0% recall (4/16)** for code-only models.
   - At a **37.5% Review Budget**, recall reaches **75.0% (12/16)**.
3. **The Inductive Bias Caveat:** Because human-remediated hybrid code is secure ($\text{Vuln}=0$) despite containing AI history ($\text{AI}=50\%$), treating provenance as a naive binary vulnerability flag causes false positives. Provenance must act as a **prioritization weight**, guarded by contextual static analysis.

---

## 3. Verified Benchmark Results

### 3.1 Function-Level Vulnerability Detection (Grouped Held-Out Test Set: 5,412 Samples)
*Zero pair leakage: all vulnerable and fixed commit pairs are partitioned strictly into the same split.*

| Model Architecture | Accuracy | Precision | Recall | F1 Score | MCC | AUROC | PR-AUC (AP) | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Char N-gram Baseline** (TF-IDF 1-4 char, LR balanced) | **52.75%** | **51.31%** | **57.18%** | **54.09%** | **+0.0575** | **0.5474** | **0.5285** | 0.2481 |
| **Word N-gram Baseline** (TF-IDF 1-2 word, LR balanced) | 50.72% | 49.43% | 54.40% | 51.80% | +0.0164 | 0.5193 | 0.5021 | 0.2512 |
| **CodeBERT** (CLS + Val-Tuned Head) | 49.08% | 49.03% | 50.58% | 49.79% | -0.0183 | 0.5010 | 0.5097 | 0.2520 |

### 3.2 Review Prioritization (Multi-CWE Benchmark, 32 Units across 8 CWEs)

| Review Budget (% Inspected) | Code-Only Recall | Code-Only Precision | Provenance-Aware Recall | Provenance-Aware Precision |
| :---: | :---: | :---: | :---: | :---: |
| **12.5% (Top 4 PRs)** | 12.5% (2/16) | 50.0% | **25.0% (4/16)** | **100.0%** |
| **25.0% (Top 8 PRs)** | 25.0% (4/16) | 50.0% | **50.0% (8/16)** | **100.0%** |
| **37.5% (Top 12 PRs)** | 43.8% (7/16) | 58.3% | **75.0% (12/16)** | **100.0%** |
| **50.0% (Top 16 PRs)** | 56.2% (9/16) | 56.2% | **87.5% (14/16)** | **87.5%** |

---

## 4. Repository Structure & Audit Deliverables

```
project/
├── research_audit.md               # Complete repository audit & flaw resolution log
├── dataset_validation_report.md    # Dataset verification & PrimeVul correction report
├── leakage_audit_report.md         # Grouped splitting & zero pair leakage audit
├── provenance_experiment_report.md # Provenance-aware research contribution & prioritization
├── verified_midsemester_results.md # Consolidated empirical benchmarks & metrics
├── mid_semester_evaluation_report.md # Updated academic submission report
├── app.py                          # Streamlit research dashboard
├── requirements.txt                # Python dependencies
├── src/
│   ├── data/
│   │   ├── dataset_loader.py       # Robust dataset loaders
│   │   ├── preprocessing.py        # Code normalization & cleaning
│   │   └── split_dataset.py        # Stratified group splitting (zero leakage)
│   ├── models/
│   │   ├── baseline.py             # TF-IDF + Logistic Regression baselines
│   │   └── code_encoder.py         # CodeBERT encoder & zero-leakage provenance features
│   ├── evaluation/
│   │   └── evaluate.py             # Classification, calibration & ROC/PR generation
│   ├── security/
│   │   └── static_analysis.py      # Flawfinder + pattern analysis with remediations
│   └── experiments/
│       └── controlled_editing.py   # 8-task, 32-version Multi-CWE evolution benchmark
├── scripts/
│   ├── prepare_dataset.py          # Preprocesses Devign, PrimeVul & HumanVsAICode
│   ├── train_baseline.py           # Trains Char and Word N-gram baselines
│   ├── train_codebert.py           # Extracts CodeBERT embeddings on MPS & tunes head
│   ├── run_controlled_experiment.py# Runs 32-version Multi-CWE benchmark & prioritization
│   └── train_provenance_experiment.py # Evaluates Model A vs Model B on real & hybrid data
├── models/checkpoints/             # Trained models (.pkl, .pt)
├── results/
│   ├── metrics/                    # JSON evaluation outputs
│   ├── plots/                      # Confusion matrices, ROC curves, PR curves, lineages
│   └── reports/                    # Experiment summaries
└── tests/
    └── test_pipeline.py            # Automated test suite (12/12 passing)
```

---

## 5. Quick Start & Exact Reproduction Commands

All commands are validated for local Apple Silicon (MPS) and CPU environments:

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Run automated test suite (12 tests)
python -m unittest tests/test_pipeline.py -v

# 3. Train baselines on un-leaked grouped splits (36,080 samples)
python scripts/train_baseline.py

# 4. Train CodeBERT classifier on Apple Silicon MPS with validation threshold tuning
python scripts/train_codebert.py --train_samples 3000 --val_samples 800 --test_samples 1200 --batch_size 32

# 5. Run the Multi-CWE Controlled Evolution Benchmark & Prioritization
python scripts/run_controlled_experiment.py

# 6. Run the Provenance-Aware Research Comparison (Model A vs Model B)
python scripts/train_provenance_experiment.py

# 7. Launch Streamlit Research Dashboard
streamlit run app.py
```

---

## 6. Security Analysis Engine

The system supports interactive code submission and returns:
1. **ML Model Predictions**: Binary classification and probability from trained models.
2. **Flawfinder Static Analysis**: Real C/C++ lexical scanner detecting buffer overflows, format strings, and race conditions.
3. **Pattern-Based Heuristics**: Comment-stripped pattern matching for CWE-120, CWE-134, CWE-190, CWE-78, CWE-416, CWE-22, CWE-327, CWE-476.
4. **Remediation Guidance**: Specific recommendations (e.g. replacing `gets` with `fgets`, using `snprintf`, canonicalizing paths with `realpath`).

---

## 7. Faculty Presentation Talking Points (5 Minutes)

1. **The Problem:** AI code generators (Copilot, ChatGPT) write plausible code that often introduces subtle security vulnerabilities. When developers merge AI suggestions with human edits, security reviewers face alert fatigue.
2. **The Core Question:** Does knowing code provenance (who wrote what, AI involvement %, and lineage) help predict vulnerabilities and prioritize security reviews?
3. **The Audit:** We discovered that earlier models collapsed into majority-class prediction (98% recall, 54% accuracy) due to uncalibrated linear heads, and random splits scattered commit pairs across train and test. We fixed the data pipeline using **Stratified Group Splitting** over 16,709 commit groups with **zero pair leakage**.
4. **The Empirical Truth:** 
   - On legacy open source, provenance provides no gain because all code is human-written.
   - On human-AI hybrid code, provenance-aware review prioritization **doubles triage recall** from 25% to 50% under a 25% review budget.
5. **The Defense:** We do not claim AI code is automatically vulnerable or human code is safe. Instead, provenance is an effective prioritization heuristic that guides human security audits to where bugs statistically cluster.
