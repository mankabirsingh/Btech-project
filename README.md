# Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code

**B.Tech Project-I — Delhi Technological University**

**Team:**
- Mankabir Singh (23/IT/199)
- Mohnish Gautam (23/IT/200)

---

## Research Question

> *Can provenance-aware analysis improve security-risk prediction and review prioritisation for human-AI hybrid code compared with provenance-agnostic vulnerability detection?*

## Project Overview

This project investigates whether information about code provenance — including AI involvement, AI generator identity, editing history, and human modifications — can improve vulnerability detection and security review prioritisation in codebases containing both human-written and AI-generated code.

### Key Components

1. **Vulnerability Detection Baseline**: TF-IDF + Logistic Regression on code text
2. **CodeBERT Fine-tuning**: Pre-trained code encoder for vulnerability classification
3. **Provenance-Aware Model**: Combines code representation with provenance features
4. **Controlled Editing Experiment**: Tracks security changes across code versions with different AI involvement levels
5. **Interactive Prototype**: Streamlit application for security analysis and experiment visualization

## Datasets

| Dataset | Source | Purpose | Samples |
|---------|--------|---------|---------|
| Devign | [DetectVul/devign](https://huggingface.co/datasets/DetectVul/devign) | Vulnerability detection (C/C++) | ~27K |
| PrimeVul | [adefu020/primevul](https://huggingface.co/datasets/adefu020/primevul) | Vulnerability detection with CWE/CVE labels | ~200K+ |
| HumanVsAICode | [OSS-forge/HumanVsAICode](https://huggingface.co/datasets/OSS-forge/HumanVsAICode) | Human vs AI code (ChatGPT, DeepSeek, Qwen) | ~507K |

### Important Notes
- Vulnerability labels come from Devign and PrimeVul (real-world open-source projects)
- HumanVsAICode provides provenance information but does NOT contain vulnerability labels
- Datasets are NOT merged blindly — vulnerability data and provenance data are maintained separately
- AI-generated ≠ vulnerable; human-written ≠ safe

## Project Structure

```
project/
├── app.py                          # Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore
├── src/
│   ├── data/
│   │   ├── dataset_loader.py       # Dataset loading utilities
│   │   ├── preprocessing.py        # Data standardization
│   │   └── split_dataset.py        # Train/val/test splitting
│   ├── models/
│   │   ├── baseline.py             # TF-IDF + LogReg baseline
│   │   └── code_encoder.py         # CodeBERT + Provenance-aware model
│   ├── evaluation/
│   │   └── evaluate.py             # Metrics, plots, model comparison
│   ├── security/
│   │   └── static_analysis.py      # Pattern-based security analysis
│   └── experiments/
│       └── controlled_editing.py   # Controlled editing experiment
├── scripts/
│   ├── prepare_dataset.py          # Download & prepare datasets
│   ├── train_baseline.py           # Train TF-IDF baseline
│   └── train_codebert.py           # Fine-tune CodeBERT
├── data/
│   ├── raw/                        # Raw downloaded data
│   ├── processed/                  # Standardized parquet files
│   ├── splits/                     # Train/val/test splits
│   └── reports/                    # Dataset inspection reports
├── models/
│   └── checkpoints/                # Saved model weights
├── results/
│   ├── metrics/                    # Evaluation metrics (JSON)
│   ├── plots/                      # Confusion matrices, ROC curves
│   └── reports/                    # Experiment reports
└── tests/
```

## Quick Start

### Prerequisites
- Python 3.9+
- macOS (Apple Silicon MPS) / Linux / Windows

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

```bash
python scripts/prepare_dataset.py
```

This downloads and processes:
- Devign vulnerability dataset
- PrimeVul vulnerability dataset (with CWE labels)
- HumanVsAICode provenance dataset

### Training

```bash
# Train TF-IDF + Logistic Regression baseline
python scripts/train_baseline.py

# Fine-tune CodeBERT (pilot experiment with subset)
python scripts/train_codebert.py --subset 500 --freeze_encoder --epochs 3

# Full CodeBERT fine-tuning (requires more time)
python scripts/train_codebert.py --epochs 5 --batch_size 8
```

### Launch Application

```bash
streamlit run app.py
```

The application provides:
1. **Security Analysis**: Paste code for vulnerability analysis + ML prediction
2. **Dataset Explorer**: Browse loaded datasets and distributions
3. **Model Results**: View evaluation metrics and comparison charts
4. **Provenance Experiment**: Controlled editing experiment visualization
5. **Project Status**: Implementation progress tracker

## Models

### Model A: TF-IDF + Logistic Regression (Baseline)
- Character 4-gram TF-IDF features (30K vocabulary)
- Balanced class weights for handling imbalanced data
- Fast training, interpretable baseline

### Model B: CodeBERT Classifier
- Pre-trained `microsoft/codebert-base` encoder
- Fine-tuned classification head
- Supports MPS (Apple Silicon), CUDA, and CPU

### Model C: Provenance-Aware Classifier
- CodeBERT code encoder + provenance feature MLP
- Fusion architecture combining code representation with:
  - AI involvement level
  - Provenance type (human/AI/mixed)
  - AI generator identity
  - Code evolution metadata
- Designed to test whether provenance improves vulnerability detection

## Evaluation

Metrics computed for each model:
- Accuracy, Precision, Recall, F1, F1-Macro, MCC
- AUROC, Average Precision
- Confusion Matrix
- Per-class classification report

## Current Limitations

1. **PrimeVul Dataset**: Large download; may require extended download time
2. **CodeBERT Fine-tuning**: Full fine-tuning requires significant compute; pilot experiments use subsets
3. **Provenance-Aware Model**: Currently trained only with controlled examples; large-scale provenance experiments planned for final project
4. **Vulnerability Labels**: Only Devign and PrimeVul have ground-truth labels; HumanVsAICode samples lack vulnerability annotations
5. **Static Analysis**: Pattern-based heuristics; not equivalent to formal verification

## Remaining Work (Final Project)

- [ ] Full CodeBERT fine-tuning on complete dataset
- [ ] Provenance-aware model training with HumanVsAICode
- [ ] Large-scale controlled editing experiment (0%–100% AI involvement)
- [ ] Provenance-aware vs provenance-agnostic comparison
- [ ] Ablation studies (feature importance)
- [ ] Unseen-generator generalization test
- [ ] CWE category evolution analysis
- [ ] Vulnerability persistence study across edit history
- [ ] Final research report with statistical analysis

## License

This is a research project for academic purposes at Delhi Technological University.
Datasets used are subject to their respective licenses.

## References

- Zhou et al., "Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks", NeurIPS 2019
- Feng et al., "CodeBERT: A Pre-Trained Model for Programming and Natural Languages", EMNLP 2020
- Ding et al., "PrimeVul: A Curated Benchmark for Vulnerability Detection", 2024
