<div align="center">

<img src="https://img.shields.io/badge/Status-Published-brightgreen?style=for-the-badge" />
<img src="https://img.shields.io/badge/Conference-Springer_LNCS-blue?style=for-the-badge" />
<img src="https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python" />
<img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch" />
<img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface" />

<br/><br/>

# 🧠 Multi-Dataset Conditional Text Generation for Emotion Augmentation in Sentiment Classification

### *A student research paper published in Springer LNCS*

<br/>

### 👨‍💻 Built & Implemented by [Hitan K](https://github.com/hitank2004)
> *AI/ML Developer · Student Researcher · Department of CSE & AI, DSATM Bangalore*

<br/>

*Co-authored with: Priyanka R, Adrian Patrick, Dinesha N, Ajay Krishna*

<br/>

> **Abstract:** We propose a unified pipeline that addresses class imbalance and label taxonomy heterogeneity across emotion datasets by combining emotion-conditioned text generation (SmolLM-135M-Instruct) with transformer-based classification (RoBERTa + Optuna). Evaluated on GoEmotions, TweetEval, and DAIR Emotion, achieving macro-F1 scores of **0.83**, **0.79**, and **0.97** respectively — with consistent gains on minority emotion classes.

<br/>

[📄 Read the Paper](#-paper) · [🚀 Quick Start](#-quick-start) · [📊 Results](#-results) · [🔬 Methodology](#-methodology) · [📁 Repo Structure](#-repo-structure)

</div>

---

## 👨‍💻 About the Developer

**Hitan K** is an AI/ML Developer and Student Researcher at Dayananda Sagar Academy of Technology and Management (DSATM), Bangalore, specializing in Generative AI, NLP, and Intelligent Systems.

On this project, Hitan was responsible for:
- 🏗️ Designing and implementing the **end-to-end pipeline** across all three datasets
- 🤖 Fine-tuning **SmolLM-135M-Instruct** for emotion-conditioned text generation
- 📊 Running **RoBERTa + Optuna** experiments and hyperparameter search
- 📓 Writing all **Jupyter notebooks** for full reproducibility
- 📈 Generating all evaluation metrics, confusion matrices, and analysis

| | |
|---|---|
| 🎓 | B.Tech CSE & AI — DSATM Bangalore |
| 🔬 | Published Researcher — Springer LNCS |
| 🛠️ | Generative AI · NLP · Transformers · PyTorch · HuggingFace |
| 📬 | hitank2004@gmail.com |
| 🐙 | [github.com/hitank2004](https://github.com/hitank2004) |

---

## 📌 Table of Contents

- [Motivation](#-motivation)
- [Key Contributions](#-key-contributions)
- [Methodology](#-methodology)
- [Results](#-results)
- [Repo Structure](#-repo-structure)
- [Quick Start](#-quick-start)
- [Datasets](#-datasets)
- [Paper](#-paper)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)

---

## 💡 Motivation

Emotion and sentiment classifiers suffer from three systemic problems that degrade real-world performance:

| Problem | Impact |
|---|---|
| **Severe class imbalance** | Models overfit to dominant classes (joy, neutral) and fail on rare ones (grief, fear, relief) |
| **Heterogeneous label taxonomies** | Hard to generalize across datasets with different emotion ontologies |
| **Sparse minority-class data** | Classical augmentation (back-translation, synonym replacement) distorts emotional context |

Classical solutions like EDA, SMOTE, and back-translation treat augmentation as a lexical problem. We treat it as a **semantic generation** problem — conditioning a language model explicitly on emotion labels to synthesize diverse, affect-faithful minority samples.

---

## 🏆 Key Contributions

- ✅ **Unified multi-dataset pipeline** — single framework evaluated across TweetEval, GoEmotions, and DAIR Emotion without dataset-specific hacks
- ✅ **Emotion-conditioned text generation** — fine-tuned SmolLM-135M-Instruct with explicit control tokens (`[JOY]`, `[FEAR]`, `[POSITIVE]`, etc.) for label-faithful synthesis
- ✅ **Minority-class targeted augmentation** — generates samples only for underrepresented classes, preserving majority-class distribution
- ✅ **Optuna-driven hyperparameter optimization** — automated search over learning rate and epoch space on validation macro-F1
- ✅ **Resource-efficient** — full pipeline runs on a single RTX 4050 via Google Colab with FP16 mixed-precision training

---

## 🔬 Methodology

The pipeline is divided into four stages:

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: Data Preprocessing                                    │
│  Load → Clean → Add Control Tokens → Stratified Split           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│  STAGE 2: Emotion-Conditioned Generation (SmolLM-135M-Instruct) │
│  Fine-tune on balanced seed data → Generate minority samples    │
│  → Quality filter → Merge with original training set            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│  STAGE 3: RoBERTa Classification (with Optuna)                  │
│  Train on augmented data → Optuna LR search → Best checkpoint   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│  STAGE 4: Evaluation                                            │
│  Macro-F1 · Weighted-F1 · Minority-class recall · Confusion     │
└─────────────────────────────────────────────────────────────────┘
```

### Generation Model — SmolLM-135M-Instruct

| Parameter | Value |
|---|---|
| Objective | Conditional Causal Language Modeling |
| Learning Rate | 2 × 10⁻⁴ (Linear Decay) |
| Effective Batch Size | 64 |
| Epochs | 6 |
| Max Sequence Length | 128 tokens |
| Precision | Mixed (FP16) |
| Control Tokens | `[POSITIVE]`, `[NEGATIVE]`, `[NEUTRAL]`, `[JOY]`, `[FEAR]`, `[SADNESS]`, ... |

**Generation hyperparameters:** `temperature=0.75`, `top_k=40`, `top_p=0.9`, `repetition_penalty=1.3`, `no_repeat_ngram_size=3`

### Classifier — RoBERTa + Optuna

| Parameter | Value |
|---|---|
| Base Model | `roberta-base` |
| Learning Rate | 5e-5 (Optuna range: 1e-5 → 5e-5) |
| Gradient Accumulation | 2 steps (effective batch size: 64) |
| Warmup Ratio | 0.1 |
| Weight Decay | 0.01 |
| Optimizer | AdamW |
| Best TweetEval Config | LR: 2.03e-05, 1 epoch → F1: 0.790 |

---

## 📊 Results

### TweetEval (3-class Sentiment)

| Method | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Baseline (No Augmentation) | 0.72 | 0.71 | 0.72 |
| Baseline (Random Oversampling) | 0.72 | 0.72 | 0.72 |
| **Ours (Conditional Generation)** | **0.79** | **0.79** | **0.79** |

*SmolLM Training — Loss: 3.08 → 4.32 eval, Perplexity: 75.10*

---

### DAIR Emotion (6-class)

| Method | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Baseline (No Augmentation) | 0.93 | 0.90 | 0.94 |
| Baseline (Random Oversampling) | 0.93 | 0.91 | 0.93 |
| **Ours (Conditional Generation)** | **0.97** | **0.97** | **0.97** |

*SmolLM Training — Loss: 1.89 → 3.97 eval, Perplexity: 52.91*

---

### GoEmotions (27-class Fine-Grained)

| Method | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Baseline (No Augmentation) | 0.58 | 0.51 | 0.57 |
| Baseline (23-Emotion Model) | 0.45 | 0.46 | 0.43 |
| **Ours (Conditional Generation)** | **0.83** | **0.83** | **0.83** |

*SmolLM Training — Loss: 2.62 → 4.15 eval, Perplexity: 63.51*

---

> **Key finding:** The largest gains appear consistently in **minority emotion classes** — fear and surprise in DAIR Emotion, relief/grief/admiration in GoEmotions, and the ambiguous neutral class in TweetEval. Augmentation reduces inter-class confusion where semantically adjacent emotions (e.g., love↔joy, disgust↔anger) are most likely to be conflated.

---

## 📁 Repo Structure

```
.
├── tweeteval/
│   ├── tweetevalfinal.ipynb        # Preprocessing + generation + classification
│   └── data/                       # Augmented split cache (auto-generated)
│
├── goemotion/
│   ├── goemotionfinal.ipynb
│   └── data/
│
├── dairemotion/
│   ├── dairemotionfinal.ipynb
│   └── data/
│
├── paper/
│   └── PID-130_research_paper_springer.pdf   # Published paper
│
└── README.md
```

Each dataset folder is self-contained — run its notebook end-to-end to reproduce generation, augmentation, and classification.

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install transformers datasets torch accelerate tokenizers optuna scikit-learn matplotlib seaborn
```

GPU with at least 6GB VRAM recommended (tested on RTX 4050 via Google Colab).

### Run on TweetEval

```python
# 1. Open tweeteval/tweetevalfinal.ipynb in Colab or Jupyter
# 2. Ensure GPU runtime is enabled
# 3. Run all cells sequentially

# The pipeline will:
# - Load cardiffnlp/tweet_eval (sentiment)
# - Fine-tune SmolLM-135M-Instruct with [POSITIVE]/[NEGATIVE]/[NEUTRAL] tokens
# - Generate synthetic minority samples
# - Fine-tune RoBERTa with Optuna
# - Output: confusion matrix, macro-F1, per-class recall
```

### Run on DAIR Emotion

```python
# Open dairemotion/dairemotionfinal.ipynb
# Control tokens: [SADNESS], [JOY], [LOVE], [ANGER], [FEAR], [SURPRISE]
```

### Run on GoEmotions

```python
# Open goemotion/goemotionfinal.ipynb
# 27 fine-grained emotion control tokens
# Note: GoEmotions is multi-label; single-label mode used here for controlled evaluation
```

---

## 📦 Datasets

| Dataset | Source | Classes | Size | Domain |
|---|---|---|---|---|
| [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) | Barbieri et al., 2020 | 3 (Neg/Neu/Pos) | ~59K tweets | Twitter |
| [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | Demszky et al., 2020 | 27 + neutral | 58,011 Reddit comments | Reddit |
| [DAIR Emotion](https://huggingface.co/datasets/dair-ai/emotion) | Saravia et al., 2018 | 6 | 20,000 sentences | Social media |

All datasets load directly from HuggingFace Hub — no manual download needed.

---

## 📄 Paper

<div align="center">

[![Read Paper](https://img.shields.io/badge/📄_Read_the_Paper-Springer_LNCS-blue?style=for-the-badge)](https://github.com/Hitan547/-project-conditional-text-generation-and-augmentation-for-sentiment-classification/blob/main/PID-130_research_paper_springer.pdf)

</div>

## 📝 Want to Use This Work?

> This project is part of a published research paper. If you'd like to use, reference, or collaborate on this work — **reach out directly.**

<div align="center">

### 📬 Contact Hitan K

[![Email](https://img.shields.io/badge/Email-hitank2004%40gmail.com-red?style=for-the-badge&logo=gmail)](mailto:hitank2004@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-hitank2004-black?style=for-the-badge&logo=github)](https://github.com/hitank2004)

*Open to collaborations, research discussions, and opportunities.*

</div>
---

## 🙏 Acknowledgements

- Dataset creators: [GoEmotions (Google Research)](https://github.com/google-research/google-research/tree/master/goemotions), [TweetEval (Cardiff NLP)](https://github.com/cardiffnlp/tweeteval), [DAIR Emotion](https://huggingface.co/datasets/dair-ai/emotion)
- Model authors: [SmolLM-135M-Instruct (HuggingFace)](https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct), [RoBERTa (Liu et al., 2019)](https://arxiv.org/abs/1907.11692)
- Compute: Google Colab (RTX 4050 GPU)
- Hyperparameter optimization: [Optuna (Akiba et al., 2019)](https://optuna.org/)

---

<div align="center">

Built by **[Hitan K](https://github.com/hitank2004)** · DSATM Bangalore

*If this work helped you, please ⭐ the repo!*

</div>
