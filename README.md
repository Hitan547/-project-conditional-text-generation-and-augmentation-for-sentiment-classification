# 🤖 Multi-Dataset Conditional Text Generation & Augmentation for Sentiment & Emotion Classification

This project implements a unified pipeline across three benchmark datasets—**TweetEval**, **GoEmotions**, and **DAIR Emotion**—to perform **conditional text generation** using *SmolLM-135M* for data augmentation and improve classification performance, especially for minority classes.

---

## 🚀 Project Overview

### 📊 Datasets

* **TweetEval** → 3-class sentiment classification *(Negative, Neutral, Positive)*
* **GoEmotions** → 27 fine-grained emotion classes
* **DAIR Emotion** → 6 core emotions *(sadness, joy, love, anger, fear, surprise)*

---

## ⚙️ Preprocessing

* Lowercasing text
* Removing URLs, mentions, hashtags, emojis, and noise
* Adding **control tokens** (e.g., `[POSITIVE]`, `[JOY]`, `[SADNESS]`)
* Tokenization with:

  * Fixed max length
  * Padding
  * Attention masks

---

## 🧠 Conditional Generation (Data Augmentation)

* Fine-tuned **SmolLM-135M-Instruct** on each dataset
* Used **control tokens** for guided generation
* Generated synthetic samples for **minority classes**
* Applied diverse prompt strategies to improve variety
* Achieved **balanced class distributions**

---

## 📈 Classifier Training

* Used **RoBERTa-based models** for classification
* Applied **Optuna** for hyperparameter tuning:

  * Learning rate
  * Batch size
  * Epochs
* Implemented:

  * Mixed precision training
  * Early stopping based on validation performance

---

## 📊 Evaluation Metrics

### Generation Models

* Training Loss
* Validation Loss
* Perplexity

### Classification Models

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

👉 Special focus on **minority-class improvements**

---

## 🔄 Pipeline Workflow

### 1. Data Loading & Cleaning

* Load datasets from Hugging Face
* Apply preprocessing & label mapping
* Visualize class imbalance

### 2. Conditional Generation & Augmentation

* Fine-tune generator models
* Generate synthetic minority samples
* Merge with original data

### 3. Classifier Fine-Tuning

* Train models on:

  * Original dataset
  * Augmented dataset
* Optimize using Optuna
* Select best model via validation F1

### 4. Evaluation & Analysis

* Compare baseline vs augmented performance
* Analyze confusion matrices
* Highlight improvements in minority classes

---

## 📊 Key Results

### TweetEval

* Original → Accuracy: **~0.72**, Macro F1: **~0.71**
* Augmented → Accuracy: **~0.79**, Macro F1: **~0.79**

### GoEmotions (Single-label)

* Original → Accuracy: **~0.58**, Macro F1: **~0.51**
* Augmented → Accuracy: **~0.83**, Macro F1: **~0.83**

### DAIR Emotion

* Original → Accuracy: **~0.93**, Macro F1: **~0.90**
* Augmented → Accuracy: **~0.97**, Macro F1: **~0.97**

✅ Significant improvements in **minority-class recall and F1-score**

---

## 📁 Repository Structure

```
dairemotion/   → DAIR Emotion pipeline
goemotion/     → GoEmotions pipeline
tweeteval/     → TweetEval pipeline
```

Each folder includes:

* Jupyter notebooks
* Data preprocessing
* Generation pipeline
* Classification workflow

---

## 🧠 Key Highlights

* Multi-dataset unified pipeline
* Conditional text generation for augmentation
* Hybrid approach combining **LLM + classical ML pipeline**
* Strong improvement in class imbalance handling
* End-to-end experimentation and evaluation

---

## 👨‍💻 Author

**Hitan K**
AI/ML Developer | Focused on Generative AI, NLP, and Intelligent Systems

---

## 📬 Contact

For queries, collaborations, or improvements:
👉 Open an issue or submit a pull request
