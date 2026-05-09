# Transformer Failure Prediction System

## AI Reliability Through Meta-Learning, Uncertainty Estimation, and Adversarial Robustness

---

# Overview

Modern transformer models are often highly confident even when incorrect.
In real-world AI systems, this becomes dangerous because users may trust predictions simply because the model sounds certain.

This project explores a different question from traditional sentiment analysis:

> Can we predict when a transformer model itself is likely to fail?

Instead of only building a sentiment classifier, this project builds a second-stage meta-learning system that analyzes transformer behavior and predicts reliability under noisy or adversarial text conditions.

The system combines:

* DistilBERT
* VADER
* TF-IDF + Logistic Regression
* Entropy-based uncertainty estimation
* Cross-model disagreement analysis
* Adversarial robustness testing
* Meta-learning for transformer failure prediction

The focus is not maximizing sentiment accuracy.
The focus is understanding when the AI model should NOT be trusted.

---

# Motivation

Large language models and transformer architectures often produce highly confident predictions even when the input becomes noisy, ambiguous, or adversarial.

Small human-readable perturbations can destabilize predictions significantly.

For example:

```text
amazing      → reliable
amazingg     → warning triggered
amazinnggg   → transformer failure
```

Humans still understand the meaning instantly.
However, the transformer progressively becomes unstable.

This project studies that instability.

The system attempts to detect transformer failures BEFORE complete prediction collapse occurs.

---

# Core Idea

The main idea behind the project is:

1. Use multiple sentiment systems simultaneously.
2. Observe their behavior on the same input.
3. Measure uncertainty and disagreement.
4. Train a second AI model to predict whether the transformer is likely to fail.

Instead of blindly trusting DistilBERT predictions, the meta-model acts like a reliability monitor.

---

# System Architecture

## Stage 1 — Base Models

The input text is passed through three independent sentiment systems.

### 1. DistilBERT

Fine-tuned SST-2 transformer model.
Acts as the primary deep learning model.

### 2. VADER

Rule-based sentiment analyzer.
Useful for lexical sentiment consistency.

### 3. TF-IDF + Logistic Regression

Classical machine learning baseline.
Provides an interpretable non-transformer comparison signal.

---

# Meta-Feature Engineering

For every input sample, the system extracts multiple reliability signals.

## Confidence Signals

* DistilBERT confidence
* Logistic Regression confidence
* VADER polarity magnitude

## Uncertainty Signals

* Transformer entropy

## Disagreement Signals

* VADER vs Logistic Regression disagreement
* Logistic Regression vs DistilBERT disagreement
* VADER vs DistilBERT disagreement

## Prediction Signals

* Raw predictions from all models

These features are combined into a structured meta-dataset.

---

# Meta-Model

A second-stage Logistic Regression meta-classifier is trained to predict:

```text
Will the transformer prediction fail?
```

The target label becomes:

```text
1 → Transformer incorrect
0 → Transformer correct
```

Because transformer failures are rare, severe class imbalance handling was required.

The project uses:

* class-weighted learning
* adversarial augmentation
* failure-focused evaluation

---

# Dataset

## Primary Dataset

* SST-2 (Stanford Sentiment Treebank)

## Meta-Dataset

* 5,000 adversarially perturbed samples

The dataset includes:

* spelling corruption
* punctuation perturbation
* casing variation
* repeated characters
* noisy text transformations
* adversarial-style perturbations

Examples:

```text
amazing → amazingg
boring → booooring
terrible → TERRIBLEEEE
```

---

# Adversarial Robustness Analysis

One major focus of the project is studying transformer robustness under noisy text conditions.

The system demonstrates:

* progressive transformer degradation
* confidence instability
* entropy growth under corruption
* disagreement escalation between models
* high-confidence failures

Example progression:

| Input      | Transformer Behavior |
| ---------- | -------------------- |
| amazing    | Reliable             |
| amazingg   | Warning Triggered    |
| amazinnggg | Failure Detected     |

This behavior became one of the strongest experimental findings in the project.

---

# Key Findings

## 1. Entropy Was the Strongest Failure Signal

Transformer entropy consistently emerged as the most important predictor of instability.

Higher entropy strongly correlated with incorrect transformer predictions.

---

## 2. Confidence Alone Was Insufficient

The transformer frequently produced:

```text
high confidence + incorrect prediction
```

This demonstrated calibration issues in transformer behavior.

---

## 3. Cross-Model Disagreement Improved Reliability Detection

When VADER, Logistic Regression, and DistilBERT disagreed strongly, transformer failure probability increased significantly.

---

## 4. Tiny Perturbations Could Destabilize Predictions

Very small human-readable spelling corruptions caused major prediction instability.

This highlighted the fragility of transformer tokenization behavior.

---

# Example Failure Case

```text
INPUT:
that was so amazinnggg

TRUE LABEL:
Positive

DISTILBERT OUTPUT:
Negative

BERT CONFIDENCE:
0.8547

BERT ENTROPY:
0.4162

FAILURE PROBABILITY:
0.8601

META-MODEL WARNING:
TRANSFORMER INSTABILITY DETECTED
```

---

# Performance Metrics

## Base Logistic Regression

| Metric              | Value |
| ------------------- | ----- |
| TF-IDF Features     | 5,000 |
| N-Grams             | 1–2   |
| Validation Accuracy | 84.4% |
| ROC-AUC             | 0.93  |
| PR-AUC              | 0.946 |

---

## Meta-Model

| Metric                          | Value   |
| ------------------------------- | ------- |
| Failure Recall                  | 0.87    |
| Severe Class Imbalance Handling | Yes     |
| Entropy Feature Importance      | Highest |

---

# Calibration Analysis

The project also studies transformer calibration.

A calibration analysis was performed to compare:

```text
model confidence vs actual correctness
```

Results showed:

* higher confidence generally improved reliability
* but high-confidence failures still existed
* confidence alone could not fully guarantee correctness

This reinforced the need for:

* entropy analysis
* disagreement signals
* meta-learning-based reliability estimation

---

# Failure Visualization System

The project includes dedicated evaluation scripts for:

* confusion matrix generation
* entropy visualization
* calibration analysis
* failure case extraction
* adversarial robustness testing
* feature importance analysis

The system can automatically surface examples where:

```text
Transformer confidence is high
BUT
prediction is incorrect
```

---

# Interactive Dashboard

The project includes a Flask-based interactive AI diagnostic dashboard.

Features include:

* Real-time sentiment analysis
* Reliability prediction
* Transformer entropy display
* Failure probability estimation
* Cross-model comparison
* Adversarial testing interface
* Retro CRT-style monitoring UI
* Interactive diagnostic tabs
* Failure case database
* Project metrics visualization

The interface was intentionally designed as a retro AI monitoring terminal rather than a traditional modern dashboard.

---

# Example Usage

## Run Locally

```bash
python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

# Example Inference Output

```text
INPUT TEXT:
not into a strangely tempting bouquet of a movie

VADER PREDICTION: 1
LR PREDICTION: 0
BERT PREDICTION: 0

BERT CONFIDENCE: 0.9341
BERT ENTROPY: 0.1739

FAILURE PROBABILITY: 0.5475

WARNING:
TRANSFORMER MAY FAIL
```

---

# Project Structure

```text
meta-model-failure-prediction/
│
├── artifacts/
│   ├── lr_model.pkl
│   ├── meta_model.pkl
│   └── tfidf.pkl
│
├── src/
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   └── predict.py
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# Technologies Used

## Machine Learning

* Scikit-learn
* Transformers
* PyTorch
* NLTK

## Data Processing

* Pandas
* NumPy

## Visualization

* Matplotlib

## Backend

* Flask

## Frontend

* HTML
* CSS
* JavaScript

---

# Challenges Faced

## 1. Severe Class Imbalance

Transformer failures were rare, making reliability prediction difficult.

## 2. Data Alignment Bugs

Multiple debugging stages were required to correctly align transformer predictions with meta-labels.

## 3. Adversarial Robustness Testing

Small perturbations sometimes caused unexpectedly large prediction instability.

## 4. Feature Engineering

Finding meaningful reliability signals required multiple iterations.

---

# Future Improvements

* Multi-dataset generalization testing
* Better adversarial attack generation
* SHAP/LIME interpretability
* Transformer ensemble reliability analysis
* Real-time logging system
* GPU batching optimization
* Hugging Face deployment
* Reliability threshold tuning
* Advanced uncertainty estimation methods

---

# Conclusion

This project demonstrates that:

```text
Model confidence alone is not enough.
```

Reliable AI systems require:

* uncertainty estimation
* disagreement analysis
* adversarial robustness evaluation
* explicit failure prediction mechanisms

The project reframes sentiment analysis as an AI reliability problem rather than only a classification problem.

---

# Author

Anushka Shakya
B.Tech CSE (AI)
