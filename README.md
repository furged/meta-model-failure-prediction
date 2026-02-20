# Meta-Learning for Transformer Failure Prediction

## Overview

Transformer models are often highly confident even when incorrect.
This project builds a meta-learning system that predicts when a fine-tuned
DistilBERT sentiment classifier is likely to fail.

The focus is not improving sentiment accuracy, but improving model reliability.

---

## Problem Statement

Deep learning models can be overconfident.
In real-world systems, detecting when a model may fail is often more valuable
than raw accuracy.

This project predicts transformer failure using:
- Confidence scores
- Cross-model disagreement
- Auxiliary model predictions

---

## Methodology

### Base Models
- VADER (rule-based sentiment)
- TF-IDF + Logistic Regression
- DistilBERT (SST-2 fine-tuned)

### Meta-Features
For each input:
- VADER prediction + confidence
- TF-IDF prediction + confidence
- Transformer confidence
- Disagreement signals

### Target
Binary label indicating whether the transformer prediction was incorrect.

### Meta-Classifier
Logistic Regression with class balancing.

---

## Dataset

- SST-2 (Stanford Sentiment Treebank)
- Meta-dataset constructed from 500 samples

---

## How to Run

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python predict.py --text "This movie was painfully boring"

#exmple output
=== Results ===
Transformer Prediction: Negative
Transformer Confidence: 0.9998
Failure Risk: LOW

