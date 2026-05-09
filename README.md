# Transformer Failure Prediction System

## Overview

Modern transformer models are often highly confident even when incorrect.
This project builds an uncertainty-aware meta-learning system that predicts when a DistilBERT sentiment classifier is likely to fail under noisy or adversarial text conditions.

Instead of improving sentiment classification accuracy itself, the project focuses on a harder and more practical problem:

> Can we predict when the AI model should NOT be trusted?

The system combines:
- DistilBERT
- VADER
- TF-IDF + Logistic Regression
- Entropy-based uncertainty analysis
- Cross-model disagreement signals
- Adversarial robustness testing

to detect transformer instability before complete prediction failure.

---

# Why This Project Matters

Transformer models can silently fail while remaining extremely confident.

For example:

```text
amazing      → reliable
amazingg     → warning triggered
amazinnggg   → transformer failure
