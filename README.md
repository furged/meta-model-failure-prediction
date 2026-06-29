# Sentinel AI — Transformer Failure Prediction

**A meta-model that watches DistilBERT-SST2 and predicts when it's about to be wrong.**

[![Docker](https://img.shields.io/badge/deployment-docker-blue)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/python-3.9+-green)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Spaces-yellow)](https://huggingface.co/spaces)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Code Style](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)

---

## About

Sentinel AI is an intelligent monitoring system that acts as a safety net for sentiment analysis models. Instead of blindly trusting a base model's predictions, Sentinel AI monitors DistilBERT-SST2 in real-time, predicts when the model is likely to make an error, flags uncertain or potentially incorrect outputs, and provides confidence scores and failure probabilities. This is particularly useful for production environments where model reliability is critical, such as customer feedback analysis, content moderation, or financial sentiment tracking.

---

## Key Features

- Failure Prediction - Predicts when the base model will make an incorrect prediction
- Meta-Model Architecture - Learns patterns from DistilBERT's internal states and outputs
- 18,000+ Sample Dataset - Trained on an expanded, diverse dataset for robust performance
- Feature Engineering - Includes specialized features like negative-intensifier detection
- Web Interface - Interactive UI with prediction history, theme toggle, and more
- Docker Support - Easy deployment with containerization
- HuggingFace Ready - Configured for Spaces deployment

---

## Architecture

### System Overview

The system consists of three primary components working in sequence:

**Component 1: Base Model Layer**
- DistilBERT model fine-tuned on SST-2 dataset for sentiment analysis
- Receives input text and produces a sentiment prediction (positive/negative)
- Outputs a confidence score between 0.0 and 1.0
- Processes 512 tokens maximum per input

**Component 2: Feature Extraction Pipeline**
- Captures 14 distinct features from the base model's inference
- Features include: confidence score, entropy of logits, prediction stability, and linguistic patterns
- Negative-intensifier detection identifies words like "very", "extremely", "absolutely"
- Ambiguity score calculation for inputs with neutral or conflicting signals
- Feature normalization and scaling for meta-model consumption

**Component 3: Meta-Model (Sentinel)**
- Lightweight classifier trained on 18,000+ labeled examples
- Input: 14-dimensional feature vector from the extraction pipeline
- Output: Failure probability score (0.0 to 1.0)
- Threshold: Scores above 0.65 trigger a "likely failure" alert
- Uses XGBoost with 100 estimators and maximum depth of 6

### Data Flow
Input Text
    ↓
DistilBERT-SST2 (Base Model)
    ↓
[Prediction: Positive/Negative] + [Confidence: 0.92]
    ↓
Feature Extraction
    ↓
[Confidence, Entropy, Stability, Intensifier_Count, Ambiguity_Score, ...]
    ↓
Meta-Model (XGBoost Classifier)
    ↓
Failure Probability: 0.73
    ↓
Alert: "High risk of incorrect prediction"


### Model Details

**Base Model Specifications:**
- Architecture: DistilBERT-base-uncased
- Fine-tuning dataset: SST-2 (67,349 training samples)
- Input length: 512 tokens
- Output classes: 2 (negative/positive)
- Validation accuracy: 91.3%
- Inference time: 0.04 seconds per sample

**Meta-Model Specifications:**
- Algorithm: XGBoost Classifier
- Training samples: 18,247
- Features: 14 numerical features
- Validation accuracy: 84.7%
- Failure detection rate: 78.2%
- False positive rate: 12.3%
- Inference time: 0.002 seconds per sample

**Feature Engineering Details:**
1. Model confidence score (0-1)
2. Logit entropy (0-2)
3. Prediction stability across 5 inference runs (standard deviation)
4. Negative intensifier presence (binary)
5. Intensifier count (0-5)
6. Sentence length (tokens)
7. Sentiment ambiguity score (0-1)
8. Negation presence (binary)
9. Adversarial token count
10. Out-of-distribution score (0-1)
11. Prediction margin (difference between top two class probabilities)
12. Maximum softmax probability
13. Second highest softmax probability
14. Feature interaction terms (confidence × ambiguity)

### Performance Metrics

| Metric | Value |
|--------|-------|
| Base Model Accuracy | 91.3% |
| Meta-Model Accuracy | 84.7% |
| Failure Detection Rate | 78.2% |
| False Positive Rate | 12.3% |
| Precision | 79.1% |
| Recall | 78.2% |
| F1 Score | 78.6% |
| AUC-ROC | 0.874 |
| Inference Time (Base) | 0.04 sec |
| Inference Time (Meta) | 0.002 sec |
| Total Pipeline Time | 0.042 sec |
| Training Time | 14.3 min |

---

## Quick Start

### Option 1: Local Deployment

1. Clone the repository
```bash
git clone https://github.com/furged/meta-model-failure-prediction.git
cd meta-model-failure-prediction
```

2. Create a virtual environment
```bash
python -m venv venv
```

3. Activate the virtual environment
```bash
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

4. Install dependencies
```bash
pip install -r requirements.txt
```

5. Run the application
```bash
python app.py
```

6. Open your browser and navigate to `http://localhost:5000`

### Option 2: Docker Deployment

```bash
docker build -t sentinel-ai .
docker run -p 5000:5000 sentinel-ai
```

### Option 3: HuggingFace Spaces

The project is configured for deployment on HuggingFace Spaces. Deployment is currently under process and will be available shortly at the HuggingFace Spaces URL. The Docker configuration and Procfile are ready for seamless deployment once the Space is created.

---

## Requirements

- `Python 3.9+`
- `Flask 2.0+`
- `Transformers 4.30+`
- `PyTorch 2.0+`
- `scikit-learn 1.2+`
- `pandas 2.0+`
- `numpy 1.24+`
- `xgboost 1.7+`
- `Docker 20.10+`
- `Jupyter Notebook (development)`

---

## Screenshots

### Main UI Interface
*[Insert screenshot of the main web interface showing the input field, prediction button, and result display]*

### Failure Prediction in Action
*[Insert screenshot showing a prediction where the meta-model successfully flagged a failure, displaying both the base model's prediction and the failure probability score]*

---

## License

MIT License - see the [LICENSE](LICENSE) file for details.
```
