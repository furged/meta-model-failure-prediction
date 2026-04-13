import argparse
import joblib
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

# Load saved artifacts
tfidf = joblib.load("artifacts/tfidf.pkl")
lr_model = joblib.load("artifacts/lr_model.pkl")
meta_model = joblib.load("artifacts/meta_model.pkl")

# Load transformer
from functools import lru_cache

@lru_cache()
def load_transformer():
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model.eval()
    return tokenizer, model

tokenizer, transformer = load_transformer()

vader = SentimentIntensityAnalyzer()

def vader_predict(text):
    score = vader.polarity_scores(text)["compound"]
    pred = 1 if score >= 0 else 0
    conf = abs(score)
    return pred, conf

def tfidf_predict(text):
    vec = tfidf.transform([text])
    pred = lr_model.predict(vec)[0]
    conf = max(lr_model.predict_proba(vec)[0])
    return pred, conf

def transformer_predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = transformer(**inputs)
    probs = F.softmax(outputs.logits, dim=1)[0]
    pred = torch.argmax(probs).item()
    conf = torch.max(probs).item()
    return pred, conf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    text = args.text

    v_pred, v_conf = vader_predict(text)
    t_pred, t_conf = tfidf_predict(text)
    tr_pred, tr_conf = transformer_predict(text)

    preds = [v_pred, t_pred, tr_pred]
    confs = [v_conf, t_conf, tr_conf]

    disagreement = len(set(preds)) / 3
    conf_std = np.std(confs)

    features = [[
        v_conf,
        t_conf,
        tr_conf,
        disagreement,
        conf_std
    ]]

    risk_prob = meta_model.predict_proba(features)[0][1]
    risk_label = "HIGH" if risk_prob > 0.5 else "LOW"

    print("\n=== Results ===")
    print("Transformer Prediction:", "Positive" if tr_pred == 1 else "Negative")
    print("Transformer Confidence:", round(tr_conf, 4))
    print("Failure Risk:", risk_label)

if __name__ == "__main__":
    main()

def predict_text(text):
    v_pred, v_conf = vader_predict(text)
    t_pred, t_conf = tfidf_predict(text)
    tr_pred, tr_conf = transformer_predict(text)

    preds = [v_pred, t_pred, tr_pred]
    confs = [v_conf, t_conf, tr_conf]

    disagreement = len(set(preds)) / 3
    conf_std = np.std(confs)

    features = [[v_conf, t_conf, tr_conf, disagreement, conf_std]]

    risk_prob = meta_model.predict_proba(features)[0][1]

    return {
        "sentiment": "Positive" if tr_pred == 1 else "Negative",
        "risk": risk_prob,
        "transformer_confidence": tr_conf,
        "vader_confidence": v_conf,
        "tfidf_confidence": t_conf
    }


