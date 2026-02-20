import argparse
import joblib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment import SentimentIntensityAnalyzer

# Load saved artifacts
tfidf = joblib.load("artifacts/tfidf.pkl")
lr_model = joblib.load("artifacts/lr_model.pkl")
meta_model = joblib.load("artifacts/meta_model.pkl")

# Load transformer
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
transformer = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
transformer.eval()

vader = SentimentIntensityAnalyzer()

def vader_predict(text):
    score = vader.polarity_scores(text)["compound"]
    pred = 1 if score > 0 else 0
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

    features = [[
        v_pred,
        v_conf,
        t_pred,
        t_conf,
        tr_conf
    ]]

    risk = meta_model.predict(features)[0]

    print("\n=== Results ===")
    print("Transformer Prediction:", "Positive" if tr_pred == 1 else "Negative")
    print("Transformer Confidence:", round(tr_conf, 4))
    print("Failure Risk:", "HIGH" if risk == 1 else "LOW")

if __name__ == "__main__":
    main()
