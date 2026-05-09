from transformers import pipeline
import math


classifier = pipeline(
    "sentiment-analysis",
    model="sshleifer/tiny-distilbert-base-uncased-finetuned-sst2",
    top_k=None
)


def calculate_entropy(probabilities):

    entropy = 0

    for p in probabilities:

        entropy -= p * math.log(p + 1e-10)

    return entropy


def get_bert_prediction(text):

    results = classifier(text)[0]

    probabilities = [r["score"] for r in results]

    entropy = calculate_entropy(probabilities)

    positive_score = 0

    for r in results:

        if r["label"] == "POSITIVE":
            positive_score = r["score"]

    prediction = 1 if positive_score >= 0.5 else 0

    confidence = max(probabilities)

    return prediction, confidence, entropy


if __name__ == "__main__":

    text = "this movie was amazing"

    prediction, confidence, entropy = get_bert_prediction(text)

    print("Prediction:", prediction)

    print("Confidence:", confidence)

    print("Entropy:", entropy)