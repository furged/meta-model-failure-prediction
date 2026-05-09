from transformers import pipeline


classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


def get_bert_prediction(text):

    result = classifier(text)[0]

    label = result["label"]

    confidence = result["score"]

    prediction = 1 if label == "POSITIVE" else 0

    entropy = 1 - confidence

    return prediction, confidence, entropy


if __name__ == "__main__":

    text = "this movie was amazing"

    prediction, confidence, entropy = (
        get_bert_prediction(text)
    )

    print("Prediction:", prediction)

    print("Confidence:", confidence)

    print("Entropy:", entropy)