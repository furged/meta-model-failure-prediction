import joblib
import pandas as pd

from src.models.vader_model import get_vader_prediction
from src.models.lr_model import get_lr_prediction
from src.models.bert_model import get_bert_prediction


# Load trained meta-model
meta_model = joblib.load(
    "artifacts/meta_model.pkl"
)


def predict_failure(text):


    vader_pred, vader_score = get_vader_prediction(text)

    lr_pred, lr_confidence = get_lr_prediction(text)

    bert_pred, bert_confidence, bert_entropy = (
        get_bert_prediction(text)
    )


    vader_lr_disagreement = int(
        vader_pred != lr_pred
    )

    lr_bert_disagreement = int(
        lr_pred != bert_pred
    )

    vader_bert_disagreement = int(
        vader_pred != bert_pred
    )


    features = pd.DataFrame([{
        "vader_pred": vader_pred,
        "vader_score": vader_score,
        "lr_pred": lr_pred,
        "lr_confidence": lr_confidence,
        "bert_pred": bert_pred,
        "bert_confidence": bert_confidence,
        "bert_entropy": bert_entropy,
        "vader_lr_disagreement": vader_lr_disagreement,
        "lr_bert_disagreement": lr_bert_disagreement,
        "vader_bert_disagreement": vader_bert_disagreement
    }])


    failure_probability = meta_model.predict_proba(
        features
    )[0][1]


    warning = (
        "TRANSFORMER MAY FAIL"
        if failure_probability >= 0.5
        else "Prediction appears reliable"
    )

    return {

    "text": text,

    "vader_prediction": vader_pred,

    "lr_prediction": lr_pred,

    "bert_prediction": bert_pred,

    "bert_confidence": round(
        bert_confidence,
        4
    ),

    "lr_confidence": round(
        lr_confidence,
        4
    ),

    "vader_score": round(
        abs(vader_score),
        4
    ),

    "bert_entropy": round(
        bert_entropy,
        4
    ),

    "failure_probability": round(
        failure_probability,
        4
    ),

    "warning": warning

}


if __name__ == "__main__":

    while True:

        text = input("\nEnter text: ")

        if text.lower() == "exit":
            break

        result = predict_failure(text)

        print("\nRESULT:\n")

        for key, value in result.items():

            print(f"{key}: {value}")