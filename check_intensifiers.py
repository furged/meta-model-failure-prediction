"""
Diagnostic: checks how many of the curated "negative-word-as-praise"
examples (disgustingly good, stupidly entertaining, etc.) the retrained
meta-model actually flags as failure-risk.

Run with:  python check_intensifiers.py
"""

import pandas as pd
import joblib

from src.config import META_MODEL_PATH, META_THRESHOLD_PATH
from src.data.curated_hard_examples import CURATED_HARD_EXAMPLES
from src.models.vader_model import get_vader_prediction
from src.models.lr_model import get_lr_prediction
from src.models.bert_model import get_bert_prediction
from src.features.intensifiers import has_negative_intensifier


INTENSIFIER_WORDS = {
    "disgustingly", "stupidly", "ridiculously", "absurdly", "annoyingly",
    "criminally", "sinfully", "painfully", "dangerously", "insanely",
    "frighteningly", "obscenely", "disturbingly", "savagely", "wickedly",
    "scandalously", "recklessly", "shamelessly", "brutally", "genuinely",
    "truly", "disappointingly", "frustratingly", "hopelessly", "tediously",
    "depressingly", "astonishingly"
}


def main():
    meta_model = joblib.load(META_MODEL_PATH)
    threshold = joblib.load(META_THRESHOLD_PATH)

    examples = [
        text for text, label in CURATED_HARD_EXAMPLES
        if text.split()[0].lower() in INTENSIFIER_WORDS
    ]

    print(f"Testing {len(examples)} intensifier-category examples\n")

    caught = 0

    for text in examples:
        vader_pred, vader_score = get_vader_prediction(text)
        lr_pred, lr_conf = get_lr_prediction(text)
        bert_pred, bert_conf, bert_entropy = get_bert_prediction(text)

        features = pd.DataFrame([{
            "vader_pred": vader_pred,
            "vader_score": vader_score,
            "lr_pred": lr_pred,
            "lr_confidence": lr_conf,
            "bert_pred": bert_pred,
            "bert_confidence": bert_conf,
            "bert_entropy": bert_entropy,
            "vader_lr_disagreement": int(vader_pred != lr_pred),
            "lr_bert_disagreement": int(lr_pred != bert_pred),
            "vader_bert_disagreement": int(vader_pred != bert_pred),
            "has_negative_intensifier": has_negative_intensifier(text)
        }])

        proba = meta_model.predict_proba(features)[0][1]
        flagged = proba >= threshold

        if flagged:
            caught += 1

        intensifier_flag = has_negative_intensifier(text)

        print(
            f"{text[:45]:45s} "
            f"bert_pred={bert_pred} conf={bert_conf:.3f} "
            f"intensifier={intensifier_flag} "
            f"fail_prob={proba:.3f} flagged={flagged}"
        )

    print()
    print(f"Flagged {caught}/{len(examples)} intensifier examples as failure-risk")


if __name__ == "__main__":
    main()
