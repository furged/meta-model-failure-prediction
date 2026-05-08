import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Load dataset
df = pd.read_csv(
    "C:\\Users\\chiya\\Documents\\meta-model-failure-prediction\\src\\data\\processed\\base_dataset.csv"
)


# Meta-features
feature_columns = [
    "vader_pred",
    "vader_score",
    "lr_pred",
    "lr_confidence",
    "bert_pred",
    "bert_confidence",
    "bert_entropy",
    "vader_lr_disagreement",
    "lr_bert_disagreement",
    "vader_bert_disagreement"
]


X = df[feature_columns]


# Target
y = df["bert_failed"]


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Meta-model
meta_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)


# Train
meta_model.fit(X_train, y_train)


# Predict
y_pred = meta_model.predict(X_test)


# Evaluation
print(classification_report(y_test, y_pred))


# ---------------- Feature Importance ----------------

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Coefficient": meta_model.coef_[0]
})


importance_df["Absolute"] = importance_df[
    "Coefficient"
].abs()


importance_df = importance_df.sort_values(
    by="Absolute",
    ascending=False
)


print("\nFeature Importance:\n")

print(importance_df)


# Save model
joblib.dump(
    meta_model,
    "artifacts/meta_model.pkl"
)

print("\nMeta-model saved successfully.")