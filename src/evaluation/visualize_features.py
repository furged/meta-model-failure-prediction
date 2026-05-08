import pandas as pd
import matplotlib.pyplot as plt
import joblib


# Load dataset
df = pd.read_csv(
    "C:\\Users\\chiya\\Documents\\meta-model-failure-prediction\\src\\data\\processed\\base_dataset.csv"
)


# Feature names
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


# Load trained model
meta_model = joblib.load(
    "artifacts/meta_model.pkl"
)


# Extract coefficients
coefficients = meta_model.coef_[0]


# Create dataframe
importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Coefficient": coefficients
})


# Absolute importance
importance_df["Absolute"] = importance_df[
    "Coefficient"
].abs()


# Sort
importance_df = importance_df.sort_values(
    by="Absolute",
    ascending=True
)


# Plot
plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Absolute"]
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title("Meta-Feature Importance for Transformer Failure Prediction")

plt.tight_layout()

plt.show()