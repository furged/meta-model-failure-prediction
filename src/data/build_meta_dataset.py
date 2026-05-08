from src.models.vader_model import get_vader_prediction
from src.models.lr_model import get_lr_prediction
from src.models.bert_model import get_bert_prediction

from src.features.noise import add_noise

from datasets import load_dataset
import pandas as pd
import random


# Load SST-2 dataset
dataset = load_dataset("sst2")


# Take training split
train_data = dataset["train"]


# Convert to pandas dataframe
df = pd.DataFrame(train_data)


# Rename columns
df = df.rename(columns={
    "sentence": "text",
    "label": "true_label"
})


# Keep only needed columns
df = df[["text", "true_label"]]


# Take sample
df = df.sample(5000, random_state=42)


# ---------------- Add Noise ----------------

random.seed(42)

df["text"] = df["text"].apply(
    lambda x: add_noise(x) if random.random() < 0.3 else x
)


# ---------------- VADER ----------------

vader_outputs = df["text"].apply(get_vader_prediction)

df["vader_pred"] = vader_outputs.apply(lambda x: x[0])

df["vader_score"] = vader_outputs.apply(lambda x: x[1])


# ---------------- Logistic Regression ----------------

lr_outputs = df["text"].apply(get_lr_prediction)

df["lr_pred"] = lr_outputs.apply(lambda x: x[0])

df["lr_confidence"] = lr_outputs.apply(lambda x: x[1])


# ---------------- DistilBERT ----------------

bert_outputs = df["text"].apply(get_bert_prediction)

df["bert_pred"] = bert_outputs.apply(lambda x: x[0])

df["bert_confidence"] = bert_outputs.apply(lambda x: x[1])

df["bert_entropy"] = bert_outputs.apply(lambda x: x[2])


# ---------------- Disagreement Features ----------------

df["vader_lr_disagreement"] = (
    df["vader_pred"] != df["lr_pred"]
).astype(int)

df["lr_bert_disagreement"] = (
    df["lr_pred"] != df["bert_pred"]
).astype(int)

df["vader_bert_disagreement"] = (
    df["vader_pred"] != df["bert_pred"]
).astype(int)


# ---------------- Failure Label ----------------

df["bert_failed"] = (
    df["bert_pred"] != df["true_label"]
).astype(int)


# Save processed dataset
df.to_csv(
    "C:\\Users\\chiya\\Documents\\meta-model-failure-prediction\\src\\data\\processed\\base_dataset.csv",
    index=False
)


# Preview dataframe
print(df.head())

print("\nDataset saved successfully.")