FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer unless requirements change)
COPY requirements.txt .

# CPU-only torch index to keep image size down. requirements.txt alone
# is sufficient here -- it already includes pandas/scikit-learn/joblib,
# everything src/models/meta_model.py needs to train. requirements-dev.txt
# adds datasets/nltk/matplotlib, which are only needed for rebuilding the
# dataset from scratch (src/data/build_meta_dataset.py), not for training
# against the already-computed CSV -- so it's deliberately left out here
# to keep the image smaller and the build faster.
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the app
COPY . .

# HuggingFace Spaces runs on port 7860
ENV PORT=7860

# Pre-download the DistilBERT model during build so it's cached in the
# image and the first request doesn't hit a cold-start download timeout.
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', device=-1)"

# Train the meta-model fresh during the build, from the already-computed
# dataset CSV (src/data/processed/base_dataset.csv, committed to the repo
# as plain text -- no Git LFS needed). This step is fast (~10-30s, pure
# sklearn on tabular data, no BERT inference involved) and means the
# trained meta_model.pkl/meta_threshold.pkl/metrics.json never need to be
# pushed to git as binary artifacts -- they're generated identically on
# every build instead.
RUN python -m src.models.meta_model

EXPOSE 7860

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1"]
