FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer unless requirements change)
COPY requirements.txt .

# CPU-only torch index to keep image size down
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the app
COPY . .

# HuggingFace Spaces runs on port 7860
ENV PORT=7860

# Pre-download the DistilBERT model during build so it's cached in the
# image and the first request doesn't hit a cold-start download timeout.
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', device=-1)"

EXPOSE 7860

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1"]
