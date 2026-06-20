from flask import Flask, render_template, request, jsonify

from src.predict import predict_failure
from src.config import METRICS_PATH

import json
import os


app = Flask(__name__)


# Load model metrics + PR curve once at startup (not per-request -- this
# file is small and only changes when the model is retrained, so there's
# no reason to re-read it from disk on every page view).
def load_metrics():
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        # Lets the app still boot if metrics.json hasn't been generated
        # yet (e.g. fresh clone before running the training script), with
        # an obvious "not available" state in the UI rather than crashing.
        return None


METRICS = load_metrics()


@app.route("/")
def home():
    return render_template("index.html", metrics=METRICS)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}

    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter some text to analyze."}), 400

    if len(text) > 2000:
        return jsonify({
            "error": "Text is too long (max 2000 characters)."
        }), 400

    result = predict_failure(text)

    return jsonify(result)


@app.route("/api/metrics")
def metrics():
    if METRICS is None:
        return jsonify({"error": "Metrics not available."}), 404

    return jsonify(METRICS)


if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 10000)
    )

    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    app.run(

        host="0.0.0.0",

        port=port,

        debug=debug_mode

    )