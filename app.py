from flask import Flask, render_template, request

from src.predict import predict_failure

import os


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    input_text = ""

    if request.method == "POST":

        input_text = request.form.get("text", "")

        result = predict_failure(input_text)

    return render_template(

        "index.html",

        result=result,

        input_text=input_text

    )


if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 10000)
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=True

    )