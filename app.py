from flask import Flask, render_template, request

from src.predict import predict_failure


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        text = request.form["text"]

        result = predict_failure(text)

    return render_template(
        "index.html",
        result=result
    )


if __name__ == "__main__":

    app.run(debug=True)