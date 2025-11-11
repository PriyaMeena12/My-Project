from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open("transformer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("Sentiment-Prediction-Model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    review = request.form["review"]
    transformed_review = vectorizer.transform([review])
    prediction = model.predict(transformed_review)[0]

    if prediction == 0:
        result = "😞 Negative Intent"
        color = "#ff4d4d"
    elif prediction == 1:
        result = "😐 Neutral Intent"
        color = "#ffd11a"
    else:
        result = "😀 Positive Intent"
        color = "#00cc66"

    return render_template("index.html", review=review, result=result, color=color)


@app.route("/feedback", methods=["POST"])
def feedback():
    feedback_text = request.form["feedback"]
    print("User Feedback:", feedback_text)  # optional: store in DB or file
    return render_template("index.html", message="Thank you for your feedback! 😊")


if __name__ == "__main__":
    app.run(debug=True)
