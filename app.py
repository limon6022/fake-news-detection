from flask import Flask, request, jsonify, render_template
import joblib
import re

app = Flask(__name__)

# Load new model bundle (trained on WELFake + ISOT)
bundle = joblib.load("outputs/fake_news_detector_v2.joblib")
model = bundle["model"]
vectorizer = bundle["vectorizer"]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text or len(text) < 20:
        return jsonify({"error": "Please enter a longer news article or headline."}), 400

    cleaned = clean_text(text)
    tfidf = vectorizer.transform([cleaned])
    prediction = model.predict(tfidf)[0]
    proba = model.predict_proba(tfidf)[0]

    label = "FAKE" if prediction == 1 else "REAL"
    confidence = float(proba[prediction])

    return jsonify({
        "label": label,
        "confidence": round(confidence * 100, 1),
        "fake_prob": round(float(proba[1]) * 100, 1),
        "real_prob": round(float(proba[0]) * 100, 1),
    })

if __name__ == "__main__":
    app.run(debug=True)