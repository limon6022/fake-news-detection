from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re

app = Flask(__name__)

# Load BERT model
print("Loading BERT model...")
MODEL_PATH = "bert_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded on {device}")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict(text):
    cleaned = clean_text(text)
    inputs = tokenizer(
        cleaned,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids      = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        proba = torch.softmax(outputs.logits, dim=1)[0]
        prediction = torch.argmax(proba).item()

    label = "FAKE" if prediction == 1 else "REAL"
    confidence = float(proba[prediction])

    return {
        "label": label,
        "confidence": round(confidence * 100, 1),
        "real_prob": round(float(proba[0]) * 100, 1),
        "fake_prob": round(float(proba[1]) * 100, 1),
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text or len(text) < 20:
        return jsonify({"error": "Please enter a longer news article or headline."}), 400

    result = predict(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)