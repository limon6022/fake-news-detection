# 🔍 Fake News Detector

> A machine learning web application that classifies news articles as **Real** or **Fake** using Random Forest with TF-IDF features — trained on 72,134 articles and explained with SHAP and LIME.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/Accuracy-96%25-brightgreen)

---

## 📌 Overview

Fake news detection is a binary text classification problem. This project builds a full pipeline from raw data to a deployed web app — including preprocessing, model comparison, explainability analysis, and a Flask API.

Three models were benchmarked (Logistic Regression, Naive Bayes, Random Forest). **Random Forest achieved the best results** and was selected for deployment.

---

## 📊 Model Performance

| Model | Accuracy | F1-Score (Real) | F1-Score (Fake) |
|---|---|---|---|
| Logistic Regression | 95% | 0.95 | 0.95 |
| Naive Bayes | 85% | 0.84 | 0.86 |
| **Random Forest** ✅ | **96%** | **0.95** | **0.96** |

> Evaluated on a held-out test set of 18,034 articles (25% split).

---

## 🧠 Explainability

A key feature of this project is **model transparency** — understanding *why* the model makes a prediction, not just *what* it predicts.

### SHAP — Global Feature Importance

SHAP (SHapley Additive exPlanations) reveals which words consistently drive predictions across the full dataset.

![SHAP Global Bar](outputs/shap_global_bar.png)

Words like `reuters` strongly push predictions toward **Real** (the word appears in legitimate news sources), while terms like `don`, `image`, and `video` are associated with **Fake** content.

### SHAP — Value Distribution (Beeswarm)

![SHAP Beeswarm](outputs/shap_beeswarm.png)

Each dot represents one prediction. Color shows feature value (red = high, blue = low). The horizontal position shows whether the word pushed the model toward Fake or Real.

### LIME — Local Explanation

LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions word-by-word.

![LIME Explanation](outputs/lime_explanation.png)

For this example article, `don` was the strongest word pushing the model toward **FAKE**, while `said` and `year` pulled it toward Real.

---

## 🏗️ Architecture

```
User Input (news text)
        │
        ▼
   Text Cleaning
   (lowercase, remove URLs, punctuation)
        │
        ▼
   TF-IDF Vectorizer
   (5,000 features, unigrams + bigrams)
        │
        ▼
   Random Forest Classifier
   (trained on WELFake dataset)
        │
        ▼
   Prediction + Confidence Score
   {"label": "FAKE", "confidence": 91.3}
```

**Stack:**
- **Frontend:** HTML / CSS / JavaScript
- **Backend:** Flask REST API
- **ML:** scikit-learn (Random Forest, TF-IDF)
- **Explainability:** SHAP, LIME
- **Dataset:** [WELFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) — 72,134 articles, 233 MB

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/limon6022/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
```

### Run the App

```bash
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### API Usage

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists confirm water found on Mars surface in new study."}'
```

**Response:**
```json
{
  "label": "REAL",
  "confidence": 87.4,
  "fake_prob": 12.6,
  "real_prob": 87.4
}
```

---

## 📁 Project Structure

```
fake-news-detection/
├── app.py                        # Flask web app & REST API
├── fake_news_detection.ipynb     # Full ML pipeline (EDA → training → explainability)
├── requirements.txt
├── templates/
│   └── index.html                # Frontend UI
└── outputs/
    ├── fake_news_detector.joblib  # Saved model + vectorizer bundle
    ├── shap_global_bar.png        # SHAP top-20 features
    ├── shap_beeswarm.png          # SHAP value distribution
    └── lime_explanation.png       # LIME local explanation example
```

---

## ⚠️ Known Limitations

| Limitation | Detail |
|---|---|
| **Domain bias** | Trained on US political news (2015–2018) only |
| **Short text** | Performs poorly on inputs under ~20 words |
| **Language style** | May flag investigative journalism as fake due to dramatic phrasing |
| **Language** | English only — does not generalize to Finnish or other languages |
| **Temporal drift** | Political language evolves; model may degrade on post-2018 articles |

---

## 🗺️ Future Work

- [ ] Add URL scraping so users can paste a link instead of full text
- [ ] Fine-tune a BERT-based model for better generalization
- [ ] Add Dockerfile for one-command deployment
- [ ] Extend dataset with post-2020 articles
- [ ] Add Finnish-language support

---

## 👤 Author

**limon6022** — University ML course project  
Built with scikit-learn, Flask, SHAP, and LIME.
