# 🔍 Fake News Detector

A machine learning web application that detects fake news using Random Forest classification trained on the WELFake dataset.

## Architecture
- **Frontend**: HTML/CSS/JavaScript (templates/index.html)
- **Backend**: Flask API (app.py)
- **Model**: Random Forest — 96% accuracy
- **Vectorizer**: TF-IDF (5000 features, bigrams)
- **Dataset**: WELFake (72,134 news articles)

## How It Works
1. User inputs a news article
2. Text is cleaned and preprocessed
3. TF-IDF vectorizer converts text to numbers
4. Random Forest model predicts Real or Fake
5. Result shown with confidence percentage

## Known Limitations
- Trained on US political news (2015-2018) only
- Performs poorly on short texts under 20 words
- May misclassify investigative journalism as fake due to dramatic language
- Does not generalize to cybersecurity, academic, or non-English content

## How To Run
```bash
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:5000

## Future Improvements
- Add URL scraping for automatic article analysis
- Integrate BERT for better generalization
- Add Finnish language support
- Add source credibility checking