# Sentiment Analyzer

A web app that compares three different sentiment analysis models — **Logistic Regression**, **Naive Bayes**, and **DistilBERT** — on the same input text, side by side.

Instead of relying on a single model, this project demonstrates how classical machine learning approaches and modern transformer-based deep learning differ in handling real-world text, including negation, sarcasm, and social media slang.

## Demo

Enter any sentence and get:
- A prediction (Positive/Negative) from each model
- A confidence score for each prediction
- A bar chart comparing confidence across all three models

## Why three models?

| Model | Type | Strengths | Weaknesses |
|---|---|---|---|
| Logistic Regression | Classical ML (TF-IDF based) | Fast, interpretable | Ignores word order and context |
| Naive Bayes | Classical ML (probabilistic) | Simple, fast to train | Assumes word independence, weak on negation |
| DistilBERT | Transformer (deep learning) | Understands context via self-attention | Sensitive to domain mismatch (e.g. informal slang) |

## Key finding

Testing revealed that **model performance depends heavily on the type of text**:

- On formal, logically complex sentences with negation (e.g. *"I don't think this could be the best... "*), DistilBERT correctly identified negative sentiment while the classical models were misled by surface-level word frequency.
- On informal, slang-heavy social media text (e.g. *"ngl this app is straight fire no cap 🔥"*), the classical models correctly predicted positive sentiment, while DistilBERT misclassified it — likely due to a **domain mismatch**, since it wasn't fine-tuned on social media-style language.

This shows that a more sophisticated model isn't always superior; it depends on whether the model's training data matches the deployment domain.

## Tech stack

- **Backend:** Python, Flask
- **Classical ML:** scikit-learn (Logistic Regression, Naive Bayes)
- **Deep learning:** Hugging Face Transformers (DistilBERT)
- **Deployment:** Render

## Project structure

```
├── app.py                 # Flask app and routes
├── models/                 # Trained model files
├── templates/               # HTML templates
├── static/                  # CSS/JS assets
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
python app.py
```

The app will run locally at `http://localhost:5000`.

## Limitations & future work

- Currently binary classification only (positive/negative) — no neutral class
- DistilBERT underperforms on informal/slang text due to domain mismatch; fine-tuning on a social-media-specific dataset (e.g. Twitter data) would likely improve accuracy
- Could integrate a hosted, domain-specific model (e.g. `cardiffnlp/twitter-roberta-base-sentiment`) via Hugging Face's Inference API for better real-world performance
