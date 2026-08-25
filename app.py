import matplotlib
matplotlib.use('Agg')  # needed so matplotlib works inside Flask (no GUI)
import matplotlib.pyplot as plt

from flask import Flask, render_template, request
import joblib
from transformers import pipeline

app = Flask(__name__)

# Load your two trained models
lr_model = joblib.load('sentiment_model.pkl')
nb_model = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load the pretrained transformer (this takes a moment on startup)
transformer = pipeline("sentiment-analysis", 
                        model="distilbert-base-uncased-finetuned-sst-2-english")

@app.route('/', methods=['GET', 'POST'])
def home():
    text = ""
    results = None

    if request.method == 'POST':
        text = request.form['text']
        text_tfidf = vectorizer.transform([text])

        lr_pred = lr_model.predict(text_tfidf)[0]
        lr_proba = max(lr_model.predict_proba(text_tfidf)[0]) * 100
        lr_label = "Positive" if lr_pred == 1 else "Negative"

        nb_pred = nb_model.predict(text_tfidf)[0]
        nb_proba = max(nb_model.predict_proba(text_tfidf)[0]) * 100
        nb_label = "Positive" if nb_pred == 1 else "Negative"

        db_result = transformer(text)[0]
        db_label = "Positive" if db_result['label'] == 'POSITIVE' else "Negative"
        db_proba = db_result['score'] * 100

        results = {
            'Logistic Regression': {'label': lr_label, 'confidence': round(lr_proba, 1)},
            'Naive Bayes': {'label': nb_label, 'confidence': round(nb_proba, 1)},
            'DistilBERT': {'label': db_label, 'confidence': round(db_proba, 1)}
        }

        # Generate a fresh chart for this prediction
        models = list(results.keys())
        confidences = [results[m]['confidence'] for m in models]
        colors = ['#4C72B0' if results[m]['label'] == 'Positive' else '#DD8452' for m in models]

        plt.figure(figsize=(6, 4))
        bars = plt.bar(models, confidences, color=colors)
        plt.ylim(0, 100)
        plt.ylabel('Confidence (%)')
        plt.title('Prediction Confidence by Model')
        for bar, conf in zip(bars, confidences):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                      f'{conf}%', ha='center', fontsize=10)
        plt.tight_layout()
        plt.savefig('static/live_chart.png')
        plt.close()

    return render_template('index.html', text=text, results=results)

if __name__ == '__main__':
    app.run(debug=True)