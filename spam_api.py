"""SpamShield service: FastAPI JSON API plus a single-page UI.

    python spamshield.py train                    # creates models/spamshield.joblib (TF-IDF + logistic regression)
    python benchmark.py                           # optional: fine-tuned transformer in models/transformer
    uvicorn spam_api:app --port 8000

Endpoints: GET /  (UI), GET /health, POST /predict {message}, POST /probe {message}
/predict returns the spam probability from the logistic regression and, when the transformer is present, from it too,
plus the words that pushed the logistic-regression score up or down. /probe re-scores the message after
obfuscation tricks (leetspeak, look-alike letters, dotted words, link tricks) to show how fragile each model is.
Messages are not stored or logged. The model files are local trusted artifacts (joblib can execute code).
"""
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MAX_CHARS = 1000


class Message(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_CHARS)


class Models:
    """Holds the classical pipeline and (optionally) the transformer; both expose spam_probability(list[str])."""

    def __init__(self, classical, transformer=None):
        self.classical, self.transformer = classical, transformer

    def classical_probability(self, texts):
        return self.classical.predict_proba(texts)[:, 1]

    def transformer_probability(self, texts):
        return None if self.transformer is None else self.transformer(texts)

    def explain(self, text, top=6):
        """Words moving the logistic-regression score: tf-idf value times coefficient (log-odds units)."""
        vectorizer, clf = self.classical.named_steps.get('tfidf'), self.classical.named_steps.get('classifier')
        if vectorizer is None or not hasattr(clf, 'coef_'):
            return []
        row = vectorizer.transform([text])
        names = vectorizer.get_feature_names_out()
        contributions = [(names[i], float(v * clf.coef_[0][i])) for i, v in zip(row.indices, row.data)]
        contributions.sort(key=lambda item: -abs(item[1]))
        return [{'term': t, 'weight': round(w, 3)} for t, w in contributions[:top]]


def load_transformer(directory):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForSequenceClassification.from_pretrained(directory).eval()

    def probability(texts):
        with torch.inference_mode():
            batch = tokenizer(list(texts), truncation=True, max_length=96, padding=True, return_tensors='pt')
            return torch.softmax(model(**batch).logits, dim=1)[:, 1].numpy()
    return probability


@lru_cache(maxsize=1)
def default_models():
    path = ROOT / 'models' / 'spamshield.joblib'
    if not path.exists():
        raise FileNotFoundError('Train first: python spamshield.py train')
    transformer_dir = ROOT / 'models' / 'transformer'
    transformer = load_transformer(transformer_dir) if transformer_dir.exists() and not os.environ.get('SPAMSHIELD_NO_TRANSFORMER') else None
    return Models(joblib.load(path), transformer)


def verdict(probability):
    return {'spam_probability': round(float(probability), 4), 'label': 'spam' if probability >= 0.5 else 'ham'}


def create_app(models_provider=default_models):
    app = FastAPI(title='SpamShield', version='2.0.0',
                  description='SMS spam classifier trained on the 2011 UCI SMS Spam Collection (English). '
                              'Not a production filter; see the README for limits.')

    def models():
        try:
            return models_provider()
        except FileNotFoundError as error:
            raise HTTPException(503, str(error)) from None

    @app.get('/', include_in_schema=False)
    def index():
        return FileResponse(ROOT / 'static' / 'index.html')

    @app.get('/health')
    def health():
        try:
            m = models_provider()
            return {'status': 'ok', 'ready': True, 'transformer': m.transformer is not None}
        except FileNotFoundError:
            return {'status': 'ok', 'ready': False, 'transformer': False}

    @app.post('/predict')
    def predict(body: Message):
        m, text = models(), body.message
        result = {'logistic_regression': verdict(m.classical_probability([text])[0]), 'explanation': m.explain(text)}
        transformer = m.transformer_probability([text])
        if transformer is not None:
            result['transformer'] = verdict(transformer[0])
        return result

    @app.post('/probe')
    def probe(body: Message):
        from benchmark import PERTURBATIONS
        m, text = models(), body.message
        variants = {'original': text, **{name: fn(text) for name, fn in PERTURBATIONS.items()}}
        names, texts = list(variants), list(variants.values())
        classical = m.classical_probability(texts)
        transformer = m.transformer_probability(texts)
        rows = []
        for i, name in enumerate(names):
            row = {'variant': name, 'text': texts[i], 'logistic_regression': round(float(classical[i]), 4)}
            if transformer is not None:
                row['transformer'] = round(float(transformer[i]), 4)
            rows.append(row)
        return {'rows': rows}

    return app


app = create_app()
