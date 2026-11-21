"""SpamShield service: FastAPI JSON API plus a single-page UI.

    python benchmark.py --export-primary          # creates models/spamshield_svm.joblib (char n-gram SVM, calibrated)
    python benchmark.py                           # optional: fine-tuned transformer in models/transformer (second opinion)
    uvicorn spam_api:app --port 8000

Endpoints: GET /  (UI), GET /health, POST /predict {message}, POST /probe {message}
/predict returns the spam probability from the primary model (character n-gram SVM, probabilities calibrated with
CalibratedClassifierCV) and, when the transformer is present, a second opinion from it, plus the character n-grams
that pushed the primary score up or down. The SVM ships as primary because it matched the transformer on F1
(McNemar p = 1.0) and held up far better under obfuscation; see the README. /probe re-scores the message after
obfuscation tricks (leetspeak, look-alike letters, dotted words, link tricks) to show how fragile each model is.
Messages are not stored or logged. The model files are local trusted artifacts (joblib can execute code).
"""
import os
import time
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MAX_CHARS = 1000
MAX_BODY_BYTES = 8 * 1024        # 1000 chars of JSON-escaped unicode fits comfortably; anything larger is rejected unread
RATE_LIMIT_PER_MINUTE = int(os.environ.get('SPAMSHIELD_RATE_LIMIT', '30'))
RATE_WINDOW_SECONDS = 60
EXEMPT_PATHS = {'/', '/health'}


class Message(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_CHARS)


class Models:
    """Holds the primary pipeline (char n-gram SVM) and, optionally, the transformer second opinion."""

    def __init__(self, primary, transformer=None):
        self.primary, self.transformer = primary, transformer

    def primary_probability(self, texts):
        return self.primary.predict_proba(texts)[:, 1]

    def transformer_probability(self, texts):
        return None if self.transformer is None else self.transformer(texts)

    def explain(self, text, top=6):
        """Character n-grams moving the score: tf-idf value times the mean linear-SVM coefficient (margin units, not probabilities)."""
        steps = self.primary.named_steps
        vectorizer, clf = steps.get('v'), steps.get('m')
        members = getattr(clf, 'calibrated_classifiers_', None)
        if vectorizer is None or not members:
            return []
        coef = np.mean([m.estimator.coef_[0] for m in members], axis=0)
        row = vectorizer.transform([text])
        names = vectorizer.get_feature_names_out()
        contributions = [(names[i], float(v * coef[i])) for i, v in zip(row.indices, row.data)]
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
    path = ROOT / 'models' / 'spamshield_svm.joblib'
    if not path.exists():
        raise FileNotFoundError('Train first: python benchmark.py --export-primary')
    transformer_dir = ROOT / 'models' / 'transformer'
    transformer = load_transformer(transformer_dir) if transformer_dir.exists() and not os.environ.get('SPAMSHIELD_NO_TRANSFORMER') else None
    return Models(joblib.load(path), transformer)


def verdict(probability):
    return {'spam_probability': round(float(probability), 4), 'label': 'spam' if probability >= 0.5 else 'ham'}


def create_app(models_provider=default_models, rate_limit=RATE_LIMIT_PER_MINUTE, clock=time.monotonic):
    app = FastAPI(title='SpamShield', version='2.0.0',
                  description='SMS spam classifier trained on the 2011 UCI SMS Spam Collection (English). '
                              'Not a production filter; see the README for limits.')

    hits = defaultdict(deque)   # client address -> timestamps of requests inside the window (in-memory, per process)

    def client_address(request):
        # Behind Render's proxy the real client is the first X-Forwarded-For hop.
        forwarded = request.headers.get('x-forwarded-for', '')
        return forwarded.split(',')[0].strip() or (request.client.host if request.client else 'unknown')

    @app.middleware('http')
    async def limits(request, call_next):
        if request.method == 'POST':
            length = request.headers.get('content-length')
            if (length is None and 'transfer-encoding' in request.headers) or (length is not None and length.isdigit() and int(length) > MAX_BODY_BYTES):
                return JSONResponse({'detail': f'Request body too large (limit {MAX_BODY_BYTES} bytes).'}, status_code=413)
        if request.url.path not in EXEMPT_PATHS and rate_limit:
            now, window = clock(), hits[client_address(request)]
            while window and now - window[0] >= RATE_WINDOW_SECONDS:
                window.popleft()
            if len(window) >= rate_limit:
                retry = max(1, int(RATE_WINDOW_SECONDS - (now - window[0])) + 1)
                return JSONResponse({'detail': f'Rate limit of {rate_limit} requests per minute exceeded. Try again shortly.'},
                                    status_code=429, headers={'Retry-After': str(retry)})
            window.append(now)
            if len(hits) > 10000:   # bound memory: drop idle addresses
                for key in [k for k, v in hits.items() if not v or now - v[-1] >= RATE_WINDOW_SECONDS]:
                    del hits[key]
        return await call_next(request)

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
        result = {'char_svm': verdict(m.primary_probability([text])[0]), 'explanation': m.explain(text)}
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
        classical = m.primary_probability(texts)
        transformer = m.transformer_probability(texts)
        rows = []
        for i, name in enumerate(names):
            row = {'variant': name, 'text': texts[i], 'char_svm': round(float(classical[i]), 4)}
            if transformer is not None:
                row['transformer'] = round(float(transformer[i]), 4)
            rows.append(row)
        return {'rows': rows}

    return app


app = create_app()
