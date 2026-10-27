"""Reproducible SMS classification experiment and local prediction CLI."""
import argparse
import csv
import hashlib
import io
import json
import platform
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
URL = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'


def normalize(text):
    return re.sub(r'\s+', ' ', text.casefold()).strip()


def load_data(path=None):
    path = Path(path) if path is not None else ROOT / 'data' / 'SMSSpamCollection'
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            archive = response.read()
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            path.write_bytes(bundle.read('SMSSpamCollection'))
    groups = {}
    raw_count = 0
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        label, message = line.split('\t', 1)
        if label not in ('ham', 'spam') or not normalize(message):
            raise ValueError('Invalid dataset record')
        raw_count += 1
        key = normalize(message)
        groups.setdefault(key, {'text': message, 'labels': set()})['labels'].add(label)
    # Remove conflicting annotations and normalized duplicates before splitting.
    rows = [(v['text'], next(iter(v['labels']))) for v in groups.values()
            if len(v['labels']) == 1]
    return rows, {'raw_rows': raw_count, 'unique_rows': len(rows),
                  'conflicting_groups_removed': sum(len(v['labels']) > 1 for v in groups.values()),
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def train():
    rows, provenance = load_data()
    x, y = map(list, zip(*rows))
    x_dev, x_test, y_dev, y_test = train_test_split(
        x, y, test_size=.2, stratify=y, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(
        x_dev, y_dev, test_size=.25, stratify=y_dev, random_state=42)
    estimators = {
        'majority_baseline': DummyClassifier(strategy='most_frequent'),
        'naive_bayes': MultinomialNB(alpha=.5),
        'logistic_regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    }
    models, validation = {}, {}
    for name, estimator in estimators.items():
        model = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                          ('classifier', estimator)])
        model.fit(x_train, y_train)
        predictions = model.predict(x_val)
        validation[name] = classification_report(y_val, predictions, output_dict=True, zero_division=0)
        models[name] = model
    best = max(models, key=lambda name: validation[name]['spam']['f1-score'])
    model = models[best]
    # Test data is never used for model selection or vocabulary fitting.
    model.fit(x_dev, y_dev)
    start = time.perf_counter()
    predictions = model.predict(x_test)
    elapsed = time.perf_counter() - start
    report = {
        'selected_model': best, 'dataset': provenance, 'seed': 42,
        'split_sizes': {'train': len(x_train), 'validation': len(x_val), 'test': len(x_test)},
        'final_fit_rows': len(x_dev), 'selection_metric': 'validation spam F1',
        'validation': validation,
        'test': classification_report(y_test, predictions, output_dict=True, zero_division=0),
        'confusion_matrix_labels': ['ham', 'spam'],
        'confusion_matrix': confusion_matrix(y_test, predictions, labels=['ham', 'spam']).tolist(),
        'batch_ms_per_message': elapsed * 1000 / len(x_test),
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__},
    }
    (ROOT / 'models').mkdir(exist_ok=True)
    (ROOT / 'reports').mkdir(exist_ok=True)
    joblib.dump(model, ROOT / 'models' / 'spamshield.joblib')
    (ROOT / 'reports' / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (ROOT / 'reports' / 'errors.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['message', 'actual', 'predicted'])
        writer.writerows((text, actual, pred) for text, actual, pred in zip(x_test, y_test, predictions) if actual != pred)
    print(json.dumps({'selected_model': best, 'test': report['test'], 'split_sizes': report['split_sizes']}, indent=2))


def predict(message):
    if not message.strip():
        raise ValueError('Enter a non-empty message')
    path = ROOT / 'models' / 'spamshield.joblib'
    if not path.exists():
        raise FileNotFoundError('Train first: python spamshield.py train')
    # Load only trusted local artifacts: joblib files can execute code.
    model = joblib.load(path)
    print(json.dumps({'message': message, 'prediction': str(model.predict([message])[0])}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('train')
    prediction = commands.add_parser('predict')
    prediction.add_argument('message')
    args = parser.parse_args()
    if args.command == 'train':
        train()
    else:
        predict(args.message)
