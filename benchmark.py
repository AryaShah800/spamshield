"""Benchmark: classical TF-IDF models vs a fine-tuned transformer on identical splits, plus robustness to obfuscation.

    python benchmark.py                      # classical models + MiniLM fine-tune (CPU), writes reports/benchmark.json
    python benchmark.py --skip-transformer   # classical only (fast)

Protocol (fixed before looking at test results):
  * Same deduplicated data and the same 60/20/20 stratified split (seed 42) as spamshield.py.
  * Hyper-parameters are selected on the validation split only; each model is then refit on train+validation
    and evaluated ONCE on the test split. The transformer trains for a fixed 3 epochs (no test-based stopping).
  * Spam threshold: 0.5 for every model (also reports PR-AUC, which is threshold free).
  * 95% intervals are percentile bootstrap over the 1,032 test messages (2,000 draws, seed 0). They capture test-set
    sampling only; training randomness of the transformer is NOT measured (one seed).
  * Robustness: deterministic text perturbations applied to the test SPAM messages only; we report spam recall.
"""
import argparse
import json
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import LinearSVC

from spamshield import ROOT, load_data

HOMOGLYPHS = str.maketrans({'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с', 'x': 'х', 'i': 'і'})
LEET = str.maketrans({'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5', 't': '7'})


def leetspeak(text):
    return text.translate(LEET)


def homoglyphs(text):
    return text.translate(HOMOGLYPHS)


def dotted_words(text):
    """Insert a dot between the letters of every word of 5+ letters (f.r.e.e style)."""
    return ' '.join('.'.join(w) if w.isalpha() and len(w) >= 5 else w for w in text.split(' '))


def spaced_digits_and_links(text):
    return text.replace('http', 'hxxp').replace('www.', 'www .').replace('.com', ' . com')


PERTURBATIONS = {'leetspeak': leetspeak, 'homoglyphs': homoglyphs, 'dotted_words': dotted_words,
                 'link_obfuscation': spaced_digits_and_links}


def splits():
    rows, provenance = load_data()
    x, y = map(list, zip(*rows))
    x_dev, x_test, y_dev, y_test = train_test_split(x, y, test_size=.2, stratify=y, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_dev, y_dev, test_size=.25, stratify=y_dev, random_state=42)
    to_int = lambda labels: np.array([1 if label == 'spam' else 0 for label in labels])
    return {'train': (x_train, to_int(y_train)), 'val': (x_val, to_int(y_val)), 'dev': (x_dev, to_int(y_dev)),
            'test': (x_test, to_int(y_test))}, provenance


def classical_candidates():
    word = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = lambda: TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    return {
        'tfidf_word_logreg': lambda c: Pipeline([('v', word()), ('m', LogisticRegression(C=c, class_weight='balanced', max_iter=2000, random_state=42))]),
        'tfidf_word_nb': lambda c: Pipeline([('v', word()), ('m', MultinomialNB(alpha=c))]),
        'tfidf_char_svm': lambda c: Pipeline([('v', char()), ('m', CalibratedClassifierCV(LinearSVC(C=c, class_weight='balanced'), cv=3))]),
    }, {'tfidf_word_logreg': [1, 10, 100], 'tfidf_word_nb': [.1, .5, 1.0], 'tfidf_char_svm': [.1, 1, 10]}


def spam_scores(model, texts):
    return model.predict_proba(texts)[:, 1]


def metrics(y, score, threshold=.5):
    pred = (score >= threshold).astype(int)
    return {'spam_precision': round(float(precision_score(y, pred, zero_division=0)), 4),
            'spam_recall': round(float(recall_score(y, pred)), 4), 'spam_f1': round(float(f1_score(y, pred)), 4),
            'pr_auc': round(float(average_precision_score(y, score)), 4),
            'false_positives': int(((pred == 1) & (y == 0)).sum()), 'false_negatives': int(((pred == 0) & (y == 1)).sum())}


def bootstrap_f1(y, score, draws=2000, seed=0):
    rng, pred = np.random.default_rng(seed), (score >= .5).astype(int)
    values = []
    for _ in range(draws):
        idx = rng.integers(0, len(y), len(y))
        values.append(f1_score(y[idx], pred[idx], zero_division=0))
    return [round(float(v), 4) for v in np.percentile(values, [2.5, 97.5])]


def mcnemar(y, score_a, score_b):
    """Exact two-sided McNemar test on which model is right where they disagree."""
    from math import comb
    a, b = (score_a >= .5).astype(int) == y, (score_b >= .5).astype(int) == y
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(0, min(only_a, only_b) + 1)) / 2 ** n)
    return {'only_first_correct': only_a, 'only_second_correct': only_b, 'p_value': round(p, 4)}


def latency_ms(predict_one, texts, runs=100):
    for t in texts[:10]:
        predict_one(t)
    samples = []
    for t in (texts * (runs // len(texts) + 1))[:runs]:
        start = time.perf_counter()
        predict_one(t)
        samples.append((time.perf_counter() - start) * 1000)
    return {'median_ms': round(statistics.median(samples), 2), 'p95_ms': round(float(np.percentile(samples, 95)), 2)}


def run_classical(data):
    builders, grids = classical_candidates()
    out, scores, models = {}, {}, {}
    for name, build in builders.items():
        best_c = max(grids[name], key=lambda c: f1_score(data['val'][1], (spam_scores(build(c).fit(*data['train']), data['val'][0]) >= .5).astype(int)))
        model = build(best_c).fit(*data['dev'])
        models[name] = model
        scores[name] = spam_scores(model, data['test'][0])
        out[name] = {'selected_param': best_c, **metrics(data['test'][1], scores[name]),
                     'f1_ci95': bootstrap_f1(data['test'][1], scores[name]),
                     'latency_batch1': latency_ms(lambda t, m=model: m.predict_proba([t]), data['test'][0])}
    return out, scores, models


def run_transformer(data, model_name, epochs, seed, out_dir):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(6)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def encode(texts):
        return tok(texts, truncation=True, max_length=96, padding=True, return_tensors='pt')

    def predict_scores(texts, batch=64):
        model.eval()
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch):
                out.append(torch.softmax(model(**encode(texts[i:i + batch])).logits, dim=1)[:, 1].numpy())
        return np.concatenate(out)

    x, y = data['dev']
    weights = torch.tensor([1.0, float((y == 0).sum() / (y == 1).sum())])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    steps = epochs * ((len(x) + 31) // 32)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, (steps - s) / steps))
    start, history, order_rng = time.perf_counter(), [], np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        order = order_rng.permutation(len(x))
        total = 0.0
        for i in range(0, len(x), 32):
            idx = order[i:i + 32]
            batch = encode([x[j] for j in idx])
            logits = model(**batch).logits
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor(y[idx]), weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total += float(loss) * len(idx)
        history.append(round(total / len(x), 4))
        print(f'epoch {epoch + 1}/{epochs} loss {history[-1]} elapsed {time.perf_counter() - start:.0f}s', flush=True)
    train_seconds = time.perf_counter() - start
    score = predict_scores(data['test'][0])
    result = {'model': model_name, 'epochs': epochs, 'seed': seed, 'train_rows': len(x), 'train_seconds': round(train_seconds, 1),
              'epoch_loss': history, 'cpu_threads': 6, **metrics(data['test'][1], score), 'f1_ci95': bootstrap_f1(data['test'][1], score),
              'latency_batch1': latency_ms(lambda t: predict_scores([t], 1), data['test'][0], runs=60)}
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)   # weights stay out of git (see .gitignore)
    tok.save_pretrained(out_dir)
    return result, score, predict_scores


def robustness(data, scorers):
    texts, y = data['test']
    spam = [t for t, label in zip(texts, y) if label == 1]
    table = {}
    for name, scorer in scorers.items():
        row = {'clean_spam_recall': round(float((scorer(spam) >= .5).mean()), 4)}
        for pname, perturb in PERTURBATIONS.items():
            row[pname] = round(float((scorer([perturb(t) for t in spam]) >= .5).mean()), 4)
        table[name] = row
    return {'spam_messages': len(spam), 'metric': 'spam recall at threshold 0.5 on perturbed test spam', 'table': table,
            'examples': {p: f(spam[0]) for p, f in PERTURBATIONS.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-transformer', action='store_true')
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, default=ROOT / 'reports' / 'benchmark.json')
    parser.add_argument('--weights-dir', type=Path, default=ROOT / 'models' / 'transformer')
    args = parser.parse_args()
    data, provenance = splits()
    results, scores, models = run_classical(data)
    scorers = {name: (lambda texts, m=model: spam_scores(m, texts)) for name, model in models.items()}
    if not args.skip_transformer:
        results['finetuned_transformer'], scores['finetuned_transformer'], predict = run_transformer(data, args.model, args.epochs, args.seed, args.weights_dir)
        scorers['finetuned_transformer'] = predict
    y = data['test'][1]
    best_classical = max((n for n in results if n != 'finetuned_transformer'), key=lambda n: results[n]['spam_f1'])
    report = {'dataset': provenance, 'split': {k: len(v[0]) for k, v in data.items()}, 'threshold': 0.5, 'results': results,
              'best_classical': best_classical, 'robustness': robustness(data, scorers),
              'environment': {'python': platform.python_version(), 'processor': platform.processor()}}
    if 'finetuned_transformer' in scores:
        report['mcnemar_best_classical_vs_transformer'] = mcnemar(y, scores[best_classical], scores['finetuned_transformer'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('results', 'robustness')}, indent=1)[:3000])


if __name__ == '__main__':
    main()
