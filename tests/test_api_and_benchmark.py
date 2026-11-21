import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import benchmark

try:
    from fastapi.testclient import TestClient
    import spam_api
except ImportError:   # serving extras not installed
    TestClient = None

TEXTS = ['win a free prize now', 'claim your free cash prize', 'free entry win cash', 'urgent claim prize call now',
         'see you at lunch', 'are we meeting tomorrow', 'call me when you get home', 'thanks for the notes today'] * 3
LABELS = [1, 1, 1, 1, 0, 0, 0, 0] * 3


def toy_models(transformer=None):
    pipe = Pipeline([('tfidf', TfidfVectorizer()), ('classifier', LogisticRegression(max_iter=200))]).fit(TEXTS, LABELS)
    return spam_api.Models(pipe, transformer)


class BenchmarkHelpers(unittest.TestCase):
    def test_perturbations_change_text_but_keep_length_class(self):
        text = 'Free prize call now http://www.win.com'
        for name, fn in benchmark.PERTURBATIONS.items():
            self.assertNotEqual(fn(text), text, name)
        self.assertEqual(benchmark.leetspeak('test'), '7357')
        self.assertNotEqual(benchmark.homoglyphs('apple'), 'apple')
        self.assertEqual(benchmark.dotted_words('free prizes'), 'free p.r.i.z.e.s')

    def test_mcnemar_exact(self):
        y = np.array([1, 1, 1, 1, 0, 0])
        a = np.array([.9, .9, .9, .9, .1, .1])      # all correct
        b = np.array([.1, .1, .1, .9, .1, .1])      # wrong on three
        r = benchmark.mcnemar(y, a, b)
        self.assertEqual((r['only_first_correct'], r['only_second_correct']), (3, 0))
        self.assertAlmostEqual(r['p_value'], 0.25)

    def test_bootstrap_interval_is_ordered_and_deterministic(self):
        y = np.array([1, 0] * 50)
        s = np.where(y == 1, .8, .2)
        s[:6] = 1 - s[:6]
        low, high = benchmark.bootstrap_f1(y, s, draws=200)
        self.assertLess(low, high)
        self.assertEqual((low, high), tuple(benchmark.bootstrap_f1(y, s, draws=200)))


@unittest.skipIf(TestClient is None, 'install requirements-serve.txt')
class Api(unittest.TestCase):
    def client(self, models):
        return TestClient(spam_api.create_app(lambda: models))

    def test_predict_returns_probability_and_explanation(self):
        r = self.client(toy_models()).post('/predict', json={'message': 'win a free cash prize now'})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body['logistic_regression']['label'], 'spam')
        self.assertTrue(any(e['weight'] > 0 for e in body['explanation']))
        self.assertNotIn('transformer', body)

    def test_transformer_included_when_available(self):
        models = toy_models(transformer=lambda texts: np.full(len(texts), .7))
        body = self.client(models).post('/predict', json={'message': 'hello there'}).json()
        self.assertEqual(body['transformer'], {'spam_probability': 0.7, 'label': 'spam'})

    def test_probe_lists_original_and_all_perturbations(self):
        rows = self.client(toy_models()).post('/probe', json={'message': 'free prize now'}).json()['rows']
        self.assertEqual([r['variant'] for r in rows], ['original'] + list(benchmark.PERTURBATIONS))

    def test_validation_and_missing_model(self):
        c = self.client(toy_models())
        self.assertEqual(c.post('/predict', json={'message': ''}).status_code, 422)
        self.assertEqual(c.post('/predict', json={'message': 'x' * 1001}).status_code, 422)

        def missing():
            raise FileNotFoundError('Train first: python spamshield.py train')
        c = TestClient(spam_api.create_app(missing))
        self.assertEqual(c.post('/predict', json={'message': 'hi'}).status_code, 503)
        self.assertFalse(c.get('/health').json()['ready'])

    def test_index_page_served(self):
        r = self.client(toy_models()).get('/')
        self.assertEqual(r.status_code, 200)
        self.assertIn('SpamShield', r.text)


if __name__ == '__main__':
    unittest.main()
