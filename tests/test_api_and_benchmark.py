import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
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
    pipe = Pipeline([('v', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))),
                     ('m', CalibratedClassifierCV(LinearSVC(), cv=3))]).fit(TEXTS, LABELS)
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
        self.assertEqual(body['char_svm']['label'], 'spam')
        self.assertTrue(any(e['weight'] > 0 for e in body['explanation']))
        self.assertNotIn('transformer', body)

    def test_primary_is_calibrated_char_svm_with_valid_probabilities(self):
        models = toy_models()
        self.assertIsInstance(models.primary.named_steps['m'], CalibratedClassifierCV)
        self.assertEqual(models.primary.named_steps['v'].analyzer, 'char_wb')
        p = models.primary_probability(['win a free cash prize now', 'see you at lunch'])
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertGreater(p[0], .5)
        self.assertLess(p[1], .5)

    def test_shipped_model_spec_is_calibrated_linear_svc(self):
        builders, _ = benchmark.classical_candidates()
        model = builders['tfidf_char_svm'](1)
        self.assertIsInstance(model.named_steps['m'], CalibratedClassifierCV)
        self.assertIsInstance(model.named_steps['m'].estimator, LinearSVC)

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


@unittest.skipIf(TestClient is None, 'install requirements-serve.txt')
class Hardening(unittest.TestCase):
    def client(self, **kwargs):
        return TestClient(spam_api.create_app(lambda: toy_models(), **kwargs))

    def test_oversized_body_rejected_before_parsing(self):
        r = self.client().post('/predict', content=b'{"message": "' + b'a' * 20000 + b'"}',
                               headers={'Content-Type': 'application/json'})
        self.assertEqual(r.status_code, 413)
        self.assertIn('detail', r.json())

    def test_rate_limit_returns_429_with_retry_after(self):
        c = self.client(rate_limit=3)
        codes = [c.post('/predict', json={'message': 'hello there'}).status_code for _ in range(5)]
        self.assertEqual(codes, [200, 200, 200, 429, 429])
        r = c.post('/predict', json={'message': 'hello there'})
        self.assertIn('retry-after', r.headers)
        self.assertIn('detail', r.json())

    def test_health_and_index_are_not_rate_limited(self):
        c = self.client(rate_limit=1)
        self.assertTrue(all(c.get('/health').status_code == 200 for _ in range(5)))
        self.assertEqual(c.get('/').status_code, 200)

    def test_rate_limit_is_per_client_address(self):
        c = self.client(rate_limit=1)
        a = {'X-Forwarded-For': '203.0.113.1'}
        b = {'X-Forwarded-For': '203.0.113.2'}
        self.assertEqual(c.post('/predict', json={'message': 'hi'}, headers=a).status_code, 200)
        self.assertEqual(c.post('/predict', json={'message': 'hi'}, headers=a).status_code, 429)
        self.assertEqual(c.post('/predict', json={'message': 'hi'}, headers=b).status_code, 200)

    def test_rate_limit_window_expires(self):
        now = [1000.0]
        c = self.client(rate_limit=1, clock=lambda: now[0])
        self.assertEqual(c.post('/predict', json={'message': 'hi'}).status_code, 200)
        self.assertEqual(c.post('/predict', json={'message': 'hi'}).status_code, 429)
        now[0] += 61
        self.assertEqual(c.post('/predict', json={'message': 'hi'}).status_code, 200)

    def test_health_reports_ready_for_probes(self):
        body = self.client().get('/health').json()
        self.assertEqual((body['status'], body['ready']), ('ok', True))


@unittest.skipIf(TestClient is None, 'install requirements-serve.txt')
class ShippedModel(unittest.TestCase):
    """The committed models/spamshield_svm.joblib that the Docker image serves (no transformer, no retraining)."""

    def test_committed_model_serves_through_the_real_provider(self):
        path = spam_api.ROOT / 'models' / 'spamshield_svm.joblib'
        self.assertTrue(path.exists(), 'models/spamshield_svm.joblib must be committed')
        self.assertLess(path.stat().st_size, 5 * 1024 * 1024)
        client = TestClient(spam_api.create_app(lambda: spam_api.Models(spam_api.joblib.load(path))))
        spam = client.post('/predict', json={'message': 'WINNER!! You have won a 1000 prize. Call 09061701461 now to claim'}).json()
        ham = client.post('/predict', json={'message': 'Are we still meeting for lunch tomorrow?'}).json()
        self.assertEqual((spam['char_svm']['label'], ham['char_svm']['label']), ('spam', 'ham'))


if __name__ == '__main__':
    unittest.main()
