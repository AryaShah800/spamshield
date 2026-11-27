import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

try:
    from test_api_and_benchmark import toy_models
    import spam_api  # noqa: F401  (serving extras)
except ImportError:
    toy_models = None

HAS_MATPLOTLIB = importlib.util.find_spec('matplotlib') is not None


@unittest.skipIf(toy_models is None, 'serving extras not installed')
class ExplainTests(unittest.TestCase):
    def test_text_with_no_known_ngrams_explains_nothing(self):
        # Characters the vectorizer never saw produce an empty tf-idf row: no terms, no crash.
        self.assertEqual(toy_models().explain('ꙮꙮꙮ ꙮꙮ'), [])
        self.assertEqual(toy_models().explain(''), [])

    def test_known_text_lists_the_strongest_terms_and_reuses_the_cached_weights(self):
        models = toy_models()
        first = models.explain('win a free prize now', top=3)
        cache = models._explain_cache
        second = models.explain('claim your free cash', top=3)

        self.assertEqual(len(first), 3)
        self.assertTrue(all({'term', 'weight'} <= set(item) for item in first + second))
        weights = [abs(item['weight']) for item in first]
        self.assertEqual(weights, sorted(weights, reverse=True))
        self.assertIs(models._explain_cache, cache)


@unittest.skipUnless(HAS_MATPLOTLIB, 'requirements-analysis.txt not installed')
class CalibrationTests(unittest.TestCase):
    def test_perfect_probabilities_have_zero_calibration_error(self):
        from error_analysis import calibration
        y = np.array([0, 0, 1, 1])
        rows, ece, brier = calibration(y, y.astype(float))
        self.assertEqual(ece, 0.0)
        self.assertEqual(brier, 0.0)
        self.assertEqual(sum(r['n'] for r in rows), 4)

    def test_overconfident_bin_is_measured(self):
        from error_analysis import calibration
        # Ten messages all scored 0.95, half of them spam: one bin, 0.45 calibration error.
        y = np.array([1, 0] * 5)
        rows, ece, brier = calibration(y, np.full(10, 0.95))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['observed_spam_rate'], 0.5)
        self.assertAlmostEqual(ece, 0.45, places=4)
        self.assertAlmostEqual(brier, 0.4525, places=4)


if __name__ == '__main__':
    unittest.main()
