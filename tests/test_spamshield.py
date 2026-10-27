"""Offline checks for data integrity and the complete model artifact workflow."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import spamshield


class SpamShieldTests(unittest.TestCase):
    def test_duplicates_and_conflicting_annotations(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / 'messages'
            path.write_text('ham\tHello there\nham\t HELLO   THERE\n'
                            'spam\tconflict\nham\tCONFLICT\nspam\tWin a prize\n', encoding='utf-8')
            rows, meta = spamshield.load_data(path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(meta['raw_rows'], 5)
        self.assertEqual(meta['conflicting_groups_removed'], 1)

    def test_invalid_label_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / 'messages'
            path.write_text('unknown\tHello\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                spamshield.load_data(path)

    def test_train_save_reload_and_predict(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            lines = [f'ham\tFamily dinner meeting schedule {i}' for i in range(80)]
            lines += [f'spam\tWin cash prize claim reward {i}' for i in range(40)]
            (root / 'data' / 'SMSSpamCollection').write_text('\n'.join(lines), encoding='utf-8')
            with patch.object(spamshield, 'ROOT', root), contextlib.redirect_stdout(io.StringIO()):
                spamshield.train()
                spamshield.predict('Family dinner meeting')
                with self.assertRaises(ValueError):
                    spamshield.predict('  ')
            metrics = json.loads((root / 'reports' / 'metrics.json').read_text())
            self.assertEqual(sum(metrics['split_sizes'].values()), 120)
            self.assertEqual(sum(map(sum, metrics['confusion_matrix'])), 24)
            self.assertTrue((root / 'models' / 'spamshield.joblib').is_file())


if __name__ == '__main__':
    unittest.main()
