"""Meaningful regression checks; synthetic data only, never final test data."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import day_block_bootstrap, save_outputs
from src.model_b import build_pipeline, NUMERICAL
from src.prepare_data import FEATURES


class Phase3ATests(unittest.TestCase):
    def test_bootstrap_matches_explicit_whole_day_resampling(self):
        # Unequal day sizes distinguish pooled hourly MAE from mean daily MAE.
        frame = pd.DataFrame({'fold': [1, 1, 1, 2, 2, 2],
            'dteday': ['2011-01-01', '2011-01-01', '2011-01-02',
                       '2011-02-01', '2011-02-02', '2011-02-02'],
            'hr': [0, 1, 0, 0, 0, 1], 'absolute_error': [1., 3., 20., 4., 10., 30.]})
        ci, actual = day_block_bootstrap(frame, repetitions=2000, seed=42)
        rng = np.random.default_rng(42)
        selected = []
        for _, group in frame.groupby('fold'):
            blocks = [day.absolute_error.to_numpy() for _, day in group.groupby('dteday')]
            draws = rng.integers(0, len(blocks), size=(2000, len(blocks)))
            selected.append([np.concatenate([blocks[j] for j in draw]) for draw in draws])
        expected = np.array([np.concatenate([selected[0][i], selected[1][i]]).mean() for i in range(2000)])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        np.testing.assert_array_equal([ci['lower'], ci['upper']], np.percentile(expected, [2.5, 97.5]))
        _, again = day_block_bootstrap(frame, repetitions=2000, seed=42)
        np.testing.assert_array_equal(actual, again)

    def test_unknown_category_does_not_refit_preprocessor(self):
        x = pd.DataFrame({name: np.arange(1., 5.) for name in FEATURES})
        estimator = build_pipeline(1.)
        estimator.fit(x, np.array([10., 20., 30., 40.]))
        before = estimator.named_steps['preprocess'].named_transformers_['scale'].mean_.copy()
        validation = x.iloc[[0]].copy()
        validation['hr'] = 99  # Deliberately unseen category.
        validation['temp'] = 10000
        self.assertTrue(np.isfinite(estimator.predict(validation)).all())
        scale = estimator.named_steps['preprocess'].named_transformers_['scale']
        np.testing.assert_array_equal(scale.mean_, before)
        np.testing.assert_array_equal(scale.mean_, x[NUMERICAL].mean())
        encoder = estimator.named_steps['preprocess'].named_transformers_['onehot']
        self.assertNotIn(99, encoder.categories_[0])

    def test_conflict_preflight_prevents_partial_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / 'existing.txt').write_bytes(b'user content')
            with self.assertRaises(ValueError):
                save_outputs(path, {'new.txt': b'new', 'existing.txt': b'replacement'})
            self.assertFalse((path / 'new.txt').exists())
            self.assertEqual((path / 'existing.txt').read_bytes(), b'user content')
            save_outputs(path, {'existing.txt': b'user content'})

    def test_test_data_open_is_blocked_before_read(self):
        # Isolate the permanent audit hook in a subprocess; no real file is opened.
        code = ('import sys; from src.evaluate import deny_test_access; '
                'sys.addaudithook(deny_test_access); open("test.csv", "rb")')
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('PermissionError: Phase 3A must not open', result.stderr)


if __name__ == '__main__':
    unittest.main()
