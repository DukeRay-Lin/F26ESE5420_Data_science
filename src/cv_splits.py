"""Shared date-based expanding folds. This module never reads the final test set."""
import json
from pathlib import Path
import pandas as pd

try:
    from .prepare_data import ROOT, FEATURES, TARGET, METADATA, CUTOFF, sha256, require, save_without_overwrite
except ImportError:
    from prepare_data import ROOT, FEATURES, TARGET, METADATA, CUTOFF, sha256, require, save_without_overwrite

TRAIN_PATH = ROOT / 'data/processed/train.csv'
MANIFEST_PATH = ROOT / 'data/processed/cv_folds.json'


def build_manifest(train_path=TRAIN_PATH):
    train_path = Path(train_path)
    train = pd.read_csv(train_path)
    require(list(train.columns) == FEATURES + [TARGET] + METADATA, 'Unexpected prepared schema')
    require(len(train) == 13891 and not train.isna().any().any(), 'Unexpected training data')
    dates = pd.to_datetime(train.dteday, format='%Y-%m-%d')
    times = pd.to_datetime(train.timestamp)
    require(dates.lt(CUTOFF).all(), 'Final test dates in CV inputs')
    require(times.is_monotonic_increasing and times.is_unique, 'Train order changed')
    require((times == dates + pd.to_timedelta(train.hr, unit='h')).all(), 'Invalid timestamp metadata')
    require(train.source_row_index.tolist() == list(range(len(train))), 'Source row order changed')
    calendar = pd.date_range(dates.min(), dates.max(), freq='D')
    window_days = len(calendar) // 6
    initial_days = len(calendar) - 5 * window_days
    require(window_days > 0, 'Insufficient dates')
    folds, seen_validation = [], set()
    previous_train, previous_validation = set(), set()
    for fold in range(5):
        start = calendar[initial_days + fold * window_days]
        end = calendar[initial_days + (fold + 1) * window_days - 1]
        ti = train.index[dates.lt(start)].tolist()
        vi = train.index[dates.between(start, end)].tolist()
        require(ti and vi, 'Empty fold')
        require(set(ti).isdisjoint(vi), 'Fold overlap')
        require(dates.iloc[ti].max() < dates.iloc[vi].min(), 'Date leakage')
        require(set(dates.iloc[ti]).isdisjoint(dates.iloc[vi]), 'Date split across train/validation')
        require(not seen_validation.intersection(vi), 'Repeated validation row')
        require(previous_train | previous_validation <= set(ti), 'Training window did not expand')
        require(ti == list(range(len(ti))) and vi == list(range(vi[0], vi[-1] + 1)), 'Non-contiguous row window')
        folds.append({'fold': fold + 1, 'train_start': str(dates.iloc[ti].min().date()),
            'train_end': str(dates.iloc[ti].max().date()), 'validation_start': str(start.date()),
            'validation_end': str(end.date()), 'train_count': len(ti), 'validation_count': len(vi),
            'train_indices': ti, 'validation_indices': vi})
        seen_validation.update(vi)
        previous_train, previous_validation = set(ti), set(vi)
    require(seen_validation == set(train.index[dates.ge(calendar[initial_days])]), 'Incomplete validation coverage')
    return {'schema_version': 1, 'strategy': '5-fold expanding window on complete calendar dates',
        'train_file': 'data/processed/train.csv', 'train_sha256': sha256(train_path),
        'index_definition': 'zero-based positional row indices into train.csv; use pandas iloc',
        'feature_columns': FEATURES, 'target_column': TARGET, 'metadata_columns': METADATA,
        'calendar_days': len(calendar), 'initial_train_days': initial_days,
        'validation_window_days': window_days, 'shuffle': False, 'gap_days': 0,
        'preprocessing_rule': 'Fit all learned preprocessing on each fold training subset only; transform validation with that fitted pipeline. Metadata must not enter model inputs.',
        'folds': folds}


def load_cv_splits(train_path=TRAIN_PATH, manifest_path=MANIFEST_PATH):
    """Return identical reusable iloc index pairs, rejecting altered data/config."""
    with open(manifest_path, encoding='utf-8') as stream:
        manifest = json.load(stream)
    require(manifest == build_manifest(train_path), 'CV manifest or train.csv changed; do not reuse mismatched folds')
    return [(f['train_indices'], f['validation_indices']) for f in manifest['folds']]


def main():
    manifest = build_manifest()
    save_without_overwrite(MANIFEST_PATH, (json.dumps(manifest, indent=2) + '\n').encode())
    require(len(load_cv_splits()) == 5, 'CV reload failed')
    for fold in manifest['folds']:
        print({k:v for k,v in fold.items() if not k.endswith('_indices')})


if __name__ == '__main__':
    main()
