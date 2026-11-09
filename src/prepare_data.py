"""Validate raw data and save chronological inputs without fitting preprocessing."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_SHA256 = 'e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f'
FEATURES = 'season yr mnth hr holiday weekday workingday weathersit temp atemp hum windspeed'.split()
TARGET = 'cnt'
METADATA = ['dteday', 'timestamp', 'source_row_index']
CUTOFF = pd.Timestamp('2012-08-07')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_without_overwrite(path, content):
    """Allow identical reruns, but never replace different existing work."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == content, f'Refusing to overwrite different content: {path}')
        return
    with path.open('xb') as stream:
        stream.write(content)


def prepare():
    raw = ROOT / 'data/raw/hour.csv'
    require(sha256(raw) == RAW_SHA256, 'Raw CSV differs from Phase 1 SHA-256')
    frame = pd.read_csv(raw, encoding='utf-8')
    require(frame.shape == (17379, 17), 'Unexpected raw dimensions')
    require(list(frame.columns) == ['instant', 'dteday'] + FEATURES + ['casual', 'registered', TARGET], 'Unexpected raw schema')
    require(not frame.isna().any().any(), 'Missing CSV cells')
    require(not frame.instant.duplicated().any(), 'Duplicate instant')
    require(not frame.duplicated(['dteday', 'hr']).any(), 'Duplicate date/hour')
    require((frame.cnt == frame.casual + frame.registered).all(), 'Inconsistent cnt')
    require(frame.hr.between(0, 23).all(), 'Invalid hour')
    dates = pd.to_datetime(frame.dteday, format='%Y-%m-%d', errors='raise')
    times = dates + pd.to_timedelta(frame.hr, unit='h')
    require(times.is_monotonic_increasing and times.is_unique, 'Input is not strictly chronological')
    grid = pd.date_range('2011-01-01', '2012-12-31 23:00', freq='h')
    missing = grid.difference(times)
    require(len(missing) == 165 and len(pd.DatetimeIndex(times).difference(grid)) == 0, 'Unexpected time coverage')
    require(frame.weathersit.isin([1, 2, 3, 4]).all(), 'Unexpected weather category')
    merged = int(frame.weathersit.eq(4).sum())
    require(merged == 3, 'Unexpected weather merge count')
    output = frame[FEATURES + [TARGET]].copy()
    output['weathersit'] = output.weathersit.replace({4: 3})
    output['dteday'] = frame.dteday
    output['timestamp'] = times.dt.strftime('%Y-%m-%d %H:%M:%S')
    output['source_row_index'] = frame.index  # Original CSV data row, zero-based.
    train_mask = dates.lt(CUTOFF)
    test_mask = dates.ge(CUTOFF)
    require(not (train_mask & test_mask).any() and (train_mask | test_mask).all(), 'Invalid partition')
    train, test = output.loc[train_mask], output.loc[test_mask]
    require((len(train), len(test)) == (13891, 3488), 'Unexpected split counts')
    require(train.timestamp.max() < test.timestamp.min(), 'Overlapping split times')
    require(set(train.source_row_index).isdisjoint(test.source_row_index), 'Overlapping source rows')
    for name, subset in [('train', train), ('test', test)]:
        save_without_overwrite(ROOT / f'data/processed/{name}.csv', subset.to_csv(index=False, lineterminator='\n').encode())
    require(sha256(raw) == RAW_SHA256, 'Raw file changed')
    summary = {'raw_rows': len(frame), 'raw_columns': len(frame.columns), 'processed_rows': len(output),
               'processed_columns': list(output.columns), 'features': FEATURES, 'target': TARGET,
               'metadata_columns': METADATA, 'merged_weather_rows': merged, 'missing_hours_not_filled': len(missing),
               'raw_sha256': sha256(raw), 'cutoff': str(CUTOFF.date()), 'partitions': {}}
    for name, subset in [('train', train), ('test', test)]:
        summary['partitions'][name] = {'rows': len(subset), 'date_start': subset.dteday.min(),
            'date_end': subset.dteday.max(), 'timestamp_start': subset.timestamp.min(),
            'timestamp_end': subset.timestamp.max(), 'sha256': sha256(ROOT / f'data/processed/{name}.csv')}
    save_without_overwrite(ROOT / 'results/preparation_summary.json', (json.dumps(summary, indent=2) + '\n').encode())
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    prepare()
