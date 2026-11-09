"""Run A/B on the saved folds; never open the final test set.

Run from the repository root: python -B -m src.evaluate --verify-repeat
Outputs refuse to overwrite different existing content. Predictions are not clipped.
"""
import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits

from . import model_a, model_b
from .cv_splits import load_cv_splits, TRAIN_PATH, MANIFEST_PATH
from .prepare_data import FEATURES, ROOT, sha256, require, save_without_overwrite


def deny_test_access(event, args):
    """Audit hook blocks Python file opens of the held-out test data."""
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0]))
        if path.name == 'test.csv' or path.resolve() == ROOT / 'data/processed/test.csv':
            raise PermissionError('Phase 3A must not open the final test set')


def run_folds(data, folds, factory, name):
    rows, predictions = [], []
    for fold_id, (ti, vi) in enumerate(folds, 1):
        tr, va = data.iloc[ti], data.iloc[vi]
        require(tr.dteday.max() < va.dteday.min(), 'Fold time overlap')
        require(set(ti).isdisjoint(vi), 'Fold row overlap')
        pipeline = factory()  # A completely new estimator for every fold/alpha.
        xtr, xva = tr[FEATURES], va[FEATURES]
        pipeline.fit(xtr, tr.cnt)
        np.testing.assert_array_equal(pipeline.feature_names_in_, FEATURES)
        if name == 'A':
            scaler, numeric = pipeline.named_steps['scale'], FEATURES
        else:
            pre = pipeline.named_steps['preprocess']
            scaler, numeric = pre.named_transformers_['scale'], model_b.NUMERICAL
            encoder = pre.named_transformers_['onehot']
            for col, categories in zip(model_b.CATEGORICAL, encoder.categories_):
                np.testing.assert_array_equal(categories, np.sort(tr[col].unique()))
        require(int(scaler.n_samples_seen_) == len(ti), 'Scaler saw non-training rows')
        np.testing.assert_allclose(scaler.mean_, xtr[numeric].mean().to_numpy(), rtol=1e-12, atol=1e-12)
        pred = pipeline.predict(xva)
        require(len(pred) == len(vi) and np.isfinite(pred).all(), 'Invalid predictions')
        error = np.abs(va.cnt.to_numpy() - pred)
        mae = float(mean_absolute_error(va.cnt, pred))
        require(np.isfinite(mae) and mae >= 0, 'Invalid MAE')
        rows.append({'model': name, 'fold': fold_id, 'train_n': len(ti), 'validation_n': len(vi),
                     'train_start': tr.dteday.min(), 'train_end': tr.dteday.max(),
                     'validation_start': va.dteday.min(), 'validation_end': va.dteday.max(), 'mae': mae})
        predictions.append(pd.DataFrame({'model': name, 'fold': fold_id, 'train_row_index': vi,
            'source_row_index': va.source_row_index.to_numpy(), 'dteday': va.dteday.to_numpy(),
            'hr': va.hr.to_numpy(), 'timestamp': va.timestamp.to_numpy(), 'y_true': va.cnt.to_numpy(),
            'y_pred': pred, 'absolute_error': error}))
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def day_block_bootstrap(predictions, repetitions=5000, seed=42):
    """Stratify by fold; resample observed whole days, retaining every row in a day.

    A block is represented by its error sum and count, exactly equivalent to
    concatenating all hourly errors for that day. Each fold draws its original
    number of days with replacement. MAE is pooled error sum / pooled row count,
    not an unweighted average of daily MAEs. Missing hours remain missing.
    """
    require(repetitions >= 2000, 'At least 2000 resamples required')
    require(not predictions.duplicated(['fold', 'dteday', 'hr']).any(), 'Duplicate prediction hour')
    require(predictions.groupby('dteday').fold.nunique().max() == 1, 'Date crosses folds')
    require(np.isfinite(predictions.absolute_error).all() and predictions.absolute_error.ge(0).all(), 'Invalid errors')
    rng = np.random.default_rng(seed)
    totals, counts = np.zeros(repetitions), np.zeros(repetitions, dtype=np.int64)
    strata = []
    for fold, group in predictions.groupby('fold', sort=True):
        blocks = group.groupby('dteday', sort=True).absolute_error.agg(['sum', 'count'])
        sampled = rng.integers(0, len(blocks), size=(repetitions, len(blocks)))
        totals += blocks['sum'].to_numpy()[sampled].sum(axis=1)
        counts += blocks['count'].to_numpy()[sampled].sum(axis=1)
        strata.append({'fold': int(fold), 'days': len(blocks), 'observations': len(group),
                       'min_day_observations': int(blocks['count'].min()),
                       'max_day_observations': int(blocks['count'].max())})
    values = totals / counts
    limits = np.percentile(values, [2.5, 97.5], method='linear')
    return {'method': 'Fold-stratified whole-day percentile bootstrap of pooled validation MAE',
            'seed': seed, 'repetitions': repetitions, 'confidence': 0.95,
            'lower': float(limits[0]), 'upper': float(limits[1]), 'strata': strata,
            'scope': 'CV validation only; conditional on selected alpha and fitted fold models; no retraining or alpha reselection',
            'limitations': 'Not a final-test interval. Does not account for alpha selection or dependence across days and overlapping training windows.'}, values


def summarize(cv, predictions):
    return {'fold_mae': cv.mae.tolist(), 'mean_fold_mae': float(cv.mae.mean()),
            'sample_sd_fold_mae': float(cv.mae.std(ddof=1)),
            'oof_mae': float(predictions.absolute_error.mean()),
            'oof_observations': len(predictions), 'negative_predictions': int(predictions.y_pred.lt(0).sum())}


def compute():
    data = pd.read_csv(TRAIN_PATH)
    folds = load_cv_splits()
    require(np.isfinite(data[FEATURES + ['cnt']].to_numpy()).all(), 'Nonfinite model input')
    expected_indices = [i for _, vi in folds for i in vi]
    require(len(expected_indices) == len(set(expected_indices)) == 11608, 'Unexpected OOF coverage')
    acv, apred = run_folds(data, folds, model_a.build_pipeline, 'A')
    tuning, best = [], None
    for alpha in model_b.ALPHAS:
        bcv, bpred = run_folds(data, folds, lambda: model_b.build_pipeline(alpha), 'B')
        mean = float(bcv.mae.mean())
        tuning.append({'alpha': float(alpha), **{f'fold_{i}_mae': float(v) for i, v in enumerate(bcv.mae, 1)},
                       'mean_mae': mean, 'sample_sd_mae': float(bcv.mae.std(ddof=1))})
        if best is None or mean < best[0]:
            best = (mean, float(alpha), bcv, bpred)
    _, best_alpha, bcv, bpred = best
    bcv['alpha'] = best_alpha
    tune = pd.DataFrame(tuning)
    tune['rank'] = tune.mean_mae.rank(method='min').astype(int)
    tune['selected'] = tune.alpha.eq(best_alpha)
    for predictions in [apred, bpred]:
        require(predictions.train_row_index.tolist() == expected_indices, 'OOF index mismatch')
        require(not predictions.isna().any().any(), 'Null OOF records')
        require(np.isfinite(predictions[['y_true', 'y_pred', 'absolute_error']]).all().all(), 'Nonfinite OOF')
        for fold_id, (_, vi) in enumerate(folds, 1):
            saved = predictions[predictions.fold == fold_id]
            require(saved.train_row_index.tolist() == vi, 'Fold prediction order mismatch')
            np.testing.assert_array_equal(saved.y_true, data.iloc[vi].cnt)
            np.testing.assert_array_equal(saved.dteday, data.iloc[vi].dteday)
            np.testing.assert_array_equal(saved.hr, data.iloc[vi].hr)
    all_predictions = pd.concat([apred, bpred], ignore_index=True)
    hourly = all_predictions.groupby(['model', 'hr'], sort=True).absolute_error.agg(n='size', mae='mean').reset_index()
    require(len(hourly) == 48 and hourly.hr.nunique() == 24, 'Missing hourly metrics')
    ci, samples = day_block_bootstrap(bpred)
    unknown = []
    for fid, (ti, vi) in enumerate(folds, 1):
        unknown.append({'fold': fid, 'validation_categories_absent_from_training': {
            col: sorted(set(data.iloc[vi][col]) - set(data.iloc[ti][col])) for col in model_b.CATEGORICAL}})
    metrics = {'units': 'bikes/hour', 'evaluation': 'Expanding-window CV validation; not final test',
        'train_sha256': sha256(TRAIN_PATH), 'cv_manifest_sha256': sha256(MANIFEST_PATH),
        'features': FEATURES, 'categorical_b': model_b.CATEGORICAL, 'numerical_b': model_b.NUMERICAL,
        'model_a': summarize(acv, apred), 'model_b': {**summarize(bcv, bpred), 'best_alpha': best_alpha},
        'alpha_grid': model_b.ALPHAS.tolist(), 'selection_rule': 'minimum unweighted mean fold MAE; smaller alpha breaks exact ties',
        'bootstrap_b': ci, 'prediction_rows': len(all_predictions), 'unique_validation_rows': len(expected_indices),
        'unknown_categories': unknown, 'predictions_clipped': False,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                     'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__},
        'runtime': {'blas_threads': 1, 'ridge_solver': 'svd', 'bootstrap_rng': 'numpy.default_rng(42)'}}
    frames = {'model_a_cv.csv': acv, 'model_b_cv.csv': bcv, 'model_b_tuning.csv': tune,
              'validation_predictions.csv': all_predictions, 'validation_mae_by_hour.csv': hourly,
              'model_b_bootstrap.csv': pd.DataFrame({'replicate': np.arange(1, len(samples)+1), 'mae': samples})}
    payloads = {name: frame.to_csv(index=False, lineterminator='\n').encode() for name, frame in frames.items()}
    payloads['baseline_metrics.json'] = (json.dumps(metrics, indent=2, allow_nan=False) + '\n').encode()
    return metrics, payloads


def report(metrics, repeated):
    a, b, ci = metrics['model_a'], metrics['model_b'], metrics['bootstrap_b']
    lines = ['# ESE 5420 — Phase 3A Model A/B Baseline', '', '**Overall Status: PASS WITH WARNINGS**', '',
        '## Results / 实测结果', '', '全部误差单位为 bikes/hour；以下均为 CV 验证结果，不是最终测试集结果。', '',
        '| Fold | Model A MAE | Model B MAE |', '| --- | ---: | ---: |']
    lines += [f'| {i} | {av:.8f} | {bv:.8f} |' for i, (av, bv) in enumerate(zip(a['fold_mae'], b['fold_mae']), 1)]
    lines += [f'| Mean（五折等权） | {a["mean_fold_mae"]:.8f} | {b["mean_fold_mae"]:.8f} |',
              f'| SD（样本标准差，ddof=1） | {a["sample_sd_fold_mae"]:.8f} | {b["sample_sd_fold_mae"]:.8f} |',
              f'| Pooled OOF MAE（按小时样本加权） | {a["oof_mae"]:.8f} | {b["oof_mae"]:.8f} |', '',
        f'- Model B 最佳 alpha：`{b["best_alpha"]:.17g}`。固定网格 `np.logspace(-4, 4, 25)`，以五折平均 MAE 最小选择，未扩展网格。',
        f'- B 的 95% Percentile CI：**[{ci["lower"]:.8f}, {ci["upper"]:.8f}]**。',
        '- A/B 每个模型 11,608 条预测，共 23,216 条长表记录；对应相同的 11,608 个未参与当折拟合的验证样本。最早 2,283 行仅用于初始训练，无 OOF 预测。',
        '- 预测不截断、不取整。A/B 负预测数量分别为 '+str(a['negative_predictions'])+' / '+str(b['negative_predictions'])+'。', '',
        '## Methods / 方法与泄漏控制', '',
        '- 只读取 train.csv，经 src.cv_splits.load_cv_splits() 加载并验证既有索引。未调用 Phase 2 生成程序，未重划分。',
        '- A：原始 12 个数值特征 → StandardScaler → LinearRegression。B：hr、weekday、mnth、season、weathersit 使用 OneHotEncoder(handle_unknown="ignore")；其他 7 列 StandardScaler；ColumnTransformer → Ridge(solver="svd")。',
        '- 每个模型、每个 alpha、每折均新建 Pipeline，仅在该折训练行 fit。检查 scaler 的样本数和均值，并检查编码器类别恰好来自该折训练行。日期、来源索引和目标没有进入 X。',
        '- 所有 alpha 与 A 共用同一 folds 对象；每折训练日期严格早于验证日期。早期未知类别使用全零编码；具体缺失类别见 baseline_metrics.json。',
        '- Python audit hook 阻止打开 test.csv。整个训练、Bootstrap 和重复运行均启用该保护；未计算最终测试 MAE。',
        '- 固定 BLAS 线程数为 1；Bootstrap seed=42；没有训练 C/C+/B+，也未做最终全训练集重拟合。', '',
        '## Bootstrap / 区间定义', '',
        '- 5,000 次重采样。每折对该折 97 个完整日期块有放回抽取 97 次，再合并五折。每个日期块保留该日所有实际观测小时，未补齐缺失小时。',
        '- 每日以绝对误差总和与样本数表示；每次计算所抽日期块的误差总和 / 小时总数，精确等价于整日拼接。不是逐小时独立抽样，也不是对每天 MAE 做等权平均。',
        '- 对 5,000 个 MAE 使用 2.5% 和 97.5% 分位数（linear interpolation）。完整重采样结果保存到 model_b_bootstrap.csv。',
        '- WARNING：最佳 alpha 在同一 CV 上选择和报告，B 的结果可能有选择乐观偏差；这不是嵌套 CV 的无偏泛化估计。',
        '- WARNING：CI 条件于已选 alpha 和固定折模型，没有重新训练或重新选择 alpha，未覆盖跨日相关性及重叠训练窗口的不确定性。不能解释为最终测试集 CI。', '',
        '- 网格边界检查：'+('WARNING：最佳 alpha 位于固定网格边界，不能据此宣称找到了网格以外的最优值；本阶段没有扩展搜索。' if b['best_alpha'] in [metrics['alpha_grid'][0], metrics['alpha_grid'][-1]] else 'PASS：最佳 alpha 位于网格内部。'), '',
        '## Verification / 核验', '', '| 项目 | 状态 |', '| --- | --- |',
        '| 已有 CV 哈希、索引、五折时间顺序及 A/B 相同验证行 | PASS |',
        '| 各折预测长度、日期、小时、真实值与源行一致 | PASS |',
        '| 输入及预测无 NaN/inf，MAE 非负，24 小时指标齐全 | PASS |',
        '| 每折独立拟合缩放和类别编码，无元数据或目标输入泄漏 | PASS |',
        '| 分层整日 Bootstrap，5,000 次，固定 seed=42 | PASS |',
        '| test.csv 文件打开保护启用；未做最终测试评估 | PASS |',
        f'| 两次完整训练、调参和 Bootstrap 的 7 个数值产物逐字节一致 | {"PASS" if repeated else "WARNING: not requested"} |',
        '| 安全保存：预检全部目标；内容不一致则拒绝覆盖 | PASS |', '',
        f'以上执行核验：{8 if repeated else 7} PASS，0 FAIL；方法学限制及网格边界合计 {2 + int(b["best_alpha"] in [metrics["alpha_grid"][0], metrics["alpha_grid"][-1]])} WARNING'+('。' if repeated else '，重复运行未验证 1 WARNING。'), '',
        '## Reproduction / 复现', '',
        '从项目根目录执行，Python 3.12；安装 requirements-phase3a.txt 中固定版本。', '', '```bash',
        'python -m pip install -r requirements-phase3a.txt',
        'python -B -m src.evaluate --verify-repeat',
        'python -B -m unittest discover -s tests -p "test_phase3a.py" -v', '```', '',
        '本次实际运行解释器：`'+sys.executable+'`。',
        '环境：`'+json.dumps(metrics['versions'], ensure_ascii=False)+'`。',
        '参考 API：[OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)、[Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)。本次行为以已安装版本与实际执行为准。',
        '现有环境缺少 sklearn，已在临时虚拟环境安装依赖，没有修改共享运行环境。',
        '重复执行不覆盖不同内容；如环境或代码变化产生不同结果，应指定新的 `--output-dir`，保留既有记录。', '',
        '## Artifacts / 文件', '',
        '- src/model_a.py、src/model_b.py：模型 Pipeline 工厂。src/evaluate.py：统一训练、调参、Bootstrap、检查及安全输出。',
        '- model_a_cv.csv、model_b_cv.csv：五折 MAE、样本数和日期范围。',
        '- model_b_tuning.csv：25 个 alpha 的各折 MAE、平均值、样本 SD、排名和选择标识。',
        '- validation_predictions.csv：模型、Fold、训练集位置、原始来源位置、日期、小时、时间、真实值、预测及绝对误差。',
        '- validation_mae_by_hour.csv：A/B 各小时 MAE 及样本数，共 48 行。',
        '- model_b_bootstrap.csv：5,000 次重采样 MAE。baseline_metrics.json：核心指标、CI、配置、版本及输入哈希。',
        '- tests/test_phase3a.py：整日抽样、折内预处理、测试集保护与安全保存的回归检查。',
        '- requirements-phase3a.txt：固定依赖版本。本报告为 phase3a_report.md。', '',
        '## Next step / 下一步', '',
        '将上述真实 CV 结果用于 11/9 Baseline Checkpoint；后续获授权再开发其他模型，并继续使用同一 folds。保留最终测试集隔离。README、Checkpoint.md、Test.py 和 Phase 2 文件不作修改；未自动提交或推送 Git。', '']
    return '\n'.join(lines).encode()


def save_outputs(folder, payloads):
    # Check all conflicts before writing anything, avoiding partial mixed runs.
    folder = Path(folder)
    for name, content in payloads.items():
        path = folder / name
        require(not path.exists() or path.read_bytes() == content, f'Refusing to overwrite different content: {path}')
    for name, content in payloads.items():
        save_without_overwrite(folder / name, content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-repeat', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    sys.addaudithook(deny_test_access)
    with threadpool_limits(limits=1):
        metrics, payloads = compute()
        print('A/B training, alpha search and 5,000 day-block resamples complete.', flush=True)
        if args.verify_repeat:
            _, repeat = compute()
            require(payloads == repeat, 'Full repeat produced different outputs')
            print('Full repeat: all numeric artifacts are byte-identical.', flush=True)
    payloads['phase3a_report.md'] = report(metrics, args.verify_repeat)
    save_outputs(args.output_dir, payloads)
    print(json.dumps({k:metrics[k] for k in ['model_a', 'model_b', 'bootstrap_b']}, indent=2))


if __name__ == '__main__':
    main()
