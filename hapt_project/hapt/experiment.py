import collections
import gzip
import csv
import importlib.metadata
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import psutil
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from .common import ResourceMonitor, digest_file, digest_value, make_logger, read_csv, write_csv, write_json
from .data import DataStore
from .features import FeatureStore, feature_names, extract_features
from .metrics import evaluate_predictions
from .selection import fit_statistics, select_features


def noise_spec(configuration, stage, fold, root, snr, noise_type='gaussian', replicate=0):
    return {'stage': stage, 'fold': fold, 'root': root, 'snr': snr, 'type': noise_type, 'replicate': replicate}


def model_key(method, k, weight):
    return f'{method}_K{k}_L{weight:g}'.replace('.', 'p')


def candidates(configuration):
    result = []
    for method in configuration['methods']:
        sizes = [0] if method['id'] == 'B0' else configuration['selection']['k_grid']
        weights = configuration['selection']['lambda_grid'] if isinstance(method['lambda'], str) else [method['lambda']]
        for size in sizes:
            for weight in weights:
                result.append({'key': model_key(method['id'], size, weight), 'method': method['id'], 'k': size, 'lambda': weight, 'beta': method['beta']})
    return result


def train_model(features, targets, statistics, candidate, seed, configuration):
    selection_started = time.perf_counter()
    if candidate['method'] == 'B0':
        selected = statistics['indices']
        trace = []
    else:
        selected, trace = select_features(statistics, candidate['k'], candidate['lambda'], candidate['beta'], configuration)
    selection_seconds = time.perf_counter() - selection_started
    parameters = dict(configuration['classifier']['parameters'], random_state=seed)
    pipeline = Pipeline([('standardize', StandardScaler()), ('classify', LinearSVC(**parameters))])
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always', ConvergenceWarning)
        pipeline.fit(features[:, selected], targets)
    retries = 0
    if any(issubclass(item.category, ConvergenceWarning) for item in captured):
        retries = 1
        pipeline.set_params(classify__max_iter=configuration['classifier']['convergence_retry_max_iter'])
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always', ConvergenceWarning)
            pipeline.fit(features[:, selected], targets)
        if any(issubclass(item.category, ConvergenceWarning) for item in captured):
            raise RuntimeError('LinearSVC failed to converge after documented retry')
    timing = {'selection_seconds': selection_seconds, 'classifier_fit_seconds': time.perf_counter() - started, 'convergence_retries': retries}
    return {'pipeline': pipeline, 'selected_indices': selected, 'trace': trace, 'configuration': candidate, 'training_seed': seed, 'timing': timing}


def statistics_for(data, store, records, seed, fold, configuration):
    clean = store.get(records)
    power = data.reference_power(records)
    stability = configuration['training_statistics']['stability']
    noisy = [store.get(records, power, noise_spec(configuration, 'training_stability', fold, seed, stability['reference_snr_db'], replicate=replicate))['features'] for replicate in stability['replicates']]
    statistics = fit_statistics(clean['features'], clean['targets'], noisy, seed, configuration)
    return clean, power, statistics


def save_statistics(directory, statistics, records, power):
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / 'statistics.npz', **statistics)
    write_json(directory / 'fit_provenance.json', {'fit_subject_ids': sorted({record['subject_id'] for record in records}), 'fit_record_ids': [record['record_id'] for record in records], 'reference_power': power, 'statistics_seconds': statistics['fit_seconds']})


def choose(rows, configuration):
    best = max(row['objective'] for row in rows)
    eligible = [row for row in rows if row['objective'] >= best - configuration['model_selection']['score_tie_tolerance']]
    return sorted(eligible, key=lambda row: (-row['clean_score'], row['k'], row['lambda']))[0]


def run_cv(data, store, configuration, result, logger):
    candidate_list = candidates(configuration)
    rows = []
    subject_rows = []
    selection_rows = []
    development = set(configuration['split']['development_subjects'])
    for seed in configuration['randomness']['training_roots']:
        for fold_id, validation_subjects in enumerate(configuration['split']['validation_subject_folds']):
            training_subjects = sorted(development - set(validation_subjects))
            assert not set(training_subjects) & set(validation_subjects)
            training_records = data.subset_records(training_subjects)
            validation_records = data.subset_records(validation_subjects)
            logger.info('CV seed=%s fold=%s: fitting statistics on %s subjects only', seed, fold_id, len(training_subjects))
            clean, power, statistics = statistics_for(data, store, training_records, seed, fold_id, configuration)
            save_statistics(result / 'fit_statistics' / f'cv_seed{seed}_fold{fold_id}', statistics, training_records, power)
            validation_clean = store.get(validation_records)
            conditions = [('clean', 0, validation_clean)]
            for snr in configuration['noise']['validation']['snr_db']:
                for noise_root in configuration['randomness']['validation_noise_roots']:
                    batch = store.get(validation_records, power, noise_spec(configuration, 'validation', fold_id, noise_root, snr))
                    conditions.append((str(snr), noise_root, batch))
            for candidate in candidate_list:
                model = train_model(clean['features'], clean['targets'], statistics, candidate, seed, configuration)
                metrics_by_condition = collections.defaultdict(list)
                for condition, noise_root, batch in conditions:
                    predictions = model['pipeline'].predict(batch['features'][:, model['selected_indices']])
                    metrics, subjects, _ = evaluate_predictions(batch['targets'], predictions, batch['subjects'], configuration)
                    metrics_by_condition[condition].append(metrics['subject_macro_f1'])
                    for subject in subjects:
                        subject_rows.append({'training_seed': seed, 'fold_id': fold_id, 'candidate': candidate['key'], 'method': candidate['method'], 'k': candidate['k'], 'lambda': candidate['lambda'], 'condition': condition, 'noise_root': noise_root, **subject})
                objective = sum(weight * np.mean(metrics_by_condition[condition]) for condition, weight in configuration['implementation']['validation_weights'].items())
                rows.append({'training_seed': seed, 'fold_id': fold_id, **candidate, 'objective': float(objective), 'clean_score': float(metrics_by_condition['clean'][0]), 'valid_dimension': len(statistics['indices']), 'selected_dimension': len(model['selected_indices']), **model['timing']})
                selection_rows.append({'training_seed': seed, 'fold_id': fold_id, 'candidate': candidate['key'], 'indices': model['selected_indices'].tolist(), 'trace': model['trace']})
            logger.info('CV completed seed=%s fold=%s, candidates=%s', seed, fold_id, len(candidate_list))
    write_csv(result / 'tables' / 'cv_fits.csv', rows)
    write_csv(result / 'tables' / 'cv_subject_metrics.csv', subject_rows)
    write_json(result / 'fit_statistics' / 'cv_selected_features.json', selection_rows)
    averaged = []
    for seed in configuration['randomness']['training_roots']:
        for candidate in candidate_list:
            selected_rows = [row for row in rows if row['training_seed'] == seed and row['key'] == candidate['key']]
            averaged.append({'training_seed': seed, **candidate, 'objective': float(np.mean([row['objective'] for row in selected_rows])), 'clean_score': float(np.mean([row['clean_score'] for row in selected_rows]))})
    write_csv(result / 'tables' / 'cv_candidate_scores.csv', averaged)
    return averaged


def source_hashes(project):
    return {str(path.relative_to(project)): digest_file(path) for path in sorted(project.rglob('*.py'))}


def final_training(data, store, configuration, result, project, averaged, logger):
    frozen_models = []
    timings = []
    chosen_rows = []
    training_records = data.subset_records(configuration['split']['development_subjects'])
    final_fold = configuration['randomness']['full_training_and_test_fold_id']
    for seed in configuration['randomness']['training_roots']:
        clean, power, statistics = statistics_for(data, store, training_records, seed, final_fold, configuration)
        save_statistics(result / 'fit_statistics' / f'final_seed{seed}', statistics, training_records, power)
        selected_candidates = {}
        for method in configuration['methods']:
            method_rows = [row for row in averaged if row['training_seed'] == seed and row['method'] == method['id']]
            chosen = choose(method_rows, configuration)
            chosen_rows.append({'role': 'deployment', **chosen})
            selected_candidates.setdefault(chosen['key'], {'candidate': chosen, 'views': []})['views'].append('deployment')
            if method['id'] != 'B0':
                for size in configuration['selection']['k_grid']:
                    matched = choose([row for row in method_rows if row['k'] == size], configuration)
                    chosen_rows.append({'role': f'matched_k{size}', **matched})
                    selected_candidates.setdefault(matched['key'], {'candidate': matched, 'views': []})['views'].append(f'matched_k{size}')
            else:
                selected_candidates[chosen['key']]['views'] += [f'matched_k{size}' for size in configuration['selection']['k_grid']]
        diagnostic = configuration['model_selection']['fixed_lambda_diagnostic']
        if diagnostic['enabled']:
            for method in configuration['methods']:
                if method['id'] not in diagnostic['methods']:
                    continue
                for size in diagnostic['k']:
                    fixed = next(row for row in averaged if row['training_seed'] == seed and row['method'] == method['id'] and row['k'] == size and row['lambda'] == diagnostic['lambda'])
                    selected_candidates.setdefault(fixed['key'], {'candidate': fixed, 'views': []})['views'].append(f'fixed_lambda_k{size}')
        for item in selected_candidates.values():
            candidate = item['candidate']
            model = train_model(clean['features'], clean['targets'], statistics, candidate, seed, configuration)
            identifier = f'seed{seed}_{candidate["key"]}'
            model['fit_subject_ids'] = configuration['split']['development_subjects']
            model['reference_power'] = power
            model['feature_names'] = [feature_names(configuration)[index] for index in model['selected_indices']]
            model['views'] = item['views']
            path = result / 'models' / f'{identifier}.joblib'
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, path)
            write_json(result / 'models' / f'{identifier}.json', {key: value for key, value in model.items() if key != 'pipeline'})
            frozen_models.append({'model_id': identifier, 'relative_path': str(path.relative_to(result)), 'sha256': digest_file(path), 'training_seed': seed, 'method': candidate['method'], 'k': candidate['k'], 'lambda': candidate['lambda'], 'effective_dimension': len(model['selected_indices']), 'views': item['views']})
            timings.append({'model_id': identifier, 'method': candidate['method'], 'training_seed': seed, 'k': candidate['k'], 'lambda': candidate['lambda'], 'windows': len(clean['targets']), 'selected_dimension': len(model['selected_indices']), **model['timing']})
        logger.info('Final training completed seed=%s, unique frozen models=%s', seed, len(selected_candidates))
    write_csv(result / 'tables' / 'chosen_configurations.csv', chosen_rows)
    write_csv(result / 'tables' / 'final_training_timing.csv', timings)
    write_json(result / 'freeze_manifest.json', {'configuration_sha256': digest_value(configuration), 'source_sha256': source_hashes(project), 'models': frozen_models, 'training_subjects': configuration['split']['development_subjects'], 'test_subjects': configuration['split']['test_subjects'], 'test_model_evaluation_started': False, 'protocol': 'All parameters selected from development subjects. No test model scores exist at freeze time.'})
    return frozen_models


def verify_freeze(configuration, result, project):
    frozen = json.loads((result / 'freeze_manifest.json').read_text(encoding='utf-8'))
    assert frozen['configuration_sha256'] == digest_value(configuration)
    assert frozen['source_sha256'] == source_hashes(project)
    for model in frozen['models']:
        assert digest_file(result / model['relative_path']) == model['sha256']
    assert not set(frozen['training_subjects']) & set(frozen['test_subjects'])
    return frozen


def run_evaluation(data, store, configuration, result, project, logger):
    if (result / 'evaluation_complete.json').exists():
        logger.info('Final evaluation already complete; refusing to reopen it for tuning')
        return
    frozen = verify_freeze(configuration, result, project)
    models = [(entry, joblib.load(result / entry['relative_path'])) for entry in frozen['models']]
    test_records = data.subset_records(configuration['split']['test_subjects'])
    power = models[0][1]['reference_power']
    assert all(np.array_equal(model['reference_power'], power) for _, model in models)
    conditions = [('clean', None)]
    for noise_type, grid_key in [('gaussian', 'final_gaussian_snr_db'), ('impulse', 'final_impulse_snr_db')]:
        for snr in configuration['noise'][grid_key]:
            for seed in configuration['randomness']['test_noise_roots']:
                specification = noise_spec(configuration, 'final_test', configuration['randomness']['full_training_and_test_fold_id'], seed, snr, noise_type)
                conditions.append((f'{noise_type}_snr{snr}_seed{seed}', specification))
    subject_rows = []
    overall_rows = []
    diagnostic_rows = []
    prediction_directory = result / 'predictions'
    prediction_directory.mkdir(parents=True, exist_ok=True)
    confusion_arrays = {}
    for condition_name, specification in conditions:
        batch = store.get(test_records, power, specification)
        condition_fields = {'noise_type': 'clean' if specification is None else specification['type'], 'snr_db': 'clean' if specification is None else specification['snr'], 'noise_seed': 0 if specification is None else specification['root']}
        for experiment, subject, channel, realized in batch['snr_diagnostics']:
            diagnostic_rows.append({**condition_fields, 'experiment_id': experiment, 'subject_id': subject, 'channel': channel, 'realized_snr_db': realized})
        with gzip.open(prediction_directory / f'{condition_name}.csv.gz', 'wt', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['model_id', 'window_id', 'y_true', 'y_pred'])
            for entry, model in models:
                started = time.perf_counter()
                prediction = model['pipeline'].predict(batch['features'][:, model['selected_indices']])
                prediction_seconds = time.perf_counter() - started
                metrics, subjects, matrix = evaluate_predictions(batch['targets'], prediction, batch['subjects'], configuration)
                common = {'model_id': entry['model_id'], 'method': entry['method'], 'training_seed': entry['training_seed'], 'k': entry['k'], 'lambda': entry['lambda'], **condition_fields}
                overall_rows.append({**common, **metrics, 'prediction_seconds': prediction_seconds, 'test_windows': len(prediction)})
                for subject in subjects:
                    subject_rows.append({**common, **subject})
                writer.writerows(zip([entry['model_id']] * len(prediction), batch['window_ids'], batch['targets'], prediction))
                confusion_arrays[f'{condition_name}__{entry["model_id"]}'] = matrix
        logger.info('Final evaluation: %s, %s windows x %s frozen models', condition_name, len(batch['targets']), len(models))
    write_csv(result / 'tables' / 'test_subject_metrics.csv', subject_rows)
    write_csv(result / 'tables' / 'test_overall_metrics.csv', overall_rows)
    write_csv(result / 'tables' / 'realized_snr.csv', diagnostic_rows)
    np.savez_compressed(result / 'tables' / 'confusion_matrices.npz', **confusion_arrays)
    write_json(result / 'evaluation_complete.json', {'conditions': len(conditions), 'models': len(models), 'subject_metric_rows': len(subject_rows), 'prediction_rows': len(conditions) * len(models) * len(batch['targets']), 'configuration_sha256': digest_value(configuration), 'no_fit_during_final_evaluation': True})


def smoke_run(data, store, configuration, scratch, logger):
    smoke = configuration['implementation']['smoke']
    assert not set(smoke['training_subjects'] + smoke['validation_subjects']) & set(configuration['split']['test_subjects'])
    training_records = data.subset_records(smoke['training_subjects'], smoke['records_per_subject'])
    validation_records = data.subset_records(smoke['validation_subjects'], smoke['records_per_subject'])
    seed = configuration['randomness']['training_roots'][0]
    clean, power, statistics = statistics_for(data, store, training_records, seed, 0, configuration)
    mask = np.zeros(len(clean['targets']), dtype=bool)
    for subject in sorted(set(clean['subjects'])):
        for label in configuration['evaluation']['label_order']:
            indices = np.flatnonzero((clean['subjects'] == subject) & (clean['targets'] == label))
            mask[indices[:smoke['max_windows_per_subject_class']]] = True
    train_features = clean['features'][mask]
    train_targets = clean['targets'][mask]
    assert set(train_targets) == set(configuration['evaluation']['label_order'])
    validation_clean = store.get(validation_records)
    validation_noisy = store.get(validation_records, power, noise_spec(configuration, 'validation', 0, configuration['randomness']['validation_noise_roots'][0], smoke['noise_snr_db']))
    rows = []
    smoke_predictions = []
    for method in configuration['methods']:
        candidate = {'key': method['id'], 'method': method['id'], 'k': smoke['k'], 'lambda': smoke['lambda'] if isinstance(method['lambda'], str) else method['lambda'], 'beta': method['beta']}
        model = train_model(train_features, train_targets, statistics, candidate, seed, configuration)
        for condition, batch in [('clean', validation_clean), ('gaussian', validation_noisy)]:
            predictions = model['pipeline'].predict(batch['features'][:, model['selected_indices']])
            metrics, _, _ = evaluate_predictions(batch['targets'], predictions, batch['subjects'], configuration)
            rows.append({'method': method['id'], 'condition': condition, **metrics})
            smoke_predictions += [{'method': method['id'], 'condition': condition, 'window_id': str(identifier), 'y_true': int(target), 'y_pred': int(prediction)} for identifier, target, prediction in zip(batch['window_ids'], batch['targets'], predictions)]
    write_csv(scratch / 'smoke' / 'metrics.csv', rows)
    write_csv(scratch / 'smoke' / 'predictions.csv', smoke_predictions)
    summary = {'configuration_sha256': digest_value(configuration), 'training_subjects': smoke['training_subjects'], 'validation_subjects': smoke['validation_subjects'], 'final_test_accessed': False, 'classifier_training_windows': len(train_targets), 'statistics_training_windows': len(clean['targets']), 'validation_windows': len(validation_clean['targets']), 'methods': len(configuration['methods']), 'metrics': rows, 'passed': True}
    write_json(scratch / 'smoke' / 'complete.json', summary)
    logger.info('Development-only smoke passed: classifier train=%s validation=%s windows; all 5 methods', len(train_targets), len(validation_clean['targets']))


def timing_benchmark(data, configuration, result, models):
    benchmark = configuration['implementation']['timing_benchmark']
    generator = np.random.default_rng(benchmark['seed'])
    records = data.subset_records(configuration['split']['development_subjects'])
    first = data.load(records[0])
    starts = generator.choice(first['starts'], size=benchmark['windows'], replace=True)
    windows = first['signals'][starts[:, None] + np.arange(configuration['window']['length_samples'])]
    rows = []
    for entry in models:
        if 'deployment' not in entry['views']:
            continue
        model = joblib.load(result / entry['relative_path'])
        for repetition in range(configuration['resources']['timing_warmup_repetitions'] + configuration['resources']['timing_measured_repetitions']):
            started = time.perf_counter()
            features = extract_features(windows, configuration)
            extracted = time.perf_counter()
            model['pipeline'].predict(features[:, model['selected_indices']])
            finished = time.perf_counter()
            if repetition >= configuration['resources']['timing_warmup_repetitions']:
                rows.append({'model_id': entry['model_id'], 'method': entry['method'], 'training_seed': entry['training_seed'], 'repetition': repetition, 'batch_windows': len(windows), 'feature_seconds': extracted - started, 'classification_seconds': finished - extracted, 'end_to_end_ms_per_window': 1000 * (finished - started) / len(windows)})
    write_csv(result / 'tables' / 'inference_benchmark.csv', rows)


def execute(command, configuration, workspace, project):
    scratch = workspace / configuration['implementation']['scratch_root']
    result = project / configuration['implementation']['result_relative_path']
    logger = make_logger(scratch / configuration['implementation']['run_id'] / 'logs')
    monitor = ResourceMonitor()
    started = time.perf_counter()
    available = psutil.virtual_memory().available
    logger.info('Command=%s; configuration_sha256=%s; available_memory_GiB=%.3f', command, digest_value(configuration), available / 2 ** 30)
    if available < configuration['resources']['minimum_available_memory_bytes']:
        logger.warning('Available memory below planned 2 GiB threshold; sequential record processing and condition caches remain enabled')
    result.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=configuration['resources']['numeric_threads']):
        data = DataStore(configuration, workspace, scratch, logger)
        data.prepare(result)
        store = FeatureStore(data, scratch, configuration, logger)
        if command in ['check', 'all']:
            from .verification import verify_implementation
            checks = verify_implementation(data, configuration)
            write_json(scratch / 'checks.json', checks)
            write_json(result / 'audit' / 'checks.json', checks)
            logger.info('Implementation checks passed: %s', list(checks))
        if command in ['smoke', 'all']:
            smoke_run(data, store, configuration, scratch, logger)
        if command in ['train', 'all']:
            marker = scratch / 'smoke' / 'complete.json'
            checks = scratch / 'checks.json'
            if not marker.exists() or not checks.exists():
                raise RuntimeError('Run check and smoke before formal training')
            smoke = json.loads(marker.read_text(encoding='utf-8'))
            assert smoke['configuration_sha256'] == digest_value(configuration)
            if (result / 'freeze_manifest.json').exists():
                logger.info('Frozen training exists; source and configuration hashes checked; no retuning')
                verify_freeze(configuration, result, project)
            else:
                write_json(result / 'configuration_snapshot.json', configuration)
                write_json(result / 'audit' / 'smoke_complete.json', smoke)
                averaged = run_cv(data, store, configuration, result, logger)
                frozen_models = final_training(data, store, configuration, result, project, averaged, logger)
                write_json(result / 'training_complete.json', {'models': len(frozen_models), 'seconds_including_prepare_and_cache': time.perf_counter() - started, 'test_evaluation_started': False})
        if command in ['evaluate', 'all']:
            run_evaluation(data, store, configuration, result, project, logger)
            frozen = verify_freeze(configuration, result, project)
            timing_benchmark(data, configuration, result, frozen['models'])
        if command in ['plot', 'all']:
            from .plotting import generate_artifacts
            generate_artifacts(configuration, result)
            logger.info('Aggregated tables and figures generated')
        if store.calls:
            write_json(scratch / configuration['implementation']['run_id'] / f'feature_calls_{command}.json', store.calls)
    resources = monitor.finish()
    resources['command'] = command
    resources['packages'] = {package: importlib.metadata.version(package) for package in ['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'joblib', 'threadpoolctl']}
    write_json(result / 'audit' / f'resources_{command}.json', resources)
    logger.info('Command complete: %s wall_seconds=%.3f sampled_peak_RSS_MiB=%.2f', command, resources['wall_seconds'], resources['sampled_peak_rss_bytes'] / 2 ** 20)
