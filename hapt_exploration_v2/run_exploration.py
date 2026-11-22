import argparse
import collections
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['smoke', 'formal'])
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
    arguments = parser.parse_args()
    workspace = arguments.workspace.resolve()
    project = Path(__file__).resolve().parent
    settings = json.loads((project / 'config.json').read_text(encoding='utf-8'))
    cfg = json.loads((workspace / settings['base_snapshot']).read_text(encoding='utf-8'))
    if arguments.stage == 'smoke':
        settings.update(settings['smoke'])
        result = workspace / 'work/hapt_exploration_v2/smoke'
    else:
        result = project / 'results' / settings['run_id']
    if (result / 'complete.json').exists():
        raise RuntimeError('Completed run exists; use a new run_id rather than overwrite')
    result.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(workspace / 'work/matplotlib-cache')
    for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[variable] = str(settings['numeric_threads'])
    sys.dont_write_bytecode = True
    import joblib
    import numpy as np
    from threadpoolctl import threadpool_limits
    sys.path.insert(0, str(project / 'core'))
    from hapt.common import ResourceMonitor, digest_file, make_logger, write_csv, write_json
    from hapt.data import DataStore
    from hapt.experiment import noise_spec, train_model
    from hapt.features import FeatureStore, feature_names
    from hapt.metrics import evaluate_predictions

    logger = make_logger(result / 'audit')
    monitor = ResourceMonitor()
    original_manifest = workspace / settings['original_results'] / 'freeze_manifest.json'
    original_digest = digest_file(original_manifest)
    manifest = json.loads(original_manifest.read_text(encoding='utf-8'))
    for relative, expected in manifest['source_sha256'].items():
        assert digest_file(workspace / settings['original_project'] / relative) == expected
    write_json(result / 'configuration_snapshot.json', {'exploration': settings, 'base': cfg, 'stage': arguments.stage})
    data = DataStore(cfg, workspace, workspace / settings['reusable_cache'], logger)
    data.prepare(result)
    store = FeatureStore(data, workspace / settings['reusable_cache'], cfg, logger)
    training_records = data.subset_records(cfg['split']['development_subjects'])
    evaluation_records = data.subset_records(cfg['split']['test_subjects'])
    clean_train = store.get(training_records)
    clean_evaluation = store.get(evaluation_records)
    assert not set(clean_train['subjects']) & set(clean_evaluation['subjects'])
    expected_subjects = sorted(cfg['split']['development_subjects'])
    statistics_by_seed = {}
    power = None
    for seed in settings['training_roots']:
        source = workspace / settings['original_results'] / 'fit_statistics' / f'final_seed{seed}'
        provenance = json.loads((source / 'fit_provenance.json').read_text(encoding='utf-8'))
        assert sorted(provenance['fit_subject_ids']) == expected_subjects
        if power is None:
            power = np.asarray(provenance['reference_power'])
        else:
            assert np.array_equal(power, provenance['reference_power'])
        with np.load(source / 'statistics.npz') as arrays:
            statistics_by_seed[seed] = {name: arrays[name] for name in arrays.files}
        destination = result / 'fit_statistics' / f'seed{seed}'
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / 'statistics.npz', destination / 'statistics.npz')
        shutil.copy2(source / 'fit_provenance.json', destination / 'fit_provenance.json')
    conditions = []
    fold = cfg['randomness']['full_training_and_test_fold_id']
    for root in settings['tuning_noise_roots']:
        specification = noise_spec(cfg, 'final_test', fold, root, settings['target_snr_db'])
        conditions.append((root, store.get(evaluation_records, power, specification)))
    candidates = []
    for method in settings['methods']:
        sizes = [0] if method == 'B0' else settings['k_grid']
        lambdas = settings['lambda_grid'] if method in ['A1', 'P'] else [0.0]
        betas = settings['beta_grid'] if method in ['B2', 'P'] else [0.0]
        for size in sizes:
            for weight in lambdas:
                for redundancy in betas:
                    key = f'{method}_K{size}_L{weight:g}_B{redundancy:g}'
                    candidates.append({'key': key, 'method': method, 'k': size, 'lambda': weight, 'beta': redundancy})
    if arguments.stage == 'formal':
        smoke_path = workspace / 'work/hapt_exploration_v2/smoke/complete.json'
        assert smoke_path.exists(), 'Run smoke first'
    candidate_scores = []
    tuning_subjects = []
    fit_rows = []
    traces = []
    models = {}
    prediction_rows = []
    prediction_index = []
    started = time.perf_counter()
    with threadpool_limits(limits=settings['numeric_threads']):
        for seed in settings['training_roots']:
            logger.info('Searching seed=%s candidates=%s; reused former test labels, exploratory scores', seed, len(candidates))
            statistics = statistics_by_seed[seed]
            for position, candidate in enumerate(candidates):
                model = train_model(clean_train['features'], clean_train['targets'], statistics, candidate, seed, cfg)
                models[(seed, candidate['key'])] = model
                fit_rows.append({'training_seed': seed, **candidate, **model['timing']})
                traces.append({'training_seed': seed, **candidate, 'indices': model['selected_indices'], 'trace': model['trace']})
                clean_prediction = model['pipeline'].predict(clean_evaluation['features'][:, model['selected_indices']])
                clean_score, _, _ = evaluate_predictions(clean_evaluation['targets'], clean_prediction, clean_evaluation['subjects'], cfg)
                scores = []
                for root, batch in conditions:
                    predicted = model['pipeline'].predict(batch['features'][:, model['selected_indices']])
                    overall, subjects, _ = evaluate_predictions(batch['targets'], predicted, batch['subjects'], cfg)
                    scores.append(overall['subject_macro_f1'])
                    tuning_subjects.extend({'training_seed': seed, 'candidate': candidate['key'], 'noise_root': root, **row} for row in subjects)
                    prediction_rows.append(predicted.astype(np.int8))
                    prediction_index.append({'prediction_row': len(prediction_rows) - 1, 'training_seed': seed, 'candidate': candidate['key'], 'noise_root': root})
                candidate_scores.append({'training_seed': seed, **candidate, 'target_score': float(np.mean(scores)), 'clean_score': clean_score['subject_macro_f1']})
                if (position + 1) % 25 == 0:
                    logger.info('seed=%s finished %s/%s candidates', seed, position + 1, len(candidates))
    write_csv(result / 'tables/candidate_seed_scores.csv', candidate_scores)
    write_csv(result / 'tables/tuning_subject_metrics.csv', tuning_subjects)
    write_csv(result / 'tables/fit_times.csv', fit_rows)
    write_json(result / 'models/all_selection_traces.json', traces)
    np.savez_compressed(result / 'tuning_predictions.npz', predictions=np.stack(prediction_rows), targets=clean_evaluation['targets'], subjects=clean_evaluation['subjects'], window_ids=clean_evaluation['window_ids'])
    write_csv(result / 'tables/tuning_prediction_index.csv', prediction_index)
    aggregated = []
    for candidate in candidates:
        matching = [row for row in candidate_scores if row['key'] == candidate['key']]
        aggregated.append({**candidate, 'target_score': float(np.mean([row['target_score'] for row in matching])), 'clean_score': float(np.mean([row['clean_score'] for row in matching]))})
    write_csv(result / 'tables/candidate_summary.csv', aggregated)

    def choose(rows):
        maximum = max(row['target_score'] for row in rows)
        eligible = [row for row in rows if row['target_score'] >= maximum - settings['score_tie_tolerance']]
        return sorted(eligible, key=lambda row: (-row['clean_score'], row['k'], row['lambda'], row['beta']))[0]

    chosen = {method: choose([row for row in aggregated if row['method'] == method]) for method in settings['methods']}
    views = []
    for method, selected in chosen.items():
        views.append({'view': 'separately_tuned', **selected})
    full = chosen['P']
    matched = {
        'B0': chosen['B0'],
        'B1': next(row for row in aggregated if row['method'] == 'B1' and row['k'] == full['k']),
        'B2': next(row for row in aggregated if row['method'] == 'B2' and row['k'] == full['k'] and row['beta'] == full['beta']),
        'A1': next(row for row in aggregated if row['method'] == 'A1' and row['k'] == full['k'] and row['lambda'] == full['lambda']),
        'P': full
    }
    for selected in matched.values():
        views.append({'view': 'matched_ablation', **selected})
    write_csv(result / 'tables/chosen_configurations.csv', views)
    write_json(result / 'selection_lock.json', {'selection_dataset_role': settings['evidence_scope'], 'selected': views, 'locked_before_fresh_noise_grid': True, 'search_seconds': time.perf_counter() - started, 'objective': settings['selection_metric']})
    selected_keys = {row['key'] for row in views}
    for (seed, key), model in models.items():
        if key in selected_keys:
            model['fit_subject_ids'] = expected_subjects
            model['reference_power'] = power
            model['feature_names'] = [feature_names(cfg)[index] for index in model['selected_indices']]
            joblib.dump(model, result / 'models' / f'seed{seed}_{key}.joblib')
    diagnostic_specs = [('clean', None)]
    for kind, grid in [('gaussian', settings['gaussian_snr_grid_db']), ('impulse', settings['impulse_snr_grid_db'])]:
        for snr in grid:
            for root in settings['confirmation_noise_roots']:
                diagnostic_specs.append((f'{kind}_{snr}_{root}', noise_spec(cfg, 'final_test', fold, root, snr, kind)))
    subject_rows = []
    matrices = []
    prediction_file = result / 'selected_predictions.csv.gz'
    import csv
    with threadpool_limits(limits=settings['numeric_threads']), gzip.open(prediction_file, 'wt', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model_id', 'noise_type', 'snr_db', 'noise_root', 'window_id', 'subject_id', 'target', 'prediction'])
        for condition, specification in diagnostic_specs:
            batch = clean_evaluation if specification is None else store.get(evaluation_records, power, specification)
            noise = 'clean' if specification is None else specification['type']
            snr = 'clean' if specification is None else specification['snr']
            noise_root = 0 if specification is None else specification['root']
            for (seed, key), model in models.items():
                if key not in selected_keys:
                    continue
                identifier = f'seed{seed}_{key}'
                predictions = model['pipeline'].predict(batch['features'][:, model['selected_indices']])
                _, subjects, matrix = evaluate_predictions(batch['targets'], predictions, batch['subjects'], cfg)
                subject_rows.extend({'training_seed': seed, 'candidate': key, 'noise_type': noise, 'snr_db': snr, 'noise_root': noise_root, **row} for row in subjects)
                matrices.append({'model_id': identifier, 'condition': condition, 'matrix': matrix})
                writer.writerows(zip([identifier] * len(predictions), [noise] * len(predictions), [snr] * len(predictions), [noise_root] * len(predictions), batch['window_ids'], batch['subjects'], batch['targets'], predictions))
            logger.info('Fresh-noise diagnostic condition completed: %s', condition)
    write_csv(result / 'tables/selected_subject_metrics.csv', subject_rows)
    write_json(result / 'tables/confusion_matrices.json', matrices)
    summary, pairs = summarize(subject_rows, views, settings, np)
    write_csv(result / 'tables/selected_summary.csv', summary)
    write_csv(result / 'tables/paired_differences.csv', pairs)
    resources = monitor.finish()
    write_json(result / 'audit/resources.json', resources)
    assert digest_file(original_manifest) == original_digest
    write_json(result / 'audit/provenance.json', {'original_freeze_manifest_sha256_unchanged': original_digest, 'original_source_hashes_verified': True, 'fit_subject_ids': expected_subjects, 'tuning_and_diagnostic_subject_ids': cfg['split']['test_subjects'], 'independent_subject_holdout': False, 'noise_roots_disjoint': not set(settings['tuning_noise_roots']) & set(settings['confirmation_noise_roots']), 'candidate_configurations': len(candidates), 'classifier_fits': len(fit_rows), 'tuning_prediction_rows': len(prediction_rows) * len(clean_evaluation['targets']), 'selected_prediction_rows': len(selected_keys) * len(settings['training_roots']) * len(diagnostic_specs) * len(clean_evaluation['targets']), 'source_sha256': {str(path.relative_to(project)): digest_file(path) for path in sorted(project.rglob('*.py'))}})
    plot_summary(summary, result, settings, np)
    write_json(result / 'complete.json', {'stage': arguments.stage, 'completed': True, 'evidence_scope': settings['evidence_scope'], 'resources': resources})
    logger.info('Completed: %s fits; target P=%s, B2=%s', len(fit_rows), chosen['P'], chosen['B2'])


def summarize(subject_rows, views, settings, np):
    grouped = collections.defaultdict(list)
    for row in subject_rows:
        grouped[(row['candidate'], row['noise_type'], str(row['snr_db']), row['subject_id'])].append(row['macro_f1'])
    conditions = sorted({(row['noise_type'], str(row['snr_db'])) for row in subject_rows})
    subjects = sorted({row['subject_id'] for row in subject_rows})
    generator = np.random.default_rng(settings['bootstrap_root'])
    indices = generator.integers(0, len(subjects), size=(settings['bootstrap_replicates'], len(subjects)))
    arrays = {}
    summary = []
    for row in views:
        for noise, snr in conditions:
            values = np.array([np.mean(grouped[(row['key'], noise, snr, subject)]) for subject in subjects])
            arrays[(row['view'], row['method'], noise, snr)] = values
            lower, upper = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
            summary.append({'view': row['view'], 'method': row['method'], 'candidate': row['key'], 'noise_type': noise, 'snr_db': snr, 'mean': float(values.mean()), 'ci95_lower': float(lower), 'ci95_upper': float(upper), 'subjects': len(subjects), 'independent_test': False})
    pairs = []
    for view in sorted({row['view'] for row in views}):
        for noise, snr in conditions:
            for baseline in ['B0', 'B1', 'B2', 'A1']:
                values = arrays[(view, 'P', noise, snr)] - arrays[(view, baseline, noise, snr)]
                lower, upper = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
                pairs.append({'view': view, 'comparison': f'P-{baseline}', 'noise_type': noise, 'snr_db': snr, 'mean_difference': float(values.mean()), 'ci95_lower': float(lower), 'ci95_upper': float(upper), 'selection_adjusted': False})
    return summary, pairs


def plot_summary(summary, result, settings, np):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = result / 'figures'
    directory.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ['#000000', '#0072B2', '#E69F00', '#009E73', '#CC79A7']
    for axis, view in zip(axes, ['separately_tuned', 'matched_ablation']):
        selected = [next(row for row in summary if row['view'] == view and row['method'] == method and row['noise_type'] == 'gaussian' and row['snr_db'] == '20') for method in settings['methods']]
        means = np.array([row['mean'] for row in selected])
        errors = np.array([[row['mean'] - row['ci95_lower'] for row in selected], [row['ci95_upper'] - row['mean'] for row in selected]])
        axis.bar(settings['methods'], means * 100, color=colors, yerr=errors * 100, capsize=3)
        axis.set_ylim(0, 100)
        axis.set_ylabel('Subject-mean six-class Macro-F1 (%)')
        axis.set_title(view.replace('_', ' '))
    figure.suptitle('Exploratory Gaussian 20 dB; original test subjects reused for tuning\nFresh noise seeds; 95% subject-bootstrap CI (not selection-adjusted)', fontsize=10)
    figure.tight_layout()
    figure.savefig(directory / 'target20_exploratory.png', dpi=220)
    figure.savefig(directory / 'target20_exploratory.svg')
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8.8, 5.2))
    for method, color in zip(settings['methods'], colors):
        selected = sorted([row for row in summary if row['view'] == 'separately_tuned' and row['method'] == method and row['noise_type'] == 'gaussian'], key=lambda row: int(row['snr_db']))
        axis.plot([int(row['snr_db']) for row in selected], [100 * row['mean'] for row in selected], marker='o', color=color, label=method)
        axis.fill_between([int(row['snr_db']) for row in selected], [100 * row['ci95_lower'] for row in selected], [100 * row['ci95_upper'] for row in selected], color=color, alpha=0.08)
    axis.set_xlabel('Training-reference SNR (dB; larger means less noise)')
    axis.set_ylabel('Subject-mean six-class Macro-F1 (%)')
    axis.set_ylim(0, 100)
    axis.set_title('Exploratory stress grid after Gaussian20 selection\nSame reused subjects; fresh noise; 95% descriptive subject-bootstrap bands')
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / 'gaussian_stress_grid.png', dpi=220)
    figure.savefig(directory / 'gaussian_stress_grid.svg')
    plt.close(figure)


if __name__ == '__main__':
    main()
