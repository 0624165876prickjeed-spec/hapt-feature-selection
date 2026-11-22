import json
import os
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parent
    result = root / 'outputs/hapt_exploration_v2/results/target20_v2'
    snapshot = json.loads((result / 'configuration_snapshot.json').read_text(encoding='utf-8'))
    settings, cfg = snapshot['exploration'], snapshot['base']
    for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[variable] = str(settings['numeric_threads'])
    sys.path.insert(0, str(root / 'outputs/hapt_exploration_v2/core'))
    import numpy as np
    from threadpoolctl import threadpool_limits
    from hapt.common import make_logger, read_csv, write_json
    from hapt.data import DataStore
    from hapt.features import FeatureStore
    from hapt.experiment import noise_spec, train_model
    from hapt.metrics import evaluate_predictions
    work = root / 'work/main_result_recheck'
    work.mkdir(parents=True, exist_ok=True)
    data = DataStore(cfg, root, work, make_logger(work / 'audit'))
    data.prepare(work)
    store = FeatureStore(data, work, cfg, data.logger)
    training = store.get(data.subset_records(cfg['split']['development_subjects']))
    records = data.subset_records(cfg['split']['test_subjects'])
    chosen = [row for row in read_csv(result / 'tables/chosen_configurations.csv') if row['view'] == 'separately_tuned']
    expected = {row['method']: float(row['mean']) for row in read_csv(result / 'tables/selected_summary.csv') if row['view'] == 'separately_tuned' and row['noise_type'] == 'gaussian' and row['snr_db'] == '20'}
    scores = {row['method']: [] for row in chosen}
    fitted = []
    with threadpool_limits(limits=settings['numeric_threads']):
        for seed in settings['training_roots']:
            source = result / f'fit_statistics/seed{seed}'
            provenance = json.loads((source / 'fit_provenance.json').read_text(encoding='utf-8'))
            assert set(provenance['fit_subject_ids']) == set(cfg['split']['development_subjects'])
            power = np.asarray(provenance['reference_power'])
            with np.load(source / 'statistics.npz') as arrays:
                statistics = {key: arrays[key] for key in arrays.files}
            for row in chosen:
                candidate = {'key': row['key'], 'method': row['method'], 'k': int(row['k']), 'lambda': float(row['lambda']), 'beta': float(row['beta'])}
                model = train_model(training['features'], training['targets'], statistics, candidate, seed, cfg)
                fitted.append((row['method'], model))
        for noise_root in settings['confirmation_noise_roots']:
            specification = noise_spec(cfg, 'final_test', cfg['randomness']['full_training_and_test_fold_id'], noise_root, settings['target_snr_db'])
            batch = store.get(records, power, specification)
            for method, model in fitted:
                predictions = model['pipeline'].predict(batch['features'][:, model['selected_indices']])
                score, _, _ = evaluate_predictions(batch['targets'], predictions, batch['subjects'], cfg)
                scores[method].append(score['subject_macro_f1'])
    values = {method: float(np.mean(repeats)) for method, repeats in scores.items()}
    for method, value in values.items():
        assert abs(value - expected[method]) < 1e-12, (method, value, expected[method])
    record = {'passed': True, 'refitted_models': len(fitted), 'new_noise_roots': settings['confirmation_noise_roots'], 'condition': 'gaussian20', 'fixed_locked_parameters': True, 'independent_subject_test': False, 'reproduced_subject_macro_f1': values, 'absolute_tolerance': 1e-12}
    write_json(work / 'main_result_recheck.json', record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
