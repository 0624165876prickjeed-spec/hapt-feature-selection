from pathlib import Path
import collections
import csv
import gzip
import json
import sys


def main():
    project = Path(__file__).resolve().parent
    settings = json.loads((project / 'config.json').read_text(encoding='utf-8'))
    root = project / 'results' / settings['run_id']
    sys.path.insert(0, str(project / 'core'))
    import numpy as np
    from sklearn.metrics import f1_score
    from hapt.common import read_csv, write_json
    from hapt.metrics import evaluate_predictions
    cfg = json.loads((root / 'configuration_snapshot.json').read_text(encoding='utf-8'))['base']
    with np.load(root / 'tuning_predictions.npz') as archive:
        targets = archive['targets']
        subjects = archive['subjects']
        window_ids = archive['window_ids']
        predictions = archive['predictions']
    assert len(set(window_ids)) == len(window_ids)
    assert set(subjects) == set(cfg['split']['test_subjects'])
    assert not set(subjects) & set(cfg['split']['development_subjects'])
    subject_metrics = read_csv(root / 'tables/tuning_subject_metrics.csv')
    lookup = {(int(row['training_seed']), row['candidate'], int(row['noise_root']), int(row['subject_id'])): float(row['macro_f1']) for row in subject_metrics}
    index = read_csv(root / 'tables/tuning_prediction_index.csv')
    checked = 0
    for row in index:
        seed, candidate, noise_root = int(row['training_seed']), row['candidate'], int(row['noise_root'])
        predicted = predictions[int(row['prediction_row'])]
        for subject in sorted(set(subjects)):
            mask = subjects == subject
            score = f1_score(targets[mask], predicted[mask], labels=cfg['evaluation']['label_order'], average='macro', zero_division=0)
            assert abs(score - lookup[(seed, candidate, noise_root, int(subject))]) < 1e-12
            checked += 1
    diagnostic_lookup = {(row['candidate'], int(row['training_seed']), row['noise_type'], row['snr_db'], int(row['noise_root']), int(row['subject_id'])): float(row['macro_f1']) for row in read_csv(root / 'tables/selected_subject_metrics.csv')}
    valid_windows = {identifier: (int(subject), int(target)) for identifier, subject, target in zip(window_ids, subjects, targets)}
    group_key = None
    buffer = []
    group_count = 0
    diagnostic_rows = 0

    def check_group(key, values):
        assert len(values) == len(window_ids)
        assert len({row['window_id'] for row in values}) == len(window_ids)
        target_array = np.array([int(row['target']) for row in values])
        predicted_array = np.array([int(row['prediction']) for row in values])
        subject_array = np.array([int(row['subject_id']) for row in values])
        for row in values:
            assert valid_windows[row['window_id']] == (int(row['subject_id']), int(row['target']))
        model, noise, snr, noise_root = key
        seed_text, candidate = model.split('_', 1)
        seed = int(seed_text.removeprefix('seed'))
        _, scores, _ = evaluate_predictions(target_array, predicted_array, subject_array, cfg)
        for score in scores:
            assert abs(score['macro_f1'] - diagnostic_lookup[(candidate, seed, noise, snr, int(noise_root), score['subject_id'])]) < 1e-12

    with gzip.open(root / 'selected_predictions.csv.gz', 'rt', encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            key = (row['model_id'], row['noise_type'], row['snr_db'], row['noise_root'])
            if group_key is not None and key != group_key:
                check_group(group_key, buffer)
                group_count += 1
                buffer = []
            group_key = key
            buffer.append(row)
            diagnostic_rows += 1
        if buffer:
            check_group(group_key, buffer)
            group_count += 1
    provenance = json.loads((root / 'audit/provenance.json').read_text(encoding='utf-8'))
    assert diagnostic_rows == provenance['selected_prediction_rows']
    result = {'passed': True, 'tuning_subject_metrics_verified_with_sklearn': checked, 'tuning_prediction_rows': int(predictions.size), 'diagnostic_prediction_rows_verified': diagnostic_rows, 'diagnostic_model_condition_groups': group_count, 'window_target_and_subject_coverage_verified': True, 'independent_subject_test': False}
    write_json(root / 'audit/independent_prediction_check.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
