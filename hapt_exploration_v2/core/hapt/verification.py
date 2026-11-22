import numpy as np
from sklearn.metrics import f1_score

from .features import add_noise, extract_features
from .metrics import evaluate_predictions
from .selection import select_features


def verify_implementation(data, configuration):
    checks = {}
    length = configuration['window']['length_samples']
    channels = len(configuration['dataset']['channels'])
    constants = np.full((2, length, channels), 3.0)
    features = extract_features(constants, configuration)
    reshaped = features[:, :60].reshape(2, channels, 10)
    assert np.allclose(reshaped[:, :, 0], 3)
    assert np.allclose(reshaped[:, :, 1], 0)
    assert np.allclose(reshaped[:, :, 2], 3)
    assert np.allclose(reshaped[:, :, 4], 9)
    assert np.allclose(reshaped[:, :, 5:], 0)
    assert np.allclose(features[:, 60:], 0)
    checks['constant_signal_analytic_features'] = True
    frequency_bin = 10
    wave = np.sin(2 * np.pi * frequency_bin * np.arange(length) / length)
    harmonic = np.tile(wave[None, :, None], (1, 1, channels))
    harmonic_features = extract_features(harmonic, configuration)
    assert np.isclose(harmonic_features[0, 2], 1 / np.sqrt(2))
    assert np.isclose(harmonic_features[0, 6], frequency_bin * configuration['dataset']['sampling_hz'] / length)
    assert np.allclose(harmonic_features[0, 60:], 1)
    checks['known_sinusoid_rms_peak_and_correlation'] = True
    generator = np.random.default_rng(configuration['implementation']['checks']['metric_seed'])
    labels = np.asarray(configuration['evaluation']['label_order'])
    targets = generator.choice(labels, size=300)
    predictions = generator.choice(labels, size=300)
    subjects = np.repeat([1, 3, 5], 100)
    measured, rows, _ = evaluate_predictions(targets, predictions, subjects, configuration)
    expected = np.mean([f1_score(targets[subjects == subject], predictions[subjects == subject], labels=labels, average='macro', zero_division=0) for subject in sorted(set(subjects))])
    assert np.isclose(measured['subject_macro_f1'], expected)
    assert np.isclose(measured['macro_f1'], f1_score(targets, predictions, labels=labels, average='macro', zero_division=0))
    perfect, _, _ = evaluate_predictions(labels, labels, np.ones(len(labels)), configuration)
    assert perfect['subject_macro_f1'] == 1
    checks['metrics_match_sklearn_and_perfect_case'] = True
    statistics = {'indices': np.array([2, 4, 6]), 'mi': np.array([1.0, 0.8, 0.1]), 'stability': np.array([1.0, 0.0, 0.0]), 'correlation': np.eye(3)}
    plain, _ = select_features(statistics, 1, 0, 0, configuration)
    robust, _ = select_features(statistics, 1, 1, 0, configuration)
    assert plain.tolist() == [2] and robust.tolist() == [4]
    statistics['mi'][:] = 0
    statistics['stability'][:] = 0
    tied, _ = select_features(statistics, 10, 0, 0, configuration)
    assert tied.tolist() == [2, 4, 6]
    checks['selection_effect_tie_and_k_clipping'] = True
    record = data.subset_records(configuration['split']['development_subjects'])[0]
    signals = data.load(record)['signals']
    power = np.arange(1, channels + 1, dtype=float)
    specification = {'stage': 'validation', 'fold': 0, 'root': configuration['implementation']['checks']['noise_seed'], 'snr': configuration['implementation']['checks']['reference_snr_db'], 'type': 'gaussian', 'replicate': 0}
    _, noise_first = add_noise(signals, power, specification, record, configuration)
    _, noise_again = add_noise(signals, power, specification, record, configuration)
    assert np.array_equal(noise_first, noise_again)
    stride = configuration['window']['stride_samples']
    assert np.array_equal(noise_first[:length][stride:], noise_first[stride:stride + length][:length - stride])
    achieved = np.mean(noise_first ** 2, axis=0)
    expected_power = power * 10 ** (-specification['snr'] / 10)
    assert np.allclose(achieved, expected_power, rtol=0.05)
    checks['record_noise_repeat_overlap_and_expected_power'] = True
    segments = np.array([[1, 1, 1, 20, 300], [1, 1, 2, 301, 320]])
    starts, targets, _, _ = data.make_windows(400, segments, {1, 2})
    assert len(starts) > 0 and (targets == 1).all()
    assert (starts >= 30).all() and (starts + length <= 290).all()
    checks['short_segments_and_window_boundaries'] = True
    development = set(configuration['split']['development_subjects'])
    test = set(configuration['split']['test_subjects'])
    assert not development & test
    for validation_subjects in configuration['split']['validation_subject_folds']:
        training_subjects = development - set(validation_subjects)
        assert not training_subjects & set(validation_subjects)
        training_records = data.subset_records(training_subjects)
        validation_records = data.subset_records(validation_subjects)
        assert not {record['record_id'] for record in training_records} & {record['record_id'] for record in validation_records}
    checks['subject_record_split_disjointness'] = True
    invalid = constants.copy()
    invalid[0, 0, 0] = np.nan
    try:
        extract_features(invalid, configuration)
    except ValueError:
        checks['nonfinite_input_rejected'] = True
    else:
        raise AssertionError('Nonfinite input accepted')
    checks['final_test_predictions_computed'] = False
    return checks
