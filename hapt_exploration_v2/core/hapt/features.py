import time

import numpy as np
from scipy.fft import rfft

from .common import digest_value


def feature_names(configuration):
    names = [f'{channel}_{feature}' for channel in configuration['dataset']['channels'] for feature in configuration['features']['per_channel_order']]
    names += [f'corr_{configuration["dataset"]["channels"][first]}_{configuration["dataset"]["channels"][second]}' for first, second in configuration['features']['correlation_pair_order']]
    return names


def extract_features(windows, configuration):
    if windows.ndim != 3 or windows.shape[1:] != (configuration['window']['length_samples'], len(configuration['dataset']['channels'])):
        raise ValueError('Unexpected window shape')
    if not np.isfinite(windows).all():
        raise ValueError('Nonfinite input')
    numeric = configuration['implementation']['numeric']
    mean = windows.mean(axis=1)
    centered = windows - mean[:, None, :]
    std = np.sqrt(np.mean(centered ** 2, axis=1))
    energy = np.mean(windows ** 2, axis=1)
    quartiles = np.quantile(windows, [0.25, 0.75], axis=1, method=configuration['features']['quantile_method'])
    iqr = quartiles[1] - quartiles[0]
    amplitude_tolerance = numeric['relative_constant_tolerance'] * np.maximum(1, np.abs(mean))
    valid = std > amplitude_tolerance
    length = windows.shape[1]
    taper = np.hanning(length)
    spectrum = np.abs(rfft(centered * taper[None, :, None], n=length, axis=1)) ** 2
    spectrum[:, 1:-1] *= 2
    spectrum = spectrum[:, 1:]
    total = spectrum.sum(axis=1)
    valid &= total > 0
    probability = np.divide(spectrum, total[:, None, :], out=np.zeros_like(spectrum), where=total[:, None, :] > 0)
    entropy = -np.sum(probability * np.log(np.maximum(probability, np.finfo(float).tiny)), axis=1) / np.log(length // 2)
    frequencies = np.fft.rfftfreq(length, 1 / configuration['dataset']['sampling_hz'])[1:]
    dominant = frequencies[np.argmax(spectrum, axis=1)]
    bands = []
    for (lower, upper), right_inclusive in zip(configuration['features']['spectrum']['bands_hz'], configuration['features']['spectrum']['band_right_inclusive']):
        mask = (frequencies >= lower) & ((frequencies <= upper) if right_inclusive else (frequencies < upper))
        bands.append(np.sum(probability[:, mask, :], axis=1))
    for array in [entropy, dominant, *bands]:
        array[~valid] = configuration['features']['spectrum']['degenerate_spectrum_output']
    per_channel = np.stack([mean, std, np.sqrt(energy), iqr, energy, entropy, dominant, *bands], axis=-1)
    result = [per_channel.reshape(len(windows), -1)]
    correlations = []
    for first, second in configuration['features']['correlation_pair_order']:
        denominator = std[:, first] * std[:, second]
        allowed = (std[:, first] > amplitude_tolerance[:, first]) & (std[:, second] > amplitude_tolerance[:, second])
        covariance = np.mean(centered[:, :, first] * centered[:, :, second], axis=1)
        correlation = np.divide(covariance, denominator, out=np.zeros(len(windows)), where=allowed)
        correlations.append(np.clip(correlation, -1, 1))
    result.append(np.stack(correlations, axis=1))
    features = np.concatenate(result, axis=1)
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite features')
    return features


def add_noise(signals, power, specification, record, configuration):
    if specification is None:
        return signals.copy(), np.zeros_like(signals)
    randomness = configuration['randomness']
    components = [specification['root'], randomness['stages'][specification['stage']], specification['fold'], record['experiment_id'], record['subject_id'], randomness['noise_type_ids'][specification['type']], specification['snr'] + 100, specification['replicate']]
    generator = np.random.default_rng(np.random.SeedSequence(components))
    variance = power * 10 ** (-specification['snr'] / 10)
    if specification['type'] == 'gaussian':
        noise = generator.normal(size=signals.shape) * np.sqrt(variance)
    elif specification['type'] == 'impulse':
        probability = configuration['noise']['impulse']['bernoulli_probability']
        active = generator.random(signals.shape) < probability
        noise = active * generator.normal(size=signals.shape) * np.sqrt(variance / probability)
    else:
        raise ValueError('Unknown noise')
    return signals + noise, noise


class FeatureStore:
    def __init__(self, data, scratch, configuration, logger):
        self.data = data
        self.cfg = configuration
        self.logger = logger
        self.cache = scratch / 'features'
        self.cache.mkdir(parents=True, exist_ok=True)
        self.calls = []

    def get(self, records, power=None, noise=None):
        key = digest_value({'archive': self.cfg['dataset']['sha256'], 'records': [(record['record_id'], record['raw_sha256']) for record in records], 'window': self.cfg['window'], 'features': self.cfg['features'], 'numeric': self.cfg['implementation']['numeric'], 'noise': noise, 'power': power, 'noise_parameters': self.cfg['noise']})
        path = self.cache / f'{key}.npz'
        if path.exists():
            with np.load(path) as loaded:
                return {name: loaded[name] for name in loaded.files}
        started = time.perf_counter()
        chunks = []
        targets = []
        subjects = []
        identifiers = []
        diagnostics = []
        for record in records:
            loaded = self.data.load(record)
            signals, noise_array = add_noise(loaded['signals'], power, noise, record, self.cfg)
            starts = loaded['starts']
            for offset in range(0, len(starts), self.cfg['resources']['feature_batch_windows']):
                batch_starts = starts[offset:offset + self.cfg['resources']['feature_batch_windows']]
                indices = batch_starts[:, None] + np.arange(self.cfg['window']['length_samples'])[None, :]
                chunks.append(extract_features(signals[indices], self.cfg))
            targets.append(loaded['targets'])
            subjects.append(np.full(len(starts), record['subject_id'], dtype=np.int64))
            identifiers += [f'{record["record_id"]}_s{start:05d}' for start in starts]
            if noise is not None:
                raw_power = np.mean((loaded['signals'] - loaded['signals'].mean(axis=0)) ** 2, axis=0)
                noise_power = np.mean(noise_array ** 2, axis=0)
                realized = 10 * np.log10(raw_power / noise_power)
                for channel, actual in zip(self.cfg['dataset']['channels'], realized):
                    diagnostics.append([record['experiment_id'], record['subject_id'], channel, float(actual)])
        batch = {'features': np.concatenate(chunks), 'targets': np.concatenate(targets), 'subjects': np.concatenate(subjects), 'window_ids': np.asarray(identifiers), 'snr_diagnostics': np.asarray(diagnostics, dtype=str).reshape(-1, 4)}
        assert batch['features'].shape == (len(batch['targets']), self.cfg['features']['dimension_before_training_constant_mask'])
        np.savez(path, **batch)
        self.calls.append({'cache_key': key, 'windows': len(batch['targets']), 'noise': noise, 'feature_seconds': time.perf_counter() - started})
        return batch
