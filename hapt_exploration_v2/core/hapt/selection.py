import time

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler


def fit_statistics(clean_features, targets, noisy_features, seed, configuration):
    started = time.perf_counter()
    numeric = configuration['implementation']['numeric']
    mean = clean_features.mean(axis=0)
    std = clean_features.std(axis=0)
    tolerance = numeric['relative_constant_tolerance'] * np.maximum(1, np.abs(mean))
    indices = np.flatnonzero(std > tolerance)
    if len(indices) == 0:
        raise ValueError('No valid features')
    values = clean_features[:, indices]
    mi_options = configuration['training_statistics']['mi']
    scaled = StandardScaler().fit_transform(values)
    mi = mutual_info_classif(scaled, targets, discrete_features=mi_options['discrete_features'], n_neighbors=mi_options['n_neighbors'], random_state=seed, n_jobs=mi_options['n_jobs'])
    normalized_mi = mi / mi.max() if mi.max() > 0 else np.zeros_like(mi)
    quartiles = np.quantile(values, [0.25, 0.75], axis=0, method=configuration['features']['quantile_method'])
    scale = np.maximum.reduce([quartiles[1] - quartiles[0], numeric['stability_std_floor_multiplier'] * std[indices], tolerance[indices]])
    differences = np.concatenate([np.abs(noisy[:, indices] - values) for noisy in noisy_features], axis=0)
    raw_penalty = np.median(differences, axis=0) / scale
    penalty = raw_penalty / (1 + raw_penalty)
    correlation = np.clip(np.abs(np.corrcoef(values, rowvar=False)), 0, 1)
    correlation = np.atleast_2d(correlation)
    assert np.isfinite(correlation).all()
    return {'indices': indices, 'mi': normalized_mi, 'mi_raw': mi, 'stability': penalty, 'stability_raw': raw_penalty, 'scale': scale, 'correlation': correlation, 'fit_seconds': time.perf_counter() - started}


def select_features(statistics, requested_k, stability_weight, redundancy_weight, configuration):
    dimension = len(statistics['indices'])
    effective = min(requested_k, dimension)
    chosen = []
    trace = []
    available = np.ones(dimension, dtype=bool)
    for step in range(effective):
        redundancy = statistics['correlation'][:, chosen].mean(axis=1) if chosen else np.zeros(dimension)
        score = statistics['mi'] - stability_weight * statistics['stability'] - redundancy_weight * redundancy
        score[~available] = -np.inf
        maximum = score.max()
        tied = np.flatnonzero(available & (score >= maximum - configuration['selection']['score_tie_tolerance']))
        selected = int(tied[0])
        chosen.append(selected)
        available[selected] = False
        trace.append({'step': step + 1, 'original_index': int(statistics['indices'][selected]), 'score': float(score[selected]), 'mi': float(statistics['mi'][selected]), 'stability': float(statistics['stability'][selected]), 'redundancy': float(redundancy[selected])})
    return statistics['indices'][chosen], trace
