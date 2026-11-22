import collections
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import read_csv, write_csv, write_json


def interval(values, configuration):
    values = np.asarray(values, dtype=float)
    bootstrap = configuration['evaluation']['bootstrap']
    generator = np.random.default_rng(configuration['randomness']['bootstrap_root'])
    indices = generator.integers(0, len(values), size=(bootstrap['replicates'], len(values)))
    draws = values[indices].mean(axis=1)
    alpha = (1 - bootstrap['confidence_level']) / 2
    lower, upper = np.quantile(draws, [alpha, 1 - alpha])
    return float(values.mean()), float(lower), float(upper)


def aggregate(configuration, result):
    freeze = json.loads((result / 'freeze_manifest.json').read_text(encoding='utf-8'))
    entries = {entry['model_id']: entry for entry in freeze['models']}
    groups = collections.defaultdict(list)
    for row in read_csv(result / 'tables' / 'test_subject_metrics.csv'):
        entry = entries[row['model_id']]
        views = list(entry['views'])
        if entry['method'] in ['B0', 'B1', 'B2']:
            views += [view.replace('matched_', 'fixed_lambda_') for view in views if view.startswith('matched_')]
        for view in views:
            groups[(view, row['method'], row['noise_type'], row['snr_db'])].append(row)
    summaries = []
    per_subject = {}
    for key, rows in groups.items():
        subject_values = collections.defaultdict(list)
        stochastic_values = collections.defaultdict(list)
        for row in rows:
            subject_values[int(row['subject_id'])].append(float(row['macro_f1']))
            stochastic_values[(row['training_seed'], row['noise_seed'])].append(float(row['macro_f1']))
        assert set(subject_values) == set(configuration['split']['test_subjects'])
        values = {subject: float(np.mean(scores)) for subject, scores in subject_values.items()}
        per_subject[key] = values
        mean, lower, upper = interval(list(values.values()), configuration)
        summaries.append({'view': key[0], 'method': key[1], 'noise_type': key[2], 'snr_db': key[3], 'subject_macro_f1': mean, 'ci95_lower': lower, 'ci95_upper': upper, 'stochastic_sd': float(np.std([np.mean(scores) for scores in stochastic_values.values()], ddof=1)), 'subjects': len(values), 'stochastic_replicates': len(stochastic_values), **{f'mean_recall_{label}': float(np.mean([float(row[f'recall_{label}']) for row in rows])) for label in configuration['evaluation']['label_order']}})
    write_csv(result / 'tables' / 'figure_source_scores.csv', summaries)
    robust_rows = []
    for view, method in sorted({(key[0], key[1]) for key in per_subject}):
        clean = per_subject[(view, method, 'clean', 'clean')]
        noisy = [per_subject[(view, method, 'gaussian', str(snr))] for snr in configuration['noise']['final_gaussian_snr_db']]
        values = {subject: float(np.mean([scores[subject] for scores in noisy])) for subject in clean}
        per_subject[(view, method, 'gaussian_grid', 'all')] = values
        mean, lower, upper = interval(list(values.values()), configuration)
        robust_rows.append({'view': view, 'method': method, 'gaussian_grid_mean': mean, 'ci95_lower': lower, 'ci95_upper': upper, 'clean_mean': float(np.mean(list(clean.values()))), 'mean_degradation': float(np.mean([clean[subject] - values[subject] for subject in clean])), 'subjects': len(values)})
    write_csv(result / 'tables' / 'gaussian_grid_summary.csv', robust_rows)
    differences = []
    for view in sorted({key[0] for key in per_subject}):
        for first, second in [('P', 'B2'), ('P', 'A1'), ('A1', 'B1'), ('B2', 'B1')]:
            conditions = sorted({(key[2], key[3]) for key in per_subject if key[:2] == (view, first)})
            for noise_type, snr in conditions:
                first_key = (view, first, noise_type, snr)
                second_key = (view, second, noise_type, snr)
                if second_key not in per_subject:
                    continue
                subjects = sorted(per_subject[first_key])
                values = [per_subject[first_key][subject] - per_subject[second_key][subject] for subject in subjects]
                mean, lower, upper = interval(values, configuration)
                differences.append({'view': view, 'comparison': f'{first}-{second}', 'noise_type': noise_type, 'snr_db': snr, 'mean_difference': mean, 'ci95_lower': lower, 'ci95_upper': upper, 'subjects': len(subjects), 'bootstrap_replicates': configuration['evaluation']['bootstrap']['replicates']})
    write_csv(result / 'tables' / 'paired_differences.csv', differences)
    selections = json.loads((result / 'fit_statistics' / 'cv_selected_features.json').read_text(encoding='utf-8'))
    selection_groups = collections.defaultdict(list)
    for row in selections:
        selection_groups[(row['training_seed'], row['candidate'])].append(row)
    jaccard = []
    for (seed, candidate), rows in selection_groups.items():
        for first in range(len(rows)):
            for second in range(first + 1, len(rows)):
                left, right = set(rows[first]['indices']), set(rows[second]['indices'])
                jaccard.append({'training_seed': seed, 'candidate': candidate, 'first_fold': rows[first]['fold_id'], 'second_fold': rows[second]['fold_id'], 'jaccard': len(left & right) / len(left | right)})
    write_csv(result / 'tables' / 'cv_selection_jaccard.csv', jaccard)
    write_json(result / 'tables' / 'subject_averaged_scores.json', [{'view': key[0], 'method': key[1], 'noise_type': key[2], 'snr_db': key[3], 'subject_scores': values} for key, values in per_subject.items()])
    return summaries


def export_figure(figure, stem, configuration):
    settings = configuration['implementation']['plots']
    for extension in settings['formats']:
        figure.savefig(stem.with_suffix('.' + extension), dpi=settings['dpi'], facecolor='white')
    plt.close(figure)


def generate_artifacts(configuration, result):
    if not (result / 'evaluation_complete.json').exists():
        raise RuntimeError('No completed frozen evaluation')
    rows = aggregate(configuration, result)
    settings = configuration['implementation']['plots']
    figure_path = result / 'figures'
    figure_path.mkdir(parents=True, exist_ok=True)
    methods = [method['id'] for method in configuration['methods']]
    labels = {'B0': 'B0 All features', 'B1': 'B1 MI', 'B2': 'B2 MI + redundancy', 'A1': 'A1 MI + stability', 'P': 'P Full method'}
    captions = []
    style = {'font.family': settings['font_family'], 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none', 'savefig.transparent': False}
    with plt.rc_context(style):
        for noise_type in ['gaussian', 'impulse']:
            figure, axes = plt.subplots(figsize=(settings['figure_width_inches'], settings['figure_height_inches']), layout='constrained')
            for index, method in enumerate(methods):
                selected = sorted([row for row in rows if row['view'] == 'deployment' and row['method'] == method and row['noise_type'] == noise_type], key=lambda row: float(row['snr_db']))
                xvalues = [float(row['snr_db']) for row in selected]
                mean = np.array([row['subject_macro_f1'] for row in selected])
                errors = np.array([mean - np.array([row['ci95_lower'] for row in selected]), np.array([row['ci95_upper'] for row in selected]) - mean])
                axes.errorbar(xvalues, mean, yerr=errors, color=settings['colors'][index], marker=settings['markers'][index], capsize=3, label=labels[method], linewidth=1.5)
            axes.set(xlabel='Training-reference SNR (dB; larger = less noise)', ylabel='Subject-mean six-class Macro-F1', ylim=(0, 1), title=f'{noise_type.title()} noise | development-selected configurations\n95% subject-bootstrap CI; 9 people; 3 training x 5 noise seeds')
            axes.set_xticks(sorted({float(row['snr_db']) for row in rows if row['noise_type'] == noise_type}))
            axes.legend(loc='lower right', fontsize=8)
            axes.grid(alpha=0.2)
            stem = f'{noise_type}_robustness'
            export_figure(figure, figure_path / stem, configuration)
            captions.append({'figure': stem, 'source': 'tables/figure_source_scores.csv', 'view': 'deployment', 'uncertainty': '95% percentile CI from 2000 resamples of 9 subjects after averaging training/noise repeats', 'snr': 'training-record-centered reference power, not per-window adaptive SNR'})
        figure, axes = plt.subplots(figsize=(settings['figure_width_inches'], settings['figure_height_inches']), layout='constrained')
        clean_rows = [next(row for row in rows if row['view'] == 'deployment' and row['method'] == method and row['noise_type'] == 'clean') for method in methods]
        values = np.array([row['subject_macro_f1'] for row in clean_rows])
        errors = np.array([values - np.array([row['ci95_lower'] for row in clean_rows]), np.array([row['ci95_upper'] for row in clean_rows]) - values])
        axes.bar(methods, values, yerr=errors, color=settings['colors'], capsize=5, edgecolor='white')
        axes.set(ylim=(0, 1), ylabel='Subject-mean six-class Macro-F1', xlabel='Method (configurations selected using development subjects only)', title='Clean, held-out subjects\n95% subject-bootstrap CI; 9 people; 3 training seeds')
        export_figure(figure, figure_path / 'clean_performance', configuration)
        captions.append({'figure': 'clean_performance', 'source': 'tables/figure_source_scores.csv', 'view': 'deployment', 'uncertainty': '95% subject-bootstrap CI, 2000 resamples; clean has no artificial noise repeats'})
        for prefix, title in [('matched_k', 'Equal feature budget; lambda selected on development set'), ('fixed_lambda_k', 'Equal feature budget; lambda fixed at 0.5 for A1 and P')]:
            sizes = configuration['selection']['k_grid']
            figure, panels = plt.subplots(1, len(sizes), figsize=(12, settings['figure_height_inches']), sharey=True, layout='constrained')
            for axes, size in zip(panels, sizes):
                selected = [next(row for row in rows if row['view'] == f'{prefix}{size}' and row['method'] == method and row['noise_type'] == 'gaussian' and float(row['snr_db']) == settings['ablation_snr_db']) for method in methods[1:]]
                values = np.array([row['subject_macro_f1'] for row in selected])
                errors = np.array([values - np.array([row['ci95_lower'] for row in selected]), np.array([row['ci95_upper'] for row in selected]) - values])
                axes.bar(methods[1:], values, yerr=errors, capsize=4, color=settings['colors'][1:], edgecolor='white')
                axes.set(title=f'K = {size}', ylim=(0, 1), xlabel='Method')
            panels[0].set_ylabel('Subject-mean six-class Macro-F1')
            figure.suptitle(f'{title}\nGaussian, reference SNR = {settings["ablation_snr_db"]} dB; 95% subject-bootstrap CI; 9 people')
            export_figure(figure, figure_path / prefix, configuration)
            captions.append({'figure': prefix, 'source': 'tables/figure_source_scores.csv', 'condition': f'gaussian {settings["ablation_snr_db"]} dB', 'uncertainty': '95% subject-bootstrap CI; see protocol for seed averaging'})
        freeze = json.loads((result / 'freeze_manifest.json').read_text(encoding='utf-8'))
        deployed = [entry for entry in freeze['models'] if entry['method'] == 'P' and 'deployment' in entry['views']]
        with np.load(result / 'tables' / 'confusion_matrices.npz') as archive:
            matrix = np.mean([archive[f'clean__{entry["model_id"]}'] for entry in deployed], axis=0)
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        figure, axes = plt.subplots(figsize=(7, 6), layout='constrained')
        image = axes.imshow(normalized, vmin=0, vmax=1, cmap='Blues', interpolation='nearest')
        class_names = ['Walk', 'Upstairs', 'Downstairs', 'Sit', 'Stand', 'Lie']
        axes.set(xticks=np.arange(6), yticks=np.arange(6), xticklabels=class_names, yticklabels=class_names, xlabel='Predicted activity', ylabel='True activity', title='Full method: clean held-out windows\nRow-normalized confusion; mean counts over 3 training seeds')
        for first in range(6):
            for second in range(6):
                axes.text(second, first, f'{normalized[first, second]:.2f}', ha='center', va='center', color='white' if normalized[first, second] > 0.55 else 'black', fontsize=9)
        figure.colorbar(image, ax=axes, label='Fraction within true class')
        export_figure(figure, figure_path / 'full_clean_confusion', configuration)
        write_csv(result / 'tables' / 'figure_source_confusion.csv', [{'true_class': first + 1, 'predicted_class': second + 1, 'mean_count': matrix[first, second], 'row_fraction': normalized[first, second]} for first in range(6) for second in range(6)])
        captions.append({'figure': 'full_clean_confusion', 'source': 'tables/figure_source_confusion.csv', 'uncertainty': 'No error bars; pooled window counts averaged over 3 training seeds then row-normalized; not the subject-mean metric'})
    write_json(figure_path / 'figure_provenance.json', {'settings': settings, 'captions': captions, 'missing_condition_policy': 'All predeclared conditions required, no interpolation', 'scope': 'Course experiment, no journal-specific compliance claim'})
