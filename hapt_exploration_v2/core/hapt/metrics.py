import numpy as np


def confusion(targets, predictions, labels):
    labels = np.asarray(labels)
    target_indices = np.searchsorted(labels, targets)
    prediction_indices = np.searchsorted(labels, predictions)
    if not (np.isin(targets, labels).all() and np.isin(predictions, labels).all()):
        raise ValueError('Unexpected class')
    return np.bincount(target_indices * len(labels) + prediction_indices, minlength=len(labels) ** 2).reshape(len(labels), len(labels))


def from_confusion(matrix):
    diagonal = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    f1 = np.divide(2 * diagonal, support + predicted, out=np.zeros(len(matrix)), where=(support + predicted) > 0)
    recall = np.divide(diagonal, support, out=np.zeros(len(matrix)), where=support > 0)
    return {'macro_f1': float(f1.mean()), 'accuracy': float(diagonal.sum() / matrix.sum()), **{f'recall_{index + 1}': float(value) for index, value in enumerate(recall)}}


def evaluate_predictions(targets, predictions, subjects, configuration):
    labels = configuration['evaluation']['label_order']
    matrix = confusion(targets, predictions, labels)
    rows = []
    for subject in sorted(set(subjects)):
        mask = subjects == subject
        metrics = from_confusion(confusion(targets[mask], predictions[mask], labels))
        rows.append({'subject_id': int(subject), 'windows': int(mask.sum()), **metrics})
    result = from_confusion(matrix)
    result['subject_macro_f1'] = float(np.mean([row['macro_f1'] for row in rows]))
    return result, rows, matrix
