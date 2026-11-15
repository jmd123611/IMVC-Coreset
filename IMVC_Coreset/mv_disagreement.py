import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

def align_to_reference(reference, labels):
    source_values = np.unique(labels)
    target_values = np.unique(reference)
    table = np.zeros((len(source_values), len(target_values)), dtype=np.int64)
    for row, source in enumerate(source_values):
        mask = labels == source
        for column, target in enumerate(target_values):
            table[row, column] = np.count_nonzero(mask & (reference == target))
    rows, columns = linear_sum_assignment(table.max() - table)
    mapping = {source_values[r]: target_values[c] for r, c in zip(rows, columns)}
    return np.asarray([mapping[value] for value in labels])

def compute_view_disagreement_from_models(base, encoders, reference, seed):
    reference = np.asarray(reference)
    view_labels = []
    for view_index, encoder in enumerate(encoders):
        standardized_view = np.asarray(getattr(base['dataset'], f'V{view_index + 1}'))
        encoder.eval()
        with torch.no_grad():
            encoded = encoder(torch.as_tensor(standardized_view, dtype=torch.float32, device=base['device'])).cpu().numpy()
        labels = KMeans(n_clusters=base['class_num'], random_state=seed, n_init=10).fit_predict(encoded)
        view_labels.append(align_to_reference(reference, labels))
    view_labels = np.stack(view_labels, axis=1)
    counts = np.stack([(view_labels == label).sum(axis=1) for label in range(base['class_num'])], axis=1)
    scores = 1.0 - counts.max(axis=1) / view_labels.shape[1]
    return (scores, view_labels)

def compute_view_disagreement(base, seed):
    from scan_experiment import restore_models
    encoders, _ = restore_models(base)
    return compute_view_disagreement_from_models(base, encoders, base['pseudo_labels'], seed)
__all__ = ['align_to_reference', 'compute_view_disagreement', 'compute_view_disagreement_from_models']
