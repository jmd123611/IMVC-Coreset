from formal_experiment_config import BUDGET_FRACTIONS as FORMAL_BUDGET_FRACTIONS, MIN_SAMPLE_SIZE as FORMAL_MIN_SAMPLE_SIZE, SAMPLE_RATIO as FORMAL_SAMPLE_RATIO, SAMPLING_MODE as FORMAL_SAMPLING_MODE, STRATUM_FRACTIONS as FORMAL_STRATUM_FRACTIONS
ini_len = 1
import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

def purity(cluster, label):
    cluster = np.array(cluster)
    label = np.array(label)
    indedata1 = {}
    for p in np.unique(label):
        indedata1[p] = np.argwhere(label == p)
    indedata2 = {}
    for q in np.unique(cluster):
        indedata2[q] = np.argwhere(cluster == q)
    count_all = []
    for i in indedata1.values():
        count = []
        for j in indedata2.values():
            a = np.intersect1d(i, j).shape[0]
            count.append(a)
        count_all.append(count)
    return sum(np.max(count_all, axis=0)) / len(cluster)

def f_measure(labels_true, labels_pred, beta=1.0):
    (tn, fp), (fn, tp) = metrics.cluster.pair_confusion_matrix(labels_true, labels_pred)
    p, r = (tp / (tp + fp), tp / (tp + fn))
    f_beta = (1 + beta ** 2) * (p * r / (beta ** 2 * p + r))
    return f_beta

def ACC(y_true, y_pred):
    y_true = np.array(y_true)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    D = int(D)
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[int(y_pred[i]), y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size

class Node:

    def __init__(self, split_feature=None, split_value=None, left=None, right=None, y_data=None, x_data=None, label=None):
        self.split_feature = split_feature
        self.split_value = split_value
        self.left = left
        self.right = right
        self.y_data = y_data
        self.x_data = x_data
        self.label = label

    def is_leaf(self):
        return self.left is None and self.right is None

    def set_condition(self, split_feature, split_value):
        self.split_feature = split_feature
        self.split_value = split_value

class Tree:

    def split_dataset(self, x_data, y_data, feature_index, split_value):
        left_mask = x_data[:, feature_index] <= split_value
        right_mask = x_data[:, feature_index] > split_value
        x_left, y_left = (x_data[left_mask], y_data[left_mask])
        x_right, y_right = (x_data[right_mask], y_data[right_mask])
        return (x_left, y_left, x_right, y_right)

    def build_tree(self, x_data, y_data, max_depth, min_samples_split, depth=1):
        if len(np.unique(y_data)) == 1 or len(y_data) < min_samples_split or depth == max_depth:
            leaf_label = np.argmax(np.bincount(y_data))
            return Node(y_data=y_data, x_data=x_data, label=leaf_label)
        split_feature, split_value = self.best_split(x_data, y_data)
        if split_feature is None:
            leaf_label = np.argmax(np.bincount(y_data))
            return Node(y_data=y_data, x_data=x_data, label=leaf_label)
        X_left, y_left, X_right, y_right = self.split_dataset(x_data, y_data, split_feature, split_value)
        left_child = self.build_tree(X_left, y_left, max_depth, min_samples_split, depth + 1)
        right_child = self.build_tree(X_right, y_right, max_depth, min_samples_split, depth + 1)
        return Node(split_feature=split_feature, split_value=split_value, left=left_child, right=right_child, y_data=y_data, x_data=x_data)

    def predict_subtree(self, node, x_data):
        if node.is_leaf():
            return node.label
        feature_value = x_data[node.split_feature]
        if feature_value <= node.split_value:
            return self.predict_subtree(node.left, x_data)
        else:
            return self.predict_subtree(node.right, x_data)

    def optimize_tree(self, root):
        if root is None:
            return
        queue = [root]
        bfs_order = []
        while queue:
            current_node = queue.pop(0)
            bfs_order.append(current_node)
            if current_node.left:
                queue.append(current_node.left)
            if current_node.right:
                queue.append(current_node.right)
        for node in reversed(bfs_order):
            self.optimize_node(node)

    def update_tree_data(self, node, x_data, y_data, parent=None, is_left_child=True):
        if node is None:
            return
        node.x_data = x_data
        node.y_data = y_data
        if node.is_leaf():
            node.label = np.argmax(np.bincount(y_data)) if len(y_data) > 0 else None
        else:
            x_left, y_left, x_right, y_right = self.split_dataset(x_data, y_data, node.split_feature, node.split_value)
            if len(x_left) == 0 or len(x_right) == 0:
                non_empty_child = node.left if len(x_left) > 0 else node.right
                non_empty_child.x_data = x_data
                non_empty_child.y_data = y_data
                if is_left_child:
                    parent.left = non_empty_child
                else:
                    parent.right = non_empty_child
                self.update_tree_data(parent.left if is_left_child else parent.right, x_data, y_data, parent, is_left_child)
                return
            else:
                self.update_tree_data(node.left, x_left, y_left, node, True)
                self.update_tree_data(node.right, x_right, y_right, node, False)

    def rebuild_tree(self, root, x_data, y_data):
        self.update_tree_data(root, x_data, y_data)

    def collect_leaf_labels(self, node, x_data):
        labels = []
        for x_sample in x_data:
            label = self.predict_subtree(node, x_sample)
            labels.append(label)
        labels = np.array(labels)
        return labels

    def count_nodes(self, node):
        if node is None:
            return 0
        return 1 + self.count_nodes(node.left) + self.count_nodes(node.right)

    def count_leaf_nodes(self, node):
        if node is None:
            return 0
        if node.is_leaf():
            return 1
        return self.count_leaf_nodes(node.left) + self.count_leaf_nodes(node.right)

    def max_depth(self, node, current_depth=0):
        if node is None:
            return current_depth - 1
        left_depth = self.max_depth(node.left, current_depth + 1)
        right_depth = self.max_depth(node.right, current_depth + 1)
        return max(left_depth, right_depth)

    def calculate_average_leaf_depth(self, node):
        total_depth = 0
        leaf_count = 0

        def traverse(node, current_depth):
            nonlocal total_depth, leaf_count
            if node is None:
                return
            if node.is_leaf():
                total_depth += current_depth
                leaf_count += 1
            else:
                traverse(node.left, current_depth + 1)
                traverse(node.right, current_depth + 1)
        traverse(node, 0)
        if leaf_count == 0:
            return 0
        return total_depth / leaf_count



    def update_node_data(self, node, x_data, new_labels):
        if node is None:
            return
        x_data = np.asarray(x_data)
        new_labels = np.asarray(new_labels)
        if len(x_data) != len(new_labels):
            raise ValueError('x_data and new_labels must have the same length')
        nodes = []
        stack = [node]
        while stack:
            current = stack.pop()
            current.sample_indices = []
            nodes.append(current)
            if current.right is not None:
                stack.append(current.right)
            if current.left is not None:
                stack.append(current.left)
        for sample_index, sample in enumerate(x_data):
            current = node
            while current is not None:
                current.sample_indices.append(sample_index)
                if current.is_leaf():
                    break
                if sample[current.split_feature] <= current.split_value:
                    current = current.left
                else:
                    current = current.right
        for current in nodes:
            current.sample_indices = np.asarray(current.sample_indices, dtype=np.int64)
            current.y_data = new_labels[current.sample_indices]
            if current.is_leaf():
                if len(current.y_data) > 0:
                    current.label = int(np.argmax(np.bincount(current.y_data)))
                else:
                    current.label = None

    @staticmethod
    def _gini_scan_feature(feature_values, class_indices, class_count):
        sample_count = len(feature_values)
        order = np.argsort(feature_values, kind='mergesort')
        sorted_values = feature_values[order]
        sorted_classes = class_indices[order]
        indicators = np.zeros((sample_count, class_count), dtype=np.int64)
        indicators[np.arange(sample_count), sorted_classes] = 1
        prefix_counts = np.cumsum(indicators, axis=0)
        total_counts = prefix_counts[-1]
        group_ends = np.flatnonzero(sorted_values[:-1] != sorted_values[1:])
        if len(group_ends) == 0:
            return (np.empty(0, dtype=sorted_values.dtype), np.empty(0))
        left_counts = prefix_counts[group_ends]
        right_counts = total_counts - left_counts
        left_sizes = (group_ends + 1).astype(np.float64)
        right_sizes = sample_count - left_sizes
        left_square_sums = np.sum(left_counts * left_counts, axis=1)
        right_square_sums = np.sum(right_counts * right_counts, axis=1)
        weighted_gini = 1.0 - (left_square_sums / left_sizes + right_square_sums / right_sizes) / sample_count
        thresholds = sorted_values[group_ends]
        return (thresholds, weighted_gini)

    def best_split(self, x_data, y_data):
        labels = np.asarray(y_data).reshape(-1)
        _, class_indices = np.unique(labels, return_inverse=True)
        class_count = int(class_indices.max()) + 1
        best_gini = np.inf
        best_split_feature = None
        best_split_value = None
        for feature_index in range(x_data.shape[1]):
            thresholds, gini_values = self._gini_scan_feature(x_data[:, feature_index], class_indices, class_count)
            thresholds = thresholds[::ini_len]
            gini_values = gini_values[::ini_len]
            if len(thresholds) == 0:
                continue
            local_index = int(np.argmin(gini_values))
            local_gini = float(gini_values[local_index])
            if local_gini < best_gini:
                best_gini = local_gini
                best_split_feature = feature_index
                best_split_value = thresholds[local_index]
        return (best_split_feature, best_split_value)

    STRATUM_FRACTIONS = FORMAL_STRATUM_FRACTIONS
    BUDGET_FRACTIONS = FORMAL_BUDGET_FRACTIONS

    def __init__(self, view_scores=None, sample_ratio=FORMAL_SAMPLE_RATIO, min_sample_size=FORMAL_MIN_SAMPLE_SIZE, random_state=42, budget_fractions=None, stratum_fractions=None, sampling_mode=FORMAL_SAMPLING_MODE):
        self.view_scores = np.asarray(view_scores, dtype=np.float64)
        self.sample_ratio = float(sample_ratio)
        self.min_sample_size = int(min_sample_size)
        self.rng = np.random.default_rng(random_state)
        if sampling_mode != 'resample':
            raise ValueError('sampling_mode must be resample')
        self.sampling_mode = sampling_mode
        self.budget_fractions = tuple(self.BUDGET_FRACTIONS if budget_fractions is None else budget_fractions)
        self.stratum_fractions = tuple(self.STRATUM_FRACTIONS if stratum_fractions is None else stratum_fractions)
        self.optimization_stats_history = []
        if not 0 < self.sample_ratio <= 1:
            raise ValueError('sample_ratio must be in (0, 1]')
        if len(self.budget_fractions) != 3 or not np.isclose(sum(self.budget_fractions), 1.0):
            raise ValueError('budget_fractions must contain three values summing to 1')
        if len(self.stratum_fractions) != 3 or not np.isclose(sum(self.stratum_fractions), 1.0):
            raise ValueError('stratum_fractions must contain three values summing to 1')
        if any((value <= 0 for value in self.stratum_fractions)):
            raise ValueError('stratum_fractions must be positive')

    @staticmethod
    def _allocate(capacities, total, fractions):
        raw = np.asarray(fractions) * total
        allocation = np.minimum(capacities, np.floor(raw).astype(int))
        while allocation.sum() < total:
            available = np.flatnonzero(allocation < capacities)
            if len(available) == 0:
                break
            priorities = raw[available] - allocation[available]
            chosen = available[int(np.argmax(priorities))]
            allocation[chosen] += 1
        return allocation

    def _stratified_sample(self, node):
        node_size = len(node.x_data)
        sample_size = min(node_size, max(self.min_sample_size, int(np.ceil(self.sample_ratio * node_size))))
        scores = self.view_scores[node.sample_indices]
        order = self._rank_scores(scores)
        first = int(np.ceil(self.stratum_fractions[0] * node_size))
        second = first + int(np.ceil(self.stratum_fractions[1] * node_size))
        strata = [order[:first], order[first:second], order[second:]]
        capacities = np.asarray([len(x) for x in strata], dtype=int)
        allocation = self._allocate(capacities, sample_size, self.budget_fractions)
        sampled, weights, stratum_stats = ([], [], [])
        for indices, count in zip(strata, allocation):
            if count == 0:
                stratum_stats.append({'population': len(indices), 'sampled': 0})
                continue
            if count == len(indices):
                chosen = indices
            else:
                chosen = self.rng.choice(indices, size=count, replace=False)
            sampled.extend(chosen.tolist())
            weights.extend([len(indices) / count] * count)
            stratum_stats.append({'population': len(indices), 'sampled': int(count), 'weight': float(len(indices) / count)})
        sampled = np.asarray(sampled, dtype=np.int64)
        weights = np.asarray(weights, dtype=np.float64)
        sort_order = np.argsort(sampled)
        return (sampled[sort_order], weights[sort_order], stratum_stats)

    def _rank_scores(self, scores):
        return np.argsort(-scores, kind='mergesort')

    def optimize_node(self, node):
        if node is None or node.is_leaf():
            return
        sample_indices, weights, strata = self._stratified_sample(node)
        sampled_x = node.x_data[sample_indices]
        sampled_y = node.y_data[sample_indices]
        deltas = np.zeros(len(sample_indices), dtype=np.float64)
        initial_loss = 0.0
        for index, (sample, label, weight) in enumerate(zip(sampled_x, sampled_y, weights)):
            left_correct = self.predict_subtree(node.left, sample) == label
            right_correct = self.predict_subtree(node.right, sample) == label
            if left_correct and (not right_correct):
                initial_loss += weight
                deltas[index] = -weight
            elif right_correct and (not left_correct):
                deltas[index] = weight
        best_score, best_feature, best_value = (np.inf, None, None)
        candidate_count = 0
        for feature_index in range(node.x_data.shape[1]):
            thresholds = np.unique(node.x_data[:, feature_index])
            order = np.argsort(sampled_x[:, feature_index], kind='mergesort')
            sorted_values = sampled_x[order, feature_index]
            prefix = np.concatenate(([0.0], np.cumsum(deltas[order])))
            positions = np.searchsorted(sorted_values, thresholds, side='right')
            losses = initial_loss + prefix[positions]
            candidate_count += len(thresholds)
            local = int(np.argmin(losses))
            if losses[local] < best_score:
                best_score = float(losses[local])
                best_feature = feature_index
                best_value = thresholds[local]
        if best_feature is not None:
            node.set_condition(best_feature, best_value)
        self.optimization_stats_history.append({'node_samples': len(node.x_data), 'sample_size': len(sample_indices), 'sample_ratio_actual': len(sample_indices) / len(node.x_data), 'candidate_splits': candidate_count, 'weighted_best_score': best_score, 'strata': strata})
