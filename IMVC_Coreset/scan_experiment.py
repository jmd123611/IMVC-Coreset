import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from cluster import generate_pseudo_labels
from dataloader import load_data
from formal_experiment_config import BUDGET_FRACTIONS as FORMAL_BUDGET_FRACTIONS, DYNAMIC_DISAGREEMENT as FORMAL_DYNAMIC_DISAGREEMENT, MIN_SAMPLE_SIZE as FORMAL_MIN_SAMPLE_SIZE, SAMPLE_RATIO as FORMAL_SAMPLE_RATIO, SAMPLING_MODE as FORMAL_SAMPLING_MODE, STRATUM_FRACTIONS as FORMAL_STRATUM_FRACTIONS
from models import MLP
from tree import ACC, Tree, f_measure, purity
from train_model import train_model, train_model_with_new_loss
DEFAULT_SEED = 42
DEFAULT_DATASET = 2005980327

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clustering_metrics(labels, truth):
    labels = np.asarray(labels)
    return {'purity': float(purity(labels, truth)), 'acc': float(ACC(truth, labels)), 'f1': float(f_measure(truth, labels))}

def prepare_base(seed=DEFAULT_SEED, dataset_name=DEFAULT_DATASET, dataset_loader=load_data):
    set_seed(seed)
    wall_start = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, dims, views, data_size, class_num, data_loader = dataset_loader(dataset_name)
    encoders = [MLP(dim, 128, 64).to(device) for dim in dims]
    decoders = [MLP(64, 128, dim).to(device) for dim in dims]
    start = time.perf_counter()
    encoders, decoders, encoded = train_model(encoders, decoders, data_loader, real_label=dataset.L, class_num=class_num)
    pretrain_seconds = time.perf_counter() - start
    start = time.perf_counter()
    pseudo_labels = generate_pseudo_labels(encoded, class_num)
    pseudo_label_seconds = time.perf_counter() - start
    raw_views = [getattr(dataset, f'OV{index + 1}') for index in range(views)]
    final_data = np.concatenate(raw_views, axis=1)
    initial_tree = Tree()
    start = time.perf_counter()
    root = initial_tree.build_tree(final_data, pseudo_labels, max_depth=10, min_samples_split=10)
    initial_tree_seconds = time.perf_counter() - start
    initial_labels = initial_tree.collect_leaf_labels(root, final_data)
    return {'seed': seed, 'dataset_name': dataset_name, 'dataset': dataset, 'dims': dims, 'views': views, 'data_size': data_size, 'class_num': class_num, 'data_loader': data_loader, 'device': device, 'final_data': final_data, 'pseudo_labels': pseudo_labels, 'initial_latent': encoded, 'initial_labels': initial_labels, 'root': root, 'encoder_states': [copy.deepcopy(model.state_dict()) for model in encoders], 'decoder_states': [copy.deepcopy(model.state_dict()) for model in decoders], 'timing': {'pretrain_seconds': pretrain_seconds, 'pseudo_label_seconds': pseudo_label_seconds, 'initial_tree_seconds': initial_tree_seconds, 'shared_initialization_seconds': time.perf_counter() - wall_start}, 'initial_pseudo_metrics': clustering_metrics(pseudo_labels, dataset.L), 'initial_tree_metrics': clustering_metrics(initial_labels, dataset.L), 'initial_tree_structure': {'nodes': initial_tree.count_nodes(root), 'leaves': initial_tree.count_leaf_nodes(root), 'max_depth': initial_tree.max_depth(root), 'average_leaf_depth': initial_tree.calculate_average_leaf_depth(root)}}

def restore_models(base):
    encoders = [MLP(dim, 128, 64).to(base['device']) for dim in base['dims']]
    decoders = [MLP(64, 128, dim).to(base['device']) for dim in base['dims']]
    for model, state in zip(encoders, base['encoder_states']):
        model.load_state_dict(state)
    for model, state in zip(decoders, base['decoder_states']):
        model.load_state_dict(state)
    return (encoders, decoders)

def run_configuration(base, method, sample_ratio=FORMAL_SAMPLE_RATIO, min_sample_size=FORMAL_MIN_SAMPLE_SIZE, coreset_seed=42, max_outer_rounds=50, view_scores=None, budget_fractions=FORMAL_BUDGET_FRACTIONS, stratum_fractions=FORMAL_STRATUM_FRACTIONS, sampling_mode=FORMAL_SAMPLING_MODE, include_tree=False, dynamic_disagreement=FORMAL_DYNAMIC_DISAGREEMENT, include_node_sampling_stats=False):
    if dynamic_disagreement:
        raise ValueError('Disagreement must remain static')
    set_seed(base['seed'])
    if method != 'mvcoreset':
        raise ValueError(f'Unsupported method: {method}')
    if view_scores is None:
        raise ValueError('view_scores are required for mvcoreset')
    tree = Tree(view_scores=view_scores, sample_ratio=sample_ratio, min_sample_size=min_sample_size, random_state=coreset_seed, budget_fractions=budget_fractions, stratum_fractions=stratum_fractions, sampling_mode=sampling_mode)
    name = f'scan_mvcoreset_{int(round(sample_ratio * 100))}pct'
    encoders, decoders = restore_models(base)
    root = copy.deepcopy(base['root'])
    initial_node_count = tree.count_nodes(root)
    final_data = base['final_data']
    truth = base['dataset'].L
    labels = np.asarray(base['initial_labels']).copy()
    previous_outer_node_count = -2
    outer_round = 0
    tree_passes = 0
    model_updates = 0
    converged = False
    history = []
    timing = {'update_node_data_seconds': 0.0, 'optimize_tree_seconds': 0.0, 'rebuild_tree_seconds': 0.0, 'collect_labels_seconds': 0.0, 'model_update_seconds': 0.0, 'disagreement_update_seconds': 0.0}
    disagreement_updates = 0
    optimization_start = time.perf_counter()
    while outer_round < max_outer_rounds and (not converged):
        outer_round += 1
        last_node_count = -1
        node_count = tree.count_nodes(root)
        while last_node_count != node_count:
            last_node_count = node_count
            start = time.perf_counter()
            tree.update_node_data(root, final_data, labels)
            timing['update_node_data_seconds'] += time.perf_counter() - start
            start = time.perf_counter()
            tree.optimize_tree(root)
            timing['optimize_tree_seconds'] += time.perf_counter() - start
            start = time.perf_counter()
            tree.rebuild_tree(root, final_data, labels)
            timing['rebuild_tree_seconds'] += time.perf_counter() - start
            start = time.perf_counter()
            labels = tree.collect_leaf_labels(root, final_data)
            timing['collect_labels_seconds'] += time.perf_counter() - start
            tree_passes += 1
            node_count = tree.count_nodes(root)
            metrics = clustering_metrics(labels, truth)
            history.append({'event': 'tree_pass', 'outer_round': outer_round, 'tree_pass': tree_passes, 'nodes': node_count, **metrics})
            if (initial_node_count - node_count) / initial_node_count > 0.5:
                converged = True
                break
            if node_count == previous_outer_node_count:
                converged = True
                break
        if converged:
            break
        previous_outer_node_count = node_count
        start = time.perf_counter()
        encoders, decoders, encoded = train_model_with_new_loss(encoders, decoders, base['data_loader'], num_epochs=100, class_num=base['class_num'], con_labels=labels)
        timing['model_update_seconds'] += time.perf_counter() - start
        model_updates += 1
        labels = generate_pseudo_labels(encoded, base['class_num'])
        history.append({'event': 'model_update', 'outer_round': outer_round, 'model_update': model_updates, **clustering_metrics(labels, truth)})
    optimization_seconds = time.perf_counter() - optimization_start
    final_labels = tree.collect_leaf_labels(root, final_data)
    final_node_count = tree.count_nodes(root)
    result = {'name': name, 'method': method, 'dataset': base['dataset_name'], 'seed': base['seed'], 'coreset_seed': coreset_seed, 'sample_ratio': sample_ratio, 'min_sample_size': min_sample_size, 'ini_len': 1, 'opt_len': 1, 'converged': converged, 'outer_rounds': outer_round, 'tree_passes': tree_passes, 'model_updates': model_updates, 'dynamic_disagreement': bool(dynamic_disagreement), 'disagreement_updates': disagreement_updates, 'final_metrics': clustering_metrics(final_labels, truth), 'tree_structure': {'nodes': final_node_count, 'leaves': tree.count_leaf_nodes(root), 'max_depth': tree.max_depth(root), 'average_leaf_depth': tree.calculate_average_leaf_depth(root)}, 'timing': {**timing, 'optimization_seconds': optimization_seconds, 'equivalent_end_to_end_seconds': base['timing']['shared_initialization_seconds'] + optimization_seconds}, 'history': history}
    if method in ('coreset', 'mvcoreset'):
        stats = tree.optimization_stats_history
        result['coreset_summary'] = {'node_coresets_built': len(stats), 'total_node_samples': int(sum((item['node_samples'] for item in stats))), 'total_sampled_samples': int(sum((item['sample_size'] for item in stats))), 'mean_actual_sample_ratio': float(np.mean([item['sample_ratio_actual'] for item in stats])) if stats else 0.0}
        if include_node_sampling_stats:
            result['node_sampling_stats'] = stats
        result['sampling_mode'] = tree.sampling_mode
    if method == 'mvcoreset':
        result['budget_fractions'] = list(tree.budget_fractions)
        result['stratum_fractions'] = list(tree.stratum_fractions)
    if include_tree:

        def serialize_node(node):
            if node is None:
                return None
            if node.is_leaf():
                return {'type': 'leaf', 'label': int(node.label), 'samples': int(len(node.y_data))}
            return {'type': 'split', 'feature': int(node.split_feature), 'threshold': float(node.split_value), 'samples': int(len(node.y_data)), 'left': serialize_node(node.left), 'right': serialize_node(node.right)}
        result['tree'] = serialize_node(root)
    return result

def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
