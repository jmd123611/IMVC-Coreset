import argparse
import inspect
import json
from secrets import randbits
from pathlib import Path
import numpy as np
from formal_experiment_config import BUDGET_FRACTIONS, DATASETS, DYNAMIC_DISAGREEMENT, MIN_SAMPLE_SIZE, SAMPLE_RATIO, SAMPLING_MODE, INDEPENDENT_RUNS, STRATUM_FRACTIONS, as_dict
from mv_disagreement import compute_view_disagreement
from scan_experiment import prepare_base, run_configuration, save_json, set_seed
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = PROJECT_ROOT / 'results' / 'formal_submission_v1'

def audit_applied_configuration(result):
    checks = {'sample_ratio': np.isclose(result['sample_ratio'], SAMPLE_RATIO), 'min_sample_size': result['min_sample_size'] == MIN_SAMPLE_SIZE, 'stratum_fractions': np.allclose(result['stratum_fractions'], STRATUM_FRACTIONS), 'budget_fractions': np.allclose(result['budget_fractions'], BUDGET_FRACTIONS), 'sampling_mode': result['sampling_mode'] == SAMPLING_MODE, 'static_disagreement': result['dynamic_disagreement'] is DYNAMIC_DISAGREEMENT and result['disagreement_updates'] == 0}
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f'Formal configuration audit failed: {failed}')
    return {name: bool(passed) for name, passed in checks.items()}

def lightweight_parameter_check():
    from tree import Tree
    default_tree = Tree(view_scores=np.zeros(6))
    explicit_tree = Tree(view_scores=np.zeros(6), sample_ratio=SAMPLE_RATIO, min_sample_size=MIN_SAMPLE_SIZE, budget_fractions=BUDGET_FRACTIONS, stratum_fractions=STRATUM_FRACTIONS, sampling_mode=SAMPLING_MODE)
    run_defaults = {name: parameter.default for name, parameter in inspect.signature(run_configuration).parameters.items()}
    observed = {'sample_ratio': explicit_tree.sample_ratio, 'stratum_fractions_high_mid_low': list(explicit_tree.stratum_fractions), 'budget_fractions_high_mid_low': list(explicit_tree.budget_fractions), 'min_sample_size': explicit_tree.min_sample_size, 'sampling_mode': explicit_tree.sampling_mode, 'dynamic_disagreement': DYNAMIC_DISAGREEMENT, 'independent_runs': INDEPENDENT_RUNS, 'tree_defaults_match': np.isclose(default_tree.sample_ratio, SAMPLE_RATIO) and np.allclose(default_tree.stratum_fractions, STRATUM_FRACTIONS) and np.allclose(default_tree.budget_fractions, BUDGET_FRACTIONS) and (default_tree.min_sample_size == MIN_SAMPLE_SIZE) and (default_tree.sampling_mode == SAMPLING_MODE), 'run_configuration_defaults_match': np.isclose(run_defaults['sample_ratio'], SAMPLE_RATIO) and run_defaults['min_sample_size'] == MIN_SAMPLE_SIZE and np.allclose(run_defaults['stratum_fractions'], STRATUM_FRACTIONS) and np.allclose(run_defaults['budget_fractions'], BUDGET_FRACTIONS) and (run_defaults['sampling_mode'] == SAMPLING_MODE) and (run_defaults['dynamic_disagreement'] is DYNAMIC_DISAGREEMENT)}
    expected = as_dict()
    assert np.isclose(observed['sample_ratio'], expected['sample_ratio'])
    assert np.allclose(observed['stratum_fractions_high_mid_low'], expected['stratum_fractions_high_mid_low'])
    assert np.allclose(observed['budget_fractions_high_mid_low'], expected['budget_fractions_high_mid_low'])
    assert observed['min_sample_size'] == expected['min_sample_size']
    assert observed['sampling_mode'] == expected['sampling_mode']
    assert observed['dynamic_disagreement'] is False
    assert observed['independent_runs'] == 1
    assert observed['tree_defaults_match']
    assert observed['run_configuration_defaults_match']
    return {'status': 'PASS', 'observed': observed}

def result_path(root, dataset, seed):
    return root / dataset / f'seed_{seed}_formal_mvcoreset.json'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--result-root', type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument('--check-config', action='store_true', help='validate parameters only; do not load data or train')
    args = parser.parse_args()
    if args.check_config:
        print(json.dumps(lightweight_parameter_check(), indent=2))
        return
    datasets = (args.dataset,) if args.dataset else DATASETS
    seeds = (args.seed,) if args.seed is not None else tuple(randbits(32) for _ in range(INDEPENDENT_RUNS))
    result_root = args.result_root.resolve()
    for dataset in datasets:
        for seed in seeds:
            output = result_path(result_root, dataset, seed)
            if output.exists():
                print(f'SKIP existing dataset={dataset} seed={seed} path={output}')
                continue
            base = prepare_base(seed=seed, dataset_name=dataset)
            scores, _ = compute_view_disagreement(base, seed)
            set_seed(seed)
            result = run_configuration(base, method='mvcoreset', sample_ratio=SAMPLE_RATIO, min_sample_size=MIN_SAMPLE_SIZE, coreset_seed=seed, view_scores=scores, budget_fractions=BUDGET_FRACTIONS, stratum_fractions=STRATUM_FRACTIONS, sampling_mode=SAMPLING_MODE, dynamic_disagreement=DYNAMIC_DISAGREEMENT)
            result['formal_configuration_audit'] = audit_applied_configuration(result)
            result['formal_submission_configuration'] = as_dict()
            save_json(result, output)
            print(json.dumps({'dataset': dataset, 'seed': seed, 'output': str(output), 'audit': 'PASS', **result['final_metrics']}, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
