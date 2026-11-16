DATASETS = ('mfeat', 'MSRCV1', 'Wikipedia', 'MNIST_USPS', 'Caltech5V')
INDEPENDENT_RUNS = 1
SAMPLE_RATIO = 0.5
STRATUM_FRACTIONS = (0.33, 0.33, 0.34)
BUDGET_FRACTIONS = (0.5, 0.3, 0.2)
MIN_SAMPLE_SIZE = 50
SAMPLING_MODE = 'resample'
DYNAMIC_DISAGREEMENT = False

def as_dict():
    return {'datasets': list(DATASETS), 'independent_runs': INDEPENDENT_RUNS, 'sample_ratio': SAMPLE_RATIO, 'stratum_fractions_high_mid_low': list(STRATUM_FRACTIONS), 'budget_fractions_high_mid_low': list(BUDGET_FRACTIONS), 'min_sample_size': MIN_SAMPLE_SIZE, 'sampling_mode': SAMPLING_MODE, 'dynamic_disagreement': DYNAMIC_DISAGREEMENT, 'disagreement_refresh': 'once_after_initialization', 'sampling_refresh': 'every_internal_node_visit'}
__all__ = ['BUDGET_FRACTIONS', 'DATASETS', 'DYNAMIC_DISAGREEMENT', 'MIN_SAMPLE_SIZE', 'SAMPLE_RATIO', 'SAMPLING_MODE', 'INDEPENDENT_RUNS', 'STRATUM_FRACTIONS', 'as_dict']
