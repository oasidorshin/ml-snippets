import numpy as np
from tadm import tqdm


def bootstrap_twosample(targets, preds, alphas, n_trials, metric):
    # Example usage: bootstrap_twosample(targets, preds, 0.05, 1000, roc_auc_score) for 95% CI

    data_len = len(targets)
    indexes = np.arange(data_len)
    
    results = []
    # begin bootstrapping
    for trial in tqdm.tqdm(range(n_trials)):
        sample_indexes = np.random.choice(indexes, data_len)
        
        results.append(metric(targets[sample_indexes], preds[sample_indexes]))
    
    for alpha in alphas:
        left = np.percentile(results, 100 * (alpha / 2.))
        right = np.percentile(results, 100 * (1 - (alpha / 2.)))
        
        print(f"{int((1-alpha)*100)}% bootstrap CI: ({left:.4f}, {right:.4f})")
        
        
def generate_random_image(h, w, c=3):
    return np.random.randint(low=0, high=255, size=(h, w, c), dtype=np.uint8)
