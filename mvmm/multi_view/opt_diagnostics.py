import numpy as np


def get_param_hist(history):
    """
    Returns the parameter history for a GMM.

    Output
    ------
    [(means[v], covs[v], weights[v]) for v in range(n_views) ]

    """
    n_views = history['model'][0].n_views
    means = [[] for _ in range(n_views)]
    covs = [[] for _ in range(n_views)]
    weights = []

    for model in history['model']:
        weights.append(model.weights_)

        for v in range(n_views):
            means[v].append(model.view_models_[v].means_)
            covs[v].append(model.view_models_[v].covariances_)

    view_param_hist = [(np.array(means[v]), np.array(covs[v]))
                       for v in range(n_views)]
    return view_param_hist, np.array(weights)
