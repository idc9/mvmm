import numpy as np
from sklearn.utils import check_random_state
from numbers import Number
from copy import deepcopy

from mvmm.single_view.toy_data import setup_rand_gmm, setup_grid_mean_gmm


def setup_grid_mean_view_params(n_view_components=[5, 8],
                                random_state=None,
                                custom_view_kws=None,
                                *args, **kwargs):
    """
    Sets up a multi-view gaussian mixture model where the means are
    put on a grid and evenly spaced apart.

    Parameters
    ----------
    n_view_components: list of ints
        Number of components in each view.

    random_state: int, None
        Seed for generating data parameters.


    custom_view_kws: None or list of dicts

    *args, **kwargs:
        See mvmm.toy_data.single_view.setup_rand_gmm parameters.
        These can either be entered as a single value which will be used
        for all views (e.g. n_features=5) or as a list to specify different
        values for each view (e.g. n_features=[5, 10]).

    """

    n_views = len(n_view_components)
    rng = check_random_state(random_state)

    if custom_view_kws is None:
        custom_view_kws = [None] * n_views

    def check_view_args(args, v):

        view_args = deepcopy(args)

        if hasattr(args, 'keys'):
            keys = args.keys()
        else:
            keys = range(len(args))

        for k in keys:
            if not (isinstance(args[k], str) or isinstance(args[k], Number)):
                view_args[k] = args[k][v]
        return view_args

    view_params = []
    for v in range(n_views):
        view_args = check_view_args(args, v)
        view_kwargs = check_view_args(kwargs, v)

        if custom_view_kws[v] is not None:
            for k in custom_view_kws[v].keys():
                view_kwargs[k] = custom_view_kws[v][k]

        mean, cov, _ = setup_grid_mean_gmm(n_components=n_view_components[v],
                                           random_state=rng,
                                           *view_args, **view_kwargs)

        view_params.append({'means': mean, 'covs': cov})

    return view_params


def setup_rand_view_params(n_view_components=[5, 8],
                           random_state=None,
                           custom_view_kws=None,
                           *args, **kwargs):
    """
    Sets up a multi-view mixture model where the means are sampled from an isotroic Gaussian.

    Parameters
    ----------
    n_view_components: list of ints
        Number of components in each view.

    random_state: int, None
        Seed for generating data parameters.


    custom_view_kws: None or list of dicts

    *args, **kwargs:
        See mvmm.toy_data.single_view.setup_rand_gmm parameters.
        These can either be entered as a single value which will be used
        for all views (e.g. n_features=5) or as a list to specify different
        values for each view (e.g. n_features=[5, 10]).


    """

    n_views = len(n_view_components)
    rng = check_random_state(random_state)

    if custom_view_kws is None:
        custom_view_kws = [None] * n_views

    def check_view_args(args, v):

        view_args = deepcopy(args)

        if hasattr(args, 'keys'):
            keys = args.keys()
        else:
            keys = range(len(args))

        for k in keys:
            if not (isinstance(args[k], str) or isinstance(args[k], Number)):
                view_args[k] = args[k][v]
        return view_args

    view_params = []
    for v in range(n_views):
        view_args = check_view_args(args, v)
        view_kwargs = check_view_args(kwargs, v)

        if custom_view_kws[v] is not None:
            for k in custom_view_kws[v].keys():
                view_kwargs[k] = custom_view_kws[v][k]

        mean, cov, _ = setup_rand_gmm(n_components=n_view_components[v],
                                      random_state=rng,
                                      *view_args, **view_kwargs)

        view_params.append({'means': mean, 'covs': cov})

    return view_params


def sample_Y(Pi, n_samples=200, random_state=None):
    """
    Samples cluster labels for multi-view mixture model.

    Parameters
    ----------
    Pi: (n_components_0, n_components_1, n_components_2, ...)
        Cluter Pi matrix.

    n_samples: int
        Number of samples to draw.

    random_state: int, None
        Seed for sampling cluster lables.

    Output
    ------
    Y: (n_samples, n_views)
        Y[i, v] = cluster index of ith obesrvation for the vth view

    y: (n_samples, )
        Cluster assignments where clusters are labeled with overall
        cluster labels
    """
    rng = check_random_state(random_state)

    n_views = Pi.ndim
    n_view_components = Pi.shape

    y_overall = rng.choice(np.arange(len(Pi.reshape(-1)), dtype=int),
                           size=n_samples,
                           p=Pi.reshape(-1))

    Y = np.zeros((n_samples, n_views), dtype=int)

    for i in range(n_samples):
        view_idxs = np.unravel_index(indices=y_overall[i],
                                     shape=n_view_components,
                                     order='C')
        for v in range(n_views):
            Y[i, v] = view_idxs[v]

    return Y, y_overall


def sample_gmm(view_params, Pi, n_samples=200, random_state=None):
    """
    Samples data from a multi-view Gaussian mixture model.


    Parameters
    ----------
    view_params: list of dicts with keys ['means', 'covariances']
        View specific cluster parameters.

    Pi: (n_components_0, n_components_1, ...)
        Pi matrix.

    n_samples: int
        Number of samples.

    random_state: int, None
        Seed to sample data.

    Output
    ------
    view_data: list of data matrices of shape (n_samples, n_features_v)
        Data for each view.

    Y: (n_samples, n_views)
        Y[i, v] = cluster index of ith obesrvation for the vth view

    """
    n_views = len(view_params)

    rng = check_random_state(random_state)

    # sample cluster labels
    Y, y_overall = sample_Y(Pi, n_samples=n_samples, random_state=rng)

    view_data = []
    for v in range(n_views):
        means = view_params[v]['means']
        covs = view_params[v]['covs']
        y = Y[:, v]
        X = np.zeros((n_samples, means.shape[1]))

        for i in range(n_samples):
            X[i, :] = rng.multivariate_normal(mean=means[y[i], :],
                                              cov=covs[y[i], :, :])

        view_data.append(X)

    return view_data, Y  # , y_overall


def get_glw_params(sigma=2.4, delta=.5):
    """
    Returns parameters for toy distribution described in Gao, Bien and Witten 2019

    two view, K=6 component GMM with 10 variables
    """
    d = 10
    K = 6

    covs = np.stack([sigma * np.eye(d) for k in range(K)])

    means_1 = np.vstack([np.hstack([2 * np.ones(5), np.zeros(5)]),
                         np.hstack([np.zeros(5), 2 * np.ones(5)]),
                         np.hstack([2 * np.ones(5), -2 * np.ones(5)]),
                         np.hstack([-2 * np.ones(5), np.zeros(5)]),
                         np.hstack([np.zeros(5), -2 * np.ones(5)]),
                         np.hstack([-2 * np.ones(5), 2 * np.ones(5)])])

    means_2 = np.vstack([np.hstack([-2 * np.ones(6), np.zeros(4)]),
                         np.hstack([np.zeros(6), - 2 * np.ones(4)]),
                         np.hstack([-2 * np.ones(6), 2 * np.ones(4)]),
                         np.hstack([2 * np.ones(6), np.zeros(4)]),
                         np.hstack([np.zeros(4), 2 * np.ones(6)]),
                         np.hstack([2 * np.ones(4), -2 * np.ones(6)])])

    Pi = ((1 - delta) / K**2) * np.ones((K, K)) + (delta / K) * np.eye(K)

    return [{'means': means_1, 'covs': covs},
            {'means': means_2, 'covs': covs}], Pi
