import numpy as np
from sklearn.utils import check_random_state


def sample_1d_gmm(n_samples=200, n_components=3, sigma=.1, random_state=None):
    """
    Samples from a 1 dimensional GMM where the means are located on the non-negative integers.

    Parameters
    ----------
    n_samples: int
        Number of samples.

    n_components: int
        Number of mixture model components.

    sigma: float
        Cluster standard deviation.

    random_state: None, int
        Seed to generate data.
    """
    rng = check_random_state(random_state)

    means = np.arange(n_components)
    pi = np.ones(n_components) / n_components

    y = rng.choice(np.arange(n_components), p=pi, size=n_samples)

    X = np.random.normal(size=n_samples, scale=sigma)
    X += y

    params = {'means': means, 'sigma': sigma, 'pi': pi}

    return X.reshape(-1, 1), y, params


def setup_grid_mean_gmm(n_components=3, n_features=10,
                        cluster_std=1.0,
                        cov_how='diag',
                        weights_how='uniform',
                        random_state=None):

    rng = check_random_state(random_state)

    ##########################
    # generate cluster means #
    ##########################
    means = np.arange(n_components).reshape(-1, 1) * \
        np.ones((n_components, n_features))

    ################################
    # generate cluster covariances #
    ################################

    if cov_how.lower() == 'diag':
        # diagonal covariances
        covariances = np.array([np.eye(n_features) * cluster_std ** 2
                                for _ in range(n_components)])
    elif cov_how.lower() == 'wishart':
        raise NotImplementedError  # TODO-FEAT: implement this
    else:
        raise ValueError("cov_how must be one of ['diag', 'wishart']"
                         "not {}".format(cov_how))

    ############################
    # generate cluster weights #
    ############################
    if weights_how.lower() == 'uniform':
        weights = rng.dirichlet(np.ones(n_components))
    elif weights_how.lower() == 'random':
        weights = np.ones(n_components) / n_components
    else:
        raise ValueError("weights_how must be one of ['random', 'uniform']"
                         "not {}".format(weights_how))

    return means, covariances, weights


def setup_rand_gmm(n_components=3, n_features=10,
                   clust_mean_std=2.0,
                   cluster_std=1.0,
                   cov_how='diag',
                   weights_how='uniform',
                   random_state=None):
    """
    Randomly assigns cluster probabilities, cluster centers
    and cluster covariances.

    Parameters
    ----------
    n_components: int
        Number of clusters.

    n_features: int
        Number of featurse.

    clust_mean_std: float
        Controlls how far apart the cluster means are.
        Diagonal entries of covariance of Gaussian
        used to generate cluster means.

    cluster_std: float
        Standard deviation of the samples for each cluster.

    cov_how: str
        How the covariance matrix is generated ['diag', 'wishart'].

    weights_how: str
        How the cluster weights are generated, ['uniform', 'random']

    random_state: None, int
        Seed for data parameters.

    Output
    ------
    means, covariances, weights

    means: (n_components, n_features)
        Cluster means.

    covariances: (n_components, n_features, n_features)
        Covariance matrix of each cluster.

    weights: (n_components, )
        Cluster weights.
    """

    rng = check_random_state(random_state)

    ##########################
    # generate cluster means #
    ##########################

    # cluster means are generated from a spherical gaussian
    center_cov = np.eye(n_features) * clust_mean_std ** 2
    means = rng.multivariate_normal(mean=np.zeros(n_features),
                                    cov=center_cov,
                                    size=n_components)

    ################################
    # generate cluster covariances #
    ################################

    if cov_how.lower() == 'diag':
        # diagonal covariances
        covariances = np.array([np.eye(n_features) * cluster_std ** 2
                                for _ in range(n_components)])
    elif cov_how.lower() == 'wishart':
        raise NotImplementedError  # TODO-FEAT: implement this!
    else:
        raise ValueError("cov_how must be one of ['diag', 'wishart']"
                         "not {}".format(cov_how))

    ############################
    # generate cluster weights #
    ############################
    if weights_how.lower() == 'uniform':
        weights = rng.dirichlet(np.ones(n_components))
    elif weights_how.lower() == 'random':
        weights = np.ones(n_components) / n_components
    else:
        raise ValueError("weights_how must be one of ['random', 'uniform']"
                         "not {}".format(weights_how))

    return means, covariances, weights


def sample_gmm_given_params(means, covariances, weights,
                            n_samples=100, random_state=None):
    """
    Samples observations from a GMM given the cluster parametres.

    Parameters
    ----------
    means: (n_components, n_features)
        Cluster means.

    covariances: (n_components, n_features, n_features)
        Covariance matrix of each cluster.

    weights: (n_components, )
        Cluster weights.

    n_samples: int
        Number of samples.

    random_state: None, int
        Seed for data.
    """

    rng = check_random_state(random_state)

    n_components = len(weights)
    n_features = means.shape[1]

    # sample cluster memberships
    y = rng.choice(a=range(n_components), size=n_samples,
                   replace=True, p=weights)

    # sample data
    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):

        X[i, :] = rng.multivariate_normal(mean=means[y[i], :],
                                          cov=covariances[y[i], :, :])

    return X, y


def sample_gmm(n_samples=100,
               n_components=3, n_features=10,
               clust_mean_std=2.0,
               cluster_std=1.0,
               cov_how='diag',
               weights_how='uniform',
               param_random_state=None,
               data_random_state=None):

    """

    Parameters
    ----------
    n_samples: int
        Number of samples.

    n_components: int
        Number of clusters.

    n_features: int
        Number of featurse.

    clust_mean_std: float
        Controlls how far apart the cluster means are.
        Diagonal entries of covariance of Gaussian
        used to generate cluster means.

    cluster_std: float
        Standard deviation of the samples for each cluster.

    cov_how: str
        How the covariance matrix is generated ['diag', 'wishart'].

    weights_how: str
        How the cluster weights are generated, ['uniform', 'random']

    param_random_state: None, int
        Seed for data parameters.

    data_random_state: None, int
        Seed for data.
    """
    means, covs, weights = setup_rand_gmm(n_components=n_components,
                                          n_features=n_features,
                                          clust_mean_std=clust_mean_std,
                                          cluster_std=cluster_std,
                                          cov_how=cov_how,
                                          weights_how=weights_how,
                                          random_state=param_random_state)

    X, y = sample_gmm_given_params(means=means,
                                   covariances=covs,
                                   weights=weights,
                                   n_samples=n_samples,
                                   random_state=data_random_state)

    return X, y
