import numpy as np
from sklearn.utils import check_random_state
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, OAS


def rand_pts_overall_cov_init(X, n_components, cov_est_method='LW',
                              covariance_type='full', random_state=None):
    """
    Sets the means to randomly selected points. Sets the covariances to the overall covariance matrix.

    Parameters
    ----------
    X: (n_samples, n_features)

    n_components: int

    cov_est_method: str
        Must be one of ['emperical', 'LW', 'OAS'] for
        empirical covariance matrix estimate, LedoitWolf and
        Oracle Approximating Shrinkage Estimator. See
        sklean.covariace for details.

    random_state: None, int, random seed
        Random seed.

    """
    assert cov_est_method in ['empirical', 'LW', 'OAS']
    assert covariance_type in ['full', 'diag', 'tied', 'spherical']
    n_samples = X.shape[0]

    # randomly select data points to start cluster centers from
    rng = check_random_state(random_state)

    # estimate global covariance
    if cov_est_method == 'empirical':
        cov_estimator = EmpiricalCovariance(store_precision=False)
    elif cov_est_method == 'LW':
        cov_estimator = LedoitWolf(store_precision=False)
    elif cov_est_method == 'OAS':
        cov_estimator = OAS(store_precision=False)
    cov_estimator.fit(X)
    cov_est = cov_estimator.covariance_

    # set covariance matrix for each cluster
    if covariance_type == 'tied':
        covs = cov_est

    elif covariance_type == 'full':
        covs = np.stack([cov_est for _ in range(n_components)])

    elif covariance_type == 'diag':
        # each components gets the diagonal of the estimated covariance matrix
        covs = np.diag(cov_est)
        covs = np.repeat(covs.reshape(1, -1),
                         repeats=n_components, axis=0)

    elif covariance_type == 'spherical':
        # each components gets the average of the variances
        covs = np.diag(cov_est).mean()
        covs = np.repeat(covs, repeats=n_components)

    # set means to random data points
    rand_idxs = rng.choice(range(n_samples), replace=False, size=n_components)

    means = [X[pt_idx, ] for pt_idx in rand_idxs]
    means = np.array(means)

    return means, covs
