import numpy as np
from sklearn.utils import check_random_state


def safe_invert(x):
    """
    Safely inverts a vector that may have 0s by setting 1/0 = 0
    """
    x = np.array(x)
    assert x.ndim == 1

    not_zero_mask = [not np.allclose(x[i], 0, rtol=1e-05, atol=1e-08)
                     for i in range(len(x))]
    inv = np.zeros(len(x))
    inv[not_zero_mask] = 1.0 / x[not_zero_mask]

    return inv


def get_seeds(n_seeds, random_state=None):
    """
    Samples a set of seeds.

    Parameters
    ----------
    n_seeds: int
        Number of seeds to generate.

    random_state: None, int
        Metaseed used to generate the seeds.
    """
    rng = check_random_state(random_state)
    # return rng.randint(low=0, high=2**32 - 1, size=n_seeds)
    return np.array([sample_seed(rng=rng) for _ in range(n_seeds)])


def sample_seed(rng):
    """
    Samples a random seed.
    """
    return rng.randint(low=0, high=2**32 - 1, size=1).item()
