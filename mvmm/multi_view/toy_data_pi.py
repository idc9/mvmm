import numpy as np
from sklearn.utils import check_random_state


def sample_sparse_Pi(n_rows_base=5, n_cols_base=8,
                     density=.8, random_state=None):
    """
    Samples a sparse Pi where entries are present as iid bernoullis.

    Parameters
    ----------
    n_rows_base, n_cols_base: int
        Number of rows/columns of the initial matrix whose entries will be sub-samples.

    sparsity: float
        Bernoulli probability -- probability of an entry being present density.

    random_state: None, int
        Seed to generate sparsity pattern.

    Output
    ------
    Pi: (n_rows, n_cols)
        Cluster probability matrix. Entries sum to 1.
        n_rows and n_cols may be less than n_rows_base, n_cols_basee
    """
    rng = check_random_state(random_state)
    non_zero_entries = rng.choice(a=[0, 1],
                                  p=(1 - density, density),
                                  size=n_rows_base * n_cols_base)

    # in degenerate case make sure we have at least one non-zero entry
    if sum(non_zero_entries) == 0:
        non_zero_entries = np.zeros(n_rows_base * n_cols_base)
        non_zero_entries[0] = 1

    A = non_zero_entries.reshape((n_rows_base, n_cols_base))

    # remove rows/cols of all zeros
    zero_rows = A.sum(axis=1) == 0
    zero_cols = A.sum(axis=0) == 0
    A = A[~zero_rows, :][:, ~zero_cols]

    # make Pi matrix
    Pi = A.astype(float)
    Pi = Pi / Pi.sum()
    return Pi


def sample_sparse_Pi_force_full_rowcols(n_rows=5, n_cols=8,
                                        density=.8, random_state=None):
    """
    Samples a sparse Pi ensuring every row has at least one non-zero column and every
    column has at least one non-zero row.

    Parameters
    ----------
    n_rows, n_cols: int
        Number of rows/columns of Pi.

    density: float
        Targeted proportion of non-zero entries of Pi.

    random_state: None, int
        Seed to generate sparsity pattern.

    Output
    ------
    Pi: (n_rows, n_cols)
        Cluster probability matrix. Entries sum to 1.
    """
    rng = check_random_state(random_state)
    q = 1 - np.sqrt(1 - density)

    Ar = np.zeros((n_rows, n_cols), dtype=bool)
    for r in range(n_rows):
        rand_mask = rng.choice(a=range(n_cols),
                               size=np.ceil(q * n_cols).astype(int),
                               replace=False)
        Ar[r, rand_mask] = True

    Ac = np.zeros((n_rows, n_cols), dtype=bool)
    for c in range(n_cols):
        rand_mask = rng.choice(a=range(n_rows),
                               size=np.ceil(q * n_rows).astype(int),
                               replace=False)
        Ac[rand_mask, c] = True

    A = Ar | Ac
    A = A.astype(float)

    n_components_tot = A.sum()

    Pi = A / n_components_tot

    return Pi


def sample_dense_pi(n_rows, n_cols, entries='equal', random_state=None):
    """
    Creates a dense Pi matrix.

    Parameters
    ----------
    n_rows, n_cols: int
        Number of rows/columns

    entries: str
        Either 'random' or 'equal'

    random_state: None, int
        Random state to generate random entries of Pi.

    Output
    ------
    Pi: array-like, (n_rows, n_cols:

    """
    assert entries in ['equal', 'random']
    if entries == 'equal':
        Pi = np.ones((n_rows, n_cols), dtype=float)

    else:
        rng = check_random_state(random_state)
        Pi = rng.uniform(size=(n_rows, n_cols))

    return Pi / Pi.sum()


def view_indep_pi(n_comp):
    """
    Pi matrix where all entries are equal i.e. views are independent.

    Parameters
    ----------
    n_comp: tuple of ints
        Number of components in each view.
    """
    Pi = np.ones((n_comp, n_comp))
    Pi = Pi / Pi.sum()
    return Pi


def motivating_ex():
    Pi = [[1] + [0] * 9,
          [0, 1] + [0] * 8,
          [0] * 2 + [1] * 3 + [0] * 5,
          [0] * 5 + [1] + [0] * 4,
          [0] * 5 + [1] + [0] * 4,
          [0] * 6 + [1] * 2 + [0] * 2,
          [0] * 6 + [1] * 2 + [0] * 2,
          [0] * 8 + [1] * 2,
          [0] * 8 + [1] * 2,
          [0] * 8 + [1, 0]]

    Pi = np.array(Pi)
    Pi = Pi / Pi.sum()
    return Pi
