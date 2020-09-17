import numpy as np
from itertools import product
from sklearn.utils import check_random_state
import networkx as nx


def get_01_block_diag(block_shapes=[(2, 2), (3, 3)],
                      block_weights=None):
    """
    Creates a block diagonal matrix of 0s and 1s.

    Parameters
    ----------
    block_shapes: list of tuples
        Block shapes.

    Output
    ------
    A: array-like, (n_rows, n_cols)
    """

    block_shapes = np.array(block_shapes)
    n_blocks = block_shapes.shape[0]

    if block_weights is not None:
        assert len(block_weights) == n_blocks
    else:
        block_weights = [1] * n_blocks

    A = [[None for _ in range(n_blocks)] for __ in range(n_blocks)]
    for i, j in product(range(n_blocks), range(n_blocks)):

        bs = (block_shapes[i][0], block_shapes[j][1])

        if i == j:
            B = np.ones(bs) * block_weights[i]
        else:
            B = np.zeros(bs)

        A[i][j] = B

    return np.bmat(A).A


def block_diag_pi(block_shapes, block_weights=None):
    """
    Returns block diagonal Pi matrix.

    Parameters
    ----------
    block_shapes: list of tuples
        Block shapes.

    Output
    ------
    Pi: array-like, (n_rows, n_cols)
        Entries sum to 1.
    """
    A = get_01_block_diag(block_shapes=block_shapes,
                          block_weights=block_weights).astype(float)
    Pi = A / A.sum()
    return Pi


def noisey_block_pi(block_shapes=[(2, 2), (3, 3)],
                    sigma=1.0, random_state=None):
    """
    Samples a noisey, block diagonal Pi matrix.
    Creates a block diagonal matrix of 0s and 1s, normalizes
    so the entries sum to 1 then adds noise in the form of U(0, sigma) to each
    entry, followed by a final normalization step.

    Parameters
    ----------
    block_shapes: list of tuples
        Block shapes.

    sigma: float
        Noise level.

    random_state: None, int
        Seed for noise.
    """
    Pi = block_diag_pi(block_shapes=block_shapes)
    Pi = add_noise(Pi, sigma=sigma, random_state=random_state)
    return Pi


def add_noise(Pi, sigma=.1, sigma_kind='rel',
              random_state=None):
    """
    Adds uniform noise to entries of Pi then normalizes so resulting
    matrix sums to 1.

    Parameters
    ----------
    Pi: array-like, (R, C)
        Matrix to add noise to.

    sigma: float
        Noise level.

    sigma_kind: str
        Must be one of ['rel', 'abs']
        If 'rel', multiplies simga by the average entry of Pi.
        If 'abs', sigma stays the same.

    random_state: None, int
        Seed for noise.

    Output
    ------
    Pi: array-like, (R, C)
        Matrix whose entries sum to 1.

    """
    assert sigma_kind in ['rel', 'abs']
    rng = check_random_state(random_state)

    if sigma_kind == 'rel':
        sigma = sigma * Pi.mean()

    E = rng.uniform(low=0, high=sigma, size=Pi.shape)

    return (Pi + E) / (Pi + E).sum()


def sample_sbm(n_blocks=4, block_size=5, p_within=.8, p_between=.4,
               random_state=None):
    """
    Samples a stochastic block model for a binary, undirected graph.

    Parameters
    ----------
    n_blocks: int
        Number of communities.

    block_size: int
        Size of each community.

    p_within, p_between: float
        Probably of within/between community edges.

    random_state: None, int
        Seed.

    Output
    ------
    A: np.array
        Adjaceny matrix.
    """
    sizes = [block_size] * n_blocks
    p = np.ones((n_blocks, n_blocks)) * p_between
    p[np.diag_indices_from(p)] = p_within

    G = nx.generators.stochastic_block_model(sizes, p, seed=random_state)
    return nx.adjacency_matrix(G).A


def sample_gaussian_noise_block_diag(block_shapes=[(2, 2), (3, 3)],
                                     sigma=.5, random_state=None):
    """
    Samples a block diagonal matrix of 0s and 1 then adds gaussian noise.

    Parameters
    ----------
    block_shapes: list of tuples
        Block shapes.

    sigma: float
        Std of the noise.

    random_state: None, int
        Seed.

    Output
    ------
    A: array-like, (n_rows, n_cols)
    """
    rng = check_random_state(random_state)
    A = get_01_block_diag(block_shapes)
    E = rng.normal(size=A.shape, scale=sigma)
    return A + E

