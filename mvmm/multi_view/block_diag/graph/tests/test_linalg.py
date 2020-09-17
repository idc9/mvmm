import numpy as np
from itertools import product

from mvmm.multi_view.block_diag.graph.linalg import get_sym_laplacian_bp, \
    eigh_Lsym_bp_from_Tsym, get_unnorm_laplacian_bp, geigh_sym_laplacian_bp, \
    get_deg_bp
from mvmm.linalg_utils import eigh_wrapper


def test_eigh_Lsym_bp_from_Tsym():
    np.random.seed(234)

    Xs = []

    X = np.random.uniform(size=(5, 10))
    Xs.append(X)

    X = np.random.uniform(size=(5, 5))
    Xs.append(X)

    # TODO: make direct handle 0 rows/columns
    # X = np.random.uniform(size=(5, 5))
    # X[0, :] = 0
    # X[:, 0] = 0
    # Xs.append(X)

    for X, rank, method in product(Xs, [None, 3], ['direct', 'tsym']):
        check_eigh_Lsym_bp_from_Tsym(X=X, rank=rank)


def check_eigh_Lsym_bp_from_Tsym(X, rank=None):
    """
    Checks the output of get_sym_laplacian_bp
    """

    Lsym = get_sym_laplacian_bp(X)
    true_evals, true_evecs = eigh_wrapper(Lsym)

    if rank is None:
        _rank = min(X.shape)
    else:
        _rank = rank

    # check largest eigenvectors
    evals, evecs = eigh_Lsym_bp_from_Tsym(X, end='largest', rank=rank)
    for k in range(len(evals)):

        # check the evals are correct
        assert np.allclose(evals[k], true_evals[k])

        if not np.allclose(evals[k], 1):  # non-unique subspace for 1 evals
            # check eigenvectors point in the same direction
            a = angle(true_evecs[:, k], evecs[:, k], subspace=True)
            assert a < 1e-4

        # check normalization
        assert np.allclose(evecs.T @ evecs, np.eye(evecs.shape[1]))

    # check smallest eigenvectors
    evals, evecs = eigh_Lsym_bp_from_Tsym(X, end='smallest', rank=rank)
    base_idx = sum(X.shape) - min(X.shape) + (min(X.shape) - _rank)
    for k in range(len(evals)):

        # check the evals are correct
        assert np.allclose(evals[k], true_evals[base_idx + k])

        if not np.allclose(evals[k], 1):  # non-unique subspace for 1 evals
            # check eigenvectors point in the same direction
            a = angle(true_evecs[:, base_idx + k], evecs[:, k], subspace=True)
            assert a < 1e-4

        # check normalization
        assert np.allclose(evecs.T @ evecs, np.eye(evecs.shape[1]))


def test_geigh_Lsym_bp_from_Tsym():
    np.random.seed(234)

    Xs = []

    X = np.random.uniform(size=(5, 10))
    Xs.append(X)

    X = np.random.uniform(size=(5, 5))
    Xs.append(X)

    X = np.random.uniform(size=(5, 5))

    X[0, :] = 0
    X[:, 0] = 0
    Xs.append(X)

    for X, rank, method in product(Xs, [None, 3], ['direct', 'tsym']):
        check_geigh_Lsym_bp_from_Tsym(X=X, rank=rank, method=method)


def check_geigh_Lsym_bp_from_Tsym(X, rank=None, method='direct'):
    """
    Checks the output of geigh_Lsym_bp_from_Tsym
    """
    Lun = get_unnorm_laplacian_bp(X)
    degs = get_deg_bp(X)
    true_gevals, true_gevecs = eigh_wrapper(A=Lun, B=np.diag(degs))

    if rank is None:
        _rank = min(X.shape)
    else:
        _rank = rank

    # check largest eigenvectors
    gevals, gevecs = geigh_sym_laplacian_bp(X=X, rank=rank, method=method,
                                            end='largest')
    for k in range(len(gevals)):

        # check the gevals are correct
        assert np.allclose(gevals[k], true_gevals[k])

        if not np.allclose(gevals[k], 1):  # non-unique subspace for 1 evals

            # check the gen evecs span the correct subspaces
            a = angle(gevecs[:, k], true_gevecs[:, k], subspace=True)
            assert a < 1e-4

        # check proper normalization
        assert np.allclose(gevecs.T @ np.diag(degs) @ gevecs,
                           np.eye(gevecs.shape[1]))

    # check smallest eigenvectors
    gevals, gevecs = geigh_sym_laplacian_bp(X=X, rank=rank, method=method,
                                            end='smallest')
    base_idx = sum(X.shape) - min(X.shape) + (min(X.shape) - _rank)
    for k in range(len(gevals)):
        # print(gevals[k], true_gevals[base_idx + k])

        # check the gevals are correct
        assert np.allclose(gevals[k], true_gevals[base_idx + k])

        if not np.allclose(gevals[k], 1):  # non-unique subspace for 1 evals
            # check the gen evecs span the correct subspaces
            a = angle(gevecs[:, k], true_gevecs[:, base_idx + k],
                      subspace=True)
            assert a < 1e-4

        # check proper normalization
        assert np.allclose(gevecs.T @ np.diag(degs) @ gevecs,
                           np.eye(gevecs.shape[1]))


def angle(u, v, subspace=False):
    """
    Computes the angle (in degrees) between two vectors (or between the subspaces spanned by two vectors).

    Parameters
    ----------
    u, v: array-like
        The vectors.

    subspace: bool
        Compute the angle between the subspaces spanned by u and v

    """
    u = np.array(u).reshape(-1)
    v = np.array(v).reshape(-1)

    c = u.T @ v / max(np.linalg.norm(u) * np.linalg.norm(u), 1e-12)
    c = np.clip(c, a_min=-1, a_max=1)

    a = np.rad2deg(np.arccos(c))

    if subspace:
        a = min(a, 180 - a)

    return a
