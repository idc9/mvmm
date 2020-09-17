"""
Linear algebra utils for graphs e.g. adjacency matrix, Laplacians, etc
"""
import numpy as np
from scipy.sparse import diags
from scipy.linalg import LinAlgError
from sklearn.utils.extmath import svd_flip

from mvmm.utils import safe_invert
from mvmm.linalg_utils import eigh_wrapper, svd_wrapper


def get_unnorm_laplacian(A):
    """
    Returns the unnormalized laplacian from an adjacency matrix.
    """
    d = A.sum(axis=1)
    return np.diag(d) - A


def get_sym_laplacian(A):
    """
    Returns the symetric laplacian from an adjacency matrix.
    """
    d = A.sum(axis=1)
    d_sqrt_inv = np.sqrt(safe_invert(d))

    return np.eye(len(d)) - diags(d_sqrt_inv) @ A @ diags(d_sqrt_inv)


def get_rw_laplacian(A):
    """
    Returns the random walk laplacian from an adjacency matrix.
    """
    d = A.sum(axis=1)

    d_inv = safe_invert(d)
    return diags(d_inv) @ get_unnorm_laplacian(A)


def get_adjmat_bp(X):
    """
    Returns the weighted adjacency matrix of the bi-partide graph whose
    vertex sets are the rows/columns of X and whose edge weights are the
    i, jth entries of X.
    """
    R, C = X.shape
    return np.block([[np.zeros((R, R)), X],
                     [X.T, np.zeros((C, C))]])


def get_unnorm_laplacian_bp(X):
    """
    Returns the laplacian of the bipartite graph
    """
    # TODO: can probably do this a touch faster
    A = get_adjmat_bp(X)
    return get_unnorm_laplacian(A)


def get_sym_laplacian_bp(X):
    """
    Returns the laplacian of the bipartite graph
    """
    # TODO: can probably do this a touch faster
    A = get_adjmat_bp(X)
    return get_sym_laplacian(A)


def get_rw_laplacian_bp(X):
    """
    Returns the random walk laplacian of the bipartite graph.
    """
    A = get_adjmat_bp(X)
    return get_unnorm_laplacian_bp(A)


def get_deg_bp(X):
    """
    Gets the degrees of A_bp(X)
    """
    A = get_adjmat_bp(X)
    return A.sum(axis=0)


def get_Tsym(X):
    """
    Computes Tsym (i.e. the upper triangluar of Lsym(A_bp(X))).
    See (Carmichael, 2020)

    Parameters
    ----------
    X: array-like, (R, C)
        The input matrix.

    Output
    ------
    T_sym: array-like, (R, C)
    """
    row_sums = X.sum(axis=1)
    col_sums = X.sum(axis=0)

    row_sums_inv_sqrt = np.sqrt(safe_invert(row_sums))
    col_sums_inv_sqrt = np.sqrt(safe_invert(col_sums))

    return diags(row_sums_inv_sqrt) @ X @ diags(col_sums_inv_sqrt)


def eigh_sym_laplacian_bp(X, rank=None):
    Lsym = get_sym_laplacian_bp(X)
    return eigh_wrapper(Lsym, rank=rank)


def eigh_Lsym_bp_from_Tsym(X, rank=None, end='smallest', zero_tol=1e-8):
    """
    Computes the smallest or largest eigenvectors of Lsym(A_bp(X)) via the SVD
    of Tsym(A_bp(X)).

    Parameters
    ----------
    X: array-like, (R, X)
        The data matrix.

    rank: None, int
        The number of eigenvectors to compute. Must be at most min(X.shape)

    end: str
        Compute the smallest or the largest eigenvectors.
        Must be one of ['smallest', 'largest'].

    zero_tol: float
        Tolerance for zero rows/columns

    Output
    ------
    evals, evec

    evals: array-like, (rank, )
        The eigenvalues in decreasing order

    evecs: array-like, (R + C, rank)
        The eigenvectors

    """
    assert end in ['smallest', 'largest']

    if rank is None:
        rank = min(X.shape)
    rank = min(rank, min(X.shape))

    # SVD of Tsym
    Tsym = get_Tsym(X)
    Tsym_left_svecs, Tsym_svals, Tsym_right_svecs = svd_wrapper(Tsym,
                                                                rank=rank)

    if end == 'smallest':
        evals = 1 - Tsym_svals
        evals = evals[::-1]

        evecs = np.vstack([Tsym_left_svecs, Tsym_right_svecs])
        evecs = evecs[:, ::-1]

    elif end == 'largest':
        evals = 1 + Tsym_svals
        evecs = np.vstack([Tsym_left_svecs, -Tsym_right_svecs])

    # ensure normalization
    col_norms = np.linalg.norm(evecs, axis=0)
    evecs = evecs @ diags(1 / col_norms)

    # deterministic output
    evecs = svd_flip(evecs, evecs.T)[0]

    return evals, evecs


# def geigh_sym_laplacian_bp(X, rank=None):
#     """
#     Computes the generalized eigenvectors of the symmetric laplacian.
#     """
#     Lun = get_unnorm_laplacian_bp(X)
#     degs = get_deg_bp(X)
#     return eigh_wrapper(A=Lun, B=np.diag(degs), rank=rank)


def geigh_sym_laplacian_bp(X, rank=None, end='smallest', method='tsym'):
    """
    Computes the smallest or largest generalized eigenvectors of
    (Lun(A_bp(X)), deg(A_bp(X))).

    Parameters
    ----------
    X: array-like, (R, C)
        The data matrix.
    rank: None, int
        The number of eigenvectors to compute. Must be at most min(X.shape)

    end: str
        Compute the smallest or the largest eigenvectors.
        Must be one of ['smallest', 'largest'].

    method: str
        Must be one of ['direct', tsym].
        If 'direct' then we directly computed the generalized eigenvectors.
        If tsym then we compute them more quickly usings the SVD of Tsym(A_bp(X)).

    Output
    ------
    gevals, gevec

    gevals: array-like, (rank, )
        The eigenvalues in decreasing order.

    gevecs: array-like, (R + C, rank)
        The generalized eigenvectors such that
        gevecs.T @ diag(deg(A_bp(X))) gevecs == I


    """
    assert method.lower() in ['direct', 'tsym']

    # TODO-FEAT
    # we have to modify direct method to handle zero rows/columns
    if method == 'direct':
        if sum(np.linalg.norm(X, axis=0) < 1e-9) > 0:
            raise NotImplementedError("Direct method does not currently"
                                      " support case when X has zero columns")
        if sum(np.linalg.norm(X, axis=1) < 1e-9) > 0:
            raise NotImplementedError("Direct method does not currently "
                                      "support case when X has rows columns")

    if rank is None:
        rank = min(X.shape)
    rank = min(rank, min(X.shape))

    if method.lower() == 'direct':
        if end == 'largest':
            Lun = get_unnorm_laplacian_bp(X)
            degs = get_deg_bp(X)
            gevals, gevecs = eigh_wrapper(A=Lun, B=np.diag(degs), rank=rank)

        elif end == 'smallest':
            Lun = get_unnorm_laplacian_bp(X)
            degs = get_deg_bp(X)
            gevals, gevecs = eigh_wrapper(A=-Lun, B=np.diag(degs), rank=rank)
            gevals = - gevals

            gevals = gevals[::-1]
            gevecs = gevecs[:, ::-1]

        return gevals, gevecs

    elif method.lower() == 'tsym':

        gevals, evecs = eigh_Lsym_bp_from_Tsym(X, rank=rank, end=end)
        degs = get_deg_bp(X)
        degs_inv_sqrt = np.sqrt(safe_invert(degs))
        gevecs = diags(degs_inv_sqrt) @ evecs

        return gevals, gevecs
