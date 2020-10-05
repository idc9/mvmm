"""
Linear algebra utils for graphs e.g. adjacency matrix, Laplacians, etc
"""
import numpy as np
from scipy.sparse import diags
from sklearn.utils.extmath import svd_flip
from scipy.sparse.linalg import ArpackNoConvergence, ArpackError
from scipy.linalg import svd as full_svd

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


def get_Tsym(X, zero_tol=None):
    """
    Computes Tsym (i.e. the upper triangluar of Lsym(A_bp(X))).
    See (Carmichael, 2020)

    Parameters
    ----------
    X: array-like, (R, C)
        The input matrix.

    zero_tol: None, float
        Tolerance to check for zero rows/columns to ensure we zero these out.

    Output
    ------
    T_sym: array-like, (R, C)
    """
    row_sums = X.sum(axis=1)
    col_sums = X.sum(axis=0)

    row_sums_inv_sqrt = np.sqrt(safe_invert(row_sums))
    col_sums_inv_sqrt = np.sqrt(safe_invert(col_sums))

    Tsym = diags(row_sums_inv_sqrt) @ X @ diags(col_sums_inv_sqrt)

    # possible make make sure zero rows/columns are handled correctly
    if zero_tol is not None:
        zero_row_mask = np.linalg.norm(X, axis=1) < zero_tol
        zero_col_mask = np.linalg.norm(X, axis=0) < zero_tol
        Tsym[zero_row_mask, :] = 0
        Tsym[:, zero_col_mask] = 0

    return Tsym


def eigh_Lsym_bp(X, rank=None):
    """
    Computes the largest eigenvectors of Lsym(A_bp(X)) directly using scipy.linalg.eigh.

    Paramters
    ---------
    X: array-like, (n_rows, n_cols)
        The data matrix.

    rank: None, int
        The rank to compute.

    Output
    ------
    evals, evecs

    evals: array-like, (rank, )
        The largest evals Lsym(A_bp(X)).

    evecs: array-like, (n_row + n_cols, rank)
        The corresponding eigenvectors.

    """
    Lsym = get_sym_laplacian_bp(X)
    return eigh_wrapper(Lsym, rank=rank)


# def geigh_sym_laplacian_bp(X, rank=None):
#     print("DELETE ME")
#     # TODO: delete this
#     return geigh_Lsym_bp(X, rank)

# def eigh_sym_laplacian_bp(X, rank=None):
#     print("DELETE ME")
#     # TODO: delete this
#     return eigh_Lsym_bp(X, rank)


def geigh_Lsym_bp(X, rank=None, zero_tol=1e-10, end='smallest'):
    """
    Computes the largest or smallest generalized eigenvectors of
    [Lun(A_bp(X)), deg(A_bp(X))] directly using scipy.linalg.eigh.

    Paramters
    ---------
    X: array-like, (n_rows, n_cols)
        The data matrix.

    rank: None, int
        The rank to compute. If None, will compute as many gevals as possible. This will depend on the number of zero rows/columns X.

    zero_tol: float
        Tolerance to identify zero rows/columns by their norm.

    end: str
        Must be one of ['smallest', 'largest'].
        Compute the smallest or largest generalized eigenvectors.

    Output
    ------
    gevals, gevecs

    gevals: array-like, (rank, )
        The smallest or largest generalized eigenvalues.

    gevecs: array-like, (n_rows + n_cols, rank)
        The corresponding generalized eigenvectors.
        Normalized such that gevecs.T @ deg(A_bp(X)) gevecs = I

    """

    assert end in ['smallest', 'largest']

    # get X without its zero rows/columns
    zero_row_mask = np.linalg.norm(X, axis=1) < zero_tol
    zero_col_mask = np.linalg.norm(X, axis=0) < zero_tol
    X_woz = X[~zero_row_mask, :][:, ~zero_col_mask]

    if rank is None:
        rank = min(X_woz.shape)
    assert 1 <= rank and rank <= sum(X.shape)

    if rank > sum(X_woz.shape):
        raise ValueError("X has too many zero rows/columns.")

    # compute generalized eigenvectors/values for X without its
    # zero rows and columns
    Lun = get_unnorm_laplacian_bp(X_woz)
    degs = get_deg_bp(X_woz)

    if end == 'largest':
        gevals, gevecs_woz = eigh_wrapper(A=Lun, B=np.diag(degs),
                                          rank=rank)

    elif end == 'smallest':
        gevals, gevecs_woz = eigh_wrapper(A=-Lun, B=np.diag(degs),
                                          rank=rank)
        gevals = - gevals
        gevals = gevals[::-1]
        gevecs_woz = gevecs_woz[:, ::-1]

    # get gen eval/vecs for X by putting zeros back into gen evectors
    gevecs = np.zeros((sum(X.shape), rank))
    non_zero_mask = ~ np.concatenate([zero_row_mask, zero_col_mask])
    gevecs[non_zero_mask, :] = gevecs_woz

    return gevals, gevecs


def safe_tsym_svd(X, rank=None, full=False):
    """
    Safely computes the SVD of Tsym.
    Computing the low rank SVD of Tsym sometimes results in arpack errors
    so we may have to resort to computing full SVD as back up.

    Parameters
    ----------
    X: array-like, (n_rows, n_cols)
        The data matrix.

    rank: int, None
        The rank of the SVD to compute.

    full: bool
        If True, computes the "full" singular vectors i.e. both singular vectors are orthonormal matrices.

    Output
    ------
    U, D, V

    """
    # SVD of Tsym
    Tsym = get_Tsym(X)

    if full:
        assert rank is None
        U, D, V = full_svd(Tsym, full_matrices=True)
        V = V.T

    else:
        try:
            # for some reason ArpackNoConvergence sometimes fails to converge
            # for low rank
            U, D, V = svd_wrapper(Tsym, rank=rank)

        except (ArpackNoConvergence, TypeError, RuntimeError, ArpackError):

            U, D, V = svd_wrapper(Tsym, rank=None)

            U = U[:, 0:rank]
            D = D[0:rank]
            V = V[:, 0:rank]

    return U, D, V


def all_non_zero(v):
    """
    Checks if all entries are non-zero.
    """
    for x in v:
        if np.allclose(x, 0):
            return False

    return True


def smallest_eigh_Lsym_bp_from_Tsym_no_zeros(X, rank=None):
    """
    Computes the smallest eigenvectors K of Lsym(A_bp(X) via the SVD
    of Tsym(A_bp(X)) assuming X has no zero rows or columns.

    Parameters
    ----------
    X: array-like, (R, X)
        The data matrix.

    rank: None, int
        The number of eigenvector/values to compute.
        If None, will compute min(X.shape)

    Output
    ------
    evals, evecs

    evals: array-like, (rank, )
        The smallest eigenvalues of Lsym(Abp(X)) in decreasing order

    evecs: array-like, (R + C, rank)
        The corresponding eigenvectors

    """

    if rank is None:
        rank = min(X.shape)
    assert 1 <= rank and rank <= sum(X.shape)

    # make sure X does not have any zero rows or columns
    # assert all_non_zero(np.linalg.norm(X, axis=1))
    # assert all_non_zero(np.linalg.norm(X, axis=0))

    # compute appropriate SVD of Tsym
    if rank <= min(X.shape):
        U, svals, V = safe_tsym_svd(X, rank=rank, full=False)
    else:
        U, svals, V = safe_tsym_svd(X, rank=None, full=True)
        svals = svals[0:min(X.shape)]

    # get evecs form svecs
    evecs = np.zeros((sum(X.shape), rank))
    for k in range(rank):

        if k < min(X.shape):
            evecs[:, k] = np.concatenate([U[:, k], V[:, k]])

        elif k >= min(X.shape) and k < max(X.shape):
            # j = k - min(X.shape)

            if X.shape[0] > X.shape[1]:
                evecs[:, k] = np.concatenate([U[:, k], np.zeros(X.shape[1])])
            elif X.shape[0] < X.shape[1]:
                evecs[:, k] = np.concatenate([np.zeros(X.shape[0]), V[:, k]])

        elif k >= max(X.shape):
            j = k - max(X.shape)
            ell = min(X.shape) - j - 1  # index of the the jth smallest singular vectors

            evecs[:, k] = np.concatenate([U[:, ell], -V[:, ell]])

    # ensure normalization
    col_norms = np.linalg.norm(evecs, axis=0)
    evecs = evecs @ diags(1 / col_norms)
    evecs = svd_flip(evecs, evecs.T)[0]  # deterministic output

    evecs = evecs[:, ::-1]  # decreasing order

    # get eigenvalues form svals
    evals = 1 - svals
    if rank > min(X.shape):
        n_ones_to_add = min(rank, max(X.shape)) - min(X.shape)
        evals = np.concatenate([evals, np.ones(n_ones_to_add)])

    if rank > max(X.shape):
        meow = rank - max(X.shape)
        evals = np.concatenate([evals, 1 + svals[-meow:]])
    evals = np.sort(evals)[::-1]

    return evals, evecs


def smallest_geigh_Lsym_bp_from_Tsym(X, rank=None, zero_tol=1e-10):
    """
    Computes the smallest generalized eigenvectors of
    [Lun(A_bp(X)), deg(A_bp(X))] via the SVD of Tsym.

    Paramters
    ---------
    X: array-like, (n_rows, n_cols)
        The data matrix.

    rank: None, int
        The rank to compute. If None, will compute as many gevals as possible. This will depend on the number of zero rows/columns X.

    zero_tol: float
        Tolerance to identify zero rows/columns by their norm.

    Output
    ------
    gevals, gevecs

    gevals: array-like, (rank, )
        The smallest or largest generalized eigenvalues.

    gevecs: array-like, (n_rows + n_cols, rank)
        The corresponding generalized eigenvectors.
        Normalized such that gevecs.T @ deg(A_bp(X)) gevecs = I
    """
    # get X without its zero rows/columns
    zero_row_mask = np.linalg.norm(X, axis=1) < zero_tol
    zero_col_mask = np.linalg.norm(X, axis=0) < zero_tol
    X_woz = X[~zero_row_mask, :][:, ~zero_col_mask]
    if rank is None:
        rank = min(X_woz.shape)
    assert 1 <= rank and rank <= sum(X.shape)

    if rank > sum(X_woz.shape):
        raise ValueError("X has too many zero rows/columns.")

    # compute eig decompot of X without zero rows/columns
    evals, evecs_woz = smallest_eigh_Lsym_bp_from_Tsym_no_zeros(X=X_woz,
                                                                rank=rank)

    # put back in zero rows
    evecs = np.zeros((sum(X.shape), rank))
    non_zero_mask = ~ np.concatenate([zero_row_mask, zero_col_mask])
    evecs[non_zero_mask, :] = evecs_woz

    # normalize gevecs
    degs = get_deg_bp(X)
    degs_inv_sqrt = np.sqrt(safe_invert(degs))
    gevecs = diags(degs_inv_sqrt) @ evecs

    return evals, gevecs


def geigh_Lsym_bp_smallest(X, rank=None, zero_tol=1e-10, method='tsym'):
    """
    Computes the smallest generalized eigenvectors of
    [Lun(A_bp(X)), deg(A_bp(X))] via the SVD of Tsym.

    Paramters
    ---------
    X: array-like, (n_rows, n_cols)
        The data matrix.

    rank: None, int
        The rank to compute. If None, will compute as many gevals as possible. This will depend on the number of zero rows/columns X.

    zero_tol: float
        Tolerance to identify zero rows/columns by their norm.

    method: str
        How to compute these generalized eigenvalues.
        Must be one of ['tsym', 'direct'].

    Output
    ------
    gevals, gevecs

    gevals: array-like, (rank, )
        The smallest or largest generalized eigenvalues.

    gevecs: array-like, (n_rows + n_cols, rank)
        The corresponding generalized eigenvectors.
        Normalized such that gevecs.T @ deg(A_bp(X)) gevecs = I
    """
    assert method in ['tsym', 'direct']

    if method == 'tsym':
        return smallest_geigh_Lsym_bp_from_Tsym(X=X, rank=rank,
                                                zero_tol=zero_tol)
    elif method == 'tsym':
        return geigh_Lsym_bp(X=X, rank=rank, zero_tol=zero_tol, end='smallest')

# def geigh_sym_laplacian_bp(X, rank=None, end='smallest', method='tsym',
#                            zero_tol=1e-8):
#     """
#     Computes the smallest or largest generalized eigenvectors of
#     (Lun(A_bp(X)), deg(A_bp(X))).

#     Parameters
#     ----------
#     X: array-like, (R, C)
#         The data matrix.
#     rank: None, int
#         The number of eigenvectors to compute. Must be at most min(X.shape)

#     end: str
#         Compute the smallest or the largest eigenvectors.
#         Must be one of ['smallest', 'largest'].

#     method: str
#         Must be one of ['direct', tsym].
#         If 'direct' then we directly computed the generalized eigenvectors.
#         If tsym then we compute them more quickly usings the SVD of Tsym(A_bp(X)).

#     zero_tol: float
#         Tolerance for L2 norm to determine zero rows/columns.

#     Output
#     ------
#     gevals, gevec

#     gevals: array-like, (rank, )
#         The eigenvalues in decreasing order.

#     gevecs: array-like, (R + C, rank)
#         The generalized eigenvectors such that
#         gevecs.T @ diag(deg(A_bp(X))) gevecs == I

#     """
#     assert method.lower() in ['direct', 'tsym']

#     if rank is None:
#         rank = min(X.shape)

#     if rank > min(X.shape):
#         raise ValueError("Only computes min(X.shape) gen eigenval/vecs")

#     # get X without its zero rows/columns
#     zero_row_mask = np.linalg.norm(X, axis=1) < zero_tol
#     zero_col_mask = np.linalg.norm(X, axis=0) < zero_tol
#     X_woz = X[~zero_row_mask, :][:, ~zero_col_mask]
#     if rank > sum(X_woz.shape):
#         raise ValueError("X has too many zero rows/columns.")

#     # TODO-FEAT: get Tsym working in this case.
#     if rank > min(X_woz.shape) and end == 'largest':
#         method = 'direct'

#     # compute generalized eigenvectors of X without zeros
#     if method.lower() == 'direct':
#         Lun = get_unnorm_laplacian_bp(X_woz)
#         degs = get_deg_bp(X_woz)

#         if end == 'largest':
#             gevals, gevecs_woz = eigh_wrapper(A=Lun, B=np.diag(degs),
#                                               rank=rank)

#         elif end == 'smallest':
#             gevals, gevecs_woz = eigh_wrapper(A=-Lun, B=np.diag(degs),
#                                               rank=rank)
#             gevals = - gevals
#             gevals = gevals[::-1]
#             gevecs_woz = gevecs_woz[:, ::-1]

#     elif method.lower() == 'tsym':
#         gevals, evecs = eigh_Lsym_bp_from_Tsym(X_woz, rank=rank, end=end)
#         degs = get_deg_bp(X_woz)
#         degs_inv_sqrt = np.sqrt(safe_invert(degs))
#         gevecs_woz = diags(degs_inv_sqrt) @ evecs

#     # get gen eval/vecs for X by putting zeros back into gen evectors
#     gevecs = np.zeros((sum(X.shape), rank))
#     non_zero_mask = ~ np.concatenate([zero_row_mask, zero_col_mask])
#     gevecs[non_zero_mask, :] = gevecs_woz

#     return gevals, gevecs
