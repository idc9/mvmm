from scipy.sparse import csr_matrix
import numpy as np
from itertools import product, combinations
from scipy.sparse import diags

from mvmm.linalg_utils import svd_wrapper


def get_lin_coef(V, shape, weights=None):
    """
    Returns the coefficient B where <X, B> = tr(V.T Lun(A_bp(X)) V diag(w))

    Parameters
    -----------
    V: array-like, (n_rows + n_cols, n_components)

    n_rows, n_cols: int


    Output
    ------
    lin_coef, array-like, (n_rows, n_cols)
    """

    if weights is None:
        weights = np.ones(V.shape[1])

    # make sure largest weights correspond to smallest evals
    # evals are in descending order so weights should be in ascending order
    weights = asc_sort(weights)

    n_rows, n_cols = shape
    assert V.shape[0] == n_rows + n_cols

    V_rows = V[0:n_rows, :]
    V_cols = V[n_rows:, :]

    lin_coef = np.zeros((n_rows, n_cols))
    for (r, c) in product(range(n_rows), range(n_cols)):
        diff = diags(weights) @ (V_rows[r, :] - V_cols[c, :])
        lin_coef[r, c] = (diff ** 2).sum()

    return lin_coef


def _basis_vec(n, i):
    """
    Returns the ith standard basis vector in R^n

    Parameters
    ----------
    n: int
        Dimension of ambient space.

    i: int
        Which basis vector.
    Output
    ------
    e: array-like, (n, )
    """
    v = np.zeros(n)
    v[i] = 1
    return v


def get_row_col_sum_mat(shape):
    """
    Returns the matrix S \in \mathbb{R}^{(R + C) \times RC)} that gives
    the row and column sums of a matrix X \in \mathbb{R}^{R \times C}


    """

    n_rows, n_cols = shape
    top = np.zeros((n_rows, n_rows * n_cols))
    for k in range(n_rows):
        top[k, k * n_cols:(k + 1) * n_cols] = np.ones(n_cols)
    bottom = np.zeros((n_cols, n_rows * n_cols))
    for k in range(n_cols):
        bottom[k, :] = np.concatenate([_basis_vec(n_cols, k)] * n_rows)
    S = np.vstack([top, bottom])
    S = csr_matrix(S)

    return S


def get_VdV_mat(V, trim_od_constrs=False):
    """
    For a vector d gets the linear transform corresponding to
    d --> V^T diag(d) V
    or
    V^T diag(d) V diag(w)

    Parameters
    ----------
    V: array-like, (n, K)
        The matrix.

    trim_od_constrs: bool
        Replace the linear equality constraints for the off diagonal terms with an equivalent, but potentially smaller matrix.

    Output
    ------
    diag_mat: array-like, (K, )

    upper_tri: (K choose 2, d)
        The upper trianglar entries in row major order.
    """
    d, K = V.shape

    # matrix corresponding to the linear constraints on the diagonal
    diag_mat = V.T ** 2

    # corresponding to the off diagonal constraints.
    # hadamard products between all columns of V
    upper_tri = np.array([V[:, i] * V[:, j] for i, j in
                         combinations(range(K), 2)])

    # Replace off_diag_constr_mat with an orthnormal basis
    # spanning its row space and possibly remove redundant constraints
    # by trimming singular vecs whose svals are 0
    if trim_od_constrs and len(upper_tri) > 0:

        sval_cutoff = 1e-10
        _, svals, right_svecs = svd_wrapper(upper_tri)
        non_zero_sval_mask = svals > sval_cutoff
        right_svecs = right_svecs[:, non_zero_sval_mask]
        upper_tri = right_svecs.T

    return diag_mat, upper_tri


def get_guess(log_coef, lin_coef, epsilon, epsilon_tilde):
    """
    Guesses the solution to the sym-lin-lin problem using the following heuristic

    argmin_{x >= 0} - a log(epsilon + x) + bx
    is given by (a/b) - epsilon if (a/b) > epsilon and 0 otherwise

    """
    shape = log_coef.shape
    assert shape == lin_coef.shape

    def _guess(a, b):

        if b == 0:
            return a

        else:
            stat_pt = (a / b) - epsilon
            if stat_pt < 0:
                return 0
            else:
                return stat_pt

    unnorm_guess = np.zeros(shape)
    for idx in np.ndindex(shape):
        unnorm_guess[idx] = _guess(a=log_coef[idx], b=lin_coef[idx])

    # if all are zero, guess uniform
    if unnorm_guess.sum() == 0:
        unnorm_guess = np.ones(shape)

    norm_guess = unnorm_guess * epsilon_tilde / unnorm_guess.sum()

    return norm_guess


def asc_sort(x):
    return np.sort(x)


def desc_sort(x):
    return np.sort(x)[::-1]
