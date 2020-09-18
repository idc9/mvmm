import numpy as np
from itertools import combinations
import cvxpy as cp
from copy import deepcopy

from mvmm.opt_utils import remove_redundant_lin_contrs
from mvmm.multi_view.block_diag.reshaping_utils import to_mat_shape, to_vec_shape

from mvmm.multi_view.block_diag.utils import get_lin_coef, \
    get_row_col_sum_mat, get_guess, get_VdV_mat


def get_cp_problem_sym_lap(Gamma,
                           eig_var,
                           epsilon,
                           B,
                           alpha,
                           eta=None,
                           weights=None,
                           init_val=None,
                           trim_od_constrs=False,
                           remove_redundant_contr=False,
                           remove_const_cols=True,
                           exclude_off_diag=False,
                           exclude_vdv_constr=False,
                           obj_mult=1):
    """
    Sets up the bd_weights_ update for the symmetric Laplacian using cvxpy.
    This is Problem (41) from (Carmichael, 2020).

    min_D - sum_{k1, k2} Gamma_{k1, k2} log(epsilon + D_{k1, k2}) +
        alpha * <D, M(eig_var, weights) >

    s.t. sum_{k1, k2} D_{k1, k1} = 1 - np.product(D.shape) * epsilon

    eig_var.T diag(deg(A_bp(D))) = I_B


    Optional constraint: deg(A_bp(D)) >= eta

    Parameters
    ----------
    Gamma:
        The coefficients of the log terms.

    eig_var:
        Current value of the eigenvector variable.

    epsilon:
        epsilon

    B:
        The number of eigenvalues to penalize.

    alpha:
        The spectral penalty weight.

    eta: None, float
        (Optional) An optional lower bound on the degrees.

    weights: None, array-like, (B, )
        Weights to put on the eigenvalues.

    init_val:
        Guess for the initial value. Note the ECOS solver does not currently
        accept inital guesses.

    trim_od_constrs: bool
        Improves numerical performace by shrinking the number of linear equality constraints. Replace the linear equality constraints of the off diagonal terms in the VDV=I constraint with an equivalent, but potentially smaller matrix.

    remove_redundant_contr: bool
        Improves numerical performace by shrinking the number of linear equality constraints. Remove all redundant linear equality constraints.
        Some solvers really struggle without this step.

    remove_const_cols: bool
        Improves numerical performace by shrinking the number of linear equality constraints. Drop columns of eig_var that are constants.

    exclude_off_diag: bool
        Exclude the constraints corresponding to the off diagonal terms of the VDV constraint. This changes the problem, but can lead to faster, approximate solutions.

    exclude_vdv_constr: bool
        Ignore the VDV=I constraints entirely. This changes the problem, but can lead to faster, approximate solutions.

    obj_mult: float
        Multiply the objective function by a constant. This does not change the problem, but can help some solvers find a solution.
    """

    shape = Gamma.shape
    var = cp.Variable(shape=np.product(shape), pos=True)

    # get data for linear-linear problem
    prob_data = get_sym_lin_lin_data(Gamma=Gamma,
                                     V=eig_var,
                                     alpha=alpha,
                                     epsilon=epsilon,
                                     eta=eta,
                                     weights=weights,
                                     trim_od_constrs=trim_od_constrs,
                                     remove_redundant_contr=remove_redundant_contr,
                                     remove_const_cols=remove_const_cols,
                                     exclude_off_diag=exclude_off_diag,
                                     exclude_vdv_constr=exclude_vdv_constr,
                                     obj_mult=obj_mult)

    # set initial value
    if type(init_val) == str and init_val == 'guess':
        var.value = prob_data['guess'].reshape(-1)
    elif init_val is not None:
        var.value = init_val.reshape(-1)

    log_coef = prob_data['log_coef'].reshape(-1)
    lin_coef = prob_data['lin_coef'].reshape(-1)

    epsilon_tilde = prob_data['epsilon_tilde']

    # setup cvxpy problem
    objective = -log_coef.T @ cp.log(epsilon + var) + lin_coef.T @ var

    constraints = [cp.sum(var) == epsilon_tilde]

    if not exclude_vdv_constr:

        # TODO: probably remove redundant linear constraint
        lin_constr_mat = to_vec_shape(prob_data['lin_constr_mat'],
                                      n_rows=shape[0], n_cols=shape[1],
                                      order='row_major')
        lin_constr_rhs = prob_data['lin_constr_rhs']

        constraints.append(lin_constr_mat @ var == lin_constr_rhs)

    if eta is not None:
        S = get_row_col_sum_mat(shape)
        S_rhs = eta * np.ones(sum(shape))
        constraints.append(S @ var >= S_rhs)

    return var, objective, constraints


def get_sym_lin_lin_data(Gamma, V, alpha, epsilon, eta=None,
                         weights=None,
                         trim_od_constrs=False,
                         remove_redundant_contr=False,
                         remove_const_cols=False,
                         exclude_off_diag=False,
                         exclude_vdv_constr=False,
                         obj_mult=1):
    """
    Returns the problem data for

    min_{X} f(X) + <lin_coef, X_pos + X_neg>
    s.t. X >= 0
         lin_constr_mat vec(X) = lin_constr_rhs


    Parameters
    ----------
    shape: tuple of ints
        Shape of the data matrix.

    V: (n_rows + n_cols, n_components)

    w: None, array-like, (n_components, )

    trim_redundant_od_constr: bool
        Remove redundant off diagonal constraints.

    remove_redundant_contr: bool
        Automatically remove redundant linear equality constraints.


    Output
    ------
    lin_coef, lin_constr_mat, lin_constr_rhs

    lin_coef: (n_rows, n_cols)

    lin_constr_mat: (n_constrs, n_rows, n_cols)

    lin_constr_rhs: (n_constrs, )
    """

    shape = Gamma.shape
    epsilon_tilde = 1 - epsilon * np.product(shape)

    log_coef = deepcopy(Gamma)

    lin_coef = alpha * get_lin_coef(V=V, shape=shape, weights=weights)

    # chaning the scale objective function may improve numerical properties
    if obj_mult is not None:
        log_coef *= obj_mult
        lin_coef *= obj_mult

    guess = get_guess(log_coef=log_coef, lin_coef=lin_coef,
                      epsilon=epsilon, epsilon_tilde=epsilon_tilde)
    # lin_coef = lin_coef.flatten(order=translate_order(order))

    if exclude_vdv_constr:
        lin_constr_mat = None
        lin_constr_rhs = None

    else:
        # get linear constraints
        diag_constr_mat, diag_constr_rhs, \
            off_diag_constr_mat, off_diag_constr_rhs = \
            get_lin_constrs(V, shape,
                            trim_od_constrs=trim_od_constrs,
                            remove_const_cols=remove_const_cols)

        # vectorize constrains so this is in matrix form
        # diag_constr_mat = to_vec_shape(diag_constr_mat, n_rows, n_cols,
        #                                order=order)
        # off_diag_constr_mat = to_vec_shape(off_diag_constr_mat, n_rows, n_cols,
        #                                    order=order)

        if exclude_off_diag:
            lin_constr_mat = diag_constr_mat
            lin_constr_rhs = diag_constr_rhs

        else:
            lin_constr_mat = np.vstack([diag_constr_mat, off_diag_constr_mat])
            lin_constr_rhs = np.concatenate([diag_constr_rhs,
                                             off_diag_constr_rhs])

        # possible drop redundant linear constraints
        if remove_redundant_contr:
            n_rows, n_cols = shape
            lin_constr_mat = to_vec_shape(lin_constr_mat, n_rows, n_cols,
                                          order='row_major')

            lin_constr_mat, lin_constr_rhs, _, __ = \
                remove_redundant_lin_contrs(lin_constr_mat, lin_constr_rhs)

            lin_constr_mat = to_mat_shape(lin_constr_mat, n_rows, n_cols,
                                          order='row_major')

    return {'lin_coef': lin_coef,
            'log_coef': log_coef,
            'lin_constr_mat': lin_constr_mat,
            'lin_constr_rhs': lin_constr_rhs,
            'epsilon_tilde': epsilon_tilde,
            'guess': guess,
            'eta': eta}


def get_lin_constrs(V, shape,
                    trim_od_constrs=False,
                    # drop_last_V_col=False,
                    non_zero_mask=None,
                    remove_const_cols=False):
    """

    Gets the constraint matrix and RHS for the X matrix in

    V^T diag(deg(A_bp(X))) V = I_K

    lin_constr_mat @ vec(X) = lin_constr_rhs


    n_diag_constr = n_components
    n_off_diag_constr <= n_components choose 2
        If trim_redundant_od_constr = True this might be smaller

    Parameters
    ----------
    V: (n_rows + n_cols, K)

    n_rows, n_cols: int
        Shape of the matrix.

    trim_redundant_od_constr: bool
        Remove redundant off diagonal constraints.

    non_zero_mask: bool
        TODO: describe  and decide if we still want this (see get_pi_sym_problem_data)

    remove_const_cols: bool
        Remove columns of the diagonal constraint matrix that are constant.
        These can cause numerical issues with other constraints.

    Output
    ------
    diag_constr_mat, diag_constr_rhs,
        off_diag_constr_mat, off_diag_constr_rhs

    diag_constr_mat: (n_diag_constr, n_rows, n_cols)

    diag_constr_rhs: (n_diag_constr, )

    off_diag_constr_mat: (n_off_diag_constr, n_rows,  n_cols)

    off_diag_constr_rhs: (n_off_diag_constr, )

    Example
    -------
    diag_constr_mat, diag_constr_rhs,
        off_diag_constr_mat, off_diag_constr_rhs = \
         get_lin_constrs(V, n_rows, n_cols)


    diag_constr_mat = to_vec_shape(diag_constr_mat,
                                   n_rows, n_cols, order='row_major')

    off_diag_constr_mat = to_vec_shape(off_diag_constr_mat,
                                       n_rows, n_cols, order='row_major')

    lin_constr_mat = np.vstack([diag_constr_mat,
                               off_diag_constr_mat])

    lin_const_rhs = np.concatenate([diag_constr_rhs, off_diag_constr_rhs])


    """
    n_rows, n_cols = shape
    n_components = V.shape[1]

    diag_constr_mat_tilde, off_diag_constr_mat_tilde = \
        get_VdV_mat(V, trim_od_constrs=trim_od_constrs)

    # possibly remove constant columns of diagonal constraint matrix
    if remove_const_cols:
        const_mask = np.std(diag_constr_mat_tilde, axis=1) < 1e-5
        diag_constr_mat_tilde = diag_constr_mat_tilde[~const_mask, :]

    # the RHS of the diagonal constraint, a vector of size (n_components, )
    diag_constr_rhs = np.ones(diag_constr_mat_tilde.shape[0])

    # off_diag_constr_mat is (n_components choose 2) x (n_rows + n_cols) matrix
    # corresponding to the off diagonal constraints.
    # hadamard products between all columns of V
    off_diag_constr_mat_tilde = np.array([V[:, i] * V[:, j] for i, j in
                                         combinations(range(n_components), 2)])

    # # Replace off_diag_constr_mat with an orthnormal basis
    # # spanning its row space and possibly remove redundant constraints
    # # by trimming singular vecs whose svals are 0
    # if trim_redundant_od_constr:
    #     sval_cutoff = 1e-10
    #     _, svals, right_svecs = svd_wrapper(off_diag_constr_mat_tilde)
    #     non_zero_sval_mask = svals > sval_cutoff
    #     right_svecs = right_svecs[:, non_zero_sval_mask]
    #     off_diag_constr_mat_tilde = right_svecs.T

    off_diag_constr_rhs = np.zeros(off_diag_constr_mat_tilde.shape[0])

    # get the matrix, S, that maps X to deg(A_bp(X))
    # S is (n_rows + n_cols) x (n_rows * n_cols) matrix
    S = get_row_col_sum_mat(shape)
    # top = np.zeros((n_rows, n_rows * n_cols))
    # for k in range(n_rows):
    #     top[k, k * n_cols:(k + 1) * n_cols] = np.ones(n_cols)
    # bottom = np.zeros((n_cols, n_rows * n_cols))
    # for k in range(n_cols):
    #     bottom[k, :] = np.concatenate([_basis_vec(n_cols, k)] * n_rows)
    # S = np.vstack([top, bottom])
    # S = csr_matrix(S)

    # TODO: do we want this? also document if we do
    if non_zero_mask is not None:
        S = S[:, non_zero_mask]

    # TODO: possibly replace S with a sparse matrix/linear operator
    diag_constr_mat = diag_constr_mat_tilde @ S
    off_diag_constr_mat = off_diag_constr_mat_tilde @ S

    # reshape to devec shape i.e. (n_constr, n_rows, n_cols)
    diag_constr_mat = to_mat_shape(diag_constr_mat, n_rows, n_cols,
                                   order='row_major')

    off_diag_constr_mat = to_mat_shape(off_diag_constr_mat, n_rows, n_cols,
                                       order='row_major')

    return diag_constr_mat, diag_constr_rhs, \
        off_diag_constr_mat, off_diag_constr_rhs
