import numpy as np
import cvxpy as cp
from copy import deepcopy

from mvmm.multi_view.block_diag.utils import \
    get_guess, get_lin_coef, get_row_col_sum_mat


def get_cp_problem_un_lap(Gamma,
                          eig_var,
                          epsilon,
                          B,
                          alpha,
                          eta=None,
                          weights=None,
                          init_val=None,
                          obj_mult=1):
    """
    Sets up the bd_weights_ update for the unnormalized Laplacian using cvxpy.

    min_D - sum_{k1, k2} Gamma_{k1, k2} log(epsilon + D_{k1, k2}) +
        alpha * <D, M(eig_var, weights) >

    s.t. sum_{k1, k2} D_{k1, k1} = 1 - np.product(D.shape) * epsilon

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

    obj_mult: float
        Multiply the objective function by a constant. This does not change the problem, but can help some solvers find a solution.

    """

    shape = Gamma.shape
    var = cp.Variable(shape=np.product(shape), pos=True)

    epsilon_tilde = 1 - epsilon * np.product(shape)
    log_coef = deepcopy(Gamma).reshape(-1)
    lin_coef = alpha * get_lin_coef(eig_var, shape,
                                    weights=weights).reshape(-1)

    # set initial value
    if type(init_val) == str and init_val == 'guess':
        guess = get_guess(log_coef, lin_coef, epsilon, epsilon_tilde)
        var.value = guess.reshape(-1)
    elif init_val is not None:
        var.value = init_val.reshape(-1)

    # setup cvxpy problem
    objective = -log_coef.T @ cp.log(epsilon + var) + lin_coef.T @ var

    if obj_mult is not None:
        objective = obj_mult * objective

    constraints = [cp.sum(var) == epsilon_tilde]

    if eta is not None:
        S = get_row_col_sum_mat(shape)
        S_rhs = eta * np.ones(sum(shape))
        constraints.append(S @ var >= S_rhs)

    return var, objective, constraints
