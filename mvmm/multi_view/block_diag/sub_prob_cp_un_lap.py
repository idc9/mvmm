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
