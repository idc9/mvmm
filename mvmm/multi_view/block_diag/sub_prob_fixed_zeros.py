import cvxpy as cp
import numpy as np


def get_cp_prob_fixed_zeros(Gamma, epsilon, zero_mask=None, eta=None):

    shape = Gamma.shape
    var = cp.Variable(shape=np.product(shape), pos=True)
    assert zero_mask.shape == Gamma.shape

    epsilon_tilde = 1 - epsilon * np.product(shape)

    # setup cvxpy problem
    objective = -Gamma.reshape(-1).T @ cp.log(epsilon + var)

    constraints = [cp.sum(var) == epsilon_tilde]

    if zero_mask is not None and zero_mask.sum() > 0:
        constraints.append(var[zero_mask.reshape(-1)] == 0)

    if eta is not None:
        constraints.append(var >= eta)

    return var, objective, constraints
