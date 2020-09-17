import numpy as np
from time import time
import cvxpy as cp
from copy import deepcopy


def project_affine(x, A, b):
    return x - A.T @ np.linalg.inv(A @ A.T) @ (A @ x - b)


def get_C(A):
    # return A.T @ np.linalg.inv(A @ A.T)
    return A.T @ np.linalg.pinv(A @ A.T, hermitian=True)


def project_affine_cached(x, A, b, C):
    """
    x - A^T (A A^T)^{-1} (Ax - b)

    C := A^T(A A^T)^{-1}

    x - C (Ax - b)

    """

    return x - C @ (A @ x - b)


def setup_aff_cp_prob(A, b):
    n = A.shape[1]

    var = cp.Variable(shape=n)
    point = cp.Parameter(n)

    constraints = [A @ var == b]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(point - var)),
                      constraints)

    return var, point, prob


class AffineProj(object):
    def __init__(self, A, b, method='lin_alg', **kwargs):

        start_time = time()
        self.A = A
        self.b = b
        self.method = method
        self.kwargs = kwargs

        if self.method == 'lin_alg':
            C = get_C(A=A)
            self.cached_data = {'C': C}

        elif self.method == 'QP':
            self.cached_data = setup_aff_cp_prob(A, b)

        else:
            raise ValueError('{} is invalid method'.format(self.method))

        self.metadata = {'setup_time': time() - start_time,
                         'proj_times': []}

    def __call__(self, x):
        start_time = time()

        if self.method == 'lin_alg':
            y = project_affine_cached(x=deepcopy(x),
                                      A=self.A, b=self.b,
                                      C=self.cached_data['C'],
                                      **self.kwargs)

        elif self.method == 'QP':
            var, point, prob = self.cached_data
            point.value = deepcopy(x)
            prob.solve(**self.kwargs)
            y = deepcopy(var.value)

        self.metadata['proj_times'].append(time() - start_time)
        return y
