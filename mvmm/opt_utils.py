# from cvxopt import spmatrix
from cvxpy import installed_solvers
import scipy.sparse as sps
import numpy as np
from scipy.optimize._remove_redundancy import _remove_redundancy, \
    _remove_redundancy_sparse, _remove_redundancy_dense
from scipy.optimize.optimize import OptimizeWarning
from warnings import warn
import cvxpy as cp


def check_stopping_criteria(abs_diff, rel_diff, abs_tol=None, rel_tol=None):
    """

    """

    if abs_tol is not None and abs_diff is not None and abs_diff <= abs_tol:
        a_stop = True
    else:
        a_stop = False

    if rel_tol is not None and rel_diff is not None and rel_diff <= rel_tol:
        r_stop = True
    else:
        r_stop = False

    if abs_tol is not None and rel_tol is not None:
        # if both critera are use both must be true
        return a_stop and r_stop
    else:
        # otherwise stop if either is True
        return a_stop or r_stop


def remove_redundant_lin_contrs(A_eq, b_eq, verbose=False):
    """
    Removes redundant linear equality constraints
    A_eq x = b_eq

    Code is borrowed from scipy.optimize._linprog_util

    Parameters
    ----------
    A_eq: array-like (n_constr, n_vars)

    b_eq: array-like, (n_constr, )

    verbose: bool

    Output
    ------
    A_eq, b_eq, status, message

    A_eq: array-like (n_constr_non_redundant, n_vars)

    b_eq: array-like, (n_constr_non_redundant, )
    """
    status = None
    message = None

    # remove redundant (linearly dependent) rows from equality constraints
    n_rows_A = A_eq.shape[0]
    redundancy_warning = ("A_eq does not appear to be of full row rank. To "
                          "improve performance, check the problem formulation "
                          "for redundant equality constraints.")

    if (sps.issparse(A_eq)):
        if A_eq.size > 0:  # TODO: Fast sparse rank check?
            A_eq, b_eq, status, message = _remove_redundancy_sparse(A_eq, b_eq)
            if A_eq.shape[0] < n_rows_A and verbose:
                warn(redundancy_warning, OptimizeWarning, stacklevel=1)

        return A_eq, b_eq, status, message

    # This is a wild guess for which redundancy removal algorithm will be
    # faster. More testing would be good.
    small_nullspace = 5
    if A_eq.size > 0:
        try:  # TODO: instead use results of first SVD in _remove_redundancy
            rank = np.linalg.matrix_rank(A_eq)
        except Exception:  # oh well, we'll have to go with _remove_redundancy_dense
            rank = 0

    if A_eq.size > 0 and rank < A_eq.shape[0]:
        if verbose:
            warn(redundancy_warning, OptimizeWarning, stacklevel=3)

        dim_row_nullspace = A_eq.shape[0] - rank

        if dim_row_nullspace <= small_nullspace:
            A_eq, b_eq, status, message = _remove_redundancy(A_eq, b_eq)

        if dim_row_nullspace > small_nullspace or status == 4:
            A_eq, b_eq, status, message = _remove_redundancy_dense(A_eq, b_eq)

        if A_eq.shape[0] < rank:
            message = ("Due to numerical issues, redundant equality "
                       "constraints could not be removed automatically. "
                       "Try providing your constraint matrices as sparse "
                       "matrices to activate sparse presolve, try turning "
                       "off redundancy removal, or try turning off presolve "
                       "altogether.")
            status = 4

    return A_eq, b_eq, status, message


def get_cp_solver(prefs):
    """
    Returns the first available cp solver from a list of preferences.

    Parameters
    ----------
    prefs: list
        Solver preferences

    Output
    ------
    A cp solver

    """
    avail_solvers = set(installed_solvers())

    for solver in prefs:
        if solver in avail_solvers:
            return solver

    raise ValueError('None of the available solvers () are on the list ()'.format(avail_solvers, prefs))


def solve_problem_cp(var, objective, constraints,
                     cp_kws={}, warm_start=False, verbosity=0):

    if cp_kws is None:
        cp_kws = {}

    prob = cp.Problem(cp.Minimize(objective), constraints)
    opt_val = prob.solve(warm_start=warm_start,
                         verbose=verbosity >= 2,
                         **cp_kws)

    if verbosity >= 1:
        print("status:", prob.status)
        print("optimal value", prob.value)

    if var.value is None:
        raise cp.SolverError('cvxpy failed to converge! {}'.
                             format(prob.status))

    return var.value, opt_val, prob


def solve_problem_cp_backups(cp_kws_backups=None,
                             *args, **kwargs):

    if cp_kws_backups is None:
        cp_kws_backups = [None]

    for cp_kws in cp_kws_backups:
        try:
            return solve_problem_cp(cp_kws=cp_kws, *args, **kwargs)

        except cp.SolverError as e:
            print(e)

    raise cp.SolverError('None of the solvers worked')
