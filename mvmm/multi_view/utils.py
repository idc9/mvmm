import numpy as np


def vec(x):
    """
    Vectorizes a matrix (rows then columns).

    Parameters
    ----------
    x: array-like, (n_rows, n_cols)

    Output
    ------
    x: np.array, (n_rows * n_cols, )
    """
    return np.array(x).reshape(-1)


def devec(x, n_rows, n_cols):
    """
    Devectorizes a vector into a matrix (inverse of vec)

    Parameters
    ----------
    x: array-like, (n_rows * n_cols, )

    n_rows: int
        Number of rows in the matrix version of x.

    n_cols: int
        Number of columns in the matrix version of x.

    Output
    ------
    x: array-like, (n_rows, n_cols)
    """
    return np.array(x).reshape((n_rows, n_cols))


# def vec2devec_idx(idx, n_rows, n_cols):
#     idxs_vec = np.arange(n_rows * n_cols)
#     idxs_devec = devec(idxs_vec, n_rows, n_cols)
#     row_idx, col_idx = np.where(idxs_devec == idx)

#     return row_idx.item(), col_idx.item()

# def devec2vec_idx(row_idx, col_idx, n_rows, n_cols):
#     idxs_vec = np.arange(n_rows * n_cols)
#     idxs_devec = devec(idxs_vec, n_rows, n_cols)
#     return idxs_devec[row_idx, col_idx]

def vec2devec_idx(idx, n_rows, n_cols):
    row_idx = idx // n_cols
    col_idx = idx % n_cols
    return row_idx, col_idx


def devec2vec_idx(row_idx, col_idx, n_rows, n_cols):
    return row_idx * n_cols + col_idx


def marginalize_to_view(a, view):
    """
    Marginalizes all views except one.
    Parameters
    ----------
    a: (n_feat_1, ..., n_feat_V)

    view: int

    Output
    ------
    s: n_view_view

    """
    axes = range(a.ndim)
    axes = np.delete(axes, view)
    return a.sum(tuple(axes))


def view_labs_to_overall(Y):
    """
    Converts view-wise labels to overall labeling index:

    Parameters
    ----------
    Y: array-like, (n_samples, n_views)

    Output
    ------
    y_overall: array-like, (n_samples, )
    """
    Y = np.array(Y)
    v2o = {tuple(pair): idx for idx, pair in enumerate(np.unique(Y, axis=0))}

    y_overall = []
    for r in range(Y.shape[0]):
        y_overall.append(v2o[tuple(Y[r, :])])

    return y_overall


def get_n_comp(Pi):
    """
    Given a Pi matrix returns the total number of components as well as the number of view specific components.

    Parameters
    ----------
    Pi: array-like, (n_comp_0, n_comp_1, ..., n_comp_V )

    Output
    ------
    n_comp_tot, n_comp_views

    n_comp_tot: int
        Total number of components (number of non-zero entries of Pi)

    n_view_components: list of len = V
        Number of components in each view.
    """
    Pi = np.array(Pi)
    n_views = Pi.ndim
    A = Pi > 0
    n_comp_tot = A.sum()
    n_view_components = [sum(marginalize_to_view(A, v) > 0)
                         for v in range(n_views)]
    return n_comp_tot, n_view_components


def unit_intval_linspace(num=50, include_zero=True):
    """
    Linearly spaced points on the unit interval either [0, 1) or (0, 1)

    Parameters
    ----------
    num: int
        Number of points

    include_zero: bool
        Whether or not to include 0.
    """
    if include_zero:
        return np.linspace(start=0, stop=1, num=num, endpoint=False)
    else:
        return np.linspace(start=0, stop=1, num=num + 1, endpoint=False)[1:]


def unit_intval_polyspace(num=50, deg=2, include_zero=True):
    """
    Polynomially spaced points on the unit interval.

    Parameters
    ----------
    num: int
        Number of points

    include_zero: bool
        Whether or not to include 0.
    """
    return unit_intval_linspace(num=num, include_zero=include_zero) ** deg


def unit_intval_logspace(num=50, stop=-2, include_zero=True):
    """
    Log-spaced points on the unit interval.

    Parameters
    ----------
    num: int
        Number of points

    include_zero: bool
        Whether or not to include 0.
    """
    vals = np.logspace(start=stop, stop=0, num=num + 1,
                       endpoint=False)[1:]

    if include_zero:
        vals = np.concatenate([[0.0], vals[:-1]])

    return vals


def linspace_zero_to(stop=1, num=50):
    """
    Lin space from 0 to stop. Does not include zero.
    """
    return np.linspace(start=stop, stop=0,
                       num=num,
                       endpoint=False)[::-1]


# def quadspace_zero_to(stop=1, num=50):
#     polyspace_zero_to(stop=stop, num=num, exp=2)


def polyspace_zero_to(stop=1, num=50, deg=2):
    vals = linspace_zero_to(stop=stop, num=num)
    vals = vals ** deg
    vals = stop * vals / max(vals)
    return vals


def expspace_zero_to(stop=1, num=50, base=10):
    vals = linspace_zero_to(stop=stop, num=num)
    vals = base ** vals
    vals = stop * vals / max(vals)
    return vals
