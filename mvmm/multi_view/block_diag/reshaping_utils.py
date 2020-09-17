

def to_mat_shape(A, n_rows, n_cols, order='row_major'):
    """

    Parameters
    ----------
    A: array-like, (n_constr, n_rows * n_cols)

    n_rows, n_cols: int
        Shape of the matrix.

    order: str, ['row_major', 'col_major']
        How the cols of A have been vectorized.

    Output
    ------
    array-like, (n_constr, n_rows, n_cols)
    """

    assert A.shape[1] == n_rows * n_cols
    n_constr = A.shape[0]
    return A.reshape(n_constr, n_rows, n_cols, order=translate_order(order))


def to_vec_shape(A, n_rows, n_cols, order='row_major'):
    """
    Vectorizes the

    Parameters
    ----------
    A: array-like, (n_constr, n_rows, n_cols)

    n_rows, n_cols: int
        Shape of the matrix.

    order: str, ['row_major', 'col_major']
        How to vectorize the matrix.

    Output
    ------
    array-like, (n_constr, n_rows * n_cols)

    """
    assert A.ndim == 3
    assert A.shape[1] == n_rows
    assert A.shape[2] == n_cols

    n_constr = A.shape[0]
    return A.reshape(n_constr, n_rows * n_cols, order=translate_order(order))


def translate_order(order):
    """
    Translates 'row_major' or 'col_major' to 'C' or 'F' respectively that numpy.reshape wants as an argument

    Parameters
    ----------
    order: str
        Must be one of ["row_major", "col_majr"]

    Output
    ------
    'C' (row major) or 'F' (col major)
    """

    if order == 'row_major':
        np_ord = 'C'
    elif order == 'col_major':
        np_ord = 'F'
    else:
        raise ValueError('order must be one of ["row_major", "col_majr"], '
                         'not {}'.format(order))

    return np_ord
