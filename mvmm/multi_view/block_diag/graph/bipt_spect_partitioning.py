from sklearn.cluster import KMeans
from itertools import product
import numpy as np
from scipy.sparse import diags


from mvmm.multi_view.block_diag.graph.linalg import get_Tsym
from mvmm.linalg_utils import svd_wrapper
from mvmm.utils import safe_invert


def to_comm_mat(y, shape):
    n_rows, n_cols = shape

    row_pred = y[:n_rows]
    col_pred = y[n_rows:]

    comm_mat = np.empty(shape)
    comm_mat[:] = np.nan

    for k in set(y):
        row_idxs = np.where(row_pred == k)[0]
        col_idxs = np.where(col_pred == k)[0]

        # print(k, row_idxs, col_idxs)

        for r, c in product(row_idxs, col_idxs):
            comm_mat[r, c] = k

    return comm_mat


def get_Z(X, K=2):

    ell = int(np.ceil(np.log2(K)))

    Tsym = get_Tsym(X)
    U, svals, V = svd_wrapper(Tsym, rank=ell + 1)

    row_sums = X.sum(axis=1)
    col_sums = X.sum(axis=0)

    row_sums_inv_sqrt = np.sqrt(safe_invert(row_sums))
    col_sums_inv_sqrt = np.sqrt(safe_invert(col_sums))

    Z = np.vstack([diags(row_sums_inv_sqrt) @ U[:, 1:],
                   diags(col_sums_inv_sqrt) @ V[:, 1:]])

    return Z


def run_bipt_spect_partitioning(X, n_blocks, kmeans_kws={}):
    if n_blocks == 1:
        return np.zeros_like(X)

    n_rows, n_cols = X.shape

    Z = get_Z(X=X, K=n_blocks)

    cl = KMeans(n_clusters=n_blocks, **kmeans_kws)
    cl.fit(Z)
    y_pred = cl.predict(Z)

    comm_mat = to_comm_mat(y=y_pred, shape=X.shape)

    return comm_mat
