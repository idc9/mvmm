import numpy as np

from mvmm.multi_view.block_diag.graph.linalg import get_sym_laplacian_bp, \
    get_unnorm_laplacian_bp, get_deg_bp
from mvmm.linalg_utils import eigh_wrapper
from mvmm.multi_view.block_diag.graph.linalg import \
    smallest_eigh_Lsym_bp_from_Tsym_no_zeros


def angle(u, v, subspace=False):
    """
    Computes the angle (in degrees) between two vectors (or between the subspaces spanned by two vectors).

    Parameters
    ----------
    u, v: array-like
        The vectors.

    subspace: bool
        Compute the angle between the subspaces spanned by u and v

    """
    u = np.array(u).reshape(-1)
    v = np.array(v).reshape(-1)

    c = u.T @ v / max(np.linalg.norm(u) * np.linalg.norm(u), 1e-12)
    c = np.clip(c, a_min=-1, a_max=1)

    a = np.rad2deg(np.arccos(c))

    if subspace:
        a = min(a, 180 - a)

    return a


def check_vs_truth_smallest_eigh_Lsym_bp_from_Tsym_no_zeros(X, rank):
    """
    Check against ground truth
    """
    evals, evecs = smallest_eigh_Lsym_bp_from_Tsym_no_zeros(X, rank=rank)

    if rank is None:
        rank = min(X.shape)

    Lsym = get_sym_laplacian_bp(X)

    evals_true, evecs_true = eigh_wrapper(A=Lsym)
    evals_true = evals_true[-rank:]
    evecs_true = evecs_true[:, -rank:]

    # check gevals match true gecals
    assert np.allclose(evals, evals_true)

    # check evecs span the correct space
    for k in range(rank):
        # ignore 1 evals since the evecs are non-unique
        if not np.allclose(evals[k], 1):
            assert angle(evecs[:, k], evecs_true[:, k], subspace=True) < 1e-4


def check_vs_internal_smallest_eigh_Lsym_bp_from_Tsym_no_zeros(X, rank):
    """
    Checks internal consistency
    """

    evals, evecs = smallest_eigh_Lsym_bp_from_Tsym_no_zeros(X, rank=rank)

    if rank is None:
        rank = min(X.shape)

    Lsym = get_sym_laplacian_bp(X)

    assert len(evals) == rank
    assert evecs.shape[0] == sum(X.shape)
    assert evecs.shape[1] == rank

    for k in range(rank):
        v = evecs[:, k]
        q = Lsym @ v / v

        # check that v is an eigenvector
        # note 0 entries give infs so we ignore these for checking
        idx = np.where(abs(v) > 1e-10)[0][0] # for non nan
        for i in range(len(q)):
            assert np.allclose(v[i], 0) or np.allclose(q[i], q[idx])

        # make sure the empirical eval is equal to the returned evals
        assert np.allclose(q[idx], evals[k])

    assert np.allclose(evecs.T @ evecs, np.eye(rank))


def check_geigh_Lsym_internal_no_zeros(X, gevals, gevecs, rank):
    Lun = get_unnorm_laplacian_bp(X)
    degs = get_deg_bp(X)

    if rank is None:
        rank = min(X.shape)

    # check shapes
    assert len(gevals) == rank
    assert gevecs.shape[0] == sum(X.shape)
    assert gevecs.shape[1] == rank

    # check gevec normalization
    assert np.allclose(gevecs.T @ np.diag(degs) @ gevecs,
                       np.eye(gevecs.shape[1]))

    # check gevecs equation
    for k in range(rank):
        v = gevecs[:, k]
        q = Lun @ v / (np.diag(degs) @ v)

        # check that v is an eigenvector
        # note 0 entries give infs so we ignore these for checking
        idx = np.where(abs(v) > 1e-10)[0][0]  # for non nan
        for i in range(len(q)):
            assert np.allclose(v[i], 0) or np.allclose(q[i], q[idx])

        # make sure the empirical geval is equal to the returned evals
        assert np.allclose(q[idx], gevals[k])


def true_gevals_Lsym(X, zero_tol=1e-10):
    Lsym = get_sym_laplacian_bp(X)
    true_evals, true_evecs = eigh_wrapper(Lsym, rank=None)

    zero_row_mask = np.linalg.norm(X, axis=1) < zero_tol
    zero_col_mask = np.linalg.norm(X, axis=0) < zero_tol
    n_iso_verts = sum(zero_row_mask) + sum(zero_col_mask)
    meow = max(X.shape) - n_iso_verts
    true_gevals = np.concatenate([true_evals[0:meow],
                                  [1] * (max(X.shape) - min(X.shape)),
                                  true_evals[-meow:]])
    true_gevals = np.sort(true_gevals)[::-1]

    true_zero_mask = np.concatenate([zero_row_mask, zero_col_mask])
    return true_gevals, true_zero_mask
