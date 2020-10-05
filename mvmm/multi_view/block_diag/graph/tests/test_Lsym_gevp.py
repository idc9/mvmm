import numpy as np
from copy import deepcopy

from mvmm.linalg_utils import eigh_wrapper
from mvmm.multi_view.block_diag.graph.linalg import get_sym_laplacian_bp, \
    geigh_Lsym_bp, smallest_geigh_Lsym_bp_from_Tsym

from mvmm.multi_view.block_diag.graph.tests.utils_test_Lsym_gevp import \
    check_vs_truth_smallest_eigh_Lsym_bp_from_Tsym_no_zeros, \
    check_vs_internal_smallest_eigh_Lsym_bp_from_Tsym_no_zeros, \
    check_geigh_Lsym_internal_no_zeros, \
    true_gevals_Lsym


def test_smallest_eigh_Lsym_bp_from_Tsym_no_zeros():

    Xs = [np.random.uniform(size=(4, 5)),
          np.random.uniform(size=(5, 5)),
          np.random.uniform(size=(5, 1))]

    for X in Xs:

        for rank in range(1, sum(X.shape) + 1):

            check_vs_truth_smallest_eigh_Lsym_bp_from_Tsym_no_zeros(X, rank)
            check_vs_internal_smallest_eigh_Lsym_bp_from_Tsym_no_zeros(X, rank)

        check_vs_truth_smallest_eigh_Lsym_bp_from_Tsym_no_zeros(X, rank=None)
        check_vs_internal_smallest_eigh_Lsym_bp_from_Tsym_no_zeros(X, rank=None)


def test_geigh_Lsym_bp():

    Xs = [np.random.uniform(size=(4, 5)),
          np.random.uniform(size=(5, 5)),
          np.random.uniform(size=(5, 1))]

    for X in Xs:

        Lsym = get_sym_laplacian_bp(X)
        true_evals, true_evecs = eigh_wrapper(Lsym, rank=None)

        for rank in range(1, sum(X.shape) + 1):

            gevals, gevecs = geigh_Lsym_bp(X, rank=rank, zero_tol=1e-10,
                                           end='largest')
            check_geigh_Lsym_internal_no_zeros(X, gevals, gevecs, rank)
            assert np.allclose(gevals[:rank], true_evals[:rank])

            gevals, gevecs = geigh_Lsym_bp(X, rank=rank, zero_tol=1e-10,
                                           end='smallest')
            check_geigh_Lsym_internal_no_zeros(X, gevals, gevecs, rank)
            assert np.allclose(gevals[:rank], true_evals[-rank:])

        rank = None
        gevals, gevecs = geigh_Lsym_bp(X, rank=rank, zero_tol=1e-10,
                                       end='largest')
        check_geigh_Lsym_internal_no_zeros(X, gevals, gevecs, rank)

        gevals, gevecs = geigh_Lsym_bp(X, rank=rank, zero_tol=1e-10,
                                       end='smallest')
        check_geigh_Lsym_internal_no_zeros(X, gevals, gevecs, rank)

    # test with zero rows/cols
    X = deepcopy(Xs)[0]
    X[0, :] = 0
    X[:, 0] = 0

    true_gevals, true_zero_mask = true_gevals_Lsym(X)
    for rank in range(1, 7 + 1):

        # make sure gen evals are correct
        gevals, gevecs = geigh_Lsym_bp(X, rank=rank, zero_tol=1e-10, end='largest')
        # check_geigh_Lsym_internal(X, gevals, gevecs, rank=rank)
        assert np.allclose(gevals, true_gevals[:rank])

        # check gen evecs have correct zero rows
        assert np.allclose(abs(gevecs[true_zero_mask]).sum(), 0)


def test_smallest_geigh_Lsym_bp_from_Tsym():

    Xs = [np.random.uniform(size=(4, 5)),
          np.random.uniform(size=(5, 5)),
          np.random.uniform(size=(5, 1))]

    for X in Xs:

        Lsym = get_sym_laplacian_bp(X)
        true_evals, true_evecs = eigh_wrapper(Lsym, rank=None)

        for rank in range(1, sum(X.shape) + 1):

            gevals, gevecs = smallest_geigh_Lsym_bp_from_Tsym(X, rank=rank,
                                                              zero_tol=1e-10)
            check_geigh_Lsym_internal_no_zeros(X, gevals, gevecs, rank)
            assert np.allclose(gevals[:rank], true_evals[-rank:])

        rank = None
        gevals, gevecs = smallest_geigh_Lsym_bp_from_Tsym(X, rank=rank,
                                                          zero_tol=1e-10)
        check_geigh_Lsym_internal_no_zeros(X, gevals, gevecs, rank)

    # test with zeros
    X = deepcopy(Xs)[0]
    X[0, :] = 0
    X[:, 0] = 0

    true_gevals, true_zero_mask = true_gevals_Lsym(X)
    for rank in range(1, 7 + 1):

        # make sure gen evals are correct
        gevals, gevecs = smallest_geigh_Lsym_bp_from_Tsym(X, rank=rank,
                                                          zero_tol=1e-10)
        # check_geigh_Lsym_internal(X, gevals, gevecs, rank=rank)
        assert np.allclose(gevals, true_gevals[-rank:])

        # check gen evecs have correct zero rows
        assert np.allclose(abs(gevecs[true_zero_mask]).sum(), 0)
