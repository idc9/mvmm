"""
Bipartite graph community utility functions.
"""
import networkx as nx
import numpy as np
from itertools import product
from copy import deepcopy

from mvmm.multi_view.block_diag.graph.linalg import get_adjmat_bp


def get_nonzero_block_mask(X, tol=1e-6):
    """
    Returns the True/False mask identifying the blocks of
    a block diagonal matrix (connected components of the bipratite graph).
    Some zero entries may be un-zeroed if its row or column contains
    at least one non-zero entry.

    Parameters
    ----------
    X: array-like, (n_rows, n_cols)
        The data matrix.

    tol: float
        The non-zero mask is abs(X) > tol

    Output
    ------
    non_zero_block_mask, n_blocks

    non_zero_block_mask: array-like of bools (n_rows, n_cols)
        The mask identifying non-zero blocks.

    C: array-like, (n_rows, n_cols)
        Array identifying the blocks.

    n_blocks: int
        The number of blocks.
    """

    entrywise_mask = abs(X) > tol

    C = get_comm_mat(entrywise_mask)

    if np.mean(np.isnan(C)) == 1:
        n_blocks = 1
        n_zero_rows = 0
        n_zero_cols = 0
        non_zero_block_mask = np.ones_like(C).astype(bool)

        return non_zero_block_mask, C, n_blocks, n_zero_cols, n_zero_rows

    n_blocks = int(np.nanmax(C)) + 1

    non_zero_block_mask = np.zeros_like(X).astype(bool)
    for com_idx in range(n_blocks):
        row_idxs = np.where((C == com_idx).sum(axis=1) >= 1)[0]
        col_idxs = np.where((C == com_idx).sum(axis=0) >= 1)[0]

        for (r, c) in product(row_idxs, col_idxs):
            non_zero_block_mask[r, col_idxs] = True
            non_zero_block_mask[row_idxs, c] = True

    n_zero_cols = sum(non_zero_block_mask.sum(axis=0) == 0)
    n_zero_rows = sum(non_zero_block_mask.sum(axis=1) == 0)

    return non_zero_block_mask, C, n_blocks, n_zero_cols, n_zero_rows


def get_comm_mat(A):
    """

    Parameters
    ----------
    A: (n_clusters_0, n_clusters_1)

    Output
    ------
    C: (n_clusters_0, n_clusters_1)
    """
    node_labels = ['row_{}'.format(i) for i in range(A.shape[0])] + \
        ['col_{}'.format(i) for i in range(A.shape[1])]

    node_labels = {i: node_labels[i] for i in range(len(node_labels))}

    # convert to nx graph
    G = nx.from_numpy_array(get_adjmat_bp(A))
    G = nx.relabel_nodes(G, node_labels)

    # get row/col indices of connected components
    comm_mat = np.empty(A.shape)
    comm_mat[:] = np.nan

    block_idx = -1
    for k, cc in enumerate(nx.connected_components(G)):
        cc_graph = G.subgraph(cc)

        if cc_graph.number_of_nodes() >= 2:
            block_idx += 1

            for e in cc_graph.edges():

                v0, idx0 = e[0].split('_')
                v1, idx1 = e[1].split('_')

                assert (v0 == 'row' and v1 == 'col') or \
                    (v0 == 'col' and v1 == 'row')

                if v0 == 'row':
                    comm_mat[int(idx0), int(idx1)] = block_idx
                else:
                    comm_mat[int(idx1), int(idx0)] = block_idx
    return comm_mat


def community_summary(Pi, zero_thresh=0):
    """
    Summarizes the community structure of a Pi matrix.

    Parameters
    ----------
    Pi: array-like

    zero_thresh: float
        Values below this are set to 0.

    Output
    ------
    summary, Pi_comm

    """
    non_zero_mask = Pi > zero_thresh
    C = get_comm_mat(non_zero_mask)

    n_communities = int(np.nanmax(C)) + 1

    n_nonzero_entries = (non_zero_mask).sum()
    n_connected_rows = sum((~np.isnan(C)).sum(axis=1) > 0)
    n_connected_cols = sum((~np.isnan(C)).sum(axis=0) > 0)
    tot_weight = Pi[non_zero_mask].sum()

    comm_n_edges = []
    comm_shapes = []
    comm_weight = []
    for idx in range(n_communities):
        comm_mask = (C == idx)

        n_edges = comm_mask.sum()
        n_rows = sum(comm_mask.sum(axis=1) > 0)
        n_cols = sum(comm_mask.sum(axis=0) > 0)
        weight = Pi[comm_mask].sum()

        comm_n_edges.append(n_edges)
        comm_shapes.append((n_rows, n_cols))
        comm_weight.append(weight)

    col_memberships = np.nanmax(C, axis=0)
    row_memberships = np.nanmax(C, axis=1)
    Pi_comm = deepcopy(Pi)
    Pi_comm[~non_zero_mask] = 0
    Pi_comm = Pi_comm[np.argsort(row_memberships), :]
    Pi_comm = Pi_comm[:, np.argsort(col_memberships)]

    summary = {'n_communities': n_communities,
               'comm_n_edges': comm_n_edges,
               'comm_shapes': comm_shapes,
               'comm_weight': comm_weight,

               'n_nonzero_entries': n_nonzero_entries,
               'tot_weight': tot_weight,
               'n_connected_rows': n_connected_rows,
               'n_connected_cols': n_connected_cols,

               'col_memberships': col_memberships,
               'row_memberships': row_memberships}

    return summary, Pi_comm
