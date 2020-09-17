import numpy as np
from itertools import product
from numbers import Number
from sklearn.utils.fixes import logsumexp

from mvmm.multi_view.MVMM import MVMM


class MaskedMVMM(MVMM):
    """
    A multi-view mixture model where some entries of Pi are set to zero.
    """

    def _pre_fit_setup(self):
        """
        Subclasses may call this before running _fit()
        """
        self.n_zeroed_comps_ = 0

    @property
    def n_components(self):
        """
        Returns the total number of clusters.
        """
        if hasattr(self, 'zero_mask_') and self.zero_mask_ is not None:
            return (~self.zero_mask_).ravel().sum()

        else:
            return np.product(self.n_view_components)

    @property
    def weights_mat_(self):
        """
        Returns weights as a matrix.
        """

        if hasattr(self, 'weights_') and self.weights_ is not None:

            weights_mat = np.zeros(self.n_view_components)
            for k in range(self.n_components):
                # idx_0, idx_1 = self._get_view_clust_idx(k)
                # weights_mat[idx_0, idx_1] = self.weights_[k]
                view_idxs = self._get_view_clust_idx(k)
                weights_mat[view_idxs] = self.weights_[k]

            return weights_mat

    def _get_view_clust_idx(self, k):
        """Retuns the view cluster indices for each view for an overall cluster index.

        Returns
        -------
        view_idxs : array, shape (n_views, )
        """

        if isinstance(k, Number):
            return self._get_view_clust_idx_for_masked(int(k))

        else:
            view_idxs = [[] for v in range(self.n_views)]
            for idx in k:
                _view_idxs = self._get_view_clust_idx_for_masked(int(idx))
                for v in range(self.n_views):
                    view_idxs[v].append(_view_idxs[v])
            return tuple(np.array(view_idxs[v]) for v in range(self.n_views))

            # row_idxs, col_idxs = [], []
            # for idx in k:
            #     r, c = self._get_view_clust_idx_for_masked(int(idx))
            #     row_idxs.append(r)
            #     col_idxs.append(c)

            # return np.array(row_idxs), np.array(col_idxs)

    def _get_view_clust_idx_for_masked(self, k):

        assert k < self.n_components

        idx = -1
        ranges = tuple(range(self.n_view_components[v])
                       for v in range(self.n_views))

        for view_idxs in product(*ranges):
            # don't count components which are zeroed out
            if not self.zero_mask_[view_idxs]:
                idx += 1

            if k == idx:
                return view_idxs

        # idx = -1
        # for idx_0, idx_1 in product(range(self.n_view_components[0]),
        #                             range(self.n_view_components[1])):
        #     # don't count components which are zeroed out
        #     if not self.zero_mask_[idx_0, idx_1]:
        #         idx += 1

        #     if k == idx:
        #         return idx_0, idx_1

        raise ValueError('No components found.')

    def _get_overall_clust_idx(self, view_idxs):

        assert not self.zero_mask_[view_idxs]

        for k in range(self.n_components):
            _view_idxs = self._get_view_clust_idx(k)

            if all(_view_idxs[v] == view_idxs[v]
                   for v in range(self.n_views)):
                return k

    # def _get_overall_clust_idx(self, idx_0, idx_1):
    #     if self.n_views > 2:
    #         raise NotImplementedError

    #     assert not self.zero_mask_[idx_0, idx_1]

    #     for k in range(self.n_components):
    #         _idx_0, _idx_1 = self._get_view_clust_idx(k)
    #         if idx_0 == _idx_0 and idx_1 == _idx_1:
    #             return k

    def _post_initialization(self, init_params, X, random_state):
        zero_mask = np.zeros(shape=self.n_view_components).astype(bool)
        init_params['zero_mask'] = zero_mask
        self.n_zeroed_comps_ = 0
        return init_params

    def _get_parameters(self):
        view_params = []
        for v in range(self.n_views):
            view_params.append(self.view_models_[v]._get_parameters())

        return {'views': view_params,
                'weights': self.weights_,
                'zero_mask': self.zero_mask_}

    def _set_parameters(self, params):
        if 'views' in params.keys():
            for v in range(self.n_views):
                self.view_models_[v]._set_parameters(params['views'][v])

        # TODO: move this after weights
        if 'zero_mask' in params.keys():
            self.zero_mask_ = params['zero_mask']

        if 'weights' in params.keys():
            self.weights_ = params['weights']

            # make sure each view's weights_ is the marginal of the weights
            for v in range(self.n_views):
                ax_to_sum = tuple([a for a in range(self.n_views) if a != v])
                view_weights = np.sum(self.weights_mat_, axis=ax_to_sum)
                self.view_models_[v]._set_parameters({'weights': view_weights})

        # drop components with 0 weight
        idxs2drop = np.where(self.weights_ == 0.0)[0]
        if len(idxs2drop) > 0:
            self.drop_component(idxs2drop)

        # TODO: something crazy is happening -- the code won't reach here
        # if 'zero_mask' in params.keys():
        #     print('here')
        #     self.zero_mask_ = params['zero_mask']

    def _m_step_clust_params(self, X, log_resp):
        """
        M step. Each view's cluster parameters can be updated independently.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        # TODO: document this as it is a critical step

        # for each view-cluster pair, which columns of log_resp to logsumsxp
        vc_axes2sum = [[[] for c in range(self.view_models_[v].n_components)]
                       for v in range(self.n_views)]

        for k in range(self.n_components):
            view_idxs = self._get_view_clust_idx(k)
            for v in range(self.n_views):
                vc_axes2sum[v][view_idxs[v]].append(k)

            # idx_0, idx_1 = self._get_view_clust_idx(k)
            # vc_axes2sum[0][idx_0].append(k)
            # vc_axes2sum[1][idx_1].append(k)

        view_params = [None for v in range(self.n_views)]

        for v in range(self.n_views):
            view_log_resp = []
            # for each view-component logsumexp the responsibilities
            for c in range(self.view_models_[v].n_components):
                axes2sum = vc_axes2sum[v][c]
                view_log_resp.append(logsumexp(log_resp[:, axes2sum], axis=1))
            view_log_resp = np.array(view_log_resp).T

            view_params[v] = self.view_models_[v].\
                _m_step_clust_params(X=X[v], log_resp=view_log_resp)

        return view_params

    def drop_component(self, comps):
        """
        Drops a component or components from the model.

        Parameters
        ----------
        comps: int, list of ints
            Which component(s) to drop. On the scal of overall indices.
        """

        # TODO: re-write using _set_parameters
        if isinstance(comps, Number):
            comps = [comps]

        self.n_zeroed_comps_ += len(comps)

        # sort componets in decreasing order so that lower indicies
        # are preserved after dropping higher indices
        comps = np.sort(comps)[::-1]

        overall_comps2drop = []
        view_comps2drop = [[] for v in range(self.n_views)]
        # view_0_comps2drop = []
        # view_1_comps2drop = []
        for k in comps:
            # idx_0, idx_1 = self._get_view_clust_idx(k)
            view_idxs = self._get_view_clust_idx(k)

            # don't drop components which are already zero
            if not self.zero_mask_[view_idxs]:
                self.zero_mask_[view_idxs] = True

                overall_comps2drop.append(k)

                for v in range(self.n_views):
                    # TODO: check this
                    meow = self.zero_mask_.take(indices=view_idxs[v], axis=v)
                    if np.mean(meow) == 1:
                        view_comps2drop[v].append(view_idxs[v])

                # # if row idx_0 is entirely zero, drop component
                # # idx_0 from the view 0 model
                # if np.mean(self.zero_mask_[idx_0, :]) == 1:
                #     view_0_comps2drop.append(idx_0)

                # # similarly drop zero columns
                # if np.mean(self.zero_mask_[:, idx_1]) == 1:
                #     view_1_comps2drop.append(idx_1)

        # drop entries from weights_ and re-normalize
        self.weights_ = np.delete(self.weights_, overall_comps2drop)
        self.weights_ = self.weights_ / sum(self.weights_)

        for v in range(self.n_views):
            self.view_models_[v].drop_component(view_comps2drop[v])
            self.zero_mask_ = np.delete(self.zero_mask_,
                                        view_comps2drop[v],
                                        axis=v)

    def reorder_components(self, new_idxs):
        """
        Re-orders the components

        Parameters
        ----------
        new_idxs_0: list
            List of the new index ordering for view 0
            i.e. new_idxs_0[0] maps old index 0 to its new index.

        new_idxs_0: list
            List of the new index ordering for view 1
            i.e. new_idxs_0[0] maps old index 0 to its new index.

        """

        for v in range(self.n_views):
            # TODO: re-write using _set_parameters()
            assert set(new_idxs[v]) == set(range(self.n_view_components[v]))

        # for new overall index ordering
        new_entry_ordering = []
        old2new = [{old: new for new, old in enumerate(new_idxs[v])}
                   for v in range(self.n_views)]

        for k in range(self.n_components):
            old_view_idxs = self._get_view_clust_idx(k)

            new_view_idxs = [old2new[v][old_view_idxs[v]]
                             for v in range(self.n_views)]

            new_entry_ordering.append(new_view_idxs)

        # reorder view cluster parameters
        for v in range(self.n_views):
            self.view_models_[v].reorder_components(new_idxs[v])

        # re-order zero mask
        for v in range(self.n_views):
            self.zero_mask_ = self.zero_mask_.take(indices=new_idxs[v],
                                                   axis=v)

        # get new overall ordering and re-order weights_
        new_idxs_overall = []
        for new_idxs in new_entry_ordering:
            k_new = self._get_overall_clust_idx(new_idxs)
            new_idxs_overall.append(k_new)

        self.weights_ = self.weights_[new_idxs_overall]

        return self

    def _n_weight_parameters(self):
        tot = np.product(self.n_view_components)
        n_zeros = self.zero_mask_.sum()

        return tot - n_zeros - 1
