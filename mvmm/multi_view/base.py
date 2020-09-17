from sklearn.utils import check_random_state
from sklearn.utils.fixes import logsumexp
from sklearn.utils.validation import check_is_fitted
from copy import deepcopy
from time import time
import numpy as np
from itertools import product

from mvmm.base import MixtureModelMixin, EMfitMMMixin, _check_X
from mvmm.multi_view.initialization import combined_weight_from_sep_view_models


class MultiViewMixtureModelMixin(MixtureModelMixin):

    def _get_parameters(self):
        view_params = []
        for v in range(self.n_views):
            view_params.append(self.view_models_[v]._get_parameters())

        return {'views': view_params,
                'weights': self.weights_}

    def _set_parameters(self, params):

        if 'views' in params.keys():
            for v in range(self.n_views):
                self.view_models_[v]._set_parameters(params['views'][v])

        if 'weights' in params.keys():
            self.weights_ = params['weights']

            # make sure each view's weights_ is the marginal of the weights
            for v in range(self.n_views):
                ax_to_sum = tuple([a for a in range(self.n_views) if a != v])
                view_weights = np.sum(self.weights_mat_, axis=ax_to_sum)
                self.view_models_[v].weights_ = view_weights

    def fit(self, X):
        """
        Fits a multi-view mixture model to the observed multi-view data.

        Parameters
        ----------
        X: list of array-like
            List of data for each view.

        """
        assert len(X) == self.n_views

        n_samples = X[0].shape[0]

        self.metadata_ = {'n_samples': X[0].shape[0],
                          'n_features': [X[v].shape[1]
                                         for v in range(self.n_views)]}

        # initalize fitted view models
        self.view_models_ = [deepcopy(self.base_view_models[v])
                             for v in range(self.n_views)]

        for v in range(self.n_views):
            assert X[v].shape[0] == n_samples

            X[v] = _check_X(X[v], self.n_view_components[v],
                            ensure_min_samples=2)

            self.view_models_[v]._check_parameters(X[v])

        self._check_parameters(X)
        self._check_fitting_parameters(X)

        start_time = time()
        self._fit(X)
        self.metadata_['fit_time'] = time() - start_time
        return self

    def _check_clust_param_values(self, X):
        """
        Checks cluster parameters.
        """
        for v in range(self.n_views):
            self.view_models_[v]._check_clust_param_values(X[v])

    def _check_clust_parameters(self, X):
        """
        Checks cluster parameters and weights.
        """
        for v in range(self.n_views):
            self.view_models_[v]._check_clust_parameters(X[v])

    @property
    def n_views(self):
        return len(self.base_view_models)

    @property
    def n_view_components(self):
        """
        Number of components in each views.
        """
        if hasattr(self, 'view_models_'):
            return [vm.n_components for vm in self.view_models_]
        else:
            return [vm.n_components for vm in self.base_view_models]

    @property
    def n_components(self):
        """
        Returns the total number of clusters.
        """
        # return np.product([self.view_models_[v].n_components
        #                   for v in range(self.n_views)])
        return np.product(self.n_view_components)

    @property
    def weights_mat_(self):
        """
        Returns weights as a matrix.
        """
        # TODO: does this mess up check_is_fitted()
        if hasattr(self, 'weights_') and self.weights_ is not None:
            return self.weights_.reshape(*self.n_view_components)

    def comp_log_probs(self, X):

        n_samples = X[0].shape[0]
        comp_lpr = np.zeros((n_samples, self.n_components))

        # log liks for each view's clusters
        # [f(x(v)| theta(v)) for v in n_views)
        view_log_probs = [self.view_models_[v].comp_log_probs(X[v])
                          for v in range(self.n_views)]

        # TODO: comment here -- this is the critical step
        # k is the overall cluster index
        # view_idxs = i0, i1 are the view specific indices
        # i.e. k = 0 -> i0 = i1 = 0, k = 1 -> i0 = 0, i1 = 1, etc

        for k in range(self.n_components):
            view_idxs = self._get_view_clust_idx(k)

            # f(x| theta_k) = sum_v f(x(v) | theta_k)
            for v in range(self.n_views):
                comp_lpr[:, k] += view_log_probs[v][:, view_idxs[v]]

        return comp_lpr

    def sample(self, n_samples=1, random_state=None):
        """
        Generate random samples from the model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        Returns
        -------
        X : list of array_like, shape (n_samples, n_features)
            List of samples
        """
        check_is_fitted(self)

        rng = check_random_state(random_state)
        pi = self.weights_
        y_overall = rng.choice(a=np.arange(len(pi)), size=n_samples,
                               replace=True, p=pi)

        samples = [np.zeros((n_samples, self.metadata_['n_features'][v]))
                   for v in range(self.n_views)]

        for i in range(n_samples):
            x = self.sample_from_comp(y=y_overall[i], random_state=rng)
            for v in range(self.n_views):
                samples[v][i, :] = x[v]

        y_views = np.array([self._get_view_clust_idx(y) for y in y_overall])

        return samples, y_overall, y_views

    def sample_from_comp(self, y, random_state=None):
        view_idxs = self._get_view_clust_idx(y)
        return [self.view_models_[v].sample_from_comp(view_idxs[v])
                for v in range(self.n_views)]

    def _n_cluster_parameters(self):
        if hasattr(self, 'view_models_'):
            return sum(vm._n_cluster_parameters()
                       for vm in self.view_models_)

    def _get_view_clust_idx(self, k):
        """Retuns the view cluster indices for each view for an overall cluster index.

        Returns
        -------
        view_idxs : array, shape (n_views, )
        """

        return np.unravel_index(indices=k,
                                shape=self.n_view_components,
                                order='C')
        # idx_0, idx_1 = vec2devec_idx(k,
        #                              n_rows=self.n_view_components[0],
        #                              n_cols=self.n_view_components[1])

        # return idx_0, idx_1

    def reorder_components(self, new_idxs):
        raise NotImplementedError

    def drop_component(self, comps):
        raise NotImplementedError

    def _drop_component_params(self, k):
        raise NotImplementedError

    def _reorder_component_params(self, new_idxs):
        raise NotImplementedError

    def bic(self, X):
        """
        Bayesian information criterion for the current model fit
        and the proposed data.

        Parameters
        ----------
        X : array of shape(n_samples, n_dimensions)
        Returns
        -------
        bic: float (the lower the better)
        """
        check_is_fitted(self)
        n = X[0].shape[0]  # only difference from single view
        return -2 * self.log_likelihood(X) + np.log(n) * self._n_parameters()

    def predict_view_labels(self, X):
        """
        Predicts the view labels the given test samples belongs to.
        This is simply a transformation of predict

        Parameters
        ----------
        X: list of array-like

        Output
        ------
        y_pred_view: array-like, (n_samples_test, n_views)

        """
        y_pred_overall = self.predict(X)
        y_pred_view = np.array([self._get_view_clust_idx(y)
                                for y in y_pred_overall])

        return y_pred_view

    def predict_view_marginal_probas(self, X):
        """
        Parameters
        ----------
        X: list of array-like
            Observed view data.

        Output
        ------
        view_clust_probas: list of array-like
            The vth entry of this list is the
            (n_samples_test, n_view_components[v]) matrix whose (i, k_v)th
            entry is the probability that sample i belongs to view cluster k_v
        """

        p_overall = self.predict_proba(X)
        n_samples = len(p_overall)

        view_clust_probas = [np.zeros((n_samples, self.n_view_components[v]))
                             for v in range(self.n_views)]

        for k in range(self.n_components):
            view_idxs = self._get_view_clust_idx(k)

            for i, v in product(range(n_samples), range(self.n_views)):
                view_clust_probas[v][i, view_idxs[v]] += p_overall[i, k]

        return view_clust_probas


class MultiViewEMixin(EMfitMMMixin):

    def initialize_parameters(self, X, random_state=None):
        """
        Parameters
        ----------
        X: list of array-like
            List of data for each view.

        random_state: int, None
            Random seed for initializations.
        """

        random_state = check_random_state(random_state)

        ##############################
        # initialize view parameters #
        ##############################

        init_view_params = []

        if self.init_params_method == 'fit':
            init_fitted_view_models = []

        for v in range(self.n_views):
            vm = deepcopy(self.base_view_models[v])

            if self.init_params_method == 'init':
                vm.initialize_parameters(X[v], random_state=random_state)

            elif self.init_params_method == 'fit':
                vm.fit(X[v])

            elif self.init_params_method == 'user':
                # make sure model is already fitted
                # check_is_fitted(vm)
                vm._set_parameters(self.init_params_value[v])

            else:
                raise ValueError('bad input for init_params_method')

            init_view_params.append(vm._get_parameters())

            if self.init_params_method == 'fit':
                init_fitted_view_models.append(deepcopy(vm))

        #####################
        # initalize weights #
        #####################

        if self.init_weights_method == 'uniform':
            init_weights = np.ones(self.n_components)
            init_weights = init_weights / sum(init_weights)

        elif 'combine' in self.init_weights_method:
            # TODO: fix for V>2 views

            if self.init_weights_method == 'combine_indep_weights':

                view_weights = [init_fitted_view_models[v].weights_
                                for v in range(self.n_views)]

                view_comps = self.n_view_components
                weights_mat = np.zeros(*view_comps)
                for i in range(np.product(view_comps)):
                    multi_idx = np.unravel_index(indices=i,
                                                 shape=view_comps,
                                                 order='C')

                    vals = [view_weights[multi_idx[v]]
                            for v in range(self.n_views)]
                    weights_mat[multi_idx] = np.product(vals)

            else:
                # TODO: should we even bother with the other strategies
                if self.n_views > 2:
                    raise NotImplementedError

                mm_0 = init_fitted_view_models[0]
                mm_1 = init_fitted_view_models[1]

                weights_mat = \
                    combined_weight_from_sep_view_models(mm_0=mm_0,
                                                         mm_1=mm_1,
                                                         X=X,
                                                         method=self.init_weights_method)
            init_weights = weights_mat.reshape(-1)
            init_weights = init_weights / sum(init_weights)

        elif self.init_weights_method == 'user':

            init_weights = np.array(self.init_weights_value).reshape(-1)

        else:
            raise ValueError('bad input for init_weights_method')

        init_params = {'views': init_view_params, 'weights': init_weights}

        init_params = self._post_initialization(init_params=init_params,
                                                X=X,
                                                random_state=random_state)

        self._set_parameters(params=init_params)
        self._check_clust_parameters(X)

    def _get_init_clust_parameters(self, X, random_state):
        # Not used by multi-view model
        raise NotImplementedError

    def _get_init_weights(self, X, random_state):
        # Not used by multi-view model
        raise NotImplementedError

    def _update_user_init(self, init_params=None, init_weights=None):
        # Not used by multi-view model
        raise NotImplementedError

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
        log_resp_mat = log_resp.reshape(-1, *self.n_view_components)

        view_params = [None for v in range(self.n_views)]
        for v in range(self.n_views):

            # sum over other views
            # note samples are on the first axis hence the +1
            axes2sum = set(range(1, self.n_views + 1))
            axes2sum = list(axes2sum.difference([v + 1]))
            view_log_resp = np.apply_over_axes(logsumexp,
                                               a=log_resp_mat,
                                               axes=axes2sum)

            # TODO: not sure why the argument to squeeze
            view_log_resp = view_log_resp.squeeze(axis=tuple(axes2sum))

            view_params[v] = self.view_models_[v].\
                _m_step_clust_params(X=X[v], log_resp=view_log_resp)

        return view_params

    def _m_step(self, X, E_out):
        log_resp = E_out['log_resp']

        view_params = self._m_step_clust_params(X=X, log_resp=log_resp)
        weights = self._m_step_weights(X=X, log_resp=log_resp)
        return {'views': view_params, 'weights': weights}
