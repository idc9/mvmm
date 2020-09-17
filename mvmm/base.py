from abc import ABCMeta, abstractmethod
from sklearn.utils import check_random_state, check_array
# from sklearn.base import BaseEstimator, DensityMixin
from sklearn.utils.fixes import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted

from datetime import datetime
from time import time
from numbers import Number
from copy import deepcopy
from warnings import warn, simplefilter, catch_warnings
from textwrap import dedent

import numpy as np

from mvmm.utils import get_seeds
from mvmm.opt_utils import check_stopping_criteria


class MixtureModelMixin(metaclass=ABCMeta):
    """
    Base mixture model class.
    """

    @abstractmethod
    def _get_parameters(self):
        raise NotImplementedError

    @abstractmethod
    def _set_parameters(self, params):
        raise NotImplementedError

    def fit(self, X):
        """
        Fits the mixture model.

        Parameters
        ----------
        X:
            The observed data.
        """
        X = _check_X(X, self.n_components, ensure_min_samples=2)
        self.metadata_ = {'n_samples': X.shape[0],
                          'n_features': X.shape[1]}
        self._check_parameters(X)
        self._check_fitting_parameters(X)

        start_time = time()
        self._fit(X)
        self.metadata_['fit_time'] = time() - start_time
        return self

    @abstractmethod
    def _fit(self, X):
        # subclass should implement this!
        raise NotImplementedError

    def _check_parameters(self, X):
        """
        Check values of the basic parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        # sub-class should overwrite
        pass

    def _check_clust_param_values(self, X):
        """Check values of the basic fitting parameters.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        pass

    def _check_clust_parameters(self, X):
        """
        Checks cluster parameters and weights.
        """
        pass

    def score_samples(self, X):
        """
        Computes the observed data log-likelihood for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log probabilities of each data point in X.
        """
        check_is_fitted(self)
        # X = _check_X(X, None, self.metadata_['n_features'])

        return logsumexp(self.log_probs(X), axis=1)

    def log_probs(self, X):
        """
        Computes the log-likelihood for each sample for each cluster including the cluster weihts.

        Parameters
        ----------
        X: array-like, (n_samples, n_features)

        Output
        ------
        log_probs: array-like, (n_samples, n_components)
        """
        check_is_fitted(self)
        # formerly _estimate_weighted_log_prob
        return self.comp_log_probs(X) + np.log(self.weights_)

    @abstractmethod
    def comp_log_probs(self, X):
        """
        Computes the log-likelihood for each sample for each cluster without the cluster weights.

        Parameters
        ----------
        X: array-like, (n_samples, n_features)

        Output
        ------
        comp_log_probs: array-like, (n_samples, n_components)

        """
        raise NotImplementedError

    def score(self, X, y=None):
        """Compute the per-sample average log-likelihood of the given data X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_dimensions)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        log_likelihood : float
            Log likelihood of the fussian mixture given X.
        """
        return self.score_samples(X).mean()

    def log_likelihood(self, X):
        """
        Computes the observed data log-likelihood.

        """
        check_is_fitted(self)
        return self.score_samples(X).sum()

    def predict(self, X):
        """
        Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        check_is_fitted(self)
        # X = _check_X(X, None, self.means_.shape[1])
        return self.log_probs(X).argmax(axis=1)

    def predict_proba(self, X):
        """
        Predict posterior probability of each component given the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Returns the probability each Gaussian (state) in
            the model given each sample.
        """
        check_is_fitted(self)
        # X = _check_X(X, None, self.means_.shape[1])
        log_prob = self.log_probs(X)
        log_resp = self.log_resps(log_prob)
        return np.exp(log_resp)

    def log_resps(self, log_prob):
        """
        Estimate log probabilities and responsibilities for each sample.
        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)
        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        # weighted_log_prob = self.log_probs(X)
        log_prob_norm = logsumexp(log_prob, axis=1)
        with np.errstate(under='ignore'):
            # ignore underflow
            log_resp = log_prob - log_prob_norm[:, np.newaxis]
        return log_resp

    def sample(self, n_samples=1, random_state=None):
        """
        Generate random samples from the model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        random_state: int, None
            Random seed.

        Returns
        -------
        X : array-like, shape (n_samples, n_features)
            List of samples

        """
        check_is_fitted(self)

        rng = check_random_state(random_state)
        pi = self.weights_
        y = rng.choice(a=np.arange(len(pi)), size=n_samples,
                       replace=True, p=pi)

        samples = [None for _ in range(n_samples)]
        for i in range(n_samples):
            samples[i] = self.sample_from_comp(y=y[i], random_state=rng)

        return np.array(samples), y

    @abstractmethod
    def sample_from_comp(self, y, random_state=None):
        """
        Samples one observation from a cluster.

        Parameters
        ----------
        y: int
            Which cluster to sample from.

        random_state: None, int
            Random seed.

        Output
        ------
        x: array-like, (n_features, )

        """
        raise NotImplementedError

    def bic(self, X):
        """
        Bayesian information criterion for the current model fit
        and the proposed data

        Parameters
        ----------
        X : array of shape(n_samples, n_featuers)

        Returns
        -------
        bic: float (the lower the better)
        """
        check_is_fitted(self)
        n = X.shape[0]
        return -2 * self.log_likelihood(X) + np.log(n) * self._n_parameters()

    def aic(self, X):
        """
        Akaike information criterion for the current model fit
        and the proposed data.

        Parameters
        ----------
        X : array of shape(n_samples, n_featuers)
        Returns
        -------
        aic: float (the lower the better)
        """
        check_is_fitted(self)
        return -2 * self.log_likelihood(X) + 2 * self._n_parameters()

    def _n_parameters(self):
        """
        Returns the number of model parameters e.g. for BIC/AIC.
        """
        check_is_fitted(self)
        return self._n_cluster_parameters() + self._n_weight_parameters()

    def _n_weight_parameters(self):
        """
        Number of weight parameters
        """
        return self.n_components - 1

    @abstractmethod
    def _n_cluster_parameters(self):
        raise NotImplementedError

    def reorder_components(self, new_idxs):
        """
        Re-orders the components
        """
        assert set(new_idxs) == set(range(self.n_components))
        params = self._reorder_component_params(new_idxs)
        params['weights'] = self.weights_[new_idxs]
        self._set_parameters(params)
        return self

    def drop_component(self, comps):
        """
        Drops a component or components from the model.

        Parameters
        ----------
        comps: int, list of ints
            Which component(s) to drop
        """
        check_is_fitted(self)
        if isinstance(comps, Number):
            comps = [comps]

        # sort componets in decreasing order so that lower indicies
        # are preserved after dropping higher indices
        comps = np.sort(comps)[::-1]

        # don't drop every component
        assert len(comps) < self.n_components

        weights = deepcopy(self.weights_)

        for k in comps:

            self.n_components = self.n_components - 1

            params = self._drop_component_params(k)
            weights = np.delete(weights, k)
            params['weights'] = weights / sum(weights)
            self._set_parameters(params)

        return self

    def _drop_component_params(self, k):
        """
        Drops the cluster parameters of a single componet.
        Subclass should overwrite.
        """
        raise NotImplementedError

    def _reorder_component_params(self, new_idxs):
        """
        Re-orders the component cluster parameters
        """
        raise NotImplementedError


class EMfitMMMixin(metaclass=ABCMeta):
    """
    Based EM mixture model class.
    """

    def __init__(self,
                 max_n_steps=200,
                 abs_tol=1e-9,
                 rel_tol=None,
                 n_init=5,
                 init_params_method='rand_resp',
                 init_params_value=None,
                 init_weights_method='uniform',
                 init_weights_value=None,
                 random_state=None,
                 verbosity=0,
                 history_tracking=0):

        self.max_n_steps = max_n_steps
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.n_init = n_init

        self.init_params_method = init_params_method
        self.init_params_value = init_params_value
        self.init_weights_method = init_weights_method
        self.init_weights_value = init_weights_value
        self.random_state = random_state

        self.verbosity = verbosity
        self.history_tracking = history_tracking

    def _fit(self, X):
        params, self.opt_data_ = self._best_em_loop(X)
        self._set_parameters(params)

    def _check_fitting_parameters(self, X):

        if self.n_components < 1:
            raise ValueError("Invalid value for 'n_components': %d "
                             "Estimation requires at least one component"
                             % self.n_components)

        if self.abs_tol is not None and self.abs_tol < 0.:
            raise ValueError("Invalid value for 'abs_tol': %.5f "
                             "Tolerance must be non-negative"
                             % self.abs_tol)

        if self.rel_tol is not None and self.rel_tol < 0.:
            raise ValueError("Invalid value for 'rel_tol': %.5f "
                             "Tolerance must be non-negative"
                             % self.rel_tol)

        if self.n_init < 1:
            raise ValueError("Invalid value for 'n_init': %d "
                             "Estimation requires at least one run"
                             % self.n_init)

        if self.max_n_steps < 0:
            raise ValueError("Invalid value for 'max_n_steps': %d "
                             ", must be non negative."
                             % self.max_n_steps)

    def initialize_parameters(self, X, random_state=None):
        random_state = check_random_state(random_state)

        init_params = self._get_init_clust_parameters(X, random_state)
        init_weights = self._get_init_weights(X, random_state)

        # over write initialized parameters with user provided parameters
        init_params, init_weights = \
            self._update_user_init(init_params=init_params,
                                   init_weights=init_weights)

        init_params['weights'] = init_weights

        init_params = self._post_initialization(init_params=init_params,
                                                X=X,
                                                random_state=random_state)
        self._set_parameters(params=init_params)
        self._check_clust_parameters(X)

    def _post_initialization(self, init_params, X, random_state):
        """
        Called after initialization; subclass may optinoally overwrite.

        Output
        ------
        dict that gets pass to _set_parameters()
        """
        return init_params

    def _get_init_clust_parameters(self, X, random_state):

        if self.init_params_method == 'rand_resp':
            n_samples = X.shape[0]
            resp = random_state.rand(n_samples, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
            init_params = self._m_step_clust_params(X=X, log_resp=np.log(resp))

        else:
            raise ValueError("Invalid value for 'init_params_method': {}"
                             "".format(self.init_params_method))

        return init_params

    def _get_init_weights(self, X, random_state):

        if self.init_weights_method == 'uniform':
            init_weights = np.ones(self.n_components) / self.n_components

        else:
            raise ValueError("Invalid value for 'init_weights_method': {}"
                             "".format(self.init_weights_method))

        return init_weights

    def _update_user_init(self, init_params=None, init_weights=None):

        if init_params is None:
            init_params = {}

        # drop parameters whom the user has provided initial values for
        if self.init_params_value is not None:
            for k in self.init_params_value:
                if k in init_params.keys():
                    init_params[k] = self.init_params_value[k]

        if self.init_weights_value is not None:
            init_weights = self.init_weights_value

        return init_params, init_weights

    def _m_step(self, X, E_out):
        log_resp = E_out['log_resp']

        new_params = self._m_step_clust_params(X=X, log_resp=log_resp)
        new_params['weights'] = self._m_step_weights(X=X, log_resp=log_resp)

        return new_params

    def _m_step_weights(self, X, log_resp):
        n_samples = log_resp.shape[0]
        resp = np.exp(log_resp)
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        return nk / n_samples

    def _m_step_clust_params(self, X, log_resp):
        raise NotImplementedError

    def _e_step(self, X):
        """

        Parameters
        ----------
        X: array-like, (n_samples, n_features)
            The data.

        Output
        ------
        out: dict

        out['log_resp']: array-like, (n_samples, n_components)
            The responsitiblities.

        out['obs_nll']: float
            The observed negative log-likelihood of the data at the current
            parameters.
        """
        log_prob = self.log_probs(X)
        log_resp = self.log_resps(log_prob)

        obs_nll = - logsumexp(log_prob, axis=1).mean()

        return {'log_resp': log_resp, 'obs_nll': obs_nll}

    def compute_tracking_data(self, X, E_out=None):
        """
        Parameters
        ----------
        X: array-like, (n_samples, n_features)
            The data.

        E_out: None, dict
            (optional) The output from _e_step which includes the
            observe neg log lik. Saves computational time.

        Output
        ------
        dict:
            out['obs_nll']: float
                The observed neg log-lik.

            out['loss_val']: float
                The loss function; in this case just obs_nll.

            out['model']: (only if history_tracking >=2)
                The current model parameters.
        """
        out = {}

        if E_out is None:
            E_out = self._e_step(X)

        if E_out is not None:
            out['obs_nll'] = E_out['obs_nll']
        else:
            out['obs_nll'] = - self.score(X)
        # obs_nll = - self.score(X)
        # log_probs = self.log_probs(X)
        # obs_nll = - logsumexp(log_probs, axis=1).mean()
        out['loss_val'] = out['obs_nll']

        # maybe track model history
        if self.history_tracking >= 2:
            out['model'] = deepcopy(self._get_parameters())

        return out

    def _em_loop(self, X):

        # make sure the cluster parameters have been properly initialized
        self._check_clust_param_values(X)

        # initialize history tracking
        history = {}

        converged = False
        prev_loss = None

        start_time = time()

        current_loss = np.nan
        converged = False
        step = -1

        if self.history_tracking >= 1:
            history['init_params'] = self._get_parameters()

        for step in range(self.max_n_steps):

            if self.verbosity >= 2:
                t = datetime.now().strftime("%H:%M:%S")
                print('EM step {} at {}'.format(step + 1, t))

            ################################
            # E-step and check convergence #
            ################################

            E_out = self._e_step(X)

            tracking_data = self.compute_tracking_data(X, E_out)
            current_loss = tracking_data['loss_val']

            # check convergence after taking at least one step
            if step >= 1:

                if current_loss > prev_loss and self.verbosity > 1:
                    warn('loss increasing from {} to {}'.format(prev_loss,
                                                                current_loss))

                abs_diff = abs(current_loss - prev_loss)
                rel_diff = abs_diff / abs(current_loss)

                converged = check_stopping_criteria(abs_diff=abs_diff,
                                                    rel_diff=rel_diff,
                                                    abs_tol=self.abs_tol,
                                                    rel_tol=self.rel_tol)

            # track data
            for k in tracking_data.keys():
                if k not in history.keys():
                    history[k] = []
                history[k].append(tracking_data[k])

            # if converged then stop, otherwise update parameters
            if converged:
                break

            else:
                ##########
                # M-step #
                ##########

                # M-step
                new_params = self._m_step(X=X, E_out=E_out)

                # update new parameters
                self._set_parameters(new_params)

                prev_loss = deepcopy(current_loss)

        params = self._get_parameters()

        opt_data = {'loss_val': current_loss,
                    'n_steps': step,
                    'converged': converged,
                    'runtime': time() - start_time,
                    'success': True,  # subclasses may have failed EM loop
                    'history': history}

        if not converged and self.verbosity >= 1:
            warn('EM did not converge', ConvergenceWarning)

        return params, opt_data

    def _best_em_loop(self, X):
        """
        Runs the EM algorithm from multiple initalizations and picks the best solution.
        """

        init_seeds = get_seeds(n_seeds=self.n_init,
                               random_state=self.random_state)

        # lower bounds for each initialization
        init_loss_vals = []

        for i in range(self.n_init):

            if self.verbosity >= 1:
                time = datetime.now().strftime("%H:%M:%S")
                print('Beginning initialization {} at {}'.format(i + 1, time))

            # initialize parameters if not warm starting
            self.initialize_parameters(X, random_state=init_seeds[i])

            # EM loop
            with catch_warnings():
                simplefilter('ignore', ConvergenceWarning)
                params, opt_data = self._em_loop(X=X)

            # update parameters if this initialization is better
            loss_val = opt_data['loss_val']

            if 'success' in opt_data.keys():
                success = opt_data['success']
            else:
                success = True

            if i == 0 or (loss_val < min(init_loss_vals) and success):
                best_params = params
                best_opt_data = opt_data
                best_opt_data['init'] = i
                best_opt_data['random_state'] = init_seeds[i]

            init_loss_vals.append(loss_val)

        best_opt_data['init_loss_vals'] = init_loss_vals

        if not best_opt_data['converged'] and self.verbosity >= 2:
            warn('Best EM initalization, {} did not'
                 'converge'.format(best_opt_data['init']), ConvergenceWarning)

        return best_params, best_opt_data


def _check_X(X, n_components=None, n_features=None, ensure_min_samples=1):
    """Check the input data X.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    n_components : int
    Returns
    -------
    X : array, shape (n_samples, n_features)
    """
    X = check_array(X, dtype=[np.float64, np.float32],
                    ensure_min_samples=ensure_min_samples)
    if n_components is not None and X.shape[0] < n_components:
        raise ValueError('Expected n_samples >= n_components '
                         'but got n_components = %d, n_samples = %d'
                         % (n_components, X.shape[0]))
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("Expected the input data X have %d features, "
                         "but got %d features"
                         % (n_features, X.shape[1]))
    return X


_em_docs = dict(
    em_param_docs=dedent("""\
    max_n_steps: int
        Maximum number of EM steps.

    abs_tol: float, None
        Absolute tolerance for EM convergence.

    rel_tol: float, None
        (optional) Relative tolerance for EM convergence.

    n_init: int
        Number of random EM initializations.

    init_params_method: str
        How to initalize the cluster parameters e.g. kmeans.

    init_params_value: None, list
        (optional) User provided value used to initalize the cluster parameters.

    init_weights_method: str
        How to initialize the cluster weights.

    init_weights_value: None, array-like
        (optional) User provided value used to initalize the cluster weights.

    random_state: None, int
        (optional) Random seed for initalization.

    verbosity: int
        How verbose the print out should be (lower means quieter).

    history_tracking: int
        How much optimization data to track as the EM algorithm progresses.
    """)
)
