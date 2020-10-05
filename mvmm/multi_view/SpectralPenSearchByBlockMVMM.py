from sklearn.base import clone
import pandas as pd
from abc import ABCMeta
from time import time
from datetime import datetime
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.base import BaseEstimator, MetaEstimatorMixin

from mvmm.utils import get_seeds
from mvmm.multi_view.utils import linspace_zero_to, \
    expspace_zero_to, polyspace_zero_to
from mvmm.multi_view.block_diag.graph.linalg import geigh_Lsym_bp_smallest
from mvmm.multi_view.block_diag.utils import asc_sort
from mvmm.clustering_measures import unsupervised_cluster_scores, \
    several_unsupervised_cluster_scores, MEASURE_MIN_GOOD


class SpectralPenSearchByBlockMVMM(MetaEstimatorMixin, BaseEstimator,
                                   metaclass=ABCMeta):
    """
    Does a grid search over the continuous hyper-parameter for the spentral penalized MVMM. Stores the best MVMM for each block.

    Parameters
    ----------
    base_mvmm_0:
        Unconstrained MVMM.

    base_wbd_mvmm: mvmm.multi_view.BlockDiagMVMM.BlockDiagMVMM()
        The base class for the spectral penalized MVMM

    eval_weights:
        The weights to put on the generalized eigenvalues.

    adapt_expon:


    max_n_blocks:
        Maximum number of blocks to get i.e. the number of eigenvalues to penalized.

    user_eval_weights:
        (Optional) User provied eignvalue weights.

    pen_max: str, float
        Largest penalty value to try. If 'default' will make an automatic, educated guess.

    n_pen_seq: int
        Number of penalty values to try.

    user_pen_vals: None, list
        (Optional) User provided penalty values to try

    default_c: float
        Multiplicative factor for infering pen_max with the default method.

    pen_seq_spacing: str
        How to space the penalty values along the penalty sequence.

    n_init: int
        Number of random initalizations.

    random_state: None, int
        Random seed.

    select_metric: str
        How to pick the best model for each fixed number of blocks.

    metrics2compute: list of st
        Model selection measures to compute for tracking purposes.

    verbosity: int
        Level of printout

    """
    def __init__(self, base_mvmm_0, base_wbd_mvmm,
                 eval_weights='adapt', adapt_expon=1,
                 max_n_blocks='default', user_eval_weights=None,
                 pen_max='default', n_pen_seq=100, user_pen_vals=None,
                 # adapt_pen=False, pen_incr=.5, max_n_pen_incr=200,
                 default_c=100, pen_seq_spacing='lin',
                 n_init=1, random_state=None,
                 select_metric='bic',
                 metrics2compute=['aic', 'bic'],
                 verbosity=0):

        self.base_mvmm_0 = base_mvmm_0
        self.base_wbd_mvmm = base_wbd_mvmm

        self.eval_weights = eval_weights
        self.adapt_expon = adapt_expon
        self.max_n_blocks = max_n_blocks
        self.user_eval_weights = user_eval_weights

        self.pen_max = pen_max
        self.n_pen_seq = n_pen_seq
        self.user_pen_vals = user_pen_vals

        self.default_c = default_c
        self.pen_seq_spacing = pen_seq_spacing

        assert pen_seq_spacing in ['lin', 'quad', 'exp']

        # self.adapt_pen = adapt_pen
        # self.pen_incr = pen_incr
        # self.max_n_pen_incr = max_n_pen_incr
        # if self.adapt_pen:
        #     assert self.user_pen_vals is None

        self.random_state = random_state
        self.n_init = n_init

        self.select_metric = select_metric
        self.metrics2compute = metrics2compute

        self.verbosity = verbosity

    def get_pen_seq_from_max(self, pen_max):

        if self.pen_seq_spacing == 'lin':
            return linspace_zero_to(stop=pen_max,
                                    num=self.n_pen_seq)

        elif self.pen_seq_spacing == 'quad':
            return polyspace_zero_to(stop=pen_max,
                                     num=self.n_pen_seq,
                                     deg=2)

        elif self.pen_seq_spacing == 'exp':
            return expspace_zero_to(stop=pen_max,
                                    num=self.n_pen_seq,
                                    base=10)

    @property
    def n_pen_vals_(self):
        if self.user_pen_vals is not None:
            return len(self.user_pen_vals) + 1
        else:
            return self.n_pen_seq + 1

    @property
    def param_grid_(self):
        """
        List of all parameter settings
        """
        if hasattr(self, 'est_n_blocks_'):
            param_grid = {'n_blocks': self.est_n_blocks_}
            return list(ParameterGrid(param_grid))
        else:
            return None

    def get_default_pen_max(self, model, X):
        # steup temp model
        temp_model = clone(model)
        temp_model.view_models_ = \
            [temp_model.base_view_models[v]
             for v in range(temp_model.n_views)]
        temp_model.initialize_parameters(X)
        eval_pen_default = temp_model.\
            get_eval_pen_guess(X=X, c=self.default_c,
                               use_bipt_sp=True,
                               K='default')

        if self.verbosity >= 1:
            print('default pen val', eval_pen_default)

        return eval_pen_default

    def fit(self, X):

        # assert all(self.pen_vals_[1:] > 0)
        # assert len(np.unique(self.pen_vals)) == len(self.pen_vals)

        init_seeds = get_seeds(n_seeds=self.n_init,
                               random_state=self.random_state)

        fit_data = pd.DataFrame()
        n_blocks_best_models = {}
        n_blocks_best_idx = {}
        init_adapt_weights = []
        for init in range(self.n_init):

            if self.verbosity >= 1:
                current_time = datetime.now().strftime("%H:%M:%S")
                print('Initialization {}/{} at {}'.
                      format(init + 1, self.n_init, current_time))

            # max number of evals to penalize
            if self.max_n_blocks == 'default':
                K = min(self.base_mvmm_0.n_view_components)
            else:
                K = int(self.max_n_blocks)

            for pen_idx in range(self.n_pen_vals_):

                if self.verbosity >= 1:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print('Penalty {}/{} at {}'.
                          format(pen_idx + 1, self.n_pen_vals_,
                                 current_time))

                data = {'pen_idx': pen_idx, 'init': init}

                start_time = time()
                if pen_idx == 0:
                    pen_val = None

                    # fit model
                    fit_model = clone(self.base_mvmm_0)
                    fit_model.set_params(random_state=init_seeds[init],
                                         n_init=1)
                    fit_model.fit(X)

                    # get current parameter values for warm starting
                    current_view_params = fit_model._get_parameters()['views']
                    current_bd_weights = fit_model.weights_mat_
                    current_bd_weights = current_bd_weights * \
                        self.base_wbd_mvmm.epsilon_tilde / \
                        current_bd_weights.sum()

                    # track data
                    data['n_blocks'] = 1
                    data['n_steps'] = fit_model.opt_data_['n_steps']

                    # compute adaptive weights
                    if self.eval_weights == 'adapt':

                        evals = geigh_Lsym_bp_smallest(X=self.bd_weights_,
                                                       rank=K,
                                                       zero_tol=1e-10,
                                                       method='tsym')

                        # deal with 0 evals by artificially setting
                        # them to the smallest non-zero eval
                        zero_evals = evals < 1e-6
                        if np.mean(zero_evals) == 1:
                            # edge case: if all evals are 0 just use uiniform
                            evals = np.ones(len(evals))
                        else:
                            evals[zero_evals] = min(evals[~zero_evals])

                        # clip for numerical stability
                        eval_weights = (1 / evals) ** self.adapt_expon

                        init_adapt_weights.append(eval_weights)

                else:

                    # setup and fit model
                    fit_model = clone(self.base_wbd_mvmm)
                    params = {'init_params_method': 'user',
                              'init_params_value': current_view_params,
                              'init_weights_method': 'user',
                              'init_weights_value': current_bd_weights
                              # 'eval_pen_base': pen_val,
                              }

                    params.update({'n_pen_tries': 1,
                                   'n_init': 1,
                                   # 'fine_tune_n_steps': None
                                   })
                    fit_model.set_params(**params)

                    # set eval weights
                    if self.user_eval_weights:
                        eval_weights = self.user_eval_weights
                    elif self.eval_weights == 'adapt':
                        eval_weights = init_adapt_weights[init]
                    elif self.eval_weights == 'uniform':
                        eval_weights = np.ones(K)
                    elif self.eval_weights == 'lin':
                        eval_weights = 1 / np.arange(1, K + 1)
                    elif self.eval_weights == 'quad':
                        eval_weights = (1 / np.arange(1, K + 1)) ** 2
                    elif self.eval_weights == 'exp':
                        eval_weights = .5 ** np.arange(1, K + 1)
                    else:
                        raise ValueError("invalid input for eval_weights: {}"
                                         .format(self.eval_weights))

                    def process(x):
                        x = np.clip(x, a_min=0, a_max=1e5)
                        return asc_sort(x * len(x) / np.sum(x))

                    # eval_weights = np.clip(eval_weights, a_min=0, a_max=1e5)
                    # superficial normalization step keeps
                    # penalty value reasonable
                    # eval_weights *= K / np.sum(eval_weights)
                    # eval_weights = desc_sort(eval_weights)
                    eval_weights = process(eval_weights)
                    fit_model.set_params(eval_weights=eval_weights)

                    # set penalty sequence for this initialization
                    if pen_idx == 1:

                        if self.user_pen_vals is not None:
                            pen_seq = np.sort(self.user_pen_vals)

                        elif self.pen_max == 'default':
                            # compute default max penalty
                            default_pen_max = \
                                self.get_default_pen_max(model=fit_model, X=X)

                            pen_seq = self.\
                                get_pen_seq_from_max(pen_max=default_pen_max)

                        elif self.pen_max != 'default':
                            pen_seq = self.\
                                get_pen_seq_from_max(pen_max=self.pen_max)
                        pen_seq = np.concatenate([[None], pen_seq])

                    # set penalty value
                    pen_val = pen_seq[pen_idx]
                    fit_model.set_params(eval_pen_base=pen_val)

                    fit_model.fit(X)

                    # get current parameter values for warm starting
                    current_view_params = fit_model._get_parameters()['views']
                    current_bd_weights = fit_model.bd_weights_

                    # track data
                    data['pen_val'] = pen_val
                    data['n_blocks'] = fit_model.opt_data_['n_blocks_est']
                    data['n_steps'] = \
                        fit_model.opt_data_['adpt_opt_data']['n_steps']

                # store tracking data
                data['fit_time'] = time() - start_time
                tracking_data = fit_model.compute_tracking_data(X)
                data['loss_val'] = tracking_data['loss_val']
                data['obs_nll'] = tracking_data['obs_nll']

                # TODO: possibly precompute distances
                model_sel_scores = \
                    unsupervised_cluster_scores(X=X,
                                                estimator=fit_model,
                                                measures=self.metrics2compute)
                for measure in model_sel_scores.keys():
                    data[measure] = model_sel_scores[measure]
                # data['bic'] = fit_model.bic(X)
                # data['aic'] = fit_model.aic(X)
                fit_data = fit_data.append(data, ignore_index=True)

                # save this model if it is the best
                current_n_blocks = data['n_blocks']  # current n_blocks
                # get th
                block_scores = fit_data.query("n_blocks == @current_n_blocks")
                if MEASURE_MIN_GOOD[self.select_metric]:
                    best_idx = block_scores[self.select_metric].idxmin()
                else:
                    best_idx = block_scores[self.select_metric].idxmax()
                # best_idx = fit_data.\
                #     query("n_blocks == @n_blocks")[self.select_metric].\
                #     idxmin()
                if fit_data.loc[best_idx, 'init'] == init:
                    n_blocks_best_models[current_n_blocks] = fit_model
                    n_blocks_best_idx[current_n_blocks] = best_idx

        self.est_n_blocks_ = np.sort(list(n_blocks_best_models.keys()))
        self.estimators_ = [n_blocks_best_models[n_blocks]
                            for n_blocks in self.est_n_blocks_]

        int_cols = ['init', 'pen_idx', 'n_blocks', 'n_steps']
        fit_data[int_cols] = fit_data[int_cols].astype(int)
        self.init_fit_data_ = fit_data
        self.fit_init_best_idxs = [n_blocks_best_idx[n_blocks]
                                   for n_blocks in self.est_n_blocks_]

        if self.eval_weights == 'adapt':
            self.init_adapt_weights_ = init_adapt_weights

        self.model_sel_scores_ = \
            several_unsupervised_cluster_scores(X=X,
                                                estimators=self.estimators_,
                                                measures=self.metrics2compute)

        return self

    def check_fit(self):
        return hasattr(self, 'estimators_')

    @property
    def best_idx_(self):
        """
        Index of selected model.
        """
        if self.check_fit():
            if MEASURE_MIN_GOOD[self.select_metric]:
                return self.model_sel_scores_[self.select_metric].idxmin()
            else:
                return self.model_sel_scores_[self.select_metric].idxmax()

        else:
            return None

    @property
    def best_estimator_(self):
        """
        Selected estimator.
        """
        if self.check_fit():
            return self.estimators_[self.best_idx_]
        else:
            return None

    def predict(self, X):
        """
        Predict the labels for the data samples in X using trained model.
        """
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """
        Predict posterior probability of each component given the data.
        """
        return self.best_estimator_.predict_proba(X)

    def sample(self, n_samples=1):
        """
        Generate random samples from the fitted Gaussian distribution.
        """
        return self.best_estimator_.sample(n_samples=n_samples)

    def score(self, X, y=None):
        """
        Compute the per-sample average log-likelihood of the given data X.
        """
        return self.best_estimator_.score(X)

    def score_samples(self, X):
        """
        Compute the weighted log probabilities for each sample.
        """
        return self.best_estimator_.score_samples(X)
