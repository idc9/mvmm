from abc import ABCMeta

# import numpy as np
from datetime import datetime
from time import time

from sklearn.base import BaseEstimator
from sklearn.base import DensityMixin
from sklearn.utils import check_random_state

from mvmm.utils import get_seeds
from copy import deepcopy
# from textwrap import dedent


class TwoStage(DensityMixin, BaseEstimator, metaclass=ABCMeta):
    """
    Initializes one model by fitting another model.

    Parameters
    ----------
    base_start:
        First estimator to fit.

    base_final:
        Second estimator to fit.

    n_init: int
        Number of initializations.

    random_state: int, None, seed
        Random seed.

    verbosity: int
        How much printout you want.

    Attributes
    ----------
    start_:
        The fit start estimator.

    final_:
        The fit final estimator.

    fit_data_:

    metadata_:

    """
    def __init__(self, base_start, base_final, n_init=1,
                 random_state=None, verbosity=0):
        self.base_start = base_start
        self.base_final = base_final

        self.n_init = n_init
        self.random_state = random_state

        self.verbosity = verbosity

    def fit(self, X, y=None):

        random_state = check_random_state(self.random_state)
        init_seeds_start = get_seeds(n_seeds=self.n_init,
                                     random_state=random_state)

        init_seeds_final = get_seeds(n_seeds=self.n_init,
                                     random_state=random_state)

        # lower bounds for each initialization
        init_loss_vals = []

        start_time = time()

        for i in range(self.n_init):

            if self.verbosity >= 1:
                now = datetime.now().strftime("%H:%M:%S")
                print('Beginning initialization {} at {}'.format(i + 1, now))

            ######################
            # fit starting model #
            ######################
            start = deepcopy(self.base_start)
            start.set_params(random_state=init_seeds_start[i])
            start.fit(X=X)

            # get initial parameters for final
            init_params_value = [start.view_models_[v]._get_parameters()
                                 for v in range(start.n_views)]
            init_weights_value = start.weights_

            if self.verbosity >= 1:
                now = datetime.now().strftime("%H:%M:%S")
                print('Beginning final estimator at {}'.format(now))

            ###################
            # fit final model #
            ###################
            final = deepcopy(self.base_final)
            final.set_params(random_state=init_seeds_final[i],
                             init_params_method='user',
                             init_params_value=init_params_value,
                             init_weights_method='user',
                             init_weights_value=init_weights_value)
            final.fit(X=X)

            loss_val = final.opt_data_['loss_val']
            if 'success' in final.opt_data_.keys():
                success = final.opt_data_['success']
            else:
                success = True

            if i == 0 or (loss_val < min(init_loss_vals) and success):
                best_start = deepcopy(start)
                best_final = deepcopy(final)
                best_init = i

            init_loss_vals.append(loss_val)

        self.start_ = best_start
        self.final_ = best_final

        self.fit_data_ = {'init_loss_vals': init_loss_vals,
                          'best_init': best_init,
                          'init_seeds_start': init_seeds_start,
                          'init_seeds_final': init_seeds_final}

        self.metadata_ = {'fit_time': time() - start_time}
        return self

    def predict(self, X):
        """
        Predict the labels for the data samples in X using trained model.
        """
        return self.final_.predict(X)

    def predict_proba(self, X):
        """
        Predict posterior probability of each component given the data.
        """
        return self.final_.predict_proba(X)

    def sample(self, n_samples=1):
        """
        Generate random samples from the fitted Gaussian distribution.
        """
        return self.final_.sample(n_samples=n_samples)

    def score(self, X, y=None):
        """
        Compute the per-sample average log-likelihood of the given data X.
        """
        return self.final_.score(X)

    def score_samples(self, X):
        """
        Compute the weighted log probabilities for each sample.
        """
        return self.final_.score_samples(X)

    def bic(self, X):
        return self.final_.bic(X)

    def aic(self, X):
        return self.final_.aic(X)

    @property
    def n_views(self):
        if hasattr(self, 'final_'):
            return self.final_.n_views
        else:
            return self.base_start.n_views

    @property
    def n_components(self):
        """
        Returns the total number of clusters.
        """
        return self.final_.n_components

    @property
    def n_view_components(self):
        """
        Number of components in each views.
        """
        self.final_.n_view_components

    @property
    def weights_mat_(self):
        """
        Returns the weights matrix.
        """
        self.final_.weights_mat_
