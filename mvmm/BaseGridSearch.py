from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
import pandas as pd
from abc import ABCMeta, abstractmethod
from time import time

from sklearn.base import BaseEstimator, MetaEstimatorMixin
from mvmm.clustering_measures import several_unsupervised_cluster_scores, \
    MEASURE_MIN_GOOD

# TODO: add random seed


class BaseGridSearch(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    def __init__(self,
                 base_estimator,
                 param_grid={},
                 select_metric='bic',
                 metrics2compute=['aic', 'bic'],
                 n_jobs=None,
                 backend=None,
                 verbose=0,
                 pre_dispatch='2*n_jobs'):

        self.base_estimator = base_estimator
        self.param_grid = param_grid
        self.select_metric = select_metric
        self.metrics2compute = metrics2compute
        self.n_jobs = n_jobs
        self.backend = backend
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch

    @abstractmethod
    def fit_and_score(self, estimator, X, parameters, return_estimator=True):
        """
        Fits and estimator on a dataset and scores results.

        Output
        ------
        scores, metadata, estimator
        """
        pass

    @property
    def param_grid_(self):
        """
        List of all parameter settings
        """
        return list(ParameterGrid(self.param_grid))

    def fit(self, X):

        if self.verbose >= 1:
            print("Fitting {} candidates".format(len(self.param_grid_)))

        start_time = time()
        if self.n_jobs is not None:
            parallel = Parallel(n_jobs=self.n_jobs,
                                backend=self.backend,
                                verbose=self.verbose,
                                pre_dispatch=self.pre_dispatch)

            with parallel:

                results = \
                    parallel(delayed(self.fit_and_score)
                             (clone(self.base_estimator), X=X,
                              parameters=params)
                             for params in self.param_grid_)
        else:
            results = [self.fit_and_score(clone(self.base_estimator),
                                          X=X, parameters=params)
                       for params in self.param_grid_]

        self.metadata_ = {'fits': [res[1] for res in results],
                          'fit_time': time() - start_time}

        self.estimators_ = [res[2] for res in results]

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
    def best_params_(self):
        """
        Parameter setting for selected model.
        """
        if self.check_fit():
            return self.metadata_['fits'][self.best_idx_]['parameters']
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
