from time import time
import pandas as pd
import matplotlib.pyplot as plt

from mvmm.BaseGridSearch import BaseGridSearch
from mvmm.viz_utils import set_xaxis_int_ticks


def fit_and_score(estimator, X, parameters):
    """
    Fits a mixture model on a dataset then comptues aic/bic scores.

    Output
    ------
    scores, metadata, estimator
    """
    metadata = {}
    scores = {}

    estimator.set_params(**parameters)

    # fit and scores
    start_time = time()
    estimator.fit(X)
    metadata['fit_time'] = time() - start_time
    metadata['n_samples'] = X.shape
    metadata['parameters'] = parameters

    scores['bic'] = estimator.bic(X)
    scores['aic'] = estimator.aic(X)

    return scores, metadata, estimator


class MMGridSearch(BaseGridSearch):
    def __init__(self,
                 base_estimator,
                 param_grid={},
                 select_metric='bic',
                 metrics2compute=['bic', 'aic'],
                 n_jobs=None,
                 verbose=0,
                 pre_dispatch='2*n_jobs'):

        super().__init__(base_estimator=base_estimator,
                         param_grid=param_grid,
                         select_metric=select_metric,
                         metrics2compute=metrics2compute,
                         n_jobs=n_jobs,
                         verbose=verbose,
                         pre_dispatch=pre_dispatch)

    def fit_and_score(self, estimator, X, parameters):
        return fit_and_score(estimator, X, parameters)


MMGridSearch.fit_and_score.__doc__ = fit_and_score.__doc__


# visualize model selection results

def plot_comp_vs_bic(gs):
    """
    Plot number of components against BIC
    """
    # TODO-feat: add functionality for grid search search
    # over other parameters than just n_components

    est_n_comp = gs.best_params_['n_components']

    params = pd.DataFrame(gs.param_grid_)
    # n_components = [p['n_components'] for p in gs.param_grid_]
    n_components = params['n_components']
    bics = gs.scores_['bic']
    plt.plot(n_components, bics, marker='.')

    plt.xlabel('n_components')
    plt.ylabel('BIC')
    set_xaxis_int_ticks()

    plt.axvline(est_n_comp,
                label='estimated {} components'.format(est_n_comp),
                color='red')
    plt.legend()
