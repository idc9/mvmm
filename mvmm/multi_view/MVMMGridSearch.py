from time import time
import re

from mvmm.BaseGridSearch import BaseGridSearch
from mvmm.multi_view.TwoStage import TwoStage


def fit_and_score(estimator, X, parameters, return_estimator=True):
    """
    Fits a multi-view mixture model to a dataset and computes BIC scores.

    parameters = {'view_0__n_components':  3,
                  'view_1__n_components':  3,
                  'n_init': 5}


    Parameters
    ----------
    estimator:

    X:

    parameters: dict


    Output
    ------
    scores, metadata, estimator
    """

    n_views = estimator.n_views
    metadata = {}
    scores = {}

    est_params = {}  # parameters for the top level estimator
    view_params = [{} for _ in range(n_views)]  # params for each view model
    for key in parameters.keys():
        is_view_param, view_idx, param_name = parse_view_param(key)

        if is_view_param:
            view_params[view_idx][param_name] = parameters[key]
        else:
            est_params[key] = parameters[key]

    # set parameters
    if isinstance(estimator, TwoStage):
        # TODO-FEAT: allow grid search over start parameters as well
        estimator.base_final.set_params(**est_params)

        if any(len(viewparm) > 0 for viewparm in view_params):
            raise ValueError('Setting final view model parameters' \
                             'not currently working')

        for v in range(n_views):
            estimator.\
                base_final.\
                base_view_models[v].\
                set_params(**view_params[v])
    else:
        estimator.set_params(**est_params)
        for v in range(n_views):
            estimator.\
                base_view_models[v].\
                set_params(**view_params[v])

    # fit and scores
    start_time = time()
    estimator.fit(X)
    metadata['fit_time'] = time() - start_time

    scores['bic'] = estimator.bic(X)
    scores['aic'] = estimator.aic(X)

    metadata['n_samples'] = X[0].shape[0]
    metadata['parameters'] = parameters

    return scores, metadata, estimator


def parse_view_param(key):
    """
    Parses the keys of the parameter dictionary to identify view parameters
    """
    view_match = re.search(r'view_\d+__', key)

    if view_match is not None:
        is_view_param = True
        view_idx = re.search('\d+', view_match.group()).group()
        view_idx = int(view_idx)
        param_name = key.split('view_{}__'.format(view_idx))[1]
    else:
        is_view_param = False
        view_idx = None
        param_name = None

    return is_view_param, view_idx, param_name


class MVMMGridSearch(BaseGridSearch):
    def __init__(self,
                 base_estimator,
                 param_grid={},
                 select_metric='bic',
                 metrics2compute=['aic', 'bic'],
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


MVMMGridSearch.fit_and_score.__doc__ = fit_and_score.__doc__
