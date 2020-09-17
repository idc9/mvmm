from sklearn.base import BaseEstimator, DensityMixin
from textwrap import dedent

from mvmm.multi_view.base import MultiViewMixtureModelMixin, MultiViewEMixin
from mvmm.base import _em_docs


class MVMM(MultiViewEMixin, MultiViewMixtureModelMixin,
           BaseEstimator, DensityMixin):

    def __init__(self,
                 base_view_models=None,
                 max_n_steps=200,
                 abs_tol=1e-9,
                 rel_tol=None,
                 n_init=1,
                 init_params_method='init',
                 init_params_value=None,
                 init_weights_method='uniform',
                 init_weights_value=None,
                 random_state=None,
                 verbosity=0,
                 history_tracking=0):

        MultiViewEMixin.__init__(self,
                                 max_n_steps=max_n_steps,
                                 abs_tol=abs_tol,
                                 rel_tol=rel_tol,
                                 n_init=n_init,
                                 init_params_method=init_params_method,
                                 init_params_value=init_params_value,
                                 init_weights_method=init_weights_method,
                                 init_weights_value=init_weights_value,
                                 random_state=random_state,
                                 verbosity=verbosity,
                                 history_tracking=history_tracking)

        self.base_view_models = base_view_models


MVMM.__doc__ = dedent("""\
Multi-view mixture model fit using an EM algorithm.

Parameters
----------
base_view_models: list of mixture models
    Mixture models for each view. These should specify the number of view components.

{em_param_docs}

Attributes
----------

weights_

weights_mat_

metadata_

""".format(**_em_docs))
