import numpy as np
import pandas as pd


def combined_weight_from_sep_view_models(mm_0, mm_1, X,
                                         method='resp_dot'):
    """
    Parameters
    ----------
    mm_0, mm_1:
        Trained mixture models on each view separately.

    X=[X0, X1]:
        The data for views 0 and 1.

    method: str
        How to get the weight matrix. Must be one of

    Output
    ------
    weight: array-like, (n_clust_0, n_clust_1)
        The estimated weight matrix.


    """
    X0, X1 = X

    y0 = mm_0.predict(X0)
    y1 = mm_1.predict(X1)
    w0 = mm_0.weights_
    w1 = mm_1.weights_

    log_prob0 = mm_0.log_probs(X0)
    log_resp_0 = mm_0.log_resps(log_prob0)

    log_prob1 = mm_1.log_probs(X1)
    log_resp_1 = mm_1.log_resps(log_prob1)

    if method == 'combine_resp_dot':
        # dot product of responsibilities
        resp_dot = np.exp(log_resp_0).T.dot(np.exp(log_resp_1))
        resp_dot = resp_dot / resp_dot.sum()
        # resp_dot = reresp_dots_dot.ravel()
        return resp_dot

    elif method == 'combine_indep_weights':
        # outer product of weight vectors (i.e. independence)
        w = np.outer(w0, w1)
        return w

    elif method == 'combine_obs_pred':
        # observed proportions of predictions
        # min_val = 1
        obs_pred = pd.crosstab(y0, y1).values
        # obs_pred = obs_pred + min_val
        obs_pred = obs_pred / obs_pred.sum()
        # obs_pred = obs_pred.ravel()
        return obs_pred

    else:
        raise ValueError("Method must be one of "
                         "['combine_resp_dot', 'combine_indep_weights', 'combine_obs_pred']"
                         "not {}".format(method))
