from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score, calinski_harabasz_score, \
    davies_bouldin_score
import numpy as np
import pandas as pd

from mvmm.dunn_index import dunn_score


MEASURE_MIN_GOOD = {'aic': True,
                    'bic': True,
                    'silhouette': False,
                    'calinski_harabasz': False,
                    'davies_bouldin': True,
                    'dunn': False}


def get_selected_model(scores):
    selected = {}
    for measure in scores.columns:
        if MEASURE_MIN_GOOD[measure]:
            selected[measure] = scores[measure].idxmin()
        else:
            selected[measure] = scores[measure].idxmax()

    return selected


def unsupervised_cluster_scores(X, estimator,
                                measures=['aic', 'bic',
                                          'silhouette',
                                          'calinski_harabasz',
                                          'davies_bouldin',
                                          'dunn'],
                                measure_kws=None,
                                metric='euclidean',
                                dist_kws={},
                                precomp_dists=None,
                                dunn_kws={'diameter_method': 'farthest',
                                          'cdist_method': 'nearest'}):

    for measure in measures:
        assert measure in ['aic', 'bic', 'silhouette', 'calinski_harabasz',
                           'davies_bouldin', 'dunn']

    scores = {}

    if 'bic' in measures:
        scores['bic'] = estimator.bic(X)

    if 'aic' in measures:
        scores['aic'] = estimator.aic(X)

    # get cluster predictions
    pred_labels = estimator.predict(X)
    n_samples = len(pred_labels)
    n_unique_pred = len(np.unique(pred_labels))

    # this makes the code compatible with multi-views data where
    # X is a list of datasets
    if type(X) == list:
        X = np.hstack(X)

    # possibly precompute pairwise distances
    if precomp_dists is None and \
            ('silhouette' in measures or 'dunn' in measures):
        precomp_dists = pairwise_distances(X=X, metric=metric, **dist_kws)

    if n_unique_pred == 1 or n_unique_pred >= n_samples:

        for x in ['silhouette', 'dunn', 'calinski_harabasz', 'davies_bouldin']:
            if x in measures:
                scores[x] = np.nan

    else:
        if 'silhouette' in measures:
            scores['silhouette'] = silhouette_score(X=precomp_dists,
                                                    labels=pred_labels,
                                                    metric='precomputed')

        if 'dunn' in measures:
            scores['dunn'] = dunn_score(X=precomp_dists, labels=pred_labels,
                                        metric='precomputed')

        if 'calinski_harabasz' in measures:
            scores['calinski_harabasz'] = \
                calinski_harabasz_score(X=X, labels=pred_labels)

        if 'davies_bouldin' in measures:
            scores['davies_bouldin'] = davies_bouldin_score(X=X,
                                                            labels=pred_labels)

    return scores


def several_unsupervised_cluster_scores(X, estimators,
                                        measures=['aic', 'bic',
                                                  'silhouette',
                                                  'calinski_harabasz',
                                                  'davies_bouldin',
                                                  'dunn'],
                                        measure_kws=None,
                                        metric='euclidean',
                                        dist_kws={},
                                        precomp_dists=None,
                                        dunn_kws={'diameter_method':
                                                  'farthest',
                                                  'cdist_method': 'nearest'}):

    if precomp_dists is None and \
            ('silhouette' in measures or 'dunn' in measures):

        precomp_dists = \
            multi_view_safe_pairwise_distances(X=X,
                                               metric=metric, **dist_kws)
    results = []
    for estimator in estimators:

        res = unsupervised_cluster_scores(X=X, estimator=estimator,
                                          measures=measures,
                                          measure_kws=measure_kws,
                                          metric=metric,
                                          dist_kws=dist_kws,
                                          precomp_dists=precomp_dists,
                                          dunn_kws=dunn_kws)

        results.append(res)

    return pd.DataFrame(results)


def multi_view_safe_pairwise_distances(X, **kwargs):
    # this makes the code compatible with multi-views data where
    # X is a list of datasets
    if type(X) == list:
        return pairwise_distances(X=np.hstack(X), **kwargs)
    else:
        return pairwise_distances(X=X, **kwargs)
