import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numbers import Number
import matplotlib.gridspec as gridspec
from copy import deepcopy

from mvmm.viz_utils import set_xaxis_int_ticks, plot_scatter_1d


def plot_est_params(gmm, n_sigma=3):
    """
    Plots the estimated means and covariances.

    Parameters
    ----------
    gmm:
        A fit gaussian mixture model

    n_sigma: int/float
        Nubmer of standard deviations to show for the variance.
    """
    pal = sns.color_palette('Set2', gmm.n_components)

    # TODO-FEAT: add weights
    for k in range(gmm.n_components):
        color = pal[k]

        # show mean
        mean = gmm.means_[k]
        std = np.sqrt(gmm.covariances_[k].item())
        weight = gmm.weights_[k]

        label = '{}: m={:1.2f}, std={:1.2f}, weight={:1.2f}'.\
            format(k, mean.item(), std, weight)

        plot_scatter_1d(mean, marker='x', s=500, color=color)

        # show cov interbal
        plt.axvspan(mean - n_sigma * std,
                    mean + n_sigma * std,
                    color=color, alpha=.1, zorder=0,
                    label=label)
        plt.legend()


def plot_param_history(history, key='loss_val',
                       true_params=None, vert=False, figsize=(21, 5)):
    """
    Plots the parameter history curves.
    """
    means, covs, weights = get_param_hist(history)

    n_components = means.shape[1]

    def plot_true_params(true_params, who):
        if true_params is None:
            return

        values = true_params[who]
        if isinstance(values, Number):
            values = np.array([values])

        for val in values.ravel():
            plt.axhline(val, color='black', alpha=.5)

    pallette = sns.color_palette("Set2", n_components)

    if vert:
        grid = gridspec.GridSpec(4, 1)
    else:
        grid = gridspec.GridSpec(1, 4)

    def get_grid(k):
        if vert:
            return grid[k, 0]
        else:
            return grid[0, k]

    if figsize is not None:
        plt.figure(figsize=figsize)

    plt.subplot(get_grid(0))
    plot_loss_val_history(history, key=key)

    # plt.subplot(3, 1, 1)
    plt.subplot(get_grid(1))
    for k in range(n_components):
        plt.plot(means[:, k], marker='.', color=pallette[k])
    plot_true_params(true_params, 'means')
    plt.xlabel('step')
    plt.ylabel('cluster mean')
    set_xaxis_int_ticks()

    # plt.subplot(3, 1, 2)
    plt.subplot(get_grid(2))
    for k in range(n_components):
        std = np.sqrt(covs[:, k]).reshape(-1)
        plt.plot(std, marker='.', color=pallette[k])
    plot_true_params(true_params, 'sigma')
    plt.xlabel('step')
    plt.ylabel('cluster std')
    set_xaxis_int_ticks()

    # plt.subplot(3, 1, 3)
    plt.subplot(get_grid(3))
    for k in range(n_components):
        plt.plot(weights[:, k], marker='.', color=pallette[k])

    plot_true_params(true_params, 'pi')
    plt.xlabel('step')
    plt.ylabel('cluster weights')
    set_xaxis_int_ticks()


def plot_loss_val_history(history, key='loss_val'):
    """
    Plots the observed log likelihood for each EM step.
    """
    loss_vals = deepcopy(history[key])
    if np.isinf(loss_vals[0]):
        loss_vals = loss_vals[1:]

    plt.plot(loss_vals, marker='.')
    plt.xlabel('step')
    plt.xlim(0)
    plt.ylabel(key)
    set_xaxis_int_ticks()


def get_param_hist(history):
    """
    Returns the parameter history for a GMM.

    Output
    ------
    means, covs, weights

    means: (n_steps, n_components, n_features)

    covs: (n_steps, n_components, n_features, n_features)

    weights: (n_steps, n_components)
    """
    means = []
    covs = []
    weights = []

    for model in history['model']:
        # means.append(model.means_)
        # covs.append(model.covariances_)
        # weights.append(model.weights_)
        means.append(model['means'])
        covs.append(model['covariances'])
        weights.append(model['weights'])

    return np.array(means), np.array(covs), np.array(weights)
