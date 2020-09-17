import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from numbers import Number
from matplotlib.ticker import MaxNLocator
import seaborn as sns


def draw_ellipse(position, covariance, ax=None, n_sig=3, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    if isinstance(n_sig, Number):
        n_sig = [n_sig]
    # Draw the Ellipse
    for ns in n_sig:
        ax.add_patch(Ellipse(position, ns * width, ns * height,
                             angle, **kwargs))


def savefig(fpath, dpi=100, close=True):
    plt.savefig(fpath, bbox_inches='tight', dpi=dpi)
    if close:
        plt.close()


def safe_heatmap(X, **kws):
    """
    Seaborn heatmap without cutting top/bottom off.
    """
    f, ax = plt.subplots()
    sns.heatmap(X, ax=ax, **kws)
    ax.set_ylim(X.shape[1] + .5, 0)
    ax.set_xlim(0, X.shape[0] + .5)


def set_xaxis_int_ticks():
    """
    Sets integer x ticks
    """
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def set_yaxis_int_ticks():
    """
    Sets integer y ticks
    """
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def axvline_with_tick(x=0, bold=False, **kwargs):
    """
    plt.axvline but atomatically adds tick to x axis

    Parameters
    ----------
    x, **kwargs: see plt.axvline arguments

    bold: bool
        Whether or not to bold the added tick.
    """
    plt.axvline(x=x, **kwargs)
    plt.xticks(list(plt.xticks()[0]) + [x])

    if bold:
        ax = plt.gca()
        ax.get_xticklabels()[-1].set_weight("bold")


def axhline_with_tick(y=0, bold=False, **kwargs):
    """
    plt.axhline but atomatically adds tick to y axis

    Parameters
    ----------
    y, **kwargs: see plt.axhline arguments

    bold: bool
        Whether or not to bold the added tick.
    """
    plt.axhline(y=y, **kwargs)
    plt.yticks(list(plt.yticks()[0]) + [y])

    if bold:
        ax = plt.gca()
        ax.get_xticklabels()[-1].set_weight("bold")


def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def plot_scatter_1d(values, **kwargs):
    """
    Plots 1d scatter plot.

    Parameters
    ----------
    values: array-like, (n, )

    **kwargs:
        key word arguments to plt.scatter
    """
    values = np.array(values).reshape(-1)

    plt.scatter(values, np.zeros(len(values)), **kwargs)

    plt.axhline(0, color='black', zorder=0, alpha=.5)
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
