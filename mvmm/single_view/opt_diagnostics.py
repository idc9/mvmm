import matplotlib.pyplot as plt
# from copy import deepcopy
import numpy as np

from mvmm.viz_utils import set_xaxis_int_ticks


def plot_opt_hist(loss_vals, init_loss_vals=None,
                  loss_name='loss value',
                  title='', step_vals=None, inches=10):

    loss_val_diffs = np.diff(loss_vals)

    # get signed log of loss val differences
    log_lvd = np.array([np.nan] * len(loss_val_diffs))
    log_lvd[loss_val_diffs > 0] = np.log10(loss_val_diffs[loss_val_diffs > 0])
    log_lvd[loss_val_diffs < 0] = np.log10(-loss_val_diffs[loss_val_diffs < 0])

#     if init_loss_vals is not None:
#         n_plots = 3
#     else:
#         n_plots = 2

    plt.figure(figsize=(2.1 * inches, inches))

    # plot loss val history
    plt.subplot(1, 2, 1)
    plt.plot(loss_vals, marker='.')
    plt.xlabel('step')
    plt.ylabel(loss_name)
    plt.title(title)

    # final initializations
    if init_loss_vals is not None:
        for i, val in enumerate(init_loss_vals):

            label = None
            if i == 0:
                label = 'init std = {:1.3f}'.format(np.std(init_loss_vals))
            plt.axhline(val, lw=.5, alpha=.5, label=label)
        plt.legend()

    if step_vals is not None:
        for s in step_vals:
            plt.axvline(s - 1, color='gray')

    set_xaxis_int_ticks()

    # plot los val differences
    plt.subplot(1, 2, 2)
    plt.plot(log_lvd, marker='.')
    plt.xlabel('step')
    plt.ylabel('log10(diff-{})'.format(loss_name))
    set_xaxis_int_ticks()

    if step_vals is not None:
        for s in step_vals:
            plt.axvline(s - 1, color='gray')

    set_xaxis_int_ticks()
