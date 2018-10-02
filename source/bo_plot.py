# coding: utf-8
#!/usr/bin/env python

# from sklearn.gaussian_process import GaussianProcessRegressor

from matplotlib import pyplot as plt
import numpy as np


# ------------------------------------------------------------------------------
def plot_mean_std(X_, y_mean, y_std, show=True, label = None, color = 'k', linewidth = 3):
    plt.plot(X_, y_mean, color=color, linewidth=linewidth, label = label)
    plt.fill_between(X_, y_mean - y_std, y_mean + y_std,
                     alpha=0.1, color=color, linewidth = 0.0)
    plt.fill_between(X_, y_mean - 2 * y_std, y_mean + 2 * y_std,
                     alpha=0.05, color=color, linewidth = 0.0)
    plt.fill_between(X_, y_mean - 3 * y_std, y_mean + 3 * y_std,
                     alpha=0.02, color=color, linewidth = 0.0)
    if show:
        plt.show()


# -----------------------------------------------------------------------------
def plot_gp(gp, X_, true_function=None, nsamples=0, show=True, label=None):
    """
    Plot a 1D Gaussian Process, with CI and samples
    """
    y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)
    if true_function is not None:
        plt.plot(X_, true_function(X_), 'r--', lw = 3, label=label)

    plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
    plt.fill_between(X_, y_mean - y_std, y_mean + y_std,
                     alpha=0.1, color='k', linewidth = 0.0)
    plt.fill_between(X_, y_mean - 2 * y_std, y_mean + 2 * y_std,
                     alpha=0.05, color='k', linewidth = 0.0)
    plt.fill_between(X_, y_mean - 3 * y_std, y_mean + 3 * y_std,
                     alpha=0.05, color='k', linewidth = 0.0)

    if nsamples > 0:
        y_samples = gp.sample_y(X_[:, np.newaxis], 10)
        plt.plot(X_, y_samples, lw=1)

    plt.plot(gp.X_train_, gp.y_train_, 'ob')
    if show:
        plt.show()


# ------------------------------------------------------------------------------
def plot_1d_strategy(gp, X_, function, nsamples, criterion, next_to_evaluate):
    plt.subplot(2, 1, 1)
    plot_gp(gp, X_, true_function = function, nsamples = nsamples, show = False)
    plt.axvline(next_to_evaluate, ls = '--', color = 'red')
    plt.subplot(2, 1, 2)
    plt.plot(X_, criterion)
    plt.axvline(next_to_evaluate, ls = '--', color = 'red')
    plt.plot(next_to_evaluate, criterion.max(), 'or')
    plt.show()


# ------------------------------------------------------------------------------
def plot_2d_strategy(gp, X_, function, criterion, next_to_evaluate = None,
                     criterion_plottitle = 'EI'):
    """
    plot contourf for a 2d problem
    """
    # X_ must be combination vector obtained by flattening a meshgrid, such that [sqn**2, :]

    if next_to_evaluate is None:
        next_to_evaluate = [np.nan, np.nan]

    sqn = int(np.sqrt(X_.shape[0]))
    vector_to_mesh = X_[:sqn, 1]
    xmesh, ymesh = np.meshgrid(vector_to_mesh, vector_to_mesh, indexing = 'ij')  # Getting back the meshgrid from X_
    y_pred_2d, y_std_2d = np.asarray(gp.predict(X_, return_std = True))
    y_pred_2d = y_pred_2d.reshape(sqn, sqn)
    y_std_2d = y_std_2d.reshape(sqn, sqn)
    true_values = function(X_)

    # ---
    plt.subplot(2, 2, 1)
    plt.contourf(xmesh, ymesh, y_pred_2d)
    plt.colorbar()
    plt.title('GP prediction')
    plt.plot(gp.X_train_[:, 0], gp.X_train_[:, 1], 'ro')
    plt.plot(next_to_evaluate[0], next_to_evaluate[1], '*w')
    # ---
    plt.subplot(2, 2, 2)
    plt.contourf(xmesh, ymesh, y_std_2d)
    plt.title('GP standard deviation')
    plt.plot(next_to_evaluate[0], next_to_evaluate[1], '*w')
    plt.colorbar()
    # ---
    plt.subplot(2, 2, 3)
    plt.contourf(xmesh, ymesh, true_values.reshape(sqn, sqn))
    plt.title('Real function')
    plt.plot(gp.X_train_[:, 0], gp.X_train_[:, 1], 'ro')
    plt.plot(next_to_evaluate[0], next_to_evaluate[1], '*w')
    plt.colorbar()
    # ---
    plt.subplot(2, 2, 4)
    plt.contourf(xmesh, ymesh, criterion.reshape(sqn, sqn))
    plt.plot(next_to_evaluate[0], next_to_evaluate[1], '*w')
    plt.colorbar()
    plt.title(criterion_plottitle)
    plt.show()