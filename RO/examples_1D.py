#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from matplotlib import pyplot as plt
import itertools
from sklearn.gaussian_process.kernels import Matern
from sklearn.cluster import KMeans
import scipy
import sys
import pyDOE
import copy

sys.path.append("/home/victor/RO_VT/RO")
import bo_plot as bplt
import bo_wrapper as bovt
import acquisition_function as acq


if __name__ == '__main__':
    rng = np.random.RandomState()
    X = rng.uniform(0, 1, 5) + [0.0, 1.0, 2.0, 3.0, 4.0]
    X = np.atleast_2d(X).T
    X = pyDOE.lhs(n=1, samples = 5)
    expl_sin = lambda X: np.exp(-np.ravel(X) / 5) * np.sin(np.ravel(X) * 2)
    higdon2002 = lambda X: 0.2 * np.sin(2 * np.pi * np.ravel(X) * 4) \
        + np.sin(2 * np.pi * np.ravel(X))

    # Evaluation
    true_function = higdon2002
    y = true_function(X)

    # Initialization GP
    gp = GaussianProcessRegressor(kernel = Matern(0.1))
    X_ = np.linspace(0, 1, 1000)
    # Fit
    gp.fit(X, y)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    bplt.plot_gp(gp, X_, true_function=true_function, nsamples=5, show=False)
    plt.title('Function to be optimized and kriging prediction')
    plt.plot(np.nan, '--r', label = 'True function')
    plt.plot(np.nan, 'k', label= 'Kriging prediction')
    plt.legend(loc = 'lower left')
    ax2 = fig.add_subplot(212)
    ln1 = ax2.plot(X_, acq.gp_EI_computation(gp, X_.T), 'r', label = 'EI')
    plt.yticks([], [])
    ax2b = ax2.twinx()
    ln2 = ax2b.plot(X_, acq.gp_PI_computation(gp, X_.T), 'b', label = 'PI')
    plt.yticks([], [])
    entropy_test = acq.conditional_entropy(gp, np.linspace(0, 1, 100), X_, M=5, nsamples = 1000)
    ax2c = ax2.twinx()
    ln3 = ax2c.plot(np.linspace(0, 1, 100), -entropy_test, 'm')
    plt.yticks([], [])
    ax2c.plot(np.nan, 'r', label = 'EI')
    ax2c.plot(np.nan, 'b', label = 'PI')
    ax2c.plot(np.nan, 'm', label = '-Conditional entropy')
    plt.legend(loc = 'upper left')
    plt.title('Optimization-oriented acquisition functions')
    plt.tight_layout()
    plt.show()


    # EGO ------------------------------------
    gp_EGO = bovt.EGO_brute(gp_ = gp, X_ = X_, true_function = true_function,
                            niterations = 1, plot = True)
    gp.fit(X, y)  # Reset

    # EGO analytical -------------------------
    gp_analytical = bovt.EGO_analytical(gp, true_function, niterations = 20, X_= X_,
                                        bounds = [(0, 5)], plot = False)
    gp.fit(X, y)  # Reset

    # qEGO ------------------------------------
    gp_qEI = bovt.qEI_brute(gp_ = gp, true_function = true_function, q=2,
                            niterations = 5, nsim = 1000)
    gp.fit(X, y)  # Reset

    # IAGO -----------------------------------
    gp_IAGO = bovt.IAGO_brute(gp, true_function, np.linspace(0, 1, 100), X_,
                              niterations = 5, M = 5, nsamples = 500, plot = True)
    gp.fit(X, y)  # Reset

    # exploEGO -------------------------------
    gp_explo_EGO = bovt.exploEGO(gp, true_function, 0, X_,
                                 bounds = np.array([[0, 1]]))

