#!/usr/bin/env python
# coding: utf-8
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from matplotlib import pyplot as plt
import itertools
from sklearn.gaussian_process.kernels import Matern
import scipy
import sys
# import os
import pyDOE
import copy
import warnings

sys.path.append("/home/victor/These/source_code/bayesian_optimization_VT")

import acquisition_function as acq
import bo_plot as bplt


# ------------------------------------------------------------------------------
def transformation_variable_to_unit(X, bounds):
    Y = (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    return Y


# ------------------------------------------------------------------------------
def inv_transformation_variable_to_unit(Y, bounds):
    X = Y * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    return X


# -----------------------------------------------------------------------------
def EGO_brute(gp_, true_function, X_, niterations=10, plot=True):
    """
    EGO performed with brute force: Brute search on vector X_

    Args:
        gp_ (GaussianProcessRegressor): GP of modelling the function to minimize
        X_ ([npoints,nfeatures] array): vector of points on which to compute and
                                        search the maximum of EI
        true_function (func): function to minimize
        niterations (int): number of iterations to perform
        plot (bool): Plot the successive iteration (only for 1D and 2D problems)

    Output:
        GaussianProcessRegressor: GP of the function after the niterations
    """

    print('Resolution of the brute search: ' + str(X_[1] - X_[0]))

    if plot and X_.ndim > 2:
        print('No plot as dim of input > 2')
        plot = False

    gp = copy.copy(gp_)
    i = 0
    while i < niterations:
        print('Iteration {} of {}'.format(str(i + 1), str(niterations)))

        EI_computed = acq.gp_EI_computation(gp, X_)
        next_to_evaluate = acq.acquisition_maxEI_brute(gp, X_)
        value_evaluated = true_function(next_to_evaluate)
        if plot:
            if X_.ndim == 1:
                bplt.plot_1d_strategy(gp, X_, true_function, nsamples = 5,
                                      criterion = EI_computed,
                                      next_to_evaluate = next_to_evaluate)
            elif X_.ndim == 2:
                bplt.plot_2d_strategy(gp, X_, true_function, criterion = EI_computed,
                                      next_to_evaluate = next_to_evaluate)

        X = np.vstack([gp.X_train_, next_to_evaluate])
        # X = X[:,np.newaxis]
        y = np.append(gp.y_train_, value_evaluated)
        gp.fit(X, y)
        i += 1
        print('Best value found so far: {}'.format(str(gp.X_train_[gp.y_train_.argmin()])))

    print('  Best value found after ' + str(niterations) + ' iterations: ' +
          str(gp.X_train_[gp.y_train_.argmin()]))
    return gp


# -----------------------------------------------------------------------------
def EGO_analytical(gp_, true_function, X_ = None, niterations = 10,
                   plot = False, nrestart = 20, bounds = None):
    """
    EGO performed with optimization on the EI

    Args:
        gp_ (GaussianProcessRegressor): GP of modelling the function to minimize
        true_function (func): function to minimize
        niterations (int): number of iterations to perform
        plot (bool): Plot the successive iteration (only for 1D and 2D problems)
        X_ ([npoints,nfeatures] array): vector of points for the plots


    Output:
        GaussianProcessRegressor: GP of the function after the niterations
    """
    # if plot and X_.ndim>2:
    #     print 'No plot as dim of input > 2'
    #     plot = False

    gp = copy.copy(gp_)
    i = 0
    while i < niterations:
        print('Iteration {} of {}'.format(str(i + 1), str(niterations)))
        if X_ is not None:
            EI_computed = acq.gp_EI_computation(gp, X_)

        next_to_evaluate = acq.acquisition_maxEI_analytical_gradientfree(gp, nrestart, bounds)
        print('  Evaluated: ' + str(next_to_evaluate))
        value_evaluated = true_function(next_to_evaluate)

        if plot:
            if X_.ndim == 1:
                bplt.plot_1d_strategy(gp, X_, true_function, nsamples = 5,
                                      criterion = EI_computed,
                                      next_to_evaluate = next_to_evaluate)

            elif X_.ndim == 2:
                bplt.plot_2d_strategy(gp, X_, true_function,
                                      criterion = EI_computed,
                                      next_to_evaluate = next_to_evaluate)

        X = np.vstack([gp.X_train_, next_to_evaluate])
        # X = X[:,np.newaxis]
        y = np.append(gp.y_train_, value_evaluated)
        gp.fit(X, y)
        i += 1

        print('  Best value yet ' + str(gp.X_train_[gp.y_train_.argmin()]))

    print('---  Best value found after ' + str(niterations) + ' iterations: ' +
          str(gp.X_train_[gp.y_train_.argmin()]))
    return gp


# -----------------------------------------------------------------------------
def EGO_LCB(gp_, true_function, kappa, X_ = None, niterations = 10,
            plot = False, nrestart = 20, bounds = None):
    """
    EGO performed with optimization on the EI

    Args:
        gp_ (GaussianProcessRegressor): GP of modelling the function to minimize
        true_function (func): function to minimize
        niterations (int): number of iterations to perform
        plot (bool): Plot the successive iteration (only for 1D and 2D problems)
        X_ ([npoints,nfeatures] array): vector of points for the plots


    Output:
        GaussianProcessRegressor: GP of the function after the niterations
    """
    # if plot and X_.ndim>2:
    #     print 'No plot as dim of input > 2'
    #     plot = False

    gp = copy.copy(gp_)
    i = 0
    while i < niterations:
        print('Iteration {} of {}'.format(str(i + 1), str(niterations)))
        if X_ is not None:
            LCB_computed = acq.gp_LCB(gp, X_, kappa)

        next_to_evaluate = acq.acquisition_LCB(gp, kappa, nrestart, bounds)
        print('   Evaluated: ' + str(next_to_evaluate))
        value_evaluated = true_function(next_to_evaluate)

        if plot:
            if X_.ndim == 1:
                bplt.plot_1d_strategy(gp, X_, true_function, nsamples = 5,
                                      criterion = LCB_computed,
                                      next_to_evaluate = next_to_evaluate)

            elif X_.ndim == 2:
                bplt.plot_2d_strategy(gp, X_, true_function,
                                      criterion = LCB_computed,
                                      next_to_evaluate = next_to_evaluate)

        X = np.vstack([gp.X_train_, next_to_evaluate])
        # X = X[:,np.newaxis]
        y = np.append(gp.y_train_, value_evaluated)
        gp.fit(X, y)
        i += 1
        print('  Best value yet ' + str(gp.X_train_[gp.y_train_.argmin()]))

    print('---  Best value found after ' + str(niterations) + ' iterations: ' +
          str(gp.X_train_[gp.y_train_.argmin()]))
    return gp


# --------------------------------------------------------------------------
def qEI_brute(gp_, true_function, X_=np.linspace(0, 1, 200), q=3,
              niterations=10, nsim=1000):
    """
    q steps EI performed with brute force: Brute search on vector X_
    """
    gp = copy.copy(gp_)
    i = 0
    nn = X_.shape[0]
    rshape = q * [nn]
    qEI_to_evaluate = np.asarray([np.vstack(np.array(comb))
                                  for comb in itertools.product(X_, repeat=q)]).squeeze()

    while i < niterations:
        bplt.plot_gp(gp, X_, true_function=true_function, nsamples=5, show=False)
        qEI_computed = acq.gp_qEI_computation_brute(gp, qEI_to_evaluate, nsim).reshape(rshape)
        next_to_evaluate = X_[np.asarray(np.unravel_index(qEI_computed.argmax(),
                                                          qEI_computed.shape))]
        value_evaluated = true_function(next_to_evaluate)
        [plt.axvline(nextpoint, ls = '--', color = 'red')
         for nextpoint in next_to_evaluate]

        X = np.append(gp.X_train_, next_to_evaluate)
        X = X[:, np.newaxis]
        y = np.append(gp.y_train_, value_evaluated)
        gp.fit(X, y)
        i += 1
        print('  Best value yet ' + str(gp.X_train_[gp.y_train_.argmin()]))

        plt.show()
    return gp


# -----------------------------------------------------------------------------
def IAGO_brute(gp_, true_function, candidates, X_, niterations=10, M=10,
               nsamples = 1000, plot = True):
    """
    Informational Approach to Global Optimization method applied

    Args:
        gp (GaussianProcessRegressor): GP of modelling the function to minimize
        true_function (func): function to minimize
        candidates ([npoints,nfeatures] array): vector of candidates
        X_ ([npoints,nfeatures] array): vector of points for the plots
        niterations (int): number of iterations to perform
        M (int): number of quantiles to compute for the estimation of entropy
        nsamples (int): number of MC samples
                        to estimate distribution of minimizer
        plot (bool): Plot the successive iteration (only for 1D and 2D problems)

    Output:
        GaussianProcessRegressor: GP of the function after the niterations
    """
    gp = copy.copy(gp_)
    i = 0
    while i < niterations:
        print('Iteration {} of {}'.format(str(i + 1), str(niterations)))

        cond_entropy = acq.conditional_entropy(gp, candidates, X_, M, nsamples)
        next_to_evaluate = candidates[cond_entropy.argmin()]
        value_evaluated = true_function(next_to_evaluate)

        if plot:
            if X_.ndim == 1:
                bplt.plot_1d_strategy(gp, candidates, true_function, nsamples = 5,
                                      criterion=-cond_entropy,
                                      next_to_evaluate=next_to_evaluate)
            elif X_.ndim == 2:
                bplt.plot_2d_strategy(gp, X_, true_function,
                                      criterion=cond_entropy,
                                      next_to_evaluate=next_to_evaluate)

        X = np.vstack([gp.X_train_, next_to_evaluate])
        # X = X[:,np.newaxis]
        y = np.append(gp.y_train_, value_evaluated)
        gp.fit(X, y)
        i += 1
        print('Best value found so far ' + str(gp.X_train_[gp.y_train_.argmin()]))

    print('  Best value found after ' + str(niterations) + ' iterations: ' +
          str(gp.X_train_[gp.y_train_.argmin()]))
    return gp


# --------------------------------------------------------------------------
def exploEGO(gp_, true_function, idx_U, X_ = None, niterations = 10,
             plot = False, nrestart = 20, bounds = None):
    """
    exploEGO performed with optimization on the EI

    Args:
        gp_ (GaussianProcessRegressor): GP of modelling the function
                                        to minimize
        true_function (func): function to minimize
        niterations (int): number of iterations to perform
        plot (bool): Plot the successive iteration
                     (only for 1D and 2D problems)
        X_ ([npoints,nfeatures] array): vector of points for the plots


    Output:
        GaussianProcessRegressor: GP of the function after the niterations
    """
    # if plot and X_.ndim>2:
    #     print 'No plot as dim of input > 2'
    #     plot = False

    gp = copy.copy(gp_)
    i = 0
    if idx_U is None:
        print('No index for exploration, classical EGO performed')
    while i < niterations:
        print('Iteration {} of {}'.format(str(i + 1), str(niterations)))


        next_to_evaluate, distance = acq.acquisition_exploEI_analytical(gp, nrestart, bounds, idx_U)
        value_evaluated = true_function(next_to_evaluate)

        if plot:
            EI_computed = acq.gp_EI_computation(gp, X_)

            if X_.ndim == 1:
                bplt.plot_1d_strategy(gp, X_, true_function, nsamples=5,
                                      criterion=EI_computed,
                                      next_to_evaluate=next_to_evaluate)

            elif X_.ndim == 2:
                bplt.plot_2d_strategy(gp, X_, true_function,
                                      criterion=EI_computed,
                                      next_to_evaluate=next_to_evaluate)

        X = np.vstack([gp.X_train_, next_to_evaluate])
        # X = X[:,np.newaxis]
        y = np.append(gp.y_train_, value_evaluated)
        gp.fit(X, y)
        i += 1
        print('  Best value yet ' + str(gp.X_train_[gp.y_train_.argmin()]))
        print('  Maximum distance: ' + str(distance))

    print('---  Best value found after ' + str(niterations) + ' iterations: ' +
          str(gp.X_train_[gp.y_train_.argmin()]))
    return gp


# --------------------------------------------------------------------------
def eval_array_separate_variables(gp, value_to_change, value_to_fix, idx_to_fix):
    _, ndim = gp.X_train_.shape
    npoints, _ = value_to_change.shape
    idx_to_change = filter(lambda i: i in range(ndim) and i not in idx_to_fix, range(ndim))

    if len(idx_to_change) + len(idx_to_fix) != ndim:
        print('Dimensions do not match!')
    # rep_fixed = np.tile(value_to_fix, npoints).reshape(npoints, len(idx_to_fix))
    eval_array = np.zeros([npoints, ndim])
    eval_array[:, idx_to_change] = value_to_change
    eval_array[:, idx_to_fix] = value_to_fix
    return eval_array


# --------------------------------------------------------------------------
def slicer_gp_predict(gp, value_to_fix, idx_to_fix, return_std=True):
    """
    Prediction function of the gp for 'value_to_fix' fixed.
    """
    def fun_to_return(value_to_change):
        evalsep = eval_array_separate_variables(gp, value_to_change, value_to_fix, idx_to_fix)
        return gp.predict(evalsep, return_std=return_std)
    return fun_to_return

# --------------------------------------------------------------------------
def find_minimum_sliced(gp, value_to_fix, idx_to_fix, bounds = None,
                        nrestart = 10, coefficient=1.0):
    """
    For a value 'value_to_fix', that is the argument indexed by 'idx_to_fix',
    Finds minimum of the prediction of the gaussian process when the other
    arguments vary
    """
    fun_ = slicer_gp_predict(gp, value_to_fix, idx_to_fix, return_std=False)
    fun = lambda X_: fun_(np.atleast_2d(X_)) * coefficient
    optim_number = 1
    rng = np.random.RandomState()
    dim = gp.X_train_.shape[1] - len(idx_to_fix)
    if bounds is None:
        bounds = np.asarray(dim * [(0, 1)])

    x0 = np.asarray([rng.uniform(bds[0], bds[1], 1) for bds in bounds]).squeeze()
    current_minimum = scipy.optimize.minimize(fun, x0=x0, bounds=bounds)
    while optim_number <= nrestart:
        x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds]
        if len(x0) == 1:
            [x0] = x0
        else:
            x0 = np.asarray(x0).reshape(-1)
        optim = scipy.optimize.minimize(fun, x0=x0, bounds=bounds)
        if optim.fun < current_minimum.fun:
            current_minimum = optim
        optim_number += 1
    return current_minimum


# --------------------------------------------------------------------------
def PI_alpha_fix(gp, alpha, value_to_fix, idx_to_fix,
                 X_, bounds = None, nrestart = 10):
    minimum_fix = find_minimum_sliced(gp, value_to_fix, idx_to_fix, bounds, nrestart)
    # minimizer = minimum_fix.x
    minimum = minimum_fix.fun
    if X_.ndim == 1:
        X_ = np.atleast_2d(X_).T
    sliced_fun = slicer_gp_predict(gp, value_to_fix, idx_to_fix, return_std = True)
    y_mean, y_std = sliced_fun(X_)
    m = alpha * minimum - y_mean
    s = y_std
    return acq.probability_of_improvement(m, s)


# --------------------------------------------------------------------------
def I_alpha_fix(gp, alpha, value_to_fix, idx_to_fix,
                X_, bounds = None, nrestart = 10):
    minimum_fix = find_minimum_sliced(gp, value_to_fix, idx_to_fix, bounds, nrestart)
    # minimizer = minimum_fix.x
    minimum = minimum_fix.fun
    if X_.ndim == 1:
        X_ = np.atleast_2d(X_).T
    sliced_fun = slicer_gp_predict(gp, value_to_fix, idx_to_fix, return_std = True)
    y_mean, y_std = sliced_fun(X_)
    m = alpha * minimum - y_mean
    return m >= 0


# --------------------------------------------------------------------------
def PI_alpha_allspace(gp, alpha, idx_to_explore, X_to_minimize,
                      X_to_explore, bounds = None, nrestart = 10, PI = True):
    """ alpha-Probability of improvement for each value in X_explore (U)
    """
    total_grid = np.empty([len(X_to_explore), len(X_to_minimize)])
    for index, x_fix in enumerate(X_to_explore):
        if PI:
            total_grid[index, :] = PI_alpha_fix(gp, alpha, x_fix, idx_to_explore,
                                                X_to_minimize, bounds, nrestart)
        else:
            total_grid[index, :] = I_alpha_fix(gp, alpha, x_fix, idx_to_explore,
                                               X_to_minimize, bounds, nrestart)
    return total_grid.mean(0)


# --------------------------------------------------------------------------
def PI_alpha_check_tol(gp, idx_to_explore, X_to_minimize, X_to_explore, ptol = 1.0,
                       bounds = None, nrestart = 10, delta_alpha = 0.01, alpha_start = 1.0,
                       PI = True):
    """ Compute the value of alpha, such that the gp prediction is below
    alpha*minimum  with probability ptol
    """



    if ptol > 1.0:
        print('No solution for ptol > 1, set at 1.0')
        ptol = 1.0
    alpha = alpha_start
    prob_alpha_min = PI_alpha_allspace(gp=gp, alpha=alpha,
                                       idx_to_explore=idx_to_explore,
                                       X_to_minimize=X_to_minimize,
                                       X_to_explore=X_to_explore,
                                       bounds=bounds, nrestart=nrestart, PI=PI)
    while np.sum(prob_alpha_min >= ptol) < 1:  # np.all(prob_alpha_min < ptol):
        alpha += delta_alpha
        prob_alpha_min = PI_alpha_allspace(gp=gp, alpha=alpha,
                                           idx_to_explore=idx_to_explore,
                                           X_to_minimize=X_to_minimize,
                                           X_to_explore=X_to_explore,
                                           bounds=bounds, nrestart=nrestart, PI=PI)
        print('alpha = ' + str(alpha) + ', max prob = ' + str(max(prob_alpha_min)))

    return alpha, X_to_minimize[prob_alpha_min.argmax()]


# --------------------------------------------------------------------------
def PI_alpha_check_tol_dichotomy(gp, idx_to_explore, X_to_minimize, X_to_explore, ptol = 1.0,
                                 bounds = None, nrestart = 10,
                                 alpha_low = 1.0, alpha_up = 10.0, ndichotomy = 10, PI = True):

    # mini, maxi = find_extrema_gp(gp, bounds_allspace, 20)
    # if mini.fun < 0:
    #     print 'Minimum of gp prediction < 0, no alpha can be found'
    #     raise NameError('Stop before infinite loop')
    # else:
    #     print 'alpha < maximum/minimum = ', maxi.fun / mini.fun

    if ptol > 1.0:
        print('No solution for ptol > 1, set at 1.0')
        ptol = 1.0
    prob_alpha_min_low = PI_alpha_allspace(gp=gp, alpha=alpha_low,
                                           idx_to_explore=idx_to_explore,
                                           X_to_minimize=X_to_minimize,
                                           X_to_explore=X_to_explore,
                                           bounds = bounds, nrestart=nrestart, PI=PI)

    prob_alpha_min_up = PI_alpha_allspace(gp=gp, alpha=alpha_up,
                                          idx_to_explore=idx_to_explore,
                                          X_to_minimize=X_to_minimize,
                                          X_to_explore=X_to_explore,
                                          bounds=bounds, nrestart=nrestart, PI=PI)
    nite = 0

    while nite < ndichotomy:
        if np.sum(prob_alpha_min_up >= ptol) >= 1.0:  # np.all(prob_alpha_min < ptol):
            alpha = (alpha_low + alpha_up) / 2.0
            prob_alpha_min = PI_alpha_allspace(gp=gp, alpha=alpha,
                                               idx_to_explore=idx_to_explore,
                                               X_to_minimize=X_to_minimize,
                                               X_to_explore=X_to_explore,
                                               bounds = bounds, nrestart = nrestart)
            if np.sum(prob_alpha_min >= ptol) >= 1.0:
                alpha_up = alpha
                prob_alpha_min_up = prob_alpha_min
            else:
                alpha_low = alpha
                prob_alpha_min_low = prob_alpha_min
        else:
            alpha_up *= 2
            prob_alpha_min_up = PI_alpha_allspace(gp=gp, alpha=alpha_up,
                                                  idx_to_explore=idx_to_explore,
                                                  X_to_minimize=X_to_minimize,
                                                  X_to_explore=X_to_explore,
                                                  bounds = bounds, nrestart = nrestart)
        print('alpha = ' + str(alpha_low) + ', ' + str(alpha_up) +
              ', max prob = ' + str(max(prob_alpha_min_low)) + ', ' +
              str(max(prob_alpha_min_up)))
        nite += 1
    return (alpha_low + alpha_up) / 2.0, X_to_minimize[prob_alpha_min.argmax()]


# --------------------------------------------------------------------------
def find_extrema_gp(gp, bounds, nrestart=20):
    fun = lambda X_: gp.predict(np.atleast_2d(X_))  # Prediction function
    # Minimization
    optim_number = 1
    rng = np.random.RandomState()
    x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds]
    if len(x0) == 1:
        [x0] = x0
    current_minimum = scipy.optimize.minimize(fun, x0=x0, bounds=bounds)
    while optim_number <= nrestart:
        x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds]
        if len(x0) == 1:
            [x0] = x0

        optim = scipy.optimize.minimize(fun, x0=x0, bounds = bounds)
        if optim.fun < current_minimum.fun:
            current_minimum = optim
        optim_number += 1
    # Maximization
    fun_max = lambda X_: -fun(X_)
    optim_number = 1
    x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds]
    if len(x0) == 1:
        [x0] = x0
    current_maximum = scipy.optimize.minimize(fun_max, x0=x0, bounds=bounds)
    while optim_number <= nrestart:
        x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds]
        if len(x0) == 1:
            [x0] = x0
        optim = scipy.optimize.minimize(fun_max, x0=x0, bounds=bounds)
        if optim.fun > current_maximum.fun:
            current_maximum = optim
        optim_number += 1
    return current_minimum, current_maximum


# --------------------------------------------------------------------------
def slicer_gp_sample(gp, value_to_fix, idx_to_fix):
    def fun_to_return(value_to_change, nsamples):
        evalsep = eval_array_separate_variables(gp, value_to_change, value_to_fix, idx_to_fix)
        return gp.sample_y(evalsep, nsamples)
    return fun_to_return


# --------------------------------------------------------------------------
def proj_mean_gp(gp, grid_K, idxU, nsamples=100, bounds=None):
    _, ndim = gp.X_train_.shape
    idxK = filter(lambda i: i in range(ndim) and i not in idxU, range(ndim))
    if bounds is None:
        bounds_U = len(idxU) * [[0, 1]]
    else:
        bounds_U = bounds[[int(x) for x in idxU], :]

    LHS_U = pyDOE.lhs(n=len(idxU), samples=nsamples)
    LHS_U = inv_transformation_variable_to_unit(LHS_U, np.asarray(bounds_U))
    mean_U = np.empty(len(grid_K))
    var_U = np.empty(len(grid_K))
    for i, kk in enumerate(grid_K):
        # array_to_eval = eval_array_separate_variables(gp, value_to_change=LHS_U,
        #                                               value_to_fix=kk, idx_to_fix=idxK)
        array_to_eval = eval_array_separate_variables(gp, value_to_change=LHS_U,
                                                      value_to_fix=np.atleast_2d(kk),
                                                      idx_to_fix=idxK)
        samples = gp.sample_y(array_to_eval, nsamples)
        mean_U[i] = samples.mean()
        var_U[i] = samples.var()
    return mean_U, var_U


# --------------------------------------------------------------------------
def gp_EI_computation_proj(gp, idxU, grid_K, nsamples=500, bounds=None):
    _, ndim = gp.X_train_.shape
    idxK = filter(lambda i: i in range(ndim) and i not in idxU, range(ndim))
    train_K = gp.X_train_[:, idxK]

    mean_train, var_train = proj_mean_gp(gp, train_K, idxU, nsamples, bounds=bounds)
    fmin = mean_train.min()
    mean_U, var_train = proj_mean_gp(gp, grid_K, idxU, nsamples, bounds=bounds)
    return acq.expected_improvement_closed_form(fmin - mean_U, np.sqrt(var_train))


# --------------------------------------------------------------------------
def acquisition_maxEI_proj(gp, idxU, nsamples, nrestart, bounds=None):
    rng = np.random.RandomState()
    _, ndim = gp.X_train_.shape
    idxK = filter(lambda i: i in range(ndim) and i not in idxU, range(ndim))
    if bounds is None:
        bounds = ndim * [(0, 1)]

    def EI_lambda(value):
        return -gp_EI_computation_proj(gp, idxU,
                                       value,
                                       nsamples=nsamples, bounds=bounds)
    bounds_K = bounds[idxK, :]


    optim_number = 1
    x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds_K]

    maxEI = scipy.optimize.minimize(EI_lambda, x0=x0, bounds=bounds_K)

    while optim_number <= nrestart:
        x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds_K]
        optim = scipy.optimize.minimize(EI_lambda, x0=x0, bounds=bounds_K)
        if optim.fun < maxEI.fun:
            maxEI = optim
        optim_number += 1
    return maxEI.x


# --------------------------------------------------------------------------
def k_vector_cov(gp, idx_U, ku_new, x_maxEI, nsamples = 500):
    KU_augmented = np.vstack([gp.X_train_, ku_new])
    kvec = np.empty([gp.X_train_.shape[0] + 1, 1, nsamples])
    for i, urnd in enumerate(np.random.uniform(0, 1, (nsamples, len(idx_U)))):
        kvec[:, :, i] = gp.kernel_(KU_augmented, np.atleast_2d(np.hstack([x_maxEI, urnd])))
    return kvec.mean(axis = 2)


# --------------------------------------------------------------------------
def cov_augmented(gp, ku_new):
    KU_augmented = np.vstack([gp.X_train_, ku_new])
    return gp.kernel_(KU_augmented)


# --------------------------------------------------------------------------
def argmin_VAR_proj(gp, idx_U, x_maxEI, bounds = None, nrestart = 20, nsamples = 500):
    def VAR_lambda(ku_new):
        kvec = k_vector_cov(gp, idx_U, ku_new, x_maxEI, nsamples = 1000)
        Cinv = np.linalg.inv(cov_augmented(gp, ku_new))  # Not optimal
        return -kvec.T.dot(Cinv).dot(kvec)

    optim_number = 1
    rng = np.random.RandomState()
    _, dim = gp.X_train_.shape
    if bounds is None:
        bounds = dim * [(0, 1)]
    x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds]

    minVAR = scipy.optimize.minimize(VAR_lambda, x0=x0, bounds=bounds)

    while optim_number <= nrestart:
        x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds]
        optim = scipy.optimize.minimize(VAR_lambda, x0=x0, bounds=bounds)
        if optim.fun < minVAR.fun:
            minVAR = optim
        optim_number += 1
    return minVAR.x


# --------------------------------------------------------------------------
def acquisition_EI_VAR(gp, idx_U, nrestart, bounds, nsamples=200):
    maxEI = acquisition_maxEI_proj(gp, idx_U, nsamples, nrestart=20, bounds=bounds)
    minVAR = argmin_VAR_proj(gp=gp, idx_U=idx_U, x_maxEI=maxEI,
                             bounds=bounds, nrestart=20, nsamples=nsamples)
    return minVAR, maxEI


# -----------------------------------------------------------------------------
def final_decision_quantile_EIVAR(gp, idx_U, nrestart, bounds, nsamples=200):
    rng = np.random.RandomState()
    _, ndim = gp.X_train_.shape

    def fun_to_minimize(grid_K):
        mean_U, var_U = proj_mean_gp(gp, grid_K, idxU=idx_U, nsamples=nsamples, bounds=bounds)
        return mean_U + 1.282 * np.sqrt(var_U)

    idxK = filter(lambda i: i in range(ndim) and i not in idx_U, range(ndim))
    bounds_K = bounds[idxK, :]
    optim_number = 1
    x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds_K]

    minquantile = scipy.optimize.minimize(fun_to_minimize, x0=x0, bounds=bounds_K)

    while optim_number <= nrestart:
        x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds_K]
        optim = scipy.optimize.minimize(fun_to_minimize, x0=x0, bounds=bounds_K)
        if optim.fun < minquantile.fun:
            minquantile = optim
        optim_number += 1
    return minquantile.x


# --------------------------------------------------------------------------
def EI_VAR(gp_, true_function, idx_U, X_ = None, niterations = 10,
           plot = False, nrestart = 20, bounds = None, nsamples = 200):
    """
    EI VAR algorithm performed with optimization on the EI, and on VAR

    Args:
        gp_ (GaussianProcessRegressor): GP of modelling the function
                                        to minimize
        true_function (func): function to minimize
        idx_U (list of int): indices corresponding to the uncertain variables
        niterations (int): number of iterations to perform
        plot (bool): Plot the successive iteration
                     (only for 1D and 2D problems)
        X_ ([npoints,nfeatures] array): vector of points for the plots


    Output:
        GaussianProcessRegressor: GP of the function after the niterations
    """
    # if plot and X_.ndim>2:
    #     print 'No plot as dim of input > 2'
    #     plot = False

    gp = copy.copy(gp_)
    i = 0
    while i < niterations:
        print('Iteration {} of {}'.format(str(i + 1), str(niterations)))
        next_to_evaluate, maxEI = acquisition_EI_VAR(gp, idx_U, nrestart, bounds, nsamples)
        print('  maxEI = ' + str(maxEI))
        print('  next to evaluate = ' + str(next_to_evaluate))
        value_evaluated = true_function(next_to_evaluate)

        if plot:
            EI_computed = acq.gp_EI_computation(gp, X_)

            if X_.ndim == 1:
                bplt.plot_1d_strategy(gp, X_, true_function, nsamples=5,
                                      criterion=EI_computed,
                                      next_to_evaluate=next_to_evaluate)

            elif X_.ndim == 2:
                bplt.plot_2d_strategy(gp, X_, true_function,
                                      criterion=EI_computed,
                                      next_to_evaluate=next_to_evaluate)

        X = np.vstack([gp.X_train_, next_to_evaluate])
        # X = X[:,np.newaxis]
        y = np.append(gp.y_train_, value_evaluated)
        gp.fit(X, y)
        i += 1
        print('  Best value yet ' + str(gp.X_train_[gp.y_train_.argmin()]))

    print('---  Best value found after ' + str(niterations) + ' iterations: ' +
          str(gp.X_train_[gp.y_train_.argmin()]))

    final_dec = final_decision_quantile_EIVAR(gp, idx_U, nrestart=500,
                                              bounds=bounds, nsamples=nsamples)

    print('--- Final decision' + str(final_dec))
    return gp, final_dec


# --------------------------------------------------------------------------
def LHS_to_eval(npointsLHS, idx_LHS, cst, idx_cst, bounds):
    ndimLHS = len(idx_LHS)
    LHS01 = pyDOE.lhs(ndimLHS, samples=npointsLHS, criterion ='m', iterations=100)
    LHS = inv_transformation_variable_to_unit(LHS01, bounds[idx_LHS, :])
    ndim = len(idx_LHS) + len(idx_cst)
    to_eval = np.zeros([npointsLHS, ndim])
    to_eval[:, idx_LHS] = LHS
    to_eval[:, idx_cst] = cst
    return to_eval


# --------------------------------------------------------------------------
def EI_MC_initial_design(true_function, idx_K, idx_U, ninitial = None,
                         npoints_LHS=30, bounds=None, LHS_K = True):
    ndim = len(idx_K) + len(idx_U)
    if bounds is None:
        bounds = ndim * [(0, 1)]

    if ninitial is None:
        ninitial = 10 * ndim
    if LHS_K:
        LHS_K = inv_transformation_variable_to_unit(pyDOE.lhs(len(idx_K),
                                                              ninitial,
                                                              criterion = 'm',
                                                              iterations=100),
                                                    bounds[idx_K, :])
    else:
        print('Only available for dimK = 1')
        LHS_K = inv_transformation_variable_to_unit(np.linspace(0, 1, ninitial),
                                                    bounds[idx_K, :])

    meanvec = np.zeros(len(LHS_K))
    varvec = np.zeros(len(LHS_K))
    for i, k in enumerate(LHS_K):
        LHS = LHS_to_eval(npoints_LHS, idx_U, k, idx_K, bounds)
        eval_sample = true_function(LHS)
        meanvec[i] = eval_sample.mean()
        varvec[i] = eval_sample.var()
    return LHS_K, meanvec, varvec


# --------------------------------------------------------------------------
def EI_MC(gp_, true_function, idx_U, X_ = None, niterations = 10,
          plot = False, nrestart = 20, bounds = None, nsamples = 200):
    """
    EI VAR algorithm performed with optimization on the EI, and on VAR

    Args:
        gp_ (GaussianProcessRegressor): GP of modelling the function
                                        to minimize
        true_function (func): function to minimize
        idx_U (list of int): indices corresponding to the uncertain variables
        niterations (int): number of iterations to perform
        plot (bool): Plot the successive iteration
                     (only for 1D and 2D problems)
        X_ ([npoints,nfeatures] array): vector of points for the plots


    Output:
        GaussianProcessRegressor: GP of the function after the niterations
    """
    # if plot and X_.ndim>2:
    #     print 'No plot as dim of input > 2'
    #     plot = False

    gp = copy.copy(gp_)
    i = 0
    while i < niterations:
        print('Iteration {} of {}'.format(str(i + 1), str(niterations)))
        next_to_evaluate, maxEI = acquisition_EI_VAR(gp, idx_U, nrestart, bounds, nsamples)
        print('  maxEI = ' + str(maxEI))
        print('  next to evaluate = ' + str(next_to_evaluate))
        value_evaluated = true_function(next_to_evaluate)

        if plot:
            EI_computed = acq.gp_EI_computation(gp, X_)

            if X_.ndim == 1:
                bplt.plot_1d_strategy(gp, X_, true_function, nsamples=5,
                                      criterion=EI_computed,
                                      next_to_evaluate=next_to_evaluate)

            elif X_.ndim == 2:
                bplt.plot_2d_strategy(gp, X_, true_function,
                                      criterion=EI_computed,
                                      next_to_evaluate=next_to_evaluate)

        X = np.vstack([gp.X_train_, next_to_evaluate])
        # X = X[:,np.newaxis]
        y = np.append(gp.y_train_, value_evaluated)
        gp.fit(X, y)
        i += 1
        print('  Best value yet ' + str(gp.X_train_[gp.y_train_.argmin()]))

    print('---  Best value found after ' + str(niterations) + ' iterations: ' +
          str(gp.X_train_[gp.y_train_.argmin()]))

    final_dec = final_decision_quantile_EIVAR(gp, idx_U, nrestart=500,
                                              bounds=bounds, nsamples=nsamples)

    print('--- Final decision' + str(final_dec))
    return gp, final_dec


# --------------------------------------------------------------------------
def gp_worst_case_fixedgrid(gp, idx_K, K_array, bounds=None, full_output=False):
    _, ndim = gp.X_train_.shape
    idx_U = filter(lambda i: i in range(ndim) and i not in idx_K, range(ndim))

    if bounds is None:
        bounds_U = len(idx_U) * [(0, 1)]
    else:
        bounds_U = bounds[idx_U, :]

    if full_output:
        worst_perf = np.zeros(np.asarray(K_array).shape)

    k_current_wc = K_array[0]
    current_minimum = find_minimum_sliced(gp, value_to_fix=K_array[0],
                                          idx_to_fix=idx_K,
                                          bounds=bounds_U,
                                          nrestart=100,
                                          coefficient=-1.0)  # Find maximum on the slice
    print(current_minimum)
    if full_output:
        worst_perf[0] = -current_minimum.fun
    for index, k in enumerate(K_array[1:]):
        minimum_at_k = find_minimum_sliced(gp, value_to_fix=k,
                                           idx_to_fix=idx_K,
                                           bounds=bounds_U,
                                           nrestart=100,
                                           coefficient=-1.0)  # Find maximum on the slice
        print((-minimum_at_k.fun, -current_minimum.fun))
        if full_output:
            worst_perf[index + 1] = -minimum_at_k.fun
        if -minimum_at_k.fun < -current_minimum.fun:
            current_minimum = minimum_at_k
            k_current_wc = k
    if full_output:
        return k_current_wc, -current_minimum.fun, worst_perf
    else:
        return k_current_wc, -current_minimum.fun


# --------------------------------------------------------------------------
def PEI_threshold(gp, u, idxU, boundsK):
<<<<<<< HEAD:bo_wrapper.py
    min_pred = find_minimum_sliced(gp, u, idxU, bounds=boundsK).fun
=======
    min_pred = find_minimum_sliced(gp, u, idxU, bounds=[boundsK.T]).fun
>>>>>>> 22d2c8cf9b675022e083e2b614234891672e65f2:RO/bo_wrapper.py
    # return min_pred
    return max([min_pred, gp.y_train_.min()])


<<<<<<< HEAD
# --------------------------------------------------------------------------
=======
<<<<<<< HEAD:bo_wrapper.py
# --------------------------------------------------------------------------
=======
>>>>>>> 22d2c8cf9b675022e083e2b614234891672e65f2:RO/bo_wrapper.py
>>>>>>> 6ee6b97291dc3792692530c38c2e5de5b72830e9
def PEI_comb(gp, comb, idxU, bounds):
    _, ndim = gp.X_train_.shape
    N = len(comb)
    idxK = filter(lambda i: i in range(ndim) and i not in idxU, range(ndim))
    vals = np.empty(N)
<<<<<<< HEAD:bo_wrapper.py
    # print 'start enum'
    for i, ku in enumerate(comb):
        thres = PEI_threshold(gp, ku[idxU], idxU, bounds[idxK, :])
        # print thres
        fun_ufixed = slicer_gp_predict(gp, ku[idxU], idxU, return_std=True)
        m, s = fun_ufixed(np.atleast_2d(ku[idxK]))
        # print acq.expected_improvement_closed_form(thres - m, s)
=======
    for i, ku in enumerate(comb):
        thres = PEI_threshold(gp, ku[idxU], idxU, bounds[idxK, :])
        fun_ufixed = slicer_gp_predict(gp, ku[idxU], idxU, return_std=True)
        m, s = fun_ufixed(np.atleast_2d(ku[idxK]).T)
>>>>>>> 22d2c8cf9b675022e083e2b614234891672e65f2:RO/bo_wrapper.py
        vals[i] = acq.expected_improvement_closed_form(thres - m, s)
    return vals


<<<<<<< HEAD
# --------------------------------------------------------------------------
=======
<<<<<<< HEAD:bo_wrapper.py
# --------------------------------------------------------------------------
=======
>>>>>>> 22d2c8cf9b675022e083e2b614234891672e65f2:RO/bo_wrapper.py
>>>>>>> 6ee6b97291dc3792692530c38c2e5de5b72830e9
def profilePEI(gp, u, idxU, boundsK):
    thres = PEI_threshold(gp, u, idxU, boundsK)
    fun_ufixed = slicer_gp_predict(gp, u, idxU, return_std=True)

    def PEI_u(k):
        m, s = fun_ufixed(np.atleast_2d(k).T)
        return -acq.expected_improvement_closed_form(m - thres, s)
    x0 = [rng.uniform(bds[0], bds[1], 1) for bds in boundsK]
    current_max = scipy.optimize.minimize(PEI_u, x0, bounds=boundsK)
    for i in xrange(10):
        x0 = [rng.uniform(bds[0], bds[1], 1) for bds in boundsK]
        temp_max = scipy.optimize.minimize(PEI_u, x0, bounds=boundsK)
        if temp_max > current_max:
            current_max = temp_max
    return current_max.x


<<<<<<< HEAD
# --------------------------------------------------------------------------
=======
<<<<<<< HEAD:bo_wrapper.py
# --------------------------------------------------------------------------
=======

>>>>>>> 22d2c8cf9b675022e083e2b614234891672e65f2:RO/bo_wrapper.py
>>>>>>> 6ee6b97291dc3792692530c38c2e5de5b72830e9
def acquisition_PEI_joint(gp, nrestart, idxU, bounds):
    optim_number = 1
    rng = np.random.RandomState()
    dim = gp.X_train_.shape[1]
    if bounds is None:
        bounds = dim * [(0, 1)]

    def fun_to_minimize(ku):
        return -PEI_comb(gp, np.atleast_2d(ku), idxU, bounds)

<<<<<<< HEAD:bo_wrapper.py
    x0 = np.asarray([rng.uniform(bds[0], bds[1], 1) for bds in bounds]).squeeze()
    # print x0, 'ok'
    # print 'b ', fun_to_minimize(x0)

    maxPEI = scipy.optimize.minimize(fun_to_minimize, x0=x0, bounds=bounds)
    print maxPEI
    while optim_number < nrestart:
        print 'loop restart'
=======
    x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds]
    maxPEI = scipy.optimize.minimize(fun_to_minimize, x0=x0, bounds=bounds)
    while optim_number < nrestart:
>>>>>>> 22d2c8cf9b675022e083e2b614234891672e65f2:RO/bo_wrapper.py
        x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds]
        maxtemp = scipy.optimize.minimize(fun_to_minimize, x0=x0, bounds=bounds)
        if maxtemp.fun < maxPEI.fun:
            maxPEI = maxtemp
        optim_number += 1
    return maxPEI.x


<<<<<<< HEAD
# --------------------------------------------------------------------------
def PEI_algo(gp_, true_function, idx_U, X_=None,
             niterations = 10,
=======
<<<<<<< HEAD:bo_wrapper.py
# --------------------------------------------------------------------------
=======



>>>>>>> 22d2c8cf9b675022e083e2b614234891672e65f2:RO/bo_wrapper.py
def PEI_algo(gp_, true_function, idx_U, X_ = None, niterations = 10,
>>>>>>> 6ee6b97291dc3792692530c38c2e5de5b72830e9
             plot = False, nrestart = 20, bounds = None):
    """
    exploEGO performed with optimization on the EI

    Args:
        gp_ (GaussianProcessRegressor): GP of modelling the function
                                        to minimize
        true_function (func): function to minimize
        niterations (int): number of iterations to perform
        plot (bool): Plot the successive iteration
                     (only for 1D and 2D problems)
        X_ ([npoints,nfeatures] array): vector of points for the plots


    Output:
        GaussianProcessRegressor: GP of the function after the niterations
    """
    # if plot and X_.ndim>2:
    #     print 'No plot as dim of input > 2'
    #     plot = False

    gp = copy.copy(gp_)
    i = 0
    while i < niterations:
        print('Iteration {} of {}'.format(str(i + 1), str(niterations)))

        next_to_evaluate = acquisition_PEI_joint(gp, nrestart, idx_U, bounds)
<<<<<<< HEAD
        print('next: {}'.format(next_to_evaluate))
=======
<<<<<<< HEAD:bo_wrapper.py
        print 'acq end'
=======
>>>>>>> 22d2c8cf9b675022e083e2b614234891672e65f2:RO/bo_wrapper.py
>>>>>>> 6ee6b97291dc3792692530c38c2e5de5b72830e9
        value_evaluated = true_function(next_to_evaluate)

        if plot:
            PEI_computed = PEI_comb(gp, X_, idx_U, bounds)

            if X_.ndim == 1:
                bplt.plot_1d_strategy(gp, X_, true_function, nsamples=5,
                                      criterion=PEI_computed,
                                      next_to_evaluate=next_to_evaluate)

            elif X_.ndim == 2:
                bplt.plot_2d_strategy(gp, X_, true_function,
                                      criterion=PEI_computed,
                                      next_to_evaluate=next_to_evaluate)

        X = np.vstack([gp.X_train_, next_to_evaluate])
        # X = X[:,np.newaxis]
        y = np.append(gp.y_train_, value_evaluated)
        gp.fit(X, y)
        i += 1
        print('  Best value yet ' + str(gp.X_train_[gp.y_train_.argmin()]))

    print('---  Best value found after ' + str(niterations) + ' iterations: ' +
          str(gp.X_train_[gp.y_train_.argmin()]))
    return gp


# --------------------------------------------------------------------------
def coverage_probability(gp, threshold, points):
    mean, std = gp.predict(points, return_std=True)
    return scipy.stats.norm.cdf((threshold - mean) / std, loc=0, scale=1)


# --------------------------------------------------------------------------
def alpha_set_quantile(gp, threshold, q, points):
    mean, std = gp.predict(points, return_std=True)
    return mean + scipy.stats.norm.ppf(q) * std <= threshold


# --------------------------------------------------------------------------
def Vorobev_quantile(gp, threshold, qstart, points, tol=1e-5):
    target = np.mean(coverage_probability(gp, threshold, points))
    q = qstart
    ql, qu = 0.0, 1.0
    proposition = np.mean(alpha_set_quantile(gp, threshold, q, points))
    nit = 0
    while (np.abs(proposition - target) > tol and (nit < 50)):
        print(nit, np.abs(proposition - target), q)
        if proposition > target:
            qu = q
            q = (ql + q) / 2.0
            proposition = np.mean(alpha_set_quantile(gp, threshold, q, points))
        else:
            ql = q
            q = (qu + q) / 2.0
            proposition = np.mean(alpha_set_quantile(gp, threshold, q, points))
        nit += 1
    return q


# --------------------------------------------------------------------------
def Vorobev_mean(gp, threshold, qstart, points, tol=1e-5):
    """ Returns indicator function of Vorobev quantile applied to points
    """
    astar = Vorobev_quantile(gp, threshold, qstart, points, tol)
    return alpha_set_quantile(gp, threshold, astar, points)


# --------------------------------------------------------------------------
def Vorobev_deviation(gp, coverage, Vorobev_mean):
    return (np.sum(coverage[Vorobev_mean]) +
            np.sum((1.0 - coverage)[~Vorobev_mean])) / len(coverage)


# --------------------------------------------------------------------------
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
    # entropy_test = acq.conditional_entropy(gp, np.linspace(0, 1, 100), X_, M=10, nsamples = 5000)
    ax2c = ax2.twinx()
    # ln3 = ax2c.plot(np.linspace(0, 1, 100), -entropy_test, 'm')
    plt.yticks([], [])
    ax2c.plot(np.nan, 'r', label = 'EI')
    ax2c.plot(np.nan, 'b', label = 'PI')
    # ax2c.plot(np.nan, 'm', label = '-Conditional entropy')
    plt.legend(loc = 'upper left')
    plt.title('Optimization-oriented acquisition functions')
    plt.tight_layout()
    plt.show()
    # EGO ------------------------------------
    gp_EGO = EGO_brute(gp_ = gp, X_ = X_, true_function = true_function,
                       niterations = 1, plot = True)
    gp.fit(X, y)  # Reset

    # EGO analytical -------------------------
    gp_analytical = EGO_analytical(gp, true_function, niterations = 20, X_= X_,
                                   bounds = [(0, 5)], plot = False)
    gp.fit(X, y)  # Reset

    # qEGO ------------------------------------
    gp_qEI = qEI_brute(gp_ = gp, true_function = true_function, q=2,
                       niterations = 5, nsim = 1000)
    gp.fit(X, y)  # Reset

    # IAGO -----------------------------------
    gp_IAGO = IAGO_brute(gp, true_function, np.linspace(0, 1, 100), X_,
                         niterations = 5, M = 5, nsamples = 500, plot = True)
    gp.fit(X, y)  # Reset

    # exploEGO -------------------------------
    gp_explo_EGO = exploEGO(gp, true_function, 0, X_,
                            bounds = np.array([[0, 1]]))

    # *. 2D test ---------------------------------------------------------------
    def quadratic(X):
        X = np.atleast_2d(X)
        # return (X[:,0] - 2.5)**2 + (X[:,1] - 1)**2
        return 1 + (X[:, 1] - 0.3)**2 + (X[:, 0] - 0.9)**2  # + (X[:,2] - 3)**2

    def branin_2d(X, damp = 1.0, switch = False):
        """ Scaled branin function:
        global minimizers are
        [0.124, 0.818], [0.54277, 0.1513], [0.96133, 0.16466]
        """
        X = np.atleast_2d(X)
        y, x = X[:, 1], X[:, 0]
        if switch:
            x, y = y, x
        x2 = 15 * y
        x1 = 15 * x - 5
        quad = (x2 - (5.1 / (4 * np.pi**2)) * x1**2 + (5 / np.pi) * x1 - 6)**2
        cosi = (10 - (10 / np.pi * 8)) * np.cos(x1) - 44.81
        return (quad + cosi) / (51.95 * damp) + 2.0
    # Xmin1 = [0.124, 0.818]
    # Xmin2 = [0.51277, 0.1513]
    # Xmin3 = [0.96133, 0.16466]

    def two_valleys(X, sigma = 1, rotation_angle = 0):
        X = np.atleast_2d(X) * 2 - 1

        X[:, 0] = np.cos(rotation_angle) * X[:, 0] - np.sin(rotation_angle) * X[:, 1]
        X[:, 1] = np.sin(rotation_angle) * X[:, 0] + np.cos(rotation_angle) * X[:, 1]
        X = (X + 1) / 2
        k = X[:, 0] * 6 - 3
        u = X[:, 1]
        return -u * np.exp(-(k - 1)**2 / sigma**2) \
            - (1 - u) * 1.01 * np.exp(-(k + 1)**2 / sigma**2) \
            + np.exp(-k**2 / sigma**2) + 1 / (sigma**2)

    def gaussian_peaks(X):
        X = np.atleast_2d(X)
        x, y = X[:, 0] * 5, X[:, 1] * 5
        return 0.8 * np.exp(-(((x)**2 + (y)**2) / 3)) \
            + 1.2 * np.exp(-(((y - 2.5)**2) + (x - 2.0)**2) / 1) \
            + np.exp(-(x - 0.5)**2 / 3 - (y - 4)**2 / 2) \
            + 0.8 * np.exp(-(x - 5)**2 / 4 - (y)**2 / 4) \
            + np.exp(-(x - 5)**2 / 4 - (y - 5)**2 / 4) \
            + (1 / (1 + x + y)) / 25  # + 50 * np.exp((-(y - 2.5)**2 + -(x - 5)**2) / 2)
    function_2d = lambda X: two_valleys(X, 1)  # , np.pi / 4)


    function_2d = branin_2d
    # function_2d = rosenbrock_general
    rng = np.random.RandomState()
    ndim = 2
    bounds = np.asarray([[0, 1], [0, 1]])

    # initial_design_2d = np.array([[1,1],[2,2],[3,3],[4,4], [5,2], [1,4],[0,0],[5,5], [4,1]])/5.0
    initial_design_2d = pyDOE.lhs(n=2, samples=10, criterion='maximin',
                                  iterations=50)
    response_2d = function_2d(initial_design_2d)
    gp = GaussianProcessRegressor(kernel = Matern(np.ones(ndim) / 5.0), n_restarts_optimizer=50)
    gp.fit(initial_design_2d, response_2d)

    X_ = np.linspace(0, 1, 100)
    xx, yy = np.meshgrid(X_, X_, indexing = 'ij')
    all_combinations = np.array([xx, yy]).T.reshape(-1, 2, order = 'F')
    EI_criterion = acq.gp_EI_computation(gp, all_combinations)
    T = 1.5
    bplt.plot_2d_strategy(gp, all_combinations, function_2d, Vorobev_mean(gp, T, 0.5, all_combinations))

    # cond_entropy_2d = acq.conditional_entropy(gp, all_combinations,
    #                                           all_combinations, M=5,
    #                                           nsamples=100)

    bplt.plot_2d_strategy(gp, all_combinations, function_2d, EI_criterion)
    # bplt.plot_2d_strategy(gp, all_combinations, function_2d, -cond_entropy_2d)
    mean_U, var_U = proj_mean_gp(gp, grid_K=np.arange(0, 1, 0.01),
                                 idxU=[1], nsamples = 1000, bounds = bounds)
    plt.plot(mean_U)
    plt.plot(mean_U + np.sqrt(var_U))
    plt.plot(mean_U - np.sqrt(var_U))
    plt.show()

    PEI_joint = PEI_algo(gp, function_2d,
                         idx_U=[1],
                         nrestart=10,
                         X_=all_combinations,
                         niterations=3,
                         plot=False,
                         bounds=np.asarray(bounds))
    PEI_crit = PEI_comb(PEI_joint, all_combinations, [1], np.asarray(bounds))
    bplt.plot_2d_strategy(PEI_joint, all_combinations, function_2d,
                          PEI_crit, criterion_plottitle = 'PEI',
                          show = True, cond_min = True)


    # EGO brute ------------------------------
    gp_brute = EGO_brute(gp, function_2d, all_combinations, niterations=100, plot=False)
    EI_criterion_brute = acq.gp_EI_computation(gp_brute, all_combinations)
    bplt.plot_2d_strategy(gp_brute, all_combinations, function_2d, EI_criterion_brute)


    # EGO analytical -------------------------
    gp_analytical = EGO_analytical(gp, function_2d, X_ = all_combinations, niterations = 100,
                                   plot = False, nrestart = 30, bounds = [(0, 1)] * 2)
    EI_criterion_analytical = acq.gp_EI_computation(gp_analytical, all_combinations)
    bplt.plot_2d_strategy(gp_analytical, all_combinations, function_2d, EI_criterion_analytical)


    # Explo EGO ------------------------------
    gp_explo_EGO = exploEGO(gp, function_2d, idx_U = [1], X_= all_combinations,
                            niterations = 50, plot = False, nrestart = 50,
                            bounds = np.array([[0, 1], [0, 1]]))
    EI_criterion_analytical = acq.gp_EI_computation(gp_explo_EGO, all_combinations)
    bplt.plot_2d_strategy(gp_explo_EGO, all_combinations, function_2d, EI_criterion_analytical)

    # EI VAR -----------------------------------
    gp_EIVAR = EI_VAR(gp, function_2d, idx_U=[1], X_=all_combinations, niterations=3,
                      nrestart=10, bounds=None, nsamples=20)

    # IAGO -------------------------------------
    X_reduced = np.linspace(0, 1, 10)
    xxr, yyr = np.meshgrid(X_, X_, indexing = 'ij')
    all_combinations_reduced = np.array([xxr, yyr]).T.reshape(-1, 2, order = 'F')
    gp_IAGO = IAGO_brute(gp, function_2d,
                         candidates=all_combinations_reduced,
                         X_=all_combinations_reduced,
                         niterations = 5, M = 3, nsamples = 100, plot = True)

    # Worst-case ------------------------------
    k_wc, _ = gp_worst_case_fixedgrid(gp, [0], X_, bounds=None)
    bplt.plot_2d_strategy(gp, all_combinations, function_2d, EI_criterion_analytical)


    # Slicer Test -----------------------------------------------------
    y_2d_pred, y_2d_std = np.asarray(gp.predict(all_combinations, return_std = True))
    plt.contourf(xx, yy, y_2d_pred.reshape(200, 200))
    plt.colorbar()
    plt.subplot(1, 2, 2)
    gppred = slicer_gp_predict(gp, [0.5], [0])
    idx = 1
    y_mean05, y_std05 = gppred(np.atleast_2d(X_).T)

    alpha_1, k_check_1 = PI_alpha_check_tol(gp, [idx], X_, X_, delta_alpha = 0.05, ptol = 1.0)
    alpha_99, k_check_99 = PI_alpha_check_tol(gp, [idx], X_, X_, delta_alpha = 0.05, ptol = 0.99)
    alpha_95, k_check_95 = PI_alpha_check_tol(gp, [idx], X_, X_, delta_alpha = 0.05, ptol = 0.95)
    alpha_90, k_check_90 = PI_alpha_check_tol(gp, [idx], X_, X_, delta_alpha = 0.05, ptol = 0.9)
    print(alpha_1, k_check_1)
    print alpha_99, k_check_99
    print alpha_95, k_check_95
    print alpha_90, k_check_90

    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, y_2d_pred.reshape(200, 200))
    plt.axvline(k_check_1, ls = '--')
    plt.axvline(k_check_99, ls = '--')
    plt.axvline(k_check_95, ls = '--')
    plt.axvline(k_check_90, ls = '--')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.plot(X_, PI_alpha_allspace(gp, 1., [idx], X_, X_))
    plt.plot(X_, PI_alpha_allspace(gp, 1.5, [idx], X_, X_))
    plt.plot(X_, PI_alpha_allspace(gp, 1.8, [idx], X_, X_))
    plt.plot(X_, PI_alpha_allspace(gp, alpha_1, [idx], X_, X_))
    plt.plot(X_, PI_alpha_allspace(gp, alpha_99, [idx], X_, X_))
    plt.plot(X_, PI_alpha_allspace(gp, alpha_95, [idx], X_, X_))
    plt.plot(X_, PI_alpha_allspace(gp, alpha_90, [idx], X_, X_))
    plt.axhline(1.00, ls ='--', color = 'black')
    plt.axhline(0.99, ls ='--', color = 'black', alpha = 0.8)
    plt.axhline(0.95, ls ='--', color = 'black', alpha = 0.6)
    plt.axhline(0.90, ls ='--', color = 'black', alpha = 0.4)
    plt.title('idx of U: ' + str(idx))
    plt.tight_layout()
    plt.show()

    gppred = slicer_gp_predict(gp, [k_check_1], [0])
    y_meancheck, y_stdcheck = gppred(np.atleast_2d(X_).T)
    bplt.plot_mean_std(X_, y_meancheck, y_stdcheck, show = False, label = 'check')

    gppred = slicer_gp_predict(gp, [k_check_95], [0])
    y_meancheck, y_stdcheck = gppred(np.atleast_2d(X_).T)
    bplt.plot_mean_std(X_, y_meancheck, y_stdcheck, show = False, label = 'check95', color = 'g')

    gppred = slicer_gp_predict(gp, [0], [0])
    y_meancheck, y_stdcheck = gppred(np.atleast_2d(X_).T)
    bplt.plot_mean_std(X_, y_meancheck, y_stdcheck, show = False, label = '0', color='b')

    gppred = slicer_gp_predict(gp, [0.2], [0])
    y_meancheck, y_stdcheck = gppred(np.atleast_2d(X_).T)
    bplt.plot_mean_std(X_, y_meancheck, y_stdcheck, show = False, label = '0.7', color = 'm')
    plt.legend()
    plt.xlabel('u')
    plt.show()





    # Ndimensional Test ---------------------------------------------------------
    def rosenbrock_general(X):
        X = np.atleast_2d(X)
        X = X * 15 - 5
        return np.sum(100.0 * (X[:, 1:] - X[:, :-1]**2.0)**2.0 + (1 - X[:, :-1])**2.0, 1)

    NDIM = 10

    initial_design_4d = pyDOE.lhs(n=NDIM, samples=10 * NDIM,
                                  criterion='maximin', iterations=50)
    response_4d = rosenbrock_general(initial_design_4d)
    gp4d = GaussianProcessRegressor(kernel = Matern(np.ones(NDIM) / 5))
    gp4d.fit(initial_design_4d, response_4d)

    gp_analytical_4d = EGO_analytical(gp4d, rosenbrock_general,
                                      X_ = initial_design_4d, niterations = 10,
                                      plot = False, nrestart=50,
                                      bounds = [(0, 1)] * NDIM)
    gp_EIVAR = EI_VAR(gp4d, rosenbrock_general, idx_U=[8, 9], X_=None, niterations=5,
                      nrestart=2, bounds=None, nsamples=2)
    
    bounds = np.asarray([(0, 1)] * NDIM)
    gp_PEI = PEI_algo(gp4d, rosenbrock_general, idx_U=[8, 9], X_=None,
                      niterations=10, plot=False, nrestart=20, bounds=bounds)
