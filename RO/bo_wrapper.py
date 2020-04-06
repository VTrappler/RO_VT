from __future__ import print_function

#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from matplotlib import pyplot as plt
import itertools
from sklearn.gaussian_process.kernels import Matern
from sklearn.cluster import KMeans
import scipy
import pyDOE
import copy
import warnings


import RO.acquisition_function as acq
import RO.bo_plot as bplt


# ------------------------------------------------------------------------------
def transformation_variable_to_unit(X, bounds):
    """bounds -> [0, 1] affine transformation

    :param X: numpy array
    :param bounds: dim N x 2
    :returns: scaled same as X
    :rtype:

    """
    Y = (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    return Y


# ------------------------------------------------------------------------------
def inv_transformation_variable_to_unit(Y, bounds):
    """[0, 1] -> boubds affine transformation

    :param Y: numpy array
    :param bounds: dim N x 2
    :returns: scaled same as X
    :rtype:

    """
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

    print('Resolution of the brute search: {}'.format(X_[1] - X_[0]))

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
        print('Best value found so far: {}'.format(gp.X_train_[gp.y_train_.argmin()]))


    print('  Best value found after {} iterations: {}'.format(niterations,
                                                              gp.X_train_[gp.y_train_.argmin()]))
    return gp


def template(gp_, true_function, acquisition_fun, criterion_fun, prefix, X_=None, niterations=10,
             plot=False, nrestart = 20, bounds = None, save=False):
    """
    template to perform an optimisation of an acquisition function,
     and to update the gp with the new point evaluated

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
    try:
        gp = copy.copy(gp_)
        i = 0
        while i < niterations:
            print('Iteration {} of {}'.format(str(i + 1), str(niterations)))
            if X_ is not None:
                criterion = criterion_fun(gp, X_)

            next_to_evaluate = acquisition_fun(gp)
            print('  Evaluated: ' + str(next_to_evaluate))
            value_evaluated = true_function(next_to_evaluate)

            if plot:
                if X_.ndim == 1:
                    bplt.plot_1d_strategy(gp, X_, true_function, nsamples = 5,
                                          criterion = criterion,
                                          next_to_evaluate = next_to_evaluate)

                elif X_.ndim == 2:
                    bplt.plot_2d_strategy(gp, X_, true_function,
                                          criterion = criterion,
                                          next_to_evaluate=next_to_evaluate,
                                          show=False,
                                          criterion_plottitle=prefix)
                    if save:
                        fn = '/home/victor/Bureau/tmp/' + prefix + '_{:02d}.png'.format(i)
                        plt.tight_layout()
                        plt.savefig(fn, transparent=False)
                        plt.close()

            X = np.vstack([gp.X_train_, next_to_evaluate])
            # X = X[:,np.newaxis]
            y = np.append(gp.y_train_, value_evaluated)
            gp.fit(X, y)
            i += 1

            print('  Best value yet ' + str(gp.X_train_[gp.y_train_.argmin()]))

        print('  Best value found after {} iterations: {}'.format(niterations,
                                                                  gp.X_train_[gp.y_train_.argmin()]))
        return gp
    except KeyboardInterrupt:
        return gp


# -----------------------------------------------------------------------------
def EGO_analytical(gp_, true_function, X_ = None, niterations = 10,
                   plot = False, nrestart = 20, bounds = None, save=False):
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
                                      next_to_evaluate = next_to_evaluate,
                                      show=False)
                if save:
                    fn = '/home/victor/Bureau/tmp/EI_{:02d}.png'.format(i)
                    plt.tight_layout()
                    plt.savefig(fn, transparent=False)
                    plt.close()

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
    """
    Build an array, accordingly to the gp dimensions, where we can choose
    which values we can set to fixed values
    """
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
def find_minimum_sliced(gp, value_to_fix, idx_to_fix, bounds=None,
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
    """ Find the extrema of the prediction of the gp
    """
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
    """ Returns a function that sample the sliced gp
    """
    def fun_to_return(value_to_change, nsamples):
        evalsep = eval_array_separate_variables(gp, value_to_change, value_to_fix, idx_to_fix)
        return gp.sample_y(evalsep, nsamples)
    return fun_to_return


# --------------------------------------------------------------------------
def proj_mean_gp(gp, grid_K, idxU, nsamples=100, bounds=None):
    """   
    """
    
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
    """Builds a LHS of npointsLHS, and puts it at the right coordinates
    

    :param npointsLHS: int
    :param idx_LHS: index where to put the LHS (its len indicates dimension of LHS)
    :param cst: values to be fixed
    :param idx_cst: index of the constant variables
    :param bounds: bounds for the LHS
    :returns: numpy array of dimension (npointsLHS, len(idx_LHS) + len(idx_cst))
    :rtype: numpy array

    """
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
    EI MC algorithm performed with optimization on the EI

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
    """find the minimum of the prediction for u fixed
    and compares it with the current evaluated minimum.
    It serves as the threshold for the PEI criterion

    :param gp: GaussianProcessRegressor
    :param u: value that is fixed
    :param idxU: index of u
    :param boundsK: bounds upon which the minimization is performed
    :returns: threshold for PEI criterion
    :rtype: float

    """
    min_pred = find_minimum_sliced(gp, u, idxU, bounds=[boundsK.T]).fun
    # return min_pred
    return max([min_pred, gp.y_train_.min()])


# --------------------------------------------------------------------------
def PEI_comb(gp, comb, idxU, bounds):
    """Computes the PEI criterion for the selected points and gp

    :param gp: GaussianProcessRegressor
    :param comb: points to eval, of dimension npoints x kudimension
    :param idxU: index of U variables
    :param bounds: bounds for the minimization
    :returns: numpy array of the evaluated PEI

    """
    _, ndim = gp.X_train_.shape
    N = len(comb)
    idxK = filter(lambda i: i in range(ndim) and i not in idxU, range(ndim))
    vals = np.empty(N)
    for i, ku in enumerate(comb):
        thres = PEI_threshold(gp, ku[idxU], idxU, bounds[idxK, :])
        fun_ufixed = slicer_gp_predict(gp, ku[idxU], idxU, return_std=True)
        m, s = fun_ufixed(np.atleast_2d(ku[idxK]).T)
        vals[i] = acq.expected_improvement_closed_form(thres - m, s)
    return vals


# --------------------------------------------------------------------------
def profilePEI(gp, u, idxU, boundsK):
    warnings.warn('May be removed bc seems unused', FutureWarning)
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


# --------------------------------------------------------------------------
def acquisition_PEI_joint(gp, nrestart, idxU, bounds):
    """Maximize the PEI criterion on the gp

    :param gp: gp to study
    :param nrestart: number of optimization restarts
    :param idxU: index of U variables
    :param bounds: bounds of the minimization
    :returns: the maximizer of the PEI
    :rtype: numpy.array

    """
    optim_number = 1
    rng = np.random.RandomState()
    dim = gp.X_train_.shape[1]
    if bounds is None:
        bounds = dim * [(0, 1)]

    def fun_to_minimize(ku):
        return -PEI_comb(gp, np.atleast_2d(ku), idxU, bounds)

    x0 = np.asarray([rng.uniform(bds[0], bds[1], 1) for bds in bounds]).squeeze()
    # print x0, 'ok'
    # print 'b ', fun_to_minimize(x0)
    maxPEI = scipy.optimize.minimize(fun_to_minimize, x0=x0, bounds=bounds)

    while optim_number < nrestart:
        x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds]
        maxtemp = scipy.optimize.minimize(fun_to_minimize, x0=x0, bounds=bounds)
        if maxtemp.fun < maxPEI.fun:
            maxPEI = maxtemp
        optim_number += 1
    return maxPEI.x


# --------------------------------------------------------------------------
def PEI_algo(gp_, true_function, idx_U, X_=None,
             niterations=10, plot=False, nrestart=20, bounds=None, save=False):
    """
    PEI performed with optimization on the EI

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
        print('next: {}'.format(next_to_evaluate))


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
                                      next_to_evaluate=next_to_evaluate,
                                      show=False, criterion_plottitle='PEI')

                if save:
                    fn = '/home/victor/Bureau/tmp/PEI_{:02d}.png'.format(i)
                    plt.tight_layout()
                    plt.savefig(fn, transparent=False)
                    plt.close()

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
def coverage_probability(arg, threshold, points):
    """returns probability of being lower than threshold

    :param arg: GaussianProcessRegressor or tuple of numpy.arrays
    :param threshold: float
    :param points: points to be evaluated by the GaussianProcessRegressor
    :returns: probability at the points
    :rtype: float

    """
    if isinstance(arg, tuple):
        m, s = arg
    else:
        m, s = gp.predict(points, return_std=True)
    return scipy.stats.norm.cdf((threshold - m) / s, loc=0, scale=1)


def alpha_set_quantile(arg, threshold, q, points):
    if isinstance(arg, tuple):
        m, s = arg
    else:
        m, s = arg.predict(points, return_std=True)
    return (m + scipy.stats.norm.ppf(q) * s <= threshold)


# --------------------------------------------------------------------------
def margin_indicator(gp, threshold, alpha_r, points):
    """
    Compute indicator function for the predicted mean to be in M_{1-alpha_r}
    """
    if gp:
        Fp1 = alpha_set_quantile(gp, threshold, alpha_r, points)
        Fm1 = alpha_set_quantile(gp, threshold, 1 - alpha_r, points)
    else:
        Fp1 = alpha_set_quantile(gp, threshold, alpha_r, points)
    return Fm1 * (1 - Fp1)



def prob_less(m, s, alpha, points, prob):
    k = scipy.stats.norm.ppf(prob)
    threshold = 0
    return scipy.stats.norm.cdf((threshold + k * s - m) / s)


# --------------------------------------------------------------------------
def margin_probability(gp, threshold, points, eta):
    """ Computes the coverage probability of the margin of uncertainty.
        By default the margin is taken with threshold 0

    :param gp: GaussianProcessRegressor
    :param threshold: level set if not 0
    :param points: points at which it should be evaluated
    :param eta: probability associated with MoU
    :returns: probability of coverage of MoU(eta)
    :rtype: numpy array of same dimension as points

    """
    if threshold is None:
        threshold = 0.0
    mean, std = gp.predict(points, return_std=True)
    k = scipy.stats.norm.ppf(eta)
    return scipy.stats.norm.cdf((threshold + k * std - mean) / std) \
        - scipy.stats.norm.cdf((threshold - k * std - mean) / std)


# --------------------------------------------------------------------------
def margin_probability_ms(m, s, points, prob):
    """Computes the coverage probability of the margin of uncertainty.
    

    :param m: 
    :param s: 
    :param points: 
    :param prob: 
    :returns: 
    :rtype: 

    """
    threshold = 0
    k = scipy.stats.norm.ppf(prob)
    return scipy.stats.norm.cdf((threshold + k * s - m) / s) \
        - scipy.stats.norm.cdf((threshold - k * s - m) / s)


# --------------------------------------------------------------------------
def Vorobev_quantile(gp, threshold, qstart, points, tol=1e-5, verbose=False):
    target = np.mean(coverage_probability(gp, threshold, points))
    q = qstart
    ql, qu = 0.0, 1.0
    proposition = np.mean(alpha_set_quantile(gp, threshold, q, points))
    nit = 0
    while (np.abs(proposition - target) > tol and (nit < 50)):
        if verbose:
            print('it: {}, abs dist: {}, q: {}'.format(nit, np.abs(proposition - target), q))
        if proposition <= target:
            qu = q
            q = (ql + q) / 2.0
            proposition = np.mean(alpha_set_quantile(gp, threshold, q, points))
        else:
            ql = q
            q = (qu + q) / 2.0
            proposition = np.mean(alpha_set_quantile(gp, threshold, q, points))
        nit += 1
    if verbose:
        print('alpha star = {}'.format(q))
    return q


# --------------------------------------------------------------------------
def Vorobev_mean(gp, threshold, qstart, points, tol=1e-5, verbose=False):
    """ Returns indicator function of Vorobev quantile applied to points
    """
    astar = Vorobev_quantile(gp, threshold, qstart, points, tol, verbose=verbose)
    print(astar)
    return alpha_set_quantile(gp, threshold, astar, points)


# --------------------------------------------------------------------------
def Vorobev_deviation(gp, coverage, Vorobev_mean):
    return (np.sum(coverage[Vorobev_mean]) +
            np.sum((1.0 - coverage)[~Vorobev_mean])) / len(coverage)


# --------------------------------------------------------------------------
def Vorobev_deviation_threshold(gp, T, qstart, points):
    coverage_prob = coverage_probability(gp, T, points)
    vor_mean = Vorobev_mean(gp, T, qstart, points, verbose=False)
    return Vorobev_deviation(gp, coverage_prob, vor_mean)


# --------------------------------------------------------------------------
def expected_Vorobev_deviation_criterion(gp, T, qstart, candidates, all_points):
    H = np.empty(len(candidates))
    vor_mean = Vorobev_mean(gp, T, qstart, all_points)
    for i, candidate in enumerate(candidates):
        H[i] = 0
        y_values = acq.compute_yvalues_quantiles(gp, candidate, M=4)
        for j, yval in enumerate(y_values):
            gp_candidate = acq.gp_fitted_candidate(gp, candidate, yval)
            coverage_added = coverage_probability(gp_candidate, T, all_points)
            H[i] += Vorobev_deviation(gp_candidate, coverage_added, vor_mean)
            # H[i] += Vorobev_deviation_threshold(gp_candidate, T, qstart, all_points)
        H[i] /= len(y_values)
    return H


# --------------------------------------------------------------------------
def mu_sigma_star(gp, u, idxU, boundsK):
    """ Minimizes the gp prediction, and returns

    :param gp: 
    :param k: 
    :param u: 
    :param idxU: 
    :param boundsK: 

    """
    # obsolete ?
    warnings.warn('Function is not used  and may be removed', FutureWarning)
    minim = find_minimum_sliced(gp, u, idxU, bounds=[boundsK.T])
    fun_ufixed = slicer_gp_predict(gp, u, idxU, return_std=True)
    m_star, sig_star = fun_ufixed(np.atleast_2d(minim.x))
    return m_star, sig_star


# --------------------------------------------------------------------------
def mean_covariance_alpha(gp, k, u, idxU, boundsK):
    """ Minimizes the gp prediction, and returns [m(k,u), m*(u)] and the covariance matrix
    """
    evalsep = eval_array_separate_variables(gp, k, u, idxU)
    minim = find_minimum_sliced(gp, u, idxU, bounds=[boundsK.T])
    evalstar = eval_array_separate_variables(gp, np.atleast_2d(minim.x), u, idxU)
    pred, co = gp.predict(np.vstack([evalsep, evalstar]), return_cov=True)
    return pred, co


# --------------------------------------------------------------------------
def mean_covariance_alpha_vector(gp, kuv, idxU, boundsK):
    res = []
    _, ndim = gp.X_train_.shape
    idxK = filter(lambda i: i in range(ndim) and i not in idxU, range(ndim))
    for i, ku in enumerate(kuv):
        res.append(mean_covariance_alpha(gp,
                                         np.atleast_2d(ku[idxK]),
                                         np.atleast_2d(ku[idxU]),
                                         idxU,
                                         boundsK))
    return res


# --------------------------------------------------------------------------
def pi_alpha(mean_cov_list_of_tuples, alpha):
    """ Computes the probability of coverage of J - alpha J^* < 0

    :param mean_cov_list_of_tuples:
    :param alpha:

    """
    pi = np.empty(len(mean_cov_list_of_tuples))
    for i, mc in enumerate(mean_cov_list_of_tuples):
        m, cov = mc
        one_alpha = np.asarray([1, -alpha])
        mu_D = one_alpha.dot(m)
        sigsqr_D = one_alpha.dot(cov.dot(one_alpha.T))
        pi[i] = scipy.stats.norm.cdf(-mu_D / np.sqrt(sigsqr_D))
    return pi


# --------------------------------------------------------------------------
def estimate_alpha_p(gp, grid, p, idxU, boundsK, alpha_lb, alpha_ub):
    """ Estimation of alpha_p by dichotomy using a gp
    Probabilistic approach
    """
    al = (alpha_lb + alpha_ub) / 2.0
    al_lb = alpha_lb
    al_ub = alpha_ub
    mean_cov_list_of_tuples = mean_covariance_alpha_vector(gp, grid, idxU, boundsK)
    ngrid = int(np.sqrt(len(grid)))
    i = 0
    prob = 0
    while (al_ub - al_lb) > 1e-7 and np.abs(prob - p) > 1e-7:
        prob = pi_alpha(mean_cov_list_of_tuples, al).reshape(ngrid, ngrid).mean(1).max()
        print(al, prob)
        if prob < p:
            (al_lb, al) = (al, (al + al_ub) / 2.0)
        else:
            al, al_ub = ((al + al_lb) / 2.0, al)
        i += 1
    return 0.5 * (al_lb + al_ub)


def estimate_alpha_quantiles(gp, grid, p, idxU, boundsK):
    """Estimate alpha solely on the prediction of the gp
    Plug-in approach

    :param gp: GaussianProcessRegressor
    :param grid: grid used to evaluate the prediction
    :param p: probability
    :param idxU: index of U variables
    :param boundsK: bounds of K for the minimization
    :returns:
    :rtype:
    """
    mdelta = [tu[0] for tu in mean_covariance_alpha_vector(gp, grid, idxU, boundsK)]

    def delta(alpha):
        return [mms[0] - alpha * mms[1] for mms in mdelta]

    ratio = np.asarray([mms[0] / mms[1] for mms in mdelta])

    return np.quantile(ratio.reshape(25, 25), p, axis=1).min()


# --------------------------------------------------------------------------
def m_s_delta(gp, k, u, alpha, idxU, boundsK):
    """Computes the mean and variance of the gaussian process constructed upon
       {Y - alpha Y*}, with Y* ~ N(m*,s2*) and m*(u) = min_k m(k,u)
       for K and U as POINTS !!

    :param gp: GaussianProcessRegressor
    :param k: points at which it should be evaluated
    :param u: points at which it should be evaluated
    :param alpha: relaxation value
    :param idxU: index of u variable
    :param boundsK: bounds for the minimization
    :returns: m_{Y - alpha Y*}, s^2_{Y - alpha Y*}
    :rtype: tuple of floats

    """
    pred, co = mean_covariance_alpha(gp, k, u, idxU, boundsK)
    one_alpha = np.asarray([1, -alpha])
    mu_D = one_alpha.dot(pred)
    sig_D = one_alpha.dot(co.dot(one_alpha.T))
    return mu_D, sig_D


def mu_sigma_delta(gp, vec_ku, alpha, idxU, boundsK):
    """ Computes the mean and variance of the gaussian process constructed upon
       D ={Y - alpha Y*}, with Y* ~ N(m*,s2*) and m*(u) = min_k m(k,u)

    :param gp: GaussianProcessRegressor
    :param vec_ku: points where it should be evaluated
    :param alpha: relaxation value
    :param idxU: index of u variables
    :param boundsK: bounds for the minimisation
    :returns: mean and variance of Delta
    :rtype: tuple of numpy.arrays

    """
    
    md, sdsquare = np.empty(len(vec_ku)), np.empty(len(vec_ku))
    _, ndim = gp.X_train_.shape
    idxK = filter(lambda i: i in range(ndim) and i not in idxU, range(ndim))
    for i, ku in enumerate(vec_ku):
        md[i], sdsquare[i] = m_s_delta(gp,
                                       np.atleast_2d(ku[idxK]),
                                       np.atleast_2d(ku[idxU]),
                                       alpha,
                                       idxU, boundsK)
    return md, np.sqrt(sdsquare)


def sample_from_criterion(Nsamples, criterion, bounds, Ncandidates):
    """ Rejection method to generate Nsamples according to a probability distribution
        proportional to the criterion
    """
    const = np.prod(np.diff(bounds))
    samples = np.empty((0, 2))
    while len(samples) < Nsamples:
            candidates = (scipy.stats.uniform
                          .rvs(size=Ncandidates * len(bounds))
                          .reshape(Ncandidates, (bounds)))
            u = scipy.stats.uniform.rvs(size=Ncandidates)
            acc = (u <= criterion(candidates) / const)
            if np.any(acc):
                samples = np.vstack([samples,
                                     np.atleast_2d([candidates[i]
                                                    for i in range(Ncandidates) if acc[i]])])
    if len(samples) > Nsamples:
        samples = samples[:Nsamples, :]
    return samples


def cluster_and_find_closest(Nclusters, samples):
    """From the samples, classify them in Nclusters, and returns the closest sample from
    the cluster centroids

    :param Nclusters: Number of clusters
    :param samples: All the samples available
    :returns:
    :rtype:

    """
    kmean = KMeans(n_clusters=Nclusters).fit(samples)
    closest = np.empty((Nclusters, 2))
    for i, kmcenter in enumerate(kmean.cluster_centers_):
        closest[i] = samples[np.sum((samples - kmcenter)**2, 1).argmin()]
    return closest, kmean


def add_points_to_design(gp, points_to_add, evaluated_points, optimize_cov=False):
    """Concatenate points to the design and their evaluation by the underlying function.

    :param gp: GaussianProcessRegressor
    :param points_to_add: Points to add to the design
    :param evaluated_points: Evaluated points of the design to be added
    :param optimize_cov: if True, a new ML estimation for the kernels parameter will be achieved
    :returns: The fitted gaussian process
    :rtype: GaussianProcessRegressor

    """
    gpp = copy.deepcopy(gp)
    X = np.vstack([gp.X_train_, points_to_add])
    # X = X[:,np.newaxis]
    y = np.append(gp.y_train_, evaluated_points)
    if not optimize_cov:
        gpp.optimizer = None
    gpp.fit(X, y)
    return gpp


def integrated_variance(gp, integration_points, alpha=1.8):
    """Computes the IMSE (Integrated Mean Square Error) of the gp (eventually with
    the relaxation alpha). Computed by summing the prediction variance at the integrated points

    :param gp: GaussianProcessRegressor
    :param integration_points: numpy.array
    :param alpha: Relaxation (float). If None, IMSE of the original gp
    :returns: IMSE
    :rtype: float

    """
    if alpha is not None:
        m, s = mu_sigma_delta(gp, integration_points, alpha, [1], np.asarray([0, 1]))  #
    else:
        m, s = gp.predict(integration_points, return_std=True)
    return np.mean(s**2)


def augmented_IMSE(gp, candidate_points, integration_points, scenarios=None, alpha=1.8):
    """IMSE of the gaussian process when a candidate point is added
    Adding a relaxation is possible when setting alpha to a float >= 1.0
    The covariance parameters of the gp are NOT optimized for each candidate

    :param gp: GaussianProcessRegressor
    :param candidate_points: design points to consider
    :param integration_points: points used to integrate the MSE
    :param scenarios: callable of the mean and variance of the GP
        that returns the set of possible values for Y(candidate)
    :param alpha: relaxation (if None, no relaxation)
    :returns: IMSE when candidate_points are added
    :rtype: float or numpy.array

    """
    if scenarios is None:
        scenarios = lambda mp, sp: scipy.stats.norm.ppf(np.linspace(0.05, 0.95, 5, endpoint=True),
                                                        loc=mp, scale=sp)

    IMSE = np.empty(len(np.atleast_2d(candidate_points)))
    for i, pt in enumerate(candidate_points):
        if alpha is not None:
            mp, sp = mu_sigma_delta(gp, np.atleast_2d(pt), alpha, [1], np.asarray([0, 1]))
        else:
            mp, sp = gp.predict(np.atleast_2d(pt), return_std=True)
        evaluated_points = scenarios(mp, sp)
        sum_ = 0
        for cand_val in evaluated_points:
            int_var = integrated_variance(add_points_to_design(gp, pt, cand_val,
                                                               optimize_cov=False),
                                          integration_points, alpha)
            sum_ += int_var
        IMSE[i] = sum_ / len(evaluated_points)
    return IMSE

# xmg, ymg = np.meshgrid(*(2 * (np.linspace(0, 1, 10), )))
# IMSE = augmented_IMSE(gp, all_combinations, all_combinations, alpha=None)



def acquisition_vpi(gp, alpha, nrestart=3, bounds=np.asarray([[0, 1], [0, 1]])):
    """Maximizes the variance of the probability of coverage of the alpha acceptable
    region

    :param gp:
    :param alpha:
    :param nrestart:
    :param bounds:
    :returns:
    :rtype:
    """
    optim_number = 0

    def fun_to_minimize(ku):
        return variance_of_prob_coverage(gp, alpha, ku)

    x0 = np.asarray([rng.uniform(bds[0], bds[1], 1) for bds in bounds]).squeeze()
    # print x0, 'ok'
    # print 'b ', fun_to_minimize(x0)
    minIMSE = scipy.optimize.minimize(fun_to_minimize, x0=x0, bounds=bounds,
                                      options={'maxiter': 5})
    while optim_number < nrestart:
        x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds]
        mintemp = scipy.optimize.minimize(fun_to_minimize, x0=x0, bounds=bounds,
                                          options={'maxiter': 5})
        if mintemp.fun < minIMSE.fun:
            minIMSE = mintemp
        optim_number += 1
    return minIMSE.x


ngrid = 50
X_ = np.linspace(0, 1, ngrid)
xx, yy = np.meshgrid(X_, X_, indexing = 'ij')
all_combinations = np.array([xx, yy]).T.reshape(-1, 2, order = 'F')


def acquisition_IMSE(gp, alpha, nrestart=3,
                     integration_points=all_combinations,
                     bounds=np.asarray([[0, 1],
                                        [0, 1]])):
    """Minimizes the augmented IMSE

    :param gp: GaussianProcessRegressor
    :param alpha: Relaxation parameter
    :param nrestart: number of restarts for the optim
    :param integration_points: points for the integration of the MSE
    :param bounds: bounds for the minimization
     :returns: minimizer of the augmented IMSE
    :rtype:

    """
    optim_number = 0

    def fun_to_minimize(ku):
        return augmented_IMSE(gp, np.atleast_2d(ku), integration_points, scenarios=None,
                              alpha=alpha)

    x0 = np.asarray([rng.uniform(bds[0], bds[1], 1) for bds in bounds]).squeeze()
    # print x0, 'ok'
    # print 'b ', fun_to_minimize(x0)
    minIMSE = scipy.optimize.minimize(fun_to_minimize, x0=x0, bounds=bounds,
                                      options={'maxiter': 5})
    while optim_number < nrestart:
        x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds]
        mintemp = scipy.optimize.minimize(fun_to_minimize, x0=x0, bounds=bounds,
                                          options={'maxiter': 5})
        if mintemp.fun < minIMSE.fun:
            minIMSE = mintemp
        optim_number += 1
    return minIMSE.x


def acquisition_IMSE_at_k(gp, kfix, alpha=None, nrestart=3,
                          bounds=np.asarray([[0, 1]])):
    """Minimizes the augmented IMSE along the axis k = kfix

    :param gp: GaussianProcessRegressor
    :param kfix:
    :param alpha:
    :param nrestart:
    :param bounds:
    :returns:
    :rtype

    """
    optim_number = 0

    def fun_to_minimize(u):
        k = kfix * np.ones_like(u)
        vec_eval = np.vstack([k, u]).T
        return augmented_IMSE(gp, np.atleast_2d(vec_eval), all_combinations, alpha=alpha)

    x0 = np.asarray([rng.uniform(bds[0], bds[1], 1) for bds in bounds]).squeeze()
    # print x0, 'ok'
    # print 'b ', fun_to_minimize(x0)
    minIMSE = scipy.optimize.minimize(fun_to_minimize, x0=x0, bounds=bounds)

    while optim_number < nrestart:
        x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds]
        mintemp = scipy.optimize.minimize(fun_to_minimize, x0=x0, bounds=bounds)
        if mintemp.fun < minIMSE.fun:
            minIMSE = mintemp
        optim_number += 1
    return minIMSE.x


# -------------------------------------------------------------------------
if __name__ == '__main__':
    # Setting of the functions and initial design -------------------------
    from RO.test_functions import branin_2d
    function_2d = lambda X: branin_2d(X, switch=False)
    rng = np.random.RandomState()
    ndim = 2
    bounds = np.asarray([[0, 1], [0, 1]])
    # initial_design_2d = np.array([[1,1],[2,2],[3,3],[4,4], [5,2], [1,4],[0,0],[5,5], [4,1]])/5.0
    initial_design_2d = pyDOE.lhs(n=2, samples=30,
                                  criterion='maximin',
                                  iterations=50)
    response_2d = function_2d(initial_design_2d)

    # Fitting of the GaussianProcess -------------------------------------
    gp = GaussianProcessRegressor(kernel=Matern(np.ones(ndim) / 5.0),
                                  n_restarts_optimizer=50)
    gp.fit(initial_design_2d, response_2d)

    # Builds a regular grid ---------------------------------------------
    ngrid = 25
    X_ = np.linspace(0, 1, ngrid)
    xx, yy = np.meshgrid(X_, X_, indexing = 'ij')
    all_combinations = np.array([xx, yy]).T.reshape(-1, 2, order = 'F')

    EI_criterion = acq.gp_EI_computation(gp, all_combinations)
    T = 1.5
    bplt.plot_2d_strategy(gp, all_combinations, function_2d,
                          Vorobev_mean(gp, T, 0.5, all_combinations))

    aa = gp.sample_y(all_combinations, n_samples=9)
    p = [0.95, .99, 1.0]
    Nsamples = 1000
    alpha_p_samples = np.empty((Nsamples, 3))
    # for j, aa in enumerate(sample_y_modified(gp, all_combinations, Nsamples)):
    #     print '{}\r'.format(j),
    #     curr = aa.reshape(ngrid, ngrid)
    #     Jstar = np.asarray([curr[curr.argmin(0)[i], i] for i in xrange(len(X_))])
    #     rho = (curr / Jstar[np.newaxis, :])
    #     alpha_p = np.quantile(rho, p, axis=0).min(1)
    #     alpha_p_samples[j, :] = alpha_p

    nan = alpha_p_samples[:, 0] <= 1.0
    alpha_p_samples = alpha_p_samples[~nan, :]
    alpha_p_samples.mean(0)
    
    plt.hist(alpha_p_samples[:, 0])
    plt.hist(alpha_p_samples[:, 1])
    plt.hist(alpha_p_samples[:, 2])
    plt.show()

    plt.style.use('ggplot')
    plt.subplot(2, 2, 1)
    plt.plot(Js.mean(0), X_)
    plt.subplot(2, 2, 2)
    plt.contourf(Js.T)
    plt.subplot(2, 2, 4)
    plt.plot(X_, Js.mean(1))
    plt.show()
    
    mp = margin_probability(gp, T, all_combinations, 1 - 0.025)
    vmp = mp * (1 - mp)
    bplt.plot_2d_strategy(gp, all_combinations, function_2d,
                          mp)
    plt.subplot(1, 2, 1)
    plt.plot(np.sum(vmp.reshape(ngrid, ngrid).T, 1), range(ngrid))
    plt.subplot(1, 2, 2)
    plt.contourf(vmp.reshape(ngrid, ngrid).T)
    plt.show()

    alpha_est = estimate_alpha_p(gp, all_combinations, 0.99, [1], np.asarray([0, 1]), 1.5, 5.0)
    m, s = mu_sigma_delta(gp, all_combinations, alpha_est, [1], np.asarray([0, 1]))
    ppi = coverage_probability((m, s), 0, None).reshape(ngrid, ngrid)
    va = ppi * (1 - ppi)
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(3, 1, 1)
    ax.contourf((ppi * (1 - ppi)).reshape(ngrid, ngrid).T)
    ax.set_title(r'Contour plot of the coverage variance')
    ax = plt.subplot(3, 1, 2)
    ax.plot(ppi.reshape(ngrid, ngrid).mean(1))
    ax.set_title(r'mean coverage probability wrt $k$, upper and lower prob bound')
    beta = 0.05
    a = prob_less(m, s, alpha_est, all_combinations, beta / 2.0)
    b = prob_less(m, s, alpha_est, all_combinations, 1 - beta / 2)
    ax.plot(a.reshape(ngrid, ngrid).mean(1))
    ax.plot(b.reshape(ngrid, ngrid).mean(1))
    ax = plt.subplot(3, 1, 3)
    ax.plot((ppi * (1 - ppi)).reshape(ngrid, ngrid).mean(1), color='red')
    ax.set_title(r'mean of coverage variance')
    plt.tight_layout()
    plt.show()

    acqIMSE = []
    acqIMSE_alpha = []
    for i in X_:
        print(i)
        acqIMSE.append(acquisition_IMSE_at_k(gp, np.atleast_2d(i), nrestart=20))
        # print('-')
        # acqIMSE_alpha.append(acquisition_IMSE_at_k(gp, np.atleast_2d(i), alpha=2.0, nrestart=4))

    plt.plot(X_, np.asarray(acqIMSE))
    # plt.plot(X_, np.asarray(acqIMSE_alpha))
    plt.show()


    marg_prob = margin_probability_ms(m, s, all_combinations, 1 - 0.05)
    global suf
    suf = 0

    def gamma_hat(k, gp, samples):
        if samples is None:
            samples = np.linspace(0, 1, 50, endpoint=True)

    def pi(gp, alpha, X):
        m, s = mu_sigma_delta(gp, X, alpha, [1], np.asarray([0, 1]))
        ppi = coverage_probability((m, s), 0, None)
        return ppi

    def variance_of_prob_coverage(gp, alpha, X):
        ppi = pi(gp, alpha, np.atleast_2d(X))
        return ppi * (1 - ppi)


    all_combinations_small = np.array(np.meshgrid(np.linspace(0, 1, 10),
                                                  np.linspace(0, 1, 10))).T.reshape(-1, 2)



    def iteration_step(gp, p, al_lb, al_ub, all_combinations):
        alpha_est = estimate_alpha_p(gp, all_combinations, p, [1], np.asarray([0, 1]),
                                     al_lb,
                                     al_ub)
        alpha_est_q = estimate_alpha_quantiles(gp, all_combinations,
                                               p, [1], np.asarray([0, 1]))

        print('==> alpha_est: {}, {}'.format(alpha_est, alpha_est_q))
        
        ngrid = int(np.sqrt(len(all_combinations)))
        m, s = mu_sigma_delta(gp, all_combinations, alpha_est_q, [1], np.asarray([0, 1]))
        ppi = coverage_probability((m, s), 0, None).reshape(ngrid, ngrid)
        # gamma_hat = ppi.reshape(ngrid, ngrid).mean(1)
        (ppi * (1 - ppi)).reshape(ngrid, ngrid).mean(1)
        kfix = X_[(ppi * (1 - ppi)).reshape(ngrid, ngrid).mean(1).argmax()]
        plt.figure(figsize = (8, 10))
        plt.subplot(3, 1, 1)
        plt.contourf((ppi * (1 - ppi)).reshape(ngrid, ngrid).T)
        plt.subplot(3, 1, 2)
        plt.plot(np.log(ppi.reshape(ngrid, ngrid).mean(1)))
        plt.axvline(ppi.reshape(ngrid, ngrid).mean(1).argmax())
        beta = 0.05
        a = prob_less(m, s, alpha_est_q, all_combinations, beta / 2.0)
        b = prob_less(m, s, alpha_est_q, all_combinations, 1 - beta / 2)
        plt.plot(np.log(a.reshape(ngrid, ngrid).mean(1)))
        plt.plot(np.log(b.reshape(ngrid, ngrid).mean(1)))
        plt.subplot(3, 1, 3)
        global suf
        plt.plot((ppi * (1 - ppi)).reshape(ngrid, ngrid).mean(1), color='red')
        plt.title('alpha: {}'.format(alpha_est_q))
        plt.tight_layout()
        plt.savefig('/home/victor/Bureau/tmp/d_imse_{:02d}.png'.format(suf))
        plt.close()
        with open('/home/victor/Bureau/tmp/alpha_est_new.txt', 'a+') as f:
            f.write('{}, {}, {}, {}\n'.format(suf,
                                          alpha_est, alpha_est_q,
                                          ppi.reshape(ngrid, ngrid).mean(1).argmax()))
        suf += 1
        print('k with highest cov variance: {}'.format(kfix))
        global al_est
        al_est = alpha_est_q
        ku = acquisition_IMSE(gp, alpha=alpha_est_q,
                              nrestart=2,
                              integration_points=all_combinations_small,
                              bounds = np.asarray([[0, 1],
                                                   [0, 1]]))
        print('ku with max variance: {}'.format(ku))
        # ku = []
        # for i in (kfix, u):
        #     try:
        #         ku.append(i[0])
        #     except:
        #         ku.append(i)
        return ku

    # ku_to_eval = iteration_step(gp, 0.99, 1.5, 5.0, all_combinations)
    
    # gp2 = add_points_to_design(gp, ku , function_2d(ku), optimize_cov=True)


    def plot_contour_probability_coverage(gp, alpha, eta=0.05, points=all_combinations,
                                          Uidx=[1], boundsU=np.asarray([0, 1])):
        m, s = mu_sigma_delta(gp, points, alpha, [1], np.asarray([0, 1]))
        ppi = coverage_probability((m, s), 0, None).reshape(50, 50).T
        fig, ax = plt.subplots()
        CS = ax.contourf(ppi)
        CS2 = ax.contour(CS, levels = [eta / 2, 0.5, 1 - eta / 2], colors=['w', 'r', 'w'])
        ax.contour(Vorobev_mean((m, s), 0, 0.5, points).reshape(50, 50).T)
        plt.show()



    global suf
    global al_est
    al_est=2.0
    suf = 0
    temp = template(gp, function_2d, lambda gp: iteration_step(gp, 0.99,
                                                               1.5,
                                                               5.0,
                                                               all_combinations),
                    criterion_fun=lambda gp, X: augmented_IMSE(gp, X, all_combinations_small, None, alpha=al_est),
                    prefix='_imse_alpha',
                    X_=all_combinations,
                    niterations=40,
                    plot=True,
                    bounds=np.asarray([[0, 1], [0, 1]]),
                    nrestart=2, save=True)


    al = []
    with open('/home/victor/Bureau/tmp/alpha_est.txt', 'r') as fi:
        firead = csv.reader(fi)
        for i in firead:
            al.append(float(i[1]))
    plt.plot(al)
    plt.show()

            
    ng = 100
    ls_ = np.linspace(0, 1, ng)
    xx, yy = np.meshgrid(ls_, ls_, indexing = 'ij')
    grid_to_estimate_alpha = np.array([xx, yy]).T.reshape(-1, 2, order = 'F')
    p = 0.99
    pi_noswitch = []
    alpha_est_vec = []
    beta = 0.025
    gp_samples = copy.copy(gp)
    N_to_evaluate = 10
    computed = 0
    for i in range(10):
        print(i, computed, beta)
        alpha_est = estimate_alpha_p(gp_samples, grid_to_estimate_alpha,
                                     p, [1],
                                     np.asarray([0, 1]), 1.5, 10.0)
        # alpha_est = 1.9
        print('--> alpha_est = {}'.format(alpha_est))

        def marg_crit(ku, gpp):
            m, s = mu_sigma_delta(gpp, ku, alpha_est, [1], np.asarray([0, 1]))
            return margin_probability_ms(m, s, ku, 1 - beta)

        plt.figure()
        m, s = mu_sigma_delta(gp_samples, all_combinations, alpha_est, [1], np.asarray([0, 1]))
        marg_prob = margin_probability_ms(m, s, all_combinations, 1 - beta)
        pi = coverage_probability((m, s), 0, None)
        pi_noswitch.append([(pi, alpha_est)])

        bplt.plot_2d_strategy(gp_samples, all_combinations, function_2d,
                              marg_prob, show=False,
                              criterion_plottitle=r'Margin Probability {}%'.format(100*(1 - beta)))
        to_add, _ = cluster_and_find_closest(N_to_evaluate,
                                             sample_from_criterion(2000,
                                                                   lambda ku: marg_crit(ku,
                                                                                        gp_samples),
                                                                   bounds, 10))
        gp_samples = add_points_to_design(gp_samples, to_add, function_2d(to_add),
                                          optimize_cov=True)
        computed += N_to_evaluate
        # beta += 0.01
        alpha_est_vec.append((alpha_est, computed))
        plt.tight_layout()
        plt.savefig('/home/victor/Bureau/tmp/pei_samples_' + str(i) + '.png')
        # plt.show()


    beta = 0.05
    a = prob_less(m, s, alpha_est, all_combinations, beta / 2.0)
    b = prob_less(m, s, alpha_est, all_combinations, 1 - beta / 2)
    plt.subplot(3, 1, 1)
    plt.plot(a.reshape(50, 50).mean(1))
    plt.plot(b.reshape(50, 50).mean(1))
    plt.plot(ppi.reshape(50, 50).mean(1), label='prob cov')
    plt.subplot(3, 1, 2)
    plt.plot((b - a).reshape(50, 50).mean(1), label=r'$\delta$')
    plt.subplot(3, 1, 3)
    plt.plot((ppi * (1 - ppi)).reshape(50, 50).mean(1), label=r'Var of coverage')
    plt.legend()
    plt.show()


    def margin_probability_enrichment(true_function, gp_, bounds, T, Niterations=10, nrestart=20):
        i = 0
        gp = copy.copy(gp_)
        while i < Niterations:
            def to_opt(x):
                m, s = mu_sigma_delta(gp, np.atleast_2d(x), 1.8, [1], np.asarray([0, 1]))

                # mp = margin_probability(gp, T, np.atleast_2d(x), 1 - 0.025)
                pi_alpha = coverage_probability((m, s), 0, None)
                return -(pi_alpha) * (1 - pi_alpha)

            print('Iter: {}'.format(i))
            optim_number = 1
            x0 = np.asarray([rng.uniform(bds[0], bds[1], 1) for bds in bounds]).squeeze()
            # print('Start: {}'.format(x0))
            current_minimum = scipy.optimize.minimize(to_opt, x0=x0, bounds=bounds)
            while optim_number <= nrestart:
                # print('Current best min: {} at {}'.format(current_minimum.fun,
                #                                           current_minimum.x))
                x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds]
                if len(x0) == 1:
                    [x0] = x0
                # print('Restart: {}'.format(x0))
                optim = scipy.optimize.minimize(to_opt, x0=x0, bounds=bounds)
                # print('computed min: {}'.format(optim.fun))
                if optim.fun < current_minimum.fun:
                    current_minimum = optim
                optim_number += 1

            next_to_evaluate = optim.x
            value_evaluated = true_function(next_to_evaluate)
            X = np.vstack([gp.X_train_, next_to_evaluate])
            y = np.append(gp.y_train_, value_evaluated)
            gp.fit(X, y)
            i += 1
        return gp


    def level_set_Vorobev(true_function, gp_, bounds, T, Niterations=10):
        i = 0
        gp = copy.copy(gp_)
        while i < Niterations:
            to_opt = lambda x: expected_Vorobev_deviation_criterion(gp,
                                                                    T,
                                                                    0.5,
                                                                    np.atleast_2d(x),
                                                                    all_combinations)
            print('Iter: {}'.format(i))
            nrestart = 1
            optim_number = 1
            x0 = np.asarray([rng.uniform(bds[0], bds[1], 1) for bds in bounds]).squeeze()
            print('Start: {}'.format(x0))
            current_minimum = scipy.optimize.minimize(to_opt, x0=x0, bounds=bounds)
            while optim_number <= nrestart:
                print('Current best min: {} at {}'.format(current_minimum.fun,
                                                          current_minimum.x))
                x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds]
                if len(x0) == 1:
                    [x0] = x0
                print('Restart: {}'.format(x0))
                optim = scipy.optimize.minimize(to_opt, x0=x0, bounds=bounds)
                print('computed min: {}'.format(optim.fun))
                if optim.fun < current_minimum.fun:
                    current_minimum = optim
                optim_number += 1

            next_to_evaluate = optim.x
            value_evaluated = true_function(next_to_evaluate)
            X = np.vstack([gp.X_train_, next_to_evaluate])
            y = np.append(gp.y_train_, value_evaluated)
            gp.fit(X, y)
            i += 1
        return gp
    gpVoro = level_set_Vorobev(function_2d, gp, bounds, T, 5)
    gp_var = margin_probability_enrichment(function_2d, gp, bounds, T, 100)
    # cond_entropy_2d = acq.conditional_entropy(gp, all_combinations,
    #                                           all_combinations, M=5,
    #                                           nsamples=100)
    plt.figure(1)
    mp = margin_probability(gp_var, T, all_combinations, 1 - 0.025)
    bplt.plot_2d_strategy(gp_var, all_combinations, function_2d, mp)
    plt.figure(2)
    mp = margin_probability(gp_var, T, all_combinations, 1 - 0.025)
    bplt.plot_2d_strategy(gp_var, all_combinations, function_2d, mp)


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
                         niterations=30,
                         plot=False, save=False,
                         bounds=np.asarray(bounds))

    PEI_crit = PEI_comb(PEI_joint, all_combinations, [1], np.asarray(bounds))
    bplt.plot_2d_strategy(PEI_joint, all_combinations, function_2d,
                          PEI_crit, criterion_plottitle = 'PEI',
                          show=True, cond_min=True)


    # EGO brute ------------------------------
    gp_brute = EGO_brute(gp, function_2d, all_combinations, niterations=100, plot=False)
    EI_criterion_brute = acq.gp_EI_computation(gp_brute, all_combinations)
    bplt.plot_2d_strategy(gp_brute, all_combinations, function_2d, EI_criterion_brute)


    # EGO analytical -------------------------
    gp_analytical = EGO_analytical(gp, function_2d, X_ = all_combinations, niterations=100,
                                   plot=True,
                                   nrestart=15,
                                   bounds = [(0, 1)] * 2, save=True)
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
    # print alpha_99, k_check_99
    # print alpha_95, k_check_95
    # print alpha_90, k_check_90

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
    from test_functions import rosenbrock_general

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


    alpha99 = 1.91089152
    alpha_est_3 = np.load('/home/victor/Bureau/tmp/3x40alpha.npy')
    alpha_est_5 = np.load('/home/victor/Bureau/tmp/5x24alpha.npy')
    alpha_est_10 = np.load('/home/victor/Bureau/tmp/10x12alpha.npy')
    alpha_est_20 = np.load('/home/victor/Bureau/tmp/20x6alpha.npy')
    alpha_est_30 = np.load('/home/victor/Bureau/tmp/30x4alpha.npy')


    alpha_3 = np.asarray([[i[1], i[0] - alpha99] for i in alpha_est_3])
    alpha_5 = np.asarray([[i[1], i[0] - alpha99] for i in alpha_est_5])
    alpha_10 = np.asarray([[i[1], i[0] - alpha99] for i in alpha_est_10])
    alpha_20 = np.asarray([[i[1], i[0] - alpha99] for i in alpha_est_20])
    alpha_30 = np.asarray([[i[1], i[0] - alpha99] for i in alpha_est_30])

    plt.style.use('default')
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}'
           r'\usepackage{amssymb}')
    # plt.rc('font','Computer Modern Roman')
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(alpha_3[:, 0], (alpha_3[:, 1]), '.-b', label=r'3x40')
    plt.plot(alpha_5[:, 0], (alpha_5[:, 1]), '.-r', label=r'5x24')
    plt.plot(alpha_10[:, 0], (alpha_10[:, 1]), '.-g', label=r'10x12')
    plt.plot(alpha_20[:, 0], (alpha_20[:, 1]), '.-m', label=r'20x6')
    plt.plot(alpha_30[:, 0], (alpha_30[:, 1]), '.-k', label=r'30x4')
    plt.title(r'$\hat{\alpha}_{.99} - \alpha_{.99}$')
    plt.xlabel(r'\# model evaluation')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(alpha_3[:, 0],  np.log(alpha_3[:, 1]), '.-b', label=r'3x40')
    plt.plot(alpha_5[:, 0],  np.log(alpha_5[:, 1]), '.-r', label=r'5x24')
    plt.plot(alpha_10[:, 0], np.log(alpha_10[:, 1]), '.-g', label=r'10x12')
    plt.plot(alpha_20[:, 0], np.log(alpha_20[:, 1]), '.-m', label=r'20x6')
    plt.plot(alpha_30[:, 0], np.log(alpha_30[:, 1]), '.-k', label=r'30x4')
    plt.legend()
    plt.title(r'$\log(\hat{\alpha}_{.99} - \alpha_{.99})$')
    plt.xlabel(r'\# model evaluation')
    plt.suptitle(r'Estimation of $\alpha_{.99}$ using sampling in $\mathbb{M}_{\eta}$ (nsamples x nsteps)\\')
    # plt.tight_layout()
    plt.show()
