# coding: utf-8
# !/usr/bin/env python

import numpy as np
import scipy
import copy


# -----------------------------------------------------------------------------
def expected_improvement_closed_form(m, s):
    """ Analytical form of the EI (m centered)
    """
    with np.errstate(divide='ignore'):
        EI = s * scipy.stats.norm.pdf(m / s) + m * scipy.stats.norm.cdf(m / s)
        EI[s < 1e-9] = 0.0
    return EI


# -----------------------------------------------------------------------------
def probability_of_improvement(m, s):
    """ probability of improvement (m centered)
    """
    with np.errstate(divide='ignore'):
        PI = scipy.stats.norm.cdf(m / s)
        PI[s < 1e-9] = 0.0
    return PI


# -----------------------------------------------------------------------------
def gp_PI_computation(gp, X_, y_mean=None, y_std=None):
    """ Compute the EI for a gp
    """
    if X_.ndim == 1:
        X_ = np.atleast_2d(X_).T

        # if y_mean is None or y_std is None:
        #     print ' No prediction in input, predictions will be computed'
    y_mean, y_std = gp.predict(np.atleast_2d(X_), return_std=True)

    pos_current_minimum = np.argmin(gp.y_train_)
    current_minimum = gp.y_train_[pos_current_minimum]
    m = current_minimum - y_mean
    s = y_std
    return probability_of_improvement(m, s)


# -----------------------------------------------------------------------------
def gp_EI_computation(gp, X_, y_mean=None, y_std=None):
    """ Compute the EI for a gp
    """
    if X_.ndim == 1:
        X_ = np.atleast_2d(X_).T

        # if y_mean is None or y_std is None:
        #     print ' No prediction in input, predictions will be computed'
    y_mean, y_std = gp.predict(np.atleast_2d(X_), return_std=True)

    pos_current_minimum = np.argmin(gp.y_train_)
    current_minimum = gp.y_train_[pos_current_minimum]
    m = current_minimum - y_mean
    s = y_std
    return expected_improvement_closed_form(m, s)


# ----------------------------------------------------------------------------
def acquisition_maxEI_brute(gp, X_):
    """
    Returns maximum value of EI computed by brute force
    """
    EI_computed = gp_EI_computation(gp, X_)
    return X_[np.argmax(EI_computed)]


# -----------------------------------------------------------------------------
def acquisition_maxEI_analytical_gradientfree(gp, nrestart, bounds):
    """
    Return maximum value of EI computed with gradient free optimization
    """
    def EI_lambda(value):
        return -gp_EI_computation(gp, np.atleast_2d(value))
    optim_number = 1
    rng = np.random.RandomState()
    dim = gp.X_train_.shape[1]
    if bounds is None:
        bounds = dim * [(0, 1)]
    x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds]

    maxEI = scipy.optimize.minimize(EI_lambda, x0=x0, bounds=bounds)

    while optim_number <= nrestart:
        x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds]
        optim = scipy.optimize.minimize(EI_lambda, x0=x0, bounds=bounds)
        if optim.fun < maxEI.fun:
            maxEI = optim
        optim_number += 1
    return maxEI.x


# -----------------------------------------------------------------------------
def gp_LCB(gp, X_, kappa):
    y_mean, y_std = gp.predict(np.atleast_2d(X_), return_std=True)
    return y_mean - kappa * y_std


# -----------------------------------------------------------------------------
def acquisition_LCB(gp, kappa, nrestart, bounds):
    """ Analytical form of the EI (m centered)
    """
    def LCB(X_):
        return gp_LCB(gp, X_, kappa)
    optim_number = 1
    rng = np.random.RandomState()
    dim = gp.X_train_.shape[1]
    if bounds is None:
        bounds = dim * [(0, 1)]
    x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds]

    minLCB = scipy.optimize.minimize(LCB, x0=x0, bounds=bounds)

    while optim_number <= nrestart:
        x0 = [rng.uniform(bds[0], bds[1], 1) for bds in bounds]
        optim = scipy.optimize.minimize(LCB, x0=x0, bounds=bounds)
        if optim.fun < minLCB.fun:
            minLCB = optim
        optim_number += 1
    return minLCB.x


# -----------------------------------------------------------------------------
def gp_qEI_computation_brute(gp, qPoints, nsim=1000):
    """
    qPoints : [npoints,q]
    """
    qPoints = np.atleast_2d(qPoints)
    npoints = qPoints.shape[0]
    qEI = np.empty(npoints)
    for i in xrange(npoints):
        y_mean, q_cov = gp.predict(qPoints[i, :, np.newaxis], return_cov=True)
        samples_MC = scipy.stats.multivariate_normal.rvs(y_mean, q_cov, nsim)
        minY = np.min(gp.y_train_)
        qEI[i] = np.mean(minY > samples_MC.min(1))
    return qEI


# -----------------------------------------------------------------------------
def find_farthest_point(datapoints, bounds, nrestart=10):
    """ find_farthest_point

    Find the point P such that P = argmax ||datapoints - P ||**2

    Arguments:
    ----------
        datapoints : array (npoints, ndim)
        bounds : array (ndim, 2)
    """
    if datapoints.ndim == 1:
        npoints, ndim = np.atleast_2d(datapoints).T.shape
        datapoints = np.atleast_2d(datapoints).T

        def fun_to_optimize(P):
            return -np.min(np.sum((P - datapoints)**2), 0)

    else:
        npoints, ndim = np.atleast_2d(datapoints).T.shape
        # print npoints, ndim

        def fun_to_optimize(P):
            return -np.min(np.sum((P - datapoints.T)**2, 1))

    bounds = np.atleast_2d(bounds)
    bnds = [tuple(bounds[i, :]) for i in xrange(ndim)]

    opt = scipy.optimize.minimize(fun_to_optimize, x0=np.mean(bounds, 1),
                                  bounds=bnds)
    best_P, best_J = opt.x, opt.fun
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1],
                                size=(nrestart, ndim)):
        opt_restart = scipy.optimize.minimize(fun_to_optimize, x0=x0,
                                              bounds=bnds)
        if opt_restart.fun < opt.fun:
            best_P = opt_restart.x
            best_J = opt_restart.fun

    return best_P, best_J


# -----------------------------------------------------------------------------
def acquisition_exploEI_analytical(gp, nrestart, bounds, idx_U):
    """
    max EI on control variable, max distance on uncertain variables
    """

    argmaxEI = acquisition_maxEI_analytical_gradientfree(gp, nrestart, bounds)
    argmax_exploEI = argmaxEI
    if idx_U is None:
        distance = np.nan
        return argmax_exploEI, distance
    else:
        idx_U = map(int, idx_U)
        # print 'idx_U: ', idx_U
        points_U = np.atleast_2d(gp.X_train_[:, idx_U]).T
        bounds_U = np.atleast_2d(bounds[idx_U, :])
        # print 'pts_U', points_U
        farthest_U, distance = find_farthest_point(points_U, bounds_U)
        argmax_exploEI[idx_U] = farthest_U
        return argmax_exploEI, -distance


# ------------------------------------------------------------------------------
def quantization_operator(y):
    """
    Quantization operator Q in IAGO strategy
    Args:
       y (array): sorted real values
    """
    M = len(y)

    def Q_y(u):
        i = 0
        yi = y[0]
        while u > yi and i < M - 1:
            i += 1
            yi = y[i]
        return y[i]

    return Q_y
# Q = quantization_operator([1,2,3,4,10,55])
# Q(5);Q(4);Q(100)


# -----------------------------------------------------------------------------
def gp_fitted_candidate(gp, xcandidate, yvalue, optim_covariance=True):
    """
    Add the pair (xcandidate, yvalue) to the experimental design, and fits gp
    """
    gp_candidate_added = copy.copy(gp)
    Xtrain = gp_candidate_added.X_train_
    ytrain = gp_candidate_added.y_train_
    if not optim_covariance:
        gp_candidate_added.optimizer = None
    new_design = np.vstack([Xtrain, xcandidate])
    new_response = np.append(ytrain, yvalue)
    gp_candidate_added.fit(new_design, new_response)
    return gp_candidate_added


# -----------------------------------------------------------------------------
def compute_yvalues_quantiles(gp, xcandidate, M=10):
    """
    Quantiles of the gaussian process at xcandidate
    """
    ndim = gp.X_train_.shape[1]
    if ndim == 1:
        xcandidate = np.array(xcandidate).reshape(-1, 1)
    else:
        xcandidate = np.atleast_2d(xcandidate)
    m, s = gp.predict(xcandidate, return_std=True)
    percentage = np.linspace(1.0, M - 1, M) / float(M)
    y_values = scipy.stats.norm.ppf(percentage, loc=m, scale=s)
    return y_values


# -----------------------------------------------------------------------------
def conditional_globalmin_brute(gp, X_, nsamples=1000):
    """
    Minimizers of the sampled gp

    Returns the minimizer by direct search over X_ of the 'nsamples' sample
    paths of gp
    TODO: Bottleneck of performance in the sampling step (in scikit...)
    """
    pos_argmin = np.empty(nsamples)
    ndim = gp.X_train_.shape[1]
    if ndim == 1:
        X_ = np.array(X_).reshape(-1, 1)
    else:
        X_ = np.atleast_2d(X_)
    y_samples = gp.sample_y(X_, nsamples)
    pos_argmin = y_samples.argmin(axis=0)
    return X_[pos_argmin.astype(int)], pos_argmin.astype(int)


# -----------------------------------------------------------------------------
def conditional_entropy_value_added(gp, xcandidate, yvalue, X_, nsamples, optim):
    """
    Conditional entropy given the observations where (xcandidate, yvalue) has
    been added
    """
    gp_candidate_added = gp_fitted_candidate(gp, xcandidate, yvalue, optim)
    Xstar, __ = conditional_globalmin_brute(gp_candidate_added, X_, nsamples)
    vals, count = np.unique(Xstar, axis=0, return_counts=True)
    return scipy.stats.entropy(count)


# -----------------------------------------------------------------------------
def conditional_entropy(gp, xcandidate, X_, M, nsamples,
                        iterative_optim = False):
    npoints = len(xcandidate)
    # nfeatures = gp.X_train_.shape[1]
    entropy_out = np.empty(npoints)
    for i, xc in enumerate(xcandidate):
        print 'candidate: ', xc
        y_values = compute_yvalues_quantiles(gp, xc, M=M)
        entropies = np.empty(M)
        for j, yval in enumerate(y_values):
            # print '  ', yval
            entropies[j] += conditional_entropy_value_added(gp, xc,
                                                            yval, X_, nsamples,
                                                            iterative_optim)
        entropy_out[i] = entropies.mean()
    return entropy_out
