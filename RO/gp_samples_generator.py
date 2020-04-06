#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal



def sample_y_modified(gp, X, n_samples, random_state=0):
    """Sample value from the GP, and evaluate them at X

    Parameters
    ----------
    gp : GaussianProcessRegressor

    X : numpy.array

    n_samples : number of samples wanted

    random_state : , optional


    Returns
    -------
    out : Samples

    """
    y_mean, y_cov = gp.predict(X, return_cov=True)
    chol = np.linalg.cholesky(y_cov)
    for i in xrange(n_samples):
        yield y_mean + np.dot(chol, (np.random.normal(size=len(X))))


if '__name__' == '__main__':

    import profile

    def test_speed():
        np.random.multivariate_normal(np.zeros(2500), np.eye(2500))
        np.random.normal(size=2500)
        multivariate_normal.rvs(np.zeros(2500), np.eye(2500))
        multivariate_normal.rvs(size=2500)

    profile.run('test_speed()')
    A = [[2, 1],
         [1, 2]]

    B = np.linalg.cholesky(A)

    X = np.asarray([[0., 0.],
                    [0.33, 0.33],
                    [0.5, 0.33],
                    [0.84, 0.3367],
                    [0.180294, 0.030394]])
