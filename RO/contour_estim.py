#!/usr/bin/env python
# coding: utf-8

from __future__ import unicode_literals, print_function, with_statement
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
import time

import RO.acquisition_function as acq
import RO.bo_plot as bplt
import RO.bo_wrapper as bow

import matplotlib as mpl
from matplotlib import cm


plt.style.use('seaborn')
mpl.rcParams['image.cmap'] = u'viridis'
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
exec(open('/home/victor/acadwriting/Manuscrit/plots_settings.py').read())
from RO.test_functions import branin_2d
from RO.gp_samples_generator import sample_y_modified



function_2d = lambda X: branin_2d(X, switch=False)
np.random.seed(3394)
ndim = 2
bounds = np.asarray([[0, 1], [0, 1]])
# initial_design_2d = np.array([[1,1],[2,2],[3,3],[4,4], [5,2], [1,4],[0,0],[5,5], [4,1]])/5.0
initial_design_2d = pyDOE.lhs(n=2,
                              samples=30,
                              criterion='maximin',
                              iterations=50)
response_2d = function_2d(initial_design_2d)

# Fitting of the GaussianProcess -------------------------------------
gp = GaussianProcessRegressor(kernel=Matern(np.ones(ndim) / 5.0),
                              n_restarts_optimizer=50)
gp.fit(initial_design_2d, response_2d)

# Builds a regular grid ---------------------------------------------
ngrid = 100
X_, Y_ = np.linspace(0, 1, ngrid), np.linspace(0, 1, ngrid)
# xx, yy = np.meshgrid(X_, Y_, indexing = 'ij')
all_combinations, (X_mg, Y_mg) = bow.pairify(X_, Y_)

ngrid_big, ngrid_big_2 = 4000, 4000
X_l, X_l2 = np.linspace(0, 1, ngrid_big), np.linspace(0, 1, ngrid_big_2)
big_comb, (mg_b1, mg_b2) = bow.pairify(X_l, X_l2)
true_fun = function_2d(big_comb).reshape(4000, 4000)
alpha_95_true = np.quantile(true_fun / true_fun.min(0), 0.95, axis=1).min()
alpha_95_true

EI_criterion = acq.gp_EI_computation(gp, all_combinations)


###
T = 1.8
ss = bow.sample_from_criterion(1000, lambda x: bow.margin_indicator(gp, T, 1 - 0.025, x),
                               bounds=np.asarray([[0, 1],
                                                  [0, 1]]),
                               Ncandidates=5)

kmeans = bow.cluster_and_find_closest(10, ss)

F1 = bow.alpha_set_quantile(gp, T, 0.025, all_combinations).reshape(ngrid, ngrid)
F2 = bow.alpha_set_quantile(gp, T, 0.975, all_combinations).reshape(ngrid, ngrid)
coverage_probability = bow.coverage_probability(gp, T, all_combinations).reshape(ngrid, ngrid)
Meta = bow.margin_probability(gp, T, all_combinations, 1 - 0.025)


plt.figure(figsize=np.array(col_full) * np.array([1.2, 1.]))
plt.subplot(1, 2, 1)
plt.contourf(X_mg, Y_mg, gp.predict(all_combinations).reshape(ngrid, ngrid))
cp = plt.contour(X_mg, Y_mg, coverage_probability, levels=[0.025, 0.975], cmap=cm.get_cmap('Dark2'))
plt.clabel(cp, fmt=r'$\pi_A$=%1.3f')

# FF = plt.contourf(X_mg, Y_mg, F1, 3, hatches=['', '\\'], alpha=0., colors='none')
# plt.contourf(X_mg, Y_mg, 1 - F2, 3, hatches=['', '/'], alpha=0., colors='none')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$u$')
plt.title('GP prediction \nand boundaries of $\mathbb{M}_{\eta}$')
# plt.contour(X_mg, Y_mg, margin_indicator(gp, T, 1 - 0.025, all_combinations).reshape(ngrid, ngrid), levels=[0.45, 0.55])
plt.subplot(1, 2, 2)
# plt.contourf(X_mg, Y_mg, bow.margin_indicator(gp, T, 1 - 0.025, all_combinations).reshape(ngrid, ngrid))
# Meta.reshape(ngrid, ngrid))
plt.contour(X_mg, Y_mg, coverage_probability, levels=[0.025, 0.975], cmap=cm.get_cmap('Dark2'))
cp = plt.contour(X_mg, Y_mg, function_2d(all_combinations).reshape(ngrid, ngrid),
                 levels=[T], cmap=cm.get_cmap('bwr'))
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(np.nan, np.nan, color=cm.get_cmap('bwr')(0), lw=2, label='Truth: $\{f(x) = T\}$')
plt.plot(np.nan, np.nan, color=cm.get_cmap('Dark2')(0.), lw=1, label='$\{\pi_A(x) = \eta/2\}$')
plt.plot(np.nan, np.nan, color=cm.get_cmap('Dark2')(7), lw=1, label='$\{\pi_A(x) = 1 - \eta/2\}$')
# plt.clabel(cp, fmt=r'$f(x)=%1.3f$')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$u$')
plt.scatter(ss[:, 0], ss[:, 1], s=3, alpha=0.5, label='samples')
plt.plot(kmeans[0][:, 0], kmeans[0][:, 1], 'r', marker='*', linestyle='', label='Centroids')
# plt.plot(gp.X_train_[:, 0], gp.X_train_[:, 1], 'w', marker='X', ls='')
plt.title(r'Samples and centroids in $\mathbb{M}_{\eta}$')
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig('/home/victor/acadwriting/Manuscrit/Text/Chapter4/img/margin_unc.pdf')
plt.show()


def reliability_index(arg, T, points):
    if isinstance(arg, tuple):
        m, s = arg
    else:
        m, s = arg.predict(points, return_std=True)
    return - np.abs(m - T) / s


def cluster_in_margin_of_uncertainty(gp, T=1.8, eta=0.975, q=5):
    samples = bow.sample_from_criterion(10,
                                        lambda x: bow.
                                        margin_indicator(bow.mu_sigma_delta(gp,
                                                                            x,
                                                                            T,
                                                                            [1],
                                                                            np.asarray([0, 1]),
                                                                            verbose=False),
                                                         T, eta, x),
                                        bounds=np.asarray([[0, 1],
                                                           [0, 1]]),
                                        Ncandidates=q)
    print('end sampling')
    return bow.cluster_and_find_closest(q, samples)[0]


def batch_in_margin_iteration(gp, ms, fun, T=1.5, eta=0.975, q=5):
    if ms is None:
        ms = gp
    print('samples')
    centroids = cluster_in_margin_of_uncertainty(gp, T, eta, q)

    centroids_adj = np.empty_like(centroids)
    for i, centro in enumerate(centroids):
        centroids_adj[i], _ = adjust_centroid(gp, centro, T)
    return bow.add_points_to_design(gp, centroids_adj, fun(centroids_adj),
                                    optimize_cov=True), centroids



def adjust_centroid(gp, centro, alpha):
    m2, cov2 = bow.mean_covariance_alpha(gp,
                                         np.atleast_2d(centro[0]),
                                         np.atleast_2d(centro[1]), [1], np.asarray([[0, 1]]))

    # _, ss = mu_sigma_delta(gp, np.atleast_2d(centro), alpha, [1], np.asarray([0, 1]))

    print('Adjust the centroid: ', cov2[0, 0] <= alpha**2 * cov2[1, 1])
    if cov2[0, 0] <= alpha**2 * cov2[1, 1]:
        curr_min = bow.find_minimum_sliced(gp, centro[1], [1], bounds=np.asarray([[0, 1]]))
        kstar = curr_min.x[0]
        f_min = curr_min.fun
        sliced_fun = bow.slicer_gp_predict(gp, np.asarray(centro[1]), [1], return_std=True)

        def EI_sliced(X_):
            y_mean, y_std = sliced_fun(np.atleast_2d(X_))
            m = f_min - y_mean
            return -acq.expected_improvement_closed_form(m, y_std)
        i = 0
        minval = np.inf
        while i < 5:
            opt = scipy.optimize.minimize(EI_sliced, np.random.uniform(),
                                          bounds=np.atleast_2d([0, 1]))
            if opt.fun < minval:
                curr = opt
                minval = curr.fun
            i += 1
            kEI = curr.x[0]
        print('kstar:', kstar)
        print('kEI:', kEI)
        newku = kEI, centro[1]
    else:
        newku = centro
    return np.asarray(newku), (cov2[0, 0], alpha**2 * cov2[1, 1])




gp1, centroids = batch_in_margin_iteration(gp, function_2d, T, 1 - 0.025, 10)
plt.subplot(3, 1, 1)
plt.contourf(X_mg, Y_mg, gp.predict(all_combinations).reshape(ngrid, ngrid))
plt.contour(X_mg, Y_mg, bow.margin_indicator(gp, T, 1 - 0.025, all_combinations).reshape(ngrid, ngrid),
            levels=[0.45, 0.55])
plt.plot(gp.X_train_[:, 0], gp.X_train_[:, 1], 'w.')
plt.plot(centroids[:, 0], centroids[:, 1], 'r.')
plt.subplot(3, 1, 2)
plt.contourf(X_mg, Y_mg, gp1.predict(all_combinations).reshape(ngrid, ngrid))
plt.contour(X_mg, Y_mg, bow.margin_indicator(gp1, T, 1 - 0.025, all_combinations).reshape(ngrid, ngrid),
            levels=[0.45, 0.55])

gp2, centroids = batch_in_margin_iteration(gp1, function_2d, T, 1 - 0.025, 10)
plt.plot(gp1.X_train_[:, 0], gp1.X_train_[:, 1], 'w.')
plt.plot(centroids[:, 0], centroids[:, 1], 'r.')

plt.subplot(3, 1, 3)
plt.contourf(X_mg, Y_mg, gp2.predict(all_combinations).reshape(ngrid, ngrid))
plt.contour(X_mg, Y_mg, bow.margin_indicator(gp2, T, 1 - 0.025, all_combinations).reshape(ngrid, ngrid),
            levels=[0.45, 0.55])
plt.show()

# gp_tmp = copy.copy(gp)
# T = 1.5
# for i in range(9):
#     plt.subplot(3, 3, i + 1)
#     plt.contourf(X_mg, Y_mg, gp_tmp.predict(all_combinations).reshape(ngrid, ngrid))
#     plt.contour(X_mg, Y_mg, margin_indicator(gp_tmp, T, 1 - 0.025, all_combinations).reshape(ngrid, ngrid),
#                 levels=[0.45, 0.55])
#     gp_, centroids = batch_in_margin_iteration(gp_tmp, None, function_2d, T, 1 - 0.025, 8)
#     plt.plot(gp_tmp.X_train_[:, 0], gp_tmp.X_train_[:, 1], 'w.')
#     plt.plot(centroids[:, 0], centroids[:, 1], 'r.')
#     gp_tmp = gp_
# plt.show()
ngrid = 200
X_, Y_ = np.linspace(0, 1, ngrid), np.linspace(0, 1, ngrid)
# xx, yy = np.meshgrid(X_, Y_, indexing = 'ij')
all_combinations, (X_mg, Y_mg) = bow.pairify(X_, Y_)

ngrid_big, ngrid_big_2 = 5000, 5000
X_l, X_l2 = np.linspace(0, 1, ngrid_big), np.linspace(0, 1, ngrid_big_2)
big_comb, (mg_b1, mg_b2) = bow.pairify(X_l, X_l2)

p = .95
out_t = function_2d(big_comb).reshape(ngrid_big, ngrid_big_2)
kstar_t = out_t.argmin(0)
Jstar_t = out_t.min(0)
rho_t = (out_t / Jstar_t[np.newaxis, :])
alpha_t = np.quantile(rho_t, p, axis=1)


out_t_allc = function_2d(all_combinations).reshape(ngrid, ngrid)
kstar_t_allc = out_t_allc.argmin(0)
Jstar_t_allc = out_t_allc.min(0)
rho_t_allc = (out_t_allc / Jstar_t_allc[np.newaxis, :])
alpha_t_allc = np.quantile(rho_t_allc, p, axis=1)
delta_t_allc = out_t_allc - 1.8 * Jstar_t_allc[np.newaxis, :] <= 0

delta = out_t - 1.8 * Jstar_t[np.newaxis, :] <= 0


gp_tmp = copy.copy(gp)
plugin = np.empty((7, ngrid, ngrid))
p = np.empty((7, ngrid, ngrid))
ppi = p
ppi = np.empty((7, ngrid, ngrid))


threshold = 1.8
for i in range(1):
    print(i)
    plt.subplot(4, 4, i + 1)
    m, s = bow.mu_sigma_delta(gp_tmp, all_combinations, threshold, [1], np.asarray([0, 1]))
    print('plugin')
    plugin[i, :, :] = m.reshape(ngrid, ngrid)
    print('ppi')
    ppi[i, :, :] = bow.coverage_probability((m, s), 0, None).reshape(ngrid, ngrid)
    # plugin = np.concatenate((plugin, m
    #                          .reshape(ngrid, ngrid)[np.newaxis, :, :]), axis=0)
    # ppi = np.concatenate((ppi,
    #                       bow.coverage_probability((m, s), 0, None)
    #                       .reshape(ngrid, ngrid)[np.newaxis, :, :]), axis=0)
    plt.contourf(X_mg, Y_mg, gp_tmp.predict(all_combinations).reshape(ngrid, ngrid))
    plt.contour(X_mg, Y_mg,
                bow.margin_indicator((m, s), 0, 1 - 0.025, all_combinations).
                reshape(ngrid, ngrid),
                levels=[0.45, 0.55])
    plt.plot(gp_tmp.X_train_[:, 0], gp_tmp.X_train_[:, 1], 'w.')
    gp_tmp, centroids = batch_in_margin_iteration(gp_tmp, (m, s),
                                                  function_2d, threshold, 1 - 0.025, 10)
    plt.plot(centroids[:, 0], centroids[:, 1], 'r.')
plt.show()

plugin = np.concatenate((plugin, m.reshape(ngrid, ngrid)[np.newaxis, :, :]), axis=0)
ppi = np.concatenate((ppi,
                      bow.coverage_probability((m, s), 0, None).
                      reshape(ngrid, ngrid)[np.newaxis, :, :]), axis=0)
gp_by_10 = gp_tmp
plugin_10 = plugin
ppi_10 = ppi

L2_PI = []
L2_PC = []
Linf_PI = []
Linf_PC = []

vol_M_eta = []
for pi in ppi:
    vol_M_eta.append(np.mean(np.logical_and(pi >= 0.025,
                                            pi <= 0.975)))

for i in range(8):
    gamma_PI = (plugin[i] <= 0).mean(1)
    gamma_PC = ppi[i].mean(1)
    plt.plot(gamma_PC)
    L2_PI.append(np.sum((gamma_PI - delta_t_allc.mean(1))**2))
    Linf_PI.append(np.abs(gamma_PI - delta_t_allc.mean(1)).max())    
    L2_PC.append(np.sum(gamma_PC - delta_t_allc.mean(1))**2)
    Linf_PC.append(np.abs(gamma_PC - delta_t_allc.mean(1)).max())

itera = np.arange(0, 80, 10)
plt.subplot(2, 2, 1)
plt.plot(itera, L2_PI, label=r'PI')
plt.plot(itera, L2_PC, label=r'$\pi$')
plt.title(r'$\|\hat{\Gamma}_{n,\alpha} - \Gamma_{\alpha}\|_2$')
plt.legend()
plt.yscale('log')
plt.subplot(2, 2, 3)
plt.plot(itera, Linf_PI, label=r'PI')
plt.plot(itera, Linf_PC, label=r'$\pi$')
plt.title(r'$\|\hat{\Gamma}_{n,\alpha} - \Gamma_{\alpha}\|_{\infty}$')
plt.yscale('log')
plt.legend()
plt.subplot(2, 2, (2, 4))
plt.title(r'Estimated volume of $\mathbb{M}_\eta$')
plt.plot(vol_M_eta)
plt.tight_layout()
plt.show()






gp_tmp = gp
from RO.gp_samples_generator import sample_y_modified
p = .95
Nsamples = 2000
alpha_p_samples = np.empty((Nsamples, len(X_)))
# rho_max = np.empty((Nsamples, len(X_)))
samples = np.empty((2000, 10, 10))
samples_rho = np.empty((2000, 10, 10))


Nsamples = 2000
rh = np.empty((ngrid, ngrid, Nsamples))
for j, aa in progressbar(list(enumerate(sample_y_modified(gp, all_combinations, Nsamples)))):
    # print '{}\r'.format(j, Nsamples),
    curr = aa.reshape(ngrid, ngrid)
    Jstar = curr.min(0)
    rho = (curr / Jstar[np.newaxis, :])
    rh[:, :, j] = rho
    alpha_p = np.quantile(rho, p, axis=1)
    alpha_p_samples[j, :] = alpha_p


mc_tu = bow.mean_covariance_alpha_vector(gp, all_combinations, [1], np.atleast_2d([0, 1]))
CV = np.asarray([np.sqrt(cov[0, 0])/mean[0] for mean, cov in mc_tu])
CV_star = np.asarray([np.sqrt(cov[1, 1]) / mean[1] for mean, cov in mc_tu])
CV.min(), CV.max()
CV_star.min(), CV_star.max()
plt.plot(CV)
plt.show()

plt.plot(CV_star)
plt.show()


def compute_plugin(mc_tuple, correc=True):
    m, co = mc_tuple
    m1, m2 = m[0], m[1]
    s1, s2 = co[0, 0], co[1, 1]
    rho = co[0, 1] / np.sqrt(s1 * s2)
    sig_of_normal = (s1 / m1**2) + (s2 / m2**2) - 2 * co[0, 1] / (m1 * m2)
    if correc:
        return (m1 / m2) * np.exp(sig_of_normal / 2)
    else:
        return (m1 / m2)


def mean_variance_logratio(gp, x, verbose=False):
    mc_tu = bow.mean_covariance_alpha_vector(gp,
                                             np.atleast_2d(x),
                                             [1],
                                             np.atleast_2d([0, 1]))
    mean = np.empty(len(x))
    var = np.empty(len(x))
    if verbose:
        gene = progressbar(list(enumerate(mc_tu)))
    else:
        gene = enumerate(mc_tu)
    for i, mc in gene:
        m, co = mc
        m1, m2 = m[0], m[1]
        s1, s2 = co[0, 0], co[1, 1]
        rho = co[0, 1] / np.sqrt(s1 * s2)
        sig_of_normal = (s1 / m1**2) + (s2 / m2**2) - 2 * co[0, 1] / (m1 * m2)
        mean[i] = np.log(m1 / m2)
        var[i] = sig_of_normal
    return mean, var



def integrated_variance_logratio(gp, integration_points):
    _, vlograt = mean_variance_logratio(gp, integration_points, verbose=False)
    return vlograt.mean()



def augmented_IMSE_logratio(x_input, gp, int_points, Nscenarios=5):
    """
    Compute the IMSE of the logratio, integrated based on int_points,
    and augmented with the point x_input
    """
    augmented_IMSElogratio = np.empty(len(np.atleast_2d(x_input)))
    for i, x_add in enumerate(np.atleast_2d(x_input)):
        mp, sp = gp.predict(np.atleast_2d(x_add), True)
        eval_x_add = scipy.stats.norm.ppf(np.linspace(0.01, 0.99, Nscenarios, endpoint=True),
                                          loc=mp, scale=sp)
        int_ = 0
        for ev in eval_x_add:
            gp_cand = bow.add_points_to_design(gp, x_add, ev, False)
            int_ += integrated_variance_logratio(gp_cand, int_points)
        augmented_IMSElogratio[i] = int_
    return augmented_IMSElogratio


def optimize_aIMSE_logratio(gp, Nintegration=25, Nscenarios=5, Noptim=3):
    int_points = pyDOE.lhs(2, Nintegration, criterion='maximin', iterations=20)
    best = np.inf
    optim_n = 0
    while optim_n < Noptim:
        x0 = np.random.uniform(size=2).reshape(1, 2)
        op = scipy.optimize.minimize(augmented_IMSE_logratio, args=(gp, int_points, 5),
                                     x0=x0, bounds=np.asarray([[0, 1],
                                                               [0, 1]]))
        if op.fun < best:
            op_best = op
        optim_n += 1
    return op_best



gp_tmp = copy.copy(gp)
aIMSE_logratio = []
minqPI = []
for i in progressbar(range(30)):
    op_best = optimize_aIMSE_logratio(gp_tmp, Nintegration=25, Nscenarios=5, Noptim=3)
    aIMSE_logratio.append(op_best.fun)
    newku = op_best.x
    gp_tmp = bow.add_points_to_design(gp_tmp, newku, function_2d(newku), True)
    mst, sigst = np.empty(len(lhsU)), np.empty(len(lhsU))
    for j, u in progressbar(list(enumerate(lhsU)), 'Zstar'):
        mst[j], sigst[j] = bow.mu_sigma_star(gp_tmp, u, [1], np.atleast_2d([0, 1]))
    be = optim_qPI(gp_tmp, 0.95, 5, mst, sigst, True)
    minqPI.append((be.fun, be.x))


qPI_iter = np.empty((70, 100))
qPI_iter_c = np.empty((70, 100))
qMC_iter = np.empty((70, 100))
qMC_iter_log = np.empty((70, 100))
Nsamples = 1000

for i in range(70):
    gp_iter = bow.rm_obs_gp(gp_tmp, 30, i)
    qMC_sam = np.empty((Nsamples, 100))
    for j, aa in progressbar(list(enumerate(sample_y_modified(gp_iter,
                                                              all_combinations,
                                                              Nsamples)))):
    # print '{}\r'.format(j, Nsamples),
        curr = aa.reshape(ngrid, ngrid)
        Jstar = curr.min(0)
        rho = (curr / Jstar[np.newaxis, :])
        q_p = np.quantile(rho, p, axis=1)
        qMC_sam[j, :] = q_p
    qMC_iter[i, :] = qMC_sam.mean(0)
    
    # mst, sigst = np.empty(len(lhsU)), np.empty(len(lhsU))
    # for j, u in progressbar(list(enumerate(lhsU)), 'Zstar'):
    #     mst[j], sigst[j] = bow.mu_sigma_star(gp_iter, u, [1], np.atleast_2d([0, 1]))
    # qPI_iter[i, :] = compute_qPI(X_, gp_iter, p, lhsU, mst, sigst)
    # mc_tu = bow.mean_covariance_alpha_vector(gp_iter, all_combinations, [1], np.atleast_2d([0, 1]))
    # PI = np.asarray([compute_plugin(mc) for mc in mc_tu]).reshape(ngrid, ngrid)
    # qPI_iter_c[i, :] = np.quantile(PI, 0.95, axis=1)



L2 = []
L2_MC = []
L2_c = []
Linf = []
Linf_c = []
Linf_MC = []
dist_target = []
dist_target_c = []
dist_target_MC = []
for qPI_, qPI_c, qMC in zip(qPI_iter, qPI_iter_c, qMC_iter):
    L2.append(np.sum((alpha_t_allc - qPI_)**2))
    Linf.append(np.max(np.abs(alpha_t_allc - qPI_)))
    L2_c.append(np.sum((alpha_t_allc - qPI_c)**2))
    Linf_c.append(np.max(np.abs(alpha_t_allc - qPI_c)))

    L2_MC.append(np.sum((alpha_t_allc - qMC)**2))
    Linf_MC.append(np.max(np.abs(alpha_t_allc - qMC)))

    dist_target.append(np.abs(alpha_t.min() - qPI_.min()))
    dist_target_c.append(np.abs(alpha_t.min() - qPI_c.min()))
    dist_target_MC.append(np.abs(alpha_t.min() - qMC.min()))

plt.figure(figsize=col_full)
plt.subplot(1, 3, 1)
plt.plot(L2)
plt.plot(L2_c)
plt.plot(L2_MC)
plt.yscale('log')
plt.title(r'$\|q_p - \hat{q}_{p,n}^{\mathrm{PI}} \|_2^2$')
plt.xlabel(r'$n$')
plt.ylabel(r'$L^2$ norm')
plt.subplot(1, 3, 2)
plt.plot(Linf)
plt.plot(Linf_c)
plt.plot(Linf_MC)
plt.title(r'$\|q_p - \hat{q}_{p,n}^{\mathrm{PI}} \|_{\infty}$')
plt.xlabel(r'$n$')
plt.ylabel(r'$L^{\infty}$ norm')
plt.yscale('log')
plt.subplot(1, 3, 3)
plt.title(r'$|\alpha_p - \hat{\alpha}_{p,n}|$')
plt.plot(dist_target)
plt.plot(dist_target_c)
plt.plot(dist_target_MC)
plt.xlabel(r'$n$')
plt.ylabel(r'$L^{\infty}$ norm')
plt.yscale('log')
plt.tight_layout()
# plt.savefig('/home/victor/acadwriting/Manuscrit/Text/Chapter4/img/qPI_aIMSE.pdf')
plt.show()

aa = (gp_tmp.X_train_, gp_tmp.y_train_, L2, L2_MC, L2_c, Linf, Linf_c, Linf_MC, dist_target, dist_target_c, dist_target_MC)
bb = ("xtrain", "ytrain", "L2", "L2_MC", "L2_c", "Linf", "Linf_c", "Linf_MC", "dist_target", "dist_target_c", "dist_target_MC")

di = dict(zip(bb, aa))
np.save('/home/victor/RO_VT/RO/aIVPC_qPI.npy', di)
plt.contourf(X_mg, Y_mg, function_2d(all_combinations).reshape(ngrid, ngrid))
plt.plot(gp.X_train_[:, 0], gp.X_train_[:, 1], '.')
plt.plot(gp_tmp.X_train_[30:, 0], gp_tmp.X_train_[30:, 1], '.')
plt.show()


def integrated_variance_logratio_at_point(x, uarray, verbose=True):
    int_var = np.empty(len(x))
    for i, x_ in progressbar(list(enumerate(x))):
        pts = np.hstack([x_ * np.ones_like(uarray), uarray])
        mlograt, vlograt = mean_variance_logratio(pts, verbose=False)
        int_var[i] = vlograt.mean()
    return int_var

vv = integrated_variance_logratio_at_point(X_, uarray)



lhsU = pyDOE.lhs(1, 100, criterion='maximin', iterations=20)
gp_ = copy.copy(gp)
for i in progressbar(range(70)):
    mst, sigst = np.empty(len(lhsU)), np.empty(len(lhsU))
    for i, u in progressbar(list(enumerate(lhsU)), 'init Z*'):
        mst[i], sigst[i] = bow.mu_sigma_star(gp_, u, [1], np.atleast_2d([0, 1]))
    be = optim_qPI(gp_, 0.95, 5, mst, sigst, True)

    ku_samples = np.random.uniform(size=2 * 500).reshape(500, 2)
    mlr, vlr = mean_variance_logratio(gp_, ku_samples)
    U = reliability_index((mlr, np.sqrt(vlr)), np.log(be.fun), None)
    new_ku = ku_samples[U.argmax()]
    gp_ = bow.add_points_to_design(gp_, new_ku, function_2d(new_ku), True)



plt.plot(gp.X_train_[:, 0], gp.X_train_[:, 1], '.')
plt.plot(gp_.X_train_[30:, 0], gp_.X_train_[30:, 1], '.')
plt.show()



def get_quantiles_at_k(gp, k, lhsU, p, Kq):
     # = np.empty(len(k))
    quantiles_array = np.empty((len(k), (Kq + 1)))
    for i, x_ in progressbar(list(enumerate(k))):
        pts = np.hstack([x_ * np.ones_like(lhsU), lhsU])
        mlograt, vlograt = mean_variance_logratio(gp, pts, verbose=False)
        qminus= np.quantile(mlograt + scipy.stats.norm.ppf(0.025) * np.sqrt(vlograt), p)
        qplus = np.quantile(mlograt + scipy.stats.norm.ppf(0.975) * np.sqrt(vlograt), p)
        q = np.asarray(np.quantile(mlograt, p))
        quantiles_array[i, :] = np.hstack([q,
                                           np.linspace(qminus, qplus, Kq, endpoint=True)])
    return quantiles_array


ql_array = get_quantiles_at_k(gp, X_, lhsU, 0.95, 3)


def margin_uncertainty_ratio_indicator(gp, X, q, eta=0.975):
    k = scipy.stats.norm.ppf(eta)
    mlograt, vlograt = mean_variance_logratio(gp, X, verbose=False)
    return np.logical_and(mlograt - k * np.sqrt(vlograt) < q, mlograt + k * np.sqrt(vlograt) > q)


def margin_uncertainty_ratio_indicator_ms((m, s), q, eta=0.975):
    k = scipy.stats.norm.ppf(eta)
    return np.logical_and(m - k * s < q, m + k * s > q)




def cluster_in_margin_of_uncertainty_ratio(gp, ql, eta=0.975, q=5, return_samples=False, **kwargs):
    samples = bow.sample_from_criterion(1000,
                                        lambda x: margin_uncertainty_ratio_indicator(gp,
                                                                                     x,
                                                                                     ql,
                                                                                     eta),
                                        bounds=np.asarray([[0, 1],
                                                           [0, 1]]),
                                        Ncandidates=q)
    print('end sampling')
    if return_samples:
        return bow.cluster_and_find_closest(q, samples, **kwargs)[0], samples
    else:
        return bow.cluster_and_find_closest(q, samples, **kwargs)[0]


gp_iter = copy.copy(gp)
lhsU = pyDOE.lhs(1, 200, criterion='maximin', iterations=100)
for j in range(10, 14):
    cl = []
    ql_array = get_quantiles_at_k(gp_iter, X_, lhsU, 0.95, 2)
    mlograt, vlograt = mean_variance_logratio(gp_iter, all_combinations, verbose=False)
    for i in range(3):
        ind = margin_uncertainty_ratio_indicator_ms((mlograt, np.sqrt(vlograt)), ql_array.min(0)[i])
        plt.subplot(2, 2, i + 1)
        plt.title(np.exp(ql_array.min(0)[i]))
        plt.contourf(X_mg, Y_mg, ind.reshape(100, 100))
        ktilde = X_[ql_array.argmin(0)[i]]
        mtilde, vtilde = mean_variance_logratio(gp_iter, np.hstack([ktilde * np.ones_like(lhsU),
                                                                    lhsU]))
        utilde = lhsU[(np.abs(mtilde - ql_array.min(0)[i]) / np.sqrt(vtilde)).argmin()]

        clus, ss = cluster_in_margin_of_uncertainty_ratio(gp_iter,
                                                          ql_array.min(0)[i], eta=0.975,
                                                          q=3,
                                                          return_samples=True)
        plt.scatter(ss[:, 0], ss[:, 1], s=5)
        plt.scatter(clus[:, 0], clus[:, 1], s=10, c='red')
        plt.scatter(ktilde, utilde, s=20, c='magenta')
        adjusted_cl = np.asarray([adjust_centroid(gp_iter, cl_, np.exp(ql_array.min(0)[i]))[0]
                                  for cl_ in clus])
        cl.append(adjusted_cl)
        plt.scatter(adjusted_cl[:, 0], adjusted_cl[:, 1], s=10, c='g')
    plt.subplot(2, 2, 4)
    plt.plot(ql_array)
    cl_array = np.asarray(cl).reshape(-1, 2)
    to_add = np.vstack([cl_array, np.atleast_2d([ktilde, utilde[0]])])
    gp_iter = bow.add_points_to_design(gp_iter, to_add, function_2d(to_add), optimize_cov=True)
    plt.tight_layout()
    plt.savefig('/home/victor/Bureau/iter_{}.png'.format(j))
    plt.close()

plt.plot(gp_iter.X_train_[30:, 0], gp_iter.X_train_[30:, 1], '.')
plt.plot(gp.X_train_[:, 0], gp.X_train_[:, 1], 'r.')
plt.show()

plt.scatter(cl_array[:, 0], cl_array[:, 1])

adjusted_cl = []
adjusted_cl = np.asarray([adjust_centroid(gp, cl_, np.exp(ql_array.min(0)[0]))[0]
                          for cl_ in cl_array])





def Geary_Hinkley(sample, m, mstar, sig2, sigstar2, rho):
    num = mstar * sample - m
    den = np.sqrt(sigstar2 * sample**2 - 2 * rho * np.sqrt(sig2) * np.sqrt(sigstar2) * sample + sig2)
    return num / den


def Geary_Hinkley_direct(gp, comb, Nsamples):
    samples = np.empty((Nsamples, len(comb)))
    samples_transformed = np.empty((Nsamples, len(comb)))
    for j, aa in progressbar(list(enumerate(sample_y_modified(gp, comb, Nsamples))),
                             'generate samples'):
        samples[j, :] = aa
    mc_tu = bow.mean_covariance_alpha_vector(gp, comb, [1], np.atleast_2d([0, 1]))
    for j, sa in progressbar(list(enumerate(samples.T))):
        m, mstar = mc_tu[j][0]
        sig2, sigstar2 = mc_tu[j][1][0, 0], mc_tu[j][1][1, 1]
        rho = mc_tu[j][1][1, 0] / np.sqrt(sigstar2) * np.sqrt(sig2)
        samples_transformed[:, j] = Geary_Hinkley(sa, m, mstar, sig2, sigstar2, rho)
    return samples_transformed, samples


PI = np.asarray([compute_plugin(mc) for mc in mc_tu]).reshape(ngrid, ngrid)
PI_nc = np.asarray([compute_plugin(mc, False) for mc in mc_tu]).reshape(ngrid, ngrid)
LBUB = np.quantile(np.asarray([compute_lb_ub(mc) for mc in mc_tu]).reshape(ngrid, ngrid, 2), p,  axis=1)

mlogratio, vlogratio = mean_variance_logratio(all_combinations)
LB, UB = (mlogratio + scipy.stats.norm.ppf(0.025) * np.sqrt(vlogratio),
          mlogratio + scipy.stats.norm.ppf(0.975) * np.sqrt(vlogratio))

plt.subplot(2, 2, 1)
plt.contourf(X_mg, Y_mg, mlogratio.reshape(ngrid, ngrid))
plt.colorbar()
plt.subplot(2, 2, 2)
plt.contourf(X_mg, Y_mg, vlogratio.reshape(ngrid, ngrid))
plt.colorbar()
plt.subplot(2, 2, 3)
plt.contourf(X_mg, Y_mg, LB.reshape(ngrid, ngrid))
plt.colorbar()
plt.subplot(2, 2, 4)
plt.contourf(X_mg, Y_mg, UB.reshape(ngrid, ngrid))
plt.colorbar()
plt.show()

plt.plot(vlogratio.reshape(100, 100).mean(1))
plt.plot((LB-UB).reshape(100, 100).mean(1))
plt.show()


np.exp(np.quantile(mlogratio.reshape(ngrid, ngrid), 0.95, 1))
qPI = np.quantile(PI, p, axis=1)
qPI_nc = np.quantile(PI_nc, p, axis=1)

plt.figure(figsize=col_full)
for al in alpha_p_samples:
    plt.plot(X_, al, alpha=0.05, color='grey')
plt.plot(np.nan, np.nan, alpha=0.1, color='grey', label=r'$q_p^{(i)}$')
np.log(alpha_p_samples)
qMC = alpha_p_samples.mean(0)
qMC2 = np.exp(np.log(alpha_p_samples).mean(0))

qSDm = np.quantile(alpha_p_samples, 0.025, axis=0)  # alpha_p_samples.std(0)
qSDp = np.quantile(alpha_p_samples, 0.975, axis=0)  # alpha_p_samples.std(0)

# plt.plot(X_, qMC, color=colors[0], lw=2, label=r'$q_p^{\mathrm{MC}}$ ar')
# plt.plot(X_, qMC2, color=colors[4], lw=2, label=r'$q_p^{\mathrm{MC}}$ geo')
# plt.plot(X_, qSDm, color=colors[0], lw=1, ls=':', label=r'$q_p^{\mathrm{MC}} \pm 1$ s.d.')
# plt.plot(X_, qSDp, color=colors[0], lw=1, ls=':')
plt.plot(X_, qPI, color=colors[1], lw=2, label=r'$q_p^{\mathrm{PI}}$')
# plt.plot(X_, qPI_nc, color=colors[3], lw=2, label=r'$q_p^{\mathrm{PI}}$ no correct')
plt.title(r'Estimation of $q_p$ based on the GP $Z$')
plt.plot(X_, LBUB, color='red', label=r'LBUB')
# plt.title(r'MC: {}, PI: {}'.format(alpha_p_samples.mean(0).min(),
#                                    qPI.min()))
plt.ylim([1.0, 8])
plt.plot(X_l, alpha_t, color=colors[2], label=r'Truth: $q_p$')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$q_p$')
plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig('/home/victor/acadwriting/Manuscrit/Text/Chapter4/img/quantile_estimation_GP.pgf')
plt.close()



mc_tu = bow.mean_covariance_alpha_vector(gp, all_combinations, [1], np.atleast_2d([0, 1]))


def compute_plugin_notuple(m1, m2, s1, s2, correc=True):
    if correc:
        return (m1 / m2) * np.exp((s1 / (2 * (m1**2))) + (s2 / (2 * (m2**2))))
    else:
        return (m1 / m2)

plugin = np.asarray([compute_plugin(tu) for tu in mc_tu])
plugin_f = np.asarray([compute_plugin(tu, False) for tu in mc_tu])
plt.plot(plugin / plugin_f)
plt.show()

samples = np.empty((20, 20, 5000))
for i, sam in enumerate(sample_y_modified(gp, all_combinations, 5000)):
    curr = sam.reshape(20, 20)
    Jstar = curr.min(0)
    rho = (curr / Jstar[np.newaxis, :])
    samples[:, :, i] = rho

plt.plot(samples.mean(2).reshape(400), label='MC')
plt.plot(plugin, label='correc')
plt.plot(plugin_f, label='f')
plt.legend()
plt.show()



nU = 100
lhsU_small = pyDOE.lhs(1, samples=nU, criterion='maximin', iterations=50)
mst, sigst = np.empty(len(lhsU)), np.empty(len(lhsU))
for i, u in progressbar(list(enumerate(lhsU))):
    mst[i], sigst[i] = bow.mu_sigma_star(gp, u, [1], np.atleast_2d([0, 1]))
qu = []


def compute_qPI(k, gp, p, lhsU, mst, sigst):
    qu = []
    if len(k) > 10:
        gen = progressbar(k)
    else:
        gen = k
    for k_ in gen:
        pts = np.hstack([k_ * np.ones_like(lhsU), lhsU])
        me, st = gp.predict(pts, return_std=True)
        qu.append(
            np.quantile(np.asarray([compute_plugin_notuple(me[i], mst[i], st[i]**2, sigst[i]**2)
                                    for i in range(len(lhsU))]), p))
    return np.asarray(qu)


def compute_qPI_bounds(k, gp, p, lhsU):
    qu = np.empty((len(k), 2))
    if len(k) > 10:
        gen = enumerate(list(progressbar(k)))
    else:
        gen = enumerate(k)
    for j, k_ in gen:
        ku = np.hstack([k_ * np.ones_like(lhsU), lhsU])
        mc_tu = bow.mean_covariance_alpha_vector(gp, ku,[1], np.atleast_2d([0, 1]))
        qu[j, :] = np.quantile(#
            np.asarray([compute_lb_ub(mc) for mc in mc_tu]), p,  axis=0)
    return qu


plt.plot(k, qu)
plt.plot(X_l, alpha_t)
plt.plot(X_, qPI)
plt.show()
Niterations = 5
counter_iter = 0
best_val = np.inf
while counter_iter < Niterations:
    x0 = np.random.uniform(0, 1)
    opt = scipy.optimize.minimize(compute_qPI, x0, args=(0.95, mst, sigst),
                                  bounds = np.atleast_2d([0, 1]))
    if opt.fun < best_val:
        best = opt
        best_val = best.fun
    counter_iter += 1
    print(best)


def reliability_ratio(gp, points, val):
    mc_tu = mean_covariance_alpha_vector(gp, points, [1], np.atleast_2d([0, 1]))
    ratio_mean = np.asarray([me[0] / me[1] for me, _ in mc_tu])
    ratio_si = np.asarray([co[0, 0] / me[0]**2 + co[1, 1] / me[1]**2 for me, co in mc_tu])
    return np.abs(ratio_mean - val) / np.sqrt(ratio_si)



def reliability_ratio_on_slice(u, gp, kf, val):
    pts = np.vstack([kf * np.ones_like(u), u]).T
    return reliability_ratio(gp, pts, val)


def optimize_reliability_ratio_slice(gp, k, val):
    Niterations = 5
    counter_iter = 0
    best_val = np.inf
    while counter_iter < Niterations:
        x0 = np.random.uniform(0, 1)
        opt = scipy.optimize.minimize(reliability_ratio_on_slice, x0, args=(gp, k, val),
                                      bounds = np.atleast_2d([0, 1]))
        if opt.fun < best_val:
            best = opt
            best_val = best.fun
        counter_iter += 1
    return best.x


def optim_qPI(gp, p, Niterations, mst, sigst, verbose=True):
    counter_iter = 0
    best_val = np.inf
    while counter_iter < Niterations:
        x0 = np.random.uniform(0, 1)
        opt = scipy.optimize.minimize(compute_qPI, x0, args=(gp, p, lhsU, mst, sigst),
                                      bounds = np.atleast_2d([0, 1]))
        if opt.fun < best_val:
            best = opt
            best_val = best.fun
        counter_iter += 1
    if verbose:
        print('min q_p: {}'.format(best.fun))
        print('diff with true {}'.format(np.abs(best.fun - alpha_t.min())))
    return best


def optim_qPI_LBUB(gp, p, Niterations, lhsU_small, verbose=True, LB=True):
    counter_iter = 0
    best_val = np.inf
    if LB is True:
        idxlb = 0
    else:
        idxlb = 1
    while counter_iter < Niterations:
        x0 = np.random.uniform(0, 1)
        to_min = lambda x: compute_qPI_bounds(x, gp, p, lhsU_small)[:, idxlb]
        opt = scipy.optimize.minimize(to_min, x0,
                                      bounds = np.atleast_2d([0, 1]))
        if opt.fun < best_val:
            best = opt
            best_val = best.fun
        counter_iter += 1
    if verbose:
        print('LB q_p: {}'.format(best.fun))
        # print('diff with true {}'.format(np.abs(best.fun - alpha_t.min())))
    return best


gp_ = copy.copy(gp)
for i in progressbar(range(5)):
    LB = optim_qPI_LBUB(gp_, p, 2, lhsU_small)
    print(LB.x)
    mst, sigst = np.empty(len(lhsU)), np.empty(len(lhsU))
    for i, u in progressbar(list(enumerate(lhsU)), 'init Z*'):
        mst[i], sigst[i] = bow.mu_sigma_star(gp_, u, [1], np.atleast_2d([0, 1]))
    val = compute_qPI(LB.x, gp_, p, lhsU, mst, sigst)
    print(val)
    u = optimize_reliability_ratio_slice(gp_, LB.x, val)
    newku = np.asarray([LB.x, u]).squeeze()
    gp_ = bow.add_points_to_design(gp_, newku, function_2d(newku), True)



def iteration_opt(gp, p, lhsU, mst, sigst):
    Niterations = 5
    counter_iter = 0
    best_val = np.inf
    while counter_iter < Niterations:
        x0 = np.random.uniform(0, 1)
        opt = scipy.optimize.minimize(compute_qPI, x0, args=(gp_, 0.95, mst, sigst),
                                      bounds = np.atleast_2d([0, 1]))
        if opt.fun < best_val:
            best = opt
            best_val = best.fun
        counter_iter += 1
    u = optimize_reliability_ratio_slice(gp, best.x, best.fun)
    return np.asarray([best.x, u]).squeeze()

nU = 1000
lhsU = pyDOE.lhs(1, samples=nU, criterion='maximin', iterations=50)
mst, sigst = np.empty(len(lhsU)), np.empty(len(lhsU))
for i, u in progressbar(list(enumerate(lhsU)), 'init Z*'):
    mst[i], sigst[i] = bow.mu_sigma_star(gp, u, [1], np.atleast_2d([0, 1]))



def iteration_MC(gp, p, lhsU, mst, sigst):
    # mst, sigst = np.empty(len(lhsU)), np.empty(len(lhsU))
    # for i, u in progressbar(list(enumerate(lhsU)), 'init Z*'):
    #     mst[i], sigst[i] = bow.mu_sigma_star(gp, u, [1], np.atleast_2d([0, 1]))

    Niterations = 5
    counter_iter = 0
    best_val = np.inf
    while counter_iter < Niterations:
        x0 = np.random.uniform(0, 1)
        opt = scipy.optimize.minimize(compute_qPI, x0, args=(gp, 0.95, mst, sigst),
                                      bounds = np.atleast_2d([0, 1]))
        if opt.fun < best_val:
            best = opt
            best_val = best.fun
        counter_iter += 1
    print('min q_p: {}'.format(best.fun))
    print('diff with true {}'.format(np.abs(best.fun - alpha_t.min())))
    # candidates = np.random.uniform(size=2 * 500).reshape(500, 2)
    candidates = np.random.uniform(size=200 * 2).reshape(200, 2)
    rel = reliability_ratio(gp, candidates, best.fun)
    return np.asarray([candidates[rel.argmin()]]).squeeze()












gp_ = copy.copy(gp)
for i in progressbar(range(70)):
    newku = iteration_MC(gp_, 0.95, lhsU, mst, sigst)
    print(newku)
    gp_ = bow.add_points_to_design(gp_, newku, function_2d(newku), True)



plt.plot(gp.X_train_[:, 0], gp.X_train_[:, 1], '.')
plt.plot(gp_.X_train_[30:, 0], gp_.X_train_[30:, 1], '.')
plt.show()
plt.plot(X_l, alpha_t, label='truth')
mst, sigst = np.empty(len(lhsU)), np.empty(len(lhsU))
for i, u in progressbar(list(enumerate(lhsU))):
    mst[i], sigst[i] = bow.mu_sigma_star(gp, u, [1], np.atleast_2d([0, 1]))
plt.plot(X_l, compute_qPI(X_l, gp, 0.95, lhsU, mst, sigst), label='init')
mst, sigst = np.empty(len(lhsU)), np.empty(len(lhsU))
for i, u in progressbar(list(enumerate(lhsU))):
    mst[i], sigst[i] = bow.mu_sigma_star(gp_, u, [1], np.atleast_2d([0, 1]))
plt.plot(X_l, compute_qPI(X_l, gp_, 0.95, lhsU, mst, sigst), label='after')
plt.legend()
plt.show()




plt.subplot(2, 1, 1)
mc_tu = bow.mean_covariance_alpha_vector(gp, all_combinations, [1], np.atleast_2d([0, 1]))
Nsamples = 1000
rh = np.empty((ngrid, ngrid, Nsamples))
alpha_p_samples = np.empty((Nsamples, ngrid))
for j, aa in progressbar(list(enumerate(sample_y_modified(gp, all_combinations, Nsamples)))):
    # print '{}\r'.format(j, Nsamples),
    curr = aa.reshape(ngrid, ngrid)
    Jstar = curr.min(0)
    rho = (curr / Jstar[np.newaxis, :])
    rh[:, :, j] = rho
    alpha_p = np.quantile(rho, p, axis=1)
    alpha_p_samples[j, :] = alpha_p

PI = np.asarray([compute_plugin(mc) for mc in mc_tu]).reshape(ngrid, ngrid)
PI_nc = np.asarray([compute_plugin(mc, False) for mc in mc_tu]).reshape(ngrid, ngrid)
qPI = np.quantile(PI, p, axis=1)
qPI_nc = np.quantile(PI_nc, p, axis=1)

for al in alpha_p_samples:
    plt.plot(X_, al, alpha=0.05, color='grey')
plt.plot(np.nan, np.nan, alpha=0.1, color='grey', label=r'$q_p^{(i)}$')
np.log(alpha_p_samples)
qMC = alpha_p_samples.mean(0)
qMC2 = np.exp(np.log(alpha_p_samples).mean(0))

qSDm = np.quantile(alpha_p_samples, 0.025, axis=0)  # alpha_p_samples.std(0)
qSDp = np.quantile(alpha_p_samples, 0.975, axis=0)  # alpha_p_samples.std(0)

plt.plot(X_, qMC, color=colors[0], lw=2, label=r'$q_p^{\mathrm{MC}}$ ar')
plt.plot(X_, qMC2, color=colors[4], lw=2, label=r'$q_p^{\mathrm{MC}}$ geo')
plt.plot(X_, qSDm, color=colors[0], lw=1, ls=':', label=r'$q_p^{\mathrm{MC}} \pm 1$ s.d.')
plt.plot(X_, qSDp, color=colors[0], lw=1, ls=':')
plt.plot(X_, qPI, color=colors[1], lw=2, label=r'$q_p^{\mathrm{PI}}$')
plt.plot(X_, qPI_nc, color=colors[3], lw=2, label=r'$q_p^{\mathrm{PI}}$ no correct')
plt.title(r'Estimation of $q_p$ based on the GP $Z$')
# plt.title(r'MC: {}, PI: {}'.format(alpha_p_samples.mean(0).min(),
#                                    qPI.min()))
plt.ylim([1.0, 8])
plt.plot(X_l, alpha_t, color=colors[2], label=r'Truth: $q_p$')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$q_p$')
plt.legend()
plt.show()
plt.subplot(2, 1, 2)

mc_tu = bow.mean_covariance_alpha_vector(gp_, all_combinations, [1], np.atleast_2d([0, 1]))
Nsamples = 1000
rh = np.empty((ngrid, ngrid, Nsamples))
alpha_p_samples = np.empty((Nsamples, ngrid))
for j, aa in progressbar(list(enumerate(sample_y_modified(gp_, all_combinations, Nsamples)))):
    # print '{}\r'.format(j, Nsamples),
    curr = aa.reshape(ngrid, ngrid)
    Jstar = curr.min(0)
    rho = (curr / Jstar[np.newaxis, :])
    rh[:, :, j] = rho
    alpha_p = np.quantile(rho, p, axis=1)
    alpha_p_samples[j, :] = alpha_p

PI = np.asarray([compute_plugin(mc) for mc in mc_tu]).reshape(ngrid, ngrid)
PI_nc = np.asarray([compute_plugin(mc, False) for mc in mc_tu]).reshape(ngrid, ngrid)
qPI = np.quantile(PI, p, axis=1)
qPI_nc = np.quantile(PI_nc, p, axis=1)

for al in alpha_p_samples:
    plt.plot(X_, al, alpha=0.05, color='grey')
plt.plot(np.nan, np.nan, alpha=0.1, color='grey', label=r'$q_p^{(i)}$')
np.log(alpha_p_samples)
qMC = alpha_p_samples.mean(0)
qMC2 = np.exp(np.log(alpha_p_samples).mean(0))

qSDm = np.quantile(alpha_p_samples, 0.025, axis=0)  # alpha_p_samples.std(0)
qSDp = np.quantile(alpha_p_samples, 0.975, axis=0)  # alpha_p_samples.std(0)

plt.plot(X_, qMC, color=colors[0], lw=2, label=r'$q_p^{\mathrm{MC}}$ ar')
plt.plot(X_, qMC2, color=colors[4], lw=2, label=r'$q_p^{\mathrm{MC}}$ geo')
plt.plot(X_, qSDm, color=colors[0], lw=1, ls=':', label=r'$q_p^{\mathrm{MC}} \pm 1$ s.d.')
plt.plot(X_, qSDp, color=colors[0], lw=1, ls=':')
plt.plot(X_, qPI, color=colors[1], lw=2, label=r'$q_p^{\mathrm{PI}}$')
plt.plot(X_, qPI_nc, color=colors[3], lw=2, label=r'$q_p^{\mathrm{PI}}$ no correct')
plt.title(r'Estimation of $q_p$ based on the GP $Z$')
# plt.title(r'MC: {}, PI: {}'.format(alpha_p_samples.mean(0).min(),
#                                    qPI.min()))
plt.ylim([1.0, 8])
plt.plot(X_l, alpha_t, color=colors[2], label=r'Truth: $q_p$')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$q_p$')
plt.legend()
plt.show()




def margin_indicator_delta(gp, x, T, eta=0.025):
    return bow.margin_indicator(bow.mu_sigma_delta(gp,
                                                   x,
                                                   T,
                                                   [1],
                                                   np.asarray([0, 1]),
                                                   verbose=False),
                                0, 1 - eta, x)



plugin = np.empty((7, ngrid, ngrid))
ppi = np.empty((7, ngrid, ngrid))
import scipy.cluster
gp_tmp = copy.copy(gp)
for i in range(7):
    print(i)
    plt.subplot(3, 3, i + 1)
    # Estimation of threshold
    mst, sigst = np.empty(len(lhsU)), np.empty(len(lhsU))
    for j, u in progressbar(list(enumerate(lhsU)), 'Zstar'):
        mst[j], sigst[j] = bow.mu_sigma_star(gp_tmp, u, [1], np.atleast_2d([0, 1]))
    # be = optim_qPI(gp_tmp, 0.95, 5, mst, sigst, True)
    threshold = 1.8
    
    m, s = bow.mu_sigma_delta(gp_tmp, all_combinations, threshold, [1], np.asarray([0, 1]))
    plugin[i, :, :] = m.reshape(ngrid, ngrid)
    ppi[i, :, :] = bow.coverage_probability((m, s), 0, None).reshape(ngrid, ngrid)

    plt.contourf(X_mg, Y_mg, gp_tmp.predict(all_combinations).reshape(ngrid, ngrid))
    plt.contour(X_mg, Y_mg,
                bow.margin_indicator((m, s), 0, 1 - 0.025, all_combinations).reshape(ngrid, ngrid),
                levels=[0.45, 0.55])
    samples_margin = bow.sample_from_criterion(1000,
                                               lambda x: margin_indicator_delta(gp_tmp,
                                                                                x, threshold),
                                               bounds=np.asarray([[0, 1],
                                                                  [0, 1]]),
                                               Ncandidates=3)
    kmeans = bow.cluster_and_find_closest(10, samples_margin)
    var_list = []
    kadj = np.empty((len(kmeans[0]), 2))
    for i, km in enumerate(kmeans[0]):
        kadj[i, :], (s2, alp_s2star) = adjust_centroid(gp_tmp, km, threshold)
        var_list.append((s2, alp_s2star))
        if np.any(kadj[i, :] != km):
            dx, dy = kadj[i, :] - km
                # plt.arrow(km[0], km[1], dx, dy, **opt)

    var_list = np.asarray(var_list)
    kadj2 = np.copy(kmeans[0])
    hier_clusters = scipy.cluster.hierarchy.fclusterdata(kadj, 0.3)
    for cluster_index in np.unique(hier_clusters):
        cl = np.asarray(hier_clusters == cluster_index).nonzero()
        print(len(cl[0]))
        to_adjust = var_list[cl][:, 0].argmin()
        kadj2[cl[0][to_adjust]] = kadj[cl[0][to_adjust]]

    plt.plot(gp_tmp.X_train_[:, 0], gp_tmp.X_train_[:, 1], 'w.')
    plt.plot(centroids[:, 0], centroids[:, 1], 'r.')
    gp_tmp = bow.add_points_to_design(gp_tmp, kadj2, function_2d(kadj2), True)







integration_points_1D = pyDOE.lhs(1, 20, criterion='maximin', iterations=50)
gp_ = copy.copy(gp)
for i in range(1, 20):
    mst, sigst = np.empty(len(lhsU)), np.empty(len(lhsU))
    for j, u in progressbar(list(enumerate(lhsU)), 'Zstar'):
        mst[j], sigst[j] = bow.mu_sigma_star(gp_, u, [1], np.atleast_2d([0, 1]))
    
    LB = optim_qPI_LBUB(gp_, p, 2, lhsU_small)
    qp = compute_qPI(LB.x, gp_, 0.95, lhsU, mst, sigst)
    threshold = qp
    int_points = np.hstack([np.ones_like(integration_points_1D) * LB.x, integration_points_1D])
    gp_ = bow.template(gp_=gp_,
                       true_function=function_2d,
                       acquisition_fun=lambda g: acquisition_IMSE(g, alpha=threshold,
                                                                  integration_points=int_points),
                       criterion_fun=lambda g, X: np.ones_like(X),
                       prefix='augmentedIMSE',
                       X_=None,
                       niterations=1,
                       plot=False,
                       bounds=np.asarray([[0, 1],
                                          [0, 1]]),
                       nrestart=2, save=False
    )


plt.plot(gp_.X_train_[:30, 0], gp_.X_train_[:30, 1], '.')
plt.plot(gp_.X_train_[30:, 0], gp_.X_train_[30:, 1], '.')
plt.show()
plt.plot(X_l, alpha_t, label='truth')
mst, sigst = np.empty(len(lhsU)), np.empty(len(lhsU))
for i, u in progressbar(list(enumerate(lhsU))):
    mst[i], sigst[i] = bow.mu_sigma_star(gp, u, [1], np.atleast_2d([0, 1]))
plt.plot(X_l, compute_qPI(X_l, gp, 0.95, lhsU, mst, sigst), label='init')
mst, sigst = np.empty(len(lhsU)), np.empty(len(lhsU))
for i, u in progressbar(list(enumerate(lhsU))):
    mst[i], sigst[i] = bow.mu_sigma_star(gp_, u, [1], np.atleast_2d([0, 1]))
plt.plot(X_l, compute_qPI(X_l, gp_, 0.95, lhsU, mst, sigst), label='after')
plt.legend()
plt.show()















lhs_croco = pyDOE.lhs(6, samples=600, criterion='maximin', iterations=100)
lhs_croco[:, :4] = (lhs_croco[:, :4] - 5e-3) / 10e-3
lhs_croco[:, :4] = lhs_croco[:, :4] * (13e-3 - 8e-3) + 8e-3
lhs_croco.min(0), lhs_croco.max(0)
np.savetxt("/home/victor/croco_dahu2/LHS_shallow.csv", lhs_croco, delimiter=",")
lhs_croco = np.genfromtxt("/home/victor/croco_dahu2/LHS_shallow.csv", delimiter=",")

# EOF ----------------------------------------------------------------------
