# coding: utf-8
#!/usr/bin/env python


import numpy as np
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from matplotlib import cm
from matplotlib import pyplot as plt
import itertools
from sklearn.gaussian_process.kernels import Matern
import scipy
from acquisition_function import expected_improvement_closed_form
