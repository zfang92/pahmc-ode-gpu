# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This is a unit test. If you would like to further develop pahmc_ode_gpu, you 
should visit here frequently.
"""


import os
from pathlib import Path

from numba import cuda, jit
import numpy as np
import torch as th

os.chdir(Path.cwd().parent)
from pahmc_ode_gpu import cuda_lib_dynamics
os.chdir(Path.cwd()/'unit_tests')


"""Prepare data, as well as variables to be compared to."""
name = 'nakl'

D = 4
M = 100000

X = np.concatenate((np.random.uniform(-100.0, 50.0, (1,M)),
                    np.random.uniform(0.0, 1.0, (D-1,M))))
par = np.array([120.0, 50.0, 20.0, -77.0, 0.3, -54.4, -40.0, 15, 
                0.1, 0.4, -60.0, -15, 1.0, 7.0, -55.0, 30, 1.0, 5.0])
stimulus \
  = np.concatenate((np.random.uniform(-30, 30, (1,M)), np.zeros((D-1,M))))

# this function has been tested in pahmc_ode_cpu
@jit(nopython=True)
def cpu_field(X, par, stimulus):
    (D, M) = np.shape(X)
    vecfield = np.zeros((D,M))

    vecfield[0, :] \
      = stimulus[0, :] \
        + par[0] * (X[1, :] ** 3) * X[2, :] * (par[1] - X[0, :]) \
        + par[2] * (X[3, :] ** 4) * (par[3] - X[0, :]) \
        + par[4] * (par[5] - X[0, :])

    tanh_m = np.tanh((X[0, :]-par[6])/par[7])
    eta_m = 1 / 2 * (1 + tanh_m)
    tau_m = par[8] + par[9] * (1 - tanh_m * tanh_m)
    vecfield[1, :] = (eta_m - X[1, :]) / tau_m

    tanh_h = np.tanh((X[0, :]-par[10])/par[11])
    eta_h = 1 / 2 * (1 + tanh_h)
    tau_h = par[12] + par[13] * (1 - tanh_h * tanh_h)
    vecfield[2, :] = (eta_h - X[2, :]) / tau_h

    tanh_n = np.tanh((X[0, :]-par[14])/par[15])
    eta_n = 1 / 2 * (1 + tanh_n)
    tau_n = par[16] + par[17] * (1 - tanh_n * tanh_n)
    vecfield[3, :] = (eta_n - X[3, :]) / tau_n
    
    return vecfield

print('\nTesting... ', end='')
field_compared = cpu_field(X, par, stimulus)

# let's tell PyTorch about our model in order to test jacobian and dfield_dpar
X = th.from_numpy(X)
par = th.from_numpy(par)
stimulus = th.from_numpy(stimulus)

X.requires_grad = True
par.requires_grad = True

vecfield = th.zeros(D, M)
vecfield[0, :] \
  = stimulus[0, :] \
    + par[0] * (X[1, :] ** 3) * X[2, :] * (par[1] - X[0, :]) \
    + par[2] * (X[3, :] ** 4) * (par[3] - X[0, :]) \
    + par[4] * (par[5] - X[0, :])

tanh_m = th.tanh((X[0, :]-par[6])/par[7])
eta_m = 1 / 2 * (1 + tanh_m)
tau_m = par[8] + par[9] * (1 - tanh_m * tanh_m)
vecfield[1, :] = (eta_m - X[1, :]) / tau_m

tanh_h = th.tanh((X[0, :]-par[10])/par[11])
eta_h = 1 / 2 * (1 + tanh_h)
tau_h = par[12] + par[13] * (1 - tanh_h * tanh_h)
vecfield[2, :] = (eta_h - X[2, :]) / tau_h

tanh_n = th.tanh((X[0, :]-par[14])/par[15])
eta_n = 1 / 2 * (1 + tanh_n)
tau_n = par[16] + par[17] * (1 - tanh_n * tanh_n)
vecfield[3, :] = (eta_n - X[3, :]) / tau_n

# fetch the variables to be compared to
scalarfield = th.sum(vecfield)
scalarfield.backward()

jacobian_compared = X.grad.numpy()
dfield_dpar_compared = par.grad.numpy()

X = X.detach().numpy()
par = par.detach().numpy()
stimulus = stimulus.numpy()


"""Fetch the kernels, transfer data, and specify grid dimensions."""
k__field = getattr(cuda_lib_dynamics, f'k__{name}_field')
k__jacobian = getattr(cuda_lib_dynamics, f'k__{name}_jacobian')
k__dfield_dpar = getattr(cuda_lib_dynamics, f'k__{name}_dfield_dpar')

d_X = cuda.to_device(X)
d_par = cuda.to_device(par)
d_stimulus = cuda.to_device(stimulus)

d_field = cuda.to_device(np.zeros((D,M)))
d_jacobian = cuda.to_device(np.zeros((D,D,M)))
d_dfield_dpar = cuda.to_device(np.zeros((D,len(par),M)))


"""Define convenience functions."""
def gtimer1():
    k__field[(16,32), (2,128)](d_X, d_par, d_stimulus, d_field)
    cuda.synchronize()

def gtimer2():
    k__jacobian[(4,4,32), (2,2,64)](d_X, d_par, d_jacobian)
    cuda.synchronize()

def gtimer3():
    k__dfield_dpar[(4,4,32), (2,2,64)](d_X, d_par, d_dfield_dpar)
    cuda.synchronize()

def gtimer4():
    gtimer1(); gtimer2(); gtimer3()

def gtimer5():
    k__field[(16,32), (2,128)](d_X, d_par, d_stimulus, d_field)
    k__jacobian[(4,4,32), (2,2,64)](d_X, d_par, d_jacobian)
    k__dfield_dpar[(4,4,32), (2,2,64)](d_X, d_par, d_dfield_dpar)
    cuda.synchronize()


"""Make sure everything is correct."""
gtimer5()

field = d_field.copy_to_host()
jacobian = np.sum(d_jacobian.copy_to_host(), axis=0)
dfield_dpar = np.sum(d_dfield_dpar.copy_to_host(), axis=(0,2))

np.testing.assert_almost_equal(field, field_compared, decimal=6)
np.testing.assert_almost_equal(jacobian, jacobian_compared, decimal=6)
np.testing.assert_almost_equal(dfield_dpar, dfield_dpar_compared, decimal=6)
print('ok.')


#======================================================================
# for profiling only
@jit(nopython=True)
def cpu_jacobian(X, par):
    (D, M) = np.shape(X)
    jacob = np.zeros((D,D,M))

    jacob[0, 0, :] = - par[0] * (X[1, :] ** 3) * X[2, :] \
                     - par[2] * (X[3, :] ** 4) - par[4]

    jacob[0, 1, :] \
      = 3 * par[0] * (X[1, :] ** 2) * X[2, :] * (par[1] - X[0, :])

    jacob[0, 2, :] = par[0] * (X[1, :] ** 3) * (par[1] - X[0, :])

    jacob[0, 3, :] = 4 * par[2] * (X[3, :] ** 3) * (par[3] - X[0, :])

    tanh_m = np.tanh((X[0, :]-par[6])/par[7])
    kernel_m = (1 - tanh_m * tanh_m)
    eta_m = 1 / 2 * (1 + tanh_m)
    tau_m = par[8] + par[9] * kernel_m
    eta_der_m = 1 / (2 * par[7]) * kernel_m
    tau_der_m = - 2 * par[9] / par[7] * tanh_m * kernel_m
    jacob[1, 0, :] \
      = eta_der_m / tau_m + tau_der_m * (X[1, :] - eta_m) / (tau_m * tau_m)

    tanh_h = np.tanh((X[0, :]-par[10])/par[11])
    kernel_h = (1 - tanh_h * tanh_h)
    eta_h = 1 / 2 * (1 + tanh_h)
    tau_h = par[12] + par[13] * kernel_h
    eta_der_h = 1 / (2 * par[11]) * kernel_h
    tau_der_h = - 2 * par[13] / par[11] * tanh_h * kernel_h
    jacob[2, 0, :] \
      = eta_der_h / tau_h + tau_der_h * (X[2, :] - eta_h) / (tau_h * tau_h)

    tanh_n = np.tanh((X[0, :]-par[14])/par[15])
    kernel_n = (1 - tanh_n * tanh_n)
    eta_n = 1 / 2 * (1 + tanh_n)
    tau_n = par[16] + par[17] * kernel_n
    eta_der_n = 1 / (2 * par[15]) * kernel_n
    tau_der_n = - 2 * par[17] / par[15] * tanh_n * kernel_n
    jacob[3, 0, :] \
      = eta_der_n / tau_n + tau_der_n * (X[3, :] - eta_n) / (tau_n * tau_n)

    jacob[1, 1, :] = - 1 / tau_m

    jacob[2, 2, :] = - 1 / tau_h

    jacob[3, 3, :] = - 1 / tau_n
    
    return jacob

@jit(nopython=True)
def cpu_dfield_dpar(X, par):
    (D, M) = np.shape(X)
    deriv_par = np.zeros((D,M,len(par)))

    deriv_par[0, :, 0] = (X[1, :] ** 3) * X[2, :] * (par[1] - X[0, :])

    deriv_par[0, :, 1] = par[0] * (X[1, :] ** 3) * X[2, :]

    deriv_par[0, :, 2] = (X[3, :] ** 4) * (par[3] - X[0, :])

    deriv_par[0, :, 3] = par[2] * (X[3, :] ** 4)

    deriv_par[0, :, 4] = par[5] - X[0, :]

    deriv_par[0, :, 5] = par[4]

    tanh_m = np.tanh((X[0, :]-par[6])/par[7])
    kernel_m = (1 - tanh_m * tanh_m)
    eta_m = 1 / 2 * (1 + tanh_m)
    tau_m = par[8] + par[9] * kernel_m
    common_m = (X[1, :] - eta_m) / (tau_m * tau_m)
    eta_der_m = - 1 / (2 * par[7]) * kernel_m
    tau_der_m = 2 * par[9] / par[7] * tanh_m * kernel_m
    deriv_par[1, :, 6] = eta_der_m / tau_m + tau_der_m * common_m

    eta_der_m = - (X[0, :] - par[6]) / (2 * (par[7] ** 2)) * kernel_m
    tau_der_m = 2 * par[9] * (X[0, :] - par[6]) / (par[7] ** 2) \
                * tanh_m * kernel_m
    deriv_par[1, :, 7] = eta_der_m / tau_m + tau_der_m * common_m

    deriv_par[1, :, 8] = common_m

    deriv_par[1, :, 9] = kernel_m * common_m

    tanh_h = np.tanh((X[0, :]-par[10])/par[11])
    kernel_h = (1 - tanh_h * tanh_h)
    eta_h = 1 / 2 * (1 + tanh_h)
    tau_h = par[12] + par[13] * kernel_h
    common_h = (X[2, :] - eta_h) / (tau_h * tau_h)
    eta_der_h = - 1 / (2 * par[11]) * kernel_h
    tau_der_h = 2 * par[13] / par[11] * tanh_h * kernel_h
    deriv_par[2, :, 10] = eta_der_h / tau_h + tau_der_h * common_h

    eta_der_h = - (X[0, :] - par[10]) / (2 * (par[11] ** 2)) * kernel_h
    tau_der_h = 2 * par[13] * (X[0, :] - par[10]) / (par[11] ** 2) \
                * tanh_h * kernel_h
    deriv_par[2, :, 11] = eta_der_h / tau_h + tau_der_h * common_h

    deriv_par[2, :, 12] = common_h

    deriv_par[2, :, 13] = kernel_h * common_h

    tanh_n = np.tanh((X[0, :]-par[14])/par[15])
    kernel_n = (1 - tanh_n * tanh_n)
    eta_n = 1 / 2 * (1 + tanh_n)
    tau_n = par[16] + par[17] * kernel_n
    common_n = (X[3, :] - eta_n) / (tau_n * tau_n)
    eta_der_n = - 1 / (2 * par[15]) * kernel_n
    tau_der_n = 2 * par[17] / par[15] * tanh_n * kernel_n
    deriv_par[3, :, 14] = eta_der_n / tau_n + tau_der_n * common_n

    eta_der_n = - (X[0, :] - par[14]) / (2 * (par[15] ** 2)) * kernel_n
    tau_der_n = 2 * par[17] * (X[0, :] - par[14]) / (par[15] ** 2) \
                * tanh_n * kernel_n
    deriv_par[3, :, 15] = eta_der_n / tau_n + tau_der_n * common_n

    deriv_par[3, :, 16] = common_n

    deriv_par[3, :, 17] = kernel_n * common_n
    
    return deriv_par

for _ in range(5):
    gtimer5()
    temp = cpu_field(X, par, stimulus)
    temp = cpu_jacobian(X, par)
    temp = cpu_dfield_dpar(X, par)
"""
%timeit -r 50 -n 10 temp = cpu_field(X, par, stimulus)
%timeit -r 50 -n 10 gtimer1()
%timeit -r 50 -n 10 temp = cpu_jacobian(X, par)
%timeit -r 50 -n 10 gtimer2()
%timeit -r 50 -n 10 temp = cpu_dfield_dpar(X, par)
%timeit -r 50 -n 10 gtimer3()
%timeit -r 50 -n 10 gtimer4()
%timeit -r 50 -n 10 gtimer5()
"""

