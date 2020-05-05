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

os.chdir(Path.cwd().parent)
from pahmc_ode_gpu import cuda_lib_dynamics
os.chdir(Path.cwd()/'unit_tests')


"""Prepare data, as well as variables to be compared to."""
name = 'lorenz96'

D = 200
M = 2000

X = np.random.uniform(-8.0, 8.0, (D,M))
par = np.array([8.17])
stimulus = np.random.uniform(-1.0, 1.0, (D,M))

# these functions have been tested in pahmc_ode_cpu
@jit(nopython=True)
def cpu_field(X, par, stimulus):
    (D, M) = np.shape(X)
    vecfield = np.zeros((D,M))

    for m in range(M):
        vecfield[0, m] = (X[1, m] - X[D-2, m]) * X[D-1, m] - X[0, m]
        vecfield[1, m] = (X[2, m] - X[D-1, m]) * X[0, m] - X[1, m]
        vecfield[D-1, m] = (X[0, m] - X[D-3, m]) * X[D-2, m] - X[D-1, m]
        for a in range(2, D-1):
            vecfield[a, m] = (X[a+1, m] - X[a-2, m]) * X[a-1, m] - X[a, m]
    
    return vecfield + par[0]

@jit(nopython=True)
def cpu_jacobian(X, par):
    (D, M) = np.shape(X)
    jacob = np.zeros((D,D,M))

    for m in range(M):
        for i in range(1, D+1):
            for j in range(1, D+1):
                jacob[i-1, j-1, m] \
                  = (1 + (i - 2) % D == j) \
                    * (X[i%D, m] - X[(i-3)%D, m]) \
                    + ((1 + i % D == j) - (1 + (i - 3) % D == j)) \
                    * X[(i-2)%D, m] - (i == j)
    
    return jacob

@jit(nopython=True)
def cpu_dfield_dpar(X, par):
    (D, M) = np.shape(X)

    return np.ones((D,len(par),M))

print('\nTesting... ', end='')
field_compared = cpu_field(X, par, stimulus)
jacobian_compared = cpu_jacobian(X, par)
dfield_dpar_compared = cpu_dfield_dpar(X, par)


"""Fetch the kernels, transfer data, and specify grid dimensions."""
k__field = getattr(cuda_lib_dynamics, f'k__{name}_field')
k__jacobian = getattr(cuda_lib_dynamics, f'k__{name}_jacobian')
k__dfield_dpar = getattr(cuda_lib_dynamics, f'k__{name}_dfield_dpar')

d_X = cuda.to_device(X)
d_par = cuda.to_device(par)
d_stimulus = cuda.to_device(stimulus)

d_field = cuda.to_device(np.zeros((D,M)))
d_jacobian = cuda.to_device(np.zeros((D,D,M)))
d_dfield_dpar = cuda.to_device(np.zeros((D,1,M)))


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
jacobian = d_jacobian.copy_to_host()
dfield_dpar = d_dfield_dpar.copy_to_host()

np.testing.assert_almost_equal(field, field_compared, decimal=12)
np.testing.assert_almost_equal(jacobian, jacobian_compared, decimal=12)
np.testing.assert_almost_equal(dfield_dpar, dfield_dpar_compared, decimal=12)
print('ok.')


#======================================================================
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

