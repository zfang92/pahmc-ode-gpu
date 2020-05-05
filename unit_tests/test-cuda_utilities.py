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
from pahmc_ode_gpu.cuda_utilities import k__action, k__diff, k__dAdX, \
    k__dAdpar, k__leapfrog_X, k__leapfrog_par, k__linearop2d, k__linearop1d, \
    k__zeros1d
os.chdir(Path.cwd()/'unit_tests')


"""Prepare data."""
D = 20
M = 200

X = np.random.uniform(-8.0, 8.0, (D,M))
par = np.concatenate((np.random.uniform(8.1, 8.2, 1),
                      np.random.uniform(1e-3, 2e-3, 199)))

# get field, jacobian, and dfield_dpar
@jit(nopython=True)
def get_field(X, par):
    (D, M) = np.shape(X)
    vecfield = np.zeros((D,M))

    for m in range(M):
        vecfield[0, m] = (X[1, m] - X[D-2, m]) * X[D-1, m] - X[0, m]
        vecfield[1, m] = (X[2, m] - X[D-1, m]) * X[0, m] - X[1, m]
        vecfield[D-1, m] = (X[0, m] - X[D-3, m]) * X[D-2, m] - X[D-1, m]
        for a in range(2, D-1):
            vecfield[a, m] = (X[a+1, m] - X[a-2, m]) * X[a-1, m] - X[a, m]
    
    return vecfield + np.sum(par)

field = get_field(X, par)

@jit(nopython=True)
def get_jacobian(X, par):
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

jacobian = get_jacobian(X, par)

dfield_dpar = np.ones((D,len(par),M))

# get the necessary constants
obsdim = np.array(list(set(np.random.randint(0, D, int(D/2)))), dtype='int64')
Y = np.random.uniform(-8.0, 8.0, (len(obsdim),M))
dt = 0.025
Rf = np.random.uniform(1e3, 1e5, D)
epsilon = 1e-3
mass_X = np.random.uniform(0.5, 1.5, (D,M))
mass_par = np.random.uniform(0.5, 1.5, len(par))
scaling = 1e5
Rm = 1.1
obs_ind = -np.ones(D, dtype='int64')
for l in range(len(obsdim)):
    obs_ind[obsdim[l]] = l

pX = np.random.normal(0, 1e3, (D,M))
leapfrog_X = np.random.uniform(-8.0, 8.0, (D,M))
ppar = np.random.normal(0, 1e3, len(par))
leapfrog_par = np.random.uniform(10.0, 20.0, len(par))
Q2d = np.random.uniform(-8.0, 8.0, (D,M))
r2d = np.random.uniform(10.0, 20.0)
S2d = np.random.uniform(-8.0, 8.0, (D,M))
Q1d = np.random.uniform(10.0, 20.0, len(par))
r1d = np.random.uniform(10.0, 20.0)
S1d = np.random.uniform(10.0, 20.0, len(par))


"""Get the comparison variables except for the ones for 'dAdX' and 'dAdpar'."""
print('\nTesting... ', end='')

# for the action
@jit(nopython=True)
def cpu_fX(X, F, D, M, dt):
    fX = np.zeros((D,M-1))
    for a in range(D):
        for m in range(M-1):
            fX[a, m] = X[a, m] + dt / 2 * (F[a, m+1] + F[a, m])

    return fX

fX = cpu_fX(X, field, D, M, dt)

@jit(nopython=True)
def cpu_action(X, fX, Rf, D, M, Y, obsdim, Rm):
    measerr = 0
    for m in range(M):
        for l in range(len(obsdim)):
            measerr = measerr + (X[obsdim[l], m] - Y[l, m]) ** 2
    measerr = Rm / (2 * M) * measerr

    modelerr = 0
    for a in range(D):
        ss_a = 0
        for m in range(M-1):
            ss_a = ss_a + (X[a, m+1] - fX[a, m]) ** 2
        modelerr = modelerr + Rf[a] / (2 * M) * ss_a

    return measerr + modelerr

action_compared = np.array([cpu_action(X, fX, Rf, D, M, Y, obsdim, Rm)])

# for 'diff'
diff_compared = X[:, 1:] - fX[:, :M-1]

# for 'leapfrog_X'
leapfrog_X_compared = leapfrog_X + epsilon * pX / mass_X

# for 'leapfrog_par'
leapfrog_par_compared = leapfrog_par + epsilon * ppar / mass_par

# for 'linearop2d'
T2d_compared = Q2d + r2d * S2d

# for 'linearop1d'
T1d_compared = Q1d + r1d * S1d


"""Use PyTorch to get 'dAdX_compared' and 'dAdpar_compared'."""
# first define the scalar field (action) for Torch
X = th.from_numpy(X)
par = th.from_numpy(par)
Rf = th.from_numpy(Rf)
Y = th.from_numpy(Y)

X.requires_grad = True
par.requires_grad = True

vecfield = th.zeros(D, M, dtype=th.float64)
vecfield[0, :] = (X[1, :] - X[D-2, :]) * X[D-1, :] - X[0, :]
vecfield[1, :] = (X[2, :] - X[D-1, :]) * X[0, :] - X[1, :]
vecfield[D-1, :] = (X[0, :] - X[D-3, :]) * X[D-2, :] - X[D-1, :]
for a in range(2, D-1):
    vecfield[a, :] = (X[a+1, :] - X[a-2, :]) * X[a-1, :] - X[a, :]
vecfield += th.sum(par)

fX = X[:, :M-1] + dt / 2 * (vecfield[:, 1:] + vecfield[:, :M-1])

scalarfield = scaling * (Rm / 2 / M * th.sum((X[obsdim, :]-Y)**2) \
                         + th.sum(Rf/2/M*th.sum((X[:, 1:]-fX)**2, dim=1)))
scalarfield.backward()

# for 'dAdX'
dAdX_compared = X.grad.numpy()

# for 'dAdpar'
dAdpar_compared = par.grad.numpy()

# detach torch variables for later use
X = X.detach().numpy()
par = par.detach().numpy()
Rf = Rf.numpy()
Y = Y.numpy()


"""Transfer data and specify grid dimensions."""
d_X = cuda.to_device(X)
d_par = cuda.to_device(par)
d_field = cuda.to_device(field)
d_jacobian = cuda.to_device(jacobian)
d_dfield_dpar = cuda.to_device(dfield_dpar)

d_obsdim = cuda.to_device(obsdim)
d_Y = cuda.to_device(Y)
d_Rf = cuda.to_device(Rf)
d_mass_X = cuda.to_device(mass_X)
d_mass_par = cuda.to_device(mass_par)
d_obs_ind = cuda.to_device(obs_ind)

d_pX = cuda.to_device(pX)
d_leapfrog_X = cuda.to_device(leapfrog_X)
d_ppar = cuda.to_device(ppar)
d_leapfrog_par = cuda.to_device(leapfrog_par)
d_Q2d = cuda.to_device(Q2d)
d_S2d = cuda.to_device(S2d)
d_Q1d = cuda.to_device(Q1d)
d_S1d = cuda.to_device(S1d)

# transfer the output arrays
d_action = cuda.device_array(1, dtype='float64')
d_diff = cuda.device_array_like(diff_compared)
d_dAdX = cuda.device_array_like(X)
d_dAdpar = cuda.device_array_like(par)
d_T2d = cuda.device_array_like(T2d_compared)
d_T1d = cuda.device_array_like(T1d_compared)


"""Define convenience functions."""
def gtimer_action():
    k__action[(16,32), (16,16)](d_X, d_field, d_Rf, d_Y, dt, d_obsdim, Rm, 
                                d_action)
    cuda.synchronize()

def gtimer_dAdX():
    k__dAdX[(32,16), (2,128)](d_X, d_diff, d_jacobian, d_Rf, scaling, d_Y, 
                              dt, d_obsdim, d_obs_ind, Rm, d_dAdX)
    cuda.synchronize()

def gtimer_dAdpar():
    k__dAdpar[(4,4,32), (2,2,64)](d_X, d_diff, d_dfield_dpar, d_Rf, 
                                  scaling, dt, d_dAdpar)
    cuda.synchronize()


"""Test for correctness."""
k__zeros1d[40, 256](d_action)  # don't forget to initialize
cuda.synchronize()  # and don't forget to synchronize after initialization
gtimer_action()
action = d_action.copy_to_host()

k__diff[(32,16), (2,128)](d_X, d_field, dt, d_diff)
cuda.synchronize()
diff = d_diff.copy_to_host()

gtimer_dAdX()
dAdX = d_dAdX.copy_to_host()

k__zeros1d[40, 256](d_dAdpar)  # don't forget to initialize
cuda.synchronize()  # and don't forget to synchronize after initialization
gtimer_dAdpar()
dAdpar = d_dAdpar.copy_to_host()

k__leapfrog_X[(32,16), (2,128)](d_pX, epsilon, d_mass_X, d_leapfrog_X)
cuda.synchronize()
leapfrog_X = d_leapfrog_X.copy_to_host()

k__leapfrog_par[40, 256](d_ppar, epsilon, d_mass_par, d_leapfrog_par)
cuda.synchronize()
leapfrog_par = d_leapfrog_par.copy_to_host()

k__linearop2d[(32,16), (2,128)](d_Q2d, r2d, d_S2d, d_T2d)
cuda.synchronize()
T2d = d_T2d.copy_to_host()

k__linearop1d[40, 256](d_Q1d, r1d, d_S1d, d_T1d)
cuda.synchronize()
T1d = d_T1d.copy_to_host()

np.testing.assert_almost_equal(action, action_compared, decimal=4)
np.testing.assert_almost_equal(diff, diff_compared, decimal=6)
np.testing.assert_almost_equal(dAdX, dAdX_compared, decimal=6)
np.testing.assert_almost_equal(dAdpar, dAdpar_compared, decimal=5)
np.testing.assert_almost_equal(leapfrog_X, leapfrog_X_compared, decimal=6)
np.testing.assert_almost_equal(leapfrog_par, leapfrog_par_compared, decimal=6)
np.testing.assert_almost_equal(T2d, T2d_compared, decimal=6)
np.testing.assert_almost_equal(T1d, T1d_compared, decimal=6)
print('ok.')


#======================================================================
# for profiling only
@jit(nopython=True)
def cpu_dAdX(X, fX, J, Rf, scaling, Y, dt, obsdim, Rm):
    D, M = X.shape

    diff = np.zeros((D,M-1))
    for a in range(D):
        for m in range(M-1):
            diff[a, m] = X[a, m+1] - fX[a, m]

    part_meas = np.zeros((D,M))
    for m in range(M):
        for l in range(len(obsdim)):
            part_meas[obsdim[l], m] = Rm * (X[obsdim[l], m] - Y[l, m])

    part_model = np.zeros((D,M))
    for a in range(D):
        # m == 0 corner case
        for i in range(D):
            part_model[a, 0] = part_model[a, 0] \
                               + Rf[i] * J[i, a, 0] * diff[i, 0]
        part_model[a, 0] = - Rf[a] * diff[a, 0] - dt / 2 * part_model[a, 0]
        # m == M-1 corner case
        for i in range(D):
            part_model[a, -1] = part_model[a, -1] \
                                + Rf[i] * J[i, a, -1] * diff[i, -1]
        part_model[a, -1] = Rf[a] * diff[a, -1] - dt / 2 * part_model[a, -1]
        # m == {1, ..., M-2}
        for m in range(1, M-1):
            for i in range(D):
                part_model[a, m] = part_model[a, m] \
                                   + Rf[i] * J[i, a, m] \
                                     * (diff[i, m-1] + diff[i, m])
            part_model[a, m] = Rf[a] * (diff[a, m-1] - diff[a, m]) \
                               - dt / 2 * part_model[a, m]

    gradX_A = np.zeros((D,M))
    for a in range(D):
        for m in range(M):
            gradX_A[a, m] = scaling / M * (part_meas[a, m] + part_model[a, m])

    return gradX_A

@jit(nopython=True)
def cpu_dAdpar(X, fX, G, Rf, scaling, dt):
    D, M = X.shape

    gradpar_A = np.zeros(G.shape[1])
    for b in range(G.shape[1]):
        for i in range(D):
            ss_i = 0
            for m in range(M-1):
                ss_i = ss_i + (X[i, m+1] - fX[i, m]) \
                              * (G[i, b, m] + G[i, b, m+1])
            gradpar_A[b] = gradpar_A[b] + Rf[i] * ss_i
        gradpar_A[b] = - scaling / M * dt / 2 * gradpar_A[b]

    return gradpar_A

# define convenience functions
def ctimer_action():
    fX = cpu_fX(X, field, D, M, dt)
    cpu_action(X, fX, Rf, D, M, Y, obsdim, Rm)

def ctimer_dAdX():
    fX = cpu_fX(X, field, D, M, dt)
    cpu_dAdX(X, fX, jacobian, Rf, scaling, Y, dt, obsdim, Rm)

def ctimer_dAdpar():
    fX = cpu_fX(X, field, D, M, dt)
    cpu_dAdpar(X, fX, dfield_dpar, Rf, scaling, dt)

for _ in range(3):
    gtimer_action(); gtimer_dAdX(); gtimer_dAdpar()
    ctimer_action(); ctimer_dAdX(); ctimer_dAdpar()
"""
%timeit -r 50 -n 10 ctimer_action()
%timeit -r 50 -n 10 gtimer_action()
%timeit -r 50 -n 10 ctimer_dAdX()
%timeit -r 50 -n 10 gtimer_dAdX()
%timeit -r 50 -n 10 ctimer_dAdpar()
%timeit -r 50 -n 10 gtimer_dAdpar()
"""

