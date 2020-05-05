# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This module tests the 'exploration' part of PAHMC.
"""


import os
from pathlib import Path

from numba import cuda
import numpy as np

os.chdir(Path.cwd().parent)
from pahmc_ode_gpu import cuda_lib_dynamics
from pahmc_ode_gpu.gd import descend
from user_results.read import get_saved
from pahmc_ode_gpu.cuda_utilities import k__action, k__diff, k__dAdX, \
    k__dAdpar, k__zeros1d
os.chdir(Path.cwd()/'unit_tests')


"""Retrieve necessary variables to get started."""
name, D, M, obsdim, dt, Rf0, alpha, betamax, \
n_iter, epsilon, S, mass, scaling, soft_dynrange, par_start, \
length, data_noisy, stimuli, noise, par_true, x0, burndata, \
burn, Rm, Rf, eta_avg, acceptance, \
action, action_meanpath, ME_meanpath, FE_meanpath, \
X_init, X_gd, X_mean, par_history, par_mean, Xfinal_history \
  = get_saved(Path.cwd(), 'test-gd')

# fetch the dynamics kernels
k__field = getattr(cuda_lib_dynamics, f'k__{name}_field')
k__jacobian = getattr(cuda_lib_dynamics, f'k__{name}_jacobian')
k__dfield_dpar = getattr(cuda_lib_dynamics, f'k__{name}_dfield_dpar')

# define inputs to 'descend'
X0 = X_init[0, :, :]
par0 = par_history[0, 0, :]
Rf = Rf[0, :]

d_stimuli = cuda.to_device(np.ascontiguousarray(stimuli[:, :M]))
d_Y = cuda.to_device(data_noisy[obsdim, :M])
d_obsdim = cuda.to_device(obsdim)
obs_ind = -np.ones(D, dtype='int64')
for l in range(len(obsdim)):
    obs_ind[obsdim[l]] = l
d_obs_ind = cuda.to_device(obs_ind)


"""Define a convenience function."""
def overview(X, par, Rf):
    # define device arrays
    d_X = cuda.to_device(X)
    d_par = cuda.to_device(par)
    d_Rf = cuda.to_device(Rf)

    d_field = cuda.device_array_like(X)
    d_jacobian = cuda.device_array((D,D,M))
    d_dfield_dpar = cuda.device_array((D,len(par),M))
    d_action = cuda.device_array((1,))
    d_diff = cuda.device_array((D,M-1))
    d_dAdX = cuda.device_array_like(X)
    d_dAdpar = cuda.device_array_like(par)

    # get the action
    k__field[(16,32), (2,128)](d_X, d_par, d_stimuli, d_field)
    k__zeros1d[40, 256](d_action)
    cuda.synchronize()
    k__action[(16,32), (16,16)](d_X, d_field, d_Rf, d_Y, dt, d_obsdim, Rm, 
                                d_action)
    action = d_action.copy_to_host()[0]

    # get model error
    field = d_field.copy_to_host()
    fX = X[:, :M-1] + dt / 2 * (field[:, 1:] + field[:, :M-1])
    FE = np.sum(Rf/2/M*np.sum((X[:, 1:]-fX)**2, axis=1))

    # get the gradients
    k__diff[(32,16), (2,128)](d_X, d_field, dt, d_diff)
    k__jacobian[(4,4,32), (2,2,64)](d_X, d_par, d_jacobian)
    cuda.synchronize()
    k__dAdX[(32,16), (2,128)](d_X, d_diff, d_jacobian, d_Rf, 1.0, d_Y, dt, 
                              d_obsdim, d_obs_ind, Rm, d_dAdX)

    k__dfield_dpar[(4,4,32), (2,2,64)](d_X, d_par, d_dfield_dpar)
    k__zeros1d[40, 256](d_dAdpar)
    cuda.synchronize()
    k__dAdpar[(4,4,32), (2,2,64)](d_X, d_diff, d_dfield_dpar, d_Rf, 1.0, dt, 
                                  d_dAdpar)
    dAdX = d_dAdX.copy_to_host()
    dAdpar = d_dAdpar.copy_to_host()

    # print results
    print(f'\n      action = {action},')
    print(f'    modelerr = {FE},\n')
    print(f'  max |dAdX| = {np.max(np.abs(dAdX))},')
    print(f'  min |dAdX| = {np.min(np.abs(dAdX))},\n')
    print(f'max |dAdpar| = {np.max(np.abs(dAdpar))},')
    print(f'min |dAdpar| = {np.min(np.abs(dAdpar))}.\n')

    return action, FE, dAdX, dAdpar


"""Get results."""
# before gradient descent
print('\n--------------------------------------------------')
print('Initial values before gradient descent:')
ov_action, ov_FE, ov_dAdX, ov_dAdpar = overview(X0, par0, Rf)

# perform gradient descent
X_gd, par_gd, action_gd, eta \
  = descend(k__field, k__jacobian, k__dfield_dpar, X0, par0, Rf, 
            d_stimuli, d_Y, dt, d_obsdim, d_obs_ind, Rm)

# after gradient descent
print('\n--------------------------------------------------')
print('After gradient descent:')
ow_action, ow_FE, ow_dAdX, ow_dAdpar = overview(X_gd, par_gd, Rf)

