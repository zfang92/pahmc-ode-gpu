# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This module runs Lorenz96 within a fixed Rf. See user manual for what to expect.
"""


import os
from pathlib import Path
import time

from numba import cuda
import numpy as np
from matplotlib import pyplot as plt

os.chdir(Path.cwd().parent)
from pahmc_ode_gpu.pahmc import anneal
from pahmc_ode_gpu.configure import Configure
from pahmc_ode_gpu import cuda_lib_dynamics
from pahmc_ode_gpu.data_preparation import generate_twin_data
from pahmc_ode_gpu.cuda_utilities import k__action, k__diff, k__dAdX, \
    k__dAdpar, k__zeros1d


#=========================type your code below=========================
"""A name for your dynamics."""
# it will be used to try to find a match in the built-ins
name = 'lorenz96'

"""Specs for the dynamics."""
# set the dimension of your dynamics
D = 20
# set the length of the observation window
M = 200
# set the observed dimensions (list with smallest possible value 1)
obsdim = [1, 2, 4, 6, 8, 10, 12, 14, 15, 17, 19, 20]
# set the discretization interval
dt = 0.025

"""Specs for precision annealing and HMC."""
# set the starting Rf value
Rf0 = 1e6
# set alpha
alpha = 1.0
# set the total number of beta values
betamax = 1
# set the number of HMC samples for each beta
n_iter = 1000
# set the HMC simulation stepsize for each beta
epsilon = 1e-3
# set the number of leapfrog steps for an HMC sample for each beta
S = 100
# set the HMC masses for each beta
mass = (1e0, 1e0, 1e0)
# set the HMC scaling parameter for each beta
scaling = 1.0
# set the "soft" dynamical range for initialization purpose
soft_dynrange = (-10, 10)
# set an initial guess for the parameters
par_start = 8.0

"""Specs for the twin-experiment data"""
# set the length of the data (must be greater than M defined above)
length = 1000
# set the noise levels (standard deviations) in the data for each dimension
noise = 0.4 * np.ones(D)
# set the true parameters (caution: order must be consistent)
par_true = 8.17
# set the initial condition for the data generation process
x0 = np.ones(D)
x0[0] = 0.01
# set the switch for discarding the first half of the generated data
burndata = True
#===============================end here===============================


"""Configure the inputs and the stimuli."""
config = Configure(name, 
                   D, M, obsdim, dt, 
                   Rf0, alpha, betamax, 
                   n_iter, epsilon, S, mass, scaling, 
                   soft_dynrange, par_start, 
                   length, noise, par_true, x0, burndata)

config.check_all()

name, \
D, M, obsdim, dt, \
Rf0, alpha, betamax, \
n_iter, epsilon, S, mass, scaling, \
soft_dynrange, par_start, \
length, noise, par_true, x0, burndata = config.regulate()

stimuli = config.get_stimuli()


"""Fetch dynamics kernels."""
k__field = getattr(cuda_lib_dynamics, f'k__{name}_field')
k__jacobian = getattr(cuda_lib_dynamics, f'k__{name}_jacobian')
k__dfield_dpar = getattr(cuda_lib_dynamics, f'k__{name}_dfield_dpar')


"""Generate twin-experiment data, also trim stimuli as needed."""
data_noisy, stimuli \
  = generate_twin_data(name, k__field, k__jacobian, 
                       D, length, dt, noise, par_true, x0, burndata, stimuli)


"""Fetch data and stimuli for the training window."""
Y = data_noisy[obsdim, :M]
stimuli_training = np.ascontiguousarray(stimuli[:, :M])


"""Do precision annealing Hamiltonian Monte Carlo."""
t0 = time.perf_counter()

burn, Rm, Rf, eta_avg, acceptance, \
action, action_meanpath, ME_meanpath, FE_meanpath, \
X_init, X_gd, X_mean, par_history, par_mean, Xfinal_history \
  = anneal(k__field, k__jacobian, k__dfield_dpar, stimuli_training, Y, 
           D, M, obsdim, dt, Rf0, alpha, betamax, 
           n_iter, epsilon, S, mass, scaling, 
           soft_dynrange, par_start)

print(f'\nTotal time = {time.perf_counter()-t0:.2f} seconds.')

os.chdir(Path.cwd()/'unit_tests')


"""Plot action vs. iteration."""
fig, ax = plt.subplots(figsize=(6,5))
textblue = (49/255, 99/255, 206/255)
ax.loglog(np.arange(1, n_iter+2), action[0, 1:], color=textblue, lw=1.5)
ax.set_xlim(1, n_iter+1)
ax.set_xlabel('iteration')
ax.set_ylabel('action')


"""Get an overview of performance."""
d_stimuli = cuda.to_device(stimuli_training)
d_Y = cuda.to_device(Y)
d_obsdim = cuda.to_device(obsdim)
obs_ind = -np.ones(D, dtype='int64')
for l in range(len(obsdim)):
    obs_ind[obsdim[l]] = l
d_obs_ind = cuda.to_device(obs_ind)


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


print('\n--------------------------------------------------')
print('Initially:')
ov1_action, ov1_FE, ov1_dAdX, ov1_dAdpar \
  = overview(X_init[0, :, :], par_history[0, 0, :], Rf[0, :])

print('--------------------------------------------------')
print('After exploration:')
ov2_action, ov2_FE, ov2_dAdX, ov2_dAdpar \
  = overview(X_gd[0, :, :], par_history[0, 1, :], Rf[0, :])

print('--------------------------------------------------')
print('After exploitation:')
ov3_action, ov3_FE, ov3_dAdX, ov3_dAdpar \
  = overview(X_mean[0, :, :], par_mean[0, :], Rf[0, :])

