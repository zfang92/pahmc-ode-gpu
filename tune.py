# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

Alternative to 'main.py', if you would like to tune the hyperparameters for  
each beta (generally a good practice), use this file as the main executable.

The user is assumed to have the following:
    1) The dynamical system. If calling one of the built-in examples, the name
    of the dynamics must have a match in 'lib_dynamics.py'; if builing from 
    scratch, 'def_dynamics.py' must be ready at this point.
    2) The data. If performing twin-experiments, the specs should be given but 
    a data file is not required; if working with real data, the data should be 
    prepared according to the user manual.
    3) If external stimuli are needed, a .npy file containing the time series; 
    4) Configuration of the code, including the hyper-parameters for PAHMC. 
    Refer to the manual for the shape and type requirements. Also note that a 
    lot of them can take either a single or an array/list of values. See user 
    manual for details.

It is suggested that the user keep a lookup table for the model paramters to 
make it easier to preserve order when working on the above steps.
"""


from datetime import date
from pathlib import Path
import time

from numba import cuda
import numpy as np
import matplotlib.pyplot as plt

from pahmc_ode_gpu.pahmc_tune import anneal
from pahmc_ode_gpu.configure import Configure
from pahmc_ode_gpu import cuda_lib_dynamics
from pahmc_ode_gpu.data_preparation import generate_twin_data
from pahmc_ode_gpu.cuda_utilities import k__action, k__diff, k__dAdX, \
    k__dAdpar, k__zeros1d


#================type your code below (stepwise tuning)================
"""Tunable hyperparameters."""
# set the beta value to be tuned
tune_beta = 0
# set the number of HMC samples for each beta
n_iter = 500
# set the HMC simulation stepsize for each beta
epsilon = 1e-3
# set the number of leapfrog steps for an HMC sample for each beta
S = 50
# set the HMC masses for each beta
mass = (1e0, 1e0, 1e0)
# set the HMC scaling parameter for each beta
scaling = 1e5
#===================type your code below (only once)===================
"""A name for your dynamics."""
# it will be used to try to find a match in the built-ins
name = 'nakl'

"""Specs for the dynamics."""
# set the dimension of your dynamics
D = 4
# set the length of the observation window
M = 5000
# set the observed dimensions (list with smallest possible value 1)
obsdim = [1]
# set the discretization interval
dt = 0.02

"""The remaining hyperparameters."""
# set the starting Rf value
Rf0 = np.array([1.0e-1, 1.2e3, 1.6e3, 2.1e3])
# set alpha
alpha = 2.0
# set the "soft" dynamical range for initialization purpose
soft_dynrange = np.array([[-120, 0], [0, 1], [0, 1], [0, 1]])
# set an initial guess for the parameters
par_start = np.array([115, 50, 25, -70, 0.2, -55, 
                      -45, 16, 0.15, 0.4,
                      -55, -16, 1.2, 6,
                      -52, 31, 0.8, 5])

"""Specs for the twin-experiment data"""
# set the length of the data (must be greater than M defined above)
length = int(1000/dt)
# set the noise levels (standard deviations) in the data for each dimension
noise = np.array([1, 0, 0, 0])
# set the true parameters (caution: order must be consistent)
par_true = np.array([120, 50, 20, -77, 0.3, -54.4, 
                     -40, 15, 0.1, 0.4, 
                     -60, -15, 1, 7, 
                     -55, 30, 1, 5])
# set the initial condition for the data generation process
x0 = np.array([-70, 0.1, 0.9, 0.1])
# set the switch for discarding the first half of the generated data
burndata = False
#===============================end here===============================


"""Prepare current Rf and set betamax."""
Rf0 = Rf0 * (alpha ** tune_beta)
betamax = 1


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
           soft_dynrange, par_start, name, tune_beta)

print(f'\nTotal time = {time.perf_counter()-t0:.2f} seconds.')


"""Save the results."""
np.savez(Path.cwd()/'user_results'/f'tune_{name}_{tune_beta}', 
         name=name, 
         D=D, M=M, obsdim=obsdim, dt=dt, 
         Rf0=Rf0, alpha=alpha, betamax=betamax, 
         n_iter=n_iter, epsilon=epsilon, S=S, mass=mass, scaling=scaling, 
         soft_dynrange=soft_dynrange, par_start=par_start, 
         length=length, data_noisy=data_noisy, stimuli=stimuli, 
         noise=noise, par_true=par_true, x0=x0, burndata=burndata, 
         burn=burn, 
         Rm=Rm, 
         Rf=Rf, 
         eta_avg=eta_avg, 
         acceptance=acceptance, 
         action=action, 
         action_meanpath=action_meanpath, 
         ME_meanpath=ME_meanpath, 
         FE_meanpath=FE_meanpath, 
         X_init=X_init, 
         X_gd=X_gd, 
         X_mean=X_mean, 
         par_history=par_history, 
         par_mean=par_mean, 
         Xfinal_history=Xfinal_history)


"""Plot action vs. iteration for current beta."""
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

# retrive the noiseless data (if doing twin experiment)
noiselessfile_name = Path.cwd() / 'user_data' / f'{name}_noiseless.npz'
if noiselessfile_name.exists():
    noiselessfile = np.load(noiselessfile_name)
    X_true = noiselessfile['data'][:, 0:M]
    noiselessfile.close()


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

print('\n--------------------------------------------------')
print('L1 distances (traveled and remaining):')
print('\n    from X_init to X_mean: '\
      +f'{np.sum(np.abs(X_mean[0, :, :]-X_init[0, :, :]))},')
if noiselessfile_name.exists():
    print('    from X_mean to X_true: '\
          +f'{np.sum(np.abs(X_true-X_mean[0, :, :]))},')
print('\nfrom par_init to par_mean: '\
      +f'{np.sum(np.abs(par_mean[0, :]-par_history[0, 0, :]))},')
if noiselessfile_name.exists():
    print('from par_mean to par_true: '\
          +f'{np.sum(np.abs(par_true-par_mean[0, :]))}.')

