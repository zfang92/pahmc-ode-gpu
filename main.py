# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This is the main executable of pahmc_ode_cpu and should be the point of entry
at which all the necessary information is provided. In particular, the user 
is assumed to have the following:
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


"""Save the results."""
day = date.today().strftime('%Y-%m-%d')
i = 1
while (Path.cwd() / 'user_results' / f'{name}_{day}_{i}.npz').exists():
    i = i + 1

np.savez(Path.cwd()/'user_results'/f'{name}_{day}_{i}', 
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

