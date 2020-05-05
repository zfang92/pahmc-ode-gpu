# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

As advised in the user manual, it is better to generate data using this test
module than directly running PAHMC since here allows you to view the generated 
data.
"""


import os
from pathlib import Path
import time

import numpy as np
from matplotlib import pyplot as plt

os.chdir(Path.cwd().parent)
from pahmc_ode_gpu import cuda_lib_dynamics
from pahmc_ode_gpu.data_preparation import generate_twin_data


"""Write down specs for the twin-experiment data."""
name = 'nakl'

D = 4
dt = 0.02
length = int(1000/dt)
noise = np.array([1.0, 0, 0, 0], dtype='float64')
par_true = np.array([120, 50, 20, -77, 0.3, -54.4, 
                     -40, 15, 0.1, 0.4, 
                     -60, -15, 1, 7, 
                     -55, 30, 1, 5], dtype='float64')
x0 = np.array([-70, 0.1, 0.9, 0.1], dtype='float64')
burndata = False

stimuli = np.load(Path.cwd()/'user_data'/f'{name}_stimuli.npy')[:, 0:2*length]


"""Generate data."""
# fetch the kernels
k__field = getattr(cuda_lib_dynamics, f'k__{name}_field')
k__jacobian = getattr(cuda_lib_dynamics, f'k__{name}_jacobian')

# run the data generator
t0 = time.perf_counter()
data_noisy, stimuli \
  = generate_twin_data(name, k__field, k__jacobian, 
                       D, length, dt, noise, par_true, x0, burndata, stimuli)
print(f'Time elapsed = {time.perf_counter()-t0:.2f} seconds.')

# get the noise level
file = np.load(Path.cwd()/'user_data'/f'{name}_noiseless.npz')
data_noiseless = file['data']
file.close()
print(f'Chi-squared = {np.sum((data_noisy-data_noiseless)**2):.4f} '
      +f'({np.sum(noise**2)*length:.4f} expected).')


"""Plot."""
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12,8))
textred = (202/255, 51/255, 0)
textblue = (49/255, 99/255, 206/255)

time = np.linspace(0, int(length*dt)-dt, length)

ax1.plot(time, data_noisy[0, :], color=textblue)
ax1.legend(['data_noisy'], loc='upper right')
ax1.set_xlim(0, int(length*dt))
ax1.set_xticks(np.linspace(0, 1000, 11))
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('V(t)', rotation='horizontal')

ax2.plot(time, stimuli[0, :], color=textred)
ax2.legend(['stimulus'], loc='upper right')
ax2.set_xlim(0, int(length*dt))
ax2.set_xticks(np.linspace(0, 1000, 11))
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('I_inj(t)', rotation='horizontal')


os.chdir(Path.cwd()/'unit_tests')

