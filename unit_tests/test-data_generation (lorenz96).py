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
name = 'lorenz96'

D = 20
length = 1000
dt = 0.025
noise = 0.4 * np.ones(D)
par_true = 8.17
x0 = np.ones(D); x0[0] = 0.01
burndata = True

stimuli = np.zeros((D,2*length))


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
fig, ax = plt.subplots(figsize=(8,4.5))
textblue = (49/255, 99/255, 206/255)

time = np.linspace(0, int(length*dt)-dt, length)

ax.plot(time, data_noisy[1, :], color=textblue, lw=1.5)
ax.legend(['data_noisy'], loc='upper right')
ax.set_xlim(0, 25)
ax.set_xticks(np.linspace(0, 25, 11))
ax.set_xlabel('Time ($\Delta t = 0.025$s)')
ax.set_ylabel('$x_1(t)$', rotation='vertical')


os.chdir(Path.cwd()/'unit_tests')

