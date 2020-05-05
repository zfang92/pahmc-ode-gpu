# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This module generates twin-experiemnt data for training and validation.
"""


from sys import exit
from pathlib import Path

from numba import cuda
import numpy as np


def generate_twin_data(name, k__field, k__jacobian, 
                       D, length, dt, noise, par_true, x0, burndata, stimuli):
    """
    This method first searches for existing data file by looking for a filename
    that matches the name of the user-defined dynamics. If found successfully, 
    it then loads the file and compares the detailed specs of its data with 
    user specs; if everything matches up, the existing data will be returned. 
    In all other cases, it integrates the user-defined dynamics using the 
    trapezoidal rule and then outputs the generated data along with the 
    stimulus, and saves the data files.

    Inputs
    ------
           name: name of the dynamics, a string.
       k__field: CUDA kernel for the vector field.
    k__jacobian: CUDA kernel for the jacobian of the vector field.
              D: model degrees of freedom, an integer.
         length: total number of time steps for the generated data, an integer.
             dt: discretization interval, a float.
          noise: standard deviation of the added noise, an 1d (shapeless) numpy 
                 array of length D.
       par_true: true parameters used to generate the data, an 1d (shapeless) 
                 numpy array.
             x0: initial condition, an 1d (shapeless) numpy array of length D.
       burndata: switch for burning the first half of the generated data, a 
                 boolean.
        stimuli: the stimuli, a 2d numpy array.

    Returns
    -------
    data_noisy: the generated noisy data, a D-by-length numpy array.
       stimuli: the tailored stimuli, a D-by-length numpy array.
    """
    print('\nGenerating data... ', end='')

    if burndata == True:
        start = length

        if stimuli.shape[1] < 2 * length:
            print('aborted. Please make sure the length of \'stimuli\' is at '
                  +'least 2*\'length\' since \'burndata\' is set to be True.')
            exit()
    else:
        start = 0

    if np.shape(par_true) == ():
        par_true = np.array([par_true])

    filepath = Path.cwd() / 'user_data'

    if (filepath / f'{name}.npz').exists():  # if a match is found
        file = np.load(filepath/f'{name}.npz')

        try:
            file['device']
        except:
            print(f'aborted. Please remove files \"{name}.npz\" and '
                  +f'\"{name}_noiseless.npz\" from the user data '
                  +'directory and run again.\n')
            exit()

        if file['device'] == 'gpu' \
        and np.shape(file['data']) == (D, length) \
        and file['dt'] == dt \
        and np.array_equal(file['noise'], noise) \
        and np.array_equal(file['par_true'], par_true) \
        and bool(file['burndata']) == burndata \
        and np.array_equal(file['stimuli'], stimuli[:, start:start+length]):
            data_noisy = file['data']
            file.close()

            print('successful (data with the same specs already exist).\n')
            return data_noisy, stimuli[:, start:start+length]

    # for all other cases
    rawdata = np.zeros((D,start+length))
    rawdata[:, 0] = x0

    d_par = cuda.to_device(par_true)
    d_field = cuda.device_array_like(np.zeros((D,1)))
    d_jacobian = cuda.device_array_like(np.zeros((D,D,1)))

    for k in range(start+length-1):
        print(f'\rGenerating data... (t={k})', end='')

        d_stimulusk = cuda.to_device(stimuli[:, [k]])
        d_stimuluskp1 = cuda.to_device(stimuli[:, [k+1]])
        d_rawdatak = cuda.to_device(rawdata[:, [k]])

        # Newton-Raphson's initial guess using the Euler method
        k__field[(16,32), (2,128)](d_rawdatak, d_par, d_stimulusk, d_field)
        x_start = rawdata[:, [k]] + dt * d_field.copy_to_host()

        # first iteration of Newton-Raphson for the trapezoidal rule
        d_xstart = cuda.to_device(x_start)

        k__field[(16,32), (2,128)](d_xstart, d_par, d_stimuluskp1, d_field)
        field1 = d_field.copy_to_host()
        k__field[(16,32), (2,128)](d_rawdatak, d_par, d_stimulusk, d_field)
        field2 = d_field.copy_to_host()
        g_x = dt / 2 * (field1[:, 0] + field2[:, 0]) \
              + rawdata[:, k] - x_start[:, 0]

        k__jacobian[(4,4,32), (2,2,64)](d_xstart, d_par, d_jacobian)
        J = dt / 2 * d_jacobian.copy_to_host()[:, :, 0] - np.identity(D)

        x_change = np.linalg.solve(J, g_x)[:, np.newaxis]
        x_new = x_start - x_change
        x_start = x_new

        # iterate until the correction reaches tolerance level
        while np.sum(abs(x_change)) > 1e-13:
            d_xstart = cuda.to_device(x_start)

            k__field[(16,32), (2,128)](d_xstart, d_par, d_stimuluskp1, d_field)
            field1 = d_field.copy_to_host()
            g_x = dt / 2 * (field1[:, 0] + field2[:, 0]) \
                  + rawdata[:, k] - x_start[:, 0]

            k__jacobian[(4,4,32), (2,2,64)](d_xstart, d_par, d_jacobian)
            J = dt / 2 * d_jacobian.copy_to_host()[:, :, 0] - np.identity(D)

            x_change = np.linalg.solve(J, g_x)[:, np.newaxis]
            x_new = x_start - x_change
            x_start = x_new

        rawdata[:, [k+1]] = x_new  # final value

    data_noiseless = rawdata[:, start:start+length]

    np.savez(filepath/f'{name}_noiseless', 
             device='gpu', 
             data=data_noiseless, 
             dt=dt, 
             noise=np.zeros(D), 
             par_true=par_true, 
             burndata=burndata, 
             stimuli=stimuli[:, start:start+length])

    data_noisy = np.zeros((D,length))
    for a in range(D):
        data_noisy[a, :] \
          = data_noiseless[a, :] + np.random.normal(0, noise[a], length)
    
    np.savez(filepath/f'{name}', 
             device='gpu', 
             data=data_noisy, 
             dt=dt, 
             noise=noise, 
             par_true=par_true, 
             burndata=burndata, 
             stimuli=stimuli[:, start:start+length])

    print('\rGenerating data... successful.\n')
    return data_noisy, stimuli[:, start:start+length]

