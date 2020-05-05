# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This module implements the gradient descent algorithm (with custom 
modifications) as the 'exploration' part of the PAHMC method.
"""


import time

from numba import cuda
import numpy as np

from pahmc_ode_gpu.cuda_utilities import k__action, k__diff, k__dAdX, \
    k__dAdpar, k__linearop2d, k__linearop1d, k__zeros1d


def descend(k__field, k__jacobian, k__dfield_dpar, X0, par0, Rf, 
            d_stimuli, d_Y, dt, d_obsdim, d_obs_ind, Rm, eta0=0.1, tmax=1000):
    """
    This method implements batch gradient descent with adaptive learning rates.
    It is found that this implementation outperforms some advanced algorithms 
    including the Nesterov Accelerated Gradient in the present context.

    Inputs
    ------
          k__field: CUDA kernel for the vector field.
       k__jacobian: CUDA kernel for the jacobian.
    k__dfield_dpar: CUDA kernel for the derivatives w.r.t. to the parameters.
                X0: initial path, a D-by-M numpy array.
              par0: initial parameters, a 1d (shapeless) numpy array.
                Rf: current Rf, a 1d (shapeless) numpy array of length D.
         d_stimuli: the stimuli (synchronous with Y), a D-by-M device array.
               d_Y: the data, a len(obsdim)-by-M device array.
                dt: the time interval, a float.
          d_obsdim: the observed dimensions, a 1d (shapeless) device array.
         d_obs_ind: a 1d (shapeless) device array of length D.
                Rm: a float.
              eta0: initial learning rate, a float.
              tmax: maximum number of gradient descent epochs.

    Returns
    ------
         X: path after gradient descent, a D-by-M numpy array.
       par: parameters after gradient descent, a 1d (shapeless) numpy array.
    action: action value after gradient descent, a float.
       eta: learning rates, a 1d (shapeless) numpy array with length tmax+1.
    """
    t0 = time.perf_counter()

    # initialize learning rates
    eta = np.zeros(tmax+1)
    eta[0] = eta0

    # set initial device arrays
    d_X = cuda.to_device(X0)
    d_par = cuda.to_device(par0)
    d_Rf = cuda.to_device(Rf)

    D, M = X0.shape
    d_field = cuda.device_array_like(X0)
    d_jacobian = cuda.device_array((D,D,M))
    d_dfield_dpar = cuda.device_array((D,len(par0),M))
    d_action = cuda.device_array((1,))
    d_diff = cuda.device_array((D,M-1))
    d_dAdX = cuda.device_array_like(X0)
    d_dAdpar = cuda.device_array_like(par0)
    d_X_try = cuda.device_array_like(X0)
    d_par_try = cuda.device_array_like(par0)
    d_action_try = cuda.device_array((1,))
    d_dummy = cuda.device_array((1,))

    # calculate initial action
    k__field[(16,32), (2,128)](d_X, d_par, d_stimuli, d_field)
    k__zeros1d[40, 256](d_action)
    cuda.synchronize()
    k__action[(16,32), (16,16)](d_X, d_field, d_Rf, d_Y, dt, d_obsdim, Rm, 
                                d_action)
    action = d_action.copy_to_host()[0]
    
    # begin gradient descent
    accel_flag = 0
    for t in range(1, tmax+1):
        print(f'\r  Exploring A(X) manifold... (step={t})', end='')

        # get the gradients
        k__diff[(32,16), (2,128)](d_X, d_field, dt, d_diff)
        k__jacobian[(4,4,32), (2,2,64)](d_X, d_par, d_jacobian)
        cuda.synchronize()
        k__dAdX[(32,16), (2,128)](d_X, d_diff, d_jacobian, d_Rf, 1.0, d_Y, 
                                  dt, d_obsdim, d_obs_ind, Rm, d_dAdX)

        k__dfield_dpar[(4,4,32), (2,2,64)](d_X, d_par, d_dfield_dpar)
        k__zeros1d[40, 256](d_dAdpar)
        cuda.synchronize()
        k__dAdpar[(4,4,32), (2,2,64)](d_X, d_diff, d_dfield_dpar, d_Rf, 
                                      1.0, dt, d_dAdpar)
        cuda.synchronize()

        # initialize current step learning rate
        if accel_flag == 1:
            eta[t] = eta[t-1] * 2
        else:
            eta[t] = eta[t-1]
            accel_flag = 1

        # get trial X and par
        k__linearop2d[(32,16), (2,128)](d_X, -eta[t], d_dAdX, d_X_try)
        k__linearop1d[40, 256](d_par, -eta[t], d_dAdpar, d_par_try)
        cuda.synchronize()

        # get trial action
        k__field[(16,32), (2,128)](d_X_try, d_par_try, d_stimuli, d_field)
        k__zeros1d[40, 256](d_action_try)
        cuda.synchronize()
        k__action[(16,32), (16,16)](d_X_try, d_field, d_Rf, d_Y, dt, 
                                    d_obsdim, Rm, d_action_try)
        action_try = d_action_try.copy_to_host()[0]

        # try to tame the trial results
        counter = 0
        while action_try >= action:
            # halve current learning rate
            accel_flag = 0
            eta[t] /= 2

            # get trial X and par
            k__linearop2d[(32,16), (2,128)](d_X, -eta[t], d_dAdX, d_X_try)
            k__linearop1d[40, 256](d_par, -eta[t], d_dAdpar, d_par_try)
            cuda.synchronize()

            # get trial action
            k__field[(16,32), (2,128)](d_X_try, d_par_try, d_stimuli, d_field)
            k__zeros1d[40, 256](d_action_try)
            cuda.synchronize()
            k__action[(16,32), (16,16)](d_X_try, d_field, d_Rf, d_Y, dt, 
                                        d_obsdim, Rm, d_action_try)
            action_try = d_action_try.copy_to_host()[0]

            # return if getting stuck
            counter += 1
            if counter == 100:
                return d_X.copy_to_host(), d_par.copy_to_host(), action, eta

        # finalize results for the current step
        d_dummy = d_X
        d_X = d_X_try
        d_X_try = d_dummy

        d_dummy = d_par
        d_par = d_par_try
        d_par_try = d_dummy

        action = action_try

    print(f'\r  Exploring A(X) manifold... '
          +f'finished in {time.perf_counter()-t0:.2f} seconds.')

    return d_X.copy_to_host(), d_par.copy_to_host(), action, eta

