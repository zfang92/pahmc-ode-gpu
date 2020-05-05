# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This file contains all necessary functions for PAHMC other than the dynamics.
The functions are to be called by 'gd' and 'hmc' in 'pahmc.py'.
"""


from numba import cuda


@cuda.jit
def k__action(X, field, Rf, Y, dt, obsdim, Rm, action):
    """
    This kernel calculates the action. The parameter 'action' must be 
    initialized before launching this kernel.

    Inputs
    ------
         X: D-by-M device array.
     field: D-by-M device array.
        Rf: one-dimensional (shapeless) device array of length D.
         Y: len(obsdim)-by-M device array.
        dt: scalar.
    obsdim: one-dimensional (shapeless) device array.
        Rm: scalar.

    Modifications
    ------
    action: one-dimenaional (shapeless) device array of length 1.
    """
    start_a, start_m = cuda.grid(2)
    stride_a, stride_m = cuda.gridsize(2)

    D, M = X.shape

    if start_a >= D or start_m >= M:
        return

    for l in range(start_a, len(obsdim), stride_a):
        for m in range(start_m, M, stride_m):
            cuda.atomic.add(action, 0, Rm/2/M*(X[obsdim[l], m]-Y[l, m])**2)

    for a in range(start_a, D, stride_a):
        for m in range(start_m, M-1, stride_m):
            fX = X[a, m] + dt / 2 * (field[a, m+1] + field[a, m])
            cuda.atomic.add(action, 0, Rf[a]/2/M*(X[a, m+1]-fX)**2)

@cuda.jit
def k__diff(X, field, dt, diff):
    """
    This kernel calculates a necessary piece in calculating dAdX and dAdpar.

    Inputs
    ------
        X: D-by-M device array.
    field: D-by-M device array.
       dt: scalar.

    Modifications
    ------
    diff: D-by-(M-1) device array.
    """
    start_a, start_m = cuda.grid(2)
    stride_a, stride_m = cuda.gridsize(2)

    D, M = X.shape

    if start_a >= D or start_m >= M:
        return

    for a in range(start_a, D, stride_a):
        for m in range(start_m, M-1, stride_m):
            diff[a, m] \
              = X[a, m+1] - (X[a, m] + dt / 2 * (field[a, m+1] + field[a, m]))

@cuda.jit
def k__dAdX(X, diff, jacobian, Rf, scaling, Y, dt, obsdim, obs_ind, Rm, dAdX):
    """
    This kernel calculates the derivatives of the action with respect to the 
    path X.

    Inputs
    ------
           X: D-by-M device array.
        diff: D-by-(M-1) device array.
    jacobian: D-by-D-by-M device array.
          Rf: one-dimensional (shapeless) device array of length D.
     scaling: scalar.
           Y: len(obsdim)-by-M device array.
          dt: scalar.
      obsdim: one-dimensional (shapeless) device array.
     obs_ind: one-dimensional (shapeless) device array of length D.
          Rm: scalar.

    Modifications
    ------
    dAdX: D-by-M device array.
    """
    start_a, start_m = cuda.grid(2)
    stride_a, stride_m = cuda.gridsize(2)

    D, M = X.shape

    if start_a >= D or start_m >= M:
        return

    # 'measurement' part
    for a in range(start_a, D, stride_a):
        if a == obsdim[obs_ind[a]]:
            for m in range(start_m, M, stride_m):
                dAdX[a, m] = Rm * (X[a, m] - Y[obs_ind[a], m])
        else:
            for m in range(start_m, M, stride_m):
                dAdX[a, m] = 0

    # 'model' part
    for m in range(start_m, M, stride_m):
        if m >= 1 and m <= M - 2:
            for a in range(start_a, D, stride_a):
                for i in range(D):
                    dAdX[a, m] -= Rf[i] * dt / 2 * jacobian[i, a, m] \
                                  * (diff[i, m-1] + diff[i, m])
                dAdX[a, m] += Rf[a] * (diff[a, m-1] - diff[a, m])
        elif m == 0:
            for a in range(start_a, D, stride_a):
                for i in range(D):
                    dAdX[a, m] \
                      -= Rf[i] * dt / 2 * jacobian[i, a, 0] * diff[i, 0]
                dAdX[a, m] -= Rf[a] * diff[a, 0]
        else:
            for a in range(start_a, D, stride_a):
                for i in range(D):
                    dAdX[a, m] \
                      -= Rf[i] * dt / 2 * jacobian[i, a, M-1] * diff[i, M-2]
                dAdX[a, m] += Rf[a] * diff[a, M-2]

    for a in range(start_a, D, stride_a):
        for m in range(start_m, M, stride_m):
            dAdX[a, m] *= scaling / M

@cuda.jit
def k__dAdpar(X, diff, dfield_dpar, Rf, scaling, dt, dAdpar):
    """
    This kernel calculates the derivatives of the action with respect to the 
    parameters 'par'. The parameter 'dAdpar' must be initialized before 
    lauching this kernel.

    Inputs
    ------
              X: D-by-M device array.
           diff: D-by-(M-1) device array.
    dfield_dpar: D-by-len(par)-by-M device array.
             Rf: one-dimensional (shapeless) device array of length D.
        scaling: scalar.
             dt: scalar.

    Modifications
    ------
    dAdpar: one-dimensional (shapeless) device array of length len(par).
    """
    start_i, start_b, start_m = cuda.grid(3)
    stride_i, stride_b, stride_m = cuda.gridsize(3)

    D, M = X.shape

    if start_i >= D or start_b >= len(dAdpar) or start_m >= M:
        return

    for i in range(start_i, D, stride_i):
        for b in range(start_b, len(dAdpar), stride_b):
            for m in range(start_m, M-1, stride_m):
                cuda.atomic.add(dAdpar, b, -scaling/M*Rf[i]*dt/2*diff[i, m]\
                                            *(dfield_dpar[i, b, m]\
                                              +dfield_dpar[i, b, m+1]))

@cuda.jit
def k__leapfrog_X(pX, epsilon, mass_X, X):
    """
    This kernel updates X as part of the leapfrog simulation procedure.

    Inputs
    ------
         pX: D-by-M device array.
    epsilon: scalar.
     mass_X: D-by-M device array.

    Modifications
    ------
    X: D-by-M device array.
    """
    start_a, start_m = cuda.grid(2)
    stride_a, stride_m = cuda.gridsize(2)

    D, M = X.shape

    if start_a >= D or start_m >= M:
        return

    for a in range(start_a, D, stride_a):
        for m in range(start_m, M, stride_m):
            X[a, m] += epsilon * pX[a, m] / mass_X[a, m]

@cuda.jit
def k__leapfrog_par(ppar, epsilon, mass_par, par):
    """
    This kernel updates 'par' as part of the leapfrog simulation procedure.

    Inputs
    ------
        ppar: one-dimensional (shapeless) device array of length len(par).
     epsilon: scalar.
    mass_par: one-dimensional (shapeless) device array of length len(par).

    Modifications
    ------
    par: one-dimensional (shapeless) device array.
    """
    start_b = cuda.grid(1)
    stride_b = cuda.gridsize(1)

    if start_b >= len(par):
        return

    for b in range(start_b, len(par), stride_b):
        par[b] += epsilon * ppar[b] / mass_par[b]

@cuda.jit
def k__linearop2d(Q, r, S, T):
    """
    This kernel calculates the linear equation T=Q+r*S for 2d arrays.

    Inputs
    ------
    Q: 2d device array.
    r: scalar.
    S: 2d device array.

    Modifications
    ------
    T: 2d device array. 
    """
    start_a, start_m = cuda.grid(2)
    stride_a, stride_m = cuda.gridsize(2)

    D, M = Q.shape

    if start_a >= D or start_m >= M:
        return

    for a in range(start_a, D, stride_a):
        for m in range(start_m, M, stride_m):
            T[a, m] = Q[a, m] + r * S[a, m]

@cuda.jit
def k__linearop1d(Q, r, S, T):
    """
    This kernel calculates the linear equation T=Q+r*S for 1d arrays.

    Inputs
    ------
    Q: one-dimensional (shapeless) device array.
    r: scalar.
    S: one-dimensional (shapeless) device array.

    Modifications
    ------
    T: one-dimensional (shapeless) device array.
    """
    start_b = cuda.grid(1)
    stride_b = cuda.gridsize(1)

    if start_b >= len(T):
        return

    for b in range(start_b, len(T), stride_b):
        T[b] = Q[b] + r * S[b]

@cuda.jit
def k__zeros1d(array):
    """
    This kernel sets the input array to zero. Doing it this way is more 
    efficient than transferring a zeros array to the GPU.

    Modifications
    ------
    array: one-dimensional (shapeless) device array. 
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    if start >= len(array):
        return

    for idx in range(start, len(array), stride):
        array[idx] = 0.0

