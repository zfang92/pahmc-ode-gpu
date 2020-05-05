# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This module contains all the built-in dynamics, each being a bundle of CUDA 
kernels, that is ready for deployment. If the user inputs a name (should be all 
lowercase) that has a match here, the 'def_dynamics' module will be ignored and
the corresponding kernels will be fetched into __init__.Fetch via main.py.

The name of each kernel below has form 'k__<name>_field/jacobian/dfield_dpar'. 
Future added kernels should be named this way.

Only three kernels need to be implemented for each dynamics. See below.
    1) k__<name>_field(X, par, stimulus, field):
        Inputs
        ------
               X: D-by-M GPU device array for any positive integer M.
             par: one-dimensional (shapeless) device array.
        stimulus: D-by-M device array for any positive integer M.

        Modifications
        -------
        field: D-by-M device array.

    2) k__<name>_jacobian(X, par, jacobian):
        Inputs
        ------
          X: D-by-M device array for any positive integer M.
        par: one-dimensional (shapeless) device array.

        Modifications
        -------
        jacobian: D-by-D-by-M device array for any positive integer M.

    3) k__<name>_dfield_dpar(X, par, dfield_dpar):
        Inputs
        ------
          X: D-by-M device array for any positive integer M.
        par: one-dimensional (shapeless) device array.

        Modifications
        -------
        dfield_dpar: D-by-len(par)-by-M device array. Each index in the third 
                     axis corresponds to a D-by-M device array that contains 
                     the derivatives with respect to the path X.
"""


import math

from numba import cuda


#===================================Lorenz96====================================
"""
Below implements the standard Lorenz96 model. Fortunately, there is only one 
representation of the model.
"""

@cuda.jit
def k__lorenz96_field(X, par, stimulus, field):
    start_a, start_m = cuda.grid(2)
    stride_a, stride_m = cuda.gridsize(2)

    D, M = X.shape

    if start_a >= D or start_m >= M:
        return

    for a in range(start_a, D, stride_a):
        if a == 0:
            for m in range(start_m, M, stride_m):
                field[a, m] \
                  = (X[1, m] - X[D-2, m]) * X[D-1, m] - X[0, m] + par[0]
        elif a == 1:
            for m in range(start_m, M, stride_m):
                field[a, m] \
                  = (X[2, m] - X[D-1, m]) * X[0, m] - X[1, m] + par[0]
        elif a == D - 1:
            for m in range(start_m, M, stride_m):
                field[a, m] \
                  = (X[0, m] - X[D-3, m]) * X[D-2, m] - X[D-1, m] + par[0]
        else:
            for m in range(start_m, M, stride_m):
                field[a, m] \
                  = (X[a+1, m] - X[a-2, m]) * X[a-1, m] - X[a, m] + par[0]

@cuda.jit
def k__lorenz96_jacobian(X, par, jacobian):
    start_i, start_j, start_m = cuda.grid(3)
    stride_i, stride_j, stride_m = cuda.gridsize(3)

    D, M = X.shape

    if start_i >= D or start_j >= D or start_m >= M:
        return
    
    for i in range(start_i+1, D+1, stride_i):
        for j in range(start_j+1, D+1, stride_j):
            for m in range(start_m, M, stride_m):
                jacobian[i-1, j-1, m] \
                  = (1 + (i - 2) % D == j) \
                    * (X[i%D, m] - X[(i-3)%D, m]) \
                    + ((1 + i % D == j) - (1 + (i - 3) % D == j)) \
                    * X[(i-2)%D, m] - (i == j)

@cuda.jit
def k__lorenz96_dfield_dpar(X, par, dfield_dpar):
    start_i, start_b, start_m = cuda.grid(3)
    stride_i, stride_b, stride_m = cuda.gridsize(3)

    D, M = X.shape

    if start_i >= D or start_b >= len(par) or start_m >= M:
        return

    for i in range(start_i, D, stride_i):
        for b in range(start_b, len(par), stride_b):
            for m in range(start_m, M, stride_m):
                dfield_dpar[i, b, m] = 1


#=====================================NaKL======================================
"""
Below implements the Hodgkin-Huxley model as described in Toth et al., 
Biological Cybernetics (2011). It has 18 parameters as follows:
g_Na, E_Na, g_K, E_K, g_L, E_L; 
Vm, dVm, tau_m0, tau_m1; 
Vh, dVh, tau_h0, tau_h1;
Vn, dVn, tau_n0, tau_n1.
"""

@cuda.jit
def k__nakl_field(X, par, stimulus, field):
    start_a, start_m = cuda.grid(2)
    stride_a, stride_m = cuda.gridsize(2)

    D, M = X.shape

    if start_a >= D or start_m >= M:
        return

    for a in range(start_a, D, stride_a):
        if a == 0:
            for m in range(start_m, M, stride_m):
                field[a, m] \
                  = stimulus[0, m] \
                    + par[0] * (X[1, m] ** 3) * X[2, m] * (par[1] - X[0, m]) \
                    + par[2] * (X[3, m] ** 4) * (par[3] - X[0, m]) \
                    + par[4] * (par[5] - X[0, m])
        if a == 1:
            for m in range(start_m, M, stride_m):
                tanh_m = math.tanh((X[0, m]-par[6])/par[7])
                field[a, m] = ((1 + tanh_m) / 2 - X[1, m]) \
                              / (par[8] + par[9] * (1 - tanh_m * tanh_m))
        if a == 2:
            for m in range(start_m, M, stride_m):
                tanh_h = math.tanh((X[0, m]-par[10])/par[11])
                field[a, m] = ((1 + tanh_h) / 2 - X[2, m]) \
                              / (par[12] + par[13] * (1 - tanh_h * tanh_h))
        if a == 3:
            for m in range(start_m, M, stride_m):
                tanh_n = math.tanh((X[0, m]-par[14])/par[15])
                field[a, m] = ((1 + tanh_n) / 2 - X[3, m]) \
                              / (par[16] + par[17] * (1 - tanh_n * tanh_n))

@cuda.jit
def k__nakl_jacobian(X, par, jacobian):
    start_i, start_j, start_m = cuda.grid(3)
    stride_i, stride_j, stride_m = cuda.gridsize(3)

    D, M = X.shape

    if start_i >= D or start_j >= D or start_m >= M:
        return
    
    for i in range(start_i, D, stride_i):
        for j in range(start_j, D, stride_j):
            if i == 0 and j == 0:
                for m in range(start_m, M, stride_m):
                    jacobian[i, j, m] = - par[0] * (X[1, m] ** 3) * X[2, m] \
                                        - par[2] * (X[3, m] ** 4) - par[4]
            elif i == 0 and j == 1:
                for m in range(start_m, M, stride_m):
                    jacobian[i, j, m] = 3 * par[0] * (X[1, m] ** 2) \
                                        * X[2, m] * (par[1] - X[0, m])
            elif i == 0 and j == 2:
                for m in range(start_m, M, stride_m):
                    jacobian[i, j, m] \
                      = par[0] * (X[1, m] ** 3) * (par[1] - X[0, m])
            elif i == 0 and j == 3:
                for m in range(start_m, M, stride_m):
                    jacobian[i, j, m] \
                      = 4 * par[2] * (X[3, m] ** 3) * (par[3] - X[0, m])
            elif i == 1 and j == 0:
                for m in range(start_m, M, stride_m):
                    tanh_m = math.tanh((X[0, m]-par[6])/par[7])
                    kernel_m = 1 - tanh_m * tanh_m
                    tau_m = par[8] + par[9] * kernel_m
                    jacobian[i, j, m] \
                      = kernel_m / (2 * par[7]) / tau_m \
                        - 2 * par[9] / par[7] * tanh_m * kernel_m \
                          * (X[1, m] - (1 + tanh_m) / 2) / (tau_m * tau_m)
            elif i == 2 and j == 0:
                for m in range(start_m, M, stride_m):
                    tanh_h = math.tanh((X[0, m]-par[10])/par[11])
                    kernel_h = 1 - tanh_h * tanh_h
                    tau_h = par[12] + par[13] * kernel_h
                    jacobian[i, j, m] \
                      = kernel_h / (2 * par[11]) / tau_h \
                        - 2 * par[13] / par[11] * tanh_h * kernel_h \
                          * (X[2, m] - (1 + tanh_h) / 2) / (tau_h * tau_h)
            elif i == 3 and j == 0:
                for m in range(start_m, M, stride_m):
                    tanh_n = math.tanh((X[0, m]-par[14])/par[15])
                    kernel_n = 1 - tanh_n * tanh_n
                    tau_n = par[16] + par[17] * kernel_n
                    jacobian[i, j, m] \
                      = kernel_n / (2 * par[15]) / tau_n \
                        - 2 * par[17] / par[15] * tanh_n * kernel_n \
                          * (X[3, m] - (1 + tanh_n) / 2) / (tau_n * tau_n)
            elif i == 1 and j == 1:
                for m in range(start_m, M, stride_m):
                    tanh_m = math.tanh((X[0, m]-par[6])/par[7])
                    jacobian[i, j, m] \
                      = - 1 / (par[8] + par[9] * (1 - tanh_m * tanh_m))
            elif i == 2 and j == 2:
                for m in range(start_m, M, stride_m):
                    tanh_h = math.tanh((X[0, m]-par[10])/par[11])
                    jacobian[i, j, m] \
                      = - 1 / (par[12] + par[13] * (1 - tanh_h * tanh_h))
            elif i == 3 and j == 3:
                for m in range(start_m, M, stride_m):
                    tanh_n = math.tanh((X[0, m]-par[14])/par[15])
                    jacobian[i, j, m] \
                      = - 1 / (par[16] + par[17] * (1 - tanh_n * tanh_n))
            else:
                for m in range(start_m, M, stride_m):
                    jacobian[i, j, m] = 0

@cuda.jit
def k__nakl_dfield_dpar(X, par, dfield_dpar):
    start_i, start_b, start_m = cuda.grid(3)
    stride_i, stride_b, stride_m = cuda.gridsize(3)

    D, M = X.shape

    if start_i >= D or start_b >= len(par) or start_m >= M:
        return

    for i in range(start_i, D, stride_i):
        for b in range(start_b, len(par), stride_b):
            if i == 0 and b == 0:
                for m in range(start_m, M, stride_m):
                    dfield_dpar[i, b, m] \
                      = (X[1, m] ** 3) * X[2, m] * (par[1] - X[0, m])
            elif i == 0 and b == 1:
                for m in range(start_m, M, stride_m):
                    dfield_dpar[i, b, m] = par[0] * (X[1, m] ** 3) * X[2, m]
            elif i == 0 and b == 2:
                for m in range(start_m, M, stride_m):
                    dfield_dpar[i, b, m] = (X[3, m] ** 4) * (par[3] - X[0, m])
            elif i == 0 and b == 3:
                for m in range(start_m, M, stride_m):
                    dfield_dpar[i, b, m] = par[2] * (X[3, m] ** 4)
            elif i == 0 and b == 4:
                for m in range(start_m, M, stride_m):
                    dfield_dpar[i, b, m] = par[5] - X[0, m]
            elif i == 0 and b == 5:
                for m in range(start_m, M, stride_m):
                    dfield_dpar[i, b, m] = par[4]
            elif i == 1 and b == 6:
                for m in range(start_m, M, stride_m):
                    tanh_m = math.tanh((X[0, m]-par[6])/par[7])
                    kernel_m = 1 - tanh_m * tanh_m
                    tau_m = par[8] + par[9] * kernel_m
                    dfield_dpar[i, b, m] \
                      = 2 * par[9] / par[7] * tanh_m * kernel_m \
                        * (X[1, m] - (1 + tanh_m) / 2) / (tau_m * tau_m) \
                        - kernel_m / (2 * par[7]) / tau_m
            elif i == 1 and b == 7:
                for m in range(start_m, M, stride_m):
                    tanh_m = math.tanh((X[0, m]-par[6])/par[7])
                    kernel_m = 1 - tanh_m * tanh_m
                    tau_m = par[8] + par[9] * kernel_m
                    dfield_dpar[i, b, m] \
                      = 2 * par[9] * (X[0, m] - par[6]) / (par[7] ** 2) \
                        * tanh_m * kernel_m * (X[1, m] - (1 + tanh_m) / 2) \
                        / (tau_m * tau_m) \
                        - (X[0, m] - par[6]) \
                          / (2 * (par[7] ** 2)) * kernel_m / tau_m
            elif i == 1 and b == 8:
                for m in range(start_m, M, stride_m):
                    tanh_m = math.tanh((X[0, m]-par[6])/par[7])
                    tau_m = par[8] + par[9] * (1 - tanh_m * tanh_m)
                    dfield_dpar[i, b, m] \
                      = (X[1, m] - (1 + tanh_m) / 2) / (tau_m * tau_m)
            elif i == 1 and b == 9:
                for m in range(start_m, M, stride_m):
                    tanh_m = math.tanh((X[0, m]-par[6])/par[7])
                    kernel_m = 1 - tanh_m * tanh_m
                    tau_m = par[8] + par[9] * kernel_m
                    dfield_dpar[i, b, m] \
                      = kernel_m * (X[1, m] - (1 + tanh_m) / 2) \
                        / (tau_m * tau_m)
            elif i == 2 and b == 10:
                for m in range(start_m, M, stride_m):
                    tanh_h = math.tanh((X[0, m]-par[10])/par[11])
                    kernel_h = 1 - tanh_h * tanh_h
                    tau_h = par[12] + par[13] * kernel_h
                    dfield_dpar[i, b, m] \
                      = 2 * par[13] / par[11] * tanh_h * kernel_h \
                        * (X[2, m] - (1 + tanh_h) / 2) / (tau_h * tau_h) \
                        - kernel_h / (2 * par[11]) / tau_h
            elif i == 2 and b == 11:
                for m in range(start_m, M, stride_m):
                    tanh_h = math.tanh((X[0, m]-par[10])/par[11])
                    kernel_h = 1 - tanh_h * tanh_h
                    tau_h = par[12] + par[13] * kernel_h
                    dfield_dpar[i, b, m] \
                      = 2 * par[13] * (X[0, m] - par[10]) / (par[11] ** 2) \
                        * tanh_h * kernel_h * (X[2, m] - (1 + tanh_h) / 2) \
                        / (tau_h * tau_h) \
                        - (X[0, m] - par[10]) \
                          / (2 * (par[11] ** 2)) * kernel_h / tau_h
            elif i == 2 and b == 12:
                for m in range(start_m, M, stride_m):
                    tanh_h = math.tanh((X[0, m]-par[10])/par[11])
                    tau_h = par[12] + par[13] * (1 - tanh_h * tanh_h)
                    dfield_dpar[i, b, m] \
                      = (X[2, m] - (1 + tanh_h) / 2) / (tau_h * tau_h)
            elif i == 2 and b == 13:
                for m in range(start_m, M, stride_m):
                    tanh_h = math.tanh((X[0, m]-par[10])/par[11])
                    kernel_h = 1 - tanh_h * tanh_h
                    tau_h = par[12] + par[13] * kernel_h
                    dfield_dpar[i, b, m] \
                      = kernel_h * (X[2, m] - (1 + tanh_h) / 2) \
                        / (tau_h * tau_h)
            elif i == 3 and b == 14:
                for m in range(start_m, M, stride_m):
                    tanh_n = math.tanh((X[0, m]-par[14])/par[15])
                    kernel_n = 1 - tanh_n * tanh_n
                    tau_n = par[16] + par[17] * kernel_n
                    dfield_dpar[i, b, m] \
                      = 2 * par[17] / par[15] * tanh_n * kernel_n \
                        * (X[3, m] - (1 + tanh_n) / 2) / (tau_n * tau_n) \
                        - kernel_n / (2 * par[15]) / tau_n
            elif i == 3 and b == 15:
                for m in range(start_m, M, stride_m):
                    tanh_n = math.tanh((X[0, m]-par[14])/par[15])
                    kernel_n = 1 - tanh_n * tanh_n
                    tau_n = par[16] + par[17] * kernel_n
                    dfield_dpar[i, b, m] \
                      = 2 * par[17] * (X[0, m] - par[14]) / (par[15] ** 2) \
                        * tanh_n * kernel_n * (X[3, m] - (1 + tanh_n) / 2) \
                        / (tau_n * tau_n) \
                        - (X[0, m] - par[14]) \
                          / (2 * (par[15] ** 2)) * kernel_n / tau_n
            elif i == 3 and b == 16:
                for m in range(start_m, M, stride_m):
                    tanh_n = math.tanh((X[0, m]-par[14])/par[15])
                    tau_n = par[16] + par[17] * (1 - tanh_n * tanh_n)
                    dfield_dpar[i, b, m] \
                      = (X[3, m] - (1 + tanh_n) / 2) / (tau_n * tau_n)
            elif i == 3 and b == 17:
                for m in range(start_m, M, stride_m):
                    tanh_n = math.tanh((X[0, m]-par[14])/par[15])
                    kernel_n = 1 - tanh_n * tanh_n
                    tau_n = par[16] + par[17] * kernel_n
                    dfield_dpar[i, b, m] \
                      = kernel_n * (X[3, m] - (1 + tanh_n) / 2) \
                        / (tau_n * tau_n)
            else:
                for m in range(start_m, M, stride_m):
                    dfield_dpar[i, b, m] = 0

