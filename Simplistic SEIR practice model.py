# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 13:13:45 2025

@author: Vincent Lomas
"""

import multi_factor_matrix_modules.modules as mfmm
import numpy as np
import scipy
import matplotlib.pyplot as plt

def sim_SEIR(t,SI):
    S,I = SI
    dS = -R0*S*I/N
    dI = R0*S*I/N - I
    
    return (dS,dI)

R0 = 1
I0_prop = 0.0001
N = 100
t_max = 400

SI0 = np.array([N*(1-I0_prop),N*I0_prop])

solution = scipy.integrate.solve_ivp(sim_SEIR,[0,t_max],SI0,t_eval=np.arange(0,t_max))

plt.plot(N-solution.y[0,:])