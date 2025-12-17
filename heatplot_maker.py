# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 16:20:45 2025

@author: Vincent X. Lomas
"""

import numpy as np
import matplotlib.pyplot as plt
import multi_factor_matrix_modules.modules as mfmm
import scipy
import seaborn as sns

is_generate_results = True
is_plot = True
is_savefig = False

# Constructing the contact matrices


age_res = 5
eth_res = 4

# for scenario_num in [1,2,3,4,5,6,7,8,9,10]:
#     N,F,a = mfmm.scenario_parameters(scenario_num)
    
#     num_age_groups, num_ethnic_groups = np.shape(N)
    
#     C_matrices = np.zeros([age_res, eth_res, num_age_groups, num_ethnic_groups, num_age_groups, num_ethnic_groups])
    
#     for age_idx in range(age_res):
#         epsilon_age = age_idx / (age_res-1)
        
#         Pij = np.zeros([num_age_groups,num_age_groups])
#         for i in range(num_age_groups):
#             for j in range(num_age_groups):
#                 Pij[i,j] = (1-epsilon_age)*a[i]*a[j]/np.sum(np.sum(N,axis=1)*a)
#                 if i==j:
#                     Pij[i,j]+=epsilon_age*a[j]/np.sum(N[j,:])
                
#         Cij_k = np.zeros([num_age_groups,num_age_groups])
#         for j in range(num_age_groups):
#             Cij_k[:,j] = Pij[:,j] * np.sum(N[j,:])
        
#         for eth_idx in range(eth_res):
#             epsilon_eth = eth_idx / (eth_res -1)
#             C_constructed = mfmm.return_C_matrix(epsilon_eth,Cij_k,F, N)
            
#             C_matrices[age_idx, eth_idx,:,:,:,:] = C_constructed
            
#     time = 100
#     gamma = 2/3
#     sigma = 1
#     attack_rate_matrix = np.zeros([age_res, eth_res])
#     ### Running SEIR model
#     S,Sv,E,I,R, In = mfmm.initial_group_populations(N,is_vacc=False,pop_vec_vacc=np.array([]))
    
#     # S,Sv,E,I,R, In = mfmm.initial_group_populations(N,is_vacc=False,pop_vec_vacc=np.array([]),initial_exposed=0)
#     # S[0] -= 0.0001*np.sum(N)
#     # E[0] = 0.0001*np.sum(N)
    
    
#     for age_idx in range(age_res):
#         for eth_idx in range(eth_res):
#             C_matrix = C_matrices[age_idx, eth_idx,:,:,:,:]
            
#             # Convert C to per capita
#             beta_matrix = mfmm.flatten_to_two_dim(C_matrix) / (N.T).flatten()
#             solution = scipy.integrate.solve_ivp(mfmm.SEIR_model, [0,time],
#                             np.concatenate(mfmm.initial_group_populations((N.T).flatten(),is_vacc=False,pop_vec_vacc=np.array([]),initial_exposed=0.0001)),
#                             t_eval=np.arange(time+1), args = (beta_matrix, sigma, gamma))
#             attack_rate_matrix[age_idx, eth_idx] = np.sum(solution.y[-10:,-1]) / np.sum(N)
    
#     heatplot = sns.heatmap(attack_rate_matrix,
#                 cmap="viridis", xticklabels=np.round(np.linspace(0,1,eth_res),2),
#                 yticklabels=np.round(np.linspace(0,1,age_res),2),cbar=True,
#                 rasterized=True)
#     heatplot.set_title(f'Senario {scenario_num}')
#     heatplot.set_xlabel("Ethnic assortativity")
#     heatplot.set_ylabel("Age assortativity")
#     plt.show()
    

curr_min = 1
curr_max = 0


eth_rel_res = 7
for scenario_num in [1,2,3,4,5,6,7,8,9,10]:
    N,F,a = mfmm.scenario_parameters(scenario_num)
    
    num_age_groups, num_ethnic_groups = np.shape(N)
    
    C_matrices = np.zeros([eth_rel_res, eth_res, num_age_groups, num_ethnic_groups, num_age_groups, num_ethnic_groups])
    
    epsilon_age= 0.3
    
    
    
    for eth_rel_idx in range(eth_rel_res):
        F= np.array([1,2**((eth_rel_idx-3)/2)])
        Pij = np.zeros([num_age_groups,num_age_groups])
        for i in range(num_age_groups):
            for j in range(num_age_groups):
                Pij[i,j] = (1-epsilon_age)*a[i]*a[j]/np.sum(np.sum(N,axis=1)*a)
                if i==j:
                    Pij[i,j]+=epsilon_age*a[j]/np.sum(N[j,:])
                
        Cij_k = np.zeros([num_age_groups,num_age_groups])
        for j in range(num_age_groups):
            Cij_k[:,j] = Pij[:,j] * np.sum(N[j,:])
        
        for eth_idx in range(eth_res):
            epsilon_eth = eth_idx / (eth_res -1)
            C_constructed = mfmm.return_C_matrix(epsilon_eth,Cij_k,F, N)
            
            C_matrices[eth_rel_idx, eth_idx,:,:,:,:] = C_constructed
            
    time = 300
    gamma = 2/3
    sigma = 1
    attack_rate_matrix = np.zeros([eth_rel_res, eth_res])
    ### Running SEIR model
    S,Sv,E,I,R, In = mfmm.initial_group_populations(N,is_vacc=False,pop_vec_vacc=np.array([]))
    
    # S,Sv,E,I,R, In = mfmm.initial_group_populations(N,is_vacc=False,pop_vec_vacc=np.array([]),initial_exposed=0)
    # S[0] -= 0.0001*np.sum(N)
    # E[0] = 0.0001*np.sum(N)
    
    
    for eth_rel_idx in range(eth_rel_res):
        for eth_idx in range(eth_res):
            C_matrix = C_matrices[eth_rel_idx, eth_idx,:,:,:,:]
            
            # Convert C to per capita
            beta_matrix = mfmm.flatten_to_two_dim(C_matrix) / (N.T).flatten()
            solution = scipy.integrate.solve_ivp(mfmm.SEIR_model, [0,time],
                            np.concatenate(mfmm.initial_group_populations((N.T).flatten(),is_vacc=False,pop_vec_vacc=np.array([]),initial_exposed=0.0001)),
                            t_eval=np.arange(time+1), args = (beta_matrix, sigma, gamma))
            attack_rate_matrix[eth_rel_idx, eth_idx] = np.sum(solution.y[-10:,-1]) / np.sum(N)
    
    heatplot = sns.heatmap(attack_rate_matrix,
                cmap="viridis", xticklabels=np.round(np.linspace(0,1,eth_res),2),
                yticklabels=np.round(np.linspace(-1.5,1.5,eth_rel_res),1),cbar=True,
                rasterized=True, vmin=0.4, vmax=0.6)
    heatplot.set_title(f'Senario {scenario_num}')
    heatplot.set_xlabel("Ethnic assortativity")
    heatplot.set_ylabel("Relative contact rate (log_2)")
    plt.show()
    
    
    curr_min = np.min([curr_min,np.min(attack_rate_matrix)])
    curr_max = np.max([curr_max,np.max(attack_rate_matrix)])