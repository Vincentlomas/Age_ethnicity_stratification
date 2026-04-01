# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 16:20:45 2025

@author: Vincent X. Lomas
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import multi_factor_matrix_modules.modules as mfmm
import scipy
import seaborn as sns

is_generate_results = False
is_plot = True
is_savefig = True

def heatplot(scenario_num,eth_assortativity_res, eth_contact_ratio_res,z_min=0.4,
             z_max=0.6, custom_title=None,has_xlabel=True, has_ylabel=True, has_y_tick_labels=True,has_x_tick_labels=True, has_color_bar=True,
             is_savefig=True, axes=None, return_plot_instance = False, ethnic_group=None, title_loc = 'center'):
    N,F,a = mfmm.scenario_parameters(scenario_num-1)
    
    num_age_groups, num_ethnic_groups = np.shape(N)
    
    C_matrices = np.zeros([eth_contact_ratio_res, eth_assortativity_res, num_age_groups, num_ethnic_groups, num_age_groups, num_ethnic_groups])
    
    epsilon_age= 0.3
    
    
    
    for eth_rel_idx in range(eth_contact_ratio_res):
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
        
        for eth_idx in range(eth_assortativity_res):
            epsilon_eth = eth_idx / (eth_assortativity_res -1)
            C_constructed = mfmm.return_C_matrix(epsilon_eth,Cij_k,F, N)
            
            C_matrices[eth_rel_idx, eth_idx,:,:,:,:] = C_constructed
            
    time = 300
    gamma = 2/3
    sigma = 1
    # attack rate matrix has shape (2,eth_res,eth_assort_res), the 2 is to store results for both ethnic groups
    attack_rate_matrix = np.zeros([2,eth_contact_ratio_res, eth_assortativity_res])
    ### Running SEIR model
    S,Sv,E,I,R, In = mfmm.initial_group_populations(N,is_vacc=False,pop_vec_vacc=np.array([]))
    
    # S,Sv,E,I,R, In = mfmm.initial_group_populations(N,is_vacc=False,pop_vec_vacc=np.array([]),initial_exposed=0)
    # S[0] -= 0.0001*np.sum(N)
    # E[0] = 0.0001*np.sum(N)
    
    
    for eth_rel_idx in range(eth_contact_ratio_res):
        for eth_idx in range(eth_assortativity_res):
            C_matrix = C_matrices[eth_rel_idx, eth_idx,:,:,:,:]
            
            # Convert C to per capita
            beta_matrix = mfmm.flatten_to_two_dim(C_matrix) / (N.T).flatten()
            solution = scipy.integrate.solve_ivp(mfmm.SEIR_model, [0,time],
                            np.concatenate(mfmm.initial_group_populations((N.T).flatten(),is_vacc=False,pop_vec_vacc=np.array([]),initial_exposed=0.0001)),
                            t_eval=np.arange(time+1), args = (beta_matrix, sigma, gamma))
            attack_rate_matrix[0,eth_rel_idx, eth_idx] = np.sum(solution.y[-10:-5,-1]) / np.sum(N[:,0])
            attack_rate_matrix[1,eth_rel_idx, eth_idx] = np.sum(solution.y[-5:,-1]) / np.sum(N[:,1])
    
    
    if has_y_tick_labels:
        y_labels = []
        pwrs = np.round(np.linspace(1.5,-1.5,eth_contact_ratio_res),1)
        for eth_rel_idx in range(eth_contact_ratio_res):
            if eth_rel_idx%2 and not abs(pwrs[eth_rel_idx]) < 10**-8:
                y_labels.append('')
            else:
                pwr = pwrs[eth_rel_idx]
                y_labels.append(np.round(2**pwr,3))
    else:
        y_labels = ['']*eth_contact_ratio_res
    
    
    if has_x_tick_labels:
        x_tick_labels=[]
        for i in range(eth_assortativity_res):
            if i%4:
                x_tick_labels.append('')
            else:
                x_tick_labels.append(np.round(i/(eth_assortativity_res-1),3))
    else:
        x_tick_labels=[None]*(eth_assortativity_res)
    
    if axes is None:
        fig, axes = plt.subplots()
    
    if ethnic_group is None:
        matrix_to_plot=np.zeros([eth_contact_ratio_res,eth_assortativity_res])
        for i in range(2):
            matrix_to_plot += attack_rate_matrix[i]*np.sum(N[:,i])/np.sum(N)
    else:
        matrix_to_plot = attack_rate_matrix[(ethnic_group-1),:,:]
    
    heatplot_instance = sns.heatmap(matrix_to_plot,ax=axes,
                cmap="viridis", xticklabels=x_tick_labels,
                yticklabels=y_labels,cbar=has_color_bar,
                rasterized=True, vmin=z_min, vmax=z_max)
    
    if custom_title is None:
        axes.set_title(f'Scenario {scenario_num}', loc=title_loc)
    else:
        axes.set_title(custom_title,loc=title_loc)
    
    if has_xlabel:
        axes.set_xlabel("Ethnic assortativity")
    if has_ylabel:
        axes.set_ylabel("Relative contact rate ratio")
    
    
    if np.min(matrix_to_plot) < z_min:
        print("WARNING: values below lower bound of colour bar")
        print(f"Current bound is {z_min} and lowest value is {np.min(matrix_to_plot)}")
    if np.max(matrix_to_plot) > z_max:
        print("WARNING: values above upper bound of colour bar")
        print(f"Current bound is {z_max} and highest value is {np.max(matrix_to_plot)}")
    
    if is_savefig:
        plt.savefig(f'images/heatplots/rel_contact_rate_vs_eth_epsilon/rel_contact_rate_vs_eth_epsilon_scen{scenario_num}.png',dpi=300)
    
    if return_plot_instance:
        return heatplot_instance

# Constructing the contact matrices


age_res = 5
eth_assortativity_res = 17
eth_contact_ratio_res = 7

# for scenario_num in [1,2,3,4,5,6,7,8,9,10]:
#     N,F,a = mfmm.scenario_parameters(scenario_num)
    
#     num_age_groups, num_ethnic_groups = np.shape(N)
    
#     C_matrices = np.zeros([age_res, eth_assortativity_res, num_age_groups, num_ethnic_groups, num_age_groups, num_ethnic_groups])
    
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
        
#         for eth_idx in range(eth_assortativity_res):
#             epsilon_eth = eth_idx / (eth_assortativity_res -1)
#             C_constructed = mfmm.return_C_matrix(epsilon_eth,Cij_k,F, N)
            
#             C_matrices[age_idx, eth_idx,:,:,:,:] = C_constructed
            
#     time = 100
#     gamma = 2/3
#     sigma = 1
#     attack_rate_matrix = np.zeros([age_res, eth_assortativity_res])
#     ### Running SEIR model
#     S,Sv,E,I,R, In = mfmm.initial_group_populations(N,is_vacc=False,pop_vec_vacc=np.array([]))
    
#     # S,Sv,E,I,R, In = mfmm.initial_group_populations(N,is_vacc=False,pop_vec_vacc=np.array([]),initial_exposed=0)
#     # S[0] -= 0.0001*np.sum(N)
#     # E[0] = 0.0001*np.sum(N)
    
    
#     for age_idx in range(age_res):
#         for eth_idx in range(eth_assortativity_res):
#             C_matrix = C_matrices[age_idx, eth_idx,:,:,:,:]
            
#             # Convert C to per capita
#             beta_matrix = mfmm.flatten_to_two_dim(C_matrix) / (N.T).flatten()
#             solution = scipy.integrate.solve_ivp(mfmm.SEIR_model, [0,time],
#                             np.concatenate(mfmm.initial_group_populations((N.T).flatten(),is_vacc=False,pop_vec_vacc=np.array([]),initial_exposed=0.0001)),
#                             t_eval=np.arange(time+1), args = (beta_matrix, sigma, gamma))
#             attack_rate_matrix[age_idx, eth_idx] = np.sum(solution.y[-10:,-1]) / np.sum(N)
    
#     heatplot = sns.heatmap(attack_rate_matrix,
#                 cmap="viridis", xticklabels=np.round(np.linspace(0,1,eth_assortativity_res),2),
#                 yticklabels=np.round(np.linspace(0,1,age_res),2),cbar=True,
#                 rasterized=True)
#     heatplot.set_title(f'Senario {scenario_num}')
#     heatplot.set_xlabel("Ethnic assortativity")
#     heatplot.set_ylabel("Age assortativity")
#     



### Ploting an example contact matrix for slideshow
# ex_matrix=np.zeros([4,4])
# ex_matrix = ex_matrix +0.3
# for i in range(4):
#     ex_matrix[i,i] = 1

# fig, axes = plt.subplots(figsize=[4,4])
# sns.heatmap(ex_matrix, axes=axes,
#             cmap="viridis", xticklabels=[1,2,3,4],
#             yticklabels=[1,2,3,4],cbar=False,
#             rasterized=True, vmin=0, vmax=1)
# plt.savefig('example_matrix.png', dpi=300)




# ### Ploting a heatplot of each scenario's attack rate when varying ethnic assortativity and realive ethnic rate ratio

# # for scenario_i in [1,2,3,4,5]:
# #     heatplot(scenario_i, eth_assortativity_res, eth_contact_ratio_res)

# fig, axs = plt.subplots(1,2, figsize=(8,3.5))
# min_val = 0
# max_val=1
# heatplot(1, eth_assortativity_res, eth_contact_ratio_res,axes=axs[0],
#          custom_title="a) Scenario 1 total",has_xlabel=False, has_color_bar=False,z_min=min_val,z_max=max_val,
#          is_savefig=False, title_loc='left')
# im = heatplot(5, eth_assortativity_res, eth_contact_ratio_res,axes=axs[1],
#          custom_title="b) Scenario 5 total", has_y_tick_labels=False,has_xlabel=False,z_min=min_val,z_max=max_val,
#          has_ylabel=False, has_color_bar=False,is_savefig=False, return_plot_instance=True, title_loc='left')
# #plt.xlabel('Ethnic assortativity')
# plt.tight_layout()
# fig.subplots_adjust(right=0.812,bottom=0.15)
# fig.text(0.445, 0.04, 'Ethnic assortativity', ha='center')
# cbar_ax = fig.add_axes([0.825, 0.15, 0.025, 0.7])
# norm = matplotlib.colors.Normalize(vmin=0.4, vmax=0.6)
# cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='viridis'),cax=cbar_ax,label='Attack rate (%)')
# cbar.set_ticks([0.4,0.45,0.5,0.55,0.6])
# cbar.set_ticklabels([40,45,50,55,60])
# plt.savefig('images/heatplots/rel_contact_rate_vs_eth_epsilon/rel_contact_rate_vs_eth_epsilon_comparrison.png', dpi=300,bbox_inches='tight')



# fig, axs = plt.subplots(2,2, figsize=(8,7))
# heatplot(1, eth_assortativity_res, eth_contact_ratio_res,axes=axs[0,0],
#          custom_title="a) Scenario 1, ethnic group 1",has_xlabel=False, has_color_bar=False,
#          is_savefig=False, ethnic_group=1,z_min=min_val,z_max=max_val,has_x_tick_labels=False,
#          has_ylabel=False, title_loc='left')
# heatplot(1, eth_assortativity_res, eth_contact_ratio_res,axes=axs[0,1],
#          custom_title="b) Scenario 1, ethnic group 2",has_xlabel=False, has_color_bar=False,
#          is_savefig=False, ethnic_group=2,z_min=min_val,z_max=max_val,has_y_tick_labels=False,
#          has_x_tick_labels=False,has_ylabel=False, title_loc='left')
# heatplot(5, eth_assortativity_res, eth_contact_ratio_res,axes=axs[1,0],
#          custom_title="c) Scenario 5, ethnic group 1", has_xlabel=False,
#          has_ylabel=False, has_color_bar=False,is_savefig=False, ethnic_group=1,z_min=min_val,z_max=max_val, title_loc='left')
# im = heatplot(5, eth_assortativity_res, eth_contact_ratio_res,axes=axs[1,1],
#          custom_title="d) Scenario 5, ethnic group 2", has_y_tick_labels=False,has_xlabel=False,
#          has_ylabel=False, has_color_bar=False,is_savefig=False, return_plot_instance=True,
#          ethnic_group=2,z_min=min_val,z_max=max_val, title_loc='left')
# #plt.xlabel('Ethnic assortativity')
# plt.tight_layout()
# fig.subplots_adjust(right=0.812,bottom=0.105,left=0.10)
# fig.text(0.445, 0.04, 'Ethnic assortativity', ha='center')
# fig.text(0.04, 0.41, 'Relative contact rate ratio', ha='center', rotation='vertical')
# cbar_ax = fig.add_axes([0.825, 0.15, 0.025, 0.7])
# norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
# cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='viridis'),cax=cbar_ax,label='Attack rate (%)')
# cbar.set_ticks(np.linspace(min_val,max_val,5))
# cbar.set_ticklabels(np.linspace(min_val*100,max_val*100,5).astype(int))
# plt.savefig('images/heatplots/rel_contact_rate_vs_eth_epsilon/rel_contact_rate_vs_eth_epsilon_ethnic_group_comparrison.png', dpi=300,bbox_inches='tight')




###
fig, axs = plt.subplots(2,3, figsize=(4*3,7))
min_val = 0
max_val=1
heatplot(1, eth_assortativity_res, eth_contact_ratio_res,axes=axs[0,0],
         custom_title="a) Scenario 1",has_xlabel=False, has_color_bar=False,
         is_savefig=False, title_loc='left',z_min=min_val,z_max=max_val,
         has_x_tick_labels=False,has_ylabel=False)
im = heatplot(5, eth_assortativity_res, eth_contact_ratio_res,axes=axs[1,0],
         custom_title="d) Scenario 5",has_xlabel=False,
         has_ylabel=False, has_color_bar=False,is_savefig=False,
         return_plot_instance=True, title_loc='left',z_min=min_val,z_max=max_val)
heatplot(1, eth_assortativity_res, eth_contact_ratio_res,axes=axs[0,1],
         custom_title="b) Scenario 1, ethnic group 1",has_xlabel=False, has_color_bar=False,
         is_savefig=False, ethnic_group=1,z_min=min_val,z_max=max_val,has_x_tick_labels=False,
         has_ylabel=False, has_y_tick_labels=False,title_loc='left')
heatplot(1, eth_assortativity_res, eth_contact_ratio_res,axes=axs[0,2],
         custom_title="c) Scenario 1, ethnic group 2",has_xlabel=False, has_color_bar=False,
         is_savefig=False, ethnic_group=2,z_min=min_val,z_max=max_val,has_y_tick_labels=False,
         has_x_tick_labels=False,has_ylabel=False, title_loc='left')
heatplot(5, eth_assortativity_res, eth_contact_ratio_res,axes=axs[1,1],
         custom_title="e) Scenario 5, ethnic group 1", has_xlabel=False,has_y_tick_labels=False,
         has_ylabel=False, has_color_bar=False,is_savefig=False, ethnic_group=1,
         z_min=min_val,z_max=max_val, title_loc='left')
im = heatplot(5, eth_assortativity_res, eth_contact_ratio_res,axes=axs[1,2],
         custom_title="f) Scenario 5, ethnic group 2", has_y_tick_labels=False,has_xlabel=False,
         has_ylabel=False, has_color_bar=False,is_savefig=False, return_plot_instance=True,
         ethnic_group=2,z_min=min_val,z_max=max_val, title_loc='left')
#plt.xlabel('Ethnic assortativity')
plt.tight_layout()
fig.subplots_adjust(right=0.812,bottom=0.105,left=0.10)
fig.text(0.447, 0.04, 'Ethnic assortativity', ha='center')
fig.text(0.065, 0.41, 'Relative contact rate ratio', ha='center', rotation='vertical')
cbar_ax = fig.add_axes([0.825, 0.15, 0.025, 0.7])
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='viridis'),cax=cbar_ax,label='Attack rate (%)')
cbar.set_ticks(np.linspace(min_val,max_val,5))
cbar.set_ticklabels(np.linspace(min_val*100,max_val*100,5).astype(int))
plt.savefig('images/heatplots/rel_contact_rate_vs_eth_epsilon/rel_contact_rate_vs_eth_epsilon_all_comparision.png', dpi=300,bbox_inches='tight')