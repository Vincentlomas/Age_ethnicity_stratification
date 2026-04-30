# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 16:20:45 2025

@author: Vincent X. Lomas

This code runs SEIR models while varying relative contact rate and Scio-demographic assortativity
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

def heatplot(scenario_num,socio_demo_assortativity_res, socio_demo_contact_ratio_res,z_min=0.4,
             z_max=0.6, custom_title=None,has_xlabel=True, has_ylabel=True,
             has_y_tick_labels=True,has_x_tick_labels=True, has_color_bar=True,
             is_savefig=True, axes=None, return_plot_instance = False,
             socio_demographic_group=None, title_loc = 'center'):
    '''Function that plots a heat plot of the SEIR model final attack rate given 
    a scenario number and some parameter values.
    
    Input:
        scenario_num: the scenaio number to consider (see paper text for table
            of values for each scenario)
        socio_demo_assortativity_res: A positive int. A number determiining the
            resolution  (or number of point to consider) for the socio-demographic
            assortativity which is the preference to socially interact within
            ones own socio-demographic group
        socio_demo_contact_ratio_res: A positive int. A number determiining the
            resolution  (or number of point to consider) for the socio-demographic
            relative contact rate ratio (or the magnitude difference between
            the socio-demographic group social contact rates)
        z_min: A float between 0 and 1. the minimum value for the colormap/bar.
            A warning will be printed if there are any results lower than this bound.
        z_max: A float between 0 and 1. the maximum value for the colormap/bar.
            A warning will be printed if there are any results larger than this bound.
        custom_title: str or None. The title of the plot, if None there will be no title
        has_xlabel: boolean. If true will set the x-axis title to "Socio-demographic assortativity"
        has_ylabel: boolean. If true will set the x-axis title to "Relative contact rate ratio"
        has_y_tick_labels: boolean. If False it will supress y-tick labels on the plot.
        has_x_tick_labels:boolean. If False it will supress x-tick labels on the plot.
        has_color_bar: boolean. If turn will add a colour bar to the plot with 
            bounds specified by z_min and z_max.
        is_savefig: boolean. If True will save the figure as a png to thefollowing
            filepath "images/heatplots/rel_contact_rate_vs_socio_demo_epsilon/" with the
            following filename "rel_contact_rate_vs_socio_demo_epsilon_scen{scenario_num}.png"
        axes: matplotlib.axes._axes.Axes. The axes to plot onto.
        return_plot_instance: boolean. Will return an instant of the heat plot
            as matplotlib.axes._axes.Axes
        socio_demographic_group: None or int of 1 or 2. If None plot the attack rate
            of the whole population, if 1 or 2, plots the attack rate of the 
            respective socio-demographic group
        title_loc: str. the location of the title.
        '''
    # Grab information about scenarios
    N,F,a = mfmm.scenario_parameters(scenario_num-1)
    num_age_groups, num_socio_demographic_groups = np.shape(N)
    
    # Make the social contact matrix storage array
    C_matrices = np.zeros([socio_demo_contact_ratio_res, socio_demo_assortativity_res, num_age_groups, num_socio_demographic_groups, num_age_groups, num_socio_demographic_groups])
    
    epsilon_age= 0.3
    
    
    ### Iterate over each relative ethnic rate ratio and assortativity rate and construct a matrix
    for socio_demo_rel_idx in range(socio_demo_contact_ratio_res):
        F= np.array([1,2**((socio_demo_rel_idx-3)/2)])
        Pij = np.zeros([num_age_groups,num_age_groups])
        for i in range(num_age_groups):
            for j in range(num_age_groups):
                Pij[i,j] = (1-epsilon_age)*a[i]*a[j]/np.sum(np.sum(N,axis=1)*a)
                if i==j:
                    Pij[i,j]+=epsilon_age*a[j]/np.sum(N[j,:])
                
        Cij_k = np.zeros([num_age_groups,num_age_groups])
        for j in range(num_age_groups):
            Cij_k[:,j] = Pij[:,j] * np.sum(N[j,:])
        
        for socio_demo_idx in range(socio_demo_assortativity_res):
            epsilon_eth = socio_demo_idx / (socio_demo_assortativity_res -1)
            C_constructed = mfmm.return_C_matrix(epsilon_eth,Cij_k,F, N)
            
            C_matrices[socio_demo_rel_idx, socio_demo_idx,:,:,:,:] = C_constructed
    
    # Speicfy SEIR model parameters
    time = 300
    gamma = 2/3
    sigma = 1
    # attack rate matrix has shape (2,socio_demo_res,socio_demo_assort_res), the 2 is to store results for both socio-demographic groups
    attack_rate_matrix = np.zeros([2,socio_demo_contact_ratio_res, socio_demo_assortativity_res])
    ### Running SEIR model
    S,Sv,E,I,R, In = mfmm.initial_group_populations(N,is_vacc=False,pop_vec_vacc=np.array([]))
    
    # Iterate over all constructed matrices and run an SEIR model
    for socio_demo_rel_idx in range(socio_demo_contact_ratio_res):
        for socio_demo_idx in range(socio_demo_assortativity_res):
            C_matrix = C_matrices[socio_demo_rel_idx, socio_demo_idx,:,:,:,:]
            
            # Convert C to per capita
            beta_matrix = mfmm.flatten_to_two_dim(C_matrix) / (N.T).flatten()
            solution = scipy.integrate.solve_ivp(mfmm.SEIR_model, [0,time],
                            np.concatenate(mfmm.initial_group_populations((N.T).flatten(),is_vacc=False,pop_vec_vacc=np.array([]),initial_exposed=0.0001)),
                            t_eval=np.arange(time+1), args = (beta_matrix, sigma, gamma))
            # Store attack rates for each socio-demographic group
            attack_rate_matrix[0,socio_demo_rel_idx, socio_demo_idx] = np.sum(solution.y[-10:-5,-1]) / np.sum(N[:,0])
            attack_rate_matrix[1,socio_demo_rel_idx, socio_demo_idx] = np.sum(solution.y[-5:,-1]) / np.sum(N[:,1])
    
    ### Start of plotting code
    if has_y_tick_labels:
        y_labels = []
        pwrs = np.round(np.linspace(1.5,-1.5,socio_demo_contact_ratio_res),1)
        for socio_demo_rel_idx in range(socio_demo_contact_ratio_res):
            pwr = pwrs[socio_demo_rel_idx]
            y_labels.append(f"{np.round(2**pwr,2):.2f}")
    else:
        y_labels = ['']*socio_demo_contact_ratio_res
    
    if has_x_tick_labels:
        x_tick_labels=[]
        for i in range(socio_demo_assortativity_res):
            if i%4:
                x_tick_labels.append('')
            else:
                x_tick_labels.append(np.round(i/(socio_demo_assortativity_res-1),3))
    else:
        x_tick_labels=[None]*(socio_demo_assortativity_res)
    
    if axes is None:
        fig, axes = plt.subplots()
    
    if socio_demographic_group is None:
        matrix_to_plot=np.zeros([socio_demo_contact_ratio_res,socio_demo_assortativity_res])
        for i in range(2):
            matrix_to_plot += attack_rate_matrix[i]*np.sum(N[:,i])/np.sum(N)
    else:
        matrix_to_plot = attack_rate_matrix[(socio_demographic_group-1),:,:]
    
    heatplot_instance = sns.heatmap(matrix_to_plot,ax=axes,
                cmap="viridis", yticklabels=y_labels, xticklabels=x_tick_labels,
                cbar=has_color_bar, rasterized=True, vmin=z_min, vmax=z_max)
    axes.tick_params(axis='y',rotation=0) 
    
    if custom_title is None:
        axes.set_title(f'Scenario {scenario_num}', loc=title_loc)
    else:
        axes.set_title(custom_title,loc=title_loc)
    
    if has_xlabel:
        axes.set_xlabel("Socio-demographic assortativity")
    if has_ylabel:
        axes.set_ylabel("Relative contact rate ratio",labelpad=25)
    
    
    if np.min(matrix_to_plot) < z_min:
        print("WARNING: values below lower bound of colour bar")
        print(f"Current bound is {z_min} and lowest value is {np.min(matrix_to_plot)}")
    if np.max(matrix_to_plot) > z_max:
        print("WARNING: values above upper bound of colour bar")
        print(f"Current bound is {z_max} and highest value is {np.max(matrix_to_plot)}")
    
    if is_savefig:
        plt.savefig(f'images/heatplots/rel_contact_rate_vs_socio_demo_epsilon/rel_contact_rate_vs_socio_demo_epsilon_scen{scenario_num}.png',dpi=300)
    
    if return_plot_instance:
        return heatplot_instance

# Constructing the contact matrices


age_res = 5
socio_demo_assortativity_res = 17
socio_demo_contact_ratio_res = 7


###
fig, axs = plt.subplots(2,3, figsize=(4*3,7))
min_val = 0
max_val=1
heatplot(1, socio_demo_assortativity_res, socio_demo_contact_ratio_res,axes=axs[0,0],
         custom_title="a) Scenario 1",has_xlabel=False, has_color_bar=False,
         is_savefig=False, title_loc='left',z_min=min_val,z_max=max_val,
         has_x_tick_labels=False,has_ylabel=False)
im = heatplot(5, socio_demo_assortativity_res, socio_demo_contact_ratio_res,axes=axs[1,0],
         custom_title="d) Scenario 5",has_xlabel=False,
         has_ylabel=False, has_color_bar=False,is_savefig=False,
         return_plot_instance=True, title_loc='left',z_min=min_val,z_max=max_val)
heatplot(1, socio_demo_assortativity_res, socio_demo_contact_ratio_res,axes=axs[0,1],
         custom_title="b) Scenario 1, ethnic group 1",has_xlabel=False, has_color_bar=False,
         is_savefig=False, socio_demographic_group=1,z_min=min_val,z_max=max_val,has_x_tick_labels=False,
         has_ylabel=False, has_y_tick_labels=False,title_loc='left')
heatplot(1, socio_demo_assortativity_res, socio_demo_contact_ratio_res,axes=axs[0,2],
         custom_title="c) Scenario 1, ethnic group 2",has_xlabel=False, has_color_bar=False,
         is_savefig=False, socio_demographic_group=2,z_min=min_val,z_max=max_val,has_y_tick_labels=False,
         has_x_tick_labels=False,has_ylabel=False, title_loc='left')
heatplot(5, socio_demo_assortativity_res, socio_demo_contact_ratio_res,axes=axs[1,1],
         custom_title="e) Scenario 5, ethnic group 1", has_xlabel=False,has_y_tick_labels=False,
         has_ylabel=False, has_color_bar=False,is_savefig=False, socio_demographic_group=1,
         z_min=min_val,z_max=max_val, title_loc='left')
im = heatplot(5, socio_demo_assortativity_res, socio_demo_contact_ratio_res,axes=axs[1,2],
         custom_title="f) Scenario 5, ethnic group 2", has_y_tick_labels=False,has_xlabel=False,
         has_ylabel=False, has_color_bar=False,is_savefig=False, return_plot_instance=True,
         socio_demographic_group=2,z_min=min_val,z_max=max_val, title_loc='left')
#plt.xlabel('Ethnic assortativity')
plt.tight_layout()
fig.subplots_adjust(right=0.812,bottom=0.105,left=0.10)
fig.text(0.447, 0.04, 'Socio-demographic assortativity', ha='center')
fig.text(0.05, 0.41, 'Relative contact rate ratio', ha='center', rotation='vertical')
cbar_ax = fig.add_axes([0.825, 0.15, 0.025, 0.7])
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='viridis'),cax=cbar_ax,label='Attack rate (%)')
cbar.set_ticks(np.linspace(min_val,max_val,5))
cbar.set_ticklabels(np.linspace(min_val*100,max_val*100,5).astype(int))
plt.savefig('images/heatplots/rel_contact_rate_vs_socio_demo_epsilon/rel_contact_rate_vs_socio_demo_epsilon_all_comparision.png', dpi=300,bbox_inches='tight')