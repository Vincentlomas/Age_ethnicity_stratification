# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 12:48:26 2025

@author: Vincent Lomas

Numerical analysis of new contact matrix construction method

0) Push X more extreme, make entries negative
Done 1) keep in mind to try diff age groups having diff contact rates - linear, oldest 50% of youngest contact rate (tack onto last age structure)
Done 2) Put together a func to calculate R0
Done 3) Start working with cum prop of I in each (i,a) group
Done 3.a) Start plotting this as a function of assortativity and relative contact rate
 - can multiply matri by a constant to scale R0
Done 3.b) average attack rate for each ethnic group (and less relevantly age)
"""

import multi_factor_matrix_modules.modules as mfmm
import numpy as np
import matplotlib.pyplot as plt
import scipy

def legend_with_extra(solid_name,dashed_name, solid_line=None, dashed_line=None,ax=None):
    # Add solid and dashed line to legend
    if solid_line is None:
        solid_line = plt.Line2D([0], [0], color='gray', linestyle='-', label=solid_name)
    if dashed_line is None:
        dashed_line = plt.Line2D([0], [0], color='gray', linestyle='--', label=dashed_name)
    if ax is None:
        handles, labels = plt.gca().get_legend_handles_labels()
    else:
        handles, labels = ax.get_legend_handles_labels()
    handles.append(solid_line)
    handles.append(dashed_line)
    labels.append(solid_name)
    labels.append(dashed_name)
    
    if ax is None:
        plt.legend(handles, labels)
    else:
        ax.legend(handles, labels)

def variance_plot(x_vals, variance_array,scenarios_to_plot, x_axis_title,
                  y_axis_title, num_matrices, num_ethnic_groups, num_age_groups,
                  is_save_fig, filename =None, reproductive_numbers=None,is_xlog=False):
    '''Input:
        x_vals: 1D array corresponding to xvalues used for plotting
        variance_array: Array of variance results of shape (number of scenarios,
            plot number, y values)
        scenarios to plot: list of scenarios to plot (scenarios being one more 
            than their index). These scenarioa will be added as a line to every
            graph
        x_axis_title: Title of x axis
        y_axis_title: Title of y axis
        is_save_fig: Boolean, True when saving plots, False if not
        is_xlog: boolean, True if x axis logged on plots
    
    Outputs:
        A series of plots about matrices
    '''
    
    plt.figure()
    plt.title('Whole population')
    for scen in scenarios_to_plot:
        plt.plot(x_vals,variance_array[(scen-1),0,:], label=f"Scenario {scen}")
    plt.legend()
    if is_xlog:
        plt.xscale('log')
    plt.ylabel(y_axis_title)
    plt.xlabel(x_axis_title)
    if is_save_fig:
        plt.savefig(f'images/{filename}_whole_population.png', dpi=300)
    plt.show()
    
    fig, axes = plt.subplots(1,num_ethnic_groups,figsize=(6*num_ethnic_groups, 4))
    for i in range(num_ethnic_groups):
        axes[i].set_title(f'Ethnic group {i+1}')
        for scen in scenarios_to_plot:
            axes[i].plot(x_vals,variance_array[(scen-1),(1+i),:], label=f"Scenario {scen}")
        axes[i].legend()
        axes[i].set_ylim([0,1])
        axes[i].set_xlim([np.min(x_vals),np.max(x_vals)])
        if is_xlog:
            axes[i].set_xscale('log')
        axes[i].set_ylabel(y_axis_title)
        axes[i].set_xlabel(x_axis_title)
    if is_save_fig:
        plt.savefig(f'images/{filename}_ethnic.png', dpi=300)
    plt.show()
    
    fig, axes = plt.subplots(1,num_age_groups,figsize=(6*num_age_groups, 4))
    for i in range(num_age_groups):
        axes[i].set_title(f'Age group {i+1}')
        for scen in scenarios_to_plot:
            axes[i].plot(x_vals,variance_array[(scen-1),(1+ num_ethnic_groups +i),:], label=f"Scenario {scen}")
        axes[i].legend()
        axes[i].set_ylim([0,1])
        axes[i].set_xlim([np.min(x_vals),np.max(x_vals)])
        if is_xlog:
            axes[i].set_xscale('log')
        axes[i].set_ylabel(y_axis_title)
        axes[i].set_xlabel(x_axis_title)
    if is_save_fig:
        plt.savefig(f'images/{filename}_age.png', dpi=300)
    plt.show()
    
    fig, axes = plt.subplots(num_ethnic_groups,num_age_groups,figsize=(6*num_age_groups, 5*num_ethnic_groups ))
    axes = axes.flatten()
    for i in range(num_age_groups*num_ethnic_groups):
        axes[i].set_title(f'Age group {i%5+1}, Ethnic group {(i*2)//num_matrices+1}')
        for scen in scenarios_to_plot:
            axes[i].plot(x_vals,variance_array[(scen-1),(1+num_ethnic_groups +num_age_groups +i),:], label=f"Scenario {scen}")
        axes[i].legend()
        if is_xlog:
            axes[i].set_xscale('log')
        axes[i].set_ylim([0,1])
        axes[i].set_xlim([np.min(x_vals),np.max(x_vals)])
        axes[i].set_ylabel(y_axis_title)
        axes[i].set_xlabel(x_axis_title)
    plt.tight_layout()
    if is_save_fig:
        plt.savefig(f'images/{filename}_age_ethnic.png', dpi=300)
    plt.show()
    
    if not reproductive_numbers is None:
        plt.figure()
        plt.title('Whole population')
        for scen in scenarios_to_plot:
            plt.plot(x_vals,reproductive_numbers[(scen-1),:], label=f"Scenario {scen}")
        plt.legend()
        if is_xlog:
            plt.xscale('log')
        plt.ylabel('Basic Reproductive Number')
        plt.xlabel(x_axis_title)
    if is_save_fig:
        plt.savefig(f'images/{filename}_R0.png', dpi=300)
    plt.show()
    
    
num_ethnic_groups = 2
num_age_groups =5

epsilon = 0.5
gamma = 2/3
sigma = 1/3 # Rate of disease development
time = 300

C_storage_prop = []
C_storage_assort = []
C_storage_list = []

num_matrices = 10

scenario_reproductive_numbers = np.zeros(num_matrices)

c = 0.3 # Age based assortativity

is_save_figs = True

is_plot_matrices = True
is_plot_SEIR = False

is_run_relative_contact_rate_variance = False
is_plot_relative_contact_rate_variance = True

is_run_R0_variance = False
is_plot_R0_variance = False

is_run_epsilon_variance = False
is_plot_epsilon_variance = True

scenarios_to_plot = [6,7,8,9,10]

for k in range(num_matrices):
    N,F,a = mfmm.scenario_parameters(k)
    
    Pij = np.zeros([num_age_groups,num_age_groups])
    for i in range(num_age_groups):
        for j in range(num_age_groups):
            Pij[i,j] = (1-c)*a[i]*a[j]/np.sum(np.sum(N,axis=1)*a)
            if i==j:
                Pij[i,j]+=c*a[j]/np.sum(N[j,:])
            
    Cij_k = np.zeros([num_age_groups,num_age_groups])
    for j in range(num_age_groups):
        Cij_k[:,j] = Pij[:,j] * np.sum(N[j,:])
    
    if k ==0:
        Cij = Cij_k.copy()
    
    C_constructed_prop = mfmm.return_C_matrix_proportionate(Cij_k,F, N)
    C_constructed_assort = mfmm.return_C_matrix_assortative(Cij_k,F, N)
    C_constructed = mfmm.return_C_matrix(epsilon,Cij_k,F, N)
    
    C_storage_list.append(C_constructed)
    
    scenario_reproductive_numbers[k] = mfmm.initial_reproduction_number(mfmm.flatten_to_two_dim(C_constructed), gamma = gamma)


### Check that constraints on matrix are satisfied (only check last matrix constructed)
print()
print()
print('-'*90)
print('Construction attempt')
print('-'*90)
mfmm.condition_checking_fixed(C_constructed, Cij_k, N)



# Heat plot of matrices
if is_plot_matrices:
    fig, axes = plt.subplots(2, num_matrices//2, figsize=(22, 8))
    axes = axes.flatten()
    vmin=0
    vmax=np.max(C_storage_list)
    for i, C_matrix in enumerate(C_storage_list):
        im = axes[i].imshow(mfmm.flatten_to_two_dim(C_matrix),vmin=vmin, vmax=vmax,cmap='viridis', aspect='auto')
        axes[i].set_title(f'{"abcdefghijklmnop"[i]})')
        # Turn off ticks
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Contact rate')
    plt.show()
    
    fig, axes = plt.subplots(2, (num_matrices//2), figsize=(22, 8))
    axes = axes.flatten()
    for i, C_matrix in enumerate(C_storage_list):
        im = axes[i].imshow(mfmm.flatten_to_two_dim(C_matrix)/np.max(C_matrix),vmin=vmin, vmax=1,cmap='viridis', aspect='auto')
        axes[i].set_title(f'{"abcdefghijklmnop"[i]})')
        # Turn off ticks
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Contact rate')
    if is_save_figs:
        plt.savefig(f'heatplot_matrix_comparison_epsilon_{epsilon}.png', dpi=300)
    plt.show()




### PLOT PARAMETERS
colour_wheel = ['red','blue','green','purple','teal']


if is_plot_matrices:
    solutions =[]
    
    ### Run SEIR models
    for k, C_matrix in enumerate(C_storage_list):
        # Grab initial populations - Note that N becomes a 1D array that has all ages within an ethnicity before moving onto the next ethnicity
        # e.g. using N_ia: N=[N_00,N_10,N_20,N_01,N_11,N_21]
        N, F, a = mfmm.scenario_parameters(k)
        N = (N.T).flatten()
            
        # Convert C to per capita
        beta_matrix = mfmm.flatten_to_two_dim(C_matrix) / (N.T).flatten()
        # Check if symmetric
        if np.all(abs(beta_matrix - beta_matrix.T) < 10**-9):
            
            S,Sv,E,I,R, In = mfmm.initial_group_populations(N,is_vacc=False,pop_vec_vacc=np.array([]),initial_exposed=0.0001)
            
            solution = scipy.integrate.solve_ivp(mfmm.SEIR_model, [0,time],
                            np.concatenate(mfmm.initial_group_populations(N,is_vacc=False,pop_vec_vacc=np.array([]),initial_exposed=0.0001)),
                            t_eval=np.arange(time+1), args = (beta_matrix, sigma, gamma))
            
            solutions.append(solution.y)
            
        else:
            print(k)
            print('Beta is not symmetric')
    
    
    # Getting age solution
    N_vec_age = np.sum(mfmm.population_examples(0),axis=1)
    SEIR_0_age = np.concatenate(mfmm.initial_group_populations(N_vec_age.flatten(),is_vacc=False,pop_vec_vacc=np.array([]),initial_exposed=0.0001))
    solution = scipy.integrate.solve_ivp(mfmm.SEIR_model, [0,time], SEIR_0_age, t_eval=np.arange(time+1), args = (Cij/(N_vec_age.flatten()), sigma, gamma))
    
    #structure_labels = ['Flat age structure', 'different age structure, same group size', 'Flat age structure, different group size', 'different age structure and group size','different age structure and group size\n different age contact rates']
    structure_labels =[]
    for i in range(num_matrices//2):
        structure_labels.append(f'Scenario {i+1}')
    
    for p in range(3):
        S0,S1 = [(0,time),(35,45),(time-10,time+1)][p]
        plt.figure()
        for k in range(0,num_matrices,1):
                
            if (2*k)//num_matrices ==0:
                plt.plot(np.arange(S0,S1),np.sum(solutions[k][50:,:],axis=0)[S0:S1],
                         label=structure_labels[k],
                         color=colour_wheel[k])
            else:
                plt.plot(np.arange(S0,S1),np.sum(solutions[k][50:,:],axis=0)[S0:S1],linestyle=(0, (3, 6)), linewidth=3,color=colour_wheel[k%(num_matrices//2)])
        
        # Plot solution with no ethnic consideration
        plt.plot(np.arange(S0,S1),np.sum(solution.y[25:,:],axis=0)[S0:S1],linestyle=':',label = 'No eth soln', linewidth = 3)
        plt.ylabel('Population')
        plt.xlabel('Day')
        legend_with_extra('Same contact rates','1/2 contact rates',dashed_line=plt.Line2D([0], [0], color='gray', linestyle=(0, (3, 6)), linewidth = 3, label='Different age structure'))
        plt.title("Cumulative number of infectious people")
        if p== 0:
            plt.savefig("images/SEIR_cum_number_of_infectious.png",dpi=300)
        
    
    fig, axes = plt.subplots(2,3,figsize=[12,6])
    axes = axes.flatten()
    plt.suptitle("Cumulative proportion of infectious people in the youngest age group")
    for k in range(5):
        for l in range(2):
            if k >1:
                idx = k +1
            else:
                idx = k
            axes[idx].plot(solutions[k+5*l][50,:]/mfmm.population_examples(k)[0,0] ,label=['Same contact rates', '1/2 contact rates'][l],color=colour_wheel[l])
            axes[idx].plot(solutions[k+5*l][55,:]/mfmm.population_examples(k)[0,1], linestyle='--',color=colour_wheel[l])
    
        axes[idx].set_title(structure_labels[k])
    
    # Putting the legend on a seperate subplot
    ax_legend = axes[2]
    ax_legend.axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color='gray', linestyle='-', label='Ethnic group 1'))
    handles.append(plt.Line2D([0], [0], color='gray', linestyle='--', label='Ethnic group 2'))
    labels.append('Ethnic group 1')
    labels.append('Ethnic group 2')
    ax_legend.legend(handles, labels, loc='center',
        fontsize=12,
        markerscale=3,
        handlelength=3,
        borderpad=1,
        labelspacing=1,
        handleheight=2,
        frameon=True,
        fancybox=False,
        shadow=False)
    
    plt.tight_layout()
    
    plt.figure()
    for k in range(num_matrices//2):
        idx =k
        k +=5
        plt.plot(solutions[k][50,:]/mfmm.population_examples(idx)[0,0],label=structure_labels[k%(num_matrices//2)],color=colour_wheel[k%(num_matrices//2)])
        plt.plot(solutions[k][55,:]/mfmm.population_examples(idx)[0,1], linestyle='--',color=colour_wheel[k%(num_matrices//2)])
    
    legend_with_extra('Ethnic group 1', 'Ethnic group 2')
    plt.title("Cumulative proportion of infectious people in the youngest age group\n (Different ethnic contact rates)")
    
    

### Change in relative contact rate

if is_run_relative_contact_rate_variance:
    
    # Want this to be odd to have zero in linspace - not required
    resolution = 41
    F_var_attack_rates = np.zeros([num_matrices//2,1+num_age_groups+num_ethnic_groups+num_age_groups*num_ethnic_groups, resolution])
    reproductive_numbers = np.zeros([num_matrices//2,resolution])
    
    F_vals = 2**np.linspace(-5, 5, resolution)
    
    for l in range(num_matrices//2):
        
        N,F,a = mfmm.scenario_parameters(l)
        
        Pij = np.zeros([num_age_groups,num_age_groups])
        for i in range(num_age_groups):
            for j in range(num_age_groups):
                Pij[i,j] = (1-c)*a[i]*a[j]/np.sum(np.sum(N,axis=1)*a)
                if i==j:
                    Pij[i,j]+=c*a[j]/np.sum(N[j,:])
                
        Cij = np.zeros([num_age_groups,num_age_groups])
        for j in range(num_age_groups):
            Cij[:,j] = Pij[:,j] * np.sum(N[j,:])
        
        
        for k in range(resolution):
            
            # Specify ethnic contact rates and grab contact matrix
            F = np.array([F_vals[k],1])
            C = mfmm.return_C_matrix(epsilon,Cij,F, N)
            
            reproductive_numbers[l,k] = mfmm.initial_reproduction_number(mfmm.flatten_to_two_dim(C), gamma = gamma)
            
            mfmm.condition_checking_fixed(C, Cij, N,is_shorthand=True)
            
            # Convert C to per capita
            beta_matrix = mfmm.flatten_to_two_dim(C) / (N.T).flatten()
            solution = scipy.integrate.solve_ivp(mfmm.SEIR_model, [0,time],
                            np.concatenate(mfmm.initial_group_populations((N.T).flatten(),is_vacc=False,pop_vec_vacc=np.array([]),initial_exposed=0.0001)),
                            t_eval=np.arange(time+1), args = (beta_matrix, sigma, gamma))
            
            F_var_attack_rates[l,0,k] = np.sum(solution.y[50:,-1])/np.sum(N)
            F_var_attack_rates[l,1,k] = np.sum(solution.y[50:55,-1])/np.sum(N[:,0])
            F_var_attack_rates[l,2,k] = np.sum(solution.y[55:,-1])/np.sum(N[:,1])
            for i in range(num_age_groups):
                F_var_attack_rates[l,(3+i),k] = (solution.y[(50+i),-1]+solution.y[(55+i),-1])/np.sum(N[i,:])
            for i in range(num_age_groups*num_ethnic_groups):
                F_var_attack_rates[l,(8+i),k] = solution.y[(50+i),-1]/np.sum(N[i%num_age_groups,i//num_age_groups])
    
    np.save('generated_results/ethnic_contact_ratio_variance_attack_rate_eth_epsilon{epsilon}_age_epsilon_{c}.npy', F_var_attack_rates)
    np.save('generated_results/ethnic_contact_ratio_variance_reprod_num_eth_epsilon{epsilon}_age_epsilon_{c}.npy', reproductive_numbers)
    np.save('generated_results/ethnic_contact_ratio_variance_Fvals_eth_epsilon{epsilon}_age_epsilon_{c}.npy', F_vals)

if is_plot_relative_contact_rate_variance:
    
    F_var_attack_rates = np.load('generated_results/ethnic_contact_ratio_variance_attack_rate_eth_epsilon{epsilon}_age_epsilon_{c}.npy')
    reproductive_numbers = np.load('generated_results/ethnic_contact_ratio_variance_reprod_num_eth_epsilon{epsilon}_age_epsilon_{c}.npy')
    F_vals = np.load('generated_results/ethnic_contact_ratio_variance_Fvals_eth_epsilon{epsilon}_age_epsilon_{c}.npy')
    
    relative_contact_rate_scens =[]
    
    for scen in scenarios_to_plot:
        if not scen in relative_contact_rate_scens:
            relative_contact_rate_scens.append(scen-5)
    
    variance_plot(F_vals, F_var_attack_rates,scenarios_to_plot=relative_contact_rate_scens,
                  x_axis_title='Ratio of ethnic contact rates (F1/F2)',
                  y_axis_title='Attack rates',
                  num_matrices=num_matrices, num_ethnic_groups=num_ethnic_groups,
                  num_age_groups=num_age_groups, is_save_fig = is_save_figs,
                  filename=f'relative_contact_rate_variation/relative_contact_rate_variation_eth_epsilon{epsilon}_age_epsilon_{c}',
                  reproductive_numbers=reproductive_numbers,is_xlog=True)
    


if is_run_R0_variance:
    
    resolution = 301
    R0_var_attack_rates = np.zeros([num_matrices,1+num_age_groups+num_ethnic_groups+num_age_groups*num_ethnic_groups, resolution])
    
    R0_vals = np.linspace(0,3, resolution)
    
    for l in range(num_matrices):
        
        N,F,a = mfmm.scenario_parameters(l)
        
        Pij = np.zeros([num_age_groups,num_age_groups])
        for i in range(num_age_groups):
            for j in range(num_age_groups):
                Pij[i,j] = (1-c)*a[i]*a[j]/np.sum(np.sum(N,axis=1)*a)
                if i==j:
                    Pij[i,j]+=c*a[j]/np.sum(N[j,:])
        # Scale to have basic reproductive number of 1
        # Pij = Pij / scenario_reproductive_numbers[l]
                
        Cij = np.zeros([num_age_groups,num_age_groups])
        for j in range(num_age_groups):
            Cij[:,j] = Pij[:,j] * np.sum(N[j,:])
        
        C = mfmm.return_C_matrix(epsilon,Cij,F, N)
        
        mfmm.condition_checking_fixed(C, Cij, N,is_shorthand=True)
        
        for k in range(resolution):
            
            # Specify epsilon and grab contact matrix
            R0 = R0_vals[k]
            
            # Convert C to per capita
            beta_matrix = mfmm.flatten_to_two_dim(C) / (N.T).flatten()
            beta_matrix = beta_matrix * R0
            
            # Run SEIR model
            solution = scipy.integrate.solve_ivp(mfmm.SEIR_model, [0,time],
                            np.concatenate(mfmm.initial_group_populations((N.T).flatten(),is_vacc=False,pop_vec_vacc=np.array([]),initial_exposed=0.0001)),
                            t_eval=np.arange(time+1), args = (beta_matrix, sigma, gamma))
            
            R0_var_attack_rates[l,0,k] = np.sum(solution.y[50:,-1])/np.sum(N)
            R0_var_attack_rates[l,1,k] = np.sum(solution.y[50:55,-1])/np.sum(N[:,0])
            R0_var_attack_rates[l,2,k] = np.sum(solution.y[55:,-1])/np.sum(N[:,1])
            for i in range(num_age_groups):
                R0_var_attack_rates[l,(3+i),k] = (solution.y[(50+i),-1]+solution.y[(55+i),-1])/np.sum(N[i,:])
            for i in range(num_age_groups*num_ethnic_groups):
                R0_var_attack_rates[l,(8+i),k] = solution.y[(50+i),-1]/np.sum(N[i%num_age_groups,i//num_age_groups])
    
    np.save('generated_results/R0_variance_attack_rate_age_epsilon_{c}.npy', R0_var_attack_rates)
    np.save('generated_results/R0_variance_R0_vals_age_epsilon_{c}.npy', R0_vals)


if is_plot_R0_variance:
    R0_var_attack_rates = np.load('generated_results/R0_variance_attack_rate_age_epsilon_{c}.npy')
    R0_vals = np.load('generated_results/R0_variance_R0_vals_age_epsilon_{c}.npy')
    
    variance_plot(R0_vals, R0_var_attack_rates,scenarios_to_plot=scenarios_to_plot,
                  x_axis_title='R0', y_axis_title='Attack rates',
                  num_matrices=num_matrices, num_ethnic_groups=num_ethnic_groups,
                  num_age_groups=num_age_groups, is_save_fig = is_save_figs,is_xlog=False)
    
    
    

if is_run_epsilon_variance:
    
    resolution = 31
    e_var_attack_rates = np.zeros([num_matrices,1+num_age_groups+num_ethnic_groups+num_age_groups*num_ethnic_groups, resolution])
    reproductive_numbers = np.zeros([num_matrices,resolution])
    
    e_vals = np.linspace(0,1, resolution)
    
    for l in range(num_matrices):
        
        N,F,a = mfmm.scenario_parameters(l)
        
        Pij = np.zeros([num_age_groups,num_age_groups])
        for i in range(num_age_groups):
            for j in range(num_age_groups):
                Pij[i,j] = (1-c)*a[i]*a[j]/np.sum(np.sum(N,axis=1)*a)
                if i==j:
                    Pij[i,j]+=c*a[j]/np.sum(N[j,:])
                
        Cij = np.zeros([num_age_groups,num_age_groups])
        for j in range(num_age_groups):
            Cij[:,j] = Pij[:,j] * np.sum(N[j,:])
        
        
        for k in range(resolution):
            
            # Specify epsilon and grab contact matrix
            epsilon = e_vals[k]
            C = mfmm.return_C_matrix(epsilon,Cij,F, N)
            
            reproductive_numbers[l,k] = mfmm.initial_reproduction_number(mfmm.flatten_to_two_dim(C), gamma = gamma)
            
            mfmm.condition_checking_fixed(C, Cij, N,is_shorthand=True)
            
            # Convert C to per capita
            beta_matrix = mfmm.flatten_to_two_dim(C) / (N.T).flatten()
            solution = scipy.integrate.solve_ivp(mfmm.SEIR_model, [0,time],
                            np.concatenate(mfmm.initial_group_populations((N.T).flatten(),is_vacc=False,pop_vec_vacc=np.array([]),initial_exposed=0.0001)),
                            t_eval=np.arange(time+1), args = (beta_matrix, sigma, gamma))
            
            e_var_attack_rates[l,0,k] = np.sum(solution.y[50:,-1])/np.sum(N)
            e_var_attack_rates[l,1,k] = np.sum(solution.y[50:55,-1])/np.sum(N[:,0])
            e_var_attack_rates[l,2,k] = np.sum(solution.y[55:,-1])/np.sum(N[:,1])
            for i in range(num_age_groups):
                e_var_attack_rates[l,(3+i),k] = (solution.y[(50+i),-1]+solution.y[(55+i),-1])/np.sum(N[i,:])
            for i in range(num_age_groups*num_ethnic_groups):
                e_var_attack_rates[l,(8+i),k] = solution.y[(50+i),-1]/np.sum(N[i%num_age_groups,i//num_age_groups])
    
    np.save('generated_results/eth_epsilon_variance_attack_rate_age_epsilon_{c}.npy', e_var_attack_rates)
    np.save('generated_results/eth_epsilon_variance_reprod_num_age_epsilon_{c}.npy', reproductive_numbers)
    np.save('generated_results/eth_epsilon_variance_e_vals_age_epsilon_{c}.npy', e_vals)


if is_plot_epsilon_variance:
    e_var_attack_rates = np.load('generated_results/eth_epsilon_variance_attack_rate_age_epsilon_{c}.npy')
    reproductive_numbers = np.load('generated_results/eth_epsilon_variance_reprod_num_age_epsilon_{c}.npy')
    e_vals = np.load('generated_results/eth_epsilon_variance_e_vals_age_epsilon_{c}.npy')
    
    variance_plot(e_vals, e_var_attack_rates,scenarios_to_plot=scenarios_to_plot,
                  x_axis_title='Ethnic epsilon value', y_axis_title='Attack rates',
                  num_matrices=num_matrices, num_ethnic_groups=num_ethnic_groups,
                  num_age_groups=num_age_groups, is_save_fig = is_save_figs,
                  filename=f'epsilon_variation/epsilon_var_age_epsilon_{c}',
                  reproductive_numbers=reproductive_numbers,is_xlog=False)

    