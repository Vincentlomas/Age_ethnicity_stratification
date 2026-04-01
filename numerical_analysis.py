# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 12:48:26 2025

@author: Vincent Lomas

Numerical analysis of new contact matrix construction method

"""

from cycler import cycler
import multi_factor_matrix_modules.modules as mfmm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as pyplottk
import scipy

def legend_with_extra(solid_name,dashed_name, solid_line=None, dashed_line=None,ax=None):
    '''
    Inputs: solid_name: str, name of first line added to legend
        dashed_name: str, name of second line added to legend
        solid_line: matplotlib.lines.Line2D, the first line to be added to the
            end of the legend, if None defaults to a solid grey line
        dashed_line: matplotlib.lines.Line2D, the second line to be added to the
            end of the legend, if None defaults to a dashed grey line
        ax: matplotlib.axes._axes.Axes, the axes to add the legend to, if None
            treats it like adding a lenend to a normal plot.
    Adds a legend to a plot (or axes) as pyplot typically does and adds a grey 
    solid and dashed line to the end'''
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
                  is_save_fig, filename =None, reproductive_numbers=None,is_xlog=False,
                  is_line_at_1=False, is_horizontal_line_at_1=False):
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
        A series of plots about matrices used in the paper
    '''
    
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    fig.text(0.5, 0.04, x_axis_title, ha='center')
    
    ### 3 image plot code (4 images if basic reproductive numbers were specified)
    title_fontsize = 16
    ### plot reproduction numbers
    if not reproductive_numbers is None:
        
        primary_plot_idx = 1
        if is_horizontal_line_at_1:
            axes[0,0].axhline(1, color='black', linestyle='dotted')
        axes[0,0].set_title('a)', loc='left', fontsize=title_fontsize)
        axes[0,1].set_title('b)', loc='left', fontsize=title_fontsize)
        axes[1,0].set_title('c)',loc='left', fontsize=title_fontsize)
        axes[1,1].set_title('d)',loc='left', fontsize=title_fontsize)
        for scen in scenarios_to_plot:
            axes[0,0].plot(x_vals,reproductive_numbers[(scen-1),:], label=f"Scenario {scen}")
        if is_xlog:
            axes[0,0].set_xscale('log')
            min_2_pwr = int(np.ceil(np.log(min(x_vals))/np.log(2)))
            max_2_pwr = int(np.floor(np.log(max(x_vals))/np.log(2)))
            
            # List of tick labels 
            tick_labels = []
            for pwr in range(min_2_pwr, max_2_pwr+1,1):
                # tick_labels.append(f'$2^{{{pwr}}}$') # used for power of two labeling (not used right now)
                # tick_labels.append(np.round(2**pwr,4))
                tick_labels.append('') # don't plot anything on the x ticks of this plot
            axes[0,0].get_xaxis().set_major_formatter(pyplottk.ScalarFormatter())
            axes[0,0].set_xticks(2.0**np.arange(min_2_pwr,max_2_pwr+1,1),labels=tick_labels)
            axes[0,0].get_xaxis().set_tick_params(which='minor', size=0)
            axes[0,0].get_xaxis().set_tick_params(which='minor', width=0) 
        axes[0,0].set_ylabel('Basic Reproductive Number')
        axes[0,0].set_xlim(min(x_vals),max(x_vals))
    else:
        primary_plot_idx = 0
        axes[0,0].set_title('a)', loc='left', fontsize=title_fontsize)
        axes[1,0].set_title('b)',loc='left', fontsize=title_fontsize)
        axes[1,1].set_title('c)',loc='left', fontsize=title_fontsize)
        
        box = axes[0,0].get_position()
        box.x0 = box.x0 + 0.21
        box.x1 = box.x1 + 0.21
        axes[0,0].set_position(box)
        
        axes[0,1].axis('off')
    
    handles = []
    for scen in scenarios_to_plot:
        line, = axes[0,primary_plot_idx].plot(x_vals,variance_array[(scen-1),0,:], label=f"Scenario {scen}")
        handles.append(line)
        
    if not reproductive_numbers is None:
        leg = axes[0,1].legend(handles, ["Scenario 1","Scenario 2","Scenario 3","Scenario 4","Scenario 5"], bbox_to_anchor=(0.98, 0.5)) #,loc = 'lower right')
    else:
        leg = axes[0,1].legend(handles, ["Scenario 1","Scenario 2","Scenario 3","Scenario 4","Scenario 5"], bbox_to_anchor=(0.98, 0.5),loc = 'center right',prop={'size': 16})
    
    for line in leg.get_lines():
        line.set_linewidth(5.0)
    
    if is_xlog:
        axes[0,1].set_xscale('log')
        min_2_pwr = int(np.ceil(np.log(min(x_vals))/np.log(2)))
        max_2_pwr = int(np.floor(np.log(max(x_vals))/np.log(2)))
        # List of tick labels 
        tick_labels = []
        for pwr in range(min_2_pwr, max_2_pwr+1,1):
            # tick_labels.append(f'$2^{{{pwr}}}$') # used for power of two labeling (not used right now)
            # tick_labels.append(np.round(2**pwr,4))
            tick_labels.append('') # don't plot anything on the x ticks of this plot
        axes[0,primary_plot_idx].get_xaxis().set_major_formatter(pyplottk.ScalarFormatter())
        axes[0,primary_plot_idx].set_xticks(2.0**np.arange(min_2_pwr,max_2_pwr+1,1),labels=tick_labels)
        axes[0,primary_plot_idx].get_xaxis().set_tick_params(which='minor', size=0)
        axes[0,primary_plot_idx].get_xaxis().set_tick_params(which='minor', width=0) 
    axes[0,primary_plot_idx].set_ylabel(y_axis_title)
    axes[0,primary_plot_idx].set_xlim(min(x_vals),max(x_vals))
    axes[0,primary_plot_idx].set_ylim(0,100)
    if is_line_at_1:
        if not reproductive_numbers is None:
            axes[0,0].axvline(1, 0,100,color='black',linestyle=':')
        axes[0,primary_plot_idx].axvline(1, 0,100,color='black',linestyle=':')
        axes[1,0].axvline(1, 0,100,color='black',linestyle=':')
        axes[1,1].axvline(1, 0,100,color='black',linestyle=':')
    
    for i in range(num_ethnic_groups):
        # axes[1,i].set_title(f'Ethnic group {i+1}')
        for scen in scenarios_to_plot:
            axes[1,i].plot(x_vals,variance_array[(scen-1),(1+i),:], label=f"Scenario {scen}")
        axes[1,i].set_ylim([0,100])
        axes[1,i].set_xlim([np.min(x_vals),np.max(x_vals)])
        if is_xlog:
            axes[1,i].set_xscale('log')
            min_2_pwr = int(np.ceil(np.log(min(x_vals))/np.log(2)))
            max_2_pwr = int(np.floor(np.log(max(x_vals))/np.log(2)))
            # List of tick labels 
            tick_labels = []
            for pwr in range(min_2_pwr, max_2_pwr+1,1):
                # tick_labels.append(f'$2^{{{pwr}}}$') # used for power of two labeling (not used right now)
                # tick_labels.append(np.round(2**pwr,4))
                tick_labels.append(np.round(2.0**pwr,4))
            axes[1,i].get_xaxis().set_major_formatter(pyplottk.ScalarFormatter())
            axes[1,i].set_xticks(2.0**np.arange(min_2_pwr,max_2_pwr+1,1),labels=tick_labels)
            axes[1,i].get_xaxis().set_tick_params(which='minor', size=0)
            axes[1,i].get_xaxis().set_tick_params(which='minor', width=0) 
        axes[1,i].set_ylabel(y_axis_title)
    if is_save_fig:
        plt.savefig(f'images/{filename}_ethnic.png', dpi=300,bbox_inches='tight')
    plt.show()
    
    fig, axes = plt.subplots(1,num_age_groups,figsize=(6*num_age_groups, 4))
    for i in range(num_age_groups):
        axes[i].set_title(f'Age group {i+1}')
        for scen in scenarios_to_plot:
            axes[i].plot(x_vals,variance_array[(scen-1),(1+ num_ethnic_groups +i),:], label=f"Scenario {scen}")
        axes[i].legend()
        axes[i].set_ylim([0,100])
        axes[i].set_xlim([np.min(x_vals),np.max(x_vals)])
        if is_xlog:
            axes[i].set_xscale('log')
        axes[i].set_ylabel(y_axis_title)
        axes[i].set_xlabel(x_axis_title)
    if is_save_fig:
        plt.savefig(f'images/{filename}_age.png', dpi=300,bbox_inches='tight')
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
        axes[i].set_ylim([0,100])
        axes[i].set_xlim([np.min(x_vals),np.max(x_vals)])
        axes[i].set_ylabel(y_axis_title)
        axes[i].set_xlabel(x_axis_title)
    plt.tight_layout()
    if is_save_fig:
        plt.savefig(f'images/{filename}_age_ethnic.png', dpi=300,bbox_inches='tight')
    plt.show()
    

# Specify plot colours
plt.rc('axes', prop_cycle=cycler(color=['#12436D', '#28A197', '#9E1962', '#F46A25','#D21D1D']))

# Specify number of ethnic groups and age groups - note that due to merging of code changing these may produce some errors
num_ethnic_groups = 2
num_age_groups =5

# specifying some parameter values
epsilon = 0.3
gamma = 2/3
sigma = 1/3 # Rate of disease development
time = 300

# specify the number of matrices, is mostly irrelevant, again may cause issues if changed due to code merging
num_matrices = 10

# setting up some storage lists to handle the scenarios
C_storage_prop = []
C_storage_assort = []
C_storage_list = []
scenario_reproductive_numbers = np.zeros(num_matrices)

c = 0.3 # Age based assortativity


### List of actions to take 

is_save_figs = True

is_plot_matrices = False
is_plot_SEIR = False

is_run_relative_contact_rate_variance = False
is_plot_relative_contact_rate_variance = True

is_run_transmission_variance = False
is_plot_transmission_variance = False

is_run_epsilon_variance = False
is_plot_epsilon_variance = False

scenarios_to_plot = [1,2,3,4,5]

# Constructing the contact matrices
for k in range(num_matrices):
    N,F,a = mfmm.scenario_parameters(k)
    
    ### Check
    if not (2*k)//num_matrices:
        F=[1,1]
    
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
        axes[i].set_title(f'{"abcdefghijklmnop"[i]})', loc='left', fontsize=16)
        # Turn off ticks
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Contact rate')
    if is_save_figs:
        plt.savefig(f'images/heatplot_matrix_comparison_epsilon_{epsilon}.png', dpi=300,bbox_inches='tight')
    plt.show()




### PLOT PARAMETERS
colour_wheel = ['red','blue','green','purple','teal']


if is_plot_SEIR:
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
            plt.tight_layout()
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
    
    plt.xlabel("days")
    plt.ylabel("cumulative proportion of group infected")
    legend_with_extra('Ethnic group 1', 'Ethnic group 2')
    plt.title("Cumulative proportion of infectious people in the youngest \n age group (Different ethnic contact rates)")
    if is_save_figs:
        plt.tight_layout()
        plt.savefig('images/cum_infectious_youngest_age_group.png', dpi=300)
    

### Change in relative contact rate

if is_run_relative_contact_rate_variance:
    
    # Want this to be odd to have zero in linspace - not required
    resolution = 41
    F_var_attack_rates = np.zeros([num_matrices//2,1+num_age_groups+num_ethnic_groups+num_age_groups*num_ethnic_groups, resolution])
    reproductive_numbers = np.zeros([num_matrices//2,resolution])
    
    F_vals = 10**np.linspace(-1, 1, resolution)
    
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
            relative_contact_rate_scens.append(scen)
    
    variance_plot(F_vals, 100* F_var_attack_rates,scenarios_to_plot=relative_contact_rate_scens,
                  x_axis_title='Ratio of ethnic contact rates (F1/F2)',
                  y_axis_title='Attack rates (%)',
                  num_matrices=num_matrices, num_ethnic_groups=num_ethnic_groups,
                  num_age_groups=num_age_groups, is_save_fig = is_save_figs,
                  filename=f'relative_contact_rate_variation/relative_contact_rate_variation_eth_epsilon{epsilon}_age_epsilon_{c}',
                  reproductive_numbers=reproductive_numbers,is_xlog=True,is_line_at_1=True)
    


if is_run_transmission_variance:
    
    # number of different transmission values to simulate
    resolution = 301
    transmission_var_attack_rates = np.zeros([num_matrices,1+num_age_groups+num_ethnic_groups+num_age_groups*num_ethnic_groups, resolution])
    
    
    transmission_vals = np.linspace(0,3, resolution)
    reproductive_numbers = np.zeros([num_matrices,resolution])
    
    
    # iterate over all scenarios
    for l in range(num_matrices):
        
        N,F,a = mfmm.scenario_parameters(l)
        
        # Form the age social contact matrix
        Pij = np.zeros([num_age_groups,num_age_groups])
        for i in range(num_age_groups):
            for j in range(num_age_groups):
                Pij[i,j] = (1-c)*a[i]*a[j]/np.sum(np.sum(N,axis=1)*a)
                if i==j:
                    Pij[i,j]+=c*a[j]/np.sum(N[j,:])
        # # Scale to have basic reproductive number of 1
        # Pij = Pij / scenario_reproductive_numbers[l]
        Cij = np.zeros([num_age_groups,num_age_groups])
        for j in range(num_age_groups):
            Cij[:,j] = Pij[:,j] * np.sum(N[j,:])
        
        # Turn the age social contact matrix into the age-ethnicity matrix
        C = mfmm.return_C_matrix(epsilon,Cij,F, N)
        # check that the matrix satisfies our conditions
        mfmm.condition_checking_fixed(C, Cij, N,is_shorthand=True)
        
        # simulate SEIR model for each transmission probability
        for k in range(resolution):
            
            q = transmission_vals[k]
            
            # basic reproductive number
            reproductive_numbers[l,k] = mfmm.initial_reproduction_number(mfmm.flatten_to_two_dim(q*C), gamma = gamma)
            
            
            # Convert C to per capita
            beta_matrix = mfmm.flatten_to_two_dim(C) / (N.T).flatten()
            beta_matrix = beta_matrix * q
            
            # Run SEIR model
            solution = scipy.integrate.solve_ivp(mfmm.SEIR_model, [0,time],
                            np.concatenate(mfmm.initial_group_populations((N.T).flatten(),is_vacc=False,pop_vec_vacc=np.array([]),initial_exposed=0.0001)),
                            t_eval=np.arange(time+1), args = (beta_matrix, sigma, gamma))
            
            # Grab the results
            transmission_var_attack_rates[l,0,k] = np.sum(solution.y[50:,-1])/np.sum(N) # whole population
            transmission_var_attack_rates[l,1,k] = np.sum(solution.y[50:55,-1])/np.sum(N[:,0]) # demographic group 1
            transmission_var_attack_rates[l,2,k] = np.sum(solution.y[55:,-1])/np.sum(N[:,1]) # demographic group 2
            for i in range(num_age_groups):
                transmission_var_attack_rates[l,(3+i),k] = (solution.y[(50+i),-1]+solution.y[(55+i),-1])/np.sum(N[i,:])
            for i in range(num_age_groups*num_ethnic_groups):
                transmission_var_attack_rates[l,(8+i),k] = solution.y[(50+i),-1]/np.sum(N[i%num_age_groups,i//num_age_groups])
    
    np.save('generated_results/transmission_variance_attack_rate_ethnic_epsilon_{epsilon}_age_epsilon_{c}.npy', transmission_var_attack_rates)
    np.save('generated_results/transmission_variance_reprod_num_ethnic_epsilon_{epsilon}_age_epsilon_{c}.npy', reproductive_numbers)
    np.save('generated_results/transmission_variance_transmission_vals_ethnic_epsilon_{epsilon}_age_epsilon_{c}.npy', transmission_vals)
    print(np.shape(reproductive_numbers))

if is_plot_transmission_variance:
    transmission_var_attack_rates = np.load('generated_results/transmission_variance_attack_rate_ethnic_epsilon_{epsilon}_age_epsilon_{c}.npy')
    transmission_vals = np.load('generated_results/transmission_variance_transmission_vals_ethnic_epsilon_{epsilon}_age_epsilon_{c}.npy')
    transmission_reprods = np.load('generated_results/transmission_variance_reprod_num_ethnic_epsilon_{epsilon}_age_epsilon_{c}.npy')
    
    variance_plot(transmission_vals/3, 100*transmission_var_attack_rates,scenarios_to_plot=scenarios_to_plot,
                  x_axis_title='transmission probability', y_axis_title='Attack rates (%)',
                  num_matrices=num_matrices, num_ethnic_groups=num_ethnic_groups,
                  num_age_groups=num_age_groups, is_save_fig = is_save_figs,is_xlog=False,
                  reproductive_numbers=transmission_reprods,
                  filename = 'transmission_variation',is_horizontal_line_at_1=True)
    
    
    

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
    
    variance_plot(e_vals, 100*e_var_attack_rates,scenarios_to_plot=scenarios_to_plot,
                  x_axis_title='Ethnic epsilon value', y_axis_title='Attack rates (%)',
                  num_matrices=num_matrices, num_ethnic_groups=num_ethnic_groups,
                  num_age_groups=num_age_groups, is_save_fig = is_save_figs,
                  filename=f'epsilon_variation/epsilon_var_age_epsilon_{c}',
                  reproductive_numbers=reproductive_numbers,is_xlog=False)

    