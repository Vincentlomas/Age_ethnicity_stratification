# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 12:48:26 2025

@author: Vincent Lomas

Numerical analysis of new contact matrix construction method

0) Push X more extreme, make entries negative
Done 1) keep in mind to try diff age groups having diff contact rates - linear, oldest 50% of youngest contact rate (tack onto last age structure)
2) Put together a func to calculate R0
3) Start working with cum prop of I in each (i,a) group
3.a) Start plotting this as a function of assortativity and relative contact rate
 - can multiply matri by a constant to scale R0
3.b) average attack rate for each ethnic group (and less relevantly age)
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

num_ethnic_groups = 2
num_age_groups =5


C_storage_list = []

num_matrices = 10
for k in range(num_matrices):
    N,F,a = mfmm.scenario_parameters(k)
    
    c = 0.3
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
    
    C_constructed = mfmm.return_C_matrix_assortative(Cij_k,F, N)
    #C_constructed = mfmm.return_C_matrix_proportionate(Cij_k,F, N)
    
    C_storage_list.append(C_constructed)


### Check that constraints on matrix are satisfied (only check last matrix constructed)
print()
print()
print('-'*90)
print('Construction attempt')
print('-'*90)
mfmm.condition_checking_fixed(C_constructed, Cij_k, N)



# Heat plot of matrices
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
plt.show()




### PLOT PARAMETERS
colour_wheel = ['red','blue','green','purple','teal']

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
        
        
        sigma = 1/3 # Rate of disease development
        gamma = 0.25 # Recovery rate
        time = 100
        
        solution = scipy.integrate.solve_ivp(mfmm.SEIR_model, [0,time], np.concatenate([S,Sv,E,I,R,In]), t_eval=np.arange(time+1), args = (beta_matrix, sigma, gamma))
        
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
    structure_labels.append(f'Scenario {i}')

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

