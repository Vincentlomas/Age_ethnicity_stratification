# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:42:57 2026

@author: Vincent Lomas

Numerical analysis of new contact matrix construction method - analysing real world example
"""


import numpy as np
import scipy
import multi_factor_matrix_modules.modules as mfmm
import matplotlib.pyplot as plt
from cycler import cycler

plot_colors = ['#12436D', '#28A197', '#9E1962', '#F46A25','#D21D1D']
plt.rc('axes', prop_cycle=cycler(color=plot_colors))


# Get population structure
N = np.genfromtxt('age_ethnicity_population_structure.csv', delimiter=',', dtype='int32',skip_header=1,usecols=(1,2,3,4))

# Get age based contact matrix
C_age = np.genfromtxt('contact_matrix_NZ.csv', delimiter=',',skip_header=True).T

# Pick ethnic assortativity and relative contact rates from previous paper - https://doi.org/10.1080/29937574.2025.2591407
# These are rough estimates of the 50% CAR assorative rates from this paper
F = np.array([2,3,0.90,1])
# F = np.array([1,1,1,1])
epsilon = 0.2


# SEIR parameters
gamma = 1/4 # rate of recovery, I --> R
sigma = 1/3 # Rate of disease development, E --> I
time = 200 # time to run SEIOR model
q = 0.03 # susceptibility of population

C_age = q* C_age

# Construct age_eth matrix
C = mfmm.return_C_matrix(epsilon,C_age,F, N)


# Convert C_matrix to per capita
beta_matrix = mfmm.flatten_to_two_dim(C,axis=1) / (N).flatten() 


age_ethnic_solution = scipy.integrate.solve_ivp(mfmm.SEIR_model, [0,time],
                np.concatenate(mfmm.initial_group_populations(N,is_vacc=False,pop_vec_vacc=np.array([]),initial_exposed=0.0001)),
                t_eval=np.arange(time+1), args = (beta_matrix, sigma, gamma))


beta_matrix_age = C_age/np.sum(N,axis=1)

time_scale_factor = 1
age_solution = scipy.integrate.solve_ivp(mfmm.SEIR_model, [0,(time_scale_factor*time)],
                np.concatenate(mfmm.initial_group_populations(np.sum(N,axis=1),is_vacc=False,pop_vec_vacc=np.array([]),initial_exposed=0.0001)),
                t_eval=np.arange((time_scale_factor*time+1)), args = (beta_matrix_age, sigma, gamma))

# Construct eth matrix
C_eth = np.zeros([4,4])
for i in range(np.shape(N)[0]):
    for a in range(np.shape(N)[1]):
        C_eth[a,:] += np.sum(C[i,a,:,:]*N[i,a],axis=0) / np.sum(N[:,a])


beta_matrix_eth = C_eth/np.sum(N,axis=0)

eth_solution = scipy.integrate.solve_ivp(mfmm.SEIR_model, [0,(time_scale_factor*time)],
                np.concatenate(mfmm.initial_group_populations(np.sum(N,axis=0),is_vacc=False,pop_vec_vacc=np.array([]),initial_exposed=0.0001)),
                t_eval=np.arange((time_scale_factor*time+1)), args = (beta_matrix_eth, sigma, gamma))


# Construct age_eth matrix with every wthnic group having the same relative ethnic contact rate
C_same_rel_contact_rates = mfmm.return_C_matrix(epsilon,C_age,np.ones(4), N)
# Convert C_matrix to per capita
beta_matrix_same_rel_contact_rates = mfmm.flatten_to_two_dim(C_same_rel_contact_rates,axis=1) / (N).flatten() 

age_ethnic_solution_same_rel_contact_rates = scipy.integrate.solve_ivp(mfmm.SEIR_model, [0,time],
                np.concatenate(mfmm.initial_group_populations(N,is_vacc=False,pop_vec_vacc=np.array([]),initial_exposed=0.0001)),
                t_eval=np.arange(time+1), args = (beta_matrix_same_rel_contact_rates, sigma, gamma))


eth_labels=['Maori','Pacific','Asian',"European/Other"]
fig,axes = plt.subplots(1,2,figsize=(6.8*1.5,4.8*0.75),sharey=True)
N_groups = np.prod(np.shape(N))
for i in range(4):
    if True:
        axes[1].plot(np.sum(age_ethnic_solution.y[(N_groups*4+i):(N_groups*5):4,:],axis=0)/np.sum(N[:,i]), label=eth_labels[i], color=plot_colors[i])
        axes[1].plot(np.sum(age_ethnic_solution_same_rel_contact_rates.y[(N_groups*4+i):(N_groups*5):4,:],axis=0)/np.sum(N[:,i]), linestyle='dashed', color=plot_colors[i])
        axes[0].plot(eth_solution.y[(4*4+i),:]/np.sum(N[:,i]), label=eth_labels[i], color=plot_colors[i])
axes[0].set_ylim(0,1)
axes[0].set_xlim(0,time_scale_factor*time)
axes[1].set_xlim(0,time_scale_factor*time)
axes[0].set_title('a)',loc='left')
axes[1].set_title('b)',loc='left')
axes[0].set_ylabel('Proportion of population')
axes[0].legend(loc='upper left',title='age groups')
fig.text(0.513, 0, 'time (days)', ha='center')
plt.xlabel('time (days)')
plt.tight_layout()
plt.savefig('images/ethnic_recovered_population_demographic_expansion.png', dpi=300, bbox_inches='tight')


fig,axes = plt.subplots(1,2,figsize=(6.8*1.5,4.8*0.75),sharey=True)
plt.subplots_adjust(wspace=0.08)
for i in range(10):
    if i <9:
        # note that the i*8 is 8 as that is double the number of eth groups and we are combining 2 age groups in this plot
        line_label = f"{10*i}-{10*i+9}"
        thing_to_plot = np.sum(age_ethnic_solution.y[(N_groups*4+i*8):(N_groups*4+(i+1)*8),:],axis=0)/np.sum(N[(i*2):(i*2+2),:])
    else:
        line_label = "90+"
        thing_to_plot = np.sum(age_ethnic_solution.y[(N_groups*4+i*8):(N_groups*4+i*8+4),:],axis=0)/np.sum(N[i*2,:])
    if i < 5:
        line_linestyle = "solid"
    else:
        line_linestyle = 'dashed'
    
    axes[1].plot(thing_to_plot, label=line_label, linestyle=line_linestyle)

axes[1].set_title('b)',loc='left')
axes[1].set_ylim(0,1)
axes[1].set_xlim(0,time)
for i in range(10):
    if i <9:
        line_label = f"{10*i}-{10*i+9}"
        thing_to_plot = np.sum(age_solution.y[(N_groups+i*2):(N_groups+i*2+2),:],axis=0)/np.sum(N[(i*2):(i*2+2),:])
    else:
        line_label = "90+"
        thing_to_plot = (age_solution.y[(N_groups+i*2),:])/np.sum(N[(i*2),:])
    if i < 5:
        line_linestyle = "solid"
    else:
        line_linestyle = 'dashed'

    axes[0].plot(thing_to_plot, label=line_label, linestyle=line_linestyle)
axes[0].set_title('a)',loc='left')
axes[0].set_ylim(0,1)
axes[0].set_xlim(0,time_scale_factor*time)
axes[0].set_ylabel('Proportion of population')
axes[0].legend(loc='upper left',title='age groups')
fig.text(0.513, 0, 'time (days)', ha='center')
plt.tight_layout()
plt.savefig('images/age_recovered_population_demographic_expansion.png', dpi=300, bbox_inches='tight')

bar_labels = []
age_bars=np.zeros(20)
for i in range(19):
    age_bars[i]=(np.sum(age_ethnic_solution.y[(N_groups*4+i*4):(N_groups*4+i*4+4),-1],axis=0)-age_solution.y[(19*4+i),-1])/np.sum(N[i,:])
    bar_labels.append(f'{5*i}')

age_bars[19]=(np.sum(age_ethnic_solution.y[(N_groups*4):(N_groups*5),-1],axis=0)-np.sum(age_solution.y[(19*4):(19*5),-1]))/np.sum(N)
bar_labels.append('all')
    
plt.figure()
plt.bar(bar_labels,age_bars*100)
plt.ylabel('Difference in attack rate (%)')
plt.xlabel('Age group')
plt.xlim(-0.5,19.5)
plt.savefig('images/Percentage_diff_of_pop_in_attack_rate.png',dpi=300)


plt.figure()
plt.imshow(mfmm.flatten_to_two_dim(C),vmin=np.min(C), vmax=np.max(C),cmap='viridis', aspect='auto')
plt.colorbar()
