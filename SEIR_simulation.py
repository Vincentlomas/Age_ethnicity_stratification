# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 12:48:26 2025

@author: Vincent Lomas

Numerical analysis of new contact matrix construction method
"""

import multi_factor_matrix_modules.modules as mfmm
import numpy as np
import matplotlib.pyplot as plt
import scipy

def legend_with_extra(solid_name,dashed_name, solid_line=None, dashed_line=None):
    # Add solid and dashed line to legend
    if solid_line is None:
        solid_line = plt.Line2D([0], [0], color='gray', linestyle='-', label=solid_name)
    if dashed_line is None:
        dashed_line = plt.Line2D([0], [0], color='gray', linestyle='--', label=dashed_name)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(solid_line)
    handles.append(dashed_line)
    labels.append(solid_name)
    labels.append(dashed_name)

    plt.legend(handles, labels)


### PLOT PARAMETERS
colour_wheel = ['red','blue','green']


### SIMULATION PARAMETERS
num_ethnic_groups = 2
num_age_groups =5

N_vec_age = np.round((-63000*np.linspace(0,1,num_age_groups)**2+44000*np.linspace(0,1,num_age_groups)+29800)/3).astype('int64')*3

c = 0.3
age_contact_rates = 1-2*((np.linspace(0,1,num_age_groups)-0.45)**2)
Pij = np.zeros([num_age_groups,num_age_groups])
for i in range(num_age_groups):
    for j in range(num_age_groups):
        Pij[i,j] = (1-c)*age_contact_rates[i]*age_contact_rates[j]/np.sum(age_contact_rates*N_vec_age)
        if i==j:
            Pij[i,j]+=c*age_contact_rates[j]/N_vec_age[j]



Cij = np.zeros([num_age_groups,num_age_groups])
for j in range(num_age_groups):
    Cij[:,j] = Pij[:,j] * N_vec_age[j]


N0 = np.zeros([num_age_groups,num_ethnic_groups], dtype='int32')
N0[:,0] = N_vec_age//3
N0[:,1] = 2*N_vec_age//3


age_structure_diff = np.round(0.3*N0[-1,0]*np.linspace(-1,1,num_age_groups)).astype('int32')

N1 = np.zeros([num_age_groups,num_ethnic_groups],dtype='int32')
N1[:,0] = N0[:,0]-age_structure_diff
N1[:,1] = N0[:,1]+age_structure_diff

C_storage_list = []

num_matrices = 6
Fs = [np.array([1,1]),np.array([2,3]),np.array([3,2])]
for k in range(num_matrices):
    F = Fs[k//2]
    
    if k%(num_matrices//len(Fs))==0:
        N = N0
    elif k%(num_matrices//len(Fs))==1:
        N = N1
    
    C_constructed = mfmm.return_C_matrix(0.3,Cij,F, N)
    
    C_storage_list.append(C_constructed)


solutions =[]

### Run SEIR models
for k, C_matrix in enumerate(C_storage_list):
    # Grab initial populations - Note that N becomes a 1D array that has all ages within an ethnicity before moving onto the next ethnicity
    # e.g. using N_ia: N=[N_00,N_10,N_20,N_01,N_11,N_21]
    if k%(num_matrices//len(Fs))==0:
        N = N0.T.flatten()
    elif k%(num_matrices//len(Fs))==1:
        N = N1.T.flatten()
        
    # Convert C to per capita
    beta_matrix = mfmm.flatten_to_two_dim(C_matrix) / N
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
SEIR_0_age = np.concatenate(mfmm.initial_group_populations(N_vec_age.flatten(),is_vacc=False,pop_vec_vacc=np.array([]),initial_exposed=0.0001))
solution = scipy.integrate.solve_ivp(mfmm.SEIR_model, [0,time], SEIR_0_age, t_eval=np.arange(time+1), args = (Cij/(N_vec_age.flatten()), sigma, gamma))

for p in range(3):
    S0,S1 = [(0,time),(35,45),(time-10,time+1)][p]
    plt.figure()
    for k in range(0,num_matrices,1):
            
        if k%2:
            plt.plot(np.arange(S0,S1),np.sum(solutions[k][50:,:],axis=0)[S0:S1],
                     label=['Same contact rates', '2/3 contact rates', '3/2 contact rates'][k//2],
                     color=colour_wheel[k//2])
        else:
            plt.plot(np.arange(S0,S1),np.sum(solutions[k][50:,:],axis=0)[S0:S1],linestyle=(0, (3, 6)), linewidth=3,color=colour_wheel[k//2])
    
    # Plot solution with no ethnic consideration
    plt.plot(np.arange(S0,S1),np.sum(solution.y[25:,:],axis=0)[S0:S1],linestyle=':',label = 'No eth soln', linewidth = 3)
    
    legend_with_extra('Same age structure','Different age structure',dashed_line=plt.Line2D([0], [0], color='gray', linestyle=(0, (3, 6)), linewidth = 3, label='Different age structure'))
    plt.title("Cumulative number of infectious people")

plt.figure()
for k in range(6):
    if k%(num_matrices//len(Fs))==0:
        plt.plot(solutions[k][50,:]/N0[0,0],label=['Same contact rates', '2/3 contact rates', '3/2 contact rates'][k//2],color=colour_wheel[k//2])
        plt.plot(solutions[k][55,:]/N0[0,1], linestyle='--',color=colour_wheel[k//2])

legend_with_extra('Ethnic group 1', 'Ethnic group 2')
plt.title("Cumulative proportion of infectious people in the youngest age group\n (Same age structure)")

plt.figure()
for k in range(6):
    if k%(num_matrices//len(Fs))==1:
        plt.plot(solutions[k][50,:]/N1[0,0],label=['Same contact rates', '2/3 contact rates', '3/2 contact rates'][k//2],color=colour_wheel[k//2])
        plt.plot(solutions[k][55,:]/N1[0,1], linestyle='--',color=colour_wheel[k//2])

legend_with_extra('Ethnic group 1', 'Ethnic group 2')
plt.title("Cumulative proportion of infectious people in the youngest age group\n (Different age structure)")

plt.figure()
for k in range(6):
    if k%(num_matrices//len(Fs))==0:
        plt.plot(np.sum(solutions[k][50:55,:],axis=0),label=['Same contact rates', '2/3 contact rates', '3/2 contact rates'][k//2],color=colour_wheel[k//2])
    else:
        plt.plot(np.sum(solutions[k][50:55,:],axis=0),linestyle='--',color=colour_wheel[k//2])
legend_with_extra('Same age structure', 'Different age structure')
plt.title("Cumulative number of infectious people in ethnic group a")

C_storage_list_epsilon = []
epsilons = np.array([0,0.5,0.8,0.95,1])
num_epsilon = len(epsilons)
for k in range(num_epsilon):
    F = np.array([2,3])
    N = N1
    
    epsilon = epsilons[k]
    
    C_constructed = mfmm.return_C_matrix(epsilon,Cij,F, N)
    
    C_storage_list_epsilon.append(C_constructed)


solutions_epsilon = []

### Run SEIR models
for k, C_matrix in enumerate(C_storage_list_epsilon):
    # Grab initial populations - Note that N becomes a 1D array that has all ages within an ethnicity before moving onto the next ethnicity
    # e.g. using N_ia: N=[N_00,N_10,N_20,N_01,N_11,N_21]
    N = N1.T.flatten()
        
    # Convert C to per capita
    beta_matrix = mfmm.flatten_to_two_dim(C_matrix) / N
    # Check if symmetric
    if np.all(abs(beta_matrix - beta_matrix.T) < 10**-9):
        
        S,Sv,E,I,R, In = mfmm.initial_group_populations(N,is_vacc=False,pop_vec_vacc=np.array([]),initial_exposed=0.0001)
        
        
        sigma = 1/3 # Rate of disease development
        gamma = 0.25 # Recovery rate
        time = 100
        
        solution = scipy.integrate.solve_ivp(mfmm.SEIR_model, [0,time], np.concatenate([S,Sv,E,I,R,In]), t_eval=np.arange(time+1), args = (beta_matrix, sigma, gamma))
        
        solutions_epsilon.append(solution.y)
        
    else:
        print(k)
        print('Beta is not symmetric')

plt.figure()
for k in range(num_epsilon):
    plt.plot(np.sum(solutions_epsilon[k][50:,:],axis=0),label=f'epsilon = {np.round(epsilons[k],decimals=2)}')
plt.legend()
plt.title("Cumulative number of infectious people\n Different age structure w/ 2/3 contact rates")
    

plt.figure()
for k in range(num_epsilon):
    plt.plot(solutions_epsilon[k][50,:]+solutions_epsilon[k][55,:],label=f'epsilon = {np.round(epsilons[k],decimals=2)}')
plt.legend()
plt.title("Cumulative number of infectious people in the youngest age group\n Different age structure w/ 2/3 contact rates")

plt.figure()
for k in range(num_epsilon):
    plt.plot(np.sum(solutions_epsilon[k][50:55,:],axis=0),label=f'epsilon = {np.round(epsilons[k],decimals=2)}')
plt.legend()
plt.title("Cumulative number of infectious people in ethnic group a\n Different age structure w/ 2/3 contact rates")

plt.figure()
for k in range(num_epsilon):
    plt.plot(np.sum(solutions_epsilon[k][55:,:],axis=0),label=f'epsilon = {np.round(epsilons[k],decimals=2)}')
plt.legend()
plt.title("Cumulative number of infectious people in ethnic group b\n Different age structure w/ 2/3 contact rates")


### When doing real world analysis - 

# Focus on measles differences

# Break down figures for presentation

# Measles - vaccination data avalible at individual level

# Plan ahead on what to focus - confirmation/paper plans
# What is the research question, data, and methodology (abstract-like)
# High level overview of project plan for papers
# how is this paper fitting into the wider thesis -essentually a methodlogical section for the next chapter

# Good justification of work
### investigation of Florida removing mandate on meseales vaccines
# impact of policy decision in the US - contentious issue, good for paper - projection of impact, whole point of my thesis
# assumption about how many people would not get the vaccines
# Maybe have data about first year
# Rare that you get data about policy reversion
# (4th? paper)
# Difference historically same as difference today

