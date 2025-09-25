# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 11:05:40 2025

@author: Vincent Lomas

Synthetic dataset

Check supp - https://www.science.org/doi/full/10.1126/sciadv.adk4606
"""

import numpy as np



def condition_checking_fixed(Ciajb,Cij, Nia, eth_rates = None, tol = 10**-8):
    '''Takes as input:
        Ciajb: An nxmxnxm matrix that is the number of people in age group j 
            and ethnic group b an individual from age group i and ethnic group
            a contacts each day
        Cij: an nxn age contact matrix that is the number of people in age
            group j an individual from age group i contacts each day
        Nia: an nxm matrix containing the populations of each age-ethnic
            group
        eth_rates: an optional m length 1D array of relative contact rates
            rates between ethnicities
        tol: the tolerated different in matrix elements before a warning is
            raised, note that this is required due to some elements being floats
            
            
        This function checks that the Ciajb matrix adheres to a few conditions:
            summation condition - summing over ethnicities recovers Cij
            symmetry condition - Ciajb[i,a,j,b]*N_matrix[i,a] == Ciajb[j,b,i,a]*N_matrix[j,b]
            row summation condition - let Cia be Ciajb summed over j and b,
                then Cia[i,a0]/Cia[i,a1]==ethnic_contact_rates[a0]/ethnic_contact_rates[a1]
                for all values of i
            
        '''
    Ni = np.array([np.sum(Nia,axis=1)]).T # Extract age populations into a column vector
    
    num_i, num_a = np.shape(Ciajb)[0:2]
    
    # Find the raw contact matrix from the contact matrix
    Riajb = np.zeros(np.shape(Ciajb))
    for i in range(num_i):
        for a in range(num_a):
            Riajb[i,a,:,:] = Ciajb[i,a,:,:]*Nia[i,a]
    
    # Aggregate over ethnicity
    Rij_est = np.sum(np.sum(Riajb,axis=3),axis=1)
    Rij = Cij*Ni
    
    
    # Check that total number of raw contacts match
    if abs(np.sum(Riajb) - np.sum(Rij)) < tol:
        print('Total number of contacts conserved')
    else:
        print('WARNING: Total number of contacts not conserved')
    print()
    
    # Check that swapping (i,a) and (j,b) results in the same values
    symmetry_conserved = True
    for i in range(num_i):
        for a in range(num_a):
            if symmetry_conserved:
                if not np.all(abs(Riajb[i,a,:,:] - Riajb[:,:,i,a]) < tol):
                    symmetry_conserved = False
    if symmetry_conserved:
        print('Symmetry is conserved')
    else:
        print('WARNING: R_iajb symmetry not conserved')
    print()
    
    # Check elements of the raw contact matrix are same (stronger than total number of raw contacts match)
    if np.all(abs(Rij_est - Rij) < tol):
        print('C_iajb aggregates to Cij')
    else:
        print('WARNING: C_iajb does not aggregate to Cij')
    print()
    
    
    if eth_rates is None:
        eth_rates = np.sum(np.sum(Ciajb[0,:,:,:],axis=-1),axis=-1)
    
    eth_rates_conserved = True
    for i in range(num_i):
        new_eth_rate0 = np.sum(np.sum(Ciajb[i,:,:,:],axis=-1),axis=-1)
        new_eth_rate = new_eth_rate0 * np.sum(eth_rates) / np.sum(new_eth_rate0)
        #print(new_eth_rate/eth_rates)
        if eth_rates_conserved:
            if not np.all(abs(new_eth_rate - eth_rates) < tol):
                eth_rates_conserved = False
    if eth_rates_conserved:
        print('intra age group eth rates conserved')
    else:
        print('WARNING: intra age group eth rates NOT conserved')


def flatten_to_two_dim(matrix,axis=0):
    '''Takes as input:
        matrix - a 4 index array A_ijkl of shape nxmxnxm
        axis - default value of 0, denotes the axis that is iterated over first
            when returning the new array (0 means axis0 iterates over fully
                                          before taking one step in axis1 index
                                          then axis0 is iterated over again)
        Output:
            a 2 index array'''
    a0,b0,c0,d0 = np.shape(matrix)
    
    if a0 == c0 and b0 == d0:
        new_matrix = np.zeros([a0*b0,c0*d0])
        
        for i in range(a0):
            for a in range(b0):
                for j in range(c0):
                    for b in range(d0):
                        if axis==0:
                            new_matrix[(i+a*a0),(j+b*c0)] = matrix[i,a,j,b]
                        else:
                            new_matrix[(i*b0+a),(j*d0+b)] = matrix[i,a,j,b]
        return new_matrix
    else:
        print('Shape of matrix is not symmetric')



def initial_group_populations(pop_vec,is_vacc,pop_vec_vacc=np.array([]),initial_exposed=0.0001,initial_infected=0,initial_recovered=0):
    '''takes as input:
        pop_vec: a vector of total populations of different groups
        pop_vec_vacc: a vector of vaccinated populations of different groups
        initial_exposed: the proportion of people initially exposed to the disease
        initial_infected: the proportion of people initially infected with the disease
        initial_recovered: the proportion of people initially recovered from the disease
    Returns initial SEIR group populations as a 5 length tuple of the form
    (susceptible, susceptible_vaccinated, exposed, infected, reovered)'''
    # Convert any 2D arrays to 1D
    pop_vec = pop_vec.flatten()
    pop_vec_vacc = pop_vec_vacc.flatten()
    
    initial_s = 1 - initial_exposed - initial_infected - initial_recovered
    
    if is_vacc:
        Sv = pop_vec_vacc
        S = initial_s*pop_vec - Sv
        E = initial_exposed*pop_vec
        I = initial_infected*pop_vec
        R = initial_recovered*pop_vec
    else:
        S = initial_s*pop_vec
        Sv = np.zeros(len(S))
        E = initial_exposed*pop_vec
        I = initial_infected*pop_vec
        R = initial_recovered*pop_vec
    In = np.zeros(len(S))
    return S,Sv,E,I,R, In



def initial_reproduction_number(C = None, P = None,N_vec_vacc=None,N_vec = None, gamma=-1):
    
    if P is None and C is None:
        raise Exception("Provide either a per capita contact matrix (P) or a contact matrix (C)")
    elif C is None and N_vec is None:
        raise Exception("If providing a per capita matrix, a population vector is needed")
    elif gamma == -1:
        raise Exception("Provide a value for gamma")
    
    if C is None:
        C = np.zeros(np.shape(P))
        for j in range(len(N_vec)):
            C[:,j] = P[:,j] * N_vec[j]
    
    beta_pop_matrix = C/gamma
            
        
    # F_Vinv = np.zeros([8,8])
    # F_Vinv[0:4,0:4] = beta_pop_matrix
    # F_Vinv[0:4,4:] = beta_pop_matrix
    
    eigs = np.linalg.eigvals(beta_pop_matrix)
    
    return max(abs(eigs))
    


def population_examples(k):
    
    if k==4:
        print('population_examples: changed k call from 4 to 3')
        k=3
    
    if k==0:
        N = 30000*np.ones([5,2], dtype='int64')
    elif k==1:
        N = np.array([[20000,40000],
                      [25000,35000],
                      [30000,30000],
                      [35000,25000],
                      [40000,20000]], dtype='int64')
    elif k==2:
        N = np.array([[54000,6000],
                      [54000,6000],
                      [54000,6000],
                      [54000,6000],
                      [54000,6000]], dtype='int64')
    elif k ==3:
        N = np.array([[36000,8000],
                      [45000,7000],
                      [54000,6000],
                      [63000,5000],
                      [72000,4000]], dtype='int64')
    return N



def return_C_matrix(epsilon,Cij,ethnic_contact_rates, N_matrix):
    '''Takes as input:
        Cij: an nxn age contact matrix that is the number of people in age
            group j an individual from age group i contacts each day
        ethnic_contact_rates: an m length 1D array of relative contact rates
            rates between ethnicities
        N_matrix: an nxm matrix containing the populations of each age-ethnic
            group
        epsilon: The percent assortative the output matrix is
    
    Outputs:
        Ciajb: An nxmxnxm matrix that is the number of people in age group j 
            and ethnic group b an individual from age group i and ethnic group
            a contacts each day. This matrix is constructed to have a linear
            combination of purely proportionate mixing and assortative mixing
            depending on the value of epsilon
            
        This matrix adheres to a few conditions:
            summation condition - summing over ethnicities recovers Cij
            symmetry condition - Ciajb[i,a,j,b]*N_matrix[i,a] == Ciajb[j,b,i,a]*N_matrix[j,b]
            row summation condition - let Cia be Ciajb summed over j and b,
                then Cia[i,a0]/Cia[i,a1]==ethnic_contact_rates[a0]/ethnic_contact_rates[a1]
                for all values of i
        '''
    return (1-epsilon)*return_C_matrix_proportionate(Cij,ethnic_contact_rates, N_matrix) + epsilon*return_C_matrix_assortative(Cij,ethnic_contact_rates, N_matrix)


        
def return_C_matrix_assortative(Cij,ethnic_contact_rates, N_matrix):
    '''Takes as input:
        Cij: an nxn age contact matrix that is the number of people in age
            group j an individual from age group i contacts each day
        ethnic_contact_rates: an m length 1D array of relative contact rates
            rates between ethnicities
        N_matrix: an nxm matrix containing the populations of each age-ethnic
            group
    
    Outputs:
        Ciajb: An nxmxnxm matrix that is the number of people in age group j 
            and ethnic group b an individual from age group i and ethnic group
            a contacts each day. This matrix is constructed to have purely 
            assortative mixing only allowing interactions within ethnicities
            
        This matrix adheres to a few conditions:
            summation condition - summing over ethnicities recovers Cij
            symmetry condition - Ciajb[i,a,j,b]*N_matrix[i,a] == Ciajb[j,b,i,a]*N_matrix[j,b]
            row summation condition - let Cia be Ciajb summed over j and b,
                then Cia[i,a0]/Cia[i,a1]==ethnic_contact_rates[a0]/ethnic_contact_rates[a1]
                for all values of i
        '''
    F = ethnic_contact_rates
    num_age_groups = np.shape(N_matrix)[0]
    num_ethnic_groups = np.shape(N_matrix)[1]
    
    # Increase int to 64 if not already as otherwise calculations will overflow
    N_matrix = N_matrix.astype('int64')
    
    # Check age contact matrix is square and matches dimention of age dimention of population matrix
    if (not np.shape(Cij)[0] == num_age_groups and np.shape(Cij)[1] == num_age_groups):
        if not np.shape(Cij)[0] == np.shape(Cij)[1]:
            print('Age matrix not square')
        else:
            print('Number of age dimention in age contact matrix and N_matrix not same')
        
        return None
    
    # Check ethnic dimention is equal for population matrix and relative ethnic contact rates
    if not np.shape(N_matrix)[1] == np.shape(ethnic_contact_rates)[0]:
        print('Number of ethnic dimention in relative ethnic contact rates and N_matrix not same')
        return None
    
    # Check that age contact matrix and population results in a symmetric raw contact matrix
    Rij = np.zeros([num_age_groups,num_age_groups])
    for i in range(num_age_groups):
        Rij[i,:] = Cij[i,:] * np.sum(N_matrix[i,:])
    if np.sum(abs(Rij - Rij.T)) > 10**-8:
        print('Age contact matrix and population matrix do not result in a symmetric raw contact matrix (to a tolerance of 10**-8)')
        return None
    
    C_assortative = np.zeros([num_age_groups,num_ethnic_groups,num_age_groups,num_ethnic_groups],dtype = 'float64')

    for i in range(num_age_groups):
        for a in range(num_ethnic_groups):
            for j in range(num_age_groups):
                for b in range(num_ethnic_groups):
                    if a==b:
                        if i==j:
                            first_term = F[a]*Rij[i,j]/np.sum(F*N_matrix[i,:])
                            diff_term = 0
                            for k in range(num_age_groups):
                                if not i == k:
                                    diff_term += F[a] * Rij[i,k]/(2 * N_matrix[i,a]) * (N_matrix[i,a]/np.sum(N_matrix[i,:]*F)-N_matrix[k,a]/np.sum(N_matrix[k,:]*F))
                            C_assortative[i,a,j,b] = first_term + diff_term
                        else:
                            C_assortative[i,a,j,b] = F[a] * Rij[i,j] * (N_matrix[i,a]*np.sum(F*N_matrix[j,:])+N_matrix[j,a]*np.sum(F*N_matrix[i,:])) / (2 * N_matrix[i,a] * np.sum(N_matrix[i,:]*F) * np.sum(N_matrix[j,:]*F))
    return C_assortative



def return_C_matrix_proportionate(Cij,ethnic_contact_rates, N_matrix):
    '''Takes as input:
        Cij: an nxn age contact matrix that is the number of people in age
            group j an individual from age group i contacts each day
        ethnic_contact_rates: an m length 1D array of relative contact rates
            rates between ethnicities
        N_matrix: an nxm matrix containing the populations of each age-ethnic
            group
    
    Outputs:
        Ciajb: An nxmxnxm matrix that is the number of people in age group j 
            and ethnic group b an individual from age group i and ethnic group
            a contacts each day. This matrix is constructed to have purely 
            proportionate mixing depending only on group size and relative 
            contact rate
            
        This matrix adheres to a few conditions:
            summation condition - summing over ethnicities recovers Cij
            symmetry condition - Ciajb[i,a,j,b]*N_matrix[i,a] == Ciajb[j,b,i,a]*N_matrix[j,b]
            row summation condition - let Cia be Ciajb summed over j and b,
                then Cia[i,a0]/Cia[i,a1]==ethnic_contact_rates[a0]/ethnic_contact_rates[a1]
                for all values of i
        '''
    F = ethnic_contact_rates
    num_age_groups = np.shape(N_matrix)[0]
    num_ethnic_groups = np.shape(N_matrix)[1]
    
    # Increase int to 64 if not already as otherwise calculations will overflow
    N_matrix = N_matrix.astype('int64')
    
    # Check age contact matrix is square and matches dimention of age dimention of population matrix
    if (not np.shape(Cij)[0] == num_age_groups and np.shape(Cij)[1] == num_age_groups):
        if not np.shape(Cij)[0] == np.shape(Cij)[1]:
            print('Age matrix not square')
        else:
            print('Number of age dimention in age contact matrix and N_matrix not same')
        
        return None
    
    # Check ethnic dimention is equal for population matrix and relative ethnic contact rates
    if not np.shape(N_matrix)[1] == np.shape(ethnic_contact_rates)[0]:
        print('Number of ethnic dimention in relative ethnic contact rates and N_matrix not same')
        return None
    
    # Check that age contact matrix and population results in a symmetric raw contact matrix
    Rij = np.zeros([num_age_groups,num_age_groups])
    for i in range(num_age_groups):
        Rij[i,:] = Cij[i,:] * np.sum(N_matrix[i,:])
    if np.sum(abs(Rij - Rij.T)) > 10**-8:
        print('Age contact matrix and population matrix do not result in a symmetric raw contact matrix (to a tolerance of 10**-8)')
        return None
    
    C_proportionate = np.zeros([num_age_groups,num_ethnic_groups,num_age_groups,num_ethnic_groups])

    for i in range(num_age_groups):
        for a in range(num_ethnic_groups):
            for j in range(num_age_groups):
                for b in range(num_ethnic_groups):
                    C_proportionate[i,a,j,b] = F[b]*F[a] * N_matrix[j,b] * Cij[i,j]*np.sum(N_matrix[i,:]) / (np.sum(N_matrix[i,:]*F)*np.sum(N_matrix[j,:]*F))
    
    return C_proportionate



def SEIR_model(t,SEIR, beta, sigma, gamma, imm_decay_rate=0, is_per_capita_contact_matrix=False):
    '''Takes as input a time (t), which is unused within the function, and
    a length 20 1D array the first 4 values coresspond to the susceptible 
    unvaccinated group for the 4 ethnic groups, next four to the suceptible
    vaccinated groups, and so on for E, I, and R as well
    The expected change in a timespan of 1 if conditions remain the same  for
    each value in the SEIR vector (i.e. the gradient) is calculated using the
    SEIR model
    The gradient of each value in the SEIR vector is returned resulting in a
    length 20 1D array'''
    
    S=SEIR[0:(len(SEIR)//6)] 
    Sv=SEIR[(len(SEIR)//6):(2*len(SEIR)//6)]
    E=SEIR[(2*len(SEIR)//6):(3*len(SEIR)//6)]
    I=SEIR[(3*len(SEIR)//6):(4*len(SEIR)//6)]
    R=SEIR[(4*len(SEIR)//6):(5*len(SEIR)//6)]
    In=SEIR[(5*len(SEIR)//6):]
    ### Equations
    S_to_E = (beta *S).T @ I
    Sv_to_S = Sv * imm_decay_rate
    Sv_to_E = (beta.T *Sv).T @ I * (1-0.3) # Vaccination effectiveness
    E_to_I = sigma * E
    I_to_R = gamma * I
    R_to_S = R * imm_decay_rate # Make people susceptible again
    
    # Calculate migration gradients
    dS =  Sv_to_S + R_to_S - S_to_E
    dSv=- Sv_to_S - Sv_to_E
    dE =   S_to_E + Sv_to_E - E_to_I
    dI =   E_to_I - I_to_R
    dR =   I_to_R - R_to_S
    dIn =  E_to_I
    return np.concatenate([dS,dSv,dE,dI,dR,dIn])



def scenario_parameters(k):
    '''Takes as input a value k and returns a population matrix,
    relative ethnic contact rates, and age contact rates'''
    
    if k // 5 == 0:
        relative_ethnic_contact_rates = np.ones(2)
    else:
        relative_ethnic_contact_rates = np.array([1,2])
    
    if k == 4 or k==9:
        age_contact_rates = np.linspace(1,0.5,5)
    else:
        age_contact_rates = np.ones(5)
    

    population_matrix = population_examples(k%5)

        
    return population_matrix, relative_ethnic_contact_rates, age_contact_rates
    
    