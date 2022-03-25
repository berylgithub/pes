# array op:
from operator import eq
import numpy as np, itertools
# file op:
import json
from os import walk
# data science:
from scipy.optimize import least_squares, minimize
from scipy.interpolate import CubicSpline
# dfbgn:
#import dfbgn
# lmfit:
#import lmfit
#from lmfit.printfuncs import report_fit
# timer:
import time
# visualization:
#import pandas as pd
import datetime
# utilities:
import warnings, sys
# related modules:
#import PES_models as pmodel
# parallelization:
import multiprocessing

'''====== Utilities ======'''
tsqrt = lambda x: np.sqrt(x) if x >= 0 else 0 #truncated sqrt
tlog = lambda x: np.log(x) if x > 0 else np.log(sys.float_info.min) #truncated log 

# rmse:
def RMSE(Y, Y_pred):
    N = Y.shape[0]
    return np.sqrt(np.sum((Y-Y_pred)**2)/N)

'''====== The fundamentals: 8.1 Bonding features ======'''
# bond strength:
def t_R_fun(R, R_up, R_low, e):
    R2 = R**2
    return ((R2 - R_low**2)/(R_up**2 - R2))**e

def s_bond_strength(R, R_up, R_low, t, t0):
    '''
    s_{ij} = s(R_{ij})
    t0 = t_R_fun(Rm, R_up, R_low, e)
    if R_m = R and R_low < R_m < R_up, then bond_strength_s = 0.5
    trainable parameters: (R_low, R_m, R_up)
    '''
    '''
    if R < R_low:
        return 1
    elif R_low <= R <= R_up:
        return t0/(t+t0)
    elif R_up < R:
        return 0
    '''
    '''
    low_idx = np.where(R < R_low)
    mid_idx = np.where((R_low <= R) & (R <= R_up))
    up_idx = np.where(R > R_up)
    R[low_idx] = 1. # low
    R[up_idx] = 0. # up
    t_mid = t[mid_idx]
    R[mid_idx] = t0/(t_mid+t0) # mid cond
    return R
    '''
    s = np.zeros(R.shape)
    low_idx = (R < R_low) 
    mid_idx = (R <= R_up) & ~low_idx #~low_idx := R_low <= R
    s[low_idx] = 1. # low
    t_mid = t[mid_idx]
    s[mid_idx] = t0/(t_mid+t0) # mid cond
    return s
    
    
s_bond_strength_vec = np.vectorize(s_bond_strength) # vectorized switch fun

# Tchebyshev polynomials functions:
s_prime_fun = lambda s: 2-4*s
def p_tchebyshev_pol(deg, s, s_prime):
    '''
    b_{ijd} = p_d(s_ij), where s_ij := s(R_ij)
    params:
        - deg = degree >= 1
        - s = s_bond_strength(R, R_up, R_low, t, t0)
        - 
    '''
    if deg == 1:
        return s
    elif deg == 2:
        return s*(1-s)
    elif deg == 3: # p_3 = s'p_2
        return s_prime*p_tchebyshev_pol(2, s, s_prime)
    elif deg > 3: # p_{d+1} = s'p_d - p_{d-1}
        return s_prime*p_tchebyshev_pol(deg-1, s, s_prime) - p_tchebyshev_pol(deg-2, s, s_prime)

# reference energy U:
def U_ref_energy_wrapper(V_fun, R):
    '''
    params:
        - V_fun := function for reference energy
        - R := distance (can be const or vector or matrix)
    '''
    return V_fun(R)

# coordination vector Y_d[i]:
@profile
def gen_bijd_mat(R_mat, max_deg, n_atom, R_up, R_m, R_low, e):
    '''
    b_ijd matrix generator \in R^{d x degree of freedom of atom i},
    b_ijd_mat, e.g, molecule with 3 atom := 
            [
                [b_{121}, b_{131}, b_{231},... ]
                [b_{122}, b_{132}, b_{232},... ]
                [b_{123}, ...] ...
            ] 
            \in R^{d x degree of freedom (DoF) of atom i}, np.array(dtype=float)
    params:
        - R_mat := [
                    [R_12, R_13, R_14, R_23, ...] := idx = 0
                    [R_12, R_13, ....] := idx = 1
                    [... ] ...
                ] \in R^{n x DoF of mol}
        - max_deg = maximum degree, d = 1,2,...,max_deg
        - n_atom = num atoms in the molecular system, e.g.: H3 -> n = 3
        - R_up for t and s(R), \in R
        - R_m for t and s(R), \in R
        - R_low for t and s(R), \in R
        - e for t and s(R), \in R
    '''
    # R_submat = R_mat[:, :n_atom-1] # all distances of atom i (len(R) x n_atom-1)
    t0 = t_R_fun(R_m, R_up, R_low, e) # const 
    t_mat = t_R_fun(R_mat, R_up, R_low, e) # matrix (len(R) x n_atom)
    s_mat = np.zeros(R_mat.shape) # matrix (len(R) x n_atom)
    '''
    # slow ver, need to check the each element one by one due to bool op:
    for i in range(R_mat.shape[0]):
        for j in range(R_mat.shape[1]):
            s_mat[i][j] = s_bond_strength(R_mat[i][j], R_up, R_low, t_mat[i][j], t0)
    #print(s_mat)
    '''
    # faster:
    #s_mat = s_bond_strength_vec(R_mat, R_up, R_low, t_mat, t0)
    s_mat = s_bond_strength(R_mat, R_up, R_low, t_mat, t0)
    #print(s_mat)
    s_prime_mat = s_prime_fun(s_mat)
    b_ijd_mat = np.array([p_tchebyshev_pol(deg, s_mat, s_prime_mat) for deg in range(1, max_deg+1)]) # tensor (max_deg, len(R), n_atom)
    return b_ijd_mat
        
def atom_indexer(num_atom):
    '''
    generates array of coordination indexes for Y[i] vector, which depends on num_atom, e.g.:
    num_atom = 3:  [[1,2],[1,3],[2,3]] - 1 (due to start at 0)
                     b_1j  b_2j b_3j
    num_atom = 4: [[1,2,3],[1,4,5],[2,4,6],[3,5,6]] - 1
                    b_1j     b_2j   b_3j     b_4j
    '''
    init_idx = list(range(num_atom-1, 0, -1))
    group_idx = []
    start = 0
    for idx in init_idx: # enumerate the indexes:
        list_index = list(range(start, idx+start))
        group_idx.append(list_index)
        start += idx
    coord_idx = [] # coordination indexes 
    for i in range(num_atom): # starts from 0 not 1!
        coord = []
        if i==0:
            coord = group_idx[i]
            coord_idx.append(coord)
        elif i==num_atom-1:
            enumerator = list(range(i-1, -1, -1))
            #print(i, enumerator)
            counter = 0
            for num in enumerator:
                coord.append(group_idx[counter][num])
                counter += 1
            coord_idx.append(coord)
        else:
            enumerator = list(range(i-1, -1, -1))
            #print(i, enumerator)
            counter = 0
            for num in enumerator:
                coord.append(group_idx[counter][num])
                counter += 1
            coord.extend(group_idx[counter])
            coord_idx.append(coord)
    return np.array(coord_idx)

       
def Y_coord_mat(b_ijd_mat, coord_idx_mat):
    '''
    returns Y := (Y[1], Y[2], ... Y[num_atom]), where Y[i] := (Y_1[i], Y_2[i], ... Y_max_d[i]),
    where Y_d[i] := \Sum_{j\neqi}b_{ijd} || shape = (num_atom, d, num_data)
    params:
        - b_ijd_mat := 
            [
                [b_{121}, b_{131}, ... ]
                [b_{122}, b_{132}, ... ]
                [b_{123}, ...] ...
            ] 
            \in R^{d x num_data x degree of freedom (DoF) of atom i}, np.array(dtype=float)
        - coord_idx_mat := list of indexes for Y[i] forall i, e.g.:
        num_atom = 4: [[1,2,3],[1,4,5],[2,4,6],[3,5,6]] - 1
                       b_1j     b_2j   b_3j     b_4j
    '''
    num_atom = coord_idx_mat.shape[0]; num_data = b_ijd_mat.shape[1]; max_d = b_ijd_mat.shape[0]
    Y = np.zeros((num_atom, max_d, num_data)) # a tensor (list of Y[i])
    # loop per atom i:
    for i, coord in enumerate(coord_idx_mat):
        Y[i] = np.sum(b_ijd_mat[:, :, coord], axis=2) # sum b_(i)jd
    return Y

# Orientation vector:
def delta_coord_matrix(X_mat):
    '''
    computes and arranges the (x_j - x_i) coodinates in such a way that the indexing is compatible with Y, r, and G,
    returns delta_mat, shape = (num_data, DoF, num_basis)
    params:
        - X_mat, list of coordinates matrix, shape = (num_data, num_atoms, num_elem)
    '''
    num_data = X_mat.shape[0]; num_atoms = X_mat.shape[1]; num_elem = X_mat.shape[2]
    n = num_atoms
    dof = int(n*(n-1)/2) # degree of freedom
    '''
    # naive way:
    delta_coord_mat = np.zeros((num_data, dof, num_elem))
    iter_atom = range(num_atoms)
    for d in range(num_data):
        # for each data, loop the atoms:
        dof_counter = 0
        for i in iter_atom:
            for j in iter_atom:
                if i < j:
                    delta_coord_mat[d][dof_counter] = X_mat[d][j] - X_mat[d][i]
                    dof_counter += 1
    '''
    # more efficient way, omit data index:
    delta_coord_mat = np.zeros((num_data, dof, 1, num_elem)) # add 1 index for numpy diff artefact
    iter_atom = range(num_atoms)
    dof_counter = 0
    for i in iter_atom:
        for j in iter_atom:
            if i < j:
                delta_coord_mat[:, dof_counter] = np.diff(X_mat[:, [i,j]], axis=1)
                dof_counter += 1
    delta_coord_mat = delta_coord_mat.reshape((num_data, dof, num_elem)) # reshape to the actual array indexes 
    
    return delta_coord_mat
        
def r_orient_vec(b_ijd_mat, delta_coord_matrix, coord_idx_mat):
    '''
    computes the r := (r[1], r[2],...), where r[i] := r_1[i], r_2[i],... ;
    where r_d[i] \in R^3 = sum(b_ij*delta_ij) for i!=j, i,j = 1,2,...;
    shape = (num_atom, d, num_data, num_elem)
    params:
        - b_ijd_mat, bond strength matrix, shape = (d, num_data, dof)
        - delta_coord_matrix, matrix containing (x_j - x_i), shape = (num_data, dof, num_elem)
    '''
    num_atom = coord_idx_mat.shape[0]; max_d = b_ijd_mat.shape[0]; num_data = delta_coord_matrix.shape[0]; 
    dof = delta_coord_matrix.shape[1]; num_elem = delta_coord_matrix.shape[2];
    # compute the matmul between b and delta:
    '''
    b_mult_delta = np.zeros((max_d, num_data, dof, num_elem)) # b*delta, shape = (d, num_data, dof, num_elem)
    # naive way:
    iter_d = range(max_d); iter_num_data = range(num_data); iter_dof = range(dof) #ranges for loop
    for d in iter_d:  # for each dimension:
        for dat in iter_num_data: # for each data:
            for deg in iter_dof: # for each degree of freedom:
                b_mult_delta[d][dat][deg] = b_ijd_mat[d][dat][deg]*delta_coord_matrix[dat][deg] # b_ijd * delta_ji
    '''
    # more efficient way, rearranging the last index vector to the front for the delta array and 2nd index for the result array:
    b_mult_delta = np.zeros((max_d, num_elem, num_data, dof)) # temporarily put the 3d cartesian vector index at 2nd pos
    delta_coord_matrix = np.transpose(delta_coord_matrix, (2,0,1)) # put the cartesian index to the front
    iter_d = range(max_d)
    for d in iter_d:
        b_mult_delta[d] = b_ijd_mat[d]*delta_coord_matrix #b_ijd * delta_ji
    b_mult_delta = np.transpose(b_mult_delta, (0,2,3,1)) # rearrange the indexes to (d, num_data, dof, num_elem)
    
    #print(b_mult_delta)
    #print(b_mult_delta.shape)
    # compute the sum, same way with Y:
    r = np.zeros((num_atom, max_d, num_data, num_elem))
    for i, coord in enumerate(coord_idx_mat):
        r[i] = np.sum(b_mult_delta[:, :, coord], axis = 2)
    return r



# Gram matrix:
def atom_indexer_G_mat(coord_idx_mat):
    '''
    # for non orientation vector version
    generate the indexes for G matrix formula, 
    depends heavily on the coord_idx_mat,
    e.g:
    [[0, 1],[0, 2],[0, 3],[1, 2],[1, 3],[2, 3]], -> i=1
    [[0, 4],[0, 5],[0, 6],[4, 5],[4, 6],[5, 6]], -> i=2
    ...
    \in R^(num_atom x ^nC_r x 2), n = coord_idx_mat.shape(1), r = 2
    
    params:
        - coord_idx_mat = matrix for indexing, shape = (DoF, 2)
    '''
    G_idx_mat = np.array([list(itertools.combinations(coord, 2)) for coord in coord_idx_mat])
    return G_idx_mat
     


def G_gram_mat(r_mat):
    '''
    makes use of orientation vectors !!
    returns Gramian matrix, G[i]_d1d2 = r_d1[i] \dot r_d2[i]... # Pg 60 of Ulrik
    shape = (num_atom, max_d, max_d, num_data)
    params:
        - r_mat = list of orientation vectors, shape = (num_atom, max_d, num_data, num_elem)
    '''
    num_atom = r_mat.shape[0]; max_d = r_mat.shape[1]; num_data = r_mat.shape[2]; num_elem = r_mat.shape[3]
    G_mat = np.zeros((num_atom, max_d, max_d, num_data))
    for i in range(num_atom): # each atom:
        for d1 in range(max_d): # each dim1:
            for d2 in range(max_d): # each dim2:
                '''
                print(i, d1, d2)
                print(r_mat[i][d1])
                print(r_mat[i][d2])
                print(np.sum(r_mat[i][d1]*r_mat[i][d2], axis=-1))
                '''
                #G_mat[i] = np.sum(r_mat[i][d1]*r_mat[i][d2], axis=-1) # r_d1[i]*r_d2[i] 
                #G_mat[i][d1][d2] = np.sum(r_mat[i][d1]*r_mat[i][d2], axis=-1) # r_d1[i]*r_d2[i] 
                # this is only n(n-1)/2 d operations instead of n^2:
                if d1 <= d2:
                    G_mat[i][d1][d2] = np.sum(r_mat[i][d1]*r_mat[i][d2], axis=-1)
                else: 
                    G_mat[i][d1][d2] = G_mat[i][d2][d1]
    return G_mat


# Neigborhood matrices Theta:

'''==== 8.2 Bonding potential with trainable reference pairpot ===='''

def V_ref_pairpot(R, C, R_h, R_C, R_0, g):
    '''
    returns energy, reference potential
    params:
        - R = distance, must be a scalar (due to comparison check)
        - C, R_h, R_C, R_0, g; scalar constants
    '''
    '''
    # slow:
    if R <= R_h:
        return np.inf
    elif R_h < R <= R_C:
        R2 = R**2
        return -C*(R_C**2 - R2)**g*( (R2 - R_0**2)/(R2 - R_h**2) )
    else:
        return 0
    '''
    '''
    # should be faster:
    low_idx = np.where(R <= R_h)
    mid_idx = np.where((R_h < R) & (R <= R_C))
    up_idx = np.where(R > R_C)
    R[up_idx] = 0. # up
    R[low_idx] = np.inf # low
    R_mid = R[mid_idx]
    R2 = R_mid**2
    R_mid = -C*(R_C**2 - R2)**g*( (R2 - R_0**2)/(R2 - R_h**2) ) # mid cond
    R[mid_idx] = R_mid
    return R
    '''
    Vref = np.zeros(R.shape)
    low_idx = (R <= R_h)
    mid_idx = (R <= R_C) & ~low_idx
    Vref[low_idx] = np.inf # low
    R_mid = R[mid_idx] # mid
    R2 = R_mid**2
    R_mid = -C*(R_C**2 - R2)**g*( (R2 - R_0**2)/(R2 - R_h**2) ) # mid cond
    Vref[mid_idx] = R_mid
    return Vref
    
    

V_ref_pairpot_vec = np.vectorize(V_ref_pairpot)
 
def U_ref_energy(R_mat, C, R_h, R_C, R_0, g, indexer):
    '''
    computes the reference energy matrix, U[i] = \sum V_ij, shape = (num_atom, num_data)
    parameters:
        - R_mat = distance matrix, shape = (num_data, dof)
        - C, R_h, R_C, R_0, g = scalar constants
        - indexer = indexer matrix, shape = (num_atom, num_atom-1)
    '''
    num_data = R_mat.shape[0]; dof = R_mat.shape[1]; num_atom = indexer.shape[0]
    Vref = np.zeros((num_data, dof))
    U = np.zeros((num_atom, num_data))
    '''
    # slow, transposed order:
    for i in range(num_data):
        for j in range(dof):
            Vref[i][j] = V_ref_pairpot(R_mat[i][j], C, R_h, R_C, R_0, g)
    #print(Vref)
    '''
    # faster:
    #Vref = V_ref_pairpot_vec(R_mat, C, R_h, R_C, R_0, g)
    Vref = V_ref_pairpot(R_mat, C, R_h, R_C, R_0, g)
    
    for i, coord in enumerate(indexer):
        U[i] = np.sum(Vref[:, coord], axis=1)
    return U
    
    
# basis functions phi:
# max degree = 5
def phi_fun(U, Y, G):
    '''
    constructs matrix containing basis functions phi, shape = (num_atom, num_basis, num_data), # Table 11 of Ulrik
    params:
        - U = reference pair potential energy, shape = (num_atom, num_data)
        - Y = matrix of coordination vectors, shape = (num_atom, max_d, num_data)
        - G = Gram matrix, shape = (num_atom, max_d, max_d, num_data)
    '''
    num_atom = U.shape[0]; num_data = U.shape[1]; num_basis = 59; #59 basis
    phi = np.zeros((num_atom, num_basis, num_data))
    #print(phi.shape)
    # degree 1:
    phi[:, 0] = U;
    phi[:, 1] = Y[:, 0];
    
    # degree 2:
    phi[:, 2] = U*Y[:, 0];
    phi[:, 3] = Y[:, 0]**2; phi[:, 4] = Y[:, 1]; phi[:, 5] = G[:, 0,0];
    
    # degree 3:
    phi[:, 6] = U*phi[:, 3]; phi[:, 7] = U*Y[:, 1]; phi[:, 8] = U*G[:, 0, 0];
    phi[:, 9] = phi[:, 3]*Y[:, 0]; phi[:, 10] = Y[:, 0]*Y[:, 1]; phi[:, 11] = Y[:, 2]; phi[:, 12] = G[:, 0, 0]*Y[:, 0]; phi[:, 13] = G[:, 0, 1];
    
    # degree 4 [14:30]:
    phi[:, 14] = U*phi[:, 9];
    phi[:, 15] = U*Y[:, 0]*Y[:, 1];
    phi[:, 16] = U*Y[:, 2];
    phi[:, 17] = U*G[:, 0,0]*Y[:, 0];
    phi[:, 18] = U*G[:, 0,1];
    phi[:, 19] = phi[:, 9]*Y[:, 0]; #Y_1^4
    phi[:, 20] = phi[:, 3]*Y[:, 1]; #Y_1^2Y_2
    phi[:, 21] = Y[:, 0]*Y[:, 2];
    phi[:, 22] = Y[:, 1]**2; #Y_2^2
    phi[:, 23] = Y[:, 3];
    phi[:, 24] = G[:, 0,0]*phi[:, 3];
    phi[:, 25] = G[:, 0,0]*Y[:, 1];
    phi[:, 26] = G[:, 0,0]**2;
    phi[:, 27] = G[:, 0,1]*Y[:, 0];
    phi[:, 28] = G[:, 0,2];
    phi[:, 29] = G[:, 1,1];
    
    # degree 5 [30:59]:
    phi[:, 30] = U*phi[:, 19];
    phi[:, 31] = U*phi[:, 3]*Y[:, 1];
    phi[:, 32] = U*phi[:, 3]*G[:, 0,0];
    phi[:, 33] = U*Y[:, 0]*Y[:, 2];
    phi[:, 34] = U*phi[:, 27];
    phi[:, 35] = U*phi[:, 22];
    phi[:, 36] = U*phi[:, 25];
    phi[:, 37] = U*Y[:, 3];
    phi[:, 38] = U*phi[:, 26];
    phi[:, 39] = U*G[:, 0,2];
    phi[:, 40] = U*G[:, 1,1];
    
    phi[:, 41] = phi[:, 19]*Y[:, 0]; # Y_1^5
    phi[:, 42] = phi[:, 9]*phi[:, 3];
    phi[:, 43] = phi[:, 9]*G[:, 0,0];
    phi[:, 44] = phi[:, 3]*Y[:, 2];
    phi[:, 45] = phi[:, 3]*G[:, 0,1];
    phi[:, 46] = Y[:, 0]*phi[:, 22];
    phi[:, 47] = phi[:, 10]*G[:, 0,0];
    phi[:, 48] = Y[:, 0]*Y[:, 3];
    phi[:, 49] = phi[:, 12]*G[:, 0,0];
    phi[:, 50] = Y[:, 0]*G[:, 0,2];
    
    phi[:, 51] = Y[:, 0]*G[:, 1,1]
    phi[:, 52] = Y[:, 1]*Y[:, 2]
    phi[:, 53] = Y[:, 1]*G[:, 0,1]
    phi[:, 54] = Y[:, 2]*G[:, 0,0]
    phi[:, 55] = Y[:, 4]
    phi[:, 56] = G[:, 0,0]*G[:, 0,1]
    phi[:, 57] = G[:, 0,3]
    phi[:, 58] = G[:, 1,2]

    return phi

# energy models:
def A_i_term(A1, A2, phi, i):
    '''
    returns partial energy model term A[i], shape = num_data. # page 61 of Ulrik
    params:
        - phi = matrix of basis functions, shape = (num_atom, num_basis, num_data)
        - A1 = vector of tuning parameters, shape = num_basis
        - A2 = vector of tuning parameters, shape = num_basis
        - i = atom index
    '''
    num_basis = phi.shape[1]
    numer = 0; denom = 0
    for k in range(num_basis):
        numer += A1[k]*phi[i,k] #scalar*vector
        denom += A2[k]*phi[i,k] #scalar*vector
    denom = denom**2 + 1
    A_i = numer/denom
    return A_i

def T0_i_term(T1, T2, phi, i):
    '''
    returns partial energy model term T0[i] \in R for B and C. shape = num_data
    params:
        - phi = matrix of basis functions, shape = (num_atom, num_basis, num_data)
        - T1 = vector of tuning parameters, shape = num_basis
        - T2 = vector of tuning parameters, shape = num_basis
        - i = atom index
    '''
    num_basis = phi.shape[1]
    numer = 0; denom = 0
    for k in range(num_basis):
        numer += T1[k]*phi[i,k] #scalar*vector
        denom += T2[k]*phi[i,k] #scalar*vector
    numer = numer**2
    denom = denom**2 + 1
    T_i = numer/denom
    return T_i

def epsilon_i_term(A_i, B_i, C_i):
    '''
    returns energy model \epsilon_i, shape = (num_data)
    params:
        - A_i = partial term, shape = num_data
        - B_i = partial term, shape = num_data
        - C_i = partial term, shape = num_data
    '''
    epsilon_i = A_i - np.sqrt(B_i + C_i)
    #print(A_i, B_i, C_i, epsilon_i)
    return epsilon_i
    
def epsilon_wrapper(phi, A1, A2, B1, B2, C1, C2):
    '''
    computes the \epsilon_0 in eq.(62) from all of the partial energy terms # pg 61
    params:
        - phi = matrix of basis functions, shape = (num_atom, num_basis, num_data)
        - A1, A2, B1, B2, C1, C2 = vectors of tuning parameters, shape = num_basis
    '''
    num_atom = phi.shape[0]
    
    A = np.array([A_i_term(A1, A2, phi, i) for i in range(num_atom)]) # A term, shape = (num_atom, num_data)
    B = np.array([T0_i_term(B1, B2, phi, i) for i in range(num_atom)]) # B term
    C = np.array([T0_i_term(C1, C2, phi, i) for i in range(num_atom)]) # C term
    
    
    epsilon = 0
    for i in range(num_atom):
        epsilon += epsilon_i_term(A[i], B[i], C[i]) # \sum \epsilon_0[i]
    return epsilon


'''====== Functional calculators ======'''
# num params = 59*6 + 7
# tuning params: C, R_h, R_low, R_0, R_m, R_up, R_C = scalar; A1, A2, B1, B2, C1, C2 = num_basis
def f_pot_bond(C, R_h, R_low, R_0, R_m, R_up, R_C, A1, A2, B1, B2, C1, C2, R, X, indexer, num_atom, max_deg, e, g=6):
    '''
    computes the energy, shape = (num_data)
    params: 
        - C, R_h, R_low, R_0, R_m, R_up, R_C; scalar || TUNING PARAMS
        - A1, A2, B1, B2, C1, C2; shape = num_basis || TUNING PARAMS
        - R: distance matrix, shape = (num_data, dof)
        - X: coordinate matrix, shape = (num_data, num_atom, 3)
        - indexer = matrix of atomic indexer, shape = (num_atom, num_atom-1)
        - num_atom: number of atoms in molecule, scalar
        - max_deg: maximum degree 
        - e: hyperparameter for bond strength, scalar > 2
        - g: hyperparameter for U, scalar (default=6)
    '''
    # compute U basis, contains tuning params (C, R_h, R_C, R_0):
    #print("C, R_h, R_C, R_0", C, R_h, R_C, R_0)
    U = U_ref_energy(R, C, R_h, R_C, R_0, g, indexer)
    #print('U', U.shape)
    #print(U)
    
    # compute Y basis, contains tuning params (R_up, R_m, R_low):
    b = gen_bijd_mat(R, max_deg, num_atom, R_up, R_m, R_low, e)
    Y = Y_coord_mat(b, indexer)
    #print("Y", Y.shape)
    #print(b)
    
    # compute G basis:
    delta = delta_coord_matrix(X)
    r = r_orient_vec(b, delta, indexer)
    G = G_gram_mat(r)
    #print('G', G.shape)
    
    # compute phi matrix:
    phi = phi_fun(U, Y, G)
    #print('phi', phi.shape)
    #print(phi)
    
    # compute the energy, contains tuning params (A1, A2, B1, B2, C1, C2):
    V = epsilon_wrapper(phi, A1, A2, B1, B2, C1, C2)
    return V

def f_pot_bond_wrapper(coeffs, *args):
    '''
    wrapper for f_pot_bond, unrolls tuning coeffs
    params:
        - coeffs: tuning coeffs in this order:
            - C, R_h, R_low, R_0, R_m, R_up, R_C; scalar
            - theta: matrix containing (A1, A2, B1, B2, C1, C2), shape = (6, num_basis)
        - *args: function arguments in this order:
            - num_basis: number of basis (column) in the phi matrix
            - R: distance matrix, shape = (num_data, dof)
            - X: coordinate matrix, shape = (num_data, num_atom, 3)
            - indexer = matrix of atomic indexer, shape = (num_atom, num_atom-1)
            - num_atom: number of atoms in molecule, scalar
            - max_deg: maximum degree 
            - e: hyperparameter for bond strength, scalar > 2
            - g: hyperparameter for U, scalar (default=6)
    '''
    # unroll args first:
    num_basis = args[0]
    R = args[1]
    X = args[2]
    indexer = args[3]
    num_atom = args[4]
    max_deg = args[5]
    e = args[6]
    g = args[7]
    
    # unroll coefficients:
    C = coeffs[0]; R_h = coeffs[1]; R_low = coeffs[2]; R_0 = coeffs[3]; R_m = coeffs[4]; R_up = coeffs[5]; R_C = coeffs[6];
    A1 = coeffs[7: num_basis+7]
    A2 = coeffs[num_basis+7: 2*num_basis+7]
    B1 = coeffs[2*num_basis+7: 3*num_basis+7]
    B2 = coeffs[3*num_basis+7: 4*num_basis+7]
    C1 = coeffs[4*num_basis+7: 5*num_basis+7]
    C2 = coeffs[5*num_basis+7: 6*num_basis+7]
    
    # compute energy from model:
    V_pred = f_pot_bond(C, R_h, R_low, R_0, R_m, R_up, R_C, A1, A2, B1, B2, C1, C2, 
                        R, X, indexer, num_atom, max_deg, e, g)
    
    return V_pred


def eq_68_converter(rho):
    '''
    converts the value of each argument by (according to eq 68 pg 64), shape = len(rho)
    '''
    pi = np.zeros(rho.shape[0])

    # compute the parameters change:
    pi[0] = tlog(rho[0])/20 # log(C)/20
    pi[1] = tlog(rho[1])/20 # log(Rh)/20
    pi[2] = tsqrt(rho[2]) # sqrt(R_low)
    pi[3] = tsqrt(rho[3] - rho[2]) # sqrt(R_0 - R_low)
    pi[4] = tsqrt(rho[4] - rho[3]) # sqrt(R_m - R_0)
    pi[5] = tsqrt(rho[5] - rho[4]) # sqrt(R_up - R_m)
    pi[6] = tsqrt(rho[6] - rho[5]) # sqrt(R_C - R_up)
    
    return pi

def eq_68_inverter(rho):
    '''
    inverts the value of each argument by (according to eq 68 pg 64), shape = len(rho)
    params:
        - rho, shape = len(rho)
    '''
    pi = np.zeros(rho.shape[0])

    # compute the inverse:
    pi[0] = np.exp(20*rho[0]) #C
    pi[1] = np.exp(20*rho[1]) #R_h
    pi[2] = np.square(rho[2]) #R_low
    pi[3] = rho[3]**2 + pi[2] #R_0
    pi[4] = rho[4]**2 + pi[3] #R_m
    pi[5] = rho[5]**2 + pi[4] #R_up
    pi[6] = rho[6]**2 + pi[5] #R_C
    return pi

def f_pot_bond_wrapper_trpp(coeffs, *args):
    '''
    "trainable reference pair potential (trpp)" version of wrapper for f_pot_bond from eq 68; unrolls tuning coeffs.
    params:
        - coeffs: tuning coeffs in this order:
            - C, R_h, R_low, R_0, R_m, R_up, R_C; scalar
            - theta: matrix containing (A1, A2, B1, B2, C1, C2), shape = (6, num_basis)
        - *args: function arguments in this order:
            - num_basis: number of basis (column) in the phi matrix
            - R: distance matrix, shape = (num_data, dof)
            - X: coordinate matrix, shape = (num_data, num_atom, 3)
            - indexer = matrix of atomic indexer, shape = (num_atom, num_atom-1)
            - num_atom: number of atoms in molecule, scalar
            - max_deg: maximum degree 
            - e: hyperparameter for bond strength, scalar > 2
            - g: hyperparameter for U, scalar (default=6)
    '''
    # unroll args first:
    num_basis = args[0]
    R = args[1]
    X = args[2]
    indexer = args[3]
    num_atom = args[4]
    max_deg = args[5]
    e = args[6]
    g = args[7]
    
    # unroll coefficients:
    #C = coeffs[0]; R_h = coeffs[1]; R_low = coeffs[2]; R_0 = coeffs[3]; R_m = coeffs[4]; R_up = coeffs[5]; R_C = coeffs[6];
    A1 = coeffs[7: num_basis+7]
    A2 = coeffs[num_basis+7: 2*num_basis+7]
    B1 = coeffs[2*num_basis+7: 3*num_basis+7]
    B2 = coeffs[3*num_basis+7: 4*num_basis+7]
    C1 = coeffs[4*num_basis+7: 5*num_basis+7]
    C2 = coeffs[5*num_basis+7: 6*num_basis+7]
    
    # compute energy from model:
    #print(C, R_h, R_low, R_0, R_m, R_up, R_C)
    #print(np.log(C), np.log(R_h)/20, np.sqrt(R_low), np.sqrt(R_0-R_low), np.sqrt(R_m-R_0), np.sqrt(R_up-R_m), np.sqrt(R_C-R_up))
    
    # the coeff input is rho = (np.log(C)/20, np.log(R_h)/20, np.sqrt(R_low), np.sqrt(R_0-R_low), np.sqrt(R_m-R_0), np.sqrt(R_up-R_m), np.sqrt(R_C-R_up))
    pi = eq_68_converter(coeffs[0:7]) # computes rho
    pi = eq_68_inverter(pi) # compute the pi := inverse of rho
    #print("initcoeffs", coeffs[0:7])
    #print("pi", pi)
    V_pred = f_pot_bond(pi[0], pi[1], pi[2], pi[3], pi[4], pi[5], pi[6], 
                        A1, A2, B1, B2, C1, C2, 
                        R, X, indexer, num_atom, max_deg, e, g)
    
    return V_pred

'''=== Objective functions ==='''
def f_obj_leastsquares(coeffs, *args):
    '''
    objective function in the form of residuals Y - Y_pred, shape = len(Y)
    params:
        - coeffs: array of coefficients, shape = len(coeffs)
        - *args:
            - F: evaluation function, F(.)
            - Y: actual data, shape = len(Y)
            - args[2:]: the rest of the arguments for F(.)
    '''
    # unroll variables:
    F = args[0]
    Y = args[1]
    Y_pred = F(coeffs, *args[2:])
    
    # residuals:
    res = Y - Y_pred

    #print(res)
    
    return res

def f_obj_standard(coeffs, *args):
    '''
    objective function in the form of computed least squares \Sum \Sqr Y-Y_pred, in order to be used by more (standard) opt routines, shape = scalar
    params:
        - coeffs: array of coefficients, shape = len(coeffs)
        - *args:
            - F: evaluation function, F(.)
            - Y: actual data, shape = len(Y)
            - args[2:]: the rest of the arguments for F(.)
    '''
    # unroll variables:
    F = args[0]
    Y = args[1]
    Y_pred = F(coeffs, *args[2:])

    return np.sum((Y-Y_pred)**2)

def f_obj_8_4(coeffs, *args):
    '''
    obj in sub 8.4 of Ulrik, includes \miu and \sigma (each shape = len(Datasets), however for testing just 1 data now shape = 1) in the tuning coeffs, Eq.66, shape = scalar
    params:
        - coeffs: array of coefficients, shape = len(coeffs)
        - *args:
            - F: evaluation function, F(.)
            - Y: actual data, shape = len(Y)
            - args[2:]: the rest of the arguments for F(.)
    '''
    # unroll variables:
    F = args[0]
    Y = args[1]
    miu = coeffs[-1]
    sigma = coeffs[-2]

    Y_pred = F(coeffs[:-2], *args[2:])

    return np.sum( ((Y-miu-Y_pred)**2)/(sigma**2) )

'''==== multistart opt ===='''
def multistart_method(F_obj, F_eval, Y_test, 
                      C_lb = -5., C_ub = 5., C_size = 100, mode = "leastsquares", method = "trf", max_nfev=None,
                      resets = 5, inner_loop_mode = False, inner_loop = 5, constant = 0.1, verbose_multi=0, verbose_min=0,
                      args_obj = None, args_eval = None
                     ):
    '''
    multi-start method revisited, same ol' (hopefully more modular/general), returns the RMSE and tuning coeff
    '''
    min_RMSE = np.inf; min_C = None; res = None
    # non inner loop mode:
    for i in range(resets):
        # re-init C:
        C = np.random.uniform(C_lb, C_ub, C_size)
        # optimize:
        while True: #NaN exception handler:
            try:
                #minimization routine and objective function here:
                if mode == "leastsquares":
                    res = least_squares(F_obj, C, args=args_obj, method=method, verbose=verbose_min, max_nfev=max_nfev) # scipy's minimizer
                elif mode == "standard":
                    res = minimize(F_obj, C, args=args_obj, method=method) # scipy's minimizer 
                break
            except ValueError:
                #reset C until no error:
                if verbose_multi == 1:
                    print("ValueError!!, resetting C")  
                # re-init C:
                C = np.random.uniform(C_lb, C_ub, C_size)  
                continue
        # compute RMSE:
        Y_pred = F_eval(res.x, *args_eval)
        rmse = RMSE(Y_test, Y_pred)
        if verbose_multi == 1:
            print(i, rmse)
        if rmse < min_RMSE:
            min_RMSE = rmse; min_C = res.x
            print("better RMSE obtained !!, rmse = ",min_RMSE)
    # if inner_loop_mode:
    
    return min_RMSE, min_C

'''
parallel multistart, restart-level parallelization:
'''
def getSeed(rank, size, count):
    modif = (rank)**11 + (size)**7 + (count)**3
    seeder = int(time.time())    
    if seeder>modif:
        #print("timer > modif")
        seeder = seeder - modif
    elif modif>(2**32)-1:
        #print("modif > 2**32")
        seeder = modif % seeder
    else:
        #print("modif > timer")
        seeder = modif - seeder
    return seeder

def generate_C(i, F_obj, C_lb, C_ub, C_size, mode, method, resets, max_nfev, verbose_multi, verbose_min, args_obj, res_return):
    # re-init C:
    count = 1
    #print("process "+str(i))
    seeder = getSeed(i,resets, count)        
    np.random.seed(seeder)
    C = np.random.uniform(C_lb, C_ub, C_size)
    # optimize:
    while True: #NaN exception handler:
        try:
            #print(i, C)
            #minimization routine and objective function here:
            if mode == "leastsquares":
                res = least_squares(F_obj, C, args=args_obj, method=method, verbose=verbose_min, max_nfev=max_nfev) # scipy's minimizer
            elif mode == "standard":
                res = minimize(F_obj, C, args=args_obj, method=method) # scipy's minimizer 
            break
        except ValueError:
            #reset C until no error:
            if verbose_multi == 1:
                print("ValueError!!, resetting C")  
            # re-init C:                
            count+=1
            seeder = getSeed(i,resets, count)        
            np.random.seed(seeder)
            print("process "+str(i)+" re-init | new seed : ",seeder," useable ", 0<seeder<2**32 -1)
            C = np.random.uniform(C_lb, C_ub, C_size)  
            continue
    #np.savetxt("res/multi "+str(i)+" of "+str(resets)+" seed "+str(seeder)+".txt", res.x, newline='\n')
    res_return[i] = res

def multistart_method_parallel(F_obj, F_eval, Y_test, 
                      C_lb = -5., C_ub = 5., C_size = 100, mode = "leastsquares", method = "trf", max_nfev=None,
                      resets = 5, inner_loop_mode = False, inner_loop = 5, constant = 0.1, verbose_multi=0, verbose_min=0,
                      args_obj = None, args_eval = None
                     ):
    '''
    multi-start method revisited, same ol' (hopefully more modular/general), returns the RMSE and tuning coeff
    '''
    min_RMSE = np.inf; min_C = None; res = None
    #if not os.path.exists('res'):
    #    os.makedirs('res')
    
    # non inner loop mode:
    manager = multiprocessing.Manager()
    res_return = manager.dict()
    jobs = []
    for i in range(resets):
        p = multiprocessing.Process(target=generate_C, args=(i, F_obj, C_lb, C_ub, C_size, mode, method, resets, max_nfev, verbose_multi, verbose_min, args_obj, res_return))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    
    res  = res_return.values()
    #print(res)
    for i in range(resets):
        # compute RMSE:
        Y_pred = F_eval(res[i].x, *args_eval)
        rmse = RMSE(Y_test, Y_pred)                
        if verbose_multi == 1:
            print(i, rmse)
        if rmse < min_RMSE:
            min_RMSE = rmse; min_C = res[i].x
            print("better RMSE obtained !!, rmse = ",min_RMSE)
    # if inner_loop_mode:
    
    return min_RMSE, min_C


if __name__=='__main__':
    '''unit tests:'''
    def basis_function_tests():
        # orient vec test:
        #deg = 3; data = 5
        num_atom = 4; deg = 5; data = 10
        b = np.array([1,2,3,4,5,6])
        b = b[np.newaxis, :]
        b = np.repeat(b, data, axis = 0)
        for i in range(b.shape[0]):
            b[i] += i
        b = b[np.newaxis, :]
        b = np.repeat(b, deg, axis=0)
        for i in range(b.shape[0]):
            b[i] *= i
            if i % 2 == 0:
                b[i] *= -1
        #b[2] += 1
        print("b = ")
        print(b, b.shape)
        delta = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6]])
        delta = delta[np.newaxis, :]
        delta = np.repeat(delta, data, axis=0)
        print("delta = ")
        print(delta, delta.shape)
        indexer = atom_indexer(num_atom)
        print("indexer = ")
        print(indexer, indexer.shape)
        print("Y")
        Y = Y_coord_mat(b, indexer)
        print(Y, Y.shape)
        print("b*delta = ")
        r = r_orient_vec(b, delta, indexer)
        print("r = ")
        print(r, r.shape)

        # Gram matrix test:
        G = G_gram_mat(r)
        print("G = ")
        print(G)
        print(G.shape)

        # reshape b into (num_data, d, dof) test:
        print(b.shape)
        b_reshape = np.transpose(b, (1, 0, 2))
        print(b_reshape, b_reshape.shape)

        # ref_pairpot_test:
        R = np.array([1,2,3, 4, 5, 6])
        R = R[np.newaxis, :]
        R = np.repeat(R, data, axis = 0)
        print(R, R.shape)
        indexer = atom_indexer(num_atom)
        print(indexer, indexer.shape)
        U = U_ref_energy(R, 1, 0, 4, 1, 1, indexer) #R_mat, C, R_h, R_C, R_0, g, indexer
        print(U, U.shape)

        print('U.shape', U.shape)
        print('Y.shape', Y.shape)
        print('G.shape', G.shape)
        phi = phi_fun(U, Y, G)
        print(phi, phi.shape)

        # epsilon fun test:
        num_atom = 4; num_basis = 3; 
        phi = np.array([1,2,3,4,5])
        phi = phi[np.newaxis, :]
        phi = np.repeat(phi, num_basis, axis=0)
        phi = phi[np.newaxis, :]
        phi = np.repeat(phi, num_atom, 0)
        for i in range(phi.shape[0]):
            phi[i] += i
        print(phi.shape)
        print(phi)
        A1 = B1 = C1 = np.array([-1, 1, 1])
        A2 = B2 = C2 = np.array([0,0,0])
        A = A_i_term(A1, A2, phi, 1)
        T = T0_i_term(A1, A2, phi, 1)
        print(A, T)
        print("==============")
        eps = epsilon_wrapper(phi, A1, A2, B1, B2, C1, C2)
        print("eps")
        print(eps)
        

    def opt_test():
        print("Single opt mode!")

        # load data and coordinates:
        H3_data = np.load("data/h3/h3_data.npy")
        R = H3_data[:, 0:3]; V = H3_data[:, 3]
        X = np.load("data/h3/h3_coord.npy")

        # fixed parameters:
        num_basis = 59; max_deg = 5; num_atom = 3; e = 3; g = 6;

        '''
        - C, R_h, R_low, R_0, R_m, R_up, R_C; scalar || TUNING PARAMS
        - A1, A2, B1, B2, C1, C2; shape = num_basis || TUNING PARAMS
        '''
        # initial tunable params:
        R_m = 0.74; R_low = R_m/np.sqrt(5); R_up = 1/R_low; num_atom = R.shape[1] 
        R_h = 0.1; R_0 = 0.5; R_C = R_up+0.5; C = 0.5
        theta = np.random.uniform(-1, 1, size=(6, num_basis)) # 6 for each coeff vector

        # indexer matrix:
        indexer = atom_indexer(num_atom) 

        '''
        #non wrapper version:
        V = f_pot_bond(C, R_h, R_low, R_0, R_m, R_up, R_C, 
                    theta[0], theta[1], theta[2], theta[3], theta[4], theta[5],
                    R, X, indexer, num_atom, max_deg, e, g)
        '''
        '''
        #wrapper version:
        coeffs = np.zeros((6*num_basis + 7))
        coeffs[0] = C; coeffs[1] = R_h; coeffs[2] = R_low; coeffs[3] = R_0; coeffs[4] = R_m; coeffs[5] = R_up; coeffs[6] = R_C
        coeffs[7:num_basis+7] = theta[0]; 
        coeffs[num_basis+7: 2*num_basis+7] = theta[1]; 
        coeffs[2*num_basis+7: 3*num_basis+7] = theta[2]; 
        coeffs[3*num_basis+7: 4*num_basis+7] = theta[3];
        coeffs[4*num_basis+7: 5*num_basis+7] = theta[4]; 
        coeffs[5*num_basis+7: 6*num_basis+7] = theta[5];
        start = time.time()
        V_pred = f_pot_bond_wrapper(coeffs, num_basis, R, X, indexer, num_atom, max_deg, e, g)
        elapsed = time.time()-start
        print(V_pred, V_pred.shape)
        print("time = ",elapsed)
        '''

        # === Optimize test: ===
        # get subset data:
        sub_V = V
        sub_R = R
        sub_X = X
        print("data size = ", sub_V.shape[0])

        num_basis = 59; max_deg = 5; num_atom = 3; e = 3; g = 6; # fixed parameters

        # random C:
        #C0 = np.random.uniform(-20, 20., 6*num_basis + 7) # non 8.4 ver
        #C0 = np.random.uniform(-.1, .1, 6*num_basis + 9) # 8.4 ver, extra 2 tuning coeffs
        #C0[[1,2,5,6]] = [-.1,-.1,4,4] # (1,2,5,6) = (R_h, R_low, R_up, R_C)

        # using pre-trained C:
        C0 = np.loadtxt("c_params_140322.out")
        # using scaling factor C0 := C0 x (hadamard) (1+c*(rand-0.5)), where rand is a vector with components uniformly distributed in [0,1] and c>0 is a constant chosen randomly in [0.2,5] or so:

        start = time.time()
        # residual mode:
        res = least_squares(f_obj_leastsquares, C0, args=(f_pot_bond_wrapper_trpp, sub_V, num_basis, sub_R, sub_X, indexer, num_atom, max_deg, e, g), verbose=2, method="lm")
        # scalar mode:
        #res = minimize(f_obj_standard, C0, args=(f_pot_bond_wrapper_trpp, sub_V, num_basis, sub_R, sub_X, indexer, num_atom, max_deg, e, g), method="BFGS")
        # 8.4 scalar mode:
        #res = minimize(f_obj_8_4, C0, args=(f_pot_bond_wrapper_trpp, sub_V, num_basis, sub_R, sub_X, indexer, num_atom, max_deg, e, g), method="BFGS")

        elapsed = time.time()-start
        #print(repr(res.x)) #print with commas

        # === RMSE: ===
        # standard obj mode:
        V_pred = f_pot_bond_wrapper_trpp(res.x, num_basis, sub_R, sub_X, indexer, num_atom, max_deg, e, g)
        
        # 8.4 mode (extra 2 params: miu and sigma):
        #V_pred = f_pot_bond_wrapper_trpp(res.x[:-2], num_basis, sub_R, sub_X, indexer, num_atom, max_deg, e, g)
        
        rmse = RMSE(V_pred, sub_V)

        # save to file:
        np.savetxt('c_params.out', res.x, delimiter=',')
        print(np.loadtxt('c_params.out'))
        print("time = ",elapsed)
        print('final rmse', rmse)

    def multistart_test():
        print("Multirestart opt mode!!")

        # load data and coordinates:
        H3_data = np.load("data/h3/h3_data.npy")
        R = H3_data[:, 0:3]; V = H3_data[:, 3]
        X = np.load("data/h3/h3_coord.npy")
        # get subset:
        n = len(H3_data)
        sub_V = V[:n]
        sub_R = R[:n]
        sub_X = X[:n]
        print("data size = ", sub_V.shape[0])

        # fixed parameters:
        num_basis = 59; max_deg = 5; num_atom = 3; e = 3; g = 6;
        indexer = atom_indexer(num_atom) 
        
        # multirestart:
        resets = 100
        print("resets = ",resets)
        start = time.time()
        
        #singlecore ver:
        '''
        rmse, C = multistart_method(f_obj_leastsquares, f_pot_bond_wrapper_trpp, 
                                    Y_test=sub_V, C_lb=-20., C_ub=20., C_size=6*num_basis+7, mode="leastsquares", max_nfev=2000,
                                    resets=resets, verbose_multi=1, verbose_min=2,
                                    args_obj=(f_pot_bond_wrapper_trpp, sub_V, num_basis, sub_R, sub_X, indexer, num_atom, max_deg, e, g),
                                    args_eval=(num_basis, sub_R, sub_X, indexer, num_atom, max_deg, e, g))
        '''
        #parallel ver:
        rmse, C = multistart_method_parallel(f_obj_leastsquares, f_pot_bond_wrapper_trpp, 
                                    Y_test=sub_V, C_lb=-20., C_ub=20., C_size=6*num_basis+7, mode="leastsquares", method = 'lm', max_nfev=5000,
                                    resets=resets, verbose_multi=1, verbose_min=2,
                                    args_obj=(f_pot_bond_wrapper_trpp, sub_V, num_basis, sub_R, sub_X, indexer, num_atom, max_deg, e, g),
                                    args_eval=(num_basis, sub_R, sub_X, indexer, num_atom, max_deg, e, g))
        
        elapsed = time.time()-start
        # save to file:
        np.savetxt('c_params.out', C, delimiter=',')
        print(np.loadtxt('c_params.out'))
        print("time = ",elapsed)
        print('final rmse', rmse)
        #print(repr(C))
        

    def opt_routine(data_index_dir = None, pretrained_C = None, multirestart = False, parallel = False, resets = 100, 
                    mode = "leastsquares", method = "lm", max_nfev = None, C_lb = -20., C_ub = 20., verbose_multi=1, verbose_min=2):
        '''
        main fun to wrap all things needed for opt, only cover the vars for data processing, just for convenience
        '''

        # load data and coordinates:
        H3_data = np.load("data/h3/h3_data.npy")
        R = H3_data[:, 0:3]; V = H3_data[:, 3]
        X = np.load("data/h3/h3_coord.npy")
        # get subset by index:
        if data_index_dir == None:
            # full data:
            sub_V = V
            sub_R = R
            sub_X = X
        else:
            idx = np.load(data_index_dir, allow_pickle=True)
            print("using crossval data splitting", idx[0].shape, idx[1].shape)
            # data for training, which is used for opt:
            sub_V = V[idx[0]]
            sub_R = R[idx[0]]
            sub_X = X[idx[0]]
        print("data size =", sub_V.shape[0])

        # fixed parameters:
        num_basis = 59; max_deg = 5; num_atom = 3; e = 3; g = 6;
        indexer = atom_indexer(num_atom)

        # multirestart:
        if multirestart:
            print("Multirestart opt mode!!")
            print("resets = ",resets)
            start = time.time()
            # parallelized ver:
            if parallel:
                rmse, C = multistart_method_parallel(f_obj_leastsquares, f_pot_bond_wrapper_trpp, 
                                    Y_test=sub_V, C_lb=C_lb, C_ub=C_ub, C_size=6*num_basis+7, mode=mode, method = method, max_nfev=max_nfev,
                                    resets=resets, verbose_multi=verbose_multi, verbose_min=verbose_min,
                                    args_obj=(f_pot_bond_wrapper_trpp, sub_V, num_basis, sub_R, sub_X, indexer, num_atom, max_deg, e, g),
                                    args_eval=(num_basis, sub_R, sub_X, indexer, num_atom, max_deg, e, g))
            # singlecore ver:
            else:
                rmse, C = multistart_method(f_obj_leastsquares, f_pot_bond_wrapper_trpp, 
                                    Y_test=sub_V, C_lb=C_lb, C_ub=C_ub, C_size=6*num_basis+7, mode=mode, method = method, max_nfev=max_nfev,
                                    resets=resets, verbose_multi=verbose_multi, verbose_min=verbose_min,
                                    args_obj=(f_pot_bond_wrapper_trpp, sub_V, num_basis, sub_R, sub_X, indexer, num_atom, max_deg, e, g),
                                    args_eval=(num_basis, sub_R, sub_X, indexer, num_atom, max_deg, e, g))
            elapsed = time.time()-start
        # single local search:
        else:
            print("Single local search mode!!")
            # pre-trained C:
            if pretrained_C:
                print("using pretrained C")
                C0 = np.loadtxt(pretrained_C)
            else:
                print("init C using random uniform")
                C0 = np.random.uniform(C_lb, C_ub, 6*num_basis + 7)

            print("opt mode =",mode)
            start = time.time()
            if mode == "leastsquares":
                res = least_squares(f_obj_leastsquares, C0, args=(f_pot_bond_wrapper_trpp, sub_V, num_basis, sub_R, sub_X, indexer, num_atom, max_deg, e, g), verbose=verbose_min, method=method, max_nfev=max_nfev)
            else:
                res = minimize(f_obj_standard, C0, args=(f_pot_bond_wrapper_trpp, sub_V, num_basis, sub_R, sub_X, indexer, num_atom, max_deg, e, g), method=method)
            elapsed = time.time()-start
            C = res.x
            # training rmse:
            V_pred = f_pot_bond_wrapper_trpp(C, num_basis, sub_R, sub_X, indexer, num_atom, max_deg, e, g)
            rmse = RMSE(V_pred, sub_V)


        # save to file:
        np.savetxt('c_params.out', C, delimiter=',')
        print(np.loadtxt('c_params.out'))
        print("time = ",elapsed)
        print('training rmse', rmse)
        # compute test rmse using test data:
        if data_index_dir:
            test_R = R[idx[1]]
            test_X = X[idx[1]]
            test_V = V[idx[1]]
            V_pred = f_pot_bond_wrapper_trpp(C, num_basis, test_R, test_X, indexer, num_atom, max_deg, e, g)
            rmse_test = RMSE(V_pred, test_V)
            print('testing rmse', rmse_test)



    #basis_function_tests()
    #opt_test()
    #multistart_test()
    #opt_routine(data_index_dir="data/h3/crossval_indices_0.npy", pretrained_C="c_params_140322.out", multirestart=True, parallel=True, resets = 3, mode = "leastsquares", method='trf', max_nfev=10)
    
    '''etc functions:'''
    def testprofile():
        # load data and coordinates:
        H3_data = np.load("data/h3/h3_data.npy")
        R = H3_data[:, 0:3]; V = H3_data[:, 3]
        print(R.dtype)
        X = np.load("data/h3/h3_coord.npy")
        sub_V = V
        sub_R = R
        sub_X = X
        # fixed parameters:
        num_basis = 59; max_deg = 5; num_atom = 3; e = 3; g = 6;

        # indexer matrix:
        indexer = atom_indexer(num_atom)

        C0 = np.loadtxt("c_params_220322_full_fold1_5e-2.out")

        start = time.time()
        #for i in range(int(10)):
        V_pred = f_pot_bond_wrapper_trpp(C0, num_basis, sub_R, sub_X, indexer, num_atom, max_deg, e, g)
        print(V_pred)
        print(time.time()-start,"s")
        
        # RMSE:
        rmse = RMSE(V_pred, sub_V)
        print('rmse', rmse)
        
    
    '''==== callers: ==='''
    testprofile()
    
