# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 20:05:41 2021

@author: Saint8312
"""

'''Query routines'''
def query_one_var(value, key, list_data):
    #to query data where data[key] = value
    query_data = []
    for data in list_data:
        if data[key] == value:
            query_data.append(data)
    return query_data

def query_one_var_indices(value, key, list_data):
    #returns the indices instead of the data
    q_idxes = []
    for i, data in enumerate(list_data):
        if data[key] == value:
            q_idxes.append(i)
    return q_idxes

def query_many_vars_indices(values, keys, list_data):
    #queries many keys at once, e.g.: "mol" == "OH", "state" == "$X^2\Pi$", "author" == "pradhan(1995)", etc
    #the entries within the values and keys list must be consistent, e.g. 1st index is state, 2nd index is author,etc..
    q_idxes = []
    for i, data in enumerate(list_data):
        hit = 0
        for j, key in enumerate(keys):
            if data[key] == values[j]:
                hit += 1 #if a key hits, increment
        if hit == len(keys):
            q_idxes.append(i)
    return q_idxes

def many_queries_many_vars_indices(dicts, list_data):
    #queries many values and many keys at once!!, e.g.: [{"mol" = "OH", "state" = "$X^2\Pi$"}, {"mol" = "OH", "state" = "$X^2\Sigma$"}, etc]
    q_idxes = []
    for i, data in enumerate(list_data):
        for j, dicti in enumerate(dicts):
            hit = 0
            for k, key in enumerate(dicti):
                value = dicti[key]
                try:
                    if data[key] == value:
                        hit += 1 #if a key hits, increment
                except KeyError: #if no key found
                    continue
            if hit == len(dicti):
                q_idxes.append(i)
    return q_idxes

'''=== utilities ==='''
def data_conversion(R, V, R_unit="angstrom", V_unit="ev"):
    #convert R to bohr, V to hartree (currently from 'angstrom' and 'ev' only)
    #distance:
    if R_unit or R:
        if R_unit=="angstrom":
            R *= 1.8897259886
    
    #energy:
    if V_unit or V:
        if V_unit == "ev":
            V *= 0.0367502
        elif V_unit=="milihartree":
            V *= 1e-3
        elif V_unit == "cm-1":
            V *= 4.55633e-6
        elif V_unit == "kjmol-1":
            V *= 0.00038088
    return R, V

# transform distance vectors to distance matrices:
def distance_vec_to_mat(R, n_atom):
    '''
    returns distance matrix D_{ij}, symmetric with D_{ii} = 0 forall i
    params:
        - R = vector of distances (R_12, R_13, R_23, ...)
        - n_atom = number of atoms in the molecule
    '''
    upper_idxes = np.triu_indices(n_atom, 1) # get the 
    dist_mat = np.zeros((n_atom, n_atom))
    dist_mat[upper_idxes] = R
    dist_mat = dist_mat + dist_mat.T # no need to consider diagonals because theyre always 0
    return dist_mat

# transform distance matrix into coordinate matrix
def distance_to_coord(dist_vec, N, verbose=False):
    '''
    from https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix
    convert distance vector r_{ij} to coordinate matrix X := [X_0, X_1, X_2], each element is one point (row wise).
    params: 
        - dist_mat = distance matrix, symmetric with 0 diagonals
    '''
    dist_square = dist_vec**2 #q_{ik}:=r_{ik}^2
    dist_square = distance_vec_to_mat(dist_square, N) # D_{ij} matrix from r_{ij}
    M_mat = np.zeros(dist_square.shape)
    #dist_square = dist_mat**2 # D^2
    vec_sum = np.sum(dist_square, axis = 0) # since it's symmetric, sums for row = col
    M_submat = M_mat[1:, 1:] # submatrix excluding 0s in the first column and row
    upper_idxes = np.triu_indices(M_submat.shape[0], 1)
    row = upper_idxes[0]; col = upper_idxes[1] # upper triangular indexes of the submatrix
    row_actual = row+1; col_actual = col+1 # corresponding (row, col) index matrix of the actual matrix
    for idx in range(row.shape[0]): # fill the off diagonals:
        i = row[idx]; j = col[idx];
        i_actual = row_actual[idx]; j_actual = col_actual[idx]
        M_submat[i][j] = (dist_square[0][j_actual] + dist_square[i_actual][0] - dist_square[i_actual][j_actual])/2
    M_submat = M_submat + M_submat.T
    #print(M_submat)
    # fill the diagonals:
    diag = np.diag_indices(M_submat.shape[0])
    M_submat[diag] = dist_square[0][1:] # M'_i-1i-1 := M_ii = D_1i^2
    # return submat to original:
    M_mat[1:, 1:] = M_submat
    if verbose:
        print("M", M_mat)
    # eigendecomposition:
    eigvals, eigvecs = np.linalg.eigh(M_mat) # symmetric matrix eigendecomposition, uses (eigh)ermitian
    #print(eigvecs @ np.diag(eigvals) @ eigvecs.T) # M = Q*lambda*Q^t
    # replace any very small |x| s.t. x<0 \in R, with 0, if |x|<delta:
    delta = -1e-6 # intuitive near 0 threshold
    eigvals[np.where((eigvals > delta) & (eigvals < 0))] = 0
    if verbose:
        print("eigval =",eigvals)
        print("eigvec =",eigvecs)
    X = eigvecs @ np.diag(np.sqrt(eigvals)) # coordinate matrix, each coordinate = X[i]
    return X

def distance_to_coord_v2(dist_vec, N, verbose=False):
    '''
    convert distance vector r_{ij} to coordinate matrix X := [X_0, X_1, X_2], each element is one point (row wise).
    params: 
        - dist_vec = distance vector, (d_1, d_2, ... )
        - N = number of points, scalar
    '''
    q = dist_vec**2 #q_{ik}:=r_{ik}^2
    Q = distance_vec_to_mat(q, N) # convert to distance matrix
    #print("Q", Q)
    gamma_i = np.sum(Q, axis=1)/N # \gamma_i:=\frac{1}{N}\sum_k q_{ik}:
    #print("gamma_i", gamma_i)
    gamma = np.sum(gamma_i)/(2*N) # \gamma:=\frac{1}{2N}\sum_i \gamma_i
    #print("gamma", gamma)
    G_diag = gamma_i - gamma # G_{ii}=\gamma_i-\gamma
    #print("G_diag", G_diag)
    G = np.diag(G_diag) # G_{ii}=\gamma_i-\gamma
    #print("G", G)
    
    # G_{ik}=\frac12(G_{ii}+G_{kk}-q_{ik}): (probably better to use upper triangular):
    for i in range(N):
        for k in range(N):
            if i != k:
                G[i][k] = (G[i][i] + G[k][k] - Q[i][k])/2
    if verbose:
        print("G:", G)
    eigvals, eigvecs = np.linalg.eigh(G)
    # replace any very small |x| s.t. x<0 \in R, with 0, if |x|<delta:
    delta = -1e-6 # intuitive near 0 threshold
    eigvals[np.where((eigvals > delta) & (eigvals < 0))] = 0
    if verbose:
        print("eigvals =", eigvals)
        print("eigvecs =", eigvecs)
    X = eigvecs @ np.diag(np.sqrt(eigvals))
    return X

'''=== cross validation utilities ==='''
def cross_val(save_dir, num_data, n_split):
    '''
    split the data into n_splits for cross validation.
    returns the indices of each split and save them to file (2D object array, [0] is the training indices, [1] is the testing indices)
    params:
        - save_dir, directory/folder, string
        - num_data, number of data, integer
        - n_split, indicates the number of splits, integer
    '''
    # only this function uses this
    from sklearn.model_selection import KFold 
    kf = KFold(n_splits=n_split, random_state=0, shuffle=True)
    X = np.zeros((num_data)) # dummy data, just for indexing
    i = 0
    for train, test in kf.split(X):
        #print("%s %s" % (train, test))
        np.save(save_dir+"crossval_indices_"+str(i), np.array([train, test], dtype=object))
        i+=1

if __name__ == "__main__":
    import numpy as np
    def conv_pd_to_pkl():
        import pandas as pd
        import pickle

        files = ["result/res_joint_OH+_271021_153617.pkl", "result/res_joint_OH+_291021_160942.pkl", "result/performance_22102021.pkl"]
        for f in files:
            data = pd.read_pickle(f)
            data = data.to_dict(orient="list")
            print(data)
            with open(f, 'wb') as handle:
                pickle.dump(data, handle)
                
    '''======call functions from here (main is from here)========'''
    #conv_pd_to_pkl()
    '''
    H3_data = np.load("data/h3/h3_data.npy")
    cross_val("data/h3/", H3_data.shape[0], 5)
    print(np.load("data/h3/crossval_indices_0.npy", allow_pickle=True)[1])
    '''
    dir = "data/h5/h5_data.txt"
    Hn = np.loadtxt(dir)
    print(Hn.shape)
    #cross_val("data/h5/", Hn.shape[0], 5)
    #print(np.load("data/h5/crossval_indices_0.npy", allow_pickle=True)[1])

    R = Hn[:, :-1]
    V = Hn[:, -1]
    X = []
    for r in R:
        #print(distance_to_coord_v2(r, 4)[:,1:])
        #X.append(distance_to_coord_v2(r, 4)[:,1:])
        X.append(distance_to_coord_v2(r, 5))
    X = np.array(X)
    np.save("data/h5/h5_coord", X)

    print(R[0])
    X = np.load("data/h5/h5_coord.npy")
    print(np.linalg.norm(X[0][1] - X[0][2]))
    print(X.shape)