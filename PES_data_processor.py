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

'''utilities'''
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