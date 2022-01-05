# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 18:54:49 2022

@author: Saint8312
"""
import numpy as np

data_normalizer = lambda x, min_val, max_val: 2*(x - min_val)/(max_val - min_val) - 1 #function to normalize any x into [-1, 1]
data_rewinder = lambda x, min_val, max_val: (x+1)*(max_val - min_val)/2 + min_val


if __name__=="__main__":
    A = np.array([0.5,1,2.1,3.3,4,5])
    min = np.min(A); max = np.max(A)
    An = data_normalizer(A, min, max)
    print(An)
    A = data_rewinder(An, min, max)
    print(A)