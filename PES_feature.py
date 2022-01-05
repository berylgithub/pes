# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 18:54:49 2022

@author: Saint8312
"""
import numpy as np


normalizer = lambda x: 2*(x - np.min(x))/(np.max(x) - np.min(x)) - 1 #function to normalize any x into [-1, 1]

if __name__=="__main__":
    A = np.array([0,1,2,3,4,5])
    print(normalizer(A))