# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 07:36:29 2021

@author: Saint8312
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
#dfbgn:
import dfbgn
#lmfit:
from lmfit import Parameters, minimize
from lmfit.printfuncs import report_fit
#timer:
import time
#visualization:
import pandas as pd




'''useful analytics/statistics functions'''

## RMSE:
def RMSE(Y_pred, Y):
    #error between Y prediction and Y actual
    error = 0
    n = len(Y)
    for i in range(n):
        diff = (Y[i]-Y_pred[i])**2
        error += diff
    return np.sqrt(error/n)

## sech:
def sech(x):
    exp = np.exp(x)
    inv_exp = 1/exp
    return 2/(exp + inv_exp)

## horner scheme:
def horner(x, C):
    y = C[-1]
    for i in range(len(C)-2, -1, -1):
        y = y*x + C[i]
    return y


'''
The models and objective functions
'''
'''Pair Potentials:'''
np.random.seed(13) #reprodicuibility

#for morse, the equilibrium distance is a good start point for r0

## First proposed model:
f_morse_pot = lambda r,a: np.exp(-r/a)

def f_poly(x, c): #general polynomial, evaluated at x by horner's scheme
    y = c[-1]
    for i in range(len(c)-2, -1, -1):
        y = y*x + c[i]
    return y
    
def f_monomial_diatomic_pot(r, coeffs, morse=False): #monomial pot - permutationally invariant functional 
    #r := r_i, the ith distance
    #coeffs := the coefficients, in increasing order of polynomial (0,1,2,...)
    #evaluate the polynomial by horner's scheme:
    y = r
    v = coeffs[-1]
    for i in range(len(coeffs)-2, -1, -1):
        #i := power of the polynomial
        if morse and i > 0: #exclude the constant for morse
            y = f_morse_pot(r,i)
        v = v*y + coeffs[i]
    return v

## second proposed model:
def f_diatomic_pot_lr(C, *args):
    #C = fitted coefficients, [0:len(C)] is the polynomials' coefficients c_i, [-1] is the nonlinear decaying exp coeff r_0
    ##uses *args because dfbgn doesnt accept **kwargs
    #args[0] := r, the internuclear distance
    #args[1] := Z = (Z_1, Z_2), the tuple of nuclear charges
    #M, the max polynomial degree (= len(C)-2)
    r0 = C[-1]
    r = args[0]
    Z = args[1]
    M = len(C)-2
    y = np.exp(-r/r0)
    #V = (Z[0]*Z[1])/r + C[0]
    V = (Z[0]*Z[1] + C[0])/r*(1+C[1]*(r**6))
    
    poly_V = C[-2]
    #for i in range(M-1, 1, -1): 
    for i in range(M-1, 2, -1):
        poly_V = poly_V*y + C[i]
        #print(i, poly_V, C[i])
    '''
    #non horner version:
    poly_V = 0
    #for i in range(2, M):
    for i in range(3, M): 
        poly_V += C[i]*(y**(i-3))
        #print(i, poly_V)
        #poly_V = poly_V*y + C[i-2]
    '''
    #poly_V *= ((1-y)/(r**3 + C[1]))**2
    poly_V *= ((1-y)/(r**3 + C[2]))**2
    V += poly_V
    return V


## third propopsed model:
def f_diatomic_vdw(C, *args):
    #m free parameters:
    R = args[0]
    Z = args[1] #Zi x Zj
    m = args[2] #the maximum polynomial power
    # V_infty = args[1]

    # C[1] -> |C[1]|:
    if C[1] < 0:
        C[1] = -C[1]
    
    #operations:
    R2 = R*R; 
    
    #R4 = R2**2; R6 = R2*R4; 
    
    '''
    s = 1 + C[0]*R2 + C[1]*R4;
    p = s/(s + C[2]*R6);
    #p = 1/(C[0] + R6)
    q = C[3] + (C[4] + C[5]*R2)**2
    
    V = (Z/R)*p - (R/q)**2 +C[6]
    '''
    
    #################################################
    '''   
    specific, m=2:

    V(r)=c_0+A(r)/C(r),

    where

    A(r)=Z_{ij}(1/r+c_1  r)+c_2+c_3 r^2,

    B(r)=1+(c_4+c_5 r +c_6 r^2)r,

    C(r)=1+c_1 r^2 B(r)^2,

    c_1>0
    '''
    
    #7 free params:
    '''
    a = Z*(1/R + C[1]*R) + C[2] + C[3]*R2
    b = 1 + (C[4] + C[5]*R + C[6]*R2)*R
    c = 1 + C[1]*R2*(b**2)
    V = C[0] + a/c
    '''    

    ##################################################
    '''
    Generalized m free parameters:
    
    V(r)=c_0+A(r)/C(r),

    where

    A(r)=Z_{ij}(1/r+c_1 r)+c_2+c_3 r2+ ...+c_{2m-1}r^{2m-2}

    B(r)=1+(c_{2m}+c_{2m+1} r +...+c_{3m} r^m)r,

    C(r)=1+c_1 (r B(r))2,

    c_1>0
    '''
    
    '''
    '''
    
    # operations on V:
    
    # a by horner:
    C_temp = C[3 : 2*m] # index [3, 2m-1]
    y = horner(R, C_temp)*R2 #multiply by R^2 at the end
    a = Z*(1/R + C[1]*R) + C[2] + y
    
    # b by horner:
    C_temp = C[2*m : 3*m + 1] # index [2m, 3m]
    y = horner(R, C_temp)
    b = 1 + y*R
    
    c = 1 + C[1]*((R*b)**2)
    
    V = C[0] + a/c
    return V

#alias:
f_diatomic_ansatz_0 = f_diatomic_vdw

'''fourth proposed model'''
def f_diatomic_ansatz_1(C, *args): #ansatz 0 was the f_diatomic_vdw
    R = args[0]
    Z = args[1]
    M = args[2] #degree of pol
    #coefficients:
    a = C[: M]
    b = C[M : 2*M]
    c = C[2*M : 3*M]
    d = C[3*M : 4*M]
    #physical params:
    s = Z/R; t=s**2
    q = t/(1+t)
    prod = 1
    for k in range(M):
        numer = (q - a[k])**2 + b[k]
        denom = (q - c[k])**2 + d[k]
        prod *= numer/denom
    
    V = (q**3)*(s - prod)
    return V


def f_diatomic_ansatz_2(C, *args):
    '''rational potential without Z, total of 4M+7 parameters'''
    R = args[0]
    M = args[1] #degree of pol
    
    #coefficients:
    a = C[: M]
    b = C[M : 2*M]
    c = C[2*M : 3*M+4]
    d = C[3*M+4 : 4*M+7]
    
    '''
    #b_i \ge 0 for i > 1:
    for i in range(1, M):
        if b[i] < 0:
            b[i] = -b[i]
    
    #d_i \ge 0 for i > 0:
    for i in range(M):
        if d[i] < 0:
            d[i] = -d[i]
    '''
    
    #evaluates P:
    #the last indexed coefficients are outside of the loop, so less index operations
    P = c[-2]
    for i in range(M): #i=0,1...m-1
        P *= (R - a[i])**2 + b[i]*R
           
    #evaluates Q:
    Q = (R + d[-1])*R
    for i in range(M+2): #i=0,1,..m+1
        Q *= (R - c[i])**2 + d[i]*R
    
    #the rational potential:
    V = c[-1] + (P/Q)
    
    return V

def f_diatomic_ansatz_2_unconstrained(C, *args):
    '''unconstrained version of ratpot2'''
    R = args[0]
    M = args[1] #degree of pol
    
    #coefficients:
    a = C[: M]
    b = C[M : 2*M]
    c = C[2*M : 3*M+4]
    d = C[3*M+4 : 4*M+7]
    
    #b_i \ge 0 for i > 1:
    for i in range(1, M):
        if b[i] < 0:
            b[i] = -b[i]
    
    #d_i \ge 0 for i > 0:
    for i in range(M):
        if d[i] < 0:
            d[i] = -d[i]
    
    #evaluates P:
    #the last indexed coefficients are outside of the loop, so less index operations
    P = c[-2]
    for i in range(M): #i=0,1...m-1
        P *= (R - a[i])**2 + b[i]*R
           
    #evaluates Q:
    Q = (R + d[-1])*R
    for i in range(M+2): #i=0,1,..m+1
        Q *= (R - c[i])**2 + d[i]*R
    
    #the rational potential:
    V = c[-1] + (P/Q)
    
    return V


def f_diatomic_ansatz_3(C, *args):
    '''RATPOT3: with coulomb term, total of 4m+8 parameters'''
    R = args[0]
    Z = args[1]
    M = args[2] #degree of pol
    
    #coefficients:
    a = C[: M]
    b = C[M : 2*M]
    c = C[2*M : 3*M+4]
    d = C[3*M+4 : 4*M+7]
    R0 = C[-1] #or C[4M+7]
    
    P = Z*(1/R - 1/R0)
    for i in range(M): #0,1,..M-1 (M length)
        P *= (1 - R/a[i])**2 + R/b[i]
    
    Q = 1
    for i in range(M+3): #0,1,..M+2 (M+3 length)
        Q *= (1 - R/c[i])**2 + R/d[i]
    
    V = c[-1] + (P/Q)
    
    return V

def f_diatomic_ansatz_3_unconstrained(C, *args):
    '''RATPOT3 unconstrained version'''
    R = args[0]
    Z = args[1]
    M = args[2] #degree of pol
    
    #coefficients:
    a = C[: M]
    b = C[M : 2*M]
    c = C[2*M : 3*M+4]
    d = C[3*M+4 : 4*M+7]
    R0 = C[-1] #or C[4M+7]
    
    for i in range(len(b)):
        if b[i] < 0:
            b[i] = -b[i]
            
    for i in range(len(d)):
        if d[i] < 0:
            d[i] = -d[i]
    
    P = Z*(1/R - 1/R0)
    for i in range(M): #0,1,..M-1 (M length)
        P *= (1 - R/a[i])**2 + R/b[i]
    
    Q = 1
    for i in range(M+3): #0,1,..M+2 (M+3 length)
        Q *= (1 - R/c[i])**2 + R/d[i]
    
    V = c[-1] + (P/Q)
    
    return V
    
    
#######################################################
### CHIPR models: 
### for OH+:
def f_diatomic_chipr_ohplus(C, *args):
    '''
    #free parameters, length = 19, for (m, M) = (4, 4): 
    R_oh = C[0] 
    omega = C[1] 
    a = C[2:6]; 
    A = C[6:10];
    zeta = C[10];
    miu = C[11:15];
    chi = C[15:19];
    
    #fixed parameters:
    R = args[0]
    
    #operations:
    R_oh_2 = R_oh**2; R_oh_3 = R_oh_2*R_oh
    
    x = (sech((R - zeta)*miu[0])**chi[0])*a[0] \
    + (sech((R - zeta*R_oh)*miu[1])**chi[1])*a[1] \
    + (sech((R - zeta*R_oh_2)*miu[2])**chi[2])*a[2] \
    + (sech((R - zeta*R_oh_3)*miu[3])**chi[3])*a[3]
    x2 = x**2; x3 = x2*x; x4 = x3*x
    V = 8*(R**(-omega))*(x*A[0] + x2*A[1] + x3*A[2] + x4*A[3])
    '''
    
    ###################################################################
    # CHIPR with (3m + M + 3) parameters:
    R = args[0]
    Z = args[1]
    M = args[2] #outer M
    m = args[3] #inner m
    
    omega = C[0]
    zeta = C[1]
    R_oh = C[2]
    miu = C[3:m+3]
    chi = C[m+3:2*m+3]
    a = C[2*m+3:3*m+3]
    A = C[3*m+3:3*m+3+M]
    
    #compute x, using inner m:
    x = 0
    Ri = 1 #R^i
    for i in range(m):
        theta = (R - zeta*Ri)*miu[i]
        x += (sech(theta)**chi[i])*a[i]
        Ri *= R_oh #compute the power of R_oh
    
    # horner for M polynom:
    y = horner(x, A)*x # the polynomial has 1 extra degree
    V = y*(Z*(R**(-omega)))
    return V

    
## Deiters + Neumaier model:
def f_diatomic_dn(C, *args):
    #5 free parameters (\alpha is placed last in the array):
    R = args[0]
    R2 = R**2; R_1 = 1/R; R8 = R2**4;
    V = (C[0]*np.exp(-2*C[3]*R))*(C[3] + R_1) - (C[1]*R2)/(C[2] + R8) + C[4]
    return V


## Deiters + Sadus model:
def f_diatomic_ds(C, *args):
    #8 free parameters:
    R = args[0]
    R2 = R**2; R6 = R2**3;
    numer = (C[0]/R)*(np.exp(C[1]*R + C[6]*R2)) + C[2]*(np.exp(C[3]*R)) + C[4]
    denom = 1 + C[5]*R6
    return numer/denom + C[7]

## LJ pot:
def f_lj_pot(C, *args):
    #2 params
    R = args[0]
    R2 = R*R; R4 = R2*R2; R6 = R4*R2; R12 = R6*R6
    return (C[0]/R12) - (C[1]/R6)

'''============= Objective functions =================='''

def f_poly_obj(R, V, coeffs, F, poly_par): #least squares for all polynomials 
    #R = vector of distances
    #V = vector of energies
    #F = polynomial function
    #coeffs = coefficients vector
    #poly_par = polynomial function parameters {"M" = max degree,"morse" = switch for morse fun}
    M = poly_par["M"]; morse = poly_par["morse"]
    ssum = 0
    length = len(R)
    for i in range(length):
        p = F(R[i], coeffs, morse)
        diff = (p-V[i])**2
        ssum += diff
    return ssum

def f_poly_obj2(C, *args): #least squares for all polynomials 
    #R = vector of distances
    #V = vector of energies
    #F = polynomial function
    #coeffs = coefficients vector
    #poly_par = polynomial function parameters {"M" = max degree,"morse" = switch for morse fun}
    F = args[0]
    R = args[1]
    V = args[2]
    poly_par = args[-1]
    M = poly_par["M"]; morse = poly_par["morse"]
    ssum = 0
    length = len(R)
    for i in range(length):
        p = F(R[i], C, morse)
        diff = (p-V[i])**2
        ssum += diff
    return ssum

def f_poly_res(C, *args): #residuals function
    #R = vector of distances
    #V = vector of energies
    #F = polynomial function
    #coeffs = coefficients vector
    #poly_par = polynomial function parameters {"M" = max degree,"morse" = switch for morse fun}
    F = args[0]
    R = args[1]
    V = args[2]
    poly_par = args[-1]
    M = poly_par["M"]; morse = poly_par["morse"]
    length = len(R)
    residuals = np.zeros(length)
    for i in range(length):
        p = F(R[i], C, morse)
        residuals[i] = (p-V[i])**2
    return residuals

def f_poly_res2(C, *args): #residuals function
    #R = vector of distances
    #V = vector of energies
    #F = polynomial function
    #coeffs = coefficients vector
    #poly_par = polynomial function parameters {"M" = max degree,"morse" = switch for morse fun}
    F = args[0]
    R = args[1]
    V = args[2]
    poly_par = args[-1]
    M = poly_par["M"]; morse = poly_par["morse"]
    length = len(R)
    residuals = np.zeros(length)
    for i in range(length):
        p = F(R[i], C, morse)
        residuals[i] = p-V[i]
    return residuals


def f_obj_diatomic_pot_res(C, *args):
    #the objective function to be fitted, in residuals form, for least squares
    #C = coefficients to be fitted
    #args[0] = function to be fitted, returns predicted data, Y_pred
    #args[1] = vector of actual data, Y
    #args[2:] = args of args[0]
    F = args[0]
    V = args[1] #the energy vector
    #R = args[2] #the distance vector
    #Z = args[3] #nuclear charges of two atoms
    #error = 0 
    n = len(V)
    residuals = np.zeros(n)
    V_pred = F(C, *args[2:])
    residuals = np.square(V_pred - V)
    #for i in range(n):
    #    v_pred = F(C, args[3:])
    #    residuals[i] = V[i]-v_pred
    return residuals.flatten()


############ lmfit version of objective functions ##############

def f_obj_diatomic_pot_res_lmfit(C_param, *args):
    #the objective function to be fitted, in residuals form, for least squares
    #C = coefficients to be fitted, dictionary form {""}
    #args[0] = function to be fitted, returns predicted data, Y_pred
    #args[1] = vector of actual data, Y
    #args[2:] = args of args[0]
    F = args[0]
    V = args[1]
    n = len(V)
    residuals = np.zeros(n)
    # Transform dictionary to array:
    C = np.array([C_param[key] for key in C_param]) #ordered array of parameters, scipy convention
    V_pred = F(C, *args[2:])
    #residuals = np.square(V_pred - V)
    residuals = V_pred - V
    return residuals.flatten()


'''======= coeff vectors generator ========='''

def coeff_generator_ansatz3(mode="initialize", C=None, M=None, prev_M=None, random=True, pwr=1e-1, const=1e-3):
    # 3 modes: generate C from previous C (append), multiply subvector with some constants (power), generate completely new C (initialize)
    # C is previous coefficient
    # const is positive
    # if C is None, then generate new vector by M, else if power is not none then it's v_J*const mode, otherwise, generate new C from prev C using prev_M
    # random  True if C is not none means the new constants will be randomized, otherwise const will be used
    
    if mode == "append":
        #a:
        if random:
            const = np.random.uniform(-1, 1, 1)
        a = C[: prev_M]; a = np.hstack((a, const))
        #b > 0:
        if random:
            const = np.random.uniform(const, 1, 1)
        b = C[prev_M : 2*prev_M]; b = np.hstack((b, const))
        #c:
        if random:
            const = np.random.uniform(-1, 1, 1)
        c = C[2*prev_M : 3*prev_M+4]
        ins_idx = len(c)-1
        c = np.insert(c, ins_idx, const)
        #d > 0:
        if random:
            const = np.random.uniform(const, 1, 1)
        d = C[3*prev_M+4 : 4*prev_M+7]
        ins_idx = len(d)
        d = np.insert(d, ins_idx, const)
        #R0:
        R0 = C[-1]
    elif mode == "power":
        # only multiply the variable constants:
        idxes = [M-1, 2*M-1, 3*M+2, 4*M+6]
        C[idxes] *= pwr
    elif mode == "initialize":
        #generate init C_params:
        a = np.random.uniform(-1, 1, M)
        b = np.random.uniform(1e-3, 1, M) #b>0
        c = np.random.uniform(-1, 1, M+4)
        d = np.random.uniform(1e-3, 1, M+3) #d>0
        R0 = np.random.uniform(-1, 1, 1)
    C = np.hstack((a,b,c,d,R0)) #union into one C
    return C

def coeff_generator_ansatz3_unconstrained(mode="initialize", C=None, M=None, prev_M=None, random=True, pwr=1e-1, const=1e-3, rand_lb=None, rand_ub=None):
    # accepts arbitrary lb and ub for randomization
    # 3 modes: generate C from previous C (append), multiply subvector with some constants (power), generate completely new C (initialize)
    # C is previous coefficient
    # const is positive
    # if C is None, then generate new vector by M, else if power is not none then it's v_J*const mode, otherwise, generate new C from prev C using prev_M
    # random  True if C is not none means the new constants will be randomized, otherwise const will be used
    
    # default lb and ub params:
    if rand_lb == None:
        rand_lb = 0
    if rand_ub == None:
        rand_ub = 1e-6
        
    if mode == "append":    
        #a:
        if random:
            const = np.random.uniform(rand_lb, rand_ub, 1)
        a = C[: prev_M]; a = np.hstack((a, const))
        #b:
        if random:
            const = np.random.uniform(rand_lb, rand_ub, 1)
        b = C[prev_M : 2*prev_M]; b = np.hstack((b, const))
        #c:
        if random:
            const = np.random.uniform(rand_lb, rand_ub, 1)
        c = C[2*prev_M : 3*prev_M+4]
        ins_idx = len(c)-1
        c = np.insert(c, ins_idx, const)
        #d:
        if random:
            const = np.random.uniform(rand_lb, rand_ub, 1)
        d = C[3*prev_M+4 : 4*prev_M+7]
        ins_idx = len(d)
        d = np.insert(d, ins_idx, const)
        #R0:
        R0 = C[-1]
    elif mode == "power":
        # only multiply the variable constants:
        idxes = [M-1, 2*M-1, 3*M+2, 4*M+6]
        C[idxes] *= pwr
    elif mode == "initialize":
        #generate init C_params:
        a = np.random.uniform(rand_lb, rand_ub, M)
        b = np.random.uniform(rand_lb, rand_ub, M) 
        c = np.random.uniform(rand_lb, rand_ub, M+4)
        d = np.random.uniform(rand_lb, rand_ub, M+3)
        R0 = np.random.uniform(rand_lb, rand_ub, 1)
    C = np.hstack((a,b,c,d,R0)) #union into one C
    return C


'''===========lmfit wrappers============'''

def lmfit_params_wrap(C, mode): #wrapper for lmfit parameters
    C_params = Parameters() #lmfit parameters
    if mode == "default":
        for i, c in enumerate(C):
            C_params.add(name="c"+str(i), value=c, min=-np.inf, max=np.inf)
    elif mode == "alternate":
        #vector with alternating sign:
        for i, c in enumerate(C):
            val = c
            if i%2 == 0:
                val = -c
            C_params.add(name="c"+str(i), value=val, min=-np.inf, max=np.inf)
    return C_params

def lmfit_params_wrap_ansatz2(C, itr=None):
    C_params = Parameters()
    M = int((len(C)-7)/4)
    for i in range(M): #a:
        C_params.add(name="c"+str(i), value=C[i], min=-np.inf, max=np.inf)
    for i in range(M, 2*M): #b:
        if i == M:
            C_params.add(name="c"+str(i), value=C[i], min=-np.inf, max=np.inf)
        else:
            C_params.add(name="c"+str(i), value=C[i], min=0, max=np.inf)
    for i in range(2*M, 3*M+4): #c:
        C_params.add(name="c"+str(i), value=C[i], min=-np.inf, max=np.inf)
    for i in range(3*M+4, 4*M+7): #d:
        C_params.add(name="c"+str(i), value=C[i], min=0, max=np.inf)
    
    return C_params

def lmfit_params_wrap_ansatz3(C, itr=None, mode="normal"):
    '''custom wrapper for ansatz3:
    mode="freeze": fixes parameters <= M-1 if M (itr) >= 1
    mode="normal": analogous to ansatz2, i.e., '''
    C_params = Parameters()
    M = int((len(C)-8)/4)
    if mode=="freeze":
        if itr == 1:
            for i in range(M): #a:
                C_params.add(name="c"+str(i), value=C[i], min=-np.inf, max=np.inf)
            for i in range(M, 2*M): #b > 0:
                C_params.add(name="c"+str(i), value=C[i], min=0, max=np.inf)
            for i in range(2*M, 3*M+4): #c:
                C_params.add(name="c"+str(i), value=C[i], min=-np.inf, max=np.inf)
            for i in range(3*M+4, 4*M+7): #d > 0:
                C_params.add(name="c"+str(i), value=C[i], min=0, max=np.inf)
            C_params.add(name="c"+str(4*M+7), value=C[4*M+7], min=-np.inf, max=np.inf) #R0
        else:
            #a:
            for i in range(M-1):
                C_params.add(name="c"+str(i), value=C[i], vary=False, min=-np.inf, max=np.inf)
            C_params.add(name="c"+str(M-1), value=C[M-1], vary=True, min=-np.inf, max=np.inf) #free the last one
            #b > 0:
            for i in range(M, 2*M-1):
                C_params.add(name="c"+str(i), value=C[i], vary=False, min=0, max=np.inf)
            C_params.add(name="c"+str(2*M-1), value=C[2*M-1], vary=True, min=0, max=np.inf) #free the last one
            #c:
            for i in range(2*M, 3*M+2):
                C_params.add(name="c"+str(i), value=C[i], vary=False, min=-np.inf, max=np.inf)
            C_params.add(name="c"+str(3*M+2), value=C[3*M+2], vary=True, min=-np.inf, max=np.inf) #free the last one
            #c_0:
            C_params.add(name="c"+str(3*M+3), value=C[3*M+3], vary=False, min=-np.inf, max=np.inf)
            #d > 0:
            for i in range(3*M+4, 4*M+6):
                C_params.add(name="c"+str(i), value=C[i], vary=False, min=0, max=np.inf)
            C_params.add(name="c"+str(4*M+6), value=C[4*M+6], vary=True, min=0, max=np.inf)
            #R0:
            C_params.add(name="c"+str(4*M+7), value=C[4*M+7], vary=False, min=-np.inf, max=np.inf)
    elif mode == "normal":
        for i in range(M): #a:
            C_params.add(name="c"+str(i), value=C[i], min=-np.inf, max=np.inf)
        for i in range(M, 2*M): #b > 0:
            C_params.add(name="c"+str(i), value=C[i], min=0, max=np.inf)
        for i in range(2*M, 3*M+4): #c:
            C_params.add(name="c"+str(i), value=C[i], min=-np.inf, max=np.inf)
        for i in range(3*M+4, 4*M+7): #d > 0:
            C_params.add(name="c"+str(i), value=C[i], min=0, max=np.inf)
        C_params.add(name="c"+str(4*M+7), value=C[4*M+7], min=-np.inf, max=np.inf) #R0
    return C_params

def lmfit_params_wrap_ansatz3_unconstrained(C, itr=None, mode="normal"):
    '''this is for unconstrained version'''
    C_params = Parameters()
    M = int((len(C)-8)/4)
    if mode=="freeze":
        if itr == 1: #custom condition, normally it shouldn't be here, for conveniences
            for i in range(M): #a:
                C_params.add(name="c"+str(i), value=C[i], min=-np.inf, max=np.inf)
            for i in range(M, 2*M): #b:
                C_params.add(name="c"+str(i), value=C[i], min=-np.inf, max=np.inf)
            for i in range(2*M, 3*M+4): #c:
                C_params.add(name="c"+str(i), value=C[i], min=-np.inf, max=np.inf)
            for i in range(3*M+4, 4*M+7): #d:
                C_params.add(name="c"+str(i), value=C[i], min=-np.inf, max=np.inf)
            C_params.add(name="c"+str(4*M+7), value=C[4*M+7], min=-np.inf, max=np.inf) #R0
        else:
            #a:
            for i in range(M-1):
                C_params.add(name="c"+str(i), value=C[i], vary=False, min=-np.inf, max=np.inf)
            C_params.add(name="c"+str(M-1), value=C[M-1], vary=True, min=-np.inf, max=np.inf) #free the last one
            #b > 0:
            for i in range(M, 2*M-1):
                C_params.add(name="c"+str(i), value=C[i], vary=False, min=-np.inf, max=np.inf)
            C_params.add(name="c"+str(2*M-1), value=C[2*M-1], vary=True, min=-np.inf, max=np.inf) #free the last one
            #c:
            for i in range(2*M, 3*M+2):
                C_params.add(name="c"+str(i), value=C[i], vary=False, min=-np.inf, max=np.inf)
            C_params.add(name="c"+str(3*M+2), value=C[3*M+2], vary=True, min=-np.inf, max=np.inf) #free the last one
            #c_0:
            C_params.add(name="c"+str(3*M+3), value=C[3*M+3], vary=False, min=-np.inf, max=np.inf)
            #d > 0:
            for i in range(3*M+4, 4*M+6):
                C_params.add(name="c"+str(i), value=C[i], vary=False, min=-np.inf, max=np.inf)
            C_params.add(name="c"+str(4*M+6), value=C[4*M+6], vary=True, min=-np.inf, max=np.inf)
            #R0:
            C_params.add(name="c"+str(4*M+7), value=C[4*M+7], vary=False, min=-np.inf, max=np.inf)
    elif mode == "normal":
        for i in range(M): #a:
            C_params.add(name="c"+str(i), value=C[i], min=-np.inf, max=np.inf)
        for i in range(M, 2*M): #b > 0:
            C_params.add(name="c"+str(i), value=C[i], min=-np.inf, max=np.inf)
        for i in range(2*M, 3*M+4): #c:
            C_params.add(name="c"+str(i), value=C[i], min=-np.inf, max=np.inf)
        for i in range(3*M+4, 4*M+7): #d > 0:
            C_params.add(name="c"+str(i), value=C[i], min=-np.inf, max=np.inf)
        C_params.add(name="c"+str(4*M+7), value=C[4*M+7], min=-np.inf, max=np.inf) #R0
    return C_params
    

'''===========multistart, multiple-start for local optimizers================'''

def multistart(n, delta, F, V, *F_args, len_C=100, C=None, 
               wrapper=None, wrapper_params=None, 
               coeff_generator=None, coeff_gen_params=None, 
               mode='default', verbose=False):
    #randomize by power x2 each loop and alternate sign:
    #n =  max loop
    #delta = minimum RMSE
    #if C is not None, then use C (in the format of lmfit params)
    #the data are global var
    pwr = 1
    min_rmse = np.inf
    #min_C = np.zeros(5)
    min_C = np.zeros(len_C)
    for k in range(n):
        #v2: the provided C is in the format of C_params :
        if (C is not None) and (k==0):
            C_params = C
        else:
            #coeff generator:
            if coeff_generator is not None:
                C0 = coeff_generator(*coeff_gen_params)
            else:
                C0 = np.random.uniform(-1, 1, len_C)*pwr
                
            #wrapper:
            if wrapper: #custom wrapper
                C_params = wrapper(C0, *wrapper_params)
            else:
                C_params = lmfit_params_wrap(C0, mode)
        while True: #NaN exception handler:
            try:
                #minimization routine and objective function here:
                out = minimize(f_obj_diatomic_pot_res_lmfit, C_params, args=(F, V, *F_args), method="bfgs", max_nfev=1e+10)
                break
            except ValueError:
                #reset C until no error:
                if verbose:
                    print("ValueError!!, resetting C")
                
                #coeff generator:
                if coeff_generator is not None:
                    C0 = coeff_generator(*coeff_gen_params)*pwr
                else:
                    C0 = np.random.uniform(-1, 1, len_C)*pwr
                
                #wrapper:
                if wrapper:
                    C_params = wrapper(C0, *wrapper_params)
                else:
                    C_params = lmfit_params_wrap(C0, mode)
                    
                continue
        #transform out.params to C array:
        C = np.array([out.params[key] for key in out.params])

        #get the predicted V
        V_pred = F(C, *F_args)
        rmse = RMSE(V_pred, V)
        #print(k, rmse)
        #get the minimum rmse:
        if rmse < min_rmse:
            min_rmse = rmse
            min_C = C
        #stop if delta is satisfied:
        if min_rmse <= delta:
            break
        #increase power or alternate sign
        if k%2 != 0:
            pwr *= 1e-1
        else:
            pwr *= -1 #alternate sign
    return min_rmse, min_C

def multiple_multistart(k, n, delta, F, V, *F_args, len_C=30, C=None, 
                        wrapper=None, wrapper_params=None, 
                        coeff_generator=None, coeff_gen_params=None, 
                        mode="default", verbose=False):
    #k = number of restarts
    min_rmse = np.inf; min_C = None
    for i in range(k):
        res = multistart(n, delta, F, V, *F_args, len_C=len_C, C=C, 
                         wrapper=wrapper, wrapper_params=wrapper_params, 
                         coeff_generator=coeff_generator, coeff_gen_params=coeff_gen_params, mode=mode)
        #print(i,"th round is done")
        rmse = res[0]; C_array = res[1];
        if rmse < min_rmse:
            min_rmse = rmse
            min_C = C_array
            print("RMSE = ",rmse)
        if rmse <= delta:
            break
    return min_rmse, min_C


'''=========performance evaluation================'''
def evaluate_efficiency(Fs, N, args, len_C):
    # function to evaluate the efficiency of a list of functions Fs by computing it N-numbers of time
    # args = list of arguments for each function
    times = []
    C = np.random.rand(len_C) #any vector works as long as it accomodates the minimium size
    for i, F in enumerate(Fs):
        #eval each function
        start = time.time()
        for j in range(N):
            F(C, *args[i])
        diff = time.time()-start
        times.append(diff)
    return times

def evaluate_efficiency_single(F, N, len_C, *arg):
    #single function only
    C = np.random.rand(len_C)
    start = time.time()
    for j in range(N):
        F(C, *arg)
    diff = time.time()-start
    return diff


if __name__ == "__main__":
    
    def test(a,b,c,d):
        print(a,b,c,d)
    params = (3,2,3,4)
    test(1,2,4,3,*params)