# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 14:35:09 2021

@author: Saint8312
"""

import numpy as np, pandas as pd, pickle, json
import PES_models as pmodel, PES_data_processor as pdata
#utilities:
import time, datetime, warnings

if __name__ == "__main__":
    def performance_comparison():
        '''compares the performance of each method'''
        '''
        performance evaluation per dataset contains:
        "mol": molecule data name
        "F": the functions list
        "F_names": the name of the functions
        "rmse": list of rmse 
        "time": list of time evaluation,
        "C_best": list of 1D array of best coefficients found,
        "M": polynomial degree
        "num_params": the number of parameters
        
        by varying the free parameters:
        CHIPR: 3(m+1) + M 
        ansatz_0: 3M+1
        ansatz_1: 4M
        M is common parameter, hence it is used as the reference,
        m = 2M-2/3, where 2M-2%3 = 0 for ansatz_0; m = M-1 for ansatz_1
        '''
        list_data = np.load('data/hxoy_data.npy', allow_pickle=True)
        list_data = list_data[()]
        
        mol = "OH+"
        qidxes = pdata.query_one_var_indices(mol, "mol", list_data) #pick one
        R = list_data[qidxes[1]]["R"]; V = list_data[qidxes[1]]["V"]
        print(R, V)
        
        
        Fs = [pmodel.f_diatomic_ansatz_1, pmodel.f_diatomic_chipr_ohplus]
        F_names = ["ansatz_1","CHIPR"]
        
        
        loop = int(1e4); n = 20; # # of loops per run and evaluation runs
        restarts = int(20); powers = int(5); # number of optimization restarts and powers for random number generations
        delta = 1e-5 #error tolerance to stop iterations

        '''
        loop = 1; n = 1; restarts = 1; powers  = 1
        delta = 1e-5 #error tolerance to stop iterations
        '''
        #physical params:
        Z = 8 #for OH or its ions
        
        data = {}
        data["num_params"] = []
        data["eval_loop"] = loop; data["eval_run"] = n;
        data["opt_restart"] = restarts; data["opt_power"] = powers; data["opt_delta"] = delta
        data["chipr_t"] = []; data["chipr_acc"] = []; data["chipr_C"] = []
        #data["ansatz_0_t"] = []; data["ansatz_0_acc"] = []; data["ansatz_0_C"] = []
        data["ansatz_1_t"] = []; data["ansatz_1_acc"] = []; data["ansatz_1_C"] = []
        data["degree"] = []
        
        max_deg = 30 #maximum polynomial degree
        init_time = time.time() #timer
        with warnings.catch_warnings(record=True): #required, otherwise the program will abruptly stops!
            for M in range(2, max_deg):
                #if (2*M - 2)%3 == 0: # must satisfy this
                    #m = int((2*M - 2)/3)
                    #ansatz_par = 3*M+1
                if M%4 == 0: #multiple of 4 only 
                    m = M - 1 #for new ansatz
                    ansatz_par = 4*M
                    #chipr_par = 3*(m+1)+M;
                    print("===========================")
                    print("M = ",M, ", m (chipr) =",m)
                    print("parameters = ",ansatz_par)
                    
                    #Time evaluation:
                    print(">>> Time evaluation:")
                    args = [(R,Z,M), (R,Z,M,m)] 
                    times_array = np.zeros(len(Fs))
                    len_C = ansatz_par #coef length, min(len) = 3M+1
                    for i in range(n):
                        times_array += np.array(pmodel.evaluate_efficiency(Fs, loop, args, len_C))
                    mean_t = times_array/n
                    print("evaluation on",loop,"runs")
                    print("functions",F_names)
                    print("running times averaged over",n,"runs",mean_t)
                    
                    #Accuracy evaluation:
                    print(">>> Accuracy evaluation:")
                    rmses = []; Cs = []
                    for i, f in enumerate(Fs):
                        rmse, C = pmodel.multiple_multistart(restarts, powers, delta, f, V, *args[i], len_C=len_C, mode="default")
                        rmses.append(rmse)
                        Cs.append(C)
                        print("rmse = ",rmse)
                        
                    #append to data:
                    data["num_params"].append(len_C); data["degree"].append(M)
                    data["ansatz_1_t"].append(mean_t[0]); data["chipr_t"].append(mean_t[1])
                    data["ansatz_1_acc"].append(rmses[0]); data["chipr_acc"].append(rmses[1])
                    data["ansatz_1_C"].append(Cs[0]); data["chipr_C"].append(Cs[1])
            
            end_time = time.time() #timer
            elapsed = end_time-init_time
            data["simulation_time"] = elapsed
            print("elapsed time =",elapsed,"s")
            print(data)
            #write to pandas, then to file:
            filename = "result/performance_"+datetime.datetime.now().strftime('%d%m%Y')+".pkl"
            with open(filename, 'wb') as handle:
                pickle.dump(data, handle)
    
    def joint_fit():
        '''joint fit for OH+'''
        list_data = np.load('data/hxoy_data.npy', allow_pickle=True)
        list_data = list_data[()]
        
        #union the data of OH+ mol:
        mol = "OH+"
        qidxes = pdata.query_one_var_indices(mol, "mol", list_data)
        #test on idx 1, train on idx 4:
        #print(list_data[qidxs])
        
        R_list = []; V_list = []
        for i in qidxes:
            R = list_data[i]["R"]
            V = list_data[i]["V"]
            R_list.append(R); V_list.append(V)
            
        R_list = np.array(R_list); V_list = np.array(V_list)
        R = np.concatenate(R_list)
        V = np.concatenate(V_list)

        
        Fs = [pmodel.f_diatomic_ansatz_1, pmodel.f_diatomic_chipr_ohplus]
        F_names = ["ansatz_1", "CHIPR"]
        #M = 7; m = 4;
        #par = 3*M+1
        M = 10; m = 9;
        par = 4*M
        #len_Cs = [par, par, 5, 8] #number of free parameters
        len_Cs = [par, par]
        Z = 8
        #args = [(R,Z,M), (R,Z,M,m), (R,), (R,)]
        args = [(R,Z,M), (R,Z,M,m)]
        rmses = []; Cs = []
        restarts = 10; powers = 5; delta = 1e-5
        with warnings.catch_warnings(record=True): #CHIPR NaN problem
            for i, f in enumerate(Fs):
                len_C = len_Cs[i]
                rmse, C = pmodel.multiple_multistart(restarts, powers, delta, f, V, *args[i], len_C=len_C, mode="default")
                rmses.append(rmse); Cs.append(C)
                print(F_names[i], "rmse = ",rmse)

        
        data = {}
        data["num_params"] = len_C #obj params
        data["opt_restart"] = restarts; data["opt_power"] = powers; data["opt_delta"] = delta #opt params
        data["mol"] = mol; #dataset descriptor
        data["ansatz_1_acc"] = rmses[0]; data["ansatz_1_C"] = Cs[0]
        #data["ansatz_acc"] = rmses[0]; data["ansatz_C"] = Cs[0]
        data["chipr_acc"] = rmses[1]; data["chipr_C"] = Cs[1]
        #data["dn_acc"] = rmses[2]; data["dn_C"] = Cs[2]
        #data["ds_acc"] = rmses[3]; data["ds_C"] = Cs[3]
        print(data)
        
        '''
        # pandas is less reliable for cross version!
        #df = pd.DataFrame.from_dict(data)
        df = pd.DataFrame.from_dict(data, orient='index')
        df = df.transpose() #because the lengths of C are not the same
        df.to_pickle("result/res_joint_"+mol+"_"+datetime.datetime.now().strftime('%d%m%y_%H%M%S')+".pkl")
        '''
        filename = "result/res_joint_"+mol+"_"+datetime.datetime.now().strftime('%d%m%y_%H%M%S')+".pkl"
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle)
        
    def joint_fit_solo():
        '''joint fit for OH+, solo for faster simulation time, for checking one formula/ansatz'''
        list_data = np.load('data/hxoy_data.npy', allow_pickle=True)
        list_data = list_data[()]
        
        #union the data of OH+ mol:
        mol = "OH+"
        qidxes = pdata.query_one_var_indices(mol, "mol", list_data)
        #test on idx 1, train on idx 4:
        #print(list_data[qidxs])
        
        R_list = []; V_list = []
        for i in qidxes:
            R = list_data[i]["R"]
            V = list_data[i]["V"]
            R_list.append(R); V_list.append(V)
            
        R_list = np.array(R_list); V_list = np.array(V_list)
        R = np.concatenate(R_list)
        V = np.concatenate(V_list)

        
        F = pmodel.f_diatomic_ansatz_1
        F_name = "ansatz_1"
        Z = 8
        restarts = 10; powers = 5; delta = 1e-5
        
        data = {}
        data["ansatz_1_acc"] = []; data["ansatz_1_C"] = []
        data["num_params"] = []; data["degree"] = []
        data["opt_restart"] = restarts; data["opt_power"] = powers; data["opt_delta"] = delta #opt params
        data["mol"] = mol; #dataset descriptor
        with warnings.catch_warnings(record=True): #CHIPR NaN problem
            for M in range(1, 7):
                print("M = ",M)
                len_C = 4*M
                arg = (R,Z,M)
                rmse, C = pmodel.multiple_multistart(restarts, powers, delta, F, V, *arg, len_C=len_C, mode="default")
                print(F_name, "rmse = ",rmse)
                
                data["num_params"].append(len_C); data["degree"].append(M)
                data["ansatz_1_acc"].append(rmse);
                data["ansatz_1_C"].append(C);
        print(data)
        
        '''
        # pandas is less reliable for cross version!
        #df = pd.DataFrame.from_dict(data)
        df = pd.DataFrame.from_dict(data, orient='index')
        df = df.transpose() #because the lengths of C are not the same
        df.to_pickle("result/res_joint_"+mol+"_"+datetime.datetime.now().strftime('%d%m%y_%H%M%S')+".pkl")
        '''
        filename = "result/res_joint_solo_"+mol+"_"+datetime.datetime.now().strftime('%d%m%y_%H%M%S')+".pkl"
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle)
        
    def each_state_fit():    
        #separated fit for each state
        list_data = np.load('data/hxoy_data.npy', allow_pickle=True)
        list_data = list_data[()]
        
        #getdata:
        mols = ["OH+", "H2", "H2+", "O2", "O2+", "OH"]
        Zs = {"OH+":8, "H2":1, "H2+":1, "O2":64, "O2+":64, "OH":8}
        #mols = ["H2+"]
        dict_list = [{'mol':mol} for mol in mols]
        qidxes = pdata.many_queries_many_vars_indices(dict_list, list_data)
        rel_datasets = list_data[qidxes]
        print(rel_datasets)
        
        #model training:
        #Fs = [pmodel.f_diatomic_vdw, pmodel.f_diatomic_chipr_ohplus]
        Fs = [pmodel.f_diatomic_vdw, pmodel.f_diatomic_chipr_ohplus, pmodel.f_diatomic_dn, pmodel.f_diatomic_ds]
        F_names = ["ansatz", "CHIPR", "Deiters-Neumaier", "Deiters-Sadus"]
        M = 7; m = 4;
        par = 3*M+1
        len_Cs = [par, par, 5, 8] #number of free parameters
        Z = 8 #change Z for each molecule!!
        restarts = 10; powers = 3; delta = 1e-5
        
        data = {}
        data["num_params"] = [] #only for variable free params, less relevant for fixed params e.g. f-dn or f-ds
        data["opt_restart"] = restarts; data["opt_power"] = powers; data["opt_delta"] = delta
        data["mol"] = []; data["state"] = []; data["author"] = []; data["method"] = []; data["Z"] = [] #dataset descriptor
        data["ansatz_acc"] = []; data["ansatz_C"] = []
        data["chipr_acc"] = []; data["chipr_C"] = []
        data["dn_acc"] = []; data["dn_C"] = []
        data["ds_acc"] = []; data["ds_C"] = []
        with warnings.catch_warnings(record=True): #CHIPR NaN problem
            for dset in rel_datasets:
                print("dataset = ", dset["mol"], dset["state"], dset["author"])
                data["mol"].append(dset["mol"]); data["state"].append(dset["state"]); data["author"].append(dset["author"]);
                if "method" in dset:
                    data["method"].append(dset["method"])
                else:
                    data["method"].append(None)
                Z = Zs[dset["mol"]] #assign Z
                V = dset["V"]; R = dset["R"]
                args = [(R,Z,M), (R,Z,M,m), (R,), (R,)]
                rmses = []; Cs = []
                for i, f in enumerate(Fs):
                    len_C = len_Cs[i]
                    rmse, C = pmodel.multiple_multistart(restarts, powers, delta, f, V, *args[i], len_C=len_C, mode="default")
                    rmses.append(rmse); Cs.append(C)
                    print(F_names[i], "rmse = ",rmse)
                    #print(rmses)
                    #print(Cs)
                    data["num_params"].append(len_C)
                data["ansatz_acc"].append(rmses[0]); data["chipr_acc"].append(rmses[1]); data["dn_acc"].append(rmses[2]); data["ds_acc"].append(rmses[3])
                data["ansatz_C"].append(Cs[0]); data["chipr_C"].append(Cs[1]); data["dn_C"].append(Cs[2]); data["ds_C"].append(Cs[3])
                data["Z"].append(Z)
                
        print(data)
        '''
        #df = pd.DataFrame.from_dict(data)
        df = pd.DataFrame.from_dict(data, orient='index') #if the lengths of C are not equal
        df = df.transpose() #because the lengths of C are not the same
        df.to_pickle("result/res_each_state_"+datetime.datetime.now().strftime('%d%m%y_%H%M%S')+".pkl")
        '''
        filename = "result/res_each_state_"+datetime.datetime.now().strftime('%d%m%y_%H%M%S')+".pkl"
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle)
        
        
    '''end of main functions, actual main starts below'''
    #performance_comparison()
    #joint_fit()
    joint_fit_solo()
    