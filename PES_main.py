# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 14:35:09 2021

@author: Saint8312
"""

import numpy as np, pandas as pd
import nbimporter #ipython importer
import PES_models as pmodel, PES_data_processor as pdata
#utilities:
import time, datetime, warnings

if __name__ == "__main__":
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

        
        Fs = [pmodel.f_diatomic_vdw, pmodel.f_diatomic_chipr_ohplus, pmodel.f_diatomic_dn, pmodel.f_diatomic_ds]
        F_names = ["ansatz", "CHIPR", "Deiters-Neumaier", "Deiters-Sadus"]
        M = 7; m = 4;
        par = 3*M+1
        len_Cs = [par, par, 5, 8] #number of free parameters
        Z = 8
        restarts = 10; powers = 3; delta = 1e-5
        args = [(R,Z,M), (R,Z,M,m), (R,), (R,)]
        rmses = []; Cs = []
        for i, f in enumerate(Fs):
            len_C = len_Cs[i]
            rmse, C = pmodel.multiple_multistart(restarts, powers, delta, f, V, *args[i], len_C=len_C, mode="default")
            rmses.append(rmse); Cs.append(C)
            print(F_names[i], "rmse = ",rmse)

        
        data = {}
        data["num_params"] = len_C #obj params
        data["opt_restart"] = restarts; data["opt_power"] = powers; data["opt_delta"] = delta #opt params
        data["mol"] = mol; #dataset descriptor
        data["ansatz_acc"] = rmses[0]; data["ansatz_C"] = Cs[0]
        data["chipr_acc"] = rmses[1]; data["chipr_C"] = Cs[1]
        data["dn_acc"] = rmses[2]; data["dn_C"] = Cs[2]
        data["ds_acc"] = rmses[3]; data["ds_C"] = Cs[3]
        print(data)
        
        #df = pd.DataFrame.from_dict(data)
        df = pd.DataFrame.from_dict(data, orient='index')
        df = df.transpose() #because the lengths of C are not the same
        df.to_pickle("result/res_joint_"+mol+"_"+datetime.datetime.now().strftime('%d%m%y_%H%M%S')+".pkl")
        
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
        #df = pd.DataFrame.from_dict(data)
        df = pd.DataFrame.from_dict(data, orient='index') #if the lengths of C are not equal
        df = df.transpose() #because the lengths of C are not the same
        df.to_pickle("result/res_each_state_"+datetime.datetime.now().strftime('%d%m%y_%H%M%S')+".pkl")
        
    '''end of main functions, actual main starts below'''
    #joint_fit()
    each_state_fit()