# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 14:35:09 2021

@author: Saint8312
"""

import numpy as np, pandas as pd, pickle, json
import PES_models as pmodel, PES_data_processor as pdata

#opt:
from lmfit import Parameters, minimize
from lmfit.printfuncs import report_fit

#utilities:
import time, datetime, warnings

if __name__ == "__main__":
    '''========time eval only:========='''
    def time_eval():
        '''evaluates the efficiency of a single function'''
        list_data = np.load('data/hxoy_data.npy', allow_pickle=True)
        list_data = list_data[()]
        
        mol = "OH+"
        qidxes = pdata.query_one_var_indices(mol, "mol", list_data) #pick one
        R = list_data[qidxes[1]]["R"]; V = list_data[qidxes[1]]["V"]
        
        #F = pmodel.f_diatomic_ansatz_2
        #F_name = "ansatz2"
        
        F = pmodel.f_diatomic_chipr_ohplus
        F_name = "chipr"
        
        loop = int(1e4); n = 20;
        
        data = {}
        data["F_name"] = F_name
        data["num_params"] = []; data["degree"] = []
        data["times"] = []
        
        max_M = 20
        for M in range(1, max_M+1): # include the last one in this case      
            # Time evaluation:
            print(">>> Time evaluation:")
            print("M = ", M)
            # for ansatz2:
            #arg = (R,M) 
            #len_C = 4*M+7
            
            # for chipr:
            m = M - 1 #for chipr only
            len_C = 3*(m+1)+M;
            arg = (R,8,M,m)
            
            print("num_params = ",len_C)

            time = 0
            for i in range(n):
                time += pmodel.evaluate_efficiency_single(F, loop, len_C, *arg)
            mean_t = time/n
            print("evaluation on",loop,"runs")
            print("functions",F_name)
            print("running times averaged over",n,"runs",mean_t)
            
            data["num_params"].append(len_C); data["degree"].append(M)
            data["times"].append(mean_t)
        
        print(data)
        filename = "result/time_eval_"+F_name+"_"+datetime.datetime.now().strftime('%d%m%y_%H%M%S')+".pkl"
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle)
         
    
    '''=======eof time eval========='''
    
    def special_split_fit():
        '''only for acc of the non-Coulomb ansatz'''
        from sklearn.model_selection import train_test_split
        
        method = "multirestart" # "direct" or "multirestart" for v2 and v3 respectively
        
        list_data = np.load('data/hxoy_data.npy', allow_pickle=True)
        list_data = list_data[()]
        
        mol = "OH+"
        qidxes = pdata.query_one_var_indices(mol, "mol", list_data) #pick one
        R = list_data[qidxes[1]]["R"]; V = list_data[qidxes[1]]["V"]
        
        test_size = 0.25
        R_train, R_test, V_train, V_test = train_test_split(R, V, test_size=test_size, random_state=0) #split data

        F = pmodel.f_diatomic_ansatz_2
        F_name = "ansatz_2"
        
        restarts = int(10); powers = int(3); # number of optimization restarts and powers for random number generations
        delta = 1e-5
        
        #no Z
        data = {}
        data["method"] = method
        data["test_size"] = test_size
        data["num_params"] = []
        data["opt_restart"] = restarts; data["opt_power"] = powers; data["opt_delta"] = delta
        data["ansatz_2_acc_train"] = []; data["ansatz_2_acc_test"] = []; data["ansatz_2_C"] = []
        data["degree"] = []
        
        max_M = 20
        init_time = time.time() #timer
        C = None; prev_M = None #placeholder
        for M in range(1, max_M+1): #include the last one in this case            
            print(">>>>> M =",M)
            arg_train = (R_train,M)
            arg_test = (R_test,M)
            
            len_C = 4*M+7 #only for special case: non-coulomb pair pot  
            
            #v2: set params here and uses standard opt method:
            #Accuracy evaluation:
            print(">>> Accuracy evaluation:")
            #special augmentation for M >= 2: a_m = b_m = c_m = d_m = 0
            if M >= 2:
                #use previous C with augmentation:
                const = 0
                a = C[: prev_M]; a = np.hstack((a, const))
                b = C[prev_M : 2*prev_M]; b = np.hstack((b, const))
                c = C[2*prev_M : 3*prev_M+4]; c = np.insert(c, M-2, const) #insert into the M index
                d = C[3*prev_M+4 : 4*prev_M+7]; d = np.insert(d, M-2, const) #insert into the M index
                
                C = np.hstack((a,b,c,d)) #union into one C
                
                #RMSE pre-check:
                V_pred_train = F(C, *arg_train)
                rmse_pc_train = pmodel.RMSE(V_pred_train, V_train)
                
                V_pred_test = F(C, *arg_test)
                rmse_pc_test = pmodel.RMSE(V_pred_test, V_test)
                print("RMSE pre-check = ", rmse_pc_train, rmse_pc_test)
                print(C)
                
            else:
                #v2:
                #generate init C_params:
                C = np.random.uniform(-1, 1, len_C)
            
            # the non zero conditions causes any x < 0 -> x = 0 !!, this lowers RMSE, will need to use lmfit custom wrapper for non coulomb!!
            C_params = pmodel.lmfit_params_wrap_ansatz2(C)
                
            #RMSE 2nd pre-check:
            C = np.array([C_params[key] for key in C_params])
            V_pred_train = F(C, *arg_train)
            rmse_pc_train = pmodel.RMSE(V_pred_train, V_train)
            
            V_pred_test = F(C, *arg_test)
            rmse_pc_test = pmodel.RMSE(V_pred_test, V_test)
            print("RMSE 2nd pre-check = ", rmse_pc_train, rmse_pc_test)
            print(C)
            
            
            #v3: multirestart with definition of C_params outside
            #for ansatz2:
            rmse_train, C = pmodel.multiple_multistart(restarts, powers, delta, F, V_train, *arg_train, len_C=len_C, C=C_params, wrapper=pmodel.lmfit_params_wrap_ansatz2, mode="default")
            
            
            '''
            #v2:
            out = minimize(pmodel.f_obj_diatomic_pot_res_lmfit, C_params, args=(F, V_train, *arg_train), method="bfgs")
            
            C = np.array([out.params[key] for key in out.params]) #reconstruct c
            #get the prediction from training data:
            V_pred = F(C, *arg_train)
            rmse_train = pmodel.RMSE(V_pred, V_train)
            # end of v2
            '''
            
            print("rmse_train =",rmse_train)

            #get test pred:
            V_pred = F(C, *arg_test)
            rmse_test = pmodel.RMSE(V_pred, V_test)
            print("rmse_test =",rmse_test)
            
            #append to data:
            data["num_params"].append(len_C); data["degree"].append(M)
            data["ansatz_2_acc_train"].append(rmse_train); #train
            data["ansatz_2_acc_test"].append(rmse_test); #test
            data["ansatz_2_C"].append(C); 
            
            #store M for next iter:
            prev_M = M
        
        end_time = time.time() #timer
        elapsed = end_time-init_time
        data["simulation_time"] = elapsed
        print("elapsed time =",elapsed,"s")
        print(data)
        
        filename = "result/spec_split_data_fit_"+method+"_"+mol+"_"+datetime.datetime.now().strftime('%d%m%y_%H%M%S')+".pkl"
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle)
    
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
        restarts = int(10); powers = int(3); # number of optimization restarts and powers for random number generations
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
                    
                    '''
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
                    '''
                    
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
                    #data["ansatz_1_t"].append(mean_t[0]); data["ansatz_1_acc"].append(rmses[0]); data["ansatz_1_C"].append(Cs[0]); 
                    #data["chipr_t"].append(mean_t[1]); 
                    data["chipr_acc"].append(rmses[1]); data["chipr_C"].append(Cs[1])
                    
            end_time = time.time() #timer
            elapsed = end_time-init_time
            data["simulation_time"] = elapsed
            print("elapsed time =",elapsed,"s")
            print(data)
            #write to pandas, then to file:
            filename = "result/performance_"+datetime.datetime.now().strftime('%d%m%Y')+".pkl"
            with open(filename, 'wb') as handle:
                pickle.dump(data, handle)
    
    
    def split_data_fit_performance():
        '''split the data for training and testing'''
        from sklearn.model_selection import train_test_split
        
        list_data = np.load('data/hxoy_data.npy', allow_pickle=True)
        list_data = list_data[()]
        
        mol = "OH+"
        qidxes = pdata.query_one_var_indices(mol, "mol", list_data) #pick one
        R = list_data[qidxes[1]]["R"]; V = list_data[qidxes[1]]["V"]

        R_train, R_test, V_train, V_test = train_test_split(R, V, test_size=0.25, random_state=0) #split data

        #Fs = [pmodel.f_diatomic_ansatz_1, pmodel.f_diatomic_chipr_ohplus]
        Fs = [pmodel.f_diatomic_chipr_ohplus]
        F_names = ["ansatz_1","CHIPR"]
        
        restarts = int(10); powers = int(3); # number of optimization restarts and powers for random number generations
        delta = 1e-5 #error tolerance to stop iterations

        #physical params:
        Z = 8 #for OH or its ions
        
        data = {}
        data["num_params"] = []
        data["opt_restart"] = restarts; data["opt_power"] = powers; data["opt_delta"] = delta
        data["chipr_t"] = []; data["chipr_acc_train"] = []; data["chipr_acc_test"] = []; data["chipr_C"] = []
        data["ansatz_1_t"] = []; data["ansatz_1_acc_train"] = []; data["ansatz_1_acc_test"] = []; data["ansatz_1_C"] = []
        data["degree"] = []
        
        max_deg = 22 #maximum polynomial degree
        init_time = time.time() #timer
        with warnings.catch_warnings(record=True): #required, otherwise the program will abruptly stops!
            for M in range(1, max_deg+1):
                '''for ansatz 0:'''
                #if (2*M - 2)%3 == 0: # must satisfy this
                    #m = int((2*M - 2)/3)
                    #ansatz_par = 3*M+1
                '''for ansatz 1:'''
                #if M%4 == 0: #multiple of 4 only 
                m = M - 1 #for new ansatz
                ansatz_par = 4*M
                chipr_par = 3*(m+1)+M;
                print("===========================")
                print("M = ",M, ", m (chipr) =",m)
                print("parameters = ",ansatz_par)
                
                #args = [(R,Z,M), (R,Z,M,m)] 
                #for both F test:
                #args_train = [(R_train,Z,M), (R_train,Z,M,m)] 
                #args_test = [(R_test,Z,M), (R_test,Z,M,m)]
                
                args_train = [(R_train,Z,M,m)]
                args_test = [(R_test,Z,M,m)]
                
                len_C = ansatz_par #coef length, min(len) = 3M+1
                print("functions",F_names)
                #Accuracy evaluation:
                print(">>> Accuracy evaluation:")
                rmses_train = []; rmses_test = []; Cs = []
                for i, f in enumerate(Fs):
                    rmse_train, C = pmodel.multiple_multistart(restarts, powers, delta, f, V_train, *args_train[i], len_C=len_C, mode="default")
                    V_pred = f(C, *args_test[i])
                    rmse_test = pmodel.RMSE(V_test, V_pred)
                    rmses_train.append(rmse_train); rmses_test.append(rmse_test)
                    Cs.append(C)
                    print("rmse test = ",rmse_test)
                    
                #append to data:
                data["num_params"].append(len_C); data["degree"].append(M)
                #data["ansatz_1_acc_train"].append(rmses_train[0]); data["ansatz_1_acc_test"].append(rmses_test[0]); data["ansatz_1_C"].append(Cs[0]); 
                data["chipr_acc_train"].append(rmses_train[0]); data["chipr_acc_test"].append(rmses_test[0]); data["chipr_C"].append(Cs[0]) #shift one if only one is tested
            
            end_time = time.time() #timer
            elapsed = end_time-init_time
            data["simulation_time"] = elapsed
            print("elapsed time =",elapsed,"s")
            print(data)
            filename = "result/split_data_fit_OH+"+datetime.datetime.now().strftime('%d%m%y_%H%M%S')+".pkl"
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
        filename = "result/res_each_state_"+datetime.datetime.now().strftime('%d%m%y_%H%M%S')+".pkl"
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle)


    '''===== cross-val ver beyond this point !!: ======'''

    def special_cross_val_fit():
        '''cross validation only for the non-Coulomb ansatz only for acc'''
        from sklearn.model_selection import KFold
        
        list_data = np.load('data/hxoy_data.npy', allow_pickle=True)
        list_data = list_data[()]
        
        mol = "OH+"
        qidxes = pdata.query_one_var_indices(mol, "mol", list_data) #pick one
        R = list_data[qidxes[1]]["R"]; V = list_data[qidxes[1]]["V"]
        
        num_fold = 4
        kf = KFold(n_splits=num_fold, random_state=0, shuffle=True) #num of testing data = 1/split*len(data)
        
        F = pmodel.f_diatomic_ansatz_2
        F_name = "ansatz_2"
        
        restarts = int(10); powers = int(3); # number of optimization restarts and powers for random number generations
        delta = 1e-5
        
        #no Z
        data = {}
        data["fold"] = num_fold
        data["num_params"] = []
        data["opt_restart"] = restarts; data["opt_power"] = powers; data["opt_delta"] = delta
        data["ansatz_2_acc_train"] = []; data["ansatz_2_acc_test"] = []; data["ansatz_2_C"] = []
        data["degree"] = []
        
        max_M = 20
        for M in range(1, max_M+1): #include the last one in this case
            min_train_rmse = np.inf; min_test_rmse = np.inf; min_C = None
            fold = 0
            for train_index, test_index in kf.split(R):
                print(">>> fold = ",fold)
                fold+=1
                R_train, R_test = R[train_index], R[test_index]
                V_train, V_test = V[train_index], V[test_index]
                
                arg_train = (R_train,M)
                arg_test = (R_test,M)
                
                len_C = 4*M+7 #only for special case: non-coulomb pair pot
        
                print("functions",F_names)
                
                #Accuracy evaluation:
                print(">>> Accuracy evaluation:")
                #special augmentation for M >= 2: C[M-1] = C[2M-1] = C[3M-2] = C[4M+2] = 0
                if M >= 2:
                     C[M-1] = C[2*M-1] = C[3*M-2] = C[4*M+2] = 0
                     rmse_train, C = pmodel.multiple_multistart(restarts, powers, delta, F, V_train, *arg_train, len_C=len_C, C=C, mode="default")
                else:
                    rmse_train, C = pmodel.multiple_multistart(restarts, powers, delta, F, V_train, *arg_train, len_C=len_C, mode="default")
                        
                if rmse_test < min_test_rmse:
                    min_test_rmse = rmse_test
                    min_train_rmse = rmse_train
                    min_C = C
                    
                
                #append to data:
                data["num_params"].append(len_C); data["degree"].append(M)
                data["ansatz_2_acc_train"].append(min_train_rmse); #train
                data["ansatz_2_acc_test"].append(min_test_rmse); #test
                data["ansatz_2_C"].append(min_C); 
        
                
                
    def cross_val_performance_fit():
        '''cross validation fit for one dataset, pick the best model'''
        from sklearn.model_selection import KFold
        
        list_data = np.load('data/hxoy_data.npy', allow_pickle=True)
        list_data = list_data[()]
        
        mol = "OH+"
        qidxes = pdata.query_one_var_indices(mol, "mol", list_data) #pick one
        R = list_data[qidxes[1]]["R"]; V = list_data[qidxes[1]]["V"]
        
        num_fold = 4
        kf = KFold(n_splits=num_fold, random_state=0, shuffle=True) #num of testing data = 1/split*len(data)
            
        Fs = [pmodel.f_diatomic_ansatz_0, pmodel.f_diatomic_chipr_ohplus]
        F_names = ["ansatz_0","CHIPR"]
        
        restarts = int(10); powers = int(3); # number of optimization restarts and powers for random number generations
        delta = 1e-5 #error tolerance to stop iterations

        #physical params:
        Z = 8 #for OH or its ions
        
        data = {}
        data["fold"] = num_fold
        data["num_params"] = []
        data["opt_restart"] = restarts; data["opt_power"] = powers; data["opt_delta"] = delta
        data["chipr_t"] = []; data["chipr_acc_train"] = []; data["chipr_acc_test"] = []; data["chipr_C"] = []
        data["ansatz_0_t"] = []; data["ansatz_0_acc_train"] = []; data["ansatz_0_acc_test"] = []; data["ansatz_0_C"] = []
        data["degree"] = []
    
        max_deg = 30 #maximum polynomial degree
        init_time = time.time() #timer
        with warnings.catch_warnings(record=True): #required, otherwise the program will abruptly stops!
            for M in range(2, max_deg):
                '''for ansatz 0:'''
                if (2*M - 2)%3 == 0: # must satisfy this
                    m = int((2*M - 2)/3)
                    ansatz_par = 3*M+1
                    '''for ansatz 1:'''
                    #if M%4 == 0: #multiple of 4 only 
                        #m = M - 1 #for new ansatz
                        #ansatz_par = 4*M
                        #chipr_par = 3*(m+1)+M;
                    print("===========================")
                    print("M = ",M, ", m (chipr) =",m)
                    print("parameters = ",ansatz_par)
                    
                    min_train_rmses = [np.inf, np.inf]; min_test_rmses = [np.inf, np.inf]; min_Cs = [None, None]
                    fold = 0
                    for train_index, test_index in kf.split(R):
                        print(">>> fold = ",fold)
                        fold+=1
                        R_train, R_test = R[train_index], R[test_index]
                        V_train, V_test = V[train_index], V[test_index]
                        
                        args_train = [(R_train,Z,M), (R_train,Z,M,m)] 
                        args_test = [(R_test,Z,M), (R_test,Z,M,m)]
                        len_C = ansatz_par #coef length, min(len) = 3M+1
                        print("functions",F_names)
                        #Accuracy evaluation:
                        print(">>> Accuracy evaluation:")
                        rmses_train = []; rmses_test = []; Cs = []
                        for i, f in enumerate(Fs):
                            rmse_train, C = pmodel.multiple_multistart(restarts, powers, delta, f, V_train, *args_train[i], len_C=len_C, mode="default")
                            V_pred = f(C, *args_test[i])
                            rmse_test = pmodel.RMSE(V_test, V_pred)
                                
                            rmses_train.append(rmse_train); rmses_test.append(rmse_test)
                            Cs.append(C)
                            print("rmse test = ",rmse_test)
                            
                        #get min test rmse:
                        for i in range(len(min_test_rmses)):
                            if rmses_test[i] < min_test_rmses[i]:
                                min_test_rmses[i] = rmses_test[i]
                                min_train_rmses[i] = rmses_train[i]
                                min_Cs[i] = Cs[i]
                                
                    print("picked rmses = ",min_test_rmses)
                        
                    #append to data:
                    data["num_params"].append(len_C); data["degree"].append(M)
                    data["ansatz_0_acc_train"].append(min_train_rmses[0]); data["chipr_acc_train"].append(min_train_rmses[1]) #train
                    data["ansatz_0_acc_test"].append(min_test_rmses[0]); data["chipr_acc_test"].append(min_test_rmses[1]) #test
                    data["ansatz_0_C"].append(min_Cs[0]); data["chipr_C"].append(min_Cs[1])
            
            end_time = time.time() #timer
            elapsed = end_time-init_time
            data["simulation_time"] = elapsed
            print("elapsed time =",elapsed,"s")
            print(data)
            #write to pandas, then to file:
            filename = "result/cross_val_fit_"+mol+"_"+datetime.datetime.now().strftime('%d%m%Y')+".pkl"
            with open(filename, 'wb') as handle:
                pickle.dump(data, handle)
                
    def cross_val_each_state_fit():    
        ''' cross validation ver - separated fit for each state, only captures test rmse '''
        from sklearn.model_selection import KFold
        
        list_data = np.load('data/hxoy_data.npy', allow_pickle=True)
        list_data = list_data[()]
        
        num_fold = 4
        kf = KFold(n_splits=num_fold, random_state=0, shuffle=True) #num of testing data = 1/split*len(data)
        
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
        Fs = [pmodel.f_diatomic_ansatz_0, pmodel.f_diatomic_chipr_ohplus]
        F_names = ["ansatz", "CHIPR"]
        M = 7; m = 4;
        par = 3*M+1
        len_Cs = [par, par] #number of free parameters
        Z = 8 #change Z for each molecule!!
        restarts = 10; powers = 3; delta = 1e-5
        
        data = {}
        data["num_params"] = [] #only for variable free params, less relevant for fixed params e.g. f-dn or f-ds
        data["opt_restart"] = restarts; data["opt_power"] = powers; data["opt_delta"] = delta
        data["mol"] = []; data["state"] = []; data["author"] = []; data["method"] = []; data["Z"] = [] #dataset descriptor
        data["ansatz_acc"] = []; data["ansatz_C"] = []
        data["chipr_acc"] = []; data["chipr_C"] = []
        #data["dn_acc"] = []; data["dn_C"] = []
        #data["ds_acc"] = []; data["ds_C"] = []
        with warnings.catch_warnings(record=True): #CHIPR NaN problem
            for dset in rel_datasets:
                print("dataset = ", dset["mol"], dset["state"], dset["author"])
                data["mol"].append(dset["mol"]); data["state"].append(dset["state"]); data["author"].append(dset["author"]);
                if "method" in dset:
                    data["method"].append(dset["method"])
                else:
                    data["method"].append(None)
                Z = Zs[dset["mol"]]; data["Z"].append(Z) #assign Z
                V = dset["V"]; R = dset["R"]
                min_test_rmses = [np.inf, np.inf]; min_Cs = [None, None]
                fold = 0
                for train_index, test_index in kf.split(R): #k-cross-val
                    print(">>> fold = ",fold)
                    fold+=1
                    R_train, R_test = R[train_index], R[test_index]
                    V_train, V_test = V[train_index], V[test_index]
                    
                    args_train = [(R_train,Z,M), (R_train,Z,M,m)] 
                    args_test = [(R_test,Z,M), (R_test,Z,M,m)]
                    rmses_test = []; Cs = []
                    for i, f in enumerate(Fs):
                        len_C = len_Cs[i]
                        rmse_train, C = pmodel.multiple_multistart(restarts, powers, delta, f, V_train, *args_train[i], len_C=len_C, mode="default")
                        V_pred = f(C, *args_test[i])
                        rmse_test = pmodel.RMSE(V_test, V_pred)

                        rmses_test.append(rmse_test); Cs.append(C)
                        print(F_names[i], "rmse = ",rmse_test)
                        data["num_params"].append(len_C)
                    
                    #get min test rmse:
                    for i in range(len(min_test_rmses)): #ignore train rmse
                        if rmses_test[i] < min_test_rmses[i]:
                            min_test_rmses[i] = rmses_test[i]
                            min_Cs[i] = Cs[i]
                    
                data["ansatz_acc"].append(min_test_rmses[0]); data["chipr_acc"].append(min_test_rmses[1]);
                data["ansatz_C"].append(min_Cs[0]); data["chipr_C"].append(min_Cs[1]);
                    
                print("picked rmses = ",min_test_rmses)
                
        print(data)
        filename = "result/cross_val_each_state_"+datetime.datetime.now().strftime('%d%m%y_%H%M%S')+".pkl"
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle)
        
    '''end of main functions, actual main starts below'''
    #cross_val_each_state_fit()
    #special_split_fit()
    time_eval()
    #special_split_fit()
    #split_data_fit_performance()