from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import random
from tigramite.independence_tests.gpdc import GPDC
from fpcmci.CPrinter import CPLevel
from fpcmci.doFPCMCI import doFPCMCI
from fpcmci.FPCMCI import FPCMCI
from fpcmci.preprocessing.data import Data
from fpcmci.selection_methods.TE import TE, TEestimator
from fpcmci.basics.constants import LabelType
import numpy as np
import res_statistics as sta


MAX_N_INT = 3
MAX_N_HID = 1
T = 1500
P_OBS = 0.85
P_INT = 0.15

if __name__ == '__main__':   

    ground_truth = {
                    'X_0' : [('X_1', -2), ('X_2', -1)],
                    'X_1' : [],
                    'X_2' : [('X_1', -2)],
                    'X_3' : [('X_3', -1)],
                    'X_4' : [('X_2', -2), ('X_3', -1)],
                    'X_5' : [('X_5', -2), ('X_0', -1)],
                    'X_6' : [],
                    }

    f_alpha = 0.1
    pcmci_alpha = 0.05
    min_lag = 1
    max_lag = 2

    resfolder = 'S2_doFPCMCI_vs_FPCMCI_wAdding_wHidden'
    Tobs = int(T)
    Tint = int(T*P_INT)
    
    np.random.seed(3)
    nfeature = range(3, 8)
    ncoeff = 10
    min_n = 0
    max_n = 1

    for n in nfeature:
        
        var_names = ['X_' + str(f) for f in range(n)]
        gt_n = {k : ground_truth[k] for k in ground_truth.keys() if int(k[-1]) < n}
        
        for nc in range(ncoeff):
            num_int = random.randint(1, MAX_N_INT - MAX_N_HID)
                
            d = np.random.random(size = (Tobs, n))
            
            res_tmp = {sta._FPCMCI : {sta._TIME : None, sta._PREC : None, sta._RECA : None, sta._F1SCORE : None, sta._SHD : None},   
                       sta._doFPCMCI : {sta._TIME : None, sta._PREC : None, sta._RECA : None, sta._F1SCORE : None, sta._SHD : None}}
            c = np.random.uniform(min_n, max_n, (n, n))
            data = deepcopy(d)
                            
            for t in range(max_lag, Tobs):
                data[t, 0] += c[0][1] * data[t-2, 1] * c[0][2] * data[t-1, 2]
                data[t, 2] += c[2][1] * data[t-2, 1] ** 2
                if n > 3: data[t, 3] += data[t-1, 3] + c[3][3]
                if n > 4: data[t, 4] += c[4][2] * data[t-2, 2] - c[4][3] * data[t-1, 3]
                if n > 5: data[t, 5] += c[5][0] * data[t-1, 0] / (1 + c[5][5] * data[t-2, 5])
            
            # Hide variables
            gt_n_hidden = deepcopy(gt_n)
            hid_var_i = list()
            for h in range(MAX_N_HID):
                hid_var = random.choice(var_names)
                hid_var_i.append(var_names.index(hid_var))
                
                for v in gt_n_hidden:
                    for s in gt_n_hidden[v]:
                        if s[0] == hid_var: gt_n_hidden[v].remove(s)
                del gt_n_hidden[hid_var]
                
            int_data = dict()
            int_vars = [v for v in var_names if var_names.index(v) not in hid_var_i]
            for ni in range(num_int):
                int_var = random.choice(int_vars)
                int_var_i = var_names.index(int_var)
                int_vars.remove(int_var)
                
                d_int = np.random.random(size = (Tint, len(var_names)))
                d_int[:, int_var_i] = random.randint(2, 6) * np.ones(shape = (Tint,)) 
                for t in range(max_lag, Tint):
                    if int_var_i != 0: data[t, 0] += c[0][1] * data[t-2, 1] * c[0][2] * data[t-1, 2]
                    if int_var_i != 2: data[t, 2] += c[2][1] * data[t-2, 1] ** 2
                    if n > 3 and int_var_i != 3: data[t, 3] += data[t-1, 3] + c[3][3]
                    if n > 4 and int_var_i != 4: data[t, 4] += c[4][2] * data[t-2, 2] - c[4][3] * data[t-1, 3]
                    if n > 5 and int_var_i != 5: data[t, 5] += c[5][0] * data[t-1, 0] / (1 + c[5][5] * data[t-2, 5])
                
                feats = deepcopy(var_names)
                for i in hid_var_i: 
                    feats.pop(i)
                    d_int = np.delete(d_int, i, 1)
                
                df_int = Data(d_int, vars = feats)
                int_data[int_var] = df_int
                
            feats = deepcopy(var_names)
            for i in hid_var_i: 
                feats.pop(i)
                data = np.delete(data, i, 1)
            df_obs = Data(data, vars = feats)
                
            resdir = resfolder + '/' + str(n) + "_" + str(nc)
            
            print("\n\n-----------------------------------------------------------------------")
            print("Iteration = " + str(nc+1) + "/" + str(ncoeff))
            print("Variables = " + str(n) + " -- " + str(var_names))
            print("Hidden variables = " + str(len(hid_var_i)) + " -- " + str([x for x in var_names if x not in feats]))
            print("Interventions = " + str(num_int) + " -- " + str(list(int_data.keys())))
            print("Observation dataset length = " + str(Tobs))
            print("Intervention dataset length = " + str(sum([tmp.T for tmp in int_data.values()])))
            print("Observation dataset size = " + str((df_obs.T , df_obs.N)))
            print("Intervention dataset size = " + str((df_obs.T + sum([tmp.T for tmp in int_data.values()]) , df_obs.N)))
            print("\n")
            
            #########################################################################################################################
            # FPCMCI
            fpcmci = FPCMCI(deepcopy(df_obs),
                            f_alpha = f_alpha, 
                            pcmci_alpha = pcmci_alpha, 
                            min_lag = min_lag, 
                            max_lag = max_lag, 
                            sel_method = TE(TEestimator.Gaussian), 
                            val_condtest = GPDC(significance = 'analytic', gp_params = None),
                            verbosity = CPLevel.NONE,
                            neglect_only_autodep = False,
                            resfolder = resdir + "_fpcmci")
    
            startFPCMCI = datetime.now()
            features, cm = fpcmci.run()
            stopFPCMCI = datetime.now()  
            fpcmci.dag(node_layout='circular', label_type=LabelType.Lag)
            
            scm = {v: list() for v in gt_n_hidden.keys()}
            if features:
                fpcmci_scm = cm.get_SCM()
                for k in fpcmci_scm:
                    scm[k] = fpcmci_scm[k]
            fpcmci_scm = scm
            res_tmp = sta.save_result(res_tmp, startFPCMCI, stopFPCMCI, fpcmci_scm, sta._FPCMCI, gt_n_hidden)
    
            #########################################################################################################################
            # doFPCMCI      
            dofpcmci = doFPCMCI(deepcopy(df_obs), 
                                int_data,
                                f_alpha = f_alpha, 
                                pcmci_alpha = pcmci_alpha, 
                                min_lag = min_lag, 
                                max_lag = max_lag, 
                                sel_method = TE(TEestimator.Gaussian), 
                                val_condtest = GPDC(significance = 'analytic', gp_params = None),
                                verbosity = CPLevel.NONE,
                                neglect_only_autodep = False,
                                resfolder = resdir + "_dofpcmci")
    
            start_doFPCMCI = datetime.now()
            features, cm = dofpcmci.run()
            stop_doFPCMCI = datetime.now()
            dofpcmci.dag(node_layout='circular', label_type=LabelType.Lag)

            scm = {v: list() for v in gt_n_hidden.keys()}
            if features:
                dofpcmci_scm = cm.get_SCM()
                for k in dofpcmci_scm:
                    scm[k] = dofpcmci_scm[k]
            dofpcmci_scm = scm
            res_tmp = sta.save_result(res_tmp, start_doFPCMCI, stop_doFPCMCI, dofpcmci_scm, sta._doFPCMCI, gt_n_hidden)
            
            # Save result
            Path(os.getcwd() + "/results/" + resfolder).mkdir(parents=True, exist_ok=True)
            filename = os.getcwd() + "/results/" + resfolder + "/" + str(n) + ".json"
            # Check if the file exists
            if os.path.exists(filename):
                # File exists, load its contents into a dictionary
                with open(filename, 'r') as file:
                    data = json.load(file)
            else:
                # File does not exist, create a new dictionary
                data = {}

            # Modify the dictionary
            data[nc] = res_tmp

            # Save the dictionary back to a JSON file
            with open(filename, 'w') as file:
                json.dump(data, file)
            res_tmp.clear()