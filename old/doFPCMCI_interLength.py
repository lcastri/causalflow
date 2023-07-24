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


MAX_N_INT = 1
PERC_STEP = 5


def save_result(startTime, stopTime, scm, i):
    res_tmp[i][sta._TIME] = str(stopTime - startTime)
    res_tmp[i][sta._PREC] = sta.precision(gt = gt, cm = scm)
    res_tmp[i][sta._RECA] = sta.recall(gt = gt, cm = scm)
    res_tmp[i][sta._F1SCORE] = sta.f1_score(res_tmp[i][sta._PREC], res_tmp[i][sta._RECA])
    res_tmp[i][sta._SHD] = sta.shd(gt = gt, cm = scm)
    print("Statistics -- |time = " + str(res_tmp[i][sta._TIME]) + " -- |F1 score = " + str(res_tmp[i][sta._F1SCORE]) + " -- |SHD = " + str(res_tmp[i][sta._SHD]))


if __name__ == '__main__':   

    gt = {
        'X_0' : [('X_0', -1), ('X_1', -1), ('X_2', -1)],
        'X_1' : [],
        'X_2' : [('X_1', -1), ('X_2', -1)],
        'X_3' : [('X_3', -1)],
        'X_4' : [('X_1', -1), ('X_2', -1), ('X_3', -1)],
        'X_5' : [],
        'X_6' : [('X_0', -1), ('X_5', -1)],
        }

    f_alpha = 0.1
    pcmci_alpha = 0.05
    min_lag = 1
    max_lag = 1

    resfolder = 'doFPCMCI_S1_interLength'
    
    np.random.seed(3)
    min_n = 0
    max_n = 1
    N = 7 
    T = 2000
    ninter = 10
    
    percentages = [((100 - i)/100, i/100) for i in range(PERC_STEP, 100, PERC_STEP)]
        
    var_names = ['X_' + str(f) for f in range(N)]
    
    for p_index, p in enumerate(percentages):
        Tobs = int(T)
        # Tobs = int(T*p[0])
        Tint = int(T*p[1])
        num_int = MAX_N_INT
        d = np.random.random(size = (Tobs, N))
        
        c = np.random.uniform(min_n, max_n, (N, N))
        data = deepcopy(d)
                        
        for t in range(max_lag, Tobs):
            data[t, 0] += c[0][0] * data[t-1, 0] - c[0][1] * data[t-1, 1] * c[0][2] * data[t-1, 2]
            data[t, 2] += c[2][1] * data[t-1, 1] / (1 + c[2][2] * data[t-1, 2])
            data[t, 3] += data[t-1, 3] ** (1/2) + c[3][3]
            data[t, 4] += c[4][1] * data[t-1, 1] * c[4][2] * data[t-1, 2] / (1 + c[4][3] * data[t-1, 3])
            data[t, 6] += c[6][0] * data[t-1, 0] / (1 + c[6][5] * data[t-1, 5])
        df_obs = Data(data, vars = var_names)
            
        obs_data = Data(data[0:Tobs, :], vars = var_names)
        
        res_tmp = {iter: {sta._TIME : None, sta._PREC : None, sta._RECA : None, sta._F1SCORE : None, sta._SHD : None} for iter in range(ninter)}
        for iter in range(ninter):
            int_data = dict()
            int_vars = deepcopy(var_names)
            for ni in range(num_int):
                int_var = random.choice(int_vars)
                int_var_i = var_names.index(int_var)
                int_vars.remove(int_var)
                
                d_int = np.random.random(size = (Tint, N))
                d_int[:, int_var_i] = random.randint(2, 6) * np.ones(shape = (Tint,)) 
                for t in range(max_lag, Tint):
                    if int_var_i != 0: d_int[t, 0] += c[0][0] * d_int[t-1, 0] - c[0][1] * d_int[t-1, 1] * c[0][2] * d_int[t-1, 2]
                    if int_var_i != 2: d_int[t, 2] += c[2][1] * d_int[t-1, 1] / (1 + c[2][2] * d_int[t-1, 2])
                    if int_var_i != 3: d_int[t, 3] += d_int[t-1, 3] ** (1/2) + c[3][3]
                    if int_var_i != 4: d_int[t, 4] += c[4][1] * d_int[t-1, 1] * c[4][2] * d_int[t-1, 2] / (1 + c[4][3] * d_int[t-1, 3])
                    if int_var_i != 6: d_int[t, 6] += c[6][0] * d_int[t-1, 0] / (1 + c[6][5] * d_int[t-1, 5])
                    
                df_int = Data(d_int, vars = var_names)
                int_data[int_var] = df_int
                
            resdir = resfolder + '/' + str()
            
            print("\n\n-----------------------------------------------------------------------")
            print("Number of variables = " + str(N))
            print("Percentage = " + str(p_index+1) + "/" + str(len(percentages)))
            print("Iteration = " + str(iter+1) + "/" + str(ninter))
            print("Observation dataset length = " + str(Tobs))
            print("Number of interventions = " + str(num_int))
            print("Intervention variables = " + str(list(int_data.keys())))
            print("Intervention dataset length = " + str(sum([tmp.T for tmp in int_data.values()])))
            print("Observation dataset size = " + str((df_obs.T , df_obs.N)))
            print("Intervention dataset size = " + str((obs_data.T + sum([tmp.T for tmp in int_data.values()]) , df_obs.N)))
            print("\n")

            #########################################################################################################################
            # doFPCMCI      
            dofpcmci = doFPCMCI(deepcopy(obs_data), 
                                deepcopy(int_data),
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
            # dofpcmci.dag(node_layout='circular', label_type=LabelType.Lag)
            scm = {v: list() for v in gt.keys()}
            if features:
                dofpcmci_scm = cm.get_SCM()
                for k in dofpcmci_scm:
                    scm[k] = dofpcmci_scm[k]
            dofpcmci_scm = scm
            save_result(start_doFPCMCI, stop_doFPCMCI, dofpcmci_scm, iter)
        
        # Save result
        Path(os.getcwd() + "/results/" + resfolder).mkdir(parents=True, exist_ok=True)
        filename = os.getcwd() + "/results/" + resfolder + "/" + str(p_index) + ".json"
        # Check if the file exists
        if os.path.exists(filename):
            # File exists, load its contents into a dictionary
            with open(filename, 'r') as file:
                data = json.load(file)
        else:
            # File does not exist, create a new dictionary
            data = {}
        # Modify the dictionary
        data[p_index] = res_tmp
        # Save the dictionary back to a JSON file
        with open(filename, 'w') as file:
            json.dump(data, file)
        res_tmp.clear()