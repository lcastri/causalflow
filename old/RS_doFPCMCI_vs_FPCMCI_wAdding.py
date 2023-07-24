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
from random_system.random_system_my import RandomSystem, NoiseType

MAX_N_INT = 1
T = 1500
P_OBS = 0.85
P_INT = 0.15


if __name__ == '__main__':

    f_alpha = 0.1
    pcmci_alpha = 0.05
    min_lag = 1
    max_lag = 2

    resfolder = 'RS_doFPCMCI_vs_FPCMCI_wAdding_1'
    Tobs = T
    Tint = int(T*P_INT)
    
    np.random.seed(3)
    nfeature = range(3, 7)
    n_attempts = 10
    min_n = 0
    max_n = 1

    for n in nfeature:
        for na in range(n_attempts):
            res_tmp = {sta._FPCMCI : {sta._TIME : None, sta._PREC : None, sta._RECA : None, sta._F1SCORE : None, sta._SHD : None},   
                       sta._doFPCMCI : {sta._TIME : None, sta._PREC : None, sta._RECA : None, sta._F1SCORE : None, sta._SHD : None}}
            
            num_int = random.randint(1, min(n, MAX_N_INT))
            err = 1
            while err is not None:
                try:
                    RS = RandomSystem(n, Tobs, max_terms = n, coeff_range = (min_n, max_n), max_exp = 2,
                                      min_lag = min_lag, max_lag = max_lag, noise_config = (NoiseType.Uniform, 0, 1.0),
                                      interv_randrange = (2, 6), functions=['', 'pow'])
                    RS.gen_equations()
                    gt = RS.get_SCM()
            
                    df_obs = RS.gen_obs_ts()
                    df_int = RS.rand_inter_vars(num_int)
                    err = None
                except Exception as e:
                    err = e
                
            resdir = resfolder + '/' + str(n) + "_" + str(na)
            
            print("\n\n-----------------------------------------------------------------------")
            print("Iteration = " + str(na+1) + "/" + str(n_attempts))
            print("Variables = " + str(n) + " -- " + str(RS.variables))
            print("S = ")
            RS.print_equations()
            print("GT = " + str(gt))
            # print("Hidden variables = " + str(len(hid_var_i)) + " -- " + str([x for x in var_names if x not in feats]))
            print("Interventions = " + str(num_int) + " -- " + str(list(df_int.keys())))
            print("Observation dataset length = " + str(Tobs))
            print("Intervention dataset length = " + str(sum([tmp.T for tmp in df_int.values()])))
            print("Observation dataset size = " + str((df_obs.T , df_obs.N)))
            print("Intervention dataset size = " + str((df_obs.T + sum([tmp.T for tmp in df_int.values()]) , df_obs.N)))
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
            
            scm = {v: list() for v in gt.keys()}
            if features:
                fpcmci_scm = cm.get_SCM()
                for k in fpcmci_scm:
                    scm[k] = fpcmci_scm[k]
            fpcmci_scm = scm
            res_tmp = sta.save_result(res_tmp, startFPCMCI, stopFPCMCI, fpcmci_scm, sta._FPCMCI, gt)
    
            #########################################################################################################################
            # doFPCMCI      
            dofpcmci = doFPCMCI(deepcopy(df_obs),
                                df_int,
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

            scm = {v: list() for v in gt.keys()}
            if features:
                dofpcmci_scm = cm.get_SCM()
                for k in dofpcmci_scm:
                    scm[k] = dofpcmci_scm[k]
            dofpcmci_scm = scm
            res_tmp = sta.save_result(res_tmp, start_doFPCMCI, stop_doFPCMCI, dofpcmci_scm, sta._doFPCMCI, gt)
            
            #########################################################################################################################
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
            data[na] = res_tmp

            # Save the dictionary back to a JSON file
            with open(filename, 'w') as file:
                json.dump(data, file)
            res_tmp.clear()