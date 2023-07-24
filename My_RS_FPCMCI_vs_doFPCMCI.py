from copy import deepcopy
import json
import os
import random
from tigramite.independence_tests.gpdc import GPDC
from fpcmci.CPrinter import CPLevel
from fpcmci.doFPCMCI import doFPCMCI
from fpcmci.FPCMCI import FPCMCI
from fpcmci.graph.DAG import DAG
from fpcmci.selection_methods.TE import TE, TEestimator
from random_system.random_system_my import NoiseType, RandomSystem
from pathlib import Path

from time import time
from datetime import timedelta
import res_statistics as sta


def get_correct_SCM(gt, scm):
    new_scm = {v: list() for v in gt.keys()}
    if features:
        for k in scm:
            new_scm[k] = scm[k]
    return new_scm
    
    
def save_result(gt, confTriple, hiddenConf, interv_var, fpcmci_t, fpcmci_scm, dofpcmci_t, dofpcmci_scm):
    print("\n")
    print("Number of variable = " + str(nfeature))
    print("Ground truth = " + str(GT))
    print("Confoundend triple: " + str(sel_confounder))
    print("Hidden confounder: " + conf_to_hide)
    print("Intervention variable: " + int_var)
    
    res_tmp["GT"] = str(gt)
    res_tmp["ConfounderTriple"] = str(confTriple)
    res_tmp["HiddenConfounder"] = str(hiddenConf)
    res_tmp["InterventionVariable"] = interv_var
    
    res_tmp[sta._FPCMCI][sta._TIME] = fpcmci_t
    res_tmp[sta._FPCMCI][sta._PREC] = sta.precision(gt = gt, cm = fpcmci_scm)
    res_tmp[sta._FPCMCI][sta._RECA] = sta.recall(gt = gt, cm = fpcmci_scm)
    res_tmp[sta._FPCMCI][sta._F1SCORE] = sta.f1_score(res_tmp[sta._FPCMCI][sta._PREC], res_tmp[sta._FPCMCI][sta._RECA])
    res_tmp[sta._FPCMCI][sta._SHD] = sta.shd(gt = gt, cm = fpcmci_scm)
    res_tmp[sta._FPCMCI]["SCM"] = str(fpcmci_scm)
    print(sta._FPCMCI + " statistics:")
    print("\t|F1 score = " + str(res_tmp[sta._FPCMCI][sta._F1SCORE]))
    print("\t|SHD = " + str(res_tmp[sta._FPCMCI][sta._SHD]))
    print("\t|time = " + str(res_tmp[sta._FPCMCI][sta._TIME]))
    
    res_tmp[sta._doFPCMCI][sta._TIME] = dofpcmci_t
    res_tmp[sta._doFPCMCI][sta._PREC] = sta.precision(gt = gt, cm = dofpcmci_scm)
    res_tmp[sta._doFPCMCI][sta._RECA] = sta.recall(gt = gt, cm = dofpcmci_scm)
    res_tmp[sta._doFPCMCI][sta._F1SCORE] = sta.f1_score(res_tmp[sta._doFPCMCI][sta._PREC], res_tmp[sta._doFPCMCI][sta._RECA])
    res_tmp[sta._doFPCMCI][sta._SHD] = sta.shd(gt = gt, cm = dofpcmci_scm)
    res_tmp[sta._doFPCMCI]["SCM"] = str(dofpcmci_scm)
    print(sta._doFPCMCI + " statistics:")
    print("\t|F1 score = " + str(res_tmp[sta._doFPCMCI][sta._F1SCORE]))
    print("\t|SHD = " + str(res_tmp[sta._doFPCMCI][sta._SHD]))
    print("\t|time = " + str(res_tmp[sta._doFPCMCI][sta._TIME]))

    
if __name__ == '__main__':   
    resdir = "My_RS_FPCMCI_vs_doFPCMCI_075intlength"
    f_alpha = 0.05
    pcmci_alpha = 0.05
    min_lag = 1
    max_lag = 2
    min_c = 0
    max_c = 1
    nsample = 1000
    nfeature = range(4, 11)
    nrun = 10
    noise = (NoiseType.Uniform, -0.1, 0.1)
    
    
    for n in nfeature:
        for nr in range(nrun):
            resfolder = resdir + '/' + str(n) + '/' + str(nr)
            os.makedirs('results/' + resfolder, exist_ok = True)
            res_tmp = {"GT" : None,
                       "ConfounderTriple" : None,
                       "HiddenConfounder" : None, 
                       "InterventionVariable" : None,
                       sta._FPCMCI : {sta._TIME : None, sta._PREC : None, sta._RECA : None, sta._F1SCORE : None, sta._SHD : None, "SCM": None},   
                       sta._doFPCMCI : {sta._TIME : None, sta._PREC : None, sta._RECA : None, sta._F1SCORE : None, sta._SHD : None, "SCM": None}}
            
            #########################################################################################################################
            # DATA
            confounders = list()
            
            while len(confounders) == 0:
                RS = RandomSystem(nvars = n, nsamples = nsample, 
                                  max_terms = 2, coeff_range = (min_c, max_c), max_exp = 2, 
                                  min_lag = min_lag, max_lag = max_lag, noise_config = noise,
                                  functions = [''], operators=['+', '-'])
                RS.gen_equations()
                confounders = RS.get_lagged_confounders()
            
            GT = RS.get_SCM()
            g = DAG(RS.variables, min_lag, max_lag, False, GT)
            g.ts_dag(max_lag, save_name = 'results/' + resfolder + '/gt')

            d_obs = RS.gen_obs_ts()
            
            sel_confounder = random.choice(confounders)
            conf_to_hide = next(iter(sel_confounder.keys()))
            int_var = min(sel_confounder[conf_to_hide], key=lambda x: abs(x[2]))[0]
            
            d_int = RS.interv_var(int_var, int(nsample*0.75), random.uniform(2, 5))
            
            RS.hide_var(conf_to_hide)
            GT = RS.get_SCM()
            g = DAG(RS.variables, min_lag, max_lag, False, GT)
            g.ts_dag(max_lag, save_name = 'results/' + resfolder + '/gt_hidden')
            
            d_obs.plot_timeseries('results/' + resfolder + '/obs_data.png')
            d_int[int_var].plot_timeseries('results/' + resfolder + '/int_data.png')
            
            print("Confoundend triple: " + str(sel_confounder))
            print("Hidden confounder: " + conf_to_hide)
            print("Intervention variable: " + int_var)
            
            #########################################################################################################################
            # FPCMCI
            fpcmci = FPCMCI(deepcopy(d_obs),
                            f_alpha = f_alpha, 
                            pcmci_alpha = pcmci_alpha, 
                            min_lag = min_lag, 
                            max_lag = max_lag, 
                            sel_method = TE(TEestimator.Gaussian), 
                            val_condtest = GPDC(significance = 'analytic', gp_params = None),
                            verbosity = CPLevel.INFO,
                            neglect_only_autodep = False,
                            resfolder = resfolder + "/fpcmci")

            new_start = time()
            features, fpcmci_cm = fpcmci.run()
            elapsed_newFPCMCI = time() - new_start
            fpcmci_time = str(timedelta(seconds = elapsed_newFPCMCI))
            print(fpcmci_time)
            fpcmci.timeseries_dag()
            
            
            #########################################################################################################################
            # doFPCMCI
            dofpcmci = doFPCMCI(deepcopy(d_obs), 
                                deepcopy(d_int),
                                f_alpha = f_alpha, 
                                pcmci_alpha = pcmci_alpha, 
                                min_lag = min_lag, 
                                max_lag = max_lag, 
                                sel_method = TE(TEestimator.Gaussian), 
                                val_condtest = GPDC(significance = 'analytic', gp_params = None),
                                verbosity = CPLevel.INFO,
                                neglect_only_autodep = False,
                                resfolder = resfolder + "/dofpcmci",
                                plot_data = False,
                                exclude_context = True)
            
            new_start = time()
            features, dofpcmci_cm = dofpcmci.run()
            elapsed_newFPCMCI = time() - new_start
            dofpcmci_time = str(timedelta(seconds = elapsed_newFPCMCI))
            print(dofpcmci_time)
            dofpcmci.timeseries_dag()


            #########################################################################################################################
            # SAVE
            save_result(GT, str(sel_confounder), conf_to_hide, int_var, 
                        fpcmci_time, get_correct_SCM(GT, fpcmci_cm.get_SCM()),
                        dofpcmci_time, get_correct_SCM(GT, dofpcmci_cm.get_SCM()))
            
            Path(os.getcwd() + "/results/" + resdir).mkdir(parents=True, exist_ok=True)
            filename = os.getcwd() + "/results/" + resdir + "/" + str(n) + ".json"
            
            # Check if the file exists
            if os.path.exists(filename):
                # File exists, load its contents into a dictionary
                with open(filename, 'r') as file:
                    data = json.load(file)
            else:
                # File does not exist, create a new dictionary
                data = {}

            # Modify the dictionary
            data[nr] = res_tmp

            # Save the dictionary back to a JSON file
            with open(filename, 'w') as file:
                json.dump(data, file)
            res_tmp.clear()