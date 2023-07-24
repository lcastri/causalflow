from copy import deepcopy
import json
import os
import random
# from tigramite.independence_tests.gpdc_torch import GPDCtorch as GPDC
from tigramite.independence_tests.gpdc import GPDC
from fpcmci.CPrinter import CPLevel
from fpcmci.doFPCMCI import doFPCMCI
from fpcmci.FPCMCI import FPCMCI
from fpcmci.selection_methods.TE import TE, TEestimator
from fpcmci.random_system.RandomSystem import NoiseType, RandomSystem
from pathlib import Path

from time import time
from datetime import timedelta
import res_statistics as sta
import gc
import shutil


EMPTY_RES = {"GT" : None,
             "Confounders" : None,
             "HiddenConfounders" : None, 
             "InterventionVariables" : None,
             "ExpectedSpuriousLinks" : None,
             "N_ExpectedSpuriousLinks" : None,
             sta._doFPCMCI : {sta._TIME : None, 
                              sta._FN : None, 
                              sta._FP : None, 
                              sta._TP : None, 
                              sta._FPR : None,
                              sta._PREC : None, 
                              sta._RECA : None, 
                              sta._F1SCORE : None, 
                              sta._SHD : None, 
                              sta._SCM: None,
                              "SpuriousLinks": None,
                              "N_SpuriousLinks": None}
             }


def remove_directory(directory_path):
    try:
        # Use shutil.rmtree() to remove the directory and its content recursively
        shutil.rmtree(directory_path)
        print(f"Directory '{directory_path}' and its content have been removed.")
    except OSError as e:
        print(f"Error: {e}")


def get_correct_SCM(gt, scm):
    new_scm = {v: list() for v in gt.keys()}
    if features:
        for k in scm:
            new_scm[k] = scm[k]
    return new_scm


def get_spurious_links(scm):
    spurious = list()
    exp_spurious = RS.expected_spurious_links
    for exp_s in exp_spurious:
        if exp_s[1] in scm and (exp_s[0], -1) in scm[exp_s[1]]:
            spurious.append(exp_s)
            
    return spurious

    
def save_result(dofpcmci_t, dofpcmci_scm):
    print("\n")
    print("Number of variable = " + str(n))
    print("Ground truth = " + str(RS.get_SCM()))
    print("Confounders: " + str(RS.confounders))
    print("Hidden confounder: " + str(list(RS.confounders.keys())))
    print("Intervention variable: " + str(list(d_int.keys())))
    
    res_tmp["GT"] = str(RS.get_SCM())
    res_tmp["Confounders"] = str(RS.confounders)
    res_tmp["HiddenConfounders"] = str(list(RS.confounders.keys()))
    res_tmp["InterventionVariables"] = str(list(d_int.keys()))
    res_tmp["ExpectedSpuriousLinks"] = str(RS.expected_spurious_links)
    res_tmp["N_ExpectedSpuriousLinks"] = len(RS.expected_spurious_links)
    
    res_tmp[sta._doFPCMCI][sta._TIME] = dofpcmci_t
    res_tmp[sta._doFPCMCI][sta._FN] = RS.get_FN(cm = dofpcmci_scm)
    res_tmp[sta._doFPCMCI][sta._FP] = RS.get_FP(cm = dofpcmci_scm)
    res_tmp[sta._doFPCMCI][sta._TP] = RS.get_TP(cm = dofpcmci_scm)
    res_tmp[sta._doFPCMCI][sta._FPR] = RS.FPR(cm = dofpcmci_scm)
    res_tmp[sta._doFPCMCI][sta._PREC] = RS.precision(cm = dofpcmci_scm)
    res_tmp[sta._doFPCMCI][sta._RECA] = RS.recall(cm = dofpcmci_scm)
    res_tmp[sta._doFPCMCI][sta._F1SCORE] = RS.f1_score(cm = dofpcmci_scm)
    res_tmp[sta._doFPCMCI][sta._SHD] = RS.shd(cm = dofpcmci_scm)
    res_tmp[sta._doFPCMCI][sta._SCM] = str(dofpcmci_scm)
    res_tmp[sta._doFPCMCI]["SpuriousLinks"] = str(get_spurious_links(dofpcmci_scm))
    res_tmp[sta._doFPCMCI]["N_SpuriousLinks"] = len(get_spurious_links(dofpcmci_scm))
    print(sta._doFPCMCI + " statistics:")
    print("\t|TP score = " + str(res_tmp[sta._doFPCMCI][sta._TP]))
    print("\t|FP score = " + str(res_tmp[sta._doFPCMCI][sta._FP]))
    print("\t|FN score = " + str(res_tmp[sta._doFPCMCI][sta._FN]))
    print("\t|F1 score = " + str(res_tmp[sta._doFPCMCI][sta._F1SCORE]))
    print("\t|SHD = " + str(res_tmp[sta._doFPCMCI][sta._SHD]))
    print("\t|FPR = " + str(res_tmp[sta._doFPCMCI][sta._FPR]))
    print("\t|TPR (Recall) = " + str(res_tmp[sta._doFPCMCI][sta._RECA]))

    
if __name__ == '__main__':   
    resdir = "nhidden_conf_1000_1000_0_0.5"
    f_alpha = 0.05
    pcmci_alpha = 0.05
    min_lag = 1
    max_lag = 2
    min_c = 0
    max_c = 0.5
    nsample = 1000
    nfeature = range(7, 8)
    nrun = 25
    noise = (NoiseType.Uniform, -0.1, 0.1)
    
    
    for n in nfeature:
        for nhidden in range(2, 5):
            for nr in range(nrun):
                #########################################################################################################################
                # DATA
                while True:
                    try:
                        resfolder = resdir + '/' + str(nhidden) + '/' + str(nr)
                        os.makedirs('results/' + resfolder, exist_ok = True)
                        res_tmp = deepcopy(EMPTY_RES)
                        
                        RS = RandomSystem(nvars = n, nsamples = nsample, 
                                            max_terms = 2, coeff_range = (min_c, max_c), max_exp = 2, 
                                            min_lag = min_lag, max_lag = max_lag, noise_config = noise,
                                            functions = [''], operators=['+', '-', '*'], n_hidden_confounders = nhidden)
                        RS.gen_equations()

                        d_obs = RS.gen_obs_ts()
                        
                        d_int = dict()
                        for int_var in RS.confintvar.values():
                            i = RS.intervene(int_var, nsample, random.uniform(2, 3))
                            d_int[int_var] = i[int_var]
                            d_int[int_var].plot_timeseries('results/' + resfolder + '/interv_' + int_var + '.png')

                    
                        GT = RS.get_SCM()
                        
                        d_obs.plot_timeseries('results/' + resfolder + '/obs_data.png')
                        
                        RS.ts_dag(withHidden = True, save_name = 'results/' + resfolder + '/gt_complete')
                        RS.ts_dag(withHidden = False, save_name = 'results/' + resfolder + '/gt')
                        
                        print("Confounders: " + str(RS.confounders))
                        print("Hidden confounder: " + str(list(RS.confounders.keys())))
                        print("Intervention variable: " + str(list(d_int.keys())))
                
                        #########################################################################################################################
                        # doFPCMCI
                        dofpcmci = doFPCMCI(deepcopy(d_obs), 
                                            deepcopy(d_int),
                                            f_alpha = f_alpha, 
                                            pcmci_alpha = pcmci_alpha, 
                                            min_lag = min_lag, 
                                            max_lag = max_lag, 
                                            sel_method = TE(TEestimator.Gaussian), 
                                            val_condtest = GPDC(significance = 'analytic'),
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
                        gc.collect()
                            
                        break
                        
                    except Exception as e:
                        print(e)
                        remove_directory(os.getcwd() + "/results/" + resfolder)
                        continue

                #########################################################################################################################
                # SAVE
                save_result(dofpcmci_time, get_correct_SCM(GT, dofpcmci_cm.get_SCM()))
                
                Path(os.getcwd() + "/results/" + resdir).mkdir(parents=True, exist_ok=True)
                filename = os.getcwd() + "/results/" + resdir + "/" + str(nhidden) + ".json"
                
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