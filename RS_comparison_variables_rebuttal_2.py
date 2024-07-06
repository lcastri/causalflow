from copy import deepcopy
import json
import os
import random
from tigramite.independence_tests.gpdc_torch import GPDCtorch as GPDC
# from tigramite.independence_tests.gpdc import GPDC
from causalflow.CPrinter import CPLevel
from causalflow.causal_discovery.CAnDOIT import CAnDOIT
from causalflow.causal_discovery.FPCMCI import FPCMCI
from causalflow.causal_discovery.baseline.PCMCI import PCMCI
from causalflow.selection_methods.TE import TE, TEestimator
from causalflow.random_system.RandomDAG import NoiseType, RandomDAG
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
             sta._PCMCI : {sta._TIME : None, 
                            sta._FN : None, 
                            sta._FP : None, 
                            sta._TP : None, 
                            sta._FPR : None,
                            sta._PREC : None, 
                            sta._PREC : None, 
                            sta._RECA : None, 
                            sta._F1SCORE : None, 
                            sta._SHD : None, 
                            sta._SCM: None,
                            "SpuriousLinks": None,
                            "N_SpuriousLinks": None},
             sta._FPCMCI : {sta._TIME : None, 
                            sta._FN : None, 
                            sta._FP : None, 
                            sta._TP : None, 
                            sta._FPR : None,
                            sta._PREC : None, 
                            sta._PREC : None, 
                            sta._RECA : None, 
                            sta._F1SCORE : None, 
                            sta._SHD : None, 
                            sta._SCM: None,
                            "SpuriousLinks": None,
                            "N_SpuriousLinks": None},   
             sta._CAnDOIT : {sta._TIME : None, 
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
    if list(scm.keys()):
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

    
def save_result(pcmci_t, pcmci_scm, fpcmci_t, fpcmci_scm, candoit_t, candoit_scm):
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
    
    for algo, scm, t in zip([sta._PCMCI, sta._FPCMCI, sta._CAnDOIT], [pcmci_scm, fpcmci_scm, candoit_scm], [pcmci_t, fpcmci_t, candoit_t]):
        res_tmp[algo][sta._TIME] = t
        res_tmp[algo][sta._FN] = RS.get_FN(cm = scm)
        res_tmp[algo][sta._FP] = RS.get_FP(cm = scm)
        res_tmp[algo][sta._TP] = RS.get_TP(cm = scm)
        res_tmp[algo][sta._FPR] = RS.FPR(cm = scm)
        res_tmp[algo][sta._PREC] = RS.precision(cm = scm)
        res_tmp[algo][sta._RECA] = RS.recall(cm = scm)
        res_tmp[algo][sta._F1SCORE] = RS.f1_score(cm = scm)
        res_tmp[algo][sta._SHD] = RS.shd(cm = scm)
        res_tmp[algo][sta._SCM] = str(scm)
        spurious_links = get_spurious_links(scm)
        res_tmp[algo]["SpuriousLinks"] = str(spurious_links)
        res_tmp[algo]["N_SpuriousLinks"] = len(spurious_links)
        res_tmp[algo]["N_EquiDAG_2exp"] = 2**len(spurious_links)
        print(algo + " statistics:")
        print("\t|TP score = " + str(res_tmp[algo][sta._TP]))
        print("\t|FP score = " + str(res_tmp[algo][sta._FP]))
        print("\t|FN score = " + str(res_tmp[algo][sta._FN]))
        print("\t|F1 score = " + str(res_tmp[algo][sta._F1SCORE]))
        print("\t|SHD = " + str(res_tmp[algo][sta._SHD]))
        print("\t|FPR = " + str(res_tmp[algo][sta._FPR]))
        print("\t|TPR (Recall) = " + str(res_tmp[algo][sta._RECA]))

    
if __name__ == '__main__':   
    nsample_obs = 1250
    nsample_int = 250
    resdir = "rebuttal_nvariable_1hconf_nonlin_" + str(nsample_obs) + "_" + str(nsample_int)
    f_alpha = 0.05
    pcmci_alpha = 0.05
    min_lag = 1
    max_lag = 2
    min_c = 0.1
    max_c = 0.5
    nfeature = range(7, 15)
    nrun = 25
    noise = (NoiseType.Uniform, -0.1, 0.1)
    
    
    for n in nfeature:
        for nr in range(nrun):
            #########################################################################################################################
            # DATA
            while True:
                try:
                    resfolder = resdir + '/' + str(n) + '/' + str(nr)
                    os.makedirs('results/' + resfolder, exist_ok = True)
                    res_tmp = deepcopy(EMPTY_RES)
                    
                    RS = RandomDAG(nvars = n, nsamples = nsample_obs+nsample_int, 
                                        link_density = 2, coeff_range = (min_c, max_c), max_exp = 2, 
                                        min_lag = min_lag, max_lag = max_lag, noise_config = noise,
                                        functions = ['', 'sin', 'cos', 'abs'], operators=['+', '-', '*'], n_hidden_confounders = 1)
                    RS.gen_equations()

                    d_obs = RS.gen_obs_ts()
                    
                    d_int = dict()
                    for int_var in RS.potentialIntervention.values():
                        i = RS.intervene(int_var, nsample_int, random.uniform(2, 3))
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
                    # FPCMCI
                    fpcmci = FPCMCI(deepcopy(d_obs),
                                    f_alpha = f_alpha, 
                                    alpha = pcmci_alpha, 
                                    min_lag = min_lag, 
                                    max_lag = max_lag, 
                                    sel_method = TE(TEestimator.Gaussian), 
                                    val_condtest = GPDC(significance = 'analytic'),
                                    verbosity = CPLevel.INFO,
                                    neglect_only_autodep = False,
                                    resfolder = resfolder + "/fpcmci")

                    new_start = time()
                    fpcmci_cm = fpcmci.run()
                    elapsed_fpcmci = time() - new_start
                    fpcmci_time = str(timedelta(seconds = elapsed_fpcmci))
                    print(fpcmci_time)
                    fpcmci.timeseries_dag()
                    gc.collect()
            
                    if len(get_spurious_links(fpcmci_cm.get_SCM())) == 0: 
                        gc.collect()
                        remove_directory(os.getcwd() + "/results/" + resfolder)
                        continue
                    
                    
                    #########################################################################################################################
                    # PCMCI
                    pcmci = PCMCI(deepcopy(d_obs),
                                    min_lag = min_lag, 
                                    max_lag = max_lag, 
                                    val_condtest = GPDC(significance = 'analytic'),
                                    verbosity = CPLevel.INFO,
                                    alpha = pcmci_alpha, 
                                    neglect_only_autodep = False,
                                    resfolder = resfolder + "/pcmci")
                    
                    new_start = time()
                    pcmci_cm = pcmci.run()
                    elapsed_pcmci = time() - new_start
                    pcmci_time = str(timedelta(seconds = elapsed_pcmci))
                    print(pcmci_time)
                    pcmci.timeseries_dag()
                    gc.collect()
                                    
            
                    #########################################################################################################################
                    # CAnDOIT
                    new_d_obs = deepcopy(d_obs)
                    new_d_obs.d = new_d_obs.d[:-nsample_int]
                    candoit = CAnDOIT(new_d_obs, 
                                      deepcopy(d_int),
                                      f_alpha = f_alpha, 
                                      alpha = pcmci_alpha, 
                                      min_lag = min_lag, 
                                      max_lag = max_lag, 
                                      sel_method = TE(TEestimator.Gaussian), 
                                      val_condtest = GPDC(significance = 'analytic'),
                                      verbosity = CPLevel.INFO,
                                      neglect_only_autodep = False,
                                      resfolder = resfolder + "/candoit",
                                      plot_data = False,
                                      exclude_context = True)
                    
                    new_start = time()
                    candoit_cm = candoit.run()
                    elapsed_candoit = time() - new_start
                    candoit_time = str(timedelta(seconds = elapsed_candoit))
                    print(candoit_time)
                    candoit.timeseries_dag()
                    gc.collect()
                        
                    break
                    
                except Exception as e:
                    print(e)
                    remove_directory(os.getcwd() + "/results/" + resfolder)
                    continue


            #########################################################################################################################
            # SAVE
            save_result(pcmci_time, get_correct_SCM(GT, pcmci_cm.get_SCM()),
                        fpcmci_time, get_correct_SCM(GT, fpcmci_cm.get_SCM()),
                        candoit_time, get_correct_SCM(GT, candoit_cm.get_SCM()))
            
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