from copy import deepcopy
import json
import os
import random
from tigramite.independence_tests.gpdc_torch import GPDCtorch as GPDC
# from tigramite.independence_tests.gpdc import GPDC
from connectingdots.CPrinter import CPLevel
from connectingdots.causal_discovery.CAnDOIT import CAnDOIT
from connectingdots.causal_discovery.FPCMCI import FPCMCI
from connectingdots.causal_discovery.baseline.DYNOTEARS import DYNOTEARS
from connectingdots.causal_discovery.baseline.PCMCI import PCMCI
from connectingdots.causal_discovery.baseline.TCDF import TCDF
from connectingdots.causal_discovery.baseline.VarLiNGAM import VarLiNGAM
from connectingdots.causal_discovery.baseline.tsFCI import tsFCI
from connectingdots.selection_methods.TE import TE, TEestimator
from connectingdots.random_system.RandomDAG import NoiseType, RandomDAG
from pathlib import Path

from time import time
from datetime import timedelta
from res_statistics_new import *
import gc
import shutil


ALGO_RES = {Metric.TIME.value : None,
            Metric.FN.value : None,
            Metric.FP.value : None,
            Metric.TP.value : None,
            Metric.FPR.value : None,
            Metric.PREC.value : None,
            Metric.RECA.value : None,
            Metric.F1SCORE.value : None,
            Metric.SHD.value : None,
            jWord.SCM.value : None,
            jWord.SpuriousLinks.value: None,
            Metric.N_ESPU.value : None,
            Metric.N_EqDAG.value: None}


EMPTY_RES = {jWord.GT.value : None,
             jWord.Confounders.value : None,
             jWord.HiddenConfounders.value : None, 
             jWord.InterventionVariables.value : None,
             jWord.ExpectedSpuriousLinks.value : None,
             jWord.N_GSPU.value : None,
             Algo.CAnDOIT.value : deepcopy(ALGO_RES),   
             Algo.FPCMCI.value : deepcopy(ALGO_RES),   
             Algo.PCMCI.value : deepcopy(ALGO_RES),
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

    
def save_result(d):
    res_tmp["coeff_range"] = str(RS.coeff_range)
    res_tmp["noise_config"] = str(RS.noise_config)
    res_tmp[jWord.GT.value] = str(RS.get_SCM())
    res_tmp[jWord.Confounders.value] = str(RS.confounders)
    res_tmp[jWord.HiddenConfounders.value] = str(list(RS.confounders.keys()))
    res_tmp[jWord.InterventionVariables.value] = str(list(d_int.keys()))
    res_tmp[jWord.ExpectedSpuriousLinks.value] = str(RS.expected_spurious_links)
    res_tmp[jWord.N_GSPU.value] = len(RS.expected_spurious_links)
    
    for a, r in d.items():
        res_tmp[a.value][Metric.TIME.value] = r["time"]
        res_tmp[a.value][Metric.FN.value] = RS.get_FN(r["scm"])
        res_tmp[a.value][Metric.FP.value] = RS.get_FP(r["scm"])
        res_tmp[a.value][Metric.TP.value] = RS.get_TP(r["scm"])
        res_tmp[a.value][Metric.FPR.value] = RS.FPR(r["scm"])
        res_tmp[a.value][Metric.PREC.value] = RS.precision(r["scm"])
        res_tmp[a.value][Metric.RECA.value] = RS.recall(r["scm"])
        res_tmp[a.value][Metric.F1SCORE.value] = RS.f1_score(r["scm"])
        res_tmp[a.value][Metric.SHD.value] = RS.shd(r["scm"])
        res_tmp[a.value][jWord.SCM.value] = str(r["scm"])
        spurious_links = get_spurious_links(r["scm"])
        res_tmp[a.value][jWord.SpuriousLinks.value] = str(spurious_links)
        res_tmp[a.value][Metric.N_ESPU.value] = len(spurious_links)
        res_tmp[a.value][Metric.N_EqDAG.value] = 2**len(spurious_links)
    
    
if __name__ == '__main__':   
    nsample_obs = 1250
    nsample_int = 250
    resdir = "S1_" + str(nsample_obs) + "_" + str(nsample_int)
    f_alpha = 0.05
    alpha = 0.05
    min_lag = 1
    max_lag = 2
    min_c = 0.1
    max_c = 0.5
    nfeature = range(7, 14)
    nrun = 25
    
    
    for n in nfeature:
        for nr in range(nrun):
            #########################################################################################################################
            # DATA
            while True:
                try:
                    resfolder = 'results/' + resdir + '/' + str(n) + '/' + str(nr)
                    os.makedirs(resfolder, exist_ok = True)
                    res_tmp = deepcopy(EMPTY_RES)
                    
                    coeff_sign = random.choice([-1, 1])
                    noise_param = random.uniform(0.5, 2)
                    noise_uniform = (NoiseType.Uniform, -noise_param, noise_param)
                    noise_gaussian = (NoiseType.Gaussian, 0, noise_param)
                    RS = RandomDAG(nvars = n, nsamples = nsample_obs + nsample_int, 
                                   max_terms = 3, coeff_range = (coeff_sign*min_c, coeff_sign*max_c), max_exp = 2, 
                                   min_lag = min_lag, max_lag = max_lag, noise_config = random.choice([noise_uniform, noise_gaussian]),
                                   functions = ['', 'sin', 'cos', 'abs'], operators=['+', '-', '*'], n_hidden_confounders = random.randint(1, 2))
                    RS.gen_equations()

                    d_obs = RS.gen_obs_ts()
                    
                    d_int = dict()
                    for int_var in RS.confintvar.values():
                        i = RS.intervene(int_var, nsample_int, random.uniform(5, 10))
                        d_int[int_var] = i[int_var]
                        d_int[int_var].plot_timeseries(resfolder + '/interv_' + int_var + '.png')

                
                    GT = RS.get_SCM()
                    
                    d_obs.plot_timeseries(resfolder + '/obs_data.png')
                    
                    RS.ts_dag(withHidden = True, save_name = resfolder + '/gt_complete')
                    RS.ts_dag(withHidden = False, save_name = resfolder + '/gt')                  
            
            
                    #########################################################################################################################
                    # FPCMCI
                    fpcmci = FPCMCI(deepcopy(d_obs),
                                    f_alpha = f_alpha, 
                                    alpha = alpha, 
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
                        remove_directory(os.getcwd() + '/' + resfolder)
                        continue
                    
                    
                    #########################################################################################################################
                    # PCMCI
                    pcmci = PCMCI(deepcopy(d_obs),
                                    min_lag = min_lag, 
                                    max_lag = max_lag, 
                                    val_condtest = GPDC(significance = 'analytic'),
                                    verbosity = CPLevel.INFO,
                                    alpha = alpha, 
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
                                      alpha = alpha, 
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
                    with open(os.getcwd() + "/" + 'results/' + resdir + '/error.txt', 'a') as f:
                        f.write(str(e))
                    remove_directory(os.getcwd() + "/" + resfolder)
                    continue


            #########################################################################################################################
            # SAVE
            res = {
                Algo.CAnDOIT: {"time":candoit_time, "scm":get_correct_SCM(GT, candoit_cm.get_SCM())},
                Algo.FPCMCI: {"time":fpcmci_time, "scm":get_correct_SCM(GT, fpcmci_cm.get_SCM())},
                Algo.PCMCI: {"time":pcmci_time, "scm":get_correct_SCM(GT, pcmci_cm.get_SCM())},
            }
            save_result(res)
            
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