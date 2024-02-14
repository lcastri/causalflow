from copy import deepcopy
import json
import os
import random
from tigramite.independence_tests.gpdc_torch import GPDCtorch as GPDC
from connectingdots.CPrinter import CPLevel
from connectingdots.causal_discovery.CAnDOIT_cont import CAnDOIT as CAnDOIT_cont
from connectingdots.preprocessing.data import Data
from connectingdots.selection_methods.TE import TE, TEestimator
from connectingdots.random_system.RandomDAG import NoiseType, RandomDAG
from pathlib import Path
import traceback

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
             Algo.CAnDOITCont.value : deepcopy(ALGO_RES),
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
    res_tmp["equations"] = str(RS.print_equations())
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
    nsample = 1500
    resdir = "CAnDOIT_bestlength"
    f_alpha = 0.5
    alpha = 0.05
    min_lag = 1
    max_lag = 2
    min_c = 0.1
    max_c = 0.5
    nrun = 25
    delta_perc = 0.05
    obs_perc = 1.0
    
    for nr in range(nrun):
        
        resfolder = 'results/' + resdir + '/' + str(nr)
        os.makedirs(resfolder, exist_ok = True)
        
        coeff_sign = random.choice([-1, 1])
        noise_param = random.uniform(0.5, 2)
        noise_uniform = (NoiseType.Uniform, -noise_param, noise_param)
        noise_gaussian = (NoiseType.Gaussian, 0, noise_param)
        RS = RandomDAG(nvars = 10, nsamples = nsample, 
                       max_terms = 3, coeff_range = (coeff_sign*min_c, coeff_sign*max_c), max_exp = 2, 
                       min_lag = min_lag, max_lag = max_lag, noise_config = random.choice([noise_uniform, noise_gaussian]),
                       functions = ['', 'sin', 'cos', 'abs'], operators=['+', '-', '*'], n_hidden_confounders = 1)
    
        RS.gen_equations()
        d_obs = RS.gen_obs_ts()
    
        d_int = dict()
        for int_var in RS.confintvar.values():
            i = RS.intervene(int_var, nsample, random.uniform(5, 10))
            d_int[int_var] = i[int_var]
            d_int[int_var].plot_timeseries(resfolder + '/interv_' + int_var + '.png')
                
        GT = RS.get_SCM()
                    
        d_obs.plot_timeseries(resfolder + '/obs_data.png')
                    
        RS.ts_dag(withHidden = True, save_name = resfolder + '/gt_complete')
        RS.ts_dag(withHidden = False, save_name = resfolder + '/gt')                  
    
    
        while obs_perc - delta_perc > 0:
            res_tmp = deepcopy(EMPTY_RES)
            obs_perc = round(obs_perc - delta_perc, 2)
            int_perc = round(1 - obs_perc, 2)
            obs_length = int(nsample * obs_perc)
            int_length = int(nsample * int_perc)     
                    
            tmp_df_obs = deepcopy(d_obs.d)
            tmp_d_obs = Data(tmp_df_obs[:obs_length])
            
            tmp_d_int = dict()
            for int_var, int_d in d_int.items():
                tmp_df_int = deepcopy(int_d.d)
                tmp_d_int[int_var] = Data(tmp_df_int[:int_length])
            
            #########################################################################################################################
            # CAnDOIT
            candoit_cont = CAnDOIT_cont(tmp_d_obs, 
                                        tmp_d_int,
                                        f_alpha = f_alpha, 
                                        alpha = alpha, 
                                        min_lag = min_lag, 
                                        max_lag = max_lag, 
                                        sel_method = TE(TEestimator.Gaussian), 
                                        val_condtest = GPDC(significance = 'analytic'),
                                        verbosity = CPLevel.INFO,
                                        neglect_only_autodep = False,
                                        resfolder = resfolder + "/" + str(obs_perc) + "_" + str(int_perc),
                                        plot_data = False,
                                        exclude_context = True)
                    
            new_start = time()
            candoit_cont_cm = candoit_cont.run()
            elapsed_candoit_cont = time() - new_start
            candoit_cont_time = str(timedelta(seconds = elapsed_candoit_cont))
            print(candoit_cont_time)
            candoit_cont.timeseries_dag()
            gc.collect()
            
                    
                # except Exception as e:
                #     traceback_info = traceback.format_exc()
                #     with open(os.getcwd() + "/" + 'results/' + resdir + '/error.txt', 'a') as f:
                #         f.write("Exception occurred: " + str(e) + "\n")
                #         f.write("Traceback:\n" + traceback_info + "\n")
                #     remove_directory(os.getcwd() + "/" + resfolder)
                #     continue


            #########################################################################################################################
            # SAVE
            res = {
                Algo.CAnDOITCont: {"time":candoit_cont_time, "scm":get_correct_SCM(GT, candoit_cont_cm.get_SCM())},
            }
            save_result(res)
                
            Path(os.getcwd() + "/results/" + resdir).mkdir(parents=True, exist_ok=True)
            filename = os.getcwd() + "/results/" + resdir + '/' + str(nr) + ".json"
                
            # Check if the file exists
            if os.path.exists(filename):
                # File exists, load its contents into a dictionary
                with open(filename, 'r') as file:
                    data = json.load(file)
            else:
                # File does not exist, create a new dictionary
                data = {}

            # Modify the dictionary
            data[str(obs_perc) + "_" + str(int_perc)] = res_tmp

            # Save the dictionary back to a JSON file
            with open(filename, 'w') as file:
                json.dump(data, file)
            res_tmp.clear()