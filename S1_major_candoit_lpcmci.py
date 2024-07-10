from copy import deepcopy
import json
import os
import random
from tigramite.independence_tests.gpdc_torch import GPDCtorch as GPDC
# from tigramite.independence_tests.gpdc import GPDC
from causalflow.CPrinter import CPLevel
from causalflow.basics.constants import ImageExt
from causalflow.causal_discovery.baseline.LPCMCI import LPCMCI
from causalflow.causal_discovery.FPCMCI import FPCMCI
from causalflow.causal_discovery.CAnDOIT_pcmciplus import CAnDOIT as CAnDOIT_pcmciplus
from causalflow.causal_discovery.CAnDOIT_lpcmci import CAnDOIT as CAnDOIT_lpcmci
from causalflow.causal_discovery.baseline.PCMCI import PCMCI
from causalflow.causal_discovery.baseline.PCMCIplus import PCMCIplus
from causalflow.preprocessing.data import Data
from causalflow.selection_methods.TE import TE, TEestimator
from causalflow.random_system.RandomDAG import NoiseType, RandomDAG
from pathlib import Path
import traceback

from time import time
from datetime import timedelta
from res_statistics_new import *
import gc
import shutil
import ast

ALGO_RES = {'done': False,
            Metric.TIME.value : None,
            Metric.FN.value : None,
            Metric.TN.value : None,
            Metric.FP.value : None,
            Metric.TP.value : None,
            Metric.FPR.value : None,
            Metric.TPR.value : None,
            Metric.FNR.value : None,
            Metric.TNR.value : None,
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
             Algo.PCMCI.value : deepcopy(ALGO_RES),
             Algo.PCMCIplus.value : deepcopy(ALGO_RES),
             Algo.LPCMCI.value : deepcopy(ALGO_RES),
             Algo.FPCMCI.value : deepcopy(ALGO_RES),
             Algo.CAnDOIT.value : deepcopy(ALGO_RES),
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


# FIXME: to be changed
def get_ambiguous_link(scm):
    spurious = list()
    for exp_s in EXP_SPURIOUS_LINKS:
        if exp_s['t'] in scm and (exp_s['s'], -abs(exp_s['lag'])) in scm[exp_s['t']]:
            spurious.append(exp_s)
            
    return spurious
        
        
def fill_res(res, r):
    res['done'] = True
    res[Metric.TIME.value] = r["time"]
    res[Metric.FN.value] = RandomDAG.get_FN(GT, r["scm"])
    res[Metric.TN.value] = RandomDAG.get_TN(GT, min_lag, max_lag, r["scm"])
    res[Metric.FP.value] = RandomDAG.get_FP(GT, r["scm"])
    res[Metric.TP.value] = RandomDAG.get_TP(GT, r["scm"])
    res[Metric.FPR.value] = RandomDAG.FPR(GT, min_lag, max_lag, r["scm"])
    res[Metric.TPR.value] = RandomDAG.TPR(GT, r["scm"])
    res[Metric.FNR.value] = RandomDAG.FNR(GT, r["scm"])
    res[Metric.TNR.value] = RandomDAG.TNR(GT, min_lag, max_lag, r["scm"])
    res[Metric.PREC.value] = RandomDAG.precision(GT, r["scm"])
    res[Metric.RECA.value] = RandomDAG.recall(GT, r["scm"])
    res[Metric.F1SCORE.value] = RandomDAG.f1_score(GT, r["scm"])
    res[Metric.SHD.value] = RandomDAG.shd(GT, r["scm"])
    res[jWord.SCM.value] = str(r["scm"])
    spurious_links = get_ambiguous_link(r["scm"])
    res[jWord.SpuriousLinks.value] = str(spurious_links)
    res[Metric.N_ESPU.value] = len(spurious_links)
    res[Metric.N_EqDAG.value] = 2**len(spurious_links)
    
    
if __name__ == '__main__':
    # Simulation params
    resdir = "S1_major_candoit_lpcmci"
    f_alpha = 0.5
    alpha = 0.05
    nfeature = range(7, 15)
    nrun = 25
    
    # RandomDAG params 
    nsample_obs = 1250
    nsample_int = 250
    min_c = 0.1
    max_c = 0.5
    link_density = 3
    max_exp = 2
    functions = ['', 'sin', 'cos', 'abs']
    operators = ['+', '-', '*']
    n_hidden_confounders = 1
    
    for n in nfeature:
        for nr in range(nrun):
            nr = str(nr)
            
            #########################################################################################################################
            # DATA
            while True:
                try:
                    # Check if the file exists
                    Path(os.getcwd() + "/results/" + resdir).mkdir(parents=True, exist_ok=True)
                    filename = os.getcwd() + "/results/" + resdir + "/" + str(n) + ".json"
                    resfolder = 'results/' + resdir + '/' + str(n) + '/' + nr
                    if os.path.exists(filename):
                        # File exists, load its contents into a dictionary
                        with open(filename, 'r') as file:
                            data = json.load(file)
                    else:
                        # File does not exist, create a new dictionary
                        data = {}
                    if nr in data and data[nr]['done']: 
                        break
                    elif nr in data and not data[nr]['done']:        
                        min_lag = int(data[nr]['min_lag'])
                        max_lag = int(data[nr]['max_lag'])
                        d_obs = Data(os.getcwd() + '/' + resfolder + '/obs_data.csv')
                        d_int = dict()
                        # List all files in the folder and filter files that start with 'interv_' and end with '.csv'
                        intvars = ast.literal_eval(data[nr]['InterventionVariables'])
                        for v in intvars: d_int[v] = Data(os.getcwd() + '/' + resfolder + f'/interv_{v}.csv')
                        
                        EQUATIONS = data[nr]["equations"]
                        COEFF_RANGE = ast.literal_eval(data[nr]["coeff_range"])
                        NOISE_CONF = data[nr]["noise_config"]
                        GT = ast.literal_eval(data[nr]['GT'])
                        CONFOUNDERS = ast.literal_eval(data[nr][jWord.Confounders.value])
                        HIDDEN_CONFOUNDERS = ast.literal_eval(data[nr][jWord.HiddenConfounders.value])
                        INT_VARS = list(d_int.keys())
                        EXP_SPURIOUS_LINKS = ast.literal_eval(data[nr]['ExpectedSpuriousLinks'])
                    else:
                        # File does not exist, create a new dictionary
                        data[nr] = deepcopy(EMPTY_RES)
                        
                        
                        # FIXME: add this also in the if and load data from the csv files  
                        min_lag = random.randint(0, 1)
                        max_lag = random.randint(2, 5)
                        os.makedirs(resfolder, exist_ok = True)
                        # res_tmp = deepcopy(EMPTY_RES)
                        
                        # Noise params 
                        noise_param = random.uniform(0.5, 2)
                        noise_uniform = (NoiseType.Uniform, -noise_param, noise_param)
                        noise_gaussian = (NoiseType.Gaussian, 0, noise_param)
                        RS = RandomDAG(nvars = n, nsamples = nsample_obs + nsample_int, 
                                    link_density = link_density, coeff_range = (min_c, max_c), max_exp = max_exp, 
                                    min_lag = min_lag, max_lag = max_lag, noise_config = random.choice([noise_uniform, noise_gaussian]),
                                    functions = functions, operators = operators, n_hidden_confounders = n_hidden_confounders)
                        RS.gen_equations()
                        RS.ts_dag(withHidden = True, save_name = resfolder + '/gt_complete')
                        RS.ts_dag(withHidden = False, save_name = resfolder + '/gt')       

                        d_obs = RS.gen_obs_ts()
                        d_obs.plot_timeseries(resfolder + '/obs_data.png')
                        d_obs.save_csv(resfolder + '/obs_data.csv')
                        
                        # This strategy allows to pick one variable for each confounder
                        # (1) if the confounder is lagged then it takes the only available option
                        # (2) if the confounder is contemporaneous then, if exists, it takes a variable that has been already chosen in the step (1)
                        # (3) if a variable that has been already chosen in the step (1) does not exist, random choice among available options
                        d_int = dict()
                        intvars = [RS.potentialIntervention[h]['vars'][0] for h in RS.potentialIntervention if RS.potentialIntervention[h]['type'] == 'lagged']
                        for h in RS.potentialIntervention:
                            if RS.potentialIntervention[h]['type'] == 'contemporaneous':
                                varFound = False
                                for v in RS.potentialIntervention[h]['vars']:
                                    if v in intvars: 
                                        varFound = True
                                        break
                                if not varFound: intvars.append(random.choice(RS.potentialIntervention[h]['vars']))
                        for intvar in intvars:
                            i = RS.intervene(intvar, nsample_int, random.uniform(5, 10))
                            d_int[intvar] = i[intvar]
                            d_int[intvar].plot_timeseries(resfolder + '/interv_' + intvar + '.png')
                            d_int[intvar].save_csv(resfolder + '/interv_' + intvar + '.csv')
                        
                        EQUATIONS = RS.print_equations()
                        COEFF_RANGE = RS.coeff_range 
                        NOISE_CONF = (RS.noise_config[0].value, RS.noise_config[1], RS.noise_config[2])
                        GT = RS.get_SCM()
                        CONFOUNDERS = RS.confounders
                        HIDDEN_CONFOUNDERS = list(RS.confounders.keys())
                        INT_VARS = list(d_int.keys())
                        EXP_SPURIOUS_LINKS= RS.expected_spurious_links
                    
                                
                    #########################################################################################################################
                    # LPCMCI
                    if Algo.LPCMCI.value not in data[nr] or (Algo.LPCMCI.value in data[nr] and not data[nr][Algo.LPCMCI.value]['done']):
                        lpcmci = LPCMCI(deepcopy(d_obs),
                                        min_lag = min_lag, 
                                        max_lag = max_lag, 
                                        val_condtest = GPDC(significance = 'analytic'),
                                        verbosity = CPLevel.INFO,
                                        alpha = alpha, 
                                        neglect_only_autodep = False,
                                        resfolder = resfolder + "/lpcmci")
                        
                        new_start = time()
                        lpcmci_cm = lpcmci.run()
                        elapsed_lpcmci = time() - new_start
                        lpcmci_time = str(timedelta(seconds = elapsed_lpcmci))
                        print(lpcmci_time)
                        lpcmci_cm.ts_dag(save_name = lpcmci.ts_dag_path, img_extention = ImageExt.PNG)
                        lpcmci_cm.ts_dag(save_name = lpcmci.ts_dag_path, img_extention = ImageExt.PDF)
                        lpcmci.save()
                        gc.collect()
                        
                        # if len(get_ambiguous_link(lpcmci_cm.get_SCM())) == 0: 
                        #     gc.collect()
                        #     remove_directory(os.getcwd() + '/' + resfolder)
                        #     continue
                        # else:
                        res = deepcopy(ALGO_RES)
                        fill_res(res, {"time":lpcmci_time, "scm":get_correct_SCM(GT, lpcmci_cm.get_SCM())})
                                
                        data[nr]['done'] = False
                        data[nr]["min_lag"] = str(min_lag)
                        data[nr]["max_lag"] = str(max_lag)
                        data[nr]["equations"] = str(EQUATIONS)
                        data[nr]["coeff_range"] = str(COEFF_RANGE)
                        data[nr]["noise_config"] = str(NOISE_CONF)
                        data[nr][jWord.GT.value] = str(GT)
                        data[nr][jWord.Confounders.value] = str(CONFOUNDERS)
                        data[nr][jWord.HiddenConfounders.value] = str(HIDDEN_CONFOUNDERS)
                        data[nr][jWord.InterventionVariables.value] = str(INT_VARS)
                        data[nr][jWord.ExpectedSpuriousLinks.value] = str(EXP_SPURIOUS_LINKS)
                        data[nr][jWord.N_GSPU.value] = len(EXP_SPURIOUS_LINKS)
                            
                        data[nr][Algo.LPCMCI.value] = res
                            
                        # Save the dictionary back to a JSON file
                        with open(filename, 'w') as file:
                            json.dump(data, file)
            
                    
                    #########################################################################################################################
                    # CAnDOIT
                    if Algo.CAnDOIT.value not in data[nr] or (Algo.CAnDOIT.value in data[nr] and not data[nr][Algo.CAnDOIT.value]['done']):
                        new_d_obs = deepcopy(d_obs)
                        new_d_obs.d = new_d_obs.d[:-nsample_int]
                        candoit = CAnDOIT_lpcmci(new_d_obs, 
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
                        candoit_cm.ts_dag(save_name = candoit.ts_dag_path, img_extention = ImageExt.PNG)
                        candoit_cm.ts_dag(save_name = candoit.ts_dag_path, img_extention = ImageExt.PDF)
                        gc.collect()
                    
                        res = deepcopy(ALGO_RES)
                        fill_res(res, {"time":candoit_time, "scm":get_correct_SCM(GT, candoit_cm.get_SCM())})
                        
                        data[nr][Algo.CAnDOIT.value] = res
                            
                        # Save the dictionary back to a JSON file
                        with open(filename, 'w') as file:
                            json.dump(data, file)
                            
                    data[nr]['done'] = True
                    # Save the dictionary back to a JSON file
                    with open(filename, 'w') as file:
                        json.dump(data, file)
                    break
                    
                except Exception as e:
                    traceback_info = traceback.format_exc()
                    with open(os.getcwd() + '/results/' + resdir + '/error.txt', 'a') as f:
                        f.write("Exception occurred: " + str(e) + "\n")
                        f.write("Traceback:\n" + traceback_info + "\n")
                    remove_directory(os.getcwd() + "/" + resfolder)
                    
                    filename = os.getcwd() + "/results/" + resdir + "/" + str(n) + ".json"
                    if os.path.exists(filename):
                        with open(filename, 'r') as file:
                            data = json.load(file)
                            if nr in data: 
                                data.pop(nr)
                                json.dump(data, file)           
                    continue