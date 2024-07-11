from pathlib import Path
import random
import os
from tigramite.independence_tests.gpdc_torch import GPDCtorch as GPDC
from causalflow.CPrinter import CPLevel
from causalflow.causal_discovery.CAnDOIT_pcmciplus import CAnDOIT as CAnDOIT_pcmciplus
from causalflow.causal_discovery.CAnDOIT_lpcmci import CAnDOIT as CAnDOIT_lpcmci
from causalflow.causal_discovery.baseline.LPCMCI import LPCMCI
from causalflow.causal_discovery.baseline.PCMCIplus import PCMCIplus
from causalflow.random_system.RandomDAG import NoiseType, RandomDAG
from causalflow.selection_methods.TE import TE, TEestimator
from causalflow.basics.constants import ImageExt
import numpy as np

from time import time
from datetime import timedelta

if __name__ == '__main__':   
    resfolder = 'results/CAnDOIT'
    Path(os.getcwd() + '/' + resfolder).mkdir(parents=True, exist_ok=True)
           
    min_lag = 0
    max_lag = 2
    # min_lag = random.randint(0, 1)
    # max_lag = random.randint(2, 5)
    nsample_obs = 1250
    nsample_int = 250
    functions = ['']
    operators = ['+', '-']
    # functions = ['', 'sin', 'cos', 'abs']
    # operators = ['+', '-', '*']
    
    # Noise params 
    noise_param = random.uniform(0.5, 2)
    noise_uniform = (NoiseType.Uniform, -noise_param, noise_param)
    noise_gaussian = (NoiseType.Gaussian, 0, noise_param)
    RS = RandomDAG(nvars = 3, nsamples = nsample_obs + nsample_int, 
                link_density = 3, coeff_range = (0.1, 0.5), max_exp = 2, 
                min_lag = min_lag, max_lag = max_lag, noise_config = random.choice([noise_uniform, noise_gaussian]),
                functions = functions, operators = operators, n_hidden_confounders = 1)
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
                        
    
    pcmciplus = PCMCIplus(d_obs, 
                          min_lag = min_lag, 
                          max_lag = max_lag, 
                          val_condtest = GPDC(significance = 'analytic'),
                          verbosity = CPLevel.DEBUG,
                          alpha = 0.05, 
                          resfolder = resfolder + "/pcmciplus",
                          neglect_only_autodep = True)
    
    new_start = time()
    pcmciplus_cm = pcmciplus.run()
    elapsed_newFPCMCI = time() - new_start
    print(str(timedelta(seconds = elapsed_newFPCMCI)))
    pcmciplus_cm.ts_dag(save_name = pcmciplus.ts_dag_path, img_extention = ImageExt.PNG, min_width=3, max_width=5)
    pcmciplus_cm.ts_dag(save_name = pcmciplus.ts_dag_path, img_extention = ImageExt.PDF, min_width=3, max_width=5)
    
    
    lpcmci = LPCMCI(d_obs, 
                    min_lag = min_lag, 
                    max_lag = max_lag, 
                    val_condtest = GPDC(significance = 'analytic'),
                    verbosity = CPLevel.DEBUG,
                    alpha = 0.05, 
                    resfolder = resfolder + "/lpcmci",
                    neglect_only_autodep = True)
    
    new_start = time()
    lpcmci_cm = lpcmci.run()
    elapsed_newFPCMCI = time() - new_start
    print(str(timedelta(seconds = elapsed_newFPCMCI)))
    lpcmci_cm.ts_dag(save_name = lpcmci.ts_dag_path, img_extention = ImageExt.PNG, min_width=3, max_width=5)
    lpcmci_cm.ts_dag(save_name = lpcmci.ts_dag_path, img_extention = ImageExt.PDF, min_width=3, max_width=5)
    
    candoit_pcmciplus = CAnDOIT_pcmciplus(d_obs, 
                      d_int,
                      f_alpha = 0.5, 
                      alpha = 0.05, 
                      min_lag = min_lag, 
                      max_lag = max_lag, 
                      sel_method = TE(TEestimator.Gaussian), 
                      val_condtest = GPDC(significance = 'analytic'),
                      verbosity = CPLevel.DEBUG,
                      neglect_only_autodep = True,
                      plot_data = False,
                      exclude_context = True,
                      resfolder = resfolder + "/candoit_pcmciplus")
    
    new_start = time()
    candoit_pcmciplus_cm = candoit_pcmciplus.run(nofilter=True)
    elapsed_newFPCMCI = time() - new_start
    print(str(timedelta(seconds = elapsed_newFPCMCI)))
    candoit_pcmciplus_cm.ts_dag(save_name = candoit_pcmciplus.ts_dag_path, img_extention = ImageExt.PNG, min_width=3, max_width=5)
    candoit_pcmciplus_cm.ts_dag(save_name = candoit_pcmciplus.ts_dag_path, img_extention = ImageExt.PDF, min_width=3, max_width=5)
    
    candoit_lpcmci = CAnDOIT_lpcmci(d_obs, 
                      d_int,
                      f_alpha = 0.5, 
                      alpha = 0.05, 
                      min_lag = min_lag, 
                      max_lag = max_lag, 
                      sel_method = TE(TEestimator.Gaussian), 
                      val_condtest = GPDC(significance = 'analytic'),
                      verbosity = CPLevel.DEBUG,
                      neglect_only_autodep = True,
                      plot_data = False,
                      exclude_context = True,
                      resfolder = resfolder + "/candoit_lpcmci")
    
    new_start = time()
    candoit_lpcmci_cm = candoit_lpcmci.run(nofilter=True)
    elapsed_newFPCMCI = time() - new_start
    print(str(timedelta(seconds = elapsed_newFPCMCI)))
    candoit_lpcmci_cm.ts_dag(save_name = candoit_lpcmci.ts_dag_path, img_extention = ImageExt.PNG, min_width=3, max_width=5)
    candoit_lpcmci_cm.ts_dag(save_name = candoit_lpcmci.ts_dag_path, img_extention = ImageExt.PDF, min_width=3, max_width=5)
    
    print("Expected bidirected links")
    for link in RS.expected_bidirected_links:
        target = list(link.keys())[0]
        source, lag = link[target]
        print(f"{(source, -lag)} <-> {target}")
        
    print("Found bidirected links")
        
        