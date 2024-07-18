from copy import deepcopy
import json
import os
import random
# from tigramite.independence_tests.gpdc_torch import GPDCtorch as GPDC
from tigramite.independence_tests.parcorr import ParCorr
# from tigramite.independence_tests.gpdc import GPDC
from causalflow.CPrinter import CPLevel
from causalflow.basics.constants import ImageExt
from causalflow.causal_discovery.baseline.LPCMCI import LPCMCI
# from causalflow.causal_discovery.FPCMCI import FPCMCI
# from causalflow.causal_discovery.CAnDOIT_pcmciplus import CAnDOIT as CAnDOIT_pcmciplus
from causalflow.causal_discovery.CAnDOIT_lpcmci import CAnDOIT
# from causalflow.causal_discovery.baseline.PCMCI import PCMCI
# from causalflow.causal_discovery.baseline.PCMCIplus import PCMCIplus
from causalflow.preprocessing.data import Data
from causalflow.selection_methods.TE import TE, TEestimator
from causalflow.random_system.RandomGraph import NoiseType, RandomGraph
import causalflow.basics.metrics as metrics
from pathlib import Path
import traceback
import timeout_decorator

from time import time
from datetime import timedelta
from res_statistics_new import *
import gc
import shutil
import ast

TIMEOUT = 5*60

def remove_directory(directory_path):
    try:
        shutil.rmtree(directory_path)
        print(f"Directory '{directory_path}' and its content have been removed.")
    except OSError as e:
        print(f"Error: {e}")


def get_ambiguous_link(scm):
    canContinue = False
    potentialIntervention = set()
    amb_links = list()
    
    for t, sources in scm.items():
        for s in sources:
            if sources[s] == 'o-o' or sources[s] == 'o->':
                canContinue = True
                potentialIntervention.add(s[0])
                amb_links.append((s, sources[s], t))
            
    return canContinue, amb_links, potentialIntervention


@timeout_decorator.timeout(TIMEOUT)
def run_algo(algo, name):
    if name == Algo.LPCMCI.value:
        return algo.run()
    elif name == Algo.CAnDOIT.value:
        return algo.run(remove_unneeded=False, nofilter=True)

        
def fill_res(r):
    res = {}
    res['done'] = True
    res[Metric.TIME.value] = r["time"]
    res["adj"] = str(r["adj"])
    res[f"adj_{Metric.FN.value}"] = metrics.get_FN(GT_ADJ, r["adj"])
    res[f"adj_{Metric.TN.value}"] = metrics.get_TN(GT_ADJ, min_lag, max_lag, r["adj"])
    res[f"adj_{Metric.FP.value}"] = metrics.get_FP(GT_ADJ, r["adj"])
    res[f"adj_{Metric.TP.value}"] = metrics.get_TP(GT_ADJ, r["adj"])
    res[f"adj_{Metric.FPR.value}"] = metrics.FPR(GT_ADJ, min_lag, max_lag, r["adj"])
    res[f"adj_{Metric.TPR.value}"] = metrics.TPR(GT_ADJ, r["adj"])
    res[f"adj_{Metric.FNR.value}"] = metrics.FNR(GT_ADJ, r["adj"])
    res[f"adj_{Metric.TNR.value}"] = metrics.TNR(GT_ADJ, min_lag, max_lag, r["adj"])
    res[f"adj_{Metric.PREC.value}"] = metrics.precision(GT_ADJ, r["adj"])
    res[f"adj_{Metric.RECA.value}"] = metrics.recall(GT_ADJ, r["adj"])
    res[f"adj_{Metric.F1SCORE.value}"] = metrics.f1_score(GT_ADJ, r["adj"])
    res[f"adj_{Metric.SHD.value}"] = metrics.shd(GT_ADJ, r["adj"])
    res["graph"] = str(r["graph"])
    res[f"graph_{Metric.FN.value}"] = metrics.get_FN(GT_GRAPH, r["graph"], alsoOrient=True)
    res[f"graph_{Metric.TN.value}"] = metrics.get_TN(GT_GRAPH, min_lag, max_lag, r["graph"], alsoOrient=True)
    res[f"graph_{Metric.FP.value}"] = metrics.get_FP(GT_GRAPH, r["graph"], alsoOrient=True)
    res[f"graph_{Metric.TP.value}"] = metrics.get_TP(GT_GRAPH, r["graph"], alsoOrient=True)
    res[f"graph_{Metric.FPR.value}"] = metrics.FPR(GT_GRAPH, min_lag, max_lag, r["graph"], alsoOrient=True)
    res[f"graph_{Metric.TPR.value}"] = metrics.TPR(GT_GRAPH, r["graph"], alsoOrient=True)
    res[f"graph_{Metric.FNR.value}"] = metrics.FNR(GT_GRAPH, r["graph"], alsoOrient=True)
    res[f"graph_{Metric.TNR.value}"] = metrics.TNR(GT_GRAPH, min_lag, max_lag, r["graph"], alsoOrient=True)
    res[f"graph_{Metric.PREC.value}"] = metrics.precision(GT_GRAPH, r["graph"], alsoOrient=True)
    res[f"graph_{Metric.RECA.value}"] = metrics.recall(GT_GRAPH, r["graph"], alsoOrient=True)
    res[f"graph_{Metric.F1SCORE.value}"] = metrics.f1_score(GT_GRAPH, r["graph"], alsoOrient=True)
    res[f"graph_{Metric.SHD.value}"] = metrics.shd(GT_GRAPH, r["graph"], alsoOrient=True)
    _, ambiguous_links, _ = get_ambiguous_link(r["graph"])
    res["ambiguous_links"] = str(ambiguous_links)
    res[Metric.UNCERTAINTY.value] = len(ambiguous_links)
    res[Metric.PAGSIZE.value] = 2**len(ambiguous_links)
    return res
    
    
if __name__ == '__main__':
    # Simulation params
    resdir = "AIS_major/AIS_major_S2"
    alpha = 0.05
    nfeature = range(5, 13)
    nrun = 25
    
    # RandomDAG params 
    nsample_obs = 1250
    nsample_int = 250
    # nsample_obs = 750
    # nsample_int = 150
    min_c = 0.1
    max_c = 0.5
    link_density = 3
    max_exp = 2
    functions = ['']
    operators = ['+', '-', '*']
    
    for n in nfeature:
        for nr in range(nrun):
            nr = str(nr)
            
            #########################################################################################################################
            # DATA
            while True:
                # try:
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
                        potentialIntervention = ast.literal_eval(data[nr]['potential_interventions'])
                        for v in potentialIntervention:
                            if os.path.exists(os.getcwd() + '/' + resfolder + f'/interv_{v}.csv'):
                                d_int[v] = Data(os.getcwd() + '/' + resfolder + f'/interv_{v}.csv')
                        
                        EQUATIONS = data[nr]["equations"]
                        COEFF_RANGE = ast.literal_eval(data[nr]["coeff_range"])
                        NOISE_CONF = data[nr]["noise_config"]
                        GT_ADJ = ast.literal_eval(data[nr]['adj'])
                        GT_GRAPH = ast.literal_eval(data[nr]['graph'])
                        CONFOUNDERS = ast.literal_eval(data[nr]["confounders"])
                        HIDDEN_CONFOUNDERS = ast.literal_eval(data[nr]["hidden_confounders"])
                        EXPECTED_AMBIGUOUS_LINKS = ast.literal_eval(data[nr]["expected_ambiguous_links"])
                        EXPECTED_UNCERTAINTY = ast.literal_eval(data[nr]["expected_uncertainty"])
                        INT_VARS = list(d_int.keys()) if len(d_int) > 0 else None
                    else:
                        # File does not exist, create a new dictionary
                        data[nr] = {}
                        
                        
                        n_hidden_confounders = random.randint(1, 2)
                        min_lag = 0
                        max_lag = random.randint(1, 3)
                        os.makedirs(resfolder, exist_ok = True)
                        
                        # Noise params 
                        noise_param = random.uniform(0.5, 2)
                        noise_uniform = (NoiseType.Uniform, -noise_param, noise_param)
                        noise_gaussian = (NoiseType.Gaussian, 0, noise_param)
                        noise_weibull = (NoiseType.Weibull, 2, 1)
                        RS = RandomGraph(nvars = n, nsamples = nsample_obs + nsample_int, 
                                       link_density = link_density, coeff_range = (min_c, max_c), max_exp = max_exp, 
                                       min_lag = min_lag, max_lag = max_lag, noise_config = random.choice([noise_uniform, noise_gaussian, noise_weibull]),
                                       functions = functions, operators = operators, n_hidden_confounders = n_hidden_confounders)
                        RS.gen_equations()
                        RS.ts_dag(withHidden = True, save_name = resfolder + '/gt_complete')
                        RS.ts_dag(withHidden = False, save_name = resfolder + '/gt')       

                        d_obs = RS.gen_obs_ts()
                        d_obs.plot_timeseries(resfolder + '/obs_data.png')
                        d_obs.save_csv(resfolder + '/obs_data.csv')

                        
                        EQUATIONS = RS.print_equations()
                        COEFF_RANGE = RS.coeff_range 
                        NOISE_CONF = (RS.noise_config[0].value, RS.noise_config[1], RS.noise_config[2])
                        GT_ADJ = RS.get_Adj()
                        GT_GRAPH = RS.get_DPAG()
                        CONFOUNDERS = RS.confounders
                        HIDDEN_CONFOUNDERS = list(RS.confounders.keys())
                        _, amb_links, _ = get_ambiguous_link(GT_GRAPH)
                        EXPECTED_AMBIGUOUS_LINKS = amb_links
                        EXPECTED_UNCERTAINTY = len(amb_links)
                        INT_VARS = None
                    
                                
                    #########################################################################################################################
                    # LPCMCI
                    if Algo.LPCMCI.value not in data[nr] or (Algo.LPCMCI.value in data[nr] and not data[nr][Algo.LPCMCI.value]['done']):
                        lpcmci = LPCMCI(deepcopy(d_obs),
                                        min_lag = 0, 
                                        max_lag = max_lag, 
                                        sys_context = [],
                                        val_condtest = ParCorr(significance = 'analytic'),
                                        verbosity = CPLevel.INFO,
                                        alpha = alpha,
                                        neglect_only_autodep = False,
                                        resfolder = resfolder + "/lpcmci")
                        
                        try:
                            new_start = time()
                            lpcmci_cm = run_algo(lpcmci, 'lpcmci')
                            # lpcmci_cm = lpcmci.run()
                            elapsed_lpcmci = time() - new_start
                            lpcmci_time = str(timedelta(seconds = elapsed_lpcmci))
                            print(lpcmci_time)
                            lpcmci_cm.ts_dag(save_name = lpcmci.ts_dag_path, img_extention = ImageExt.PNG, node_size=6, min_width=2, max_width=5, x_disp=1)
                            lpcmci_cm.ts_dag(save_name = lpcmci.ts_dag_path, img_extention = ImageExt.PDF, node_size=6, min_width=2, max_width=5, x_disp=1)
                            lpcmci.save()
                            gc.collect()
                            
                            canContinue, _, potentialIntervention = get_ambiguous_link(lpcmci_cm.get_Graph())
                            
                            if not canContinue: 
                                gc.collect()
                                remove_directory(os.getcwd() + '/' + resfolder)
                                continue
                            else:
                                res = fill_res({"time": lpcmci_time, 
                                                "adj": lpcmci_cm.get_Adj(), 
                                                "graph": lpcmci_cm.get_Graph()})
                                
                                data[nr] = {}
                                        
                                data[nr]['done'] = False
                                data[nr]["min_lag"] = str(min_lag)
                                data[nr]["max_lag"] = str(max_lag)
                                data[nr]["equations"] = str(EQUATIONS)
                                data[nr]["coeff_range"] = str(COEFF_RANGE)
                                data[nr]["noise_config"] = str(NOISE_CONF)
                                data[nr]["adj"] = str(GT_ADJ)
                                data[nr]["graph"] = str(GT_GRAPH)
                                data[nr]["potential_interventions"] = str(potentialIntervention)
                                data[nr]["confounders"] = str(CONFOUNDERS)
                                data[nr]["hidden_confounders"] = str(HIDDEN_CONFOUNDERS)
                                data[nr]["expected_ambiguous_links"] = str(EXPECTED_AMBIGUOUS_LINKS)
                                data[nr]["expected_uncertainty"] = str(EXPECTED_UNCERTAINTY)
                                
                                d_int = dict()
                                for intvar in potentialIntervention:
                                    i = RS.intervene(intvar, nsample_int, random.uniform(5, 10), d_obs.d)
                                    d_int[intvar] = i[intvar]
                                    d_int[intvar].plot_timeseries(resfolder + '/interv_' + intvar + '.png')
                                    d_int[intvar].save_csv(resfolder + '/interv_' + intvar + '.csv')
                                INT_VARS = list(d_int.keys())
                                data[nr]["intervention_variables"] = str(INT_VARS)
                                
                                data[nr][Algo.LPCMCI.value] = res
                                    
                                # Save the dictionary back to a JSON file
                                with open(filename, 'w') as file:
                                    json.dump(data, file)
                                    
                                gc.collect()
                        except timeout_decorator.timeout_decorator.TimeoutError:
                            gc.collect()
                            remove_directory(os.getcwd() + '/' + resfolder)
                            continue
            
                    
                    #########################################################################################################################
                    # CAnDOIT                        
                    if Algo.CAnDOIT.value not in data[nr] or (Algo.CAnDOIT.value in data[nr] and not data[nr][Algo.CAnDOIT.value]['done']):
                        noIntervention = True
                        for selected_intvar in potentialIntervention:
                            tmp_d_int = {intvar: d_int[intvar] for intvar in d_int.keys() if intvar == selected_intvar}
                                                    
                            new_d_obs = deepcopy(d_obs)
                            new_d_obs.d = new_d_obs.d[:-nsample_int]
                            candoit = CAnDOIT(new_d_obs, 
                                            deepcopy(tmp_d_int),
                                            alpha = alpha, 
                                            min_lag = 0, 
                                            max_lag = max_lag, 
                                            sel_method = TE(TEestimator.Gaussian), 
                                            val_condtest = ParCorr(significance = 'analytic'),
                                            verbosity = CPLevel.INFO,
                                            neglect_only_autodep = False,
                                            resfolder = resfolder + f"/candoit_{selected_intvar}",
                                            plot_data = False,
                                            exclude_context = True)
                            try:
                                new_start = time()
                                candoit_cm = run_algo(candoit, 'candoit')
                                # candoit_cm = candoit.run(remove_unneeded=False, nofilter=True)
                                elapsed_candoit = time() - new_start
                                candoit_time = str(timedelta(seconds = elapsed_candoit))
                                print(candoit_time)
                                candoit_cm.ts_dag(save_name = candoit.ts_dag_path, img_extention = ImageExt.PNG, node_size=6, min_width=2, max_width=5, x_disp=1)
                                candoit_cm.ts_dag(save_name = candoit.ts_dag_path, img_extention = ImageExt.PDF, node_size=6, min_width=2, max_width=5, x_disp=1)
                                gc.collect()
                            
                                res = fill_res({"time": candoit_time, 
                                                "adj": candoit_cm.get_Adj(), 
                                                "graph": candoit_cm.get_Graph()})
                                
                                data[nr][f"{Algo.CAnDOIT.value}__{selected_intvar}"] = res
                                noIntervention = False
                                    
                                # Save the dictionary back to a JSON file
                                with open(filename, 'w') as file:
                                    json.dump(data, file)
                                                                 
                            except timeout_decorator.timeout_decorator.TimeoutError:
                                if noIntervention:
                                    gc.collect()
                                    remove_directory(os.getcwd() + '/' + resfolder)
                                    
                                    del data[nr]
                                    
                                    with open(filename, 'w') as file:
                                        json.dump(data, file)
                                    
                                continue

                    data[nr]['done'] = True
                    # Save the dictionary back to a JSON file
                    with open(filename, 'w') as file:
                        json.dump(data, file)
                    break
                    
                # except Exception as e:
                #     traceback_info = traceback.format_exc()
                #     with open(os.getcwd() + '/results/' + resdir + '/error.txt', 'a') as f:
                #         f.write("Exception occurred: " + str(e) + "\n")
                #         f.write("Traceback:\n" + traceback_info + "\n")
                #     remove_directory(os.getcwd() + "/" + resfolder)
                    
                #     filename = os.getcwd() + "/results/" + resdir + "/" + str(n) + ".json"
                #     if os.path.exists(filename):
                #         with open(filename, 'r') as file:
                #             data = json.load(file)
                #             if nr in data: 
                #                 data.pop(nr)
                #                 json.dump(data, file)           
                #     continue