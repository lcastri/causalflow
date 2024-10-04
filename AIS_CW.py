import copy
import os
import pandas as pd
# from tigramite.independence_tests.gpdc_torch import GPDCtorch as GPDC
from tigramite.independence_tests.gpdc import GPDC
from causalflow.CPrinter import CPLevel
from causalflow.basics.constants import ImageExt
from causalflow.causal_discovery.CAnDOIT import CAnDOIT
from causalflow.causal_discovery.baseline.LPCMCI import LPCMCI
from causalflow.preprocessing.data import Data
from causalflow.selection_methods.TE import TE, TEestimator

if __name__ == '__main__':   
    resfolder = "results/AIS_major/CW"
    datafolder = "CW_data"
    obscsv = "/obs.csv"
    intcsvFc = "/int_Fc.csv"
    intcsvCc = "/int_Cc.csv"
    obslen = 600
    intlen = 125
    f_alpha = 0.05
    pcmci_alpha = 0.01
    min_lag = 0
    max_lag = 1
    
    df_obs = Data(pd.read_csv(os.getcwd() + "/" + datafolder + obscsv)[:obslen])
        
    # ######################################################################################################
    # OBS NO HIDDEN CONFOUNDER
    # lpcmci = LPCMCI(copy.deepcopy(df_obs),
    #                 min_lag = min_lag, 
    #                 max_lag = max_lag,
    #                 sys_context = [],
    #                 val_condtest = GPDC(significance = 'analytic'),
    #                 verbosity = CPLevel.DEBUG,
    #                 alpha = pcmci_alpha, 
    #                 neglect_only_autodep = False,
    #                 resfolder = resfolder + "/lpcmci_noh")
    
    # lpcmci_cm = lpcmci.run()
    # lpcmci_cm.ts_dag(save_name = lpcmci.ts_dag_path, img_extention = ImageExt.PNG, node_size=6, min_width=2, max_width=5, x_disp=1, font_size=14, node_color=['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple'])
    # lpcmci_cm.ts_dag(save_name = lpcmci.ts_dag_path, img_extention = ImageExt.PDF, node_size=6, min_width=2, max_width=5, x_disp=1, font_size=14, node_color=['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple'])
    # lpcmci.save()
    
    
    shrinkVars = ['F_c', 'C_c', 'v', 'd_c']
    df_obs.shrink(shrinkVars)
    # ######################################################################################################
    # # OBS HIDDEN CONFOUNDER
    # lpcmci = LPCMCI(copy.deepcopy(df_obs),
    #                 min_lag = min_lag, 
    #                 max_lag = max_lag,
    #                 sys_context = [],
    #                 val_condtest = GPDC(significance = 'analytic'),
    #                 verbosity = CPLevel.DEBUG,
    #                 alpha = pcmci_alpha, 
    #                 neglect_only_autodep = False,
    #                 resfolder = resfolder + "/lpcmci")
    
    # lpcmci_cm = lpcmci.run()
    # lpcmci_cm.ts_dag(save_name = lpcmci.ts_dag_path, img_extention = ImageExt.PNG, node_size=6, min_width=2, max_width=5, x_disp=1, font_size=14, node_color=['tab:blue', 'tab:orange', 'tab:red', 'tab:purple'])
    # lpcmci_cm.ts_dag(save_name = lpcmci.ts_dag_path, img_extention = ImageExt.PDF, node_size=6, min_width=2, max_width=5, x_disp=1, font_size=14, node_color=['tab:blue', 'tab:orange', 'tab:red', 'tab:purple'])
    # lpcmci.save()
                        
    
    ######################################################################################################
    # INTERVENTION
    shrinkVars = ['F_c', 'C_c', 'v', 'd_c']
    
    int_dataFc = Data(pd.read_csv(os.getcwd() + "/" + datafolder + intcsvFc)[2:intlen+2])
    int_dataFc.shrink(shrinkVars)
    # int_dataFc.plot_timeseries()
    # int_dataCc = Data(pd.read_csv(os.getcwd() + "/" + datafolder + intcsvCc)[:intlen])
    # int_dataCc.shrink(shrinkVars)
    # int_dataCc.plot_timeseries()
    df_int = {'F_c': int_dataFc}
    # df_int = {'C_c': int_dataCc}
    # df_int = {'F_c': int_dataFc, 'C_c': int_dataCc}
    df_obsCAnDOIT = copy.deepcopy(df_obs)
    df_obsCAnDOIT = Data(df_obsCAnDOIT.d[:- sum([df_int[k].T for k in df_int])])
    df_obsCAnDOIT.shrink(shrinkVars)
    candoit = CAnDOIT(copy.deepcopy(df_obs), 
                      copy.deepcopy(df_int),
                      min_lag = min_lag, 
                      max_lag = max_lag, 
                      sel_method = TE(TEestimator.Gaussian), 
                      val_condtest = GPDC(significance = 'analytic'),
                      verbosity = CPLevel.DEBUG,
                      f_alpha = f_alpha, 
                      alpha = pcmci_alpha, 
                      neglect_only_autodep = False,
                      resfolder = resfolder + "/candoit",
                      plot_data = True,
                      exclude_context = True)
    
    candoit_cm = candoit.run(remove_unneeded=False, nofilter=True)
    candoit_cm.ts_dag(save_name = candoit.ts_dag_path, img_extention = ImageExt.PNG, node_size=6, min_width=2, max_width=5, x_disp=1, font_size=14, node_color=['tab:blue', 'tab:orange', 'tab:red', 'tab:purple'])
    candoit_cm.ts_dag(save_name = candoit.ts_dag_path, img_extention = ImageExt.PDF, node_size=6, min_width=2, max_width=5, x_disp=1, font_size=14, node_color=['tab:blue', 'tab:orange', 'tab:red', 'tab:purple'])
    candoit.save()
