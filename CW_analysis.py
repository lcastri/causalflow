import copy
import os
import pandas as pd
# from tigramite.independence_tests.gpdc_torch import GPDCtorch as GPDC
from tigramite.independence_tests.gpdc import GPDC
from causalflow.CPrinter import CPLevel
from causalflow.basics.constants import ImageExt
from causalflow.causal_discovery.CAnDOIT import CAnDOIT
from causalflow.causal_discovery.FPCMCI import FPCMCI
from causalflow.preprocessing.data import Data
from causalflow.selection_methods.TE import TE, TEestimator

if __name__ == '__main__':   
    resfolder = "CW_result"
    datafolder = "CW_data"
    obscsv = "/old/ReachingSingleFinger_obs.csv"
    intcsv = "/old/ReachingSingleFinger_int.csv"
    obslen = 600
    intlen = 225
    f_alpha = 0.05
    pcmci_alpha = 0.05
    min_lag = 1
    max_lag = 2
    
    df_obs = Data(pd.read_csv(os.getcwd() + "/" + datafolder + obscsv)[:obslen])
    
    # ######################################################################################################
    # # OBS NO HIDDEN CONFOUNDER
    # fpcmci = FPCMCI(copy.deepcopy(df_obs),
    #                 f_alpha = f_alpha, 
    #                 alpha = 0.001, 
    #                 min_lag = min_lag, 
    #                 max_lag = max_lag, 
    #                 sel_method = TE(TEestimator.Gaussian), 
    #                 val_condtest = GPDC(significance = 'analytic'),
    #                 verbosity = CPLevel.DEBUG,
    #                 neglect_only_autodep = True,
    #                 resfolder = resfolder + "/fpcmci")
    
    # cm = fpcmci.run()
    # fpcmci.timeseries_dag(font_size = 14, node_color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'])
    # fpcmci.timeseries_dag(font_size = 14, img_ext=ImageExt.PDF, node_color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'])
    
    
    # shrinkVars = ['F_c', 'C_c', 'v', 'd_c']
    # df_obs.shrink(shrinkVars)
    # ######################################################################################################
    # # OBS HIDDEN CONFOUNDER
    # fpcmci = FPCMCI(copy.deepcopy(df_obs),
    #                 f_alpha = f_alpha, 
    #                 alpha = pcmci_alpha, 
    #                 min_lag = min_lag, 
    #                 max_lag = max_lag, 
    #                 sel_method = TE(TEestimator.Gaussian), 
    #                 val_condtest = GPDC(significance = 'analytic'),
    #                 verbosity = CPLevel.DEBUG,
    #                 neglect_only_autodep = True,
    #                 resfolder = resfolder + "/fpcmci_hc")
    
    # cm = fpcmci.run(remove_unneeded=False)
    # fpcmci.timeseries_dag(font_size = 14, node_color=['tab:blue', 'tab:orange', 'tab:red', 'tab:purple'])
    # fpcmci.timeseries_dag(font_size = 14, img_ext=ImageExt.PDF, node_color=['tab:blue', 'tab:orange', 'tab:red', 'tab:purple'])
    
    
    ######################################################################################################
    # INTERVENTION
    shrinkVars = ['F_c', 'C_c', 'v', 'd_c']
    
    # int_data = Data(os.getcwd() + "/" + datafolder + intcsv)
    int_data = Data(pd.read_csv(os.getcwd() + "/" + datafolder + intcsv)[:intlen])
    int_data.shrink(shrinkVars)
    df_obsCAnDOIT = copy.deepcopy(df_obs)
    df_obsCAnDOIT = Data(df_obsCAnDOIT.d[:-int_data.T])
    df_obsCAnDOIT.shrink(shrinkVars)
    df_int = {'C_c': int_data}
    
    candoit = CAnDOIT(df_obsCAnDOIT, 
                        copy.deepcopy(df_int),
                        f_alpha = f_alpha, 
                        alpha = pcmci_alpha, 
                        min_lag = min_lag, 
                        max_lag = max_lag, 
                        sel_method = TE(TEestimator.Gaussian), 
                        val_condtest = GPDC(significance = 'analytic'),
                        verbosity = CPLevel.DEBUG,
                        neglect_only_autodep = True,
                        resfolder = resfolder + "/candoit",
                        plot_data = False,
                        exclude_context = True)
    
    cm = candoit.run(remove_unneeded=False)
    candoit.timeseries_dag(font_size = 14, node_color=['tab:blue', 'tab:orange', 'tab:red', 'tab:purple'])
    candoit.timeseries_dag(font_size = 14, img_ext=ImageExt.PDF, node_color=['tab:blue', 'tab:orange', 'tab:red', 'tab:purple'])
    