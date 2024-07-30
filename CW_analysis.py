import copy
import os
# from tigramite.independence_tests.gpdc_torch import GPDCtorch as GPDC
from tigramite.independence_tests.gpdc import GPDC
from causalflow.CPrinter import CPLevel
from causalflow.basics.constants import ImageExt
from causalflow.causal_discovery.CAnDOIT import CAnDOIT
from causalflow.causal_discovery.baseline.LPCMCI import LPCMCI
from causalflow.preprocessing.data import Data
from causalflow.selection_methods.TE import TE, TEestimator

if __name__ == '__main__':   
    resfolder = "CW_result"
    datafolder = "CW_data"
    f_alpha = 0.05
    pcmci_alpha = 0.01
    min_lag = 1
    max_lag = 2
    
    df_obs = Data(os.getcwd() + "/" + datafolder + "/ReachingSingleFinger_obs.csv")
    
    # # # ######################################################################################################
    # # OBS NO HIDDEN CONFOUNDER
    # fpcmci = FPCMCI(copy.deepcopy(df_obs),
    #                 f_alpha = f_alpha, 
    #                 pcmci_alpha = pcmci_alpha, 
    #                 min_lag = min_lag, 
    #                 max_lag = max_lag, 
    #                 sel_method = TE(TEestimator.Gaussian), 
    #                 val_condtest = GPDC(significance = 'analytic'),
    #                 verbosity = CPLevel.DEBUG,
    #                 neglect_only_autodep = True,
    #                 resfolder = resfolder + "/fpcmci")
    
    # features, cm = fpcmci.run()
    # fpcmci.timeseries_dag(font_size = 14)
    # fpcmci.timeseries_dag(font_size = 14, img_ext=ImageExt.PDF)
    
    
    shrinkVars = ['F_c', 'C_c', 'v', 'd_c']
    df_obs.shrink(shrinkVars)
    # # ######################################################################################################
    # OBS HIDDEN CONFOUNDER
    lpcmci = LPCMCI(copy.deepcopy(df_obs),
                    min_lag = min_lag, 
                    max_lag = max_lag,
                    sys_context = [],
                    val_condtest = GPDC(significance = 'analytic'),
                    verbosity = CPLevel.DEBUG,
                    alpha = pcmci_alpha, 
                    neglect_only_autodep = False,
                    resfolder = resfolder + "/lpcmci")
    
    lpcmci_cm = lpcmci.run()
    lpcmci_cm.ts_dag(save_name = lpcmci.ts_dag_path, img_extention = ImageExt.PNG, node_size=6, min_width=2, max_width=5, x_disp=0.5)
    lpcmci_cm.ts_dag(save_name = lpcmci.ts_dag_path, img_extention = ImageExt.PDF, node_size=6, min_width=2, max_width=5, x_disp=0.5)
    lpcmci.save()
                        
    
    ######################################################################################################
    # INTERVENTION
    int_data = Data(os.getcwd() + "/" + datafolder + "/ReachingSingleFinger_int.csv")
    int_data.shrink(shrinkVars)
    df_int = {'C_c': int_data}
    candoit = CAnDOIT(copy.deepcopy(df_obs), 
                      copy.deepcopy(df_int),
                      min_lag = min_lag, 
                      max_lag = max_lag, 
                      sel_method = TE(TEestimator.Gaussian), 
                      val_condtest = GPDC(significance = 'analytic'),
                      verbosity = CPLevel.DEBUG,
                      f_alpha = f_alpha, 
                      pcmci_alpha = pcmci_alpha, 
                      neglect_only_autodep = False,
                      resfolder = resfolder + "/candoit",
                      plot_data = False,
                      exclude_context = True)
    
    candoit_cm = candoit.run(remove_unneeded=False, nofilter=True)
    candoit_cm.ts_dag(save_name = lpcmci.ts_dag_path, img_extention = ImageExt.PNG, node_size=6, min_width=2, max_width=5, x_disp=0.5)
    candoit_cm.ts_dag(save_name = lpcmci.ts_dag_path, img_extention = ImageExt.PDF, node_size=6, min_width=2, max_width=5, x_disp=0.5)
    candoit.save()
