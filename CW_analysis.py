import copy
import os
# from tigramite.independence_tests.gpdc_torch import GPDCtorch as GPDC
from tigramite.independence_tests.gpdc import GPDC
from connectingdots.CPrinter import CPLevel
from connectingdots.basics.constants import ImageExt
from connectingdots.CAnDOIT import CAnDOIT
from connectingdots.FPCMCI import FPCMCI
from connectingdots.preprocessing.data import Data
from connectingdots.selection_methods.TE import TE, TEestimator

if __name__ == '__main__':   
    resfolder = "CW_result"
    datafolder = "CW_data"
    f_alpha = 0.05
    pcmci_alpha = 0.01
    min_lag = 1
    max_lag = 2
    
    df_obs = Data(os.getcwd() + "/" + datafolder + "/ReachingSingleFinger_obs.csv")
    
    # # ######################################################################################################
    # OBS NO HIDDEN CONFOUNDER
    fpcmci = FPCMCI(copy.deepcopy(df_obs),
                    f_alpha = f_alpha, 
                    pcmci_alpha = pcmci_alpha, 
                    min_lag = min_lag, 
                    max_lag = max_lag, 
                    sel_method = TE(TEestimator.Gaussian), 
                    val_condtest = GPDC(significance = 'analytic'),
                    verbosity = CPLevel.DEBUG,
                    neglect_only_autodep = True,
                    resfolder = resfolder + "/fpcmci")
    
    features, cm = fpcmci.run()
    fpcmci.timeseries_dag(font_size = 14)
    fpcmci.timeseries_dag(font_size = 14, img_ext=ImageExt.PDF)
    
    
    shrinkVars = ['F_c', 'C_c', 'v', 'd_c']
    df_obs.shrink(shrinkVars)
    # # ######################################################################################################
    # OBS HIDDEN CONFOUNDER
    fpcmci = FPCMCI(copy.deepcopy(df_obs),
                    f_alpha = f_alpha, 
                    pcmci_alpha = pcmci_alpha, 
                    min_lag = min_lag, 
                    max_lag = max_lag, 
                    sel_method = TE(TEestimator.Gaussian), 
                    val_condtest = GPDC(significance = 'analytic'),
                    verbosity = CPLevel.DEBUG,
                    neglect_only_autodep = True,
                    resfolder = resfolder + "/fpcmci_hc")
    
    features, cm = fpcmci.run(remove_unneeded=False)
    fpcmci.timeseries_dag(font_size = 14)
    fpcmci.timeseries_dag(font_size = 14, img_ext=ImageExt.PDF)
    
    
    ######################################################################################################
    # INTERVENTION
    int_data = Data(os.getcwd() + "/" + datafolder + "/ReachingSingleFinger_int.csv")
    int_data.shrink(shrinkVars)
    df_int = {'C_c': int_data}
    candoit = CAnDOIT(copy.deepcopy(df_obs), 
                        copy.deepcopy(df_int),
                        f_alpha = f_alpha, 
                        pcmci_alpha = pcmci_alpha, 
                        min_lag = min_lag, 
                        max_lag = max_lag, 
                        sel_method = TE(TEestimator.Gaussian), 
                        val_condtest = GPDC(significance = 'analytic'),
                        verbosity = CPLevel.DEBUG,
                        neglect_only_autodep = True,
                        resfolder = resfolder + "/candoit",
                        plot_data = False,
                        exclude_context = True)
    
    features, cm = candoit.run(remove_unneeded=False)
    candoit.timeseries_dag(font_size = 14)
    candoit.timeseries_dag(font_size = 14, img_ext=ImageExt.PDF)
