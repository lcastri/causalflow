import copy
import os
# from tigramite.independence_tests.gpdc_torch import GPDCtorch as GPDC
from tigramite.independence_tests.gpdc import GPDC
from fpcmci.CPrinter import CPLevel
from fpcmci.basics.constants import ImageExt
from fpcmci.doFPCMCI import doFPCMCI
from fpcmci.FPCMCI import FPCMCI
from fpcmci.preprocessing.data import Data
from fpcmci.selection_methods.TE import TE, TEestimator

if __name__ == '__main__':   
    resfolder = "CW_result"
    datafolder = "CW_data"
    f_alpha = 0.05
    pcmci_alpha = 0.01
    min_lag = 1
    max_lag = 2
    
    noise = 0.15
    df_obs = Data(os.getcwd() + "/" + datafolder + "/ReachingSingleFinger_obs_noise" + str(noise) + ".csv")
    
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
                    resfolder = resfolder + "/fpcmci" + str(noise))
    
    features, cm = fpcmci.run()
    fpcmci.timeseries_dag()
    fpcmci.timeseries_dag(img_ext=ImageExt.PDF)
    
    
    shrinkVars = ['F_c', 'B_c', 'v', 'd_b']
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
                    resfolder = resfolder + "/fpcmci" + str(noise) + "_hc")
    
    features, cm = fpcmci.run()
    fpcmci.timeseries_dag()
    fpcmci.timeseries_dag(img_ext=ImageExt.PDF)
    
    
    ######################################################################################################
    # INTERVENTION
    int_data = Data(os.getcwd() + "/" + datafolder + "/ReachingSingleFinger_int_noise" + str(noise) + ".csv")
    int_data.shrink(shrinkVars)
    df_int = {'B_c': int_data}
    dofpcmci = doFPCMCI(copy.deepcopy(df_obs), 
                        copy.deepcopy(df_int),
                        f_alpha = f_alpha, 
                        pcmci_alpha = pcmci_alpha, 
                        min_lag = min_lag, 
                        max_lag = max_lag, 
                        sel_method = TE(TEestimator.Gaussian), 
                        val_condtest = GPDC(significance = 'analytic'),
                        verbosity = CPLevel.DEBUG,
                        neglect_only_autodep = True,
                        resfolder = resfolder + "/dofpcmci" + str(noise),
                        plot_data = False,
                        exclude_context = True)
    
    features, cm = dofpcmci.run()
    dofpcmci.timeseries_dag()
    dofpcmci.timeseries_dag(img_ext=ImageExt.PDF)
