from copy import deepcopy
from tigramite.independence_tests.gpdc import GPDC
from fpcmci.CPrinter import CPLevel
from fpcmci.doFPCMCI_new import doFPCMCI as doFPCMCI_new
from fpcmci.doFPCMCI import doFPCMCI
from fpcmci.FPCMCI import FPCMCI
from fpcmci.preprocessing.data import Data
from fpcmci.selection_methods.TE import TE, TEestimator
from fpcmci.basics.constants import LabelType
import numpy as np

from time import time
from datetime import datetime, timedelta

# System of equations
# X_0(t) = noise_0(t)
# X_1(t) = 0.5X_0(t-1) + noise_1(t)
# X_2(t) = 0.3X_0(t-2) + 0.75X_3(t-1) + noise_2(t)
# X_3(t) = noise_3(t)

if __name__ == '__main__':   
    resdir = "FPCMCI_vs_doFPCMCI_may22"
    f_alpha = 0.1
    pcmci_alpha = 0.05
    min_lag = 1
    max_lag = 2

    np.random.seed(1)
    nsample = 1500
    nfeature = 4 
    d = np.random.random(size = (nsample, nfeature))
    for t in range(max_lag, nsample):
        d[t, 1] += 0.5 * d[t-1, 0]
        d[t, 2] += 0.3 * d[t-2, 0] + 0.75 * d[t-1, 3] 
    
    
    
    df_obs = Data(d, vars = ['X_0', 'X_1', 'X_2', 'X_3'])
    
    df_obs_hidden = deepcopy(df_obs)
    df_obs_hidden.shrink(['X_1', 'X_2', 'X_3'])
    df_obs_hidden.plot_timeseries()
    
    #########################################################################################################################
    # # FPCMCI
    # fpcmci = FPCMCI(deepcopy(df_obs),
    #                 f_alpha = f_alpha, 
    #                 pcmci_alpha = pcmci_alpha, 
    #                 min_lag = min_lag, 
    #                 max_lag = max_lag, 
    #                 sel_method = TE(TEestimator.Gaussian), 
    #                 val_condtest = GPDC(significance = 'analytic', gp_params = None),
    #                 verbosity = CPLevel.INFO,
    #                 neglect_only_autodep = False,
    #                 resfolder = resdir + "/fpcmci")

    # startFPCMCI = datetime.now()
    # features, cm = fpcmci.run()
    # stopFPCMCI = datetime.now()  
    # fpcmci.dag(node_layout='circular', label_type = LabelType.Lag)
    # fpcmci.timeseries_dag()
    
    # #########################################################################################################################
    # # FPCMCI
    # fpcmci = FPCMCI(deepcopy(df_obs_hidden),
    #                 f_alpha = f_alpha, 
    #                 pcmci_alpha = pcmci_alpha, 
    #                 min_lag = min_lag, 
    #                 max_lag = max_lag, 
    #                 sel_method = TE(TEestimator.Gaussian), 
    #                 val_condtest = GPDC(significance = 'analytic', gp_params = None),
    #                 verbosity = CPLevel.INFO,
    #                 neglect_only_autodep = False,
    #                 resfolder = resdir + "/fpcmci_hidden")

    # startFPCMCI = datetime.now()
    # features, cm = fpcmci.run()
    # stopFPCMCI = datetime.now()  
    # fpcmci.dag(node_layout='circular', label_type = LabelType.Lag)
    # fpcmci.timeseries_dag()
    
    ######################################################## INTERVENTION ########################################################
    Tint = nsample
    # Tint = int(nsample*0.5)
    int_data = dict()
    
    # X_1
    d_intX1 = np.random.random(size = (Tint, nfeature))
    d_intX1[:, 1] = 15 * np.ones(shape = (Tint,)) 
    for t in range(max_lag, Tint):
        d_intX1[t, 2] += 0.3 * d_intX1[t-2, 0] + 0.75 * d_intX1[t-1, 3] 

        
    df_int = Data(d_intX1, vars = ['X_0', 'X_1', 'X_2', 'X_3'])
    df_int.shrink(['X_1', 'X_2', 'X_3'])
    int_data['X_1'] =  df_int
    
    # dofpcmci = doFPCMCI(deepcopy(df_obs_hidden), 
    #                     deepcopy(int_data),
    #                     f_alpha = f_alpha, 
    #                     pcmci_alpha = pcmci_alpha, 
    #                     min_lag = min_lag, 
    #                     max_lag = max_lag, 
    #                     sel_method = TE(TEestimator.Gaussian), 
    #                     val_condtest = GPDC(significance = 'analytic', gp_params = None),
    #                     verbosity = CPLevel.INFO,
    #                     neglect_only_autodep = False,
    #                     resfolder = resdir + "/dofpcmci_hidden",
    #                     plot_data = True,
    #                     exclude_context = True)
    
    # new_start = time()
    # features, cm = dofpcmci.run()
    # elapsed_newFPCMCI = time() - new_start
    # print(str(timedelta(seconds = elapsed_newFPCMCI)))
    # dofpcmci.dag(label_type = LabelType.Lag, node_layout = 'dot')
    # dofpcmci.timeseries_dag()


    dofpcmci = doFPCMCI_new(deepcopy(df_obs_hidden), 
                        deepcopy(int_data),
                        f_alpha = f_alpha, 
                        pcmci_alpha = pcmci_alpha, 
                        min_lag = min_lag, 
                        max_lag = max_lag, 
                        sel_method = TE(TEestimator.Gaussian), 
                        val_condtest = GPDC(significance = 'analytic', gp_params = None),
                        verbosity = CPLevel.INFO,
                        neglect_only_autodep = False,
                        resfolder = resdir + "/dofpcmci_new_hidden",
                        plot_data = True,
                        exclude_context = True)
    
    new_start = time()
    features, cm = dofpcmci.run()
    elapsed_newFPCMCI = time() - new_start
    print(str(timedelta(seconds = elapsed_newFPCMCI)))
    dofpcmci.dag(label_type = LabelType.Lag, node_layout = 'dot')
    dofpcmci.timeseries_dag()