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
# X_0(t) = 0.15X_0(t-1) + noise_0(t)
# X_1(t) = 0.5X_0(t-1)**2 + noise_1(t)
# X_2(t) = 0.75X_1(t-1) + 0.9X_4(t-2) + noise_2(t)
# X_3(t) = 0.33X_4(t-1)X_1(t-1) + noise_3(t)
# X_4(t) = noise_4(t)

if __name__ == '__main__':   
    resdir = "FPCMCI_vs_doFPCMCI_simple"
    f_alpha = 0.1
    pcmci_alpha = 0.05
    min_lag = 1
    max_lag = 2

    np.random.seed(1)
    nsample = 750
    nfeature = 5
    d = np.random.random(size = (nsample, nfeature))
    for t in range(max_lag, nsample):
        d[t, 0] += 0.15 * d[t-1, 0]
        d[t, 1] += 0.5 * d[t-1, 0]**2
        d[t, 2] += 0.75 * d[t-1, 1] + 0.9 * d[t-2, 4] 
        d[t, 3] += 0.33 * d[t-1, 4] * d[t-1, 1] + d[t-1, 3]
        
       
    df_obs = Data(d, vars = ['A', 'B', 'C', 'D', 'E'])
    
    df_obs_hidden = deepcopy(df_obs)
    df_obs_hidden.shrink(['A', 'B', 'C', 'D'])
    # df_obs_hidden.plot_timeseries()
    
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
    int_data = dict()
    
    # # A
    # d_intA = np.random.random(size = (Tint, nfeature))
    # d_intA[:, 0] = 2 * np.ones(shape = (Tint,)) 
    # for t in range(max_lag, Tint):
    #     d_intA[t, 1] += 0.5 * d_intA[t-1, 0]**2
    #     d_intA[t, 2] += 0.75 * d_intA[t-1, 1] + 0.9 * d_intA[t-2, 4] 
    #     d_intA[t, 3] += 0.33 * d_intA[t-1, 4] * d_intA[t-1, 2]
        
    # df_int = Data(d_intA, vars = ['A', 'B', 'C', 'D', 'E'])
    # df_int.shrink(['A', 'C', 'D', 'E'])
    # int_data['A'] =  df_int
    
    # C
    # d_intC = np.random.random(size = (Tint, nfeature))
    # d_intC[:, 2] = 3.5 * np.ones(shape = (Tint,)) 
    # for t in range(max_lag, Tint):
    #     d_intC[t, 0] += 0.15 * d_intC[t-1, 0]
    #     d_intC[t, 1] += 0.5 * d_intC[t-1, 0]**2
    #     d_intC[t, 3] += 0.33 * d_intC[t-1, 4] * d_intC[t-1, 2]
        
    # df_int = Data(d_intC, vars = ['A', 'B', 'C', 'D', 'E'])
    # df_int.shrink(['A', 'C', 'D', 'E'])
    # int_data['C'] =  df_int
    
    # D
    d_intD = np.random.random(size = (Tint, nfeature))
    d_intD[:, 3] = 15 * np.ones(shape = (Tint,)) 
    for t in range(max_lag, Tint):
        d_intD[t, 0] += 0.15 * d_intD[t-1, 0]
        d_intD[t, 1] += 0.5 * d_intD[t-1, 0]**2
        d_intD[t, 2] += 0.75 * d_intD[t-1, 1] + 0.9 * d_intD[t-2, 4] 
        
    df_int = Data(d_intD, vars = ['A', 'B', 'C', 'D', 'E'])
    df_int.shrink(['A', 'B', 'C', 'D'])
    int_data['D'] =  df_int
    
    
    # dofpcmci = doFPCMCI_new(deepcopy(df_obs_hidden), 
    #                     deepcopy(int_data),
    #                     f_alpha = f_alpha, 
    #                     pcmci_alpha = pcmci_alpha, 
    #                     min_lag = min_lag, 
    #                     max_lag = max_lag, 
    #                     sel_method = TE(TEestimator.Gaussian), 
    #                     val_condtest = GPDC(significance = 'analytic', gp_params = None),
    #                     verbosity = CPLevel.INFO,
    #                     neglect_only_autodep = False,
    #                     resfolder = resdir + "/dofpcmci_new_hidden",
    #                     plot_data = False,
    #                     exclude_context = True)
    
    # new_start = time()
    # features, cm = dofpcmci.run()
    # elapsed_newFPCMCI = time() - new_start
    # print(str(timedelta(seconds = elapsed_newFPCMCI)))
    # dofpcmci.dag(label_type = LabelType.Lag, node_layout = 'dot')
    # dofpcmci.timeseries_dag()
    
    
    dofpcmci = doFPCMCI(deepcopy(df_obs_hidden), 
                        deepcopy(int_data),
                        f_alpha = f_alpha, 
                        pcmci_alpha = pcmci_alpha, 
                        min_lag = min_lag, 
                        max_lag = max_lag, 
                        sel_method = TE(TEestimator.Gaussian), 
                        val_condtest = GPDC(significance = 'analytic', gp_params = None),
                        verbosity = CPLevel.INFO,
                        neglect_only_autodep = False,
                        resfolder = resdir + "/dofpcmci_hidden",
                        plot_data = True,
                        exclude_context = True)
    
    new_start = time()
    features, cm = dofpcmci.run()
    elapsed_newFPCMCI = time() - new_start
    print(str(timedelta(seconds = elapsed_newFPCMCI)))
    dofpcmci.dag(label_type = LabelType.Lag, node_layout = 'dot')
    dofpcmci.timeseries_dag()
