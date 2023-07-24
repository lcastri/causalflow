from copy import deepcopy
from tigramite.independence_tests.gpdc import GPDC
from fpcmci.CPrinter import CPLevel
from fpcmci.doFPCMCI_hardcoded import doFPCMCI
from fpcmci.FPCMCI import FPCMCI
from fpcmci.preprocessing.data import Data
from fpcmci.selection_methods.TE import TE, TEestimator
from fpcmci.basics.constants import LabelType
import numpy as np


if __name__ == '__main__':   
    f_alpha = 0.1
    pcmci_alpha = 0.05
    min_lag = 1
    max_lag = 2

    np.random.seed(1)
    nsample = 1500
    nfeature = 4
    d = np.random.random(size = (nsample, nfeature))
    d_int = deepcopy(d)
    for t in range(max_lag, nsample):
        d[t, 1] += 0.5 * d[t-1, 0]
        d[t, 2] += 0.3 * d[t-2, 0] + 0.75 * d[t-1, 3]
       
    d = d[:,1:]
    df = Data(d, vars = ['X_1', 'X_2', 'X_3'])
    # df.plot_timeseries()
    
    # PCMCI
    pcmci = FPCMCI(df, 
                   f_alpha = f_alpha, 
                   pcmci_alpha = pcmci_alpha, 
                   min_lag = min_lag, 
                   max_lag = max_lag, 
                   sel_method = TE(TEestimator.Gaussian), 
                   val_condtest = GPDC(significance = 'analytic', gp_params = None),
                   verbosity = CPLevel.DEBUG,
                   neglect_only_autodep = False,
                   resfolder = 'PCMCI')
      
    features, cm = pcmci.run_pcmci()
    pcmci.dag(label_type = LabelType.Lag, node_layout = 'dot')
    pcmci.timeseries_dag()
    
    # FPCMCI
    fpcmci = FPCMCI(df, 
                    f_alpha = f_alpha, 
                    pcmci_alpha = pcmci_alpha, 
                    min_lag = min_lag, 
                    max_lag = max_lag, 
                    sel_method = TE(TEestimator.Gaussian), 
                    val_condtest = GPDC(significance = 'analytic', gp_params = None),
                    verbosity = CPLevel.DEBUG,
                    neglect_only_autodep = False,
                    resfolder = 'FPCMCI')
      
    features, cm = fpcmci.run()
    fpcmci.dag(label_type = LabelType.Lag, node_layout = 'dot')
    fpcmci.timeseries_dag()
    
    
    ######################################################## INTERVENTIONS ########################################################
    
    
    # doPCMCI
    # I want to do an intervention on X_1. So, I create a context variable CX_1 which models the intervention
    # CX_1 does not have any parent and it is connected ONLY to X_1 @ time t-1
    int_data = np.r_[0 * np.ones(shape = (int(nsample/2)-1, 1)), 15 * np.ones(shape = (int(nsample/2)+1, 1))]
    for t in range(max_lag, nsample):
        d_int[t, 2] += 0.3 * d_int[t-2, 0] + 0.75 * d_int[t-1, 3]
        if t < int(nsample/2):
            d_int[t, 1] += 0.5 * d_int[t-1, 0]
        else:
            d_int[t, 1] = int_data[t, 0]
    d_int = np.c_[d_int, int_data]
    d_int = d_int[:,1:]
    df_int = Data(d_int, vars = ['X_1', 'X_2', 'X_3', 'CX_1'])
    # df_int.plot_timeseries()
    dopcmci = FPCMCI(df_int,
                     f_alpha = f_alpha, 
                     pcmci_alpha = pcmci_alpha, 
                     min_lag = min_lag, 
                     max_lag = max_lag, 
                     sel_method = TE(TEestimator.Gaussian), 
                     val_condtest = GPDC(significance = 'analytic', gp_params = None),
                     verbosity = CPLevel.DEBUG,
                     neglect_only_autodep = False,
                     resfolder = 'doPCMCI')
    
    sel_links = {df_int.features.index(f) : dict() for f in df_int.features}
    for t in df_int.features:
        if t == 'CX_1': 
            continue
        sources = deepcopy(df_int.features)
        for s in sources:
            if s != 'CX_1':
                sel_links[df_int.features.index(t)][(df_int.features.index(s), -1)] = '-?>'
            elif t == 'X_1' and s == 'CX_1':
                sel_links[df_int.features.index(t)][(df_int.features.index(s), -1)] = '-->'
    
    
    dopcmci.validator.run(link_assumptions=sel_links)
    dopcmci.result = ['X_1', 'X_2', 'X_3', 'CX_1']
    dopcmci.dag(label_type = LabelType.Lag, node_layout = 'dot')
    dopcmci.timeseries_dag()
    
    # doFPCMCI
    dofpcmci = doFPCMCI(df, 
                        df_int,
                        f_alpha = f_alpha, 
                        pcmci_alpha = pcmci_alpha, 
                        min_lag = min_lag, 
                        max_lag = max_lag, 
                        sel_method = TE(TEestimator.Gaussian), 
                        val_condtest = GPDC(significance = 'analytic', gp_params = None),
                        verbosity = CPLevel.DEBUG,
                        neglect_only_autodep = False,
                        resfolder = 'doFPCMCI')
    
    features, cm = dofpcmci.run()
    dofpcmci.dag(label_type = LabelType.Lag, node_layout = 'dot')
    dofpcmci.timeseries_dag()