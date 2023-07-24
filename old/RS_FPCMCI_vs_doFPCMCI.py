from copy import deepcopy
import os
import random
from matplotlib import pyplot as plt
from tigramite.independence_tests.gpdc import GPDC
from fpcmci.CPrinter import CPLevel
from fpcmci.doFPCMCI_new import doFPCMCI as doFPCMCI_new
from fpcmci.doFPCMCI import doFPCMCI
from fpcmci.FPCMCI import FPCMCI
from fpcmci.preprocessing.data import Data
from fpcmci.selection_methods.TE import TE, TEestimator
from fpcmci.basics.constants import LabelType
import numpy as np
from random_system.random_system import RandomSystem

from time import time
from datetime import timedelta
import res_statistics as sta


def print_res(alg, cm, time):
    scm = {v: list() for v in GT.keys()}
    if len(cm.features) != 0:
        _scm = cm.get_SCM()
        for k in _scm:
            scm[k] = _scm[k]
    _scm = scm
    
    p = sta.precision(GT, cm = scm)
    r = sta.recall(GT, cm = scm)
    print(alg + " statistics:")
    print("\t|F1 score = " + str(round(sta.f1_score(p, r), 3)))
    print("\t|SHD = " + str(sta.shd(GT, cm = scm)))
    print("\t|time = " + str(time))

    
if __name__ == '__main__':   
    resdir = "RS_FPCMCI_vs_doFPCMCI"
    f_alpha = 0.05
    pcmci_alpha = 0.05
    min_lag = 1
    max_lag = 2
    nsample = 1500
    nfeature = 4
    complexity = 20
    
    
    #########################################################################################################################
    # DATA
    confounders = list()
    
    while len(confounders) == 0:
        RS = RandomSystem(max_lag, min_lag, nfeature, nsample, resfolder = resdir)
        GT, vars = RS.randSCM()
        
        confounders = RS.get_lagged_confounders(GT)

    data = RS.gen_obs_ts()
    d_obs = Data(data, RS.varlist)
    
    sel_confounder = random.choice(confounders)
    conf_to_hide = next(iter(sel_confounder.keys()))
    int_var = min(sel_confounder[conf_to_hide], key=lambda x: abs(x[2]))[0]
    
    data_int = RS.gen_int_ts(int_var, random.uniform(2, 5), nsample)
    d_int = Data(data_int, RS.varlist)
    
    vars.remove(conf_to_hide)
    d_obs.shrink(vars)
    d_int.shrink(vars)
    
    int_data = {int_var: d_int}

    d_obs.plot_timeseries()
    d_int.plot_timeseries() 
    print("Confoundend triple: " + str(sel_confounder))
    print("Hidden confounder: " + conf_to_hide)
    print("Intervention variable: " + int_var)
    
    #########################################################################################################################
    # FPCMCI
    fpcmci = FPCMCI(deepcopy(d_obs),
                    f_alpha = f_alpha, 
                    pcmci_alpha = pcmci_alpha, 
                    min_lag = min_lag, 
                    max_lag = max_lag, 
                    sel_method = TE(TEestimator.Gaussian), 
                    val_condtest = GPDC(significance = 'analytic', gp_params = None),
                    verbosity = CPLevel.INFO,
                    neglect_only_autodep = False,
                    resfolder = resdir + "/5/fpcmci_hidden")

    new_start = time()
    features, fpcmci_cm = fpcmci.run()
    elapsed_newFPCMCI = time() - new_start
    fpcmci_time = str(timedelta(seconds = elapsed_newFPCMCI))
    print(fpcmci_time)
    fpcmci.timeseries_dag()
    
    #########################################################################################################################
    # doFPCMCI new
    dofpcmci_new = doFPCMCI_new(deepcopy(d_obs), 
                        deepcopy(int_data),
                        f_alpha = f_alpha, 
                        pcmci_alpha = pcmci_alpha, 
                        min_lag = min_lag, 
                        max_lag = max_lag, 
                        sel_method = TE(TEestimator.Gaussian), 
                        val_condtest = GPDC(significance = 'analytic', gp_params = None),
                        verbosity = CPLevel.INFO,
                        neglect_only_autodep = False,
                        resfolder = resdir + "/5/dofpcmci_new_hidden",
                        plot_data = False,
                        exclude_context = True)
    
    new_start = time()
    features, dofpcmci_new_cm = dofpcmci_new.run()
    elapsed_newFPCMCI = time() - new_start
    dofpcmci_new_time = str(timedelta(seconds = elapsed_newFPCMCI))
    print(dofpcmci_new_time)
    dofpcmci_new.timeseries_dag()
    
    
    #########################################################################################################################
    # doFPCMCI
    dofpcmci = doFPCMCI(deepcopy(d_obs), 
                        deepcopy(int_data),
                        f_alpha = f_alpha, 
                        pcmci_alpha = pcmci_alpha, 
                        min_lag = min_lag, 
                        max_lag = max_lag, 
                        sel_method = TE(TEestimator.Gaussian), 
                        val_condtest = GPDC(significance = 'analytic', gp_params = None),
                        verbosity = CPLevel.INFO,
                        neglect_only_autodep = False,
                        resfolder = resdir + "/5/dofpcmci_hidden",
                        plot_data = False,
                        exclude_context = True)
    
    new_start = time()
    features, dofpcmci_cm = dofpcmci.run()
    elapsed_newFPCMCI = time() - new_start
    dofpcmci_time = str(timedelta(seconds = elapsed_newFPCMCI))
    print(dofpcmci_time)
    dofpcmci.timeseries_dag()

    print("\n")
    print("Number of variable = " + str(nfeature))
    print("Ground truth = " + str(GT))
    print("Confoundend triple: " + str(sel_confounder))
    print("Hidden confounder: " + conf_to_hide)
    print("Intervention variable: " + int_var)
    print_res("FPCMCI", fpcmci_cm, fpcmci_time)
    print_res("DoFPCMCI_new", dofpcmci_new_cm, dofpcmci_new_time)
    print_res("DoFPCMCI", dofpcmci_cm, dofpcmci_time)