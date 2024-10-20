"""
This module provides the CAnDOIT class.

Classes:
    CAnDOIT: class containing the CAnDOIT causal discovery algorithm.
"""

import copy
import pickle
import numpy as np
import pandas as pd
from causalflow.causal_discovery.tigramite.independence_tests.independence_tests_base import CondIndTest
from causalflow.selection_methods.SelectionMethod import SelectionMethod
from causalflow.CPrinter import CPLevel, CP
from causalflow.basics.constants import *
from causalflow.basics.utils import *
from causalflow.preprocessing.data import Data 
from causalflow.causal_discovery.CausalDiscoveryMethod import CausalDiscoveryMethod 
from causalflow.causal_discovery.support.myLPCMCI import myLPCMCI
from causalflow.graph.DAG import DAG

class CAnDOIT(CausalDiscoveryMethod):
    """CAnDOIT causal discovery method."""

    def __init__(self, 
                 observation_data: Data, 
                 intervention_data: dict, 
                 max_lag,
                 sel_method: SelectionMethod, val_condtest: CondIndTest, 
                 verbosity: CPLevel, 
                 f_alpha = 0.05, 
                 alpha = 0.05, 
                 resfolder = None,
                 neglect_only_autodep = False,
                 exclude_context = True,
                 plot_data = False,
                 clean_cls = True):
        """
        Class contructor.

        Args:
            observation_data (Data): observational data to analyse.
            intervention_data (dict): interventional data to analyse in the form {INTERVENTION_VARIABLE : Data (same variables of observation_data)}.
            max_lag (int): maximum time lag.
            sel_method (SelectionMethod): selection method.
            val_condtest (CondIndTest): validation method.
            verbosity (CPLevel): verbosity level.
            f_alpha (float, optional): filter significance level. Defaults to 0.05.
            alpha (float, optional): PCMCI significance level. Defaults to 0.05.
            resfolder (string, optional): result folder to create. Defaults to None.
            neglect_only_autodep (bool, optional): Bit for neglecting variables with only autodependency. Defaults to False.
            exclude_context (bool, optional): Bit for neglecting context variables. Defaults to False.
            plot_data (bool, optional): Bit for plotting your data. Defaults to False.
            clean_cls (bool): Clean console bit. Default to True.
        """
        min_lag = 0
        
        self.obs_data = observation_data
        self.systems = observation_data.features
        self.contexts = []
        self.sys_context = {}
        for k in intervention_data.keys():
            self.contexts.append("C" + k)
            self.sys_context[k] = "C" + k
        self.vars = self.systems + self.contexts
        
        self.f_alpha = f_alpha
        self.sel_method = sel_method
        self.val_condtest = val_condtest
        self.exclude_context = exclude_context
        super().__init__(self.obs_data, min_lag, max_lag, verbosity, alpha, resfolder, neglect_only_autodep, clean_cls)
        
        # Create filter and validator data
        self.filter_data, self.validator_data = self._prepare_data(self.obs_data, intervention_data, plot_data)
        
        
        CP.info("\n")
        CP.info(DASH)
        CP.info("Observational data length: " + str(observation_data.T))
        CP.info("Interventional data length: " + str(sum([d.T for d in intervention_data.values()])))
        CP.info("Min lag time: " + str(min_lag))
        CP.info("Max lag time: " + str(max_lag))
        CP.info("Filter significance level: " + str(f_alpha))
        CP.info("PCMCI significance level: " + str(alpha))
        CP.info("Selection method: " + sel_method.name)
        
            
    @property    
    def isThereInterv(self) -> bool:
        """
        Check whether an intervention is present or not.

        Returns:
            bool: flag to identify if an intervention is present or not.
        """
        return len(list(self.sys_context.keys())) > 0
    
    
    def JCI_assumptions(self):
        """Initialise the algorithm initial causal structure with the JCI assumptions."""
        # ! JCI Assmpution 1: No system variable causes any context variable
        # ! JCI Assmpution 2: No context variable is confounded with a system variable
        # ! JCI Assmpution 3: The context distribution contains no (conditional) independences

        knowledge = {self.vars.index(f): dict() for f in self.vars}
        
        # ! JCI Assmpution 1
        for k in self.contexts:
            for x in self.systems:
                for tau_i in range(0, self.max_lag + 1):
                    knowledge[self.vars.index(k)][(self.vars.index(x), -tau_i)] = ''
            
        # ! JCI Assmpution 2
        for k in self.contexts:
            for x in self.systems:
                if x not in self.sys_context or (x in self.sys_context and k != self.sys_context[x]):
                    for tau_i in range(0, self.max_lag + 1): knowledge[self.vars.index(x)][(self.vars.index(k), -tau_i)] = ''
                elif x in self.sys_context and k == self.sys_context[x]:
                    knowledge[self.vars.index(x)][(self.vars.index(k), 0)] = '-->'
                    knowledge[self.vars.index(k)][(self.vars.index(x), 0)] = '<--'
                    for tau_i in range(1, self.max_lag + 1): knowledge[self.vars.index(x)][(self.vars.index(k), -tau_i)] = ''
                    for tau_i in range(1, self.max_lag + 1): knowledge[self.vars.index(k)][(self.vars.index(x), -tau_i)] = ''
        
        # ! JCI Assmpution 3
        for k1 in self.contexts:
            for k2 in remove_from_list(self.contexts, k1):
                knowledge[self.vars.index(k1)][(self.vars.index(k2), 0)] = '<->'
                # for tau_i in range(0, self.max_lag + 1): knowledge[self.vars.index(k1)][(self.vars.index(k2), -tau_i)] = '<->'
        
        # ! This models the context variables as chain across different time steps
        for k in self.contexts:
            for tau_i in range(1, self.max_lag + 1):
                knowledge[self.vars.index(k)][(self.vars.index(k), -tau_i)] = '-->' if tau_i == 1 else ''
        
                          
                
                                              
        out = {}
        for j in range(len(self.vars)):
            inner_dict = {} 
            
            for i in range(len(self.vars)):
                for tau_i in range(0, self.max_lag + 1):
                    if tau_i > 0 or i != j:
                        value = "o?>" if tau_i > 0 else "o?o"
                        inner_dict[(i, -tau_i)] = value
                           
            out[j] = inner_dict

        for j, links_j in knowledge.items():
            for (i, lag_i), link_ij in links_j.items():
                if link_ij == "":
                    del out[j][(i, lag_i)]
                else: 
                    out[j][(i, lag_i)] = link_ij
        return out
    

    def run_filter(self):
        """Run filter method."""
        CP.info("Selecting relevant features among: " + str(self.filter_data.features))
       
        self.sel_method.initialise(self.obs_data, self.f_alpha, self.min_lag, self.max_lag, self.CM)
        self.CM = self.sel_method.compute_dependencies()
        
    
    def run_validator(self, link_assumptions = None) -> DAG:
        """
        Run Validator (LPCMCI).

        Args:
            link_assumptions (dict, optional): link assumption with context. Defaults to None.

        Returns:
            DAG: causal model with context.
        """
        self.validator = myLPCMCI(self.validator_data,
                                self.max_lag,
                                self.sys_context,
                                self.val_condtest,
                                CP.verbosity,
                                self.alpha)
        causal_model = self.validator.run(link_assumptions)
        causal_model.sys_context = self.CM.sys_context      
 
        return causal_model
      
    
    def run(self, remove_unneeded = True, nofilter = True) -> DAG:
        """
        Run CAnDOIT.
        
        Args:
            remove_unneeded (bool, optional): Bit to remove unneeded (isolated) variables. Defaults to True.
            nofilter (bool, optional): Bit to run CAnDOIT without filter. Defaults to False.
            
        Returns:
            DAG: causal model.
        """
        link_assumptions = None
            
        if not nofilter:
            #FIXME: to include also the filter. for now this is wrong
            ## 1. FILTER
            self.run_filter()
        
            # list of selected features based on filter dependencies
            self.CM.remove_unneeded_features()
            if not self.CM.features: return None, None
            
            self.obs_data.shrink(self.CM.features)
            f_dag = copy.deepcopy(self.CM)
        
            ## 2. VALIDATOR
            # Add dependencies corresponding to the context variables 
            # ONLY if the the related system variable is still present
            self.CM.add_context() 

            # shrink dataframe d by using the filter result
            self.validator_data.shrink(self.CM.features)
                    
            # selected links to check by the validator
            link_assumptions = self.CM.get_link_assumptions()
                
        else:
            # fullg = DAG(self.validator_data.features, self.min_lag, self.max_lag, False)
            # fullg.sys_context = self.CM.sys_context
            link_assumptions = self.JCI_assumptions()
            
        # calculate dependencies on selected links
        self.CM = self.run_validator(link_assumptions)
               
        # list of selected features based on validator dependencies
        if remove_unneeded: self.CM.remove_unneeded_features()
        if self.exclude_context: self.CM.remove_context()

        self.save()
            
        return self.CM
            
    
    def load(self, res_path):
        """
        Load previously estimated result.

        Args:
            res_path (str): pickle file path.
        """
        with open(res_path, 'rb') as f:
            r = pickle.load(f)
            self.CM = r['causal_model']
            self.f_alpha = r['filter_alpha']
            self.alpha = r['alpha']
            self.dag_path = r['dag_path']
            self.ts_dag_path = r['ts_dag_path']
            
            
    def save(self):
        """Save causal discovery result as pickle file if resfolder is set."""
        if self.respath is not None:
            if self.CM:
                res = dict()
                res['causal_model'] = copy.deepcopy(self.CM)
                res['features'] = copy.deepcopy(self.CM.features)
                res['filter_alpha'] = self.f_alpha
                res['alpha'] = self.alpha
                res['dag_path'] = self.dag_path
                res['ts_dag_path'] = self.ts_dag_path
                with open(self.respath, 'wb') as resfile:
                    pickle.dump(res, resfile)
            else:
                CP.warning("Causal model impossible to save")
                
                
    def _prepare_data(self, obser_data, inter_data, plot_data):
        """
        Prepare data for filter and validator phases.
        
        Args:
            obser_data (Data): observational data.
            inter_data (Data): interventional data.
            plot_data (bool): boolean bit to plot the generated data.

        Returns:
            Data, Data: filter data obj and validator data obj.
        """
        # Filter phase data preparation
        filter_data = copy.deepcopy(obser_data.d)
        for int_data in inter_data.values(): filter_data = pd.concat([filter_data, int_data.d], axis = 0, ignore_index = True)
        filter_data = Data(filter_data, vars = obser_data.features)
        
        # Validator phase data preparation
        validator_data = copy.deepcopy(obser_data.d)
        context_vars = dict()
        for int_var, int_data in inter_data.items():
            
            # Create context variable name
            context_varname = 'C' + int_var
            
            # Store a dict of context variable and system variable corresponding to an intervention
            self.CM.sys_context[int_var] = context_varname
            
            # Create context variable data
            # context_data = np.ones(shape=int_data.d[int_var].shape)
            context_data = int_data.d[int_var]
            context_start = len(validator_data)
            context_end = context_start + len(context_data)
            context_vars[context_varname] = {'data': context_data, 'start': context_start, 'end': context_end}
            
            validator_data = pd.concat([validator_data, int_data.d], axis = 0, ignore_index = True)
            
        for var in context_vars:
            new_column = np.zeros(shape = (len(validator_data),))
            new_column[context_vars[var]['start']: context_vars[var]['end']] = context_vars[var]['data']
            validator_data[var] = new_column
        
        validator_data = Data(validator_data, vars = list(validator_data.columns))
        
        if plot_data: validator_data.plot_timeseries()
        return filter_data, validator_data