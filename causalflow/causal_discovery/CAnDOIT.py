import copy
import pickle
import numpy as np
import pandas as pd
from tigramite.independence_tests.independence_tests_base import CondIndTest
from causalflow.causal_discovery.FPCMCI import FPCMCI
from causalflow.selection_methods.SelectionMethod import SelectionMethod
from causalflow.CPrinter import CPLevel, CP
from causalflow.basics.constants import *
from causalflow.graph.DAG import DAG
from causalflow.causal_discovery.myPCMCI import myPCMCI
from causalflow.preprocessing.data import Data 
from causalflow.causal_discovery.CausalDiscoveryMethod import CausalDiscoveryMethod 


class CAnDOIT(CausalDiscoveryMethod):
    """
    CAnDOIT causal discovery method.

    CAnDOIT is a causal discovery method that uses observational and interventional data to reconstruct 
    causal models from large-scale time series datasets.
    Starting from a Data object and it selects the main features responsible for the
    evolution of the analysed system. Based on the selected features, the framework outputs a causal model.
    It extends F-PCMCI by introducing the possibility to perform the causal analysis using 
    observational and interventional data.
    """

    def __init__(self, 
                 observation_data: Data, 
                 intervention_data: dict, 
                 min_lag, max_lag,
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
        CAnDOIT class contructor

        Args:
            observation_data (Data): observational data to analyse
            intervention_data (dict): interventional data to analyse in the form {INTERVENTION_VARIABLE : Data (same variables of observation_data)}
            min_lag (int): minimum time lag
            max_lag (int): maximum time lag
            sel_method (SelectionMethod): selection method
            val_condtest (CondIndTest): validation method
            verbosity (CPLevel): verbosity level
            f_alpha (float, optional): filter significance level. Defaults to 0.05.
            alpha (float, optional): PCMCI significance level. Defaults to 0.05.
            resfolder (string, optional): result folder to create. Defaults to None.
            neglect_only_autodep (bool, optional): Bit for neglecting variables with only autodependency. Defaults to False.
            exclude_context (bool, optional): Bit for neglecting context variables. Defaults to False.
            clean_cls (bool): Clean console bit. Default to True.

        """
        
        self.obs_data = observation_data
        self.f_alpha = f_alpha
        self.sel_method = sel_method
        self.val_condtest = val_condtest
        self.exclude_context = exclude_context
        super().__init__(self.obs_data, min_lag, max_lag, verbosity, alpha, resfolder, neglect_only_autodep, clean_cls)
        
        # Create filter and validator data
        self.filter_data, self.validator_data = self._prepare_data(self.obs_data, intervention_data, plot_data)
        
        self.validator = myPCMCI(self.alpha, self.min_lag, self.max_lag, self.val_condtest, verbosity, self.CM.sys_context)
    
    @property    
    def isThereInterv(self):
        """
        is there an intervention?

        Returns:
            bool: flag to identify if an intervention is present or not
        """
        return len(list(self.CM.sys_context.keys())) > 0
        

    def run_filter(self):
        """
        Run filter method
        """
        CP.info("\n")
        CP.info(DASH)
        CP.info("Selecting relevant features among: " + str(self.filter_data.features))
        CP.info("Selection method: " + self.sel_method.name)
        CP.info("Significance level: " + str(self.f_alpha))
        CP.info("Max lag time: " + str(self.max_lag))
        CP.info("Min lag time: " + str(self.min_lag))
        CP.info("Data length: " + str(self.filter_data.T))
       
        self.sel_method.initialise(self.obs_data, self.f_alpha, self.min_lag, self.max_lag, self.CM)
        self.CM = self.sel_method.compute_dependencies()
        
    
    def run_validator(self, link_assumptions = None):
        """
        Runs Validator (PCMCI)

        Args:
            link_assumptions (dict, optional): link assumption with context. Defaults to None.

        Returns:
            DAG: causal model with context
        """
        # Run PC algorithm on selected links
        tmp_dag = self.validator.run_pc(self.validator_data, self.min_lag, link_assumptions)
        tmp_dag.sys_context = self.CM.sys_context
        
        if tmp_dag.autodep_nodes:
        
            # Remove context from parents
            tmp_dag.remove_context()
            
            tmp_link_assumptions = tmp_dag.get_link_assumptions()
            
            # Auto-dependency Check
            tmp_dag = self.validator.check_autodependency(self.obs_data, tmp_dag, tmp_link_assumptions, self.min_lag)
            
            # Add again context for final MCI test on obs and inter data
            tmp_dag.add_context()
        
        # Causal Model
        causal_model = self.validator.run_mci(self.validator_data, tmp_dag, self.min_lag)
        causal_model = self._change_score_and_pval(tmp_dag, causal_model) 
        return causal_model
    
    
    def _change_score_and_pval(self, orig_cm: DAG, dest_cm: DAG):
        for t in dest_cm.g:
            if dest_cm.g[t].is_autodependent:
                for s in dest_cm.g[t].autodependency_links:
                    dest_cm.g[t].sources[s][SCORE] = orig_cm.g[t].sources[s][SCORE]
                    dest_cm.g[t].sources[s][PVAL] = orig_cm.g[t].sources[s][PVAL]
        return dest_cm
    
    
    def run(self, remove_unneeded = True, nofilter = False) -> DAG:
        """
        Run CAnDOIT
        
        Returns:
            DAG: causal model
        """
        
        if not self.isThereInterv:
            
            fpcmci = FPCMCI(self.obs_data,
                            self.min_lag,
                            self.max_lag,
                            self.sel_method,
                            self.val_condtest,
                            CP.verbosity,
                            self.f_alpha,
                            self.alpha,
                            self.resfolder,
                            self.neglect_only_autodep)
            self.CM = fpcmci.run()
            
        else:
        
            link_assumptions = None
            
            if not nofilter:
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
            
            # calculate dependencies on selected links
            self.CM = self.run_validator(link_assumptions)
                   
            # list of selected features based on validator dependencies
            if remove_unneeded: self.CM.remove_unneeded_features()
            if self.exclude_context: self.CM.remove_context()
            
            # Print and save final causal model
            if not nofilter: self.__print_differences(f_dag, self.CM)
            self.save()
            
        if self.resfolder is not None: self.logger.close()
        return self.CM
    
            
    
    def load(self, res_path):
        """
        Loads previously estimated result 

        Args:
            res_path (str): pickle file path
        """
        with open(res_path, 'rb') as f:
            r = pickle.load(f)
            self.CM = r['causal_model']
            self.f_alpha = r['filter_alpha']
            self.alpha = r['alpha']
            self.dag_path = r['dag_path']
            self.ts_dag_path = r['ts_dag_path']
            
            
    def save(self):
        """
        Save causal discovery result as pickle file if resfolder is set
        """
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
     

    def __print_differences(self, old_dag : DAG, new_dag : DAG):
        """
        Print difference between old and new dependencies

        Args:
            old_dep (DAG): old dag
            new_dep (DAG): new dag
        """
        # Check difference(s) between validator and filter dependencies
        list_diffs = list()
        tmp = copy.deepcopy(old_dag)
        for t in tmp.g:
            if t not in new_dag.g:
                list_diffs.append(t)
                continue
                
            for s in tmp.g[t].sources:
                if s not in new_dag.g[t].sources:
                    list_diffs.append((s[0], s[1], t))
        
        if list_diffs:
            CP.info("\n")
            CP.info(DASH)
            CP.info("Difference(s):")
            for diff in list_diffs: 
                if type(diff) is tuple:
                    CP.info("Removed (" + str(diff[0]) + " -" + str(diff[1]) +") --> (" + str(diff[2]) + ")")
                else:
                    CP.info(diff + " removed")
                
                
    def _prepare_data(self, obser_data, inter_data, plot_data):
        """
        Prepares data for filter and validator phases
        
        Args:
            obser_data (Data): observational data
            inter_data (Data): interventional data
            plot_data (bool): boolean bit to plot the generated data

        Returns:
            Data, Data: filter data obj and validator data obj     
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
            context_data = int_data.d[int_var]
            # FIXME: context_start = len(validator_data) - 1
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