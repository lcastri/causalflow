"""
This module provides the FPCMCI class.

Classes:
    FPCMCI: class containing the FPCMCI causal discovery algorithm.
"""

import copy
import pickle
from tigramite.independence_tests.independence_tests_base import CondIndTest
from causalflow.graph.DAG import DAG
from causalflow.selection_methods.SelectionMethod import SelectionMethod
from causalflow.CPrinter import CPLevel, CP
from causalflow.basics.constants import *
from causalflow.causal_discovery.support.myPCMCI import myPCMCI
from causalflow.preprocessing.data import Data 
from causalflow.causal_discovery.CausalDiscoveryMethod import CausalDiscoveryMethod 


class FPCMCI(CausalDiscoveryMethod):
    """F-PCMCI causal discovery method."""

    def __init__(self, 
                 data: Data, 
                 min_lag, max_lag, 
                 sel_method: SelectionMethod, val_condtest: CondIndTest, 
                 verbosity: CPLevel, 
                 f_alpha = 0.05, 
                 alpha = 0.05, 
                 resfolder = None,
                 neglect_only_autodep = False,
                 clean_cls = True):
        """
        Class contructor.

        Args:
            data (Data): data to analyse.
            min_lag (int): minimum time lag.
            max_lag (int): maximum time lag.
            sel_method (SelectionMethod): selection method.
            val_condtest (CondIndTest): validation method.
            verbosity (CPLevel): verbosity level.
            f_alpha (float, optional): filter significance level. Defaults to 0.05.
            alpha (float, optional): PCMCI significance level. Defaults to 0.05.
            resfolder (string, optional): result folder to create. Defaults to None.
            neglect_only_autodep (bool, optional): Bit for neglecting variables with only autodependency. Defaults to False.
            clean_cls (bool): Clean console bit. Default to True.
        """
        super().__init__(data, min_lag, max_lag, verbosity, alpha, resfolder, neglect_only_autodep, clean_cls)
        
        self.f_alpha = f_alpha
        self.sel_method = sel_method
        
        self.validator = myPCMCI(self.alpha, min_lag, max_lag, val_condtest, verbosity, neglect_only_autodep = neglect_only_autodep)       


    def run_filter(self):
        """Run filter method."""
        CP.info("\n")
        CP.info(DASH)
        CP.info("Selecting relevant features among: " + str(self.data.features))
        CP.info("Selection method: " + self.sel_method.name)
        CP.info("Significance level: " + str(self.f_alpha))
        CP.info("Max lag time: " + str(self.max_lag))
        CP.info("Min lag time: " + str(self.min_lag))
        CP.info("Data length: " + str(self.data.T))
       
        self.sel_method.initialise(self.data, self.f_alpha, self.min_lag, self.max_lag, self.CM)
        self.CM = self.sel_method.compute_dependencies()  

   
    def run(self, remove_unneeded = True, nofilter = False) -> DAG:
        """
        Run F-PCMCI.
        
        Args:
            remove_unneeded (bool, optional): Bit to remove unneeded (isolated) variables. Defaults to True.
            nofilter (bool, optional): Bit to run F-PCMCI without filter. Defaults to False.

        Returns:
            DAG: causal model.
        """
        link_assumptions = None
        
        if not nofilter:
            ## 1. FILTER
            self.run_filter()
        
            # list of selected features based on filter dependencies
            self.CM.remove_unneeded_features()
            if not self.CM.features: return None, None
        
            ## 2. VALIDATOR
            # shrink dataframe d by using the filter result
            self.data.shrink(self.CM.features)
        
            # selected links to check by the validator
            link_assumptions = self.CM.get_link_assumptions()
            
            # calculate dependencies on selected links
            f_dag = copy.deepcopy(self.CM)
            
        if self.min_lag != 0:
            self.CM = self.validator.run(self.data, link_assumptions)
        else:
            self.CM = self.validator.run_plus(self.data, link_assumptions)
            
        # list of selected features based on validator dependencies
        if remove_unneeded: self.CM.remove_unneeded_features()
    
        # Saving final causal model
        if not nofilter: self._print_differences(f_dag, self.CM)
        self.save()
        
        if self.resfolder is not None: self.logger.close()
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
    
    
    def _print_differences(self, old_dag : DAG, new_dag : DAG):
        """
        Print difference between old and new dependencies.

        Args:
            old_dep (DAG): old dag.
            new_dep (DAG): new dag.
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