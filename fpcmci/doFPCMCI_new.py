import sys
import copy
import pickle
import numpy as np
import pandas as pd
from tigramite.independence_tests.independence_tests_base import CondIndTest
from fpcmci.selection_methods.SelectionMethod import SelectionMethod
from fpcmci.CPrinter import CPLevel, CP
from fpcmci.basics.constants import *
from fpcmci.graph.DAG import DAG
from fpcmci.basics.logger import Logger
import fpcmci.basics.utils as utils
from fpcmci.PCMCI_new import PCMCI
from fpcmci.preprocessing.data import Data 


class doFPCMCI():
    """
    doFPCMCI class.

    doFPCMCI is a causal feature selector framework for large-scale time series datasets. 
    Starting from a Data object and it selects the main features responsible for the
    evolution of the analysed system. Based on the selected features, the framework outputs a causal model.
    """

    def __init__(self, 
                 observation_data: Data, 
                 intervention_data: dict, 
                 min_lag, max_lag,
                 sel_method: SelectionMethod, val_condtest: CondIndTest, 
                 verbosity: CPLevel, 
                 f_alpha = 0.05, 
                 pcmci_alpha = 0.05, 
                 resfolder = None,
                 neglect_only_autodep = False,
                 exclude_context = True,
                 plot_data = False):
        """
        FPCMCI class contructor

        Args:
            observation_data (Data): observational data to analyse
            intervention_data (dict): interventional data to analyse in the form {INTERVENTION_VARIABLE : Data (same variables of observation_data)}
            min_lag (int): minimum time lag
            max_lag (int): maximum time lag
            sel_method (SelectionMethod): selection method
            val_condtest (CondIndTest): validation method
            verbosity (CPLevel): verbosity level
            f_alpha (float, optional): filter significance level. Defaults to 0.05.
            pcmci_alpha (float, optional): PCMCI significance level. Defaults to 0.05.
            resfolder (string, optional): result folder to create. Defaults to None.
            neglect_only_autodep (bool, optional): Bit for neglecting variables with only autodependency. Defaults to False.
            exclude_context (bool, optional): Bit for neglecting context variables. Defaults to False.
        """
        
        self.obs_data = observation_data
        self.f_alpha = f_alpha
        self.pcmci_alpha = pcmci_alpha
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.sel_method = sel_method
        self.CM = DAG(self.obs_data.features, min_lag, max_lag, neglect_only_autodep)
        self.neglect_only_autodep = neglect_only_autodep
        self.exclude_context = exclude_context
        
        # Create filter and validator data
        self.filter_data, self.validator_data = self.__prepare_data(self.obs_data, intervention_data, plot_data)

        self.respath, self.dag_path, self.ts_dag_path = None, None, None
        if resfolder is not None:
            utils.create_results_folder()
            logpath, self.respath, self.dag_path, self.ts_dag_path = utils.get_selectorpath(resfolder)  
            sys.stdout = Logger(logpath)
        
        self.validator = PCMCI(self.pcmci_alpha, self.min_lag, self.max_lag, val_condtest, verbosity, self.CM.sys_context)
        
        CP.set_verbosity(verbosity)


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
        tmp_dag = self.validator.run_pc(self.obs_data, link_assumptions)
        
        # if tmp_dag.autodep_nodes:
                    
        #     tmp_link_assumptions = tmp_dag.get_link_assumptions()
            
        #     # Auto-dependency Check
        #     tmp_dag = self.validator.check_autodependency(self.obs_data, tmp_dag, tmp_link_assumptions)
            
        # Add again context for final MCI test on obs and inter data
        # tmp_dag.add_context()
        
        # shrink dataframe d by using the filter result
        self.validator_data.shrink(tmp_dag.features)
                
        # Causal Model
        link_assumptions = tmp_dag.get_link_assumptions()
        causal_model = self.validator.run_mci(self.validator_data, link_assumptions, tmp_dag.get_parents())
        return causal_model
    

    def run_pcmci(self):
        """
        Run PCMCI
        
        Returns:
            list(str): list of selected variable names
            DAG: causal model
        """
        CP.info("Significance level: " + str(self.pcmci_alpha))
        CP.info("Max lag time: " + str(self.max_lag))
        CP.info("Min lag time: " + str(self.min_lag))
        CP.info("Data length: " + str(self.validator_data.T))
        
        # add context to the fake link assumptions (complete set of links)
        self.CM.fully_connected_dag()
        self.CM.add_context()
        # selected links to check by the validator
        link_assumptions = self.CM.get_link_assumptions()

        # calculate dependencies on selected links
        self.CM = self.run_validator(link_assumptions)
        
        # list of selected features based on validator dependencies
        self.CM.remove_unneeded_features()
        
        if self.exclude_context: self.CM.remove_context()
        
        # Saving final causal model
        self.save()
        
        return self.CM.features, self.CM
    
    
    def run(self):
        """
        Run Selector and Validator
        
        Returns:
            list(str): list of selected variable names
            DAG: causal model
        """
        
        ## 1. FILTER
        self.run_filter()
        
        # list of selected features based on filter dependencies
        self.CM.remove_unneeded_features()
        if not self.CM.features: return None, None
        
        ## 2. VALIDATOR
        # shrink dataframe d by using the filter result
        self.obs_data.shrink(self.CM.features)
        
        # selected links to check by the validator
        link_assumptions = self.CM.get_link_assumptions()
            
        # calculate dependencies on selected links
        f_dag = copy.deepcopy(self.CM)
        self.CM = self.validator.run(self.obs_data, link_assumptions)
        
        # selected links to check by the validator
        self.CM = self.validator.check_interventions(self.filter_data, self.CM)
        
        # list of selected features based on validator dependencies
        self.CM.remove_unneeded_features()
    
        # Saving final causal model
        self.__print_differences(f_dag, self.CM)
        self.save()
        
        return self.CM.features, self.CM
    
    
    # def run(self):
    #     """
    #     Run Selector and Validator
        
    #     Returns:
    #         list(str): list of selected variable names
    #         DAG: causal model
    #     """
        
    #     ## 1. FILTER
    #     self.run_filter()
        
    #     # list of selected features based on filter dependencies
    #     self.CM.remove_unneeded_features()
    #     if not self.CM.features: return None, None
    #     self.obs_data.shrink(self.CM.features)
        
    #     ## 2. VALIDATOR
    #     # Add dependencies corresponding to the context variables 
    #     # ONLY if the the related system variable is still present
    #     # self.CM.add_context() 
        
    #     # selected links to check by the validator
    #     link_assumptions = self.CM.get_link_assumptions()
            
    #     # calculate dependencies on selected links
    #     f_dag = copy.deepcopy(self.CM)
    #     self.CM = self.run_validator(link_assumptions)
        
    #     # list of selected features based on validator dependencies
    #     self.CM.remove_unneeded_features()
    #     self.__print_differences(f_dag, self.CM)
        
    #     if self.exclude_context: self.CM.remove_context()
        
    #     # Saving final causal model
    #     self.save()
        
    #     return self.CM.features, self.CM
      
    
    def dag(self,
            node_layout = 'dot',
            min_width = 1,
            max_width = 5,
            min_score = 0,
            max_score = 1,
            node_size = 8,
            node_color = 'orange',
            edge_color = 'grey',
            font_size = 12,
            label_type = LabelType.Lag,
            save_name = None,
            img_ext = ImageExt.PNG):
        """
        Saves dag plot if resfolder has been set otherwise it shows the figure
        
        Args:
            node_layout (str, optional): Node layout. Defaults to 'dot'.
            min_width (int, optional): minimum linewidth. Defaults to 1.
            max_width (int, optional): maximum linewidth. Defaults to 5.
            min_score (int, optional): minimum score range. Defaults to 0.
            max_score (int, optional): maximum score range. Defaults to 1.
            node_size (int, optional): node size. Defaults to 8.
            node_color (str, optional): node color. Defaults to 'orange'.
            edge_color (str, optional): edge color. Defaults to 'grey'.
            font_size (int, optional): font size. Defaults to 12.
            label_type (LabelType, optional): enum to set whether to show the lag time (LabelType.Lag) or the strength (LabelType.Score) of the dependencies on each link/node or not showing the labels (LabelType.NoLabels). Default LabelType.Lag.
            img_ext (ImageExt, optional): dag image extention (.png, .pdf, ..). Default ImageExt.PNG.
        """
        
        if self.CM:
            if save_name is None: save_name = self.dag_path
            self.CM.dag(node_layout, min_width, 
                        max_width, min_score, max_score,
                        node_size, node_color, edge_color,
                        font_size, label_type, save_name,
                        img_ext)
        else:
            CP.warning("Dag impossible to create: causal model not estimated yet")
    
        
    def timeseries_dag(self,
                       min_width = 1,
                       max_width = 5,
                       min_score = 0,
                       max_score = 1,
                       node_size = 8,
                       font_size = 12,
                       node_color = 'orange',
                       edge_color = 'grey',
                       save_name = None,
                       img_ext = ImageExt.PNG):
        """
        Saves timeseries dag plot if resfolder has been set otherwise it shows the figure
        
        Args:
            min_width (int, optional): minimum linewidth. Defaults to 1.
            max_width (int, optional): maximum linewidth. Defaults to 5.
            min_score (int, optional): minimum score range. Defaults to 0.
            max_score (int, optional): maximum score range. Defaults to 1.
            node_size (int, optional): node size. Defaults to 8.
            node_color (str, optional): node color. Defaults to 'orange'.
            edge_color (str, optional): edge color. Defaults to 'grey'.
            font_size (int, optional): font size. Defaults to 12.
            img_ext (ImageExt, optional): dag image extention (.png, .pdf, ..). Default ImageExt.PNG.
        """
        
        if self.CM:
            if save_name is None: save_name = self.ts_dag_path
            self.CM.ts_dag(self.max_lag, min_width,
                           max_width, min_score, max_score,
                           node_size, node_color, edge_color,
                           font_size, save_name, img_ext)
        else:
            CP.warning("Timeseries dag impossible to create: causal model not estimated yet")
            
    
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
            self.pcmci_alpha = r['pcmci_alpha']
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
                res['pcmci_alpha'] = self.pcmci_alpha
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
                
                
    def __prepare_data(self, obser_data, inter_data, plot_data):
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
            self.CM.g[int_var].intervention_node = True
            
            # Create context variable data
            context_data = int_data.d[int_var]
            context_start = len(validator_data) - 1
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