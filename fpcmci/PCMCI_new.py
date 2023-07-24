from tigramite.pcmci import PCMCI as VAL
from tigramite.independence_tests.independence_tests_base import CondIndTest
import tigramite.data_processing as pp
import numpy as np
from fpcmci.CPrinter import CPLevel, CP
from fpcmci.basics.constants import *
from fpcmci.graph.DAG import DAG
from fpcmci.preprocessing.data import Data


class PCMCI():
    """
    PCMCI class.

    PCMCI works with FSelector in order to find the causal 
    model starting from a prefixed set of variables and links.
    """
    def __init__(self, alpha, min_lag, max_lag, val_condtest: CondIndTest, verbosity: CPLevel, sys_context = dict()):
        """
        PCMCI class constructor

        Args:
            alpha (float): significance level
            min_lag (int): minimum time lag
            max_lag (int): maximum time lag
            val_condtest (CondIndTest): validation method
            verbosity (CPLevel): verbosity level
        """
        self.alpha = alpha
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.result = None
        self.dependencies = None
        self.val_method = None
        self.val_condtest = val_condtest
        self.verbosity = verbosity.value
        self.sys_context = sys_context
        

    def run(self, data: Data, link_assumptions = None):
        """
        Run causal discovery algorithm

        Args:
            data (Data): Data obj to analyse
            link_assumptions (dict, optional): prior assumptions on causal model links. Defaults to None.

        Returns:
            (DAG): estimated causal model
        """
        
        CP.info('\n')
        CP.info(DASH)
        CP.info("Running Causal Discovery Algorithm")

        # build tigramite dataset
        vector = np.vectorize(float)
        d = vector(data.d)
        dataframe = pp.DataFrame(data = d, var_names = data.features)
        
        # init and run pcmci
        self.val_method = VAL(dataframe = dataframe,
                              cond_ind_test = self.val_condtest,
                              verbosity = self.verbosity)

        self.result = self.val_method.run_pcmci(link_assumptions = link_assumptions,
                                                tau_max = self.max_lag,
                                                tau_min = self.min_lag,
                                                alpha_level = self.alpha)
        
        self.result['var_names'] = data.features
        self.result['pretty_var_names'] = data.pretty_features
        
        self.dependencies = self.__PCMCI_to_DAG()
        return self.dependencies
    
    
    def run_pc(self, data: Data, link_assumptions = None):
        """
        Run PC causal discovery algorithm

        Args:
            data (Data): Data obj to analyse
            link_assumptions (dict, optional): prior assumptions on causal model links. Defaults to None.

        Returns:
            (DAG): estimated parents
        """
        
        CP.info('\n')
        CP.info(DASH)
        CP.info("Running Causal Discovery Algorithm")

        # build tigramite dataset
        vector = np.vectorize(float)
        d = vector(data.d)
        dataframe = pp.DataFrame(data = d, var_names = data.features)
        
        # init and run pcmci
        self.val_method = VAL(dataframe = dataframe,
                              cond_ind_test = self.val_condtest,
                              verbosity = self.verbosity)
        
        parents = self.val_method.run_pc_stable(link_assumptions = link_assumptions,
                                                tau_max = self.max_lag,
                                                tau_min = self.min_lag,
                                                pc_alpha = 0.05)
    
        return self.__PC_to_DAG(parents, data.features)
    
    
    def check_interventions(self, data: Data, dag: DAG):
        """
        Performs MCI test

        Args:
            link_assumptions (dict, optional): link assumptions. Defaults to None.
            parents (dict, optional): parents dictionary. Defaults to None.

        Returns:
            (dict): MCI result
        """
        CP.info("\n##")
        CP.info("## Interventions test analysis")
        CP.info("##")

        # build tigramite dataset
        vector = np.vectorize(float)
        d = vector(data.d)
        dataframe = pp.DataFrame(data = d, var_names = data.features)
        
        # init and run pcmci
        self.val_method = VAL(dataframe = dataframe,
                              cond_ind_test = self.val_condtest,
                              verbosity = 0)
        
        link_assumptions = dag.get_link_assumptions_for_interventions()
        parents = dag.get_parents()
        
        _int_link_assumptions = self.val_method._set_link_assumptions(link_assumptions, self.min_lag, self.max_lag)

        # Set the maximum condition dimension for Y and X
        max_conds_py = self.val_method._set_max_condition_dim(None, self.min_lag, self.max_lag)
        max_conds_px = 0
        
        # Get the parents that will be checked
        _int_parents = self.val_method._get_int_parents(parents)
         
        # Get the conditions as implied by the input arguments
        for j, i, tau, Z in self.val_method._iter_indep_conds(_int_parents,
                                                              _int_link_assumptions,
                                                              max_conds_py,
                                                              max_conds_px):
            
            # Set X and Y (for clarity of code)
            X = [(i, tau)]
            Y = [(j, 0)]
            if (data.features[i], abs(tau), data.features[j]) not in dag.interventions_links: continue
            
            CP.info("\tlink: (" + data.features[i] + " " + str(tau) + ") -?> (" + data.features[j] + "):")
            val, pval = self.val_method.cond_ind_test.run_test(X, Y, Z=Z, tau_max=self.max_lag)

            if pval > self.alpha:
                dag.del_source(data.features[j], data.features[i], abs(tau))
                CP.info("\t|val = " + str(round(val,3)) + " |pval = " + str(str(round(pval,3))) + " -- removed\n")

            else:
                dag.g[data.features[j]].sources[(data.features[i], abs(tau))][SCORE] = val
                dag.g[data.features[j]].sources[(data.features[i], abs(tau))][PVAL] = pval
                CP.info("\t|val = " + str(round(val,3)) + " |pval = " + str(str(round(pval,3))) + " -- ok\n")

        return dag
    
    
    def check_autodependency(self, data: Data, dag:DAG, link_assumptions) -> DAG:
        """
        Run MCI test on observational data using the causal structure computed by the validator 

        Args:
            data (Data): Data obj to analyse
            dag (DAG): causal model
            link_assumptions (dict): prior assumptions on causal model links. Defaults to None.

        Returns:
            (DAG): estimated causal model
        """
        
        CP.info("\n##")
        CP.info("## Auto-dependency check on intervention variables by MCI test analysis")
        CP.info("##")
        
        # build tigramite dataset
        vector = np.vectorize(float)
        d = vector(data.d)
        dataframe = pp.DataFrame(data = d, var_names = data.features)
        
        # init and run pcmci
        self.val_method = VAL(dataframe = dataframe,
                              cond_ind_test = self.val_condtest,
                              verbosity = 0)
        
        _int_link_assumptions = self.val_method._set_link_assumptions(link_assumptions, self.min_lag, self.max_lag)

        # Set the maximum condition dimension for Y and X
        max_conds_py = self.val_method._set_max_condition_dim(None, self.min_lag, self.max_lag)
        max_conds_px = self.val_method._set_max_condition_dim(None, self.min_lag, self.max_lag)
        
        # Get the parents that will be checked
        _int_parents = self.val_method._get_int_parents(dag.get_parents())

        # Get the conditions as implied by the input arguments
        links_tocheck = self.val_method._iter_indep_conds(_int_parents, _int_link_assumptions, max_conds_py, max_conds_px)
        for j, i, tau, Z in links_tocheck:
            if data.features[j] not in dag.autodep_int_nodes or j != i: continue
            else:
                # Set X and Y (for clarity of code)
                X = [(i, tau)]
                Y = [(j, 0)]
                
                CP.info("\tlink: (" + data.features[i] + " " + str(tau) + ") -?> (" + data.features[j] + "):")
                # Run the independence tests and record the results
                val, pval = self.val_method.cond_ind_test.run_test(X, Y, Z = Z, tau_max = self.max_lag)
                if pval > self.alpha:
                    dag.del_source(data.features[j], data.features[j], abs(tau))
                    CP.info("\t|val = " + str(round(val,3)) + " |pval = " + str(str(round(pval,3))) + " -- removed")

                else:
                    CP.info("\t|val = " + str(round(val,3)) + " |pval = " + str(str(round(pval,3))) + " -- ok")
                
        return dag
 
    
    def run_mci(self, data: Data, dag:DAG, link_assumptions, parents):
        """
        Run MCI test on observational data using the causal structure computed by the validator 

        Args:
            data (Data): Data obj to analyse
            link_assumptions (dict): prior assumptions on causal model links. Defaults to None.
            parents (dict): causal structure

        Returns:
            (DAG): estimated causal model
        """
        
        CP.info("\n##")
        CP.info("## MCI test analysis")
        CP.info("##")

        # build tigramite dataset
        vector = np.vectorize(float)
        d = vector(data.d)
        dataframe = pp.DataFrame(data = d, var_names = data.features)
        
        # init and run pcmci
        self.val_method = VAL(dataframe = dataframe,
                              cond_ind_test = self.val_condtest,
                              verbosity = self.verbosity)

        self.result = self.__my_mci(link_assumptions = link_assumptions,
                                    parents = parents)
        
        self.result['var_names'] = data.features
        self.result['pretty_var_names'] = data.pretty_features
        
        self.dependencies = self.__PCMCI_to_DAG()
        return self.dependencies
    
    
    def __PC_to_DAG(self, parents, features):
        """
        Re-elaborates the PC result in a dag

        Args:
            parents (dict): causal structure
            features (list): feature list

        Returns:
            (DAG): pc result re-elaborated
        """
        tmp_dag = DAG(features, self.min_lag, self.max_lag)
        tmp_dag.sys_context = self.sys_context
        for t in parents:
            for s in parents[t]:
                if features[t] in self.sys_context.keys() and features[s[0]] == self.sys_context[features[t]]:
                    tmp_dag.g[features[t]].intervention_node = True
                    tmp_dag.g[features[t]].associated_context = features[s[0]]
                tmp_dag.add_source(features[t], features[s[0]], 1, 0, s[1])
        return tmp_dag
    
    
    def __PCMCI_to_DAG(self):
        """
        Re-elaborates the PCMCI result in a new dictionary

        Returns:
            (DAG): pcmci result re-elaborated
        """
        vars = self.result['var_names']
        tmp_dag = DAG(vars, self.min_lag, self.max_lag)
        tmp_dag.sys_context = self.sys_context
        for k in tmp_dag.sys_context: tmp_dag.g[k].intervention_node = True
        N, lags = self.result['graph'][0].shape
        for s in range(len(self.result['graph'])):
            for t in range(N):
                for lag in range(lags):
                    if self.result['graph'][s][t,lag] == '-->':
                        if vars[t] in self.sys_context.keys() and vars[s] == self.sys_context[vars[t]]:
                            tmp_dag.g[vars[t]].intervention_node = True
                            tmp_dag.g[vars[t]].associated_context = vars[s]
                        tmp_dag.add_source(vars[t], 
                                           vars[s],
                                           self.result['val_matrix'][s][t,lag],
                                           self.result['p_matrix'][s][t,lag],
                                           lag)
        return tmp_dag