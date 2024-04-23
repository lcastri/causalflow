from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
import numpy as np
import pandas as pd

from causalflow.dbn.Density import Density
from causalflow.dbn.Process import Process
import copy
    
class DynamicBayesianNetwork():
    def __init__(self, dag: DAG, data: Data):
        """
        DynamicBayesianNetwork contructer

        Args:
            dag (DAG): DAG to convert into a DBN
            data (Data): observational data
        """
        # observation variables
        self.dag = dag
        self.dbn = dict()
        self.obsD = data
        self._DAG2DBN()
        
        # intervention variables
        self.dags = {node: copy.deepcopy(self.dag) for node in self.dag.g}
        for target in self.dags.keys():
            for s in self.dag.g[target].sources:
                self.dags[target].del_source(target, s[0], s[1])
            
        self.intD = {}
        self.dbn_int = dict()
        
        
    def _DAG2DBN(self):
        """
        converts the DAG to a DBN
        """
        self.dbn = {node: None for node in self.dag.g}
        for node in self.dbn:
            Y = Process(self.obsD.d[node].to_numpy(), node, 0)
            parents = self._extract_parents(self.obsD.d, self.dag, node)
            self.dbn[node] = Density(Y, parents)            
            self.dbn[node].ComputeDensity()
            
            
    def whatHappensTo(self, target) -> Density:
        """
        wrap method that extracts the density of a function from the dbn

        Args:
            target (str): target variable name

        Returns:
            Density: density function of the target variable
        """
        return self.dbn[target]
    
    
    def predictEffect(self, var: str, data: np.ndarray):
        """
        predicts the effect of the system if a variable assumes a certain value(s)

        Args:
            var (str): variable name
            data (np.ndarray): data

        Returns:
            np.array: time-series data representing the effect 
        """
        estimated_data = np.zeros((len(data) + self.dag.max_lag, len(self.dag.features)))
        estimated_data[:self.dag.max_lag] = self.obsD.d[len(self.obsD.d) - self.dag.max_lag:]
        estimated_data[self.dag.max_lag:, self.dag.features.index(var)] = data

        for t in range(self.dag.max_lag, estimated_data.shape[0]):
            for f_idx, f in enumerate(self.dag.features):
                if f != var:
                    given_parents = {s[0]: estimated_data[t - s[1], self.dag.features.index(s[0])] for s in self.dag.g[f].sources}
                    _, expectation = self.whatHappensTo(f).If(given_parents)
                    estimated_data[t, f_idx] = expectation
        
        for f_idx, f in enumerate(self.dag.features): 
            if np.any(np.isnan(estimated_data[:, f_idx])):
                estimated = pd.Series(estimated_data[:, f_idx])
                estimated.interpolate(inplace=True)
                estimated_data[:, f_idx] = estimated.values
        
        return estimated_data[self.dag.max_lag:]
    
    
    # FIXME: need to understand how to model this. I think I need to create a new Desity for
    # the intervention variable (no longer with parents!) and add the interventional dataset
    # to the dataset self.data (obs + int) for all the other variables
    def addInterventionData(self, target: str, data: Data):
        """
        adds interventional data

        Args:
            target (_type_): _description_
            data (_type_): _description_
        """
        self.intD[target] = data
        #FIXME: handle multiple intervention on the same variable
        self.dbn_int[target] = {node: None for node in self.dag.g}
                
        for node in self.dbn_int[target]:
            Y = Process(self.intD[target].d[node].to_numpy(), node, 0)
            parents = self._extract_parents(self.intD[target].d, self.dags[target], node)
            self.dbn_int[target][node] = Density(Y, parents)            
            self.dbn_int[target][node].ComputeDensity()
    
    
    # # FIXME: need to understand how to model this
    # def predictDoEffect(self, var: str, data: np.ndarray):
    #     """
    #     predicts the effect of the system caused by an intervention

    #     Args:
    #         var (str): manipulated variable name
    #         data (np.ndarray): manipulated data

    #     Returns:
    #         np.array: time-series data representing the effect of a certain manipulation
    #     """
    #     if not self.intD:
    #         raise ValueError("No interventional data provided")

    #     estimated_data = np.zeros((len(data) + self.dag.max_lag, len(self.dag.features)))
    #     estimated_data[:self.dag.max_lag] = self.intD.d[len(self.obsD.d) - self.dag.max_lag:]
    #     estimated_data[self.dag.max_lag:, self.dag.features.index(var)] = data

    #     for t in range(self.dag.max_lag, estimated_data.shape[0]):
    #         for f_idx, f in enumerate(self.dag.features):
    #             if f != var:
    #                 given_parents = {s[0]: estimated_data[t - s[1], self.dag.features.index(s[0])] for s in self.dag.g[f].sources}
    #                 _, expectation = self.whatHappensTo(f).If(given_parents)
    #                 estimated_data[t, f_idx] = expectation
        
    #     for f_idx, f in enumerate(self.dag.features): 
    #         if np.any(np.isnan(estimated_data[:, f_idx])):
    #             estimated = pd.Series(estimated_data[:, f_idx])
    #             estimated.interpolate(inplace=True)
    #             estimated_data[:, f_idx] = estimated.values
        
    #     return estimated_data[self.dag.max_lag:]
    
    
    def _extract_parents(self, d, dag, node):
        parents = {s[0]: Process(d[s[0]].to_numpy(), s[0], s[1]) for s in dag.g[node].sources}
        if not parents: return None
        return parents