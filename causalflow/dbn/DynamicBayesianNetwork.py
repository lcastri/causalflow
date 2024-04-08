from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
import numpy as np
import pandas as pd

from causalflow.dbn.Density import Density
from causalflow.dbn.Process import Process

OBS = "obs"
INT = "int"
    
class DynamicBayesianNetwork():
    def __init__(self, dag: DAG, data: Data):
        """
        DynamicBayesianNetwork contructer

        Args:
            dag (DAG): DAG to convert into a DBN
            data (Data): data used to generate the DAG
        """
        self.dag = dag
        self.obsD = data
        self._DAG2DBN()
        
        
    def _DAG2DBN(self):
        """
        converts the DAG to a DBN
        """
        self.dbn = {node: {OBS: None, INT: None} for node in self.dag.g}
        for node in self.dbn:
            Y = Process(self.obsD.d[node].to_numpy(), node, 0)
            parents = {s[0]: Process(self.obsD.d[s[0]].to_numpy(), s[0], s[1]) for s in self.dag.g[node].sources}
            if not parents: parents = None
            self.dbn[node][OBS] = Density(Y, parents)            
            self.dbn[node][OBS].ConditionalDensity()
            
            
    def whatHappensTo(self, target) -> Density:
        """
        wrap method which extract the density of a function from the dbn

        Args:
            target (str): target variable name

        Returns:
            Density: density function of the target variable
        """
        return self.dbn[target][OBS]
    
    
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
        _summary_

        Args:
            target (_type_): _description_
            data (_type_): _description_
        """
        self.intD = data
        
        Y = Process(self.intD.d[target], target, 0)
        self.dbn[target][INT] = Density(Y)
        
        for node in self.dbn:
            if node == target: continue
            Y = Process(self.intD.d[node].to_numpy(), node, 0)
            parents = {s[0]: Process(self.intD.d[s[0]].to_numpy(), s[0], s[1]) for s in self.dag.g[node].sources}
            self.dbn[node][INT] = Density(Y, parents)            
            self.dbn[node][INT].ConditionalDensity()
    
    
    # FIXME: need to understand how to model this
    # def predictDoEffect(self, var: str, data: np.ndarray):
    #     """
    #     predicts the effect of the system caused by an intervention

    #     Args:
    #         var (str): manipulated variable name
    #         data (np.ndarray): manipulated data

    #     Returns:
    #         np.array: time-series data representing the effect of a certain manipulation
    #     """
    #     estimated_data = np.zeros((len(data) + self.dag.max_lag, len(self.dag.features)))
    #     estimated_data[:self.dag.max_lag] = self.data.d[len(self.data.d) - self.dag.max_lag:]
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