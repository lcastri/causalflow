from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
import numpy as np
import pandas as pd
from causalflow.causal_inference.Density import Density
from causalflow.causal_inference.Process import Process
import copy
from tigramite.causal_effects import CausalEffects



class CausalEngine():
    def __init__(self, dag: DAG, obs_data: Data):
        """
        CausalEngine contructer
        """
        # observation variables
        self.DAGs = {('obs', 0): dag}
        self.Ds = {('obs', 0): obs_data}
        
        
    @property        
    def nextObs(self):
        return max((key for key in self.DAGs.keys() if key[0] == 'obs'), key=lambda x: x[1]) + 1
    
    
    @property
    def nextInt(self):
        return max((key for key in self.DAGs.keys() if key[0] == 'int'), key=lambda x: x[1]) + 1
    
    
    @property
    def DAG(self):
        return self.DAGs[('obs', 0)]
        
        
    def addObsData(self, data):
        self.DAGs = {('obs', self.nextObs): self.DAG}
        self.Ds = {('obs', self.nextObs): data}
        
        
    def addIntData(self, target, data):
        dag = copy.deepcopy(self.DAG)
        for s in self.DAG.g[target].sources:
            dag.del_source(target, s[0], s[1])
            
        k = 'int_' + str(target)
        self.DAGs = {(k, self.nextInt): dag}
        self.Ds = {(k, self.nextInt): data}