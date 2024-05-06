from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
from causalflow.causal_inference.Density import Density
from causalflow.causal_inference.Process import Process
import copy



class CausalEngine():
    def __init__(self, dag: DAG, obs_data: Data):
        """
        CausalEngine contructer
        """
        # observation variables
        self.DAGs = {('obs', 0): dag}
        self.Ds = {('obs', 0): obs_data}
        self.DBNs = {('obs', 0): self._DAG2DBN(dag, obs_data)}
        
        
    @property        
    def nextObs(self):
        return max((key for key in self.DAGs.keys() if key[0] == 'obs'), key=lambda x: x[1]) + 1
    
    
    @property
    def nextInt(self):
        return max((key for key in self.DAGs.keys() if key[0] == 'int'), key=lambda x: x[1]) + 1
    
    
    @property
    def dag(self):
        return self.DAGs[('obs', 0)]
    
    
    @property
    def dbn(self):
        return self.DBNs[('obs', 0)]
        
        
    def addObsData(self, data):
        self.DAGs = {('obs', self.nextObs): self.dag}
        self.Ds = {('obs', self.nextObs): data}
        self.DBNs = {('obs', self.nextObs): self._DAG2DBN(self.dag, data)}
        
        
    def addIntData(self, target, data):
        dag = copy.deepcopy(self.dag)
        for s in self.dag.g[target].sources:
            dag.del_source(target, s[0], s[1])
            
        k = 'int_' + str(target)
        self.DAGs = {(k, self.nextInt): dag}
        self.Ds = {(k, self.nextInt): data}
        
        
    @staticmethod
    def _DAG2DBN(dag: DAG, d: Data):
        """
        converts the DAG to a DBN
        """
        dbn = {node: None for node in dag}
        for node in dbn:
            Y = Process(d[node].to_numpy(), node, 0)
            parents = CausalEngine._extract_parents(d, dag, node)
            dbn[node] = Density(Y, parents)            
            dbn[node].ComputeDensity()
        return dbn
            
            
    @staticmethod
    def _extract_parents(d, dag, node):
        parents = {s[0]: Process(d[s[0]].to_numpy(), s[0], s[1]) for s in dag.g[node].sources}
        if not parents: return None
        return parents