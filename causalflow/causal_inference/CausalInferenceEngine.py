import numpy as np
from causalflow.causal_inference.DynamicBayesianNetwork import DynamicBayesianNetwork
from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
import copy



class CausalInferenceEngine():
    def __init__(self, dag: DAG, obs_data: Data, nsample = 100):
        """
        CausalEngine contructer
        """
        self.nsample = nsample
        self.outcome_var = None
        self.DAGs = {('obs', 0): dag}
        self.Ds = {('obs', 0): obs_data}
        self.DBNs = {('obs', 0): DynamicBayesianNetwork(dag, obs_data, self.nsample)}
        
        
    @property        
    def nextObs(self):
        """
        Returns next observation ID

        Returns:
            int: next observation ID
        """
        return max((key for key in self.DAGs.keys() if key[0] == 'obs'), key=lambda x: x[1]) + 1
    
    
    @property
    def nextInt(self):
        """
        Returns next intervention ID

        Returns:
            int: next intervention ID
        """
        return max((key for key in self.DAGs.keys() if key[0] == 'int'), key=lambda x: x[1]) + 1
    
    
    @property
    def dag(self):
        """
        Returns default observational DAG

        Returns:
            DAG: default observational DAG
        """
        return self.DAGs[('obs', 0)]
    
    
    @property
    def dbn(self):
        """
        Returns the Dynamic Bayesian Network associated to the default observational DAG

        Returns:
            DynamicBayesianNetwork: Dynamic Bayesian Network associated to the default observational DAG
        """
        return self.DBNs[('obs', 0)]
    
    
    @property
    def d(self):
        """
        Returns the Data associated to the default observational DAG

        Returns:
            Data: Data associated to the default observational DAG
        """
        return self.Ds[('obs', 0)]
        
        
    def addObsData(self, data: Data):
        """
        Adds new observational dataset

        Args:
            data (Data): new observational dataset
        """
        self.DAGs = {('obs', self.nextObs): self.dag}
        self.Ds = {('obs', self.nextObs): data}
        self.DBNs = {('obs', self.nextObs): DynamicBayesianNetwork(self.dag, data, self.nsample)}
        
        
    def addIntData(self, target: str, data: Data):
        """
        Adds new interventional dataset

        Args:
            target (str): Intervention treatment variable
            data (Data): Interventional data
        """
        dag = copy.deepcopy(self.dag)
        for s in self.dag.g[target].sources:
            dag.del_source(target, s[0], s[1])
            
        k = 'int_' + str(target)
        self.DAGs = {(k, self.nextInt): dag}
        self.Ds = {(k, self.nextInt): data}
        self.DBNs = {(k, self.nextInt): DynamicBayesianNetwork(self.dag, data, self.nsample)}
        
        
    def whatHappensTo(self, outcome: str):
        """
        initialises the query taking in input the outcome variable

        Args:
            outcome (str): outcome variable

        Returns:
            CausalInferenceEngine: self
        """
        self.outcome_var = outcome
        return self
    
    
    def If(self, treatment: str, value):
        """
        finalises the query, which has been initialised by whatHappenTo, taking in input the treatment variable and its value

        Args:
            treatment (str): treatment variable
            value (float): treatment value
        """
                
        # TODO: check if data for this intervention already exists
        if ('int_' + str(treatment)) in self.Ds:
            # TODO: search for the most similar intervention. if not present, go to else
            # TODO: apply the transportability formula to estimate the effect of the intervention on the desired population
            pass
        else:
            p_y_do_X_x, E_p_y_do_X_x = self.dbn.evalDoDensity(treatment, self.outcome_var, value)
            
            # TODO: once found the estimated cause-effect, use the transportability formula to estimate the effect of the intervention on the desired population 
            return p_y_do_X_x, E_p_y_do_X_x
        
        
    def transport(self, sourceP: tuple, targetP: tuple, treatment: str, outcome: str):
        """
        Computes the target population's p_y_do(x) from the source population by using the transportability formula [1].
        
        [1] Bareinboim, Elias, and Judea Pearl. "Causal inference and the data-fusion problem." 
            Proceedings of the National Academy of Sciences 113.27 (2016): 7345-7352.

        Args:
            sourceP (tuple): source population ID
            targetP (tuple): target population ID
            treatment (str): treatment variable
            outcome (str): outcome variable

        Returns:
            nd.array: Target population's p_y_do(x)
        """
        adjset = self.DBNs[sourceP][outcome].DO[treatment]['adj']
        
        # Source population's p(output|do(treatment), adjustment)
        pS_y_given_xadj = self.DBNs[sourceP][outcome].DO[treatment]['p_y_given_xadj']
        
        # Compute the adjustment density for the target population
        pT_adj = np.ones((self.nsample, 1)).squeeze()
            
        for node in adjset: pT_adj = pT_adj * self.DBNs[targetP][self.data.features[node[0]]].CondDensity
        
        # Compute the p(outcome|do(treatment)) density
        if len(pS_y_given_xadj.shape) > 2: 
            # Sum over the adjustment set
            p_y_do_x = np.sum(pS_y_given_xadj * pT_adj, axis=tuple(range(2, len(pS_y_given_xadj.shape)))) #* np.sum(p_adj, axis=tuple(range(0, len(p_adj.shape))))
        else:
            p_y_do_x = pS_y_given_xadj
        
        return p_y_do_x