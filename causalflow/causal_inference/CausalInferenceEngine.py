import numpy as np
from causalflow.basics.constants import *
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
            p_y_do_x = self.transport(('int_' + str(treatment)), )
        else:
            # TODO: which observational data?
            # TODO: I need to specify the source and target populations
            p_y_do_x = self.transport()
            # TODO: I have to evaluate the p_y_do_x with a specific interventional value
            # p_y_do_X_x, E_p_y_do_X_x = self.dbn.evalDoDensity(treatment, self.outcome_var, value)
            
            # return p_y_do_X_x, E_p_y_do_X_x
        
        
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
        # adjset = self.DBNs[sourceP][outcome].DO[treatment][ADJ] # FIXME: I think I need to compute again the adjset for the target population
        adjset = self.DBNs[targetP].get_adjset(treatment, outcome) # TODO: to test
        
        # Source population's p(output|do(treatment), adjustment)
        pS_y_do_x_adj = self.DBNs[sourceP][outcome].DO[treatment][P_Y_GIVEN_DOX_ADJ]
        
        # Compute the adjustment density for the target population
        pT_adj = np.ones((self.nsample, 1)).squeeze()
            
        for node in adjset: pT_adj = pT_adj * self.DBNs[targetP][self.data.features[node[0]]].CondDensity
        
        # Compute the p(outcome|do(treatment)) density
        if len(pS_y_do_x_adj.shape) > 2: 
            # Sum over the adjustment set
            p_y_do_x = np.sum(pS_y_do_x_adj * pT_adj, axis=tuple(range(2, len(pS_y_do_x_adj.shape)))) #* np.sum(p_adj, axis=tuple(range(0, len(p_adj.shape))))
        else:
            p_y_do_x = pS_y_do_x_adj
        
        return p_y_do_x
    
    
    
        # def evalDoDensity(self, treatment: str, outcome: str, value):
    #     """
    #     Evaluates the p(outcome|treatment = t)

    #     Args:
    #         treatment (str): treatment variable
    #         outcome (str): outcome variable
    #         value (float): treatment value

    #     Returns:
    #         tuple: p(outcome|treatment = t), E[p(outcome|treatment = t)]
    #     """
    #     indices_X = None
    #     column_indices = np.where(np.isclose(self.dbn[outcome].X[:, treatment], value, atol=0.25))[0]
    #     if indices_X is None:
    #         indices_X = set(column_indices)
    #     else:
    #         # The intersection is needed to take the common indices 
    #         indices_X = indices_X.intersection(column_indices)
        
    #     indices_X = np.array(sorted(indices_X))
        
    #     X_dens = np.zeros_like(self.dbn[treatment].MarginalDensity)
    #     zero_array = np.zeros_like(X_dens)
    #     p_y_do_X_x = copy.deepcopy(self.dbn[outcome].DO[treatment])
    #     p_y_do_X_x[~np.isin(np.arange(len(X_dens)), indices_X)] = zero_array[~np.isin(np.arange(len(X_dens)), indices_X)]
    #     p_y_do_X_x = p_y_do_X_x.reshape(-1, 1)
        
    #     # TODO: once found the estimated cause-effect, use the transportability formula to estimate the effect of the intervention on the desired population 
    #     return p_y_do_X_x, self.dbn[outcome].expectation(p_y_do_X_x)