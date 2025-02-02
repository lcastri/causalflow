from causalflow.CPrinter import CP
from causalflow.causal_reasoning.Utils import *
from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
from causalflow.causal_reasoning.Density import Density
from causalflow.causal_reasoning.Process import Process
from causalflow.basics.constants import *
from typing import Dict
from pgmpy.factors.continuous import ContinuousFactor

class DynamicBayesianNetwork():
    def __init__(self, 
                 dag: DAG, 
                 data: Data, 
                 data_type: Dict[str, DataType], 
                 node_type: Dict[str, NodeType],
                 recycle = None,
                 max_components = 50):
        """
        Class constructor.

        Args:
            dag (DAG): DAG from which deriving the DBN.
            data (Data): Data associated with the DAG.
            data_type (dict[str:DataType]): data type for each node (continuous|discrete). E.g., {"X_2": DataType.Continuous}.
            node_type (dict[str:NodeType]): node type for each node (system|context). E.g., {"X_2": NodeType.Context}.
        """
        self.dag = dag
        self.data = data
        self.data_type = data_type
        self.node_type = node_type
        self.max_components = max_components
        
        nodes = [(f, -abs(l)) for f in self.dag.features for l in range(self.dag.max_lag + 1)]
        self.dbn_data = {node: None for node in nodes}
        self.dbn = DAG.get_DBN(self.dag.get_Adj(), self.dag.max_lag)
            
        self.compute_density(recycle)

           
            

        

        
        
    def _get_Y_X(self, data, node, dag):
        f, lag = node
        Y = Process(data[f].to_numpy(), f, abs(lag), self.data_type[f], self.node_type[f])
        X = {}
        for s in dag.g[f].sources:
            if s[1] + abs(lag) > dag.max_lag: continue
            if s[0] == 'WP': continue
            X[s[0]] = Process(data[s[0]].to_numpy(), s[0], s[1] + abs(lag), self.data_type[s[0]], self.node_type[s[0]])
        # X = {s[0]: Process(data[s[0]].to_numpy(), s[0], s[1], self.data_type[s[0]], self.node_type[s[0]])
        #     for s in dag.g[node].sources}
        return Y, X
    
    
    def gmm_factor(self, variables, gmm_params):
        """
        Create a pgmpy ContinuousFactor using stored GMM parameters.
        
        Args:
            variables (list): List of variable names.
            gmm_params (dict): Contains 'means', 'covariances', and 'weights'.
        
        Returns:
            ContinuousFactor
        """
        means, covariances, weights = gmm_params["means"], gmm_params["covariances"], gmm_params["weights"]
        
        def factor_pdf(*args):
            x = np.array(args).reshape(1, -1)  # Reshape input for compatibility
            return self.gmm_pdf_manual(x, means, covariances, weights)
        
        return ContinuousFactor(variables, factor_pdf)
    
    
    def gmm_pdf_manual(self, x, means, covariances, weights):
        """
        Compute the probability density for a given point `x` using a pre-trained GMM.
        
        Args:
            x (numpy array): The input point.
            means (numpy array): The means of the GMM components.
            covariances (numpy array): The covariance matrices of the components.
            weights (numpy array): The mixture weights.
        
        Returns:
            float: The probability density at `x`.
        """
        num_components = len(weights)
        prob = 0.0

        for i in range(num_components):
            prob += weights[i] * multivariate_normal.pdf(x, mean=means[i], cov=covariances[i])

        return prob
            
       
    def compute_density(self, recycle = None):
        for node in self.dbn_data.keys():
            if node[0] == 'WP': continue
            # Y and Y's parents retrieved from data                                             
            Y, X = self._get_Y_X(self.data.d, node, self.dag)
            parents_str = []
            for x in X.keys():
                parents_str.append(f"{X[x].varname}_t{-abs(X[x].lag) if X[x].lag != 0 else ''}")
            parents_str = f" {', '.join(parents_str)}" if len(parents_str) > 0 else ''
            CP.info(f"\n    ### Variable: {node[0]}_t{-abs(node[1]) if node[1] != 0 else ''}{f' -- Parent(s): {parents_str}' if X else ''}")
            self.dbn_data[node] = Density(Y, X if X else None, max_components=self.max_components)
            
            
            
            
            # # Here, you create the GMM factor (i.e., the Conditional Probability Distribution)
            # gmm_params = self.dbn_data[node].pJoint
            # factor = self.gmm_factor([node], gmm_params)  # Create a factor using the GMM parameters

            # # Add the factor as a CPD to the Dynamic Bayesian Network (DBN)
            # self.dbn.add_cpds(factor)
