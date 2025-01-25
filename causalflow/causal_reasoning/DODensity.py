import numpy as np
import warnings
from causalflow.CPrinter import CP
from causalflow.causal_reasoning.Process import Process
from causalflow.basics.constants import *
from causalflow.causal_reasoning.Utils import *
from typing import Dict
import causalflow.causal_reasoning.Utils as DensityUtils
from scipy.integrate import dblquad
from itertools import product
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
    
class DOType(Enum):
    pY = 'pY'
    pY_given_X = 'pY|X'
    pY_given_X_Adj = 'sum(pY|X,Adj*pAdj)'
    pY_given_X_Cond_Adj = 'sum(pY|X,Cond,Adj*pAdj)'


class DODensity():       
    def __init__(self, 
                 outcome: Process, 
                 treatment: Process,
                 conditions: Dict[str, Process] = None,
                 adjustments: Dict[str, Process] = None,
                 doType: DOType = DOType.pY_given_X_Adj,
                 max_components = 50,
                 pY = None,
                 pY_X = None):
        """
        Class constructor.
        """
        self.outcome = outcome
        self.treatment = treatment
        self.conditions = conditions
        self.adjustments = adjustments
        self.max_components = max_components
        self.doType = doType
        
        self.pAdj = None
        self.pJoint = None
    
        if self.doType in [DOType.pY_given_X_Adj, DOType.pY_given_X_Cond_Adj]:
            self._preprocess()
            
            self.pY = self.compute_pY() if pY is None else pY
            self.pY_X = self.compute_pY_X() if pY_X is None else pY_X
            self.pAdj = self.compute_pAdj()
            if self.doType is DOType.pY_given_X_Adj:
                self.pJoint = self.compute_pY_X_Adj()
            elif self.doType is DOType.pY_given_X_Cond_Adj:
                self.pJoint = self.compute_pY_X_Cond_Adj()
    
            
    def _preprocess(self):
        ALL = {}
        ALL[self.outcome.pvarname] = self.outcome
        ALL[self.treatment.pvarname] = self.treatment
        for pname, cond in self.conditions.items():
            ALL[pname] = cond
        for pname, adj in self.adjustments.items():
            ALL[pname] = adj
        maxLag = DensityUtils.get_max_lag(ALL)
        
        # Outcome
        self.outcome.align(maxLag)
        
        # Treatment
        self.treatment.align(maxLag)
        
        # Conditions
        for p in self.conditions.values():
            p.align(maxLag)
            
        # Adjustments
        for p in self.adjustments.values():
            p.align(maxLag)
            
            
    def compute_pY(self):
        """
        Compute the prior density p(y) using GMM.

        Returns:
            dict: GMM parameters for the prior density.
        """
        CP.info("    - Prior density p(Y)", noConsole=True)
        return DensityUtils.fit_gmm(self.max_components, 'p(Y)', self.outcome.aligndata)
    
    
    def compute_pY_X(self):
        ALL = {}
        ALL[self.outcome.pvarname] = self.outcome
        ALL[self.treatment.pvarname] = self.treatment
        CP.info("    - Joint density p(Y,X)", noConsole=True)
        all_data = np.column_stack([p.aligndata for p in ALL.values()])
        return DensityUtils.fit_gmm(self.max_components, 'p(Y,X)', all_data)
                    
                    
    def compute_pAdj(self):
        CP.info("    - Adjustment density p(Adj)", noConsole=True)
        adj_data = np.column_stack([p.aligndata for p in self.adjustments.values()])
        return DensityUtils.fit_gmm(self.max_components, 'p(Adj)', adj_data)


    def compute_pY_X_Adj(self):
        ALL = {}
        ALL[self.outcome.pvarname] = self.outcome
        ALL[self.treatment.pvarname] = self.treatment
        for pname, adj in self.adjustments.items():
            ALL[pname] = adj
        CP.info("    - Joint density p(Y,X,Adj)", noConsole=True)
        all_data = np.column_stack([p.aligndata for p in ALL.values()])
        return DensityUtils.fit_gmm(self.max_components, 'p(Y,X,Adj)', all_data)
    
    def compute_pY_X_Cond_Adj(self):
        ALL = {}
        ALL[self.outcome.pvarname] = self.outcome
        ALL[self.treatment.pvarname] = self.treatment
        for pname, c in self.conditions.items():
            ALL[pname] = c
        for pname, adj in self.adjustments.items():
            ALL[pname] = adj
        CP.info("    - Joint density p(Y,X,Cond,Adj)", noConsole=True)
        all_data = np.column_stack([p.aligndata for p in ALL.values()])
        return DensityUtils.fit_gmm(self.max_components, 'p(Y,X,Cond,Adj)', all_data)


    def compute_p_y_do_x(self, x_value, adj_values):
        """
        Compute p(Y | do(X=x)) for both discrete and continuous adjustment sets.

        Args:
            x_value (float): Value of X=x.
            adj_values (list): List of discrete adjustment values (for discrete Adj).

        Returns:
            float: Marginal interventional density p(Y | do(X=x)).
        """
        total_density = 0

        # Discrete Adjustment Set
        for adj_value in adj_values:
            adj_value = np.array(adj_value)
            p_adj = DensityUtils.get_density(self.pAdj, adj_value)
            parent_values = np.concatenate(([x_value], adj_value))
            cond_params = DensityUtils.compute_conditional(self.pJoint, parent_values)

            for k in range(len(cond_params["weights"])):
                weight = cond_params["weights"][k]
                total_density += weight * p_adj

        return total_density
    
    
    def predict(self, x: float = None, conditions: Dict[str, float] = None):
        """
        Predict the conditional density p(y | parents) and the expected value of y.

        Args:
            x (float, optional): A dictionary of treatment variable values.

        Returns:
            float: Expected value of y.
        """
        if self.doType is DOType.pY_given_X_Cond_Adj and conditions is None:
            raise ValueError("conditions must be provided for DOType.pY_given_X_Cond_Adj")
        
        if self.doType is DOType.pY:
            conditional_params = self.pY
            
        elif self.doType is DOType.pY_given_X:
            parent_values = np.array(x).reshape(-1, 1)
            conditional_params = DensityUtils.compute_conditional(self.pY_X, parent_values)
            
        elif self.doType in [DOType.pY_given_X_Adj, DOType.pY_given_X_Cond_Adj]:
            total_density = 0
            
            adj_combo = list(product([p.unique_values for p in self.adjustments.values()]))

            # Discrete Adjustment Set
            for adj_value in adj_combo:
                adj_value = np.array(adj_value)
                p_adj = DensityUtils.get_density(self.pAdj, adj_value)
                if self.doType is DOType.pY_given_X_Adj:
                    parent_values = np.concatenate(([x], adj_value))
                elif self.doType is DOType.pY_given_X_Cond_Adj:
                    parent_values = np.concatenate(([x], [conditions[pname] for pname in self.conditions], adj_value))
                cond_params = DensityUtils.compute_conditional(self.pJoint, parent_values)

                for k in range(len(cond_params["weights"])):
                    weight = cond_params["weights"][k]
                    total_density += weight * p_adj
                                
        # dens = Density.get_density(self.y.aligndata, conditional_params)
        # dens = dens / np.sum(dens)

        # Find the most likely value (mode)
        expected_value = DensityUtils.expectation_from_params(conditional_params['means'], conditional_params['weights'])

        # return dens, expected_value
        return expected_value