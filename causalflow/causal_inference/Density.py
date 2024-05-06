from copy import deepcopy
from matplotlib import pyplot as plt
from causalflow.causal_inference.Process import Process
from typing import Dict
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from tigramite.causal_effects import CausalEffects
from causalflow.graph.DAG import DAG
from scipy.interpolate import interp1d
from scipy.integrate import simps


class Density():
    def __init__(self, y: Process, parents: Dict[str, Process] = None):
        self.y = y
        self.parents = parents
        self.Y = None
        self.X = None
        self.DO = {}
        self._preprocess()
        
        # init density variables
        self.joint_density = None
        self.parent_joint_density = None
        self.marginal_density = None
        self.cond_density = None
        self.prior_density = None

        
    @property
    def MaxLag(self):
        if self.parents is not None: 
            return max(p.lag for p in self.parents.values())
        return 0
        
        
    def _preprocess(self):
        # target
        self.Y = self.y.align(self.MaxLag)
        
        if self.parents is not None:
            # parents
            X = [p.align(self.MaxLag) for p in self.parents.values()]
            self.X = np.column_stack(X)
            
            
    def JointDensity(self):
        if self.joint_density is None: 
            self.joint_density = self.estimate(self.Y, self.X)
        return self.joint_density
    
    
    def PriorDensity(self):
        if self.prior_density is None: 
            self.prior_density = self.estimate(self.Y)
        return self.prior_density
        
        
    def MarginalDensity(self):
        if self.marginal_density is None:
            if self.parents is None:
                self.marginal_density = self.PriorDensity()
            else:
                # Compute conditional density P(Y | parents of Y)
                conditional_density = self.ConditionalDensity()
                
                # Integrate over all possible values of parent variables to obtain marginal density of Y
                # Assuming self.parents contains all possible values of parent variables
                self.marginal_density = conditional_density
                for p in self.parents:
                    # FIXME: it is negative when I do the integral
                    self.marginal_density *= np.trapz(self.marginal_density, x = self.X[:, list(self.parents.keys()).index(p)], axis = 0)
                        
        return self.marginal_density
         
        
    def ConditionalDensity(self):
        if self.cond_density is None:
            if self.parents is not None:
                self.cond_density = self.JointDensity() / self.ParentJointDensity()
            else:
                self.cond_density = self.PriorDensity()
        return self.cond_density
    
    
    def ComputeDensity(self):
        self.PriorDensity()
        self.ConditionalDensity()
        self.MarginalDensity()


    def ParentJointDensity(self):
        if self.parent_joint_density is None:
            if self.parents is not None: 
                self.parent_joint_density = self.estimate(self.X)
        return self.parent_joint_density
    
    
    @staticmethod
    def fixLen(density, desired_length):
        # Define the interpolation function for density
        interp_func_1 = interp1d(np.arange(len(density)), density, kind='linear', fill_value="extrapolate")

        # Interpolate density to match the maximum length
        return interp_func_1(np.linspace(0, len(density) - 1, desired_length))
    

    @staticmethod
    def estimate(target, Z = None):
        if Z is None:
            data = target
        else:
            data = np.column_stack((target, Z))
            
        # Create the grid search
        bandwidths = [0.1, 0.5, 1]
        Ks = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
        grid_search = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths, 'kernel': Ks})
        grid_search.fit(data)

        # Fit a kernel density model to the data
        kde = KernelDensity(bandwidth=grid_search.best_params_['bandwidth'], kernel=grid_search.best_params_['kernel'])
        kde.fit(data)

        # Compute the density
        log_density = kde.score_samples(data)
        density = np.exp(log_density)
        return density
           
           
    def expectation(self, cond_density):
        if np.sum(cond_density) == 0:
            # raise ValueError("Given value(s) out of distributions")
            return np.nan
        expectation_Y_given_X = np.sum(self.Y * cond_density) / np.sum(cond_density)
        return expectation_Y_given_X
        
    
    # def If(self, given_p: Dict[str, float]):
    #     if self.parents is None: 
    #         dens = self.MarginalDensity()
    #     else:
    #         # self.given_p = given_p
    #         indices_X = None
    #         for p in given_p.keys():
    #             # if p not in self.parents:
    #             #     marginal_density = self._get_marginal_density()
    #             #     return marginal_density, self._expectation(marginal_density)
    #             column_indices = np.where(np.isclose(self.X[:, list(self.parents.keys()).index(p)], given_p[p], atol=0.25))[0]
                
    #             if indices_X is None:
    #                 indices_X = set(column_indices)
    #             else:
    #                 indices_X = indices_X.intersection(column_indices)
            
    #         indices_X = np.array(sorted(indices_X)) 

    #         zero_array = np.zeros_like(self.ParentMarginalDensity())
    #         eval_cond_density = deepcopy(self.ConditionalDensity())
    #         eval_cond_density[~np.isin(np.arange(len(self.ParentMarginalDensity())), indices_X)] = zero_array[~np.isin(np.arange(len(self.ParentMarginalDensity())), indices_X)]
    #         dens = eval_cond_density.reshape(-1, 1)
            
    #     return dens, self._expectation(dens)
    
    
