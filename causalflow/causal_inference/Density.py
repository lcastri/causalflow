from copy import deepcopy
import copy
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
        self.DO = {}
        self._preprocess()
        
        # init density variables
        self.JointDensity = None
        self.ParentJointDensity = None
        self.MarginalDensity = None
        self.CondDensity = None
        self.PriorDensity = None
        
        self.computePriorDensity()
        self.computeJointDensity()
        self.computeParentJointDensity()
        self.computeConditionalDensity()
        self.computeMarginalDensity()

        
    @property
    def MaxLag(self):
        if self.parents is not None: 
            return max(p.lag for p in self.parents.values())
        return 0
        
        
    def _preprocess(self):
        # target
        self.y.align(self.MaxLag)
        
        if self.parents is not None:
            # parents
            for p in self.parents.values():
                p.align(self.MaxLag)
            
            
    def computeJointDensity(self):
        if self.JointDensity is None:
            if self.parents is not None:
                yz = [p for p in self.parents.values()]
                yz.insert(0, self.y)
                self.JointDensity = self.estimate(yz)
            else:
                self.JointDensity = self.estimate(self.y)
        return self.JointDensity
    
    
    def computePriorDensity(self):
        if self.PriorDensity is None: 
            self.PriorDensity = self.estimate(self.y)
        return self.PriorDensity
        
        
    def computeMarginalDensity(self):
        if self.MarginalDensity is None:
            if self.parents is None:
                self.MarginalDensity = self.PriorDensity
            else:                
                # Sum over parents axis
                self.MarginalDensity = copy.deepcopy(self.JointDensity)
                self.MarginalDensity = np.sum(self.MarginalDensity, axis=tuple(range(1, len(self.JointDensity.shape))))  

        return self.MarginalDensity
         
        
    def computeConditionalDensity(self):
        if self.CondDensity is None:
            if self.parents is not None:
                self.CondDensity = self.JointDensity / self.ParentJointDensity
            else:
                self.CondDensity = self.PriorDensity
        return self.CondDensity


    def computeParentJointDensity(self):
        if self.ParentJointDensity is None:
            if self.parents is not None: 
                self.ParentJointDensity = self.estimate([p for p in self.parents.values()])
        return self.ParentJointDensity
    
    
    @staticmethod
    def estimate(YZ):
        if not isinstance(YZ, list): YZ = [YZ]
        
        YZ_data = np.column_stack([yz.aligndata for yz in YZ])
        YZ_mesh = np.meshgrid(*[yz.samples for yz in YZ])
        YZ_samples = np.column_stack([yz.ravel() for yz in YZ_mesh])
        
        # Create the grid search
        bandwidths = [0.1, 0.5, 1]
        Ks = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
        grid_search = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths, 'kernel': Ks})
        grid_search.fit(YZ_data)

        # Fit a kernel density model to the data
        kde = KernelDensity(bandwidth=grid_search.best_params_['bandwidth'], kernel=grid_search.best_params_['kernel'])
        kde.fit(YZ_data)

        # Compute the density
        log_density = kde.score_samples(YZ_samples)
        density = np.exp(log_density)
        density = density.reshape(YZ_mesh[0].shape)
        return density
           
           
    def expectation(self, cond_density):
        if np.sum(cond_density) == 0:
            # raise ValueError("Given value(s) out of distributions")
            return np.nan
        expectation_Y_given_X = np.sum(self.y.samples * cond_density) / np.sum(cond_density)
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
    
    
