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
    def __init__(self, y: Process, parents: Dict[str, Process] = None, nsample = 100):
        self.nsamples = nsample
        
        self.y = y
        self.parents = parents
        self.Y = None
        self.X = None
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
        # self.computeMarginalDensity()

        
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
            
            
    def computeJointDensity(self):
        if self.JointDensity is None:
            if self.parents is not None:
                self.JointDensity = self.estimate(np.column_stack([self.Y, self.X]))
            else:
                self.JointDensity = self.estimate(self.Y)
        return self.JointDensity
    
    def computePriorDensity(self):
        if self.PriorDensity is None: 
            self.PriorDensity = self.estimate(self.Y)
        return self.PriorDensity
        
        
    # FIXME: to fix with the new density
    # def computeMarginalDensity(self):
    #     if self.MarginalDensity is None:
    #         if self.parents is None:
    #             self.MarginalDensity = self.PriorDensity()
    #         else:
    #             # Compute conditional density P(Y | parents of Y)
    #             conditional_density = self.CondDensity()
                
    #             # Integrate over all possible values of parent variables to obtain marginal density of Y
    #             # Assuming self.parents contains all possible values of parent variables
    #             self.MarginalDensity = copy.deepcopy(conditional_density)
    #             # for p in self.parents:
    #             #     # FIXME: it is negative when I do the integral
    #             #     self.marginal_density *= np.trapz(self.marginal_density, x = self.X[:, list(self.parents.keys()).index(p)], axis = 0)
                        
    #             # FIXME: not sure about it
    #             for p in self.parents:
    #                 # Get the x values corresponding to the current parent variable
    #                 x_values = copy.deepcopy(self.X[:, list(self.parents.keys()).index(p)])
                    
    #                 # Sort the x values and the conditional density array based on x values
    #                 sorted_indices = np.argsort(x_values)
    #                 x_sorted = x_values[sorted_indices]
    #                 density_sorted = self.MarginalDensity[sorted_indices]
                    
    #                 marginal_density_partial = np.trapz(density_sorted, x=x_sorted)
                    
    #                 self.MarginalDensity *= marginal_density_partial

                
    #     return self.MarginalDensity
         
        
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
                self.ParentJointDensity = self.estimate(self.X)
        return self.ParentJointDensity
    
    
    
    # FIXME: by using self.nsample, this method might be useless 
    @staticmethod
    def fixLen(density, desired_length):
        # Define the interpolation function for density
        interp_func_1 = interp1d(np.arange(len(density)), density, kind='linear', fill_value="extrapolate")

        # Interpolate density to match the maximum length
        return interp_func_1(np.linspace(0, len(density) - 1, desired_length))
    

    def estimate(self, YZ):

        data = YZ
        XZ_mesh = np.meshgrid(*[np.linspace(min(YZ[:, i]), max(YZ[:, i]), self.nsamples) for i in range(YZ.shape[1])])
        XY_samples = np.column_stack([xz.ravel() for xz in XZ_mesh])


        # Create the grid search
        bandwidths = [0.1, 0.5, 1]
        Ks = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
        grid_search = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths, 'kernel': Ks})
        grid_search.fit(data)

        # Fit a kernel density model to the data
        kde = KernelDensity(bandwidth=grid_search.best_params_['bandwidth'], kernel=grid_search.best_params_['kernel'])
        kde.fit(data)

        # Compute the density
        log_density = kde.score_samples(XY_samples)
        density = np.exp(log_density)
        density = density.reshape(XZ_mesh[0].shape)
        return density
    # def estimate(self, target, Z = None):
    #     if Z is None:
    #         data = target
    #         YZ = np.linspace(min(target), max(target), self.nsamples)
    #     else:
    #         data = np.column_stack((target, Z))
    #         XZ_mesh = np.meshgrid(np.linspace(min(target), max(target), self.nsamples), *[np.linspace(min(Z[:, i]), max(Z[:, i]), self.nsamples) for i in range(Z.shape[1])])
    #         YZ = np.column_stack([xz.ravel() for xz in XZ_mesh])


    #     # Create the grid search
    #     bandwidths = [0.1, 0.5, 1]
    #     Ks = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
    #     grid_search = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths, 'kernel': Ks})
    #     grid_search.fit(data)

    #     # Fit a kernel density model to the data
    #     kde = KernelDensity(bandwidth=grid_search.best_params_['bandwidth'], kernel=grid_search.best_params_['kernel'])
    #     kde.fit(data)

    #     # Compute the density
    #     log_density = kde.score_samples(YZ)
    #     density = np.exp(log_density)
    #     if Z is not None:
    #         density = density.reshape(XZ_mesh[0].shape)
    #     return density
           
           
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
    
    
