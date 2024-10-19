from copy import deepcopy
from matplotlib import pyplot as plt
from causalflow.causal_inference.Process import Process
from typing import Dict
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

class Density():
    def __init__(self, y: Process, parents: Dict[str, Process] = None):
        self.y = y
        self.parents = parents
        self.Y = None
        self.X = None
        self.joint_density = None
        self.parent_marginal_density = None
        self.marginal_density = None
        self.dens = None
        self._preprocess()

        
    @property
    def MaxLag(self):
        if self.parents is not None: 
            return max(p.lag for p in self.parents.values())
        return 0
        
        
    def JointDensity(self):
        if self.joint_density is None: 
            self.joint_density = self.estimate(self.Y, self.X)
        return self.joint_density
        
        
    def MarginalDensity(self):
        if self.marginal_density is None: 
            self.marginal_density = self.estimate(self.Y)
        return self.marginal_density
         
        
    def ConditionalDensity(self):
        if self.dens is None:
            if self.parents is not None:
                self.dens = self.JointDensity() / self.ParentMarginalDensity()
            else:
                self.dens = self.MarginalDensity()

        return self.dens
    
    
    def ComputeDensity(self):
        if self.parents is not None:
            self.ConditionalDensity()
        else:
            self.MarginalDensity()


    def ParentMarginalDensity(self):
        if self.parent_marginal_density is None:
            if self.parents is not None: 
                self.parent_marginal_density = self.estimate(self.X)
        return self.parent_marginal_density


    def _preprocess(self):
        # target
        self.Y = self.y.data[self.MaxLag - self.y.lag : self.y.T - self.y.lag]
        
        if self.parents is not None:
            # parents
            X = list()
            for p in self.parents.values():
                X_p = p.data[self.MaxLag - p.lag : p.T - p.lag]
                X.append(X_p)
            self.X = np.column_stack(X)


    @staticmethod
    def estimate(target, Z = None):
        if Z is None:
            data = target
        else:
            data = np.column_stack((target, Z))
            
        # Create the grid search
        bandwidths = np.logspace(-5, 5, 100)
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
           
           
    @staticmethod        
    def _expectation(self, cond_density):
        if np.sum(cond_density) == 0:
            # raise ValueError("Given value(s) out of distributions")
            return np.nan
        expectation_Y_given_X = np.sum(self.Y * cond_density) / np.sum(cond_density)
        return expectation_Y_given_X
        
    
    def If(self, given_p: Dict[str, float]):
        if self.parents is None: 
            dens = self.MarginalDensity()
        else:
            # self.given_p = given_p
            indices_X = None
            for p in given_p.keys():
                # if p not in self.parents:
                #     marginal_density = self._get_marginal_density()
                #     return marginal_density, self._expectation(marginal_density)
                column_indices = np.where(np.isclose(self.X[:, list(self.parents.keys()).index(p)], given_p[p], atol=0.25))[0]
                
                if indices_X is None:
                    indices_X = set(column_indices)
                else:
                    indices_X = indices_X.intersection(column_indices)
            
            indices_X = np.array(sorted(indices_X)) 

            zero_array = np.zeros_like(self.ParentMarginalDensity())
            eval_cond_density = deepcopy(self.ConditionalDensity())
            eval_cond_density[~np.isin(np.arange(len(self.ParentMarginalDensity())), indices_X)] = zero_array[~np.isin(np.arange(len(self.ParentMarginalDensity())), indices_X)]
            dens = eval_cond_density.reshape(-1, 1)
            
        return dens, self._expectation(dens)
    
        
    # def plot_conditional_density(self, cond_density):
    #     sorted_indices = np.argsort(np.squeeze(self.Y))
    #     sorted_Y = self.Y[sorted_indices].reshape(-1, 1)
    #     sorted_cd = cond_density[sorted_indices].reshape(-1, 1)
    #     parents_string = '(' 
    #     parents_string += ', '.join(self.parents[p].varname + ('=' + str(self.given_p[p]) if (self.given_p is not None and p in self.given_p) else '') for p in self.parents.keys()) 
    #     parents_string += ')'
        
    #     plt.plot(sorted_Y, sorted_cd)
    #     plt.xlabel(self.y.pvarname)
    #     plt.ylabel('Conditional Density')
    #     plt.title('$f_{' + self.y.varname + '|' + parents_string + '}$')
    #     plt.show()