import copy
import warnings
import os

from joblib import Parallel, delayed
from multiprocessing import Pool

from tqdm import tqdm
from causalflow.CPrinter import CP
from causalflow.causal_reasoning.Process import Process
from typing import Dict
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from causalflow.basics.constants import *

# Define the compute_density function as a top-level function
def compute_density(samples, kde):
    """
    Compute density for a subset of samples.

    Args:
        samples (ndarray): The input samples for which to compute density.
        kde (KernelDensity): Fitted KernelDensity model.

    Returns:
        ndarray: Density for the input samples.
    """
    samples = samples.copy()  # Ensure writable samples
    log_density = kde.score_samples(samples)
    return np.exp(log_density)
# Define a helper function for imap that takes a chunk and kde
def process_chunk(chunk_kde):
    chunk, kde = chunk_kde
    return compute_density(chunk, kde)
 
class Density():
    def __init__(self, y: Process, parents: Dict[str, Process] = None, atol = 0.25):
        """
        Class constructor.

        Args:
            y (Process): target process.
            parents (Dict[str, Process], optional): Target's parents. Defaults to None.
        """
        self.y = y
        self.parents = parents
        self.atol = atol
        self.DO = {}
        if self.parents is not None:
            self.DO = {treatment: {ADJ: None, 
                                   P_Y_GIVEN_DOX_ADJ: None, 
                                   P_Y_GIVEN_DOX: None} for treatment in self.parents.keys()}
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
        """
        Return max time lag between target and all its parents.

        Returns:
            int: max time lag.
        """
        if self.parents is not None: 
            return max(p.lag for p in self.parents.values())
        return 0
        
        
    def _preprocess(self):
        """Preprocess the data to have all the same length by using the maxlag."""
        # target
        self.y.align(self.MaxLag)
        
        if self.parents is not None:
            # parents
            for p in self.parents.values():
                p.align(self.MaxLag)
            
            
    def computeJointDensity(self):
        """
        Compute the joint density p(y, parents).

        Returns:
            ndarray: joint density p(y, parents).
        """
        # CP.debug("- Joint density")
        if self.JointDensity is None:
            if self.parents is not None:
                yz = [p for p in self.parents.values()]
                yz.insert(0, self.y)
                self.JointDensity = Density.estimate('Joint', yz)
            else:
                self.JointDensity = Density.estimate('Joint', self.y)
        self.JointDensity = Density.normalise(self.JointDensity)
        return self.JointDensity
    
    
    def computePriorDensity(self):
        """
        Compute the prior density p(y).

        Returns:
            ndarray: prior density p(y).
        """
        # CP.debug("- Prior density")
        if self.PriorDensity is None: 
            self.PriorDensity = Density.estimate('Prior', self.y)
        self.PriorDensity = Density.normalise(self.PriorDensity)
        return self.PriorDensity
        
        
    def computeMarginalDensity(self):
        """
        Compute the marginal density p(y) = \sum_parents p(y, parents).

        Returns:
            ndarray: marginal density p(y) = \sum_parents p(y, parents).
        """
        CP.debug("- Marginal density")
        if self.MarginalDensity is None:
            if self.parents is None:
                self.MarginalDensity = self.PriorDensity
            else:                
                # Sum over parents axis
                self.MarginalDensity = copy.deepcopy(self.JointDensity)
                self.MarginalDensity = np.sum(self.MarginalDensity, axis=tuple(range(1, len(self.JointDensity.shape))))  
        self.MarginalDensity = Density.normalise(self.MarginalDensity)
        return self.MarginalDensity
         
        
    def computeConditionalDensity(self):
        """
        Compute the conditional density p(y|parents) = p(y, parents) / p(parents).

        Returns:
            ndarray: conditional density p(y|parents) = p(y, parents) / p(parents).
        """
        CP.debug("- Conditional density")
        if self.CondDensity is None:
            if self.parents is not None:
                self.CondDensity = self.JointDensity / self.ParentJointDensity + np.finfo(float).eps
            else:
                self.CondDensity = self.PriorDensity
        self.CondDensity = Density.normalise(self.CondDensity)
        return self.CondDensity


    def computeParentJointDensity(self):
        """
        Compute the parents's joint density p(parents).

        Returns:
            ndarray: parents's joint density p(parents).
        """
        # CP.debug("- Parent density")
        if self.ParentJointDensity is None:
            if self.parents is not None: 
                self.ParentJointDensity = Density.estimate('Parent', [p for p in self.parents.values()])
                self.ParentJointDensity = Density.normalise(self.ParentJointDensity)
        return self.ParentJointDensity
    
    
    @staticmethod
    def compute_density_parallel(caller, kde, YZ_samples):
        """
        Compute the density for a set of samples in parallel.

        Args:
            kde (KernelDensity): Fitted KernelDensity model.
            YZ_samples (ndarray): Samples for which to compute the density.

        Returns:
            ndarray: Computed density for the input samples.
        """
        YZ_samples = np.array(YZ_samples, copy=True)

        # Determine how to split YZ_samples into chunks
        num_samples = len(YZ_samples)

        # Use all available cores
        n_jobs = -1  # Use all available cores
        if num_samples == 0:
            return np.array([])  # Return empty array if there are no samples

        # Get the number of CPU cores available
        available_cores = os.cpu_count()
        n_jobs = min(available_cores, num_samples) if n_jobs == -1 else n_jobs

        chunk_size = max(1, num_samples // n_jobs)  # Ensure at least one sample per chunk
        # chunks = [YZ_samples[i:i + chunk_size] for i in range(0, num_samples, chunk_size)]
        chunks = [YZ_samples[i:i + chunk_size].copy() for i in range(0, num_samples, chunk_size)]

        # Parallel computation of densities for each chunk
        # densities = Parallel(n_jobs=n_jobs)(delayed(compute_density)(chunk) for chunk in chunks)
        # Combine chunks and kde into a tuple for processing
        chunk_kde_pairs = [(chunk, kde) for chunk in chunks]

        # Parallel computation of densities for each chunk using Pool and tqdm
        with Pool(n_jobs) as pool:
            densities = list(tqdm(pool.imap(process_chunk, chunk_kde_pairs), 
                                  total=len(chunks), desc=f"- {caller} density", unit="chunk"))
            
        # Combine the densities from all chunks
        density = np.concatenate(densities)

        return density
    
    
    @staticmethod
    def estimate(caller, YZ):
        """
        Estimate the density through KDE.

        Args:
            YZ (Process or [Process]): Process(es) for density estimation.

        Returns:
            ndarray: density.
        """
        if not isinstance(YZ, list): YZ = [YZ]
        
        YZ_data = np.column_stack([yz.aligndata for yz in YZ])
        YZ_mesh = np.meshgrid(*[yz.samples for yz in YZ])
        YZ_samples = np.column_stack([yz.ravel() for yz in YZ_mesh])
        
        # Create the grid search
        bandwidths = [0.1, 0.5, 1]
        Ks = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # grid_search = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths, 'kernel': Ks})
            # grid_search.fit(YZ_data)

            # Fit a kernel density model to the data
            # kde = KernelDensity(bandwidth=grid_search.best_params_['bandwidth'], kernel=grid_search.best_params_['kernel'])
            kde = KernelDensity(bandwidth=0.5, kernel='gaussian', algorithm='ball_tree')
            kde.fit(YZ_data)

            # Compute the density
            density = Density.compute_density_parallel(caller, kde, YZ_samples)
            density = density.reshape(YZ_mesh[0].shape)
        return density
              
           
    @staticmethod
    def expectation(y, p):
        """
        Compute the expectation E[y*p(y)/sum(p(y))].

        Args:
            y (ndarray): process samples.
            p (ndarray): probability density function. Note it must be ALREADY NORMALISED.

        Returns:
            float: expectation E[y*p(y)/sum(p(y))].
        """
        if np.sum(p) == 0:
            return np.nan
        expectation_Y_given_X = np.sum(y * p)
        return expectation_Y_given_X
    
    
    @staticmethod
    def mode(y, p):
        """
        Compute the mode, which is the most likely valueof y.

        Args:
            y (ndarray): process samples.
            p (ndarray): probability density function. Note it must be ALREADY NORMALISED.

        Returns:
            float: mode Mode(y*p(y)).
        """
        return y[np.argmax(p)]

    
    @staticmethod
    def normalise(p):
        """
        Normalise the probability density function to ensure it sums to 1.

        Args:
            p (ndarray): probability density function.

        Returns:
            ndarray: normalised probability density function.
        """
        if np.sum(p) != 1:
            return p / np.sum(p)
        return p
         
    
    def predict(self, given_p: Dict[str, float] = None):
        if self.parents is None: 
            dens = self.MarginalDensity
        else:
            indices_X = {}
            for p in given_p.keys():
                # column_indices = np.where(np.isclose(self.parents[p].samples, given_p[p], atol=0.25))[0]                
                column_indices = np.where(np.isclose(self.parents[p].samples, given_p[p], atol=self.atol))[0]                
                indices_X[p] = np.array(sorted(set(column_indices)))

            eval_cond_density = copy.deepcopy(self.CondDensity)

            # For each parent, apply the conditions independently
            for p in indices_X.keys():
                parent_axis = list(self.parents.keys()).index(p) + 1
                parent_indices = indices_X[p]

                # Create a mask to zero out entries not matching parent indices along the specified axis
                mask = np.ones(eval_cond_density.shape[parent_axis], dtype=bool)
                mask[parent_indices] = False

                # Apply mask along the specified axis
                eval_cond_density = np.where(np.expand_dims(mask, axis=tuple(i for i in range(eval_cond_density.ndim) if i != parent_axis)), 
                                             0, 
                                             eval_cond_density)
                
            eval_cond_density = np.sum(eval_cond_density, axis=tuple([list(self.parents.keys()).index(p) + 1 for p in indices_X.keys()]))

            # Reshape eval_cond_density
            dens = Density.normalise(eval_cond_density.reshape(-1, 1))
            
        # expectation = Density.expectation(self.y.samples, dens)
        most_likely = Density.mode(self.y.samples, dens)
        return dens, most_likely