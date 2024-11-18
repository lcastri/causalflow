import copy
from concurrent.futures import ProcessPoolExecutor

import os
from multiprocessing import Pool
import warnings
from tqdm import tqdm
from causalflow.CPrinter import CP
from causalflow.causal_reasoning.Process import Process
from typing import Dict
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from causalflow.basics.constants import *
from causalflow.causal_reasoning.Density_utils import *
from scipy.stats import multivariate_normal, norm

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
    def __init__(self, y: Process, parents: Dict[str, Process] = None):
        """
        Class constructor.

        Args:
            y (Process): target process.
            parents (Dict[str, Process], optional): Target's parents. Defaults to None.
        """
        self.y = y
        self.parents = parents
        self.DO = {}

        if self.parents is not None:
            self.DO = {treatment: {ADJ: None, 
                                P_Y_GIVEN_DOX_ADJ: None, 
                                P_Y_GIVEN_DOX: None} for treatment in self.parents.keys()}
        self._preprocess()
            
        # init density variables
        self.PriorDensity = None
        self.JointDensity = None
        self.ParentJointDensity = None
        self.CondDensity = None
        self.MarginalDensity = None
        self.PriorDensity = self.computePriorDensity()
        self.JointDensity = self.computeJointDensity()
        self.ParentJointDensity = self.computeParentJointDensity()
        self.CondDensity = self.computeConditionalDensity()
        self.MarginalDensity = self.computeMarginalDensity()
            
        
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
            
            
    def computePriorDensity(self):
        """
        Compute the prior density p(y).

        Returns:
            ndarray: prior density p(y).
        """
        if self.PriorDensity is None: 
            self.PriorDensity = Density.estimate('Prior', self.y)
        self.PriorDensity = normalise(self.PriorDensity)
        return self.PriorDensity
        
        
    def computeJointDensity(self):
        """
        Compute the joint density p(y, parents).

        Returns:
            ndarray: joint density p(y, parents).
        """
        if self.JointDensity is None:
            if self.parents is not None:
                yz = [p for p in self.parents.values()]
                yz.insert(0, self.y)
                self.JointDensity = Density.estimate('Joint', yz)
            else:
                self.JointDensity = Density.estimate('Joint', self.y)
        self.JointDensity = normalise(self.JointDensity)
        return self.JointDensity
    
    
    def computeParentJointDensity(self):
        """
        Compute the parents's joint density p(parents).

        Returns:
            ndarray: parents's joint density p(parents).
        """
        if self.ParentJointDensity is None:
            if self.parents is not None: 
                self.ParentJointDensity = Density.estimate('Parent', [p for p in self.parents.values()])
                self.ParentJointDensity = normalise(self.ParentJointDensity)
        return self.ParentJointDensity
    
         
    def computeConditionalDensity(self):
        """
        Compute the conditional density p(y|parents) = p(y, parents) / p(parents).

        Returns:
            ndarray: conditional density p(y|parents) = p(y, parents) / p(parents).
        """
        CP.info("- Conditional density")
        if self.CondDensity is None:
            if self.parents is not None:
                self.CondDensity = self.JointDensity / self.ParentJointDensity[np.newaxis, :] + np.finfo(float).eps
            else:
                self.CondDensity = self.PriorDensity
        self.CondDensity = normalise(self.CondDensity)
        return self.CondDensity
    
    
    def computeMarginalDensity(self):
        """
        Compute the marginal density p(y) = \sum_parents p(y, parents).

        Returns:
            ndarray: marginal density p(y) = \sum_parents p(y, parents).
        """
        CP.info("- Marginal density")
        if self.MarginalDensity is None:
            if self.parents is None:
                self.MarginalDensity = self.PriorDensity
            else:                
                # Sum over parents axis
                self.MarginalDensity = copy.deepcopy(self.JointDensity)
                self.MarginalDensity = np.sum(self.MarginalDensity, axis=tuple(range(1, len(self.JointDensity.shape))))  
        self.MarginalDensity = normalise(self.MarginalDensity)
        return self.MarginalDensity
    
    @staticmethod
    def compute_density_parallel(caller, kde, YZ_samples, batch_size=10000):
        """
        Compute the density for a set of samples in parallel, processing in batches to reduce memory usage.

        Args:
            caller (str): Identifier for the caller, used in tqdm display.
            kde (KernelDensity): Fitted KernelDensity model.
            YZ_samples (ndarray): Samples for which to compute the density.
            batch_size (int): Number of samples to process in each batch.

        Returns:
            ndarray: Computed density for the input samples.
        """
        YZ_samples = np.array(YZ_samples, copy=True)
        num_samples = len(YZ_samples)
        if num_samples == 0:
            return np.array([])

        # Get the number of CPU cores available
        available_cores = os.cpu_count()
        n_jobs = min(available_cores, num_samples)

        # Initialize list to collect densities from each batch
        density = []

        # Create a persistent pool for all batch processing
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Process in batches
            for batch_start in tqdm(range(0, num_samples, batch_size), desc=f"- {caller} density", unit="batch"):
                batch = YZ_samples[batch_start:batch_start + batch_size]
                num_batch_samples = len(batch)
                
                # Split the current batch into chunks for parallel processing
                chunk_size = max(1, num_batch_samples // n_jobs)
                chunks = [batch[i:i + chunk_size].copy() for i in range(0, num_batch_samples, chunk_size)]
                chunk_kde_pairs = [(chunk, kde) for chunk in chunks]

                # Parallel computation of densities for each chunk in the batch
                batch_densities = list(executor.map(process_chunk, chunk_kde_pairs))

                # Concatenate the densities from all chunks in the current batch
                density.append(np.concatenate(batch_densities))

        # Concatenate densities from all batches
        return np.concatenate(density, dtype=np.float32)
 
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
        YZ_mesh = np.meshgrid(*[yz.samples for yz in YZ], indexing='ij')
        YZ_samples = np.column_stack([yz.ravel() for yz in YZ_mesh])
        mesh_shape = YZ_mesh[0].shape
        del YZ_mesh
        # Create the grid search
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Fit a kernel density model to the data
            kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
            kde.fit(YZ_data)

            # Compute the density
            density = Density.compute_density_parallel(caller, kde, YZ_samples)
            density = density.reshape(mesh_shape)
            
        return density

    
    # def predict(self, given_p: Dict[str, float] = None, tol = 0.25):
    #     if self.parents is None: 
    #         dens = self.MarginalDensity
    #     else:
    #         indices_X = {}
    #         for p in given_p.keys():
    #             if isinstance(tol, dict):
    #                 column_indices = np.where(np.isclose(self.parents[p].sorted_samples, given_p[p], atol=tol[p]))[0]                
    #             else:
    #                 column_indices = np.where(np.isclose(self.parents[p].sorted_samples, given_p[p], atol=tol))[0]                
    #             indices_X[p] = np.array(sorted(set(column_indices)))

    #         eval_cond_density = copy.deepcopy(self.CondDensity)

    #         # For each parent, apply the conditions independently
    #         for p, indices in indices_X.items():
    #             parent_axis = list(self.parents.keys()).index(p) + 1
    #             eval_cond_density = np.take(eval_cond_density, indices, axis=parent_axis)
                    
    #         eval_cond_density = np.sum(eval_cond_density, axis=tuple([list(self.parents.keys()).index(p) + 1 for p in indices_X.keys()]))

    #         # Reshape eval_cond_density
    #         dens = normalise(eval_cond_density.reshape(-1, 1))
                
    #     # expectation = expectation(self.y.sorted_samples, dens)
    #     most_likely = mode(self.y.sorted_samples, dens)
    #     return dens, most_likely
    
    def predict(self, given_p: Dict[str, float] = None, tol = 0.25):
        if self.parents is None: 
            dens = self.MarginalDensity
        else:
            indices_X = {}
            for p in given_p.keys():
                closest_index = np.argmin(np.abs(self.parents[p].original_samples - given_p[p]))
                indices_X[p] = closest_index

            # Extract the specific slice of eval_cond_density corresponding to the closest match
            eval_cond_density = copy.deepcopy(self.CondDensity)
            for p, idx in indices_X.items():
                parent_axis = list(self.parents.keys()).index(p) + 1
                eval_cond_density = np.take(eval_cond_density, indices=[idx], axis=parent_axis)

            dens = normalise(eval_cond_density.flatten())  # Normalize after extracting closest match
            
        most_likely = mode(self.y.original_samples, dens)
        return dens, most_likely
