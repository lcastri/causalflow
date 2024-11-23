import copy
from itertools import product
import math
import os
import warnings
from multiprocessing import Manager
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from causalflow.CPrinter import CP
from causalflow.causal_reasoning.Process import Process
from typing import Dict
from sklearn.neighbors import KernelDensity
from causalflow.basics.constants import *
from causalflow.causal_reasoning.Density_utils import *


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


def process_chunk(chunk_kde):
    """
    Define a helper function for imap that takes a chunk and kde

    Args:
        chunk_kde (_type_): _description_

    Returns:
        _type_: _description_
    """
    chunk, kde = chunk_kde
    return compute_density(chunk, kde)
 
 
class Density():
    # def __init__(self, 
    #              y: Process, 
    #              batch_size: int, 
    #              parents: Dict[str, Process] = None):
    #     """
    #     Class constructor.

    #     Args:
    #         y (Process): target process.
    #         batch_size (int): Batch size.
    #         parents (Dict[str, Process], optional): Target's parents. Defaults to None.
    #     """
    #     self.y = y
    #     self.batch_size = batch_size
    #     self.parents = parents
    #     self.DO = {}

    #     if self.parents is not None:
    #         self.DO = {treatment: {ADJ: None, 
    #                             P_Y_GIVEN_DOX_ADJ: None, 
    #                             P_Y_GIVEN_DOX: None} for treatment in self.parents.keys()}
    #     self._preprocess()
            
    #     # init density variables
    #     self.PriorDensity = None
    #     self.JointDensity = None
    #     self.ParentJointDensity = None
    #     self.CondDensity = None
    #     self.MarginalDensity = None
    #     self.PriorDensity = self.computePriorDensity()
    #     self.JointDensity = self.computeJointDensity()
    #     self.ParentJointDensity = self.computeParentJointDensity()
    #     self.CondDensity = self.computeConditionalDensity()
    #     self.MarginalDensity = self.computeMarginalDensity()
        
    def __init__(self, 
                 y: Process, 
                 batch_size: int, 
                 parents: Dict[str, Process] = None,
                 prior_density=None, 
                 joint_density=None, 
                 parent_joint_density=None, 
                 cond_density=None, 
                 marginal_density=None):
        """
        Class constructor.

        Args:
            y (Process): target process.
            batch_size (int): Batch size.
            parents (Dict[str, Process], optional): Target's parents. Defaults to None.
            prior_density (np.array, optional): Precomputed prior density. Defaults to None.
            joint_density (np.array, optional): Precomputed joint density. Defaults to None.
            parent_joint_density (np.array, optional): Precomputed parent joint density. Defaults to None.
            cond_density (np.array, optional): Precomputed conditional density. Defaults to None.
            marginal_density (np.array, optional): Precomputed marginal density. Defaults to None.
        """
        self.y = y
        self.batch_size = batch_size
        self.parents = parents
        self.DO = {}

        if self.parents is not None:
            self.DO = {treatment: {ADJ: None, 
                                P_Y_GIVEN_DOX_ADJ: None, 
                                P_Y_GIVEN_DOX: None} for treatment in self.parents.keys()}
        
        # If precomputed densities are provided, set them directly
        self.PriorDensity = prior_density
        self.JointDensity = joint_density
        self.ParentJointDensity = parent_joint_density
        self.CondDensity = cond_density
        self.MarginalDensity = marginal_density
        
        # Check if any density is None and run _preprocess() if needed
        self._preprocess()
            
        # Only compute densities if they were not provided
        if self.PriorDensity is None: self.PriorDensity = self.computePriorDensity()
        if self.JointDensity is None: self.JointDensity = self.computeJointDensity()
        if self.ParentJointDensity is None: self.ParentJointDensity = self.computeParentJointDensity()
        if self.CondDensity is None: self.CondDensity = self.computeConditionalDensity()
        if self.MarginalDensity is None: self.MarginalDensity = self.computeMarginalDensity()
            
        
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
            self.PriorDensity = Density.estimate('Prior', self.y, self.batch_size)
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
                self.JointDensity = Density.estimate('Joint', yz, self.batch_size)
                del yz
            else:
                self.JointDensity = Density.estimate('Joint', self.y, self.batch_size)
        return self.JointDensity
    
    
    def computeParentJointDensity(self):
        """
        Compute the parents's joint density p(parents).

        Returns:
            ndarray: parents's joint density p(parents).
        """
        if self.ParentJointDensity is None:
            if self.parents is not None: 
                self.ParentJointDensity = Density.estimate('Parent', [p for p in self.parents.values()], self.batch_size)
                self.ParentJointDensity = self.ParentJointDensity
        return self.ParentJointDensity
    
         
    def computeConditionalDensity(self):
        """
        Compute the conditional density p(y|parents) = p(y, parents) / p(parents).

        Returns:
            ndarray: conditional density p(y|parents) = p(y, parents) / p(parents).
        """
        CP.info("    - Conditional density")
        if self.CondDensity is None:
            if self.parents is not None:
                self.CondDensity = self.JointDensity / self.ParentJointDensity[np.newaxis, :] + np.finfo(float).eps
            else:
                self.CondDensity = self.PriorDensity
        return self.CondDensity
    
    
    def computeMarginalDensity(self):
        """
        Compute the marginal density p(y) = \sum_parents p(y, parents).

        Returns:
            ndarray: marginal density p(y) = \sum_parents p(y, parents).
        """
        CP.info("    - Marginal density")
        if self.MarginalDensity is None:
            if self.parents is None:
                self.MarginalDensity = self.PriorDensity
            else:
                # Sum over parents axis
                self.MarginalDensity = np.sum(
                    self.JointDensity, axis=tuple(range(1, len(self.JointDensity.shape)))
                )
        return self.MarginalDensity
    
    # @staticmethod
    # def compute_density_parallel(caller, kde, YZ_samples, batch_size):
    #     """
    #     Compute the density for a set of samples in parallel, processing in batches to reduce memory usage.

    #     Args:
    #         caller (str): Identifier for the caller, used in tqdm display.
    #         kde (KernelDensity): Fitted KernelDensity model.
    #         YZ_samples (ndarray): Samples for which to compute the density.
    #         batch_size (int): Number of samples to process in each batch.

    #     Returns:
    #         ndarray: Computed density for the input samples.
    #     """
    #     YZ_samples = np.array(YZ_samples, copy=True)
    #     num_samples = len(YZ_samples)
    #     if num_samples == 0:
    #         return np.array([])

    #     # Get the number of CPU cores available
    #     available_cores = os.cpu_count()
    #     n_jobs = min(available_cores, num_samples)

    #     # Initialize list to collect densities from each batch
    #     density = np.empty(num_samples, dtype=np.float32)  # Pre-allocate the array

    #     # Create a persistent pool for all batch processing
    #     with ProcessPoolExecutor(max_workers=n_jobs) as executor:
    #         # Process in batches
    #         for batch_start in tqdm(range(0, num_samples, batch_size), desc=f"    - {caller} density", unit="batch"):
    #             batch = YZ_samples[batch_start:batch_start + batch_size]
    #             num_batch_samples = len(batch)
                
    #             # Split the current batch into chunks for parallel processing
    #             chunk_size = max(1, num_batch_samples // n_jobs)
    #             chunk_kde_pairs = [(batch[i:i + chunk_size], kde) for i in range(0, num_batch_samples, chunk_size)]

    #             # Parallel computation of densities for each chunk in the batch
    #             batch_densities = list(executor.map(process_chunk, chunk_kde_pairs))

    #             # Concatenate the densities from all chunks in the current batch
    #             density[batch_start:batch_start + num_batch_samples] = np.concatenate(batch_densities)
        
    #     # Concatenate densities from all batches
    #     return density
 
    # @staticmethod
    # def estimate(caller, YZ, batch_size):
    #     """
    #     Estimate the density through KDE.

    #     Args:
    #         YZ (Process or [Process]): Process(es) for density estimation.

    #     Returns:
    #         ndarray: density.
    #     """
    #     if not isinstance(YZ, list): YZ = [YZ]
        
    #     YZ_data = np.column_stack([yz.aligndata for yz in YZ])
    #     YZ_mesh = np.meshgrid(*[yz.samples for yz in YZ], indexing='ij')
    #     YZ_samples = np.column_stack([yz.ravel() for yz in YZ_mesh])
        
    #     grids = [yz.samples for yz in YZ]
    #     mesh_shape = tuple(len(grid) for grid in grids)
        
    #     mesh_shape = YZ_mesh[0].shape
    #     del YZ_mesh
    #     # Create the grid search
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")

    #         # Fit a kernel density model to the data
    #         kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
    #         kde.fit(YZ_data)

    #         # Compute the density
    #         density = Density.compute_density_parallel(caller, kde, YZ_samples, batch_size)
    #         density = density.reshape(mesh_shape)
            
    #     return density
    


    @staticmethod
    def estimate(caller, YZ, batch_size):
        """
        Estimate the density through KDE.

        Args:
            YZ (Process or [Process]): Process(es) for density estimation.

        Returns:
            ndarray: density.
        """
        if not isinstance(YZ, list):
            YZ = [YZ]
        
        # Prepare the input data for KDE fitting
        YZ_data = np.column_stack([yz.aligndata for yz in YZ])
        grids = [yz.samples for yz in YZ]
        mesh_shape = tuple(len(grid) for grid in grids)
        
        # Fit the KDE model
        kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
        kde.fit(YZ_data)
        
        # Create a generator to lazily generate batches of YZ_samples
        def lazy_sample_generator():
            for combination in product(*grids):
                yield np.array(combination)

        # Create the batch generator to yield samples in chunks
        def batch_generator():
            batch = []
            for sample in lazy_sample_generator():
                batch.append(sample)
                if len(batch) == batch_size:
                    yield np.array(batch)
                    batch = []
            if batch:
                yield np.array(batch)

        # Calculate total batches without fully iterating over the generator
        total_samples = np.prod([len(grid) for grid in grids])  # Total number of samples
        total_batches = math.ceil(total_samples / batch_size)  # Total batches (rounded up)        
        
        # Manager for progress bar
        manager = Manager()
        progress = manager.Value('i', 0)  # Shared value for progress tracking

        # Compute the density in parallel for each batch
        density_batches = []
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            CP.info(f"    - {caller} density", noConsole=True)
            for batch in tqdm(batch_generator(), desc=f"[INFO]:     - {caller} density", total=total_batches, unit="batch"):
                num_batch_samples = len(batch)
                
                # Split the current batch into chunks for parallel processing
                chunk_size = max(1, num_batch_samples // os.cpu_count())
                chunk_kde_pairs = [(batch[i:i + chunk_size], kde) for i in range(0, num_batch_samples, chunk_size)]

                # Parallel computation of densities for each chunk in the batch
                batch_densities = list(executor.map(process_chunk, chunk_kde_pairs))

                progress.value += 1

                # Flatten the list of densities from all chunks in the current batch
                density_batches.extend(batch_densities)

        # Concatenate and reshape to match the grid
        density = np.concatenate(density_batches).reshape(mesh_shape)

        del YZ_data, grids, mesh_shape, kde, total_samples, density_batches
        return density





    
    # def predict(self, given_p: Dict[str, float] = None, tol = 0.25):
    #     if self.parents is None: 
    #         dens = self.MarginalDensity
    #     else:
    #         indices_X = {}
    #         for p in given_p.keys():
    #             if isinstance(tol, dict):
    #                 column_indices = np.where(np.isclose(self.parents[p].samples, given_p[p], atol=tol[p]))[0]                
    #             else:
    #                 column_indices = np.where(np.isclose(self.parents[p].samples, given_p[p], atol=tol))[0]                
    #             indices_X[p] = np.array(sorted(set(column_indices)))

    #         eval_cond_density = copy.deepcopy(self.CondDensity)

    #         # For each parent, apply the conditions independently
    #         for p, indices in indices_X.items():
    #             parent_axis = list(self.parents.keys()).index(p) + 1
    #             indices = np.array(indices, dtype=np.int64)
    #             eval_cond_density = np.take(eval_cond_density, indices, axis=parent_axis)
                    
    #         eval_cond_density = np.sum(eval_cond_density, axis=tuple([list(self.parents.keys()).index(p) + 1 for p in indices_X.keys()]))

    #         # Reshape eval_cond_density
    #         dens = normalise(eval_cond_density.reshape(-1, 1))
                
    #     # expectation = expectation(self.y.samples, dens)
    #     most_likely = mode(self.y.samples, dens)
    #     return dens, most_likely
    
    def predict(self, given_p: Dict[str, float] = None):
        if self.parents is None: 
            dens = self.MarginalDensity
        else:
            indices_X = {}
            for p in given_p.keys():
                closest_index = np.argmin(np.abs(self.parents[p].samples - given_p[p]))
                indices_X[p] = closest_index

            # Extract the specific slice of eval_cond_density corresponding to the closest match
            eval_cond_density = copy.deepcopy(self.CondDensity)
            for p, idx in indices_X.items():
                parent_axis = list(self.parents.keys()).index(p) + 1
                eval_cond_density = np.take(eval_cond_density, indices=[idx], axis=parent_axis)

            dens = normalise(eval_cond_density.flatten())  # Normalize after extracting closest match
            
        most_likely = mode(self.y.samples, dens)
        return dens, most_likely
