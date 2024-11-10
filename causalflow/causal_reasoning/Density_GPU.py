import copy
import warnings

import cupy as cp
from cuml import KernelDensity

# Check if a GPU is available
if cp.cuda.is_available():
    print("Is GPU available?: True")

    # Get the number of GPUs
    num_gpus = cp.cuda.runtime.getDeviceCount()
    print("Number of GPUs available:", num_gpus)

    # Print GPU properties
    for i in range(num_gpus):
        properties = cp.cuda.runtime.getDeviceProperties(i)
        gpu_name = properties['name']  # Access the GPU name
        print(f"GPU {i}: {gpu_name}")
else:
    print("No GPU available.")

from causalflow.CPrinter import CP
from causalflow.causal_reasoning.Process import Process
from typing import Dict
import numpy as np
from causalflow.basics.constants import *
from tqdm import tqdm
from causalflow.causal_reasoning.Density_utils import *

 
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
        self.JointDensity = normalise(self.JointDensity)
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
        self.PriorDensity = normalise(self.PriorDensity)
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
        self.MarginalDensity = normalise(self.MarginalDensity)
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
        self.CondDensity = normalise(self.CondDensity)
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
                self.ParentJointDensity = normalise(self.ParentJointDensity)
        return self.ParentJointDensity
       
    @staticmethod
    def estimate(caller, YZ):
        """
        Estimate the density through KDE using CuML.

        Args:
            YZ (Process or [Process]): Process(es) for density estimation.

        Returns:
            ndarray: density.
        """
        if not isinstance(YZ, list):
            YZ = [YZ]
        
        # Ensure aligndata is a CuPy array before stacking
        YZ_data = cp.column_stack([cp.asarray(yz.aligndata) for yz in YZ])  # Convert to CuPy

        # Create a meshgrid for samples
        YZ_mesh = np.meshgrid(*[yz.samples for yz in YZ])
        YZ_samples = cp.column_stack([cp.ravel(cp.array(yz)) for yz in YZ_mesh])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Initialize and fit CuML's KernelDensity
            kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
            kde.fit(YZ_data)  # Fit the model on GPU

            # Use batch processing to compute log density for samples with tqdm progress bar
            # @profile
            def batch_score_samples(kde, YZ_samples, batch_size = 1000):
                """Scores samples in batches to avoid memory overflow."""
                num_samples = YZ_samples.shape[0]
                densities = []

                # Create a tqdm progress bar
                for i in tqdm(range(0, num_samples, batch_size), desc=f"- {caller} Density"):
                    batch = YZ_samples[i:i + batch_size]
                    log_density = kde.score_samples(batch)
                    densities.append(cp.exp(log_density))  # Compute densities from log densities
                    
                    # Explicitly free GPU memory after processing the batch
                    cp._default_memory_pool.free_all_blocks()

                return cp.concatenate(densities)  # Concatenate all batches

            # Compute densities using batch processing
            densities = batch_score_samples(kde, YZ_samples)

            # Convert the densities back to numpy array (if needed)
            density = cp.asnumpy(densities)  # Move back to CPU if you need a numpy array
            density = density.reshape(YZ_mesh[0].shape)  # Reshape to match the meshgrid

        return density
         
    
    def predict(self, given_p: Dict[str, float] = None):
        if self.parents is None: 
            dens = self.MarginalDensity
        else:
            indices_X = {}
            for p in given_p.keys():
                column_indices = np.where(np.isclose(self.parents[p].samples, given_p[p], atol=0.25))[0]                
                # column_indices = np.where(np.isclose(self.parents[p].samples, given_p[p], atol=self.atol))[0]                
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
            dens = normalise(eval_cond_density.reshape(-1, 1))
            
        # expectation = expectation(self.y.samples, dens)
        most_likely = mode(self.y.samples, dens)
        return dens, most_likely