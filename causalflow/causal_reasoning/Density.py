import copy
import cupy as cp
from cuml import KernelDensity as cuKDE
from sklearn.neighbors import KernelDensity as KDE
from causalflow.causal_reasoning.NewDensity import DensityManager

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
from causalflow.causal_reasoning.Density_utils import *


# Helper function for CPU processing
def cpu_process(kde_cpu, batch):
    log_density = kde_cpu.score_samples(batch)
    density = np.exp(log_density)
    return density

# Helper function for GPU processing
def gpu_process(kde_gpu, batch):
    batch = cp.array(batch)
    log_density = kde_gpu.score_samples(batch)
    density = cp.exp(log_density)
    cp._default_memory_pool.free_all_blocks()
    return density

 
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
    def estimate(caller, YZ, use_gpu=True, batch_size=50):
        if not isinstance(YZ, list):
            YZ = [YZ]

        YZ_data_gpu = cp.column_stack([cp.asarray(yz.aligndata) for yz in YZ]) if use_gpu else None
        YZ_data_cpu = np.column_stack([yz.aligndata for yz in YZ])

        YZ_mesh = np.meshgrid(*[yz.samples for yz in YZ])
        YZ_samples = np.column_stack([np.ravel(yz) for yz in YZ_mesh])

        kde_gpu = cuKDE(bandwidth=0.5, kernel='gaussian') if use_gpu else None
        kde_cpu = KDE(bandwidth=0.5, kernel='gaussian')

        if use_gpu:
            kde_gpu.fit(YZ_data_gpu)
        kde_cpu.fit(YZ_data_cpu)

        # Combine results from both GPU and CPU
        density_manager = DensityManager(YZ_samples, kde_gpu, kde_cpu)
        density_combined = density_manager.run(caller)

        if density_combined.size != YZ_mesh[0].size:
            raise ValueError(f"Cannot reshape array of size {density_combined.size} into shape {YZ_mesh[0].shape}.")

        density = density_combined.reshape(YZ_mesh[0].shape)
        return density
         
    
    def predict(self, given_p: Dict[str, float] = None):
        if self.parents is None: 
            dens = self.MarginalDensity
        else:
            indices_X = {}
            for p in given_p.keys():
                column_indices = np.where(np.isclose(self.parents[p].sorted_samples, given_p[p], atol=0.25))[0]                
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
        most_likely = mode(self.y.sorted_samples, dens)
        return dens, most_likely