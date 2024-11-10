import concurrent.futures
import numpy as np
import cupy as cp
import time
from tqdm import tqdm
import os

class DensityManager:
    def __init__(self, YZ_samples, kde_gpu, kde_cpu, batch_size=1000):
        self.YZ_samples = YZ_samples  # Data to be processed
        self.kde_gpu = kde_gpu  # GPU model
        self.kde_cpu = kde_cpu  # CPU model
        self.batch_size = batch_size  # Size of each batch
        self.max_cpu_workers = os.cpu_count() 
        self.total_batches = int(np.ceil(len(YZ_samples) / batch_size))  # Total number of batches
        
        # Create the dictionary {batch_id: batch_samples}
        self.batches = {
            batch_idx: np.array(self.YZ_samples[batch_idx * batch_size : min((batch_idx + 1) * batch_size, len(self.YZ_samples))])
            for batch_idx in range(self.total_batches)
        }
        
        self.gpu_tasks = []  # List to track GPU tasks
        self.cpu_tasks = []  # List to track CPU tasks

    def gpu_process(self, batch):
        """
        Processes a batch on the GPU.
        """
        log_density = self.kde_gpu.score_samples(batch)
        density = cp.exp(log_density)
        cp._default_memory_pool.free_all_blocks()  # Free memory to avoid memory issues
        return density

    def cpu_process(self, batch):
        """
        Processes a batch on the CPU.
        """
        log_density = self.kde_cpu.score_samples(batch)
        density = np.exp(log_density)
        return density

    def run(self, caller):
        """
        Manages GPU and CPU workers, processes the batches, and waits for free resources.
        """
        gpu_densities, cpu_densities = [], []

        # Create the progress bar
        with tqdm(total=self.total_batches, desc=f"- {caller} Computation", position=0) as pbar:

            # Create the CPU worker pool (15 cores)
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_cpu_workers) as cpu_executor:

                # Loop over batch ids
                batch_ids = list(self.batches.keys())  # Extract batch IDs
                while batch_ids:
                    # Check if GPU is free, assign to GPU if possible
                    if len(self.gpu_tasks) == 0 and batch_ids:  # No active GPU tasks
                        batch_id = batch_ids.pop(0)
                        batch = self.batches[batch_id]
                        # Assign batch to GPU
                        gpu_future = cpu_executor.submit(self.gpu_process, batch)  # Submit to GPU task
                        self.gpu_tasks.append(gpu_future)  # Track GPU task

                    # If GPU is busy, assign tasks to CPU if there are available cores
                    if len(self.cpu_tasks) < self.max_cpu_workers and batch_ids:
                        batch_id = batch_ids.pop(0)
                        batch = self.batches[batch_id]
                        cpu_future = cpu_executor.submit(self.cpu_process, batch)  # Assign batch to CPU
                        self.cpu_tasks.append(cpu_future)  # Track CPU task

                    # # Wait for at least one GPU or CPU task to complete
                    # if self.gpu_tasks or self.cpu_tasks:
                    #     # Check GPU tasks (if any)
                    #     completed_gpu_tasks = [task for task in self.gpu_tasks if task.done()]
                    #     for task in completed_gpu_tasks:
                    #         gpu_densities.append(task.result())  # Collect result from GPU
                    #         self.gpu_tasks.remove(task)  # Remove completed GPU task
                    #         pbar.update(1)  # Update progress bar for GPU task

                    #     # Check CPU tasks (if any)
                    #     completed_cpu_tasks = [task for task in self.cpu_tasks if task.done()]
                    #     for task in completed_cpu_tasks:
                    #         cpu_densities.append(task.result())  # Collect result from CPU
                    #         self.cpu_tasks.remove(task)  # Remove the completed CPU task
                    #         pbar.update(1)  # Update progress bar for each CPU task

                    # Wait for all remaining GPU tasks to finish
                    while self.gpu_tasks:
                        completed_gpu_tasks = [task for task in self.gpu_tasks if task.done()]
                        for task in completed_gpu_tasks:
                            gpu_densities.append(task.result())  # Collect result from GPU
                            self.gpu_tasks.remove(task)  # Remove completed GPU task
                            pbar.update(1)

                    # Wait for all remaining CPU tasks to finish
                    while self.cpu_tasks:
                        completed_cpu_tasks = [task for task in self.cpu_tasks if task.done()]
                        for task in completed_cpu_tasks:
                            cpu_densities.append(task.result())  # Collect result from CPU
                            self.cpu_tasks.remove(task)  # Remove the completed CPU task
                            pbar.update(1)
                    # If both GPU and CPU are busy, wait before checking again
                    time.sleep(0.01)
                    

        # Combine the results
        density_combined = np.concatenate([cp.asnumpy(d).reshape(-1) if isinstance(d, cp.ndarray) else np.array([d]) for d in gpu_densities + cpu_densities])

        return density_combined
