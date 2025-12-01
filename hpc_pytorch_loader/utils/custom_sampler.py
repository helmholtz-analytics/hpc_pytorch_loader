from typing import Optional, Iterator
from torch.utils.data import Dataset, Sampler
import torch.distributed as dist

class CustomDistributedSampler(Sampler):
    """
    A custom sampler for distributed training that divides a dataset among multiple processes.

    This sampler partitions a dataset across multiple replicas (processes) in a distributed setting.
    Each replica is assigned a subset of data for processing. The sampler ensures that the dataset
    is divided among all replicas, with the last replica handling any remaining data if the dataset
    size is not perfectly divisible.

    Parameters
    ----------
    dataset : Dataset
        The dataset to sample from.
    images_per_array : int
        The number of images per array in the dataset.
    num_replicas : int, optional
        The total number of processes among which the data is distributed. If None, it defaults to the 
        world size in the distributed group.
    rank : int, optional
        The rank of the current process within `num_replicas`. If None, it defaults to the current process rank.
    """
    def __init__(
        self,
        dataset: Dataset,
        images_per_array: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        """
        Initialize the CustomDistributedSampler.

        Parameters
        ----------
        dataset : Dataset
            The dataset to sample from.
        images_per_array : int
            The number of images per array in the dataset.
        num_replicas : int, optional
            The total number of processes among which the data is distributed. If None, it defaults to the 
            world size in the distributed group.
        rank : int, optional
            The rank of the current process within `num_replicas`. If None, it defaults to the current process rank.

        Raises
        ------
        RuntimeError
            If the distributed package is not available when `num_replicas` or `rank` are not provided.
        ValueError
            If the provided `rank` is not within the valid range [0, num_replicas - 1].
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.images_per_array = images_per_array
        self.epoch = 0

        total_number_of_arrays = len(self.dataset)// images_per_array
        
        #NOTE we divide amongs all tasks except the last one
        #NOTE because the last one gets the reminder of the arrays
        self.array_per_task = total_number_of_arrays // (num_replicas - 1)

        #NOTE all tasks exept the last one
        if rank < (num_replicas - 1): 
            self.start_index = rank * images_per_array * self.array_per_task
            self.end_index = (rank + 1) * images_per_array * self.array_per_task
            self.num_samples = images_per_array * self.array_per_task
        #NOTE the last task
        else:
            self.start_index = rank * images_per_array * self.array_per_task
            self.end_index = len(self.dataset)
            self.num_samples = len(self.dataset) - self.start_index

    def __iter__(self) -> Iterator:
        """
        Return an iterator over the indices assigned to the current process.

        This method generates the list of indices that the current process should process, 
        based on its rank and the total number of replicas.

        Returns
        -------
        Iterator
            An iterator over the indices assigned to the current process.
        """
        indices = list(range(self.start_index, self.end_index))
        return iter(indices)

    def __len__(self) -> int:
        """
        Return the number of samples assigned to the current process.

        This method returns the number of samples that the current process is responsible for processing.

        Returns
        -------
        int
            The number of samples assigned to the current process.
        """
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch



