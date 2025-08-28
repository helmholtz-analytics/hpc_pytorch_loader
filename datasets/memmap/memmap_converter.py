import os
import numpy as np

from utils.utils import get_number_of_channels, distributedConverter
from utils.converter_utils import ConverterFixedSize

@distributedConverter
class MemmapConverter(ConverterFixedSize):
    """
    A class to convert datasets to memmap format.

    This class provides functionality to convert datasets into memory-mapped files
    for efficient data handling. It initializes memory-mapped arrays for storing
    images and labels, and handles distributed processing if applicable.

    Attributes
    ----------
    input_dataset : Dataset
        The input dataset to be converted.
    output_path : str
        Path to the directory where output files will be saved.
    images_per_file : int
        Number of images to be stored in each memmap file.
    im_mode : str
        Image mode (e.g., 'RGB').
    shape : tuple
        Shape of the images.
    batch_size : int
        Size of batches to process at a time.
    num_workers : int
        Number of worker threads for data loading.
    num_arrays : int
        Number of complete arrays.
    len_last_array : int
        Length of the last array.
    class_to_idx : dict
        Mapping from class names to indices.
    length : int
        Total number of images in the dataset.
    """

    @staticmethod
    def _custom_collate_fn(batch):
        """
        Custom collate function for the DataLoader.

        Converts each image in the batch into a numpy array and separates images and
        labels into different batches.

        Parameters
        ----------
        batch : list of tuples
            List where each tuple contains an image and its label.

        Returns
        -------
        tuple
            A tuple containing:
            - images (np.ndarray): Array of images.
            - labels (np.ndarray): Array of labels.
        """
        images = []
        labels = []
        for image, label in batch:
            # Convert the image to a numpy array and append to the images list
            images.append(np.array(image))
            # Append the label to the labels list
            labels.append(label)
        # Return a tuple of numpy arrays for images and labels
        return np.array(images), np.array(labels)

    def _init_arrays(self):
        """
        Initialize memmap arrays for storing images and labels.

        This method sets up memory-mapped files for storing images and labels based 
        on the specified format and parameters. It handles the initialization of files 
        for distributed conversion if applicable, ensuring that the last array is also 
        correctly initialized if it has a different length.

        Raises
        ------
        ValueError
            If the image mode is not supported.
        """
        # Determine the number of channels in the images based on the image mode
        number_of_channels = get_number_of_channels(self.im_mode)

        # Raise an error if the image mode is not supported
        if number_of_channels is None:
            raise ValueError(f"The provided mode {self.im_mode} is not supported")
        
        # Determine the shape of the images based on the number of channels
        if number_of_channels != 1:
            shape = (self.shape[0], self.shape[1], number_of_channels)
        else:
            shape = self.shape

        is_last = False
        # Check if distributed conversion is being used
        if self.sampler:
            # Determine the start and end indices for each replica in distributed mode
            if (self.rank < (self.num_replicas - 1)):
                start = self.sampler.array_per_task * self.rank
                end = self.sampler.array_per_task * (self.rank + 1)
            else:
                start = self.rank * self.sampler.array_per_task
                end = self.num_arrays
                # Determine if the last array has a different length
                is_last = self.len_last_array != 0
        else:
            # Non-distributed mode: process all arrays
            start = 0
            end = self.num_arrays
            # Determine if the last array has a different length
            is_last = self.len_last_array != 0

        # Create directories for storing memory-mapped files
        images_path = os.path.join(self.output_path, "images")
        labels_path = os.path.join(self.output_path, "labels")
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(labels_path, exist_ok=True)

        # Lists to store memory-mapped objects for images and labels
        self.im_list = []
        self.labels_list = []

        # Initialize memory-mapped arrays for each file in the specified range
        for n in range(start, end):
            # Create a memory-mapped file for images and append to the list
            arr_path = os.path.join(images_path, f"images_{n}.npy")
            self.im_list.append(np.memmap(arr_path, dtype=np.uint8, mode='w+', shape=(self.images_per_file, *shape)))

            # Create a memory-mapped file for labels and append to the list
            arr_path = os.path.join(labels_path, f"labels_{n}.npy")
            self.labels_list.append(np.memmap(arr_path, dtype=np.int64, mode='w+', shape=(self.images_per_file,)))

        # If the last array has a different length, initialize it separately
        if is_last:
            # Create a memory-mapped file for the remaining images
            arr_path = os.path.join(images_path, f"images_{self.num_arrays}.npy")
            self.im_list.append(np.memmap(arr_path, dtype=np.uint8, mode='w+', shape=(self.len_last_array, *shape)))

            # Create a memory-mapped file for the remaining labels
            arr_path = os.path.join(labels_path, f"labels_{self.num_arrays}.npy")
            self.labels_list.append(np.memmap(arr_path, dtype=np.int64, mode='w+', shape=(self.len_last_array,)))
            # Increment the number of arrays since an extra one was created
            self.num_arrays += 1
