import os
import numpy as np
import h5py

from utils.utils import get_number_of_channels, distributedConverter
from utils.converter_utils import ConverterFixedSize


@distributedConverter
class Hdf5Converter(ConverterFixedSize):
    """
    A class to convert datasets to HDF5 format.

    This converter handles the transformation of an input dataset into a series of HDF5 files. 
    It supports distributed processing by dividing the dataset among multiple processes and 
    storing the results in separate HDF5 files.

    Attributes
    ----------
    input_dataset : Dataset
        The input dataset to be converted.
    output_path : str
        Path to the directory where output files will be saved.
    images_per_file : int
        Number of images to be stored in each HDF5 file.
    im_mode : str
        Image mode (e.g., 'RGB', 'L', etc.).
    shape : tuple
        Shape of the images.
    batch_size : int
        Size of batches to process at a time.
    num_workers : int
        Number of worker threads for data loading.
    num_arrays : int
        Number of complete arrays (HDF5 files) to be created.
    len_last_array : int
        Length of the last array if the dataset size is not perfectly divisible by images_per_file.
    class_to_idx : dict
        Mapping from class names to indices.
    length : int
        Total number of images in the dataset.
    """

    @staticmethod
    def _custom_collate_fn(batch):
        """
        Custom collate function for the DataLoader.

        Converts each image in the batch to a NumPy array and collects the images 
        and labels into separate arrays.

        Parameters
        ----------
        batch : list
            List of tuples where each tuple contains an image and a label.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple containing two NumPy arrays:
            - images (numpy.ndarray): The images as NumPy arrays.
            - labels (numpy.ndarray): The corresponding labels as integers.
        """
        images = []
        labels = []
        for image, label in batch:
            images.append(np.array(image))
            labels.append(label)
        return np.array(images), np.array(labels)
    

    def _init_arrays(self):
        """
        Initialize HDF5 files for storing images and labels.

        This method creates the necessary HDF5 files and datasets for storing images 
        and labels based on the dataset's configuration. It supports both standard 
        and distributed conversion modes. In distributed mode, it partitions the 
        dataset among multiple processes, with each process handling its assigned 
        portion of the dataset.

        Raises
        ------
        ValueError
            If the provided image mode (`im_mode`) is not supported.
        """
        number_of_channels = get_number_of_channels(self.im_mode)

        if number_of_channels is None:
            raise ValueError(f"The provided mode {self.im_mode} is not supported")
        
        # Adjust the shape of the images depending on the number of channels
        if number_of_channels != 1:
            shape = (self.shape[0], self.shape[1], number_of_channels) # Include channel dimension
        else:
            shape = self.shape # Grayscale or single-channel images

        is_last = False

        # Check if distributed conversion is being used
        if self.sampler:

            # Determine the range of arrays (files) this process will handle
            if (self.rank < (self.num_replicas - 1)):
                start = self.sampler.array_per_task * self.rank
                end = self.sampler.array_per_task * (self.rank+1)
            
            # The last process may have to handle fewer images if the dataset size isn't perfectly divisible
            else:
                start = self.rank * self.sampler.array_per_task
                end = self.num_arrays
                is_last = (self.len_last_array != 0)

        # Not using distributed conversion
        else:
            start = 0
            end = self.num_arrays
            is_last = (self.len_last_array != 0)

        # Create directories for storing HDF5 files for images and labels
        images_path = os.path.join(self.output_path, "images")
        labels_path = os.path.join(self.output_path, "labels")
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(labels_path, exist_ok=True)

        # Initialize lists to store the HDF5 dataset objects
        self.im_list = []
        self.labels_list = []

        # Create HDF5 files and datasets for the assigned range of arrays
        for n in range(start, end):

            # Create HDF5 file for images
            arr_path = os.path.join(images_path, f"images_{n}.h5")
            h5_file = h5py.File(arr_path, 'w')

            # Create the 'images' dataset with the specified shape and data type
            self.im_list.append(h5_file.create_dataset('images', shape=(self.images_per_file, *shape), dtype=np.uint8))

            # Create HDF5 file for labels
            arr_path = os.path.join(labels_path, f"labels_{n}.h5")
            h5_file = h5py.File(arr_path, 'w')

            # Create the 'labels' dataset with the specified shape and data type
            self.labels_list.append(h5_file.create_dataset('labels', shape=(self.images_per_file,), dtype=np.int64))
            

        # If the dataset size is not perfectly divisible by images_per_file, create one more file for the remaining images
        # This is only the case by the last array or file
        if is_last:
            arr_path = os.path.join(images_path, f"images_{self.num_arrays}.h5")
            h5_file = h5py.File(arr_path, 'w')
            self.im_list.append(h5_file.create_dataset('images', shape=(self.len_last_array, *shape), dtype=np.uint8))

            arr_path = os.path.join(labels_path, f"labels_{self.num_arrays}.h5")
            h5_file = h5py.File(arr_path, 'w')
            self.labels_list.append(h5_file.create_dataset('labels', shape=(self.len_last_array,), dtype=np.int64))
            self.num_arrays += 1 # Increment the total number of arrays or files
