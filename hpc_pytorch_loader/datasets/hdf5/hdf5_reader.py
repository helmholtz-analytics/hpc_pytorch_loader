import os
import h5py
import numpy as np
from PIL import Image
from hpc_pytorch_loader.utils.reader_utils import Reader

class Hdf5Reader(Reader):
    """
    A PyTorch Dataset class for reading images and labels from HDF5 files.

    This class provides an interface to load images and labels from HDF5 files 
    and supports transformations on the images. It manages HDF5 file handles 
    for both images and labels and provides methods to retrieve specific 
    images and labels by index.

    Attributes
    ----------
    hdf5_dataset_path : str
        Path to the HDF5 dataset directory.
    transform : callable, optional
        A function/transform to apply to the images. Defaults to None.
    images_path : str
        Path to the directory containing HDF5 image files.
    labels_path : str
        Path to the directory containing HDF5 label files.
    hdf5_images_handles : list
        List of open HDF5 file handles for images.
    hdf5_labels_handles : list
        List of open HDF5 file handles for labels.
    class_to_idx : dict
        Mapping from class names to indices.
    shape : tuple
        Shape of the images.
    len : int
        Total number of images in the dataset.
    images_per_file : int
        Number of images per HDF5 file.
    len_last_array : int
        Number of images in the last HDF5 file.
    total_num_arrays : int
        Total number of HDF5 files.
    mode : str
        Image mode (e.g., 'RGB').
    """

    @staticmethod
    def custom_sort(elem):
        """
        Custom sorting function to sort file names based on the numeric value 
        at the end of the file name (before the extension).

        This function extracts the numeric part from the file name to ensure that 
        files are processed in the correct order.

        Parameters
        ----------
        elem : str
            The file name.

        Returns
        -------
        int
            The numeric value extracted from the file name, used for sorting.
        """
        # Split the filename, remove the extension, then extract the numeric part for sorting
        return int(os.path.splitext(elem)[0].split('_')[-1])

    def _read(self):
        """
        Read and open HDF5 files for images and labels.

        This method iterates over the sorted list of HDF5 files for images and labels, 
        opens each file, and stores the file handles in lists. These handles are used 
        later to access the image and label data.

        Returns
        -------
        tuple
            A tuple containing two lists:
            - hdf5_images_handles (list): List of HDF5 file handles for images.
            - hdf5_labels_handles (list): List of HDF5 file handles for labels.
        """

        # Paths to the directories containing HDF5 files for images and labels
        images_path = os.path.join(self.dataset_path, "images")
        labels_path = os.path.join(self.dataset_path, "labels")

        # Get a sorted list of image and label files using the custom_sort method
        images_files = sorted(os.listdir(images_path), key=self.custom_sort)
        labels_files = sorted(os.listdir(labels_path), key=self.custom_sort)

        hdf5_images_handles = []
        hdf5_labels_handles = []

        # Open each image and label file and store the file handles
        for image_file, label_file in zip(images_files, labels_files):
            # Open HDF5 files for images in read mode and add to the list
            hdf5_images_handles.append(h5py.File(os.path.join(images_path, image_file), 'r'))
            # Open HDF5 files for labels in read mode and add to the list
            hdf5_labels_handles.append(h5py.File(os.path.join(labels_path, label_file), 'r'))
        
        # Return the lists of HDF5 file handles for images and labels
        return hdf5_images_handles, hdf5_labels_handles

    def __getitem__(self, idx):
        """
        Get an image and its label by index.

        This method retrieves an image and its associated label from the HDF5 files 
        using the provided index. It converts the image from a NumPy array to a PIL 
        Image and applies any specified transformations.

        Parameters
        ----------
        idx : int
            The index of the image and label to retrieve.

        Returns
        -------
        tuple
            A tuple containing:
            - image (PIL.Image): The retrieved image.
            - label (int): The label associated with the image.
        """
        # Calculate the index of the HDF5 file and the index within that file
        list_ind, arr_ind = divmod(idx, self.images_per_file)
        
        # Retrieve the image data from the HDF5 file using the computed indices
        images_data = self.im_list[list_ind]['images'][arr_ind]
        # Retrieve the label data, ensuring it is converted to int64
        labels_data = (self.labels_list[list_ind]['labels'][arr_ind]).astype(np.int64)
        
        # Convert the NumPy array to a PIL image
        image = Image.fromarray(images_data)
        
        # Apply the specified transformation, if any
        if self.transform is not None:
            image = self.transform(image)
        
        # Return the transformed image and the corresponding label
        return image, labels_data
