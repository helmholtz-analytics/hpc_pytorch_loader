import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.utils import get_number_of_channels
from utils.reader_utils import Reader
#from torch import from_numpy as t_from_numpy
from torchvision import transforms
to_tensor = transforms.ToTensor()

class MemmapReader(Reader):
    """
    A PyTorch Dataset class for reading images and labels from memory-mapped files.

    This class reads and loads images and labels from memory-mapped NumPy arrays. It supports
    handling both distributed data processing and non-distributed scenarios.

    Attributes
    ----------
    images_path : str
        Path to the directory containing memory-mapped image files.
    labels_path : str
        Path to the directory containing memory-mapped label files.
    im_list : list
        List of memory-mapped image arrays.
    labels_list : list
        List of memory-mapped label arrays.
    mode : str
        Image mode (e.g., 'RGB').
    shape : tuple
        Shape of the images.
    len : int
        Total number of images in the dataset.
    images_per_file : int
        Number of images per memory-mapped file.
    len_last_array : int
        Number of images in the last memory-mapped file.
    total_num_arrays : int
        Total number of memory-mapped files.
    """

    def _read_single_array(self, image, label, array_length):
        """
        Read a single memory-mapped array for images and labels.

        This method loads a memory-mapped NumPy array for images and labels from disk.

        Parameters
        ----------
        image : str
            The image file name.
        label : str
            The label file name.
        array_length : int
            The number of elements in the array.

        Returns
        -------
        tuple
            A tuple containing:
            - image_array (np.memmap): Memory-mapped array of images.
            - label_array (np.memmap): Memory-mapped array of labels.
        """
        # Determine the number of channels in the image based on the mode
        number_of_channels = get_number_of_channels(self.mode)
        # Set the shape of the image based on the number of channels
        shape = (self.shape[0], self.shape[1], number_of_channels) if number_of_channels != 1 else self.shape

        # Construct the paths for the image and label memory-mapped files
        arr_path = os.path.join(self.images_path, image)
        label_path = os.path.join(self.labels_path, label)

        # Load the memory-mapped array for the images
        image_array = np.memmap(arr_path, dtype=np.uint8, mode='r', shape=(array_length, *shape))
        # Load the memory-mapped array for the labels
        label_array = np.memmap(label_path, dtype=np.int64, mode='r', shape=(array_length,))
        return image_array, label_array

    def _read(self):
        """
        Read and open memory-mapped files for images and labels.

        This method loads all memory-mapped image and label files into lists. It also handles
        the case where the last file may contain a different number of images.

        Returns
        -------
        tuple
            A tuple containing:
            - images_list (list of np.memmap): List of memory-mapped arrays for images.
            - labels_list (list of np.memmap): List of memory-mapped arrays for labels.
        """

        images_list = []
        labels_list = []

        # Define paths to the image and label directories
        self.images_path = os.path.join(self.dataset_path, "images")
        self.labels_path = os.path.join(self.dataset_path, "labels")

        # Get the sorted list of image and label file names
        images = sorted(os.listdir(self.images_path), key=self.custom_sort)
        labels = sorted(os.listdir(self.labels_path), key=self.custom_sort)

        # Progress bar to indicate the processing status
        progress_bar = tqdm(total=self.total_num_arrays, desc='Processing Files', unit='file')

        # Iterate through all image and label files except the last one
        for image, label in zip(images[:-1], labels[:-1]):
            # Read and append each memory-mapped array to the lists
            image_array, label_array = self._read_single_array(image, label, self.images_per_file)
            images_list.append(image_array)
            labels_list.append(label_array)
            progress_bar.update(1)  # Update the progress bar

        # Handle the last file separately if it has a different number of images
        if self.len_last_array != 0:
            image_array, label_array = self._read_single_array(images[-1], labels[-1], self.len_last_array)
        else:
            # If the last array has the same length as the others
            image_array, label_array = self._read_single_array(images[-1], labels[-1], self.images_per_file)

        # Append the last array to the lists
        images_list.append(image_array)
        labels_list.append(label_array)

        progress_bar.update(1)  # Update the progress bar for the last file
        progress_bar.close()  # Close the progress bar

        return images_list, labels_list

    def __getitem__(self, idx):
        """
        Get an image and its label by index.

        This method retrieves an image and its corresponding label based on the provided index.
        The image is converted from a NumPy array to a PIL Image, and any specified transformations
        are applied.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve.

        Returns
        -------
        tuple
            A tuple containing:
            - image (PIL.Image): The image at the specified index.
            - label (int): The label associated with the image.
        """
        # Calculate which memory-mapped file and array index the data is in
        list_ind, arr_ind = divmod(idx, self.images_per_file)
        
        # Convert the memory-mapped array to a PIL image
        image = Image.fromarray(np.array(self.im_list[list_ind][arr_ind]))

        # Retrieve the corresponding label
        label = self.labels_list[list_ind][arr_ind]
        # Apply transformation if specified
        if self.transform:
            return self.transform(image), label

        return image, label
    
    @staticmethod
    def custom_sort(elem):
        """
        Custom sorting function to sort file names based on the numeric value 
        at the end of the file name (before the extension).

        This function extracts the numeric part from the file name to ensure files
        are processed in the correct order.

        Parameters
        ----------
        elem : str
            The file name to sort.

        Returns
        -------
        int
            The numeric value extracted from the file name.
        """
        # Extract the numeric part from the file name for sorting
        return int(elem.rstrip('.npy').split('_')[-1])
