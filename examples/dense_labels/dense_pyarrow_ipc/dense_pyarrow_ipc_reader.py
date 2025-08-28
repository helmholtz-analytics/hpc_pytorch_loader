import os
from PIL import Image
from tqdm import tqdm
import pyarrow as pa
import io

import sys
sys.path.append(r"C:\Users\a.hmouda\Desktop\clean_code\dataloader_bachelor_project")

from utils.reader_utils import Reader

class DenseIpcReader(Reader):
    """
    A PyTorch Dataset class for reading images and labels from IPC (Inter-Process Communication) files.

    This class reads and loads images and labels from IPC files using Apache Arrow's IPC format. It supports
    processing datasets that are split into multiple IPC files.

    Attributes
    ----------
    memmap_dataset_path : str
        Path to the IPC dataset.
    transform : callable, optional
        A function or transform to apply to the images.
    im_list : list
        List of IPC file objects for images.
    labels_list : list
        List of IPC file objects for labels.
    class_to_idx : dict
        Mapping from class names to indices.
    len : int
        Total number of images in the dataset.
    images_per_file : int
        Number of images per IPC file.
    len_last_array : int
        Number of images in the last IPC file.
    total_num_arrays : int
        Total number of IPC files.
    images_path : str
        Path to the directory containing IPC image files.
    labels_path : str
        Path to the directory containing IPC label files.
    """

    def __init__(self, dataset_path, transform=None, target_transform=None):
        """
        Initialize the IpcReader class.

        Parameters
        ----------
        dataset_path : str
            Path to the IPC dataset.
        transform : callable, optional
            A function or transform to apply to the images.
        target_transform : callable, optional
            A function or transform to apply to the labels.
        """
        super().__init__(dataset_path, transform)
        self.target_transform = target_transform

    @staticmethod
    def custom_sort(elem):
        """
        Custom sorting function to sort file names based on the numeric value 
        at the end of the file name (before the extension).

        This function is used to ensure IPC files are processed in the correct order.

        Parameters
        ----------
        elem : str
            The file name to sort.

        Returns
        -------
        int
            The numeric value extracted from the file name.
        """
        # Extract the numeric value from the file name for sorting
        return int(elem.rstrip('.parquet').split('_')[-1])

    def _read(self):
        """
        Read and open IPC files for images and labels.

        This method loads all IPC files for images and labels into lists. Each file is opened
        using Apache Arrow's IPC format for further processing.

        Returns
        -------
        tuple
            A tuple containing:
            - im_list (list of pyarrow.Table): List of IPC file objects for images.
            - labels_list (list of pyarrow.Table): List of IPC file objects for labels.
        """
        # Define paths for images and labels directories
        self.images_path = os.path.join(self.dataset_path, "images")
        self.labels_path = os.path.join(self.dataset_path, "labels")

        # Get sorted lists of image and label files
        images = sorted(os.listdir(self.images_path), key=self.custom_sort)
        labels = sorted(os.listdir(self.labels_path), key=self.custom_sort)

        # Initialize a progress bar for tracking file processing
        progress_bar = tqdm(total=len(images), desc='Processing Files', unit='file')

        im_list = []  # List to store image IPC file objects
        labels_list = []  # List to store label IPC file objects

        # Open each IPC file and add it to the respective lists
        for image, label in zip(images, labels):
            # Open IPC files using memory mapping for efficient file access
            im_list.append(pa.ipc.open_file(pa.memory_map(os.path.join(self.images_path, image), 'rb')))
            labels_list.append(pa.ipc.open_file(pa.memory_map(os.path.join(self.labels_path, label), 'rb')))

        # Update progress bar after processing files
        progress_bar.update(1)
        progress_bar.close()

        # Return the lists of image and label IPC file objects
        return im_list, labels_list

    def __getitem__(self, idx):
        """
        Get an image and its label by index.

        This method retrieves an image and its corresponding label based on the provided index.
        The image is read from the IPC file as a binary object and then converted to a PIL Image.
        Any specified transformations are applied to the image.

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
        # Determine which IPC file and which record within the file correspond to the given index
        list_ind, arr_ind = divmod(idx, self.images_per_file)

        # Retrieve the binary image data and label from the IPC files
        image_binary = self.im_list[list_ind].get_batch(arr_ind)[0][0].as_py()
        label_binary = self.labels_list[list_ind].get_batch(arr_ind)[0][0].as_py()

        # Convert binary image data to a PIL image
        image = Image.open(io.BytesIO(image_binary))
        label = Image.open(io.BytesIO(label_binary))

        # Apply any specified transformations to the image
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            label = self.target_transform(label)

        # Return the image and label
        return image, label
