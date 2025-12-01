from torch.utils.data import Dataset, get_worker_info
import os
from torchvision import transforms
from PIL import Image
import json
from tqdm import tqdm
import pyarrow as pa
import io
from abc import ABC, abstractmethod

class Reader(Dataset, ABC):
    """
    Initialize the Reader with dataset path and transform.

    This constructor initializes the `Reader` by setting up the dataset path and 
    applying any specified transformations. It also parses the dataset metadata 
    and reads the image and label data from the dataset.

    Parameters
    ----------
    dataset_path : str, optional
        Path to the memory-mapped dataset. Defaults to None.
    transform : callable, optional
        A function/transform to apply to the images. This can include operations 
        like resizing, cropping, or normalization. Defaults to None.
    """

    def _parse_metadata(self):
        """
        Parse metadata from JSON files.

        This method reads and parses metadata files associated with the dataset. The 
        metadata includes information such as the total number of images, the number 
        of images per file, the length of the last file, and additional attributes 
        like image mode and shape if the dataset has a fixed size elements.

        Attributes
        ----------
        class_to_idx : dict
            Mapping from class names to integer labels.
        len : int
            Total number of images in the dataset.
        images_per_file : int
            Number of images per file.
        len_last_array : int
            Number of images in the last file.
        total_num_arrays : int
            Total number of files in the dataset.
        mode : str, optional
            Image mode (e.g., 'RGB', 'L') if specified in the metadata.
        shape : tuple, optional
            Shape of the images (height, width) if specified in the metadata.
        """
        metadata_path = os.path.join(self.dataset_path, "metadata")
        with open(os.path.join(metadata_path, "metadata.json"), "r") as input_file:
            metadata = json.load(input_file)

        try:
            with open(os.path.join(metadata_path, "class_to_integer_mapping.json"), 'r') as input_file:
                self.class_to_idx = json.load(input_file)
        except FileNotFoundError:
            self.class_to_idx = None
            

        self.len = metadata["len"]
        self.images_per_file = metadata["num_images_per_array"]
        self.len_last_array = metadata["len_last_array"]
        self.total_num_arrays = metadata["total_num_arrays"]

        #these attributes are exclusive to the fixed size datasets
        mode = metadata.get("mode", None)
        if mode:
            self.mode= mode

        shape = metadata.get("shape",None)
        if shape:
            self.shape = tuple(shape)

    def __init__(self, dataset_path=None, transform=None):
        """
        Initialize the IpcReader.

        Args:
            dataset_path (str, optional): Path to the memory-mapped dataset. Defaults to None.
            transform (callable, optional): A function/transform to apply to the images. Defaults to None.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.transform = transform
        self._parse_metadata()
        self.im_list, self.labels_list = self._read()

    def __len__(self):
        """
        Return the total number of images in the dataset.

        This method provides the length of the dataset, which is determined during 
        metadata parsing.

        Returns
        -------
        int
            Total number of images in the dataset.
        """
        return self.len
    
    def __getitem__(self, index):
        """
        Retrieve an image and its corresponding label by index.

        This method should be implemented by subclasses to return a specific image 
        and its label from the dataset. The implementation will depend on the format 
        of the dataset.

        Parameters
        ----------
        index : int
            Index of the image and label to retrieve.

        Returns
        -------
        tuple
            A tuple containing the image and its corresponding label.
        """
        pass

    @staticmethod
    def worker_init_fn(worker_id):
        """
        NOTE: This is needed only on windows operating system.
        Initialize worker process by reading dataset files.

        This static method is used to initialize each worker process in a data loader. 
        It reads the image and label data into memory for each worker, allowing 
        multiprocessing to work efficiently with the dataset.

        Parameters
        ----------
        worker_id : int
            Worker ID assigned by the data loader.
        """
        dataset = get_worker_info().dataset
        dataset.im_list, dataset.labels_list = dataset._read()

    @staticmethod
    @abstractmethod
    def custom_sort(elem):
        pass

    @abstractmethod
    def _read(self):
        """
        Abstract method to read data from the dataset.

        This method must be implemented by subclasses to read image and label data 
        from the dataset. The implementation will depend on the format of the dataset.

        Returns
        -------
        tuple
            A tuple containing the lists of images and labels.
        """
        pass

    def __getstate__(self):
        """
        NOTE: This is needed only on windows operating system.
        Get the state for pickling the dataset object.

        This method returns the state dictionary for pickling the dataset object, 
        excluding any file handles or other resources that cannot be shared across 
        processes. These resources need to be initialized separately by each worker 
        process, especially in multiprocessing contexts.

        Returns
        -------
        dict
            State dictionary without non-pickleable resources like file handles.
        """
        #check if running on windows then do not forward the file handlers
        if os.name == 'nt':
            state = self.__dict__.copy()
            state.pop('im_list', None)
            state.pop('labels_list', None)

            #relevant only for tar files
            state.pop('tar_files_list', None)
        return state

    def __setstate__(self, state):
        """
        NOTE: This is needed only on windows operating system.
        Set the state of the dataset object during unpickling.

        This method restores the state of the dataset object from the state dictionary 
        during unpickling.

        Parameters
        ----------
        state : dict
            State of the object to be restored.
        """
        self.__dict__.update(state)
