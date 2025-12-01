from abc import ABC, abstractmethod
import json
import os
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from tqdm import tqdm
from hpc_pytorch_loader.utils.utils import check_input_type, ResizeAndConvert

class Converter(ABC):
    """
    Abstract base class for dataset converters.

    This class provides the core utilities and structure required for converting datasets
    into various formats.

    Parameters
    ----------
    output_path : str
        Path where the converted dataset will be saved.
    images_per_file : int
        Number of images per output file.
    batch_size : int
        Batch size to be used during the conversion process.
    num_workers : int
        Number of workers for data loading.

    Attributes
    ----------
    sampler : None or sampler object
        Sampler used during data loading, if any.
    rank : None or int
        Rank of the current process in distributed settings.
    num_replicas : None or int
        Number of processes in distributed settings.
    output_path : str
        Path where the converted dataset will be saved.
    images_per_file : int
        Number of images per output file.
    batch_size : int
        Batch size used during the conversion process.
    num_workers : int
        Number of workers for data loading.
    num_arrays : int
        Number of output files that will be created.
    len_last_array : int
        Number of images in the last file, which may be less than `images_per_file`.
    class_to_idx : dict
        Mapping from class names to integer labels.
    length : int
        Total number of images in the input dataset.
    """
    def __init__(self,output_path, images_per_file, 
                 batch_size, num_workers):
        """
        Initialize the Converter with common parameters.

        This method sets up the core attributes needed for the conversion process,
        including the output path, batch size, and the number of workers. It also 
        calculates the number of output arrays and the size of the last array.

        Parameters
        ----------
        output_path : str
            Path where the converted dataset will be saved.
        images_per_file : int
            Number of images to be stored per output file.
        batch_size : int
            Batch size to be used during the conversion process.
        num_workers : int
            Number of workers for data loading.
        """
        super().__init__()
        self.sampler = None
        self.rank = None
        self.num_replicas = None
        self.output_path = output_path
        self.images_per_file = images_per_file
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.num_arrays, self.len_last_array = divmod(len(self.input_dataset), self.images_per_file)

        self.class_to_idx = getattr(self.input_dataset, "class_to_idx", None)

        #this is present in the calstec256 dataset
        if getattr(self.input_dataset, "categories", None):
            self.class_to_idx = {i: category for i, category in enumerate(self.input_dataset.categories)}

        self.length = len(self.input_dataset)


    def _save_metadata(self):
        """
        Save metadata related to the conversion process.

        This method writes out important metadata to the output path, including the 
        number of images per file, the total number of files, and the class-to-index
        mapping. This metadata is essential for understanding the structure of the 
        converted dataset and for loading it later.

        Metadata saved includes:
        - Number of images per file.
        - Total number of images.
        - Length of the last file.
        - Total number of arrays/files created.
        - Optional attributes like image mode and shape, for fixed-sized converters
        """
        metadata = {
            "num_images_per_array": self.images_per_file,
            "len": self.length,
            "len_last_array": self.len_last_array,
            "total_num_arrays": self.num_arrays
        }

        #these attributes are exclusive to the fixed size datasets
        if getattr(self,"im_mode", None):
            metadata.update({"mode":self.im_mode})
        
        if getattr(self,"shape",None):
            metadata.update({"shape":self.shape})



        metadata_path = os.path.join(self.output_path, "metadata")
        os.makedirs(metadata_path, exist_ok=True)

        with open(os.path.join(metadata_path, "metadata.json"), 'w') as output:
            json.dump(metadata, output)

        #some datasets do not provide class to integer mapping
        if self.class_to_idx:
            with open(os.path.join(metadata_path, "class_to_integer_mapping.json"), 'w') as output:
                json.dump(self.class_to_idx, output)
        #else:
        #    with open(os.path.join(metadata_path, "class_to_integer_mapping.json"), 'w') as output:
        #        json.dump(output, )

######################################################################################
################## Subclass for Converting into Fixed Size Datasets ##################
######################################################################################

class ConverterFixedSize(Converter, ABC):
    """
    Converter for datasets with fixed image sizes.

    This class provides the commun utilities and structure required for converting datasets
    into fixed-sized formats

    Parameters
    ----------
    input_data : Dataset or str
        The input dataset or path to the dataset.
    output_path : str
        Path where the converted dataset will be saved.
    images_per_file : int
        Number of images per output file.
    batch_size : int
        Batch size to be used during the conversion process.
    num_workers : int
        Number of workers for data loading.
    shape : tuple, optional
        Desired shape for resizing the images.
    im_mode : str, optional
        Image mode (e.g., 'RGB', 'L') for converting images.
    root_in_archive : str, optional
        Root directory inside an archive, if applicable.
    
    Attributes
    ----------
    shape : tuple or None
        Desired shape for resizing the images.
    im_mode : str or None
        Image mode (e.g., 'RGB', 'L') for converting images.
    input_dataset : Dataset
        The processed input dataset.
    """
    def __init__(self, input_data, output_path, images_per_file, 
                 batch_size, num_workers,shape=None, im_mode=None, root_in_archive = None):
        """
        Initialize the ConverterFixedSize with dataset-specific parameters.

        This method prepares the input dataset, applying necessary transformations such as resizing
        and converting the image mode, and initializes the base Converter attributes.

        Parameters
        ----------
        input_data : Dataset or str
            The input dataset or path to the dataset.
        output_path : str
            Path where the converted dataset will be saved.
        images_per_file : int
            Number of images per output file.
        batch_size : int
            Batch size to be used during the conversion process.
        num_workers : int
            Number of workers for data loading.
        shape : tuple, optional
            Desired shape for resizing the images. If None, no resizing is applied.
        im_mode : str, optional
            Image mode (e.g., 'RGB', 'L') for converting images. If None, no conversion is applied.
        root_in_archive : str, optional
            Root directory inside a tar archive, used with Tar converters
        """
        self.shape = shape
        self.im_mode = im_mode
        transform = Compose([ResizeAndConvert(shape, im_mode),])
        self.input_dataset = check_input_type(input_data, transform=transform, root_in_archive=root_in_archive)
        
        super().__init__(output_path, images_per_file, 
                 batch_size, num_workers)
        
    def _convert_to_format(self):
        """
        Convert the dataset to a fixed-size format and save it to disk.

        This method handles the conversion of the input dataset into a fixed-size format,
        where each file contains a fixed number of images. The method ensures that images are
        processed and stored in batches, and handles the progress tracking through a progress bar.

        Raises
        ------
        ValueError
            If the batch size does not evenly divide the number of images per file.
        """
        if self.images_per_file % self.batch_size != 0:
            raise ValueError("Batch size should be chosen in a way it divides the number of images per array")

        data_loader = DataLoader(self.input_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 sampler=self.sampler,
                                 pin_memory=False,
                                 collate_fn=self._custom_collate_fn)
        
        if self.sampler:
            progress_bar = tqdm(total=self.sampler.num_samples, desc=f'Task {self.rank}', unit='image')
        else:
            progress_bar = tqdm(total=len(self.input_dataset), desc=f'processed images', unit='image')

        curr_array_indx = 0
        curr_pos = 0
        for images, labels in iter(data_loader):
            #handle arrays except the last one
            if (curr_pos + 1) * self.batch_size < self.im_list[curr_array_indx].shape[0] :
                self.im_list[curr_array_indx][curr_pos * self.batch_size : (curr_pos + 1) * self.batch_size, :] = images
                self.labels_list[curr_array_indx][curr_pos * self.batch_size : (curr_pos + 1) * self.batch_size] = labels
            #handle the last array as it can have a number of elements not divisible by the batch size
            else:
                self.im_list[curr_array_indx][curr_pos * self.batch_size :, :] = images
                self.labels_list[curr_array_indx][curr_pos * self.batch_size :] = labels
            progress_bar.update(self.batch_size)
            curr_pos += 1
            #check if end of current array is reached and pas to the next
            #sets curr_pos to 0 because it srats from the beginning of the next array
            if ((curr_pos * self.batch_size) % self.images_per_file) == 0:
                curr_array_indx += 1
                curr_pos = 0

        progress_bar.close()

    def convert(self):
        """
        Start the dataset conversion process.

        This method initializes the necessary storage arrays, performs the conversion by
        calling the `_convert_to_format` method, and saves the metadata once the conversion
        is complete. If the conversion is distributed, only the process with rank 0 saves the metadata.
        """
        self._init_arrays()
        self._convert_to_format()
        if not(self.dist) or (self.rank == 0) :
            self._save_metadata()

    @staticmethod
    @abstractmethod
    def _custom_collate_fn(batch):
        pass

    @abstractmethod
    def _init_arrays(self):
        pass

######################################################################################
################## Subclass for Converting into Flexible Size Datasets ###############
######################################################################################

class ConverterFlexibleSize(Converter, ABC):
    """
    Initialize the ConverterFlexibleSize with dataset-specific parameters.

    This method prepares the input dataset without applying transformations, and initializes 
    the base Converter attributes.

    Parameters
    ----------
    input_data : Dataset or str
        The input dataset or path to the dataset.
    output_path : str
        Path where the converted dataset will be saved.
    images_per_file : int
        Number of images per output file.
    batch_size : int
        Batch size to be used during the conversion process.
    num_workers : int
        Number of workers for data loading.
    root_in_archive : str, optional
        Root directory inside a tar archive, used with Tar converters
    """
    def __init__(self, input_data, output_path, images_per_file, 
                 batch_size, num_workers, root_in_archive = None):
        
        self.input_dataset = check_input_type(input_data, transform=None, root_in_archive=root_in_archive)
        super().__init__(output_path, images_per_file, 
                 batch_size, num_workers,)
        
    
    def _convert_to_format(self):
        """
        Convert the dataset to a flexible-size format and save it to disk.

        This method handles the conversion of the input dataset into a Parquet format, 
        where each file can vary in size based on the batch processing. The method manages
        the creation of directories, data loading, and the writing of data to disk.

        Raises
        ------
        ValueError
            If the batch size does not evenly divide the number of images per file.
        """
        if self.images_per_file % self.batch_size != 0:
            raise ValueError("Batch size should be chosen in a way it divides the number of images per array")

        os.makedirs(os.path.join(self.output_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'labels'), exist_ok=True)

        #use a torch dataloader to iterate through the instances of the dataset
        #load then into memory tp apply necessary processing for conversion
        data_loader = DataLoader(self.input_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True,
                                 collate_fn=self._custom_collate_fn,
                                 sampler = self.sampler)

        if self.sampler:
            progress_bar = tqdm(total=self.sampler.num_samples, desc=f'Task {self.rank}', unit='image')
        else:
            progress_bar = tqdm(total=len(self.input_dataset), desc=f'processed images', unit='image')


        is_last = False #indicates wether last array is handled by current task

        #in case of distributed conversion
        #determine which subset of the dataset is processed by the current task
        if self.sampler:
            
            if (self.rank < (self.num_replicas - 1)):
                start = self.sampler.array_per_task * self.rank
            else:
                start = self.rank * self.sampler.array_per_task
                is_last = self.len_last_array != 0
        #no distributed conversion: all instances are handled by the task    
        else:
            start = 0
            is_last = self.len_last_array != 0

        #all images stored currently in memory
        images_global_batch = []

        #all labels stored currently in memory
        labels_global_batch = []

        #tracks the total number of instances stored curently in memory
        global_batch_size = 0
        
        file_number = start

        for (images, labels) in iter(data_loader):
            #read images and labels from disc and append them to the same list
            images_global_batch.extend(images)
            labels_global_batch.extend(labels)

            #update number of instances loaded currently in memory
            global_batch_size += self.batch_size

            #number of images per file is reached
            #save the data to disk and free up memory
            if global_batch_size >= self.images_per_file:
                self._write_data_to_disk(images_global_batch, labels_global_batch, file_number)

                images_global_batch = []
                labels_global_batch = []

                file_number += 1
                global_batch_size = 0

            progress_bar.update(self.batch_size)

        #the above loop is often existed without saving the last global batch
        #due to its size being smaller than the previous ones
        #save it individually to disc
        if is_last and images_global_batch:
            self._write_data_to_disk(images_global_batch, labels_global_batch, file_number)
            self.num_arrays +=1

        progress_bar.close()

    def convert(self):
        """
        Start the dataset conversion process.

        This method performs the conversion by calling the `_convert_to_format` method,
        and saves the metadata once the conversion is complete. If the conversion is distributed,
        only the process with rank 0 saves the metadata.
        """
        self._convert_to_format()
        if not(self.dist) or (self.rank == 0) :
            self._save_metadata()
    
    @staticmethod
    @abstractmethod
    def _custom_collate_fn(batch):
        pass

    @abstractmethod
    def _write_data_to_disk(self):
        pass
    
