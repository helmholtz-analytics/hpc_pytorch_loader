import os
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from utils.utils import image_to_binary, distributedConverter
from utils.converter_utils import ConverterFlexibleSize


@distributedConverter
class ParquetConverter(ConverterFlexibleSize):
    """
    A class to convert datasets into Parquet format.

    This class handles the conversion of datasets into Parquet files, which are efficient for storage
    and retrieval in data processing pipelines. It supports handling images and labels, and can
    operate in a distributed manner to process large datasets.

    Attributes
    ----------
    input_dataset : Dataset
        The input dataset to be converted.
    output_path : str
        Path to the directory where output Parquet files will be saved.
    images_per_file : int
        Number of images to be stored in each Parquet file.
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

        Converts each image into its binary representation and collects images and labels into separate batches.

        Parameters
        ----------
        batch : list of tuple
            List of tuples where each tuple contains an image and a label.

        Returns
        -------
        tuple
            A tuple containing:
            - images (list of bytes): List of binary representations of images.
            - labels (list of int): List of labels corresponding to the images.
        """
        images = []
        labels = []
        for (image, label) in batch:
            images.append(image_to_binary(image))
            labels.append(label)
        return images, labels

    def _write_data_to_disk(self, images, labels, file_number):
        """
        Write images and labels to disk as Parquet files.

        Saves the provided images and labels into separate Parquet files. Each file contains multiple
        images or labels, organized into tables for efficient storage and retrieval.

        Parameters
        ----------
        images : list of bytes
            List of binary representations of images.
        labels : list of int
            List of labels corresponding to the images.
        file_number : int
            The file number suffix to be used in the filenames for the Parquet files.
        """
        image_array = pa.array(images, type=pa.binary())
        labels_array = pa.array(labels, type=pa.int64())

        image_table = pa.Table.from_arrays([image_array], names=["images"])
        labels_table = pa.Table.from_arrays([labels_array], names=["labels"])

        images_path = os.path.join(self.output_path, 'images', f'images_{file_number}.parquet')
        labels_path = os.path.join(self.output_path, 'labels', f'labels_{file_number}.parquet')

        pq.write_table(image_table, images_path)
        pq.write_table(labels_table, labels_path)
