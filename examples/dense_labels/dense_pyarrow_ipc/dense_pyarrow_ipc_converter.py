import os
import pyarrow as pa

import sys
sys.path.append(r"C:\Users\a.hmouda\Desktop\clean_code\dataloader_bachelor_project")

from utils.converter_utils import ConverterFlexibleSize
from utils.utils import image_to_binary, distributedConverter

@distributedConverter
class DenseIpcConverter(ConverterFlexibleSize):
    """
    A class to convert datasets into IPC (Inter-Process Communication) format using Parquet files.

    This class converts images and labels into Parquet files for efficient inter-process communication.
    It supports distributed data processing to handle large datasets.

    Attributes
    ----------
    output_path : str
        Path to the directory where the output Parquet files will be saved.
    batch_size : int
        Number of images and labels to process in each batch.
    num_workers : int
        Number of worker threads for data loading.
    im_mode : str
        Image mode (e.g., 'RGB').
    shape : tuple
        Shape of the images.
    """

    @staticmethod
    def _custom_collate_fn(batch):
        """
        Custom collate function for the DataLoader.

        Converts each image into a Parquet record_batch containing the binary representation of the image.
        Collects the images and labels into separate batches.

        Parameters
        ----------
        batch : list of tuple
            List of tuples where each tuple contains an image and a label.

        Returns
        -------
        tuple
            A tuple containing:
            - images (list of pa.RecordBatch): List of record batches where each batch contains binary images.
            - labels (list of pa.RecordBatch): List of record batches where each batch contains labels.
        """
        # Define the schema for images and labels to store them in record batches
        images_schema = pa.schema([pa.field('image_data', pa.binary())])
        labels_schema = pa.schema([pa.field('label', pa.binary())])

        images = []  # List to store the record batches of images
        labels = []  # List to store the record batches of labels

        # Process each image and label in the batch
        for (image, label) in batch:
            # Convert the image to binary format
            binary_image = image_to_binary(image)
            # Create an array with the binary image
            binary_array = pa.array([binary_image], type=pa.binary())
            # Create a record batch for the image
            images_batch = pa.record_batch([binary_array], schema=images_schema)
            # Add the record batch to the images list
            images.append(images_batch)

            # Convert the image to binary format
            binary_label = image_to_binary(label)
            # Create an array with the binary image
            binary_array = pa.array([binary_label], type=pa.binary())
            # Create a record batch for the image
            labels_batch = pa.record_batch([binary_array], schema=labels_schema)
            # Add the record batch to the images list
            labels.append(labels_batch)

        # Return the lists of image and label record batches
        return images, labels

    def _write_data_to_disk(self, images, labels, file_number):
        """
        Write images and labels to disk as Parquet files.

        This method saves the images and labels into separate Parquet files using the provided file number
        as the suffix for the filenames. Each file contains multiple record batches.

        Parameters
        ----------
        images : list of pa.RecordBatch
            List of record batches where each batch contains binary images.
        labels : list of pa.RecordBatch
            List of record batches where each batch contains labels.
        file_number : int
            The file number suffix for the Parquet files, used to differentiate between different files.
        """
        # Define the schema for images and labels to write them into Parquet files
        images_schema = pa.schema([pa.field('image_data', pa.binary())])
        labels_schema = pa.schema([pa.field('label', pa.binary())])

        # Construct file paths for images and labels Parquet files
        images_path = os.path.join(self.output_path, 'images', f'images_{file_number}.parquet')
        labels_path = os.path.join(self.output_path, 'labels', f'labels_{file_number}.parquet')

        # Write image data to the Parquet file
        with pa.OSFile(images_path, 'wb') as f:
            with pa.ipc.new_file(f, images_schema) as writer:
                # Write each record batch to the file
                for im in images:
                    writer.write(im)

        # Write label data to the Parquet file
        with pa.OSFile(labels_path, 'wb') as f:
            with pa.ipc.new_file(f, labels_schema) as writer:
                # Write each record batch to the file
                for lb in labels:
                    writer.write(lb)
