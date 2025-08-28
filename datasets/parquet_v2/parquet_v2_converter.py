import os
import pyarrow as pa
import pyarrow.parquet as pq
from utils.utils import image_to_binary, distributedConverter
from utils.converter_utils import ConverterFlexibleSize


@distributedConverter
class ParquetConverter(ConverterFlexibleSize):
    """
    A class to convert datasets to Parquet format.

    This class handles the conversion of images and labels from a dataset into Parquet format files.
    It uses PyArrow to manage the conversion and storage of data in Parquet format, optimizing for 
    efficient storage and access.

    Attributes
    ----------
    input_dataset : Dataset
        The input dataset to be converted.
    output_path : str
        Path to the directory where output files will be saved.
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

        Converts each image into its binary representation and collects the images and labels into separate
        batches. This function is used to prepare data for conversion to Parquet format.

        Parameters
        ----------
        batch : list
            List of tuples where each tuple contains an image and a label.

        Returns
        -------
        tuple
            Two lists:
            - images (list of pa.Array): Binary representations of images.
            - labels (list of pa.Array): Labels as integers.
        """
        images = []
        labels = []

        # Iterate over each image and label in the batch
        for (image, label) in batch:
            # Convert the image to its binary representation
            binary_image = image_to_binary(image)
            # Append the binary image and label to their respective lists
            images.append(pa.array([binary_image], type=pa.binary()))  # Use pa.binary() for image data
            labels.append(pa.array([label], type=pa.int64()))  # Use pa.int64() for labels

        # Return lists of pyarrow arrays for images and labels
        return images, labels

    def _write_data_to_disk(self, images, labels, file_number):
        """
        Write images and labels to disk as Parquet files.

        This method writes the provided images and labels to Parquet files on disk. Each file is
        named using a file number suffix to distinguish between different batches of data.

        Parameters
        ----------
        images : list
            List of pa.Array objects containing the binary representations of images.
        labels : list
            List of pa.Array objects containing the labels.
        file_number : int
            The file number suffix for the Parquet files.
        """
        # Create a PyArrow table from the list of image arrays
        image_table = pa.Table.from_arrays(images, names=[str(i) for i in range(len(images))])
        # Create a PyArrow table from the list of label arrays
        labels_table = pa.Table.from_arrays(labels, names=[str(i) for i in range(len(labels))])

        # Construct paths for the output Parquet files for images and labels
        images_path = os.path.join(self.output_path, 'images', f'images_{file_number}.parquet')
        labels_path = os.path.join(self.output_path, 'labels', f'labels_{file_number}.parquet')

        # Write the image table to a Parquet file
        pq.write_table(image_table, images_path)
        # Write the labels table to a Parquet file
        pq.write_table(labels_table, labels_path)
