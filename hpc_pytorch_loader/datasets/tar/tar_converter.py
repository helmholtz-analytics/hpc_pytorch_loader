import os
import json
import tarfile
import io
from hpc_pytorch_loader.utils.converter_utils import ConverterFlexibleSize
from hpc_pytorch_loader.utils.utils import image_to_binary, distributedConverter

@distributedConverter
class TarConverter(ConverterFlexibleSize):
    """
    A class to convert datasets into Tar format for images and JSON format for labels.

    This class handles the conversion of images and labels into Tar and JSON files for efficient data storage
    and access. It supports distributed data processing to handle large datasets.

    Attributes
    ----------
    output_path : str
        Path to the directory where the output Tar and JSON files will be saved.
    batch_size : int
        Number of images and labels to process in each batch.
    num_workers : int
        Number of worker threads for data loading.
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
            # Convert the image to binary format
            images.append(image_to_binary(image))
            # Append the label
            labels.append(label)
        return images, labels

    def _write_data_to_disk(self, images, labels, file_number):
        """
        Write images and labels to disk as Tar and JSON files.

        This method saves the images into a Tar file and the labels into a JSON file. Each file is named using the
        provided file number as the suffix to differentiate between different batches of data.

        Parameters
        ----------
        images : list of bytes
            List of binary representations of images.
        labels : list of int
            List of labels corresponding to the images.
        file_number : int
            The file number suffix for the output files, used to differentiate between different files.
        """
        # Define file paths for the Tar and JSON files
        images_path = os.path.join(self.output_path, 'images', f'images_{file_number}.tar')
        labels_path = os.path.join(self.output_path, 'labels', f'labels_{file_number}.json')

        # Write images to a Tar file
        with tarfile.open(images_path, 'w') as tar:
            for idx, img in enumerate(images):
                # Create a TarInfo object for each image
                tarinfo = tarfile.TarInfo(name=f"image_{idx}")
                tarinfo.size = len(img)
                # Add the image bytes buffer to the Tar file
                tar.addfile(tarinfo, io.BytesIO(img))
    
        # Write labels to a JSON file
        with open(labels_path, 'w') as jsonfile:
            # Serialize the list of labels to a JSON formatted string
            json.dump(labels, jsonfile)
