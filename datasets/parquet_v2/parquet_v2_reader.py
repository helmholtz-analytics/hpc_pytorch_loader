import os
from PIL import Image
from tqdm import tqdm
import pyarrow.parquet as pq
import io
from utils.reader_utils import Reader

class ParquetReader(Reader):
    """
    A PyTorch Dataset class for reading images and labels from Parquet files.

    This class allows reading images and their associated labels from Parquet files. It supports
    efficient data loading and transformation, making it suitable for large datasets.

    Attributes
    ----------
    memmap_dataset_path : str
        Path to the memory-mapped dataset.
    transform : callable, optional
        A function or transform to apply to the images.
    im_list : list
        List of ParquetFile objects for images.
    labels_list : list
        List of ParquetFile objects for labels.
    sizes_list : list
        List of sizes for each Parquet file.
    class_to_idx : dict
        Mapping from class names to indices.
    len : int
        Total number of images in the dataset.
    images_per_file : int
        Number of images per Parquet file.
    len_last_array : int
        Number of images in the last Parquet file.
    total_num_arrays : int
        Total number of Parquet files.
    images_path : str
        Path to the directory containing Parquet image files.
    labels_path : str
        Path to the directory containing Parquet label files.
    """

    @staticmethod
    def custom_sort(elem):
        """
        Custom sorting function to sort file names based on the numeric value 
        at the end of the file name (before the extension).

        This function is used to ensure that files are processed in the correct order
        based on a numeric suffix in the file names.

        Parameters
        ----------
        elem : str
            The file name to sort.

        Returns
        -------
        int
            The numeric value extracted from the file name.
        """
        # Extract the numeric suffix from the file name (assumes filenames like "images_1.parquet")
        return int(elem.rstrip('.parquet').split('_')[-1])

    def _read(self):
        """
        Read and open Parquet files for images and labels.

        This method loads all Parquet files for images and labels into lists of ParquetFile
        objects. It ensures that files are processed in the correct order based on their
        numeric suffix.

        Returns
        -------
        tuple
            A tuple containing:
            - im_list (list of pq.ParquetFile): List of ParquetFile objects for images.
            - labels_list (list of pq.ParquetFile): List of ParquetFile objects for labels.
        """
        # Define paths for images and labels
        self.images_path = os.path.join(self.dataset_path, "images")
        self.labels_path = os.path.join(self.dataset_path, "labels")

        # List all files in the images and labels directories, and sort them using custom_sort
        images = sorted(os.listdir(self.images_path), key=self.custom_sort)
        labels = sorted(os.listdir(self.labels_path), key=self.custom_sort)

        # Initialize the progress bar to monitor the loading process
        progress_bar = tqdm(total=len(images), desc='Processing Files', unit='file')

        im_list = []  # List to store ParquetFile objects for images
        labels_list = []  # List to store ParquetFile objects for labels

        # Load each pair of image and label files
        for image, label in zip(images, labels):
            # Read the Parquet files into memory-mapped ParquetFile objects
            im_list.append(pq.ParquetFile(os.path.join(self.images_path, image), memory_map=True))
            labels_list.append(pq.ParquetFile(os.path.join(self.labels_path, label), memory_map=True))

        # Update and close the progress bar after loading
        progress_bar.update(1)
        progress_bar.close()

        # Return lists of ParquetFile objects for images and labels
        return im_list, labels_list

    def __getitem__(self, idx):
        """
        Get an image and its label by index.

        Retrieves an image and its corresponding label from the Parquet files based on the
        provided index. The image is read from the Parquet file, converted to a PIL Image,
        and any specified transformations are applied.

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
        # Calculate which Parquet file and which element in that file correspond to the given index
        list_ind, arr_ind = divmod(idx, self.images_per_file)

        # Read the binary image data from the Parquet file
        image_binary = self.im_list[list_ind].read(columns=[str(arr_ind)])[0][0].as_py()
        # Read the label from the Parquet file
        label = self.labels_list[list_ind].read(columns=[str(arr_ind)])[0][0].as_py()

        # Convert the binary image data into a PIL image
        image = Image.open(io.BytesIO(image_binary))

        # Apply any transformations if they are defined
        if self.transform is not None:
            image = self.transform(image)

        # Return the image and its corresponding label
        return image, label
