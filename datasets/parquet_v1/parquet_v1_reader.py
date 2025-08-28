import os
from PIL import Image
from tqdm import tqdm
import pyarrow.parquet as pq
import io
from utils.reader_utils import Reader

class ParquetReader(Reader):
    """
    A PyTorch Dataset class for reading images and labels from Parquet files.

    This class provides functionality to read images and labels stored in Parquet files.
    It supports handling both images and labels, and can process large datasets efficiently.

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
        return int(elem.rstrip('.parquet').split('_')[-1])

    def _read(self):
        """
        Read and open Parquet files for images and labels.

        This method loads all Parquet files for images and labels into lists of ParquetFile
        objects. It ensures that the files are processed in the correct order based on their
        numeric suffix.

        Returns
        -------
        tuple
            A tuple containing:
            - im_list (list of pq.ParquetFile): List of ParquetFile objects for images.
            - labels_list (list of pq.ParquetFile): List of ParquetFile objects for labels.
        """
        self.images_path = os.path.join(self.dataset_path, "images")
        self.labels_path = os.path.join(self.dataset_path, "labels")

        images = sorted(os.listdir(self.images_path), key=self.custom_sort)
        labels = sorted(os.listdir(self.labels_path), key=self.custom_sort)

        progress_bar = tqdm(total=len(images), desc='Processing Files', unit='file')

        im_list = []
        labels_list = []

        for image, label in zip(images, labels):
            im_list.append(pq.ParquetFile(os.path.join(self.images_path, image), memory_map=True))
            labels_list.append(pq.ParquetFile(os.path.join(self.labels_path, label), memory_map=True))

        progress_bar.update(1)
        progress_bar.close()

        return im_list, labels_list

    def __getitem__(self, idx):
        """
        Get an image and its label by index.

        Retrieves the image and its corresponding label based on the provided index. The image
        is read from a Parquet file, converted to a PIL Image, and any specified transformations
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
        list_ind, arr_ind = divmod(idx, self.images_per_file)

        image_binary = self.im_list[list_ind].read()[0][arr_ind].as_py()
        label = self.labels_list[list_ind].read()[0][arr_ind].as_py()

        image = Image.open(io.BytesIO(image_binary))

        if self.transform is not None:
            image = self.transform(image)
        return image, label