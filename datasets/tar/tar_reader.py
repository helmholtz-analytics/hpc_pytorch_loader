import os
from PIL import Image
import json
from tqdm import tqdm
import tarfile
import io
from utils.reader_utils import Reader
from PIL import  UnidentifiedImageError

class TarReader(Reader):
    """
    A PyTorch Dataset class for reading images and labels from Tar files and JSON files.

    This class reads and loads images from Tar files and their associated labels from JSON files. It supports
    processing datasets that are split into multiple Tar and JSON files.

    Attributes
    ----------
    dataset_path : str
        Path to the directory containing the Tar files for images and JSON files for labels.
    transform : callable, optional
        A function or transform to apply to the images.
    im_list : list
        List of Tar file member lists for images.
    labels_list : list
        List of labels loaded from JSON files.
    tar_files_list : list
        List of TarFile objects.
    images_per_file : int
        Number of images per Tar file.
    """

    @staticmethod
    def custom_sort(elem):
        """
        Custom sorting function to sort Tar files based on numeric suffix in file names.

        Parameters
        ----------
        elem : str
            The file name to sort.

        Returns
        -------
        int
            The numeric value extracted from the file name.
        """
        return int(elem.rstrip('.tar').split('_')[-1])

    @staticmethod
    def custom_sort_labels(elem):
        """
        Custom sorting function to sort JSON label files based on numeric suffix in file names.

        Parameters
        ----------
        elem : str
            The file name to sort.

        Returns
        -------
        int
            The numeric value extracted from the file name.
        """
        return int(elem.rstrip('.json').split('_')[-1])

    def _read(self):
        """
        Read and open Tar files and JSON files for images and labels.

        This method loads all Tar files for images and JSON files for labels into lists. Each Tar file is opened
        and its members (images) are added to the list. JSON files are read to get the labels.

        Returns
        -------
        tuple
            A tuple containing:
            - im_list (list of list): List of Tar file member lists for images.
            - labels_list (list of list): List of labels loaded from JSON files.
        """
        # Define paths to images and labels directories
        self.images_path = os.path.join(self.dataset_path, "images")
        self.labels_path = os.path.join(self.dataset_path, "labels")

        # Get and sort the file names
        images = sorted(os.listdir(self.images_path), key=self.custom_sort)
        labels = sorted(os.listdir(self.labels_path), key=self.custom_sort_labels)

        # Initialize progress bar
        progress_bar = tqdm(total=len(images), desc='Processing Files', unit='file')

        im_list = []
        labels_list = []
        tar_files_list = []

        # Load images and labels
        for image, label in zip(images, labels):
            # Open Tar file and get its members
            tar_file = tarfile.open(os.path.join(self.images_path, image))
            tar_files_list.append(tar_file)
            im_list.append(tar_file.getmembers())

            # Read labels from JSON file
            with open(os.path.join(self.labels_path, label), 'r') as jsonfile:
                labels = json.load(jsonfile)

            labels_list.append(labels)

        # Update and close the progress bar
        progress_bar.update(1)
        progress_bar.close()

        # Store Tar file objects for later use
        self.tar_files_list = tar_files_list

        return im_list, labels_list

    def __getitem__(self, idx):
        """
        Get an image and its label by index.

        This method retrieves an image and its corresponding label based on the provided index. The image is
        read from the Tar file as a binary object and then converted to a PIL Image. Any specified transformations
        are applied to the image.

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
        # Determine Tar file and image index based on the global index
        list_ind, arr_ind = divmod(idx, self.images_per_file)

        # Get the specific image member from the Tar file
        tar_member = self.im_list[list_ind][arr_ind]
        image_binary = self.tar_files_list[list_ind].extractfile(tar_member)
        #image_binary = self.im_list[list_ind].extractfile(f"image_{arr_ind}")

        # Retrieve the label for the image
        label = self.labels_list[list_ind][arr_ind]

        # Convert binary image data to a PIL Image
        try:
            image = Image.open(io.BytesIO(image_binary.read()))
        except UnidentifiedImageError:
            print(image_binary)
            self.num_skips = self.num_skips +1
            print(self.num_skips)


        # Apply transformation if specified
        if self.transform is not None:
            image = self.transform(image)
        return image, label
