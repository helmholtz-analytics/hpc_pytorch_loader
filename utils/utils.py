from utils.custom_sampler import CustomDistributedSampler
from torchvision.datasets import ImageFolder
import io

def get_number_of_channels(mode):
        """
        Get the number of channels corresponding to a given image mode.

        This function maps an image mode (e.g., 'RGB', 'L') to the number of channels 
        associated with that mode.

        Parameters
        ----------
        mode : str
            The mode of the image (e.g., 'RGB', 'L', 'CMYK').

        Returns
        -------
        int or None
            The number of channels for the given mode, or None if the mode is unrecognized.
        """
        mode_to_channels = {
            "1": 1,     # 1-bit pixels, black and white
            "L": 1,     # 8-bit pixels, grayscale
            "P": 1,     # 8-bit pixels, mapped to any other mode using a color palette
            "RGB": 3,   # 3x8-bit pixels, true color
            "RGBA": 4,  # 4x8-bit pixels, true color with transparency mask
            "CMYK": 4,  # 4x8-bit pixels, color separation
            "YCbCr": 3, # 3x8-bit pixels, color video format
            "LAB": 3,   # 3x8-bit pixels, the L*a*b color space
            "HSV": 3,   # 3x8-bit pixels, Hue, Saturation, Value color space
            "I": 1,     # 32-bit signed integer pixels
            "F": 1      # 32-bit floating point pixels
        }

        return mode_to_channels.get(mode, None)

def check_input_type(input_data, transform, root_in_archive=None):
    """
    Determine the appropriate dataset class based on the input type.

    This function checks whether the input data is a path to a directory or a tar 
    archive and returns the corresponding dataset class (`ImageFolder` or 
    `TarImageFolder`). If the input is not a string, it is returned as is (Assumed to be
    an instance of torch.data.utils.Dataset).

    Parameters
    ----------
    input_data : str or Dataset
        The input data, which can be a path to a dataset directory or tar archive, 
        or Dataset object.
    transform : callable
        A function/transform to apply to the images.
    root_in_archive : str, optional
        Root directory in the tar archive (if applicable). Defaults to None.

    Returns
    -------
    Dataset
        A dataset object suitable for loading the images.
    """
    if isinstance(input_data, str):
                return ImageFolder(root=input_data, transform=transform)
    return input_data

def image_to_binary(img):
    """
    Convert an image to its binary representation.

    This function converts a PIL Image object to a binary stream (bytes), which 
    can be saved or transmitted as binary data.

    Parameters
    ----------
    img : PIL.Image.Image
        The image to convert.

    Returns
    -------
    bytes
        The binary representation of the image.
    """
    img_byte_arr = io.BytesIO()
    if img.format:
        img.save(img_byte_arr, format=img.format)
    else:
        img.save(img_byte_arr, format="JPEG")
    return img_byte_arr.getvalue()

class ResizeAndConvert:
    """
    Resize and convert an image to a specified mode.

    This class provides a callable object that resizes an image to a specified 
    shape and converts it to a specified mode (e.g., 'RGB', 'L').

    Parameters
    ----------
    shape : tuple of int
        The desired output shape of the image (height, width).
    mode : str
        The desired output mode of the image (e.g., 'RGB', 'L').
    """
    def __init__(self, shape, mode):
        self.shape = shape
        self.mode = mode
    def __call__(self, img):
        """
        Apply the resizing and mode conversion to an image.

        Parameters
        ----------
        img : PIL.Image.Image
            The image to be resized and converted.

        Returns
        -------
        PIL.Image.Image
            The transformed image with the specified shape and mode.
        """
        img = img.convert(self.mode)
        img = img.resize((self.shape[0], self.shape[1]))
        return img

def distributedConverter(cls):
    """
    A decorator to add distributed processing capabilities to a class.

    This function decorates a class to include distributed processing features. 
    If distribution is enabled, a custom sampler is used to handle distributed 
    data loading across multiple workers or nodes.

    Parameters
    ----------
    cls : class
        The class to be decorated.

    Returns
    -------
    class
        A new class with added distributed processing functionality.
    """
    class NewClass(cls):
        """
        A wrapper class for distributed processing.

        This class extends the functionality of the input class (`cls`) by adding 
        support for distributed data loading using a custom sampler.

        Parameters
        ----------
        *args : tuple
            Positional arguments for the base class constructor.
        dist : bool, optional
            Whether to enable distributed processing. Defaults to False.
        rank : int, optional
            Rank of the current process in distributed processing. Required if `dist` is True.
        num_replicas : int, optional
            Total number of processes in distributed processing. Required if `dist` is True.
        **kwargs : dict
            Keyword arguments for the base class constructor.
        """
        def __init__(self, *args, dist = False,
                     rank = None,
                     num_replicas = None,
                     **kwargs,):
            super().__init__(*args, **kwargs)

            self.dist = dist

            if dist:
                self.sampler = CustomDistributedSampler(self.input_dataset,self.images_per_file,
                                                        rank = rank,
                                                        num_replicas = num_replicas)
                self.rank = self.sampler.rank
                self.num_replicas = self.sampler.num_replicas
    return NewClass