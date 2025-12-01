import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as T

import argparse

import os
import sys


from dense_pyarrow_ipc.dense_pyarrow_ipc_reader import DenseIpcReader
from dense_memmap.dense_memmap_reader import DenseMemmapReader

def custom_collate_fn(batch):
        """
        Custom collate function for the DataLoader.

        Converts each image in the batch into a numpy array and separates images and
        labels into different batches.

        Parameters
        ----------
        batch : list of tuples
            List where each tuple contains an image and its label.

        Returns
        -------
        tuple
            A tuple containing:
            - images (np.ndarray): Array of images.
            - labels (np.ndarray): Array of labels.
        """
        images = []
        labels = []
        for image, label in batch:
            # Convert the image to a numpy array and append to the images list
            images.append(image)
            # Append the label to the labels list
            labels.append(label)
        # Return a tuple of numpy arrays for images and labels
        return images, labels

def show_batch(images, masks, batch_idx):
        fig, axs = plt.subplots(4, 2, figsize=(8, 12))
        for i in range(4):
            # Convert to PIL if needed
            img = images[i]
            mask = masks[i]

            if isinstance(img, torch.Tensor):
                img = to_pil(img)
            if isinstance(mask, torch.Tensor):
                mask = to_pil(mask)

            axs[i, 0].imshow(img)
            axs[i, 0].set_title(f'Image {batch_idx*4 + i}')
            axs[i, 0].axis('off')

            axs[i, 1].imshow(mask, cmap='gray')
            axs[i, 1].set_title(f'Mask {batch_idx*4 + i}')
            axs[i, 1].axis('off')
        plt.tight_layout()
        #plt.show()
        plt.savefig(f"output_{i}.png")

READERCLASSES = {
    "memmap": DenseMemmapReader,
    "pyarrow_ipc": DenseIpcReader,
    #"hdf5": Hdf5Converter,
    #"parquet_v2": ParquetConverter_v2,
    #"tar": TarConverter,
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read Converted Pascal dataset")
    parser.add_argument("--input_path", type=str, default=".",
                        help="Base directory where the data will be stored")

    parser.add_argument('--format', default="memmap", type=str,
                        help='format to convert the dataset into')
    args = parser.parse_args()
    
    # Dataset and DataLoader setup
    dataset = READERCLASSES[args.format](dataset_path =os.path.join(args.input_path, args.format), transform=None, target_transform=None)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

    # Optional: If images/masks are PIL, convert them to tensors for grid display
    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()

    # Iterate and visualize
    total_images_shown = 0
    for batch_idx, (images, masks) in enumerate(dataloader):
        show_batch(images, masks, batch_idx)
        total_images_shown += len(images)
        if total_images_shown >= 12:
            break
