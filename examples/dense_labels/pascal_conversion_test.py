import argparse
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),'..','..')))

from dense_pyarrow_ipc.dense_pyarrow_ipc_converter import DenseIpcConverter
from dense_memmap.dense_memmap_converter import DenseMemmapConverter

from torchvision.datasets import VOCSegmentation
from utils.utils import ResizeAndConvert




CONVERTERCLASSES = {
    "memmap": DenseMemmapConverter,
    "pyarrow_ipc": DenseIpcConverter,
    #"hdf5": Hdf5Converter,
    #"parquet_v2": ParquetConverter_v2,
    #"tar": TarConverter,
}

FIXED_SIZE_CONVERTERS = ["memmap","hdf5"]


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download and Convert Pascal dataset")
    parser.add_argument("--input_path", type=str, default=".",
                        help="Base directory where the data will be stored")

    parser.add_argument('--format', default="memmap", type=str,
                        help='format to convert the dataset into')

    parser.add_argument('--img_per_file', default=1000, type=int,
                        help='number of images per file')
    
    parser.add_argument('--batch_size', default=250, type=int,
                        help='batch size for the conversion process')
    
    parser.add_argument('--num_workers', default=0, type=int,
                        help='number of workers for the conversion process')

    parser.add_argument("--output_dir", type=str, default=".",
                        help="Ourput directory where the data will be stored")
    args = parser.parse_args()


    transform = ResizeAndConvert(shape=(256, 256), mode="RGB")
    dataset = VOCSegmentation(root=args.input_path, year="2012", image_set="train", download=True, transform=transform, target_transform=transform)

    output_path = os.path.join(args.output_dir, args.format)
    os.makedirs(output_path, exist_ok=True)


    kwargs = {
                "input_data": dataset,
                "output_path": output_path,
                "images_per_file": args.img_per_file,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "dist": False
            }
    
    if args.format not in CONVERTERCLASSES.keys():
        raise TypeError("Not supported type!")
    
    if args.format in FIXED_SIZE_CONVERTERS:
        kwargs["im_mode"] = "RGB"
        kwargs["shape"] = (256,256)

    converter = CONVERTERCLASSES[args.format](**kwargs)
    converter.convert()

    print(f"{args.format}-{args.img_per_file} Converted Successfully")

if __name__ == '__main__':
    main()

  
