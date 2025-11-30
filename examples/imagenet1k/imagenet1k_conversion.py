import argparse
import os
import sys
import torch.distributed as dist

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..','..')))


from datasets.parquet_v2.parquet_v2_converter import ParquetConverter as ParquetConverter_v2
from datasets.pyarrow_ipc.pyarrow_ipc_converter import IpcConverter
from datasets.hdf5.hdf5_converter import Hdf5Converter
from datasets.memmap.memmap_converter import MemmapConverter
from datasets.tar.tar_converter import TarConverter




CONVERTERCLASSES = {
    "memmap": MemmapConverter,
    "hdf5": Hdf5Converter,
    "parquet_v2": ParquetConverter_v2,
    "pyarrow_ipc": IpcConverter,
    "tar": TarConverter,
}

FIXED_SIZE_CONVERTERS = ["memmap","hdf5"]


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download and extract Caltech 256 dataset")
    parser.add_argument("--input_path", type=str, default=".",
                        help="Base directory where the data is stored")

    parser.add_argument('--format', default="memmap", type=str,
                        help='format to convert the dataset into')

    parser.add_argument('--img_per_file', default=5000, type=int,
                        help='number of images per file')
    
    parser.add_argument('--batch_size', default=500, type=int,
                        help='batch size for the conversion process')
    
    parser.add_argument('--num_workers', default=2, type=int,
                        help='number of workers for the conversion process')
    
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Ourput directory where the data will be stored")
    args = parser.parse_args()


    # Set up distributed training
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NPROCS'])
    dist.init_process_group('nccl',world_size=world_size, rank=rank)


    imagenet1k_path = args.input_path

    output_path = args.output_dir
    if rank == 0:
        print(f"Converting {imagenet1k_path} to {output_path}")
        os.makedirs(output_path, exist_ok=True)


    kwargs = {
                "input_data": imagenet1k_path,
                "output_path": output_path,
                "images_per_file": args.img_per_file,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "dist": True
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

  
