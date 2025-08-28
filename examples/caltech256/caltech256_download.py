import subprocess
import tarfile
import os
import argparse

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download and extract Caltech 256 dataset")
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory where the data will be stored")
    args = parser.parse_args()

    # Create the base directory if it doesn't exist
    base_dir = args.base_dir
    os.makedirs(base_dir, exist_ok=True)

    # The link to the file
    caltech256_link = "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1"

    # Output filename
    output_file = os.path.join(base_dir,"256_ObjectCategories.tar")

    # Download the file using subprocess and wget
    subprocess.run(["wget", "-O", output_file, caltech256_link])

    # Extract the tar file
    with tarfile.open(output_file, "r:*") as tar:
        tar.extractall(path=base_dir)

    # Remove the tar file after extraction
    os.remove(output_file)

    print(f"Files extracted to: {base_dir}")
