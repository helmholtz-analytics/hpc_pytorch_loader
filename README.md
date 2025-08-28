# Dataset Conversion Library

This library provides tools for efficient conversion and reading of large image datasets. It supports multiple formats (e.g., Memmap, HDF5, Parquet) and is designed for both fixed-size and flexible-size datasets. With distributed processing capabilities, it ensures fast and scalable dataset preparation for machine learning workflows.

---

## Features

- **Multi-format Support**:
  - **Fixed-size**: Memory-mapped (Memmap), HDF5
  - **Flexible-size**: Parquet, Tar, PyArrow IPC (Inter-Process Communication)
- **Dataset Compatibility**:
  - Supports conversion from image folders (e.g., ImageNet1k structure) or any `torch.Dataset`.
- **Distributed Processing**:
  - Accelerates dataset conversion using distributed processing.
- **Custom Data Loaders**:
  - Efficient reading of converted datasets with PyTorch-compatible loaders.
- **Image Format Support**:
  - Handles various image formats and modes (e.g., RGB, grayscale).

---

## Main Components

1. **Converters**:
   - `MemmapConverter`, `HDF5Converter`, `ParquetConverter`, `IpcConverter`, `TarConverter`
2. **Readers**:
   - `MemmapReader`, `HDF5Reader`, `ParquetReader`, `IpcReader`, `TarReader`
3. **Utilities**:
   - Custom samplers, preprocessing tools, and distributed processing decorators.

---

## Installation

### Local Setup

1. Ensure Python 3 is installed on your system.
2. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/dataloader_bachelor_project.git
   cd dataloader_bachelor_project
   ```
3. Run the setup script:
   ```bash
   bash setup_local.sh
   ```
   This will:
   - Create a virtual environment (`dataloadenv`).
   - Install required dependencies (e.g., PyTorch, torchvision, tqdm, etc.).

4. Activate the virtual environment:
   ```bash
   source dataloadenv/bin/activate
   ```

### Cluster Setup

1. Connect to your cluster and navigate to the project directory.
2. Run the cluster setup script:
   ```bash
   source setup_cluster.sh
   ```
   This will:
   - Load necessary modules (e.g., PyTorch, CUDA).
   - Create and activate a virtual environment.
   - Install required dependencies.

**Note**: The `setup_cluster.sh` script uses the `ml` command to load modules. Adjust it based on your cluster's configuration.

---

## Usage
You can provide either:
### Folder Structure

The input dataset must be organized into a root directory with one subfolder per class. Each subfolder contains the images belonging to that class. Example:

```python
root/
  ├── class1/
  │     ├── img1.jpg
  │     ├── img2.jpg
  ├── class2/
  │     ├── img3.jpg
  │     ├── img4.jpg
```

- The folder name is used as the class label.

- Supported image formats: .jpg, .png, etc.

### Pytorch Dataset
The library also supports input as a PyTorch torch.utils.data.Dataset object.
However, when using a FixedSizeConverter (e.g., MemmapConverter, HDF5Converter), preprocessing is required to ensure that all images have the same size and mode (e.g., RGB).

In this case, it is mandatory to apply the provided ResizeAndConvert transformation:

```python
from utils.utils import ResizeAndConvert

transform = ResizeAndConvert(size=(224, 224), mode="RGB")
```
This guarantees that the dataset is compatible with fixed-size storage formats.

### Fixed-Size Datasets

#### Example: Using `MemmapConverter`

```python
from datasets.memmap.memmap_converter import MemmapConverter

# Initialize the converter
converter = MemmapConverter(
    input_data="path/to/your/dataset",
    output_path="path/to/output",
    images_per_file=1000,
    batch_size=100,
    num_workers=4,
    shape=(224, 224),
    im_mode="RGB"
)

# Convert the dataset
converter.convert()
```

**Notes**:
- Ensure the batch size divides evenly into the number of images per file.

#### Example: Using a Torch Dataset

```python
from datasets.memmap.memmap_converter import MemmapConverter
from utils.utils import ResizeAndConvert
from torchvision.datasets import MNIST

# Prepare the dataset
dataset = MNIST(
    root="path/to/mnist",
    train=True,
    download=True,
    transform=ResizeAndConvert((224, 224), "RGB")
)

# Initialize the converter
converter = MemmapConverter(
    input_data=dataset,
    output_path="path/to/output",
    images_per_file=1000,
    batch_size=100,
    num_workers=4,
    shape=(224, 224),
    im_mode="RGB"
)

# Convert the dataset
converter.convert()
```
---

### Flexible-Size Datasets

#### Example: Using `IpcConverter`

```python
from datasets.pyarrow_ipc.pyarrow_ipc_converter import IpcConverter

# Initialize the converter
converter = IpcConverter(
    input_data="path/to/your/dataset",
    output_path="path/to/output",
    images_per_file=1000,
    batch_size=32,
    num_workers=4
)

# Convert the dataset
converter.convert()
```

**Note**: For flexible-size datasets, image size and type are automatically inferred from the original dataset.

---

### Reading Converted Datasets

#### Example: Using `MemmapReader`

```python
from datasets.memmap.memmap_reader import MemmapReader
from torch.utils.data import DataLoader
from torchvision import transforms

# Define transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Initialize the reader
reader = MemmapReader(dataset_path="path/to/converted/dataset")

# Create a DataLoader
dataloader = DataLoader(
    reader,
    batch_size=32,
    num_workers=4
)

# Iterate through the dataset
for images, labels in dataloader:
    # Your processing code here
    pass
```

**Note**: On Windows, use the `worker_init_fn` method to handle file handler pickling:
```python
dataloader = DataLoader(
    reader,
    batch_size=32,
    num_workers=4,
    worker_init_fn=MemmapReader.worker_init_fn
)
```

---

### Distributed Conversion

#### Python Script

```python
import torch.distributed as dist

def main():
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NPROCS'])
    
    dist.init_process_group('nccl', world_size=world_size, rank=rank)
    # Initialize the MemmapReader with distributed support
    dataset = MemmapConverter(dataset_path="path/to/dataset", dist=True, **kwargs)
```

#### Batch Script

```bash
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:0

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_ADDR="${MASTER_ADDR}i"
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
export MASTER_PORT=6000
```

---

## Extending the Library

To add support for new formats:
1. Subclass `ConverterFixedSize` or `ConverterFlexibleSize`.
2. Implement the `_custom_collate_fn` method for preprocessing batches.
3. Define methods for saving data to disk (`_write_data_to_disk`) or initializing arrays (`_init_arrays`).

### Example: Extention to Dense Segmentation Labels

To support datasets with dense segmentation labels (e.g., per-pixel class annotations), we provide an example extension of the library.
We created two new subclasses:
- FixedSizeConverter:
  - DenseMemmapConverter

  - DenseMemmapReader
- FlexibleSizeConverter
  - DenseIpcConverter

  - DenseIpcReader
  
These classes are adapted versions of the standard MemmapConverter and MemmapReader, with the following changes:

Label Shape:
- Instead of storing labels as single integers (shape: (num_instances,)), dense labels are stored as arrays with shape (num_instances, height, width, channels).

Target Transform Support:
- A target_transform can be applied to the labels during loading to handle preprocessing (e.g., resizing, normalization).

---

## Usage Examples

Refer to the `examples` directory for detailed use cases:
- **Caltech256**: Local testing.
- **ImageNet1k**: Cluster environments.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

