python3 -m venv dataloadenv
source dataloadenv/bin/activate
pip install --upgrade pip
pip install torch
pip install torchvision
pip install tqdm
pip install requests
pip install pyarrow
pip install h5py
pip install pillow
pip install nvidia-ml-py3
pip install pandas

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export PATH=$PATH:$CUDA_HOME/bin
