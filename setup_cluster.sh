ml Stages/2025  
ml  GCC/13.3.0
ml OpenMPI/5.0.5
ml mpi4py/4.0.1
ml CUDA/12
ml cuDNN/9.5.0.50-CUDA-12
ml Python/3.12.3

python -m venv dataloadenv
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

