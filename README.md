
## Install

### Anaconda

1. [Python 3.10](https://www.python.org/downloads/release/python-31011/)
2. [Anaconda](https://www.anaconda.com/products/distribution)
3. [Cuda](https://developer.nvidia.com/cuda-downloads)

### Virtual Environment

### CPU-Only (Windows)

```shell
conda create -n ml-cpu python=3.10
conda activate ml-cpu

pip install torch torchvision torchaudio torchdata transformers[torch]
pip install numpy matplotlib openpyxl notebook
conda install pandas 
```

### GPU (Mac >= M1)

```shell
conda create -n ml-gpu python=3.10
conda activate ml-gpu

pip install torch torchvision torchaudio torchdata transformers
pip install numpy matplotlib openpyxl notebook
conda install pandas 
```
