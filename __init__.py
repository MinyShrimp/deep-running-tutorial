import os
import sys
import torch

sys.path.append(os.getcwd())

if not torch.backends.mps.is_built():
    print("MPS is not built")
    sys.exit(1)

if not torch.backends.mps.is_available():
    print("MPS is not available")
    sys.exit(1)
