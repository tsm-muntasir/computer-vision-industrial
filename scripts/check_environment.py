import platform
import sys

import matplotlib
import numpy as np
import pandas as pd
import PIL
import sklearn
import torch
import torchvision

print("=" * 55)
print("COMPUTER VISION PROJECT: ENVIRONMENT CHECK")
print("=" * 55)

print(f"Python:       {sys.version.split()[0]}")
print(f"Platform:     {platform.platform()}")
print(f"PyTorch:      {torch.__version__}")
print(f"TorchVision:  {torchvision.__version__}")
print(f"NumPy:        {np.__version__}")
print(f"Pandas:       {pd.__version__}")
print(f"Matplotlib:   {matplotlib.__version__}")
print(f"Pillow:       {PIL.__version__}")
print(f"scikit-learn: {sklearn.__version__}")

print(f"\nCUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU. This is fine for Phase 1.")

# Test a basic tensor operation.
x = torch.tensor([1.0, 2.0, 3.0])
y = x * 2

print(f"\nInput tensor:  {x}")
print(f"Output tensor: {y}")
print("\nEnvironment check completed.")