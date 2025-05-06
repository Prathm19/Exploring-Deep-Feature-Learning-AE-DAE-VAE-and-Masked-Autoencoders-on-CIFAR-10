import torch

# Check if CUDA (GPU) is available
print("CUDA Available:", torch.cuda.is_available())

# If available, print GPU name
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
