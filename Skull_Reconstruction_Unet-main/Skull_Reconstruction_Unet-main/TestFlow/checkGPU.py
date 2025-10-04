import torch
print(torch.__version__)  # Check PyTorch version
print(torch.version.cuda) # Check CUDA version used by PyTorch
print(torch.cuda.is_available()) # Check if CUDA is detected

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

if cuda_available:
    print(f"CUDA is available. Your GPU is: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available.")
    