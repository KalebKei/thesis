import gc
import sys
import argparse
import os
import tonic
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
from pathlib import Path

import nmnistencodings
import gestureencodings



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# yeah the name is def not confusing - trust

# custom funcs
import Models.snn_baseline as snn_models
import Models.snn_wavelet as wavelet_models
import Models.snn_wavelet_integrated as integrated_wavelet_models

### Command line arguments
debug = False
plot = False
encoding = ""
model_type = ""
dataset = ""
# setup args
parser = argparse.ArgumentParser(description="Plotting for pretrained models.")

parser.add_argument("dataset", type=int, help="Datasets. 0: NMNIST, 1: DVSGesture.")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output")


# get args
args = parser.parse_args()

# Debug
if(args.debug == True):
    debug = True

if(args.dataset == 0):
    dataset = "NMNIST"
elif(args.dataset == 1):
    dataset = "DVSGesture"
else:
    sys.exit(f"Error, incorrect dataset type: {args.dataset}")

# Model type
encodings = []
encoding_type = ["Binary Event Frames", "Event Frames", "DCT", "Truncated DCT", "Voxel Grids"]
match dataset:
    case "NMNIST": 
        encodings = [nmnistencodings.spiketrain_transform, nmnistencodings.voxel_grids_transform, nmnistencodings.dct_transform, nmnistencodings.truncated_dct_transform, nmnistencodings.interpolated_voxel_grid_transform]
    case "DVSGesture": 
        encodings = [gestureencodings.spiketrain_transform, gestureencodings.voxel_grids_transform, gestureencodings.dct_transform, gestureencodings.truncated_dct_transform, gestureencodings.interpolated_voxel_grid_transform]


i = 0
for encoding in encodings:
    match dataset:
        case "NMNIST":
            train_dataset = tonic.datasets.NMNIST(
                save_to="../Datasets/NMNIST/", train=True, transform=encoding
            )
            batch_size = 650
        case "DVSGesture":
            train_dataset = tonic.datasets.DVSGesture(
                save_to="../Datasets/DVSGESTURE/", train=True, transform=encoding
            )
            batch_size = 40
    # Pull out a single uncollated sample directly from the dataset
    raw_sample, label = train_dataset[0] 

    # 1. Inspect the true structural dimensions
    print(f"--- {encoding_type[i]} Structural Profile ---")
    print(f"Tensor Shape: {list(raw_sample.shape)}")
    print(f"Data Type:    {raw_sample.dtype}")

    # 2. Compute individual sample bytes
    sample_bytes = raw_sample.size * raw_sample.itemsize if hasattr(raw_sample, 'size') else raw_sample.nelement() * raw_sample.element_size()
    print(f"Single Sample Size: {sample_bytes / 1024:.2f} KB\n")

            
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    batch_iter = iter(train_loader)
    batch, _ = next(batch_iter)
    batch = batch.float()

    # Calculate CPU tensor size
    cpu_data_bytes = batch.nelement() * batch.element_size()
    cpu_mb = cpu_data_bytes / (1024 ** 2)

    # --- FIX 1: Clean slate before measuring ---
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize() 

    # Measure baseline VRAM
    vram_before = torch.cuda.memory_allocated() / (1024 ** 2)

    # Transfer to GPU
    images_gpu = batch.to('cuda')

    # --- FIX 2: Wait for GPU transfer to completely finish ---
    torch.cuda.synchronize() 

    # Measure final VRAM
    vram_after = torch.cuda.memory_allocated() / (1024 ** 2)
    vram_delta = vram_after - vram_before

    print(f"Encoding {encoding_type[i]} calculated CPU size: {cpu_mb:.2f} MB")
    print(f"Encoding {encoding_type[i]} actual GPU size: {vram_delta:.2f} MB\n")
    
    # --- FIX 3: Explicitly delete the GPU tensor so it doesn't leak into iteration i+1 ---
    del images_gpu
    
    i += 1
