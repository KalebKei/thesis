import sys
import argparse
import os
import tonic
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
from pathlib import Path


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
batch_size = 128
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
model_type = []
models = []
model_type.append("SNN")
match dataset:
    case "NMNIST": 
        models.append(snn_models.SNNModel())
    case "DVSGesture": 
        models.append(snn_models.SNNModel_Gesture())
model_type.append("FrontEndWaveletSNN")
match dataset:
    case "NMNIST":
        models.append(wavelet_models.WaveletModel())
    case "DVSGesture": 
        models.append(wavelet_models.WaveletModel_Gesture())
model_type.append("IntegratedWaveletSNN")
match dataset:
    case "NMNIST":
        models.append(integrated_wavelet_models.IntegratedWaveletBlockModel_Gesture())
    case "DVSGesture": 
        models.append(integrated_wavelet_models.IntegratedWaveletBlockModel_Gesture())


i = 0
for model in models:
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"Trainable parameters for {model_type[i]} model: {trainable_params:,}")
    i += 1