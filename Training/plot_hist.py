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
import trainhelpers as th
import Models.snn_baseline as models
import Models.snn_wavelet as wavelet_models

import Encodings.gestureencodings as encodings

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
parser.add_argument("encoding", type=int, help="Encoding type. 0: spiketrain, 1: voxel grids, 2: DCT, 3: truncated DCT, 4: aggressive DCT")
parser.add_argument("model", type=int, help="Model type for training. 0: traditional snn, 1: front-end 2d haar wavelet snn")
parser.add_argument("model_hist_filename", default="", help="Filename of the model's history to validate.")
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

if(args.encoding > 4):
    sys.exit(f"Error, incorrect encoding type: {args.encoding}")
if(args.encoding == 0): 
    transform = encodings.spiketrain_transform
    encoding = "spike_train"
    checkpoint_file = "SpikeTrain"
elif(args.encoding == 1):
    transform = encodings.voxel_grids_transform
    encoding = "voxel_grid"
    checkpoint_file = "VoxelGrids"
elif(args.encoding == 2):
    transform = encodings.dct_transform
    encoding = "dct"
    checkpoint_file = "DCT"
elif(args.encoding == 3):
    transform = encodings.truncated_dct_transform
    encoding = "trunc_dct"
    checkpoint_file = "TruncatedDCT"
elif(args.encoding == 4):
    transform = encodings.interpolated_voxel_grid_transform
    encoding = "interp_voxel_grid"
    checkpoint_file = "InterpolatedVoxelGrids"

# Model type
if(args.model > 1):
    sys.exit(f"Error, incorrect model type: {args.model}")
if(args.model == 0):
    model_type = "SNN"
    # match dataset:
    #     case "NMNIST": 
    #         model = models.SNNModel
    #     case "DVSGesture": 
    #         model = models.SNNModel_Gesture
elif(args.model == 1):
    model_type = "FrontEndWaveletSNN"
    # match dataset:
    #     case "NMNIST": 
    #         model = wavelet_models.WaveletModel
    #     case "DVSGesture": 
    #         model = wavelet_models.WaveletModel_Gesture

# Model training continuation
if(args.model_hist_filename != ""):
    hist_file_path = Path(args.model_hist_filename)
    if not hist_file_path.is_file():
        sys.exit(f"Model history file path {args.model_hist_filename} does not exist.")

    # checkpoint = torch.load(args.model_filename, map_location=torch.device('cpu'), weights_only=True)
    # model.load_state_dict(checkpoint['model_state_dict'])
    history = th.load_hist(args.model_hist_filename, model_type=model_type, dataset=dataset)
else:
    sys.exit(f"Error, pretrained model required for plotting: {args.model_hist_filename}")

filename = f"{dataset}_{model_type}"

th.plot_hist(history=history, epochs=len(history["train_loss"]), filename_ext=filename)