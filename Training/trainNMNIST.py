import sys
import argparse
import os
import tonic
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from pathlib import Path


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# yeah the name is def not confusing - trust

# custom funcs
import trainhelpers as th
from Models.snn_baseline import SNNModel
from Models.snn_wavelet import WaveletModel

import Encodings.nmnistencodings as encodings

### Command line arguments
debug = False
plot = False
epochs = 4
batch_size = 32
encoding = ""
checkpoint_file = ""
model_type = ""
# setup args
parser = argparse.ArgumentParser(description="Training for NMNIST SNN models.")

parser.add_argument("encoding", type=int, help="Encoding type. 0: spiketrain, 1: voxel grids, 2: DCT, 3: truncated DCT, 4: aggressive DCT")
parser.add_argument("model", type=int, help="Model type for training. 0: traditional snn, 1: front-end 2d haar wavelet snn")
parser.add_argument("epochs", type=int, default=4, help="Number of epochs for training.")
parser.add_argument("-bs", "--batch_size", type=int, default=batch_size, help="Batch size for training.")
parser.add_argument("-mf", "--model_filename", default="", help="Filename for model to continue training. Optional")
parser.add_argument("-mhf", "--model_hist_filename", default="", help="Filename of the model's history to continue training. Optional")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output")
parser.add_argument("-p", "--plot", action="store_true", help="Enable plotting after training")

# get args
args = parser.parse_args()

# parse args

# Encodings
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
    transform = encodings.aggressive_dct_transform
    encoding = "aggr_dct"
    checkpoint_file = "AggressiveDCT"

# Model type
if(args.model > 1):
    sys.exit(f"Error, incorrect model type: {args.model}")
if(args.model == 0):
    model = SNNModel()
    model_type = "SNN"
elif(args.model == 1):
    model = WaveletModel()
    model_type = "FrontEndWaveletSNN"

# Epochs and batch size
epochs = args.epochs
batch_size = args.batch_size

# Model training continuation
if(args.model_filename != ""):
    file_path = Path(args.model_filename)
    if not file_path.is_file():
        sys.exit(f"Model file path {args.model_filename} does not exist.")
    hist_file_path = Path(args.model_filename)
    if not hist_file_path.is_file():
        sys.exit(f"Model history file path {args.model_hist_filename} does not exist.")

    checkpoint = torch.load(args.model_filename, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    history = th.load_hist(args.model_hist_filename)
else:
    history = None

# Debug and plotting
if(args.debug == True):
    debug = True
if(args.plot == True):
    plot = True


# Load the dataset and encode

train_dataset = tonic.datasets.NMNIST(
    save_to="../Datasets/NMNIST/",
    train=True,
    transform=transform
)

test_dataset = tonic.datasets.NMNIST(
    save_to="../Datasets/NMNIST/",
    train=False,
    transform=transform
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


if(debug):
    print("Testing one forward pass of model")
    frames, labels = next(iter(train_loader))

    print(f"Input: {frames.shape}")
    frames = frames.float()

    output, spikes_count = model(frames)

    print("Successful pass")
    print(f"\tOutput: {output.shape}")
    print("\tFirst layer firing rate:", spikes_count["layer1fr"].item()*100, '%')
    print("\tSecond layer firing rate:", spikes_count["layer2fr"].item()*100, '%')
    print("\tOutput layer firing rate:", spikes_count["outputfr"].item()*100, '%')


# Now time for some train time

loss_fun = nn.CrossEntropyLoss() # #nofun

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3 # i never know what to put this guy at
)

history = th.train(model=model, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, loss_fun=loss_fun, epochs=epochs, checkpoint_dir=f"ModelCheckpoints/NMNIST/{model_type}/{checkpoint_file}", encoding=encoding, model_type=model_type, history=history, debug=debug)

if(plot):
    th.plot_hist(history=history, epochs=epochs)
