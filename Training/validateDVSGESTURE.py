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
from Models.snn_baseline import SNNModel_Gesture
from Models.snn_wavelet import WaveletModel_Gesture

import Encodings.gestureencodings as encodings

### Command line arguments
debug = False
plot = False
encoding = ""
batch_size = 32
model_type = ""
# setup args
parser = argparse.ArgumentParser(description="Training for NMNIST SNN models.")

parser.add_argument("encoding", type=int, help="Encoding type. 0: spiketrain, 1: voxel grids, 2: DCT, 3: truncated DCT, 4: aggressive DCT")
parser.add_argument("model", type=int, help="Model type for training. 0: traditional snn, 1: front-end 2d haar wavelet snn")
parser.add_argument("model_filename", default="", help="Filename for model to validate.")
parser.add_argument("model_hist_filename", default="", help="Filename of the model's history to validate.")
parser.add_argument("-bs", "--batch_size", type=int, default=32, help="Batch size used during training model.")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output")
parser.add_argument("-p", "--plot", action="store_true", help="Enable plotting after training")
parser.add_argument("-g", "--gpu", action="store_true", help="Enable gpu acceleration")



# get args
args = parser.parse_args()

# parse args

# Encodings
if(args.encoding > 4):
    sys.exit(f"Error, incorrect encoding type: {args.encoding}")
if(args.encoding == 0): 
    transform = encodings.spiketrain_transform
    encoding = "spike_train"
elif(args.encoding == 1):
    transform = encodings.voxel_grids_transform
    encoding = "voxel_grid"
elif(args.encoding == 2):
    transform = encodings.dct_transform
    encoding = "dct"
elif(args.encoding == 3):
    transform = encodings.truncated_dct_transform
    encoding = "trunc_dct"
elif(args.encoding == 4):
    transform = encodings.aggressive_dct_transform
    encoding = "aggr_dct"

# Model type
if(args.model > 1):
    sys.exit(f"Error, incorrect model type: {args.model}")
if(args.model == 0):
    model = SNNModel_Gesture()
    model_type = "SNN"
elif(args.model == 1):
    model = WaveletModel_Gesture()
    model_type = "FrontEndWaveletSNN"

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
    history = th.load_hist(args.model_hist_filename, model_type)

# Debug and plotting
if(args.debug == True):
    debug = True
if(args.plot == True):
    plot = True
if(args.gpu == True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(debug):
        print(f"Current training device: {device}")
    model = model.to(device)

model.eval()
# Load the dataset and encode

tonic.datasets.cifar10dvs.CIFAR10DVS.url = "https://figshare.com/ndownloader/files/38023437"

# Load the dataset and encode

train_dataset = tonic.datasets.DVSGesture(
    save_to="../Datasets/DVSGESTURE/",
    train=True,
    transform=transform
)

test_dataset = tonic.datasets.DVSGesture(
    save_to="../Datasets/DVSGESTURE/",
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


# update epochs based on hist
epochs = len(history["layer1_fr"])


if(debug):
    print(f"Testing one forward pass of the {model_type} model on {encoding} encoding type")
    frames, labels = next(iter(train_loader))

    print(f"Input: {frames.shape}")
    frames = frames.float()

    output, spikes_count = model(frames)

    print("Successful pass")
    print(f"\tOutput: {output.shape}")
    print("\tFirst layer firing rate:", spikes_count["layer1fr"].item()*100, '%')
    print("\tSecond layer firing rate:", spikes_count["layer2fr"].item()*100, '%')
    print("\tOutput layer firing rate:", spikes_count["outputfr"].item()*100, '%')


# Now time for some validate time

loss_fun = nn.CrossEntropyLoss() # #nofun

validation_metrics = {
    "loss": [],
    "acc": [],
    "layer1_fr": [],
    "layer2_fr": [],
    "output_fr": []
}
validation_metrics = th.validate(model=model, val_loader=test_loader, loss_fun=loss_fun, debug=debug, model_type=model_type, device=device)

print(f"Validation metrics for model with {epochs} epochs:")
print("\tLoss: ", validation_metrics["loss"])
print("\tAccuracy: ", validation_metrics["acc"])
print("\tLayer 1 firing rate: ", validation_metrics["layer1_fr"])
print("\tLayer 2 firing rate: ", validation_metrics["layer2_fr"])
print("\tOutput firing rate: ", validation_metrics["output_fr"])

# show charts again for debug purposes
if(plot):
    print("Printing charts")
    th.plot_hist(history=history, epochs=epochs, filename_ext="val_")


