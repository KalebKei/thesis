import tonic
from tonic import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import numpy as np
import sys 
import argparse

import nmnistencodings as encodings



### Command line arguments
debug = False
plot = False
encoding = ""
checkpoint_file = ""

# setup args
parser = argparse.ArgumentParser(description="Training for NMNIST SNN models.")

parser.add_argument("encoding", type=int, help="Encoding type. 0: spiketrain, 1: voxel grids, 2: DCT, 3: truncated DCT, 4: aggressive DCT")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output")
parser.add_argument("-p", "--plot", action="store_true", help="Enable plotting after training")

# get args
args = parser.parse_args()

# parse args

# Encodings
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

# Debug and plotting
if(args.debug == True):
    debug = True
if(args.plot == True):
    plot = True

# Translate to frame

tonic.datasets.cifar10dvs.CIFAR10DVS.url = "https://figshare.com/ndownloader/files/38023437"

# Load the dataset and encode
raw_dataset = tonic.datasets.CIFAR10DVS(
    save_to="../Datasets/CIFAR10DVS/",
)

train_size = int(0.8 * len(raw_dataset))
test_size = len(raw_dataset) - train_size

# 3. Randomly split the dataset
train_dataset, test_dataset = random_split(raw_dataset, [train_size, test_size])

# Let's do some debug output
if(debug): 

    print("Dataset information")
    frames, label = train_dataset[0]
    print("dtype:", frames.dtype)
    print("shape:", frames.shape)
    print("min:", frames.min())
    print("max:", frames.max())
    print("nonzero:", (frames > 0).sum())

    vals = np.unique(frames)

    print(vals[:20])
    print("count:", len(vals))
    


if plot:
    print("Generating images")
    for t in range(10):
        plt.figure()
        plt.imshow(frames[t,0])
        plt.title(f"Timestep {t}")
        # plt.show()
        plt.savefig(f"../Plots/NMNIST/{checkpoint_file}/{encoding}_timestep{t}")
    summed = frames.sum(axis=0)

    plt.figure()
    plt.imshow(summed[0] + summed[1])
    plt.title("Summed Digit")
    plt.savefig(f"../Plots/NMNIST/{checkpoint_file}/{encoding}_summed_digit")

    print(f"Saving plots to ../Plots/NMNIST/{checkpoint_file}/")

print(f"Encoding: {encoding}")
print(f"Shape = {train_dataset[0][0].shape}")
print("Window = 1000 μs")

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# Debug 
if(debug):
    frames, labels = next(iter(train_loader))

    print("Train dataloader debug")
    print(f"\tFrame shape: {frames.shape}")
    print(f"\tFrame dtype: {frames.dtype}")
    print(f"\tFrame min: {frames.min()}")
    print(f"\tFrame max: {frames.max()}")
