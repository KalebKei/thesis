import sys
import os
import tonic
from tonic import transforms
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

# required for the transform
import numpy as np
from scipy.fft import dct

# import snntorch as snn
# from tqdm import tqdm
# from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# yeah the name is def not confusing - trust

# custom funcs
import trainhelpers as th
from Models.snn_baseline import SNNModel


# these are args and such
debug = True
plot = True
epochs = 4


# Translate to custom dct transform
# encoding
class DCT:
    def __init__(self, keep_coeffs=None):
        self.keep_coeffs = keep_coeffs

    def __call__(self, frames):

        # DCT along temporal dimension
        coeffs = dct(
            frames,
            axis=0,
            norm='ortho'
        )

        # Optional coefficient truncation
        if self.keep_coeffs is not None:
            coeffs[self.keep_coeffs:, :, :, :] = 0

        return coeffs.astype(np.float32)

# Translate to frame using new encoding class
transform = transforms.Compose([
    transforms.ToFrame(
        sensor_size=(34, 34, 2),
        n_time_bins=50
    ),
    DCT()
])

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
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


model = SNNModel()


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

history = th.train(model=model, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, loss_fun=loss_fun, epochs=epochs, checkpoint_dir="ModelCheckpoints/DCT", encoding="dct", debug=True)


th.plot_hist(history=history, epochs=epochs)
