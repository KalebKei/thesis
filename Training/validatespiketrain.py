import sys
import os
import tonic
from tonic import transforms
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
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
skip_train = True
model_path = "ModelCheckpoints/VoxelGrids/20260618_133607_checkpoint_epoch_1.pt"
hist_path = "ModelCheckpoints/VoxelGrids/20260618_133607_hist_checkpoint_epoch_1.pt"

# Translate to frame
# encoding
transform = transforms.Compose([
    transforms.ToFrame(
        sensor_size=(34,34,2),
        n_time_bins=50
    ),
    lambda x: (x > 0).astype(np.float32)
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

# load pretrained model
model = SNNModel()
checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# load model hist
history = th.load_hist(filename=hist_path)
# update epochs based on hist
epochs = len(history["layer1_fr"])


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


# Now time for some validate time

loss_fun = nn.CrossEntropyLoss() # #nofun

validation_metrics = {
    "loss": [],
    "acc": [],
    "layer1_fr": [],
    "layer2_fr": [],
    "output_fr": []
}
validation_metrics = th.validate(model=model, val_loader=test_loader, loss_fun=loss_fun, debug=debug)

print(f"Validation metrics for model with {epochs} epochs:")
print("\tLoss: ", validation_metrics["loss"])
print("\tAccuracy: ", validation_metrics["acc"])
print("\tLayer 1 firing rate: ", validation_metrics["layer1_fr"])
print("\tLayer 2 firing rate: ", validation_metrics["layer2_fr"])
print("\tOutput firing rate: ", validation_metrics["output_fr"])

# show charts again for debug purposes
if(debug):
    print("Printing charts again for debug purposes")
    th.plot_hist(history=history, epochs=epochs, filename_ext="val_")


