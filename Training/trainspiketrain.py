
import sys
import os
import tonic
from tonic import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import snntorch as snn
from tqdm import tqdm
from datetime import datetime


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# yeah the name is def not confusing - trust


from Models.snn_baseline import SNNModel

debug = True
plot = True
epochs = 4

# Translate to frame

transform = transforms.ToFrame(
    sensor_size=(34, 34, 2),
    n_time_bins=50
)

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

if(debug):
    print("Begin training")
sum_loss = 0
sum_correct = 0
sum_total = 0
model.train()

for ep in enumerate(tqdm(range(1,epochs+1), desc=f"Training with {epochs} epochs")):
    now = datetime.now()
    checkpoint_path = f"ModelCheckpoints/SpikeTrain/checkpoint_epoch_{ep}_{now.strftime('%Y%m%d_%H%M%S')}.pt"

    for curr_batch, (frames, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {ep}"), start=1):
        # call me thomas the way i be training

        # get the info and init
        frames = frames.float()
        optimizer.zero_grad()

        # Pass them frames and learn to listen
        spikes, stats = model(frames)
        logits = spikes.sum(dim=0) # say sum
        loss = loss_fun(logits,labels)

        # Run it back (three steps this time)
        loss.backward()
        optimizer.step()

        # current metrics and such
        predictions = logits.argmax(dim=1)
        correct = (predictions == labels).sum().item()
        accuracy = correct / labels.size(0)

        # get running metrics
        sum_loss += loss.item()
        sum_correct += correct
        sum_total += labels.size(0) # convenient
        epoch_accuracy = sum_correct / sum_total

        if(debug and curr_batch % 100 == 0):
            print("\nTrain stats")
            print(f"\tLoss: {loss.item():.4f} | Acc: {epoch_accuracy*100:.3f}%")
            print(f"\tOutput: {spikes.shape}")
            print("\tFirst layer firing rate:", stats["layer1fr"].item()*100, '%')
            print("\tSecond layer firing rate:", stats["layer2fr"].item()*100, '%')
            print("\tOutput layer firing rate:", stats["outputfr"].item()*100, '%')
            print("\tBin count", torch.bincount(predictions, minlength=10), flush=True) # debug in case we are only guessing like 0 or smth

    if(debug):
        print("Saving checkpoint with:")
        print(f"\tLoss: {loss.item():.4f} | Acc: {epoch_accuracy*100:.3f}%")
        print(f"\tOutput: {spikes.shape}")
        print("\tFirst layer firing rate:", stats["layer1fr"].item()*100, '%')
        print("\tSecond layer firing rate:", stats["layer2fr"].item()*100, '%')
        print("\tOutput layer firing rate:", stats["outputfr"].item()*100, '%')
        print("\tBin count", torch.bincount(predictions, minlength=10), flush=True) # debug in case we are only guessing like 0 or smth


    checkpoint = {
        'epoch': ep,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)


    
    


