import tonic
from tonic import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


debug = True
plot = True


# Translate to frame

transform = transforms.ToFrame(
    sensor_size=(34, 34, 2),
    n_time_bins=50
)

# Load the dataset and encode
raw_dataset = tonic.datasets.NMNIST(
    save_to="../Datasets/NMNIST/",
    train=True
)

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



# Let's do some debug output
if(debug): 

    print("Dataset information")
    frames, label = train_dataset[0]
    print("dtype:", frames.dtype)
    print("shape:", frames.shape)
    print("min:", frames.min())
    print("max:", frames.max())
    print("nonzero:", (frames > 0).sum())


if plot:
    print("Generating images")
    for t in range(10):
        plt.figure()
        plt.imshow(frames[t,0])
        plt.title(f"Timestep {t}")
        # plt.show()
        plt.savefig(f"../Plots/SpikeTrain/SpikeTrainTimestep{t}")
    summed = frames.sum(axis=0)

    plt.figure()
    plt.imshow(summed[0] + summed[1])
    plt.title("Summed Digit")
    plt.savefig(f"../Plots/SpikeTrain/SummedDigit")

print("Encoding A:")
print("Temporal Spike Frames")
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


