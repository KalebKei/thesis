import tonic
from tonic import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from scipy.fft import dct
from scipy.fft import idct


debug = True
plot = True

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
    DCT(keep_coeffs=25)
])

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

    vals = np.unique(frames)

    print(vals[:20])
    print("count:", len(vals))
    


if plot:
    print("Generating images")
    for t in range(10):
        plt.figure()
        plt.imshow(frames[t,0], cmap='gray')
        plt.colorbar()
        plt.title(f"Timestep {t}")
        # plt.show()
        plt.savefig(f"../Plots/TruncatedDCT/TruncDCTTimestep{t}")
    summed = frames.sum(axis=0)

    plt.figure()
    plt.imshow(summed[0] + summed[1])
    plt.colorbar()
    plt.title("Summed Digit")
    plt.savefig(f"../Plots/TruncatedDCT/SummedDigit")

print("Encoding D:")
print("Truncated Temporal DCT")
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

# sanity check; rebuild and check if error is close to 0 as intended

reconstructed = idct(
    frames,
    axis=0,
    norm='ortho'
)

original_frames, _ = train_dataset[0]

error = np.mean(np.abs(
    reconstructed - original_frames
))

print("Reconstruction Error:", error)