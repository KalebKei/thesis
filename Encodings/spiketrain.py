import tonic
from tonic import transforms
import matplotlib.pyplot as plt

debug = True


# Translate to frame
transform = transforms.ToFrame(
    sensor_size=(34, 34, 2),
    time_window=1000
)

# Load the dataset
raw_dataset = tonic.datasets.NMNIST(
    save_to="../Datasets/NMNIST/",
    train=True
)
frame_dataset = tonic.datasets.NMNIST(
    save_to="../Datasets/NMNIST/",
    train=True,
    transform=transform
)


# Let's do some debug output
if(debug): 

    frames, label = frame_dataset[0]
    print("dtype:", frames.dtype)
    print("shape:", frames.shape)
    print("min:", frames.min())
    print("max:", frames.max())
    print("nonzero:", (frames > 0).sum())



for t in range(10):
    plt.figure()
    plt.imshow(frames[t,0])
    plt.title(f"Timestep {t}")
    # plt.show()
    plt.savefig(f"../Plots/SpikeTrain/SpikeTrainTimestep{t}")
