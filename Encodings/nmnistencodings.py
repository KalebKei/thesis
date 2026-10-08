from tonic import transforms
import numpy as np
from scipy.fft import dct


spiketrain_transform = transforms.Compose([
    transforms.ToFrame(
        sensor_size=(34,34,2),
        n_time_bins=50
    ),
    lambda x: (x > 0).astype(np.float32)
])

voxel_grids_transform = transforms.ToFrame(
    sensor_size=(34, 34, 2),
    n_time_bins=50
)

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
    
dct_transform = transforms.Compose([
    transforms.ToFrame(
        sensor_size=(34, 34, 2),
        n_time_bins=50
    ),
    DCT()
])
    
truncated_dct_transform = transforms.Compose([
    transforms.ToFrame(
        sensor_size=(34, 34, 2),
        n_time_bins=50
    ),
    DCT(keep_coeffs=25)
])

aggressive_dct_transform = transforms.Compose([
    transforms.ToFrame(
        sensor_size=(34, 34, 2),
        n_time_bins=50
    ),
    DCT(keep_coeffs=10)
])

class InterpolateVoxelPolarities:
    def __call__(self, grid):
        # from tonic output [T, 1, H, W]
        neg = np.maximum(-grid,0)
        pos = np.maximum(grid,0)

        # output [T,2,H,W] to match model input
        return np.concatenate((neg, pos), axis=1).astype(np.float32)

interpolated_voxel_grid_transform = transforms.Compose([
    transforms.ToVoxelGrid(
        sensor_size=sensor_size,
        n_time_bins=50,
    ),
    InterpolateVoxelPolarities(),
])