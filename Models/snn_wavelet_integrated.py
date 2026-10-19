import math
import torch
from torch import nn
import torch.nn.functional as F
from Models.snn_wavelet import Haar2DTransform


class SurrogateStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage):
        ctx.save_for_backward(voltage)
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, gradient):
        voltage, = ctx.saved_tensors
        return gradient / (1 + 5.0 * voltage.abs()).square()


class LIFNeuron(nn.Module):
    """Threshold=1; immediate reset=0; detached reset; slope=5.
    tau=2: voltage = 0.5*membrane + 0.5*current (not default snn.Leaky).
    State persists over time within a call, and resets between calls.
    """
    def __init__(self, tau=2.0):
        super().__init__()
        if not math.isfinite(tau) or tau < 1:
            raise ValueError("tau must be finite and >= 1")
        self.tau = tau

    def forward(self, currents):
        membrane = torch.zeros_like(currents[:, 0])
        spike_sequence = []
        for step in range(currents.shape[1]):
            voltage = membrane + (currents[:, step] - membrane) / self.tau
            spikes = SurrogateStep.apply(voltage - 1.0)
            membrane = voltage * (1 - spikes.detach())
            spike_sequence.append(spikes)
        return torch.stack(spike_sequence, dim=1)


def apply_frames(layer, sequence):
    """Apply spatial layers to [B*T,C,H,W]. BN includes all time bins."""
    batch, time = sequence.shape[:2]
    frames = layer(sequence.flatten(0, 1))
    return frames.reshape(batch, time, *frames.shape[1:])


def record_spikes(records, name, spikes):
    records[name] = (spikes.detach().float().sum(), spikes.numel())


class WaveletBlock(nn.Module):
    """WSC spatial residual followed by the two-stage channel residual."""
    def __init__(self, channels, haar_factory=Haar2DTransform, tau=2.0):
        super().__init__()
        self.haar = haar_factory()
        if tuple(self.haar.haar_kernels.shape) != (4, 1, 2, 2):
            raise ValueError("Haar kernels must have shape [4,1,2,2]")

        # Depthwise WSC: one learned kernel per feature channel/subband.
        self.base_lif = LIFNeuron(tau)
        self.base_conv = nn.Conv2d(channels, channels, 5, padding=2,
                                   groups=channels, bias=False)
        self.band_lif = LIFNeuron(tau)
        self.band_conv = nn.Conv2d(4*channels, 4*channels, 5, padding=2,
                                   groups=4*channels, bias=False)
        self.spatial_bn = nn.BatchNorm2d(channels)

        # Original compact channel-mixer default: expansion=1.
        self.lif1 = LIFNeuron(tau)
        self.conv1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.lif2 = LIFNeuron(tau)
        self.conv2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x, records):
        batch, time, channels, height, width = x.shape
        if min(height, width) < 2 or height % 2 or width % 2:
            raise ValueError("Wavelet features must have even H/W >= 2")

        # Base path: SN -> depthwise convolution.
        spikes = self.base_lif(x)
        record_spikes(records, "block1.spatial.base", spikes)
        base_features = apply_frames(self.base_conv, spikes)

        # Haar output: [ch0 LL,LH,HL,HH; ch1 LL,LH,HL,HH; ...].
        coefficients = apply_frames(self.haar, x)
        band_h, band_w = coefficients.shape[-2:]
        coefficients = coefficients.reshape(batch, time, channels, 4, band_h, band_w)

        # Band-major convolution layout: [all LL; all LH; all HL; all HH].
        bands = coefficients.permute(0, 1, 3, 2, 4, 5).reshape(
            batch, time, 4*channels, band_h, band_w)
        spikes = self.band_lif(bands)
        separated = spikes.reshape(batch, time, 4, channels, band_h, band_w)
        for index, name in enumerate(self.haar.subband_names):
            record_spikes(records, f"block1.spatial.level1.{name}", separated[:, :, index])
        processed = apply_frames(self.band_conv, spikes)

        # Inverse Haar: restore channel-major layout, then reconstruct.
        processed = processed.reshape(batch, time, 4, channels, band_h, band_w)
        synthesis_input = processed.permute(0, 1, 3, 2, 4, 5).reshape(
            batch*time*channels, 4, band_h, band_w)
        reconstructed = F.conv_transpose2d(synthesis_input, self.haar.haar_kernels, stride=2)
        reconstructed = reconstructed.reshape(batch, time, channels, height, width)

        # First residual: x + WSC(x).
        spatial_features = apply_frames(self.spatial_bn, base_features + reconstructed)
        x = x + spatial_features

        # Second residual: x + ChannelMixer(x).
        spikes = self.lif1(x)
        record_spikes(records, "block1.mixer.lif1", spikes)
        mixed = apply_frames(self.conv1, spikes)
        mixed = apply_frames(self.bn1, mixed)
        spikes = self.lif2(mixed)
        record_spikes(records, "block1.mixer.lif2", spikes)
        mixed = apply_frames(self.conv2, spikes)
        mixed = apply_frames(self.bn2, mixed)
        return x + mixed


class IntegratedWaveletBlockModel(nn.Module):
    def __init__(self, haar_factory = Haar2DTransform, num_classes=10, in_channels=2,
                 embedding_channels=(16, 24), tau=2.0, readout="spikes"):
        super().__init__()
        if readout not in ("logits", "spikes"):
            raise ValueError("readout must be 'logits' or 'spikes'")
        self.in_channels = in_channels
        self.readout = readout
        c1, c2 = embedding_channels

        self.conv1 = nn.Conv2d(in_channels, c1, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.lif1 = LIFNeuron(tau)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.lif2 = LIFNeuron(tau)
        self.pool2 = nn.MaxPool2d(2)

        # Continuous currents prevent consecutive LIFs seeing only 0/1 input.
        self.projection_conv = nn.Conv2d(c2, c2, 1, bias=False)
        self.projection_bn = nn.BatchNorm2d(c2)
        self.block = WaveletBlock(c2, haar_factory, tau)
        self.head_lif = LIFNeuron(tau)
        self.fc1 = nn.Linear(c2, num_classes)
        self.lif_out = LIFNeuron(tau) if readout == "spikes" else None

    def forward(self, frames):
        if frames.ndim != 5 or frames.shape[2] != self.in_channels:
            raise ValueError(
                f"Expected [B,T,{self.in_channels},H,W], "
                f"received {tuple(frames.shape)}"
            )

        if not frames.is_floating_point() or min(frames.shape) < 1:
            raise ValueError("Frames must be nonempty floating-point tensors")

        if min(frames.shape[-2:]) < 8:
            raise ValueError("Input is too small for the embedding stages")

        if self.lif_out is None:
            raise ValueError(
                "Use readout='spikes' for the original model's "
                "outputsp/outputfr history interface."
            )

        # Each entry is:
        # name -> (detached spike count, number of neuron updates)
        records = {}

        # First embedding stage
        x = apply_frames(self.conv1, frames)
        x = apply_frames(self.bn1, x)
        spikes = self.lif1(x)
        record_spikes(records, "embedding1", spikes)
        x = apply_frames(self.pool1, spikes)

        # Second embedding stage
        x = apply_frames(self.conv2, x)
        x = apply_frames(self.bn2, x)
        spikes = self.lif2(x)
        record_spikes(records, "embedding2", spikes)
        x = apply_frames(self.pool2, spikes)

        # Current projection and integrated wavelet block
        x = apply_frames(self.projection_conv, x)
        x = apply_frames(self.projection_bn, x)
        x = self.block(x, records)

        # Spiking head, spatial pooling, and output neurons
        spikes = self.head_lif(x)
        record_spikes(records, "head.pre_pool", spikes)

        currents = self.fc1(spikes.mean(dim=(-2, -1)))
        output_spikes = self.lif_out(currents)
        record_spikes(records, "head.output", output_spikes)

        # Diagnostics must not retain the training computation graph.
        counts = {
            name: count.detach().float()
            for name, (count, _) in records.items()
        }
        updates = {
            name: num_updates
            for name, (_, num_updates) in records.items()
        }
        rates = {
            name: counts[name] / updates[name]
            for name in records
        }

        total_spikes = torch.stack(list(counts.values())).sum()
        total_updates = sum(updates.values())

        stats = {
            # Original model's history interface:
            "layer1sp": counts["embedding1"],
            "layer2sp": counts["embedding2"],
            "outputsp": counts["head.output"],
            "layer1fr": rates["embedding1"],
            "layer2fr": rates["embedding2"],
            "outputfr": rates["head.output"],

            # Detailed diagnostics, including the wavelet block:
            "spike_counts": counts,
            "neuron_updates": updates,
            "firing_rates": rates,
            "total_spikes": total_spikes,
            "total_neuron_updates": total_updates,
            "overall_fire_rate": total_spikes / total_updates,
        }

        # Original prediction layout: [T, B, num_classes]
        return output_spikes.transpose(0, 1).contiguous(), stats

def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class IntegratedWaveletBlockModel_Gesture(nn.Module):
    def __init__(
        self,
        haar_factory=Haar2DTransform,
        num_classes=11,
        in_channels=2,
        tau=2.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        # Learned downsampling replaces the original input Haar reduction:
        # 128 -> 64
        self.conv1 = nn.Conv2d(
            in_channels, 16,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.lif1 = LIFNeuron(tau)
        self.pool1 = nn.MaxPool2d(2)  # 64 -> 32

        self.conv2 = nn.Conv2d(
            16, 32,
            kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.lif2 = LIFNeuron(tau)
        self.pool2 = nn.MaxPool2d(2)  # 32 -> 16

        # Convert pooled spikes into continuous currents for the block.
        self.projection_conv = nn.Conv2d(
            32, 32, kernel_size=1, bias=False
        )
        self.projection_bn = nn.BatchNorm2d(32)

        # Preserves [B, T, 32, 16, 16] for 128x128 input.
        self.block = WaveletBlock(32, haar_factory, tau)

        # Original Gesture model's third stage; no third max pool.
        self.conv3 = nn.Conv2d(
            32, 64,
            kernel_size=3, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(64)
        self.lif3 = LIFNeuron(tau)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, num_classes)
        self.lif_out = LIFNeuron(tau)

    def forward(self, frames):
        if frames.ndim != 5 or frames.shape[2] != self.in_channels:
            raise ValueError(
                f"Expected [B,T,{self.in_channels},H,W], "
                f"received {tuple(frames.shape)}"
            )

        if not frames.is_floating_point() or min(frames.shape) < 1:
            raise ValueError(
                "Frames must be nonempty floating-point tensors"
            )

        if min(frames.shape[-2:]) < 8:
            raise ValueError(
                "Input is too small for the embedding stages"
            )

        records = {}

        # First stage
        x = apply_frames(self.conv1, frames)
        x = apply_frames(self.bn1, x)
        spikes = self.lif1(x)
        record_spikes(records, "embedding1", spikes)
        x = apply_frames(self.pool1, spikes)

        # Second stage
        x = apply_frames(self.conv2, x)
        x = apply_frames(self.bn2, x)
        spikes = self.lif2(x)
        record_spikes(records, "embedding2", spikes)
        x = apply_frames(self.pool2, spikes)

        # Integrated wavelet processing
        x = apply_frames(self.projection_conv, x)
        x = apply_frames(self.projection_bn, x)
        x = self.block(x, records)

        # Third stage
        x = apply_frames(self.conv3, x)
        x = apply_frames(self.bn3, x)
        spikes = self.lif3(x)
        record_spikes(records, "embedding3", spikes)

        # Global average pooling and spiking classification
        x = apply_frames(self.avgpool, spikes)
        x = x.flatten(start_dim=2)  # [B, T, 64]

        currents = self.fc1(x)
        output_spikes = self.lif_out(currents)
        record_spikes(records, "head.output", output_spikes)

        # Detached diagnostics for this batch
        counts = {
            name: count.detach().float()
            for name, (count, _) in records.items()
        }
        updates = {
            name: num_updates
            for name, (_, num_updates) in records.items()
        }
        rates = {
            name: counts[name] / updates[name]
            for name in records
        }

        total_spikes = torch.stack(list(counts.values())).sum()
        total_updates = sum(updates.values())

        stats = {
            # Original Gesture history keys
            "layer1sp": counts["embedding1"],
            "layer2sp": counts["embedding2"],
            "layer3sp": counts["embedding3"],
            "outputsp": counts["head.output"],

            "layer1fr": rates["embedding1"],
            "layer2fr": rates["embedding2"],
            "layer3fr": rates["embedding3"],
            "outputfr": rates["head.output"],

            # Includes internal spikes recorded by WaveletBlock
            "spike_counts": counts,
            "neuron_updates": updates,
            "firing_rates": rates,
            "total_spikes": total_spikes,
            "total_neuron_updates": total_updates,
            "overall_fire_rate": total_spikes / total_updates,
        }

        return output_spikes.transpose(0, 1).contiguous(), stats