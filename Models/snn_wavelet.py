import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
import math

class Haar2DTransform(nn.Module):
    # Fixed 2D Haar wavelet transform
    # Input: (B,C,H,W)
    # Output: Dict[str, Tensor]
    # Each tensor: (B,2*C*4, H/2,W/2)
    def __init__(self):
        super().__init__()

        scale = 1 / math.sqrt(2)

        h0 = torch.tensor([scale, scale])
        h1 = torch.tensor([scale, -scale])

        filters = []

        # 2D Haar: LL, LH, HL, HH
        self.subband_names = ["LL", "LH", "HL", "HH"]

        for fy in [h0, h1]:
            for fx in [h0, h1]:
                kernel = fy[:, None] * fx[None, :]
                filters.append(kernel)

        # shape: (4, 1, 2, 2)
        kernels = torch.stack(filters).unsqueeze(1)

        self.register_buffer("haar_kernels", kernels)

    def forward(self, x):   
        """
        x: (B, C, H, W)
        """
        b, c, h, w = x.shape
        
        assert h % 2 == 0, "Height must be even"
        assert w % 2 == 0, "Width must be even"

        # merge batch and channels for grouped conv
        x = x.reshape(b * c, 1, h, w)

        out = F.conv2d(
            x,
            self.haar_kernels,
            stride=2
        )

        # (B*C, 4, H/2, W/2)
        out = out.reshape(b, c, 4, h // 2, w // 2)

        # merge polarity + subbands → channels
        out = out.reshape(b, c * 4, h // 2, w // 2)

        return out
    
# TODO rework for future 3DWSNet implementation
class Haar3DTransform(nn.Module):
    # Fixed 3D Haar wavelet transform
    # Input: (B,C,D,H,W)
    # Output: Dict[str, Tensor]
    # Each tensor: (B,C,D/2,H/2,W/2)

    def __init__(self):
        super().__init__()
        with torch.no_grad():
            # define high and low
            scale = 1 / math.sqrt(2)
            h0 = torch.tensor([scale, scale])
            h1 = torch.tensor([scale, -scale])

            # combine high and low into the 8 combinations for 3D
            filters = []
            self.subband_names = ['LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH']

            for fz in [h0,h1]:
                for fy in [h0,h1]:
                    for fx in [h0,h1]:
                        kernel = fz[:, None, None] * fy[None, :, None] * fx[None, None,:]
                        filters.append(kernel)
            
            self.register_buffer('haar_kernels', torch.stack(filters).unsqueeze(1))


    def inverse(self, subbands):
        stacked = torch.stack([subbands[name] for name in self.subband_names], dim=2)
        b, c, _, d, h, w = stacked.shape

        # reshape
        stack_reshaped = stacked.reshape(b * c, 8, d, h, w)

        reconstructed = F.conv_transpose3d(stack_reshaped, self.haar_kernels, stride=2)

        # reconstruct the reconstruction
        reconstructed = reconstructed.reshape( 
            b,
            c,
            reconstructed.shape[-3],
            reconstructed.shape[-2],
            reconstructed.shape[-1]
        )

        return reconstructed

    def forward(self, x):
        # HAAR ts
        b, c, d, h, w = x.shape
        assert d % 2 == 0, "Depth must be even." # they gotta be even for the math (if not use padding)
        assert h % 2 == 0, "Depth must be even."
        assert w % 2 == 0, "Depth must be even."

        x_reshaped = x.reshape(b*c, 1, d, h, w)

        out = F.conv3d(x_reshaped, self.haar_kernels, stride=2)

        out = out.reshape(b, c, 8, d // 2, h // 2, w //2) # beauty of haar

        subbands = {self.subband_names[i]: out[:, :, i] for i in range(8)} # all 8 types

        return out, subbands


# Spiking convolutional network. Wavelet slapped on the front end ; best for comparison
class WaveletModel(nn.Module):
    def __init__(self, num_classes=10, beta=0.9):
        super().__init__()

        self.haar = Haar2DTransform()

        self.conv1 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.lif1 = snn.Leaky(beta=beta)

        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.lif2 = snn.Leaky(beta=beta)

        self.pool2 = nn.MaxPool2d(2)

        # 34 → Haar(17) → Pool(8) → Pool(4)
        self.fc1 = nn.Linear(
            32 * 4 * 4,
            num_classes
        )

        self.lif_out = snn.Leaky(beta=beta)
    
    def forward(self,x):
        # we can vectorize prior to the stepping through time thanks to packages
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.haar(x)
        x = x.reshape(B, T, 8, H//2, W//2) # 2 polarities * 4 haar subbands = 8 channels

        # same process but now different var names
        batch_size = B
        num_steps = T
        
        # for architecture purposes
        layer1_h = H // 2 # Haar transform
        layer2_h = layer1_h // 2
        
        # important tools to be used later
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()

        # return values
        spike_record = []
        layer1_spikes = 0.0
        layer2_spikes = 0.0
        output_spikes = 0.0



        for step in range(num_steps): # heavy lifting time

            # start 'er up
            cur = x[:, step] # (B,C,H,W)

            # first layer
            cur = self.conv1(cur) 
            spik1,mem1 = self.lif1(cur, mem1)
            
            # pool 1 layer
            cur = self.pool1(spik1)

            # second layer
            cur = self.conv2(cur)
            spik2,mem2 = self.lif2(cur,mem2)

            # pool 2 layer
            cur = self.pool2(spik2)

            # thirdish layer i think (linear)
            cur = cur.reshape(batch_size, -1)

            # linearing
            cur = self.fc1(cur)

            # we out
            spik_out, mem_out = self.lif_out(cur, mem_out)

            # individual layers let me do math that help good
            layer1_spikes += spik1.sum()
            layer2_spikes += spik2.sum()
            output_spikes += spik_out.sum()

            spike_record.append(spik_out) # helpful tool; for later ts

        # rates change now bc the model is a lil diff
        layer1_fire_rate = layer1_spikes / (batch_size * num_steps * 16 * layer1_h * layer1_h) # rates are also helpul
        layer2_fire_rate = layer2_spikes / (batch_size * num_steps * 32 * layer2_h * layer2_h)
        output_fire_rate = output_spikes / (batch_size * num_steps * 10)

        return torch.stack(spike_record, dim=0), {
            "layer1sp": layer1_spikes, # spikes ts
            "layer2sp": layer2_spikes,
            "outputsp": output_spikes,
            "layer1fr": layer1_fire_rate, # fr here means firing rate, not for real (frfr)
            "layer2fr": layer2_fire_rate,
            "outputfr": output_fire_rate} # yippie yay yippie

class WaveletModel_CIFAR(nn.Module):
    def __init__(self, num_classes=10, beta=0.9):
        super().__init__()

        self.num_classes = num_classes

        self.haar = Haar2DTransform() # fixed haar transform

        self.conv1 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.lif1 = snn.Leaky(beta=beta)

        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.lif2 = snn.Leaky(beta=beta)

        self.pool2 = nn.MaxPool2d(2)

        # no third pool
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        
        self.lif3 = snn.Leaky(beta=beta)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(
            64,
            num_classes
        )

        self.lif_out = snn.Leaky(beta=beta)
    
    def forward(self,x):
        # we can vectorize prior to the stepping through time thanks to packages
        # batch, num time steps, channels, height, width
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.haar(x)
        x = x.reshape(B, T, 8, H//2, W//2) # 2 polarities * 4 haar subbands = 8 channels

        # same process but now different var names
        batch_size = B
        num_steps = T

        # for architecture purposes
        layer1_h = H // 2 # Haar transform
        layer2_h = layer1_h // 2
        layer3_h = layer2_h // 2
        
        # important tools to be used later
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_out = self.lif_out.init_leaky()

        # return values
        spike_record = []
        layer1_spikes = 0.0
        layer2_spikes = 0.0
        layer3_spikes = 0.0
        output_spikes = 0.0


        for step in range(num_steps): # heavy lifting time

            # start 'er up
            cur = x[:, step] # (B,C,H,W)

            # first layer
            cur = self.conv1(cur) 
            spik1,mem1 = self.lif1(cur, mem1)            
            # pool 1 layer
            cur = self.pool1(spik1)

            # second layer
            cur = self.conv2(cur)
            spik2,mem2 = self.lif2(cur,mem2)
            # pool 2 layer
            cur = self.pool2(spik2)

            # third layer
            cur = self.conv3(cur)
            spik3,mem3 = self.lif3(cur, mem3)

            # avg pool
            cur = self.avgpool(spik3)

            # classifier layer (linear)
            cur = cur.view(B, -1)

            cur = self.fc1(cur)

            # linearing
            cur = cur.reshape(batch_size, -1)

            # we out
            spik_out, mem_out = self.lif_out(cur, mem_out)

            # individual layers let me do math that help good
            layer1_spikes += spik1.sum()
            layer2_spikes += spik2.sum()
            layer3_spikes += spik3.sum()
            output_spikes += spik_out.sum()

            spike_record.append(spik_out) # helpful tool; for later ts

        # rates change now bc the model is a lil diff
        layer1_fire_rate = layer1_spikes / (batch_size * num_steps * 16 * layer1_h * layer1_h) # rates are also helpul
        layer2_fire_rate = layer2_spikes / (batch_size * num_steps * 32 * layer2_h * layer2_h)
        layer3_fire_rate = layer3_spikes / (batch_size * num_steps * 64 * layer3_h * layer3_h)
        output_fire_rate = output_spikes / (batch_size * num_steps * self.num_classes)

        return torch.stack(spike_record, dim=0), {
            "layer1sp": layer1_spikes, # spikes ts
            "layer2sp": layer2_spikes,
            "layer3sp": layer3_spikes,
            "outputsp": output_spikes,
            "layer1fr": layer1_fire_rate, # fr here means firing rate, not for real (frfr)
            "layer2fr": layer2_fire_rate,
            "layer3fr": layer3_fire_rate,
            "outputfr": output_fire_rate} # yippie yay yippie
    

class WaveletModel_Gesture(nn.Module):
    def __init__(self, num_classes=11, beta=0.9):
        super().__init__()

        self.num_classes = num_classes

        self.haar = Haar2DTransform() # fixed haar transform

        self.conv1 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.lif1 = snn.Leaky(beta=beta)

        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.lif2 = snn.Leaky(beta=beta)

        self.pool2 = nn.MaxPool2d(2)

        # no third pool
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        
        self.lif3 = snn.Leaky(beta=beta)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(
            128,
            num_classes
        )

        self.lif_out = snn.Leaky(beta=beta)
    
    def forward(self,x):
        # we can vectorize prior to the stepping through time thanks to packages
        # batch, num time steps, channels, height, width
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.haar(x)
        x = x.reshape(B, T, 8, H//2, W//2) # 2 polarities * 4 haar subbands = 8 channels

        # same process but now different var names
        batch_size = B
        num_steps = T

        # for architecture purposes
        layer1_h = H // 2 # Haar transform
        layer2_h = layer1_h // 2
        layer3_h = layer2_h // 2
        
        # important tools to be used later
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_out = self.lif_out.init_leaky()

        # return values
        spike_record = []
        layer1_spikes = 0.0
        layer2_spikes = 0.0
        layer3_spikes = 0.0
        output_spikes = 0.0


        for step in range(num_steps): # heavy lifting time

            # start 'er up
            cur = x[:, step] # (B,C,H,W)

            # first layer
            cur = self.conv1(cur) 
            spik1,mem1 = self.lif1(cur, mem1)            
            # pool 1 layer
            cur = self.pool1(spik1)

            # second layer
            cur = self.conv2(cur)
            spik2,mem2 = self.lif2(cur,mem2)
            # pool 2 layer
            cur = self.pool2(spik2)

            # third layer
            cur = self.conv3(cur)
            spik3,mem3 = self.lif3(cur, mem3)

            # avg pool
            cur = self.avgpool(spik3)

            # classifier layer (linear)
            cur = cur.view(B, -1)

            cur = self.fc1(cur)

            # linearing
            cur = cur.reshape(batch_size, -1)

            # we out
            spik_out, mem_out = self.lif_out(cur, mem_out)

            # individual layers let me do math that help good
            layer1_spikes += spik1.sum()
            layer2_spikes += spik2.sum()
            layer3_spikes += spik3.sum()
            output_spikes += spik_out.sum()

            spike_record.append(spik_out) # helpful tool; for later ts

        # rates change now bc the model is a lil diff
        layer1_fire_rate = layer1_spikes / (batch_size * num_steps * 16 * layer1_h * layer1_h) # rates are also helpul
        layer2_fire_rate = layer2_spikes / (batch_size * num_steps * 32 * layer2_h * layer2_h)
        layer3_fire_rate = layer3_spikes / (batch_size * num_steps * 128 * layer3_h * layer3_h)
        output_fire_rate = output_spikes / (batch_size * num_steps * self.num_classes)

        return torch.stack(spike_record, dim=0), {
            "layer1sp": layer1_spikes, # spikes ts
            "layer2sp": layer2_spikes,
            "layer3sp": layer3_spikes,
            "outputsp": output_spikes,
            "layer1fr": layer1_fire_rate, # fr here means firing rate, not for real (frfr)
            "layer2fr": layer2_fire_rate,
            "layer3fr": layer3_fire_rate,
            "outputfr": output_fire_rate} # yippie yay yippie