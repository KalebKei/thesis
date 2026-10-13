import torch
import torch.nn as nn
import snntorch as snn
from Models.snn_wavelet import Haar2DTransform

# Spiking NMNIST perceptron network 
class WaveletPerceptronModel(nn.Module):
    def __init__(self, num_classes=10, beta=0.9):
        super().__init__()

        self.haar = Haar2DTransform()

        self.linear1 = nn.Linear(
            in_features=8,
            out_features=16
        )
        self.lif1 = snn.Leaky(beta=beta)
        self.pool1 = nn.MaxPool2d(2)

        self.linear2 = nn.Linear(
            in_features=16,
            out_features=32
        )
        self.lif2 = snn.Leaky(beta=beta)
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(
            32*4*4,
            num_classes
        )

        self.lif_out = snn.Leaky(beta=beta)


    def forward(self, x):
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
            cur = cur.permute(0,2,3,1) # move channels to last dimension for Linear layers

            # first layer
            cur = self.linear1(cur) 
            spik1,mem1 = self.lif1(cur, mem1)

            cur = spik1.permute(0,3,1,2) # move back for pooling
            cur = self.pool1(cur)

            # second layer
            cur = cur.permute(0,2,3,1) # move channels to last dimension for Linear layers

            cur = self.linear2(cur)
            spik2,mem2 = self.lif2(cur,mem2)
            
            cur = spik2.permute(0,3,1,2) # move back for pooling
            cur = self.pool2(cur)


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
