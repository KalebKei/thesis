import torch
import torch.nn as nn
import snntorch as snn

# Spiking convolutional network ; best for comparison
class SNNModel(nn.Module):
    def __init__(self, num_classes=10, beta=0.9):
        super().__init__()
    
        # Conv feature extractor
        self.conv1 = nn.Conv2d( # simple first conv layer
            in_channels=2,
            out_channels=16,
            kernel_size=3,
            padding=1
        )    
        
        self.lif1 = snn.Leaky(beta=beta) # Leaky relu

        self.pool1 = nn.MaxPool2d(2) # Pooling; 34x34 -> 17x17

        self.conv2 = nn.Conv2d( # second conv layer
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.lif2 = snn.Leaky(beta=beta)

        self.pool2 = nn.MaxPool2d(2) # Pooling; 17x17 -> 8x8

        # 34 x 34 -- (pool1) --> 17x17 -- (pool2) --> 8x8
        self.fc1 = nn.Linear( # map to output
            32 * 8 * 8,
            num_classes
        )

        self.lif_out = snn.Leaky(beta=beta)
    
    def forward(self,x):
        batch_size = x.size(0) # it better be
        num_steps = x.size(1)
        
        # important tools to be used later
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()

        # return values
        spike_record = []
        layer1_spikes = 0
        layer2_spikes = 0
        output_spikes = 0

        for step in range(num_steps): # heavy lifting time
            # start 'er up
            cur = x[:, step]

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
            cur = cur.view(batch_size, -1)

            # linearing
            cur = self.fc1(cur)

            # we out
            spk_out, mem_out = self.lif_out(cur, mem_out)

            # individual layers let me do math that help good
            layer1_spikes += spik1.sum()
            layer2_spikes += spik2.sum()
            output_spikes += spk_out.sum()

            spike_record.append(spk_out) # helpful tool; for later ts

        layer1_fire_rate = layer1_spikes / (batch_size * num_steps * 16 * 34 * 34) # rates are also helpul
        layer2_fire_rate = layer2_spikes / (batch_size * num_steps * 32 * 17 * 17)
        output_fire_rate = output_spikes / (batch_size * num_steps * 10)

        return torch.stack(spike_record, dim=0), {
            "layer1sp": layer1_spikes, # spikes ts
            "layer1sp": layer2_spikes,
            "outputsp": output_spikes,
            "layer1fr": layer1_fire_rate, # fr here means firing rate, not for real (frfr)
            "layer2fr": layer2_fire_rate,
            "outputfr": output_fire_rate} # yippie yay yippie
