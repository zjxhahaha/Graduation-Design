import math
import torch
import torch.nn as nn
from torchsummary import summary

class conv_block(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(conv_block,self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,3,1,bias=False,padding=1),
            # nn.ReLU(),
            nn.ELU(),
            nn.BatchNorm3d(outchannels),
            nn.Conv3d(outchannels,outchannels,3,1,bias=False,padding=1),
            # nn.ReLU(),
            nn.ELU(),
            nn.BatchNorm3d(outchannels),
            nn.MaxPool3d(2,2),
        )

    def forward(self,x):
        x = self.block(x)
        return x

class CNN(nn.Module):
    def __init__(self,nb_filter,nb_block):
        super(CNN,self).__init__()
        self.nb_block = nb_block
        self.pre = nn.Sequential(
            nn.Conv3d(1,nb_filter,3,1,padding=1),
            nn.ReLU())
        self.block, last_channels = self._make_block(nb_filter,nb_block)
        self.gap = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Sequential(
            nn.Linear(last_channels,1),
            nn.ELU(),
            )

        self.male_fc = nn.Sequential(
            nn.Linear(2,16),
            nn.Linear(16,8),
            # nn.ReLU(),
            nn.ELU(),
            )
        self.end_fc = nn.Sequential(
            nn.Linear(32,16),
            # nn.Dropout(0.5),
            nn.Linear(16,8),
            nn.Linear(8,1),
            # nn.ReLU(),
            nn.ELU()
            )

    def _make_block(self,nb_filter,nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):
            outchannels = inchannels * 2
            blocks.append(conv_block(inchannels,outchannels))
            inchannels = outchannels 
        return nn.Sequential(*blocks),inchannels

    def forward(self,x):
        x = self.pre(x)
        x = self.block(x)
        x = self.gap(x)
        x = torch.reshape(x,(x.size(0),-1))
        x = self.fc(x)
        # x = self.end_fc(x)
        return x

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
# model = CNN(8,5).to(device)
# print(model)

# iuput = torch.autograd.Variable(torch.rand(10,1,91,109,91)).to(device)
# male_input = torch.autograd.Variable(torch.rand(10,2)).to(device)
# out = model(iuput,male_input)
# print(out)
# print(out.size())