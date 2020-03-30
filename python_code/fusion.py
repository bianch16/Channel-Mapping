import torch
from torch import nn
import torch.nn.functional as F

drop_fac = 0.4
sub6G_dim = 256
pilot_dim = 64
neuron = 2048
catagory = 64

class Fusion(nn.Module):
    
    def __init__(self):
        # Trying to use the batch_normalization to replace the imageinputlayer in Matlab

        super(Fusion, self).__init__()
        
        #######   The Nerual network for sub6G channel   #######
        self.r = nn.ReLU()
        self.drop = nn.Dropout(drop_fac)
        # total five stacks
        self.fc1 = nn.Linear(sub6G_dim,neuron)
        self.bn1 = nn.BatchNorm1d(neuron)
        self.fc2 = nn.Linear(neuron,neuron)
        self.bn2 = nn.BatchNorm1d(neuron)
        self.fc3 = nn.Linear(neuron,neuron)
        self.bn3 = nn.BatchNorm1d(neuron)
        self.fc4 = nn.Linear(neuron,neuron)
        self.bn4 = nn.BatchNorm1d(neuron)
        self.fc5 = nn.Linear(neuron,neuron)
        self.bn5 = nn.BatchNorm1d(neuron)
        self.fc6 = nn.Linear(neuron,neuron)
        self.bn6 = nn.BatchNorm1d(neuron)
        self.bn7 = nn.BatchNorm1d(neuron)
        self.fc7 = nn.Linear(neuron,catagory)
        self.bn8 = nn.BatchNorm1d(catagory)
        self.fc8 = nn.Linear(catagory,catagory)

        ######    The Nerual network for pilot signal    #######
        self.pfc1 = nn.Linear(pilot_dim,neuron)
        self.pbn1 = nn.BatchNorm1d(neuron)
        self.pfc2 = nn.Linear(neuron,neuron)
        self.pbn2 = nn.BatchNorm1d(neuron)
        self.pfc3 = nn.Linear(neuron,neuron)
        self.pbn3 = nn.BatchNorm1d(neuron)
        self.pfc4 = nn.Linear(neuron,neuron)
        self.pbn4 = nn.BatchNorm1d(neuron)
        self.pfc5 = nn.Linear(neuron,neuron)
        self.pbn5 = nn.BatchNorm1d(neuron)
        self.pfc6 = nn.Linear(neuron,neuron)
        self.pbn6 = nn.BatchNorm1d(neuron)
        self.pbn7 = nn.BatchNorm1d(neuron)
        self.pfc7 = nn.Linear(neuron,catagory)
        self.pbn8 = nn.BatchNorm1d(catagory)
        self.pfc8 = nn.Linear(catagory,catagory)

        ######    The fusion network     #######
        self.fbn1 = nn.BatchNorm1d(catagory*2)
        self.ffc1 = nn.Linear(catagory*2,catagory)
        self.fbn2 = nn.BatchNorm1d(catagory)
        self.ffc2 = nn.Linear(catagory,catagory)
    
    def forward(self, x, pilot):
        #[x,pilot] = input
        ###### For the sub6G channel ######      
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.r(x)
        x = self.drop(x)

        x = self.bn2(x)
        x = self.fc2(x)
        x = self.r(x)
        x = self.drop(x)

        x = self.bn3(x)
        x = self.fc3(x)
        x = self.r(x)
        x = self.drop(x)

        x = self.bn4(x)
        x = self.fc4(x)
        x = self.r(x)
        x = self.drop(x)

        x = self.bn5(x)
        x = self.fc5(x)
        x = self.r(x)
        x = self.drop(x)

        # final classification layer
        x = self.bn7(x)
        x = self.fc7(x)
        #x = self.r(x)   
        #x = self.bn8(x)
        #x = self.fc8(x)
        
        ######  Then for the pilot channel ######
        pilot = self.pfc1(pilot)
        pilot = self.pbn1(pilot)
        pilot = self.r(pilot)
        pilot = self.drop(pilot)

        pilot = self.pfc2(pilot)
        pilot = self.pbn2(pilot)
        pilot = self.r(pilot)
        pilot = self.drop(pilot)

        pilot = self.pfc3(pilot)
        pilot = self.pbn3(pilot)
        pilot = self.r(pilot)
        pilot = self.drop(pilot) 

        pilot = self.pfc4(pilot)
        pilot = self.pbn4(pilot)
        pilot = self.r(pilot)
        pilot = self.drop(pilot)

        pilot = self.pfc5(pilot)
        pilot = self.pbn5(pilot)
        pilot = self.r(pilot)
        pilot = self.drop(pilot)

        pilot = self.pbn7(pilot)
        pilot = self.pfc7(pilot)

        ###### Now concatenate the trained network ######
        fused = torch.cat((x,pilot),1)        # now with size [batch_size,2*catagory]

        ###### Finally for the classification layer ######
        fused = self.fbn1(fused)
        fused = self.ffc1(fused)
        fused = self.r(fused)

        fused = self.fbn2(fused)
        fused = self.ffc2(fused)


        return fused
