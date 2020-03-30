import torch
from torch import nn
import torch.nn.functional as F
drop_fac = 0.4



class FCN(nn.Module):
    
    def __init__(self):
        # Trying to use the batch_normalization to replace the imageinputlayer in Matlab

        super(FCN, self).__init__()
        
        self.r = nn.ReLU()
        self.drop = nn.Dropout(drop_fac)
        # total five stacks
        self.fc1 = nn.Linear(256,2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048,2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048,2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.fc4 = nn.Linear(2048,2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc5 = nn.Linear(2048,2048)
        self.bn5 = nn.BatchNorm1d(2048)

        # the final classification layer for the beam prediction
        self.fc6 = nn.Linear(2048,64)

    
    def forward(self, x):
        
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

        # final classification layer
        x = self.fc6(x)   
            
        return x

class Sub_FCN(nn.Module):
    
    def __init__(self):
        # This nerual network is designed for the pilot data

        super(FCN, self).__init__()
        
        self.r = nn.ReLU()
        self.drop = nn.Dropout(drop_fac)
        # total five stacks
        self.fc1 = nn.Linear(256,2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048,2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048,2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.fc4 = nn.Linear(2048,2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc5 = nn.Linear(2048,2048)
        self.bn5 = nn.BatchNorm1d(2048)

        # the final classification layer for the beam prediction
        self.fc6 = nn.Linear(2048,64)

    
    def forward(self, x):
        
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

        # final classification layer
        x = self.fc6(x)   
            
        return x

class Fusion(nn.Module):
    
    def __init__(self):
        # This neural network is designed for the feature fusion

        super(FCN, self).__init__()
        
        self.r = nn.ReLU()
        self.drop = nn.Dropout(drop_fac)
        # total five stacks
        self.fc1 = nn.Linear(256,2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048,2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048,2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.fc4 = nn.Linear(2048,2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc5 = nn.Linear(2048,2048)
        self.bn5 = nn.BatchNorm1d(2048)

        # the final classification layer for the beam prediction
        self.fc6 = nn.Linear(2048,64)

    
    def forward(self, x):
        
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

        # final classification layer
        x = self.fc6(x)   
            
        return x