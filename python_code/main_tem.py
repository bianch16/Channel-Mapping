import numpy as np 
from network import FCN
from torch import optim
import torch
import matplotlib.pyplot as plt
import math
import torch.nn as nn 
import torch.nn.functional as F
import os
'''
# for the channel NMSE fitting  
train_mmw = data[0,0]['highFreqChTrain']
val_mmw = data[0,0]['highFreqChVal']
'''
# system paras
dataset_ratio = [0.005,0.01,0.05,0.1,0.3,0.5,0.7,1]
tx_power = [-37.3375  -32.3375  -27.3375  -22.3375  -17.3375  -12.3375   -7.3375]
#dataset_ratio = [0.1]
num_antenna = 4
num_subc = 32
beam_set = 64
overall_pred = []
overall_tar = []
overall_low = []

# ML paras
epoches = 1
batch_size = 1024
lr = 0.0007
val_rate = 0.2
lr_decay = 1e-4

# setting gpu devices, then change the input data & the model into .gpu()
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def to_label(label):
    num = label.shape[0]
    for i in range(num):        
        label[i] = label[i]-1
    return label

#### loading data from mat files   ###
import h5py
data=h5py.File('dataset1.mat','r')
data = data['dataset']

# for the beam prediction
# sub6G channel
train_x = np.squeeze(data[('inpTrain')])
val_x = np.squeeze(data[('inpVal')])
train_label = np.squeeze(data[('labelTrain')])   # have some problem in labels
train_labels = train_label.astype(np.int16)
val_label = np.squeeze(data[('labelVal')])
val_labels = val_label.astype(np.int16)

train_labels = to_label(train_labels)      
val_labels = to_label(val_labels)

# pilot channel in 28GHz
mmwave_chT = data[('highFreqChTrain')]
mmwave_chV = data[('highFreqChVal')]
# channel via pilot signal
OFDM_sample = 8
ant_sample = 8
num_ant = 64
num_OFDM = 32
num_train = mmwave_chT.shape[0]
num_val = mmwave_chV.shape[0]
def de_complex(pilot):
    # concatenate the real part with the image part
    num = pilot.shape[0]
    col = pilot.shape[1]
    real_pilot = np.zeros([num,col])
    imag_pilot = np.zeros([num,col])
    for i in range(num):
        tem = pilot[i]
        for j in range(col):
            real_pilot[i,j] = tem[j][0]
            imag_pilot[i,j] = tem[j][1]
    pilot = np.concatenate((real_pilot,imag_pilot),axis=1)
    return pilot

pilot_train = mmwave_chT[:,::OFDM_sample,::ant_sample]
pilot_val   = mmwave_chV[:,::OFDM_sample,::ant_sample]
# changing the dimension and change them to real number
pilot_train = np.array(pilot_train)
pilot_train = pilot_train.reshape(num_train,-1)
pilot_train = de_complex(pilot_train)

pilot_val = np.array(pilot_val)
pilot_val = pilot_val.reshape(num_val,-1)
pilot_val = de_complex(pilot_val)

# concatenate the data:the most intuitive way for training
total_train = np.concatenate((train_x,pilot_train),axis = 1)
total_val   = np.concatenate((val_x,pilot_val),axis=1)
# adjusting the learning rate
def adjust_lr(optimizer, epoch):
    is_decay = epoch/25
    if(is_decay):
        adapt_lr = 0.1*lr
    else:
        adapt_lr = lr
    for para_group in optimizer.param_groups:
        para_group['lr'] = adapt_lr

'''
def loss_fun(pred, label):
    # note pred with shape (batch_size, 2048)
    
    diff = pred - label
    num = torch.diag(torch.matmul(torch.t(diff), diff), 0)
    den = torch.diag(torch.matmul(torch.t(label), label), 0)
    nmseVec = num/den
    loss = (torch.sum(nmseVec))/(2*diff.shape[0])

    return loss

def to_complex(val):
    num_sample = val.shape[0]
    half = int(val.shape[1]/2)
    channel = np.zeros([num_sample, half], dtype = 'complex')
    for sample in range(num_sample):
        for i in range(half):
            channel[sample, i] = complex(val[sample, i], val[sample, half + i])
    return channel
'''

        

#####   The training phase!   ######

#print('###################dataset size is:', int(151402 * (1-val_rate) * dataset_ratio[iter+1]),'#####################')
num_sample = train_x.shape[0]

fcn = FCN()
optimizer = optim.Adam(fcn.parameters(), lr = lr)  # may need to adapt the lr
# using the Crossentropy loss for the classification

for epoch in range(epoches):
    print('epoch', epoch)
    adjust_lr(optimizer, epoch)
    # have shuffled already in the data preprocessing process
    
    for i in range(int(num_sample/batch_size)+1):

        if((i+1)*batch_size <= num_sample):
            X = total_train[i*batch_size:(i+1)*batch_size, :]
            Y = train_labels[i*batch_size:(i+1)*batch_size]
        else:
            X = total_train[i*batch_size:num_sample, :]
            Y = train_labels[i*batch_size:num_sample]

        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).long()
        pred = fcn(X)
        loss = F.cross_entropy(pred, Y)
        if(i%100 == 0):
            print('classification_loss : ', loss)  

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
'''
# save the pre_trained paras to the given path:
PATH = './model/model_' + str(iter) + '.pt'
torch.save(fcn.state_dict(), PATH)
'''
# calculating the Ber
val_xt = torch.from_numpy(total_val).float()
pred_val = fcn(val_xt)
pred_val = pred_val.cpu().detach().numpy()
pred_label = pred_val.argmax(axis = 1)
count = 0
for i in range(val_xt.shape[0]):
    if(pred_label[i]==val_labels[i]):
        count = count+1
print(count/val_xt.shape[0])
'''
# caluculating the NMSE and visualize
val_xt = torch.from_numpy(val_x).float().cuda()
pred_val = fcn(val_xt)
val_yt = torch.from_numpy(val_label).float().cuda()

loss_val = loss_fun(pred_val, val_yt)
loss_val = loss_val.cpu().detach().numpy()
print(loss_val)
NMSE.append(loss_val)
# plotting the NMSE loss
print(NMSE)
plt.xlabel('dataset_ration')
plt.ylabel('NMSE')
plt.plot(dataset_ratio,NMSE)
plt.show()'''