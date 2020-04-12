import numpy as np 
from network import FCN
from fusion import Fusion
from torch import optim
import torch
import matplotlib.pyplot as plt
import math
import torch.nn as nn 
import os
import codebook
from mod_ant import *

# system paras
tx_power = [-37.3375  -32.3375  -27.3375  -22.3375  -17.3375  -12.3375   -7.3375]
num_antenna = 4
num_subc = 32
beam_set = 64

# ML paras
epoches = 10
batch_size = 512
lr = 0.001


# setting gpu devices, then change the input data & the model into .gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def to_label(label):
    num = label.shape[0]
    for i in range(num):
        label[i] = label[i]-1
    return label

# adjusting the learning rate
def adjust_lr(optimizer, epoch):
    # changing the lr according to the epoch.
    if(epoch<epoches/2):
        adapt_lr = lr
    else:
        if(epoch<0.9*epoches):
            adapt_lr = 0.1*lr
        else:
            adapt_lr = 0.01*lr
    for para_group in optimizer.param_groups:
        para_group['lr'] = adapt_lr

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

def reform(pilot):
    # just change the form of the original .mat file
    num = pilot.shape[0]
    col = pilot.shape[1]
    real_pilot = np.zeros([num,col])
    imag_pilot = np.zeros([num,col])
    for i in range(num):
        tem = pilot[i]
        for j in range(col):
            real_pilot[i,j] = tem[j][0]
            imag_pilot[i,j] = tem[j][1]
    complex_pilot = np.zeros([num,col],dtype = 'complex64')
    # note that we may alternate the number of antennas, but will not change the number of subcarriers
    for i in range(num):
        for j in range(col):
            complex_pilot[i,j] = complex(real_pilot[i,j],imag_pilot[i,j])
    complex_pilot = complex_pilot.reshape(int(num),32,int(col/32))
    return complex_pilot

def add_noise(pilot,SNR):
    num = pilot.shape[0]
    col = pilot.shape[1]
    for i in range(num):
        norm = 0
        for j in range(col):
            norm += pilot[i,j]*pilot[i,j]
        snr = 10**(SNR/10)
        var = norm/snr
        noise = np.random.randn(col)*np.sqrt(var/col)
        pilot[i] += noise
    return pilot 

acc = []
comm_rate = []
#### loading data from mat files   ###
import h5py
from codebook import upa_codebook



for iter in range(1):
    name = '/home/yyw/chbian/graduation_design/Dataset/'+'dataset'+str(iter+1)+'.mat'
    data=h5py.File(name,'r')
    data = data['dataset']

    # for the beam prediction
    train_x = np.squeeze(data[('inpTrain')])
    val_x = np.squeeze(data[('inpVal')])
    train_label = np.squeeze(data[('labelTrain')])
    train_label = train_label.astype(np.int16)
    val_label = np.squeeze(data[('labelVal')])
    val_label = val_label.astype(np.int16)

    train_labels = to_label(train_label)      
    val_labels = to_label(val_label)

    # will include the pilot signal
    mmwave_chT = data[('highFreqChTrain')]
    mmwave_chV = data[('highFreqChVal')]
    # would like to use some pilot signal
    OFDM_sample = 8
    ant_sample = 8
    num_ant = 64
    num_OFDM = 32
    num_train = mmwave_chT.shape[0]
    num_val = mmwave_chV.shape[0]

    pilot_train = mmwave_chT[:,::OFDM_sample,::ant_sample]
    pilot_val   = mmwave_chV[:,::OFDM_sample,::ant_sample]
    # changing the dimension and change them to real number
    # also given the SNR, add noise to the signal
    pilot_train = np.array(pilot_train)
    pilot_train = pilot_train.reshape(num_train,-1)
    pilot_train = de_complex(pilot_train)
    pilot_train = add_noise(pilot_train,5)

    pilot_val = np.array(pilot_val)
    pilot_val = pilot_val.reshape(num_val,-1)
    pilot_val = de_complex(pilot_val)
    pilot_val = add_noise(pilot_val,5)
    
    
    #merging the pilot with the sub6G signal
    #total_train = np.concatenate((train_x,pilot_train),axis = 1)
    #total_val   = np.concatenate((val_x,pilot_val),axis = 1) 
    total_train = pilot_train
    total_val = pilot_val
    #####   The training phase!   ######

    num_sample = train_x.shape[0]

    fcn = Fusion().cuda()
    optimizer = optim.Adam(fcn.parameters(), lr = lr)
    loss_fun = nn.CrossEntropyLoss()

    fcn.train()
    for epoch in range(epoches):
        print('epoch', epoch)
        adjust_lr(optimizer, epoch)
        # have shuffled already in the data preprocessing process
        
        for i in range(int(num_sample/batch_size)+1):

            if((i+1)*batch_size <= num_sample):
                X = train_x[i*batch_size:(i+1)*batch_size, :]
                P = pilot_train[i*batch_size:(i+1)*batch_size, :]
                Y = train_labels[i*batch_size:(i+1)*batch_size]
            else:
                X = train_x[i*batch_size:num_sample, :]
                P = pilot_train[i*batch_size:num_sample, :]
                Y = train_labels[i*batch_size:num_sample]

            X = torch.from_numpy(X).float().cuda()
            P = torch.from_numpy(P).float().cuda()
            Y = torch.from_numpy(Y).long().cuda()
            pred = fcn(X,P)
            loss = loss_fun(pred, Y)
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
    #############  The Testing Phase!  ############

    fcn.eval()
    val_xt = torch.from_numpy(val_x).float().cuda()
    p = torch.from_numpy(pilot_val).float().cuda()
    pred_val = fcn(val_xt,p)
    pred_val = pred_val.cpu().detach().numpy()

    # calculating the accuracy
    pred_label = pred_val.argmax(axis = 1)
    count = 0
    for i in range(val_xt.shape[0]):
        if(val_labels[i]==pred_label[i]):
            count = count+1
    acc.append(count/val_xt.shape[0])
    print(count/val_xt.shape[0])

    # then calculating the achieve rate


    W = upa_codebook(1,64,1,0.5)
    mmwave_ch = np.array(mmwave_chV)
    row = mmwave_ch.shape[0]
    mmwave_ch = mmwave_ch.reshape(row,-1)
    mmwave_ch = reform(mmwave_ch)    # need revising because of the unknown grammar.

    # assuming everything goes right
    rate = []
    for user in range(num_val):
        H = mmwave_ch[user,:,:]    # 32*64
        w = W[pred_label[user],:]  # 1*64
        rec_power = abs(np.dot(H,np.conjugate(w.T)))*abs(np.dot(H,np.conjugate(w.T))) 
        tem = 0
        for sub in range(32):
            tem += np.log2(1+rec_power[sub])
        tem = tem/32
        rate.append(tem)
    comm_rate.append(np.mean(rate))


print(acc)    # show the accracy for the total process.
print(comm_rate)