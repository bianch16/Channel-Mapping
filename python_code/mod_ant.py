# this python files contains functions to modify the mmwave antennas

import numpy as np

def antenna_label(channel,W):
    # channel is the mmwave channel with given antennas while W is the codebook, 
    # assuming channel: num_users x num_OFDM x num_antennas
    num = channel.shape[0]
    label = np.zeros([num,1],dtype = 'np.int16')
    for i in range(num):
        tem = abs(np.dot(np.conjugate(channel[i,:,:]),W))  # 32*antennas
        tem = tem*tem
        tem = np.sum(tem,0)
        label[i] = np.argmax(tem)
    return label

