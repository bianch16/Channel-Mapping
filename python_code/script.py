import h5py
import numpy as np
name = '/home/yyw/chbian/graduation_design/Dataset/'+'dataset'+str(1)+'.mat'
data=h5py.File(name,'r')
data = data['dataset']

max_rate = data[('maxRateVal')]
print(np.mean(max_rate))