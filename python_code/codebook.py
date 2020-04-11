import numpy as np 
def phase_shifter(tx,spacing):
    code = np.zeros([tx,tx],dtype = 'complex32')
    theta = np.arange(0,np.pi-1e-10,np.pi/tx)
    for i in range(tx):
        for k in range(tx):
            code[i,k] = 1/np.sqrt(tx)*np.exp(-1j*k*2*np.pi*spacing*np.cos(theta[i]))
    return code 

def upa_codebook(tx,ty,tz,spacing):
    code_x = phase_shifter(tx,spacing)
    code_y = phase_shifter(ty,spacing)
    code_z = phase_shifter(tz,spacing)

    # then use the kron function in Numpy package
    code_xy = np.kron(code_y,code_x)
    code = np.kron(code_z,code_xy)

    return code
    