import numpy as np

def smoothing_grid(arr, noc=None, ratio=None, opt='mean'):
    ''' It coarse a grid from shape=(mesh mesh mesh) to shape=(noc noc noc), a tot of ratio^3 original cells are used to average and create noc cube.
        Parameters:
            * arr (narray)	: 3D array of data
            * noc (int)		: dimension of coarsening of data
            * ratio (int)   : scaling ratio between the two resolution
            * opt (string)  : operation on the smoothing (mean, min or max)
    '''
    
    if(ratio == None and noc != None):
        ratio = int(float(arr.shape[0])/noc)
    elif(ratio != None and noc == None):
        noc = int(float(arr.shape[0])/ratio)
    elif(ratio != None and noc != None):
        ValueError("Specify just one of the two quantity 'noc' or 'ratio'.")
    else:
        ValueError("Specify at least one of the two quantity 'noc' or 'ratio'.")

    if(opt == 'mean'):
        operator = np.mean
    elif(opt == 'max'):
        operator = np.max
    elif(opt == 'min'):
        operator = np.min
    elif(opt == 'sum'):
        operator = np.sum
    else:
        ValueError("Operation on the smoothing must be string ('mean', 'min' or 'max').")
    
    coarse_data = np.zeros([noc, noc])  # nocXnocXnoc
    for i in range(noc):
        for j in range(noc):
            cut = arr[(ratio*i):(ratio*(i+1)), (ratio*j):(ratio*(j+1))]
            coarse_data[i][j] = np.mean(cut)
    return coarse_data