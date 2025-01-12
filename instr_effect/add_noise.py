"""
Created on Fri May 5, 2023
@author: Chris Finley, modified by Michele Bianco
"""

import numpy as np, os, dask.array as da, random, sys, gc
import casacore.tables as tb

from daskms import xds_from_table
from time import time
from datetime import timedelta

import sys
from calculate_noise import add_noise, read_SEFD, write_vis, get_vis

start_time = time()

root_name = sys.argv[1]
path_out = sys.argv[2]

ms_input = path_out+'ms/'+root_name.replace('dT', 'dTgain')+'.MS'
ms_output = ms_input.replace('dT', 'dTnoise')

if not (os.path.exists(ms_output)):
    os.system('cp -r %s %s' %(ms_input, ms_output))
    #os.system('mv %s %s' %(ms_input, ms_output))
else:
    #raise FileExistsError(' %s do not exist. Stopping.' %ms_input)
    print(' folder %s exist.' %ms_output)

# define noise parameters
n_days = 250
SEFD_path = '/store/ska/sk014/dataset_sdc3/inputs/SEFD_tables/AAVS2_sensitivity_ra0.00deg_dec_-30.00deg_0.00hours.txt'
rseed = random.randint(0, 1e9)
#write_col = 'MODEL_DATA'
write_col = 'DATA'

# allocate MS with Dask
ds = xds_from_table(ms_input, chunks={'row': 511*256*10, 'chan': 1, 'corr': 4, 'uvw': 3})[0]
print(f" MS file loaded in :     {timedelta(seconds=time()-start_time)}")
get_time = time()

# calculate noise
SEFD = read_SEFD(SEFD_path)
vis = get_vis(ms_input, write_col)
noisy_vis = add_noise(vis, SEFD/da.sqrt(n_days), ms_input, ds, seed=rseed)

print(f" Noise applied in :      {timedelta(seconds=time()-get_time)}")
get_time = time()

# free memory
del ds, vis
gc.collect()

# apply noise and gain to visibility of new measurment set
ms = tb.table(ms_output, ack=False, readonly=False)
ms.putcol(write_col, noisy_vis.values)
ms.close()

print(f" MS file written in :    {timedelta(seconds=time()-get_time)}")
print(f" Total time :            {timedelta(seconds=time()-start_time)}")
