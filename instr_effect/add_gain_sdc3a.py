"""
Created on Fri May 5, 2023
@author: Shreyam Parth Krishna, modified by Michele Bianco
"""

import numpy as np, os, gc, random, sys
import casacore.tables as tb

from tqdm import tqdm
from time import time
from datetime import timedelta

from utils_gain import calculate_SDC3_gains
from calculate_gain import DIAntennaErrorSDC3A, DIBaselineErrorSDC3A

start_time = time()

root_name = sys.argv[1]
path_out = sys.argv[2]

ms_input = path_out+'ms/'+root_name+'.MS'
ms_output = ms_input.replace('dT', 'dTgain')

if not (os.path.exists(ms_output)):
    os.system('cp -r %s %s' %(ms_input, ms_output))
    #os.system('mv %s %s' %(ms_input, ms_output))
else:
    #raise FileExistsError(' %s do not exist. Stopping.' %ms_input)
    print(' folder %s exist... over-writing to MODEL_DATA.' %ms_input)

# Open the MeasurementSet
ms = tb.table(ms_output, ack=False, readonly=False)

# Get visibility
vis_data = ms.getcol('DATA')
print(f"Get visibility data : {timedelta(seconds=time()-start_time)}")
get_time = time()
print("------------------------------------------------")

# Get the number of frequency channels
frequency = ms.SPECTRAL_WINDOW.CHAN_FREQ[:]
num_chan = frequency[0].size

# Get the unique antenna IDs, number of antennas and baselines
ant1 = ms.getcol('ANTENNA1')
ant2 = ms.getcol('ANTENNA2')
antennas = np.unique(np.concatenate((ant1, ant2)))
num_antennas = len(antennas)
num_baselines = int(num_antennas * (num_antennas - 1) / 2)
del ant1, ant2, antennas

# Get the number of time steps
times = ms.getcol('TIME')
num_times = len(np.unique(times))
del times

# garbage collector
gc.collect()
print(f" Get unique time and antennas : {timedelta(seconds=time()-get_time)}")
get_time = time()

# combine gain error to visibility data
amp_mean, amp_sig = 1, 2e-4
pahse_mean, phase_sig = 0, 0.02*np.pi/180
rseed = random.randint(0, 1e9)

print(' DI Antenna Error')
antennaErrorSKALow = DIAntennaErrorSDC3A(numAntenna=num_antennas, frequencyChannels=np.array([frequency]), timeSteps=num_times)

print(' DI Baseline Error')
gain_matrix = DIBaselineErrorSDC3A(gainMatrix = antennaErrorSKALow)
gain = gain_matrix.reshape(gain_matrix.shape[0]*gain_matrix.shape[1], gain_matrix.shape[2], 1) # Now Ntimes*Nbaselines, Nchannels
gain_vis_data = (gain*vis_data).reshape(num_times*num_baselines, 1, 4)

print(f" Apply gains to visibility : {timedelta(seconds=time()-start_time)}")
get_time = time()

# Add the MODEL_DATA column to the table and copy DATA values
"""
try:
    data_desc = ms.getcoldesc('DATA')
    data_desc['name'] = 'MODEL_DATA'
    ms.addcols(desc=data_desc)
except:
    print(' column data MODEL_DATA already exist... Over-writing data.')

ms.putcol('MODEL_DATA', gain_vis_data)
"""
ms.putcol('DATA', gain_vis_data)

# Close the MeasurementSet
ms.close()

print(f" gain effect written in MS: {timedelta(seconds=time()-get_time)}")
print(f" Total time :            {timedelta(seconds=time()-start_time)}")