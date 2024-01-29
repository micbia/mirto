import numpy as np
import astropy.units as u
from datetime import timedelta

from create_screen import simulate_screen

i = 0
path = '/scratch/snx3000/mibianco/output_sdc3/dataLC_256_train_090523/atmo/'
screen_file = '%sscreen_4h_i%d.npy' %(path, i)
#iono_fits = '%sscreen_4h_i%d.fits' %(path, i)

idx_f = 750
freqs_sdc = np.loadtxt('/store/ska/sk09/dataset_sdc3/freqs_sdc3.txt')
freq = freqs_sdc[idx_f]

# number of time steps
t_obs = timedelta(hours=4, minutes=0, seconds=0, milliseconds=0)
t_int = timedelta(seconds=10)
nr_tsteps = int(t_obs.total_seconds() / t_int.total_seconds())

# simulate screen
simulate_screen(screen_width_metres=200e3, r0=7e3, bmax=20e3, sampling=100.0, speed=150*u.km/u.h, rate=0.1, alpha_mag=0.999, num_times=nr_tsteps, filename=screen_file, rseed=29119541)

