"""
Created on Wed Apr 12, 2023
@author: Michele Bianco

Modified from Karabo test script
"""

from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation, ObservationLong
from karabo.simulation.visibility import Visibility

from oskar.imager import Imager
import numpy as np, os, sys, pandas as pd, toml
import tools21cm as t2c
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

import astropy.units as u
import astropy.constants as cst
from astropy.io import fits
from astropy.wcs import WCS
from astropy.cosmology import Planck18
from astropy.coordinates import Angle

sys.path.append("/users/mibianco/codes/sdc3")
from utils.smoothing import smoothing_grid
from gain.iono.create_screen import simulate_TEC

root_name = sys.argv[1]
path_out = sys.argv[2]

# name of the outputfile to change manually to the time selected
run_name = path_out+'ms/'+root_name

idx = int(root_name[root_name.rfind('_i')+2:root_name.rfind('_i')+root_name[root_name.rfind('_i'):].find('_dT'):])
idx_f = int(root_name[root_name.rfind('ch')+2:root_name.rfind('ch')+root_name[root_name.rfind('ch'):].find('_'):])

FRG_GF = 'gf' in root_name
FRG_EXGF = 'gleam' in root_name or 'point' in root_name
PBEAM = 'beam' in root_name
ION = 'ion' in root_name
PLOT_IMG = False

print(' i = %d   i_channel = %d' %(idx, idx_f))
print(' Primary beam effect: %s' %PBEAM)
print(' Ionospheric effect: %s' %ION)
print(' Galactic foreground: %s' %FRG_GF)
print(' Extra-galactic point source foreground: %s' %FRG_EXGF)

# move to output path
os.chdir(path_out)

# --- Sky model ---
path_in = path_out #'/scratch/snx3000/mibianco/output_sdc3/dataLC_130923/'
params = toml.load('%sparams/par_i%d.toml' %(path_in, idx))
random_seed = params['user_params']['seed']

with fits.open('%slightcones/%s.fits' %(path_in, root_name[:root_name.rfind('dT')+2]), mode="readonly", memmap=True) as hdulist:
    hdr = hdulist[0].header
    Nx, Ny = hdr['NAXIS1'], hdr['NAXIS2']
    FoV = (abs(hdr['CDELT2']*Nx) * u.deg).to('rad')
    beam_sim = (FoV.value / Nx)**2     # from grid to point source
    RA, DEC = hdr['CRVAL1']*u.deg, hdr['CRVAL2']*u.deg
    
    freqs = hdr['CRVAL3']+np.arange(hdr['NAXIS3']) * hdr['CDELT3'] # Hz
    z = t2c.nu_to_z(freqs[idx_f]*1e-6)

    # add galactic foreground
    if(FRG_GF):
        zmax = t2c.nu_to_z(200.)
        fov_mpc = (Planck18.comoving_transverse_distance(zmax).value * FoV.value)
        data_gf = t2c.galactic_synch_fg(z=[z], ncells=Nx, boxsize=fov_mpc, rseed=random_seed//2)*1e-3 #* u.K    
        data = (hdulist[0].data[idx_f] + data_gf) * beam_sim
    else:
        data = hdulist[0].data[idx_f] * beam_sim

    w = WCS(hdr).celestial

# get coordinates
idx_ra, idx_dec = np.arange(0, Nx).reshape(Nx, 1), np.arange(0, Ny).reshape(1, Ny)
lon, lat = w.celestial.all_pix2world(idx_ra, idx_dec, 1)
sky_grid =  np.vstack((lon[np.newaxis, ...], lat[np.newaxis, ...])).reshape(2,lon.shape[0]*lon.shape[1]).T

freq = freqs[idx_f]

print(' FoV = %.2f %s' %(FoV.to('deg').value, FoV.to('deg').unit))
print(' RA, DEC = (%f, %f) %s' %(RA.value, DEC.value, DEC.unit))
print(' nu = %.5e Hz, z = %.3f (idx_f = %d)' %(freq, z, idx_f))

# convert K to Jy
Jy2kel = (u.Jy * cst.c * cst.c / (2 * cst.k_B * (freq*u.Hz)**2)).cgs
data *= (u.K / Jy2kel).cgs.value

if(PBEAM):
    # get and apply primary beam
    path_beam = '/store/ska/sk014/sdc3/station_beam.fits'
    beam_data = fits.getdata(path_beam)
    smooth_primary_beam = smoothing_grid(arr=beam_data[idx_f], noc=Nx, opt='mean')
    data *= smooth_primary_beam

# use only non-zero grid points
idx_nozero = np.nonzero(data.flatten())[0]

# create sky model with columns:
# RA [deg], Dec [deg], I [Jy], Q [Jy], U [Jy], V [Jy], ref freq [Hz], alpha, rot, maj ax [arcsec], min ax [arcsec], pos angle [deg], object ID
sky_data = np.zeros((idx_nozero.size, 13))
sky_data[:,:3] = np.hstack((sky_grid[idx_nozero, :], data.flatten()[idx_nozero, np.newaxis]))
sky_data[:,9] += 2*FoV.to('arcsec').value/Nx
sky_data[:,10] += 2*FoV.to('arcsec').value/Nx
sky = SkyModel(sky_data)

# add GLEAM point sources foreground
if(FRG_EXGF):
    path_point = '/store/ska/sk014/dataset_sdc3/inputs/frg/exgf/'
    
    if('point' in root_name):
        print(' Using SDC3 catalog')
        points_data = np.loadtxt(path_point+'rohit_sdc3cat_skymodel_4deg.txt')
    elif('gleam' in root_name):
        print(' Using GLEAM catalog')
        points_data = np.loadtxt(path_point+'gleamcat_skymodel_4deg.txt')

    # create inner & outter sky
    ra_wrap, dec_wrap = Angle([points_data[:,0], points_data[:,1]], unit='degree').wrap_at(180 * u.deg).deg
    inner_mask = np.sqrt((ra_wrap-RA.value)**2+(dec_wrap-DEC.value)**2) <= 2
    inner_sky = points_data[inner_mask]
    outter_mask = np.sqrt((ra_wrap-RA.value)**2+(dec_wrap-DEC.value)**2) > 2
    outter_sky = points_data[outter_mask]
    outter_sky[:,2] *= 1e-3

    #sky = SkyModel()
    sky.add_point_sources(inner_sky)
    sky.add_point_sources(outter_sky)

    #gauss_data = np.loadtxt(path_point+'trecscat_gauss_skymodel_4deg.txt')
    #ra_wrap, dec_wrap = Angle([gauss_data[:,0], gauss_data[:,1]], unit='degree').wrap_at(180 * u.deg).deg
    #inner_mask = np.sqrt((ra_wrap-RA.value)**2+(dec_wrap-DEC.value)**2) <= 2
    #inner_gauss_sky = gauss_data[inner_mask]
    #sky.add_point_sources(inner_gauss_sky)

    #diffuse_data = np.loadtxt(path_point+'trecscat_diffuse_skymodel_4deg.txt')
    #ra_wrap, dec_wrap = Angle([diffuse_data[:,0], diffuse_data[:,1]], unit='degree').wrap_at(180 * u.deg).deg
    #inner_mask = np.sqrt((ra_wrap-RA.value)**2+(dec_wrap-DEC.value)**2) <= 2
    #inner_diffuse_sky = diffuse_data[inner_mask]
    #sky.add_point_sources(inner_diffuse_sky)

# -----------------
path_telescope = '/users/mibianco/codes/sdc3/data/files/telescope.tm'
telescope = Telescope.read_from_file(path_telescope)

# HA between -2h to +2h, obs start at '2021-09-21 14:12:40.1'
t_start = datetime(2021, 9, 21, 14, 12, 40, 0)
t_obs = timedelta(hours=4, minutes=0, seconds=0, milliseconds=0)
t_day = timedelta(hours=4, minutes=0)
t_int = timedelta(seconds=10)
nr_tsteps = int(t_day.total_seconds() / t_int.total_seconds())
nr_days_obs = int(t_obs.total_seconds() / t_day.total_seconds())

print(' Simulating %d days observation\n time steps: %d' %(nr_days_obs, nr_tsteps))

observation_settings = Observation(phase_centre_ra_deg=RA.value,
                                    phase_centre_dec_deg=DEC.value,
                                    start_date_and_time=t_start,
                                    start_frequency_hz=freq,
                                    number_of_channels=1,
                                    number_of_time_steps=nr_tsteps,
                                    length=t_day)


if(ION):
    path_ion = '/scratch/snx3000/mibianco/output_sdc3/dataLC_130923/'
    screen_file = path_ion+'atmo/screen_4h_i0.npy'
    my_screen = np.load(file=screen_file, mmap_mode='r')

    iono_fits = screen_file.replace('.npy', '_ch%d.fits' %idx_f)
    r0, sampling = 7e3, 100.0
    if not (os.path.exists(iono_fits)):
        simulate_TEC(screens=my_screen, screen_width_metres=200e3, rate=0.1, bmax=20e3, sampling=sampling, num_times=nr_tsteps, frequency=freq, fits_filename=iono_fits)

    simulation = InterferometerSimulation(ms_file_path=run_name+'.MS',
                                            vis_path=run_name+'.vis',
                                            use_gpus=True, use_dask=False,
                                            channel_bandwidth_hz=1e5,
                                            noise_enable=False,
                                            ionosphere_fits_path=iono_fits,
                                            ionosphere_screen_type="External",
                                            ionosphere_screen_height_km=r0,
                                            ionosphere_screen_pixel_size_m=sampling,
                                            ionosphere_isoplanatic_screen=True)
else:
    simulation = InterferometerSimulation(ms_file_path=run_name+'.MS',
                                            vis_path=run_name+'.vis',
                                            use_gpus=True, use_dask=False,
                                            channel_bandwidth_hz=1e5,
                                            noise_enable=False)

visibilities = simulation.run_simulation(telescope, sky, observation_settings)

if(PLOT_IMG):
    # Make Image
    imager = Imager('single')
    imager.set(fov_deg=FoV.value, image_size=2048)
    imager.set(input_file=run_name+'.vis', output_root=run_name)
    output = imager.run(return_images=1)
    image = output["images"][0]

    # Plot
    fig, axs = plt.subplots(figsize=(10, 5), ncols=2, nrows=1, constrained_layout=True)
    axs[0].set_title('OSKAR image')
    im = axs[0].imshow(image, origin='lower', cmap="jet")
    plt.colorbar(im, ax=axs[0], pad=0.02, fraction=0.048)

    im = axs[1].imshow(data.T, origin='lower', cmap='jet')
    plt.colorbar(im, ax=axs[1], pad=0.02, fraction=0.048)
    plt.savefig(run_name+'_oskar.png', bbox_inches='tight')
    plt.clf()
