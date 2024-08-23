import os, numpy as np, os
import matplotlib.pyplot as plt
import tools21cm as t2c
import cupy as cp
import itertools
import time

import astropy.units as u
import astropy.constants as cst

from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from tqdm import tqdm

def vectorized_visibility_computation(uvw, sky_coord, freq, I_sky, beam_pattern, max_norm, chunk_size):
    # Convert to CuPy arrays
    uvw = cp.asarray(uvw)
    sky_coord = cp.asarray(sky_coord)
    I_sky = cp.asarray(I_sky)
    beam_pattern = cp.asarray(beam_pattern)
    
    l_coord, m_coord = cp.meshgrid(sky_coord, sky_coord)
    
    uvw_norms = cp.linalg.norm(uvw, axis=1)
    
    if max_norm is not None:
        valid_uvw_ids = cp.where(uvw_norms < max_norm)[0]
        filter_uvw = uvw[valid_uvw_ids]
        filter_uvw_norms = uvw_norms[valid_uvw_ids]
    else:
        valid_uvw_ids = cp.arange(len(uvw))
        filter_uvw = uvw
        filter_uvw_norms = uvw_norms
    
    l_coord = l_coord.flatten()
    m_coord = m_coord.flatten()
    I_sky = I_sky.flatten()
    beam_pattern = beam_pattern.flatten()
    
    visibility_list = cp.empty(len(valid_uvw_ids), dtype = cp.complex128)
    
    # Process in chunks
    for start in range(0, len(valid_uvw_ids), chunk_size):
        end = min(start + chunk_size, len(valid_uvw_ids))
        chunk_uvw = filter_uvw[start:end]
        
        fringe = cp.exp(-2j * cp.pi * (chunk_uvw[:,0,None] * l_coord + chunk_uvw[:,1,None] * m_coord))
        vis_chunk = cp.dot(fringe, I_sky * beam_pattern)
        visibility_list[start:end] = vis_chunk
    
    return cp.asnumpy(visibility_list), cp.asnumpy(cp.max(filter_uvw_norms))



def process_timesteps(uvw, start_timestep, end_timestep, file_path, sky_coord, freq, I_sky, beam_pattern, max_norm, chunk_size):
    visibilities = []
    processed_timesteps = set()

    # Load existing data if the file exists
    try:
        existing_uvw = np.load(file_path)
        existing_timesteps = existing_uvw.shape[0]
        processed_timesteps.update(range(existing_timesteps))
    except FileNotFoundError:
        existing_uvw = None
    
    for t in range(start_timestep, end_timestep):
        if t in processed_timesteps:
            print(f"Timestep {t} already processed. Skipping.")
            continue
        
        start_time = time.time()
        visls, max_uvw = vectorized_visibility_computation(uvw[t], sky_coord, freq, I_sky, beam_pattern, max_norm, chunk_size)
        end_time = time.time()
        print(f"Processed timestep {t}, took {round(end_time-start_time, 3)} seconds.")
        visibilities.append(visls)
    
    if visibilities:
        visibilities = np.array(visibilities)  # Shape will be (new_timesteps, 130816)
        
        if existing_uvw is not None:
            visibilities = np.concatenate((existing_uvw, visibilities), axis=0)
        
        np.save(file_path, visibilities)
        print(f"Processed and stored visibility for timesteps {start_timestep} to {end_timestep - 1}.")
    else:
        print(f"No new timesteps to process between {start_timestep} and {end_timestep}.")

        

def galactic_synch_fg_custom(z, ncells, boxsize, rseed=False):
    if(isinstance(z, float)):
        z = np.array([z])
    else:
        z = np.array(z, copy=False)
    gf_data = np.zeros((ncells, ncells, z.size))

    if(rseed): np.random.seed(rseed)
    X  = np.random.normal(size=(ncells, ncells))
    Y  = np.random.normal(size=(ncells, ncells))
    nu_s,A150,beta_,a_syn,Da_syn = 150,513,2.34,2.8,0.1

    for i in range(0, z.size):
        nu = t2c.cosmo.z_to_nu(z[i])
        U_cb  = (np.mgrid[-ncells/2:ncells/2,-ncells/2:ncells/2]+0.5)*t2c.cosmo.z_to_cdist(z[i])/boxsize
        l_cb  = 2*np.pi*np.sqrt(U_cb[0,:,:]**2+U_cb[1,:,:]**2)
        #C_syn = A150*(1000/l_cb)**beta_*(nu/nu_s)**(-2*a_syn-2*Da_syn*np.log(nu/nu_s))
        C_syn = A150*(1000/l_cb)**beta_
        #C_syn = A150
        solid_angle = boxsize**2/t2c.cosmo.z_to_cdist(z[i])**2
        AA = np.sqrt(solid_angle*C_syn/2)
        T_four = AA*(X+Y*1j) * np.sqrt(2)
        T_real = np.abs(np.fft.ifft2(T_four))   #in Jansky
        #gf_data[..., i] = t2c.jansky_2_kelvin(T_real*1e6, z[i], boxsize=boxsize, ncells=ncells)
        gf_data[..., i] = T_real
    return gf_data.squeeze()



# Loading the uvw coords:
#all_uvw = np.load('bl_lc_256_train_130923_i0_dT_ch600_4h1d_256.npy', mmap_mode= 'r')
all_uvw = np.load('bl_MWA_syngf_ch600_1024.npy', mmap_mode= 'r')


freq = 166000000
lam = cst.c.value / freq
z = t2c.nu_to_z(freq/ 1e6)
#D = 37.315 #Antenna base diameter
D = 4.51
print('frequency [Hz]:', freq)
print('wavelength [m]:', lam)
print('redshift:', z)
print('dish diameter [m]:', D)
print(20*'-')
#--------------------------------------------------
theta_fwhm = (1.03 * lam / D) * u.rad #D is the diameter of the antenna base
theta_0 = 0.6 * theta_fwhm
V_0 = (np.pi * theta_0**2 / 2).to('sr')
N_pix = 1024
print("theta_fwhm:", theta_fwhm)
print("theta_0:", theta_0)
print("V_0:", V_0)
print(20*'-')
#--------------------------------------------------

#--------------------------------------------------
FoV = np.sqrt(V_0).to('deg')
L_box = FoV.to('rad').value * (1+z)*cosmo.angular_diameter_distance(z)
dthet = L_box / ((1+z)*cosmo.angular_diameter_distance(z)*N_pix) * u.rad
print('FoV:', FoV)
print('Boxsize:', L_box)
print('dthet:', dthet)
print(20*'-')
#--------------------------------------------------


dT_jy = galactic_synch_fg_custom(z=z, ncells=N_pix, boxsize=L_box.value, rseed=918)


#--------------------------------------------------
# define a 1D sky and get RA coordinate
thet = np.linspace(-FoV/2, FoV/2, N_pix).to('rad').value
# Create a grid of points (we ignore third dimension, i.e. n-axis)
l_coord, m_coord = np.meshgrid(thet, thet)
lmn_coord = np.dstack((l_coord, m_coord)).reshape(-1, 2)
# Create a beam pattern
beam_pattern = np.exp(-((l_coord**2 + m_coord**2) / theta_0.value**2))
#beam_pattern = np.ones((N_pix, N_pix))


visibility_file_path = 'MWA_all_visibilities_virgin_4.51diam_beam.npy'
#visibility_file_path = 'SKA_all_visibilities_virgin_37.315diam_nobeam_fullsynchfunc.npy'

#Running the function
start_time = time.time()
process_timesteps(all_uvw, 0, 360, visibility_file_path, thet, z, dT_jy, beam_pattern, None, 128)
end_time = time.time()
print(f"Total time: {round(end_time-start_time, 3)} seconds.")