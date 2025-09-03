import numpy as np, itertools, os
import cupy as cp
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import tools21cm as t2c

import astropy.units as u
import astropy.constants as cst

from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from tqdm import tqdm


# define a 2D gaussian function
def gaussian_2d(prefactor, x, y, mean, cov):
    """ a simple 2D gaussian distribution """
    x_diff = x - mean[0]
    y_diff = y - mean[1]
    inv_cov = np.linalg.inv(cov)
    exponent = -0.5 * (x_diff**2 * inv_cov[0, 0] + y_diff**2 * inv_cov[1, 1] + 2 * x_diff * y_diff * inv_cov[0, 1])
    return prefactor * np.exp(exponent)

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
    # empty visibilities
    visibilities = []
    processed_timesteps = set()

    # Load existing data if the file exists
    if os.path.exists(file_path):
        existing_uvw = np.load(file_path)
        existing_timesteps = existing_uvw.shape[0]
        processed_timesteps.update(range(existing_timesteps))
    else:
        existing_uvw = None

    # loop over timesteps
    for t in tqdm(range(start_timestep, end_timestep), desc='Calculate visibility: '):
        visls, max_uvw = vectorized_visibility_computation(uvw[t], sky_coord, freq, I_sky, beam_pattern, max_norm, chunk_size)
        visibilities.append(visls)
    
    if visibilities:
        visibilities = np.array(visibilities)  # Shape will be (new_timesteps, 130816)
        
        if existing_uvw is not None:
            visibilities = np.concatenate((existing_uvw, visibilities), axis=0)
        
        #np.save(file_path, visibilities)
        print(f"Processed and stored visibility for timesteps {start_timestep} to {end_timestep - 1}.")
    else:
        print(f"No new timesteps to process between {start_timestep} and {end_timestep}.")
    return visibilities
        
# --------------- Define Telescope ---------------
# Loading the telescope and station layout coords:
telescope_layout = np.loadtxt('skalow_AAstar_layout.txt') * u.m
station_layout = np.loadtxt('station_layout.txt') * u.m

N_ant = telescope_layout.shape[0]
N_B = int(N_ant*(N_ant-1)/2)

freq = 166. * u.MHz
#freq = 300. * u.MHz
lam = (cst.c / freq).to('m')
z = 1.42*u.GHz/freq - 1.
print(' frequency [MHz]:', freq)
print(' wavelength [m]:', lam)
print(' redshift:', z)

# get pair of baselines
pair_comb = list(itertools.combinations(range(N_ant), 2))

assert np.shape(pair_comb)[0] == N_B

uv_coord = np.empty((N_B, telescope_layout.shape[1]))
for i in tqdm(range(N_B), desc='Calculate baselines: '):
    ii, jj = pair_comb[i]
    
    # calculate the distance between antennas
    uv_coord[i] = (telescope_layout[ii] - telescope_layout[jj])/lam

# sort by increasing baseline distance
idx_sort = np.argsort(np.linalg.norm(uv_coord.squeeze(), axis=1))
pair_comb = np.array(pair_comb)[idx_sort]
uv_coord = uv_coord.squeeze()[idx_sort]

# add time axis TODO: now is just one timestep
uv_coord = uv_coord[None,...]

# Antenna base diameter
D = np.linalg.norm(station_layout, axis=1).max()*2
print(' dish diameter [m]:', D)
print(20*'-')

#--------------------------------------------------
theta_fwhm = (1.03 * lam / D) * u.rad
theta_0 = 0.6 * theta_fwhm
V_0 = (np.pi * theta_0**2 / 2).to('sr')
N_pix = 256
print(' theta_fwhm:', theta_fwhm)
print(' theta_0:', theta_0)
print(' V_0:', V_0)
print(20*'-')

# --------------- Define Observational Quantities ---------------
FoV = np.sqrt(V_0).to('rad')
L_box = FoV.to('rad').value * (1+z)*cosmo.angular_diameter_distance(z)
dthet = L_box / ((1+z)*cosmo.angular_diameter_distance(z)*N_pix) * u.rad
print(' FoV:', FoV)
print(' Boxsize:', L_box)
print(' dthet:', dthet)
print(20*'-')


#--------------------------------------------------
# define a 1D sky and get RA coordinate
thet = np.linspace(-FoV/2, FoV/2, N_pix).to('rad').value
# Create a grid of points (we ignore third dimension, i.e. n-axis)
l_coord, m_coord = np.meshgrid(thet, thet)
lmn_coord = np.dstack((l_coord, m_coord)).reshape(-1, 2)
# Create a beam pattern
beam_pattern = np.exp(-((l_coord**2 + m_coord**2) / theta_0.value**2))

visibility_file_path = 'test.npy'

# --------------- Define Sky Model ---------------
# Define parameters for the 2D Gaussian source
mu = np.array([0.005, -0.004])  # Mean
sigma = np.array([[(8e-1*u.arcsec).to('rad').value, 0], [0, (1e-1*u.arcsec).to('rad').value]])  # Covariance matrix

# Get the sky model
#dT_jy = np.zeros((N_pix, N_pix))
#for ix in range(10):
    #dT_jy[N_pix//2+ix, N_pix//2+ix] = 1e-3
    #dT_jy[N_pix//2, N_pix//2+ix] = 1e-3
#dT_jy = gaussian_2d(prefactor=1e-3, x=l_coord, y=m_coord, mean=mu, cov=sigma)
dT_jy = galactic_synch_fg_custom(z=[z], ncells=N_pix, boxsize=L_box.value, rseed=918)
#dT_jy = np.random.normal(loc=1e-3, scale=1e-4, size=(N_pix, N_pix))

#Running the function
vis = process_timesteps(uv_coord, 0, 1, visibility_file_path, thet, z, dT_jy, beam_pattern, None, 128)

# Define visibilities matrix
V_matrix = np.zeros((N_ant, N_ant))

pair_comb_sort = list(itertools.combinations(range(N_ant), 2))

for i_b in range(N_B):
    ii, jj = pair_comb_sort[i_b]

    # store visibility matrix
    V_matrix[ii, jj] = np.abs(vis[0, i_b])

# pairs are only for one corner
V_matrix = V_matrix + np.conj(V_matrix.T)

# plot visibility matrix
fig, axs = plt.subplots(figsize=(12, 5), ncols=2, nrows=1, constrained_layout=True)
axs[0].set_title('Sky Model')
im = axs[0].pcolormesh((l_coord*u.rad).to('arcmin').value, (m_coord*u.rad).to('arcmin').value, dT_jy, vmin=0., vmax=0.002, cmap='viridis')
axs[0].set_xlabel(r'l [arcmin]'), axs[0].set_ylabel(r'm [arcmin]')
plt.colorbar(im, ax=axs[0], label='I [Jy]', pad=0.01)

axs[1].set_title('Visibility Matrix')
im = axs[1].imshow(V_matrix, origin='lower', cmap='viridis', norm=LogNorm(vmin=1e-3, vmax=10))
axs[1].set_xlabel(r'Id$_\mathrm{ant}$'), axs[1].set_ylabel(r'Id$_\mathrm{ant}$')
plt.colorbar(im, ax=axs[1], label='V [Jy]', pad=0.01)
plt.show(), plt.clf()