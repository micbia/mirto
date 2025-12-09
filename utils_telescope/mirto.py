import numpy as np, itertools, os
import cupy as cp
import itertools

import astropy.units as u
import astropy.constants as cst

from astropy.io import fits
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from tqdm import tqdm

class MIRTO():
    def __init__(self, N_ant, wavelength, layout, lmn, RA, Dec, I_sky, flat_sky=False):
        self.N_ant = N_ant
        self.wavelength = wavelength
        self.layout = layout
        self.lmn = cp.asarray(lmn)
        self.RA, self.Dec = RA, Dec

        self.I_sky = cp.asarray(I_sky)
        
        self.flat_sky = flat_sky
        """
        # Criterion for flat-sky validity
        uv_max = cp.max(uvw_norms)
        l_max = cp.max(np.abs(lmn[:,0]))
        m_max = cp.max(np.abs(lmn[:,1]))
        phase_error = cp.pi * uv_max * (l_max**2 + m_max**2)
        flat_sky = phase_error < 0.1  # threshold ~0.1 rad phase error

        if flat_sky:
            print(" Using FLAT-SKY approximation (phase error %.3e rad)" %phase_error)
        else:
            print(" Using FULL-SKY model (phase error %.3e rad)" %phase_error)    
        """

        # get number of baselines (pair  of antennas)
        self.N_B = int(self.N_ant*(self.N_ant-1)/2)


    def earth_rotation_effect(self, HA, delta):
        """ Earth Rotation matrix calculation"""
        HA, delta = HA.to('rad').value, delta.to('rad').value
        
        return cp.array([[np.sin(HA), np.cos(HA), 0], 
                        [-np.sin(delta)*np.cos(HA), np.sin(delta)*np.sin(HA), np.cos(delta)],
                        [np.cos(delta)*np.cos(HA), -np.cos(delta)*np.sin(HA), np.sin(delta)]])


    def get_baselines(self):
        pair_comb = list(itertools.combinations(range(self.N_ant), 2))

        assert np.shape(pair_comb)[0] == self.N_B

        uv_coord = np.empty((self.N_B, self.layout.shape[1]))
        for i in tqdm(range(self.N_B), desc='Calculate baselines: '):
            ii, jj = pair_comb[i]
            
            # calculate the distance between antennas
            uv_coord[i] = (self.layout[ii] - self.layout[jj])/self.wavelength

        # sort by increasing baseline distance
        uv_length = np.linalg.norm(uv_coord.squeeze(), axis=1)
        idx_sort = np.argsort(uv_length)
        pair_comb = np.array(pair_comb)[idx_sort]
        uv_coord = uv_coord.squeeze()[idx_sort]

        return uv_coord, uv_length


    def compute_visibility(self, uvw, beam_pattern=None, max_norm=None, chunk_size=1024):
        """
        Compute interferometric visibilities, automatically deciding whether to
        use the flat-sky approximation or full-sky model.

        Parameters
        ----------
        uvw : (N_baselines, 3) array
            Baselines in metres.
        lmn : (N_pixel, 2) arrays
            Direction cosines or small-angle radians (if small FOV).
        I_sky, beam_pattern : 2D arrays
            Sky brightness and primary beam pattern on (l,m) grid.
        max_norm : float or None
            Optional cutoff in |uvw| (wavelengths).
        chunk_size : int
            Number of baselines per processing chunk.

        Returns
        -------
        vis : complex numpy array
            Computed visibilities.
        uvw_norms : float
            Baselines norm (after max_norm).
        uvw : bool
            Baselines employed in the calculation (after max_norm).
        """
        # Convert to CuPy arrays
        if type(uvw) is not type(cp.array([])):
            uvw = cp.asarray(uvw)
        else:
            pass

        Nx, Ny = self.I_sky.shape
        assert Nx == Ny

        # pixel-wise product between beam and sky model
        if beam_pattern is not None:
            if type(beam_pattern) is not type(cp.array([])):
                beam_pattern = cp.asarray(beam_pattern)
            img = (self.I_sky * beam_pattern).ravel()
        else:
            img = self.I_sky.ravel()
        
        # uvw norm
        uvw_norms = cp.linalg.norm(uvw, axis=1)
        
        if(self.flat_sky):
            l_coord, m_coord = self.lmn[:,0].ravel(), self.lmn[:,1].ravel()
        else:
            n_grid = cp.sqrt(1 - self.lmn[:,0]**2 - self.lmn[:,1]**2)
            l_coord, m_coord, n_coord = self.lmn[:,0].ravel(), self.lmn[:,1].ravel(), n_grid.ravel()
        
        # TODO: maybe to keep the full cp.diff instead than using just the first value?
        dl = (l_coord.max()-l_coord.min())/Nx
        dm = (m_coord.max()-m_coord.min())/Nx

        # Area element for integration
        pixel_area = dl * dm

        # filters uvw data if a maximum baseline is give
        if max_norm is not None:
            mask_baseline = cp.where(uvw_norms < max_norm)[0]
            uvw = uvw[mask_baseline]
            uvw_norms = uvw_norms[mask_baseline]

        vis = cp.empty(len(uvw), dtype=cp.complex128)
        for i0 in range(0, len(uvw), chunk_size):
            i1 = min(i0 + chunk_size, len(uvw))
            chunk_uvw = uvw[i0:i1]
            u = chunk_uvw[:, 0][:, None]
            v = chunk_uvw[:, 1][:, None]
            w = chunk_uvw[:, 2][:, None]

            if(self.flat_sky):
                # to deal with the quadratic phase correction (first order Taylor expansion) when flat-sky approximation break down
                fresnel = cp.exp(1j * cp.pi * w * (l_coord**2 + m_coord**2))
                phase = (u * l_coord + v * m_coord)        
                fringe = cp.exp(-2j * cp.pi * phase) * fresnel
            else:
                phase = (u * l_coord + v * m_coord + w * (n_coord - 1))
                fringe = cp.exp(-2j * cp.pi * phase)

            vis[i0:i1] = cp.dot(fringe, img) * pixel_area

        return cp.asnumpy(vis), cp.asnumpy(uvw_norms), cp.asnumpy(uvw)


    def visibility_timesteps(self, h_angle, uvw, beam_pattern, max_norm=None, chunk_size=1024):
        
        for i_t, h in enumerate(h_angle):
            # comput visibility
            vis, uvw_len, uvw = self.compute_visibility(self, uvw, beam_pattern=None, max_norm=None, chunk_size=chunk_size)

            # calculate rotation for next time step
            rotation_matrix = self.earth_rotation_effect(HA=h, delta=self.Dec)
            
            # rotate uvw coordinates for next time step
            uwv = cp.dot(rotation_matrix, uvw.T).T

        return vis

"""
def visibility_timesteps(uvw, start_timestep, end_timestep, file_path, sky_coord, freq, I_sky, beam_pattern, max_norm, chunk_size):
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

"""