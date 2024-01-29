import os
import pathlib
import shutil
from typing import Union

import dask
import dask.array as da
import xarray as xr
from daskms import xds_from_ms, xds_to_table
from numpy.random import default_rng
import numpy as np

def read_SEFD(SEFD_path: str) -> xr.DataArray:
    '''Read the SEFD text file downloaded from the SEFD calculator for SKA low
    http://skalowsensitivitybackup-env.eba-daehsrjt.ap-southeast-2.elasticbeanstalk.com/sensitivity_radec_vs_freq/ 
    
    Parameters:
    -----------
    SEFD_path: str
        Path to the SEFD file.
        
    Returns:
    --------
    SEFD: xr.DataArray
        SEFD values in Jy along frequency coordinates in Hz
    '''
    SEFD_freqs, sens_x, SEFD_X, sens_y, SEFD_y, sens_I, SEFD_I = np.loadtxt(SEFD_path, skiprows=2).T
    SEFD = xr.DataArray(SEFD_I, coords={'chan': SEFD_freqs*1e6})
    return SEFD

def get_t_int_and_chan_specs(ms_path: str = None, ms_ds: xr.Dataset = None) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Read the integration time per time step and channel width from a Measurement Set (MS).

    Parameters:
    -----------
    ms_file: str
        Path to Measurement set.

    Returns:
    --------
    t_int: float
        Integration time per time step in seconds.
    chan_freq: xr.DataArray (n_chan,)
        Channel frequencies in Hz.
    chan_width: xr.DataArray (n_chan,)
        Channel widths in Hz.
    """
    # Get the integration time, frequencies and channel widths
    if isinstance(ms_ds, xr.Dataset):
        t_int = ms_ds.INTERVAL[0]
    else:
        t_int = xds_from_ms(ms_path)[0].INTERVAL[0]
    ds_spw = xds_from_ms(
        os.path.join(ms_path, "::SPECTRAL_WINDOW"), group_cols="__row__"
    )[0]
    chan_freq = ds_spw.CHAN_FREQ[0].rename({"CHAN_FREQ-1": "chan"})
    chan_width = ds_spw.CHAN_WIDTH[0].rename({"CHAN_WIDTH-1": "chan"})
    return t_int, chan_freq, chan_width


def get_sigma_vis(SEFD: xr.DataArray, chan_freqs: xr.DataArray, chan_width: Union[float, xr.DataArray], t_int: float) -> xr.DataArray:
    """Calculate the standard deviation for normally distributed noise to visibilities for real and imaginary parts.
    Following OSKAR/oskar/vis/src/oskar_vis_block_add_system_noise.c lines 61-101.

    Parameters:
    -----------
    SEFD: xr.DataArray (n_chan1,)
        System equivalent flux density over frequency in Jansky. Assumes the SEFD is the same acrosss all antennas.
    chan_freqs: array_like (n_chan2,)
        Frequencies for the channels in Hz.
    chan_width: float | array_like (n_chan2,)
        Channel width in Hz.
    t_int: float
        Integration time for a time step in seconds.

    Returns:
    --------
    sigma: array_like (n_chan2,)
        Noise standard deviation over frequency in Jy.
    """
    # Interpolate SEFD to channel frequencies
    SEFD = SEFD.interp(chan=chan_freqs)
    # Calculate the noise standard deviation
    sigma = SEFD / da.sqrt(2.0 * t_int * chan_width)
    return sigma


def add_noise_to_vis(noise_std: xr.DataArray, vis: xr.DataArray, seed: int = 0) -> xr.DataArray:
    """Add complex normally distributed noise to visibilities.

    Parameters:
    -----------
    noise_std: array_like (n_chan,)
        Standard deviation of the complex noise in the real and imaginary components.
    vis: array_like (n_row, n_chan, n_corr)
        Complex visibilities to add noise to.
    seed: int
        Random seed.

    Returns:
    --------
    noisy_vis: array_like
        Visibilities with noise added.
    """
    # Set the random seed
    rng = default_rng(seed=seed)
    # Expand the noise standard deviation to match the shape of the visibilities
    out_shape = vis.shape
    noise_std = noise_std.expand_dims(dim=("row", "corr"), axis=(0, 2))
    # Add noise to the visibilities
    noise = noise_std * (rng.standard_normal(size=out_shape, dtype=np.float32) + 1.0j * rng.standard_normal(size=out_shape, dtype=np.float32))
    noisy_vis = vis + noise
    return noisy_vis

def add_noise(vis: xr.DataArray, SEFD: xr.DataArray, ms_path: str = None, ms_ds: xr.Dataset = None, seed: int = 0) -> xr.DataArray:
    '''Add noise to the visibilities according to the SEFD at specific frequencies.
    
    Parameters:
    -----------
    ms_path: str
        Path to the MS file to read frequencies, channel widths and integration times.
    vis: xr.DataArray
        Noiseless visibilities to add noise to.
    SEFD: xr.DataArray
        SEFD values at specic frequency values. Frequency coordinates must be defined under "chan".
    seed: int
        Random seed to use for the noise generation.
    '''
    # Get the integration time and channel specifications
    t_int, chan_freqs, chan_width = get_t_int_and_chan_specs(ms_path, ms_ds)
    # Calculate the noise standard deviation
    noise_std = get_sigma_vis(SEFD, chan_freqs, chan_width, t_int)
    # Add noise to the visibilities
    noisy_vis = add_noise_to_vis(noise_std, vis, seed)
    
    return noisy_vis


def get_vis(ms_file: str, type_data: str) -> xr.DataArray:
    """Get the visibility Dataset from the Measurement Set file.

    Parameters:
    -----------
    ms_file: str
        Path of the MS file.
    """
    # Get the visibility data
    if(type_data == 'DATA'):
        vis = xds_from_ms(ms_file)[0].DATA
    else:
        vis = xds_from_ms(ms_file)[0].MODEL_DATA
    return vis


def write_vis(ms_ds: xr.Dataset, ms_file: str, vis: xr.DataArray, col: str="MODEL_DATA"):
    """Write visibilitity data to Measurement Set.

    Parameters:
    -----------
    ms_file: str
        Path of the MS file to write to.
    vis: xr.Dataset ("row", "chan", "corr")
        Visibility data to write to MS file.
    col: str
        Column in MS file to write to.
    """
    ms_ds = ms_ds.assign({col: vis})
    
    # Write the visibility data to the MS file
    writes = xds_to_table(ms_ds, ms_file, [col])
    dask.compute(writes)


def duplicate(ms_file_path: str, new_path: str = "", overwrite: bool = True):
    """Duplicate a file of directory tree such as a Measurement Set.

    Parameters:
    -----------
    ms_file: str
        Path of the MS file to duplicate.
    new_path: str
        Path to new MS file location and name. Deault is "old_path/oldname_new.ms".
    overwrite: bool
        Overwrite the new_path directory.
    """
    # Set the new path
    if new_path == "":
        path = pathlib.Path(ms_file_path)
        new_path = f"{os.path.join(path.parent, path.stem)}_new{path.suffix}"
        print(new_path)
    # Duplicate the MS file
    if overwrite:
        shutil.rmtree(new_path, ignore_errors=True)
    shutil.copytree(ms_file_path, new_path)
    