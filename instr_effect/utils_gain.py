import numpy as np
import colorednoise as cn

import dask.array as da
from daskms import xds_from_table

from numpy.random import default_rng
from numpy.typing import ArrayLike

import xarray as xr

def expand_gains(gains: xr.DataArray, antennas: ArrayLike) -> xr.DataArray:
    '''Expand gains along antenna dimension to be length of the number of baselines such that the position of the gain on the baseline axis corresponds to the antenna for that baseline.  
    
    Parameters:
    -----------
    gains: xr.DataArray
        The complex gains to expand.
    antennas: array_like
        Antenna numbers array the length of the number of baselines.
        
    Returns:
    --------
    gains_bl: xr.DataArray
    '''
    return gains.isel({'ant': antennas}).rename({'ant': 'bl'})

def apply_gains_to_vis(vis: xr.DataArray, gains: xr.DataArray, antenna1: ArrayLike, antenna2: ArrayLike) -> xr.DataArray:
    '''Apply complex gains to the visibilities as G_p V_pq G_q^H.
    
    Parameters:
    -----------
    vis: xr.DataArray
        Visibilities with no DI gains applied.
    gains: xr.DataArray
        Complex polarized gain matrix per time step, antenna and frequency channel.
    antenna1: array_like
        Antenna 1 index for the baselines.
    antenna2: array_like
        Antenna 2 index for the baselines.
        
    Returns:
    --------
    vis_mod: xr.DataArray
        Visibilities with gains applied.
    '''
    g1 = expand_gains(gains, antenna1)
    g2_conj = da.conj(expand_gains(gains, antenna2))
    g1_vis = da.einsum('...ij,...jk->...ik', g1, vis)
    g1_vis_g2_conj = da.einsum('...ik,...jk->...ij', g1_vis, g2_conj)
    
    return g1_vis_g2_conj


def calculate_SDC3_gains(timeSteps: int, numAntenna: int, numChannels: int, numBaselines: int, amplitudeMean=1, amplitudeSigma=2e-4, phaseMean=0, phaseSigma=0.02*np.pi/180, seed: int=123) -> xr.DataArray:
    rng0 = default_rng(seed=seed)
    rng1 = default_rng(seed=seed+1)

    # --- DIAntennaErrorSDC3A ---
    noiseExponent = 2 # 2 represents red/brown noise

    # for time amplitude
    if (timeSteps > 1):
        timeBrownNoiseArray = cn.powerlaw_psd_gaussian(exponent=noiseExponent, size=(numAntenna, timeSteps), random_state=rng0)
        timePhaseBrownNoiseArray = cn.powerlaw_psd_gaussian(exponent=noiseExponent, size=(numAntenna, timeSteps), random_state=rng1)

        timeAmplitudeArray = (timeBrownNoiseArray/ np.std(timeBrownNoiseArray) * amplitudeSigma) + (amplitudeMean - (amplitudeSigma /np.std(timeBrownNoiseArray) * np.mean(timeBrownNoiseArray) ))
        timePhaseArray = (timePhaseBrownNoiseArray/ np.std(timePhaseBrownNoiseArray) * phaseSigma) + (phaseMean - (phaseSigma /np.std(timePhaseBrownNoiseArray) * np.mean(timePhaseBrownNoiseArray) ))
    else: 
        timeAmplitudeArray = rng0.normal(loc=amplitudeMean, scale=amplitudeSigma, size=(numAntenna,1))
        timePhaseArray  = rng1.normal(loc=phaseMean, scale=phaseSigma, size=(numAntenna,1))

    timeArray = timeAmplitudeArray * np.exp(1j * 2 * np.pi * timePhaseArray)

    time3DArray = np.tile(timeArray, (numChannels, 1, 1)) # frequency, numAntenna, TimeSteps we want times, baselines, channels
    time3DArray = time3DArray.T # time, numAntennas, channels - but channels should be same
    
    # for frequency amplitude
    rng2 = default_rng(seed=seed+2)
    rng3 = default_rng(seed=seed+3)

    if (numChannels>1):
        frequencyBrownNoiseArray = cn.powerlaw_psd_gaussian(exponent=noiseExponent, size=(numAntenna, numChannels), random_state=rng2)
        frequencyPhaseBrownNoiseArray = cn.powerlaw_psd_gaussian(exponent=noiseExponent, size=(numAntenna, numChannels), random_state=rng3)

        frequencyAmplitudeArray = (frequencyBrownNoiseArray/np.std(frequencyBrownNoiseArray) * amplitudeSigma) + (amplitudeMean - (amplitudeSigma /np.std(frequencyBrownNoiseArray) * np.mean(frequencyBrownNoiseArray)))
        frequencyPhaseArray = (frequencyPhaseBrownNoiseArray/np.std(frequencyPhaseBrownNoiseArray) * phaseSigma) + (phaseMean - (phaseSigma /np.std(frequencyPhaseBrownNoiseArray) * np.mean(frequencyPhaseBrownNoiseArray)))
    else:
        frequencyAmplitudeArray = rng2.normal(loc=amplitudeMean, scale=amplitudeSigma, size=(numAntenna, 1))
        frequencyPhaseArray = rng3.normal(loc=phaseMean, scale=phaseSigma, size=(numAntenna, 1))
    
    # (1 + q_i) = (A_it * exp(2 * pi * 1j * (B_it))) * (A_if*exp(2* pi * 1j * (B_if))) ~ g
    frequencyArray = frequencyAmplitudeArray * np.exp(1j * 2 * np.pi * frequencyPhaseArray)

    frequency3DArray = np.tile(frequencyArray, (timeSteps, 1, 1)) # Timesteps, numAntenna, numFrequencies but times should be same

    DIError = time3DArray * frequency3DArray

    # --- DIBaselineErrorSDC3A ---
    baselineError = np.zeros((numBaselines, 0))
    for t in np.arange(timeSteps): 
        for f in np.arange(numChannels): # lazy implementation - try to do f, baselines together
            antennaError1, antennaError2 = np.meshgrid(DIError[t, :, f], DIError[t, :, f])
            timeFreqError = antennaError1 * np.conj(antennaError2)
            baselineError = np.append(baselineError, timeFreqError[np.triu_indices(numAntenna, k=1)].reshape(numBaselines, 1), axis=1)
    
    #baselineError = baselineError.reshape(-1, timeSteps, numChannels) # nBaselines, nTimes, nFreq
    #baselineError = np.moveaxis(baselineError, [0,1], [1,0]) # nTimes, nBaselines, nFreq
    baselineError = baselineError.reshape(-1, numChannels) # (nBaselines * nTimes), nFreq
    return baselineError


def apply_SDC3_gains(ms_input: str, seed: int=123) -> xr.DataArray:
    '''Read a Measurement set file and apply gains.
    
    Parameters:
    -----------
    ms_input: str
        Path to the measurement set file.
        
    Returns:
    --------
    vis_mod: xr.DataArray
        Visibilities with gains applied.
    '''
    ds = xds_from_table(ms_input, chunks={'row': 511*256*10, 'chan': 1, 'corr': 4, 'uvw': 3})[0]
    ds_spw = xds_from_table(ms_input+'::SPECTRAL_WINDOW')[0]
    
    time = ds.TIME.data
    a1 = ds.ANTENNA1.data
    a2 = ds.ANTENNA2.data
    freq = ds_spw.CHAN_FREQ[0].data
    
    times = da.asarray(da.unique(time).compute()).rechunk(10)
    antennas = da.asarray(da.unique(da.concatenate([a1, a2])).compute())

    n_time = times.size
    n_ant = antennas.size
    n_bl = n_ant * (n_ant-1) // 2
    n_freq = ds.DATA.sizes['chan']
    n_pol = ds.DATA.sizes['corr']
    n_row = ds.DATA.sizes['row']

    ant1 = a1.reshape(n_time, n_bl)[0].compute()
    ant2 = a2.reshape(n_time, n_bl)[0].compute()
    
    vis = xr.DataArray(ds.DATA.data.reshape(n_time, n_bl, n_freq, 2, 2), dims=('time', 'bl', 'chan', 'cor1', 'cor2'))
    
    gains = calculate_SDC3_gains(timeSteps=n_time, numAntenna=n_ant, numChannels=n_freq, numBaselines=n_bl, amplitudeMean=1, amplitudeSigma=2e-4, phaseMean=0, phaseSigma=0.02*np.pi/180, seed=seed)
    vis_mod = apply_gains_to_vis(vis, gains, ant1, ant2)
    
    vis_mod = xr.DataArray(vis_mod.reshape(n_row, n_freq, n_pol), dims=('row', 'chan', 'corr'))
    
    return vis_mod


def calculate_gains(times: ArrayLike, numAntenna: int, frequencyChannels: ArrayLike, antennaSigma: float=1e-4, bandpassSigma: float=1e-4, delaySigma: float=1e-10, statisticalSigma: float=1e-4, statisticalPhaseSigma: float=1e-5, seed: int=123) -> xr.DataArray:
    """
    Return (1 + q_nu_i) = ( 1 + p_i + h_nu + delta_nu_i) * e^(2*pi*1j(nu*tau_i + epsilon_vu_i)) ~ g
    V = V_theoretical * g[i, :] * g*[j, :]
    where "_nu" denotes a frequency dependent variable and "_i" denotes an antenna dependent variable

    the standard deviations of the following quantities are given by (typical values pre assigned):
    p_i = antennaSigma, 1e-4
    h_nu = bandpassSigma, 1e-4
    delta_nu_i = statisticalSigma, 1e-4
    nu = frequencies, 
    tau_i = delay_sigma, 1e-6 
    epsilon_nu_i = statisticalPhaseSigma, 1e-5
    """
    numTime = times.shape[0]
    numChannels = frequencyChannels.shape[0]
    frequencyChannels = frequencyChannels.astype(np.float32)
    rng = default_rng(seed=seed)
    
    # Construct a gain error array of shape (n_time, n_ant, n_chan, (x, y)) -> (n_time, n_ant, n_chan, ( (x, 0), (0, y) ) i.e. (n_time, n_ant, n_chan, 2, 2)
    antennaError = antennaSigma*rng.standard_normal(size=(numTime, numAntenna, 1, 2), dtype=np.float32)
    bandpassError = bandpassSigma*rng.standard_normal(size=(numTime, 1, numChannels, 2), dtype=np.float32)
    delayError = delaySigma*rng.standard_normal(size=(numTime, numAntenna, 1, 2), dtype=np.float32)
    statisticalError = statisticalSigma*rng.standard_normal(size=(numTime, numAntenna, numChannels, 2), dtype=np.float32)
    statisticalPhaseError = statisticalPhaseSigma*rng.standard_normal(size=(numTime, numAntenna, numChannels, 2), dtype=np.float32)

    totalAntennaError = (1 + antennaError + statisticalError) * np.exp(2 * np.pi * 1.j * (frequencyChannels[None,None,:,None] * delayError + statisticalPhaseError))
    # totalAntennaError = ( 1 + antennaError + bandpassError + statisticalError) * np.exp(2 * np.pi * 1.j * (frequencyChannels[None,None,:,None] * delayError + statisticalPhaseError))
    zeros = da.zeros_like(totalAntennaError[...,0])
    totalAntennaError = da.stack([totalAntennaError[...,0], zeros, zeros, totalAntennaError[...,1]], axis=3).reshape(numTime, numAntenna, numChannels, 2, 2)
    totalAntennaError = xr.DataArray(totalAntennaError, dims=('time', 'ant', 'chan', 'cor1', 'cor2')).chunk({'time': 10, 'ant': numAntenna, 'chan': 1, 'cor1': 2, 'cor2': 2})
    
    return totalAntennaError

def apply_gains(ms_input: str, antennaSigma: float=1e-4, bandpassSigma: float=1e-4, delaySigma: float=1e-10, statisticalSigma: float=1e-4, statisticalPhaseSigma: float=1e-5, seed: int=123) -> xr.DataArray:
    '''Read a Measurement set file and apply gains.
    
    Parameters:
    -----------
    ms_input: str
        Path to the measurement set file.
        
    Returns:
    --------
    vis_mod: xr.DataArray
        Visibilities with gains applied.
    '''
    ds = xds_from_table(ms_input, chunks={'row': 511*256*10, 'chan': 1, 'corr': 4, 'uvw': 3})[0]
    ds_spw = xds_from_table(ms_input+'::SPECTRAL_WINDOW')[0]
    
    time = ds.TIME.data
    a1 = ds.ANTENNA1.data
    a2 = ds.ANTENNA2.data
    freq = ds_spw.CHAN_FREQ[0].data
    
    times = da.asarray(da.unique(time).compute()).rechunk(10)
    antennas = da.asarray(da.unique(da.concatenate([a1, a2])).compute())

    n_time = times.size
    n_ant = antennas.size
    n_bl = n_ant * (n_ant-1) // 2
    n_freq = ds.DATA.sizes['chan']
    n_pol = ds.DATA.sizes['corr']
    n_row = ds.DATA.sizes['row']

    ant1 = a1.reshape(n_time, n_bl)[0].compute()
    ant2 = a2.reshape(n_time, n_bl)[0].compute()
    
    vis = xr.DataArray(ds.DATA.data.reshape(n_time, n_bl, n_freq, 2, 2), dims=('time', 'bl', 'chan', 'cor1', 'cor2'))
    gains = calculate_gains(times, n_ant, freq, antennaSigma, bandpassSigma, delaySigma, statisticalSigma, statisticalPhaseSigma, seed)
    vis_mod = apply_gains_to_vis(vis, gains, ant1, ant2)
    vis_mod = xr.DataArray(vis_mod.reshape(n_row, n_freq, n_pol), dims=('row', 'chan', 'corr'))
    
    return vis_mod
