import numpy as np, sys
from time import time
from datetime import timedelta
import colorednoise as cn
import matplotlib.pyplot as plt

def DIAntennaErrorSDC3A(numAntenna, frequencyChannels, timeSteps, amplitudeMean=1, amplitudeSigma=2e-4,phaseMean=0, phaseSigma=0.02*np.pi/180):
    """
    As per the document, 
    "We have constructed a gain model for the telescope array which represents the amplitude and phase response of each of the 512 stations for every 10 second time interval and every frequency channel of the observation.We have fixed the mean amplitude response at unity and the mean phase response at zero, so that there are no systematic calibration errors. However, each time and frequency interval is assigned a value taken from an error distribution. The time and frequency fluctuations are assumed to be independent of each other and the gain error at each specific (time, frequency) is given by the complex product of the two, one-dimensional distributions. The one-dimensional gain distributions in each of time and frequency are first assigned random values from a Gaussian distribution with a specified standard deviation in each of amplitude and phase. We then make use of the “colorednoise” python code [RD10] to take these “white-noise” distributions that have equal fluctuation power on all sampled scales and produce “red-noise” distributions with a -2 power-law index, whereby the greatest  fluctuation power is present on the longest time intervals and over the largest frequency increments.Both the standard deviation and mean values are preserved in this process of coloring the noise. The resulting fluctuations are reminiscent of gain error patterns that are encountered in actual observations. In assigning a numerical value to the standard deviation of the DI amplitude and phase errors we have considered both what might be realistically achieved for the calibration quality of a single observation as well as a plausible degree of
    improvement that might be achieved by averaging the outcomes of a large number of independent observations. Specifically, we have simulated a continuous four-hour duration tracking observation. However, we are using this to simulate the outcome of a 1000 hour deep integration, which would be achieved in practice by carrying out 250 repetitions of such a four-hour track. The final post-calibration and post-averaging standard deviations we have assumed to apply are 0.02 degrees in phase and 0.02% in amplitude for each of the time and the frequency domains. The time domain represents residual broadband gain calibration errors and the frequency domain represents residual bandpass calibration errors."

    Amplitude and phase for EVERY 10 second time interval and every frequency channcel

    A e(i*theta)
    A drawn from gaussian with mean 1, theta drawn from gaussian with mean 0

    A_t e(i* theta_t) & A_f e(t*theta_f)

    time and frequency independent of one another specific (time, frequency) given by complex product of 2 1d distributions. 


    Return (1 + q_i) = (A_it*exp(2 * pi * 1j * (B_it))) * (A_if*exp(2* pi * 1j * (B_if))) ~ g
    V = g[i] @ V_theoretical @ g_j*[j]
    where "_i" denotes an antenna dependent variable
    g is per antenna or per station per time step and per frequency
    
    A_it/f is drawn from a gaussian of mean 1 and sigma 0.0002
    B_it/f is drawn from a gaussian of mean 0 and sigma 0.02 * pi/180

    Futher the individual samples of A_i and B_i are correlated in frequency and time (red noise NOT white noise)
    
    """
    noiseExponent = 2 # 2 represents red noise

    if (timeSteps > 1):

        timeBrownNoiseArray = cn.powerlaw_psd_gaussian(noiseExponent, (numAntenna, timeSteps)) # for time amplitude
        
        timePhaseBrownNoiseArray = cn.powerlaw_psd_gaussian(noiseExponent, (numAntenna, timeSteps))

        timeAmplitudeArray = (timeBrownNoiseArray/ np.std(timeBrownNoiseArray) * amplitudeSigma) + (amplitudeMean - (amplitudeSigma /np.std(timeBrownNoiseArray) * np.mean(timeBrownNoiseArray) ))
        timePhaseArray = (timePhaseBrownNoiseArray/ np.std(timePhaseBrownNoiseArray) * phaseSigma) + (phaseMean - (phaseSigma /np.std(timePhaseBrownNoiseArray) * np.mean(timePhaseBrownNoiseArray) ))
    else: 
        timeAmplitudeArray = np.random.normal(loc=amplitudeMean, scale=amplitudeSigma, size=(numAntenna,1))
        timePhaseArray  = np.random.normal(loc=phaseMean, scale=phaseSigma, size=(numAntenna,1))

    timeArray = timeAmplitudeArray * np.exp(1j * 2 * np.pi * timePhaseArray)

    time3DArray = np.tile(timeArray, (frequencyChannels.shape[0], 1, 1)) # frequency, numAntenna, TimeSteps we want times, baselines, channels
    time3DArray = time3DArray.T # time, numAntennas, channels - but channels should be same

    if (frequencyChannels.shape[0]>1):

        frequencyBrownNoiseArray = cn.powerlaw_psd_gaussian(noiseExponent, (numAntenna,frequencyChannels.shape[0])) # for frequency amplitude
        frequencyPhaseBrownNoiseArray = cn.powerlaw_psd_gaussian(noiseExponent, (numAntenna,frequencyChannels.shape[0]))

        frequencyAmplitudeArray = (frequencyBrownNoiseArray/np.std(frequencyBrownNoiseArray) * amplitudeSigma) + (amplitudeMean - (amplitudeSigma /np.std(frequencyBrownNoiseArray) * np.mean(frequencyBrownNoiseArray) ))
        frequencyPhaseArray = (frequencyPhaseBrownNoiseArray/np.std(frequencyPhaseBrownNoiseArray) * phaseSigma) + (phaseMean - (phaseSigma /np.std(frequencyPhaseBrownNoiseArray) * np.mean(frequencyPhaseBrownNoiseArray) ))

    else:
        frequencyAmplitudeArray = np.random.normal(loc = amplitudeMean, scale = amplitudeSigma, size = (numAntenna, 1))
        frequencyPhaseArray = np.random.normal(loc = phaseMean, scale = phaseSigma, size = (numAntenna, 1))
    
    
    frequencyArray = frequencyAmplitudeArray * np.exp(1j * 2 * np.pi * frequencyPhaseArray)

    frequency3DArray = np.tile(frequencyArray, (timeSteps, 1, 1)) # Timesteps, numAntenna, numFrequencies but times should be same

    DIError = time3DArray* frequency3DArray   
    '''
    #Plotting test:
    #fig, ax = plt.subplots(2,2, figsize = (20,20)) # make 2, 3 when histogram to be plotted - possibly need to multiply values by 1000 or 100 before sorting
    fig,ax = plt.subplots(2, 1, figsize = (15,15)) # make 2, 3 when histogram to be plotted - possibly need to multiply values by 1000 or 100 before sorting
    ax = ax[None,:]

    for i in np.arange(numAntenna):
        ax[0,0].plot(np.arange(timeArray.shape[-1]), timeArray[i, :])
        #ax[0,0].set_title("Time Series")
        ax[0,0].set_xlabel(r"$t_{obs}$ [$\times 10$ s]", size=18), ax[0,0].set_ylabel(r"$q_{\nu, i}$", size=18)
        ax[0,0].tick_params(axis='x', labelsize=18)
        ax[0,0].tick_params(axis='y', labelsize=18)
        f = np.fft.rfftfreq(timeArray.shape[-1])
        ax[0,1].loglog(f, abs(np.fft.rfft(timeArray[i, :])))# FFT of original time series
        #ax[0,1].set_title(f"Time Series FFT", size=18)
        ax[0,1].set_xlabel(r"$Frequency$ [Hz]", size=18), ax[0,1].set_ylabel(r"Power Spectral Density [dB]", size=18)
        ax[0,1].set_ylim(5e-6, 1e1)
        ax[0,1].tick_params(axis='x', labelsize=18)
        ax[0,1].tick_params(axis='y', labelsize=18)

        #ax[0,2].hist(timeArray * 100, bins = 10)
        #ax[0,2].set_title("Time Histogram")
        """
        ax[1,0].plot(np.arange(frequencyArray.shape[-1]), frequencyArray[i, :])
        ax[1,0].set_title("Frequency Series")

        f = np.fft.rfftfreq(frequencyArray.shape[-1])
        ax[1,1].loglog(f, abs(np.fft.rfft(frequencyArray[i, :])))# FFT of original time series
        ax[1,1].set_title(f"Frequency Series FFT")

        """
        #ax[1,2].hist(frequencyArray * 100, bins = 10)
        #ax[1,2].set_title("Frequency Histogram")
    #plt.savefig('rednoise.png', bbox_inches='tight')
    plt.show()
    '''
    return DIError

def DIBaselineErrorSDC3A(gainMatrix, verbose=True):
    """
    Takes a Ntime, NAntenna, NChannel gain matrix and returns a Ntime, Nbaseline, NChannel error
    """

    baselineError = np.zeros((int((gainMatrix.shape[1]**2 - gainMatrix.shape[1])/2),0))

    loop_time = time()
    for t in np.arange(gainMatrix.shape[0]): 
        for f in np.arange(gainMatrix.shape[2]): # lazy implementation - try to do f, baselines together
            antennaError1, antennaError2 = np.meshgrid(gainMatrix[t,:, f], gainMatrix[t, :, f])
            timeFreqError = antennaError1* np.conj(antennaError2)
            baselineError = np.append(baselineError, timeFreqError[np.triu_indices(gainMatrix.shape[1], k=1)].reshape(int((gainMatrix.shape[1]**2 - gainMatrix.shape[1])/2),int(1)), axis=1)
    get_time = time()
    if(verbose): print(f"Loop took : {timedelta(seconds=time()-loop_time)}")

    #baselineError = baselineError.reshape(-1, gainMatrix.shape[0], gainMatrix.shape[2]) # nBaselines, nTimes, nFreq
    baselineError = baselineError[..., None]
    baselineError = np.moveaxis(baselineError, [0,1], [1, 0])
    
    #baselineError = baselineError.flatten()[..., None]
    if(verbose): print(f"Reshape took : {timedelta(seconds=time()-get_time)}")
    get_time = time()
    return baselineError