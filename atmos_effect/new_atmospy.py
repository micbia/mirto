import numpy as np, h5py
from scipy import fftpack
import astropy.io.fits as pyfits
from ARatmospy.create_multilayer_arbase import create_multilayer_arbase
from tqdm import tqdm

class NewArScreens(object):
    """
    Class to generate atmosphere phase screens using an autoregressive
    process to add stochastic noise to an otherwise frozen flow.
    @param n          Number of subapertures across the screen
    @param m          Number of pixels per subaperature
    @param pscale     Pixel scale
    @param rate       A0 system rate (Hz)
    @param paramcube  Parameter array describing each layer of the atmosphere
                      to be modeled.  Each row contains a tuple of 
                      (r0 (m), velocity (m/s), direction (deg), altitude (m))
                      describing the corresponding layer.
    @param alpha_mag  magnitude of autoregressive parameter.  (1-alpha_mag)
                      is the fraction of the phase from the prior time step
                      that is "forgotten" and replaced by Gaussian noise.
    """
    def __init__(self, n, m, pscale, rate, paramcube, alpha_mag, ranseed=None, bit=np.float32):
        self.pl, self.alpha = create_multilayer_arbase(n, m, pscale, rate, paramcube, alpha_mag)
        self.num_pix = n * m
        self.num_param = paramcube.shape[0]
        self._phaseFT = None
        self.bit = bit
        np.random.seed(ranseed)
    
    def get_ar_atmos(self):
        shape = self.alpha.shape
        newphFT = np.zeros_like(self.alpha)
        newphase = np.zeros_like(self.alpha)
        for i, powerlaw, alpha in zip(range(shape[0]), self.pl, self.alpha):
            noise = np.random.normal(size=shape[1:3])
            noisescalefac = np.sqrt(1. - np.abs(alpha**2))
            noiseFT = fftpack.fft2(noise)*powerlaw
            if self._phaseFT is None:
                newphFT[i] = noiseFT
            else:
                newphFT[i] = alpha*self._phaseFT[i] + noiseFT*noisescalefac
            newphase[i] = fftpack.ifft2(newphFT[i]).real
        return newphFT, newphase
    
    def run(self, nframes):
        self.screens = np.zeros((self.num_param, nframes, self.num_pix, self.num_pix), dtype=self.bit)
        for j in tqdm(range(nframes)):
            self._phaseFT, screens = self.get_ar_atmos()
            for i, item in enumerate(screens):
                self.screens[i, j] = item
    
    def write(self, outfile):
        np.save(outfile, self.screens)