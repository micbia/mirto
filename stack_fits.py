"""
Created on Sun Jun 25, 2023
@author: Michele Bianco
"""
import numpy as np, os, sys
from tqdm import tqdm

import astropy.units as u
from astropy.io import fits

from utils.other_utils import write_fits, RescaleData

# ---> Rremark:
# Here the i_min indicates high redshift (low frequency), while i_max is related to the lowest redshfit (highest frequency). Moreover, here the targets are stored with the shape (Nx, Ny, N_freq).

i, i_bin = 0, 4
freqs_sdc = np.loadtxt('/store/ska/sk014/dataset_sdc3/freqs_sdc3.txt')
freq_bins = np.loadtxt('/store/ska/sk014/sdc3/testdataset/bins_frequency.txt')

freq_min, freq_max = freq_bins[i_bin]
i_min, i_max = np.argmin(np.abs(freq_min-freqs_sdc)), np.argmin(np.abs(freq_max-freqs_sdc))+1
print(' frequency range: [%.2f, %.2f] MHz' %(freq_min, freq_max))
print(' idx_f = [%d, %d]' %(i_min, i_max))

path_in = '/scratch/snx3000/mibianco/output_sdc3/dataLC_130923/'
path_data = path_in+'lightcones/'
path_out = path_in+'data/'

# read xHI lightcone simulated with 21cmFAST
xH_fits = '%slc_256_train_130923_i%d_xHI.fits' %(path_data, i)
xHI_data = fits.getdata(xH_fits)
print(' data upsampled shape: %s' %str(xHI_data.shape))

# upsample the xHI maps to the mesh size of the dirty image
upsample_data = xHI_data.repeat(8, axis=1).repeat(8, axis=2)
print(' data upsampled shape: %s' %str(upsample_data.shape))

# read psf for smoothing xHI field (natural weight)
psf_file = '/store/ska/sk01/sdc3-new/Image/ZW3.msn_psf.fits'
psf_data = fits.getdata(psf_file)
hdr = fits.getheader(psf_file)

# cut to the central 4 deg (see SDC3 documentation)
cut = 574
angular_shape = int(2048-2*cut)
print(' data dT shape: %s' %str((angular_shape, angular_shape, i_max-i_min)))

if not (os.path.exists('%sxH_21cm_i%d_ch%d-%d.npy' %(path_out, i, i_min, i_max))):
    mask_xH_data = np.zeros((angular_shape, angular_shape, i_max-i_min), dtype=np.float32)

for typ in ['dT', 'dTgf', 'dTgfpoint', 'dTiongfpoint', 'dTnoisegainiongf', 'dTnoisegainiongfpoint', 'gf', 'point']:
    true_dT_data = np.zeros((angular_shape, angular_shape, i_max-i_min), dtype=np.float32)
    true_dT_fits = np.zeros((i_max-i_min, 2048, 2048))
    idx = 0
    freqs = []
    for idx_f in tqdm(range(i_min, i_max), desc='%s_i%d' %(typ, i)):
        # convolve the xHI image with psf (smooth field)
        if not (os.path.exists('%sxH_21cm_i%d_ch%d-%d.' %(path_out, i, i_min, i_max))):
            fft_img = np.fft.rfft2(upsample_data[idx_f], norm='ortho')
            sampling = np.fft.rfft2(psf_data[idx_f], norm='ortho')
            obs_img = np.fft.fftshift(np.fft.irfft2(fft_img * sampling, norm='ortho'))
            smooth_xH = RescaleData(obs_img, a=0, b=1)
            
            # apply threshold
            mask_xH = smooth_xH > 0.5
            mask_xH_data[..., idx] = mask_xH[cut:psf_data.shape[1]-cut,cut:psf_data.shape[2]-cut].T     # for some weird reason we nee to transpose to match the dirty image
        
        # get dirty-image created with WSCLEAN
        wsclean_file = '%slc_256_train_130923_i%d_%s_ch%d_4h1d_256-dirty.fits' %(path_in+'ms/', i, typ, idx_f)

        # get header for unit conversion from Jy/beam to K
        hdr = fits.getheader(wsclean_file)
        if(idx_f == i_min):
            hdr0 = hdr
        Nx, Ny = hdr['NAXIS1'], hdr['NAXIS2']
        RA, DEC = hdr['CRVAL1']*u.deg, hdr['CRVAL2']*u.deg
        freq = hdr['CRVAL3']*u.Hz
        freqs.append(freq.value)

        bmaj, bmin = hdr['BMAJ']*u.deg, hdr['BMIN']*u.deg
        fwhm_to_sigma = 1./(8*np.log(2))**0.5
        solid_beam = (2.*np.pi*bmaj*bmin*fwhm_to_sigma**2).to('sr')

        equiv = u.brightness_temperature(frequency=freq)
        Jyb2K = (u.Jy/solid_beam).to(u.mK, equivalencies=equiv)

        # get dirty image
        wsclean_data = fits.getdata(wsclean_file).squeeze()

        assert Jyb2K.unit == u.mK
        true_dT_fits[idx] = wsclean_data * Jyb2K
        true_dT_data[..., idx] = wsclean_data[cut:psf_data.shape[1]-cut,cut:psf_data.shape[2]-cut] * Jyb2K.value  # in mK
        idx += 1

    if not (os.path.exists('%sfrequency.txt' %path_out)):
        data_freq = np.zeros((len(freqs), 2))
        data_freq[:, 0] = np.arange(i_min, i_max)
        data_freq[:, 1] = freqs
        np.savetxt('%sfrequency.txt' %path_out, data_freq, fmt='%d\t%.2f', header='N_ch\tfreq [Hz]')

    # save binary lightcone
    if not (os.path.exists('%sxH_21cm_i%d_ch%d-%d.npy' %(path_out, i, i_min, i_max))):
        np.save('%sxH_21cm_i%d_ch%d-%d.npy' %(path_out, i, i_min, i_max), mask_xH_data)

    output_name = '%s_21cm_i%d_ch%d-%d.npy' %(path_out+typ, i, i_min, i_max)
    np.save(output_name, true_dT_data)

    hdr0['CRPIX3'] = true_dT_fits.shape[0]
    hdr0['CDELT3'] = np.diff(freqs)[0]
    hdr0['BUNIT'] = str(Jyb2K.unit)
    write_fits(fname=output_name[:-4], data=true_dT_fits, hdr=hdr0)
