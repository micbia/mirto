#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os, oskar, sys

from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS
from astropy.coordinates import Angle

import astropy.units as u
import astropy.constants as cst

idx_f = int(sys.argv[1]) 
idx_f += 400

path_in = '/scratch/snx3000/mibianco/output_sdc3/dataLC_256_train_090523/'
path_out = path_in+'ms/'

fits_file = '%sdata/coevalLC_256_train_190922_i0_dT.fits' %path_in
vis_file = '%sms/coevalLC_256_train_190922_i0_dTgf_ch%d_4h1d_256.vis' %(path_in, idx_f)

with fits.open(fits_file, mode="readonly", memmap=True) as hdulist:
    hdr = hdulist[0].header
    #freqs = hdr['CRVAL3']+np.arange(hdr['NAXIS3']) * hdr['CDELT3'] # Hz
    #Jy2kel = (u.Jy * cst.c * cst.c / (2 * cst.k_B * (freqs[idx_f]*u.Hz)**2)).cgs.value
    #lc = hdulist[0].data
    #data = lc[idx_f] / Jy2kel
    #w = WCS(hdr).celestial

Nx = 2048
FoV = abs(hdr['CDELT1']*hdr['NAXIS1'])

# Make Image
imager = oskar.Imager('single')
imager.set(fov_deg=FoV, image_size=Nx)
imager.set(input_file=vis_file, output_root=vis_file.replace('.vis', ''))
output = imager.run(return_images=1)
image = output["images"][0]

np.save('%scoevalLC_256_train_190922_i0_dTgf_ch%d_4h1d_256.MS_I.npy' %(path_out, idx_f), image)
"""
# Plot
fig, axs = plt.subplots(figsize=(10,5 ), ncols=2, nrows=1)
axs[0].set_title('OSKAR image')
im = axs[0].imshow(image, origin='lower', cmap="jet")
#im = axs[0].imshow(lc.squeeze(), origin='lower', cmap="jet")

plt.colorbar(im, ax=axs[0])
im = axs[1].imshow(data.T, origin='lower', cmap='jet')
plt.colorbar(im, ax=axs[1])
#plt.savefig('%s_oskar.png' %(path_out+imager.output_root[imager.output_root.rfind('/')+1:]), bbox_inches='tight')
plt.show(), plt.clf()
"""