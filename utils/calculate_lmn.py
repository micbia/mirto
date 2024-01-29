import numpy as np

path_data = "/store/ska/sk01/sdc3-new/Image/"

with fits.open(path_data + "ZW3.msn_image.fits", mode="readonly", memmap=True) as hdulist:
    hdr = hdulist[0].header
    Nx, Ny = hdr['NAXIS1'], hdr['NAXIS2']
    RA, DEC = hdr['CRVAL1']*u.deg, hdr['CRVAL2']*u.deg
    freq = hdr['CRVAL3'] 
    
    print(freq, t2c.nu_to_z(freq*1e-6))
    
    # calculate beam size
    bmaj, bmin = hdr['BMAJ']*u.deg, hdr['BMIN']*u.deg
    fwhm_to_sigma = 1./(8*np.log(2))**0.5
    solid_beam = (2.*np.pi*bmaj*bmin*fwhm_to_sigma**2).to('sr')
    print('beam size = %.3e %s' %(solid_beam.value, solid_beam.unit))
    
    # unit conversion from Jy/beam to K
    equiv = u.brightness_temperature(frequency=freq*u.Hz)
    Jyb2K = (u.Jy/solid_beam).to(u.K, equivalencies=equiv)
    
    wsclean_data = hdulist[0].data * Jyb2K.value
    w = WCS(hdr).celestial

idx_ra, idx_dec = np.arange(0, Nx).reshape(Nx, 1), np.arange(0, Ny).reshape(1, Ny)
lon, lat = w.celestial.all_pix2world(idx_ra, idx_dec, 1)
sky_grid =  np.vstack((lon[np.newaxis, ...], lat[np.newaxis, ...]))#.reshape(2,lon.shape[0]*lon.shape[1]).T
sky_grid.shape

alpha = sky_grid[0] # deg
delta = sky_grid[1] # deg

def radec_to_lmn(ra, dec, phase_centre):
    """
    Convert right-ascension and declination positions of a set of sources to
    direction cosines.

    Parameters
    ----------
    ra : ndarray (n_src,)
        Right-ascension in degrees.
    dec : ndarray (n_src,)
        Declination in degrees.
    phase_centre : ndarray (2,)
        The ra and dec coordinates of the phase centre in degrees.

    Returns
    -------
    lmn : ndarray (n_src, 3)
        The direction cosines, (l,m,n), coordinates of each source.
    """
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    phase_centre = np.asarray(phase_centre)
    ra, dec = np.deg2rad(jnp.array([ra, dec]))
    phase_centre = np.deg2rad(phase_centre)

    delta_ra = ra - phase_centre[0]
    dec_0 = phase_centre[1]

    l = np.cos(dec) * np.sin(delta_ra)
    m = np.sin(dec) * np.cos(dec_0) - np.cos(dec) * np.sin(dec_0) * np.cos(delta_ra)
    n = np.sqrt(1 - l**2 - m**2)

    return np.array([l, m, n]).T

gd = radec_to_lmn(ra=alpha, dec=delta, phase_centre=[0, -30])
n = gd[...,-1]

plt.imshow(gd[...,-1])
plt.colorbar()