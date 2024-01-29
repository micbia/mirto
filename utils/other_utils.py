import numpy as np, matplotlib.pyplot as plt
import astropy.units as u

from datetime import datetime
from astropy.io import fits

def RescaleData(arr, a=-1, b=1):
    scaled_arr = (arr.astype(np.float32) - np.min(arr))/(np.max(arr) - np.min(arr)) * (b-a) + a
    return scaled_arr

def read_fits(fname, unit='mK'):
    data, hdr = fits.getdata(fname, header=True)
    Nx, Ny = hdr['NAXIS1'], hdr['NAXIS2']
    FoV = (abs(hdr['CDELT2']*Nx) * u.deg).to('rad')
    beam_sim = (FoV.value / Nx)**2     # from grid to point source

    freq = (hdr['CRVAL3'] + np.arange(hdr['NAXIS3']) * hdr['CDELT3']) * u.Hz

    fov_deg = hdr['CDELT2']*hdr['NAXIS1'] * u.deg
    bmaj, bmin = hdr['BMAJ']*u.deg, hdr['BMIN']*u.deg
    solid_beam = (np.pi*bmaj*bmin/(4*np.log(2))).to('sr')

    equiv = u.brightness_temperature(frequency=freq)
    Jyb2K = (u.Jy/solid_beam).to(u.K, equivalencies=equiv)

    if(data.shape[0] != data.shape[1]):
        data = np.moveaxis(data, 0, -1) * Jyb2K[None, None, :].to(unit).value
    else:
        data = data * Jyb2K[None, None, :].to(unit).value
    
    return data


def smoothing_grid(arr, noc=None, ratio=None, opt='mean'):
    ''' It coarse a grid from shape=(mesh mesh mesh) to shape=(noc noc noc), a tot of ratio^3 original cells are used to average and create noc cube.
        Parameters:
            * arr (narray)	: 3D array of data
            * noc (int)		: dimension of coarsening of data
            * ratio (int)   : scaling ratio between the two resolution
            * opt (string)  : operation on the smoothing (mean, min or max)
    '''
    
    if(ratio == None and noc != None):
        ratio = int(float(arr.shape[0])/noc)
    elif(ratio != None and noc == None):
        noc = int(float(arr.shape[0])/ratio)
    elif(ratio != None and noc != None):
        ValueError("Specify just one of the two quantity 'noc' or 'ratio'.")
    else:
        ValueError("Specify at least one of the two quantity 'noc' or 'ratio'.")

    if(opt == 'mean'):
        operator = np.mean
    elif(opt == 'max'):
        operator = np.max
    elif(opt == 'min'):
        operator = np.min
    elif(opt == 'sum'):
        operator = np.sum
    else:
        ValueError("Operation on the smoothing must be string ('mean', 'min' or 'max').")
    
    coarse_data = np.zeros([noc, noc])  # nocXnocXnoc
    for i in range(noc):
        for j in range(noc):
            cut = arr[(ratio*i):(ratio*(i+1)), (ratio*j):(ratio*(j+1))]
            coarse_data[i][j] = np.mean(cut)
    return coarse_data


def plot_lightcone(lightcone, loc_axis, fov,
                   xlabel = 'z', ylabel = 'L (cMpc)',
                   fig = None, ax = None, title = None, savefig = False):
    """
    Plot the Epoch of Reionisation's Hydrogen 21cm lightcone

    Parameters
    ----------
    lightcone
        Lightcone data
    loc_axis
        Line of sight axis (e.g. redshift) data
    fov
        Field of view [Mpc]
    xlabel
        Axis x-label
    ylabel
        Axis y-label
    fig
        matplotlib.pylab.Figure instance to plot to. If None, will create new
        figure instance
    ax
        matplotlib.pylab.Axes instance to plot to. If None, will create new
        axes instance
    title
        Plot title
    savefig
        Whether to save the plot. If False (default), will not save. Otherwise
        must be the full path to the save file

    Returns
    -------
    2-Tuple of (matplotlib.pylab.Figure instance,
    matplotlib.pylab.Axes instance) plotted to
    """
    data = {'lc': lightcone, 'z': loc_axis}
    xi = np.array([data['z'] for _ in range(data['lc'].shape[1])])
    yi = np.array([np.linspace(0, fov, data['lc'].shape[1]) for _ in range(xi.shape[1])]).T
    zj = (data['lc'][100, 1:, 1:] + data['lc'][100, 1:, :-1] + data['lc'][100, :-1, 1:] + data['lc'][100, :-1, :-1]) / 4

    if fig is None or ax is None:
        PAGE_WIDTH_INCHES = 6.97522
        fig, ax = plt.subplots(1, 1, figsize=(PAGE_WIDTH_INCHES, PAGE_WIDTH_INCHES / 3.))

    if title is not None:
        ax.set_title(title)

    im = ax.pcolor(xi, yi, zj, cmap='jet')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if loc_axis[0] > loc_axis[-1]:
        ax.invert_xaxis()

    ax.tick_params(axis='both', which='major')
    fig.subplots_adjust(bottom=0.11, right=0.91, top=0.95, left=0.06)
    cax = plt.axes([0.92, 0.15, 0.02, 0.75])
    fig.colorbar(im, cax=cax)

    if savefig:
        fig.savefig(savefig, dpi=300, bbox_inches='tight')

    return fig, ax


def write_fits(fname, data, freqs=None, cdelt=None, ra=None, dec=None, hdr=None):
    hdu = fits.PrimaryHDU(data)
    hdul = fits.HDUList([hdu])
    if(hdr):
        hdul[0].header = hdr
    else:
        hdul[0].header.set('CTYPE1', 'RA---SIN')
        hdul[0].header.set('CTYPE2', 'DEC--SIN')
        try:
            hdul[0].header.set('CRVAL1', ra.value)
            hdul[0].header.set('CRVAL2', dec.value)
        except:
            hdul[0].header.set('CRVAL1', ra)
            hdul[0].header.set('CRVAL2', dec)
        hdul[0].header.set('CUNIT1', 'deg     ')
        hdul[0].header.set('CUNIT2', 'deg     ')
 
        hdul[0].header.set('CTYPE3', 'FREQ    ')
        try:
            hdul[0].header.set('CRVAL3', np.min(freqs.value))
            hdul[0].header.set('CDELT3', np.abs(np.diff(freqs[:2].value)[0]))
        except:
            hdul[0].header.set('CRVAL3', np.min(freqs))
            hdul[0].header.set('CDELT3', np.abs(np.diff(freqs[:2])[0]))
        hdul[0].header.set('CUNIT3', 'Hz      ')

        hdul[0].header.set('CRPIX1', data.shape[1]//2+1)
        hdul[0].header.set('CRPIX2', data.shape[2]//2+1)
        hdul[0].header.set('CRPIX3', freqs.size)
        try:
            hdul[0].header.set('CDELT1', -cdelt.value)
            hdul[0].header.set('CDELT2', cdelt.value)
        except:
            hdul[0].header.set('CDELT1', -cdelt)
            hdul[0].header.set('CDELT2', cdelt)

        hdul[0].header.set('BUNIT', 'K       ')
    
    hdul.writeto(fname+'.fits', overwrite=True)


def timeseed():
    # create a random seed based on the current second and millisecond
    seed = [var for var in datetime.now().strftime('%d%H%M%S')]
    np.random.shuffle(seed)
    return int(''.join(seed))
