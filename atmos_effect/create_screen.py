"""
Generate a test ionsopheric screen with multiple layers.
ARatmospy must be in the PYTHONPATH https://github.com/shrieks/ARatmospy
"""

import numpy as np, gc, sys
from astropy.io import fits
from astropy.wcs import WCS

sys.path.append("/users/mibianco/codes/sdc3")
from gain.iono.new_atmospy import NewArScreens
#from ARatmospy.ArScreens import ArScreens
from tqdm import tqdm

def simulate_screen(screen_width_metres, r0, bmax, sampling, speed, rate, alpha_mag, num_times, filename, rseed=2023):
    """
    Inputs:
    - screen_width_metres (float): self explanatory
    - r0 (float): Scale size in km
    - bmax (float): sub-aperture size in km
    - sampling (float): meter per pixel
    - speed (float): speed of the layer in m/s (default for 150 km/s)
    - rate (float): the inverse frame rate in 1/s
    - alpha (float): evolve the screen slovely
    - num_times (float): length of the observation in sec
    - frequency (float): in Hz
    - fits_filename (string): where to store the simulated screen
    """ 
    m = int(bmax / sampling)  # Pixels per sub-aperture
    n = int(screen_width_metres / bmax)  # Sub-apertures across the screen
    num_pix = n * m
    pscale = screen_width_metres / (n * m)  # Pixel scale (in m/pixel).

    print("Number of pixels %d, pixel size %.3f m" % (num_pix, pscale))
    print("Field of view %.1f (m)" % (num_pix * pscale))

    speed = speed.to('m/s').value

    # Parameters for each layer (scale size [m], speed [m/s], direction [deg], layer height [m]).
    layer_params = np.array([(r0, speed, 0., 300e3), (r0, speed/2.0, -30.0, 310e3)])

    my_screens = NewArScreens(n=n, m=m, pscale=pscale, rate=rate, paramcube=layer_params, alpha_mag=alpha_mag, ranseed=rseed)
    my_screens.run(nframes=num_times, verbose=True)

    np.save(filename, my_screens.screens)
    del my_screen
    gc.collect()
    return 0


def simulate_TEC(screens, screen_width_metres, bmax, rate, sampling, num_times, frequency, fits_filename):
    """
    Inputs:
    - screen_width_metres (float): self explanatory
    - r0 (float): Scale size in km
    - bmax (float): sub-aperture size in km
    - sampling (float): meter per pixel
    - speed (float): speed of the layer in m/s (default for 150 km/s)
    - rate (float): the inverse frame rate in 1/s
    - alpha (float): evolve the screen slovely
    - num_times (float): length of the observation in sec
    - frequency (float): in Hz
    - fits_filename (string): where to store the simulated screen
    """ 
    m = int(bmax / sampling)  # Pixels per sub-aperture
    n = int(screen_width_metres / bmax)  # Sub-apertures across the screen
    num_pix = n * m
    pscale = screen_width_metres / (n * m)  # Pixel scale (in m/pixel).

    print("Number of pixels %d, pixel size %.3f m" % (num_pix, pscale))
    print("Field of view %.1f (m)" % (num_pix * pscale))

    # Convert to TEC
    phase2tec = -frequency / 8.44797245e9 * 1e-2 # TODO: Here the factor 1e-2 is introduced by the SDC3, and it represent the outcome of a successful DD calibration.


    w = WCS(naxis=4)
    w.naxis = 4
    w.wcs.cdelt = [pscale, pscale, 1.0/rate, 1.0]
    w.wcs.crpix = [num_pix // 2 + 1, num_pix // 2 + 1, num_times // 2 + 1, 1.0]
    w.wcs.ctype = ['XX', 'YY', 'TIME', 'FREQ']
    w.wcs.crval = [0.0, 0.0, 0.0, frequency]
    
    data = phase2tec*screens.sum(axis=0)[None, ...]

    fits.writeto(filename=fits_filename, data=data, header=w.to_header(), overwrite=True)
    
    del data, screens
    gc.collect()

    return 0


def simulate_IonoSky(screen_width_metres, r0, bmax, sampling, speed, rate, alpha_mag, num_times, frequency, fits_filename):
    """
    Inputs:
    - screen_width_metres (float): self explanatory
    - r0 (float): Scale size in km
    - bmax (float): sub-aperture size in km
    - sampling (float): meter per pixel
    - speed (float): speed of the layer in m/s (default for 150 km/s)
    - rate (float): the inverse frame rate in 1/s
    - alpha (float): evolve the screen slovely
    - num_times (float): length of the observation in sec
    - frequency (float): in Hz
    - fits_filename (string): where to store the simulated screen
    """ 
    m = int(bmax / sampling)  # Pixels per sub-aperture
    n = int(screen_width_metres / bmax)  # Sub-apertures across the screen
    num_pix = n * m
    pscale = screen_width_metres / (n * m)  # Pixel scale (in m/pixel).

    print("Number of pixels %d, pixel size %.3f m" % (num_pix, pscale))
    print("Field of view %.1f (m)" % (num_pix * pscale))

    speed = speed.to('m/s').value

    # Parameters for each layer (scale size [m], speed [m/s], direction [deg], layer height [m]).
    layer_params = np.array([(r0, speed, 0., 300e3), (r0, speed/2.0, -30.0, 310e3)])

    my_screens = ArScreens(n=n, m=m, pscale=pscale, rate=rate, paramcube=layer_params, alpha_mag=alpha_mag, ranseed=2023)
    my_screens.run(nframes=num_times, verbose=True)

    # Convert to TEC
    phase2tec = -frequency / 8.44797245e9

    w = WCS(naxis=4)
    w.naxis = 4
    w.wcs.cdelt = [pscale, pscale, 1.0/rate, 1.0]
    w.wcs.crpix = [num_pix // 2 + 1, num_pix // 2 + 1, num_times // 2 + 1, 1.0]
    w.wcs.ctype = ['XX', 'YY', 'TIME', 'FREQ']
    w.wcs.crval = [0.0, 0.0, 0.0, frequency]
    data = np.zeros([1, num_times, num_pix, num_pix])

    for layer in tqdm(range(len(my_screens.screens))):
        for i, screen in enumerate(my_screens.screens[layer]):
            data[:, i, ...] += phase2tec * screen[np.newaxis, ...]
    fits.writeto(filename=fits_filename, data=data, header=w.to_header(), overwrite=True)

    return my_screens