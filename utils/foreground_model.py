import tools21cm as t2c
import numpy as np

""" Michele Bianco, 23 Aug 2024, 11:25am CEST
    To date there is a small bug in the version of tools21cm that Karabo install.
    Since it will take a bit of time to get a new release of Karabo (with the newest tools21cm) here a quick fix.
    This function is originally in foreground_model.py in tools21cm with the modification to avoid NaN when the ncells is an uneven number."""

def galactic_synch_fg(z, ncells, boxsize, rseed=False):
    """
    @ Ghara et al. (2017)

    Parameters
    ----------
    z           : float
        Redshift observed with 21-cm.
    ncells      : int
        Number of cells on each axis.
    boxsize     : float
        Size of the FOV in Mpc.
    rseed: int
        random seed to have the same realisation (Default: False).
    Returns
    -------
    A 2D numpy array of brightness temperature in mK.
    """
    if(isinstance(z, float)):
        z = np.array([z])
    else:
        z = np.array(z, copy=False)
    gf_data = np.zeros((ncells, ncells, z.size))

    if(rseed): np.random.seed(rseed)
    
    # np.fft require even number of cells (otherwise give NaN)
    if(ncells%2 != 0):
        X  = np.random.normal(size=(ncells+1, ncells+1))
        Y  = np.random.normal(size=(ncells+1, ncells+1))
    else:
        X  = np.random.normal(size=(ncells, ncells))
        Y  = np.random.normal(size=(ncells, ncells))

    nu_s,A150,beta_,a_syn,Da_syn = 150,513,2.34,2.8,0.1

    for i in range(0, z.size):
        nu = t2c.z_to_nu(z[i])
        if(ncells%2 != 0):
            U_cb  = (np.mgrid[(-ncells-1)/2:(ncells+1)/2,(-ncells-1)/2:(ncells+1)/2]+0.5)*t2c.z_to_cdist(z[i])/boxsize
        else:
            U_cb  = (np.mgrid[-ncells/2:ncells/2,-ncells/2:ncells/2]+0.5)*t2c.z_to_cdist(z[i])/boxsize
        
        l_cb  = 2*np.pi*np.sqrt(U_cb[0,:,:]**2+U_cb[1,:,:]**2)
        C_syn = A150*(1000/l_cb)**beta_*(nu/nu_s)**(-2*a_syn-2*Da_syn*np.log(nu/nu_s))
        solid_angle = boxsize**2/t2c.z_to_cdist(z[i])**2
        AA = np.sqrt(solid_angle*C_syn/2)
        T_four = AA*(X+Y*1j) * np.sqrt(2)   # sum of two distribution changes the std of 1/sqrt(2)
        T_real = np.abs(np.fft.ifft2(T_four))   #in Jansky
        if(ncells%2 != 0):
            gf_data[..., i] = t2c.jansky_2_kelvin(T_real*1e6, z[i], boxsize=boxsize, ncells=ncells)[:-1,:-1]
        else:
            gf_data[..., i] = t2c.jansky_2_kelvin(T_real*1e6, z[i], boxsize=boxsize, ncells=ncells)
    return gf_data.squeeze()

