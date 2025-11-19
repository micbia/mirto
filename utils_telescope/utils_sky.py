import numpy as np
import tools21cm as t2c

# define a 2D gaussian function
def gaussian_2d(prefactor, x, y, mean, cov):
    """ a simple 2D gaussian distribution """
    x_diff = x - mean[0]
    y_diff = y - mean[1]
    inv_cov = np.linalg.inv(cov)
    exponent = -0.5 * (x_diff**2 * inv_cov[0, 0] + y_diff**2 * inv_cov[1, 1] + 2 * x_diff * y_diff * inv_cov[0, 1])
    return prefactor * np.exp(exponent)

def galactic_synch_fg_custom(z, ncells, boxsize, A150=513., beta_=2.34, rseed=False):
    if(isinstance(z, float)):
        z = np.array([z])
    else:
        z = np.array(z, copy=False)
    gf_data = np.zeros((ncells, ncells, z.size))

    if(rseed): np.random.seed(rseed)
    X  = np.random.normal(size=(ncells, ncells))
    Y  = np.random.normal(size=(ncells, ncells))
    nu_s,a_syn,Da_syn = 150,2.8,0.1

    for i in range(0, z.size):
        nu = t2c.cosmo.z_to_nu(z[i])
        U_cb  = (np.mgrid[-ncells/2:ncells/2,-ncells/2:ncells/2]+0.5)*t2c.cosmo.z_to_cdist(z[i])/boxsize
        l_cb  = 2*np.pi*np.sqrt(U_cb[0,:,:]**2+U_cb[1,:,:]**2)
        #C_syn = A150*(1000/l_cb)**beta_*(nu/nu_s)**(-2*a_syn-2*Da_syn*np.log(nu/nu_s))
        C_syn = A150*(1000/l_cb)**beta_
        #C_syn = A150
        solid_angle = boxsize**2/t2c.cosmo.z_to_cdist(z[i])**2
        AA = np.sqrt(solid_angle*C_syn/2)
        T_four = AA*(X+Y*1j) * np.sqrt(2)
        T_real = np.abs(np.fft.ifft2(T_four))   #in Jansky
        #gf_data[..., i] = t2c.jansky_2_kelvin(T_real*1e6, z[i], boxsize=boxsize, ncells=ncells)
        gf_data[..., i] = T_real
    return gf_data.squeeze()
