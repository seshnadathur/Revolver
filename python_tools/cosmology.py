import numpy as np
from scipy.integrate import quad


class Cosmology:

    def __init__(self, omega_m=0.308, h=0.676):
        # print('Initializing cosmology with omega_m = %.3f' % omega_m)
        c = 299792.458
        omega_l = 1.-omega_m
        ztab = np.linspace(0, 4, 1000)
        rtab = np.zeros_like(ztab)
        for i in range(len(ztab)):
            rtab[i] = quad(lambda x: 0.01 * c / np.sqrt(omega_m * (1 + x) ** 3 + omega_l), 0, ztab[i])[0]

        self.h = h
        self.c = c
        self.omega_m = omega_m
        self.omegaL = omega_l
        self.ztab = ztab
        self.rtab = rtab

    # comoving distance in Mpc/h
    def get_comoving_distance(self, z):
        return np.interp(z, self.ztab, self.rtab)

    def get_redshift(self, r):
        return np.interp(r, self.rtab, self.ztab)
