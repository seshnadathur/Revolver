import numpy as np
from scipy.integrate import quad
from scipy import constants

class Cosmology:
    """
    Calculate background cosmological quantities for 0<z<4
    """

    def __init__(self, omega_m=0.31, omega_l=0.69, h=0.676):
        c = constants.c / 1000
        omega_k = 1. - omega_m - omega_l
        ztab = np.linspace(0, 4, 1000)
        rtab = np.zeros_like(ztab)
        for i in range(len(ztab)):
            rtab[i] = quad(lambda x: 0.01 * c \
            / np.sqrt(omega_m * (1 + x)**3 + omega_k * (1 + x)**2 + omega_l), 0, ztab[i])[0]

        self.h = h
        self.omega_m = omega_m
        self.omegaL = omega_l
        self.omegaK = omega_k
        self.ztab = ztab
        self.rtab = rtab

    # comoving distance in Mpc/h
    def get_comoving_distance(self, z):
        return np.interp(z, self.ztab, self.rtab)

    def get_ez(self, z):
        return 100 * np.sqrt(self.omega_m * (1 + z)**3 +
                             self.omegaK * (1 + z)**2 + self.omegaL)

    def get_hubble(self, z):
        return self.h * self.get_ez(z)

    def get_redshift(self, r):
        return np.interp(r, self.rtab, self.ztab)
