import numpy as np
import sys
from astropy.io import fits
from cosmology import Cosmology


class GalaxyCatalogue:

    def __init__(self, catalogue_file, is_box=False, box_length=1000., randoms=False, boss_like=True,
                 special_patchy=False, posn_cols=np.array([0, 1, 2]), omega_m=0.308, fkp=True, noz=True,
                 cp=True, systot=True, veto=True, verbose=True):

        if not is_box and randoms and catalogue_file == '':
            sys.exit('ERROR: no randoms file provided! Randoms required for survey reconstruction')

        if verbose:
            if randoms:
                print('Loading randoms data from file...')
            else:
                print('Loading galaxy data from file...')

        if boss_like:
            with fits.open(catalogue_file) as hdul:
                a = hdul[1].data
            for f in a.names:
                self.__dict__[f.lower()] = a.field(f)
            # NOTE: this takes the weights, in particular the FKP weights, directly from file.
            # If a different FKP weighting is desired (e.g., to work better on smaller scales)
            # this would need to be recalculated!

            self.size = self.ra.size
            self.names = a.names
            self.veto = np.ones(self.size)  # all vetoes have already been applied!

            # change the name 'z'-->'redshift' to avoid confusion
            self.redshift = self.z.astype('float64')  # explicit float64 specification necessary for Cython use

            # initialize Cartesian positions and observer distance
            cosmo = Cosmology(omega_m=omega_m)
            self.dist = cosmo.get_comoving_distance(self.redshift)
            self.x = self.dist * np.cos(self.dec * np.pi / 180) * np.cos(self.ra * np.pi / 180)
            self.y = self.dist * np.cos(self.dec * np.pi / 180) * np.sin(self.ra * np.pi / 180)
            self.z = self.dist * np.sin(self.dec * np.pi / 180)
            self.newx = self.x * 1.
            self.newy = self.y * 1.
            self.newz = self.z * 1.

        else:
            # ASCII and NPY formatted input files are supported
            if '.npy' in catalogue_file:
                data = np.load(catalogue_file)
            else:
                data = np.loadtxt(catalogue_file)

            if is_box:
                self.box_length = box_length

                # for uniform box, position data is in Cartesian format
                self.x = data[:, posn_cols[0]]
                self.y = data[:, posn_cols[1]]
                self.z = data[:, posn_cols[2]]
                self.newx = 1.0 * self.x
                self.newy = 1.0 * self.y
                self.newz = 1.0 * self.z
                self.size = self.x.size

                # for uniform box ra, dec, redshift, distance information are not used!

                # set uniform weights for all galaxies
                self.weight = np.ones(self.size)
            else:
                # position information is ra, dec and redshift
                self.ra = data[:, posn_cols[0]]
                self.dec = data[:, posn_cols[1]]
                self.redshift = data[:, posn_cols[2]]
                self.size = self.ra.size

                # Cartesian positions and observer distance initialized to zero, calculated later
                cosmo = Cosmology(omega_m=omega_m)
                self.dist = cosmo.get_comoving_distance(self.redshift)
                self.x = self.dist * np.cos(self.dec * np.pi / 180) * np.cos(self.ra * np.pi / 180)
                self.y = self.dist * np.cos(self.dec * np.pi / 180) * np.sin(self.ra * np.pi / 180)
                self.z = self.dist * np.sin(self.dec * np.pi / 180)
                self.newx = self.x * 1.
                self.newy = self.y * 1.
                self.newz = self.z * 1.

                # by default, set all weights and veto to 1
                self.weight_fkp = np.ones(self.size)
                self.weight_cp = np.ones(self.size)
                self.weight_noz = np.ones(self.size)
                self.weight_systot = np.ones(self.size)
                self.veto = np.ones(self.size)

                if special_patchy:  # special routine for dealing with unusual PATCHY mock data formatting
                    if randoms:  # randoms are not formatted the same way as data
                        self.weight_fkp = 1./(10000 * data[:, 3])
                        self.veto = data[:, 5]
                        self.weight_cp = data[:, 6]
                    else:
                        self.weight_fkp = 1. / (10000 * data[:, 4])
                        self.veto = data[:, 6]
                        self.weight_cp = data[:, 7]
                # NOTE: this calculation of FKP weights takes P0=10000, which is optimized for
                # case of (a) BOSS CMASS/LOWZ galaxies and (b) BAO reconstruction. For better performance
                # at smaller scales, or for different galaxy samples, this may need to be re-calculated!
                else:
                    count = 1
                    if fkp:
                        self.weight_fkp = data[:, posn_cols[2] + count]
                        count += 1
                    if cp:
                        self.weight_cp = data[:, posn_cols[2] + count]
                        count += 1
                    if noz:
                        self.weight_noz = data[:, posn_cols[2] + count]
                        count += 1
                    if systot:
                        self.weight_systot = data[:, posn_cols[2] + count]
                        count += 1
                    if veto:
                        self.veto = data[:, posn_cols[2] + count]

    def cut(self, w):
        """Trim catalog columns using a boolean array"""

        size = self.size
        for f in self.__dict__.items():
            if hasattr(f[1], 'size') and f[1].size % size == 0:
                self.__dict__[f[0]] = f[1][w]
        self.size = self.x.size

    def get_weights(self, fkp=1, noz=1, cp=1, syst=0):
        """Combine different galaxy weights"""

        if cp == 1 and noz == 1:
            weights = (self.weight_cp + self.weight_noz - 1)
        elif cp == 1:
            weights = 1.*self.weight_cp
        elif noz == 1:
            weights = 1.*self.weight_noz
        else:
            weights = np.ones(self.size)

        if fkp:
            weights *= self.weight_fkp
        if syst:
            weights *= self.weight_systot

        return weights.astype('float64')  # have to specify this because sometimes it was returning float32!
