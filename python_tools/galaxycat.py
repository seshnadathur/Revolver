import numpy as np
import sys
import healpy as hp
import os
from astropy.io import fits
from python_tools.cosmology import Cosmology


class GalaxyCatalogue:

    def __init__(self, parms, randoms=False):

        if randoms:
            input_file = parms.random_file
            input_file_type = parms.random_file_type
            posn_cols = parms.random_posn_cols
        else:
            input_file = parms.tracer_file
            input_file_type = parms.tracer_file_type
            posn_cols = parms.tracer_posn_cols

        if parms.verbose:
            if randoms:
                print('Loading randoms data from file...')
            else:
                print('Loading galaxy data from file...')

        if not os.access(input_file, os.F_OK):
            sys.exit("Can't find input file %s, aborting" % input_file)

        if parms.is_box:  # dealing with uniform cubic simulation box with periodic boundary conditions

            self.box_length = parms.box_length

            if parms.input_file_type == 1:
                # FITS file format: assumes position fields are called X,Y,Z
                with fits.open(input_file) as hdul:
                    a = hdul[1].data
                for f in a.names:
                    self.__dict__[f.lower()] = a.field(f)

                self.newx = 1.0 * self.x
                self.newy = 1.0 * self.y
                self.newz = 1.0 * self.z
                self.size = self.x.size
                self.weight = np.ones(self.x.size)  # uniform box so all weights are 1

            else:
                if parms.input_file_type == 2:
                    # Numpy pickle format ('.npy' files)
                    data = np.load(input_file)
                else:
                    # default is assumed to be ASCII file format
                    data = np.loadtxt(input_file)

                self.x = data[:, posn_cols[0]]
                self.y = data[:, posn_cols[1]]
                self.z = data[:, posn_cols[2]]
                self.newx = 1.0 * self.x
                self.newy = 1.0 * self.y
                self.newz = 1.0 * self.z
                self.size = self.x.size

                # set uniform weights for all tracers
                self.weight = np.ones(self.size)

        else:  # dealing with survey-like data on the lightcone

            if input_file_type == 1:
                # FITS file format, assumes field names match BOSS/eBOSS data model
                with fits.open(input_file) as hdul:
                    a = hdul[1].data
                for f in a.names:
                    self.__dict__[f.lower()] = a.field(f)
                self.size = self.ra.size
                self.names = a.names
                # assume all vetoes have already been applied (as for BOSS/eBOSS public catalogues)
                self.veto = np.ones(self.size)  # all vetoes have already been applied!
                # change the name 'z'-->'redshift' to avoid confusion
                self.redshift = self.z.astype('float64')  # explicit float64 specification necessary for Cython use
                self.weights_model = parms.weights_model
                # check if weights are provided; if not, initialize to defaults
                if 'weight_cp' not in (name.casefold() for name in self.names):
                    # set close pair weights to 1 by default
                    self.weight_cp = np.ones(self.size)
                if 'weight_noz' not in (name.casefold() for name in self.names):
                    # set missing redshift weights to 1 by default
                    self.weight_noz = np.ones(self.size)
                if 'weight_systot' not in (name.casefold() for name in self.names):
                    # set total systematic weights to 1 by default
                    self.weight_systot = np.ones(self.size)
                if 'weight_fkp' not in (name.casefold() for name in self.names):
                    # set FKP weights to 1 by default
                    print('No FKP weights found for survey data. Proceeding without, but are you sure?')
                    self.weight_fkp = np.ones(self.size)
                    # NOTE: if provided, we take the FKP weights directly from file
                    # If an FKP weighting different to the BOSS/eBOSS standard is desired
                    # (e.g., to work better on smaller scales), this would need to be recalculated
                if self.weights_model == 1:
                    # like BOSS, so completeness is specified as COMP
                    if 'comp' not in (name.casefold() for name in self.names):
                        # assume a uniform completeness over the sky
                        self.comp = np.ones(self.size)
                elif self.weights_model == 2:
                    # like eBOSS, so completeness is specified as COMP_BOSS
                    if 'comp_boss' not in (name.casefold() for name in self.names):
                        # assume a uniform completeness over the sky
                        self.comp = np.ones(self.size)
                    else:
                        self.comp = self.comp_boss
                elif self.weights_model == 3:
                    # for the joint BOSS+eBOSS LRG sample, completeness is not provided at all
                    self.comp = np.ones(self.size)
                    # and all systematic weights should be amalgamated into one
                    if 'weight_all_nofkp' not in (name.casefold() for name in self.names):
                        # set total systematic weights to 1 by default
                        self.weight_all_nofkp = np.ones(self.size)

            elif input_file_type == 2 or input_file_type == 3:
                # data is provided as an array
                if input_file_type == 2:
                    # Numpy pickle format ('.npy' files)
                    data = np.load(input_file)
                else:
                    # default is assumed to be ASCII file format
                    data = np.loadtxt(input_file)

                # position information is ra, dec and redshift
                self.ra = data[:, posn_cols[0]]
                self.dec = data[:, posn_cols[1]]
                self.redshift = data[:, posn_cols[2]]
                self.size = self.ra.size

                # by default, initialize all weights and vetos to 1
                self.weight_fkp = np.ones(self.size)
                self.weight_cp = np.ones(self.size)
                self.weight_noz = np.ones(self.size)
                self.weight_systot = np.ones(self.size)
                self.comp = np.ones(self.size)
                self.veto = np.ones(self.size)

                # if weight information is present, get it from the input
                self.weights_model = 1  # TODO: currently only allows BOSS weights model for array input; extend?
                count = 1
                if parms.fkp:
                    self.weight_fkp = data[:, posn_cols[2] + count]
                    count += 1
                # NOTE: randoms file should not have the following weights information (and we ignore it if present!)
                if parms.cp and not randoms:
                    self.weight_cp = data[:, posn_cols[2] + count]
                    count += 1
                if parms.noz and not randoms:
                    self.weight_noz = data[:, posn_cols[2] + count]
                    count += 1
                if parms.systot and not randoms:
                    self.weight_systot = data[:, posn_cols[2] + count]
                    count += 1
                if parms.veto and not randoms:
                    self.veto = data[:, posn_cols[2] + count]
                    count += 1
                if parms.comp and not randoms:
                    self.comp = data[:, posn_cols[2] + count]

            elif input_file_type == 4:
                # this is a special case used to deal with the format of BOSS DR12 Patchy mocks on Sciama
                # should not be relevant anywhere else
                if '.npy' in input_file:
                    data = np.load(input_file)
                else:
                    data = np.loadtxt(input_file)

                # ra, dec and redshift are always first three columns
                self.ra = data[:, 0]
                self.dec = data[:, 1]
                self.redshift = data[:, 2]
                self.size = self.ra.size

                # by default, initialize all weights and vetos to 1
                self.weight_fkp = np.ones(self.size)
                self.weight_cp = np.ones(self.size)
                self.weight_noz = np.ones(self.size)
                self.weight_systot = np.ones(self.size)
                self.comp = np.ones(self.size)
                self.veto = np.ones(self.size)

                if randoms:  # Patchy randoms are not formatted the same way as the Patchy data
                    self.weight_fkp = 1. / (10000 * data[:, 3])
                    self.veto = data[:, 5]  # weird format means veto should be used
                    # self.weight_cp = data[:, 6]
                    # don't know why cp weights are provided in the Patchy randoms, but they shouldn't be used
                else:
                    self.weight_fkp = 1. / (10000 * data[:, 4])
                    self.veto = data[:, 6]  # weird format means veto should be used
                    # self.weight_cp = data[:, 7]
                    # don't know why cp weights are provided in the Patchy randoms, but they shouldn't be used

                # the Patchy mocks do not have completeness weights as in the data files
                # therefore in this case they have to be taken from the mask
                # this means the mask file has to contain that information rather than be a binary mask
                # all this is hard-coded here because this code section should not be used EXCEPT when
                # dealing with this special case!
                print('SPECIAL CASE: input_file_type=4 is for BOSS DR12 Patchy mocks – are you sure you wanted this?')
                print('Proceeding to extract completeness information from the mask ...')
                if not os.access(parms.mask_file, os.F_OK):
                    sys.exit('Mask file %s not specified or not found, aborting' % parms.mask_file)

                completeness_mask = hp.read_map(parms.mask_file, verbose=False)
                nside = hp.get_nside(completeness_mask)
                pixels = hp.ang2pix(nside, np.deg2rad(90 - self.dec), np.deg2rad(self.ra))
                self.comp = completeness_mask[pixels]

            # now initialize Cartesian positions and observer distance
            cosmo = Cosmology(omega_m=parms.omega_m)
            self.dist = cosmo.get_comoving_distance(self.redshift)
            self.x = self.dist * np.cos(self.dec * np.pi / 180) * np.cos(self.ra * np.pi / 180)
            self.y = self.dist * np.cos(self.dec * np.pi / 180) * np.sin(self.ra * np.pi / 180)
            self.z = self.dist * np.sin(self.dec * np.pi / 180)
            self.newx = self.x * 1.
            self.newy = self.y * 1.
            self.newz = self.z * 1.

    def cut(self, w):
        """Trim catalog columns using a boolean array"""

        size = self.size
        for f in self.__dict__.items():
            if hasattr(f[1], 'size') and f[1].size % size == 0:
                self.__dict__[f[0]] = f[1][w]
        self.size = self.x.size

    def get_weights(self, fkp=True, syst_wts=True):
        """
        Combine different galaxy weights
        :param fkp:      bool, default True
                         whether to include FKP weights
        :param syst_wts: bool, default True
                         whether to include observational systematic weights
        :return: composite weights for each galaxy
        """

        weights = np.ones(self.size)
        if fkp:
            weights *= self.weight_fkp
        if syst_wts:
            if self.weights_model == 1:
                # BOSS model for weights; cp + noz - 1
                weights *= self.weight_systot * (self.weight_noz + self.weight_cp - 1)
            elif self.weights_model == 2:
                # eBOSS model for weights; cp * noz
                weights *= self.weight_systot * self.weight_cp * self.weight_noz
            elif self.weights_model == 3:
                # joint BOSS+eBOSS weights model; just use custom field
                weights *= self.weight_all_nofkp

        return weights.astype('float64')  # have to specify this because sometimes it was returning float32!
