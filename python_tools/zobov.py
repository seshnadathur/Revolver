from __future__ import print_function
import sys
import os
import numpy as np
import healpy as hp
import random
import subprocess
import glob
from scipy.spatial import cKDTree
from scipy.integrate import quad
from python_tools.cosmology import Cosmology
from python_tools.galaxycat import GalaxyCatalogue
from scipy.signal import savgol_filter
from scipy.interpolate import InterpolatedUnivariateSpline


class ZobovVoids:

    def __init__(self, parms):

        print("\n ==== Starting the void-finding with ZOBOV ==== ")
        sys.stdout.flush()

        self.verbose = parms.verbose

        # the prefix/handle used for all output file names
        self.handle = parms.handle

        # output folder
        self.output_folder = parms.output_folder
        if not os.access(self.output_folder, os.F_OK):
            os.makedirs(self.output_folder)

        # file path for ZOBOV-formatted tracer data
        self.posn_file = self.output_folder + parms.handle + "_pos.dat"

        # load up the tracer catalogue information
        cat = GalaxyCatalogue(parms, randoms=False)

        # (Boolean) choice between cubic simulation box and sky survey
        self.is_box = parms.is_box

        # ZOBOV run-time options
        self.use_mpi = parms.use_mpi
        self.zobov_box_div = parms.zobov_box_div
        self.zobov_buffer = parms.zobov_buffer
        self.guard_nums = parms.guard_nums

        # data preparation steps
        if self.is_box:
            if parms.box_length <= 0:
                sys.exit("Zero or negative box length, aborting")
            self.box_length = parms.box_length

            # assign the tracer positions within the box
            self.num_tracers = cat.size
            tracers = np.empty((self.num_tracers, 3))
            tracers[:, 0] = cat.x
            tracers[:, 1] = cat.y
            tracers[:, 2] = cat.z
            # check that tracer positions lie within the box, wrap using PBC if not
            tracers[tracers[:, 0] > self.box_length, 0] -= self.box_length
            tracers[tracers[:, 1] > self.box_length, 1] -= self.box_length
            tracers[tracers[:, 2] > self.box_length, 2] -= self.box_length
            tracers[tracers[:, 0] < 0, 0] += self.box_length
            tracers[tracers[:, 1] < 0, 1] += self.box_length
            tracers[tracers[:, 2] < 0, 2] += self.box_length

            # determine mean tracer number density
            self.tracer_dens = 1.0 * self.num_tracers / (self.box_length ** 3.)

            self.num_mocks = 0
            self.num_part_total = self.num_tracers
            self.tracers = tracers
        else:
            # set cosmology
            self.omega_m = parms.omega_m
            cosmo = Cosmology(omega_m=self.omega_m)
            self.cosmo = cosmo

            # apply veto if any is required (only in unusual cases)
            veto = np.asarray(cat.veto, dtype=bool)
            cat.cut(veto)
            self.num_tracers = cat.size

            # convert input tracer information to standard format
            syswt = cat.get_weights(fkp=False, syst_wts=True)
            comp = cat.comp
            self.coords_radecz2std(cat.ra, cat.dec, cat.redshift, syswt, comp)
            # after this step, self.tracers contains all the reqd tracer information, no longer need cat

            self.z_min = parms.z_min
            self.z_max = parms.z_max
            # check and cut on the provided redshift limits
            if np.min(self.tracers[:, 5]) < self.z_min or np.max(self.tracers[:, 5]) > self.z_max:
                print('Cutting galaxies outside the redshift limits provided')
                zselect = (self.z_min < self.tracers[:, 5]) & (self.tracers[:, 5] < self.z_max)
                self.tracers = self.tracers[zselect, :]

            # sky mask file (should be in Healpy FITS format)
            if not os.access(parms.mask_file, os.F_OK):
                print("Sky mask not provided or not found, generating approximate one")
                sys.stdout.flush()
                self.mask_file = self.output_folder + self.handle + '_mask.fits'
                self.f_sky = self.generate_mask()
            else:
                mask = hp.read_map(parms.mask_file, verbose=False)
                self.mask_file = parms.mask_file
                # check whether the mask is correct
                ra = self.tracers[:, 3]
                dec = self.tracers[:, 4]
                nside = hp.get_nside(mask)
                pixels = hp.ang2pix(nside, np.deg2rad(90 - dec), np.deg2rad(ra))
                if np.any(mask[pixels] == 0):
                    print('Galaxies exist where mask=0. Maybe check the input?')
                    sys.stdout.flush()
                    # all_indices = np.arange(len(self.tracers))
                    # bad_inds = np.where(mask[pixels] == 0)[0]
                    # good_inds = all_indices[np.logical_not(np.in1d(all_indices, bad_inds))]
                    # self.tracers = self.tracers[good_inds, :]

                # effective sky fraction
                self.f_sky = 1.0 * np.sum(mask) / len(mask)

            # finally, remove any instances of two galaxies at the same location, otherwise tessellation will fail
            # (this is a problem with DR12 Patchy mocks, I've not seen any such instances in real survey data ...)
            # NOTE: the following line will not work with older versions of numpy!!
            unique_tracers = np.unique(self.tracers, axis=0)
            if unique_tracers.shape[0] < self.tracers.shape[0]:
                print('Removing %d galaxies with duplicate positions' %
                      (self.tracers.shape[0] - unique_tracers.shape[0]))
                sys.stdout.flush()
            self.tracers = unique_tracers

            # update galaxy stats
            self.num_tracers = self.tracers.shape[0]
            print('Kept %d tracers after all cuts' % self.num_tracers)
            sys.stdout.flush()

            # calculate mean density
            self.r_near = self.cosmo.get_comoving_distance(self.z_min)
            self.r_far = self.cosmo.get_comoving_distance(self.z_max)
            survey_volume = self.f_sky * 4 * np.pi * (self.r_far ** 3. - self.r_near ** 3.) / 3.
            self.tracer_dens = self.num_tracers / survey_volume

            # weights options: correct for z-dependent selection and angular completeness
            self.use_z_wts = parms.use_z_wts
            if parms.use_z_wts:
                self.selection_fn_file = self.output_folder + self.handle + '_selFn.txt'
                self.generate_selfn(nbins=15)
            self.use_syst_wts = parms.use_syst_wts
            self.use_completeness_wts = parms.use_completeness_wts

            if parms.do_tessellation:
                # options for buffer mocks around survey boundaries
                if not os.access(parms.mock_file, os.F_OK):
                    if not parms.mock_file == '':
                        print('Could not find file %s containing buffer mocks!' % parms.mock_file)
                    print('Generating buffer mocks around survey edges ...')
                    print('\tbuffer mocks will have %0.1f x the galaxy number density' % parms.mock_dens_ratio)
                    sys.stdout.flush()
                    self.mock_dens_ratio = parms.mock_dens_ratio
                    self.generate_buffer()
                    # self.tracers now contains all tracer information (galaxies + buffers + guards)
                else:
                    print('Loading pre-computed buffer mocks from file %s' % parms.mock_file)
                    sys.stdout.flush()
                    if '.npy' in parms.mock_file:
                        buffers = np.load(parms.mock_file)
                    else:
                        buffers = np.loadtxt(parms.mock_file)
                    # recalculate the box length
                    select = buffers[:, 3] > -60  # exclude the guard particles
                    self.box_length, self.middle = self.get_box_length(buffers[select, :3])
                    if self.verbose:
                        print("\tUsing box length %0.2f" % self.box_length)
                    self.num_mocks = buffers.shape[0]
                    # join the buffers to the galaxy tracers
                    self.tracers = np.vstack([self.tracers, buffers])
                    # self.tracers now contains all tracer information (galaxies + buffers + guards)
                    self.num_part_total = self.num_tracers + self.num_mocks
                    self.mock_file = parms.mock_file

            # shift X, Y, Z to the Zobov box coordinate system
            new_box_posns = self.obs2zobovbox(self.tracers[:, :3])
            self.tracers[:, :3] = new_box_posns

        # for easy debugging: write all tracer positions to file
        # np.save(self.posn_file.replace('pos.dat', 'pos.npy'), self.tracers)

        self.num_non_edge = self.num_tracers

        # options for void-finding
        self.min_dens_cut = parms.min_dens_cut
        self.void_min_num = parms.void_min_num
        self.use_barycentres = parms.use_barycentres

        # prefix for naming void files
        self.void_prefix = 'zobov-' + parms.void_prefix

        # options for finding 'superclusters'
        self.find_clusters = parms.find_clusters
        if parms.find_clusters:
            self.cluster_min_num = parms.cluster_min_num
            self.max_dens_cut = parms.max_dens_cut
            self.cluster_prefix = 'zobov-' + parms.cluster_prefix

    def coords_radecz2std(self, ra, dec, redshift, syswt=1, comp=1):
        """Converts sky coordinates in (RA,Dec,redshift) to standard form, including comoving
        Cartesian coordinate information
        """

        # convert galaxy redshifts to comoving distances
        rdist = self.cosmo.get_comoving_distance(redshift)

        # convert RA, Dec angles in degrees to theta, phi in radians
        phi = ra * np.pi / 180.
        theta = np.pi / 2. - dec * np.pi / 180.

        # obtain Cartesian coordinates
        galaxies = np.zeros((self.num_tracers, 8))
        galaxies[:, 0] = rdist * np.sin(theta) * np.cos(phi)  # r*cos(ra)*cos(dec)
        galaxies[:, 1] = rdist * np.sin(theta) * np.sin(phi)  # r*sin(ra)*cos(dec)
        galaxies[:, 2] = rdist * np.cos(theta)  # r*sin(dec)
        # standard format includes RA, Dec, redshift info
        galaxies[:, 3] = ra
        galaxies[:, 4] = dec
        galaxies[:, 5] = redshift
        # add weights and completeness information
        galaxies[:, 6] = syswt
        galaxies[:, 7] = comp

        # remove any galaxies for which the syswt value is 0 (occasionally true in some mocks)
        galaxies = galaxies[galaxies[:, 6] > 0]

        self.tracers = galaxies

    def get_box_length(self, positions, pad=200):
        """Calculate the extent of a cubic box that will comfortably enclose the whole survey

        :param positions: Nx3 array of positions (of galaxies + buffers) to enclose
        :param pad: additional padding desired for safety

        :returns:
        ----
        box_len: box length required
        middle: coordinates of central point of the survey
        """

        extents = np.asarray([np.max(positions[:, i]) - np.min(positions[:, i]) for i in range(3)])
        middle = np.asarray([0.5 * (np.max(positions[:, i]) + np.min(positions[:, i])) for i in range(3)])
        max_extent = np.max(extents)
        box_len = max_extent + pad

        return box_len, middle

    def obs2zobovbox(self, positions):
        """Convert from Cartesian coordinates in observers ref. frame to ZOBOV box coords

        :param positions: Nx3 array of position coordinates to be converted

        :return: array of new coordinates
        """

        shift = self.middle - self.box_length / 2
        new_pos = positions - shift

        return new_pos

    def zobovbox2obs(self, positions):
        """Convert from Cartesian coordinates in ZOBOV box coords to observers ref. frame

        :param positions: Nx3 array of position coordinates to be converted

        :return: array of new coordinates
        """

        shift = self.middle - self.box_length / 2
        new_pos = positions + shift

        return new_pos

    def generate_mask(self):
        """Generates an approximate survey sky mask if none is provided, and saves to file

        :return: sky fraction covered by survey mask
        """

        nside = 128  # seems ok for BOSS data, but for very sparse surveys nside=64 might be required
        npix = hp.nside2npix(nside)

        # use tracer RA,Dec info to see which sky pixels are occupied
        phi = self.tracers[:, 3] * np.pi / 180.
        theta = np.pi / 2. - self.tracers[:, 4] * np.pi / 180.
        pixels = hp.ang2pix(nside, theta, phi)

        # crude binary mask
        mask = np.zeros(npix)
        mask[pixels] = 1.

        # write this mask to file
        hp.write_map(self.mask_file, mask)

        # return sky fraction
        f_sky = 1.0 * sum(mask) / len(mask)
        return f_sky

    def find_mask_boundary(self):
        """
        Finds pixels adjacent to but outside the survey mask

        :return: a healpy map instance with pixels set =1 if they are on the survey boundary, =0 if not
        """

        mask = hp.read_map(self.mask_file, verbose=False)
        nside = 512
        mask = hp.ud_grade(mask, nside)
        npix = hp.nside2npix(nside)
        boundary = np.zeros(npix)

        # find pixels outside the mask that neighbour pixels within it
        # do this step in a loop, to get a thicker boundary layer
        for j in range(int(3 + nside / 64)):
            if j == 0:
                filled_inds = np.nonzero(mask)[0]
            else:
                filled_inds = np.nonzero(boundary)[0]
            theta, phi = hp.pix2ang(nside, filled_inds)
            neigh_pix = hp.get_all_neighbours(nside, theta, phi)
            for i in range(neigh_pix.shape[1]):
                outsiders = neigh_pix[(mask[neigh_pix[:, i]] == 0) & (neigh_pix[:, i] > -1)
                                      & (boundary[neigh_pix[:, i]] == 0), i]
                # >-1 condition takes care of special case where neighbour wasn't found
                if j == 0:
                    boundary[outsiders] = 2
                else:
                    boundary[outsiders] = 1
        boundary[boundary == 2] = 0  # this sets a 1-pixel gap between boundary and survey so buffers are not too close

        return boundary

    def generate_buffer(self):
        """Method to generate buffer particles around the edges of survey volume to prevent and detect leakage of
        Voronoi cells outside survey region during the tessellation stage"""

        # set the buffer particle density
        buffer_dens = self.mock_dens_ratio * self.tracer_dens

        # get the survey mask
        mask = hp.read_map(self.mask_file, verbose=False)
        nside = hp.get_nside(mask)
        survey_pix = np.nonzero(mask)[0]
        numpix = len(survey_pix)

        # estimate the mean inter-particle separation
        mean_nn_distance = self.tracer_dens ** (-1. / 3)

        # ---- Step 1: buffer particles along the high-redshift cap---- #
        # get the maximum redshift of the survey galaxies
        z_high = np.max([np.max(self.tracers[:, 5]), self.z_max])

        # define the radial extents of the layer in which we will place the buffer particles
        # these choices are somewhat arbitrary, and could be optimized
        r_low = self.cosmo.get_comoving_distance(z_high) + mean_nn_distance * 0.5 #self.mock_dens_ratio ** (-1. / 3)
        r_high = r_low + mean_nn_distance * 1.5
        cap_volume = self.f_sky * 4. * np.pi * (r_high ** 3. - r_low ** 3.) / 3.

        # how many buffer particles fit in this cap
        num_high_mocks = int(np.ceil(buffer_dens * cap_volume))
        high_mocks = np.zeros((num_high_mocks, 8))

        # generate random radial positions within the cap
        rdist = (r_low ** 3. + (r_high ** 3. - r_low ** 3.) * np.random.rand(num_high_mocks)) ** (1. / 3)

        # generate mock angular positions within the survey mask
        # NOTE: these are not random positions, since they are all centred in a Healpix pixel
        # but for buffer particles this is not important (and is generally faster)
        while num_high_mocks > numpix:
            # more mock posns required than mask pixels, so upgrade mask to get more pixels
            nside *= 2
            mask = hp.ud_grade(mask, nside)
            survey_pix = np.nonzero(mask)[0]
            numpix = len(survey_pix)
        rand_pix = survey_pix[random.sample(range(numpix), num_high_mocks)]
        theta, phi = hp.pix2ang(nside, rand_pix)

        # convert to standard format
        high_mocks[:, 0] = rdist * np.sin(theta) * np.cos(phi)
        high_mocks[:, 1] = rdist * np.sin(theta) * np.sin(phi)
        high_mocks[:, 2] = rdist * np.cos(theta)
        high_mocks[:, 3] = phi * 180. / np.pi
        high_mocks[:, 4] = 90 - theta * 180. / np.pi
        high_mocks[:, 5] = -1  # all buffer particles are given redshift -1 to aid identification
        high_mocks[:, 6] = 0   # all buffer particles are given weight 0 to aid identification
        high_mocks[:, 7] = 0   # all buffer particles are given weight 0 to aid identification

        # farthest buffer particle
        self.r_far = np.max(rdist)

        if self.verbose:
            print("\tplaced %d buffer mocks at high-redshift cap" % num_high_mocks)

        buffers = high_mocks
        self.num_mocks = num_high_mocks
        # ------------------------------------------------------------- #

        # ----- Step 2: buffer particles along the low-redshift cap---- #
        z_low = np.min([np.min(self.tracers[:, 5]), self.z_min])
        if z_low > 0:
            # define the radial extents of the layer in which we will place the buffer particles
            # these choices are somewhat arbitrary, and could be optimized
            r_high = self.cosmo.get_comoving_distance(z_low) - mean_nn_distance * 0.5 #self.mock_dens_ratio ** (-1. / 3)
            r_low = r_high - mean_nn_distance * 1.5
            if r_high < 0:
                r_high = self.cosmo.get_comoving_distance(z_low)
            if r_low < 0:
                r_low = 0
            cap_volume = self.f_sky * 4. * np.pi * (r_high ** 3. - r_low ** 3.) / 3.

            # how many buffer particles fit in this cap
            num_low_mocks = int(np.ceil(buffer_dens * cap_volume))
            low_mocks = np.zeros((num_low_mocks, 8))

            # generate random radial positions within the cap
            rdist = (r_low ** 3. + (r_high ** 3. - r_low ** 3.) * np.random.rand(num_low_mocks)) ** (1. / 3)

            # generate mock angular positions within the survey mask
            # same as above -- these are not truly random but that's ok
            while num_low_mocks > numpix:
                # more mock posns required than mask pixels, so upgrade mask to get more pixels
                nside *= 2
                mask = hp.ud_grade(mask, nside)
                survey_pix = np.nonzero(mask)[0]
                numpix = len(survey_pix)
            rand_pix = survey_pix[random.sample(range(numpix), num_low_mocks)]
            theta, phi = hp.pix2ang(nside, rand_pix)

            # convert to standard format
            low_mocks[:, 0] = rdist * np.sin(theta) * np.cos(phi)
            low_mocks[:, 1] = rdist * np.sin(theta) * np.sin(phi)
            low_mocks[:, 2] = rdist * np.cos(theta)
            low_mocks[:, 3] = phi * 180. / np.pi
            low_mocks[:, 4] = 90 - theta * 180. / np.pi
            low_mocks[:, 5] = -1.  # all buffer particles are given redshift -1 to aid later identification
            low_mocks[:, 6] = 0    # all buffer particles are given weight 0 to aid identification
            low_mocks[:, 7] = 0    # all buffer particles are given weight 0 to aid identification

            # closest buffer particle
            self.r_near = np.min(rdist)

            if self.verbose:
                print("\tplaced %d buffer mocks at low-redshift cap" % num_low_mocks)

            buffers = np.vstack([buffers, low_mocks])
            self.num_mocks += num_low_mocks
        else:
            if self.verbose:
                print("\tno buffer mocks required at low-redshift cap")
        sys.stdout.flush()
        # ------------------------------------------------------------- #

        # ------ Step 3: buffer particles along the survey edges-------- #
        if self.f_sky < 1.0:
            # get the survey boundary
            boundary = self.find_mask_boundary()

            # where we will place the buffer mocks
            boundary_pix = np.nonzero(boundary)[0]
            numpix = len(boundary_pix)
            boundary_f_sky = 1.0 * len(boundary_pix) / len(boundary)
            boundary_nside = hp.get_nside(boundary)

            # how many buffer particles
            # boundary_volume = boundary_f_sky * 4. * np.pi * (self.r_far ** 3. - self.r_near ** 3.) / 3.
            boundary_volume = boundary_f_sky * 4. * np.pi * quad(lambda y: y ** 2, self.r_near, self.r_far)[0]
            num_bound_mocks = int(np.ceil(buffer_dens * boundary_volume))
            bound_mocks = np.zeros((num_bound_mocks, 8))

            # generate random radial positions within the boundary layer
            rdist = (self.r_near ** 3. + (self.r_far ** 3. - self.r_near ** 3.) *
                     np.random.rand(num_bound_mocks)) ** (1. / 3)

            # generate mock angular positions within the boundary layer
            # and same as above -- not truly random, but ok
            while num_bound_mocks > numpix:
                # more mocks required than pixels in which to place them, so upgrade mask
                boundary_nside *= 2
                boundary = hp.ud_grade(boundary, boundary_nside)
                boundary_pix = np.nonzero(boundary)[0]
                numpix = len(boundary_pix)
            rand_pix = boundary_pix[random.sample(range(numpix), num_bound_mocks)]
            theta, phi = hp.pix2ang(boundary_nside, rand_pix)

            # convert to standard format
            bound_mocks[:, 0] = rdist * np.sin(theta) * np.cos(phi)
            bound_mocks[:, 1] = rdist * np.sin(theta) * np.sin(phi)
            bound_mocks[:, 2] = rdist * np.cos(theta)
            bound_mocks[:, 3] = phi * 180. / np.pi
            bound_mocks[:, 4] = 90 - theta * 180. / np.pi
            bound_mocks[:, 5] = -1.  # all buffer particles are given redshift -1 to aid identification
            bound_mocks[:, 6] = 0    # all buffer particles are given weight 0 to aid identification
            bound_mocks[:, 7] = 0    # all buffer particles are given weight 0 to aid identification

            if self.verbose:
                print("\tplaced %d buffer mocks along the survey boundary edges" % num_bound_mocks)

            buffers = np.vstack([buffers, bound_mocks])
            self.num_mocks += num_bound_mocks
        else:
            if self.verbose:
                print("\tdata covers the full sky, no buffer mocks required along edges")
        sys.stdout.flush()
        # ------------------------------------------------------------- #

        # determine the size of the cubic box required
        self.box_length, self.middle = self.get_box_length(buffers[:, :3])
        if self.verbose:
            print("\tUsing box length %0.2f" % self.box_length)

        # ------ Step 4: guard buffers to stabilize the tessellation-------- #
        # (strictly speaking, this gives a lot of redundancy as the box is very big;
        # but it doesn't slow the tessellation too much and keeps coding simpler)

        # generate guard particle positions
        x = np.linspace(0.1, self.box_length - 0.1, self.guard_nums)
        guards = np.vstack(np.meshgrid(x, x, x)).reshape(3, -1).T

        # make a kdTree instance using all the galaxies and buffer mocks
        all_posns_obs = np.vstack([self.tracers[:, :3], buffers[:, :3]])
        # positions in Zobov box coordinates
        all_posns_box = self.obs2zobovbox(all_posns_obs)
        tree = cKDTree(all_posns_box, boxsize=self.box_length)

        # find the nearest neighbour distance for each of the guard particles
        nn_dist = np.empty(len(guards))
        for i in range(len(guards)):
            nn_dist[i], nnind = tree.query(guards[i, :], k=1)

        # drop all guards that are too close to existing points
        guards = guards[nn_dist > (self.box_length - 0.2) / self.guard_nums]
        # shift back to observer coordinates
        guards_obs = self.zobovbox2obs(guards)

        # convert to standard format
        num_guard_mocks = len(guards)
        guard_mocks = np.zeros((num_guard_mocks, 8))
        guard_mocks[:, :3] = guards_obs
        guard_mocks[:, 3:5] = -60.  # guards are given RA and Dec -60 as well to distinguish them from other buffers
        guard_mocks[:, 5] = -1.
        guard_mocks[:, 6] = 0
        guard_mocks[:, 7] = 0

        if self.verbose:
            print("\tadded %d guards to stabilize the tessellation" % num_guard_mocks)

        buffers = np.vstack([buffers, guard_mocks])
        self.num_mocks += num_guard_mocks
        # ------------------------------------------------------------------ #

        # write the buffer information to file for later reference
        mock_file = self.posn_file.replace('pos.dat', 'mocks.npy')
        if self.verbose:
            print('Buffer mocks written to file %s' % mock_file)
        np.save(mock_file, buffers)
        self.mock_file = mock_file
        sys.stdout.flush()

        # now add buffer particles to tracers
        self.tracers = np.vstack([self.tracers, buffers])

        self.num_part_total = self.num_tracers + self.num_mocks

    def generate_selfn(self, nbins=20):
        """
        Measures the redshift-dependence of the galaxy number density in equal-volume redshift bins, and writes to file

        :param nbins: number of bins to use
        :return:
        """

        if self.verbose:
            print('Determining survey redshift selection function ...')
        sys.stdout.flush()

        # first determine the equal volume bins
        r_near = self.cosmo.get_comoving_distance(self.z_min)
        r_far = self.cosmo.get_comoving_distance(self.z_max)
        rvals = np.linspace(r_near ** 3, r_far ** 3, nbins + 1)
        rvals = rvals ** (1. / 3)
        zsteps = self.cosmo.get_redshift(rvals)
        volumes = self.f_sky * 4 * np.pi * (rvals[1:] ** 3. - rvals[:-1] ** 3.) / 3.
        # (all elements of volumes should be equal)

        # get the tracer galaxy redshifts
        redshifts = self.tracers[:, 5]

        # histogram and calculate number density
        hist, zsteps = np.histogram(redshifts, bins=zsteps)
        nofz = hist / volumes
        # previously this was set to be the mean redshift of all galaxies in the bin
        # but that sometimes caused nans to appear if the bin were empty, leading to crashes later
        # so now it stores the central value of the bin instead
        zmeans = 0.5 * (zsteps[1:] + zsteps[:-1])

        output = np.zeros((len(zmeans), 3))
        output[:, 0] = zmeans
        output[:, 1] = nofz
        output[:, 2] = nofz / self.tracer_dens
        # write to file
        np.savetxt(self.selection_fn_file, output, fmt='%0.3f %0.4e %0.4f',
                   header='z n(z)[h^3/Mpc^3] f(z)[normed in units of overall mean]')

    def write_box_zobov(self):
        """
        Writes the tracer and mock position information to file in a ZOBOV-readable format
        :return:
        """

        with open(self.posn_file, 'w') as F:
            npart = np.array(self.num_part_total, dtype=np.int32)
            npart.tofile(F, format='%d')
            data = self.tracers[:, 0]
            data.tofile(F, format='%f')
            data = self.tracers[:, 1]
            data.tofile(F, format='%f')
            data = self.tracers[:, 2]
            data.tofile(F, format='%f')
            if not self.is_box:  # write RA, Dec, redshift, weights and completeness info too
                data = self.tracers[:, 3]
                data.tofile(F, format='%f')
                data = self.tracers[:, 4]
                data.tofile(F, format='%f')
                data = self.tracers[:, 5]
                data.tofile(F, format='%f')
                data = self.tracers[:, 6]
                data.tofile(F, format='%f')
                data = self.tracers[:, 7]
                data.tofile(F, format='%f')

    def delete_tracer_info(self):
        """
        removes the tracer information if no longer required, to save memory
        """

        self.tracers = 0

    def reread_tracer_info(self):
        """
        re-reads tracer information from Zobov-formatted file if required after previous deletion
        """

        self.tracers = np.empty((self.num_part_total, 8))
        with open(self.posn_file, 'r') as F:
            nparts = np.fromfile(F, dtype=np.int32, count=1)[0]
            if not nparts == self.num_part_total:  # sanity check
                sys.exit("nparts = %d in %s_pos.dat file does not match num_part_total = %d!"
                         % (nparts, self.handle, self.num_part_total))
            self.tracers[:, 0] = np.fromfile(F, dtype=np.float64, count=nparts)
            self.tracers[:, 1] = np.fromfile(F, dtype=np.float64, count=nparts)
            self.tracers[:, 2] = np.fromfile(F, dtype=np.float64, count=nparts)
            if not self.is_box:
                self.tracers[:, 3] = np.fromfile(F, dtype=np.float64, count=nparts)
                self.tracers[:, 4] = np.fromfile(F, dtype=np.float64, count=nparts)
                self.tracers[:, 5] = np.fromfile(F, dtype=np.float64, count=nparts)
                self.tracers[:, 6] = np.fromfile(F, dtype=np.float64, count=nparts)
                self.tracers[:, 7] = np.fromfile(F, dtype=np.float64, count=nparts)

    def write_config(self):
        """method to write configuration information for the ZOBOV run to file for later lookup"""

        info = 'handle = \'%s\'\nis_box = %s\nnum_tracers = %d\n' % (self.handle, self.is_box, self.num_tracers)
        info += 'num_mocks = %d\nnum_non_edge = %d\nbox_length = %f\n' % (self.num_mocks, self.num_non_edge,
                                                                          self.box_length)
        if not self.is_box:
            info += 'middle = np.array([%0.6e, %0.6e, %0.6e])\n' % (self.middle[0], self.middle[1], self.middle[2])
        info += 'tracer_dens = %e' % self.tracer_dens
        info_file = self.output_folder + self.handle + '_sample_info.py'
        with open(info_file, 'w') as F:
            F.write(info)

    def read_config(self):
        """method to read configuration file for information about previous ZOBOV run"""

        info_file = self.output_folder + self.handle + '_sample_info.py'
        if sys.version_info.major <= 2:
            import imp
            config = imp.load_source("name", info_file)
        elif sys.version_info.major == 3 and sys.version_info.minor <= 4:
            from importlib.machinery import SourceFileLoader
            config = SourceFileLoader("name", info_file).load_module()
        else:
            import importlib.util
            spec = importlib.util.spec_from_file_location("name", info_file)
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)

        self.num_mocks = config.num_mocks
        self.num_non_edge = config.num_non_edge
        self.box_length = config.box_length
        self.tracer_dens = config.tracer_dens
        self.num_part_total = config.num_mocks + config.num_tracers
        self.num_tracers = config.num_tracers
        if not self.is_box:
            self.middle = config.middle
        if not self.handle == config.handle:
            print('Warning: handle in config file does not match input! Check?')

    def zobov_wrapper(self):
        """
        Wrapper function to call C-based ZOBOV codes
        :return: 1 if tessellation successfully completed
        """

        # get the path to where the C executables are stored
        binpath = os.path.dirname(__file__).replace('python_tools', 'bin/')

        # ---run the tessellation--- #
        # there are 2 options: use MPI or not, chosen by the user
        # If the data is in a cubic box with PBC, tessellation must be done by voz1b1 and voztie
        # irrespective of choice of MPI  or not. If the data is from a survey, we use vozisol if no MPI
        # is available, or voz1b1_mpi and voztie if it is. If using voz1b1_mpi and voztie, we need to
        # flag edge galaxies separately using checkedges (handled automatically by vozisol).
        logfolder = self.output_folder + 'log/'
        if not os.access(logfolder, os.F_OK):
            os.makedirs(logfolder)
        if not self.use_mpi:
            if self.is_box:  # cannot use vozisol with PBC
                print("Calling voz1b1 and voztie to do the tessellation...")
                sys.stdout.flush()

                # ---Step 1: run voz1b1 on the sub-boxes, in series--- #
                logfile = logfolder + self.handle + '-voz1b1.out'
                log = open(logfile, "w")
                for b1 in range(self.zobov_box_div):
                    for b2 in range(self.zobov_box_div):
                        for b3 in range(self.zobov_box_div):

                            cmd = [binpath + 'voz1b1', self.posn_file, str(self.zobov_buffer), str(self.box_length),
                                    self.handle,str(self.zobov_box_div),str(b1),str(b2),str(b3)]
                            subprocess.call(cmd, stdout=log, stderr=log)
                log.close()

                # ---Step 2: tie the sub-boxes together using voztie--- #
                log = open(logfile, "a")
                cmd = [binpath + "voztie", str(self.zobov_box_div), self.handle]
                subprocess.call(cmd, stdout=log, stderr=log)
                log.close()

            else:  # no PBC, so use vozisol
                print("Calling vozisol to do the tessellation...")
                sys.stdout.flush()
                logfile = logfolder + self.handle + '-vozisol.out'
                log = open(logfile, "w")
                cmd = [binpath + "vozisol", self.posn_file, self.handle, str(self.box_length),
                       str(self.num_tracers), str(0.9e30)]
                subprocess.call(cmd, stdout=log, stderr=log)
                log.close()

            # check the tessellation was successful
            if not os.access("%s.vol" % self.handle, os.F_OK):
                sys.exit("Something went wrong with the tessellation. Aborting ...")
            
            # copy the .vol files to .trvol
            cmd = ["cp", "%s.vol" % self.handle, "%s.trvol" % self.handle]
            subprocess.call(cmd)
        else:
            print("MPI run: calling voz1b1 and voztie to do the tessellation...")
            sys.stdout.flush()

            # ---Step 1: run voz1b1 on the sub-boxes in parallel using voz1b1_mpi--- #
            logfile = logfolder + self.handle + '-voz1b1_mpi.out'
            log = open(logfile, "w")
            cmd = ['mpirun', binpath + 'voz1b1_mpi', self.posn_file, str(self.zobov_buffer), str(self.box_length),
                   str(self.zobov_box_div), self.handle]
            subprocess.call(cmd, stdout=log, stderr=log)
            log.close()

            # ---Step 2: tie the sub-boxes together using voztie--- #
            log = open(logfile, "a")
            cmd = [binpath + "voztie", str(self.zobov_box_div), self.handle]
            subprocess.call(cmd, stdout=log, stderr=log)
            log.close()

            # ---Step 3: check the tessellation was successful--- #
            if not os.access("%s.vol" % self.handle, os.F_OK):
                print("Something went wrong with the tessellation. Check log file!\nAborting ...")
                return 0

            # ---Step 4: copy the .vol files to .trvol--- #
            cmd = ["cp", "%s.vol" % self.handle, "%s.trvol" % self.handle]
            subprocess.call(cmd)

            # ---Step 5: if buffer mocks were used, remove them and flag edge galaxies--- #
            # (necessary because voz1b1 and voztie do not do this automatically)
            if self.num_mocks > 0:
                cmd = [binpath + "checkedges", self.handle, str(self.num_tracers), str(0.9e30)]
                logfile = logfolder + self.handle + '-checkedges.out'
                log = open(logfile, 'a')
                subprocess.call(cmd, stdout=log, stderr=log)
                log.close()

        print("Tessellation done.")
        sys.stdout.flush()

        # ---prepare files for running jozov--- #
        if self.is_box:
            # no preparation is required for void-finding
            if self.find_clusters:
                cmd = ["cp", "%s.vol" % self.handle, "%sc.vol" % self.handle]
                subprocess.call(cmd)
        else:
            # we need to account for buffer mocks, selection effects and systematics on relative densities
            # Voronoi volumes were calculated in units of the mean of all particles; we first change this to units of
            # the mean of galaxies only (no buffer mocks), then scale volumes up or down to account for variations in
            # the local means density (selection function and completeness, if desired) and galaxy systematics weights
            # (if desired). Original Voronoi volumes are also kept to calculate void volumes in correct units

            # ---Step 1: read the edge-modified Voronoi volumes--- #
            with open('./%s.vol' % self.handle, 'r') as F:
                npreal = np.fromfile(F, dtype=np.int32, count=1)[0]
                modvols = np.fromfile(F, dtype=np.float64, count=npreal)

            # ---Step 2: renormalize volumes in units of mean volume per galaxy--- #
            # (this step is necessary because otherwise the buffer mocks and bounding box affect the calculation)
            edgemask = modvols == 1.0 / 0.9e30  # effectively zero volume / infinite density
            modvols[np.logical_not(edgemask)] *= (self.tracer_dens * self.box_length ** 3.) / self.num_part_total
            # check for failures!
            if np.any(modvols[np.logical_not(edgemask)] == 0):
                sys.exit('Tessellation gave some zero-volume Voronoi cells - check log file!!\nAborting...')

            # Basic principle in next few steps is that the cell density relative to local mean is the inverse of the
            # cell volume relative to the local mean volume. Where the local mean density is higher, we increase the
            # relative volume to compensate and vice versa. Cell volumes are also adjusted by the inverse of systematic
            # weights.

            # ---Step 3: scale volumes accounting for z-dependent selection--- #
            if self.use_z_wts:
                redshifts = self.tracers[:self.num_tracers, 5]
                selfnbins = np.loadtxt(self.selection_fn_file)
                selfn = InterpolatedUnivariateSpline(selfnbins[:, 0], selfnbins[:, 2], k=1)
                # smooth with a Savitzky-Golay filter to remove high-frequency noise
                x = np.linspace(redshifts.min(), redshifts.max(), 1000)
                y = savgol_filter(selfn(x), 101, 3)
                # then linearly interpolate the filtered interpolation
                selfn = InterpolatedUnivariateSpline(x, y, k=1)
                # scale the cell volumes
                modfactors = selfn(redshifts[np.logical_not(edgemask)])
                modvols[np.logical_not(edgemask)] *= modfactors
                # check for failures!
                if np.any(modvols[np.logical_not(edgemask)] == 0):
                    sys.exit('Use of z-weights caused some zero-volume Voronoi cells - check input!!\nAborting...')

            # ---Step 4: scale volumes to account for systematics weights--- #
            if self.use_syst_wts:
                modfactors = self.tracers[:self.num_tracers, 6]
                # note division in next step
                modvols[np.logical_not(edgemask)] /= modfactors[np.logical_not(edgemask)]
                if np.any(modvols[np.logical_not(edgemask)] == 0):
                    sys.exit('Use of systematics weights caused some zero-volume Voronoi cells - check input!!' +
                             '\nAborting...')

            # ---Step 5: scale volumes accounting for angular completeness--- #
            if self.use_completeness_wts:
                modfactors = self.tracers[:self.num_tracers, 7]
                modvols[np.logical_not(edgemask)] *= modfactors[np.logical_not(edgemask)]
                if np.any(modvols[np.logical_not(edgemask)] == 0):
                    sys.exit('Use of completeness weights caused some zero-volume Voronoi cells - check input!!' +
                             '\nAborting...')

            # ---Step 6: write the scaled volumes to file--- #
            with open("./%s.vol" % self.handle, 'w') as F:
                npreal.tofile(F, format="%d")
                modvols.tofile(F, format="%f")

            # ---Step 7: if finding clusters, create the files required--- #
            if self.find_clusters:
                modvols[edgemask] = 0.9e30
                # and write to c.vol file
                with open("./%sc.vol" % self.handle, 'w') as F:
                    npreal.tofile(F, format="%d")
                    modvols.tofile(F, format="%f")

            # ---Step 8: set the number of non-edge galaxies--- #
            self.num_non_edge = self.num_tracers - sum(edgemask)

        # write a config file
        self.write_config()

        # ---run jozov to perform the void-finding--- #
        cmd = [binpath + "jozovtrvol", "v", self.handle, str(0), str(0)]
        logfile = logfolder + self.handle + '-jozov.out'
        log = open(logfile, 'w')
        subprocess.call(cmd, stdout=log, stderr=log)
        log.close()
        # this call to (modified version of) jozov sets NO density threshold, so ALL voids are merged without limit
        # and the FULL merged void heirarchy is output to file; distinct non-overlapping voids are later
        # obtained in post-processing

        # ---if finding clusters, run jozov again--- #
        if self.find_clusters:
            print(" ==== bonus: overdensity-finding with ZOBOV ==== ")
            sys.stdout.flush()
            cmd = [binpath + "jozovtrvol", "c", self.handle, str(0), str(0)]
            log = open(logfile, 'a')
            subprocess.call(cmd, stdout=log, stderr=log)
            log.close()

        # ---clean up: remove unnecessary files--- #
        for fileName in glob.glob("./part." + self.handle + ".*"):
            os.unlink(fileName)

        # ---clean up: move all other files to appropriate directory--- #
        raw_dir = self.output_folder + "rawZOBOV/"
        if not os.access(raw_dir, os.F_OK):
            os.makedirs(raw_dir)
        for fileName in glob.glob("./" + self.handle + "*"):
            cmd = ["mv", fileName, "%s." % raw_dir]
            subprocess.call(cmd)

        return 1

    def postprocess_voids(self):
        """
        Method to post-process raw ZOBOV output to obtain discrete set of non-overlapping voids. This method
        is hard-coded to NOT allow any void merging, since no objective (non-arbitrary) criteria can be defined
        to control merging, if allowed.
        """

        print('Post-processing voids ...')

        # ------------NOTE----------------- #
        # Actually, the current code is built from previous code that did have merging
        # functionality. This functionality is still technically present, but is controlled
        # by the following hard-coded parameters. If you know what you are doing, you can
        # change them.
        # --------------------------------- #
        dont_merge = True
        use_r_threshold = False
        r_threshold = 1.
        use_link_density_threshold = False
        link_density_threshold = 1.
        count_all_voids = True
        use_stripping = False
        strip_density_threshold = 1.
        if use_stripping:
            if (strip_density_threshold < self.min_dens_cut) or (strip_density_threshold < link_density_threshold):
                print('ERROR: incorrect use of strip_density_threshold\nProceeding with automatically corrected value')
                strip_density_threshold = max(self.min_dens_cut, link_density_threshold)
        # --------------------------------- #

        # the files with ZOBOV output
        zone_file = self.output_folder + 'rawZOBOV/' + self.handle + '.zone'
        void_file = self.output_folder + 'rawZOBOV/' + self.handle + '.void'
        list_file = self.output_folder + 'rawZOBOV/' + self.handle + '.txt'
        volumes_file = self.output_folder + 'rawZOBOV/' + self.handle + '.trvol'
        densities_file = self.output_folder + 'rawZOBOV/' + self.handle + '.vol'

        # new files after post-processing
        new_void_file = self.output_folder + self.void_prefix + ".void"
        new_list_file = self.output_folder + self.void_prefix + "_list.txt"

        # load the list of void candidates
        voidsread = np.loadtxt(list_file, skiprows=2)
        # sort in ascending order of minimum dens
        sorted_order = np.argsort(voidsread[:, 3])
        voidsread = voidsread[sorted_order]

        num_voids = len(voidsread[:, 0])
        vid = np.asarray(voidsread[:, 0], dtype=int)
        edgelist = np.asarray(voidsread[:, 1], dtype=int)
        vollist = voidsread[:, 4]
        numpartlist = np.asarray(voidsread[:, 5], dtype=int)
        rlist = voidsread[:, 9]

        # load the void hierarchy
        with open(void_file, 'r') as Fvoid:
            hierarchy = Fvoid.readlines()
        # sanity check
        nvoids = int(hierarchy[0])
        if nvoids != num_voids:
            sys.exit('Unequal void numbers in voidfile and listfile, %d and %d!' % (nvoids, num_voids))
        hierarchy = hierarchy[1:]

        # load the particle-zone info
        zonedata = np.loadtxt(zone_file, dtype='int', skiprows=1)

        # load the VTFE volume information
        with open(volumes_file, 'r') as File:
            npart = np.fromfile(File, dtype=np.int32, count=1)[0]
            if not npart == self.num_tracers:  # sanity check
                sys.exit('npart = %d in %s.trvol file does not match num_tracers = %d!'
                         % (npart, self.handle, self.num_tracers))
            vols = np.fromfile(File, dtype=np.float64, count=npart)

        # load the VTFE density information
        with open(densities_file, 'r') as File:
            npart = np.fromfile(File, dtype=np.int32, count=1)[0]
            if not npart == self.num_tracers:  # sanity check
                sys.exit("npart = %d in %s.vol file does not match num_tracers = %d!"
                         % (npart, self.handle, self.num_tracers))
            densities = np.fromfile(File, dtype=np.float64, count=npart)
            densities = 1. / densities

        # mean volume per particle in box (including all buffer mocks)
        meanvol_trc = (self.box_length ** 3.) / self.num_part_total

        # parse the list of structures, separating distinct voids and performing minimal pruning
        with open(new_void_file, 'w') as Fnewvoid:
            with open(new_list_file, 'w') as Fnewlist:

                # initialize variables
                counted_zones = np.empty(0, dtype=int)
                edge_flag = np.empty(0, dtype=int)
                wtd_avg_dens = np.empty(0, dtype=int)
                num_acc = 0

                for i in range(num_voids):
                    coredens = voidsread[i, 3]
                    voidline = hierarchy[sorted_order[i]].split()
                    pos = 1
                    num_zones_to_add = int(voidline[pos])
                    finalpos = pos + num_zones_to_add + 1
                    rval = float(voidline[pos + 1])
                    rstopadd = rlist[i]
                    num_adds = 0
                    if rval >= 1 and coredens < self.min_dens_cut and numpartlist[i] >= self.void_min_num \
                            and (count_all_voids or vid[i] not in counted_zones):
                        # this void passes basic pruning
                        add_more = True
                        num_acc += 1
                        zonelist = vid[i]
                        total_vol = vollist[i]
                        total_num_parts = numpartlist[i]
                        zonestoadd = []
                        while num_zones_to_add > 0 and add_more:  # more zones can potentially be added
                            zonestoadd = np.asarray(voidline[pos + 2:pos + num_zones_to_add + 2], dtype=int)
                            dens = rval * coredens
                            rsublist = rlist[np.in1d(vid, zonestoadd)]
                            volsublist = vollist[np.in1d(vid, zonestoadd)]
                            partsublist = numpartlist[np.in1d(vid, zonestoadd)]
                            if dont_merge or (use_link_density_threshold and dens > link_density_threshold) or \
                                    (use_r_threshold > 0 and max(rsublist) > r_threshold):
                                # cannot add these zones
                                rstopadd = rval
                                add_more = False
                                finalpos -= (num_zones_to_add + 1)
                            else:
                                # keep adding zones
                                zonelist = np.append(zonelist, zonestoadd)
                                num_adds += num_zones_to_add
                                total_vol += np.sum(volsublist)  #
                                total_num_parts += np.sum(partsublist)  #
                            pos += num_zones_to_add + 2
                            num_zones_to_add = int(voidline[pos])
                            rval = float(voidline[pos + 1])
                            if add_more:
                                finalpos = pos + num_zones_to_add + 1

                        counted_zones = np.append(counted_zones, zonelist)
                        if use_stripping:
                            member_ids = np.logical_and(densities[:] < strip_density_threshold,
                                                        np.in1d(zonedata, zonelist))
                        else:
                            member_ids = np.in1d(zonedata, zonelist)

                        # if using void "stripping" functionality, recalculate void volume and number of particles
                        if use_stripping:
                            total_vol = np.sum(vols[member_ids])
                            total_num_parts = len(vols[member_ids])

                        # check if the void is edge-contaminated (useful for observational surveys only)
                        if 1 in edgelist[np.in1d(vid, zonestoadd)]:
                            edge_flag = np.append(edge_flag, 1)
                        else:
                            edge_flag = np.append(edge_flag, 0)

                        # average density of member cells weighted by cell volumes
                        w_a_d = np.sum(vols[member_ids] * densities[member_ids]) / np.sum(vols[member_ids])
                        wtd_avg_dens = np.append(wtd_avg_dens, w_a_d)

                        # set the new line for the .void file
                        newvoidline = voidline[:finalpos]
                        if not add_more:
                            newvoidline.append(str(0))
                        newvoidline.append(str(rstopadd))
                        # write line to the output .void file
                        for j in range(len(newvoidline)):
                            Fnewvoid.write(newvoidline[j] + '\t')
                        Fnewvoid.write('\n')
                        if rstopadd > 10 ** 20:
                            rstopadd = -1  # only structures entirely surrounded by edge particles
                        # write line to the output _list.txt file
                        Fnewlist.write('%d\t%d\t%f\t%d\t%d\t%d\t%f\t%f\n' % (vid[i], int(voidsread[i, 2]), coredens,
                                                                             int(voidsread[i, 5]), num_adds + 1,
                                                                             total_num_parts, total_vol * meanvol_trc,
                                                                             rstopadd))

        # tidy up the files
        # insert first line with number of voids to the new .void file
        with open(new_void_file, 'r+') as Fnewvoid:
            old = Fnewvoid.read()
            Fnewvoid.seek(0)
            topline = "%d\n" % num_acc
            Fnewvoid.write(topline + old)

        # insert header to the _list.txt file
        listdata = np.loadtxt(new_list_file)
        header = '%d non-edge tracers in %s, %d voids\n' % (self.num_non_edge, self.handle, num_acc)
        header = header + 'VoidID CoreParticle CoreDens Zone#Parts Void#Zones Void#Parts VoidVol(Mpc/h^3) VoidDensRatio'
        np.savetxt(new_list_file, listdata, fmt='%d %d %0.6f %d %d %d %0.6f %0.6f', header=header)

        # now find void centres and create the void catalogue files
        edge_flag = self.find_void_circumcentres(num_acc, wtd_avg_dens, edge_flag)

        if self.use_barycentres:
            if not os.access(self.output_folder + "barycentres/", os.F_OK):
                os.makedirs(self.output_folder + "barycentres/")
            self.find_void_barycentres(num_acc, edge_flag, use_stripping, strip_density_threshold)

    def find_void_circumcentres(self, num_struct, wtd_avg_dens, edge_flag):
        """Method that checks a list of processed voids, finds the void minimum density centres and writes
        the void catalogue file.

        Arguments:
            num_struct: integer number of voids after pruning
            wtd_avg_dens: float array of shape (num_struct,), weighted average void densities from post-processing
            edge_flag: integer array of shape (num_struct,), edge contamination flags
        """

        print("Identified %d potential voids. Now extracting circumcentres ..." % num_struct)
        sys.stdout.flush()

        # set the filenames
        densities_file = self.output_folder + "rawZOBOV/" + self.handle + ".vol"
        adjacency_file = self.output_folder + "rawZOBOV/" + self.handle + ".adj"
        list_file = self.output_folder + self.void_prefix + "_list.txt"
        info_file = self.output_folder + self.void_prefix + "_cat.txt"

        # load the VTFE density information
        with open(densities_file, 'r') as File:
            npart = np.fromfile(File, dtype=np.int32, count=1)[0]
            if not npart == self.num_tracers:  # sanity check
                sys.exit("npart = %d in %s.vol file does not match num_tracers = %d!"
                         % (npart, self.handle, self.num_tracers))
            densities = np.fromfile(File, dtype=np.float64, count=npart)
            densities = 1. / densities

        # check whether tracer information is present, re-read in if required
        if not len(self.tracers) == self.num_part_total:
            self.reread_tracer_info()
        # extract the x,y,z positions of the galaxies only (no buffer mocks)
        positions = self.tracers[:self.num_tracers, :3]

        list_array = np.loadtxt(list_file)
        v_id = np.asarray(list_array[:, 0], dtype=int)
        corepart = np.asarray(list_array[:, 1], dtype=int)

        # read and assign adjacencies from ZOBOV output
        with open(adjacency_file, 'r') as AdjFile:
            npfromadj = np.fromfile(AdjFile, dtype=np.int32, count=1)[0]
            if not npfromadj == self.num_tracers:
                sys.exit("npart = %d from adjacency file does not match num_tracers = %d!"
                         % (npfromadj, self.num_tracers))
            partadjs = [[] for i in range(npfromadj)]  # list of lists to record adjacencies - is there a better way?
            partadjcount = np.zeros(npfromadj, dtype=np.int32)  # counter to monitor adjacencies
            nadj = np.fromfile(AdjFile, dtype=np.int32, count=npfromadj)  # number of adjacencies for each particle
            # load up the adjacencies from ZOBOV output
            for i in range(npfromadj):
                numtomatch = np.fromfile(AdjFile, dtype=np.int32, count=1)[0]
                if numtomatch > 0:
                    # particle numbers of adjacent particles
                    adjpartnumbers = np.fromfile(AdjFile, dtype=np.int32, count=numtomatch)
                    # keep track of how many adjacencies had already been assigned
                    oldcount = partadjcount[i]
                    newcount = oldcount + len(adjpartnumbers)
                    partadjcount[i] = newcount
                    # and now assign new adjacencies
                    partadjs[i][oldcount:newcount] = adjpartnumbers
                    # now also assign the reverse adjacencies
                    # (ZOBOV records only (i adj j) or (j adj i), not both)
                    for index in adjpartnumbers:
                        partadjs[index].append(i)
                    partadjcount[adjpartnumbers] += 1

        if self.is_box:
            info_output = np.zeros((num_struct, 9))
        else:
            info_output = np.zeros((num_struct, 11))
        circumcentres = np.empty((num_struct, 3))
        eff_rad = (3.0 * list_array[:, 6] / (4 * np.pi)) ** (1.0 / 3)

        # loop over void cores, calculating circumcentres and writing to file
        for i in range(num_struct):
            # get adjacencies of the core particle
            coreadjs = partadjs[corepart[i]]
            adj_dens = densities[coreadjs]

            # get the 3 lowest density mutually adjacent neighbours of the core particle
            first_nbr = coreadjs[np.argmin(adj_dens)]
            mutualadjs = np.intersect1d(coreadjs, partadjs[first_nbr], assume_unique=True)
            if len(mutualadjs) == 0:
                circumcentres[i] = np.asarray([0, 0, 0])
                edge_flag[i] = 2
            else:
                mutualadj_dens = densities[mutualadjs]
                second_nbr = mutualadjs[np.argmin(mutualadj_dens)]
                finaladjs = np.intersect1d(mutualadjs, partadjs[second_nbr], assume_unique=True)
                if len(finaladjs) == 0:  # something has gone wrong at tessellation stage!
                    circumcentres[i] = np.asarray([0, 0, 0])
                    edge_flag[i] = 2
                else:  # can calculate circumcentre position
                    finaladj_dens = densities[finaladjs]
                    third_nbr = finaladjs[np.argmin(finaladj_dens)]

                    # collect positions of the vertices
                    vertex_pos = np.zeros((4, 3))
                    vertex_pos[0, :] = positions[corepart[i], :]
                    vertex_pos[1, :] = positions[first_nbr, :]
                    vertex_pos[2, :] = positions[second_nbr, :]
                    vertex_pos[3, :] = positions[third_nbr, :]
                    if self.is_box:  # need to adjust for periodic BC
                        shift_inds = abs(vertex_pos[0, 0] - vertex_pos[:, 0]) > self.box_length / 2.0
                        vertex_pos[shift_inds, 0] += self.box_length * np.sign(vertex_pos[0, 0] -
                                                                               vertex_pos[shift_inds, 0])
                        shift_inds = abs(vertex_pos[0, 1] - vertex_pos[:, 1]) > self.box_length / 2.0
                        vertex_pos[shift_inds, 1] += self.box_length * np.sign(vertex_pos[0, 1] -
                                                                               vertex_pos[shift_inds, 1])
                        shift_inds = abs(vertex_pos[0, 2] - vertex_pos[:, 2]) > self.box_length / 2.0
                        vertex_pos[shift_inds, 2] += self.box_length * np.sign(vertex_pos[0, 2] -
                                                                               vertex_pos[shift_inds, 2])

                    # solve for the circumcentre; for more details on this method and its stability,
                    # see http://www.ics.uci.edu/~eppstein/junkyard/circumcentre.html
                    a = np.bmat([[2 * np.dot(vertex_pos, vertex_pos.T), np.ones((4, 1))],
                                 [np.ones((1, 4)), np.zeros((1, 1))]])
                    b = np.hstack((np.sum(vertex_pos * vertex_pos, axis=1), np.ones((1))))
                    x = np.linalg.solve(a, b)
                    bary_coords = x[:-1]
                    circumcentres[i, :] = np.dot(bary_coords, vertex_pos)

            if self.is_box:
                # put centre coords back within the fiducial box if they have leaked out
                if circumcentres[i, 0] < 0 or circumcentres[i, 0] > self.box_length:
                    circumcentres[i, 0] -= self.box_length * np.sign(circumcentres[i, 0])
                if circumcentres[i, 1] < 0 or circumcentres[i, 1] > self.box_length:
                    circumcentres[i, 1] -= self.box_length * np.sign(circumcentres[i, 1])
                if circumcentres[i, 2] < 0 or circumcentres[i, 2] > self.box_length:
                    circumcentres[i, 2] -= self.box_length * np.sign(circumcentres[i, 2])

        info_output[:, 0] = v_id
        info_output[:, 4] = eff_rad
        info_output[:, 5] = list_array[:, 2] - 1.
        info_output[:, 6] = wtd_avg_dens - 1.
        info_output[:, 7] = (wtd_avg_dens - 1.) * eff_rad**1.2
        info_output[:, 8] = list_array[:, 7]
        if self.is_box:
            info_output[:, 1:4] = circumcentres[:, :3]
        else:
            centre_obs = self.zobovbox2obs(circumcentres)  # circumcentres - self.box_length / 2.0
            rdist = np.linalg.norm(centre_obs, axis=1)
            eff_angrad = np.degrees(eff_rad / rdist)
            centre_redshifts = self.cosmo.get_redshift(rdist)
            centre_dec = 90 - np.degrees((np.arccos(centre_obs[:, 2] / rdist)))
            centre_ra = np.degrees(np.arctan2(centre_obs[:, 1], centre_obs[:, 0]))
            centre_ra[centre_ra < 0] += 360
            mask = hp.read_map(self.mask_file, verbose=False)
            nside = hp.get_nside(mask)
            pixel = hp.ang2pix(nside, np.deg2rad(90 - centre_dec), np.deg2rad(centre_ra))
            centre_redshifts[mask[pixel] == 0] = -1
            centre_dec[mask[pixel] == 0] = -60
            centre_ra[mask[pixel] == 0] = -60
            eff_angrad[mask[pixel] == 0] = 0
            edge_flag[mask[pixel] == 0] = 2
            out_of_redshift = np.logical_or(centre_redshifts < self.z_min, centre_redshifts > self.z_max)
            centre_redshifts[out_of_redshift] = -1
            centre_dec[out_of_redshift] = -60
            centre_ra[out_of_redshift] = -60
            eff_angrad[out_of_redshift] = 0
            edge_flag[out_of_redshift] = 2

            info_output[:, 1] = centre_ra
            info_output[:, 2] = centre_dec
            info_output[:, 3] = centre_redshifts
            info_output[:, 9] = eff_angrad
            info_output[:, 10] = edge_flag

            info_output = info_output[edge_flag < 2]  # remove all the tessellation failures
            print('Removed %d edge-contaminated voids' % (num_struct - len(info_output)))

        # save output data to file
        header = "%d voids from %s\n" % (len(info_output), self.handle)
        if self.is_box:
            header = header + 'VoidID XYZ[3](Mpc/h) R_eff(Mpc/h) delta_min delta_avg lambda_v DensRatio'
            np.savetxt(info_file, info_output, fmt='%d %0.6f %0.6f %0.6f %0.3f %0.6f %0.6f %0.6f %0.6f', header=header)
        else:
            header = header + 'VoidID RA(deg) Dec(deg) redshift R_eff(Mpc/h) delta_min delta_avg lambda_v ' + \
                     'DensRatio Theta_eff(deg) EdgeFlag'
            np.savetxt(info_file, info_output, fmt='%d %0.6f %0.3f %0.3f %0.4f %0.3f %0.6f %0.6f %0.6f %0.6f %d',
                       header=header)

        return edge_flag

    def find_void_barycentres(self, num_struct, edge_flag, use_stripping=False, strip_density_threshold=1.):
        """Method that checks a list of processed voids, finds the void barycentres and writes
        the void catalogue file.

        Arguments:
            num_struct: integer number of voids after pruning
            edge_flag: integer array of shape (num_struct,), edge contamination flags
            use_stripping: bool,optional (default is False, don't change unless you know what you're doing!)
            strip_density_threshold: float, optional (default 1.0, not required unless use_stripping is True)
        """

        print('Now extracting void barycentres ...')
        sys.stdout.flush()

        # set the filenames
        vol_file = self.output_folder + 'rawZOBOV/' + self.handle + '.trvol'
        dens_file = self.output_folder + 'rawZOBOV/' + self.handle + '.vol'
        zone_file = self.output_folder + 'rawZOBOV/' + self.handle + '.zone'
        hierarchy_file = self.output_folder + self.void_prefix + '.void'
        list_file = self.output_folder + self.void_prefix + '_list.txt'
        info_file = self.output_folder + 'barycentres/' + self.void_prefix + '_baryC_cat.txt'

        # load up the particle-zone info
        zonedata = np.loadtxt(zone_file, dtype='int', skiprows=1)

        # load the VTFE volume information
        with open(vol_file, 'r') as File:
            npart = np.fromfile(File, dtype=np.int32, count=1)[0]
            if not npart == self.num_tracers:  # sanity check
                sys.exit('npart = %d in %s.trvol file does not match num_tracers = %d!'
                         % (npart, self.handle, self.num_tracers))
            vols = np.fromfile(File, dtype=np.float64, count=npart)

        # load the VTFE density information
        with open(dens_file, 'r') as File:
            npart = np.fromfile(File, dtype=np.int32, count=1)[0]
            if not npart == self.num_tracers:  # sanity check
                sys.exit("npart = %d in %s.vol file does not match num_tracers = %d!"
                         % (npart, self.handle, self.num_tracers))
            densities = np.fromfile(File, dtype=np.float64, count=npart)
            densities = 1. / densities

        # mean volume per particle in box (including all buffer mocks)
        meanvol_trc = (self.box_length ** 3.) / self.num_part_total

        # check whether tracer information is present, re-read in if required
        if not len(self.tracers) == self.num_part_total:
            self.reread_tracer_info()
        # extract the x,y,z positions of the galaxies only (no buffer mocks)
        positions = self.tracers[:self.num_tracers, :3]

        list_array = np.loadtxt(list_file, skiprows=2)
        if self.is_box:
            info_output = np.zeros((num_struct, 9))
        else:
            info_output = np.zeros((num_struct, 11))
        centres = np.empty((num_struct, 3))
        eff_rad = np.zeros(num_struct)
        wtd_avg_dens = np.zeros(num_struct)

        with open(hierarchy_file, 'r') as FHierarchy:
            FHierarchy.readline()  # skip the first line, contains total number of structures
            for i in range(num_struct):
                # get the member zones of the structure
                structline = (FHierarchy.readline()).split()
                pos = 1
                add_zones = int(structline[pos]) > 0
                member_zones = np.asarray(structline[0], dtype=int)
                while add_zones:
                    num_zones_to_add = int(structline[pos])
                    zonestoadd = np.asarray(structline[pos + 2:pos + num_zones_to_add + 2], dtype=int)
                    member_zones = np.append(member_zones, zonestoadd)
                    pos += num_zones_to_add + 2
                    add_zones = int(structline[pos]) > 0

                # get the member particles for these zones
                if use_stripping:
                    member_ids = np.logical_and(densities[:] < strip_density_threshold, np.in1d(zonedata, member_zones))
                else:  # stripDens functionality disabled
                    member_ids = np.in1d(zonedata, member_zones)
                member_x = positions[member_ids, 0] - positions[int(list_array[i, 1]), 0]
                member_y = positions[member_ids, 1] - positions[int(list_array[i, 1]), 1]
                member_z = positions[member_ids, 2] - positions[int(list_array[i, 1]), 2]
                member_vols = vols[member_ids]
                member_dens = densities[member_ids]

                if self.is_box:
                    # must account for periodic boundary conditions, assume box coordinates in range [0,box_length]!
                    shift_vec = np.zeros((len(member_x), 3))
                    shift_x_ids = abs(member_x) > self.box_length / 2.0
                    shift_y_ids = abs(member_y) > self.box_length / 2.0
                    shift_z_ids = abs(member_z) > self.box_length / 2.0
                    shift_vec[shift_x_ids, 0] = -np.copysign(self.box_length, member_x[shift_x_ids])
                    shift_vec[shift_y_ids, 1] = -np.copysign(self.box_length, member_y[shift_y_ids])
                    shift_vec[shift_z_ids, 2] = -np.copysign(self.box_length, member_z[shift_z_ids])
                    member_x += shift_vec[:, 0]
                    member_y += shift_vec[:, 1]
                    member_z += shift_vec[:, 2]

                # volume-weighted barycentre of the structure
                centres[i, 0] = np.average(member_x, weights=member_vols) + positions[int(list_array[i, 1]), 0]
                centres[i, 1] = np.average(member_y, weights=member_vols) + positions[int(list_array[i, 1]), 1]
                centres[i, 2] = np.average(member_z, weights=member_vols) + positions[int(list_array[i, 1]), 2]

                # put centre coords back within the fiducial box if they have leaked out
                if self.is_box:
                    if centres[i, 0] < 0 or centres[i, 0] > self.box_length:
                        centres[i, 0] -= self.box_length * np.sign(centres[i, 0])
                    if centres[i, 1] < 0 or centres[i, 1] > self.box_length:
                        centres[i, 1] -= self.box_length * np.sign(centres[i, 1])
                    if centres[i, 2] < 0 or centres[i,2] > self.box_length:
                        centres[i, 2] -= self.box_length * np.sign(centres[i, 2])

                # total volume of structure in Mpc/h, and effective radius
                void_vol = np.sum(member_vols) * meanvol_trc
                eff_rad[i] = (3.0 * void_vol / (4 * np.pi)) ** (1.0 / 3)

                # average density of member cells weighted by cell volumes
                wtd_avg_dens[i] = np.average(member_dens, weights=member_vols)

            info_output[:, 0] = list_array[:, 0]
            info_output[:, 4] = eff_rad
            info_output[:, 5] = list_array[:, 2] - 1.
            info_output[:, 6] = wtd_avg_dens - 1.
            info_output[:, 7] = (wtd_avg_dens - 1.) * eff_rad ** 1.2
            info_output[:, 8] = list_array[:, 7]
            if self.is_box:
                info_output[:, 1:4] = centres[:, :3]
            else:
                centre_obs = self.zobovbox2obs(centres) # centres - self.box_length / 2.0
                rdist = np.linalg.norm(centre_obs, axis=1)
                eff_angrad = np.degrees(eff_rad / rdist)
                centre_redshifts = self.cosmo.get_redshift(rdist)
                centre_dec = 90 - np.degrees((np.arccos(centre_obs[:, 2] / rdist)))
                centre_ra = np.degrees(np.arctan2(centre_obs[:, 1], centre_obs[:, 0]))
                centre_ra[centre_ra < 0] += 360
                mask = hp.read_map(self.mask_file, verbose=False)
                nside = hp.get_nside(mask)
                pixel = hp.ang2pix(nside, np.deg2rad(90 - centre_dec), np.deg2rad(centre_ra))
                centre_redshifts[mask[pixel] == 0] = -1
                centre_dec[mask[pixel] == 0] = -60
                centre_ra[mask[pixel] == 0] = -60
                eff_angrad[mask[pixel] == 0] = 0
                edge_flag[mask[pixel] == 0] = 2
                out_of_redshift = np.logical_or(centre_redshifts < self.z_min, centre_redshifts > self.z_max)
                centre_redshifts[out_of_redshift] = -1
                centre_dec[out_of_redshift] = -60
                centre_ra[out_of_redshift] = -60
                eff_angrad[out_of_redshift] = 0
                edge_flag[out_of_redshift] = 2

                info_output[:, 1] = centre_ra
                info_output[:, 2] = centre_dec
                info_output[:, 3] = centre_redshifts
                info_output[:, 9] = eff_angrad
                info_output[:, 10] = edge_flag
                info_output = info_output[edge_flag < 2]  # remove all the tessellation failures

        # save output data to file
        header = "%d voids from %s\n" % (len(info_output), self.handle)
        if self.is_box:
            header = header + 'VoidID XYZ[3](Mpc/h) R_eff(Mpc/h) delta_min delta_avg lambda_v DensRatio'
            np.savetxt(info_file, info_output, fmt='%d %0.6f %0.6f %0.6f %0.3f %0.6f %0.6f %0.6f %0.6f', header=header)
        else:
            header = header + 'VoidID RA(deg) Dec(deg) redshift R_eff(Mpc/h) delta_min delta_avg lambda_v' + \
                     'DensRatio Theta_eff(deg) EdgeFlag'
            np.savetxt(info_file, info_output, fmt='%d %0.6f %0.3f %0.3f %0.4f %0.3f %0.6f %0.6f %0.6f %0.6f %d',
                       header=header)

    def postprocess_clusters(self):
        """
        Method to post-process raw ZOBOV output to obtain discrete set of non-overlapping 'superclusters'. This method
        is hard-coded to NOT allow any supercluster merging, since no objective (non-arbitrary) criteria can be defined
        to control merging anyway.
        """

        print('Post-processing superclusters ...')
        sys.stdout.flush()

        # ------------NOTE----------------- #
        # Actually, the current code is built from previous code that did have merging
        # functionality. This functionality is still technically present, but is controlled
        # by the following hard-coded parameters. If you know what you are doing, you can
        # change them.
        # --------------------------------- #
        dont_merge = True
        use_r_threshold = False
        r_threshold = 2.
        use_link_density_threshold = False
        link_density_threshold = 1.
        count_all_clusters = True
        use_stripping = False
        strip_density_threshold = 1.
        if use_stripping:
            if (strip_density_threshold > self.max_dens_cut) or (strip_density_threshold > link_density_threshold):
                print('ERROR: incorrect use of strip_density_threshold\nProceeding with automatically corrected value')
                strip_density_threshold = max(self.max_dens_cut, link_density_threshold)
        # --------------------------------- #

        # the files with ZOBOV output
        zone_file = self.output_folder + "rawZOBOV/" + self.handle + "c.zone"
        clust_file = self.output_folder + "rawZOBOV/" + self.handle + "c.void"
        list_file = self.output_folder + "rawZOBOV/" + self.handle + "c.txt"
        vol_file = self.output_folder + "rawZOBOV/" + self.handle + ".trvol"
        dens_file = self.output_folder + "rawZOBOV/" + self.handle + ".vol"
        info_file = self.output_folder + self.cluster_prefix + "_cat.txt"

        # new files after post-processing
        new_clust_file = self.output_folder + self.cluster_prefix + ".void"
        new_list_file = self.output_folder + self.cluster_prefix + "_list.txt"

        # load the list of supercluster candidates
        clustersread = np.loadtxt(list_file, skiprows=2)
        # sort in desc order of max dens
        sorted_order = np.argsort(1. / clustersread[:, 3])
        clustersread = clustersread[sorted_order]

        num_clusters = len(clustersread[:, 0])
        vid = np.asarray(clustersread[:, 0], dtype=int)
        edgelist = np.asarray(clustersread[:, 1], dtype=int)
        vollist = clustersread[:, 4]
        numpartlist = np.asarray(clustersread[:, 5], dtype=int)
        rlist = clustersread[:, 9]

        # load up the cluster hierarchy
        with open(clust_file, 'r') as Fclust:
            hierarchy = Fclust.readlines()
        nclusters = int(hierarchy[0])
        if nclusters != num_clusters:
            sys.exit('Unequal void numbers in clustfile and listfile, %d and %d!' % (nclusters, num_clusters))
        hierarchy = hierarchy[1:]

        # load up the particle-zone info
        zonedata = np.loadtxt(zone_file, dtype='int', skiprows=1)

        # load the VTFE volume information
        with open(vol_file, 'r') as File:
            npart = np.fromfile(File, dtype=np.int32, count=1)[0]
            if not npart == self.num_tracers:  # sanity check
                sys.exit('npart = %d in %s.trvol file does not match num_tracers = %d!'
                         % (npart, self.handle, self.num_tracers))
            vols = np.fromfile(File, dtype=np.float64, count=npart)

        # load the VTFE density information
        with open(dens_file, 'r') as File:
            npart = np.fromfile(File, dtype=np.int32, count=1)[0]
            if not npart == self.num_tracers:  # sanity check
                sys.exit("npart = %d in %s.cvol file does not match num_tracers = %d!"
                         % (npart, self.handle, self.num_tracers))
            densities = np.fromfile(File, dtype=np.float64, count=npart)
            densities = 1. / densities

        # check whether tracer information is present, re-read in if required
        if not len(self.tracers) == self.num_part_total:
            self.reread_tracer_info()
        # extract the x,y,z positions of the galaxies only (no buffer mocks)
        positions = self.tracers[:self.num_tracers, :3]

        # mean volume per tracer particle
        meanvol_trc = (self.box_length ** 3.) / self.num_part_total

        with open(new_clust_file, 'w') as Fnewclust:
            with open(new_list_file, 'w') as Fnewlist:

                # initialize variables
                counted_zones = np.empty(0, dtype=int)
                edge_flag = np.empty(0, dtype=int)
                wtd_avg_dens = np.empty(0, dtype=int)
                num_acc = 0

                for i in range(num_clusters):
                    coredens = clustersread[i, 3]
                    clustline = hierarchy[sorted_order[i]].split()
                    pos = 1
                    num_zones_to_add = int(clustline[pos])
                    finalpos = pos + num_zones_to_add + 1
                    rval = float(clustline[pos + 1])
                    rstopadd = rlist[i]
                    num_adds = 0
                    if rval >= 1 and coredens > self.max_dens_cut and numpartlist[i] >= self.cluster_min_num \
                            and (count_all_clusters or vid[i] not in counted_zones):
                        # this zone qualifies as a seed zone
                        add_more = True
                        num_acc += 1
                        zonelist = [vid[i]]
                        total_vol = vollist[i]
                        total_num_parts = numpartlist[i]
                        zonestoadd = []
                        while num_zones_to_add > 0 and add_more:
                            zonestoadd = np.asarray(clustline[pos + 2:pos + num_zones_to_add + 2], dtype=int)
                            dens = coredens / rval
                            rsublist = rlist[np.in1d(vid, zonestoadd)]
                            volsublist = vollist[np.in1d(vid, zonestoadd)]
                            partsublist = numpartlist[np.in1d(vid, zonestoadd)]
                            if dont_merge or (use_link_density_threshold and dens < link_density_threshold) or \
                                    (use_r_threshold and max(rsublist) > r_threshold):
                                # cannot add these zones
                                rstopadd = rval
                                add_more = False
                                finalpos -= (num_zones_to_add + 1)
                            else:
                                # keep adding zones
                                zonelist = np.append(zonelist, zonestoadd)
                                num_adds += num_zones_to_add
                                total_vol += np.sum(volsublist)
                                total_num_parts += np.sum(partsublist)
                            pos += num_zones_to_add + 2
                            num_zones_to_add = int(clustline[pos])
                            rval = float(clustline[pos + 1])
                            if add_more:
                                finalpos = pos + num_zones_to_add + 1

                        counted_zones = np.append(counted_zones, zonelist)
                        member_ids = np.logical_and(
                            np.logical_or(use_stripping, densities[:] > strip_density_threshold),
                            np.in1d(zonedata, zonelist))
                        if use_stripping:  # need to recalculate total_vol and total_num_parts after stripping
                            total_vol = np.sum(vols[member_ids])
                            total_num_parts = len(vols[member_ids])

                        if 1 in edgelist[np.in1d(vid, zonestoadd)]:
                            edge_flag = np.append(edge_flag, 1)
                        else:
                            edge_flag = np.append(edge_flag, 0)

                        # average density of member cells weighted by cell volumes
                        w_a_d = np.sum(vols[member_ids] * densities[member_ids]) / np.sum(vols[member_ids])
                        wtd_avg_dens = np.append(wtd_avg_dens, w_a_d)

                        newclustline = clustline[:finalpos]
                        if not add_more:
                            newclustline.append(str(0))
                        newclustline.append(str(rstopadd))

                        # write line to the output .void file
                        for j in range(len(newclustline)):
                            Fnewclust.write(newclustline[j] + '\t')
                        Fnewclust.write('\n')

                        if rstopadd > 10 ** 20:
                            rstopadd = -1  # will be true for structures entirely surrounded by edge particles
                        # write line to the output _list.txt file
                        Fnewlist.write('%d\t%d\t%f\t%d\t%d\t%d\t%f\t%f\n' % (vid[i], int(clustersread[i, 2]), coredens,
                                                                             int(clustersread[i, 5]), num_adds + 1,
                                                                             total_num_parts, total_vol * meanvol_trc,
                                                                             rstopadd))

        # tidy up the files
        # insert first line with number of clusters to the new .void file
        with open(new_clust_file, 'r+') as Fnewclust:
            old = Fnewclust.read()
            Fnewclust.seek(0)
            topline = "%d\n" % num_acc
            Fnewclust.write(topline + old)

        # insert header to the output _list.txt file
        listdata = np.loadtxt(new_list_file)
        header = "%d non-edge tracers in %s, %d clusters\n" % (self.num_non_edge, self.handle, num_acc)
        header = header + "ClusterID CoreParticle CoreDens Zone#Parts Cluster#Zones Cluster#Parts" + \
                 "ClusterVol(Mpc/h^3) ClusterDensRatio"
        np.savetxt(new_list_file, listdata, fmt='%d %d %0.6f %d %d %d %0.6f %0.6f', header=header)

        # now find the maximum density centre locations of the superclusters
        list_array = np.loadtxt(new_list_file)
        if self.is_box:
            info_output = np.zeros((num_acc, 9))
        else:
            info_output = np.zeros((num_acc, 11))
        eff_rad = np.zeros(num_acc)
        wtd_avg_dens = np.zeros(num_acc)
        centres = np.empty((num_acc, 3))

        with open(new_clust_file, 'r') as FHierarchy:
            FHierarchy.readline()  # skip the first line, contains total number of structures
            for i in range(num_acc):
                # get the member zones of the structure
                structline = (FHierarchy.readline()).split()
                pos = 1
                add_zones = int(structline[pos]) > 0
                member_zones = np.asarray(structline[0], dtype=int)
                while add_zones:
                    num_zones_to_add = int(structline[pos])
                    zonestoadd = np.asarray(structline[pos + 2:pos + num_zones_to_add + 2], dtype=int)
                    member_zones = np.append(member_zones, zonestoadd)
                    pos += num_zones_to_add + 2
                    add_zones = int(structline[pos]) > 0

                # get the member particles for these zones
                if use_stripping:
                    member_ids = np.logical_and(densities[:] > strip_density_threshold, np.in1d(zonedata, member_zones))
                else:  # stripDens functionality disabled
                    member_ids = np.in1d(zonedata, member_zones)
                member_vol = vols[member_ids]
                member_dens = densities[member_ids]

                # centre location is position of max. density member particle
                core_part_id = int(list_array[i, 1])
                centres[i, :] = positions[core_part_id]

                # total volume of structure in Mpc/h, and effective radius
                cluster_vol = np.sum(member_vol) * meanvol_trc
                eff_rad[i] = (3.0 * cluster_vol / (4 * np.pi)) ** (1.0 / 3)

                # average density of member cells weighted by cell volumes
                wtd_avg_dens[i] = np.sum(member_dens * member_vol) / np.sum(member_vol)

            info_output[:, 0] = list_array[:, 0]
            info_output[:, 4] = eff_rad
            info_output[:, 5] = list_array[:, 2] - 1.
            info_output[:, 6] = wtd_avg_dens - 1.
            info_output[:, 7] = (wtd_avg_dens - 1.) * eff_rad ** 1.6
            info_output[:, 8] = list_array[:, 7]
            if self.is_box:
                info_output[:, 1:4] = centres[:, :3]
            else:
                centre_obs = self.zobovbox2obs(centres)  # centres - self.box_length / 2.0
                rdist = np.linalg.norm(centre_obs, axis=1)
                eff_angrad = np.degrees(eff_rad / rdist)
                centre_redshifts = self.cosmo.get_redshift(rdist)
                centre_dec = 90 - np.degrees((np.arccos(centre_obs[:, 2] / rdist)))
                centre_ra = np.degrees(np.arctan2(centre_obs[:, 1], centre_obs[:, 0]))
                centre_ra[centre_ra < 0] += 360

                info_output[:, 1] = centre_ra
                info_output[:, 2] = centre_dec
                info_output[:, 3] = centre_redshifts
                info_output[:, 9] = eff_angrad
                info_output[:, 10] = edge_flag

        # save output data to file
        header = "%d superclusters from %s\n" % (num_acc, self.handle)
        if self.is_box:
            header = header + 'ClusterID XYZ[3](Mpc/h) R_eff(Mpc/h) delta_max delta_avg lambda_c DensRatio'
            np.savetxt(info_file, info_output, fmt='%d %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %d %d',
                       header=header)
        else:
            header = header + 'ClusterID RA(deg) Dec(deg) redshift R_eff(Mpc/h) delta_max delta_avg lambda_c ' + \
                     'DensRatio Theta_eff(deg) EdgeFlag'
            np.savetxt(info_file, info_output, fmt='%d %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %d',
                       header=header)
