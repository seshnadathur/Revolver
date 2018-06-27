from __future__ import print_function
import sys
import os
import numpy as np
import healpy as hp
import random
import imp
from scipy.spatial import cKDTree
from scipy.integrate import quad
from astropy.io import fits

class Cosmology:

    def __init__(self, omega_m=0.308, h=0.676):
        print('Initializing cosmology with omega_m = %.2f' % omega_m)
        c = 299792.458
        omega_l = 1.-omega_m
        ztab = np.linspace(0, 4, 10000)
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


class VoidSample:

    def __init__(self, run_zobov=True, tracer_file="", handle="", output_folder="", is_box=True, boss_like=False, 
                 special_patchy=False, posn_cols=np.array([0, 1, 2]), box_length=2500.0, omega_m=0.308, mask_file="",
                 use_z_wts=True, use_ang_wts=True, z_min=0.43, z_max=0.7, mock_file="", mock_dens_ratio=10,
                 min_dens_cut=1.0, void_min_num=1, use_barycentres=True, void_prefix="", find_clusters=False,
                 max_dens_cut=1.0, cluster_min_num=1, cluster_prefix=""):

        # the prefix/handle used for all output file names
        self.handle = handle

        # output folder
        self.output_folder = output_folder

        # file path for ZOBOV-formatted tracer data
        self.posn_file = output_folder + handle + "_pos.dat"

        # input file
        self.tracer_file = tracer_file
        # check input file exists ...
        if not os.access(tracer_file, os.F_OK):
            sys.exit("Can't find tracer file %s, aborting" % tracer_file)

        # (Boolean) choice between cubic simulation box and sky survey
        self.is_box = is_box

        # load tracer information
        print("Loading tracer positions from file %s" % tracer_file)
        if boss_like:  # FITS format input file
            if self.is_box:
                print('Both boss_like and is_box cannot be simultaneously True! Setting is_box = False')
                self.is_box = False
            input_data = fits.open(self.tracer_file)[1].data
            ra = input_data.RA
            dec = input_data.DEC
            redshift = input_data.Z
            self.num_tracers = len(ra)
            tracers = np.empty((self.num_tracers, 3))
            tracers[:, 0] = ra
            tracers[:, 1] = dec
            tracers[:, 2] = redshift
        else:
            if '.npy' in tracer_file:
                tracers = np.load(tracer_file)
            else:
                tracers = np.loadtxt(tracer_file)
            # test that tracer information is valid
            if not tracers.shape[1] >= 3:
                sys.exit("Not enough columns, need 3D position information. Aborting")
            if not len(posn_cols) == 3:
                sys.exit("You must specify 3 columns containing tracer position information. Aborting")

            if special_patchy:
                # if the special PATCHY mock format is used and the vetomask has not already been applied
                # during reconstruction, apply it now
                veto_cut = tracers[:, 6] == 1
                tracers = tracers[veto_cut, :]
            self.num_tracers = tracers.shape[0]
            print("%d tracers found" % self.num_tracers)

            # keep only the tracer position information
            tracers = tracers[:, posn_cols]

        # now complete the rest of the data preparation
        if self.is_box:  # dealing with cubic simulation box
            if box_length <= 0:
                sys.exit("Zero or negative box length, aborting")
            self.box_length = box_length

            # check that tracer positions lie within the box, wrap using PBC if not
            tracers[tracers[:, 0] > box_length, 0] -= box_length
            tracers[tracers[:, 1] > box_length, 1] -= box_length
            tracers[tracers[:, 2] > box_length, 2] -= box_length
            tracers[tracers[:, 0] < 0, 0] += box_length
            tracers[tracers[:, 1] < 0, 1] += box_length
            tracers[tracers[:, 2] < 0, 2] += box_length

            # determine mean tracer number density
            self.tracer_dens = 1.0*self.num_tracers/(box_length**3.)

            self.num_mocks = 0
            self.num_part_total = self.num_tracers
            self.tracers = tracers
        else:
            # set cosmology
            self.omega_m = omega_m
            cosmo = Cosmology(omega_m=omega_m)
            self.cosmo = cosmo

            # convert input tracer information to standard format
            self.coords_radecz2std(tracers[:, 0], tracers[:, 1], tracers[:, 2])

            self.z_min = z_min
            self.z_max = z_max

            # check and cut on the provided redshift limits
            if np.min(self.tracers[:, 5]) < self.z_min or np.max(self.tracers[:, 5]) > self.z_max:
                print('Cutting galaxies outside the redshift limits')
                zselect = (self.z_min < self.tracers[:, 5])&(self.tracers[:, 5] < self.z_max)
                self.tracers = self.tracers[zselect, :]

            # sky mask file (should be in Healpy FITS format)
            if not os.access(mask_file, os.F_OK):
                print("Sky mask not provided or not found, generating approximate one")
                self.mask_file = self.output_folder + self.handle + '_mask.fits'
                self.f_sky = self.generate_mask()
            else:
                mask = hp.read_map(mask_file, verbose=False)
                self.mask_file = mask_file
                # check whether the mask is correct
                ra = self.tracers[:, 3]
                dec = self.tracers[:, 4]
                nside = hp.get_nside(mask)
                pixels = hp.ang2pix(nside, np.deg2rad(90-dec), np.deg2rad(ra))
                if np.any(mask[pixels] == 0):
                    print('Galaxies exist where mask=0. Removing these to avoid errors later.')
                    all_indices = np.arange(len(self.tracers))
                    bad_inds = np.where(mask[pixels] == 0)[0]
                    good_inds = all_indices[np.logical_not(np.in1d(all_indices, bad_inds))]
                    self.tracers = self.tracers[good_inds, :]

                # effective sky fraction
                self.f_sky = 1.0*np.sum(mask)/len(mask)

            # finally, remove any instances of two galaxies at the same location otherwise tessellation will fail
            # (this is a problem with PATCHY mocks, not seen any such instances in real survey data ...)
            unique_tracers = np.unique(self.tracers, axis=0)
            if unique_tracers.shape[0] < self.tracers.shape[0]:
                print('Removing %d galaxies with duplicate positions'
                      % ((self.tracers.shape[0]-unique_tracers.shape[0])/2))
            self.tracers = unique_tracers

            # update galaxy stats
            self.num_tracers = self.tracers.shape[0]
            print('Kept %d tracers after all cuts' % self.num_tracers)

            # calculate mean density
            self.r_near = self.cosmo.get_comoving_distance(self.z_min)
            self.r_far = self.cosmo.get_comoving_distance(self.z_max)
            survey_volume = self.f_sky*4*np.pi*(self.r_far**3. - self.r_near**3.)/3.
            self.tracer_dens = self.num_tracers/survey_volume

            # weights options: correct for z-dependent selection and angular completeness
            self.use_z_wts = use_z_wts
            if use_z_wts:
                self.selection_fn_file = self.output_folder + self.handle + '_selFn.txt'
                self.generate_selfn(nbins=15)
            self.use_ang_wts = use_ang_wts

            if run_zobov:
                # options for buffer mocks around survey boundaries
                if mock_file == '':
                    # no buffer mocks provided, so generate new
                    print('Generating buffer mocks around survey edges ...')
                    print('\tbuffer mocks will have %0.1f x the galaxy number density' % mock_dens_ratio)
                    self.mock_dens_ratio = mock_dens_ratio
                    self.generate_buffer()
                elif not os.access(mock_file, os.F_OK):
                    print('Could not find file %s containing buffer mocks!' % mock_file)
                    print('Generating buffer mocks around survey edges ...')
                    print('\tbuffer mocks will have %0.1f x the galaxy number density' % mock_dens_ratio)
                    self.mock_dens_ratio = mock_dens_ratio
                    self.generate_buffer()
                else:
                    print('Loading pre-computed buffer mocks from file %s' % mock_file)
                    if '.npy' in mock_file:
                        buffers = np.load(mock_file)
                    else:
                        buffers = np.loadtxt(mock_file)
                    # recalculate the box length
                    self.box_length = 2.0 * np.max(np.abs(buffers[:, :3])) + 1
                    self.num_mocks = buffers.shape[0]
                    # join the buffers to the galaxy tracers
                    self.tracers = np.vstack([self.tracers, buffers])
                    self.num_part_total = self.num_tracers + self.num_mocks
                    self.mock_file = mock_file
                # shift Cartesian positions from observer to box coordinates
                self.tracers[:, :3] += 0.5 * self.box_length

        # for easy debugging: write all tracer positions to file
        # np.save(self.posn_file.replace('pos.dat', 'pos.npy'), self.tracers)

        self.num_non_edge = self.num_tracers

        # options for void-finding
        self.min_dens_cut = min_dens_cut
        self.void_min_num = void_min_num
        self.use_barycentres = use_barycentres

        # prefix for naming void files
        self.void_prefix = void_prefix

        # options for finding 'superclusters'
        self.find_clusters = find_clusters
        if find_clusters:
            self.cluster_min_num = cluster_min_num
            self.max_dens_cut = max_dens_cut
            self.cluster_prefix = cluster_prefix

    def coords_radecz2std(self, ra, dec, redshift):
        """Converts sky coordinates in (RA,Dec,redshift) to standard form, including comoving
        Cartesian coordinate information
        """

        # convert galaxy redshifts to comoving distances
        rdist = self.cosmo.get_comoving_distance(redshift)

        # convert RA, Dec angles in degrees to theta, phi in radians
        phi = ra*np.pi/180.
        theta = np.pi/2. - dec*np.pi/180.

        # obtain Cartesian coordinates
        galaxies = np.zeros((self.num_tracers, 6))
        galaxies[:, 0] = rdist*np.sin(theta)*np.cos(phi)  # r*cos(ra)*cos(dec)
        galaxies[:, 1] = rdist*np.sin(theta)*np.sin(phi)  # r*sin(ra)*cos(dec)
        galaxies[:, 2] = rdist*np.cos(theta)  # r*sin(dec)
        # standard format includes RA, Dec, redshift info
        galaxies[:, 3] = ra
        galaxies[:, 4] = dec
        galaxies[:, 5] = redshift

        self.tracers = galaxies

    def generate_mask(self):
        """Generates an approximate survey sky mask if none is provided, and saves to file

        Returns: f_sky
        """

        nside = 64
        npix = hp.nside2npix(nside)

        # use tracer RA,Dec info to see which sky pixels are occupied
        phi = self.tracers[:, 3] * np.pi / 180.
        theta = np.pi / 2. - self.tracers[:, 4] * np.pi / 180.
        pixels = hp.ang2pix(nside, theta, phi)

        # very crude binary mask
        mask = np.zeros(npix)
        mask[pixels] = 1.

        # write this mask to file
        hp.write_map(self.mask_file, mask)

        # return sky fraction
        f_sky = 1.0 * sum(mask) / len(mask)
        return f_sky

    def find_mask_boundary(self, completeness_limit=0.):
        """Finds pixels adjacent to but outside the survey mask

        Arguments:
            completeness_limit: value in range (0,1), sets completeness lower limit for boundary determination

        Returns:
            boundary: a binary Healpix map with the survey mask boundary"""

        mask = hp.read_map(self.mask_file, verbose=False)
        mask = hp.ud_grade(mask, 512)
        nside = hp.get_nside(mask)
        npix = hp.nside2npix(nside)
        boundary = np.zeros(npix)

        # find pixels outside the mask that neighbour pixels within it
        # do this step in a loop, to get a thicker boundary layer
        for j in range(2 + nside/128):
            if j == 0:
                filled_inds = np.nonzero(mask > completeness_limit)[0]
            else:
                filled_inds = np.nonzero(boundary)[0]
            theta, phi = hp.pix2ang(nside, filled_inds)
            neigh_pix = hp.get_all_neighbours(nside, theta, phi)
            for i in range(neigh_pix.shape[1]):
                outsiders = neigh_pix[(mask[neigh_pix[:, i]] <= completeness_limit) & (neigh_pix[:, i] > -1)
                                      & (boundary[neigh_pix[:, i]] == 0), i]
                # >-1 condition takes care of special case where neighbour wasn't found
                if j == 0:
                    boundary[outsiders] = 2
                else:
                    boundary[outsiders] = 1
        boundary[boundary == 2] = 0

        if nside <= 128:
            # upgrade the boundary to aid placement of buffer mocks
            boundary = hp.ud_grade(boundary, 128)

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
        r_low = self.cosmo.get_comoving_distance(z_high) + mean_nn_distance * self.mock_dens_ratio**(-1./3)
        r_high = r_low + mean_nn_distance
        cap_volume = self.f_sky*4.*np.pi*(r_high**3. - r_low**3.)/3.

        # how many buffer particles fit in this cap
        num_high_mocks = int(np.ceil(buffer_dens * cap_volume))
        high_mocks = np.zeros((num_high_mocks, 6))

        # generate random radial positions within the cap
        rdist = (r_low**3. + (r_high**3. - r_low**3.)*np.random.rand(num_high_mocks))**(1./3)

        # generate random angular positions within the survey mask
        while num_high_mocks > numpix:
            # more mock posns required than mask pixels, so upgrade mask to get more pixels
            nside *= 2
            mask = hp.ud_grade(mask, nside)
            survey_pix = np.nonzero(mask)[0]
            numpix = len(survey_pix)
        rand_pix = survey_pix[random.sample(np.arange(numpix), num_high_mocks)]
        theta, phi = hp.pix2ang(nside, rand_pix)

        # convert to standard format
        high_mocks[:, 0] = rdist * np.sin(theta) * np.cos(phi)
        high_mocks[:, 1] = rdist * np.sin(theta) * np.sin(phi)
        high_mocks[:, 2] = rdist * np.cos(theta)
        high_mocks[:, 3] = phi * 180. / np.pi
        high_mocks[:, 4] = 90 - theta * 180. / np.pi
        high_mocks[:, 5] = -1    # all buffer particles are given redshift -1 to aid identification

        # farthest buffer particle
        self.r_far = np.max(rdist)

        print("\tplaced %d buffer mocks at high-redshift cap" % num_high_mocks)

        buffers = high_mocks
        self.num_mocks = num_high_mocks
        # ------------------------------------------------------------- #

        # ----- Step 2: buffer particles along the low-redshift cap---- #
        z_low = np.min([np.min(self.tracers[:, 5]), self.z_min])
        if z_low > 0:
            # define the radial extents of the layer in which we will place the buffer particles
            # these choices are somewhat arbitrary, and could be optimized
            r_high = self.cosmo.get_comoving_distance(z_low) - mean_nn_distance * self.mock_dens_ratio**(-1./3)
            r_low = r_high - mean_nn_distance
            if r_high < 0:
                r_high = self.cosmo.get_comoving_distance(z_low)
            if r_low < 0:
                r_low = 0
            cap_volume = self.f_sky * 4. * np.pi * (r_high ** 3. - r_low ** 3.) / 3.

            # how many buffer particles fit in this cap
            num_low_mocks = int(np.ceil(buffer_dens * cap_volume))
            low_mocks = np.zeros((num_low_mocks, 6))

            # generate random radial positions within the cap
            rdist = (r_low ** 3. + (r_high ** 3. - r_low ** 3.) * np.random.rand(num_low_mocks)) ** (1. / 3)

            # generate random angular positions within the survey mask
            while num_low_mocks > numpix:
                # more mock posns required than mask pixels, so upgrade mask to get more pixels
                nside *= 2
                mask = hp.ud_grade(mask, nside)
                survey_pix = np.nonzero(mask)[0]
                numpix = len(survey_pix)
            rand_pix = survey_pix[random.sample(np.arange(numpix), num_low_mocks)]
            theta, phi = hp.pix2ang(nside, rand_pix)

            # convert to standard format
            low_mocks[:, 0] = rdist * np.sin(theta) * np.cos(phi)
            low_mocks[:, 1] = rdist * np.sin(theta) * np.sin(phi)
            low_mocks[:, 2] = rdist * np.cos(theta)
            low_mocks[:, 3] = phi * 180. / np.pi
            low_mocks[:, 4] = 90 - theta * 180. / np.pi
            low_mocks[:, 5] = -1.  # all buffer particles are given redshift -1 to aid later identification

            # closest buffer particle
            self.r_near = np.min(rdist)

            print("\tplaced %d buffer mocks at low-redshift cap" % num_low_mocks)

            buffers = np.vstack([buffers, low_mocks])
            self.num_mocks += num_low_mocks
        else:
            print("\tno buffer mocks required at low-redshift cap")
        # ------------------------------------------------------------- #

        # ------ Step 3: buffer particles along the survey edges-------- #
        if self.f_sky < 1.0:
            # get the survey boundary
            boundary = self.find_mask_boundary(completeness_limit=0.0)

            # where we will place the buffer mocks
            boundary_pix = np.nonzero(boundary)[0]
            numpix = len(boundary_pix)
            boundary_f_sky = 1.0 * len(boundary_pix) / len(boundary)
            boundary_nside = hp.get_nside(boundary)

            # how many buffer particles
            # boundary_volume = boundary_f_sky * 4. * np.pi * (self.r_far ** 3. - self.r_near ** 3.) / 3.
            boundary_volume = boundary_f_sky * 4. * np.pi * quad(lambda y: y**2, self.r_near, self.r_far)[0]
            num_bound_mocks = int(np.ceil(buffer_dens * boundary_volume))
            bound_mocks = np.zeros((num_bound_mocks, 6))

            # generate random radial positions within the boundary layer
            rdist = (self.r_near ** 3. + (self.r_far ** 3. - self.r_near ** 3.) *
                     np.random.rand(num_bound_mocks)) ** (1. / 3)

            # generate random angular positions within the boundary layer
            while num_bound_mocks > numpix:
                # more mocks required than pixels in which to place them, so upgrade mask
                boundary_nside *= 2
                boundary = hp.ud_grade(boundary, boundary_nside)
                boundary_pix = np.nonzero(boundary)[0]
                numpix = len(boundary_pix)
            rand_pix = boundary_pix[random.sample(np.arange(numpix), num_bound_mocks)]
            theta, phi = hp.pix2ang(boundary_nside, rand_pix)

            # convert to standard format
            bound_mocks[:, 0] = rdist * np.sin(theta) * np.cos(phi)
            bound_mocks[:, 1] = rdist * np.sin(theta) * np.sin(phi)
            bound_mocks[:, 2] = rdist * np.cos(theta)
            bound_mocks[:, 3] = phi * 180. / np.pi
            bound_mocks[:, 4] = 90 - theta * 180. / np.pi
            bound_mocks[:, 5] = -1.  # all buffer particles are given redshift -1 to aid identification

            print("\tplaced %d buffer mocks along the survey boundary edges" % num_bound_mocks)

            buffers = np.vstack([buffers, bound_mocks])
            self.num_mocks += num_bound_mocks
        else:
            print("\tdata covers the full sky, no buffer mocks required along edges")
        # ------------------------------------------------------------- #

        # determine the size of the cubic box required
        self.box_length = 2.0 * np.max(np.abs(buffers[:, :3])) + 1.
        print("\tUsing box length %0.2f" % self.box_length)

        # ------ Step 4: guard buffers to stabilize the tessellation-------- #
        # (strictly speaking, this gives a lot of redundancy as the box is very big;
        # but it doesn't slow the tessellation too much and keeps coding simpler)

        # generate guard particle positions
        x = np.linspace(0.1, self.box_length - 0.1, 20)
        guards = np.vstack(np.meshgrid(x, x, x)).reshape(3, -1).T

        # make a kdTree instance using all the galaxies and buffer mocks
        all_positions = np.vstack([self.tracers[:, :3], buffers[:, :3]])
        all_positions += self.box_length/2.  # from observer to box coordinates
        tree = cKDTree(all_positions, boxsize=self.box_length)

        # find the nearest neighbour distance for each of the guard particles
        nn_dist = np.empty(len(guards))
        for i in range(len(guards)):
            nn_dist[i], nnind = tree.query(guards[i, :], k=1)

        # drop all guards that are too close to existing points
        guards = guards[nn_dist > (self.box_length - 0.2)/20.]
        guards = guards - self.box_length/2.    # guard positions back in observer coordinates

        # convert to standard format
        num_guard_mocks = len(guards)
        guard_mocks = np.zeros((num_guard_mocks, 6))
        guard_mocks[:, :3] = guards
        guard_mocks[:, 3:5] = -60.     # guards are given RA and Dec -60 as well to distinguish them from other buffers
        guard_mocks[:, 5] = -1.

        print("\tadded %d guards to stabilize the tessellation" % num_guard_mocks)

        buffers = np.vstack([buffers, guard_mocks])
        self.num_mocks += num_guard_mocks
        # ------------------------------------------------------------------ #

        # write the buffer information to file for later reference
        mock_file = self.posn_file.replace('pos.dat', 'mocks.npy')
        print('Buffer mocks written to file %s' % mock_file)
        np.save(mock_file, buffers)
        self.mock_file = mock_file

        # now add buffer particles to tracers
        self.tracers = np.vstack([self.tracers, buffers])

        self.num_part_total = self.num_tracers + self.num_mocks

    def generate_selfn(self, nbins=20):
        """Measures the redshift-dependence of the galaxy number density in equal-volume redshift bins,
         and writes the selection function to file.

        Arguments:
          nbins: number of bins to use
        """

        print('Determining survey redshift selection function ...')

        # first determine the equal volume bins
        r_near = self.cosmo.get_comoving_distance(self.z_min)
        r_far = self.cosmo.get_comoving_distance(self.z_max)
        rvals = np.linspace(r_near**3, r_far**3, nbins+1)
        rvals = rvals**(1./3)
        zsteps = self.cosmo.get_redshift(rvals)
        volumes = self.f_sky*4*np.pi*(rvals[1:]**3. - rvals[:-1]**3.)/3.
        # (all elements of volumes should be equal)

        # get the tracer galaxy redshifts
        redshifts = self.tracers[:, 5]

        # histogram and calculate number density
        hist, zsteps = np.histogram(redshifts, bins=zsteps)
        nofz = hist/volumes
        zmeans = np.zeros(len(hist))
        for i in range(len(hist)):
            zmeans[i] = np.mean(redshifts[np.logical_and(redshifts >= zsteps[i], redshifts < zsteps[i+1])])

        output = np.zeros((len(zmeans), 3))
        output[:, 0] = zmeans
        output[:, 1] = nofz
        output[:, 2] = nofz/self.tracer_dens

        # write to file
        np.savetxt(self.selection_fn_file, output, fmt='%0.3f %0.4e %0.4f', header='z n(z) f(z)')

    def write_box_zobov(self):
        """Writes the tracer and mock position information to file in a ZOBOV-readable format"""

        with open(self.posn_file, 'w') as F:
            npart = np.array(self.num_part_total, dtype=np.int32)
            npart.tofile(F, format='%d')
            data = self.tracers[:, 0]
            data.tofile(F, format='%f')
            data = self.tracers[:, 1]
            data.tofile(F, format='%f')
            data = self.tracers[:, 2]
            data.tofile(F, format='%f')
            if not self.is_box:  # write RA, Dec and redshift too
                data = self.tracers[:, 3]
                data.tofile(F, format='%f')
                data = self.tracers[:, 4]
                data.tofile(F, format='%f')
                data = self.tracers[:, 5]
                data.tofile(F, format='%f')

    def delete_tracer_info(self):
        """removes the tracer information if no longer required, to save memory"""

        self.tracers = 0

    def reread_tracer_info(self):
        """re-reads tracer information from file if required after previous deletion"""

        self.tracers = np.empty((self.num_part_total, 6))
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

    def write_config(self):
        """method to write configuration information for the ZOBOV run to file for later lookup"""

        info = 'handle = \'%s\'\nis_box = %s\nnum_tracers = %d\n' % (self.handle, self.is_box, self.num_tracers)
        info += 'num_mocks = %d\nnum_non_edge = %d\nbox_length = %f\n' % (self.num_mocks, self.num_non_edge,
                                                                          self.box_length)
        info += 'tracer_dens = %e' % self.tracer_dens
        info_file = self.output_folder + 'sample_info.txt'
        with open(info_file, 'w') as F:
            F.write(info)

    def read_config(self):
        """method to read configuration file for information about previous ZOBOV run"""

        info_file = self.output_folder + 'sample_info.txt'
        parms = imp.load_source('name', info_file)
        self.num_mocks = parms.num_mocks
        self.num_non_edge = parms.num_non_edge
        self.box_length = parms.box_length
        self.tracer_dens = parms.tracer_dens


class GalaxyCatalogue:

    def __init__(self, catalogue_file, is_box=False, box_length=1000., randoms=False, boss_like=True,
                 special_patchy=False, posn_cols=np.array([0, 1, 2]), fkp=1, noz=1, cp=1, systot=1, veto=1):

        if randoms:
            print('=== Loading randoms data from file ====')
        else:
            print('=== Loading galaxy data from file ====')

        if boss_like:
            a = fits.open(catalogue_file)[1].data
            for f in a.names:
                self.__dict__[f.lower()] = a.field(f)
            # NOTE: this takes the weights, in particular the FKP weights, directly from file.
            # If a different FKP weighting is desired (e.g., to work better on smaller scales)
            # this would need to be recalculated!

            self.size = self.ra.size
            self.names = a.names
            self.veto = np.ones(self.size)  # all vetoes have already been applied!

            # change the name 'z'-->'redshift' to avoid confusion
            self.redshift = self.z

            # initialize Cartesian positions and observer distance
            self.x = np.zeros_like(self.ra)
            self.y = np.zeros_like(self.ra)
            self.z = np.zeros_like(self.ra)
            self.newx = np.zeros_like(self.ra)
            self.newy = np.zeros_like(self.ra)
            self.newz = np.zeros_like(self.ra)
            self.dist = np.zeros_like(self.ra)

        else:
            # ASCII and NPY formatted input files are supported
            if '.npy' in catalogue_file:
                data = np.load(catalogue_file)
            else:
                data = np.loadtxt(catalogue_file)

            if is_box:
                # for uniform box, galaxy weights, vetos, ra, dec, redshift, distance information are not used!
                self.box_length = box_length

                # for uniform box, position data is in Cartesian format
                self.x = data[:, posn_cols[0]]
                self.y = data[:, posn_cols[1]]
                self.z = data[:, posn_cols[2]]
                self.newx = 1.0 * self.x
                self.newy = 1.0 * self.y
                self.newz = 1.0 * self.z
                self.size = self.x.size
            else:
                # position information is ra, dec and redshift
                self.ra = data[:, posn_cols[0]]
                self.dec = data[:, posn_cols[1]]
                self.redshift = data[:, posn_cols[2]]
                self.size = self.ra.size

                # Cartesian positions and observer distance initialized to zero, calculated later
                self.x = np.zeros(self.size)
                self.y = np.zeros(self.size)
                self.z = np.zeros(self.size)
                self.newx = np.zeros(self.size)
                self.newy = np.zeros(self.size)
                self.newz = np.zeros(self.size)
                self.dist = np.zeros(self.size)

                # by default, set all weights and veto to 1
                self.weight_fkp = np.ones(self.size)
                self.weight_cp = np.ones(self.size)
                self.weight_noz = np.ones(self.size)
                self.weight_syst = np.ones(self.size)
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
                        self.weight_syst = data[:, posn_cols[2] + count]
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
            weights *= self.weight_syst

        return weights
