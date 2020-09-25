from __future__ import print_function
import os
import sys
import numpy as np
import subprocess
import python_tools.fastmodules as fastmodules
from scipy.ndimage.filters import gaussian_filter
from python_tools.cosmology import Cosmology


class VoxelVoids:

    def __init__(self, cat, ran, parms):

        print("\n ==== Starting the void-finding with voxel-based method ==== ")
        sys.stdout.flush()

        self.is_box = parms.is_box
        self.handle = parms.handle
        self.output_folder = parms.output_folder
        if not os.access(self.output_folder, os.F_OK):
            os.makedirs(self.output_folder)
        self.min_dens_cut = parms.min_dens_cut
        self.use_barycentres = parms.use_barycentres
        self.void_prefix = 'voxel-' + parms.void_prefix
        self.find_clusters = parms.find_clusters
        self.max_dens_cut = parms.max_dens_cut
        self.cluster_prefix = 'voxel-' + parms.cluster_prefix
        self.rhog = np.array(0.)  # this gets changed later
        self.mask_cut = []
        self.z_min = parms.z_min
        self.z_max = parms.z_max
        self.verbose = parms.verbose

        print("%d tracers found" % cat.size)

        if self.is_box:
            self.box_length = parms.box_length
            self.cat = cat

            # determine an appropriate bin size
            mean_dens = cat.size / parms.box_length ** 3.
            self.nbins = int(np.floor(parms.box_length / (0.5 * (4 * np.pi * mean_dens / 3.) ** (-1. / 3))))
            self.binsize = parms.box_length / self.nbins
            print('Bin size [Mpc/h]: %0.2f, nbins = %d' % (self.binsize, self.nbins))

            # choose an appropriate smoothing scale
            self.smooth = mean_dens ** (-1. / 3)
            print('Smoothing scale [Mpc/h]:', self.smooth)
            sys.stdout.flush()

            self.xmin = 0
            self.ymin = 0
            self.zmin = 0

        else:
            cosmo = Cosmology(omega_m=parms.omega_m)
            # get the weights for data and randoms
            cat.weight = cat.get_weights(fkp=False, syst_wts=True)
            if cat.weights_model == 2 or cat.weights_model == 3:
                # for eBOSS or joint BOSS+eBOSS catalogues, systematic weights are included for randoms
                ran.weight = ran.get_weights(fkp=False, syst_wts=True)
            # for BOSS catalogues, systematic weights are NOT included for randoms
            ran.weight = ran.get_weights(fkp=False, syst_wts=False)

            # relative weighting of galaxies and randoms
            sum_wgal = np.sum(cat.weight)
            sum_wran = np.sum(ran.weight)
            alpha = sum_wgal / sum_wran
            self.alpha = alpha
            self.deltar = 0

            self.cosmo = cosmo
            self.ran = ran
            self.cat = cat

            # put the data into a box
            mean_dens = self.make_sky_box()

            # set a cutoff for defining cells as empty if they have < 10% the mean number of randoms
            ran_min = (0.1 * mean_dens * self.binsize**3.) / self.alpha
            self.ran_min = ran_min

    def make_sky_box(self, padding=5.):

        maxx = max(np.max(self.ran.x), np.max(self.cat.x))
        minx = min(np.min(self.ran.x), np.min(self.cat.x))
        maxy = max(np.max(self.ran.y), np.max(self.cat.y))
        miny = min(np.min(self.ran.y), np.min(self.cat.y))
        maxz = max(np.max(self.ran.z), np.max(self.cat.z))
        minz = min(np.min(self.ran.z), np.min(self.cat.z))
        dx = maxx - minx
        dy = maxy - miny
        dz = maxz - minz
        x0 = 0.5 * (maxx + minx)
        y0 = 0.5 * (maxy + miny)
        z0 = 0.5 * (maxz + minz)

        redo_padding = True
        while redo_padding:
            box = max([dx, dy, dz]) + 2 * padding  # a bit bigger than strictly necessary
            xmin = x0 - box / 2
            ymin = y0 - box / 2
            zmin = z0 - box / 2

            self.xmin = xmin
            self.ymin = ymin
            self.zmin = zmin
            self.box_length = box

            # this is clearly a major underestimate
            mean_dens = np.sum(self.cat.weight) / self.box_length**3.

            # starting estimate for bin size
            self.nbins = int(np.floor(box / (0.5 * (4 * np.pi * mean_dens / 3.) ** (-1. / 3))))
            self.binsize = self.box_length / self.nbins

            # check that the zero-padding is greater than 1 bin size
            if padding < self.binsize:
                padding *= 2
            else:
                redo_padding = False

        print('Box size [Mpc/h]: %0.3f' % self.box_length)
        if self.verbose:
            print('Initial bin size [Mpc/h]: %0.2f, nbins = %d' % (self.binsize, self.nbins))

        # now approximately check true survey volume
        ran = self.ran
        rhor = np.zeros((self.nbins, self.nbins, self.nbins), dtype='float64')
        fastmodules.allocate_gal_cic(rhor, ran.x, ran.y, ran.z, ran.weight, ran.size, self.xmin, self.ymin,
                                     self.zmin, self.box_length, self.nbins, 1.)
        # re-estimate the mean density: this will still be a slight underestimate
        filled_cells = np.sum(rhor.flatten() > 0)
        mean_dens = np.sum(self.cat.weight) / (filled_cells * self.binsize**3.)
        # thus get better choice of bin size (this is sufficient for current purposes)
        self.nbins = int(np.floor(box / (0.5 * (4 * np.pi * mean_dens / 3.) ** (-1. / 3))))
        self.binsize = self.box_length / self.nbins
        print('Final bin size [Mpc/h]: %0.2f, nbins = %d' % (self.binsize, self.nbins))

        # choose an appropriate smoothing scale
        self.smooth = mean_dens ** (-1./3)
        print('Smoothing scale [Mpc/h]: %0.2f' % self.smooth)
        sys.stdout.flush()

        return mean_dens

    def run_voidfinder(self):

        # create the folder in which to store various raw outputs
        raw_dir = self.output_folder + "rawVoxelInfo/"
        if not os.access(raw_dir, os.F_OK):
            os.makedirs(raw_dir)

        # get the path to where the C executables are stored
        binpath = os.path.dirname(__file__).replace('python_tools', 'bin/')

        if self.is_box:
            # measure the galaxy density field
            if self.verbose:
                print('Allocating galaxies in cells...')
            sys.stdout.flush()
            rhog = np.zeros((self.nbins, self.nbins, self.nbins), dtype='float64')
            fastmodules.allocate_gal_cic(rhog, self.cat.x, self.cat.y, self.cat.z, self.cat.weight, self.cat.size,
                                         self.xmin, self.ymin, self.zmin, self.box_length, self.nbins, 1)

            # smooth with pre-determined smoothing scale
            if self.verbose:
                print('Smoothing galaxy density field ...')
            sys.stdout.flush()
            rhog = gaussian_filter(rhog, self.smooth / self.binsize, mode='wrap')

            # then normalize number counts to get density in units of mean (i.e. 1 + delta)
            fastmodules.normalize_rho_box(rhog, self.cat.size)
            self.rhoflat = rhog.flatten()
            self.mask_cut = np.zeros(self.nbins**3, dtype='int')  # we don't mask any voxels in a box
        else:
            # measure the galaxy density field
            if self.verbose:
                print('Allocating galaxies in cells...')
            sys.stdout.flush()
            rhog = np.zeros((self.nbins, self.nbins, self.nbins), dtype='float64')
            fastmodules.allocate_gal_cic(rhog, self.cat.x, self.cat.y, self.cat.z, self.cat.weight, self.cat.size,
                                         self.xmin, self.ymin, self.zmin, self.box_length, self.nbins, 1)

            if self.verbose:
                print('Allocating randoms in cells...')
            sys.stdout.flush()
            rhor = np.zeros((self.nbins, self.nbins, self.nbins), dtype='float64')
            fastmodules.allocate_gal_cic(rhor, self.ran.x, self.ran.y, self.ran.z, self.ran.weight, self.ran.size,
                                         self.xmin, self.ymin, self.zmin, self.box_length, self.nbins, 1)

            # identify "empty" cells for later cuts on void catalogue
            mask_cut = np.zeros(self.nbins**3, dtype='int')
            fastmodules.survey_mask(mask_cut, rhor, self.ran_min)
            self.mask_cut = mask_cut

            # smooth both galaxy and randoms with pre-determined smoothing scale
            if self.verbose:
                print('Smoothing density fields ...')
            sys.stdout.flush()
            rhog = gaussian_filter(rhog, self.smooth / self.binsize, mode='nearest')
            rhor = gaussian_filter(rhor, self.smooth / self.binsize, mode='nearest')

            rho = np.empty((self.nbins, self.nbins, self.nbins), dtype='float64')
            fastmodules.normalize_rho_survey(rho, rhog, rhor, self.alpha, self.ran_min)
            self.rhoflat = rho.flatten()

        # write this to file for jozov-grid to read
        rhogflat = np.array(self.rhoflat, dtype=np.float32)
        with open(raw_dir + 'density_n%d.dat' % self.nbins, 'w') as F:
            rhogflat.tofile(F, format='%f')

        # now call jozov-grid
        cmd = [binpath + "jozov-grid", "v", raw_dir + "density_n%d.dat" % self.nbins,
               raw_dir + self.handle, str(self.nbins)]
        subprocess.call(cmd)

        # postprocess void data
        self.postprocess_voids()

        # if reqd, find superclusters
        if self.find_clusters:
            print("\n ==== bonus: overdensity-finding with voxel-based method ==== ")
            sys.stdout.flush()
            cmd = [binpath + "jozov-grid", "c", raw_dir + "density_n%d.dat" % self.nbins,
                   raw_dir + self.handle, str(self.nbins)]
            subprocess.call(cmd)
            self.postprocess_clusters()

        print(" ==== Finished with voxel-based method ==== ")
        sys.stdout.flush()

    def postprocess_voids(self):

        print("Post-processing voids")

        raw_dir = self.output_folder + "rawVoxelInfo/"
        rawdata = np.loadtxt(raw_dir + self.handle + ".txt", skiprows=2)
        nvox = self.nbins ** 3
        # masked_vox = np.arange(nvox)[self.mask_cut]

        # load the void hierarchy data to record void leak density ratio, even though this is
        # possibly not useful for anything at all
        voidfile = raw_dir + self.handle + ".void"
        with open(voidfile, 'r') as F:
            hierarchy = F.readlines()
        densratio = np.zeros(len(rawdata))
        for i in range(len(rawdata)):
            densratio[i] = np.fromstring(hierarchy[i + 1], dtype=float, sep=' ')[2]

        # load zone membership data
        zonefile = raw_dir + self.handle + ".zone"
        with open(zonefile, 'r') as F:
            hierarchy = F.readlines()
        hierarchy = np.asarray(hierarchy, dtype=str)

        # remove voids that: a) don't meet minimum density cut, b) are edge voids, or c) lie in a masked voxel
        select = np.zeros(rawdata.shape[0], dtype='int')
        fastmodules.voxelvoid_cuts(select, self.mask_cut, rawdata, self.min_dens_cut)
        select = np.asarray(select, dtype=bool)
        rawdata = rawdata[select]
        densratio = densratio[select]
        hierarchy = hierarchy[select]

        # void minimum density centre locations
        xpos, ypos, zpos = self.voxel_position(rawdata[:, 2])

        if not self.is_box:  # convert void centre coordinates from box Cartesian to sky positions
            xpos += self.xmin
            ypos += self.ymin
            zpos += self.zmin
            dist = np.sqrt(xpos**2 + ypos**2 + zpos**2)
            redshift = self.cosmo.get_redshift(dist)
            ra = np.degrees(np.arctan2(ypos, xpos))
            dec = 90 - np.degrees(np.arccos(zpos / dist))
            ra[ra < 0] += 360
            xpos = ra
            ypos = dec
            zpos = redshift
            # and an additional cut on any voids with min. dens. centre outside specified redshift range
            select_z = np.logical_and(zpos > self.z_min, zpos < self.z_max)
            rawdata = rawdata[select_z]
            densratio = densratio[select_z]
            hierarchy = hierarchy[select_z]
            xpos = xpos[select_z]
            ypos = ypos[select_z]
            zpos = zpos[select_z]

        # void effective radii
        vols = (rawdata[:, 5] * self.binsize ** 3.)
        rads = (3. * vols / (4. * np.pi)) ** (1. / 3)
        # void minimum densities (as delta)
        mindens = rawdata[:, 3] - 1.
        # void average densities and barycentres
        avgdens = np.zeros(len(rawdata))
        barycentres = np.zeros((len(rawdata), 3))
        for i in range(len(rawdata)):
            member_voxels = np.fromstring(hierarchy[i], dtype=int, sep=' ')[1:]
            member_dens = np.zeros(len(member_voxels), dtype='float64')
            fastmodules.get_member_densities(member_dens, member_voxels, self.rhoflat)
            # member_dens = self.rhoflat[member_voxels]
            avgdens[i] = np.mean(member_dens) - 1.
            if self.use_barycentres:
                member_x, member_y, member_z = self.voxel_position(member_voxels)
                barycentres[i, 0] = np.average(member_x, weights=1. / member_dens)
                barycentres[i, 1] = np.average(member_y, weights=1. / member_dens)
                barycentres[i, 2] = np.average(member_z, weights=1. / member_dens)
        if self.use_barycentres and not self.is_box:
            barycentres[:, 0] += self.xmin
            barycentres[:, 1] += self.ymin
            barycentres[:, 2] += self.zmin
            dist = np.linalg.norm(barycentres, axis=1)
            redshift = self.cosmo.get_redshift(dist)
            ra = np.degrees(np.arctan2(barycentres[:, 1], barycentres[:, 0]))
            dec = 90 - np.degrees(np.arccos(barycentres[:, 2] / dist))
            ra[ra < 0] += 360
            barycentres[:, 0] = ra
            barycentres[:, 1] = dec
            barycentres[:, 2] = redshift

        # record void lambda value, even though usefulness of this has only really been shown for ZOBOV voids so far
        void_lambda = avgdens * (rads ** 1.2)

        # create output array
        output = np.zeros((len(rawdata), 9))
        output[:, 0] = rawdata[:, 0]
        output[:, 1] = xpos
        output[:, 2] = ypos
        output[:, 3] = zpos
        output[:, 4] = rads
        output[:, 5] = mindens
        output[:, 6] = avgdens
        output[:, 7] = void_lambda
        output[:, 8] = densratio

        print('Total %d voids pass all cuts' % len(output))
        sys.stdout.flush()

        # sort in increasing order of minimum density
        sort_order = np.argsort(output[:, 5])
        output = output[sort_order]
        if self.use_barycentres:
            barycentres = barycentres[sort_order]
        # save to file
        catalogue_file = self.output_folder + self.void_prefix + '_cat.txt'
        header = '%d voxels, %d voids\n' % (nvox, len(output))
        if self.is_box:
            header += 'VoidID XYZ[3](Mpc/h) R_eff(Mpc/h) delta_min delta_avg lambda_v DensRatio'
        else:
            header += 'VoidID RA Dec z R_eff(Mpc/h) delta_min delta_avg lambda_v DensRatio'
        np.savetxt(catalogue_file, output, fmt='%d %0.4f %0.4f %0.4f %0.4f %0.6f %0.6f %0.6f %0.6f', header=header)

        if self.use_barycentres:
            if not os.access(self.output_folder + "barycentres/", os.F_OK):
                os.makedirs(self.output_folder + "barycentres/")
            catalogue_file = self.output_folder + 'barycentres/' + self.void_prefix + '_baryC_cat.txt'
            output[:, 1:4] = barycentres
            np.savetxt(catalogue_file, output, fmt='%d %0.4f %0.4f %0.4f %0.4f %0.6f %0.6f %0.6f %0.6f',
                       header=header)

    def postprocess_clusters(self):

        print("Post-processing clusters")

        raw_dir = self.output_folder + "rawVoxelInfo/"
        rawdata = np.loadtxt(raw_dir + self.handle + "c.txt", skiprows=2)

        # load the void hierarchy data to record void leak density ratio, even though this is
        # possibly not useful for anything at all
        voidfile = raw_dir + self.handle + ".void"
        with open(voidfile, 'r') as F:
            hierarchy = F.readlines()
        densratio = np.zeros(len(rawdata))
        for i in range(len(rawdata)):
            densratio[i] = np.fromstring(hierarchy[i + 1], dtype=float, sep=' ')[2]

        # load zone membership data
        zonefile = raw_dir + self.handle + ".zone"
        with open(zonefile, 'r') as F:
            hierarchy = F.readlines()

        nvox = self.nbins ** 3
        # masked_vox = np.arange(nvox)[self.mask_cut]

        select = np.zeros(rawdata.shape[0], dtype='int')
        fastmodules.voxelcluster_cuts(select, self.mask_cut, rawdata, self.min_dens_cut)
        rawdata = rawdata[select]
        densratio = densratio[select]
        hierarchy = hierarchy[select]

        # cluster effective radii
        vols = (rawdata[:, 5] * self.binsize ** 3.)
        rads = (3. * vols / (4. * np.pi)) ** (1. / 3)
        # cluster maximum density centre locations
        xpos, ypos, zpos = self.voxel_position(rawdata[:, 2])
        # cluster maximum densities (as delta)
        maxdens = rawdata[:, 3] - 1.
        # cluster average densities
        avgdens = np.zeros(len(rawdata))
        for i in range(len(rawdata)):
            member_voxels = np.fromstring(hierarchy[i], dtype=int, sep=' ')[1:]
            member_dens = np.zeros(len(member_voxels), dtype='float64')
            fastmodules.get_member_densities(member_dens, member_voxels, self.rhoflat)
            # member_dens = self.rhoflat[member_voxels]
            avgdens[i] = np.mean(member_dens) - 1.
        # record cluster lambda value, even though usefulness of this has only been shown for ZOBOV clusters so far
        cluster_lambda = avgdens * (rads ** 1.6)

        if not self.is_box:  # convert void centre coordinates from box Cartesian to sky positions
            xpos += self.xmin
            ypos += self.ymin
            zpos += self.zmin
            dist = np.sqrt(xpos**2 + ypos**2 + zpos**2)
            redshift = self.cosmo.get_redshift(dist)
            ra = np.degrees(np.arctan2(ypos, xpos))
            dec = 90 - np.degrees(np.arccos(zpos / dist))
            ra[ra < 0] += 360
            xpos = ra
            ypos = dec
            zpos = redshift

        # create output array
        output = np.zeros((len(rawdata), 9))
        output[:, 0] = rawdata[:, 0]
        output[:, 1] = xpos
        output[:, 2] = ypos
        output[:, 3] = zpos
        output[:, 4] = rads
        output[:, 5] = maxdens
        output[:, 6] = avgdens
        output[:, 7] = cluster_lambda
        output[:, 8] = densratio

        print('Total %d clusters pass all cuts' % len(output))
        sys.stdout.flush()
        # sort in decreasing order of maximum density
        output = output[np.argsort(output[:, 5])[::-1]]
        catalogue_file = self.output_folder + self.cluster_prefix + '_cat.txt'
        header = '%d voxels, %d clusters\n' % (nvox, len(output))
        if self.is_box:
            header += 'ClusterID XYZ[3](Mpc/h) R_eff(Mpc/h) delta_max delta_avg lambda_c DensRatio'
        else:
            header += 'ClusterID RA Dec z R_eff(Mpc/h) delta_max delta_avg lambda_c DensRatio'
        np.savetxt(catalogue_file, output, fmt='%d %0.4f %0.4f %0.4f %0.4f %0.6f %0.6f %0.6f %0.6f', header=header)

    def voxel_position(self, voxel):

        xind = np.array(voxel / (self.nbins ** 2), dtype=int)
        yind = np.array((voxel - xind * self.nbins ** 2) / self.nbins, dtype=int)
        zind = np.array(voxel % self.nbins, dtype=int)
        xpos = xind * self.box_length / self.nbins
        ypos = yind * self.box_length / self.nbins
        zpos = zind * self.box_length / self.nbins

        return xpos, ypos, zpos
