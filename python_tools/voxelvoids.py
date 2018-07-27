from __future__ import print_function
import os
import numpy as np
import subprocess
from scipy.ndimage.filters import gaussian_filter
from cosmology import Cosmology


class VoxelVoids:

    def __init__(self, cat, ran, handle="", output_folder="", is_box=True, box_length=2500.0, omega_m=0.308,
                 min_dens_cut=1.0, use_barycentres=True, void_prefix="", find_clusters=False, max_dens_cut=1.0,
                 cluster_prefix=""):

        print(" ==== Starting the void-finding with voxel-based method ==== ")

        self.is_box = is_box
        self.handle = handle
        self.output_folder = output_folder
        self.min_dens_cut = min_dens_cut
        self.use_barycentres = use_barycentres
        self.void_prefix = void_prefix
        self.find_clusters = find_clusters
        self.max_dens_cut = max_dens_cut
        self.cluster_prefix = cluster_prefix
        self.rhog = np.array(0.)  # this gets changed later

        print("%d tracers found" % cat.size)

        if self.is_box:

            self.box_length = box_length
            self.cat = cat

            # determine an appropriate bin size
            mean_dens = cat.size / box_length ** 3.
            self.nbins = int(np.floor(box_length / (0.25 * (4 * np.pi * mean_dens / 3.) ** (-1. / 3))))
            self.binsize = box_length / self.nbins

            # choose an appropriate smoothing scale
            self.smooth = mean_dens ** (-1. / 3)

            print('Bin size [Mpc/h]: %0.2f, nbins = %d' % (self.binsize, self.nbins))
            print('Smoothing scale [Mpc/h]:', self.smooth)

            self.xmin = 0
            self.ymin = 0
            self.zmin = 0

        else:

            # get the weights for data and randoms
            cat.weight = cat.get_weights(fkp=0, noz=1, cp=1, syst=1)
            ran.weight = ran.get_weights(fkp=0, noz=1, cp=1, syst=1)

            # relative weighting of galaxies and randoms
            sum_wgal = np.sum(cat.weight)
            sum_wran = np.sum(ran.weight)
            alpha = sum_wgal / sum_wran
            ran_min = 0.01 * sum_wran / ran.size
            self.ran_min = ran_min
            self.alpha = alpha
            self.deltar = 0

            # convert sky coords to Cartesian
            cosmo = Cosmology(omega_m=omega_m)
            cat.dist = cosmo.get_comoving_distance(cat.redshift)
            cat.x = cat.dist * np.cos(cat.dec * np.pi / 180) * np.cos(cat.ra * np.pi / 180)
            cat.y = cat.dist * np.cos(cat.dec * np.pi / 180) * np.sin(cat.ra * np.pi / 180)
            cat.z = cat.dist * np.sin(cat.dec * np.pi / 180)
            ran.dist = cosmo.get_comoving_distance(ran.redshift)
            ran.x = ran.dist * np.cos(ran.dec * np.pi / 180) * np.cos(ran.ra * np.pi / 180)
            ran.y = ran.dist * np.cos(ran.dec * np.pi / 180) * np.sin(ran.ra * np.pi / 180)
            ran.z = ran.dist * np.sin(ran.dec * np.pi / 180)
            self.cosmo = cosmo
            self.ran = ran
            self.cat = cat

            # put the data into a box
            self.make_sky_box()

    def make_sky_box(self, padding=200.):

        dx = max(self.ran.x) - min(self.ran.x)
        dy = max(self.ran.y) - min(self.ran.y)
        dz = max(self.ran.z) - min(self.ran.z)
        x0 = 0.5 * (max(self.ran.x) + min(self.ran.x))
        y0 = 0.5 * (max(self.ran.y) + min(self.ran.y))
        z0 = 0.5 * (max(self.ran.z) + min(self.ran.z))

        box = max([dx, dy, dz]) + 2 * padding  # a bit bigger than strictly necessary
        xmin = x0 - box / 2
        ymin = y0 - box / 2
        zmin = z0 - box / 2

        # determine an appropriate bin size
        mean_dens = self.cat.size / box**3.
        # starting estimate
        self.nbins = int(np.floor(box / (0.25 * (4 * np.pi * mean_dens / 3.) ** (-1. / 3))))
        self.binsize = box / self.nbins
        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.box_length = box
        # now approximately check true survey volume and recalculate mean density
        ran = self.ran
        rhor = self.allocate_gal_cic(ran)
        filled_cells = np.sum(rhor.flatten() >= 0.)
        mean_dens = self.cat.size / (filled_cells * self.binsize**3.)
        # thus get better choice of bin size
        self.nbins = int(np.floor(box / (0.25 * (4 * np.pi * mean_dens / 3.) ** (-1. / 3))))
        self.binsize = box / self.nbins
        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.box_length = box

        # choose an appropriate smoothing scale
        smooth = mean_dens ** (-1./3)
        self.smooth = smooth

        print('Box size [Mpc/h]:', box)
        print('Bin size [Mpc/h]: %0.2f, nbins = %d' % (self.binsize, self.nbins))
        print('Smoothing scale [Mpc/h]:', smooth)

        # return xmin, ymin, zmin, box, smooth

    def allocate_gal_cic(self, c):
        """ Allocate galaxies to grid cells using a CIC scheme in order to determine galaxy
        densities on the grid"""

        xmin = self.xmin
        ymin = self.ymin
        zmin = self.zmin
        binsize = self.binsize
        nbins = self.nbins

        xpos = (c.x - xmin) / binsize
        ypos = (c.y - ymin) / binsize
        zpos = (c.z - zmin) / binsize

        i = xpos.astype(int)
        j = ypos.astype(int)
        k = zpos.astype(int)

        ddx = xpos - i
        ddy = ypos - j
        ddz = zpos - k

        delta = np.zeros((nbins, nbins, nbins))
        edges = [np.linspace(0, nbins, nbins + 1),
                 np.linspace(0, nbins, nbins + 1),
                 np.linspace(0, nbins, nbins + 1)]

        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    if self.is_box:
                        # PBC, so wrap around the box
                        pos = np.array([(i + ii) % self.nbins, (j + jj) % self.nbins,
                                        (k + kk) % self.nbins]).transpose()
                    else:
                        pos = np.array([i + ii, j + jj, k + kk]).transpose()
                    weight = (((1 - ddx) + ii * (-1 + 2 * ddx)) *
                              ((1 - ddy) + jj * (-1 + 2 * ddy)) *
                              ((1 - ddz) + kk * (-1 + 2 * ddz))) * c.weight
                    delta_t, edges = np.histogramdd(pos, bins=edges, weights=weight)
                    delta += delta_t

        return delta

    def run_voidfinder(self):

        # create the folder in which to store various raw outputs
        raw_dir = self.output_folder + "rawVoxelInfo/"
        if not os.access(raw_dir, os.F_OK):
            os.makedirs(raw_dir)

        # measure the galaxy density field
        print('Allocating galaxies in cells...')
        rhog = self.allocate_gal_cic(self.cat)
        if self.is_box:
            # normalize number counts to get density in units of mean (i.e. 1 + delta)
            rhog = (rhog * self.box_length ** 3.) / (self.cat.size * self.binsize ** 3.)
            # then smooth with pre-determined smoothing scale
            print('Smoothing galaxy density field ...')
            rhog = gaussian_filter(rhog, self.smooth / self.binsize, mode='wrap')
        else:
            print('Allocating randoms in cells...')
            rhor = self.allocate_gal_cic(self.ran)
            # normalize densities using the randoms, avoiding possible divide-by-zero errors
            delta = rhog - self.alpha * rhor
            w = np.where(rhor > self.ran_min)
            delta[w] = delta[w] / (self.alpha * rhor[w])
            w2 = np.where((rhor <= self.ran_min))  # possible divide-by-zero sites, density estimate not reliable here
            delta[w2] = 0.
            rhog = delta + 1.
            del w
            del w2
            # then smooth with pre-determined smoothing scale
            print('Smoothing galaxy density field ...')
            rhog = gaussian_filter(rhog, self.smooth / self.binsize, mode='nearest')

        self.rhog = rhog

        # write this to file for jozov-grid to read
        rhogflat = np.array(np.copy(rhog.flatten()), dtype=np.float32)
        print('Debug: rhogflat[0] = %0.4e, rhogflat[-1] = %0.4e' % (rhogflat[0], rhogflat[-1]))
        with open(raw_dir + 'density_n%d.dat' % self.nbins, 'w') as F:
            rhogflat.tofile(F, format='%f')

        # now call jozov-grid
        logfolder = self.output_folder + 'log/'
        if not os.access(logfolder, os.F_OK):
            os.makedirs(logfolder)
        logfile = logfolder + self.handle + '-voxel.out'
        cmd = ["./bin/jozov-grid", "v", raw_dir + "density_n%d.dat" % self.nbins, self.handle, str(self.nbins)]
        log = open(logfile, 'w')
        subprocess.call(cmd)
        log.close()

        # move the raw output files to appropriate folder
        cmd = ["mv", self.handle + ".txt", raw_dir + self.handle + ".txt"]
        subprocess.call(cmd)
        cmd = ["mv", self.handle + ".void", raw_dir + self.handle + ".void"]
        subprocess.call(cmd)
        cmd = ["mv", self.handle + ".zone", raw_dir + self.handle + ".zone"]
        subprocess.call(cmd)

        # postprocess void data
        self.postprocess_voids()

        # if reqd, find superclusters
        if self.find_clusters:
            logfolder = self.output_folder + 'log/'
            if not os.access(logfolder, os.F_OK):
                os.makedirs(logfolder)
            logfile = logfolder + self.handle + '-voxel.out'
            cmd = ["./bin/jozov-grid", "c", raw_dir + "density_n%d.dat" % self.nbins, self.handle, str(self.nbins)]
            log = open(logfile, 'w')
            subprocess.call(cmd)
            log.close()

            # move the raw output files to appropriate folder
            cmd = ["mv", self.handle + "c.txt", raw_dir + self.handle + "c.txt"]
            subprocess.call(cmd)
            cmd = ["mv", self.handle + "c.void", raw_dir + self.handle + "c.void"]
            subprocess.call(cmd)
            cmd = ["mv", self.handle + "c.zone", raw_dir + self.handle + "c.zone"]
            subprocess.call(cmd)

            self.postprocess_clusters()

    def postprocess_voids(self):

        raw_dir = self.output_folder + "rawVoxelInfo/"
        rawdata = np.loadtxt(raw_dir + self.handle + ".txt", skiprows=2)

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
        nvox = self.nbins**3

        print("Post-processing voids")

        # void effective radii
        vols = (rawdata[:, 5] * self.binsize ** 3.)
        rads = (3. * vols / (4. * np.pi)) ** (1. / 3)
        # void minimum density centre locations
        xpos, ypos, zpos = self.voxel_position(rawdata[:, 2])
        # void minimum densities (as delta)
        mindens = rawdata[:, 3] - 1.
        # void average densities and barycentres
        avgdens = np.zeros(len(rawdata))
        barycentres = np.zeros((len(rawdata), 3))
        rhoflat = self.rhog.flatten()
        for i in range(len(rawdata)):
            member_voxels = np.fromstring(hierarchy[i], dtype=int, sep=' ')[1:]
            member_dens = rhoflat[member_voxels]
            avgdens[i] = np.mean(member_dens) - 1.
            if self.use_barycentres:
                member_x, member_y, member_z = self.voxel_position(member_voxels)
                barycentres[i, 0] = np.average(member_x, weights=1. / member_dens)
                barycentres[i, 1] = np.average(member_y, weights=1. / member_dens)
                barycentres[i, 2] = np.average(member_z, weights=1. / member_dens)
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

        # cut on minimum density criterion
        output = output[rawdata[:, 3] < self.min_dens_cut]
        barycentres = barycentres[rawdata[:, 3] < self.min_dens_cut]
        print('Total %d voids pass basic density cuts' % len(output))
        # sort in increasing order of minimum density
        output = output[np.argsort(output[:, 5])]
        # save to file
        catalogue_file = self.output_folder + self.void_prefix + '_cat.txt'
        np.savetxt(catalogue_file, output, fmt='%d %0.4f %0.4f %0.4f %0.4f %0.6f %0.6f %0.6f %0.6f',
                   header='%d voxels, %d voids\n' % (nvox, len(output)) +
                          'VoidID XYZ[3](Mpc/h) R_eff(Mpc/h) delta_min delta_avg lambda_v DensRatio')

        if self.use_barycentres:
            if not os.access(self.output_folder + "barycentres/", os.F_OK):
                os.makedirs(self.output_folder + "barycentres/")
            catalogue_file = self.output_folder + 'barycentres/' + self.void_prefix + '_baryC_cat.txt'
            output[:, 1:4] = barycentres
            np.savetxt(catalogue_file, output, fmt='%d %0.4f %0.4f %0.4f %0.4f %0.6f %0.6f %0.6f %0.6f',
                       header='%d voxels, %d voids\n' % (nvox, len(output)) +
                              'VoidID XYZ[3](Mpc/h) R_eff(Mpc/h) delta_min delta_avg lambda_v DensRatio')

    def postprocess_clusters(self):

        raw_dir = self.output_folder + "rawVoxelInfo/"
        rawdata = np.loadtxt(raw_dir + self.handle + "c.txt", skiprows=2)

        print("Post-processing clusters")

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

        # cluster effective radii
        vols = (rawdata[:, 5] * self.binsize ** 3.)
        rads = (3. * vols / (4. * np.pi)) ** (1. / 3)
        # cluster maximum density centre locations
        xpos, ypos, zpos = self.voxel_position(rawdata[:, 2])
        # cluster maximum densities (as delta)
        maxdens = rawdata[:, 3] - 1.
        # cluster average densities
        avgdens = np.zeros(len(rawdata))
        rhoflat = self.rhog.flatten()
        for i in range(len(rawdata)):
            member_voxels = np.fromstring(hierarchy[i], dtype=int, sep=' ')[1:]  # allvoxels[zones[:] == zoneid]
            member_dens = rhoflat[member_voxels]
            avgdens[i] = np.mean(member_dens) - 1.
        # record cluster lambda value, even though usefulness of this has only been shown for ZOBOV clusters so far
        cluster_lambda = avgdens * (rads ** 1.6)

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
        # cut on maximum density criterion
        output = output[rawdata[:, 3] > self.max_dens_cut]
        print('Total %d clusters pass basic density cuts' % len(output))
        # sort in decreasing order of maximum density
        output = output[np.argsort(output[:, 5])[::-1]]
        catalogue_file = self.output_folder + self.cluster_prefix + '_cat.txt'
        np.savetxt(catalogue_file, output, fmt='%d %0.4f %0.4f %0.4f %0.4f %0.6f %0.6f %0.6f %0.6f',
                   header='%d voxels, %d clusters\n' % (nvox, len(output)) +
                          'ClusterID XYZ[3](Mpc/h) R_eff(Mpc/h) delta_max delta_avg lambda_c DensRatio')

    def voxel_position(self, voxel):

        xind = np.array(voxel / (self.nbins ** 2), dtype=int)
        yind = np.array((voxel - xind * self.nbins ** 2) / self.nbins, dtype=int)
        zind = np.array(voxel % self.nbins, dtype=int)
        xpos = (xind + 0.5) * self.box_length / self.nbins
        ypos = (yind + 0.5) * self.box_length / self.nbins
        zpos = (zind + 0.5) * self.box_length / self.nbins

        return xpos, ypos, zpos
