from __future__ import print_function
import numpy as np
import os
import json
from scipy.ndimage.filters import gaussian_filter
from scipy.fftpack import fftfreq
from classes import Cosmology
import pyfftw


class Recon:

    def __init__(self, cat, ran, is_box=False, box_length=1000., omega_m=0.308, bias=2.3, f=0.817, smooth=10.,
                 nbins=256, padding=200., opt_box=1, nthreads=1):

        beta = f / bias
        self.is_box = is_box

        # -- parameters of box
        cosmo = Cosmology(omega_m=omega_m)
        print('Number of bins:', nbins)
        print('Smoothing scale [Mpc/h]:', smooth)

        # initialize the basics common to all analyses
        self.nbins = nbins
        self.bias = bias
        self.f = f
        self.beta = beta
        self.smooth = smooth
        self.cat = cat
        self.cosmo = cosmo
        self.nthreads = nthreads
        self.is_box = is_box

        # if dealing with data in a simulation box with PBC and uniform selection
        # the random data is not used and many of the processing steps are not required,
        # so check and proceed on this basis
        if not self.is_box:

            # get the weights for data and randoms
            cat.weight = cat.get_weights(fkp=1, noz=1, cp=1, syst=1)
            ran.weight = ran.get_weights(fkp=1, noz=0, cp=0, syst=0)

            # compute Cartesian positions for data and randoms
            cat.dist = cosmo.get_comoving_distance(cat.redshift)
            cat.x = cat.dist * np.cos(cat.dec * np.pi / 180) * np.sin(cat.ra * np.pi / 180)
            cat.y = cat.dist * np.cos(cat.dec * np.pi / 180) * np.cos(cat.ra * np.pi / 180)
            cat.z = cat.dist * np.sin(cat.dec * np.pi / 180)
            cat.newx = cat.x * 1.
            cat.newy = cat.y * 1.
            cat.newz = cat.z * 1.
            ran.dist = cosmo.get_comoving_distance(ran.redshift)
            ran.x = ran.dist * np.cos(ran.dec * np.pi / 180) * np.sin(ran.ra * np.pi / 180)
            ran.y = ran.dist * np.cos(ran.dec * np.pi / 180) * np.cos(ran.ra * np.pi / 180)
            ran.z = ran.dist * np.sin(ran.dec * np.pi / 180)
            ran.newx = ran.x * 1.
            ran.newy = ran.y * 1.
            ran.newz = ran.z * 1.

            # print('Randoms min of x, y, z', min(ran.x), min(ran.y), min(ran.z))
            # print('Randoms max of x, y, z', max(ran.x), max(ran.y), max(ran.z))

            sum_wgal = np.sum(cat.weight)
            sum_wran = np.sum(ran.weight)
            # relative weighting of galaxies and randoms
            alpha = sum_wgal / sum_wran
            ran_min = 0.01 * sum_wran / ran.size

            self.ran = ran
            self.ran_min = ran_min
            self.alpha = alpha
            self.deltar = 0
            self.xmin, self.ymin, self.zmin, self.box, self.binsize = \
                self.compute_box(padding=padding, optimize_box=opt_box)

        else:
            self.ran = cat  # we will not use this!
            self.xmin = 0
            self.ymin = 0
            self.zmin = 0
            self.box = box_length
            self.binsize = self.box / self.nbins
            print('Box size [Mpc/h]:', self.box)
            print('Bin size [Mpc/h]:', self.binsize)

        # initialize a bunch of things to zero, will be set later
        self.delta = 0
        self.deltak = 0
        self.psi_x = 0
        self.psi_y = 0
        self.psi_z = 0
        self.fft_obj = 0
        self.ifft_obj = 0

    def compute_box(self, padding=200., optimize_box=1):

        if optimize_box:
            dx = max(self.ran.x) - min(self.ran.x)
            dy = max(self.ran.y) - min(self.ran.y)
            dz = max(self.ran.z) - min(self.ran.z)
            x0 = 0.5 * (max(self.ran.x) + min(self.ran.x))
            y0 = 0.5 * (max(self.ran.y) + min(self.ran.y))
            z0 = 0.5 * (max(self.ran.z) + min(self.ran.z))

            box = max([dx, dy, dz]) + 2 * padding
            xmin = x0 - box / 2
            ymin = y0 - box / 2
            zmin = z0 - box / 2
            box = box
            binsize = box / self.nbins
        else:
            box = self.cosmo.get_comoving_distance(1.05)
            xmin = -box
            ymin = -box
            zmin = -box
            box = box * 2.
            binsize = box / self.nbins

        print('Box size [Mpc/h]:', box)
        print('Bin size [Mpc/h]:', binsize)

        return xmin, ymin, zmin, box, binsize

    def iterate(self, iloop, save_wisdom=1):
        cat = self.cat
        ran = self.ran
        smooth = self.smooth
        binsize = self.binsize
        beta = self.beta
        bias = self.bias
        f = self.f
        nbins = self.nbins

        # -- Creating arrays for FFTW
        if iloop == 0:  # first iteration requires initialization
            delta = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
            deltak = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
            psi_x = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
            psi_y = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
            psi_z = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')

            if self.is_box:
                deltar = 0
            else:
                print('Allocating randoms in cells...')
                deltar = self.allocate_gal_cic(ran)
                print('Smoothing...')
                deltar = gaussian_filter(deltar, smooth/binsize)

            # -- Initialize FFT objects and load wisdom if available
            wisdom_file = "wisdom." + str(nbins) + "." + str(self.nthreads)
            if os.path.isfile(wisdom_file):
                print('Reading wisdom from ', wisdom_file)
                g = open(wisdom_file, 'r')
                wisd = json.load(g)
                pyfftw.import_wisdom(wisd)
                g.close()
            print('Creating FFTW objects...')
            fft_obj = pyfftw.FFTW(delta, delta, axes=[0, 1, 2], threads=self.nthreads)
            ifft_obj = pyfftw.FFTW(deltak, psi_x, axes=[0, 1, 2],
                                   threads=self.nthreads,
                                   direction='FFTW_BACKWARD')
        else:
            delta = self.delta
            deltak = self.deltak
            deltar = self.deltar
            psi_x = self.psi_x
            psi_y = self.psi_y
            psi_z = self.psi_z
            fft_obj = self.fft_obj
            ifft_obj = self.ifft_obj

        # -- Allocate galaxies and randoms to grid with CIC method
        # -- using new positions
        print('Allocating galaxies in cells...')
        deltag = self.allocate_gal_cic(cat)
        print('Smoothing galaxy density field ...')
        if self.is_box:
            deltag = gaussian_filter(deltag, smooth/binsize, mode='wrap')  # mode='wrap' for PBC
        else:
            # in this case the filter mode should not be important, due to the box padding
            # but just in case, use mode='nearest' to continue with zeros
            deltag = gaussian_filter(deltag, smooth / binsize, mode='nearest')

        print('Computing density fluctuations, delta...')
        if self.is_box:
            # simply normalize based on (constant) mean galaxy number density
            delta[:] = (deltag * self.box**3.)/(cat.size * self.binsize**3.) - 1.
        else:
            # normalize using the randoms, avoiding possible divide-by-zero errors
            delta[:] = deltag - self.alpha * deltar
            w = np.where(deltar > self.ran_min)
            delta[w] = delta[w] / (self.alpha * deltar[w])
            w2 = np.where((deltar <= self.ran_min))  # possible divide-by-zero sites, density estimate not reliable here
            delta[w2] = 0.
            # additionally, remove the highest peaks of delta to reduce noise (if required?)
            # w3 = np.where(delta > np.percentile(delta[w].ravel(), 99))
            # delta[w3] = 0.
            # del w3
            del w
            del w2
        del deltag  # deltag no longer required anywhere

        print('Fourier transforming delta field...')
        fft_obj(input_array=delta, output_array=delta)

        # -- delta/k**2
        k = fftfreq(self.nbins, d=binsize) * 2 * np.pi
        ksq = k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2
        # avoid divide by zero
        ksq[ksq == 0] = 1.
        delta /= ksq
        # set zero mode to 1
        delta[0, 0, 0] = 1

        # now solve the basic building block: IFFT[-i k delta(k)/(b k^2)]
        print('Inverse Fourier transforming to get psi...')
        deltak[:] = delta * -1j * k[:, None, None] / bias
        ifft_obj(input_array=deltak, output_array=psi_x)
        deltak[:] = delta * -1j * k[None, :, None] / bias
        ifft_obj(input_array=deltak, output_array=psi_y)
        deltak[:] = delta * -1j * k[None, None, :] / bias
        ifft_obj(input_array=deltak, output_array=psi_z)

        # from grid values of Psi_est = IFFT[-i k delta(k)/(b k^2)], compute the values at the galaxy positions
        shift_x, shift_y, shift_z = self.get_shift(cat, psi_x.real, psi_y.real, psi_z.real, use_newpos=True)
        # for debugging:
        for i in range(10):
            print('%0.3f %0.3f %0.3f %0.3f' %(shift_x[i], shift_y[i], shift_z[i], cat.newz[i]))

        # now we update estimates of the Psi field in the following way:
        if iloop == 0:
            # starting estimate chosen according to Eq. 12 of Burden et al 2015, in order to improve convergence
            if self.is_box:
                # line-of-sight direction is along the z-axis (hard-coded)
                psi_dot_rhat = shift_z
                shift_z -= beta / (1 + beta) * psi_dot_rhat
            else:
                # line-of-sight direction determined by galaxy coordinates
                psi_dot_rhat = (shift_x * cat.x + shift_y * cat.y + shift_z * cat.z) / cat.dist
                shift_x -= beta / (1 + beta) * psi_dot_rhat * cat.x / cat.dist
                shift_y -= beta / (1 + beta) * psi_dot_rhat * cat.y / cat.dist
                shift_z -= beta / (1 + beta) * psi_dot_rhat * cat.z / cat.dist
        # given estimate of Psi, subtract approximate RSD to get estimate of real-space galaxy positions
        if self.is_box:
            # line-of-sight direction is along the z-axis (hard-coded)
            cat.newz = cat.z + f * shift_z
            # check PBC
            cat.newz[cat.newz >= cat.box_length] -= cat.box_length
            cat.newz[cat.newz < 0] += cat.box_length
        else:
            psi_dot_rhat = (shift_x * cat.x + shift_y * cat.y + shift_z * cat.z) / cat.dist
            cat.newx = cat.x + f * psi_dot_rhat * cat.x / cat.dist
            cat.newy = cat.y + f * psi_dot_rhat * cat.y / cat.dist
            cat.newz = cat.z + f * psi_dot_rhat * cat.z / cat.dist

        # in the next loop of iteration, these new positions are used to compute next approximation of
        # the (real-space) galaxy density, and then this is used to get new estimate of Psi, etc.
        # at the end of the iterations, newx, newy, newz should be the real-space galaxy positions (or best
        # estimate thereof)

        self.deltar = deltar
        self.delta = delta
        self.deltak = deltak
        self.psi_x = psi_x
        self.psi_y = psi_y
        self.psi_z = psi_z
        self.fft_obj = fft_obj
        self.ifft_obj = ifft_obj

        # -- save wisdom
        wisdom_file = "wisdom." + str(nbins) + "." + str(self.nthreads)
        if iloop == 0 and save_wisdom and not os.path.isfile(wisdom_file):
            wisd = pyfftw.export_wisdom()
            f = open(wisdom_file, 'w')
            json.dump(wisd, f)
            f.close()
            print('Wisdom saved at', wisdom_file)

    def apply_shifts_rsd(self):
        """ Subtract RSD to get the estimated real-space positions of randoms
        (no need to do this for galaxies, since it already happens during the iteration loop)
        """

        if self.is_box:
            print('Mistaken call to apply_shifts_rsd()? No randoms to correct, galaxy positions already corrected')
        else:
            shift_x, shift_y, shift_z = \
                self.get_shift(self.ran, self.psi_x.real, self.psi_y.real, self.psi_z.real)
            psi_dot_rhat = (shift_x * self.ran.x + shift_y * self.ran.y + shift_z * self.ran.z) / self.ran.dist
            self.ran.newx = self.ran.x + self.f * psi_dot_rhat * self.ran.x / self.ran.dist
            self.ran.newy = self.ran.y + self.f * psi_dot_rhat * self.ran.y / self.ran.dist
            self.ran.newz = self.ran.z + self.f * psi_dot_rhat * self.ran.z / self.ran.dist

    def apply_shifts_full(self):
        """ Use the estimated displacement field to shift the positions of galaxies (and randoms).
        This method subtracts full displacement field as in standard BAO reconstruction"""

        for c in [self.cat, self.ran]:
            shift_x, shift_y, shift_z = \
                self.get_shift(c, self.psi_x.real, self.psi_y.real, self.psi_z.real, use_newpos=True)
            # note that Julian's eBOSS pipeline code has use_newpos=False at this point, which is not correct
            c.newx += shift_x
            c.newy += shift_y
            c.newz += shift_z
            if self.is_box:  # account for PBC
                c.newx[c.newx >= c.box_length] -= c.box_length
                c.newx[c.newx < 0] += c.box_length
                c.newy[c.newy >= c.box_length] -= c.box_length
                c.newy[c.newy < 0] += c.box_length
                c.newz[c.newz >= c.box_length] -= c.box_length
                c.newz[c.newz < 0] += c.box_length

    def summary(self):

        cat = self.cat
        sx = cat.newx - cat.x
        sy = cat.newy - cat.y
        sz = cat.newz - cat.z
        print('Shifts stats:')
        for s in [sx, sy, sz]:
            print(np.std(s), np.percentile(s, 16), np.percentile(s, 84), np.min(s), np.max(s))

    def allocate_gal_cic(self, c):
        """ Allocate galaxies to grid cells using a CIC scheme in order to determine galaxy
        densities on the grid"""

        xmin = self.xmin
        ymin = self.ymin
        zmin = self.zmin
        binsize = self.binsize
        nbins = self.nbins

        xpos = (c.newx - xmin) / binsize
        ypos = (c.newy - ymin) / binsize
        zpos = (c.newz - zmin) / binsize

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

    def get_shift(self, c, f_x, f_y, f_z, use_newpos=False):
        """Given grid of f_x, f_y and f_z values, uses interpolation scheme to compute
        appropriate values at the galaxy positions"""

        xmin = self.xmin
        ymin = self.ymin
        zmin = self.zmin
        binsize = self.binsize

        if use_newpos:
            xpos = (c.newx - xmin) / binsize
            ypos = (c.newy - ymin) / binsize
            zpos = (c.newz - zmin) / binsize
        else:
            xpos = (c.x - xmin) / binsize
            ypos = (c.y - ymin) / binsize
            zpos = (c.z - zmin) / binsize

        i = xpos.astype(int)
        j = ypos.astype(int)
        k = zpos.astype(int)

        ddx = xpos - i
        ddy = ypos - j
        ddz = zpos - k

        shift_x = np.zeros(c.size)
        shift_y = np.zeros(c.size)
        shift_z = np.zeros(c.size)

        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    weight = (((1 - ddx) + ii * (-1 + 2 * ddx)) *
                              ((1 - ddy) + jj * (-1 + 2 * ddy)) *
                              ((1 - ddz) + kk * (-1 + 2 * ddz)))
                    if self.is_box:
                        pos = ((i + ii) % self.nbins, (j + jj) % self.nbins, (k + kk) % self.nbins)
                    else:
                        pos = (i + ii, j + jj, k + kk)
                    shift_x += f_x[pos] * weight
                    shift_y += f_y[pos] * weight
                    shift_z += f_z[pos] * weight

        return shift_x, shift_y, shift_z

    def export_shift_pos(self, root1, root2='', rsd_only=True):
        """method to write the shifted positions to file"""

        if self.is_box:
            output = np.zeros((self.cat.size, 3))
            output[:, 0] = self.cat.newx
            output[:, 1] = self.cat.newy
            output[:, 2] = self.cat.newz
            out_file = root1 + '_shift.npy'
            np.save(out_file, output)
        else:
            output = np.zeros((self.cat.size, 4))
            output[:, 0] = self.cat.ra
            output[:, 1] = self.cat.dec
            output[:, 2] = self.cat.redshift
            output[:, 3] = self.cat.weight
            out_file = root1 + '_shift.npy'
            np.save(out_file, output)

            output = np.zeros((self.cat.size, 3))
            output[:, 0] = self.cat.newx
            output[:, 1] = self.cat.newy
            output[:, 2] = self.cat.newz
            out_file = root1 + '_shift_xyz.npy'
            np.save(out_file, output)

            if not rsd_only:
                output = np.zeros((self.ran.size, 4))
                output[:, 0] = self.ran.ra
                output[:, 1] = self.ran.dec
                output[:, 2] = self.ran.redshift
                output[:, 3] = self.ran.weight
                out_file = root2 + '_shift.npy'
                np.save(out_file, output)

    def cart_to_radecz(self, x, y, z):

        dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        dec = np.arcsin(z / dist) * 180. / np.pi
        ra = np.arctan(x / y) * 180. / np.pi + 180
        redshift = self.cosmo.get_redshift(dist)
        return ra, dec, redshift

    def get_new_radecz(self, c):

        return self.cart_to_radecz(c.newx, c.newy, c.newz)