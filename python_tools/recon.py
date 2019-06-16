from __future__ import print_function
import numpy as np
import os
import json
import sys
from scipy.fftpack import fftfreq
from cosmology import Cosmology
import pyfftw
import fastmodules


class Recon:

    def __init__(self, cat, ran, is_box=False, box_length=1000., omega_m=0.308, bias=2.3, f=0.817, smooth=10.,
                 nbins=256, padding=200., opt_box=1, nthreads=1, verbose=False):

        beta = f / bias
        self.is_box = is_box
        self.verbose = verbose

        # -- parameters of box
        cosmo = Cosmology(omega_m=omega_m)
        print("\n ==== Starting the reconstruction ==== ")
        print('Using values of growth rate f = %0.3f and bias b = %0.3f' % (f, bias))
        print('Smoothing scale [Mpc/h]:', smooth)
        if self.verbose:
            print('Number of bins:', nbins)
        sys.stdout.flush()

        # initialize the basics common to all analyses
        self.nbins = nbins
        self.bias = bias
        self.f = f
        self.beta = beta
        self.smooth = smooth
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
            if self.verbose:
                print('Box size [Mpc/h]: %0.2f' % self.box)
                print('Bin size [Mpc/h]: %0.2f' % self.binsize)

        sys.stdout.flush()
        self.cat = cat

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
            maxx = np.max(self.ran.x)
            minx = np.min(self.ran.x)
            maxy = np.max(self.ran.y)
            miny = np.min(self.ran.y)
            maxz = np.max(self.ran.z)
            minz = np.min(self.ran.z)
            dx = maxx - minx
            dy = maxy - miny
            dz = maxz - minz
            x0 = 0.5 * (maxx + minx)
            y0 = 0.5 * (maxy + miny)
            z0 = 0.5 * (maxz + minz)

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

        if self.verbose:
            print('Box size [Mpc/h]: %0.2f' % box)
            print('Bin size [Mpc/h]: %0.2f' % binsize)

        return xmin, ymin, zmin, box, binsize

    def iterate(self, iloop, save_wisdom=1, debug=False):

        cat = self.cat
        ran = self.ran
        binsize = self.binsize
        beta = self.beta
        bias = self.bias
        f = self.f
        nbins = self.nbins

        print("Loop %d" % iloop)
        # -- Creating arrays for FFTW
        if iloop == 0:  # first iteration requires initialization
            delta = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
            deltak = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
            rho = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
            rhok = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
            psi_x = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
            psi_y = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
            psi_z = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')

            # -- Initialize FFT objects and load wisdom if available
            wisdom_file = "wisdom." + str(nbins) + "." + str(self.nthreads)
            if os.path.isfile(wisdom_file):
                print('Reading wisdom from ', wisdom_file)
                g = open(wisdom_file, 'r')
                wisd = json.load(g)
                pyfftw.import_wisdom(wisd)
                g.close()
            print('Creating FFTW objects...')
            sys.stdout.flush()
            fft_obj = pyfftw.FFTW(delta, delta, axes=[0, 1, 2], threads=self.nthreads)
            ifft_obj = pyfftw.FFTW(deltak, psi_x, axes=[0, 1, 2],
                                   threads=self.nthreads,
                                   direction='FFTW_BACKWARD')
            kr = fftfreq(nbins, d=binsize) * 2 * np.pi * self.smooth
            norm = np.exp(-0.5 * (kr[:, None, None] ** 2 + kr[None, :, None] ** 2 + kr[None, None, :] ** 2))

            if self.is_box:
                deltar = 0
            else:
                if self.verbose:
                    print('Allocating randoms in cells...')
                sys.stdout.flush()
                deltar = np.zeros((nbins, nbins, nbins), dtype='float64')
                fastmodules.allocate_gal_cic(deltar, ran.x, ran.y, ran.z, ran.weight, ran.size, self.xmin, self.ymin,
                                             self.zmin, self.box, nbins, 1.)
                if self.verbose:
                    print('Smoothing...')
                sys.stdout.flush()
                # NOTE - we do the smoothing via FFTs rather than scipy's gaussian_filter because if using several
                # threads for pyfftw it is much faster this way (if only using 1 thread gains are negligible)
                rho = deltar + 0.0j
                fft_obj(input_array=rho, output_array=rhok)
                fastmodules.mult_norm(rhok, rhok, norm)
                ifft_obj(input_array=rhok, output_array=rho)
                deltar = rho.real

        else:
            delta = self.delta
            deltak = self.deltak
            deltar = self.deltar
            rho = self.rho
            rhok = self.rhok
            psi_x = self.psi_x
            psi_y = self.psi_y
            psi_z = self.psi_z
            fft_obj = self.fft_obj
            ifft_obj = self.ifft_obj
            norm = self.norm

        # -- Allocate galaxies and randoms to grid with CIC method
        # -- using new positions
        if self.verbose:
            print('Allocating galaxies in cells...')
        sys.stdout.flush()
        deltag = np.zeros((nbins, nbins, nbins), dtype='float64')
        fastmodules.allocate_gal_cic(deltag, cat.newx, cat.newy, cat.newz, cat.weight, cat.size, self.xmin, self.ymin,
                                  self.zmin, self.box, nbins, 1.)
        if self.verbose:
            print('Smoothing galaxy density field ...')
        sys.stdout.flush()
        # NOTE - smoothing via FFTs
        rho = deltag + 0.0j
        fft_obj(input_array=rho, output_array=rhok)
        fastmodules.mult_norm(rhok, rhok, norm)
        ifft_obj(input_array=rhok, output_array=rho)
        deltag = rho.real

        if self.verbose:
            print('Computing density fluctuations, delta...')
        sys.stdout.flush()
        if self.is_box:
            # simply normalize based on (constant) mean galaxy number density
            fastmodules.normalize_delta_box(delta, deltag, cat.size)
        else:
            # normalize using the randoms, avoiding possible divide-by-zero errors
            fastmodules.normalize_delta_survey(delta, deltag, deltar, self.alpha, self.ran_min)
        del deltag  # deltag no longer required anywhere

        if self.verbose:
            print('Fourier transforming delta field...')
        sys.stdout.flush()
        fft_obj(input_array=delta, output_array=delta)

        # -- delta/k**2
        k = fftfreq(self.nbins, d=binsize) * 2 * np.pi
        fastmodules.divide_k2(delta, delta, k)

        # now solve the basic building block: IFFT[-i k delta(k)/(b k^2)]
        if self.verbose:
            print('Inverse Fourier transforming to get psi...')
        sys.stdout.flush()
        fastmodules.mult_kx(deltak, delta, k, bias)
        ifft_obj(input_array=deltak, output_array=psi_x)
        fastmodules.mult_ky(deltak, delta, k, bias)
        ifft_obj(input_array=deltak, output_array=psi_y)
        fastmodules.mult_kz(deltak, delta, k, bias)
        ifft_obj(input_array=deltak, output_array=psi_z)

        # from grid values of Psi_est = IFFT[-i k delta(k)/(b k^2)], compute the values at the galaxy positions
        if self.verbose:
            print('Calculating shifts...')
        sys.stdout.flush()
        shift_x, shift_y, shift_z = self.get_shift(cat, psi_x.real, psi_y.real, psi_z.real, use_newpos=True)

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

        # for debugging:
        if self.verbose and debug:
            if self.is_box:
                print('Debug: first 10 x,y,z shifts and old and new z positions')
                for i in range(10):
                    print('%0.3f %0.3f %0.3f %0.3f %0.3f' % (shift_x[i], shift_y[i], shift_z[i], cat.z[i], cat.newz[i]))

            else:
                print('Debug: first 10 x,y,z shifts and old and new observer distances')
                for i in range(10):
                    oldr = np.sqrt(cat.x[i] ** 2 + cat.y[i] ** 2 + cat.z[i] ** 2)
                    newr = np.sqrt(cat.newx[i] ** 2 + cat.newy[i] ** 2 + cat.newz[i] ** 2)
                    print('%0.3f %0.3f %0.3f %0.3f %0.3f' % (shift_x[i], shift_y[i], shift_z[i], oldr, newr))

        # in the next loop of iteration, these new positions are used to compute next approximation of
        # the (real-space) galaxy density, and then this is used to get new estimate of Psi, etc.
        # at the end of the iterations, newx, newy, newz should be the real-space galaxy positions (or best
        # estimate thereof)
        self.deltar = deltar
        self.delta = delta
        self.deltak = deltak
        self.rho = rho
        self.rhok = rhok
        self.psi_x = psi_x
        self.psi_y = psi_y
        self.psi_z = psi_z
        self.fft_obj = fft_obj
        self.ifft_obj = ifft_obj
        self.norm = norm

        # -- save wisdom
        wisdom_file = "wisdom." + str(nbins) + "." + str(self.nthreads)
        if iloop == 0 and save_wisdom and not os.path.isfile(wisdom_file):
            wisd = pyfftw.export_wisdom()
            f = open(wisdom_file, 'w')
            json.dump(wisd, f)
            f.close()
            print('Wisdom saved at', wisdom_file)

    def apply_shifts_rsd(self):
        """
        Subtract RSD to get the estimated real-space positions of randoms
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
        """
        Uses the estimated displacement field to shift the positions of galaxies (and randoms).
        This method subtracts full displacement field as in standard BAO reconstruction
        """

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
            # recalculate weights, as we don't want the FKP weighting for void-finding
            self.cat.weight = self.cat.get_weights(fkp=0, noz=1, cp=1, syst=1)
            output = np.zeros((self.cat.size, 4))
            output[:, 0] = self.cat.ra
            output[:, 1] = self.cat.dec
            output[:, 2] = self.cat.redshift
            output[:, 3] = self.cat.weight
            out_file = root1 + '_shift.npy'
            np.save(out_file, output)

            if not rsd_only:
                # same as above, but for the randoms as well
                self.ran.weight = self.ran.get_weights(fkp=0, noz=0, cp=0, syst=0)
                output = np.zeros((self.ran.size, 4))
                output[:, 0] = self.ran.ra
                output[:, 1] = self.ran.dec
                output[:, 2] = self.ran.redshift
                output[:, 3] = self.ran.weight
                out_file = root2 + '_shift.npy'
                np.save(out_file, output)

    def cart_to_radecz(self, x, y, z):

        dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        dec = 90 - np.degrees(np.arccos(z / dist))
        ra = np.degrees(np.arctan2(y, x))
        ra[ra < 0] += 360
        redshift = self.cosmo.get_redshift(dist)
        return ra, dec, redshift

    def get_new_radecz(self, c):

        return self.cart_to_radecz(c.newx, c.newy, c.newz)
