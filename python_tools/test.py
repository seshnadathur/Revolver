import numpy as np
import time
import os
import pyfftw
import fastmult
from scipy.ndimage.filters import gaussian_filter
import json
from scipy.fftpack import fftfreq

def sesh_cic(x,y,z,nbins):

    xpos = x * nbins
    ypos = y * nbins
    zpos = z * nbins

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
                # pos = np.array([i + ii, j + jj, k + kk]).transpose()
                pos = np.array([(i + ii) % nbins, (j + jj) % nbins,
                                (k + kk) % nbins]).transpose()
                weight = (((1 - ddx) + ii * (-1 + 2 * ddx)) *
                          ((1 - ddy) + jj * (-1 + 2 * ddy)) *
                          ((1 - ddz) + kk * (-1 + 2 * ddz))) * 1.0
                delta_t, edges = np.histogramdd(pos, bins=edges, weights=weight)
                delta += delta_t

    return delta

def allocate_gal_cic_fast(x, y, z, xmin, ymin, zmin, binsize, nbins):
    """ Allocate galaxies to grid cells using a CIC scheme in order to determine galaxy
    densities on the grid"""

    boxsize = 1. #binsize * nbins

    # Scale positions to lie inside [0,1]
    xpos = (x - xmin) / boxsize
    ypos = (y - ymin) / boxsize
    zpos = (z - zmin) / boxsize

    # Make output array (as written it works with doubles only, but can be changed)
    delta = np.zeros((nbins, nbins, nbins), dtype='float64')

    # need to add a wrap condition here if self.is_box is True!
    cic.perform_cic_3D_w(delta, xpos, ypos, zpos, np.ones_like(xpos))

    return delta


def gaussian_filter_fftw(deltagal, Rsmooth, binsize, nbins, nthreads):
    """
      A faster way of doing the smoothing using FFT's (25sec -> 10sec)
      The smoothing radius is in Mpc/h (same units as binsize and the same as 'smooth')
      Replace deltar = gaussian_filter(...) with deltar = gaussian_filter_fftw(delta_r, smooth, ...)
      and similar for deltag
    """
    # print("Doing gaussian smoothing R = ", Rsmooth)

    # Load wisdom
    wisdom_file = "../wisdom." + str(nbins) + "." + str(nthreads)
    if os.path.isfile(wisdom_file):
        print('Reading wisdom from ', wisdom_file)
        g = open(wisdom_file, 'r')
        wisd = json.load(g)
        pyfftw.import_wisdom(wisd)
        g.close()

    # Allocate array and set up fft plans
    delta   = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
    delta_k = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
    fft_s   = pyfftw.FFTW(delta, delta_k, axes=[0, 1, 2], threads=nthreads)
    ifft_s  = pyfftw.FFTW(delta_k, delta, axes=[0, 1, 2], threads=nthreads, direction='FFTW_BACKWARD')

    # Assign deltagal -> delta
    delta = deltagal + 1j * 0.

    # Transform to fourier space
    fft_s()

    # Gaussian factor e^[-1/2 * (k*R)^2]
    kR = fftfreq(nbins, d=binsize) * 2 * np.pi * Rsmooth
    norm = np.exp(-0.5 * (kR[:, None, None] ** 2 + kR[None, :, None] ** 2 + kR[None, None, :] ** 2))
    delta_k = delta_k * norm
    # fastmult.mult_kr2(delta_k, delta, kR)

    # Transform back
    ifft_s()

    # Return array is float64
    delta_out = np.real(delta)

    # Free memory and return real array
    del delta_k
    del delta

    return delta_out

# Set parameters
npart = 6000000
npartg = 600000
ngrid = 512
nbins = ngrid
nthreads = 1
binsize = 2500./ngrid
smooth = 10
bias = 1.85
print "We have npart = ", npart, " npartg = ", npartg, " ngrid = ", ngrid

# Generate random data
x = np.random.rand(npart) * 0.9
y = np.random.rand(npart) * 0.9
z = np.random.rand(npart) * 0.9
xg = np.random.rand(npartg) * 0.9
yg = np.random.rand(npartg) * 0.9
zg = np.random.rand(npartg) * 0.9
grid3d_sesh = np.zeros((ngrid,ngrid,ngrid),dtype='float64')
grid3d_cython = np.zeros((ngrid,ngrid,ngrid),dtype='float64')
grid3d_cic = np.zeros((ngrid,ngrid,ngrid),dtype='float64')

# ====== Test CIC routines ======== #
start = time.time()
fastmult.allocate_gal_cic(grid3d_cython,x,y,z,None,npart,0.0,0.0,0.0,1.0,ngrid,0)
end = time.time(); print "Time CIC 3D grid cython: ", end-start

# start = time.time()
# grid3d_sesh = sesh_cic(x,y,z,ngrid)
# end = time.time(); print "Time CIC 3D grid Sesh: ", end-start
#
# start = time.time()
# grid3d_cic = allocate_gal_cic_fast(x, y, z, 0., 0., 0., binsize, ngrid)
# end = time.time(); print "Time CIC 3D grid SWIG: ", end-start
#
# diff = np.abs(grid3d_sesh-grid3d_cython)
# print "Max difference wrt cython: ", np.amax(diff)
# diff = np.abs(grid3d_sesh-grid3d_cic)
# print "Max difference wrt SWIG: ", np.amax(diff)
# ================================= #

# ======= Test FFTW operation ======= #
delta = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
deltak = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
psi_x = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')

wisdom_file = "../wisdom." + str(nbins) + "." + str(nthreads)
if os.path.isfile(wisdom_file):
    print('Reading wisdom from ', wisdom_file)
    g = open(wisdom_file, 'r')
    wisd = json.load(g)
    pyfftw.import_wisdom(wisd)
    g.close()
print('Creating FFTW objects...')
begin = time.time()
fft_obj = pyfftw.FFTW(delta, delta, axes=[0, 1, 2], threads=nthreads)
finish = time.time(); print 'Time creating FFT object %0.3f' % (finish - begin)
begin = time.time()
ifft_obj = pyfftw.FFTW(deltak, delta, axes=[0, 1, 2], threads=nthreads, direction='FFTW_BACKWARD')
finish = time.time(); print 'Time creating IFFT object %0.3f' % (finish - begin)

start = time.time()
delta[:] = (grid3d_cython * ngrid**3.)/(npart) - 1.
end = time.time(); print "Time delta: ", end-start
print 'delta max:', np.max(delta)
print 'delta min:', np.min(delta)
print 'grid3d max:', np.max(grid3d_cython)
print 'grid3d min:', np.min(grid3d_cython)


begin = time.time()
fft_obj(input_array=delta, output_array=deltak)
finish = time.time(); print 'Time FFT %0.3f' % (finish - begin)
psi_x[:] = np.copy(delta)
newdelta = np.real(delta)


# begin = time.time()
# ifft_obj(input_array=deltak, output_array=delta)
# finish = time.time(); print 'Time IFFT %0.3f' % (finish - begin)
# diff = np.abs(delta-psi_x)
# print "Max difference wrt original: ", np.amax(diff)

# ======= Test fast multiplication routines ======= #
# k = fftfreq(nbins, d=binsize) * 2 * np.pi
# begin = time.time()
# fastmult.divide_k2(delta, deltak, k)
# finish = time.time(); print 'Time fast ksq %0.3f' % (finish - begin)
# begin = time.time()
# ksq = k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2
# ksq[ksq == 0] = 1.
# deltak /= ksq
# deltak[0, 0, 0] = 0
# finish = time.time(); print 'Time slow ksq %0.3f' % (finish - begin)
# diff = np.abs(deltak-delta)
# print "Max difference wrt cython: ", np.amax(diff)
#
# begin = time.time()
# fastmult.mult_kx(delta, newdeltak, k, bias)
# finish = time.time(); print 'Time fast mult_kx %0.3f' % (finish - begin)
# begin = time.time()
# deltak[:] = newdeltak * -1j * k[:, None, None] / bias
# finish = time.time(); print 'Time slow mult_kx %0.3f' % (finish - begin)
# diff = np.abs(deltak-delta)
# print "Max difference wrt cython: ", np.amax(diff)

# begin = time.time()
# fastmult.mult_ky(deltak, delta, k, bias)
# finish = time.time(); print 'Time fast mult_ky %0.3f' % (finish - begin)
# newdelta = np.copy(deltak)
# begin = time.time()
# deltak[:] = delta * -1j * k[None, :, None] / bias
# finish = time.time(); print 'Time slow mult_ky %0.3f' % (finish - begin)
# diff = np.abs(deltak-newdelta)
# print "Max difference wrt cython: ", np.amax(diff)

start = time.time()
deltag = gaussian_filter(newdelta, smooth/binsize, mode='wrap')
end = time.time(); print "Time scipy smooth: ", end-start
print 'scipy smoothed max:', np.max(deltag)
print 'scipy smoothed min:', np.min(deltag)

start = time.time()
fft_obj(input_array=delta, output_array=deltak)
# start = time.time()
kR = fftfreq(nbins, d=binsize) * 2 * np.pi * smooth
norm = np.exp(-0.5 * (kR[:, None, None] ** 2 + kR[None, :, None] ** 2 + kR[None, None, :] ** 2))
deltak = deltak * norm
# end = time.time(); print "Time slow kR mult: ", end-start
# start = time.time()
# fastmult.mult_kr2(delta, deltak, kR)
# end = time.time(); print "Time fast kR mult: ", end-start
# diff = np.abs(delta-psi_x)
# print "Max difference : ", np.amax(diff)
ifft_obj(input_array=deltak, output_array=delta)
delta = np.real(delta)
end = time.time(); print "Time fftw smooth: ", end-start
diff = np.abs(delta-deltag)
print "Max difference wrt scipy smooth: ", np.amax(diff)

# start = time.time()
# fastsmooth = gaussian_filter_fftw(grid3d_cython, smooth, binsize, ngrid, nthreads)
# end = time.time(); print "Time fftw smooth: ", end-start
# diff = np.abs(fastsmooth-deltag)
# print "Max difference wrt scipy smooth: ", np.amax(diff)
# print 'fftw smoothed max:', np.max(fastsmooth)
# print 'fftw smoothed min:', np.min(fastsmooth)
