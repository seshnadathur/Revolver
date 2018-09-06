from __future__ import print_function
import os
import sys
import time
import numpy as np
from pyCUTE import pycute
from zobov import ZobovVoids
from voxelvoids import VoxelVoids
from galaxycat import GalaxyCatalogue
from recon import Recon
from fastmodules import survey_cuts_logical


cap = sys.argv[1]
index = int(sys.argv[2])
nthreads = int(sys.argv[3])

catalogue = 'CMASS'
if cap == 'N':
    fullcap = 'North'
else:
    fullcap = 'South'

handle = catalogue + '-' + cap + '-%04d' % index
tracer_file = '/mnt/lustre/hectorgm/DR12_Mocks/Patchy_V6.0C/%s/%sGC/Patchy-Mocks-DR12' % (catalogue, cap) + \
              '%s-%s-V6C-Portsmouth-mass_%04d.dat' % (catalogue, cap, index)
output_folder = '/mnt/lustre/nadathur/BOSS_DR12_voidRSD/Patchy_voids/' + handle + '/'
randoms_file = '/mnt/lustre/nadathur/BOSS_DR12_voidRSD/tracer_files_for_CUTE/gals/' \
               + 'Random-DR12%s-%s-V6C-x50.npy' % (catalogue, cap)
mask_file = '/users/nadathur/Revolver/masks/unified_DR12v5_%s_%s_completeness_n128.fits' % (catalogue, fullcap)
mock_file = '/mnt/lustre/nadathur/BOSS_DR12_voidRSD/fiducial_DR12_voids/%s-%s/%s-%s_mocks.npy' % (catalogue, cap,
                                                                                                  catalogue, cap)
void_rand_file = '/mnt/lustre/nadathur/BOSS_DR12_voidRSD/tracer_files_for_CUTE/zobov_voids/' \
                 + 'Zobov-reconVoids-Rcut-randoms-DR12-%s-%s-x50.npy' % (catalogue, cap)

omega_m = 0.307115
do_recon = True
nbins = 512
padding = 200.
smooth = 10.
bias = 2.1
f = 0.757
niter = 3
special_patchy = True
z_low_cut = 0.4
z_high_cut = 0.73
z_min = 0.43
z_max = 0.7
void_prefix = handle + '-Voids'
min_dens_cut = 1.0

if not os.access(output_folder, os.F_OK):
    os.makedirs(output_folder)


# ===== load random catalogue ===== #
ran = GalaxyCatalogue(randoms_file, is_box=False, boss_like=False, randoms=True, special_patchy=True)
wran = np.empty(ran.size, dtype=int)
survey_cuts_logical(wran, ran.veto, ran.redshift, z_low_cut, z_high_cut)
wran = np.asarray(wran, dtype=bool)
ran.cut(wran)
# ================================= #

# ===== load redshift-space galaxy catalogue ==== #
# galaxy and random catalogues ===== #
cat = GalaxyCatalogue(tracer_file, is_box=False, randoms=False, boss_like=False, special_patchy=True)
wgal = np.empty(cat.size, dtype=int)
survey_cuts_logical(wgal, cat.veto, cat.redshift, z_low_cut, z_high_cut)
wgal = np.asarray(wgal, dtype=bool)
cat.cut(wgal)
# ============================================ #

# ======= run reconstruction ======== #
start = time.time()
print('\n ==== Running reconstruction for real-space positions ==== ')
recon = Recon(cat, ran, is_box=False, omega_m=omega_m, bias=bias, f=f, smooth=smooth,
              nbins=nbins, padding=padding, nthreads=nthreads, verbose=True)
# now run the iteration loop to solve for displacement field
for i in range(niter):
    recon.iterate(i, debug=False)
cat.ra, cat.dec, cat.redshift = recon.get_new_radecz(cat)
# save real-space positions to file
root = output_folder + handle + '_pos'
recon.export_shift_pos(root, rsd_only=True)
print(" ==== Done reconstruction ====\n")
end = time.time()
print("Reconstruction took %0.3f seconds" % (end - start))

# === reload the original redshift-space galaxy positions ==== #
oldcat = GalaxyCatalogue(tracer_file, is_box=False, randoms=False, boss_like=False, special_patchy=True)
wgal = np.empty(oldcat.size, dtype=int)
survey_cuts_logical(wgal, oldcat.veto, oldcat.redshift, z_low_cut, z_high_cut)
wgal = np.asarray(wgal, dtype=bool)
oldcat.cut(wgal)
# ============================================================ #

# === special case: if using previous reconstruction run, load reconstructed catalogue === #
root = output_folder + handle + '_pos'
cat = GalaxyCatalogue(root + '_shift.npy', is_box=False, randoms=False, boss_like=False, special_patchy=False,
                         posn_cols=[0, 1, 2], fkp=0, noz=0, cp=0, systot=1, veto=0)
wgal = np.empty(cat.size, dtype=int)
survey_cuts_logical(wgal, cat.veto, cat.redshift, z_low_cut, z_high_cut)
wgal = np.asarray(wgal, dtype=bool)
cat.cut(wgal)
# ======================================================================================== #

# ====== run voxel void-finding ====== #
start = time.time()
voidcat = VoxelVoids(cat, ran, handle=handle, output_folder=output_folder, is_box=False, omega_m=omega_m,
                     z_min=z_min, z_max=z_max, min_dens_cut=min_dens_cut, use_barycentres=True,
                     void_prefix=void_prefix, find_clusters=False, verbose=True)
voidcat.run_voidfinder()
end = time.time()
print("Voxel voids took %0.3f seconds" % (end - start))

# ======= run ZOBOV ======= #
start = time.time()
void_prefix = handle + '-Zobov-Voids'

voidcat = ZobovVoids(do_tessellation=True, tracer_file=root + '_shift.npy', handle=handle, output_folder=output_folder,
                     is_box=False, boss_like=False, special_patchy=False, posn_cols=[0, 1, 2], omega_m=omega_m,
                     mask_file=mask_file, use_z_wts=True, use_ang_wts=True, z_min=z_min, z_max=z_max,
                     mock_file=mock_file, min_dens_cut=min_dens_cut, void_min_num=5, use_barycentres=True,
                     void_prefix=void_prefix, find_clusters=False)
voidcat.write_box_zobov()
voidcat.write_config()
voidcat.zobov_wrapper(use_mpi=True, zobov_box_div=4, zobov_buffer=0.05, nthreads=nthreads)
voidcat.postprocess_voids()
print(" ==== Finished with ZOBOV-based method ==== ")
end = time.time()
print("ZOBOV took %0.3f seconds\n" % (end - start))
sys.stdout.flush()

'''
# ====== run CUTE on the ZOBOV outputs ====== #

# 1. create the CUTE catalogues
voids = np.loadtxt(output_folder + handle + '-Zobov-Voids_cat.txt')
voids = voids[voids[:, 10] < 2]  # remove edge failures
select = voids[:, 4] > np.median(voids[:, 4])
voids = voids[select]
rweight = ran.get_weights(fkp=0, noz=1, cp=1, syst=1)
void_rand_cat = np.load(void_rand_file)
sweight = oldcat.get_weights(fkp=0, noz=1, cp=1, syst=1)
pweight = cat.get_weights(fkp=0, noz=1, cp=1, syst=1)

# 2. call pycute for the redshift-space xi(s, mu)
output_filename = '/mnt/lustre/nadathur/BOSS_DR12_voidRSD/Patchy_CCFs/zobov-voids/rmu/' \
                  + 'Patchy-%s-%s-%04d-reconZobovVoids-Rcut-x-sGals.txt' % (catalogue, cap, index)
pycute.set_CUTE_parameters(data_filename='', data_filename2='', random_filename='', random_filename2='',
                           output_filename=output_filename, mask_filename='', z_dist_filename='',
                           corr_type='3D_rm_cross', corr_estimator='LS', omega_M=omega_m, omega_L=1-omega_m,
                           w=-1, log_bin=0, dim1_max=120., dim1_nbin=30, dim2_max=1.0, dim2_nbin=80,
                           dim3_min=z_low_cut, dim3_max=z_high_cut, dim3_nbin=1)
void_catalog = pycute.createCatalogFromNumpy_radecz(voids[:, 1], voids[:, 2], voids[:, 3])
gal_random_catalog = pycute.createCatalogFromNumpy_radecz(ran.ra, ran.dec, ran.redshift, rweight)
void_random_catalog = pycute.createCatalogFromNumpy_radecz(void_rand_cat[:, 0], void_rand_cat[:, 1],
                                                           void_rand_cat[:, 2])
s_galaxy_catalog = pycute.createCatalogFromNumpy_radecz(oldcat.ra, oldcat.dec, oldcat.redshift, sweight)
x, y, corr, D1D2, D1R2, D2R1, R1R2 = pycute.runCUTE(paramfile=None, galaxy_catalog=void_catalog,
                                                    galaxy_catalog2=s_galaxy_catalog,
                                                    random_catalog=void_random_catalog,
                                                    random_catalog2=gal_random_catalog)
# overwrite the CUTE output file to control formatting better
r, mu = np.meshgrid(x, y)
r = r.flatten(order='F')
mu = mu.flatten(order='F')
output = np.empty((len(r), 7))
output[:, 0] = r
output[:, 1] = mu
output[:, 2] = corr.flatten()
output[:, 3] = D1D2.flatten()
output[:, 4] = D1R2.flatten()
output[:, 5] = D2R1.flatten()
output[:, 6] = R1R2.flatten()
np.savetxt(output_filename, output, fmt='%0.2f %0.6f %0.6e %0.6e %0.6e %0.6e %0.6e',
           header='Void-galaxy correlation in redshift space\ns[Mpc/h] mu xi(s,mu) D1D2 D1R2 D2R1 R1R2')

# 3. call pycute for the redshift-space xi(s)
output_filename = '/mnt/lustre/nadathur/BOSS_DR12_voidRSD/Patchy_CCFs/zobov-voids/monopole/' \
                  + 'Patchy-%s-%s-%04d-reconZobovVoids-Rcut-x-sGals.txt' % (catalogue, cap, index)
pycute.set_CUTE_parameters(data_filename='', data_filename2='', random_filename='', random_filename2='',
                           output_filename=output_filename, mask_filename='', z_dist_filename='',
                           corr_type='monopole_cross', corr_estimator='LS', omega_M=omega_m, omega_L=1-omega_m,
                           w=-1, log_bin=0, dim1_max=144., dim1_nbin=36, dim2_max=1.0, dim2_nbin=1,
                           dim3_min=z_low_cut, dim3_max=z_high_cut, dim3_nbin=1)
void_catalog = pycute.createCatalogFromNumpy_radecz(voids[:, 1], voids[:, 2], voids[:, 3])
gal_random_catalog = pycute.createCatalogFromNumpy_radecz(ran.ra, ran.dec, ran.redshift, rweight)
void_random_catalog = pycute.createCatalogFromNumpy_radecz(void_rand_cat[:, 0], void_rand_cat[:, 1],
                                                           void_rand_cat[:, 2])
s_galaxy_catalog = pycute.createCatalogFromNumpy_radecz(oldcat.ra, oldcat.dec, oldcat.redshift, sweight)
x, corr, D1D2, D1R2, D2R1, R1R2 = pycute.runCUTE(paramfile=None, galaxy_catalog=void_catalog,
                                                 galaxy_catalog2=s_galaxy_catalog,
                                                 random_catalog=void_random_catalog,
                                                 random_catalog2=gal_random_catalog)
# overwrite the CUTE output file to control formatting better
output = np.empty((len(x), 6))
output[:, 0] = x
output[:, 1] = corr
output[:, 2] = D1D2
output[:, 3] = D1R2
output[:, 4] = D2R1
output[:, 5] = R1R2
np.savetxt(output_filename, output, fmt='%0.2f %0.6e %0.6e %0.6e %0.6e %0.6e',
           header='Void-galaxy correlation in redshift space\ns[Mpc/h] xi(s) D1D2 D1R2 D2R1 R1R2')

# 4. call pycute for the pseudo-real-space xi(r)
output_filename = '/mnt/lustre/nadathur/BOSS_DR12_voidRSD/Patchy_CCFs/zobov-voids/monopole/' \
                  + 'Patchy-%s-%s-%04d-reconZobovVoids-Rcut-x-pGals.txt' % (catalogue, cap, index)
pycute.set_CUTE_parameters(data_filename='', data_filename2='', random_filename='', random_filename2='',
                           output_filename=output_filename, mask_filename='', z_dist_filename='',
                           corr_type='monopole_cross', corr_estimator='LS', omega_M=omega_m, omega_L=1-omega_m,
                           w=-1, log_bin=0, dim1_max=144., dim1_nbin=36, dim2_max=1.0, dim2_nbin=1,
                           dim3_min=z_low_cut, dim3_max=z_high_cut, dim3_nbin=1)
void_catalog = pycute.createCatalogFromNumpy_radecz(voids[:, 1], voids[:, 2], voids[:, 3])
gal_random_catalog = pycute.createCatalogFromNumpy_radecz(ran.ra, ran.dec, ran.redshift, rweight)
void_random_catalog = pycute.createCatalogFromNumpy_radecz(void_rand_cat[:, 0], void_rand_cat[:, 1],
                                                           void_rand_cat[:, 2])
p_galaxy_catalog = pycute.createCatalogFromNumpy_radecz(cat.ra, cat.dec, cat.redshift, pweight)
x, corr, D1D2, D1R2, D2R1, R1R2 = pycute.runCUTE(paramfile=None, galaxy_catalog=void_catalog,
                                                 galaxy_catalog2=p_galaxy_catalog,
                                                 random_catalog=void_random_catalog,
                                                 random_catalog2=gal_random_catalog)
# overwrite the CUTE output file to control formatting better
output = np.empty((len(x), 6))
output[:, 0] = x
output[:, 1] = corr
output[:, 2] = D1D2
output[:, 3] = D1R2
output[:, 4] = D2R1
output[:, 5] = R1R2
np.savetxt(output_filename, output, fmt='%0.2f %0.6e %0.6e %0.6e %0.6e %0.6e',
           header='Void-galaxy correlation in reconstructed real space\nr[Mpc/h] xi(r) D1D2 D1R2 D2R1 R1R2')

# 5. call pycute for the pseudo-real-space xi(r, mu)
output_filename = '/mnt/lustre/nadathur/BOSS_DR12_voidRSD/Patchy_CCFs/zobov-voids/rmu/' \
                  + 'Patchy-%s-%s-%04d-reconZobovVoids-Rcut-x-pGals.txt' % (catalogue, cap, index)
pycute.set_CUTE_parameters(data_filename='', data_filename2='', random_filename='', random_filename2='',
                           output_filename=output_filename, mask_filename='', z_dist_filename='',
                           corr_type='3D_rm_cross', corr_estimator='LS', omega_M=omega_m, omega_L=1-omega_m,
                           w=-1, log_bin=0, dim1_max=120., dim1_nbin=30, dim2_max=1.0, dim2_nbin=80,
                           dim3_min=z_low_cut, dim3_max=z_high_cut, dim3_nbin=1)
void_catalog = pycute.createCatalogFromNumpy_radecz(voids[:, 1], voids[:, 2], voids[:, 3])
gal_random_catalog = pycute.createCatalogFromNumpy_radecz(ran.ra, ran.dec, ran.redshift, rweight)
void_random_catalog = pycute.createCatalogFromNumpy_radecz(void_rand_cat[:, 0], void_rand_cat[:, 1],
                                                           void_rand_cat[:, 2])
p_galaxy_catalog = pycute.createCatalogFromNumpy_radecz(cat.ra, cat.dec, cat.redshift, pweight)
x, y, corr, D1D2, D1R2, D2R1, R1R2 = pycute.runCUTE(paramfile=None, galaxy_catalog=void_catalog,
                                                    galaxy_catalog2=p_galaxy_catalog,
                                                    random_catalog=void_random_catalog,
                                                    random_catalog2=gal_random_catalog)
# overwrite the CUTE output file to control formatting better
r, mu = np.meshgrid(x, y)
r = r.flatten(order='F')
mu = mu.flatten(order='F')
output = np.empty((len(r), 7))
output[:, 0] = r
output[:, 1] = mu
output[:, 2] = corr.flatten()
output[:, 3] = D1D2.flatten()
output[:, 4] = D1R2.flatten()
output[:, 5] = D2R1.flatten()
output[:, 6] = R1R2.flatten()
np.savetxt(output_filename, output, fmt='%0.2f %0.6f %0.6e %0.6e %0.6e %0.6e %0.6e',
           header='Void-galaxy correlation in reconstructed real space\nr[Mpc/h] mu xi(r,mu) D1D2 D1R2 D2R1 R1R2')
'''
