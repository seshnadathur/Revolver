import argparse
import os
import sys
import imp
import numpy as np
from python_tools.classes import VoidSample, GalaxyCatalogue
from python_tools.tools import zobov_wrapper, postprocess_voids, postprocess_clusters
from python_tools.recon import Recon

# Read in settings
parser = argparse.ArgumentParser(description='options')
parser.add_argument('--par', dest='par', default="", help='path to parameter file')
args = parser.parse_args()
filename = args.par
if os.access(filename, os.F_OK):
    print('Loading parameters from %s' % filename)
    parms = imp.load_source("name", filename)
else:
    sys.exit('Did not find settings file %s, aborting' % filename)

if not os.access(parms.output_folder, os.F_OK):
    os.makedirs(parms.output_folder)

if parms.do_recon:
    print('Reconstructing real-space tracer positions ...')

    cat = GalaxyCatalogue(parms.tracer_file, boss_like=parms.boss_like, randoms=False, is_box=parms.is_box,
                          box_length=parms.box_length, posn_cols=parms.posn_cols, wts_col=parms.wts_col,
                          ang_coords=parms.ang_coords, obs_posn=parms.observer_posn)

    if parms.is_box:
        recon = Recon(cat, ran=cat, is_box=True, box_length=parms.box_length, bias=parms.bias, f=parms.f,
                      smooth=parms.smooth, nbins=parms.nbins, padding=parms.padding, nthreads=parms.nthreads)
    else:
        if not os.access(parms.randoms_file, os.F_OK):
            sys.exit('ERROR: randoms data required but not provided! Aborting.')

        ran = GalaxyCatalogue(parms.randoms_file, boss_like=parms.boss_like, randoms=True, is_box=False,
                              box_length=parms.box_length, posn_cols=parms.posn_cols, wts_col=parms.wts_col,
                              ang_coords=parms.ang_coords, obs_posn=parms.observer_posn, omega_m=parms.omega_m)

        # perform basic cuts on the data
        if parms.boss_like:
            wgal = np.logical_and((cat.veto == 1), (parms.z_min < cat.redshift < parms.z_max))
            wran = np.logical_and((ran.veto == 1), (parms.z_min < ran.redshift < parms.z_max))
        else:
            wgal = (parms.z_min < cat.redshift < parms.z_max)
            wran = (parms.z_min < ran.redshift < parms.z_max)
        cat.cut(wgal)
        ran.cut(wran)

        recon = Recon(cat, ran, is_box=False, omega_m=parms.omega_m, bias=parms.bias, f=parms.f, smooth=parms.smooth,
                      nbins=parms.nbins, padding=parms.padding, nthreads=parms.nthreads)

    # now run the iteration loop to solve for displacement field
    for i in range(parms.niter):
        recon.iterate(i)

    # apply shifts to obtain real-space positions
    recon.apply_shifts_rsd()
    if not parms.is_box:
        cat.ra, cat.dec, cat.redshift = recon.get_new_radecz(recon.cat)
        ran.ra, ran.dec, ran.redshift = recon.get_new_radecz(recon.ran)

    # save real-space positions to file
    root = parms.tracer_file.replace('.txt', '').replace('.dat', '').replace('.npy', '')
    rand_root = parms.randoms_file.replace('.txt', '').replace('.dat', '').replace('.npy', '')
    recon.export_shift_pos(root, rand_root)

    # initialize the void sample
    sample = VoidSample(run_zobov=parms.run_zobov, tracer_file=root + '_shift.npy', handle=parms.handle,
                        output_folder=parms.output_folder, posn_cols=parms.posn_cols, is_box=parms.is_box,
                        box_length=parms.box_length, omega_m=parms.omega_m, ang_coords=True,
                        mask_file=parms.mask_file, use_z_wts=parms.use_z_wts, use_ang_wts=parms.use_ang_wts,
                        z_min=parms.z_min, z_max=parms.z_max, mock_file=parms.mock_file,
                        mock_dens_ratio=parms.mock_dens_ratio, min_dens_cut=parms.min_dens_cut,
                        void_min_num=parms.void_min_num, use_barycentres=parms.use_barycentres,
                        void_prefix=parms.void_prefix, find_clusters=parms.find_clusters,
                        max_dens_cut=parms.max_dens_cut, cluster_min_n=parms.cluster_min_n,
                        cluster_prefix=parms.cluster_prefix)
else:
    sample = VoidSample(run_zobov=parms.run_zobov, tracer_file=parms.tracer_file, handle=parms.handle,
                        output_folder=parms.output_folder, posn_cols=parms.posn_cols, is_box=parms.is_box,
                        box_length=parms.box_length, omega_m=parms.omega_m, ang_coords=parms.ang_coords,
                        observer_posn=parms.observer_posn, mask_file=parms.mask_file, use_z_wts=parms.use_z_wts,
                        use_ang_wts=parms.use_ang_wts, z_min=parms.z_min, z_max=parms.z_max, mock_file=parms.mock_file,
                        mock_dens_ratio=parms.mock_dens_ratio, min_dens_cut=parms.min_dens_cut,
                        void_min_num=parms.void_min_num, use_barycentres=parms.use_barycentres,
                        void_prefix=parms.void_prefix, find_clusters=parms.find_clusters,
                        max_dens_cut=parms.max_dens_cut, cluster_min_n=parms.cluster_min_n,
                        cluster_prefix=parms.cluster_prefix)

if parms.run_zobov:
    # write the tracer information to ZOBOV-readable format
    sample.write_box_zobov()
    # write a config file
    sample.write_config()
    # run ZOBOV
    zobov_wrapper(sample, use_vozisol=parms.use_vozisol, zobov_box_div=parms.zobov_box_div,
                  zobov_buffer=parms.zobov_buffer)
else:
    # read the config file from a previous run
    sample.read_config()

postprocess_voids(sample)
if sample.find_clusters:
    postprocess_clusters(sample)
