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
    print('Running reconstruction for real-space positions ...')

    cat = GalaxyCatalogue(parms.tracer_file, is_box=parms.is_box, box_length=parms.box_length, randoms=False,
                          boss_like=parms.boss_like, special_patchy=parms.special_patchy, posn_cols=parms.posn_cols,
                          fkp=parms.fkp, noz=parms.noz, cp=parms.cp, systot=parms.systot, veto=parms.veto)

    if parms.is_box:
        recon = Recon(cat, ran=cat, is_box=True, box_length=parms.box_length, omega_m=parms.omega_m, bias=parms.bias,
                      f=parms.f, smooth=parms.smooth, nbins=parms.nbins, padding=parms.padding, nthreads=parms.nthreads)
    else:
        if not os.access(parms.randoms_file, os.F_OK):
            sys.exit('ERROR: randoms data required but not provided! Aborting.')

        # initializing randoms: note that for custom input files (boss_like and special_patchy both False)
        # we have hard-coded the assumption that no noz, cp, or systot weights and no veto information is to be used
        ran = GalaxyCatalogue(parms.randoms_file, is_box=False, box_length=parms.box_length, boss_like=parms.boss_like,
                              randoms=True, special_patchy=parms.special_patchy, posn_cols=parms.posn_cols,
                              fkp=parms.fkp, noz=0, cp=0, systot=0, veto=0)

        # perform basic cuts on the data
        wgal = np.logical_and((cat.veto == 1), (parms.z_min < cat.redshift)&(cat.redshift < parms.z_max))
        wran = np.logical_and((ran.veto == 1), (parms.z_min < ran.redshift)&(ran.redshift < parms.z_max))
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

    # save real-space positions to file
    root = parms.tracer_file.replace('.txt', '').replace('.dat', '').replace('.npy', '')
    recon.export_shift_pos(root, rsd_only=True)

    # initialize the void sample
    sample = VoidSample(run_zobov=parms.run_zobov, tracer_file=root + '_shift.npy', handle=parms.handle,
                        output_folder=parms.output_folder, is_box=parms.is_box, boss_like=False, posn_cols=[0, 1, 2],
                        box_length=parms.box_length, omega_m=parms.omega_m, mask_file=parms.mask_file,
                        use_z_wts=parms.use_z_wts, use_ang_wts=parms.use_ang_wts, z_min=parms.z_min, z_max=parms.z_max,
                        mock_file=parms.mock_file, mock_dens_ratio=parms.mock_dens_ratio,
                        min_dens_cut=parms.min_dens_cut, void_min_num=parms.void_min_num,
                        use_barycentres=parms.use_barycentres, void_prefix=parms.void_prefix,
                        find_clusters=parms.find_clusters, max_dens_cut=parms.max_dens_cut,
                        cluster_min_num=parms.cluster_min_num, cluster_prefix=parms.cluster_prefix)
else:
    sample = VoidSample(run_zobov=parms.run_zobov, tracer_file=parms.tracer_file, handle=parms.handle,
                        output_folder=parms.output_folder, is_box=parms.is_box, boss_like=parms.boss_like,
                        special_patchy=parms.special_patchy, posn_cols=parms.posn_cols, box_length=parms.box_length,
                        omega_m=parms.omega_m, mask_file=parms.mask_file, use_z_wts=parms.use_z_wts,
                        use_ang_wts=parms.use_ang_wts, z_min=parms.z_min, z_max=parms.z_max, mock_file=parms.mock_file,
                        mock_dens_ratio=parms.mock_dens_ratio, min_dens_cut=parms.min_dens_cut,
                        void_min_num=parms.void_min_num, use_barycentres=parms.use_barycentres,
                        void_prefix=parms.void_prefix, find_clusters=parms.find_clusters,
                        max_dens_cut=parms.max_dens_cut, cluster_min_num=parms.cluster_min_num,
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

# post-process the raw ZOBOV output to make catalogues
postprocess_voids(sample)
if sample.find_clusters:
    postprocess_clusters(sample)
