import argparse
import os
import sys
import imp
import time
import numpy as np
from python_tools.zobov import ZobovVoids
from python_tools.voxelvoids import VoxelVoids
from python_tools.galaxycat import GalaxyCatalogue
from python_tools.recon import Recon
from python_tools.fastmodules import survey_cuts_logical

# ==== Read in settings ==== #
parser = argparse.ArgumentParser(description='options')
parser.add_argument('-p', '--par', dest='par', default="", help='path to parameter file')
args = parser.parse_args()
# read the default parameters
parms = imp.load_source('parameters/default_params.py')
globals().update(vars(parms))
# then override these with the user-provided settings
filename = args.par
if os.access(filename, os.F_OK):
    print('Loading parameters from %s' % filename)
    parms = imp.load_source("name", filename)
else:
    sys.exit('Did not find settings file %s, aborting' % filename)
globals().update(vars(parms))
# ========================= #

# === check output path === #
if not os.access(output_folder, os.F_OK):
    os.makedirs(output_folder)
# ========================= #

# ==== run reconstruction ==== #
if do_recon:
    print('\n ==== Running reconstruction for real-space positions ==== ')

    cat = GalaxyCatalogue(tracer_file, is_box=is_box, box_length=box_length, randoms=False,
                          boss_like=boss_like, special_patchy=special_patchy, posn_cols=posn_cols,
                          fkp=fkp, noz=noz, cp=cp, systot=systot, veto=veto)

    if is_box:
        recon = Recon(cat, ran=cat, is_box=True, box_length=box_length, omega_m=omega_m, bias=bias,
                      f=f, smooth=smooth, nbins=nbins, padding=padding, nthreads=nthreads,
                      verbose=verbose)
    else:
        if not os.access(parms.randoms_file, os.F_OK):
            sys.exit('ERROR: randoms data required for reconstruction but randoms file not provided or not found!' +
                     'Aborting.')

        # initializing randoms: note that in general we assume only FKP weights are provided for randoms
        # this is overridden for special_patchy input format (where veto flags are provided and need to be used)
        ran = GalaxyCatalogue(parms.randoms_file, is_box=False, box_length=parms.box_length, boss_like=parms.boss_like,
                              randoms=True, special_patchy=parms.special_patchy, posn_cols=parms.posn_cols,
                              fkp=parms.fkp, noz=False, cp=False, systot=False, veto=False)

        # perform basic cuts on the data: vetomask and low redshift extent
        wgal = np.empty(cat.size, dtype=int)
        survey_cuts_logical(wgal, cat.veto, cat.redshift, parms.z_low_cut, parms.z_high_cut)
        wgal = np.asarray(wgal, dtype=bool)
        wran = np.empty(ran.size, dtype=int)
        survey_cuts_logical(wran, ran.veto, ran.redshift, parms.z_low_cut, parms.z_high_cut)
        wran = np.asarray(wran, dtype=bool)
        cat.cut(wgal)
        ran.cut(wran)

        recon = Recon(cat, ran, is_box=False, omega_m=parms.omega_m, bias=parms.bias, f=parms.f, smooth=parms.smooth,
                      nbins=parms.nbins, padding=parms.padding, nthreads=parms.nthreads, verbose=parms.verbose)

    start = time.time()
    # now run the iteration loop to solve for displacement field
    for i in range(parms.niter):
        recon.iterate(i, debug=parms.debug)

    # get new ra, dec and redshift for real-space positions
    if not parms.is_box:
        cat.ra, cat.dec, cat.redshift = recon.get_new_radecz(recon.cat)

    # save real-space positions to file
    root = parms.output_folder + parms.handle + '_pos'
    recon.export_shift_pos(root, rsd_only=True)

    print(" ==== Done reconstruction ====\n")
    end = time.time()
    print("Reconstruction took %0.3f seconds" % (end - start))

    # galaxy input for void-finding will now be read from new file with shifted data
    parms.tracer_file = root + '_shift.npy'
# ============================ #

# === run voxel void-finding === #
if parms.run_voxelvoids:

    if parms.do_recon:
        # new tracer file after reconstruction only contains single consolidated weights column
        cat = GalaxyCatalogue(parms.tracer_file, is_box=parms.is_box, box_length=parms.box_length, randoms=False,
                              boss_like=False, special_patchy=False, posn_cols=[0, 1, 2], fkp=0, noz=0, cp=0,
                              systot=1, veto=0)
    else:
        # no reconstruction was performed: use original catalogue file with original weights specification
        cat = GalaxyCatalogue(parms.tracer_file, is_box=parms.is_box, box_length=parms.box_length, randoms=False,
                              boss_like=parms.boss_like, special_patchy=parms.special_patchy, posn_cols=parms.posn_cols,
                              fkp=parms.fkp, noz=parms.noz, cp=parms.cp, systot=parms.systot, veto=parms.veto)

    if not parms.is_box:
        # perform basic cuts on the data: vetomask and low redshift extent
        wgal = np.empty(cat.size, dtype=int)
        survey_cuts_logical(wgal, cat.veto, cat.redshift, parms.z_low_cut, parms.z_high_cut)
        wgal = np.asarray(wgal, dtype=bool)
        cat.cut(wgal)

        # randoms are required
        if not parms.do_recon:
            # randoms were not previously loaded
            if not os.access(parms.randoms_file, os.F_OK):
                sys.exit('ERROR: randoms data required for voxel voids but randoms file not provided or not found!' +
                         'Aborting.')

            # initializing randoms: note that in general we assume only FKP weights are provided for randoms
            # this is overridden for special_patchy input format (where veto flags are provided and need to be used)
            ran = GalaxyCatalogue(parms.randoms_file, is_box=False, box_length=parms.box_length,
                                  boss_like=parms.boss_like, randoms=True, special_patchy=parms.special_patchy,
                                  posn_cols=parms.posn_cols, fkp=parms.fkp, noz=False, cp=False, systot=False,
                                  veto=False)

            # perform basic cuts on the randoms: vetomask and low redshift extent
            wran = np.empty(ran.size, dtype=int)
            survey_cuts_logical(wran, ran.veto, ran.redshift, parms.z_low_cut, parms.z_high_cut)
            wran = np.asarray(wran, dtype=bool)
            ran.cut(wran)

            pre_calc_ran = False
        else:
            # we already have the randoms, and their coordinates have already been calculated
            pre_calc_ran = True
    else:
        # no randoms are required, so set to zero
        ran = 0.
        pre_calc_ran = False  # irrelevant anyway

    # initialize ...
    voidcat = VoxelVoids(cat, ran, handle=parms.handle, output_folder=parms.output_folder, is_box=parms.is_box,
                         box_length=parms.box_length, omega_m=parms.omega_m, z_min=parms.z_min, z_max=parms.z_max,
                         min_dens_cut=parms.min_dens_cut, use_barycentres=parms.use_barycentres,
                         void_prefix=parms.void_prefix, find_clusters=parms.find_clusters,
                         max_dens_cut=parms.max_dens_cut, cluster_prefix=parms.cluster_prefix, verbose=parms.verbose)
    # ... and run the void-finder
    start = time.time()
    voidcat.run_voidfinder()
    end = time.time()
    print("Voxel voids took %0.3f seconds" % (end - start))
# ============================== #

# === run ZOBOV void-finding === #
if parms.run_zobov:

    if parms.run_voxelvoids:
        # need to differentiate the output file names
        parms.void_prefix = parms.void_prefix + '-zobov'
        parms.cluster_prefix = parms.cluster_prefix + '-zobov'

    parms.z_min = max(parms.z_min, parms.z_low_cut)
    parms.z_max = min(parms.z_max, parms.z_high_cut)

    if parms.do_recon:
        voidcat = ZobovVoids(do_tessellation=parms.do_tessellation, tracer_file=parms.tracer_file, handle=parms.handle,
                             output_folder=parms.output_folder, is_box=parms.is_box, boss_like=False,
                             special_patchy=False, posn_cols=[0, 1, 2], box_length=parms.box_length,
                             omega_m=parms.omega_m, mask_file=parms.mask_file, use_z_wts=parms.use_z_wts,
                             use_ang_wts=parms.use_ang_wts, z_min=parms.z_min, z_max=parms.z_max,
                             mock_file=parms.mock_file, mock_dens_ratio=parms.mock_dens_ratio,
                             min_dens_cut=parms.min_dens_cut, void_min_num=parms.void_min_num,
                             use_barycentres=parms.use_barycentres, void_prefix=parms.void_prefix,
                             find_clusters=parms.find_clusters, max_dens_cut=parms.max_dens_cut,
                             cluster_min_num=parms.cluster_min_num, cluster_prefix=parms.cluster_prefix,
                             verbose=parms.verbose)
    else:
        voidcat = ZobovVoids(do_tessellation=parms.do_tessellation, tracer_file=parms.tracer_file, handle=parms.handle,
                             output_folder=parms.output_folder, is_box=parms.is_box, boss_like=parms.boss_like,
                             special_patchy=parms.special_patchy, posn_cols=parms.posn_cols,
                             box_length=parms.box_length, omega_m=parms.omega_m, mask_file=parms.mask_file,
                             use_z_wts=parms.use_z_wts, use_ang_wts=parms.use_ang_wts, z_min=parms.z_min,
                             z_max=parms.z_max, mock_file=parms.mock_file, mock_dens_ratio=parms.mock_dens_ratio,
                             min_dens_cut=parms.min_dens_cut, void_min_num=parms.void_min_num,
                             use_barycentres=parms.use_barycentres, void_prefix=parms.void_prefix,
                             find_clusters=parms.find_clusters, max_dens_cut=parms.max_dens_cut,
                             cluster_min_num=parms.cluster_min_num, cluster_prefix=parms.cluster_prefix,
                             verbose=parms.verbose)

    start = time.time()
    if parms.do_tessellation:
        # write the tracer information to ZOBOV-readable format
        voidcat.write_box_zobov()
        # run ZOBOV
        success = voidcat.zobov_wrapper(use_mpi=parms.use_mpi, zobov_box_div=parms.zobov_box_div,
                                        zobov_buffer=parms.zobov_buffer, nthreads=parms.nthreads)
    else:
        # read the config file from a previous run
        voidcat.read_config()
        success = True

    if success:
        # post-process the raw ZOBOV output to make catalogues
        voidcat.postprocess_voids()
        if voidcat.find_clusters:
            voidcat.postprocess_clusters()
    print(" ==== Finished with ZOBOV-based method ==== ")
    end = time.time()
    print("ZOBOV took %0.3f seconds" % (end - start))
# ============================== #