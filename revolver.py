import argparse
import os
import sys
import imp
import numpy as np
from python_tools.zobov import ZobovVoids
from python_tools.voxelvoids import VoxelVoids
from python_tools.galaxycat import GalaxyCatalogue
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
    print('\n ==== Running reconstruction for real-space positions ==== ')

    cat = GalaxyCatalogue(parms.tracer_file, is_box=parms.is_box, box_length=parms.box_length, randoms=False,
                          boss_like=parms.boss_like, special_patchy=parms.special_patchy, posn_cols=parms.posn_cols,
                          fkp=parms.fkp, noz=parms.noz, cp=parms.cp, systot=parms.systot, veto=parms.veto)

    if parms.is_box:
        recon = Recon(cat, ran=cat, is_box=True, box_length=parms.box_length, omega_m=parms.omega_m, bias=parms.bias,
                      f=parms.f, smooth=parms.smooth, nbins=parms.nbins, padding=parms.padding, nthreads=parms.nthreads)
    else:
        if not os.access(parms.randoms_file, os.F_OK):
            sys.exit('ERROR: randoms data required but randoms file not provided or not found! Aborting.')

        # initializing randoms: note that in general we assume only FKP and systot weights are provided for randoms
        # this is overridden for special input formats (either boss_like and special_patchy are True)
        ran = GalaxyCatalogue(parms.randoms_file, is_box=False, box_length=parms.box_length, boss_like=parms.boss_like,
                              randoms=True, special_patchy=parms.special_patchy, posn_cols=parms.posn_cols,
                              fkp=parms.fkp, noz=0, cp=0, systot=0, veto=0)

        # perform basic cuts on the data (vetomask)
        wgal = cat.veto == 1
        wran = ran.veto == 1
        cat.cut(wgal)
        ran.cut(wran)

        recon = Recon(cat, ran, is_box=False, omega_m=parms.omega_m, bias=parms.bias, f=parms.f, smooth=parms.smooth,
                      nbins=parms.nbins, padding=parms.padding, nthreads=parms.nthreads)

    # now run the iteration loop to solve for displacement field
    for i in range(parms.niter):
        recon.iterate(i)

    # get new ra, dec and redshift for real-space positions
    if not parms.is_box:
        cat.ra, cat.dec, cat.redshift = recon.get_new_radecz(recon.cat)

    # save real-space positions to file
    root = parms.output_folder + parms.handle + '_pos'
    recon.export_shift_pos(root, rsd_only=True)

    print(" ==== Done reconstruction ====\n")

    # galaxy input for void-finding will now be read from new file with shifted data
    parms.tracer_file = root + '_shift.npy'

if parms.run_voxelvoids:

    if parms.do_recon:
        # new tracer file after reconstruction only contains single consolidated weights column
        cat = GalaxyCatalogue(parms.tracer_file, is_box=parms.is_box, box_length=parms.box_length, randoms=False,
                              boss_like=False, special_patchy=False, posn_cols=[0, 1, 2], fkp=0, noz=0, cp=0,
                              systot=1, veto=0)
    else:
        # no reconstruction was performed: use original weights specification
        cat = GalaxyCatalogue(parms.tracer_file, is_box=parms.is_box, box_length=parms.box_length, randoms=False,
                              boss_like=parms.boss_like, special_patchy=parms.special_patchy, posn_cols=parms.posn_cols,
                              fkp=parms.fkp, noz=parms.noz, cp=parms.cp, systot=parms.systot, veto=parms.veto)

    if not parms.is_box:
        # randoms are required
        if not os.access(parms.randoms_file, os.F_OK):
            sys.exit('ERROR: randoms data required but randoms file not provided or not found! Aborting.')

        # initializing randoms: note that in general we assume only FKP and systot weights are provided for randoms
        # this is overridden for special input formats (if either boss_like and special_patchy are True)
        ran = GalaxyCatalogue(parms.randoms_file, is_box=False, box_length=parms.box_length, boss_like=parms.boss_like,
                              randoms=True, special_patchy=parms.special_patchy, posn_cols=parms.posn_cols,
                              fkp=parms.fkp, noz=0, cp=0, systot=1, veto=0)

        # perform basic cuts on the data (vetomask)
        wgal = cat.veto == 1
        wran = ran.veto == 1
        cat.cut(wgal)
        ran.cut(wran)
    else:
        # no randoms are required, so set to zero
        ran = 0.

    # initialize ...
    voidcat = VoxelVoids(cat, ran, handle=parms.handle, output_folder=parms.output_folder, is_box=parms.is_box,
                         box_length=parms.box_length, omega_m=parms.omega_m, min_dens_cut=parms.min_dens_cut,
                         use_barycentres=parms.use_barycentres, void_prefix=parms.void_prefix,
                         find_clusters=parms.find_clusters, max_dens_cut=parms.max_dens_cut,
                         cluster_prefix=parms.cluster_prefix)
    # ... and run the void-finder
    voidcat.run_voidfinder()

if parms.run_zobov:

    if parms.run_voxelvoids:
        # need to differentiate the output file names
        parms.void_prefix = parms.void_prefix + '-zobov'
        parms.cluster_prefix = parms.cluster_prefix + '-zobov'

    # NOTE: Reconstruction and voxel void-finding use all the galaxies for the density estimation
    # (and normalize by the randoms) but ZOBOV cuts survey galaxy data on redshift, parms.z_min < z < parms.z_max
    # This is necessary for the tessellation implementation. ZOBOV does not use the randoms.

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
                             cluster_min_num=parms.cluster_min_num, cluster_prefix=parms.cluster_prefix)
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
                             cluster_min_num=parms.cluster_min_num, cluster_prefix=parms.cluster_prefix)

    if parms.do_tessellation:
        # write the tracer information to ZOBOV-readable format
        voidcat.write_box_zobov()
        # write a config file
        voidcat.write_config()
        # run ZOBOV
        voidcat.zobov_wrapper(use_vozisol=parms.use_vozisol, zobov_box_div=parms.zobov_box_div,
                              zobov_buffer=parms.zobov_buffer)
    else:
        # read the config file from a previous run
        voidcat.read_config()

    # post-process the raw ZOBOV output to make catalogues
    voidcat.postprocess_voids()
    if voidcat.find_clusters:
        voidcat.postprocess_clusters()
