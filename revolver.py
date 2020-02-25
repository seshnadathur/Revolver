import argparse
import os
import sys
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
# read in default parameter values
if sys.version_info.major <= 2:
    import imp
    parms = imp.load_source("name", 'parameters/default_params.py')
elif sys.version_info.major == 3 and sys.version_info.minor <= 4:
    from importlib.machinery import SourceFileLoader
    parms = SourceFileLoader("name", 'parameters/default_params.py').load_module()
else:
    import importlib.util
    spec = importlib.util.spec_from_file_location("name",'parameters/default_params.py')
    parms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parms)

# then override these with the user-provided settings
filename = args.par
if os.access(filename, os.F_OK):
    print('Loading parameters from %s' % filename)
    if sys.version_info.major <= 2:
        user_parms = imp.load_source("name", filename)
    elif sys.version_info.major == 3 and sys.version_info.minor <= 4:
        user_parms = SourceFileLoader("name", filename).load_module()
    else:
        spec = importlib.util.spec_from_file_location("name", filename)
        user_parms = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_parms)
else:
    sys.exit('Did not find settings file %s, aborting' % filename)
for name in vars(user_parms):
    parms.__dict__[name] = user_parms.__dict__[name]
# ========================= #

# === check output path === #
if not os.access(parms.output_folder, os.F_OK):
    os.makedirs(parms.output_folder)
# ========================= #

# ==== run reconstruction ==== #
if parms.do_recon:
    print('\n ==== Running reconstruction for real-space positions ==== ')

    cat = GalaxyCatalogue(parms, randoms=False)

    if parms.is_box:
        recon = Recon(cat, ran=None, parms=parms)
    else:
        if not os.access(parms.random_file, os.F_OK):
            sys.exit('ERROR: randoms data required for reconstruction but randoms file not provided or not found!' +
                     'Aborting.')

        # initializing randoms
        ran = GalaxyCatalogue(parms, randoms=True)

        # perform basic cuts on the data: vetomask and low redshift extent
        wgal = np.empty(cat.size, dtype=int)
        survey_cuts_logical(wgal, cat.veto, cat.redshift, parms.z_low_cut, parms.z_high_cut)
        wgal = np.asarray(wgal, dtype=bool)
        wran = np.empty(ran.size, dtype=int)
        survey_cuts_logical(wran, ran.veto, ran.redshift, parms.z_low_cut, parms.z_high_cut)
        wran = np.asarray(wran, dtype=bool)
        cat.cut(wgal)
        ran.cut(wran)

        recon = Recon(cat, ran, parms)

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
    # adjust input parameters for subsequent steps to match shifted tracer file
    parms.tracer_file_type = 2
    # following lines set to match recon output; ignored for box data anyway
    parms.weights_model = 1
    parms.fkp = False; parms.cp = False; parms.noz = False; parms.veto = False
    parms.systot = True
    parms.comp = True
# ============================ #

# === run voxel void-finding === #
if parms.run_voxelvoids:

    # read in input catalogue again in case it was changed by reconstruction
    cat = GalaxyCatalogue(parms, randoms=False)

    if not parms.is_box:
        # perform basic cuts on the data: vetomask and low redshift extent
        wgal = np.empty(cat.size, dtype=int)
        survey_cuts_logical(wgal, cat.veto, cat.redshift, parms.z_low_cut, parms.z_high_cut)
        wgal = np.asarray(wgal, dtype=bool)
        cat.cut(wgal)

        # randoms are required
        if not parms.do_recon:
            # randoms were not previously loaded
            if not os.access(parms.random_file, os.F_OK):
                sys.exit('ERROR: randoms data required for voxel voids but randoms file not provided or not found!' +
                         'Aborting.')

            # initializing randoms: note that in general we assume only FKP weights are provided for randoms
            # this is overridden for special_patchy input format (where veto flags are provided and need to be used)
            ran = GalaxyCatalogue(parms, randoms=True)

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
        ran = None
        pre_calc_ran = False  # irrelevant anyway

    # initialize ...
    voidcat = VoxelVoids(cat, ran, parms)
    # ... and run the void-finder
    start = time.time()
    voidcat.run_voidfinder()
    end = time.time()
    print("Voxel voids took %0.3f seconds" % (end - start))
# ============================== #

# === run ZOBOV void-finding === #
if parms.run_zobov:

    parms.z_min = max(parms.z_min, parms.z_low_cut)
    parms.z_max = min(parms.z_max, parms.z_high_cut)

    if parms.do_recon:
        voidcat = ZobovVoids(parms)
    else:
        voidcat = ZobovVoids(parms)

    start = time.time()
    if parms.do_tessellation:
        # write the tracer information to ZOBOV-readable format
        voidcat.write_box_zobov()
        # run ZOBOV
        success = voidcat.zobov_wrapper()
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
