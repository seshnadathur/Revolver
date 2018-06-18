import argparse
from python_tools import *

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

sample = Sample(run_zobov=parms.run_zobov, tracer_file=parms.tracer_file, handle=parms.handle,
                output_folder=parms.output_folder, posn_cols=parms.posn_cols, is_box=parms.is_box, 
                box_length=parms.box_length, omega_m=parms.omega_m, ang_coords=parms.ang_coords, 
                observer_posn=parms.observer_posn, mask_file=parms.mask_file, use_z_wts=parms.use_z_wts, 
                use_ang_wts=parms.use_ang_wts, z_min=parms.z_min, z_max=parms.z_max, mock_file=parms.mock_file, 
                mock_dens_ratio=parms.mock_dens_ratio, min_dens_cut=parms.min_dens_cut, void_min_num=parms.void_min_num,
                use_barycentres=parms.use_barycentres, void_prefix=parms.void_prefix, find_clusters=parms.find_clusters,
                max_dens_cut=parms.max_dens_cut, cluster_min_n=parms.cluster_min_n, cluster_prefix=parms.cluster_prefix)
# initialization of sample takes care of all of the following necessary steps:
# 1. setting file paths 2. loading tracer information 3. setting cosmology (if reqd)
# 4. converting coordinates to standard form (if reqd) 5. reading/generating survey
# mask (if reqd) 6. generating redshift weights (if reqd) 7. setting structure-finding
# options 8. (if parms.run_zobov is True and parms.is_box is False) reading/generating
# buffer mocks

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
