# all of the following quantities must be assigned a value in this file otherwise an error will result
# however, depending on options chosen, not all quantities may be used in the code
# distance units in this code are calculated in Mpc/h by default

import numpy as np

# ZOBOV run-time options
run_zobov = True    # if True, does tessellation; if False, only post-processes a previous run
use_vozisol = False # set True for survey data or small simulation boxes
zobov_box_div = 2   # no. of subdivisions of ZOBOV box to save memory; only used if use_voz_isol is False
zobov_buffer = 0.1  # fraction of box length overlap between subdivisions; only used if use_voz_isol is False

# file handling
handle = ''         # string; used to identify the sample and set filenames
output_folder = ''  # path to folder where output should be placed

# basic tracer data description
tracer_file = ''    # path to input tracer data file
posn_cols = np.array([0, 1, 2]) # which columns of tracer input array contain position information
is_box = True       # True if tracers cover a cubic simulation box with periodic boundaries; False for survey data
box_length = 2500.  # if is_box is True, the box side length (if is_box is False, ignored); same units as tracer posns

# survey data options - only used if is_box is False
omega_m = 0.308     # cosmology, needed for distance-redshift relation (code assumes flat Universe!)
ang_coords = True   # True if tracer position data is in (ra, dec, redshift) coordinates; False if in Cartesian
observer_posn = np.array([0, 0, 0]) # if ang_coords is False, the Cartesian coords of the observer posn
mask_file = ''      # path to Healpix FITS file containing the survey mask
use_z_wts = True    # if True, densities are weighted by survey n(z) selection function
use_ang_wts = True  # if True, densities are weighted by survey angular completeness function
z_min = 0.43        # minimum redshift extent of survey data
z_max = 0.7         # maximum redshift extent of survey data
mock_file = ''      # path to file containing pre-computed buffer mocks (saves time)
# if mock_file is not specified, new buffer mock positions are computed
mock_dens_ratio = 10.   # if computing buffer mock positions, the ratio of buffer mock densities to galaxy density

# void-finding options
min_dens_cut = 1.0  # void minimum galaxy number density (in units of mean density) reqd to qualify
void_min_num = 5    # minimum number of void member galaxies reqd to qualify (for surveys, set = 5 to be safe)
use_barycentres = False # if True, additionally calculate void barycentre positions
void_prefix = 'Voids'   # prefix used for naming void catalogue files

# 'supercluster'-finding options
find_clusters = False   # set to False unless really needed, finding superclusters is slow
max_dens_cut = 1.0  # cluster maximum galaxy density (in units of mean density) reqd to qualify
cluster_min_n = 5   # minimum number of void member galaxies reqd to qualify
cluster_prefix = 'Clusters' # prefix used for naming supercluster catalogue files