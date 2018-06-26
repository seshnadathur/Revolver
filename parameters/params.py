# all of the following quantities must be assigned a value in this file otherwise an error will result
# however, depending on options chosen, not all quantities may be used in the code
# distance units in this code are calculated in Mpc/h by default

import numpy as np

# ========= file handling options ========= #
handle = ''         # string; used to identify the sample and set filenames
output_folder = ''  # path to folder where output should be placed
# ========================================= #

# ========== cosmology ============ #
omega_m = 0.308  # used for reconstruction and to convert redshifts to distances (assumes flat Universe!)
# ================================= #

# ======= reconstruction options ========== #
do_recon = True     # if False, no reconstruction is performed and other recon options are ignored
nbins = 256     # the number of grid cells per side of the box
padding = 200.  # for survey data, the extra 'padding' for the cubic box, in Mpc/h
smooth = 10.    # smoothing scale in Mpc/h
bias = 2.0      # the linear galaxy/tracer bias value
f = 0.8         # the assumed growth rate at the mean redshift
nthreads = 1    # number of threads used by pyFFTW
niter = 3       # number of iterations in the FFT reconstruction method
# ========================================= #

# ======= input tracer data options =========== #
tracer_file = ''    # path to file with input tracer data; either ASCII or numpy NPY format
is_box = False      # True if tracers cover a cubic simulation box with periodic boundaries; False for survey data
box_length = 1500.  # if is_box, the box side length in Mpc/h; else ignored
posn_cols = np.array([0, 1, 2])  # which columns of tracer input array contain position information
# NOTE: for box data, reconstruction assumes plane-parallel approximation with single l-o-s along the box z-axis!!
# for box data, the next 5 options are ignored:
randoms_file = ''   # path to file containing randoms; required for reconstruction of survey data
boss_like = True    # are data columns formatted as for BOSS DR12 value-added catalogues?
# if not is_box and not boss_like, further information is required:
ang_coords = True   # True if posn_cols contain ra, dec, redshift information; False if they contain x, y, z
observer_posn = np.array([0, 0, 0])  # if ang_coords==False, the Cartesian coords of the observer position
wts_col = 3  # column in tracer input array specifying weights (combination of FKP and fibre collision weights)
# (weights are only required for reconstruction!)
# ============================================= #

# ========== void-finding options ============= #

# -- Tessellation options -- #
run_zobov = True    # if True, does tessellation; if False, only post-processes a previous run
# vozisol code performs entire tessellation in one shot: more memory-intensive, but handles survey data better
use_vozisol = True  # set True for survey data or small simulation boxes
# if not using vozisol, tessellation code divides data into chunks to save memory and requires following two options
zobov_box_div = 2   # no. of subdivisions per box side
zobov_buffer = 0.1  # fraction of box length overlap between subdivisions
# -------------------------- #

# -- survey data handling options -- #
# (if is_box==True, these options are ignored)
z_min = 0.15        # minimum redshift extent of survey data
z_max = 0.43        # maximum redshift extent of survey data
mask_file = ''      # path to Healpix FITS file containing the survey mask (geometry, completeness, missing pixels etc.)
use_z_wts = True    # if True, densities are weighted by survey n(z) selection function
use_ang_wts = True  # if True, densities are weighted by survey angular completeness function
mock_file = ''      # path to file containing pre-computed buffer mocks (saves time)
# if mock_file is not specified, new buffer mock positions are computed
mock_dens_ratio = 10.   # if computing buffer mocks, ratio of buffer mock densities to mean galaxy number density
# ---------------------------------- #

# --- void options ---- #
void_prefix = 'Voids'   # prefix used for naming void catalogue files
min_dens_cut = 1.0  # void minimum galaxy number density (in units of mean density) reqd to qualify
void_min_num = 1    # minimum number of void member galaxies reqd to qualify (for surveys, set = 5 to be safe)
use_barycentres = True  # if True, additionally calculate void barycentre positions
# --------------------- #

# -- additional bonus: 'supercluster' options -- #
find_clusters = False   # set to False unless really needed, finding superclusters is slow
cluster_prefix = 'Clusters'  # prefix used for naming supercluster catalogue files
max_dens_cut = 1.0  # cluster maximum galaxy density (in units of mean density) reqd to qualify
cluster_min_n = 5   # minimum number of void member galaxies reqd to qualify
# ---------------------------------------------- #
# ============================================= #
