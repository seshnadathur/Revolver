# This file explains the input parameters for the code and assigns reasonable default values to each of them.
# Don't alter this file – instead use a separate parameters file to overwrite any input values you wish to change
# Note: distance units in this code are calculated in Mpc/h by default

# ======= runtime options ======== #
verbose = False  # True for more informative output statements
debug = False    # True for output checks during reconstruction
nthreads = 4     # set to the number of CPUs available, more is better
# ================================ #

# ========= file handling options ========= #
handle = 'default'  # string to identify the run; used to set filenames
output_folder = ''   # /path/to/folder/ where output should be placed
# ========================================= #

# ========== cosmology ============ #
omega_m = 0.31  # used for reconstruction and to convert redshifts to distances (assumes flat Universe!)
# ================================= #

# ======= reconstruction options ========== #
do_recon = True     # if False, no reconstruction is performed and other recon options are ignored
nbins = 512     # the number of grid cells per side of the box
padding = 200.  # for survey data, the extra 'padding' for the cubic box, in Mpc/h
smooth = 10.    # smoothing scale in Mpc/h
bias = 2        # the linear galaxy/tracer bias value
f = 0.78        # the linear growth rate at the mean redshift
niter = 3       # number of iterations in the FFT reconstruction method, 3 is sufficient
# NOTE: for box data, reconstruction assumes plane-parallel approximation with single l-o-s along the box z-axis!!
# ========================================= #

# ======= input galaxy/tracer data options =========== #
tracer_file = ''     # /path/to/file with input data
tracer_file_type = 1  # 1 for FITS file, 2 for array in numpy pickle format (.npy), 3 for array in ASCII format
# NOTE: for FITS files, the tracer coordinates should be specified using appropriate field names
# current options are 'RA', 'DEC' and 'Z' for survey-like data on the sky, or 'X', 'Y', 'Z' for simulation boxes
# For array data (tracer_file_type = 2 or 3), specify which columns of the array contain the tracer coordinates
tracer_posn_cols = [0, 1, 2]  # columns of tracer input array containing 3D position information
# specify data type:
is_box = False       # True for cubic simulation box with periodic boundaries; False for survey-like data on the sky
box_length = 1500.   # if is_box, the box side length in Mpc/h; else ignored
# the following cuts useful for more efficient reconstruction and voxel void-finding for BOSS CMASS data, where a tiny
# fraction of data extends to very high or very low redshifts (and even redshifts < 0)
z_low_cut = 0.4      # lower redshift cut (ignored if not survey)
z_high_cut = 0.73    # higher redshift cut (ignored if not survey)
# what is the model for applying weights? 1 = like BOSS; 2 = like eBOSS; 3 = like joint BOSS+eBOSS LRG sample
# (unfortunately things change as surveys progress)
weights_model = 1
# 1. For FITS files (tracer_file_type = 1) weights are automatically extracted using field names based on BOSS/eBOSS data
# model (https://data.sdss.org/datamodel/files/BOSS_LSS_REDUX/galaxy_DRX_SAMPLE_NS.html)
# 2. for simulation box data (is_box = True) all weights information is ignored as assumed uniform
# -----------
# Most users will only use weights with survey data in FITS files
# If for some reason you have survey(-like) data in array format (tracer_file_type = 2 or 3), specify what info is
# present in the file using following flags (FKP, close-pair, missing redshift, total systematics, veto flag,
# completeness). Weights MUST be given in consecutive columns starting immediately after the column with redshifts,
# and with column numbers in the order fkp<cp<noz<systot<veto<comp
fkp = False     # FKP weights (used for reconstruction when n(z) is not constant)
cp = False      # close-pair or fibre collision weights
noz = False     # missing redshift / redshift failure weights
systot = False  # total systematic weights
veto = False    # veto mask (if present, galaxies with veto!=1 are discarded)
comp = False    # sector completeness
# ============================================= #

# ====== input randoms options ======= #
# for survey-like data, randoms characterize the window function and MUST be provided for reconstruction and
# voxel void-finding (not necessary for ZOBOV alone)
random_file = ''   # /path/to/file containing randoms data
random_file_type = 1  # 1 for FITS file, 2 for array in numpy pickle format (.npy), 3 for array in ASCII format
# if random_file_type = 2 or 3, specify which columns of the array contain the (RA, Dec, redshift) coordinates
random_posn_cols = [0, 1, 2]
# if galaxy data has FKP weights, randoms are assumed to have FKP weights too
# all other galaxy weights are ignored for randoms
# =========================== #

# ========== void-finding choices ============= #
run_voxelvoids = True  # watershed void-finding based on particle-mesh density field interpolation in voxels
run_zobov = True   # watershed void-finding (using ZOBOV) based on Voronoi tessellation
# these two options are not mutually exclusive - 2 sets of voids can be produced if desired

# for survey-like data only: set redshift limits
z_min = 0.43        # minimum redshift extent of the data
z_max = 0.70        # maximum redshift extent of the data
# these limits are used to prune output void catalogues and to terminate the tessellation in ZOBOV
# NOTES: 1. always set z_min >= z_low_cut and z_max <= z_high_cut
# 2. do not set z_min < minimum redshift of the data or z_max > max redshift – will cause tessellation leakage!

void_prefix = 'Voids'   # string used in naming void output files
min_dens_cut = 1.0  # void minimum galaxy number density (in units of mean density) reqd to qualify
use_barycentres = True  # if True, additionally calculate void barycentre positions

# -- additional bonus: 'supercluster' options -- #
find_clusters = False   # if run_zobov is True, this step will be significantly slower
cluster_prefix = 'Clusters'  # prefix used for naming supercluster catalogue files
max_dens_cut = 1.0  # cluster maximum galaxy density (in units of mean density) reqd to qualify
# ============================================= #

# ========== ZOBOV-specific options ============ #
# all ignored if run_zobov = False

# -- Tessellation options -- #
do_tessellation = True    # if True, does tessellation; if False, only post-processes previous run with same handle
# guards are used to stabilise the tessellation for surveys; increase this number if the survey volume is a
# small fraction of that of the smallest cube required to fully enclose it
guard_nums = 30     
use_mpi = False
# use MPI if you have several (~10) CPUs available, otherwise it is generally faster to run without
zobov_box_div = 2   # partition tessellation job into (zobov_box_div)^3 chunks (run in parallel, if using MPI)
zobov_buffer = 0.08  # fraction of box length overlap between sub-boxes
# -------------------------- #

# -- survey data handling options -- #
# (if is_box==True, these options are ignored)
mask_file = ''       # path to Healpix FITS file containing the survey mask
use_z_wts = True     # set True if survey n(z) is not uniform
use_syst_wts = True  # set True to use galaxy systematic weights
use_completeness_wts = True  # set True to account for angular variations in survey completeness
mock_file = ''       # path to file containing pre-computed buffer mocks (saves time)
# if mock_file is not specified, new buffer mock positions are computed
mock_dens_ratio = 10.   # if computing buffer mocks, ratio of buffer mock densities to mean galaxy number density
# ---------------------------------- #

# --- void options ---- #
void_min_num = 5    # minimum number of void member galaxies reqd to qualify (for surveys, set = 5 to be conservative)
# --------------------- #

# -- bonus 'supercluster' options -- #
cluster_min_num = 5   # minimum number of void member galaxies reqd to qualify
# ---------------------------------- #
# ===================================== #
