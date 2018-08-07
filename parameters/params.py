# all of the following quantities must be assigned a value in this file otherwise an error will result
# however, depending on options chosen, not all quantities may be used in the code
# distance units in this code are calculated in Mpc/h by default

# ========= file handling options ========= #
handle = ''         # string; used to identify the sample and set filenames
output_folder = ''  # /path/to/folder/ where output should be placed
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
f = 0.8         # the linear growth rate at the mean redshift
nthreads = 1    # number of threads used by pyFFTW
niter = 3       # number of iterations in the FFT reconstruction method
# ========================================= #

# ======= input tracer data options =========== #
tracer_file = ''    # /path/to/file with input data
is_box = False      # True if tracers cover a cubic simulation box with periodic boundaries; False for survey data
box_length = 1500.  # if is_box, the box side length in Mpc/h; else ignored
boss_like = True    # True if the input data file is in FITS format with same data fields as BOSS data
special_patchy = False  # set True if input array is in the special PATCHY format provided by Hector
# if not boss_like or special_patchy, data file must contain array data in ASCII or NPY format
posn_cols = [0, 1, 2]  # columns of tracer input array containing 3D position information
# if is_box == True, these columns should contain x,y,z Cartesian coordinates; otherwise RA, Dec, redshift
# NOTE: for box data, reconstruction assumes plane-parallel approximation with single l-o-s along the box z-axis!!
z_low_cut = 0.4      # lower redshift cut (ignored if not survey)
z_high_cut = 0.8     # higher redshift cut (ignored if not survey)
# minimum cuts required for efficient functioning of reconstruction and voxel void-finding in cases where a tiny
# fraction of data extends to very high or very low redshifts (or even redshifts < 0), as for BOSS data
# ============================================= #

# ======= weights options ========= #
# the following options are used to identify the correct galaxy weights information in ASCII/NPY formatted input arrays
# FKP weights are only used for reconstruction; Voronoi tessellation does not use weights
# if is_box is True, all weights are ignored even if provided
# a special case is hard-coded for use if special_patchy == True, in which case these options are ignored
fkp = True  # are FKP weights (WEIGHT_FKP) provided?
cp = True   # are fibre collision weights (WEIGHT_CP) provided?
noz = False  # are noz weights (WEIGHT_NOZ) provided?
systot = False  # are total systematic weights (WEIGHT_SYSTOT) provided?
veto = True  # is a vetomask column provided? (data is dropped if veto != 1)
# if any of the above weights are provided, they must start from the column immediately after redshift data
# any weights provided MUST be in consecutive columns and with column numbers in the order fkp<cp<noz<systot<veto
# ================================= #

# ====== randoms file ======= #
# for survey data (i.e. when is_box == False) randoms MUST be provided if either do_recon or run_pmvoids are True
randoms_file = ''   # /path/to/file containing randoms data: must be formatted similarly to input data
# NOTE: for randoms, only FKP weights are used, other weights and vetos are ignored (except in the
# special case where special_patchy == True)
# =========================== #

# ========== void-finding choices ============= #
run_voxelvoids = True  # watershed void-finding based on particle-mesh density field interpolation in voxels
run_zobov = False   # watershed void-finding (using ZOBOV) based on Voronoi tessellation
# note that these two options are not mutually exclusive - 2 sets of voids can be produced if desired

z_min = 0.43        # minimum redshift extent
z_max = 0.70        # maximum redshift extent
# voids/cluster catalogues will be cut to have z_min < z < z_max (if survey data): these cuts should be as tight as
# or tighter than the z_low_cut and z_high_cut values specified above

void_prefix = 'Voids'   # prefix used for naming void catalogue files
min_dens_cut = 1.0  # void minimum galaxy number density (in units of mean density) reqd to qualify
use_barycentres = True  # if True, additionally calculate void barycentre positions

# -- additional bonus: 'supercluster' options -- #
find_clusters = False   # if run_zobov is True, this step will be significantly slower
cluster_prefix = 'Clusters'  # prefix used for naming supercluster catalogue files
max_dens_cut = 1.0  # cluster maximum galaxy density (in units of mean density) reqd to qualify
# ============================================= #

# ========== ZOBOV options ============ #
# all ignored if run_zobov = False

# -- Tessellation options -- #
do_tessellation = True    # if True, does tessellation; if False, only post-processes a previous run
# vozisol code performs entire tessellation in one shot: more memory-intensive, but handles survey data better
use_vozisol = True  # set True for survey data or small simulation boxes
# if not using vozisol, tessellation code divides data into chunks to save memory and requires following two options
zobov_box_div = 2   # no. of subdivisions per box side
zobov_buffer = 0.1  # fraction of box length overlap between subdivisions
# -------------------------- #

# -- survey data handling options -- #
# (if is_box==True, these options are ignored)
mask_file = ''      # path to Healpix FITS file containing the survey mask (geometry, completeness, missing pixels etc.)
use_z_wts = True    # if True, densities are weighted by survey n(z) selection function
use_ang_wts = True  # if True, densities are weighted by survey angular completeness function
mock_file = ''      # path to file containing pre-computed buffer mocks (saves time)
# if mock_file is not specified, new buffer mock positions are computed
mock_dens_ratio = 10.   # if computing buffer mocks, ratio of buffer mock densities to mean galaxy number density
# ---------------------------------- #

# --- void options ---- #
void_min_num = 1    # minimum number of void member galaxies reqd to qualify (for surveys, set = 5 to be safe)
# --------------------- #

# -- bonus 'supercluster' options -- #
cluster_min_num = 5   # minimum number of void member galaxies reqd to qualify
# ---------------------------------- #
# ===================================== #
