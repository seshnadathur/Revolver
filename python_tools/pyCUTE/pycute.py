import CUTEPython as cute
import numpy as np

"""
 Run CUTE from within Python using CUTEPython
 
 Allows to load a random_catalog and pass it to CUTE to avoid 
 having to read this from file every time in case of multiple calls.
 
 Input:
    * Filename of CUTE parameterfile (standard CUTE format). If not provided
      then we assume the parameters have been set by first calling set_CUTE_parameters(...)
    * Catalog(s) in CUTE format
      If not provided then the catalogs is read from file inside CUTE
      The catalog can be created by calling readCatalog(filename) 

 Output:
    * For corr_type = 0 [x, corr] where [x] is [Dz]
      For corr_type = 1 [x, corr] where [x] is [theta]
      For corr_type = 2 [x, corr] where [x] is [r]
      For corr_type = 3 [x, y, corr] where [x], [y] is (rt,rl)
      For corr_type = 4 [x, y, corr] where [x], [y] is (r,mu)
      For corr_type = 5 [x, y, z, corr] where [x], [y], [z] is (z_mean,Dz,theta)
      For corr_type = 6 [x, y, z, corr] where [x], [y], [z] is (z1,z2,theta)
      For corr_type = 7 [x, corr] where [x] is [r]
      For corr_type = 8 [x, y, corr] where [x], [y] is (r,mu)
      See the CUTE readme or src/io.c for more info
      [corr] is the correlation function (radial, angular, monopole etc.).

 If paircounts are outputted then we also output (after corr)
 [ DD, DR, RR ] for corr_type < 7 and [ D1D2, D1R2, D2R1, R1R2 ] for corr_type = 7,8

 We can also fetch paircount from [result] if wanted, see CUTEPython.i for 
 availiable functions.

 Python has responsibillity for the memory of [result] and [catalog] (if 
 they are passed to CUTE). Deallocation should be handled automatically, but not tested
 so to be sure of no memory leaks we can always call cute.free_result_struct(result) 
 and cute.free_Catalog(random_catalog) 

 MPI support is implemented, but not well tested. To run with MPI
 run as OMP_NUM_THREADS=1 mpirun -np N python2.7 script.py and remember 
 to call cute.finalize_mpi() in the end of the script.

"""
def runCUTE(paramfile = None, galaxy_catalog = None, galaxy_catalog2 = None, random_catalog = None, random_catalog2 = None):
  
  if(paramfile is not None): 
    cute.read_run_params(paramfile)
  
  # Check for errors in parameters
  err = cute.verify_parameters()
  if(err > 0): return

  result = cute.make_empty_result_struct()
  cute.runCUTE(galaxy_catalog,galaxy_catalog2,random_catalog,random_catalog2,result)

  # Fetch results
  corr_type_oneD = [0,1,2,7]; corr_type_twoD   = [3,4,8]; corr_type_threeD = [5,6]
  if(cute.get_corr_type() in corr_type_oneD): 
    nx = result.get_nx()
    x    = np.array([result.get_x(i)    for i in range(nx)])
    corr = np.array([result.get_corr(i) for i in range(nx)])
    
    #===============================================
    # Fetch paircounts
    #===============================================
    if(cute.get_corr_type() == 7):
      D1D2   = np.array([result.get_D1D2(i) for i in range(nx)])
      D1R2   = np.array([result.get_D1R2(i) for i in range(nx)])
      D2R1   = np.array([result.get_D2R1(i) for i in range(nx)])
      R1R2   = np.array([result.get_R1R2(i) for i in range(nx)])
      return x, corr, D1D2, D1R2, D2R1, R1R2
    else:
      DD = np.array([result.get_D1D1(i) for i in range(nx)])
      DR = np.array([result.get_D1R1(i) for i in range(nx)])
      RR = np.array([result.get_R1R1(i) for i in range(nx)])
      return x, corr, DD, DR, RR
    #===============================================
    
    return x, corr
  elif(cute.get_corr_type() in corr_type_twoD):
    nx = result.get_nx()
    ny = result.get_ny()
    x = np.array([result.get_x(i) for i in range(nx)])
    y = np.array([result.get_y(i) for i in range(ny)])
    corr = np.zeros((nx,ny),dtype='float64')
    for i in range(nx):
      for j in range(ny):
        corr[i][j] = result.get_corr(j + ny * i)

    #===============================================
    # Fetch paircounts
    #===============================================
    if(cute.get_corr_type() == 8):
      D1D2 = np.zeros((nx,ny),dtype='float64')
      D1R2 = np.zeros((nx,ny),dtype='float64')
      D2R1 = np.zeros((nx,ny),dtype='float64')
      R1R2 = np.zeros((nx,ny),dtype='float64')
      for i in range(nx):
        for j in range(ny):
          D1D2[i][j] = result.get_D1D2(j + ny * i)
          D1R2[i][j] = result.get_D1R2(j + ny * i)
          D2R1[i][j] = result.get_D2R1(j + ny * i)
          R1R2[i][j] = result.get_R1R2(j + ny * i)
      return x, y, corr, D1D2, D1R2, D2R1, R1R2
    else:
      DD = np.zeros((nx,ny),dtype='float64')
      DR = np.zeros((nx,ny),dtype='float64')
      RR = np.zeros((nx,ny),dtype='float64')
      for i in range(nx):
        for j in range(ny):
          DD[i][j] = result.get_D1D1(j + ny * i)
          DR[i][j] = result.get_D1R1(j + ny * i)
          RR[i][j] = result.get_R1R1(j + ny * i)
    #===============================================
    
    return x, y, corr
  elif(cute.get_corr_type() in corr_type_threeD):
    nx = result.get_nx()
    ny = result.get_ny()
    nz = result.get_nz()
    x = np.array([result.get_x(i) for i in range(nx)])
    y = np.array([result.get_y(i) for i in range(ny)])
    z = np.array([result.get_z(i) for i in range(nz)])
    corr = np.zeros((nx,ny,nz),dtype='float64')
    for i in range(nx):
      for j in range(ny):
        for k in range(nz):
          corr[i][j][k] = result.get_corr(k + nz * (j + ny * i))
    
    #===============================================
    # Fetch paircounts
    #===============================================
    DD   = np.zeros((nx,ny,nz),dtype='float64')
    DR   = np.zeros((nx,ny,nz),dtype='float64')
    RR   = np.zeros((nx,ny,nz),dtype='float64')
    for i in range(nx):
      for j in range(ny):
        for k in range(nz):
          DD[i][j][k]   = result.get_D1D1(k + nz * (j + ny * i))
          DR[i][j][k]   = result.get_D1R1(k + nz * (j + ny * i))
          RR[i][j][k]   = result.get_R1R1(k + nz * (j + ny * i))
    return x, y, z, corr, DD, DR, RR
    #===============================================
    
    return x, y, z, corr
  else:
    return None

"""
 Set parameters in CUTE either by providing a parameterfile or 
 by setting them directly
 If paramfile is not None then we read the parameterfile
 NB: if we read the parameterfile and there are critical errors then
 C calls exit() (standard in CUTE). If we set them one by one then 
 errors will be shown, but no call to exit()
"""
def set_CUTE_parameters(
    paramfile=None,
    data_filename="",
    data_filename2="",
    random_filename="",
    random_filename2="",
    reuse_randoms=0,
    num_lines="all",
    input_format=2,
    mask_filename="",
    z_dist_filename="",
    output_filename="",
    corr_estimator="LS",
    corr_type="monopole",
    np_rand_fact=1,
    omega_M=0.3,
    omega_L=0.7,
    w=-1.0,
    radial_aperture=1.0,
    dim1_max=1,
    dim2_max=1,
    dim3_max=1,
    dim3_min=0,
    dim1_nbin=1,
    dim2_nbin=1,
    dim3_nbin=1,
    log_bin=0,
    n_logint=10,
    use_pm=1,
    n_pix_sph=2048):

  if(paramfile is not None):
    cute.read_run_params(paramfile)
    return

  cute.initialize_binner();

  # Strings
  cute.set_data_filename(data_filename)
  cute.set_data_filename2(data_filename2)
  cute.set_random_filename(random_filename);
  cute.set_random_filename2(random_filename2);
  cute.set_output_filename(output_filename)
  cute.set_corr_type(corr_type)
  cute.set_num_lines(num_lines)
  cute.set_mask_filename(mask_filename)
  cute.set_z_dist_filename(z_dist_filename)
  cute.set_corr_estimator(corr_estimator)

  # Doubles
  cute.set_omega_M(omega_M)
  cute.set_omega_L(omega_L)
  cute.set_w(w)
  cute.set_radial_aperture(radial_aperture)
  cute.set_dim1_max(dim1_max)
  cute.set_dim2_max(dim2_max)
  cute.set_dim3_max(dim3_max)
  cute.set_dim3_min(dim3_min)

  # Integers
  cute.set_input_format(input_format)
  cute.set_n_logint(n_logint)
  cute.set_dim1_nbin(dim1_nbin)
  cute.set_dim2_nbin(dim2_nbin)
  cute.set_dim3_nbin(dim3_nbin)
  cute.set_log_bin(log_bin)
  cute.set_np_rand_fact(np_rand_fact)
  cute.set_reuse_randoms(reuse_randoms)
  cute.set_use_pm(use_pm)
  cute.set_n_pix_sph(n_pix_sph)

  # Check if parameters are good 
  # err = cute.verify_parameters()

"""
 If we run with MPI then we need to Finalize MPI within C
"""
def Finalize():
  cute.finalize_mpi()

"""
 Free up memory of a C catalog
"""
def freeCatalog(catalog):
  cute.free_Catalog(catalog)

"""
 Use CUTE to read a catalog an store it in C format
 If paramfile = None we assume the parameters have already been set in CUTE by set_CUTE_parameters
"""
def readCatalog(paramfile, filename):
  if(paramfile is not None):
    cute.read_run_params(paramfile)
  return cute.read_Catalog(filename)

"""
 A galaxy catalog in Python format
 Use convert_to_python to convert a catalog from C format to Python format
"""
class PyCatalog:
  
  def __init__(self, with_weights = False, ccatalog = None):
    if(ccatalog is not None):
      self.convert_to_python(with_weights,ccatalog)

  n = 0
  with_weights = False
  phi = np.zeros(n)
  cth = np.zeros(n)
  red = np.zeros(n)
  weight = np.zeros(n)

  @classmethod
  def get_n(self):
    return self.n
  @classmethod
  def get_phi(self):
    return self.phi
  @classmethod
  def set_phi(self,phi):
    self.n = phi.size
    self.phi = phi
  @classmethod
  def get_cth(self):
    return self.cth
  @classmethod
  def set_cth(self,cth):
    self.n = cth.size
    self.cth = cth
  @classmethod
  def get_red(self):
    return self.red
  @classmethod
  def set_red(self,red):
    self.n = red.size
    self.red = red
  @classmethod
  def get_weight(self):
    if(self.with_weights):
      return self.weight
    else:
      return None
  @classmethod
  def set_weight(self,weight):
    self.n = weight.size
    self.weight = weight

  @classmethod
  def convert_to_python(self, with_weights, ccatalog):
    self.with_weights = with_weights
    self.n       = ccatalog.get_np()
    self.phi     = np.array([ccatalog.get_phi(i) for i in range(self.n)])
    self.cth     = np.array([ccatalog.get_cth(i) for i in range(self.n)])
    self.red     = np.array([ccatalog.get_red(i) for i in range(self.n)])
    if(self.with_weights):
      self.weight = np.array([ccatalog.get_weight(i) for i in range(self.n)])

"""
  Create a CUTE galaxy catalog in C format from numpy arrays of phi, cos(theta), redshift and weight
"""
def createCatalogFromNumpy_phicthz(phi,cth,red,weight = None):
  ok = True
  if (type(phi)     is not np.ndarray): ok = False
  if (type(cth)     is not np.ndarray): ok = False
  if (type(red)     is not np.ndarray): ok = False
  if(weight is not None):
    if (type(weight) is not np.ndarray): ok = False
  if( not ok ):
    print "Error: all input needs to be numpy double arrays (weight can be None)"
    return None
  if(weight is None):
    tmp = np.ones(phi.size,dtype='float64')
    return cute.create_catalog_from_numpy(phi,cth,red,tmp)
  return cute.create_catalog_from_numpy(phi,cth,red,weight)

"""
  Create a CUTE galaxy catalog in C format from numpy arrays of RA, Dec, redshift, and weight (angles in degrees)
"""
def createCatalogFromNumpy_radecz(ra, dec, red, weight=None):
  ok = True
  if (type(ra)     is not np.ndarray): ok = False
  if (type(dec)     is not np.ndarray): ok = False
  if (type(red)     is not np.ndarray): ok = False
  if(weight is not None):
    if (type(weight) is not np.ndarray): ok = False
  if( not ok ):
    print "Error: all input needs to be numpy double arrays (weight can be None)"
    return None
  phi = np.deg2rad(ra)
  cth = np.cos(np.deg2rad(90 - dec))
  if(weight is None):
    tmp = np.ones(ra.size,dtype='float64')
    return cute.create_catalog_from_numpy(phi,cth,red,tmp)
  return cute.create_catalog_from_numpy(phi,cth,red,weight)

"""
 Print CUTE parameters
"""
def print_param():
  cute.print_parameters()
