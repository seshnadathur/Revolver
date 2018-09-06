# REVOLVER

### REal-space VOid Locations from surVEy Reconstruction

Repository containing code to:

   - reconstruct real space positions from redshift-space tracer data, by subtracting RSD through FFT-based reconstruction (optional)
   - apply void-finding algorithm to create catalogue of voids in these tracers
   
The tracers used will normally be galaxies from a redshift survey, but could also be halos or dark matter 
particles from a simulation box.

Two different void-finding routines are provided. One is based on ZOBOV (Neyrinck 2008, arXiv:0712.3049, 
http://skysrv.pha.jhu.edu/~neyrinck/zobov/zobovhelp.html) and uses Voronoi tessellation of the tracer field to estimate 
the local density, followed by a watershed void-finding step. The other is a voxel-based method, which uses a 
particle-mesh interpolation to estimate the tracer density, and then uses a similar watershed algorithm.

Input data files can be in FITS format (with data fields as for BOSS/eBOSS), or ASCII- or NPY-formatted data arrays.
Read the comments in parameters/params.py for more information about the input file formatting. 

For galaxy survey data, the reconstruction and voxel-based void-finding require a file containing with appropriate 
randoms for the survey. The ZOBOV-based void-finding does not require randoms, but instead requires an appropriate
survey mask file (in HEALPix FITS format) combining the survey geometry, completeness, missing pixels etc. Example masks
for the BOSS DR12 public data releases are provided with this code. If a mask file is not specified, the code will 
attempt to generate an approximate one, but the acccuracy of results will be compromised.

For survey data, pre-computed FKP weights (and other galaxy weight/veto information) are not necessary, but should be 
provided for best performance. 

##### Requirements:
   - python 2.7, but should be compatible with python 3
   - numpy 1.14.5
   - scipy 0.18.1
   - healpy 1.9.0
   - pyfftw 0.10.3
   - astropy 1.0.6
   - cython 0.25.2

Some earlier versions of numpy and scipy will fail due to changes in functionality of some methods (numpy.unique 
and scipy.spatial.cKDTree). The code has only been tested with the stated versions of the other packages: other versions
may or may not work!

##### MPI and parallel processing:
For ZOBOV-based void-finding, there is an option to perform the slow tessellation step in 
separate chunks run in parallel, achieved using MPI. If you have several (i.e. ~10) CPUs available, this will 
be faster than doing it in one shot. If not, single-shot tessellation is usually faster (set use_mpi = False in the 
params.py file). 

Separately, the FFTs used in reconstruction can be performed over multiple CPUs (without using MPI). This is *always*
beneficial, more is faster. Set nthreads in the parameter file to the number of cores available.
 
##### Installation and running:
To install and run:
   - if you don't have MPI compilers/headers, in the Makefile change the line 'make -C src all' to 'make -C src all_nompi' 
   - in the top-level directory, do 'make clean', then 'make'
   - edit parameters/params.py according to instructions given there
   - python revolver.py --par parameters/params.py
   
##### Acknowledgments:

The following people contributed to the concept, development and testing of this code in various ways:
   - Hans Winther
   - Julian Bautista
   - Paul Carter
   - Will Percival
   - Shaun Hotchkiss
