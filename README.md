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

Requirements:
   - python 2.7, but should be compatible with python 3
   - numpy 1.14.5
   - scipy 0.18.1
   - healpy 1.9.0
   - pyfftw 0.10.3
   - astropy 1.0.6
   - SWIG-3.0.12

Some earlier versions of numpy and scipy will fail due to changes in functionality of some methods (numpy.unique 
and scipy.spatial.cKDTree). The code has only been tested with the stated versions of the other packages: other versions
may or may not work!
  
To install and run:
   - edit the python paths in the Makefile, and choose the compiler to use in src/Makefile 
   - in the top-level directory, do 'make clean', then 'make'
   - edit parameters/params.py
   - python revolver.py --par parameters/params.py
   
Acknowledgments:

The reconstruction section of this code was developed from a version written by Julian Bautista and with input from 
Paul Carter, Will Percival and (indirectly) Angela Burden. Hans Winther made several improvements. Some elements of the ZOBOV void-finding section of code were 
developed with Shaun Hotchkiss. 