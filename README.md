# REVOLVER

### REal-space VOid Locations from surVEy Reconstruction
(until/unless a better name is found!)

Repository containing code to:

   - reconstruct real space positions from redshift-space tracer data, by subtracting RSD through FFT-based reconstruction (optional)
   - apply void-finding algorithm to create catalogue of voids in these tracers
   
The tracers used will normally be galaxies from a redshift survey, but could also be halos or dark matter 
particles from a simulation box.

Input data files can be in FITS format (with BOSS data fields), or ASCII- or NPY-formatted data arrays.
Read the comments in parameters/params.py for more information about the input file formatting. For galaxy
survey data, the reconstruction requires a file with appropriate randoms. Pre-computed FKP weights (and 
other weight/veto information) are not necessary but should be provided for best performance. 

For galaxy survey data, an optional path to an appropriate survey mask file (in HEALPix FITS format) combining 
the survey geometry, completeness, missing pixels etc. should be provided for best performance of the 
void-finding step. Masks for the BOSS DR12 public data releases are provided with this code.

Requirements:
   - python 2.7 or python 3
   - numpy 1.14.5
   - scipy 0.18.1
   - healpy 1.9.0
   - pyfftw 0.10.3

Earlier versions of these packages may lack some required functionality (especially numpy and scipy). 
   
To run:
   - in the top-level directory, do 'make clean', then 'make'
   - edit parameters/params.py
   - python revolver.py --par parameters/params.py