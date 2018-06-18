# REVOLVER

### REal-space VOid Locations from surVEy Reconstruction
(until/unless a better name is found!)

Repository containing code to:

   - reconstruct pseudo-real space tracer positions by subtracting RSD through reconstruction (optional, not yet 
   implemented)
   - apply void-finding algorithm to create catalogue of voids
   
The tracers used will normally be galaxies from a redshift survey, but could also be halos or dark matter particles 
from a simulation box.

Requirements:
   - python 2.7 or python 3
   - numpy 1.11.3
   - scipy 0.18.1
   - healpy 1.9.0
   
To run:
   - in the top-level directory, do 'make clean', then 'make'
   - edit parameters/params.py
   - python revolver.py --par parameters/params.py