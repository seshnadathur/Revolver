# REVOLVER

### REal-space VOid Locations from surVEy Reconstruction

Repository containing code to:

   - reconstruct real space positions from redshift-space tracer data, by subtracting RSD through FFT-based
   reconstruction (optional)
   - apply void-finding algorithm to create catalogue of voids in these tracers

The tracers used will normally be galaxies from a redshift survey, but could also be halos or dark matter
particles from a simulation box.

Two different void-finding routines are provided. Both operate by identifying minima of the tracer density field, but
differ in the method of reconstructing the tracer density.    
   - The ```ZOBOV``` method is based on Neyrinck 2008 ([arXiv:0712.3049](https://arxiv.org/abs/0712.3049)) and uses
   Voronoi tessellation to estimate the local tracer density from the discrete tracer input.
   - The ```voxel``` method estimates the density field using a particle-mesh interpolation of tracer positions on a
   grid. For survey data this is normalized by the values for the random catalogue characterizing the survey window
   function and selection effects. This method is intended to make more efficient use of fragmented survey data, but
   has not been tested to publication-standard yet – if you are interested in helping with this, please get in touch!

Both methods then use a modified version of the original ```ZOBOV``` watershed algorithm to grow voids around these
minima, and employ additional post-processing and quality control steps to generate the final output catalogues.

#### Requirements:
   - python (default Python3, but for now should still work with 2.7 as well)
   - ```numpy``` 1.14.5
   - ```scipy``` 0.18.1
   - ```healpy``` 1.9.0
   - ```pyfftw``` 0.10.3
   - ```astropy``` 1.0.6
   - ```cython``` 0.25.2

Some earlier versions of ```numpy``` and ```scipy``` will fail due to changes in functionality of some methods
(```numpy.unique``` and ```scipy.spatial.cKDTree```). The code has only been tested with the stated versions of the
other packages: other versions may or may not work!

#### Installation and running:
To install and run:
   - if you don't have MPI compilers/headers, in the [Makefile](Makefile) change the line ```make -C src all``` to
   ```make -C src all_nompi```
   - in the top-level directory, do ```make clean```, then ```make```
   - edit input parameters as required in [parameters/params.py](parameters/params.py) (for full parameter list and
   instructions, see [parameters/default_params.py](parameters/default_params.py))
   - run ```python revolver.py --par parameters/params.py```

##### Input data
Input tracer data can be provided in FITS files, native ```numpy``` .npy files, or plain ASCII files. At a minimum
these files should contain tracer positions, either as (RA, Dec, z) or Cartesian (X, Y, Z). Additional information on
systematics or FKP weights for survey data can also be provided. For FITS files these are assumed to be in data fields
with names
[as for BOSS/eBOSS](https://data.sdss.org/datamodel/files/BOSS_LSS_REDUX/galaxy_DRX_SAMPLE_NS.html "SDSS Data Model").
For ```numpy``` arrays and ASCII files they can be given as additional columns;
see the comments in [parameters/default_params.py](parameters/default_params.py) for more information about the file formatting.

For **uniform simulation data** in a cubic box, *only the tracer data is required*.

For **survey-like data** on the sky, an additional input file containing the randoms characterising the survey
visibility mask (as used for galaxy clustering studies) is required for reconstruction and voxel-based void-finding. If
you are using data from a survey, this file should be easy to obtain. If not (e.g. analysing mock survey data from a
simulation), you can generate your own randoms file. Make sure it has a much higher number density (>=50x) than the
tracers.

For operation on survey-like data on the sky, the ```ZOBOV``` method needs a binary survey mask file (in ```HEALPix```
FITS format) combining the survey geometry, holes, missing pixels etc. Example masks for the BOSS DR12 public data
releases are provided with this code. If no mask file is provided, an approximate one will be generated on the fly from
the survey data, but this may result in less accurate results. If using ```ZOBOV``` void-finding as a standalone (i.e.
no reconstruction and no ```voxel``` void-finding), then no randoms catalogue is required.


##### MPI and parallel processing:
For ```ZOBOV```-based void-finding, there is an option to perform the slow tessellation step in
separate chunks run in parallel, achieved using MPI (set ```use_mpi = True``` in
the [params.py](parameters/params.py) file). If you have many (i.e. >~10) CPUs available, this can
be faster than doing it in one shot. If not, single-shot tessellation is usually faster.

If your data are in a simulation box with periodic boundary conditions, the code will always break the tessellation
into chunks (single-shot is not available in this case). If ```use_mpi``` is False, these chunks will be run serially.
It may then be more efficient to reduce the value of input parameter ```zobov_box_div``` (integer, minimum 2,
default 4).

Separately, the FFTs used in reconstruction can be performed over multiple CPUs (without using MPI). This is *always*
beneficial, more is faster. Set ```nthreads``` in the parameter file to the number of cores available.


##### Troubleshooting:
Log files for various steps are generated and stored in in the ```log/``` subfolder in the output directory. Check
these logs (even if the code appears to have produced successful output):
   - if you see warnings about the number of guard particles, try increasing the value of ```zobov_buffer```
   - if you see warnings about cells with zero volume, check your input data for bad/duplicate tracer positions (these
   will cause the tessellation to fail)


#### Acknowledgments:

The following people contributed to the concept, development and testing of this code in various ways:
   - Hans Winther
   - Slađana Radinovic
   - Julian Bautista
   - Paul Carter
   - Will Percival
   - Shaun Hotchkiss
