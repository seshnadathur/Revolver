import numpy as np
import os
import sys
import glob
import subprocess
import random
import healpy as hp
from scipy.integrate import quad
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.signal import savgol_filter
from scipy.optimize import brentq
from scipy.spatial import cKDTree

LIGHT_SPEED = 299792.458

#-------select elements of a vector between two limits---------#
#--------------------------------------------------------------#
def test_bin(vector,lower_lim,upper_lim):
	"""Finds the elements of a vector lying between two limits

	Arguments:
	  vector -- 1d array of values
	  lower_lim -- lower limit value (assumed closed limit)
	  upper_lim -- upper limit value (assumed open limit)

	Returns a boolean mask specifying the required elements
	"""
	return np.logical_and(vector[:]>=lower_lim,vector[:]<upper_lim) 
#--------------------------------------------------------------#

#-------find mean value in a bin--------#
#---------------------------------------#	
def bin_mean_val(xmin,xmax,xvector,values):
	"""Finds the mean value and std. error of elements of a vector 
	corresponding to elements of a second vector lying between two limits

	Arguments:
	  xmin -- lower limit value (assumed closed limit)
	  xmax -- upper limit value (assumed open limit)
	  xvector -- array whose values are to be tested against the limits
	  values -- arrays whose corresponding elements are to be averaged

	Returns mean, stderr of required values of values
	"""	
	bin_vals = values[test_bin(xvector,xmin,xmax)]
	if len(bin_vals)>0:
		return np.mean(bin_vals), np.std(bin_vals)/np.sqrt(len(bin_vals))
	else:
		return np.nan, np.nan
#---------------------------------------#

#--------bin up one vector according to values of another------#
#--------------------------------------------------------------#
def binner(xvector,values,nbins):
	"""Finds bin means and std devs for elements in one vector binned up
	according to values of another

	Arguments:
	  xvector -- array which determines the binning
	  values -- array of elements whose binned means and std devs are required
	  nbins -- number of bins (uniform bin widths), or array of user-defined bin edges

	Returns:
	  xedges -- array of values defining bin edges in xvector
	  masked_bin_means -- array of binned means (masked in case of bins which contain no elements)
	  masked_bin_errs -- array of binned stddevs (also masked)
	"""
	H, xedges = np.histogram(xvector,bins=nbins)
	bin_means, bin_err = np.empty(H.shape),np.empty(H.shape)
	for i in range(H.shape[0]):
			bin_means[i],bin_err[i] = bin_mean_val(xedges[i],xedges[i+1],xvector,values)
	masked_bin_means = np.ma.masked_where(np.isnan(bin_means),bin_means)
	masked_bin_err = np.ma.masked_where(np.isnan(bin_err),bin_err)
	return xedges, masked_bin_means, masked_bin_err
#--------------------------------------------------------------#

#-------comoving distance to redshift z ------#
#---------------------------------------------#
def comovr (z = 0, OmegaM = 0.308):
	"""Calculates the comoving distance to redshift z assuming a flat Universe.

	Arguments:
	   z -- desired redshift, default 0
	   OmegaM -- matter density ratio, default 0.27

	Returns r_c(z)
	"""
	d = quad(lambda x: 0.01*LIGHT_SPEED / np.sqrt(OmegaM*(1+x)**3 + 1 - OmegaM), 0, z)
	return d[0]
#---------------------------------------------#

#--------comoving volume between two redshifts--------#
def full_volume_z(z=[0,1], OmegaM=0.27):
	"""Calculates the comoving volume between two input redshifts, over the full sky. Assumes a flat Universe.

	Arguments:
	  z -- array of shape (2,) containing start and end redshifts, default [0,1]
	  OmegaM -- matter density ratio, default 0.27

	Returns volume
	"""
	r1 = comovr(z[0],OmegaM)
	r2 = comovr(z[1],OmegaM)
	volume = 4*np.pi*quad(lambda x: x**2, r1, r2)[0]
	return volume
#-----------------------------------------------------#

#--------measure galaxy number density in redshift bins---------#
#---------------------------------------------------------------#
def generate_selFn(sample,nbins=15):
	"""Measures the galaxy number density in equal volume redshift bins and writes the values to file.

	Arguments:
	  sample -- object of type Sample
	  nbins -- number of bins
	"""
	print "Determining survey redshift selection function ..."
	#first calculate redshift values to make equal volume bins
	zsteps = np.linspace(sample.zMin,sample.zMax,nbins+1)
	Vstep = full_volume_z([sample.zMin,sample.zMax],sample.OmegaM)/nbins
	for i in range(1,nbins):
		zsteps[i] = brentq(lambda x: full_volume_z([zsteps[i-1],x],sample.OmegaM)-Vstep,0,1.0)

	#now load up the galaxy data - 6th column should contain redshifts
	if not os.access(sample.tracerFile,os.F_OK):
		print 'Could not find galaxy data file %s!' %sample.tracerFile
		sys.exit(-1)
	galaxies = np.loadtxt(sample.tracerFile) #,skiprows=2)

	galHist, zsteps = np.histogram(galaxies[:,5],bins=zsteps)
	galNumDens = galHist/(sample.fSky*Vstep)
	zsteps, zmeans, zerr = binner(galaxies[:,5],galaxies[:,5],zsteps)

	#write to file	
	sample.selFnFile = sample.outputFolder+sample.sampleHandle+'_selFn.txt'
	with open(sample.selFnFile,'w') as F:
		F.write("# z n(z) f(z)\n")
		F.write("%0.3f %0.4e %0.4f\n" %(sample.zMin, galNumDens[0], galNumDens[0]/sample.tracerDens))
		for i in xrange(len(zmeans)):
			F.write("%0.3f %0.4e %0.4f\n" %(zmeans[i], galNumDens[i], galNumDens[i]/sample.tracerDens))
		F.write("%0.3f %0.4e %0.4f\n" %(sample.zMax, galNumDens[i], galNumDens[i]/sample.tracerDens))
#---------------------------------------------------------------#

#-------determine survey mask from galaxy distribution---------#
#--------------------------------------------------------------#
def generate_mask(tracerFile,maskFile):
	"""Creates an approximate survey mask from galaxy positions, and 
	writes to a FITS file with healpix
	Note that this is VERY approximate, only use if a proper survey mask
	is not available!

	Arguments:
	  tracerFile -- filename for file containing galaxy positions
	  maskFile -- output FITS filename
	"""
	#load up the galaxy data - 1st and 2nd columns should contain RA, dec
	if not os.access(tracerFile,os.F_OK):
		print 'Could not find galaxy data file %s!' %tracerFile
		sys.exit(-1)
	galaxies = np.loadtxt(tracerFile) #,skiprows=2)

	nside = 64
	npix = hp.nside2npix(nside)
	mask = np.zeros((npix))

	phi = galaxies[:,0]*np.pi/180.
	theta = np.pi/2. - galaxies[:,1]*np.pi/180.
	pixels = hp.ang2pix(nside,theta,phi)
	mask[pixels] = 1.

	fsky = 1.0*sum(mask)/len(mask)
	hp.write_map(maskFile,mask,verbose=False)
	return fsky
#--------------------------------------------------------------#

#-------find boundary pixels given a survey mask---------#
#--------------------------------------------------------#
def find_boundary(mask,c_limit=0):
	"""Finds the boundary of a survey mask

	Arguments:
	  mask -- Healpix map with pixel values giving survey completeness
	  c_limit -- value in range (0,1), sets completeness lower limit for boundary determination

	Returns:
	  boundary -- Healpix map with pixels 1 in thin boundary outside survey mask, 0 elsewhere
	"""
	nside = hp.get_nside(mask)
	npix = hp.nside2npix(nside)
	boundary = np.zeros((npix))

	#find pixels outside the mask that neighbour pixels within it
	filled_inds = np.nonzero(mask>c_limit)[0]
	theta, phi = hp.pix2ang(nside,filled_inds)
	neigh_pix = hp.get_all_neighbours(nside,theta,phi)	
	for i in range(neigh_pix.shape[1]):
		outsiders = neigh_pix[(mask[neigh_pix[:,i]]<=c_limit) & (neigh_pix[:,i]>-1),i]
		#>-1 condition takes care of special case where neighbour wasn't found
		boundary[outsiders] = 2

	#do iteration to get boundary slightly removed
	filled_inds = np.nonzero(boundary)[0]
	theta, phi = hp.pix2ang(nside,filled_inds)
	neigh_pix = hp.get_all_neighbours(nside,theta,phi)
	for i in range(neigh_pix.shape[1]):
		outsiders = neigh_pix[(mask[neigh_pix[:,i]]<=c_limit) & (neigh_pix[:,i]>-1) \
					& (boundary[neigh_pix[:,i]]==0),i]
		boundary[outsiders] = 1
	
	if nside>128:
		#do one more round
		filled_inds = np.nonzero(boundary)[0]
		theta, phi = hp.pix2ang(nside,filled_inds)
		neigh_pix = hp.get_all_neighbours(nside,theta,phi)
		for i in range(neigh_pix.shape[1]):
			outsiders = neigh_pix[(mask[neigh_pix[:,i]]<=c_limit) & (neigh_pix[:,i]>-1) \
						& (boundary[neigh_pix[:,i]]==0),i]
			boundary[outsiders] = 1
	boundary[boundary==2] = 0

	if nside<=128:
		#upgrade the boundary to aid placement of buffer mocks
		boundary = hp.ud_grade(boundary,2*nside)

	return boundary
#--------------------------------------------------------#

#------convert galaxy posn file given in (RA,Dec,z) to standard form used------#
#------------------------------------------------------------------------------#
def coords_ang2std(sample):
	"""Converts galaxy positions given in (RA,Dec,redshift) to comoving Cartesian coordinates 
	using assumed cosmology and writes to a new file

	Arguments:
	   sample -- object of type Sample
	"""
	#load up the galaxy positions
	if not os.access(sample.tracerFile,os.F_OK):
		print 'Could not find galaxy positions data file %s!' %sample.tracerFile
		sys.exit(-1)
	galaxies = np.loadtxt(sample.tracerFile) #,skiprows=2)

	#convert to Cartesian coordinates
	# 1st 3 columns are RA, dec, z; 4th column is abs mag M
	vfunc = np.vectorize(comovr)
	zrange = np.linspace(sample.zMin,sample.zMax,10)
	rvals = vfunc(zrange,sample.OmegaM)
	rinterp = interp1d(zrange,rvals)
	rdist = rinterp(galaxies[:,2])
	phi = galaxies[:,0]*np.pi/180.
	theta = np.pi/2. - galaxies[:,1]*np.pi/180.
	galX = rdist*np.sin(theta)*np.cos(phi)	#r*cos(ra)*cos(dec)
	galY = rdist*np.sin(theta)*np.sin(phi)	#r*sin(ra)*cos(dec)
	galZ = rdist*np.cos(theta)			#r*sin(dec)

	#first calculation of box size - may be superseded after adding buffer
	maxX, maxY, maxZ = np.max(galX), np.max(galY), np.max(galZ)
	minX, minY, minZ = np.min(galX), np.min(galY), np.min(galZ)
	sample.boxLen = 2.0*np.max([maxX,abs(minX),maxY,abs(minY),maxZ,abs(minZ)])+0.1

	sample.tracerFile = sample.posnFile+'.temp'
	with open(sample.tracerFile,'w') as F:
		F.write("# NumGals %d\n" %len(galaxies))
		F.write("# X Y Z RA Dec redshift M r_c\n")
		for i in range(len(galaxies)):
			F.write("%0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.2f\n" %(galX[i],galY[i],galZ[i],galaxies[i,0],galaxies[i,1],galaxies[i,2],galaxies[i,3],rdist[i]))
#------------------------------------------------------------------------------#

#------convert galaxy posn file given in Cartesian to standard form used-------#
#------------------------------------------------------------------------------#
def coords_Cartesian2std(sample):
	"""Converts galaxy positions given in comoving Cartesian coordinates to standard form
	including RA,Dec and redshift information

	Arguments:
	   sample -- object of type Sample
	"""
	#load up the galaxy positions
	if not os.access(sample.tracerFile,os.F_OK):
		print 'Could not find galaxy positions data file %s!' %sample.tracerFile
		sys.exit(-1)
	galaxies = np.loadtxt(sample.tracerFile) #,skiprows=2)

	#convert Cartesian coordinates to RA,Dec,redshift
	#assumes first three columns contain X,Y,Z, 4th column contains abs mag M
	rdist = np.linalg.norm(galaxies[:,:3],axis=1)
	Dec = 90 - np.degrees(np.arccos(galaxies[:,2]/rdist))
	RA = np.degrees(np.arctan2(galaxies[:,1],galaxies[:,0]))
	RA[RA<0] += 360 #to ensure RA is in the range 0 to 360
	rrange = np.linspace(min(rdist)-1,max(rdist+1),20)
	zvals = np.asarray([brentq(lambda x: comovr(x,sample.OmegaM) - rr, 0.0, 1.0) for rr in rrange])
	zinterp = interp1d(rrange,zvals)
	redshifts = zinterp(rdist)

	sample.tracerFile = sample.posnFile+'.temp'
	with open(sample.tracerFile,'w') as F:
		F.write("# NumGals %d\n" %len(galaxies))
		F.write("# X Y Z RA Dec redshift M r_c\n")
		for i in range(len(galaxies)):
			F.write("%0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.2f\n" %(galaxies[i,0],galaxies[i,1],galaxies[i,2],RA[i],Dec[i],redshifts[i],galaxies[i,3],rdist[i]))
#------------------------------------------------------------------------------#

#--------generate buffer mocks around a survey----------#
#-------------------------------------------------------#
def generate_buffer(sample, useGuards):
	"""Places buffer mocks around the survey boundaries to prevent leakage of Voronoi cells
	out of surveyed volume during the tessellation stage

	Arguments:
	  sample -- object of type Sample
	  useGuard -- boolean, indicating whether to use additional gridded guard particles to stabilize box
	
	Adds buffer positions to those of galaxies and writes both to file
	"""
	print "Generating buffer mocks around survey edges ..."
	#load up the galaxy positions
	if not os.access(sample.tracerFile,os.F_OK):
		print 'Could not find galaxy data file %s!' %sample.tracerFile
		sys.exit(-1)
	galaxies = np.loadtxt(sample.tracerFile)
	galX, galY, galZ = galaxies[:,0], galaxies[:,1], galaxies[:,2]

	#start collecting maximum and minimum coordinate values for box size calculation
	maxX, maxY, maxZ = np.max(galX), np.max(galY), np.max(galZ)
	minX, minY, minZ = np.min(galX), np.min(galY), np.min(galZ)

	if sample.fSky==1:
		#covers the whole sky without exception (e.g., Jubilee simulation)
		numpix = hp.nside2npix(256)
		mask = np.ones((numpix))
	else:
		#get the survey mask from file
		if not os.access(sample.maskFile,os.F_OK):
			print 'Could not find survey mask file %s!' %sample.maskFile
			sys.exit(-1)
		mask = hp.read_map(sample.maskFile,verbose=False)
	nside = hp.get_nside(mask)
	survey_pix = np.nonzero(mask)[0]
	numpix = len(survey_pix)

	#place buffer mocks along the high redshift cap
	#mocks are placed randomly in a thin shell beyond maximum radial extent of survey
	meanNNdist = (sample.tracerDens)**(-1./3)
	minDist = comovr(sample.zMax,sample.OmegaM) + meanNNdist*sample.mockDensRat**(-1./3)
	maxDist = minDist + meanNNdist

	boundmaxDist = maxDist
	boundminDist = comovr(sample.zMin,sample.OmegaM)

	buffervol = sample.fSky*4*np.pi*quad(lambda x: x**2, minDist, maxDist)[0]
	numHighMocks = int(np.ceil(sample.mockDensRat*sample.tracerDens*buffervol))
	HighMocks = np.zeros((numHighMocks,6))

	#radial distances of buffer mocks
	HighMocks[:,0] = (minDist**3. + (maxDist**3. - minDist**3.)*np.random.rand(numHighMocks))**(1.0/3)

	#angular positions of mocks
	while numHighMocks>numpix:
		#more mock posns required than mask pixels, so upgrade mask to get more pixels
		nside *= 2
		mask = hp.ud_grade(mask, nside)
		survey_pix = np.nonzero(mask)[0]
		numpix = len(survey_pix)
	rand_pix = survey_pix[random.sample(np.arange(numpix),numHighMocks)]
	HighMocks[:,1], HighMocks[:,2] = hp.pix2ang(nside,rand_pix)

	#convert to Cartesian
	HighMocks[:,3] = HighMocks[:,0]*np.sin(HighMocks[:,1])*np.cos(HighMocks[:,2])
	HighMocks[:,4] = HighMocks[:,0]*np.sin(HighMocks[:,1])*np.sin(HighMocks[:,2])
	HighMocks[:,5] = HighMocks[:,0]*np.cos(HighMocks[:,1])
	maxX, maxY, maxZ = max(maxX,np.max(HighMocks[:,3])), max(maxY,np.max(HighMocks[:,4])), max(maxZ,np.max(HighMocks[:,5]))
	minX, minY, minZ = min(minX,np.min(HighMocks[:,3])), min(minY,np.min(HighMocks[:,4])), min(minZ,np.min(HighMocks[:,5]))
	sample.numMock = numHighMocks
	print "	%d mocks at the high-redshift cap" %numHighMocks

	if sample.zMin>0:
		#place buffer mocks along the low redshift cap
		#if possible, these mocks are placed randomly in a thin shell below the low redshift survey edge
		#if this is not possible, they are placed from the origin up to the low redshift edge
		meanNNdist = sample.tracerDens**(-1./3)
		minDist = comovr(sample.zMin,sample.OmegaM) - meanNNdist*(1+sample.mockDensRat**(-1./3))
		maxDist = comovr(sample.zMin,sample.OmegaM) - meanNNdist*sample.mockDensRat**(-1./3)
		if minDist<0: minDist = 0
		if maxDist<0: maxDist = comovr(sample.zMin,sample.OmegaM)
		boundminDist = minDist

		buffervol = sample.fSky*4*np.pi*quad(lambda x: x**2, minDist, maxDist)[0]
		numLowMocks = int(np.ceil(sample.mockDensRat*sample.tracerDens*buffervol))
		LowMocks = np.zeros((numLowMocks,6))

		#radial distances of buffer mocks
		LowMocks[:,0] = (minDist**3. + (maxDist**3. - minDist**3.)*np.random.rand(numLowMocks))**(1.0/3)

		#angular positions of mocks
		while numLowMocks>numpix:
			#more mocks required than pixels in which to place them, so upgrade mask
			nside *= 2
			mask = hp.ud_grade(mask, nside)
			survey_pix = np.nonzero(mask)[0]
			numpix = len(survey_pix)
		rand_pix = survey_pix[random.sample(np.arange(numpix),numLowMocks)]
		LowMocks[:,1], LowMocks[:,2] = hp.pix2ang(nside,rand_pix)

		#convert to Cartesian
		LowMocks[:,3] = LowMocks[:,0]*np.sin(LowMocks[:,1])*np.cos(LowMocks[:,2])
		LowMocks[:,4] = LowMocks[:,0]*np.sin(LowMocks[:,1])*np.sin(LowMocks[:,2])
		LowMocks[:,5] = LowMocks[:,0]*np.cos(LowMocks[:,1])
		maxX, maxY, maxZ = max(maxX,np.max(LowMocks[:,3])), max(maxY,np.max(LowMocks[:,4])), max(maxZ,np.max(LowMocks[:,5]))
		minX, minY, minZ = min(minX,np.min(LowMocks[:,3])), min(minY,np.min(LowMocks[:,4])), min(minZ,np.min(LowMocks[:,5]))
		sample.numMock += numLowMocks
		print "	%d mocks at the low-redshift cap" %numLowMocks

	if sample.fSky<1.0:
		#place buffer mocks along the survey edges
		#these mocks are placed randomly in the boundary pixels
		boundary = find_boundary(mask)
		boundary_pix = np.nonzero(boundary)[0]
		numpix = len(boundary_pix)
		boundary_fSky = 1.0*len(boundary_pix)/len(boundary)
		boundary_nside = hp.get_nside(boundary)

		buffervol = boundary_fSky*4*np.pi*quad(lambda x: x**2, boundminDist, boundmaxDist)[0]
		numBoundMocks = int(np.ceil(sample.mockDensRat*sample.tracerDens*buffervol))
		BoundMocks = np.zeros((numBoundMocks,6))

		#radial distances of buffer mocks
		BoundMocks[:,0] = (boundminDist**3. + (boundmaxDist**3. - boundminDist**3.)*np.random.rand(numBoundMocks))**(1.0/3)
		
		#angular positions of buffer mocks
		while numBoundMocks>numpix:
			#more mocks required than pixels in which to place them, so upgrade mask
			boundary_nside *= 2
			boundary = hp.ud_grade(boundary, boundary_nside)
			boundary_pix = np.nonzero(boundary)[0]
			numpix = len(boundary_pix)
		rand_pix = boundary_pix[random.sample(np.arange(numpix),numBoundMocks)]
		BoundMocks[:,1], BoundMocks[:,2] = hp.pix2ang(boundary_nside,rand_pix)

		#convert to Cartesian
		BoundMocks[:,3] = BoundMocks[:,0]*np.sin(BoundMocks[:,1])*np.cos(BoundMocks[:,2])
		BoundMocks[:,4] = BoundMocks[:,0]*np.sin(BoundMocks[:,1])*np.sin(BoundMocks[:,2])
		BoundMocks[:,5] = BoundMocks[:,0]*np.cos(BoundMocks[:,1])
		maxX, maxY, maxZ = max(maxX,np.max(BoundMocks[:,3])), max(maxY,np.max(BoundMocks[:,4])), max(maxZ,np.max(BoundMocks[:,5]))
		minX, minY, minZ = min(minX,np.min(BoundMocks[:,3])), min(minY,np.min(BoundMocks[:,4])), min(minZ,np.min(BoundMocks[:,5]))
		sample.numMock += numBoundMocks
		print "	%d mocks around the survey boundary" %numBoundMocks
	
	#calculate the required box size to enclose all particles
	sample.boxLen = 2.0*np.max([maxX,abs(minX),maxY,abs(minY),maxZ,abs(minZ)])+0.1
	print "    Box of length %0.3f" %sample.boxLen 
	#NOTE: This will often produce a much bigger box than strictly necessary to 
	#enclose the galaxies. This will not significantly slow the tessellation, but it
	#does help to keep the code simple! :)

	if useGuards:
		#add some sparse guard particles well outside the survey area, so that qhull doesn't complain
		galPos = galaxies[:,:3] + sample.boxLen/2.0	#half-box shift so we can use periodic KD Tree
		galTree = cKDTree(galPos,boxsize=sample.boxLen)
		print "Made tree"
		X = np.linspace(0.1,sample.boxLen-0.1,20)
		guards = np.vstack(np.meshgrid(X,X,X)).reshape(3,-1).T
		print "Made guards"

		nndist = np.empty(len(guards))
		print "%d guards" %len(guards)
		for i in range(len(guards)):
			nndist[i], nnind = galTree.query(guards[i,:],k=1)
		guards = guards[nndist>sample.boxLen/20.0,:]
		print "%d meet criteria" %len(nndist>sample.boxLen/20.)
		print "sifted guards"

		#undo half-box shift
		guards = guards - sample.boxLen/2.0
		sample.numMock += len(guards)
		del galTree
		print "	%d gridded guard mocks to stabilize the box" %len(guards)

	#now write these positions to a temporary file
	sample.tracerFile = sample.posnFile+'.temp'
	with open(sample.tracerFile,'w') as F:
		F.write("NumGals %d\nNumBufferMocks %d\nBoxLen %0.3f\n" %(len(galaxies),sample.numMock,sample.boxLen))
		F.write("X Y Z RA Dec redshift ignore r_c\n")
		for i in range(len(galaxies)):
			F.write("%0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.2f\n" %(galX[i],galY[i],galZ[i],galaxies[i,3],galaxies[i,4],galaxies[i,5],\
				galaxies[i,6],galaxies[i,7]))
		HighMocks[:,2] = HighMocks[:,2]*180./np.pi
		HighMocks[:,1] = 90 - HighMocks[:,1]*180./np.pi
		for i in range(len(HighMocks)):
			F.write("%0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.2f\n" %(HighMocks[i,3],HighMocks[i,4],HighMocks[i,5],HighMocks[i,2],\
				HighMocks[i,1],-1,0,HighMocks[i,0]))
		if sample.zMin>0:
			LowMocks[:,2] = LowMocks[:,2]*180./np.pi
			LowMocks[:,1] = 90 - LowMocks[:,1]*180./np.pi
			for i in range(len(LowMocks)):
				F.write("%0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.2f\n" %(LowMocks[i,3],LowMocks[i,4],LowMocks[i,5],LowMocks[i,2],\
					LowMocks[i,1],-1,0,LowMocks[i,0]))
		if sample.fSky<1.0:
			BoundMocks[:,2] = BoundMocks[:,2]*180./np.pi
			BoundMocks[:,1] = 90 - BoundMocks[:,1]*180./np.pi
			for i in range(len(BoundMocks)):
				F.write("%0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.2f\n" %(BoundMocks[i,3],BoundMocks[i,4],BoundMocks[i,5],\
					BoundMocks[i,2],BoundMocks[i,1],-1,0,BoundMocks[i,0]))
		if useGuards:
			dist = np.linalg.norm(guards,axis=1)
			for i in range(len(guards)):
				F.write("%0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.2f\n" %(guards[i,0],guards[i,1],guards[i,2],\
					-60,-60,-1,0,dist[i]))

#-------------------------------------------------------#

#--------get indices of points within given radius of a point------------#
def getBall(Tree,centre,radius):
	"""Queries a kd-tree object and returns all points within given distance of a given point

	Arguments:
	   Tree -- kd-Tree object
	   centre -- coordinates of the central point
	   radius -- maximum distance from centre
	"""

	return Tree.query_ball_point(centre,radius)
#-------------------------------------------------------------------------#

#----------run ZOBOV scripts for sample--------#
#----------------------------------------------#
def runZOBOV(sample, useIsol=False, numZobovDivisions=2, ZobovBufferSize=0.1, doSC=False):
	"""Wrapper function to call C-based ZOBOV codes based on chosen options.

	Arguments:
	   sample -- object of type Sample
	   useIsol -- boolean, True if using vozisol.c
	   isBox -- boolean, True only if tracer positions occupy full cubic box with PBC
	   numZobovDivisions -- number of ZOBOV box sub-divisions per side, ignored if using vozisol.c
	   ZobovBufferSize -- size of buffer region about each box sub-division, in units of box side
	"""

	if useIsol: #tessellate the entire box in one go using vozisol

		print "Calling vozisol to do the tessellation\n"
		logFile = "./log/"+sample.sampleHandle+".out"
		log = open(logFile,"w")
		cmd = ["./bin/vozisol", sample.posnFile, sample.sampleHandle, str(sample.boxLen), \
			str(sample.numTracer), str(0.9e30)]	
		subprocess.call(cmd,stdout=log,stderr=log)
		log.close()
		# (vozisol will also account for buffer mocks etc automatically)

	else:	#tessellate the box in chunks, using voz1b1 and voztie

		print "Calling vozinit, voz1b1 and voztie to do the tessellation\n"
		# first call vozinit to write the script used to call voz1b1 and voztie		
		logFile = "./log/"+sample.sampleHandle+".out"
		log = open(logFile,"w")
		cmd = ["./bin/vozinit", sample.posnFile, str(ZobovBufferSize), str(sample.boxLen), \
			str(numZobovDivisions), sample.sampleHandle]
		subprocess.call(cmd,stdout=log,stderr=log)
		log.close()

		# now call this script to do the tessellation
		vozScript = "scr"+sample.sampleHandle
		cmd = ["./%s" % vozScript]
		log = open(logFile, 'a')
		subprocess.call(cmd, stdout=log, stderr=log)
		log.close()
	
		# check to see if the tessellation was successful
		if not os.access("%s.vol"%sample.sampleHandle,os.F_OK):
			print "Something went wrong with the tessellation. Aborting ..."
			sys.exit(-1)
			
		# remove the script file
		if os.access(vozScript, os.F_OK): os.unlink(vozScript)

		# copy .vol file to .trvol
		cmd = ["cp","%s.vol" %sample.sampleHandle, "%s.trvol" %sample.sampleHandle]
		subprocess.call(cmd)

		if sample.numMock>0:
			# remove buffer mocks and flag edge particles using checkedges.c
			# (voz1b1 and voztie do not account for buffer mocks automatically)
			cmd = ["./bin/checkedges", sample.sampleHandle, str(sample.numTracer), str(0.9e30)]
			log = open(logFile, 'a')
			subprocess.call(cmd, stdout=log, stderr=log)
			log.close()

	print "Tessellation done\n"

	if sample.isBox:	
		#simply copy %s.vol file to %sc.vol
		cmd = ["cp","%s.vol" %sample.sampleHandle, "%sc.vol" %sample.sampleHandle]
		subprocess.call(cmd)

	else:	#renormalize densities to account for buffer mocks in the box
		#read original Voronoi volumes
		with open('./%s.trvol' %sample.sampleHandle,'r') as File:
			Npreal = np.fromfile(File, dtype=np.int32,count=1)
			#sanity check
			if not Npreal==sample.numTracer:
				print "Npreal = %d in .vol file does not match numTracer = %d!" %(Npreal, sample.numTracer)
				sys.exit(-1)
			trvols = np.fromfile(File, dtype=np.float64,count=Npreal)
		#and read the edge-flagged values
		with open('./%s.vol' %sample.sampleHandle,'r') as File:
			Npreal = np.fromfile(File, dtype=np.int32,count=1)
			modvols = np.fromfile(File, dtype=np.float64,count=Npreal)

		#renormalize volumes in units of mean volume per real galaxy
		edgemask = modvols==1.0/0.9e30
		modvols[edgemask==False] *= (sample.tracerDens*sample.boxLen**3.)/sample.numPartTot

		#survey volume lost to buffer particle encroachment (only for info, not used anywhere)
		totalSurveyVol = np.sum(trvols[edgemask==False])	
		lostfrac = (sample.numTracer/sample.tracerDens - (sample.boxLen**3.0)*(totalSurveyVol/sample.numPartTot))*(sample.tracerDens/sample.numTracer)
		print "Fraction of survey volume that is usable = %0.3f\n" %(1-lostfrac)

		#additionally scale densities in order to account for redshift-dependent selection function
		gal_data = np.loadtxt(sample.tracerFile,skiprows=4)		
		if sample.useSelFn:
			#read in the individual redshifts
			redshifts = gal_data[:sample.numTracer,5]
			#read in the selection function
			selfnbins = np.loadtxt(sample.selFnFile,skiprows=1)
			#create linear interpolation for normalized sel. fn.
			selfn = InterpolatedUnivariateSpline(selfnbins[:,0],selfnbins[:,2],k=1)
			#smooth this with a Savitzky-Golay filter to remove high-frequency noise
			x = np.linspace(sample.zMin,sample.zMax,1000)
			y = savgol_filter(selfn(x),101,3)
			#then linearly interpolate the filtered interpolation itself
			selfn = InterpolatedUnivariateSpline(x,y,k=1)
			#scale the densities according to this
			modfactors = selfn(redshifts[edgemask==False])
			modvols[edgemask==False] *= modfactors

		#additionally scale densities to account for sky-varying completeness
		if sample.fSky<1.0:
			#read in the individual sky positions
			ra = gal_data[:sample.numTracer,3]
			dec = gal_data[:sample.numTracer,4]
			#fetch the survey mask
			mask = hp.read_map(sample.maskFile,verbose=False)
			nside = hp.get_nside(mask)
			#weight the densities by completeness
			pixels = hp.ang2pix(nside,np.deg2rad(90-dec),np.deg2rad(ra))
			modfactors = mask[pixels]
			modvols[edgemask==False] *= modfactors[edgemask==False]

		#write the modified volumes to file for jozov to use to find densities
		with open("./%s.vol" %sample.sampleHandle,'w') as F:
			Npreal.tofile(F,format="%d")
			modvols.tofile(F,format="%f")

		#invert the edge flag value for finding clusters
		modvols[edgemask] = 0.9e30
		#and write to c.vol file
		with open("./%sc.vol" %sample.sampleHandle,'w') as F:
			Npreal.tofile(F,format="%d")
			modvols.tofile(F,format="%f")

		#set the number of non-edge galaxies
		sample.numNonEdge = sample.numTracer - sum(edgemask)
	
	# now call jozov, once for voids ...
	cmd = ["./bin/jozovtrvol", "v", sample.sampleHandle, str(0), str(0)]
	log = open(logFile,'a')
	subprocess.call(cmd)
	log.close()
	# ... and, if requested, once for superclusters
	if doSC:
		cmd = ["./bin/jozovtrvol", "c", sample.sampleHandle, str(0), str(0)]
		log = open(logFile,'a')
		subprocess.call(cmd)
		log.close()

	# remove unnecessary files
	for fileName in glob.glob("./part."+sample.sampleHandle+".*"):
		os.unlink(fileName)

	# ... and move others to the appropriate directory
	rawDir = sample.outputFolder+"rawZOBOV/"	
	if not os.access(rawDir, os.F_OK):
		os.makedirs(rawDir)
	for fileName in glob.glob("./"+sample.sampleHandle+"*"):
		cmd = ["mv", fileName, "%s." %rawDir]
		subprocess.call(cmd)
			 
#----------------------------------------------#

#------------get total DM mass in an annular region about given centre using healpy----------#
#------------------------(DMdens is the DM density on a cubic grid)--------------------------#
def annular_DM_healpy(DMdens, centre, rmin, rmax, resolution, boxLen):
	"""Calculates the total mass within a shell rmin<r<rmax about a given centre. Uses healpy.

	Arguments:
	   DMDens -- array containing gridded density field, dimensions (resolution,resolution,resolution)
	   centre -- coordinates of centre
	   rmin -- inner radius of shell
	   rmax -- maximum radius of shell
	   resolution -- resolution/dimensions of DMDens
	   boxLen -- simulation box length in Mpc/h
	"""
	 
	DeltaR = rmax - rmin
	ang_res_factor = resolution/512.0
	rad_res_factor = 0.75*resolution/512.0
	Np_rad = np.max([np.round(rad_res_factor*DeltaR),3])
	Nside = np.max([2**(np.round(np.log(rmax*ang_res_factor/2.0)/np.log(2.0))),1]).astype(int)
	if Nside>8: Nside = 8	#for speed! Nside=8 should be enough sampling

	r_axis = np.linspace(rmin,rmax,Np_rad)
	pixel_list = np.arange(12*Nside**2)
	r_vals,pixels=np.ix_(r_axis,pixel_list)
	x_vals,y_vals,z_vals = hp.pix2vec(Nside,pixels)
	
	#convert to Cartesian coordinate indices
	xind = np.mod(np.floor((centre[0]+r_vals*x_vals)*resolution/boxLen),resolution)
	yind = np.mod(np.floor((centre[1]+r_vals*y_vals)*resolution/boxLen),resolution)
	zind = np.mod(np.floor((centre[2]+r_vals*z_vals)*resolution/boxLen),resolution)

	#get the densities at these locations
	dens_vals = DMdens[xind.astype(int),yind.astype(int),zind.astype(int)]

	#volume integral of these values to get the enclosed mass
	ang_avg = np.mean(dens_vals,axis=1)
	mass = (4*np.pi/3) * np.trapz(ang_avg,r_axis**3)

	return mass
	
#--------------------------------------------------------------------------------------------#

#------------get average Phi in an annular region about given centre using healpy----------#
def annular_Phi_healpy(Phi, centre, rmin, rmax, resolution, boxLen):
	"""Calculates the average gravitational potential value within a shell rmin<r<rmax about a given centre. Uses healpy.

	Arguments:
	   Phi -- array containing gridded grav. pot. field, dimensions (resolution,resolution,resolution)
	   centre -- coordinates of centre
	   rmin -- inner radius of shell
	   rmax -- maximum radius of shell
	   resolution -- resolution/dimensions of Phi
	   boxLen -- simulation box length in Mpc/h
	"""
	 
	DeltaR = rmax - rmin
	ang_res_factor = resolution/512.0
	rad_res_factor = 0.75*resolution/512.0
	Np_rad = np.max([np.round(rad_res_factor*DeltaR),3])
	Nside = np.max([2**(np.round(np.log(rmax*ang_res_factor/2.0)/np.log(2.0))),1]).astype(int)
	if Nside>8: Nside = 8	#for speed! Nside=8 should be enough sampling

	r_axis = np.linspace(rmin,rmax,Np_rad)
	pixel_list = np.arange(12*Nside**2)
	r_vals,pixels=np.ix_(r_axis,pixel_list)
	x_vals,y_vals,z_vals = hp.pix2vec(Nside,pixels)
	
	#convert to Cartesian coordinate indices
	xind = np.mod(np.floor((centre[0]+r_vals*x_vals)*resolution/boxLen),resolution)
	yind = np.mod(np.floor((centre[1]+r_vals*y_vals)*resolution/boxLen),resolution)
	zind = np.mod(np.floor((centre[2]+r_vals*z_vals)*resolution/boxLen),resolution)

	#get the Phi values at these locations
	Phi_vals = Phi[xind.astype(int),yind.astype(int),zind.astype(int)]

	#volume integral of these values to get the enclosed mass
	ang_avg = np.mean(Phi_vals,axis=1)
	Phi_avg = np.trapz(ang_avg,r_axis**3)	#this is 3/4pi times the volume integral of Phi over the annulus

	return Phi_avg
	
#--------------------------------------------------------------------------------------------#
