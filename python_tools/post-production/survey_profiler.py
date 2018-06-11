# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 13:51:48 2016

@author: seshadri
"""

#!/usr/bin/python

import imp
import argparse
from periodic_kdtree import PeriodicCKDTree
from scipy.spatial import cKDTree
from scipy.optimize import fsolve
from classes import *
from tools import *

def HSWprofile(r, rs, dc, alpha, beta):
	return 1 + dc*(1 - (r/rs)**alpha)/(1 + (r)**beta)

#----------from input options select a defined sub-section of specified catalogue-------#
#-------------returns Nstruct, void centres, radii, mean radius and densities-----------#
def subsection(catalogue):

	InfoFile = os.getenv('HOME') + "/Workspace/structures/BigMDPl/" + catalogue.sHandle + \
				"/obs_centres/" + catalogue.prefix + "_info.txt"
	ListFile = os.getenv('HOME') + "/Workspace/structures/BigMDPl/" + catalogue.sHandle + \
				"/" + catalogue.prefix + "_list.txt"
	CatArray = np.loadtxt(InfoFile,skiprows=2)
	ListArray = np.loadtxt(ListFile,skiprows=2)
	
	#select the sub-section matching the given criteria
	# 1. max no. of merged zones (if reqd)
	CatArray = CatArray[np.logical_or(catalogue.zoneMax==0,ListArray[:,5]<catalogue.zoneMax)]
	# 2. central density cuts
	CatArray = CatArray[np.logical_and(CatArray[:,4]>catalogue.CentDensMin, 
				CatArray[:,4]<catalogue.CentDensMax)]
	# 3. wtd. avg. density cuts
	CatArray = CatArray[np.logical_and(CatArray[:,5]>catalogue.AvgDensMin, 
				CatArray[:,5]<catalogue.AvgDensMax)]
	# 4. radius cuts
	CatArray = CatArray[np.logical_and(CatArray[:,6]>catalogue.Rmin, 
				CatArray[:,6]<catalogue.Rmax)]
	
	Nstruct = len(CatArray)
	meanCentDens = np.mean(CatArray[:,4])
	CentDensErr = np.std(CatArray[:,4])/np.sqrt(Nstruct) if Nstruct>0 else 0
	meanAvgDens = np.mean(CatArray[:,5])
	AvgDensErr = np.std(CatArray[:,5])/np.sqrt(Nstruct) if Nstruct>0 else 0
	meanRad = np.mean(CatArray[:,6])
	RadErr = np.std(CatArray[:,6])/np.sqrt(Nstruct) if Nstruct>0 else 0
	centres = CatArray[:,1:4]
	radii = CatArray[:,6]

	return Nstruct, centres, radii, meanRad, RadErr, meanCentDens, CentDensErr, meanAvgDens, AvgDensErr
#---------------------------------------------------------------------------------------#

#-------------obtain tracer density profiles for list of centres and radii-----------#
def PoissonProfile(centres, radii, meanDens, T, maxDist=3.0, nbins=30, diff=True):
		
	Nstruct = len(radii)
	scale = maxDist/float(nbins)
	rbins = np.fromfunction(lambda i,j: (j+1)*radii[i]*scale,(Nstruct,nbins),dtype=int)

	print "Obtaining the average tracer density profile for %i structures ..." %Nstruct		
	numbers = np.empty((Nstruct,nbins),dtype=int)
	for i in range(numbers.shape[0]):
		for j in range(numbers.shape[1]):
			numbers[i,j] = len(getBall(T,centres[i],rbins[i,j]))
	Volumes = (4*np.pi*rbins**3)/3.0
	
	#CUMULATIVE density - total number per volume within given radius
	CTotals = np.sum(numbers,axis=0)
	CDensMean = (CTotals+1)/np.sum(Volumes,axis=0)	# +1 to account for Poisson statistics!
	CDensMean /= meanDens
	#print CTotals
	#print np.sum(Volumes,axis=0)
	#print meanDens
	
	#DIFFERENTIAL density - number per volume within each bin
	Dnumbers = np.fromfunction(lambda i,j: numbers[i,j+1]-numbers[i,j],(Nstruct,nbins-1),dtype=int)
	Dnumbers = np.insert(Dnumbers,0,numbers[:,0],axis=1)
	DVolumes = np.fromfunction(lambda i,j: Volumes[i,j+1]-Volumes[i,j],(Nstruct,nbins-1),dtype=int)
	DVolumes = np.insert(DVolumes,0,Volumes[:,0],axis=1)
	DTotals = np.sum(Dnumbers,axis=0)
	DDensMean = (DTotals+1)/np.sum(DVolumes,axis=0)	# +1 to account for Poisson statistics!
	DDensMean /= meanDens
		
	#Error bars
	CHErr = np.empty((nbins)) 
	CLErr = np.empty((nbins)) 
	DHErr = np.empty((nbins))
	DLErr = np.empty((nbins))
	for j in range(nbins):
		CHErr[j] = PoissHErr(CTotals[j]+1)
		CLErr[j] = PoissLErr(CTotals[j]+1)
		DHErr[j] = PoissHErr(DTotals[j]+1)
		DLErr[j] = PoissLErr(DTotals[j]+1)
	CHErr /= (meanDens*np.sum(Volumes,axis=0))
	CLErr /= (meanDens*np.sum(Volumes,axis=0))
	DHErr /= (meanDens*np.sum(DVolumes,axis=0))
	DLErr /= (meanDens*np.sum(DVolumes,axis=0))
				
	#calculate and return desired result
	if diff:
		return DDensMean, DHErr, DLErr
	else:
		return CDensMean, CHErr, CLErr
#----------------------------------------------------------------------------#

#-------------obtain tracer density profiles for list of centres and radii, using VTFE method-----------#
def VTFEProfile(centres, radii, T, vols, dens, maxDist=3.0, nbins=30, diff=True):
		
	Nstruct = len(radii)
	scale = maxDist/float(nbins)
	rbins = np.fromfunction(lambda i,j: (nbins-j)*radii[i]*scale,(Nstruct,nbins),dtype=int)

	print "Obtaining the average VTFE tracer density profile for %i structures ..." %Nstruct
		
	#start by calculating cumulative values within spheres
	dens_sum = np.empty((Nstruct,nbins),dtype=float)
	wts_sum = np.empty((Nstruct,nbins),dtype=float)
	numbers = np.zeros((Nstruct,nbins),dtype=int)
	for i in range(Nstruct):
		for j in range(nbins):
			part_inds = getBall(T,centres[i],rbins[i,j])
			numbers[i,j] = len(part_inds) #number of particles within rbins[i,j] of ith void centre 
			if numbers[i,j]>0:	
				#calculate new values for this sphere
				dens_sum[i,j] = np.sum(dens[part_inds]*vols[part_inds])
				wts_sum[i,j] = np.sum(vols[part_inds])
			else:		
				if j==0: 	#idiot-proof check
					print 'No particles found anywhere near void %d, some crazy input!' %i
					print 'bin %d' %j
					print 'Centre = ',centres[i]
					print 'radius = ',rbins[i,j]
					print 'numbers = ',numbers[i,j]
					sys.exit(-1)
				#no further particles closer to centre, so keep value of previous sphere
				dens_sum[i,j] = dens_sum[i,j-1]
				wts_sum[i,j] = wts_sum[i,j-1]

	
	if diff:
		#from these obtain the values for annular shells
		shell_dens_sum = dens_sum
		shell_wts_sum = wts_sum
		for i in range(Nstruct):
			shell_numbers = numbers[i,:-1] - numbers[i,1:]
			shell_numbers = np.insert(shell_numbers,nbins-1,numbers[i,nbins-1])
			nonzeroind = np.nonzero(shell_numbers)[0] 
			zeroind = np.where(shell_numbers==0)[0]
			#first calculate shell values for filled shells ...
			reset_ind = nonzeroind[:-1]	#don't change value for innermost filled shell
			shell_dens_sum[i,reset_ind] = dens_sum[i,reset_ind] - dens_sum[i,reset_ind+1]
			shell_wts_sum[i,reset_ind] = wts_sum[i,reset_ind] - wts_sum[i,reset_ind+1]
			#... now deal with the empty shells
			for j in zeroind:
				#if there are no particles closer to the void centre either, don't change default assignment
				if numbers[i,j]==0: break
				if nonzeroind[j-nonzeroind>0].shape[0]==0:	#special case - no particles in an outer shell!
					highind = lowind = np.min(nonzeroind[nonzeroind-j>0])
				else:
					highind = np.min(nonzeroind[nonzeroind-j>0])
					lowind = np.max(nonzeroind[j-nonzeroind>0])
				#current implementation differs slightly from original in arXiv:1407.1295!
				shell_dens_sum[i,j] = np.sum(shell_dens_sum[i,[lowind,highind]])
				shell_wts_sum[i,j] = np.sum(shell_wts_sum[i,[lowind,highind]])
		#reassign back
		dens_sum = shell_dens_sum
		wts_sum = shell_wts_sum

	#swap the columns back to conventional order
	dens_sum = dens_sum[:,::-1]
	wts_sum = wts_sum[:,::-1]

	#create the jackknife samples
	Dens_Jack = np.empty((Nstruct,nbins))	#create jackknife samples from the individual structures directly
	for i in range(Nstruct):
		Dens_Jack[i,:] = (np.sum(dens_sum,axis=0)-dens_sum[i])/(np.sum(wts_sum,axis=0)-wts_sum[i])
#	if Nstruct<250:
#		Dens_Jack = np.empty((Nstruct,nbins))	#create jackknife samples from the individual structures directly
#		for i in range(Nstruct):
#			Dens_Jack[i,:] = (np.sum(dens_sum,axis=0)-dens_sum[i])/(np.sum(wts_sum,axis=0)-wts_sum[i])
#	else: 
#		numJack = min(Nstruct/25, 100)
#		step = np.ceil(Nstruct/numJack)
#		Dens_Jack = np.empty((numJack,nbins))	#create jackknife samples from the individual structures directly
#		for i in range(numJack):
#			Dens_Jack[i,:] = (np.sum(dens_sum,axis=0)-np.sum(dens_sum[i::step],axis=0))/(np.sum(wts_sum,axis=0)-np.sum(wts_sum[i::step],axis=0))
	Jack_mean = np.mean(Dens_Jack,axis=0)
	Jack_err = np.std(Dens_Jack,axis=0)*np.sqrt(Nstruct)

	return Jack_mean, Jack_err, dens_sum/wts_sum
	
#------------------------------------------------------------------------------------------------------#


#----------Poisson error bar routines---------#
def PoissLErr (elam, prob = 0.342):

	if elam <=140:
		integ = lambda x: quad(lambda y: y**(elam-1)*np.exp(-y)/np.math.factorial(elam-1),x,elam)[0] - prob
		result = fsolve(integ,elam-np.sqrt(elam-1)) 
		return (elam - result)[0]
	else: #avoid overflow errors for large elam
		return np.sqrt(elam)

def PoissHErr (elam, prob = 0.342):

	if elam <= 140:
		integ = lambda x: quad(lambda y: y**(elam-1)*np.exp(-y)/np.math.factorial(elam-1),elam,x)[0] - prob
		result = fsolve(integ,elam+np.sqrt(elam-1)) 
		return (result - elam)[0]
	else: #avoid overflow errors for large elam
		return np.sqrt(elam)
#---------------------------------------------#


stackOptionsFile = "/home/seshadri/Workspace/ZOBOV/datasets/stackOptions/stackOptions.py" #default, will be overridden by the --par option

# Read in settings from file
parser = argparse.ArgumentParser(description='options')
parser.add_argument('--par', dest='par',default="",help='path to parameter file')
args = parser.parse_args()

filename = args.par
if os.access(filename, os.F_OK):
	print "Loading parameters from", filename
	parms = imp.load_source("name", filename)
	globals().update(vars(parms))
else:
	print "\nDid not find settings file %s, proceeding with default settings\n" % filename
	defaultFile = os.getenv('HOME')+"/Workspace/ZOBOV/datasets/stackOptions/stackOptions.py"
	parms = imp.load_source("name", stackOptionsFile)
	globals().update(vars(parms))

#pick up additional details - tracer average number density and isBox
sampleInfoFile = os.getenv('HOME')+"/Workspace/structures/BigMDPl/"+sHandle+"/sample_info.dat"
parms = imp.load_source("name", sampleInfoFile)
globals().update(vars(parms))
meanDens = parms.tracerDens
selFnFile = os.getenv('HOME')+"/Workspace/structures/BigMDPl/"+sHandle+"/"+sHandle+"_selFn.dat"

diffStr = 'differential/' if differential else 'cumulative/'
profileDir = os.getenv('HOME')+"/Workspace/structures/BigMDPl/"+sHandle+"/profiles/"+diffStr
if not os.access(profileDir, os.F_OK): 
	os.makedirs(profileDir)
if useVTFE:
	VTFEprofileDir = profileDir+"VTFE/"
	if not os.access(VTFEprofileDir, os.F_OK): 
		os.makedirs(VTFEprofileDir)

#load up the tracer particle positions
posnFile = os.getenv('HOME')+"/Workspace/structures/BigMDPl/"+sHandle+"/"+sHandle+"_pos.dat"
with open(posnFile,'r') as File:
	Np = np.fromfile(File, dtype=np.int32,count=1)
	Posns = np.empty([Np,6])
	Posns[:,0] = np.fromfile(File, dtype=np.float64,count=Np)
	Posns[:,1] = np.fromfile(File, dtype=np.float64,count=Np)
	Posns[:,2] = np.fromfile(File, dtype=np.float64,count=Np)
	Posns[:,3] = np.fromfile(File, dtype=np.float64,count=Np)
	Posns[:,4] = np.fromfile(File, dtype=np.float64,count=Np)
	Posns[:,5] = np.fromfile(File, dtype=np.float64,count=Np)
Posns = Posns[:parms.numTracer]

#load the VTFE volumes
volFile = os.getenv('HOME')+"/Workspace/structures/BigMDPl/"+sHandle+"/rawZOBOV/"+sHandle+".trvol"
with open(volFile,'r') as File:
	Np = np.fromfile(File, dtype=np.int32,count=1)
	vols = np.fromfile(File, dtype=np.float64,count=Np)
vols = vols[:parms.numTracer]
vols *= (parms.boxLen**3.)/(parms.numTracer+parms.numMock) 	

#load the VTFE density information
densFile = os.getenv('HOME')+"/Workspace/structures/BigMDPl/"+sHandle+"/rawZOBOV/"+sHandle+".vol"
with open(densFile,'r') as File:
	Np = np.fromfile(File, dtype=np.int32,count=1)
	modvols = np.fromfile(File, dtype=np.float64,count=Np)
modvols = modvols[:parms.numTracer]

edgemask = modvols==1.0/0.9e30 	#find the edge-flagged galaxies

edgevol = np.sum(vols[edgemask])
nonedgevol = np.sum(vols[edgemask==False])
surveyvol = parms.fSky*4*np.pi*(comovr(zMax,OmegaM)**3. - comovr(zMin,OmegaM)**3.)/3
expected_edge_vol = surveyvol-nonedgevol
encroachment = (expected_edge_vol-edgevol)/edgevol
print "encroachment = %0.6f" %encroachment
vols[edgemask] *= (1+encroachment)


#find densities for these edge-flagged galaxies
redshifts = Posns[:,5]
selfnbins = np.loadtxt(selFnFile,skiprows=1)
selfn = InterpolatedUnivariateSpline(selfnbins[:,0],selfnbins[:,2],k=1)
x = np.linspace(zMin,zMax,1000)
y = savgol_filter(selfn(x),101,3)
selfn = InterpolatedUnivariateSpline(x,y,k=1)
modfactors = selfn(redshifts[edgemask])
modvols[edgemask] = vols[edgemask]*modfactors*meanDens
densities = 1./modvols

# and build the kd-tree
T = cKDTree(Posns[:,:3])	

# obtain the tracer density profiles and write to file
for thisStack in stackList:
#	binCentres = np.fromfunction(lambda i,j: 0.5*(2*j+1)*MaxDist/float(Nbins),(1,Nbins),dtype=int)
	binEdges = MaxDist*np.arange(Nbins+1)/Nbins	
	binCentres = 0.5*(binEdges[1:]+binEdges[:-1])
	binCentres = binCentres.reshape((Nbins,1))
	
	catalogue = Catalogue(sHandle = sHandle, prefix = prefix, stackName = thisStack['stackName'], 
		Rmin = thisStack['Rmin'], Rmax = thisStack['Rmax'], CentDensMin = thisStack['CentDensMin'], 
		CentDensMax = thisStack['CentDensMax'], AvgDensMin = thisStack['AvgDensMin'], 
		AvgDensMax = thisStack['AvgDensMax'], zoneMax = thisStack['zoneMax'], Nbins = Nbins, 
		MaxDist = MaxDist, isBox = isBox, boxLen = boxLen, tracerDens = meanDens)
	Nstruct, centres, radii, meanRad, RadErr, meanCentDens, CentDensErr, meanAvgDens, AvgDensErr = subsection(catalogue)
	
	if Nstruct>0:
		if useVTFE:
			VTFEmean, VTFEerr, VTFEprofiles = VTFEProfile(centres, radii, T, vols, densities, MaxDist, Nbins, differential)
			profileFile = VTFEprofileDir+thisStack['stackName']
			output = np.hstack([binCentres,np.transpose(VTFEprofiles)])
#			np.savetxt(profileFile,output,fmt='%0.5f',header='%d structures'%Nstruct)
			with open(profileFile,'w') as F:
				F.write("%i structures\nn_min: %0.3f(%0.3f), n_avg: %0.2f(%0.2f), R_v: %0.2f(%0.2f)\n" \
					%(Nstruct,meanCentDens,CentDensErr,meanAvgDens,AvgDensErr,meanRad,RadErr))
				for i in xrange(Nbins):
					F.write("%0.2f %f %f\n" %(binCentres[i,0],VTFEmean[i],VTFEerr[i]))
		TracerMean, HErr, LErr = PoissonProfile(centres, radii, meanDens, T, MaxDist, Nbins, differential)
		profileFile = profileDir+thisStack['stackName']
		with open(profileFile,'w') as F:
			F.write("%i structures\nn_min: %0.3f(%0.3f), n_avg: %0.2f(%0.2f), R_v: %0.2f(%0.2f)\n" \
				%(Nstruct,meanCentDens,CentDensErr,meanAvgDens,AvgDensErr,meanRad,RadErr))
			for i in xrange(Nbins):
				F.write("%0.2f %f %f %f\n" %(binCentres[i,0],TracerMean[i],HErr[i],LErr[i]))
	else:
		print "No structures satisfy these criteria"

