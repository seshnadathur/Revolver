import numpy as np
import glob
import sys
import os

#----------recalculate DM and Phi-dependent quantities for all _metrics.txt files in a folder----------------#
def recalcMetrics(structdir,PhiFile,DMFile,NgridPhi,NgridDM,boxLen):

	InfoList = glob.glob(structdir+'*_info.txt')
	print 'Loading Phi ...'
	sys.stdout.flush()
	with open(PhiFile,'r') as Fin:
		Phi = np.fromfile(Fin,dtype=np.float32)
	Phi = Phi.reshape((NgridPhi,NgridPhi,NgridPhi))
#	Phi = np.load(os.getenv('HOME')+'/Workspace/Sims/MultiDark/MDR1_Phi_1024.npy')
	for name in InfoList:
		Info = np.loadtxt(name,skiprows=2)
		centres = Info[:,1:4]
		radii = Info[:,6]
		metname = name.replace('_info.txt','_metrics.txt')
		Metrics = np.loadtxt(metname,skiprows=1)
		xind = np.mod(np.floor(centres[:,0]*float(NgridPhi)/boxLen),NgridPhi)
		yind = np.mod(np.floor(centres[:,1]*float(*NgridPhi)/boxLen),NgridPhi)
		zind = np.mod(np.floor(centres[:,2]*float(NgridPhi)/boxLen),NgridPhi)
		#get the Phi values at these grid cells
		Phi_cent_vals = Phi[xind.astype(int),yind.astype(int),zind.astype(int)]*10**5
		with open(metname,'w') as Fout:
			Fout.write("StructID R_eff(Mpc/h) CentNumDens WtdAvgNumDens CentDMDens Phi_cent*10^5 DeltaN(Rv) Delta(Rv) DeltaN(3Rv) Delta(3Rv)\n")
			for i in range(Metrics.shape[0]):
				Fout.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" %(Metrics[i,0], Metrics[i,1], Metrics[i,2], Metrics[i,3], \
					Metrics[i,4], Phi_cent_vals[i], Metrics[i,6], Metrics[i,7], Metrics[i,8], Metrics[i,9]))
	del Phi
	print 'Loading DM density ...'
	sys.stdout.flush()
	with open(DMFile,'r') as Fin:
		DMDens = np.fromfile(Fin,dtype=np.float32)
	DMDens = DMDens.reshape((NgridDM,NgridDM,NgridDM))
#	DMDens = np.load(os.getenv('HOME')+'/Workspace/Sims/MultiDark/MDR1_Dens_1024.npy')
	DMDens+=1
	print 'now calculating Delta quantities ...'
	sys.stdout.flush()
	for name in InfoList:
		Info = np.loadtxt(name,skiprows=2)
		centres = Info[:,1:4]
		radii = Info[:,6]
		metname = name.replace('_info.txt','_metrics.txt')
		Metrics = np.loadtxt(metname,skiprows=1)
		xind = np.mod(np.floor(centres[:,0]*float(NgridDM)/boxLen),NgridDM)
		yind = np.mod(np.floor(centres[:,1]*float(NgridDM)/boxLen),NgridDM)
		zind = np.mod(np.floor(centres[:,2]*float(NgridDM)/boxLen),NgridDM)
		#get the Phi values at these grid cells
		DM_cent_vals = DMDens[xind.astype(int),yind.astype(int),zind.astype(int)]
		with open(metname,'w') as Fout:
			Fout.write("StructID R_eff(Mpc/h) CentNumDens WtdAvgNumDens CentDMDens Phi_cent*10^5 DeltaN(Rv) Delta(Rv) DeltaN(3Rv) Delta(3Rv)\n")
			for i in range(len(radii)):
				small_vol = (4*np.pi*(radii[i])**3)/3.0
				big_vol = (4*np.pi*(3.0*radii[i])**3)/3.0
				small_mass = annular_DM_healpy(DMDens,centres[i],0,radii[i],1024)
				big_mass = small_mass + annular_DM_healpy(DMDens,centres[i],radii[i],3.0*radii[i],1024)
				Fout.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" %(Metrics[i,0], Metrics[i,1], Metrics[i,2], Metrics[i,3], \
					DM_cent_vals[i], Metrics[i,5], Metrics[i,6], (small_mass/small_vol-1), Metrics[i,8], (big_mass/big_vol-1)))
	

#----------------------------------------------------------------------------------------------------------#

#----------find maxima/minima of Phi, using jozov-like algorithm----------------#
def jozovPhi(PhiFile, Nside, boxLen, outFile, maxima=True, doMerge=False, MergeVal=0):
	"""
	find the maxima or minima of Phi using a watershed algorithm

	Arguments:
	  PhiFile - file in which the gridded Phi data is stored (in numpy .npy format)
	  Nside - resolution of the grid (total Nside*Nside*Nside voxels/grid cells)
	  boxLen - physical length of the simulation box
	  outFile - output filename
	  maxima - if True, will find maxima of Phi, else will find minima
	  doMerge - if True, will merge some watershed zones according to MergeVal value
	  MergeVal - cutoff to use for merging zones
	"""

	#load the Phi data
	print 'Loading Phi data ...'
	Phi = np.load(PhiFile)
	#flatten to C-type indexing
	Phi = Phi.flat

	#set the adjacencies for the grid
	print 'Setting grid adjacencies ...'
	Nvox = Nside*Nside*Nside
	gridadj = [-1*np.ones((6)) for i in Nvox]
	for i in xrange(Nvox):
		iz = i%Nside
		iy = i%(Nside*Nside)/Nside
		ix = i/(Nside*Nside)
		ind = np.asarray([ix,iy,iz],dtype=int)
		#six neighbouring voxels
		neighbours = np.asarray([ind+[0,0,1],ind+[0,0,-1],ind+[0,1,0],ind+[0,-1,0],ind+[1,0,0],ind+[-1,0,0]])
		#periodic BC
		neighbours[neighbours<0] += Nside
		neighbours[neighbours>Nside] -= Nside
		#flat indices of neighbouring voxels
		gridadj[i] = neighbours[:,0]*Nside*Nside + neighbours[:,1]*Nside + neighbours[:,2]

	jumper = -1*np.ones((Nvox),dtype=int)
	jumped = np.arange(Nvox)
	numinh = np.zeros((Nvox),dtype=int)

	#set jumper for each voxel
	print 'Setting jumper ...'
	for i in xrange(Nvox):
		if maxima:
			maxPhi = np.max(Phi[gridadj[i]])
			if maxPhi>Phi[i]: jumper[i] = gridadj[i][np.argmax(Phi[gridadj[i]])]
		else:
			minPhi = np.min(Phi[gridadj[i]])
			if minPhi<Phi[i]: jumper[i] = gridadj[i][np.argmin(Phi[gridadj[i]])]

	#jump voxels
	print 'About to jump ...'
	for i in xrange(Nvox):
		while jumper[jumped[i]] > -1:
			jumped[i] = jumper[jumped[i]]
		numinh[jumped[i]]+=1	#count how many jumped to this voxel
	print 'Jumped all voxels'

	nzones = np.count_nonzero(numinh)
	print 'Found %d initial zones' %nzones

