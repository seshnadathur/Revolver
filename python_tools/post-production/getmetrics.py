# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 13:57:04 2015

@author: seshadri
"""

import numpy as np
import glob
import sys
import os
from python_tools.tools import annular_DM_healpy

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
		yind = np.mod(np.floor(centres[:,1]*float(NgridPhi)/boxLen),NgridPhi)
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
			for i in range(Metrics.shape[0]):
				small_vol = (4*np.pi*(radii[i])**3)/3.0
				big_vol = (4*np.pi*(3.0*radii[i])**3)/3.0
				small_mass = annular_DM_healpy(DMDens,centres[i],0,radii[i],NgridDM)
				big_mass = small_mass + annular_DM_healpy(DMDens,centres[i],radii[i],3.0*radii[i],NgridDM)
				Fout.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" %(Metrics[i,0], Metrics[i,1], Metrics[i,2], Metrics[i,3], \
					DM_cent_vals[i], Metrics[i,5], Metrics[i,6], (small_mass/small_vol-1), Metrics[i,8], (big_mass/big_vol-1)))
	

#----------------------------------------------------------------------------------------------------------#

structdir = os.getenv('HOME')+'/voids/BigMDPl_CMASS/'
PhiFile = os.getenv('HOME')+'/cic/BigMDPl_Phi1175_flat_g1_z052.bin'
DMFile = os.getenv('HOME')+'/cic/BigMDPl_Dens2350_flat_z052.bin'
NgridPhi = 1175
NgridDM = 2350
boxLen = 2500.0

#print structdir
#print PhiFile
recalcMetrics(structdir,PhiFile,DMFile,NgridPhi,NgridDM,boxLen)