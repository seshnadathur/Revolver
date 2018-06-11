import os
import numpy as np
from scipy.stats.mstats import mquantiles

#specify the sample
sHandle = "MDR1_DM_Dense"
useDM = True
dmFile = os.getenv('HOME')+"/Workspace/Sims/MultiDark/MDR1_Dens_1024.npy"
usePhi = False
PhiFile = os.getenv('HOME')+"/Workspace/Sims/MultiDark/MDR1_Phi_1024.npy"

#structure catalogue to use
prefix = "VIDEbVoids"

#stack selection and binning choices
MaxDist = 3	#in units of the void radius, max distance for which profile is required
Nbins = 30	#number of bins
differential = True  #set True for differential density profile, False for cumulative
useVTFE = False

catFile = os.getenv('HOME')+"/Workspace/structures/"+sHandle+"/"+prefix+"_info.txt"
catalogue = np.loadtxt(catFile,skiprows=2)
shortprefix = prefix.replace("Voids",'V')

#profiles are calculated for the voids in the stacks specified by stackList
#set stackList equal to one of the examples below or choose your own criteria

#no density cuts, just different radius quantiles
SimpleRadiusQuantiles = []
nbins = 15
Redges = mquantiles(catalogue[:,6],np.linspace(0,1,nbins))
for i in range(Redges.shape[0]-1):
	rstring = '_%0.1fR%0.1f' %(Redges[i],Redges[i+1])
	dstring = '_RQ'
	name = shortprefix+dstring+rstring
	SimpleRadiusQuantiles.append({'stackName':name,'Rmin':Redges[i],'Rmax':Redges[i+1],'CentDensMin':0,\
		'CentDensMax':max(catalogue[:,4])+0.01,'AvgDensMin':0,'AvgDensMax':100,'zoneMax':0})

#special radius bins
SpecialRadiusBins = []
Redges = [0,4,6,8,9,10,11,12,14,16,20,25,30,35,100]
for i in range(len(Redges)-1):
	rstring = '_%0.1fR%0.1f' %(Redges[i],Redges[i+1])
	dstring = '_SQ'
	name = shortprefix+dstring+rstring
	SpecialRadiusBins.append({'stackName':name,'Rmin':Redges[i],'Rmax':Redges[i+1],'CentDensMin':0,\
		'CentDensMax':max(catalogue[:,4])+0.01,'AvgDensMin':0,'AvgDensMax':100,'zoneMax':0})

#no radius cuts, just different central density quantiles
CentDensQuantiles = []
nbins = 10
Dedges = mquantiles(catalogue[:,4],np.linspace(0,1,nbins))
for i in range(Dedges.shape[0]-1):
	rstring = '_%0.2fD%0.2f' %(Dedges[i],Dedges[i+1])
	dstring = '_DQ'
	name = shortprefix+dstring+rstring
	CentDensQuantiles.append({'stackName':name,'Rmin':0,'Rmax':max(catalogue[:,6])+0.01,\
		'CentDensMin':Dedges[i],'CentDensMax':Dedges[i+1],'AvgDensMin':0,'AvgDensMax':100,'zoneMax':0})

#equally spaced radial bins in log space, no density cuts
LogRadiusBins = []
nbins = 9
H, lRedges = np.histogram(np.log10(catalogue[catalogue[:,6]>13,6]),nbins)
for i in range(lRedges.shape[0]-1):
	rstring = '_%0.1fR%0.1f' %(10**lRedges[i],10**lRedges[i+1])
	dstring = '_logRQ'
	name = shortprefix+dstring+rstring
	LogRadiusBins.append({'stackName':name,'Rmin':10**lRedges[i],'Rmax':10**lRedges[i+1],'CentDensMin':0,\
		'CentDensMax':max(catalogue[:,4])+0.01,'AvgDensMin':0,'AvgDensMax':100,'zoneMax':0})

#2d bins of min dens and radius values
nmin_Rv_Bins = []
H,Redges,Dedges = np.histogram2d(catalogue[:,6],catalogue[:,4],bins=(10,15))
for i in range(H.shape[0]):
	for j in range(H.shape[1]):
		if H[i,j]>0:
			rstring = '_%0.1fR%0.1f' %(Redges[i],Redges[i+1])
			dstring = '_%0.2fD%0.2f' %(Dedges[j],Dedges[j+1])
			name = shortprefix+dstring+rstring
			nmin_Rv_Bins.append({'stackName':name,'Rmin':Redges[i],'Rmax':Redges[i+1],'CentDensMin':Dedges[j],\
				'CentDensMax':Dedges[j+1],'AvgDensMin':0,'AvgDensMax':100,'zoneMax':0})
#print Redges, Dedges

#radius quantiles for all voids below specified min dens threshold
LowDensRadiusQuantiles = []
nbins = 10
DensThresh = 0.2
Redges = mquantiles(catalogue[catalogue[:,4]<DensThresh,6],np.linspace(0,1,nbins))
for i in range(Redges.shape[0]-1):
	rstring = '_%0.1fR%0.1f' %(Redges[i],Redges[i+1])
	dstring = '_LDRQ'
	name = shortprefix+dstring+rstring
	LowDensRadiusQuantiles.append({'stackName':name,'Rmin':Redges[i],'Rmax':Redges[i+1],'CentDensMin':0,\
		'CentDensMax':DensThresh,'AvgDensMin':0,'AvgDensMax':100,'zoneMax':0})

#radius quantiles for all voids below specified avg dens threshold
LowAvgRadiusQuantiles = []
nbins = 10
DensThresh = 1.0
Redges = mquantiles(catalogue[catalogue[:,5]<DensThresh,6],np.linspace(0,1,nbins))
for i in range(Redges.shape[0]-1):
	rstring = '_%0.1fR%0.1f' %(Redges[i],Redges[i+1])
	dstring = '_LARQ'
	name = shortprefix+dstring+rstring
	LowAvgRadiusQuantiles.append({'stackName':name,'Rmin':Redges[i],'Rmax':Redges[i+1],'CentDensMin':0,\
		'CentDensMax':max(catalogue[:,4])+0.1,'AvgDensMin':0,'AvgDensMax':DensThresh,'zoneMax':0})

#quantiles of wtd avg dens
AvgDensQuantiles = []
nbins = 10
Aedges = mquantiles(catalogue[:,5],np.linspace(0,1,nbins))
for i in range(Aedges.shape[0]-1):
	rstring = '_%0.2fA%0.2f' %(Aedges[i],Aedges[i+1])
	dstring = '_AQ'
	name = shortprefix+dstring+rstring
	AvgDensQuantiles.append({'stackName':name,'Rmin':min(catalogue[:,6])-0.1,'Rmax':max(catalogue[:,6])+0.1,\
		'CentDensMin':0,'CentDensMax':max(catalogue[:,4])+0.01,'AvgDensMin':Aedges[i],'AvgDensMax':Aedges[i+1],'zoneMax':0})

#custom specification
CustomStack = [
  {'stackName'  : shortprefix+"_15.1R20.4", 
   'Rmin'	: 15.1, 'Rmax' : 20.4,
   'CentDensMin': 0,'CentDensMax' : 1,
   'AvgDensMin'	: 0, 'AvgDensMax' : 10,
   'zoneMax'	: 0,
  },

]

#set stackList here
stackList = SimpleRadiusQuantiles
#stackList = []
