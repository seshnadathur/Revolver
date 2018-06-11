import os
import numpy as np
from scipy.stats.mstats import mquantiles

#specify the sample
sHandle = "MDR1_Main1"
useDM = False
dmFile = os.getenv('HOME')+"/Workspace/Sims/MultiDark/MDR1_Dens_1024.npy"
usePhi = False
PhiFile = os.getenv('HOME')+"/Workspace/Sims/MultiDark/MDR1_Phi_1024.npy"

#structure catalogue to use
prefix = "IsolatedVoids"

#binning choices
MaxDist = 3	#in units of the void radius, max distance for which profile is required
Nbins = 30	#number of bins
differential = True  #set True for differential density profile, False for cumulative
useVTFE = True

catFile = os.getenv('HOME')+"/Workspace/structures/"+sHandle+"/"+prefix+"_info.txt"
catalogue = np.loadtxt(catFile,skiprows=2)
shortprefix = prefix.replace("Voids",'V')
   
#profiles are calculated for the voids in the stacks specified by stackList
#set stackList equal to one of the examples below or choose your own criteria

#no density cuts, just different radius quantiles
SimpleRadiusQuantiles = []
nbins = 10
Redges = mquantiles(catalogue[:,6],np.linspace(0,1,nbins))
for i in range(Redges.shape[0]-1):
	rstring = '_%0.1fR%0.1f' %(Redges[i],Redges[i+1])
	dstring = '_RQ'
	name = shortprefix+dstring+rstring
	SimpleRadiusQuantiles.append({'stackName':name,'Rmin':Redges[i],'Rmax':Redges[i+1],'CentDensMin':0,\
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

#2d bins of min dens and radius values
nmin_Rv_Bins = []
H,Redges,Dedges = np.histogram2d(catalogue[:,6],catalogue[:,4],bins=10)
for i in range(H.shape[0]):
	for j in range(H.shape[1]):
		if H[i,j]>0:
			rstring = '_%0.1fR%0.1f' %(Redges[i],Redges[i+1])
			dstring = '_%0.2fD%0.2f' %(Dedges[j],Dedges[j+1])
			name = shortprefix+dstring+rstring
			nmin_Rv_Bins.append({'stackName':name,'Rmin':Redges[i],'Rmax':Redges[i+1],'CentDensMin':Dedges[j],\
				'CentDensMax':Dedges[j+1],'AvgDensMin':0,'AvgDensMax':100,'zoneMax':0})

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


#equally spaced radial bins in log space, no density cuts
LogRadiusBins = []
H,Redges = np.histogram(np.log10(catalogue[:,6]),bins=9)
for i in range(H.shape[0]):
	rstring = '_%0.1fR%0.1f' %(10**Redges[i],10**Redges[i+1])
	dstring = '_lR'
	name = shortprefix+dstring+rstring
	LogRadiusBins.append({'stackName':name,'Rmin':10**Redges[i],'Rmax':10**Redges[i+1],'CentDensMin':0,\
		'CentDensMax':max(catalogue[:,4])+0.01,'AvgDensMin':0,'AvgDensMax':100,'zoneMax':0})
	
#quantiles of wtd avg dens
AvgDensQuantiles = []
nbins = 10
Aedges = mquantiles(catalogue[:,5],np.linspace(0,1,nbins))
for i in range(Aedges.shape[0]-1):
	rstring = '_%0.2fA%0.2f' %(Redges[i],Redges[i+1])
	dstring = '_AQ'
	name = shortprefix+dstring+rstring
	AvgDensQuantiles.append({'stackName':name,'Rmin':min(catalogue[:,6])-0.1,'Rmax':max(catalogue[:,6])+0.1,\
		'CentDensMin':0,'CentDensMax':max(catalogue[:,4])+0.01,'AvgDensMin':Aedges[i],'AvgDensMax':Aedges[i+1],'zoneMax':0})

#custom specification
CustomStack = [
  {'stackName'  : shortprefix+"_0D0.3", 
   'Rmin'	: 0, 'Rmax' : max(catalogue[:,6])+0.1,
   'CentDensMin': 0,'CentDensMax' : 0.3,
   'AvgDensMin'	: 0, 'AvgDensMax' : max(catalogue[:,5])+0.1,
   'zoneMax'	: 0,
  },

]

#set stackList here
stackList = SimpleRadiusQuantiles
